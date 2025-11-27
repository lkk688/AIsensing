#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISAC Simulation (Corrected & Optimized)
- Proper pilots + LS channel estimation + nearest 2D interpolation (OFDM & OTFS)
- Pre-equalized, domain-aligned tokens (learnable denoising/refinement)
- Train SNR ~ Uniform[0,20] dB
- Modulation-agnostic I/Q regression head; demap to BPSK/QPSK/QAMxx at eval (no retrain)
- Masked loss over data REs (pilots excluded)
- BER vs SNR curves, and OTFS DD sanity figure

Author: AISensing (2025-10-30)
"""

import os
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Config
# -------------------------------
@dataclass
class ISACConfig:
    # Repro / device
    seed: int = 42
    device: str = "cpu"

    # Grids
    N_SC: int = 32      # subcarriers / delay bins
    M_SYM: int = 16     # symbols / Doppler bins
    CP_LEN: int = 8

    # Pilot comb (OFDM frequency-time; OTFS delay-Doppler)
    ofdm_df: int = 8
    ofdm_dt: int = 4
    otfs_df: int = 8
    otfs_dt: int = 4
    pilot_value: complex = 1.0 + 0.0j  # known pilot symbol

    # Training
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-3
    train_samples: int = 6000
    val_samples: int = 800
    test_samples: int = 800

    # SNR
    snr_train_min: float = 0.0
    snr_train_max: float = 20.0
    snr_eval_list: tuple = (0, 5, 10, 15, 20)

    # Modulations
    train_mods: tuple = ("BPSK", "QPSK", "QAM16", "QAM64")
    eval_mods: tuple = ("BPSK", "QPSK", "QAM16", "QAM64", "QAM256")

    # Model
    d_model: int = 128
    nhead: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # Output
    outdir: str = "./outputs_isac"


# -------------------------------
# Utility
# -------------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    p = np.mean(np.abs(signal)**2) + 1e-12
    nvar = p / (10**(snr_db/10))
    noise = np.sqrt(nvar/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return (signal + noise.astype(np.complex64)).astype(np.complex64)

def make_pos_enc(L, d_model, device="cpu"):
    pos = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, dtype=torch.float32, device=device).unsqueeze(0)
    angle_rates = 1.0 / torch.pow(10000, (2*(i//2))/d_model)
    angles = pos * angle_rates
    pe = torch.zeros(L, d_model, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


# -------------------------------
# Constellations (Gray)
# -------------------------------
def gray_to_binary(g):
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b

def bits_from_int(x, k):
    return [(x >> i) & 1 for i in range(k-1, -1, -1)]

def qam_square_constellation(M):
    k = int(round(math.log2(M)))
    m_side = int(round(math.sqrt(M)))
    assert m_side*m_side == M
    bps = k//2
    levels = np.arange(-(m_side-1), (m_side-1)+2, 2)
    pts=[]; bits_list=[]
    for gi in range(m_side):
        i_idx = gray_to_binary(gi)
        for gq in range(m_side):
            q_idx = gray_to_binary(gq)
            I = levels[i_idx]; Q = levels[q_idx]
            pts.append(I + 1j*Q)
            bI = bits_from_int(gi, bps); bQ = bits_from_int(gq, bps)
            bits_list.append(bI + bQ)
    pts = np.array(pts, np.complex64)
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    bits_arr = np.array(bits_list, np.int64)
    return pts, bits_arr

def bpsk_constellation():
    pts = np.array([1+0j, -1+0j], np.complex64); bits = np.array([[0],[1]], np.int64)
    return pts, bits

def get_constellation(mod: str):
    m = mod.upper()
    if m=="BPSK": return bpsk_constellation()
    if m=="QPSK": return qam_square_constellation(4)
    if m=="QAM16": return qam_square_constellation(16)
    if m=="QAM64": return qam_square_constellation(64)
    if m=="QAM256": return qam_square_constellation(256)
    raise ValueError("Unsupported modulation: "+mod)

def modulate_bits(bits, mod):
    pts, bits_tbl = get_constellation(mod)
    k = bits_tbl.shape[1]
    bits = np.array(bits, np.int64).reshape(-1, k)
    pow2 = (1 << np.arange(k-1, -1, -1)).astype(np.int64)
    tbl_idx = (bits_tbl*pow2).sum(1)
    LUT = -np.ones(2**k, np.int64); LUT[tbl_idx] = np.arange(len(tbl_idx), dtype=np.int64)
    idx = (bits*pow2).sum(1)
    return pts[LUT[idx]]

def nearest_bits(symbols, mod):
    pts, bits_tbl = get_constellation(mod)
    d2 = np.abs(symbols.reshape(-1,1) - pts.reshape(1,-1))**2
    idx = np.argmin(d2, axis=1)
    return bits_tbl[idx].reshape(-1)


# -------------------------------
# Channels (toy DD)
# -------------------------------
def rand_delay_doppler_paths(n_paths=3, max_delay=6, max_doppler=0.10):
    delays = np.random.randint(0, max_delay+1, size=n_paths)
    dopplers = np.random.uniform(-max_doppler, max_doppler, size=n_paths)
    gains = (np.random.randn(n_paths) + 1j*np.random.randn(n_paths)).astype(np.complex64) / np.sqrt(2*n_paths)
    return delays, dopplers, gains

def apply_dd_channel(signal, delays, dopplers, gains):
    n = np.arange(len(signal))
    out = np.zeros_like(signal, dtype=np.complex64)
    for d, fd, g in zip(delays, dopplers, gains):
        doppler_shift = np.exp(1j*2*np.pi*fd*n).astype(np.complex64)
        out += g * np.roll(signal, int(d)) * doppler_shift
    return out


# -------------------------------
# OTFS
# -------------------------------
def otfs_modulate_dd_grid(X_nm):
    N, M = X_nm.shape
    x_tf = np.fft.ifft2(X_nm)
    tx=[]
    for m in range(M):
        tx.append(np.fft.ifft(x_tf[:, m]))
    return np.concatenate(tx).astype(np.complex64)

def otfs_demodulate_to_dd(rx, N, M):
    x_tf = np.zeros((N, M), dtype=np.complex64)
    for m in range(M):
        blk = rx[m*N:(m+1)*N]
        x_tf[:, m] = np.fft.fft(blk)
    return np.fft.fft2(x_tf)


# -------------------------------
# OFDM
# -------------------------------
def ofdm_modulate_grid(X_fm, N, M, CP_LEN):
    tx=[]
    for m in range(M):
        sym_t = np.fft.ifft(X_fm[:, m])
        with_cp = np.concatenate([sym_t[-CP_LEN:], sym_t])
        tx.append(with_cp)
    return np.concatenate(tx).astype(np.complex64)

def ofdm_demodulate(rx, N, M, CP_LEN):
    X = np.zeros((N,M), dtype=np.complex64)
    idx=0
    for m in range(M):
        blk = rx[idx:idx+N+CP_LEN]; idx += N+CP_LEN
        t = blk[CP_LEN:]
        X[:, m] = np.fft.fft(t)
    return X


# -------------------------------
# Pilots & interpolation
# -------------------------------
def build_pilot_mask(N, M, df, dt):
    mask = np.zeros((N, M), dtype=bool)
    for m in range(0, M, max(1, dt)):
        for n in range(0, N, max(1, df)):
            mask[n, m] = True
    return mask

def nearest_interp_from_pilots(Hls, pilot_mask, df, dt):
    """Nearest-neighbor interpolation aligned to the comb pilot grid."""
    N, M = pilot_mask.shape
    n_grid = np.arange(0, N, max(1, df))
    m_grid = np.arange(0, M, max(1, dt))
    # nearest pilot index for each n and m
    n_idx = np.clip(np.round(np.arange(N)/max(1,df)).astype(int), 0, len(n_grid)-1)
    m_idx = np.clip(np.round(np.arange(M)/max(1,dt)).astype(int), 0, len(m_grid)-1)
    Hhat = Hls[n_grid[n_idx][:,None], m_grid[m_idx][None,:]]
    return Hhat

def estimate_channel_from_pilots(Xh, pilot_mask, pilot_value, df, dt):
    Hls = np.zeros_like(Xh)
    Hls[pilot_mask] = Xh[pilot_mask] / (pilot_value + 1e-12)
    Hhat = nearest_interp_from_pilots(Hls, pilot_mask, df, dt)
    return Hhat


# -------------------------------
# Datasets (with pilots & masked loss)
# -------------------------------
def _fill_grid_with_bits(N, M, data_mask, bits, mod):
    """Fill an N×M complex grid: data_mask-> modulated bits, else zeros (caller sets pilots)."""
    pts, bits_tbl = get_constellation(mod)
    k = bits_tbl.shape[1]
    assert bits.size == data_mask.sum()*k
    data_syms = modulate_bits(bits, mod)
    X = np.zeros((N, M), dtype=np.complex64)
    X[data_mask] = data_syms.astype(np.complex64)
    return X

class CommTrainDataset(torch.utils.data.Dataset):
    """Training: random mod from cfg.train_mods, SNR~U[0,20] dB, pilots added, masked loss on data REs."""
    def __init__(self, cfg: ISACConfig, n_samples, waveform='OTFS'):
        super().__init__()
        self.cfg = cfg
        self.n = n_samples
        self.waveform = waveform

    def __len__(self): return self.n

    def __getitem__(self, idx):
        c = self.cfg
        N, M, CP = c.N_SC, c.M_SYM, c.CP_LEN
        mod = np.random.choice(c.train_mods)

        if self.waveform == 'OFDM':
            pilot_mask = build_pilot_mask(N, M, c.ofdm_df, c.ofdm_dt)
        else:
            pilot_mask = build_pilot_mask(N, M, c.otfs_df, c.otfs_dt)
        data_mask = ~pilot_mask

        pts, bits_tbl = get_constellation(mod)
        k = bits_tbl.shape[1]
        bits = np.random.randint(0, 2, size=(data_mask.sum()*k,), dtype=np.int64)

        # Build TX grid with pilots + data
        X_data = _fill_grid_with_bits(N, M, data_mask, bits, mod)
        X_tx = X_data.copy()
        X_tx[pilot_mask] = c.pilot_value

        # Modulate
        if self.waveform == 'OFDM':
            tx = ofdm_modulate_grid(X_tx, N, M, CP)
        else:
            tx = otfs_modulate_dd_grid(X_tx)

        # Channel + noise
        delays, dopplers, gains = rand_delay_doppler_paths()
        rx = apply_dd_channel(tx, delays, dopplers, gains)
        snr_db = np.random.uniform(c.snr_train_min, c.snr_train_max)
        rx = add_awgn(rx, snr_db)

        # Demod + LS channel est from pilots + pre-eq tokens
        if self.waveform == 'OFDM':
            Xh = ofdm_demodulate(rx, N, M, CP)
            Hhat = estimate_channel_from_pilots(Xh, pilot_mask, c.pilot_value, c.ofdm_df, c.ofdm_dt)
        else:
            Xh = otfs_demodulate_to_dd(rx, N, M)
            Hhat = estimate_channel_from_pilots(Xh, pilot_mask, c.pilot_value, c.otfs_df, c.otfs_dt)

        Xeq = Xh / (Hhat + 1e-8)  # equalized tokens
        tokens = np.stack([Xeq.real, Xeq.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
        y_iq = np.stack([X_data.real, X_data.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
        mask_flat = data_mask.reshape(-1)  # bool

        return torch.from_numpy(tokens), torch.from_numpy(y_iq), torch.from_numpy(mask_flat)


class CommEvalDataset(torch.utils.data.Dataset):
    """Evaluation: fixed mod & SNR; returns tokens, labels, bits for DATA REs, and data mask."""
    def __init__(self, cfg: ISACConfig, n_samples, waveform='OTFS', snr_db=10, modulation="QPSK"):
        super().__init__()
        self.cfg = cfg; self.n = n_samples
        self.waveform = waveform; self.snr_db = snr_db; self.mod = modulation

    def __len__(self): return self.n

    def __getitem__(self, idx):
        c = self.cfg
        N, M, CP = c.N_SC, c.M_SYM, c.CP_LEN
        if self.waveform == 'OFDM':
            pilot_mask = build_pilot_mask(N, M, c.ofdm_df, c.ofdm_dt)
        else:
            pilot_mask = build_pilot_mask(N, M, c.otfs_df, c.otfs_dt)
        data_mask = ~pilot_mask

        pts, bits_tbl = get_constellation(self.mod)
        k = bits_tbl.shape[1]
        bits = np.random.randint(0,2,size=(data_mask.sum()*k,), dtype=np.int64)

        X_data = _fill_grid_with_bits(N, M, data_mask, bits, self.mod)
        X_tx = X_data.copy(); X_tx[pilot_mask] = c.pilot_value

        if self.waveform == 'OFDM':
            tx = ofdm_modulate_grid(X_tx, N, M, CP)
        else:
            tx = otfs_modulate_dd_grid(X_tx)

        delays, dopplers, gains = rand_delay_doppler_paths()
        rx = apply_dd_channel(tx, delays, dopplers, gains)
        rx = add_awgn(rx, self.snr_db)

        if self.waveform == 'OFDM':
            Xh = ofdm_demodulate(rx, N, M, CP)
            Hhat = estimate_channel_from_pilots(Xh, pilot_mask, c.pilot_value, c.ofdm_df, c.ofdm_dt)
        else:
            Xh = otfs_demodulate_to_dd(rx, N, M)
            Hhat = estimate_channel_from_pilots(Xh, pilot_mask, c.pilot_value, c.otfs_df, c.otfs_dt)

        Xeq = Xh / (Hhat + 1e-8)
        tokens = np.stack([Xeq.real, Xeq.imag], axis=-1).astype(np.float32).reshape(N*M, 2)
        y_iq = np.stack([X_data.real, X_data.imag], axis=-1).astype(np.float32).reshape(N*M, 2)

        return (torch.from_numpy(tokens),
                torch.from_numpy(y_iq),
                torch.from_numpy(bits.astype(np.int64)),
                torch.from_numpy(data_mask.reshape(-1)))


# -------------------------------
# Model
# -------------------------------
class CommTransformer(nn.Module):
    def __init__(self, seq_len, d_model=128, nhead=4, n_layers=2, dropout=0.1, device="cpu"):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=2*d_model, dropout=dropout,
                                         batch_first=True)
        self.backbone = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)  # I/Q
        self.register_buffer("pos", make_pos_enc(seq_len, d_model, device))

    def forward(self, x):
        h = self.input_proj(x) + self.pos[:x.shape[1], :]
        z = self.backbone(h)
        return self.head(z)


# -------------------------------
# Training / Eval
# -------------------------------
def train_comm_model(cfg: ISACConfig, waveform='OTFS'):
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    seq_len = cfg.N_SC * cfg.M_SYM

    model = CommTransformer(seq_len=seq_len, d_model=cfg.d_model, nhead=cfg.nhead,
                            n_layers=cfg.n_layers, dropout=cfg.dropout, device=cfg.device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    tr_ds = CommTrainDataset(cfg, cfg.train_samples, waveform=waveform)
    va_ds = CommTrainDataset(cfg, cfg.val_samples, waveform=waveform)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)

    best_val = 1e9; best_state=None
    for ep in range(cfg.epochs):
        # train
        model.train(); s=0.0; n=0
        for x, y, mask in tr_loader:
            x=x.to(device); y=y.to(device); mask=mask.to(device).bool()
            yhat = model(x)
            # masked MSE on data REs only
            mask3 = mask.unsqueeze(-1).expand_as(y)
            loss = F.mse_loss(yhat[mask3], y[mask3])
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item() * x.shape[0]; n += x.shape[0]
        tr = s/max(1,n)

        # val
        model.eval(); s=0.0; n=0
        with torch.no_grad():
            for x, y, mask in va_loader:
                x=x.to(device); y=y.to(device); mask=mask.to(device).bool()
                yhat = model(x)
                mask3 = mask.unsqueeze(-1).expand_as(y)
                loss = F.mse_loss(yhat[mask3], y[mask3])
                s += loss.item() * x.shape[0]; n += x.shape[0]
        va = s/max(1,n)

        if va < best_val:
            best_val = va
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        print(f"[COMM {waveform}] Epoch {ep+1}/{cfg.epochs} train={tr:.4f} val={va:.4f}")

    model.load_state_dict(best_state)
    return model


def eval_comm_ber(model: nn.Module, cfg: ISACConfig, waveform='OTFS',
                  snr_db=10, modulation="QPSK", n_samples=512):
    device = torch.device(cfg.device)
    model.eval()
    ds = CommEvalDataset(cfg, n_samples, waveform=waveform, snr_db=snr_db, modulation=modulation)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    total_bits=0; err_bits=0
    with torch.no_grad():
        for x, y, bits_true, data_mask in loader:
            pred = model(x.to(device)).cpu().numpy()  # [B,T,2]
            sym = (pred[...,0] + 1j*pred[...,1])      # [B,T]
            mask = data_mask.numpy().astype(bool)
            sym_data = sym.reshape(-1)[mask.reshape(-1)]
            bits_hat = nearest_bits(sym_data, modulation)

            bt = bits_true.numpy().reshape(-1)
            L = min(len(bt), len(bits_hat))
            total_bits += L
            err_bits   += np.sum(bits_hat[:L] != bt[:L])

    return err_bits/max(1,total_bits)


# -------------------------------
# Classical baselines (with pilots)
# -------------------------------
def classical_comm_ber_ofdm(cfg: ISACConfig, snr_db, modulation="QPSK"):
    N,M,CP = cfg.N_SC, cfg.M_SYM, cfg.CP_LEN
    pilot_mask = build_pilot_mask(N,M,cfg.ofdm_df,cfg.ofdm_dt)
    data_mask = ~pilot_mask
    pts, bits_tbl = get_constellation(modulation)
    k = bits_tbl.shape[1]
    bits = np.random.randint(0,2,size=(data_mask.sum()*k,))
    X_data = _fill_grid_with_bits(N,M,data_mask,bits,modulation)
    X_tx   = X_data.copy(); X_tx[pilot_mask] = cfg.pilot_value

    tx = ofdm_modulate_grid(X_tx, N, M, CP)
    delays, dopplers, gains = rand_delay_doppler_paths()
    rx = apply_dd_channel(tx, delays, dopplers, gains)
    rx = add_awgn(rx, snr_db)

    Xh = ofdm_demodulate(rx, N, M, CP)
    Hhat = estimate_channel_from_pilots(Xh, pilot_mask, cfg.pilot_value, cfg.ofdm_df, cfg.ofdm_dt)
    Xeq = Xh / (Hhat + 1e-8)
    sym = Xeq.reshape(-1)[data_mask.reshape(-1)]
    bits_hat = nearest_bits(sym, modulation)
    return np.mean(bits_hat != bits[:len(bits_hat)])


def classical_comm_ber_otfs(cfg: ISACConfig, snr_db, modulation="QPSK"):
    N,M = cfg.N_SC, cfg.M_SYM
    pilot_mask = build_pilot_mask(N,M,cfg.otfs_df,cfg.otfs_dt)
    data_mask = ~pilot_mask
    pts, bits_tbl = get_constellation(modulation)
    k = bits_tbl.shape[1]
    bits = np.random.randint(0,2,size=(data_mask.sum()*k,))
    X_data = _fill_grid_with_bits(N,M,data_mask,bits,modulation)
    X_tx   = X_data.copy(); X_tx[pilot_mask] = cfg.pilot_value

    tx = otfs_modulate_dd_grid(X_tx)
    delays, dopplers, gains = rand_delay_doppler_paths()
    rx = apply_dd_channel(tx, delays, dopplers, gains)
    rx = add_awgn(rx, snr_db)

    Xh = otfs_demodulate_to_dd(rx, N, M)
    Hhat = estimate_channel_from_pilots(Xh, pilot_mask, cfg.pilot_value, cfg.otfs_df, cfg.otfs_dt)
    Xeq = Xh / (Hhat + 1e-8)
    sym = Xeq.reshape(-1)[data_mask.reshape(-1)]
    bits_hat = nearest_bits(sym, modulation)
    return np.mean(bits_hat != bits[:len(bits_hat)])


# -------------------------------
# Debug visualization (sanity)
# -------------------------------
def debug_otfs_single_target_fig(N, M, snr_db=20, delay_bin=3, doppler_bin=5, save_png=None):
    X_nm = np.zeros((N, M), dtype=np.complex64)
    X_nm[delay_bin, doppler_bin] = 1+0j
    tx = otfs_modulate_dd_grid(X_nm)
    delays = np.array([delay_bin]); dopplers = np.array([(doppler_bin - M//2)/M * 0.1])
    gains = np.array([1+0j], dtype=np.complex64)
    rx = apply_dd_channel(tx, delays, dopplers, gains)
    rx = add_awgn(rx, snr_db)
    Xh = otfs_demodulate_to_dd(rx, N, M)
    mag = np.abs(Xh)
    plt.figure()
    plt.imshow(mag, aspect='auto')
    plt.title(f"OTFS single target @ (delay={delay_bin}, doppler={doppler_bin})")
    plt.xlabel("Doppler"); plt.ylabel("Delay"); plt.colorbar(); plt.tight_layout()
    if save_png: plt.savefig(save_png); plt.close()
    else: plt.show()


# -------------------------------
# Orchestration
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="./outputs_isac")
    args = ap.parse_args()

    cfg = ISACConfig(device=args.device, epochs=args.epochs, outdir=args.outdir)
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    print("=== Training neural receivers (pilots + pre-eq, SNR~U[0,20] dB, modulation-agnostic) ===")
    comm_ofdm = train_comm_model(cfg, 'OFDM')
    comm_otfs = train_comm_model(cfg, 'OTFS')

    # Evaluate BER across SNR & modulations
    rows=[]
    for mod in cfg.eval_mods:
        for snr in cfg.snr_eval_list:
            ber_ofdm_class = classical_comm_ber_ofdm(cfg, snr, modulation=mod)
            ber_otfs_class = classical_comm_ber_otfs(cfg, snr, modulation=mod)
            ber_ofdm_nn = eval_comm_ber(comm_ofdm, cfg, waveform='OFDM', snr_db=snr, modulation=mod, n_samples=cfg.test_samples)
            ber_otfs_nn = eval_comm_ber(comm_otfs, cfg, waveform='OTFS', snr_db=snr, modulation=mod, n_samples=cfg.test_samples)
            rows += [
                {"Waveform":"OFDM","Model":"Classical","Mod":mod,"SNR(dB)":snr,"BER":ber_ofdm_class},
                {"Waveform":"OTFS","Model":"Classical","Mod":mod,"SNR(dB)":snr,"BER":ber_otfs_class},
                {"Waveform":"OFDM","Model":"Transformer","Mod":mod,"SNR(dB)":snr,"BER":ber_ofdm_nn},
                {"Waveform":"OTFS","Model":"Transformer","Mod":mod,"SNR(dB)":snr,"BER":ber_otfs_nn},
            ]
            print(f"[SNR={snr:2d} | {mod:6s}]  OFDM cls/nn: {ber_ofdm_class:.3e}/{ber_ofdm_nn:.3e} | "
                  f"OTFS cls/nn: {ber_otfs_class:.3e}/{ber_otfs_nn:.3e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.outdir, "comm_metrics.csv")
    df.to_csv(csv_path, index=False)

    # BER plots
    for mod in cfg.eval_mods:
        sub = df[(df.Mod==mod)]
        if len(sub)==0: continue
        plt.figure()
        for (wave, model) in [("OFDM","Classical"),("OFDM","Transformer"),
                              ("OTFS","Classical"),("OTFS","Transformer")]:
            xs=[]; ys=[]
            for snr in cfg.snr_eval_list:
                q = sub[(sub.Waveform==wave)&(sub.Model==model)&(sub["SNR(dB)"]==snr)]
                if len(q)==0: continue
                xs.append(snr); ys.append(q["BER"].values[0])
            if len(xs)>0: plt.plot(xs, ys, marker='o', label=f"{wave} {model}")
        plt.yscale('log'); plt.xlabel("SNR (dB)"); plt.ylabel("BER"); plt.title(f"BER vs SNR — {mod}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(cfg.outdir, f"ber_{mod}.png")); plt.close()

    # Sanity figure
    debug_otfs_single_target_fig(cfg.N_SC, cfg.M_SYM, snr_db=15, delay_bin=2, doppler_bin=7,
                                 save_png=os.path.join(cfg.outdir, "debug_otfs_dd.png"))

    print(f"Saved metrics to {csv_path}")
    print(f"Figures in {cfg.outdir}")

if __name__ == "__main__":
    main()