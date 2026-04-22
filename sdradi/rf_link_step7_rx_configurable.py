#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_link_step7_rx_configurable.py

Configurable RX for "Step7" packetized OFDM video link (PlutoSDR).
- JSON config + CLI overrides
- Fast Schmidl-Cox sync + CFO estimate
- LTF channel estimate
- Pilot-aided FLL+PLL tracking
- Header-only probe (MAGIC + header) to compute required demod length
- Bounded demod (avoid tail contamination)
- Bit-slip search (0..7) for robust byte alignment
- Logging to CSV + summary plots

Packet format:
  MAGIC("VID7", 4B) | FRAME_ID(2B) | SEQ(2B) | TOTAL(2B) | LEN(2B) | PAYLOAD | CRC32(4B)
CRC covers MAGIC..PAYLOAD.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MAGIC = b"VID7"

# ==============================
# OFDM Parameters (match TX)
# ==============================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array([k for k in range(-26, 27)
                              if k != 0 and k not in set(PILOT_SUBCARRIERS)], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)                   # 48
BITS_PER_QPSK_SYM = 2
BITS_PER_OFDM_SYM = N_DATA * BITS_PER_QPSK_SYM   # 96 raw bits per OFDM symbol (coded bits)

def sc_to_bin(k: int) -> int:
    """Unshifted FFT bin index mapping for subcarrier k in [-N/2..N/2-1]."""
    return (k + N_FFT) % N_FFT

PILOT_BINS = np.array([sc_to_bin(int(k)) for k in PILOT_SUBCARRIERS], dtype=int)
DATA_BINS  = np.array([sc_to_bin(int(k)) for k in DATA_SUBCARRIERS], dtype=int)

USED_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
USED_BINS = np.array([sc_to_bin(int(k)) for k in USED_SUBCARRIERS], dtype=int)


# ==============================
# Config
# ==============================
@dataclass
class SDRConfig:
    uri: str = "ip:192.168.2.2"
    fc_hz: float = 2.3e9
    fs_hz: float = 3e6
    rx_gain_db: float = 30.0
    rx_bw_hz: float = 3.6e6
    rx_buffer_size: int = 65536
    kernel_buffers: int = 4

@dataclass
class PreambleConfig:
    stf_repeats: int = 6
    ltf_symbols: int = 4

@dataclass
class DemodConfig:
    repeat: int = 2  # 1/2/4
    kp: float = 0.15
    ki: float = 0.005
    sync_peak_med_ratio_th: float = 6.0
    sc_threshold: float = 0.20
    search_len: int = 60000
    xcorr_refine_win: int = 2000

    # PROBE behavior
    probe_syms: int = 16
    bit_slip_max: int = 7

    # bounded demod safety cap
    max_payload_syms_cap: int = 300

    # optional early stop on bad EVM streak
    bad_evm_th: float = 0.85
    bad_evm_patience: int = 5

@dataclass
class RuntimeConfig:
    max_captures: int = 5000
    max_frames: int = 0
    frame_timeout_s: float = 60.0
    max_inflight: int = 10
    output_dir: str = "rf_link_step7_rx_runs"
    output_video: str = "received_video.avi"
    output_fps: float = 2.0
    width: int = 320
    height: int = 240

    # logging / plotting
    verbose: bool = False
    sparse_every: int = 200
    save_csv: bool = True
    save_plots: bool = True
    plot_every_frame: int = 1
    save_frames_dir: str = ""

@dataclass
class AppConfig:
    # ✅ FIX: use default_factory for dataclass fields that are objects
    sdr: SDRConfig = field(default_factory=SDRConfig)
    preamble: PreambleConfig = field(default_factory=PreambleConfig)
    demod: DemodConfig = field(default_factory=DemodConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = AppConfig()
    for section_name in ["sdr", "preamble", "demod", "runtime"]:
        if section_name in raw:
            sec_obj = getattr(cfg, section_name)
            for k, v in raw[section_name].items():
                if hasattr(sec_obj, k):
                    setattr(sec_obj, k, v)
    return cfg

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ==============================
# Packet parsing
# ==============================
def parse_video_packet_full(data: bytes) -> Tuple[bool, int, int, int, bytes, int, str]:
    if len(data) < 16:
        return False, -1, -1, 0, b"", 0, "too_short"
    if data[:4] != MAGIC:
        return False, -1, -1, 0, b"", 0, "no_magic"
    frame_id = int.from_bytes(data[4:6], "little")
    seq      = int.from_bytes(data[6:8], "little")
    total    = int.from_bytes(data[8:10], "little")
    plen     = int.from_bytes(data[10:12], "little")
    expected = 12 + plen + 4
    if len(data) < expected:
        return False, frame_id, seq, total, b"", expected, "need_more_bytes"
    content  = data[:12 + plen]
    payload  = data[12:12 + plen]
    crc_rx   = int.from_bytes(data[12 + plen:12 + plen + 4], "little")
    crc_calc = zlib.crc32(content) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return False, frame_id, seq, total, b"", expected, "crc_mismatch"
    return True, frame_id, seq, total, payload, expected, "ok"

def parse_header_only(data: bytes) -> Tuple[bool, int, int, int, int]:
    if len(data) < 12:
        return False, -1, -1, 0, 0
    if data[:4] != MAGIC:
        return False, -1, -1, 0, 0
    frame_id = int.from_bytes(data[4:6], "little")
    seq      = int.from_bytes(data[6:8], "little")
    total    = int.from_bytes(data[8:10], "little")
    plen     = int.from_bytes(data[10:12], "little")
    expected = 12 + plen + 4
    return True, frame_id, seq, total, expected


# ==============================
# DSP helpers
# ==============================
def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(len(samples), dtype=np.float32)
    return samples * np.exp(-1j * 2 * np.pi * (cfo_hz / fs) * n).astype(np.complex64)

def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
    for i, s in enumerate(symbols):
        rr = np.real(s) >= 0
        ii = np.imag(s) >= 0
        if rr and ii:
            bits[2*i], bits[2*i+1] = 0, 0
        elif (not rr) and ii:
            bits[2*i], bits[2*i+1] = 0, 1
        elif (not rr) and (not ii):
            bits[2*i], bits[2*i+1] = 1, 1
        else:
            bits[2*i], bits[2*i+1] = 1, 0
    return bits

def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    if L <= 0:
        return np.array([], dtype=np.uint8)
    m = bits[:L].reshape(-1, repeat)
    return (np.sum(m, axis=1) >= (repeat / 2)).astype(np.uint8)

def packbits_with_slip(bits: np.ndarray, slip: int) -> bytes:
    if slip > 0:
        if len(bits) <= slip:
            return b""
        bits = bits[slip:]
    L = (len(bits) // 8) * 8
    if L <= 0:
        return b""
    return np.packbits(bits[:L]).tobytes()


# ==============================
# Preamble builders
# ==============================
def create_schmidl_cox_stf(num_repeats: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i] + 0j
    x = np.fft.ifft(X) * np.sqrt(N_FFT)
    stf = np.tile(x, num_repeats).astype(np.complex64)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp, X

def create_ltf_ref(num_symbols: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(X) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X


# ==============================
# Schmidl-Cox + CFO
# ==============================
def schmidl_cox_metric(rx: np.ndarray, L: int, search_len: int) -> Tuple[np.ndarray, np.ndarray]:
    N = len(rx)
    search_len = min(search_len, N - 2*L - 1)
    if search_len <= 2:
        return np.array([]), np.array([])
    P = np.zeros(search_len, dtype=np.complex64)
    R = np.zeros(search_len, dtype=np.float32)
    for n in range(search_len):
        a = rx[n:n+L]
        b = rx[n+L:n+2*L]
        P[n] = np.sum(a * np.conj(b))
        R[n] = np.sum(np.abs(b)**2)
    M = (np.abs(P)**2) / (R**2 + 1e-12)
    return M, P

def detect_stf_autocorr(rx: np.ndarray, fs: float, half_period: int, threshold: float,
                        search_len: int, ratio_th: float) -> Tuple[bool, int, float, float, float]:
    L = half_period
    M, P = schmidl_cox_metric(rx, L=L, search_len=search_len)
    if M.size == 0:
        return False, 0, 0.0, 0.0, 0.0

    plateau = 2 * L
    if M.size <= plateau + 2:
        peak = float(np.max(M))
        ratio = float(peak / (np.median(M) + 1e-12))
        return False, int(np.argmax(M)), peak, 0.0, ratio

    kernel = np.ones(plateau, dtype=np.float32) / plateau
    M_s = np.convolve(M, kernel, mode="valid")
    pk = int(np.argmax(M_s))
    peak_s = float(M_s[pk])
    med = float(np.median(M_s) + 1e-12)
    ratio = peak_s / med

    idx = pk
    for i in range(pk, max(0, pk - 5*L), -1):
        if M[i] < 0.1 * peak_s:
            idx = i + 1
            break

    p_sum = np.sum(P[idx: idx + plateau])
    cfo_hz = -np.angle(p_sum) * (fs / (2*np.pi*L))

    ok = (peak_s > threshold) and (ratio >= ratio_th)
    return ok, idx, peak_s, float(cfo_hz), float(ratio)

def detect_stf_crosscorr(rx: np.ndarray, stf_ref: np.ndarray, search_len: int) -> Tuple[int, float]:
    L = len(stf_ref)
    search_len = min(search_len, len(rx) - L - 1)
    if search_len <= 0:
        return -1, 0.0
    seg = rx[:search_len + L]
    corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
    denom = np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12
    corr_n = corr / denom
    idx = int(np.argmax(corr_n))
    pk = float(corr_n[idx])
    return idx, pk


# ==============================
# OFDM symbol extraction / channel estimate
# ==============================
def extract_ofdm_symbol_freq(rx: np.ndarray, start_idx: int) -> Optional[np.ndarray]:
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    td = rx[start_idx + N_CP : start_idx + SYMBOL_LEN]
    return np.fft.fft(td) / np.sqrt(N_FFT)

def channel_estimate_from_ltf(rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray,
                              num_symbols: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    Ys: List[np.ndarray] = []
    for i in range(num_symbols):
        Y = extract_ofdm_symbol_freq(rx, ltf_start + i*SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if not Ys:
        return None, None

    Yavg = np.mean(np.stack(Ys, axis=0), axis=0)
    H = np.ones(N_FFT, dtype=np.complex64)
    for b in USED_BINS:
        if np.abs(ltf_freq_ref[b]) > 1e-6:
            H[b] = Yavg[b] / ltf_freq_ref[b]

    if len(Ys) >= 2:
        Ystk = np.stack(Ys, axis=0)[:, USED_BINS]
        noise_var = np.var(Ystk, axis=0)
        sig_var = np.abs(np.mean(Ystk, axis=0))**2
        snr_sc = 10*np.log10(sig_var / (noise_var + 1e-12) + 1e-12)
    else:
        snr_sc = np.zeros(len(USED_BINS), dtype=np.float32)

    return H, snr_sc


# ==============================
# Payload demod + pilot tracking
# ==============================
def demod_payload_symbols(rx_cfo: np.ndarray,
                          payload_start: int,
                          H: np.ndarray,
                          num_syms: int,
                          kp: float,
                          ki: float,
                          repeat: int,
                          bad_evm_th: float,
                          bad_evm_patience: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
    ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)

    phase_acc = 0.0
    freq_acc = 0.0
    prev_phase = None

    all_data_syms = []
    evm_per_sym = []
    bad_streak = 0

    for si in range(num_syms):
        Y = extract_ofdm_symbol_freq(rx_cfo, payload_start + si*SYMBOL_LEN)
        if Y is None:
            break

        Yeq = Y.copy()
        Yeq[USED_BINS] = Yeq[USED_BINS] / (H[USED_BINS] + 1e-12)

        pilot_sign = 1 if (si % 2 == 0) else -1
        exp_p = pilot_sign * pilot_pattern
        rp = Yeq[PILOT_BINS]
        pilot_corr = np.sum(rp * np.conj(exp_p))
        ph = float(np.angle(pilot_corr))

        if prev_phase is not None:
            dph = ph - prev_phase
            while dph > np.pi:
                dph -= 2*np.pi
            while dph < -np.pi:
                dph += 2*np.pi
            freq_acc += ki * dph
        prev_phase = ph

        phase_acc += freq_acc + kp * ph
        Yeq = Yeq * np.exp(-1j * phase_acc).astype(np.complex64)

        data_syms = Yeq[DATA_BINS]
        nearest = ideal[np.argmin(np.abs(data_syms[:, None] - ideal[None, :]), axis=1)]
        evm = float(np.sqrt(np.mean(np.abs(data_syms - nearest)**2)))

        all_data_syms.append(data_syms)
        evm_per_sym.append(evm)

        if evm > bad_evm_th:
            bad_streak += 1
            if bad_streak >= bad_evm_patience:
                break
        else:
            bad_streak = 0

    if not all_data_syms:
        return None, {"num_syms": 0, "mean_evm": 0.0}

    all_data_syms = np.concatenate(all_data_syms, axis=0)
    bits_raw = qpsk_demap(all_data_syms)
    bits_eff = majority_vote(bits_raw, repeat)

    return bits_eff, {
        "num_syms": len(evm_per_sym),
        "mean_evm": float(np.mean(evm_per_sym)) if evm_per_sym else 0.0,
    }

def required_syms_for_bytes(expected_len_bytes: int, repeat: int, slip_bits: int) -> int:
    eff_bits_per_sym = BITS_PER_OFDM_SYM // max(1, repeat)
    needed_bits = expected_len_bytes * 8 + slip_bits
    return int(np.ceil(needed_bits / max(1, eff_bits_per_sym)))


# ==============================
# Demod one capture
# ==============================
def demod_one_capture(rx_raw_iq: np.ndarray,
                      cfg: AppConfig,
                      stf_ref: np.ndarray,
                      ltf_freq_ref: np.ndarray) -> Dict[str, Any]:

    fs = float(cfg.sdr.fs_hz)
    dcfg = cfg.demod
    pcfg = cfg.preamble

    rx = rx_raw_iq.astype(np.complex64)
    if np.median(np.abs(rx)) > 100:
        rx = rx / (2**14)
    rx = rx - np.mean(rx)

    ok_sc, sc_idx, _, cfo_hz, ratio = detect_stf_autocorr(
        rx, fs=fs,
        half_period=N_FFT//2,
        threshold=dcfg.sc_threshold,
        search_len=min(dcfg.search_len, len(rx)-1),
        ratio_th=dcfg.sync_peak_med_ratio_th
    )

    rx_cfo = apply_cfo(rx, cfo_hz, fs)

    if ok_sc:
        win = int(dcfg.xcorr_refine_win)
        s0 = max(0, sc_idx - win)
        s1 = min(len(rx_cfo), sc_idx + win + len(stf_ref))
        seg = rx_cfo[s0:s1]
        corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
        denom = np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12
        corr_n = corr / denom
        loc = int(np.argmax(corr_n))
        stf_idx = s0 + loc
        stf_peak = float(corr_n[loc])
        sync_method = "sc+xc_refine"
    else:
        stf_idx, stf_peak = detect_stf_crosscorr(rx_cfo, stf_ref, search_len=min(dcfg.search_len, len(rx_cfo)-1))
        sync_method = "xc_global"

    if (not ok_sc) and (stf_peak < 0.02):
        return {"status": "no_signal", "reason": "weak_xcorr", "cfo_hz": float(cfo_hz),
                "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
                "sync_method": sync_method}

    if ok_sc and ratio < dcfg.sync_peak_med_ratio_th:
        return {"status": "no_signal", "reason": "ratio_gate", "cfo_hz": float(cfo_hz),
                "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
                "sync_method": sync_method}

    ltf_start = stf_idx + len(stf_ref)
    H, _ = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref, num_symbols=pcfg.ltf_symbols)
    if H is None:
        return {"status": "ltf_fail", "reason": "no_ltf", "cfo_hz": float(cfo_hz),
                "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
                "sync_method": sync_method}

    payload_start = ltf_start + pcfg.ltf_symbols * SYMBOL_LEN

    bits_probe, _ = demod_payload_symbols(
        rx_cfo, payload_start, H,
        num_syms=int(dcfg.probe_syms),
        kp=float(dcfg.kp), ki=float(dcfg.ki),
        repeat=int(dcfg.repeat),
        bad_evm_th=float(dcfg.bad_evm_th),
        bad_evm_patience=int(dcfg.bad_evm_patience)
    )
    if bits_probe is None or bits_probe.size == 0:
        return {"status": "no_packet", "reason": "probe_empty", "cfo_hz": float(cfo_hz),
                "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
                "sync_method": sync_method}

    best = None
    for slip in range(int(dcfg.bit_slip_max) + 1):
        bb = packbits_with_slip(bits_probe, slip)
        ok_h, fid_h, seq_h, total_h, expected_len = parse_header_only(bb)
        if ok_h:
            best = (slip, fid_h, seq_h, total_h, expected_len)
            break

    if best is None:
        return {"status": "no_packet", "reason": "no_magic_in_probe", "cfo_hz": float(cfo_hz),
                "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
                "sync_method": sync_method}

    slip, fid_h, seq_h, total_h, expected_len = best
    req_syms = required_syms_for_bytes(expected_len, repeat=int(dcfg.repeat), slip_bits=int(slip))
    req_syms = min(req_syms, int(dcfg.max_payload_syms_cap))

    bits_full, _ = demod_payload_symbols(
        rx_cfo, payload_start, H,
        num_syms=req_syms,
        kp=float(dcfg.kp), ki=float(dcfg.ki),
        repeat=int(dcfg.repeat),
        bad_evm_th=float(dcfg.bad_evm_th),
        bad_evm_patience=int(dcfg.bad_evm_patience)
    )
    if bits_full is None or bits_full.size == 0:
        return {"status": "no_packet", "reason": "full_empty", "cfo_hz": float(cfo_hz),
                "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
                "sync_method": sync_method, "req_syms": int(req_syms), "slip": int(slip)}

    bb_full = packbits_with_slip(bits_full, int(slip))
    ok, fid, seq, total, payload, expected2, reason = parse_video_packet_full(bb_full)

    if not ok:
        # try slips on full bits
        for slip2 in range(int(dcfg.bit_slip_max) + 1):
            bb2 = packbits_with_slip(bits_full, slip2)
            ok2, fid2, seq2, total2, payload2, expected3, reason2 = parse_video_packet_full(bb2)
            if ok2:
                ok, fid, seq, total, payload, expected2, reason = True, fid2, seq2, total2, payload2, expected3, "ok_slip_fallback"
                slip = slip2
                bb_full = bb2
                break

    if not ok:
        return {"status": "crc_fail", "reason": reason, "cfo_hz": float(cfo_hz),
                "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
                "sync_method": sync_method, "frame_id": int(fid_h), "seq": int(seq_h), "total": int(total_h),
                "expected_len": int(expected_len), "req_syms": int(req_syms), "slip": int(slip)}

    return {"status": "demod_ok", "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio), "stf_peak": float(stf_peak), "rx_peak": float(np.max(np.abs(rx))),
            "sync_method": sync_method,
            "frame_id": int(fid), "seq": int(seq), "total": int(total),
            "payload": payload, "expected_len": int(expected2), "req_syms": int(req_syms), "slip": int(slip)}


# ==============================
# Minimal main (kept short)
# ==============================
def main():
    ap = argparse.ArgumentParser("RF Link Step7 RX (configurable)")
    ap.add_argument("--config", default="", help="Path to JSON config")
    ap.add_argument("--uri", default=None)
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--rx_gain", type=float, default=None)
    ap.add_argument("--buf_size", type=int, default=None)
    ap.add_argument("--repeat", type=int, choices=[1,2,4], default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else AppConfig()
    if args.uri is not None: cfg.sdr.uri = args.uri
    if args.fc is not None: cfg.sdr.fc_hz = args.fc
    if args.fs is not None: cfg.sdr.fs_hz = args.fs
    if args.rx_gain is not None: cfg.sdr.rx_gain_db = args.rx_gain
    if args.buf_size is not None: cfg.sdr.rx_buffer_size = args.buf_size
    if args.repeat is not None: cfg.demod.repeat = args.repeat
    if args.verbose: cfg.runtime.verbose = True

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.runtime.output_dir, f"run_{ts}")
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "effective_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "sdr": asdict(cfg.sdr),
            "preamble": asdict(cfg.preamble),
            "demod": asdict(cfg.demod),
            "runtime": asdict(cfg.runtime),
        }, f, indent=2)

    stf_ref, _ = create_schmidl_cox_stf(num_repeats=cfg.preamble.stf_repeats)
    _, ltf_freq_ref = create_ltf_ref(num_symbols=cfg.preamble.ltf_symbols)

    print("\n" + "="*86)
    print("RF Link Step7 RX (Configurable) - FIXED dataclass defaults")
    print("="*86)
    print(f"run_dir={run_dir}")
    print(f"uri={cfg.sdr.uri}  fc={cfg.sdr.fc_hz/1e6:.1f}MHz  fs={cfg.sdr.fs_hz/1e6:.1f}Msps  rx_gain={cfg.sdr.rx_gain_db}")
    print(f"buf={cfg.sdr.rx_buffer_size}  stf_repeats={cfg.preamble.stf_repeats}  ltf_syms={cfg.preamble.ltf_symbols}")
    print(f"repeat={cfg.demod.repeat}  probe_syms={cfg.demod.probe_syms}  sync_ratio_th={cfg.demod.sync_peak_med_ratio_th}")
    print("="*86)

    import adi
    sdr = adi.Pluto(uri=cfg.sdr.uri)
    sdr.sample_rate = int(cfg.sdr.fs_hz)
    sdr.rx_lo = int(cfg.sdr.fc_hz)
    sdr.rx_rf_bandwidth = int(cfg.sdr.rx_bw_hz if cfg.sdr.rx_bw_hz else cfg.sdr.fs_hz * 1.2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(cfg.sdr.rx_gain_db)
    sdr.rx_enabled_channels = [0]
    sdr.rx_buffer_size = int(cfg.sdr.rx_buffer_size)
    try:
        if hasattr(sdr, "_rxadc") and hasattr(sdr._rxadc, "set_kernel_buffers_count"):
            sdr._rxadc.set_kernel_buffers_count(int(cfg.sdr.kernel_buffers))
    except Exception:
        pass

    for _ in range(4):
        _ = sdr.rx()

    print("Listening ... Ctrl+C to stop")
    start = time.time()
    try:
        for i in range(int(cfg.runtime.max_captures)):
            rx_raw = sdr.rx()
            r = demod_one_capture(rx_raw, cfg, stf_ref, ltf_freq_ref)
            if cfg.runtime.verbose:
                if i < 20 or (i+1) % int(cfg.runtime.sparse_every) == 0:
                    print(f"[{i+1:05d}] {r.get('status','?'):9s} ratio={r.get('sync_ratio',0):6.2f} "
                          f"CFO={r.get('cfo_hz',0):+8.1f}Hz peak={r.get('rx_peak',0):.1f} {r.get('reason','')}")
            else:
                if i < 10 or (i+1) % int(cfg.runtime.sparse_every) == 0:
                    print(f"[{i+1:05d}] {r.get('status','?'):9s} ratio={r.get('sync_ratio',0):6.2f} "
                          f"CFO={r.get('cfo_hz',0):+8.1f}Hz")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
    print(f"elapsed={time.time()-start:.1f}s")


if __name__ == "__main__":
    main()

"""
python3 rf_link_step7_rx_configurable.py --config rx_config_cable_20db_repeat2.json

python3 rf_link_step7_rx_configurable.py \
  --config rx_config_cable_20db_repeat2.json \
  --rx_gain 25 --probe_syms 12 --verbose


python3 rf_link_step7_rx_configurable.py \
  --config rx_config_cable_20db_repeat2.json \
  --rx_gain 40 \
  --buf_size 131072 \
  --repeat 2 \
  --verbose
"""
