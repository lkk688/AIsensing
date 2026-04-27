#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_rx_step6phy.py  – Step 6 RX

Builds on step5 (PlutoSDR acquisition thread → queue → DSP thread, ring buffer,
Numba-JIT xcorr, Schmidl-Cox STF, multi-LTF, 8-combination demod search) and adds:

  • Multi-modulation: BPSK, QPSK (default), QAM8 (8-PSK), QAM16, QAM32
  • BER evaluation mode: compare decoded bits vs known PRNG reference
      --ref_seed / --ref_len  (must match TX)
  • RX-gain sweep within a single run for BER-vs-SNR curves
      --rx_gain_sweep "60,50,40,30"  --gain_step_s 15
  • Per-subcarrier SNR from multi-LTF averaging (added to CSV)
  • Run-end summary JSON: captures.csv + run_summary.json → consumed by analyzer
  • NPZ saves enhanced with modulation + BER metadata

Offline diagnostics: feed captures.csv / run_summary.json to
  rf_stream/analyze_rf_stream_captures.py  (no impact on RX thread speed)
"""

import argparse
import csv
import json
import os
import queue
import threading
import time
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Optional Numba JIT
# ─────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*a, **kw):
        def deco(f): return f
        return deco


# ─────────────────────────────────────────────────────────────────────────────
# PHY parameters  (must match TX)
# ─────────────────────────────────────────────────────────────────────────────
N_FFT    = 64
N_CP     = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array(
    [k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)],
    dtype=int,
)
N_DATA   = len(DATA_SUBCARRIERS)          # 48
MAGIC    = b"AIS1"

def sc_to_bin(k: int) -> int:
    return (k + N_FFT) % N_FFT

DATA_BINS  = np.array([sc_to_bin(int(k)) for k in DATA_SUBCARRIERS], dtype=int)
PILOT_BINS = np.array([sc_to_bin(int(k)) for k in PILOT_SUBCARRIERS], dtype=int)
USED_BINS  = np.array([sc_to_bin(int(k)) for k in range(-26, 27) if k != 0], dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Modulation tables  (exact copy of TX definitions)
# ─────────────────────────────────────────────────────────────────────────────
MOD_BPS = {"bpsk": 1, "qpsk": 2, "qam8": 3, "qam16": 4, "qam32": 5}


def make_constellation(mod: str) -> np.ndarray:
    mod = mod.lower()
    if mod == "bpsk":
        return np.array([-1+0j, 1+0j], dtype=np.complex64)
    if mod == "qpsk":
        return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    if mod == "qam8":
        phase_idx = [0, 1, 3, 2, 6, 7, 5, 4]
        return np.exp(1j * np.pi / 4.0 * np.array(phase_idx, dtype=float)).astype(np.complex64)
    if mod == "qam16":
        g = {0b00: -3, 0b01: -1, 0b11: 1, 0b10: 3}
        t = np.array([g[(i >> 2) & 3] + 1j * g[i & 3] for i in range(16)], dtype=np.complex64)
        return t / np.sqrt(10.0)
    if mod == "qam32":
        pts = np.array(
            [r + 1j * m for r in (-5, -3, -1, 1, 3, 5) for m in (-5, -3, -1, 1, 3, 5)
             if not (abs(r) == 5 and abs(m) == 5)],
            dtype=np.complex64,
        )
        pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
        order = np.lexsort((pts.real, -pts.imag))
        return pts[order]
    raise ValueError(f"Unknown modulation: {mod!r}")


def symbols_to_bits(syms: np.ndarray, table: np.ndarray, bps: int) -> np.ndarray:
    """Nearest-neighbour hard-decision demap → MSB-first bit array."""
    dist    = np.abs(syms[:, np.newaxis] - table[np.newaxis, :]) ** 2  # (N, M)
    indices = np.argmin(dist, axis=1).astype(np.int32)                  # (N,)
    shifts  = np.arange(bps - 1, -1, -1, dtype=np.int32)
    return ((indices[:, np.newaxis] >> shifts) & 1).astype(np.uint8).ravel()


def bits_to_symbols(bits: np.ndarray, table: np.ndarray, bps: int) -> np.ndarray:
    n = len(bits) // bps
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    b      = bits[:n * bps].astype(np.int32).reshape(n, bps)
    powers = 2 ** np.arange(bps - 1, -1, -1, dtype=np.int32)
    return table[(b * powers).sum(axis=1).astype(np.int32)]


def mod_evm(syms: np.ndarray, table: np.ndarray) -> float:
    dist = np.abs(syms[:, np.newaxis] - table[np.newaxis, :]) ** 2
    return float(np.sqrt(np.mean(np.min(dist, axis=1))))


def get_rotation_candidates(mod: str) -> list:
    if mod == "bpsk":
        return [1.0 + 0j, -1.0 + 0j]
    if mod in ("qpsk", "qam16", "qam32"):
        return [1.0 + 0j, 1j, -1.0 + 0j, -1j]
    if mod == "qam8":
        return [np.exp(1j * np.pi / 4.0 * k) for k in range(8)]
    return [1.0 + 0j]


# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT kernels
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def xcorr_mag_valid(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    nx, nh = x.shape[0], h.shape[0]
    nout = nx - nh + 1
    if nout <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(nout, dtype=np.float32)
    for i in range(nout):
        ar, ai = 0.0, 0.0
        for k in range(nh):
            a = x[i + k]; b = h[k]
            ar += a.real * b.real + a.imag * b.imag   # re(a·conj(b))
            ai += a.imag * b.real - a.real * b.imag   # im(a·conj(b))
        out[i] = (ar * ar + ai * ai) ** 0.5
    return out

@njit(cache=True)
def moving_energy(x: np.ndarray, win: int) -> np.ndarray:
    n = x.shape[0]
    if win <= 0 or n < win:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n - win + 1, dtype=np.float32)
    s   = 0.0
    for i in range(win):
        a = x[i]; s += a.real * a.real + a.imag * a.imag
    out[0] = s / win
    for i in range(1, out.shape[0]):
        a0 = x[i - 1]; a1 = x[i + win - 1]
        s -= a0.real * a0.real + a0.imag * a0.imag
        s += a1.real * a1.real + a1.imag * a1.imag
        out[i] = s / win
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHY helpers
# ─────────────────────────────────────────────────────────────────────────────
def bits_from_bytes(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='big').astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    L = (len(bits) // 8) * 8
    return np.packbits(bits[:L]).tobytes() if L > 0 else b""

def scramble_bits(bits: np.ndarray, seed: int = 0x7F) -> np.ndarray:
    state, out = seed, np.zeros_like(bits)
    for i in range(len(bits)):
        b7 = (state >> 6) & 1
        b4 = (state >> 3) & 1
        out[i] = bits[i] ^ b7
        state = ((state << 1) | (b7 ^ b4)) & 0x7F
    return out

def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    if L <= 0:
        return np.zeros(0, dtype=np.uint8)
    return (np.sum(bits[:L].reshape(-1, repeat), axis=1) >= (repeat / 2)).astype(np.uint8)

def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    X   = np.zeros(N_FFT, dtype=np.complex64)
    for i, k in enumerate(np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)):
        X[sc_to_bin(int(k))] = rng.choice([-1.0, 1.0])
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    s = np.tile(x.astype(np.complex64), num_repeats)
    return np.concatenate([s[-N_CP:], s]).astype(np.complex64)

def create_ltf_ref(num_symbols: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    X    = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = 1.0 if (i % 2 == 0) else -1.0
    x       = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    return np.tile(ltf_sym, num_symbols).astype(np.complex64), X

def extract_ofdm_symbol(rx: np.ndarray, start: int) -> Optional[np.ndarray]:
    if start + SYMBOL_LEN > rx.shape[0]:
        return None
    return np.fft.fftshift(np.fft.fft(rx[start + N_CP: start + SYMBOL_LEN]))

def channel_estimate_from_ltf(
    rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray, ltf_symbols: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    Ys = []
    for i in range(ltf_symbols):
        Y = extract_ofdm_symbol(rx, ltf_start + i * SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if not Ys:
        return None, None
    Yavg = np.mean(np.stack(Ys), axis=0)
    H    = np.ones(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for k in used:
        idx = sc_to_bin(int(k))
        if np.abs(ltf_freq_ref[idx]) > 1e-6:
            H[idx] = Yavg[idx] / ltf_freq_ref[idx]
    # Per-subcarrier SNR from variance across LTF symbols
    snr_db = None
    if len(Ys) >= 2:
        Y_arr      = np.stack(Ys)[:, USED_BINS]      # (L, 52)
        noise_var  = np.var(Y_arr, axis=0).real + 1e-10
        sig_var    = np.abs(Yavg[USED_BINS]) ** 2
        snr_db     = 10.0 * np.log10(sig_var / noise_var + 1e-10)
    return H, snr_db

def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, bytes]:
    if len(bb) < 14:
        return False, "too_short", -1, b""
    if bb[:4] != MAGIC:
        return False, "bad_magic", -1, b""
    seq  = int.from_bytes(bb[4:6], "little")
    plen = int.from_bytes(bb[8:10], "little")
    need = 10 + plen + 4
    if len(bb) < need:
        return False, "need_more", seq, b""
    body   = bb[:10 + plen]
    crc_rx = int.from_bytes(bb[10 + plen: 10 + plen + 4], "little")
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc_rx:
        return False, "crc_fail", seq, b""
    return True, "ok", seq, bb[10: 10 + plen]


# ─────────────────────────────────────────────────────────────────────────────
# Ring buffer
# ─────────────────────────────────────────────────────────────────────────────
class RingBuffer:
    def __init__(self, size: int):
        self.size   = int(size)
        self.buf    = np.zeros(self.size, dtype=np.complex64)
        self.w      = 0
        self.filled = False

    def push(self, x: np.ndarray):
        n = x.shape[0]
        if n >= self.size:
            self.buf[:] = x[-self.size:]
            self.w = 0; self.filled = True
            return
        end = self.w + n
        if end <= self.size:
            self.buf[self.w: end] = x
        else:
            n1 = self.size - self.w
            self.buf[self.w:]          = x[:n1]
            self.buf[:end - self.size] = x[n1:]
        self.w = end % self.size
        if not self.filled and self.w == 0:
            self.filled = True

    def get_window(self, length: int) -> np.ndarray:
        length = int(length)
        if length > self.size:
            raise ValueError("window > ring")
        if not self.filled and self.w < length:
            return np.zeros(0, dtype=np.complex64)
        start = (self.w - length) % self.size
        if start < self.w:
            return self.buf[start: self.w].copy()
        return np.concatenate([self.buf[start:], self.buf[:self.w]]).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RxConfig:
    uri: str; fc: float; fs: float
    rx_gain: float; rx_bw: float
    rx_buf: int; kernel_buffers: int

    repeat: int; stf_repeats: int; ltf_symbols: int
    modulation: str; bps: int

    ring_size: int; proc_window: int; proc_hop: int
    energy_win: int; energy_mult: float
    xcorr_search: int; xcorr_topk: int; xcorr_min_peak: float
    ltf_off_sweep: int; max_ofdm_syms_probe: int; max_ofdm_syms_cap: int
    kp: float; ki: float

    ref_seed: int; ref_len: int; chunk_bytes: int
    save_dir: str; save_npz: bool; mode: str

    rx_gain_sweep: list          # [] → single fixed gain
    gain_step_s: float


# ─────────────────────────────────────────────────────────────────────────────
# RX acquisition worker
# ─────────────────────────────────────────────────────────────────────────────
def rx_acq_worker(
    stop_ev: threading.Event,
    q: "queue.Queue[np.ndarray]",
    cfg: RxConfig,
    current_gain: list,          # mutable one-element list: current RX gain value
):
    import adi
    sdr = adi.Pluto(uri=cfg.uri)
    sdr.sample_rate             = int(cfg.fs)
    sdr.rx_lo                   = int(cfg.fc)
    sdr.rx_rf_bandwidth         = int(cfg.rx_bw)
    sdr.gain_control_mode_chan0  = "manual"
    sdr.rx_hardwaregain_chan0    = float(cfg.rx_gain)
    sdr.rx_buffer_size          = int(cfg.rx_buf)
    sdr.rx_enabled_channels     = [0]
    try:
        if hasattr(sdr, "_rxadc") and sdr._rxadc is not None:
            sdr._rxadc.set_kernel_buffers_count(int(cfg.kernel_buffers))
    except Exception:
        pass
    for _ in range(4):
        sdr.rx()

    # Gain-sweep state
    sweep = cfg.rx_gain_sweep
    gain_idx   = 0
    sweep_start = time.time()
    if sweep:
        sdr.rx_hardwaregain_chan0 = float(sweep[0])
        current_gain[0]          = float(sweep[0])
        print(f"[RX] gain sweep start → {sweep[0]} dB")

    print("[RX] acq worker started. rx_buf =", cfg.rx_buf)
    try:
        while not stop_ev.is_set():
            # Possibly advance gain sweep
            if sweep and (time.time() - sweep_start) >= cfg.gain_step_s:
                gain_idx    = (gain_idx + 1) % len(sweep)
                new_g       = float(sweep[gain_idx])
                sdr.rx_hardwaregain_chan0 = new_g
                current_gain[0]          = new_g
                sweep_start = time.time()
                print(f"[RX] gain sweep → {new_g} dB")

            x = sdr.rx().astype(np.complex64)
            if np.median(np.abs(x)) > 100:
                x = x / (2 ** 14)
            x = x - np.mean(x)
            q.put(x, block=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        print("[RX] acq worker stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# DSP thread
# ─────────────────────────────────────────────────────────────────────────────
def dsp_thread(
    stop_ev: threading.Event,
    q: "queue.Queue[np.ndarray]",
    cfg: RxConfig,
    current_gain: list,
):
    import scipy.signal  # noqa: F401 (imported for sweep mode)

    os.makedirs(cfg.save_dir, exist_ok=True)
    good_dir = os.path.join(cfg.save_dir, "good_packets")
    os.makedirs(good_dir, exist_ok=True)

    stf_ref       = create_schmidl_cox_stf(cfg.stf_repeats).astype(np.complex64)
    stf_ref_e     = float(np.sqrt(np.sum(np.abs(stf_ref) ** 2)) + 1e-12)
    ltf_ref_full, ltf_freq_ref = create_ltf_ref(cfg.ltf_symbols)
    ltf_td_ref    = ltf_ref_full[:SYMBOL_LEN].astype(np.complex64)

    const_tbl     = make_constellation(cfg.modulation)
    bps           = cfg.bps
    rot_cands     = get_rotation_candidates(cfg.modulation)

    # Build reference bits for BER tracking
    ref_bits_tx: Optional[np.ndarray] = None
    ref_expected_frames: Optional[list] = None   # list of (seq, expected_frame_bytes)
    if cfg.ref_len > 0:
        rng      = np.random.default_rng(cfg.ref_seed)
        ref_data = rng.integers(0, 256, size=cfg.ref_len, dtype=np.uint8).tobytes()
        cs       = cfg.chunk_bytes if cfg.chunk_bytes > 0 else cfg.ref_len
        ref_chunks = [ref_data[i: i + cs] for i in range(0, len(ref_data), cs)]
        total    = len(ref_chunks)
        ref_expected_frames = []
        for seq, ch in enumerate(ref_chunks):
            fb = _build_packet_bytes(seq, total, ch)
            ref_expected_frames.append((seq, fb))
        print(f"[RX] BER mode: {cfg.ref_len}B ref payload, {total} packets, seed={cfg.ref_seed}")

    ring         = RingBuffer(cfg.ring_size)
    samples_since = 0
    cap = 0; good = 0

    # CSV
    csv_path = os.path.join(cfg.save_dir, "captures.csv")
    fcsv     = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(fcsv, fieldnames=[
        "cap", "status", "reason", "peak",
        "p10", "eg_th", "maxe",
        "xc_best_peak", "xc_best_idx",
        "stf_idx", "ltf_start", "payload_start",
        "probe_evm", "cfo_hz", "snr_db",
        "seq", "payload_len",
        "modulation", "bps", "rx_gain",
        "ber", "n_bits", "n_bit_errors",
    ])
    writer.writeheader()

    # Per-gain accumulator for BER sweep
    gain_ber: dict = {}       # rx_gain → {"n_bits": int, "n_errors": int, "n_pkts": int}

    def _record_ber(rx_gain: float, n_bits: int, n_errors: int):
        k = round(float(rx_gain), 2)
        if k not in gain_ber:
            gain_ber[k] = {"n_bits": 0, "n_errors": 0, "n_pkts": 0}
        gain_ber[k]["n_bits"]   += n_bits
        gain_ber[k]["n_errors"] += n_errors
        gain_ber[k]["n_pkts"]   += 1

    # ─── inner demod function ────────────────────────────────────────────────
    def try_demod_at(rxw: np.ndarray, stf_idx: int):
        ltf0   = stf_idx + len(stf_ref)

        # Schmidl-Cox CFO estimate over STF
        P      = N_FFT // 2
        sc_s   = stf_idx
        sc_e   = min(rxw.shape[0] - P, stf_idx + len(stf_ref) - P)
        if sc_e > sc_s:
            R       = np.sum(rxw[sc_s + P: sc_e + P].astype(np.complex64) *
                             np.conj(rxw[sc_s: sc_e].astype(np.complex64)))
            cfo_est = float(np.angle(R)) * cfg.fs / (2.0 * np.pi * P)
        else:
            cfo_est = 0.0

        n_arr   = np.arange(rxw.shape[0], dtype=np.float32)
        rxw_cfo = (rxw * np.exp(-1j * 2.0 * np.pi * (cfo_est / cfg.fs) * n_arr)).astype(np.complex64)

        # LTF timing xcorr (CFO-corrected)
        search_s = max(0, ltf0 - cfg.ltf_off_sweep)
        search_e = min(rxw.shape[0] - SYMBOL_LEN, ltf0 + cfg.ltf_off_sweep)
        if search_e > search_s and NUMBA_OK:
            sb        = rxw_cfo[search_s: search_e + SYMBOL_LEN].astype(np.complex64)
            corr_ltf  = xcorr_mag_valid(sb, ltf_td_ref)
            ltf_start = search_s + int(np.argmax(corr_ltf)) if corr_ltf.size > 0 else ltf0
        else:
            ltf_start = ltf0

        payload_start = ltf_start + cfg.ltf_symbols * SYMBOL_LEN
        pilot_vals    = np.array([1, 1, 1, -1], dtype=np.complex64)
        rxw_cfo_conj  = np.conj(rxw_cfo)

        bpos = N_DATA * bps  # bits per OFDM symbol for current modulation

        best_ok = False; best_reason = "bad_magic"; best_seq = -1
        best_payload = b""; best_evm = 1e9; best_diag = {}
        best_n_bits = 0; best_n_errors = 0

        for conj_comp in [False, True]:
            rxw_proc = rxw_cfo_conj if conj_comp else rxw_cfo

            H, snr_db_arr = channel_estimate_from_ltf(rxw_proc, ltf_start, ltf_freq_ref, cfg.ltf_symbols)
            if H is None:
                continue
            snr_mean = float(np.mean(snr_db_arr)) if snr_db_arr is not None else 0.0

            data_syms_all = []; evm_list = []
            for si in range(cfg.max_ofdm_syms_probe):
                Y = extract_ofdm_symbol(rxw_proc, payload_start + si * SYMBOL_LEN)
                if Y is None:
                    break
                Ye = Y.copy()
                for k in range(-26, 27):
                    if k == 0: continue
                    idx = sc_to_bin(k)
                    if np.abs(H[idx]) > 1e-6:
                        Ye[idx] = Ye[idx] / (H[idx] + 1e-12)
                # Per-symbol pilot phase correction
                sign = 1 if (si % 2 == 0) else -1
                rp   = Ye[PILOT_BINS]
                ph   = float(np.angle(np.sum(rp * np.conj(sign * pilot_vals))))
                Ye  *= np.exp(-1j * ph).astype(np.complex64)
                ds   = Ye[DATA_BINS]
                evm_list.append(mod_evm(ds, const_tbl))
                data_syms_all.append(ds)

            if not data_syms_all:
                continue
            data_syms_all = np.concatenate(data_syms_all).astype(np.complex64)
            cur_evm       = float(np.mean(evm_list)) if evm_list else 1e9

            for rot in rot_cands:
                syms_rot  = data_syms_all * rot
                bits_raw  = symbols_to_bits(syms_rot, const_tbl, bps)
                bits_mv   = majority_vote(bits_raw, cfg.repeat)
                bits_ds   = scramble_bits(bits_mv)     # descramble
                bb        = bits_to_bytes(bits_ds)
                ok, reason, seq, payload = parse_packet_bytes(bb)
                if ok:
                    pkt_bytes_count  = 4 + 2 + 2 + 2 + len(payload) + 4
                    pkt_syms_needed  = int(np.ceil(pkt_bytes_count * 8 / bpos))
                    pkt_evm          = float(np.mean(evm_list[:pkt_syms_needed])) if evm_list else cur_evm

                    # BER vs reference
                    n_bits = 0; n_errors = 0
                    if ref_expected_frames is not None:
                        seq_mod = seq % len(ref_expected_frames)
                        _, ref_fb = ref_expected_frames[seq_mod]
                        ref_bits_tx_frame = scramble_bits(bits_from_bytes(ref_fb))
                        n_compare  = min(len(bits_mv), len(ref_bits_tx_frame))
                        n_bits     = int(n_compare)
                        n_errors   = int(np.sum(bits_mv[:n_compare] != ref_bits_tx_frame[:n_compare]))

                    best_ok      = True
                    best_reason  = reason
                    best_seq     = seq
                    best_payload = payload
                    best_evm     = pkt_evm
                    best_n_bits  = n_bits
                    best_n_errors = n_errors
                    best_diag    = {
                        "ltf_start": ltf_start, "payload_start": payload_start,
                        "probe_evm": pkt_evm,   "cfo_hz": cfo_est,
                        "snr_db":    snr_mean,
                    }
                    break
                elif best_reason == "bad_magic":
                    best_reason = reason

            if not best_ok and cur_evm < best_evm:
                best_evm = cur_evm
                best_diag = {"ltf_start": ltf_start, "payload_start": payload_start,
                             "probe_evm": cur_evm, "cfo_hz": cfo_est, "snr_db": snr_mean}

            if best_ok:
                break

        if not best_diag:
            best_diag = {"ltf_start": ltf_start, "payload_start": payload_start,
                         "probe_evm": 0.0, "cfo_hz": 0.0, "snr_db": 0.0}
        return best_ok, best_reason, best_seq, best_payload, best_diag, best_n_bits, best_n_errors

    # ─────────────────────────────────────────────────────────────────────────
    print(f"[RX] DSP thread started.  NUMBA_OK={NUMBA_OK}  mod={cfg.modulation}  bps={bps}")
    try:
        while not stop_ev.is_set():
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                continue
            ring.push(x)
            samples_since += x.shape[0]
            if samples_since < cfg.proc_hop:
                continue
            samples_since = 0

            rxw = ring.get_window(cfg.proc_window)
            if rxw.size == 0:
                continue
            cap += 1
            rx_gain_now = float(current_gain[0])
            peak        = float(np.max(np.abs(rxw)))

            e   = (moving_energy(rxw.astype(np.complex64), cfg.energy_win)
                   if NUMBA_OK else np.convolve(np.abs(rxw) ** 2,
                                                np.ones(cfg.energy_win) / cfg.energy_win,
                                                mode="valid"))
            if e.size == 0:
                continue
            p10   = float(np.percentile(e, 10))
            maxe  = float(np.max(e))
            eg_th = float(p10 * cfg.energy_mult)

            if maxe < p10 * 1.1:
                writer.writerow({
                    "cap": cap, "status": "skip", "reason": "energy_low", "peak": peak,
                    "p10": p10, "eg_th": eg_th, "maxe": maxe,
                    "xc_best_peak": 0.0, "xc_best_idx": -1,
                    "stf_idx": -1, "ltf_start": -1, "payload_start": -1,
                    "probe_evm": "", "cfo_hz": "", "snr_db": "",
                    "seq": "", "payload_len": "",
                    "modulation": cfg.modulation, "bps": bps,
                    "rx_gain": rx_gain_now,
                    "ber": "", "n_bits": "", "n_bit_errors": "",
                })
                continue

            search_len = min(cfg.xcorr_search, rxw.size)
            xs         = rxw[:search_len].astype(np.complex64)
            corr       = xcorr_mag_valid(xs, stf_ref) if NUMBA_OK else _numpy_xcorr(xs, stf_ref)
            if corr.size == 0:
                continue
            corr_norm  = corr / stf_ref_e

            k          = min(cfg.xcorr_topk, corr_norm.size)
            top_idx    = np.argpartition(corr_norm, -k)[-k:]
            top_idx    = top_idx[np.argsort(corr_norm[top_idx])[::-1]]
            best_xc_pk = float(corr_norm[top_idx[0]])

            best_ok = False; best_reason = "no_try"; best_seq = -1
            best_payload = b""; best_diag = {}; best_stf = -1
            best_n_bits = 0; best_n_errors = 0

            for cand in top_idx:
                if corr_norm[cand] < cfg.xcorr_min_peak:
                    continue
                ok, reason, seq, payload, diag, nb, ne = try_demod_at(rxw, int(cand))
                if ok:
                    best_ok = True; best_reason = "ok"; best_seq = int(seq)
                    best_payload = payload; best_diag = diag; best_stf = int(cand)
                    best_n_bits = nb; best_n_errors = ne
                    break
                elif best_stf < 0:
                    best_reason = reason; best_diag = diag; best_stf = int(cand)

            status     = "ok" if best_ok else "no_crc"
            payload_len = len(best_payload) if best_ok else 0
            ber_val    = (best_n_errors / best_n_bits) if best_n_bits > 0 else ""

            writer.writerow({
                "cap": cap, "status": status, "reason": best_reason, "peak": peak,
                "p10": p10, "eg_th": eg_th, "maxe": maxe,
                "xc_best_peak": best_xc_pk, "xc_best_idx": int(top_idx[0]),
                "stf_idx": best_stf,
                "ltf_start":    int(best_diag.get("ltf_start",     -1)),
                "payload_start": int(best_diag.get("payload_start", -1)),
                "probe_evm":  float(best_diag.get("probe_evm", 0.0)),
                "cfo_hz":     float(best_diag.get("cfo_hz",    0.0)),
                "snr_db":     float(best_diag.get("snr_db",    0.0)),
                "seq":        (best_seq if best_ok else ""),
                "payload_len": (payload_len if best_ok else ""),
                "modulation": cfg.modulation, "bps": bps,
                "rx_gain":    rx_gain_now,
                "ber":        (float(ber_val) if ber_val != "" else ""),
                "n_bits":     (best_n_bits   if best_ok else ""),
                "n_bit_errors": (best_n_errors if best_ok else ""),
            })

            if best_ok:
                good += 1
                if best_n_bits > 0:
                    _record_ber(rx_gain_now, best_n_bits, best_n_errors)
                outp = os.path.join(good_dir, f"seq_{best_seq:08d}.bin")
                with open(outp, "wb") as f:
                    f.write(best_payload)
                try:
                    ps = best_payload.decode("utf-8")
                except Exception:
                    ps = repr(best_payload[:32])
                ber_str = f" ber={float(ber_val):.4f}" if ber_val != "" else ""
                cfo_str = f" cfo={best_diag.get('cfo_hz', 0):.0f}Hz" if best_diag.get("cfo_hz") else ""
                snr_str = f" snr={best_diag.get('snr_db', 0):.1f}dB" if best_diag.get("snr_db") else ""
                print(f"[RX] OK seq={best_seq} {payload_len}B "
                      f"evm={best_diag.get('probe_evm', 0):.3f} "
                      f"xc={best_xc_pk:.3f}{cfo_str}{snr_str}{ber_str} | {ps}")

                if cfg.save_npz:
                    np.savez_compressed(
                        os.path.join(cfg.save_dir, f"cap_{cap:06d}_ok.npz"),
                        rxw=rxw.astype(np.complex64),
                        corr_norm=corr_norm.astype(np.float32),
                        meta_json=np.bytes_(json.dumps({
                            "cap": cap, "seq": best_seq, "peak": peak,
                            "xc_best_peak": best_xc_pk, "stf_idx": best_stf,
                            "modulation": cfg.modulation, "bps": bps,
                            "rx_gain": rx_gain_now,
                            "n_bits": best_n_bits, "n_bit_errors": best_n_errors,
                            "diag": best_diag, "cfg": asdict(cfg),
                        }).encode()),
                    )

            if cap % 20 == 0:
                fcsv.flush()

    except KeyboardInterrupt:
        pass
    finally:
        fcsv.flush(); fcsv.close()
        _save_run_summary(cfg, cap, good, gain_ber)
        print(f"[RX] DSP thread stopped.  cap={cap}  good={good}")


def _numpy_xcorr(xs: np.ndarray, stf_ref: np.ndarray) -> np.ndarray:
    L    = stf_ref.size
    nout = xs.size - L + 1
    if nout <= 0:
        return np.zeros(0, dtype=np.float32)
    hc   = np.conj(stf_ref)
    corr = np.zeros(nout, dtype=np.float32)
    for i in range(nout):
        corr[i] = np.abs(np.vdot(hc, xs[i: i + L]))
    return corr


def _build_packet_bytes(seq: int, total: int, payload: bytes) -> bytes:
    hdr  = (MAGIC
            + int(seq).to_bytes(2, "little")
            + int(total).to_bytes(2, "little")
            + int(len(payload)).to_bytes(2, "little"))
    body = hdr + payload
    crc  = zlib.crc32(body) & 0xFFFFFFFF
    return body + int(crc).to_bytes(4, "little")


def bits_from_bytes(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='big').astype(np.uint8)


def _save_run_summary(cfg: RxConfig, cap: int, good: int, gain_ber: dict):
    ber_per_gain = {}
    for g, v in gain_ber.items():
        nb = v["n_bits"]
        ne = v["n_errors"]
        ber_per_gain[str(g)] = {
            "rx_gain_dB": g,
            "modulation": cfg.modulation,
            "n_bits":     nb,
            "n_errors":   ne,
            "ber":        ne / nb if nb > 0 else None,
            "n_pkts":     v["n_pkts"],
        }

    summary = {
        "run_id":      os.path.basename(cfg.save_dir),
        "modulation":  cfg.modulation,
        "bps":         cfg.bps,
        "fc_MHz":      cfg.fc / 1e6,
        "fs_MHz":      cfg.fs / 1e6,
        "rx_gain_dB":  cfg.rx_gain,
        "gain_sweep":  cfg.rx_gain_sweep,
        "ref_seed":    cfg.ref_seed,
        "ref_len":     cfg.ref_len,
        "total_cap":   cap,
        "good_pkts":   good,
        "decode_rate": good / max(cap, 1),
        "ber_per_gain": ber_per_gain,
        "end_time":    datetime.now().isoformat(),
    }
    path = os.path.join(cfg.save_dir, "run_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[RX] run summary → {path}")
    if ber_per_gain:
        print("[RX] BER per gain level:")
        for g, v in sorted(ber_per_gain.items(), key=lambda x: float(x[0])):
            ber = v["ber"]
            print(f"     rx_gain={v['rx_gain_dB']:6.1f} dB  "
                  f"mod={v['modulation']:6s}  "
                  f"BER={ber:.4e}  n_bits={v['n_bits']}  pkts={v['n_pkts']}")


# ─────────────────────────────────────────────────────────────────────────────
# Sweep DSP thread (spectrum analyser mode – unchanged from step5)
# ─────────────────────────────────────────────────────────────────────────────
def dsp_sweep_thread(stop_ev, q, cfg, current_gain):
    import scipy.signal
    os.makedirs(cfg.save_dir, exist_ok=True)
    ring          = RingBuffer(cfg.ring_size)
    samples_since = 0; cap = 0
    print("[RX] Sweep thread started.")
    try:
        while not stop_ev.is_set():
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                continue
            ring.push(x); samples_since += x.shape[0]
            if samples_since < cfg.proc_window:
                continue
            samples_since = 0; cap += 1
            rxw   = ring.get_window(cfg.proc_window)
            f, Px = scipy.signal.welch(rxw, fs=cfg.fs, return_onesided=False, nperseg=4096)
            f     = np.fft.fftshift(f); Px = np.fft.fftshift(Px)
            Pdb   = 10 * np.log10(Px + 1e-12)
            top3  = np.argsort(Pdb)[-3:][::-1]
            print(f"[Sweep] cap={cap}  "
                  + "  ".join(f"{f[i]/1e6:+.3f}MHz({Pdb[i]:.1f}dB)" for i in top3))
    except KeyboardInterrupt:
        pass
    finally:
        print("[RX] Sweep thread stopped. cap=", cap)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Step6 PHY RX – multi-modulation + BER tracking")
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc",  type=float, required=True)
    ap.add_argument("--fs",  type=float, required=True)
    ap.add_argument("--rx_gain",  type=float, default=30.0)
    ap.add_argument("--rx_bw",   type=float, default=0.0)
    ap.add_argument("--rx_buf",  type=int,   default=131072)
    ap.add_argument("--kernel_buffers", type=int, default=4)

    ap.add_argument("--modulation", default="qpsk", choices=list(MOD_BPS.keys()))
    ap.add_argument("--repeat",      type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--ring_size",   type=int, default=524288)
    ap.add_argument("--proc_window", type=int, default=262144)
    ap.add_argument("--proc_hop",    type=int, default=65536)
    ap.add_argument("--energy_win",  type=int,   default=512)
    ap.add_argument("--energy_mult", type=float, default=2.5)
    ap.add_argument("--xcorr_search",   type=int,   default=200000)
    ap.add_argument("--xcorr_topk",     type=int,   default=8)
    ap.add_argument("--xcorr_min_peak", type=float, default=0.2)
    ap.add_argument("--ltf_off_sweep",  type=int,   default=16)
    ap.add_argument("--probe_syms",     type=int,   default=16)
    ap.add_argument("--max_syms_cap",   type=int,   default=260)
    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.005)

    ap.add_argument("--ref_seed",  type=int, default=0)
    ap.add_argument("--ref_len",   type=int, default=0,
                    help="BER mode: known PRNG payload length in bytes (must match TX)")
    ap.add_argument("--chunk_bytes", type=int, default=512)

    # Gain sweep: e.g. "60,50,40,30" – change RX gain every gain_step_s seconds
    ap.add_argument("--rx_gain_sweep", type=str, default="",
                    help="Comma-separated RX gain levels for BER sweep (dB)")
    ap.add_argument("--gain_step_s", type=float, default=15.0,
                    help="Seconds per gain level in sweep mode")

    ap.add_argument("--out_root", default="rf_stream_rx_runs")
    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--mode", default="packet", choices=["packet", "sweep"])
    args = ap.parse_args()

    sweep_gains = ([float(x.strip()) for x in args.rx_gain_sweep.split(",") if x.strip()]
                   if args.rx_gain_sweep else [])
    bps         = MOD_BPS[args.modulation]
    run_id      = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir     = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg = RxConfig(
        uri=args.uri, fc=args.fc, fs=args.fs,
        rx_gain=sweep_gains[0] if sweep_gains else args.rx_gain,
        rx_bw=(args.rx_bw if args.rx_bw > 0 else args.fs * 1.2),
        rx_buf=args.rx_buf, kernel_buffers=args.kernel_buffers,
        repeat=args.repeat, stf_repeats=args.stf_repeats, ltf_symbols=args.ltf_symbols,
        modulation=args.modulation, bps=bps,
        ring_size=args.ring_size, proc_window=args.proc_window, proc_hop=args.proc_hop,
        energy_win=args.energy_win, energy_mult=args.energy_mult,
        xcorr_search=args.xcorr_search, xcorr_topk=args.xcorr_topk,
        xcorr_min_peak=args.xcorr_min_peak, ltf_off_sweep=args.ltf_off_sweep,
        max_ofdm_syms_probe=args.probe_syms, max_ofdm_syms_cap=args.max_syms_cap,
        kp=args.kp, ki=args.ki,
        ref_seed=args.ref_seed, ref_len=args.ref_len, chunk_bytes=args.chunk_bytes,
        save_dir=out_dir, save_npz=bool(args.save_npz), mode=args.mode,
        rx_gain_sweep=sweep_gains, gain_step_s=args.gain_step_s,
    )

    # Save config
    with open(os.path.join(out_dir, "rx_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\n" + "=" * 70)
    print("Streaming RX  (Step6 PHY)")
    print("=" * 70)
    print(f"  out_dir={out_dir}  NUMBA_OK={NUMBA_OK}")
    print(f"  modulation={cfg.modulation}  bps={bps}  rx_gain={cfg.rx_gain} dB")
    if sweep_gains:
        print(f"  gain_sweep={sweep_gains}  step={args.gain_step_s}s")
    if cfg.ref_len > 0:
        print(f"  BER mode: ref_seed={cfg.ref_seed}  ref_len={cfg.ref_len}B")
    print("=" * 70)

    current_gain = [float(cfg.rx_gain)]   # shared mutable
    q       = queue.Queue(maxsize=32)
    stop_ev = threading.Event()

    t_acq = threading.Thread(target=rx_acq_worker, args=(stop_ev, q, cfg, current_gain), daemon=True)
    if cfg.mode == "sweep":
        t_dsp = threading.Thread(target=dsp_sweep_thread, args=(stop_ev, q, cfg, current_gain), daemon=True)
    else:
        t_dsp = threading.Thread(target=dsp_thread, args=(stop_ev, q, cfg, current_gain), daemon=True)

    t_acq.start()
    t_dsp.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        time.sleep(0.5)
        print("[RX] exit.")


if __name__ == "__main__":
    main()

"""
# ── Quick-start commands ──────────────────────────────────────────────────────

# Default QPSK (same behavior as step5):
python3 rf_stream_rx_step6phy.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 --rx_gain 30

# QAM16 with BER tracking (ref_seed/ref_len must match TX):
python3 rf_stream_rx_step6phy.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --modulation qam16 --ref_seed 42 --ref_len 512 --save_npz

# BER vs RX gain sweep (automated – single run, 15s per gain level):
python3 rf_stream_rx_step6phy.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --modulation qpsk --ref_seed 42 --ref_len 512 \
  --rx_gain_sweep "60,55,50,45,40,35,30" --gain_step_s 15 \
  --save_npz

# Then analyze:
python3 analyze_rf_stream_captures.py \
  --run_dir rf_stream_rx_runs/run_20260427_120000
"""
