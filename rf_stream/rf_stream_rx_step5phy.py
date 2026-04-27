#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_rx_step5phy.py

PlutoSDR streaming RX with:
- acquisition thread -> queue
- DSP thread with numpy ring buffer
- Numba JIT for heavy kernels:
    - STF xcorr (valid) magnitude
    - moving energy (sliding average)

Decode the same packet format as TX:
MAGIC(4)|SEQ(4)|LEN(2)|PAYLOAD|CRC32(4), CRC over MAGIC..PAYLOAD
"""

import argparse
import csv
import json
import os
import queue
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np


# -------------------------
# Optional numba
# -------------------------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco


# =========================
# Step5 PHY params
# =========================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array(
    [k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))],
    dtype=int,
)
N_DATA = len(DATA_SUBCARRIERS)
BITS_PER_SYM = N_DATA * 2
MAGIC = b"AIS1"

def sc_to_bin(k: int) -> int:
    return (k + N_FFT) % N_FFT

DATA_BINS = np.array([sc_to_bin(int(k)) for k in DATA_SUBCARRIERS], dtype=int)
PILOT_BINS = np.array([sc_to_bin(int(k)) for k in PILOT_SUBCARRIERS], dtype=int)

IDEAL_QPSK = (np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0))


# =========================
# Numba kernels
# =========================
@njit(cache=True)
def xcorr_mag_valid(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Valid-mode correlation magnitude:
      out[i] = |sum_k x[i+k] * conj(h[k])|
    x: complex64, h: complex64
    """
    nx = x.shape[0]
    nh = h.shape[0]
    nout = nx - nh + 1
    if nout <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(nout, dtype=np.float32)
    for i in range(nout):
        acc_re = 0.0
        acc_im = 0.0
        for k in range(nh):
            a = x[i+k]
            b = h[k]
            # a * conj(b)
            br = b.real
            bi = -b.imag
            acc_re += a.real*br - a.imag*bi
            acc_im += a.real*bi + a.imag*br
        out[i] = np.sqrt(acc_re*acc_re + acc_im*acc_im)
    return out

@njit(cache=True)
def moving_energy(x: np.ndarray, win: int) -> np.ndarray:
    """
    Sliding window mean(|x|^2) for win samples. Output length = len(x)-win+1
    """
    n = x.shape[0]
    if win <= 0 or n < win:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n - win + 1, dtype=np.float32)
    s = 0.0
    for i in range(win):
        a = x[i]
        s += a.real*a.real + a.imag*a.imag
    out[0] = s / win
    for i in range(1, out.shape[0]):
        a0 = x[i-1]
        a1 = x[i+win-1]
        s -= a0.real*a0.real + a0.imag*a0.imag
        s += a1.real*a1.real + a1.imag*a1.imag
        out[i] = s / win
    return out


# =========================
# PHY helpers
# =========================
def bits_to_bytes(bits: np.ndarray) -> bytes:
    L = (len(bits)//8)*8
    if L <= 0:
        return b""
    return np.packbits(bits[:L]).tobytes()

def scramble_bits(bits: np.ndarray, seed: int = 0x7F) -> np.ndarray:
    state = seed
    out = np.zeros_like(bits)
    for i in range(len(bits)):
        b7 = (state >> 6) & 1
        b4 = (state >> 3) & 1
        feedback = b7 ^ b4
        out[i] = bits[i] ^ b7
        state = ((state << 1) | feedback) & 0x7F
    return out

def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    # hard decision by quadrant
    bits = np.zeros(symbols.size * 2, dtype=np.uint8)
    re = (symbols.real >= 0)
    im = (symbols.imag >= 0)
    # mapping inverse of TX Gray mapping
    # 00: re>=0 im>=0
    # 01: re<0  im>=0
    # 11: re<0  im<0
    # 10: re>=0 im<0
    bits[0::2] = np.where(re, np.where(im, 0, 1), np.where(im, 0, 1))  # b0
    bits[1::2] = np.where(re, np.where(im, 0, 0), np.where(im, 1, 1))  # b1
    # Above is a bit tricky; safer explicit:
    # We'll overwrite with explicit per-sample (still fast enough).
    for i in range(symbols.size):
        r = symbols[i].real >= 0
        m = symbols[i].imag >= 0
        if r and m:
            bits[2*i], bits[2*i+1] = 0, 0
        elif (not r) and m:
            bits[2*i], bits[2*i+1] = 0, 1
        elif (not r) and (not m):
            bits[2*i], bits[2*i+1] = 1, 1
        else:
            bits[2*i], bits[2*i+1] = 1, 0
    return bits

def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits
    L = (len(bits)//repeat)*repeat
    if L <= 0:
        return np.zeros(0, dtype=np.uint8)
    x = bits[:L].reshape(-1, repeat)
    return (np.sum(x, axis=1) >= (repeat/2)).astype(np.uint8)

def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i] + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x.astype(np.complex64), num_repeats)
    stf_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_cp

def create_ltf_ref(num_symbols: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X

def extract_ofdm_symbol(rx: np.ndarray, start: int) -> Optional[np.ndarray]:
    if start + SYMBOL_LEN > rx.shape[0]:
        return None
    td = rx[start + N_CP : start + SYMBOL_LEN]
    return np.fft.fftshift(np.fft.fft(td))

def channel_estimate_from_ltf(rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray, ltf_symbols: int) -> Optional[np.ndarray]:
    Ys = []
    for i in range(ltf_symbols):
        Y = extract_ofdm_symbol(rx, ltf_start + i*SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if not Ys:
        return None
    Yavg = np.mean(np.stack(Ys, axis=0), axis=0)
    H = np.ones(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for k in used:
        idx = sc_to_bin(int(k))
        if np.abs(ltf_freq_ref[idx]) > 1e-6:
            H[idx] = Yavg[idx] / ltf_freq_ref[idx]
    return H

def apply_cfo(rx: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(rx.shape[0], dtype=np.float32)
    return (rx * np.exp(-1j*2*np.pi*(cfo_hz/fs)*n)).astype(np.complex64)

def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, bytes]:
    """
    returns: (ok, reason, seq, payload)
    Packet format: MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    CRC covers MAGIC..PAYLOAD.
    """
    if len(bb) < 14:
        return False, "too_short", -1, b""
    if bb[:4] != MAGIC:
        return False, "bad_magic", -1, b""
    seq = int.from_bytes(bb[4:6], "little")
    plen = int.from_bytes(bb[8:10], "little")
    need = 10 + plen + 4
    if len(bb) < need:
        return False, "need_more", seq, b""
    body = bb[:10+plen]
    crc_rx = int.from_bytes(bb[10+plen:10+plen+4], "little")
    crc_ok = (zlib.crc32(body) & 0xFFFFFFFF) == crc_rx
    if not crc_ok:
        return False, "crc_fail", seq, b""
    payload = bb[10:10+plen]
    return True, "ok", seq, payload


# =========================
# Ring buffer
# =========================
class RingBuffer:
    def __init__(self, size: int):
        self.size = int(size)
        self.buf = np.zeros(self.size, dtype=np.complex64)
        self.w = 0
        self.filled = False

    def push(self, x: np.ndarray):
        n = x.shape[0]
        if n >= self.size:
            self.buf[:] = x[-self.size:]
            self.w = 0
            self.filled = True
            return
        end = self.w + n
        if end <= self.size:
            self.buf[self.w:end] = x
        else:
            n1 = self.size - self.w
            self.buf[self.w:] = x[:n1]
            self.buf[:end - self.size] = x[n1:]
        self.w = end % self.size
        if not self.filled and self.w == 0:
            self.filled = True

    def get_window(self, length: int) -> np.ndarray:
        length = int(length)
        if length > self.size:
            raise ValueError("window length > ring size")
        if not self.filled and self.w < length:
            # not enough yet
            return np.zeros(0, dtype=np.complex64)
        start = (self.w - length) % self.size
        if start < self.w:
            return self.buf[start:self.w].copy()
        else:
            return np.concatenate([self.buf[start:], self.buf[:self.w]]).copy()


# =========================
# Config
# =========================
@dataclass
class RxConfig:
    uri: str
    fc: float
    fs: float
    rx_gain: float
    rx_bw: float
    rx_buf: int
    kernel_buffers: int

    repeat: int
    stf_repeats: int
    ltf_symbols: int

    ring_size: int
    proc_window: int            # DSP operates on last proc_window samples
    proc_hop: int               # how often to run detection (samples)
    energy_win: int
    energy_mult: float          # energy gate threshold = p10 * energy_mult

    xcorr_search: int           # search region (samples) inside proc_window
    xcorr_topk: int
    xcorr_min_peak: float

    ltf_off_sweep: int          # +- sweep for ltf start fine
    max_ofdm_syms_probe: int
    max_ofdm_syms_cap: int
    kp: float
    ki: float

    save_dir: str
    save_npz: bool
    mode: str


# =========================
# Threads
# =========================
def rx_acq_worker(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: RxConfig):
    import adi
    sdr = adi.Pluto(uri=cfg.uri)
    sdr.sample_rate = int(cfg.fs)
    sdr.rx_lo = int(cfg.fc)
    sdr.rx_rf_bandwidth = int(cfg.rx_bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(cfg.rx_gain)
    sdr.rx_buffer_size = int(cfg.rx_buf)
    sdr.rx_enabled_channels = [0]
    try:
        if hasattr(sdr, "_rxadc") and sdr._rxadc is not None:
            sdr._rxadc.set_kernel_buffers_count(int(cfg.kernel_buffers))
    except Exception:
        pass

    # flush
    for _ in range(4):
        _ = sdr.rx()

    print("[RX] acq worker started. rx_buf =", cfg.rx_buf)
    try:
        while not stop_ev.is_set():
            x = sdr.rx().astype(np.complex64)
            # autoscale
            if np.median(np.abs(x)) > 100:
                x = x / (2**14)
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


def dsp_sweep_thread(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: RxConfig):
    import scipy.signal
    os.makedirs(cfg.save_dir, exist_ok=True)
    ring = RingBuffer(cfg.ring_size)
    samples_since = 0
    cap = 0
    
    print("[RX] Sweep thread started.")
    try:
        while not stop_ev.is_set():
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                continue

            ring.push(x)
            samples_since += x.shape[0]
            if samples_since < cfg.proc_window:
                continue
            samples_since = 0
            
            rxw = ring.get_window(cfg.proc_window)
            if rxw.size == 0:
                continue
            cap += 1
            
            f, Pxx = scipy.signal.welch(rxw, fs=cfg.fs, return_onesided=False, nperseg=4096)
            f = np.fft.fftshift(f)
            Pxx = np.fft.fftshift(Pxx)
            Pxx_dB = 10 * np.log10(Pxx + 1e-12)
            
            top_indices = np.argsort(Pxx_dB)[-3:][::-1]
            peaks_str = ", ".join([f"{f[i]/1e6:+.3f} MHz ({Pxx_dB[i]:.1f} dB)" for i in top_indices])
            
            print(f"[RX Sweep] cap={cap} Peaks: {peaks_str}")
            
    except KeyboardInterrupt:
        pass
    finally:
        print("[RX] Sweep thread stopped. cap=", cap)


def dsp_thread(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: RxConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    good_dir = os.path.join(cfg.save_dir, "good_packets")
    os.makedirs(good_dir, exist_ok=True)

    stf_ref = create_schmidl_cox_stf(cfg.stf_repeats).astype(np.complex64)
    stf_ref_e = float(np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
    ltf_ref_full, ltf_freq_ref = create_ltf_ref(cfg.ltf_symbols)
    ltf_td_ref = ltf_ref_full[:SYMBOL_LEN].astype(np.complex64)  # one LTF symbol for timing xcorr

    ring = RingBuffer(cfg.ring_size)
    samples_since = 0

    csv_path = os.path.join(cfg.save_dir, "captures.csv")
    fcsv = open(csv_path, "w", newline="")
    writer = csv.DictWriter(
        fcsv,
        fieldnames=[
            "cap","status","reason","peak",
            "p10","eg_th","maxe",
            "xc_best_peak","xc_best_idx",
            "stf_idx","ltf_start","payload_start",
            "probe_evm","cfo_hz","seq","payload_len"
        ]
    )
    writer.writeheader()

    cap = 0
    good = 0

    def try_demod_at(rxw: np.ndarray, stf_idx: int) -> Tuple[bool, str, int, bytes, dict]:
        """
        Demod starting at stf_idx (index within rxw).
        Tries 8 combinations: 2 conjugation modes (normal + conj) x 4 QPSK rotations.
        TX hardware conjugates signal (np.conj(buf)*4096), so conj(rxw) undoes this.
        """
        ltf0 = stf_idx + len(stf_ref)

        # --- CFO estimation via Schmidl-Cox on the STF ---
        # STF has period P = N_FFT//2 = 32 samples.  Use as many complete
        # periods as available inside the STF region.
        P = N_FFT // 2
        sc_s = stf_idx
        sc_e = min(rxw.shape[0] - P, stf_idx + len(stf_ref) - P)
        if sc_e > sc_s:
            R = np.sum(rxw[sc_s + P : sc_e + P].astype(np.complex64) *
                       np.conj(rxw[sc_s : sc_e].astype(np.complex64)))
            cfo_est = float(np.angle(R)) * cfg.fs / (2.0 * np.pi * P)
        else:
            cfo_est = 0.0

        # Apply CFO correction to the whole window.
        # After correction conj(rxw_cfo) implicitly has the correct -cfo,
        # so a single correction handles both conj_comp modes.
        n_arr = np.arange(rxw.shape[0], dtype=np.float32)
        rxw_cfo = (rxw * np.exp(-1j * 2.0 * np.pi * (cfo_est / cfg.fs) * n_arr)
                   ).astype(np.complex64)

        # LTF timing: xcorr magnitude is conjugation-invariant, use CFO-corrected rxw
        search_s = max(0, ltf0 - cfg.ltf_off_sweep)
        search_e = min(rxw.shape[0] - SYMBOL_LEN, ltf0 + cfg.ltf_off_sweep)
        if search_e > search_s and NUMBA_OK:
            search_buf = rxw_cfo[search_s : search_e + SYMBOL_LEN].astype(np.complex64)
            corr_ltf = xcorr_mag_valid(search_buf, ltf_td_ref)
            ltf_start = search_s + int(np.argmax(corr_ltf)) if corr_ltf.size > 0 else ltf0
        else:
            ltf_start = ltf0

        payload_start = ltf_start + cfg.ltf_symbols * SYMBOL_LEN
        pilot_vals = np.array([1, 1, 1, -1], dtype=np.complex64)

        # Precompute conjugated CFO-corrected window (for TX conj compensation)
        rxw_cfo_conj = np.conj(rxw_cfo)

        best_ok = False
        best_reason = "bad_magic"
        best_seq = -1
        best_payload = b""
        best_evm = 1e9
        best_diag_inner = {}

        # Outer loop: try with and without conjugate compensation
        for conj_comp in [False, True]:
            rxw_proc = rxw_cfo_conj if conj_comp else rxw_cfo

            H = channel_estimate_from_ltf(rxw_proc, ltf_start, ltf_freq_ref, cfg.ltf_symbols)
            if H is None:
                continue

            data_syms_all = []
            evm_list = []

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

                # Per-symbol pilot phase correction: measure phase from pilots and apply directly.
                # This is exact for AWGN + slow CFO (no inter-symbol phase accumulation error).
                sign = 1 if (si % 2 == 0) else -1
                rp = Ye[PILOT_BINS]
                ph = float(np.angle(np.sum(rp * np.conj(sign * pilot_vals))))
                Ye *= np.exp(-1j * ph).astype(np.complex64)

                ds = Ye[DATA_BINS]
                nearest = np.empty_like(ds)
                for i in range(ds.size):
                    best_j = 0
                    md = 1e9
                    for j in range(4):
                        d = ds[i] - IDEAL_QPSK[j]
                        dd = d.real * d.real + d.imag * d.imag
                        if dd < md:
                            md = dd
                            best_j = j
                    nearest[i] = IDEAL_QPSK[best_j]
                evm_list.append(float(np.sqrt(np.mean(np.abs(ds - nearest) ** 2))))
                data_syms_all.append(ds)

            if not data_syms_all:
                continue

            data_syms_all = np.concatenate(data_syms_all).astype(np.complex64)
            cur_evm = float(np.mean(evm_list)) if evm_list else 1e9

            # Try all 4 QPSK rotations; TX scrambles bits so descramble before parsing
            for rot in range(4):
                syms_rot = data_syms_all * (1j ** rot)
                bits_raw = qpsk_demap(syms_rot)
                bits = majority_vote(bits_raw, cfg.repeat)
                bits = scramble_bits(bits)
                bb = bits_to_bytes(bits)
                ok, reason, seq, payload = parse_packet_bytes(bb)
                if ok:
                    # EVM only over the OFDM symbols that carry the actual packet bytes
                    pkt_byte_count = 4 + 2 + 2 + 2 + len(payload) + 4
                    pkt_syms_needed = int(np.ceil(pkt_byte_count * 8 / BITS_PER_SYM))
                    pkt_evm = float(np.mean(evm_list[:pkt_syms_needed])) if evm_list else cur_evm
                    best_ok = True
                    best_reason = reason
                    best_seq = seq
                    best_payload = payload
                    best_evm = pkt_evm
                    best_diag_inner = {"ltf_start": ltf_start, "payload_start": payload_start, "probe_evm": pkt_evm, "cfo_hz": cfo_est}
                    break
                elif best_reason == "bad_magic":
                    best_reason = reason

            # Track lowest EVM for diagnostics even if not decoded
            if not best_ok and cur_evm < best_evm:
                best_evm = cur_evm
                best_diag_inner = {"ltf_start": ltf_start, "payload_start": payload_start, "probe_evm": cur_evm}

            if best_ok:
                break

        if not best_diag_inner:
            best_diag_inner = {"ltf_start": ltf_start, "payload_start": payload_start, "probe_evm": 0.0}

        diag = {"ltf_q": 0.0, **best_diag_inner}
        return best_ok, best_reason, best_seq, best_payload, diag

    last_proc_w = None

    print("[RX] DSP thread started. NUMBA_OK =", NUMBA_OK)

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

            peak = float(np.max(np.abs(rxw)))
            # energy gate
            e = moving_energy(rxw.astype(np.complex64), cfg.energy_win) if NUMBA_OK else np.convolve(np.abs(rxw)**2, np.ones(cfg.energy_win)/cfg.energy_win, mode="valid")
            if e.size == 0:
                continue
            p10 = float(np.percentile(e, 10))
            maxe = float(np.max(e))
            eg_th = float(p10 * cfg.energy_mult)

            # Relaxed energy gating: only skip if maxe is practically indistinguishable from noise
            if maxe < p10 * 1.1:
                writer.writerow({
                    "cap": cap, "status": "skip", "reason": "energy_low", "peak": peak,
                    "p10": p10, "eg_th": eg_th, "maxe": maxe,
                    "xc_best_peak": 0.0, "xc_best_idx": -1,
                    "stf_idx": -1, "ltf_start": -1, "payload_start": -1,
                    "probe_evm": "", "cfo_hz": "", "seq": "", "payload_len": ""
                })
                continue

            # xcorr on search region
            search_len = min(cfg.xcorr_search, rxw.size)
            xs = rxw[:search_len].astype(np.complex64)

            if NUMBA_OK:
                corr = xcorr_mag_valid(xs, stf_ref)
            else:
                # numpy fallback (slower)
                L = stf_ref.size
                nout = xs.size - L + 1
                if nout <= 0:
                    corr = np.zeros(0, dtype=np.float32)
                else:
                    corr = np.zeros(nout, dtype=np.float32)
                    hc = np.conj(stf_ref)
                    for i in range(nout):
                        corr[i] = np.abs(np.vdot(hc, xs[i:i+L]))

            if corr.size == 0:
                continue

            # normalize by stf energy for comparability
            corr_norm = corr / stf_ref_e
            # take top-k peaks
            k = min(cfg.xcorr_topk, corr_norm.size)
            top_idx = np.argpartition(corr_norm, -k)[-k:]
            top_idx = top_idx[np.argsort(corr_norm[top_idx])[::-1]]

            best_ok = False
            best_reason = "no_try"
            best_seq = -1
            best_payload = b""
            best_diag = {}
            best_stf = -1
            best_xc_peak = float(corr_norm[top_idx[0]])

            # try each candidate
            for cand in top_idx:
                if corr_norm[cand] < cfg.xcorr_min_peak:
                    continue
                ok, reason, seq, payload, diag = try_demod_at(rxw, int(cand))
                if ok:
                    best_ok = True
                    best_reason = "ok"
                    best_seq = int(seq)
                    best_payload = payload
                    best_diag = diag
                    best_stf = int(cand)
                    break
                else:
                    # keep the best diag of the first candidate (for logging)
                    if best_stf < 0:
                        best_reason = reason
                        best_diag = diag
                        best_stf = int(cand)

            status = "ok" if best_ok else "no_crc"
            payload_len = len(best_payload) if best_ok else 0

            writer.writerow({
                "cap": cap, "status": status, "reason": best_reason, "peak": peak,
                "p10": p10, "eg_th": eg_th,
                "maxe": maxe,
                "xc_best_peak": best_xc_peak, "xc_best_idx": int(top_idx[0]),
                "stf_idx": best_stf,
                "ltf_start": int(best_diag.get("ltf_start", -1)),
                "payload_start": int(best_diag.get("payload_start", -1)),
                "probe_evm": float(best_diag.get("probe_evm", 0.0)),
                "cfo_hz": float(best_diag.get("cfo_hz", 0.0)),
                "seq": (best_seq if best_ok else ""),
                "payload_len": (payload_len if best_ok else ""),
            })

            if best_ok:
                good += 1
                outp = os.path.join(good_dir, f"seq_{best_seq:08d}.bin")
                with open(outp, "wb") as f:
                    f.write(best_payload)
                try:
                    payload_str = best_payload.decode("utf-8")
                except Exception:
                    payload_str = repr(best_payload[:32])
                cfo_str = f" cfo={best_diag.get('cfo_hz',0):.0f}Hz" if best_diag.get('cfo_hz') else ""
                print(f"[RX] OK seq={best_seq} payload={payload_len}B evm={best_diag.get('probe_evm',0):.3f} xc={best_xc_peak:.3f}{cfo_str} | {payload_str}")

                if cfg.save_npz:
                    npz_path = os.path.join(cfg.save_dir, f"cap_{cap:06d}_ok.npz")
                    meta = {
                        "cap": cap,
                        "seq": best_seq,
                        "peak": peak,
                        "xc_best_peak": best_xc_peak,
                        "stf_idx": best_stf,
                        "diag": best_diag,
                        "cfg": cfg.__dict__,
                    }
                    np.savez_compressed(
                        npz_path,
                        rxw=rxw.astype(np.complex64),
                        corr_norm=corr_norm.astype(np.float32),
                        meta_json=np.bytes_(json.dumps(meta).encode("utf-8")),
                    )

            # flush csv
            if cap % 20 == 0:
                fcsv.flush()

    except KeyboardInterrupt:
        pass
    finally:
        fcsv.flush()
        fcsv.close()
        print("[RX] DSP thread stopped. cap=", cap, "good=", good)


def main():
    ap = argparse.ArgumentParser("Streaming Step5 PHY RX (ring buffer + threads + numba)")
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, required=True)
    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--rx_gain", type=float, default=30.0)
    ap.add_argument("--rx_bw", type=float, default=0.0)
    ap.add_argument("--rx_buf", type=int, default=131072)
    ap.add_argument("--kernel_buffers", type=int, default=4)

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--ring_size", type=int, default=524288)
    ap.add_argument("--proc_window", type=int, default=262144)
    ap.add_argument("--proc_hop", type=int, default=65536)

    ap.add_argument("--energy_win", type=int, default=512)
    ap.add_argument("--energy_mult", type=float, default=2.5)

    ap.add_argument("--xcorr_search", type=int, default=200000)
    ap.add_argument("--xcorr_topk", type=int, default=8)
    ap.add_argument("--xcorr_min_peak", type=float, default=0.2)

    ap.add_argument("--ltf_off_sweep", type=int, default=16)
    ap.add_argument("--probe_syms", type=int, default=16)
    ap.add_argument("--max_syms_cap", type=int, default=260)
    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.005)

    ap.add_argument("--out_root", default="rf_stream_rx_runs")
    ap.add_argument("--save_npz", action="store_true")
    
    ap.add_argument("--mode", type=str, default="packet", choices=["packet", "sweep"])

    args = ap.parse_args()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, run_id)

    cfg = RxConfig(
        uri=args.uri,
        fc=args.fc,
        fs=args.fs,
        rx_gain=args.rx_gain,
        rx_bw=(args.rx_bw if args.rx_bw > 0 else args.fs*1.2),
        rx_buf=args.rx_buf,
        kernel_buffers=args.kernel_buffers,

        repeat=args.repeat,
        stf_repeats=args.stf_repeats,
        ltf_symbols=args.ltf_symbols,

        ring_size=args.ring_size,
        proc_window=args.proc_window,
        proc_hop=args.proc_hop,

        energy_win=args.energy_win,
        energy_mult=args.energy_mult,

        xcorr_search=args.xcorr_search,
        xcorr_topk=args.xcorr_topk,
        xcorr_min_peak=args.xcorr_min_peak,

        ltf_off_sweep=args.ltf_off_sweep,
        max_ofdm_syms_probe=args.probe_syms,
        max_ofdm_syms_cap=args.max_syms_cap,
        kp=args.kp,
        ki=args.ki,

        save_dir=out_dir,
        save_npz=bool(args.save_npz),
        mode=args.mode,
    )

    print("\n" + "="*78)
    print("Streaming RX (Step5 PHY)")
    print("="*78)
    print("out_dir:", out_dir)
    print("NUMBA_OK:", NUMBA_OK)
    print("cfg:", cfg)
    print("="*78)

    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
    stop_ev = threading.Event()

    t_acq = threading.Thread(target=rx_acq_worker, args=(stop_ev, q, cfg), daemon=True)
    if cfg.mode == "sweep":
        t_dsp = threading.Thread(target=dsp_sweep_thread, args=(stop_ev, q, cfg), daemon=True)
    else:
        t_dsp = threading.Thread(target=dsp_thread, args=(stop_ev, q, cfg), daemon=True)
        
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
* 采集线程：固定 rx_buffer_size 不断 sdr.rx() → 入队（不做 DSP）
* DSP 线程：维护 ring buffer，定期从 ring 中取窗口做检测/解调
* Numba：
    * xcorr_mag_valid()：STF 互相关（complex dot，输出 |corr|）
    * moving_energy()：滑动能量（用于能量门控、粗定位）
* 输出：
    * run_dir/captures.csv（每次检测记录）
    * run_dir/good_packets/seq_XXXX.bin（成功的 payload）
    * 可选：保存 debug npz（窗口波形+索引+指标），便于离线复现

Added a corresponding dsp_sweep_thread that uses scipy.signal.welch to calculate the Power Spectral Density (PSD) of the incoming signal every window. It locates the peaks in the spectrum and outputs the top 3 frequencies and their corresponding power to validate the physical link.

python3 rf_stream_rx_step5phy.py --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 --rx_gain 30 --mode sweep


python3 rf_stream_rx_step5phy.py \
  --uri ip:192.168.2.2 \
  --fc 2.3e9 --fs 3e6 \
  --rx_gain 30 \
  --rx_buf 131072 \
  --ring_size 524288 \
  --proc_window 262144 \
  --proc_hop 65536 \
  --xcorr_topk 8 \
  --xcorr_min_peak 0.2 \
  --energy_mult 2.5 \
  --probe_syms 16 \
  --save_npz

  
python3 rf_stream_rx_step5phy.py --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 --rx_gain 30 --rx_buf 131072 --ring_size 524288 --proc_window 262144 --proc_hop 65536 --xcorr_topk 8 --xcorr_min_peak 0.2 --energy_mult 2.5 --probe_syms 16 --save_npz
"""