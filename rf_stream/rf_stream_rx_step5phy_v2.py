#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming RX (Step5 PHY) - v2
- Ring buffer + acquisition thread + DSP thread
- Robust energy gating: median + MAD -> sigma, threshold = med + k*sigma
- Run-length gating: require continuous run over threshold
- Normalized cross-correlation (NCC): corr / (||xseg|| * ||stf||)
- Optional one-pole highpass in acquisition to suppress LO leakage / DC drift
- Sweep mode: prints top peaks ignoring DC neighborhood
- Prints decoded packet in terminal (payload utf-8 if possible)

Packet format (TX must match):
  MAGIC(4) | SEQ(4) | LEN(2) | PAYLOAD | CRC32(4) , where CRC32 covers (MAGIC..PAYLOAD) i.e. header+payload.

Test:
  RX:
    python3 rf_stream_rx_step5phy_v2.py --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 --rx_gain 30 \
      --rx_buf 131072 --ring_size 524288 --proc_window 262144 --proc_hop 65536 \
      --xcorr_topk 8 --xcorr_min_peak 0.2 --probe_syms 16 --save_npz --verbose

  Sweep (ignore DC):
    python3 rf_stream_rx_step5phy_v2.py --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 --rx_gain 30 --mode sweep
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
import matplotlib
matplotlib.use('Agg') # non-interactive
import matplotlib.pyplot as plt

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




def make_reference_payload(seed: int, length: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()

def ber_bits(a: bytes, b: bytes) -> tuple[int,int,float]:
    n = min(len(a), len(b))
    if n == 0:
        return 0, 0, 0.0
    aa = np.unpackbits(np.frombuffer(a[:n], dtype=np.uint8), bitorder='big')
    bb = np.unpackbits(np.frombuffer(b[:n], dtype=np.uint8), bitorder='big')
    err = int(np.sum(aa != bb))
    tot = int(len(aa))
    return err, tot, float(err/(tot+1e-12))

def figure_worker_thread(stop_ev: threading.Event, fig_q: queue.Queue):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("[FIG] Figure worker started.")
    while not stop_ev.is_set():
        try:
            item = fig_q.get(timeout=0.5)
        except queue.Empty:
            continue
            
        (path, rx, fs, tone_info, corr, idxs, peaks, chosen, ltf_info, demod_info, parse_info, title) = item
        try:
            stf_idx = chosen.get("stf_idx", -1)
            stf_ncc = chosen.get("stf_ncc", 0.0)

            fig = plt.figure(figsize=(18, 10))

            # 1) time |rx|
            ax = fig.add_subplot(3,4,1)
            N = min(len(rx), 80000)
            ax.plot(np.abs(rx[:N]))
            if stf_idx >= 0 and stf_idx < N:
                ax.axvline(stf_idx, color="r", linestyle="--", label="chosen STF")
            ax.set_title("Time |rx| (first 80k)")
            ax.grid(True)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper right")

            # 2) tone FFT
            ax = fig.add_subplot(3,4,2)
            if tone_info is not None:
                freqs, db, detected, pk_db = tone_info
                ax.plot(freqs/1e3, db)
                ax.set_title(f"Tone FFT (dB)\ndetected={detected/1e3:.1f}kHz peak={pk_db:.1f}dB")
                ax.grid(True)
            else:
                ax.axis("off")

            # 3) STF NCC corr
            ax = fig.add_subplot(3,4,3)
            if len(corr) > 0:
                ax.plot(idxs, corr)
                for s,v in peaks:
                    ax.axvline(s, color="k", alpha=0.12)
                if stf_idx >= 0:
                    ax.axvline(stf_idx, color="r", linestyle="--", label=f"chosen {stf_ncc:.3f}")
                ax.set_title("STF NCC corr")
                ax.grid(True)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
            else:
                ax.axis("off")

            # 4) LTF mags Y0/Y1
            ax = fig.add_subplot(3,4,4)
            if ltf_info is not None and ltf_info.get("Y0") is not None:
                ax.plot(np.abs(ltf_info["Y0"]), label="|Y0|")
                ax.plot(np.abs(ltf_info["Y1"]), label="|Y1|", alpha=0.8)
                ax.set_title(f"LTF FFT mags (2 repeats)\nscore={ltf_info.get('score',0):.3f}")
                ax.grid(True)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
            else:
                ax.axis("off")

            # 5) |H|
            ax = fig.add_subplot(3,4,5)
            if ltf_info is not None and ltf_info.get("H") is not None:
                ax.plot(np.abs(ltf_info["H"]))
                ax.set_title("|H| (fftshift bins)")
                ax.grid(True)
            else:
                ax.axis("off")

            # 6) angle(H)
            ax = fig.add_subplot(3,4,6)
            if ltf_info is not None and ltf_info.get("H") is not None:
                ax.plot(np.unwrap(np.angle(ltf_info["H"])))
                ax.set_title("angle(H) unwrap")
                ax.grid(True)
            else:
                ax.axis("off")

            # 7) pilot power
            ax = fig.add_subplot(3,4,7)
            if demod_info is not None and demod_info.get("pilot_pwr") is not None and len(demod_info["pilot_pwr"]) > 0:
                ax.plot(demod_info["pilot_pwr"])
                ax.set_title("Pilot power per OFDM symbol")
                ax.grid(True)
            else:
                ax.axis("off")

            # 8) pilot loop traces
            ax = fig.add_subplot(3,4,8)
            if demod_info is not None and demod_info.get("phase_err") is not None and len(demod_info["phase_err"]) > 0:
                ax.plot(demod_info["phase_err"], label="phase_err (unwrap)")
                ax.plot(demod_info["phase_acc"], label="phase_acc")
                ax.plot(demod_info["freq_acc"], label="freq_acc")
                ax.set_title("Pilot loop traces")
                ax.grid(True)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
            else:
                ax.axis("off")

            # 9) constellation pre/post CPE
            ax = fig.add_subplot(3,4,9)
            if demod_info is not None and demod_info.get("data_pre") is not None and len(demod_info["data_pre"]) > 0:
                pre = demod_info["data_pre"]
                post = demod_info["data_post"]
                ax.scatter(np.real(pre), np.imag(pre), s=4, alpha=0.35, label="pre-CPE")
                ax.scatter(np.real(post), np.imag(post), s=4, alpha=0.35, label="post-CPE")
                ax.set_title("Constellation (data bins)")
                ax.grid(True)
                ax.axis("equal")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
            else:
                ax.axis("off")

            # 10) EVM per symbol
            ax = fig.add_subplot(3,4,10)
            if demod_info is not None and demod_info.get("evm_db") is not None and len(demod_info["evm_db"]) > 0:
                ax.plot(demod_info["evm_db"])
                ax.set_title("EVM per OFDM symbol (dB)")
                ax.grid(True)
            else:
                ax.axis("off")

            # 11) angle histogram
            ax = fig.add_subplot(3,4,11)
            if demod_info is not None and demod_info.get("data_post") is not None and len(demod_info["data_post"]) > 0:
                ang = np.angle(demod_info["data_post"])
                ax.hist(ang, bins=60)
                ax.set_title("Angle hist (post-CPE)")
                ax.grid(True)
            else:
                ax.axis("off")

            # 12) text box
            ax = fig.add_subplot(3,4,12)
            ax.axis("off")
            txt = []
            txt.append(title)
            txt.append(f"CFO_use={chosen.get('cfo_use',0):+.1f} Hz")
            txt.append(f"STF_idx={stf_idx}  STF_ncc={stf_ncc:.3f}  LTF_score={ltf_info.get('score',0) if ltf_info else 0:.3f}")
            if parse_info is None:
                txt.append("PARSE: FAIL")
            else:
                txt.append(f"PARSE: start={parse_info.get('start', -1)} ok={parse_info.get('ok', False)}")
                txt.append(f" seq={parse_info.get('seq', -1)} total={parse_info.get('total', -1)} plen={parse_info.get('plen', -1)}")
            if demod_info is not None and demod_info.get("nsyms", 0) > 0:
                txt.append(f"Demod: ofdm_syms={demod_info['nsyms']} bytes={demod_info.get('bytes_len',0)}")
            ax.text(0.02, 0.98, "\n".join(txt), va="top", family="monospace")

            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
        except Exception as e:
            import traceback
            print(f"[FIG] Error saving figure {path}: {e}")
            traceback.print_exc()

    print("[FIG] Figure worker stopped.")

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
def moving_energy_mean(x: np.ndarray, win: int) -> np.ndarray:
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

@njit(cache=True)
def longest_run_over_thr(e: np.ndarray, thr: float):
    best = 0
    best_idx = 0
    cur = 0
    start = 0
    for i in range(e.shape[0]):
        if e[i] > thr:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best = cur
                best_idx = start
        else:
            cur = 0
    return best, best_idx

@njit(cache=True)
def one_pole_highpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    y[n] = alpha*(y[n-1] + x[n] - x[n-1])
    alpha ~ exp(-2*pi*fc/fs)
    """
    y = np.empty_like(x)
    y0r = 0.0
    y0i = 0.0
    xpr = x[0].real
    xpi = x[0].imag
    y[0] = 0j
    for n in range(1, x.shape[0]):
        xr = x[n].real
        xi = x[n].imag
        y0r = alpha * (y0r + xr - xpr)
        y0i = alpha * (y0i + xi - xpi)
        y[n] = y0r + 1j*y0i
        xpr = xr
        xpi = xi
    return y


# =========================
# PHY helpers
# =========================
def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits, bitorder='big').tobytes()

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
    # IMPORTANT: match working v1 convention
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x.astype(np.complex64), num_repeats)
    stf_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_cp

def create_ltf_ref(num_symbols: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    # IMPORTANT: match working v1 convention
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

def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, int, bytes]:
    if len(bb) < 10:
        return False, "too_short", -1, -1, b""
    midx = bb.find(MAGIC)
    if midx < 0:
        return False, "bad_magic", -1, -1, b""
    b = bb[midx:]
    
    # Try V1 format: MAGIC(4) + SEQ(4) + PLEN(2) + DATA(PLEN) + CRC(4)
    if len(b) >= 14:
        seq = int.from_bytes(b[4:8], "little")
        plen = int.from_bytes(b[8:10], "little")
        if len(b) >= 10 + plen + 4:
            body = b[:10+plen]
            crc_rx = int.from_bytes(b[10+plen:10+plen+4], "little")
            if (zlib.crc32(body) & 0xFFFFFFFF) == crc_rx:
                return True, "ok_v1", seq, -1, b[10:10+plen]
                
    # Try V2 format: MAGIC(4) + SEQ(2) + TOTAL(2) + PLEN(2) + DATA(PLEN) + CRC(4)
    if len(b) >= 14:
        seq = int.from_bytes(b[4:6], "little")
        total = int.from_bytes(b[6:8], "little")
        plen = int.from_bytes(b[8:10], "little")
        if len(b) >= 10 + plen + 4:
            body = b[:10+plen]
            crc_rx = int.from_bytes(b[10+plen:10+plen+4], "little")
            if (zlib.crc32(body) & 0xFFFFFFFF) == crc_rx:
                return True, "ok_v2", seq, total, b[10:10+plen]

    return False, "crc_fail", -1, -1, b""


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
    uri: str = ""
    fc: float = 2.3e9
    fs: float = 3e6
    rx_gain: float = 40.0
    rx_bw: float = 3.6e6
    rx_buf: int = 131072
    kernel_buffers: int = 4
    repeat: int = 1
    stf_repeats: int = 6
    ltf_symbols: int = 4

    ring_size: int = 524288
    proc_window: int = 262144
    proc_hop: int = 65536

    # robust gating
    energy_win: int = 512
    energy_k: float = 3.0
    min_run: int = 128
    hard_peak: float = 0.0
    hard_maxe: float = 0.0

    # xcorr/NCC
    xcorr_search: int = 200000
    xcorr_topk: int = 8
    xcorr_min_peak: float = 0.2
    use_ncc: bool = True

    ltf_off_sweep: int = 16
    max_ofdm_syms_probe: int = 16
    max_ofdm_syms_cap: int = 260
    kp: float = 0.05
    ki: float = 0.0005

    # LO leakage suppression
    hp_fc: float = 3000.0
    sweep_ignore_dc_hz: float = 5000.0

    save_dir: str = "."
    save_npz: bool = False
    mode: str = "packet"
    verbose: bool = True
    ref_seed: int = 1234
    ref_len: int = 4096


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

    for _ in range(4):
        _ = sdr.rx()

    alpha = None
    if cfg.hp_fc > 0:
        alpha = float(np.exp(-2*np.pi*cfg.hp_fc/cfg.fs))

    print("[RX] acq worker started. rx_buf =", cfg.rx_buf, "hp_fc =", cfg.hp_fc)
    try:
        while not stop_ev.is_set():
            x = sdr.rx().astype(np.complex64)

            # autoscale
            if np.median(np.abs(x)) > 100:
                x = x / (2**14)

            # remove DC per block
            x = x - np.mean(x)

            # highpass to suppress LO leakage / drift (optional)
            if alpha is not None:
                if NUMBA_OK:
                    x = one_pole_highpass(x, alpha)
                else:
                    # python fallback (slower but fine)
                    y = np.empty_like(x)
                    y[0] = 0j
                    y0 = 0j
                    xp = x[0]
                    for n in range(1, x.size):
                        y0 = alpha * (y0 + x[n] - xp)
                        y[n] = y0
                        xp = x[n]
                    x = y

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
    cap = 0
    samples_since = 0

    print("[RX] Sweep thread started. ignore_dc_hz =", cfg.sweep_ignore_dc_hz)
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
            Pxx_dB = 10*np.log10(Pxx + 1e-12)

            # DC peak
            dc_i = int(np.argmin(np.abs(f)))
            dc_str = f"DC~{f[dc_i]/1e3:+.1f} kHz ({Pxx_dB[dc_i]:.1f} dB)"

            # non-DC peaks
            mask = (np.abs(f) >= float(cfg.sweep_ignore_dc_hz))
            ff = f[mask]
            pp = Pxx_dB[mask]
            if pp.size == 0:
                print(f"[RX Sweep] cap={cap} {dc_str} | no non-DC bins")
                continue

            top = np.argsort(pp)[-5:][::-1]
            peaks_str = ", ".join([f"{ff[i]/1e6:+.3f} MHz ({pp[i]:.1f} dB)" for i in top[:3]])
            print(f"[RX Sweep] cap={cap} {dc_str} | Top(nonDC): {peaks_str}")

    except KeyboardInterrupt:
        pass
    finally:
        print("[RX] Sweep thread stopped. cap=", cap)


def dsp_thread(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", fig_q: queue.Queue, cfg: RxConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    good_dir = os.path.join(cfg.save_dir, "good_packets")
    os.makedirs(good_dir, exist_ok=True)

    stf_ref = create_schmidl_cox_stf(cfg.stf_repeats).astype(np.complex64)
    stf_ref_e = float(np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
    stf_len = int(stf_ref.size)
    _, ltf_freq_ref = create_ltf_ref(cfg.ltf_symbols)

    ring = RingBuffer(cfg.ring_size)
    samples_since = 0

    csv_path = os.path.join(cfg.save_dir, "captures.csv")
    fcsv = open(csv_path, "w", newline="")
    fieldnames = [
        "cap","status","reason",
        "peak",
        "med","mad","sigma","thr","runlen",
        "maxe",
        "xc_best_peak","xc_best_idx",
        "stf_idx","ltf_start","payload_start",
        "probe_evm","seq","payload_len"
    ]
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    cap = 0
    good = 0
    got_packets = {}
    total_expected = None

    def try_demod_at(rxw: np.ndarray, stf_idx: int) -> Tuple[bool, str, int, int, bytes, dict, dict, dict]:
        """
        Demod starting at stf_idx (index within rxw).
        """
        ltf0 = stf_idx + stf_len
        
        # 1. Estimate CFO from STF (period=16 samples)
        cfo_hz = 0.0
        stf_start = stf_idx + 16
        if stf_start + 64 <= rxw.shape[0]:
            p0 = rxw[stf_start : stf_start + 48]
            p1 = rxw[stf_start + 16 : stf_start + 64]
            angle = float(np.angle(np.sum(p1 * np.conj(p0))))
            cfo_hz = angle / (2 * np.pi * (16.0 / cfg.fs))
            if cfg.verbose:
                print(f"[STF] stf_idx={stf_idx} angle={angle:.3f} cfo={cfo_hz/1e3:.1f}kHz")
            
        def extract_sym(start, do_swap=False):
            if start + SYMBOL_LEN > rxw.shape[0]:
                return None
            td = rxw[start + N_CP : start + SYMBOL_LEN]
            Y = np.fft.fftshift(np.fft.fft(td))
            if do_swap:
                # In fftshifted domain, bin 32 is DC. 
                # Subcarriers are -26 to 26.
                # Swap k -> -k
                Y = np.conj(Y[::-1])
                # Small shift if needed to align DC exactly? 
                # fftshift([0 1 2 3 4]) -> [3 4 0 1 2]. 
                # DC was at index 0, now at index 32.
                # np.conj(Y[::-1]) correctly flips -k to k.
            return Y

        best_xc = -1.0
        best_off = 0
        used_bins = np.array([sc_to_bin(k) for k in range(-26,27) if k!=0], dtype=int)
        for off in range(-cfg.ltf_off_sweep, cfg.ltf_off_sweep+1):
            Y = extract_sym(ltf0 + off)
            if Y is None: continue
            Ht = np.zeros(N_FFT, dtype=np.complex64)
            for k in range(-26,27):
                if k==0: continue
                idx = sc_to_bin(k)
                if np.abs(ltf_freq_ref[idx]) > 1e-6:
                    Ht[idx] = Y[idx] / ltf_freq_ref[idx]
            m = np.abs(Ht[used_bins])
            qv = float((np.mean(m)**2) / (np.var(m) + 1e-12))
            if qv > best_xc:
                best_xc = qv
                best_off = off
        
        ltf_start = ltf0 + best_off
        
        # LTF ICFO and H estimate
        Ys = []
        for i in range(cfg.ltf_symbols):
            Y = extract_sym(ltf_start + i*SYMBOL_LEN)
            if Y is None: break
            Ys.append(Y)
        H = None
        if Ys:
            Yavg = np.mean(np.stack(Ys, axis=0), axis=0)
            
            # Disable Super Search to match v1
            best_dk = 0
            best_swap = False
            icfo_hz = 0.0
            # cfo_hz already contains stf_cfo_hz
            
            # Fine CFO removed to match v1
            
            Ys_new = []
            for i in range(cfg.ltf_symbols):
                Y = extract_sym(ltf_start + i*SYMBOL_LEN, do_swap=best_swap)
                if Y is None: break
                Ys_new.append(Y)
                
            if Ys_new:
                Yavg = np.mean(np.stack(Ys_new, axis=0), axis=0)
                H = np.ones(N_FFT, dtype=np.complex64)
                # Estimate channel and timing slope
                phases = []
                ks = []
                for k in range(-26,27):
                    if k == 0: continue
                    idx = sc_to_bin(k)
                    if np.abs(ltf_freq_ref[idx]) > 1e-6:
                        val = Yavg[idx] / ltf_freq_ref[idx]
                        H[idx] = val
                        phases.append(np.angle(val))
                        ks.append(k)
                
                    # Timing correction (v2 experimental) disabled to match v1
                    pass

        ltf_info = {"score": best_xc, "H": H, "Y0": None, "Y1": None}
        if H is None:
            return False, "ltf_fail", -1, -1, b"", {"ltf_q": best_xc, "ltf_start": ltf_start}, None, ltf_info

        payload_start = ltf_start + cfg.ltf_symbols*SYMBOL_LEN

        # probe demod
        pilot_vals = np.array([1,1,1,-1], dtype=np.complex64)
        phase_acc = 0.0
        freq_acc = 0.0
        prev_phase = None

        data_syms_all = []
        evm_log = []
        pilot_pwr_log = []
        phase_err_log = []
        phase_acc_log = []
        freq_acc_log = []
        all_data_pre = []
        all_data_post = []

        for si in range(cfg.max_ofdm_syms_cap):
            Y = extract_sym(payload_start + si*SYMBOL_LEN, do_swap=best_swap)
            if Y is None:
                break
            Ye = Y.copy()
            for k in range(-26,27):
                if k == 0:
                    continue
                idx = sc_to_bin(k)
                if np.abs(H[idx]) > 1e-6:
                    Ye[idx] = Ye[idx] / (H[idx] + 1e-12)

            sign = 1 if (si % 2 == 0) else -1
            rp = Ye[PILOT_BINS]
            ph = float(np.angle(np.sum(rp * np.conj(sign*pilot_vals))))

            if prev_phase is not None:
                dph = ph - prev_phase
                while dph > np.pi:
                    dph -= 2*np.pi
                while dph < -np.pi:
                    dph += 2*np.pi
                freq_acc += cfg.ki * dph
            prev_phase = ph

            if si == 0:
                # Jump start phase tracking!
                phase_acc = ph
            else:
                phase_acc += freq_acc + cfg.kp * ph
            
            phase_err_log.append(ph)
            phase_acc_log.append(phase_acc)
            freq_acc_log.append(freq_acc)
            pilot_pwr_log.append(float(np.mean(np.abs(rp)**2)))
            all_data_pre.append(Ye[DATA_BINS].copy())
            
            Ye *= np.exp(-1j*phase_acc).astype(np.complex64)

            ds = Ye[DATA_BINS]
            all_data_post.append(ds.copy())

            # EVM
            nearest = np.empty_like(ds)
            for i in range(ds.size):
                bestj = 0
                md = 1e9
                for j in range(4):
                    d = ds[i] - IDEAL_QPSK[j]
                    dd = (d.real*d.real + d.imag*d.imag)
                    if dd < md:
                        md = dd
                        bestj = j
                nearest[i] = IDEAL_QPSK[bestj]

            evm = float(np.sqrt(np.mean(np.abs(ds - nearest)**2)))
            evm_log.append(evm)
            data_syms_all.append(ds)

        demod_info = {
            "nsyms": len(data_syms_all),
            "pilot_pwr": np.array(pilot_pwr_log, dtype=np.float32),
            "phase_err": np.array(phase_err_log, dtype=np.float32),
            "phase_acc": np.array(phase_acc_log, dtype=np.float32),
            "freq_acc": np.array(freq_acc_log, dtype=np.float32),
            "evm_db": np.array(evm_log, dtype=np.float32),
            "data_pre": np.concatenate(all_data_pre) if len(all_data_pre) else np.array([]),
            "data_post": np.concatenate(all_data_post) if len(all_data_post) else np.array([]),
        }
        if not data_syms_all:
            return False, "no_payload", -1, -1, b"", {"ltf_q": best_xc, "ltf_start": ltf_start, "payload_start": payload_start}, demod_info, ltf_info

        data_syms_all_concat = np.concatenate(data_syms_all).astype(np.complex64)
        
        # Super Search: 4 rotations and Swap
        best_ok = False
        best_bb = b""
        best_reason = "bad_magic"
        best_seq = -1
        best_total = -1
        best_payload = b""
        
        for conj in [False, True]:
            syms_base = np.conj(data_syms_all_concat) if conj else data_syms_all_concat
            for rot in [0, 1, 2, 3]:
                syms_rot = syms_base * (1j**rot)
                bits_raw = qpsk_demap(syms_rot)
                bits = majority_vote(bits_raw, cfg.repeat)
                bits = scramble_bits(bits)   # descramble (TX scrambles before modulation)
                bb = bits_to_bytes(bits)

                res_ok, res_reason, res_seq, res_total, res_payload = parse_packet_bytes(bb)
                if res_ok:
                    best_ok = True
                    best_reason = res_reason
                    best_seq = res_seq
                    best_total = res_total
                    best_payload = res_payload
                    best_bb = bb
                    if cfg.verbose:
                        print(f"[SEARCH] SUCCESS! conj={conj} rot={rot} seq={res_seq}")
                    break
                else:
                    # Keep track of the "least bad" attempt for logging
                    if best_reason == "bad_magic" or best_reason == "no_try":
                        best_reason = res_reason
                        best_bb = bb
            if best_ok: break
            
        diag = {
            "ltf_q": best_xc,
            "ltf_start": ltf_start,
            "payload_start": payload_start,
            "probe_evm": float(np.mean(evm_log)) if evm_log else 0.0,
        }
        
        if not best_ok:
            if cfg.verbose:
                ph0 = phases[0] if (phases is not None and len(phases) > 0) else 0.0
                print(f"[DEMOD] FAIL {best_reason}: dk=0 off={best_off} ph0={ph0:.3f} evm={diag['probe_evm']:.3f}")
                print(f"        hex: {bb[:16].hex()}")
        
        return best_ok, best_reason, best_seq, best_total, best_payload, diag, demod_info, ltf_info

    print("[RX] DSP thread started. NUMBA_OK =", NUMBA_OK, "use_ncc =", cfg.use_ncc)

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

            # energy (sum power like v1)
            if NUMBA_OK:
                e = moving_energy(rxw.astype(np.complex64), cfg.energy_win)
            else:
                win = cfg.energy_win
                p = (np.abs(rxw)**2).astype(np.float32)
                ker = np.ones(win, dtype=np.float32)
                e = np.convolve(p, ker, mode="valid")

            if e.size == 0:
                continue

            maxe = float(np.max(e))
            p10 = float(np.percentile(e, 10))
            eg_th = float(p10 * cfg.energy_mult) if hasattr(cfg, 'energy_mult') else float(p10 * cfg.energy_k)
            
            if NUMBA_OK:
                runlen, run_start = longest_run_over_thr(e.astype(np.float32), eg_th)
            else:
                runlen = np.max(np.convolve((e > eg_th).astype(np.int32), np.ones(64, dtype=np.int32), 'same'))
                run_start = 0 # suboptimal fallback

            # robust gate decision (+ hard override)
            hard_override = False
            if cfg.hard_peak > 0 and peak >= cfg.hard_peak:
                hard_override = True
            if cfg.hard_maxe > 0 and maxe >= cfg.hard_maxe:
                hard_override = True

            # Relaxed gate: V1 style (skip only if no signal above noise floor)
            if (not hard_override) and (maxe < p10 * 1.1):
                writer.writerow({
                    "cap": cap, "status": "skip", "reason": "energy_low_v1",
                    "peak": peak,
                    "med": p10, "mad": 0, "sigma": 0, "thr": eg_th, "runlen": runlen,
                    "maxe": maxe,
                    "xc_best_peak": 0.0, "xc_best_idx": -1,
                    "stf_idx": -1, "ltf_start": -1, "payload_start": -1,
                    "probe_evm": "", "seq": "", "payload_len": ""
                })
                if cfg.verbose and (cap <= 5 or cap % 200 == 0):
                    print(f"[RX] cap={cap} SKIP energy_low peak={peak:.3f} maxe={maxe:.4g} p10={p10:.4g}")
                continue

            # 2. Tone Mode (Spectrum)
            if cfg.mode == "tone":
                win_len = rxw.size
                if win_len >= 4096:
                    seg = rxw[-4096:]
                    psd = np.abs(np.fft.fftshift(np.fft.fft(seg)))**2
                    freqs = np.fft.fftshift(np.fft.fftfreq(seg.size, 1/cfg.fs))
                    pk_idx = np.argmax(psd)
                    pk_freq = freqs[pk_idx]
                    if cap % 5 == 0:
                        print(f"[TONE] cap={cap} peak_freq={pk_freq/1e3:.1f} kHz")
                        plt.figure(figsize=(8, 4))
                        plt.semilogy(freqs/1e3, psd)
                        plt.grid(True)
                        plt.title(f"Tone Spectrum (Peak: {pk_freq/1e3:.1f} kHz)")
                        plt.xlabel("Freq (kHz)")
                        plt.ylabel("PSD")
                        plt.tight_layout()
                        plt.savefig(f"{cfg.save_dir}/tone_cap_{cap:04d}.png")
                        plt.close()
                cap += 1
                continue

            # xcorr on search region
            search_len = min(cfg.xcorr_search, rxw.size)
            xs = rxw[:search_len].astype(np.complex64)

            if NUMBA_OK:
                corr = xcorr_mag_valid(xs, stf_ref)
            else:
                L = stf_len
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

            # normalize correlation
            if cfg.use_ncc:
                # seg power mean over L; convert to ||seg|| by sqrt(mean*L)
                if NUMBA_OK:
                    seg_pow = moving_energy(xs, stf_len) / float(stf_len)
                else:
                    p = (np.abs(xs)**2).astype(np.float32)
                    ker = np.ones(stf_len, dtype=np.float32)/float(stf_len)
                    seg_pow = np.convolve(p, ker, mode="valid")
                seg_norm = np.sqrt(seg_pow * float(stf_len))  # ||seg||
                corr_norm = corr / (seg_norm * stf_ref_e + 1e-12)
            else:
                corr_norm = corr / stf_ref_e

            # top-k peaks
            k = min(cfg.xcorr_topk, corr_norm.size)
            top_idx = np.argpartition(corr_norm, -k)[-k:]
            top_idx = top_idx[np.argsort(corr_norm[top_idx])[::-1]]

            best_ok = False
            best_reason = "no_try"
            best_seq = -1
            best_total = -1
            best_payload = b""
            best_diag = {}
            best_stf = -1
            best_xc_peak = float(corr_norm[top_idx[0]])
            best_xc_idx = int(top_idx[0])
            if cfg.verbose:
                print(f"[RX] best_xc={best_xc_peak:.3f} at {best_xc_idx}")
            best_demod = None
            best_ltf = None

            for cand in top_idx:
                if corr_norm[cand] < cfg.xcorr_min_peak:
                    continue
                ok, reason, seq, total, payload, diag, d_info, l_info = try_demod_at(rxw, int(cand))
                if ok:
                    best_ok = True
                    best_reason = "ok"
                    best_seq = int(seq)
                    best_total = int(total)
                    best_payload = payload
                    best_diag = diag
                    best_stf = int(cand)
                    best_demod = d_info
                    best_ltf = l_info
                    break
                else:
                    if best_stf < 0:
                        best_reason = reason
                        best_diag = diag
                        best_stf = int(cand)
                        best_demod = d_info
                        best_ltf = l_info

            if best_stf < 0 and peak > cfg.energy_k * mad + med:
                # Force try around the energy start
                print(f"[RX] FORCING sweep around {run_start} (peak={peak:.1f})")
                for offset in range(-500, 2000, 100):
                    cand = max(0, run_start + offset)
                    ok, reason, seq, total, payload, diag, d_info, l_info = try_demod_at(rxw, int(cand))
                    if ok or (d_info and d_info.get("evm", 10.0) < 1.5):
                        best_stf = int(cand)
                        best_ok = ok
                        best_reason = reason + "_forced"
                        best_diag = diag
                        best_demod = d_info
                        best_ltf = l_info
                        if ok:
                            best_seq = int(seq)
                            best_total = int(total)
                            best_payload = payload
                            print(f"  [FORCE] FOUND! cand={cand} evm={d_info.get('evm',0):.2f}")
                            break
                        else:
                            print(f"  [FORCE] Potential at {cand} evm={d_info.get('evm',0):.2f}")

            status = "ok" if best_ok else "no_crc"
            payload_len = len(best_payload) if best_ok else 0

            writer.writerow({
                "cap": cap,
                "status": status,
                "reason": best_reason,
                "peak": peak,
                "med": med, "mad": mad, "sigma": sigma, "thr": thr, "runlen": runlen,
                "maxe": maxe,
                "xc_best_peak": best_xc_peak,
                "xc_best_idx": best_xc_idx,
                "stf_idx": best_stf,
                "ltf_start": int(best_diag.get("ltf_start", -1)),
                "payload_start": int(best_diag.get("payload_start", -1)),
                "probe_evm": float(best_diag.get("probe_evm", 0.0)),
                "seq": (best_seq if best_ok else ""),
                "payload_len": (payload_len if best_ok else ""),
            })

            # Save raw debug NPZ
            if cap < 5:
                npz_path = os.path.join(cfg.save_dir, f"cap_{cap:06d}_raw.npz")
                np.savez(npz_path, rxw=rxw, best_stf=best_stf, best_reason=best_reason)
            cap += 1

            
            # Enqueue figure saving
            if best_stf >= 0:
                chosen = {"stf_idx": best_stf, "stf_ncc": best_xc_peak, "cfo_use": 0.0}
                parse_info = {"ok": best_ok, "seq": best_seq, "total": best_total, "plen": len(best_payload)}
                title = f"cap={cap} {'OK' if best_ok else 'FAIL'} {best_reason}"
                fig_path = os.path.join(cfg.save_dir, f"cap_{cap:06d}_{'ok' if best_ok else 'fail'}.png")
                fig_q.put((fig_path, rxw.copy(), cfg.fs, None, corr_norm, np.arange(len(corr_norm)), [(int(x), float(corr_norm[x])) for x in top_idx], chosen, best_ltf, best_demod, parse_info, title))

            if best_ok:
                good += 1
                got_packets[best_seq] = best_payload
                if total_expected is None and best_total > 0:
                    total_expected = best_total
                    
                outp = os.path.join(good_dir, f"seq_{best_seq:08d}.bin")
                with open(outp, "wb") as f:
                    f.write(best_payload)

                # terminal print payload
                try:
                    s = best_payload.decode("utf-8", errors="replace")
                    print(f"[RX] OK seq={best_seq} payload={payload_len}B evm={best_diag.get('probe_evm',0):.3f} stf={best_stf} xc={best_xc_peak:.3f} | '{s}'")
                except Exception:
                    print(f"[RX] OK seq={best_seq} payload={payload_len}B evm={best_diag.get('probe_evm',0):.3f} stf={best_stf} xc={best_xc_peak:.3f} | hex={best_payload[:64].hex()}")
                if cfg.save_npz:
                    npz_path = os.path.join(cfg.save_dir, f"cap_{cap:06d}_ok.npz")
                    meta = {
                        "cap": cap,
                        "seq": best_seq,
                        "peak": peak,
                        "best_xc_peak": best_xc_peak,
                        "best_xc_idx": best_xc_idx,
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
            else:
                if (not cfg.verbose) and (cap % 50 == 0):
                    print(f"[RX] cap={cap} good={good} (last status={best_reason}, xc={best_xc_peak:.3f}, peak={peak:.1f})")

            if cap % 50 == 0:
                fcsv.flush()
                if cfg.verbose:
                    print(f"[RX] cap={cap} good={good} (last status={status}, xc={best_xc_peak:.3f}, peak={peak:.3f})")

    except KeyboardInterrupt:
        pass
    finally:
        fcsv.flush()
        fcsv.close()
        print("[RX] DSP thread stopped. cap=", cap, "good=", good)
        if total_expected is not None and len(got_packets) >= total_expected:
            full = b"".join(got_packets[i] for i in range(total_expected) if i in got_packets)
            outf = os.path.join(cfg.save_dir, "recovered_payload.bin")
            with open(outf, "wb") as f:
                f.write(full)
            print(f"\n[RX] Reassembled payload: {len(full)} bytes -> {outf}")
            if cfg.ref_len > 0:
                ref_payload = make_reference_payload(cfg.ref_seed, cfg.ref_len)
                err, tot, ber = ber_bits(full, ref_payload)
                print(f"[BER] compare_len={min(len(full),len(ref_payload))} bytes bit_err={err} bit_tot={tot} BER={ber:.3e}")
        else:
            print("\n[RX] Incomplete payload reassembly.")
            if total_expected:
                print(f"  got {len(got_packets)}/{total_expected} packets")


def main():
    ap = argparse.ArgumentParser("Streaming Step5 PHY RX v2 (ring buffer + threads + numba + MAD gate + NCC)")
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, required=True)
    ap.add_argument("--fs", type=float, default=5e6)
    ap.add_argument("--rx_gain", type=float, default=30.0)
    ap.add_argument("--rx_bw", type=float, default=0.0)
    ap.add_argument("--rx_buf", type=int, default=131072)
    ap.add_argument("--kernel_buffers", type=int, default=4)

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--stf_repeats", type=int, default=20)
    ap.add_argument("--ltf_symbols", type=int, default=10)

    ap.add_argument("--ring_size", type=int, default=524288)
    ap.add_argument("--proc_window", type=int, default=262144)
    ap.add_argument("--proc_hop", type=int, default=65536)

    # robust gating params
    ap.add_argument("--energy_win", type=int, default=512)
    ap.add_argument("--energy_k", type=float, default=8.0, help="thr = median + k*sigma(MAD). Typical 6~12")
    ap.add_argument("--min_run", type=int, default=128, help="require runlen>=min_run for energy gate")
    ap.add_argument("--hard_peak", type=float, default=0.0, help="if peak>=hard_peak, bypass energy gate (optional)")
    ap.add_argument("--hard_maxe", type=float, default=0.0, help="if maxe>=hard_maxe, bypass energy gate (optional)")

    # xcorr
    ap.add_argument("--xcorr_search", type=int, default=200000)
    ap.add_argument("--xcorr_topk", type=int, default=8)
    ap.add_argument("--xcorr_min_peak", type=float, default=0.2)
    ap.add_argument("--no_ncc", action="store_true", help="disable normalized cross correlation")

    ap.add_argument("--ltf_off_sweep", type=int, default=16)
    ap.add_argument("--probe_syms", type=int, default=16)
    ap.add_argument("--max_syms_cap", type=int, default=260)
    ap.add_argument("--kp", type=float, default=0.05)
    ap.add_argument("--ki", type=float, default=0.0005)

    # LO leakage suppress
    ap.add_argument("--hp_fc", type=float, default=0, help="one-pole highpass cutoff Hz (0 disables).")
    ap.add_argument("--sweep_ignore_dc_hz", type=float, default=5000.0, help="sweep: ignore |f| < this when printing peaks")

    ap.add_argument("--out_root", default="rf_stream_rx_runs")
    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--mode", type=str, default="packet", choices=["packet","sweep","tone"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ref_seed", type=int, default=0)
    ap.add_argument("--ref_len", type=int, default=0)

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
        energy_k=args.energy_k,
        min_run=args.min_run,
        hard_peak=args.hard_peak,
        hard_maxe=args.hard_maxe,

        xcorr_search=args.xcorr_search,
        xcorr_topk=args.xcorr_topk,
        xcorr_min_peak=args.xcorr_min_peak,
        use_ncc=(not args.no_ncc),

        ltf_off_sweep=args.ltf_off_sweep,
        max_ofdm_syms_probe=args.probe_syms,
        max_ofdm_syms_cap=args.max_syms_cap,
        kp=args.kp,
        ki=args.ki,

        hp_fc=args.hp_fc,
        sweep_ignore_dc_hz=args.sweep_ignore_dc_hz,

        save_dir=out_dir,
        save_npz=bool(args.save_npz),
        mode=args.mode,
        verbose=bool(args.verbose),
        ref_seed=args.ref_seed,
        ref_len=args.ref_len,
    )

    print("\n" + "="*78)
    print("Streaming RX (Step5 PHY) - v2")
    print("="*78)
    print("out_dir:", out_dir)
    print("NUMBA_OK:", NUMBA_OK)
    print("cfg:", cfg)
    print("="*78)

    os.makedirs(out_dir, exist_ok=True)

    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
    fig_q = queue.Queue()
    stop_ev = threading.Event()

    t_acq = threading.Thread(target=rx_acq_worker, args=(stop_ev, q, cfg), daemon=True)
    t_fig = threading.Thread(target=figure_worker_thread, args=(stop_ev, fig_q), daemon=True)

    if cfg.mode == "sweep":
        t_dsp = threading.Thread(target=dsp_sweep_thread, args=(stop_ev, q, cfg), daemon=True)
    else:
        t_dsp = threading.Thread(target=dsp_thread, args=(stop_ev, q, fig_q, cfg), daemon=True)

    t_acq.start()
    t_fig.start()
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
python3 rf_stream_rx_step5phy_v2.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --rx_gain 30 --rx_buf 131072 \
  --ring_size 524288 --proc_window 262144 --proc_hop 65536 \
  --energy_k 8 --min_run 128 \
  --xcorr_topk 8 --xcorr_min_peak 0.2 \
  --probe_syms 16 --save_npz --verbose

python3 -u rf_stream_rx_step5phy_v2.py --uri ip:192.168.2.2 --fc 2.3e9 --fs 5e6 --rx_gain 50 --mode packet --verbose --energy_k 3.0 --xcorr_min_peak 0.3

python3 rf_stream_rx_step5phy_v2.py --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 --rx_gain 30 --rx_buf 131072 --stf_repeats 6 --ltf_symbols 4 --kp 0.15 --ki 0.005 --no_ncc --xcorr_min_peak 0.1

"""