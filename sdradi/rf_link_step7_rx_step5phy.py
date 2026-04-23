#!/usr/bin/env python3
"""
Step5-based Step7 RX (VID7 video packets) - PlutoSDR

Why this version:
- Keeps EXACT Step5 FFT convention and preamble mapping to match the known-working PHY.
- Uses Schmidl-Cox for timing + CFO (CFO from SC by default; tone CFO optional).
- Adds probe + bit-slip + bounded demod (prevents tail contamination).
- Adds logging + CSV + plots + frame reassembly.

Packet format (must match TX):
  MAGIC("VID7", 4B) | FRAME_ID(2B) | SEQ(2B) | TOTAL(2B) | LEN(2B) | PAYLOAD | CRC32(4B)
  CRC covers MAGIC..PAYLOAD (header+payload).

Outputs:
- run_dir with:
  - packets.csv (per-capture log)
  - summary.png
  - per-frame diagnostics (optional)
  - received_video.avi (optional) + optional saved frames
"""

import argparse
import os
import time
import csv
import zlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAGIC = b"VID7"

# =========================
# OFDM (Step5 convention)
# =========================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)           # 48
BITS_PER_OFDM_SYM = N_DATA * 2           # 96

def bits_to_bytes(bits: np.ndarray) -> bytes:
    L = (len(bits)//8)*8
    if L <= 0:
        return b""
    return np.packbits(bits[:L]).tobytes()

def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits
    L = (len(bits)//repeat)*repeat
    if L <= 0:
        return np.zeros(0, dtype=np.uint8)
    x = bits[:L].reshape(-1, repeat)
    return (np.sum(x, axis=1) >= (repeat/2)).astype(np.uint8)

def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols)*2, dtype=np.uint8)
    for i, s in enumerate(symbols):
        re = np.real(s) >= 0
        im = np.imag(s) >= 0
        if re and im:
            bits[2*i], bits[2*i+1] = 0, 0
        elif (not re) and im:
            bits[2*i], bits[2*i+1] = 0, 1
        elif (not re) and (not im):
            bits[2*i], bits[2*i+1] = 1, 1
        else:
            bits[2*i], bits[2*i+1] = 1, 0
    return bits

def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(len(samples), dtype=np.float32)
    return samples * np.exp(-1j * 2*np.pi*(cfo_hz/fs)*n).astype(np.complex64)

# =========================
# Step5 Preamble refs
# =========================
def create_schmidl_cox_stf(num_repeats: int = 6):
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[(k + N_FFT) % N_FFT] = bpsk[i] + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x, num_repeats).astype(np.complex64)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp, X

def create_ltf_ref(num_symbols: int = 4):
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N_FFT) % N_FFT] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X

# =========================
# Schmidl-Cox metric (vectorized)
# =========================
def schmidl_cox_metric(rx: np.ndarray, half_period: int = 32, window_len: int = 60000):
    L = half_period
    N = len(rx)
    window_len = min(window_len, N - 2*L)
    if window_len <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.complex64), np.array([], dtype=np.float32)

    c = rx[:window_len+L] * np.conj(rx[L:window_len+2*L])
    cs = np.zeros(window_len+L+1, dtype=np.complex64)
    cs[1:] = np.cumsum(c)
    P = cs[L:L+window_len] - cs[:window_len]

    pow2 = np.abs(rx[:window_len+2*L])**2
    cr = np.zeros(window_len+2*L+1, dtype=np.float32)
    cr[1:] = np.cumsum(pow2)
    R = cr[2*L:2*L+window_len] - cr[L:L+window_len]

    M = (np.abs(P)**2) / (R**2 + 1e-12)
    return M, P, R

def detect_stf_sc(rx: np.ndarray, fs: float, half_period: int = 32,
                  sc_threshold: float = 0.2, search_len: int = 60000,
                  peak_med_ratio_th: float = 6.0):
    M, P, _ = schmidl_cox_metric(rx, half_period=half_period, window_len=search_len)
    if M.size == 0:
        return False, 0, 0.0, 0.0, 0.0, M

    plateau = 2*half_period
    if len(M) <= plateau+1:
        pk = float(np.max(M))
        ratio = float(pk / (np.median(M)+1e-12))
        return False, int(np.argmax(M)), pk, 0.0, ratio, M

    kernel = np.ones(plateau, dtype=np.float32) / plateau
    Ms = np.convolve(M, kernel, mode="valid")
    pk_i = int(np.argmax(Ms))
    pk_v = float(Ms[pk_i])
    med = float(np.median(Ms) + 1e-12)
    ratio = pk_v / med

    # refine start: walk backward on raw M
    st = pk_i
    for i in range(pk_i, max(0, pk_i - 5*half_period), -1):
        if M[i] < 0.1*pk_v:
            st = i + 1
            break

    # CFO from phase(Psum)
    p_sum = np.sum(P[st: st + plateau])
    cfo_hz = -np.angle(p_sum) * (fs / (2*np.pi*half_period))

    ok = (pk_v > sc_threshold) and (ratio >= peak_med_ratio_th)
    return ok, int(st), pk_v, float(cfo_hz), float(ratio), M

def detect_stf_xcorr(rx: np.ndarray, stf_ref: np.ndarray, search_len: int = 60000):
    L = len(stf_ref)
    search_len = min(search_len, len(rx)-L)
    if search_len <= 0:
        return -1, 0.0
    seg = rx[:search_len+L]
    corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
    corr = corr / (np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
    idx = int(np.argmax(corr))
    return idx, float(corr[idx])

def extract_ofdm_symbol(rx: np.ndarray, start_idx: int):
    """Step5 convention: fftshift(fft())"""
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    td = rx[start_idx + N_CP : start_idx + SYMBOL_LEN]
    return np.fft.fftshift(np.fft.fft(td))

def channel_estimate_from_ltf(rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray, num_symbols: int = 4):
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    used_idx = np.array([(k + N_FFT) % N_FFT for k in used], dtype=int)

    Ys = []
    for i in range(num_symbols):
        Y = extract_ofdm_symbol(rx, ltf_start + i*SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if not Ys:
        return None, None

    Yavg = np.mean(np.stack(Ys, axis=0), axis=0)
    H = np.ones(N_FFT, dtype=np.complex64)
    for k in used:
        idx = (k + N_FFT) % N_FFT
        if np.abs(ltf_freq_ref[idx]) > 1e-6:
            H[idx] = Yavg[idx] / ltf_freq_ref[idx]

    if len(Ys) >= 2:
        Ystk = np.stack(Ys, axis=0)[:, used_idx]
        noise_var = np.var(Ystk, axis=0)
        sig_var = np.abs(np.mean(Ystk, axis=0))**2
        snr_sc = 10*np.log10(sig_var / (noise_var + 1e-12) + 1e-12)
    else:
        snr_sc = np.zeros(len(used), dtype=np.float32)

    return H, snr_sc

def demod_payload(rx: np.ndarray, payload_start: int, H: np.ndarray,
                  num_syms: int, kp: float, ki: float,
                  bad_evm_th: float, bad_evm_patience: int):
    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
    pilot_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS], dtype=int)
    data_idx  = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS], dtype=int)
    ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)

    phase_acc = 0.0
    freq_acc = 0.0
    prev_phase = None
    bad_cnt = 0

    all_data = []
    evm_sym = []
    ph_errs = []
    fr_log  = []

    for si in range(num_syms):
        Y = extract_ofdm_symbol(rx, payload_start + si*SYMBOL_LEN)
        if Y is None:
            break

        # equalize used subcarriers
        Yeq = Y.copy()
        for k in range(-26, 27):
            if k == 0:
                continue
            idx = (k + N_FFT) % N_FFT
            if np.abs(H[idx]) > 1e-6:
                Yeq[idx] = Yeq[idx] / (H[idx] + 1e-12)

        # pilot-aided FLL + PLL
        pilot_sign = 1 if (si % 2 == 0) else -1
        exp_p = pilot_sign * pilot_pattern
        rp = Yeq[pilot_idx]
        ph = float(np.angle(np.sum(rp * np.conj(exp_p))))

        if prev_phase is not None:
            dph = ph - prev_phase
            while dph > np.pi:
                dph -= 2*np.pi
            while dph < -np.pi:
                dph += 2*np.pi
            freq_acc += ki * dph
        prev_phase = ph
        phase_acc += freq_acc + kp * ph
        Yeq *= np.exp(-1j * phase_acc).astype(np.complex64)

        ds = Yeq[data_idx]

        nearest = ideal[np.argmin(np.abs(ds[:, None] - ideal[None, :]), axis=1)]
        evm = float(np.sqrt(np.mean(np.abs(ds - nearest)**2)))

        all_data.append(ds)
        evm_sym.append(evm)
        ph_errs.append(ph)
        fr_log.append(freq_acc)

        if evm > bad_evm_th:
            bad_cnt += 1
            if bad_cnt >= bad_evm_patience:
                break
        else:
            bad_cnt = 0

    if not all_data:
        return None, {}

    all_data = np.concatenate(all_data)
    diag = {
        "all_data_syms": all_data,
        "evm_per_sym": np.array(evm_sym, dtype=np.float32),
        "phase_err": np.array(ph_errs, dtype=np.float32),
        "freq_log": np.array(fr_log, dtype=np.float32),
    }
    return all_data, diag

def packbits_with_slip(bits: np.ndarray, slip: int) -> bytes:
    if slip > 0:
        if len(bits) <= slip:
            return b""
        bits = bits[slip:]
    return bits_to_bytes(bits)

def parse_vid7_packet(data: bytes):
    if len(data) < 16:
        return False, -1, -1, 0, b"", 0, "too_short"
    if data[:4] != MAGIC:
        return False, -1, -1, 0, b"", 0, "bad_magic"
    fid = int.from_bytes(data[4:6], "little")
    seq = int.from_bytes(data[6:8], "little")
    total = int.from_bytes(data[8:10], "little")
    plen = int.from_bytes(data[10:12], "little")
    need = 12 + plen + 4
    if len(data) < need:
        return False, fid, seq, total, b"", need, "need_more"
    content = data[:12+plen]
    payload = data[12:12+plen]
    crc_rx = int.from_bytes(data[12+plen:12+plen+4], "little")
    crc_ok = (zlib.crc32(content) & 0xFFFFFFFF) == crc_rx
    if not crc_ok:
        return False, fid, seq, total, b"", need, "crc_mismatch"
    return True, fid, seq, total, payload, need, "ok"

# =========================
# Frame accumulator
# =========================
class FrameAccumulator:
    def __init__(self, max_inflight: int = 10, timeout_s: float = 60.0):
        self.frames = {}   # fid -> entry
        self.done = set()
        self.max_inflight = max_inflight
        self.timeout_s = timeout_s

    def add(self, fid: int, seq: int, total: int, payload: bytes, evm: float, cfo: float, data_syms: np.ndarray):
        if fid in self.done:
            return False
        now = time.time()
        if fid not in self.frames:
            self.frames[fid] = {
                "pkts": {},
                "total": total,
                "t0": now,
                "t1": now,
                "evm_sum": 0.0,
                "evm_cnt": 0,
                "cfo_list": [],
                "meta": {},  # seq -> {evm, data_syms}
            }
        e = self.frames[fid]
        e["t1"] = now
        if seq not in e["pkts"]:
            e["pkts"][seq] = payload
            e["evm_sum"] += float(evm)
            e["evm_cnt"] += 1
            e["cfo_list"].append(float(cfo))
            e["meta"][seq] = {"evm": float(evm), "data_syms": data_syms.copy() if isinstance(data_syms, np.ndarray) else np.array([], dtype=np.complex64)}

        return len(e["pkts"]) >= e["total"]

    def get_frame_bytes(self, fid: int):
        if fid not in self.frames:
            return None
        e = self.frames[fid]
        out = b""
        for i in range(e["total"]):
            if i not in e["pkts"]:
                return None
            out += e["pkts"][i]
        return out

    def stats(self, fid: int):
        if fid not in self.frames:
            return {}
        e = self.frames[fid]
        r = len(e["pkts"])
        t = e["total"]
        return {
            "received": r,
            "total": t,
            "completion": (r/t) if t else 0.0,
            "mean_evm": (e["evm_sum"]/e["evm_cnt"]) if e["evm_cnt"] else 0.0,
            "mean_cfo": float(np.mean(e["cfo_list"])) if e["cfo_list"] else 0.0,
            "latency": e["t1"] - e["t0"],
        }

    def mark_done(self, fid: int):
        self.done.add(fid)
        if fid in self.frames:
            del self.frames[fid]

    def expire(self):
        now = time.time()
        expired = []
        for fid in list(self.frames.keys()):
            e = self.frames[fid]
            if now - e["t1"] > self.timeout_s:
                expired.append((fid, self.stats(fid)))
                del self.frames[fid]

        while len(self.frames) > self.max_inflight:
            oldest = min(self.frames.keys(), key=lambda k: self.frames[k]["t1"])
            expired.append((oldest, self.stats(oldest)))
            del self.frames[oldest]
        return expired

    def active_summary(self):
        return {fid: f'{len(e["pkts"])}/{e["total"]}' for fid, e in self.frames.items()}

# =========================
# Plots
# =========================
def plot_packet_diag(out_png: str, diag: dict, title: str):
    syms = diag.get("all_data_syms", np.array([], dtype=np.complex64))
    evm = diag.get("evm_per_sym", np.array([], dtype=np.float32))
    ph  = diag.get("phase_err", np.array([], dtype=np.float32))
    fr  = diag.get("freq_log", np.array([], dtype=np.float32))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    if syms.size:
        n = min(syms.size, 3000)
        ax.scatter(np.real(syms[:n]), np.imag(syms[:n]), s=3, alpha=0.5)
        ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        ax.scatter(np.real(ideal), np.imag(ideal), c="r", s=80, marker="x")
    ax.set_title("Constellation (subset)")
    ax.axis("equal")
    ax.grid(True)

    ax = axes[0, 1]
    if evm.size:
        ax.plot(evm, "-o", markersize=3)
        ax.axhline(float(np.mean(evm)), linestyle="--")
    ax.set_title("EVM per OFDM symbol")
    ax.grid(True)

    ax = axes[1, 0]
    if ph.size:
        ax.plot(np.degrees(ph), "-o", markersize=3)
    ax.set_title("Pilot phase error (deg)")
    ax.grid(True)

    ax = axes[1, 1]
    if fr.size:
        ax.plot(fr, "-o", markersize=3)
    ax.set_title("Freq accumulator (rad/sym)")
    ax.grid(True)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def plot_summary(out_png: str, rows: list[dict], frames_ok: int, frames_expired: int, elapsed: float):
    cfos = [r["cfo_hz"] for r in rows if r.get("status") == "demod_ok"]
    evms = [r["mean_evm"] for r in rows if (r.get("status") == "demod_ok" and r.get("mean_evm") is not None)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    if cfos:
        ax.plot(cfos, ".", markersize=2)
        ax.set_title(f"CFO per OK packet (std={np.std(cfos):.1f} Hz)")
    ax.grid(True)

    ax = axes[0, 1]
    if evms:
        ax.hist(evms, bins=30, edgecolor="black", alpha=0.8)
        ax.set_title(f"EVM distribution (mean={np.mean(evms):.4f})")
    ax.grid(True)

    ax = axes[1, 0]
    status_map = {}
    for r in rows:
        s = r.get("status", "na")
        status_map[s] = status_map.get(s, 0) + 1
    keys = sorted(status_map.keys())
    ax.bar(keys, [status_map[k] for k in keys])
    ax.set_title("Capture status counts")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"Frames OK: {frames_ok}\n"
        f"Frames expired: {frames_expired}\n"
        f"Elapsed: {elapsed:.1f}s\n"
        f"Effective FPS: {(frames_ok/elapsed):.4f}\n" if elapsed > 0 else ""
    )
    ax.text(0.05, 0.95, txt, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

    fig.suptitle("Step5-based Step7 RX Summary", fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

# =========================
# Config
# =========================
@dataclass
class RXArgs:
    uri: str = "ip:192.168.2.2"
    fc: float = 2.3e9
    fs: float = 3e6
    rx_gain: float = 30.0
    rx_bw: float = 3.6e6
    rx_buffer_size: int = 131072
    kernel_buffers: int = 4

    stf_repeats: int = 6
    ltf_symbols: int = 4

    repeat: int = 1
    kp: float = 0.15
    ki: float = 0.005

    sc_threshold: float = 0.2
    search_len: int = 60000
    sync_peak_med_ratio_th: float = 6.0
    xcorr_refine: bool = True
    xcorr_window: int = 2000

    probe_syms: int = 16
    bit_slip_max: int = 7
    max_payload_syms_cap: int = 260

    bad_evm_th: float = 0.85
    bad_evm_patience: int = 5

    max_captures: int = 5000
    max_frames: int = 3
    frame_timeout_s: float = 60.0
    max_inflight: int = 10

    out_root: str = "rf_link_step7_rx_runs"
    output_video: str = "received_video.avi"
    output_fps: float = 2.0
    width: int = 320
    height: int = 240
    save_video: bool = True
    save_frames_dir: str = ""
    plot_every_frame: int = 1
    verbose: bool = True
    sparse_every: int = 200
    save_csv: bool = True
    save_plots: bool = True

# =========================
# Demod one capture (probe + bounded)
# =========================
def demod_one_capture_step5phy(rx_raw: np.ndarray, cfg: RXArgs, stf_ref: np.ndarray, ltf_freq_ref: np.ndarray):
    # normalize / DC remove
    rx = rx_raw.astype(np.complex64)
    if np.median(np.abs(rx)) > 100:
        rx = rx / (2**14)
    rx = rx - np.mean(rx)

    peak = float(np.max(np.abs(rx))) if rx.size else 0.0

    # SC timing + CFO
    ok_sc, sc_idx, sc_peak, cfo_hz, ratio, sc_M = detect_stf_sc(
        rx, fs=cfg.fs,
        half_period=N_FFT//2,
        sc_threshold=cfg.sc_threshold,
        search_len=cfg.search_len,
        peak_med_ratio_th=cfg.sync_peak_med_ratio_th
    )

    # If SC not ok, still try global xcorr as fallback
    rx_cfo = apply_cfo(rx, cfo_hz, cfg.fs)

    stf_idx = sc_idx
    stf_peak = sc_peak
    method = "sc"

    if cfg.xcorr_refine:
        if ok_sc:
            s0 = max(0, sc_idx - cfg.xcorr_window)
            s1 = min(len(rx_cfo), sc_idx + cfg.xcorr_window + len(stf_ref))
            seg = rx_cfo[s0:s1]
            if len(seg) > len(stf_ref) + 10:
                corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
                corr /= (np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
                loc = int(np.argmax(corr))
                stf_idx = s0 + loc
                stf_peak = float(corr[loc])
                method = "sc+xc_refine"
        else:
            xi, xp = detect_stf_xcorr(rx_cfo, stf_ref, search_len=cfg.search_len)
            stf_idx, stf_peak = xi, xp
            method = "xc_fallback"

    # gate: require SC ratio if SC says ok; otherwise require reasonable xcorr peak
    if ok_sc and ratio < cfg.sync_peak_med_ratio_th:
        return {"status": "no_signal", "reason": "ratio_gate", "ratio": ratio, "cfo_hz": cfo_hz, "peak": peak, "method": method}
    if (not ok_sc) and (stf_peak < 0.02):
        return {"status": "no_signal", "reason": "xcorr_low", "ratio": ratio, "cfo_hz": cfo_hz, "peak": peak, "method": method}

    # channel estimate
    ltf_start = stf_idx + len(stf_ref)
    H, snr_sc = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref, num_symbols=cfg.ltf_symbols)
    if H is None:
        return {"status": "ltf_fail", "ratio": ratio, "cfo_hz": cfo_hz, "peak": peak, "method": method}

    payload_start = ltf_start + cfg.ltf_symbols * SYMBOL_LEN

    # ---- probe ----
    data_syms_probe, diag_probe = demod_payload(
        rx_cfo, payload_start, H,
        num_syms=int(cfg.probe_syms),
        kp=float(cfg.kp), ki=float(cfg.ki),
        bad_evm_th=float(cfg.bad_evm_th),
        bad_evm_patience=int(cfg.bad_evm_patience),
    )
    if data_syms_probe is None:
        return {"status": "no_payload", "ratio": ratio, "cfo_hz": cfo_hz, "peak": peak, "method": method}

    bits_raw = qpsk_demap(diag_probe["all_data_syms"])
    bits = majority_vote(bits_raw, int(cfg.repeat))

    found = None
    found_slip = None
    found_need = None
    for slip in range(int(cfg.bit_slip_max)+1):
        bb = packbits_with_slip(bits, slip)
        ok, fid, seq, total, payload, need, why = parse_vid7_packet(bb)
        if ok:
            found = (fid, seq, total, payload)
            found_slip = slip
            found_need = need
            break

    if found is None:
        return {
            "status": "no_packet",
            "reason": "no_magic_in_probe",
            "ratio": ratio,
            "cfo_hz": cfo_hz,
            "peak": peak,
            "method": method,
            "mean_evm": float(np.mean(diag_probe["evm_per_sym"])) if diag_probe["evm_per_sym"].size else None,
        }

    # ---- bounded demod ----
    required_bits = found_need * 8 + int(found_slip)
    required_syms = int(np.ceil(required_bits / BITS_PER_OFDM_SYM))
    required_syms = min(required_syms, int(cfg.max_payload_syms_cap))

    data_syms_full, diag_full = demod_payload(
        rx_cfo, payload_start, H,
        num_syms=required_syms,
        kp=float(cfg.kp), ki=float(cfg.ki),
        bad_evm_th=float(cfg.bad_evm_th),
        bad_evm_patience=int(cfg.bad_evm_patience),
    )
    if data_syms_full is None:
        return {"status": "no_payload", "ratio": ratio, "cfo_hz": cfo_hz, "peak": peak, "method": method}

    bits_raw2 = qpsk_demap(diag_full["all_data_syms"])
    bits2 = majority_vote(bits_raw2, int(cfg.repeat))

    bb2 = packbits_with_slip(bits2, int(found_slip))
    ok2, fid2, seq2, total2, payload2, need2, why2 = parse_vid7_packet(bb2)
    if not ok2:
        return {
            "status": "crc_fail",
            "reason": why2,
            "ratio": ratio,
            "cfo_hz": cfo_hz,
            "peak": peak,
            "method": method,
            "mean_evm": float(np.mean(diag_full["evm_per_sym"])) if diag_full["evm_per_sym"].size else None,
        }

    mean_evm = float(np.mean(diag_full["evm_per_sym"])) if diag_full["evm_per_sym"].size else 0.0

    return {
        "status": "demod_ok",
        "ratio": ratio,
        "cfo_hz": cfo_hz,
        "peak": peak,
        "method": method,
        "frame_id": int(fid2),
        "seq": int(seq2),
        "total": int(total2),
        "payload": payload2,
        "need_bytes": int(need2),
        "mean_evm": mean_evm,
        "diag": diag_full,
        "snr_sc": snr_sc,
        "stf_idx": int(stf_idx),
        "stf_peak": float(stf_peak),
    }

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("Step5-based Step7 RX (VID7)")
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--rx_gain", type=float, default=30.0)
    ap.add_argument("--rx_bw", type=float, default=3.6e6)
    ap.add_argument("--rx_buffer_size", type=int, default=131072)
    ap.add_argument("--kernel_buffers", type=int, default=4)

    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.005)

    ap.add_argument("--probe_syms", type=int, default=16)
    ap.add_argument("--bit_slip_max", type=int, default=7)
    ap.add_argument("--max_payload_syms_cap", type=int, default=260)

    ap.add_argument("--sync_ratio", type=float, default=6.0)
    ap.add_argument("--sc_threshold", type=float, default=0.2)
    ap.add_argument("--search_len", type=int, default=60000)

    ap.add_argument("--bad_evm_th", type=float, default=0.85)
    ap.add_argument("--bad_evm_patience", type=int, default=5)

    ap.add_argument("--max_captures", type=int, default=5000)
    ap.add_argument("--max_frames", type=int, default=3)
    ap.add_argument("--frame_timeout_s", type=float, default=60.0)
    ap.add_argument("--max_inflight", type=int, default=10)

    ap.add_argument("--out_root", default="rf_link_step7_rx_runs")
    ap.add_argument("--output_video", default="received_video.avi")
    ap.add_argument("--output_fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--save_frames_dir", default="")
    ap.add_argument("--plot_every_frame", type=int, default=1)

    ap.add_argument("--sparse_every", type=int, default=200)
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = RXArgs(
        uri=args.uri, fc=args.fc, fs=args.fs,
        rx_gain=args.rx_gain, rx_bw=args.rx_bw,
        rx_buffer_size=args.rx_buffer_size,
        kernel_buffers=args.kernel_buffers,
        stf_repeats=args.stf_repeats, ltf_symbols=args.ltf_symbols,
        repeat=args.repeat, kp=args.kp, ki=args.ki,
        sc_threshold=args.sc_threshold, search_len=args.search_len,
        sync_peak_med_ratio_th=args.sync_ratio,
        probe_syms=args.probe_syms, bit_slip_max=args.bit_slip_max,
        max_payload_syms_cap=args.max_payload_syms_cap,
        bad_evm_th=args.bad_evm_th, bad_evm_patience=args.bad_evm_patience,
        max_captures=args.max_captures, max_frames=args.max_frames,
        frame_timeout_s=args.frame_timeout_s, max_inflight=args.max_inflight,
        out_root=args.out_root, output_video=args.output_video,
        output_fps=args.output_fps, width=args.width, height=args.height,
        save_video=bool(args.save_video),
        save_frames_dir=args.save_frames_dir,
        plot_every_frame=args.plot_every_frame,
        sparse_every=args.sparse_every,
        save_csv=not args.no_csv,
        save_plots=not args.no_plots,
        verbose=bool(args.verbose),
    )

    import adi
    import cv2

    # run dir
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    if cfg.save_frames_dir:
        os.makedirs(cfg.save_frames_dir, exist_ok=True)

    # refs
    stf_ref, _ = create_schmidl_cox_stf(cfg.stf_repeats)
    _, ltf_freq_ref = create_ltf_ref(cfg.ltf_symbols)

    print("\n" + "="*90)
    print("Step5-based Step7 RX (VID7) - Config")
    print("="*90)
    print(f"run_dir={run_dir}")
    print(f"uri={cfg.uri} fc={cfg.fc/1e6:.1f}MHz fs={cfg.fs/1e6:.1f}Msps rx_gain={cfg.rx_gain}")
    print(f"buf={cfg.rx_buffer_size} stf_repeats={cfg.stf_repeats} ltf_syms={cfg.ltf_symbols} repeat={cfg.repeat}")
    print(f"probe_syms={cfg.probe_syms} bit_slip_max={cfg.bit_slip_max} max_payload_syms_cap={cfg.max_payload_syms_cap}")
    print(f"sync_ratio_th={cfg.sync_peak_med_ratio_th} sc_th={cfg.sc_threshold} kp={cfg.kp} ki={cfg.ki}")
    print("="*90)

    # SDR
    sdr = adi.Pluto(uri=cfg.uri)
    sdr.sample_rate = int(cfg.fs)
    sdr.rx_lo = int(cfg.fc)
    sdr.rx_rf_bandwidth = int(cfg.rx_bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(cfg.rx_gain)
    sdr.rx_buffer_size = int(cfg.rx_buffer_size)
    sdr.rx_enabled_channels = [0]

    # kernel buffers
    try:
        if hasattr(sdr, "_rxadc") and sdr._rxadc is not None:
            sdr._rxadc.set_kernel_buffers_count(int(cfg.kernel_buffers))
    except Exception:
        pass

    # flush
    for _ in range(4):
        _ = sdr.rx()

    # video writer
    vw = None
    if cfg.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vw = cv2.VideoWriter(os.path.join(run_dir, cfg.output_video), fourcc, float(cfg.output_fps), (int(cfg.width), int(cfg.height)))

    acc = FrameAccumulator(max_inflight=cfg.max_inflight, timeout_s=cfg.frame_timeout_s)

    # CSV log
    csv_path = os.path.join(run_dir, "packets.csv")
    csv_f = None
    csv_w = None
    if cfg.save_csv:
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.DictWriter(csv_f, fieldnames=[
            "capture","status","reason","ratio","cfo_hz","peak","mean_evm",
            "frame_id","seq","total","payload_len","method"
        ])
        csv_w.writeheader()

    rows = []
    frames_written = 0
    frames_expired = 0

    t0 = time.time()
    print("Listening ... Ctrl+C to stop")

    try:
        for ci in range(1, cfg.max_captures+1):
            rx_raw = sdr.rx()
            r = demod_one_capture_step5phy(rx_raw, cfg, stf_ref, ltf_freq_ref)

            status = r.get("status", "na")
            reason = r.get("reason","")
            ratio = float(r.get("ratio", 0.0))
            cfo_hz = float(r.get("cfo_hz", 0.0))
            peak = float(r.get("peak", 0.0))
            mean_evm = r.get("mean_evm", None)
            method = r.get("method","")

            fid = r.get("frame_id", -1)
            seq = r.get("seq", -1)
            total = r.get("total", 0)
            payload_len = len(r.get("payload", b"")) if isinstance(r.get("payload", b""), (bytes, bytearray)) else 0

            row = {
                "capture": ci,
                "status": status,
                "reason": reason,
                "ratio": ratio,
                "cfo_hz": cfo_hz,
                "peak": peak,
                "mean_evm": float(mean_evm) if mean_evm is not None else "",
                "frame_id": fid if fid is not None else "",
                "seq": seq if seq is not None else "",
                "total": total if total is not None else "",
                "payload_len": payload_len,
                "method": method
            }
            rows.append(row)
            if csv_w:
                csv_w.writerow(row)

            if cfg.verbose and (ci <= 20 or (ci % cfg.sparse_every == 0)):
                tag = ""
                if status == "demod_ok":
                    tag = f"F{fid} pkt{seq}/{total-1} payload={payload_len}B"
                print(f"[{ci:05d}] {status:8s} ratio={ratio:6.2f} CFO={cfo_hz:+8.1f}Hz peak={peak:5.1f} {reason} {tag}")

            # if packet OK, add to frame
            if status == "demod_ok":
                diag = r.get("diag", {})
                data_syms = diag.get("all_data_syms", np.array([], dtype=np.complex64))
                complete = acc.add(fid, seq, total, r["payload"], evm=float(r.get("mean_evm", 0.0)), cfo=cfo_hz, data_syms=data_syms)

                if complete:
                    jpg = acc.get_frame_bytes(fid)
                    stt = acc.stats(fid)
                    img = None
                    if jpg is not None and len(jpg) > 0:
                        nparr = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        if img.shape[1] != cfg.width or img.shape[0] != cfg.height:
                            img = cv2.resize(img, (cfg.width, cfg.height))
                        if vw is not None:
                            vw.write(img)
                        frames_written += 1

                        if cfg.save_frames_dir:
                            cv2.imwrite(os.path.join(cfg.save_frames_dir, f"frame_{fid:04d}.jpg"), img)

                        print(f"  >>> Frame {fid} COMPLETE: jpeg={len(jpg)}B latency={stt.get('latency',0):.2f}s "
                              f"EVM={stt.get('mean_evm',0):.4f} [{frames_written}]")

                        # plot per-frame diag (use aggregated symbols from last packet as a proxy)
                        if cfg.save_plots and (frames_written % cfg.plot_every_frame == 0):
                            out_png = os.path.join(run_dir, f"frame_{fid:04d}_diag.png")
                            plot_packet_diag(out_png, diag, f"Frame {fid} (last packet diag)")

                    else:
                        print(f"  >>> Frame {fid} COMPLETE but JPEG decode FAILED ({len(jpg) if jpg else 0}B)")

                    acc.mark_done(fid)
                    if cfg.max_frames > 0 and frames_written >= cfg.max_frames:
                        print(f"Reached max_frames={cfg.max_frames}.")
                        break

            # expire old frames
            expired = acc.expire()
            for efid, est in expired:
                frames_expired += 1
                print(f"  [EXPIRED] Frame {efid}: {est.get('received',0)}/{est.get('total',0)} pkts")

            if ci % 500 == 0:
                elapsed = time.time() - t0
                print(f"\n--- Status @{ci} captures ---")
                print(f"frames_written={frames_written}, active={acc.active_summary()}, elapsed={elapsed:.1f}s\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        elapsed = time.time() - t0
        if vw is not None:
            vw.release()
        if csv_f:
            csv_f.close()
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

        print("\n" + "="*90)
        print("FINAL SUMMARY")
        print("="*90)
        print(f"run_dir: {run_dir}")
        print(f"frames_written: {frames_written}")
        print(f"frames_expired: {frames_expired}")
        print(f"elapsed: {elapsed:.1f}s  fps={(frames_written/elapsed):.4f}" if elapsed > 0 else "")
        if cfg.save_csv:
            print(f"csv: {csv_path}")

        if cfg.save_plots:
            out_png = os.path.join(run_dir, "summary.png")
            plot_summary(out_png, rows, frames_written, frames_expired, elapsed)
            print(f"summary plot: {out_png}")
        print("="*90)

if __name__ == "__main__":
    main()

"""
RX (Intel) — match repeat, keep probe small at first
python3 rf_link_step7_rx_step5phy.py \
  --uri ip:192.168.2.2 \
  --fc 2.3e9 --fs 3e6 \
  --rx_gain 30 \
  --repeat 1 \
  --probe_syms 24 \
  --max_payload_syms_cap 260 \
  --save_video \
  --verbose
"""
