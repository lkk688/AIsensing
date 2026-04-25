#!/usr/bin/env python3
"""
Step5 Debug Capture RX (AIS1) - PlutoSDR
Purpose: white-box data collection for offline analysis (not just "try decode").

What it saves per capture:
- rx (normalized, DC-removed) complex64
- optionally rx_cfo (CFO corrected)
- tone matched-filter curve (to locate tone robustly)
- Schmidl-Cox metric curve (timing)
- STF cross-correlation curve (global and local refine)
- chosen STF index / LTF index / candidate list
- channel estimate H (from averaged LTF symbols)
- probe demod diagnostics: constellation subset, EVM per symbol, pilot phase, freq_acc
- first bytes of probe packet (so you can see if magic is close)

Outputs:
run_dir/
  captures.csv                # one-line summary per capture
  cap_0001.npz ...            # per-capture arrays + metadata
  cap_0001.png ...            # optional debug plot per capture
  summary.png                 # optional summary plot

Notes:
- Designed to compare "good vs bad" captures.
- Default parameters assume Step5 TX frame:
  gap_long=3000, tone=10ms@100k, gap_short=1000, STF, LTF, payload.
"""

import argparse
import os
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
import zlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# PHY constants (must match TX)
# -----------------------------
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array([k for k in range(-26, 27)
                              if (k != 0 and k not in set(PILOT_SUBCARRIERS))], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)         # 48
BITS_PER_OFDM_SYM = N_DATA * 2         # 96

MAGIC = b"AIS1"

# -----------------------------
# Helpers
# -----------------------------
def sc_to_bin(k: int) -> int:
    """Subcarrier index (-32..31) to FFT bin index (0..63) for fftshifted convention."""
    return (k + N_FFT) % N_FFT

PILOT_BINS = np.array([sc_to_bin(int(k)) for k in PILOT_SUBCARRIERS], dtype=int)
DATA_BINS  = np.array([sc_to_bin(int(k)) for k in DATA_SUBCARRIERS], dtype=int)

def autoscale_and_dc_remove(rx_raw: np.ndarray) -> np.ndarray:
    rx = rx_raw.astype(np.complex64)
    # If pyadi-iio returns int14-like range, median magnitude will be huge
    if np.median(np.abs(rx)) > 100:
        rx = rx / (2**14)
    rx = rx - np.mean(rx)
    return rx.astype(np.complex64)

def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
    re = (np.real(symbols) >= 0)
    im = (np.imag(symbols) >= 0)
    # Gray mapping consistent with TX:
    # 00: + + ; 01: - + ; 11: - - ; 10: + -
    bits[0::2] = (~re).astype(np.uint8)  # b0
    bits[1::2] = (re ^ im).astype(np.uint8)  # b1
    return bits

def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    if L <= 0:
        return np.zeros(0, dtype=np.uint8)
    x = bits[:L].reshape(-1, repeat)
    return (np.sum(x, axis=1) >= (repeat / 2)).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    L = (len(bits) // 8) * 8
    if L <= 0:
        return b""
    return np.packbits(bits[:L]).tobytes()

def parse_packet(data: bytes):
    """Step4/5 packet: MAGIC(4) LEN(2) PAYLOAD CRC32(payload)(4)"""
    if len(data) < 10:
        return False, "too_short", b"", 0, 0
    if data[:4] != MAGIC:
        return False, "bad_magic", b"", 0, 0
    plen = int.from_bytes(data[4:6], "little")
    need = 6 + plen + 4
    if len(data) < need:
        return False, "need_more", b"", 0, 0
    payload = data[6:6+plen]
    crc_rx = int.from_bytes(data[6+plen:6+plen+4], "little")
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return False, "crc_mismatch", payload, crc_rx, crc_calc
    return True, "ok", payload, crc_rx, crc_calc

# -----------------------------
# Preamble refs (Step5)
# -----------------------------
def create_schmidl_cox_stf(num_repeats: int = 6):
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i] + 0j
    # ifftshift(X) because X currently is in fftshifted bin indexing
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x, num_repeats).astype(np.complex64)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp, X

def create_ltf_ref(num_symbols: int = 4):
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X

# -----------------------------
# Core metrics
# -----------------------------
def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(len(samples), dtype=np.float32)
    return (samples * np.exp(-1j * 2*np.pi*(cfo_hz/fs)*n)).astype(np.complex64)

def schmidl_cox_metric(rx: np.ndarray, half_period: int = 32, window_len: int = 60000):
    L = half_period
    N = len(rx)
    window_len = min(window_len, N - 2 * L)
    if window_len <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.complex64)

    c = rx[:window_len + L] * np.conj(rx[L:window_len + 2 * L])
    cs = np.zeros(window_len + L + 1, dtype=np.complex64)
    cs[1:] = np.cumsum(c)
    P = cs[L:L + window_len] - cs[:window_len]

    pow2 = np.abs(rx[:window_len + 2 * L])**2
    cr = np.zeros(window_len + 2 * L + 1, dtype=np.float32)
    cr[1:] = np.cumsum(pow2)
    R = cr[2 * L:2 * L + window_len] - cr[L:L + window_len]

    M = (np.abs(P)**2) / (R**2 + 1e-12)
    return M.astype(np.float32), P

def detect_stf_sc(rx: np.ndarray, fs: float,
                  sc_threshold: float, peak_med_ratio_th: float,
                  search_len: int = 60000, half_period: int = 32):
    M, P = schmidl_cox_metric(rx, half_period=half_period, window_len=search_len)
    if M.size == 0:
        return False, 0, 0.0, 0.0, 0.0, M

    plateau = 2 * half_period
    if len(M) <= plateau + 1:
        pk = float(np.max(M))
        ratio = float(pk / (np.median(M) + 1e-12))
        return False, int(np.argmax(M)), pk, 0.0, ratio, M

    kernel = np.ones(plateau, dtype=np.float32) / plateau
    Ms = np.convolve(M, kernel, mode="valid")
    pk_i = int(np.argmax(Ms))
    pk_v = float(Ms[pk_i])
    med = float(np.median(Ms) + 1e-12)
    ratio = pk_v / med

    # refine rising edge
    st = pk_i
    for i in range(pk_i, max(0, pk_i - 5 * half_period), -1):
        if M[i] < 0.1 * pk_v:
            st = i + 1
            break

    # CFO from phase(Psum)
    p_sum = np.sum(P[st: st + plateau])
    cfo_hz = -np.angle(p_sum) * (fs / (2 * np.pi * half_period))

    ok = (pk_v > sc_threshold) and (ratio >= peak_med_ratio_th)
    return ok, int(st), pk_v, float(cfo_hz), float(ratio), M

def detect_stf_xcorr(rx: np.ndarray, stf_ref: np.ndarray, search_len: int = 60000):
    L = len(stf_ref)
    search_len = min(search_len, len(rx) - L)
    if search_len <= 0:
        return -1, 0.0, np.array([], dtype=np.float32)
    seg = rx[:search_len + L]
    corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
    corr = corr / (np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
    idx = int(np.argmax(corr))
    return idx, float(corr[idx]), corr.astype(np.float32)

def detect_tone_matched_filter(rx: np.ndarray, fs: float, tone_hz: float,
                               win_len: int = 8192, hop: int = 512):
    """
    Robustly locate a known single-tone using matched filtering.
    Returns (best_start, peak, peak/median, metric_vector, idx_vector).
    """
    if len(rx) < win_len + 1:
        return -1, 0.0, 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    n = np.arange(win_len, dtype=np.float32)
    ref = np.exp(-1j * 2*np.pi*tone_hz*n/fs).astype(np.complex64)
    ref /= (np.linalg.norm(ref) + 1e-12)

    vals = []
    idxs = []
    for s in range(0, len(rx) - win_len, hop):
        seg = rx[s:s+win_len]
        v = np.abs(np.vdot(seg, ref))  # |sum(seg*conj(ref))|
        vals.append(v)
        idxs.append(s)

    vals = np.asarray(vals, dtype=np.float32)
    idxs = np.asarray(idxs, dtype=np.int32)
    k = int(np.argmax(vals))
    peak = float(vals[k])
    med = float(np.median(vals) + 1e-12)
    return int(idxs[k]), peak, peak / med, vals, idxs

def extract_ofdm_symbol(rx: np.ndarray, start_idx: int):
    """Step5 FFT convention: fftshift(fft()) on CP-removed symbol."""
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    td = rx[start_idx + N_CP : start_idx + SYMBOL_LEN]
    return np.fft.fftshift(np.fft.fft(td))

def channel_estimate_from_ltf(rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray, num_symbols: int = 4):
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    used_bins = np.array([sc_to_bin(int(k)) for k in used], dtype=int)

    Ys = []
    for i in range(num_symbols):
        Y = extract_ofdm_symbol(rx, ltf_start + i * SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if not Ys:
        return None, None, 0.0

    Yavg = np.mean(np.stack(Ys, axis=0), axis=0)

    H = np.ones(N_FFT, dtype=np.complex64)
    for k in used:
        b = sc_to_bin(int(k))
        if np.abs(ltf_freq_ref[b]) > 1e-6:
            H[b] = Yavg[b] / ltf_freq_ref[b]

    # LTF quality: mean(|H|)^2 / var(|H|)
    Hm = np.abs(H[used_bins])
    q = float((np.mean(Hm)**2) / (np.var(Hm) + 1e-12))

    # per-subcarrier SNR proxy (variance across LTF symbols)
    if len(Ys) >= 2:
        Ystk = np.stack(Ys, axis=0)[:, used_bins]
        noise_var = np.var(Ystk, axis=0)
        sig_var = np.abs(np.mean(Ystk, axis=0))**2
        snr_sc = 10*np.log10(sig_var / (noise_var + 1e-12) + 1e-12)
    else:
        snr_sc = np.zeros(len(used), dtype=np.float32)

    return H, snr_sc.astype(np.float32), q

def demod_probe(rx: np.ndarray, payload_start: int, H: np.ndarray,
                probe_syms: int, kp: float, ki: float):
    pilot_pat = np.array([1, 1, 1, -1], dtype=np.complex64)
    ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)

    phase_acc = 0.0
    freq_acc = 0.0
    prev_phase = None

    all_data = []
    evm = []
    ph = []
    fr = []

    for si in range(probe_syms):
        Y = extract_ofdm_symbol(rx, payload_start + si * SYMBOL_LEN)
        if Y is None:
            break

        # Equalize (only on used bins; others left as-is)
        Ye = Y.copy()
        for k in range(-26, 27):
            if k == 0:
                continue
            b = sc_to_bin(k)
            if np.abs(H[b]) > 1e-6:
                Ye[b] = Ye[b] / (H[b] + 1e-12)

        # Pilot tracking (FLL + PLL)
        pilot_sign = 1 if (si % 2 == 0) else -1
        exp_p = pilot_sign * pilot_pat
        rp = Ye[PILOT_BINS]
        phase_err = float(np.angle(np.sum(rp * np.conj(exp_p))))

        if prev_phase is not None:
            dph = phase_err - prev_phase
            while dph > np.pi:
                dph -= 2*np.pi
            while dph < -np.pi:
                dph += 2*np.pi
            freq_acc += ki * dph
        prev_phase = phase_err
        phase_acc += freq_acc + kp * phase_err
        Ye *= np.exp(-1j * phase_acc).astype(np.complex64)

        ds = Ye[DATA_BINS]
        nearest = ideal[np.argmin(np.abs(ds[:, None] - ideal[None, :]), axis=1)]
        ev = float(np.sqrt(np.mean(np.abs(ds - nearest)**2)))

        all_data.append(ds)
        evm.append(ev)
        ph.append(phase_err)
        fr.append(freq_acc)

    if not all_data:
        return None

    all_data = np.concatenate(all_data).astype(np.complex64)
    return {
        "data_syms": all_data,
        "evm_per_sym": np.array(evm, dtype=np.float32),
        "pilot_phase": np.array(ph, dtype=np.float32),
        "freq_acc": np.array(fr, dtype=np.float32),
    }

# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    uri: str
    fc: float
    fs: float
    rx_gain: float
    rx_bw: float
    buf_size: int
    kernel_buffers: int

    # expected Step5 TX framing
    tone_hz: float
    tone_ms: float
    gap_short: int
    gap_long: int

    stf_repeats: int
    ltf_symbols: int
    repeat: int

    sc_threshold: float
    sc_ratio_th: float
    search_len: int

    # selection behavior
    use_tone_anchor: bool
    tone_win: int
    tone_hop: int
    tone_ratio_th: float
    stf_refine_win: int
    ltf_off_sweep: int

    kp: float
    ki: float
    probe_syms: int
    bit_slip_max: int

    tries: int
    out_root: str
    save_rx: bool
    save_rx_cfo: bool
    save_curves: bool
    save_plots: bool
    plot_every: int

# -----------------------------
# Plot
# -----------------------------
def plot_capture(path_png: str, meta: dict, arrays: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SC metric
    ax = axes[0, 0]
    M = arrays.get("sc_metric", None)
    if M is not None and len(M) > 0:
        ax.plot(M[:min(60000, len(M))])
        ax.axvline(meta.get("stf_idx", 0), linestyle="--")
    ax.set_title("Schmidl-Cox metric")
    ax.grid(True)

    # Spectrum quicklook
    ax = axes[0, 1]
    rx = arrays.get("rx", None)
    if rx is not None and len(rx) > 0:
        N = min(16384, len(rx))
        win = np.hanning(N).astype(np.float32)
        F = np.fft.fftshift(np.fft.fft(rx[:N] * win))
        fk = np.fft.fftshift(np.fft.fftfreq(N, 1.0/meta["fs"])) / 1e3
        ax.plot(fk, 20*np.log10(np.abs(F) + 1e-10))
        ax.set_xlim([-300, 300])
    ax.set_title("Spectrum (kHz)")
    ax.grid(True)

    # Constellation subset
    ax = axes[1, 0]
    ds = arrays.get("probe_syms", None)
    if ds is not None and len(ds) > 0:
        n = min(len(ds), 2000)
        ax.scatter(np.real(ds[:n]), np.imag(ds[:n]), s=4, alpha=0.5)
        ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        ax.scatter(np.real(ideal), np.imag(ideal), c="r", s=80, marker="x")
    ax.set_title("Constellation (probe subset)")
    ax.axis("equal")
    ax.grid(True)

    # EVM / pilot phase
    ax = axes[1, 1]
    evm = arrays.get("probe_evm", None)
    ph = arrays.get("probe_phase", None)
    if evm is not None and len(evm) > 0:
        ax.plot(evm, "-o", markersize=3, label="EVM")
    if ph is not None and len(ph) > 0:
        ax.plot(np.degrees(ph), "-o", markersize=3, label="Pilot phase (deg)")
    ax.legend()
    ax.grid(True)
    ax.set_title("Probe EVM / Pilot phase")

    fig.suptitle(
        f"cap={meta.get('cap',-1)} status={meta.get('status','?')} "
        f"CFO={meta.get('cfo_hz',0):+.1f}Hz ratio={meta.get('sc_ratio',0):.2f} "
        f"stf={meta.get('stf_idx',0)} ltfQ={meta.get('ltf_q',0):.2f}",
        fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path_png, dpi=140)
    plt.close(fig)

# -----------------------------
# Main capture
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Step5 Debug Capture RX (AIS1)")
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--rx_gain", type=float, default=30.0)
    ap.add_argument("--rx_bw", type=float, default=3.6e6)
    ap.add_argument("--buf_size", type=int, default=262144)
    ap.add_argument("--kernel_buffers", type=int, default=4)

    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])

    ap.add_argument("--tone_hz", type=float, default=100e3)
    ap.add_argument("--tone_ms", type=float, default=10.0)
    ap.add_argument("--gap_short", type=int, default=1000)
    ap.add_argument("--gap_long", type=int, default=3000)

    ap.add_argument("--tries", type=int, default=50)

    ap.add_argument("--sc_threshold", type=float, default=0.08)
    ap.add_argument("--sc_ratio_th", type=float, default=6.0)
    ap.add_argument("--search_len", type=int, default=60000)

    ap.add_argument("--use_tone_anchor", action="store_true", help="Use tone matched filter to anchor STF search window")
    ap.add_argument("--tone_win", type=int, default=8192)
    ap.add_argument("--tone_hop", type=int, default=512)
    ap.add_argument("--tone_ratio_th", type=float, default=6.0)
    ap.add_argument("--stf_refine_win", type=int, default=1500)
    ap.add_argument("--ltf_off_sweep", type=int, default=12)

    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.005)
    ap.add_argument("--probe_syms", type=int, default=16)
    ap.add_argument("--bit_slip_max", type=int, default=16)

    ap.add_argument("--out_root", default="rf_link_step5_debug_runs")
    ap.add_argument("--save_rx", action="store_true", help="Save rx in NPZ (recommended)")
    ap.add_argument("--save_rx_cfo", action="store_true", help="Save rx_cfo in NPZ (bigger)")
    ap.add_argument("--save_curves", action="store_true", help="Save sc/xcorr/tone curves (recommended)")
    ap.add_argument("--save_plots", action="store_true")
    ap.add_argument("--plot_every", type=int, default=1)

    args = ap.parse_args()

    cfg = Cfg(
        uri=args.uri, fc=args.fc, fs=args.fs,
        rx_gain=args.rx_gain, rx_bw=args.rx_bw,
        buf_size=args.buf_size, kernel_buffers=args.kernel_buffers,
        tone_hz=args.tone_hz, tone_ms=args.tone_ms,
        gap_short=args.gap_short, gap_long=args.gap_long,
        stf_repeats=args.stf_repeats, ltf_symbols=args.ltf_symbols, repeat=args.repeat,
        sc_threshold=args.sc_threshold, sc_ratio_th=args.sc_ratio_th, search_len=args.search_len,
        use_tone_anchor=bool(args.use_tone_anchor),
        tone_win=args.tone_win, tone_hop=args.tone_hop, tone_ratio_th=args.tone_ratio_th,
        stf_refine_win=args.stf_refine_win,
        ltf_off_sweep=args.ltf_off_sweep,
        kp=args.kp, ki=args.ki, probe_syms=args.probe_syms, bit_slip_max=args.bit_slip_max,
        tries=args.tries,
        out_root=args.out_root,
        save_rx=bool(args.save_rx),
        save_rx_cfo=bool(args.save_rx_cfo),
        save_curves=bool(args.save_curves),
        save_plots=bool(args.save_plots),
        plot_every=max(1, args.plot_every),
    )

    # run dir
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # refs
    stf_ref, _ = create_schmidl_cox_stf(cfg.stf_repeats)
    _, ltf_freq_ref = create_ltf_ref(cfg.ltf_symbols)

    print("\n" + "="*78)
    print("Step5 Debug Capture RX (AIS1)")
    print("="*78)
    print(f"run_dir={run_dir}")
    print(f"uri={cfg.uri} fc={cfg.fc/1e6:.1f}MHz fs={cfg.fs/1e6:.1f}Msps rx_gain={cfg.rx_gain}")
    print(f"buf={cfg.buf_size} bw={cfg.rx_bw/1e6:.2f}MHz kernel_buffers={cfg.kernel_buffers}")
    print(f"stf_repeats={cfg.stf_repeats} ltf_symbols={cfg.ltf_symbols} repeat={cfg.repeat}")
    print(f"use_tone_anchor={cfg.use_tone_anchor} tone={cfg.tone_hz/1e3:.1f}kHz {cfg.tone_ms:.1f}ms")
    print("="*78)

    import adi

    # SDR
    sdr = adi.Pluto(uri=cfg.uri)
    sdr.sample_rate = int(cfg.fs)
    sdr.rx_lo = int(cfg.fc)
    sdr.rx_rf_bandwidth = int(cfg.rx_bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(cfg.rx_gain)
    sdr.rx_buffer_size = int(cfg.buf_size)
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

    csv_path = os.path.join(run_dir, "captures.csv")
    csv_f = open(csv_path, "w", newline="")
    fieldnames = [
        "cap","status","reason",
        "peak","sc_peak","sc_ratio","sc_idx","cfo_hz",
        "tone_idx","tone_ratio",
        "xc_idx","xc_peak","stf_idx","stf_method",
        "ltf_start","ltf_q",
        "payload_start",
        "probe_mean_evm",
        "probe_head_hex",
        "parse_status",
    ]
    w = csv.DictWriter(csv_f, fieldnames=fieldnames)
    w.writeheader()

    t0 = time.time()
    success_any = False

    try:
        for cap in range(1, cfg.tries + 1):
            rx_raw = sdr.rx()
            rx = autoscale_and_dc_remove(rx_raw)

            peak = float(np.max(np.abs(rx))) if rx.size else 0.0

            # Tone anchor (optional)
            tone_idx = -1
            tone_ratio = 0.0
            tone_curve = np.array([], dtype=np.float32)
            tone_curve_idx = np.array([], dtype=np.int32)

            if cfg.use_tone_anchor and cfg.tone_ms > 0:
                ti, tp, tr, tv, tix = detect_tone_matched_filter(
                    rx, cfg.fs, cfg.tone_hz, win_len=cfg.tone_win, hop=cfg.tone_hop
                )
                tone_idx, tone_ratio = ti, tr
                if cfg.save_curves:
                    tone_curve, tone_curve_idx = tv, tix

            # SC detection (always compute for debug)
            sc_ok, sc_idx, sc_peak, sc_cfo, sc_ratio, sc_M = detect_stf_sc(
                rx, cfg.fs,
                sc_threshold=cfg.sc_threshold,
                peak_med_ratio_th=cfg.sc_ratio_th,
                search_len=cfg.search_len,
                half_period=N_FFT//2
            )

            # Choose CFO for this capture: SC CFO (debug focus)
            cfo_hz = float(sc_cfo)
            rx_cfo = apply_cfo(rx, cfo_hz, cfg.fs)

            # Global xcorr for debug
            xc_idx, xc_peak, xc_curve = detect_stf_xcorr(rx_cfo, stf_ref, search_len=cfg.search_len)

            # Decide STF search window / refine
            stf_method = "sc"
            stf_idx = sc_idx

            if cfg.use_tone_anchor and cfg.tone_ms > 0 and tone_idx >= 0 and tone_ratio >= cfg.tone_ratio_th:
                tone_samples = int(cfg.tone_ms * cfg.fs / 1000.0)
                stf_expect = tone_idx + tone_samples + cfg.gap_short
                stf_method = "tone_anchor"
                stf_idx = int(stf_expect)

                # local refine by xcorr in a window
                s0 = max(0, stf_idx - cfg.stf_refine_win)
                s1 = min(len(rx_cfo) - len(stf_ref), stf_idx + cfg.stf_refine_win)
                if s1 > s0:
                    seg = rx_cfo[s0:s1 + len(stf_ref)]
                    lc = np.abs(np.correlate(seg, stf_ref, mode="valid"))
                    lc = lc / (np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
                    loc = int(np.argmax(lc))
                    stf_idx = int(s0 + loc)
                    stf_method = "tone_anchor+xc_refine"
            else:
                # If SC not confident, fall back to global xcorr peak
                if (not sc_ok) and (xc_peak > 0.02):
                    stf_idx = xc_idx
                    stf_method = "xc_fallback"
                elif sc_ok:
                    # If SC ok, optionally refine locally around sc_idx by xcorr
                    s0 = max(0, sc_idx - cfg.stf_refine_win)
                    s1 = min(len(rx_cfo) - len(stf_ref), sc_idx + cfg.stf_refine_win)
                    if s1 > s0:
                        seg = rx_cfo[s0:s1 + len(stf_ref)]
                        lc = np.abs(np.correlate(seg, stf_ref, mode="valid"))
                        lc = lc / (np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
                        loc = int(np.argmax(lc))
                        stf_idx = int(s0 + loc)
                        stf_method = "sc+xc_refine"

            # LTF alignment sweep (±ltf_off_sweep)
            ltf_base = stf_idx + len(stf_ref)
            best_q = -1.0
            best_off = 0
            best_H = None
            best_snr = None
            for off in range(-cfg.ltf_off_sweep, cfg.ltf_off_sweep + 1):
                H, snr_sc, q = channel_estimate_from_ltf(rx_cfo, ltf_base + off, ltf_freq_ref, num_symbols=cfg.ltf_symbols)
                if H is None:
                    continue
                if q > best_q:
                    best_q = q
                    best_off = off
                    best_H = H
                    best_snr = snr_sc

            status = "ok"
            reason = ""
            parse_status = ""
            probe_head_hex = ""
            probe_mean_evm = ""

            arrays_to_save = {}
            meta = {
                "cap": cap,
                "uri": cfg.uri,
                "fc": cfg.fc,
                "fs": cfg.fs,
                "rx_gain": cfg.rx_gain,
            }

            if best_H is None:
                status = "ltf_fail"
                reason = "no_valid_ltf"
            else:
                ltf_start = int(ltf_base + best_off)
                payload_start = int(ltf_start + cfg.ltf_symbols * SYMBOL_LEN)

                # probe demod
                probe = demod_probe(rx_cfo, payload_start, best_H, cfg.probe_syms, cfg.kp, cfg.ki)
                if probe is None:
                    status = "no_payload"
                    reason = "probe_failed"
                else:
                    bits_raw = qpsk_demap(probe["data_syms"])
                    bits = majority_vote(bits_raw, cfg.repeat)

                    # try bit-slip parse
                    ok_pkt = False
                    best_parse = ("bad_magic", None, None, None, None)
                    for slip in range(cfg.bit_slip_max + 1):
                        bb = bits_to_bytes(bits[slip:])
                        ok, why, payload, crc_rx, crc_calc = parse_packet(bb)
                        if ok:
                            ok_pkt = True
                            best_parse = ("ok", payload, crc_rx, crc_calc, slip)
                            break
                        # keep the "closest" info: if bad_magic, store its first 16 bytes
                        best_parse = (why, payload, crc_rx, crc_calc, slip)

                    evm = probe["evm_per_sym"]
                    probe_mean_evm = float(np.mean(evm)) if len(evm) else ""
                    head = bits_to_bytes(bits[:128])  # first 16 bytes
                    probe_head_hex = head[:16].hex()

                    parse_status = best_parse[0]
                    if ok_pkt:
                        status = "crc_ok"
                        success_any = True
                        reason = f"slip={best_parse[4]}"
                    else:
                        status = "no_crc"
                        reason = best_parse[0]

                    # arrays for debug
                    arrays_to_save.update({
                        "H": best_H.astype(np.complex64),
                        "snr_sc": (best_snr if best_snr is not None else np.array([], dtype=np.float32)).astype(np.float32),
                        "probe_syms": probe["data_syms"].astype(np.complex64),
                        "probe_evm": probe["evm_per_sym"].astype(np.float32),
                        "probe_phase": probe["pilot_phase"].astype(np.float32),
                        "probe_freq": probe["freq_acc"].astype(np.float32),
                    })

                    meta.update({
                        "ltf_start": ltf_start,
                        "payload_start": payload_start,
                        "ltf_q": float(best_q),
                        "probe_mean_evm": probe_mean_evm,
                    })

            # always save core meta
            meta.update({
                "status": status,
                "reason": reason,
                "peak": peak,
                "sc_idx": sc_idx,
                "sc_peak": float(sc_peak),
                "sc_ratio": float(sc_ratio),
                "cfo_hz": cfo_hz,
                "tone_idx": tone_idx,
                "tone_ratio": float(tone_ratio),
                "xc_idx": xc_idx,
                "xc_peak": float(xc_peak),
                "stf_idx": int(stf_idx),
                "stf_method": stf_method,
                "ltf_off": int(best_off),
                "parse_status": parse_status,
                "probe_head_hex": probe_head_hex,
            })

            # save arrays requested
            if cfg.save_rx:
                arrays_to_save["rx"] = rx.astype(np.complex64)
            if cfg.save_rx_cfo:
                arrays_to_save["rx_cfo"] = rx_cfo.astype(np.complex64)
            if cfg.save_curves:
                arrays_to_save["sc_metric"] = sc_M.astype(np.float32)
                arrays_to_save["xcorr_curve"] = xc_curve.astype(np.float32)
                if cfg.use_tone_anchor:
                    arrays_to_save["tone_curve"] = tone_curve.astype(np.float32)
                    arrays_to_save["tone_curve_idx"] = tone_curve_idx.astype(np.int32)

            # write NPZ + plot
            npz_path = os.path.join(run_dir, f"cap_{cap:04d}.npz")
            np.savez_compressed(npz_path, **arrays_to_save, meta_json=np.bytes_(json.dumps(meta).encode("utf-8")))

            if cfg.save_plots and (cap % cfg.plot_every == 0):
                png_path = os.path.join(run_dir, f"cap_{cap:04d}.png")
                # for plotting we want rx + sc_metric + probe fields if present
                plot_arrays = {
                    "rx": arrays_to_save.get("rx", rx),
                    "sc_metric": arrays_to_save.get("sc_metric", sc_M),
                    "probe_syms": arrays_to_save.get("probe_syms", np.array([], dtype=np.complex64)),
                    "probe_evm": arrays_to_save.get("probe_evm", np.array([], dtype=np.float32)),
                    "probe_phase": arrays_to_save.get("probe_phase", np.array([], dtype=np.float32)),
                }
                plot_capture(png_path, {"fs": cfg.fs, **meta}, plot_arrays)

            # CSV row
            row = {
                "cap": cap,
                "status": status,
                "reason": reason,
                "peak": f"{peak:.2f}",
                "sc_peak": f"{sc_peak:.4f}",
                "sc_ratio": f"{sc_ratio:.2f}",
                "sc_idx": sc_idx,
                "cfo_hz": f"{cfo_hz:+.1f}",
                "tone_idx": tone_idx,
                "tone_ratio": f"{tone_ratio:.2f}",
                "xc_idx": xc_idx,
                "xc_peak": f"{xc_peak:.4f}",
                "stf_idx": stf_idx,
                "stf_method": stf_method,
                "ltf_start": meta.get("ltf_start", ""),
                "ltf_q": f"{meta.get('ltf_q', 0.0):.2f}" if "ltf_q" in meta else "",
                "payload_start": meta.get("payload_start", ""),
                "probe_mean_evm": f"{probe_mean_evm:.4f}" if isinstance(probe_mean_evm, (float,int)) else probe_mean_evm,
                "probe_head_hex": probe_head_hex,
                "parse_status": parse_status,
            }
            w.writerow(row)

            # console
            head = probe_head_hex
            print(f"[{cap:04d}] {status:7s} CFO={cfo_hz:+8.1f}Hz "
                  f"scR={sc_ratio:5.2f} scP={sc_peak:6.3f} "
                  f"toneR={tone_ratio:5.2f} xcP={xc_peak:6.3f} "
                  f"stf={stf_idx:6d} off={best_off:3d} ltfQ={best_q:7.2f} "
                  f"EVM={probe_mean_evm if probe_mean_evm!='' else '':>6} "
                  f"head={head} {reason}")

    finally:
        csv_f.close()
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

        elapsed = time.time() - t0
        print("\n" + "="*78)
        print("DONE")
        print("="*78)
        print(f"run_dir: {run_dir}")
        print(f"csv: {csv_path}")
        print(f"elapsed: {elapsed:.1f}s")
        print(f"success_any: {success_any}")
        print("="*78)

if __name__ == "__main__":
    main()

"""
python3 rf_step5_rx_debug_capture.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --rx_gain 30 --rx_bw 3.6e6 --buf_size 262144 \
  --tries 80 \
  --save_rx --save_curves --save_plots --plot_every 1

#then open tone anchor
python3 rf_step5_rx_debug_capture.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --rx_gain 30 --rx_bw 3.6e6 --buf_size 262144 \
  --tries 80 \
  --use_tone_anchor --tone_ms 10 --tone_hz 100e3 \
  --save_rx --save_curves --save_plots --plot_every 1

captures.csv
"""