#!/usr/bin/env python3
"""
rf_link_step7_rx_fixed_v2.py
RF Link Test - Step 7: RX (Video Reception) - FIXED v2 + logging + plots

What this version fixes (vs. earlier Step7 RX variants):
1) **Consistent FFT bin indexing**
   - Use UN-SHIFTED FFT for OFDM symbols (no fftshift).
   - Map subcarrier k in [-N/2..N/2-1] to bin (k+N)%N.

2) **Robust STF sync + CFO**
   - Schmidl-Cox metric computed in O(N) via moving sums (fast, stable).
   - CFO estimated from STF repetition phase:
        angle(P) = -2π * CFO * L/fs  ->  CFO = -angle(P)*fs/(2πL)
   - Cross-correlation refinement uses STF reference WITHOUT CP to avoid CP peak ambiguity.

3) **Tail-contamination prevention**
   - First pass: probe a small number of OFDM symbols.
   - Bit-slip search (0..7) to locate MAGIC + header even if decode is byte-misaligned.
   - Once header is found, compute required symbol count and demod ONLY that many symbols.

4) **PLL/FLL stability**
   - Pilot-based PLL/FLL with:
       - phase unwrap
       - dphi clipping to avoid integrator blow-up
       - freeze integrator on very bad EVM symbols

5) **Logging + plots**
   - Writes CSV capture log: captures.csv
   - Saves NPZ session arrays: session.npz
   - Per-frame diagnostic plots (optional): frame_XXXX.png
   - Session summary plot: video_summary.png

Packet format (must match TX):
  MAGIC("VID7",4B) | FRAME_ID(2B) | SEQ(2B) | TOTAL(2B) | LEN(2B) | PAYLOAD | CRC32(4B)
  CRC covers MAGIC..PAYLOAD.

Example:
  python rf_link_step7_rx_fixed_v2.py \
    --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
    --rx_gain 60 --buf_size 65536 \
    --stf_repeats 6 --ltf_symbols 4 \
    --repeat 1 \
    --sync_ratio 6 \
    --kp 0.15 --ki 0.005 \
    --max_syms_probe 40 \
    --plot_every_frame 1 \
    --verbose
"""

import argparse
import os
import time
import zlib
import csv
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
N_DATA = len(DATA_SUBCARRIERS)          # 48
BITS_PER_QPSK_SYM = 2
BITS_PER_OFDM_SYM = N_DATA * BITS_PER_QPSK_SYM  # 96

def sc_to_bin(k: int) -> int:
    """Subcarrier k in [-32..31] -> unshifted FFT bin index."""
    return (k + N_FFT) % N_FFT

PILOT_BINS = np.array([sc_to_bin(k) for k in PILOT_SUBCARRIERS], dtype=int)
DATA_BINS  = np.array([sc_to_bin(k) for k in DATA_SUBCARRIERS], dtype=int)

IDEAL_QPSK = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)
PILOT_PATTERN = np.array([1, 1, 1, -1], dtype=np.complex64)

# ==============================
# Bit/packet helpers
# ==============================
def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    """Hard QPSK demap matching TX mapping: (00:+,+), (01:-,+), (11:-,-), (10:+,-)."""
    bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
    for i, s in enumerate(symbols):
        rr = (np.real(s) >= 0)
        ii = (np.imag(s) >= 0)
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
    bits = bits[:L].reshape(-1, repeat)
    return (np.sum(bits, axis=1) >= (repeat / 2)).astype(np.uint8)

def packbits_with_slip(bits: np.ndarray, slip: int) -> bytes:
    """Drop first slip bits (0..7), then pack to bytes."""
    if slip > 0:
        if len(bits) <= slip:
            return b""
        bits = bits[slip:]
    L = (len(bits) // 8) * 8
    if L <= 0:
        return b""
    return np.packbits(bits[:L]).tobytes()

def try_parse_header_only(data: bytes):
    """
    Header-only parsing:
      MAGIC(4) | FRAME_ID(2) | SEQ(2) | TOTAL(2) | LEN(2)
    Returns (ok, frame_id, seq, total, plen, expected_total_len_bytes)
    expected_total_len_bytes = 12 + plen + 4
    """
    if len(data) < 12:
        return False, -1, -1, 0, 0, 0
    if data[:4] != MAGIC:
        return False, -1, -1, 0, 0, 0
    frame_id = int.from_bytes(data[4:6], "little")
    seq      = int.from_bytes(data[6:8], "little")
    total    = int.from_bytes(data[8:10], "little")
    plen     = int.from_bytes(data[10:12], "little")
    expected = 12 + plen + 4
    return True, frame_id, seq, total, plen, expected

def parse_video_packet_full(data: bytes):
    """
    Full parse + CRC.
    Returns: (ok, frame_id, seq, total, payload, expected_len)
    """
    okh, fid, seq, total, plen, expected = try_parse_header_only(data)
    if not okh:
        return False, -1, -1, 0, b"", 0
    if len(data) < expected:
        return False, fid, seq, total, b"", expected
    content = data[:12 + plen]
    payload = data[12:12 + plen]
    crc_rx  = int.from_bytes(data[12 + plen:12 + plen + 4], "little")
    crc_calc = zlib.crc32(content) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return False, fid, seq, total, b"", expected
    return True, fid, seq, total, payload, expected

# ==============================
# Training sequences (match TX)
# ==============================
def create_schmidl_cox_stf_payload(num_repeats: int):
    """
    STF payload (NO CP): repeated time-domain chunk length N_FFT.
    Use even subcarriers with BPSK.
    """
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i] + 0j
    x = np.fft.ifft(X) * np.sqrt(N_FFT)
    stf_payload = np.tile(x, num_repeats).astype(np.complex64)
    return stf_payload, X

def create_ltf_ref(num_symbols: int):
    """LTF reference in frequency domain + time-domain LTF (cp+fft) repeated."""
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(X) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf_td = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf_td, X

# ==============================
# CFO / sync helpers
# ==============================
def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(len(samples), dtype=np.float32)
    return (samples * np.exp(-1j * 2 * np.pi * (cfo_hz / fs) * n)).astype(np.complex64)

def schmidl_cox_fast(rx: np.ndarray, L: int, search_len: int):
    """
    Fast Schmidl-Cox metric using moving sums (O(N)):
      P[n] = sum_{m=0..L-1} rx[n+m] * conj(rx[n+m+L])
      R[n] = sum_{m=0..L-1} |rx[n+m+L]|^2
    """
    N = len(rx)
    max_n = min(search_len, N - 2*L)
    if max_n <= 8:
        return None, None, None

    a = rx[:max_n + L]              # covers n..n+L-1
    b = rx[L:L + max_n + L]         # covers n+L..n+2L-1
    s = a * np.conj(b)              # length max_n+L

    # moving sum of s over window L => P[n]
    cs = np.cumsum(s, dtype=np.complex64)
    cs = np.concatenate([np.zeros(1, dtype=np.complex64), cs])
    P = cs[L:] - cs[:-L]            # length max_n+1

    # moving sum of |b|^2 over window L => R[n]
    pb = np.abs(b)**2
    cb = np.cumsum(pb, dtype=np.float32)
    cb = np.concatenate([np.zeros(1, dtype=np.float32), cb])
    R = cb[L:] - cb[:-L]            # length max_n+1

    # Use first max_n entries (consistent)
    P = P[:max_n]
    R = R[:max_n]
    M = (np.abs(P)**2) / (R**2 + 1e-12)
    return M, P, R

def detect_stf_autocorr(rx: np.ndarray, fs: float, half_period: int, threshold: float,
                        ratio_th: float, search_len: int):
    """
    Returns: ok, idx, peak_smooth, ratio(peak/median), cfo_hz, M_raw, M_smooth
    """
    L = half_period
    M, P, _ = schmidl_cox_fast(rx, L=L, search_len=search_len)
    if M is None:
        return False, 0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    plateau = 2 * L
    if len(M) < plateau + 8:
        return False, 0, float(np.max(M)), float(np.max(M)/(np.median(M)+1e-12)), 0.0, M, M

    kernel = np.ones(plateau, dtype=np.float32) / plateau
    M_s = np.convolve(M, kernel, mode="valid")
    pk = int(np.argmax(M_s))
    peak_s = float(M_s[pk])
    med_s = float(np.median(M_s) + 1e-12)
    ratio = peak_s / med_s

    # refine idx: walk back from pk in raw M until it drops
    idx = pk
    for i in range(pk, max(0, pk - 5*L), -1):
        if M[i] < 0.1 * peak_s:
            idx = i + 1
            break

    # CFO from sum of P across plateau
    p_sum = np.sum(P[idx: idx + plateau])
    cfo_hz = -np.angle(p_sum) * (fs / (2*np.pi*L))

    ok = (peak_s > threshold) and (ratio >= ratio_th)
    return ok, idx, peak_s, float(ratio), float(cfo_hz), M, M_s

def crosscorr_refine(rx: np.ndarray, ref: np.ndarray, center_idx: int, win: int):
    """Refine correlation peak near center_idx in [center-win, center+win]."""
    L = len(ref)
    s0 = max(0, center_idx - win)
    s1 = min(len(rx), center_idx + win + L)
    if s1 - s0 <= L + 8:
        return -1, 0.0
    seg = rx[s0:s1]
    corr = np.abs(np.correlate(seg, ref, mode="valid"))
    denom = np.sqrt(np.sum(np.abs(ref)**2)) + 1e-12
    corr_n = corr / denom
    loc = int(np.argmax(corr_n))
    return s0 + loc, float(corr_n[loc])

# ==============================
# OFDM helpers
# ==============================
def extract_ofdm_symbol_freq(rx: np.ndarray, start_idx: int):
    """start_idx points to CP start; returns UN-SHIFTED FFT bins."""
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    td = rx[start_idx + N_CP : start_idx + SYMBOL_LEN]
    return np.fft.fft(td) / np.sqrt(N_FFT)

def channel_estimate_from_ltf(rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray, num_symbols: int):
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    used_bins = np.array([sc_to_bin(int(k)) for k in used], dtype=int)

    Ys = []
    for i in range(num_symbols):
        Y = extract_ofdm_symbol_freq(rx, ltf_start + i*SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if not Ys:
        return None, None

    Yavg = np.mean(np.stack(Ys, axis=0), axis=0)
    H = np.ones(N_FFT, dtype=np.complex64)
    for k in used:
        b = sc_to_bin(int(k))
        if np.abs(ltf_freq_ref[b]) > 1e-3:
            H[b] = Yavg[b] / ltf_freq_ref[b]

    # Rough SNR-per-sc estimate from variance across LTF symbols
    if len(Ys) >= 2:
        Ystk = np.stack(Ys, axis=0)[:, used_bins]
        noise_var = np.var(Ystk, axis=0)
        sig_var = np.abs(np.mean(Ystk, axis=0))**2
        snr_sc = 10*np.log10(sig_var / (noise_var + 1e-12) + 1e-12)
    else:
        snr_sc = np.zeros(len(used), dtype=np.float32)

    return H, snr_sc

def wrap_to_pi(x: float) -> float:
    while x > np.pi:
        x -= 2*np.pi
    while x < -np.pi:
        x += 2*np.pi
    return x

def demod_payload_symbols(rx: np.ndarray,
                         payload_start: int,
                         H: np.ndarray,
                         num_syms: int,
                         kp: float,
                         ki: float,
                         repeat: int,
                         dphi_clip: float = 0.35,
                         freeze_evm: float = 1.20):
    """
    Demodulate num_syms OFDM payload symbols -> logical bits (after majority vote).
    Returns (bits, diag_dict). If no symbols, returns (None, diag_dict).
    """
    phase_acc = 0.0
    freq_acc = 0.0
    prev_phase = None

    all_data_syms = []
    evm_per_sym = []
    phase_errs = []
    freq_log = []

    used_bins = np.array([sc_to_bin(k) for k in range(-26, 27) if k != 0], dtype=int)

    for si in range(num_syms):
        Y = extract_ofdm_symbol_freq(rx, payload_start + si*SYMBOL_LEN)
        if Y is None:
            break

        # Equalize used bins
        Yeq = Y.copy()
        Yeq[used_bins] = Yeq[used_bins] / (H[used_bins] + 1e-12)

        # Pilot phase error
        pilot_sign = 1 if (si % 2 == 0) else -1
        exp_p = pilot_sign * PILOT_PATTERN
        rp = Yeq[PILOT_BINS]
        pilot_corr = np.sum(rp * np.conj(exp_p))
        ph = float(np.angle(pilot_corr))

        # FLL (freq from phase diff) + PLL (phase)
        if prev_phase is not None:
            dph = wrap_to_pi(ph - prev_phase)
            dph = float(np.clip(dph, -dphi_clip, dphi_clip))
            # Freeze integrator if symbol is too bad later (after we compute EVM)
            f_update = ki * dph
        else:
            f_update = 0.0

        # Apply current correction first (using prev integrator value)
        phase_acc_tmp = phase_acc + freq_acc + kp * ph
        Yeq2 = Yeq * np.exp(-1j * phase_acc_tmp).astype(np.complex64)

        data_syms = Yeq2[DATA_BINS]
        nearest = IDEAL_QPSK[np.argmin(np.abs(data_syms[:, None] - IDEAL_QPSK[None, :]), axis=1)]
        evm = float(np.sqrt(np.mean(np.abs(data_syms - nearest)**2)))

        # Update integrators (freeze if EVM too large)
        if prev_phase is not None and evm < freeze_evm:
            freq_acc += f_update
        # else: freeze freq_acc

        phase_acc = phase_acc_tmp
        prev_phase = ph

        all_data_syms.append(data_syms)
        evm_per_sym.append(evm)
        phase_errs.append(ph)
        freq_log.append(freq_acc)

    if not all_data_syms:
        return None, {
            "num_syms": 0,
            "all_data_syms": np.array([], dtype=np.complex64),
            "evm_per_sym": np.array([], dtype=np.float32),
            "phase_err": np.array([], dtype=np.float32),
            "freq_log": np.array([], dtype=np.float32),
        }

    all_data_syms = np.concatenate(all_data_syms, axis=0)
    bits_raw = qpsk_demap(all_data_syms)
    bits = majority_vote(bits_raw, repeat)

    return bits, {
        "num_syms": len(evm_per_sym),
        "all_data_syms": all_data_syms,
        "evm_per_sym": np.array(evm_per_sym, dtype=np.float32),
        "phase_err": np.array(phase_errs, dtype=np.float32),
        "freq_log": np.array(freq_log, dtype=np.float32),
    }

# ==============================
# Frame accumulator
# ==============================
class FrameAccumulator:
    """Track packets for multiple frames; handle timeouts and reassembly."""
    def __init__(self, max_inflight=10, timeout=60.0):
        self.frames = {}     # fid -> entry
        self.completed = set()
        self.max_inflight = max_inflight
        self.timeout = timeout

    def add_packet(self, frame_id, seq, total, payload, evm=0.0, cfo=0.0, data_syms=None):
        if frame_id in self.completed:
            return False, True  # (is_new, complete)

        now = time.time()
        if frame_id not in self.frames:
            self.frames[frame_id] = {
                "pkts": {},
                "total": int(total),
                "ts_first": now,
                "ts_last": now,
                "evm_sum": 0.0,
                "evm_cnt": 0,
                "cfo_list": [],
                "pkt_meta": {}  # seq -> {evm, data_syms}
            }
        e = self.frames[frame_id]
        e["ts_last"] = now

        is_new = (seq not in e["pkts"])
        if is_new:
            e["pkts"][seq] = payload
            e["evm_sum"] += float(evm)
            e["evm_cnt"] += 1
            e["cfo_list"].append(float(cfo))
            e["pkt_meta"][seq] = {
                "evm": float(evm),
                "data_syms": data_syms.copy() if isinstance(data_syms, np.ndarray) else np.array([], dtype=np.complex64),
            }

        complete = (len(e["pkts"]) >= e["total"])
        return is_new, complete

    def get_frame_bytes(self, frame_id):
        if frame_id not in self.frames:
            return None
        e = self.frames[frame_id]
        out = b""
        for i in range(e["total"]):
            if i not in e["pkts"]:
                return None
            out += e["pkts"][i]
        return out

    def get_stats(self, frame_id):
        if frame_id not in self.frames:
            return {}
        e = self.frames[frame_id]
        r = len(e["pkts"])
        t = e["total"]
        return {
            "received": r,
            "total": t,
            "completion": r / t if t else 0,
            "mean_evm": e["evm_sum"] / e["evm_cnt"] if e["evm_cnt"] else 0.0,
            "mean_cfo": float(np.mean(e["cfo_list"])) if e["cfo_list"] else 0.0,
            "latency": e["ts_last"] - e["ts_first"],
        }

    def mark_completed(self, frame_id):
        self.completed.add(frame_id)
        if frame_id in self.frames:
            del self.frames[frame_id]

    def expire_old(self):
        now = time.time()
        expired = []
        for fid in list(self.frames.keys()):
            if now - self.frames[fid]["ts_last"] > self.timeout:
                expired.append((fid, self.get_stats(fid)))
                del self.frames[fid]

        while len(self.frames) > self.max_inflight:
            oldest = min(self.frames.keys(), key=lambda k: self.frames[k]["ts_last"])
            expired.append((oldest, self.get_stats(oldest)))
            del self.frames[oldest]
        return expired

    def active_summary(self):
        return {fid: f'{len(e["pkts"])}/{e["total"]}' for fid, e in self.frames.items()}

# ==============================
# Diagnostics plots
# ==============================
def plot_frame_diagnostics(frame_id, frame_img_bgr, pkt_meta, total_pkts, out_dir):
    """2x2 per-frame diagnostics plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) image
    ax = axes[0, 0]
    if frame_img_bgr is not None:
        ax.imshow(frame_img_bgr[:, :, ::-1])
        ax.set_title(f"Frame {frame_id} (decoded)")
    else:
        ax.text(0.5, 0.5, "Decode Failed", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="red")
        ax.set_title(f"Frame {frame_id} (FAILED)")
    ax.axis("off")

    # (0,1) packet map
    ax = axes[0, 1]
    colors = ["green" if i in pkt_meta else "red" for i in range(total_pkts)]
    ax.barh(range(total_pkts), [1]*total_pkts, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlim([0, 1.2])
    ax.set_xlabel("Received")
    ax.set_ylabel("Packet Seq")
    ax.set_title(f"Packets: {len(pkt_meta)}/{total_pkts}")

    # (1,0) EVM per packet
    ax = axes[1, 0]
    if pkt_meta:
        seqs = sorted(pkt_meta.keys())
        evms = [pkt_meta[s].get("evm", 0.0) for s in seqs]
        ax.bar(seqs, evms, alpha=0.85)
        ax.axhline(np.mean(evms), linestyle="--", label=f"Mean={np.mean(evms):.4f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Packet Seq")
    ax.set_ylabel("EVM")
    ax.set_title("EVM per Packet")
    ax.grid(True)

    # (1,1) constellation
    ax = axes[1, 1]
    all_syms = []
    for s in sorted(pkt_meta.keys()):
        syms = pkt_meta[s].get("data_syms", np.array([], dtype=np.complex64))
        if syms.size:
            all_syms.append(syms)
    if all_syms:
        z = np.concatenate(all_syms)
        n = len(z)
        ax.scatter(np.real(z), np.imag(z), c=np.arange(n), cmap="viridis", s=2, alpha=0.5)
        ax.scatter(np.real(IDEAL_QPSK), np.imag(IDEAL_QPSK), c="red", s=80, marker="x", linewidths=2)
    ax.axhline(0, color="k", linewidth=0.3)
    ax.axvline(0, color="k", linewidth=0.3)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("QPSK Constellation (all packets)")
    ax.grid(True)

    fig.suptitle(f"RF Link Step7 RX - Frame {frame_id}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"frame_{frame_id:04d}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

def plot_video_summary(all_frame_stats, capture_arrays, out_png, start_time, end_time):
    """3x3 session summary plot."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    elapsed = end_time - start_time

    cap_ok   = np.array(capture_arrays["cap_ok"], dtype=float)
    cap_cfo  = np.array(capture_arrays["cap_cfo"], dtype=float)
    cap_evm  = np.array(capture_arrays["cap_evm"], dtype=float)
    cap_ratio = np.array(capture_arrays["cap_ratio"], dtype=float)
    cap_st = capture_arrays["cap_status"]

    completed = sorted([k for k,v in all_frame_stats.items() if v.get("completion",0) >= 1.0])
    times = [all_frame_stats[f]["complete_time"] - start_time for f in completed] if completed else []

    # (0,0) frames completion step
    ax = axes[0,0]
    if times:
        ax.step(times, range(1, len(times)+1), where="post")
        fps = len(times)/elapsed if elapsed>0 else 0
        ax.set_title(f"Frame Completion (FPS={fps:.3f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frames Completed")
    ax.grid(True)

    # (0,1) PER over captures (moving window)
    ax = axes[0,1]
    if len(cap_ok) > 0:
        win = min(50, len(cap_ok))
        if win >= 5:
            per = 1.0 - np.convolve(cap_ok, np.ones(win)/win, mode="valid")
            ax.plot(per)
            ax.set_title(f"PER (window={win})")
        else:
            ax.axhline(1.0 - cap_ok.mean())
            ax.set_title("PER")
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Capture")
    ax.set_ylabel("PER")
    ax.grid(True)

    # (0,2) sync ratio
    ax = axes[0,2]
    if len(cap_ratio) > 0:
        ax.plot(cap_ratio, ".", markersize=2, alpha=0.6)
        ax.axhline(np.median(cap_ratio), linestyle="--", alpha=0.6)
    ax.set_title("Schmidl-Cox Peak/Median Ratio")
    ax.set_xlabel("Capture")
    ax.set_ylabel("Ratio")
    ax.grid(True)

    # (1,0) CFO over captures (only for ok captures)
    ax = axes[1,0]
    ok_idx = np.where(cap_ok > 0.5)[0]
    if ok_idx.size:
        ax.plot(ok_idx, cap_cfo[ok_idx], ".", markersize=2, alpha=0.6)
        ax.axhline(np.mean(cap_cfo[ok_idx]), linestyle="--", alpha=0.6, label=f"Mean={np.mean(cap_cfo[ok_idx]):.1f}Hz")
        ax.legend(fontsize=8)
        ax.set_title(f"CFO (std={np.std(cap_cfo[ok_idx]):.1f}Hz)")
    ax.set_xlabel("Capture")
    ax.set_ylabel("CFO (Hz)")
    ax.grid(True)

    # (1,1) EVM histogram (ok captures)
    ax = axes[1,1]
    if ok_idx.size:
        ev = cap_evm[ok_idx]
        ev = ev[np.isfinite(ev)]
        if ev.size:
            ax.hist(ev, bins=30, edgecolor="black", alpha=0.85)
            ax.axvline(np.mean(ev), linestyle="--", label=f"Mean={np.mean(ev):.4f}")
            ax.legend(fontsize=8)
    ax.set_title("Mean EVM (per packet)")
    ax.set_xlabel("EVM")
    ax.set_ylabel("Count")
    ax.grid(True)

    # (1,2) throughput (cumulative jpeg size)
    ax = axes[1,2]
    if completed:
        t = np.array(times)
        sz = np.array([all_frame_stats[f]["jpeg_size"] for f in completed], dtype=float)
        cum_kb = np.cumsum(sz)/1024.0
        ax.plot(t, cum_kb, "-o", markersize=3)
        thr = (np.sum(sz)/elapsed) if elapsed>0 else 0
        ax.set_title(f"Cumulative (KB), Throughput={thr:.0f} B/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative KB")
    ax.grid(True)

    # (2,0) frame latencies
    ax = axes[2,0]
    if completed:
        lat = [all_frame_stats[f].get("latency",0) for f in completed]
        ax.bar(range(len(lat)), lat, alpha=0.85)
        ax.axhline(np.mean(lat), linestyle="--", label=f"Mean={np.mean(lat):.2f}s")
        ax.legend(fontsize=8)
    ax.set_title("Per-frame latency")
    ax.set_xlabel("Frame order")
    ax.set_ylabel("Latency (s)")
    ax.grid(True)

    # (2,1) completion %
    ax = axes[2,1]
    if all_frame_stats:
        fids = sorted(all_frame_stats.keys())
        comp = [all_frame_stats[f].get("completion",0)*100 for f in fids]
        ax.bar(range(len(comp)), comp, alpha=0.85)
        ax.axhline(100, linestyle=":", alpha=0.6)
    ax.set_title("Frame completion %")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Completion (%)")
    ax.set_ylim([0, 110])
    ax.grid(True)

    # (2,2) summary text
    ax = axes[2,2]
    ax.axis("off")
    n_caps = len(cap_ok)
    n_ok = int(np.sum(cap_ok))
    # packet-level PER estimate (ok vs crc_fail among demod attempts)
    n_crc = capture_arrays["crc_fail"]
    n_nosig = capture_arrays["no_signal"]
    n_dup = capture_arrays["dup"]
    n_exp = capture_arrays["frames_expired"]
    n_frames_ok = capture_arrays["frames_ok"]

    per = 1.0 - (n_ok / max(1, (n_ok + n_crc)))
    fer = 1.0 - (n_frames_ok / max(1, (n_frames_ok + n_exp)))

    cfo_std = float(np.std(cap_cfo[ok_idx])) if ok_idx.size else float("nan")
    evm_mean = float(np.mean(cap_evm[ok_idx])) if ok_idx.size else float("nan")
    fps = (n_frames_ok / elapsed) if elapsed>0 else 0.0

    summary = f"""
VIDEO RX SUMMARY

Captures: {n_caps}
  Packets OK: {n_ok}
  CRC/NoPacket: {n_crc}
  No signal/LTF: {n_nosig}
  Duplicates: {n_dup}
  PER: {per:.3f}

Frames completed: {n_frames_ok}
Frames expired:   {n_exp}
FER: {fer:.3f}

Elapsed: {elapsed:.1f}s
Effective FPS: {fps:.4f}

Mean EVM (ok): {evm_mean:.4f}
CFO std (ok):  {cfo_std:.1f} Hz
"""
    ax.text(0.05, 0.95, summary, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6), fontsize=11)

    fig.suptitle("RF Link Step7 RX - Session Summary", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png

# ==============================
# Core demod for one capture
# ==============================
def demod_one_capture(rx_raw_iq: np.ndarray,
                      fs: float,
                      stf_payload_ref: np.ndarray,
                      ltf_freq_ref: np.ndarray,
                      stf_repeats: int,
                      ltf_symbols: int,
                      kp: float,
                      ki: float,
                      repeat: int,
                      max_syms_probe: int,
                      sync_ratio_th: float,
                      verbose: bool = False):
    """
    Returns dict:
      status: demod_ok / no_signal / ltf_fail / no_packet / crc_fail
      cfo_hz, sync_ratio, stf_idx, stf_peak
      if demod_ok: frame_id, seq, total, payload, mean_evm, constellation
    """
    rx = rx_raw_iq.astype(np.complex64)

    # scale heuristic: Pluto often returns ~int16-range complex
    if np.median(np.abs(rx)) > 100:
        rx = rx / (2**14)

    rx = rx - np.mean(rx)

    # 1) Schmidl-Cox detect
    search_len = min(40000, len(rx) - 1)
    ok_sc, sc_idx, sc_peak, ratio, cfo_hz, _, _ = detect_stf_autocorr(
        rx, fs=fs, half_period=N_FFT//2,
        threshold=0.20,
        ratio_th=sync_ratio_th,
        search_len=search_len
    )
    if not ok_sc:
        return {
            "status": "no_signal",
            "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio),
            "stf_idx": int(sc_idx),
            "stf_peak": float(sc_peak),
            "mean_evm": np.nan,
            "rx_peak": float(np.max(np.abs(rx)))
        }

    # 2) CFO correction (coarse)
    rx_cfo = apply_cfo(rx, cfo_hz, fs)

    # 3) Crosscorr refine around sc_idx using STF payload ref (NO CP)
    # sc_idx points into plateau; it is close to STF payload start.
    stf_idx, stf_peak = crosscorr_refine(rx_cfo, stf_payload_ref, center_idx=sc_idx, win=2000)
    if stf_idx < 0 or stf_peak < 0.02:
        return {
            "status": "no_signal",
            "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio),
            "stf_idx": int(stf_idx if stf_idx >= 0 else sc_idx),
            "stf_peak": float(stf_peak),
            "mean_evm": np.nan,
            "rx_peak": float(np.max(np.abs(rx)))
        }

    # 4) LTF start: immediately after STF payload
    stf_payload_len = stf_repeats * N_FFT
    ltf_start = stf_idx + stf_payload_len

    H, snr_sc = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref, num_symbols=ltf_symbols)
    if H is None:
        return {
            "status": "ltf_fail",
            "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "mean_evm": np.nan,
            "rx_peak": float(np.max(np.abs(rx)))
        }

    payload_start = ltf_start + ltf_symbols * SYMBOL_LEN

    # 5) Probe demod: enough symbols to cover header typically
    bits_probe, diag_probe = demod_payload_symbols(
        rx_cfo, payload_start, H,
        num_syms=max_syms_probe,
        kp=kp, ki=ki,
        repeat=repeat
    )
    if bits_probe is None:
        return {
            "status": "no_packet",
            "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "mean_evm": np.nan,
            "rx_peak": float(np.max(np.abs(rx)))
        }

    # 6) Bit-slip search for MAGIC+header (allow incomplete payload)
    best_slip = None
    best_hdr = None
    for slip in range(8):
        bb = packbits_with_slip(bits_probe, slip)
        okh, fid, seq, total, plen, expected_len = try_parse_header_only(bb)
        if okh:
            best_slip = slip
            best_hdr = (fid, seq, total, plen, expected_len)
            break

    if best_hdr is None:
        return {
            "status": "no_packet",
            "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "mean_evm": float(np.mean(diag_probe["evm_per_sym"])) if diag_probe["evm_per_sym"].size else np.nan,
            "rx_peak": float(np.max(np.abs(rx)))
        }

    fid, seq, total, plen, expected_len = best_hdr
    required_bits = expected_len * 8 + best_slip
    required_syms = int(np.ceil(required_bits / BITS_PER_OFDM_SYM))
    required_syms = max(1, required_syms)

    # 7) Full demod only needed symbols
    bits_full, diag_full = demod_payload_symbols(
        rx_cfo, payload_start, H,
        num_syms=required_syms,
        kp=kp, ki=ki,
        repeat=repeat
    )
    if bits_full is None:
        return {
            "status": "no_packet",
            "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "mean_evm": np.nan,
            "rx_peak": float(np.max(np.abs(rx)))
        }

    # 8) Full CRC check (try best_slip first, then all slips)
    bb = packbits_with_slip(bits_full, best_slip)
    ok, fid2, seq2, total2, payload, expected2 = parse_video_packet_full(bb)
    if not ok:
        for slip in range(8):
            bb2 = packbits_with_slip(bits_full, slip)
            ok2, fid2, seq2, total2, payload2, expected2 = parse_video_packet_full(bb2)
            if ok2:
                ok, fid2, seq2, total2, payload = True, fid2, seq2, total2, payload2
                best_slip = slip
                break

    mean_evm = float(np.mean(diag_full["evm_per_sym"])) if diag_full["evm_per_sym"].size else np.nan

    if verbose:
        print(f"    [SYNC] ratio={ratio:.2f} stf_peak={stf_peak:.3f} idx={stf_idx} "
              f"cfo={cfo_hz:+.1f}Hz slip={best_slip} req_syms={required_syms} evm={mean_evm:.4f}")

    if not ok:
        return {
            "status": "crc_fail",
            "cfo_hz": float(cfo_hz),
            "sync_ratio": float(ratio),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "frame_id": int(fid),
            "seq": int(seq),
            "total": int(total),
            "mean_evm": mean_evm,
            "rx_peak": float(np.max(np.abs(rx))),
            "all_data_syms": diag_full["all_data_syms"],
            "evm_per_sym": diag_full["evm_per_sym"],
            "phase_err": diag_full["phase_err"],
            "freq_log": diag_full["freq_log"],
            "snr_per_sc": snr_sc
        }

    return {
        "status": "demod_ok",
        "cfo_hz": float(cfo_hz),
        "sync_ratio": float(ratio),
        "stf_idx": int(stf_idx),
        "stf_peak": float(stf_peak),
        "frame_id": int(fid2),
        "seq": int(seq2),
        "total": int(total2),
        "payload": payload,
        "mean_evm": mean_evm,
        "rx_peak": float(np.max(np.abs(rx))),
        "all_data_syms": diag_full["all_data_syms"],
        "evm_per_sym": diag_full["evm_per_sym"],
        "phase_err": diag_full["phase_err"],
        "freq_log": diag_full["freq_log"],
        "snr_per_sc": snr_sc
    }

# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser(description="RF Link Step7 RX - FIXED v2 (log+plot)")
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--rx_gain", type=float, default=60)
    ap.add_argument("--buf_size", type=int, default=65536)

    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.005)
    ap.add_argument("--sync_ratio", type=float, default=6.0)
    ap.add_argument("--max_syms_probe", type=int, default=40)

    ap.add_argument("--max_captures", type=int, default=5000)
    ap.add_argument("--max_frames", type=int, default=0, help="Stop after N decoded frames (0=unlimited)")

    ap.add_argument("--output_dir", default="rf_link_step7_results_rx_v2")
    ap.add_argument("--output_video", default="received_video_v2.avi")
    ap.add_argument("--output_fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)

    ap.add_argument("--frame_timeout", type=float, default=60.0)
    ap.add_argument("--max_inflight", type=int, default=10)

    ap.add_argument("--plot_every_frame", type=int, default=1, help="Save frame diagnostics every N completed frames (0=disable)")
    ap.add_argument("--save_frames_dir", default="", help="If set, save decoded frames as JPEG here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    # Hardware deps
    import cv2
    import adi

    # Refs
    stf_payload_ref, _ = create_schmidl_cox_stf_payload(num_repeats=args.stf_repeats)
    _, ltf_freq_ref = create_ltf_ref(num_symbols=args.ltf_symbols)

    print("\n" + "="*86)
    print("RF Link Step 7 RX (FIXED v2 - fast SC + header probe + bounded demod + log/plot)")
    print("="*86)
    print(f"uri={args.uri}  fc={args.fc/1e6:.1f}MHz  fs={args.fs/1e6:.1f}Msps  rx_gain={args.rx_gain}")
    print(f"buf={args.buf_size}  stf_repeats={args.stf_repeats}  ltf_syms={args.ltf_symbols}  repeat={args.repeat}")
    print(f"sync_ratio_th={args.sync_ratio}  probe_syms={args.max_syms_probe}  kp={args.kp} ki={args.ki}")
    print("="*86)

    # SDR config
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.fs * 1.2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_enabled_channels = [0]
    sdr.rx_buffer_size = int(args.buf_size)

    # flush
    for _ in range(4):
        _ = sdr.rx()

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(args.output_video, fourcc, float(args.output_fps), (args.width, args.height))

    # logging
    csv_path = os.path.join(args.output_dir, "captures.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "capture", "status", "sync_ratio", "stf_peak", "stf_idx",
        "cfo_hz", "mean_evm", "rx_peak",
        "frame_id", "seq", "total", "payload_len"
    ])
    csv_f.flush()

    acc = FrameAccumulator(max_inflight=args.max_inflight, timeout=args.frame_timeout)

    # session arrays for summary plots
    cap_ok = []
    cap_cfo = []
    cap_evm = []
    cap_ratio = []
    cap_status = []

    capture_stats = {
        "pkt_ok": 0,
        "crc_fail": 0,
        "no_signal": 0,
        "dup": 0,
        "frames_ok": 0,
        "frames_expired": 0,
    }

    all_frame_stats = {}
    frames_written = 0

    start_time = time.time()

    try:
        print("Listening for packets ...")
        print("-"*86)
        for ci in range(args.max_captures):
            rx_raw = sdr.rx()

            r = demod_one_capture(
                rx_raw_iq=rx_raw,
                fs=float(args.fs),
                stf_payload_ref=stf_payload_ref,
                ltf_freq_ref=ltf_freq_ref,
                stf_repeats=args.stf_repeats,
                ltf_symbols=args.ltf_symbols,
                kp=float(args.kp),
                ki=float(args.ki),
                repeat=int(args.repeat),
                max_syms_probe=int(args.max_syms_probe),
                sync_ratio_th=float(args.sync_ratio),
                verbose=bool(args.verbose)
            )

            status = r["status"]
            sync_ratio = float(r.get("sync_ratio", 0.0))
            stf_peak = float(r.get("stf_peak", 0.0))
            stf_idx = int(r.get("stf_idx", 0))
            cfo = float(r.get("cfo_hz", 0.0))
            mean_evm = r.get("mean_evm", np.nan)
            rx_peak = float(r.get("rx_peak", 0.0))

            fid = r.get("frame_id", -1)
            seq = r.get("seq", -1)
            total = r.get("total", 0)
            payload_len = len(r.get("payload", b"")) if "payload" in r else 0

            ok_capture = 1 if status == "demod_ok" else 0
            cap_ok.append(ok_capture)
            cap_cfo.append(cfo if ok_capture else np.nan)
            cap_evm.append(mean_evm if ok_capture else np.nan)
            cap_ratio.append(sync_ratio)
            cap_status.append(status)

            csv_w.writerow([ci, status, sync_ratio, stf_peak, stf_idx, cfo, mean_evm, rx_peak, fid, seq, total, payload_len])
            if (ci % 50) == 0:
                csv_f.flush()

            if status == "demod_ok":
                # store packet
                payload = r["payload"]
                is_new, complete = acc.add_packet(
                    frame_id=int(fid), seq=int(seq), total=int(total),
                    payload=payload,
                    evm=float(mean_evm) if np.isfinite(mean_evm) else 0.0,
                    cfo=float(cfo),
                    data_syms=r.get("all_data_syms", None)
                )

                if not is_new:
                    capture_stats["dup"] += 1
                else:
                    capture_stats["pkt_ok"] += 1

                if args.verbose and (ci < 20 or (ci+1) % 50 == 0 or complete):
                    fs_ = acc.get_stats(int(fid))
                    tag = "NEW" if is_new else "DUP"
                    print(f"[{ci+1:05d}] OK  F{fid} pkt{seq}/{total-1} {payload_len}B "
                          f"ratio={sync_ratio:6.2f} CFO={cfo:+7.1f}Hz EVM={float(mean_evm):.4f} [{tag}] "
                          f"({fs_.get('received',0)}/{fs_.get('total',0)})")

                if complete:
                    jpg = acc.get_frame_bytes(int(fid))
                    fs_ = acc.get_stats(int(fid))

                    img = None
                    if jpg is not None and len(jpg) > 0:
                        nparr = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        if img.shape[1] != args.width or img.shape[0] != args.height:
                            img = cv2.resize(img, (args.width, args.height))
                        vw.write(img)
                        frames_written += 1
                        capture_stats["frames_ok"] += 1

                        if args.save_frames_dir:
                            cv2.imwrite(os.path.join(args.save_frames_dir, f"frame_{int(fid):04d}.jpg"), img)

                        print(f"  >>> Frame {fid} COMPLETE: {len(jpg)}B  latency={fs_.get('latency',0):.2f}s  "
                              f"meanEVM={fs_.get('mean_evm',0):.4f}  [frames={frames_written}]")
                    else:
                        print(f"  >>> Frame {fid} COMPLETE but JPEG decode FAILED ({len(jpg) if jpg else 0}B)")

                    # per-frame stats
                    all_frame_stats[int(fid)] = {
                        "complete_time": time.time(),
                        "jpeg_size": len(jpg) if jpg else 0,
                        "latency": fs_.get("latency", 0.0),
                        "mean_evm": fs_.get("mean_evm", 0.0),
                        "completion": 1.0,
                        "total_pkts": int(total),
                    }

                    # per-frame diagnostics
                    if args.plot_every_frame > 0 and (frames_written % args.plot_every_frame == 0):
                        pkt_meta = acc.frames.get(int(fid), {}).get("pkt_meta", {})
                        plot_frame_diagnostics(int(fid), img, pkt_meta, int(total), args.output_dir)

                    acc.mark_completed(int(fid))

                    if args.max_frames > 0 and frames_written >= args.max_frames:
                        print(f"Reached max_frames={args.max_frames}.")
                        break

            else:
                if status in ("no_signal", "ltf_fail"):
                    capture_stats["no_signal"] += 1
                else:
                    capture_stats["crc_fail"] += 1

                if args.verbose and (ci < 20 or (ci+1) % 200 == 0):
                    print(f"[{ci+1:05d}] {status:>10s}  ratio={sync_ratio:6.2f}  CFO={cfo:+7.1f}Hz  peak={rx_peak:.1f}")

            # expire
            expired = acc.expire_old()
            for efid, est in expired:
                capture_stats["frames_expired"] += 1
                print(f"  [EXPIRED] Frame {efid}: {est.get('received',0)}/{est.get('total',0)} pkts "
                      f"(completion={est.get('completion',0)*100:.1f}%)")

            if (ci+1) % 500 == 0:
                elapsed = time.time() - start_time
                print(f"\n--- Status @{ci+1} captures ---")
                print(f"frames_written={frames_written}, active={acc.active_summary()}, elapsed={elapsed:.1f}s\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        end_time = time.time()
        vw.release()
        csv_f.flush()
        csv_f.close()

        # summary
        elapsed = end_time - start_time
        print("\n" + "="*86)
        print("FINAL SUMMARY (RX v2)")
        print("="*86)
        print(f"Frames written: {frames_written}")
        print(f"Frames expired: {capture_stats['frames_expired']}")
        print(f"Packets OK:     {capture_stats['pkt_ok']}")
        print(f"CRC/NoPacket:   {capture_stats['crc_fail']}")
        print(f"No signal/LTF:  {capture_stats['no_signal']}")
        print(f"Duplicates:     {capture_stats['dup']}")
        if elapsed > 0:
            print(f"Elapsed:        {elapsed:.1f}s  FPS={frames_written/elapsed:.4f}")
        print(f"Output video:   {args.output_video}")
        print(f"Capture log:    {csv_path}")

        # save arrays
        session_npz = os.path.join(args.output_dir, "session.npz")
        np.savez_compressed(
            session_npz,
            cap_ok=np.array(cap_ok, dtype=np.int8),
            cap_cfo=np.array(cap_cfo, dtype=np.float32),
            cap_evm=np.array(cap_evm, dtype=np.float32),
            cap_ratio=np.array(cap_ratio, dtype=np.float32),
            cap_status=np.array(cap_status, dtype=object),
        )
        print(f"Session npz:    {session_npz}")

        # summary plot
        summary_png = os.path.join(args.output_dir, "video_summary.png")
        plot_video_summary(
            all_frame_stats=all_frame_stats,
            capture_arrays={
                "cap_ok": cap_ok,
                "cap_cfo": cap_cfo,
                "cap_evm": cap_evm,
                "cap_ratio": cap_ratio,
                "cap_status": cap_status,
                "crc_fail": capture_stats["crc_fail"],
                "no_signal": capture_stats["no_signal"],
                "dup": capture_stats["dup"],
                "frames_expired": capture_stats["frames_expired"],
                "frames_ok": capture_stats["frames_ok"],
            },
            out_png=summary_png,
            start_time=start_time,
            end_time=end_time
        )
        print(f"Summary plot:   {summary_png}")
        print(f"Diagnostics:   {args.output_dir}/frame_XXXX.png")
        print("="*86)

        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

if __name__ == "__main__":
    main()