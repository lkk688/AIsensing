#!/usr/bin/env python3
"""
RF Link Test - Step 7: RX (Video Reception) - FIXED VERSION

Key fixes vs. previous Step7 RX:
1) Consistent FFT bin indexing:
   - Use *unshifted* FFT for OFDM symbols (no fftshift).
   - Use (k + N_FFT) % N_FFT mapping for subcarriers k in [-N/2..N/2-1].
2) CFO estimation:
   - Primary CFO estimate from Schmidl-Cox STF repetition (robust for OFDM preamble).
   - Optional tone CFO is kept as diagnostic only (disabled by default).
3) Prevent tail contamination:
   - First pass: coarse demod a small number of symbols and try to parse MAGIC+CRC with 0..7 bit slip.
   - Once a valid packet is found, compute required OFDM symbols and demod ONLY that many symbols.
4) Bit-slip search:
   - Try 0..7 bit offsets to find MAGIC+CRC.
5) More stable sync gating:
   - Use peak-to-median ratio to reject false peaks.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import zlib
import time

MAGIC = b"VID7"

# ==============================
# OFDM Parameters (match TX)
# ==============================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

# IEEE 802.11-like used range (-26..-1, +1..+26)
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array([k for k in range(-26, 27)
                              if k != 0 and k not in set(PILOT_SUBCARRIERS)], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)        # 48
BITS_PER_QPSK_SYM = 2
BITS_PER_OFDM_SYM = N_DATA * BITS_PER_QPSK_SYM  # 96

# Subcarrier index mapping for UN-SHIFTED FFT:
# k in [-32..31], bin index = (k + N) % N
def sc_to_bin(k: int) -> int:
    return (k + N_FFT) % N_FFT

PILOT_BINS = np.array([sc_to_bin(k) for k in PILOT_SUBCARRIERS], dtype=int)
DATA_BINS  = np.array([sc_to_bin(k) for k in DATA_SUBCARRIERS], dtype=int)

# ==============================
# Helpers
# ==============================
def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(len(samples), dtype=np.float32)
    return samples * np.exp(-1j * 2 * np.pi * (cfo_hz / fs) * n).astype(np.complex64)

def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    # Gray-ish mapping consistent with your earlier demap
    bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
    re = (np.real(symbols) >= 0).astype(np.uint8)
    im = (np.imag(symbols) >= 0).astype(np.uint8)
    # 00: + +, 01: - +, 11: - -, 10: + -
    # Your earlier mapping:
    # (+,+)->00, (-,+)->01, (-,-)->11, (+,-)->10
    bits[0::2] = (1 - re) & (1 - im)  # not used directly; keep exact with branch below
    # Keep explicit for correctness:
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
    bits = bits[:L].reshape(-1, repeat)
    return (np.sum(bits, axis=1) >= (repeat / 2)).astype(np.uint8)

def packbits_with_slip(bits: np.ndarray, slip: int) -> bytes:
    # slip in [0..7] means drop first slip bits
    if slip > 0:
        if len(bits) <= slip:
            return b""
        bits2 = bits[slip:]
    else:
        bits2 = bits
    # truncate to multiple of 8
    L = (len(bits2) // 8) * 8
    if L <= 0:
        return b""
    return np.packbits(bits2[:L]).tobytes()

def parse_video_packet(data: bytes):
    """
    MAGIC(4) | FRAME_ID(2) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    CRC covers MAGIC..PAYLOAD
    """
    if len(data) < 16:
        return False, -1, -1, 0, b"", 0
    if data[:4] != MAGIC:
        return False, -1, -1, 0, b"", 0
    frame_id = int.from_bytes(data[4:6], "little")
    seq      = int.from_bytes(data[6:8], "little")
    total    = int.from_bytes(data[8:10], "little")
    plen     = int.from_bytes(data[10:12], "little")
    expected = 12 + plen + 4
    if len(data) < expected:
        return False, frame_id, seq, total, b"", expected
    content  = data[:12 + plen]
    payload  = data[12:12 + plen]
    crc_rx   = int.from_bytes(data[12 + plen:12 + plen + 4], "little")
    crc_calc = zlib.crc32(content) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return False, frame_id, seq, total, b"", expected
    return True, frame_id, seq, total, payload, expected

# ==============================
# STF/LTF Construction (match TX)
# ==============================
def create_schmidl_cox_stf(num_repeats=6):
    """
    STF: time-domain sequence with half-symbol repetition property.
    Here we generate an OFDM-like training symbol on even subcarriers, then repeat.
    """
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i] + 0j
    x = np.fft.ifft(X) * np.sqrt(N_FFT)
    stf = np.tile(x, num_repeats).astype(np.complex64)
    # prepend CP once for the whole STF block (same as your old code)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp, X

def create_ltf_ref(num_symbols=4):
    """
    LTF: deterministic BPSK on all used subcarriers.
    """
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(X) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X

# ==============================
# Schmidl-Cox Metric + CFO
# ==============================
def schmidl_cox_metric(rx: np.ndarray, L: int, search_len: int):
    """
    M[n] = |P[n]|^2 / (R[n]^2)
    P[n] = sum_{m=0..L-1} r[n+m] * conj(r[n+m+L])
    R[n] = sum_{m=0..L-1} |r[n+m+L]|^2
    """
    N = len(rx)
    search_len = min(search_len, N - 2*L - 1)
    if search_len <= 1:
        return None, None, None
    P = np.zeros(search_len, dtype=np.complex64)
    R = np.zeros(search_len, dtype=np.float32)
    # vectorized-ish using sliding windows (still O(N*L), but search_len is capped)
    for n in range(search_len):
        a = rx[n:n+L]
        b = rx[n+L:n+2*L]
        P[n] = np.sum(a * np.conj(b))
        R[n] = np.sum(np.abs(b)**2)
    M = (np.abs(P)**2) / (R**2 + 1e-12)
    return M, P, R

def detect_stf_autocorr(rx: np.ndarray, half_period=32, threshold=0.25, search_len=60000):
    """
    Return:
      ok, idx, peak_smooth, cfo_hz, M_raw, peak_to_median
    CFO estimate from angle(P) over plateau:
      angle(Psum) = -2*pi*CFO*(L/fs) => CFO = -angle(Psum)*fs/(2*pi*L)
    """
    L = half_period
    MPR = 0.0
    M, P, R = schmidl_cox_metric(rx, L=L, search_len=search_len)
    if M is None:
        return False, 0, 0.0, 0.0, np.array([]), 0.0

    plateau = 2*L
    if len(M) <= plateau+1:
        return False, 0, float(np.max(M)), 0.0, M, float(np.max(M) / (np.median(M)+1e-12))

    kernel = np.ones(plateau, dtype=np.float32) / plateau
    M_s = np.convolve(M, kernel, mode="valid")
    pk = int(np.argmax(M_s))
    peak_s = float(M_s[pk])
    med = float(np.median(M_s) + 1e-12)
    MPR = peak_s / med

    # refine idx: earliest point before peak where M drops
    idx = pk
    for i in range(pk, max(0, pk - 5*L), -1):
        if M[i] < 0.1 * peak_s:
            idx = i + 1
            break

    # CFO from P over plateau window
    p_sum = np.sum(P[idx: idx + plateau])
    cfo_hz = -np.angle(p_sum) * (FS_GLOBAL / (2*np.pi*L))

    ok = (peak_s > threshold) and (MPR > 6.0)  # MPR gate helps reject false sync
    return ok, idx, peak_s, float(cfo_hz), M, float(MPR)

def detect_stf_crosscorr(rx: np.ndarray, stf_ref: np.ndarray, search_len=60000):
    """
    Cross-correlation peak on a bounded window.
    """
    L = len(stf_ref)
    search_len = min(search_len, len(rx) - L - 1)
    if search_len <= 0:
        return -1, 0.0, np.array([])
    seg = rx[:search_len + L]
    corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
    denom = np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12
    corr_n = corr / denom
    idx = int(np.argmax(corr_n))
    pk = float(corr_n[idx])
    return idx, pk, corr_n

def extract_ofdm_symbol_freq(rx: np.ndarray, start_idx: int):
    """
    Extract one OFDM symbol from rx at symbol boundary start_idx (points to CP start).
    Return UN-SHIFTED FFT bins.
    """
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    td = rx[start_idx + N_CP : start_idx + SYMBOL_LEN]
    return np.fft.fft(td) / np.sqrt(N_FFT)

def channel_estimate_from_ltf(rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray, num_symbols=4):
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

    if len(Ys) >= 2:
        Ystk = np.stack(Ys, axis=0)[:, used_bins]
        noise_var = np.var(Ystk, axis=0)
        sig_var = np.abs(np.mean(Ystk, axis=0))**2
        snr_sc = 10*np.log10(sig_var / (noise_var + 1e-12) + 1e-12)
    else:
        snr_sc = np.zeros(len(used), dtype=np.float32)

    return H, snr_sc

# ==============================
# Demod + PLL/FLL
# ==============================
def demod_payload_symbols(rx: np.ndarray,
                         payload_start: int,
                         H: np.ndarray,
                         num_syms: int,
                         kp: float,
                         ki: float,
                         repeat: int):
    """
    Demodulate num_syms OFDM payload symbols into bits and also return diagnostics.
    - Equalize with H (from LTF)
    - Pilot-based PLL/FLL for residual phase/CFO tracking
    """
    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
    ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)

    phase_acc = 0.0
    freq_acc = 0.0
    prev_phase = None

    all_data_syms = []
    evm_per_sym = []
    phase_errs = []
    freq_log = []

    for si in range(num_syms):
        Y = extract_ofdm_symbol_freq(rx, payload_start + si*SYMBOL_LEN)
        if Y is None:
            break

        # Equalize only used bins
        Yeq = Y.copy()
        used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
        used_bins = np.array([sc_to_bin(int(k)) for k in used], dtype=int)
        H_used = H[used_bins]
        Yeq[used_bins] = Yeq[used_bins] / (H_used + 1e-12)

        # Pilot phase error
        pilot_sign = 1 if (si % 2 == 0) else -1
        exp_p = pilot_sign * pilot_pattern
        rp = Yeq[PILOT_BINS]
        pilot_corr = np.sum(rp * np.conj(exp_p))
        ph = float(np.angle(pilot_corr))

        if prev_phase is not None:
            dph = ph - prev_phase
            # wrap
            while dph > np.pi:
                dph -= 2*np.pi
            while dph < -np.pi:
                dph += 2*np.pi
            freq_acc += ki * dph
        prev_phase = ph
        phase_acc += freq_acc + kp * ph

        # apply residual correction
        Yeq = Yeq * np.exp(-1j * phase_acc).astype(np.complex64)

        data_syms = Yeq[DATA_BINS]

        # EVM
        nearest = ideal[np.argmin(np.abs(data_syms[:, None] - ideal[None, :]), axis=1)]
        evm = float(np.sqrt(np.mean(np.abs(data_syms - nearest)**2)))

        all_data_syms.append(data_syms)
        evm_per_sym.append(evm)
        phase_errs.append(ph)
        freq_log.append(freq_acc)

    if not all_data_syms:
        return None, {
            "evm_per_sym": np.array([]),
            "phase_err": np.array([]),
            "freq_log": np.array([]),
            "all_data_syms": np.array([], dtype=np.complex64),
            "num_syms": 0
        }

    all_data_syms = np.concatenate(all_data_syms, axis=0)
    bits_raw = qpsk_demap(all_data_syms)
    bits = majority_vote(bits_raw, repeat)

    return bits, {
        "evm_per_sym": np.array(evm_per_sym, dtype=np.float32),
        "phase_err": np.array(phase_errs, dtype=np.float32),
        "freq_log": np.array(freq_log, dtype=np.float32),
        "all_data_syms": all_data_syms,
        "num_syms": len(evm_per_sym)
    }

# ==============================
# Demod One Capture: robust + bounded
# ==============================
def demod_one_capture(rx_raw_iq: np.ndarray,
                      fs: float,
                      stf_ref: np.ndarray,
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
    Returns dict with:
      status: demod_ok / no_signal / ltf_fail / no_packet
      cfo_hz, stf_idx, ltf_start, payload_start
      bits_bytes (best decoded bytes)
      diagnostics (evm/snr/constellation etc.)
    """
    # normalize from Pluto int scale if needed; you may already get complex64 with int16 range
    rx = rx_raw_iq.astype(np.complex64)
    # Heuristic: if magnitude is large, scale down
    if np.median(np.abs(rx)) > 100:
        rx = rx / (2**14)
    rx = rx - np.mean(rx)

    # --- Step 1: STF autocorr detect + CFO
    ok_sc, sc_idx, sc_peak, cfo_hz, M, ratio = detect_stf_autocorr(
        rx, half_period=N_FFT//2, threshold=0.20, search_len=min(60000, len(rx)-1)
    )

    # --- Step 2: apply coarse CFO (from STF)
    rx_cfo = apply_cfo(rx, cfo_hz, fs)

    # --- Step 3: STF crosscorr refine around sc_idx (or globally if sc failed)
    if ok_sc:
        win = 2000
        s0 = max(0, sc_idx - win)
        s1 = min(len(rx_cfo), sc_idx + win + len(stf_ref))
        seg = rx_cfo[s0:s1]
        corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
        denom = np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12
        corr_n = corr / denom
        loc = int(np.argmax(corr_n))
        stf_idx = s0 + loc
        stf_peak = float(corr_n[loc])
        method = "sc+xc_refine"
    else:
        stf_idx0, stf_peak, _ = detect_stf_crosscorr(rx_cfo, stf_ref, search_len=min(60000, len(rx_cfo)-1))
        stf_idx = stf_idx0
        method = "xc_global"

    # gate by ratio and peak
    if (not ok_sc) and (stf_peak < 0.02):
        return {
            "status": "no_signal",
            "cfo_hz": float(cfo_hz),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "sync_ratio": float(ratio),
            "method": method,
            "bits_bytes": b"",
            "all_data_syms": np.array([], dtype=np.complex64),
            "mean_evm": 0.0
        }

    if ok_sc and ratio < sync_ratio_th:
        # even if we found something, likely false sync
        return {
            "status": "no_signal",
            "cfo_hz": float(cfo_hz),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "sync_ratio": float(ratio),
            "method": method,
            "bits_bytes": b"",
            "all_data_syms": np.array([], dtype=np.complex64),
            "mean_evm": 0.0
        }

    # --- Step 4: LTF channel estimate
    ltf_start = stf_idx + len(stf_ref)
    H, snr_sc = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref, num_symbols=ltf_symbols)
    if H is None:
        return {
            "status": "ltf_fail",
            "cfo_hz": float(cfo_hz),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "sync_ratio": float(ratio),
            "method": method,
            "bits_bytes": b"",
            "all_data_syms": np.array([], dtype=np.complex64),
            "mean_evm": 0.0
        }

    payload_start = ltf_start + ltf_symbols * SYMBOL_LEN

    # --- Step 5: PROBE a small number of symbols, search MAGIC+CRC with bit slips
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
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "sync_ratio": float(ratio),
            "method": method,
            "bits_bytes": b"",
            "all_data_syms": np.array([], dtype=np.complex64),
            "mean_evm": 0.0
        }

    best = None
    best_slip = None
    best_expect = None

    for slip in range(8):
        bb = packbits_with_slip(bits_probe, slip)
        ok, fid, seq, total, payload, expected_len = parse_video_packet(bb)
        if ok:
            best = (fid, seq, total, payload, bb, expected_len)
            best_slip = slip
            best_expect = expected_len
            break

    if best is None:
        # no valid packet in probe
        return {
            "status": "no_packet",
            "cfo_hz": float(cfo_hz),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "sync_ratio": float(ratio),
            "method": method,
            "bits_bytes": b"",
            "all_data_syms": diag_probe["all_data_syms"],
            "mean_evm": float(np.mean(diag_probe["evm_per_sym"])) if diag_probe["evm_per_sym"].size else 0.0,
            "probe_evm": diag_probe["evm_per_sym"]
        }

    # --- Step 6: Now we know expected packet byte length, decode ONLY required OFDM symbols
    required_bits = (best_expect * 8) + best_slip
    # after repetition decoding, bits length is after majority_vote -> "logical bits"
    # Our demod_payload_symbols returns bits after majority_vote already.
    # So we need enough logical bits to cover required_bits (including slip).
    required_syms = int(np.ceil(required_bits / BITS_PER_OFDM_SYM))

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
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "sync_ratio": float(ratio),
            "method": method,
            "bits_bytes": b"",
            "all_data_syms": np.array([], dtype=np.complex64),
            "mean_evm": 0.0
        }

    bb_full = packbits_with_slip(bits_full, best_slip)
    ok, fid, seq, total, payload, expected_len = parse_video_packet(bb_full)
    if not ok:
        # fallback: still try all slips on full bits
        for slip in range(8):
            bb2 = packbits_with_slip(bits_full, slip)
            ok2, fid2, seq2, total2, payload2, expected_len2 = parse_video_packet(bb2)
            if ok2:
                ok, fid, seq, total, payload, expected_len = True, fid2, seq2, total2, payload2, expected_len2
                best_slip = slip
                bb_full = bb2
                break

    if not ok:
        return {
            "status": "crc_fail",
            "cfo_hz": float(cfo_hz),
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "sync_ratio": float(ratio),
            "method": method,
            "bits_bytes": b"",
            "all_data_syms": diag_full["all_data_syms"],
            "mean_evm": float(np.mean(diag_full["evm_per_sym"])) if diag_full["evm_per_sym"].size else 0.0
        }

    mean_evm = float(np.mean(diag_full["evm_per_sym"])) if diag_full["evm_per_sym"].size else 0.0

    if verbose:
        print(f"    [SYNC] method={method} sc_ratio={ratio:.2f} stf_peak={stf_peak:.3f} "
              f"cfo={cfo_hz:+.1f}Hz slip={best_slip} req_syms={required_syms}")

    return {
        "status": "demod_ok",
        "cfo_hz": float(cfo_hz),
        "stf_idx": int(stf_idx),
        "stf_peak": float(stf_peak),
        "sync_ratio": float(ratio),
        "method": method,
        "frame_id": int(fid),
        "seq": int(seq),
        "total": int(total),
        "payload": payload,
        "bits_bytes": bb_full,
        "all_data_syms": diag_full["all_data_syms"],
        "mean_evm": mean_evm,
        "evm_per_sym": diag_full["evm_per_sym"],
        "phase_err": diag_full["phase_err"],
        "freq_log": diag_full["freq_log"],
        "snr_per_sc": snr_sc
    }

# ==============================
# Frame accumulator (same as yours, slightly simplified)
# ==============================
class FrameAccumulator:
    def __init__(self, max_inflight=10, timeout=60.0):
        self.frames = {}
        self.completed = set()
        self.max_inflight = max_inflight
        self.timeout = timeout

    def add_packet(self, frame_id, seq, total, payload, evm=0.0, cfo=0.0, data_syms=None):
        if frame_id in self.completed:
            return False

        now = time.time()
        if frame_id not in self.frames:
            self.frames[frame_id] = {
                "pkts": {},
                "total": total,
                "ts_first": now,
                "ts_last": now,
                "evm_sum": 0.0,
                "evm_cnt": 0,
                "cfo_list": [],
                "pkt_meta": {}  # seq -> (evm, syms)
            }
        e = self.frames[frame_id]
        e["ts_last"] = now

        if seq not in e["pkts"]:
            e["pkts"][seq] = payload
            e["evm_sum"] += float(evm)
            e["evm_cnt"] += 1
            e["cfo_list"].append(float(cfo))
            e["pkt_meta"][seq] = {
                "evm": float(evm),
                "data_syms": data_syms.copy() if isinstance(data_syms, np.ndarray) else np.array([], dtype=np.complex64)
            }

        return (len(e["pkts"]) >= e["total"])

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
            "latency": e["ts_last"] - e["ts_first"]
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
# Diagnostics plots (reuse your style)
# ==============================
def plot_video_summary_simple(all_frame_stats, capture_stats, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    if capture_stats["cfos"]:
        ax.plot(capture_stats["cfos"], ".", markersize=2)
        ax.set_title(f"CFO per Packet (std={np.std(capture_stats['cfos']):.1f} Hz)")
    ax.set_xlabel("Packet")
    ax.set_ylabel("CFO (Hz)")
    ax.grid(True)

    ax = axes[0, 1]
    if capture_stats["evms"]:
        ax.hist(capture_stats["evms"], bins=30, edgecolor="black", alpha=0.8)
        ax.set_title(f"EVM distribution (mean={np.mean(capture_stats['evms']):.4f})")
    ax.set_xlabel("Mean EVM per packet")
    ax.set_ylabel("Count")
    ax.grid(True)

    ax = axes[1, 0]
    if all_frame_stats:
        fids = sorted(all_frame_stats.keys())
        comp = [all_frame_stats[f]["completion"] for f in fids]
        ax.plot(comp, "b-o", markersize=3)
        ax.set_title("Frame completion ratio")
    ax.set_xlabel("Frame index (order)")
    ax.set_ylabel("Completion")
    ax.grid(True)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"Packets OK: {capture_stats['pkt_ok']}\n"
        f"CRC fails: {capture_stats['crc_fail']}\n"
        f"No signal: {capture_stats['no_signal']}\n"
        f"Duplicates: {capture_stats['dup']}\n"
        f"Frames OK: {capture_stats['frames_ok']}\n"
        f"Frames expired: {capture_stats['frames_expired']}\n"
    )
    ax.text(0.05, 0.95, txt, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))
    fig.suptitle("RF Link Step7 RX Summary (Fixed)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

# ==============================
# Main
# ==============================
FS_GLOBAL = 3e6  # updated in main

def main():
    ap = argparse.ArgumentParser("RF Link Step7 RX - FIXED")
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
    ap.add_argument("--sync_ratio", type=float, default=6.0, help="Schmidl-Cox peak/median ratio gate")
    ap.add_argument("--max_syms_probe", type=int, default=40, help="symbols for probing MAGIC+CRC before slicing")
    ap.add_argument("--max_captures", type=int, default=5000)
    ap.add_argument("--output_dir", default="rf_link_step7_results_fixed")
    ap.add_argument("--output_video", default="received_video_fixed.avi")
    ap.add_argument("--output_fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--frame_timeout", type=float, default=60.0)
    ap.add_argument("--max_inflight", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global FS_GLOBAL
    FS_GLOBAL = float(args.fs)

    import cv2
    import adi

    os.makedirs(args.output_dir, exist_ok=True)

    stf_ref, _ = create_schmidl_cox_stf(num_repeats=args.stf_repeats)
    _, ltf_freq_ref = create_ltf_ref(num_symbols=args.ltf_symbols)

    print("\n" + "="*78)
    print("RF Link Step 7 RX (FIXED)")
    print("="*78)
    print(f"uri={args.uri}  fc={args.fc/1e6:.1f}MHz  fs={args.fs/1e6:.1f}Msps  rx_gain={args.rx_gain}")
    print(f"buf={args.buf_size}  stf_repeats={args.stf_repeats} ltf_syms={args.ltf_symbols} repeat={args.repeat}")
    print(f"sync_ratio_th={args.sync_ratio}  probe_syms={args.max_syms_probe}")
    print("="*78)

    # SDR
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
    vw = cv2.VideoWriter(args.output_video, fourcc, args.output_fps, (args.width, args.height))

    acc = FrameAccumulator(max_inflight=args.max_inflight, timeout=args.frame_timeout)
    frames_written = 0
    all_frame_stats = {}

    capture_stats = {
        "pkt_ok": 0,
        "crc_fail": 0,
        "no_signal": 0,
        "dup": 0,
        "frames_ok": 0,
        "frames_expired": 0,
        "cfos": [],
        "evms": [],
    }

    start = time.time()

    try:
        for ci in range(args.max_captures):
            rx_raw = sdr.rx()

            r = demod_one_capture(
                rx_raw, args.fs,
                stf_ref, ltf_freq_ref,
                stf_repeats=args.stf_repeats,
                ltf_symbols=args.ltf_symbols,
                kp=args.kp, ki=args.ki, repeat=args.repeat,
                max_syms_probe=args.max_syms_probe,
                sync_ratio_th=args.sync_ratio,
                verbose=args.verbose
            )

            st = r["status"]

            if st == "demod_ok":
                fid = r["frame_id"]
                seq = r["seq"]
                total = r["total"]
                payload = r["payload"]
                evm = r.get("mean_evm", 0.0)
                cfo = r.get("cfo_hz", 0.0)

                # duplicates
                is_dup = False
                if fid in acc.frames and seq in acc.frames[fid]["pkts"]:
                    is_dup = True
                    capture_stats["dup"] += 1
                else:
                    capture_stats["pkt_ok"] += 1
                    capture_stats["cfos"].append(cfo)
                    capture_stats["evms"].append(evm)

                complete = acc.add_packet(fid, seq, total, payload, evm=evm, cfo=cfo,
                                          data_syms=r.get("all_data_syms", None))

                if args.verbose and (ci < 10 or (ci+1) % 50 == 0):
                    tag = "DUP" if is_dup else "NEW"
                    fs = acc.get_stats(fid)
                    print(f"[{ci+1:04d}] F{fid} pkt{seq}/{total-1} {len(payload)}B EVM={evm:.4f} CFO={cfo:+.0f}Hz [{tag}] "
                          f"({fs.get('received',0)}/{fs.get('total',0)})")

                if complete:
                    jpg = acc.get_frame_bytes(fid)
                    fs = acc.get_stats(fid)

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
                        print(f"  >>> Frame {fid} COMPLETE: {len(jpg)}B latency={fs.get('latency',0):.2f}s "
                              f"EVM={fs.get('mean_evm',0):.4f} [{frames_written}]")
                    else:
                        print(f"  >>> Frame {fid} COMPLETE but JPEG decode FAILED ({len(jpg) if jpg else 0}B)")

                    all_frame_stats[fid] = {
                        "complete_time": time.time(),
                        "jpeg_size": len(jpg) if jpg else 0,
                        "latency": fs.get("latency", 0),
                        "mean_evm": fs.get("mean_evm", 0),
                        "completion": 1.0,
                    }
                    acc.mark_completed(fid)

                    if args.max_frames > 0 and frames_written >= args.max_frames:
                        print(f"Reached max_frames={args.max_frames}.")
                        break

            elif st in ("no_signal", "ltf_fail"):
                capture_stats["no_signal"] += 1
                if args.verbose and (ci < 10 or (ci+1) % 100 == 0):
                    print(f"[{ci+1:04d}] {st} (sync_ratio={r.get('sync_ratio',0):.2f})")
            else:
                # crc_fail / no_packet
                capture_stats["crc_fail"] += 1
                if args.verbose and (ci < 10 or (ci+1) % 100 == 0):
                    print(f"[{ci+1:04d}] {st} (sync_ratio={r.get('sync_ratio',0):.2f})")

            # expire old frames
            expired = acc.expire_old()
            for fid, stt in expired:
                capture_stats["frames_expired"] += 1
                print(f"  [EXPIRED] Frame {fid}: {stt.get('received',0)}/{stt.get('total',0)} pkts")

            if (ci+1) % 200 == 0:
                elapsed = time.time() - start
                print(f"\n--- Status @{ci+1} captures ---")
                print(f"frames_written={frames_written}, active={acc.active_summary()}, elapsed={elapsed:.1f}s\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    end = time.time()
    vw.release()

    elapsed = end - start
    print("\n" + "="*78)
    print("FINAL SUMMARY (FIXED)")
    print("="*78)
    print(f"Frames written: {frames_written}")
    print(f"Frames expired: {capture_stats['frames_expired']}")
    print(f"Packets OK:     {capture_stats['pkt_ok']}")
    print(f"CRC/NoPacket:   {capture_stats['crc_fail']}")
    print(f"No signal/LTF:  {capture_stats['no_signal']}")
    print(f"Duplicates:     {capture_stats['dup']}")
    print(f"Elapsed:        {elapsed:.1f}s  FPS={frames_written/elapsed:.4f}" if elapsed > 0 else "")
    if capture_stats["cfos"]:
        print(f"CFO mean/std:   {np.mean(capture_stats['cfos']):.1f} / {np.std(capture_stats['cfos']):.1f} Hz")
    if capture_stats["evms"]:
        print(f"EVM mean/min/max: {np.mean(capture_stats['evms']):.4f} / {np.min(capture_stats['evms']):.4f} / {np.max(capture_stats['evms']):.4f}")
    print(f"Output video:   {args.output_video}")

    # summary plot
    out_png = os.path.join(args.output_dir, "video_summary_fixed.png")
    plot_video_summary_simple(all_frame_stats, capture_stats, out_png)
    print(f"Summary plot:   {out_png}")
    print("="*78)

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

if __name__ == "__main__":
    main()
    
"""
python rf_link_step7_rx_fixed.py \
  --uri ip:192.168.2.2 \
  --fc 2.3e9 --fs 3e6 \
  --rx_gain 60 \
  --buf_size 65536 \
  --stf_repeats 6 --ltf_symbols 4 \
  --repeat 1 \
  --sync_ratio 6 \
  --kp 0.15 --ki 0.005 \
  --max_syms_probe 40 \
  --verbose
"""
