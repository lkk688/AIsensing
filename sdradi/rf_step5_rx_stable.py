#!/usr/bin/env python3
"""
rf_step5_rx_stable.py

Step-5 RX "stable" version for PlutoSDR OFDM packets (MAGIC="AIS1").

What this script fixes / improves vs your current Step5 RX:

1) CFO:
   - Default CFO from Schmidl-Cox (STF autocorr) which is CFO-tolerant.
   - Optional tone CFO mode kept, but SC is recommended for stability.
   - CFO clipping (guard) to prevent crazy CFO when sync is wrong.

2) Sync robustness:
   - Schmidl-Cox timing metric to find STF plateau + CFO.
   - Optional cross-corr refine around SC index.
   - Candidate selection scores *header hit* first (MAGIC/CRC), then EVM.
     This avoids "EVM looks ok but wrong boundary" false locks.

3) Two-stage decode (probe -> bounded full decode):
   - Probe a small number of OFDM symbols.
   - Bit-slip search (0..7) to find MAGIC and/or CRC.
   - Once LENGTH is known, demod ONLY required OFDM symbols to avoid tail contamination.

4) Logging + plots:
   - Per-capture PNG diagnostic saved in output_dir.
   - CSV log saved for later analysis.
   - Summary plot at end.

TX assumptions (must match your Step5 TX):
- N_FFT=64, N_CP=16
- pilots: [-21,-7,7,21], data: 48 subcarriers in [-26..26] excluding pilots and DC
- QPSK mapping consistent with your previous scripts
- STF: Schmidl-Cox with num_repeats (default 6) + one CP prepended to whole STF block
- LTF: deterministic BPSK on used subcarriers, num_symbols (default 4)
- Packet format:
    MAGIC(4) | LEN(2 little) | PAYLOAD(LEN) | CRC32(payload)(4 little)

Example:
python3 rf_step5_rx_stable.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 --rx_gain 30 \
  --tries 30 --cfo_mode sc --probe_syms 16 --verbose

"""

import argparse
import csv
import os
import time
import zlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAGIC = b"AIS1"

# ==============================
# OFDM Parameters (match TX)
# ==============================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)              # 48
BITS_PER_OFDM_SYM = N_DATA * 2              # 96 (QPSK)

# Step5 convention uses fftshift(fft()) bins but still indexes with (k+N)%N like older scripts.
# We'll keep it EXACTLY as your Step5 RX/TX (fftshift + (k+N)%N).
PILOT_BINS = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS], dtype=int)
DATA_BINS  = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS], dtype=int)


# ==============================
# Helpers
# ==============================
def bits_to_bytes(bits: np.ndarray) -> bytes:
    L = (len(bits) // 8) * 8
    if L <= 0:
        return b""
    return np.packbits(bits[:L]).tobytes()

def packbits_with_slip(bits: np.ndarray, slip: int) -> bytes:
    if slip > 0:
        if len(bits) <= slip:
            return b""
        bits = bits[slip:]
    return bits_to_bytes(bits)

def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    if L <= 0:
        return np.zeros(0, dtype=np.uint8)
    x = bits[:L].reshape(-1, repeat)
    return (np.sum(x, axis=1) >= (repeat / 2)).astype(np.uint8)

def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
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

def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(len(samples), dtype=np.float32)
    return (samples * np.exp(-1j * 2 * np.pi * (cfo_hz / fs) * n)).astype(np.complex64)


# ==============================
# Packet parse
# ==============================
def parse_packet_bytes(bb: bytes):
    """
    Frame format:
      MAGIC(4) | LEN(2 little) | PAYLOAD | CRC32(payload)(4 little)
    Return:
      ok, payload, need_bytes, why, plen, crc_rx, crc_calc
    """
    if len(bb) < 10:
        return False, b"", 0, "too_short", 0, 0, 0
    if bb[:4] != MAGIC:
        return False, b"", 0, "bad_magic", 0, 0, 0
    plen = int.from_bytes(bb[4:6], "little")
    need = 6 + plen + 4
    if len(bb) < need:
        return False, b"", need, "need_more", plen, 0, 0
    payload = bb[6:6+plen]
    crc_rx = int.from_bytes(bb[6+plen:6+plen+4], "little")
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return False, b"", need, "crc_mismatch", plen, crc_rx, crc_calc
    return True, payload, need, "ok", plen, crc_rx, crc_calc


# ==============================
# Preamble refs (match TX)
# ==============================
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


# ==============================
# Schmidl-Cox metric (vectorized)
# ==============================
def schmidl_cox_metric(rx: np.ndarray, half_period: int = 32, window_len: int = 60000):
    """
    Returns M, P, R
      P[n] = sum rx[n:n+L]*conj(rx[n+L:n+2L])
      R[n] = sum |rx[n+L:n+2L]|^2
      M[n] = |P|^2 / R^2
    """
    L = half_period
    N = len(rx)
    window_len = min(window_len, N - 2*L)
    if window_len <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.complex64), np.array([], dtype=np.float32)

    c = rx[:window_len+L] * np.conj(rx[L:window_len+2*L])
    cs = np.zeros(window_len+L+1, dtype=np.complex64)
    cs[1:] = np.cumsum(c)
    P = cs[L:L+window_len] - cs[:window_len]

    p2 = np.abs(rx[:window_len+2*L])**2
    cr = np.zeros(window_len+2*L+1, dtype=np.float32)
    cr[1:] = np.cumsum(p2)
    R = cr[2*L:2*L+window_len] - cr[L:L+window_len]

    M = (np.abs(P)**2) / (R**2 + 1e-12)
    return M, P, R

def detect_stf_sc(rx: np.ndarray, fs: float,
                  sc_threshold: float = 0.1,
                  search_len: int = 60000,
                  peak_med_ratio_th: float = 6.0):
    """
    SC detect + CFO estimate from phase(Psum).
    Returns:
      ok, stf_idx, sc_peak_smooth, cfo_hz, ratio, M_raw
    """
    L = N_FFT // 2
    M, P, _ = schmidl_cox_metric(rx, half_period=L, window_len=search_len)
    if M.size == 0:
        return False, 0, 0.0, 0.0, 0.0, M

    plateau = 2 * L
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

    # refine to rising edge on raw M
    st = pk_i
    for i in range(pk_i, max(0, pk_i - 5*L), -1):
        if M[i] < 0.1 * pk_v:
            st = i + 1
            break

    p_sum = np.sum(P[st: st + plateau])
    cfo_hz = -np.angle(p_sum) * (fs / (2*np.pi*L))

    ok = (pk_v > sc_threshold) and (ratio >= peak_med_ratio_th)
    return ok, int(st), pk_v, float(cfo_hz), float(ratio), M

def detect_stf_xcorr(rx: np.ndarray, stf_ref: np.ndarray, search_len: int = 60000):
    L = len(stf_ref)
    search_len = min(search_len, len(rx) - L)
    if search_len <= 0:
        return -1, 0.0
    seg = rx[:search_len + L]
    corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
    corr /= (np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
    idx = int(np.argmax(corr))
    return idx, float(corr[idx])


# ==============================
# OFDM extraction & channel
# ==============================
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
        Y = extract_ofdm_symbol(rx, ltf_start + i * SYMBOL_LEN)
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


# ==============================
# Demod payload with pilot PLL/FLL and EVM tracking
# ==============================
def demod_payload_bits(rx: np.ndarray,
                       payload_start: int,
                       H: np.ndarray,
                       num_syms: int,
                       kp: float,
                       ki: float,
                       repeat: int,
                       bad_evm_th: float = 0.95,
                       bad_evm_patience: int = 6):
    """
    Demodulate `num_syms` OFDM symbols into *logical* bits (after majority_vote).
    Also returns diagnostics.

    NOTE: We do NOT early-stop aggressively by default; but we keep a mild
    "bad EVM patience" guard to avoid running off into noise forever.
    """
    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
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
        Y = extract_ofdm_symbol(rx, payload_start + si * SYMBOL_LEN)
        if Y is None:
            break

        # equalize used
        Yeq = Y.copy()
        for k in range(-26, 27):
            if k == 0:
                continue
            idx = (k + N_FFT) % N_FFT
            if np.abs(H[idx]) > 1e-6:
                Yeq[idx] = Yeq[idx] / (H[idx] + 1e-12)

        # pilots
        pilot_sign = 1 if (si % 2 == 0) else -1
        exp_p = pilot_sign * pilot_pattern
        rp = Yeq[PILOT_BINS]
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
        Yeq = Yeq * np.exp(-1j * phase_acc).astype(np.complex64)

        ds = Yeq[DATA_BINS]

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
        return None, {
            "num_syms": 0,
            "all_data_syms": np.array([], dtype=np.complex64),
            "evm_per_sym": np.array([], dtype=np.float32),
            "phase_err": np.array([], dtype=np.float32),
            "freq_log": np.array([], dtype=np.float32),
        }

    all_data_syms = np.concatenate(all_data, axis=0)
    bits_raw = qpsk_demap(all_data_syms)
    bits = majority_vote(bits_raw, repeat)

    return bits, {
        "num_syms": len(evm_sym),
        "all_data_syms": all_data_syms,
        "evm_per_sym": np.array(evm_sym, dtype=np.float32),
        "phase_err": np.array(ph_errs, dtype=np.float32),
        "freq_log": np.array(fr_log, dtype=np.float32),
    }


# ==============================
# Candidate evaluation
# ==============================
@dataclass
class CandidateResult:
    stf_idx: int
    stf_peak: float
    method: str
    cfo_hz: float
    ratio: float
    probe_slip: int
    probe_status: str   # ok / crc_mismatch / need_more / bad_magic / ...
    probe_need_bytes: int
    probe_plen: int
    probe_crc_rx: int
    probe_crc_calc: int
    probe_mean_evm: float

def evaluate_candidate(rx_raw: np.ndarray,
                       fs: float,
                       stf_ref: np.ndarray,
                       ltf_freq_ref: np.ndarray,
                       stf_idx: int,
                       stf_peak: float,
                       method: str,
                       cfo_hz: float,
                       ratio: float,
                       ltf_symbols: int,
                       probe_syms: int,
                       repeat: int,
                       kp: float,
                       ki: float,
                       bad_evm_th: float,
                       bad_evm_patience: int,
                       bit_slip_max: int,
                       max_payload_syms_cap: int):
    """
    Apply CFO, do LTF channel est, probe demod and look for MAGIC/CRC with bit-slip.
    Returns CandidateResult plus (optional) decoded final packet if we already can.
    """

    # CFO correction
    rx_cfo = apply_cfo(rx_raw, cfo_hz, fs)

    # LTF
    ltf_start = stf_idx + len(stf_ref)
    H, snr_sc = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref, num_symbols=ltf_symbols)
    if H is None:
        return None, None, {"snr_sc": np.array([])}

    payload_start = ltf_start + ltf_symbols * SYMBOL_LEN

    # Probe bits
    bits_probe, diag_probe = demod_payload_bits(
        rx_cfo, payload_start, H,
        num_syms=probe_syms,
        kp=kp, ki=ki,
        repeat=repeat,
        bad_evm_th=bad_evm_th,
        bad_evm_patience=bad_evm_patience
    )
    if bits_probe is None or len(bits_probe) == 0:
        cr = CandidateResult(
            stf_idx=stf_idx, stf_peak=stf_peak, method=method,
            cfo_hz=cfo_hz, ratio=ratio,
            probe_slip=-1, probe_status="no_payload",
            probe_need_bytes=0, probe_plen=0, probe_crc_rx=0, probe_crc_calc=0,
            probe_mean_evm=float(np.mean(diag_probe["evm_per_sym"])) if diag_probe["evm_per_sym"].size else 0.0
        )
        return cr, None, {"diag": diag_probe, "H": H, "snr_sc": snr_sc}

    # Search bit slips for best header/CRC
    best_hit = None
    for slip in range(bit_slip_max + 1):
        bb = packbits_with_slip(bits_probe, slip)
        ok, payload, need, why, plen, crc_rx, crc_calc = parse_packet_bytes(bb)
        if ok:
            best_hit = ("ok", slip, need, plen, crc_rx, crc_calc, payload)
            break
        # Prefer "crc_mismatch" over "bad_magic"
        if best_hit is None:
            if why in ("crc_mismatch", "need_more"):
                best_hit = (why, slip, need, plen, crc_rx, crc_calc, b"")
        else:
            # upgrade bad_magic -> need_more/crc_mismatch
            if best_hit[0] == "bad_magic" and why in ("need_more", "crc_mismatch"):
                best_hit = (why, slip, need, plen, crc_rx, crc_calc, b"")

    if best_hit is None:
        best_hit = ("bad_magic", 0, 0, 0, 0, 0, b"")

    status, slip, need_bytes, plen, crc_rx, crc_calc, payload0 = best_hit
    mean_evm = float(np.mean(diag_probe["evm_per_sym"])) if diag_probe["evm_per_sym"].size else 0.0

    cr = CandidateResult(
        stf_idx=stf_idx,
        stf_peak=stf_peak,
        method=method,
        cfo_hz=cfo_hz,
        ratio=ratio,
        probe_slip=int(slip),
        probe_status=str(status),
        probe_need_bytes=int(need_bytes),
        probe_plen=int(plen),
        probe_crc_rx=int(crc_rx),
        probe_crc_calc=int(crc_calc),
        probe_mean_evm=mean_evm
    )

    # If probe already CRC OK and payload complete, we can return success immediately
    if status == "ok" and len(payload0) == plen:
        return cr, {
            "payload": payload0,
            "crc_ok": True,
            "need_bytes": need_bytes,
            "slip": slip,
            "used_syms": diag_probe["num_syms"],
            "diag": diag_probe,
            "H": H,
            "snr_sc": snr_sc,
            "payload_start": payload_start
        }, {"diag": diag_probe, "H": H, "snr_sc": snr_sc}

    # Otherwise do bounded full demod if we learned LENGTH/need_bytes
    if need_bytes > 0 and slip >= 0:
        required_bits = need_bytes * 8 + slip
        required_syms = int(np.ceil(required_bits / BITS_PER_OFDM_SYM))
        required_syms = min(required_syms, max_payload_syms_cap)

        bits_full, diag_full = demod_payload_bits(
            rx_cfo, payload_start, H,
            num_syms=required_syms,
            kp=kp, ki=ki,
            repeat=repeat,
            bad_evm_th=bad_evm_th,
            bad_evm_patience=bad_evm_patience
        )
        if bits_full is None:
            return cr, None, {"diag": diag_probe, "H": H, "snr_sc": snr_sc}

        bb_full = packbits_with_slip(bits_full, slip)
        ok2, payload2, need2, why2, plen2, crc_rx2, crc_calc2 = parse_packet_bytes(bb_full)
        if ok2:
            return cr, {
                "payload": payload2,
                "crc_ok": True,
                "need_bytes": need2,
                "slip": slip,
                "used_syms": diag_full["num_syms"],
                "diag": diag_full,
                "H": H,
                "snr_sc": snr_sc,
                "payload_start": payload_start
            }, {"diag": diag_full, "H": H, "snr_sc": snr_sc}
        else:
            # return failure but with diag_full
            return cr, {
                "payload": b"",
                "crc_ok": False,
                "need_bytes": need2,
                "slip": slip,
                "used_syms": diag_full["num_syms"],
                "diag": diag_full,
                "H": H,
                "snr_sc": snr_sc,
                "payload_start": payload_start,
                "fail_reason": why2,
                "crc_rx": crc_rx2,
                "crc_calc": crc_calc2,
                "plen": plen2,
            }, {"diag": diag_full, "H": H, "snr_sc": snr_sc}

    return cr, None, {"diag": diag_probe, "H": H, "snr_sc": snr_sc}


def candidate_score(c: CandidateResult) -> float:
    """
    Higher is better.
    Priority:
      CRC OK > CRC mismatch / need_more > bad_magic
    Tie-break: lower probe_mean_evm, higher stf_peak.
    """
    base = 0.0
    if c.probe_status == "ok":
        base = 1000.0
    elif c.probe_status == "crc_mismatch":
        base = 500.0
    elif c.probe_status == "need_more":
        base = 300.0
    else:
        base = 0.0

    evm_term = 1.0 / (c.probe_mean_evm + 0.02)
    peak_term = min(50.0, c.stf_peak)
    return base + 5.0 * evm_term + 0.1 * peak_term


# ==============================
# Plotting
# ==============================
def plot_capture_diag(out_png: str,
                      rx: np.ndarray,
                      fs: float,
                      sc_metric: np.ndarray,
                      stf_idx: int,
                      title: str,
                      diag: dict):
    syms = diag.get("all_data_syms", np.array([], dtype=np.complex64))
    evm = diag.get("evm_per_sym", np.array([], dtype=np.float32))
    ph  = diag.get("phase_err", np.array([], dtype=np.float32))
    fr  = diag.get("freq_log", np.array([], dtype=np.float32))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) SC metric
    ax = axes[0, 0]
    if sc_metric is not None and sc_metric.size:
        L = min(len(sc_metric), 60000)
        ax.plot(sc_metric[:L], linewidth=0.8)
        if 0 <= stf_idx < L:
            ax.axvline(stf_idx, linestyle="--")
    ax.set_title("Schmidl-Cox metric (raw)")
    ax.grid(True)

    # (0,1) Spectrum snapshot
    ax = axes[0, 1]
    N = min(16384, len(rx))
    if N >= 1024:
        win = np.hanning(N)
        X = np.fft.fftshift(np.fft.fft(rx[:N] * win))
        fk = np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) / 1e3
        ax.plot(fk, 20*np.log10(np.abs(X) + 1e-12), linewidth=0.8)
        ax.set_xlim([-300, 300])
    ax.set_title("Spectrum (kHz)")
    ax.grid(True)

    # (1,0) Constellation subset
    ax = axes[1, 0]
    if syms.size:
        n = min(len(syms), 2000)
        ax.scatter(np.real(syms[:n]), np.imag(syms[:n]), s=3, alpha=0.5)
        ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        ax.scatter(np.real(ideal), np.imag(ideal), s=80, marker="x")
    ax.axis("equal")
    ax.set_title("Constellation (subset)")
    ax.grid(True)

    # (1,1) EVM / phase / freq
    ax = axes[1, 1]
    if evm.size:
        ax.plot(evm, "-o", markersize=3, label="EVM")
    if ph.size:
        ax.plot(np.degrees(ph), "-o", markersize=3, label="Pilot phase (deg)")
    if fr.size:
        ax.plot(fr, "-o", markersize=3, label="Freq acc")
    ax.grid(True)
    ax.legend()

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def plot_summary(out_png: str, rows: list[dict]):
    cfos = [r["cfo_hz"] for r in rows if r["final_status"] == "crc_ok"]
    evms = [r["mean_evm"] for r in rows if (r["final_status"] == "crc_ok" and r["mean_evm"] is not None)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if cfos:
        ax.plot(cfos, ".", markersize=3)
        ax.set_title(f"CFO (crc_ok) std={np.std(cfos):.1f} Hz")
    ax.grid(True)

    ax = axes[0, 1]
    if evms:
        ax.hist(evms, bins=30, edgecolor="black", alpha=0.8)
        ax.set_title(f"EVM (crc_ok) mean={np.mean(evms):.4f}")
    ax.grid(True)

    ax = axes[1, 0]
    # status counts
    mp = {}
    for r in rows:
        s = r["final_status"]
        mp[s] = mp.get(s, 0) + 1
    keys = sorted(mp.keys())
    ax.bar(keys, [mp[k] for k in keys])
    ax.set_title("Final status counts")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    txt = "\n".join([f"{k}: {mp[k]}" for k in keys])
    ax.text(0.05, 0.95, txt, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

    fig.suptitle("Step5 RX Stable Summary", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ==============================
# Main demod per capture
# ==============================
def demod_one_capture_stable(rx_raw_iq: np.ndarray,
                             fs: float,
                             stf_ref: np.ndarray,
                             ltf_freq_ref: np.ndarray,
                             cfg: dict):
    """
    Returns:
      final_status: crc_ok / crc_fail / no_magic / no_signal / ltf_fail / no_payload
      plus debug fields
    """
    rx = rx_raw_iq.astype(np.complex64)

    # If Pluto gives small numbers already, don't scale. If int14-like, scale.
    if np.median(np.abs(rx)) > 100:
        rx = rx / (2**14)

    rx = rx - np.mean(rx)
    peak = float(np.max(np.abs(rx))) if rx.size else 0.0

    # --- SC detect (on raw, no CFO needed) ---
    ok_sc, sc_idx, sc_peak, cfo_sc, ratio, sc_M = detect_stf_sc(
        rx, fs=fs,
        sc_threshold=cfg["sc_threshold"],
        search_len=cfg["search_len"],
        peak_med_ratio_th=cfg["sync_ratio"]
    )

    # CFO choose
    cfo_hz = cfo_sc
    if cfg["cfo_mode"] == "none":
        cfo_hz = 0.0

    # clip CFO if requested
    cfo_clip = float(cfg["cfo_clip_hz"])
    if cfo_clip > 0:
        if abs(cfo_hz) > cfo_clip:
            cfo_hz = float(np.clip(cfo_hz, -cfo_clip, cfo_clip))

    # Apply CFO
    rx_cfo = apply_cfo(rx, cfo_hz, fs)

    # --- STF refine ---
    stf_idx = sc_idx
    stf_peak = sc_peak
    method = "sc"

    if cfg["xcorr_refine"]:
        if ok_sc:
            w = int(cfg["xcorr_window"])
            s0 = max(0, sc_idx - w)
            s1 = min(len(rx_cfo), sc_idx + w + len(stf_ref))
            seg = rx_cfo[s0:s1]
            if len(seg) > len(stf_ref) + 10:
                corr = np.abs(np.correlate(seg, stf_ref, mode="valid"))
                corr /= (np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
                loc = int(np.argmax(corr))
                stf_idx = s0 + loc
                stf_peak = float(corr[loc])
                method = "sc+xc_refine"
        else:
            xi, xp = detect_stf_xcorr(rx_cfo, stf_ref, search_len=cfg["search_len"])
            stf_idx, stf_peak = xi, xp
            method = "xc_fallback"

    # gate
    if ok_sc:
        if ratio < cfg["sync_ratio"]:
            return {
                "final_status": "no_signal",
                "reason": "ratio_gate",
                "cfo_hz": float(cfo_hz),
                "ratio": float(ratio),
                "peak": peak,
                "stf_idx": int(stf_idx),
                "stf_peak": float(stf_peak),
                "method": method,
                "sc_metric": sc_M,
                "diag": {}
            }
    else:
        if stf_peak < cfg["xcorr_min_peak"]:
            return {
                "final_status": "no_signal",
                "reason": "xcorr_low",
                "cfo_hz": float(cfo_hz),
                "ratio": float(ratio),
                "peak": peak,
                "stf_idx": int(stf_idx),
                "stf_peak": float(stf_peak),
                "method": method,
                "sc_metric": sc_M,
                "diag": {}
            }

    # --- Candidate list (local fine search around stf_idx) ---
    # This helps when SC idx is slightly off; we test a small grid of offsets.
    cand_offsets = list(range(-cfg["cand_offset_span"], cfg["cand_offset_span"] + 1, cfg["cand_offset_step"]))
    cands = []

    for off in cand_offsets:
        cand_idx = int(stf_idx + off)
        if cand_idx < 0 or cand_idx + len(stf_ref) + cfg["ltf_symbols"] * SYMBOL_LEN + SYMBOL_LEN > len(rx_cfo):
            continue

        cr, final, aux = evaluate_candidate(
            rx_raw=rx,
            fs=fs,
            stf_ref=stf_ref,
            ltf_freq_ref=ltf_freq_ref,
            stf_idx=cand_idx,
            stf_peak=stf_peak,
            method=f"{method}+off{off}",
            cfo_hz=cfo_hz,
            ratio=ratio,
            ltf_symbols=cfg["ltf_symbols"],
            probe_syms=cfg["probe_syms"],
            repeat=cfg["repeat"],
            kp=cfg["kp"],
            ki=cfg["ki"],
            bad_evm_th=cfg["bad_evm_th"],
            bad_evm_patience=cfg["bad_evm_patience"],
            bit_slip_max=cfg["bit_slip_max"],
            max_payload_syms_cap=cfg["max_payload_syms_cap"]
        )
        if cr is None:
            continue
        cands.append((candidate_score(cr), cr, final, aux))

    if not cands:
        return {
            "final_status": "ltf_fail",
            "reason": "no_candidate",
            "cfo_hz": float(cfo_hz),
            "ratio": float(ratio),
            "peak": peak,
            "stf_idx": int(stf_idx),
            "stf_peak": float(stf_peak),
            "method": method,
            "sc_metric": sc_M,
            "diag": {}
        }

    cands.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cr, best_final, best_aux = cands[0]

    # Determine final outcome
    if best_final is not None and best_final.get("crc_ok", False):
        diag = best_final.get("diag", {})
        mean_evm = float(np.mean(diag.get("evm_per_sym", np.array([], dtype=np.float32)))) if diag.get("evm_per_sym", np.array([])).size else None
        return {
            "final_status": "crc_ok",
            "reason": "",
            "payload": best_final["payload"],
            "payload_len": len(best_final["payload"]),
            "cfo_hz": float(cfo_hz),
            "ratio": float(ratio),
            "peak": peak,
            "stf_idx": int(best_cr.stf_idx),
            "stf_peak": float(best_cr.stf_peak),
            "method": best_cr.method,
            "probe_status": best_cr.probe_status,
            "probe_slip": best_cr.probe_slip,
            "need_bytes": best_cr.probe_need_bytes,
            "mean_evm": mean_evm,
            "sc_metric": sc_M,
            "diag": diag
        }

    # If we had magic but CRC fail -> report crc_fail
    if best_cr.probe_status in ("crc_mismatch",) or (best_final and best_final.get("fail_reason") == "crc_mismatch"):
        diag = (best_final or {}).get("diag", best_aux.get("diag", {}))
        mean_evm = float(np.mean(diag.get("evm_per_sym", np.array([], dtype=np.float32)))) if diag.get("evm_per_sym", np.array([])).size else None
        return {
            "final_status": "crc_fail",
            "reason": (best_final or {}).get("fail_reason", "crc_mismatch"),
            "cfo_hz": float(cfo_hz),
            "ratio": float(ratio),
            "peak": peak,
            "stf_idx": int(best_cr.stf_idx),
            "stf_peak": float(best_cr.stf_peak),
            "method": best_cr.method,
            "probe_status": best_cr.probe_status,
            "probe_slip": best_cr.probe_slip,
            "need_bytes": best_cr.probe_need_bytes,
            "mean_evm": mean_evm,
            "sc_metric": sc_M,
            "diag": diag
        }

    # Otherwise no_magic / no_packet
    diag = (best_final or {}).get("diag", best_aux.get("diag", {}))
    mean_evm = float(np.mean(diag.get("evm_per_sym", np.array([], dtype=np.float32)))) if diag.get("evm_per_sym", np.array([])).size else None
    return {
        "final_status": "no_magic",
        "reason": best_cr.probe_status,
        "cfo_hz": float(cfo_hz),
        "ratio": float(ratio),
        "peak": peak,
        "stf_idx": int(best_cr.stf_idx),
        "stf_peak": float(best_cr.stf_peak),
        "method": best_cr.method,
        "probe_status": best_cr.probe_status,
        "probe_slip": best_cr.probe_slip,
        "need_bytes": best_cr.probe_need_bytes,
        "mean_evm": mean_evm,
        "sc_metric": sc_M,
        "diag": diag
    }


# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser("Step5 RX Stable (AIS1)")
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--rx_gain", type=float, default=30.0)
    ap.add_argument("--rx_bw", type=float, default=0.0, help="0 => auto use 1.2*fs")
    ap.add_argument("--buf_size", type=int, default=131072)
    ap.add_argument("--kernel_buffers", type=int, default=4)

    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.005)

    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--outfile", default="recovered.bin")
    ap.add_argument("--output_dir", default="rf_link_step5_results_stable")

    # CFO
    ap.add_argument("--cfo_mode", choices=["sc", "none"], default="sc")
    ap.add_argument("--cfo_clip_hz", type=float, default=20000.0)

    # SC detect
    ap.add_argument("--sc_threshold", type=float, default=0.08)
    ap.add_argument("--sync_ratio", type=float, default=6.0)
    ap.add_argument("--search_len", type=int, default=60000)

    # xcorr refine
    ap.add_argument("--xcorr_refine", action="store_true")
    ap.add_argument("--xcorr_window", type=int, default=2000)
    ap.add_argument("--xcorr_min_peak", type=float, default=0.02)

    # probe / bounded
    ap.add_argument("--probe_syms", type=int, default=16)
    ap.add_argument("--bit_slip_max", type=int, default=7)
    ap.add_argument("--max_payload_syms_cap", type=int, default=260)

    # EVM guard
    ap.add_argument("--bad_evm_th", type=float, default=0.95)
    ap.add_argument("--bad_evm_patience", type=int, default=6)

    # candidate offsets
    ap.add_argument("--cand_offset_span", type=int, default=8, help="+/- samples around stf_idx")
    ap.add_argument("--cand_offset_step", type=int, default=2)

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--no_csv", action="store_true")
    args = ap.parse_args()

    import adi

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    stf_ref, _ = create_schmidl_cox_stf(num_repeats=args.stf_repeats)
    _, ltf_freq_ref = create_ltf_ref(num_symbols=args.ltf_symbols)

    # SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    bw = args.rx_bw if args.rx_bw and args.rx_bw > 0 else float(args.fs) * 1.2
    sdr.rx_rf_bandwidth = int(bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf_size)
    sdr.rx_enabled_channels = [0]

    # kernel buffers
    try:
        if hasattr(sdr, "_rxadc") and sdr._rxadc is not None:
            sdr._rxadc.set_kernel_buffers_count(int(args.kernel_buffers))
    except Exception:
        pass

    # flush
    for _ in range(4):
        _ = sdr.rx()

    cfg = {
        "cfo_mode": args.cfo_mode,
        "cfo_clip_hz": float(args.cfo_clip_hz),
        "sc_threshold": float(args.sc_threshold),
        "sync_ratio": float(args.sync_ratio),
        "search_len": int(args.search_len),
        "xcorr_refine": bool(args.xcorr_refine),
        "xcorr_window": int(args.xcorr_window),
        "xcorr_min_peak": float(args.xcorr_min_peak),

        "ltf_symbols": int(args.ltf_symbols),
        "probe_syms": int(args.probe_syms),
        "bit_slip_max": int(args.bit_slip_max),
        "max_payload_syms_cap": int(args.max_payload_syms_cap),

        "repeat": int(args.repeat),
        "kp": float(args.kp),
        "ki": float(args.ki),

        "bad_evm_th": float(args.bad_evm_th),
        "bad_evm_patience": int(args.bad_evm_patience),

        "cand_offset_span": int(args.cand_offset_span),
        "cand_offset_step": int(args.cand_offset_step),
    }

    # CSV log
    rows = []
    csv_path = os.path.join(run_dir, "captures.csv")
    csv_f = None
    csv_w = None
    if not args.no_csv:
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.DictWriter(csv_f, fieldnames=[
            "try",
            "final_status",
            "reason",
            "cfo_hz",
            "ratio",
            "peak",
            "stf_idx",
            "stf_peak",
            "method",
            "probe_status",
            "probe_slip",
            "need_bytes",
            "mean_evm",
            "payload_len",
        ])
        csv_w.writeheader()

    print("\n" + "="*78)
    print("Step5 RX STABLE (AIS1)")
    print("="*78)
    print(f"run_dir={run_dir}")
    print(f"uri={args.uri} fc={args.fc/1e6:.1f}MHz fs={args.fs/1e6:.1f}Msps rx_gain={args.rx_gain}")
    print(f"buf={args.buf_size} kernel_buffers={args.kernel_buffers} bw={bw/1e6:.2f}MHz")
    print(f"stf_repeats={args.stf_repeats} ltf_symbols={args.ltf_symbols} repeat={args.repeat}")
    print(f"cfo_mode={args.cfo_mode} cfo_clip_hz={args.cfo_clip_hz}")
    print(f"probe_syms={args.probe_syms} bit_slip_max={args.bit_slip_max} max_payload_syms_cap={args.max_payload_syms_cap}")
    print("="*78)
    print(f"Starting {args.tries} captures ...")
    print("-"*78)

    t0 = time.time()
    success = False

    try:
        for ti in range(1, args.tries + 1):
            rx_raw = sdr.rx()
            r = demod_one_capture_stable(
                rx_raw_iq=rx_raw,
                fs=float(args.fs),
                stf_ref=stf_ref,
                ltf_freq_ref=ltf_freq_ref,
                cfg=cfg
            )

            final_status = r.get("final_status", "na")
            reason = r.get("reason", "")
            row = {
                "try": ti,
                "final_status": final_status,
                "reason": reason,
                "cfo_hz": float(r.get("cfo_hz", 0.0)),
                "ratio": float(r.get("ratio", 0.0)),
                "peak": float(r.get("peak", 0.0)),
                "stf_idx": int(r.get("stf_idx", -1)),
                "stf_peak": float(r.get("stf_peak", 0.0)),
                "method": r.get("method", ""),
                "probe_status": r.get("probe_status", ""),
                "probe_slip": int(r.get("probe_slip", -1)) if r.get("probe_slip", None) is not None else -1,
                "need_bytes": int(r.get("need_bytes", 0)) if r.get("need_bytes", None) is not None else 0,
                "mean_evm": float(r.get("mean_evm")) if r.get("mean_evm", None) is not None else None,
                "payload_len": int(r.get("payload_len", 0)) if r.get("payload_len", None) is not None else 0,
            }
            rows.append(row)
            if csv_w:
                csv_w.writerow(row)

            if args.verbose or ti <= 10 or (ti % 10 == 0):
                evm_str = f"{row['mean_evm']:.4f}" if row["mean_evm"] is not None else "-"
                print(f"[{ti:02d}] {final_status:8s} "
                      f"CFO={row['cfo_hz']:+8.1f}Hz ratio={row['ratio']:6.2f} peak={row['peak']:5.1f} "
                      f"slip={row['probe_slip']:2d} need={row['need_bytes']:4d} EVM={evm_str} "
                      f"{row['method']} {reason}")

            # Plot per capture if enabled (not too heavy for small tries)
            if not args.no_plots:
                diag = r.get("diag", {})
                out_png = os.path.join(run_dir, f"capture_{ti:02d}_{final_status}.png")
                title = (f"Capture {ti} - {final_status} "
                         f"CFO={row['cfo_hz']:+.1f}Hz ratio={row['ratio']:.2f} "
                         f"slip={row['probe_slip']} need={row['need_bytes']}")
                plot_capture_diag(out_png, (rx_raw.astype(np.complex64) / (2**14) if np.median(np.abs(rx_raw)) > 100 else rx_raw.astype(np.complex64)),
                                  float(args.fs), r.get("sc_metric", np.array([])), row["stf_idx"], title, diag)

            if final_status == "crc_ok":
                payload = r.get("payload", b"")
                with open(os.path.join(run_dir, args.outfile), "wb") as f:
                    f.write(payload)
                # print payload preview
                try:
                    print(f"  >>> CRC OK! payload={len(payload)}B saved={os.path.join(run_dir, args.outfile)}")
                    if len(payload) <= 128:
                        print(f"      Payload: {payload}")
                except Exception:
                    pass
                success = True
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        elapsed = time.time() - t0
        try:
            if csv_f:
                csv_f.close()
        except Exception:
            pass
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

        # Summary plot
        if not args.no_plots and rows:
            out_png = os.path.join(run_dir, "summary.png")
            plot_summary(out_png, rows)

        print("\n" + "="*78)
        print("SUMMARY")
        print("="*78)
        print(f"run_dir: {run_dir}")
        print(f"success: {success}")
        print(f"elapsed: {elapsed:.1f}s")
        if not args.no_csv:
            print(f"csv: {csv_path}")
        if not args.no_plots:
            print(f"plots: {run_dir}/capture_*.png and summary.png")
        print("="*78)


if __name__ == "__main__":
    main()

"""
python3 rf_step5_rx_stable.py \
  --uri ip:192.168.2.2 \
  --fc 2.3e9 --fs 3e6 \
  --rx_gain 30 \
  --buf_size 262144 \
  --tries 30 \
  --cfo_mode sc \
  --sc_threshold 0.08 \
  --probe_syms 16 \
  --xcorr_refine \
  --verbose
"""