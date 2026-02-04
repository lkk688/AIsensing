#!/usr/bin/env python3
"""
RF Link Test - Step 6: RX (Image Reception)

Receives multi-packet image transmission:
1. Continuously captures RX buffer
2. Demodulates using Step 5 robust pipeline (Schmidl-Cox, FLL+PLL)
3. Parses image packets with sequence numbers
4. Accumulates chunks, reassembles image when complete
5. Outputs detailed diagnostic figures and evaluation metrics

Packet format (must match TX):
  MAGIC("IMG6", 4B) | SEQ(2B) | TOTAL(2B) | LEN(2B) | PAYLOAD(var) | CRC32(4B)

Run on the local RX device.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import zlib
import time

# ============================================================================
# OFDM Parameters (must match TX)
# ============================================================================
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in PILOT_SUBCARRIERS])
N_DATA = len(DATA_SUBCARRIERS)  # 48
SYMBOL_LEN = N_FFT + N_CP       # 80
BITS_PER_OFDM_SYM = N_DATA * 2  # 96

MAGIC = b"IMG6"


# ============================================================================
# Signal Processing Functions (from Step 5)
# ============================================================================

def estimate_cfo_from_tone(rx_samples, fs, expected_freq=100e3):
    """Estimate CFO from pilot tone using FFT peak with parabolic interpolation."""
    N = len(rx_samples)
    window = np.hanning(N).astype(np.float32)
    fft = np.fft.fft(rx_samples * window)
    fft_mag = np.abs(fft)
    fft_mag[0] = 0
    fft_mag[1:30] = 0
    fft_mag[-30:] = 0
    peak_idx = np.argmax(fft_mag)
    freq_bins = np.fft.fftfreq(N, 1 / fs)
    if 1 < peak_idx < N - 1:
        alpha = np.log(fft_mag[peak_idx - 1] + 1e-10)
        beta = np.log(fft_mag[peak_idx] + 1e-10)
        gamma = np.log(fft_mag[peak_idx + 1] + 1e-10)
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            p = 0.5 * (alpha - gamma) / denom
        else:
            p = 0.0
        detected_freq = freq_bins[peak_idx] + p * (freq_bins[1] - freq_bins[0])
    else:
        detected_freq = freq_bins[peak_idx]
    tone_snr = fft_mag[peak_idx] / (np.median(fft_mag) + 1e-10)
    return detected_freq - expected_freq, tone_snr


def apply_cfo_correction(samples, cfo, fs):
    """Apply CFO correction to entire buffer."""
    n = np.arange(len(samples))
    return samples * np.exp(-1j * 2 * np.pi * cfo * n / fs).astype(np.complex64)


# ============================================================================
# Preamble Generation (must match TX exactly)
# ============================================================================

def create_schmidl_cox_stf(N=64, num_repeats=6):
    """Create Schmidl-Cox STF with period N/2 (must match TX)."""
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0])
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    stf = np.tile(x, num_repeats).astype(np.complex64)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp, X


def create_ltf_ref(N=64, num_symbols=4):
    """Create LTF reference (must match TX)."""
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if i % 2 == 0 else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X


# ============================================================================
# Robust Timing Sync: Schmidl-Cox Autocorrelation
# ============================================================================

def schmidl_cox_metric(rx, half_period=32, window_len=None):
    """Compute Schmidl-Cox timing metric."""
    L = half_period
    N = len(rx)
    if window_len is None:
        window_len = N - 2 * L
    M = np.zeros(window_len, dtype=np.float32)
    P_arr = np.zeros(window_len, dtype=np.complex64)
    R_arr = np.zeros(window_len, dtype=np.float32)
    for n in range(window_len):
        if n + 2 * L > N:
            break
        seg1 = rx[n:n + L]
        seg2 = rx[n + L:n + 2 * L]
        P_arr[n] = np.sum(seg1 * np.conj(seg2))
        R_arr[n] = np.sum(np.abs(seg2)**2)
    eps = 1e-10
    M = np.abs(P_arr)**2 / (R_arr**2 + eps)
    return M, P_arr, R_arr


def detect_stf_autocorr(rx, half_period=32, threshold=0.3):
    """Detect STF using Schmidl-Cox autocorrelation."""
    search_len = min(60000, len(rx) - 2 * half_period)
    M, P, R = schmidl_cox_metric(rx, half_period, search_len)
    plateau_len = 2 * half_period
    if len(M) < plateau_len:
        return False, 0, 0.0, 0.0, M
    kernel = np.ones(plateau_len) / plateau_len
    M_smooth = np.convolve(M, kernel, mode='valid')
    peak_smooth_idx = np.argmax(M_smooth)
    peak_val = float(M_smooth[peak_smooth_idx])
    start_idx = peak_smooth_idx
    for i in range(peak_smooth_idx, max(0, peak_smooth_idx - 5 * half_period), -1):
        if M[i] < 0.1 * peak_val:
            start_idx = i + 1
            break
    if start_idx + half_period < len(P):
        p_sum = np.sum(P[start_idx:start_idx + 2 * half_period])
        cfo_phase = np.angle(p_sum)
        cfo_norm = -cfo_phase / (2 * np.pi * half_period)
    else:
        cfo_norm = 0.0
    return peak_val > threshold, int(start_idx), peak_val, cfo_norm, M


def detect_stf_crosscorr(rx, stf_ref, search_range=60000):
    """Cross-correlation STF detection (backup method)."""
    search_len = min(search_range, len(rx) - len(stf_ref))
    if search_len <= 0:
        return -1, 0.0, np.array([])
    corr = np.abs(np.correlate(rx[:search_len + len(stf_ref)], stf_ref, mode='valid'))
    stf_energy = np.sqrt(np.sum(np.abs(stf_ref)**2))
    corr_norm = corr / (stf_energy + 1e-10)
    peak_idx = np.argmax(corr_norm)
    return int(peak_idx), float(corr_norm[peak_idx]), corr_norm


def detect_frame_by_energy(rx, fs, tone_gap=1000, stf_len=400):
    """Detect frame structure using energy envelope."""
    win = 500
    energy = np.abs(rx)**2
    if len(energy) < win:
        return []
    cumsum = np.cumsum(energy)
    cumsum = np.insert(cumsum, 0, 0)
    avg_energy = (cumsum[win:] - cumsum[:-win]) / win
    noise_floor = np.percentile(avg_energy, 10)
    signal_thresh = noise_floor * 10
    is_signal = avg_energy > signal_thresh
    candidates = []
    for i in range(1, len(is_signal) - tone_gap - stf_len):
        if is_signal[i] and not is_signal[i + 1]:
            for j in range(i + 1, min(i + 3000, len(is_signal))):
                if is_signal[j] and not is_signal[j - 1]:
                    stf_candidate = j + win // 2
                    candidates.append(stf_candidate)
                    break
    return candidates


# ============================================================================
# Channel Estimation and OFDM Demodulation
# ============================================================================

def extract_ofdm_symbol(rx, start_idx):
    """Extract one OFDM symbol frequency domain, removing CP."""
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    sym = rx[start_idx + N_CP:start_idx + SYMBOL_LEN]
    return np.fft.fftshift(np.fft.fft(sym))


def channel_estimate_from_ltf(rx, ltf_start, ltf_freq_ref, num_symbols=4):
    """Estimate channel from multiple LTF symbols."""
    used = np.array([k for k in range(-26, 27) if k != 0])
    used_idx = np.array([(k + N_FFT) % N_FFT for k in used])
    Ys = []
    for i in range(num_symbols):
        Y = extract_ofdm_symbol(rx, ltf_start + i * SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if len(Ys) == 0:
        return None, None
    Y_avg = np.mean(Ys, axis=0)
    H = np.ones(N_FFT, dtype=np.complex64)
    for k in used:
        idx = (k + N_FFT) % N_FFT
        if np.abs(ltf_freq_ref[idx]) > 0.1:
            H[idx] = Y_avg[idx] / ltf_freq_ref[idx]
    if len(Ys) >= 2:
        noise_var = np.var(np.array(Ys)[:, used_idx], axis=0)
        signal_var = np.abs(Y_avg[used_idx])**2
        snr_per_sc = 10 * np.log10(signal_var / (noise_var + 1e-10) + 1e-10)
    else:
        snr_per_sc = np.zeros(len(used))
    return H, snr_per_sc


def qpsk_demap(symbols):
    """Demap QPSK symbols to bits."""
    bits = np.zeros(len(symbols) * 2, dtype=np.uint8)
    for i, s in enumerate(symbols):
        re = np.real(s) >= 0
        im = np.imag(s) >= 0
        if re and im:
            bits[2 * i], bits[2 * i + 1] = 0, 0
        elif not re and im:
            bits[2 * i], bits[2 * i + 1] = 0, 1
        elif not re and not im:
            bits[2 * i], bits[2 * i + 1] = 1, 1
        else:
            bits[2 * i], bits[2 * i + 1] = 1, 0
    return bits


def majority_vote(bits, repeat):
    """Apply majority voting for repeated bits."""
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    bits = bits[:L].reshape(-1, repeat)
    return (np.sum(bits, axis=1) >= (repeat / 2)).astype(np.uint8)


# ============================================================================
# Image Packet Parsing
# ============================================================================

def parse_image_packet(data: bytes):
    """
    Parse image packet from demodulated bytes.
    Format: MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    CRC covers MAGIC through PAYLOAD.

    Returns: (success, seq, total, payload)
    """
    if len(data) < 14:
        return False, -1, 0, b""
    if data[:4] != MAGIC:
        return False, -1, 0, b""
    try:
        seq = int.from_bytes(data[4:6], "little")
        total = int.from_bytes(data[6:8], "little")
        plen = int.from_bytes(data[8:10], "little")
        expected_len = 10 + plen + 4
        if len(data) < expected_len:
            return False, seq, total, b""
        content = data[:10 + plen]
        payload = data[10:10 + plen]
        crc_rx = int.from_bytes(data[10 + plen:10 + plen + 4], "little")
        crc_calc = zlib.crc32(content) & 0xFFFFFFFF
        if crc_rx == crc_calc:
            return True, seq, total, payload
        else:
            return False, seq, total, b""
    except Exception:
        return False, -1, 0, b""


# ============================================================================
# Demodulation Pipeline (from Step 5, adapted for image packets)
# ============================================================================

def demod_one_capture(
    rx_raw, fs, stf_ref, ltf_freq_ref,
    stf_repeats, ltf_symbols, max_ofdm_syms,
    kp, ki, repeat
):
    """
    Full demodulation pipeline for one RX capture.
    Returns raw demodulated bytes (packet parsing done by caller).
    """
    rx = rx_raw.astype(np.complex64) / (2**14)
    rx = rx - np.mean(rx)

    # ---- Multi-segment CFO estimation ----
    seg_len = 20000
    best_cfo = 0.0
    best_snr = 0.0
    for seg_start in range(0, min(len(rx) - seg_len, 80000), 5000):
        seg = rx[seg_start:seg_start + seg_len]
        c, s = estimate_cfo_from_tone(seg, fs, 100e3)
        if s > best_snr:
            best_snr = s
            best_cfo = c

    cfo = best_cfo
    tone_snr = best_snr
    rx_cfo = apply_cfo_correction(rx, cfo, fs)

    # ---- STF detection: Schmidl-Cox ----
    sc_detected, sc_idx, sc_peak, sc_cfo_norm, sc_M = detect_stf_autocorr(
        rx_cfo, half_period=N_FFT // 2, threshold=0.1
    )

    # ---- STF detection: cross-correlation ----
    xc_idx, xc_peak, xc_corr = detect_stf_crosscorr(rx_cfo, stf_ref)

    # ---- Combined detection ----
    energy_candidates = detect_frame_by_energy(rx_cfo, fs)
    all_candidates = list(energy_candidates)
    if sc_detected:
        all_candidates.append(sc_idx)
    if xc_peak > 0.02:
        all_candidates.append(xc_idx)

    used = np.array([k for k in range(-26, 27) if k != 0])
    used_ltf_idx = np.array([(k + N_FFT) % N_FFT for k in used])
    pilot_pattern_ref = np.array([1, 1, 1, -1], dtype=np.complex64)
    p_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS])
    d_idx = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS])

    best_stf_idx = -1
    best_score = -1.0
    best_xc_peak = 0.0
    best_method = "none"

    for cand in all_candidates:
        search_start = max(0, cand - 500)
        search_end = min(len(rx_cfo) - len(stf_ref), cand + 500)
        if search_end <= search_start:
            continue
        local_seg = rx_cfo[search_start:search_end + len(stf_ref)]
        local_corr = np.abs(np.correlate(local_seg, stf_ref, mode='valid'))
        stf_e = np.sqrt(np.sum(np.abs(stf_ref)**2))
        local_corr_norm = local_corr / (stf_e + 1e-10)
        local_peak_idx = np.argmax(local_corr_norm)
        refined_idx = search_start + int(local_peak_idx)
        refined_peak = float(local_corr_norm[local_peak_idx])

        # Fine-tune with LTF quality sweep
        ltf_base = refined_idx + len(stf_ref)
        best_fine = 0
        best_fine_q = -1.0
        for off in range(-4, 5):
            Yt = extract_ofdm_symbol(rx_cfo, ltf_base + off)
            if Yt is None:
                continue
            Ht = np.zeros(N_FFT, dtype=np.complex64)
            for k in used:
                idx_k = (k + N_FFT) % N_FFT
                if np.abs(ltf_freq_ref[idx_k]) > 0.1:
                    Ht[idx_k] = Yt[idx_k] / ltf_freq_ref[idx_k]
            H_m = np.abs(Ht[used_ltf_idx])
            q = float(np.mean(H_m)**2 / (np.var(H_m) + 1e-10))
            if q > best_fine_q:
                best_fine_q = q
                best_fine = off
        refined_idx += best_fine

        # Trial demodulation
        ltf_s = refined_idx + len(stf_ref)
        Y_ltf = extract_ofdm_symbol(rx_cfo, ltf_s)
        if Y_ltf is None:
            continue
        H_trial = np.ones(N_FFT, dtype=np.complex64)
        for k in used:
            idx_k = (k + N_FFT) % N_FFT
            if np.abs(ltf_freq_ref[idx_k]) > 0.1:
                H_trial[idx_k] = Y_ltf[idx_k] / ltf_freq_ref[idx_k]

        pay_start = ltf_s + ltf_symbols * SYMBOL_LEN
        trial_evm = []
        trial_ph = 0.0
        trial_freq = 0.0
        for si in range(min(5, max_ofdm_syms)):
            Ys = extract_ofdm_symbol(rx_cfo, pay_start + si * SYMBOL_LEN)
            if Ys is None:
                break
            Ye = np.zeros_like(Ys)
            for k in range(-26, 27):
                if k == 0:
                    continue
                idx_k = (k + N_FFT) % N_FFT
                if np.abs(H_trial[idx_k]) > 1e-6:
                    Ye[idx_k] = Ys[idx_k] / H_trial[idx_k]
            ps = 1 if si % 2 == 0 else -1
            pe = np.angle(np.sum(Ye[p_idx] * np.conj(ps * pilot_pattern_ref)))
            trial_freq += 0.01 * pe
            trial_ph += trial_freq + 0.1 * pe
            Ye *= np.exp(-1j * trial_ph)
            ideal_pts = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
            nearest = np.array([ideal_pts[np.argmin(np.abs(s - ideal_pts))] for s in Ye[d_idx]])
            trial_evm.append(float(np.sqrt(np.mean(np.abs(Ye[d_idx] - nearest)**2))))

        if len(trial_evm) == 0:
            continue
        avg_evm = np.mean(trial_evm)
        score = 1.0 / (avg_evm + 0.01)
        if score > best_score:
            best_score = score
            best_stf_idx = refined_idx
            best_xc_peak = refined_peak
            best_method = f"trial(evm={avg_evm:.3f})"

    if best_stf_idx >= 0:
        stf_idx = best_stf_idx
        stf_peak = best_xc_peak
        method = best_method
    elif xc_peak > 0.02:
        stf_idx = xc_idx
        stf_peak = xc_peak
        method = "xcorr-fallback"
    else:
        stf_idx = sc_idx
        stf_peak = sc_peak
        method = "sc-fallback"

    result = {
        'cfo': cfo, 'tone_snr': tone_snr,
        'stf_idx': stf_idx, 'stf_peak': stf_peak, 'stf_method': method,
        'sc_peak': sc_peak, 'sc_cfo_norm': sc_cfo_norm,
        'xc_peak': xc_peak, 'sc_metric': sc_M,
        'xc_metric': xc_corr, 'xc_idx': xc_idx,
        'rx': rx, 'rx_cfo': rx_cfo,
    }

    if stf_peak < 0.01:
        result['status'] = 'no_signal'
        result['H'] = np.ones(N_FFT, dtype=np.complex64)
        result['snr_per_sc'] = np.zeros(52)
        result['all_data_syms'] = np.array([], dtype=np.complex64)
        result['phase_errors'] = np.array([])
        result['freq_log'] = np.array([])
        result['pilot_powers'] = np.array([])
        result['evm_per_sym'] = np.array([])
        result['bits_bytes'] = b""
        return result

    # ---- Channel estimation ----
    ltf_start = stf_idx + len(stf_ref)
    ch_result = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref, ltf_symbols)

    if ch_result[0] is None:
        result['status'] = 'ltf_fail'
        result['H'] = np.ones(N_FFT, dtype=np.complex64)
        result['snr_per_sc'] = np.zeros(52)
        result['all_data_syms'] = np.array([], dtype=np.complex64)
        result['phase_errors'] = np.array([])
        result['freq_log'] = np.array([])
        result['pilot_powers'] = np.array([])
        result['evm_per_sym'] = np.array([])
        result['bits_bytes'] = b""
        return result

    H, snr_per_sc = ch_result
    result['H'] = H
    result['snr_per_sc'] = snr_per_sc

    # ---- Demodulate with FLL+PLL ----
    payload_start = ltf_start + ltf_symbols * SYMBOL_LEN
    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
    pilot_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS])
    data_idx = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS])

    all_data_syms = []
    phase_errors = []
    freq_log = []
    pilot_powers = []
    evm_per_sym = []
    phase_acc = 0.0
    freq_acc = 0.0
    prev_pilot_phase = None
    bad_evm_count = 0
    EVM_CUTOFF = 0.8

    for sym_idx in range(max_ofdm_syms):
        sym_start = payload_start + sym_idx * SYMBOL_LEN
        Y = extract_ofdm_symbol(rx_cfo, sym_start)
        if Y is None:
            break
        Y_eq = np.zeros_like(Y)
        for k in range(-26, 27):
            if k == 0:
                continue
            idx = (k + N_FFT) % N_FFT
            if np.abs(H[idx]) > 1e-6:
                Y_eq[idx] = Y[idx] / H[idx]

        pilot_sign = 1 if sym_idx % 2 == 0 else -1
        expected_pilots = pilot_sign * pilot_pattern
        rx_pilots = Y_eq[pilot_idx]
        pilot_corr = np.sum(rx_pilots * np.conj(expected_pilots))
        phase_err = np.angle(pilot_corr)

        if prev_pilot_phase is not None:
            freq_err = phase_err - prev_pilot_phase
            while freq_err > np.pi:
                freq_err -= 2 * np.pi
            while freq_err < -np.pi:
                freq_err += 2 * np.pi
            freq_acc += ki * freq_err
        prev_pilot_phase = phase_err
        phase_acc += freq_acc + kp * phase_err
        Y_eq *= np.exp(-1j * phase_acc)

        data_syms = Y_eq[data_idx]
        ideal_pts = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        nearest = np.array([ideal_pts[np.argmin(np.abs(s - ideal_pts))] for s in data_syms])
        sym_evm = float(np.sqrt(np.mean(np.abs(data_syms - nearest)**2)))

        if sym_evm > EVM_CUTOFF:
            bad_evm_count += 1
            if bad_evm_count >= 3:
                break
        else:
            bad_evm_count = 0

        all_data_syms.append(data_syms)
        phase_errors.append(phase_err)
        freq_log.append(freq_acc)
        pilot_powers.append(float(np.mean(np.abs(rx_pilots)**2)))
        evm_per_sym.append(sym_evm)

    result['all_data_syms'] = np.concatenate(all_data_syms) if all_data_syms else np.array([], dtype=np.complex64)
    result['phase_errors'] = np.array(phase_errors)
    result['freq_log'] = np.array(freq_log)
    result['pilot_powers'] = np.array(pilot_powers)
    result['evm_per_sym'] = np.array(evm_per_sym)

    if len(all_data_syms) == 0:
        result['status'] = 'no_payload'
        result['bits_bytes'] = b""
        return result

    # Bit recovery -> raw bytes (packet parsing done by caller)
    bits_raw = qpsk_demap(result['all_data_syms'])
    bits = majority_vote(bits_raw, repeat)
    bits_bytes = np.packbits(bits).tobytes()
    result['bits_bytes'] = bits_bytes
    result['status'] = 'demod_ok'
    result['mean_evm'] = float(np.mean(evm_per_sym)) if evm_per_sym else 0.0
    return result


# ============================================================================
# Diagnostic Plot Generation
# ============================================================================

def plot_capture_diagnostics(
    r, capture_id, output_dir, fs,
    pkt_status="", progress_str=""
):
    """Generate 4x3 diagnostic plot for one capture (adapted from Step 5)."""
    rx = r['rx']
    rx_cfo = r['rx_cfo']
    cfo = r['cfo']
    stf_idx = r['stf_idx']
    stf_peak = r['stf_peak']
    sc_metric = r['sc_metric']
    xcorr_metric = r['xc_metric']
    xcorr_idx = r['xc_idx']
    H = r['H']
    snr_per_sc = r['snr_per_sc']
    all_data_syms = r['all_data_syms']
    phase_errors = r['phase_errors']
    freq_log = r['freq_log']
    evm_per_sym = r['evm_per_sym']

    fig, axes = plt.subplots(4, 3, figsize=(20, 22))

    # (0,0) Power envelope
    ax = axes[0, 0]
    power_env = np.abs(rx[:min(60000, len(rx))])**2
    ds = max(1, len(power_env) // 5000)
    ax.plot(np.arange(0, len(power_env), ds) / fs * 1000, power_env[::ds])
    if stf_idx > 0:
        ax.axvline(stf_idx / fs * 1000, color='r', linestyle='--', label=f'STF @{stf_idx}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power')
    ax.set_title('Signal Power Envelope')
    ax.legend(fontsize=8)
    ax.grid(True)

    # (0,1) Spectrum
    ax = axes[0, 1]
    N_fft = min(16384, len(rx))
    win = np.hanning(N_fft)
    fft_before = np.fft.fftshift(np.fft.fft(rx[:N_fft] * win))
    fft_after = np.fft.fftshift(np.fft.fft(rx_cfo[:N_fft] * win))
    freq_kHz = np.fft.fftshift(np.fft.fftfreq(N_fft, 1 / fs)) / 1e3
    ax.plot(freq_kHz, 20 * np.log10(np.abs(fft_before) + 1e-10), alpha=0.6, label='Before CFO')
    ax.plot(freq_kHz, 20 * np.log10(np.abs(fft_after) + 1e-10), alpha=0.6, label='After CFO')
    ax.axvline(100, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Spectrum (CFO = {cfo:.1f} Hz)')
    ax.legend(fontsize=8)
    ax.set_xlim([-300, 300])
    ax.grid(True)

    # (0,2) Schmidl-Cox
    ax = axes[0, 2]
    disp_len = min(50000, len(sc_metric))
    ax.plot(sc_metric[:disp_len])
    if 0 < stf_idx < disp_len:
        ax.axvline(stf_idx, color='r', linestyle='--')
    ax.set_xlabel('Sample')
    ax.set_title(f'SC Autocorrelation (peak={stf_peak:.4f})')
    ax.grid(True)

    # (1,0) Cross-correlation
    ax = axes[1, 0]
    if xcorr_metric is not None and len(xcorr_metric) > 0:
        disp2 = min(50000, len(xcorr_metric))
        ax.plot(xcorr_metric[:disp2])
        if 0 <= xcorr_idx < disp2:
            ax.axvline(xcorr_idx, color='r', linestyle='--')
    ax.set_xlabel('Sample')
    ax.set_title('Cross-Correlation STF')
    ax.grid(True)

    # (1,1) Channel estimate
    ax = axes[1, 1]
    H_shift = np.fft.fftshift(H)
    H_mag = 20 * np.log10(np.abs(H_shift) + 1e-10)
    sc_axis = np.arange(-N_FFT // 2, N_FFT // 2)
    ax.plot(sc_axis, H_mag, 'b-')
    ax.set_xlabel('Subcarrier')
    ax.set_ylabel('|H| (dB)')
    ax.set_title('Channel Estimate')
    ax.grid(True)

    # (1,2) SNR per subcarrier
    ax = axes[1, 2]
    if snr_per_sc is not None and len(snr_per_sc) > 0:
        used_sc = np.array([k for k in range(-26, 27) if k != 0])
        ax.bar(used_sc, snr_per_sc, width=0.8, alpha=0.7)
        ax.axhline(np.mean(snr_per_sc), color='r', linestyle='--',
                    label=f'Mean={np.mean(snr_per_sc):.1f} dB')
        ax.legend(fontsize=8)
    ax.set_xlabel('Subcarrier')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Per-Subcarrier SNR')
    ax.grid(True)

    # (2,0) Constellation
    ax = axes[2, 0]
    if all_data_syms is not None and len(all_data_syms) > 0:
        n_syms = len(all_data_syms)
        colors = np.arange(n_syms)
        sc_plot = ax.scatter(np.real(all_data_syms), np.imag(all_data_syms),
                             c=colors, cmap='viridis', s=3, alpha=0.6)
        plt.colorbar(sc_plot, ax=ax, label='Symbol Index')
        ideal = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        ax.scatter(np.real(ideal), np.imag(ideal), c='red', s=100, marker='x', linewidths=2)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.axvline(0, color='k', linewidth=0.3)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title('QPSK Constellation')
    ax.axis('equal')
    ax.grid(True)

    # (2,1) EVM per symbol
    ax = axes[2, 1]
    if evm_per_sym is not None and len(evm_per_sym) > 0:
        ax.plot(evm_per_sym, 'b-', alpha=0.7)
        ax.axhline(np.mean(evm_per_sym), color='r', linestyle='--',
                    label=f'Mean={np.mean(evm_per_sym):.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('OFDM Symbol')
    ax.set_ylabel('EVM')
    ax.set_title('EVM per OFDM Symbol')
    ax.grid(True)

    # (2,2) Phase histogram
    ax = axes[2, 2]
    if all_data_syms is not None and len(all_data_syms) > 0:
        angles = np.degrees(np.angle(all_data_syms))
        ax.hist(angles, bins=72, range=(-180, 180), alpha=0.7, edgecolor='black', linewidth=0.3)
        for a in [45, 135, -135, -45]:
            ax.axvline(a, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Phase (deg)')
    ax.set_title('Constellation Angle Histogram')
    ax.grid(True)

    # (3,0) Phase error
    ax = axes[3, 0]
    if phase_errors is not None and len(phase_errors) > 0:
        ax.plot(np.degrees(phase_errors), 'b-', alpha=0.7, label='Raw')
        ax.plot(np.degrees(np.unwrap(phase_errors)), 'r-', alpha=0.7, label='Unwrapped')
        ax.legend(fontsize=8)
    ax.set_xlabel('OFDM Symbol')
    ax.set_ylabel('Phase Error (deg)')
    ax.set_title('Pilot Phase Error')
    ax.grid(True)

    # (3,1) Freq tracking
    ax = axes[3, 1]
    if freq_log is not None and len(freq_log) > 0:
        ax.plot(freq_log, 'b-')
    ax.set_xlabel('OFDM Symbol')
    ax.set_ylabel('Freq Acc (rad/sym)')
    ax.set_title('Residual CFO Tracking')
    ax.grid(True)

    # (3,2) Summary text
    ax = axes[3, 2]
    ax.axis('off')
    mean_evm = r.get('mean_evm', 0)
    mean_snr = np.mean(snr_per_sc) if snr_per_sc is not None and len(snr_per_sc) > 0 else 0
    n_ofdm = len(all_data_syms) // N_DATA if all_data_syms is not None and len(all_data_syms) > 0 else 0
    summary = f"""
{pkt_status}

Signal:
  RX Power: {10 * np.log10(np.mean(np.abs(rx)**2) + 1e-10):.1f} dB
CFO: {cfo:.1f} Hz
Sync:
  SC peak: {r['sc_peak']:.4f}
  Method: {r['stf_method']}
  STF index: {stf_idx}
Channel:
  Mean SNR: {mean_snr:.1f} dB
  Mean EVM: {mean_evm:.4f}
  OFDM syms: {n_ofdm}

{progress_str}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'RF Link Step 6 - Capture {capture_id}', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = os.path.join(output_dir, f'capture_{capture_id:03d}.png')
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def plot_reception_summary(
    received_chunks, total_expected, per_pkt_stats,
    output_dir, output_image, start_time, end_time
):
    """Generate image reception summary figure (2x3)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (0,0) Packet reception map
    ax = axes[0, 0]
    colors = []
    for i in range(total_expected):
        if i in received_chunks:
            colors.append('green')
        else:
            colors.append('red')
    ax.barh(range(total_expected), [1] * total_expected, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Received')
    ax.set_ylabel('Packet Seq')
    ax.set_title(f'Packet Map ({len(received_chunks)}/{total_expected})')
    ax.set_xlim([0, 1.2])

    # (0,1) EVM per packet
    ax = axes[0, 1]
    if per_pkt_stats:
        seqs = sorted(per_pkt_stats.keys())
        evms = [per_pkt_stats[s]['evm'] for s in seqs]
        ax.bar(seqs, evms, color='steelblue', alpha=0.8)
        ax.axhline(np.mean(evms), color='r', linestyle='--',
                    label=f'Mean={np.mean(evms):.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Packet Seq')
    ax.set_ylabel('Mean EVM')
    ax.set_title('EVM per Received Packet')
    ax.grid(True)

    # (0,2) CFO over time
    ax = axes[0, 2]
    if per_pkt_stats:
        seqs = sorted(per_pkt_stats.keys())
        cfos = [per_pkt_stats[s]['cfo'] for s in seqs]
        times = [per_pkt_stats[s]['time'] - start_time for s in seqs]
        ax.plot(times, cfos, 'ro-', markersize=4)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('CFO (Hz)')
        ax.set_title(f'CFO per Packet (std={np.std(cfos):.1f} Hz)')
        ax.grid(True)

    # (1,0) Throughput timeline
    ax = axes[1, 0]
    if per_pkt_stats:
        seqs = sorted(per_pkt_stats.keys())
        times = [per_pkt_stats[s]['time'] - start_time for s in seqs]
        cum_bytes = np.cumsum([per_pkt_stats[s]['payload_len'] for s in seqs])
        ax.plot(times, cum_bytes / 1024, 'b-o', markersize=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cumulative (KB)')
        total_time = max(times) - min(times) if len(times) > 1 else 1
        throughput = cum_bytes[-1] / total_time if total_time > 0 else 0
        ax.set_title(f'Throughput ({throughput:.0f} bytes/s)')
        ax.grid(True)

    # (1,1) Received image thumbnail
    ax = axes[1, 1]
    if os.path.isfile(output_image):
        try:
            img = plt.imread(output_image)
            ax.imshow(img)
            ax.set_title(f'Received Image ({os.path.getsize(output_image)} bytes)')
        except Exception:
            ax.text(0.5, 0.5, 'Image decode failed', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Received Image (FAILED)')
    else:
        ax.text(0.5, 0.5, 'Incomplete', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Image Not Complete')
    ax.axis('off')

    # (1,2) Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    elapsed = end_time - start_time
    n_received = len(received_chunks)
    completion = n_received / total_expected * 100 if total_expected > 0 else 0
    total_bytes = sum(len(v) for v in received_chunks.values())

    all_evms = [per_pkt_stats[s]['evm'] for s in per_pkt_stats] if per_pkt_stats else [0]
    all_cfos = [per_pkt_stats[s]['cfo'] for s in per_pkt_stats] if per_pkt_stats else [0]

    stats_text = f"""
IMAGE RECEPTION SUMMARY

Packets: {n_received} / {total_expected} ({completion:.0f}%)
Missing: {total_expected - n_received}
Total bytes: {total_bytes}
Elapsed: {elapsed:.1f}s

EVM:
  Mean: {np.mean(all_evms):.4f}
  Min: {np.min(all_evms):.4f}
  Max: {np.max(all_evms):.4f}

CFO:
  Mean: {np.mean(all_cfos):.1f} Hz
  Std: {np.std(all_cfos):.1f} Hz

Output: {output_image}
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if completion == 100 else 'lightyellow',
                      alpha=0.5))

    fig.suptitle('RF Link Step 6 - Image Reception Summary', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'reception_summary.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="RF Link Test - Step 6: Image RX")
    ap.add_argument("--uri", default="ip:192.168.2.2", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--rx_gain", type=float, default=40, help="RX gain (dB)")
    ap.add_argument("--buf_size", type=int, default=2**17, help="RX buffer size")
    ap.add_argument("--max_ofdm_syms", type=int, default=120, help="Max OFDM symbols")
    ap.add_argument("--stf_repeats", type=int, default=6, help="STF repeats (must match TX)")
    ap.add_argument("--ltf_symbols", type=int, default=4, help="LTF symbols (must match TX)")
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Bit repetition")
    ap.add_argument("--kp", type=float, default=0.15, help="PLL Kp")
    ap.add_argument("--ki", type=float, default=0.005, help="FLL Ki")
    ap.add_argument("--max_captures", type=int, default=500, help="Max capture attempts")
    ap.add_argument("--output_dir", default="rf_link_step6_results", help="Output directory")
    ap.add_argument("--output_image", default="received_image.jpg", help="Output image file")
    ap.add_argument("--plot_every", type=int, default=5, help="Plot diagnostics every N captures")
    args = ap.parse_args()

    import adi

    os.makedirs(args.output_dir, exist_ok=True)

    # Create reference signals
    stf_ref, stf_freq = create_schmidl_cox_stf(N_FFT, num_repeats=args.stf_repeats)
    ltf_ref, ltf_freq_ref = create_ltf_ref(N_FFT, num_symbols=args.ltf_symbols)

    print(f"RF Link Test - Step 6: Image RX")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc / 1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs / 1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
    print(f"  Max captures: {args.max_captures}")
    print(f"  Output: {args.output_image}")
    print()

    # Configure SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.fs * 1.2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = args.buf_size

    # Flush
    for _ in range(3):
        sdr.rx()

    # State
    received_chunks = {}       # seq -> payload bytes
    per_pkt_stats = {}         # seq -> {evm, cfo, stf_peak, time, payload_len}
    total_expected = None
    capture_count = 0
    crc_fail_count = 0
    no_signal_count = 0
    dup_count = 0

    start_time = time.time()

    print(f"Listening for image packets...")
    print("-" * 70)

    try:
        for cap_idx in range(args.max_captures):
            rx_raw = sdr.rx()
            capture_count += 1

            r = demod_one_capture(
                rx_raw, args.fs, stf_ref, ltf_freq_ref,
                args.stf_repeats, args.ltf_symbols,
                args.max_ofdm_syms,
                args.kp, args.ki, args.repeat
            )

            status = r['status']

            if status in ('no_signal', 'ltf_fail', 'no_payload'):
                no_signal_count += 1
                if cap_idx < 5 or cap_idx % 20 == 0:
                    print(f"[{cap_idx + 1:03d}] {status} (SC={r['sc_peak']:.4f})")
                continue

            # Try to parse image packet
            bits_bytes = r['bits_bytes']
            success, seq, total, payload = parse_image_packet(bits_bytes)

            if success:
                if total_expected is None:
                    total_expected = total
                    print(f"  Detected image: {total} packets total")

                if seq in received_chunks:
                    dup_count += 1
                    tag = "DUP"
                else:
                    received_chunks[seq] = payload
                    per_pkt_stats[seq] = {
                        'evm': r.get('mean_evm', 0),
                        'cfo': r['cfo'],
                        'stf_peak': r['stf_peak'],
                        'time': time.time(),
                        'payload_len': len(payload),
                    }
                    tag = "NEW"

                n_recv = len(received_chunks)
                pct = n_recv / total * 100 if total > 0 else 0
                print(f"[{cap_idx + 1:03d}] PKT seq={seq}/{total - 1} "
                      f"({len(payload)}B) CFO={r['cfo']:.0f}Hz "
                      f"EVM={r.get('mean_evm', 0):.4f} [{tag}] "
                      f"Progress: {n_recv}/{total} ({pct:.0f}%)")

                pkt_status = f"PKT OK: seq={seq}/{total - 1} ({len(payload)}B) [{tag}]"
                progress = f"Progress: {n_recv}/{total} ({pct:.0f}%)"

            else:
                crc_fail_count += 1
                if seq >= 0:
                    print(f"[{cap_idx + 1:03d}] CRC_FAIL seq={seq} CFO={r['cfo']:.0f}Hz "
                          f"EVM={r.get('mean_evm', 0):.4f}")
                    pkt_status = f"CRC FAIL (seq={seq})"
                elif bits_bytes[:4] == MAGIC:
                    print(f"[{cap_idx + 1:03d}] CRC_FAIL (header corrupt) "
                          f"CFO={r['cfo']:.0f}Hz")
                    pkt_status = "CRC FAIL (header corrupt)"
                else:
                    print(f"[{cap_idx + 1:03d}] NO_MAGIC CFO={r['cfo']:.0f}Hz "
                          f"EVM={r.get('mean_evm', 0):.4f}")
                    pkt_status = "Magic not found"
                n_recv = len(received_chunks)
                t_exp = total_expected if total_expected else "?"
                progress = f"Progress: {n_recv}/{t_exp}"

            # Plot diagnostics periodically or on new packets
            if (cap_idx + 1) % args.plot_every == 0 or (success and tag == "NEW"):
                plot_capture_diagnostics(
                    r, cap_idx + 1, args.output_dir, args.fs,
                    pkt_status=pkt_status, progress_str=progress
                )

            # Check if complete
            if total_expected is not None and len(received_chunks) >= total_expected:
                print(f"\n  All {total_expected} packets received!")
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    end_time = time.time()

    # ---- Reassemble image ----
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if total_expected is not None and len(received_chunks) > 0:
        n_recv = len(received_chunks)
        missing = [i for i in range(total_expected) if i not in received_chunks]

        print(f"  Packets received: {n_recv}/{total_expected}")
        if missing:
            print(f"  Missing packets: {missing}")

        if n_recv == total_expected:
            # Reassemble
            image_data = b""
            for i in range(total_expected):
                image_data += received_chunks[i]

            with open(args.output_image, "wb") as f:
                f.write(image_data)
            print(f"  Image saved: {args.output_image} ({len(image_data)} bytes)")
        else:
            # Save partial
            partial_data = b""
            for i in range(total_expected):
                if i in received_chunks:
                    partial_data += received_chunks[i]
                else:
                    partial_data += b"\x00" * (args.max_ofdm_syms * BITS_PER_OFDM_SYM // 8)
            partial_path = args.output_image + ".partial"
            with open(partial_path, "wb") as f:
                f.write(partial_data)
            print(f"  Partial image saved: {partial_path}")
    else:
        print("  No image packets received.")

    elapsed = end_time - start_time
    print(f"\n  Statistics:")
    print(f"    Total captures: {capture_count}")
    print(f"    No signal: {no_signal_count}")
    print(f"    CRC failures: {crc_fail_count}")
    print(f"    Duplicates: {dup_count}")
    print(f"    Elapsed: {elapsed:.1f}s")

    if per_pkt_stats:
        evms = [per_pkt_stats[s]['evm'] for s in per_pkt_stats]
        print(f"    Mean EVM: {np.mean(evms):.4f}")

    # Generate summary plot
    if total_expected is not None and total_expected > 0:
        summary_path = plot_reception_summary(
            received_chunks, total_expected, per_pkt_stats,
            args.output_dir, args.output_image, start_time, end_time
        )
        print(f"\n  Summary plot: {summary_path}")
    print(f"  Diagnostic plots: {args.output_dir}/")

    # Cleanup
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


if __name__ == "__main__":
    main()
