#!/usr/bin/env python3
"""
RF Link Test - Step 7: RX (Video Reception)

Receives video frames transmitted as multi-packet JPEG streams:
1. Continuously captures RX buffer
2. Demodulates using Step 5 robust pipeline (Schmidl-Cox, FLL+PLL)
3. Parses video packets with frame_id and sequence numbers
4. Accumulates packets per frame, decodes JPEG when complete
5. Writes decoded frames to output video file
6. Outputs detailed diagnostic figures and evaluation metrics

Packet format (must match TX):
  MAGIC("VID7", 4B) | FRAME_ID(2B) | SEQ(2B) | TOTAL(2B) | LEN(2B) |
  PAYLOAD(var) | CRC32(4B)

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

MAGIC = b"VID7"


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
    n = np.arange(len(samples))
    return samples * np.exp(-1j * 2 * np.pi * cfo * n / fs).astype(np.complex64)


def create_schmidl_cox_stf(N=64, num_repeats=6):
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
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if i % 2 == 0 else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X


def schmidl_cox_metric(rx, half_period=32, window_len=None):
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
    M = np.abs(P_arr)**2 / (R_arr**2 + 1e-10)
    return M, P_arr, R_arr


def detect_stf_autocorr(rx, half_period=32, threshold=0.3):
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
        cfo_norm = -np.angle(p_sum) / (2 * np.pi * half_period)
    else:
        cfo_norm = 0.0
    return peak_val > threshold, int(start_idx), peak_val, cfo_norm, M


def detect_stf_crosscorr(rx, stf_ref, search_range=60000):
    search_len = min(search_range, len(rx) - len(stf_ref))
    if search_len <= 0:
        return -1, 0.0, np.array([])
    corr = np.abs(np.correlate(rx[:search_len + len(stf_ref)], stf_ref, mode='valid'))
    stf_energy = np.sqrt(np.sum(np.abs(stf_ref)**2))
    corr_norm = corr / (stf_energy + 1e-10)
    peak_idx = np.argmax(corr_norm)
    return int(peak_idx), float(corr_norm[peak_idx]), corr_norm


def detect_frame_by_energy(rx, fs, tone_gap=1000, stf_len=400):
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
                    candidates.append(j + win // 2)
                    break
    return candidates


def extract_ofdm_symbol(rx, start_idx):
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    sym = rx[start_idx + N_CP:start_idx + SYMBOL_LEN]
    return np.fft.fftshift(np.fft.fft(sym))


def channel_estimate_from_ltf(rx, ltf_start, ltf_freq_ref, num_symbols=4):
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
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    bits = bits[:L].reshape(-1, repeat)
    return (np.sum(bits, axis=1) >= (repeat / 2)).astype(np.uint8)


# ============================================================================
# Video Packet Parsing
# ============================================================================

def parse_video_packet(data: bytes):
    """
    Parse video packet from demodulated bytes.
    Format: MAGIC(4) | FRAME_ID(2) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    CRC covers MAGIC through PAYLOAD.

    Returns: (success, frame_id, seq, total, payload)
    """
    if len(data) < 16:
        return False, -1, -1, 0, b""
    if data[:4] != MAGIC:
        return False, -1, -1, 0, b""
    try:
        frame_id = int.from_bytes(data[4:6], "little")
        seq = int.from_bytes(data[6:8], "little")
        total = int.from_bytes(data[8:10], "little")
        plen = int.from_bytes(data[10:12], "little")
        expected_len = 12 + plen + 4
        if len(data) < expected_len:
            return False, frame_id, seq, total, b""
        content = data[:12 + plen]
        payload = data[12:12 + plen]
        crc_rx = int.from_bytes(data[12 + plen:12 + plen + 4], "little")
        crc_calc = zlib.crc32(content) & 0xFFFFFFFF
        if crc_rx == crc_calc:
            return True, frame_id, seq, total, payload
        else:
            return False, frame_id, seq, total, b""
    except Exception:
        return False, -1, -1, 0, b""


# ============================================================================
# Frame Accumulator
# ============================================================================

class FrameAccumulator:
    """Tracks multiple in-flight video frames, handles timeout and reassembly."""

    def __init__(self, max_inflight=10, timeout=60.0):
        self.frames = {}       # frame_id -> {pkts: {seq: payload}, total, ts, stats}
        self.completed = set()
        self.expired_ids = set()
        self.max_inflight = max_inflight
        self.timeout = timeout

    def add_packet(self, frame_id, seq, total, payload, evm=0, cfo=0):
        """Add a packet. Returns (frame_complete, frame_id) if complete."""
        if frame_id in self.completed:
            return False, frame_id  # Already completed

        if frame_id not in self.frames:
            self.frames[frame_id] = {
                'pkts': {},
                'total': total,
                'ts_first': time.time(),
                'ts_last': time.time(),
                'evm_sum': 0.0,
                'evm_count': 0,
                'cfo_list': [],
            }

        entry = self.frames[frame_id]
        entry['ts_last'] = time.time()

        if seq not in entry['pkts']:
            entry['pkts'][seq] = payload
            entry['evm_sum'] += evm
            entry['evm_count'] += 1
            entry['cfo_list'].append(cfo)

        # Check completion
        if len(entry['pkts']) >= entry['total']:
            return True, frame_id

        return False, frame_id

    def get_frame_data(self, frame_id):
        """Reassemble JPEG bytes for a completed frame."""
        if frame_id not in self.frames:
            return None
        entry = self.frames[frame_id]
        data = b""
        for i in range(entry['total']):
            if i in entry['pkts']:
                data += entry['pkts'][i]
            else:
                return None  # Missing packet
        return data

    def get_frame_stats(self, frame_id):
        """Get stats for a frame."""
        if frame_id not in self.frames:
            return {}
        entry = self.frames[frame_id]
        n_recv = len(entry['pkts'])
        return {
            'received': n_recv,
            'total': entry['total'],
            'completion': n_recv / entry['total'] if entry['total'] > 0 else 0,
            'mean_evm': entry['evm_sum'] / entry['evm_count'] if entry['evm_count'] > 0 else 0,
            'mean_cfo': np.mean(entry['cfo_list']) if entry['cfo_list'] else 0,
            'latency': entry['ts_last'] - entry['ts_first'],
        }

    def mark_completed(self, frame_id):
        self.completed.add(frame_id)
        if frame_id in self.frames:
            del self.frames[frame_id]

    def expire_old(self):
        """Remove frames older than timeout. Returns list of expired frame_ids with stats."""
        now = time.time()
        expired = []
        for fid in list(self.frames.keys()):
            if now - self.frames[fid]['ts_last'] > self.timeout:
                stats = self.get_frame_stats(fid)
                expired.append((fid, stats))
                self.expired_ids.add(fid)
                del self.frames[fid]

        # Enforce max inflight
        while len(self.frames) > self.max_inflight:
            oldest_fid = min(self.frames.keys(), key=lambda k: self.frames[k]['ts_last'])
            stats = self.get_frame_stats(oldest_fid)
            expired.append((oldest_fid, stats))
            self.expired_ids.add(oldest_fid)
            del self.frames[oldest_fid]

        return expired

    def get_active_frames(self):
        """Return summary of active frames."""
        result = {}
        for fid, entry in self.frames.items():
            result[fid] = f"{len(entry['pkts'])}/{entry['total']}"
        return result


# ============================================================================
# Demodulation Pipeline (from Step 5, adapted for video packets)
# ============================================================================

def demod_one_capture(
    rx_raw, fs, stf_ref, ltf_freq_ref,
    stf_repeats, ltf_symbols, max_ofdm_syms,
    kp, ki, repeat
):
    """Full demodulation pipeline. Returns raw bytes for packet parsing."""
    rx = rx_raw.astype(np.complex64) / (2**14)
    rx = rx - np.mean(rx)

    # Multi-segment CFO estimation
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
    rx_cfo = apply_cfo_correction(rx, cfo, fs)

    # STF detection
    sc_detected, sc_idx, sc_peak, sc_cfo_norm, sc_M = detect_stf_autocorr(
        rx_cfo, half_period=N_FFT // 2, threshold=0.1
    )
    xc_idx, xc_peak, xc_corr = detect_stf_crosscorr(rx_cfo, stf_ref)

    # Combined detection
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
        'cfo': cfo, 'tone_snr': best_snr,
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
        result['evm_per_sym'] = np.array([])
        result['bits_bytes'] = b""
        return result

    ltf_start = stf_idx + len(stf_ref)
    ch_result = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref, ltf_symbols)

    if ch_result[0] is None:
        result['status'] = 'ltf_fail'
        result['H'] = np.ones(N_FFT, dtype=np.complex64)
        result['snr_per_sc'] = np.zeros(52)
        result['all_data_syms'] = np.array([], dtype=np.complex64)
        result['phase_errors'] = np.array([])
        result['freq_log'] = np.array([])
        result['evm_per_sym'] = np.array([])
        result['bits_bytes'] = b""
        return result

    H, snr_per_sc = ch_result
    result['H'] = H
    result['snr_per_sc'] = snr_per_sc

    # Demodulate with FLL+PLL
    payload_start = ltf_start + ltf_symbols * SYMBOL_LEN
    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
    pilot_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS])
    data_idx = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS])

    all_data_syms = []
    phase_errors = []
    freq_log = []
    evm_per_sym = []
    phase_acc = 0.0
    freq_acc = 0.0
    prev_pilot_phase = None
    bad_evm_count = 0

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

        if sym_evm > 0.8:
            bad_evm_count += 1
            if bad_evm_count >= 3:
                break
        else:
            bad_evm_count = 0

        all_data_syms.append(data_syms)
        phase_errors.append(phase_err)
        freq_log.append(freq_acc)
        evm_per_sym.append(sym_evm)

    result['all_data_syms'] = np.concatenate(all_data_syms) if all_data_syms else np.array([], dtype=np.complex64)
    result['phase_errors'] = np.array(phase_errors)
    result['freq_log'] = np.array(freq_log)
    result['evm_per_sym'] = np.array(evm_per_sym)

    if len(all_data_syms) == 0:
        result['status'] = 'no_payload'
        result['bits_bytes'] = b""
        return result

    bits_raw = qpsk_demap(result['all_data_syms'])
    bits = majority_vote(bits_raw, repeat)
    result['bits_bytes'] = np.packbits(bits).tobytes()
    result['status'] = 'demod_ok'
    result['mean_evm'] = float(np.mean(evm_per_sym)) if evm_per_sym else 0.0
    return result


# ============================================================================
# Diagnostic Plots
# ============================================================================

def plot_frame_diagnostics(frame_id, frame_img, pkt_stats, total_pkts, output_dir):
    """Generate 2x2 diagnostic figure for a completed frame."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Decoded frame thumbnail
    ax = axes[0, 0]
    if frame_img is not None:
        # Convert BGR to RGB for matplotlib
        ax.imshow(frame_img[:, :, ::-1])
        ax.set_title(f'Frame {frame_id} (decoded)')
    else:
        ax.text(0.5, 0.5, 'Decode Failed', ha='center', va='center',
                transform=ax.transAxes, fontsize=16, color='red')
        ax.set_title(f'Frame {frame_id} (FAILED)')
    ax.axis('off')

    # (0,1) Packet reception map
    ax = axes[0, 1]
    colors = []
    for i in range(total_pkts):
        if i in pkt_stats:
            colors.append('green')
        else:
            colors.append('red')
    ax.barh(range(total_pkts), [1] * total_pkts, color=colors,
            edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Received')
    ax.set_ylabel('Packet Seq')
    ax.set_title(f'Packets: {len(pkt_stats)}/{total_pkts}')
    ax.set_xlim([0, 1.2])

    # (1,0) EVM per packet
    ax = axes[1, 0]
    if pkt_stats:
        seqs = sorted(pkt_stats.keys())
        evms = [pkt_stats[s].get('evm', 0) for s in seqs]
        ax.bar(seqs, evms, color='steelblue', alpha=0.8)
        if evms:
            ax.axhline(np.mean(evms), color='r', linestyle='--',
                        label=f'Mean={np.mean(evms):.4f}')
            ax.legend(fontsize=8)
    ax.set_xlabel('Packet Seq')
    ax.set_ylabel('EVM')
    ax.set_title('EVM per Packet')
    ax.grid(True)

    # (1,1) Constellation from all packets
    ax = axes[1, 1]
    all_syms = []
    for s in sorted(pkt_stats.keys()):
        syms = pkt_stats[s].get('data_syms', np.array([]))
        if len(syms) > 0:
            all_syms.append(syms)
    if all_syms:
        all_syms = np.concatenate(all_syms)
        n = len(all_syms)
        ax.scatter(np.real(all_syms), np.imag(all_syms),
                   c=np.arange(n), cmap='viridis', s=2, alpha=0.5)
        ideal = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        ax.scatter(np.real(ideal), np.imag(ideal), c='red', s=80, marker='x', linewidths=2)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.axvline(0, color='k', linewidth=0.3)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title('QPSK Constellation (all packets)')
    ax.axis('equal')
    ax.grid(True)

    fig.suptitle(f'RF Link Step 7 - Frame {frame_id}', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, f'frame_{frame_id:04d}.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_video_summary(all_frame_stats, capture_stats, output_dir, start_time, end_time):
    """Generate video session summary figure (3x3)."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))

    elapsed = end_time - start_time
    completed_fids = sorted(all_frame_stats.keys())
    n_completed = len(completed_fids)

    # (0,0) Frame completion over time
    ax = axes[0, 0]
    if completed_fids:
        times = [all_frame_stats[f]['complete_time'] - start_time for f in completed_fids]
        ax.step(times, range(1, n_completed + 1), 'b-', where='post')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frames Completed')
        effective_fps = n_completed / elapsed if elapsed > 0 else 0
        ax.set_title(f'Frame Completion ({effective_fps:.3f} fps)')
    ax.grid(True)

    # (0,1) PER over time
    ax = axes[0, 1]
    if capture_stats['per_capture_ok']:
        window = 20
        ok = np.array(capture_stats['per_capture_ok'])
        if len(ok) >= window:
            per = 1.0 - np.convolve(ok, np.ones(window) / window, mode='valid')
            ax.plot(per, 'r-')
            ax.set_title(f'PER (window={window})')
        else:
            total_ok = np.sum(ok)
            per = 1.0 - total_ok / len(ok) if len(ok) > 0 else 1.0
            ax.axhline(per, color='r')
            ax.set_title(f'PER = {per:.3f}')
    ax.set_xlabel('Capture')
    ax.set_ylabel('PER')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True)

    # (0,2) Throughput over time
    ax = axes[0, 2]
    if completed_fids:
        times = [all_frame_stats[f]['complete_time'] - start_time for f in completed_fids]
        sizes = [all_frame_stats[f]['jpeg_size'] for f in completed_fids]
        cum_bytes = np.cumsum(sizes)
        ax.plot(times, np.array(cum_bytes) / 1024, 'b-o', markersize=3)
        total_bytes = cum_bytes[-1] if len(cum_bytes) > 0 else 0
        throughput = total_bytes / elapsed if elapsed > 0 else 0
        ax.set_title(f'Throughput ({throughput:.0f} B/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative (KB)')
    ax.grid(True)

    # (1,0) Latency per frame
    ax = axes[1, 0]
    if completed_fids:
        latencies = [all_frame_stats[f].get('latency', 0) for f in completed_fids]
        ax.bar(range(len(latencies)), latencies, color='steelblue', alpha=0.8)
        ax.axhline(np.mean(latencies), color='r', linestyle='--',
                    label=f'Mean={np.mean(latencies):.1f}s')
        ax.legend(fontsize=8)
    ax.set_xlabel('Frame (order)')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Per-Frame Latency')
    ax.grid(True)

    # (1,1) Packet reception heatmap
    ax = axes[1, 1]
    if completed_fids:
        max_pkts = max(all_frame_stats[f].get('total_pkts', 1) for f in completed_fids)
        n_frames = len(completed_fids)
        heatmap = np.zeros((n_frames, max_pkts))
        for i, fid in enumerate(completed_fids):
            total = all_frame_stats[fid].get('total_pkts', 0)
            recv = all_frame_stats[fid].get('received_pkts', 0)
            heatmap[i, :recv] = 1.0
            heatmap[i, recv:total] = 0.5  # partial
        ax.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_xlabel('Packet Seq')
        ax.set_ylabel('Frame (order)')
        ax.set_title('Packet Heatmap (green=ok)')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Packet Heatmap')

    # (1,2) EVM distribution
    ax = axes[1, 2]
    all_evms = []
    for fid in completed_fids:
        evm = all_frame_stats[fid].get('mean_evm', 0)
        if evm > 0:
            all_evms.append(evm)
    if all_evms:
        ax.hist(all_evms, bins=20, color='steelblue', alpha=0.8, edgecolor='black')
        ax.axvline(np.mean(all_evms), color='r', linestyle='--',
                    label=f'Mean={np.mean(all_evms):.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Mean EVM')
    ax.set_ylabel('Count')
    ax.set_title('EVM Distribution (per frame)')
    ax.grid(True)

    # (2,0) CFO stability
    ax = axes[2, 0]
    if capture_stats['cfos']:
        ax.plot(capture_stats['cfos'], 'r.', markersize=2, alpha=0.5)
        ax.axhline(np.mean(capture_stats['cfos']), color='b', linestyle='--',
                    label=f'Mean={np.mean(capture_stats["cfos"]):.1f} Hz')
        ax.legend(fontsize=8)
    ax.set_xlabel('Capture')
    ax.set_ylabel('CFO (Hz)')
    ax.set_title(f'CFO Stability (std={np.std(capture_stats["cfos"]):.1f} Hz)' if capture_stats['cfos'] else 'CFO')
    ax.grid(True)

    # (2,1) Per-frame completion percentage
    ax = axes[2, 1]
    if completed_fids:
        completions = [all_frame_stats[f].get('completion', 0) * 100 for f in completed_fids]
        ax.bar(range(len(completions)), completions, color='green', alpha=0.8)
        ax.axhline(100, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frame (order)')
    ax.set_ylabel('Completion (%)')
    ax.set_title('Frame Completion %')
    ax.set_ylim([0, 110])
    ax.grid(True)

    # (2,2) Summary text
    ax = axes[2, 2]
    ax.axis('off')

    n_captures = capture_stats['total_captures']
    n_no_signal = capture_stats['no_signal']
    n_crc_fail = capture_stats['crc_fail']
    n_pkt_ok = capture_stats['pkt_ok']
    n_dup = capture_stats['dup']
    n_expired = capture_stats.get('expired', 0)

    per = 1.0 - n_pkt_ok / (n_pkt_ok + n_crc_fail) if (n_pkt_ok + n_crc_fail) > 0 else 1.0
    fer = 1.0 - n_completed / (n_completed + n_expired) if (n_completed + n_expired) > 0 else 1.0

    mean_evm_str = f"{np.mean(all_evms):.4f}" if all_evms else "N/A"
    cfo_std_str = f"{np.std(capture_stats['cfos']):.1f}" if capture_stats['cfos'] else "N/A"
    fps_str = f"{n_completed / elapsed:.4f}" if elapsed > 0 else "N/A"

    summary = f"""
VIDEO RECEPTION SUMMARY

Frames completed: {n_completed}
Frames expired: {n_expired}
FER: {fer:.3f}

Captures: {n_captures}
  Packets OK: {n_pkt_ok}
  CRC fails: {n_crc_fail}
  No signal: {n_no_signal}
  Duplicates: {n_dup}
PER: {per:.3f}

Elapsed: {elapsed:.1f}s
Effective FPS: {fps_str}

Mean EVM: {mean_evm_str}
CFO std: {cfo_std_str} Hz
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.suptitle('RF Link Step 7 - Video Session Summary', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'video_summary.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="RF Link Test - Step 7: Video RX")
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
    ap.add_argument("--max_captures", type=int, default=5000, help="Max capture attempts")
    ap.add_argument("--output_dir", default="rf_link_step7_results", help="Output directory")
    ap.add_argument("--output_video", default="received_video.avi", help="Output video")
    ap.add_argument("--output_fps", type=float, default=2.0, help="Output video FPS")
    ap.add_argument("--width", type=int, default=320, help="Frame width (must match TX)")
    ap.add_argument("--height", type=int, default=240, help="Frame height (must match TX)")
    ap.add_argument("--frame_timeout", type=float, default=60.0,
                    help="Seconds before expiring incomplete frame")
    ap.add_argument("--max_inflight", type=int, default=10, help="Max concurrent partial frames")
    ap.add_argument("--save_frames_dir", default="", help="Dir to save decoded frames as JPEG")
    ap.add_argument("--plot_every_frame", type=int, default=1,
                    help="Plot diagnostics every Nth completed frame")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Stop after N completed frames (0=unlimited)")
    args = ap.parse_args()

    import cv2
    import adi

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    # Create reference signals
    stf_ref, _ = create_schmidl_cox_stf(N_FFT, num_repeats=args.stf_repeats)
    _, ltf_freq_ref = create_ltf_ref(N_FFT, num_symbols=args.ltf_symbols)

    print(f"RF Link Test - Step 7: Video RX")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc / 1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs / 1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
    print(f"  Frame size: {args.width}x{args.height}")
    print(f"  Frame timeout: {args.frame_timeout}s")
    print(f"  Max captures: {args.max_captures}")
    print(f"  Output: {args.output_video}")
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

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(
        args.output_video, fourcc, args.output_fps,
        (args.width, args.height)
    )

    # State
    accumulator = FrameAccumulator(
        max_inflight=args.max_inflight,
        timeout=args.frame_timeout
    )
    all_frame_stats = {}     # frame_id -> {complete_time, jpeg_size, latency, ...}
    per_frame_pkt_stats = {} # frame_id -> {seq -> {evm, data_syms}}
    capture_stats = {
        'total_captures': 0,
        'no_signal': 0,
        'crc_fail': 0,
        'pkt_ok': 0,
        'dup': 0,
        'expired': 0,
        'per_capture_ok': [],
        'cfos': [],
    }

    start_time = time.time()
    frames_written = 0

    print(f"Listening for video packets...")
    print("-" * 70)

    try:
        for cap_idx in range(args.max_captures):
            rx_raw = sdr.rx()
            capture_stats['total_captures'] += 1

            r = demod_one_capture(
                rx_raw, args.fs, stf_ref, ltf_freq_ref,
                args.stf_repeats, args.ltf_symbols,
                args.max_ofdm_syms, args.kp, args.ki, args.repeat
            )

            status = r['status']

            if status in ('no_signal', 'ltf_fail', 'no_payload'):
                capture_stats['no_signal'] += 1
                capture_stats['per_capture_ok'].append(0)
                if cap_idx < 3 or cap_idx % 50 == 0:
                    print(f"[{cap_idx + 1:04d}] {status}")
                continue

            # Parse video packet
            bits_bytes = r['bits_bytes']
            success, frame_id, seq, total, payload = parse_video_packet(bits_bytes)

            if success:
                capture_stats['pkt_ok'] += 1
                capture_stats['per_capture_ok'].append(1)
                capture_stats['cfos'].append(r['cfo'])

                evm = r.get('mean_evm', 0)
                complete, fid = accumulator.add_packet(
                    frame_id, seq, total, payload, evm=evm, cfo=r['cfo']
                )

                # Track per-packet data for diagnostics
                if frame_id not in per_frame_pkt_stats:
                    per_frame_pkt_stats[frame_id] = {}
                is_new = seq not in per_frame_pkt_stats[frame_id]

                if is_new:
                    per_frame_pkt_stats[frame_id][seq] = {
                        'evm': evm,
                        'data_syms': r['all_data_syms'].copy() if len(r['all_data_syms']) > 0 else np.array([]),
                    }
                    tag = "NEW"
                else:
                    capture_stats['dup'] += 1
                    tag = "DUP"

                # Progress
                fstats = accumulator.get_frame_stats(frame_id)
                n_recv = fstats.get('received', 0) if not complete else total
                pct = n_recv / total * 100 if total > 0 else 0

                # Print status
                if is_new or cap_idx < 10:
                    print(f"[{cap_idx + 1:04d}] F{frame_id}:pkt{seq}/{total - 1} "
                          f"({len(payload)}B) EVM={evm:.4f} [{tag}] "
                          f"({n_recv}/{total} {pct:.0f}%)")

                # Handle completed frame
                if complete:
                    jpeg_data = accumulator.get_frame_data(frame_id)
                    fstats_final = accumulator.get_frame_stats(frame_id)

                    frame_img = None
                    if jpeg_data is not None:
                        nparr = np.frombuffer(jpeg_data, dtype=np.uint8)
                        frame_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame_img is not None:
                        # Resize if needed
                        if frame_img.shape[1] != args.width or frame_img.shape[0] != args.height:
                            frame_img = cv2.resize(frame_img, (args.width, args.height))
                        video_writer.write(frame_img)
                        frames_written += 1

                        # Save individual frame
                        if args.save_frames_dir:
                            frame_path = os.path.join(
                                args.save_frames_dir, f"frame_{frame_id:04d}.jpg"
                            )
                            cv2.imwrite(frame_path, frame_img)

                        print(f"  >>> Frame {frame_id} COMPLETE: "
                              f"{len(jpeg_data)}B JPEG, "
                              f"latency={fstats_final.get('latency', 0):.1f}s, "
                              f"EVM={fstats_final.get('mean_evm', 0):.4f} "
                              f"[{frames_written} frames written]")
                    else:
                        print(f"  >>> Frame {frame_id}: JPEG decode FAILED "
                              f"({len(jpeg_data) if jpeg_data else 0}B)")

                    # Record stats
                    all_frame_stats[frame_id] = {
                        'complete_time': time.time(),
                        'jpeg_size': len(jpeg_data) if jpeg_data else 0,
                        'latency': fstats_final.get('latency', 0),
                        'mean_evm': fstats_final.get('mean_evm', 0),
                        'total_pkts': total,
                        'received_pkts': total,
                        'completion': 1.0,
                        'decode_ok': frame_img is not None,
                    }

                    # Plot diagnostics
                    if frames_written % args.plot_every_frame == 0:
                        pkt_st = per_frame_pkt_stats.get(frame_id, {})
                        plot_frame_diagnostics(
                            frame_id, frame_img, pkt_st, total, args.output_dir
                        )

                    accumulator.mark_completed(frame_id)

            else:
                capture_stats['crc_fail'] += 1
                capture_stats['per_capture_ok'].append(0)
                if cap_idx < 10 or cap_idx % 50 == 0:
                    if frame_id >= 0:
                        print(f"[{cap_idx + 1:04d}] CRC_FAIL F{frame_id}:pkt{seq} "
                              f"CFO={r['cfo']:.0f}Hz")
                    else:
                        print(f"[{cap_idx + 1:04d}] NO_MAGIC CFO={r['cfo']:.0f}Hz")

            # Expire old frames
            expired = accumulator.expire_old()
            for exp_fid, exp_stats in expired:
                capture_stats['expired'] += 1
                print(f"  [EXPIRED] Frame {exp_fid}: "
                      f"{exp_stats.get('received', 0)}/{exp_stats.get('total', '?')} pkts")
                all_frame_stats[exp_fid] = {
                    'complete_time': time.time(),
                    'jpeg_size': 0,
                    'latency': exp_stats.get('latency', 0),
                    'mean_evm': exp_stats.get('mean_evm', 0),
                    'total_pkts': exp_stats.get('total', 0),
                    'received_pkts': exp_stats.get('received', 0),
                    'completion': exp_stats.get('completion', 0),
                    'decode_ok': False,
                }

            # Check if we've received enough frames
            if args.max_frames > 0 and frames_written >= args.max_frames:
                print(f"\n  Reached max_frames ({args.max_frames}).")
                break

            # Periodic status
            if (cap_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                active = accumulator.get_active_frames()
                print(f"\n  --- Status at capture {cap_idx + 1} ---")
                print(f"  Frames written: {frames_written}")
                print(f"  Active frames: {active}")
                print(f"  Elapsed: {elapsed:.1f}s\n")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    end_time = time.time()
    video_writer.release()

    # ---- Final summary ----
    elapsed = end_time - start_time
    print()
    print("=" * 70)
    print("VIDEO RECEPTION SUMMARY")
    print("=" * 70)
    print(f"  Frames written: {frames_written}")
    print(f"  Frames expired: {capture_stats['expired']}")
    print(f"  Total captures: {capture_stats['total_captures']}")
    print(f"  Packets OK: {capture_stats['pkt_ok']}")
    print(f"  CRC failures: {capture_stats['crc_fail']}")
    print(f"  No signal: {capture_stats['no_signal']}")
    print(f"  Duplicates: {capture_stats['dup']}")
    print(f"  Elapsed: {elapsed:.1f}s")
    if frames_written > 0:
        print(f"  Effective FPS: {frames_written / elapsed:.4f}")

    per = (1.0 - capture_stats['pkt_ok'] / (capture_stats['pkt_ok'] + capture_stats['crc_fail'])
           if (capture_stats['pkt_ok'] + capture_stats['crc_fail']) > 0 else 1.0)
    fer = (1.0 - frames_written / (frames_written + capture_stats['expired'])
           if (frames_written + capture_stats['expired']) > 0 else 1.0)
    print(f"  PER: {per:.4f}")
    print(f"  FER: {fer:.4f}")
    print(f"  Output video: {args.output_video}")

    # Generate summary plot
    if all_frame_stats:
        summary_path = plot_video_summary(
            all_frame_stats, capture_stats,
            args.output_dir, start_time, end_time
        )
        print(f"  Summary plot: {summary_path}")

    print(f"  Diagnostic plots: {args.output_dir}/")

    # Cleanup
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


if __name__ == "__main__":
    main()
