#!/usr/bin/env python3
"""
RF Link Test - Step 5: RX with Robust Sync and Diagnostics

Improvements over Step 4:
1. Schmidl-Cox autocorrelation for timing sync (CFO-tolerant)
2. Multi-symbol LTF channel estimation averaging
3. Comprehensive diagnostic plots saved as images
4. BER stress test mode
5. Per-symbol EVM, phase error, and SNR tracking

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

MAGIC = b"AIS1"


# ============================================================================
# Signal Processing Functions
# ============================================================================

def estimate_cfo_from_tone(rx_samples, fs, expected_freq=100e3):
    """Estimate CFO from pilot tone using FFT peak with parabolic interpolation."""
    N = len(rx_samples)
    # Apply Hanning window for better spectral leakage
    window = np.hanning(N).astype(np.float32)
    fft = np.fft.fft(rx_samples * window)
    fft_mag = np.abs(fft)

    # Remove DC and near-DC
    fft_mag[0] = 0
    fft_mag[1:30] = 0
    fft_mag[-30:] = 0

    peak_idx = np.argmax(fft_mag)
    freq_bins = np.fft.fftfreq(N, 1/fs)

    # Parabolic interpolation for sub-bin accuracy
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
    """
    Compute Schmidl-Cox timing metric.

    The STF has period N/2 = 32, so correlating rx[n:n+L] with rx[n+L:n+2L]
    gives a high peak at the start of the STF, even in the presence of CFO.

    Returns:
        M: timing metric array |P|^2 / R^2
        P: complex autocorrelation array
    """
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
    """
    Detect STF using Schmidl-Cox autocorrelation.

    More robust than cross-correlation because it does NOT require
    CFO correction before detection. CFO only causes a phase rotation
    of P, not a reduction of |P|.

    Returns: (detected, start_idx, metric_peak, cfo_coarse, M)
    """
    search_len = min(60000, len(rx) - 2 * half_period)
    M, P, R = schmidl_cox_metric(rx, half_period, search_len)

    # Find plateau region (STF gives sustained high metric)
    # Use sliding window to find the sustained high region
    plateau_len = 2 * half_period  # Expect at least 2 half-periods of plateau
    if len(M) < plateau_len:
        return False, 0, 0.0, 0.0, M

    # Smooth the metric
    kernel = np.ones(plateau_len) / plateau_len
    M_smooth = np.convolve(M, kernel, mode='valid')

    peak_smooth_idx = np.argmax(M_smooth)
    peak_val = float(M_smooth[peak_smooth_idx])

    # The actual STF start is near the beginning of the plateau
    # Search for the rising edge before the peak
    start_idx = peak_smooth_idx
    for i in range(peak_smooth_idx, max(0, peak_smooth_idx - 5 * half_period), -1):
        if M[i] < 0.1 * peak_val:
            start_idx = i + 1
            break

    # CFO estimate from the phase of P at the peak
    if start_idx + half_period < len(P):
        # Average P over several positions for better estimate
        p_sum = np.sum(P[start_idx:start_idx + 2 * half_period])
        cfo_phase = np.angle(p_sum)
        # CFO in normalized frequency: f_cfo = -phase / (2*pi*L)
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
    """
    Detect frame structure using energy envelope.

    Frame: ... gap(3000) | TONE(20000) | gap(1000) | STF | LTF | payload | gap(3000) ...

    Strategy:
    1. Compute smoothed energy envelope
    2. Find tone regions (high energy, narrowband)
    3. Find the gap after tone (low energy)
    4. STF starts ~tone_gap samples after tone ends

    Returns: list of candidate STF start positions
    """
    # Smoothed energy envelope (window ~500 samples)
    win = 500
    energy = np.abs(rx)**2
    if len(energy) < win:
        return []

    # Cumulative sum for fast moving average
    cumsum = np.cumsum(energy)
    cumsum = np.insert(cumsum, 0, 0)
    avg_energy = (cumsum[win:] - cumsum[:-win]) / win

    # Threshold: distinguish signal from noise
    noise_floor = np.percentile(avg_energy, 10)
    signal_thresh = noise_floor * 10  # 10 dB above noise floor

    # Find transitions from high energy to low energy (end of tone/payload → gap)
    is_signal = avg_energy > signal_thresh

    candidates = []

    # Find falling edges (signal → gap) then rising edges (gap → signal)
    for i in range(1, len(is_signal) - tone_gap - stf_len):
        # Look for: signal region → gap → signal region
        # The gap between tone and STF is ~1000 samples
        if is_signal[i] and not is_signal[i + 1]:
            # Found a falling edge at position i
            # Look for the next rising edge (start of STF)
            for j in range(i + 1, min(i + 3000, len(is_signal))):
                if is_signal[j] and not is_signal[j - 1]:
                    # Rising edge at j - this could be STF start
                    # Adjust for the window averaging offset
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

    # Average across LTF symbols
    Y_avg = np.mean(Ys, axis=0)

    # Channel estimate
    H = np.ones(N_FFT, dtype=np.complex64)
    for k in used:
        idx = (k + N_FFT) % N_FFT
        if np.abs(ltf_freq_ref[idx]) > 0.1:
            H[idx] = Y_avg[idx] / ltf_freq_ref[idx]

    # Channel SNR estimate per subcarrier
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
            bits[2*i], bits[2*i+1] = 0, 0
        elif not re and im:
            bits[2*i], bits[2*i+1] = 0, 1
        elif not re and not im:
            bits[2*i], bits[2*i+1] = 1, 1
        else:
            bits[2*i], bits[2*i+1] = 1, 0
    return bits


def majority_vote(bits, repeat):
    """Apply majority voting for repeated bits."""
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    bits = bits[:L].reshape(-1, repeat)
    return (np.sum(bits, axis=1) >= (repeat / 2)).astype(np.uint8)


def parse_packet(data: bytes):
    """Parse received packet, returns (success, payload, crc_rx, crc_calc)."""
    if len(data) < 10:
        return False, b"", 0, 0
    if data[:4] != MAGIC:
        return False, b"", 0, 0
    plen = int.from_bytes(data[4:6], "little")
    need = 6 + plen + 4
    if len(data) < need:
        return False, b"", 0, 0
    payload = data[6:6 + plen]
    crc_rx = int.from_bytes(data[6 + plen:6 + plen + 4], "little")
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
    return (crc_rx == crc_calc), payload, crc_rx, crc_calc


# ============================================================================
# Diagnostic Plot Generation
# ============================================================================

def plot_full_diagnostics(
    rx, rx_cfo, cfo, stf_idx, stf_peak,
    sc_metric, xcorr_metric, xcorr_idx,
    H, snr_per_sc,
    all_data_syms, phase_errors, freq_log, pilot_powers, evm_per_sym,
    result_str, capture_id, output_dir, fs,
    stf_cfo_norm=0.0
):
    """Generate comprehensive 4x3 diagnostic plot."""

    fig, axes = plt.subplots(4, 3, figsize=(20, 22))

    # Row 0: Signal Overview
    # (0,0) Time domain power envelope
    ax = axes[0, 0]
    power_env = np.abs(rx[:min(60000, len(rx))])**2
    # Downsample for plotting
    ds = max(1, len(power_env) // 5000)
    ax.plot(np.arange(0, len(power_env), ds) / fs * 1000, power_env[::ds])
    if stf_idx > 0:
        ax.axvline(stf_idx / fs * 1000, color='r', linestyle='--', label=f'STF @{stf_idx}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power')
    ax.set_title('Signal Power Envelope')
    ax.legend(fontsize=8)
    ax.grid(True)

    # (0,1) Spectrum before/after CFO correction
    ax = axes[0, 1]
    N_fft = min(16384, len(rx))
    win = np.hanning(N_fft)
    fft_before = np.fft.fftshift(np.fft.fft(rx[:N_fft] * win))
    fft_after = np.fft.fftshift(np.fft.fft(rx_cfo[:N_fft] * win))
    freq_kHz = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/fs)) / 1e3
    ax.plot(freq_kHz, 20 * np.log10(np.abs(fft_before) + 1e-10), alpha=0.6, label='Before CFO')
    ax.plot(freq_kHz, 20 * np.log10(np.abs(fft_after) + 1e-10), alpha=0.6, label='After CFO')
    ax.axvline(100, color='r', linestyle=':', alpha=0.5, label='100 kHz')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Spectrum (CFO = {cfo:.1f} Hz)')
    ax.legend(fontsize=8)
    ax.set_xlim([-300, 300])
    ax.grid(True)

    # (0,2) Schmidl-Cox timing metric
    ax = axes[0, 2]
    disp_len = min(50000, len(sc_metric))
    ax.plot(sc_metric[:disp_len], label='SC Metric')
    if stf_idx > 0 and stf_idx < disp_len:
        ax.axvline(stf_idx, color='r', linestyle='--', label=f'STF detected')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Metric')
    ax.set_title(f'Schmidl-Cox Autocorrelation (peak={stf_peak:.4f})')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Row 1: Sync Details
    # (1,0) Cross-correlation (backup method)
    ax = axes[1, 0]
    if xcorr_metric is not None and len(xcorr_metric) > 0:
        disp_len2 = min(50000, len(xcorr_metric))
        ax.plot(xcorr_metric[:disp_len2])
        if xcorr_idx >= 0 and xcorr_idx < disp_len2:
            ax.axvline(xcorr_idx, color='r', linestyle='--',
                       label=f'XCorr peak={xcorr_metric[xcorr_idx]:.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Correlation')
    ax.set_title('Cross-Correlation STF Detection')
    ax.grid(True)

    # (1,1) Channel Estimate |H|
    ax = axes[1, 1]
    H_shift = np.fft.fftshift(H)
    H_mag = 20 * np.log10(np.abs(H_shift) + 1e-10)
    H_phase = np.angle(H_shift)
    sc_axis = np.arange(-N_FFT // 2, N_FFT // 2)
    ax.plot(sc_axis, H_mag, 'b-', label='|H| (dB)')
    ax2 = ax.twinx()
    ax2.plot(sc_axis, np.degrees(H_phase), 'r-', alpha=0.5, label='Phase (deg)')
    ax2.set_ylabel('Phase (deg)', color='r')
    ax.set_xlabel('Subcarrier Index')
    ax.set_ylabel('|H| (dB)', color='b')
    ax.set_title('Channel Estimate')
    ax.grid(True)

    # (1,2) Channel SNR per subcarrier
    ax = axes[1, 2]
    if snr_per_sc is not None and len(snr_per_sc) > 0:
        used = np.array([k for k in range(-26, 27) if k != 0])
        ax.bar(used, snr_per_sc, width=0.8, alpha=0.7)
        ax.axhline(np.mean(snr_per_sc), color='r', linestyle='--',
                    label=f'Mean SNR = {np.mean(snr_per_sc):.1f} dB')
        ax.legend(fontsize=8)
    ax.set_xlabel('Subcarrier')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Per-Subcarrier SNR (from LTF)')
    ax.grid(True)

    # Row 2: Demodulation
    # (2,0) QPSK Constellation
    ax = axes[2, 0]
    if all_data_syms is not None and len(all_data_syms) > 0:
        n_syms = len(all_data_syms)
        colors = np.arange(n_syms)
        sc = ax.scatter(np.real(all_data_syms), np.imag(all_data_syms),
                        c=colors, cmap='viridis', s=3, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='Symbol Index')
        # Mark ideal QPSK points
        ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        ax.scatter(np.real(ideal), np.imag(ideal), c='red', s=100, marker='x', linewidths=2)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.axvline(0, color='k', linewidth=0.3)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title('QPSK Constellation (colored by time)')
    ax.axis('equal')
    ax.grid(True)

    # (2,1) EVM per symbol
    ax = axes[2, 1]
    if evm_per_sym is not None and len(evm_per_sym) > 0:
        ax.plot(evm_per_sym, 'b-', alpha=0.7)
        ax.axhline(np.mean(evm_per_sym), color='r', linestyle='--',
                    label=f'Mean EVM = {np.mean(evm_per_sym):.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('OFDM Symbol Index')
    ax.set_ylabel('EVM (RMS)')
    ax.set_title('EVM per OFDM Symbol')
    ax.grid(True)

    # (2,2) Phase angle histogram
    ax = axes[2, 2]
    if all_data_syms is not None and len(all_data_syms) > 0:
        angles = np.degrees(np.angle(all_data_syms))
        ax.hist(angles, bins=72, range=(-180, 180), alpha=0.7, edgecolor='black', linewidth=0.3)
        for ideal_angle in [45, 135, -135, -45]:
            ax.axvline(ideal_angle, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Phase (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Constellation Angle Histogram')
    ax.grid(True)

    # Row 3: Tracking
    # (3,0) Pilot phase error
    ax = axes[3, 0]
    if phase_errors is not None and len(phase_errors) > 0:
        ax.plot(np.degrees(phase_errors), 'b-', alpha=0.7, label='Raw')
        ax.plot(np.degrees(np.unwrap(phase_errors)), 'r-', alpha=0.7, label='Unwrapped')
        ax.legend(fontsize=8)
    ax.set_xlabel('OFDM Symbol Index')
    ax.set_ylabel('Phase Error (degrees)')
    ax.set_title('Pilot Phase Error')
    ax.grid(True)

    # (3,1) Frequency tracking
    ax = axes[3, 1]
    if freq_log is not None and len(freq_log) > 0:
        ax.plot(freq_log, 'b-')
        ax.axhline(0, color='k', linewidth=0.3)
    ax.set_xlabel('OFDM Symbol Index')
    ax.set_ylabel('Freq Acc (rad/sym)')
    ax.set_title('Residual CFO Tracking (PI Integrator)')
    ax.grid(True)

    # (3,2) Summary text
    ax = axes[3, 2]
    ax.axis('off')
    if all_data_syms is not None and len(all_data_syms) > 0:
        mean_evm = np.mean(evm_per_sym) if evm_per_sym is not None and len(evm_per_sym) > 0 else 0
        mean_snr = np.mean(snr_per_sc) if snr_per_sc is not None and len(snr_per_sc) > 0 else 0
    else:
        mean_evm = 0
        mean_snr = 0

    summary = f"""
{result_str}

Signal:
  RX Power: {10*np.log10(np.mean(np.abs(rx)**2)+1e-10):.1f} dB

CFO:
  Tone-based: {cfo:.1f} Hz
  SC autocorr: {stf_cfo_norm * fs:.1f} Hz

Sync:
  SC metric peak: {stf_peak:.4f}
  STF index: {stf_idx}

Channel:
  Mean SNR: {mean_snr:.1f} dB
  Mean EVM: {mean_evm:.4f}
  OFDM syms: {len(all_data_syms)//N_DATA if all_data_syms is not None and len(all_data_syms)>0 else 0}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'RF Link Step 5 - Capture {capture_id}', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = os.path.join(output_dir, f'capture_{capture_id:02d}.png')
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    return plot_path


def plot_ber_summary(ber_results, output_dir):
    """Plot BER stress test summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    captures = [r['capture'] for r in ber_results]
    bers = [r['ber'] for r in ber_results]
    evms = [r['mean_evm'] for r in ber_results]
    cfos = [r['cfo'] for r in ber_results]
    stf_peaks = [r['stf_peak'] for r in ber_results]

    # BER over captures
    ax = axes[0, 0]
    ax.semilogy(captures, [max(b, 1e-6) for b in bers], 'bo-', markersize=6)
    ax.axhline(0.01, color='r', linestyle='--', alpha=0.5, label='1% BER')
    ax.axhline(0.001, color='g', linestyle='--', alpha=0.5, label='0.1% BER')
    ax.set_xlabel('Capture')
    ax.set_ylabel('BER')
    ax.set_title(f'BER per Capture (mean={np.mean(bers):.2e})')
    ax.legend(fontsize=8)
    ax.grid(True)

    # EVM over captures
    ax = axes[0, 1]
    ax.plot(captures, evms, 'go-', markersize=6)
    ax.set_xlabel('Capture')
    ax.set_ylabel('Mean EVM')
    ax.set_title(f'EVM per Capture (mean={np.mean(evms):.4f})')
    ax.grid(True)

    # CFO stability
    ax = axes[1, 0]
    ax.plot(captures, cfos, 'ro-', markersize=6)
    ax.set_xlabel('Capture')
    ax.set_ylabel('CFO (Hz)')
    ax.set_title(f'CFO Stability (std={np.std(cfos):.1f} Hz)')
    ax.grid(True)

    # STF peak stability
    ax = axes[1, 1]
    ax.plot(captures, stf_peaks, 'mo-', markersize=6)
    ax.set_xlabel('Capture')
    ax.set_ylabel('STF Peak')
    ax.set_title(f'STF Detection Peak (mean={np.mean(stf_peaks):.4f})')
    ax.grid(True)

    n_ok = sum(1 for b in bers if b < 0.01)
    fig.suptitle(
        f'BER Stress Test Summary: {n_ok}/{len(bers)} captures below 1% BER',
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(output_dir, 'ber_stress_summary.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ============================================================================
# Main RX Logic
# ============================================================================

def demod_one_capture(
    rx_raw, fs, stf_ref, ltf_freq_ref,
    stf_repeats, ltf_symbols, max_ofdm_syms,
    kp, ki, repeat,
    ber_test=False, ber_ref_bits=None
):
    """
    Full demodulation pipeline for one RX capture.

    Returns a dict with all results and diagnostics.
    """
    rx = rx_raw.astype(np.complex64) / (2**14)
    rx = rx - np.mean(rx)  # DC removal

    # ---- CFO from tone (robust: scan multiple segments, pick highest SNR) ----
    # The tone may not be at the start of the buffer due to cyclic frame timing.
    # Scan several overlapping segments, estimate CFO from each, pick the one
    # with the best tone SNR (strongest narrowband peak).
    seg_len = 20000  # Long enough for good freq resolution
    best_cfo = 0.0
    best_snr = 0.0
    cfo_estimates = []
    for seg_start in range(0, min(len(rx) - seg_len, 80000), 5000):
        seg = rx[seg_start:seg_start + seg_len]
        c, s = estimate_cfo_from_tone(seg, fs, 100e3)
        cfo_estimates.append((c, s, seg_start))
        if s > best_snr:
            best_snr = s
            best_cfo = c

    cfo = best_cfo
    tone_snr = best_snr
    rx_cfo = apply_cfo_correction(rx, cfo, fs)

    # ---- STF detection: Schmidl-Cox autocorrelation (robust to residual CFO) ----
    sc_detected, sc_idx, sc_peak, sc_cfo_norm, sc_M = detect_stf_autocorr(
        rx_cfo, half_period=N_FFT // 2, threshold=0.1
    )

    # ---- STF detection: cross-correlation (backup) ----
    xc_idx, xc_peak, xc_corr = detect_stf_crosscorr(rx_cfo, stf_ref)

    # ---- Combined detection: energy envelope + cross-corr refinement ----
    # Strategy:
    #   1. Find frame boundaries using energy envelope (tone→gap→STF transition)
    #   2. Cross-correlate STF in narrow window around each candidate
    #   3. Pick best candidate using LTF channel estimate quality

    energy_candidates = detect_frame_by_energy(rx_cfo, fs)

    # Also add SC and global XC candidates
    all_candidates = list(energy_candidates)
    if sc_detected:
        all_candidates.append(sc_idx)
    if xc_peak > 0.02:
        all_candidates.append(xc_idx)

    # For each candidate: refine with cross-corr, estimate channel, trial-demod 5 symbols
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
        # Local cross-corr refinement ±500 samples around candidate
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

        # Fine-tune with ±4 sample LTF quality sweep
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

        # Trial demodulation: channel est + demod first 5 OFDM symbols
        ltf_s = refined_idx + len(stf_ref)
        # Quick channel estimate from first LTF symbol
        Y_ltf = extract_ofdm_symbol(rx_cfo, ltf_s)
        if Y_ltf is None:
            continue
        H_trial = np.ones(N_FFT, dtype=np.complex64)
        for k in used:
            idx_k = (k + N_FFT) % N_FFT
            if np.abs(ltf_freq_ref[idx_k]) > 0.1:
                H_trial[idx_k] = Y_ltf[idx_k] / ltf_freq_ref[idx_k]

        # Demod 5 payload symbols, compute average EVM
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
            # Pilot phase
            ps = 1 if si % 2 == 0 else -1
            pe = np.angle(np.sum(Ye[p_idx] * np.conj(ps * pilot_pattern_ref)))
            trial_freq += 0.01 * pe
            trial_ph += trial_freq + 0.1 * pe
            Ye *= np.exp(-1j * trial_ph)
            # EVM
            ideal_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            nearest = np.array([ideal_pts[np.argmin(np.abs(s - ideal_pts))] for s in Ye[d_idx]])
            trial_evm.append(float(np.sqrt(np.mean(np.abs(Ye[d_idx] - nearest)**2))))

        if len(trial_evm) == 0:
            continue

        avg_evm = np.mean(trial_evm)
        # Score: lower EVM = better. Invert so higher score = better.
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
        'cfo': cfo,
        'tone_snr': tone_snr,
        'stf_idx': stf_idx,
        'stf_peak': stf_peak,
        'stf_method': method,
        'sc_peak': sc_peak,
        'sc_cfo_norm': sc_cfo_norm,
        'xc_peak': xc_peak,
        'sc_metric': sc_M,
        'xc_metric': xc_corr,
        'xc_idx': xc_idx,
        'rx': rx,
        'rx_cfo': rx_cfo,
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
        return result

    # ---- Channel estimation from LTF ----
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
        return result

    H, snr_per_sc = ch_result
    result['H'] = H
    result['snr_per_sc'] = snr_per_sc

    # ---- Demodulate OFDM payload with FLL+PLL phase tracking ----
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
    prev_pilot_phase = None  # For FLL

    # Track consecutive high-EVM symbols to detect end of payload
    bad_evm_count = 0
    EVM_CUTOFF = 0.8

    for sym_idx in range(max_ofdm_syms):
        sym_start = payload_start + sym_idx * SYMBOL_LEN
        Y = extract_ofdm_symbol(rx_cfo, sym_start)
        if Y is None:
            break

        # Equalize with channel estimate
        Y_eq = np.zeros_like(Y)
        for k in range(-26, 27):
            if k == 0:
                continue
            idx = (k + N_FFT) % N_FFT
            if np.abs(H[idx]) > 1e-6:
                Y_eq[idx] = Y[idx] / H[idx]

        # Pilot-based phase tracking: combined FLL + PLL
        pilot_sign = 1 if sym_idx % 2 == 0 else -1
        expected_pilots = pilot_sign * pilot_pattern
        rx_pilots = Y_eq[pilot_idx]

        # Phase error from pilot correlation
        pilot_corr = np.sum(rx_pilots * np.conj(expected_pilots))
        phase_err = np.angle(pilot_corr)

        # FLL: estimate frequency from rate of change of pilot phase
        # This handles large residual CFO without phase wrapping issues
        if prev_pilot_phase is not None:
            # Frequency error = change in phase between consecutive symbols
            freq_err = phase_err - prev_pilot_phase
            # Unwrap frequency error to handle ±π wrapping
            while freq_err > np.pi:
                freq_err -= 2 * np.pi
            while freq_err < -np.pi:
                freq_err += 2 * np.pi
            # FLL update: adjust frequency accumulator
            freq_acc += ki * freq_err
        prev_pilot_phase = phase_err

        # PLL update: adjust phase accumulator
        phase_acc += freq_acc + kp * phase_err

        # Apply phase correction to all subcarriers
        Y_eq *= np.exp(-1j * phase_acc)

        data_syms = Y_eq[data_idx]

        # EVM for this symbol
        ideal_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        nearest = np.array([ideal_pts[np.argmin(np.abs(s - ideal_pts))] for s in data_syms])
        sym_evm = float(np.sqrt(np.mean(np.abs(data_syms - nearest)**2)))

        # Detect end of payload
        if not ber_test and sym_evm > EVM_CUTOFF:
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

    result['all_data_syms'] = np.concatenate(all_data_syms) if len(all_data_syms) > 0 else np.array([], dtype=np.complex64)
    result['phase_errors'] = np.array(phase_errors)
    result['freq_log'] = np.array(freq_log)
    result['pilot_powers'] = np.array(pilot_powers)
    result['evm_per_sym'] = np.array(evm_per_sym)

    if len(all_data_syms) == 0:
        result['status'] = 'no_payload'
        return result

    # ---- Bit recovery ----
    all_syms_flat = result['all_data_syms']
    bits_raw = qpsk_demap(all_syms_flat)
    bits = majority_vote(bits_raw, repeat)
    bits_bytes = np.packbits(bits).tobytes()

    if ber_test and ber_ref_bits is not None:
        # BER test mode
        num_bits = min(len(bits_raw), len(ber_ref_bits))
        bit_errors = int(np.sum(bits_raw[:num_bits] != ber_ref_bits[:num_bits]))
        ber = bit_errors / num_bits
        result['status'] = 'ber_test'
        result['ber'] = ber
        result['bit_errors'] = bit_errors
        result['total_bits'] = num_bits
    else:
        # Packet mode
        success, payload, crc_rx, crc_calc = parse_packet(bits_bytes)
        result['crc_rx'] = crc_rx
        result['crc_calc'] = crc_calc
        result['payload'] = payload
        if success:
            result['status'] = 'crc_ok'
        elif bits_bytes[:4] == MAGIC:
            result['status'] = 'crc_fail'
        else:
            result['status'] = 'no_magic'

    result['mean_evm'] = float(np.mean(evm_per_sym)) if len(evm_per_sym) > 0 else 0.0
    return result


# ============================================================================
# Entry Point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="RF Link Test - RX (Step 5)")
    ap.add_argument("--uri", default="ip:192.168.2.2", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--rx_gain", type=float, default=40, help="RX gain (dB)")
    ap.add_argument("--buf_size", type=int, default=2**17, help="RX buffer size")
    ap.add_argument("--max_ofdm_syms", type=int, default=200, help="Max OFDM symbols")
    ap.add_argument("--stf_repeats", type=int, default=6, help="STF repeats (must match TX)")
    ap.add_argument("--ltf_symbols", type=int, default=4, help="LTF symbols (must match TX)")
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Bit repetition")
    ap.add_argument("--tries", type=int, default=10, help="Number of captures")
    ap.add_argument("--output_dir", default="rf_link_step5_results", help="Output directory")
    ap.add_argument("--outfile", default="recovered.bin", help="Output file")
    ap.add_argument("--kp", type=float, default=0.1, help="Phase tracking Kp")
    ap.add_argument("--ki", type=float, default=0.01, help="Phase tracking Ki")
    ap.add_argument("--ber_test", action="store_true", help="BER stress test mode")
    ap.add_argument("--ber_num_syms", type=int, default=100, help="OFDM symbols for BER test")
    ap.add_argument("--ber_captures", type=int, default=20, help="Captures for BER test")
    args = ap.parse_args()

    import adi

    os.makedirs(args.output_dir, exist_ok=True)

    # Create reference signals (must match TX parameters)
    stf_ref, stf_freq = create_schmidl_cox_stf(N_FFT, num_repeats=args.stf_repeats)
    ltf_ref, ltf_freq_ref = create_ltf_ref(N_FFT, num_symbols=args.ltf_symbols)

    # BER test reference bits
    ber_ref_bits = None
    if args.ber_test:
        rng = np.random.default_rng(99999)
        ber_ref_bits = rng.integers(0, 2, size=args.ber_num_syms * BITS_PER_OFDM_SYM).astype(np.uint8)

    print(f"RF Link Test - Step 5: RX (Robust Sync + Diagnostics)")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
    print(f"  STF repeats: {args.stf_repeats}, LTF symbols: {args.ltf_symbols}")
    if args.ber_test:
        print(f"  MODE: BER stress test ({args.ber_captures} captures, {args.ber_num_syms} syms)")
    else:
        print(f"  MODE: Packet reception (max {args.tries} tries)")
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

    num_captures = args.ber_captures if args.ber_test else args.tries
    ber_results = []

    print(f"Starting {num_captures} captures...")
    print("-" * 70)

    for t in range(num_captures):
        rx_raw = sdr.rx()

        r = demod_one_capture(
            rx_raw, args.fs, stf_ref, ltf_freq_ref,
            args.stf_repeats, args.ltf_symbols,
            args.ber_num_syms if args.ber_test else args.max_ofdm_syms,
            args.kp, args.ki, args.repeat,
            ber_test=args.ber_test, ber_ref_bits=ber_ref_bits
        )

        # Status line
        status = r['status']
        if status == 'no_signal':
            print(f"[{t+1:02d}] No signal detected (SC={r['sc_peak']:.4f}, XC={r['xc_peak']:.4f})")
        elif status == 'ltf_fail':
            print(f"[{t+1:02d}] LTF extraction failed (STF@{r['stf_idx']}, {r['stf_method']})")
        elif status == 'no_payload':
            print(f"[{t+1:02d}] No payload decoded")
        elif status == 'ber_test':
            ber = r['ber']
            ber_results.append({
                'capture': t + 1,
                'ber': ber,
                'mean_evm': r['mean_evm'],
                'cfo': r['cfo'],
                'stf_peak': r['stf_peak'],
                'bit_errors': r['bit_errors'],
                'total_bits': r['total_bits'],
            })
            sym_indicator = "OK" if ber < 0.01 else "HIGH"
            print(f"[{t+1:02d}] BER={ber:.2e} ({r['bit_errors']}/{r['total_bits']}) "
                  f"EVM={r['mean_evm']:.4f} CFO={r['cfo']:.1f}Hz "
                  f"STF={r['stf_peak']:.4f}({r['stf_method']}) [{sym_indicator}]")
        elif status == 'crc_ok':
            payload = r['payload']
            print(f"[{t+1:02d}] CRC OK! {len(payload)} bytes, "
                  f"CFO={r['cfo']:.1f}Hz, EVM={r['mean_evm']:.4f}")
            with open(args.outfile, "wb") as f:
                f.write(payload)
            try:
                print(f"      Payload: {payload}")
            except:
                print(f"      Payload (hex): {payload[:32].hex()}...")
        elif status == 'crc_fail':
            print(f"[{t+1:02d}] CRC FAIL rx=0x{r['crc_rx']:08X} calc=0x{r['crc_calc']:08X} "
                  f"CFO={r['cfo']:.1f}Hz EVM={r['mean_evm']:.4f}")
        elif status == 'no_magic':
            print(f"[{t+1:02d}] Magic not found "
                  f"CFO={r['cfo']:.1f}Hz STF={r['stf_peak']:.4f}({r['stf_method']})")

        # Build result string for plot
        if status == 'ber_test':
            result_str = f"BER = {r['ber']:.2e}  ({r['bit_errors']}/{r['total_bits']} errors)"
        elif status == 'crc_ok':
            result_str = f"CRC OK - {len(r['payload'])} bytes received"
        elif status == 'crc_fail':
            result_str = f"CRC FAIL: 0x{r['crc_rx']:08X} vs 0x{r['crc_calc']:08X}"
        else:
            result_str = f"Status: {status}"

        # Generate diagnostic plot for every capture
        plot_path = plot_full_diagnostics(
            r['rx'], r['rx_cfo'], r['cfo'],
            r['stf_idx'], r['stf_peak'],
            r['sc_metric'], r['xc_metric'], r['xc_idx'],
            r['H'], r['snr_per_sc'],
            r['all_data_syms'], r['phase_errors'], r['freq_log'],
            r['pilot_powers'], r['evm_per_sym'],
            result_str, t + 1, args.output_dir, args.fs,
            stf_cfo_norm=r['sc_cfo_norm']
        )

        # Stop early on successful packet reception
        if status == 'crc_ok' and not args.ber_test:
            print(f"\n  Diagnostic plot: {plot_path}")
            print(f"  Payload saved: {args.outfile}")
            break

    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if args.ber_test and len(ber_results) > 0:
        bers = [r['ber'] for r in ber_results]
        evms = [r['mean_evm'] for r in ber_results]
        n_ok = sum(1 for b in bers if b < 0.01)

        print(f"  BER Stress Test: {len(ber_results)} captures")
        print(f"  BER: mean={np.mean(bers):.2e}, min={np.min(bers):.2e}, max={np.max(bers):.2e}")
        print(f"  EVM: mean={np.mean(evms):.4f}")
        print(f"  Captures < 1% BER: {n_ok}/{len(ber_results)}")
        print(f"  Captures = 0% BER: {sum(1 for b in bers if b == 0)}/{len(ber_results)}")

        # Generate BER summary plot
        summary_path = plot_ber_summary(ber_results, args.output_dir)
        print(f"\n  Summary plot: {summary_path}")

        if np.mean(bers) == 0:
            print("\n  PERFECT - 0% BER across all captures!")
        elif np.mean(bers) < 0.001:
            print("\n  EXCELLENT - BER < 0.1%")
        elif np.mean(bers) < 0.01:
            print("\n  GOOD - BER < 1%")
        else:
            print("\n  NEEDS IMPROVEMENT - BER > 1%")
    else:
        print("  See diagnostic plots in:", args.output_dir)

    # Cleanup
    try:
        sdr.rx_destroy_buffer()
    except:
        pass


if __name__ == "__main__":
    main()
