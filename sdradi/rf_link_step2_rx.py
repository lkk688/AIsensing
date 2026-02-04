#!/usr/bin/env python3
"""
RF Link Test - Step 2: RX with CFO Correction

Builds on Step 1:
1. Estimates CFO from tone
2. Applies CFO correction to full signal
3. Performs preamble detection on corrected signal
4. Verifies CFO-corrected constellation is stable

Run on the local RX device.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def create_schmidl_cox_preamble(N=64, num_repeats=4):
    """Create the same preamble as TX for correlation."""
    rng = np.random.default_rng(12345)
    half_n = N // 2
    bpsk = rng.choice([-1.0, 1.0], size=half_n).astype(np.float32)
    X = np.zeros(N, dtype=np.complex64)
    X[::2] = bpsk
    x = np.fft.ifft(X) * np.sqrt(N)
    preamble = np.tile(x, num_repeats).astype(np.complex64)
    return preamble


def estimate_cfo_from_tone(rx_samples, fs, expected_tone_freq=100e3):
    """Estimate CFO by finding the tone in the received signal."""
    N = len(rx_samples)
    fft = np.fft.fft(rx_samples)
    fft_mag = np.abs(fft)

    # Remove DC and near-DC
    fft_mag[0] = 0
    fft_mag[1:20] = 0
    fft_mag[-20:] = 0

    peak_idx = np.argmax(fft_mag)
    freq_bins = np.fft.fftfreq(N, 1/fs)
    detected_freq = freq_bins[peak_idx]

    # Parabolic interpolation for better precision
    if 1 < peak_idx < N-1:
        alpha = np.log(fft_mag[peak_idx-1] + 1e-10)
        beta = np.log(fft_mag[peak_idx] + 1e-10)
        gamma = np.log(fft_mag[peak_idx+1] + 1e-10)
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        detected_freq = freq_bins[peak_idx] + p * (freq_bins[1] - freq_bins[0])

    cfo = detected_freq - expected_tone_freq
    tone_power = fft_mag[peak_idx]**2 / N

    return detected_freq, cfo, tone_power


def apply_cfo_correction(samples, cfo, fs):
    """Apply CFO correction to the signal."""
    n = np.arange(len(samples))
    correction = np.exp(-1j * 2 * np.pi * cfo * n / fs).astype(np.complex64)
    return samples * correction


def schmidl_cox_sync(rx_samples, half_period=32):
    """Schmidl-Cox timing and CFO estimation."""
    N = len(rx_samples)
    L = half_period

    P = np.zeros(N - 2*L, dtype=np.complex64)
    R = np.zeros(N - 2*L, dtype=np.float32)

    for n in range(len(P)):
        seg1 = rx_samples[n:n+L]
        seg2 = rx_samples[n+L:n+2*L]
        P[n] = np.sum(seg1 * np.conj(seg2))
        R[n] = np.sum(np.abs(seg2)**2)

    eps = 1e-10
    M = np.abs(P)**2 / (R**2 + eps)

    peak_idx = np.argmax(M)
    phase_diff = np.angle(P[peak_idx])
    coarse_cfo = -phase_diff / (2 * np.pi * L)

    return M, coarse_cfo, peak_idx


def correlate_with_preamble(rx_samples, preamble):
    """Cross-correlation with known preamble."""
    corr = np.abs(np.correlate(rx_samples, preamble, mode='valid'))
    preamble_energy = np.sqrt(np.sum(np.abs(preamble)**2))
    corr = corr / preamble_energy

    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]

    return corr, peak_idx, peak_val


def fine_cfo_from_preamble(rx_samples, preamble_start, symbol_len=64):
    """
    Fine CFO estimation from preamble using phase difference between identical halves.
    The Schmidl-Cox preamble has period N/2, so we can measure phase drift.
    """
    L = symbol_len // 2  # Half period

    # Use multiple half-periods for averaging
    phases = []
    for i in range(4):  # 4 pairs of half-periods
        start = preamble_start + i * L
        if start + 2*L > len(rx_samples):
            break
        seg1 = rx_samples[start:start+L]
        seg2 = rx_samples[start+L:start+2*L]
        P = np.sum(seg1 * np.conj(seg2))
        phases.append(np.angle(P))

    if len(phases) == 0:
        return 0.0

    # Average phase difference per L samples
    avg_phase = np.mean(phases)
    return avg_phase


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - RX (Step 2)")
    ap.add_argument("--uri", default="ip:192.168.2.2", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--rx_gain", type=float, default=40, help="RX gain (dB)")
    ap.add_argument("--buf_size", type=int, default=2**17, help="RX buffer size")
    ap.add_argument("--num_captures", type=int, default=5, help="Number of captures")
    ap.add_argument("--output_dir", default="rf_link_step2_results", help="Output directory")
    args = ap.parse_args()

    import adi

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate reference preamble
    N = 64
    preamble = create_schmidl_cox_preamble(N, num_repeats=10)

    print(f"RF Link Test - Step 2: RX with CFO Correction")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
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

    print("Starting captures...")
    print("-" * 60)

    results = []

    for i in range(args.num_captures):
        rx_raw = sdr.rx()
        rx = rx_raw.astype(np.complex64) / (2**14)

        # Remove DC
        dc = np.mean(rx)
        rx_no_dc = rx - dc

        # Step 1: Estimate CFO from tone
        tone_freq, tone_cfo, tone_power = estimate_cfo_from_tone(rx_no_dc, args.fs, 100e3)

        # Step 2: Apply coarse CFO correction
        rx_cfo_corrected = apply_cfo_correction(rx_no_dc, tone_cfo, args.fs)

        # Step 3: Preamble detection on CFO-corrected signal
        corr_before, pre_idx_before, pre_peak_before = correlate_with_preamble(rx_no_dc, preamble)
        corr_after, pre_idx_after, pre_peak_after = correlate_with_preamble(rx_cfo_corrected, preamble)

        # Step 4: Schmidl-Cox on corrected signal
        M, sc_cfo_norm, sc_peak_idx = schmidl_cox_sync(rx_cfo_corrected, half_period=32)
        sc_cfo_hz = sc_cfo_norm * args.fs

        # Step 5: Fine CFO from preamble phase
        fine_phase = fine_cfo_from_preamble(rx_cfo_corrected, pre_idx_after, symbol_len=64)
        fine_cfo_hz = -fine_phase * args.fs / (2 * np.pi * 32)  # Per sample normalized

        # Verify residual CFO after correction
        residual_cfo = estimate_cfo_from_tone(rx_cfo_corrected, args.fs, 100e3)[1]

        result = {
            'capture': i + 1,
            'tone_cfo': tone_cfo,
            'pre_peak_before': pre_peak_before,
            'pre_peak_after': pre_peak_after,
            'residual_cfo': residual_cfo,
            'sc_cfo_hz': sc_cfo_hz,
            'improvement': pre_peak_after / (pre_peak_before + 1e-10),
        }
        results.append(result)

        print(f"Capture {i+1}/{args.num_captures}:")
        print(f"  Tone CFO: {tone_cfo:.1f} Hz")
        print(f"  Preamble peak BEFORE CFO corr: {pre_peak_before:.4f}")
        print(f"  Preamble peak AFTER CFO corr:  {pre_peak_after:.4f}")
        print(f"  Improvement: {result['improvement']:.2f}x")
        print(f"  Residual CFO: {residual_cfo:.1f} Hz")
        print(f"  Schmidl-Cox residual: {sc_cfo_hz:.1f} Hz")
        print()

        # Create diagnostic plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Spectrum before and after CFO correction
        ax = axes[0, 0]
        N_fft = min(16384, len(rx_no_dc))
        fft_before = np.fft.fftshift(np.fft.fft(rx_no_dc[:N_fft]))
        fft_after = np.fft.fftshift(np.fft.fft(rx_cfo_corrected[:N_fft]))
        freq = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/args.fs)) / 1e3
        ax.plot(freq, 20*np.log10(np.abs(fft_before) + 1e-10), alpha=0.7, label='Before')
        ax.plot(freq, 20*np.log10(np.abs(fft_after) + 1e-10), alpha=0.7, label='After CFO corr')
        ax.axvline(100, color='r', linestyle='--', alpha=0.5, label='100 kHz')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'Spectrum (CFO={tone_cfo:.1f} Hz)')
        ax.legend()
        ax.grid(True)
        ax.set_xlim([-200, 200])

        # 2. Preamble correlation comparison
        ax = axes[0, 1]
        disp_len = min(20000, len(corr_before), len(corr_after))
        ax.plot(corr_before[:disp_len], alpha=0.7, label=f'Before: {pre_peak_before:.4f}')
        ax.plot(corr_after[:disp_len], alpha=0.7, label=f'After: {pre_peak_after:.4f}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Preamble Correlation ({result["improvement"]:.2f}x improvement)')
        ax.legend()
        ax.grid(True)

        # 3. IQ Constellation before CFO correction (subset around preamble)
        ax = axes[0, 2]
        start = max(0, pre_idx_before - 500)
        end = min(len(rx_no_dc), pre_idx_before + 1500)
        subset = rx_no_dc[start:end:5]
        ax.scatter(np.real(subset), np.imag(subset), s=2, alpha=0.5)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title('IQ Before CFO Correction')
        ax.axis('equal')
        ax.grid(True)

        # 4. IQ Constellation after CFO correction
        ax = axes[1, 0]
        start = max(0, pre_idx_after - 500)
        end = min(len(rx_cfo_corrected), pre_idx_after + 1500)
        subset = rx_cfo_corrected[start:end:5]
        ax.scatter(np.real(subset), np.imag(subset), s=2, alpha=0.5)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title('IQ After CFO Correction')
        ax.axis('equal')
        ax.grid(True)

        # 5. Schmidl-Cox timing metric on corrected signal
        ax = axes[1, 1]
        ax.plot(M[:min(20000, len(M))])
        ax.axvline(sc_peak_idx, color='r', linestyle='--', label=f'Peak at {sc_peak_idx}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Timing Metric')
        ax.set_title(f'Schmidl-Cox (residual CFO={sc_cfo_hz:.1f} Hz)')
        ax.legend()
        ax.grid(True)

        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""
RF Link Test - Step 2 - Capture {i+1}

CFO Estimation:
  Tone-based CFO: {tone_cfo:.1f} Hz
  After correction residual: {residual_cfo:.1f} Hz

Preamble Detection:
  Before CFO corr: {pre_peak_before:.4f}
  After CFO corr:  {pre_peak_after:.4f}
  Improvement: {result['improvement']:.2f}x

Schmidl-Cox (on corrected):
  Residual CFO: {sc_cfo_hz:.1f} Hz
  Peak index: {sc_peak_idx}
"""
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        fig.suptitle(f'RF Link Step 2 - Capture {i+1}', fontsize=14)
        fig.tight_layout()

        plot_path = os.path.join(args.output_dir, f'capture_{i+1:02d}.png')
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {plot_path}")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    tone_cfos = [r['tone_cfo'] for r in results]
    improvements = [r['improvement'] for r in results]
    pre_peaks_after = [r['pre_peak_after'] for r in results]

    print(f"Tone CFO: mean={np.mean(tone_cfos):.1f} Hz, std={np.std(tone_cfos):.1f} Hz")
    print(f"Preamble peak (after): mean={np.mean(pre_peaks_after):.4f}, std={np.std(pre_peaks_after):.4f}")
    print(f"Improvement: mean={np.mean(improvements):.2f}x")

    if np.mean(pre_peaks_after) > 0.5:
        print("\n✅ CFO correction significantly improved preamble detection!")
        print("   Ready for Step 3: OFDM demodulation")
    elif np.mean(pre_peaks_after) > 0.3:
        print("\n⚠️  CFO correction helped, but preamble detection still marginal")
        print("   May need to adjust TX power or RX gain")
    else:
        print("\n❌ Preamble detection still weak after CFO correction")
        print("   Check: TX power, cable connection, attenuator")

    # Cleanup
    try:
        sdr.rx_destroy_buffer()
    except:
        pass


if __name__ == "__main__":
    main()
