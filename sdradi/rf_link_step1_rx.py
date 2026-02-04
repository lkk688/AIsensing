#!/usr/bin/env python3
"""
RF Link Test - Step 1: RX (Tone Detection + CFO Measurement)

Receives and analyzes the test signal from TX:
1. Detects the tone and measures frequency offset (CFO)
2. Detects the preamble and measures correlation strength
3. Provides diagnostic information about the RF link

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
    """
    Estimate CFO by finding the tone in the received signal.

    Returns:
        detected_freq: Detected tone frequency
        cfo: Carrier frequency offset (detected - expected)
        tone_power: Power at the detected frequency
    """
    # FFT to find tone
    N = len(rx_samples)
    fft = np.fft.fft(rx_samples)
    fft_mag = np.abs(fft)

    # Find peak (excluding DC)
    fft_mag[0] = 0  # Remove DC
    fft_mag[1:10] = 0  # Remove near-DC
    fft_mag[-10:] = 0

    peak_idx = np.argmax(fft_mag)
    freq_bins = np.fft.fftfreq(N, 1/fs)
    detected_freq = freq_bins[peak_idx]

    # CFO = detected - expected
    cfo = detected_freq - expected_tone_freq

    # Tone power
    tone_power = fft_mag[peak_idx]**2 / N

    return detected_freq, cfo, tone_power


def schmidl_cox_sync(rx_samples, half_period=32):
    """
    Schmidl-Cox timing and CFO estimation.

    Uses the property that the preamble has period N/2 in time domain.

    Returns:
        timing_metric: P[n] / (R[n] + eps) for each sample
        coarse_cfo: Estimated CFO from phase of correlation peak
        peak_idx: Index of detected preamble start
    """
    N = len(rx_samples)
    L = half_period  # Half period length

    # P[n] = sum of r[n+k] * conj(r[n+k+L]) for k=0..L-1
    # This correlates the signal with itself delayed by L
    P = np.zeros(N - 2*L, dtype=np.complex64)
    R = np.zeros(N - 2*L, dtype=np.float32)

    for n in range(len(P)):
        seg1 = rx_samples[n:n+L]
        seg2 = rx_samples[n+L:n+2*L]
        P[n] = np.sum(seg1 * np.conj(seg2))
        R[n] = np.sum(np.abs(seg2)**2)

    # Timing metric
    eps = 1e-10
    M = np.abs(P)**2 / (R**2 + eps)

    # Find peak
    peak_idx = np.argmax(M)

    # CFO from phase of P at peak
    # Phase accumulates over L samples
    phase_diff = np.angle(P[peak_idx])
    coarse_cfo = -phase_diff / (2 * np.pi * L)  # Normalized frequency

    return M, coarse_cfo, peak_idx, P[peak_idx]


def correlate_with_preamble(rx_samples, preamble):
    """
    Cross-correlation with known preamble.

    Returns correlation magnitude and best alignment index.
    """
    corr = np.abs(np.correlate(rx_samples, preamble, mode='valid'))

    # Normalize
    preamble_energy = np.sqrt(np.sum(np.abs(preamble)**2))
    corr = corr / preamble_energy

    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]

    return corr, peak_idx, peak_val


def analyze_dc_offset(rx_samples):
    """Analyze DC offset in the received signal."""
    dc = np.mean(rx_samples)
    dc_power = np.abs(dc)**2
    signal_power = np.mean(np.abs(rx_samples - dc)**2)
    dc_ratio_db = 10 * np.log10(dc_power / (signal_power + 1e-10) + 1e-10)
    return dc, dc_power, signal_power, dc_ratio_db


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - RX (Step 1)")
    ap.add_argument("--uri", default="ip:192.168.2.2", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--rx_gain", type=float, default=40, help="RX gain (dB)")
    ap.add_argument("--buf_size", type=int, default=2**17, help="RX buffer size")
    ap.add_argument("--num_captures", type=int, default=5, help="Number of captures")
    ap.add_argument("--output_dir", default="rf_link_step1_results", help="Output directory")
    args = ap.parse_args()

    import adi

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate reference preamble
    N = 64
    preamble = create_schmidl_cox_preamble(N, num_repeats=10)

    print(f"RF Link Test - Step 1: RX")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
    print(f"  Buffer size: {args.buf_size}")
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
        # Receive
        rx_raw = sdr.rx()
        rx = rx_raw.astype(np.complex64) / (2**14)

        # Remove DC
        dc, dc_power, sig_power, dc_ratio_db = analyze_dc_offset(rx)
        rx_no_dc = rx - dc

        # Tone detection and CFO
        tone_freq, tone_cfo, tone_power = estimate_cfo_from_tone(rx_no_dc, args.fs, 100e3)

        # Schmidl-Cox sync
        M, sc_cfo_norm, sc_peak_idx, sc_P = schmidl_cox_sync(rx_no_dc, half_period=32)
        sc_cfo_hz = sc_cfo_norm * args.fs

        # Preamble correlation
        corr, pre_peak_idx, pre_peak_val = correlate_with_preamble(rx_no_dc, preamble)

        # Signal stats
        rx_power_db = 10 * np.log10(np.mean(np.abs(rx)**2) + 1e-10)

        result = {
            'capture': i + 1,
            'rx_power_db': rx_power_db,
            'dc_ratio_db': dc_ratio_db,
            'tone_freq': tone_freq,
            'tone_cfo': tone_cfo,
            'tone_power': tone_power,
            'sc_cfo_hz': sc_cfo_hz,
            'sc_peak_metric': np.max(M),
            'pre_peak_val': pre_peak_val,
        }
        results.append(result)

        print(f"Capture {i+1}/{args.num_captures}:")
        print(f"  RX Power: {rx_power_db:.1f} dB")
        print(f"  DC Ratio: {dc_ratio_db:.1f} dB")
        print(f"  Tone: detected={tone_freq/1e3:.2f} kHz, CFO={tone_cfo:.1f} Hz")
        print(f"  Schmidl-Cox: CFO={sc_cfo_hz:.1f} Hz, metric_peak={np.max(M):.4f}")
        print(f"  Preamble corr: peak={pre_peak_val:.4f} at idx={pre_peak_idx}")
        print()

        # Create diagnostic plot for this capture
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Time domain signal
        ax = axes[0, 0]
        t_ms = np.arange(min(10000, len(rx))) / args.fs * 1000
        ax.plot(t_ms, np.real(rx[:len(t_ms)]), alpha=0.7, label='I')
        ax.plot(t_ms, np.imag(rx[:len(t_ms)]), alpha=0.7, label='Q')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Time Domain (first 10k samples)')
        ax.legend()
        ax.grid(True)

        # 2. Spectrum
        ax = axes[0, 1]
        fft = np.fft.fftshift(np.fft.fft(rx_no_dc[:args.buf_size//4]))
        freq = np.fft.fftshift(np.fft.fftfreq(len(fft), 1/args.fs)) / 1e3
        ax.plot(freq, 20*np.log10(np.abs(fft) + 1e-10))
        ax.axvline(100, color='r', linestyle='--', label='Expected tone')
        ax.axvline(tone_freq/1e3, color='g', linestyle=':', label=f'Detected: {tone_freq/1e3:.1f}kHz')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'Spectrum (CFO={tone_cfo:.1f} Hz)')
        ax.legend()
        ax.grid(True)

        # 3. Schmidl-Cox timing metric
        ax = axes[0, 2]
        ax.plot(M[:min(20000, len(M))])
        ax.axvline(sc_peak_idx, color='r', linestyle='--', label=f'Peak at {sc_peak_idx}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Timing Metric')
        ax.set_title(f'Schmidl-Cox (CFO={sc_cfo_hz:.1f} Hz)')
        ax.legend()
        ax.grid(True)

        # 4. Preamble correlation
        ax = axes[1, 0]
        ax.plot(corr[:min(20000, len(corr))])
        ax.axvline(pre_peak_idx, color='r', linestyle='--', label=f'Peak at {pre_peak_idx}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Preamble Correlation (peak={pre_peak_val:.4f})')
        ax.legend()
        ax.grid(True)

        # 5. IQ constellation (subset)
        ax = axes[1, 1]
        subset = rx_no_dc[::10][:5000]
        ax.scatter(np.real(subset), np.imag(subset), s=1, alpha=0.3)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title('IQ Constellation')
        ax.axis('equal')
        ax.grid(True)

        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""
RF Link Test - Capture {i+1}

Signal Level:
  RX Power: {rx_power_db:.1f} dB
  DC Ratio: {dc_ratio_db:.1f} dB

Tone Analysis:
  Expected: 100 kHz
  Detected: {tone_freq/1e3:.2f} kHz
  CFO: {tone_cfo:.1f} Hz

Schmidl-Cox Sync:
  CFO estimate: {sc_cfo_hz:.1f} Hz
  Timing metric peak: {np.max(M):.4f}

Preamble Detection:
  Correlation peak: {pre_peak_val:.4f}
  Peak location: {pre_peak_idx}
"""
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        fig.suptitle(f'RF Link Step 1 - Capture {i+1}', fontsize=14)
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
    sc_cfos = [r['sc_cfo_hz'] for r in results]
    pre_peaks = [r['pre_peak_val'] for r in results]

    print(f"Tone CFO: mean={np.mean(tone_cfos):.1f} Hz, std={np.std(tone_cfos):.1f} Hz")
    print(f"SC CFO:   mean={np.mean(sc_cfos):.1f} Hz, std={np.std(sc_cfos):.1f} Hz")
    print(f"Preamble: mean_peak={np.mean(pre_peaks):.4f}, std={np.std(pre_peaks):.4f}")

    if np.mean(pre_peaks) > 0.1:
        print("\n✅ RF Link appears to be working!")
        print(f"   CFO is approximately {np.mean(tone_cfos):.0f} Hz")
    else:
        print("\n❌ RF Link may have issues - low preamble correlation")
        print("   Check: TX running? Cable connected? Gain settings?")

    # Cleanup
    try:
        sdr.rx_destroy_buffer()
    except:
        pass


if __name__ == "__main__":
    main()
