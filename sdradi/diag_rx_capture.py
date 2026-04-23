#!/usr/bin/env python3
"""
Minimal diagnostic: capture raw IQ from PlutoSDR and analyze.
No demodulation - just check power levels and spectrum.
"""
import argparse
import numpy as np
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--rx_gain", type=float, default=30)
    ap.add_argument("--buf", type=int, default=2**17)
    ap.add_argument("--num", type=int, default=20, help="Number of captures")
    args = ap.parse_args()

    import adi

    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.fs * 1.2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = args.buf

    # Flush
    for _ in range(3):
        sdr.rx()

    print(f"Capturing {args.num} buffers @ fc={args.fc/1e6:.1f}MHz fs={args.fs/1e6:.1f}Msps gain={args.rx_gain}dB buf={args.buf}")
    print(f"{'#':>4}  {'max':>8}  {'rms':>8}  {'peak_dB':>8}  {'tone_100k':>10}  {'ratio':>6}")
    print("-" * 60)

    for i in range(args.num):
        rx_raw = sdr.rx()
        rx = rx_raw.astype(np.complex64)
        if np.max(np.abs(rx)) > 10:
            rx = rx / (2 ** 14)

        mx = float(np.max(np.abs(rx)))
        rms = float(np.sqrt(np.mean(np.abs(rx) ** 2)))
        peak_db = 20 * np.log10(mx + 1e-12)

        # Check for 100kHz tone
        N = len(rx)
        fft = np.abs(np.fft.fft(rx * np.hanning(N)))
        fft[0] = 0
        freq_bins = np.fft.fftfreq(N, 1/args.fs)

        # Find bin closest to 100kHz
        tone_bin = int(np.argmin(np.abs(freq_bins - 100e3)))
        # Search ±5 bins around expected 100kHz
        lo, hi = max(1, tone_bin - 5), min(N // 2, tone_bin + 6)
        tone_peak = float(np.max(fft[lo:hi]))
        median_fft = float(np.median(fft[1:N//2]))
        tone_ratio = tone_peak / (median_fft + 1e-12)

        # Also find the global peak frequency
        peak_idx = int(np.argmax(fft[1:N//2])) + 1
        peak_freq = freq_bins[peak_idx]

        print(f"{i+1:4d}  {mx:8.4f}  {rms:8.4f}  {peak_db:8.1f}  {tone_ratio:10.1f}  {peak_freq/1e3:6.1f}kHz")

    try:
        sdr.rx_destroy_buffer()
    except:
        pass

if __name__ == "__main__":
    main()
