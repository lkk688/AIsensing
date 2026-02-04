#!/usr/bin/env python3
"""
RF Link Test - Step 1: TX (Tone + Preamble)

Transmits a simple signal for testing RF connectivity:
1. Continuous tone for easy detection and CFO measurement
2. Known preamble pattern for sync testing

Run on the remote TX device (Jetson).
"""

import argparse
import time
import numpy as np

def create_schmidl_cox_preamble(N=64, num_repeats=4):
    """
    Create Schmidl-Cox style preamble with repeated structure.

    The preamble consists of a pattern repeated multiple times.
    This allows:
    - Timing sync via autocorrelation
    - CFO estimation from phase difference between repeats
    """
    # Generate random BPSK sequence for N/2 samples
    rng = np.random.default_rng(12345)  # Fixed seed for reproducibility
    half_n = N // 2
    bpsk = rng.choice([-1.0, 1.0], size=half_n).astype(np.float32)

    # Create one period: BPSK in freq domain, only on even subcarriers
    X = np.zeros(N, dtype=np.complex64)
    X[::2] = bpsk  # Only even subcarriers

    # IFFT to get time domain
    x = np.fft.ifft(X) * np.sqrt(N)

    # The key property: x has period N/2 in time domain!
    # So first half equals second half

    # Repeat the full symbol multiple times for robust sync
    preamble = np.tile(x, num_repeats).astype(np.complex64)

    return preamble


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - TX (Step 1)")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=-10, help="TX gain (dB)")
    ap.add_argument("--mode", choices=["tone", "preamble", "both"], default="both")
    args = ap.parse_args()

    import adi

    # Create signals
    N = 64

    # Tone: 100 kHz offset from center
    t = np.arange(8192) / args.fs
    tone = np.exp(2j * np.pi * 100e3 * t).astype(np.complex64)

    # Schmidl-Cox preamble
    preamble = create_schmidl_cox_preamble(N, num_repeats=10)  # 640 samples

    # Gap (zeros)
    gap = np.zeros(2000, dtype=np.complex64)

    if args.mode == "tone":
        tx_signal = tone
    elif args.mode == "preamble":
        tx_signal = np.concatenate([gap, preamble, gap])
    else:  # both
        # Continuous signal: gap, preamble, tone, gap (repeated)
        tx_signal = np.concatenate([gap, preamble, gap, tone, gap])

    # Normalize
    tx_signal = tx_signal / (np.max(np.abs(tx_signal)) + 1e-9) * 0.7

    # Scale for DAC
    tx_scaled = (tx_signal * 2**14).astype(np.complex64)

    print(f"TX Signal: {len(tx_scaled)} samples")
    print(f"  Mode: {args.mode}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  TX gain: {args.tx_gain} dB")

    # Configure SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    sdr.tx_cyclic_buffer = True

    # Transmit
    sdr.tx(tx_scaled)

    print("\nTX Running (Ctrl+C to stop)")
    print(f"  Tone at {args.fc/1e6:.3f} MHz + 100 kHz = {(args.fc + 100e3)/1e6:.3f} MHz")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping TX...")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except:
            pass


if __name__ == "__main__":
    main()
