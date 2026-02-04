#!/usr/bin/env python3
"""
RF Link Test - Step 3: TX (OFDM with Pilot Tone)

Transmits OFDM signal with:
1. Leading pilot tone for CFO estimation (important!)
2. STF (Short Training Field) for timing sync
3. LTF (Long Training Field) for channel estimation
4. OFDM data symbols with QPSK and pilot subcarriers

Run on the remote TX device (Jetson).
"""

import argparse
import time
import numpy as np

# OFDM Parameters
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])  # IEEE 802.11 style
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in PILOT_SUBCARRIERS])
N_DATA = len(DATA_SUBCARRIERS)  # 48 data subcarriers


def qpsk_map(bits):
    """Map bits to QPSK symbols (Gray coded)."""
    symbols = np.zeros(len(bits)//2, dtype=np.complex64)
    for i in range(len(bits)//2):
        b0, b1 = bits[2*i], bits[2*i+1]
        if b0 == 0 and b1 == 0:
            symbols[i] = 1 + 1j
        elif b0 == 0 and b1 == 1:
            symbols[i] = -1 + 1j
        elif b0 == 1 and b1 == 1:
            symbols[i] = -1 - 1j
        else:  # 1, 0
            symbols[i] = 1 - 1j
    return symbols / np.sqrt(2)


def create_stf(N=64):
    """
    Create Short Training Field for timing sync.
    Uses only even subcarriers -> period N/2 in time domain.
    """
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)
    # BPSK on even used subcarriers
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0])
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i]

    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    # Repeat twice for better sync
    stf = np.tile(x, 2).astype(np.complex64)
    return stf


def create_ltf(N=64):
    """
    Create Long Training Field for channel estimation.
    Known sequence on all used subcarriers.
    """
    X = np.zeros(N, dtype=np.complex64)
    # Alternating BPSK pattern
    used = np.array([k for k in range(-26, 27) if k != 0])
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if i % 2 == 0 else -1.0

    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    # Repeat twice with CP
    ltf_sym = np.concatenate([x[-N_CP:], x])
    ltf = np.tile(ltf_sym, 2).astype(np.complex64)
    return ltf, X  # Return freq domain for RX reference


def create_ofdm_symbol(data_symbols, pilot_values, sym_idx):
    """Create one OFDM symbol with data and pilots."""
    X = np.zeros(N_FFT, dtype=np.complex64)

    # Insert data
    for i, k in enumerate(DATA_SUBCARRIERS):
        if i < len(data_symbols):
            X[(k + N_FFT) % N_FFT] = data_symbols[i]

    # Insert pilots (alternating pattern based on symbol index)
    pilot_sign = 1 if sym_idx % 2 == 0 else -1
    for i, k in enumerate(PILOT_SUBCARRIERS):
        X[(k + N_FFT) % N_FFT] = pilot_sign * pilot_values[i]

    # IFFT
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)

    # Add cyclic prefix
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - TX (Step 3)")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=-10, help="TX gain (dB)")
    ap.add_argument("--num_ofdm_syms", type=int, default=20, help="Number of OFDM data symbols")
    ap.add_argument("--tone_duration_ms", type=float, default=5, help="Pilot tone duration (ms)")
    args = ap.parse_args()

    import adi

    # Create pilot tone for CFO estimation (100 kHz offset)
    tone_samples = int(args.tone_duration_ms * args.fs / 1000)
    t = np.arange(tone_samples) / args.fs
    pilot_tone = np.exp(2j * np.pi * 100e3 * t).astype(np.complex64) * 0.5

    # Create preamble
    stf = create_stf(N_FFT)
    ltf, ltf_freq = create_ltf(N_FFT)

    # Add CP to STF
    stf_with_cp = np.concatenate([stf[-N_CP:], stf])

    # Create OFDM data symbols with known pattern
    rng = np.random.default_rng(12345)  # Fixed seed for debugging
    all_bits = rng.integers(0, 2, size=args.num_ofdm_syms * N_DATA * 2)

    pilot_values = np.array([1, 1, 1, -1], dtype=np.complex64)  # Standard pilot pattern

    ofdm_symbols = []
    for sym_idx in range(args.num_ofdm_syms):
        start_bit = sym_idx * N_DATA * 2
        end_bit = start_bit + N_DATA * 2
        data_bits = all_bits[start_bit:end_bit]
        data_syms = qpsk_map(data_bits)
        ofdm_sym = create_ofdm_symbol(data_syms, pilot_values, sym_idx)
        ofdm_symbols.append(ofdm_sym)

    ofdm_payload = np.concatenate(ofdm_symbols)

    # Build complete frame: gap -> tone -> gap -> STF -> LTF -> payload -> gap
    gap_short = np.zeros(500, dtype=np.complex64)
    gap_long = np.zeros(2000, dtype=np.complex64)

    frame = np.concatenate([
        gap_long,
        pilot_tone,
        gap_short,
        stf_with_cp,
        ltf,
        ofdm_payload,
        gap_long
    ])

    # Normalize
    frame = frame / (np.max(np.abs(frame)) + 1e-9) * 0.7

    # Scale for DAC
    tx_scaled = (frame * 2**14).astype(np.complex64)

    print(f"RF Link Test - Step 3: TX (OFDM)")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  TX gain: {args.tx_gain} dB")
    print(f"  Frame structure:")
    print(f"    Pilot tone: {len(pilot_tone)} samples ({args.tone_duration_ms:.1f} ms)")
    print(f"    STF: {len(stf_with_cp)} samples")
    print(f"    LTF: {len(ltf)} samples")
    print(f"    OFDM payload: {len(ofdm_payload)} samples ({args.num_ofdm_syms} symbols)")
    print(f"    Total frame: {len(frame)} samples")

    # Configure SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    sdr.tx_cyclic_buffer = True

    sdr.tx(tx_scaled)

    print("\nTX Running (Ctrl+C to stop)")
    print(f"  Pilot tone at {args.fc/1e6:.3f} MHz + 100 kHz")

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
