#!/usr/bin/env python3
"""
RF Link Test - Step 5: TX (Robust Sync with Diagnostics)

Improvements over Step 4:
1. Longer preamble for better detection
2. Schmidl-Cox style STF (period N/2) for autocorrelation-based sync
3. Multiple LTF symbols for better channel estimation
4. BER stress test mode with known patterns

Run on the remote TX device (Jetson).
"""

import argparse
import time
import zlib
import numpy as np

# OFDM Parameters
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in PILOT_SUBCARRIERS])
N_DATA = len(DATA_SUBCARRIERS)
BITS_PER_OFDM_SYM = N_DATA * 2
SYMBOL_LEN = N_FFT + N_CP

MAGIC = b"AIS1"


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
        else:
            symbols[i] = 1 - 1j
    return symbols / np.sqrt(2)


def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def create_schmidl_cox_stf(N=64, num_repeats=4):
    """
    Create Schmidl-Cox STF with period N/2.
    Using only even subcarriers creates a signal with period N/2 in time domain.
    This enables robust timing sync via autocorrelation.
    """
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)

    # BPSK on even subcarriers only (creates period N/2)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0])
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i]

    # IFFT
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)

    # Repeat multiple times for robust detection
    stf = np.tile(x, num_repeats).astype(np.complex64)

    # Add CP
    stf_with_cp = np.concatenate([stf[-N_CP:], stf])
    return stf_with_cp.astype(np.complex64), X


def create_ltf(N=64, num_symbols=2):
    """
    Create LTF with known sequence on all used subcarriers.
    Multiple symbols for better channel estimation averaging.
    """
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])

    # Alternating BPSK pattern (deterministic)
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if i % 2 == 0 else -1.0

    # Create time domain with CP
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])

    # Repeat for averaging
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X


def create_ofdm_symbol(data_symbols, pilot_values, sym_idx):
    """Create one OFDM symbol with data and pilots."""
    X = np.zeros(N_FFT, dtype=np.complex64)
    for i, k in enumerate(DATA_SUBCARRIERS):
        if i < len(data_symbols):
            X[(k + N_FFT) % N_FFT] = data_symbols[i]
    pilot_sign = 1 if sym_idx % 2 == 0 else -1
    for i, k in enumerate(PILOT_SUBCARRIERS):
        X[(k + N_FFT) % N_FFT] = pilot_sign * pilot_values[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)


def build_packet(payload: bytes, repeat: int = 1):
    """Build packet with framing."""
    plen = len(payload)
    header = MAGIC + plen.to_bytes(2, "little")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    frame = header + payload + crc.to_bytes(4, "little")
    bits = bits_from_bytes(frame)
    if repeat > 1:
        bits = np.repeat(bits, repeat)
    num_ofdm_syms = int(np.ceil(len(bits) / BITS_PER_OFDM_SYM))
    total_bits = num_ofdm_syms * BITS_PER_OFDM_SYM
    if len(bits) < total_bits:
        bits = np.pad(bits, (0, total_bits - len(bits)))
    return bits, num_ofdm_syms, len(frame)


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - TX (Step 5)")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=-5, help="TX gain (dB)")
    ap.add_argument("--tone_duration_ms", type=float, default=10, help="Pilot tone duration (ms)")
    ap.add_argument("--stf_repeats", type=int, default=6, help="STF symbol repeats")
    ap.add_argument("--ltf_symbols", type=int, default=4, help="LTF symbols for averaging")
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Bit repetition")
    ap.add_argument("--payload", type=str, default="", help="Payload string")
    ap.add_argument("--payload_len", type=int, default=64, help="Random payload length")
    ap.add_argument("--ber_test", action="store_true", help="BER test mode with known pattern")
    ap.add_argument("--ber_num_syms", type=int, default=100, help="Number of OFDM symbols for BER test")
    args = ap.parse_args()

    import adi

    # Determine payload/mode
    if args.ber_test:
        # BER test: known PRBS pattern
        print("BER TEST MODE - Transmitting known PRBS pattern")
        rng = np.random.default_rng(99999)  # Fixed seed for BER test
        num_bits = args.ber_num_syms * BITS_PER_OFDM_SYM
        bits = rng.integers(0, 2, size=num_bits).astype(np.uint8)
        num_ofdm_syms = args.ber_num_syms
        frame_len = num_bits // 8
        payload = b""
    else:
        # Packet mode
        if args.payload:
            payload = args.payload.encode('utf-8')
        else:
            rng = np.random.default_rng(54321)
            payload = rng.integers(0, 256, size=args.payload_len, dtype=np.uint8).tobytes()
        bits, num_ofdm_syms, frame_len = build_packet(payload, args.repeat)

    # Create pilot tone (longer for more reliable CFO estimation)
    tone_samples = int(args.tone_duration_ms * args.fs / 1000)
    t = np.arange(tone_samples) / args.fs
    pilot_tone = np.exp(2j * np.pi * 100e3 * t).astype(np.complex64) * 0.5

    # Create improved preamble
    stf, stf_freq = create_schmidl_cox_stf(N_FFT, num_repeats=args.stf_repeats)
    ltf, ltf_freq = create_ltf(N_FFT, num_symbols=args.ltf_symbols)

    # Create OFDM payload
    pilot_values = np.array([1, 1, 1, -1], dtype=np.complex64)
    ofdm_symbols = []

    for sym_idx in range(num_ofdm_syms):
        start_bit = sym_idx * BITS_PER_OFDM_SYM
        end_bit = start_bit + BITS_PER_OFDM_SYM
        sym_bits = bits[start_bit:end_bit]
        data_syms = qpsk_map(sym_bits)
        ofdm_sym = create_ofdm_symbol(data_syms, pilot_values, sym_idx)
        ofdm_symbols.append(ofdm_sym)

    ofdm_payload = np.concatenate(ofdm_symbols)

    # Build frame with longer gaps for sync
    gap_short = np.zeros(1000, dtype=np.complex64)
    gap_long = np.zeros(3000, dtype=np.complex64)

    frame = np.concatenate([
        gap_long,
        pilot_tone,
        gap_short,
        stf,
        ltf,
        ofdm_payload,
        gap_long
    ])

    # Normalize
    frame = frame / (np.max(np.abs(frame)) + 1e-9) * 0.7
    tx_scaled = (frame * 2**14).astype(np.complex64)

    print(f"\nRF Link Test - Step 5: TX (Robust Sync)")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  TX gain: {args.tx_gain} dB")
    print(f"\n  Preamble structure:")
    print(f"    Pilot tone: {len(pilot_tone)} samples ({args.tone_duration_ms:.1f} ms)")
    print(f"    STF: {len(stf)} samples ({args.stf_repeats} repeats)")
    print(f"    LTF: {len(ltf)} samples ({args.ltf_symbols} symbols)")
    if args.ber_test:
        print(f"\n  BER Test Mode:")
        print(f"    OFDM symbols: {num_ofdm_syms}")
        print(f"    Total bits: {len(bits)}")
    else:
        print(f"\n  Packet info:")
        print(f"    Payload: {len(payload)} bytes")
        print(f"    Frame: {frame_len} bytes")
        print(f"    Repetition: {args.repeat}x")
        print(f"    OFDM symbols: {num_ofdm_syms}")
    print(f"\n  Total signal: {len(frame)} samples")

    # Configure SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    sdr.tx_cyclic_buffer = True

    sdr.tx(tx_scaled)

    print("\nTX Running (Ctrl+C to stop)")

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
