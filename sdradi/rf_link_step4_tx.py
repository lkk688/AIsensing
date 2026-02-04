#!/usr/bin/env python3
"""
RF Link Test - Step 4: TX (Full Packet with CRC)

Transmits complete packets with:
1. Magic header for packet detection
2. Length field
3. Payload data
4. CRC32 for error detection
5. Optional bit repetition for reliability

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
N_DATA = len(DATA_SUBCARRIERS)  # 48 data subcarriers
BITS_PER_OFDM_SYM = N_DATA * 2  # QPSK = 2 bits per symbol

MAGIC = b"AIS1"  # 4-byte magic header


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
    """Convert bytes to bit array."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def create_stf(N=64):
    """Create Short Training Field."""
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0])
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    stf = np.tile(x, 2).astype(np.complex64)
    return stf


def create_ltf(N=64):
    """Create Long Training Field."""
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if i % 2 == 0 else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])
    ltf = np.tile(ltf_sym, 2).astype(np.complex64)
    return ltf


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
    """
    Build complete packet with framing.

    Frame format:
    - MAGIC (4 bytes): "AIS1"
    - LENGTH (2 bytes): payload length, little endian
    - PAYLOAD (N bytes): user data
    - CRC32 (4 bytes): CRC of payload, little endian
    """
    # Build frame
    plen = len(payload)
    header = MAGIC + plen.to_bytes(2, "little")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    frame = header + payload + crc.to_bytes(4, "little")

    # Convert to bits
    bits = bits_from_bytes(frame)

    # Optional repetition for reliability
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    # Calculate number of OFDM symbols needed
    num_ofdm_syms = int(np.ceil(len(bits) / BITS_PER_OFDM_SYM))

    # Pad bits to fill last symbol
    total_bits = num_ofdm_syms * BITS_PER_OFDM_SYM
    if len(bits) < total_bits:
        bits = np.pad(bits, (0, total_bits - len(bits)))

    return bits, num_ofdm_syms, len(frame)


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - TX (Step 4)")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=-5, help="TX gain (dB)")
    ap.add_argument("--tone_duration_ms", type=float, default=5, help="Pilot tone duration (ms)")
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Bit repetition")
    ap.add_argument("--payload", type=str, default="", help="Payload string (or use --payload_len)")
    ap.add_argument("--payload_len", type=int, default=64, help="Random payload length")
    ap.add_argument("--infile", type=str, default="", help="File to transmit")
    args = ap.parse_args()

    import adi

    # Determine payload
    if args.infile:
        with open(args.infile, "rb") as f:
            payload = f.read()
        print(f"Using file payload: {len(payload)} bytes from {args.infile}")
    elif args.payload:
        payload = args.payload.encode('utf-8')
        print(f"Using string payload: {len(payload)} bytes")
    else:
        rng = np.random.default_rng(54321)
        payload = rng.integers(0, 256, size=args.payload_len, dtype=np.uint8).tobytes()
        print(f"Using random payload: {len(payload)} bytes")

    # Build packet
    bits, num_ofdm_syms, frame_len = build_packet(payload, args.repeat)

    # Create pilot tone
    tone_samples = int(args.tone_duration_ms * args.fs / 1000)
    t = np.arange(tone_samples) / args.fs
    pilot_tone = np.exp(2j * np.pi * 100e3 * t).astype(np.complex64) * 0.5

    # Create preamble
    stf = create_stf(N_FFT)
    ltf = create_ltf(N_FFT)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf])

    # Create OFDM payload symbols
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

    # Build frame
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
    tx_scaled = (frame * 2**14).astype(np.complex64)

    # Calculate CRC for display
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    print(f"\nRF Link Test - Step 4: TX (Full Packet)")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  TX gain: {args.tx_gain} dB")
    print(f"\n  Packet info:")
    print(f"    Payload: {len(payload)} bytes")
    print(f"    Frame: {frame_len} bytes (with header+CRC)")
    print(f"    Repetition: {args.repeat}x")
    print(f"    OFDM symbols: {num_ofdm_syms}")
    print(f"    Total bits: {len(bits)}")
    print(f"    CRC32: 0x{crc:08X}")
    print(f"\n  Signal samples: {len(frame)}")

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
