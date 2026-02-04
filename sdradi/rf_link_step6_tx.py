#!/usr/bin/env python3
"""
RF Link Test - Step 6: TX (Image Transmission)

Transmits an image file split into multiple packets:
1. Reads image file, splits into fixed-size chunks
2. Each chunk gets a packet header with sequence number, total count, CRC
3. Packets are transmitted one at a time via cyclic buffer
4. TX cycles through all packets in round-robin for reliability

Packet format (14 bytes overhead):
  MAGIC("IMG6", 4B) | SEQ(2B) | TOTAL(2B) | LEN(2B) | PAYLOAD(var) | CRC32(4B)
  CRC covers MAGIC through PAYLOAD.

Run on the remote TX device (Jetson).
"""

import argparse
import os
import time
import zlib
import numpy as np

# ============================================================================
# OFDM Parameters (must match RX)
# ============================================================================
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in PILOT_SUBCARRIERS])
N_DATA = len(DATA_SUBCARRIERS)  # 48
BITS_PER_OFDM_SYM = N_DATA * 2  # 96
SYMBOL_LEN = N_FFT + N_CP       # 80

MAGIC = b"IMG6"


# ============================================================================
# PHY Functions (from Step 5)
# ============================================================================

def qpsk_map(bits):
    """Map bits to QPSK symbols (Gray coded)."""
    symbols = np.zeros(len(bits) // 2, dtype=np.complex64)
    for i in range(len(bits) // 2):
        b0, b1 = bits[2 * i], bits[2 * i + 1]
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
    """Create Schmidl-Cox STF with period N/2."""
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0])
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    stf = np.tile(x, num_repeats).astype(np.complex64)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf])
    return stf_with_cp.astype(np.complex64), X


def create_ltf(N=64, num_symbols=2):
    """Create LTF with known sequence on all used subcarriers."""
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if i % 2 == 0 else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])
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


# ============================================================================
# Image Packet Functions
# ============================================================================

def chunk_file(data: bytes, chunk_size: int):
    """Split data into chunks, returns list of (seq, total, chunk_bytes)."""
    total = (len(data) + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(total):
        start = i * chunk_size
        end = min(start + chunk_size, len(data))
        chunks.append((i, total, data[start:end]))
    return chunks


def build_image_packet(seq, total, payload, repeat=1):
    """
    Build image packet bits.
    Format: MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    CRC covers MAGIC through PAYLOAD.
    """
    plen = len(payload)
    header = (MAGIC
              + seq.to_bytes(2, "little")
              + total.to_bytes(2, "little")
              + plen.to_bytes(2, "little"))
    content = header + payload
    crc = zlib.crc32(content) & 0xFFFFFFFF
    frame_bytes = content + crc.to_bytes(4, "little")

    bits = bits_from_bytes(frame_bytes)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    num_ofdm_syms = int(np.ceil(len(bits) / BITS_PER_OFDM_SYM))
    total_bits = num_ofdm_syms * BITS_PER_OFDM_SYM
    if len(bits) < total_bits:
        bits = np.pad(bits, (0, total_bits - len(bits)))

    return bits, num_ofdm_syms, len(frame_bytes)


def build_tx_waveform(bits, num_ofdm_syms, fs, tone_duration_ms, stf, ltf):
    """Build complete TX waveform for one packet."""
    # Pilot tone
    tone_samples = int(tone_duration_ms * fs / 1000)
    t = np.arange(tone_samples) / fs
    pilot_tone = np.exp(2j * np.pi * 100e3 * t).astype(np.complex64) * 0.5

    # OFDM modulate
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

    # Assemble frame
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

    # Normalize and scale
    frame = frame / (np.max(np.abs(frame)) + 1e-9) * 0.7
    return (frame * 2**14).astype(np.complex64)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="RF Link Test - Step 6: Image TX")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=-5, help="TX gain (dB)")
    ap.add_argument("--tone_duration_ms", type=float, default=10, help="Pilot tone duration (ms)")
    ap.add_argument("--stf_repeats", type=int, default=6, help="STF symbol repeats")
    ap.add_argument("--ltf_symbols", type=int, default=4, help="LTF symbols for averaging")
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Bit repetition")
    ap.add_argument("--image", type=str, required=True, help="Path to image file")
    ap.add_argument("--chunk_size", type=int, default=1000, help="Payload bytes per packet")
    ap.add_argument("--dwell_time", type=float, default=2.0, help="Seconds to dwell on each packet")
    ap.add_argument("--rounds", type=int, default=0, help="Round-robin cycles (0=infinite)")
    args = ap.parse_args()

    # Read image file
    if not os.path.isfile(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    with open(args.image, "rb") as f:
        image_data = f.read()

    file_size = len(image_data)
    chunks = chunk_file(image_data, args.chunk_size)
    total_packets = len(chunks)

    print(f"\nRF Link Test - Step 6: Image TX")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc / 1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs / 1e6:.3f} Msps")
    print(f"  TX gain: {args.tx_gain} dB")
    print(f"\n  Image: {args.image}")
    print(f"  File size: {file_size} bytes")
    print(f"  Chunk size: {args.chunk_size} bytes")
    print(f"  Total packets: {total_packets}")
    print(f"  Dwell time: {args.dwell_time}s per packet")
    print(f"  Repetition: {args.repeat}x")
    print(f"  Rounds: {'infinite' if args.rounds == 0 else args.rounds}")

    # Create preamble (shared across all packets)
    stf, _ = create_schmidl_cox_stf(N_FFT, num_repeats=args.stf_repeats)
    ltf, _ = create_ltf(N_FFT, num_symbols=args.ltf_symbols)

    # Pre-build all waveforms
    print(f"\n  Building {total_packets} waveforms...")
    waveforms = []
    for seq, total, chunk in chunks:
        bits, num_ofdm_syms, frame_len = build_image_packet(seq, total, chunk, args.repeat)
        waveform = build_tx_waveform(bits, num_ofdm_syms, args.fs, args.tone_duration_ms, stf, ltf)
        waveforms.append(waveform)
        print(f"    Packet {seq}/{total - 1}: {len(chunk)} bytes payload, "
              f"{num_ofdm_syms} OFDM symbols, {len(waveform)} samples")

    # Configure SDR
    import adi
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    sdr.tx_cyclic_buffer = True

    print(f"\nTX Starting (Ctrl+C to stop)")
    print("-" * 60)

    round_num = 0
    try:
        while True:
            round_num += 1
            if args.rounds > 0 and round_num > args.rounds:
                print(f"\nCompleted {args.rounds} rounds.")
                break

            for seq, total, chunk in chunks:
                # Load new waveform
                try:
                    sdr.tx_destroy_buffer()
                except Exception:
                    pass
                sdr.tx(waveforms[seq])

                print(f"\r  Round {round_num}: Packet {seq + 1}/{total} "
                      f"({len(chunk)} bytes) dwelling {args.dwell_time}s...",
                      end="", flush=True)
                time.sleep(args.dwell_time)

            print(f"\n  Round {round_num} complete.")

    except KeyboardInterrupt:
        print(f"\n\nStopping TX after {round_num} rounds...")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass


if __name__ == "__main__":
    main()
