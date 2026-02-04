#!/usr/bin/env python3
"""
RF Link Test - Step 7: TX (Video Transmission)

Transmits video frames over RF link:
1. Captures frames from video file or webcam
2. JPEG-encodes each frame, splits into packets
3. Transmits packets sequentially via cyclic buffer
4. Supports file mode (finite) and stream mode (continuous)

Packet format (16 bytes overhead):
  MAGIC("VID7", 4B) | FRAME_ID(2B) | SEQ(2B) | TOTAL(2B) | LEN(2B) | PAYLOAD(var) | CRC32(4B)
  CRC covers MAGIC through PAYLOAD.

Run on the remote TX device (Jetson).
"""

import argparse
import os
import sys
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

MAGIC = b"VID7"


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
# Video Packet Functions
# ============================================================================

def jpeg_encode_frame(frame, width, height, quality):
    """Encode a BGR frame as JPEG, return raw bytes."""
    import cv2
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode('.jpg', frame, encode_params)
    if not success:
        return None
    return encoded.tobytes()


def chunk_data(data: bytes, chunk_size: int):
    """Split data into chunks, returns list of (seq, total, chunk_bytes)."""
    total = (len(data) + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(total):
        start = i * chunk_size
        end = min(start + chunk_size, len(data))
        chunks.append((i, total, data[start:end]))
    return chunks


def build_video_packet(frame_id, seq, total, payload, repeat=1):
    """
    Build video packet bits.
    Format: MAGIC(4) | FRAME_ID(2) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    CRC covers MAGIC through PAYLOAD.
    """
    plen = len(payload)
    header = (MAGIC
              + (frame_id & 0xFFFF).to_bytes(2, "little")
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
    tone_samples = int(tone_duration_ms * fs / 1000)
    t = np.arange(tone_samples) / fs
    pilot_tone = np.exp(2j * np.pi * 100e3 * t).astype(np.complex64) * 0.5

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

    frame = frame / (np.max(np.abs(frame)) + 1e-9) * 0.7
    return (frame * 2**14).astype(np.complex64)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="RF Link Test - Step 7: Video TX")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=-5, help="TX gain (dB)")
    ap.add_argument("--tone_duration_ms", type=float, default=10, help="Pilot tone (ms)")
    ap.add_argument("--stf_repeats", type=int, default=6, help="STF repeats")
    ap.add_argument("--ltf_symbols", type=int, default=4, help="LTF symbols")
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Bit repetition")
    ap.add_argument("--video", type=str, required=True,
                    help="Video file path, or '0' for webcam")
    ap.add_argument("--mode", type=str, default="file", choices=["file", "stream"],
                    help="file: send all frames then stop; stream: continuous webcam")
    ap.add_argument("--width", type=int, default=320, help="Frame width")
    ap.add_argument("--height", type=int, default=240, help="Frame height")
    ap.add_argument("--quality", type=int, default=30, help="JPEG quality (1-100)")
    ap.add_argument("--chunk_size", type=int, default=1000, help="Payload per packet (bytes)")
    ap.add_argument("--dwell_time", type=float, default=0.5, help="Seconds per packet")
    ap.add_argument("--max_frames", type=int, default=30, help="Max frames (file mode)")
    ap.add_argument("--frame_step", type=int, default=1, help="Send every Nth source frame")
    ap.add_argument("--rounds_per_frame", type=int, default=1,
                    help="How many times to cycle through a frame's packets")
    args = ap.parse_args()

    import cv2

    # Open video source
    if args.video == "0":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        source_desc = "Webcam"
    else:
        if not os.path.isfile(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {args.video}")
            return
        source_desc = args.video

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Create preamble
    stf, _ = create_schmidl_cox_stf(N_FFT, num_repeats=args.stf_repeats)
    ltf, _ = create_ltf(N_FFT, num_symbols=args.ltf_symbols)

    print(f"\nRF Link Test - Step 7: Video TX")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc / 1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs / 1e6:.3f} Msps")
    print(f"  TX gain: {args.tx_gain} dB")
    print(f"\n  Video source: {source_desc}")
    print(f"  Source FPS: {src_fps:.1f}, Total frames: {src_total}")
    print(f"  Mode: {args.mode}")
    print(f"  Output resolution: {args.width}x{args.height}")
    print(f"  JPEG quality: {args.quality}")
    print(f"  Chunk size: {args.chunk_size} bytes")
    print(f"  Dwell time: {args.dwell_time}s per packet")
    print(f"  Frame step: every {args.frame_step} frame(s)")
    if args.mode == "file":
        print(f"  Max frames: {args.max_frames}")
    print(f"  Rounds per frame: {args.rounds_per_frame}")

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

    frame_id = 0
    source_frame_idx = 0
    total_packets_sent = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.mode == "file":
                    print(f"\n  End of video file after {frame_id} frames.")
                    break
                else:
                    # Stream mode: try reopening webcam
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    continue

            source_frame_idx += 1

            # Skip frames per frame_step
            if (source_frame_idx - 1) % args.frame_step != 0:
                continue

            # JPEG encode
            jpeg_bytes = jpeg_encode_frame(frame, args.width, args.height, args.quality)
            if jpeg_bytes is None:
                print(f"  Warning: JPEG encode failed for frame {frame_id}")
                continue

            # Split into packets
            chunks = chunk_data(jpeg_bytes, args.chunk_size)
            total_pkts = len(chunks)

            print(f"\n  Frame {frame_id}: {len(jpeg_bytes)} bytes JPEG, "
                  f"{total_pkts} packets")

            # Transmit all packets for this frame
            for rnd in range(args.rounds_per_frame):
                if args.rounds_per_frame > 1:
                    print(f"    Round {rnd + 1}/{args.rounds_per_frame}")

                for seq, total, chunk in chunks:
                    bits, num_ofdm_syms, frame_len = build_video_packet(
                        frame_id, seq, total, chunk, args.repeat
                    )
                    waveform = build_tx_waveform(
                        bits, num_ofdm_syms, args.fs, args.tone_duration_ms, stf, ltf
                    )

                    try:
                        sdr.tx_destroy_buffer()
                    except Exception:
                        pass
                    sdr.tx(waveform)

                    total_packets_sent += 1
                    print(f"\r    Pkt {seq + 1}/{total} "
                          f"({len(chunk)}B, {num_ofdm_syms} syms) "
                          f"dwell={args.dwell_time}s",
                          end="", flush=True)
                    time.sleep(args.dwell_time)

                print()

            frame_id += 1
            elapsed = time.time() - t_start
            effective_fps = frame_id / elapsed if elapsed > 0 else 0
            print(f"    Elapsed: {elapsed:.1f}s, Effective: {effective_fps:.3f} fps, "
                  f"Total pkts: {total_packets_sent}")

            if args.mode == "file" and frame_id >= args.max_frames:
                print(f"\n  Reached max_frames ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        print(f"\n\nStopping TX after {frame_id} frames...")
    finally:
        cap.release()
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

    elapsed = time.time() - t_start
    print(f"\nSummary:")
    print(f"  Frames sent: {frame_id}")
    print(f"  Packets sent: {total_packets_sent}")
    print(f"  Elapsed: {elapsed:.1f}s")
    if frame_id > 0:
        print(f"  Effective FPS: {frame_id / elapsed:.3f}")


if __name__ == "__main__":
    main()
