#!/usr/bin/env python3
"""
Step5-based Step7 TX (VID7 video packets) - FIXED LENGTH CYCLIC TX

Fixes:
1) Cyclic buffer length must stay constant on Pluto. We enforce a fixed TX waveform length
   by forcing ALL packets to use the same number of OFDM payload symbols.
2) tx_cyclic_buffer is set ONCE at init, never toggled inside loop.
3) Step5 PHY convention is preserved:
   - Frequency-domain uses fftshift/ifftshift mapping.
   - OFDM symbol time domain: ifft(ifftshift(X))*sqrt(N), CP appended.

Packet format (VID7):
  MAGIC("VID7", 4B) | FRAME_ID(2B) | SEQ(2B) | TOTAL(2B) | LEN(2B) | PAYLOAD | CRC32(4B)
  CRC covers MAGIC..PAYLOAD (header+payload).

Usage examples:
- Send one image as 3 frames (repeat same image):
  python3 rf_link_step7_tx_step5phy_fixed.py \
    --uri usb:1.4.5 --fc 2.3e9 --fs 3e6 --tx_gain -20 \
    --image rx_video_frame.jpg --width 320 --height 240 --quality 30 \
    --chunk_size 600 --repeat 1 --tone_duration_ms 10 \
    --dwell_time 0.25 --rounds_per_frame 1 --max_frames 3 \
    --cyclic --print_every_packet

- If you want to reduce required OFDM symbols, increase chunk_size or reduce repeat.
"""

import argparse
import os
import time
import zlib
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


MAGIC = b"VID7"

# =========================
# OFDM (Step5 convention)
# =========================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)              # 48
BITS_PER_OFDM_SYM = N_DATA * 2              # QPSK -> 96 bits/symbol


def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Gray-like mapping consistent with your RX demap."""
    bits = bits.astype(np.uint8)
    assert bits.ndim == 1
    n = len(bits) // 2
    b = bits[: 2*n].reshape(n, 2)
    out = np.zeros(n, dtype=np.complex64)
    # 00 -> + + ; 01 -> - + ; 11 -> - - ; 10 -> + -
    for i in range(n):
        b0, b1 = int(b[i, 0]), int(b[i, 1])
        if b0 == 0 and b1 == 0:
            out[i] = 1 + 1j
        elif b0 == 0 and b1 == 1:
            out[i] = -1 + 1j
        elif b0 == 1 and b1 == 1:
            out[i] = -1 - 1j
        else:
            out[i] = 1 - 1j
    return out / np.sqrt(2)


def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    """Match Step5 STF generation: fftshift mapping and ifft(ifftshift(X))*sqrt(N)."""
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[(k + N_FFT) % N_FFT] = bpsk[i] + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x.astype(np.complex64), num_repeats)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp


def create_ltf(num_symbols: int = 4) -> np.ndarray:
    """Match Step5 LTF generation."""
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N_FFT) % N_FFT] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf


def create_ofdm_symbol(data_symbols: np.ndarray, sym_idx: int) -> np.ndarray:
    """Build one OFDM symbol (time domain with CP). Step5 convention."""
    X = np.zeros(N_FFT, dtype=np.complex64)

    # map data
    nfill = min(len(data_symbols), len(DATA_SUBCARRIERS))
    for i in range(nfill):
        k = int(DATA_SUBCARRIERS[i])
        X[(k + N_FFT) % N_FFT] = data_symbols[i]

    # map pilots (polarity alternation)
    pilot_values = np.array([1, 1, 1, -1], dtype=np.complex64)
    pilot_sign = 1 if (sym_idx % 2 == 0) else -1
    for i, k in enumerate(PILOT_SUBCARRIERS):
        X[(int(k) + N_FFT) % N_FFT] = pilot_sign * pilot_values[i]

    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)


def build_vid7_packet(frame_id: int, seq: int, total: int, payload: bytes) -> bytes:
    plen = len(payload)
    header = (
        MAGIC
        + (frame_id & 0xFFFF).to_bytes(2, "little")
        + (seq & 0xFFFF).to_bytes(2, "little")
        + (total & 0xFFFF).to_bytes(2, "little")
        + (plen & 0xFFFF).to_bytes(2, "little")
    )
    content = header + payload
    crc = zlib.crc32(content) & 0xFFFFFFFF
    pkt = content + crc.to_bytes(4, "little")
    return pkt


def chunk_bytes(data: bytes, chunk_size: int) -> List[Tuple[int, int, bytes]]:
    total = (len(data) + chunk_size - 1) // chunk_size
    out = []
    for i in range(total):
        s = i * chunk_size
        e = min(s + chunk_size, len(data))
        out.append((i, total, data[s:e]))
    return out


def build_tx_waveform_for_packet(
    pkt_bytes: bytes,
    fs: float,
    stf: np.ndarray,
    ltf: np.ndarray,
    fixed_ofdm_syms: int,
    repeat: int,
    tone_duration_ms: float,
    tone_freq_hz: float,
    tone_amp: float,
    gap_short: int,
    gap_long: int,
    tx_scale: float,
    include_preamble: bool = True,
) -> np.ndarray:
    """
    Returns complex64 waveform scaled to Pluto int14 range (2**14).
    LENGTH IS CONSTANT for all packets if fixed_ofdm_syms is constant.
    """
    # bits + repetition
    bits = bits_from_bytes(pkt_bytes)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    # pad bits to fixed_ofdm_syms
    total_bits = fixed_ofdm_syms * BITS_PER_OFDM_SYM
    if len(bits) > total_bits:
        raise RuntimeError(
            f"Packet bits exceed fixed_ofdm_syms capacity: bits={len(bits)} > {total_bits}. "
            f"Increase --fixed_ofdm_syms or reduce --chunk_size/--repeat."
        )
    if len(bits) < total_bits:
        bits = np.pad(bits, (0, total_bits - len(bits)))

    # OFDM payload
    ofdm_syms = []
    for si in range(fixed_ofdm_syms):
        b0 = si * BITS_PER_OFDM_SYM
        b1 = b0 + BITS_PER_OFDM_SYM
        sym_bits = bits[b0:b1]
        data_syms = qpsk_map(sym_bits)
        ofdm_syms.append(create_ofdm_symbol(data_syms, si))
    ofdm_payload = np.concatenate(ofdm_syms).astype(np.complex64)

    # optional tone
    if tone_duration_ms > 0:
        tone_samples = int(round(tone_duration_ms * fs / 1000.0))
        t = (np.arange(tone_samples, dtype=np.float32) / float(fs))
        tone = (tone_amp * np.exp(2j * np.pi * tone_freq_hz * t)).astype(np.complex64)
    else:
        tone = np.zeros(0, dtype=np.complex64)

    gS = np.zeros(int(gap_short), dtype=np.complex64)
    gL = np.zeros(int(gap_long), dtype=np.complex64)

    if include_preamble:
        frame = np.concatenate([gL, tone, gS, stf, ltf, ofdm_payload, gL]).astype(np.complex64)
    else:
        frame = np.concatenate([gL, tone, gS, ofdm_payload, gL]).astype(np.complex64)

    # normalize + scale to int14
    frame = frame / (np.max(np.abs(frame)) + 1e-9) * float(tx_scale)
    tx = (frame * (2**14)).astype(np.complex64)
    return tx


def safe_set_tx_cyclic_once(sdr, enable: bool):
    """Set tx_cyclic_buffer only once safely (before any tx buffer exists)."""
    try:
        sdr.tx_cyclic_buffer = bool(enable)
    except Exception as e:
        raise RuntimeError(f"Failed to set tx_cyclic_buffer={enable}. Set it before first tx(). Error: {e}")


def safe_tx_destroy(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass


@dataclass
class TxConfig:
    uri: str
    fc: float
    fs: float
    tx_gain: float
    tx_scale: float
    stf_repeats: int
    ltf_symbols: int

    repeat: int
    chunk_size: int
    dwell_time: float
    rounds_per_frame: int
    packet_order: str
    max_frames: int

    width: int
    height: int
    quality: int

    tone_duration_ms: float
    tone_freq_hz: float
    tone_amp: float
    gap_short: int
    gap_long: int
    include_preamble: bool

    cyclic: bool
    burst: bool
    print_every_packet: bool

    fixed_ofdm_syms: int
    log_csv: str
    out_dir: str


def main():
    ap = argparse.ArgumentParser("Step5-based Step7 TX (VID7) - FIXED")
    ap.add_argument("--uri", default="usb:1.4.5")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--tx_gain", type=float, default=-20.0)
    ap.add_argument("--tx_scale", type=float, default=0.7)

    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4])

    ap.add_argument("--chunk_size", type=int, default=600)
    ap.add_argument("--dwell_time", type=float, default=0.25)
    ap.add_argument("--rounds_per_frame", type=int, default=1)
    ap.add_argument("--packet_order", type=str, default="sequential", choices=["sequential", "reverse", "random"])
    ap.add_argument("--max_frames", type=int, default=3)

    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--quality", type=int, default=30)

    ap.add_argument("--tone_duration_ms", type=float, default=10.0)
    ap.add_argument("--tone_freq_hz", type=float, default=100e3)
    ap.add_argument("--tone_amp", type=float, default=0.5)
    ap.add_argument("--gap_short", type=int, default=1000)
    ap.add_argument("--gap_long", type=int, default=3000)
    ap.add_argument("--no_preamble", action="store_true")

    ap.add_argument("--cyclic", action="store_true", help="Use cyclic buffer (recommended).")
    ap.add_argument("--burst", action="store_true", help="Non-cyclic burst per packet (experiment).")
    ap.add_argument("--print_every_packet", action="store_true")

    # Source options
    ap.add_argument("--image", type=str, default="", help="Send a single image repeatedly.")
    ap.add_argument("--video", type=str, default="", help="Send frames from a video file.")
    ap.add_argument("--webcam", type=int, default=-1, help="Webcam index, e.g., 0.")

    # Fixed OFDM symbols:
    ap.add_argument("--fixed_ofdm_syms", type=int, default=0,
                    help="Force fixed OFDM symbols per packet. 0=auto from chunk_size+repeat.")

    ap.add_argument("--out_dir", default="rf_link_step7_tx_runs")
    ap.add_argument("--log_csv", default="tx_packets.csv")
    args = ap.parse_args()

    # Validate source
    src_count = int(bool(args.image)) + int(bool(args.video)) + int(args.webcam >= 0)
    if src_count != 1:
        raise SystemExit("Choose exactly one source: --image OR --video OR --webcam <idx>.")

    import cv2
    import adi

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.log_csv)

    # Prepare preamble
    stf = create_schmidl_cox_stf(args.stf_repeats)
    ltf = create_ltf(args.ltf_symbols)

    # Determine fixed OFDM symbols from "max packet bytes"
    # pkt_bytes_max = header(12) + payload(chunk_size) + crc(4)
    pkt_bytes_max = 12 + int(args.chunk_size) + 4
    bits_max = pkt_bytes_max * 8 * int(args.repeat)
    auto_syms = int(np.ceil(bits_max / BITS_PER_OFDM_SYM))
    fixed_syms = int(args.fixed_ofdm_syms) if int(args.fixed_ofdm_syms) > 0 else auto_syms

    if fixed_syms <= 0:
        raise SystemExit("fixed_ofdm_syms computed invalid.")

    # Precompute constant TX length (in samples) by building a dummy max packet waveform
    dummy_payload = bytes([0xAB]) * int(args.chunk_size)
    dummy_pkt = build_vid7_packet(frame_id=0, seq=0, total=1, payload=dummy_payload)
    dummy_tx = build_tx_waveform_for_packet(
        dummy_pkt,
        fs=float(args.fs),
        stf=stf,
        ltf=ltf,
        fixed_ofdm_syms=fixed_syms,
        repeat=int(args.repeat),
        tone_duration_ms=float(args.tone_duration_ms),
        tone_freq_hz=float(args.tone_freq_hz),
        tone_amp=float(args.tone_amp),
        gap_short=int(args.gap_short),
        gap_long=int(args.gap_long),
        tx_scale=float(args.tx_scale),
        include_preamble=(not args.no_preamble),
    )
    TX_LEN = int(len(dummy_tx))

    # Print summary
    print("\n" + "=" * 88)
    print("Step5-based Step7 TX (VID7) - FIXED LENGTH")
    print("=" * 88)
    print(f"uri={args.uri} fc={args.fc/1e6:.1f}MHz fs={args.fs/1e6:.1f}Msps tx_gain={args.tx_gain:.1f}dB tx_scale={args.tx_scale}")
    print(f"stf_repeats={args.stf_repeats} ltf_symbols={args.ltf_symbols} repeat={args.repeat}")
    print(f"chunk_size={args.chunk_size}B  pkt_bytes_max={pkt_bytes_max}  fixed_ofdm_syms={fixed_syms}")
    print(f"TX_LEN(samples)={TX_LEN}  SYMBOL_LEN={SYMBOL_LEN}  payload_samples={fixed_syms*SYMBOL_LEN}")
    print(f"tone: {args.tone_freq_hz/1e3:.1f}kHz dur={args.tone_duration_ms:.1f}ms amp={args.tone_amp}")
    print(f"gaps: short={args.gap_short} long={args.gap_long} preamble={'OFF' if args.no_preamble else 'ON'}")
    mode = "cyclic" if args.cyclic or (not args.burst) else "burst"
    print(f"TX mode: {mode}  dwell={args.dwell_time}s  rounds/frame={args.rounds_per_frame} order={args.packet_order}")
    if args.image:
        print(f"source=image:{args.image}")
    elif args.video:
        print(f"source=video:{args.video}")
    else:
        print(f"source=webcam:{args.webcam}")
    print(f"log_csv: {csv_path}")
    print("=" * 88)

    # Open source
    if args.image:
        img0 = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img0 is None:
            raise RuntimeError(f"Cannot read image: {args.image}")
        # Keep one image, resend multiple frames
        def next_frame():
            return img0.copy(), True
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        def next_frame():
            ok, frame = cap.read()
            return frame, ok
    else:
        cap = cv2.VideoCapture(int(args.webcam))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index={args.webcam}")
        def next_frame():
            ok, frame = cap.read()
            return frame, ok

    # Configure SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)

    # Select cyclic/burst
    use_cyclic = bool(args.cyclic) or (not args.burst)
    safe_tx_destroy(sdr)                 # ensure no old buffer
    safe_set_tx_cyclic_once(sdr, use_cyclic)

    # CSV logger
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "frame_id", "seq", "total", "payload_len", "pkt_bytes",
        "fixed_ofdm_syms", "tx_len", "mode", "dwell_s", "timestamp"
    ])
    csv_w.writeheader()

    frame_id = 0
    pkt_sent = 0
    t0 = time.time()

    try:
        while True:
            if args.max_frames > 0 and frame_id >= args.max_frames:
                print("Reached max_frames. Done.")
                break

            frame, ok = next_frame()
            if not ok or frame is None:
                # for image mode, ok always True
                if args.video:
                    print("End of video file.")
                    break
                # webcam: try again
                continue

            # Resize + JPEG encode
            if frame.shape[1] != int(args.width) or frame.shape[0] != int(args.height):
                frame = cv2.resize(frame, (int(args.width), int(args.height)))
            enc_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.quality)]
            ok2, enc = cv2.imencode(".jpg", frame, enc_params)
            if not ok2:
                print(f"[WARN] JPEG encode failed for frame_id={frame_id}")
                continue
            jpeg_bytes = enc.tobytes()

            # Chunk into packets
            chunks = chunk_bytes(jpeg_bytes, int(args.chunk_size))
            total_pkts = len(chunks)

            # Packet order
            order = list(range(total_pkts))
            if args.packet_order == "reverse":
                order = list(reversed(order))
            elif args.packet_order == "random":
                rng = np.random.default_rng(2026 + frame_id)
                rng.shuffle(order)

            print(f"\nFrame {frame_id}: JPEG={len(jpeg_bytes)}B pkts={total_pkts}")

            for rnd in range(int(args.rounds_per_frame)):
                if int(args.rounds_per_frame) > 1:
                    print(f"  Round {rnd+1}/{args.rounds_per_frame}")

                for idx_in_order in order:
                    seq, total, payload = chunks[idx_in_order]
                    pkt = build_vid7_packet(frame_id=frame_id, seq=seq, total=total, payload=payload)

                    tx = build_tx_waveform_for_packet(
                        pkt,
                        fs=float(args.fs),
                        stf=stf,
                        ltf=ltf,
                        fixed_ofdm_syms=fixed_syms,
                        repeat=int(args.repeat),
                        tone_duration_ms=float(args.tone_duration_ms),
                        tone_freq_hz=float(args.tone_freq_hz),
                        tone_amp=float(args.tone_amp),
                        gap_short=int(args.gap_short),
                        gap_long=int(args.gap_long),
                        tx_scale=float(args.tx_scale),
                        include_preamble=(not args.no_preamble),
                    )

                    # Hard guard: ensure constant length
                    if len(tx) != TX_LEN:
                        raise RuntimeError(f"BUG: tx length changed: {len(tx)} != {TX_LEN}")

                    # Send
                    if use_cyclic:
                        # In cyclic mode, length is constant, so tx() can update content.
                        sdr.tx(tx)
                        time.sleep(float(args.dwell_time))
                    else:
                        # Burst mode: send once, not cyclic.
                        # NOTE: Pluto TX will enqueue and return quickly.
                        sdr.tx(tx)
                        time.sleep(float(args.dwell_time))

                    pkt_sent += 1
                    if args.print_every_packet:
                        print(f"    pkt {seq+1}/{total} payload={len(payload)}B pkt_bytes={len(pkt)} "
                              f"fixed_syms={fixed_syms} tx_len={TX_LEN} dwell={args.dwell_time}s")

                    csv_w.writerow({
                        "frame_id": frame_id,
                        "seq": seq,
                        "total": total,
                        "payload_len": len(payload),
                        "pkt_bytes": len(pkt),
                        "fixed_ofdm_syms": fixed_syms,
                        "tx_len": TX_LEN,
                        "mode": "cyclic" if use_cyclic else "burst",
                        "dwell_s": float(args.dwell_time),
                        "timestamp": time.time(),
                    })

            frame_id += 1
            elapsed = time.time() - t0
            eff_fps = frame_id / elapsed if elapsed > 0 else 0.0
            print(f"  Sent frames={frame_id}, packets={pkt_sent}, elapsed={elapsed:.1f}s, eff_fps={eff_fps:.3f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        csv_f.close()
        try:
            safe_tx_destroy(sdr)
        except Exception:
            pass
        try:
            if args.video or args.webcam >= 0:
                cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()

"""
python3 rf_link_step7_tx_step5phy.py \
  --uri usb:1.4.5 \
  --fc 2.3e9 --fs 3e6 \
  --tx_gain -20 \
  --image rx_video_frame.jpg \
  --width 320 --height 240 --quality 30 \
  --chunk_size 600 \
  --repeat 1 \
  --tone_duration_ms 10 \
  --dwell_time 0.25 \
  --rounds_per_frame 1 \
  --max_frames 3 \
  --cyclic \
  --print_every_packet
"""
