#!/usr/bin/env python3
"""
Step5-based Step7 TX (VID7) - FIXED v2

Key fix:
- pyadi-iio Pluto limitation:
  If tx_cyclic_buffer=True, you can call sdr.tx() only once per created buffer.
  To send a new waveform in cyclic mode, you MUST destroy and recreate the buffer.

This script supports:
  --mode burst            : Non-cyclic, send each packet once (recommended).
  --mode cyclic_per_packet: For each packet: destroy -> set cyclic -> tx -> sleep(dwell) -> destroy.

Also enforces fixed waveform length across packets via fixed_ofdm_syms.

Packet format:
  MAGIC("VID7",4) | FRAME_ID(2) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
  CRC covers MAGIC..PAYLOAD.

"""

import argparse
import os
import time
import zlib
import csv
from typing import List, Tuple

import numpy as np

MAGIC = b"VID7"

# ===== OFDM (Step5 convention) =====
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27)
                             if (k != 0 and k not in set(PILOT_SUBCARRIERS))],
                            dtype=int)
N_DATA = len(DATA_SUBCARRIERS)        # 48
BITS_PER_OFDM_SYM = N_DATA * 2        # 96 (QPSK)


def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def qpsk_map(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(np.uint8)
    n = len(bits) // 2
    b = bits[:2*n].reshape(n, 2)
    out = np.zeros(n, dtype=np.complex64)
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


def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
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
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N_FFT) % N_FFT] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf


def create_ofdm_symbol(data_symbols: np.ndarray, sym_idx: int) -> np.ndarray:
    X = np.zeros(N_FFT, dtype=np.complex64)

    # data mapping
    nfill = min(len(data_symbols), len(DATA_SUBCARRIERS))
    for i in range(nfill):
        k = int(DATA_SUBCARRIERS[i])
        X[(k + N_FFT) % N_FFT] = data_symbols[i]

    # pilot mapping
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
    return content + crc.to_bytes(4, "little")


def chunk_bytes(data: bytes, chunk_size: int) -> List[Tuple[int, int, bytes]]:
    total = (len(data) + chunk_size - 1) // chunk_size
    out = []
    for i in range(total):
        s = i * chunk_size
        e = min(s + chunk_size, len(data))
        out.append((i, total, data[s:e]))
    return out


def build_waveform(
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
    include_preamble: bool,
) -> np.ndarray:
    bits = bits_from_bytes(pkt_bytes)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    total_bits = fixed_ofdm_syms * BITS_PER_OFDM_SYM
    if len(bits) > total_bits:
        raise RuntimeError(
            f"Packet too large for fixed_ofdm_syms: {len(bits)} > {total_bits}. "
            f"Increase fixed_ofdm_syms or reduce chunk/repeat."
        )
    if len(bits) < total_bits:
        bits = np.pad(bits, (0, total_bits - len(bits)))

    # payload OFDM
    ofdm_syms = []
    for si in range(fixed_ofdm_syms):
        b0 = si * BITS_PER_OFDM_SYM
        b1 = b0 + BITS_PER_OFDM_SYM
        data_syms = qpsk_map(bits[b0:b1])
        ofdm_syms.append(create_ofdm_symbol(data_syms, si))
    ofdm_payload = np.concatenate(ofdm_syms).astype(np.complex64)

    # tone
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

    frame = frame / (np.max(np.abs(frame)) + 1e-9) * float(tx_scale)
    return (frame * (2**14)).astype(np.complex64)


def safe_tx_destroy(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser("Step5-based Step7 TX (VID7) - FIXED v2")
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
    ap.add_argument("--packet_order", type=str, default="sequential",
                    choices=["sequential", "reverse", "random"])
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

    ap.add_argument("--fixed_ofdm_syms", type=int, default=0,
                    help="0=auto from chunk_size+repeat; otherwise force.")
    ap.add_argument("--mode", type=str, default="burst",
                    choices=["burst", "cyclic_per_packet"],
                    help="burst: send once per packet (recommended). "
                         "cyclic_per_packet: cyclic + destroy/recreate each packet.")
    ap.add_argument("--print_every_packet", action="store_true")

    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--webcam", type=int, default=-1)

    ap.add_argument("--out_dir", default="rf_link_step7_tx_runs")
    ap.add_argument("--log_csv", default="tx_packets.csv")
    args = ap.parse_args()

    src_count = int(bool(args.image)) + int(bool(args.video)) + int(args.webcam >= 0)
    if src_count != 1:
        raise SystemExit("Choose exactly one: --image OR --video OR --webcam <idx>")

    import cv2
    import adi

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.log_csv)

    stf = create_schmidl_cox_stf(args.stf_repeats)
    ltf = create_ltf(args.ltf_symbols)

    pkt_bytes_max = 12 + int(args.chunk_size) + 4
    bits_max = pkt_bytes_max * 8 * int(args.repeat)
    auto_syms = int(np.ceil(bits_max / BITS_PER_OFDM_SYM))
    fixed_syms = int(args.fixed_ofdm_syms) if int(args.fixed_ofdm_syms) > 0 else auto_syms

    dummy_payload = bytes([0xAB]) * int(args.chunk_size)
    dummy_pkt = build_vid7_packet(0, 0, 1, dummy_payload)
    dummy_tx = build_waveform(
        dummy_pkt, args.fs, stf, ltf, fixed_syms, args.repeat,
        args.tone_duration_ms, args.tone_freq_hz, args.tone_amp,
        args.gap_short, args.gap_long, args.tx_scale, (not args.no_preamble)
    )
    TX_LEN = len(dummy_tx)

    print("\n" + "=" * 88)
    print("Step5-based Step7 TX (VID7) - FIXED v2")
    print("=" * 88)
    print(f"uri={args.uri} fc={args.fc/1e6:.1f}MHz fs={args.fs/1e6:.1f}Msps tx_gain={args.tx_gain}dB tx_scale={args.tx_scale}")
    print(f"chunk_size={args.chunk_size} pkt_bytes_max={pkt_bytes_max} repeat={args.repeat} fixed_ofdm_syms={fixed_syms}")
    print(f"TX_LEN(samples)={TX_LEN}  mode={args.mode}  dwell={args.dwell_time}s")
    print("=" * 88)

    # source
    if args.image:
        img0 = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img0 is None:
            raise RuntimeError(f"Cannot read image: {args.image}")
        def next_frame():
            return img0.copy(), True
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        def next_frame():
            ok, fr = cap.read()
            return fr, ok
    else:
        cap = cv2.VideoCapture(int(args.webcam))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam: {args.webcam}")
        def next_frame():
            ok, fr = cap.read()
            return fr, ok

    # SDR config
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)

    # CSV
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
        while frame_id < int(args.max_frames):
            frame, ok = next_frame()
            if not ok or frame is None:
                if args.video:
                    print("End of video.")
                    break
                continue

            if frame.shape[1] != int(args.width) or frame.shape[0] != int(args.height):
                frame = cv2.resize(frame, (int(args.width), int(args.height)))

            ok2, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(args.quality)])
            if not ok2:
                print("[WARN] JPEG encode failed")
                continue
            jpeg_bytes = enc.tobytes()

            chunks = chunk_bytes(jpeg_bytes, int(args.chunk_size))
            total_pkts = len(chunks)

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

                for j in order:
                    seq, total, payload = chunks[j]
                    pkt = build_vid7_packet(frame_id, seq, total, payload)
                    tx = build_waveform(
                        pkt, args.fs, stf, ltf, fixed_syms, args.repeat,
                        args.tone_duration_ms, args.tone_freq_hz, args.tone_amp,
                        args.gap_short, args.gap_long, args.tx_scale, (not args.no_preamble)
                    )
                    if len(tx) != TX_LEN:
                        raise RuntimeError(f"BUG: tx length changed {len(tx)} != {TX_LEN}")

                    # --- hard reset any previous TX buffer state (important!) ---
                    try:
                        sdr.tx_destroy_buffer()
                    except Exception:
                        pass

                    # Set cyclic mode ONCE and never change it afterwards
                    sdr.tx_cyclic_buffer = (args.mode == "cyclic_per_packet")

                    if args.mode == "burst":
                        # non-cyclic, just push buffers; do NOT touch tx_cyclic_buffer here
                        sdr.tx(tx)
                        time.sleep(args.dwell_time)
                    else:
                        # cyclic_per_packet: must destroy/recreate buffer each packet
                        try:
                            sdr.tx_destroy_buffer()
                        except Exception:
                            pass
                        # tx_cyclic_buffer already True (set once at init)
                        sdr.tx(tx)
                        time.sleep(args.dwell_time)
                        try:
                            sdr.tx_destroy_buffer()
                        except Exception:
                            pass

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
                        "mode": args.mode,
                        "dwell_s": float(args.dwell_time),
                        "timestamp": time.time(),
                    })

            frame_id += 1
            elapsed = time.time() - t0
            print(f"  Sent frames={frame_id}, packets={pkt_sent}, elapsed={elapsed:.1f}s, eff_fps={frame_id/elapsed:.3f}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        csv_f.close()
        safe_tx_destroy(sdr)
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
  --mode burst \
  --print_every_packet
"""
