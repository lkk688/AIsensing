#!/usr/bin/env python3
"""
Step5-based Step7 TX (VID7 video packets) - PlutoSDR

Key points:
- Uses the SAME "Step5 FFT convention":
  * TX builds frequency bins X using idx=(k+N)%N, then time via ifft(ifftshift(X))
  * RX expects fftshift(fft()) and uses idx=(k+N)%N for subcarrier k
- Frame format (per packet):
  MAGIC("VID7",4) | FRAME_ID(2) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
  CRC covers MAGIC..PAYLOAD (i.e., header+payload)

TX supports:
- --image <path>   (single image repeatedly)
- --video <path>   (file frames)
- --payload <str>  (single small packet)
- --random_bytes N (single random payload)
"""

import argparse
import os
import time
import zlib
import numpy as np

MAGIC = b"VID7"

# =========================
# OFDM (802.11-like)
# =========================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array(
    [k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))],
    dtype=int
)
N_DATA = len(DATA_SUBCARRIERS)  # 48
BITS_PER_OFDM_SYM = N_DATA * 2  # QPSK

def bits_from_bytes(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)

def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Gray-ish mapping consistent with your Step4/5 mapping."""
    bits = bits.astype(np.uint8)
    if len(bits) % 2 == 1:
        bits = np.pad(bits, (0, 1))
    out = np.zeros(len(bits)//2, dtype=np.complex64)
    for i in range(len(out)):
        b0, b1 = bits[2*i], bits[2*i+1]
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
    """
    Step5-style STF:
    - Fill even subcarriers with BPSK in X[(k+N)%N]
    - Time via ifft(ifftshift(X))*sqrt(N)
    - Repeat num_repeats, and prepend a CP once for the whole block
    """
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[(k + N_FFT) % N_FFT] = bpsk[i] + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x, num_repeats).astype(np.complex64)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp

def create_ltf(num_symbols: int = 4) -> np.ndarray:
    """
    Step5-style LTF:
    - Deterministic BPSK on all used subcarriers in X[(k+N)%N]
    - Time via ifft(ifftshift(X))*sqrt(N)
    - Each symbol has its own CP, repeated num_symbols
    """
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N_FFT) % N_FFT] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf

def create_ofdm_symbol(data_symbols: np.ndarray, pilot_values: np.ndarray, sym_idx: int) -> np.ndarray:
    X = np.zeros(N_FFT, dtype=np.complex64)
    # map data
    for i, k in enumerate(DATA_SUBCARRIERS):
        if i < len(data_symbols):
            X[(k + N_FFT) % N_FFT] = data_symbols[i]
    # pilots with alternating sign
    pilot_sign = 1 if (sym_idx % 2 == 0) else -1
    for i, k in enumerate(PILOT_SUBCARRIERS):
        X[(k + N_FFT) % N_FFT] = pilot_sign * pilot_values[i]
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

def chunk_bytes(data: bytes, chunk_size: int):
    total = (len(data) + chunk_size - 1) // chunk_size
    out = []
    for i in range(total):
        out.append(data[i*chunk_size : min((i+1)*chunk_size, len(data))])
    return out

def build_packet_waveform(pkt_bytes: bytes,
                          repeat: int,
                          stf: np.ndarray,
                          ltf: np.ndarray,
                          tone_freq_hz: float,
                          tone_duration_ms: float,
                          tone_amp: float,
                          fs: float,
                          gap_short: int,
                          gap_long: int) -> tuple[np.ndarray, int]:
    bits = bits_from_bytes(pkt_bytes)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    num_syms = int(np.ceil(len(bits) / BITS_PER_OFDM_SYM))
    need = num_syms * BITS_PER_OFDM_SYM
    if len(bits) < need:
        bits = np.pad(bits, (0, need - len(bits)))

    # payload OFDM
    pilot_values = np.array([1, 1, 1, -1], dtype=np.complex64)
    ofdm_syms = []
    for si in range(num_syms):
        b = bits[si*BITS_PER_OFDM_SYM:(si+1)*BITS_PER_OFDM_SYM]
        data_syms = qpsk_map(b)
        ofdm_syms.append(create_ofdm_symbol(data_syms, pilot_values, si))
    ofdm_payload = np.concatenate(ofdm_syms).astype(np.complex64)

    # optional tone
    if tone_duration_ms > 0:
        n_tone = int(tone_duration_ms * fs / 1000.0)
        t = np.arange(n_tone) / fs
        tone = (tone_amp * np.exp(2j*np.pi*tone_freq_hz*t)).astype(np.complex64)
    else:
        tone = np.zeros(0, dtype=np.complex64)

    gS = np.zeros(int(gap_short), dtype=np.complex64)
    gL = np.zeros(int(gap_long), dtype=np.complex64)

    frame = np.concatenate([gL, tone, gS, stf, ltf, ofdm_payload, gL]).astype(np.complex64)
    return frame, num_syms

def main():
    ap = argparse.ArgumentParser("Step5-based Step7 TX (VID7)")
    ap.add_argument("--uri", default="usb:1.4.5")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--tx_gain", type=float, default=-20)
    ap.add_argument("--tx_scale", type=float, default=0.7)

    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--tone_freq_hz", type=float, default=100e3)
    ap.add_argument("--tone_duration_ms", type=float, default=10.0)
    ap.add_argument("--tone_amp", type=float, default=0.5)

    ap.add_argument("--gap_short", type=int, default=1000)
    ap.add_argument("--gap_long", type=int, default=3000)

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--chunk_size", type=int, default=600)
    ap.add_argument("--dwell_time", type=float, default=0.25)
    ap.add_argument("--rounds_per_frame", type=int, default=1)
    ap.add_argument("--packet_order", choices=["sequential","reverse","random"], default="sequential")
    ap.add_argument("--destroy_each_packet", action="store_true")
    ap.add_argument("--print_every_packet", action="store_true")

    # sources
    ap.add_argument("--payload", type=str, default="")
    ap.add_argument("--random_bytes", type=int, default=0)
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--mode", choices=["file","stream"], default="file")

    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--quality", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=3)
    ap.add_argument("--frame_step", type=int, default=1)

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import adi

    # preamble
    stf = create_schmidl_cox_stf(args.stf_repeats)
    ltf = create_ltf(args.ltf_symbols)

    # choose mode
    use_cv = bool(args.image or args.video)
    if use_cv:
        import cv2

    # open source
    cap = None
    img_static = None

    if args.image:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(args.image)
        img_static = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img_static is None:
            raise RuntimeError(f"Cannot read image: {args.image}")

    elif args.video:
        if args.video == "0":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open webcam (video=0).")
        else:
            if not os.path.isfile(args.video):
                raise FileNotFoundError(args.video)
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {args.video}")

    # configure SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)

    print("\n" + "="*80)
    print("Step5-based Step7 TX (VID7)")
    print("="*80)
    print(f"uri={args.uri} fc={args.fc/1e6:.1f}MHz fs={args.fs/1e6:.1f}Msps tx_gain={args.tx_gain}dB tx_scale={args.tx_scale}")
    print(f"stf_repeats={args.stf_repeats} ltf_symbols={args.ltf_symbols} repeat={args.repeat}")
    print(f"tone={args.tone_freq_hz/1e3:.1f}kHz dur={args.tone_duration_ms}ms amp={args.tone_amp}")
    print(f"chunk={args.chunk_size}B dwell={args.dwell_time}s rounds/frame={args.rounds_per_frame} order={args.packet_order}")
    if args.image:
        print(f"source=image: {args.image}")
    elif args.video:
        print(f"source=video: {args.video} mode={args.mode}")
    elif args.payload:
        print(f"source=payload string len={len(args.payload.encode('utf-8'))}")
    elif args.random_bytes > 0:
        print(f"source=random_bytes={args.random_bytes}")
    else:
        print("source=EMPTY (will send small test payload)")
    print("="*80)

    def jpeg_encode(frame_bgr):
        import cv2
        if frame_bgr.shape[1] != args.width or frame_bgr.shape[0] != args.height:
            frame_bgr = cv2.resize(frame_bgr, (args.width, args.height))
        ok, enc = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(args.quality)])
        if not ok:
            return None
        return enc.tobytes()

    rng = np.random.default_rng(1234)
    frame_id = 0
    total_pkts_sent = 0
    t0 = time.time()

    try:
        while True:
            # build "frame bytes" to transmit (either JPEG or raw payload)
            if img_static is not None:
                frame_bytes = jpeg_encode(img_static)
                if frame_bytes is None:
                    print("[WARN] jpeg encode failed")
                    continue
            elif cap is not None:
                ret, fr = cap.read()
                if not ret:
                    if args.mode == "file":
                        print("End of stream.")
                        break
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    continue
                frame_bytes = jpeg_encode(fr)
                if frame_bytes is None:
                    print("[WARN] jpeg encode failed")
                    continue
            else:
                # single-packet modes
                if args.payload:
                    frame_bytes = args.payload.encode("utf-8")
                elif args.random_bytes > 0:
                    frame_bytes = rng.integers(0, 256, size=int(args.random_bytes), dtype=np.uint8).tobytes()
                else:
                    frame_bytes = b"HELLO_VID7"

            # chunk into packets (for non-JPEG payload, still send as a "frame" with total>=1)
            chunks = chunk_bytes(frame_bytes, int(args.chunk_size))
            total = len(chunks)

            # packet order
            idxs = list(range(total))
            if args.packet_order == "reverse":
                idxs = list(reversed(idxs))
            elif args.packet_order == "random":
                rng.shuffle(idxs)

            print(f"\nFrame {frame_id}: bytes={len(frame_bytes)} pkts={total}")

            for rnd in range(args.rounds_per_frame):
                if args.rounds_per_frame > 1:
                    print(f"  Round {rnd+1}/{args.rounds_per_frame}")

                for ii, seq in enumerate(idxs):
                    payload = chunks[seq]
                    pkt_bytes = build_vid7_packet(frame_id, seq, total, payload)
                    wf, nsyms = build_packet_waveform(
                        pkt_bytes=pkt_bytes,
                        repeat=args.repeat,
                        stf=stf, ltf=ltf,
                        tone_freq_hz=args.tone_freq_hz,
                        tone_duration_ms=args.tone_duration_ms,
                        tone_amp=args.tone_amp,
                        fs=args.fs,
                        gap_short=args.gap_short,
                        gap_long=args.gap_long
                    )

                    # scale and send
                    wf = wf / (np.max(np.abs(wf)) + 1e-9) * float(args.tx_scale)
                    tx = (wf * (2**14)).astype(np.complex64)

                    # Pluto cyclic behavior:
                    # We use cyclic buffer per packet for robustness, and optionally destroy each packet.
                    sdr.tx_cyclic_buffer = True
                    if args.destroy_each_packet:
                        try:
                            sdr.tx_destroy_buffer()
                        except Exception:
                            pass
                    sdr.tx(tx)

                    total_pkts_sent += 1
                    if args.print_every_packet or args.verbose:
                        print(f"    pkt {ii+1}/{len(idxs)} seq={seq} payload={len(payload)}B pkt_bytes={len(pkt_bytes)} "
                              f"ofdm_syms={nsyms} nsamp={len(tx)} dwell={args.dwell_time}s")
                    else:
                        print(f"\r    pkt {ii+1}/{len(idxs)} seq={seq} payload={len(payload)}B ofdm_syms={nsyms}",
                              end="", flush=True)

                    time.sleep(float(args.dwell_time))

                if not (args.print_every_packet or args.verbose):
                    print()

            frame_id += 1
            elapsed = time.time() - t0
            eff_fps = frame_id / elapsed if elapsed > 0 else 0.0
            print(f"  Sent frames={frame_id}, packets={total_pkts_sent}, elapsed={elapsed:.1f}s, eff_fps={eff_fps:.3f}")

            if img_static is not None:
                if frame_id >= args.max_frames:
                    print("Reached max_frames. Done.")
                    break
            elif cap is not None:
                if args.mode == "file" and frame_id >= args.max_frames:
                    print("Reached max_frames. Done.")
                    break
            else:
                # single-payload modes: send once
                break

    except KeyboardInterrupt:
        print("\nStopping TX...")

    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        if cap is not None:
            cap.release()

if __name__ == "__main__":
    main()

"""
TX (Jetson) — start with chunk_size 600, tone ON (10 ms), repeat 1
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
  --print_every_packet
"""
