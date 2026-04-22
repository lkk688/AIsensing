#!/usr/bin/env python3
"""
rf_link_step7_tx_fixed.py  (Step-7 TX, FIXED + experiment switches)

A practical PHY+APP TX script for PlutoSDR video transmission.

Key fixes vs earlier TX:
- Consistent FFT bin mapping with the RX pipeline:
  * NO fftshift/ifftshift in OFDM/STF/LTF generation.
  * Subcarrier mapping uses k in [-26..26] with DC excluded, mapped by (k+N)%N.
- Pilot tone is optional (disabled by default). When enabled, it is placed before STF.
- Cyclic TX handling is safe:
  * You can keep a cyclic buffer per packet, or update it per packet by destroying buffer.

Experiment switches:
- Cable / antenna experiments via preset + gain/scale knobs
- Optional pilot tone, optional gaps
- Repetition coding (bit repeat)
- Packet dwell time, rounds per frame, packet order
- Optional padding/truncation controls for frame payload mapping
- Optional “TX-only” (no camera/video) test pattern

Packet format (must match RX):
  MAGIC("VID7",4) | FRAME_ID(2) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)

Author: (your lab)
"""

import argparse
import os
import time
import zlib
import numpy as np

MAGIC = b"VID7"

# ----------------------------
# OFDM parameters (MUST match RX)
# ----------------------------
N_FFT = 64
N_CP  = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array([k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)          # 48
BITS_PER_SYM = N_DATA * 2               # QPSK: 2 bits/symbol => 96 bits/OFDM sym

# Default pilot values on pilot subcarriers (BPSK)
PILOT_VALUES = np.array([1, 1, 1, -1], dtype=np.complex64)


# ----------------------------
# Helpers: bits/bytes/QPSK
# ----------------------------
def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)

def bytes_from_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()

def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK:
        00 -> +1 +j
        01 -> -1 +j
        11 -> -1 -j
        10 -> +1 -j
    """
    bits = bits.astype(np.uint8)
    if len(bits) % 2 != 0:
        bits = np.pad(bits, (0, 1))
    b0 = bits[0::2]
    b1 = bits[1::2]
    syms = np.empty(len(b0), dtype=np.complex64)

    # Map
    # 00
    m00 = (b0 == 0) & (b1 == 0)
    syms[m00] = (1 + 1j)
    # 01
    m01 = (b0 == 0) & (b1 == 1)
    syms[m01] = (-1 + 1j)
    # 11
    m11 = (b0 == 1) & (b1 == 1)
    syms[m11] = (-1 - 1j)
    # 10
    m10 = (b0 == 1) & (b1 == 0)
    syms[m10] = (1 - 1j)

    return (syms / np.sqrt(2)).astype(np.complex64)


# ----------------------------
# PHY: STF/LTF/OFDM builders (NO fftshift/ifftshift)
# ----------------------------
def k2i(k: int, N: int) -> int:
    return (k + N) % N

def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    """
    Schmidl-Cox STF: half-period repetition property
    We fill even subcarriers with BPSK, DC excluded.
    IMPORTANT: no ifftshift; RX must use fft with consistent indexing.
    """
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)

    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for kk, vv in zip(even_subs, bpsk):
        X[k2i(int(kk), N_FFT)] = vv + 0j

    x = np.fft.ifft(X) * np.sqrt(N_FFT)     # <-- FIX: no ifftshift
    stf = np.tile(x, num_repeats).astype(np.complex64)
    stf_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_cp

def create_ltf(num_symbols: int = 4) -> np.ndarray:
    """
    LTF with known BPSK on all used subcarriers (DC excluded).
    IMPORTANT: no ifftshift.
    """
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, kk in enumerate(used):
        X[k2i(int(kk), N_FFT)] = (1.0 if (i % 2 == 0) else -1.0) + 0j

    x = np.fft.ifft(X) * np.sqrt(N_FFT)     # <-- FIX: no ifftshift
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    return np.tile(ltf_sym, num_symbols).astype(np.complex64)

def create_ofdm_symbol(data_symbols: np.ndarray, sym_idx: int) -> np.ndarray:
    """
    One OFDM symbol with pilots and data.
    IMPORTANT: no ifftshift.
    """
    X = np.zeros(N_FFT, dtype=np.complex64)

    # Map data
    for i, kk in enumerate(DATA_SUBCARRIERS):
        X[k2i(int(kk), N_FFT)] = data_symbols[i]

    # Map pilots with alternating sign
    pilot_sign = 1 if (sym_idx % 2 == 0) else -1
    for i, kk in enumerate(PILOT_SUBCARRIERS):
        X[k2i(int(kk), N_FFT)] = pilot_sign * PILOT_VALUES[i]

    x = np.fft.ifft(X) * np.sqrt(N_FFT)     # <-- FIX: no ifftshift
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)


# ----------------------------
# Video packetization
# ----------------------------
def chunk_data(data: bytes, chunk_size: int):
    total = (len(data) + chunk_size - 1) // chunk_size
    return [(i, total, data[i * chunk_size: min((i + 1) * chunk_size, len(data))]) for i in range(total)]

def build_video_packet_bytes(frame_id: int, seq: int, total: int, payload: bytes) -> bytes:
    plen = len(payload)
    header = (
        MAGIC +
        (frame_id & 0xFFFF).to_bytes(2, "little") +
        (seq & 0xFFFF).to_bytes(2, "little") +
        (total & 0xFFFF).to_bytes(2, "little") +
        (plen & 0xFFFF).to_bytes(2, "little")
    )
    content = header + payload
    crc = zlib.crc32(content) & 0xFFFFFFFF
    return content + crc.to_bytes(4, "little")

def build_packet_bits(pkt_bytes: bytes, repeat: int, pad_to_ofdm: bool = True) -> tuple[np.ndarray, int]:
    """
    Returns (bits, num_ofdm_syms)
    """
    bits = bits_from_bytes(pkt_bytes)
    if repeat > 1:
        bits = np.repeat(bits, repeat).astype(np.uint8)

    num_ofdm_syms = int(np.ceil(len(bits) / BITS_PER_SYM))
    if pad_to_ofdm:
        total_bits = num_ofdm_syms * BITS_PER_SYM
        if len(bits) < total_bits:
            bits = np.pad(bits, (0, total_bits - len(bits)), constant_values=0)

    return bits.astype(np.uint8), num_ofdm_syms


# ----------------------------
# TX waveform builder
# ----------------------------
def build_tx_waveform(
    bits: np.ndarray,
    num_ofdm_syms: int,
    fs: float,
    stf: np.ndarray,
    ltf: np.ndarray,
    tx_scale: float,
    gap_short: int,
    gap_long: int,
    tone_duration_ms: float,
    tone_amp: float,
    tone_freq_hz: float,
    include_stf_ltf: bool,
) -> np.ndarray:
    """
    Assemble:
      [gap_long] + [optional tone] + [gap_short] + [STF+LTF] + [OFDM payload] + [gap_long]
    Scale to complex64 * 2^14 for Pluto.
    """

    # Optional tone
    tone = np.zeros(0, dtype=np.complex64)
    if tone_duration_ms > 0:
        tone_samples = int(max(1, tone_duration_ms * fs / 1000.0))
        t = np.arange(tone_samples, dtype=np.float32) / float(fs)
        tone = (tone_amp * np.exp(2j * np.pi * float(tone_freq_hz) * t)).astype(np.complex64)

    # OFDM payload
    ofdm_syms = []
    for si in range(num_ofdm_syms):
        b = bits[si * BITS_PER_SYM: (si + 1) * BITS_PER_SYM]
        data_syms = qpsk_map(b)  # length = 48 QPSK symbols
        ofdm_syms.append(create_ofdm_symbol(data_syms[:N_DATA], si))
    payload = np.concatenate(ofdm_syms).astype(np.complex64)

    gs = np.zeros(int(max(0, gap_short)), dtype=np.complex64)
    gl = np.zeros(int(max(0, gap_long)), dtype=np.complex64)

    pre = np.concatenate([stf, ltf]).astype(np.complex64) if include_stf_ltf else np.zeros(0, dtype=np.complex64)

    frame = np.concatenate([gl, tone, gs, pre, payload, gl]).astype(np.complex64)

    # Normalize and scale
    peak = np.max(np.abs(frame)) + 1e-12
    frame = (frame / peak) * float(tx_scale)

    # Pluto expects complex64 with amplitude roughly within [-1,1] before internal scaling,
    # but most pyadi-iio examples scale by 2^14 in complex64.
    return (frame * (2**14)).astype(np.complex64)


# ----------------------------
# JPEG source helpers
# ----------------------------
def jpeg_encode_frame(frame_bgr, width: int, height: int, quality: int) -> bytes | None:
    import cv2
    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
        frame_bgr = cv2.resize(frame_bgr, (width, height))
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, enc = cv2.imencode(".jpg", frame_bgr, params)
    if not ok:
        return None
    return enc.tobytes()

def synthetic_frame(width: int, height: int, frame_id: int):
    """Simple test pattern without camera dependency."""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # moving bar
    x = (frame_id * 7) % width
    img[:, max(0, x-10):min(width, x+10), :] = (0, 255, 255)
    cv2.putText(img, f"VID7 TX {frame_id}", (10, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return img


# ----------------------------
# Presets (you can extend)
# ----------------------------
PRESETS = {
    "antenna_close": dict(tx_gain=0.0,  tx_scale=0.70),
    "antenna_far":   dict(tx_gain=0.0,  tx_scale=0.90),
    "cable_30db":    dict(tx_gain=-10.0, tx_scale=0.50),
    "cable_direct":  dict(tx_gain=-30.0, tx_scale=0.10),
}


def main():
    ap = argparse.ArgumentParser(description="RF Link Step-7 TX (fixed + experiment switches)")
    # SDR
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=2.3e9, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=3e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=None, help="TX gain (dB), overrides preset")
    ap.add_argument("--preset", type=str, default="antenna_close",
                    choices=list(PRESETS.keys()), help="Experiment preset")
    ap.add_argument("--tx_scale", type=float, default=None,
                    help="Waveform scale after normalization (0..1), overrides preset")

    # Preamble / tone / gaps
    ap.add_argument("--stf_repeats", type=int, default=6, help="STF repeats (must match RX)")
    ap.add_argument("--ltf_symbols", type=int, default=4, help="LTF symbols (must match RX)")
    ap.add_argument("--no_preamble", action="store_true", help="Disable STF/LTF (debug only)")
    ap.add_argument("--tone_duration_ms", type=float, default=0.0,
                    help="Optional pilot tone duration before STF (ms). Default=0 (OFF)")
    ap.add_argument("--tone_amp", type=float, default=0.5, help="Pilot tone amplitude (0..1)")
    ap.add_argument("--tone_freq_hz", type=float, default=100e3, help="Pilot tone frequency (Hz)")
    ap.add_argument("--gap_short", type=int, default=1000, help="Gap (samples) between tone and preamble")
    ap.add_argument("--gap_long", type=int, default=3000, help="Gap (samples) at frame edges")

    # Coding / packet / timing
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Bit repetition factor")
    ap.add_argument("--chunk_size", type=int, default=1000, help="Payload bytes per packet")
    ap.add_argument("--dwell_time", type=float, default=0.20, help="Seconds to dwell per packet")
    ap.add_argument("--rounds_per_frame", type=int, default=1, help="Repeat a frame's packets N rounds")
    ap.add_argument("--packet_order", choices=["sequential", "reverse", "random"], default="sequential",
                    help="Packet ordering within a frame")

    # Source
    ap.add_argument("--video", type=str, default="0",
                    help="Video file path, '0' for webcam, or 'synthetic'")
    ap.add_argument("--mode", choices=["file", "stream"], default="stream",
                    help="file: finite, stream: continuous (webcam/synthetic)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--quality", type=int, default=30, help="JPEG quality (1-100)")
    ap.add_argument("--max_frames", type=int, default=30, help="Max frames in file mode")
    ap.add_argument("--frame_step", type=int, default=1, help="Send every Nth source frame")

    # TX buffer behavior
    ap.add_argument("--cyclic", action="store_true",
                    help="Use cyclic buffer for each packet waveform (recommended)")
    ap.add_argument("--destroy_each_packet", action="store_true",
                    help="Destroy TX buffer before each packet (safe for switching). "
                         "If not set, we destroy on waveform length change only.")
    ap.add_argument("--print_every_packet", action="store_true",
                    help="Print each packet line (otherwise compact)")

    args = ap.parse_args()

    # Apply preset defaults
    preset = PRESETS[args.preset]
    tx_gain = float(args.tx_gain) if args.tx_gain is not None else float(preset["tx_gain"])
    tx_scale = float(args.tx_scale) if args.tx_scale is not None else float(preset["tx_scale"])

    # Sanity
    if not (0.0 < tx_scale <= 1.0):
        raise ValueError("--tx_scale should be within (0,1].")
    if args.quality < 1 or args.quality > 100:
        raise ValueError("--quality must be 1..100")

    # Source open
    import cv2
    cap = None
    source_desc = ""
    if args.video == "synthetic":
        source_desc = "synthetic"
    elif args.video == "0":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam (video=0).")
        source_desc = "webcam"
    else:
        if not os.path.isfile(args.video):
            raise RuntimeError(f"Video file not found: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        source_desc = args.video

    # Build preamble refs
    stf = create_schmidl_cox_stf(num_repeats=args.stf_repeats)
    ltf = create_ltf(num_symbols=args.ltf_symbols)
    include_stf_ltf = (not args.no_preamble)

    print("\n" + "="*78)
    print("RF Link Step-7 TX (FIXED)")
    print("="*78)
    print(f"URI={args.uri}  fc={args.fc/1e6:.3f} MHz  fs={args.fs/1e6:.3f} Msps")
    print(f"preset={args.preset}  tx_gain={tx_gain} dB  tx_scale={tx_scale}")
    print(f"STF repeats={args.stf_repeats}  LTF symbols={args.ltf_symbols}  preamble={'ON' if include_stf_ltf else 'OFF'}")
    print(f"tone_ms={args.tone_duration_ms}  tone_amp={args.tone_amp}  tone_f={args.tone_freq_hz/1e3:.1f} kHz")
    print(f"repeat={args.repeat}  chunk={args.chunk_size}B  dwell={args.dwell_time}s  rounds/frame={args.rounds_per_frame}")
    print(f"order={args.packet_order}  cyclic={'ON' if args.cyclic else 'OFF'}  destroy_each_packet={args.destroy_each_packet}")
    print(f"source={source_desc}  mode={args.mode}  size={args.width}x{args.height}  jpegQ={args.quality}")
    print("="*78)

    # SDR config
    import adi
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(tx_gain)
    sdr.tx_cyclic_buffer = bool(args.cyclic)

    # TX loop
    frame_id = 0
    src_idx = 0
    total_pkts = 0
    t0 = time.time()
    last_wave_len = None

    def maybe_destroy_buffer(force: bool = False):
        nonlocal last_wave_len
        if force:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            last_wave_len = None

    try:
        while True:
            # Get frame
            if args.video == "synthetic":
                frame = synthetic_frame(args.width, args.height, frame_id)
                ret = True
            else:
                ret, frame = cap.read()

            if not ret:
                if args.mode == "file":
                    print(f"\n[TX] End of source. Sent {frame_id} frames.")
                    break
                # stream mode: reopen webcam if possible
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(0)
                time.sleep(0.2)
                continue

            src_idx += 1
            if (src_idx - 1) % args.frame_step != 0:
                continue

            jpeg = jpeg_encode_frame(frame, args.width, args.height, args.quality)
            if jpeg is None:
                print(f"[TX] JPEG encode failed for frame {frame_id}, skipping.")
                continue

            chunks = chunk_data(jpeg, args.chunk_size)
            n_pkts = len(chunks)

            # Packet ordering
            if args.packet_order == "reverse":
                chunks_iter = list(reversed(chunks))
            elif args.packet_order == "random":
                chunks_iter = chunks.copy()
                np.random.default_rng(frame_id + 123).shuffle(chunks_iter)
            else:
                chunks_iter = chunks

            print(f"\n[TX] Frame {frame_id}: jpeg={len(jpeg)}B  packets={n_pkts}")

            for rnd in range(args.rounds_per_frame):
                if args.rounds_per_frame > 1:
                    print(f"  round {rnd+1}/{args.rounds_per_frame}")

                for (seq, total, payload) in chunks_iter:
                    pkt_bytes = build_video_packet_bytes(frame_id, seq, total, payload)
                    bits, num_syms = build_packet_bits(pkt_bytes, repeat=args.repeat, pad_to_ofdm=True)

                    wave = build_tx_waveform(
                        bits=bits,
                        num_ofdm_syms=num_syms,
                        fs=args.fs,
                        stf=stf,
                        ltf=ltf,
                        tx_scale=tx_scale,
                        gap_short=args.gap_short,
                        gap_long=args.gap_long,
                        tone_duration_ms=args.tone_duration_ms,
                        tone_amp=args.tone_amp,
                        tone_freq_hz=args.tone_freq_hz,
                        include_stf_ltf=include_stf_ltf,
                    )

                    # Safe cyclic buffer handling:
                    # - If user requested destroy_each_packet, always destroy.
                    # - Otherwise, destroy only when waveform length changes (pyadi can be picky).
                    if args.destroy_each_packet or (last_wave_len is not None and len(wave) != last_wave_len):
                        maybe_destroy_buffer(force=True)

                    # Set cyclic ONCE BEFORE first tx() after destroy.
                    # If buffer exists, pyadi refuses changing tx_cyclic_buffer.
                    # So we only assign tx_cyclic_buffer at init, not here.
                    sdr.tx(wave)
                    last_wave_len = len(wave)

                    total_pkts += 1
                    if args.print_every_packet:
                        print(f"    pkt {seq}/{total-1}  payload={len(payload)}B  ofdm_syms={num_syms}  wave={len(wave)}  dwell={args.dwell_time}s")
                    else:
                        print(f"\r    pkt {seq+1:>3}/{total:<3}  payload={len(payload):>4}B  syms={num_syms:<3}  dwell={args.dwell_time:.2f}s",
                              end="", flush=True)

                    time.sleep(max(0.0, args.dwell_time))

                if not args.print_every_packet:
                    print()

            frame_id += 1
            elapsed = time.time() - t0
            eff_fps = frame_id / elapsed if elapsed > 0 else 0.0
            print(f"[TX] elapsed={elapsed:.1f}s  eff_fps={eff_fps:.3f}  total_pkts={total_pkts}")

            if args.mode == "file" and frame_id >= args.max_frames:
                print(f"\n[TX] Reached max_frames={args.max_frames}.")
                break

    except KeyboardInterrupt:
        print("\n[TX] Stopped by user.")
    finally:
        if cap is not None:
            cap.release()
        maybe_destroy_buffer(force=True)

    elapsed = time.time() - t0
    print("\n" + "-"*78)
    print("TX SUMMARY")
    print(f"frames_sent={frame_id}  packets_sent={total_pkts}  elapsed={elapsed:.1f}s  eff_fps={(frame_id/elapsed if elapsed>0 and frame_id>0 else 0):.3f}")
    print("-"*78)


if __name__ == "__main__":
    main()

"""
#Best “start-safe” antenna-close experiment (tone off):
python rf_link_step7_tx_fixed.py \
  --uri ip:192.168.2.1 --fc 2.3e9 --fs 3e6 \
  --preset antenna_close \
  --tone_duration_ms 0 \
  --video 0 --mode stream \
  --chunk_size 1000 --dwell_time 0.2

#Cable + 30 dB attenuator (lower TX gain):
python rf_link_step7_tx_fixed.py \
  --uri ip:192.168.2.1 --fc 915e6 --fs 2e6 \
  --preset cable_30db \
  --video synthetic --mode stream \
  --dwell_time 0.15

#Turn tone on for diagnosis (not recommended until stable):
python rf_link_step7_tx_fixed.py \
  --preset antenna_close \
  --tone_duration_ms 10 --tone_amp 0.3 --tone_freq_hz 100e3

#Packet-order stress test (random) + repetition coding:
python rf_link_step7_tx_fixed.py \
  --preset antenna_close \
  --repeat 2 --packet_order random --rounds_per_frame 2
"""
