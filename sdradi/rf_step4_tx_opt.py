#!/usr/bin/env python3
"""
RF Link Test - Step 4: TX (Full Packet with CRC) - OPT version
- Keep the same on-air format as your working version.
- Only optimize implementation + add optional TX self-check figures.
"""

import argparse
import time
import zlib
import numpy as np

# OFDM Parameters
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)  # 48
BITS_PER_OFDM_SYM = N_DATA * 2  # QPSK

MAGIC = b"AIS1"

# ---- IMPORTANT NOTE ----
# This code keeps your original bin indexing convention:
#   idx = (k + N) % N
# together with ifft(ifftshift(X)).
# It is not the "textbook" mapping, but since your TX/RX already worked once,
# we keep it unchanged to preserve self-consistency.


def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def qpsk_map_gray(bits: np.ndarray) -> np.ndarray:
    """
    Vectorized Gray QPSK mapping.
      00 -> +1 + j
      01 -> -1 + j
      11 -> -1 - j
      10 -> +1 - j
    """
    bits = bits.astype(np.uint8)
    if len(bits) % 2 != 0:
        bits = np.pad(bits, (0, 1))
    b0 = bits[0::2]
    b1 = bits[1::2]

    # Start with signs
    re = np.where(b0 == 0, 1.0, -1.0)
    im = np.where(b1 == 0, 1.0, -1.0)

    # Fix Gray mapping for 10 case:
    # Our re/im gives (b0=1,b1=0)->(-1,+1j) but we want (+1,-1j).
    # Easiest: explicit table via boolean masks:
    syms = re + 1j * im
    m10 = (b0 == 1) & (b1 == 0)
    syms[m10] = (1.0 - 1.0j)
    m11 = (b0 == 1) & (b1 == 1)
    syms[m11] = (-1.0 - 1.0j)
    m01 = (b0 == 0) & (b1 == 1)
    syms[m01] = (-1.0 + 1.0j)
    m00 = (b0 == 0) & (b1 == 0)
    syms[m00] = (1.0 + 1.0j)

    return (syms / np.sqrt(2)).astype(np.complex64)


def create_stf(N=64) -> np.ndarray:
    """Create STF exactly as your working code: random BPSK on even subcarriers, repeat x2, then CP later."""
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i] + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    stf = np.tile(x, 2).astype(np.complex64)  # length 2N
    return stf


def create_ltf(N=64) -> np.ndarray:
    """Create LTF exactly as your working code: used bins BPSK alternating, (CP+N) then repeat x2."""
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N) % N] = (1.0 if i % 2 == 0 else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])
    ltf = np.tile(ltf_sym, 2).astype(np.complex64)
    return ltf


def create_ofdm_symbol(data_symbols: np.ndarray, pilot_values: np.ndarray, sym_idx: int) -> np.ndarray:
    """Create one OFDM symbol (CP+N) using your original bin mapping convention."""
    X = np.zeros(N_FFT, dtype=np.complex64)

    # Fill data bins
    # NOTE: keep idx = (k+N)%N to preserve compatibility with your RX.
    nfill = min(len(data_symbols), N_DATA)
    for i in range(nfill):
        k = DATA_SUBCARRIERS[i]
        X[(k + N_FFT) % N_FFT] = data_symbols[i]

    # Pilots with alternating sign
    pilot_sign = 1.0 if (sym_idx % 2 == 0) else -1.0
    for i, k in enumerate(PILOT_SUBCARRIERS):
        X[(k + N_FFT) % N_FFT] = pilot_sign * pilot_values[i]

    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)


def build_packet_bits(payload: bytes, repeat: int):
    """Frame: MAGIC(4) + LEN(2) + PAYLOAD + CRC32(4). Return repeated bits and symbol count."""
    plen = len(payload)
    header = MAGIC + plen.to_bytes(2, "little")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    frame = header + payload + crc.to_bytes(4, "little")

    bits = bits_from_bytes(frame)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    num_syms = int(np.ceil(len(bits) / BITS_PER_OFDM_SYM))
    total_bits = num_syms * BITS_PER_OFDM_SYM
    if len(bits) < total_bits:
        bits = np.pad(bits, (0, total_bits - len(bits)))

    return bits.astype(np.uint8), num_syms, len(frame), crc


def save_tx_debug_plots(x: np.ndarray, fs: float, outdir: str, tag: str):
    """Optional: save TX waveform self-check plots (time, PSD, magnitude histogram)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs(outdir, exist_ok=True)
    N = len(x)
    x0 = x.astype(np.complex64)

    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(np.real(x0[:6000]), label="I")
    ax1.plot(np.imag(x0[:6000]), label="Q")
    ax1.set_title("TX time-domain (first 6k)")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    mag = np.abs(x0)
    ax2.plot(mag[:6000])
    ax2.set_title("TX |x| (first 6k)")
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(np.real(x0), bins=80, alpha=0.7, label="I")
    ax3.hist(np.imag(x0), bins=80, alpha=0.7, label="Q")
    ax3.set_title("TX I/Q histogram")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(2, 3, 4)
    # PSD
    win = np.hanning(N).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x0 * win))
    psd = (np.abs(X) ** 2) / (np.sum(win**2) + 1e-12)
    psd_db = 10 * np.log10(psd + 1e-20)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / fs))
    ax4.plot(freqs / 1e3, psd_db)
    ax4.set_title("TX PSD (dB)")
    ax4.set_xlabel("kHz")
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(np.real(x0[:20000]), np.imag(x0[:20000]), s=2, alpha=0.25)
    ax5.set_title("TX IQ scatter (first 20k)")
    ax5.axis("equal")
    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    papr = 10*np.log10((np.max(np.abs(x0))**2 + 1e-12)/(np.mean(np.abs(x0)**2)+1e-12))
    ax6.axis("off")
    ax6.text(0.02, 0.98,
             f"N={N}\nfs={fs:.0f} Hz\nPAPR~{papr:.2f} dB\nmax|x|={np.max(np.abs(x0)):.3f}\nmean|x|={np.mean(np.abs(x0)):.3f}",
             va="top", family="monospace")

    fig.tight_layout()
    outp = os.path.join(outdir, f"tx_debug_{tag}.png")
    fig.savefig(outp, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - TX (Step 4) OPT")
    ap.add_argument("--uri", default="ip:192.168.2.1")
    ap.add_argument("--fc", type=float, default=915e6)
    ap.add_argument("--fs", type=float, default=2e6)
    ap.add_argument("--bw", type=float, default=0.0, help="0=auto use 1.2*fs")
    ap.add_argument("--tx_gain", type=float, default=-5)
    ap.add_argument("--tone_duration_ms", type=float, default=5)
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--payload", type=str, default="")
    ap.add_argument("--payload_len", type=int, default=64)
    ap.add_argument("--infile", type=str, default="")
    ap.add_argument("--tone_hz", type=float, default=100e3)
    ap.add_argument("--save_tx_debug", action="store_true")
    ap.add_argument("--tx_debug_dir", default="tx_debug_step4")
    args = ap.parse_args()

    import adi

    # Payload selection
    if args.infile:
        with open(args.infile, "rb") as f:
            payload = f.read()
        print(f"Using file payload: {len(payload)} bytes from {args.infile}")
    elif args.payload:
        payload = args.payload.encode("utf-8")
        print(f"Using string payload: {len(payload)} bytes")
    else:
        rng = np.random.default_rng(54321)
        payload = rng.integers(0, 256, size=args.payload_len, dtype=np.uint8).tobytes()
        print(f"Using random payload: {len(payload)} bytes")

    # Packet bits
    bits, num_ofdm_syms, frame_len, crc = build_packet_bits(payload, args.repeat)

    # Pilot tone (for CFO est on RX)
    tone_samples = int(args.tone_duration_ms * args.fs / 1000)
    t = np.arange(tone_samples, dtype=np.float64) / args.fs
    pilot_tone = (np.exp(2j * np.pi * args.tone_hz * t).astype(np.complex64) * 0.5)

    # Preamble
    stf = create_stf(N_FFT)
    ltf = create_ltf(N_FFT)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)

    # OFDM payload
    pilot_values = np.array([1, 1, 1, -1], dtype=np.complex64)
    ofdm_syms = []
    for sym_idx in range(num_ofdm_syms):
        sb = sym_idx * BITS_PER_OFDM_SYM
        eb = sb + BITS_PER_OFDM_SYM
        sym_bits = bits[sb:eb]
        data_syms = qpsk_map_gray(sym_bits)
        ofdm_syms.append(create_ofdm_symbol(data_syms, pilot_values, sym_idx))
    ofdm_payload = np.concatenate(ofdm_syms).astype(np.complex64)

    # Frame layout (keep your original gaps)
    gap_short = np.zeros(500, dtype=np.complex64)
    gap_long  = np.zeros(2000, dtype=np.complex64)

    frame = np.concatenate([
        gap_long,
        pilot_tone,
        gap_short,
        stf_with_cp,
        ltf,
        ofdm_payload,
        gap_long
    ]).astype(np.complex64)

    # Normalize & scale (avoid clipping)
    frame = frame / (np.max(np.abs(frame)) + 1e-9)
    frame = (frame * 0.7).astype(np.complex64)
    tx_scaled = (frame * (2**14)).astype(np.complex64)

    print("\nRF Link Test - Step 4: TX (Full Packet) OPT")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  TX gain: {args.tx_gain} dB")
    print(f"  Tone: {args.tone_hz/1e3:.1f} kHz, dur={args.tone_duration_ms:.2f} ms")
    print("\n  Packet info:")
    print(f"    Payload: {len(payload)} bytes")
    print(f"    Frame: {frame_len} bytes (with header+CRC)")
    print(f"    Repetition: {args.repeat}x")
    print(f"    OFDM symbols: {num_ofdm_syms}")
    print(f"    Total bits (after repeat+pad): {len(bits)}")
    print(f"    CRC32(payload): 0x{crc:08X}")
    print(f"\n  Signal samples: {len(frame)}")
    print("")

    if args.save_tx_debug:
        save_tx_debug_plots(frame, args.fs, args.tx_debug_dir, tag="step4")

    # SDR setup
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    bw = int(args.bw) if args.bw > 0 else int(args.fs * 1.2)
    sdr.tx_rf_bandwidth = bw
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    sdr.tx_cyclic_buffer = True

    sdr.tx(tx_scaled)

    print("TX Running (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping TX...")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass


if __name__ == "__main__":
    main()

"""
python3 rf_step4_tx_opt.py \
  --uri "ip:192.168.3.2" \
  --fc 915e6 --fs 2e6 \
  --tx_gain -5 \
  --payload "Hello from Jetson! This is a test message via OFDM RF link." \
  --repeat 1 \
  --tone_duration_ms 5 \
  --save_tx_debug
"""