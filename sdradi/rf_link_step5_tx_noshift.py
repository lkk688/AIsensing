#!/usr/bin/env python3
"""
RF Link Test - Step 5: TX (NO-SHIFT FFT convention, matches rf_step5_rx_stable.py)

Key point:
- Subcarrier mapping uses UN-SHIFTED FFT bins:
    bin = (k + N) % N
- IFFT uses np.fft.ifft(X)  (NO ifftshift)
Therefore RX must use np.fft.fft() (NO fftshift) and the same bin mapping.

Packet format (AIS1):
  MAGIC("AIS1", 4B) | LEN(2B) | PAYLOAD | CRC32(4B)
  CRC32 covers PAYLOAD only (same as your existing Step4/Step5).

Preamble:
  gap_long | optional tone | gap_short | STF(with CP) | LTF(with CP, repeated) | OFDM payload | gap_long

Default mode is cyclic TX (repeats the same frame). Use --burst to send once.

Run on TX device (Jetson).
"""

import argparse
import time
import zlib
from dataclasses import dataclass
import numpy as np

# =========================
# OFDM Parameters
# =========================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array(
    [k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))],
    dtype=int
)
N_DATA = len(DATA_SUBCARRIERS)          # 48
BITS_PER_OFDM_SYM = N_DATA * 2          # QPSK

MAGIC = b"AIS1"

def sc_to_bin(k: int, N: int = N_FFT) -> int:
    """Map subcarrier index k in [-N/2..N/2-1] to UN-SHIFTED FFT bin index [0..N-1]."""
    return (k + N) % N

PILOT_BINS = np.array([sc_to_bin(k) for k in PILOT_SUBCARRIERS], dtype=int)
DATA_BINS  = np.array([sc_to_bin(k) for k in DATA_SUBCARRIERS], dtype=int)

# =========================
# Modulation / framing
# =========================
def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK mapping, returns complex64 symbols, normalized by sqrt(2)."""
    bits = bits.astype(np.uint8)
    assert bits.size % 2 == 0
    out = np.zeros(bits.size // 2, dtype=np.complex64)
    for i in range(out.size):
        b0, b1 = int(bits[2*i]), int(bits[2*i+1])
        if b0 == 0 and b1 == 0:
            s = 1 + 1j
        elif b0 == 0 and b1 == 1:
            s = -1 + 1j
        elif b0 == 1 and b1 == 1:
            s = -1 - 1j
        else:
            s = 1 - 1j
        out[i] = s
    return out / np.sqrt(2)

def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)

def build_packet(payload: bytes, repeat: int = 1):
    """
    MAGIC(4) | LEN(2) | PAYLOAD | CRC32(4)
    CRC32 covers PAYLOAD only.
    """
    plen = len(payload)
    header = MAGIC + plen.to_bytes(2, "little")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    frame = header + payload + crc.to_bytes(4, "little")

    bits = bits_from_bytes(frame)
    if repeat > 1:
        bits = np.repeat(bits, repeat).astype(np.uint8)

    num_syms = int(np.ceil(bits.size / BITS_PER_OFDM_SYM))
    total_bits = num_syms * BITS_PER_OFDM_SYM
    if bits.size < total_bits:
        bits = np.pad(bits, (0, total_bits - bits.size)).astype(np.uint8)

    return frame, bits, num_syms, crc

# =========================
# Preamble (NO-SHIFT FFT)
# =========================
def create_schmidl_cox_stf(num_repeats: int = 6):
    """
    STF with N/2 periodicity in time domain by using only even subcarriers.
    Build X in UN-SHIFTED bins and do ifft(X) (NO ifftshift).
    """
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i] + 0j

    x = np.fft.ifft(X) * np.sqrt(N_FFT)  # NO ifftshift
    stf = np.tile(x, num_repeats).astype(np.complex64)

    # Add CP once for entire STF block (match many of your RX refs)
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp, X

def create_ltf(num_symbols: int = 4):
    """
    LTF with deterministic BPSK on all used subcarriers.
    Build X in UN-SHIFTED bins and do ifft(X) (NO ifftshift).
    """
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j

    x = np.fft.ifft(X) * np.sqrt(N_FFT)  # NO ifftshift
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf, X

def create_ofdm_symbol(data_symbols: np.ndarray, pilot_values: np.ndarray, sym_idx: int) -> np.ndarray:
    """
    One OFDM symbol: fill UN-SHIFTED bins, then ifft(X) (NO ifftshift), add CP.
    """
    X = np.zeros(N_FFT, dtype=np.complex64)

    nfill = min(data_symbols.size, DATA_BINS.size)
    if nfill > 0:
        X[DATA_BINS[:nfill]] = data_symbols[:nfill]

    pilot_sign = 1 if (sym_idx % 2 == 0) else -1
    X[PILOT_BINS] = pilot_sign * pilot_values

    x = np.fft.ifft(X) * np.sqrt(N_FFT)  # NO ifftshift
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)

def build_waveform(
    bits: np.ndarray,
    num_ofdm_syms: int,
    fs: float,
    tone_duration_ms: float,
    tone_freq_hz: float,
    tone_amp: float,
    stf_repeats: int,
    ltf_symbols: int,
    gap_short: int,
    gap_long: int,
    tx_scale: float,
    include_preamble: bool = True,
):
    # Optional tone
    if tone_duration_ms > 0:
        tone_samples = int(tone_duration_ms * fs / 1000.0)
        t = np.arange(tone_samples, dtype=np.float32) / float(fs)
        pilot_tone = (tone_amp * np.exp(2j * np.pi * tone_freq_hz * t)).astype(np.complex64)
    else:
        pilot_tone = np.zeros(0, dtype=np.complex64)

    stf, _ = create_schmidl_cox_stf(num_repeats=stf_repeats)
    ltf, _ = create_ltf(num_symbols=ltf_symbols)

    pilot_values = np.array([1, 1, 1, -1], dtype=np.complex64)
    ofdm_syms = []
    for si in range(num_ofdm_syms):
        sb = si * BITS_PER_OFDM_SYM
        eb = sb + BITS_PER_OFDM_SYM
        sym_bits = bits[sb:eb]
        data_syms = qpsk_map(sym_bits)
        ofdm_syms.append(create_ofdm_symbol(data_syms, pilot_values, si))
    ofdm_payload = np.concatenate(ofdm_syms).astype(np.complex64) if ofdm_syms else np.zeros(0, dtype=np.complex64)

    gS = np.zeros(int(gap_short), dtype=np.complex64)
    gL = np.zeros(int(gap_long), dtype=np.complex64)

    if include_preamble:
        frame = np.concatenate([gL, pilot_tone, gS, stf, ltf, ofdm_payload, gL]).astype(np.complex64)
    else:
        frame = np.concatenate([gL, ofdm_payload, gL]).astype(np.complex64)

    # Normalize + scale (Pluto expects complex float-ish; many examples use int14 scale)
    m = float(np.max(np.abs(frame)) + 1e-9)
    frame = frame / m
    frame = frame * float(tx_scale)  # keep headroom
    tx = (frame * (2**14)).astype(np.complex64)
    return tx, len(pilot_tone), len(stf), len(ltf), len(ofdm_payload), len(frame)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("Step5 TX (NO-SHIFT) - matches rf_step5_rx_stable.py")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI (usb:... or ip:...)")
    ap.add_argument("--fc", type=float, default=2.3e9, help="Center freq (Hz)")
    ap.add_argument("--fs", type=float, default=3e6, help="Sample rate (Hz)")
    ap.add_argument("--tx_gain", type=float, default=-20, help="TX gain (dB)")

    ap.add_argument("--tone_duration_ms", type=float, default=10.0, help="Pilot tone duration (ms); 0 disables")
    ap.add_argument("--tone_freq_hz", type=float, default=100e3, help="Pilot tone offset (Hz)")
    ap.add_argument("--tone_amp", type=float, default=0.5, help="Tone amplitude before normalization")

    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--gap_short", type=int, default=1000)
    ap.add_argument("--gap_long", type=int, default=3000)
    ap.add_argument("--tx_scale", type=float, default=0.7, help="Final waveform scale after normalization")

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4], help="Bit repetition")
    ap.add_argument("--payload", type=str, default="", help="Payload string (packet mode)")
    ap.add_argument("--payload_len", type=int, default=64, help="Random payload length if --payload empty")

    ap.add_argument("--ber_test", action="store_true", help="BER mode: send known pseudo-random bits (no AIS1 framing)")
    ap.add_argument("--ber_num_syms", type=int, default=100, help="OFDM symbols in BER mode")
    ap.add_argument("--seed", type=int, default=99999, help="Seed for BER bits / random payload")

    ap.add_argument("--no_preamble", action="store_true", help="Disable preamble (for debugging only)")
    ap.add_argument("--burst", action="store_true", help="Send once (non-cyclic) then exit")
    ap.add_argument("--dwell_s", type=float, default=0.0, help="If burst, sleep this long after tx() before destroy")
    args = ap.parse_args()

    import adi

    # Build bits
    if args.ber_test:
        rng = np.random.default_rng(int(args.seed))
        bits = rng.integers(0, 2, size=int(args.ber_num_syms * BITS_PER_OFDM_SYM), dtype=np.uint8)
        num_ofdm_syms = int(args.ber_num_syms)
        frame_bytes = b""
        crc = 0
        mode_str = f"BER_TEST syms={num_ofdm_syms} seed={args.seed}"
    else:
        if args.payload:
            payload = args.payload.encode("utf-8")
        else:
            rng = np.random.default_rng(int(args.seed))
            payload = rng.integers(0, 256, size=int(args.payload_len), dtype=np.uint8).tobytes()
        frame_bytes, bits, num_ofdm_syms, crc = build_packet(payload, repeat=int(args.repeat))
        mode_str = f"PACKET payload={len(payload)}B frame={len(frame_bytes)}B repeat={args.repeat}"

    tx, tone_len, stf_len, ltf_len, pay_len, total_len = build_waveform(
        bits=bits,
        num_ofdm_syms=num_ofdm_syms,
        fs=float(args.fs),
        tone_duration_ms=float(args.tone_duration_ms),
        tone_freq_hz=float(args.tone_freq_hz),
        tone_amp=float(args.tone_amp),
        stf_repeats=int(args.stf_repeats),
        ltf_symbols=int(args.ltf_symbols),
        gap_short=int(args.gap_short),
        gap_long=int(args.gap_long),
        tx_scale=float(args.tx_scale),
        include_preamble=(not args.no_preamble),
    )

    print("\n" + "="*78)
    print("RF Link Step5 TX (NO-SHIFT FFT) - matches rf_step5_rx_stable.py")
    print("="*78)
    print(f"uri={args.uri}  fc={args.fc/1e6:.1f}MHz  fs={args.fs/1e6:.1f}Msps  tx_gain={args.tx_gain}dB")
    print(f"FFT_MODE=no_shift  bin=(k+N)%N  IFFT=ifft(X)  (NO ifftshift)")
    print(f"mode={mode_str}")
    if not args.ber_test:
        print(f"crc32(payload)=0x{crc:08X}")
    print(f"preamble={'OFF' if args.no_preamble else 'ON'}  tone={args.tone_duration_ms:.1f}ms @ {args.tone_freq_hz/1e3:.1f}kHz")
    print(f"stf_repeats={args.stf_repeats}  ltf_symbols={args.ltf_symbols}")
    print(f"lens: tone={tone_len} stf={stf_len} ltf={ltf_len} payload={pay_len} total={total_len} samples")
    print(f"tx dtype={tx.dtype}  tx max|x|={np.max(np.abs(tx)):.1f}")
    print("="*78)

    # SDR setup
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    sdr.tx_enabled_channels = [0]

    try:
        # Ensure clean TX state
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

        if args.burst:
            # Non-cyclic one-shot
            sdr.tx_cyclic_buffer = False
            sdr.tx(tx)
            if args.dwell_s > 0:
                time.sleep(float(args.dwell_s))
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            print("TX burst sent. Done.")
        else:
            # Cyclic continuous
            sdr.tx_cyclic_buffer = True
            sdr.tx(tx)
            print("TX running in CYCLIC mode (Ctrl+C to stop)")
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
python3 rf_link_step5_tx_noshift.py \
  --uri ip:192.168.3.2 \
  --fc 2.3e9 --fs 3e6 \
  --tx_gain -20 \
  --tone_duration_ms 0 \
  --payload "step5 noshift no-tone"


python3 rf_step5_rx_stable.py \
  --uri ip:192.168.2.2 \
  --fc 2.3e9 --fs 3e6 \
  --rx_gain 30 \
  --buf_size 262144 \
  --tries 30 \
  --cfo_mode sc \
  --sc_threshold 0.08 \
  --probe_syms 16 \
  --xcorr_refine \
  --verbose
"""
