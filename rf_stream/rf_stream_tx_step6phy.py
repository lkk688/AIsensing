#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_tx_step6phy.py  – Step 6 TX

Builds on step5 (non-cyclic streaming, ring buffer, Schmidl-Cox STF, multi-LTF,
TX hardware conjugation) and adds:
  • Multi-modulation: BPSK, QPSK (default), QAM8 (8-PSK), QAM16, QAM32
  • BER evaluation mode: transmit known PRNG payload (--ref_seed / --ref_len)
  • Run-metadata JSON saved to --out_root at startup (consumed by analyzer)
  • Sweep mode unchanged from step5

Hardware note: TX sends  np.conj(buf) * 4096  (IQ-swap compensation for PlutoSDR).
RX step6 compensates with conj(rxw) in its outer demod loop.
"""

import argparse
import json
import os
import queue
import threading
import time
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np


# =============================================================================
# Step6 PHY params  (must match RX)
# =============================================================================
N_FFT    = 64
N_CP     = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array(
    [k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)],
    dtype=int,
)
N_DATA   = len(DATA_SUBCARRIERS)          # 48
MAGIC    = b"AIS1"

def sc_to_bin(k: int) -> int:
    return (k + N_FFT) % N_FFT

DATA_BINS  = np.array([sc_to_bin(int(k)) for k in DATA_SUBCARRIERS], dtype=int)
PILOT_BINS = np.array([sc_to_bin(int(k)) for k in PILOT_SUBCARRIERS], dtype=int)


# =============================================================================
# Modulation tables  (shared logic duplicated in RX for standalone use)
# =============================================================================
MOD_BPS = {"bpsk": 1, "qpsk": 2, "qam8": 3, "qam16": 4, "qam32": 5}


def make_constellation(mod: str) -> np.ndarray:
    """Return normalized complex64 constellation table.
    table[i] = symbol for bit pattern i (MSB-first, i.e. i=0b...b0b1).
    """
    mod = mod.lower()
    if mod == "bpsk":
        return np.array([-1+0j, 1+0j], dtype=np.complex64)

    if mod == "qpsk":
        # Gray: 00→NE  01→NW  11→SW  10→SE
        return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)

    if mod == "qam8":
        # 8-PSK Gray coded:
        # bit-pattern 0(000)→0°, 1(001)→45°, 2(010)→135°, 3(011)→90°
        #             4(100)→315°, 5(101)→270°, 6(110)→180°, 7(111)→225°
        phase_idx = [0, 1, 3, 2, 6, 7, 5, 4]  # maps int-value to PSK-phase index
        return np.exp(1j * np.pi / 4.0 * np.array(phase_idx, dtype=float)).astype(np.complex64)

    if mod == "qam16":
        # Standard Gray-coded 16-QAM:  top-2-bits → I, bottom-2-bits → Q
        # Gray decode: 00→-3, 01→-1, 11→+1, 10→+3  (divide by sqrt(10) to normalize)
        g = {0b00: -3, 0b01: -1, 0b11: 1, 0b10: 3}
        t = np.array([g[(i >> 2) & 3] + 1j * g[i & 3] for i in range(16)], dtype=np.complex64)
        return t / np.sqrt(10.0)

    if mod == "qam32":
        # Cross 32-QAM: 6×6 Cartesian grid minus 4 corner points (|re|=5 AND |im|=5)
        pts = np.array(
            [r + 1j * m for r in (-5, -3, -1, 1, 3, 5) for m in (-5, -3, -1, 1, 3, 5)
             if not (abs(r) == 5 and abs(m) == 5)],
            dtype=np.complex64,
        )
        pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
        # Consistent ordering: row descending (im), column ascending (re)
        order = np.lexsort((pts.real, -pts.imag))
        return pts[order]

    raise ValueError(f"Unknown modulation: {mod!r}")


def bits_to_symbols(bits: np.ndarray, table: np.ndarray, bps: int) -> np.ndarray:
    """Map MSB-first bit array to complex symbols."""
    n = len(bits) // bps
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    b = bits[:n * bps].astype(np.int32).reshape(n, bps)
    powers = 2 ** np.arange(bps - 1, -1, -1, dtype=np.int32)
    indices = (b * powers).sum(axis=1).astype(np.int32)
    return table[indices]


# =============================================================================
# Helpers
# =============================================================================
def bits_from_bytes(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='big').astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits, bitorder='big').tobytes()

def scramble_bits(bits: np.ndarray, seed: int = 0x7F) -> np.ndarray:
    state, out = seed, np.zeros_like(bits)
    for i in range(len(bits)):
        b7 = (state >> 6) & 1
        b4 = (state >> 3) & 1
        out[i] = bits[i] ^ b7
        state = ((state << 1) | (b7 ^ b4)) & 0x7F
    return out

def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    X   = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i]
    x   = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x.astype(np.complex64), num_repeats)
    return np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)

def create_ltf(num_symbols: int = 4) -> np.ndarray:
    X    = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = 1.0 if (i % 2 == 0) else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    return np.tile(ltf_sym, num_symbols).astype(np.complex64)

def create_ofdm_symbol(data_syms: np.ndarray, pilot_vals: np.ndarray, sym_idx: int) -> np.ndarray:
    X = np.zeros(N_FFT, dtype=np.complex64)
    nfill = min(len(data_syms), len(DATA_BINS))
    X[DATA_BINS[:nfill]] = data_syms[:nfill]
    pilot_sign = 1 if (sym_idx % 2 == 0) else -1
    X[PILOT_BINS] = pilot_sign * pilot_vals
    x  = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)

def build_packet_bytes(seq: int, total: int, payload: bytes) -> bytes:
    hdr  = (MAGIC
            + int(seq).to_bytes(2, "little")
            + int(total).to_bytes(2, "little")
            + int(len(payload)).to_bytes(2, "little"))
    body = hdr + payload
    crc  = zlib.crc32(body) & 0xFFFFFFFF
    return body + int(crc).to_bytes(4, "little")

def bytes_to_ofdm_samples(
    frame_bytes: bytes,
    repeat: int,
    stf: np.ndarray,
    ltf: np.ndarray,
    const_table: np.ndarray,
    bps: int,
    fs: float,
    tone_duration_ms: float,
    tone_freq_hz: float,
    gap_short: int,
    gap_long: int,
    tx_scale: float,
) -> np.ndarray:
    bits = bits_from_bytes(frame_bytes)
    bits = scramble_bits(bits)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    bpos     = N_DATA * bps               # bits per OFDM symbol
    num_syms = int(np.ceil(len(bits) / bpos))
    need     = num_syms * bpos
    if len(bits) < need:
        bits = np.pad(bits, (0, need - len(bits)))

    pilot_vals = np.array([1, 1, 1, -1], dtype=np.complex64)
    ofdm = []
    for si in range(num_syms):
        sb = bits[si * bpos: (si + 1) * bpos]
        ds = bits_to_symbols(sb, const_table, bps)
        ofdm.append(create_ofdm_symbol(ds, pilot_vals, si))
    ofdm = np.concatenate(ofdm).astype(np.complex64)

    tone_samps = int(tone_duration_ms * fs / 1000.0)
    if tone_samps > 0:
        t    = np.arange(tone_samps, dtype=np.float32) / float(fs)
        tone = np.exp(2j * np.pi * float(tone_freq_hz) * t).astype(np.complex64) * 0.5
    else:
        tone = np.zeros(0, dtype=np.complex64)

    gS  = np.zeros(int(gap_short), dtype=np.complex64)
    gL  = np.zeros(int(gap_long),  dtype=np.complex64)
    sig = np.concatenate([gL, tone, gS, stf, ltf, ofdm, gL]).astype(np.complex64)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * float(tx_scale)
    return sig

def fit_to_fixed_len(sig: np.ndarray, fixed_len: int) -> np.ndarray:
    if len(sig) == fixed_len:
        return sig
    if len(sig) > fixed_len:
        return sig[:fixed_len].copy()
    out = np.zeros(fixed_len, dtype=np.complex64)
    out[:len(sig)] = sig
    return out


# =============================================================================
# Config
# =============================================================================
@dataclass
class TxConfig:
    uri: str
    fc: float
    fs: float
    tx_gain: float
    tx_bw: float
    fixed_len: int
    repeat: int
    stf_repeats: int
    ltf_symbols: int
    modulation: str
    bps: int
    tone_duration_ms: float
    tone_freq_hz: float
    gap_short: int
    gap_long: int
    tx_scale: float
    idle_amp: float
    send_interval_s: float
    beacon_period_s: float
    mode: str
    sweep_freqs: list
    sweep_time_s: float
    out_root: str
    run_id: str


def make_idle_buffer(cfg: TxConfig) -> np.ndarray:
    if cfg.idle_amp <= 0:
        return np.zeros(cfg.fixed_len, dtype=np.complex64)
    rng = np.random.default_rng(0)
    n   = (rng.standard_normal(cfg.fixed_len) + 1j * rng.standard_normal(cfg.fixed_len)).astype(np.complex64)
    return (n / (np.max(np.abs(n)) + 1e-9) * cfg.idle_amp).astype(np.complex64)


# =============================================================================
# TX worker  (non-cyclic streaming – unchanged from step5)
# =============================================================================
def tx_worker(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: TxConfig):
    import adi
    sdr = adi.Pluto(uri=cfg.uri)
    sdr.sample_rate           = int(cfg.fs)
    sdr.tx_lo                 = int(cfg.fc)
    sdr.tx_rf_bandwidth       = int(cfg.tx_bw)
    sdr.tx_hardwaregain_chan0  = float(cfg.tx_gain)
    sdr.tx_enabled_channels   = [0]
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False

    idle      = fit_to_fixed_len(make_idle_buffer(cfg), cfg.fixed_len).astype(np.complex64)
    last_sig: Optional[np.ndarray] = None
    last_beacon_t = 0.0

    print(f"[TX] worker started.  fixed_len={cfg.fixed_len}  modulation={cfg.modulation}  tx_gain={cfg.tx_gain} dB")
    try:
        while not stop_ev.is_set():
            now = time.time()
            buf = None
            try:
                buf = q.get_nowait()
            except queue.Empty:
                pass

            if buf is not None:
                buf        = fit_to_fixed_len(buf, cfg.fixed_len).astype(np.complex64)
                last_sig   = buf.copy()
                last_beacon_t = now
            elif cfg.beacon_period_s > 0 and last_sig is not None and (now - last_beacon_t) >= cfg.beacon_period_s:
                buf = last_sig
                last_beacon_t = now
            else:
                buf = idle

            # Hardware IQ-swap compensation (PlutoSDR TX DAC inverts IQ)
            tx_data = (np.conj(buf) * 4096.0).astype(np.complex64)
            sdr.tx(tx_data)

            if cfg.send_interval_s > 0:
                time.sleep(cfg.send_interval_s)
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        print("[TX] worker stopped.")


# =============================================================================
# Producer: packet mode
# =============================================================================
def producer_packets(
    stop_ev: threading.Event,
    q: "queue.Queue[np.ndarray]",
    cfg: TxConfig,
    infile: str,
    payload_str: str,
    payload_len: int,
    chunk_bytes: int,
    ref_seed: int = 0,
    ref_len: int = 0,
):
    stf        = create_schmidl_cox_stf(cfg.stf_repeats)
    ltf        = create_ltf(cfg.ltf_symbols)
    const_tbl  = make_constellation(cfg.modulation)
    bps        = cfg.bps

    if ref_len > 0:
        rng  = np.random.default_rng(ref_seed)
        data = rng.integers(0, 256, size=ref_len, dtype=np.uint8).tobytes()
        print(f"[TX] BER-mode reference payload: {len(data)} bytes  seed={ref_seed}")
    elif infile:
        with open(infile, "rb") as fh:
            data = fh.read()
    elif payload_str:
        data = payload_str.encode("utf-8")
    else:
        rng  = np.random.default_rng(54321)
        data = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()

    chunks = [data[i: i + chunk_bytes] for i in range(0, len(data), chunk_bytes)] if chunk_bytes > 0 else [data]
    total  = len(chunks)

    while not stop_ev.is_set():
        for seq, ch in enumerate(chunks):
            if stop_ev.is_set():
                break
            fb  = build_packet_bytes(seq, total, ch)
            sig = bytes_to_ofdm_samples(
                fb, cfg.repeat, stf, ltf, const_tbl, bps,
                cfg.fs, cfg.tone_duration_ms, cfg.tone_freq_hz,
                cfg.gap_short, cfg.gap_long, cfg.tx_scale,
            )
            sig = fit_to_fixed_len(sig, cfg.fixed_len)
            q.put(sig, block=True)
            bpos      = N_DATA * bps
            num_syms  = int(np.ceil((len(fb) * 8 + (cfg.repeat - 1)) / bpos))
            print(f"[TX] enq seq={seq}/{total-1}  payload={len(ch)}B  "
                  f"frame={len(fb)}B  ofdm_syms={num_syms}  mod={cfg.modulation}")

    print("[TX] producer done – idling.")
    while not stop_ev.is_set():
        time.sleep(0.2)


# =============================================================================
# Producer: sweep mode (unchanged)
# =============================================================================
def producer_sweep(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: TxConfig):
    print(f"[TX] Sweep mode  freqs={cfg.sweep_freqs}  step={cfg.sweep_time_s}s")
    t        = np.arange(cfg.fixed_len, dtype=np.float32) / float(cfg.fs)
    freq_idx = 0
    start_t  = time.time()
    print(f"[TX] → {cfg.sweep_freqs[freq_idx]} Hz")
    while not stop_ev.is_set():
        if time.time() - start_t >= cfg.sweep_time_s:
            freq_idx = (freq_idx + 1) % len(cfg.sweep_freqs)
            start_t  = time.time()
            print(f"[TX] → {cfg.sweep_freqs[freq_idx]} Hz")
        tone = np.exp(2j * np.pi * cfg.sweep_freqs[freq_idx] * t).astype(np.complex64)
        tone = tone / (np.max(np.abs(tone)) + 1e-9) * float(cfg.tx_scale)
        try:
            q.put(tone, timeout=0.1)
        except queue.Full:
            pass
    print("[TX] producer_sweep done.")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser("Step6 PHY TX – multi-modulation streaming (non-cyclic)")
    ap.add_argument("--uri",     required=True)
    ap.add_argument("--fc",      type=float, required=True)
    ap.add_argument("--fs",      type=float, required=True)
    ap.add_argument("--tx_gain", type=float, default=-20.0)
    ap.add_argument("--tx_bw",   type=float, default=0.0)

    ap.add_argument("--modulation", default="qpsk",
                    choices=list(MOD_BPS.keys()), help="Modulation scheme")

    ap.add_argument("--fixed_len",     type=int,   default=65536)
    ap.add_argument("--send_interval", type=float, default=0.0)
    ap.add_argument("--queue_depth",   type=int,   default=8)

    ap.add_argument("--repeat",      type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--tone_duration_ms", type=float, default=0.0)
    ap.add_argument("--tone_freq_hz",     type=float, default=100e3)
    ap.add_argument("--gap_short",  type=int,   default=1000)
    ap.add_argument("--gap_long",   type=int,   default=3000)
    ap.add_argument("--tx_scale",   type=float, default=0.8)
    ap.add_argument("--idle_amp",   type=float, default=0.0)

    ap.add_argument("--payload",     type=str, default="")
    ap.add_argument("--payload_len", type=int, default=64)
    ap.add_argument("--infile",      type=str, default="")
    ap.add_argument("--chunk_bytes", type=int, default=512)
    ap.add_argument("--ref_seed",    type=int, default=0)
    ap.add_argument("--ref_len",     type=int, default=0,
                    help="BER mode: transmit known PRNG payload of this length in bytes")
    ap.add_argument("--beacon_period", type=float, default=0.2)

    ap.add_argument("--mode", default="packet", choices=["packet", "sweep"])
    ap.add_argument("--sweep_freqs", default="-1e6,-5e5,0,5e5,1e6")
    ap.add_argument("--sweep_time",  type=float, default=2.0)

    ap.add_argument("--out_root", default="rf_stream_tx_runs")
    args = ap.parse_args()

    sweep_freqs = [float(x.strip()) for x in args.sweep_freqs.split(",") if x.strip()]
    bps         = MOD_BPS[args.modulation]
    run_id      = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir     = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg = TxConfig(
        uri=args.uri, fc=args.fc, fs=args.fs,
        tx_gain=args.tx_gain,
        tx_bw=(args.tx_bw if args.tx_bw > 0 else args.fs * 1.2),
        fixed_len=args.fixed_len,
        repeat=args.repeat,
        stf_repeats=args.stf_repeats,
        ltf_symbols=args.ltf_symbols,
        modulation=args.modulation,
        bps=bps,
        tone_duration_ms=args.tone_duration_ms,
        tone_freq_hz=args.tone_freq_hz,
        gap_short=args.gap_short,
        gap_long=args.gap_long,
        tx_scale=args.tx_scale,
        idle_amp=args.idle_amp,
        send_interval_s=args.send_interval,
        beacon_period_s=args.beacon_period,
        mode=args.mode,
        sweep_freqs=sweep_freqs,
        sweep_time_s=args.sweep_time,
        out_root=args.out_root,
        run_id=run_id,
    )

    # Save run metadata
    meta = {
        **asdict(cfg),
        "ref_seed": args.ref_seed,
        "ref_len":  args.ref_len,
        "chunk_bytes": args.chunk_bytes,
        "start_time": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "tx_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n" + "=" * 70)
    print("Streaming TX  (Step6 PHY)")
    print("=" * 70)
    print(f"  URI={cfg.uri}  fc={cfg.fc/1e6:.3f}MHz  fs={cfg.fs/1e6:.1f}Msps")
    print(f"  tx_gain={cfg.tx_gain}dB  modulation={cfg.modulation}  bps={bps}")
    print(f"  run_id={run_id}  out_dir={out_dir}")
    print("=" * 70)

    q       = queue.Queue(maxsize=int(args.queue_depth))
    stop_ev = threading.Event()

    t_tx = threading.Thread(target=tx_worker, args=(stop_ev, q, cfg), daemon=True)
    if cfg.mode == "packet":
        t_prod = threading.Thread(
            target=producer_packets,
            args=(stop_ev, q, cfg, args.infile, args.payload,
                  args.payload_len, args.chunk_bytes, args.ref_seed, args.ref_len),
            daemon=True,
        )
    else:
        t_prod = threading.Thread(target=producer_sweep, args=(stop_ev, q, cfg), daemon=True)

    t_tx.start()
    t_prod.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        time.sleep(0.5)
        print("[TX] exit.")


if __name__ == "__main__":
    main()

"""
# ── Quick-start commands ──────────────────────────────────────────────────────

# Default QPSK (same as step5, backward compatible):
python3 rf_stream_tx_step6phy.py \
  --uri ip:192.168.3.2 --fc 2.3e9 --fs 3e6 --tx_gain 0 \
  --payload "step6 hello @2.3G 3Msps" --idle_amp 0.002 --beacon_period 0.2

# BPSK test:
python3 rf_stream_tx_step6phy.py \
  --uri ip:192.168.3.2 --fc 2.3e9 --fs 3e6 --tx_gain -10 \
  --modulation bpsk --ref_seed 42 --ref_len 512

# QAM16 test:
python3 rf_stream_tx_step6phy.py \
  --uri ip:192.168.3.2 --fc 2.3e9 --fs 3e6 --tx_gain -5 \
  --modulation qam16 --ref_seed 42 --ref_len 512

# BER sweep (run once per TX gain level, change --tx_gain each time):
# Step 1 – high SNR:  --tx_gain 0
# Step 2 – mid  SNR:  --tx_gain -10
# Step 3 – low  SNR:  --tx_gain -20
# Then use analyze_rf_stream_captures.py --ber_dirs ... to plot BER curves
"""
