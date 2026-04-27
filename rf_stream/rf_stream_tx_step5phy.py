#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_tx_step5phy.py

PlutoSDR streaming TX WITHOUT tx_cyclic_buffer.
- Thread A (producer): build fixed-length OFDM packets, push into queue
- Thread B (tx_worker): continuously call sdr.tx(buf_fixed_len)
    - if queue has packet: send it
    - else: send idle buffer (near-silent) to keep DMA alive

PHY: Step5-style STF (Schmidl-Cox, period N/2) + multi-LTF + QPSK OFDM
Packet bytes: MAGIC|SEQ|LEN|PAYLOAD|CRC32
"""

import argparse
import queue
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =========================
# Step5 PHY params
# =========================
N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array(
    [k for k in range(-26, 27) if (k != 0 and k not in set(PILOT_SUBCARRIERS))],
    dtype=int,
)
N_DATA = len(DATA_SUBCARRIERS)       # 48
BITS_PER_SYM = N_DATA * 2            # QPSK
MAGIC = b"AIS1"

def sc_to_bin(k: int) -> int:
    return (k + N_FFT) % N_FFT

DATA_BINS = np.array([sc_to_bin(int(k)) for k in DATA_SUBCARRIERS], dtype=int)
PILOT_BINS = np.array([sc_to_bin(int(k)) for k in PILOT_SUBCARRIERS], dtype=int)


# =========================
# Helpers
# =========================
def bits_from_bytes(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='big').astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits, bitorder='big').tobytes()

def scramble_bits(bits: np.ndarray, seed: int = 0x7F) -> np.ndarray:
    state = seed
    out = np.zeros_like(bits)
    for i in range(len(bits)):
        b7 = (state >> 6) & 1
        b4 = (state >> 3) & 1
        feedback = b7 ^ b4
        out[i] = bits[i] ^ b7
        state = ((state << 1) | feedback) & 0x7F
    return out

def qpsk_map(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(np.uint8)
    n = (len(bits) // 2)
    bits = bits[: 2*n]
    b0 = bits[0::2]
    b1 = bits[1::2]
    # Gray mapping: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
    re = np.where((b0 == 0) & (b1 == 0),  1.0,
         np.where((b0 == 0) & (b1 == 1), -1.0,
         np.where((b0 == 1) & (b1 == 1), -1.0,  1.0)))
    im = np.where((b0 == 0) & (b1 == 0),  1.0,
         np.where((b0 == 0) & (b1 == 1),  1.0,
         np.where((b0 == 1) & (b1 == 1), -1.0, -1.0)))
    syms = (re + 1j*im).astype(np.complex64) / np.sqrt(2.0)
    return syms

def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[sc_to_bin(int(k))] = bpsk[i] + 0j
    # IMPORTANT: match RX fftshift convention
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x.astype(np.complex64), num_repeats)
    stf_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_cp

def create_ltf(num_symbols: int = 4) -> np.ndarray:
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = (1.0 if (i % 2 == 0) else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, num_symbols).astype(np.complex64)
    return ltf

def create_ofdm_symbol(data_syms: np.ndarray, pilot_vals: np.ndarray, sym_idx: int) -> np.ndarray:
    X = np.zeros(N_FFT, dtype=np.complex64)
    nfill = min(len(data_syms), len(DATA_BINS))
    X[DATA_BINS[:nfill]] = data_syms[:nfill]

    pilot_sign = 1 if (sym_idx % 2 == 0) else -1
    X[PILOT_BINS] = pilot_sign * pilot_vals

    # ifftshift so RX fftshift(fft()) recovers X directly (same convention as STF/LTF)
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    td = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    return td

def build_packet_bytes(seq: int, total: int, payload: bytes) -> bytes:
    # MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    hdr = MAGIC + int(seq).to_bytes(2, "little") + int(total).to_bytes(2, "little") + int(len(payload)).to_bytes(2, "little")
    body = hdr + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + int(crc).to_bytes(4, "little")

def bytes_to_ofdm_samples(frame_bytes: bytes, repeat: int, stf: np.ndarray, ltf: np.ndarray,
                          fs: float, tone_duration_ms: float, tone_freq_hz: float,
                          gap_short: int, gap_long: int,
                          tx_scale: float) -> np.ndarray:
    bits = bits_from_bytes(frame_bytes)
    bits = scramble_bits(bits)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    num_syms = int(np.ceil(len(bits) / BITS_PER_SYM))
    need_bits = num_syms * BITS_PER_SYM
    if len(bits) < need_bits:
        bits = np.pad(bits, (0, need_bits - len(bits)))

    pilot_vals = np.array([1, 1, 1, -1], dtype=np.complex64)
    ofdm = []
    for si in range(num_syms):
        sb = bits[si*BITS_PER_SYM : (si+1)*BITS_PER_SYM]
        ds = qpsk_map(sb)
        ofdm.append(create_ofdm_symbol(ds, pilot_vals, si))
    ofdm = np.concatenate(ofdm).astype(np.complex64)

    # Optional tone (can set duration 0)
    tone_samps = int(tone_duration_ms * fs / 1000.0)
    if tone_samps > 0:
        t = np.arange(tone_samps, dtype=np.float32) / float(fs)
        tone = (np.exp(2j*np.pi*float(tone_freq_hz)*t).astype(np.complex64) * 0.5)
    else:
        tone = np.zeros(0, dtype=np.complex64)

    gS = np.zeros(int(gap_short), dtype=np.complex64)
    gL = np.zeros(int(gap_long), dtype=np.complex64)

    sig = np.concatenate([gL, tone, gS, stf, ltf, ofdm, gL]).astype(np.complex64)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * float(tx_scale)
    return sig

def fit_to_fixed_len(sig: np.ndarray, fixed_len: int) -> np.ndarray:
    if len(sig) == fixed_len:
        return sig
    if len(sig) > fixed_len:
        return sig[:fixed_len].copy()
    # pad zeros
    out = np.zeros(fixed_len, dtype=np.complex64)
    out[:len(sig)] = sig
    return out


@dataclass
class TxConfig:
    uri: str
    fc: float
    fs: float
    tx_gain: float
    tx_bw: float
    fixed_len: int               # IMPORTANT: constant length for every sdr.tx()
    repeat: int
    stf_repeats: int
    ltf_symbols: int
    tone_duration_ms: float
    tone_freq_hz: float
    gap_short: int
    gap_long: int
    tx_scale: float
    idle_amp: float              # idle amplitude (0 -> strict zeros)
    send_interval_s: float       # pacing per buffer
    beacon_period_s: float   # e.g. 0.2
    mode: str
    sweep_freqs: list
    sweep_time_s: float


def make_idle_buffer(cfg: TxConfig) -> np.ndarray:
    if cfg.idle_amp <= 0:
        return np.zeros(cfg.fixed_len, dtype=np.complex64)
    # very small complex noise to keep DAC active but near-silent
    rng = np.random.default_rng(0)
    n = (rng.standard_normal(cfg.fixed_len) + 1j*rng.standard_normal(cfg.fixed_len)).astype(np.complex64)
    n = n / (np.max(np.abs(n)) + 1e-9) * cfg.idle_amp
    return n.astype(np.complex64)


def tx_worker(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: TxConfig):
    import adi
    sdr = adi.Pluto(uri=cfg.uri)
    sdr.sample_rate = int(cfg.fs)
    sdr.tx_lo = int(cfg.fc)
    sdr.tx_rf_bandwidth = int(cfg.tx_bw)
    sdr.tx_hardwaregain_chan0 = float(cfg.tx_gain)
    sdr.tx_enabled_channels = [0]

    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False

    idle = make_idle_buffer(cfg)
    idle = fit_to_fixed_len(idle, cfg.fixed_len).astype(np.complex64)

    last_sig: Optional[np.ndarray] = None
    last_beacon_t = 0.0

    print("[TX] worker started. Continuous non-cyclic push. fixed_len =", cfg.fixed_len)

    try:
        while not stop_ev.is_set():
            now = time.time()

            # 1) try get new buffer
            buf = None
            try:
                buf = q.get_nowait()
            except queue.Empty:
                buf = None

            if buf is not None:
                # got new packet
                buf = fit_to_fixed_len(buf, cfg.fixed_len).astype(np.complex64)
                last_sig = buf.copy()
                last_beacon_t = now

            else:
                # no new packet: maybe beacon
                if (cfg.beacon_period_s > 0) and (last_sig is not None) and ((now - last_beacon_t) >= cfg.beacon_period_s):
                    buf = last_sig
                    last_beacon_t = now
                else:
                    buf = idle

            # FIX: Conjugate to flip spectrum to match RX.
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


def producer_packets(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: TxConfig,
                     infile: str, payload_str: str, payload_len: int, chunk_bytes: int, ref_seed: int = 0, ref_len: int = 0):
    stf = create_schmidl_cox_stf(cfg.stf_repeats)
    ltf = create_ltf(cfg.ltf_symbols)

    # build payload stream
    if ref_len > 0:
        rng = np.random.default_rng(ref_seed)
        data = rng.integers(0, 256, size=ref_len, dtype=np.uint8).tobytes()
        print(f"[TX] Generating reference payload: {len(data)} bytes (seed={ref_seed})")
    elif infile:
        with open(infile, "rb") as f:
            data = f.read()
    elif payload_str:
        data = payload_str.encode("utf-8")
    else:
        rng = np.random.default_rng(54321)
        data = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()

    # fragment to chunks so each packet has bounded payload size
    if chunk_bytes <= 0:
        chunks = [data]
    else:
        chunks = [data[i:i+chunk_bytes] for i in range(0, len(data), chunk_bytes)]

    total = len(chunks)
    while not stop_ev.is_set():
        seq = 0
        for ch in chunks:
            if stop_ev.is_set():
                break
            frame_bytes = build_packet_bytes(seq, total, ch)
            sig = bytes_to_ofdm_samples(
                frame_bytes, cfg.repeat, stf, ltf,
                fs=cfg.fs,
                tone_duration_ms=cfg.tone_duration_ms,
                tone_freq_hz=cfg.tone_freq_hz,
                gap_short=cfg.gap_short,
                gap_long=cfg.gap_long,
                tx_scale=cfg.tx_scale
            )
            sig = fit_to_fixed_len(sig, cfg.fixed_len)
            q.put(sig, block=True)
            print(f"[TX] enqueued packet seq={seq} payload={len(ch)}B frame_bytes={len(frame_bytes)} sig_len={len(sig)}")
            seq += 1

    print("[TX] producer done. (TX will continue idling)")
    # keep alive until stop
    while not stop_ev.is_set():
        time.sleep(0.2)


def producer_sweep(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: TxConfig):
    print(f"[TX] Sweep mode started. Freqs: {cfg.sweep_freqs} Hz, Time/freq: {cfg.sweep_time_s} s")
    
    tone_samps = cfg.fixed_len
    t = np.arange(tone_samps, dtype=np.float32) / float(cfg.fs)
    
    freq_idx = 0
    start_t = time.time()
    print(f"[TX] Sweeping to frequency: {cfg.sweep_freqs[freq_idx]} Hz")
    
    while not stop_ev.is_set():
        now = time.time()
        elapsed = now - start_t
        if elapsed >= cfg.sweep_time_s:
            freq_idx = (freq_idx + 1) % len(cfg.sweep_freqs)
            start_t = now
            print(f"[TX] Sweeping to frequency: {cfg.sweep_freqs[freq_idx]} Hz")
            
        f = cfg.sweep_freqs[freq_idx]
        tone = (np.exp(2j * np.pi * f * t).astype(np.complex64))
        tone = tone / (np.max(np.abs(tone)) + 1e-9) * float(cfg.tx_scale)
        
        try:
            q.put(tone, timeout=0.1)
        except queue.Full:
            pass
            
    print("[TX] producer_sweep done.")


def main():
    ap = argparse.ArgumentParser("Streaming Step5 PHY TX (no cyclic)")
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, required=True)
    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--tx_gain", type=float, default=-20.0)
    ap.add_argument("--tx_bw", type=float, default=0.0)

    ap.add_argument("--fixed_len", type=int, default=65536, help="Fixed TX buffer length (samples) ALWAYS constant")
    ap.add_argument("--send_interval", type=float, default=0.0, help="Optional sleep between tx() pushes")
    ap.add_argument("--queue_depth", type=int, default=8)

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--tone_duration_ms", type=float, default=0.0)
    ap.add_argument("--tone_freq_hz", type=float, default=100e3)
    ap.add_argument("--gap_short", type=int, default=1000)
    ap.add_argument("--gap_long", type=int, default=3000)
    ap.add_argument("--tx_scale", type=float, default=0.8)
    ap.add_argument("--idle_amp", type=float, default=0.0)

    ap.add_argument("--payload", type=str, default="")
    ap.add_argument("--payload_len", type=int, default=64)
    ap.add_argument("--infile", type=str, default="")
    ap.add_argument("--chunk_bytes", type=int, default=512, help="fragment file/payload into chunks")
    ap.add_argument("--ref_seed", type=int, default=0)
    ap.add_argument("--ref_len", type=int, default=0)
    ap.add_argument("--beacon_period", type=float, default=0.2, help="If no new packet, resend last packet every N seconds (0 disables)")

    ap.add_argument("--mode", type=str, default="packet", choices=["packet", "sweep"], help="TX mode: packet or sweep")
    ap.add_argument("--sweep_freqs", type=str, default="-1e6,-5e5,0,5e5,1e6", help="Comma separated frequencies for sweep mode")
    ap.add_argument("--sweep_time", type=float, default=2.0, help="Duration per frequency in sweep mode")

    args = ap.parse_args()

    sweep_freqs = [float(x.strip()) for x in args.sweep_freqs.split(",") if x.strip()]

    cfg = TxConfig(
        uri=args.uri,
        fc=args.fc,
        fs=args.fs,
        tx_gain=args.tx_gain,
        tx_bw=(args.tx_bw if args.tx_bw > 0 else args.fs*1.2),
        fixed_len=args.fixed_len,
        repeat=args.repeat,
        stf_repeats=args.stf_repeats,
        ltf_symbols=args.ltf_symbols,
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
    )

    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=int(args.queue_depth))
    stop_ev = threading.Event()

    t_tx = threading.Thread(target=tx_worker, args=(stop_ev, q, cfg), daemon=True)
    
    if cfg.mode == "packet":
        t_prod = threading.Thread(
            target=producer_packets,
            args=(stop_ev, q, cfg, args.infile, args.payload, args.payload_len, args.chunk_bytes, args.ref_seed, args.ref_len),
            daemon=True
        )
    else:
        t_prod = threading.Thread(
            target=producer_sweep,
            args=(stop_ev, q, cfg),
            daemon=True
        )

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
* 协议：Step5 PHY（Schmidl-Cox STF + 多符号 LTF + QPSK OFDM）
* 负载：默认发字符串；也支持 --infile 发文件（会自动分片成多个包）
* 帧格式（字节）：
    MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    CRC 覆盖 MAGIC..PAYLOAD

Added a new producer_sweep thread. When running in --mode sweep, the TX node will iterate through a list of frequencies (-1e6, -5e5, 0, 5e5, 1e6 Hz by default) and continuously transmit a single complex tone for exactly 2 seconds per frequency.

python3 rf_stream_tx_step5phy.py --uri ip:192.168.3.2 --fc 2.3e9 --fs 3e6 --tx_gain 0 --fixed_len 65536 --mode sweep --sweep_freqs "-1e6, -5e5, 0, 5e5, 1e6" --sweep_time 2.0

cmpe@cmpe-jetson:~$ source .venv/bin/activate
python3 rf_stream_tx_step5phy.py \
  --uri ip:192.168.3.2 \
  --fc 2.3e9 --fs 3e6 \
  --tx_gain 0 \
  --fixed_len 65536 \
  --repeat 1 \
  --tone_duration_ms 0 \
  --payload "step5 streaming hello @2.3G 3Msps" \
  --chunk_bytes 256 \
  --idle_amp 0.002 \
  --beacon_period 0.2

如果你想“保持 DMA 活跃但极低泄漏”，用严格静默：
--idle_amp 0


"""