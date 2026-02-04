#!/usr/bin/env python3
"""
RF Link Step5 RX - MultiPacket + BER + Full Debug Figures

Key robustness upgrade:
  - STF detection uses NCC (normalized cross-correlation)
  - Candidate peaks -> LTF_score verification -> choose best peak
  - Works reliably under cyclic / multi-packet superframe

Demod:
  - channel estimate from LTF (2 repeats averaged)
  - per-symbol pilot-based CPE tracking (PI loop)
  - QPSK hard demap
  - optional repetition majority vote
  - parse packet bytes by MAGIC scan + CRC32
  - store by seq, reassemble full payload when all received

BER:
  - if --ref_seed/--ref_len provided, compare recovered payload to reference random payload

Debug figure for every capture:
  - time |rx| (first 80k)
  - tone FFT + CFO
  - STF NCC corr curve (peaks + chosen)
  - LTF: |Y0| vs |Y1|, |H|, angle(H)
  - pilot loop traces (phase_err, phase_acc, freq_acc)
  - pilot power per OFDM symbol
  - constellation (data bins): pre/post CPE
  - EVM per symbol
  - parse status text (MAGIC/SEQ/TOTAL/CRC)
"""

import argparse
import os
import zlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import adi

# OFDM params (must match TX)
N_FFT = 64
N_CP  = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array([k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)
BITS_PER_OFDM_SYM = 2 * N_DATA

MAGIC = b"AIS1"


def pilot_pattern(sym_idx: int) -> np.ndarray:
    base = np.array([1, 1, 1, -1], dtype=np.complex64)
    if sym_idx % 2 == 1:
        base = -base
    return base

def create_stf_ref():
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        #X[(k + N_FFT) % N_FFT] = bpsk[i]
        X[(k + N_FFT//2) % N_FFT] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x, 2).astype(np.complex64)  # 2N
    stf_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)  # CP + 2N
    return stf_cp

def create_ltf_ref_freq_fftshift():
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        #X[(k + N_FFT) % N_FFT] = 1.0 if i % 2 == 0 else -1.0
        X[(k + N_FFT//2) % N_FFT] = 1.0 if i % 2 == 0 else -1.0
    return np.fft.fftshift(X).astype(np.complex64)

def apply_cfo(x, cfo, fs):
    n = np.arange(len(x), dtype=np.float32)
    return (x * np.exp(-1j * 2*np.pi*cfo*n/fs)).astype(np.complex64)

def estimate_cfo_from_tone(rx, fs, expected=100e3, tone_win=32768):
    N = min(tone_win, len(rx))
    if N < 2048:
        return 0.0, None
    x = rx[:N]
    X = np.fft.fft(x)
    mag = np.abs(X) + 1e-12
    mag[0] = 0
    guard = 30
    mag[:guard] = 0
    mag[-guard:] = 0
    k = int(np.argmax(mag))
    freqs = np.fft.fftfreq(N, d=1/fs)
    detected = freqs[k]
    cfo = detected - expected
    db = 20*np.log10((mag + 1e-12) / (np.max(mag) + 1e-12))
    return float(cfo), (freqs, db, detected, float(db[k]))

def ncc_corr(rx, ref, search_len, step=1):
    L = len(ref)
    search_len = min(search_len, len(rx)-L)
    if search_len <= 0:
        return np.array([]), np.array([]), -1, 0.0

    idxs = np.arange(0, search_len, step, dtype=int)
    corr = np.zeros(len(idxs), dtype=np.float32)
    refE = np.sqrt(np.sum(np.abs(ref)**2)) + 1e-12

    for ii, k in enumerate(idxs):
        seg = rx[k:k+L]
        segE = np.sqrt(np.sum(np.abs(seg)**2)) + 1e-12
        corr[ii] = np.abs(np.vdot(ref, seg)) / (refE * segE)

    pk = int(np.argmax(corr))
    return corr, idxs, int(idxs[pk]), float(corr[pk])

def topk_peaks(corr, idxs, k=8, guard=1200):
    order = np.argsort(corr)[::-1]
    chosen = []
    for oi in order:
        s = int(idxs[oi])
        ok = True
        for ss, _ in chosen:
            if abs(s - ss) < guard:
                ok = False
                break
        if ok:
            chosen.append((s, float(corr[oi])))
            if len(chosen) >= k:
                break
    chosen.sort(key=lambda x: x[0])
    return chosen

def extract_symbol_fftshift(rx, start):
    if start + SYMBOL_LEN > len(rx):
        return None
    sym = rx[start+N_CP : start+SYMBOL_LEN]
    Y = np.fft.fftshift(np.fft.fft(sym) / np.sqrt(N_FFT))
    return Y.astype(np.complex64)

def cfo_from_stf(rx, fs, stf_idx):
    s = stf_idx + N_CP
    if s + 2*N_FFT > len(rx):
        return None
    x1 = rx[s:s+N_FFT]
    x2 = rx[s+N_FFT:s+2*N_FFT]
    v = np.vdot(x1, x2)
    ang = np.angle(v)
    return float(ang / (2*np.pi*(N_FFT/fs)))

def ltf_score(Y, Xref):
    mask = (np.abs(Xref) > 0.1)
    if np.sum(mask) < 10:
        return 0.0
    num = np.abs(np.sum(Y[mask] * np.conj(Xref[mask])))
    den = np.sum(np.abs(Y[mask])) + 1e-12
    return float(num / den)

def qpsk_demap_gray(symbols):
    # inverse of TX mapping:
    # 00:+ +, 01:- +, 11:- -, 10:+ -
    bits = np.zeros(2*len(symbols), dtype=np.uint8)
    for i, s in enumerate(symbols):
        re = np.real(s) >= 0
        im = np.imag(s) >= 0
        if re and im:       b0,b1 = 0,0
        elif (not re) and im: b0,b1 = 0,1
        elif (not re) and (not im): b0,b1 = 1,1
        else:               b0,b1 = 1,0
        bits[2*i] = b0
        bits[2*i+1] = b1
    return bits

def majority_vote(bits, repeat):
    if repeat <= 1:
        return bits
    L = (len(bits)//repeat)*repeat
    bits = bits[:L].reshape(-1, repeat)
    return (np.sum(bits, axis=1) >= (repeat/2)).astype(np.uint8)

def scan_magic_and_parse(b: bytes):
    """
    Scan for MAGIC and parse a single packet starting at that MAGIC.
    Returns dict or None.
    """
    # Need at least: MAGIC(4)+SEQ(2)+TOTAL(2)+PLEN(2)+CRC(4) = 14 bytes
    if len(b) < 14:
        return None

    start = b.find(MAGIC)
    if start < 0:
        return None
    if start + 14 > len(b):
        return None

    seq   = int.from_bytes(b[start+4:start+6], "little")
    total = int.from_bytes(b[start+6:start+8], "little")
    plen  = int.from_bytes(b[start+8:start+10], "little")
    need = 10 + plen + 4
    if start + need > len(b):
        return None

    payload = b[start+10:start+10+plen]
    crc_rx = int.from_bytes(b[start+10+plen:start+10+plen+4], "little")
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF

    ok = (crc_rx == crc_calc)
    return {
        "start": start,
        "seq": seq,
        "total": total,
        "plen": plen,
        "payload": payload,
        "crc_rx": crc_rx,
        "crc_calc": crc_calc,
        "ok": ok
    }

def evm_db(rx_syms, ideal_syms):
    # EVM = rms(error)/rms(ideal)
    if len(rx_syms) == 0:
        return np.array([], dtype=np.float32)
    err = rx_syms - ideal_syms
    num = np.mean(np.abs(err)**2, axis=1)
    den = np.mean(np.abs(ideal_syms)**2, axis=1) + 1e-12
    evm = np.sqrt(num/den)
    return (20*np.log10(evm + 1e-12)).astype(np.float32)

def make_reference_payload(seed: int, length: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()

def ber_bits(a: bytes, b: bytes) -> tuple[int,int,float]:
    # compare up to min length
    n = min(len(a), len(b))
    if n == 0:
        return 0, 0, 0.0
    aa = np.unpackbits(np.frombuffer(a[:n], dtype=np.uint8))
    bb = np.unpackbits(np.frombuffer(b[:n], dtype=np.uint8))
    err = int(np.sum(aa != bb))
    tot = int(len(aa))
    return err, tot, float(err/(tot+1e-12))


def save_debug_figure(path, rx, fs, tone_info, corr, idxs, peaks, chosen, ltf_info, demod_info, parse_info, title):
    """
    One capture -> one figure (OK/FAIL). Includes all evidence.
    """
    stf_idx = chosen.get("stf_idx", -1)
    stf_ncc = chosen.get("stf_ncc", 0.0)

    fig = plt.figure(figsize=(18, 10))

    # 1) time |rx|
    ax = fig.add_subplot(3,4,1)
    N = min(len(rx), 80000)
    ax.plot(np.abs(rx[:N]))
    if stf_idx >= 0 and stf_idx < N:
        ax.axvline(stf_idx, color="r", linestyle="--", label="chosen STF")
    ax.set_title("Time |rx| (first 80k)")
    ax.grid(True)
    ax.legend(loc="upper right")

    # 2) tone FFT
    ax = fig.add_subplot(3,4,2)
    if tone_info is not None:
        freqs, db, detected, pk_db = tone_info
        ax.plot(freqs/1e3, db)
        ax.set_title(f"Tone FFT (dB)\ndetected={detected/1e3:.1f}kHz peak={pk_db:.1f}dB")
        ax.grid(True)
    else:
        ax.axis("off")

    # 3) STF NCC corr
    ax = fig.add_subplot(3,4,3)
    if len(corr) > 0:
        ax.plot(idxs, corr)
        for s,v in peaks:
            ax.axvline(s, color="k", alpha=0.12)
        if stf_idx >= 0:
            ax.axvline(stf_idx, color="r", linestyle="--", label=f"chosen {stf_ncc:.3f}")
        ax.set_title("STF NCC corr")
        ax.grid(True)
        ax.legend()
    else:
        ax.axis("off")

    # 4) LTF mags Y0/Y1
    ax = fig.add_subplot(3,4,4)
    if ltf_info.get("Y0") is not None:
        ax.plot(np.abs(ltf_info["Y0"]), label="|Y0|")
        ax.plot(np.abs(ltf_info["Y1"]), label="|Y1|", alpha=0.8)
        ax.set_title(f"LTF FFT mags (2 repeats)\nscore={ltf_info.get('score',0):.3f}")
        ax.grid(True)
        ax.legend()
    else:
        ax.axis("off")

    # 5) |H|
    ax = fig.add_subplot(3,4,5)
    if ltf_info.get("H") is not None:
        ax.plot(np.abs(ltf_info["H"]))
        ax.set_title("|H| (fftshift bins)")
        ax.grid(True)
    else:
        ax.axis("off")

    # 6) angle(H)
    ax = fig.add_subplot(3,4,6)
    if ltf_info.get("H") is not None:
        ax.plot(np.unwrap(np.angle(ltf_info["H"])))
        ax.set_title("angle(H) unwrap")
        ax.grid(True)
    else:
        ax.axis("off")

    # 7) pilot power
    ax = fig.add_subplot(3,4,7)
    if demod_info.get("pilot_pwr") is not None and len(demod_info["pilot_pwr"]) > 0:
        ax.plot(demod_info["pilot_pwr"])
        ax.set_title("Pilot power per OFDM symbol")
        ax.grid(True)
    else:
        ax.axis("off")

    # 8) pilot loop traces
    ax = fig.add_subplot(3,4,8)
    if demod_info.get("phase_err") is not None and len(demod_info["phase_err"]) > 0:
        ax.plot(demod_info["phase_err"], label="phase_err (unwrap)")
        ax.plot(demod_info["phase_acc"], label="phase_acc")
        ax.plot(demod_info["freq_acc"], label="freq_acc")
        ax.set_title("Pilot loop traces")
        ax.grid(True)
        ax.legend()
    else:
        ax.axis("off")

    # 9) constellation pre/post CPE
    ax = fig.add_subplot(3,4,9)
    if demod_info.get("data_pre") is not None and len(demod_info["data_pre"]) > 0:
        pre = demod_info["data_pre"]
        post = demod_info["data_post"]
        ax.scatter(np.real(pre), np.imag(pre), s=4, alpha=0.35, label="pre-CPE")
        ax.scatter(np.real(post), np.imag(post), s=4, alpha=0.35, label="post-CPE")
        ax.set_title("Constellation (data bins)")
        ax.grid(True)
        ax.axis("equal")
        ax.legend()
    else:
        ax.axis("off")

    # 10) EVM per symbol
    ax = fig.add_subplot(3,4,10)
    if demod_info.get("evm_db") is not None and len(demod_info["evm_db"]) > 0:
        ax.plot(demod_info["evm_db"])
        ax.set_title("EVM per OFDM symbol (dB, lower better)")
        ax.grid(True)
    else:
        ax.axis("off")

    # 11) angle histogram (QPSK clusters) from post-CPE data
    ax = fig.add_subplot(3,4,11)
    if demod_info.get("data_post") is not None and len(demod_info["data_post"]) > 0:
        ang = np.angle(demod_info["data_post"])
        ax.hist(ang, bins=60)
        ax.set_title("Angle hist (post-CPE) expect 4 clusters")
        ax.grid(True)
    else:
        ax.axis("off")

    # 12) text box
    ax = fig.add_subplot(3,4,12)
    ax.axis("off")
    txt = []
    txt.append(title)
    txt.append(f"CFO_tone={chosen.get('cfo_tone',0):+.1f} Hz  CFO_stf={chosen.get('cfo_stf',float('nan')):+.1f} Hz  CFO_use={chosen.get('cfo_use',0):+.1f} Hz")
    txt.append(f"STF_idx={stf_idx}  STF_ncc={stf_ncc:.3f}  LTF_score={ltf_info.get('score',0):.3f}")
    if parse_info is None:
        txt.append("PARSE: no MAGIC")
    else:
        txt.append(f"PARSE: start={parse_info['start']} ok={parse_info['ok']}")
        txt.append(f" seq={parse_info['seq']} total={parse_info['total']} plen={parse_info['plen']}")
        txt.append(f" crc_rx=0x{parse_info['crc_rx']:08X} crc_calc=0x{parse_info['crc_calc']:08X}")
    if demod_info.get("nsyms", 0) > 0:
        txt.append(f"Demod: ofdm_syms={demod_info['nsyms']}  bytes_out={demod_info.get('bytes_len',0)}")
    ax.text(0.02, 0.98, "\n".join(txt), va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=915e6)
    ap.add_argument("--fs", type=float, default=2e6)
    ap.add_argument("--bw", type=float, default=2.4e6)
    ap.add_argument("--rx_gain", type=float, default=40.0)
    ap.add_argument("--buf_size", type=int, default=131072)

    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--output_dir", type=str, default="rf_step5_rx_dbg")
    ap.add_argument("--outfile", type=str, default="recovered_payload.bin")

    ap.add_argument("--tone_freq", type=float, default=100e3)
    ap.add_argument("--tone_win", type=int, default=32768)

    ap.add_argument("--search_len", type=int, default=120000)
    ap.add_argument("--scan_step", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--guard", type=int, default=1200)

    ap.add_argument("--max_ofdm_syms", type=int, default=260, help="max OFDM symbols to demod after LTF")
    ap.add_argument("--kp", type=float, default=0.10)
    ap.add_argument("--ki", type=float, default=0.01)

    ap.add_argument("--ref_seed", type=int, default=0)
    ap.add_argument("--ref_len", type=int, default=0, help="if >0, generate reference random payload for BER")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ref_payload = b""
    if args.ref_len > 0:
        ref_payload = make_reference_payload(args.ref_seed, args.ref_len)
        print(f"[BER] Reference payload available: {len(ref_payload)} bytes (seed={args.ref_seed})")

    stf_ref = create_stf_ref()
    ltf_ref = create_ltf_ref_freq_fftshift()

    # indices
    #pilot_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS], dtype=int)
    #data_idx  = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS], dtype=int)
    pilot_idx = np.array([(k + N_FFT//2) % N_FFT for k in PILOT_SUBCARRIERS], dtype=int)
    data_idx  = np.array([(k + N_FFT//2) % N_FFT for k in DATA_SUBCARRIERS], dtype=int)

    print("RF Link Step5 RX - MultiPacket + BER + Full Debug")
    print(f"  uri={args.uri} fc={args.fc/1e6:.3f} MHz fs={args.fs/1e6:.3f} Msps bw={args.bw/1e6:.3f} MHz gain={args.rx_gain} dB")
    print(f"  repeat={args.repeat} buf={args.buf_size} tries={args.tries}")
    print(f"  STF: search_len={args.search_len} step={args.scan_step} topk={args.topk} guard={args.guard}")
    print("------------------------------------------------------------------------")

    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf_size)
    
    # QPSK constellation points (Gray mapping, normalized)
    R = np.array([ (1+1j), (-1+1j), (-1-1j), (1-1j) ], dtype=np.complex64) / np.sqrt(2)

    # flush
    for _ in range(3):
        _ = sdr.rx()

    got = {}         # seq -> payload bytes
    total_expected = None

    for ti in range(1, args.tries+1):
        # capture with timeout resilience
        try:
            rx_raw = sdr.rx()
        except TimeoutError:
            # save minimal fail info
            print(f"[cap {ti:02d}] RX TIMEOUT (iio dequeue). Try increasing USB stability/buf_size.")
            continue

        rx = (rx_raw.astype(np.complex64) / (2**14)).astype(np.complex64)
        rx = rx - np.mean(rx)

        # tone CFO
        cfo_tone, tone_info = estimate_cfo_from_tone(rx, args.fs, expected=args.tone_freq, tone_win=args.tone_win)
        rx1 = apply_cfo(rx, cfo_tone, args.fs)

        # STF NCC corr
        corr, idxs, best_idx, best_val = ncc_corr(rx1, stf_ref, search_len=args.search_len, step=max(1, args.scan_step))
        peaks = topk_peaks(corr, idxs, k=args.topk, guard=args.guard) if len(corr) else []

        # For each peak: compute LTF score and choose best
        best_choice = None
        best_score = -1.0

        ltf_best = {"Y0": None, "Y1": None, "H": None, "score": 0.0}

        for (stf_idx, stf_ncc) in peaks:
            cfo_stf = cfo_from_stf(rx, args.fs, stf_idx)
            if cfo_stf is None:
                continue
            cfo_use = 0.5*cfo_tone + 0.5*cfo_stf
            rxc = apply_cfo(rx, cfo_use, args.fs)

            ltf_start = stf_idx + len(stf_ref)
            Y0 = extract_symbol_fftshift(rxc, ltf_start)
            Y1 = extract_symbol_fftshift(rxc, ltf_start + SYMBOL_LEN)
            if (Y0 is None) or (Y1 is None):
                continue

            Y = (Y0 + Y1) / 2.0
            score = ltf_score(Y, ltf_ref)

            if score > best_score:
                best_score = score
                best_choice = {
                    "stf_idx": stf_idx,
                    "stf_ncc": stf_ncc,
                    "cfo_tone": cfo_tone,
                    "cfo_stf": cfo_stf,
                    "cfo_use": cfo_use,
                    "ltf_start": ltf_start
                }
                # channel estimate H on used bins
                mask = (np.abs(ltf_ref) > 0.1)
                H = np.ones(N_FFT, dtype=np.complex64)
                H[mask] = Y[mask] / (ltf_ref[mask] + 1e-12)
                ltf_best = {"Y0": Y0, "Y1": Y1, "H": H, "score": score}

        # if no choice, save FAIL fig and continue
        if best_choice is None:
            dbg_path = os.path.join(args.output_dir, f"cap{ti:02d}_FAIL_nopeak.png")
            save_debug_figure(
                dbg_path, rx, args.fs, tone_info,
                corr, idxs, peaks,
                chosen={"stf_idx": -1, "stf_ncc": 0.0, "cfo_tone": cfo_tone, "cfo_stf": float("nan"), "cfo_use": cfo_tone},
                ltf_info=ltf_best,
                demod_info={},
                parse_info=None,
                title=f"cap={ti:02d} FAIL (no valid peak)"
            )
            print(f"[cap {ti:02d}] tone_CFO={cfo_tone:+.1f} Hz | no valid STF/LTF peak (saved {os.path.basename(dbg_path)})")
            continue

        # proceed demod with chosen peak
        stf_idx = best_choice["stf_idx"]
        rxc = apply_cfo(rx, best_choice["cfo_use"], args.fs)
        H = ltf_best["H"]
        ltf_start = best_choice["ltf_start"]
        payload_start = ltf_start + 2*SYMBOL_LEN

        # equalize + pilot CPE
        phase_acc = 0.0
        freq_acc = 0.0

        all_data_pre = []
        all_data_post = []
        bits_out = []

        phase_err_log = []
        phase_acc_log = []
        freq_acc_log = []
        pilot_pwr_log = []
        evm_log = []

        # precompute used bins list for equalization
        used_bins = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
        #used_idx  = np.array([(k + N_FFT) % N_FFT for k in used_bins], dtype=int)
        used_idx = np.array([(k + N_FFT//2) % N_FFT for k in used_bins], dtype=int)

        # Ideal pilots for EVM reference in pilot bins only; for data bins, ideal unknown.
        # We'll compute EVM vs nearest hard-decision constellation per-symbol (post-CPE) for a proxy.
        const_points = np.array([ (1+1j), (-1+1j), (-1-1j), (1-1j) ], dtype=np.complex64) / np.sqrt(2)

        nsyms = 0
        for sym_idx in range(args.max_ofdm_syms):
            start = payload_start + sym_idx*SYMBOL_LEN
            Y = extract_symbol_fftshift(rxc, start)
            if Y is None:
                break

            # Equalize on used bins
            Yeq = np.zeros_like(Y)
            # only on used_idx
            denom = H[used_idx]
            good = (np.abs(denom) > 1e-6)
            tmp = np.zeros_like(denom)
            tmp[good] = Y[used_idx][good] / denom[good]
            Yeq[used_idx] = tmp

            # pilots
            expected_p = pilot_pattern(sym_idx)
            rx_p = Yeq[pilot_idx]
            pilot_pwr = float(np.mean(np.abs(rx_p)**2))
            pilot_pwr_log.append(pilot_pwr)

            # phase error (sum correlation)
            phase_err = np.angle(np.sum(rx_p * np.conj(expected_p)))
            # unwrap log for viewing (do not affect control)
            if len(phase_err_log) > 0:
                prev = phase_err_log[-1]
                # manual unwrap
                while phase_err - prev > np.pi: phase_err -= 2*np.pi
                while phase_err - prev < -np.pi: phase_err += 2*np.pi
            phase_err_log.append(phase_err)

            # PI loop
            # freq_acc += args.ki * phase_err
            # phase_acc += freq_acc + args.kp * phase_err
            
            if pilot_pwr < 1e-5:
                # skip loop update, keep last phase_acc/freq_acc
                pass
            else:
                freq_acc += args.ki * phase_err
                phase_acc += freq_acc + args.kp * phase_err
                
            phase_acc_log.append(phase_acc)
            freq_acc_log.append(freq_acc)

            rot = np.exp(-1j * phase_acc).astype(np.complex64)
            Yrot = Yeq * rot

            data_pre = Yeq[data_idx]
            data_post = Yrot[data_idx]
            all_data_pre.append(data_pre)
            all_data_post.append(data_post)

            # hard decision for bits
            bits_out.append(qpsk_demap_gray(data_post))

            # EVM proxy: compare to nearest constellation point (per symbol, per subcarrier)
            # (This is a useful debug signal even without knowing actual transmitted bits.)
            dp = data_post
            # nearest point
            # dist shape: [len(dp), 4]
            dist = np.abs(dp.reshape(-1,1) - R.reshape(1,-1))**2
            nearest = R[np.argmin(dist, axis=1)]
            evm = evm_db(dp.reshape(1,-1), nearest.reshape(1,-1))[0]
            evm_log.append(float(evm))

            nsyms += 1

        demod_info = {
            "nsyms": nsyms,
            "pilot_pwr": np.array(pilot_pwr_log, dtype=np.float32),
            "phase_err": np.array(phase_err_log, dtype=np.float32),
            "phase_acc": np.array(phase_acc_log, dtype=np.float32),
            "freq_acc": np.array(freq_acc_log, dtype=np.float32),
            "evm_db": np.array(evm_log, dtype=np.float32),
        }

        if nsyms == 0:
            dbg_path = os.path.join(args.output_dir, f"cap{ti:02d}_FAIL_nosyms.png")
            save_debug_figure(
                dbg_path, rx, args.fs, tone_info,
                corr, idxs, peaks,
                chosen=best_choice, ltf_info=ltf_best,
                demod_info=demod_info,
                parse_info=None,
                title=f"cap={ti:02d} FAIL (no OFDM symbols)"
            )
            print(f"[cap {ti:02d}] FAIL no OFDM symbols (saved {os.path.basename(dbg_path)})")
            continue

        data_pre = np.concatenate(all_data_pre) if len(all_data_pre) else np.array([], dtype=np.complex64)
        data_post = np.concatenate(all_data_post) if len(all_data_post) else np.array([], dtype=np.complex64)
        bits_raw = np.concatenate(bits_out) if len(bits_out) else np.array([], dtype=np.uint8)

        bits = majority_vote(bits_raw, args.repeat)
        bbytes = np.packbits(bits).tobytes()

        demod_info["data_pre"] = data_pre
        demod_info["data_post"] = data_post
        demod_info["bytes_len"] = len(bbytes)

        parse = scan_magic_and_parse(bbytes)

        ok = (parse is not None and parse["ok"])
        dbg_path = os.path.join(args.output_dir, f"cap{ti:02d}_{'OK' if ok else 'FAIL'}.png")

        save_debug_figure(
            dbg_path, rx, args.fs, tone_info,
            corr, idxs, peaks,
            chosen=best_choice, ltf_info=ltf_best,
            demod_info=demod_info,
            parse_info=parse,
            title=f"cap={ti:02d} {'OK' if ok else 'FAIL'}"
        )

        if parse is None:
            print(f"[cap {ti:02d}] tone_CFO={best_choice['cfo_tone']:+.1f} Hz STF_ncc={best_choice['stf_ncc']:.3f} LTF={ltf_best['score']:.3f} -> no MAGIC (saved {os.path.basename(dbg_path)})")
            continue

        if not parse["ok"]:
            print(f"[cap {ti:02d}] MAGIC but CRC FAIL: seq={parse['seq']} total={parse['total']} plen={parse['plen']} "
                  f"rx=0x{parse['crc_rx']:08X} calc=0x{parse['crc_calc']:08X} (saved {os.path.basename(dbg_path)})")
            continue

        # store packet
        seq = parse["seq"]
        total = parse["total"]
        if total_expected is None:
            total_expected = total
        elif total_expected != total:
            # keep the first seen total; still store
            pass

        got[seq] = parse["payload"]
        print(f"[cap {ti:02d}] OK seq={seq}/{total-1} plen={parse['plen']} CRC=0x{parse['crc_calc']:08X} "
              f"CFO={best_choice['cfo_use']:+.1f}Hz STF={best_choice['stf_ncc']:.3f} LTF={ltf_best['score']:.3f} "
              f"(saved {os.path.basename(dbg_path)})")

        # if complete
        if total_expected is not None and len(got) >= total_expected:
            # reassemble
            full = b"".join(got[i] for i in range(total_expected) if i in got)
            with open(args.outfile, "wb") as f:
                f.write(full)
            print(f"\n[RX] Reassembled payload: {len(full)} bytes -> {args.outfile}")

            # BER
            if len(ref_payload) > 0:
                err, tot, ber = ber_bits(full, ref_payload)
                print(f"[BER] compare_len={min(len(full),len(ref_payload))} bytes  bit_err={err} bit_tot={tot}  BER={ber:.3e}")
            break

    else:
        print("\n[RX] Incomplete: no full payload reassembled.")
        if total_expected is not None:
            print(f"  got {len(got)}/{total_expected} packets: {sorted(got.keys())}")

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
python3 rf_step5_rx_multipkt_dbg.py \
  --uri "ip:192.168.2.2" \
  --fc 915e6 --fs 2e6 --bw 2.4e6 \
  --rx_gain 40 \
  --buf_size 131072 \
  --repeat 1 \
  --tries 20 \
  --output_dir rf_step5_rx_dbg \
  --outfile recovered_payload.bin \
  --ref_seed 12345 --ref_len 4096
"""