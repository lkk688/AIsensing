#!/usr/bin/env python3
"""
pluto_ofdm_rx_dbg_v3.py

Two-Pluto robust OFDM RX with heavy debug.

Core improvements vs v2:
  (A) Packet start disambiguation:
      - Schmidl–Cox gives candidate peaks
      - Full PREAMBLE matched filter (STF+LTF) chooses true packet start
      This fixes "pilot power is ~0 for first ~200 symbols" (wrong payload start).

  (B) CFO:
      - coarse CFO from Schmidl–Cox (rough)
      - refined CFO from STF symbol-to-symbol repetition (separation = N+CP)

  (C) Residual CFO + CPE tracking:
      - 2nd-order pilot loop (FLL + PLL):
          freq_acc += kf * ph_err
          ph_acc   += freq_acc + kp * ph_err
      - Apply exp(-j*ph_acc) per symbol.

  (D) Debug figures:
      - Schmidl metric neighborhood
      - preamble correlation scan (normalized) + chosen start
      - LTF correlation around expected LTF0
      - channel |H| and angle(H)
      - constellation pre/post
      - pilot power per symbol
      - pilot phase error / residual CFO estimate
      - EVM per symbol
      - post-CPE angle histogram

Assumes TX matches the earlier TX design:
  N=64 CP=16 fs ~1e6
  STF: 2 identical OFDM symbols (random BPSK on used bins, rng seed 0)
  LTF: 2 identical OFDM symbols (deterministic alternating BPSK on used bins)
  Payload: QPSK on 48 data bins, pilots on 4 bins with +/- alternation
  TX appends CRC32 (4 bytes, little-endian) to payload bytes.
"""

import argparse
import os
import zlib
import numpy as np
import adi
import matplotlib.pyplot as plt


# ----------------------------
# OFDM plan (N=64, 802.11-like)
# ----------------------------
def make_subcarrier_plan(N: int):
    if N != 64:
        raise ValueError("This implementation is written for N=64 only.")
    pilots = np.array([-21, -7, 7, 21], dtype=int)
    used = np.r_[np.arange(-26, 0), np.arange(1, 27)]
    data = np.array([k for k in used if k not in set(pilots)], dtype=int)
    return data, pilots, used


def pilot_pattern(sym_idx: int, pilots_len: int):
    base = np.array([1, 1, 1, -1], dtype=np.float32)
    if pilots_len != 4:
        base = np.ones(pilots_len, dtype=np.float32)
    if sym_idx % 2 == 1:
        base = -base
    return base + 0j


def ofdm_symbol_from_bins(X_shift: np.ndarray, N: int, CP: int) -> np.ndarray:
    """X_shift is fftshift ordering (DC at center). Return CP+N time-domain symbol."""
    x = np.fft.ifft(np.fft.ifftshift(X_shift)).astype(np.complex64)
    return np.r_[x[-CP:], x]


def build_stf_ltf(N: int, CP: int, data_bins, pilot_bins, used_bins):
    """
    Reconstruct STF and LTF exactly like TX.

    STF:
      - Random BPSK on used bins, rng seed 0
      - One OFDM symbol (CP+N) repeated twice

    LTF:
      - Deterministic alternating BPSK on used bins
      - One OFDM symbol (CP+N) repeated twice
    """
    used = np.r_[data_bins, pilot_bins]
    used = np.array(sorted(set(used.tolist())), dtype=int)

    # STF
    Xstf = np.zeros(N, dtype=np.complex64)
    rng = np.random.default_rng(0)
    bpsk = rng.choice([-1.0, 1.0], size=len(used)).astype(np.float32)
    Xstf[(used + N // 2) % N] = bpsk + 0j
    stf_sym = ofdm_symbol_from_bins(Xstf, N, CP)
    STF = np.r_[stf_sym, stf_sym].astype(np.complex64)

    # LTF
    Xltf = np.zeros(N, dtype=np.complex64)
    ltf_bpsk = np.ones(len(used), dtype=np.float32)
    ltf_bpsk[::2] = -1.0
    Xltf[(used + N // 2) % N] = ltf_bpsk + 0j
    ltf_sym = ofdm_symbol_from_bins(Xltf, N, CP)
    LTF = np.r_[ltf_sym, ltf_sym].astype(np.complex64)

    preamble = np.r_[STF, LTF].astype(np.complex64)
    return STF, LTF, preamble, Xltf, used


# ----------------------------
# QPSK demap (Gray)
# ----------------------------
def qpsk_demap_gray(syms: np.ndarray) -> np.ndarray:
    bits = np.zeros((len(syms), 2), dtype=np.uint8)
    re = np.real(syms) >= 0
    im = np.imag(syms) >= 0
    # + + : 00
    # - + : 01
    # - - : 11
    # + - : 10
    for i in range(len(syms)):
        if re[i] and im[i]:
            bits[i] = [0, 0]
        elif (not re[i]) and im[i]:
            bits[i] = [0, 1]
        elif (not re[i]) and (not im[i]):
            bits[i] = [1, 1]
        else:
            bits[i] = [1, 0]
    return bits.reshape(-1)


def majority_vote_repetition(bits_rep: np.ndarray, rep: int) -> np.ndarray:
    if rep == 1:
        return bits_rep.astype(np.uint8)
    L = (len(bits_rep) // rep) * rep
    bits_rep = bits_rep[:L].reshape(-1, rep)
    return (np.sum(bits_rep, axis=1) >= (rep / 2)).astype(np.uint8)


# ----------------------------
# OFDM symbol extraction
# ----------------------------
def extract_ofdm_td(rx: np.ndarray, start_cp: int, N: int, CP: int):
    a = start_cp + CP
    b = a + N
    if a < 0 or b > len(rx):
        return None
    return rx[a:b]


def extract_ofdm_fd(rx: np.ndarray, start_cp: int, N: int, CP: int):
    td = extract_ofdm_td(rx, start_cp, N, CP)
    if td is None:
        return None
    return np.fft.fftshift(np.fft.fft(td))


# ----------------------------
# Schmidl–Cox (candidate peaks)
# ----------------------------
def schmidl_metric(rx: np.ndarray, N: int, step: int = 4):
    """
    P(d) = sum r[d+n] * conj(r[d+n+N])
    angle(P) ≈ -2*pi*CFO*N/fs  => CFO = -angle(P)*fs/(2*pi*N)
    """
    L = len(rx)
    Lm = L - 2 * N
    if Lm <= 0:
        return None, None

    M = np.zeros(Lm, dtype=np.float32)
    P = np.zeros(Lm, dtype=np.complex64)
    eps = 1e-12

    for d in range(0, Lm, step):
        seg1 = rx[d : d + N]
        seg2 = rx[d + N : d + 2 * N]
        Pd = np.sum(seg1 * np.conj(seg2))
        denom = np.sum(np.abs(seg2) ** 2) + eps
        P[d] = Pd
        M[d] = (np.abs(Pd) ** 2) / (denom**2)

    return M, P


def top_k_indices(x: np.ndarray, k: int):
    if len(x) == 0:
        return []
    k = min(k, len(x))
    idx = np.argpartition(x, -k)[-k:]
    idx = idx[np.argsort(-x[idx])]
    return idx.tolist()


# ----------------------------
# Preamble matched filter (normalized)
# ----------------------------
def preamble_corr_scan(rx: np.ndarray, preamble: np.ndarray, start: int, stop: int, step: int = 2):
    """
    Normalized correlation:
      c[i] = | <rx_seg, preamble> | / (||rx_seg|| * ||preamble||)

    Returns:
      xs (candidate starts), cs (corr values)
    """
    Lp = len(preamble)
    eps = 1e-12
    pnorm = np.sqrt(np.sum(np.abs(preamble) ** 2)) + eps

    xs = []
    cs = []
    for s in range(start, stop, step):
        if s < 0 or (s + Lp) > len(rx):
            continue
        seg = rx[s : s + Lp]
        snorm = np.sqrt(np.sum(np.abs(seg) ** 2)) + eps
        v = np.abs(np.vdot(seg, preamble)) / (snorm * pnorm)
        xs.append(s)
        cs.append(float(v))
    return np.array(xs, dtype=int), np.array(cs, dtype=np.float32)


def refine_max(xs: np.ndarray, cs: np.ndarray):
    if len(cs) == 0:
        return None, None
    i = int(np.argmax(cs))
    return int(xs[i]), float(cs[i])


# ----------------------------
# CFO refine from STF repetition (symbol-to-symbol)
# ----------------------------
def refine_cfo_from_stf(rx: np.ndarray, stf_cp: int, fs: float, N: int, CP: int):
    """
    Use STF: two identical OFDM symbols (CP+N each).
    Use their DATA parts (length N) separated by (N+CP).

      P = sum d1[n] * conj(d2[n])
      angle(P) ≈ -2*pi*CFO*(N+CP)/fs
      CFO = -angle(P)*fs/(2*pi*(N+CP))
    """
    Lsym = N + CP
    d1 = extract_ofdm_td(rx, stf_cp, N, CP)
    d2 = extract_ofdm_td(rx, stf_cp + Lsym, N, CP)
    if d1 is None or d2 is None:
        return None
    P = np.sum(d1 * np.conj(d2))
    cfo = -(np.angle(P) * fs) / (2 * np.pi * Lsym)
    return float(cfo)


# ----------------------------
# Metrics
# ----------------------------
def evm_db(rx_syms: np.ndarray, ref_syms: np.ndarray):
    e = rx_syms - ref_syms
    num = np.mean(np.abs(e) ** 2) + 1e-12
    den = np.mean(np.abs(ref_syms) ** 2) + 1e-12
    evm = np.sqrt(num / den)
    return 20 * np.log10(evm + 1e-12)


def hard_ref_syms_from_bits(bits: np.ndarray):
    hb = bits.reshape(-1, 2)
    ref = np.empty(len(hb), dtype=np.complex64)
    for i, (b0, b1) in enumerate(hb):
        if b0 == 0 and b1 == 0:
            ref[i] = 1 + 1j
        elif b0 == 0 and b1 == 1:
            ref[i] = -1 + 1j
        elif b0 == 1 and b1 == 1:
            ref[i] = -1 - 1j
        else:
            ref[i] = 1 - 1j
    return ref / np.sqrt(2)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=1e6)
    ap.add_argument("--bw", type=float, default=1.2e6)
    ap.add_argument("--rx_gain", type=float, default=55.0)
    ap.add_argument("--buf", type=int, default=2**20)
    ap.add_argument("--repeat", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--num_syms", type=int, default=300)
    ap.add_argument("--expect_len", type=int, default=2048, help="bytes BEFORE CRC32")
    ap.add_argument("--outfile", default="recovered.jpg")
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--debug_dir", default="", help="save debug plots here")
    # search parameters
    ap.add_argument("--k_peaks", type=int, default=6, help="how many Schmidl peaks to try")
    ap.add_argument("--win", type=int, default=8000, help="preamble search half-window around each peak")
    ap.add_argument("--scan_step", type=int, default=2, help="preamble scan step")
    ap.add_argument("--refine_span", type=int, default=64, help="final refine +/- span (step=1)")
    # equalization safety
    ap.add_argument("--H_min", type=float, default=1e-2, help="min |H| to trust equalization")
    # pilot loop gains (2nd order)
    ap.add_argument("--kp", type=float, default=0.08, help="PLL proportional gain")
    ap.add_argument("--kf", type=float, default=0.0015, help="FLL gain (freq integrator)")
    args = ap.parse_args()

    N, CP = 64, 16
    Lsym = N + CP

    data_bins, pilot_bins, used_bins = make_subcarrier_plan(N)
    didx = (data_bins + N // 2) % N
    pidx = (pilot_bins + N // 2) % N
    used_idx = ((np.r_[data_bins, pilot_bins] + N // 2) % N).astype(int)

    STF, LTF, PREAMBLE, Xltf, used_all_bins = build_stf_ltf(N, CP, data_bins, pilot_bins, used_bins)
    Lp = len(PREAMBLE)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf)

    # flush
    for _ in range(3):
        _ = sdr.rx()

    print("RX running")
    print(f"  uri={args.uri} fc={args.fc/1e6:.3f} MHz fs={args.fs/1e6:.3f} Msps bw={args.bw/1e6:.3f} MHz gain={args.rx_gain} dB")
    print(f"  N={N} CP={CP} data_sc={len(data_bins)} pilots={len(pilot_bins)} repeat={args.repeat} payload_syms={args.num_syms}")
    print(f"  Preamble matched filter: k_peaks={args.k_peaks} win={args.win} scan_step={args.scan_step} refine_span={args.refine_span}")
    print(f"  Pilot loop: kp={args.kp} kf={args.kf} |H|>={args.H_min}")

    try:
        for t in range(args.tries):
            rx_raw = sdr.rx().astype(np.complex64) / (2**14)
            rx = rx_raw - np.mean(rx_raw)  # DC remove

            # --- 1) Schmidl–Cox candidates ---
            M, P = schmidl_metric(rx, N, step=4)
            if M is None:
                print(f"[{t+1:02d}] buffer too short")
                continue

            peaks = top_k_indices(M, args.k_peaks)
            if len(peaks) == 0:
                print(f"[{t+1:02d}] no peaks")
                continue

            # Pick best candidate by preamble correlation search
            best_start = None
            best_corr = -1.0
            best_peak = None
            best_scan = None  # store scan trace for debug

            # Use CFO from the strongest peak (rough), then CFO-correct for scanning
            # (We also try per-peak CFO; but usually rough CFO is enough for correlation)
            # Choose the first peak as "strongest" since peaks is sorted by metric desc.
            peak0 = peaks[0]
            Pd0 = P[peak0]
            cfo_coarse = -(np.angle(Pd0) * args.fs) / (2 * np.pi * N)

            n = np.arange(len(rx), dtype=np.float32)
            rx_cfo0 = rx * np.exp(-1j * 2 * np.pi * cfo_coarse * n / args.fs)

            for pk in peaks:
                # search window around pk
                lo = pk - args.win
                hi = pk + args.win
                xs, cs = preamble_corr_scan(rx_cfo0, PREAMBLE, lo, hi, step=args.scan_step)
                s0, c0 = refine_max(xs, cs)
                if s0 is None:
                    continue
                if c0 > best_corr:
                    best_corr = c0
                    best_start = s0
                    best_peak = pk
                    best_scan = (xs, cs, lo, hi)

            if best_start is None:
                print(f"[{t+1:02d}] preamble not found (coarse CFO={cfo_coarse:+.1f}Hz)")
                continue

            # final refine around best_start with step=1
            xs2, cs2 = preamble_corr_scan(
                rx_cfo0, PREAMBLE,
                best_start - args.refine_span,
                best_start + args.refine_span + 1,
                step=1
            )
            stf_cp, pre_corr = refine_max(xs2, cs2)
            if stf_cp is None:
                print(f"[{t+1:02d}] refine failed")
                continue

            # --- 2) Refine CFO from STF repetition at aligned start ---
            cfo_ref = refine_cfo_from_stf(rx, stf_cp, args.fs, N, CP)
            if cfo_ref is None:
                cfo_ref = cfo_coarse

            rx_cfo = rx * np.exp(-1j * 2 * np.pi * cfo_ref * n / args.fs)

            # --- 3) Channel estimate from LTF (2 symbols) ---
            ltf0_cp = stf_cp + 2 * Lsym
            ltf1_cp = ltf0_cp + Lsym

            Y0 = extract_ofdm_fd(rx_cfo, ltf0_cp, N, CP)
            Y1 = extract_ofdm_fd(rx_cfo, ltf1_cp, N, CP)
            if Y0 is None or Y1 is None:
                print(f"[{t+1:02d}] insufficient samples for LTF")
                continue

            Yltf = 0.5 * (Y0 + Y1)

            # Build Xltf in fftshift indexing (already built in build_stf_ltf)
            # Channel on all bins, but only trust used bins
            H = np.ones(N, dtype=np.complex64)
            eps = 1e-9
            Xltf_shift = Xltf  # fftshift vector
            used_ltf_idx = ((used_all_bins + N // 2) % N).astype(int)
            H[used_ltf_idx] = Yltf[used_ltf_idx] / (Xltf_shift[used_ltf_idx] + eps)

            # equalization mask
            H_used = H[used_idx]
            mask = np.abs(H_used) >= float(args.H_min)

            # --- 4) Payload demod with 2nd-order pilot loop ---
            pay_cp0 = stf_cp + len(PREAMBLE)  # since PREAMBLE = STF(2syms)+LTF(2syms)
            bits_hat = []

            # logs for debug
            pilot_pow = []
            ph_err_log = []
            ph_acc_log = []
            freq_acc_log = []
            evm_log = []
            const_pre = []
            const_post = []

            ph_acc = 0.0
            freq_acc = 0.0  # radians/symbol (residual CFO in discrete-time symbol domain)

            syms_decoded = 0
            for s in range(args.num_syms):
                sym_cp = pay_cp0 + s * Lsym
                Y = extract_ofdm_fd(rx_cfo, sym_cp, N, CP)
                if Y is None:
                    break

                # Equalize only used bins
                Ye = np.zeros_like(Y)
                Y_used = Y[used_idx]
                Ye_used = np.zeros_like(Y_used)
                Ye_used[mask] = Y_used[mask] / (H_used[mask] + eps)
                Ye[used_idx] = Ye_used

                # Pilot phase error
                pilots_rx = Ye[pidx]
                pilots_ref = pilot_pattern(s, len(pilot_bins))
                # complex error -> robust phase
                e = np.sum(pilots_rx * np.conj(pilots_ref))
                ph_err = np.angle(e)

                # 2nd-order loop (FLL + PLL)
                freq_acc += float(args.kf) * ph_err
                ph_acc += freq_acc + float(args.kp) * ph_err

                ph_err_log.append(ph_err)
                ph_acc_log.append(ph_acc)
                freq_acc_log.append(freq_acc)
                pilot_pow.append(float(np.mean(np.abs(pilots_rx) ** 2)))

                # apply correction
                data_pre = Ye[didx].copy()
                Ye[used_idx] *= np.exp(-1j * ph_acc)
                data_post = Ye[didx].copy()

                const_pre.append(data_pre)
                const_post.append(data_post)

                hb = qpsk_demap_gray(data_post)
                bits_hat.append(hb)

                # EVM vs nearest hard-decision ref
                ref = hard_ref_syms_from_bits(hb)
                evm_log.append(float(evm_db(data_post, ref)))

                syms_decoded += 1

            if len(bits_hat) == 0:
                print(f"[{t+1:02d}] locked but no payload decoded")
                continue

            bits_hat = np.concatenate(bits_hat).astype(np.uint8)
            bits_dec = majority_vote_repetition(bits_hat, args.repeat)
            bb = np.packbits(bits_dec).tobytes()

            total = args.expect_len + 4
            bb = bb[:total]
            payload = bb[:-4]
            crc_rx = int.from_bytes(bb[-4:], "little")
            crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
            ok = (crc_calc == crc_rx)

            # --- 5) Extra debug: LTF-only correlation around expected LTF0 ---
            # (should show a sharp peak at offset 0 if stf_cp is correct)
            ltf_sym = LTF[:Lsym]  # one LTF symbol (CP+N)
            lo2 = ltf0_cp - 120
            hi2 = ltf0_cp + 120
            xs_ltf, cs_ltf = preamble_corr_scan(rx_cfo, ltf_sym, lo2, hi2, step=1)
            # shift x-axis to offsets
            off_ltf = xs_ltf - ltf0_cp if len(xs_ltf) else np.array([0])

            # --- 6) Debug plot per try ---
            if args.debug_dir:
                cp = np.concatenate(const_pre) if len(const_pre) else np.zeros(1, dtype=np.complex64)
                cpost = np.concatenate(const_post) if len(const_post) else np.zeros(1, dtype=np.complex64)

                # Schmidl neighborhood around best_peak
                w = 600
                loM = max(0, best_peak - w)
                hiM = min(len(M) - 1, best_peak + w)
                xsM = np.arange(loM, hiM)
                Ms = M[loM:hiM]

                # Preamble scan trace (best window)
                xs_scan, cs_scan, scan_lo, scan_hi = best_scan

                # Channel plots
                Hmag = 20 * np.log10(np.abs(H) + 1e-12)
                Hph = np.unwrap(np.angle(H + 1e-12))

                fig = plt.figure(figsize=(20, 12))

                ax1 = fig.add_subplot(3, 4, 1)
                ax1.plot(xsM, Ms)
                ax1.axvline(best_peak, linestyle="--", label="chosen peak")
                ax1.set_title("Schmidl–Cox metric (neighborhood)")
                ax1.set_xlabel("sample")
                ax1.grid(True)
                ax1.legend()

                ax2 = fig.add_subplot(3, 4, 2)
                ax2.plot(xs_scan, cs_scan)
                ax2.axvline(stf_cp, linestyle="--", label="STF cp chosen")
                ax2.set_title("Preamble matched filter (normalized corr)")
                ax2.set_xlabel("candidate STF cp start")
                ax2.grid(True)
                ax2.legend()

                ax3 = fig.add_subplot(3, 4, 3)
                # time magnitude around preamble
                zlo = max(0, stf_cp - 2 * Lsym)
                zhi = min(len(rx_cfo), stf_cp + Lp + 2 * Lsym)
                ax3.plot(np.arange(zlo, zhi), np.abs(rx_cfo[zlo:zhi]))
                ax3.axvline(stf_cp, linestyle="--", label="STF cp")
                ax3.axvline(ltf0_cp, linestyle="--", label="LTF0 cp")
                ax3.axvline(pay_cp0, linestyle="--", label="PAY cp0")
                ax3.set_title("|rx_cfo| around preamble+start")
                ax3.grid(True)
                ax3.legend(loc="upper right")

                ax4 = fig.add_subplot(3, 4, 4)
                ax4.plot(np.arange(N), Hmag)
                ax4.set_title("Estimated channel |H[k]| (dB)")
                ax4.set_xlabel("bin (fftshift order)")
                ax4.grid(True)

                ax5 = fig.add_subplot(3, 4, 5)
                ax5.plot(np.arange(N), Hph)
                ax5.set_title("Estimated channel angle(H) (unwrap)")
                ax5.set_xlabel("bin (fftshift order)")
                ax5.grid(True)

                ax6 = fig.add_subplot(3, 4, 6)
                ax6.scatter(np.real(cp), np.imag(cp), s=4, alpha=0.35)
                ax6.set_title("Constellation (eq pre-loop)")
                ax6.grid(True)

                ax7 = fig.add_subplot(3, 4, 7)
                ax7.scatter(np.real(cpost), np.imag(cpost), s=4, alpha=0.35)
                ax7.set_title("Constellation (post loop: FLL+PLL)")
                ax7.grid(True)

                ax8 = fig.add_subplot(3, 4, 8)
                ax8.plot(np.unwrap(np.array(ph_acc_log)), label="ph_acc (unwrap)")
                ax8.plot(np.array(ph_err_log), alpha=0.6, label="ph_err (wrapped)")
                ax8.set_title("Pilot loop phase")
                ax8.set_xlabel("OFDM symbol")
                ax8.grid(True)
                ax8.legend()

                ax9 = fig.add_subplot(3, 4, 9)
                ax9.plot(np.array(evm_log))
                ax9.set_title("EVM per symbol (dB) — lower is better")
                ax9.set_xlabel("OFDM symbol")
                ax9.grid(True)

                ax10 = fig.add_subplot(3, 4, 10)
                ax10.plot(np.array(pilot_pow))
                ax10.set_title("Pilot power per symbol (should be non-zero immediately if aligned)")
                ax10.set_xlabel("OFDM symbol")
                ax10.grid(True)

                ax11 = fig.add_subplot(3, 4, 11)
                ax11.plot(np.array(freq_acc_log))
                ax11.set_title("Residual CFO estimator (freq_acc, rad/symbol)")
                ax11.set_xlabel("OFDM symbol")
                ax11.grid(True)

                ax12 = fig.add_subplot(3, 4, 12)
                if len(off_ltf) > 1:
                    ax12.plot(off_ltf, cs_ltf)
                    ax12.axvline(0, linestyle="--")
                ax12.set_title("LTF-only correlation around expected LTF0 (peak at 0 = good)")
                ax12.set_xlabel("offset (samples)")
                ax12.grid(True)

                fig.suptitle(
                    f"try={t+1:02d} ok={ok}  pre_corr={pre_corr:.3f}  "
                    f"CFO_coarse={cfo_coarse:+.1f}Hz CFO_ref={cfo_ref:+.1f}Hz  "
                    f"syms_decoded={syms_decoded}  crc_calc={crc_calc:08x} crc_rx={crc_rx:08x}"
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                out = os.path.join(args.debug_dir, f"rx_dbg_v3_try{t+1:02d}_{'OK' if ok else 'FAIL'}.png")
                fig.savefig(out, dpi=140)
                plt.close(fig)

                # Additional small figure: post-CPE angle histogram
                fig2 = plt.figure(figsize=(7, 4))
                axh = fig2.add_subplot(1, 1, 1)
                ang = np.angle(cpost) if len(cpost) > 1 else np.zeros(1)
                axh.hist(ang, bins=60)
                axh.set_title("Post-loop angle histogram (expect 4 peaks for QPSK)")
                axh.grid(True)
                fig2.tight_layout()
                out2 = os.path.join(args.debug_dir, f"rx_ang_v3_try{t+1:02d}_{'OK' if ok else 'FAIL'}.png")
                fig2.savefig(out2, dpi=140)
                plt.close(fig2)

            # Print outcome
            if ok:
                open(args.outfile, "wb").write(payload)
                print(f"[{t+1:02d}] ✅ CRC OK! wrote {len(payload)} bytes -> {args.outfile}")
                break
            else:
                print(
                    f"[{t+1:02d}] CRC fail  pre_corr={pre_corr:.3f}  "
                    f"CFO_coarse={cfo_coarse:+.1f}Hz CFO_ref={cfo_ref:+.1f}Hz  "
                    f"calc={crc_calc:08x} rx={crc_rx:08x}"
                )

        else:
            print("No valid packet recovered.")
            print("If v3 still fails, the debug plot will tell us which knob:")
            print("  - If preamble corr is low/flat: SNR/overload problem")
            print("  - If LTF-only corr peak not at 0: still timing error")
            print("  - If pilot power starts near 0 then rises later: payload start still wrong (usually cyclic/gap issue)")
            print("  - If freq_acc ramps: residual CFO needs stronger kf or lower bandwidth")

    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass


if __name__ == "__main__":
    main()

# python3 pluto_ofdm_rx_dbg_v3.py \
#   --uri "usb:1.37.5" \
#   --fc 2.3e9 --fs 1e6 --bw 1.2e6 \
#   --rx_gain 55 \
#   --repeat 4 --num_syms 300 \
#   --expect_len 2048 \
#   --outfile recovered.jpg \
#   --tries 50 \
#   --debug_dir rx_debug_v3