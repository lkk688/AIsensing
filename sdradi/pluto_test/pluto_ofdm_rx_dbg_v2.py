#!/usr/bin/env python3
"""
pluto_ofdm_rx_dbg_v2.py

Robust OFDM RX for 2x Pluto with heavy debug figures.

Key fixes vs previous dbg version:
  1) CFO sign FIX for Schmidl–Cox:
       CFO = -angle(P) * fs / (2*pi*N)
  2) Fine timing FIX using full (CP+N) LTF matched filter (sharp peak expected)
  3) Equalization safety: ignore bins where |H| is too small (prevents huge blowups)
  4) Much richer debug figures saved per try:
       - coarse Schmidl–Cox metric around peak
       - fine LTF correlation vs offset (FULL symbol matched filter)
       - estimated channel magnitude + phase
       - constellation (eq pre-CPE) and (post-CPE)
       - CPE (pilot phase) / EVM / pilot power
       - residual CFO trend proxy (phase slope) and raw time magnitude around preamble

Assumes TX matches the earlier TX script:
  N=64, CP=16
  STF: 2 identical OFDM symbols (each CP+N)
  LTF: 2 identical OFDM symbols with deterministic BPSK on used bins
  Payload: QPSK on 48 data bins, pilots on 4 bins with simple +/- alternation

Usage:
  python3 pluto_ofdm_rx_dbg_v2.py \
    --uri "usb:1.37.5" \
    --fc 2.3e9 --fs 1e6 --bw 1.2e6 \
    --rx_gain 55 \
    --repeat 4 --num_syms 300 \
    --expect_len 2048 \
    --outfile recovered.jpg \
    --tries 50 \
    --debug_dir rx_debug

Tip:
  Verify file size on TX host:
    ls -l stress_2kb.jpg
  expect_len must match BYTES before CRC32 (TX appends 4 bytes CRC).
"""

import argparse
import os
import zlib
import numpy as np
import adi
import matplotlib.pyplot as plt


# ----------------------------
# Subcarrier plan (N=64)
# ----------------------------
def make_subcarrier_plan(N: int):
    if N != 64:
        raise ValueError("This implementation is for N=64.")
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
# OFDM extraction helpers
# ----------------------------
def extract_ofdm_symbol_td(rx: np.ndarray, start_cp: int, N: int, CP: int):
    """Return time-domain OFDM data part (length N), CP removed."""
    a = start_cp + CP
    b = a + N
    if a < 0 or b > len(rx):
        return None
    return rx[a:b]


def extract_ofdm_symbol_fd(rx: np.ndarray, start_cp: int, N: int, CP: int):
    s = extract_ofdm_symbol_td(rx, start_cp, N, CP)
    if s is None:
        return None
    return np.fft.fftshift(np.fft.fft(s))


# ----------------------------
# Schmidl–Cox metric
# ----------------------------
def schmidl_cox_metric(rx: np.ndarray, N: int, step: int = 1):
    """
    Repeated-block metric correlating N samples with the next N samples.

    We define:
      P(d) = sum_{n=0..N-1} r[d+n] * conj(r[d+n+N])
    For repeated block under CFO, angle(P) ≈ -2*pi*CFO*N/fs

    Returns arrays M[d], P[d] for d in [0..len-2N).
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
        Pd = np.sum(seg1 * np.conj(seg2))  # <- explicit definition
        denom = np.sum(np.abs(seg2) ** 2) + eps
        P[d] = Pd
        M[d] = (np.abs(Pd) ** 2) / (denom**2)

    return M, P


def find_coarse_start(rx: np.ndarray, N: int):
    """
    Coarse peak search with a stepped scan + local refine.
    Returns (d_best, P_best, M_best, coarse_debug_dict)
    """
    step = 4
    M_step, P_step = schmidl_cox_metric(rx, N, step=step)
    if M_step is None:
        return None, None, None, None

    d0 = int(np.argmax(M_step))
    # refine in +/- 64 samples at step=1 using fresh metric
    lo = max(0, d0 - 64)
    hi = min(len(rx) - 2 * N - 1, d0 + 64)
    M_ref, P_ref = schmidl_cox_metric(rx[lo : hi + 2 * N + 1], N, step=1)
    if M_ref is None:
        return d0, P_step[d0], M_step[d0], {"M_step": M_step, "d0": d0}

    d1_rel = int(np.argmax(M_ref))
    d1 = lo + d1_rel
    dbg = {"M_step": M_step, "d0": d0, "ref_lo": lo, "M_ref": M_ref, "d1": d1}
    return d1, P_ref[d1_rel], M_ref[d1_rel], dbg


# ----------------------------
# Fine sync with full LTF matched filter (CP+N)
# ----------------------------
def fine_sync_with_ltf_full(rx_cfo: np.ndarray, ltf_cp_guess: int, Xltf_shift: np.ndarray, N: int, CP: int, search: int = 240):
    """
    Search around ltf_cp_guess for CP-start that best matches the full LTF symbol (CP+N).

    This is MUCH less ambiguous than correlating only the N-sample data portion.
    Expect a sharp peak if the preamble is present and CFO is reasonably corrected.
    """
    # Build time-domain LTF symbol used by TX (with CP)
    ltf_td = np.fft.ifft(np.fft.ifftshift(Xltf_shift)).astype(np.complex64)
    ltf_sym = np.r_[ltf_td[-CP:], ltf_td]  # length CP+N

    offsets = np.arange(-search, search + 1)
    corrs = np.zeros_like(offsets, dtype=np.float32)
    best_cp = None
    best_val = -1.0

    Lsym = N + CP
    for i, off in enumerate(offsets):
        cp = ltf_cp_guess + int(off)
        a = cp
        b = cp + Lsym
        if a < 0 or b > len(rx_cfo):
            corrs[i] = 0.0
            continue
        seg = rx_cfo[a:b]
        v = np.abs(np.vdot(seg, ltf_sym))
        corrs[i] = float(v)
        if v > best_val:
            best_val = v
            best_cp = cp

    return best_cp, offsets, corrs


# ----------------------------
# Metrics
# ----------------------------
def evm_db(rx_syms: np.ndarray, ref_syms: np.ndarray):
    e = rx_syms - ref_syms
    num = np.mean(np.abs(e) ** 2) + 1e-12
    den = np.mean(np.abs(ref_syms) ** 2) + 1e-12
    evm = np.sqrt(num / den)
    return 20 * np.log10(evm + 1e-12)


def qpsk_hard_ref_from_bits(bits: np.ndarray):
    """
    bits length multiple of 2; returns reference symbols per Gray map:
      00 -> +1 + j
      01 -> -1 + j
      11 -> -1 - j
      10 -> +1 - j
    normalized to 1/sqrt(2)
    """
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
    ap.add_argument("--expect_len", type=int, default=2048, help="bytes BEFORE CRC32 (TX appends 4 bytes)")
    ap.add_argument("--outfile", default="recovered.jpg")
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--debug_dir", default="", help="If set, save debug plots here.")
    ap.add_argument("--ltf_search", type=int, default=240, help="fine timing search +/- this many samples")
    ap.add_argument("--H_min", type=float, default=1e-2, help="min |H| to trust equalization")
    ap.add_argument("--cpe_alpha", type=float, default=0.25, help="CPE tracking smoothing (0..1)")
    args = ap.parse_args()

    N, CP = 64, 16
    Lsym = N + CP

    data_bins, pilot_bins, used_bins = make_subcarrier_plan(N)
    used_idx = (used_bins + N // 2) % N
    pidx = (pilot_bins + N // 2) % N
    didx = (data_bins + N // 2) % N

    # Known LTF in freq (fftshift indexing), must match TX
    Xltf = np.zeros(N, dtype=np.complex64)
    ltf_bpsk = np.ones(len(used_bins), dtype=np.float32)
    ltf_bpsk[::2] = -1.0
    Xltf[used_idx] = ltf_bpsk + 0j

    # SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf)

    # Flush
    for _ in range(3):
        _ = sdr.rx()

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    print("RX running")
    print(f"  uri={args.uri} fc={args.fc/1e6:.3f} MHz fs={args.fs/1e6:.3f} Msps bw={args.bw/1e6:.3f} MHz gain={args.rx_gain} dB")
    print(f"  N={N} CP={CP} data_sc={len(data_bins)} pilots={len(pilot_bins)} repeat={args.repeat} payload_syms={args.num_syms}")
    print(f"  FineSync: LTF matched filter search +/-{args.ltf_search} samples |H|>={args.H_min} CPE_alpha={args.cpe_alpha}")

    packet_found = False

    try:
        for t in range(args.tries):
            rx_raw = sdr.rx().astype(np.complex64) / (2**14)
            rx = rx_raw - np.mean(rx_raw)  # DC removal

            # Coarse Schmidl–Cox on repeated N block
            d, Pd, m, coarse_dbg = find_coarse_start(rx, N)
            if d is None:
                print(f"[{t+1:02d}] no lock")
                continue

            # --- CFO SIGN FIX ---
            # P angle ~ -2*pi*CFO*N/fs  =>  CFO = -angle(P)*fs/(2*pi*N)
            cfo_hz = -(np.angle(Pd) * args.fs) / (2 * np.pi * N)
            n = np.arange(len(rx), dtype=np.float32)
            rx_cfo = rx * np.exp(-1j * 2 * np.pi * cfo_hz * n / args.fs)

            # Guess CP-start of STF and LTF based on packet layout
            stf_guess = d
            stf_cp_guess = max(0, stf_guess - CP)
            ltf0_cp_guess = stf_cp_guess + 2 * Lsym

            # Fine timing using FULL LTF symbol matched filter
            ltf0_cp, offs, ltf_corr = fine_sync_with_ltf_full(
                rx_cfo, ltf0_cp_guess, Xltf, N, CP, search=args.ltf_search
            )
            if ltf0_cp is None:
                print(f"[{t+1:02d}] lock but fine LTF sync failed (CFO={cfo_hz:+.1f}Hz metric={m:.3f})")
                continue

            ltf1_cp = ltf0_cp + Lsym

            Y0 = extract_ofdm_symbol_fd(rx_cfo, ltf0_cp, N, CP)
            Y1 = extract_ofdm_symbol_fd(rx_cfo, ltf1_cp, N, CP)
            if Y0 is None or Y1 is None:
                print(f"[{t+1:02d}] lock but insufficient samples around LTF")
                continue

            Yltf = 0.5 * (Y0 + Y1)

            # Channel estimate on used bins
            H = np.ones(N, dtype=np.complex64)
            eps = 1e-9
            H[used_idx] = Yltf[used_idx] / (Xltf[used_idx] + eps)

            # Payload start after LTF1
            pay_cp0 = ltf1_cp + Lsym

            bits_hat = []
            cpe_log = []
            evm_log = []
            pilot_pow_log = []
            const_pre = []
            const_post = []

            # Extra: residual CFO proxy from pilots (phase slope)
            # We measure raw pilot phase error per symbol before smoothing
            cpe_inst_log = []

            ph = 0.0
            alpha = float(args.cpe_alpha)

            # Equalization mask to avoid |H| too small
            H_used = H[used_idx]
            H_mask = np.abs(H_used) >= float(args.H_min)

            sym_count = 0
            for s in range(args.num_syms):
                sym_cp = pay_cp0 + s * Lsym
                Y = extract_ofdm_symbol_fd(rx_cfo, sym_cp, N, CP)
                if Y is None:
                    break

                # Equalize used bins with mask
                Ye = np.zeros_like(Y)
                Y_used = Y[used_idx]
                Ye_used = np.zeros_like(Y_used)
                Ye_used[H_mask] = Y_used[H_mask] / (H_used[H_mask] + eps)
                Ye[used_idx] = Ye_used

                # Pilot-based CPE
                pilots_rx = Ye[pidx]
                pilots_ref = pilot_pattern(s, len(pilot_bins))
                # cpe = angle( sum conj(ref)*rx )
                cpe_inst = np.angle(np.vdot(pilots_ref, pilots_rx))
                cpe_inst_log.append(float(cpe_inst))

                ph = (1 - alpha) * ph + alpha * cpe_inst
                cpe_log.append(float(ph))

                pilot_pow_log.append(float(np.mean(np.abs(pilots_rx) ** 2)))

                # Data constellation pre/post CPE
                data_syms_pre = Ye[didx].copy()
                Ye[used_idx] *= np.exp(-1j * ph)
                data_syms_post = Ye[didx].copy()

                const_pre.append(data_syms_pre)
                const_post.append(data_syms_post)

                # Hard demap + EVM
                hard_bits = qpsk_demap_gray(data_syms_post)
                ref = qpsk_hard_ref_from_bits(hard_bits)
                evm_log.append(float(evm_db(data_syms_post, ref)))

                bits_hat.append(hard_bits)
                sym_count += 1

            if len(bits_hat) == 0:
                print(f"[{t+1:02d}] lock but no payload symbols decoded (CFO={cfo_hz:+.1f}Hz metric={m:.3f})")
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

            # ----------------------------
            # Debug figure per try
            # ----------------------------
            if args.debug_dir:
                # For plots, make sure arrays are non-empty
                cpe_arr = np.array(cpe_log) if len(cpe_log) else np.zeros(1)
                cpe_inst_arr = np.array(cpe_inst_log) if len(cpe_inst_log) else np.zeros(1)
                evm_arr = np.array(evm_log) if len(evm_log) else np.zeros(1)
                pp_arr = np.array(pilot_pow_log) if len(pilot_pow_log) else np.zeros(1)

                # Gather constellations
                cp = np.concatenate(const_pre) if len(const_pre) else np.zeros(1, dtype=np.complex64)
                cpost = np.concatenate(const_post) if len(const_post) else np.zeros(1, dtype=np.complex64)

                # Build a “preamble zoom” view in time domain (magnitude)
                # Show around stf_cp_guess and ltf0_cp (best)
                zoom_len = 6 * Lsym
                a0 = max(0, stf_cp_guess - 2 * Lsym)
                b0 = min(len(rx_cfo), a0 + zoom_len)
                tmag = np.abs(rx_cfo[a0:b0])

                # Channel plots
                Hmag_db = 20 * np.log10(np.abs(H) + 1e-12)
                Hph = np.unwrap(np.angle(H + 1e-12))

                # Residual CFO proxy: derivative of instantaneous CPE (unwrap then diff)
                # If timing good and CFO corrected, this should be near 0 mean.
                cpe_u = np.unwrap(cpe_inst_arr)
                cpe_slope = np.diff(cpe_u) if len(cpe_u) > 2 else np.zeros(1)

                fig = plt.figure(figsize=(18, 12))

                # (1) Coarse metric around peak (stepped)
                ax1 = fig.add_subplot(3, 4, 1)
                M_step = coarse_dbg["M_step"]
                d0 = coarse_dbg["d0"]
                w = 500
                lo = max(0, d0 - w)
                hi = min(len(M_step) - 1, d0 + w)
                xs = np.arange(lo, hi)
                ax1.plot(xs, M_step[lo:hi])
                ax1.axvline(d0, linestyle="--")
                ax1.set_title("Schmidl–Cox metric (coarse, stepped)")
                ax1.set_xlabel("sample index")
                ax1.grid(True)

                # (2) Fine LTF correlation vs offset (FULL symbol)
                ax2 = fig.add_subplot(3, 4, 2)
                ax2.plot(offs, ltf_corr)
                ax2.axvline(0, linestyle="--")
                ax2.set_title("Fine sync: FULL-LTF matched filter vs offset")
                ax2.set_xlabel("offset (samples)")
                ax2.grid(True)

                # (3) Time-domain magnitude around preamble (after CFO corr)
                ax3 = fig.add_subplot(3, 4, 3)
                ax3.plot(np.arange(a0, b0), tmag)
                ax3.axvline(stf_cp_guess, linestyle="--", label="STF cp guess")
                ax3.axvline(ltf0_cp, linestyle="--", label="LTF0 cp chosen")
                ax3.set_title("|rx_cfo| zoom near preamble")
                ax3.set_xlabel("sample index")
                ax3.legend(loc="upper right")
                ax3.grid(True)

                # (4) Channel magnitude
                ax4 = fig.add_subplot(3, 4, 4)
                ax4.plot(np.arange(N), Hmag_db)
                ax4.set_title("Estimated channel |H[k]| (dB)")
                ax4.set_xlabel("FFT bin index (fftshift order)")
                ax4.grid(True)

                # (5) Channel phase
                ax5 = fig.add_subplot(3, 4, 5)
                ax5.plot(np.arange(N), Hph)
                ax5.set_title("Estimated channel ∠H[k] (unwrap)")
                ax5.set_xlabel("FFT bin index (fftshift order)")
                ax5.grid(True)

                # (6) Constellation pre-CPE
                ax6 = fig.add_subplot(3, 4, 6)
                ax6.scatter(np.real(cp), np.imag(cp), s=4, alpha=0.35)
                ax6.set_title("Constellation (equalized, pre-CPE)")
                ax6.grid(True)

                # (7) Constellation post-CPE
                ax7 = fig.add_subplot(3, 4, 7)
                ax7.scatter(np.real(cpost), np.imag(cpost), s=4, alpha=0.35)
                ax7.set_title("Constellation (post-CPE)")
                ax7.grid(True)

                # (8) CPE track (smoothed) + instantaneous
                ax8 = fig.add_subplot(3, 4, 8)
                ax8.plot(np.unwrap(cpe_inst_arr), label="CPE inst (unwrap)", alpha=0.7)
                ax8.plot(np.unwrap(cpe_arr), label="CPE smooth (unwrap)", linewidth=2.0)
                ax8.set_title("Pilot CPE tracking")
                ax8.set_xlabel("OFDM symbol")
                ax8.legend()
                ax8.grid(True)

                # (9) EVM per symbol
                ax9 = fig.add_subplot(3, 4, 9)
                ax9.plot(evm_arr)
                ax9.set_title("EVM per symbol (dB, lower is better)")
                ax9.set_xlabel("OFDM symbol")
                ax9.grid(True)

                # (10) Pilot power per symbol
                ax10 = fig.add_subplot(3, 4, 10)
                ax10.plot(pp_arr)
                ax10.set_title("Pilot power per symbol")
                ax10.set_xlabel("OFDM symbol")
                ax10.grid(True)

                # (11) Residual CFO proxy (CPE slope)
                ax11 = fig.add_subplot(3, 4, 11)
                ax11.plot(cpe_slope)
                ax11.set_title("Residual CFO proxy: Δ(CPE_inst_unwrap)")
                ax11.set_xlabel("OFDM symbol")
                ax11.grid(True)

                # (12) Histogram of post-CPE angles (should cluster into 4)
                ax12 = fig.add_subplot(3, 4, 12)
                ang = np.angle(cpost) if len(cpost) > 1 else np.zeros(1)
                ax12.hist(ang, bins=60)
                ax12.set_title("Angle hist (post-CPE) — expect 4 clusters")
                ax12.grid(True)

                fig.suptitle(
                    f"try={t+1:02d} ok={ok}  CFO={cfo_hz:+.1f}Hz  metric={m:.3f}  "
                    f"syms_decoded={sym_count}  crc_calc={crc_calc:08x} crc_rx={crc_rx:08x}"
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                out = os.path.join(args.debug_dir, f"rx_dbg_v2_try{t+1:02d}_{'OK' if ok else 'FAIL'}.png")
                fig.savefig(out, dpi=140)
                plt.close(fig)

            # Print status
            if ok:
                open(args.outfile, "wb").write(payload)
                print(f"[{t+1:02d}] ✅ CRC OK! wrote {len(payload)} bytes -> {args.outfile}")
                packet_found = True
                break
            else:
                print(f"[{t+1:02d}] CRC fail (CFO={cfo_hz:+.1f} Hz metric={m:.3f}) calc={crc_calc:08x} rx={crc_rx:08x}")

    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

    if not packet_found:
        print("No valid packet recovered.")
        print("Next knobs to try (in order):")
        print("  1) Increase SNR: closer antennas / coax+atten / raise TX gain slightly (avoid clipping).")
        print("  2) Increase rx_buffer_size so it always captures a full packet.")
        print("  3) Increase --ltf_search (e.g., 400) if fine peak still ambiguous.")
        print("  4) Increase --repeat (already 4 max here) or reduce bandwidth.")
        print("  5) If post-CPE still rings: add residual CFO tracking (FLL) using pilot slope.")


if __name__ == "__main__":
    main()