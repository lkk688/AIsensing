#!/usr/bin/env python3
import argparse
import os
import zlib
import numpy as np
import adi
import matplotlib.pyplot as plt

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

def schmidl_cox_metric(rx: np.ndarray, N: int, step: int = 1):
    """
    Classic Schmidl-Cox for repeated block of length N (we correlate N vs next N).
    Returns metric array M[d], complex P[d], and power R[d] for d in [0..len-2N).
    """
    L = len(rx)
    Lm = L - 2*N
    if Lm <= 0:
        return None, None, None
    M = np.zeros(Lm, dtype=np.float32)
    P = np.zeros(Lm, dtype=np.complex64)
    R = np.zeros(Lm, dtype=np.float32)

    eps = 1e-12
    # For debugging clarity, do a stepped loop (fast enough for 1-2^20 buffers)
    for d in range(0, Lm, step):
        seg1 = rx[d:d+N]
        seg2 = rx[d+N:d+2*N]
        Pd = np.vdot(seg2, seg1)   # sum(conj(seg2)*seg1)
        Rd = np.sum(np.abs(seg2)**2) + eps
        P[d] = Pd
        R[d] = Rd
        # normalized metric (0..1-ish), but can exceed 1 if very clean + scaling differences
        M[d] = (np.abs(Pd)**2) / (Rd**2)
    return M, P, R

def find_coarse_start(rx: np.ndarray, N: int):
    # coarse: compute metric in steps, pick max, refine nearby
    step = 4
    M, P, _ = schmidl_cox_metric(rx, N, step=step)
    if M is None:
        return None, None, None, None
    d0 = int(np.argmax(M))
    # refine in +/-16
    lo = max(0, d0 - 16)
    hi = min(len(rx) - 2*N - 1, d0 + 16)
    M2, P2, _ = schmidl_cox_metric(rx[lo:hi+2*N+1], N, step=1)
    if M2 is None:
        return d0, P[d0], M[d0], (M, P, d0)
    d1_rel = int(np.argmax(M2))
    d1 = lo + d1_rel
    return d1, P2[d1_rel], M2[d1_rel], (M, P, d0)

def fine_sync_with_ltf(rx_cfo: np.ndarray, ltf_cp_guess: int, ltf_td_ref: np.ndarray, N: int, CP: int, search: int = 40):
    """
    Search around ltf_cp_guess for the CP-start that best correlates with known LTF time-domain (data part).
    """
    best = None
    best_val = -1
    corrs = []
    offsets = list(range(-search, search+1))
    for off in offsets:
        cp = ltf_cp_guess + off
        seg = extract_ofdm_symbol_td(rx_cfo, cp, N, CP)
        if seg is None:
            corrs.append(0.0)
            continue
        v = np.abs(np.vdot(seg, ltf_td_ref))
        corrs.append(float(v))
        if v > best_val:
            best_val = v
            best = cp
    return best, np.array(offsets), np.array(corrs)

def evm_db(rx_syms: np.ndarray, ref_syms: np.ndarray):
    # EVM = sqrt(E[|e|^2]/E[|ref|^2])
    e = rx_syms - ref_syms
    num = np.mean(np.abs(e)**2) + 1e-12
    den = np.mean(np.abs(ref_syms)**2) + 1e-12
    evm = np.sqrt(num/den)
    return 20*np.log10(evm + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=1e6)
    ap.add_argument("--bw", type=float, default=1.2e6)
    ap.add_argument("--rx_gain", type=float, default=55.0)
    ap.add_argument("--buf", type=int, default=2**20)
    ap.add_argument("--repeat", type=int, default=4, choices=[1,2,4])
    ap.add_argument("--num_syms", type=int, default=300)
    ap.add_argument("--expect_len", type=int, default=2048)  # bytes before CRC
    ap.add_argument("--outfile", default="recovered.jpg")
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--debug_dir", default="", help="If set, save debug plots here.")
    args = ap.parse_args()

    N, CP = 64, 16
    data_bins, pilot_bins, used_bins = make_subcarrier_plan(N)
    used_idx = (used_bins + N//2) % N
    pidx = (pilot_bins + N//2) % N
    didx = (data_bins + N//2) % N

    # Known LTF in freq (must match TX)
    Xltf = np.zeros(N, dtype=np.complex64)
    ltf_bpsk = np.ones(len(used_bins), dtype=np.float32)
    ltf_bpsk[::2] = -1.0
    Xltf[used_idx] = ltf_bpsk + 0j

    # LTF time-domain reference (data part only)
    ltf_td = np.fft.ifft(np.fft.ifftshift(Xltf)).astype(np.complex64)

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

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    for t in range(args.tries):
        rx_raw = sdr.rx().astype(np.complex64) / (2**14)
        rx = rx_raw - np.mean(rx_raw)  # DC removal

        # coarse detect
        d, Pd, m, coarse_dbg = find_coarse_start(rx, N)
        if d is None:
            print(f"[{t+1:02d}] no lock")
            continue

        # CFO estimate
        cfo_hz = (np.angle(Pd) * args.fs) / (2*np.pi*N)
        n = np.arange(len(rx), dtype=np.float32)
        rx_cfo = rx * np.exp(-1j * 2*np.pi * cfo_hz * n / args.fs)

        # Packet layout must match TX:
        # zeros gap, STF has 2*(N+CP), LTF has 2*(N+CP)
        # Our Schmidl-Cox d is near start of the repeated STF content (not perfect CP-aligned).
        stf_guess = d
        stf_cp_guess = max(0, stf_guess - CP)

        ltf0_cp_guess = stf_cp_guess + 2*(N+CP)
        # Fine sync using LTF correlation
        ltf0_cp, offs, ltf_corr = fine_sync_with_ltf(rx_cfo, ltf0_cp_guess, ltf_td, N, CP, search=48)
        if ltf0_cp is None:
            print(f"[{t+1:02d}] lock but fine LTF sync failed (cfo={cfo_hz:+.1f} Hz metric={m:.3f})")
            continue
        ltf1_cp = ltf0_cp + (N+CP)

        Y0 = extract_ofdm_symbol_fd(rx_cfo, ltf0_cp, N, CP)
        Y1 = extract_ofdm_symbol_fd(rx_cfo, ltf1_cp, N, CP)
        if Y0 is None or Y1 is None:
            print(f"[{t+1:02d}] lock but insufficient samples around LTF")
            continue

        Yltf = 0.5*(Y0 + Y1)
        H = np.ones(N, dtype=np.complex64)
        eps = 1e-9
        H[used_idx] = Yltf[used_idx] / (Xltf[used_idx] + eps)

        # payload start after LTF1
        pay_cp0 = ltf1_cp + (N+CP)

        bits_hat = []
        cpe_log = []
        evm_log = []
        const_pre = []
        const_post = []

        ph = 0.0
        alpha = 0.2  # a bit stronger than before

        for s in range(args.num_syms):
            sym_cp = pay_cp0 + s*(N+CP)
            Y = extract_ofdm_symbol_fd(rx_cfo, sym_cp, N, CP)
            if Y is None:
                break

            # Equalize on used bins
            Ye = np.zeros_like(Y)
            Ye[used_idx] = Y[used_idx] / (H[used_idx] + eps)

            # Pilot-based CPE
            pilots_rx = Ye[pidx]
            pilots_ref = pilot_pattern(s, len(pilot_bins))
            # Estimate common phase between rx and ref
            cpe = np.angle(np.vdot(pilots_ref, pilots_rx))
            ph = (1 - alpha)*ph + alpha*cpe
            cpe_log.append(float(ph))

            # Constellation before/after CPE
            data_syms_pre = Ye[didx].copy()
            Ye[used_idx] *= np.exp(-1j * ph)
            data_syms_post = Ye[didx].copy()

            const_pre.append(data_syms_pre)
            const_post.append(data_syms_post)

            # EVM vs ideal QPSK points (nearest hard decision)
            hard_bits = qpsk_demap_gray(data_syms_post)
            # Map bits back to symbols for EVM reference
            # (same Gray map as TX)
            ref = np.empty(len(data_syms_post), dtype=np.complex64)
            hb = hard_bits.reshape(-1, 2)
            for i, (b0, b1) in enumerate(hb):
                if b0 == 0 and b1 == 0: ref[i] = 1+1j
                elif b0 == 0 and b1 == 1: ref[i] = -1+1j
                elif b0 == 1 and b1 == 1: ref[i] = -1-1j
                else: ref[i] = 1-1j
            ref /= np.sqrt(2)
            evm_log.append(float(evm_db(data_syms_post, ref)))

            bits_hat.append(hard_bits)

        if len(bits_hat) == 0:
            print(f"[{t+1:02d}] lock but no payload symbols decoded (cfo={cfo_hz:+.1f} Hz metric={m:.3f})")
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

        # Debug plots
        if args.debug_dir:
            fig = plt.figure(figsize=(14, 10))
            ax1 = fig.add_subplot(2,3,1)
            ax2 = fig.add_subplot(2,3,2)
            ax3 = fig.add_subplot(2,3,3)
            ax4 = fig.add_subplot(2,3,4)
            ax5 = fig.add_subplot(2,3,5)
            ax6 = fig.add_subplot(2,3,6)

            # (1) coarse timing metric around peak
            M_step, P_step, d0 = coarse_dbg
            dpk = d0
            w = 400
            lo = max(0, dpk-w)
            hi = min(len(M_step)-1, dpk+w)
            xs = np.arange(lo, hi)
            ax1.plot(xs, M_step[lo:hi])
            ax1.axvline(dpk, linestyle="--")
            ax1.set_title("Schmidl-Cox metric (coarse, stepped)")

            # (2) fine LTF correlation sweep
            ax2.plot(offs, ltf_corr)
            ax2.axvline(0, linestyle="--")
            ax2.set_title("Fine sync: LTF correlation vs offset")

            # (3) channel magnitude
            ax3.plot(np.arange(N), 20*np.log10(np.abs(H)+1e-12))
            ax3.set_title("Estimated channel |H[k]| (dB)")

            # (4) constellation pre-CPE
            cp = np.concatenate(const_pre)
            ax4.scatter(np.real(cp), np.imag(cp), s=4, alpha=0.4)
            ax4.set_title("Constellation (equalized, pre-CPE)")
            ax4.grid(True)

            # (5) constellation post-CPE
            cpost = np.concatenate(const_post)
            ax5.scatter(np.real(cpost), np.imag(cpost), s=4, alpha=0.4)
            ax5.set_title("Constellation (post-CPE)")
            ax5.grid(True)

            # (6) phase + EVM
            ax6.plot(np.unwrap(np.array(cpe_log)), label="CPE (rad)")
            ax6b = ax6.twinx()
            ax6b.plot(np.array(evm_log), linestyle="--", label="EVM (dB)")
            ax6.set_title("CPE track and EVM per symbol")

            fig.suptitle(f"try={t+1:02d} ok={ok}  CFO={cfo_hz:+.1f}Hz  metric={m:.3f}  crc_calc={crc_calc:08x} crc_rx={crc_rx:08x}")
            fig.tight_layout(rect=[0,0,1,0.95])
            out = os.path.join(args.debug_dir, f"rx_dbg_try{t+1:02d}_{'OK' if ok else 'FAIL'}.png")
            fig.savefig(out, dpi=140)
            plt.close(fig)

        if ok:
            open(args.outfile, "wb").write(payload)
            print(f"[{t+1:02d}] ✅ CRC OK! wrote {len(payload)} bytes -> {args.outfile}")
            break
        else:
            print(f"[{t+1:02d}] CRC fail (cfo={cfo_hz:+.1f} Hz metric={m:.3f}) calc={crc_calc:08x} rx={crc_rx:08x}")

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

if __name__ == "__main__":
    main()