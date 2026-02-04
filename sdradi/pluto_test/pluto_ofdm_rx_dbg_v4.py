#!/usr/bin/env python3
import argparse, os, zlib
import numpy as np
import adi
import matplotlib.pyplot as plt

MAGIC = b"AIS1"

def make_subcarrier_plan(N: int):
    if N != 64:
        raise ValueError("N=64 only.")
    pilots = np.array([-21, -7, 7, 21], dtype=int)
    used = np.r_[np.arange(-26, 0), np.arange(1, 27)]
    data = np.array([k for k in used if k not in set(pilots)], dtype=int)
    return data, pilots, used

def ofdm_symbol_from_bins(X_shift: np.ndarray, N: int, CP: int) -> np.ndarray:
    x = np.fft.ifft(np.fft.ifftshift(X_shift)).astype(np.complex64)
    return np.r_[x[-CP:], x]

def pilot_pattern(sym_idx: int):
    base = np.array([1,1,1,-1], dtype=np.float32)
    if sym_idx % 2 == 1:
        base = -base
    return base.astype(np.complex64)

def build_preamble(N: int, CP: int, data_bins, pilot_bins):
    used = np.r_[data_bins, pilot_bins]

    Xstf = np.zeros(N, dtype=np.complex64)
    rng = np.random.default_rng(0)
    bpsk = rng.choice([-1.0, 1.0], size=len(used)).astype(np.float32)
    Xstf[(used + N//2) % N] = bpsk + 0j
    stf_sym = ofdm_symbol_from_bins(Xstf, N, CP)
    STF = np.r_[stf_sym, stf_sym].astype(np.complex64)

    Xltf = np.zeros(N, dtype=np.complex64)
    ltf_bpsk = np.ones(len(used), dtype=np.float32)
    ltf_bpsk[::2] = -1.0
    Xltf[(used + N//2) % N] = ltf_bpsk + 0j
    ltf_sym = ofdm_symbol_from_bins(Xltf, N, CP)
    LTF = np.r_[ltf_sym, ltf_sym].astype(np.complex64)

    PRE = np.r_[STF, LTF].astype(np.complex64)
    return STF, LTF, PRE, Xltf, used

def qpsk_demap_gray(syms: np.ndarray) -> np.ndarray:
    bits = np.zeros((len(syms), 2), dtype=np.uint8)
    re = np.real(syms) >= 0
    im = np.imag(syms) >= 0
    for i in range(len(syms)):
        if re[i] and im[i]:
            bits[i] = [0,0]
        elif (not re[i]) and im[i]:
            bits[i] = [0,1]
        elif (not re[i]) and (not im[i]):
            bits[i] = [1,1]
        else:
            bits[i] = [1,0]
    return bits.reshape(-1)

def majority_vote(bits_rep: np.ndarray, rep: int) -> np.ndarray:
    if rep <= 1:
        return bits_rep.astype(np.uint8)
    L = (len(bits_rep)//rep)*rep
    bits_rep = bits_rep[:L].reshape(-1, rep)
    return (np.sum(bits_rep, axis=1) >= (rep/2)).astype(np.uint8)

def extract_fd(rx: np.ndarray, sym_cp: int, N: int, CP: int):
    a = sym_cp + CP
    b = a + N
    if a < 0 or b > len(rx):
        return None
    td = rx[a:b]
    return np.fft.fftshift(np.fft.fft(td))

def preamble_corr_scan(rx: np.ndarray, pre: np.ndarray, start: int, stop: int, step: int):
    Lp = len(pre)
    pnorm = np.sqrt(np.sum(np.abs(pre)**2)) + 1e-12
    xs, cs = [], []
    for s in range(start, stop, step):
        if s < 0 or (s+Lp) > len(rx):
            continue
        seg = rx[s:s+Lp]
        sn = np.sqrt(np.sum(np.abs(seg)**2)) + 1e-12
        c = np.abs(np.vdot(seg, pre)) / (sn*pnorm)
        xs.append(s); cs.append(float(c))
    return np.array(xs, dtype=int), np.array(cs, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=1e6)
    ap.add_argument("--bw", type=float, default=1.2e6)
    ap.add_argument("--rx_gain", type=float, default=55.0)
    ap.add_argument("--buf", type=int, default=131072)
    ap.add_argument("--repeat", type=int, default=1, choices=[1,2,4])
    ap.add_argument("--num_syms", type=int, default=300)
    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--debug_dir", default="rx_debug_ofdm64")
    ap.add_argument("--outfile", default="recovered.bin")
    ap.add_argument("--scan_step", type=int, default=2)
    ap.add_argument("--scan_stride", type=int, default=4, help="coarse scan stride in samples for speed")
    ap.add_argument("--kp", type=float, default=0.08)
    ap.add_argument("--kf", type=float, default=0.0015)
    args = ap.parse_args()

    N, CP = 64, 16
    Lsym = N + CP

    data_bins, pilot_bins, used_bins = make_subcarrier_plan(N)
    didx = (data_bins + N//2) % N
    pidx = (pilot_bins + N//2) % N
    used_idx = ((np.r_[data_bins, pilot_bins] + N//2) % N).astype(int)

    STF, LTF, PRE, Xltf, used_all = build_preamble(N, CP, data_bins, pilot_bins)
    Lp = len(PRE)

    os.makedirs(args.debug_dir, exist_ok=True)

    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf)

    for _ in range(2):
        _ = sdr.rx()

    print("RX running")
    print(f"  uri={args.uri} fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz gain={args.rx_gain}dB buf={args.buf}")
    print(f"  repeat={args.repeat} payload_syms={args.num_syms}")

    try:
        for t in range(args.tries):
            rx_raw = sdr.rx().astype(np.complex64) / (2**14)
            rx = rx_raw - np.mean(rx_raw)

            # coarse scan: try multiple start positions to find best preamble corr
            best_s = None
            best_c = -1.0
            best_trace = None

            for s0 in range(0, len(rx) - Lp, args.scan_stride):
                seg = rx[s0:s0+Lp]
                if len(seg) < Lp:
                    break
                # cheap energy gate
                if np.mean(np.abs(seg)) < 1e-3:
                    continue
                c = np.abs(np.vdot(seg, PRE)) / ((np.linalg.norm(seg)*np.linalg.norm(PRE)) + 1e-12)
                if c > best_c:
                    best_c = float(c)
                    best_s = s0

            if best_s is None:
                print(f"[{t+1:02d}] no candidate")
                continue

            # refine scan around best_s
            xs, cs = preamble_corr_scan(rx, PRE, best_s-2000, best_s+2000, step=args.scan_step)
            if len(cs) == 0:
                print(f"[{t+1:02d}] refine failed")
                continue
            stf_cp = int(xs[int(np.argmax(cs))])
            pre_corr = float(np.max(cs))
            best_trace = (xs, cs)

            # CFO from STF repetition (2 identical symbols)
            # Use the DATA parts of two STF symbols separated by Lsym
            a0 = stf_cp + CP
            a1 = a0 + Lsym
            if a1 + N > len(rx):
                print(f"[{t+1:02d}] short buffer after stf_cp")
                continue
            d0 = rx[a0:a0+N]
            d1 = rx[a1:a1+N]
            P = np.sum(d0 * np.conj(d1))
            cfo = -(np.angle(P) * args.fs) / (2*np.pi*Lsym)

            n = np.arange(len(rx), dtype=np.float32)
            rx_cfo = rx * np.exp(-1j * 2*np.pi*cfo*n/args.fs)

            # Channel from LTF (2 symbols)
            ltf0 = stf_cp + 2*Lsym
            ltf1 = ltf0 + Lsym
            Y0 = extract_fd(rx_cfo, ltf0, N, CP)
            Y1 = extract_fd(rx_cfo, ltf1, N, CP)
            if Y0 is None or Y1 is None:
                print(f"[{t+1:02d}] no LTF")
                continue
            Yltf = 0.5*(Y0+Y1)

            H = np.ones(N, dtype=np.complex64)
            eps = 1e-9
            used_ltf_idx = ((used_all + N//2) % N).astype(int)
            H[used_ltf_idx] = Yltf[used_ltf_idx] / (Xltf[used_ltf_idx] + eps)

            # Payload demod with FLL+PLL on pilot CPE
            pay0 = stf_cp + len(PRE)

            ph_acc = 0.0
            freq_acc = 0.0

            bits_rep = []
            pilot_pow = []
            evm = []
            const_post = []
            ph_err_log = []
            freq_log = []

            for s in range(args.num_syms):
                sym_cp = pay0 + s*Lsym
                Y = extract_fd(rx_cfo, sym_cp, N, CP)
                if Y is None:
                    break

                Ye = np.zeros_like(Y)
                Ye[used_idx] = Y[used_idx] / (H[used_idx] + eps)

                # pilot error
                p = Ye[pidx]
                pref = pilot_pattern(s)
                e = np.sum(p * np.conj(pref))
                ph_err = np.angle(e)

                freq_acc += args.kf * ph_err
                ph_acc += freq_acc + args.kp * ph_err

                Ye[used_idx] *= np.exp(-1j*ph_acc)

                d = Ye[didx]
                const_post.append(d)
                pilot_pow.append(float(np.mean(np.abs(p)**2)))
                ph_err_log.append(float(ph_err))
                freq_log.append(float(freq_acc))

                bits_rep.append(qpsk_demap_gray(d))

            if len(bits_rep) == 0:
                print(f"[{t+1:02d}] no payload")
                continue

            bits_rep = np.concatenate(bits_rep).astype(np.uint8)
            bits = majority_vote(bits_rep, args.repeat)
            bb = np.packbits(bits).tobytes()

            # parse frame
            ok = False
            payload = b""
            crc_calc = 0
            crc_rx = 0
            if len(bb) >= 6:
                if bb[:4] == MAGIC:
                    plen = int.from_bytes(bb[4:6], "little")
                    need = 6 + plen + 4
                    if len(bb) >= need:
                        payload = bb[6:6+plen]
                        crc_rx = int.from_bytes(bb[6+plen:6+plen+4], "little")
                        crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
                        ok = (crc_calc == crc_rx)

            # debug plot
            cpst = np.concatenate(const_post) if len(const_post) else np.zeros(1, np.complex64)
            xs2, cs2 = best_trace

            fig = plt.figure(figsize=(18,10))
            ax1 = fig.add_subplot(2,3,1)
            ax1.plot(xs2, cs2); ax1.axvline(stf_cp, ls="--")
            ax1.set_title("Preamble corr (refined)"); ax1.grid(True)

            ax2 = fig.add_subplot(2,3,2)
            ax2.scatter(np.real(cpst), np.imag(cpst), s=4, alpha=0.35)
            ax2.set_title("Constellation post-loop"); ax2.grid(True)

            ax3 = fig.add_subplot(2,3,3)
            ax3.plot(pilot_pow)
            ax3.set_title("Pilot power per symbol"); ax3.grid(True)

            ax4 = fig.add_subplot(2,3,4)
            ax4.plot(np.unwrap(np.array(ph_err_log)))
            ax4.set_title("Pilot phase error (unwrap)"); ax4.grid(True)

            ax5 = fig.add_subplot(2,3,5)
            ax5.plot(freq_log)
            ax5.set_title("Residual CFO integrator (rad/sym)"); ax5.grid(True)

            ax6 = fig.add_subplot(2,3,6)
            ax6.hist(np.angle(cpst), bins=60)
            ax6.set_title("Angle histogram (QPSK expect 4 peaks)"); ax6.grid(True)

            fig.suptitle(f"try={t+1:02d} ok={ok} pre_corr={pre_corr:.3f} CFO={cfo:+.1f}Hz crc_calc={crc_calc:08x} crc_rx={crc_rx:08x}")
            fig.tight_layout(rect=[0,0,1,0.95])
            outp = os.path.join(args.debug_dir, f"rx_ofdm_try{t+1:02d}_{'OK' if ok else 'FAIL'}.png")
            fig.savefig(outp, dpi=140)
            plt.close(fig)

            if ok:
                open(args.outfile, "wb").write(payload)
                print(f"[{t+1:02d}] ✅ CRC OK payload_len={len(payload)} -> {args.outfile}")
                break
            else:
                print(f"[{t+1:02d}] CRC fail pre_corr={pre_corr:.3f} CFO={cfo:+.1f}Hz (saved {outp})")

        else:
            print("No valid packet recovered.")

    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

if __name__ == "__main__":
    main()

"""
python3 pluto_ofdm_rx_dbg_v4.py \
  --uri "usb:1.39.5" \
  --fc 2.3e9 --fs 1e6 --bw 1.2e6 \
  --rx_gain 55 \
  --buf 262144 \
  --repeat 1 --num_syms 400 \
  --tries 20 \
  --outfile recovered_payload.bin \
  --debug_dir rx_debug_ofdm64
"""