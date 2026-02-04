#!/usr/bin/env python3
"""
pluto_zc_ofdm32_rx_dbg.py

Receiver that matches the user's existing TX structure:

TX (as provided by user):
  - N=32, CP=32  => symbol length = 64
  - ZC preamble: time-domain zc_td (length N) repeated 40 times (NO CP)
      header = [ifft(zc)*5.6] repeated 40
  - payload: 250 symbols, each symbol is [CP(32) + data(32)] where CP is last 32 of time-domain OFDM symbol
      payload.append(concat(ifft(X)[-32:], ifft(X)))
  - data bins: [-12, -11, 11, 12]
  - pilots: [-14, 14] (very large amplitude in TX)
  - modulation: DQPSK phase increments (binary mapping):
      k = (b0*2 + b1) in {0,1,2,3}
      delta_phase = k * (pi/2)
      per-subcarrier phase accumulates over time

This RX does:
  1) Capture IQ, DC remove
  2) Coarse timing: correlate with zc_td and detect best header region
  3) Coarse CFO from repeated ZC segments (average across many repeats)
  4) Refined timing: choose the best boundary aligned to N (32)
  5) Demod payload symbols:
      - remove CP(32), FFT(32), fftshift
      - use pilots to estimate common phase error (CPE) per symbol
      - optional residual CFO tracking from pilot CPE slope (simple FLL)
      - DQPSK differential decode across symbols per subcarrier
  6) CRC check if expected length is provided

Debug outputs:
  - corr curve around detected header
  - CFO estimate
  - constellation (pre/post CPE)
  - pilot CPE over time
  - residual CFO estimate over time
  - DQPSK angle histogram (should show 4 peaks)
  - pilot power / SNR proxy
"""

import argparse
import os
import zlib
import numpy as np
import adi
import matplotlib.pyplot as plt


def zc_time_domain(N: int, root: int = 17, scale: float = 5.6):
    n = np.arange(N)
    zc = np.exp(-1j * np.pi * root * n * (n + 1) / N)
    td = np.fft.ifft(zc).astype(np.complex64) * scale
    return td


def fft_bin_idx(k: int, N: int):
    # fftshift indexing: bins are [-N/2..N/2-1]
    return (k + N // 2) % N


def dqpsk_bits_from_phase_delta(phi: np.ndarray):
    """
    Binary mapping consistent with your TX:
      k = round(phi / (pi/2)) mod 4
      bits = [k>>1, k&1]
    """
    # wrap to [0, 2pi)
    phi = (phi + 2*np.pi) % (2*np.pi)
    k = np.floor((phi + (np.pi/4)) / (np.pi/2)).astype(int) % 4
    b0 = (k >> 1) & 1
    b1 = k & 1
    out = np.empty((len(k), 2), dtype=np.uint8)
    out[:, 0] = b0
    out[:, 1] = b1
    return out.reshape(-1)


def majority_vote(bits_rep: np.ndarray, rep: int):
    if rep <= 1:
        return bits_rep.astype(np.uint8)
    L = (len(bits_rep) // rep) * rep
    bits_rep = bits_rep[:L].reshape(-1, rep)
    return (np.sum(bits_rep, axis=1) >= (rep/2)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=1e6)
    ap.add_argument("--bw", type=float, default=1.2e6)
    ap.add_argument("--rx_gain", type=float, default=55.0)
    ap.add_argument("--buf", type=int, default=2**20)

    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--CP", type=int, default=32)
    ap.add_argument("--zc_root", type=int, default=17)
    ap.add_argument("--zc_reps", type=int, default=40)

    ap.add_argument("--num_syms", type=int, default=250)
    ap.add_argument("--repeat_bits", type=int, default=1, choices=[1,2,4], help="optional repetition coding at bit level")
    ap.add_argument("--expect_len", type=int, default=2048, help="bytes BEFORE CRC32; set 0 to skip CRC check")
    ap.add_argument("--outfile", default="recovered.jpg")

    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--debug_dir", default="rx_debug_zc")
    ap.add_argument("--corr_search_start", type=int, default=0)
    ap.add_argument("--corr_search_stop", type=int, default=0, help="0 means full buffer")

    # tracking knobs
    ap.add_argument("--cpe_alpha", type=float, default=0.25, help="pilot CPE smoothing")
    ap.add_argument("--fll_beta", type=float, default=0.02, help="residual CFO loop gain on CPE slope")
    args = ap.parse_args()

    N = args.N
    CP = args.CP
    Lsym = N + CP

    DATA_SC = np.array([-12, -11, 11, 12], dtype=int)
    PILOT_SC = np.array([-14, 14], dtype=int)

    didx = np.array([fft_bin_idx(k, N) for k in DATA_SC], dtype=int)
    pidx = np.array([fft_bin_idx(k, N) for k in PILOT_SC], dtype=int)

    # Known ZC time-domain ref
    zc_td = zc_time_domain(N, root=args.zc_root, scale=5.6)

    # SDR setup
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

    os.makedirs(args.debug_dir, exist_ok=True)

    print("RX(ZC/OFDM32/DQPSK) running")
    print(f"  uri={args.uri} fc={args.fc/1e6:.3f} MHz fs={args.fs/1e6:.3f} Msps bw={args.bw/1e6:.3f} MHz gain={args.rx_gain} dB")
    print(f"  N={N} CP={CP} header_zc_reps={args.zc_reps} payload_syms={args.num_syms}")

    for t in range(args.tries):
        rx_raw = sdr.rx().astype(np.complex64) / (2**14)
        rx = rx_raw - np.mean(rx_raw)  # DC removal

        # --- 1) Correlation search for ZC header (length N) ---
        start = args.corr_search_start
        stop = args.corr_search_stop if args.corr_search_stop > 0 else (len(rx) - N - 1)
        if stop <= start + N + 1:
            print(f"[{t+1:02d}] buffer too short for correlation search")
            continue

        # compute correlation magnitude (valid)
        # use vdot for each shift (fast enough at 1e6 samples? we restrict range)
        corr = np.zeros(stop - start, dtype=np.float32)
        zc_conj = np.conj(zc_td)

        # step a bit for speed, then refine
        step = 2
        for i, idx in enumerate(range(start, stop, step)):
            seg = rx[idx:idx+N]
            if len(seg) < N:
                break
            corr[i] = np.abs(np.vdot(zc_conj, seg))  # vdot conjugates first arg -> conj(zc_conj)=zc, not desired
        # fix: do explicit sum(seg * conj(zc))
        corr = np.zeros(((stop-start)//step)+1, dtype=np.float32)
        xs = np.zeros_like(corr, dtype=int)
        j = 0
        for idx in range(start, stop, step):
            seg = rx[idx:idx+N]
            if len(seg) < N:
                break
            corr[j] = np.abs(np.sum(seg * np.conj(zc_td)))
            xs[j] = idx
            j += 1
        corr = corr[:j]
        xs = xs[:j]

        if len(corr) == 0:
            print(f"[{t+1:02d}] no corr computed")
            continue

        kmax = int(np.argmax(corr))
        coarse_idx = int(xs[kmax])

        # --- 2) Refine peak in a small window step=1 ---
        ref_lo = max(start, coarse_idx - 200)
        ref_hi = min(stop, coarse_idx + 200)
        corr2 = []
        xs2 = []
        for idx in range(ref_lo, ref_hi):
            seg = rx[idx:idx+N]
            if len(seg) < N:
                break
            corr2.append(np.abs(np.sum(seg * np.conj(zc_td))))
            xs2.append(idx)
        corr2 = np.array(corr2, dtype=np.float32)
        xs2 = np.array(xs2, dtype=int)
        fine_idx = int(xs2[int(np.argmax(corr2))])

        # Align to ZC repetition grid: choose the boundary such that we can take zc_reps segments
        # We search a few offsets to maximize the sum corr across zc_reps segments
        best0 = None
        bestv = -1.0
        for off in range(-N, N+1):
            s0 = fine_idx + off
            if s0 < 0 or (s0 + args.zc_reps*N) > len(rx):
                continue
            v = 0.0
            for r in range(args.zc_reps):
                seg = rx[s0 + r*N : s0 + (r+1)*N]
                v += np.abs(np.sum(seg * np.conj(zc_td)))
            if v > bestv:
                bestv = v
                best0 = s0

        if best0 is None:
            print(f"[{t+1:02d}] header grid align failed")
            continue

        header_start = best0

        # --- 3) CFO estimate from repeated ZC segments ---
        # Use phase difference between consecutive ZC segments:
        # P = sum seg0 * conj(seg1) ~ exp(-j 2pi CFO N/fs)
        # CFO = -angle(P)*fs/(2pi*N)
        Pacc = 0+0j
        pairs = min(args.zc_reps - 1, 30)  # average first 30 pairs
        for r in range(pairs):
            s0 = rx[header_start + r*N : header_start + (r+1)*N]
            s1 = rx[header_start + (r+1)*N : header_start + (r+2)*N]
            if len(s0) < N or len(s1) < N:
                break
            Pacc += np.sum(s0 * np.conj(s1))
        cfo_hz = -(np.angle(Pacc) * args.fs) / (2*np.pi*N)

        n = np.arange(len(rx), dtype=np.float32)
        rx_cfo = rx * np.exp(-1j * 2*np.pi * cfo_hz * n / args.fs)

        # --- 4) Payload start ---
        payload_start = header_start + args.zc_reps * N  # immediately after ZC header (no CP in header)

        # Ensure we have enough samples for payload
        need = payload_start + args.num_syms * Lsym
        if need > len(rx_cfo):
            print(f"[{t+1:02d}] insufficient samples for full payload (have={len(rx_cfo)}, need={need})")
            continue

        # --- 5) Demod loop ---
        bits_out = []
        last_syms = None

        # logs
        const_pre = []
        const_post = []
        cpe_inst_log = []
        cpe_smooth_log = []
        res_cfo_log = []
        pilot_pow_log = []

        cpe_smooth = 0.0
        # residual CFO integrator (rad/symbol)
        w_sym = 0.0

        for s in range(args.num_syms):
            sym_cp = payload_start + s * Lsym

            # remove CP=32, take N=32
            td = rx_cfo[sym_cp + CP : sym_cp + CP + N]
            if len(td) < N:
                break

            Y = np.fft.fftshift(np.fft.fft(td))

            # pilots
            p = Y[pidx]
            pilot_pow = float(np.mean(np.abs(p)**2))
            pilot_pow_log.append(pilot_pow)

            # CPE estimate: use mean pilot phase (robust even if amplitude differs)
            cpe_inst = np.angle(np.mean(p))
            cpe_inst_log.append(float(cpe_inst))

            # simple FLL: adjust w_sym using slope of CPE
            if s > 0:
                dphi = np.angle(np.exp(1j*(cpe_inst_log[-1] - cpe_inst_log[-2])))
                w_sym += args.fll_beta * dphi
            res_cfo_log.append(float(w_sym))

            # smooth CPE with alpha, and include w_sym
            cpe_smooth = (1 - args.cpe_alpha)*cpe_smooth + args.cpe_alpha*cpe_inst
            cpe_total = cpe_smooth + w_sym * s
            cpe_smooth_log.append(float(cpe_smooth))

            # data
            data_pre = Y[didx].copy()
            data_post = data_pre * np.exp(-1j * cpe_total)

            const_pre.append(data_pre)
            const_post.append(data_post)

            # DQPSK differential decode
            if last_syms is not None:
                diff = data_post * np.conj(last_syms)
                phi = np.angle(diff)
                bits_sym = dqpsk_bits_from_phase_delta(phi)
                bits_out.append(bits_sym)

            last_syms = data_post

        if len(bits_out) == 0:
            print(f"[{t+1:02d}] decoded 0 bits")
            continue

        bits_out = np.concatenate(bits_out).astype(np.uint8)

        # Optional repetition coding (if you used it at TX; default 1)
        bits_dec = majority_vote(bits_out, args.repeat_bits)

        bb = np.packbits(bits_dec).tobytes()

        ok = False
        crc_calc = 0
        crc_rx = 0

        if args.expect_len > 0:
            total = args.expect_len + 4
            bb2 = bb[:total]
            if len(bb2) >= total:
                payload = bb2[:-4]
                crc_rx = int.from_bytes(bb2[-4:], "little")
                crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
                ok = (crc_calc == crc_rx)
            else:
                payload = bb2
        else:
            payload = bb

        # --- 6) Debug figures ---
        cp = np.concatenate(const_pre) if len(const_pre) else np.zeros(1, dtype=np.complex64)
        cpost = np.concatenate(const_post) if len(const_post) else np.zeros(1, dtype=np.complex64)

        # Angle histogram after CPE (should show 4 peaks if DQPSK diff works)
        # For DQPSK, angle peaks appear in diff angles; we also plot diff hist:
        if len(const_post) > 2:
            dp = []
            last = None
            for sym in const_post:
                if last is not None:
                    dp.append(np.angle(sym * np.conj(last)))
                last = sym
            dp = np.concatenate(dp) if len(dp) else np.zeros(1)
        else:
            dp = np.zeros(1)

        fig = plt.figure(figsize=(18, 10))

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(xs2, corr2)
        ax1.axvline(header_start, linestyle="--", label="header_start")
        ax1.set_title("ZC correlation (refined window)")
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.scatter(np.real(cp), np.imag(cp), s=4, alpha=0.35)
        ax2.set_title("Constellation pre-CPE (data bins)")
        ax2.grid(True)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.scatter(np.real(cpost), np.imag(cpost), s=4, alpha=0.35)
        ax3.set_title("Constellation post-CPE (data bins)")
        ax3.grid(True)

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(np.unwrap(np.array(cpe_inst_log)), label="CPE inst (unwrap)", alpha=0.7)
        ax4.plot(np.unwrap(np.array(cpe_smooth_log)), label="CPE smooth (unwrap)", linewidth=2)
        ax4.set_title("Pilot CPE tracking")
        ax4.grid(True)
        ax4.legend()

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(pilot_pow_log)
        ax5.set_title("Pilot power per symbol (should be stable/non-zero)")
        ax5.grid(True)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(dp, bins=60)
        ax6.set_title("DQPSK diff-angle histogram (expect 4 peaks)")
        ax6.grid(True)

        fig.suptitle(
            f"try={t+1:02d} ok={ok}  CFO={cfo_hz:+.1f}Hz  "
            f"crc_calc={crc_calc:08x} crc_rx={crc_rx:08x}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out = os.path.join(args.debug_dir, f"rx_zc_try{t+1:02d}_{'OK' if ok else 'FAIL'}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)

        # Save payload if ok (or always save first N bytes for inspection)
        if ok:
            open(args.outfile, "wb").write(payload)
            print(f"[{t+1:02d}] ✅ CRC OK! wrote {len(payload)} bytes -> {args.outfile}")
            break
        else:
            # also dump a small binary slice to inspect if needed
            dump = os.path.join(args.debug_dir, f"rx_zc_try{t+1:02d}_head.bin")
            open(dump, "wb").write(payload[:256])
            print(f"[{t+1:02d}] CRC fail CFO={cfo_hz:+.1f}Hz calc={crc_calc:08x} rx={crc_rx:08x}  (saved {out})")

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


if __name__ == "__main__":
    main()

# python3 pluto_zc_ofdm32_rx_dbg.py \
#   --uri "usb:1.37.5" \
#   --fc 2.3e9 --fs 1e6 --bw 1.2e6 \
#   --rx_gain 55 --buf 4194304 \
#   --num_syms 250 --zc_reps 40 \
#   --expect_len 2048 \
#   --outfile recovered.jpg \
#   --tries 50 \
#   --debug_dir rx_debug_zc
"""
iio_attr -a -C fw_version
python3 pluto_ofdm_tx.py   --uri "usb:1.7.5"   --fc 2.3e9 --fs 1e6 --bw 1.2e6   --tx_gain -15   --repeat 4 --num_syms 300   --infile stress_2kb.jpg   --cyclic


python3 pluto_zc_ofdm32_rx_dbg.py \
  --uri "usb:1.39.5" \
  --fc 2.3e9 --fs 1e6 --bw 8e5 \
  --rx_gain 55 --buf 262144 \
  --num_syms 250 --zc_reps 40 \
  --expect_len 2048 \
  --outfile recovered.jpg \
  --tries 50 \
  --debug_dir rx_debug_zc

python3 pluto_zc_ofdm32_rx_dbg.py \
  --uri "usb:1.39.5" \
  --fc 2.3e9 --fs 1e6 --bw 8e5 \
  --rx_gain 55 --buf 524288 \
  --num_syms 250 --zc_reps 40 \
  --expect_len 0 \
  --outfile recovered.jpg \
  --tries 10 \
  --debug_dir rx_debug_zc
"""