#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import adi

N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)], dtype=int)

def create_stf_ref():
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        X[(k + N_FFT) % N_FFT] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x, 2).astype(np.complex64)          # 2N
    stf_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)  # CP + 2N
    return stf_cp

def create_ltf_ref_freq_shift():
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N_FFT) % N_FFT] = 1.0 if i % 2 == 0 else -1.0
    return np.fft.fftshift(X)

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
    db = 20*np.log10(mag/np.max(mag))
    return float(cfo), (freqs, db, detected)

def ncc_corr(rx, ref, search_len, step=1):
    L = len(ref)
    search_len = min(search_len, len(rx)-L)
    idxs = np.arange(0, search_len, step, dtype=int)
    refE = np.sqrt(np.sum(np.abs(ref)**2)) + 1e-12
    corr = np.zeros(len(idxs), dtype=np.float32)

    # NCC: |<r,s>| / (||r||*||s||)
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
        for ss,_ in chosen:
            if abs(s-ss) < guard:
                ok = False
                break
        if ok:
            chosen.append((s, float(corr[oi])))
            if len(chosen) >= k:
                break
    chosen.sort(key=lambda x: x[0])
    return chosen

def extract_sym_fftshift(rx, start):
    if start + SYMBOL_LEN > len(rx):
        return None
    sym = rx[start+N_CP : start+SYMBOL_LEN]
    Y = np.fft.fftshift(np.fft.fft(sym) / np.sqrt(N_FFT))
    return Y.astype(np.complex64)

def cfo_from_stf(rx, fs, stf_idx):
    # CFO from STF two halves (after CP), using N_FFT spacing
    s = stf_idx + N_CP
    if s + 2*N_FFT > len(rx):
        return None
    x1 = rx[s:s+N_FFT]
    x2 = rx[s+N_FFT:s+2*N_FFT]
    v = np.vdot(x1, x2)
    ang = np.angle(v)
    return float(ang / (2*np.pi*(N_FFT/fs)))

def ltf_score(Y, Xref):
    # compare only used bins (where Xref != 0)
    mask = (np.abs(Xref) > 0.1)
    if np.sum(mask) < 10:
        return 0.0
    # normalized matched score
    num = np.abs(np.sum(Y[mask] * np.conj(Xref[mask])))
    den = np.sum(np.abs(Y[mask])) + 1e-12
    return float(num / den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=915e6)
    ap.add_argument("--fs", type=float, default=2e6)
    ap.add_argument("--bw", type=float, default=2.4e6)
    ap.add_argument("--rx_gain", type=float, default=40.0)
    ap.add_argument("--buf_size", type=int, default=131072)

    ap.add_argument("--tone_freq", type=float, default=100e3)
    ap.add_argument("--search_len", type=int, default=120000)
    ap.add_argument("--scan_step", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--guard", type=int, default=1200)

    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--outdir", default="preamble_sanity_dbg")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    stf_ref = create_stf_ref()
    ltf_ref = create_ltf_ref_freq_shift()

    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf_size)

    # flush
    for _ in range(3):
        _ = sdr.rx()

    for t in range(1, args.tries+1):
        rx_raw = sdr.rx()
        rx = (rx_raw.astype(np.complex64) / (2**14)).astype(np.complex64)
        rx = rx - np.mean(rx)

        cfo_tone, tone_info = estimate_cfo_from_tone(rx, args.fs, expected=args.tone_freq)
        rx1 = apply_cfo(rx, cfo_tone, args.fs)

        corr, idxs, best_idx, best_val = ncc_corr(rx1, stf_ref, search_len=args.search_len, step=max(1,args.scan_step))
        peaks = topk_peaks(corr, idxs, k=args.topk, guard=args.guard)

        # overview fig
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1,3,1)
        ax1.plot(np.abs(rx[:min(80000,len(rx))]))
        ax1.set_title(f"Time |rx| (first 80k)\ntry={t}")
        ax1.grid(True)

        ax2 = fig.add_subplot(1,3,2)
        if tone_info is not None:
            freqs, db, detected = tone_info
            ax2.plot(freqs/1e3, db)
            ax2.set_title(f"Tone FFT(dB)\ndetected={detected/1e3:.1f}kHz CFO={cfo_tone:+.1f}Hz")
            ax2.grid(True)
        else:
            ax2.axis("off")

        ax3 = fig.add_subplot(1,3,3)
        ax3.plot(idxs, corr)
        ax3.axvline(best_idx, color="r", linestyle="--", label=f"best {best_val:.3f}")
        for s,v in peaks:
            ax3.axvline(s, color="k", alpha=0.15)
        ax3.set_title("STF NCC corr (higher is better)")
        ax3.grid(True)
        ax3.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"try{t:02d}_overview.png"), dpi=140)
        plt.close(fig)

        # per-peak LTF verification
        for pi,(stf_idx, stf_v) in enumerate(peaks):
            cfo_stf = cfo_from_stf(rx, args.fs, stf_idx)
            cfo_use = cfo_tone if cfo_stf is None else 0.5*cfo_tone + 0.5*cfo_stf
            rxc = apply_cfo(rx, cfo_use, args.fs)

            ltf_start = stf_idx + len(stf_ref)
            Y0 = extract_sym_fftshift(rxc, ltf_start)
            Y1 = extract_sym_fftshift(rxc, ltf_start + SYMBOL_LEN)
            if Y0 is None or Y1 is None:
                continue
            Y = (Y0+Y1)/2.0
            score = ltf_score(Y, ltf_ref)

            # plots
            used_mask = (np.abs(ltf_ref) > 0.1)

            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(2,3,1)
            ax.plot(np.abs(rxc[stf_idx:stf_idx+len(stf_ref)]))
            ax.set_title(f"STF window |rx|\nidx={stf_idx} ncc={stf_v:.3f}")
            ax.grid(True)

            ax = fig.add_subplot(2,3,2)
            ax.plot(np.abs(Y0), label="|Y0|")
            ax.plot(np.abs(Y1), label="|Y1|", alpha=0.8)
            ax.set_title("LTF FFT mags (two repeats)\nshould be similar")
            ax.grid(True); ax.legend()

            ax = fig.add_subplot(2,3,3)
            ax.plot(np.abs(ltf_ref), label="|Xref|")
            ax.plot(np.abs(Y), label="|Yavg|", alpha=0.8)
            ax.set_title("LTF ref vs received (fftshift)")
            ax.grid(True); ax.legend()

            H = np.ones(N_FFT, dtype=np.complex64)
            H[used_mask] = Y[used_mask] / (ltf_ref[used_mask] + 1e-12)

            ax = fig.add_subplot(2,3,4)
            ax.plot(np.abs(H))
            ax.set_title("|H| on used bins (should be smooth-ish)")
            ax.grid(True)

            ax = fig.add_subplot(2,3,5)
            ax.plot(np.unwrap(np.angle(H)))
            ax.set_title("angle(H) unwrap (used bins)")
            ax.grid(True)

            ax = fig.add_subplot(2,3,6)
            ax.axis("off")
            ax.text(0.02, 0.98,
                    f"try={t:02d} peak={pi}\n"
                    f"STF_ncc={stf_v:.3f}\n"
                    f"tone_CFO={cfo_tone:+.1f}Hz\n"
                    f"stf_CFO={(cfo_stf if cfo_stf is not None else float('nan')):+.1f}Hz\n"
                    f"cfo_use={cfo_use:+.1f}Hz\n"
                    f"LTF_score={score:.3f}\n",
                    va="top", family="monospace")

            fig.suptitle("Preamble Sanity Check (STF/LTF)", fontsize=14)
            fig.tight_layout(rect=[0,0,1,0.95])
            fig.savefig(os.path.join(args.outdir, f"try{t:02d}_peak{pi:02d}_stf{stf_v:.3f}_score{score:.3f}.png"),
                        dpi=140)
            plt.close(fig)

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
python3 rf_step0_preamble_sanity_rx.py \
  --uri "ip:192.168.2.2" \
  --fc 915e6 --fs 2e6 --bw 2.4e6 \
  --rx_gain 40 --buf_size 131072 \
  --tone_freq 100e3 \
  --search_len 120000 \
  --scan_step 1 \
  --topk 8 --guard 1200 \
  --tries 5 \
  --outdir preamble_sanity_dbg
"""