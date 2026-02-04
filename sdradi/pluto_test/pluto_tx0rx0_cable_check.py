#!/usr/bin/env python3
import argparse, time
import numpy as np
import adi

def set_if(sdr, name, val):
    if hasattr(sdr, name):
        try:
            setattr(sdr, name, val)
            return True
        except Exception:
            return False
    return False

def cfg_pluto(sdr, fc, fs, bw, rx_buf, tx_gain, rx_gain):
    set_if(sdr, "rx_lo", int(fc))
    set_if(sdr, "tx_lo", int(fc))
    set_if(sdr, "sample_rate", int(fs))
    set_if(sdr, "rx_rf_bandwidth", int(bw))
    set_if(sdr, "tx_rf_bandwidth", int(bw))
    set_if(sdr, "rx_buffer_size", int(rx_buf))

    # manual gain
    set_if(sdr, "gain_control_mode_chan0", "manual")
    set_if(sdr, "gain_control_mode_chan1", "manual")
    set_if(sdr, "rx_hardwaregain_chan0", float(rx_gain))
    set_if(sdr, "tx_hardwaregain_chan0", float(tx_gain))

def make_tone(fs, tone_hz, amp, n):
    t = np.arange(n, dtype=np.float64) / fs
    x = amp * np.exp(1j * 2*np.pi*tone_hz*t)
    return x.astype(np.complex64)

def rx_one(sdr, n):
    x = sdr.rx()
    if isinstance(x, (list, tuple)):
        x = x[0]
    x = np.asarray(x)
    if x.size > n:
        x = x[:n]
    return x.astype(np.complex64, copy=False)

def fft_metrics(x, fs, tone_hz, dc_guard_hz=5e3, tone_guard_hz=2e3):
    N = len(x)
    # time-domain DC remove
    x = x - np.mean(x)

    w = np.hanning(N).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * w))
    mag = np.abs(X) + 1e-12
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    # noise floor: median outside DC guard and tone bins
    mask = np.ones_like(f, dtype=bool)
    mask &= (np.abs(f) > dc_guard_hz)
    mask &= (np.abs(f - tone_hz) > tone_guard_hz)
    mask &= (np.abs(f + tone_hz) > tone_guard_hz)
    noise = np.median(mag[mask]) if np.any(mask) else np.median(mag)

    # measure bins
    dc_i = int(np.argmin(np.abs(f)))
    t_i  = int(np.argmin(np.abs(f - tone_hz)))
    it_i = int(np.argmin(np.abs(f + tone_hz)))

    dc = mag[dc_i]
    t  = mag[t_i]
    it = mag[it_i]

    # global peak excluding DC guard
    mag2 = mag.copy()
    mag2[np.abs(f) < dc_guard_hz] = 0.0
    pk = int(np.argmax(mag2))

    return {
        "peak_f": float(f[pk]),
        "peak_mag": float(mag[pk]),
        "dc_mag": float(dc),
        "tone_mag": float(t),
        "img_mag": float(it),
        "tone_snr_db": float(20*np.log10(t/noise)),
        "img_snr_db": float(20*np.log10(it/noise)),
        "dc_over_tone_db": float(20*np.log10(dc/t)),
    }

def sat_hint(x):
    a = np.abs(x)
    # 不知道缩放就做相对判据：99.9%分位/中位数过大通常意味着饱和/强尖峰
    p999 = float(np.percentile(a, 99.9))
    med = float(np.median(a) + 1e-12)
    return p999/med

def run(sdr, fs, tone_hz, tx_amp, n, loops, tx_on):
    sdr.rx_enabled_channels = [0]
    sdr.tx_enabled_channels = [0]

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    if tx_on:
        set_if(sdr, "tx_cyclic_buffer", True)
        sdr.tx(make_tone(fs, tone_hz, tx_amp, n))
        time.sleep(0.2)

    hits = 0
    best = None
    for k in range(loops):
        x = rx_one(sdr, n)
        met = fft_metrics(x, fs, tone_hz)
        sat = sat_hint(x)

        # hit: tone SNR>15dB 且 global peak 接近 ±tone
        peak_is_tone = (abs(met["peak_f"] - tone_hz) < 2e3) or (abs(met["peak_f"] + tone_hz) < 2e3)
        hit = (met["tone_snr_db"] > 15.0) and peak_is_tone
        hits += int(hit)

        if best is None or met["tone_snr_db"] > best["tone_snr_db"]:
            best = met

        print(
            f"[{k:02d}] peak_f={met['peak_f']/1e3:+7.1f}kHz  "
            f"tone_snr={met['tone_snr_db']:5.1f}dB img_snr={met['img_snr_db']:5.1f}dB  "
            f"DC/tone={met['dc_over_tone_db']:5.1f}dB  sat_ratio(p99.9/med)={sat:6.1f}  hit={hit}"
        )

    if tx_on:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

    return hits, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2405e6)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--n", type=int, default=262144)
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--tone_hz", type=float, default=600e3)
    ap.add_argument("--tx_gain", type=float, default=-80)
    ap.add_argument("--rx_gain", type=float, default=0)
    ap.add_argument("--tx_amp", type=float, default=0.005)
    args = ap.parse_args()

    print("==== Pluto TX0<->RX0 cable check (1R1T in pyadi sense) ====")
    print(f"[ARGS] uri={args.uri} tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain} tx_amp={args.tx_amp}")

    sdr = adi.Pluto(args.uri)
    print("rx_chans:", getattr(sdr, "_rx_channel_names", None))
    print("tx_chans:", getattr(sdr, "_tx_channel_names", None))

    cfg_pluto(sdr, args.fc, args.fs, args.bw, args.rx_buf, args.tx_gain, args.rx_gain)

    # sanity: only ch0 should work
    try:
        sdr.rx_enabled_channels = [0]
        sdr.tx_enabled_channels = [0]
    except Exception as e:
        print("[FATAL] cannot set complex channel 0:", e)
        return

    print("\n== Baseline (no TX) ==")
    h0, b0 = run(sdr, args.fs, args.tone_hz, args.tx_amp, args.n, max(4, args.loops//2), tx_on=False)
    print(f"baseline best tone_snr={b0['tone_snr_db']:.1f}dB  peak_f={b0['peak_f']/1e3:+.1f}kHz")

    print("\n== TX ON ==")
    h1, b1 = run(sdr, args.fs, args.tone_hz, args.tx_amp, args.n, args.loops, tx_on=True)
    print(f"TXON hits={h1}/{args.loops} best tone_snr={b1['tone_snr_db']:.1f}dB  peak_f={b1['peak_f']/1e3:+.1f}kHz")

    print("\n==== Interpretation ====")
    print("- 如果 TXON 的 best tone_snr 比 baseline 高很多(>10dB)，且 peak_f≈±tone：链路是通的。")
    print("- 如果 DC/tone 很大(>20dB) 且 sat_ratio 很大：还是 DC/泄漏/饱和主导，继续减 tx_amp 或加衰减。")
    print("- 建议把 tone_hz 设到 600k~1.0M，远离 DC 再做 OFDM/同步。")

if __name__ == "__main__":
    main()