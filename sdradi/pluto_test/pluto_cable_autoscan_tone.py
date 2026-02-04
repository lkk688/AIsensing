#!/usr/bin/env python3
import argparse, time
import numpy as np
import adi

def safe_set(obj, name, value):
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False
    return False

def safe_get(obj, name, default=None):
    return getattr(obj, name, default)

def make_tone(fs, tone_hz, amp, n):
    t = np.arange(n, dtype=np.float64) / fs
    x = amp * np.exp(1j * 2*np.pi*tone_hz*t)
    return x.astype(np.complex64)

def rx_one(sdr, n):
    x = sdr.rx()
    # pyadi: single chan -> np.ndarray (n,)
    # multi chan -> list/tuple of arrays
    if isinstance(x, (list, tuple)):
        # return first enabled channel stream
        x = x[0]
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[:, 0]
    if x.size > n:
        x = x[:n]
    return x

def analyze_fft(x, fs, tone_hz, guard_hz=2e3):
    """
    Return metrics:
      - tone_mag_db, img_mag_db, dc_mag_db, noise_db, tone_snr_db, img_snr_db
      - peak_f (global peak excluding dc-guard)
    """
    x = x.astype(np.complex64, copy=False)

    # remove DC mean (time-domain)
    x0 = x - np.mean(x)

    # FFT
    N = len(x0)
    w = np.hanning(N).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x0 * w))
    mag = np.abs(X) + 1e-12

    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/fs))

    # noise estimate: median outside a small DC guard
    dc_mask = np.abs(freqs) < guard_hz
    noise_mag = np.median(mag[~dc_mask]) if np.any(~dc_mask) else np.median(mag)
    noise_db = 20*np.log10(noise_mag + 1e-12)

    # DC bin (closest to 0 Hz)
    dc_i = int(np.argmin(np.abs(freqs - 0.0)))
    dc_mag = mag[dc_i]
    dc_db = 20*np.log10(dc_mag)

    # tone bin
    ti = int(np.argmin(np.abs(freqs - tone_hz)))
    tone_mag = mag[ti]
    tone_db = 20*np.log10(tone_mag)
    tone_snr = 20*np.log10(tone_mag / (noise_mag + 1e-12))

    # image bin (-tone)
    ii = int(np.argmin(np.abs(freqs + tone_hz)))
    img_mag = mag[ii]
    img_db = 20*np.log10(img_mag)
    img_snr = 20*np.log10(img_mag / (noise_mag + 1e-12))

    # global peak excluding DC guard
    mag2 = mag.copy()
    mag2[dc_mask] = 0.0
    pk = int(np.argmax(mag2))
    peak_f = freqs[pk]
    peak_db = 20*np.log10(mag[pk])

    return {
        "tone_db": tone_db, "tone_snr": tone_snr,
        "img_db": img_db, "img_snr": img_snr,
        "dc_db": dc_db, "noise_db": noise_db,
        "peak_f": float(peak_f), "peak_db": float(peak_db),
    }

def config_common(sdr, fc, fs, bw, rx_buf, rx_gain, tx_gain):
    # Basic RF
    safe_set(sdr, "rx_lo", int(fc))
    safe_set(sdr, "tx_lo", int(fc))
    safe_set(sdr, "sample_rate", int(fs))
    safe_set(sdr, "rx_rf_bandwidth", int(bw))
    safe_set(sdr, "tx_rf_bandwidth", int(bw))
    safe_set(sdr, "rx_buffer_size", int(rx_buf))

    # Gain mode
    # Pluto/AD936x often has gain_control_mode_chan0/1
    safe_set(sdr, "gain_control_mode_chan0", "manual")
    safe_set(sdr, "gain_control_mode_chan1", "manual")
    # Set gains
    # Some firmware uses rx_hardwaregain_chan0/1 and tx_hardwaregain_chan0/1
    safe_set(sdr, "rx_hardwaregain_chan0", float(rx_gain))
    safe_set(sdr, "rx_hardwaregain_chan1", float(rx_gain))
    safe_set(sdr, "tx_hardwaregain_chan0", float(tx_gain))
    safe_set(sdr, "tx_hardwaregain_chan1", float(tx_gain))

def run_case(sdr, tx_ch, rx_ch, fs, tone_hz, tx_amp, n_fft, loops, do_tx):
    sdr.rx_enabled_channels = [int(rx_ch)]
    sdr.tx_enabled_channels = [int(tx_ch)]

    # flush
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    if do_tx:
        # Cyclic TX tone
        tone = make_tone(fs, tone_hz, tx_amp, n_fft)
        safe_set(sdr, "tx_cyclic_buffer", True)
        sdr.tx(tone)
        time.sleep(0.15)

    hits = 0
    best = None

    for k in range(loops):
        x = rx_one(sdr, n_fft)
        met = analyze_fft(x, fs, tone_hz)

        # hit rule: tone SNR > 15dB AND global peak close to tone (or -tone)
        peak_f = met["peak_f"]
        peak_is_tone = (abs(peak_f - tone_hz) < 2e3) or (abs(peak_f + tone_hz) < 2e3)
        hit = (met["tone_snr"] > 15.0 and peak_is_tone)

        hits += int(hit)
        if (best is None) or (met["tone_snr"] > best["tone_snr"]):
            best = met

        print(
            f"  [{k:02d}] peak_f={peak_f/1e3:+7.1f}kHz "
            f"tone_snr={met['tone_snr']:5.1f}dB img_snr={met['img_snr']:5.1f}dB "
            f"dc={met['dc_db']:6.1f}dB noise={met['noise_db']:6.1f}dB "
            f"hit={hit}"
        )

    if do_tx:
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
    ap.add_argument("--tone_hz", type=float, default=100e3)
    ap.add_argument("--tx_gain", type=float, default=-80.0)
    ap.add_argument("--rx_gain", type=float, default=0.0)
    ap.add_argument("--tx_amp", type=float, default=0.005)
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--n_fft", type=int, default=262144)
    ap.add_argument("--loops", type=int, default=6)
    ap.add_argument("--no_tx", action="store_true", help="Only measure RX baseline (no TX)")
    ap.add_argument("--sweep", action="store_true", help="Sweep (tx_ch,rx_ch) in {0,1}x{0,1}")
    args = ap.parse_args()

    print("==== Pluto cable autoscan tone ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain} tx_amp={args.tx_amp}")
    print(f"[ARGS] n_fft={args.n_fft} loops={args.loops} sweep={args.sweep} no_tx={args.no_tx}")

    sdr = adi.Pluto(args.uri)
    config_common(sdr, args.fc, args.fs, args.bw, args.rx_buf, args.rx_gain, args.tx_gain)

    # show channel names
    print("rx_chans:", safe_get(sdr, "_rx_channel_names", None))
    print("tx_chans:", safe_get(sdr, "_tx_channel_names", None))

    cases = []
    if args.sweep:
        for tx in [0, 1]:
            for rx in [0, 1]:
                cases.append((tx, rx))
    else:
        cases.append((0, 0))

    # 0) RX baseline
    if args.no_tx or args.sweep:
        print("\n== RX baseline (no TX) ==")
        for (tx, rx) in cases:
            print(f"\n[CASE] tx_ch={tx} rx_ch={rx} (no TX)")
            hits, best = run_case(
                sdr, tx, rx, args.fs, args.tone_hz, args.tx_amp,
                args.n_fft, max(2, args.loops//2), do_tx=False
            )
            print(f"  -> baseline best tone_snr={best['tone_snr']:.1f}dB peak_f={best['peak_f']/1e3:+.1f}kHz")

    if args.no_tx:
        return

    # 1) TX+RX
    print("\n== TX+RX tone test ==")
    summary = []
    for (tx, rx) in cases:
        print(f"\n[CASE] tx_ch={tx} rx_ch={rx} (TX ON)")
        hits, best = run_case(
            sdr, tx, rx, args.fs, args.tone_hz, args.tx_amp,
            args.n_fft, args.loops, do_tx=True
        )
        summary.append((tx, rx, hits, best["tone_snr"], best["peak_f"]))
        print(f"  -> hits={hits}/{args.loops}, best tone_snr={best['tone_snr']:.1f}dB peak_f={best['peak_f']/1e3:+.1f}kHz")

    print("\n==== SUMMARY ====")
    summary.sort(key=lambda x: x[3], reverse=True)
    for tx, rx, hits, snr, pf in summary:
        print(f"tx={tx} rx={rx}  hit={hits}/{args.loops}  best_tone_snr={snr:5.1f}dB  best_peak_f={pf/1e3:+7.1f}kHz")

if __name__ == "__main__":
    main()