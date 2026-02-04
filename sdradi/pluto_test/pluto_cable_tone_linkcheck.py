#!/usr/bin/env python3
# pluto_cable_tone_linkcheck.py
import argparse
import time
import numpy as np
import adi

def db(x, eps=1e-12):
    return 20*np.log10(np.maximum(x, eps))

def dbp(x, eps=1e-12):
    return 10*np.log10(np.maximum(x, eps))

def make_tone(fs, n, f0, amp=0.25):
    t = np.arange(n) / fs
    # complex exponential at +f0
    return (amp * np.exp(1j * 2*np.pi * f0 * t)).astype(np.complex64)

def fft_metrics(x, fs, tone_hz, tone_tol_hz=2000.0, guard_bins=5):
    """
    Return:
      peak_f, peak_mag, tone_mag, snr_db, tone_hit(bool)
    """
    n = len(x)
    # window to reduce leakage
    w = np.hanning(n).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * w))
    mag = np.abs(X)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))

    # overall peak
    k_peak = int(np.argmax(mag))
    peak_f = freqs[k_peak]
    peak_mag = mag[k_peak]

    # tone bin neighborhood
    k_tone = int(np.argmin(np.abs(freqs - tone_hz)))
    k0 = max(0, k_tone - guard_bins)
    k1 = min(n, k_tone + guard_bins + 1)
    tone_mag = float(np.max(mag[k0:k1]))

    # estimate noise floor excluding vicinity of DC and tone neighborhood
    exclude = np.zeros(n, dtype=bool)
    # exclude DC neighborhood
    k_dc = int(np.argmin(np.abs(freqs - 0.0)))
    dc0 = max(0, k_dc - 20)
    dc1 = min(n, k_dc + 21)
    exclude[dc0:dc1] = True
    # exclude tone neighborhood
    exclude[k0:k1] = True

    noise = mag[~exclude]
    noise_floor = float(np.median(noise)) if noise.size else 1e-9
    snr_db = 20*np.log10((tone_mag + 1e-12) / (noise_floor + 1e-12))

    tone_hit = (abs(peak_f - tone_hz) <= tone_tol_hz) or (abs(peak_f + tone_hz) <= tone_tol_hz)
    return peak_f, peak_mag, tone_mag, snr_db, tone_hit

def set_channel_list(obj, prop_name, ch):
    # pyadi expects a list of enabled channels (0-based)
    setattr(obj, prop_name, [int(ch)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", type=str, required=True)
    ap.add_argument("--fc", type=float, default=2.405e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--tone_hz", type=float, default=100e3)
    ap.add_argument("--tx_gain", type=float, default=-50.0, help="dB (Pluto attenuation style)")
    ap.add_argument("--rx_gain", type=float, default=0.0, help="dB")
    ap.add_argument("--tx_ch", type=int, default=0, help="0 or 1")
    ap.add_argument("--rx_ch", type=int, default=0, help="0 or 1")
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--tx_buf", type=int, default=262144)
    ap.add_argument("--loops", type=int, default=20)
    ap.add_argument("--tone_tol_hz", type=float, default=2000.0)
    ap.add_argument("--cal_dc", action="store_true", help="remove RX mean (DC)")
    ap.add_argument("--no_tx", action="store_true", help="RX only (no transmit)")
    args = ap.parse_args()

    print("==== Pluto cable tone link check ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tx_ch={args.tx_ch} rx_ch={args.rx_ch} tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain}")
    print(f"[ARGS] rx_buf={args.rx_buf} loops={args.loops} cal_dc={args.cal_dc} no_tx={args.no_tx}")

    sdr = adi.Pluto(args.uri)

    # Basic RF config
    sdr.rx_lo = int(args.fc)
    sdr.tx_lo = int(args.fc)
    sdr.sample_rate = int(args.fs)
    try:
        sdr.rx_rf_bandwidth = int(args.bw)
    except Exception:
        pass
    try:
        sdr.tx_rf_bandwidth = int(args.bw)
    except Exception:
        pass

    # Gains
    # Pluto TX gain is typically "attenuation" in dB (negative values used in some wrappers);
    # We'll set whatever pyadi exposes.
    try:
        sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    except Exception:
        try:
            sdr.tx_hardwaregain = float(args.tx_gain)
        except Exception:
            pass
    try:
        sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    except Exception:
        try:
            sdr.rx_hardwaregain = float(args.rx_gain)
        except Exception:
            pass

    # Buffers
    sdr.rx_buffer_size = int(args.rx_buf)

    # Enable desired channels (0-based!)
    set_channel_list(sdr, "rx_enabled_channels", args.rx_ch)
    set_channel_list(sdr, "tx_enabled_channels", args.tx_ch)

    # Create TX tone
    tx = make_tone(args.fs, args.tx_buf, args.tone_hz, amp=0.20)

    # Warmup read to flush
    for _ in range(3):
        _ = sdr.rx()

    # Start TX
    if not args.no_tx:
        # cyclic=True keeps tone continuously
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx)

    hit = 0
    clipped = 0
    snrs = []
    rms_list = []

    try:
        for i in range(args.loops):
            x = sdr.rx()
            if x is None or len(x) == 0:
                print(f"[{i:03d}] RX empty")
                time.sleep(0.05)
                continue

            x = np.array(x).astype(np.complex64)
            if args.cal_dc:
                x = x - np.mean(x)

            rms = float(np.sqrt(np.mean(np.abs(x)**2)))
            rms_dbfs = db(rms)
            rms_list.append(rms_dbfs)

            # crude clipping check: if too many samples near max
            mx = float(np.max(np.abs(x)))
            if mx > 0.98:
                clipped += 1

            peak_f, peak_mag, tone_mag, snr_db, tone_hit = fft_metrics(
                x, args.fs, args.tone_hz, tone_tol_hz=args.tone_tol_hz
            )
            snrs.append(snr_db)
            if tone_hit:
                hit += 1

            print(f"[{i:03d}] n={len(x)} rms={rms_dbfs:+.2f} dBFS  max|x|={mx:.3f}  "
                  f"peak_f={peak_f:+.1f} Hz  snr~{snr_db:.1f} dB  tone_hit={tone_hit}")

        print("\n==== SUMMARY ====")
        print(f"Tone hit: {hit}/{args.loops}  hit_rate={hit/max(args.loops,1):.2f}")
        if snrs:
            print(f"SNR: median={np.median(snrs):.1f} dB  min={np.min(snrs):.1f} dB  max={np.max(snrs):.1f} dB")
        if rms_list:
            print(f"RMS(dBFS): median={np.median(rms_list):+.2f} dBFS")
        print(f"Clipping blocks (max|x|>0.98): {clipped}/{args.loops}")

        print("\nInterpretation tips:")
        print("- hit_rate 接近 1.0 且 SNR > ~15 dB：这条 cable 链路（TX->RX）基本正常。")
        print("- hit_rate 很低但 rms 很大/peak_f 常在 0Hz：多半是 DC/泄漏/饱和，先把 tx_gain 更小(rx_gain 更小)，加衰减器。")
        print("- rx 一直为空：RX 数据路径没起来（uri/设备/驱动/通道设置/流未启动）。")
    finally:
        # Stop TX
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

if __name__ == "__main__":
    main()