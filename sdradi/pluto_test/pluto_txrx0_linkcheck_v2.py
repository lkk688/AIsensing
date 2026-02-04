#!/usr/bin/env python3
import argparse, time
import numpy as np
import adi

def try_set(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False

def try_get(obj, name):
    try:
        return getattr(obj, name)
    except Exception:
        return None

def make_tone(fs, n, f0, amp):
    t = np.arange(n) / fs
    return (amp * np.exp(1j * 2*np.pi * f0 * t)).astype(np.complex64)

def fft_report(x, fs, tone_hz, topk=5):
    n = len(x)
    w = np.hanning(n).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * w))
    mag = np.abs(X)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))

    k_dc = int(np.argmin(np.abs(freqs - 0.0)))
    k_tone = int(np.argmin(np.abs(freqs - tone_hz)))
    k_itone = int(np.argmin(np.abs(freqs + tone_hz)))

    # around-bin max to handle leakage
    def around(k, r=6):
        a = max(0, k-r); b = min(n, k+r+1)
        kk = a + int(np.argmax(mag[a:b]))
        return freqs[kk], float(mag[kk])

    fdc, mdc = around(k_dc, 20)
    ft, mt = around(k_tone, 8)
    fit, mit = around(k_itone, 8)

    # noise floor (median), excluding DC/tone neighborhoods
    exclude = np.zeros(n, dtype=bool)
    def excl(k, r):
        a = max(0, k-r); b = min(n, k+r+1)
        exclude[a:b] = True
    excl(k_dc, 50)
    excl(k_tone, 20)
    excl(k_itone, 20)
    noise = mag[~exclude]
    nf = float(np.median(noise)) if noise.size else 1e-12
    snr_t = 20*np.log10((mt+1e-12)/(nf+1e-12))
    snr_it = 20*np.log10((mit+1e-12)/(nf+1e-12))

    # top peaks
    idx = np.argpartition(mag, -topk)[-topk:]
    idx = idx[np.argsort(mag[idx])[::-1]]
    peaks = [(float(freqs[i]), float(mag[i])) for i in idx]

    return {
        "dc": (fdc, mdc),
        "tone": (ft, mt, snr_t),
        "itone": (fit, mit, snr_it),
        "nf": nf,
        "peaks": peaks
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2.405e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--tone_hz", type=float, default=100e3)
    ap.add_argument("--tx_ch", type=int, default=0)
    ap.add_argument("--rx_ch", type=int, default=0)
    ap.add_argument("--tx_gain", type=float, default=-80.0)
    ap.add_argument("--rx_gain", type=float, default=0.0)
    ap.add_argument("--tx_amp", type=float, default=0.02)  # 关键：数字幅度
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--tx_buf", type=int, default=262144)
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--no_tx", action="store_true")
    ap.add_argument("--rm_dc", action="store_true", help="remove mean")
    args = ap.parse_args()

    print("==== Pluto TX/RX linkcheck v2 ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tx_ch={args.tx_ch} rx_ch={args.rx_ch} tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain} tx_amp={args.tx_amp}")

    sdr = adi.Pluto(args.uri)

    # RF
    sdr.rx_lo = int(args.fc)
    sdr.tx_lo = int(args.fc)
    sdr.sample_rate = int(args.fs)
    try_set(sdr, "rx_rf_bandwidth", int(args.bw))
    try_set(sdr, "tx_rf_bandwidth", int(args.bw))
    sdr.rx_buffer_size = int(args.rx_buf)

    # Enable channels (0-based)
    sdr.rx_enabled_channels = [int(args.rx_ch)]
    sdr.tx_enabled_channels = [int(args.tx_ch)]

    # Force manual gain control if possible
    # (different pyadi versions use different names)
    manual_set = False
    for name in [
        f"gain_control_mode_chan{args.rx_ch}",
        "gain_control_mode_chan0",
        "gain_control_mode"
    ]:
        if try_set(sdr, name, "manual"):
            manual_set = True
            break

    # Set gains (try chan-specific first)
    tx_set = False
    rx_set = False
    for name in [f"tx_hardwaregain_chan{args.tx_ch}", "tx_hardwaregain_chan0", "tx_hardwaregain"]:
        if try_set(sdr, name, float(args.tx_gain)):
            tx_set = True
            tx_name = name
            break
    for name in [f"rx_hardwaregain_chan{args.rx_ch}", "rx_hardwaregain_chan0", "rx_hardwaregain"]:
        if try_set(sdr, name, float(args.rx_gain)):
            rx_set = True
            rx_name = name
            break

    # Read back
    print(f"[CFG] manual_gain={manual_set}")
    print(f"[CFG] tx_gain_set={tx_set} via {locals().get('tx_name', None)}  readback={try_get(sdr, locals().get('tx_name',''))}")
    print(f"[CFG] rx_gain_set={rx_set} via {locals().get('rx_name', None)}  readback={try_get(sdr, locals().get('rx_name',''))}")

    # Prepare TX
    tx = make_tone(args.fs, args.tx_buf, args.tone_hz, args.tx_amp)

    # Warmup RX
    for _ in range(3):
        _ = sdr.rx()

    if not args.no_tx:
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx)

    try:
        hit = 0
        for i in range(args.loops):
            x = np.array(sdr.rx()).astype(np.complex64)
            if args.rm_dc:
                x = x - np.mean(x)

            rms = float(np.sqrt(np.mean(np.abs(x)**2)))
            mx = float(np.max(np.abs(x)))

            rep = fft_report(x, args.fs, args.tone_hz, topk=5)
            ft, mt, snr_t = rep["tone"]
            fdc, mdc = rep["dc"]
            fit, mit, snr_it = rep["itone"]

            # tone hit if tone bin clearly above noise
            tone_hit = (snr_t > 12.0) or (snr_it > 12.0)
            hit += int(tone_hit)

            peaks_str = ", ".join([f"{pf/1e3:+.1f}kHz:{pm:.1f}" for pf,pm in rep["peaks"]])

            print(f"[{i:03d}] rms={20*np.log10(max(rms,1e-12)):+.2f}dB  max|x|={mx:.3f}  "
                  f"DC@{fdc:+.1f}Hz:{mdc:.1f}  "
                  f"T@{ft/1e3:+.1f}kHz:{mt:.1f} snr~{snr_t:.1f}dB  "
                  f"IT@{fit/1e3:+.1f}kHz:{mit:.1f} snr~{snr_it:.1f}dB  hit={tone_hit}")
            print(f"      top5: {peaks_str}")
            time.sleep(0.05)

        print("\n==== SUMMARY ====")
        print(f"tone_hit: {hit}/{args.loops}  rate={hit/max(args.loops,1):.2f}")
        print("If max|x| stays huge / DC dominates -> reduce tx_amp, reduce tx_gain, add attenuator, ensure gains read back correctly.")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

if __name__ == "__main__":
    main()