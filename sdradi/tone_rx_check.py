#!/usr/bin/env python3
import numpy as np, time, adi, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--rx_gain", type=float, default=10)
    ap.add_argument("--tone", type=float, default=100e3)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--loops", type=int, default=10)
    args = ap.parse_args()

    s = adi.Pluto(args.uri)
    s.sample_rate = int(args.fs)
    s.rx_lo = int(args.fc)
    s.rx_rf_bandwidth = int(args.fs)
    s.gain_control_mode_chan0 = "manual"
    s.rx_hardwaregain_chan0 = float(args.rx_gain)
    s.rx_enabled_channels = [0]
    s.rx_buffer_size = args.n

    for _ in range(3): s.rx()

    k = int(round(args.tone * args.n / args.fs))
    print(f"uri={args.uri} fc={args.fc/1e6:.1f}MHz fs={args.fs/1e6:.1f}Msps rx_gain={args.rx_gain} tone={args.tone/1e3:.1f}kHz kbin={k}")

    for i in range(args.loops):
        x = s.rx()
        X = np.fft.fft(x)
        tone_mag = np.abs(X[k])
        noise = np.median(np.abs(X))
        snr = 20*np.log10((tone_mag+1e-12)/(noise+1e-12))
        peak = np.max(np.abs(x))
        print(f"[{i:02d}] peak={peak:8.1f}  tone_mag={tone_mag:10.1f}  snr={snr:6.1f} dB")
        time.sleep(0.2)

if __name__ == "__main__":
    main()