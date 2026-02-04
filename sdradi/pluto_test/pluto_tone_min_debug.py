#!/usr/bin/env python3
import argparse, time
import numpy as np

def mk_tone(fs, f0, n, amp=0.2):
    t = np.arange(n, dtype=np.float64) / float(fs)
    x = amp * np.exp(1j * 2*np.pi * float(f0) * t)
    return x.astype(np.complex64)

def fft_peak_hz(x, fs):
    x = np.asarray(x)
    # 去 DC（否则 DC 泄漏很容易被选成 peak）
    x = x - np.mean(x)
    w = np.hanning(len(x)).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * w))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0/fs))
    mag = np.abs(X)
    k = int(np.argmax(mag))
    return float(f[k]), float(mag[k]), float(np.median(mag))

def config_pluto(uri, fc, fs, bw, tx_gain, rx_gain, rx_buf):
    import adi
    sdr = adi.Pluto(uri)

    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    sdr.sample_rate = int(fs)
    sdr.rx_rf_bandwidth = int(bw)
    sdr.tx_rf_bandwidth = int(bw)

    # Pluto 1R1T: 通道固定为 [0]
    sdr.rx_enabled_channels = [0]
    sdr.tx_enabled_channels = [0]

    # gain
    sdr.tx_hardwaregain_chan0 = float(tx_gain)   # dB (Pluto: -89.75 ~ 0)
    sdr.rx_hardwaregain_chan0 = float(rx_gain)   # dB

    sdr.rx_buffer_size = int(rx_buf)

    # 关键：循环发射 tone
    sdr.tx_cyclic_buffer = True

    return sdr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2405e6)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--tone_hz", type=float, default=200e3)
    ap.add_argument("--tx_gain", type=float, default=-40)
    ap.add_argument("--rx_gain", type=float, default=20)
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--tx_len", type=int, default=65536, help="TX cyclic buffer length")
    ap.add_argument("--loops", type=int, default=10)
    args = ap.parse_args()

    print("==== Pluto tone MIN debug ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain} rx_buf={args.rx_buf}")

    sdr = config_pluto(args.uri, args.fc, args.fs, args.bw, args.tx_gain, args.rx_gain, args.rx_buf)

    # 先清 RX（避免读到旧缓冲）
    for _ in range(3):
        _ = sdr.rx()

    # 发 cyclic tone
    tx = mk_tone(args.fs, args.tone_hz, args.tx_len, amp=0.2)
    sdr.tx(tx)
    time.sleep(0.2)  # 给硬件稳定时间

    ok = 0
    for i in range(args.loops):
        rx = sdr.rx()
        rx = np.asarray(rx).astype(np.complex64)
        peak_f, peak_mag, med = fft_peak_hz(rx, args.fs)

        # tone bin 简单匹配（允许 2kHz 偏差）
        if abs(abs(peak_f) - args.tone_hz) < 2000:
            ok += 1

        snr_like = 20*np.log10((peak_mag + 1e-12) / (med + 1e-12))
        rms = np.sqrt(np.mean(np.abs(rx)**2))
        print(f"[{i:03d}] rx_n={len(rx)} rms={rms:.4e}  peak_f={peak_f:+.1f} Hz  snr~{snr_like:.1f} dB")

    print(f"==== SUMMARY ====\nPeak near tone: {ok}/{args.loops}")

    # 停 TX
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

if __name__ == "__main__":
    main()