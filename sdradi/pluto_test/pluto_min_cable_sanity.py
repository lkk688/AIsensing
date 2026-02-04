#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np

try:
    import adi
except ImportError:
    raise SystemExit("ERROR: need pyadi-iio. Try: pip install pyadi-iio")


def dbfs_rms(x: np.ndarray) -> float:
    """粗略 dBFS（以 1.0 满幅为 0dBFS）"""
    if x is None or len(x) == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms <= 1e-12:
        return float("-inf")
    return 20 * np.log10(rms)


def iq_stats(x: np.ndarray):
    if x is None:
        return {"n": 0, "rms": 0.0, "peak": 0.0, "rms_dbfs": float("-inf")}
    x = np.asarray(x)
    if x.size == 0:
        return {"n": 0, "rms": 0.0, "peak": 0.0, "rms_dbfs": float("-inf")}
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    peak = float(np.max(np.abs(x)))
    return {"n": int(x.size), "rms": rms, "peak": peak, "rms_dbfs": float(dbfs_rms(x))}


def peak_freq_hz(x: np.ndarray, fs: float):
    """返回 FFT 主峰频率（Hz，带符号）和峰值幅度（线性）"""
    if x is None or len(x) == 0:
        return 0.0, 0.0
    x = np.asarray(x)
    N = len(x)
    # 去直流、加窗，减少泄漏
    xw = (x - np.mean(x)) * np.hanning(N)
    X = np.fft.fftshift(np.fft.fft(xw))
    mag = np.abs(X)
    k = int(np.argmax(mag))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/fs))
    return float(freqs[k]), float(mag[k])


def config_pluto(
    uri: str,
    fc: float,
    fs: float,
    bw: float,
    tx_gain: float,
    rx_gain: float,
    tx_ch: int,
    rx_ch: int,
    tx_port: int,
    rx_port: int,
    rx_buf: int,
):
    """
    只配置到 “能收能发” 的最小集合。
    tx_ch/rx_ch: 0->通道1, 1->通道2
    tx_port/rx_port: 0->A, 1->B (对应 Pluto 的端口映射；不同固件可能名字不同)
    """
    sdr = adi.Pluto(uri)

    # 基本射频参数
    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    sdr.sample_rate = int(fs)

    # 一般 Pluto 用 rx_rf_bandwidth/tx_rf_bandwidth
    if hasattr(sdr, "rx_rf_bandwidth"):
        sdr.rx_rf_bandwidth = int(bw)
    if hasattr(sdr, "tx_rf_bandwidth"):
        sdr.tx_rf_bandwidth = int(bw)

    # 增益
    # RX：建议先手动固定，不用 AGC
    if hasattr(sdr, "gain_control_mode_chan0"):
        sdr.gain_control_mode_chan0 = "manual"
    if hasattr(sdr, "gain_control_mode_chan1"):
        sdr.gain_control_mode_chan1 = "manual"

    if rx_ch == 0 and hasattr(sdr, "rx_hardwaregain_chan0"):
        sdr.rx_hardwaregain_chan0 = float(rx_gain)
    if rx_ch == 1 and hasattr(sdr, "rx_hardwaregain_chan1"):
        sdr.rx_hardwaregain_chan1 = float(rx_gain)

    # TX：Pluto 通常是 tx_hardwaregain_chan0/1 (单位 dB，负数表示衰减)
    if tx_ch == 0 and hasattr(sdr, "tx_hardwaregain_chan0"):
        sdr.tx_hardwaregain_chan0 = float(tx_gain)
    if tx_ch == 1 and hasattr(sdr, "tx_hardwaregain_chan1"):
        sdr.tx_hardwaregain_chan1 = float(tx_gain)

    # buffer
    sdr.rx_buffer_size = int(rx_buf)

    # 通道使能（关键！）
    # pyadi-iio: sdr.rx_enabled_channels / tx_enabled_channels
    # 这里用索引
    sdr.rx_enabled_channels = [int(rx_ch)]
    sdr.tx_enabled_channels = [int(tx_ch)]

    # 端口选择（不同固件属性名可能不同，所以多尝试）
    # 常见：rx_rf_port_select / tx_rf_port_select
    # 值通常是 "A_BALANCED"/"B_BALANCED" 或 "A"/"B"
    port_map = {
        0: ["A", "A_BALANCED", "a", "a_balanced"],
        1: ["B", "B_BALANCED", "b", "b_balanced"],
    }

    rx_ok = tx_ok = False
    for attr in ["rx_rf_port_select", "rx_rf_port_select_chan0", "rx_rf_port_select_chan1"]:
        if hasattr(sdr, attr):
            for v in port_map[int(rx_port)]:
                try:
                    setattr(sdr, attr, v)
                    rx_ok = True
                    break
                except Exception:
                    pass
    for attr in ["tx_rf_port_select", "tx_rf_port_select_chan0", "tx_rf_port_select_chan1"]:
        if hasattr(sdr, attr):
            for v in port_map[int(tx_port)]:
                try:
                    setattr(sdr, attr, v)
                    tx_ok = True
                    break
                except Exception:
                    pass

    return sdr, rx_ok, tx_ok


def main():
    ap = argparse.ArgumentParser(description="Pluto TX2<->RX2 cable minimal sanity test")
    ap.add_argument("--uri", required=True, help="e.g. ip:192.168.3.2 or usb:1.2.3")
    ap.add_argument("--fc", type=float, default=2405e6)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--tx_gain", type=float, default=-40.0, help="TX hardwaregain (dB), usually <= 0")
    ap.add_argument("--rx_gain", type=float, default=20.0, help="RX hardwaregain (dB)")
    ap.add_argument("--tx_ch", type=int, default=1, choices=[0,1], help="0=TX1, 1=TX2")
    ap.add_argument("--rx_ch", type=int, default=1, choices=[0,1], help="0=RX1, 1=RX2")
    ap.add_argument("--tx_port", type=int, default=1, choices=[0,1], help="0=A, 1=B")
    ap.add_argument("--rx_port", type=int, default=1, choices=[0,1], help="0=A, 1=B")
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--tone_hz", type=float, default=200e3)
    ap.add_argument("--amp", type=float, default=0.25, help="TX tone amplitude (0~1)")
    ap.add_argument("--nfft", type=int, default=16384)
    ap.add_argument("--loops", type=int, default=50)
    ap.add_argument("--stage", type=str, default="all", choices=["rx_only","tx_only","txrx","all"])
    args = ap.parse_args()

    print("==== Pluto MIN sanity (direct pyadi-iio) ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tx_ch={args.tx_ch} rx_ch={args.rx_ch} tx_port={args.tx_port} rx_port={args.rx_port}")
    print(f"[ARGS] tx_gain={args.tx_gain} rx_gain={args.rx_gain} rx_buf={args.rx_buf}")

    sdr, rx_port_ok, tx_port_ok = config_pluto(
        uri=args.uri, fc=args.fc, fs=args.fs, bw=args.bw,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain,
        tx_ch=args.tx_ch, rx_ch=args.rx_ch,
        tx_port=args.tx_port, rx_port=args.rx_port,
        rx_buf=args.rx_buf
    )

    print(f"[CFG] rx_enabled_channels={getattr(sdr,'rx_enabled_channels',None)}  tx_enabled_channels={getattr(sdr,'tx_enabled_channels',None)}")
    print(f"[CFG] rx_port_set={'YES' if rx_port_ok else 'NO'}  tx_port_set={'YES' if tx_port_ok else 'NO'}")
    print("[NOTE] 如果 rx_port_set/tx_port_set=NO，说明你固件/pyadi 的端口属性名不同，但不影响先验证“是否能收/能发”。")

    # 先生成 tone（循环发送）
    Ntx = max(args.nfft, 8192)
    t = np.arange(Ntx) / args.fs
    tone = args.amp * np.exp(1j * 2*np.pi*args.tone_hz * t).astype(np.complex64)

    try:
        if args.stage in ("rx_only", "all"):
            print("\n== Stage 1: RX only (no TX) ==")
            for k in range(10):
                rx = sdr.rx()
                st = iq_stats(rx)
                pf, pm = peak_freq_hz(rx[:args.nfft] if len(rx) >= args.nfft else rx, args.fs)
                print(f"[RX {k:02d}] {st}  peak_f={pf/1e3:+.1f}kHz mag={pm:.1f}")
                time.sleep(0.05)

        if args.stage in ("tx_only", "all"):
            print("\n== Stage 2: TX only (cyclic tone) ==")
            # cyclic=True 让 Pluto DAC 循环播放
            sdr.tx_cyclic_buffer = True
            sdr.tx(tone)
            print(f"[TX] cyclic tone on: f={args.tone_hz/1e3:.1f}kHz amp={args.amp} (Now wait 1s)")
            time.sleep(1.0)
            # 不立刻关，给下一阶段用
            if args.stage == "tx_only":
                print("[TX] Done. (Ctrl+C to stop, or it will stop when script exits)")
                return

        if args.stage in ("txrx", "all"):
            print("\n== Stage 3: TX+RX (expect tone peak near tone_hz) ==")
            # 若上一步没开 TX，这里也开
            if not getattr(sdr, "tx_cyclic_buffer", False):
                sdr.tx_cyclic_buffer = True
                sdr.tx(tone)

            ok = 0
            for k in range(args.loops):
                rx = sdr.rx()
                st = iq_stats(rx)
                # 用前 nfft 做频谱峰
                seg = rx[:args.nfft] if len(rx) >= args.nfft else rx
                pf, pm = peak_freq_hz(seg, args.fs)
                # 简单判定：峰值频率接近 tone_hz（允许几 kHz 误差）
                if abs(pf - args.tone_hz) < 5e3 and st["n"] > 0:
                    ok += 1
                print(f"[TXRX {k:03d}] n={st['n']} rms_dbfs={st['rms_dbfs']:.2f} peak={st['peak']:.3f}  peak_f={pf/1e3:+.1f}kHz mag={pm:.1f}")
                time.sleep(0.02)

            print("\n==== SUMMARY ====")
            print(f"Peak-detect OK count: {ok}/{args.loops}")
            if ok == 0:
                print("结论：RX 能否读到样本/是否看到 tone 峰？若一直 n=0 -> RX 读通道没工作；若 n>0 但没有 tone 峰 -> 端口/线缆/通道映射/tx未生效。")

    finally:
        # 关 TX，避免 Pluto 一直发
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass


if __name__ == "__main__":
    main()