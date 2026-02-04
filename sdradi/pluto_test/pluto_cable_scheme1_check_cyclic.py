#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np

def db(x, floor=1e-20):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))

def rms_db(x, floor=1e-20):
    r = np.sqrt(np.mean(np.abs(x) ** 2))
    return db(r, floor=floor)

def remove_dc(x):
    return x - np.mean(x)

def build_tone(fs, tone_hz, n, amp):
    t = np.arange(n) / float(fs)
    x = amp * np.exp(1j * 2*np.pi * float(tone_hz) * t)
    return x.astype(np.complex64)

def fft_tone_snr(x, fs, tone_hz, guard_bins=5):
    n = len(x)
    w = np.hanning(n).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * w))
    mag = np.abs(X)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0/fs))

    tone = float(tone_hz)
    tone_bin = int(np.argmin(np.abs(freqs - tone)))
    img_bin  = int(np.argmin(np.abs(freqs + tone)))
    dc_bin   = int(np.argmin(np.abs(freqs - 0.0)))

    tone_mag = mag[tone_bin]
    img_mag  = mag[img_bin]
    dc_mag   = mag[dc_bin]

    mask = np.ones(n, dtype=bool)
    # exclude tone/img/DC neighborhoods from noise estimate
    for b in range(-guard_bins, guard_bins+1):
        for center in (tone_bin, img_bin, dc_bin):
            k = center + b
            if 0 <= k < n:
                mask[k] = False

    noise = np.median(mag[mask]) + 1e-12
    tone_snr = 20*np.log10((tone_mag + 1e-12) / noise)

    # global peak (debug)
    pk = int(np.argmax(mag))
    peak_f = freqs[pk]
    peak_mag = mag[pk]

    return {
        "tone_mag": float(tone_mag),
        "img_mag": float(img_mag),
        "dc_mag": float(dc_mag),
        "noise": float(noise),
        "tone_snr_db": float(tone_snr),
        "peak_f": float(peak_f),
        "peak_mag": float(peak_mag),
    }

def lockin_amp(x, fs, tone_hz):
    n = len(x)
    t = np.arange(n) / float(fs)
    lo = np.exp(-1j * 2*np.pi * float(tone_hz) * t)
    y = x * lo
    return float(np.abs(np.mean(y)))

def safe_set_attr(obj, candidates, value):
    for name in candidates:
        if hasattr(obj, name):
            try:
                setattr(obj, name, value)
                rb = getattr(obj, name)
                return True, name, rb
            except Exception:
                pass
    return False, None, None

def configure_pluto(uri, fc, fs, bw, tx_gain, rx_gain, rx_buf, tx_ch, rx_ch):
    import adi
    sdr = adi.Pluto(uri)

    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    try: sdr.rx_rf_bandwidth = int(bw)
    except: pass
    try: sdr.tx_rf_bandwidth = int(bw)
    except: pass

    sdr.rx_buffer_size = int(rx_buf)
    sdr.rx_enabled_channels = [int(rx_ch)]
    sdr.tx_enabled_channels = [int(tx_ch)]

    # manual gain
    mode_attr = f"gain_control_mode_chan{int(rx_ch)}"
    if hasattr(sdr, mode_attr):
        try: setattr(sdr, mode_attr, "manual")
        except: pass

    rx_ok, rx_attr, rx_rb = safe_set_attr(
        sdr, [f"rx_hardwaregain_chan{int(rx_ch)}", "rx_hardwaregain"], float(rx_gain)
    )
    tx_ok, tx_attr, tx_rb = safe_set_attr(
        sdr, [f"tx_hardwaregain_chan{int(tx_ch)}", "tx_hardwaregain"], float(tx_gain)
    )

    return sdr, {
        "rx_ok": rx_ok, "rx_attr": rx_attr, "rx_rb": rx_rb,
        "tx_ok": tx_ok, "tx_attr": tx_attr, "tx_rb": tx_rb
    }

def rx_grab(sdr, rm_dc, fs, tone_hz):
    rx = np.asarray(sdr.rx()).astype(np.complex64)
    rx2 = remove_dc(rx) if rm_dc else rx
    met = fft_tone_snr(rx2, fs, tone_hz)
    lin = lockin_amp(rx2, fs, tone_hz)
    return rx, rx2, met, lin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", type=str, required=True)
    ap.add_argument("--fc", type=float, default=2405e6)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--tx_ch", type=int, default=0)
    ap.add_argument("--rx_ch", type=int, default=0)
    ap.add_argument("--tone_hz", type=float, default=900e3)
    ap.add_argument("--tx_gain", type=float, default=-80.0)
    ap.add_argument("--rx_gain", type=float, default=0.0)
    ap.add_argument("--tx_amp", type=float, default=2e-4)
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--tx_len", type=int, default=262144)  # 让 TX 波形长度≈RX buffer，方便对齐
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--rm_dc", action="store_true")
    ap.add_argument("--warmup_rx", type=int, default=3, help="TX切换后丢弃的RX包数")
    args = ap.parse_args()

    print("==== Pluto cable Scheme-1 check (CYCLIC TX) ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tx_ch={args.tx_ch} rx_ch={args.rx_ch} tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain} tx_amp={args.tx_amp}")
    print(f"[ARGS] rx_buf={args.rx_buf} tx_len={args.tx_len} loops={args.loops} rm_dc={args.rm_dc}")

    sdr, info = configure_pluto(args.uri, args.fc, args.fs, args.bw, args.tx_gain, args.rx_gain,
                               args.rx_buf, args.tx_ch, args.rx_ch)
    print(f"[CFG] tx_gain_set={info['tx_ok']} via {info['tx_attr']} readback={info['tx_rb']}")
    print(f"[CFG] rx_gain_set={info['rx_ok']} via {info['rx_attr']} readback={info['rx_rb']}")

    # --- KEY FIX: cyclic TX ---
    try:
        sdr.tx_cyclic_buffer = True
    except Exception:
        pass

    x_zero = np.zeros(args.tx_len, dtype=np.complex64)
    x_tone = build_tone(args.fs, args.tone_hz, args.tx_len, args.tx_amp)

    # helper: switch TX waveform safely
    def set_tx_wave(x):
        try:
            # some versions need destroy buffer before re-tx
            if hasattr(sdr, "tx_destroy_buffer"):
                sdr.tx_destroy_buffer()
        except Exception:
            pass
        sdr.tx(x)
        time.sleep(0.05)
        for _ in range(int(args.warmup_rx)):
            try:
                _ = sdr.rx()
            except Exception:
                pass

    # Warmup RX
    try:
        _ = sdr.rx()
        time.sleep(0.05)
        _ = sdr.rx()
    except Exception:
        pass

    base_lockins = []
    tx_lockins = []

    print("\n== Baseline (TX = zeros, cyclic) ==")
    set_tx_wave(x_zero)
    for i in range(max(3, args.loops // 2)):
        rx_raw, rx2, met, lin = rx_grab(sdr, args.rm_dc, args.fs, args.tone_hz)
        base_lockins.append(lin)
        print(f"[BASE {i:02d}] rms={rms_db(rx_raw):+.2f}dB  toneSNR={met['tone_snr_db']:+6.1f}dB  "
              f"lockin={lin:.3e}  peak_f={met['peak_f']/1e3:+7.1f}kHz  "
              f"DC={db(met['dc_mag']):.1f}dB  Tmag={db(met['tone_mag']):.1f}dB")

    print("\n== TX ON (tone, cyclic) ==")
    set_tx_wave(x_tone)
    hits = 0
    for i in range(args.loops):
        rx_raw, rx2, met, lin = rx_grab(sdr, args.rm_dc, args.fs, args.tone_hz)
        tx_lockins.append(lin)

        # hit: lockin比baseline中位数高，且toneSNR也不太差
        base_med_tmp = float(np.median(base_lockins)) if base_lockins else 1e-12
        delta_i = 20*np.log10((lin + 1e-20) / (base_med_tmp + 1e-20))
        hit = (delta_i > 8.0) and (met["tone_snr_db"] > 6.0)
        hits += int(hit)

        print(f"[TXON {i:02d}] rms={rms_db(rx_raw):+.2f}dB  toneSNR={met['tone_snr_db']:+6.1f}dB  "
              f"lockin={lin:.3e}  ΔvsBASEmed={delta_i:+5.1f}dB  "
              f"peak_f={met['peak_f']/1e3:+7.1f}kHz  "
              f"DC={db(met['dc_mag']):.1f}dB  Tmag={db(met['tone_mag']):.1f}dB  hit={hit}")

    base_med = float(np.median(base_lockins)) if base_lockins else 1e-12
    tx_med = float(np.median(tx_lockins)) if tx_lockins else 1e-12
    delta_db = 20*np.log10((tx_med + 1e-20) / (base_med + 1e-20))

    print("\n==== SUMMARY ====")
    print(f"BASE lockin median = {base_med:.3e}")
    print(f"TXON lockin median = {tx_med:.3e}")
    print(f"Delta(lockin)      = {delta_db:+.1f} dB")
    print(f"TXON hit count     = {hits}/{args.loops}")

    print("\nInterpretation:")
    print("- Delta >= +10 dB 且 hit 很多：TX->ATT->RX 这条 cable 链路基本通。")
    print("- Delta ~ 0 dB 或负：要么 TX 没持续发（cyclic没生效/tx buffer没起来），要么线/衰减器/接法不对，或其实接到同一口造成强自泄漏但非tone。")
    print("- 如果 DC 一直巨大且 peak_f 总在 0 附近：继续减 tx_amp（1e-4/5e-5），必要时再加衰减。")

    # stop TX
    try:
        set_tx_wave(x_zero)
        if hasattr(sdr, "tx_destroy_buffer"):
            sdr.tx_destroy_buffer()
    except Exception:
        pass

if __name__ == "__main__":
    main()