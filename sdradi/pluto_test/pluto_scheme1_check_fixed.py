#!/usr/bin/env python3
import argparse, time
import numpy as np
import adi

def db(x, eps=1e-12):
    return 20*np.log10(np.maximum(np.abs(x), eps))

def lockin_amp(x, fs, f0, rm_dc=True):
    """Complex lock-in at f0: amplitude of correlation with exp(-j2πf0t)."""
    x = x.astype(np.complex64)
    if rm_dc:
        x = x - np.mean(x)
    n = len(x)
    t = np.arange(n, dtype=np.float64) / fs
    ref = np.exp(-1j * 2*np.pi * f0 * t).astype(np.complex64)
    # Coherent amplitude estimate (rough): |mean(x*ref)|
    return np.abs(np.mean(x * ref))

def tone_snr_fft(x, fs, f0, rm_dc=True, guard_bins=6, noise_bins=200):
    """SNR around tone bin using FFT; returns (snr_db, tone_mag_db, dc_mag_db, peak_f)."""
    x = x.astype(np.complex64)
    if rm_dc:
        x = x - np.mean(x)

    n = len(x)
    w = np.hanning(n).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * w))
    mag = np.abs(X) + 1e-12
    mag_db = 20*np.log10(mag)

    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))
    # locate expected tone bin
    k0 = int(np.argmin(np.abs(freqs - f0)))
    kdc = int(np.argmin(np.abs(freqs - 0.0)))

    # allow small search around expected bin (freq error)
    search = 30
    ks = np.arange(max(0, k0-search), min(n, k0+search+1))
    k_peak = ks[np.argmax(mag[ks])]
    peak_f = freqs[k_peak]

    tone_mag_db = mag_db[k_peak]
    dc_mag_db = mag_db[kdc]

    # noise estimate: take bins away from tone and away from DC
    mask = np.ones(n, dtype=bool)
    mask[max(0, kdc-guard_bins):min(n, kdc+guard_bins+1)] = False
    mask[max(0, k_peak-guard_bins):min(n, k_peak+guard_bins+1)] = False

    noise_vals = mag_db[mask]
    if len(noise_vals) > noise_bins:
        # take median as robust noise floor
        noise_db = np.median(noise_vals)
    else:
        noise_db = np.median(mag_db)

    snr_db = tone_mag_db - noise_db
    return snr_db, tone_mag_db, dc_mag_db, peak_f

def rx_once(sdr, flush=2):
    for _ in range(flush):
        _ = sdr.rx()
    return sdr.rx()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2405e6)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--tx_ch", type=int, default=0)
    ap.add_argument("--rx_ch", type=int, default=0)
    ap.add_argument("--tone_hz", type=float, default=900e3)
    ap.add_argument("--tx_gain", type=float, default=-80)
    ap.add_argument("--rx_gain", type=float, default=0)
    ap.add_argument("--tx_amp", type=float, default=1e-4, help="tone amplitude (complex float)")
    ap.add_argument("--n", type=int, default=262144)
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--rm_dc", action="store_true")
    ap.add_argument("--settle", type=float, default=0.3)
    args = ap.parse_args()

    print("==== Pluto Scheme-1 FIXED check (TX OFF baseline, cyclic tone TX) ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tx_ch={args.tx_ch} rx_ch={args.rx_ch} tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain} tx_amp={args.tx_amp}")
    print(f"[ARGS] n={args.n} loops={args.loops} rm_dc={args.rm_dc}")

    sdr = adi.Pluto(args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.rx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.bw)
    sdr.rx_rf_bandwidth = int(args.bw)

    # RX gain
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)

    # channel enables
    sdr.rx_enabled_channels = [int(args.rx_ch)]
    sdr.tx_enabled_channels = [int(args.tx_ch)]

    sdr.rx_buffer_size = int(args.n)

    # ---- Baseline: TX truly OFF ----
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False
    time.sleep(args.settle)

    base_amps, base_snrs = [], []
    print("\n== BASELINE (TX OFF) ==")
    for i in range(max(3, args.loops//2)):
        x = rx_once(sdr, flush=2)
        amp = lockin_amp(x, args.fs, args.tone_hz, rm_dc=args.rm_dc)
        snr_db, tone_db, dc_db, peak_f = tone_snr_fft(x, args.fs, args.tone_hz, rm_dc=args.rm_dc)
        base_amps.append(amp); base_snrs.append(snr_db)
        print(f"[BASE {i:02d}] lockin={amp:.3e}  toneSNR={snr_db:+6.1f}dB  peak_f={peak_f/1e3:+7.1f}kHz  DC={dc_db:5.1f}dB  Tone={tone_db:5.1f}dB")

    base_med = np.median(base_amps)
    base_snr_med = np.median(base_snrs)

    # ---- TX ON: cyclic tone ----
    print("\n== TX ON (cyclic tone) ==")
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)  # channel0 gain attribute works for tx_ch=0
    # generate tone buffer
    n = int(args.n)
    t = np.arange(n, dtype=np.float64) / args.fs
    tone = (args.tx_amp * np.exp(1j*2*np.pi*args.tone_hz*t)).astype(np.complex64)
    sdr.tx_cyclic_buffer = True
    sdr.tx(tone)
    time.sleep(args.settle)

    tx_amps, tx_snrs = [], []
    for i in range(args.loops):
        x = rx_once(sdr, flush=2)
        amp = lockin_amp(x, args.fs, args.tone_hz, rm_dc=args.rm_dc)
        snr_db, tone_db, dc_db, peak_f = tone_snr_fft(x, args.fs, args.tone_hz, rm_dc=args.rm_dc)
        tx_amps.append(amp); tx_snrs.append(snr_db)
        delta_db = 20*np.log10((amp+1e-12)/(base_med+1e-12))
        print(f"[TXON {i:02d}] lockin={amp:.3e}  Δ={delta_db:+5.1f}dB  toneSNR={snr_db:+6.1f}dB  peak_f={peak_f/1e3:+7.1f}kHz")

    tx_med = np.median(tx_amps)
    tx_snr_med = np.median(tx_snrs)
    delta_med_db = 20*np.log10((tx_med+1e-12)/(base_med+1e-12))

    # cleanup
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    print("\n==== SUMMARY ====")
    print(f"BASE lockin med = {base_med:.3e}   toneSNR med = {base_snr_med:+.1f} dB")
    print(f"TXON lockin med = {tx_med:.3e}   toneSNR med = {tx_snr_med:+.1f} dB")
    print(f"Delta(lockin)   = {delta_med_db:+.1f} dB")
    print("\nInterpretation:")
    print("- Delta >= +10 dB 且 TXON toneSNR 明显高于 baseline：TX->ATT->RX 链路通。")
    print("- Delta ~ 0 dB：要么 TX tone 没真正出来，要么线/衰减器/接法不对，或 RX 主要是泄漏/杂散。")
    print("- peak_f 若仍总在 0 附近没关系，关键看 lockin/toneSNR 的提升。")

if __name__ == "__main__":
    main()

# python3 pluto_scheme1_check_fixed.py --uri ip:192.168.2.2 \
#   --tx_ch 0 --rx_ch 0 --tone_hz 900e3 \
#   --tx_gain -80 --rx_gain 0 --tx_amp 0.0001 \
#   --rm_dc --loops 10