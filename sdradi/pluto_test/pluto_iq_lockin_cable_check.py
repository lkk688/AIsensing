#!/usr/bin/env python3
import time
import argparse
import numpy as np
import adi

def lockin_metrics(x: np.ndarray, fs: float, tone_hz: float):
    """
    Coherent lock-in at tone_hz:
      A = mean( x * exp(-j2πft) )   -> complex tone amplitude estimate
      residual -> noise/interference estimate
    """
    x = x.astype(np.complex64, copy=False)

    # remove DC (important for Pluto leakage)
    x = x - np.mean(x)

    N = len(x)
    t = (np.arange(N, dtype=np.float32) / np.float32(fs))
    lo = np.exp(-1j * 2*np.pi*np.float32(tone_hz)*t).astype(np.complex64)

    y = x * lo
    A = np.mean(y)                         # complex
    tone_amp = np.abs(A)

    y_res = y - A
    noise_rms = np.sqrt(np.mean(np.abs(y_res)**2) + 1e-12)

    tone_snr_db = 20*np.log10((tone_amp + 1e-12) / (noise_rms + 1e-12))

    rms = np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
    p999 = np.percentile(np.abs(x), 99.9)
    p50  = np.percentile(np.abs(x), 50.0)
    sat_ratio = float(p999 / (p50 + 1e-12))

    return {
        "tone_amp": float(tone_amp),
        "tone_snr_db": float(tone_snr_db),
        "rms": float(rms),
        "sat_ratio": float(sat_ratio),
    }

def make_tone(fs: float, tone_hz: float, N: int, amp: float):
    t = np.arange(N, dtype=np.float32) / np.float32(fs)
    tone = amp * np.exp(1j * 2*np.pi*np.float32(tone_hz)*t)
    return tone.astype(np.complex64)

def _set_channels_with_fallback(sdr, kind: str, wanted):
    """
    kind: 'rx' or 'tx'
    wanted: list of ints
    """
    prop = "rx_enabled_channels" if kind == "rx" else "tx_enabled_channels"
    try:
        setattr(sdr, prop, wanted)
        return wanted, True
    except Exception as e:
        # fallback to [0]
        try:
            setattr(sdr, prop, [0])
            return [0], False
        except Exception as e2:
            raise RuntimeError(f"Failed to set {prop} with {wanted} and fallback [0]. "
                               f"err1={e}, err2={e2}")

def config_pluto(uri, fc, fs, bw, tx_gain, rx_gain, rx_buf, want_iq_pair: bool):
    sdr = adi.Pluto(uri)

    # sample rate / LO / BW
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    sdr.rx_rf_bandwidth = int(bw)
    sdr.tx_rf_bandwidth = int(bw)

    # IMPORTANT:
    # Some Pluto setups expose only ONE "enabled channel" (complex IQ stream),
    # even though _rx_channel_names shows voltage0/voltage1.
    wanted = [0, 1] if want_iq_pair else [0]
    rx_set, rx_full = _set_channels_with_fallback(sdr, "rx", wanted)
    tx_set, tx_full = _set_channels_with_fallback(sdr, "tx", wanted)

    # Manual gains
    try:
        sdr.gain_control_mode_chan0 = "manual"
    except Exception:
        pass

    # RX gain
    rx_gain_set = False
    for attr in ("rx_hardwaregain_chan0", "rx_hardwaregain"):
        try:
            setattr(sdr, attr, float(rx_gain))
            rx_gain_set = True
            break
        except Exception:
            pass

    # TX gain
    tx_gain_set = False
    for attr in ("tx_hardwaregain_chan0", "tx_hardwaregain"):
        try:
            setattr(sdr, attr, float(tx_gain))
            tx_gain_set = True
            break
        except Exception:
            pass

    sdr.rx_buffer_size = int(rx_buf)

    # Make sure TX is off initially
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False

    print(f"[CFG] rx_enabled_channels={rx_set} (wanted {wanted}, full={rx_full})")
    print(f"[CFG] tx_enabled_channels={tx_set} (wanted {wanted}, full={tx_full})")
    print(f"[CFG] rx_gain_set={rx_gain_set} tx_gain_set={tx_gain_set}")

    return sdr

def run_block(sdr, fs, tone_hz, loops, label):
    amps = []
    snrs = []
    rmss = []
    sats = []
    for k in range(loops):
        x = sdr.rx()
        met = lockin_metrics(x, fs, tone_hz)
        amps.append(met["tone_amp"])
        snrs.append(met["tone_snr_db"])
        rmss.append(met["rms"])
        sats.append(met["sat_ratio"])
        print(f"[{label} {k:02d}] tone_amp={met['tone_amp']:.3e}  "
              f"tone_snr={met['tone_snr_db']:+5.1f} dB  "
              f"rms={met['rms']:.3e}  sat_ratio={met['sat_ratio']:.2f}")
        time.sleep(0.05)
    return {
        "tone_amp_med": float(np.median(amps)),
        "tone_snr_med": float(np.median(snrs)),
        "rms_med": float(np.median(rmss)),
        "sat_ratio_med": float(np.median(sats)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc", type=float, default=2405e6)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)
    ap.add_argument("--tone_hz", type=float, default=900e3)
    ap.add_argument("--tx_gain", type=float, default=-30)
    ap.add_argument("--rx_gain", type=float, default=0)
    ap.add_argument("--tx_amp", type=float, default=0.05)
    ap.add_argument("--rx_buf", type=int, default=262144)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tx_len", type=int, default=65536)
    ap.add_argument("--try_iq_pair", action="store_true",
                    help="try enabling [0,1] (will auto-fallback to [0] if unsupported)")
    args = ap.parse_args()

    print("==== Pluto IQ lock-in cable check (fallback-safe) ====")
    print(f"[ARGS] uri={args.uri} fc={args.fc/1e6:.3f}MHz fs={args.fs/1e6:.3f}Msps bw={args.bw/1e6:.3f}MHz")
    print(f"[ARGS] tone={args.tone_hz/1e3:.1f}kHz tx_gain={args.tx_gain} rx_gain={args.rx_gain} tx_amp={args.tx_amp}")
    print(f"[ARGS] rx_buf={args.rx_buf} loops={args.loops} tx_len={args.tx_len} try_iq_pair={args.try_iq_pair}")

    sdr = config_pluto(args.uri, args.fc, args.fs, args.bw,
                      args.tx_gain, args.rx_gain, args.rx_buf,
                      want_iq_pair=args.try_iq_pair)

    print("\n== Baseline (TX OFF) ==")
    base = run_block(sdr, args.fs, args.tone_hz, args.loops, "BASE")

    # TX ON (cyclic)
    tone = make_tone(args.fs, args.tone_hz, args.tx_len, args.tx_amp)
    sdr.tx_cyclic_buffer = True
    sdr.tx(tone)
    time.sleep(0.25)

    print("\n== TX ON ==")
    txon = run_block(sdr, args.fs, args.tone_hz, args.loops, "TXON")

    # Stop TX
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False

    gain_db = 20*np.log10((txon["tone_amp_med"] + 1e-12) / (base["tone_amp_med"] + 1e-12))

    print("\n==== SUMMARY ====")
    print(f"BASE: tone_amp_med={base['tone_amp_med']:.3e}  tone_snr_med={base['tone_snr_med']:+.1f} dB  rms_med={base['rms_med']:.3e}")
    print(f"TXON: tone_amp_med={txon['tone_amp_med']:.3e}  tone_snr_med={txon['tone_snr_med']:+.1f} dB  rms_med={txon['rms_med']:.3e}")
    print(f"Delta: tone_amp gain = {gain_db:+.1f} dB")

    print("\nInterpretation:")
    print("- Delta >= +10 dB 且 TXON tone_snr 明显上升：TX->RX cable 链路通。")
    print("- Delta ~ 0 dB：TX 没进入 RX（线/口/衰减器路径/连接）或 TX 没真正发出。")

if __name__ == "__main__":
    main()