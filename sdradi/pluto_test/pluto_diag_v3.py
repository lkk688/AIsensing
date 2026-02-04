#!/usr/bin/env python3
# pluto_diag_v3.py
#
# Robust Pluto/AD936x diagnostic:
#  - Best-effort chip ID evidence (AD9361 vs AD9363)
#  - Digital loopback test (ad9361-phy debug_attr: loopback)
#  - RF path test (real OTA/cable path)
#  - Optional matrix sweep across TX/RX channels and RF port selects (if exposed)
#
# Notes:
#  - On some designs, loopback may only route to RX0 (not RX1). Use --matrix to confirm.
#  - Some firmwares expose port selects:
#      adi,rx-rf-port-input-select and adi,tx-rf-port-input-select
#    If present, you can set via --rx_port/--tx_port or sweep via --port_sweep.

import argparse
import time
import numpy as np
import adi
import sys
import traceback

FULLSCALE = 2**14

def db10(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))

def dbfs_from_rms(rms, fullscale=FULLSCALE):
    return 20.0 * np.log10(np.maximum(rms / fullscale, 1e-12))

def tone_metrics(iq, fs, tone_hz, guard_bins=6):
    iq = np.asarray(iq).astype(np.complex64)
    N = iq.size
    if N < 1024:
        return 0.0, -200.0, -200.0, -200.0

    w = np.hanning(N).astype(np.float64)
    X = np.fft.fftshift(np.fft.fft(iq * w))
    P = (np.abs(X) ** 2).astype(np.float64)
    Pdb = db10(P)

    k = int(np.round(tone_hz / fs * N))          # bin index in [-N/2, N/2)
    k = np.clip(k, -N//2 + 1, N//2 - 1)
    idx = k + N//2

    lo = max(0, idx - guard_bins)
    hi = min(N, idx + guard_bins + 1)

    mask = np.ones(N, dtype=bool)
    mask[lo:hi] = False

    tone_db = float(np.max(Pdb[lo:hi]))
    noise_db = float(np.median(Pdb[mask]))
    snr_db = tone_db - noise_db
    peak_db = float(np.max(Pdb))
    return snr_db, tone_db, noise_db, peak_db

def gen_tone(fs, tone_hz, amp=0.2, n=32768):
    t = np.arange(n) / fs
    x = amp * np.exp(1j * 2*np.pi*tone_hz*t)
    return (x * FULLSCALE).astype(np.complex64)

def connect(uri, prefer_ad9361=True):
    if prefer_ad9361:
        try:
            s = adi.ad9361(uri=uri)
            return s, "adi.ad9361"
        except Exception:
            pass
    s = adi.Pluto(uri=uri)
    return s, "adi.Pluto"

def stop_buffers(sdr):
    try: sdr.tx_destroy_buffer()
    except Exception: pass
    try: sdr.rx_destroy_buffer()
    except Exception: pass

def get_ctx_phy(sdr):
    ctx = getattr(sdr, "ctx", None)
    if ctx is None:
        return None, None
    try:
        phy = ctx.find_device("ad9361-phy")
    except Exception:
        phy = None
    return ctx, phy

def set_debug_attr(phy, key, value_str):
    if phy is None:
        return False, "phy=None"
    try:
        if key not in phy.debug_attrs:
            return False, f"missing:{key}"
        phy.debug_attrs[key].value = str(value_str)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def set_loopback(phy, mode_int):
    # mode_int: 0(off), 1(loopback), 2(alt) - depends on firmware
    ok, msg = set_debug_attr(phy, "loopback", str(int(mode_int)))
    return ok, msg

def set_port_selects(phy, rx_port=None, tx_port=None):
    notes = []
    if rx_port is not None:
        ok, msg = set_debug_attr(phy, "adi,rx-rf-port-input-select", str(int(rx_port)))
        notes.append(("rx_port", ok, msg))
    if tx_port is not None:
        ok, msg = set_debug_attr(phy, "adi,tx-rf-port-input-select", str(int(tx_port)))
        notes.append(("tx_port", ok, msg))
    return notes

def has_ch(sdr, kind, ch):
    # kind: "rx" or "tx"
    try:
        if kind == "rx":
            getattr(sdr, f"rx_hardwaregain_chan{ch}")
        else:
            getattr(sdr, f"tx_hardwaregain_chan{ch}")
        return True
    except Exception:
        return False

def force_manual_gain(sdr, rx_gain0=None, rx_gain1=None):
    # Force both channels to manual if available (prevents weird CH1 states affecting things)
    if rx_gain0 is not None:
        try:
            sdr.gain_control_mode_chan0 = "manual"
            sdr.rx_hardwaregain_chan0 = int(rx_gain0)
        except Exception:
            pass
    if rx_gain1 is not None:
        try:
            sdr.gain_control_mode_chan1 = "manual"
            sdr.rx_hardwaregain_chan1 = int(rx_gain1)
        except Exception:
            pass

def maybe_set_agc(sdr, enable, ch):
    if not enable:
        return
    try:
        if ch == 0:
            sdr.gain_control_mode_chan0 = "fast_attack"
        else:
            sdr.gain_control_mode_chan1 = "fast_attack"
    except Exception:
        pass

def config_common(sdr, fc, fs, bw):
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    try: sdr.rx_rf_bandwidth = int(bw)
    except Exception: pass
    try: sdr.tx_rf_bandwidth = int(bw)
    except Exception: pass

def capture_one(sdr, rx_ch, n, kbuf, flush):
    sdr.rx_enabled_channels = [rx_ch]
    sdr.rx_buffer_size = int(n)

    if hasattr(sdr, "_rxadc") and hasattr(sdr._rxadc, "set_kernel_buffers_count"):
        try: sdr._rxadc.set_kernel_buffers_count(int(kbuf))
        except Exception: pass

    for _ in range(int(flush)):
        _ = sdr.rx()

    iq = sdr.rx()
    iq = np.asarray(iq)
    # Robust: if firmware returns multi-chan array anyway, pick the channel we asked for
    if iq.ndim == 2:
        # common shapes: (2, N) or (N, 2)
        if iq.shape[0] in (1, 2, 4) and iq.shape[1] == n:
            iq = iq[min(rx_ch, iq.shape[0]-1), :]
        elif iq.shape[1] in (1, 2, 4) and iq.shape[0] == n:
            iq = iq[:, min(rx_ch, iq.shape[1]-1)]
        else:
            iq = iq.reshape(-1)
    return iq.astype(np.complex64)

def measure_path(sdr, phy, *, name, loopback_mode, fc, fs, bw,
                 tx_ch, rx_ch, tx_gain, rx_gain, tone_hz,
                 n, settle, kbuf, flush, iters,
                 rx_port=None, tx_port=None, amp=0.2, guard_bins=6, quiet=False):

    if not quiet:
        print(f"\n=== {name} ===")

    stop_buffers(sdr)
    time.sleep(0.05)

    # Loopback
    if phy is not None:
        ok, msg = set_loopback(phy, loopback_mode)
        if not quiet:
            print(f"[LOOP] loopback={loopback_mode} -> {msg}")
    else:
        if not quiet:
            print("[LOOP] phy=None (cannot control loopback)")

    # Port selects (optional)
    port_notes = set_port_selects(phy, rx_port=rx_port, tx_port=tx_port) if phy is not None else []
    if (not quiet) and port_notes:
        for who, ok, msg in port_notes:
            print(f"[PORT] {who} set -> {msg}")

    config_common(sdr, fc, fs, bw)

    # Gains
    if rx_gain is not None:
        # Keep both channels in sane manual unless AGC requested
        if isinstance(rx_gain, (list, tuple)) and len(rx_gain) >= 2:
            force_manual_gain(sdr, rx_gain0=rx_gain[0], rx_gain1=rx_gain[1])
        else:
            force_manual_gain(sdr, rx_gain0=rx_gain, rx_gain1=rx_gain)

    # TX gain
    try:
        setattr(sdr, f"tx_hardwaregain_chan{tx_ch}", int(tx_gain))
    except Exception:
        pass

    # TX enable + send tone
    tx = gen_tone(fs, tone_hz, amp=amp, n=n)
    sdr.tx_enabled_channels = [tx_ch]
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx)
    time.sleep(settle)

    # RX capture & average metrics
    snrs, rms_dbfss, peaks = [], [], []
    tone_dbs, noise_dbs = [], []
    for _ in range(int(iters)):
        iq = capture_one(sdr, rx_ch=rx_ch, n=n, kbuf=kbuf, flush=flush)
        mag = np.abs(iq)
        rms = float(np.sqrt(np.mean(mag**2)))
        peak = float(np.max(mag))
        rms_dbfs = dbfs_from_rms(rms)
        snr, tone_db, noise_db, peak_db = tone_metrics(iq, fs, tone_hz, guard_bins=guard_bins)

        snrs.append(snr); rms_dbfss.append(rms_dbfs); peaks.append(peak)
        tone_dbs.append(tone_db); noise_dbs.append(noise_db)

    stop_buffers(sdr)

    res = {
        "tx_ch": tx_ch, "rx_ch": rx_ch,
        "loopback": loopback_mode,
        "rx_port": rx_port, "tx_port": tx_port,
        "rms_dbfs_mean": float(np.mean(rms_dbfss)),
        "snr_db_mean": float(np.mean(snrs)),
        "peak_mean": float(np.mean(peaks)),
        "tone_db_mean": float(np.mean(tone_dbs)),
        "noise_db_mean": float(np.mean(noise_dbs)),
    }

    if not quiet:
        print(f"[MEAS] TX{tx_ch}->RX{rx_ch}  rms_dbfs={res['rms_dbfs_mean']:.2f}  "
              f"SNR={res['snr_db_mean']:.2f} dB  peak={res['peak_mean']:.2f}")
        print(f"[MEAS] tone_db={res['tone_db_mean']:.2f}  noise_db={res['noise_db_mean']:.2f} "
              f"(rx_port={rx_port}, tx_port={tx_port})")
    return res

def best_effort_chip_id(phy):
    evidence = []
    guess = "UNKNOWN"
    if phy is None:
        return guess, evidence

    # gain_table_config often contains "AD9361"
    try:
        gtc = phy.attrs["gain_table_config"].value
        snippet = (gtc[:140] + "...") if len(gtc) > 140 else gtc
        evidence.append(("gain_table_config", snippet))
        up = gtc.upper()
        if "AD9363" in up:
            guess = "AD9363"
        elif "AD9361" in up:
            guess = "AD9361 (or AD9363 using AD9361 driver)"
    except Exception:
        pass

    # scan attrs/debug_attrs for useful strings
    def scan(d, tag):
        nonlocal guess
        try:
            keys = list(d.keys())
        except Exception:
            return
        for k in sorted(keys):
            lk = k.lower()
            if any(t in lk for t in ["part", "product", "chip", "id", "revision", "silicon", "variant"]):
                try:
                    v = d[k].value
                    vs = str(v)
                    evidence.append((f"{tag}:{k}", (vs[:200] + "...") if len(vs) > 200 else vs))
                    up = vs.upper()
                    if "AD9363" in up:
                        guess = "AD9363"
                    elif "AD9361" in up and guess == "UNKNOWN":
                        guess = "AD9361 (or AD9363 using AD9361 driver)"
                except Exception:
                    pass

    try: scan(phy.attrs, "attr")
    except Exception: pass
    try: scan(phy.debug_attrs, "dbg")
    except Exception: pass

    return guess, evidence

def main():
    ap = argparse.ArgumentParser(description="Robust Pluto/AD936x diag (chip id, loopback, RF path, channel/port sweeps)")
    ap.add_argument("--uri", default="ip:192.168.3.2", help="IIO URI (default: ip:192.168.3.2)")
    ap.add_argument("--prefer_ad9361", action="store_true", help="Try adi.ad9361 first (default behavior is to try it anyway)")

    # RF params
    ap.add_argument("--fc", type=float, default=2405e6, help="Center freq (Hz)")
    ap.add_argument("--fs", type=float, default=3e6, help="Sample rate (Hz)")
    ap.add_argument("--bw", type=float, default=None, help="RF bandwidth (Hz), default=fs")
    ap.add_argument("--tone", type=float, default=100e3, help="Tone offset (Hz)")
    ap.add_argument("--amp", type=float, default=0.2, help="Tone amplitude (0..1, before scaling)")

    # Buffers / stability
    ap.add_argument("--n", type=int, default=32768, help="RX buffer size / tone length")
    ap.add_argument("--settle", type=float, default=0.4, help="Seconds to wait after TX start")
    ap.add_argument("--kbuf", type=int, default=4, help="Kernel buffers count (if supported)")
    ap.add_argument("--flush", type=int, default=6, help="RX flush reads before capture")
    ap.add_argument("--iters", type=int, default=1, help="Measure iterations to average")

    # Gains
    ap.add_argument("--tx_gain", type=float, default=-20, help="TX gain (dB), Pluto often -60..0")
    ap.add_argument("--rx_gain", type=float, default=40, help="RX gain (dB) for both channels (manual)")
    ap.add_argument("--agc", action="store_true", help="Use AGC (fast_attack) on the selected RX channel (overrides manual for that channel)")

    # Port selects (if exposed by firmware)
    ap.add_argument("--rx_port", type=int, default=None, help="Set adi,rx-rf-port-input-select (e.g., 0/1) if available")
    ap.add_argument("--tx_port", type=int, default=None, help="Set adi,tx-rf-port-input-select (e.g., 0/1) if available")
    ap.add_argument("--port_sweep", action="store_true", help="Sweep rx_port/tx_port over {0,1} (if available)")

    # Which tests to run
    ap.add_argument("--loopback_mode", type=int, default=1, help="Loopback mode value to use for DLB test (default 1)")
    ap.add_argument("--rf_test", action="store_true", help="Run RF path test (loopback=0)")
    ap.add_argument("--dlb_test", action="store_true", help="Run digital loopback test (loopback=--loopback_mode)")
    ap.add_argument("--matrix", action="store_true", help="Sweep channel pairs TX{0,1} x RX{0,1} (skips unavailable ch)")
    ap.add_argument("--quiet", action="store_true", help="Less prints (still prints summary)")

    args = ap.parse_args()
    bw = args.bw if args.bw is not None else args.fs

    sdr, drv = connect(args.uri, prefer_ad9361=True)
    ctx, phy = get_ctx_phy(sdr)

    if not args.quiet:
        print(f"[CONN] uri={args.uri}  driver={drv}")
    tx1 = has_ch(sdr, "tx", 1)
    rx1 = has_ch(sdr, "rx", 1)
    if not args.quiet:
        print(f"[CAP] TX1={tx1}  RX1={rx1}")

    # Chip ID evidence (best effort)
    guess, evidence = best_effort_chip_id(phy)
    print(f"\n[CHIP] Best-effort guess: {guess}")
    if evidence:
        for k, v in evidence[:6]:
            print(f"  - {k}: {v}")

    # Decide what to run if user didn't choose
    if not (args.rf_test or args.dlb_test):
        # default: run both
        args.rf_test = True
        args.dlb_test = True

    # Build channel list
    tx_chs = [0] + ([1] if tx1 else [])
    rx_chs = [0] + ([1] if rx1 else [])

    # Build port list
    if args.port_sweep:
        port_pairs = [(0,0), (0,1), (1,0), (1,1)]
    else:
        port_pairs = [(args.rx_port, args.tx_port)]

    results = []

    def run_group(group_name, loopback_val):
        nonlocal results
        for (rp, tp) in port_pairs:
            if args.matrix:
                for t in tx_chs:
                    for r in rx_chs:
                        res = measure_path(
                            sdr, phy,
                            name=f"{group_name} TX{t}->RX{r}",
                            loopback_mode=loopback_val,
                            fc=args.fc, fs=args.fs, bw=bw,
                            tx_ch=t, rx_ch=r,
                            tx_gain=args.tx_gain,
                            rx_gain=args.rx_gain,
                            tone_hz=args.tone,
                            n=args.n, settle=args.settle,
                            kbuf=args.kbuf, flush=args.flush, iters=args.iters,
                            rx_port=rp, tx_port=tp,
                            amp=args.amp,
                            quiet=args.quiet
                        )
                        res["group"] = group_name
                        results.append(res)
            else:
                # default pair: TX1->RX1 if available, else TX0->RX0
                t = 1 if tx1 else 0
                r = 1 if rx1 else 0
                res = measure_path(
                    sdr, phy,
                    name=f"{group_name} TX{t}->RX{r}",
                    loopback_mode=loopback_val,
                    fc=args.fc, fs=args.fs, bw=bw,
                    tx_ch=t, rx_ch=r,
                    tx_gain=args.tx_gain,
                    rx_gain=args.rx_gain,
                    tone_hz=args.tone,
                    n=args.n, settle=args.settle,
                    kbuf=args.kbuf, flush=args.flush, iters=args.iters,
                    rx_port=rp, tx_port=tp,
                    amp=args.amp,
                    quiet=args.quiet
                )
                res["group"] = group_name
                results.append(res)

    try:
        if args.dlb_test:
            run_group("DLB", args.loopback_mode)
        if args.rf_test:
            run_group("RF", 0)

    finally:
        # Always leave loopback off
        if phy is not None:
            try: set_loopback(phy, 0)
            except Exception: pass
        stop_buffers(sdr)

    # Summary table (compact)
    print("\n=== SUMMARY (mean over iters) ===")
    print("group  tx->rx  lb  rx_port tx_port  rms_dbfs    snr_db")
    print("-----  ------  --  ------- ------  ---------  --------")
    for res in results:
        txrx = f"{res['tx_ch']}->{res['rx_ch']}"
        rp = "-" if res["rx_port"] is None else str(res["rx_port"])
        tp = "-" if res["tx_port"] is None else str(res["tx_port"])
        print(f"{res['group']:<5}  {txrx:<6}  {res['loopback']:<2}  {rp:^7} {tp:^6}  "
              f"{res['rms_dbfs_mean']:>8.2f}  {res['snr_db_mean']:>8.2f}")

    print("\n[INTERPRET]")
    print("  - RF: SNR > ~10-15 dB => tone clearly received on that TX/RX/port combination.")
    print("  - DLB: If loopback is wired for that channel/path, SNR is usually very high and stable.")
    print("  - If DLB works on RX0 but not RX1, that’s normal on some designs (loopback routed to RX0 only).")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C")
        sys.exit(0)
    except Exception as e:
        print("\n[MAIN] FATAL:", e)
        traceback.print_exc()
        sys.exit(1)

#python3 pluto_diag_v3.py --uri ip:192.168.3.2