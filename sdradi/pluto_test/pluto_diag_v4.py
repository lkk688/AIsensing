#!/usr/bin/env python3
"""
pluto_diag_v4.py

Robust Pluto/AD936x diagnostic script:
- Best-effort chip identification evidence (AD9361 vs AD9363 is often not uniquely readable via IIO)
- Digital loopback (BIST) test
- RF path tone test (real cable/air path)
- Matrix test (TXch x RXch)
- Port sweep (rx_port x tx_port)
- Optional unplug-check (interactive) to prove cable effect

Examples:
  # Quick RF test on chosen channel/ports
  python3 pluto_diag_v4.py --uri ip:192.168.3.2 --mode rf --tx_ch 1 --rx_ch 1 --tx_port 1 --rx_port 1

  # RF matrix over channels
  python3 pluto_diag_v4.py --uri ip:192.168.3.2 --mode matrix --tx_port 1 --rx_port 1

  # RF port sweep (rx_port in {0,1}, tx_port in {0,1})
  python3 pluto_diag_v4.py --uri ip:192.168.3.2 --mode port_sweep --tx_ch 1 --rx_ch 1

  # Digital loopback test (BIST)
  python3 pluto_diag_v4.py --uri ip:192.168.3.2 --mode dlb --tx_ch 0 --rx_ch 0

  # Compare RF vs Digital loopback (same tx/rx selection)
  python3 pluto_diag_v4.py --uri ip:192.168.3.2 --mode compare --tx_ch 1 --rx_ch 1 --tx_port 1 --rx_port 1

  # Interactive unplug check (run once plugged, then asks you to unplug and re-run)
  python3 pluto_diag_v4.py --uri ip:192.168.3.2 --mode unplug_check --tx_ch 1 --rx_ch 1 --tx_port 1 --rx_port 1
"""

import argparse
import time
import sys
import numpy as np

try:
    import adi
except Exception as e:
    print("[FATAL] Failed to import pyadi-iio (adi). Ensure pyadi-iio is installed.")
    raise

FULLSCALE = float(2**14)  # Pluto/AD936x typical I/Q full scale in pyadi-iio examples


# -------------------------- Utility helpers -------------------------- #

def _safe_set(obj, name, value):
    """Set attribute if exists; returns (ok, errstr)."""
    try:
        setattr(obj, name, value)
        return True, ""
    except Exception as e:
        return False, str(e)

def _safe_get(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default

def _db(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.asarray(x), floor))

def gen_tone(fs, tone_hz, n, amp=0.5):
    """
    Generate complex tone centered at baseband +tone_hz.
    Returns complex64 scaled to FULLSCALE.
    """
    t = np.arange(n) / float(fs)
    sig = amp * np.exp(1j * 2 * np.pi * float(tone_hz) * t)
    return (sig * FULLSCALE).astype(np.complex64)

def fft_snr_tone(x, fs, tone_hz, exclude_bins=10):
    """
    Compute SNR w.r.t. expected tone bin.
    - tone_db: magnitude at expected bin (dB, arbitrary ref)
    - noise_db: median magnitude excluding neighborhood around tone (dB)
    - snr_db = tone_db - noise_db
    """
    x = np.asarray(x)
    n = len(x)
    if n <= 8:
        return -200.0, -200.0, 0.0

    X = np.fft.fft(x)
    mag = np.abs(X)

    k = int(np.round((float(tone_hz) / float(fs)) * n)) % n
    tone_mag = mag[k]
    tone_db = float(_db(tone_mag, floor=1e-12))

    # Exclude bins around tone and DC
    mask = np.ones(n, dtype=bool)
    for kk in range(-exclude_bins, exclude_bins + 1):
        mask[(k + kk) % n] = False
    for kk in range(-exclude_bins, exclude_bins + 1):
        mask[(0 + kk) % n] = False  # DC neighborhood

    noise_db = float(np.median(_db(mag[mask], floor=1e-12))) if np.any(mask) else -200.0
    snr_db = tone_db - noise_db
    return tone_db, noise_db, snr_db

def summarize_samples(x, fs, tone_hz):
    """
    Return dict of peak/rms/crest, rms_dbfs, tone_db/noise_db/snr_db.
    """
    x = np.asarray(x)
    if x.size == 0:
        return dict(
            peak=0.0, rms=0.0, crest=0.0,
            rms_dbfs=-240.0, peak_dbfs=-240.0,
            tone_db=-200.0, noise_db=-200.0, snr_db=0.0,
        )
    a = np.abs(x)
    peak = float(np.max(a))
    rms = float(np.sqrt(np.mean(a**2)))
    crest = float(peak / (rms + 1e-12))
    rms_dbfs = float(_db(rms / FULLSCALE, floor=1e-12))
    peak_dbfs = float(_db(peak / FULLSCALE, floor=1e-12))
    tone_db, noise_db, snr_db = fft_snr_tone(x, fs, tone_hz)
    return dict(
        peak=peak, rms=rms, crest=crest,
        rms_dbfs=rms_dbfs, peak_dbfs=peak_dbfs,
        tone_db=tone_db, noise_db=noise_db, snr_db=snr_db
    )

def print_iio_summary(sdr):
    ctx = getattr(sdr, "ctx", None)
    if ctx is None:
        print("[IIO] No ctx available via pyadi-iio object.")
        return None, None
    try:
        devs = list(ctx.devices)
        print("\n[IIO] ==== Context / Device Tree (summary) ====")
        print(f"[IIO] Devices: {len(devs)}")
        for i, d in enumerate(devs):
            print(f"[IIO]  - dev[{i}]: {d.name}")
        phy = ctx.find_device("ad9361-phy")
        return ctx, phy
    except Exception as e:
        print(f"[IIO] Failed to enumerate devices: {e}")
        return ctx, None

def best_effort_chip_guess(phy):
    """
    Distinguishing AD9361 vs AD9363 purely from IIO is often not reliable:
    - Many stacks still report AD9361 even if silicon is AD9363.
    We provide evidence strings only.
    """
    guess = "AD9361 (or AD9363 using AD9361 driver)"
    evidence = []

    try:
        # Some builds expose 'model' or similar
        for key in ["model", "compatible", "name"]:
            if key in getattr(phy, "attrs", {}):
                v = phy.attrs[key].value
                evidence.append(f"attr:{key}: {v}")

        # gain_table_config often prints "AD9361" regardless of 9363
        if "gain_table_config" in phy.attrs:
            gt = phy.attrs["gain_table_config"].value
            head = gt.splitlines()[0] if gt else ""
            evidence.append(f"gain_table_config: {head}")

        # a few debug attrs that commonly exist
        for k in ["adi,rf-rx-bandwidth-hz", "adi,rf-tx-bandwidth-hz", "multichip_sync", "loopback"]:
            if k in phy.debug_attrs:
                evidence.append(f"dbg:{k}: {phy.debug_attrs[k].value}")

    except Exception as e:
        evidence.append(f"(chip probe error: {e})")

    # If any evidence explicitly mentions AD9363, update guess
    if any("AD9363" in s for s in evidence):
        guess = "AD9363 (explicit in attrs)"
    return guess, evidence[:6]


# -------------------------- AD936x control -------------------------- #

def connect(uri):
    # Prefer adi.ad9361 if available (2R2T)
    try:
        sdr = adi.ad9361(uri=uri)
        drv = "adi.ad9361"
        return sdr, drv
    except Exception:
        sdr = adi.Pluto(uri=uri)
        drv = "adi.Pluto"
        return sdr, drv

def detect_capabilities(sdr):
    """
    Return dict with RX1/TX1 (channel index 1) availability.
    """
    caps = {"RX1": False, "TX1": False}
    # RX1
    try:
        _ = sdr.rx_hardwaregain_chan1
        caps["RX1"] = True
    except Exception:
        caps["RX1"] = False
    # TX1
    try:
        _ = sdr.tx_hardwaregain_chan1
        caps["TX1"] = True
    except Exception:
        caps["TX1"] = False
    return caps

def destroy_buffers(sdr):
    # Not all versions expose both
    for fn in ["tx_destroy_buffer", "rx_destroy_buffer"]:
        f = getattr(sdr, fn, None)
        if callable(f):
            try:
                f()
            except Exception:
                pass

def set_kernel_buffers(sdr, count):
    # Works on some pyadi-iio builds
    try:
        if hasattr(sdr, "_rxadc") and hasattr(sdr._rxadc, "set_kernel_buffers_count"):
            sdr._rxadc.set_kernel_buffers_count(int(count))
            return True
    except Exception:
        pass
    return False

def set_loopback_bist(phy, enable):
    """
    Enable AD936x internal digital loopback via debug attr 'loopback' when present.
    """
    if phy is None:
        return False, "no phy"
    try:
        if "loopback" in phy.debug_attrs:
            phy.debug_attrs["loopback"].value = "1" if enable else "0"
            return True, ""
        return False, "phy.debug_attrs['loopback'] not found"
    except Exception as e:
        return False, str(e)

def set_ports(phy, rx_port=None, tx_port=None):
    """
    Try to set RF port select. Different trees expose these as debug_attrs.
    We try the keys observed in your logs:
      - adi,rx-rf-port-input-select
      - adi,tx-rf-port-input-select
    """
    if phy is None:
        return False, "no phy"
    ok_all = True
    errs = []
    try:
        if rx_port is not None:
            key = "adi,rx-rf-port-input-select"
            if key in phy.debug_attrs:
                phy.debug_attrs[key].value = str(int(rx_port))
            elif key in phy.attrs:
                phy.attrs[key].value = str(int(rx_port))
            else:
                ok_all = False
                errs.append(f"missing {key}")
        if tx_port is not None:
            key = "adi,tx-rf-port-input-select"
            if key in phy.debug_attrs:
                phy.debug_attrs[key].value = str(int(tx_port))
            elif key in phy.attrs:
                phy.attrs[key].value = str(int(tx_port))
            else:
                ok_all = False
                errs.append(f"missing {key}")
    except Exception as e:
        return False, str(e)
    return ok_all, "; ".join(errs)

def config_common(sdr, fs, fc, bw):
    # Sample rate
    sdr.sample_rate = int(fs)
    # LOs
    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    # Bandwidth (some stacks clamp)
    sdr.rx_rf_bandwidth = int(bw)
    sdr.tx_rf_bandwidth = int(bw)

# def config_gains(sdr, rx_gain, tx_gain, agc=False, caps=None, quiet_other_tx=True):
#     """
#     Configure manual gains and optionally silence the "other" TX channel to avoid leakage confusion.
#     """
#     if caps is None:
#         caps = detect_capabilities(sdr)

#     if agc:
#         _safe_set(sdr, "gain_control_mode_chan0", "slow_attack")
#         if caps["RX1"]:
#             _safe_set(sdr, "gain_control_mode_chan1", "slow_attack")
#     else:
#         _safe_set(sdr, "gain_control_mode_chan0", "manual")
#         _safe_set(sdr, "rx_hardwaregain_chan0", float(rx_gain))
#         if caps["RX1"]:
#             _safe_set(sdr, "gain_control_mode_chan1", "manual")
#             _safe_set(sdr, "rx_hardwaregain_chan1", float(rx_gain))

#     # TX gains
#     _safe_set(sdr, "tx_hardwaregain_chan0", float(tx_gain))
#     if caps["TX1"]:
#         _safe_set(sdr, "tx_hardwaregain_chan1", float(tx_gain))

#     # Silence other TX if requested (helps debugging a LOT)
#     if quiet_other_tx and caps["TX1"]:
#         # The minimum attenuation depends on variant; -89 is common safe "almost off"
#         try:
#             _safe_set(sdr, "tx_hardwaregain_chan0", float(tx_gain))
#             _safe_set(sdr, "tx_hardwaregain_chan1", -89.0 if tx_gain > -89.0 else float(tx_gain))
#         except Exception:
#             pass

def config_gains(sdr, rx_gain, tx_gain, agc=False, caps=None, quiet_other_tx=True, active_tx_ch=0):
    if caps is None:
        caps = detect_capabilities(sdr)

    # RX gain
    if agc:
        _safe_set(sdr, "gain_control_mode_chan0", "slow_attack")
        if caps["RX1"]:
            _safe_set(sdr, "gain_control_mode_chan1", "slow_attack")
    else:
        _safe_set(sdr, "gain_control_mode_chan0", "manual")
        _safe_set(sdr, "rx_hardwaregain_chan0", float(rx_gain))
        if caps["RX1"]:
            _safe_set(sdr, "gain_control_mode_chan1", "manual")
            _safe_set(sdr, "rx_hardwaregain_chan1", float(rx_gain))

    # TX gain: set both first
    _safe_set(sdr, "tx_hardwaregain_chan0", float(tx_gain))
    if caps["TX1"]:
        _safe_set(sdr, "tx_hardwaregain_chan1", float(tx_gain))

    # Quiet the *inactive* TX only
    if quiet_other_tx and caps["TX1"]:
        mute = -89.0
        if int(active_tx_ch) == 0:
            _safe_set(sdr, "tx_hardwaregain_chan1", mute)
        else:
            _safe_set(sdr, "tx_hardwaregain_chan0", mute)


def enable_channels(sdr, tx_ch, rx_ch, caps=None):
    if caps is None:
        caps = detect_capabilities(sdr)
    # Validate
    if tx_ch not in (0, 1) or rx_ch not in (0, 1):
        raise ValueError("tx_ch/rx_ch must be 0 or 1")
    if tx_ch == 1 and not caps["TX1"]:
        raise RuntimeError("TX channel 1 not available")
    if rx_ch == 1 and not caps["RX1"]:
        raise RuntimeError("RX channel 1 not available")

    sdr.tx_enabled_channels = [int(tx_ch)]
    sdr.rx_enabled_channels = [int(rx_ch)]

def start_tx_tone(sdr, fs, tone, n, amp, settle_s):
    """
    Robust TX start:
    - destroy old buffers
    - set cyclic
    - push tone
    - settle
    """
    destroy_buffers(sdr)
    sdr.tx_cyclic_buffer = True
    tx = gen_tone(fs, tone, n, amp=amp)
    sdr.tx(tx)
    time.sleep(float(settle_s))

def stop_tx(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

def rx_capture(sdr, n, flush=3):
    """
    Robust RX capture:
    - set rx_buffer_size
    - flush a few reads
    - return one read
    """
    sdr.rx_buffer_size = int(n)
    for _ in range(int(flush)):
        try:
            _ = sdr.rx()
        except Exception:
            pass
    data = sdr.rx()
    return data


# -------------------------- Measurement flows -------------------------- #

def measure_once(
    sdr, phy,
    fs, fc, bw,
    tx_ch, rx_ch,
    tx_gain, rx_gain,
    tone_hz, n,
    amp=0.5,
    agc=False,
    loopback=False,
    rx_port=None, tx_port=None,
    settle_s=0.25,
    flush=3,
    kernel_buffers=4,
    quiet_other_tx=True,
    verbose=False,
):
    """
    One robust measurement with TX tone ON.
    Returns dict with metrics and metadata.
    """

    caps = detect_capabilities(sdr)

    # Hard reset state: buffers first
    destroy_buffers(sdr)

    # Optionally set kernel buffers
    if kernel_buffers and kernel_buffers > 0:
        set_kernel_buffers(sdr, kernel_buffers)

    # Configure loopback
    if loopback:
        ok, err = set_loopback_bist(phy, True)
        if verbose:
            print(f"[LOOP] loopback=1 -> {'OK' if ok else 'FAIL'} {err if err else ''}".rstrip())
    else:
        ok, err = set_loopback_bist(phy, False)
        if verbose:
            print(f"[LOOP] loopback=0 -> {'OK' if ok else 'FAIL'} {err if err else ''}".rstrip())

    # Ports
    if rx_port is not None or tx_port is not None:
        okp, errp = set_ports(phy, rx_port=rx_port, tx_port=tx_port)
        if verbose:
            print(f"[PORT] set rx_port={rx_port} tx_port={tx_port} -> {'OK' if okp else 'PART/FAIL'} {errp if errp else ''}".rstrip())

    # Common config
    config_common(sdr, fs=fs, fc=fc, bw=bw)
    # Gains
    #config_gains(sdr, rx_gain=rx_gain, tx_gain=tx_gain, agc=agc, caps=caps, quiet_other_tx=quiet_other_tx)
    config_gains(sdr, rx_gain=rx_gain, tx_gain=tx_gain, agc=agc, caps=caps, quiet_other_tx=quiet_other_tx, active_tx_ch=tx_ch)
    # Enable channels (single channel only to avoid confusion)
    enable_channels(sdr, tx_ch=tx_ch, rx_ch=rx_ch, caps=caps)

    # Ensure cyclic TX tone is ON
    start_tx_tone(sdr, fs=fs, tone=tone_hz, n=n, amp=amp, settle_s=settle_s)

    # Capture
    data = rx_capture(sdr, n=n, flush=flush)

    # Stop TX for cleanliness (optional; but helps avoid state bleeding between tests)
    stop_tx(sdr)

    # Metrics
    m = summarize_samples(data, fs=fs, tone_hz=tone_hz)
    m.update(dict(
        fs=float(fs), fc=float(fc), bw=float(bw),
        tx_ch=int(tx_ch), rx_ch=int(rx_ch),
        tx_gain=float(tx_gain), rx_gain=float(rx_gain),
        tone_hz=float(tone_hz), n=int(n),
        loopback=int(1 if loopback else 0),
        rx_port=rx_port, tx_port=tx_port,
    ))
    return m

def print_meas(tag, m):
    rxp = "-" if m.get("rx_port", None) is None else str(m["rx_port"])
    txp = "-" if m.get("tx_port", None) is None else str(m["tx_port"])
    print(f"[MEAS] {tag}  rms_dbfs={m['rms_dbfs']:.2f}  SNR={m['snr_db']:.2f} dB  peak={m['peak']:.2f}  (rx_port={rxp}, tx_port={txp})")
    print(f"[MEAS]         tone_db={m['tone_db']:.2f}  noise_db={m['noise_db']:.2f}  rms={m['rms']:.2f} crest={m['crest']:.2f}")

def run_rf_single(args, sdr, phy):
    m = measure_once(
        sdr, phy,
        fs=args.fs, fc=args.fc, bw=args.bw,
        tx_ch=args.tx_ch, rx_ch=args.rx_ch,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain,
        tone_hz=args.tone, n=args.n,
        amp=args.amp,
        agc=args.agc,
        loopback=False,
        rx_port=args.rx_port, tx_port=args.tx_port,
        settle_s=args.settle,
        flush=args.flush,
        kernel_buffers=args.kernel_buffers,
        quiet_other_tx=not args.no_quiet_other_tx,
        verbose=True,
    )
    print_meas(f"RF TX{args.tx_ch}->RX{args.rx_ch}", m)

def run_dlb_single(args, sdr, phy):
    m = measure_once(
        sdr, phy,
        fs=args.fs, fc=args.fc, bw=args.bw,
        tx_ch=args.tx_ch, rx_ch=args.rx_ch,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain,
        tone_hz=args.tone, n=args.n,
        amp=args.amp,
        agc=args.agc,
        loopback=True,
        rx_port=args.rx_port, tx_port=args.tx_port,
        settle_s=args.settle,
        flush=args.flush,
        kernel_buffers=args.kernel_buffers,
        quiet_other_tx=not args.no_quiet_other_tx,
        verbose=True,
    )
    print_meas(f"DLB TX{args.tx_ch}->RX{args.rx_ch}", m)
    ok, err = set_loopback_bist(phy, False)
    print(f"[LOOP] loopback=0 -> {'OK' if ok else 'FAIL'} {err if err else ''}".rstrip())

def run_compare(args, sdr, phy):
    print("\n==============================")
    print(" MODE: compare (RF vs Digital loopback)")
    print("==============================")
    print(f"[ARGS] uri={args.uri}  fc={args.fc/1e6:.3f} MHz  fs={args.fs/1e6:.3f} Msps  bw={args.bw/1e6:.3f} MHz")
    print(f"[ARGS] tx_ch={args.tx_ch} rx_ch={args.rx_ch}  tx_gain={args.tx_gain}  rx_gain={args.rx_gain}  tone={args.tone/1e3:.1f} kHz  N={args.n}")
    print(f"[ARGS] rx_port={args.rx_port} tx_port={args.tx_port}  agc={args.agc} settle={args.settle}s")

    m_rf = measure_once(
        sdr, phy,
        fs=args.fs, fc=args.fc, bw=args.bw,
        tx_ch=args.tx_ch, rx_ch=args.rx_ch,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain,
        tone_hz=args.tone, n=args.n,
        amp=args.amp,
        agc=args.agc,
        loopback=False,
        rx_port=args.rx_port, tx_port=args.tx_port,
        settle_s=args.settle,
        flush=args.flush,
        kernel_buffers=args.kernel_buffers,
        quiet_other_tx=not args.no_quiet_other_tx,
        verbose=True,
    )
    print_meas("RF", m_rf)

    m_dlb = measure_once(
        sdr, phy,
        fs=args.fs, fc=args.fc, bw=args.bw,
        tx_ch=args.tx_ch, rx_ch=args.rx_ch,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain,
        tone_hz=args.tone, n=args.n,
        amp=args.amp,
        agc=args.agc,
        loopback=True,
        rx_port=args.rx_port, tx_port=args.tx_port,
        settle_s=args.settle,
        flush=args.flush,
        kernel_buffers=args.kernel_buffers,
        quiet_other_tx=not args.no_quiet_other_tx,
        verbose=True,
    )
    print_meas("DLB", m_dlb)
    ok, err = set_loopback_bist(phy, False)
    print(f"[LOOP] loopback=0 -> {'OK' if ok else 'FAIL'} {err if err else ''}".rstrip())

    print("\n[SUMMARY]")
    print(f"  RF : rms_dbfs={m_rf['rms_dbfs']:.2f}  snr={m_rf['snr_db']:.2f} dB  tone_db={m_rf['tone_db']:.2f}")
    print(f"  DLB: rms_dbfs={m_dlb['rms_dbfs']:.2f}  snr={m_dlb['snr_db']:.2f} dB  tone_db={m_dlb['tone_db']:.2f}")
    print("[HINT] DLB 通常很强很稳定；若 DLB 强但 RF 弱，多半是物理链路/端口/增益/连接问题。")

def run_matrix(args, sdr, phy, loopback=False):
    print("\n==============================")
    print(f" MODE: {'DLB' if loopback else 'RF'} matrix (TXch x RXch)")
    print("==============================")
    caps = detect_capabilities(sdr)
    tx_chs = [0, 1] if caps["TX1"] else [0]
    rx_chs = [0, 1] if caps["RX1"] else [0]

    results = []
    for txc in tx_chs:
        for rxc in rx_chs:
            try:
                m = measure_once(
                    sdr, phy,
                    fs=args.fs, fc=args.fc, bw=args.bw,
                    tx_ch=txc, rx_ch=rxc,
                    tx_gain=args.tx_gain, rx_gain=args.rx_gain,
                    tone_hz=args.tone, n=args.n,
                    amp=args.amp,
                    agc=args.agc,
                    loopback=loopback,
                    rx_port=args.rx_port, tx_port=args.tx_port,
                    settle_s=args.settle,
                    flush=args.flush,
                    kernel_buffers=args.kernel_buffers,
                    quiet_other_tx=not args.no_quiet_other_tx,
                    verbose=False,
                )
                results.append(m)
                tag = f"{'DLB' if loopback else 'RF'} TX{txc}->RX{rxc}"
                print_meas(tag, m)
            except Exception as e:
                print(f"[ERR] {'DLB' if loopback else 'RF'} TX{txc}->RX{rxc}: {e}")

    if loopback:
        set_loopback_bist(phy, False)

def run_port_sweep(args, sdr, phy, loopback=False):
    print("\n==============================")
    print(f" MODE: {'DLB' if loopback else 'RF'} port_sweep (rx_port x tx_port)")
    print("==============================")
    results = []
    for rxp in [0, 1]:
        for txp in [0, 1]:
            try:
                m = measure_once(
                    sdr, phy,
                    fs=args.fs, fc=args.fc, bw=args.bw,
                    tx_ch=args.tx_ch, rx_ch=args.rx_ch,
                    tx_gain=args.tx_gain, rx_gain=args.rx_gain,
                    tone_hz=args.tone, n=args.n,
                    amp=args.amp,
                    agc=args.agc,
                    loopback=loopback,
                    rx_port=rxp, tx_port=txp,
                    settle_s=args.settle,
                    flush=args.flush,
                    kernel_buffers=args.kernel_buffers,
                    quiet_other_tx=not args.no_quiet_other_tx,
                    verbose=False,
                )
                results.append(m)
                tag = f"{'DLB' if loopback else 'RF'} TX{args.tx_ch}->RX{args.rx_ch}"
                print_meas(tag, m)
            except Exception as e:
                print(f"[ERR] {'DLB' if loopback else 'RF'} rx_port={rxp} tx_port={txp}: {e}")

    if loopback:
        set_loopback_bist(phy, False)

def run_unplug_check(args, sdr, phy):
    print("\n==============================")
    print(" MODE: unplug_check (interactive)")
    print("==============================")
    print("[STEP] Measurement with cable CONNECTED (as-is).")
    m1 = measure_once(
        sdr, phy,
        fs=args.fs, fc=args.fc, bw=args.bw,
        tx_ch=args.tx_ch, rx_ch=args.rx_ch,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain,
        tone_hz=args.tone, n=args.n,
        amp=args.amp,
        agc=args.agc,
        loopback=False,
        rx_port=args.rx_port, tx_port=args.tx_port,
        settle_s=args.settle,
        flush=args.flush,
        kernel_buffers=args.kernel_buffers,
        quiet_other_tx=not args.no_quiet_other_tx,
        verbose=True,
    )
    print_meas("RF (CONNECTED)", m1)

    print("\n[STEP] Now UNPLUG only the RX-side of the cable (or remove attenuator) and press Enter.")
    try:
        input()
    except KeyboardInterrupt:
        print("\n[ABORT]")
        return

    m2 = measure_once(
        sdr, phy,
        fs=args.fs, fc=args.fc, bw=args.bw,
        tx_ch=args.tx_ch, rx_ch=args.rx_ch,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain,
        tone_hz=args.tone, n=args.n,
        amp=args.amp,
        agc=args.agc,
        loopback=False,
        rx_port=args.rx_port, tx_port=args.tx_port,
        settle_s=args.settle,
        flush=args.flush,
        kernel_buffers=args.kernel_buffers,
        quiet_other_tx=not args.no_quiet_other_tx,
        verbose=True,
    )
    print_meas("RF (UNPLUGGED)", m2)

    print("\n[VERDICT]")
    dr = m1["rms_dbfs"] - m2["rms_dbfs"]
    ds = m1["snr_db"] - m2["snr_db"]
    print(f"  delta_rms_dbfs (connected - unplugged) = {dr:.2f} dB")
    print(f"  delta_snr_db (connected - unplugged)   = {ds:.2f} dB")
    if dr > 10 and m1["snr_db"] > 20:
        print("  => STRONG evidence of direct/cable coupling on the selected path.")
    elif m1["snr_db"] > 20 and m2["snr_db"] > 20:
        print("  => Both look strong: likely leakage/other path still active or wrong port/channel selection.")
    else:
        print("  => Weak or ambiguous: increase RX gain a bit, ensure TX really outputs, verify port/channel mapping.")


# -------------------------- Main -------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--mode", default="rf",
                    choices=["rf", "dlb", "compare", "matrix", "dlb_matrix", "port_sweep", "dlb_port_sweep", "unplug_check"])

    ap.add_argument("--fc", type=float, default=2405e6)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--bw", type=float, default=3e6)

    ap.add_argument("--tx_ch", type=int, default=1, help="TX logical channel index: 0 or 1")
    ap.add_argument("--rx_ch", type=int, default=1, help="RX logical channel index: 0 or 1")

    ap.add_argument("--tx_gain", type=float, default=-20.0, help="TX hardware gain (dB), typical range ~[-89..0]")
    ap.add_argument("--rx_gain", type=float, default=20.0, help="RX hardware gain (dB), typical range ~[0..70+]")

    ap.add_argument("--tone", type=float, default=100e3, help="Baseband tone frequency (Hz)")
    ap.add_argument("--n", type=int, default=32768, help="Number of samples for TX/RX buffers")
    ap.add_argument("--amp", type=float, default=0.5, help="Tone amplitude (0..1). Reduce if you clip/saturate.")

    ap.add_argument("--rx_port", type=int, default=None, help="RF rx port select (0/1) if supported")
    ap.add_argument("--tx_port", type=int, default=None, help="RF tx port select (0/1) if supported")

    ap.add_argument("--agc", action="store_true", help="Use AGC (slow_attack) instead of manual gain")
    ap.add_argument("--settle", type=float, default=0.25, help="Seconds to wait after starting TX")
    ap.add_argument("--flush", type=int, default=3, help="Number of RX flush reads before capture")
    ap.add_argument("--kernel_buffers", type=int, default=4, help="Kernel RX buffer count (if supported)")

    ap.add_argument("--verbose_phy", action="store_true", help="Print IIO device summary and chip evidence")
    ap.add_argument("--no_quiet_other_tx", action="store_true", help="Do NOT try to silence the other TX channel")

    args = ap.parse_args()

    print(f"[CONN] uri={args.uri}")
    sdr, drv = connect(args.uri)
    print(f"[CONN] driver={drv}")
    caps = detect_capabilities(sdr)
    print(f"[CAP] TX1={caps['TX1']}  RX1={caps['RX1']}")

    ctx, phy = print_iio_summary(sdr)

    if args.verbose_phy:
        if phy is None:
            print("[PHY] ad9361-phy not found.")
        else:
            guess, evidence = best_effort_chip_guess(phy)
            print("\n[CHIP] Best-effort guess:", guess)
            for e in evidence:
                print("  -", e)
            # Print a few relevant debug attrs if present
            for k in ["ensm_mode", "ensm_mode_available", "rx_path_rates", "tx_path_rates"]:
                try:
                    if k in phy.attrs:
                        print(f"[PHY] attr:{k}: {phy.attrs[k].value}")
                except Exception:
                    pass
            for k in ["adi,rx-rf-port-input-select", "adi,tx-rf-port-input-select", "loopback"]:
                try:
                    if phy and k in phy.debug_attrs:
                        print(f"[PHY] dbg:{k}: {phy.debug_attrs[k].value}")
                except Exception:
                    pass

    try:
        if args.mode == "rf":
            print("\n=== RF single test ===")
            run_rf_single(args, sdr, phy)
        elif args.mode == "dlb":
            print("\n=== Digital loopback (BIST) single test ===")
            run_dlb_single(args, sdr, phy)
        elif args.mode == "compare":
            run_compare(args, sdr, phy)
        elif args.mode == "matrix":
            run_matrix(args, sdr, phy, loopback=False)
        elif args.mode == "dlb_matrix":
            run_matrix(args, sdr, phy, loopback=True)
        elif args.mode == "port_sweep":
            run_port_sweep(args, sdr, phy, loopback=False)
        elif args.mode == "dlb_port_sweep":
            run_port_sweep(args, sdr, phy, loopback=True)
        elif args.mode == "unplug_check":
            run_unplug_check(args, sdr, phy)
        else:
            raise ValueError("Unknown mode")
    finally:
        # Always attempt to disable loopback and stop TX
        try:
            if phy is not None:
                set_loopback_bist(phy, False)
        except Exception:
            pass
        stop_tx(sdr)
        destroy_buffers(sdr)

if __name__ == "__main__":
    main()