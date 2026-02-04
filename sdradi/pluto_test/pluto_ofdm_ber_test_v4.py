#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pluto_ofdm_ber_test_v4.py
- ZC sync with *repeated* ZC preamble: ZC || ZC
- CFO estimated from repetition (Schmidl-Cox style)
- 64-FFT, CP=16, 48 data + 4 pilots (802.11-like)
- fftshift-consistent mapping
"""

import argparse
import time
import numpy as np
import sys

try:
    import adi
except ImportError:
    print("[FATAL] pyadi-iio not installed. pip install pyadi-iio")
    sys.exit(1)

# -----------------------------
# Presets
# -----------------------------
PRESETS = {
    "cable_30db":     {"tx_gain": -10, "rx_gain": 40, "tx_amp": 0.3},
    "cable_direct":   {"tx_gain": -30, "rx_gain": 20, "tx_amp": 0.08},
    "antenna_close":  {"tx_gain": 0,   "rx_gain": 60, "tx_amp": 0.6},
    "antenna_far":    {"tx_gain": 0,   "rx_gain": 70, "tx_amp": 0.8},
}

# -----------------------------
# QPSK (Gray-ish self-consistent)
# -----------------------------
def qpsk_mod(bits):
    bits = bits.astype(np.int32)
    bits = bits.reshape(-1, 2)
    # 00 -> +1 +j
    # 01 -> +1 -j
    # 10 -> -1 +j
    # 11 -> -1 -j
    re = 1 - 2*bits[:, 0]
    im = 1 - 2*bits[:, 1]
    return (re + 1j*im) / np.sqrt(2)

def qpsk_demod(sym):
    b0 = (np.real(sym) < 0).astype(np.int32)
    b1 = (np.imag(sym) < 0).astype(np.int32)
    out = np.stack([b0, b1], axis=1).reshape(-1)
    return out

# -----------------------------
# ZC preamble (repeated)
# -----------------------------
def make_zc(L=1024, root=29):
    # L should be prime-ish relative to root; 1024 isn't prime but works ok for sync.
    # If you want "true CAZAC", pick prime length (e.g., 839, 1021).
    n = np.arange(L)
    zc = np.exp(-1j * np.pi * root * n * (n + 1) / L)
    zc = zc / np.sqrt(np.mean(np.abs(zc)**2))
    return zc.astype(np.complex64)

def sync_zc_valid(rx, zc, ratio_th=6.0, verbose=False):
    rx0 = rx - np.mean(rx)
    zc0 = zc / (np.sqrt(np.mean(np.abs(zc)**2)) + 1e-12)

    # valid correlation: idx = zc start
    corr = np.abs(np.correlate(rx0, zc0, mode="valid"))
    peak = float(np.max(corr))
    med  = float(np.median(corr) + 1e-12)
    idx  = int(np.argmax(corr))
    ratio = peak / med
    if verbose:
        print(f"    [SYNC] peak={peak:.2e} med={med:.2e} ratio={ratio:.2f} idx={idx}")
    return (ratio >= ratio_th), idx, ratio

def refine_start(rx, zc, idx, search=16):
    L = len(zc)
    best = idx
    bestv = -1.0
    for d in range(-search, search+1):
        s = idx + d
        if s < 0 or s+L > len(rx):
            continue
        # maximize matched filter output magnitude
        v = np.abs(np.vdot(zc, rx[s:s+L]))  # sum conj(zc)*rx
        if v > bestv:
            bestv = v
            best = s
    return best

def estimate_cfo_from_repetition(rx, idx, L, fs, verbose=False):
    """
    CFO from repeated blocks:
      cfo = angle( sum( r2 * conj(r1) ) ) * fs / (2*pi*L)
    where r1=rx[idx:idx+L], r2=rx[idx+L:idx+2L]
    """
    if idx + 2*L > len(rx):
        return 0.0
    r1 = rx[idx:idx+L]
    r2 = rx[idx+L:idx+2*L]
    metric = np.sum(r2 * np.conj(r1))
    ang = np.angle(metric)
    cfo = ang * fs / (2*np.pi*L)
    if verbose:
        print(f"    [CFO] angle={ang:.3e} rad -> cfo={cfo:.1f} Hz")
    return float(cfo)

def apply_cfo_correction(x, cfo_hz, fs):
    n = np.arange(len(x), dtype=np.float64)
    rot = np.exp(-1j * 2*np.pi * cfo_hz * n / fs)
    return x * rot

# -----------------------------
# OFDM mapping (802.11-like)
# N=64: used subcarriers are [-26..-1, +1..+26]
# Pilots at [-21,-7,+7,+21], Data are remaining 48.
# Use fftshift domain: DC at index N/2.
# -----------------------------
def sc_to_bin(sc, N):
    return (sc + N//2) % N  # in fftshifted spectrum

def ofdm_params(N=64):
    pilots_sc = np.array([-21, -7, 7, 21], dtype=np.int32)
    used_sc = np.concatenate([np.arange(-26, 0), np.arange(1, 27)]).astype(np.int32)
    data_sc = np.array([sc for sc in used_sc if sc not in set(pilots_sc)], dtype=np.int32)
    assert len(data_sc) == 48
    return used_sc, data_sc, pilots_sc

def ofdm_modulate(bits, N=64, CP=16, n_sym=14):
    used_sc, data_sc, pilots_sc = ofdm_params(N)
    bits = bits.reshape(-1)
    n_data = len(data_sc)
    bits_per_sym = n_data * 2
    assert len(bits) == bits_per_sym * n_sym

    pilots = np.array([1, 1, 1, -1], dtype=np.float32) + 0j  # fixed BPSK pilots

    out = []
    ptr = 0
    for _ in range(n_sym):
        b = bits[ptr:ptr+bits_per_sym]; ptr += bits_per_sym
        Xd = qpsk_mod(b)  # 48 symbols

        X_shift = np.zeros(N, dtype=np.complex64)  # fftshifted bins
        # pilots
        for k, sc in enumerate(pilots_sc):
            X_shift[sc_to_bin(sc, N)] = pilots[k]
        # data
        for k, sc in enumerate(data_sc):
            X_shift[sc_to_bin(sc, N)] = Xd[k]

        # to ifft input: ifftshift
        X = np.fft.ifftshift(X_shift)
        x = np.fft.ifft(X) * np.sqrt(N)
        x_cp = np.concatenate([x[-CP:], x])
        out.append(x_cp.astype(np.complex64))

    return np.concatenate(out)

def ofdm_demodulate(rx, N=64, CP=16, n_sym=14):
    used_sc, data_sc, pilots_sc = ofdm_params(N)
    n_data = len(data_sc)
    pilots = np.array([1, 1, 1, -1], dtype=np.float32) + 0j

    sym_len = N + CP
    need = n_sym * sym_len
    if len(rx) < need:
        return None, {"error": "short"}

    rx = rx[:need]
    rx_syms = rx.reshape(n_sym, sym_len)
    bits_out = []
    snr_list = []
    evm_list = []

    # interpolation axis: subcarrier index (in fftshifted domain)
    sc_all = np.concatenate([pilots_sc, data_sc])
    sc_all_sorted = np.sort(sc_all)
    bin_all_sorted = np.array([sc_to_bin(sc, N) for sc in sc_all_sorted], dtype=np.int32)

    for i in range(n_sym):
        y = rx_syms[i, CP:CP+N]
        Y = np.fft.fft(y) / np.sqrt(N)
        Y_shift = np.fft.fftshift(Y)

        # channel on pilots
        Yp = np.array([Y_shift[sc_to_bin(sc, N)] for sc in pilots_sc], dtype=np.complex64)
        Hp = Yp / (pilots + 1e-12)

        # interpolate complex H across used carriers (real/imag)
        sc_p = pilots_sc.astype(np.float64)
        Hr = np.interp(sc_all_sorted, np.sort(pilots_sc), np.real(Hp[np.argsort(pilots_sc)]))
        Hi = np.interp(sc_all_sorted, np.sort(pilots_sc), np.imag(Hp[np.argsort(pilots_sc)]))
        H_all = (Hr + 1j*Hi).astype(np.complex64)

        # build a dict from sorted sc to H
        H_map = {sc: H_all[k] for k, sc in enumerate(sc_all_sorted)}

        # equalize data
        Yd = np.array([Y_shift[sc_to_bin(sc, N)] for sc in data_sc], dtype=np.complex64)
        Hd = np.array([H_map[sc] for sc in data_sc], dtype=np.complex64)
        Xd_hat = Yd / (Hd + 1e-12)

        # demod
        b_hat = qpsk_demod(Xd_hat)
        bits_out.append(b_hat)

        # SNR/EVM quick metrics (hard decision ref)
        Xh = qpsk_mod(b_hat)
        err = Xd_hat - Xh
        sigp = np.mean(np.abs(Xh)**2)
        errp = np.mean(np.abs(err)**2) + 1e-12
        snr = 10*np.log10(sigp/errp)
        evm = 100*np.sqrt(errp/(sigp+1e-12))
        snr_list.append(float(snr))
        evm_list.append(float(evm))

    bits_out = np.concatenate(bits_out).astype(np.int32)
    return bits_out, {"snr_db": float(np.mean(snr_list)), "evm_pct": float(np.mean(evm_list))}

# -----------------------------
# Pluto Link
# -----------------------------
class PlutoLink:
    def __init__(self, uri, fc, fs, tx_gain, rx_gain, tx_amp, rx_buf, verbose=False):
        self.uri = uri
        self.fc = float(fc)
        self.fs = float(fs)
        self.tx_gain = float(tx_gain)
        self.rx_gain = float(rx_gain)
        self.tx_amp = float(tx_amp)
        self.rx_buf = int(rx_buf)
        self.verbose = verbose

        self.sdr = None

        self.N = 64
        self.CP = 16
        self.n_sym = 14

        self.zc_len = 1024
        self.zc = make_zc(self.zc_len, root=29)
        self.preamble = np.concatenate([self.zc, self.zc]).astype(np.complex64)  # repeated
        self.sync_ratio_th = 6.0

    def connect(self):
        if self.verbose:
            print(f"[SDR] Connecting {self.uri} ...")
        self.sdr = adi.Pluto(uri=self.uri)
        self.sdr.sample_rate = int(self.fs)
        self.sdr.tx_lo = int(self.fc)
        self.sdr.rx_lo = int(self.fc)
        self.sdr.tx_rf_bandwidth = int(self.fs)
        self.sdr.rx_rf_bandwidth = int(self.fs)
        self.sdr.tx_hardwaregain_chan0 = self.tx_gain
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = self.rx_gain
        self.sdr.rx_buffer_size = self.rx_buf
        self.sdr.tx_enabled_channels = [0]
        self.sdr.rx_enabled_channels = [0]

        # stability
        try:
            if hasattr(self.sdr, "_rxadc") and self.sdr._rxadc is not None:
                self.sdr._rxadc.set_kernel_buffers_count(4)
        except:
            pass

        if self.verbose:
            print(f"[SDR] OK fc={self.fc/1e6:.1f}MHz fs={self.fs/1e6:.1f}Msps TXgain={self.tx_gain} RXgain={self.rx_gain} tx_amp={self.tx_amp}")

    def close(self):
        if self.sdr is None:
            return
        try:
            self.sdr.tx_destroy_buffer()
        except:
            pass
        try:
            self.sdr.rx_destroy_buffer()
        except:
            pass

    def tx_frame_once(self, tx_samples_c64):
        # Always destroy before (avoid tx_cyclic_buffer error)
        try:
            self.sdr.tx_destroy_buffer()
        except:
            pass
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(tx_samples_c64)

    def rx_capture(self):
        # flush
        for _ in range(3):
            _ = self.sdr.rx()
        return self.sdr.rx()

    def build_frame(self, bits):
        ofdm = ofdm_modulate(bits, N=self.N, CP=self.CP, n_sym=self.n_sym)
        # No gap between preamble and payload; simpler slicing
        frame = np.concatenate([self.preamble, ofdm]).astype(np.complex64)

        # scale to tx_amp
        frame = frame / (np.max(np.abs(frame)) + 1e-12) * self.tx_amp
        tx = (frame * (2**14)).astype(np.complex64)
        return frame, tx

    def decode_from_capture(self, rx, ratio_th, verbose=False):
        rx_peak = float(np.max(np.abs(rx)))
        ok, idx, ratio = sync_zc_valid(rx, self.zc, ratio_th=ratio_th, verbose=verbose)
        if not ok:
            return None, {"sync_ok": False, "ratio": ratio, "rx_peak": rx_peak}

        idx = refine_start(rx, self.zc, idx, search=16)
        cfo = estimate_cfo_from_repetition(rx, idx, self.zc_len, self.fs, verbose=verbose)

        # CFO correct whole stream from idx onward (safer)
        rx2 = rx.copy()
        rx2[idx:] = apply_cfo_correction(rx2[idx:], cfo, self.fs)

        payload_start = idx + 2*self.zc_len
        payload_len = self.n_sym * (self.N + self.CP)
        if payload_start + payload_len > len(rx2):
            return None, {"sync_ok": True, "ratio": ratio, "cfo_hz": cfo, "rx_peak": rx_peak, "error": "short_payload"}

        payload = rx2[payload_start:payload_start+payload_len]

        # normalize payload power (optional)
        p = np.sqrt(np.mean(np.abs(payload)**2)) + 1e-12
        payload = payload / p

        bits_hat, m = ofdm_demodulate(payload, N=self.N, CP=self.CP, n_sym=self.n_sym)
        if bits_hat is None:
            return None, {"sync_ok": True, "ratio": ratio, "cfo_hz": cfo, "rx_peak": rx_peak, "error": "demod_fail"}

        m.update({"sync_ok": True, "ratio": ratio, "cfo_hz": cfo, "rx_peak": rx_peak})
        return bits_hat, m

# -----------------------------
def ber(tx_bits, rx_bits):
    n = min(len(tx_bits), len(rx_bits))
    if n <= 0:
        return 1.0, 0, 0
    e = int(np.sum(tx_bits[:n] != rx_bits[:n]))
    return e/n, e, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="ip:192.168.2.2")
    ap.add_argument("--freq", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--preset", default="antenna_close", choices=list(PRESETS.keys()))
    ap.add_argument("--tx_gain", type=float, default=None)
    ap.add_argument("--rx_gain", type=float, default=None)
    ap.add_argument("--tx_amp", type=float, default=None)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--rx_buf", type=int, default=65536)
    ap.add_argument("--sync_ratio", type=float, default=6.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ps = PRESETS[args.preset].copy()
    if args.tx_gain is not None: ps["tx_gain"] = args.tx_gain
    if args.rx_gain is not None: ps["rx_gain"] = args.rx_gain
    if args.tx_amp is not None:  ps["tx_amp"]  = args.tx_amp

    print("\n" + "="*78)
    print("PLUTO OFDM BER TEST (v4 - repeated ZC preamble + proper CFO)")
    print("="*78)
    print(f"ip={args.ip}  fc={args.freq/1e6:.1f}MHz  fs={args.fs/1e6:.1f}Msps  preset={args.preset}")
    print(f"TXgain={ps['tx_gain']}  RXgain={ps['rx_gain']}  tx_amp={ps['tx_amp']}  rx_buf={args.rx_buf}")
    print("OFDM: N=64 CP=16 data=48 pilots=4 syms=14 bits/frame=1344")
    print(f"Preamble: ZC len=1024 repeated x2  sync_ratio_th={args.sync_ratio}")
    print("="*78)

    link = PlutoLink(args.ip, args.freq, args.fs, ps["tx_gain"], ps["rx_gain"], ps["tx_amp"], args.rx_buf, verbose=args.verbose)
    link.connect()

    total_e = 0
    total_n = 0
    ok_frames = 0
    snrs = []
    evms = []

    print("\nFrame  BER        Err    Bits   SyncRatio   CFO(Hz)     SNR(dB)  EVM(%)  RXpeak  Status")
    print("-"*100)

    try:
        for fi in range(args.frames):
            np.random.seed(1000 + fi)
            tx_bits = np.random.randint(0, 2, 48*14*2).astype(np.int32)

            _, tx = link.build_frame(tx_bits)
            link.tx_frame_once(tx)

            # small settle
            time.sleep(0.15)

            rx = link.rx_capture()

            bits_hat, m = link.decode_from_capture(rx, ratio_th=args.sync_ratio, verbose=args.verbose)

            # stop tx after capture
            try:
                link.sdr.tx_destroy_buffer()
            except:
                pass

            if bits_hat is None:
                ratio = m.get("ratio", 0.0)
                rxp = m.get("rx_peak", 0.0)
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:8.2f}   N/A        N/A      N/A   {rxp:7.0f}  SYNC/DECODE FAIL")
                continue

            b, e, n = ber(tx_bits, bits_hat)
            total_e += e
            total_n += n
            ok_frames += 1

            cfo = m.get("cfo_hz", 0.0)
            snr = m.get("snr_db", float("nan"))
            evm = m.get("evm_pct", float("nan"))
            rxp = m.get("rx_peak", 0.0)
            ratio = m.get("ratio", 0.0)

            if np.isfinite(snr): snrs.append(snr)
            if np.isfinite(evm): evms.append(evm)

            status = "OK" if b < 1e-2 else "ERRORS"
            print(f"{fi:<5}  {b:8.2e} {e:6d} {n:7d}  {ratio:8.2f}  {cfo:9.1f}  {snr:8.1f}  {evm:7.1f}  {rxp:7.0f}  {status}")

            time.sleep(0.05)

        print("-"*100)
        print("\nSUMMARY")
        print(f"frames={args.frames}, decoded={ok_frames} ({100*ok_frames/max(1,args.frames):.1f}%)")
        if total_n > 0:
            print(f"bits={total_n}, errors={total_e}, BER={total_e/total_n:.3e}")
            if snrs:
                print(f"SNR mean={np.mean(snrs):.1f} dB, min={np.min(snrs):.1f}, max={np.max(snrs):.1f}")
            if evms:
                print(f"EVM mean={np.mean(evms):.1f} %,  min={np.min(evms):.1f}, max={np.max(evms):.1f}")
        else:
            print("No successful decodes.")
        print("="*78)

    finally:
        link.close()

if __name__ == "__main__":
    main()