#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pluto_ofdm_ber_test_v2.py
- Robust OTA OFDM BER test for PlutoSDR (1R1T)
Key upgrades vs previous:
  * Long Zadoff-Chu preamble (1024) + FFT-based correlation sync
  * CFO estimation from phase slope of rx_preamble * conj(tx_preamble)
  * Larger RX buffer to ensure frame capture
  * Cleaner OFDM subcarrier mapping (signed indices)
"""

import argparse
import time
import numpy as np
import sys

try:
    import adi
    ADI_AVAILABLE = True
except ImportError:
    ADI_AVAILABLE = False


# -----------------------------
# Presets (safe-ish defaults)
# -----------------------------
PRESETS = {
    "cable_30db":    {"tx_gain": -10, "rx_gain": 40, "tx_amp": 0.5},
    "cable_direct":  {"tx_gain": -30, "rx_gain": 20, "tx_amp": 0.1},
    "antenna_close": {"tx_gain": 0,   "rx_gain": 60, "tx_amp": 0.6},  # NOTE: start less than 0.9
    "antenna_far":   {"tx_gain": 0,   "rx_gain": 70, "tx_amp": 0.9},
}


# -----------------------------
# Utilities
# -----------------------------
def db(x, eps=1e-12):
    return 20 * np.log10(np.abs(x) + eps)

def pwr_db(x, eps=1e-12):
    return 10 * np.log10(np.mean(np.abs(x)**2) + eps)

def rms(x, eps=1e-12):
    return np.sqrt(np.mean(np.abs(x)**2) + eps)


# -----------------------------
# Zadoff-Chu preamble
# -----------------------------
def zc_sequence(N=1024, u=29):
    """
    ZC sequence length N (N prime preferred; 1024 isn't prime but still works as a good CAZAC-like preamble in practice).
    We'll use a constant-modulus polyphase seq:
      x[n] = exp(-j*pi*u*n*(n+1)/N)
    """
    n = np.arange(N)
    x = np.exp(-1j * np.pi * u * n * (n + 1) / N).astype(np.complex64)
    return x

def fft_correlate_valid(x, h):
    """
    Compute valid correlation of x with h (h known, length M):
      corr[k] = sum_{m=0..M-1} x[k+m] * conj(h[m])
    Returns corr length = len(x)-len(h)+1
    """
    x = x.astype(np.complex64)
    h = h.astype(np.complex64)
    M = len(h)
    L = len(x)
    if L < M:
        return np.array([], dtype=np.complex64)

    # FFT-based: corr = ifft( FFT(x) * conj(FFT(h_rev)) )
    # where h_rev[n] = conj(h[M-1-n])? careful with correlation conventions.
    # We'll implement correlation using convolution:
    # corr[k] = sum x[k+m]*conj(h[m]) = (x * conj(h[::-1])) at index k+M-1
    g = np.conj(h[::-1])
    nfft = 1
    while nfft < (L + M - 1):
        nfft *= 2
    X = np.fft.fft(x, nfft)
    G = np.fft.fft(g, nfft)
    y = np.fft.ifft(X * G)
    y = y[:L + M - 1]
    corr_full = y  # convolution result

    # valid part starts at index M-1, length L-M+1
    corr_valid = corr_full[M-1:M-1 + (L - M + 1)]
    return corr_valid

def coarse_sync_zc(rx, preamble, ratio_th=8.0, verbose=False):
    """
    Find preamble start by correlation peak.
    Returns dict: ok, start, peak, med, ratio
    """
    # remove DC
    rx0 = rx - np.mean(rx)
    corr = fft_correlate_valid(rx0, preamble)
    if len(corr) == 0:
        return {"ok": False, "reason": "rx shorter than preamble"}

    mag = np.abs(corr)
    peak_i = int(np.argmax(mag))
    peak_v = float(mag[peak_i])
    med_v = float(np.median(mag))
    ratio = peak_v / (med_v + 1e-12)

    if verbose:
        print(f"    [SYNC-ZC] peak={peak_v:.2e} med={med_v:.2e} ratio={ratio:.2f} idx={peak_i}")

    ok = ratio >= ratio_th
    return {"ok": ok, "start": peak_i, "peak": peak_v, "med": med_v, "ratio": ratio}

def estimate_cfo_from_preamble(rx_pre, tx_pre, fs, verbose=False):
    """
    Robust CFO estimate:
      z[n] = rx[n] * conj(tx[n]) ≈ A * exp(j*2π*cfo*n/fs)
      mean phase increment = angle(sum z[n+1]*conj(z[n]))
    """
    z = rx_pre * np.conj(tx_pre)
    d = np.sum(z[1:] * np.conj(z[:-1]))  # complex average of phase increment
    w = np.angle(d)                      # in [-pi, pi]
    cfo = w * fs / (2*np.pi)

    # wrap to [-fs/2, fs/2] for safety
    cfo = ((cfo + fs/2) % fs) - fs/2

    if verbose:
        print(f"    [CFO] angle={w:.3e} rad -> cfo={cfo:.1f} Hz")
    return float(cfo)


# -----------------------------
# QPSK / 16QAM (Gray-ish)
# -----------------------------
class QAM:
    def __init__(self, M=4):
        self.M = M
        self.bps = int(np.log2(M))
        if M == 4:
            # indices from bits LSB-first: 00,01,10,11
            self.const = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / np.sqrt(2)
        elif M == 16:
            levels = np.array([-3,-1,1,3])
            grid = np.array([x+1j*y for y in levels for x in levels], dtype=np.complex64)
            self.const = grid / np.sqrt(10)
        else:
            raise ValueError("M must be 4 or 16")

    def mod(self, bits):
        bits = np.asarray(bits).astype(np.int8)
        pad = (-len(bits)) % self.bps
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, np.int8)])
        b = bits.reshape(-1, self.bps)
        idx = np.zeros(len(b), dtype=np.int32)
        for i in range(self.bps):
            idx |= (b[:, i].astype(np.int32) << i)
        return self.const[idx]

    def demod_hard(self, syms):
        syms = np.asarray(syms).astype(np.complex64)
        dist = np.abs(syms[:, None] - self.const[None, :])
        idx = np.argmin(dist, axis=1)
        out = np.zeros((len(idx), self.bps), dtype=np.int8)
        for i in range(self.bps):
            out[:, i] = (idx >> i) & 1
        return out.reshape(-1)


# -----------------------------
# OFDM mapping (N=64 style)
# -----------------------------
class OFDM:
    def __init__(self, N=64, CP=16, M=4, n_sym=14):
        self.N = N
        self.CP = CP
        self.n_sym = n_sym
        self.qam = QAM(M)

        # Use IEEE802.11-like occupied bins (excluding DC):
        # data: 48, pilots: 4 at k = [-21, -7, +7, +21]
        self.pilot_ks = np.array([-21, -7, 7, 21], dtype=np.int32)
        used_ks = np.concatenate([np.arange(-26,0), np.arange(1,27)])
        data_ks = np.array([k for k in used_ks if k not in set(self.pilot_ks)], dtype=np.int32)
        self.data_ks = data_ks[:48]  # 48 data carriers

        # map signed k to FFT bin index
        self.k2bin = lambda k: (k % self.N)
        self.k2i = lambda k: int(k + self.N//2)  # signed k in [-N/2, N/2-1] -> [0..N-1]

        # pilots (BPSK)
        np.random.seed(42)
        self.pilots = (np.sign(np.random.randn(len(self.pilot_ks))) + 0j).astype(np.complex64)

    @property
    def bits_per_frame(self):
        return len(self.data_ks) * self.n_sym * self.qam.bps

    @property
    def samples_per_frame(self):
        return self.n_sym * (self.N + self.CP)

    # def modulate(self, bits):
    #     bits = np.asarray(bits).astype(np.int8)
    #     need = self.bits_per_frame
    #     if len(bits) < need:
    #         bits = np.concatenate([bits, np.zeros(need - len(bits), np.int8)])
    #     bits = bits[:need]

    #     data_syms = self.qam.mod(bits).reshape(self.n_sym, len(self.data_ks))

    #     out = []
    #     for i in range(self.n_sym):
    #         X = np.zeros(self.N, dtype=np.complex64)
    #         # pilots
    #         for pk, pv in zip(self.pilot_ks, self.pilots):
    #             X[self.k2bin(pk)] = pv
    #         # data
    #         for dk, dv in zip(self.data_ks, data_syms[i]):
    #             X[self.k2bin(dk)] = dv
    #         x = np.fft.ifft(X) * np.sqrt(self.N)
    #         cp = x[-self.CP:]
    #         out.append(np.concatenate([cp, x]))
    #     return np.concatenate(out).astype(np.complex64)

    def modulate(self, bits):
        bits = np.asarray(bits).astype(np.int8)
        need = self.bits_per_frame
        if len(bits) < need:
            bits = np.concatenate([bits, np.zeros(need - len(bits), np.int8)])
        bits = bits[:need]

        data_syms = self.qam.mod(bits).reshape(self.n_sym, len(self.data_ks))

        out = []
        for i in range(self.n_sym):
            Xs = np.zeros(self.N, dtype=np.complex64)  # SHIFTED spectrum: bins correspond to signed k
            for pk, pv in zip(self.pilot_ks, self.pilots):
                Xs[self.k2i(pk)] = pv
            for dk, dv in zip(self.data_ks, data_syms[i]):
                Xs[self.k2i(dk)] = dv

            x = np.fft.ifft(np.fft.ifftshift(Xs)) * np.sqrt(self.N)
            cp = x[-self.CP:]
            out.append(np.concatenate([cp, x]))
        return np.concatenate(out).astype(np.complex64)

    def demodulate(self, rx_td):
        rx_td = np.asarray(rx_td).astype(np.complex64)
        Lsym = self.N + self.CP
        ns = min(len(rx_td) // Lsym, self.n_sym)
        if ns <= 0:
            return np.array([], np.int8), {"error": "no symbols"}

        pilot_i = np.array([self.k2i(k) for k in self.pilot_ks], dtype=np.int32)
        data_i  = np.array([self.k2i(k) for k in self.data_ks], dtype=np.int32)

        all_data = []
        snr_list, evm_list = [], []

        # signed axis for interpolation: [-N/2 .. N/2-1]
        axis = np.arange(-self.N//2, self.N//2, dtype=np.float32)

        for i in range(ns):
            s = rx_td[i*Lsym + self.CP : i*Lsym + self.CP + self.N]
            Ys = np.fft.fftshift(np.fft.fft(s) / np.sqrt(self.N))  # SHIFTED

            Yp = Ys[pilot_i]
            Hp = Yp / (self.pilots + 1e-12)

            # interpolate complex H over signed axis
            pilot_k = np.array(self.pilot_ks, dtype=np.float32)
            order = np.argsort(pilot_k)
            pilot_k = pilot_k[order]
            Hp = Hp[order]

            Hr = np.interp(axis, pilot_k, np.real(Hp))
            Hi = np.interp(axis, pilot_k, np.imag(Hp))
            Hs = (Hr + 1j*Hi).astype(np.complex64)

            Hd = Hs[data_i]
            Xeq = Ys[data_i] / (Hd + 1e-12)
            all_data.append(Xeq)

            hard_bits = self.qam.demod_hard(Xeq.reshape(-1))
            hard_syms = self.qam.mod(hard_bits).reshape(-1)
            err = Xeq.reshape(-1) - hard_syms

            evm = 100 * np.sqrt(np.mean(np.abs(err)**2) / (np.mean(np.abs(hard_syms)**2) + 1e-12))
            evm_list.append(float(evm))

            sigp = np.mean(np.abs(hard_syms)**2)
            noisep = np.mean(np.abs(err)**2)
            snr = 10*np.log10(sigp/(noisep+1e-12))
            snr_list.append(float(snr))

        all_data = np.concatenate(all_data).astype(np.complex64)
        bits = self.qam.demod_hard(all_data.reshape(-1))

        return bits, {
            "nsym": int(ns),
            "snr_db": float(np.mean(snr_list)) if snr_list else None,
            "evm_pct": float(np.mean(evm_list)) if evm_list else None,
        }


# -----------------------------
# Pluto link
# -----------------------------
class PlutoLink:
    def __init__(self, ip, fc, fs, tx_gain, rx_gain, tx_amp, rx_buf, verbose=False):
        self.ip = ip
        self.fc = fc
        self.fs = fs
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.tx_amp = tx_amp
        self.rx_buf = rx_buf
        self.verbose = verbose
        self.sdr = None

    def connect(self):
        if not ADI_AVAILABLE:
            print("[Error] pyadi-iio not available")
            return False
        print(f"[SDR] Connecting {self.ip} ...")
        self.sdr = adi.Pluto(uri=self.ip)
        self.sdr.sample_rate = int(self.fs)
        self.sdr.tx_lo = int(self.fc)
        self.sdr.rx_lo = int(self.fc)
        self.sdr.tx_rf_bandwidth = int(self.fs)
        self.sdr.rx_rf_bandwidth = int(self.fs)

        self.sdr.tx_hardwaregain_chan0 = float(self.tx_gain)
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = float(self.rx_gain)

        self.sdr.tx_enabled_channels = [0]
        self.sdr.rx_enabled_channels = [0]
        self.sdr.rx_buffer_size = int(self.rx_buf)

        # stability
        if hasattr(self.sdr, "_rxadc") and self.sdr._rxadc is not None:
            try:
                self.sdr._rxadc.set_kernel_buffers_count(4)
            except:
                pass

        print(f"[SDR] OK  fc={self.fc/1e6:.1f}MHz fs={self.fs/1e6:.1f}Msps TXgain={self.tx_gain} RXgain={self.rx_gain} tx_amp={self.tx_amp}")
        return True

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

    def tx_cyclic(self, x):
        x = x.astype(np.complex64)
        x = x / (np.max(np.abs(x)) + 1e-12) * float(self.tx_amp)
        tx = (x * (2**14)).astype(np.complex64)

        # destroy old buffer before re-creating a cyclic buffer
        try:
            self.sdr.tx_destroy_buffer()
        except:
            pass

        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(tx)

    def rx_once(self):
        # flush a bit
        for _ in range(2):
            _ = self.sdr.rx()
        return self.sdr.rx()


# -----------------------------
# BER
# -----------------------------
def ber(tx_bits, rx_bits):
    n = min(len(tx_bits), len(rx_bits))
    if n <= 0:
        return None, None, None
    e = int(np.sum(tx_bits[:n] != rx_bits[:n]))
    return e / n, e, n


# -----------------------------
# Main test
# -----------------------------
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
    ap.add_argument("--mod", type=int, default=4, choices=[4,16])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sync_ratio", type=float, default=8.0, help="corr peak/median threshold")
    args = ap.parse_args()

    s = PRESETS[args.preset].copy()
    if args.tx_gain is not None: s["tx_gain"] = args.tx_gain
    if args.rx_gain is not None: s["rx_gain"] = args.rx_gain
    if args.tx_amp  is not None: s["tx_amp"]  = args.tx_amp

    ofdm = OFDM(N=64, CP=16, M=args.mod, n_sym=14)
    pre = zc_sequence(1024, u=29).astype(np.complex64)

    # build one frame (cyclic)
    guard1 = np.zeros(512, np.complex64)
    guard2 = np.zeros(128, np.complex64)
    guard3 = np.zeros(512, np.complex64)

    print("\n" + "="*78)
    print("PLUTO OFDM BER TEST (v2 - ZC sync)")
    print("="*78)
    print(f"ip={args.ip}  fc={args.freq/1e6:.1f}MHz  fs={args.fs/1e6:.1f}Msps  preset={args.preset}")
    print(f"mod={args.mod}  TXgain={s['tx_gain']}  RXgain={s['rx_gain']}  tx_amp={s['tx_amp']}  rx_buf={args.rx_buf}")
    print(f"OFDM: N=64 CP=16 data=48 pilots=4 syms=14  bits/frame={ofdm.bits_per_frame}")
    print(f"Preamble: ZC len={len(pre)}  sync_ratio_th={args.sync_ratio}")
    print("="*78)

    link = PlutoLink(args.ip, args.freq, args.fs, s["tx_gain"], s["rx_gain"], s["tx_amp"], args.rx_buf, verbose=args.verbose)
    if not link.connect():
        sys.exit(1)

    total_e = 0
    total_n = 0
    ok_frames = 0

    print("\nFrame  BER        Err    Bits   SyncRatio  CFO(Hz)    SNR(dB)  EVM(%)  RXpeak  Status")
    print("-"*100)

    try:
        for fi in range(args.frames):
            np.random.seed(1000 + fi)
            tx_bits = np.random.randint(0,2, ofdm.bits_per_frame).astype(np.int8)
            payload = ofdm.modulate(tx_bits)

            frame = np.concatenate([guard1, pre, guard2, payload, guard3]).astype(np.complex64)
            link.tx_cyclic(frame)

            # allow settle a bit
            time.sleep(0.10)

            rx = link.rx_once().astype(np.complex64)
            rxpk = float(np.max(np.abs(rx)))

            # sync
            sync = coarse_sync_zc(rx, pre, ratio_th=args.sync_ratio, verbose=args.verbose)
            if not sync["ok"]:
                print(f"{fi:<5}  N/A       N/A    N/A    {sync.get('ratio',0):>8.2f}    N/A       N/A      N/A    {rxpk:>6.0f}  SYNC FAIL")
                continue

            st = sync["start"]
            ratio = sync["ratio"]

            # extract preamble region
            if st + len(pre) > len(rx):
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:>8.2f}    N/A       N/A      N/A    {rxpk:>6.0f}  CUT PRE")
                continue

            rx_pre = rx[st:st+len(pre)]
            # CFO
            cfo = estimate_cfo_from_preamble(rx_pre, pre, args.fs, verbose=args.verbose)

            # payload start
            pay_start = st + len(pre) + len(guard2)
            pay_len = ofdm.samples_per_frame
            if pay_start + pay_len > len(rx):
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:>8.2f}  {cfo:>8.1f}    N/A      N/A    {rxpk:>6.0f}  CUT PAY")
                continue

            pay = rx[pay_start:pay_start+pay_len]

            # CFO correction
            n = np.arange(len(pay), dtype=np.float32)
            pay = pay * np.exp(-1j * 2*np.pi * (cfo/args.fs) * n).astype(np.complex64)

            # demod
            rx_bits, met = ofdm.demodulate(pay)
            if rx_bits is None or len(rx_bits) == 0 or "error" in met:
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:>8.2f}  {cfo:>8.1f}    N/A      N/A    {rxpk:>6.0f}  DECODE FAIL")
                continue

            b, e, ncmp = ber(tx_bits, rx_bits)
            snr_db = met.get("snr_db", None)
            evm = met.get("evm_pct", None)

            ok_frames += 1
            total_e += e
            total_n += ncmp

            status = "OK" if b is not None and b < 1e-2 else "ERRORS"
            print(f"{fi:<5}  {b:>8.2e}  {e:>5}  {ncmp:>6}   {ratio:>8.2f}  {cfo:>8.1f}  {snr_db:>7.1f}  {evm:>6.1f}  {rxpk:>6.0f}  {status}")

            time.sleep(0.05)

    finally:
        try:
            link.sdr.tx_destroy_buffer()
        except:
            pass
        link.close()

    print("-"*100)
    print("\nSUMMARY")
    print(f"frames={args.frames}, decoded={ok_frames} ({(100*ok_frames/max(1,args.frames)):.1f}%)")
    if total_n > 0:
        overall = total_e / total_n
        print(f"bits={total_n}, errors={total_e}, BER={overall:.2e}")
    else:
        print("No successful decodes.")


if __name__ == "__main__":
    main()