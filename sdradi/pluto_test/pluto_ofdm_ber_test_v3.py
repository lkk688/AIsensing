#!/usr/bin/env python3
import argparse, time, sys
import numpy as np

try:
    import adi
except Exception as e:
    print("pyadi-iio not available:", e)
    sys.exit(1)

# --------------------------
# QPSK / 16QAM (hard)
# --------------------------
class QAM:
    def __init__(self, M=4):
        self.M = M
        self.bps = int(np.log2(M))
        self.const = self._constellation()

    def _constellation(self):
        if self.M == 4:  # QPSK Gray-ish (00,01,10,11) with LSB-first mapping below
            c = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64) / np.sqrt(2)
            return c
        if self.M == 16:
            lv = np.array([-3,-1,1,3], dtype=np.float32)
            g = np.array([x + 1j*y for y in lv for x in lv], dtype=np.complex64) / np.sqrt(10)
            return g
        raise ValueError("Unsupported M")

    def mod(self, bits):
        bits = np.asarray(bits).astype(np.int8)
        pad = (-len(bits)) % self.bps
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, np.int8)])
        b = bits.reshape(-1, self.bps)
        idx = np.zeros(len(b), dtype=np.int32)
        # LSB-first packing (must match demod)
        for i in range(self.bps):
            idx |= (b[:, i].astype(np.int32) << i)
        return self.const[idx]

    def demod_hard(self, syms):
        syms = np.asarray(syms).astype(np.complex64).reshape(-1)
        d = np.abs(syms[:, None] - self.const[None, :])
        idx = np.argmin(d, axis=1).astype(np.int32)
        b = np.zeros((len(idx), self.bps), dtype=np.int8)
        for i in range(self.bps):
            b[:, i] = (idx >> i) & 1
        return b.reshape(-1)

# --------------------------
# ZC preamble
# --------------------------
def zc_seq(N=1024, root=25):
    n = np.arange(N)
    # ZC constant amplitude
    x = np.exp(-1j * np.pi * root * n * (n + 1) / N).astype(np.complex64)
    return x

# def sync_zc(rx, zc, ratio_th=6.0, verbose=False):
#     # matched filter via correlation (valid)
#     zc_n = zc / (np.sqrt(np.mean(np.abs(zc)**2)) + 1e-12)
#     rx_n = rx - np.mean(rx)
#     # use conj-reversed for correlation (matched filter)
#     mf = np.abs(np.convolve(rx_n, np.conj(zc_n[::-1]), mode="valid"))
#     peak = float(np.max(mf))
#     med = float(np.median(mf) + 1e-12)
#     idx = int(np.argmax(mf))  # start index in rx of zc (because we used reversed)
#     ratio = peak / med
#     if verbose:
#         print(f"    [SYNC-ZC] peak={peak:.2e} med={med:.2e} ratio={ratio:.2f} idx={idx}")
#     ok = ratio >= ratio_th
#     return ok, idx, ratio

def sync_zc(rx, zc, ratio_th=6.0, verbose=False):
    rx_n = rx - np.mean(rx)
    zc_n = zc / (np.sqrt(np.mean(np.abs(zc)**2)) + 1e-12)

    # np.correlate 对第二个参数会取共轭：sum rx[m+i] * conj(zc[i])
    corr = np.abs(np.correlate(rx_n, zc_n, mode="valid"))

    peak = float(np.max(corr))
    med  = float(np.median(corr) + 1e-12)
    idx  = int(np.argmax(corr))          # 这里 idx 就是 zc 在 rx 中的起始位置
    ratio = peak / med

    if verbose:
        print(f"    [SYNC-ZC] peak={peak:.2e} med={med:.2e} ratio={ratio:.2f} idx={idx}")

    return (ratio >= ratio_th), idx, ratio

def estimate_cfo_known_preamble(rx_pre, tx_pre, fs, verbose=False):
    # remove known preamble phase
    z = rx_pre * np.conj(tx_pre)
    d = np.sum(z[1:] * np.conj(z[:-1]))
    w = np.angle(d)
    cfo = w * fs / (2*np.pi)
    cfo = ((cfo + fs/2) % fs) - fs/2
    if verbose:
        print(f"    [CFO] angle={w:.3e} rad -> cfo={cfo:.1f} Hz")
    return float(cfo)

def apply_cfo(x, fs, cfo_hz):
    n = np.arange(len(x), dtype=np.float32)
    return x * np.exp(-1j * 2*np.pi * cfo_hz * n / fs)

# --------------------------
# OFDM (fftshift / signed axis)
# --------------------------
class OFDM:
    def __init__(self, N=64, CP=16, n_sym=14, M=4):
        self.N = N
        self.CP = CP
        self.n_sym = n_sym
        self.qam = QAM(M)
        self.bps = self.qam.bps

        # use 802.11-like subcarriers on signed axis
        # data: 48, pilots: 4 at [-21,-7,7,21]
        self.pilot_ks = np.array([-21, -7, 7, 21], dtype=np.int32)
        used = np.hstack([np.arange(-26,0), np.arange(1,27)]).astype(np.int32)  # 52 used excl DC
        # remove pilots from used to form data
        self.data_ks = np.array([k for k in used if k not in set(self.pilot_ks)], dtype=np.int32)[:48]
        assert len(self.data_ks) == 48

        # fixed pilots (BPSK)
        rng = np.random.RandomState(7)
        self.pilots = (np.sign(rng.randn(len(self.pilot_ks))) + 0j).astype(np.complex64)

        self.k2i = lambda k: int(k + self.N//2)  # signed k -> fftshift index

    @property
    def sym_len(self): return self.N + self.CP

    @property
    def bits_per_frame(self): return len(self.data_ks) * self.n_sym * self.bps

    @property
    def samples_per_frame(self): return self.n_sym * self.sym_len

    def modulate(self, bits):
        bits = np.asarray(bits).astype(np.int8)
        need = self.bits_per_frame
        if len(bits) < need:
            bits = np.concatenate([bits, np.zeros(need-len(bits), np.int8)])
        bits = bits[:need]

        data_syms = self.qam.mod(bits).reshape(self.n_sym, len(self.data_ks))
        out = []
        for i in range(self.n_sym):
            Xs = np.zeros(self.N, dtype=np.complex64)  # fftshift spectrum
            # pilots
            for k, p in zip(self.pilot_ks, self.pilots):
                Xs[self.k2i(k)] = p
            # data
            for k, d in zip(self.data_ks, data_syms[i]):
                Xs[self.k2i(k)] = d

            x = np.fft.ifft(np.fft.ifftshift(Xs)) * np.sqrt(self.N)
            cp = x[-self.CP:]
            out.append(np.concatenate([cp, x]))
        return np.concatenate(out).astype(np.complex64)

    def demodulate(self, rx_td):
        rx_td = np.asarray(rx_td).astype(np.complex64)
        ns = min(len(rx_td)//self.sym_len, self.n_sym)
        if ns <= 0:
            return np.array([], np.int8), {"error": "no symbols"}

        pilot_i = np.array([self.k2i(k) for k in self.pilot_ks], dtype=np.int32)
        data_i  = np.array([self.k2i(k) for k in self.data_ks], dtype=np.int32)

        axis = np.arange(-self.N//2, self.N//2, dtype=np.float32)
        pilot_k = self.pilot_ks.astype(np.float32)
        order = np.argsort(pilot_k)
        pilot_k = pilot_k[order]

        all_eq = []
        snr_list, evm_list, cpe_list = [], [], []

        for si in range(ns):
            s = rx_td[si*self.sym_len + self.CP : si*self.sym_len + self.CP + self.N]
            Ys = np.fft.fftshift(np.fft.fft(s) / np.sqrt(self.N))

            # pilots
            Yp = Ys[pilot_i]
            Yp = Yp[order]
            Hp = Yp / (self.pilots[order] + 1e-12)

            # interpolate complex H over signed axis
            Hr = np.interp(axis, pilot_k, np.real(Hp))
            Hi = np.interp(axis, pilot_k, np.imag(Hp))
            Hs = (Hr + 1j*Hi).astype(np.complex64)

            # equalize data
            Xeq = Ys[data_i] / (Hs[data_i] + 1e-12)

            # --- CPE (common phase error) correction per symbol using pilots ---
            # estimate residual phase from equalized pilots
            Xp_eq = Ys[pilot_i] / (Hs[pilot_i] + 1e-12)
            cpe = np.angle(np.sum(Xp_eq * np.conj(self.pilots) ))
            Xeq *= np.exp(-1j * cpe)
            cpe_list.append(float(cpe))

            all_eq.append(Xeq)

            # quick SNR/EVM estimate vs hard decisions
            bits_h = self.qam.demod_hard(Xeq)
            sym_h = self.qam.mod(bits_h).reshape(-1)
            err = Xeq.reshape(-1) - sym_h
            evm = 100*np.sqrt(np.mean(np.abs(err)**2)/(np.mean(np.abs(sym_h)**2)+1e-12))
            evm_list.append(float(evm))
            snr = 10*np.log10(np.mean(np.abs(sym_h)**2)/(np.mean(np.abs(err)**2)+1e-12))
            snr_list.append(float(snr))

        all_eq = np.concatenate(all_eq).astype(np.complex64)
        bits = self.qam.demod_hard(all_eq)

        return bits, {
            "nsym": int(ns),
            "snr_db": float(np.mean(snr_list)) if snr_list else None,
            "evm_pct": float(np.mean(evm_list)) if evm_list else None,
            "cpe_rad": float(np.mean(cpe_list)) if cpe_list else None,
        }

# --------------------------
# Pluto link
# --------------------------
PRESETS = {
    "cable_30db":     dict(tx_gain=-10, rx_gain=40, tx_amp=0.4),
    "cable_direct":   dict(tx_gain=-30, rx_gain=20, tx_amp=0.1),
    "antenna_close":  dict(tx_gain=0,   rx_gain=60, tx_amp=0.6),
    "antenna_far":    dict(tx_gain=0,   rx_gain=70, tx_amp=0.8),
}

class Link:
    def __init__(self, ip, fc, fs, tx_gain, rx_gain, tx_amp, rx_buf):
        self.ip = ip; self.fc = fc; self.fs = fs
        self.tx_gain = tx_gain; self.rx_gain = rx_gain
        self.tx_amp = tx_amp; self.rx_buf = rx_buf
        self.sdr = None

    def connect(self):
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
        self.sdr.rx_buffer_size = int(self.rx_buf)
        self.sdr.tx_enabled_channels = [0]
        self.sdr.rx_enabled_channels = [0]
        # kernel buffers helps stability
        try:
            self.sdr._rxadc.set_kernel_buffers_count(4)
        except:
            pass
        print(f"[SDR] OK  fc={self.fc/1e6:.1f}MHz fs={self.fs/1e6:.1f}Msps TXgain={self.tx_gain} RXgain={self.rx_gain} tx_amp={self.tx_amp}")

    def tx_cyclic(self, x):
        x = np.asarray(x).astype(np.complex64)
        x = x / (np.max(np.abs(x)) + 1e-12) * float(self.tx_amp)
        tx = (x * (2**14)).astype(np.complex64)
        try:
            self.sdr.tx_destroy_buffer()
        except:
            pass
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(tx)

    def stop_tx(self):
        try:
            self.sdr.tx_destroy_buffer()
        except:
            pass

    def rx(self):
        # flush a bit
        for _ in range(3):
            _ = self.sdr.rx()
        return self.sdr.rx()

    def close(self):
        self.stop_tx()
        try:
            self.sdr.rx_destroy_buffer()
        except:
            pass

# --------------------------
# BER helpers
# --------------------------
def ber(tx_bits, rx_bits):
    L = min(len(tx_bits), len(rx_bits))
    if L <= 0:
        return None, None, None
    e = int(np.sum(tx_bits[:L] != rx_bits[:L]))
    return e / L, e, L

def refine_start(rx, zc, idx, search=16):
    L = len(zc)
    best = idx
    bestv = -1.0
    for d in range(-search, search+1):
        s = idx + d
        if s < 0 or s+L > len(rx):
            continue
        v = np.abs(np.vdot(zc, rx[s:s+L]))  # vdot = sum conj(zc)*rx
        if v > bestv:
            bestv = v
            best = s
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="ip:192.168.2.2")
    ap.add_argument("--freq", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=3e6)
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="antenna_close")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--rx_buf", type=int, default=65536)
    ap.add_argument("--sync_ratio", type=float, default=6.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--M", type=int, default=4)  # 4 or 16
    args = ap.parse_args()

    p = PRESETS[args.preset]
    ofdm = OFDM(N=64, CP=16, n_sym=14, M=args.M)
    zc = zc_seq(1024, root=25)
    gap = np.zeros(256, dtype=np.complex64)  # make sure preamble has room

    print("\n" + "="*78)
    print("PLUTO OFDM BER TEST (v3 - ZC sync + correct CFO + fftshift OFDM)")
    print("="*78)
    print(f"ip={args.ip}  fc={args.freq/1e6:.1f}MHz  fs={args.fs/1e6:.1f}Msps  preset={args.preset}")
    print(f"mod={args.M}  TXgain={p['tx_gain']}  RXgain={p['rx_gain']}  tx_amp={p['tx_amp']}  rx_buf={args.rx_buf}")
    print(f"OFDM: N=64 CP=16 data=48 pilots=4 syms=14  bits/frame={ofdm.bits_per_frame}")
    print(f"Preamble: ZC len={len(zc)}  sync_ratio_th={args.sync_ratio}")
    print("="*78)

    link = Link(args.ip, args.freq, args.fs, p["tx_gain"], p["rx_gain"], p["tx_amp"], args.rx_buf)
    link.connect()

    total_e = 0
    total_b = 0
    ok = 0

    print("\nFrame  BER        Err    Bits   SyncRatio  CFO(Hz)    SNR(dB)  EVM(%)  RXpeak  Status")
    print("-"*100)

    try:
        for fi in range(args.frames):
            rng = np.random.RandomState(1000 + fi)
            tx_bits = rng.randint(0, 2, ofdm.bits_per_frame).astype(np.int8)

            payload = ofdm.modulate(tx_bits)
            frame = np.concatenate([gap, zc, payload, gap])

            link.tx_cyclic(frame)
            time.sleep(0.15)

            rx = link.rx()
            rxpeak = float(np.max(np.abs(rx)))

            ok_sync, idx, ratio = sync_zc(rx, zc, ratio_th=args.sync_ratio, verbose=args.verbose)
            if not ok_sync:
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:8.2f}  N/A       N/A     N/A    {rxpeak:7.0f}  SYNC FAIL")
                link.stop_tx()
                continue

            # extract preamble & estimate CFO using known zc (critical!)
            if idx + len(zc) > len(rx):
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:8.2f}  N/A       N/A     N/A    {rxpeak:7.0f}  SHORT")
                link.stop_tx()
                continue

            if ok_sync:
                idx = refine_start(rx, zc, idx, search=16)
                rx_pre = rx[idx:idx+len(zc)]
                cfo = estimate_cfo_known_preamble(rx_pre, zc, args.fs, verbose=args.verbose)

            # payload start (after zc)
            p0 = idx + len(zc)
            p1 = p0 + ofdm.samples_per_frame
            if p1 > len(rx):
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:8.2f}  {cfo:8.1f}  N/A     N/A    {rxpeak:7.0f}  SHORT")
                link.stop_tx()
                continue

            rx_pay = rx[p0:p1]
            rx_pay = apply_cfo(rx_pay, args.fs, cfo)

            rx_bits, met = ofdm.demodulate(rx_pay)
            link.stop_tx()

            if rx_bits is None or len(rx_bits) < ofdm.bits_per_frame:
                print(f"{fi:<5}  N/A       N/A    N/A    {ratio:8.2f}  {cfo:8.1f}  N/A     N/A    {rxpeak:7.0f}  DECODE FAIL")
                continue

            rx_bits = rx_bits[:ofdm.bits_per_frame]
            b, e, L = ber(tx_bits, rx_bits)
            snr = met.get("snr_db", None)
            evm = met.get("evm_pct", None)

            status = "OK" if (b is not None and b < 1e-2) else "ERRORS"
            print(f"{fi:<5}  {b:8.2e}  {e:5d}  {L:5d}  {ratio:8.2f}  {cfo:8.1f}  {snr:7.1f}  {evm:6.1f}  {rxpeak:7.0f}  {status}")

            ok += 1
            total_e += e
            total_b += L

        print("-"*100)
        if total_b > 0:
            print(f"\nSUMMARY: frames={args.frames}, decoded={ok} ({100*ok/max(1,args.frames):.1f}%) bits={total_b} errors={total_e} BER={total_e/total_b:.3e}")
        else:
            print("\nSUMMARY: no successful decodes")
    finally:
        link.close()

if __name__ == "__main__":
    main()

#python pluto_ofdm_ber_test_v3.py --preset antenna_close --frames 3 --sync_ratio 6 --verbose