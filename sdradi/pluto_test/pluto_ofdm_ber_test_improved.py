#!/usr/bin/env python3
"""
pluto_ofdm_ber_test_improved.py - Physical OFDM Communication BER Test for PlutoSDR (Improved)

Key improvements vs your version:
1) Correct, consistent QPSK/16QAM mapping (Gray-coded for both).
2) Robust SNR/EVM estimate using nearest-constellation hard decisions (works for QPSK/16QAM).
3) Clean OFDM subcarrier plan (802.11a-like for FFT=64):
   - Used subcarriers: k = [-26..-1, +1..+26] (52 total, DC=0 unused)
   - Pilots default: k = [-21, -7, +7, +21] (4 pilots)
   - Data: remaining 48
   Configurable via OFDMConfig.pilot_bins / used_bins.
4) Better preamble:
   - Schmidl-Cox short preamble (time-domain half repetition) for coarse timing + CFO
   - One long training OFDM symbol (known BPSK on used subcarriers) for initial channel estimate
5) Timing refine: small search around coarse index to align to OFDM symbol boundary via CP correlation.
6) Pluto robustness: kernel buffers count; optional rx_destroy_buffer retry.

Usage examples:
  python pluto_ofdm_ber_test_improved.py --ip ip:192.168.2.2 --freq 2.3e9 --preset antenna_close --frames 20
  python pluto_ofdm_ber_test_improved.py --ip ip:192.168.2.2 --preset cable_30db --frames 30 --mod 4
  python pluto_ofdm_ber_test_improved.py --ip ip:192.168.2.2 --tx_gain_sweep

Notes:
- Keep antennas/cable safe: avoid RX saturation. If RX peak > ~28000, reduce RX gain or TX gain/amp.
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


# ==============================================================================
# Config
# ==============================================================================

class OFDMConfig:
    def __init__(
        self,
        fft_size: int = 64,
        cp_length: int = 16,
        num_symbols: int = 14,
        mod_order: int = 4,          # 4=QPSK, 16=16QAM
        sync_threshold: float = 10.0, # correlation peak / median threshold
        used_bins: tuple = None,      # FFT bin indices centered at DC (k in [-N/2..N/2-1], excluding 0 typically)
        pilot_bins: tuple = (-21, -7, 7, 21),
    ):
        self.fft_size = fft_size
        self.cp_length = cp_length
        self.num_symbols = num_symbols
        self.mod_order = mod_order
        self.sync_threshold = sync_threshold

        # Default to 802.11a-like bins for N=64
        if used_bins is None:
            # 52 used carriers: [-26..-1, +1..+26]
            used = list(range(-26, 0)) + list(range(1, 27))
            self.used_bins = tuple(used)
        else:
            self.used_bins = tuple(used_bins)

        self.pilot_bins = tuple(pilot_bins)

        # Derived bins
        self.data_bins = tuple([b for b in self.used_bins if b not in self.pilot_bins])

        # 802.11a expects 48 data carriers
        if len(self.data_bins) < 1:
            raise ValueError("No data bins configured. Check used_bins/pilot_bins.")
        self.num_data_carriers = len(self.data_bins)

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.mod_order))

    @property
    def bits_per_frame(self) -> int:
        return self.num_data_carriers * self.num_symbols * self.bits_per_symbol

    @property
    def samples_per_symbol(self) -> int:
        return self.fft_size + self.cp_length

    @property
    def samples_per_payload(self) -> int:
        return self.num_symbols * self.samples_per_symbol

    @property
    def samples_per_frame(self) -> int:
        # frame = short preamble + long training + payload
        return self.short_preamble_len + self.samples_per_symbol + self.samples_per_payload

    @property
    def short_preamble_len(self) -> int:
        # We build a Schmidl-Cox short preamble with L=N/2 repetition, repeated reps times
        # Default: reps=10 => length = 10*(N/2)
        return (self.fft_size // 2) * 10


class PlutoConfig:
    def __init__(
        self,
        ip: str = "ip:192.168.2.2",
        fc: float = 2.3e9,
        fs: float = 3e6,
        tx_gain: float = 0.0,
        rx_gain: float = 50.0,
        rx_buffer_size: int = 2**14,
        kernel_buffers: int = 4,
    ):
        self.ip = ip
        self.fc = fc
        self.fs = fs
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.rx_buffer_size = rx_buffer_size
        self.kernel_buffers = kernel_buffers


PRESETS = {
    "cable_30db":     {"tx_gain": -10, "rx_gain": 40, "tx_amp": 0.5},
    "cable_direct":   {"tx_gain": -30, "rx_gain": 20, "tx_amp": 0.1},
    "antenna_close":  {"tx_gain": 0,   "rx_gain": 50, "tx_amp": 0.9},
    "antenna_far":    {"tx_gain": 0,   "rx_gain": 70, "tx_amp": 0.9},
}


# ==============================================================================
# Utility: FFT bin conversion
# ==============================================================================

def k_to_index(k: int, N: int) -> int:
    """Convert centered FFT bin index k (can be negative) to numpy index [0..N-1]."""
    return k % N


# ==============================================================================
# QAM Modulation (Gray-coded)
# ==============================================================================

class QAMModulator:
    """Gray-coded QPSK / 16QAM with unit average power."""

    def __init__(self, mod_order: int = 4):
        if mod_order not in (4, 16):
            raise ValueError("Only 4(QPSK) and 16(16QAM) supported.")
        self.mod_order = mod_order
        self.bits_per_symbol = int(np.log2(mod_order))
        self.constellation, self.bit_labels = self._build_constellation()

    def _build_constellation(self):
        if self.mod_order == 4:
            # Gray QPSK mapping (b0 b1):
            # 00 -> +1+1j
            # 01 -> -1+1j
            # 11 -> -1-1j
            # 10 -> +1-1j
            pts = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)
            bits = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.int8)  # (b0,b1) little-endian in our pack/unpack
            return pts, bits

        # 16QAM Gray:
        # 2-bit Gray for I and Q: 00->-3, 01->-1, 11->+1, 10->+3
        def gray2_to_level(b0, b1):
            if (b0,b1) == (0,0): return -3
            if (b0,b1) == (1,0): return -1
            if (b0,b1) == (1,1): return  1
            if (b0,b1) == (0,1): return  3
            raise ValueError

        pts = []
        bits = []
        for q0,q1 in [(0,0),(1,0),(1,1),(0,1)]:  # Gray order for Q
            for i0,i1 in [(0,0),(1,0),(1,1),(0,1)]:  # Gray order for I
                I = gray2_to_level(i0,i1)
                Q = gray2_to_level(q0,q1)
                pts.append(I + 1j*Q)
                # Pack bits little-endian: [i0,i1,q0,q1] (you can reorder if you want; TX/RX consistent)
                bits.append([i0,i1,q0,q1])
        pts = np.array(pts, dtype=np.complex64) / np.sqrt(10)  # avg power = 1
        bits = np.array(bits, dtype=np.int8)
        return pts, bits

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.astype(np.int8).flatten()
        bps = self.bits_per_symbol
        pad = (-len(bits)) % bps
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])

        bits_reshaped = bits.reshape(-1, bps)

        # Find constellation point whose bit label matches. For speed, build index map.
        # Since small constellations, brute compare is fine.
        out = np.empty(bits_reshaped.shape[0], dtype=np.complex64)
        for i, b in enumerate(bits_reshaped):
            # locate row in bit_labels equal to b
            idx = np.where(np.all(self.bit_labels == b, axis=1))[0][0]
            out[i] = self.constellation[idx]
        return out

    def demodulate_hard(self, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hard demod: returns (bits, hard_symbols)."""
        x = symbols.flatten()
        const = self.constellation
        d = np.abs(x[:, None] - const[None, :])
        idx = np.argmin(d, axis=1)
        hard = const[idx]
        bits = self.bit_labels[idx].reshape(-1).astype(np.int8)
        return bits, hard


# ==============================================================================
# OFDM Transceiver
# ==============================================================================

class OFDMTransceiver:
    def __init__(self, cfg: OFDMConfig):
        self.cfg = cfg
        self.mod = QAMModulator(cfg.mod_order)

        # Convert bins to FFT indices
        N = cfg.fft_size
        self.used_idx = np.array([k_to_index(k, N) for k in cfg.used_bins], dtype=int)
        self.pilot_idx = np.array([k_to_index(k, N) for k in cfg.pilot_bins], dtype=int)
        self.data_idx = np.array([k_to_index(k, N) for k in cfg.data_bins], dtype=int)

        # Pilot symbols (BPSK), fixed
        rng = np.random.RandomState(42)
        self.pilot_symbols = np.sign(rng.randn(len(self.pilot_idx))).astype(np.float32) + 0j

        # Long training symbol (known BPSK on used carriers)
        rng2 = np.random.RandomState(7)
        self.long_train_bpsk = np.sign(rng2.randn(len(self.used_idx))).astype(np.float32) + 0j

    def build_short_preamble(self) -> np.ndarray:
        """
        Schmidl-Cox short preamble: time-domain half repetition.
        Build freq domain with random BPSK on even subcarriers of used set, odd = 0,
        so time-domain has N/2 periodicity. Repeat reps times.
        """
        N = self.cfg.fft_size
        L = N // 2
        reps = 10

        X = np.zeros(N, dtype=np.complex64)

        # Fill only even indices within used carriers to enforce repetition
        # We operate in FFT index domain; ensure DC=0 stays 0.
        used = self.used_idx
        even_used = used[used % 2 == 0]
        rng = np.random.RandomState(12345)
        bpsk = np.sign(rng.randn(len(even_used))).astype(np.float32) + 0j
        X[even_used] = bpsk

        x = np.fft.ifft(X) * np.sqrt(N)  # time-domain
        block = x[:L]  # due to repetition property, x[0:L] ~ x[L:2L]
        pre = np.tile(block, reps)
        # Normalize
        pre = pre / (np.max(np.abs(pre)) + 1e-12)
        return pre.astype(np.complex64)

    def build_long_training_symbol(self) -> np.ndarray:
        """
        One full OFDM symbol with CP, known BPSK on all used carriers.
        Used for initial channel estimate.
        """
        N = self.cfg.fft_size
        X = np.zeros(N, dtype=np.complex64)
        X[self.used_idx] = self.long_train_bpsk
        x = np.fft.ifft(X) * np.sqrt(N)
        cp = x[-self.cfg.cp_length:]
        sym = np.concatenate([cp, x])
        sym = sym / (np.max(np.abs(sym)) + 1e-12)
        return sym.astype(np.complex64)

    def ofdm_modulate_payload(self, bits: np.ndarray) -> np.ndarray:
        """Map bits to OFDM payload (num_symbols OFDM symbols with CP)."""
        cfg = self.cfg
        N = cfg.fft_size
        bps = cfg.bits_per_symbol
        n_data = len(self.data_idx)
        bits_needed = cfg.num_symbols * n_data * bps

        bits = bits.astype(np.int8).flatten()
        if len(bits) < bits_needed:
            bits = np.concatenate([bits, np.zeros(bits_needed - len(bits), dtype=np.int8)])
        else:
            bits = bits[:bits_needed]

        qam = self.mod.modulate(bits).reshape(cfg.num_symbols, n_data)

        out = []
        for s in range(cfg.num_symbols):
            X = np.zeros(N, dtype=np.complex64)
            X[self.data_idx] = qam[s]
            X[self.pilot_idx] = self.pilot_symbols
            x = np.fft.ifft(X) * np.sqrt(N)
            cp = x[-cfg.cp_length:]
            out.append(np.concatenate([cp, x]))
        payload = np.concatenate(out)
        payload = payload / (np.max(np.abs(payload)) + 1e-12)
        return payload.astype(np.complex64)

    def ofdm_demodulate_payload(self, payload: np.ndarray, H0: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Demod payload using channel estimate H0 (length N complex).
        Simple 1-tap equalization per subcarrier.
        """
        cfg = self.cfg
        N = cfg.fft_size
        sym_len = cfg.samples_per_symbol
        n_syms = len(payload) // sym_len

        rx_syms = []
        h_used_mag = []

        for s in range(min(n_syms, cfg.num_symbols)):
            start = s * sym_len + cfg.cp_length
            end = start + N
            if end > len(payload):
                break
            x = payload[start:end]
            X = np.fft.fft(x) / np.sqrt(N)

            # Pilot-based refinement: estimate residual common phase error (CPE)
            Hp = H0[self.pilot_idx]
            rp = X[self.pilot_idx]
            # Estimate phase error using pilots
            cpe = np.angle(np.sum(rp * np.conj(self.pilot_symbols * Hp)))
            X = X * np.exp(-1j * cpe)

            # Equalize
            Hd = H0[self.data_idx]
            Yd = X[self.data_idx] / (Hd + 1e-12)
            rx_syms.append(Yd)
            h_used_mag.append(np.mean(np.abs(H0[self.used_idx])))

        if not rx_syms:
            return np.array([], dtype=np.int8), {"error": "No OFDM symbols demodulated"}

        rx_syms = np.array(rx_syms).reshape(-1)

        bits_hat, hard = self.mod.demodulate_hard(rx_syms)

        # Robust SNR/EVM estimate from hard decisions
        err = rx_syms - hard
        sig_pwr = np.mean(np.abs(hard) ** 2)
        noise_pwr = np.mean(np.abs(err) ** 2)
        snr_db = 10 * np.log10(sig_pwr / (noise_pwr + 1e-12))
        evm_rms = np.sqrt(noise_pwr / (sig_pwr + 1e-12))
        evm_pct = 100 * evm_rms

        metrics = {
            "num_data_symbols": int(len(rx_syms)),
            "snr_est_db": float(snr_db),
            "evm_percent": float(evm_pct),
            "chan_gain_db": float(20 * np.log10(np.mean(h_used_mag) + 1e-12)),
            "constellation": rx_syms[:256].copy(),
        }
        return bits_hat, metrics

    def estimate_channel_from_long_training(self, rx_long_sym: np.ndarray) -> np.ndarray:
        """
        Estimate channel H0 from received long training OFDM symbol (with CP).
        """
        cfg = self.cfg
        N = cfg.fft_size
        start = cfg.cp_length
        end = start + N
        x = rx_long_sym[start:end]
        X = np.fft.fft(x) / np.sqrt(N)

        # Known training in freq
        Xref = np.zeros(N, dtype=np.complex64)
        Xref[self.used_idx] = self.long_train_bpsk

        H = np.ones(N, dtype=np.complex64)
        # Only estimate on used carriers; elsewhere keep 1 to avoid division issues
        H[self.used_idx] = X[self.used_idx] / (Xref[self.used_idx] + 1e-12)
        return H


# ==============================================================================
# Sync (Schmidl-Cox) + timing refine
# ==============================================================================

def schmidl_cox_coarse_sync(rx: np.ndarray, N: int, reps: int = 10, threshold_ratio: float = 10.0):
    """
    Coarse timing + CFO using repeated blocks length L=N/2.
    Metric: M(d) = |P(d)|^2 / (R(d)^2)
      P(d) = sum_{n=0..W-1} r[d+n+L] * conj(r[d+n])
      R(d) = sum_{n=0..W-1} |r[d+n+L]|^2
    Choose W = L.
    """
    L = N // 2
    W = L
    r = rx.astype(np.complex64)

    # Need at least: d+L+W <= len(r)
    if len(r) < (reps + 2) * L:
        return {"ok": False, "reason": "rx too short"}

    a = r[:-L]         # length = len(r)-L
    b = r[L:]          # length = len(r)-L
    prod = b * np.conj(a)
    power = (np.abs(b) ** 2).astype(np.float32)

    # Cumulative sums with leading zero so window sum is easy
    cP = np.concatenate([np.array([0+0j], dtype=np.complex64), np.cumsum(prod)])
    cR = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(power)])

    # Window sums length: len(r)-L-W+1
    d_len = len(r) - L - W + 1
    if d_len <= 0:
        return {"ok": False, "reason": "rx too short for window"}

    # For d=0..d_len-1: sum over [d, d+W)
    P = cP[W:W + d_len] - cP[0:d_len]
    R = cR[W:W + d_len] - cR[0:d_len]

    M = (np.abs(P) ** 2) / (R ** 2 + 1e-12)

    peak_idx = int(np.argmax(M))
    peak_val = float(M[peak_idx])
    med = float(np.median(M))

    ok = (peak_val / (med + 1e-12)) >= threshold_ratio
    cfo_phase = float(np.angle(P[peak_idx]))  # phase between repeated halves

    return {
        "ok": ok,
        "peak_idx": peak_idx,
        "peak_val": peak_val,
        "median": med,
        "cfo_phase": cfo_phase,
    }

def refine_symbol_boundary(rx: np.ndarray, coarse: int, cfg: OFDMConfig, search: int = None):
    """
    Refine timing near coarse index to align to OFDM symbol boundary by CP correlation.
    Search +/-search samples (default = cp_length).
    """
    N = cfg.fft_size
    cp = cfg.cp_length
    sym_len = cfg.samples_per_symbol
    if search is None:
        search = cp

    best = coarse
    best_metric = -1.0

    for d in range(max(0, coarse - search), min(len(rx) - sym_len - 1, coarse + search + 1)):
        # CP region: [d, d+cp), tail: [d+N, d+N+cp)
        if d + sym_len > len(rx):
            break
        cp_seg = rx[d:d+cp]
        tail = rx[d+cp+N-cp:d+cp+N]  # last cp samples of FFT part
        # correlation magnitude
        m = np.abs(np.vdot(cp_seg, tail)) / (np.linalg.norm(cp_seg)*np.linalg.norm(tail) + 1e-12)
        if m > best_metric:
            best_metric = float(m)
            best = d

    return best, best_metric


def apply_cfo(x: np.ndarray, fs: float, cfo_hz: float):
    n = np.arange(len(x), dtype=np.float64)
    rot = np.exp(-1j * 2*np.pi * cfo_hz * n / fs)
    return x * rot


# ==============================================================================
# Pluto Link
# ==============================================================================

class PlutoOFDMLink:
    def __init__(self, pcfg: PlutoConfig, ocfg: OFDMConfig, tx_amp: float):
        self.pcfg = pcfg
        self.ocfg = ocfg
        self.tx_amp = float(tx_amp)

        self.ofdm = OFDMTransceiver(ocfg)
        self.short_pre = self.ofdm.build_short_preamble()
        self.long_train = self.ofdm.build_long_training_symbol()

        self.sdr = None

    def connect(self) -> bool:
        if not ADI_AVAILABLE:
            print("[Error] pyadi-iio not available.")
            return False
        try:
            print(f"[SDR] Connecting {self.pcfg.ip} ...")
            self.sdr = adi.Pluto(uri=self.pcfg.ip)

            self.sdr.sample_rate = int(self.pcfg.fs)
            self.sdr.tx_lo = int(self.pcfg.fc)
            self.sdr.rx_lo = int(self.pcfg.fc)
            self.sdr.tx_rf_bandwidth = int(self.pcfg.fs)
            self.sdr.rx_rf_bandwidth = int(self.pcfg.fs)

            self.sdr.tx_hardwaregain_chan0 = float(self.pcfg.tx_gain)
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = float(self.pcfg.rx_gain)

            self.sdr.tx_enabled_channels = [0]
            self.sdr.rx_enabled_channels = [0]
            self.sdr.rx_buffer_size = int(self.pcfg.rx_buffer_size)

            # Robustness
            if hasattr(self.sdr, "_rxadc") and hasattr(self.sdr._rxadc, "set_kernel_buffers_count"):
                try:
                    self.sdr._rxadc.set_kernel_buffers_count(int(self.pcfg.kernel_buffers))
                except Exception:
                    pass

            print(f"[SDR] OK  fc={self.pcfg.fc/1e6:.1f}MHz fs={self.pcfg.fs/1e6:.1f}Msps "
                  f"TXgain={self.pcfg.tx_gain} RXgain={self.pcfg.rx_gain} tx_amp={self.tx_amp}")
            return True
        except Exception as e:
            print(f"[SDR] Connection failed: {e}")
            return False

    def disconnect(self):
        if self.sdr is None:
            return
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass

    def build_frame(self, bits: np.ndarray) -> np.ndarray:
        payload = self.ofdm.ofdm_modulate_payload(bits)
        # frame = short preamble + long training + payload
        frame = np.concatenate([self.short_pre, self.long_train, payload]).astype(np.complex64)
        # normalize + scale
        frame = frame / (np.max(np.abs(frame)) + 1e-12) * self.tx_amp
        return frame

    def transmit_cyclic(self, frame: np.ndarray):
        # Pluto expects complex64 scaled by 2^14 typical
        tx = (frame * (2**14)).astype(np.complex64)
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(tx)

    def stop_tx(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass

    def receive_once(self) -> np.ndarray:
        # flush a couple buffers
        for _ in range(2):
            _ = self.sdr.rx()
        return self.sdr.rx()

    def receive_and_decode(self, expected_bits: np.ndarray, num_captures: int = 5, verbose: bool = False):
        """
        Try multiple captures, return best (lowest BER / highest SNR).
        """
        ocfg = self.ocfg
        best = None

        for ci in range(num_captures):
            try:
                rx_raw = self.receive_once()
                peak = float(np.max(np.abs(rx_raw)))
                if verbose:
                    print(f"  [RX] cap={ci} len={len(rx_raw)} peak={peak:.0f}")

                # Normalize for processing (keep complex)
                pwr = np.sqrt(np.mean(np.abs(rx_raw)**2)) + 1e-12
                rx = (rx_raw / pwr).astype(np.complex64)

                # Coarse sync on short preamble
                sc = schmidl_cox_coarse_sync(rx, ocfg.fft_size, reps=10, threshold_ratio=ocfg.sync_threshold)
                if not sc["ok"]:
                    if verbose:
                        print(f"    [SYNC] fail peak/med={sc.get('peak_val',0):.2e}/{sc.get('median',0):.2e}")
                    # optional: reset DMA if repeated failures
                    if ci == 2:
                        try:
                            self.sdr.rx_destroy_buffer()
                        except Exception:
                            pass
                    continue

                coarse = int(sc["peak_idx"])
                L = ocfg.fft_size // 2
                # CFO estimate (Hz): angle(P)*fs/(2*pi*L)
                cfo_hz = sc["cfo_phase"] * self.pcfg.fs / (2*np.pi*L)

                # Apply CFO correction to whole stream (starting near coarse helps, but ok)
                rx_c = apply_cfo(rx, self.pcfg.fs, cfo_hz)

                # Refine to OFDM symbol boundary near end of short preamble
                # We expect long training starts after short preamble length
                long_start_coarse = coarse + ocfg.short_preamble_len
                long_start, cp_metric = refine_symbol_boundary(rx_c, long_start_coarse, ocfg, search=ocfg.cp_length)

                if verbose:
                    print(f"    [SYNC] coarse={coarse} cfo={cfo_hz:.1f}Hz long_start={long_start} cpM={cp_metric:.3f}")

                # Extract long training symbol (with CP)
                sym_len = ocfg.samples_per_symbol
                if long_start + sym_len > len(rx_c):
                    continue
                rx_long = rx_c[long_start:long_start + sym_len]

                # Channel estimate
                H0 = self.ofdm.estimate_channel_from_long_training(rx_long)

                # Payload begins right after long training symbol
                payload_start = long_start + sym_len
                payload_len = ocfg.samples_per_payload
                if payload_start + payload_len > len(rx_c):
                    continue
                payload = rx_c[payload_start:payload_start + payload_len]

                # Demod
                bits_hat, met = self.ofdm.ofdm_demodulate_payload(payload, H0)
                if bits_hat.size == 0:
                    continue

                # BER
                ber, errs, ncmp = calculate_ber(expected_bits, bits_hat)

                out = {
                    "ber": ber,
                    "errors": errs,
                    "compared": ncmp,
                    "snr_est_db": met.get("snr_est_db", -np.inf),
                    "evm_percent": met.get("evm_percent", np.nan),
                    "chan_gain_db": met.get("chan_gain_db", np.nan),
                    "cfo_hz": float(cfo_hz),
                    "rx_peak": peak,
                    "sync_cp_metric": cp_metric,
                    "decoded_bits": bits_hat,
                }

                # Choose best: primary BER, tie-breaker by SNR
                if best is None:
                    best = out
                else:
                    if (out["ber"] < best["ber"]) or (out["ber"] == best["ber"] and out["snr_est_db"] > best["snr_est_db"]):
                        best = out

            except Exception as e:
                if verbose:
                    print(f"  [RX] cap={ci} error: {e}")
                continue

        return best


# ==============================================================================
# BER helpers
# ==============================================================================

def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray):
    tx_bits = tx_bits.astype(np.int8).flatten()
    rx_bits = rx_bits.astype(np.int8).flatten()
    n = min(len(tx_bits), len(rx_bits))
    if n <= 0:
        return 1.0, 0, 0
    errs = int(np.sum(tx_bits[:n] != rx_bits[:n]))
    return errs / n, errs, n


# ==============================================================================
# Main test routines
# ==============================================================================

def run_ber_test(ip: str, freq: float, preset: str, frames: int, mod_order: int,
                 tx_gain=None, rx_gain=None, verbose=False):
    settings = PRESETS.get(preset, PRESETS["antenna_close"]).copy()
    if tx_gain is not None:
        settings["tx_gain"] = tx_gain
    if rx_gain is not None:
        settings["rx_gain"] = rx_gain

    pcfg = PlutoConfig(ip=ip, fc=freq, tx_gain=settings["tx_gain"], rx_gain=settings["rx_gain"])
    ocfg = OFDMConfig(mod_order=mod_order)

    link = PlutoOFDMLink(pcfg, ocfg, tx_amp=settings["tx_amp"])

    print("\n" + "="*78)
    print("PLUTO OFDM BER TEST (Improved)")
    print("="*78)
    print(f"ip={ip}  fc={freq/1e6:.1f}MHz  fs={pcfg.fs/1e6:.1f}Msps  preset={preset}")
    print(f"mod={mod_order}  TXgain={pcfg.tx_gain}  RXgain={pcfg.rx_gain}  tx_amp={settings['tx_amp']}")
    print(f"OFDM: N={ocfg.fft_size} CP={ocfg.cp_length} dataCarriers={ocfg.num_data_carriers} pilots={len(ocfg.pilot_bins)} syms={ocfg.num_symbols}")
    print("="*78)

    if not link.connect():
        return None

    total_err = 0
    total_bits = 0
    ok_frames = 0
    snrs = []
    evms = []
    peaks = []

    print(f"\n{'Frame':<6} {'BER':<12} {'Err':<8} {'Bits':<8} {'SNR(dB)':<8} {'EVM(%)':<8} {'CFO(Hz)':<10} {'RXpeak':<8} {'Status'}")
    print("-"*90)

    try:
        for fi in range(frames):
            # deterministic random per frame
            rng = np.random.RandomState(1000 + fi)
            tx_bits = rng.randint(0, 2, size=ocfg.bits_per_frame, dtype=np.int8)

            frame = link.build_frame(tx_bits)
            link.transmit_cyclic(frame)

            # short settle
            time.sleep(0.08)

            best = link.receive_and_decode(tx_bits, num_captures=6, verbose=verbose)
            link.stop_tx()

            if best is None:
                print(f"{fi:<6} {'N/A':<12} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<8} SYNC/DECODE FAIL")
                time.sleep(0.05)
                continue

            ber = best["ber"]
            errs = best["errors"]
            ncmp = best["compared"]
            snr = best["snr_est_db"]
            evm = best["evm_percent"]
            cfo = best["cfo_hz"]
            peak = best["rx_peak"]

            total_err += errs
            total_bits += ncmp
            ok_frames += 1
            snrs.append(snr)
            evms.append(evm)
            peaks.append(peak)

            status = "OK"
            if peak > 28000:
                status = "SATURATED?"
            elif ber > 1e-2:
                status = "ERRORS"

            print(f"{fi:<6} {ber:<12.2e} {errs:<8d} {ncmp:<8d} {snr:<8.1f} {evm:<8.1f} {cfo:<10.1f} {peak:<8.0f} {status}")

            time.sleep(0.05)

        print("-"*90)
        print("\n" + "="*78)
        print("SUMMARY")
        print("="*78)
        print(f"Frames total: {frames}")
        print(f"Frames decoded: {ok_frames} ({(ok_frames/frames*100):.1f}%)")
        if total_bits > 0:
            overall_ber = total_err / total_bits
            print(f"Total bits compared: {total_bits}")
            print(f"Total bit errors:    {total_err}")
            print(f"Overall BER:         {overall_ber:.2e}")
        else:
            overall_ber = 1.0
            print("No successful decodes.")

        if snrs:
            print(f"SNR(dB): mean={np.mean(snrs):.1f} min={np.min(snrs):.1f} max={np.max(snrs):.1f}")
        if evms:
            print(f"EVM(%):  mean={np.mean(evms):.1f} min={np.min(evms):.1f} max={np.max(evms):.1f}")
        if peaks:
            print(f"RX peak: mean={np.mean(peaks):.0f} max={np.max(peaks):.0f}")

        print("-"*78)
        if total_bits > 0 and overall_ber < 1e-3 and ok_frames >= int(0.8 * frames):
            print("\033[92m[PASSED]\033[0m BER < 1e-3 and >=80% frames decoded")
        elif total_bits > 0 and overall_ber < 1e-2:
            print("\033[93m[MARGINAL]\033[0m BER < 1e-2 (improve gain/distance/timing)")
        else:
            print("\033[91m[FAILED]\033[0m Too many failures or BER too high")
        print("="*78)

        return {
            "frames": frames,
            "decoded": ok_frames,
            "total_bits": total_bits,
            "total_err": total_err,
            "ber": overall_ber,
            "snr_mean": float(np.mean(snrs)) if snrs else None,
            "evm_mean": float(np.mean(evms)) if evms else None,
        }

    finally:
        link.disconnect()


def run_tx_gain_sweep(ip: str, freq: float, preset: str, mod_order: int):
    tx_gains = [-30, -20, -10, -5, 0]
    print("\n" + "="*78)
    print("TX GAIN SWEEP")
    print("="*78)
    print(f"ip={ip}  fc={freq/1e6:.1f}MHz  preset={preset}  mod={mod_order}")
    print("-"*78)
    print(f"{'TXgain':<8} {'BER':<12} {'SNRmean(dB)':<12} {'EVMmean(%)':<12} {'FramesOK':<10}")
    print("-"*78)

    best = None
    for g in tx_gains:
        r = run_ber_test(ip, freq, preset, frames=8, mod_order=mod_order, tx_gain=g, rx_gain=None, verbose=False)
        if not r:
            continue
        ber = r["ber"]
        snr = r["snr_mean"]
        evm = r["evm_mean"]
        ok = r["decoded"]
        print(f"{g:<8} {ber:<12.2e} {('%.1f'%snr if snr is not None else 'N/A'):<12} {('%.1f'%evm if evm is not None else 'N/A'):<12} {ok:<10}")

        if best is None or ber < best["ber"]:
            best = {"tx_gain": g, "ber": ber}

    print("-"*78)
    if best:
        print(f"Best TX gain: {best['tx_gain']} dB (BER={best['ber']:.2e})")
    else:
        print("No valid results.")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Improved Pluto OFDM BER Test")
    parser.add_argument("--ip", default="ip:192.168.2.2")
    parser.add_argument("--freq", type=float, default=2.3e9)
    parser.add_argument("--preset", default="antenna_close",
                        choices=["cable_30db", "cable_direct", "antenna_close", "antenna_far"])
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--mod", type=int, default=4, choices=[4, 16], help="4=QPSK, 16=16QAM")
    parser.add_argument("--tx_gain", type=float, default=None)
    parser.add_argument("--rx_gain", type=float, default=None)
    parser.add_argument("--tx_gain_sweep", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not ADI_AVAILABLE:
        print("[Error] pyadi-iio not available. Install with: pip install pyadi-iio")
        sys.exit(1)

    if args.tx_gain_sweep:
        run_tx_gain_sweep(args.ip, args.freq, args.preset, args.mod)
    else:
        run_ber_test(args.ip, args.freq, args.preset, args.frames, args.mod,
                     tx_gain=args.tx_gain, rx_gain=args.rx_gain, verbose=args.verbose)


if __name__ == "__main__":
    main()