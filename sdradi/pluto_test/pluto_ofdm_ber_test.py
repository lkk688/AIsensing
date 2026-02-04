#!/usr/bin/env python3
"""
pluto_ofdm_ber_test.py - Physical OFDM Communication BER Test for PlutoSDR

This script performs real OFDM communication over the air using PlutoSDR hardware.
It transmits OFDM frames with known data, receives them, and calculates BER.

Based on the link_test results from sdr_auto_tune.py and OFDM components from
sdr_video_commv2_lab.py.

Features:
- Physical TX/RX via PlutoSDR (1R1T mode)
- OFDM modulation with QPSK/16QAM
- Preamble-based synchronization
- Channel estimation and equalization
- BER calculation with known transmitted data
- Support for different frequencies to avoid WiFi

Usage:
    # Basic BER test at 2.3 GHz (recommended - avoids WiFi)
    python pluto_ofdm_ber_test.py --ip ip:192.168.2.2 --freq 2.3e9

    # Test with antenna preset
    python pluto_ofdm_ber_test.py --ip ip:192.168.2.2 --preset antenna_close

    # Sweep SNR by adjusting TX gain
    python pluto_ofdm_ber_test.py --ip ip:192.168.2.2 --tx_gain_sweep

Author: AI-assisted development
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
    print("[Error] pyadi-iio not available. Install with: pip install pyadi-iio")


# ==============================================================================
# Configuration Classes
# ==============================================================================

class OFDMConfig:
    """OFDM Waveform Configuration."""
    def __init__(
        self,
        fft_size: int = 64,
        cp_length: int = 16,
        num_data_carriers: int = 48,
        pilot_carriers: tuple = (6, 20, 34, 48),  # Indices in used carriers
        mod_order: int = 4,  # QPSK
        num_symbols: int = 14,
        sync_threshold: float = 30.0
    ):
        self.fft_size = fft_size
        self.cp_length = cp_length
        self.num_data_carriers = num_data_carriers
        self.pilot_carriers = pilot_carriers
        self.mod_order = mod_order
        self.num_symbols = num_symbols
        self.sync_threshold = sync_threshold

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.mod_order))

    @property
    def bits_per_frame(self) -> int:
        return self.num_data_carriers * self.num_symbols * self.bits_per_symbol

    @property
    def samples_per_frame(self) -> int:
        return self.num_symbols * (self.fft_size + self.cp_length)


class PlutoConfig:
    """PlutoSDR Configuration."""
    def __init__(
        self,
        ip: str = "ip:192.168.2.2",
        fc: float = 2.3e9,  # 2.3 GHz - outside WiFi band
        fs: float = 3e6,    # 3 MSPS
        tx_gain: float = 0,
        rx_gain: float = 50,
        rx_buffer_size: int = 2**14
    ):
        self.ip = ip
        self.fc = fc
        self.fs = fs
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.rx_buffer_size = rx_buffer_size


# Presets for different scenarios
PRESETS = {
    "cable_30db": {"tx_gain": -10, "rx_gain": 40, "tx_amp": 0.5},
    "cable_direct": {"tx_gain": -30, "rx_gain": 20, "tx_amp": 0.1},
    "antenna_close": {"tx_gain": 0, "rx_gain": 50, "tx_amp": 0.9},
    "antenna_far": {"tx_gain": 0, "rx_gain": 70, "tx_amp": 0.9},
}


# ==============================================================================
# QAM Modulation
# ==============================================================================

class QAMModulator:
    """QAM modulation and demodulation with Gray coding."""

    def __init__(self, mod_order: int = 4):
        self.mod_order = mod_order
        self.bits_per_symbol = int(np.log2(mod_order))
        self.constellation = self._create_constellation()

    def _create_constellation(self) -> np.ndarray:
        """Create Gray-coded QAM constellation."""
        if self.mod_order == 4:  # QPSK
            return np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif self.mod_order == 16:  # 16-QAM
            levels = np.array([-3, -1, 1, 3])
            grid = np.array([[x + 1j*y for x in levels] for y in levels])
            return grid.flatten() / np.sqrt(10)
        else:
            raise ValueError(f"Unsupported modulation order: {self.mod_order}")

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bit array to QAM symbols."""
        num_bits = len(bits)
        pad_len = (self.bits_per_symbol - num_bits % self.bits_per_symbol) % self.bits_per_symbol
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])

        bits_reshaped = bits.reshape(-1, self.bits_per_symbol)
        indices = np.zeros(len(bits_reshaped), dtype=int)
        for i in range(self.bits_per_symbol):
            indices += bits_reshaped[:, i].astype(int) << i

        return self.constellation[indices]

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Convert QAM symbols to bit array (hard decision)."""
        distances = np.abs(symbols[:, None] - self.constellation[None, :])
        indices = np.argmin(distances, axis=1)

        bits = np.zeros((len(indices), self.bits_per_symbol), dtype=int)
        for i in range(self.bits_per_symbol):
            bits[:, i] = (indices >> i) & 1

        return bits.flatten()


# ==============================================================================
# OFDM Transceiver
# ==============================================================================

class OFDMTransceiver:
    """OFDM Modulator/Demodulator for physical communication."""

    def __init__(self, config: OFDMConfig = None):
        self.config = config or OFDMConfig()
        self.modulator = QAMModulator(self.config.mod_order)
        self._setup_carriers()

        # Generate pilot symbols (BPSK, reproducible)
        np.random.seed(42)
        self.pilot_symbols = np.sign(np.random.randn(len(self.config.pilot_carriers))) + 0j

    def _setup_carriers(self):
        """Setup data and pilot carrier indices."""
        fft_size = self.config.fft_size
        num_data = self.config.num_data_carriers
        num_pilot = len(self.config.pilot_carriers)
        total_used = num_data + num_pilot

        if total_used % 2 != 0:
            total_used -= 1

        half_used = total_used // 2

        # Positive frequencies
        pos_carriers = np.arange(1, half_used + 1)
        # Negative frequencies
        neg_carriers = np.arange(fft_size - half_used, fft_size)

        used_carriers = np.sort(np.concatenate([pos_carriers, neg_carriers]))

        # Pilot carriers (evenly spaced)
        pilot_spacing = len(used_carriers) // num_pilot
        self.pilot_indices = used_carriers[::pilot_spacing][:num_pilot]

        # Data carriers
        self.data_indices = np.array([c for c in used_carriers if c not in self.pilot_indices])
        self.data_indices = self.data_indices[:num_data]

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to OFDM time-domain signal."""
        cfg = self.config
        bits_per_symbol = cfg.bits_per_symbol
        num_data = len(self.data_indices)

        bits_per_frame = num_data * cfg.num_symbols * bits_per_symbol
        num_frames = (len(bits) + bits_per_frame - 1) // bits_per_frame

        total_bits = num_frames * bits_per_frame
        if len(bits) < total_bits:
            bits = np.concatenate([bits, np.zeros(total_bits - len(bits), dtype=int)])

        all_tx_signal = []

        for frame_idx in range(num_frames):
            frame_bits = bits[frame_idx * bits_per_frame:(frame_idx + 1) * bits_per_frame]

            data_symbols = self.modulator.modulate(frame_bits)
            data_symbols = data_symbols.reshape(cfg.num_symbols, num_data)

            for sym_idx in range(cfg.num_symbols):
                freq_sym = np.zeros(cfg.fft_size, dtype=complex)
                freq_sym[self.data_indices] = data_symbols[sym_idx]
                freq_sym[self.pilot_indices] = self.pilot_symbols

                time_sym = np.fft.ifft(freq_sym) * np.sqrt(cfg.fft_size)
                cp = time_sym[-cfg.cp_length:]
                all_tx_signal.append(np.concatenate([cp, time_sym]))

        return np.concatenate(all_tx_signal)

    def demodulate(self, signal: np.ndarray) -> tuple:
        """Demodulate OFDM signal to bits."""
        cfg = self.config
        symbol_len = cfg.fft_size + cfg.cp_length
        num_data = len(self.data_indices)

        total_symbols = len(signal) // symbol_len
        all_rx_symbols = []
        channel_gains = []

        for sym_idx in range(total_symbols):
            start = sym_idx * symbol_len + cfg.cp_length
            end = start + cfg.fft_size
            if end > len(signal):
                break

            time_sym = signal[start:end]
            freq_sym = np.fft.fft(time_sym) / np.sqrt(cfg.fft_size)

            # Channel estimation from pilots
            rx_pilots = freq_sym[self.pilot_indices]
            h_pilots = rx_pilots / self.pilot_symbols

            # Interpolate channel across all carriers
            h_mag = np.abs(h_pilots)
            h_interp_mag = np.interp(np.arange(cfg.fft_size), self.pilot_indices, h_mag)

            h_phase = np.angle(h_pilots)
            h_phase_unwrapped = np.unwrap(h_phase)
            h_interp_phase = np.interp(np.arange(cfg.fft_size), self.pilot_indices, h_phase_unwrapped)

            h_interp = h_interp_mag * np.exp(1j * h_interp_phase)
            channel_gains.append(np.mean(np.abs(h_interp)))

            # Equalize data carriers
            h_data = h_interp[self.data_indices]
            eq_data = freq_sym[self.data_indices] / (h_data + 1e-10)
            all_rx_symbols.append(eq_data)

        if not all_rx_symbols:
            return np.array([], dtype=int), {'error': 'No symbols decoded'}

        all_rx_symbols = np.array(all_rx_symbols)
        bits = self.modulator.demodulate(all_rx_symbols.flatten())

        # Metrics
        sig_pwr = np.mean(np.abs(all_rx_symbols)**2)
        hard = np.sign(all_rx_symbols.real) + 1j*np.sign(all_rx_symbols.imag)
        err = all_rx_symbols - hard
        noise_pwr = np.mean(np.abs(err)**2)
        snr_est = 10 * np.log10(sig_pwr / (noise_pwr + 1e-10))

        metrics = {
            'num_symbols': len(all_rx_symbols),
            'channel_gain_db': 20 * np.log10(np.mean(channel_gains) + 1e-10),
            'snr_est_db': snr_est,
            'constellation': all_rx_symbols.flatten()[:256],
        }

        return bits, metrics


# ==============================================================================
# Preamble Generation and Synchronization
# ==============================================================================

def generate_preamble(block_len: int = 16, repetitions: int = 10) -> np.ndarray:
    """Generate Schmidl-Cox style preamble for synchronization."""
    rng = np.random.RandomState(12345)  # Fixed seed for reproducibility
    block = (rng.choice([-1, 1], block_len) + 1j * rng.choice([-1, 1], block_len)) / np.sqrt(2)
    return np.tile(block, repetitions)


def synchronize(signal: np.ndarray, preamble: np.ndarray, threshold: float = 30.0, fs: float = 3e6, verbose: bool = False) -> dict:
    """
    Synchronize received signal using preamble correlation.

    Returns dict with sync info including payload start index.
    """
    # Remove DC offset
    signal = signal - np.mean(signal)

    # Normalize signal for consistent correlation
    sig_power = np.sqrt(np.mean(np.abs(signal)**2))
    if sig_power > 1e-10:
        signal_norm = signal / sig_power
    else:
        signal_norm = signal

    # Correlation-based detection
    corr = np.abs(np.correlate(signal_norm, preamble / np.sqrt(np.mean(np.abs(preamble)**2)), mode='valid'))

    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]

    # Calculate noise floor for dynamic threshold
    noise_floor = np.median(corr)
    dynamic_threshold = max(threshold, noise_floor * 3)

    if verbose:
        print(f"    [Sync] Peak: {peak_val:.1f}, Noise: {noise_floor:.1f}, Threshold: {dynamic_threshold:.1f}")

    if peak_val < dynamic_threshold:
        return {
            'sync_success': False,
            'peak_val': peak_val,
            'peak_idx': peak_idx,
            'payload_start': None,
            'noise_floor': noise_floor,
            'incomplete': len(signal) < len(preamble) * 2
        }

    # CFO estimation using Schmidl-Cox
    L = len(preamble) // 10  # Block length (preamble is repeated)
    preamble_start = peak_idx

    if preamble_start + 2*L > len(signal):
        return {
            'sync_success': False,
            'peak_val': peak_val,
            'peak_idx': peak_idx,
            'payload_start': None,
            'incomplete': True
        }

    # Extract two consecutive blocks for CFO estimation
    block1 = signal[preamble_start:preamble_start + L]
    block2 = signal[preamble_start + L:preamble_start + 2*L]

    # CFO estimate using phase difference
    cfo_metric = np.sum(block2 * np.conj(block1))
    cfo_phase = np.angle(cfo_metric)
    cfo_hz = cfo_phase * fs / (2 * np.pi * L)

    if verbose:
        print(f"    [Sync] CFO estimate: {cfo_hz:.1f} Hz")

    # Payload starts after preamble
    payload_start = preamble_start + len(preamble)

    return {
        'sync_success': True,
        'peak_val': peak_val,
        'peak_idx': peak_idx,
        'cfo_phase': cfo_phase,
        'cfo_hz': cfo_hz,
        'payload_start': payload_start,
        'noise_floor': noise_floor
    }


# ==============================================================================
# PlutoSDR Interface
# ==============================================================================

class PlutoOFDMLink:
    """PlutoSDR OFDM communication link."""

    def __init__(self, pluto_config: PlutoConfig, ofdm_config: OFDMConfig, tx_amp: float = 0.9):
        self.pluto_cfg = pluto_config
        self.ofdm_cfg = ofdm_config
        self.tx_amp = tx_amp

        self.ofdm = OFDMTransceiver(ofdm_config)
        self.preamble = generate_preamble(block_len=16, repetitions=10)

        self.sdr = None

    def connect(self) -> bool:
        """Connect to PlutoSDR."""
        if not ADI_AVAILABLE:
            print("[Error] pyadi-iio not available")
            return False

        try:
            print(f"[SDR] Connecting to {self.pluto_cfg.ip}...")
            self.sdr = adi.Pluto(uri=self.pluto_cfg.ip)

            # Configure
            self.sdr.sample_rate = int(self.pluto_cfg.fs)
            self.sdr.tx_lo = int(self.pluto_cfg.fc)
            self.sdr.rx_lo = int(self.pluto_cfg.fc)
            self.sdr.tx_rf_bandwidth = int(self.pluto_cfg.fs)
            self.sdr.rx_rf_bandwidth = int(self.pluto_cfg.fs)
            self.sdr.tx_hardwaregain_chan0 = self.pluto_cfg.tx_gain
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = self.pluto_cfg.rx_gain
            self.sdr.rx_buffer_size = self.pluto_cfg.rx_buffer_size
            self.sdr.tx_enabled_channels = [0]
            self.sdr.rx_enabled_channels = [0]

            print(f"[SDR] Connected. TX LO: {self.sdr.tx_lo/1e6:.1f} MHz, "
                  f"TX Gain: {self.pluto_cfg.tx_gain} dB, RX Gain: {self.pluto_cfg.rx_gain} dB")
            return True

        except Exception as e:
            print(f"[SDR] Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from PlutoSDR."""
        if self.sdr:
            try:
                self.sdr.tx_destroy_buffer()
            except:
                pass
            try:
                self.sdr.rx_destroy_buffer()
            except:
                pass

    def transmit_frame(self, bits: np.ndarray) -> np.ndarray:
        """
        Build and transmit an OFDM frame with preamble.

        Returns the transmitted signal (for BER reference).
        """
        # Modulate data
        ofdm_signal = self.ofdm.modulate(bits)

        # Add gap before and after
        gap = np.zeros(100, dtype=complex)

        # Build frame: gap + preamble + ofdm_signal + gap
        tx_frame = np.concatenate([gap, self.preamble, ofdm_signal, gap])

        # Scale for DAC
        tx_frame = tx_frame / (np.max(np.abs(tx_frame)) + 1e-10) * self.tx_amp

        # Convert to int16 for PlutoSDR
        tx_samples = (tx_frame * 2**14).astype(np.complex64)

        # Transmit (cyclic)
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(tx_samples)

        return tx_frame

    def receive_and_decode(self, num_captures: int = 3, verbose: bool = False) -> tuple:
        """
        Receive signal, synchronize, and decode OFDM.

        Returns (decoded_bits, metrics, raw_rx_signal)
        """
        all_metrics = []
        best_bits = None
        best_snr = -np.inf
        raw_rx = None

        for capture_idx in range(num_captures):
            try:
                # Flush buffers
                for _ in range(3):
                    self.sdr.rx()

                # Capture
                rx_signal = self.sdr.rx()
                raw_rx = rx_signal

                if verbose:
                    print(f"  [RX] Capture {capture_idx}: {len(rx_signal)} samples, "
                          f"peak={np.max(np.abs(rx_signal)):.0f}")

                # Normalize (keep phase, normalize amplitude)
                rx_power = np.sqrt(np.mean(np.abs(rx_signal)**2))
                if rx_power > 1e-10:
                    rx_signal = rx_signal / rx_power * 0.5
                else:
                    continue

                # Synchronize
                sync_result = synchronize(
                    rx_signal, self.preamble,
                    self.ofdm_cfg.sync_threshold,
                    fs=self.pluto_cfg.fs,
                    verbose=verbose
                )

                if not sync_result['sync_success']:
                    all_metrics.append({
                        'capture': capture_idx,
                        'sync_success': False,
                        'peak_val': sync_result['peak_val']
                    })
                    continue

                # Extract payload
                payload_start = sync_result['payload_start']
                expected_len = self.ofdm_cfg.samples_per_frame

                if payload_start + expected_len > len(rx_signal):
                    all_metrics.append({
                        'capture': capture_idx,
                        'sync_success': True,
                        'error': 'Insufficient samples after sync'
                    })
                    continue

                payload = rx_signal[payload_start:payload_start + expected_len]

                # CFO correction
                if 'cfo_hz' in sync_result:
                    cfo_hz = sync_result['cfo_hz']
                    t = np.arange(len(payload)) / self.pluto_cfg.fs
                    cfo_correction = np.exp(-1j * 2 * np.pi * cfo_hz * t)
                    payload = payload * cfo_correction

                    if verbose:
                        print(f"    [CFO] Applied correction: {cfo_hz:.1f} Hz")

                # Demodulate
                bits, demod_metrics = self.ofdm.demodulate(payload)

                demod_metrics['capture'] = capture_idx
                demod_metrics['sync_success'] = True
                demod_metrics['sync_peak'] = sync_result['peak_val']
                demod_metrics['cfo_hz'] = sync_result.get('cfo_hz', 0)
                all_metrics.append(demod_metrics)

                # Keep best SNR result
                if demod_metrics.get('snr_est_db', -np.inf) > best_snr:
                    best_snr = demod_metrics['snr_est_db']
                    best_bits = bits

            except Exception as e:
                all_metrics.append({
                    'capture': capture_idx,
                    'error': str(e)
                })
                if verbose:
                    print(f"  [RX] Error: {e}")

        return best_bits, all_metrics, raw_rx


# ==============================================================================
# BER Test Functions
# ==============================================================================

def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> tuple:
    """Calculate Bit Error Rate."""
    min_len = min(len(tx_bits), len(rx_bits))
    if min_len == 0:
        return 1.0, 0, 0

    tx_bits = tx_bits[:min_len]
    rx_bits = rx_bits[:min_len]

    errors = np.sum(tx_bits != rx_bits)
    ber = errors / min_len

    return ber, errors, min_len


def run_ber_test(
    ip: str,
    freq: float,
    preset: str,
    num_frames: int,
    tx_gain: float = None,
    rx_gain: float = None
):
    """Run a BER test with the specified parameters."""

    print("\n" + "=" * 70)
    print("  PLUTO SDR OFDM BER TEST")
    print("=" * 70)

    # Get preset settings
    if preset in PRESETS:
        settings = PRESETS[preset].copy()
    else:
        settings = PRESETS["antenna_close"].copy()

    # Override with explicit parameters
    if tx_gain is not None:
        settings["tx_gain"] = tx_gain
    if rx_gain is not None:
        settings["rx_gain"] = rx_gain

    print(f"IP: {ip}")
    print(f"Frequency: {freq/1e6:.1f} MHz")
    print(f"Preset: {preset}")
    print(f"TX Gain: {settings['tx_gain']} dB, RX Gain: {settings['rx_gain']} dB")
    print(f"Number of frames: {num_frames}")
    print("-" * 70)

    # Configure
    pluto_cfg = PlutoConfig(
        ip=ip,
        fc=freq,
        tx_gain=settings["tx_gain"],
        rx_gain=settings["rx_gain"]
    )
    ofdm_cfg = OFDMConfig()

    # Create link
    link = PlutoOFDMLink(pluto_cfg, ofdm_cfg, tx_amp=settings["tx_amp"])

    if not link.connect():
        print("[Error] Failed to connect to PlutoSDR")
        return None

    try:
        total_errors = 0
        total_bits = 0
        successful_frames = 0
        snr_estimates = []

        print(f"\n{'Frame':<8} {'TX Bits':<10} {'RX Bits':<10} {'Errors':<10} {'BER':<12} {'SNR (dB)':<10} {'Status'}")
        print("-" * 80)

        for frame_idx in range(num_frames):
            # Generate random data
            np.random.seed(frame_idx + 1000)  # Reproducible but different each frame
            tx_bits = np.random.randint(0, 2, ofdm_cfg.bits_per_frame)

            # Transmit
            link.transmit_frame(tx_bits)
            time.sleep(0.3)  # Let TX stabilize

            # Receive and decode
            rx_bits, metrics, _ = link.receive_and_decode(num_captures=3)

            # Stop TX
            link.sdr.tx_destroy_buffer()

            # Check if we got valid bits
            if rx_bits is None or len(rx_bits) == 0:
                print(f"{frame_idx:<8} {len(tx_bits):<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} SYNC FAIL")
                continue

            # Calculate BER
            ber, errors, compared = calculate_ber(tx_bits, rx_bits)
            total_errors += errors
            total_bits += compared

            # Get SNR estimate
            snr = -np.inf
            for m in metrics:
                if m.get('sync_success') and 'snr_est_db' in m:
                    snr = m['snr_est_db']
                    snr_estimates.append(snr)
                    break

            successful_frames += 1
            status = "OK" if ber < 0.01 else "ERRORS"
            print(f"{frame_idx:<8} {len(tx_bits):<10} {len(rx_bits):<10} {errors:<10} {ber:<12.2e} {snr:<10.1f} {status}")

            time.sleep(0.1)

        # Summary
        print("-" * 80)
        print("\n" + "=" * 70)
        print("  BER TEST SUMMARY")
        print("=" * 70)

        if total_bits > 0:
            overall_ber = total_errors / total_bits
            print(f"Total Frames Sent:     {num_frames}")
            print(f"Successful Frames:     {successful_frames}")
            print(f"Frame Success Rate:    {successful_frames/num_frames*100:.1f}%")
            print(f"Total Bits Compared:   {total_bits}")
            print(f"Total Bit Errors:      {total_errors}")
            print(f"Overall BER:           {overall_ber:.2e}")

            if snr_estimates:
                print(f"Average SNR:           {np.mean(snr_estimates):.1f} dB")
                print(f"Min SNR:               {np.min(snr_estimates):.1f} dB")
                print(f"Max SNR:               {np.max(snr_estimates):.1f} dB")

            # Pass/Fail
            print("-" * 70)
            if overall_ber < 1e-3 and successful_frames >= num_frames * 0.8:
                print("\033[92m[PASSED]\033[0m BER < 1e-3 and >80% frames decoded")
            elif overall_ber < 1e-2:
                print("\033[93m[MARGINAL]\033[0m BER < 1e-2 but not ideal")
            else:
                print("\033[91m[FAILED]\033[0m BER too high or too many sync failures")
        else:
            print("\033[91m[FAILED]\033[0m No frames successfully decoded")

        print("=" * 70)

        return {
            'total_frames': num_frames,
            'successful_frames': successful_frames,
            'total_bits': total_bits,
            'total_errors': total_errors,
            'ber': overall_ber if total_bits > 0 else 1.0,
            'snr_mean': np.mean(snr_estimates) if snr_estimates else None
        }

    finally:
        link.disconnect()


def run_tx_gain_sweep(ip: str, freq: float, preset: str):
    """Sweep TX gain to find optimal operating point."""
    print("\n" + "=" * 70)
    print("  TX GAIN SWEEP TEST")
    print("=" * 70)

    settings = PRESETS.get(preset, PRESETS["antenna_close"]).copy()

    print(f"IP: {ip}")
    print(f"Frequency: {freq/1e6:.1f} MHz")
    print(f"RX Gain: {settings['rx_gain']} dB (fixed)")
    print("-" * 70)

    results = []
    tx_gains = [-30, -20, -10, -5, 0]

    print(f"\n{'TX Gain (dB)':<15} {'BER':<15} {'SNR (dB)':<15} {'Frames OK':<15}")
    print("-" * 60)

    for tx_gain in tx_gains:
        result = run_ber_test(
            ip=ip,
            freq=freq,
            preset=preset,
            num_frames=5,
            tx_gain=tx_gain
        )

        if result:
            results.append({
                'tx_gain': tx_gain,
                'ber': result['ber'],
                'snr': result.get('snr_mean'),
                'frames_ok': result['successful_frames']
            })

            snr_str = f"{result.get('snr_mean', 0):.1f}" if result.get('snr_mean') else "N/A"
            print(f"{tx_gain:<15} {result['ber']:<15.2e} {snr_str:<15} {result['successful_frames']:<15}")

    # Find best TX gain
    print("-" * 60)
    if results:
        best = min(results, key=lambda x: x['ber'])
        print(f"\nBest TX Gain: {best['tx_gain']} dB (BER: {best['ber']:.2e})")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PlutoSDR OFDM BER Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic BER test at 2.3 GHz (avoids WiFi)
  python pluto_ofdm_ber_test.py --ip ip:192.168.2.2 --freq 2.3e9

  # Test with antenna preset and more frames
  python pluto_ofdm_ber_test.py --ip ip:192.168.2.2 --preset antenna_close --frames 20

  # TX gain sweep to find optimal settings
  python pluto_ofdm_ber_test.py --ip ip:192.168.2.2 --tx_gain_sweep

  # Custom gains
  python pluto_ofdm_ber_test.py --ip ip:192.168.2.2 --tx_gain 0 --rx_gain 60
        """
    )
    parser.add_argument("--ip", default="ip:192.168.2.2", help="PlutoSDR IP address")
    parser.add_argument("--freq", type=float, default=2.3e9, help="Center frequency (Hz)")
    parser.add_argument("--preset", default="antenna_close",
                        choices=["cable_30db", "cable_direct", "antenna_close", "antenna_far"],
                        help="Test preset")
    parser.add_argument("--tx_gain", type=float, default=None, help="Override TX gain (dB)")
    parser.add_argument("--rx_gain", type=float, default=None, help="Override RX gain (dB)")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to test")
    parser.add_argument("--tx_gain_sweep", action="store_true", help="Run TX gain sweep")

    args = parser.parse_args()

    if not ADI_AVAILABLE:
        print("[Error] pyadi-iio not available. Cannot run test.")
        sys.exit(1)

    if args.tx_gain_sweep:
        run_tx_gain_sweep(args.ip, args.freq, args.preset)
    else:
        run_ber_test(
            ip=args.ip,
            freq=args.freq,
            preset=args.preset,
            num_frames=args.frames,
            tx_gain=args.tx_gain,
            rx_gain=args.rx_gain
        )


if __name__ == "__main__":
    main()
