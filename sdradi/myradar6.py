"""
myradar6.py - Radar Device Driver with Pluto/AD9361 Loopback Testing

This module extends myradar5.py to support local testing of AD9361/Pluto SDR devices
via loopback (TX→30dB attenuator→RX) before using the full CN0566 phaser system.

Device Modes:
- SIMULATION: No hardware, simulated data for development
- PLUTO_DIGITAL_LOOPBACK: Internal digital loopback (BIST mode)
- PLUTO_RF_LOOPBACK: RF loopback via external attenuator
- PHASER: Full CN0566 phaser system (from myradar5)

Usage:
    # Simulation mode (no hardware)
    python myradar6.py --mode simulation
    
    # Digital loopback test
    python myradar6.py --mode digital_loopback --ip ip:192.168.2.2
    
    # RF loopback test (requires TX-30dB-RX cable)
    python myradar6.py --mode rf_loopback --ip ip:192.168.2.2
    
    # Comprehensive test
    python myradar6.py --mode comprehensive --ip ip:192.168.2.2

Author: AI-assisted development
Date: 2026-01-31
"""

import time
import sys
import os
import argparse
import numpy as np
from datetime import datetime
from timeit import default_timer as timer
from enum import Enum
import logging
from scipy import signal

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeviceMode(Enum):
    """Device operation modes."""
    SIMULATION = "simulation"
    PLUTO_DIGITAL_LOOPBACK = "digital_loopback"
    PLUTO_RF_LOOPBACK = "rf_loopback"
    PHASER = "phaser"


class TestResult:
    """Container for test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.metrics = {}
        self.message = ""
    
    def __str__(self):
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        metrics_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in self.metrics.items())
        return f"{self.name}: {status} | {metrics_str} | {self.message}"


def generate_complex_sinusoid(fs: int, fc: int = 100000, N: int = 1024 * 16) -> np.ndarray:
    """
    Generate a complex sinusoid test signal.
    
    Args:
        fs: Sample rate in Hz
        fc: Tone frequency in Hz
        N: Number of samples
    
    Returns:
        Complex IQ samples scaled for SDR (2^14)
    """
    ts = 1.0 / fs
    t = np.arange(N) * ts
    i = np.cos(2 * np.pi * fc * t) * (2**14)
    q = np.sin(2 * np.pi * fc * t) * (2**14)
    return (i + 1j * q).astype(np.complex64)


def calculate_snr(rx_data: np.ndarray, signal_freq: float, fs: float) -> dict:
    """
    Calculate SNR and related metrics from received data.
    
    Args:
        rx_data: Received complex samples
        signal_freq: Expected signal frequency in Hz
        fs: Sample rate in Hz
    
    Returns:
        Dictionary with SNR, peak amplitude, peak frequency, etc.
    """
    # Time domain metrics
    peak_amplitude = np.max(np.abs(rx_data))
    mean_amplitude = np.mean(np.abs(rx_data))
    
    # Frequency domain analysis
    N = len(rx_data)
    fft_data = np.fft.fftshift(np.fft.fft(rx_data))
    fft_mag = np.abs(fft_data)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    
    # Find peak
    peak_idx = np.argmax(fft_mag)
    peak_freq = freqs[peak_idx]
    peak_power_db = 20 * np.log10(fft_mag[peak_idx] + 1e-12)
    
    # Estimate noise (median of spectrum)
    noise_floor_db = 20 * np.log10(np.median(fft_mag) + 1e-12)
    
    # SNR estimation
    snr_db = peak_power_db - noise_floor_db
    
    return {
        "peak_amplitude": peak_amplitude,
        "mean_amplitude": mean_amplitude,
        "peak_freq_hz": peak_freq,
        "peak_power_db": peak_power_db,
        "noise_floor_db": noise_floor_db,
        "snr_db": snr_db,
    }


def calculate_correlation(tx_data: np.ndarray, rx_data: np.ndarray) -> dict:
    """
    Calculate cross-correlation between TX and RX signals.
    
    Returns:
        Dictionary with correlation coefficient, peak offset, etc.
    """
    # Normalize
    tx_norm = tx_data / (np.max(np.abs(tx_data)) + 1e-12)
    rx_norm = rx_data / (np.max(np.abs(rx_data)) + 1e-12)
    
    # Cross-correlation
    corr_real = np.correlate(np.real(rx_norm), np.real(tx_norm), mode='valid')
    corr_imag = np.correlate(np.imag(rx_norm), np.imag(tx_norm), mode='valid')
    corr = np.abs(corr_real + 1j * corr_imag)
    
    peak_corr = np.max(corr)
    peak_offset = np.argmax(corr)
    
    # Pearson correlation on aligned data
    if len(rx_norm) >= len(tx_norm):
        aligned_rx = rx_norm[peak_offset:peak_offset + len(tx_norm)]
        if len(aligned_rx) == len(tx_norm):
            pearson_r = np.corrcoef(np.abs(tx_norm), np.abs(aligned_rx))[0, 1]
        else:
            pearson_r = 0.0
    else:
        pearson_r = 0.0
    
    return {
        "peak_correlation": float(peak_corr),
        "peak_offset_samples": int(peak_offset),
        "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
    }


class PlutoRadarDevice:
    """
    Pluto/AD9361 Radar Device for loopback testing.
    
    This class provides a test interface for AD9361-based SDR devices
    before moving to the full CN0566 phaser system.
    """
    
    def __init__(self,
                 sdr_ip: str = "ip:192.168.2.2",
                 mode: DeviceMode = DeviceMode.PLUTO_RF_LOOPBACK,
                 sample_rate: int = 3000000,
                 center_freq: int = 2400000000,
                 rx_buffer_size: int = 1024 * 16,
                 bandwidth: int = 3000000,
                 tx_gain: int = -10,
                 rx_gain: int = 30,
                 signal_freq: int = 100000,
                 driver_type: str = "pluto"):
        """
        Initialize the Pluto/AD9361 device.
        
        Args:
            sdr_ip: IP address of the SDR device
            mode: Operating mode (simulation, digital_loopback, rf_loopback, phaser)
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            rx_buffer_size: RX buffer size in samples
            bandwidth: RF bandwidth in Hz
            tx_gain: TX gain in dB (negative attenuation)
            rx_gain: RX gain in dB
            signal_freq: Test signal frequency offset in Hz
            driver_type: SDR driver type ("pluto" or "ad9361")
        """
        self.sdr_ip = sdr_ip
        self.mode = mode
        self.sample_rate = int(sample_rate)
        self.center_freq = int(center_freq)
        self.rx_buffer_size = int(rx_buffer_size)
        self.bandwidth = int(bandwidth)
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.signal_freq = signal_freq
        self.driver_type = driver_type
        
        self.sdr = None
        self.phy = None
        self.ctx = None
        self.connected = False
        
        # Data storage
        self.tx_data = None
        self.rx_data = None
        self.all_rx_data = np.empty(0, dtype=np.complex64)
        
        # Initialize based on mode
        if mode == DeviceMode.SIMULATION:
            logger.info("Initializing in SIMULATION mode (no hardware)")
            self._init_simulation()
        else:
            self._init_hardware()
    
    def _init_simulation(self):
        """Initialize simulation mode."""
        logger.info("Simulation mode: Generating synthetic test data")
        self.connected = True
    
    def _init_hardware(self):
        """Initialize hardware connection."""
        try:
            import adi
            logger.info(f"Connecting to SDR at {self.sdr_ip} with driver={self.driver_type}...")
            
            # Select driver based on driver_type parameter
            if self.driver_type == "ad9361":
                try:
                    self.sdr = adi.ad9361(uri=self.sdr_ip)
                    logger.info("Using adi.ad9361 driver (2R2T support)")
                except Exception as e:
                    logger.warning(f"ad9361 driver failed: {e}, falling back to Pluto")
                    self.sdr = adi.Pluto(uri=self.sdr_ip)
                    logger.info("Using adi.Pluto driver (fallback)")
            else:
                # Default: Use Pluto driver (more stable for single-channel operation)
                self.sdr = adi.Pluto(uri=self.sdr_ip)
                logger.info("Using adi.Pluto driver")
            
            # Configure SDR
            self._configure_sdr()
            
            # Get IIO context for advanced features
            self.ctx = self.sdr.ctx
            self.phy = self.ctx.find_device("ad9361-phy")
            
            self.connected = True
            logger.info(f"Successfully connected to SDR at {self.sdr_ip}")
            
        except ImportError:
            logger.error("ADI library not installed. Install with: pip install pyadi-iio")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to SDR: {e}")
            raise
    
    def _configure_sdr(self):
        """Configure SDR parameters."""
        # CRITICAL: Configure channels FIRST to avoid segfault (ad9361 driver only)
        # Pluto driver doesn't have enabled_channels properties
        if hasattr(self.sdr, 'tx_enabled_channels'):
            self.sdr.tx_enabled_channels = [0]
        if hasattr(self.sdr, 'rx_enabled_channels'):
            self.sdr.rx_enabled_channels = [0]
        
        # Destroy any existing buffers before configuration
        try:
            self.sdr.tx_destroy_buffer()
        except:
            pass
        try:
            self.sdr.rx_destroy_buffer()
        except:
            pass
        
        # Now configure parameters
        self.sdr.sample_rate = self.sample_rate
        self.sdr.tx_lo = self.center_freq
        self.sdr.rx_lo = self.center_freq
        self.sdr.tx_rf_bandwidth = self.bandwidth
        self.sdr.rx_rf_bandwidth = self.bandwidth
        self.sdr.rx_buffer_size = self.rx_buffer_size
        
        # Gain configuration
        self.sdr.tx_hardwaregain_chan0 = self.tx_gain
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = self.rx_gain
        
        # Set kernel buffers for robustness
        if hasattr(self.sdr, "_rxadc") and hasattr(self.sdr._rxadc, "set_kernel_buffers_count"):
            self.sdr._rxadc.set_kernel_buffers_count(4)
        
        # Increase context timeout
        if hasattr(self.sdr, "_ctx"):
            self.sdr._ctx.set_timeout(30000)
        
        logger.info(f"SDR configured: SR={self.sample_rate/1e6}MHz, FC={self.center_freq/1e9}GHz, "
                    f"BW={self.bandwidth/1e6}MHz, TX_GAIN={self.tx_gain}dB, RX_GAIN={self.rx_gain}dB")
    
    def enable_digital_loopback(self, enable: bool = True):
        """
        Enable/disable digital loopback (BIST mode).
        
        The AD9361 has built-in digital loopback for testing.
        Mode 1 = Digital TX → Digital RX
        """
        if self.mode == DeviceMode.SIMULATION:
            logger.info(f"Simulation: Digital loopback {'enabled' if enable else 'disabled'}")
            return True
        
        try:
            if self.phy:
                loopback_value = '1' if enable else '0'
                self.phy.debug_attrs['loopback'].value = loopback_value
                logger.info(f"Digital loopback {'enabled' if enable else 'disabled'}")
                return True
        except Exception as e:
            logger.error(f"Failed to set digital loopback: {e}")
            return False
    
    def generate_tx_signal(self, signal_type: str = "sinusoid") -> np.ndarray:
        """
        Generate TX test signal.
        
        Args:
            signal_type: Type of signal ("sinusoid", "chirp", "random")
        
        Returns:
            Complex IQ samples
        """
        if signal_type == "sinusoid":
            self.tx_data = generate_complex_sinusoid(
                self.sample_rate, 
                self.signal_freq, 
                self.rx_buffer_size
            )
        elif signal_type == "chirp":
            # Linear chirp
            t = np.arange(self.rx_buffer_size) / self.sample_rate
            f0 = self.signal_freq / 2
            f1 = self.signal_freq * 2
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1]))
            self.tx_data = (np.exp(1j * phase) * (2**14)).astype(np.complex64)
        elif signal_type == "random":
            # Random QPSK-like
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], self.rx_buffer_size)
            self.tx_data = (symbols * (2**14) * 0.5).astype(np.complex64)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        logger.info(f"Generated {signal_type} TX signal: {len(self.tx_data)} samples")
        return self.tx_data
    
    def transmit(self, cyclic: bool = True):
        """
        Transmit the TX data.
        
        Args:
            cyclic: If True, use cyclic buffer for continuous transmission
        """
        if self.tx_data is None:
            self.generate_tx_signal()
        
        if self.mode == DeviceMode.SIMULATION:
            logger.info("Simulation: TX signal queued")
            return
        
        try:
            self.sdr.tx_destroy_buffer()
            self.sdr.tx_cyclic_buffer = cyclic
            self.sdr.tx(self.tx_data)
            logger.info(f"Transmitting {len(self.tx_data)} samples (cyclic={cyclic})")
        except Exception as e:
            logger.error(f"TX failed: {e}")
            raise
    
    def receive(self, flush_count: int = 3) -> np.ndarray:
        """
        Receive data from the SDR.
        
        Args:
            flush_count: Number of buffers to flush before capturing
        
        Returns:
            Complex IQ samples
        """
        if self.mode == DeviceMode.SIMULATION:
            # Simulate received signal with noise
            if self.tx_data is not None:
                # Add delay, attenuation, and noise
                delay = np.random.randint(10, 100)
                attenuation = 0.1 if self.mode == DeviceMode.PLUTO_RF_LOOPBACK else 1.0
                noise = (np.random.randn(len(self.tx_data)) + 
                         1j * np.random.randn(len(self.tx_data))) * 100
                
                self.rx_data = np.roll(self.tx_data * attenuation, delay) + noise
            else:
                self.rx_data = (np.random.randn(self.rx_buffer_size) + 
                                1j * np.random.randn(self.rx_buffer_size)).astype(np.complex64) * 100
            
            logger.info("Simulation: Generated synthetic RX data")
            return self.rx_data
        
        try:
            # Flush stale buffers
            for _ in range(flush_count):
                self.sdr.rx()
            
            self.rx_data = self.sdr.rx()
            logger.info(f"Received {len(self.rx_data)} samples")
            
            # Store for later analysis
            self.all_rx_data = np.concatenate([self.all_rx_data, self.rx_data])
            
            return self.rx_data
            
        except Exception as e:
            logger.error(f"RX failed: {e}")
            raise
    
    def stop(self):
        """Stop TX/RX and clean up."""
        if self.mode == DeviceMode.SIMULATION:
            logger.info("Simulation: Stopped")
            return
        
        try:
            if self.sdr:
                self.sdr.tx_destroy_buffer()
                self.sdr.rx_destroy_buffer()
            logger.info("SDR stopped")
        except Exception as e:
            logger.warning(f"Error stopping SDR: {e}")
    
    # ===== TEST METHODS =====
    
    def run_connectivity_test(self) -> TestResult:
        """Test basic device connectivity and parameters."""
        result = TestResult("Connectivity Test")
        
        if self.mode == DeviceMode.SIMULATION:
            result.passed = True
            result.message = "Simulation mode - always passes"
            result.metrics = {"sample_rate": self.sample_rate, "center_freq": self.center_freq}
            return result
        
        try:
            # Read back parameters
            actual_sr = self.sdr.sample_rate
            actual_fc = self.sdr.tx_lo
            
            result.metrics = {
                "sample_rate": actual_sr,
                "center_freq": actual_fc,
                "connected": self.connected,
            }
            
            # Verify parameters match
            sr_match = abs(actual_sr - self.sample_rate) < 1000
            fc_match = abs(actual_fc - self.center_freq) < 1000
            
            result.passed = sr_match and fc_match and self.connected
            result.message = "Parameters verified" if result.passed else "Parameter mismatch"
            
        except Exception as e:
            result.passed = False
            result.message = f"Error: {e}"
        
        return result
    
    def run_digital_loopback_test(self) -> TestResult:
        """Run digital loopback test using AD9361 BIST mode."""
        result = TestResult("Digital Loopback Test")
        
        try:
            # Enable digital loopback
            if not self.enable_digital_loopback(True):
                result.passed = False
                result.message = "Failed to enable digital loopback"
                return result
            
            time.sleep(0.2)
            
            # Generate and transmit
            self.generate_tx_signal("sinusoid")
            self.transmit(cyclic=True)
            time.sleep(0.5)
            
            # Receive
            rx_data = self.receive(flush_count=5)
            
            # Analyze
            snr_metrics = calculate_snr(rx_data, self.signal_freq, self.sample_rate)
            corr_metrics = calculate_correlation(self.tx_data[:len(rx_data)], rx_data)
            
            result.metrics = {
                "snr_db": snr_metrics["snr_db"],
                "peak_amplitude": snr_metrics["peak_amplitude"],
                "correlation": corr_metrics["pearson_r"],
            }
            
            # Pass criteria: SNR > 25dB and correlation > 0.7
            result.passed = snr_metrics["snr_db"] > 25 and corr_metrics["pearson_r"] > 0.7
            result.message = "Digital path verified" if result.passed else "Low SNR or correlation"
            
            # Disable loopback
            self.enable_digital_loopback(False)
            self.stop()
            
        except Exception as e:
            result.passed = False
            result.message = f"Error: {e}"
            try:
                self.enable_digital_loopback(False)
                self.stop()
            except:
                pass
        
        return result
    
    def run_rf_loopback_test(self, expected_attenuation_db: float = 30.0) -> TestResult:
        """
        Run RF loopback test with external attenuator.
        
        Args:
            expected_attenuation_db: Expected attenuation in the loopback path (default 30dB)
        """
        result = TestResult("RF Loopback Test")
        
        try:
            # Make sure digital loopback is disabled
            self.enable_digital_loopback(False)
            time.sleep(0.2)
            
            # Generate and transmit
            self.generate_tx_signal("sinusoid")
            self.transmit(cyclic=True)
            time.sleep(0.5)
            
            # Receive
            rx_data = self.receive(flush_count=5)
            
            # Analyze
            snr_metrics = calculate_snr(rx_data, self.signal_freq, self.sample_rate)
            corr_metrics = calculate_correlation(self.tx_data[:len(rx_data)], rx_data)
            
            # Calculate approximate path loss
            tx_power = 20 * np.log10(np.max(np.abs(self.tx_data)) + 1e-12)
            rx_power = 20 * np.log10(np.max(np.abs(rx_data)) + 1e-12)
            measured_loss_db = tx_power - rx_power
            
            result.metrics = {
                "snr_db": snr_metrics["snr_db"],
                "peak_amplitude": snr_metrics["peak_amplitude"],
                "correlation": corr_metrics["pearson_r"],
                "path_loss_db": measured_loss_db,
                "peak_freq_hz": snr_metrics["peak_freq_hz"],
            }
            
            # Pass criteria for RF loopback with 30dB attenuator:
            # - SNR > 10dB (signal present above noise)
            # - Detectable signal (peak amplitude > noise threshold - lowered for attenuated signals)
            # - Peak at expected frequency (within 20% tolerance)
            snr_ok = snr_metrics["snr_db"] > 10
            signal_detected = snr_metrics["peak_amplitude"] > 50  # Lowered for attenuated signals
            freq_tolerance = 0.2 * self.signal_freq  # 20% tolerance
            freq_ok = abs(abs(snr_metrics["peak_freq_hz"]) - self.signal_freq) < freq_tolerance
            
            result.passed = snr_ok and signal_detected and freq_ok
            
            if result.passed:
                result.message = f"RF path verified (loss={measured_loss_db:.1f}dB)"
            else:
                issues = []
                if not snr_ok:
                    issues.append(f"low SNR ({snr_metrics['snr_db']:.1f}dB)")
                if not signal_detected:
                    issues.append("no signal detected")
                if not freq_ok:
                    issues.append(f"wrong freq (got {snr_metrics['peak_freq_hz']/1000:.1f}kHz, expect {self.signal_freq/1000:.0f}kHz)")
                result.message = f"Issues: {', '.join(issues)}"
            
            self.stop()
            
        except Exception as e:
            result.passed = False
            result.message = f"Error: {e}"
            try:
                self.stop()
            except:
                pass
        
        return result
    
    def run_spectrum_sweep_test(self, freqs: list = None) -> TestResult:
        """
        Test multiple frequencies to verify RF path across bandwidth.
        
        Args:
            freqs: List of test frequencies (Hz). Default: 50k, 100k, 200k, 500k
        """
        result = TestResult("Spectrum Sweep Test")
        
        if freqs is None:
            freqs = [50000, 100000, 200000, 500000]
        
        try:
            results_per_freq = {}
            all_passed = True
            
            for freq in freqs:
                self.signal_freq = freq
                self.generate_tx_signal("sinusoid")
                self.transmit(cyclic=True)
                time.sleep(0.3)
                
                rx_data = self.receive(flush_count=3)
                snr_metrics = calculate_snr(rx_data, freq, self.sample_rate)
                
                freq_passed = snr_metrics["snr_db"] > 8
                results_per_freq[f"{freq/1000:.0f}kHz"] = {
                    "snr": snr_metrics["snr_db"],
                    "passed": freq_passed
                }
                
                if not freq_passed:
                    all_passed = False
                
                self.stop()
                time.sleep(0.1)
            
            result.metrics = results_per_freq
            result.passed = all_passed
            result.message = "All frequencies OK" if all_passed else "Some frequencies failed"
            
        except Exception as e:
            result.passed = False
            result.message = f"Error: {e}"
        
        return result
    
    def run_comprehensive_test(self) -> list:
        """Run all tests and return results."""
        results = []
        
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE PLUTO/AD9361 TEST SUITE")
        logger.info("=" * 60)
        
        # 1. Connectivity
        logger.info("\n[1/4] Running Connectivity Test...")
        results.append(self.run_connectivity_test())
        logger.info(str(results[-1]))
        
        # 2. Digital Loopback
        if self.mode != DeviceMode.SIMULATION:
            logger.info("\n[2/4] Running Digital Loopback Test...")
            results.append(self.run_digital_loopback_test())
            logger.info(str(results[-1]))
        
        # 3. RF Loopback
        logger.info("\n[3/4] Running RF Loopback Test...")
        results.append(self.run_rf_loopback_test())
        logger.info(str(results[-1]))
        
        # 4. Spectrum Sweep
        logger.info("\n[4/4] Running Spectrum Sweep Test...")
        results.append(self.run_spectrum_sweep_test())
        logger.info(str(results[-1]))
        
        # Summary
        logger.info("\n" + "=" * 60)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
        logger.info("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Pluto/AD9361 Radar Device Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python myradar6.py --mode simulation
  python myradar6.py --mode digital_loopback --ip ip:192.168.2.2
  python myradar6.py --mode rf_loopback --ip ip:192.168.2.2
  python myradar6.py --mode comprehensive --ip ip:192.168.2.2
        """
    )
    
    parser.add_argument("--mode", type=str, default="rf_loopback",
                        choices=["simulation", "digital_loopback", "rf_loopback", "comprehensive"],
                        help="Test mode to run")
    parser.add_argument("--ip", type=str, default="ip:192.168.2.2",
                        help="SDR IP address (default: ip:192.168.2.2)")
    parser.add_argument("--sample_rate", type=float, default=3e6,
                        help="Sample rate in Hz (default: 3e6)")
    parser.add_argument("--center_freq", type=float, default=2.4e9,
                        help="Center frequency in Hz (default: 2.4e9)")
    parser.add_argument("--tx_gain", type=int, default=-10,
                        help="TX gain in dB (default: -10)")
    parser.add_argument("--rx_gain", type=int, default=30,
                        help="RX gain in dB (default: 30)")
    parser.add_argument("--signal_freq", type=float, default=100e3,
                        help="Test signal frequency in Hz (default: 100e3)")
    parser.add_argument("--attenuation", type=float, default=30.0,
                        help="Expected loopback attenuation in dB (default: 30)")
    parser.add_argument("--driver", type=str, default="pluto",
                        choices=["pluto", "ad9361"],
                        help="SDR driver to use (default: pluto)")
    
    args = parser.parse_args()
    
    # Map mode string to enum
    mode_map = {
        "simulation": DeviceMode.SIMULATION,
        "digital_loopback": DeviceMode.PLUTO_DIGITAL_LOOPBACK,
        "rf_loopback": DeviceMode.PLUTO_RF_LOOPBACK,
        "comprehensive": DeviceMode.PLUTO_RF_LOOPBACK,  # Will run all tests
    }
    
    device_mode = mode_map.get(args.mode, DeviceMode.SIMULATION)
    
    logger.info(f"Initializing PlutoRadarDevice...")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  IP: {args.ip}")
    logger.info(f"  Sample Rate: {args.sample_rate/1e6} MHz")
    logger.info(f"  Center Freq: {args.center_freq/1e9} GHz")
    logger.info(f"  Driver: {args.driver}")
    
    try:
        device = PlutoRadarDevice(
            sdr_ip=args.ip,
            mode=device_mode,
            sample_rate=int(args.sample_rate),
            center_freq=int(args.center_freq),
            tx_gain=args.tx_gain,
            rx_gain=args.rx_gain,
            signal_freq=int(args.signal_freq),
            driver_type=args.driver,
        )
        
        if args.mode == "comprehensive":
            results = device.run_comprehensive_test()
        elif args.mode == "simulation":
            result = device.run_rf_loopback_test()
            logger.info(str(result))
        elif args.mode == "digital_loopback":
            result = device.run_digital_loopback_test()
            logger.info(str(result))
        elif args.mode == "rf_loopback":
            result = device.run_rf_loopback_test(expected_attenuation_db=args.attenuation)
            logger.info(str(result))
        
        device.stop()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
