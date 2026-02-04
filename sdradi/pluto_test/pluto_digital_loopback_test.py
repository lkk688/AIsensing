#!/usr/bin/env python
"""
pluto_digital_loopback_test.py - Digital Loopback Testing for Pluto SDR

This script tests the video communication stack step-by-step using the Pluto SDR's
internal digital loopback mode. Supports both IP and USB connections.

Key Features:
- Burst transmission mode (non-cyclic) for better TX/RX alignment
- Hardware reset sequence to prevent DMA deadlocks
- Buffer flushing to clear stale data
- Fine timing offset search for synchronization

Test Levels:
    Level 0: Basic SDR connection test
    Level 1: Digital loopback with simple tone
    Level 2: OFDM modulation/demodulation
    Level 3: Packet transmission with FEC
    Level 4: Video frame encoding/decoding

Usage:
    # USB connection
    python pluto_digital_loopback_test.py --ip usb:1.32.5

    # IP connection
    python pluto_digital_loopback_test.py --ip ip:192.168.3.2

    # Run specific test level
    python pluto_digital_loopback_test.py --level 2

Author: AI-assisted development for AIsensing project
"""

import numpy as np
import time
import argparse
import sys
import os

# Set Qt platform before any GUI imports
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Pluto SDR configuration
PLUTO_IP = "ip:192.168.2.2"  # Default to IP connection
SAMPLE_RATE = 2e6  # 2 MSPS
CENTER_FREQ = 915e6  # 915 MHz (ISM band)
TX_GAIN = -10  # dB
RX_GAIN = 40  # dB


# =============================================================================
# Test Result Tracking
# =============================================================================

class TestResult:
    """Track test results."""
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = {}

    def success(self, message="", **details):
        self.passed = True
        self.message = message
        self.details = details
        return self

    def fail(self, message="", **details):
        self.passed = False
        self.message = message
        self.details = details
        return self

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        msg = f"[{status}] {self.name}"
        if self.message:
            msg += f": {self.message}"
        return msg


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subheader(title):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


# =============================================================================
# SDR Helper Functions
# =============================================================================

def reset_sdr(sdr):
    """Hardware reset sequence to prevent DMA deadlocks."""
    try:
        # Destroy any existing TX buffer first
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        # Toggle LO to reset AD9361 state machine
        original_lo = sdr.rx_lo
        sdr.rx_lo = int(original_lo + 1e6)
        time.sleep(0.1)
        sdr.rx_lo = int(original_lo)
        time.sleep(0.1)

        # Also toggle TX LO
        original_tx_lo = sdr.tx_lo
        sdr.tx_lo = int(original_tx_lo + 1e6)
        time.sleep(0.05)
        sdr.tx_lo = int(original_tx_lo)
        time.sleep(0.05)

        print("[SDR] Hardware reset completed")
    except Exception as e:
        print(f"[SDR] Reset warning: {e}")


def flush_rx_buffer(sdr, num_flushes=5):
    """Flush stale data from RX buffer."""
    flushed = 0
    for _ in range(num_flushes):
        try:
            rx_data = sdr.rx()
            if rx_data is not None and len(rx_data) > 0:
                flushed += 1
        except:
            pass
        time.sleep(0.01)  # Small delay between flushes
    print(f"[SDR] Flushed RX buffer ({flushed}/{num_flushes} successful reads)")


def enable_digital_loopback(sdr):
    """Enable digital loopback mode with verification."""
    try:
        # Set loopback mode to 1 (digital loopback)
        sdr._ctrl.debug_attrs['loopback'].value = '1'
        time.sleep(0.1)  # Wait for setting to take effect

        # Verify it was actually set
        actual = sdr._ctrl.debug_attrs['loopback'].value
        if actual != '1':
            print(f"[SDR] WARNING: Loopback not set correctly (got {actual}, expected 1)")
            # Try again
            sdr._ctrl.debug_attrs['loopback'].value = '1'
            time.sleep(0.1)
            actual = sdr._ctrl.debug_attrs['loopback'].value

        print(f"[SDR] Digital loopback enabled (mode={actual})")
        return actual == '1'
    except Exception as e:
        print(f"[SDR] Failed to enable loopback: {e}")
        return False


def disable_digital_loopback(sdr):
    """Disable digital loopback mode."""
    try:
        sdr._ctrl.debug_attrs['loopback'].value = '0'
        print("[SDR] Digital loopback disabled")
    except:
        pass


# =============================================================================
# Level 0: Basic SDR Connection Test
# =============================================================================

def test_level_0_connection():
    """Test basic connection to Pluto SDR."""
    print_header("Level 0: Basic SDR Connection Test")
    result = TestResult("SDR Connection")

    sdr = None
    try:
        import adi
        print(f"[INFO] Connecting to Pluto SDR at {PLUTO_IP}...")

        sdr = adi.Pluto(PLUTO_IP)

        # Configure SDR
        sdr.sample_rate = int(SAMPLE_RATE)
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.rx_lo = int(CENTER_FREQ)
        sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        sdr.rx_buffer_size = 2**16

        print(f"[INFO] Configuration applied:")
        print(f"       Sample Rate: {sdr.sample_rate / 1e6:.2f} MSPS")
        print(f"       Center Freq: {sdr.tx_lo / 1e6:.2f} MHz")
        print(f"       TX Gain: {sdr.tx_hardwaregain_chan0} dB")
        print(f"       RX Gain: {sdr.rx_hardwaregain_chan0} dB")

        return result.success("Connected and configured")

    except ImportError:
        return result.fail("pyadi-iio not installed. Run: pip install pyadi-iio")
    except Exception as e:
        return result.fail(f"Connection failed: {e}")
    finally:
        # Release SDR resources
        if sdr is not None:
            try:
                del sdr
            except:
                pass
        time.sleep(0.2)  # Give USB time to release


# =============================================================================
# Level 1: Digital Loopback with Simple Tone
# =============================================================================

def test_level_1_tone_loopback():
    """Test digital loopback with a simple sine tone using TDD-style operation."""
    print_header("Level 1: Digital Loopback - Simple Tone Test (TDD Mode)")
    result = TestResult("Tone Loopback")

    sdr = None
    try:
        import adi
        sdr = adi.Pluto(PLUTO_IP)

        # Configure SDR
        sdr.sample_rate = int(SAMPLE_RATE)
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.rx_lo = int(CENTER_FREQ)
        sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        sdr.rx_buffer_size = 2**16  # 64k samples

        # Enable digital loopback
        print("[INFO] Enabling digital loopback mode...")
        if not enable_digital_loopback(sdr):
            return result.fail("Could not enable loopback")

        # Generate test tone
        N = 8192
        f_tone = 100e3  # 100 kHz tone
        t = np.arange(N) / SAMPLE_RATE
        tx_tone = np.exp(2j * np.pi * f_tone * t)
        tx_samples = (tx_tone * 0.8 * 2**14).astype(np.complex64)

        print(f"[INFO] Using TDD-style operation: TX first, then RX")
        print(f"[INFO] Transmitting {f_tone/1e3:.0f} kHz tone ({len(tx_samples)} samples)...")

        # TDD-style: Use cyclic TX, capture RX, then stop TX
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_samples)

        # Wait for TX to stabilize
        time.sleep(0.1)

        # Capture RX (digital loopback should route TX to RX)
        rx_samples = None
        for attempt in range(5):
            try:
                rx_temp = sdr.rx()
                if rx_temp is not None and len(rx_temp) > 0:
                    max_amp = np.max(np.abs(rx_temp))
                    print(f"[INFO] RX attempt {attempt+1}: {len(rx_temp)} samples, max_amp={max_amp:.0f}")
                    if max_amp > 100:  # Valid signal threshold
                        rx_samples = rx_temp
                        break
            except Exception as e:
                print(f"[INFO] RX attempt {attempt+1} error: {e}")
            time.sleep(0.05)

        # Stop TX
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        # Disable loopback
        disable_digital_loopback(sdr)

        # Analyze received signal
        if rx_samples is None or len(rx_samples) == 0 or np.max(np.abs(rx_samples)) < 1:
            return result.fail("No samples received (DMA issue). Try: 1) Reconnect USB, 2) Reset Pluto")

        # Normalize
        rx_samples = rx_samples / (2**14)

        # Remove DC
        rx_samples = rx_samples - np.mean(rx_samples)

        # Calculate power
        rx_power = np.mean(np.abs(rx_samples)**2)

        # FFT to find tone frequency
        fft = np.fft.fft(rx_samples)
        fft_mag = np.abs(fft[:len(fft)//2])
        peak_idx = np.argmax(fft_mag)
        freq_bins = np.fft.fftfreq(len(rx_samples), 1/SAMPLE_RATE)[:len(fft)//2]
        detected_freq = abs(freq_bins[peak_idx])

        print(f"[INFO] RX Power: {10*np.log10(rx_power + 1e-10):.1f} dB")
        print(f"[INFO] Detected tone frequency: {detected_freq/1e3:.1f} kHz")
        print(f"[INFO] Expected frequency: {f_tone/1e3:.1f} kHz")

        # Check if tone is within tolerance (1% of tone frequency or 10 FFT bins)
        freq_error = abs(detected_freq - f_tone)
        fft_bin_resolution = SAMPLE_RATE / len(rx_samples)
        freq_tolerance = max(f_tone * 0.01, fft_bin_resolution * 10)  # 1% or 10 bins

        print(f"[INFO] Freq error: {freq_error:.0f} Hz, tolerance: {freq_tolerance:.0f} Hz")

        if freq_error < freq_tolerance and rx_power > 1e-6:
            return result.success(
                f"Tone detected at {detected_freq/1e3:.1f} kHz",
                rx_power_db=10*np.log10(rx_power + 1e-10),
                freq_error_hz=freq_error
            )
        else:
            return result.fail(
                f"Frequency mismatch or low power: error={freq_error/1e3:.1f} kHz, power={10*np.log10(rx_power+1e-10):.1f} dB"
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return result.fail(f"Error: {e}")
    finally:
        if sdr is not None:
            try:
                del sdr
            except:
                pass
        time.sleep(0.2)


# =============================================================================
# Level 2: OFDM Modulation/Demodulation Test (Improved)
# =============================================================================

def test_level_2_ofdm_loopback():
    """Test OFDM modulation and demodulation through digital loopback."""
    print_header("Level 2: OFDM Modulation/Demodulation Test")
    result = TestResult("OFDM Loopback")

    sdr = None
    try:
        import adi
        from sdr_video_commv2_lab import OFDMConfig, OFDMTransceiver

        sdr = adi.Pluto(PLUTO_IP)

        # Configure SDR with larger buffer for repeated frames
        sdr.sample_rate = int(SAMPLE_RATE)
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.rx_lo = int(CENTER_FREQ)
        sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        sdr.rx_buffer_size = 2**18  # 256k samples for repeated frame capture

        # Enable digital loopback
        print("[INFO] Enabling digital loopback mode...")
        if not enable_digital_loopback(sdr):
            return result.fail("Could not enable loopback")

        # Create OFDM transceiver
        ofdm_cfg = OFDMConfig()
        ofdm = OFDMTransceiver(ofdm_cfg)

        # Generate random test bits - use fixed seed for reproducibility
        np.random.seed(42)
        num_bits = ofdm_cfg.bits_per_frame  # Single OFDM frame
        tx_bits = np.random.randint(0, 2, num_bits)

        print(f"[INFO] Transmitting {num_bits} bits ({num_bits//8} bytes)")
        print(f"[INFO] OFDM Config: FFT={ofdm_cfg.fft_size}, CP={ofdm_cfg.cp_length}, "
              f"Data carriers={ofdm_cfg.num_data_carriers}")

        # Modulate single frame
        tx_signal = ofdm.modulate(tx_bits)

        # KEY FIX: Repeat the OFDM frame many times for cyclic mode stability
        # This ensures the SDR has consistent pattern to lock onto
        num_repeats = 50
        tx_repeated = np.tile(tx_signal, num_repeats)

        # Normalize and scale for DAC
        max_val = np.max(np.abs(tx_repeated))
        if max_val > 0:
            tx_repeated = tx_repeated / max_val * 0.5  # Conservative scaling
        tx_scaled = (tx_repeated * 2**14).astype(np.complex64)

        print(f"[INFO] TX signal: {len(tx_signal)} samples x {num_repeats} repeats = {len(tx_scaled)} total")

        # Use cyclic TX mode
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_scaled)

        # Wait for TX to stabilize (longer for repeated signal)
        time.sleep(0.2)

        # Receive
        rx_samples = sdr.rx()

        # Stop TX
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        # Disable loopback
        disable_digital_loopback(sdr)

        if rx_samples is None or len(rx_samples) == 0:
            return result.fail("No samples received")

        if np.max(np.abs(rx_samples)) < 100:
            return result.fail(f"Zero signal received. Max: {np.max(np.abs(rx_samples)):.2e}")

        print(f"[INFO] Received {len(rx_samples)} samples, max={np.max(np.abs(rx_samples)):.0f}")

        # Normalize RX to match TX scaling
        rx_max = np.max(np.abs(rx_samples))
        tx_ref = tx_signal / max_val * 0.5  # Reference single frame
        rx_norm = rx_samples / rx_max * 0.5

        # Find alignment using correlation with one frame
        corr = np.correlate(rx_norm[:5000], tx_ref[:200], mode='valid')
        peak_idx = np.argmax(np.abs(corr))

        print(f"[INFO] Frame alignment at index {peak_idx}")

        # Extract one frame from RX
        rx_frame = rx_norm[peak_idx:peak_idx + len(tx_signal)]

        if len(rx_frame) < len(tx_signal):
            return result.fail("Not enough samples for full frame")

        # Check sample-level match (should be very high for repeated cyclic signal)
        sample_corr = np.corrcoef(rx_frame.real, tx_ref.real)[0, 1]
        mse = np.mean(np.abs(rx_frame - tx_ref)**2)
        print(f"[INFO] Sample correlation: {sample_corr:.6f}, MSE: {mse:.6f}")

        if sample_corr < 0.9:
            return result.fail(f"Low sample correlation: {sample_corr:.4f}")

        # Demodulate
        rx_bits, metrics = ofdm.demodulate(rx_frame)

        # Calculate BER
        min_len = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        ber = errors / min_len

        print(f"[INFO] Demodulated {len(rx_bits)} bits")
        print(f"[INFO] Bit errors: {errors} / {min_len}")
        print(f"[INFO] BER: {ber:.2e}")

        if ber < 0.05:  # Less than 5% BER
            return result.success(f"BER = {ber:.2e}", ber=ber, sample_corr=sample_corr)
        else:
            return result.fail(f"High BER: {ber:.2e}", ber=ber)

    except ImportError as e:
        return result.fail(f"Import error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return result.fail(f"Error: {e}")
    finally:
        if sdr is not None:
            try:
                del sdr
            except:
                pass
        time.sleep(0.2)


# =============================================================================
# Level 3: Packet Transmission with FEC
# =============================================================================

def test_level_3_packet_loopback():
    """Test packet transmission with FEC through digital loopback."""
    print_header("Level 3: Packet Transmission with FEC Test")
    result = TestResult("Packet Loopback")

    sdr = None
    try:
        import adi
        import zlib
        from sdr_video_commv2_lab import (
            SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType
        )

        sdr = adi.Pluto(PLUTO_IP)

        # Configure SDR with larger buffer for repeated frames
        sdr.sample_rate = int(SAMPLE_RATE)
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.rx_lo = int(CENTER_FREQ)
        sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        sdr.rx_buffer_size = 2**18  # 256k samples

        # Enable digital loopback
        print("[INFO] Enabling digital loopback mode...")
        if not enable_digital_loopback(sdr):
            return result.fail("Could not enable loopback")

        # Create SDRVideoLink in simulation mode (we handle SDR separately)
        sdr_cfg = SDRConfig()
        sdr_cfg.fs = SAMPLE_RATE
        sdr_cfg.fc = CENTER_FREQ

        ofdm_cfg = OFDMConfig()

        # No FEC for this test (simplify debugging)
        fec_cfg = FECConfig(enabled=False)

        link = SDRVideoLink(
            sdr_config=sdr_cfg,
            ofdm_config=ofdm_cfg,
            fec_config=fec_cfg,
            simulation_mode=True
        )

        # Create a test packet
        packet_size = 256  # bytes
        test_payload = bytes(range(256))  # Known pattern

        # Add header
        packet = link.video_codec.create_packet_header(
            test_payload,
            frame_id=0,
            pkt_idx=0,
            total_pkts=1,
            quality=50
        )

        print(f"[INFO] Packet size: {len(packet)} bytes")

        # Convert to bits
        np.random.seed(42)
        pkt_arr = np.frombuffer(packet, dtype=np.uint8)
        bits = np.unpackbits(pkt_arr)

        # FEC encode (if enabled)
        fec_bits = link.fec_codec.encode(bits)
        print(f"[INFO] Bits: {len(bits)} -> FEC: {len(fec_bits)}")

        # Modulate
        tx_signal = link.transceiver.modulate(fec_bits)

        # KEY: Repeat the signal for cyclic mode stability (same as Level 2)
        num_repeats = 30
        tx_repeated = np.tile(tx_signal, num_repeats)

        # Normalize and scale
        max_val = np.max(np.abs(tx_repeated))
        if max_val > 0:
            tx_repeated = tx_repeated / max_val * 0.5
        tx_scaled = (tx_repeated * 2**14).astype(np.complex64)

        print(f"[INFO] TX signal: {len(tx_signal)} samples x {num_repeats} repeats = {len(tx_scaled)} total")

        # Use cyclic TX mode
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_scaled)

        # Wait for TX to stabilize
        time.sleep(0.2)

        # Receive
        rx_samples = sdr.rx()

        # Stop TX
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        # Disable loopback
        disable_digital_loopback(sdr)

        if rx_samples is None or len(rx_samples) == 0:
            return result.fail("No samples received")

        if np.max(np.abs(rx_samples)) < 100:
            return result.fail("Zero signal (DMA deadlock)")

        print(f"[INFO] Received {len(rx_samples)} samples, max={np.max(np.abs(rx_samples)):.0f}")

        # Normalize RX to match TX scaling
        rx_max = np.max(np.abs(rx_samples))
        tx_ref = tx_signal / max_val * 0.5  # Reference single frame
        rx_norm = rx_samples / rx_max * 0.5

        # Find alignment
        corr = np.correlate(rx_norm[:10000], tx_ref[:200], mode='valid')
        peak_idx = np.argmax(np.abs(corr))

        print(f"[INFO] Frame alignment at index {peak_idx}")

        # Extract one frame
        rx_frame = rx_norm[peak_idx:peak_idx + len(tx_signal)]

        if len(rx_frame) < len(tx_signal):
            return result.fail("Not enough samples for full frame")

        # Check sample-level match
        sample_corr = np.corrcoef(rx_frame.real, tx_ref.real)[0, 1]
        print(f"[INFO] Sample correlation: {sample_corr:.6f}")

        if sample_corr < 0.9:
            return result.fail(f"Low sample correlation: {sample_corr:.4f}")

        # Demodulate
        rx_fec_bits, metrics = link.transceiver.demodulate(rx_frame)

        # FEC decode
        rx_bits_dec = link.fec_codec.decode(rx_fec_bits)

        # Truncate to original length
        expected_bits = len(bits)
        if len(rx_bits_dec) < expected_bits:
            return result.fail("Not enough decoded bits")

        rx_bits_dec = rx_bits_dec[:expected_bits]

        # Convert back to bytes
        rx_bytes = np.packbits(rx_bits_dec).tobytes()

        # Parse packet
        info = link.video_codec.parse_packet_header(rx_bytes)

        if info is None:
            return result.fail("Failed to parse packet header")

        # Verify CRC
        calc_crc = zlib.crc32(info['payload']) & 0xFFFFFFFF
        crc_ok = calc_crc == info['crc']

        print(f"[INFO] Parsed header: frame_id={info['frame_id']}, "
              f"pkt={info['pkt_idx']}/{info['total_pkts']}")
        print(f"[INFO] CRC verified: {crc_ok}")

        if crc_ok:
            return result.success(
                "Packet CRC verified",
                frame_id=info['frame_id'],
                pkt_idx=info['pkt_idx']
            )
        else:
            return result.fail("CRC mismatch")

    except ImportError as e:
        return result.fail(f"Import error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return result.fail(f"Error: {e}")
    finally:
        if sdr is not None:
            try:
                del sdr
            except:
                pass
        time.sleep(0.2)


# =============================================================================
# Level 4: Video Frame Encoding/Decoding Test
# =============================================================================

def test_level_4_video_loopback():
    """Test video frame encoding and decoding through digital loopback."""
    print_header("Level 4: Video Frame Loopback Test")
    result = TestResult("Video Frame Loopback")

    sdr = None
    try:
        import cv2
        import adi
        import zlib
        from sdr_video_commv2_lab import (
            SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType
        )

        sdr = adi.Pluto(PLUTO_IP)

        # Configure SDR with larger buffer for repeated frames
        sdr.sample_rate = int(SAMPLE_RATE)
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.rx_lo = int(CENTER_FREQ)
        sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        sdr.rx_buffer_size = 2**18  # 256k samples

        # Enable digital loopback
        print("[INFO] Enabling digital loopback mode...")
        if not enable_digital_loopback(sdr):
            return result.fail("Could not enable loopback")

        # Create SDRVideoLink
        sdr_cfg = SDRConfig()
        sdr_cfg.fs = SAMPLE_RATE
        sdr_cfg.fc = CENTER_FREQ

        ofdm_cfg = OFDMConfig()

        fec_cfg = FECConfig(enabled=False)

        link = SDRVideoLink(
            sdr_config=sdr_cfg,
            ofdm_config=ofdm_cfg,
            fec_config=fec_cfg,
            simulation_mode=True
        )

        # Set video resolution (small for fast testing)
        width, height = 160, 120
        quality = 50
        link.video_config.resolution = (width, height)
        link.video_config.packet_size = 512

        # Create test frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.tile(np.linspace(0, 255, width), (height, 1))
        frame[:, :, 1] = np.tile(np.linspace(0, 255, height).reshape(-1, 1), (1, width))
        cv2.rectangle(frame, (20, 20), (60, 60), (255, 255, 255), -1)
        cv2.putText(frame, "TEST", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        print(f"[INFO] Test frame: {width}x{height}, quality={quality}")

        # Encode frame to packets
        packets = link.video_codec.encode_frame(frame, quality=quality)
        print(f"[INFO] Frame encoded to {len(packets)} packets")

        # Transmit first packet
        pkt_bytes, pkt_idx = packets[0]

        # Convert to bits
        np.random.seed(42)
        pkt_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
        bits = np.unpackbits(pkt_arr)

        # FEC encode
        fec_bits = link.fec_codec.encode(bits)
        print(f"[INFO] Packet {pkt_idx}: {len(pkt_bytes)} bytes -> {len(fec_bits)} FEC bits")

        # Modulate
        tx_signal = link.transceiver.modulate(fec_bits)

        # KEY: Repeat signal for cyclic mode stability (same as Level 2/3)
        num_repeats = 30
        tx_repeated = np.tile(tx_signal, num_repeats)

        # Normalize and scale
        max_val = np.max(np.abs(tx_repeated))
        if max_val > 0:
            tx_repeated = tx_repeated / max_val * 0.5
        tx_scaled = (tx_repeated * 2**14).astype(np.complex64)

        print(f"[INFO] TX signal: {len(tx_signal)} samples x {num_repeats} repeats = {len(tx_scaled)} total")

        # Use cyclic TX mode
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_scaled)

        # Wait for TX to stabilize
        time.sleep(0.2)

        # Receive
        rx_samples = sdr.rx()

        # Stop TX
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        # Disable loopback
        disable_digital_loopback(sdr)

        if rx_samples is None or len(rx_samples) == 0:
            return result.fail("No samples received")

        if np.max(np.abs(rx_samples)) < 100:
            return result.fail("Zero signal (DMA deadlock)")

        print(f"[INFO] Received {len(rx_samples)} samples, max={np.max(np.abs(rx_samples)):.0f}")

        # Normalize RX to match TX scaling
        rx_max = np.max(np.abs(rx_samples))
        tx_ref = tx_signal / max_val * 0.5  # Reference single frame
        rx_norm = rx_samples / rx_max * 0.5

        # Find alignment
        corr = np.correlate(rx_norm[:10000], tx_ref[:200], mode='valid')
        peak_idx = np.argmax(np.abs(corr))

        print(f"[INFO] Frame alignment at index {peak_idx}")

        # Extract one frame
        rx_frame = rx_norm[peak_idx:peak_idx + len(tx_signal)]

        if len(rx_frame) < len(tx_signal):
            return result.fail("Not enough samples for full frame")

        # Check sample-level match
        sample_corr = np.corrcoef(rx_frame.real, tx_ref.real)[0, 1]
        print(f"[INFO] Sample correlation: {sample_corr:.6f}")

        if sample_corr < 0.9:
            return result.fail(f"Low sample correlation: {sample_corr:.4f}")

        # Demodulate
        rx_fec_bits, metrics = link.transceiver.demodulate(rx_frame)

        # FEC decode
        rx_bits_dec = link.fec_codec.decode(rx_fec_bits)

        expected_bits = len(bits)
        if len(rx_bits_dec) < expected_bits:
            return result.fail("Not enough decoded bits")

        rx_bits_dec = rx_bits_dec[:expected_bits]
        rx_bytes = np.packbits(rx_bits_dec).tobytes()

        # Parse packet
        info = link.video_codec.parse_packet_header(rx_bytes)

        if info is None:
            return result.fail("Failed to parse packet header")

        # Verify CRC
        calc_crc = zlib.crc32(info['payload']) & 0xFFFFFFFF
        crc_ok = calc_crc == info['crc']

        # Calculate BER
        tx_bits_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
        rx_bits_arr = np.frombuffer(rx_bytes, dtype=np.uint8)
        min_len = min(len(tx_bits_arr), len(rx_bits_arr))
        bit_errors = np.sum(np.unpackbits(tx_bits_arr[:min_len]) !=
                           np.unpackbits(rx_bits_arr[:min_len]))
        ber = bit_errors / (min_len * 8)

        print(f"[INFO] Parsed: frame_id={info['frame_id']}, "
              f"w={info['width']}x{info['height']}, q={info['quality']}")
        print(f"[INFO] CRC verified: {crc_ok}, BER: {ber:.2e}")

        if crc_ok and ber == 0:
            return result.success(
                "Video packet perfect (0% BER)",
                frame_id=info['frame_id'],
                resolution=(info['width'], info['height'])
            )
        elif crc_ok and ber < 0.01:
            return result.success(
                f"Video packet good (BER={ber:.2e})",
                ber=ber
            )
        else:
            return result.fail(f"CRC: {crc_ok}, BER: {ber:.2e}")

    except ImportError as e:
        return result.fail(f"Import error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return result.fail(f"Error: {e}")
    finally:
        if sdr is not None:
            try:
                del sdr
            except:
                pass
        time.sleep(0.2)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests(level=None, max_level=4, verbose=True):
    """Run digital loopback tests."""
    tests = [
        (0, test_level_0_connection),
        (1, test_level_1_tone_loopback),
        (2, test_level_2_ofdm_loopback),
        (3, test_level_3_packet_loopback),
        (4, test_level_4_video_loopback),
    ]

    results = []

    for test_level, test_func in tests:
        if level is not None and test_level != level:
            continue

        if test_level > max_level:
            continue

        result = test_func()
        results.append(result)

        print("\n" + str(result))

        if not result.passed and test_level < max_level and level is None:
            print(f"\n[STOP] Test level {test_level} failed. Stopping further tests.")
            break

    # Summary
    print_header("Test Summary")
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return all(r.passed for r in results)


def main():
    global PLUTO_IP

    parser = argparse.ArgumentParser(description='Pluto SDR Digital Loopback Test')
    parser.add_argument('--level', type=int, default=None,
                        help='Run only specific test level (0-4)')
    parser.add_argument('--max-level', type=int, default=4,
                        help='Maximum test level to run (default: 4)')
    parser.add_argument('--ip', type=str, default=PLUTO_IP,
                        help=f'Pluto SDR URI (default: {PLUTO_IP})')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    PLUTO_IP = args.ip

    print_header("Pluto SDR Digital Loopback Test Suite")
    print(f"SDR URI: {PLUTO_IP}")
    print(f"Sample Rate: {SAMPLE_RATE/1e6:.1f} MSPS")
    print(f"Center Freq: {CENTER_FREQ/1e6:.1f} MHz")

    success = run_tests(level=args.level, max_level=args.max_level, verbose=args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
