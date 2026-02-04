#!/usr/bin/env python
"""
run_video_loopback_tdd.py - SDR Video Loopback Test with TDD Support

Based on run_video_loopback_digital.py but adds TDD mode support.
"""

import sys
import time
import numpy as np
import argparse
import json
import os
import zlib
import iio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Qt Wayland issue
os.environ["QT_QPA_PLATFORM"] = "xcb"

from sdr_video_commv2 import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

# =============================================================================
# TDD Controller
# =============================================================================

class TDDController:
    def __init__(self, uri='ip:192.168.2.2'):
        self.uri = uri
        self.tdd = None
        self.enabled = False
        
    def connect(self):
        try:
            import adi
            self.tdd = adi.tddn(self.uri)
            print(f"[TDD] Connected to AXI TDD Controller")
            return True
        except Exception as e:
            print(f"[TDD] Failed to connect: {e}")
            return False
    
    def configure(self, frame_length_ms=20.0, tx_duration_ms=8.0, 
                  guard_ms=2.0, rx_duration_ms=8.0):
        if self.tdd is None:
            return False
        
        try:
            self.tdd.enable = False
            self.tdd.frame_length_ms = frame_length_ms
            self.tdd.startup_delay_ms = 0.1
            
            # Assuming Ch1 = ENABLE, Ch0 = TXNRX (or vice versa)
            # We want ENABLE high for both TX and RX.
            # We want TXNRX high for TX, low for RX.
            
            # Try Configuration:
            # Ch1 (ENABLE): ON from 0 to rx_end
            # Ch0 (TXNRX): ON from 0 to tx_duration
            
            # Ch0 (TXNRX)
            if len(self.tdd.channel) > 0:
                self.tdd.channel[0].on_ms = 0.0
                self.tdd.channel[0].off_ms = tx_duration_ms
                self.tdd.channel[0].enable = True
                self.tdd.channel[0].polarity = 1 # Try Inverted Polarity
            
            # Ch1 (ENABLE) - Covers both TX and RX
            rx_start = tx_duration_ms + guard_ms
            rx_end = rx_start + rx_duration_ms
            
            if len(self.tdd.channel) > 1:
                self.tdd.channel[1].on_ms = 0.0
                self.tdd.channel[1].off_ms = rx_end
                self.tdd.channel[1].enable = True
                self.tdd.channel[1].polarity = 1 # Try Inverted Polarity
            
            self.tdd.burst_count = 0
            self.tdd.sync_internal = True
            self.tdd.internal_sync_period_ms = frame_length_ms
            
            print(f"[TDD] Configured (ENABLE/TXNRX mode): frame={frame_length_ms}ms, "
                  f"TXNRX (Ch0)={0.0}-{tx_duration_ms}ms, ENABLE (Ch1)={0.0}-{rx_end}ms")
            return True
        except Exception as e:
            print(f"[TDD] Configuration failed: {e}")
            return False
    
    def enable(self):
        if self.tdd is None: return False
        try:
            self.tdd.enable = True
            self.enabled = True
            print("[TDD] Enabled")
            return True
        except Exception as e:
            print(f"[TDD] Enable failed: {e}")
            return False
    
    def disable(self):
        if self.tdd is None: return False
        try:
            self.tdd.enable = False
            self.enabled = False
            print("[TDD] Disabled")
            return True
        except Exception as e:
            print(f"[TDD] Disable failed: {e}")
            return False

# =============================================================================
# Digital Loopback Control
# =============================================================================

def set_loopback_mode(sdr_ip, mode):
    try:
        ctx = iio.Context(sdr_ip)
        phy = ctx.find_device('ad9361-phy')
        if phy:
            phy.debug_attrs['loopback'].value = str(mode)
            actual = phy.debug_attrs['loopback'].value
            print(f"[SDR] Loopback mode set to {actual}")
            return True
    except Exception as e:
        print(f"[SDR] Failed to set loopback: {e}")
    return False

# =============================================================================
# Digital AGC
# =============================================================================

class DigitalAGC:
    def __init__(self, target_level=0.5, alpha=0.1): 
        self.gain = 1.0
        self.target = target_level
        self.alpha = alpha 
        
    def process(self, signal):
        curr_amp = np.mean(np.abs(signal))
        if curr_amp < 1e-9: 
            curr_amp = 1e-9
        ideal_gain = self.target / curr_amp
        self.gain = (1 - self.alpha) * self.gain + self.alpha * ideal_gain
        if self.gain > 10000.0: 
            self.gain = 10000.0
        if self.gain < 0.01: 
            self.gain = 0.01
        return signal * self.gain

# =============================================================================
# Buffer Windowing Sync (Optional helper for TDD)
# =============================================================================

def sync_with_windowing(rx_signal, preamble, payload_samples, threshold=15.0, 
                        sample_rate=2e6, verbose=False):
    # This can be used if standard sync fails in TDD mode
    # For now, we will try to use the standard link._synchronize which is robust enough
    # if we capture enough signal.
    pass

# =============================================================================
# Main
# =============================================================================

def main():
    import cv2
    
    parser = argparse.ArgumentParser(description='SDR Video Loopback Test (TDD/Digital/RF)')
    parser.add_argument('--ip', default='ip:192.168.2.2', help='SDR IP address')
    parser.add_argument('--config', default='sdr_tuned_config.json', help='SDR config file')
    parser.add_argument('--mode', default='tdd', choices=['tdd', 'cyclic', 'digital', 'rf'], 
                        help='Mode: tdd (burst), cyclic (continuous), digital/rf (legacy aliases for cyclic)')
    parser.add_argument('--max-frames', type=int, default=5, help='Max frames to test')
    parser.add_argument('--quality', type=int, default=50, help='JPEG quality')
    parser.add_argument('--width', type=int, default=320, help='Frame width')
    parser.add_argument('--height', type=int, default=240, help='Frame height')
    parser.add_argument('--fec', type=str, default='ldpc', choices=['ldpc', 'repetition', 'none'], help='FEC type')
    parser.add_argument('--packet-size', type=int, default=1024, help='Packet size')
    parser.add_argument('--mac-repeats', type=int, default=2, help='MAC layer packet repetitions')
    parser.add_argument('--sync-threshold', type=float, default=15.0, help='Sync threshold')
    parser.add_argument('--tx-gain', type=int, default=0, help='TX gain override')
    parser.add_argument('--rx-gain', type=int, default=40, help='RX gain override')
    parser.add_argument('--loopback', type=str, default='digital', choices=['rf', 'digital', 'disabled'], help='Loopback type')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Map legacy modes
    if args.mode == 'digital':
        args.mode = 'cyclic'
        args.loopback = 'digital'
    elif args.mode == 'rf':
        args.mode = 'cyclic'
        args.loopback = 'rf'
        
    print("=" * 60)
    print(f"SDR Video Loopback Test - {args.mode.upper()} Mode")
    print("=" * 60)
    
    # Load Config
    if os.path.exists(args.config):
        sdr_cfg = SDRConfig.load_from_json(args.config)
    else:
        sdr_cfg = SDRConfig()
    
    sdr_cfg.sdr_ip = args.ip
    sdr_cfg.rx_uri = None # Force single device mode
    sdr_cfg.tx_gain = args.tx_gain
    sdr_cfg.rx_gain = args.rx_gain
    
    # Override for Loopback Test stability
    if args.loopback != 'disabled':
        print("[SDR] Overriding config for Loopback Test (FS=3M, BW=3M, LO=2405M)")
        sdr_cfg.fs = 3000000
        sdr_cfg.bandwidth = 3000000
        sdr_cfg.fc = 2405000000  # 2.405 GHz has better loopback response than 915 MHz
        
        # Auto-set safe gain for loopback if using defaults
        # RF loopback with 30dB attenuator needs high gains
        if args.tx_gain == 0 and args.rx_gain == 40:  # Both at defaults
            print("[SDR] Auto-setting gains for RF loopback: TX=0dB, RX=70dB")
            sdr_cfg.tx_gain = 0
            sdr_cfg.rx_gain = 70
        else:
            # Use user-specified gains
            sdr_cfg.tx_gain = args.tx_gain
            sdr_cfg.rx_gain = args.rx_gain
    
    # FEC Config
    fec_map = {"none": FECType.NONE, "repetition": FECType.REPETITION, "ldpc": FECType.LDPC}
    fec_type = fec_map.get(args.fec.lower(), FECType.LDPC)
    fec_cfg = FECConfig(enabled=fec_type != FECType.NONE, fec_type=fec_type)
    
    # OFDM Config
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = args.sync_threshold
    
    # Initialize SDR Link
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg)
    link.video_config.packet_size = args.packet_size
    link.video_config.resolution = (args.width, args.height)
    
    # Connect to SDR
    print(f"\n[SDR] Connecting to {args.ip}...")
    if not link.connect_sdr():
        print("[SDR] ERROR: Failed to connect")
        return
        
    # Get reference to low-level device
    adi_dev = link.sdr.sdr
    
    # Configure for Loopback (Digital or RF)
    if args.loopback != 'disabled':
        print(f"[SDR] Configuring for {args.loopback.upper()} Loopback...")
        
        # 1. Force FDD Mode - Disabled for now as it might conflict with auto-mode
        # try:
        #     adi_dev._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
        #     adi_dev._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
        #     adi_dev._ctrl.debug_attrs["initialize"].value = "1"
        #     print("[SDR] FDD Mode Enabled")
        # except Exception as e:
        #     print(f"[SDR] Warning: Failed to set FDD: {e}")

        # 2. Set Gains - Use the configured values, NOT hardcoded values!
        try:
            adi_dev.gain_control_mode_chan0 = "manual"
            adi_dev.tx_hardwaregain_chan0 = sdr_cfg.tx_gain
            adi_dev.rx_hardwaregain_chan0 = sdr_cfg.rx_gain
            print(f"[SDR] Gains set: TX={sdr_cfg.tx_gain}dB, RX={sdr_cfg.rx_gain}dB")
        except Exception as e:
            print(f"[SDR] Warning: Failed to set gains: {e}")

        # 3. Enable Loopback Mode
        if args.loopback == 'digital':
            print("[SDR] Enabling Digital Loopback (Mode 1)...")
            try:
                # Try property first
                adi_dev.loopback = 1 
            except:
                # Fallback to debug_attrs
                try:
                    adi_dev.ctx.find_device('ad9361-phy').debug_attrs['loopback'].value = '1'
                except Exception as e:
                    print(f"[SDR] Failed to set loopback: {e}")
        elif args.loopback == 'rf':
            print("[SDR] Using RF Loopback (External Cable)")
            set_loopback_mode(args.ip, 0)

        # 4. Set Buffer Size (Increased for robustness)
        adi_dev.rx_buffer_size = 65536
        
        # 5. Sanity Check
        print("--- SANITY CHECK ---")
        time.sleep(2.0) # Wait for settings to settle
        try:
            # Generate tone (Same scaling as sdr_auto_tune)
            t = np.arange(1024)
            # sdr_auto_tune uses: 2**14 * 0.5
            tone = np.exp(1j * 2 * np.pi * 0.1 * t) * (2**14 * 0.5)
            tone = tone.astype(np.complex64)
            
            # Reset Buffer
            adi_dev.tx_destroy_buffer()
            adi_dev.tx_cyclic_buffer = True
            adi_dev.tx(tone)
            time.sleep(0.5)
            
            # Read
            rx_sanity = adi_dev.rx()
            max_sanity = np.max(np.abs(rx_sanity))
            print(f"Sanity RX Max: {max_sanity}")
            
            if max_sanity < 100:
                print("\033[91mSANITY CHECK FAILED: No Signal (Check Loopback/Mode)\033[0m")
            else:
                print("\033[92mSANITY CHECK PASSED\033[0m")
                
            adi_dev.tx_destroy_buffer()
            
        except Exception as e:
            print(f"SANITY CHECK ERROR: {e}")
            
    # Readback Gain
    try:
        tg = adi_dev.tx_hardwaregain_chan0
        rg = adi_dev.rx_hardwaregain_chan0
        if hasattr(adi_dev, 'gain_control_mode_chan0'):
            mode = adi_dev.gain_control_mode_chan0
        else:
            mode = "unknown"
        print(f"[SDR] Actual Gain State: TX={tg}dB, RX={rg}dB, Mode={mode}")
    except:
        print("[SDR] Could not read back gain state")

    print("[SDR] Connected successfully!")
    
    # adi_dev is already set above
    
    # TDD Setup
    tdd_ctrl = None
    tx_duration_ms = 0
    guard_ms = 1.0
    
    if args.mode == 'tdd':
        # Configure AD9361 for TDD Mode (Pin Control)
        # ... (keep existing code)
        try:
            print("[SDR] Configuring AD9361 for TDD Pin Control...")
            # Disable FDD (Enable TDD)
            adi_dev._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "0"
            # Enable Pin Control (ENSM controlled by TDD core)
            adi_dev._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "1"
            # Apply changes
            adi_dev._ctrl.debug_attrs["initialize"].value = "1"
            print("[SDR] AD9361 TDD Mode Configured")
        except Exception as e:
            print(f"[SDR] Warning: Failed to configure TDD Pin Control: {e}")
            
    else:
        # Force FDD Mode for Cyclic/Digital Loopback
        try:
            print("[SDR] Ensuring AD9361 is in FDD Mode (SPI Control)...")
            # Always apply to ensure Pin Control is disabled
            adi_dev._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
            adi_dev._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
            adi_dev._ctrl.debug_attrs["initialize"].value = "1"
            
            # Force ENSM to FDD state (if property exists)
            try:
                # Check current state
                if hasattr(adi_dev, 'ensm_mode'):
                     print(f"[SDR] Current ENSM Mode: {adi_dev.ensm_mode}")
                     # adi_dev.ensm_mode = 'fdd' # Some drivers use 'fdd', some auto
            except: pass
            
            print("[SDR] AD9361 FDD Mode Configured")
        except Exception as e:
            print(f"[SDR] Warning: Failed to configure FDD Mode: {e}")

    if args.mode == 'tdd':
        # Calculate timing
        sample_rate = sdr_cfg.fs
        # Estimate packet duration
        # We need to know payload size first, but we can estimate
        # Assuming max packet size
        est_bits = args.packet_size * 8 * 2 # 2x for FEC
        est_samples = (est_bits / ofdm_cfg.bits_per_frame) * ofdm_cfg.samples_per_frame + 500
        tx_duration_ms = (est_samples / sample_rate) * 1000 + 4.0 # Increased Margin
        rx_duration_ms = tx_duration_ms * 2
        frame_length_ms = tx_duration_ms + guard_ms + rx_duration_ms + 2.0
        
        # DEBUG: Force RX ON for longer
        # rx_duration_ms = frame_length_ms - tx_duration_ms - guard_ms
        
        tdd_ctrl = TDDController(args.ip)
        if tdd_ctrl.connect():
            tdd_ctrl.configure(frame_length_ms, tx_duration_ms, guard_ms, rx_duration_ms)
            tdd_ctrl.enable()
        else:
            print("[TDD] Failed to connect, falling back to cyclic")
            args.mode = 'cyclic'

    # Buffer Config
    # Calculate expected samples based on packet size and FEC
    base_packet_bytes = link.video_config.packet_size
    expected_packet_bits = base_packet_bytes * 8
    
    # Calculate FEC output size
    if fec_cfg.enabled and fec_cfg.fec_type == FECType.LDPC and hasattr(link.fec_codec, 'n'):
        blocks = int(np.ceil(expected_packet_bits / link.fec_codec.k))
        expected_fec_bits = blocks * link.fec_codec.n
    elif fec_cfg.enabled and fec_cfg.fec_type == FECType.REPETITION:
        expected_fec_bits = expected_packet_bits * fec_cfg.repetitions
    else:
        expected_fec_bits = expected_packet_bits
    
    bits_per_frame = link.ofdm_config.bits_per_frame
    samples_per_frame = link.ofdm_config.samples_per_frame
    num_frames = int(np.ceil(expected_fec_bits / bits_per_frame))
    expected_payload_samples = num_frames * samples_per_frame
    preamble_len = len(link._generate_preamble())
    min_capture = preamble_len + expected_payload_samples
    
    # Set RX buffer size dynamically
    # For TDD, we use a smaller buffer to avoid timeouts if the RX window is tight
    if args.mode == 'tdd':
         rx_buffer_size = 4096 * 4 # 16k samples, should fit in 18ms window easily
    else:
         rx_buffer_size = 32768 # Standard buffer size

    
    # Increase timeout to prevent "Unable to dequeue block" errors
    # Default is often 1-3s, which might be too short for large buffers or slow USB
    if hasattr(adi_dev, "ctx") and hasattr(adi_dev.ctx, "set_timeout"):
        print("[SDR] Setting libiio timeout to 30000ms")
        adi_dev.ctx.set_timeout(30000)
    
    # Configure buffer size
    adi_dev.rx_buffer_size = int(rx_buffer_size)
    if hasattr(adi_dev, "_rxadc") and hasattr(adi_dev._rxadc, "set_kernel_buffers_count"):
        adi_dev._rxadc.set_kernel_buffers_count(4)
        
    print(f"[SDR] RX buffer: {rx_buffer_size} samples (Min capture: {min_capture})")
    
    # Initialize AGC
    agc = DigitalAGC(target_level=0.5, alpha=0.1)
    
    stats = {
        'tx_packets': 0,      # Physical TX bursts
        'rx_packets': 0,      # Successfully received logical packets (CRC OK)
        'total_packets': 0,   # Total logical packets attempted
        'crc_fails': 0,
        'bit_errors': 0,
        'total_bits_rx': 0,
        'total_frames': 0,
        'recovered_frames': 0
    }
    
    print("\n" + "=" * 60)
    print("Starting Loopback Test...")
    print("=" * 60 + "\n")
    
    try:
        for frame_idx in range(args.max_frames):
            print(f"\n--- Frame {frame_idx + 1}/{args.max_frames} ---")
            stats['total_frames'] += 1
            
            # Generate test frame
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10, 10), (50, 50), (0, 255, 0), -1)
            
            # Encode frame
            packets = link.video_codec.encode_frame(frame, quality=args.quality)
            print(f"[TX] Encoded: {len(packets)} packets")
            
            packet_rx_success = 0
            # Track received packets across the entire frame transmission
            packets_received_mask = [False] * len(packets)
            
            for pkt_idx, (pkt_bytes, pkt_i) in enumerate(packets):
                stats['total_packets'] += 1
                
                # Prepare signal
                pkt_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                fec_bits = link.fec_codec.encode(bits)
                tx_signal = link.transceiver.modulate(fec_bits)
                preamble = link._generate_preamble()
                
                # Boost preamble power to match OFDM PAPR
                # OFDM Peak can be 8-10x average. Preamble is constant envelope 1.0.
                # If we normalize to Payload Peak, Preamble becomes tiny (~0.1), causing Sync/CFO failure.
                # We scale preamble up to ensure it uses good dynamic range.
                payload_peak = np.max(np.abs(tx_signal)) if len(tx_signal) > 0 else 1.0
                preamble_scale = max(1.0, payload_peak * 0.8) # 80% of payload peak
                preamble = preamble * preamble_scale
                
                tx_signal_full = np.concatenate([preamble, tx_signal])
                
                # Pad to 1024 samples alignment for DMA safety
                align = 1024
                pad_len = (align - len(tx_signal_full) % align) % align
                if pad_len > 0:
                    tx_signal_full = np.concatenate([tx_signal_full, np.zeros(pad_len, dtype=np.complex64)])
                
                # Normalize
                max_val = np.max(np.abs(tx_signal_full))
                if max_val > 0: tx_signal_full /= max_val
                
                # Scale for Pluto (expecting -1.0 to 1.0 complex floats, or raw ints?)
                # sdr_auto_tune uses 2**14 * 0.5 (~8192) for Digital Loopback
                # If we send large values, pyadi-iio might clip them or interpret them wrong.
                # Let's try 0.5 scale like diagnose_sdr.py
                
                # Digital Loopback REQUIRES large values (it bypasses DAC scaling?)
                scale_factor = 2**14 * 0.5 # Back to 0.5 to match sdr_auto_tune
                tx_signal_full = (tx_signal_full * scale_factor).astype(np.complex64)
                
                if args.verbose:
                    print(f"    [TX Debug] Max Amp: {np.max(np.abs(tx_signal_full)):.2f}, Dtype: {tx_signal_full.dtype}")
                
                # Transmit
                rx_success = False
                
                if args.mode == 'cyclic':
                    # Cyclic Mode - Start TX once, then try to RX
                    repeats = max(5, 65536 // len(tx_signal_full) + 1)
                    tx_cyclic = np.tile(tx_signal_full, repeats)
                    
                    try:
                        adi_dev.tx_destroy_buffer()
                        # Explicitly set cyclic buffer property BEFORE transmit
                        adi_dev.tx_cyclic_buffer = True
                        adi_dev.tx(tx_cyclic)
                        stats['tx_packets'] += 1
                        time.sleep(0.5) # Wait for stabilization (increased from 0.1)
                        
                        # Receive attempts
                        # In Cyclic mode, we continuously receive and process a stream
                        # to handle packet fragmentation and alignment issues.
                        
                        rx_stream = np.array([], dtype=np.complex64)
                        packets_received_mask = [False] * len(packets)
                        max_rx_attempts = 20 # Standard attempts
                        
                        for attempt in range(max_rx_attempts):
                            # Break if we have all packets
                            if all(packets_received_mask):
                                break
                                
                            try:
                                rx_samples = adi_dev.rx()
                                if isinstance(rx_samples, (tuple, list)):
                                     rx_samples = rx_samples[0] if len(rx_samples) > 0 else rx_samples
                                
                                if rx_samples is None: continue
                                
                                # Append RAW samples to stream (do NOT normalize per-buffer!)
                                # Per-buffer normalization causes amplitude discontinuities at buffer
                                # boundaries which corrupts OFDM symbols spanning multiple buffers.
                                max_rx = np.max(np.abs(rx_samples))
                                if args.verbose:
                                     print(f"    [RX Raw] Max Level: {max_rx:.2f}")
                                
                                rx_stream = np.concatenate([rx_stream, rx_samples])
                                
                                # Process stream while it has enough data
                                while len(rx_stream) > min_capture:
                                    try:
                                        # Normalize the ENTIRE stream before sync (not per-buffer)
                                        max_stream = np.max(np.abs(rx_stream))
                                        if max_stream > 0:
                                            rx_stream_norm = rx_stream / max_stream
                                        else:
                                            rx_stream_norm = rx_stream
                                        
                                        # Debug: Check signal level
                                        if args.verbose:
                                            max_lvl = np.max(np.abs(rx_stream_norm))
                                            print(f"    [RX Debug] Buffer Level: {max_lvl:.4f}, Len: {len(rx_stream)}")

                                        # Sync & Decode (on normalized stream)
                                        synced_sig, sync_metrics = link._synchronize(rx_stream_norm)
                                        
                                        if args.verbose and sync_metrics.get('sync_success'):
                                            print(f"    [RX Sync] Success! Peak: {sync_metrics.get('peak_val',0):.2f}, Pos: {sync_metrics.get('payload_start',0)}, CFO: {sync_metrics.get('cfo_est',0):.2f} Hz")
                                        elif args.verbose:
                                             pass # Don't spam on failure
                                        
                                        if not sync_metrics.get('sync_success'):
                                            if len(rx_stream) > rx_buffer_size * 2:
                                                rx_stream = rx_stream[-rx_buffer_size:]
                                            break
                                        
                                        # Sync Success!
                                        payload_start = sync_metrics.get('payload_start', 0)
                                        peak_idx = sync_metrics.get('peak_idx', 0)
                                        
                                        # Extract payload
                                        if args.verbose:
                                            print(f"    [Payload] synced_sig={len(synced_sig)}, expected={len(tx_signal)}, peak_idx={peak_idx}, payload_start={payload_start}")
                                        
                                        # Check if we have enough samples for the payload
                                        if len(synced_sig) < len(tx_signal):
                                            if args.verbose:
                                                print(f"    [RX] Skipping: Not enough samples ({len(synced_sig)} < {len(tx_signal)})")
                                            # Need more data - don't advance past incomplete packet
                                            break
                                        
                                        payload = synced_sig[:len(tx_signal)]
                                        
                                        # Advance stream past this packet
                                        packet_len_samples = payload_start + len(tx_signal)
                                        rx_stream = rx_stream[packet_len_samples:]
                                        
                                        # Demodulate & Decode
                                        rx_fec_bits, demod_metrics = link.transceiver.demodulate(payload)
                                        if args.verbose and demod_metrics:
                                            snr_est = demod_metrics.get('snr_est_db', 0)
                                            ch_gain = demod_metrics.get('channel_gain_db', 0)
                                            print(f"    [Demod] SNR={snr_est:.1f}dB, ChGain={ch_gain:.1f}dB, Bits={len(rx_fec_bits)}")
                                            
                                            # Save debug signals on first packet for analysis
                                            if pkt_idx == 0 and snr_est < 5:
                                                np.savez('/Developer/AIsensing/sdradi/debug_signals.npz',
                                                        tx_signal=tx_signal,
                                                        rx_payload=payload,
                                                        synced_sig=synced_sig[:min(len(synced_sig), 20000)],
                                                        sync_metrics=sync_metrics,
                                                        cfo=sync_metrics.get('cfo_est', 0),
                                                        payload_start=payload_start)
                                                print(f"    [DEBUG] Saved signals to debug_signals.npz")
                                        try:
                                            rx_bits = link.fec_codec.decode(rx_fec_bits)
                                            if len(rx_bits) >= len(bits):
                                                rx_bits = rx_bits[:len(bits)]
                                                rx_bytes = np.packbits(rx_bits).tobytes()
                                                info = link.video_codec.parse_packet_header(rx_bytes)
                                                
                                                if info:
                                                    calc_crc = (zlib.crc32(info['payload']) & 0xFFFFFFFF)
                                                    if calc_crc == info['crc']:
                                                        rx_pkt_idx = info['pkt_idx']
                                                        # ... (rest of code)
                                                    else:
                                                        if args.verbose: 
                                                            print(f"    [RX] CRC Fail: {calc_crc:08x} != {info['crc']:08x} (pkt_idx={info['pkt_idx']})")
                                                        # Debug BER
                                                        try:
                                                            idx = info['pkt_idx']
                                                            if idx < len(packets):
                                                                ref_pkt_bytes, _ = packets[idx]
                                                                ref_pkt_arr = np.frombuffer(ref_pkt_bytes, dtype=np.uint8)
                                                                ref_bits = np.unpackbits(ref_pkt_arr)
                                                                # rx_bits might be truncated/padded
                                                                min_len = min(len(ref_bits), len(rx_bits))
                                                                if min_len > 0:
                                                                    bit_diff = np.sum(ref_bits[:min_len] != rx_bits[:min_len])
                                                                    ber = bit_diff / min_len
                                                                    if args.verbose: print(f"    [RX Debug] BER (CRC Fail): {ber:.4f}")
                                                        except: pass
                                                else:
                                                    if args.verbose: print("    [RX] Header Parse Fail")
                                                
                                                if info and (zlib.crc32(info['payload']) & 0xFFFFFFFF) == info['crc']:
                                                    rx_pkt_idx = info['pkt_idx']
                                                    
                                                    if rx_pkt_idx >= len(packets):
                                                        continue
                                                        
                                                    if packets_received_mask[rx_pkt_idx]:
                                                        continue
                                                    
                                                    packets_received_mask[rx_pkt_idx] = True
                                                    stats['rx_packets'] += 1
                                                    packet_rx_success += 1
                                                    
                                                    ref_pkt_bytes, _ = packets[rx_pkt_idx]
                                                    ref_pkt_arr = np.frombuffer(ref_pkt_bytes, dtype=np.uint8)
                                                    ref_bits = np.unpackbits(ref_pkt_arr)
                                                    
                                                    bit_diff = np.sum(ref_bits != rx_bits)
                                                    stats['bit_errors'] += bit_diff
                                                    stats['total_bits_rx'] += len(ref_bits)
                                                    
                                                    rx_success = True
                                                    if args.verbose: print(f"  [RX] Packet {rx_pkt_idx} OK (attempt={attempt}) - BER: {bit_diff/len(ref_bits):.2e}")
                                        
                                        except Exception as e:
                                            if args.verbose: print(f"    [RX] Decode error: {e}")

                                    except Exception as e:
                                        if args.verbose: print(f"    [RX] Stream processing error: {e}")
                                        break
                        
                            except Exception as e:
                                if args.verbose: print(f"    [RX] Sync/Processing error: {e}")
                                # Break inner loop to get more data
                                pass
                    
                    except Exception as e:
                        if args.verbose: print(f"  [RX] Error: {e}")
                        # If error (timeout), try to continue
                        continue
                                
                    finally:
                        adi_dev.tx_destroy_buffer()
                        
                else:
                    # TDD Mode
                    for rep in range(args.mac_repeats):
                        # TDD Burst
                        adi_dev.tx_destroy_buffer()
                        adi_dev.tx_cyclic_buffer = False
                        adi_dev.tx(tx_signal_full)
                        time.sleep((tx_duration_ms + guard_ms) / 1000.0)
                        
                        stats['tx_packets'] += 1
                        
                        # Receive
                        for attempt in range(3):
                            try:
                                rx_samples = adi_dev.rx()
                                if isinstance(rx_samples, (tuple, list)):
                                     rx_samples = rx_samples[0] if len(rx_samples) > 0 else rx_samples
                                
                                if rx_samples is None or len(rx_samples) < len(tx_signal_full): continue
                                
                                # Normalize & AGC
                                max_rx = np.max(np.abs(rx_samples))
                                rx_norm = rx_samples / max_rx if max_rx > 0 else rx_samples
                                rx_signal = agc.process(rx_norm)
                                
                                # Sync & Decode
                                synced_sig, sync_metrics = link._synchronize(rx_signal)
                                
                                if sync_metrics.get('sync_success'):
                                    payload = synced_sig[:len(tx_signal)] # Approximate length
                                    rx_fec_bits, _ = link.transceiver.demodulate(payload)
                                    try:
                                        rx_bits = link.fec_codec.decode(rx_fec_bits)
                                        if len(rx_bits) >= len(bits):
                                            rx_bits = rx_bits[:len(bits)]
                                            rx_bytes = np.packbits(rx_bits).tobytes()
                                            info = link.video_codec.parse_packet_header(rx_bytes)
                                            
                                            if info and (zlib.crc32(info['payload']) & 0xFFFFFFFF) == info['crc']:
                                                # Verify it's the correct packet
                                                if info['pkt_idx'] != pkt_i:
                                                    if args.verbose: print(f"    [RX] Stale packet: got {info['pkt_idx']}, expected {pkt_i}")
                                                    continue
                                                    
                                                stats['rx_packets'] += 1
                                                packet_rx_success += 1
                                                
                                                # Calculate BER
                                                bit_diff = np.sum(bits != rx_bits)
                                                stats['bit_errors'] += bit_diff
                                                stats['total_bits_rx'] += len(bits)
                                                
                                                rx_success = True
                                                if args.verbose: print(f"  [RX] OK (rep={rep}) - BER: {bit_diff/len(bits):.2e}")
                                                break
                                    except:
                                        pass
                            except Exception as e:
                                if args.verbose: print(f"  [RX] Error: {e}")
                                
                        if rx_success: break
            
            print(f"[Result] {packet_rx_success}/{len(packets)} OK")
            if packet_rx_success == len(packets):
                stats['recovered_frames'] += 1
            
    except KeyboardInterrupt:
        print("\n[Test] Interrupted")
    finally:
        if tdd_ctrl:
            tdd_ctrl.disable()
        if args.mode != 'tdd' and args.loopback == 'digital':
             set_loopback_mode(args.ip, 0)
             
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        print(f"Frames Transmitted: {stats['total_frames']}")
        print(f"Frames Recovered:   {stats['recovered_frames']} (FER: {1.0 - stats['recovered_frames']/stats['total_frames']:.2%})")
        print("-" * 60)
        print(f"Packets Transmitted (Logical): {stats['total_packets']}")
        print(f"Packets Received (CRC OK):     {stats['rx_packets']} (PER: {1.0 - stats['rx_packets']/stats['total_packets']:.2%})")
        print(f"Physical TX Bursts:            {stats['tx_packets']}")
        print("-" * 60)
        if stats['total_bits_rx'] > 0:
            print(f"Total Bits Received: {stats['total_bits_rx']}")
            print(f"Bit Errors:          {stats['bit_errors']}")
            print(f"Post-FEC BER:        {stats['bit_errors']/stats['total_bits_rx']:.2e}")
        else:
            print("No bits received to calculate BER")
        print("=" * 60)

if __name__ == "__main__":
    main()
