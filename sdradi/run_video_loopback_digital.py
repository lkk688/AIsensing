#!/usr/bin/env python
"""
run_video_loopback_digital.py - SDR Video Loopback Test with Digital Loopback

Tests the video communication stack using AD9361's internal digital loopback
mode (BIST), which bypasses the RF frontend for near-perfect signal quality.

Usage:
    python run_video_loopback_digital.py --mode digital --max-frames 5
    python run_video_loopback_digital.py --mode rf --max-frames 5
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
# Digital Loopback Control
# =============================================================================

def set_loopback_mode(sdr_ip, mode):
    """
    Set AD9361 loopback mode.
    
    Args:
        sdr_ip: SDR IP address (e.g., 'ip:192.168.2.2')
        mode: 0 = disabled, 1 = digital TX->RX loopback, 2 = FPGA RX->TX
    """
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
    """Simple Digital AGC to normalize signal levels."""
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
# Loopback Test
# =============================================================================

def main():
    import cv2
    
    parser = argparse.ArgumentParser(description='SDR Video Loopback Test (Digital/RF)')
    parser.add_argument('--ip', default='ip:192.168.2.2', help='SDR IP address')
    parser.add_argument('--config', default='sdr_tuned_config.json', help='SDR config file')
    parser.add_argument('--mode', default='digital', choices=['digital', 'rf'], 
                        help='Loopback mode: digital (internal) or rf (cable)')
    parser.add_argument('--max-frames', type=int, default=5, help='Max frames to test')
    parser.add_argument('--quality', type=int, default=50, help='JPEG quality')
    parser.add_argument('--width', type=int, default=320, help='Frame width')
    parser.add_argument('--height', type=int, default=240, help='Frame height')
    parser.add_argument('--fec', type=str, default='none', choices=['ldpc', 'repetition', 'none'], help='FEC type')
    parser.add_argument('--packet-size', type=int, default=768, help='Packet size')
    parser.add_argument('--sync-threshold', type=float, default=30.0, help='Sync threshold')
    parser.add_argument('--tx-gain', type=int, default=0, help='TX gain override')
    parser.add_argument('--rx-gain', type=int, default=40, help='RX gain override')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"SDR Video Loopback Test - {args.mode.upper()} Mode")
    print("=" * 60)
    
    # Load Config
    config_file = args.config
    if os.path.exists(config_file):
        print(f"[Config] Loading from {config_file}")
        sdr_cfg = SDRConfig.load_from_json(config_file)
    else:
        print(f"[Config] Using defaults")
        sdr_cfg = SDRConfig()
    
    sdr_cfg.sdr_ip = args.ip
    sdr_cfg.rx_uri = None
    sdr_cfg.device = "pluto"
    sdr_cfg.tx_gain = args.tx_gain
    sdr_cfg.rx_gain = args.rx_gain
    
    print(f"[Config] SDR IP: {sdr_cfg.sdr_ip}")
    print(f"[Config] Mode: {args.mode.upper()}")
    print(f"[Config] TX Gain: {sdr_cfg.tx_gain} dB, RX Gain: {sdr_cfg.rx_gain} dB")
    
    # Enable digital loopback if requested
    if args.mode == 'digital':
        print("\n[SDR] Enabling digital loopback (AD9361 internal TX->RX)...")
        set_loopback_mode(args.ip, 1)
    else:
        print("\n[SDR] Using RF loopback (external cable)...")
        set_loopback_mode(args.ip, 0)
    
    # FEC Config
    fec_map = {
        "none": FECType.NONE,
        "repetition": FECType.REPETITION,
        "ldpc": FECType.LDPC
    }
    fec_type = fec_map.get(args.fec.lower(), FECType.NONE)
    fec_cfg = FECConfig(enabled=fec_type != FECType.NONE, fec_type=fec_type)
    
    # OFDM Config
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = args.sync_threshold
    
    print(f"[Config] FEC: {fec_type.value}")
    
    # Lower sync threshold for RF loopback (signal is weaker)
    if args.mode == 'rf':
        ofdm_cfg.sync_threshold = 15.0  # Lower for RF path losses
    else:
        ofdm_cfg.sync_threshold = 10.0  # Digital loopback is cleaner
    
    print(f"[Config] Sync Threshold: {ofdm_cfg.sync_threshold}")
    
    # Initialize SDR Link
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg)
    
    # Update video settings
    if args.packet_size:
        link.video_config.packet_size = int(args.packet_size)
    link.video_config.resolution = (args.width, args.height)
    
    # Connect to SDR
    print(f"\n[SDR] Connecting to {args.ip}...")
    if not link.connect_sdr():
        print("[SDR] ERROR: Failed to connect")
        return
    print("[SDR] Connected successfully!")
    
    # Get underlying adi device
    adi_dev = link.sdr.sdr
    
    # Calculate expected samples
    base_packet_bytes = link.video_config.packet_size
    expected_packet_bits = base_packet_bytes * 8
    
    # Calculate FEC output size
    if fec_cfg.enabled and fec_cfg.fec_type == FECType.LDPC and hasattr(link.fec_codec, 'n'):
        # LDPC: output is n bits per k-bit block
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
    
    # Set RX buffer size to accommodate larger LDPC packets
    rx_buffer_size = max(32768, min_capture * 2)
    adi_dev.rx_buffer_size = int(rx_buffer_size)
    if hasattr(adi_dev, "_rxadc") and hasattr(adi_dev._rxadc, "set_kernel_buffers_count"):
        adi_dev._rxadc.set_kernel_buffers_count(4)
    
    print(f"\n[PHY] Packet bits: {expected_packet_bits}, FEC bits: {expected_fec_bits}")
    print(f"[PHY] Min capture: {min_capture} samples, Preamble: {preamble_len}")
    print(f"[PHY] RX buffer: {rx_buffer_size} samples")
    
    # Initialize
    agc = DigitalAGC(target_level=0.5, alpha=0.1)
    payload_size = link.video_codec.config.packet_size - link.video_codec.HEADER_SIZE
    
    # Stats
    stats = {
        'tx_packets': 0,
        'rx_packets': 0,
        'crc_fails': 0,
        'frames_recovered': 0,
        'sync_attempts': 0,
        'sync_successes': 0,
    }
    
    print("\n" + "=" * 60)
    print("Starting Loopback Test...")
    print("=" * 60 + "\n")
    
    try:
        for frame_idx in range(args.max_frames):
            print(f"\n--- Frame {frame_idx + 1}/{args.max_frames} ---")
            
            # Generate test frame
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            x = (frame_idx * 20) % (args.width - 50)
            y = (frame_idx * 15) % (args.height - 50)
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)
            cv2.putText(frame, f"F{frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Encode frame
            packets = link.video_codec.encode_frame(frame, quality=args.quality)
            print(f"[TX] Encoded: {len(packets)} packets")
            
            # Process each packet
            packet_rx_success = 0
            
            for pkt_idx, (pkt_bytes, pkt_i) in enumerate(packets):
                # Convert to bits
                pkt_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                actual_packet_bits = len(bits)
                
                # FEC encode
                fec_bits = link.fec_codec.encode(bits)
                actual_fec_bits = len(fec_bits)
                
                # Calculate expected payload samples for THIS packet
                num_ofdm_frames = int(np.ceil(actual_fec_bits / bits_per_frame))
                actual_payload_samples = num_ofdm_frames * samples_per_frame
                
                # Modulate
                tx_signal = link.transceiver.modulate(fec_bits)
                preamble = link._generate_preamble()
                
                # Build complete signal - just preamble + payload (no zeros for cyclic mode)
                tx_signal_full = np.concatenate([preamble, tx_signal])
                
                # Normalize for DAC (scale to 14-bit range)
                max_val = np.max(np.abs(tx_signal_full))
                if max_val > 0:
                    tx_signal_full = tx_signal_full / max_val
                tx_signal_full = tx_signal_full.astype(np.complex64) * 2**14
                
                # Repeat signal for cyclic mode - need multiple copies for reliable sync
                min_buffer = 32768  # Minimum 32k samples for reliable operation
                tx_repeats = max(5, int(np.ceil(min_buffer / len(tx_signal_full))))
                tx_cyclic = np.tile(tx_signal_full, tx_repeats)
                
                # Transmit in CYCLIC mode
                adi_dev.tx_destroy_buffer()
                adi_dev.tx_cyclic_buffer = True
                adi_dev.tx(tx_cyclic)
                stats['tx_packets'] += 1
                
                # Wait for cyclic buffer to stabilize
                time.sleep(0.1)
                
                # Receive and process
                rx_success = False
                for rx_attempt in range(5):
                    try:
                        rx_samples = adi_dev.rx()
                        if isinstance(rx_samples, (tuple, list)):
                            rx_samples = rx_samples[0] if len(rx_samples) > 0 else rx_samples
                    except Exception as e:
                        continue
                    
                    if rx_samples is None or len(rx_samples) < min_capture:
                        continue
                    
                    # Normalize and AGC
                    max_val = np.max(np.abs(rx_samples))
                    if max_val > 0:
                        rx_norm = rx_samples / max_val
                    else:
                        rx_norm = rx_samples
                    rx_signal = agc.process(rx_norm)
                    
                    # Synchronize
                    stats['sync_attempts'] += 1
                    synced_sig, sync_metrics = link._synchronize(rx_signal)
                    
                    if not sync_metrics.get('sync_success'):
                        continue
                    
                    stats['sync_successes'] += 1
                    
                    if args.verbose:
                        print(f"  [Sync] Peak: {sync_metrics.get('peak_val', 0):.1f}, "
                              f"CFO: {sync_metrics.get('cfo_est', 0):.1f} Hz")
                    
                    # Get payload - use actual_payload_samples for this specific packet
                    if len(synced_sig) < actual_payload_samples:
                        if args.verbose:
                            print(f"    [Debug] synced_sig too short: {len(synced_sig)} < {actual_payload_samples}")
                        continue
                    
                    payload = synced_sig[:actual_payload_samples]
                    
                    # Demodulate
                    rx_fec_bits, met = link.transceiver.demodulate(payload)
                    
                    # FEC decode
                    try:
                        rx_bits_dec = link.fec_codec.decode(rx_fec_bits)
                    except Exception as e:
                        if args.verbose:
                            print(f"    [Debug] FEC decode failed: {e}")
                        continue
                    
                    # Use actual_packet_bits for this specific packet
                    if len(rx_bits_dec) < actual_packet_bits:
                        if args.verbose:
                            print(f"    [Debug] Decoded bits too short: {len(rx_bits_dec)} < {actual_packet_bits}")
                        continue
                    
                    rx_bits_dec = rx_bits_dec[:actual_packet_bits]
                    rx_bytes = np.packbits(rx_bits_dec).tobytes()
                    
                    # Parse packet
                    info = link.video_codec.parse_packet_header(rx_bytes)
                    
                    if info:
                        if args.verbose:
                            print(f"    [Debug] Parsed: w={info['width']}, h={info['height']}, "
                                  f"pkt={info['pkt_idx']}/{info['total_pkts']}, q={info['quality']}")
                        
                        # Verify CRC
                        calc_crc = zlib.crc32(info['payload']) & 0xFFFFFFFF
                        if calc_crc == info['crc']:
                            stats['rx_packets'] += 1
                            packet_rx_success += 1
                            rx_success = True
                            if args.verbose:
                                print(f"  [RX] Packet {info['pkt_idx']}/{info['total_pkts']} OK")
                            break
                        else:
                            stats['crc_fails'] += 1
                            if args.verbose:
                                print(f"    [Debug] CRC fail: calc={calc_crc:08x} vs pkt={info['crc']:08x}")
                
                # Stop cyclic TX
                adi_dev.tx_destroy_buffer()
            
            success_rate = 100*packet_rx_success/max(1, len(packets))
            print(f"[Result] TX: {len(packets)}, RX OK: {packet_rx_success}, Success: {success_rate:.1f}%")
            
            if packet_rx_success >= len(packets) // 2:  # At least 50% packets
                stats['frames_recovered'] += 1
        
    except KeyboardInterrupt:
        print("\n[Test] Interrupted")
    
    finally:
        # Stop TX and disable loopback
        try:
            adi_dev.tx_destroy_buffer()
        except:
            pass
        
        # Disable digital loopback
        if args.mode == 'digital':
            print("\n[SDR] Disabling digital loopback...")
            set_loopback_mode(args.ip, 0)
        
        print("\n" + "=" * 60)
        print(f"Loopback Test Results ({args.mode.upper()} Mode):")
        print(f"  TX Packets: {stats['tx_packets']}")
        print(f"  RX Packets: {stats['rx_packets']}")
        print(f"  CRC Failures: {stats['crc_fails']}")
        print(f"  Frames Recovered: {stats['frames_recovered']}/{args.max_frames}")
        if stats['sync_attempts'] > 0:
            print(f"  Sync Rate: {100*stats['sync_successes']/stats['sync_attempts']:.1f}%")
        if stats['tx_packets'] > 0:
            print(f"  Packet Success Rate: {100*stats['rx_packets']/stats['tx_packets']:.1f}%")
        print("=" * 60)

if __name__ == "__main__":
    main()
