#!/usr/bin/env python
"""
run_video_loopback_test.py - SDR Video Loopback Test

Tests the video communication stack using Pluto SDR's full-duplex capability.
TX and RX run in the same process, sharing the SDR device.

Usage:
    python run_video_loopback_test.py --ip ip:192.168.2.2 --max-frames 10
"""

import sys
import time
import numpy as np
import argparse
import json
import os
import zlib
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Qt Wayland issue
os.environ["QT_QPA_PLATFORM"] = "xcb"

from sdr_video_commv2 import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

# =============================================================================
# GF(256) Arithmetic for MAC-layer FEC
# =============================================================================

_GF256_EXP = np.zeros(512, dtype=np.uint8)
_GF256_LOG = np.zeros(256, dtype=np.int16)

def _gf256_init():
    x = 1
    for i in range(255):
        _GF256_EXP[i] = x
        _GF256_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11d
    for i in range(255, 512):
        _GF256_EXP[i] = _GF256_EXP[i - 255]

_gf256_init()

def gf256_pow(a, p):
    if p == 0:
        return 1
    if a == 0:
        return 0
    return int(_GF256_EXP[(int(_GF256_LOG[a]) * int(p)) % 255])

def gf256_mul_vec(vec, coeff):
    if coeff == 0:
        return np.zeros_like(vec)
    if coeff == 1:
        return vec.copy()
    log_coeff = int(_GF256_LOG[coeff])
    logs = _GF256_LOG[vec]
    res = _GF256_EXP[(logs + log_coeff) % 255]
    res = res.astype(np.uint8, copy=False)
    res[vec == 0] = 0
    return res

def gf256_inv(a):
    if a == 0:
        return 0
    return int(_GF256_EXP[255 - int(_GF256_LOG[a])])

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
    
    parser = argparse.ArgumentParser(description='SDR Video Loopback Test')
    parser.add_argument('--ip', default='ip:192.168.2.2', help='SDR IP address')
    parser.add_argument('--config', default='sdr_tuned_config.json', help='SDR config file')
    parser.add_argument('--max-frames', type=int, default=10, help='Max frames to test')
    parser.add_argument('--quality', type=int, default=50, help='JPEG quality')
    parser.add_argument('--width', type=int, default=320, help='Frame width')
    parser.add_argument('--height', type=int, default=240, help='Frame height')
    parser.add_argument('--fec', type=str, default='none', choices=['ldpc', 'repetition', 'none'], help='FEC type')
    parser.add_argument('--mac-fec-parity', type=int, default=2, help='MAC-layer parity packets')
    parser.add_argument('--header-repeat-bytes', type=int, default=12, help='Header repeat bytes')
    parser.add_argument('--packet-size', type=int, default=768, help='Packet size')
    parser.add_argument('--sync-threshold', type=float, default=30.0, help='Sync threshold')
    parser.add_argument('--tx-gain', type=int, default=None, help='TX gain override')
    parser.add_argument('--rx-gain', type=int, default=None, help='RX gain override')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDR Video Loopback Test")
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
    
    if args.tx_gain is not None:
        sdr_cfg.tx_gain = args.tx_gain
    if args.rx_gain is not None:
        sdr_cfg.rx_gain = args.rx_gain
    
    print(f"[Config] SDR IP: {sdr_cfg.sdr_ip}")
    print(f"[Config] TX Gain: {sdr_cfg.tx_gain} dB, RX Gain: {sdr_cfg.rx_gain} dB")
    print(f"[Config] Sample Rate: {sdr_cfg.fs/1e6:.2f} MHz")
    print(f"[Config] Carrier Freq: {sdr_cfg.fc/1e6:.2f} MHz")
    
    # FEC Config - use none for simpler debugging
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
    print(f"[Config] Sync Threshold: {args.sync_threshold}")
    
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
    
    # Setup RX buffer
    rx_buffer_size = int(sdr_cfg.rx_buffer_size)
    if hasattr(link.sdr, 'SDR_RX_setup'):
        link.sdr.SDR_RX_setup(n_SAMPLES=rx_buffer_size)
    
    # Parameters
    repeat_len = max(0, int(args.header_repeat_bytes))
    mac_fec_parity = int(args.mac_fec_parity)
    mac_fec_seed = 1
    payload_size = link.video_config.packet_size - link.video_codec.HEADER_SIZE
    
    # Calculate expected samples
    base_packet_bytes = link.video_config.packet_size
    packet_bytes = base_packet_bytes + repeat_len
    expected_packet_bits = packet_bytes * 8
    
    if fec_cfg.enabled and fec_cfg.fec_type == FECType.REPETITION:
        expected_fec_bits = expected_packet_bits * fec_cfg.repetitions
    else:
        expected_fec_bits = expected_packet_bits
    
    bits_per_frame = link.ofdm_config.bits_per_frame
    samples_per_frame = link.ofdm_config.samples_per_frame
    num_frames = int(np.ceil(expected_fec_bits / bits_per_frame))
    expected_payload_samples = num_frames * samples_per_frame
    preamble_len = len(link._generate_preamble())
    min_capture = preamble_len + expected_payload_samples
    
    print(f"\n[PHY] Packet bits: {expected_packet_bits}, FEC bits: {expected_fec_bits}")
    print(f"[PHY] Min capture: {min_capture} samples, Preamble: {preamble_len}")
    
    # Initialize
    agc = DigitalAGC(target_level=0.5, alpha=0.1)
    allowed_set = {(args.width, args.height)}
    max_total_pkts = 60 + mac_fec_parity
    
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
            
            # Build final packets with MAC-FEC
            final_packets = []
            frame_info = None
            
            if mac_fec_parity > 0 and len(packets) > 0:
                data_total = len(packets)
                total_pkts = data_total + mac_fec_parity
                data_payloads = []
                
                for pkt_bytes, pkt_i in packets:
                    payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                    info = link.video_codec.parse_packet_header(pkt_bytes)
                    if info and frame_info is None:
                        frame_info = info
                    data_payloads.append(payload)
                
                if frame_info:
                    for pkt_bytes, pkt_i in packets:
                        payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                        rebuilt = link.video_codec.create_packet_header(
                            payload, frame_info['frame_id'], pkt_i, total_pkts, frame_info['quality']
                        )
                        final_packets.append(rebuilt)
                    
                    # Generate parity packets
                    base = (mac_fec_seed + frame_info['frame_id']) % 255
                    if base == 0:
                        base = 1
                    
                    for p_idx in range(mac_fec_parity):
                        alpha = (base + p_idx) % 255
                        if alpha == 0:
                            alpha = 1
                        coeffs = np.fromiter(
                            (gf256_pow(alpha, i) for i in range(data_total)),
                            dtype=np.uint8, count=data_total
                        )
                        parity_payload = np.zeros(payload_size, dtype=np.uint8)
                        for i in range(data_total):
                            coeff = int(coeffs[i])
                            if coeff:
                                dp = np.frombuffer(data_payloads[i], dtype=np.uint8)
                                if len(dp) < payload_size:
                                    dp = np.pad(dp, (0, payload_size - len(dp)))
                                parity_payload ^= gf256_mul_vec(dp, coeff)
                        
                        parity_pkt = link.video_codec.create_packet_header(
                            parity_payload.tobytes(),
                            frame_info['frame_id'],
                            data_total + p_idx,
                            total_pkts,
                            frame_info['quality']
                        )
                        final_packets.append(parity_pkt)
            else:
                final_packets = [p[0] for p in packets]
            
            print(f"[TX] Total packets (with parity): {len(final_packets)}")
            
            # Transmit and receive each packet
            packet_rx_success = 0
            
            for pkt_idx, pkt_bytes in enumerate(final_packets):
                # Prepare wire packet
                if repeat_len > 0:
                    header = pkt_bytes[:link.video_codec.HEADER_SIZE]
                    payload = pkt_bytes[link.video_codec.HEADER_SIZE:]
                    pkt_wire = header + header[:repeat_len] + payload
                else:
                    pkt_wire = pkt_bytes
                
                # Convert to bits
                pkt_arr = np.frombuffer(pkt_wire, dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                
                # FEC encode
                fec_bits = link.fec_codec.encode(bits)
                
                # Modulate
                tx_signal = link.transceiver.modulate(fec_bits)
                preamble = link._generate_preamble()
                
                # Build complete signal with leading/trailing zeros for timing margin
                leading_zeros = np.zeros(500, dtype=np.complex64)
                trailing_zeros = np.zeros(500, dtype=np.complex64)
                tx_signal = np.concatenate([leading_zeros, preamble, tx_signal, trailing_zeros])
                
                # Normalize for DAC (scale to 14-bit range)
                max_val = np.max(np.abs(tx_signal))
                if max_val > 0:
                    tx_signal = tx_signal / max_val
                tx_signal = tx_signal.astype(np.complex64) * 2**14
                
                # Get underlying adi device
                adi_dev = link.sdr.sdr
                
                # Transmit in CYCLIC mode for loopback (directly using adi device)
                adi_dev.tx_destroy_buffer()
                adi_dev.tx_cyclic_buffer = True
                adi_dev.tx(tx_signal)
                stats['tx_packets'] += 1
                
                # Wait for cyclic buffer to stabilize
                time.sleep(0.05)
                
                # Receive multiple times to find the packet
                rx_success = False
                for rx_attempt in range(5):  # Increased attempts
                    try:
                        # Use adi device directly for RX too
                        rx_samples = adi_dev.rx()
                        
                        if isinstance(rx_samples, tuple) or isinstance(rx_samples, list):
                            rx_samples = rx_samples[0] if len(rx_samples) > 0 else rx_samples
                    except Exception as e:
                        if args.verbose:
                            print(f"[RX] Error: {e}")
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
                    
                    # Get payload
                    payload_start = sync_metrics.get('payload_start', None)
                    if payload_start is None:
                        if args.verbose:
                            print(f"    [Debug] No payload_start in sync_metrics")
                        continue
                    
                    if len(synced_sig) < expected_payload_samples:
                        if args.verbose:
                            print(f"    [Debug] synced_sig too short: {len(synced_sig)} < {expected_payload_samples}")
                        continue
                    
                    payload = synced_sig[:expected_payload_samples]
                    
                    # Demodulate
                    rx_fec_bits, met = link.transceiver.demodulate(payload)
                    if args.verbose:
                        print(f"    [Debug] Demodulated {len(rx_fec_bits)} bits")
                    
                    # FEC decode
                    try:
                        rx_bits_dec = link.fec_codec.decode(rx_fec_bits)
                        if args.verbose:
                            print(f"    [Debug] FEC decoded {len(rx_bits_dec)} bits")
                    except Exception as e:
                        if args.verbose:
                            print(f"    [Debug] FEC decode error: {e}")
                        continue
                    
                    if len(rx_bits_dec) < expected_packet_bits:
                        if args.verbose:
                            print(f"    [Debug] Decoded bits too short: {len(rx_bits_dec)} < {expected_packet_bits}")
                        continue
                    
                    rx_bits_dec = rx_bits_dec[:expected_packet_bits]
                    rx_bytes = np.packbits(rx_bits_dec).tobytes()
                    
                    if len(rx_bytes) < packet_bytes:
                        continue
                    rx_bytes = rx_bytes[:packet_bytes]
                    
                    if args.verbose:
                        print(f"    [Debug] rx_bytes length: {len(rx_bytes)}, packet_bytes: {packet_bytes}")
                        print(f"    [Debug] First 24 bytes (hex): {rx_bytes[:24].hex()}")
                    
                    # Header scanning
                    info = None
                    for shift in range(0, repeat_len + 1):
                        header_start = shift
                        header_end = header_start + link.video_codec.HEADER_SIZE
                        repeat_start = header_end
                        repeat_end = repeat_start + repeat_len
                        payload_start_byte = repeat_end
                        payload_end_byte = payload_start_byte + payload_size
                        
                        if payload_end_byte > len(rx_bytes):
                            break
                        
                        header_bytes = rx_bytes[header_start:header_end]
                        
                        if repeat_len > 0:
                            repeat_bytes = rx_bytes[repeat_start:repeat_end]
                            if repeat_bytes != header_bytes[:repeat_len]:
                                if args.verbose and shift == 0:
                                    print(f"    [Debug] Header repeat mismatch at shift {shift}")
                                continue
                        
                        payload_bytes = rx_bytes[payload_start_byte:payload_end_byte]
                        candidate = header_bytes + payload_bytes
                        candidate_info = link.video_codec.parse_packet_header(candidate)
                        
                        if candidate_info:
                            if args.verbose:
                                print(f"    [Debug] Parsed: w={candidate_info['width']}, h={candidate_info['height']}, "
                                      f"pkt={candidate_info['pkt_idx']}/{candidate_info['total_pkts']}, "
                                      f"q={candidate_info['quality']}, frame={candidate_info['frame_id']}")
                            
                            if (candidate_info['width'], candidate_info['height']) not in allowed_set:
                                if args.verbose:
                                    print(f"    [Debug] Resolution mismatch: ({candidate_info['width']}, {candidate_info['height']})")
                                continue
                            if candidate_info['total_pkts'] <= 0 or candidate_info['total_pkts'] > max_total_pkts:
                                if args.verbose:
                                    print(f"    [Debug] total_pkts out of range: {candidate_info['total_pkts']}")
                                continue
                            
                            # Verify CRC
                            calc_crc = zlib.crc32(candidate_info['payload']) & 0xFFFFFFFF
                            if calc_crc == candidate_info['crc']:
                                info = candidate_info
                                break
                            else:
                                stats['crc_fails'] += 1
                                if args.verbose:
                                    print(f"    [Debug] CRC mismatch: calc={calc_crc:08x} vs pkt={candidate_info['crc']:08x}")
                        elif args.verbose and shift == 0:
                            print(f"    [Debug] parse_packet_header returned None at shift {shift}")
                    
                    if info:
                        stats['rx_packets'] += 1
                        packet_rx_success += 1
                        rx_success = True
                        if args.verbose:
                            print(f"  [RX] Packet {info['pkt_idx']}/{info['total_pkts']} OK (Frame {info['frame_id']})")
                        break  # Success, move to next packet
                
                # Stop cyclic TX for this packet
                adi_dev.tx_destroy_buffer()
            
            print(f"[Result] TX: {len(final_packets)}, RX OK: {packet_rx_success}, "
                  f"Success: {100*packet_rx_success/max(1, len(final_packets)):.1f}%")
            
            if packet_rx_success >= len(final_packets) - mac_fec_parity:
                stats['frames_recovered'] += 1
        
    except KeyboardInterrupt:
        print("\n[Test] Interrupted")
    
    finally:
        # Stop TX
        try:
            link.sdr.sdr.tx_destroy_buffer()
            link.sdr.SDR_TX_stop()
        except:
            pass
        
        print("\n" + "=" * 60)
        print("Loopback Test Results:")
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

