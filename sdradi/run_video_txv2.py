#!/usr/bin/env python
"""
run_video_txv2.py - SDR Video Transmitter with improved PHY/MAC

This script ports the working TX logic from sim_video_e2e_asyncv3.py to real Pluto SDR hardware.

Key Features:
- LDPC FEC encoding
- MAC-layer FEC with GF(256) parity packets  
- Header redundancy for bit-alignment recovery
- Proper signal normalization for DAC

Usage:
    python run_video_txv2.py --ip ip:192.168.2.2 --video test_video.mp4
"""

import sys
import time
import numpy as np
import argparse
import json
import datetime
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# =============================================================================
# Logging
# =============================================================================

LOG_FILE = "tx_debug_v2.jsonl"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        if isinstance(obj, complex):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

def log_event(event_type, data):
    """Log an event to the JSONL file."""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "data": data
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")

# =============================================================================
# Main TX Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SDR Video TX v2 (Improved PHY/MAC)')
    
    # SDR Config
    parser.add_argument('--ip', default='ip:192.168.2.2', help='SDR IP address')
    parser.add_argument('--config', default='sdr_tuned_config.json', help='SDR config file')
    
    # Video Config
    parser.add_argument('--video', default='test_video.mp4', help='Video file path')
    parser.add_argument('--quality', type=int, default=32, help='JPEG Quality (1-100)')
    parser.add_argument('--width', type=int, default=320, help='Frame width')
    parser.add_argument('--height', type=int, default=240, help='Frame height')
    parser.add_argument('--test-pattern', action='store_true', help='Generate test pattern instead of file')
    parser.add_argument('--max-frames', type=int, default=120, help='Max frames to transmit')
    parser.add_argument('--frame-step', type=int, default=1, help='Transmit every Nth source frame')
    
    # PHY Config
    parser.add_argument('--fec', type=str, default='ldpc', choices=['ldpc', 'repetition', 'none'], 
                        help='FEC type')
    parser.add_argument('--waveform', choices=['ofdm', 'otfs'], default='ofdm', help='Waveform Type')
    parser.add_argument('--tx-gain', type=int, default=None, help='TX Gain Override (dB)')
    
    # MAC Config
    parser.add_argument('--mac-fec-parity', type=int, default=2, help='MAC-layer parity packets per frame')
    parser.add_argument('--mac-fec-seed', type=int, default=1, help='MAC-layer parity seed')
    parser.add_argument('--header-repeat-bytes', type=int, default=12, help='Header bytes to repeat for alignment')
    parser.add_argument('--packet-size', type=int, default=768, help='Packet size in bytes')
    parser.add_argument('--max-packets-per-frame', type=int, default=60, help='Max packets per frame')
    
    # Timing
    parser.add_argument('--pkt-repeats', type=int, default=1, help='Repeat each packet N times')
    parser.add_argument('--sleep-factor', type=float, default=0.5, help='TX pacing factor')
    parser.add_argument('--inter-packet-delay', type=float, default=0.01, help='Delay between packets (seconds)')
    parser.add_argument('--frame-gap-ms', type=int, default=50, help='Gap between frames (ms)')
    
    # Debug
    parser.add_argument('--log-interval', type=float, default=2.0, help='Stats log interval')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDR Video TX v2 - Improved PHY/MAC Stack")
    print("=" * 60)
    
    # Load/Create Config
    config_file = args.config
    if not os.path.exists(config_file):
        print(f"[Config] Creating default config: {config_file}")
        with open(config_file, 'w') as f:
            json.dump({
                "sdr_ip": "ip:192.168.2.2",
                "device": "pluto",
                "fc": 915e6,
                "fs": 2e6,
                "bandwidth": 2e6,
                "tx_gain": -10,
                "rx_gain": 40,
                "rx_buffer_size": 262144
            }, f, indent=2)
    
    # Load SDR Config
    print(f"[Config] Loading from {config_file}")
    sdr_cfg = SDRConfig.load_from_json(config_file)
    sdr_cfg.sdr_ip = args.ip
    sdr_cfg.rx_uri = None  # Single device mode for TX
    sdr_cfg.device = "pluto"
    
    if args.tx_gain is not None:
        sdr_cfg.tx_gain = args.tx_gain
    
    print(f"[Config] SDR IP: {sdr_cfg.sdr_ip}")
    print(f"[Config] TX Gain: {sdr_cfg.tx_gain} dB")
    print(f"[Config] Sample Rate: {sdr_cfg.fs/1e6:.2f} MHz")
    print(f"[Config] Carrier Freq: {sdr_cfg.fc/1e6:.2f} MHz")
    
    # FEC Config
    fec_map = {
        "none": FECType.NONE,
        "repetition": FECType.REPETITION,
        "ldpc": FECType.LDPC
    }
    fec_type = fec_map.get(args.fec.lower(), FECType.LDPC)
    fec_cfg = FECConfig(enabled=fec_type != FECType.NONE, fec_type=fec_type)
    
    print(f"[Config] FEC Type: {fec_type.value}")
    print(f"[Config] MAC FEC Parity: {args.mac_fec_parity}")
    print(f"[Config] Header Repeat: {args.header_repeat_bytes} bytes")
    
    # Initialize SDR Link
    waveform_enum = WaveformType.OTFS if args.waveform == 'otfs' else WaveformType.OFDM
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, waveform=waveform_enum)
    
    # Update video codec settings
    if args.packet_size:
        link.video_config.packet_size = int(args.packet_size)
    if args.max_packets_per_frame:
        link.video_config.max_packets_per_frame = int(args.max_packets_per_frame)
    link.video_config.resolution = (args.width, args.height)
    
    # Connect to SDR
    print(f"\n[SDR] Connecting to {args.ip}...")
    if not link.connect_sdr():
        print("[SDR] ERROR: Failed to connect to SDR")
        return
    print("[SDR] Connected successfully!")
    
    # Log Configuration
    log_event("CONFIG", {
        "sdr_config": sdr_cfg.__dict__,
        "args": vars(args)
    })
    
    # Open Video Source
    cap = None
    if not args.test_pattern:
        import cv2
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[Video] Warning: Cannot open {args.video}, using test pattern")
            cap = None
    
    if cap is None:
        print("[Video] Using synthetic test pattern")
    else:
        print(f"[Video] Source: {args.video}")
    
    print(f"[Video] Resolution: {args.width}x{args.height}")
    print(f"[Video] Quality: {args.quality}")
    print(f"[Video] Max Frames: {args.max_frames}")
    
    # TX Parameters
    repeat_len = max(0, int(args.header_repeat_bytes))
    mac_fec_parity = int(args.mac_fec_parity)
    mac_fec_seed = int(args.mac_fec_seed)
    payload_size = link.video_config.packet_size - link.video_codec.HEADER_SIZE
    
    print("\n" + "=" * 60)
    print("Starting Transmission...")
    print("=" * 60 + "\n")
    
    frame_idx = 0
    src_frame_idx = 0
    tx_packets = 0
    last_log = time.time()
    
    try:
        while frame_idx < args.max_frames:
            # Capture Frame
            if cap:
                import cv2
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                src_frame_idx += 1
                if args.frame_step > 1 and (src_frame_idx % args.frame_step) != 1:
                    continue
                frame = cv2.resize(frame, (args.width, args.height))
            else:
                # Generate test pattern
                import cv2
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                # Moving rectangle
                x = (frame_idx * 5) % (args.width - 40)
                y = (frame_idx * 3) % (args.height - 40)
                cv2.rectangle(frame, (x, y), (x + 40, y + 40), (0, 255, 0), -1)
                cv2.putText(frame, f"F{frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Encode Frame to Packets
            packets = link.video_codec.encode_frame(frame, quality=args.quality)
            
            if len(packets) == 0:
                print(f"[TX] Warning: Frame {frame_idx} encoded to 0 packets")
                continue
            
            # Apply MAC-layer FEC (add parity packets)
            final_packets = []
            frame_info = None
            
            if mac_fec_parity > 0:
                data_total = len(packets)
                parity_count = mac_fec_parity
                total_pkts = data_total + parity_count
                data_payloads = []
                
                for pkt_bytes, pkt_i in packets:
                    payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                    info = link.video_codec.parse_packet_header(pkt_bytes)
                    if info and frame_info is None:
                        frame_info = info
                    data_payloads.append(payload)
                
                if frame_info:
                    # Rebuild data packets with updated total_pkts count
                    for pkt_bytes, pkt_i in packets:
                        payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                        rebuilt = link.video_codec.create_packet_header(
                            payload, frame_info['frame_id'], pkt_i, total_pkts, frame_info['quality']
                        )
                        final_packets.append(rebuilt)
                    
                    # Generate parity packets
                    base = (int(mac_fec_seed) + int(frame_info['frame_id'])) % 255
                    if base == 0:
                        base = 1
                    
                    for p_idx in range(parity_count):
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
            
            # Transmit each packet
            for pkt_bytes in final_packets:
                # Add header repeat for alignment slack
                if repeat_len > 0:
                    header = pkt_bytes[:link.video_codec.HEADER_SIZE]
                    payload = pkt_bytes[link.video_codec.HEADER_SIZE:]
                    header_repeat = header[:repeat_len]
                    pkt_wire = header + header_repeat + payload
                else:
                    pkt_wire = pkt_bytes
                
                # Convert to bits
                pkt_arr = np.frombuffer(pkt_wire, dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                
                # FEC Encode
                fec_bits = link.fec_codec.encode(bits)
                
                # Transmit (with preamble)
                for rep in range(max(1, args.pkt_repeats)):
                    tx_signal = link.transmit(fec_bits)
                    tx_packets += 1
                    
                    # Pacing
                    if args.inter_packet_delay > 0:
                        time.sleep(args.inter_packet_delay)
            
            frame_idx += 1
            
            # Log progress
            now = time.time()
            if now - last_log >= args.log_interval:
                print(f"[TX] Frame: {frame_idx}/{args.max_frames}, Packets: {tx_packets}")
                last_log = now
                
                log_event("TX_PROGRESS", {
                    "frame_idx": frame_idx,
                    "tx_packets": tx_packets
                })
            
            # Frame gap
            if args.frame_gap_ms > 0:
                time.sleep(args.frame_gap_ms / 1000.0)
    
    except KeyboardInterrupt:
        print("\n[TX] Interrupted by user")
    
    finally:
        print("\n" + "=" * 60)
        print(f"[TX] Finished: {frame_idx} frames, {tx_packets} packets")
        print("=" * 60)
        
        if cap:
            cap.release()
        
        # Stop TX
        try:
            link.sdr.SDR_TX_stop()
        except:
            pass
        
        log_event("TX_COMPLETE", {
            "frames": frame_idx,
            "packets": tx_packets
        })

if __name__ == "__main__":
    main()
