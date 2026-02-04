#!/usr/bin/env python
"""
run_video_rxv2.py - SDR Video Receiver with improved PHY/MAC

This script ports the working RX logic from sim_video_e2e_asyncv3.py to real Pluto SDR hardware.

Key Features:
- LDPC FEC decoding
- Digital AGC for signal normalization
- Robust synchronization with CFO correction
- Header scanning with bit-alignment recovery
- MAC-layer FEC recovery for missing packets
- Frame assembly with timeout-based expiration

Usage:
    python run_video_rxv2.py --ip ip:192.168.2.2 --output received_video.avi
"""

import sys
import time
import numpy as np
import argparse
import json
import datetime
import os
import zlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Qt Wayland issue
os.environ["QT_QPA_PLATFORM"] = "xcb"

from sdr_video_commv2 import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

# =============================================================================
# GF(256) Arithmetic for MAC-layer FEC Recovery
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

def gf256_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return int(_GF256_EXP[int(_GF256_LOG[a]) + int(_GF256_LOG[b])])

def gf256_inv(a):
    if a == 0:
        return 0
    return int(_GF256_EXP[255 - int(_GF256_LOG[a])])

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
        
        # Hybrid update
        ideal_gain = self.target / curr_amp
        self.gain = (1 - self.alpha) * self.gain + self.alpha * ideal_gain
        
        # Limits
        if self.gain > 10000.0: 
            self.gain = 10000.0
        if self.gain < 0.01: 
            self.gain = 0.01
        
        return signal * self.gain

# =============================================================================
# Logging
# =============================================================================

LOG_FILE = "rx_debug_v2.jsonl"

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
# MAC FEC Recovery Functions
# =============================================================================

def build_coeffs(total_data, parity_idx, frame_id, mac_fec_seed):
    """Build GF(256) coefficients for MAC-FEC recovery."""
    base = (int(mac_fec_seed) + int(frame_id)) % 255
    if base == 0:
        base = 1
    alpha = (base + parity_idx) % 255
    if alpha == 0:
        alpha = 1
    coeffs = np.fromiter((gf256_pow(alpha, i) for i in range(total_data)), dtype=np.uint8, count=total_data)
    return coeffs

def solve_missing(missing_indices, parity_indices, data_total, frame_id, payloads, payload_size, mac_fec_seed):
    """Solve for missing packets using GF(256) Gaussian elimination."""
    m = len(missing_indices)
    if m == 0 or len(parity_indices) < m:
        return {}
    
    A = np.zeros((m, m), dtype=np.uint8)
    B = [None] * m
    
    for row in range(m):
        parity_idx = parity_indices[row]
        coeffs = build_coeffs(data_total, parity_idx - data_total, frame_id, mac_fec_seed)
        eq = np.frombuffer(payloads[parity_idx], dtype=np.uint8).copy()
        
        for i in range(data_total):
            coeff = int(coeffs[i])
            if coeff and i in payloads:
                eq ^= gf256_mul_vec(np.frombuffer(payloads[i], dtype=np.uint8), coeff)
        
        for col, miss_idx in enumerate(missing_indices):
            A[row, col] = coeffs[miss_idx]
        B[row] = eq
    
    # Gaussian elimination
    for col in range(m):
        pivot = None
        for row in range(col, m):
            if A[row, col]:
                pivot = row
                break
        if pivot is None:
            return None
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            B[col], B[pivot] = B[pivot], B[col]
        
        inv_pivot = gf256_inv(int(A[col, col]))
        if inv_pivot == 0:
            return None
        
        A[col] = gf256_mul_vec(A[col], inv_pivot)
        B[col] = gf256_mul_vec(B[col], inv_pivot)
        
        for row in range(m):
            if row != col and A[row, col]:
                factor = int(A[row, col])
                A[row] ^= gf256_mul_vec(A[col], factor)
                B[row] ^= gf256_mul_vec(B[col], factor)
    
    recovered = {}
    for i, miss_idx in enumerate(missing_indices):
        recovered[miss_idx] = B[i].tobytes()
    
    return recovered

# =============================================================================
# Main RX Function
# =============================================================================

def main():
    import cv2
    
    parser = argparse.ArgumentParser(description='SDR Video RX v2 (Improved PHY/MAC)')
    
    # SDR Config
    parser.add_argument('--ip', default='ip:192.168.2.2', help='SDR IP address')
    parser.add_argument('--config', default='sdr_tuned_config.json', help='SDR config file')
    
    # Output Config
    parser.add_argument('--output', default='received_video_v2.avi', help='Output video file')
    parser.add_argument('--width', type=int, default=320, help='Frame width')
    parser.add_argument('--height', type=int, default=240, help='Frame height')
    parser.add_argument('--output-fps', type=float, default=10.0, help='Output video FPS')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    parser.add_argument('--save-frames-dir', type=str, default='', help='Save frames to directory')
    
    # PHY Config
    parser.add_argument('--fec', type=str, default='ldpc', choices=['ldpc', 'repetition', 'none'], 
                        help='FEC type')
    parser.add_argument('--waveform', choices=['ofdm', 'otfs'], default='ofdm', help='Waveform Type')
    parser.add_argument('--sync-threshold', type=float, default=30.0, help='Sync correlation threshold')
    parser.add_argument('--rx-gain', type=int, default=None, help='RX Gain Override (dB)')
    
    # MAC Config
    parser.add_argument('--mac-fec-parity', type=int, default=2, help='MAC-layer parity packets per frame')
    parser.add_argument('--mac-fec-seed', type=int, default=1, help='MAC-layer parity seed')
    parser.add_argument('--header-repeat-bytes', type=int, default=12, help='Header repeat bytes for alignment')
    parser.add_argument('--packet-size', type=int, default=768, help='Packet size in bytes')
    parser.add_argument('--max-packets-per-frame', type=int, default=60, help='Max packets per frame')
    
    # Timing
    parser.add_argument('--frame-timeout', type=float, default=5.0, help='Frame buffer timeout (seconds)')
    parser.add_argument('--max-runtime', type=float, default=0, help='Max runtime in seconds (0=infinite)')
    
    # Debug
    parser.add_argument('--log-interval', type=float, default=2.0, help='Stats log interval')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--diag-level', type=str, default='basic', 
                        choices=['basic', 'phy', 'mac', 'all'], help='Diagnostic level')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDR Video RX v2 - Improved PHY/MAC Stack")
    print("=" * 60)
    
    # Load Config
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
    sdr_cfg.rx_uri = None  # Single device mode
    sdr_cfg.device = "pluto"
    
    if args.rx_gain is not None:
        sdr_cfg.rx_gain = args.rx_gain
    
    print(f"[Config] SDR IP: {sdr_cfg.sdr_ip}")
    print(f"[Config] RX Gain: {sdr_cfg.rx_gain} dB")
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
    
    # OFDM Config with sync threshold
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = args.sync_threshold
    
    print(f"[Config] FEC Type: {fec_type.value}")
    print(f"[Config] Sync Threshold: {args.sync_threshold}")
    print(f"[Config] MAC FEC Parity: {args.mac_fec_parity}")
    print(f"[Config] Header Repeat: {args.header_repeat_bytes} bytes")
    
    # Initialize SDR Link
    waveform_enum = WaveformType.OTFS if args.waveform == 'otfs' else WaveformType.OFDM
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg, waveform=waveform_enum)
    
    # Update video codec settings
    if args.packet_size:
        link.video_config.packet_size = int(args.packet_size)
    if args.max_packets_per_frame:
        link.video_config.max_packets_per_frame = int(args.max_packets_per_frame)
    
    # Connect to SDR
    print(f"\n[SDR] Connecting to {args.ip}...")
    if not link.connect_sdr():
        print("[SDR] ERROR: Failed to connect to SDR")
        return
    print("[SDR] Connected successfully!")
    
    # Setup RX buffer
    rx_buffer_size = int(sdr_cfg.rx_buffer_size)
    if hasattr(link.sdr, 'SDR_RX_setup'):
        link.sdr.SDR_RX_setup(n_SAMPLES=rx_buffer_size)
    
    # Log Configuration
    log_event("CONFIG", {
        "sdr_config": sdr_cfg.__dict__,
        "args": vars(args)
    })
    
    # Initialize AGC
    agc = DigitalAGC(target_level=0.5, alpha=0.1)
    
    # Calculate expected packet structure
    base_packet_bytes = link.video_config.packet_size
    repeat_len = max(0, int(args.header_repeat_bytes))
    packet_bytes = base_packet_bytes + repeat_len
    expected_packet_bits = packet_bytes * 8
    payload_size = link.video_codec.config.packet_size - link.video_codec.HEADER_SIZE
    max_total_pkts = link.video_config.max_packets_per_frame + (int(args.mac_fec_parity) if args.mac_fec_parity else 0)
    
    # Calculate expected FEC bits
    if fec_cfg.enabled:
        if fec_cfg.fec_type == FECType.LDPC and hasattr(link.fec_codec, 'k'):
            blocks = int(np.ceil(expected_packet_bits / link.fec_codec.k))
            expected_fec_bits = blocks * link.fec_codec.n
        elif fec_cfg.fec_type == FECType.REPETITION:
            expected_fec_bits = expected_packet_bits * fec_cfg.repetitions
        else:
            expected_fec_bits = expected_packet_bits
    else:
        expected_fec_bits = expected_packet_bits
    
    # Calculate sample requirements
    if link.waveform == WaveformType.OTFS:
        bits_per_frame = link.otfs_config.bits_per_frame
        samples_per_frame = link.otfs_config.N_delay * link.otfs_config.N_doppler
    else:
        bits_per_frame = link.ofdm_config.bits_per_frame
        samples_per_frame = link.ofdm_config.samples_per_frame
    
    num_frames = int(np.ceil(expected_fec_bits / bits_per_frame))
    expected_payload_samples = num_frames * samples_per_frame
    preamble_len = len(link._generate_preamble())
    min_capture = preamble_len + expected_payload_samples
    
    print(f"\n[PHY] Expected packet bits: {expected_packet_bits}")
    print(f"[PHY] Expected FEC bits: {expected_fec_bits}")
    print(f"[PHY] Min capture samples: {min_capture}")
    print(f"[PHY] Preamble length: {preamble_len}")
    
    # Create output directory
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)
    
    # Initialize video writer
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MJPG'), 
                          float(args.output_fps), (args.width, args.height))
    
    # Allowed resolutions for validation
    allowed_set = {(args.width, args.height)}
    
    # Create display window
    if not args.headless:
        cv2.namedWindow("SDR Video RX v2", cv2.WINDOW_NORMAL)
    
    print("\n" + "=" * 60)
    print("Starting Reception...")
    print("=" * 60 + "\n")
    
    # Stats
    stats = {
        'rx_packets': 0,
        'crc_fails': 0,
        'header_invalid': 0,
        'frames_recovered': 0,
        'frames_started': 0,
        'frames_expired': 0,
        'sync_attempts': 0,
        'sync_successes': 0,
        'mac_fec_recovered_packets': 0,
    }
    
    frame_buffer = {}
    recovered_ids = set()
    rx_stream = np.array([], dtype=complex)
    
    last_log = time.time()
    start_time = time.time()
    
    try:
        while True:
            # Check max runtime
            if args.max_runtime > 0 and (time.time() - start_time) > args.max_runtime:
                print(f"\n[RX] Max runtime reached ({args.max_runtime}s)")
                break
            
            # Receive samples from SDR
            try:
                if hasattr(link.sdr, 'SDR_RX_receive'):
                    rx_samples = link.sdr.SDR_RX_receive(normalize=False)
                else:
                    rx_samples = link.sdr.sdr.rx()
                
                if isinstance(rx_samples, tuple):
                    rx_samples = rx_samples[0]
            except Exception as e:
                if args.verbose:
                    print(f"[RX] Receive error: {e}")
                time.sleep(0.1)
                continue
            
            if rx_samples is None or len(rx_samples) == 0:
                time.sleep(0.01)
                continue
            
            # Normalize for processing
            max_val = np.max(np.abs(rx_samples))
            if max_val > 0:
                rx_signal_norm = rx_samples / max_val
            else:
                rx_signal_norm = rx_samples
            
            # Apply AGC
            rx_signal = agc.process(rx_signal_norm)
            rx_stream = np.concatenate([rx_stream, rx_signal])
            
            # Process buffer when we have enough samples
            while len(rx_stream) >= min_capture:
                window_len = min(len(rx_stream), min_capture * 2)
                rx_window = rx_stream[:window_len]
                
                stats['sync_attempts'] += 1
                
                # Synchronize
                synced_sig, sync_metrics = link._synchronize(rx_window)
                
                if not sync_metrics.get('sync_success'):
                    # No sync, discard some samples and try again
                    if len(rx_stream) > min_capture * 4:
                        rx_stream = rx_stream[-min_capture * 2:]
                    break
                
                stats['sync_successes'] += 1
                
                if args.verbose:
                    peak_val = sync_metrics.get('peak_val', 0)
                    cfo_val = sync_metrics.get('cfo_est', 0)
                    print(f"[RX] Sync! Peak: {peak_val:.1f}, CFO: {cfo_val:.1f} Hz, AGC: {agc.gain:.1f}")
                
                # Get payload start
                payload_start = sync_metrics.get('payload_start', None)
                if payload_start is None:
                    break
                
                total_needed = payload_start + expected_payload_samples
                if len(rx_stream) < total_needed:
                    break
                
                # Extract payload
                payload = synced_sig[:expected_payload_samples]
                rx_stream = rx_stream[total_needed:]
                
                # Demodulate
                rx_fec_bits, met = link.transceiver.demodulate(payload)
                
                # FEC Decode
                try:
                    rx_bits_dec = link.fec_codec.decode(rx_fec_bits)
                except Exception as e:
                    if args.verbose:
                        print(f"[RX] FEC decode error: {e}")
                    continue
                
                if len(rx_bits_dec) < expected_packet_bits:
                    continue
                
                rx_bits_dec = rx_bits_dec[:expected_packet_bits]
                rx_bytes = np.packbits(rx_bits_dec).tobytes()
                
                if len(rx_bytes) < packet_bytes:
                    continue
                rx_bytes = rx_bytes[:packet_bytes]
                
                # Header scanning with alignment recovery
                header_valid_found = False
                info = None
                rx_packet_clean = None
                scan_max = repeat_len if repeat_len > 0 else 0
                
                for shift in range(0, scan_max + 1):
                    header_start = shift
                    header_end = header_start + link.video_codec.HEADER_SIZE
                    repeat_start = header_end
                    repeat_end = repeat_start + repeat_len
                    payload_start_byte = repeat_end
                    payload_end_byte = payload_start_byte + payload_size
                    
                    if payload_end_byte > len(rx_bytes):
                        break
                    
                    header_bytes = rx_bytes[header_start:header_end]
                    
                    # Verify header repeat
                    if repeat_len > 0:
                        repeat_bytes = rx_bytes[repeat_start:repeat_end]
                        if repeat_bytes != header_bytes[:repeat_len]:
                            continue
                    
                    payload_bytes = rx_bytes[payload_start_byte:payload_end_byte]
                    candidate = header_bytes + payload_bytes
                    candidate_info = link.video_codec.parse_packet_header(candidate)
                    
                    if candidate_info:
                        # Validate header fields
                        if (
                            (candidate_info['width'], candidate_info['height']) not in allowed_set
                            or candidate_info['quality'] <= 0
                            or candidate_info['quality'] > 100
                            or candidate_info['total_pkts'] <= 0
                            or candidate_info['total_pkts'] > max_total_pkts
                            or candidate_info['pkt_idx'] < 0
                            or candidate_info['pkt_idx'] >= candidate_info['total_pkts']
                        ):
                            continue
                        
                        header_valid_found = True
                        
                        # Verify CRC
                        if zlib.crc32(candidate_info['payload']) & 0xFFFFFFFF == candidate_info['crc']:
                            info = candidate_info
                            rx_packet_clean = candidate
                            break
                
                if info is None:
                    if header_valid_found:
                        stats['crc_fails'] += 1
                    else:
                        stats['header_invalid'] += 1
                    continue
                
                stats['rx_packets'] += 1
                
                fid = info['frame_id']
                
                # Initialize frame buffer
                if fid not in frame_buffer:
                    frame_buffer[fid] = {
                        'pkts': {},
                        'payloads': {},
                        'total': info['total_pkts'],
                        'quality': info['quality'],
                        'ts': time.time(),
                    }
                    stats['frames_started'] += 1
                else:
                    frame_buffer[fid]['ts'] = time.time()
                    frame_buffer[fid]['total'] = info['total_pkts']
                
                frame_buffer[fid]['pkts'][info['pkt_idx']] = rx_packet_clean
                frame_buffer[fid]['payloads'][info['pkt_idx']] = info['payload']
                
                # Try MAC-FEC recovery
                if args.mac_fec_parity and frame_buffer[fid]['total'] > 1:
                    total = frame_buffer[fid]['total']
                    parity_count = int(args.mac_fec_parity)
                    data_total = total - parity_count
                    
                    if data_total > 0:
                        pkts = frame_buffer[fid]['pkts']
                        payloads = frame_buffer[fid]['payloads']
                        missing = [i for i in range(data_total) if i not in pkts]
                        parity_indices = [i for i in range(data_total, total) if i in payloads]
                        
                        if missing and len(parity_indices) >= len(missing):
                            recovered = solve_missing(
                                missing, parity_indices[:len(missing)], 
                                data_total, fid, payloads, payload_size, 
                                args.mac_fec_seed
                            )
                            
                            if recovered:
                                for miss_idx, payload_bytes in recovered.items():
                                    rebuilt = link.video_codec.create_packet_header(
                                        payload_bytes, fid, miss_idx, total,
                                        frame_buffer[fid]['quality']
                                    )
                                    pkts[miss_idx] = rebuilt
                                    payloads[miss_idx] = payload_bytes
                                stats['mac_fec_recovered_packets'] += len(recovered)
                        
                        # Check if data packets are complete
                        data_ready = all(i in frame_buffer[fid]['pkts'] for i in range(data_total))
                        
                        if data_ready and fid not in recovered_ids:
                            pkt_list = [frame_buffer[fid]['pkts'][i] for i in range(data_total)]
                            frame = link.video_codec.decode_packets(pkt_list)
                            
                            if frame is not None:
                                write_frame = frame
                                if (frame.shape[1], frame.shape[0]) != (args.width, args.height):
                                    write_frame = cv2.resize(frame, (args.width, args.height))
                                
                                out.write(write_frame)
                                recovered_ids.add(fid)
                                stats['frames_recovered'] += 1
                                
                                if not args.headless:
                                    cv2.imshow("SDR Video RX v2", write_frame)
                                    cv2.waitKey(1)
                                
                                if args.save_frames_dir:
                                    save_name = f"frame_{fid:06d}.jpg"
                                    cv2.imwrite(os.path.join(args.save_frames_dir, save_name), write_frame)
                                
                                print(f"[RX] Frame {fid} recovered. AGC: {agc.gain:.1f}")
                                del frame_buffer[fid]
                
                else:
                    # Simple frame assembly (no MAC-FEC)
                    if len(frame_buffer[fid]['pkts']) == frame_buffer[fid]['total']:
                        pkt_list = [frame_buffer[fid]['pkts'][i] for i in range(frame_buffer[fid]['total'])]
                        frame = link.video_codec.decode_packets(pkt_list)
                        
                        if frame is not None and fid not in recovered_ids:
                            write_frame = frame
                            if (frame.shape[1], frame.shape[0]) != (args.width, args.height):
                                write_frame = cv2.resize(frame, (args.width, args.height))
                            
                            out.write(write_frame)
                            recovered_ids.add(fid)
                            stats['frames_recovered'] += 1
                            
                            if not args.headless:
                                cv2.imshow("SDR Video RX v2", write_frame)
                                cv2.waitKey(1)
                            
                            print(f"[RX] Frame {fid} complete. AGC: {agc.gain:.1f}")
                        del frame_buffer[fid]
            
            # Expire old frames
            now = time.time()
            expired = [k for k, v in frame_buffer.items() if now - v['ts'] > args.frame_timeout]
            for k in expired:
                stats['frames_expired'] += 1
                del frame_buffer[k]
            
            # Log stats periodically
            if now - last_log >= args.log_interval:
                sync_rate = stats['sync_successes'] / stats['sync_attempts'] if stats['sync_attempts'] > 0 else 0
                print(f"[RX] Frames: {stats['frames_recovered']}, Packets: {stats['rx_packets']}, "
                      f"CRC Fail: {stats['crc_fails']}, Sync: {stats['sync_successes']}/{stats['sync_attempts']} "
                      f"({sync_rate:.1%}), AGC: {agc.gain:.1f}")
                last_log = now
                
                log_event("RX_STATS", stats.copy())
            
            # Handle GUI events
            if not args.headless:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[RX] Quit requested")
                    break
    
    except KeyboardInterrupt:
        print("\n[RX] Interrupted by user")
    
    finally:
        print("\n" + "=" * 60)
        print("[RX] Final Statistics:")
        print(f"  Frames Recovered: {stats['frames_recovered']}")
        print(f"  Packets Received: {stats['rx_packets']}")
        print(f"  CRC Failures: {stats['crc_fails']}")
        print(f"  Header Invalid: {stats['header_invalid']}")
        print(f"  Frames Started: {stats['frames_started']}")
        print(f"  Frames Expired: {stats['frames_expired']}")
        print(f"  MAC-FEC Recovered: {stats['mac_fec_recovered_packets']} packets")
        if stats['sync_attempts'] > 0:
            print(f"  Sync Rate: {stats['sync_successes']}/{stats['sync_attempts']} "
                  f"({100*stats['sync_successes']/stats['sync_attempts']:.1f}%)")
        print("=" * 60)
        
        out.release()
        if not args.headless:
            cv2.destroyAllWindows()
        
        log_event("RX_COMPLETE", stats)

if __name__ == "__main__":
    main()
