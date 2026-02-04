import sys
import time
import cv2
import numpy as np
import argparse
import os
import zlib  # Added for CRC check
import json
import datetime

# Fix Qt Wayland issue
os.environ["QT_QPA_PLATFORM"] = "xcb"

from sdr_video_comm import SDRVideoLink, SDRConfig, VideoCodec, FECCodec, FECConfig, WaveformType

LOG_FILE = "rx_debug.jsonl"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):  # Handle bytes
            return obj.decode('utf-8', errors='ignore')
        if isinstance(obj, complex): # Handle complex
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

def main():
    parser = argparse.ArgumentParser(description='SDR Video RX (Local)')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    parser.add_argument('--ip', default='ip:192.168.3.2', help='SDR IP address')
    parser.add_argument('--waveform', choices=['ofdm', 'otfs'], default='ofdm', help='Waveform Type')
    parser.add_argument('--freq-offset', type=int, default=0, help='Manual Rx LO Offset (Hz)')
    args = parser.parse_args()

    sdr_cfg = SDRConfig.load_from_json()
    sdr_cfg.sdr_ip = args.ip
    sdr_cfg.rx_uri = None # DISABLE Dual-Device Mode - We only control local RX
    sdr_cfg.device = "pluto"
    
    # Apply Manual Offset
    sdr_cfg.fc += args.freq_offset
    print(f"Applying Manual Freq Offset: {args.freq_offset} Hz -> FC: {sdr_cfg.fc}")
    
    # Log Configuration
    log_event("CONFIG", {
        "sdr_config": sdr_cfg.__dict__,
        "args": vars(args)
    })
    
    # Init Link
    waveform_enum = WaveformType.OTFS if args.waveform == 'otfs' else WaveformType.OFDM
    link = SDRVideoLink(sdr_config=sdr_cfg, waveform=waveform_enum)
    
    # link.ofdm_config.sync_threshold = 15.0 # This only applies if OFDM. 
    # OTFS might need tuning? Default should be fine.
    
    if link.waveform == WaveformType.OFDM:
        link.ofdm_config.sync_threshold = 15.0
        
    if not link.connect_sdr():
        print("Failed to connect to SDR")
        return

    # Verify Hardware State
    actual_lo = int(link.sdr.sdr.rx_lo)
    print(f"[Hardware Check] Config FC: {link.sdr_config.fc}, Hardware LO: {actual_lo}")
    
    if abs(actual_lo - link.sdr_config.fc) > 1000:
        print(f"[Correction] Hardware LO mismatch! Forcing to {link.sdr_config.fc}")
        link.sdr.sdr.rx_lo = int(link.sdr_config.fc)
        print(f"[Correction] New LO: {link.sdr.sdr.rx_lo}")
        
    # Start RX Loop

    # Video Tools
    video_codec = VideoCodec()
    fec_config = FECConfig(enabled=True, code_rate=0.5) 
    fec_codec = FECCodec(fec_config)

    print(f"Starting Video RX: {args.ip}")
    print("Waiting for frames...")
    print("-" * 50)

    if not args.headless:
        cv2.namedWindow("SDR Video Feed", cv2.WINDOW_NORMAL)

    frame_buffer = {} # {frame_id: {pkts: {idx: bytes}, total: N, ts: time}}
    
    try:
        while True:
            # Receive (Blocking-ish)
            try:
                rx_bits_coded, metrics = link.receive()
            except ValueError as e:
                # Handle zero-size array (empty buffer/timeout)
                # print(f"[RX] Error receiving: {e}")
                time.sleep(0.1)
                continue
            
            if metrics.get('sync_success') and len(rx_bits_coded) > 0:
                # Debug CFO
                cfo_est = metrics.get('cfo_est', 0)
                snr_est = metrics.get('snr_est', 0)
                print(f"[RX] Locked. SNR: {snr_est:.1f}dB, CFO: {cfo_est/1000:.1f}kHz")
                
                # Auto-Tune Logic
                if abs(cfo_est) > 50000:
                   current_rx_lo = int(link.sdr.sdr.rx_lo)
                   tune_step = int(cfo_est * 0.1) # Slow AutoTune
                   new_fc = current_rx_lo + tune_step
                   print(f"[AutoTune] Adjusting LO: {current_rx_lo} -> {new_fc} Hz (Step {tune_step})")
                   link.sdr.sdr.rx_lo = int(new_fc)
                   # Update config reference too
                   link.sdr_config.fc = float(new_fc)
                
                # Log Metrics
                log_data = metrics.copy()
                # Remove large arrays to keep log compact
                keys_to_remove = ['channel_est', 'rx_constellation', 'rx_symbols', 'tx_symbols']
                for k in keys_to_remove:
                    if k in log_data:
                        del log_data[k]
                
                log_event("METRICS", log_data)
                
                
                try:
                    rx_bits = fec_codec.decode(rx_bits_coded)
                    rx_bytes = video_codec.bits_to_bytes(rx_bits)
                    
                    # Parse Header
                    info = video_codec.parse_packet_header(rx_bytes)
                    if info:
                        # CRITICAL: Verify CRC before processing!
                        payload = info['payload']
                        expected_crc = info['crc']
                        if zlib.crc32(payload) & 0xFFFFFFFF != expected_crc:
                            print(f"  -> CRC Failed for Frame {info['frame_id']} (Pk {info['pkt_idx']})")
                            continue
                            
                        fid = info['frame_id']
                        tot = info['total_pkts']
                        idx = info['pkt_idx']
                        
                        # Validate sanity
                        if tot == 0 or tot > 200: 
                             continue
                        
                        # Initialize buffer for this frame
                        if fid not in frame_buffer:
                            frame_buffer[fid] = {'pkts': {}, 'total': tot, 'ts': time.time()}
                        
                        # Store packet
                        frame_buffer[fid]['pkts'][idx] = rx_bytes
                        frame_buffer[fid]['ts'] = time.time() # Update timestamp
                        
                        # Check complete
                        current_frame = frame_buffer[fid]
                        if len(current_frame['pkts']) == current_frame['total']:
                            print(f"[RX] Frame {fid} Complete ({tot} pkts)! Decoding...")
                            
                            # Compile list
                            pkt_list = list(current_frame['pkts'].values())
                            frame = video_codec.decode_packets(pkt_list)
                            
                            if frame is not None:
                                print(f"[RX] Decoded Frame {fid}. Shape: {frame.shape}")
                                if not args.headless:
                                    cv2.imshow("SDR Video Feed", frame)
                                    cv2.waitKey(1)
                            else:
                                print(f"  -> JPEG Decode Failed for Frame {fid} (Size: {len(pkt_list)} pkts)")
                                # Check incomplete JPEG headers?
                                full_data = b''.join(pkt_list)
                                if len(full_data) > 0:
                                    print(f"     First 10 bytes: {full_data[:10].hex()}")
                                    print(f"     Last 10 bytes: {full_data[-10:].hex()}")
                            
                            # Remove processed frame
                            del frame_buffer[fid]
                            
                        else:
                            print(f"[RX] Frame {fid} Part {idx+1}/{tot} Received")
                    
                    # Cleanup old frames (> 2 seconds)
                    now = time.time()
                    expired = [k for k, v in frame_buffer.items() if now - v['ts'] > 2.0]
                    for k in expired:
                        del frame_buffer[k]
                        
                except Exception as e:
                    print(f"  -> Decode Error: {e}")
            else:
                # Debug: Show peak
                peak = metrics.get('peak_val', 0)
                # if peak > 15: 
                #    print(f"[RX Scanning] Peak: {peak:.1f}")
                pass

    except KeyboardInterrupt:
        print("\nStopping...")
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
