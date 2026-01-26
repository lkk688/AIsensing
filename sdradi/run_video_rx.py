import sys
import time
import cv2
import numpy as np
import argparse
import os

# Fix Qt Wayland issue
os.environ["QT_QPA_PLATFORM"] = "xcb"

from sdr_video_comm import SDRVideoLink, SDRConfig, VideoCodec, FECCodec, FECConfig

def main():
    parser = argparse.ArgumentParser(description='SDR Video RX (Local)')
    parser.add_argument('--ip', default='ip:192.168.3.2', help='SDR IP address')
    args = parser.parse_args()

    sdr_cfg = SDRConfig.load_from_json()
    sdr_cfg.sdr_ip = args.ip
    sdr_cfg.rx_uri = None # DISABLE Dual-Device Mode - We only control local RX
    sdr_cfg.device = "pluto"
    
    # Init Link
    link = SDRVideoLink(sdr_config=sdr_cfg)
    if not link.connect_sdr():
        print("Failed to connect to SDR")
        return

    # Video Tools
    video_codec = VideoCodec()
    fec_config = FECConfig(enabled=True, code_rate=0.5) 
    fec_codec = FECCodec(fec_config)

    print(f"Starting Video RX: {args.ip}")
    print("Waiting for frames...")
    print("-" * 50)

    cv2.namedWindow("SDR Video Feed", cv2.WINDOW_NORMAL)

    frame_buffer = {} # {frame_id: {pkts: {idx: bytes}, total: N, ts: time}}
    
    try:
        while True:
            # Receive (Blocking-ish)
            rx_bits_coded, metrics = link.receive()
            
            if metrics.get('sync_success') and len(rx_bits_coded) > 0:
                try:
                    rx_bits = fec_codec.decode(rx_bits_coded)
                    rx_bytes = video_codec.bits_to_bytes(rx_bits)
                    
                    # Parse Header
                    info = video_codec.parse_packet_header(rx_bytes)
                    if info:
                        fid = info['frame_id']
                        tot = info['total_pkts']
                        idx = info['pkt_idx']
                        
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
                                cv2.imshow("SDR Video Feed", frame)
                                cv2.waitKey(1)
                            else:
                                print("  -> JPEG Decode Failed (Corrupt data)")
                            
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
                if peak > 15: 
                    print(f"[RX Scanning] Peak: {peak:.1f}")
                pass

    except KeyboardInterrupt:
        print("\nStopping...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
