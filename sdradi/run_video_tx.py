import sys
import time
import cv2
import numpy as np
import argparse
from sdr_video_comm import SDRVideoLink, SDRConfig, OFDMConfig, WaveformType, VideoCodec, FECCodec, FECConfig

def main():
    parser = argparse.ArgumentParser(description='SDR Video TX (Jetson)')
    parser.add_argument('--ip', default='ip:192.168.2.2', help='SDR IP address')
    parser.add_argument('--file', default='test_video.mp4', help='Video file path')
    parser.add_argument('--quality', type=int, default=30, help='JPEG Quality (1-100)')
    parser.add_argument('--scale', type=float, default=0.5, help='Resize factor')
    args = parser.parse_args()

    # Load Configs
    sdr_cfg = SDRConfig.load_from_json()
    sdr_cfg.sdr_ip = args.ip
    sdr_cfg.rx_uri = None # DISABLE Dual-Device Mode - We only control local TX
    sdr_cfg.device = "pluto" # Force pluto for consistency
    # Ensure buffer is large enough for a frame if we were receiving, 
    # but for TX, we just need to send big buffers.
    
    # Init Link
    link = SDRVideoLink(sdr_config=sdr_cfg)
    if not link.connect_sdr():
        print("Failed to connect to SDR")
        return

    # Video Tools
    video_codec = VideoCodec()
    fec_config = FECConfig(enabled=True, code_rate=0.5) # Use robust FEC
    fec_codec = FECCodec(fec_config)

    # Open Video
    cap = cv2.VideoCapture(args.file)
    if not cap.isOpened():
        print(f"Error opening video file: {args.file}")
        return

    print(f"Starting Video TX: {args.file} -> {args.ip}")
    print(f"Modulation: {link.waveform.name}, Quality: {args.quality}")
    print("-" * 50)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of file. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize
            if args.scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=args.scale, fy=args.scale)

            # Encode
            # 1. JPEG
            # Returns list of (header+payload, packet_idx)
            packets = video_codec.encode_frame(frame, quality=args.quality)
            
            print(f"Frame {frame_idx}: {len(packets)} packets")
            
            start_t = time.time()
            
            # Send each packet individually
            for pkt_data, pkt_idx in packets:
                # 2. FEC
                data_bits = video_codec.bytes_to_bits(pkt_data)
                tx_bits = fec_codec.encode(data_bits)
                
                # 3. Transmit
                # transmit() adds the preamble and sends via SDR_TX_send
                link.transmit(tx_bits)
                
                # Small gap to let RX process?
                # time.sleep(0.01) 
            
            dur = time.time() - start_t
            print(f"  -> TX: {dur*1000:.1f}ms")
            
            frame_idx += 1
            # throttle slightly 
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")
        cap.release()

if __name__ == "__main__":
    main()
