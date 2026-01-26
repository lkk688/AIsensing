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
            packets = video_codec.encode_frame(frame, quality=args.quality)
            all_bytes = b''.join([p[0] for p in packets])
            data_bits = video_codec.bytes_to_bits(all_bytes)
            
            # 2. FEC
            tx_bits = fec_codec.encode(data_bits)
            
            # 3. Transmit
            # transmit() adds the preamble and sends via SDR_TX_send
            # Check length - if too big, warn
            if len(tx_bits) > 100000: # Arbitrary warning threshold
                 print(f"Warning: Large Frame ({len(tx_bits)} bits)")
            
            start_t = time.time()
            link.transmit(tx_bits)
            dur = time.time() - start_t
            
            print(f"Frame {frame_idx}: {len(data_bits)/8/1024:.1f} KB -> TX: {dur*1000:.1f}ms")
            
            frame_idx += 1
            # throttle slightly to allow RX to process?
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
        cap.release()

if __name__ == "__main__":
    main()
