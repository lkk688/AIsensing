import sys
import time
import numpy as np
import argparse
import json
import datetime
from sdr_video_comm import SDRVideoLink, SDRConfig, OFDMConfig, WaveformType, VideoCodec, FECCodec, FECConfig

LOG_FILE = "tx_debug.jsonl"

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
    parser = argparse.ArgumentParser(description='SDR Video TX (Jetson)')
    parser.add_argument('--ip', default='ip:192.168.2.2', help='SDR IP address')
    parser.add_argument('--file', default='test_video.mp4', help='Video file path')
    parser.add_argument('--quality', type=int, default=30, help='JPEG Quality (1-100)')
    parser.add_argument('--scale', type=float, default=0.5, help='Resize factor')
    parser.add_argument('--waveform', choices=['ofdm', 'otfs'], default='ofdm', help='Waveform Type')
    parser.add_argument('--gain', type=int, default=None, help='TX Gain Override (dB)')
    parser.add_argument('--test-pattern', action='store_true', help='Generate test pattern instead of file')
    parser.add_argument('--cyclic', action='store_true', help='Transmit single frame in cyclic buffer (Benchmark)')
    args = parser.parse_args()

    # Load Configs
    sdr_cfg = SDRConfig.load_from_json()
    sdr_cfg.sdr_ip = args.ip # CRITICAL FIX: Use the arg!
    print(f"[DEBUG] SDR IP override: {sdr_cfg.sdr_ip}")
    
    if args.gain is not None:
        sdr_cfg.tx_gain = args.gain
    sdr_cfg.rx_uri = None 
    sdr_cfg.device = "pluto"
    sdr_cfg.rx_channels = 0 # DISABLE RX to prevent TX power loss
    
    # Log Configuration
    log_event("CONFIG", {
        "sdr_config": sdr_cfg.__dict__,
        "args": vars(args)
    })
    
    # Ensure buffer is large enough for a frame if we were receiving, 
    # but for TX, we just need to send big buffers.
    
    # Init Link
    waveform_enum = WaveformType.OTFS if args.waveform == 'otfs' else WaveformType.OFDM
    link = SDRVideoLink(sdr_config=sdr_cfg, waveform=waveform_enum)
    if not link.connect_sdr():
        print("Failed to connect to SDR")
        return

    # CYCLIC MODE CONFIGURATION
    if args.cyclic:
        print(">>> CYCLIC MODE ENABLED: Loading buffer once to FPGA <<<")
        link.sdr.tx_cyclic_buffer = True

    # Video Tools
    video_codec = VideoCodec()
    fec_config = FECConfig(enabled=False, code_rate=0.5) # DISABLE FEC to avoid Torch
    fec_codec = FECCodec(fec_config)

    # Open Video Source
    cap = None
    if not args.test_pattern and not args.cyclic:
        import cv2
        cap = cv2.VideoCapture(args.file)
        if not cap.isOpened():
            print(f"Error opening video file: {args.file}")
            return
            
    print(f"Starting TX (IP={args.ip}, Gain={args.gain}, Waveform={args.waveform})")
    
    import sys
    torch_loaded = [m for m in sys.modules if 'torch' in m]
    if torch_loaded:
        print(f"CRITICAL WARNING: TORCH IS LOADED: {torch_loaded}")
    else:
        print("CONFIRMED: Torch is NOT loaded.")

    frame_idx = 0
    try:
        while True:
            # 1. Capture Frame
            if args.test_pattern or args.cyclic:
                # Generate Pattern (Moving Square)
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                # Draw Rectangle (Moving) using NUMPY ONLY (No CV2)
                x = (frame_idx * 5) % 320
                y = (frame_idx * 3) % 240
                
                # frame[y:y+h, x:x+w]
                end_x = min(x+40, 320)
                end_y = min(y+40, 240)
                
                # Green box (BGR: 0, 255, 0)
                frame[y:end_y, x:end_x] = [0, 255, 0]
                # cv2.putText ... REMOVED to avoid import
                # cv2.putText(frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("End of video, looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Resize if needed (Skip for Test Pattern to avoid CV2)
            if not args.test_pattern and (frame.shape[1] != 320 or frame.shape[0] != 240):
                 # import cv2 inside if needed
                 pass 
                 # frame = cv2.resize(frame, (320, 240))
            
            # Encode
            # 1. JPEG
            # Returns list of (header+payload, packet_idx)
            packets = video_codec.encode_frame(frame, quality=args.quality)
            
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: {len(packets)} packets")
            
            start_t = time.time()
            
            # Send each packet individually
            for pkt_data, pkt_idx in packets:
                # 2. FEC
                data_bits = video_codec.bytes_to_bits(pkt_data)
                tx_bits = fec_codec.encode(data_bits)
                
                # 3. Transmit
                # transmit() adds the preamble and sends via SDR_TX_send
                link.transmit(tx_bits, cyclic=args.cyclic)
                
                # If Cyclic, we transmitted ONCE. Now we stop feeding bits.
                if args.cyclic:
                    print("Cyclic Buffer Loaded. Transmitting in HW Loop...")
                    while True:
                        time.sleep(1)
                
                # Full Speed Ahead
                # time.sleep(0.001) 
            
            dur = time.time() - start_t
            if frame_idx % 30 == 0:
                print(f"  -> TX: {dur*1000:.1f}ms")
            
            # Log Frame Stats
            log_event("TX_FRAME", {
                "frame_idx": frame_idx,
                "packets": len(packets),
                "duration_sec": dur,
                "fps_est": 1.0/dur if dur > 0 else 0
            })
            
            frame_idx += 1
            # throttle slightly 
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")
        if cap: cap.release()

if __name__ == "__main__":
    main()
