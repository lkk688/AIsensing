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

    try:
        while True:
            # Receive (Blocking-ish)
            # receive() returns raw decoded bits from the demodulator
            rx_bits_coded, metrics = link.receive()
            
            if metrics.get('sync_success') and len(rx_bits_coded) > 0:
                # We have a candidate frame!
                print(f"[RX] Preamble Locked. SNR: {metrics.get('snr_est', 0):.1f} dB. Bits: {len(rx_bits_coded)}")
                
                # 1. FEC Decode
                # We don't know exact length, so we decode what we have
                # FECCodec usually expects exact block sizes?
                # Convolutional code can decode stream.
                try:
                    rx_bits = fec_codec.decode(rx_bits_coded)
                    
                    # 2. Bytes
                    rx_bytes = video_codec.bits_to_bytes(rx_bits)
                    
                    # 3. Accumulate and Decode
                    # We don't know sequence yet, just try to decode the chunk as a packet
                    # decode_packets wants a list of bytes
                    frame = video_codec.decode_packets([rx_bytes])
                    
                    if frame is not None:
                        cv2.imshow("SDR Video Feed", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        print("  -> Frame Displayed!")
                        
                except Exception as e:
                    # It's normal to have partial packets fail
                    print(f"  -> Decode Error: {e}")
            else:
                # Debug: Show peak so we know if we are seeing the burst
                peak = metrics.get('peak_val', 0)
                if peak > 15: # Filter pure noise (~10-50 depends on gain, usually noise < 20)
                    print(f"[RX Scanning] Peak: {peak:.1f}")
                pass

    except KeyboardInterrupt:
        print("\nStopping...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
