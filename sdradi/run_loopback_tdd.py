
import time
import cv2
import numpy as np
import zlib
import sys
import os

# Ensure local modules are found
sys.path.append(os.getcwd())

from sdr_video_comm import SDRVideoLink, SDRConfig, FECConfig, FECType, OFDMConfig

# TDD Parameters
BURST_FRAMES = 1
RX_TIMEOUT = 2.0 # Wait up to 2s for burst to return
RX_BUFFER_SAMPLES = 262144 # enough for ~100ms at 2Msps

def run_tdd_loopback(video_file, config_file="sdr_tuned_config.json"):
    print(f"[TDD] Loading config from {config_file}...")
    sdr_cfg = SDRConfig.load_from_json(config_file)
    sdr_cfg.rx_uri = "" # Force single device
    
    # Enable Torch LDPC
    try:
        import torch
        print(f"[TDD] Torch available: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        fec_cfg = FECConfig(enabled=True, fec_type=FECType.LDPC)
    except ImportError:
        print("[TDD] Torch not found, using CPU convolution")
        fec_cfg = FECConfig(enabled=True, fec_type=FECType.CONVOLUTIONAL)

    # Low Sync Threshold for loopback
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = 20.0 # Increased to avoid noise
    
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg)
    
    if not link.connect_sdr():
        print("Failed to connect to SDR.")
        return

    # Open Video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Could not open video.")
        return

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter("tdd_loopback_out.avi", fourcc, 10.0, link.video_config.resolution)
    
    frame_idx = 0
    total_recovered = 0
    
    try:
        while True:
            # 1. TX Phase: Encode and Buffer Burst
            packets_to_send = []
            for _ in range(BURST_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                frame = cv2.resize(frame, tuple(link.video_config.resolution))
                pkts = link.video_codec.encode_frame(frame, quality=50)
                
                # Modulate all packets
                for pkt_bytes, pkt_i in pkts:
                    # Create Header
                    header = link.video_codec.create_packet_header(pkt_bytes, frame_idx, pkt_i, len(pkts))
                    # FEC Encode
                    pkt_arr = np.frombuffer(header, dtype=np.uint8)
                    bits = np.unpackbits(pkt_arr)
                    fec_bits = link.fec_codec.encode(bits)
                    # Modulate
                    tx_signal = link.transmit(fec_bits)
                    packets_to_send.append(tx_signal)
                
                frame_idx += 1
            
            if not packets_to_send: continue
            
            # Concatenate all signals
            burst_signal = np.concatenate(packets_to_send)
            
            # 2. Transmit Burst
            print(f"[TDD] TX Burst: {len(packets_to_send)} packets ({len(burst_signal)} samples)...")
            # Clear RX buffer first to remove old garbage
            link.sdr.sdr.rx_destroy_buffer()
            
            # Send
            # We use the raw sdr object to burst
            # cyclic=True allows us to catch the signal in the RX phase (single thread limitation)
            # normalize=True is CRITICAL to prevent integer overflow in DAC scaling (2^14)
            link.sdr.SDR_TX_send(burst_signal, normalize=True, cyclic=True) 
            
            # 3. RX Phase: Listen immediately
            print("[TDD] RX Listening (Cyclic TX)...")
            # Read multiple chunks to cover the burst
            total_rx_samples = 0
            # Wait for 1 second of data or success
            start_rx = time.time()
            while time.time() - start_rx < 1.0:
                rx_bits, metrics = link.receive()
                
                if metrics.get('sync_success'):
                    peak = metrics.get('peak_val', 0)
                    cfo = metrics.get('cfo_est', 0)
                    rx_amp = metrics.get('rx_max', 0)
                    print(f"  [RX] Sync! Peak:{peak:.1f} CFO:{cfo:.1f} RxAmp:{rx_amp:.3f}")
                    
                    # Store if valid
                    try:
                        rx_bytes = np.packbits(rx_bits.astype(np.uint8)).tobytes()
                        info = link.video_codec.parse_packet_header(rx_bytes)
                        if info and zlib.crc32(info['payload']) & 0xFFFFFFFF == info['crc']:
                             print(f"  [RX] Packet OK (Frame {info['frame_id']})")
                             
                             # Decode (simplified: just verify CRC for loopback test)
                             total_recovered += 1
                             # Exit inner loop if we got a frame (to proceed to next burst)
                             break 
                    except Exception as e:
                        pass
                
                # Check error
                if metrics.get('error'):
                   # If SDR error, maybe timeout
                   time.sleep(0.01)
            
            # Stop TX after RX is done
            link.sdr.SDR_TX_stop()
            time.sleep(0.1) # Guard interval
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        out.release()
        print(f"[TDD] Done. Total Valid Packets: {total_recovered}")

if __name__ == "__main__":
    run_tdd_loopback("test_video.mp4")
