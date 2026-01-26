
import cv2
import time
import numpy as np
from sdr_video_comm import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

def run_e2e_sim(video_path="test_video.mp4", output_path="sim_output.avi"):
    print(f"=== Running E2E Video Link Simulation ===")
    print(f"Input: {video_path}")
    
    # Setup
    # Enable LDPC for best performance if verified 
    # (Rate 0.5 is standard, ensure it's fast enough or use Repetition if slower but safer)
    # We verified LDPC is fast (25fps).
    fec_cfg = FECConfig(enabled=True, fec_type=FECType.LDPC)
    # fec_cfg = FECConfig(enabled=True, fec_type=FECType.REPETITION, repetitions=3)
    
    ofdm_cfg = OFDMConfig(fft_size=64, cp_length=16)
    ofdm_cfg.sync_threshold = 20.0
    
    link = SDRVideoLink(
        sdr_config=SDRConfig(),
        ofdm_config=ofdm_cfg,
        fec_config=fec_cfg,
        waveform=WaveformType.OFDM,
        simulation_mode=True
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video file. Generating synthetic frames.")
        cap = None
        
    width, height = 640, 480
    if cap:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    # Resize for transmission speed
    target_w, target_h = 320, 240
    link.video_config.resolution = (target_w, target_h)
    
    # Writer for output
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (target_w, target_h))
    
    frame_count = 0
    total_latency = 0
    
    try:
        while True:
            if cap:
                ret, frame = cap.read()
                if not ret: break
            else:
                # Synthetic
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(frame, (frame_count*5 % 600, 100), (frame_count*5 % 600 + 50, 150), (0, 255, 0), -1)
                cv2.putText(frame, f"Cnt: {frame_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if frame_count > 50: break
                
            frame_resized = cv2.resize(frame, (target_w, target_h))
            
            t_start = time.time()
            
            # 1. TX Chain
            packets = link.video_codec.encode_frame(frame_resized, quality=40)
            
            recovered_packets = []
            
            for pkt_bytes, pkt_idx in packets:
                # Packet -> Bits
                # We need to serialize packet bytes to bits.
                # VideoCodec doesn't have bytes_to_bits helper for single packet payload?
                # It has `decode_packets` taking bytes.
                # But link.transmit takes bits.
                
                # We need a helper. np.unpackbits?
                # sdr_video_comm uses `bits_to_bytes` and `bytes_to_bits` internally?
                # I'll use numpy.
                pkt_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                
                # FEC + Mod + Preamble
                # link.transmit adds preamble.
                # link.transceiver.modulate does not.
                # We also need FEC.
                
                # Manual chain to simulate "Link" layer
                fec_bits = link.fec_codec.encode(bits)
                tx_signal = link.transmit(fec_bits)
                
                # 2. Channel Impairments
                # Delay + CFO + Noise
                rx_signal = link.simulate_channel(tx_signal, snr_db=25, channel_type='awgn')
                
                # Add delay/CFO explicitly if simulate_channel doesn't
                # simulate_channel only does AWGN/Fading.
                
                cfo_hz = 500.0
                t = np.arange(len(rx_signal)) / 2e6
                rx_signal = rx_signal * np.exp(1j * 2 * np.pi * cfo_hz * t)
                
                rx_signal = np.concatenate([np.zeros(100, dtype=complex), rx_signal])
                
                # 3. RX Chain
                # _synchronize -> Demod
                synced_sig, sync_metrics = link._synchronize(rx_signal)
                
                if sync_metrics.get('sync_success'):
                    # Demod
                    rx_fec_bits, met = link.transceiver.demodulate(synced_sig)
                    # FEC Decode
                    rx_bits_dec = link.fec_codec.decode(rx_fec_bits)
                    
                    # Bits -> Bytes
                    rx_bytes = np.packbits(rx_bits_dec).tobytes()
                    # Trim padding (unpackbits pads to 8)
                    rx_bytes = rx_bytes[:len(pkt_bytes)]
                    
                    recovered_packets.append(rx_bytes)
                else:
                    print(f"Packet {pkt_idx} lost (Sync Fail)")
            
            # 4. Reassemble Frame
            rec_frame = link.video_codec.decode_packets(recovered_packets)
            
            t_end = time.time()
            latency = (t_end - t_start) * 1000
            total_latency += latency
            
            if rec_frame is not None:
                out.write(rec_frame)
                print(f"Frame {frame_count}: Latency={latency:.1f}ms, Pkts={len(recovered_packets)}/{len(packets)}")
            else:
                print(f"Frame {frame_count}: FAILED")
                
            frame_count += 1
            
            if frame_count >= 30: # Limit test
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        if cap: cap.release()
        out.release()
        print(f"Simulation Complete. Avg Latency: {total_latency/frame_count:.1f}ms")

if __name__ == "__main__":
    run_e2e_sim()
