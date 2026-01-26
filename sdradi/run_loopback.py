
import threading
import time
import cv2
import numpy as np
import zlib
import queue
from sdr_video_comm import SDRVideoLink, SDRConfig, FECConfig, FECType, OFDMConfig

# Thread-safe queue for RX frames to be displayed
rx_frame_queue = queue.Queue()
stop_event = threading.Event()

def run_tx_thread(link, video_file):
    print("[TX] Starting Loopback Stream...")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("[TX] Error: Could not open video file.")
        return

    frame_idx = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Encode
            frame = cv2.resize(frame, tuple(link.video_config.resolution))
            packets = link.video_codec.encode_frame(frame, quality=50)
            
            for pkt_bytes, pkt_i in packets:
                if stop_event.is_set(): break
                
                # Transmit Packet
                # Note: transmit() in sdr_video_comm.py calls sdr.SDR_TX_send which handles normalization
                link.transmit_packet(pkt_bytes, frame_idx, pkt_i, len(packets))
                
                # Throttle slightly to avoid USB buffer overflow if PC is too fast
                # At 2Msps, 16k samples take ~8ms.
                time.sleep(0.005) 
            
            frame_idx += 1
            # print(f"[TX] Sent Frame {frame_idx}")
            
    except Exception as e:
        print(f"[TX] Error: {e}")
    finally:
        cap.release()
        print("[TX] Stopped.")

def run_rx_thread(link, output_file="loopback_out.avi"):
    print("[RX] Starting Loopback Listener...")
    frame_buffer = {}
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file, fourcc, 10.0, link.video_config.resolution)
    
    total_recovered = 0
    
    try:
        while not stop_event.is_set():
            # Receive Packet
            # This calls sdr.rx() which returns a chunk of samples
            rx_bits, metrics = link.receive()
            
            # Diagnostic
            if metrics.get('sync_success'):
                 peak = metrics.get('peak_val', 0)
                 cfo = metrics.get('cfo_est', 0)
                 print(f"[RX Diag] Sync! Peak:{peak:.1f} CFO:{cfo:.1f} Hz")
            
            if metrics.get('error'):
                continue
                
            # Parse
            # Ensure bits are uint8/bool for packbits
            if rx_bits.size > 0:
                rx_bytes = np.packbits(rx_bits.astype(np.uint8)).tobytes()
            else:
                continue
            
            info = link.video_codec.parse_packet_header(rx_bytes)
            
            if info:
                # CRC Check
                if zlib.crc32(info['payload']) & 0xFFFFFFFF != info['crc']:
                    print(f"[RX] CRC Fail (Frame {info['frame_id']} Pkt {info['pkt_idx']})")
                    continue
                
                # Reassemble
                fid = info['frame_id']
                if fid not in frame_buffer:
                    frame_buffer[fid] = {'pkts': {}, 'total': info['total_pkts'], 'ts': time.time()}
                
                frame_buffer[fid]['pkts'][info['pkt_idx']] = rx_bytes
                
                if len(frame_buffer[fid]['pkts']) == frame_buffer[fid]['total']:
                    pkt_list = [frame_buffer[fid]['pkts'][i] for i in range(frame_buffer[fid]['total'])]
                    frame = link.video_codec.decode_packets(pkt_list)
                    if frame is not None:
                        total_recovered += 1
                        print(f"[RX] Recovered Frame {fid} (Total: {total_recovered})")
                        out.write(frame)
                        rx_frame_queue.put(frame) # For UI/Display
                    del frame_buffer[fid]
            
            # Cleanup old frames
            now = time.time()
            expired = [k for k,v in frame_buffer.items() if now - v['ts'] > 2.0]
            for k in expired: del frame_buffer[k]
            
    except Exception as e:
        print(f"[RX] Error: {e}")
    finally:
        out.release()
        print(f"[RX] Stopped. Total Recovered: {total_recovered}")

def main():
    config_file = "sdr_tuned_config.json"
    print(f"Loading config from {config_file}...")
    
    sdr_cfg = SDRConfig.load_from_json(config_file)
    # Ensure SINGLE DEVICE mode by clearing rx_uri if it matches sdr_ip or is empty
    sdr_cfg.rx_uri = "" 
    
    # Robust Config
    fec_cfg = FECConfig(enabled=True, fec_type=FECType.LDPC)
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = 10.0 # Lower usage for debugging
    
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg)
    
    if not link.connect_sdr():
        print("Failed to connect to SDR.")
        return

    # Start Threads
    tx_thread = threading.Thread(target=run_tx_thread, args=(link, "test_video.mp4"))
    rx_thread = threading.Thread(target=run_rx_thread, args=(link,))
    
    tx_thread.start()
    rx_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        tx_thread.join()
        rx_thread.join()
        print("Done.")

if __name__ == "__main__":
    main()
