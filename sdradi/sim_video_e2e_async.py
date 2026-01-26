
import threading
import queue
import time
import numpy as np
import cv2
import struct
import zlib
import json
import os
from sdr_video_comm import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

# =============================================================================
# Channel Simulation (The "Air")
# =============================================================================

class SharedChannel:
    """
    Simulates the physical medium between TX and RX.
    Handles shared buffering, non-linearities, and noise.
    """
    def __init__(self, fs=2e6, max_buffer_len=10000000):
        self.lock = threading.Lock()
        self.buffer = [] 
        self.fs = fs
        self.max_buffer_len = max_buffer_len
        
        # Hardware Constraints & Impairments
        self.tx_power_db = -10.0 # dBFS
        self.path_loss_db = 70.0 # Path loss
        self.rx_gain_db = 40.0   # RX LNA + VGA gain
        self.base_noise_dbfs = -60.0
        
        self.cfo_hz = 0.0
        self.saturation_limit = 1.0 
        
        self.total_samples_written = 0
        self.total_samples_read = 0
        
    def write(self, samples):
        """TX pushes samples to the air."""
        if len(samples) == 0: return
        
        with self.lock:
            # Apply TX Power
            gain_lin = 10 ** (self.tx_power_db / 20.0)
            samples = samples * gain_lin
            
            # Apply Saturation (DAC)
            samples = np.clip(samples, -self.saturation_limit-1j*self.saturation_limit, self.saturation_limit+1j*self.saturation_limit)
            
            self.buffer.extend(samples.tolist())
            self.total_samples_written += len(samples)
            
            if len(self.buffer) > self.max_buffer_len:
                drop = len(self.buffer) - self.max_buffer_len
                self.buffer = self.buffer[drop:]
                
    def read(self, num_samples):
        """RX pulls samples from the air."""
        with self.lock:
            available = len(self.buffer)
            if available < num_samples:
                return None
            
            chunk = np.array(self.buffer[:num_samples], dtype=complex)
            self.buffer = self.buffer[num_samples:]
            self.total_samples_read += num_samples
            
        # Channel Physics
        # Total Gain
        path_gain_db = -self.path_loss_db
        rx_amp_db = self.rx_gain_db
        total_gain_db = path_gain_db + rx_amp_db
        gain_lin = 10 ** (total_gain_db / 20.0)
        chunk = chunk * gain_lin
        
        # CFO
        if self.cfo_hz != 0.0:
            start_idx = self.total_samples_read - num_samples
            t = (np.arange(num_samples) + start_idx) / self.fs
            cfo_vec = np.exp(1j * 2 * np.pi * self.cfo_hz * t)
            chunk = chunk * cfo_vec
            
        # Noise
        noise_amp = 10 ** (self.base_noise_dbfs / 20.0)
        noise = (np.random.randn(len(chunk)) + 1j * np.random.randn(len(chunk))) / np.sqrt(2) * noise_amp
        chunk += noise
        
        # ADC Saturation
        chunk = np.clip(chunk, -self.saturation_limit-1j*self.saturation_limit, self.saturation_limit+1j*self.saturation_limit)
        
        return chunk

# Global Channel
channel = SharedChannel(fs=2e6)
stop_event = threading.Event()

# =============================================================================
# Helper: Digital AGC
# =============================================================================
class DigitalAGC:
    """Simple Digital AGC to normalize signal levels."""
    def __init__(self, target_level=0.5, alpha=0.1): 
        self.gain = 1.0
        self.target = target_level
        self.alpha = alpha 
        
    def process(self, signal):
        curr_amp = np.mean(np.abs(signal))
        if curr_amp < 1e-9: curr_amp = 1e-9
        
        # Hybrid update
        ideal_gain = self.target / curr_amp
        self.gain = (1 - self.alpha) * self.gain + self.alpha * ideal_gain
        
        # Limits
        if self.gain > 10000.0: self.gain = 10000.0
        if self.gain < 0.01: self.gain = 0.01
        
        return signal * self.gain

# =============================================================================
# TX Node
# =============================================================================

def run_tx_node(video_file, config_file="sdr_tuned_config.json"):
    print(f"[TX] Loading config from {config_file}...")
    sdr_cfg = SDRConfig.load_from_json(config_file)
    fec_cfg = FECConfig(enabled=True, fec_type=FECType.LDPC)
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, simulation_mode=True)
    
    # Update Channel Impairments
    if sdr_cfg.tx_gain > -80:
        channel.tx_power_db = float(sdr_cfg.tx_gain)
    
    print(f"[TX] Simulating TX Power: {channel.tx_power_db} dB")
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"[TX] Using Synthetic Video")
        cap = None
        
    width, height = 320, 240
    link.video_config.resolution = (width, height)
    frame_idx = 0
    
    try:
        while not stop_event.is_set():
            if cap:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            else:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.rectangle(frame, (frame_idx*5 % 300, 50), (frame_idx*5 % 300 + 30, 80), (0, 0, 255), -1)
                time.sleep(0.05)
                
            frame = cv2.resize(frame, (width, height))
            packets = link.video_codec.encode_frame(frame, quality=40)
            
            for pkt_bytes, pkt_i in packets:
                pkt_arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                fec_bits = link.fec_codec.encode(bits)
                tx_signal = link.transmit(fec_bits)
                
                # Normalize manually as SDR_TX_send(normalize=True) does
                # Peak normalization is standard for DACs to avoid clipping.
                max_val = np.max(np.abs(tx_signal))
                if max_val > 0:
                    tx_signal = tx_signal / max_val
                
                channel.write(tx_signal)
                
                duration = len(tx_signal) / sdr_cfg.fs
                time.sleep(duration * 0.9) 
            
            frame_idx += 1
            if frame_idx >= 60: break
            
    except Exception as e:
        print(f"[TX] Error: {e}")
    finally:
        print("[TX] Stopped.")

# =============================================================================
# RX Node
# =============================================================================

def run_rx_node(output_file, config_file="sdr_tuned_config.json"):
    print(f"[RX] Loading config from {config_file}...")
    sdr_cfg = SDRConfig.load_from_json(config_file)
    fec_cfg = FECConfig(enabled=True, fec_type=FECType.LDPC)
    
    # Custom OFDM Config for Robustness
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = 60.0 # Increased from 5.0 to reject noise. Signal Peak ~160, Noise ~9. 
    
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg, simulation_mode=True)
    
    channel.rx_gain_db = float(sdr_cfg.rx_gain)
    channel.fs = sdr_cfg.fs
    print(f"[RX] Simulating RX Gain: {channel.rx_gain_db} dB, Fs: {channel.fs} Hz")
    
    agc = DigitalAGC(target_level=0.5, alpha=0.1) 
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (320, 240))
    buffer_chunk_size = 65536 
    frame_buffer = {}
    recovered_count = 0
    crc_fails = 0
    
    try:
        while not stop_event.is_set():
            rx_signal = channel.read(buffer_chunk_size)
            if rx_signal is None:
                time.sleep(0.01)
                continue
            
            # Robustness: AGC
            rx_signal_norm = agc.process(rx_signal)
            
            # Debug AGC & SNR
            if np.random.rand() < 0.1:
               sig_pwr = np.mean(np.abs(rx_signal)**2)
               print(f"[RX Debug] AGC Gain: {agc.gain:.1f}, Rx Pwr: {sig_pwr:.6f}")
            
            # Sync
            synced_sig, sync_metrics = link._synchronize(rx_signal_norm)
            
            if sync_metrics.get('sync_success'):
                # Diagnostic: Check Quality of Sync
                peak_val = sync_metrics.get('peak_val', 0)
                cfo_val = sync_metrics.get('cfo_est', 0)
                print(f"[RX Diag] Sync! Peak: {peak_val:.1f}, CFO: {cfo_val:.1f} Hz")
                
                rx_fec_bits, met = link.transceiver.demodulate(synced_sig)
                try:
                    rx_bits_dec = link.fec_codec.decode(rx_fec_bits)
                except:
                    continue
                
                rx_bytes = np.packbits(rx_bits_dec).tobytes()
                info = link.video_codec.parse_packet_header(rx_bytes)
                if info:
                    if zlib.crc32(info['payload']) & 0xFFFFFFFF != info['crc']:
                        crc_fails += 1
                        continue
                        
                    fid = info['frame_id']
                    if fid not in frame_buffer:
                        frame_buffer[fid] = {'pkts': {}, 'total': info['total_pkts'], 'ts': time.time()}
                        
                    frame_buffer[fid]['pkts'][info['pkt_idx']] = rx_bytes
                    
                    if len(frame_buffer[fid]['pkts']) == frame_buffer[fid]['total']:
                         pkt_list = [frame_buffer[fid]['pkts'][i] for i in range(frame_buffer[fid]['total'])]
                         frame = link.video_codec.decode_packets(pkt_list)
                         if frame is not None:
                             out.write(frame)
                             recovered_count += 1
                             print(f"[RX] Recovered Frame {fid}. AGC Gain: {agc.gain:.1f}")
                         del frame_buffer[fid]

            now = time.time()
            expired = [k for k,v in frame_buffer.items() if now - v['ts'] > 2.0]
            for k in expired: del frame_buffer[k]
                    
    except Exception as e:
        print(f"[RX] Error: {e}")
    finally:
        out.release()
        print(f"[RX] Stopped. Recovered: {recovered_count}, CRC Fails: {crc_fails}")

# =============================================================================
# Main
# =============================================================================

def main():
    print("=== Enhanced Async E2E Simulation ===")
    
    config_file = "sdr_tuned_config.json"
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump({
                "sdr_ip": "ip:192.168.2.1",
                "rx_uri": "ip:192.168.2.2",
                "device": "pluto",
                "fc": 915e6,
                "fs": 2e6,
                "bandwidth": 2e6,
                "tx_gain": 0,
                "rx_gain": 40,
                "rx_buffer_size": 262144
            }, f)

    # 1. Weak Signal
    print("\n[Scenario] Weak Signal (Input -30dBFS)...")
    channel.path_loss_db = 40.0 # Ideal
    channel.tx_power_db = 0.0
    channel.rx_gain_db = 40.0
    channel.base_noise_dbfs = -120.0 # Practically zero noise
    channel.cfo_hz = 100.0 # Small CFO
    
    print("Starting Threads...")
    rx_thread = threading.Thread(target=run_rx_node, args=("sim_enhanced_out.avi", config_file))
    tx_thread = threading.Thread(target=run_tx_node, args=("test_video.mp4", config_file))
    
    rx_thread.start()
    tx_thread.start()
    
    try:
        # Run 15s Weak
        time.sleep(15)
        
        # 2. Saturation
        print("\n[Scenario] Strong Signal (Input > 0dBFS) -> Saturation!")
        channel.path_loss_db = 10.0
        # Wait for AGC
        
        time.sleep(15)
        
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        stop_event.set()
        tx_thread.join()
        rx_thread.join()
        print("Done.")

if __name__ == "__main__":
    main()
