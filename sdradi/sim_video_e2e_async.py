
import threading
import queue
import time
import numpy as np
import cv2
import struct
import zlib
import json
import os
import argparse
import collections
from sdr_video_comm import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType, OTFSConfig, OFDMTransceiver, OTFSTransceiver, PacketFramer, robust_synchronize, FECCodec, LDPC5GCoder
from sdr_mac import SDRMACLayer

# =============================================================================
# Channel Simulation (The "Air")
# =============================================================================

# =============================================================================
# Helper: Latency Tracker
# =============================================================================

class LatencyTracker:
    """Tracks latency across pipeline stages."""
    def __init__(self):
        self.stats = collections.defaultdict(list)
        self.lock = threading.Lock()
        
    def log(self, stage, duration_sec):
        with self.lock:
            self.stats[stage].append(duration_sec)
            # Keep history small
            if len(self.stats[stage]) > 100:
                self.stats[stage] = self.stats[stage][-100:]
                
    def get_avg(self, stage):
        with self.lock:
            if stage not in self.stats or not self.stats[stage]:
                return 0.0
            return np.mean(self.stats[stage])
            
    def report(self):
        with self.lock:
            s = []
            for stage, vals in self.stats.items():
                if vals:
                    avg = np.mean(vals) * 1000.0 # ms
                    s.append(f"{stage}={avg:.1f}ms")
            return ", ".join(s)

latency_tracker = LatencyTracker()

# =============================================================================
# Helper: PRBS Generator
# =============================================================================

class PRBSGenerator:
    """Pseudo-Random Bit Sequence Generator/Checker."""
    def __init__(self, seed=0xACE1):
        self.state = seed
        self.seed = seed
        
    def generate(self, n_bits):
        if not hasattr(self, 'rng'):
             self.rng = np.random.RandomState(self.seed)
        bits = self.rng.randint(0, 2, size=n_bits)
        return bits
        
    def reset(self):
        self.rng = np.random.RandomState(self.seed)

class ChannelState:
    """Thread-safe Channel Buffer."""
    def __init__(self, capacity=1000000):
        self.buffer = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()
        
    def write(self, samples):
        with self.lock:
            self.buffer.extend(samples)
            
    def read(self, num_samples):
        with self.lock:
            if len(self.buffer) < num_samples:
                return None
            
            # Pop left
            chunk = np.array([self.buffer.popleft() for _ in range(num_samples)])
            return chunk

channel = ChannelState()
stop_event = threading.Event()

# Shared Statistics for FER
stats = {
    'tx_packets': 0,
    'rx_packets': 0,
    'rx_syncs': 0,
    'rx_decodes': 0
}

class SharedChannel:
    """
    Simulates the physical medium between TX and RX.
    Handles shared buffering, non-linearities, frequency offsets, and noise.
    """
    def __init__(self, fs=2e6, max_buffer_len=10000000):
        self.lock = threading.Lock()
        self.buffer = [] 
        self.fs = fs
        self.max_buffer_len = max_buffer_len
        
        # Hardware Constraints & Impairments
        self.tx_power_db = 0.0 # dBFS relative to unity
        self.path_loss_db = 0.0 # Path loss
        self.rx_gain_db = 0.0   # RX LNA + VGA gain
        self.base_noise_dbfs = -120.0 # Start clean
        
        self.cfo_hz = 0.0
        self.jitter_std_dev = 0.0  # Jitter intensity
        self.saturation_limit = 1.0 
        
        self.total_samples_written = 0
        self.total_samples_read = 0
        
    def write(self, samples):
        """TX pushes samples to the air."""
        if len(samples) == 0: return
        
        with self.lock:
            # Apply TX Power
            # If TX is normalized to 1.0, power is 0 dBFS.
            # tx_power_db scales it down/up.
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
        
        # Jitter (Time Shift / Sample Drop/Add)
        if self.jitter_std_dev > 0 and len(chunk) > 10:
            # Simple jitter: Randomly drop or duplicate a sample occasionally
            # Rate: ~1 event per 10000 samples scaled by jitter factor
            if np.random.rand() < 0.001 * self.jitter_std_dev:
                # 50% chance drop, 50% dup
                if np.random.rand() > 0.5:
                     # Drop one sample
                     chunk = chunk[:-1]
                else:
                     # Dup one sample
                     chunk = np.concatenate([chunk, chunk[-1:]])

        # CFO
        if self.cfo_hz != 0.0:
            # We must maintain phase continuity based on absolute time
            start_idx = self.total_samples_read - num_samples
            t = (np.arange(num_samples) + start_idx) / self.fs
            # Handle potential length mismatch from jitter
            if len(t) != len(chunk):
                t = t[:len(chunk)] 
            cfo_vec = np.exp(1j * 2 * np.pi * self.cfo_hz * t)
            chunk = chunk * cfo_vec
            
        # Noise
        noise_amp = 10 ** (self.base_noise_dbfs / 20.0)
        # Random noise
        noise = (np.random.randn(len(chunk)) + 1j * np.random.randn(len(chunk))) / np.sqrt(2) * noise_amp
        chunk += noise
        
        # ADC Saturation
        chunk = np.clip(chunk, -self.saturation_limit-1j*self.saturation_limit, self.saturation_limit+1j*self.saturation_limit)
        
        # ADC Saturation
        chunk = np.clip(chunk, -self.saturation_limit-1j*self.saturation_limit, self.saturation_limit+1j*self.saturation_limit)
        
        return chunk

    def clear(self):
        with self.lock:
            self.buffer = []
            self.total_samples_written = 0
            self.total_samples_read = 0

    def set_impairments(self, noise_db=-120, cfo=0, jitter=0):

        with self.lock:
            self.base_noise_dbfs = noise_db
            self.cfo_hz = cfo
            self.jitter_std_dev = jitter
            print(f"[Channel] Config: Noise={noise_db}dB, CFO={cfo}Hz, Jitter={jitter}")


# Global Channel
channel = SharedChannel(fs=2e6)
stop_event = threading.Event()
stop_tx_event = threading.Event()

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
# Zadoff-Chu Generator (Matches Transceiver)
# =============================================================================
def generate_zc_sequence(length, root):
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)

# =============================================================================
# TX Node
# =============================================================================

def run_tx_node(video_file, waveform="otfs", target_fps=15):
    print(f"[TX] Starting TX Node (Waveform: {waveform})...")
    
    # Initialize Transceiver
    if waveform == "otfs":
        cfg = OTFSConfig(mod_order=2) # BPSK
        tr = OTFSTransceiver(cfg)
    else:
        cfg = OFDMConfig(mod_order=2) # BPSK
        tr = OFDMTransceiver(cfg)
        
    # FEC (LDPC)
    # Using Rate 1/2, Block Size k=8192, n=16384
    # This matches OTFS Frame Size (64x256 * 1 bit/sym = 16384) perfectly.
    fec_cfg = FECConfig(enabled=True, fec_type=FECType.LDPC, code_rate="1/3") 
    try:
        fec = LDPC5GCoder(fec_cfg)
    except Exception as e:
        print(f"[TX] Error initializing LDPC: {e}")
        return

    # Check Alignment
    bits_per_frame = cfg.bits_per_frame
    if fec.n != bits_per_frame:
        print(f"[TX Warning] FEC Block Size ({fec.n}) != PHY Frame Size ({bits_per_frame}). Padding will be used.")
        
    # ZC Sequence for Preamble/Sync
    zc = generate_zc_sequence(127, 25)
    
    # Video Source
    cap = cv2.VideoCapture(video_file)
    use_synthetic = False
    if not cap.isOpened():
        print(f"[TX] Video file not found. Using Synthetic Video.")
        use_synthetic = True
        
    width, height = 320, 240
    frame_idx = 0
    seq_counter = 0
    
    last_report_time = time.time()

    
    try:
        while not stop_tx_event.is_set():
            if not use_synthetic:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            else:
                # Synthetic bouncing box
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (frame_idx*5 % (width-30), 50), (frame_idx*5 % (width-30) + 30, 80), (0, 0, 255), -1)
                cv2.putText(frame, f"Seq: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                time.sleep(0.05)
                
            frame = cv2.resize(frame, (width, height))
            
            # Compress Frame (JPEG)
            t_enc_start = time.time()
            ret, jpg_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame_bytes = jpg_bytes.tobytes()
            
            # Frame Payload
            # We just send the raw JPEG bytes. The PacketFramer handles segmentation.
            
            # Manual Chunking and Framing (PacketFramer)
            current_idx = 0
            MAX_PAYLOAD = 1000 # Keep it under standard MTU
            packets = []

            while current_idx < len(frame_bytes):
                chunk_len = min(MAX_PAYLOAD, len(frame_bytes) - current_idx)
                chunk = frame_bytes[current_idx : current_idx + chunk_len]
                
                pkt = PacketFramer.frame(chunk, seq_counter)
                packets.append(pkt)
                
                stats['tx_packets'] += 1
                seq_counter = (seq_counter + 1) % 65536
                current_idx += chunk_len
            
            # 1. Stream bits
            stream_bytes = b"".join(packets)
            stream_bits = np.unpackbits(np.frombuffer(stream_bytes, dtype=np.uint8))
            
            # 2. FEC Encoding (Rate 1/3) & Fragmentation (MAC)
            # Re-init FEC at Rate 1/3 just to be safe (or assume fec object is Rate 1/3 if updated above)
            # We assume 'fec' is already initialized.
            
            # We must slice stream_bits into 'k' sized blocks
            # k for Rate 1/3 (if n=16384) ~ 5461. 
            # We'll use fec.k dynamically.
            
            mac = SDRMACLayer()
            bits_per_frame = cfg.bits_per_frame # 16384
            # Leave room for header (4096 Pad + 32 Magic + 32 Fields = 4160)
            mac_frag_size = bits_per_frame - 4160 
            
            current_bit_idx = 0
            
            while current_bit_idx < len(stream_bits):
                # Take 1 Code Block (k bits)
                k_size = fec.k
                end = min(current_bit_idx + k_size, len(stream_bits))
                chunk = stream_bits[current_bit_idx : end]
                current_bit_idx = end
                
                # Encode (Output size n)
                encoded_chunk = fec.encode(chunk)
                
                # DEBUG: Force Calibration Pattern 0x55 (0,1,0,1...)
                encoded_chunk = np.tile([0, 1], len(encoded_chunk)//2)
                
                # MAC Fragmentation
                # Split Encoded Block into Fragments that fit in PHY Frame
                fragments = mac.fragment_packet(encoded_chunk, mac_frag_size)
                
                # Combine Fragments for Continuous Transmission (Burst)
                # Ensure each fragment is padded to frame align if needed?
                # mac_frag_size + 32 = bits_per_frame. 
                # So fragments are exactly bits_per_frame size.
                # If last fragment is short? 
                # SDRMACLayer.fragment_packet produces short last fragment.
                # We must PAD the last fragment to bits_per_frame to maintain timing alignment.
                
                aligned_fragments = []
                for frag in fragments:
                     if len(frag) < bits_per_frame:
                         padding = np.zeros(bits_per_frame - len(frag), dtype=int)
                         aligned_fragments.append(np.concatenate([frag, padding]))
                     else:
                         aligned_fragments.append(frag)
                         
                # Loop Fragments and Send Independently
                for frag in aligned_fragments:
                    # Modulate
                    t_phy_start = time.time()
                    tx_signal = tr.modulate(frag)
                    t_phy_end = time.time()
                    latency_tracker.log("TX_Phy", t_phy_end - t_phy_start)
                    
                    # Sync Preamble (Per Fragment)
                    preamble = np.concatenate([np.zeros(50), zc, np.zeros(50)])
                    # Tone for CFO
                    tone = np.exp(1j * 2 * np.pi * np.arange(256) * (16.0/256.0))
                    sync_block = np.concatenate([preamble, tone, np.zeros(50)]) 
                    
                    full_signal = np.concatenate([sync_block, tx_signal, np.zeros(1000)]) 
                    
                    # Normalize
                    max_val = np.max(np.abs(full_signal))
                    if max_val > 0:
                        full_signal = full_signal / max_val
                    
                    # Send
                    channel.write(full_signal)
                    
                    # Rate Pacing (OS Sleep gap)
                    # duration = len(full_signal) / 2e6 
                    # Reduced to avoid starvation width. Use 10MSPS rate.
                    duration = len(full_signal) / 10e6
                    time.sleep(duration * 0.5) 
            
            # Maintain Video FPS
            elapsed = time.time() - t_enc_start
            wait = (1.0 / target_fps) - elapsed
            if wait > 0:
                time.sleep(wait)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"[TX] Sent Frame {frame_idx} ({len(packets)} packets)")
                if time.time() - last_report_time > 5.0:
                    print(f"[TX] {latency_tracker.report()}")
                    last_report_time = time.time()

            
    except Exception as e:
        print(f"[TX] Error: {e}")
    finally:
        print("[TX] Stopped.")

# =============================================================================
# RX Node
# =============================================================================

def run_rx_node(output_file, waveform="otfs"):
    print(f"[RX] Starting RX Node (Waveform: {waveform})...")
    
    # Initialize Transceiver
    if waveform == "otfs":
        cfg = OTFSConfig(mod_order=2)
        tr = OTFSTransceiver(cfg)
    else:
        cfg = OFDMConfig(mod_order=2)
        tr = OFDMTransceiver(cfg)
        
    # FEC
    # Matching TX Update: Use LDPC5GCoder directly with Rate 1/3 (Fragmented)
    fec_cfg = FECConfig(enabled=True, fec_type=FECType.LDPC, code_rate="1/3")
    try:
         fec = LDPC5GCoder(fec_cfg)
    except:
         print("[RX] Error init LDPC decoder (LDPC5GCoder)")
         return
        
    zc = generate_zc_sequence(127, 25)
    agc = DigitalAGC(target_level=0.5, alpha=0.1)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (320, 240))
    
    # Buffers
    rx_buffer = np.array([], dtype=complex)
    read_chunk_size = 4096
    
    # Assembler
    byte_accumulator = bytearray()
    
    total_packets = 0
    valid_packets = 0
    
    try:
        # Persistent MAC
        mac = SDRMACLayer()
        while not stop_event.is_set():
            # 1. Read from Channel
            new_samples = channel.read(read_chunk_size)
            if new_samples is None:
                time.sleep(0.001)
                continue
                
            rx_buffer = np.concatenate([rx_buffer, new_samples])
            
            # AGC
            rx_buffer = agc.process(rx_buffer)
            
            # 2. Process Buffer
            # Buffer Flow Control: If we are too far behind, drop old samples
            # Increased to 500,000 (approx 250ms) to handle severe jitter without dropping packets
            if len(rx_buffer) > 500000:
                print(f"[RX] Buffer Overflow ({len(rx_buffer)}). Dropping oldest 200000 samples.")
                rx_buffer = rx_buffer[200000:]
            
            # We search for ZC correlation peaks
            # OTFS Frame is ~17000 samples. Burst is ~4 Frames (68k).
            # We need window large enough to capture FULL burst.
            while len(rx_buffer) > 18000: 
                # Extract a working window (Max 150k to cover multiple bursts)
                window_len = min(len(rx_buffer), 150000)
                window = rx_buffer[:window_len]
                
                # Synchronize
                payload, H_est, metrics = robust_synchronize(window, zc)
                
                if metrics['sync_success']:
                    stats['rx_syncs'] += 1
                    peak = metrics['peak_val']
                    noise = metrics['noise_floor']
                    # ratio = peak / (noise + 1e-9)
                    # print(f"[RX Debug] Sync: Peak={peak:.1f} Noise={noise:.1f}")

                    peak_idx = metrics['peak_idx']
                    cfo_rad = metrics['cfo_rad']
                    corr_complex = metrics['corr_complex']
                    
                    # --- Robust Decoding Logic (Triple Search) ---
                    # 1. Timing Search Loop (+/- 2 samples)
                    # 2. Phase Hypothesis Loop (Normal / Inverted)
                    # 3. Bit Shift Loop
                    
                    found_good_packet = False
                    
                    # Optimization: Early Exit Search Order
                    # Prioritize t_off=0, then +/-1.
                    timing_offsets = [0, -1, 1, -2, 2] 
                    base_payload_start = peak_idx + 127 + 50 + 256 + 50
                    
                    for t_off in range(-10, 51):
                        valid_hypothesis = False
                        
                        # Process Single Frame
                        p_start = base_payload_start + t_off
                        L_payload = 16384
                        
                        if p_start + L_payload > len(window):
                            break
                            
                        # Extract and Correct
                        t_payload_raw = window[p_start : p_start+L_payload]
                        t_vec = np.arange(len(t_payload_raw))
                        correction = np.exp(-1j * cfo_rad * t_vec)
                        t_payload = t_payload_raw * correction
                        
                        pwr = np.mean(np.abs(t_payload)**2)
                        if pwr > 0: t_payload /= np.sqrt(pwr)

                        if waveform == "otfs":
                            bits, _ = tr.demodulate(t_payload, channel_est=H_est)
                        else:
                            bits, _ = tr.demodulate(t_payload)
                        
                        rx_bits = bits
                        # Tolerance 4 (Stricter)
                        valid = mac.validate_header(rx_bits, tolerance=4)
                        pola = "Normal"
                        if not valid:
                            rx_bits = -bits
                            if mac.validate_header(rx_bits, tolerance=4):
                                valid = True
                                pola = "Inverted"
                                
                        if not valid: continue
                        
                        # Found valid header 
                        # Debug Header (Skip DC Pad 4096)
                        h_bits = (rx_bits[4096:4160] < 0).astype(np.uint8)
                        h_hex = np.packbits(h_bits).tobytes().hex()
                        print(f"[RX Debug] Sync Lock! Peak={peak_idx}, Off={t_off}, Pol={pola}, Head={h_hex}")

                        # Feed to Persistent MAC
                        reassembled = mac.process_fragment(rx_bits)
                        valid_hypothesis = True
                        
                        if reassembled is not None:
                            try:
                                decoded_bits = fec.decode(reassembled)
                                stats['rx_decodes'] += 1
                                print(f"[RX Debug] Reassembled Len={len(reassembled)}, Decoded Len={len(decoded_bits)}")
                                
                                rx_bytes = np.packbits(decoded_bits).tobytes()
                            except:
                                print(f"[RX] MAC Reassembled but Decode Failed")
                                continue
                            
                            byte_accumulator.extend(rx_bytes)
                            
                            packets_found = PacketFramer.deframe(byte_accumulator)
                            v_cnt = sum(1 for _, _, v in packets_found if v)
                            
                            if v_cnt > 0:
                                stats['rx_packets'] += v_cnt
                                found_good_packet = True
                                valid_hypothesis = True
                                
                                for p_bytes, _, p_valid in packets_found:
                                    if p_valid:
                                        try:
                                            frame_arr = np.frombuffer(p_bytes, dtype=np.uint8)
                                            frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
                                            if frame is not None:
                                                out.write(frame)
                                        except: pass
                                
                                # Cleanup
                                if len(byte_accumulator) > 20000:
                                    del byte_accumulator[:-4000]
                                
                                break
                                
                        if found_good_packet: break
                                
                    # Advance buffer relative to detected peak to ensure we don't skip next frame
                    # Frame ~ 17000. Advance safe amount to skip current Preamble+Header but keep next.
                    # If we skip 20000, we might skip next preamble if frames are back-to-back.
                    consume_len = peak_idx + 15000
                    rx_buffer = rx_buffer[consume_len:] 
                    
                else:
                    rx_buffer = rx_buffer[1000:]
                    
            # Try decoding Accumulator (Naive Stream Decoder)
            if len(byte_accumulator) > 10000:
                pass
                
            if valid_packets > 0 and total_packets > 0:
                 if valid_packets % 10 == 0:
                     print(f"[RX] PER: {(1 - valid_packets/total_packets)*100:.1f}% ({valid_packets}/{total_packets})")
            
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc(file=sys.stdout)
        print(f"[RX] Error: {e}")
    finally:
        out.release()
        print(f"[RX] Stopped. Total Pkts: {total_packets}, Valid: {valid_packets}")
        fer = 1.0 - (stats['rx_packets']/max(1, stats['tx_packets']))
        print("-" * 60)
        print("=== FINAL STATISTICS REPORT ===")
        print(f"1. Transmitted Packets (TX App): {stats['tx_packets']}")
        print(f"   - Total video data packets sent by Transmitter.")
        print(f"2. Detected Frames (RX PHY):     {stats['rx_syncs']}")
        print(f"   - Number of physical layer frames synchronized (Zadoff-Chu detected).")
        print(f"   - If this is low, signals are missed due to noise or buffer drops.")
        print(f"3. Decoded FEC Blocks (RX FEC):  {stats['rx_decodes']}")
        print(f"   - Frames successfully LDPC decoded.")
        print(f"   - If high but 'Decoded Packets' is low => CRC/Parsing Failure (Payload Corruption).")
        print(f"4. Decoded Packets (RX App):     {stats['rx_packets']}")
        print(f"   - Packets successfully extracted, CRC-validated, and passed to Application.")
        print(f"5. Frame Error Rate (FER):       {fer*100:.2f}%")
        print(f"   - Loss calculated as 1 - (RX_App / TX_App).")
        print(f"   - High FER (>10%) in async sim is often due to stopping alignment (TX stops before RX finishes).")
        print("-" * 60)

# =============================================================================
# Main
# =============================================================================


def run_prbs_test(waveform="otfs", snr_levels=[-5, 0, 5, 10, 20], jitter_levels=[0]):
    print(f"=== PRBS Impairment Test (Waveform: {waveform}) ===")
    
    # Setup
    prbs_gen = PRBSGenerator(seed=0x1234)
    if waveform == "otfs":
        cfg = OTFSConfig(mod_order=2)
        tr = OTFSTransceiver(cfg)
    else:
        cfg = OFDMConfig(mod_order=2)
        tr = OFDMTransceiver(cfg)
        
    # We transmit a fixed number of frames per scenario
    FRAMES_PER_TEST = 20
    BITS_PER_FRAME = cfg.bits_per_frame
    
    for snr in snr_levels:
        for jitter in jitter_levels:
            print(f"\n[Test Case] SNR={snr}dB, Jitter={jitter}")
            channel.set_impairments(noise_db=-120, cfo=0, jitter=0) # Reset first
            time.sleep(1) # Clear buffer
            
            # Configure Channel
            # Calculate Noise Power for desired SNR
            # Signal Power is roughly 1.0 (normalized) -> 0 dB
            # SNR = S_db - N_db => N_db = S_db - SNR = 0 - SNR = -SNR
            # Base noise dbfs = -SNR.
            channel.set_impairments(noise_db=-snr, cfo=0, jitter=jitter)
            
            # Update Transceiver Config for MMSE
            # Pass the known SNR to the receiver (Genie-Aided MMSE)
            if hasattr(tr.config, 'snr_est_db'):
                tr.config.snr_est_db = float(snr)
            
            total_bits = 0
            error_bits = 0
            latencies = []
            
            for i in range(FRAMES_PER_TEST):
                # Clear channel to ensure we get corresponding frame
                channel.clear()
                
                # Generate Data

                tx_bits = prbs_gen.generate(BITS_PER_FRAME)
                
                # Encode/Modulate
                t0 = time.time()
                # No FEC in this raw mode, or use FEC? 
                # User asked: "Impairment Testing (No FEC/Video)"
                # So we test raw Phy performance or Uncoded BER.
                
                mod_syms = tr.modulate(tx_bits)
                
                # Sync Preamble
                zc = generate_zc_sequence(127, 25)
                preamble = np.concatenate([np.zeros(50), zc, np.zeros(50)])
                # Add significant padding at end to accommodate jitter and prevent underflow on read
                full_signal = np.concatenate([preamble, mod_syms, np.zeros(1000)])
                
                # Send
                t1 = time.time()
                channel.write(full_signal)
                
                # Receive (Direct Read for Sync Test)
                time.sleep(len(full_signal)/2e6 * 1.5) # generous wait
                
                # We expect to read back roughly len(full_signal)
                # But to cover jitter, we ask for slightly less than the full padded amount?
                # Or we just assume the 1000 zeros covered us.
                # Let's read len(full_signal) - 100 to be safe from minor drops?
                # NO. If we jitter ADD, we have more. If DROP, less.
                # If we read too much, we fail.
                # Better: Read all available? channel class doesn't support it directly but we can try.
                # Helper: Peek length? 
                # Let's just rely on the padding.
                # If we wrote N + 1000.
                # We read N. We surely have N samples unless jitter dropped > 1000 (unlikely).
                
                read_len = len(full_signal) - 500
                rx_samples = channel.read(read_len)
                if rx_samples is None:
                     print("  [Warn] Buffer underflow/dropped packet")
                     error_bits += len(tx_bits) // 2 
                     continue

                     
                # Sync
                payload, H_est, info = robust_synchronize(rx_samples, zc)
                
                if not info['sync_success']:
                    print("  [Fail] Sync lost")
                    error_bits += len(tx_bits) // 2 
                else:
                    peak_idx = info['peak_idx']
                    # Standard offset from ZC start to payload start
                    # Preamble: 50(Zero) + 127(ZC) + 50(Zero)
                    # PRBS Mode DOES NOT include the Tone (256) + 50(Zero) used in Video Mode!
                    # So offset is 50 + 127 + 50 = 227? 
                    # Let's check construction:
                    # preamble = np.concatenate([np.zeros(50), zc, np.zeros(50)]) (Len 227)
                    # full_signal = np.concatenate([preamble, mod_syms, zeros(100)])
                    # ZC starts at index 50 of full_signal.
                    # Sync returns peak_idx relative to rx_samples start.
                    # If peak corresponds to ZC start (or center? robust_synch usually Start?),
                    # then Payload starts at peak_idx + 127 + 50.
                    
                    # Search range
                    base_start = peak_idx + 127 + 50 
                    found_len = len(mod_syms)
                    
                    # Try offsets
                    best_rx_bits = None
                    best_ber = 1.0
                    
                    peak_idx = info['peak_idx']
                    # Revert to Manual Extraction (Robust Sync CFO is unreliable on PRBS data)
                    base_start = peak_idx + 127 + 50
                    found_len = len(mod_syms)
                    
                    best_ber = 1.0
                    best_rx_bits = None
                    
                    # Search range
                    for t_off in range(-2, 3):
                        start = base_start + t_off
                        if start < 0 or start + found_len > len(rx_samples):
                            continue
                            
                        rx_chunk = rx_samples[start : start + found_len]
                        
                        # Demodulate
                        try:
                            # Pass channel estimate if available (for MMSE)
                            rx_bits_cand, metrics = tr.demodulate(rx_chunk)
                        except: continue
                            
                        n = min(len(tx_bits), len(rx_bits_cand))
                        errs = np.sum(tx_bits[:n] != rx_bits_cand[:n])
                        len_diff = abs(len(tx_bits) - len(rx_bits_cand))
                        errs += len_diff
                        ber_cand = errs / len(tx_bits)
                        
                        if ber_cand < best_ber:
                            best_ber = ber_cand
                            best_rx_bits = rx_bits_cand

                    error_bits += (best_ber * len(tx_bits))
                    
                t2 = time.time()
                latencies.append((t2-t0)*1000)
                total_bits += len(tx_bits)
                
            ber = error_bits / total_bits if total_bits > 0 else 1.0
            avg_lat = np.mean(latencies) if latencies else 0
            print(f"  -> Result: BER={ber:.6f}, Avg Latency={avg_lat:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description="Async E2E Simulation")
    parser.add_argument("--mode", type=str, default="video", choices=["video", "prbs"], help="Simulation mode")
    parser.add_argument("--waveform", type=str, default="otfs", choices=["otfs", "ofdm"], help="Waveform type")
    args = parser.parse_args()
    
    if args.mode == "prbs":
        run_prbs_test(waveform=args.waveform)
        return

    print(f"=== Enhanced Async E2E Simulation ({args.waveform.upper()}) ===")
    
    # Scenarios
    print("\n[Scenario 1] Weak Signal (-120dB Noise, Ideal Path)")
    channel.set_impairments(noise_db=-120)
    
    print("Starting Threads...")
    rx_thread = threading.Thread(target=run_rx_node, args=("sim_out.avi", args.waveform))
    tx_thread = threading.Thread(target=run_tx_node, args=("test_video.mp4", args.waveform))
    
    rx_thread.start()
    tx_thread.start()
    
    try:
        # Phase 1: Good Channel
        time.sleep(2)
        print(f"[Report] Scenario 1 Stats: Sent={stats['tx_packets']}, RX_PHY={stats['rx_syncs']}, RX_Dec={stats['rx_decodes']}, RX_App={stats['rx_packets']}")
        
        # Phase 2: Add Noise
        print("\n[Scenario 2] Adding Noise (-60dB)")
        channel.set_impairments(noise_db=-60)
        time.sleep(2)
        print(f"[Report] Scenario 2 Stats: Sent={stats['tx_packets']}, RX_PHY={stats['rx_syncs']}, RX_Dec={stats['rx_decodes']}, RX_App={stats['rx_packets']}")
        
        # Phase 3: Add CFO & Timing Jitter
        print("\n[Scenario 3] Adding CFO (+200Hz) & Saturation")
        channel.set_impairments(noise_db=-60, cfo=200, jitter=0.5)
        # channel.tx_power_db = 10.0 # Drive into saturation (optional)
        time.sleep(5)
        print(f"[Report] Scenario 3 Stats: Sent={stats['tx_packets']}, RX_PHY={stats['rx_syncs']}, RX_Dec={stats['rx_decodes']}, RX_App={stats['rx_packets']}")
        
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        stop_tx_event.set()
        tx_thread.join()
        print("TX Stopped. Draining RX buffer (5s)...")
        time.sleep(5.0)
        stop_event.set()
        rx_thread.join()
        print("Done.")

if __name__ == "__main__":
    main()
