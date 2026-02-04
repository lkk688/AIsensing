
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
from sdr_video_commv3 import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

_GF256_EXP = np.zeros(512, dtype=np.uint8)
_GF256_LOG = np.zeros(256, dtype=np.int16)

def _gf256_init():
    x = 1
    for i in range(255):
        _GF256_EXP[i] = x
        _GF256_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11d
    for i in range(255, 512):
        _GF256_EXP[i] = _GF256_EXP[i - 255]

_gf256_init()

def gf256_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return int(_GF256_EXP[int(_GF256_LOG[a]) + int(_GF256_LOG[b])])

def gf256_inv(a):
    if a == 0:
        return 0
    return int(_GF256_EXP[255 - int(_GF256_LOG[a])])

def gf256_pow(a, p):
    if p == 0:
        return 1
    if a == 0:
        return 0
    return int(_GF256_EXP[(int(_GF256_LOG[a]) * int(p)) % 255])

def gf256_mul_vec(vec, coeff):
    if coeff == 0:
        return np.zeros_like(vec)
    if coeff == 1:
        return vec.copy()
    log_coeff = int(_GF256_LOG[coeff])
    logs = _GF256_LOG[vec]
    res = _GF256_EXP[(logs + log_coeff) % 255]
    res = res.astype(np.uint8, copy=False)
    res[vec == 0] = 0
    return res

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
stats_lock = threading.Lock()
stats = {
    'tx_packets': 0,
    'rx_packets': 0,
    'crc_fails': 0,
    'frames_recovered': 0,
    'tx_frames': 0,
    'frames_started': 0,
    'frames_expired': 0,
    'frames_dropped': 0,
    'frames_decode_fail': 0,
    'frames_oversize': 0,
    'mac_recovered_frames': 0,
    'mac_recovered_packets': 0,
    'header_invalid': 0,
    'tx_lookup_miss': 0,
    'pre_fec_bit_errors': 0,
    'pre_fec_bit_total': 0,
    'post_fec_bit_errors': 0,
    'post_fec_bit_total': 0,
    'sync_attempts': 0,
    'sync_successes': 0,
    'sync_incomplete': 0,
    'header_resync': 0,
    'frames_total_pkts_sum': 0,
    'frames_total_pkts_count': 0,
    'frames_rx_pkts_sum': 0,
    'frames_missing_pkts_sum': 0,
    'frames_expired_total_pkts_sum': 0,
    'frames_expired_missing_pkts_sum': 0,
    'frames_total_pkts_min': None,
    'frames_total_pkts_max': 0,
    'mac_fec_recovered_frames': 0,
    'mac_fec_recovered_packets': 0,
    'mac_fec_decode_fail': 0
}
tx_packet_store = {}
tx_packet_lock = threading.Lock()

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

def run_tx_node(video_file, config_file="sdr_tuned_config.json", allow_synthetic=True, width=320, height=240, quality=40, fec_cfg=None, gap_samples=0, sleep_factor=0.9, max_frames=60, pkt_repeats=1, log_interval=2.0, store_timeout=15.0, mac_parity=False, diag_level="basic", mac_parity_groups=1, header_repeat_bytes=0, repeat_strategy="per_packet", mac_fec_parity=0, mac_fec_seed=1, frame_gap_ms=0, frame_step=1, packet_size=None, max_packets_per_frame=None, adapt=False, adapt_interval=3.0, target_fer=0.2, target_sync_rate=0.7, target_post_ber=1e-3, quality_min=20, quality_max=70, quality_step=5, resolution_steps=None, hardware_mode=False, sdr_ip_override=None, rx_uri_override=None):
    print(f"[TX] Loading config from {config_file}...")
    sdr_cfg = SDRConfig.load_from_json(config_file)
    if sdr_ip_override:
        sdr_cfg.sdr_ip = sdr_ip_override
    if rx_uri_override is not None:
        sdr_cfg.rx_uri = rx_uri_override
    if hardware_mode:
        sdr_cfg.device = "pluto"
    fec_cfg = fec_cfg or FECConfig(enabled=True, fec_type=FECType.LDPC)
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, simulation_mode=not hardware_mode)
    if hardware_mode:
        if not link.connect_sdr():
            print("[TX] SDR connect failed")
            return
    else:
        if sdr_cfg.tx_gain > -80:
            channel.tx_power_db = float(sdr_cfg.tx_gain)
        print(f"[TX] Simulating TX Power: {channel.tx_power_db} dB")
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        if allow_synthetic:
            print(f"[TX] Using Synthetic Video")
            cap = None
        else:
            print(f"[TX] Error: Cannot open video file: {video_file}")
            return
        
    if packet_size is not None:
        packet_size = int(packet_size)
        if packet_size <= link.video_codec.HEADER_SIZE:
            packet_size = link.video_codec.HEADER_SIZE + 16
        link.video_config.packet_size = packet_size
    if max_packets_per_frame is not None:
        max_packets_per_frame = int(max_packets_per_frame)
        if max_packets_per_frame > 0:
            link.video_config.max_packets_per_frame = max_packets_per_frame

    if resolution_steps is None or len(resolution_steps) == 0:
        resolution_steps = [(width, height)]
    if (width, height) not in resolution_steps:
        resolution_steps = [(width, height)] + list(resolution_steps)
    try:
        res_index = resolution_steps.index((width, height))
    except ValueError:
        res_index = 0

    quality_min = int(quality_min)
    quality_max = int(quality_max)
    quality_step = max(1, int(quality_step))
    current_quality = int(quality)
    if current_quality < quality_min:
        current_quality = quality_min
    if current_quality > quality_max:
        current_quality = quality_max

    link.video_config.resolution = (width, height)
    frame_idx = 0
    src_frame_idx = 0
    
    last_log = time.time()
    next_adapt = time.time() + float(adapt_interval)
    try:
        while not stop_event.is_set():
            if adapt and time.time() >= next_adapt:
                with stats_lock:
                    tx_frames = stats['tx_frames']
                    frames_total = stats['frames_recovered']
                    pre_err = stats['pre_fec_bit_errors']
                    pre_total = stats['pre_fec_bit_total']
                    post_err = stats['post_fec_bit_errors']
                    post_total = stats['post_fec_bit_total']
                    sync_attempts = stats['sync_attempts']
                    sync_successes = stats['sync_successes']
                    frames_started = stats['frames_started']
                    frames_expired = stats['frames_expired']
                    total_pkts_sum = stats['frames_total_pkts_sum']
                    missing_pkts_sum = stats['frames_missing_pkts_sum']
                fer = 1.0 - (frames_total / tx_frames) if tx_frames > 0 else 0.0
                post_ber = (post_err / post_total) if post_total > 0 else 0.0
                sync_rate = (sync_successes / sync_attempts) if sync_attempts > 0 else 0.0
                missing_ratio = (missing_pkts_sum / total_pkts_sum) if total_pkts_sum > 0 else 0.0
                expire_ratio = (frames_expired / frames_started) if frames_started > 0 else 0.0
                target_missing_ratio = min(0.3, max(0.05, target_fer * 0.8))
                degrade = (
                    fer > target_fer
                    or sync_rate < target_sync_rate
                    or post_ber > target_post_ber
                    or missing_ratio > target_missing_ratio
                    or expire_ratio > (target_fer * 0.8)
                )
                improve = (
                    fer < (target_fer * 0.5)
                    and sync_rate >= target_sync_rate
                    and post_ber < (target_post_ber * 0.5)
                    and missing_ratio < (target_missing_ratio * 0.5)
                    and expire_ratio < (target_fer * 0.5)
                )
                changed = False
                if degrade:
                    if current_quality > quality_min:
                        current_quality = max(quality_min, current_quality - quality_step)
                        changed = True
                    elif res_index > 0:
                        res_index -= 1
                        width, height = resolution_steps[res_index]
                        link.video_config.resolution = (width, height)
                        changed = True
                elif improve:
                    if current_quality < quality_max:
                        current_quality = min(quality_max, current_quality + quality_step)
                        changed = True
                    elif res_index < len(resolution_steps) - 1:
                        res_index += 1
                        width, height = resolution_steps[res_index]
                        link.video_config.resolution = (width, height)
                        changed = True
                if changed and diag_level in ("basic", "phy", "mac", "all"):
                    print(f"[TX] Adapt -> {width}x{height} Q{current_quality}")
                next_adapt = time.time() + float(adapt_interval)
            if cap:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                src_frame_idx += 1
                if frame_step > 1 and (src_frame_idx % frame_step) != 1:
                    continue
            else:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.rectangle(frame, (frame_idx*5 % 300, 50), (frame_idx*5 % 300 + 30, 80), (0, 0, 255), -1)
                time.sleep(0.05)
                
            max_pkts_limit = int(link.video_config.max_packets_per_frame) if link.video_config.max_packets_per_frame else 0
            oversize_logged = False
            while True:
                frame = cv2.resize(frame, (width, height))
                packets = link.video_codec.encode_frame(frame, quality=current_quality)
                oversize = max_pkts_limit > 0 and len(packets) >= max_pkts_limit
                if oversize and not oversize_logged:
                    with stats_lock:
                        stats['frames_oversize'] += 1
                    oversize_logged = True
                if not oversize:
                    break
                if current_quality > quality_min:
                    current_quality = max(quality_min, current_quality - quality_step)
                elif res_index > 0:
                    res_index -= 1
                    width, height = resolution_steps[res_index]
                    link.video_config.resolution = (width, height)
                else:
                    break
            payload_size = link.video_codec.config.packet_size - link.video_codec.HEADER_SIZE
            repeat_len = max(0, int(header_repeat_bytes))
            final_packets = []
            frame_info = None

            if mac_fec_parity and int(mac_fec_parity) > 0:
                data_total = len(packets)
                parity_count = int(mac_fec_parity)
                if parity_count < 1:
                    parity_count = 1
                total_pkts = data_total + parity_count
                data_payloads = []
                for pkt_bytes, pkt_i in packets:
                    payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                    info = link.video_codec.parse_packet_header(pkt_bytes)
                    if info and frame_info is None:
                        frame_info = info
                    data_payloads.append(payload)
                if frame_info:
                    for pkt_bytes, pkt_i in packets:
                        payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                        rebuilt = link.video_codec.create_packet_header(payload, frame_info['frame_id'], pkt_i, total_pkts, frame_info['quality'])
                        final_packets.append(rebuilt)
                    base = (int(mac_fec_seed) + int(frame_info['frame_id'])) % 255
                    if base == 0:
                        base = 1
                    for p_idx in range(parity_count):
                        alpha = (base + p_idx) % 255
                        if alpha == 0:
                            alpha = 1
                        coeffs = np.fromiter((gf256_pow(alpha, i) for i in range(data_total)), dtype=np.uint8, count=data_total)
                        parity_payload = np.zeros(payload_size, dtype=np.uint8)
                        for i in range(data_total):
                            coeff = int(coeffs[i])
                            if coeff:
                                parity_payload ^= gf256_mul_vec(np.frombuffer(data_payloads[i], dtype=np.uint8), coeff)
                        parity_pkt = link.video_codec.create_packet_header(
                            parity_payload.tobytes(),
                            frame_info['frame_id'],
                            data_total + p_idx,
                            total_pkts,
                            frame_info['quality']
                        )
                        final_packets.append(parity_pkt)
            elif mac_parity:
                group_count = int(mac_parity_groups) if mac_parity_groups else 1
                if group_count < 1:
                    group_count = 1
                parity_payloads = [np.zeros(payload_size, dtype=np.uint8) for _ in range(group_count)]
                for pkt_bytes, pkt_i in packets:
                    info = link.video_codec.parse_packet_header(pkt_bytes)
                    payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                    if info and frame_info is None:
                        frame_info = info
                    parity_payloads[pkt_i % group_count] ^= np.frombuffer(payload, dtype=np.uint8)
                if frame_info:
                    data_total = frame_info['total_pkts']
                    if data_total < group_count:
                        group_count = max(1, data_total)
                        parity_payloads = parity_payloads[:group_count]
                    total_pkts = data_total + group_count
                    for pkt_bytes, pkt_i in packets:
                        payload = pkt_bytes[link.video_codec.HEADER_SIZE:link.video_codec.HEADER_SIZE + payload_size]
                        rebuilt = link.video_codec.create_packet_header(payload, frame_info['frame_id'], pkt_i, total_pkts, frame_info['quality'])
                        final_packets.append(rebuilt)
                    for gid in range(group_count):
                        parity_pkt = link.video_codec.create_packet_header(
                            parity_payloads[gid].tobytes(),
                            frame_info['frame_id'],
                            data_total + gid,
                            total_pkts,
                            frame_info['quality']
                        )
                        final_packets.append(parity_pkt)
            else:
                final_packets = [p[0] for p in packets]

            def _wire_packet(pkt_bytes):
                if repeat_len > 0:
                    header = pkt_bytes[:link.video_codec.HEADER_SIZE]
                    payload = pkt_bytes[link.video_codec.HEADER_SIZE:]
                    header_repeat = header[:repeat_len]
                    pkt_wire = header + header_repeat + payload
                else:
                    pkt_wire = pkt_bytes
                pkt_arr = np.frombuffer(pkt_wire, dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                fec_bits = link.fec_codec.encode(bits)
                tx_signal = link.transmit(fec_bits)
                info = link.video_codec.parse_packet_header(pkt_bytes)
                if info:
                    key = (info['frame_id'], info['pkt_idx'])
                    with tx_packet_lock:
                        tx_packet_store[key] = {
                            'bits': bits.copy(),
                            'fec': fec_bits.copy(),
                            'ts': time.time()
                        }
                if not hardware_mode:
                    max_val = np.max(np.abs(tx_signal))
                    if max_val > 0:
                        tx_signal = tx_signal / max_val
                    if gap_samples > 0:
                        gap = np.zeros(gap_samples, dtype=tx_signal.dtype)
                        tx_signal = np.concatenate([tx_signal, gap])
                return tx_signal, bits, fec_bits

            if repeat_strategy == "per_frame":
                rounds = max(1, int(pkt_repeats))
                for r in range(rounds):
                    for pkt_bytes in final_packets:
                        tx_signal, bits, fec_bits = _wire_packet(pkt_bytes)
                        with stats_lock:
                            stats['tx_packets'] += 1
                        if not hardware_mode:
                            channel.write(tx_signal)
                        duration = len(tx_signal) / sdr_cfg.fs
                        time.sleep(duration * sleep_factor)
            else:
                for pkt_bytes in final_packets:
                    for _ in range(max(1, int(pkt_repeats))):
                        tx_signal, bits, fec_bits = _wire_packet(pkt_bytes)
                        with stats_lock:
                            stats['tx_packets'] += 1
                        if not hardware_mode:
                            channel.write(tx_signal)
                        duration = len(tx_signal) / sdr_cfg.fs
                        time.sleep(duration * sleep_factor) 
            
            frame_idx += 1
            with stats_lock:
                stats['tx_frames'] += 1
            now = time.time()
            with tx_packet_lock:
                expired = [k for k,v in tx_packet_store.items() if now - v['ts'] > store_timeout]
                for k in expired:
                    del tx_packet_store[k]
            if now - last_log >= log_interval:
                with stats_lock:
                    tx_pkts = stats['tx_packets']
                    tx_frames = stats['tx_frames']
                if diag_level in ("basic", "phy", "mac", "all"):
                    print(f"[TX] Frames: {tx_frames}, Packets: {tx_pkts}")
                last_log = now
            if frame_gap_ms and frame_gap_ms > 0:
                time.sleep(frame_gap_ms / 1000.0)
            if frame_idx >= max_frames: break
            
    except Exception as e:
        print(f"[TX] Error: {e}")
    finally:
        print("[TX] Stopped.")

# =============================================================================
# RX Node
# =============================================================================

def run_rx_node(output_file, config_file="sdr_tuned_config.json", width=320, height=240, fec_cfg=None, sync_threshold=40.0, chunk_samples=0, frame_timeout=5.0, log_interval=2.0, store_timeout=15.0, mac_parity=False, diag_level="basic", mac_parity_groups=1, header_repeat_bytes=0, mac_fec_parity=0, mac_fec_seed=1, save_frames_dir="", save_frames_interval=1, packet_size=None, max_packets_per_frame=None, allowed_resolutions=None, output_fps=10.0, max_inflight_frames=0):
    print(f"[RX] Loading config from {config_file}...")
    sdr_cfg = SDRConfig.load_from_json(config_file)
    fec_cfg = fec_cfg or FECConfig(enabled=True, fec_type=FECType.LDPC)
    
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = sync_threshold
    
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg, simulation_mode=True)
    if packet_size is not None:
        packet_size = int(packet_size)
        if packet_size <= link.video_codec.HEADER_SIZE:
            packet_size = link.video_codec.HEADER_SIZE + 16
        link.video_config.packet_size = packet_size
    if max_packets_per_frame is not None:
        max_packets_per_frame = int(max_packets_per_frame)
        if max_packets_per_frame > 0:
            link.video_config.max_packets_per_frame = max_packets_per_frame
    
    channel.rx_gain_db = float(sdr_cfg.rx_gain)
    channel.fs = sdr_cfg.fs
    print(f"[RX] Simulating RX Gain: {channel.rx_gain_db} dB, Fs: {channel.fs} Hz")
    
    agc = DigitalAGC(target_level=0.5, alpha=0.1) 

    base_packet_bytes = link.video_config.packet_size
    repeat_len = max(0, int(header_repeat_bytes))
    packet_bytes = base_packet_bytes + repeat_len
    expected_packet_bits = packet_bytes * 8
    payload_size = link.video_codec.config.packet_size - link.video_codec.HEADER_SIZE
    max_total_pkts = link.video_config.max_packets_per_frame + (int(mac_parity_groups) if mac_parity else 0) + (int(mac_fec_parity) if mac_fec_parity else 0)
    if fec_cfg.enabled:
        if fec_cfg.fec_type == FECType.LDPC and hasattr(link.fec_codec, 'k'):
            blocks = int(np.ceil(expected_packet_bits / link.fec_codec.k))
            expected_fec_bits = blocks * link.fec_codec.n
        elif fec_cfg.fec_type == FECType.REPETITION:
            expected_fec_bits = expected_packet_bits * fec_cfg.repetitions
        elif fec_cfg.fec_type == FECType.CONVOLUTIONAL or (fec_cfg.fec_type == FECType.LDPC and not hasattr(link.fec_codec, 'k')):
            # Fallback for FECCodec (Rate 1/2, K=7)
            # Or explicit Convolutional
            k_constraint = 7
            expected_fec_bits = 2 * (expected_packet_bits + k_constraint - 1)
        else:
            expected_fec_bits = expected_packet_bits
    else:
        expected_fec_bits = expected_packet_bits

    if link.waveform == WaveformType.OTFS:
        bits_per_frame = link.otfs_config.bits_per_frame
        samples_per_frame = link.otfs_config.N_delay * link.otfs_config.N_doppler
    else:
        bits_per_frame = link.ofdm_config.bits_per_frame
        samples_per_frame = link.ofdm_config.samples_per_frame

    num_frames = int(np.ceil(expected_fec_bits / bits_per_frame))
    expected_payload_samples = num_frames * samples_per_frame
    preamble_len = len(link._generate_preamble())
    min_capture = preamble_len + expected_payload_samples

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), float(output_fps), (width, height))
    buffer_chunk_size = int(chunk_samples) if int(chunk_samples) > 0 else min_capture
    rx_stream = np.array([], dtype=complex)
    frame_buffer = {}
    recovered_count = 0
    crc_fails = 0
    recovered_ids = set()
    mac_recovered_ids = set()
    if allowed_resolutions:
        allowed_set = set(tuple(item) for item in allowed_resolutions)
    else:
        allowed_set = {(width, height)}
    
    last_log = time.time()
    try:
        while True:
            rx_signal = channel.read(buffer_chunk_size)
            if rx_signal is None:
                if stop_event.is_set():
                    if len(rx_stream) < min_capture:
                        break
                time.sleep(0.01)
                continue
            
            rx_signal = agc.process(rx_signal)
            rx_stream = np.concatenate([rx_stream, rx_signal])

            if len(rx_stream) < min_capture:
                continue
            
            # Debug AGC & SNR
            if diag_level in ("phy", "all") and np.random.rand() < 0.03:
               sig_pwr = np.mean(np.abs(rx_signal)**2)
               print(f"[RX Debug] AGC Gain: {agc.gain:.1f}, Rx Pwr: {sig_pwr:.6f}")
            
            while True:
                if len(rx_stream) < min_capture:
                    break
                window_len = min(len(rx_stream), min_capture * 2)
                rx_stream_norm = rx_stream[:window_len]
                with stats_lock:
                    stats['sync_attempts'] += 1
                synced_sig, sync_metrics = link._synchronize(rx_stream_norm)

                if not sync_metrics.get('sync_success'):
                    if sync_metrics.get('incomplete'):
                        with stats_lock:
                            stats['sync_incomplete'] += 1
                    if len(rx_stream) > buffer_chunk_size * 8:
                        rx_stream = rx_stream[-buffer_chunk_size * 4:]
                    break
                with stats_lock:
                    stats['sync_successes'] += 1

                peak_val = sync_metrics.get('peak_val', 0)
                cfo_val = sync_metrics.get('cfo_est', 0)
                if diag_level in ("phy", "all") and np.random.rand() < 0.02:
                    print(f"[RX Diag] Sync! Peak: {peak_val:.1f}, CFO: {cfo_val:.1f} Hz")

                payload_start = sync_metrics.get('payload_start', None)
                if payload_start is None:
                    break
                total_needed = payload_start + expected_payload_samples
                if len(rx_stream) < total_needed:
                    break

                payload = synced_sig[:expected_payload_samples]
                rx_stream = rx_stream[total_needed:]

                rx_fec_bits, met = link.transceiver.demodulate(payload)
                try:
                    rx_bits_dec = link.fec_codec.decode(rx_fec_bits)
                except:
                    continue

                if len(rx_bits_dec) < expected_packet_bits:
                    continue
                rx_bits_dec = rx_bits_dec[:expected_packet_bits]
                rx_bytes = np.packbits(rx_bits_dec).tobytes()
                if len(rx_bytes) < packet_bytes:
                    continue
                rx_bytes = rx_bytes[:packet_bytes]
                header_valid_found = False
                info = None
                rx_packet_clean = None
                selected_shift = 0
                scan_max = repeat_len if repeat_len > 0 else 0
                for shift in range(0, scan_max + 1):
                    header_start = shift
                    header_end = header_start + link.video_codec.HEADER_SIZE
                    repeat_start = header_end
                    repeat_end = repeat_start + repeat_len
                    payload_start = repeat_end
                    payload_end = payload_start + payload_size
                    if payload_end > len(rx_bytes):
                        break
                    header_bytes = rx_bytes[header_start:header_end]
                    if repeat_len > 0:
                        repeat_bytes = rx_bytes[repeat_start:repeat_end]
                        if repeat_bytes != header_bytes[:repeat_len]:
                            continue
                    payload_bytes = rx_bytes[payload_start:payload_end]
                    candidate = header_bytes + payload_bytes
                    candidate_info = link.video_codec.parse_packet_header(candidate)
                    if candidate_info:
                        if (
                            (candidate_info['width'], candidate_info['height']) not in allowed_set
                            or candidate_info['quality'] <= 0
                            or candidate_info['quality'] > 100
                            or candidate_info['total_pkts'] <= 0
                            or candidate_info['total_pkts'] > max_total_pkts
                            or candidate_info['pkt_idx'] < 0
                            or candidate_info['pkt_idx'] >= candidate_info['total_pkts']
                        ):
                            continue
                        header_valid_found = True
                        if zlib.crc32(candidate_info['payload']) & 0xFFFFFFFF == candidate_info['crc']:
                            info = candidate_info
                            rx_packet_clean = candidate
                            selected_shift = shift
                            break
                if info is None:
                    if header_valid_found:
                        crc_fails += 1
                        with stats_lock:
                            stats['crc_fails'] += 1
                    else:
                        with stats_lock:
                            stats['header_invalid'] += 1
                    continue
                if selected_shift > 0:
                    with stats_lock:
                        stats['header_resync'] += 1
                key = (info['frame_id'], info['pkt_idx'])
                with tx_packet_lock:
                    tx_entry = tx_packet_store.get(key)
                if tx_entry is not None:
                    tx_bits = tx_entry['bits']
                    tx_fec_bits = tx_entry['fec']
                    min_len_pre = min(len(rx_fec_bits), len(tx_fec_bits))
                    if min_len_pre > 0:
                        pre_err = np.sum(rx_fec_bits[:min_len_pre] != tx_fec_bits[:min_len_pre])
                        with stats_lock:
                            stats['pre_fec_bit_errors'] += int(pre_err)
                            stats['pre_fec_bit_total'] += int(min_len_pre)
                    min_len_post = min(len(rx_bits_dec), len(tx_bits))
                    if min_len_post > 0:
                        post_err = np.sum(rx_bits_dec[:min_len_post] != tx_bits[:min_len_post])
                        with stats_lock:
                            stats['post_fec_bit_errors'] += int(post_err)
                            stats['post_fec_bit_total'] += int(min_len_post)
                else:
                    with stats_lock:
                        stats['tx_lookup_miss'] += 1

                with stats_lock:
                    stats['rx_packets'] += 1

                fid = info['frame_id']
                if fid not in frame_buffer:
                    frame_buffer[fid] = {
                        'pkts': {},
                        'payloads': {},
                        'total': info['total_pkts'],
                        'quality': info['quality'],
                        'ts': time.time(),
                        'counted': False
                    }
                    with stats_lock:
                        stats['frames_started'] += 1
                else:
                    frame_buffer[fid]['ts'] = time.time()
                    frame_buffer[fid]['total'] = info['total_pkts']
                    frame_buffer[fid]['quality'] = info['quality']

                frame_buffer[fid]['pkts'][info['pkt_idx']] = rx_packet_clean
                frame_buffer[fid]['payloads'][info['pkt_idx']] = info['payload']
                if max_inflight_frames and len(frame_buffer) > int(max_inflight_frames):
                    drop_fid, drop_info = min(frame_buffer.items(), key=lambda kv: kv[1]['ts'])
                    if drop_fid in frame_buffer:
                        del frame_buffer[drop_fid]
                        with stats_lock:
                            stats['frames_dropped'] += 1

                def build_coeffs(total_data, parity_idx, frame_id):
                    base = (int(mac_fec_seed) + int(frame_id)) % 255
                    if base == 0:
                        base = 1
                    alpha = (base + parity_idx) % 255
                    if alpha == 0:
                        alpha = 1
                    coeffs = np.fromiter((gf256_pow(alpha, i) for i in range(total_data)), dtype=np.uint8, count=total_data)
                    return coeffs

                def solve_missing(missing_indices, parity_indices, data_total, frame_id, payloads, payload_size):
                    m = len(missing_indices)
                    if m == 0 or len(parity_indices) < m:
                        return {}
                    A = np.zeros((m, m), dtype=np.uint8)
                    B = [None] * m
                    for row in range(m):
                        parity_idx = parity_indices[row]
                        coeffs = build_coeffs(data_total, parity_idx - data_total, frame_id)
                        eq = np.frombuffer(payloads[parity_idx], dtype=np.uint8).copy()
                        for i in range(data_total):
                            coeff = int(coeffs[i])
                            if coeff and i in payloads:
                                eq ^= gf256_mul_vec(np.frombuffer(payloads[i], dtype=np.uint8), coeff)
                        for col, miss_idx in enumerate(missing_indices):
                            A[row, col] = coeffs[miss_idx]
                        B[row] = eq
                    for col in range(m):
                        pivot = None
                        for row in range(col, m):
                            if A[row, col]:
                                pivot = row
                                break
                        if pivot is None:
                            return None
                        if pivot != col:
                            A[[col, pivot]] = A[[pivot, col]]
                            B[col], B[pivot] = B[pivot], B[col]
                        inv_pivot = gf256_inv(int(A[col, col]))
                        if inv_pivot == 0:
                            return None
                        A[col] = gf256_mul_vec(A[col], inv_pivot)
                        B[col] = gf256_mul_vec(B[col], inv_pivot)
                        for row in range(m):
                            if row != col and A[row, col]:
                                factor = int(A[row, col])
                                A[row] ^= gf256_mul_vec(A[col], factor)
                                B[row] ^= gf256_mul_vec(B[col], factor)
                    recovered = {}
                    for i, miss_idx in enumerate(missing_indices):
                        recovered[miss_idx] = B[i].tobytes()
                    return recovered

                if mac_fec_parity and frame_buffer[fid]['total'] > 1:
                    total = frame_buffer[fid]['total']
                    parity_count = int(mac_fec_parity)
                    data_total = total - parity_count
                    if data_total > 0:
                        pkts = frame_buffer[fid]['pkts']
                        payloads = frame_buffer[fid]['payloads']
                        missing = [i for i in range(data_total) if i not in pkts]
                        parity_indices = [i for i in range(data_total, total) if i in payloads]
                        if missing and len(parity_indices) >= len(missing):
                            recovered = solve_missing(missing, parity_indices[:len(missing)], data_total, fid, payloads, payload_size)
                            if recovered is None:
                                with stats_lock:
                                    stats['mac_fec_decode_fail'] += 1
                            else:
                                for miss_idx, payload_bytes in recovered.items():
                                    rebuilt = link.video_codec.create_packet_header(
                                        payload_bytes,
                                        fid,
                                        miss_idx,
                                        total,
                                        frame_buffer[fid]['quality']
                                    )
                                    pkts[miss_idx] = rebuilt
                                    payloads[miss_idx] = payload_bytes
                                with stats_lock:
                                    stats['mac_fec_recovered_packets'] += len(recovered)
                                if fid not in mac_recovered_ids:
                                    with stats_lock:
                                        stats['mac_fec_recovered_frames'] += 1
                                    mac_recovered_ids.add(fid)
                    data_ready = all(i in frame_buffer[fid]['pkts'] for i in range(data_total))
                    if data_ready:
                        pkt_list = [frame_buffer[fid]['pkts'][i] for i in range(data_total)]
                        frame = link.video_codec.decode_packets(pkt_list)
                        if frame is not None and fid not in recovered_ids:
                            if not frame_buffer[fid].get('counted'):
                                total_pkts = frame_buffer[fid]['total']
                                rx_pkts = len(frame_buffer[fid]['pkts'])
                                missing_pkts = max(0, total_pkts - rx_pkts)
                                with stats_lock:
                                    stats['frames_total_pkts_sum'] += total_pkts
                                    stats['frames_total_pkts_count'] += 1
                                    stats['frames_rx_pkts_sum'] += rx_pkts
                                    stats['frames_missing_pkts_sum'] += missing_pkts
                                    stats['frames_total_pkts_min'] = total_pkts if stats['frames_total_pkts_min'] is None else min(stats['frames_total_pkts_min'], total_pkts)
                                    stats['frames_total_pkts_max'] = max(stats['frames_total_pkts_max'], total_pkts)
                                frame_buffer[fid]['counted'] = True
                            write_frame = frame
                            if (frame.shape[1], frame.shape[0]) != (width, height):
                                write_frame = cv2.resize(frame, (width, height))
                            out.write(write_frame)
                            recovered_count += 1
                            if save_frames_dir and save_frames_interval > 0:
                                if fid % int(save_frames_interval) == 0:
                                    save_name = f"frame_{fid:06d}.jpg"
                                    cv2.imwrite(os.path.join(save_frames_dir, save_name), write_frame)
                            recovered_ids.add(fid)
                            with stats_lock:
                                stats['frames_recovered'] += 1
                            if diag_level in ("basic", "phy", "mac", "all"):
                                print(f"[RX] Recovered Frame {fid}. AGC Gain: {agc.gain:.1f}")
                            del frame_buffer[fid]
                elif mac_parity and frame_buffer[fid]['total'] > 1:
                    total = frame_buffer[fid]['total']
                    pkts = frame_buffer[fid]['pkts']
                    payloads = frame_buffer[fid]['payloads']
                    group_count = int(mac_parity_groups) if mac_parity_groups else 1
                    if group_count < 1:
                        group_count = 1
                    if total <= group_count:
                        group_count = 1
                    data_total = total - group_count
                    if data_total > 0:
                        recovered_pkts = 0
                        for gid in range(group_count):
                            parity_idx = data_total + gid
                            if parity_idx not in pkts:
                                continue
                            missing = [i for i in range(gid, data_total, group_count) if i not in pkts]
                            if len(missing) == 1:
                                missing_idx = missing[0]
                                recovered_payload = np.frombuffer(payloads[parity_idx], dtype=np.uint8).copy()
                                for i in range(gid, data_total, group_count):
                                    if i == missing_idx:
                                        continue
                                    if i in payloads:
                                        recovered_payload ^= np.frombuffer(payloads[i], dtype=np.uint8)
                                rebuilt = link.video_codec.create_packet_header(
                                    recovered_payload.tobytes(),
                                    fid,
                                    missing_idx,
                                    total,
                                    frame_buffer[fid]['quality']
                                )
                                pkts[missing_idx] = rebuilt
                                payloads[missing_idx] = recovered_payload.tobytes()
                                recovered_pkts += 1
                        if recovered_pkts > 0:
                            with stats_lock:
                                stats['mac_recovered_packets'] += recovered_pkts
                            if fid not in mac_recovered_ids:
                                with stats_lock:
                                    stats['mac_recovered_frames'] += 1
                                mac_recovered_ids.add(fid)
                    data_ready = all(i in pkts for i in range(data_total))
                    if data_ready:
                        pkt_list = [pkts[i] for i in range(data_total)]
                        frame = link.video_codec.decode_packets(pkt_list)
                        if frame is not None and fid not in recovered_ids:
                            if not frame_buffer[fid].get('counted'):
                                total_pkts = frame_buffer[fid]['total']
                                rx_pkts = len(pkts)
                                missing_pkts = max(0, total_pkts - rx_pkts)
                                with stats_lock:
                                    stats['frames_total_pkts_sum'] += total_pkts
                                    stats['frames_total_pkts_count'] += 1
                                    stats['frames_rx_pkts_sum'] += rx_pkts
                                    stats['frames_missing_pkts_sum'] += missing_pkts
                                    stats['frames_total_pkts_min'] = total_pkts if stats['frames_total_pkts_min'] is None else min(stats['frames_total_pkts_min'], total_pkts)
                                    stats['frames_total_pkts_max'] = max(stats['frames_total_pkts_max'], total_pkts)
                                frame_buffer[fid]['counted'] = True
                            write_frame = frame
                            if (frame.shape[1], frame.shape[0]) != (width, height):
                                write_frame = cv2.resize(frame, (width, height))
                            out.write(write_frame)
                            recovered_count += 1
                            if save_frames_dir and save_frames_interval > 0:
                                if fid % int(save_frames_interval) == 0:
                                    save_name = f"frame_{fid:06d}.jpg"
                                    cv2.imwrite(os.path.join(save_frames_dir, save_name), write_frame)
                            recovered_ids.add(fid)
                            with stats_lock:
                                stats['frames_recovered'] += 1
                            if diag_level in ("basic", "phy", "mac", "all"):
                                print(f"[RX] Recovered Frame {fid}. AGC Gain: {agc.gain:.1f}")
                            del frame_buffer[fid]
                    else:
                        if len(frame_buffer[fid]['pkts']) == frame_buffer[fid]['total']:
                            pkt_list = [frame_buffer[fid]['pkts'][i] for i in range(frame_buffer[fid]['total'])]
                            frame = link.video_codec.decode_packets(pkt_list)
                            if frame is not None and fid not in recovered_ids:
                                if not frame_buffer[fid].get('counted'):
                                    total_pkts = frame_buffer[fid]['total']
                                    rx_pkts = len(frame_buffer[fid]['pkts'])
                                    missing_pkts = max(0, total_pkts - rx_pkts)
                                    with stats_lock:
                                        stats['frames_total_pkts_sum'] += total_pkts
                                        stats['frames_total_pkts_count'] += 1
                                        stats['frames_rx_pkts_sum'] += rx_pkts
                                        stats['frames_missing_pkts_sum'] += missing_pkts
                                        stats['frames_total_pkts_min'] = total_pkts if stats['frames_total_pkts_min'] is None else min(stats['frames_total_pkts_min'], total_pkts)
                                        stats['frames_total_pkts_max'] = max(stats['frames_total_pkts_max'], total_pkts)
                                    frame_buffer[fid]['counted'] = True
                                write_frame = frame
                                if (frame.shape[1], frame.shape[0]) != (width, height):
                                    write_frame = cv2.resize(frame, (width, height))
                                out.write(write_frame)
                                recovered_count += 1
                                if save_frames_dir and save_frames_interval > 0:
                                    if fid % int(save_frames_interval) == 0:
                                        save_name = f"frame_{fid:06d}.jpg"
                                        cv2.imwrite(os.path.join(save_frames_dir, save_name), write_frame)
                                recovered_ids.add(fid)
                                with stats_lock:
                                    stats['frames_recovered'] += 1
                                if diag_level in ("basic", "phy", "mac", "all"):
                                    print(f"[RX] Recovered Frame {fid}. AGC Gain: {agc.gain:.1f}")
                            del frame_buffer[fid]
                else:
                    if len(frame_buffer[fid]['pkts']) == frame_buffer[fid]['total']:
                        pkt_list = [frame_buffer[fid]['pkts'][i] for i in range(frame_buffer[fid]['total'])]
                        frame = link.video_codec.decode_packets(pkt_list)
                        if frame is not None and fid not in recovered_ids:
                            if not frame_buffer[fid].get('counted'):
                                total_pkts = frame_buffer[fid]['total']
                                rx_pkts = len(frame_buffer[fid]['pkts'])
                                missing_pkts = max(0, total_pkts - rx_pkts)
                                with stats_lock:
                                    stats['frames_total_pkts_sum'] += total_pkts
                                    stats['frames_total_pkts_count'] += 1
                                    stats['frames_rx_pkts_sum'] += rx_pkts
                                    stats['frames_missing_pkts_sum'] += missing_pkts
                                    stats['frames_total_pkts_min'] = total_pkts if stats['frames_total_pkts_min'] is None else min(stats['frames_total_pkts_min'], total_pkts)
                                    stats['frames_total_pkts_max'] = max(stats['frames_total_pkts_max'], total_pkts)
                                frame_buffer[fid]['counted'] = True
                            write_frame = frame
                            if (frame.shape[1], frame.shape[0]) != (width, height):
                                write_frame = cv2.resize(frame, (width, height))
                            out.write(write_frame)
                            recovered_count += 1
                            if save_frames_dir and save_frames_interval > 0:
                                if fid % int(save_frames_interval) == 0:
                                    save_name = f"frame_{fid:06d}.jpg"
                                    cv2.imwrite(os.path.join(save_frames_dir, save_name), write_frame)
                            recovered_ids.add(fid)
                            with stats_lock:
                                stats['frames_recovered'] += 1
                            if diag_level in ("basic", "phy", "mac", "all"):
                                print(f"[RX] Recovered Frame {fid}. AGC Gain: {agc.gain:.1f}")
                        del frame_buffer[fid]

            now = time.time()
            expired = [k for k,v in frame_buffer.items() if now - v['ts'] > frame_timeout]
            for k in expired:
                total_pkts = frame_buffer[k]['total']
                rx_pkts = len(frame_buffer[k]['pkts'])
                missing_pkts = max(0, total_pkts - rx_pkts)
                with stats_lock:
                    stats['frames_expired'] += 1
                    stats['frames_expired_total_pkts_sum'] += total_pkts
                    stats['frames_expired_missing_pkts_sum'] += missing_pkts
                    stats['frames_total_pkts_sum'] += total_pkts
                    stats['frames_total_pkts_count'] += 1
                    stats['frames_rx_pkts_sum'] += rx_pkts
                    stats['frames_missing_pkts_sum'] += missing_pkts
                    stats['frames_total_pkts_min'] = total_pkts if stats['frames_total_pkts_min'] is None else min(stats['frames_total_pkts_min'], total_pkts)
                    stats['frames_total_pkts_max'] = max(stats['frames_total_pkts_max'], total_pkts)
                del frame_buffer[k]
            if now - last_log >= log_interval:
                with stats_lock:
                    tx_pkts = stats['tx_packets']
                    rx_pkts = stats['rx_packets']
                    crc_total = stats['crc_fails']
                    frames_total = stats['frames_recovered']
                    tx_frames = stats['tx_frames']
                    frames_started = stats['frames_started']
                    frames_expired = stats['frames_expired']
                    frames_dropped = stats['frames_dropped']
                    mac_rec_frames = stats['mac_recovered_frames']
                    mac_rec_pkts = stats['mac_recovered_packets']
                    header_invalid = stats['header_invalid']
                    header_resync = stats['header_resync']
                    tx_lookup_miss = stats['tx_lookup_miss']
                    mac_fec_frames = stats['mac_fec_recovered_frames']
                    mac_fec_pkts = stats['mac_fec_recovered_packets']
                    mac_fec_fail = stats['mac_fec_decode_fail']
                    total_pkts_sum = stats['frames_total_pkts_sum']
                    total_pkts_count = stats['frames_total_pkts_count']
                    rx_pkts_sum = stats['frames_rx_pkts_sum']
                    missing_pkts_sum = stats['frames_missing_pkts_sum']
                    expired_total_pkts_sum = stats['frames_expired_total_pkts_sum']
                    expired_missing_pkts_sum = stats['frames_expired_missing_pkts_sum']
                    total_pkts_min = stats['frames_total_pkts_min']
                    total_pkts_max = stats['frames_total_pkts_max']
                    pre_err = stats['pre_fec_bit_errors']
                    pre_total = stats['pre_fec_bit_total']
                    post_err = stats['post_fec_bit_errors']
                    post_total = stats['post_fec_bit_total']
                    sync_attempts = stats['sync_attempts']
                    sync_successes = stats['sync_successes']
                    sync_incomplete = stats['sync_incomplete']
                pkt_success = (rx_pkts / tx_pkts) if tx_pkts > 0 else 0.0
                pre_ber = (pre_err / pre_total) if pre_total > 0 else 0.0
                post_ber = (post_err / post_total) if post_total > 0 else 0.0
                fer = 1.0 - (frames_total / tx_frames) if tx_frames > 0 else 0.0
                sync_rate = (sync_successes / sync_attempts) if sync_attempts > 0 else 0.0
                avg_total_pkts = (total_pkts_sum / total_pkts_count) if total_pkts_count > 0 else 0.0
                avg_rx_pkts = (rx_pkts_sum / total_pkts_count) if total_pkts_count > 0 else 0.0
                avg_missing_pkts = (missing_pkts_sum / total_pkts_count) if total_pkts_count > 0 else 0.0
                exp_missing_avg = (expired_missing_pkts_sum / stats['frames_expired']) if stats['frames_expired'] > 0 else 0.0
                exp_total_avg = (expired_total_pkts_sum / stats['frames_expired']) if stats['frames_expired'] > 0 else 0.0
                pkt_success_est = pkt_success
                frame_success_est = (pkt_success_est ** avg_total_pkts) if avg_total_pkts > 0 else 0.0
                if diag_level in ("basic", "phy", "mac", "all"):
                    print(f"[RX] Frames: {frames_total}/{tx_frames}, Dropped: {frames_dropped}, Packets: {rx_pkts}/{tx_pkts}, CRC: {crc_total}, FER: {fer:.3f}, Success: {pkt_success:.3f}")
                if diag_level in ("phy", "all"):
                    print(f"[RX] PreBER: {pre_ber:.4e}, PostBER: {post_ber:.4e}")
                    print(f"[RX] Sync: {sync_successes}/{sync_attempts}, Incomplete: {sync_incomplete}, SyncRate: {sync_rate:.3f}")
                if diag_level in ("mac", "all"):
                    print(f"[RX] FramesStarted: {frames_started}, FramesExpired: {frames_expired}, MACRecoveredFrames: {mac_rec_frames}, MACRecoveredPackets: {mac_rec_pkts}")
                    print(f"[RX] HeaderInvalid: {header_invalid}, HeaderResync: {header_resync}, TxLookupMiss: {tx_lookup_miss}")
                    print(f"[RX] MACFEC Frames: {mac_fec_frames}, Packets: {mac_fec_pkts}, Fail: {mac_fec_fail}")
                    print(f"[RX] FramePkts AvgTotal: {avg_total_pkts:.2f}, AvgRx: {avg_rx_pkts:.2f}, AvgMissing: {avg_missing_pkts:.2f}, Min: {total_pkts_min}, Max: {total_pkts_max}")
                    print(f"[RX] Expired AvgTotal: {exp_total_avg:.2f}, AvgMissing: {exp_missing_avg:.2f}, EstFrameSuccess: {frame_success_est:.3e}")
                last_log = now
                    
    except Exception as e:
        print(f"[RX] Error: {e}")
    finally:
        out.release()
        with stats_lock:
            tx_packets = stats['tx_packets']
            rx_packets = stats['rx_packets']
            crc_total = stats['crc_fails']
            frames_total = stats['frames_recovered']
            tx_frames = stats['tx_frames']
            frames_started = stats['frames_started']
            frames_expired = stats['frames_expired']
            frames_dropped = stats['frames_dropped']
            mac_rec_frames = stats['mac_recovered_frames']
            mac_rec_pkts = stats['mac_recovered_packets']
            header_invalid = stats['header_invalid']
            header_resync = stats['header_resync']
            tx_lookup_miss = stats['tx_lookup_miss']
            mac_fec_frames = stats['mac_fec_recovered_frames']
            mac_fec_pkts = stats['mac_fec_recovered_packets']
            mac_fec_fail = stats['mac_fec_decode_fail']
            total_pkts_sum = stats['frames_total_pkts_sum']
            total_pkts_count = stats['frames_total_pkts_count']
            rx_pkts_sum = stats['frames_rx_pkts_sum']
            missing_pkts_sum = stats['frames_missing_pkts_sum']
            expired_total_pkts_sum = stats['frames_expired_total_pkts_sum']
            expired_missing_pkts_sum = stats['frames_expired_missing_pkts_sum']
            total_pkts_min = stats['frames_total_pkts_min']
            total_pkts_max = stats['frames_total_pkts_max']
            pre_err = stats['pre_fec_bit_errors']
            pre_total = stats['pre_fec_bit_total']
            post_err = stats['post_fec_bit_errors']
            post_total = stats['post_fec_bit_total']
            sync_attempts = stats['sync_attempts']
            sync_successes = stats['sync_successes']
            sync_incomplete = stats['sync_incomplete']
        pkt_success = (rx_packets / tx_packets) if tx_packets > 0 else 0.0
        pre_ber = (pre_err / pre_total) if pre_total > 0 else 0.0
        post_ber = (post_err / post_total) if post_total > 0 else 0.0
        fer = 1.0 - (frames_total / tx_frames) if tx_frames > 0 else 0.0
        sync_rate = (sync_successes / sync_attempts) if sync_attempts > 0 else 0.0
        avg_total_pkts = (total_pkts_sum / total_pkts_count) if total_pkts_count > 0 else 0.0
        avg_rx_pkts = (rx_pkts_sum / total_pkts_count) if total_pkts_count > 0 else 0.0
        avg_missing_pkts = (missing_pkts_sum / total_pkts_count) if total_pkts_count > 0 else 0.0
        exp_missing_avg = (expired_missing_pkts_sum / frames_expired) if frames_expired > 0 else 0.0
        exp_total_avg = (expired_total_pkts_sum / frames_expired) if frames_expired > 0 else 0.0
        frame_success_est = (pkt_success ** avg_total_pkts) if avg_total_pkts > 0 else 0.0
        if diag_level in ("basic", "phy", "mac", "all"):
            print(f"[RX] Stopped. Recovered: {recovered_count}, CRC Fails: {crc_fails}")
            print(f"[RX] Packets: TX={tx_packets}, RX_OK={rx_packets}, CRC_FAIL={crc_total}, Success={pkt_success:.3f}")
            print(f"[RX] Frames: TX={tx_frames}, RX_OK={frames_total}, Started={frames_started}, Expired={frames_expired}, Dropped={frames_dropped}, FER={fer:.3f}")
        if diag_level in ("phy", "all"):
            print(f"[RX] PreBER: {pre_ber:.4e}, PostBER: {post_ber:.4e}")
            print(f"[RX] Sync: {sync_successes}/{sync_attempts}, Incomplete: {sync_incomplete}, SyncRate: {sync_rate:.3f}")
        if diag_level in ("mac", "all"):
            print(f"[RX] MACRecoveredFrames: {mac_rec_frames}, MACRecoveredPackets: {mac_rec_pkts}")
            print(f"[RX] HeaderInvalid: {header_invalid}, HeaderResync: {header_resync}, TxLookupMiss: {tx_lookup_miss}")
            print(f"[RX] MACFEC Frames: {mac_fec_frames}, Packets: {mac_fec_pkts}, Fail: {mac_fec_fail}")
            print(f"[RX] FramePkts AvgTotal: {avg_total_pkts:.2f}, AvgRx: {avg_rx_pkts:.2f}, AvgMissing: {avg_missing_pkts:.2f}, Min: {total_pkts_min}, Max: {total_pkts_max}")
            print(f"[RX] Expired AvgTotal: {exp_total_avg:.2f}, AvgMissing: {exp_missing_avg:.2f}, EstFrameSuccess: {frame_success_est:.3e}")

# =============================================================================
# Main
# =============================================================================

def normalize_video_path(path: str) -> str:
    if len(path) >= 2 and path[1] == ':' and '\\' in path:
        drive = path[0].lower()
        rest = path[2:].lstrip('\\/').replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    return path


def main():
    print("=== Enhanced Async E2E Simulation ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="Input video file")
    parser.add_argument("--output", type=str, default="sim_enhanced_out.avi", help="Output video file")
    parser.add_argument("--config", type=str, default="sdr_tuned_config.json", help="SDR config file")
    parser.add_argument("--duration", type=int, default=15, help="Duration per scenario (seconds)")
    parser.add_argument("--width", type=int, default=320, help="Frame width")
    parser.add_argument("--height", type=int, default=240, help="Frame height")
    parser.add_argument("--quality", type=int, default=32, help="JPEG quality")
    parser.add_argument("--fec", type=str, default="ldpc", help="FEC type: ldpc, repetition, convolutional, none")
    parser.add_argument("--repetitions", type=int, default=5, help="Repetition FEC factor")
    parser.add_argument("--code-rate", type=str, default="1/2", help="FEC code rate")
    parser.add_argument("--sync-threshold", type=float, default=30.0, help="Sync correlation threshold")
    parser.add_argument("--chunk-samples", type=int, default=0, help="RX chunk size; 0 uses min packet size")
    parser.add_argument("--gap-samples", type=int, default=0, help="Zero samples inserted between packets")
    parser.add_argument("--sleep-factor", type=float, default=1.0, help="TX pacing factor")
    parser.add_argument("--max-frames", type=int, default=120, help="Max frames to send")
    parser.add_argument("--pkt-repeats", type=int, default=4, help="TX repeat count per packet")
    parser.add_argument("--frame-timeout", type=float, default=10.0, help="RX frame buffer timeout (seconds)")
    parser.add_argument("--mode", type=str, default="file", choices=["file", "stream"], help="Transmission mode")
    parser.add_argument("--log-interval", type=float, default=2.0, help="Stats log interval (seconds)")
    parser.add_argument("--store-timeout", type=float, default=20.0, help="TX packet store timeout (seconds)")
    parser.add_argument("--tx-power", type=float, default=None, help="Simulated TX power dBFS")
    parser.add_argument("--path-loss", type=float, default=None, help="Simulated path loss dB")
    parser.add_argument("--rx-gain", type=float, default=None, help="Simulated RX gain dB")
    parser.add_argument("--noise-dbfs", type=float, default=None, help="Simulated noise level dBFS")
    parser.add_argument("--cfo-hz", type=float, default=None, help="Simulated CFO Hz")
    parser.add_argument("--strong-tx-power", type=float, default=None, help="Strong-signal TX power dBFS")
    parser.add_argument("--strong-path-loss", type=float, default=None, help="Strong-signal path loss dB")
    parser.add_argument("--strong-rx-gain", type=float, default=None, help="Strong-signal RX gain dB")
    parser.add_argument("--strong-noise-dbfs", type=float, default=None, help="Strong-signal noise level dBFS")
    parser.add_argument("--strong-cfo-hz", type=float, default=None, help="Strong-signal CFO Hz")
    parser.add_argument("--run-strong-scenario", action="store_true", help="Enable strong-signal scenario")
    parser.add_argument("--mac-parity", action="store_true", help="Enable MAC parity recovery")
    parser.add_argument("--mac-parity-groups", type=int, default=1, help="MAC parity group count")
    parser.add_argument("--mac-fec-parity", type=int, default=None, help="Cross-packet parity count per frame")
    parser.add_argument("--mac-fec-seed", type=int, default=1, help="Cross-packet parity seed")
    parser.add_argument("--diag-level", type=str, default="basic", choices=["basic", "phy", "mac", "all"], help="Diagnostic output level")
    parser.add_argument("--header-repeat-bytes", type=int, default=12, help="Repeat header bytes in packet body for alignment slack")
    parser.add_argument("--repeat-strategy", type=str, default="per_packet", choices=["per_packet","per_frame"], help="Packet repeat scheduling strategy")
    parser.add_argument("--channel-buffer-len", type=int, default=None, help="Channel buffer length in samples")
    parser.add_argument("--rx-drain-timeout", type=float, default=6.0, help="RX drain time after TX completes (seconds)")
    parser.add_argument("--frame-gap-ms", type=int, default=0, help="Gap between frames at TX (ms)")
    parser.add_argument("--save-frames-dir", type=str, default="", help="Directory to save recovered frames")
    parser.add_argument("--save-frames-interval", type=int, default=1, help="Save every N frame_id")
    parser.add_argument("--tx-frame-step", type=int, default=1, help="Send every Nth source frame")
    parser.add_argument("--packet-size", type=int, default=768, help="Video packet size in bytes (including header)")
    parser.add_argument("--max-packets-per-frame", type=int, default=60, help="Max packets per frame for video codec")
    parser.add_argument("--adapt", action="store_true", help="Enable adaptive quality/resolution control at TX")
    parser.add_argument("--no-adapt", action="store_true", help="Disable adaptive quality/resolution control at TX")
    parser.add_argument("--adapt-interval", type=float, default=2.0, help="Adaptation interval seconds")
    parser.add_argument("--target-fer", type=float, default=0.05, help="Target frame error rate")
    parser.add_argument("--target-sync-rate", type=float, default=0.9, help="Target sync success rate")
    parser.add_argument("--target-post-ber", type=float, default=1e-4, help="Target post-FEC BER")
    parser.add_argument("--quality-min", type=int, default=15, help="Min JPEG quality for adaptation")
    parser.add_argument("--quality-max", type=int, default=65, help="Max JPEG quality for adaptation")
    parser.add_argument("--quality-step", type=int, default=4, help="JPEG quality step for adaptation")
    parser.add_argument("--allowed-resolutions", type=str, default="320x240,640x480", help="Comma list of WxH allowed resolutions for RX (e.g., 320x240,640x480)")
    parser.add_argument("--output-fps", type=float, default=10.0, help="Output video FPS at RX")
    parser.add_argument("--max-inflight-frames", type=int, default=0, help="Max in-flight frames in RX buffer (drop oldest if exceeded)")
    args = parser.parse_args()

    config_file = args.config
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump({
                "sdr_ip": "ip:192.168.2.1",
                "rx_uri": "ip:192.168.2.2",
                "device": "pluto",
                "fc": 915e6,
                "fs": 2e6,
                "bandwidth": 10e6,
                "tx_gain": 0,
                "rx_gain": 40,
                "rx_buffer_size": 262144
            }, f)

    fec_map = {
        "none": FECType.NONE,
        "repetition": FECType.REPETITION,
        "convolutional": FECType.CONVOLUTIONAL,
        "ldpc": FECType.LDPC
    }
    fec_type = fec_map.get(args.fec.lower(), FECType.LDPC)
    fec_cfg = FECConfig(enabled=fec_type != FECType.NONE, fec_type=fec_type, repetitions=args.repetitions, code_rate=args.code_rate)

    # 1. Weak Signal
    print("\n[Scenario] Weak Signal (Input -30dBFS)...")
    channel.path_loss_db = 40.0
    channel.tx_power_db = 0.0
    channel.rx_gain_db = 40.0
    channel.base_noise_dbfs = -120.0
    channel.cfo_hz = 100.0
    if args.path_loss is not None:
        channel.path_loss_db = float(args.path_loss)
    if args.tx_power is not None:
        channel.tx_power_db = float(args.tx_power)
    if args.rx_gain is not None:
        channel.rx_gain_db = float(args.rx_gain)
    if args.noise_dbfs is not None:
        channel.base_noise_dbfs = float(args.noise_dbfs)
    if args.cfo_hz is not None:
        channel.cfo_hz = float(args.cfo_hz)
    
    buffer_len = args.channel_buffer_len
    if buffer_len is None:
        buffer_len = 60000000 if args.mode == "file" else 30000000
    channel.max_buffer_len = int(buffer_len)
    print("Starting Threads...")
    video_path = normalize_video_path(args.video) if args.video else "test_video.mp4"
    allow_synthetic = not bool(args.video)
    pkt_repeats = args.pkt_repeats
    frame_timeout = args.frame_timeout
    store_timeout = args.store_timeout
    sleep_factor = args.sleep_factor
    mac_fec_parity = args.mac_fec_parity
    adapt = args.adapt
    if not args.adapt and not args.no_adapt:
        adapt = True
    if args.mode == "stream":
        pkt_repeats = 1
        if frame_timeout < 12.0:
            frame_timeout = 12.0
        if store_timeout < (frame_timeout * 2.0):
            store_timeout = frame_timeout * 2.0
        if mac_fec_parity is None:
            mac_fec_parity = 4
    elif args.mode == "file" and pkt_repeats < 3:
        pkt_repeats = 3
    if args.mode == "file" and sleep_factor < 1.0:
        sleep_factor = 1.0
    if mac_fec_parity is None:
        mac_fec_parity = 2 if args.mode == "file" else 4
    if store_timeout < (frame_timeout * 2.0):
        store_timeout = frame_timeout * 2.0
    allowed_res_list = []
    if args.allowed_resolutions:
        parts = [p.strip() for p in args.allowed_resolutions.split(',') if p.strip()]
        for p in parts:
            if 'x' in p:
                try:
                    w, h = p.split('x')
                    allowed_res_list.append((int(w), int(h)))
                except:
                    pass
    if (args.width, args.height) not in allowed_res_list:
        allowed_res_list.insert(0, (args.width, args.height))

    rx_thread = threading.Thread(
        target=run_rx_node,
        args=(args.output, config_file, args.width, args.height, fec_cfg, args.sync_threshold, args.chunk_samples, frame_timeout, args.log_interval, store_timeout, args.mac_parity, args.diag_level, args.mac_parity_groups, args.header_repeat_bytes, mac_fec_parity, args.mac_fec_seed, args.save_frames_dir, args.save_frames_interval, args.packet_size, args.max_packets_per_frame, allowed_res_list, args.output_fps, args.max_inflight_frames)
    )
    tx_thread = threading.Thread(
        target=run_tx_node,
        args=(video_path, config_file, allow_synthetic, args.width, args.height, args.quality, fec_cfg, args.gap_samples, sleep_factor, args.max_frames, pkt_repeats, args.log_interval, store_timeout, args.mac_parity, args.diag_level, args.mac_parity_groups, args.header_repeat_bytes, args.repeat_strategy, mac_fec_parity, args.mac_fec_seed, args.frame_gap_ms, args.tx_frame_step, args.packet_size, args.max_packets_per_frame, adapt, args.adapt_interval, args.target_fer, args.target_sync_rate, args.target_post_ber, args.quality_min, args.quality_max, args.quality_step, allowed_res_list)
    )
    
    rx_thread.start()
    tx_thread.start()
    
    try:
        if args.mode == "file":
            tx_thread.join()
            drain_start = time.time()
            while time.time() - drain_start < args.rx_drain_timeout:
                with channel.lock:
                    remaining = len(channel.buffer)
                if remaining == 0:
                    break
                time.sleep(0.05)
            time.sleep(min(2.0, max(0.5, args.frame_timeout * 0.25)))
        else:
            time.sleep(args.duration)
            run_strong = args.run_strong_scenario or any([
                args.strong_tx_power is not None,
                args.strong_path_loss is not None,
                args.strong_rx_gain is not None,
                args.strong_noise_dbfs is not None,
                args.strong_cfo_hz is not None
            ])
            if run_strong and tx_thread.is_alive():
                print("\n[Scenario] Strong Signal (Input > 0dBFS) -> Saturation!")
                channel.path_loss_db = 10.0
                if args.strong_path_loss is not None:
                    channel.path_loss_db = float(args.strong_path_loss)
                if args.strong_tx_power is not None:
                    channel.tx_power_db = float(args.strong_tx_power)
                if args.strong_rx_gain is not None:
                    channel.rx_gain_db = float(args.strong_rx_gain)
                if args.strong_noise_dbfs is not None:
                    channel.base_noise_dbfs = float(args.strong_noise_dbfs)
                if args.strong_cfo_hz is not None:
                    channel.cfo_hz = float(args.strong_cfo_hz)
                time.sleep(args.duration)
        
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
