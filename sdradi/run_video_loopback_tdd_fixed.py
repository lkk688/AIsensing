#!/usr/bin/env python
"""
run_video_loopback_tdd_fixed.py - Fixed TDD Video Loopback

Based on run_video_loopback_tdd.py but with fixes for timeout issues.
"""

import sys
import time
import numpy as np
import argparse
import os
import zlib
import iio
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Qt Wayland issue
os.environ["QT_QPA_PLATFORM"] = "xcb"

from sdr_video_commv2 import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType

def set_loopback_mode(sdr_ip, mode, ctx=None):
    """
    Set AD9361 loopback mode.
    """
    try:
        import iio
        if ctx is None:
            ctx = iio.Context(sdr_ip)
        phy = ctx.find_device('ad9361-phy')
        if phy:
            phy.debug_attrs['loopback'].value = str(mode)
            actual = phy.debug_attrs['loopback'].value
            print(f"[SDR] Loopback mode set to {actual}")
            return True
    except Exception as e:
        print(f"[SDR] Failed to set loopback: {e}")
    return False

# =============================================================================
# TDD Controller
# =============================================================================

class TDDController:
    def __init__(self, uri='ip:192.168.2.2'):
        self.uri = uri
        self.tdd = None
        self.enabled = False
        
    def connect(self):
        try:
            import adi
            self.tdd = adi.tddn(self.uri)
            print(f"[TDD] Connected to AXI TDD Controller")
            return True
        except Exception as e:
            print(f"[TDD] Failed to connect: {e}")
            return False
    
    def configure(self, frame_length_ms=20.0, tx_duration_ms=8.0, 
                  guard_ms=2.0, rx_duration_ms=8.0):
        if self.tdd is None:
            return False
        
        try:
            self.tdd.enable = False
            self.tdd.frame_length_ms = frame_length_ms
            self.tdd.startup_delay_ms = 0.1
            
            # TX burst (channel 0)
            if len(self.tdd.channel) > 0:
                self.tdd.channel[0].on_ms = 0.0
                self.tdd.channel[0].off_ms = tx_duration_ms
                self.tdd.channel[0].enable = True
                self.tdd.channel[0].polarity = 0
            
            # RX burst (channel 1)
            rx_start = tx_duration_ms + guard_ms
            rx_end = rx_start + rx_duration_ms
            if len(self.tdd.channel) > 1:
                self.tdd.channel[1].on_ms = rx_start
                self.tdd.channel[1].off_ms = rx_end
                self.tdd.channel[1].enable = True
                self.tdd.channel[1].polarity = 0
            
            self.tdd.burst_count = 0
            self.tdd.sync_internal = True
            self.tdd.internal_sync_period_ms = frame_length_ms
            
            print(f"[TDD] Configured: frame={frame_length_ms}ms, "
                  f"TX={0.0}-{tx_duration_ms}ms, RX={rx_start}-{rx_end}ms")
            return True
        except Exception as e:
            print(f"[TDD] Configuration failed: {e}")
            return False
    
    def enable(self):
        if self.tdd is None: return False
        try:
            self.tdd.enable = True
            self.enabled = True
            print("[TDD] Enabled")
            return True
        except Exception as e:
            print(f"[TDD] Enable failed: {e}")
            return False
    
    def disable(self):
        if self.tdd is None: return False
        try:
            self.tdd.enable = False
            self.enabled = False
            print("[TDD] Disabled")
            return True
        except Exception as e:
            print(f"[TDD] Disable failed: {e}")
            return False

# =============================================================================
# Buffer Windowing Sync
# =============================================================================

def sync_with_windowing(rx_signal, preamble, payload_samples, threshold=15.0, 
                        sample_rate=2e6, verbose=False):
    corr = np.abs(np.correlate(rx_signal, preamble, mode='valid'))
    peaks = []
    min_distance = len(preamble) // 2
    
    for i in range(len(corr)):
        if corr[i] > threshold:
            is_local_max = True
            for j in range(max(0, i - min_distance), min(len(corr), i + min_distance)):
                if corr[j] > corr[i]:
                    is_local_max = False
                    break
            
            if is_local_max:
                samples_after = len(rx_signal) - (i + len(preamble))
                if samples_after >= payload_samples:
                    peaks.append({'idx': i, 'corr': corr[i]})
    
    if not peaks:
        return None, {'sync_success': False, 'peak_val': np.max(corr) if len(corr) > 0 else 0}
    
    best = max(peaks, key=lambda x: x['corr'])
    payload_start = best['idx'] + len(preamble)
    
    # CFO estimation
    half = len(preamble) // 2
    preamble_rx = rx_signal[best['idx']:best['idx'] + len(preamble)]
    if len(preamble_rx) >= len(preamble):
        phase1 = np.angle(np.sum(preamble_rx[:half] * preamble[:half].conj()))
        phase2 = np.angle(np.sum(preamble_rx[half:] * preamble[half:].conj()))
        cfo = (phase2 - phase1) / (half / sample_rate) / (2 * np.pi)
    else:
        cfo = 0
    
    t = np.arange(payload_samples) / sample_rate
    cfo_correction = np.exp(-2j * np.pi * cfo * t)
    payload = rx_signal[payload_start:payload_start + payload_samples] * cfo_correction
    
    return payload, {
        'sync_success': True,
        'peak_val': best['corr'],
        'peak_idx': best['idx'],
        'cfo_est': cfo,
        'num_peaks': len(peaks)
    }

# =============================================================================
# Main
# =============================================================================

def main():
    import adi
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='ip:192.168.2.2')
    parser.add_argument('--config', default='sdr_tuned_config.json')
    parser.add_argument('--mode', default='tdd', choices=['tdd', 'cyclic'])
    parser.add_argument('--max-frames', type=int, default=5)
    parser.add_argument('--quality', type=int, default=35)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--fec', type=str, default='ldpc')
    parser.add_argument('--packet-size', type=int, default=1024)
    parser.add_argument('--mac-repeats', type=int, default=2)
    parser.add_argument('--sync-threshold', type=float, default=15.0)
    parser.add_argument('--tx-gain', type=int, default=0)
    parser.add_argument('--rx-gain', type=int, default=60)
    parser.add_argument('--loopback', type=str, default='rf', choices=['rf', 'digital', 'disabled'])
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    print(f"SDR Video Loopback Fixed - {args.mode.upper()} Mode")
    
    # Load Config
    if os.path.exists(args.config):
        sdr_cfg = SDRConfig.load_from_json(args.config)
    else:
        sdr_cfg = SDRConfig()
    sdr_cfg.sdr_ip = args.ip
    sdr_cfg.tx_gain = args.tx_gain
    sdr_cfg.rx_gain = args.rx_gain
    
    # FEC
    fec_map = {"none": FECType.NONE, "repetition": FECType.REPETITION, "ldpc": FECType.LDPC}
    fec_type = fec_map.get(args.fec.lower(), FECType.LDPC)
    fec_cfg = FECConfig(enabled=fec_type != FECType.NONE, fec_type=fec_type)
    
    # OFDM
    ofdm_cfg = OFDMConfig()
    ofdm_cfg.sync_threshold = args.sync_threshold
    
    # Initialize Link
    link = SDRVideoLink(sdr_config=sdr_cfg, fec_config=fec_cfg, ofdm_config=ofdm_cfg)
    link.video_config.packet_size = args.packet_size
    link.video_config.resolution = (args.width, args.height)
    
    # Calculate Timing
    bits_per_frame = ofdm_cfg.bits_per_frame
    samples_per_frame = ofdm_cfg.samples_per_frame
    sample_rate = sdr_cfg.fs
    
    expected_fec_bits = link.fec_codec.n if fec_cfg.enabled and hasattr(link.fec_codec, 'n') else args.packet_size * 8
    num_ofdm_frames = int(np.ceil(expected_fec_bits / bits_per_frame))
    expected_payload_samples = num_ofdm_frames * samples_per_frame
    preamble = link._generate_preamble()
    preamble_len = len(preamble)
    
    tx_samples = preamble_len + expected_payload_samples + 500
    tx_duration_ms = (tx_samples / sample_rate) * 1000
    guard_ms = 1.0
    rx_duration_ms = tx_duration_ms * 2
    frame_length_ms = tx_duration_ms + guard_ms + rx_duration_ms + 2.0
    
    print(f"[TDD] TX duration: {tx_duration_ms:.2f} ms, Frame: {frame_length_ms:.2f} ms")
    
    # TDD Controller
    tdd_ctrl = None
    if args.mode == 'tdd':
        tdd_ctrl = TDDController(args.ip)
        if tdd_ctrl.connect():
            tdd_ctrl.configure(frame_length_ms, tx_duration_ms, guard_ms, rx_duration_ms)
            tdd_ctrl.enable() # Enable immediately
        else:
            print("[TDD] Failed to connect, falling back to cyclic")
            args.mode = 'cyclic'
            
    # Connect SDR using simplified approach from digital script
    # But we need direct access to adi object for TDD bursting
    tx_sdr = adi.Pluto(args.ip)
    rx_sdr = tx_sdr
    
    # Apply Config
    tx_sdr.sample_rate = int(sample_rate)
    tx_sdr.tx_lo = int(sdr_cfg.fc)
    tx_sdr.tx_hardwaregain_chan0 = args.tx_gain
    rx_sdr.rx_lo = int(sdr_cfg.fc)
    rx_sdr.rx_hardwaregain_chan0 = args.rx_gain
    
    # Buffer Config
    rx_buffer_size = 32768 # Safe default
    rx_sdr.rx_buffer_size = rx_buffer_size
    if hasattr(rx_sdr, "_rxadc") and hasattr(rx_sdr._rxadc, "set_kernel_buffers_count"):
        rx_sdr._rxadc.set_kernel_buffers_count(4)
        
    print(f"[SDR] RX buffer: {rx_buffer_size}")
    
    # Loopback
    if args.loopback != 'disabled':
        mode_val = 1 if args.loopback == 'digital' else 0
        set_loopback_mode(args.ip, mode_val)
        
    # Main Loop
    stats = {'tx': 0, 'rx': 0}
    
    try:
        for frame_idx in range(args.max_frames):
            print(f"\n--- Frame {frame_idx + 1} ---")
            
            # Encode (dummy frame for speed)
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10,10), (50,50), (255,255,255), -1)
            packets = link.video_codec.encode_frame(frame, quality=args.quality)
            print(f"[TX] Packets: {len(packets)}")
            
            rx_count = 0
            for pkt in packets:
                # Prepare Signal
                pkt_arr = np.frombuffer(pkt[0], dtype=np.uint8)
                bits = np.unpackbits(pkt_arr)
                fec_bits = link.fec_codec.encode(bits)
                tx_signal = link.transceiver.modulate(fec_bits)
                tx_full = np.concatenate([preamble, tx_signal])
                max_val = np.max(np.abs(tx_full))
                if max_val > 0: tx_full /= max_val
                tx_scaled = (tx_full * 2**14).astype(np.complex64)
                
                # Transmit
                for rep in range(args.mac_repeats):
                    if args.mode == 'tdd':
                        tx_sdr.tx_destroy_buffer()
                        tx_sdr.tx_cyclic_buffer = False
                        tx_sdr.tx(tx_scaled)
                        time.sleep((tx_duration_ms + guard_ms) / 1000.0)
                    else:
                        tx_sdr.tx_destroy_buffer()
                        tx_sdr.tx_cyclic_buffer = True
                        tx_repeats = max(5, 32768 // len(tx_scaled) + 1)
                        tx_sdr.tx(np.tile(tx_scaled, tx_repeats))
                        time.sleep(0.1)
                        
                    stats['tx'] += 1
                    
                    # Receive
                    success = False
                    for attempt in range(3):
                        try:
                            # NO destroy buffer here!
                            rx_data = rx_sdr.rx()
                            if len(rx_data) < len(tx_scaled): continue
                            
                            # Normalize
                            rx_norm = rx_data / np.max(np.abs(rx_data)) if np.max(np.abs(rx_data)) > 0 else rx_data
                            
                            # Sync
                            payload, met = sync_with_windowing(rx_norm, preamble, expected_payload_samples, 
                                                             threshold=args.sync_threshold, sample_rate=sample_rate)
                            
                            if payload is not None and met['sync_success']:
                                # Demod/Decode
                                rx_fec_bits, _ = link.transceiver.demodulate(payload)
                                rx_bits = link.fec_codec.decode(rx_fec_bits)
                                rx_bytes = np.packbits(rx_bits[:len(bits)]).tobytes()
                                
                                # Verify (simple CRC check via zlib)
                                info = link.video_codec.parse_packet_header(rx_bytes)
                                if info and (zlib.crc32(info['payload']) & 0xFFFFFFFF) == info['crc']:
                                    rx_count += 1
                                    stats['rx'] += 1
                                    success = True
                                    if args.verbose: print(f"  [RX] OK (rep={rep})")
                                    break
                        except Exception as e:
                            if args.verbose: print(f"  [RX] Error: {e}")
                    
                    if success: break
                
                if args.mode == 'cyclic':
                    tx_sdr.tx_destroy_buffer()
            
            print(f"[Result] {rx_count}/{len(packets)} OK")

    except KeyboardInterrupt:
        pass
    finally:
        if tdd_ctrl: tdd_ctrl.disable()
        print(f"\nStats: TX={stats['tx']}, RX={stats['rx']}")

if __name__ == "__main__":
    main()
