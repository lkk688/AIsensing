import numpy as np
import time
import argparse
import sys
import adi

from sdr_video_comm import OFDMTransceiver, OFDMConfig, OTFSTransceiver, OTFSConfig, PacketFramer

def generate_zc_sequence(length, root):
    """Generate Zadoff-Chu sequence."""
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)

def synchronize(signal, zc):
    """
    Robust Synchronization & Channel Estimation routine.
    1. Coarse Preamble Detection (Correlate with ZC*)
    2. Channel Estimation (Extract CIR -> FFT)
    3. CFO Estimation (Schmidl-Cox)
    4. Payload Extraction
    """
    # 1. Remove DC
    signal = signal - np.mean(signal)
    
    # 2. Coarse Detection (Original Logic)
    # Note: numpy correlate does not conjugate second arg by default.
    # We use zc directly here as it worked before (maybe due to phase structure?)
    corr_complex = np.correlate(signal, zc, mode='valid')
    corr_mag = np.abs(corr_complex)
    
    peak_idx = np.argmax(corr_mag)
    max_val = corr_mag[peak_idx]
    
    # Threshold
    noise_floor = np.median(corr_mag)
    if max_val < noise_floor * 5.0:
        return np.array([]), None, {'sync_success': False, 'peak_val': max_val}

    # 3. Channel Estimation from Preamble
    # CIR is contained in the correlation peak loop
    # We extract a window around the peak to capture multipath
    # Window: [peak, peak + 32] (Assume delay spread < 32 samples)
    # ZC autocorrelation is perfect 1/N, so corr_complex[peak+lag] approx h[lag] * N
    
    # Scale: Divide by N (Energy of ZC)
    N_zc = len(zc)
    cir_est = np.zeros(64, dtype=complex) # Allocate reasonable delay spread 
    
    # Copy available taps
    L_extract = min(64, len(corr_complex) - peak_idx)
    if L_extract > 0:
        cir_est[:L_extract] = corr_complex[peak_idx : peak_idx+L_extract] / N_zc
    
    # Frequency Domain Channel Estimate (for Equalizer)
    # Target size: 256 (OTFS N_delay)
    H_est = np.fft.fft(cir_est, n=256)

    # 4. Payload Extraction
    tone_start = peak_idx + 127 + 50
    tone_end = tone_start + 256
    payload_start = tone_end + 50
    
    if payload_start >= len(signal):
         return np.array([]), None, {'sync_success': False, 'peak_val': max_val}

    # 5. CFO Estimation
    tone_rx = signal[tone_start:tone_end]
    d_phi_meas = np.angle(np.mean(tone_rx[1:] * np.conj(tone_rx[:-1])))
    d_phi_nominal = 2 * np.pi / 16.0
    cfo_rad = d_phi_meas - d_phi_nominal
    
    # Apply Correction
    corrected_payload = signal[payload_start:]
    t = np.arange(len(corrected_payload))
    correction = np.exp(-1j * cfo_rad * t)
    corrected_payload = corrected_payload * correction
    
    return corrected_payload, H_est, {
        'sync_success': True, 
        'peak_val': max_val, 
        'peak_idx': peak_idx,
        'cfo_est': cfo_rad * (2e6/(2*np.pi)),
        'cfo_rad': cfo_rad,
        'corr_complex': corr_complex
    }

def main():
    parser = argparse.ArgumentParser(description="Realistic RX (OFDM Transceiver)")
    parser.add_argument("--ip", type=str, default="ip:192.168.2.2", help="SDR IP")
    parser.add_argument("--gain", type=int, default=60, help="RX Gain")
    parser.add_argument("--waveform", type=str, default="ofdm", choices=["ofdm", "otfs"], help="Waveform type")
    args = parser.parse_args()

    # Config
    if args.waveform == "ofdm":
        cfg = OFDMConfig(mod_order=2)
        tr = OFDMTransceiver(cfg)
        print("Selected Waveform: OFDM (BPSK)")
    else:
        cfg = OTFSConfig(mod_order=2)
        tr = OTFSTransceiver(cfg)
        print("Selected Waveform: OTFS (BPSK)")
    
    # Preamble for Sync
    zc = generate_zc_sequence(127, 25)
    
    # SDR Setup
    print(f"Connecting to SDR {args.ip}...")
    sdr = adi.Pluto(args.ip)
    sdr.rx_lo = int(2.3e9) + 6561
    sdr.sample_rate = int(2e6)
    sdr.rx_rf_bandwidth = int(1e6)
    sdr.rx_buffer_size = 1024 * 32 # Explicit reasonable size (32k), removing 65k
    sdr.rx_buffer_size = 32768 # Use 32k as compromise
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = args.gain
    
    print("Waiting for frames...")
    
    while True:
        try:
            samples = sdr.rx()
            
            # Synchronize
            payload, H_est, metrics = synchronize(samples, zc)
            
            if metrics['sync_success']:
                print(f"Locked! Peak={metrics['peak_val']:.1f} CFO={metrics['cfo_est']:.1f}Hz")
                
                found_packets = False
                
                # Timing Search: Try offsets around the detected peak
                # Simulation showed OTFS is very sensitive to +/- 1 sample offset.
                # Expanding search to +/- 20 to cover potential gross misalignment
                timing_offsets = range(-20, 21, 2) # Step 2 for speed
                
                # Retrieve sync info to reconstruct extraction
                peak_idx = metrics['peak_idx']
                cfo_rad = metrics['cfo_rad']
                corr_complex = metrics['corr_complex']
                N_zc = 127 # Hardcoded from generate_zc_sequence(127, 25)
                
                # Derived from synchronize() logic:
                base_payload_start = peak_idx + 127 + 50 + 256 + 50
                
                for t_off in timing_offsets:
                    # 1. Re-extract Payload with timing offset
                    p_start = base_payload_start + t_off
                    
                    if p_start < 0 or p_start >= len(samples):
                        continue
                        
                    # Extract Raw Payload (No CFO yet)
                    t_payload_raw = samples[p_start:]
                    
                    # Apply CFO Correction
                    t_vec = np.arange(len(t_payload_raw))
                    correction = np.exp(-1j * cfo_rad * t_vec)
                    t_payload = t_payload_raw * correction
                    
                    # Normalize Payload
                    pwr_p = np.mean(np.abs(t_payload)**2)
                    if pwr_p > 0:
                        t_payload = t_payload / np.sqrt(pwr_p)
                        
                    # 2. Re-estimate Channel (H_est) for this specific timing offset
                    # We must extract CIR from corr_complex starting at peak_idx + t_off
                    # Window: 64 samples
                    current_peak = peak_idx + t_off
                    L_extract = min(64, len(corr_complex) - current_peak)
                    
                    if L_extract <= 0:
                        continue
                        
                    cir_est = np.zeros(64, dtype=complex)
                    cir_est[:L_extract] = corr_complex[current_peak : current_peak+L_extract] / N_zc
                    
                    # FFT to Freq Domain (OTFS uses 256 delays)
                    H_est_shifted = np.fft.fft(cir_est, n=256)
                    
                    # Normalize H_est
                    pwr_h = np.mean(np.abs(H_est_shifted)**2)
                    if pwr_h > 0:
                        H_est_shifted = H_est_shifted / np.sqrt(pwr_h)

                    # Hypothesis Search: Normal vs Spectral Inversion
                    hypotheses = [
                        ("Normal", t_payload),
                        ("Spectrally Inverted", np.conj(t_payload))
                    ]
                    
                    for h_name, h_payload in hypotheses:
                        # Demodulate this hypothesis
                        if args.waveform == "otfs":
                            # Use the aligned H_est
                            bits, dem_m = tr.demodulate(h_payload, channel_est=H_est_shifted)
                        else:
                            bits, dem_m = tr.demodulate(h_payload)
                        demod_metrics = dem_m
                            
                        # Bit-level scanning
                        for bit_offset in range(8):
                            if bit_offset == 0:
                                b = bits
                            else:
                                b = bits[bit_offset:]
                                
                            rx_bytes = np.packbits(b).tobytes()
                            packets = PacketFramer.deframe(rx_bytes)
                            valid_cnt = sum(1 for _, _, v in packets if v)
                            
                            if valid_cnt > 0:
                                print(f"  Frame Decode [Time {t_off}, {h_name}, Bit {bit_offset}]: Found {len(packets)} packets, {valid_cnt} Valid CRC.")
                                for i, (p, seq, valid) in enumerate(packets[:5]):
                                    status = "OK" if valid else "CRC-FAIL"
                                    print(f"    Pkt {i}: Seq {seq}, Len {len(p)}, {status}")
                                found_packets = True
                                break
                        
                        if found_packets: break
                    if found_packets: break
                
                if not found_packets:
                     print("  Frame Decode: No valid packets found (Scanned Time/Phase/Bits).")

                snr = demod_metrics.get('snr_est', 0.0)
                num_frames = demod_metrics.get('num_frames', 1)
                print(f"  SNR Est: {snr:.2f} dB, Frames: {num_frames}")
                
                if snr > 5.0 and found_packets:
                    print("  SUCCESS: Signal Locked & Valid Packets Decoding.")
                elif snr > 5.0:
                    print("  SUCCESS: Signal Locked, High SNR (Framing Error).")
                
            else:
                # print(".", end="", flush=True)
                pass
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
