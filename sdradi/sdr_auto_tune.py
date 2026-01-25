import sys
import time
import json
import numpy as np
import adi
import matplotlib.pyplot as plt
from dataclasses import asdict

# Configuration Headers
CONFIG_FILE = "sdr_tuned_config.json"

def generate_tone(fs, freq, duration, amplitude=1.0):
    t = np.arange(int(fs * duration)) / fs
    return amplitude * np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)

def measure_signal(rx_data, fs, signal_freq):
    # FFT
    N = len(rx_data)
    fft_data = np.fft.fftshift(np.fft.fft(rx_data))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
    
    # Find peak near signal_freq
    # Define a window around target freq
    window_hz = 50e3 # +/- 50kHz search window
    mask = (freqs >= signal_freq - window_hz) & (freqs <= signal_freq + window_hz)
    
    if np.sum(mask) == 0:
        return -100, -100 # Should not happen
        
    peak_pwr = np.max(psd[mask])
    
    # Noise floor: Median of PSD outside signal band
    noise_mask = (np.abs(freqs - signal_freq) > 200e3)
    noise_floor = np.median(psd[noise_mask])
    
    return peak_pwr, noise_floor

def run_auto_tune(ip="ip:192.168.2.1"):
    print(f"=== SDR Auto-Tuning Tool ===")
    print(f"Connecting to {ip}...")
    
    try:
        sdr = adi.Pluto(uri=ip)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Basic Setup
    fs = 2e6 
    fc = 2400e6
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    sdr.rx_rf_bandwidth = int(fs)
    sdr.tx_rf_bandwidth = int(fs)
    sdr.rx_gain = 20 # Fixed RX gain (start safe)
    sdr.rx_buffer_size = 1024*16
    
    # Test Signal
    tone_freq = 100e3 # +100 kHz offset
    tx_signal = generate_tone(fs, tone_freq, duration=0.1) # 100ms
    # Scale for hardware 14-bit
    tx_signal *= 2**14 * 0.5 
    
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_signal)
    
    print("\nStarting Gain Sweep...")
    print(f"{'TX Gain':>10} | {'Peak (dB)':>10} | {'Noise (dB)':>10} | {'SNR (dB)':>10}")
    print("-" * 50)
    
    results = []
    
    # Sweep from -80 to 0
    start_gain = -80
    end_gain = 0
    step = 5
    
    tuned_gain = None
    max_snr = 0
    
    current_gain = start_gain
    prev_peak = -100
    
    saturation_counts = 0
    
    while current_gain <= end_gain:
        sdr.tx_hardwaregain_chan0 = current_gain
        time.sleep(0.1) # Settle
        
        # Measure
        rx = sdr.rx()
        peak, noise = measure_signal(rx, fs, tone_freq)
        snr = peak - noise
        
        print(f"{current_gain:10d} | {peak:10.1f} | {noise:10.1f} | {snr:10.1f}")
        
        results.append({
            'gain': current_gain,
            'peak': peak,
            'noise': noise,
            'snr': snr
        })
        
        # Check Saturation/Linearity
        # Expected: +5dB gain -> +5dB peak
        # Non-linear detection: If delta peak < delta gain - 2dB (compression)
        
        delta_gain = step
        delta_peak = peak - prev_peak
        
        # Only check linearity if we measure a signal above noise floor
        if peak > noise + 10 and current_gain > start_gain + step:
            if delta_peak < delta_gain - 1.5:
                # Compression detected
                print(f"  [!] Non-linear growth (Delta Peak={delta_peak:.1f} vs Gain={delta_gain}). Saturation?")
                saturation_counts += 1
                if saturation_counts >= 2:
                    print("  [!] Saturation confirmed. Stopping sweep.")
                    break
            else:
                saturation_counts = 0 # Reset if linear again (unlikely but possible noise)
        
        # Update best
        # Criteria: Max SNR before saturation
        if snr > max_snr:
            max_snr = snr
            tuned_gain = current_gain
            
        prev_peak = peak
        current_gain += step
        
    sdr.tx_destroy_buffer()
    
    # Verify Tuned Result
    # Back off 3-6dB from peak to be safe/linear
    safe_gain = tuned_gain - 5 if tuned_gain is not None else -30
    print(f"\nOptimization Complete.")
    print(f"Max SNR Reached: {max_snr:.1f} dB at Gain {tuned_gain}")
    print(f"Recommended Safe TX Gain: {safe_gain} dB")
    
    # Save Config
    config_data = {
        "sdr_ip": "ip:192.168.2.1",
        "device": "pluto",
        "fc": fc,
        "fs": fs,
        "bandwidth": fs,
        "tx_gain": safe_gain,
        "rx_gain": 20, # Stick to 20
        "timestamp": time.time()
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)
        
    print(f"Configuration saved to {CONFIG_FILE}")

if __name__ == "__main__":
    run_auto_tune()
