import sys
import time
import json
import numpy as np
import adi

CONFIG_FILE = "sdr_tuned_config.json"

def generate_sine(fs, freq, duration, amplitude=0.9):
    t = np.arange(int(fs * duration)) / fs
    # 0.9 amplitude to avoid DAC clipping, let the hardware gain do the work
    return amplitude * np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)

def measure_metrics(rx_data, fs, signal_freq):
    # Time Domain Metrics
    max_val = np.max(np.abs(rx_data))
    
    # Frequency Domain Metrics
    N = len(rx_data)
    if N == 0: return -100, -100, 0
    
    fft_data = np.fft.fftshift(np.fft.fft(rx_data))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
    
    # Peak search window +/- 50kHz
    mask = (freqs >= signal_freq - 50e3) & (freqs <= signal_freq + 50e3)
    if np.sum(mask) == 0: peak_pwr = -100
    else: peak_pwr = np.max(psd[mask])
    
    # Noise floor (far from signal)
    noise_mask = (np.abs(freqs - signal_freq) > 200e3)
    if np.sum(noise_mask) == 0: noise_floor = -100
    else: noise_floor = np.median(psd[noise_mask])
    
    return peak_pwr, noise_floor, max_val

def run_desense_tune(ip="ip:192.168.2.1"):
    print("=== Receiver Desensitization Tuning ===")
    print(f"Connecting to {ip}...")
    
    try:
        sdr = adi.Pluto(uri=ip)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Setup
    fs = 2e6 
    fc = 2400e6
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(fc)
    sdr.rx_rf_bandwidth = int(fs)
    sdr.rx_buffer_size = 1024*16
    
    # CRITICAL: Desensitize Receiver
    print("Setting RX Gain to 0 dB (Minimum)...")
    sdr.rx_hardwaregain_chan0 = 0
    
    # Prepare TX Sine
    tone_freq = 100e3 
    tx_signal = generate_sine(fs, tone_freq, 0.1)
    tx_signal *= 2**14 # Scale to hardware
    
    # Enable TX
    sdr.tx_lo = int(fc)
    sdr.tx_rf_bandwidth = int(fs)
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_signal)
    
    print("\nStarting TX Gain Sweep (-60 to 0)...")
    print(f"{'TX Gain':>8} | {'Peak (dB)':>10} | {'Noise (dB)':>10} | {'SNR':>6} | {'Max Amp':>8} | {'Status'}")
    print("-" * 75)
    
    best_gain = None
    max_snr = 0
    
    prev_peak = -999
    
    # Sweep
    for tx_gain in range(-60, 5, 5): # Up to 0
        sdr.tx_hardwaregain_chan0 = tx_gain
        time.sleep(0.15)
        
        rx = sdr.rx()
        peak, noise, max_amp = measure_metrics(rx, fs, tone_freq)
        snr = peak - noise
        
        # Analysis
        delta_gain = 5
        delta_peak = peak - prev_peak
        
        status = "Linear"
        
        # Check Saturation (Non-linearity)
        # If we added 5dB gain but got < 3dB signal increase
        if prev_peak > -100 and delta_peak < 3.0:
            status = "Saturated (Comp)"
        
        # Check Clipping (Max Amp) -> Pluto raw is usually signed 12-bit (2048) or 16-bit? 
        # Usually pyadi-iio returns values that can go up to ~2048 or 32768 depending on scale.
        # Let's assume > 1800 is risky if it's 12-bit mode.
        if max_amp > 2000:
             status = "CLIPPING!"
            
        print(f"{tx_gain:8d} | {peak:10.1f} | {noise:10.1f} | {snr:6.1f} | {max_amp:8.0f} | {status}")
        
        # Pick Best
        # Must be Linear and not Clipping
        if "Linear" in status and snr > max_snr:
            max_snr = snr
            best_gain = tx_gain
            
        prev_peak = peak
        
        if "Saturated" in status or "CLIPPING" in status:
            if best_gain is None: best_gain = tx_gain - 5 # Back off if we started saturated
            break
            
    sdr.tx_destroy_buffer()
    
    print("-" * 75)
    print(f"Best Linear TX Gain found: {best_gain} dB (SNR: {max_snr:.1f} dB)")
    
    if best_gain is None:
        best_gain = -60
        print("Warning: Could not find linear region. Defaulting to -60.")
        
    # Save
    cfg = {
        "sdr_ip": "ip:192.168.2.1",
        "device": "pluto",
        "fc": fc,
        "fs": fs,
        "bandwidth": fs,
        "tx_gain": best_gain,
        "rx_gain": 0, # Force 0
        "timestamp": time.time()
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"Saved optimized config to {CONFIG_FILE}")

if __name__ == "__main__":
    run_desense_tune()
