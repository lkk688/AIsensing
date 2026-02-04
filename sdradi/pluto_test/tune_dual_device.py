import sys
import time
import json
import numpy as np
import adi

CONFIG_FILE = "sdr_tuned_config.json"

# Auto-detected URIs from previous step
TX_URI = "usb:1.5.5" # Serial ...294
RX_URI = "usb:9.2.5" # Serial ...eb8

def generate_tone(fs, freq, duration, amplitude=1.0):
    t = np.arange(int(fs * duration)) / fs
    return amplitude * np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)

def measure_signal(rx_data, fs, signal_freq):
    N = len(rx_data)
    if N == 0: return -100, -100
    
    fft_data = np.fft.fftshift(np.fft.fft(rx_data))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
    
    # Peak search window +/- 50kHz
    mask = (freqs >= signal_freq - 50e3) & (freqs <= signal_freq + 50e3)
    if np.sum(mask) == 0: peak_pwr = -100
    else: peak_pwr = np.max(psd[mask])
    
    # Noise floor
    noise_mask = (np.abs(freqs - signal_freq) > 200e3)
    if np.sum(noise_mask) == 0: noise_floor = -100
    else: noise_floor = np.median(psd[noise_mask])
    
    return peak_pwr, noise_floor

def run_dual_tune():
    print("=== Dual-Device SDR Tuning ===")
    print(f"TX Device: {TX_URI}")
    print(f"RX Device: {RX_URI}")
    
    try:
        print("Connecting to TX SDR...")
        sdr_tx = adi.Pluto(uri=TX_URI)
        print("Connecting to RX SDR...")
        sdr_rx = adi.Pluto(uri=RX_URI)
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # Common Settings
    fs = 2e6 
    fc = 2400e6
    
    # Setup TX
    sdr_tx.sample_rate = int(fs)
    sdr_tx.tx_lo = int(fc)
    sdr_tx.tx_rf_bandwidth = int(fs)
    sdr_tx.tx_cyclic_buffer = True
    
    # Setup RX
    sdr_rx.sample_rate = int(fs)
    sdr_rx.rx_lo = int(fc)
    sdr_rx.rx_rf_bandwidth = int(fs)
    sdr_rx.rx_gain = 20 # Start with 20dB
    sdr_rx.rx_buffer_size = 1024*16
    
    # Generate Signal
    col_tone_freq = 100e3
    tx_signal = generate_tone(fs, col_tone_freq, 0.1)
    tx_signal *= 2**14 * 0.5 
    
    # Start TX
    print("Starting Transmitter...")
    sdr_tx.tx(tx_signal)
    
    print("\nStarting Gain Sweep...")
    print(f"{'TX Gain':>8} | {'Peak (dB)':>10} | {'Noise (dB)':>10} | {'SNR':>6} | {'Status'}")
    print("-" * 65)
    
    best_gain = None
    max_snr = 0
    prev_peak = -999
    
    # Sweep from -40 to 0 (Assuming separating devices adds ~40dB path loss compared to loopback)
    # Actually, let's start safe at -80 again
    for tx_gain in range(-80, 5, 5): # Up to 0
        sdr_tx.tx_hardwaregain_chan0 = tx_gain
        time.sleep(0.2)
        
        # Measure on RX Device
        rx = sdr_rx.rx()
        peak, noise = measure_signal(rx, fs, col_tone_freq)
        snr = peak - noise
        
        # Linearity Check
        delta_gain = 5
        delta_peak = peak - prev_peak
        
        status = "Linear"
        
        # If signal is visible (above noise)
        if peak > noise + 5:
             # Check for 1:1 slope
             if prev_peak > -100 and delta_peak < 3.0:
                 status = "Sat/Comp"
                 
        print(f"{tx_gain:8d} | {peak:10.1f} | {noise:10.1f} | {snr:6.1f} | {status}")
        
        if "Linear" in status and snr > max_snr:
            max_snr = snr
            best_gain = tx_gain
            
        if "Sat" in status:
             if best_gain is None: best_gain = tx_gain - 5
             break
             
        prev_peak = peak
        
    sdr_tx.tx_destroy_buffer()
    
    print("-" * 65)
    print(f"Best Linear TX Gain: {best_gain} dB (SNR: {max_snr:.1f} dB)")
    
    if best_gain is None:
        best_gain = -50
        print("Defaulting to -50 dB")
        
    # Save Dual Config
    # We need to support dual URIs in the config structure now!
    cfg = {
        "sdr_ip": TX_URI, # Primary (TX)
        "rx_uri": RX_URI, # Secondary (RX) 
        "device": "pluto_dual",
        "fc": fc,
        "fs": fs,
        "bandwidth": fs,
        "tx_gain": best_gain,
        "rx_gain": 20,
        "timestamp": time.time()
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"Saved dual-device config to {CONFIG_FILE}")

if __name__ == "__main__":
    run_dual_tune()
