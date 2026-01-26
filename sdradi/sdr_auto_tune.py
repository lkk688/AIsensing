import sys
import time
import json
import numpy as np
import adi
import argparse

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
    
    # Calculate Power Spectral Density (dB)
    # Normalize by N to get consistent magnitude independent of FFT size
    psd = 20 * np.log10(np.abs(fft_data) / N + 1e-12)
    
    # Find peak near signal_freq
    window_hz = 50e3 # +/- 50kHz search window
    mask = (freqs >= signal_freq - window_hz) & (freqs <= signal_freq + window_hz)
    
    if np.sum(mask) == 0:
        return -100, -100
        
    peak_pwr = np.max(psd[mask])
    
    # Noise floor: Median of PSD outside signal band and DC
    # Exclude DC (center) and Signal
    noise_mask = (np.abs(freqs - signal_freq) > 100e3) & (np.abs(freqs) > 10e3)
    noise_floor = np.median(psd[noise_mask])
    
    return peak_pwr, noise_floor

def run_diagnostics(ip="ip:192.168.3.2"):
    print(f"=== SDR Link Diagnostics & Auto-Tune ===")
    print(f"Target IP: {ip}")
    
    try:
        sdr = adi.Pluto(uri=ip)
    except Exception as e:
        print(f"FATAL: Failed to connect to SDR: {e}")
        return

    # --- 1. Load & Apply Config ---
    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            print(f"Loaded config: FC={cfg['fc']/1e6}MHz, FS={cfg['fs']/1e6}MHz, TX Gain={cfg['tx_gain']}")
    except:
        print("Warning: Could not load config, using defaults.")
        cfg = {"fc": 2400e6, "fs": 2e6, "tx_gain": -10, "bandwidth": 2000000}

    # Setup SDR
    fs = float(cfg['fs'])
    fc = float(cfg['fc'])
    
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(fc)
    sdr.tx_lo = int(fc)
    sdr.rx_rf_bandwidth = int(cfg['bandwidth'])
    sdr.tx_rf_bandwidth = int(cfg['bandwidth'])
    sdr.rx_buffer_size = 1024*32
    
    # Ensure manual gain mode
    sdr.gain_control_mode_chan0 = "manual"
    
    # --- 2. Measure Noise Floor (TX OFF) ---
    print("\n[Step 1] Measuring Noise Floor (TX OFF)...")
    sdr.tx_destroy_buffer() # Stop any TX
    sdr.rx_hardwaregain_chan0 = 60 # Max Sensitivity
    
    # Take a few readings
    noise_levels = []
    for _ in range(5):
        rx = sdr.rx()
        # Measure noise floor across whole band (median)
        _, nf = measure_signal(rx, fs, 0) # Freq 0 irrelevant for noise floor
        noise_levels.append(nf)
        time.sleep(0.1)
        
    avg_noise = np.mean(noise_levels)
    print(f"  -> Noise Floor at Max Gain (60dB): {avg_noise:.1f} dB")
    
    if avg_noise > -50:
        print("  [WARNING] High Noise Floor! Possible interference or local electronics noise.")
    else:
        print("  [OK] Noise floor looks healthy (quiet).")

    # --- 3. Signal Presence Check (TX ON) ---
    print("\n[Step 2] Checking Signal Presence...")
    tone_freq = 200e3 # +200 kHz offset to avoid DC
    tx_signal = generate_tone(fs, tone_freq, duration=0.1)
    tx_signal *= 2**14 * 0.5 # 50% Full Scale
    
    sdr.tx_hardwaregain_chan0 = int(cfg['tx_gain']) # Use configured TX gain
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_signal)
    
    # Quick Check at medium RX gain
    sdr.rx_hardwaregain_chan0 = 30
    time.sleep(0.5)
    rx = sdr.rx()
    peak, noise = measure_signal(rx, fs, tone_freq)
    snr = peak - noise
    
    print(f"  -> TX Gain: {cfg['tx_gain']} dB")
    print(f"  -> RX Gain: 30 dB")
    print(f"  -> Peak: {peak:.1f} dB, Noise: {noise:.1f} dB, SNR: {snr:.1f} dB")
    
    if snr < 5:
        print("  [CRITICAL] Signal barely visible! Link is extremely weak.")
    elif snr > 20:
        print("  [OK] Strong signal detected.")
    else:
        print("  [OK] Signal detected, but could be stronger.")

    # --- 4. RX Gain Sweep (AGC Emulation) ---
    print("\n[Step 3] Sweeping RX Gain (Optimization)...")
    print(f"{'RX Gain':>10} | {'Peak (dB)':>10} | {'Noise (dB)':>10} | {'SNR (dB)':>10} | {'Status'}")
    print("-" * 70)
    
    best_gain = 30
    max_snr = -100
    
    for gain in range(0, 75, 5): # Pluto goes up to ~73dB? Limit to 70 normally.
        sdr.rx_hardwaregain_chan0 = gain
        time.sleep(0.1)
        
        rx = sdr.rx()
        peak, nf = measure_signal(rx, fs, tone_freq)
        snr = peak - nf
        
        raw_peak = np.max(np.abs(rx))
        status = ""
        
        # Detect Saturation (ADC Clipping)
        # 12-bit ADC -> +/- 2048. Complex magnitude approx 2048*1.4? 
        # Safest bet: > 2000 is danger zone.
        if raw_peak > 2000:
            status = "CLIPPING!"
            # Penalty for clipping in optimization
            valid_snr = -10
        else:
            valid_snr = snr
            
        print(f"{gain:10d} | {peak:10.1f} | {nf:10.1f} | {snr:10.1f} | {status}")
        
        if valid_snr > max_snr:
            max_snr = valid_snr
            best_gain = gain
            
    print("-" * 70)
    print(f"\nDiagnostic Result:")
    print(f"  -> Optimal RX Gain: {best_gain} dB (Max SNR: {max_snr:.1f} dB)")
    
    # Save optimized RX gain?
    # Actually, we implemented Software AGC in the main loop. 
    # But updating the default starting point is good.
    
    if max_snr < 10:
        print("\n[CONCLUSION] STARTLINGLY LOW SNR.")
        print("  Even at optimal gain, SNR is poor.")
        print("  Potential Causes:")
        print("  1. Distance too great for current TX Power (-10dB). -> Try increasing TX Gain.")
        print("  2. Antennas Disconnected or Wrong Port. -> CHECK HARDWARE.")
        print("  3. Frequency Mismatch/Interference. -> Try slightly different frequency.")
        
        # Suggest Action
        print(f"  -> Recommendation: Increase TX Gain in {CONFIG_FILE}")
        
    elif best_gain > 50:
         print("\n[CONCLUSION] Link is operational but requires High Gain.")
         print("  -> AGC will handle this, but verify antennas.")
         
    else:
         print("\n[CONCLUSION] Link is Healthy.")

    # Stop TX
    sdr.tx_destroy_buffer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="ip:192.168.3.2", help="SDR IP address")
    args = parser.parse_args()
    
    run_diagnostics(ip=args.ip)
