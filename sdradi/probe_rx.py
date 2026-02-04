import argparse
import adi
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.signal

# Configuration
SDR_IP = "ip:192.168.1.10" # AntSDR E200
FC = 2300e6
FS = 2e6
BW = 1e6

def generate_zadoff_chu(seq_len=127, root=25):
    n = np.arange(seq_len)
    cf = seq_len % 2
    return np.exp(-1j * np.pi * root * n * (n + cf + 2 * 0) / seq_len)

def estimate_channel(rx_samples, gain_db):
    """Detect Preamble, Measure SNR, Estimate Channel."""
    zc = generate_zadoff_chu(127, 25)
    
    # Correlation
    corr = np.correlate(rx_samples, zc, mode='valid')
    peak_idx = np.argmax(np.abs(corr))
    peak_val = np.abs(corr[peak_idx])
    
    # Threshold for Lock
    # ZC AutoCorr Peak is ~127. Noise floor depends on gain.
    # We look for distinct peak.
    
    results = {
        'locked': False,
        'peak_idx': peak_idx,
        'snr': -99,
        'signal_power': 0,
        'noise_power': 0,
        'saturation': False,
        'channel_response': None
    }
    
    # Check Saturation (Pluto ADC Max ~2048)
    max_amp = np.max(np.abs(rx_samples))
    if max_amp > 2000:
        results['saturation'] = True

    # Validate Peak
    # If peak is 10x median, we probably have it.
    noise_floor = np.median(np.abs(corr))
    if peak_val > noise_floor * 5:
        results['locked'] = True
        
        # Structure: [ZC(127)] [Guard(50)] [Tone(512)] [Guard(50)] [Silence(256)]
        # Peak Index corresponds to START of ZC match? 
        # numpy correlate `valid`: index 0 means alignment at start.
        # So refined start index = peak_idx.
        
        start_idx = peak_idx
        
        # Extract Regions (Safe bounds check needed)
        # Tone starts at 127 + 50 = 177 samples after start
        tone_start = start_idx + 177
        tone_end = tone_start + 512
        
        # Tone Power
        if tone_end < len(rx_samples):
            tone_seg = rx_samples[tone_start:tone_end]
            signal_power = np.mean(np.abs(tone_seg)**2)
            results['signal_power'] = signal_power
            
            # CFO Estimation (Phase Diff)
            # Use first 100 samples to be safe (Probe has 512, Video has 256)
            safe_len = min(len(tone_seg), 100)
            if safe_len > 10:
                short_seg = tone_seg[:safe_len]
                phase_diff = np.angle(short_seg[1:] * np.conj(short_seg[:-1]))
                avg_phase_diff = np.mean(phase_diff)
                fs = 2e6 # Hardcoded FS
                freq_measured = avg_phase_diff * fs / (2 * np.pi)
                # probe_tx.py uses 50kHz tone. sdr_video_comm.py uses "Fs/16" = 125kHz
                # We need to know WHICH source is running to calc offset.
                # Assuming sdr_video_comm.py (Video) -> Target 125kHz
                # If probe_tx.py -> Target 50kHz
                # We can print both or just raw freq.
                results['freq_measured'] = freq_measured
                print(f"  [CFO Est] Measured Freq: {freq_measured:.1f} Hz")
        
        # Silence starts at 177 + 512 + 50 = 739
        silence_start = start_idx + 739
        silence_end = silence_start + 256
        
        if silence_end < len(rx_samples):
            silence_seg = rx_samples[silence_start:silence_end]
            noise_power = np.mean(np.abs(silence_seg)**2) + 1e-9
            results['noise_power'] = noise_power
            results['snr'] = 10 * np.log10(results['signal_power'] / results['noise_power'])
            
        # Channel Impulse Response (CIR)
        # H = FFT(RX_ZC) / FFT(TX_ZC) ??
        # Or just use the correlation peak shape itself as coarse CIR
        # Let's extract window around peak
        win = 20
        if peak_idx > win and peak_idx < len(corr) - win:
             results['channel_response'] = np.abs(corr[peak_idx-win : peak_idx+win])
             
    else:
        results['signal_power'] = max_amp**2 # Just raw energy if no lock

    return results, rx_samples

def auto_gain_control(sdr, channel=0):
    """Adjust Gain to target Peak ~1000 (50% ADC)."""
    current_gain = 30 # Start guess
    
    print("Starting Auto-Gain...")
    for i in range(10): # Max 10 steps
        # Set Gain
        if channel == 0: sdr.rx_hardwaregain_chan0 = int(current_gain)
        else: sdr.rx_hardwaregain_chan1 = int(current_gain)
        
        # Receive
        samples = sdr.rx()
        peak = np.max(np.abs(samples))
        
        print(f"  [AGC] Gain: {current_gain}dB -> Peak: {peak:.0f}")
        
        # Logic
        if peak > 1900:
            current_gain -= 10 # Fast backoff
        elif peak > 1500:
            current_gain -= 3
        elif peak < 200:
            current_gain += 10 # Boost
        elif peak < 800:
            current_gain += 5
        else:
            # Good range (800 - 1500)
            return int(current_gain)
            
        # Clamp
        current_gain = max(0, min(70, current_gain))
        
    return int(current_gain)

def main():
    parser = argparse.ArgumentParser(description='SDR Probe Receiver')
    parser.add_argument('--ip', default='ip:192.168.1.10', help='Local SDR IP')
    parser.add_argument('--channel', type=int, default=1, help='RX Channel (0 or 1)') # Default to Div antenna
    parser.add_argument('--loopback', action='store_true', help='Enable Local TX Loopback')
    args = parser.parse_args()

    print(f"Connecting to {args.ip}...")
    try:
        sdr = adi.Pluto(uri=args.ip)
    except:
        print("Failed to connect.")
        return

    sdr.sample_rate = int(FS)
    sdr.rx_lo = int(FC)
    sdr.rx_rf_bandwidth = int(BW)

    if args.loopback:
        print("Enabling Local TX Loopback...")
        sdr.tx_enabled_channels = [0]
        sdr.tx_lo = int(FC)
        sdr.tx_cyclic_buffer = True
        sdr.tx_hardwaregain_chan0 = -30
        
        # Generate Frame
        zc = generate_zadoff_chu(127, 25)
        t = np.arange(512)/FS
        tone = 0.5 * np.exp(1j*2*np.pi*50e3*t)
        guard = np.zeros(50, dtype=complex)
        silence = np.zeros(256, dtype=complex)
        frame = np.concatenate([zc, guard, tone, guard, silence])
        buffer = np.tile(frame, 10).astype(np.complex64) * (2**14) # Scale for int16/complex? ADI handles float 0-1 usually but let's be safe
        
        # ADI Python scaling: it expects complex float -1 to 1? Or unscaled?
        # Usually unscaled -1 to 1 is fine if strictly typed.
        buffer = np.tile(frame, 10) * 0.5
        sdr.tx([buffer]) # Wrap in list for channel 0
    
    sdr.rx_enabled_channels = [0]
    sdr.rx_buffer_size = 1024*32 # Enough to capture frames
    
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"

    # 1. Run AGC
    best_gain = auto_gain_control(sdr, args.channel)
    print(f"Optimal Gain Found: {best_gain}dB")
    
    # 2. Capture Data for Analysis
    print("Capturing Probe Data...")
    samples = sdr.rx()
    # If single channel, sdr.rx() returns the array directly
    if len(sdr.rx_enabled_channels) == 1:
        rx_data = samples
    else:
        if args.channel == 0:
            rx_data = samples[0]
        else:
            rx_data = samples[1]
        
    # 3. Analyze
    stats, raw = estimate_channel(rx_data, best_gain)
    
    print("="*30)
    print("CHANNEL ESTIMATION REPORT")
    print("="*30)
    print(f"Locked: {stats['locked']}")
    print(f"Saturation: {stats['saturation']}")
    print(f"Signal Power: {stats['signal_power']:.1f}")
    print(f"Noise Power:  {stats['noise_power']:.1f}")
    print(f"SNR Est:      {stats['snr']:.1f} dB")
    
    # 4. Plotting
    plt.figure(figsize=(12, 10))
    
    # Time Domain
    plt.subplot(3, 1, 1)
    plt.plot(np.abs(raw[:2000]))
    plt.title(f"Time Domain (CH{args.channel}, Gain {best_gain}dB)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Channel Response (Correlation Peak)
    if stats['channel_response'] is not None:
        plt.subplot(3, 1, 2)
        plt.plot(stats['channel_response'], 'r-o')
        plt.title("Estimated Channel Impulse Response (Multipath)")
        plt.grid(True)
        
    # Spectrogram / Spectrum
    plt.subplot(3, 1, 3)
    plt.psd(raw, NFFT=1024, Fs=FS/1e6)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (MHz)")
    
    plt.tight_layout()
    plt.savefig('probe_results.png')
    print("Saved plot to 'probe_results.png'")

if __name__ == "__main__":
    main()
