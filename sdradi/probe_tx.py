import argparse
import adi
import numpy as np
import time
import sys

# Configuration
SDR_IP = "ip:192.168.2.2" # Default Remote IP (if running on remote)
FC = 2300e6
FS = 2e6 # 2 MHz Sample Rate
BW = 1e6 # 1 MHz Bandwidth

def generate_zadoff_chu(seq_len=127, root=25):
    """Generate Zadoff-Chu sequence for synchronization."""
    n = np.arange(seq_len)
    cf = seq_len % 2
    zadoff_chu = np.exp(-1j * np.pi * root * n * (n + cf + 2 * 0) / seq_len)
    return zadoff_chu

def generate_probe_frame(fs):
    """
    Frame Structure:
    [ZC Sync (127)] + [Guard (50)] + [Tone (512)] + [Guard (50)] + [Silence (256)]
    """
    # 1. Sync Preamble
    zc = generate_zadoff_chu(127, 25)
    
    # 2. Tone (Low Frequency to see shape clearly)
    t = np.arange(512)/fs
    tone_freq = 50e3 # 50 kHz
    tone = 0.8 * np.exp(1j*2*np.pi*tone_freq*t) # 0.8 Amp (-2dBFS)
    
    # 3. Silence (Noise Floor)
    silence = np.zeros(256, dtype=complex)
    
    # Assembly
    guard = np.zeros(50, dtype=complex)
    frame = np.concatenate([zc, guard, tone, guard, silence])
    
    # Repeat Frame 10 times to buffer
    buffer = np.tile(frame, 10)
    return buffer

def main():
    parser = argparse.ArgumentParser(description='SDR Probe Transmitter')
    parser.add_argument('--ip', default='ip:192.168.1.10', help='SDR IP Address')
    parser.add_argument('--gain', type=int, default=-10, help='TX Gain (dB)')
    args = parser.parse_args()

    print(f"Connecting to {args.ip}...")
    try:
        sdr = adi.Pluto(uri=args.ip)
    except Exception as e:
        print(f"Error: {e}")
        return

    sdr.sample_rate = int(FS)
    sdr.tx_lo = int(FC)
    sdr.tx_rf_bandwidth = int(BW)
    
    # Configure TX2 (Channel 1) - Default for Remote
    print(f"Configuring TX at {args.gain}dB...")
    sdr.tx_enabled_channels = [0] # Single Channel
    sdr.tx_hardwaregain_chan0 = args.gain
    sdr.tx_hardwaregain_chan1 = args.gain

    # Generate Signal
    frame = generate_probe_frame(FS)
    # Scale to 2^14
    tx_signal = frame * (2**14) 
    
    # Enable single channel
    tx_data = tx_signal # Pass 1D array for single channel

    print(f"Transmitting Probe (Size {len(tx_signal)} x 2)... (Ctrl+C to stop)")
    
    try:
        sdr.tx_cyclic_buffer = True # Continuous Transmit
        sdr.tx(tx_data)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        sdr.tx_destroy_buffer()

if __name__ == "__main__":
    main()
