import argparse
import adi
import numpy as np
import time
import sys
import torch # <--- SUSPECT

# Configuration
SDR_IP = "ip:192.168.3.2"
FC = 2300e6
FS = 2e6
BW = 1e6

def generate_zadoff_chu(seq_len=127, root=25):
    n = np.arange(seq_len)
    cf = seq_len % 2
    zadoff_chu = np.exp(-1j * np.pi * root * n * (n + cf + 2 * 0) / seq_len)
    return zadoff_chu

def generate_probe_frame(fs):
    zc = generate_zadoff_chu(127, 25)
    t = np.arange(512)/fs
    tone_freq = 50e3
    tone = 0.8 * np.exp(1j*2*np.pi*tone_freq*t)
    silence = np.zeros(256, dtype=complex)
    guard = np.zeros(50, dtype=complex)
    frame = np.concatenate([zc, guard, tone, guard, silence])
    return np.tile(frame, 10)

def main():
    print(f"Connecting to {SDR_IP} (Torch Version: {torch.__version__})...")
    try:
        sdr = adi.Pluto(uri=SDR_IP)
    except Exception as e:
        print(f"Error: {e}")
        return

    sdr.sample_rate = int(FS)
    sdr.tx_lo = int(FC)
    sdr.tx_rf_bandwidth = int(BW)
    
    print(f"Configuring TX...")
    sdr.tx_enabled_channels = [0]
    sdr.tx_hardwaregain_chan0 = -5
    sdr.tx_hardwaregain_chan1 = -5

    frame = generate_probe_frame(FS)
    tx_data = frame * (2**14) 
    
    print(f"Transmitting Probe... (Ctrl+C to stop)")
    sdr.tx_cyclic_buffer = True 
    sdr.tx(tx_data)
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
