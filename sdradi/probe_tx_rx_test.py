import argparse
import adi
import numpy as np
import time

# Configuration
SDR_IP = "ip:192.168.3.2"
FC = 2300e6
FS = 2e6
BW = 1e6

def generate_probe_frame(fs):
    t = np.arange(512)/fs
    tone_freq = 50e3
    tone = 0.8 * np.exp(1j*2*np.pi*tone_freq*t)
    # Tile to ensure buffer is large enough
    return np.tile(tone * (2**14), 100)

def main():
    print(f"Connecting to {SDR_IP}...")
    sdr = adi.Pluto(uri=SDR_IP)
    sdr.sample_rate = int(FS)
    sdr.tx_lo = int(FC)
    sdr.tx_rf_bandwidth = int(BW)
    
    print("Configuring TX and RX...")
    sdr.tx_enabled_channels = [0]
    sdr.tx_hardwaregain_chan0 = -5
    sdr.tx_hardwaregain_chan1 = -5
    
    # HYPOTHESIS TEST: Enable RX but don't read
    sdr.rx_enabled_channels = [0] 
    
    # Generate Signal
    tx_data = generate_probe_frame(FS)
    
    print(f"Transmitting with RX Enabled... (Ctrl+C to stop)")
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_data)
    
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
