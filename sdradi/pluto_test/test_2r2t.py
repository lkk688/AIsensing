import adi
import numpy as np
import time
import sys

# Configuration
SDR_IP = "ip:192.168.3.2" # Local Device
FC = 2300e6
FS = 3e6

def generate_tone(fs, freq=100e3):
    N = 1024*16
    t = np.arange(N)/fs
    return 0.5 * np.exp(1j*2*np.pi*freq*t)

print(f"Connecting to {SDR_IP} using adi.ad9361...")
try:
    sdr = adi.ad9361(uri=SDR_IP)
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)

sdr.sample_rate = int(FS)
sdr.tx_lo = int(FC)
sdr.rx_lo = int(FC)
sdr.tx_rf_bandwidth = int(FS)
sdr.rx_rf_bandwidth = int(FS)
sdr.tx_cyclic_buffer = True

# Matrix Test
tone = generate_tone(FS)
channels = [0, 1]

print("\n=== 2R2T LOOPBACK MATRIX ===")
print(f"{'TX CH':<6} -> {'RX CH':<6} | {'PEAK':>8} | {'dBFS':>8} | {'STATUS':<15}")
print("-" * 55)

for tx_ch in channels:
    # Setup TX
    try:
        if tx_ch == 0:
            sdr.tx_hardwaregain_chan0 = -10
            sdr.tx_enabled_channels = [0]
        else:
            sdr.tx_hardwaregain_chan1 = -10
            sdr.tx_enabled_channels = [1]
            
        sdr.tx(tone)
        time.sleep(0.2)
    except Exception as e:
        print(f" TX{tx_ch} Error: {e}")
        continue

    # Read RX
    for rx_ch in channels:
        try:
            sdr.rx_enabled_channels = [rx_ch]
            sdr.rx_buffer_size = 1024*16
            
            # Gain
            if rx_ch == 0: sdr.rx_hardwaregain_chan0 = 30
            else: sdr.rx_hardwaregain_chan1 = 30
            
            # Flush
            for _ in range(2): sdr.rx()
            
            data = sdr.rx()
            peak = np.max(np.abs(data))
            dbfs = 20*np.log10(peak+1)
            
            status = "SIGNAL" if peak > 500 else "NOISE"
            if peak > 1500: status = "STRONG!"
            
            print(f" {tx_ch:<6} -> {rx_ch:<6} | {peak:8.0f} | {dbfs:8.1f} | {status:<15}")
            
        except Exception as e:
            print(f" {tx_ch:<6} -> {rx_ch:<6} |   ERROR  |          | {e}")

    # Stop TX
    try: sdr.tx_destroy_buffer()
    except: pass
