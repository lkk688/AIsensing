import adi
import time
import numpy as np

try:
    print("Connecting to Pluto...")
    sdr = adi.Pluto(uri="ip:192.168.2.2")
    print("Connected.")
    sdr.sample_rate = int(3e6)
    sdr.rx_lo = int(2.3e9)
    sdr.rx_rf_bandwidth = int(3e6)
    sdr.rx_buffer_size = 131072
    print("Configured.")
    for i in range(5):
        data = sdr.rx()
        print(f"Captured {len(data)} samples. Peak abs={np.max(np.abs(data)):.4f}")
    print("Done.")
except Exception as e:
    print(f"Error: {e}")
