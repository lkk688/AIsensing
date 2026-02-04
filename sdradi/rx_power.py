import adi
import numpy as np
import time

IP = "ip:192.168.1.10"
FC = 2412e6 # WiFi Ch 1
FS = 1e6

print(f"Connecting to {IP}...")
sdr = adi.ad9361(uri=IP)

sdr.sample_rate = int(FS)
sdr.rx_lo = int(FC)
sdr.rx_rf_bandwidth = int(FS)
sdr.rx_enabled_channels = [0, 1]
sdr.rx_buffer_size = 1024*16
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = 50
sdr.rx_hardwaregain_chan1 = 70

print("Monitoring Power (Peak/Mean)...")
print(f"Ch0 (Peak/Mean) | Ch1 (Peak/Mean)")
print("-" * 30)

while True:
    try:
        data = sdr.rx()
        if len(data[0]) == 0: continue
            
        d0 = data[0]
        d1 = data[1]
        
        p0_peak = np.max(np.abs(d0))
        p0_mean = np.mean(np.abs(d0))
        
        p1_peak = np.max(np.abs(d1))
        p1_mean = np.mean(np.abs(d1))
        
        print(f"{p0_peak:6.0f} / {p0_mean:4.1f} | {p1_peak:6.0f} / {p1_mean:4.1f}")
        time.sleep(0.5)
    except Exception as e:
        print(f"Read Error: {e}")
        time.sleep(0.1)
