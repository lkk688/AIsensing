import adi
import numpy as np
import matplotlib.pyplot as plt

def test_pluto(ip="ip:192.168.2.1"):
    print(f"Connecting to PlutoSDR at {ip}...")
    try:
        sdr = adi.Pluto(uri=ip)
        print("Successfully connected!")
        print(f"Device URI: {sdr.uri}")
        
        # Configure
        sdr.sample_rate = int(1e6)
        sdr.rx_lo = int(2.4e9)
        sdr.tx_lo = int(2.4e9)
        sdr.rx_rf_bandwidth = int(1e6)
        sdr.tx_rf_bandwidth = int(1e6)
        sdr.rx_buffer_size = 1024
        
        print(f"LO Frequency: {sdr.rx_lo/1e9} GHz")
        print(f"Sample Rate: {sdr.sample_rate/1e6} MSPS")
        
        # Read RX
        print("Reading 1024 samples...")
        rx = sdr.rx()
        print(f"Received {len(rx)} samples.")
        print(f"Min: {np.min(rx)}, Max: {np.max(rx)}, Mean: {np.mean(rx)}")
        
        if len(rx) > 0:
            print("SUCCESS: PlutoSDR is working.")
        else:
            print("FAILURE: No data received.")
            
    except Exception as e:
        print(f"FAILURE: Could not connect to PlutoSDR: {e}")
        # Hint about drivers or USB networking
        print("\nTroubleshooting Tips:")
        print("1. Ensure device is pingable (ping 192.168.2.1).")
        print("2. Unplug and replug the USB cable.")
        print("3. Check dmesg for USB enumeration.")

if __name__ == "__main__":
    test_pluto()
