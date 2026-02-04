import adi
import time

print("Attempting to connect to ip:192.168.2.2")
try:
    sdr = adi.Pluto("ip:192.168.2.2")
    print("Connected!")
    print("Getting samples...")
    data = sdr.rx()
    print(f"Got {len(data)} samples")
except Exception as e:
    print(f"Error: {e}")
