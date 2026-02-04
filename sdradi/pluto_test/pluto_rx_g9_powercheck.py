import numpy as np
import adi
import matplotlib.pyplot as plt

def run_power_audit():
    sdr = None
    try:
        sdr = adi.Pluto(uri="usb:1.37.5")
        sdr.rx_buffer_size = 2**20
        print("📡 Auditing RX power level...")
        rx = sdr.rx()
        
        # Calculate Magnitude
        mag = np.abs(rx)
        pwr_db = 10 * np.log10(np.mean(mag**2) + 1e-12)
        
        plt.figure(figsize=(10, 4))
        plt.plot(mag[:5000])
        plt.title(f"Time Domain Power Audit (Avg Power: {pwr_db:.2f} dB)")
        plt.xlabel("Samples"); plt.ylabel("Linear Magnitude")
        plt.savefig("power_audit.png")
        
        if np.max(mag) < 10:
            print("⚠️ WARNING: Extremely low signal level. Is the TX running?")
        else:
            print(f"✅ Signal Detected. Max Mag: {np.max(mag):.2f}")
            
    except Exception as e: print(f"Error: {e}")
    finally: 
        if sdr: sdr.rx_destroy_buffer()

run_power_audit()