import numpy as np
import adi
import matplotlib.pyplot as plt

# --- RX CONFIG ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
EXPECTED_HZ = 100e3

def run_cfo_discriminator():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI)
        sdr.sample_rate, sdr.rx_lo = int(FS), int(FC)
        sdr.rx_buffer_size = 2**16
        sdr.rx_hardwaregain_chan0 = 50

        print(f"Measuring frequency error from {URI}...")
        rx = sdr.rx()
        
        # 1. Calculate Power Spectral Density (PSD)
        fft_size = len(rx)
        psd = np.abs(np.fft.fftshift(np.fft.fft(rx)))**2
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/FS))
        
        # 2. Find the physical peak
        peak_idx = np.argmax(psd)
        measured_hz = freqs[peak_idx]
        cfo_error = measured_hz - EXPECTED_HZ
        
        print(f"\n--- CALIBRATION RESULTS ---")
        print(f"Expected Tone: {EXPECTED_HZ/1e3:.2f} kHz")
        print(f"Measured Tone: {measured_hz/1e3:.2f} kHz")
        print(f"REAL HARDWARE CFO: {cfo_error:.2f} Hz") # This is your "Magic Number"
        
        # 3. Plot for visual confirmation
        plt.figure(figsize=(10, 5))
        plt.plot(freqs/1e3, 10*np.log10(psd))
        plt.axvline(EXPECTED_HZ/1e3, color='r', linestyle='--', label='Expected')
        plt.axvline(measured_hz/1e3, color='g', label='Measured')
        plt.title(f"Hardware CFO: {cfo_error:.2f} Hz")
        plt.xlabel("Frequency (kHz)"); plt.ylabel("dB"); plt.legend(); plt.grid(True)
        plt.savefig("cfo_calibration.png")
        print("Diagnostic saved to cfo_calibration.png")

    except Exception as e: print(f"RX Error: {e}")
    finally:
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_cfo_discriminator()