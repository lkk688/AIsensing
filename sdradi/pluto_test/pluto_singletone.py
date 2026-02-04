import numpy as np
import adi
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 3e6
N, CP = 64, 16
TEST_BIN = 10 # We will put a signal ONLY in subcarrier +10

def run_mapping_test():
    try:
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate, sdr.tx_lo, sdr.rx_lo = int(FS), int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -10, 30

        # Create 1 OFDM symbol with only ONE active bin
        X = np.zeros(N, dtype=complex)
        X[(TEST_BIN + N//2) % N] = 1.0 + 1j # Signal at +10
        
        x_time = np.fft.ifft(np.fft.ifftshift(X))
        tx_sig = np.concatenate([x_time[-CP:], x_time])
        
        sdr.tx_cyclic_buffer = True
        sdr.tx((tx_sig / np.max(np.abs(tx_sig)) * 0.5 * 2**14).astype(np.complex64))
        
        plt.pause(0.5)
        rx = sdr.rx()
        sdr.tx_destroy_buffer()

        # FFT and look at the spectrum
        Y = np.fft.fftshift(np.fft.fft(rx[100:100+N])) # Take a chunk
        
        plt.figure(figsize=(10, 4))
        plt.plot(np.abs(Y))
        plt.title(f"Single Tone Test (Expected Bin: {TEST_BIN + N//2})")
        plt.xlabel("FFT Bin Index")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_mapping_test()

#Single Tone peak at Bin 42, passed