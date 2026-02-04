import numpy as np
import adi
import matplotlib.pyplot as plt

# --- DIAGNOSTIC V3 ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 1e6 
N, CP = 64, 16 

def deep_debug_scan():
    sdr = adi.Pluto(uri=IP)
    sdr.sample_rate = int(FS)
    sdr.tx_lo, sdr.rx_lo = int(FC), int(FC)
    sdr.rx_buffer_size = 2**18
    
    # Capture current state
    rx = sdr.rx()
    
    # 1. FFT Surface Search
    # We look for the "spectral footprint" of your OFDM symbols
    psd = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(rx.reshape(-1, N), axis=1)))**2)
    
    # 2. Plot Detailed Subcarrier Phase
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(psd, aspect='auto', interpolation='none')
    plt.title("Spectral Surface (Time vs Frequency)")
    plt.colorbar(label='dB')

    # 3. Decision-Directed Phase History
    # This shows if the "Spin" is constant (easy to fix) or jittery (hard)
    plt.subplot(1, 2, 2)
    plt.plot(np.angle(rx[10000:10500]), 'r.', markersize=1)
    plt.title("Raw Phase Noise Pattern")
    
    plt.tight_layout()
    plt.savefig("deep_internal_state.png")
    print("Internal states saved to deep_internal_state.png")

if __name__ == "__main__":
    deep_debug_scan()