import numpy as np
import adi
import matplotlib.pyplot as plt

URI, FC, FS = "usb:1.37.5", 2300e6, 1e6 
N, CP = 32, 32

def run_peak_validated_rx():
    sdr = adi.Pluto(uri=URI); sdr.rx_buffer_size = 2**20
    rx_raw = sdr.rx() / 2**14
    
    # 1. PEAK-VALIDATED SYNC
    zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 17 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
    corr = np.abs(np.correlate(rx_raw, zc_ref, mode='valid'))
    
    # Filter correlation to find the "true" mountain
    smooth_corr = np.convolve(corr, np.ones(10)/10, mode='same')
    c_idx = np.argmax(smooth_corr[10000:]) + 10000 
    
    # 2. CFO & DECODE
    f_lock = np.angle(np.sum(np.conj(rx_raw[c_idx:c_idx+N]) * rx_raw[c_idx+N:c_idx+2*N])) / N
    rx_p = rx_raw * np.exp(-1j * f_lock * np.arange(len(rx_raw)))
    
    # Skip Header & Preamble
    data_start = c_idx + (20 * N) + (2 * (N+CP)) + CP
    const, diff_angles = [], []
    last_syms = None

    for i in range(100):
        idx = data_start + i*(N+CP)
        # Large CP allows for +/- 16 samples of timing error
        sym = rx_p[idx : idx+N] 
        Y = np.fft.fftshift(np.fft.fft(sym))
        
        DATA_SC = np.array([-4, -3, 3, 4])
        curr_data = Y[(DATA_SC + N//2) % N]
        if last_syms is not None:
            diff = curr_data * np.conj(last_syms)
            diff_angles.extend(np.degrees(np.angle(diff)))
            const.extend(curr_data)
        last_syms = curr_data

    # --- VALIDATION FIGURE ---
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1); plt.hist(diff_angles, bins=180); plt.title("Angle Distribution (Must have 4 Peaks)")
    plt.subplot(2,1,2); plt.scatter(np.real(const), np.imag(const), s=10); plt.title("Constellation")
    plt.tight_layout(); plt.savefig("peak_validation_report.png")
    print(f"🏁 Validation complete. CFO: {f_lock*FS/(2*np.pi):.2f} Hz.")

if __name__ == "__main__": run_peak_validated_rx()