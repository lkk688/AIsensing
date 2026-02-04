import numpy as np
import adi
import matplotlib.pyplot as plt

# --- HARDWARE & NARROW-BAND PARAMS ---
URI, FC, FS = "usb:1.37.5", 2300e6, 1e6 
N, CP = 32, 32 

def run_ultra_search_rx():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI); sdr.rx_buffer_size = 2**20
        print("📡 Initiating Ultra-Wideband Frequency Search...")
        rx_raw = sdr.rx() / 2**14
        
        # 1. GENERATE ZC-REF
        n = np.arange(N); zc = np.exp(-1j * np.pi * 17 * n * (n + 1) / N)
        zc_ref = np.fft.ifft(zc) * np.sqrt(N)
        
        # 2. FREQUENCY SEARCH (±15 kHz)
        freq_offsets = np.arange(-15000, 15000, 500)
        best_peak, best_f, best_idx = 0, 0, 0
        
        for f_off in freq_offsets:
            # Counter-rotate raw signal to test this frequency hypothesis
            rx_test = rx_raw * np.exp(-1j * 2 * np.pi * f_off / FS * np.arange(len(rx_raw)))
            corr = np.abs(np.correlate(rx_test[10000:30000], zc_ref, mode='valid'))
            peak = np.max(corr)
            if peak > best_peak:
                best_peak, best_f, best_idx = peak, f_off, np.argmax(corr) + 10000
        
        print(f"🎯 SEARCH COMPLETE. Best Lock: {best_f} Hz at index {best_idx}")
        
        # 3. APPLY MASTER LOCK
        rx_locked = rx_raw * np.exp(-1j * 2 * np.pi * best_f / FS * np.arange(len(rx_raw)))
        
        # 4. DECODE GOLDEN CORE
        data_start = best_idx + (32 * N) + CP # Skip expanded header
        const, diff_angles = [], []
        last_syms = None
        DATA_SC = np.array([-4, -3, 3, 4])

        for i in range(100):
            idx = data_start + i*(N+CP)
            if idx+N+CP > len(rx_locked): break
            sym = rx_locked[idx+CP : idx+CP+N]
            Y = np.fft.fftshift(np.fft.fft(sym))
            
            curr_data = Y[(DATA_SC + N//2) % N]
            if last_syms is not None:
                diff = curr_data * np.conj(last_syms)
                diff_angles.extend(np.degrees(np.angle(diff)))
                const.extend(curr_data)
            last_syms = curr_data

        # --- FINAL GLASS BOX AUDIT ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(np.real(const), np.imag(const), s=8, color='blue', alpha=0.5)
        axes[0].set_title(f"1. Constellation (CFO: {best_f} Hz)"); axes[0].grid(True)
        
        axes[1].hist(diff_angles, bins=180); axes[1].set_title("2. DQPSK Angle Peaks")
        plt.tight_layout(); plt.savefig("ultra_search_final.png")
        print("✅ Report saved to ultra_search_final.png.")

    except Exception as e: print(f"Error: {e}")
    finally: 
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_ultra_search_rx()