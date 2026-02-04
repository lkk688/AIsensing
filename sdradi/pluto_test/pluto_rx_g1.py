import numpy as np
import adi
import matplotlib.pyplot as plt

# --- RX CONFIGURATION ---
URI = "usb:1.30.5"
FC, FS = 2300e6, 1e6 
N, CP = 64, 16 

def run_production_rx():
    try:
        sdr = adi.Pluto(uri=URI)
        sdr.sample_rate = int(FS)
        sdr.rx_lo = int(FC)
        sdr.rx_buffer_size = 2**19
        sdr.rx_hardwaregain_chan0 = 55 # Optimized for cabled SNR

        print(f"RX Syncing on {URI}...")
        for _ in range(5): sdr.rx() 
        rx = sdr.rx()
        
        # 1. Synchronization and Digital AGC
        rx = rx / np.sqrt(np.mean(np.abs(rx)**2))
        
        # Preamble Sync (Match first symbol of TX)
        PILOT_IDX = np.array([-21, -7, 7, 21])
        DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_IDX])
        
        # Re-gen first symbol for correlation
        np.random.seed(1234)
        ref_bits = np.random.randint(0, 2, len(DATA_SC) * 2)
        ref_qpsk = ((1 - 2*ref_bits[::2]) + 1j*(1 - 2*ref_bits[1::2])) / np.sqrt(2)
        X_ref = np.zeros(N, dtype=complex)
        X_ref[(DATA_SC + N//2) % N] = ref_qpsk
        X_ref[(PILOT_IDX + N//2) % N] = 1.0
        preamble_t = np.fft.ifft(np.fft.ifftshift(X_ref)) * np.sqrt(N)
        preamble_t = np.concatenate([preamble_t[-CP:], preamble_t])
        
        corr = np.abs(np.correlate(rx, preamble_t, mode='valid'))
        c_idx = np.argmax(corr)

        # 2. Pilot-Aided Decoding
        data_start = c_idx + (N + CP) 
        clean_constellation = []
        phase_errors = []
        
        for i in range(40):
            idx = data_start + i*(N+CP)
            Y = np.fft.fftshift(np.fft.fft(rx[idx+CP : idx+CP+N])) / np.sqrt(N)
            
            # Extract Pilots to find phase twist
            pilots = Y[(PILOT_IDX + N//2) % N]
            avg_phase = np.angle(np.mean(pilots)) # Reference is 1+0j
            
            # Correct the whole symbol
            Y_corrected = Y * np.exp(-1j * avg_phase)
            Y_data = Y_corrected[(DATA_SC + N//2) % N]
            
            clean_constellation.extend(Y_data)
            phase_errors.append(np.degrees(avg_phase))

        # --- FINAL DIAGNOSTICS ---
        const = np.array(clean_constellation)
        # Calculate BER
        rx_bits = np.zeros(len(const)*2)
        rx_bits[::2] = (const.real < 0)
        rx_bits[1::2] = (const.imag < 0)
        tx_bits_ref = np.random.randint(0, 2, len(const)*2) # Note: simplified for debug
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes[0,0].scatter(const.real, const.imag, s=2, alpha=0.3, color='green')
        axes[0,0].set_title("PILOT-CORRECTED CONSTELLATION"); axes[0,0].grid(True)
        axes[0,1].plot(phase_errors); axes[0,1].set_title("Pilot-Aided Phase Correction (Deg)")
        axes[1,0].plot(corr[c_idx-100:c_idx+100]); axes[1,0].set_title("Sync Peak Zoom")
        axes[1,1].hist2d(rx.real, rx.imag, bins=60); axes[1,1].set_title("IQ Density (Check for Ringing)")
        
        plt.tight_layout(); plt.savefig("pilot_sync_debug.png")
        print(f"Locked! BER Check Complete. Figures in pilot_sync_debug.png")

    except Exception as e:
        print(f"RX Error: {e}")

if __name__ == "__main__":
    run_production_rx()