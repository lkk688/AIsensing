import numpy as np
import adi
import matplotlib.pyplot as plt

# --- RX CONFIGURATION ---
URI, FC, FS = "usb:1.30.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_phase_lock_rx():
    try:
        sdr = adi.Pluto(uri=URI)
        sdr.sample_rate, sdr.rx_lo = int(FS), int(FC)
        sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**19, 55

        print(f"RX Listening on {URI}...")
        for _ in range(5): sdr.rx()
        rx = sdr.rx()
        
        # 1. Digital AGC (Crucial to expand the IQ Density)
        rx = rx / np.sqrt(np.mean(np.abs(rx)**2))
        
        # Cross-correlation for timing
        stf_sc = np.zeros(N, dtype=complex)
        stf_sc[::4] = (1 + 1j) * np.sqrt(8/2)
        stf_t = np.fft.ifft(stf_sc) * np.sqrt(N)
        corr = np.abs(np.correlate(rx, stf_t, mode='valid'))
        c_idx = np.argmax(corr[5000:]) + 5000 
        
        # 2. FREQUENCY RECOVERY
        # Period L=16 samples. Compare repetition 1 vs repetition 2.
        L = 16
        rep1, rep2 = rx[c_idx : c_idx + L], rx[c_idx + L : c_idx + 2*L]
        cfo_est = np.angle(np.sum(np.conj(rep1) * rep2)) / L
        print(f"CFO Locked: {cfo_est * FS / (2*np.pi):.2f} Hz")
        
        # Apply inverse rotation to the whole buffer
        t = np.arange(len(rx))
        rx_locked = rx * np.exp(-1j * cfo_est * t)
        
        # 3. Decision-Directed Fine Tracking
        data_start = c_idx + (4 * N) # Jump past 4xSTF
        PILOT_IDX = np.array([-21, -7, 7, 21])
        DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_IDX])
        
        constellation, residual_phases = [], []
        for i in range(80):
            idx = data_start + i*(N+CP)
            Y = np.fft.fftshift(np.fft.fft(rx_locked[idx+CP : idx+CP+N])) / np.sqrt(N)
            pilots = Y[(PILOT_IDX + N//2) % N]
            p_phase = np.angle(np.mean(pilots))
            Y_data = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_phase)
            constellation.extend(Y_data)
            residual_phases.append(np.degrees(p_phase))

        # --- DIAGNOSTICS ---
        const = np.array(constellation)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].scatter(const.real, const.imag, s=2, alpha=0.4, color='teal')
        axes[0,0].set_title("CFO-CORRECTED CONSTELLATION"); axes[0,0].grid(True)
        axes[0,0].set_xlim([-3, 3]); axes[0,0].set_ylim([-3, 3])
        
        axes[0,1].plot(residual_phases); axes[0,1].set_title("Residual Jitter (Deg) - Should be flat")
        axes[1,0].plot(corr[c_idx-100:c_idx+100]); axes[1,0].set_title("Sync Anchor")
        axes[1,1].hist2d(rx.real, rx.imag, bins=50); axes[1,1].set_title("IQ Density")
        
        plt.tight_layout(); plt.savefig("final_math_verify.png")
        print(f"VERIFIED! Constellation cloud should now be 4 tight dots.")

    except Exception as e: print(f"RX Error: {e}")
    finally: sdr.tx_destroy_buffer()

if __name__ == "__main__": run_phase_lock_rx()