import numpy as np
import adi
import matplotlib.pyplot as plt

# --- RX CONFIGURATION ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_audit_rx():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI)
        sdr.sample_rate, sdr.rx_lo = int(FS), int(FC)
        sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**19, 45 

        print(f"RX Capturing Diagnostic Audit...")
        for _ in range(10): sdr.rx() # Flush DMA
        rx = sdr.rx()
        rx = rx / (np.sqrt(np.mean(np.abs(rx)**2)) + 1e-12)
        
        # 1. SYNC & MULTI-STAGE FREQUENCY LOCK
        stf_sc = np.zeros(N, dtype=complex); stf_sc[::4] = (1 + 1j) * np.sqrt(4)
        corr = np.abs(np.correlate(rx, np.fft.ifft(stf_sc), mode='valid'))
        c_idx = np.argmax(corr[10000:]) + 10000 
        
        L = 16
        phases = [np.angle(np.sum(np.conj(rx[c_idx+i*L : c_idx+(i+1)*L]) * rx[c_idx+(i+1)*L : c_idx+(i+2)*L])) for i in range(8)]
        coarse_cfo = np.median(phases) / L 
        rx_locked = rx * np.exp(-1j * coarse_cfo * np.arange(len(rx)))
        
        # 2. LTF-AIDED CHANNEL AUDIT
        ltf_start = c_idx + (10 * 16) + CP
        ltf_raw = np.fft.fftshift(np.fft.fft(rx_locked[ltf_start:ltf_start+N]))
        
        # 3. DATA EXTRACTION WITH PER-SYMBOL TRACKING
        PILOT_IDX = np.array([-21, -7, 7, 21])
        DATA_SC = np.array([sc for sc in range(-26, 27) if sc not in range(-2, 3) and sc not in PILOT_IDX])
        
        const, tracking_log, snr_per_sc = [], [], []
        data_start = ltf_start + (2 * N) + CP
        
        # Equalization stage (avoid DC null bins)
        np.random.seed(42)
        ltf_ref = np.fft.ifftshift(np.random.choice([1, -1], N).astype(complex))
        H_est = np.fft.fft(rx_locked[ltf_start:ltf_start+N]) / (ltf_ref + 1e-12)

        for i in range(150):
            idx = data_start + i*(N+CP)
            if idx + N + CP > len(rx_locked): break
            Y = np.fft.fftshift(np.fft.fft(rx_locked[idx+CP : idx+CP+N])) / np.sqrt(N)
            Y_eq = Y / (np.fft.fftshift(H_est) + 1e-12)
            
            # Pilot tracking to handle residual drift
            p_phase = np.angle(np.mean(Y_eq[(PILOT_IDX + N//2) % N]))
            symbols = Y_eq[(DATA_SC + N//2) % N] * np.exp(-1j * p_phase)
            const.extend(symbols)
            tracking_log.append(np.degrees(p_phase))

        # --- EXTENDED DEBUG AUDIT ---
        const = np.array(const)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Constellation Check
        axes[0,0].scatter(const.real, const.imag, s=2, alpha=0.3, color='navy')
        axes[0,0].set_title("1. AUDITED CONSTELLATION"); axes[0,0].grid(True)
        axes[0,0].set_xlim([-4, 4]); axes[0,0].set_ylim([-4, 4])
        
        # 2. Spectral Audit (Magnitude)
        axes[0,1].bar(np.arange(-32, 32), np.abs(ltf_raw))
        axes[0,1].set_title("2. Subcarrier Power Map (Look for Clean DC Hole)"); axes[0,1].grid(True)
        
        # 3. Time-Domain Phase Lock Check
        axes[1,0].plot(np.unwrap(np.angle(rx_locked[c_idx:c_idx+1000])))
        axes[1,0].set_title("3. Phase Continuity (MUST BE A FLAT LINE)")
        
        # 4. Residual Jitter Stability
        axes[1,1].plot(tracking_log)
        axes[1,1].set_title("4. Symbol-by-Symbol Phase Track (Should be Horizontal)")
        
        # 5. SNR Heatmap
        axes[2,0].plot(DATA_SC, 20*np.log10(np.abs(const[:len(DATA_SC)]) + 1e-12), 'x')
        axes[2,0].set_title("5. Subcarrier SNR Audit (Are edges or center dying?)")
        
        # 6. Raw IQ Spread
        axes[2,1].hist2d(rx.real, rx.imag, bins=50); axes[2,1].set_title("6. Normalized IQ Density")
        
        plt.tight_layout(); plt.savefig("deep_audit_report.png")
        print("Success! Deep Audit Report saved to deep_audit_report.png")

    except Exception as e: print(f"Processing Error: {e}")
    finally:
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_audit_rx()