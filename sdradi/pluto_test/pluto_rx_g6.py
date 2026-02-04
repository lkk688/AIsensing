import numpy as np
import adi
import matplotlib.pyplot as plt

# --- RX CONFIGURATION ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_continuous_auditor_rx():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI); sdr.sample_rate, sdr.tx_lo = int(FS), int(FC)
        sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**19, 22 # Minimal gain for purity

        print("\n--- CONTINUOUS FLOW AUDITOR ---")
        rx_raw = sdr.rx(); rx = rx_raw / (np.sqrt(np.mean(np.abs(rx_raw)**2)) + 1e-12)
        
        # 1. ZC-SYNC & RESOLVED CFO
        zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
        corr = np.abs(np.correlate(rx, zc_ref, mode='valid'))
        c_idx = np.argmax(corr[10000:]) + 10000 
        f_lock = np.angle(np.sum(np.conj(rx[c_idx:c_idx+N]) * rx[c_idx+N:c_idx+2*N])) / N
        rx_locked = rx * np.exp(-1j * f_lock * np.arange(len(rx)))
        
        # 2. CHANNEL STATE
        ltf_start = c_idx + (12 * N) + CP
        np.random.seed(42); ltf_ref = np.fft.ifftshift(np.random.choice([1, -1], N).astype(complex)); ltf_ref[range(-9, 10)] = 0
        H_est = np.fft.fft(rx_locked[ltf_start:ltf_start+N]) / (ltf_ref + 1e-12)

        # 3. VELOCITY-INTEGRATED TRACKING
        PILOT_IDX = np.array([-10, 10]); DATA_SC = np.array([-7, -6, -5, 5, 6, 7])
        const, track_log, pll_check = [], [], []
        data_start = ltf_start + (2 * N) + CP
        
        phase_accum, velocity = 0, 0
        alpha, beta = 0.05, 0.005 # Dual-loop gains for momentum tracking

        for i in range(60):
            idx = data_start + i*(N+CP)
            if idx + N + CP > len(rx_locked): break
            
            # Predict & Rotate
            phase_pred = phase_accum + velocity
            symbol_vec = rx_locked[idx+CP : idx+CP+N] * np.exp(-1j * phase_pred)
            
            Y = np.fft.fftshift(np.fft.fft(symbol_vec)) / (np.fft.fftshift(H_est) + 1e-12)
            p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
            
            # Update Feedback Loop
            phase_accum = phase_pred + alpha * p_err
            velocity = velocity + beta * p_err
            
            curr_data = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
            const.extend(curr_data)
            track_log.append(np.degrees(p_err)); pll_check.append(np.angle(symbol_vec[0] * np.exp(-1j * p_err)))

        # --- INTERNAL STATE DIAGNOSTICS ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0,0].scatter(np.array(const).real, np.array(const).imag, s=10, alpha=0.5, color='darkgreen')
        axes[0,0].set_title("1. CONTINUOUS CONSTELLATION"); axes[0,0].grid(True)
        axes[0,1].plot(track_log); axes[0,1].set_title("2. Frequency Steering Log")
        axes[1,0].plot(np.unwrap(pll_check)); axes[1,0].set_title("3. PLL Continuity (HORIZONTAL CHECK)")
        axes[1,1].bar(np.arange(-32, 32), np.abs(np.fft.fftshift(np.fft.fft(rx[ltf_start:ltf_start+N]))))
        axes[1,1].set_title("4. DC Hole Audit (Check Golden Window)")
        plt.tight_layout(); plt.savefig("ironclad_audit_report.png")
        print("Success! Diagnostic in ironclad_audit_report.png.")

    except Exception as e: print(f"ERROR: {e}")
    finally: 
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_continuous_auditor_rx()