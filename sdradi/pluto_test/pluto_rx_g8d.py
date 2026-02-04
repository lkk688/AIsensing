import numpy as np
import adi
import matplotlib.pyplot as plt

# --- HARDWARE & SYNC PARAMS ---
URI, FC, FS = "usb:1.33.5", 2300e6, 1e6
N, CP = 64, 16
DATA_SC = np.array([-10, -9, 9, 10]) # Golden Window
PILOT_IDX = np.array([-12, 12])

def run_glass_box_step_audit():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI); sdr.rx_buffer_size = 2**20
        rx_raw = sdr.rx() / 2**14
        
        # --- STEP 1: SYNC AUDIT ---
        zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * 8
        corr = np.abs(np.correlate(rx_raw, zc_ref, mode='valid'))
        c_idx = np.argmax(corr[10000:]) + 10000 
        print(f"[STEP 1] Sync Lock at index: {c_idx} (Magnitude: {corr[c_idx]:.2f})")
        
        # --- STEP 2: CFO RESOLUTION ---
        # Resolve offset over Long Training Field (LTF)
        f_lock = np.angle(np.sum(np.conj(rx_raw[c_idx:c_idx+N]) * rx_raw[c_idx+N:c_idx+2*N])) / N
        rx_p = rx_raw * np.exp(-1j * f_lock * np.arange(len(rx_raw)))
        print(f"[STEP 2] CFO Resolved: {f_lock * FS / (2*np.pi):.2f} Hz")
        
        # --- STEP 3: CHANNEL EQ AUDIT ---
        ltf_start = c_idx + (12 * N) + CP
        np.random.seed(42); ltf_ref = np.fft.ifftshift(np.random.choice([1, -1], N).astype(complex))
        ltf_ref[range(-7, 8)] = 0
        H_est = np.fft.fft(rx_p[ltf_start:ltf_start+N]) / (ltf_ref + 1e-12)
        
        # --- STEP 4: PHASE VELOCITY & DATA DECODE ---
        data_start = ltf_start + (2 * N) + CP
        const, phase_err_log, diff_angles = [], [], []
        phase_acc, freq_acc, last_syms = 0, 0, None
        
        for i in range(150):
            idx = int(data_start + i*(N+CP))
            if idx+N+CP > len(rx_p): break
            # Symbol-level DC scrub
            sym = rx_p[idx+CP : idx+CP+N] - np.mean(rx_p[idx+CP : idx+CP+N])
            Y = np.fft.fftshift(np.fft.fft(sym * np.exp(-1j * phase_acc))) / (np.fft.fftshift(H_est) + 1e-12)
            
            p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
            freq_acc += 0.025 * p_err; phase_acc += 0.18 * p_err + freq_acc
            
            curr_data = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
            if last_syms is not None:
                diff = curr_data * np.conj(last_syms)
                diff_angles.extend(np.degrees(np.angle(diff)))
                const.extend(curr_data)
            last_syms = curr_data
            phase_err_log.append(np.degrees(p_err))

        # --- GENERATE 6-STEP AUDIT FIGURE ---
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        
        # 1. Sync Sharpness
        axes[0,0].plot(corr[c_idx-500:c_idx+500]); axes[0,0].set_title("1. Sync Sharpness (Timing)")
        # 2. Channel Magnitude
        axes[0,1].plot(np.arange(-32,32), 20*np.log10(np.abs(np.fft.fftshift(H_est))+1e-12))
        axes[0,1].set_title("2. Channel Magnitude (EQ Stability)")
        # 3. Constellation
        axes[1,0].scatter(np.real(const), np.imag(const), s=5, alpha=0.5)
        axes[1,0].set_title("3. Audited Constellation"); axes[1,0].grid(True)
        # 4. Phase Decision Histogram
        axes[1,1].hist(diff_angles, bins=180); axes[1,1].set_title("4. DQPSK Angle Distribution (Must have 4 Peaks)")
        # 5. PLL Flatness
        axes[2,0].plot(phase_err_log); axes[2,0].set_title("5. PLL Phase Error (Stability)")
        # 6. Time Domain Frame Start
        axes[2,1].plot(rx_p[c_idx:c_idx+1000].real); axes[2,1].set_title("6. Raw Frame Start (Time Domain)")

        plt.tight_layout(); plt.savefig("step_audit_v4.png")
        print(f"🏁 Step Audit complete. Report: step_audit_v4.png.")

    except Exception as e: print(f"❌ Error: {e}")
    finally: 
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_glass_box_step_audit()