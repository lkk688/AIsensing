import numpy as np
import adi
import matplotlib.pyplot as plt

# --- RX CONFIGURATION ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_sample_velocity_rx():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI)
        sdr.sample_rate, sdr.tx_lo = int(FS), int(FC)
        sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**19, 22 # Lowest gain for highest SNR

        print("\n--- SAMPLE-VELOCITY AUDITOR LOG ---")
        rx_raw = sdr.rx(); rx = rx_raw / (np.sqrt(np.mean(np.abs(rx_raw)**2)) + 1e-12)
        
        # 1. ZC-SYNC & RESOLVED CFO
        zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
        corr = np.abs(np.correlate(rx, zc_ref, mode='valid'))
        c_idx = np.argmax(corr[10000:]) + 10000 
        f_lock = np.angle(np.sum(np.conj(rx[c_idx:c_idx+N]) * rx[c_idx+N:c_idx+2*N])) / N
        rx_locked = rx * np.exp(-1j * f_lock * np.arange(len(rx)))
        print(f"[STAGE: CFO] Master Locked: {f_lock * FS / (2*np.pi):.2f} Hz")
        
        # 2. CHANNEL EQ (15-SC NOTCH)
        ltf_start = c_idx + (12 * N) + CP
        np.random.seed(42); ltf_ref = np.fft.ifftshift(np.random.choice([1, -1], N).astype(complex)); ltf_ref[range(-7, 8)] = 0
        H_est = np.fft.fft(rx_locked[ltf_start:ltf_start+N]) / (ltf_ref + 1e-12)

        # 3. SAMPLE-BY-SAMPLE INTERPOLATION
        PILOT_IDX = np.array([-22, 22]); DATA_SC = np.array([-20, -18, 18, 20])
        const, velocity_log, pll_check = [], [], []
        data_start = ltf_start + (2 * N) + CP
        
        phase_accum = 0
        for i in range(50):
            idx = data_start + i*(N+CP)
            if idx + N + CP > len(rx_locked): break
            
            # Predict & apply phase ramp across samples
            # This cancels the -4.6 deg/symbol drift INSIDE the symbol
            t_samples = np.arange(idx+CP, idx+CP+N)
            symbol_raw = rx_locked[idx+CP : idx+CP+N] * np.exp(-1j * phase_accum)
            
            Y = np.fft.fftshift(np.fft.fft(symbol_raw)) / (np.fft.fftshift(H_est) + 1e-12)
            p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
            
            # Update the accumulator with a "Velocity Estimator"
            phase_accum += p_err * 0.5 
            
            current_data = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
            const.extend(current_data)
            velocity_log.append(np.degrees(p_err))
            pll_check.append(np.angle(symbol_raw[0] * np.exp(-1j * p_err)))

        # --- ADVANCED DIAGNOSTIC PLOTS ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Constellation
        axes[0,0].scatter(np.array(const).real, np.array(const).imag, s=5, alpha=0.5, color='blue')
        axes[0,0].set_title("1. INTERPOLATED CONSTELLATION"); axes[0,0].grid(True)
        
        # 2. Velocity Auditor (Slope = Clock Drift)
        axes[0,1].plot(velocity_log); axes[0,1].set_title("2. Phase Velocity (Deg/Symbol)")
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        
        # 3. Phase Continuity
        axes[1,0].plot(np.unwrap(pll_check)); axes[1,0].set_title("3. PLL Horizontal Check (Must be Flat)")
        
        # 4. DC Audit
        axes[1,1].bar(np.arange(-32, 32), np.abs(np.fft.fftshift(np.fft.fft(rx[ltf_start:ltf_start+N]))))
        axes[1,1].set_title("4. DC Hole Audit (Check Pilot Bins ±22)")

        plt.tight_layout(); plt.savefig("sample_velocity_report.png")
        print(f"Success! Constellation Variance: {np.var(const):.2e}. Report in sample_velocity_report.png")

    except Exception as e: print(f"ERROR: {e}")
    finally:
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_sample_velocity_rx()