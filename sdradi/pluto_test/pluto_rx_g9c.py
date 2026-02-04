import numpy as np
import adi
import matplotlib.pyplot as plt

# --- PRODUCTION PARAMS ---
URI_RX = "usb:1.37.5"
N, CP = 32, 32
DATA_SC = np.array([-12, -11, 11, 12]) 
PILOT_IDX = np.array([-14, 14])

def run_production_recovery():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI_RX)
        sdr.rx_buffer_size = 2**22 
        print("📡 Capture started. Listening for High-Vis stream...")
        rx_raw = sdr.rx() / 2**14
        
        # --- FIX: ZC REFERENCE (Correcting Broadcast Error) ---
        n = np.arange(N)
        # Sequence must be N elements long
        zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 17 * n * (n + 1) / N)) * 5.6
        
        # 1. TIMING & CFO MASTER LOCK
        # Use previous best lock of -9500 Hz
        f_master = -9500 
        rx_cfo = rx_raw * np.exp(-1j * 2 * np.pi * f_master / 1e6 * np.arange(len(rx_raw)))
        
        corr = np.abs(np.correlate(rx_cfo[10000:60000], zc_ref, mode='valid'))
        c_idx = np.argmax(corr) + 10000
        
        # 2. DEEP STATE TRACKING
        data_start = c_idx + (40 * N) + CP
        recovered_bits, last_syms = [], None
        ph_acc, sco_acc = 0, 0
        
        # Diagnostic Logs
        const, ph_err_log, sco_log, bit_angles = [], [], [], []

        for i in range(250):
            idx = data_start + i*(N+CP) + sco_acc
            int_idx = int(np.floor(idx))
            if int_idx + N + 1 > len(rx_cfo): break
            
            # Linear Resampling for thermal drift
            frac = idx - int_idx
            sym = (1-frac)*rx_cfo[int_idx : int_idx+N] + frac*rx_cfo[int_idx+1 : int_idx+N+1]
            
            # FFT & Rotation
            Y = np.fft.fftshift(np.fft.fft(sym * np.exp(-1j * ph_acc)))
            p_vals = Y[(PILOT_IDX + N//2) % N]
            
            # Dual-Loop Update (PI Controller)
            p_err = np.angle(np.mean(p_vals))
            sco_err = np.angle(p_vals[1] * np.conj(p_vals[0]))
            
            ph_acc += 0.15 * p_err
            sco_acc += 0.005 * sco_err
            
            # Data Extraction
            curr_data = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
            const.extend(curr_data)
            ph_err_log.append(np.degrees(p_err))
            sco_log.append(sco_acc)
            
            if last_syms is not None:
                diff = curr_data * np.conj(last_syms)
                angles = (np.angle(diff) + np.pi/4) % (2*np.pi)
                bit_angles.extend(angles)
                bits = (angles // (np.pi/2)).astype(int)
                for b in bits: recovered_bits.extend([int(b >> 1), int(b & 1)])
            last_syms = curr_data

        # --- GENERATE MULTI-STATE DIAGNOSTIC ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Constellation (Post-Resample)
        axes[0,0].scatter(np.real(const), np.imag(const), s=5, color='darkgreen', alpha=0.5)
        axes[0,0].set_title("1. Thermal-Corrected Constellation"); axes[0,0].grid(True)
        
        # 2. SCO Convergence (Thermal Tracking)
        axes[0,1].plot(sco_log); axes[0,1].set_title("2. SCO Drift Tracking (Samples)")
        
        # 3. Phase Error History
        axes[1,0].plot(ph_err_log); axes[1,0].set_title("3. PLL Phase Residual (Degrees)")
        
        # 4. DQPSK Angle Decision Peaks
        axes[1,1].hist(bit_angles, bins=100); axes[1,1].set_title("4. Angle Histogram (Must show 4 Spikes)")
        
        plt.tight_layout(); plt.savefig("thermal_leash_report.png")
        print("🏁 Diagnostic generated: thermal_leash_report.png")

        # Save JPEG
        with open("recovered_final.jpg", "wb") as f: f.write(np.packbits(recovered_bits))
        print("🏁 Recovery finished. Check recovered_final.jpg!")

    except Exception as e: print(f"❌ Error: {e}")
    finally: 
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_production_recovery()