import numpy as np
import adi
import matplotlib.pyplot as plt

# --- HARDWARE & RESYNC PARAMS ---
URI, FC, FS = "usb:1.37.5", 2300e6, 1e6 
N, CP = 32, 32 # Robust narrow-band mode

def run_coherent_resync_rx():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI); sdr.rx_buffer_size = 2**20
        rx_raw = sdr.rx() / 2**14
        
        # 1. APPLY MASTER LOCK (-9500 Hz)
        f_master = -9500 
        rx_cfo = rx_raw * np.exp(-1j * 2 * np.pi * f_master / FS * np.arange(len(rx_raw)))
        
        # 2. ZC-SYNC & TIMING
        zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 17 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
        corr = np.abs(np.correlate(rx_cfo[10000:40000], zc_ref, mode='valid'))
        c_idx = np.argmax(corr) + 10000 
        
        # 3. SAMPLE-LEVEL RESAMPLING LOOP
        PILOT_IDX = np.array([-10, 10]); DATA_SC = np.array([-4, -3, 3, 4])
        data_start = c_idx + (32 * N) + (2 * (N+CP)) + CP
        const, phase_log, sco_log = [], [], []
        
        phase_acc, sco_acc = 0, 0
        Kp_ph, Ki_ph = 0.15, 0.02 # Phase PI
        Kp_sco, Ki_sco = 0.005, 0.0005 # Sampling Clock PI

        for i in range(100):
            # Calculate fractional sample offset
            idx = data_start + i*(N+CP) + sco_acc
            int_idx = int(np.floor(idx))
            if int_idx + N + CP > len(rx_cfo): break
            
            # Simple Linear Interpolation for sub-sample alignment
            frac = idx - int_idx
            sym_raw = (1-frac)*rx_cfo[int_idx : int_idx+N] + frac*rx_cfo[int_idx+1 : int_idx+N+1]
            
            # Phase Rotation
            sym_rot = sym_raw * np.exp(-1j * phase_acc)
            Y = np.fft.fftshift(np.fft.fft(sym_rot))
            
            # Pilot Error Calculation
            p_vals = Y[(PILOT_IDX + N//2) % N]
            p_err = np.angle(np.mean(p_vals))
            
            # SCO Error: Phase difference between pilots at +10 and -10
            sco_err = np.angle(p_vals[1] * np.conj(p_vals[0]))
            
            # Dual PI Loop Update
            phase_acc += Kp_ph * p_err
            sco_acc += Kp_sco * sco_err
            
            const.extend(Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err))
            phase_log.append(np.degrees(p_err))
            sco_log.append(sco_acc)

        # --- DIAGNOSTIC REPORT ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(np.real(const), np.imag(const), s=8, alpha=0.5, color='darkblue')
        axes[0].set_title("1. RESAMPLED CONSTELLATION"); axes[0].grid(True)
        
        axes[1].plot(sco_log); axes[1].set_title("2. Sampling Clock Drift (Samples)")
        plt.tight_layout(); plt.savefig("resync_audit_final.png")
        print(f"✅ Coherent Audit Complete. Final SCO: {sco_acc:.4f} samples.")

    except Exception as e: print(f"Error: {e}")
    finally: 
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": 
    run_coherent_resync_rx()