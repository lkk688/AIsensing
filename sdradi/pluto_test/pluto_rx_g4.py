import numpy as np
import adi
import matplotlib.pyplot as plt

# --- RX HARDWARE CONFIG ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_state_space_rx():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI)
        sdr.sample_rate, sdr.rx_lo = int(FS), int(FC)
        sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**19, 35 # Pre-Saturation Zone

        print(f"\n--- STATE-SPACE DEBUG LOG ---")
        rx_raw = sdr.rx(); rx = rx_raw / (np.sqrt(np.mean(np.abs(rx_raw)**2)) + 1e-12)
        
        # 1. TIMING STATE
        stf_sc = np.zeros(N, dtype=complex); stf_sc[::4] = (1 + 1j) * np.sqrt(4)
        corr = np.abs(np.correlate(rx, np.fft.ifft(stf_sc), mode='valid'))
        c_idx = np.argmax(corr[10000:]) + 10000 
        print(f"[STATE: TIMING] Sync Index: {c_idx} | Peak SNR: {20*np.log10(corr[c_idx]/np.mean(corr)):.2f} dB")
        
        # 2. FREQUENCY STATE
        L = 16
        drifts = [np.angle(np.sum(np.conj(rx[c_idx+i*L:c_idx+(i+1)*L])*rx[c_idx+(i+1)*L:c_idx+(i+2)*L])) for i in range(8)]
        coarse_f = np.median(drifts) / L
        print(f"[STATE: CFO] Residual Offset: {coarse_f * FS / (2*np.pi):.2f} Hz")
        rx_locked = rx * np.exp(-1j * coarse_f * np.arange(len(rx)))
        
        # 3. CHANNEL STATE
        ltf_start = c_idx + (10 * 16) + CP
        np.random.seed(42); ltf_ref = np.fft.ifftshift(np.random.choice([1, -1], N).astype(complex)); ltf_ref[range(-3, 4)] = 0
        H_est = np.fft.fft(rx_locked[ltf_start:ltf_start+N]) / (ltf_ref + 1e-12)
        print(f"[STATE: EQ] Channel Magnitude: {np.mean(np.abs(H_est)):.2e}")

        # 4. DIFFERENTIAL DECODING STATE
        PILOT_IDX = np.array([-21, -7, 7, 21]); DATA_SC = np.array([sc for sc in range(-26, 27) if sc not in range(-3, 4) and sc not in PILOT_IDX])
        const, diff_const, track_log = [], [], []
        data_start = ltf_start + (2 * N) + CP
        
        last_symbol_fft = None
        for i in range(80):
            idx = data_start + i*(N+CP)
            if idx + N + CP > len(rx_locked): break
            
            Y = np.fft.fftshift(np.fft.fft(rx_locked[idx+CP : idx+CP+N])) / (np.fft.fftshift(H_est) + 1e-12)
            p_phase = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
            current_symbol = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_phase)
            
            if last_symbol_fft is not None:
                # DIFFERENTIAL STEP: Extract change in phase
                diff_const.extend(current_symbol * np.conj(last_symbol_fft))
            
            last_symbol_fft = current_symbol
            const.extend(current_symbol)
            track_log.append(np.degrees(p_phase))

        print(f"[STATE: DATA] Decoded {len(diff_const)} differential symbols.")

        # --- SIGNAL INTEGRITY MAP ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0,0].scatter(np.array(diff_const).real, np.array(diff_const).imag, s=2, alpha=0.3, color='forestgreen')
        axes[0,0].set_title("1. DIFFERENTIAL CONSTELLATION (Should be 4 clusters)"); axes[0,0].grid(True)
        
        axes[0,1].plot(track_log); axes[0,1].set_title("2. Residual Phase Wave (The 'Plot 3' Killer)")
        
        axes[1,0].plot(np.unwrap(np.deg2rad(track_log))); axes[1,0].set_title("3. PLL Phase Continuity (Unwrapped)")
        
        axes[1,1].bar(np.arange(-32, 32), np.abs(np.fft.fftshift(np.fft.fft(rx[ltf_start:ltf_start+N]))))
        axes[1,1].set_title("4. DC Audit (Check for Hardware Saturation)")
        
        plt.tight_layout(); plt.savefig("state_space_victory.png")
        print(f"Success! State-Space report in state_space_victory.png. Variance: {np.var(track_log):.2f}")

    except Exception as e: print(f"ERROR: {e}")
    finally:
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_state_space_rx()