import numpy as np
import adi
import matplotlib.pyplot as plt
import time

# --- PRODUCTION OTA CONFIG ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 1e6 
N, CP = 64, 32 
PILOT_SC = np.array([-26, -18, -10, -2, 2, 10, 18, 26]) 
PILOT_VALS = np.ones(8) + 0j
DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_SC])

def run_deep_space_diagnostic():
    try:
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate, sdr.tx_lo, sdr.rx_lo = int(FS), int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -15, 40
        sdr.rx_buffer_size = 2**18

        # 1. Build Burst
        zc_len = 1024
        zc = np.exp(-1j * np.pi * 29 * np.arange(zc_len) * (np.arange(zc_len) + 1) / zc_len)
        preamble = np.concatenate([zc, zc]).astype(np.complex64)
        
        np.random.seed(42)
        tx_bits = np.random.randint(0, 2, len(DATA_SC) * 30 * 2) # Longer burst for tracking
        qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
        
        tx_payload = []
        for i in range(30):
            X = np.zeros(N, dtype=complex)
            X[(PILOT_SC + N//2) % N] = PILOT_VALS
            X[(DATA_SC + N//2) % N] = qpsk[i*len(DATA_SC) : (i+1)*len(DATA_SC)]
            x_t = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
            tx_payload.append(np.concatenate([x_t[-CP:], x_t]))
            
        tx_sig = np.concatenate([np.zeros(10000, dtype=complex), preamble, *tx_payload])
        sdr.tx_cyclic_buffer = False
        sdr.tx((tx_sig * 0.6 * 2**14).astype(np.complex64))
        
        time.sleep(0.2)
        rx = sdr.rx()
        sdr.tx_destroy_buffer()

        # 2. Sync & Symbol-by-Symbol Tracking
        rx = rx - np.mean(rx)
        corr = np.abs(np.correlate(rx, zc, mode='valid'))
        c_idx = np.argmax(corr)
        best_idx = c_idx + 2*zc_len - 9 # Using your verified -9 offset

        all_eq_syms = []
        phase_tracking_log = []
        
        for i in range(30):
            idx = best_idx + i*(N+CP)
            Y = np.fft.fftshift(np.fft.fft(rx[idx+CP : idx+CP+N])) / np.sqrt(N)
            
            # Step A: LS Channel Estimate
            Hp = Y[(PILOT_SC + N//2) % N] / PILOT_VALS
            H_full = np.interp(np.arange(-N//2, N//2), PILOT_SC, Hp)
            Y_eq = Y / (H_full + 1e-12)
            
            # Step B: Common Phase Error (CPE) Tracking
            # This detects the "spin" for THIS specific symbol
            cpe_rad = np.angle(np.sum(Y_eq[(PILOT_SC + N//2) % N] * np.conj(PILOT_VALS)))
            phase_tracking_log.append(np.degrees(cpe_rad))
            
            # Rotate back
            all_eq_syms.extend(Y_eq[(DATA_SC + N//2) % N] * np.exp(-1j * cpe_rad))

        # 3. Figure Export
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # A. CPE Tracking (Does the phase drift over time?)
        axes[0,0].plot(phase_tracking_log, 'o-')
        axes[0,0].set_title("Symbol Phase Drift (Degrees)")
        axes[0,0].set_xlabel("Symbol Index")
        
        # B. Constellation (Post-Tracking)
        all_eq_syms = np.array(all_eq_syms)
        all_eq_syms /= np.mean(np.abs(all_eq_syms))
        axes[0,1].scatter(all_eq_syms.real, all_eq_syms.imag, marker='.', alpha=0.5, color='purple')
        axes[0,1].set_title("Phase-Tracked Constellation")
        axes[0,1].axis([-2, 2, -2, 2]); axes[0,1].grid(True)
        
        # C. LS Magnitude Stability
        axes[1,0].plot(np.abs(H_full))
        axes[1,0].set_title("Final Channel Magnitude")
        
        # D. Time-Domain Burst Envelope
        axes[1,1].plot(np.abs(rx[c_idx:c_idx+5000]))
        axes[1,1].set_title("Burst Capture Envelope")
        
        plt.tight_layout(); plt.savefig("deep_space_results.png")
        print("Diagnostic Complete. Check deep_space_results.png")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_deep_space_diagnostic()