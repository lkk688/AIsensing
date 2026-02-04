import numpy as np
import adi
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d # Fix: Explicit import for interpolation

# --- DIAGNOSTIC MASTER CONFIG ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 1e6 
N, CP = 64, 32 
# Maximize pilot density for diagnostic "Pathology Mapping"
PILOT_SC = np.array([-26, -20, -14, -8, -2, 4, 10, 16, 22]) 
PILOT_VALS = np.ones(len(PILOT_SC)) + 0j
DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_SC])

def run_diagnostic_master_v2():
    try:
        print(f"Connecting to Pluto at {IP}...")
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate = int(FS)
        sdr.tx_lo, sdr.rx_lo = int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -15, 45
        sdr.rx_buffer_size = 2**18

        # 1. Build a "Known-Truth" Frame
        zc_len = 1024
        zc = np.exp(-1j * np.pi * 29 * np.arange(zc_len) * (np.arange(zc_len) + 1) / zc_len)
        preamble = np.concatenate([zc, zc]).astype(np.complex64)
        
        # Fixed reference symbol for all carriers
        X_ref = np.zeros(N, dtype=complex)
        X_ref[(PILOT_SC + N//2) % N] = PILOT_VALS
        x_t = np.fft.ifft(np.fft.ifftshift(X_ref)) * np.sqrt(N)
        tx_sig = np.concatenate([np.zeros(20000, dtype=complex), preamble, 
                                 np.tile(np.concatenate([x_t[-CP:], x_t]), 50)])
        
        sdr.tx_cyclic_buffer = False
        sdr.tx((tx_sig * 0.5 * 2**14).astype(np.complex64))
        time.sleep(0.2)
        rx = sdr.rx()
        sdr.tx_destroy_buffer()

        # 2. Sync and CFO
        rx = rx - np.mean(rx) # DC Offset Removal
        corr = np.abs(np.correlate(rx, zc, mode='valid'))
        c_idx = np.argmax(corr)
        base_idx = c_idx + 2*zc_len - 9 # Applying your verified hardware offset

        # 3. MAPPING PHASE ERROR ACROSS TIME & FREQUENCY
        error_surface = []
        magnitudes = []
        for i in range(20):
            idx = base_idx + i*(N+CP)
            # Fix: Correct call to np.fft.fft()
            Y = np.fft.fftshift(np.fft.fft(rx[idx+CP : idx+CP+N])) 
            Hp = Y[(PILOT_SC + N//2) % N] / PILOT_VALS
            error_surface.append(np.angle(Hp))
            magnitudes.append(np.abs(Hp))

        # 4. EXPORT INTERNAL PATHOLOGY FIGURES
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # A. Phase Error Surface (Heatmap)
        im = axes[0,0].imshow(np.array(error_surface), aspect='auto', cmap='hsv')
        axes[0,0].set_title("Phase Error Surface (Symbol vs Pilot Index)")
        fig.colorbar(im, ax=axes[0,0])
        
        # B. Average Channel Magnitude
        avg_mag = np.mean(np.array(magnitudes), axis=0)
        axes[0,1].bar(PILOT_SC, avg_mag)
        axes[0,1].set_title("Average Magnitude across Pilots (Check for Nulls)")
        
        # C. Constellation of Raw Pilots
        raw_pilots = (np.array(error_surface)).flatten()
        axes[1,0].scatter(np.cos(raw_pilots), np.sin(raw_pilots), alpha=0.5)
        axes[1,0].set_title("Pilot Phase Distribution (Should cluster at 0 deg)")
        axes[1,0].axis([-1.5, 1.5, -1.5, 1.5])
        
        # D. Phase Unwrapping Diagnostic
        for i in range(5): # Plot first 5 symbols
            axes[1,1].plot(PILOT_SC, np.unwrap(error_surface[i]), label=f'Sym {i}')
        axes[1,1].set_title("Unwrapped Phase Profile (Slope = Timing Error)")
        axes[1,1].legend()

        plt.tight_layout()
        plt.savefig("master_pathology_v2.png")
        print(f"Master Pathology V2 saved. Sync Index: {base_idx}")

    except Exception as e:
        print(f"Master Error: {e}")

if __name__ == "__main__":
    run_diagnostic_master_v2()