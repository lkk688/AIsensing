import numpy as np
import adi
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

# --- MASTER PRODUCTION CONFIG ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 1e6 
N, CP = 64, 16 
PILOT_SC = np.array([-21, -7, 7, 21]) 
PILOT_VALS = np.array([1, 1, 1, -1]) 
# DATA_SC has 48 elements for a 64-FFT (standard 802.11 style)
DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_SC])

def run_mathworks_style_sync_v2():
    try:
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate, sdr.tx_lo, sdr.rx_lo = int(FS), int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -15, 40
        sdr.rx_buffer_size = 2**18

        # 1. Preamble & Payload Construction
        np.random.seed(99)
        ltf_sc = np.random.choice([-1, 1], N)
        ltf_t = np.fft.ifft(np.fft.ifftshift(ltf_sc)) * np.sqrt(N)
        preamble = np.concatenate([ltf_t, ltf_t]).astype(np.complex64)
        
        num_symbols = 30
        tx_bits = np.random.randint(0, 2, len(DATA_SC) * num_symbols * 2)
        qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
        
        payload = []
        for i in range(num_symbols):
            X = np.zeros(N, dtype=complex)
            X[(PILOT_SC + N//2) % N] = PILOT_VALS
            X[(DATA_SC + N//2) % N] = qpsk[i*len(DATA_SC) : (i+1)*len(DATA_SC)]
            p = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
            payload.append(np.concatenate([p[-CP:], p]))
            
        tx_sig = np.concatenate([np.zeros(10000, dtype=complex), preamble, *payload])
        sdr.tx_cyclic_buffer = False
        sdr.tx((tx_sig * 0.5 * 2**14).astype(np.complex64))
        time.sleep(0.2)
        rx = sdr.rx()
        sdr.tx_destroy_buffer()

        # 2. Sync and Best Window Search
        rx = rx - np.mean(rx)
        corr = np.abs(np.correlate(rx, ltf_t, mode='valid'))
        c_idx = np.argmax(corr)
        
        # Scan for best timing offset
        best_evm, best_offset = float('inf'), 0
        for offset in range(-8, 9):
            idx = c_idx + N + offset
            Y = np.fft.fftshift(np.fft.fft(rx[idx+CP : idx+CP+N]))
            evm = np.var(np.angle(Y[(PILOT_SC + N//2) % N] / PILOT_VALS))
            if evm < best_evm:
                best_evm, best_offset = evm, offset

        # 3. Equalization with Fix for Shape Mismatch
        final_idx = c_idx + N + best_offset
        all_syms = []
        phase_errors = [] # For Heatmap
        
        for i in range(num_symbols):
            idx = final_idx + i*(N+CP)
            Y = np.fft.fftshift(np.fft.fft(rx[idx+CP : idx+CP+N]))
            
            # Channel Estimate at Pilots
            Hp = Y[(PILOT_SC + N//2) % N] / PILOT_VALS
            phase_errors.append(np.angle(Hp))
            
            # Interpolate to ALL used subcarriers (Data + Pilots)
            used_indices = np.sort(np.concatenate([DATA_SC, PILOT_SC]))
            f_interp = interp1d(PILOT_SC, Hp, kind='linear', fill_value="extrapolate")
            H_used = f_interp(used_indices)
            
            # Equalize only the USED subcarriers
            Y_used = Y[(used_indices + N//2) % N]
            Y_eq = Y_used / (H_used + 1e-12)
            
            # Extract Data Carriers only from the equalized set
            data_mask = np.isin(used_indices, DATA_SC)
            all_syms.extend(Y_eq[data_mask])

        # 4. DIAGNOSTIC PLOTTING
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # A. Final Constellation
        all_syms = np.array(all_syms)
        all_syms /= np.mean(np.abs(all_syms))
        axes[0,0].scatter(all_syms.real, all_syms.imag, s=5, alpha=0.4, color='blue')
        axes[0,0].set_title(f"Constellation (Offset: {best_offset})")
        axes[0,0].axis([-2, 2, -2, 2]); axes[0,0].grid(True)
        
        # B. Phase Error Heatmap (Symbol vs Pilot)
        im = axes[0,1].imshow(np.array(phase_errors), aspect='auto', cmap='hsv')
        axes[0,1].set_title("Phase Rotation Heatmap (Symbol vs Pilot)")
        fig.colorbar(im, ax=axes[0,1])
        
        # C. Correlation Surface (Sharpness)
        axes[1,0].plot(corr[c_idx-100:c_idx+100])
        axes[1,0].set_title("Sync Peak Stability")
        
        # D. Channel Phase Slope
        axes[1,1].plot(PILOT_SC, np.angle(Hp), 'o-')
        axes[1,1].set_title("Current Symbol Phase Slope")

        plt.tight_layout()
        plt.savefig("fixed_sync_debug.png")
        print(f"Debug figures saved to fixed_sync_debug.png. BER Estimate: {np.mean(np.abs(all_syms) > 1.5)}")

    except Exception as e:
        print(f"Sync Error: {e}")

if __name__ == "__main__":
    run_mathworks_style_sync_v2()