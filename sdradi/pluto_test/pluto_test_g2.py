import numpy as np
import adi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION (Optimized for Cabled Loopback) ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 2e6 
N, CP = 64, 16 
# Dense pilots to prevent phase aliasing and handle steep slopes
PILOT_SC = np.array([-26, -18, -10, -2, 2, 10, 18, 26]) 
PILOT_VALS = np.ones(8) + 0j
DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_SC])

def run_integrated_loopback():
    try:
        print(f"Initializing Pluto at {IP}...")
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate, sdr.tx_lo, sdr.rx_lo = int(FS), int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -10, 20
        sdr.rx_buffer_size = 2**18 # 256k buffer to catch the burst

        # 1. Burst Waveform Construction
        zc_len = 1024
        zc = np.exp(-1j * np.pi * 29 * np.arange(zc_len) * (np.arange(zc_len) + 1) / zc_len)
        preamble = np.concatenate([zc, zc]).astype(np.complex64)
        
        np.random.seed(123)
        tx_bits = np.random.randint(0, 2, len(DATA_SC) * 20 * 2)
        qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
        
        tx_payload = []
        for i in range(20):
            X = np.zeros(N, dtype=complex)
            X[(PILOT_SC + N//2) % N] = PILOT_VALS
            X[(DATA_SC + N//2) % N] = qpsk[i*len(DATA_SC) : (i+1)*len(DATA_SC)]
            x_t = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
            tx_payload.append(np.concatenate([x_t[-CP:], x_t]))
            
        # THE CRITICAL FIX: Add leading/trailing zeros to create a clear "Event"
        tx_sig = np.concatenate([np.zeros(20000, dtype=complex), preamble, *tx_payload, np.zeros(5000, dtype=complex)])
        
        # 2. Execution (Burst Mode)
        sdr.tx_cyclic_buffer = False
        print("Sending Triggered Burst...")
        sdr.tx((tx_sig * 0.5 * 2**14).astype(np.complex64))
        
        time.sleep(0.1)
        rx = sdr.rx()
        sdr.tx_destroy_buffer()

        # 3. Synchronization (Burst-Aware)
        rx_no_dc = rx - np.mean(rx) # Remove central spike
        corr = np.abs(np.correlate(rx_no_dc, zc, mode='valid'))
        c_idx = np.argmax(corr)
        
        # CFO Calculation
        r1, r2 = rx[c_idx:c_idx+zc_len], rx[c_idx+zc_len:c_idx+2*zc_len]
        cfo_rad = np.angle(np.sum(r2 * np.conj(r1))) / zc_len
        corrected = rx_no_dc * np.exp(-1j * cfo_rad * np.arange(len(rx)))
        
        # Apply verified physical offset (+4 samples)
        idx = c_idx + (2 * zc_len) + 4 

        # 4. Equalization & Fine-Timing (STO Correction)
        all_eq_syms = []
        debug_slopes = []
        for i in range(20):
            Y = np.fft.fftshift(np.fft.fft(corrected[idx + i*(N+CP) + CP : idx + i*(N+CP) + CP + N])) / np.sqrt(N)
            
            # Frequency-Domain STO correction via Pilot Phase Slope
            H_pilots = Y[(PILOT_SC + N//2) % N] / PILOT_VALS
            phases = np.unwrap(np.angle(H_pilots))
            slope, _ = np.polyfit(PILOT_SC, phases, 1)
            debug_slopes.append(phases)
            
            # De-rotate linear phase slope
            Y_sto_corr = Y * np.exp(-1j * slope * np.arange(-N//2, N//2))
            
            # LS Channel Interpolation
            Hp = Y_sto_corr[(PILOT_SC + N//2) % N] / PILOT_VALS
            H_eff = np.interp(np.arange(N), (PILOT_SC + N//2) % N, Hp)
            Y_eq = Y_sto_corr / (H_eff + 1e-12)
            
            # Symbol-by-Symbol CPE Correction
            cpe = np.angle(np.sum(Y_eq[(PILOT_SC + N//2) % N] * np.conj(PILOT_VALS)))
            all_eq_syms.extend(Y_eq[(DATA_SC + N//2) % N] * np.exp(-1j * cpe))

        # 5. BER and Internal State Figures
        all_eq_syms = np.array(all_eq_syms)
        rx_bits = np.concatenate([[(s.real < 0).astype(int), (s.imag < 0).astype(int)] for s in all_eq_syms])
        ber = np.sum(tx_bits[:len(rx_bits)] != rx_bits) / len(rx_bits)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # Correlation: Verify single peak
        axes[0,0].plot(corr); axes[0,0].set_title(f"Correlation Surface (Peak: {c_idx})")
        # Constellation: Verify clusters
        axes[0,1].scatter(all_eq_syms.real, all_eq_syms.imag, marker='.', alpha=0.5, color='green')
        axes[0,1].set_title(f"LOCKED CONSTELLATION | BER: {ber:.4f}")
        axes[0,1].axis([-2, 2, -2, 2]); axes[0,1].grid(True)
        # Phase Slope: Verify STO correction
        axes[1,0].plot(PILOT_SC, debug_slopes[0], 'o-'); axes[1,0].set_title("Initial Phase Slope (STO Check)")
        # Power spectrum
        axes[1,1].specgram(rx, Fs=FS); axes[1,1].set_title("Spectrogram (Burst Capture)")
        
        plt.tight_layout(); plt.savefig("final_link_debug.png")
        print(f"Link Test Success! BER: {ber:.4e}. Data saved to final_link_debug.png")

    except Exception as e:
        print(f"Integrated Link Error: {e}")

if __name__ == "__main__":
    run_integrated_loopback()