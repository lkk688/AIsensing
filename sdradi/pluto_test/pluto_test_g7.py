import numpy as np
import adi
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

# --- PRODUCTION CONFIG (MathWorks 802.11 Optimized) ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 1e6 
N, CP = 64, 16 
PILOT_SC = np.array([-21, -7, 7, 21]) 
PILOT_VALS = np.array([1, 1, 1, -1]) 
DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_SC])

def run_mathworks_final_lock():
    try:
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate, sdr.tx_lo, sdr.rx_lo = int(FS), int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -15, 40
        sdr.rx_buffer_size = 2**18

        # 1. Waveform: LTF + 50 Symbols of Data
        np.random.seed(99)
        ltf_sc = np.random.choice([-1, 1], N)
        ltf_t = np.fft.ifft(np.fft.ifftshift(ltf_sc)) * np.sqrt(N)
        preamble = np.concatenate([ltf_t, ltf_t]).astype(np.complex64)
        
        num_symbols = 50
        tx_bits = np.random.randint(0, 2, len(DATA_SC) * num_symbols * 2)
        qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
        
        payload = []
        for i in range(num_symbols):
            X = np.zeros(N, dtype=complex)
            X[(PILOT_SC + N//2) % N] = PILOT_VALS
            X[(DATA_SC + N//2) % N] = qpsk[i*len(DATA_SC) : (i+1)*len(DATA_SC)]
            p = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
            payload.append(np.concatenate([p[-CP:], p]))
            
        sdr.tx_cyclic_buffer = False
        sdr.tx((np.concatenate([np.zeros(10000, dtype=complex), preamble, *payload]) * 0.5 * 2**14).astype(np.complex64))
        time.sleep(0.2); rx = sdr.rx(); sdr.tx_destroy_buffer()

        # 2. Sync & Coarse CFO
        rx = rx - np.mean(rx)
        corr = np.abs(np.correlate(rx, ltf_t, mode='valid'))
        c_idx = np.argmax(corr)
        
        # Fine CFO Estimation from LTF
        l1, l2 = rx[c_idx : c_idx+N], rx[c_idx+N : c_idx+2*N]
        f_cfo = np.angle(np.sum(l2 * np.conj(l1))) / N
        rx_cfo = rx * np.exp(-1j * f_cfo * np.arange(len(rx)))

        # 3. Decision-Directed Equalization
        base_idx = c_idx + 2*N + 7 # Forced +7 offset from your best result
        all_syms = []
        phase_tracking = []

        # Initial Channel Estimate from LTF
        Y_ltf = np.fft.fftshift(np.fft.fft(rx_cfo[c_idx : c_idx+N]))
        H_initial = Y_ltf / ltf_sc

        for i in range(num_symbols):
            idx = base_idx + i*(N+CP)
            Y = np.fft.fftshift(np.fft.fft(rx_cfo[idx+CP : idx+CP+N]))
            
            # Step A: Coarse Pilot-Aided Phase Fix
            Hp = Y[(PILOT_SC + N//2) % N] / PILOT_VALS
            cpe_coarse = np.angle(np.sum(Hp))
            Y = Y * np.exp(-1j * cpe_coarse)
            
            # Step B: LS Equalization
            f_interp = interp1d(PILOT_SC, Hp * np.exp(-1j*cpe_coarse), kind='linear', fill_value="extrapolate")
            H_curr = f_interp(DATA_SC)
            Y_data = Y[(DATA_SC + N//2) % N] / (H_curr + 1e-12)
            
            # Step C: Decision-Directed "Fine" Lock
            # Snaps the cluster into the quadrant
            snapped = (np.sign(Y_data.real) + 1j*np.sign(Y_data.imag)) / np.sqrt(2)
            fine_phase = np.angle(np.sum(Y_data * np.conj(snapped)))
            Y_final = Y_data * np.exp(-1j * fine_phase)
            
            all_syms.extend(Y_final)
            phase_tracking.append(np.degrees(cpe_coarse + fine_phase))

        # 4. FINAL STATE VISUALIZATION
        all_syms = np.array(all_syms)
        all_syms /= np.mean(np.abs(all_syms))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0,0].scatter(all_syms.real, all_syms.imag, s=2, alpha=0.3, color='blue')
        axes[0,0].set_title("LOCKED QPSK CONSTELLATION")
        axes[0,0].axis([-2, 2, -2, 2]); axes[0,0].grid(True)
        
        axes[0,1].plot(phase_tracking); axes[0,1].set_title("Total Phase Correction (Deg)")
        axes[1,0].plot(np.abs(Y_ltf)); axes[1,1].plot(np.unwrap(np.angle(H_initial)))
        axes[1,0].set_title("LTF Spectrum"); axes[1,1].set_title("Initial Phase Slope")
        
        plt.tight_layout(); plt.savefig("final_production_lock.png")
        
        rx_bits = np.concatenate([[(s.real < 0).astype(int), (s.imag < 0).astype(int)] for s in all_syms])
        print(f"Lock Attempt Complete. Diagnostic saved to final_production_lock.png")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_mathworks_final_lock()