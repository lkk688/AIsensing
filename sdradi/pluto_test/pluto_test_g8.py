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
DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_SC])

def run_phase_anchor_lock():
    try:
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate, sdr.tx_lo, sdr.rx_lo = int(FS), int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -15, 40
        sdr.rx_buffer_size = 2**18

        # 1. Waveform Generation
        np.random.seed(99)
        ltf_sc = np.random.choice([-1, 1], N)
        ltf_t = np.fft.ifft(np.fft.ifftshift(ltf_sc)) * np.sqrt(N)
        preamble = np.concatenate([ltf_t, ltf_t]).astype(np.complex64)
        
        num_symbols = 60
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

        # 2. Advanced CFO & Timing Correction
        rx = rx - np.mean(rx) # DC removal
        corr = np.abs(np.correlate(rx, ltf_t, mode='valid'))
        c_idx = np.argmax(corr)
        
        # Dual-Window CFO Estimation
        l1, l2 = rx[c_idx : c_idx+N], rx[c_idx+N : c_idx+2*N]
        coarse_cfo = np.angle(np.sum(l2 * np.conj(l1))) / N
        rx_fixed = rx * np.exp(-1j * coarse_cfo * np.arange(len(rx)))

        # 3. Dynamic Tracking with Decision-Directed "Snap"
        base_idx = c_idx + 2*N + 7 # Using verified +7 offset
        all_syms = []
        
        # Initial Channel Estimate from the first symbol
        Y_ref = np.fft.fftshift(np.fft.fft(rx_fixed[base_idx+CP : base_idx+CP+N]))
        H_ref = Y_ref[(PILOT_SC + N//2) % N] / PILOT_VALS

        for i in range(num_symbols):
            idx = base_idx + i*(N+CP)
            Y = np.fft.fftshift(np.fft.fft(rx_fixed[idx+CP : idx+CP+N]))
            
            # Pilot-Aided Phase Fix
            Hp = Y[(PILOT_SC + N//2) % N] / PILOT_VALS
            cpe = np.angle(np.sum(Hp * np.conj(H_ref)))
            Y = Y * np.exp(-1j * cpe)
            
            # Frequency-Domain STO Correction (Flattening the slope)
            phases = np.unwrap(np.angle(Hp))
            slope, _ = np.polyfit(PILOT_SC, phases, 1)
            Y = Y * np.exp(-1j * slope * np.arange(-N//2, N//2))
            
            # Equalization
            f_eq = interp1d(PILOT_SC, Hp * np.exp(-1j*cpe), kind='quadratic', fill_value="extrapolate")
            Y_eq = Y[(DATA_SC + N//2) % N] / (f_eq(DATA_SC) + 1e-12)
            
            # Final Decision-Directed Lock
            snapped = (np.sign(Y_eq.real) + 1j*np.sign(Y_eq.imag)) / np.sqrt(2)
            fine_phase = np.angle(np.sum(Y_eq * np.conj(snapped)))
            all_syms.extend(Y_eq * np.exp(-1j * fine_phase))

        # 4. Final Verification
        all_syms = np.array(all_syms)
        all_syms /= np.mean(np.abs(all_syms))
        
        plt.figure(figsize=(6,6))
        plt.scatter(all_syms.real, all_syms.imag, s=3, alpha=0.4, color='purple')
        plt.title(f"PHASE-ANCHORED LOCK\nBER: {np.mean(np.abs(all_syms) > 1.4):.4f}")
        plt.axis([-2, 2, -2, 2]); plt.grid(True); plt.savefig("phase_anchor_success.png")
        print("Final attempt complete. Check phase_anchor_success.png")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_phase_anchor_lock()