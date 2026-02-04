import numpy as np
import adi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# --- OTA DEBUG CONFIG ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 1e6 
N, CP = 64, 32 
PILOT_SC = np.array([-26, -18, -10, -2, 2, 10, 18, 26]) 
PILOT_VALS = np.ones(8) + 0j
DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0 and sc not in PILOT_SC])

def run_link_investigator():
    try:
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate, sdr.tx_lo, sdr.rx_lo = int(FS), int(FC), int(FC)
        sdr.tx_hardwaregain_chan0, sdr.rx_hardwaregain_chan0 = -10, 45
        sdr.rx_buffer_size = 2**18

        # 1. Build Burst Waveform
        zc_len = 1024
        zc = np.exp(-1j * np.pi * 29 * np.arange(zc_len) * (np.arange(zc_len) + 1) / zc_len)
        preamble = np.concatenate([zc, zc]).astype(np.complex64)
        
        np.random.seed(42)
        tx_bits = np.random.randint(0, 2, len(DATA_SC) * 10 * 2)
        qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
        
        tx_payload = []
        for i in range(10):
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

        # 2. Sync & Sliding Window Search
        rx_no_dc = rx - np.mean(rx)
        corr = np.abs(np.correlate(rx_no_dc, zc, mode='valid'))
        c_idx = np.argmax(corr)
        
        # SEARCH: Test ±10 samples around the peak to find the tightest constellation
        best_var = float('inf')
        best_idx = c_idx + 2*zc_len
        
        for offset in range(-10, 11):
            test_idx = c_idx + 2*zc_len + offset
            # Grab one symbol
            Y = np.fft.fftshift(np.fft.fft(rx_no_dc[test_idx+CP : test_idx+CP+N]))
            Yp = Y[(PILOT_SC + N//2) % N]
            # Metric: Pilot Magnitude Variance (Timing alignment affects power)
            var = np.var(np.abs(Yp))
            if var < best_var:
                best_var, best_idx = var, test_idx

        # 3. Final Equalization with LS
        all_syms = []
        Y_final = np.fft.fftshift(np.fft.fft(rx_no_dc[best_idx+CP : best_idx+CP+N]))
        Hp = Y_final[(PILOT_SC + N//2) % N] / PILOT_VALS
        H_full = np.interp(np.arange(N), (PILOT_SC + N//2) % N, Hp)
        
        # 4. EXPORT INTERNAL STATES
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # A. Channel Magnitude (Flatness Check)
        axes[0,0].plot(np.abs(H_full)); axes[0,0].set_title("Channel Magnitude (LS)")
        
        # B. Channel Phase (Timing Slope Check)
        axes[0,1].plot(np.unwrap(np.angle(H_full))); axes[0,1].set_title("Unwrapped Phase (Slope)")
        
        # C. Constellation (Post-Equalization)
        Y_eq = Y_final / (H_full + 1e-12)
        # Software Gain Control
        Y_eq /= np.mean(np.abs(Y_eq))
        axes[1,0].scatter(Y_eq.real, Y_eq.imag, marker='.', color='red')
        axes[1,0].set_title(f"Diagnostic Constellation (Best Offset: {best_idx-c_idx-2*zc_len})")
        axes[1,0].axis([-2, 2, -2, 2]); axes[1,0].grid(True)
        
        # D. Correlation Peak (Zoomed)
        axes[1,1].plot(corr[c_idx-50:c_idx+50]); axes[1,1].set_title("Sync Peak Stability")
        
        plt.tight_layout(); plt.savefig("link_investigator_results.png")
        print(f"Investigation Complete. Check link_investigator_results.png")

    except Exception as e:
        print(f"Link Error: {e}")

if __name__ == "__main__":
    run_link_investigator()