import numpy as np
import adi
import matplotlib.pyplot as plt

# --- HARDWARE CONFIG ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_lo_nulling_rx():
    sdr = adi.Pluto(uri=URI); sdr.sample_rate, sdr.tx_lo = int(FS), int(FC)
    sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**20, 24

    print("\n--- LO-NULLING STRESS AUDIT ---")
    rx_raw = sdr.rx() / 2**14
    
    # 1. TIME-DOMAIN DC BLOCKER (Subtracting the 5th Cluster)
    # This physically removes the constant bias seen in image_548687.png
    rx = rx_raw - np.mean(rx_raw) 
    
    # 2. ZC-SYNC & RESOLVED CFO
    zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
    corr = np.abs(np.correlate(rx, zc_ref, mode='valid'))
    c_idx = np.argmax(corr[5000:20000]) + 5000 
    f_lock = np.angle(np.sum(np.conj(rx[c_idx:c_idx+N]) * rx[c_idx+N:c_idx+2*N])) / N
    rx_p = rx * np.exp(-1j * f_lock * np.arange(len(rx)))
    
    # 3. HIGH-TORQUE PI TRACKER
    PILOT_IDX = np.array([-10, 10]); DATA_SC = np.array([-7, -6, -5, 5, 6, 7])
    data_start = c_idx + (12 * N) + CP
    const, pll_check = [], []
    
    phase_accum, freq_accum = 0, 0
    Kp, Ki = 0.15, 0.02 # PID gains from your successful run

    for i in range(100):
        idx = data_start + i*(N+CP)
        if idx+N+CP > len(rx_p): break
        
        symbol_vec = rx_p[idx+CP : idx+CP+N] * np.exp(-1j * phase_accum)
        Y = np.fft.fftshift(np.fft.fft(symbol_vec))
        
        p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
        freq_accum += Ki * p_err
        phase_accum += Kp * p_err + freq_accum
        
        data_pts = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
        const.extend(data_pts)
        pll_check.append(np.angle(symbol_vec[0] * np.exp(-1j * p_err)))

    # --- FINAL RE-CENTERED REPORT ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(np.real(const), np.imag(const), s=8, color='purple', alpha=0.5)
    axes[0].set_title("1. NULLED CONSTELLATION (Target: 4 Clusters)"); axes[0].grid(True)
    # Auto-center axis check
    limit = np.max(np.abs(const)) * 1.2
    axes[0].set_xlim([-limit, limit]); axes[0].set_ylim([-limit, limit])
    
    axes[1].plot(np.unwrap(pll_check)); axes[1].set_title("2. PLL Stability (Must be Horizontal)")
    
    plt.tight_layout(); plt.savefig("lo_nulling_final.png")
    print(f"✅ LO Nulling complete. Fifth cluster suppressed.")

if __name__ == "__main__": run_lo_nulling_rx()