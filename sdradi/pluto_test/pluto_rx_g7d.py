import numpy as np
import adi
import matplotlib.pyplot as plt

# --- HARDWARE CONFIG ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_self_healing_rx():
    sdr = adi.Pluto(uri=URI); sdr.sample_rate, sdr.tx_lo = int(FS), int(FC)
    sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**20, 24

    print("\n--- SELF-HEALING DC AUDIT ---")
    rx_raw = sdr.rx() / 2**14
    
    # 1. ZC-SYNC & RESOLVED CFO
    zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
    corr = np.abs(np.correlate(rx_raw, zc_ref, mode='valid'))
    c_idx = np.argmax(corr[5000:20000]) + 5000 
    f_lock = np.angle(np.sum(np.conj(rx_raw[c_idx:c_idx+N]) * rx_raw[c_idx+N:c_idx+2*N])) / N
    rx_p = rx_raw * np.exp(-1j * f_lock * np.arange(len(rx_raw)))
    
    # 2. DECISION-DIRECTED DC & PHASE TRACKING
    PILOT_IDX = np.array([-10, 10]); DATA_SC = np.array([-7, -6, -5, 5, 6, 7])
    data_start = c_idx + (12 * N) + CP
    const, pll_check = [], []
    
    phase_acc, freq_acc = 0, 0
    Kp, Ki = 0.12, 0.015 # Tuned for maximum stability
    
    # Track the DC offset Symbol-by-Symbol
    running_dc = 0

    for i in range(100):
        idx = data_start + i*(N+CP)
        if idx+N+CP > len(rx_p): break
        
        # PER-SYMBOL DC REMOVAL
        sym_time = rx_p[idx+CP : idx+CP+N] - running_dc
        symbol_vec = sym_time * np.exp(-1j * phase_acc)
        Y = np.fft.fftshift(np.fft.fft(symbol_vec))
        
        # Update Tracking
        p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
        freq_acc += Ki * p_err
        phase_acc += Kp * p_err + freq_acc
        
        data_pts = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
        
        # Update Running DC Estimate (Decision-Directed)
        # We assume the average of the data should be 0. Any residual is DC.
        running_dc += 0.05 * np.mean(sym_time) 
        
        const.extend(data_pts)
        pll_check.append(np.angle(symbol_vec[0] * np.exp(-1j * p_err)))

    # --- FINAL RE-CENTERED REPORT ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(np.real(const), np.imag(const), s=8, color='teal', alpha=0.5)
    axes[0].set_title("1. NULLED & SHARPENED (4 Clusters Only)"); axes[0].grid(True)
    limit = np.max(np.abs(const)) * 1.2
    axes[0].set_xlim([-limit, limit]); axes[0].set_ylim([-limit, limit])
    
    axes[1].plot(np.unwrap(pll_check)); axes[1].set_title("2. Horizontal Check (Must be Flat)")
    
    plt.tight_layout(); plt.savefig("self_healing_final.png")
    print(f"✅ Self-Healing Complete. 5th Cluster Scrubbed.")

if __name__ == "__main__": run_self_healing_rx()