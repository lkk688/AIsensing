import numpy as np
import adi
import matplotlib.pyplot as plt

URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_stress_rx():
    sdr = adi.Pluto(uri=URI); sdr.sample_rate, sdr.tx_lo = int(FS), int(FC)
    sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**20, 22 # Minimal gain for purity

    print("\n--- DEEP-STATE STRESS AUDIT ---")
    rx = sdr.rx() / 2**14
    
    # 1. ZC-SYNC & CFO
    zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
    corr = np.abs(np.correlate(rx, zc_ref, mode='valid'))
    c_idx = np.argmax(corr[10000:]) + 10000 
    f_lock = np.angle(np.sum(np.conj(rx[c_idx:c_idx+N]) * rx[c_idx+N:c_idx+2*N])) / N
    rx_p = rx * np.exp(-1j * f_lock * np.arange(len(rx)))
    
    # 2. DATA TRACKING (Golden Window)
    PILOT_IDX = np.array([-10, 10]); DATA_SC = np.array([-7, -6, -5, 5, 6, 7])
    data_start = c_idx + (12 * N) + CP
    const, velocity_log, evm_log, pll_check = [], [], [], []
    
    phase_accum, velocity = 0, 0
    alpha, beta = 0.06, 0.006 # Aggressive P/I for stress tracking

    for i in range(100): # STRESS: Track 100 symbols
        idx = data_start + i*(N+CP)
        if idx+N+CP > len(rx_p): break
        
        # Sample-Level Rotation
        symbol_vec = rx_p[idx+CP : idx+CP+N] * np.exp(-1j * (phase_accum + velocity * np.arange(N)))
        Y = np.fft.fftshift(np.fft.fft(symbol_vec))
        
        p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
        phase_accum += alpha * p_err
        velocity += beta * p_err # Momentum update
        
        data_pts = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
        const.extend(data_pts)
        velocity_log.append(np.degrees(velocity))
        pll_check.append(np.angle(symbol_vec[0]))
        # EVM calculation
        ideal = np.sign(data_pts.real) + 1j*np.sign(data_pts.imag)
        evm_log.append(10*np.log10(np.mean(np.abs(data_pts - ideal)**2) + 1e-12))

    # --- 6-PLOT STRESS REPORT ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes[0,0].scatter(np.array(const).real, np.array(const).imag, s=5, color='darkred', alpha=0.5)
    axes[0,0].set_title("1. STRESS CONSTELLATION"); axes[0,0].grid(True)
    
    axes[0,1].plot(velocity_log); axes[0,1].set_title("2. Phase Velocity (Deg/Symbol) - Must be Stable")
    
    axes[1,0].plot(np.unwrap(pll_check)); axes[1,0].set_title("3. PLL Continuity (Horizontal Check)")
    
    axes[1,1].bar(np.arange(-32, 32), np.abs(np.fft.fftshift(np.fft.fft(rx_p[c_idx:c_idx+N]))))
    axes[1,1].set_title("4. DC Hole Audit (Check Spike Stability)")
    
    axes[2,0].plot(evm_log); axes[2,0].set_title("5. EVM Stability (dB) - Lower is Better")
    
    axes[2,1].plot(corr[c_idx-500:c_idx+500]); axes[2,1].set_title("6. Sync Sharpness (Timing Jitter Check)")
    
    plt.tight_layout(); plt.savefig("stress_test_report.png")
    print(f"✅ Stress Report saved. Final CFO: {f_lock*FS/(2*np.pi):.2f} Hz. EVM: {np.mean(evm_log):.2f} dB")

if __name__ == "__main__": run_stress_rx()