import numpy as np
import adi
import matplotlib.pyplot as plt

URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_deep_glass_rx():
    sdr = adi.Pluto(uri=URI); sdr.sample_rate, sdr.tx_lo = int(FS), int(FC)
    sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**20, 22 

    print("\n--- DEEP GLASS BOX STRESS LOG ---")
    rx_raw = sdr.rx() / 2**14
    
    # 1. ZC-SYNC & CFO LOCK
    zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * 8
    corr = np.abs(np.correlate(rx_raw, zc_ref, mode='valid'))
    c_idx = np.argmax(corr[5000:20000]) + 5000 
    f_lock = np.angle(np.sum(np.conj(rx_raw[c_idx:c_idx+N]) * rx_raw[c_idx+N:c_idx+2*N])) / N
    rx_p = rx_raw * np.exp(-1j * f_lock * np.arange(len(rx_raw)))
    
    # 2. ADAPTIVE STATE TRACKERS
    PILOT_IDX = np.array([-12, 12]); DATA_SC = np.array([-10, -9, 9, 10])
    data_start = c_idx + (12 * N) + CP
    const, vel_log, dc_log, pll_check = [], [], [], []
    
    phase_acc, freq_acc = 0, 0
    running_dc = 0
    # PI Loop Gains (Tuned for 200 symbol duration)
    Kp, Ki = 0.18, 0.025 

    for i in range(200): # 2x Time Duration
        idx = data_start + i*(N+CP)
        if idx+N+CP > len(rx_p): break
        
        # SLIDING WINDOW DC BLOCKER
        raw_sym = rx_p[idx+CP : idx+CP+N]
        running_dc = 0.95 * running_dc + 0.05 * np.mean(raw_sym)
        
        symbol_vec = (raw_sym - running_dc) * np.exp(-1j * phase_acc)
        Y = np.fft.fftshift(np.fft.fft(symbol_vec))
        
        # Phase Error Correction
        p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
        freq_acc += Ki * p_err
        phase_acc += Kp * p_err + freq_acc
        
        data_pts = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
        const.extend(data_pts)
        vel_log.append(np.degrees(freq_acc))
        dc_log.append(np.abs(running_dc))
        pll_check.append(np.angle(symbol_vec[0] * np.exp(-1j * p_err)))

    # --- 6-PANEL STRESS DASHBOARD ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes[0,0].scatter(np.real(const), np.imag(const), s=4, color='green', alpha=0.4)
    axes[0,0].set_title("1. STRESS CONSTELLATION (200 Syms)"); axes[0,0].grid(True)
    
    axes[0,1].plot(vel_log); axes[0,1].set_title("2. Frequency Integrator (Acceleration Track)")
    
    axes[1,0].plot(np.unwrap(pll_check)); axes[1,0].set_title("3. Horizontal Check (MUST BE FLAT)")
    
    axes[1,1].plot(dc_log); axes[1,1].set_title("4. DC Leakage magnitude (LO Drift)")
    
    axes[2,0].hist(np.angle(const), bins=100); axes[2,0].set_title("5. Phase Distribution (Target: 4 Peaks)")
    
    axes[2,1].bar(np.arange(-32, 32), np.abs(np.fft.fftshift(np.fft.fft(rx_p[c_idx:c_idx+N]))))
    axes[2,1].set_title("6. Spectrum Hole Audit (17-SC Notch)")
    
    plt.tight_layout(); plt.savefig("deep_glass_report.png")
    print(f"🏁 Stress test complete. Final Drift: {vel_log[-1]:.2f} deg/sym. DC: {dc_log[-1]:.2e}")

if __name__ == "__main__": run_deep_glass_rx()