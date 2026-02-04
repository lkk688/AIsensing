import numpy as np
import adi
import matplotlib.pyplot as plt

# --- HARDWARE CONFIG ---
URI, FC, FS = "usb:1.31.5", 2300e6, 1e6 
N, CP = 64, 16 

def run_high_torque_rx():
    sdr = adi.Pluto(uri=URI); sdr.sample_rate, sdr.tx_lo = int(FS), int(FC)
    sdr.rx_buffer_size, sdr.rx_hardwaregain_chan0 = 2**20, 24 # Slight gain boost to fight the noise floor

    print("\n--- HIGH-TORQUE PI TRACKER LOG ---")
    rx = sdr.rx() / 2**14
    
    # 1. ZC-SYNC (Robust against the 4.1kHz measured offset)
    zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * np.sqrt(N)
    corr = np.abs(np.correlate(rx, zc_ref, mode='valid'))
    c_idx = np.argmax(corr[5000:20000]) + 5000 
    
    # 2. RESOLVED CFO LOCK
    f_lock = np.angle(np.sum(np.conj(rx[c_idx:c_idx+N]) * rx[c_idx+N:c_idx+2*N])) / N
    rx_p = rx * np.exp(-1j * f_lock * np.arange(len(rx)))
    
    # 3. PI TRACKING LOOP (Handling the 22 deg/sym Acceleration)
    PILOT_IDX = np.array([-10, 10]); DATA_SC = np.array([-7, -6, -5, 5, 6, 7])
    data_start = c_idx + (12 * N) + CP
    const, velocity_log, pll_check = [], [], []
    
    phase_accum, frequency_accum = 0, 0
    # Higher gains to "catch" the runaway drift seen in Plot 2
    Kp, Ki = 0.15, 0.02 

    for i in range(100):
        idx = data_start + i*(N+CP)
        if idx+N+CP > len(rx_p): break
        
        # Apply current phase and frequency estimates
        symbol_vec = rx_p[idx+CP : idx+CP+N] * np.exp(-1j * (phase_accum))
        Y = np.fft.fftshift(np.fft.fft(symbol_vec))
        
        # Phase Error from Ultra-Leash Pilots
        p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
        
        # PI Loop Update: Tracking the acceleration
        frequency_accum += Ki * p_err
        phase_accum += Kp * p_err + frequency_accum
        
        data_pts = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
        const.extend(data_pts)
        velocity_log.append(np.degrees(frequency_accum))
        pll_check.append(np.angle(symbol_vec[0] * np.exp(-1j * p_err)))

    # --- PI TRACKER DIAGNOSTIC ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].scatter(np.real(const), np.imag(const), s=8, color='blue', alpha=0.5)
    axes[0,0].set_title("1. PI-TRACKED CONSTELLATION"); axes[0,0].grid(True)
    
    axes[0,1].plot(velocity_log); axes[0,1].set_title("2. Frequency Integrator (Catching the Ramp)")
    axes[1,0].plot(np.unwrap(pll_check)); axes[1,0].set_title("3. PLL Final Flatness (Target: Horizontal)")
    
    axes[1,1].bar(np.arange(-32, 32), np.abs(np.fft.fftshift(np.fft.fft(rx_p[c_idx:c_idx+N]))))
    axes[1,1].set_title("4. DC Hole Audit")
    
    plt.tight_layout(); plt.savefig("high_torque_report.png")
    print(f"✅ PI-Lock complete. Final Frequency Integrator: {velocity_log[-1]:.2f} deg/sym")

if __name__ == "__main__": run_high_torque_rx()