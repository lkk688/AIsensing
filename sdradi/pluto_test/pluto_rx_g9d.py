import numpy as np
import adi
import matplotlib.pyplot as plt

def run_recovery_v5():
    sdr = adi.Pluto(uri="usb:1.37.5"); sdr.rx_buffer_size = 2**22
    # Lock to the best known CFO of -9500 Hz
    rx_raw = sdr.rx() / 2**14
    rx_cfo = rx_raw * np.exp(-1j * 2 * np.pi * -9500 / 1e6 * np.arange(len(rx_raw)))
    
    zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 17 * np.arange(32) * (np.arange(32)+1) / 32)) * 10.0
    c_idx = np.argmax(np.abs(np.correlate(rx_cfo[20000:80000], zc_ref, mode='valid'))) + 20000
    
    DATA_SC = np.array([-12, -11, 11, 12]); PILOT_IDX = np.array([-14, 14])
    data_start = c_idx + (50 * 32) + 32
    recovered_bits, last_syms = [], None
    ph_acc, freq_acc, sco_acc = 0, 0, 0
    
    # Internal Audit Logs
    const, ph_res_log, bit_angles = [], [], []

    for i in range(300):
        idx = data_start + i*64 + sco_acc
        int_idx = int(np.floor(idx))
        frac = idx - int_idx
        sym = (1-frac)*rx_cfo[int_idx : int_idx+32] + frac*rx_cfo[int_idx+1 : int_idx+33]
        
        # AGGRESSIVE Phase Tracker
        Y = np.fft.fftshift(np.fft.fft(sym * np.exp(-1j * (ph_acc + freq_acc))))
        p_vals = Y[(PILOT_IDX + 16) % 32]
        p_err = np.angle(np.mean(p_vals))
        
        freq_acc += 0.04 * p_err # Increased Integral Gain
        ph_acc += 0.25 * p_err + freq_acc # Increased Proportional Gain
        sco_acc += 0.008 * np.angle(p_vals[1] * np.conj(p_vals[0]))
        
        curr_data = Y[(DATA_SC + 16) % 32] * np.exp(-1j * p_err)
        # AGC: Normalize constellation
        curr_data /= (np.mean(np.abs(curr_data)) + 1e-12)
        
        const.extend(curr_data)
        ph_res_log.append(np.degrees(p_err))
        
        if last_syms is not None:
            diff = curr_data * np.conj(last_syms)
            angles = (np.angle(diff) + np.pi/4) % (2*np.pi)
            bit_angles.append(angles)
            bits = (angles // (np.pi/2)).astype(int)
            for b in bits: recovered_bits.extend([int(b >> 1), int(b & 1)])
        last_syms = curr_data

    # Final Audit Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(np.real(const), np.imag(const), s=5); axes[0].set_title("1. NORM-CONSTELLATION")
    axes[1].hist(np.array(bit_angles).flatten(), bins=100); axes[1].set_title("2. ANGLE HISTOGRAM (NEED 4 PEAKS)")
    plt.savefig("v5_audit.png")

    with open("recovered_final.jpg", "wb") as f: f.write(np.packbits(recovered_bits))
    print("🏁 Recovery v5 complete. Check v5_audit.png for the 4 spikes!")

if __name__ == "__main__": run_recovery_v5()