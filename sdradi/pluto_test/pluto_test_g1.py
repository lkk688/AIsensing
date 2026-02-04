import numpy as np
import adi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# --- DIAGNOSTIC CONFIG ---
IP, FC, FS = "ip:192.168.3.2", 2300e6, 2e6 
N, CP = 64, 16 
ALL_USED = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])

def run_pathologist_v3():
    try:
        print(f"Connecting to Pluto at {IP}...")
        sdr = adi.Pluto(uri=IP)
        sdr.sample_rate = int(FS)
        sdr.tx_lo = int(FC)
        sdr.rx_lo = int(FC)
        
        # Increased gains to ensure preamble visibility above DC noise
        sdr.tx_hardwaregain_chan0 = -10 
        sdr.rx_hardwaregain_chan0 = 25
        sdr.rx_buffer_size = 131072 

        # 1. Waveform Construction
        zc_len = 1024
        zc = np.exp(-1j * np.pi * 29 * np.arange(zc_len) * (np.arange(zc_len) + 1) / zc_len)
        preamble = np.concatenate([zc, zc]).astype(np.complex64)
        
        np.random.seed(42)
        bits = np.random.randint(0, 2, len(ALL_USED) * 14 * 2)
        qpsk = ((1 - 2*bits[::2]) + 1j*(1 - 2*bits[1::2])) / np.sqrt(2)
        
        tx_payload = []
        for i in range(14):
            X = np.zeros(N, dtype=complex)
            X[(ALL_USED + N//2) % N] = qpsk[i*len(ALL_USED) : (i+1)*len(ALL_USED)]
            x_t = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
            tx_payload.append(np.concatenate([x_t[-CP:], x_t]))
        
        tx_frame = np.concatenate([preamble, *tx_payload])
        
        # 2. Execution with Buffer Flushing
        sdr.tx_cyclic_buffer = True
        sdr.tx((tx_frame * 0.5 * 2**14).astype(np.complex64))
        
        print("Flushing and Capturing...")
        for _ in range(3): _ = sdr.rx() # Clear stale DMA buffers
        rx = sdr.rx()
        sdr.tx_destroy_buffer()

        if rx is None or len(rx) == 0:
            raise ValueError("SDR returned empty buffer.")

        # 3. INTERNAL STATE ANALYSIS
        # Remove DC to improve correlation peak
        rx_no_dc = rx - np.mean(rx)
        corr = np.abs(np.correlate(rx_no_dc, zc, mode='valid'))
        
        # Calculate Peak-to-Average Power Ratio (PAPR) of the correlation
        peak_val = np.max(corr)
        avg_val = np.mean(corr)
        peak_idx = np.argmax(corr)

        # 4. DIAGNOSTIC PLOTTING
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # A. Time Domain Energy (Capture Check)
        axes[0,0].plot(np.abs(rx[:5000]))
        axes[0,0].set_title("Time Domain Magnitude (Capture Envelope)")
        
        # B. Full Correlation Surface (Sync Check)
        axes[0,1].plot(corr)
        axes[0,1].set_title(f"Correlation Surface (Peak Index: {peak_idx})")
        axes[0,1].axhline(avg_val * 10, color='r', linestyle='--', label='10x Mean Threshold')
        
        # C. CFO Check (Repeated ZC Phase)
        if peak_idx + 2*zc_len < len(rx):
            r1 = rx[peak_idx : peak_idx+zc_len]
            r2 = rx[peak_idx+zc_len : peak_idx+2*zc_len]
            phase_diff = np.angle(r2 * np.conj(r1))
            axes[1,0].plot(phase_diff)
            axes[1,0].set_title("Inter-ZC Phase Difference (CFO Diagnostic)")

        # D. Power Spectrum (LO Leakage Check)
        axes[1,1].specgram(rx, Fs=FS)
        axes[1,1].set_title("Spectrogram (LO/Leakage Stability)")
        
        plt.tight_layout()
        plt.savefig("pathologist_v3_results.png")
        
        # Final Guard against empty FFT
        if peak_val < avg_val * 5:
            raise ValueError(f"Sync failed. Peak {peak_val:.1f} too low compared to mean {avg_val:.1f}")

        print(f"Diagnostics saved to pathologist_v3_results.png. Peak Index: {peak_idx}")

    except Exception as e:
        print(f"Pathologist V3 Error: {e}")

if __name__ == "__main__":
    run_pathologist_v3()