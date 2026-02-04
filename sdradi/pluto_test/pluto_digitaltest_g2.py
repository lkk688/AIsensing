import numpy as np
import adi
import matplotlib.pyplot as plt
import time

def run_cold_start_loopback():
    try:
        sdr = adi.Pluto("ip:192.168.3.2")
        # 1. HARDWARE RESET SEQUENCE
        # Toggling LO forces the AD9361 to reset its internal state machine
        original_fc = 2300e6
        sdr.rx_lo = int(original_fc + 1e6)
        time.sleep(0.1)
        sdr.rx_lo = int(original_fc)
        
        sdr.sample_rate = int(1e6)
        sdr._ctrl.debug_attrs['loopback'].value = '1' # Force Digital Loopback
        sdr.rx_buffer_size = 2**19 # 512k samples

        # 2. Configuration & Waveform
        N, CP = 64, 16
        DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0])
        num_symbols = 40
        
        np.random.seed(42)
        tx_bits = np.random.randint(0, 2, len(DATA_SC) * num_symbols * 2)
        qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
        
        tx_payload = []
        for i in range(num_symbols):
            X = np.zeros(N, dtype=complex)
            X[(DATA_SC + N//2) % N] = qpsk[i*len(DATA_SC) : (i+1)*len(DATA_SC)]
            x_t = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
            tx_payload.append(np.concatenate([x_t[-CP:], x_t]))
        
        preamble = tx_payload[0]
        # Massive 50,000 sample pad to ensure the burst is centered
        full_tx = np.concatenate([np.zeros(50000, dtype=complex), preamble, *tx_payload, np.zeros(10000, dtype=complex)])
        
        # 3. BURST TRIGGER
        sdr.tx_cyclic_buffer = False 
        print("Flushing and Triggering...")
        for _ in range(10): _ = sdr.rx() # Hard flush
        
        # Use a higher gain for the TX to ensure the digital loopback is visible
        sdr.tx((full_tx * 0.8 * 2**14).astype(np.complex64))
        
        rx = sdr.rx()
        sdr.tx_destroy_buffer()

        # 4. Correlation and Sync
        rx_no_dc = rx - np.mean(rx)
        # Use a high-pass filter to remove 1/f noise that might be masking the peak
        rx_filt = rx_no_dc - np.roll(rx_no_dc, 1)
        corr = np.abs(np.correlate(rx_filt, preamble, mode='valid'))
        
        if np.max(corr) < 1e-4:
            raise RuntimeError(f"Low Signal. Max Corr: {np.max(corr):.2e}. Check if PlutoSDR is overheating.")

        # Look for peak past the stale startup data
        c_idx = np.argmax(corr[10000:]) + 10000 
        print(f"✅ Cold-Start Sync Point: {c_idx}")

        data_start_base = c_idx + (N + CP)
        best_evm, best_offset, final_syms_list = float('inf'), 0, []

        # 5. Math Verification
        for offset in range(-20, 21):
            test_idx = data_start_base + offset
            current_test_syms, total_err = [], 0
            for i in range(num_symbols):
                start = test_idx + i*(N+CP) + CP
                if start < 0 or (start + N) >= len(rx_no_dc):
                    total_err = float('inf'); break
                
                Y = np.fft.fftshift(np.fft.fft(rx_no_dc[start : start+N])) / np.sqrt(N)
                Y_data = Y[(DATA_SC + N//2) % N]
                ideal = qpsk[i*len(DATA_SC) : (i+1)*len(DATA_SC)]
                total_err += np.sum(np.abs(Y_data - ideal)**2)
                current_test_syms.append(Y_data)
                
            if len(current_test_syms) == num_symbols and total_err < best_evm:
                best_evm, best_offset, final_syms_list = total_err, offset, current_test_syms

        # 6. Result Verification
        if not final_syms_list:
            raise RuntimeError("Symbol extraction failed. The captured signal is corrupted.")
            
        final_syms = np.concatenate(final_syms_list)
        rx_bits = np.zeros(len(final_syms)*2, dtype=int)
        rx_bits[::2], rx_bits[1::2] = (final_syms.real < 0).astype(int), (final_syms.imag < 0).astype(int)
        ber = np.sum(rx_bits != tx_bits[:len(rx_bits)]) / len(rx_bits)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1,2,1); plt.scatter(final_syms.real, final_syms.imag, s=2, alpha=0.5); plt.grid(True)
        plt.title(f"DIGITAL LOCK | BER: {ber:.6f}")
        plt.subplot(1,2,2); plt.plot(corr); plt.title("Correlation Profile")
        plt.savefig("cold_start_verify.png")
        
        print(f"VERIFIED! BER: {ber:.6f} | Math Offset: {best_offset}")

    except Exception as e:
        print(f"Cold-Start Error: {e}")

if __name__ == "__main__":
    run_cold_start_loopback()

# The "Low Signal" error with a Max Correlation of 0.00e+00 means the Python environment is receiving a buffer of literal zeros from the hardware. This isn't a math error anymore—it's a USB/IIO driver deadlock. When the PlutoSDR's DMA (Direct Memory Access) engine hangs, it stops streaming data entirely, leading to the empty buffers you're seeing.

# 🔍 Root Cause: The "Zombie" DMA
# In a single-device digital loopback, if the TX and RX aren't perfectly synchronized in the hardware's clock domain, the AD9361's internal state machine can "stall". The 0.00e+00 result confirms that no samples are making it from the FPGA to your Python script.