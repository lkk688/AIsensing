import numpy as np
import adi
import matplotlib.pyplot as plt
import time

def run_iio_flush_loopback():
    sdr = None
    try:
        print("Initiating IIO-Level Flush...")
        # Use a fresh context and a smaller buffer to reduce DMA strain
        sdr = adi.Pluto("ip:192.168.3.2")
        sdr.sample_rate = int(1e6)
        sdr.rx_buffer_size = 2**17 # 128k samples
        sdr._ctrl.debug_attrs['loopback'].value = '1' # Digital Loopback

        # 1. Baseband Math (Simplified for verification)
        N, CP = 64, 16
        DATA_SC = np.array([sc for sc in range(-26, 27) if sc != 0])
        np.random.seed(42)
        tx_bits = np.random.randint(0, 2, len(DATA_SC) * 20 * 2)
        qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
        
        # Preamble for Sync
        X = np.zeros(N, dtype=complex)
        X[(DATA_SC + N//2) % N] = qpsk[:len(DATA_SC)]
        preamble = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
        preamble_t = np.concatenate([preamble[-CP:], preamble])
        
        # Massive zero padding to help DMA stabilization
        full_tx = np.concatenate([np.zeros(10000, dtype=complex), preamble_t, np.zeros(5000, dtype=complex)])
        
        # 2. TRIGGERED DATA POLL
        # We use a lower amplitude (0.4) to stay within safe hardware limits
        sdr.tx_cyclic_buffer = True 
        sdr.tx((full_tx * 0.4 * 2**14).astype(np.complex64))
        
        print("Polling IIO Pipe...")
        rx_data = None
        for attempt in range(100):
            temp_rx = sdr.rx()
            pwr = np.mean(np.abs(temp_rx)**2)
            if pwr > 1e-9:
                print(f"✅ IIO Flush Success! Signal Power: {10*np.log10(pwr):.1f} dB (Attempt {attempt})")
                rx_data = temp_rx
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("IIO Pipe Blocked: Hardware reset required.")

        sdr.tx_destroy_buffer()

        # 3. Math Verification
        rx_no_dc = rx_data - np.mean(rx_data)
        corr = np.abs(np.correlate(rx_no_dc, preamble_t, mode='valid'))
        c_idx = np.argmax(corr[5000:]) + 5000
        
        # FFT on the aligned symbol
        rx_aligned = rx_no_dc[c_idx + CP : c_idx + CP + N]
        Y = np.fft.fftshift(np.fft.fft(rx_aligned)) / np.sqrt(N)
        Y_data = Y[(DATA_SC + N//2) % N]
        
        # Calculate Phase Error (Should be 0.0 in loopback)
        phase_err = np.angle(Y_data / qpsk[:len(DATA_SC)])
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1,2,1); plt.scatter(Y_data.real, Y_data.imag, color='orange'); plt.grid(True)
        plt.title("Math Verification Constellation")
        plt.subplot(1,2,2); plt.plot(phase_err); plt.title("Residual Phase (rad)")
        plt.savefig("iio_flush_math.png")
        
        print(f"Verified Sync Index: {c_idx}")
        print(f"Max Phase Error: {np.max(np.abs(phase_err)):.4f} rad")

    except Exception as e:
        print(f"Diagnostic Error: {e}")
    finally:
        if sdr: del sdr # Force context release

if __name__ == "__main__":
    run_iio_flush_loopback()