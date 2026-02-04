import numpy as np
import adi

# --- FINAL PRODUCTION CONFIG ---
URI_RX = "usb:1.32.5"
N, CP = 64, 16
# Golden Window: Bins +/- 9 and +/- 10
DATA_SC = np.array([-10, -9, 9, 10]) 
PILOT_IDX = np.array([-12, 12])

def run_final_image_recovery():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI_RX)
        sdr.rx_buffer_size = 2**22 # 4MB Buffer for full capture
        print("📡 Capture started. Waiting for the stress-test stream...")
        rx_raw = sdr.rx() / 2**14
        
        # 1. TIMING SYNC (ZC-Sequence)
        zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * 8
        corr = np.abs(np.correlate(rx_raw, zc_ref, mode='valid'))
        c_idx = np.argmax(corr[10000:]) + 10000 
        
        # 2. FREQUENCY LOCK
        f_lock = np.angle(np.sum(np.conj(rx_raw[c_idx:c_idx+N]) * rx_raw[c_idx+N:c_idx+2*N])) / N
        rx_p = rx_raw * np.exp(-1j * f_lock * np.arange(len(rx_raw)))
        
        # 3. PRODUCTION DECODE LOOP
        data_start = c_idx + (12 * N) + CP
        recovered_bits = []
        phase_acc, freq_acc, last_syms = 0, 0, None
        
        print(f"📦 Decoding 190 symbols (CFO: {f_lock*1e6/(2*np.pi):.2f} Hz)...")

        for i in range(190):
            idx = data_start + i*(N+CP)
            # SURGICAL SCRUB: Remove DC bias per symbol
            raw_sym = rx_p[idx+CP : idx+CP+N]
            sym_clean = raw_sym - np.mean(raw_sym) 
            
            symbol_vec = sym_clean * np.exp(-1j * phase_acc)
            Y = np.fft.fftshift(np.fft.fft(symbol_vec))
            
            # Phase Tracking (Alpha/Beta from image_5495e6.jpg)
            p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
            freq_acc += 0.025 * p_err
            phase_acc += 0.18 * p_err + freq_acc
            
            curr_data = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
            
            if last_syms is not None:
                # Differential DQPSK Logic
                diff = curr_data * np.conj(last_syms)
                angles = (np.angle(diff) + np.pi/4) % (2*np.pi)
                bits = (angles // (np.pi/2)).astype(int)
                for b in bits:
                    recovered_bits.extend([int(b >> 1), int(b & 1)])
            
            last_syms = curr_data

        # 4. RECONSTRUCT JPEG
        out_bytes = np.packbits(recovered_bits)
        with open("recovered_stress.jpg", "wb") as f:
            f.write(out_bytes)
        print("🏁 SUCCESS! Image saved as 'recovered_stress.jpg'.")

    except Exception as e: print(f"❌ ERROR: {e}")
    finally:
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__":
    run_final_image_recovery()