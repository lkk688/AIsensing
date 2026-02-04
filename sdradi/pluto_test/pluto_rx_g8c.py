import numpy as np
import adi

# --- FINAL PRODUCTION ALIGNER ---
URI_RX = "usb:1.32.5"
N, CP = 64, 16
DATA_SC = np.array([-10, -9, 9, 10]) 
PILOT_IDX = np.array([-12, 12])

def find_jpeg_header(bits):
    """Searches for FF D8 FF in the bitstream across all possible 1-bit shifts."""
    target = np.array([1,1,1,1,1,1,1,1, 1,1,0,1,1,0,0,0, 1,1,1,1,1,1,1,1], dtype=np.uint8)
    for shift in range(32): # Search up to 4 symbols of bit-offset
        test_bits = bits[shift:]
        if len(test_bits) < 24: continue
        # Simple cross-correlation for bit-alignment
        if np.array_equal(test_bits[:24], target):
            return shift
    return None

def run_universal_search_rx():
    sdr = None
    try:
        sdr = adi.Pluto(uri=URI_RX); sdr.rx_buffer_size = 2**21
        rx_raw = sdr.rx() / 2**14
        
        # 1. PEAK-SEARCH SYNC & CFO
        zc_ref = np.fft.ifft(np.exp(-1j * np.pi * 31 * np.arange(N) * (np.arange(N) + 1) / N)) * 8
        corr = np.abs(np.correlate(rx_raw, zc_ref, mode='valid'))
        c_idx = np.argmax(corr[10000:]) + 10000 
        f_lock = np.angle(np.sum(np.conj(rx_raw[c_idx:c_idx+N]) * rx_raw[c_idx+N:c_idx+2*N])) / N
        rx_p = rx_raw * np.exp(-1j * f_lock * np.arange(len(rx_raw)))
        
        # 2. LTF CHANNEL PINNING
        ltf_idx = c_idx + (12 * N) + CP
        np.random.seed(42); ltf_ref = np.fft.ifftshift(np.random.choice([1, -1], N).astype(complex))
        ltf_ref[range(-7, 8)] = 0
        H_est = np.fft.fft(rx_p[ltf_idx:ltf_idx+N]) / (ltf_ref + 1e-12)

        # 3. FULL DECODE
        data_start = ltf_idx + (2 * N) + CP
        raw_bit_pool = []
        phase_acc, freq_acc, last_syms = 0, 0, None

        for i in range(190):
            idx = int(data_start + i*(N+CP))
            if idx+N+CP > len(rx_p): break
            sym_clean = rx_p[idx+CP : idx+CP+N] - np.mean(rx_p[idx+CP : idx+CP+N])
            Y = np.fft.fftshift(np.fft.fft(sym_clean * np.exp(-1j * phase_acc))) / (np.fft.fftshift(H_est) + 1e-12)
            
            p_err = np.angle(np.mean(Y[(PILOT_IDX + N//2) % N]))
            freq_acc += 0.025 * p_err; phase_acc += 0.18 * p_err + freq_acc
            curr_data = Y[(DATA_SC + N//2) % N] * np.exp(-1j * p_err)
            
            if last_syms is not None:
                diff = curr_data * np.conj(last_syms)
                angles = (np.angle(diff) + np.pi/4) % (2*np.pi)
                bits = (angles // (np.pi/2)).astype(int)
                for b in bits: raw_bit_pool.extend([int(b >> 1), int(b & 1)])
            last_syms = curr_data

        # 4. UNIVERSAL ALIGNMENT SEARCH
        bit_array = np.array(raw_bit_pool, dtype=np.uint8)
        best_shift = find_jpeg_header(bit_array)
        
        if best_shift is not None:
            print(f"🎯 ALIGNMENT FOUND! Shift: {best_shift} bits.")
            final_bits = bit_array[best_shift:]
            out_bytes = np.packbits(final_bits)
            print(f"Header (Hex): {' '.join([f'{b:02X}' for b in out_bytes[:8]])}")
            with open("recovered_final.jpg", "wb") as f: f.write(out_bytes)
            print("🏁 Final Image saved as 'recovered_final.jpg'.")
        else:
            print("❌ No JPEG header found in any bit-shift. SNR may be too low.")
            # Print current state for final debug
            out_bytes = np.packbits(bit_array)
            print(f"Current Raw Header: {' '.join([f'{b:02X}' for b in out_bytes[:8]])}")

    except Exception as e: print(f"❌ ERROR: {e}")
    finally: 
        if sdr: sdr.rx_destroy_buffer()

if __name__ == "__main__": run_universal_search_rx()