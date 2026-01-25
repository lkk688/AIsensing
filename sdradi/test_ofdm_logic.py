import numpy as np
from sdr_video_comm import OFDMTransceiver, OFDMConfig, QAMModulator

def test_ofdm_logic():
    print("=== OFDM Logic Unit Test (Software Loopback) ===")
    
    # 1. Setup
    # Use exact config from video test
    cfg = OFDMConfig(mod_order=4, fft_size=64, num_data_carriers=48, num_pilot_carriers=8)
    tr = OFDMTransceiver(cfg)
    
    print(f"Config: FFT={cfg.fft_size}, Used={len(tr.data_indices)+len(tr.pilot_indices)}")
    print(f"Data Indices: {tr.data_indices}")
    print(f"Pilot Indices: {tr.pilot_indices}")
    
    # 2. Random Data
    # 10 frames worth of data
    bits_per_frame = len(tr.data_indices) * cfg.num_symbols * cfg.bits_per_symbol
    total_bits = bits_per_frame * 10
    
    tx_bits = np.random.randint(0, 2, total_bits)
    
    # 3. Modulate
    tx_signal = tr.modulate(tx_bits)
    print(f"Modulated Signal: {len(tx_signal)} samples")
    print(f"Sample Power: {np.mean(np.abs(tx_signal)**2):.4f}")
    
    # 4. Channel (Identity)
    rx_signal = tx_signal
    
    # 5. Demodulate
    rx_bits, metrics = tr.demodulate(rx_signal)
    
    # 6. Verify
    # We might have padding bits in rx_bits if length mismatches slightly?
    # Truncate to tx length
    n = min(len(tx_bits), len(rx_bits))
    
    errors = np.sum(tx_bits[:n] != rx_bits[:n])
    ber = errors / n
    
    print(f"BER: {ber:.6f} ({errors} errors)")
    print(f"SNR Est: {metrics['snr_est_db']:.2f} dB")
    
    if ber == 0.0:
        print("[PASS] Logic is perfect.")
    else:
        print("[FAIL] Logic is broken.")
        
        # Debug: Check First Symbol
        print("\n--- Debugging First Symbol ---")
        # Extract first symbol manually
        sym_len = cfg.fft_size + cfg.cp_length
        s0_time = tx_signal[cfg.cp_length : cfg.cp_length + cfg.fft_size]
        s0_freq = np.fft.fft(s0_time) / np.sqrt(cfg.fft_size)
        
        # Check Pilots
        rx_pilots = s0_freq[tr.pilot_indices]
        h_est = rx_pilots / tr.pilot_symbols
        print(f"Pilots (Tx): {tr.pilot_symbols[:5]}")
        print(f"Pilots (Rx): {rx_pilots[:5]}")
        print(f"H_est (First 5): {h_est[:5]}")
        
        # Check Data
        # Re-modulate first few bits
        first_bits = tx_bits[:cfg.bits_per_symbol] # 2 bits
        # Manually modulate
        # indices = 1*b0 + 2*b1 ? (Little Endian)
        b0 = first_bits[0]
        b1 = first_bits[1]
        idx = b0 + (b1 << 1)
        expected_sym = tr.modulator.constellation[idx]
        
        # Where is it mapped?
        # modulate() puts it at data_indices[0]?
        # data_symbols = modulator.modulate(frame_bits) .reshape(...)
        # data_indices[0] gets data_symbols[0]
        
        rx_sym_0 = s0_freq[tr.data_indices[0]]
        print(f"Bits[0:2]: {first_bits} -> Index {idx}")
        print(f"Expected Constellation: {expected_sym}")
        print(f"Received Frequency Val: {rx_sym_0}")
        
        if np.abs(rx_sym_0 - expected_sym) < 1e-1:
            print("Symbol 0 matches.")
        else:
            print("Symbol 0 MISMATCH.")

if __name__ == "__main__":
    test_ofdm_logic()
