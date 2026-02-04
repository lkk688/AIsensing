
import numpy as np
import sys
import os

# Add path
sys.path.append(os.getcwd())

from sdr_video_comm import OTFSConfig, OTFSTransceiver, QAMModulator, LDPC5GCoder

def test_otfs_loopback():
    print("=== OTFS Loopback Test (Noiseless) ===")
    
    # Config
    cfg = OTFSConfig()
    cfg.mod_order = 2 # BPSK
    cfg.pilot_val = 0.0 # Disable pilot for pure data test first
    cfg.ptr_carriers = () # Disable PTRS
    
    print(f"Config: Ns={cfg.N_delay}, Nc={cfg.N_doppler}, Mod={cfg.mod_order}")
    
    tr = OTFSTransceiver(cfg)
    
    # 1. Test Grid Mapping (Symbol Level)
    print("\n[Test 1] Symbol Mapping Consistency")
    Ns, Nc = cfg.N_delay, cfg.N_doppler
    total_symbols = Ns * Nc
    
    # Generate sequential symbols to detect scrambling
    # 0, 1, 2, ... (normalized to constellation later, but let's use integers for tracking)
    # Actually, let's use BPSK bits 0, 1
    
    num_bits = total_symbols * cfg.bits_per_symbol
    tx_bits = np.random.randint(0, 2, num_bits)
    
    # Modulate
    tx_signal = tr.modulate(tx_bits)
    
    # Channel (Identity)
    rx_signal = tx_signal
    
    # Demodulate
    # We need to bypass channel estimation or provide perfect one
    H_perfect = np.ones(Ns, dtype=complex)
    rx_bits, _ = tr.demodulate(rx_signal, channel_est=H_perfect)
    
    # Verify
    # Truncate rx_bits to match tx_bits (ignore PHY padding if any, though here it fits perfectly)
    rx_llrs = rx_bits[:len(tx_bits)]
    
    # LLR < 0 means Bit 1, LLR > 0 means Bit 0
    rx_hard = (rx_llrs < 0).astype(int)
    
    bit_errors = np.sum(np.abs(tx_bits - rx_hard))
    ber = bit_errors / len(tx_bits)
    
    print(f"Tx Bits: {len(tx_bits)}")
    print(f"Rx Bits: {len(rx_hard)}")
    print(f"Bit Errors: {bit_errors}")
    print(f"BER: {ber:.6f}")
    
    if ber == 0:
        print(">> PASSED: Perfect Bit Match")
    else:
        print(">> FAILED: Bit Mismatch")
        # Analyze mismatch pattern
        print("First 20 Tx:", tx_bits[:20])
        print("First 20 Rx:", rx_hard[:20])
        print("First 20 LLR:", rx_llrs[:20])
    
    # 2. Test LDPC Loopback
    print("\n[Test 2] LDPC Loopback")
    # LDPC5GCoder hardcodes k=8192 internally
    ldpc = LDPC5GCoder()
    
    # Generate one block of data
    # LDPC5GCoder expects bits to match K exactly?
    data_bits = np.random.randint(0, 2, 8192)
    
    # Encode
    encoded_bits = ldpc.encode(data_bits)
    print(f"Encoded Size: {len(encoded_bits)}")
    
    # Transmit encoded bits
    tx_sig_ldpc = tr.modulate(encoded_bits)
    
    # Receive
    rx_bits_ldpc, _ = tr.demodulate(tx_sig_ldpc, channel_est=H_perfect)
    
    # LLR Conversion (Demodulate returns Soft bits? No, hard/soft depending on config)
    # Wait, OTFSTransceiver.demodulate currently returns LLRs?
    # Let's check sdr_video_comm.py
    # "frame_bits = self.modulator.soft_demodulate..."
    # Yes, it returns LLRs.
    
    # LDPC Decode
    # Expects LLRs
    decoded_bits = ldpc.decode(rx_bits_ldpc[:len(encoded_bits)])
    
    ldpc_errors = np.sum(np.abs(data_bits - decoded_bits))
    print(f"LDPC Errors: {ldpc_errors}")
    
    if ldpc_errors == 0:
         print(">> PASSED: LDPC Success")
    else:
         print(">> FAILED: LDPC Failure")
         
if __name__ == "__main__":
    test_otfs_loopback()
