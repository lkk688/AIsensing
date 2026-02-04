
import numpy as np
import time
from sdr_video_comm import OTFSConfig, OTFSTransceiver

def test_phy_loopback():
    print("=== PHY Loopback Test (BPSK) ===")
    
    # 1. Config
    conf = OTFSConfig(mod_order=2) # BPSK
    tr = OTFSTransceiver(conf)
    
    # 2. Generate Data (Known Pattern)
    # 0xAA = 10101010
    pattern_len = 1000
    bits = np.tile([1, 0], pattern_len // 2)
    
    print(f"TX Bits: {len(bits)}")
    
    # 3. Modulate
    tx_signal = tr.modulate(bits)
    print(f"TX Signal Power: {np.mean(np.abs(tx_signal)**2):.4f}")
    
    # 4. Channel (Ideal)
    # No noise, no CFO
    msk = np.ones(len(tx_signal), dtype=complex)
    rx_signal = tx_signal * msk
    
    # 5. Demodulate (Returns LLRs)
    rx_llrs, metrics = tr.demodulate(rx_signal)
    
    # LLR < 0 -> 1, LLR > 0 -> 0
    rx_bits = np.where(rx_llrs < 0, 1, 0)
    
    print(f"RX Bits: {len(rx_bits)}")
    
    # 6. Verify
    # Truncate to min
    n = min(len(bits), len(rx_bits))
    
    print(f"TX Bits (first 20): {bits[:20]}")
    print(f"RX Bits (first 20): {rx_bits[:20]}")
    
    errors = np.sum(np.abs(bits[:n] - rx_bits[:n]))
    ber = errors / n
    
    print(f"BER (Ideal): {ber:.6f} ({errors}/{n})")
    
    if ber == 0:
        print("PASS: Ideal Channel")
    else:
        print("FAIL: Ideal Channel")

    # 7. Channel (Noisy)
    print("\n--- Adding Noise (-60dB -> Very Low Noise) ---")
    noise_amp = 10**(-60/20)
    noise = (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))/np.sqrt(2) * noise_amp
    rx_signal_noisy = tx_signal + noise
    
    rx_bits_n, _ = tr.demodulate(rx_signal_noisy)
    n_n = min(len(bits), len(rx_bits_n))
    errors_n = np.sum(np.abs(bits[:n_n] - rx_bits_n[:n_n]))
    ber_n = errors_n / n_n
    
    print(f"BER (Noisy): {ber_n:.6f}")
    if ber_n < 1e-3:
        print("PASS: Noisy Channel")
    else:
        print("FAIL: Noisy Channel")
        
    # 8. Channel with Sync Offset (Simulate Preamble Detection Error)
    print("\n--- Adding Time Offset (+3 samples) ---")
    rx_signal_shifted = np.concatenate([np.zeros(3), tx_signal])
    # Receiver "thinks" it starts at 0, but data is at 3
    # We must simulate the "Robust Synchronizer" extraction
    # Here we just feed it shifted data to see if equalization/demod handles it?
    # No, demod expects frame aligned. Even 1 sample off destroys OFDM/OTFS orthogonality.
    # So we prove that alignment MUST be exact.
    
    rx_bits_s, _ = tr.demodulate(rx_signal_shifted) # This should fail significantly
    n_s = min(len(bits), len(rx_bits_s))
    # We need to align bits to compare
    # But usually it just produces garbage.
    print(f"RX Bits (Shifted): {len(rx_bits_s)}")
    
    # Check if any part matches?
    # Just print BER assuming aligned
    errors_s = np.sum(np.abs(bits[:n_s] - rx_bits_s[:n_s]))
    ber_s = errors_s / n_s
    print(f"BER (Shifted): {ber_s:.6f} (Expected to be high)")

if __name__ == "__main__":
    test_phy_loopback()
