import numpy as np
from sdr_video_comm import OFDMTransceiver, OFDMConfig

def test_pilot_consistency():
    print("=== Pilot Consistency Test ===")
    
    # Instance 1
    print("Creating Transceiver A...")
    tx_a = OFDMTransceiver()
    pilots_a = tx_a.pilot_symbols
    indices_a = tx_a.pilot_indices
    print(f"Pilots A (First 5): {pilots_a[:5]}")
    print(f"Indices A (First 5): {indices_a[:5]}")
    
    # Generate some random usage of np to mess up state
    np.random.rand(100)
    
    # Instance 2
    print("\nCreating Transceiver B...")
    tx_b = OFDMTransceiver()
    pilots_b = tx_b.pilot_symbols
    indices_b = tx_b.pilot_indices
    print(f"Pilots B (First 5): {pilots_b[:5]}")
    
    # Compare
    if np.allclose(pilots_a, pilots_b):
        print("\n[PASS] Pilots match.")
    else:
        print("\n[FAIL] Pilots mismatch!")
        print(f"Diff: {np.sum(pilots_a != pilots_b)}")

    # Ideal Channel Test
    print("\n=== Ideal Channel Equalization Test ===")
    # Simulate TX -> RX with perfect channel (H=1)
    # If pilots match, H_est should be 1.0 + 0j
    
    # Create simple freq domain symbol
    cfg = tx_a.config
    fft_size = cfg.fft_size
    
    # "Received" pilots = "Transmitted" pilots (H=1)
    rx_pilots = pilots_a # Perfect
    
    # Receiver B estimates channel
    # h = rx / known_pilots
    h_est_pilots = rx_pilots / tx_b.pilot_symbols
    
    print(f"Mean Channel Est (Target 1.0): {np.mean(h_est_pilots)}")
    
    if np.allclose(h_est_pilots, 1.0):
        print("[PASS] Channel Estimation Logic Correct.")
    else:
        print("[FAIL] Channel Estimation Logic Inverted/Wrong.")
        print(f"Sample: {h_est_pilots[:5]}")

if __name__ == "__main__":
    test_pilot_consistency()
