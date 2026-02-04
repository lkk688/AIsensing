import numpy as np
import argparse
import sys
import os

# Add local directory to path to import sdr_video_comm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sdr_video_comm import OFDMTransceiver, OFDMConfig, OTFSTransceiver, OTFSConfig, PacketFramer

def test_ofdm_loopback():
    print("\n=== Testing OFDM Loopback (Noiseless) ===")
    
    # Configs to test
    configs = [
        OFDMConfig(mod_order=2, num_symbols=14), # BPSK
        OFDMConfig(mod_order=4, num_symbols=14)  # QPSK
    ]
    
    for cfg in configs:
        mod_name = "BPSK" if cfg.mod_order == 2 else "QPSK"
        print(f"\nTesting OFDM {mod_name}:")
        
        tr = OFDMTransceiver(cfg)
        
        # Generate Random Bits
        bits_per_frame = cfg.bits_per_frame
        tx_bits = np.random.randint(0, 2, bits_per_frame)
        
        # Modulate
        tx_signal = tr.modulate(tx_bits)
        
        # Channel (Identity)
        rx_signal = tx_signal
        
        # Demodulate
        rx_bits, metrics = tr.demodulate(rx_signal)
        
        # Check BER
        # Demodulator might return more bits if it pads?
        L = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:L] != rx_bits[:L])
        ber = errors / L
        
        print(f"  Tx Bits: {len(tx_bits)}")
        print(f"  Rx Bits: {len(rx_bits)}")
        print(f"  Errors : {errors}")
        print(f"  BER    : {ber:.6f}")
        
        if ber == 0.0:
            print("  [PASS] DSP Logic Correct.")
        else:
            print("  [FAIL] DSP Logic Incorrect!")
            # Debug
            print(f"  First 20 Tx: {tx_bits[:20]}")
            print(f"  First 20 Rx: {rx_bits[:20]}")

def test_otfs_loopback():
    print("\n=== Testing OTFS Loopback (Noiseless) ===")
    
    # Configs (Reduced size for speed?)
    # use default size to match realistic usage
    configs = [
        OTFSConfig(mod_order=2), # BPSK
        OTFSConfig(mod_order=4)  # QPSK
    ]
    
    for cfg in configs:
        mod_name = "BPSK" if cfg.mod_order == 2 else "QPSK"
        print(f"\nTesting OTFS {mod_name} (64x256):")
        
        tr = OTFSTransceiver(cfg)
        
        # Generate Random Bits
        bits_per_frame = cfg.bits_per_frame
        tx_bits = np.random.randint(0, 2, bits_per_frame)
        print(f"  Tx Bits: {len(tx_bits)}")
        
        # Modulate
        tx_signal = tr.modulate(tx_bits)
        
        # Channel (Identity)
        # Note: OTFSTransceiver.demodulate expects channel_est for equalization
        # In ideal channel, H = 1.
        rx_signal = tx_signal
        identity_H = np.ones(cfg.N_delay, dtype=complex)
        
        # Demodulate
        rx_bits, metrics = tr.demodulate(rx_signal, channel_est=identity_H)
        
        # Check BER
        L = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:L] != rx_bits[:L])
        ber = errors / L
        
        print(f"  Rx Bits: {len(rx_bits)}")
        print(f"  Errors : {errors}")
        print(f"  BER    : {ber:.6f}")
        
        if ber == 0.0:
            print("  [PASS] DSP Logic Correct.")
        else:
            print("  [FAIL] DSP Logic Incorrect!")
            print(f"  First 20 Tx: {tx_bits[:20]}")
            print(f"  First 20 Rx: {rx_bits[:20]}")

def test_framing():
    print("\n=== Testing PacketFramer ===")
    payload = b"Hello World! This is a test packet."
    seq = 123
    
    frame = PacketFramer.frame(payload, seq)
    print(f"Framed {len(payload)} bytes -> {len(frame)} bytes")
    
    packets = PacketFramer.deframe(frame)
    if len(packets) == 1:
        rx_payload, rx_seq, valid = packets[0]
        if rx_payload == payload and rx_seq == seq and valid:
            print("  [PASS] Deframed correctly.")
        else:
            print(f"  [FAIL] Content mismatch: {rx_payload[:20]}... Seq={rx_seq} Valid={valid}")
    else:
        print(f"  [FAIL] Found {len(packets)} packets.")

def test_otfs_phase_robustness():
    print("\n=== Testing OTFS Phase Robustness (Noiseless + Random Phase) ===")
    
    cfg = OTFSConfig(mod_order=2) # BPSK
    tr = OTFSTransceiver(cfg)
    
    # Generate Random Bits
    bits_per_frame = cfg.bits_per_frame
    tx_bits = np.random.randint(0, 2, bits_per_frame)
    tx_signal = tr.modulate(tx_bits)
    
    # Test different static phase offsets
    offsets = [0, np.pi/4, np.pi/2, np.pi, -np.pi/2]
    
    for phi in offsets:
        print(f"\nPhase Offset: {phi:.2f} rad ({np.degrees(phi):.0f} deg)")
        
        # Channel: Identity * Phase
        rx_signal = tx_signal * np.exp(1j * phi)
        identity_H = np.ones(cfg.N_delay, dtype=complex)
        
        # Demodulate
        rx_bits, metrics = tr.demodulate(rx_signal, channel_est=identity_H)
        
        # Check BER
        L = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:L] != rx_bits[:L])
        ber = errors / L
        
        print(f"  BER: {ber:.6f}")
        
        if ber == 0.0:
            print("  [PASS] Robust.")
        else:
            print("  [FAIL] Phase tracking failed.")

def test_otfs_timing_sensitivity():
    print("\n=== Testing OTFS Timing Sensitivity (Noiseless + Time Shift) ===")
    
    cfg = OTFSConfig(mod_order=2) # BPSK
    tr = OTFSTransceiver(cfg)
    
    # Generate Random Bits
    bits_per_frame = cfg.bits_per_frame
    tx_bits = np.random.randint(0, 2, bits_per_frame)
    tx_signal = tr.modulate(tx_bits)
    
    # Test shifts
    shifts = [0, 1, -1, 2, -2]
    
    for shift in shifts:
        print(f"\nTiming Shift: {shift} samples")
        
        # Channel: Identity shifted
        rx_signal = np.roll(tx_signal, shift)
        # Handle wrap-around for simulation (though signals are long enough usually)
        if shift > 0:
            rx_signal[:shift] = 0
        elif shift < 0:
            rx_signal[shift:] = 0
            
        identity_H = np.ones(cfg.N_delay, dtype=complex)
        
        # Demodulate (Assuming perfect Channel Est but imperfect Timing)
        # Note: In reality, channel est would also be shifted.
        # Here we test if the Demodulator (FFT/SFFT) breaks with offset.
        rx_bits, metrics = tr.demodulate(rx_signal, channel_est=identity_H)
        
        # Check BER
        L = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:L] != rx_bits[:L])
        ber = errors / L
        
        print(f"  BER: {ber:.6f}")
        
        if ber == 0.0:
            print("  [PASS] Robust.")
        else:
            print("  [FAIL] Timing sensitivity.")

def test_otfs_timing_recovery():
    print("\n=== Testing OTFS Timing Recovery (Search Strategy) ===")
    
    cfg = OTFSConfig(mod_order=2) # BPSK
    tr = OTFSTransceiver(cfg)
    
    bits_per_frame = cfg.bits_per_frame
    tx_bits = np.random.randint(0, 2, bits_per_frame)
    tx_signal = tr.modulate(tx_bits)
    
    # Simulate a channel with unknown delay (e.g. +2 samples)
    true_delay = 2
    rx_signal_delayed = np.roll(tx_signal, true_delay)
    # Zero out wrapped part
    rx_signal_delayed[:true_delay] = 0
    
    print(f"Simulated Delay: {true_delay} samples.")
    
    # Search range around assumed 0
    search_range = range(-3, 4)
    found = False
    
    for offset in search_range:
        # We "guess" the start is at 'offset'
        # So we shift BACK by 'offset' to align 0
        # If offset == true_delay, then rx_shifted == tx_signal
        
        # Actually, if we received a signal that is delayed by D,
        # we need to start reading from index D.
        # Here we simulate that by rolling 'rx_signal_delayed' by -offset.
        
        candidate_signal = np.roll(rx_signal_delayed, -offset)
        
        identity_H = np.ones(cfg.N_delay, dtype=complex)
        rx_bits, _ = tr.demodulate(candidate_signal, channel_est=identity_H)
        
        # Check BER
        L = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:L] != rx_bits[:L])
        ber = errors / L
        
        if ber == 0.0:
            print(f"  Offset {offset}: BER={ber:.6f} [MATCH!]")
            found = True
            break
        else:
            print(f"  Offset {offset}: BER={ber:.6f}")
            
    if found:
        print("  [PASS] Timing recovery successful.")
    else:
        print("  [FAIL] Could not recover timing.")

if __name__ == "__main__":
    # test_framing()
    # test_ofdm_loopback()
    # test_otfs_phase_robustness()
    # test_otfs_timing_sensitivity()
    test_otfs_timing_recovery()
