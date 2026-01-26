import numpy as np
import matplotlib.pyplot as plt
from sdr_video_comm import OFDMTransceiver, OFDMConfig

def test_chain():
    # Setup
    cfg = OFDMConfig(num_subcarriers=64, cyclic_prefix_len=16)
    tr = OFDMTransceiver(cfg)
    
    # Data
    bits = np.random.randint(0, 2, 1000)
    
    # Modulate
    tx_sig = tr.modulate(bits)
    
    # Channel Effect (Rotation + Attenuation + Noise)
    # 45 degree rotation, 0.5 amplitude
    channel = 0.5 * np.exp(1j * np.pi/4)
    rx_sig = tx_sig * channel
    
    # Add noise (High SNR)
    rx_sig += (np.random.randn(len(rx_sig)) + 1j*np.random.randn(len(rx_sig))) * 0.01
    
    # Demodulate (Blind - Should Fail or be poor without EQ)
    print("Demodulating without EQ...")
    rx_bits_blind, metrics_blind = tr.demodulate(rx_sig)
    print(f"Blind SNR: {metrics_blind['snr_est']:.2f} dB")
    
    # Manual Channel Est (Perfect)
    # H_est should be array of 'channel'.
    # In time domain? No demodulate expects freq domain est if passed.
    # But wait, demodulate() takes channel_est.
    # Let's see if we can implement internal estimation.
    
    print("\nSimulating 'channel_est' logic...")
    # Does demodulate support extracting pilots? 
    # Current codebase: demodulate() calculates nothing.
    
    # We need to test the FIX.
    
if __name__ == "__main__":
    test_chain()
