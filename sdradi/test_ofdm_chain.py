import numpy as np
import matplotlib.pyplot as plt
from sdr_video_comm import OFDMTransceiver, OFDMConfig

def test_chain():
    # Setup
    cfg = OFDMConfig(fft_size=64, cp_length=16)
    tr = OFDMTransceiver(cfg)
    
    # Data
    bits = np.random.randint(0, 2, 1000)
    
    # Modulate
    tx_sig = tr.modulate(bits)
    
    # Channel Effect (Rotation + Attenuation + Noise + Usage of Preamble)
    # 45 degree rotation, 0.5 amplitude
    channel = 0.5 * np.exp(1j * np.pi/4)
    rx_sig = tx_sig * channel
    
    # Add noise (High SNR)
    rx_sig += (np.random.randn(len(rx_sig)) + 1j*np.random.randn(len(rx_sig))) * 0.01
    
    # Inject TIMING OFFSET
    delay = 123
    rx_sig_delayed = np.concatenate([np.zeros(delay, dtype=complex), rx_sig, np.zeros(100, dtype=complex)])
    
    print(f"Injecting Delay: {delay} samples")
    
    # We must use the TRANSPORTER's receive() logic which calls _synchronize()
    # BUT tr.demodulate() is low level. 
    # We need to manually call _synchronize logic here or instantiate SDRVideoLink?
    # SDRVideoLink is hard to instantiate due to SDR dependencies.
    # Let's inspect sdr_video_comm.py to pull the `_synchronize` method or make a test helper.
    # Actually, SDRVideoLink has logic. Let's try to verify _synchronize isolation.
    # Since we can't easily import private methods without the class, 
    # Let's assume the user wants end-to-end.
    
    # Instead, let's use the SDRVideoLink class in SIMULATION mode? 
    # (The previous test_video_sim.py did this).
    # Let's update test_ofdm_chain.py to use SDRVideoLink's full chain including sync.
    
    from sdr_video_comm import SDRVideoLink, SDRConfig
    
    sim_link = SDRVideoLink(SDRConfig())
    # Bypass SDR
    sim_link.sdr_config.device = "sim" 
    
    print("Synchronizing...")
    # Manually call internal sync
    rx_synced, sync_meta = sim_link._synchronize(rx_sig_delayed)
    
    if not sync_meta['sync_success']:
        print("FAIL: Sync failed to find preamble")
        # continue testing demod anyway to see
    else:
        print(f"Sync Success! Peak: {sync_meta['peak_val']:.1f}")
        
    # Demodulate (Blind - Should Fail or be poor without EQ)
    print("Demodulating...")
    rx_bits_blind, metrics_blind = tr.demodulate(rx_synced)
    print(f"Blind SNR: {metrics_blind['snr_est']:.2f} dB")
    
    # Was: rx_bits_blind, metrics_blind = tr.demodulate(rx_sig)
    # Replaced by synced version above.
    pass
    
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
