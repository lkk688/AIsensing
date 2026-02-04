
import numpy as np
import matplotlib.pyplot as plt
from sdr_video_comm import SDRVideoLink, SDRConfig, OFDMConfig, WaveformType

def test_fine_sync():
    print("=== Testing Fine Synchronization & CFO Correction ===")
    
    # Setup
    sdr_cfg = SDRConfig()
    ofdm_cfg = OFDMConfig(fft_size=64, cp_length=16, num_symbols=14)
    # Increase sync threshold to avoid false positives in noise
    ofdm_cfg.sync_threshold = 20.0
    
    link = SDRVideoLink(
        sdr_config=sdr_cfg,
        ofdm_config=ofdm_cfg,
        waveform=WaveformType.OFDM,
        simulation_mode=True
    )
    
    # 1. Generate Signal
    # Encodes random bits into OFDM frames + Preamble
    num_bits = 1000
    bits = np.random.randint(0, 2, num_bits)
    tx_signal = link.transmit(bits)  # normalize=True inside? No, transmit returns raw.
    
    # 2. Add Impairments
    print("\n--- Applying Impairments ---")
    
    # A. Carrier Frequency Offset (CFO)
    # e.g., 2 kHz offset. Fs=2e6. 
    # Normalized offset = 2e3 / 2e6 = 1e-3
    cfo_hz = 2000.0 
    fs = sdr_cfg.fs
    t = np.arange(len(tx_signal)) / fs
    cfo_vec = np.exp(1j * 2 * np.pi * cfo_hz * t)
    rx_signal = tx_signal * cfo_vec
    print(f"Injected CFO: {cfo_hz} Hz")
    
    # B. Integer Delay
    delay_samples = 150
    rx_signal = np.concatenate([np.zeros(delay_samples, dtype=complex), rx_signal, np.zeros(100, dtype=complex)])
    print(f"Injected Integer Delay: {delay_samples} samples")
    
    # D. Noise (High SNR)
    snr_db = 30
    p_sig = np.mean(np.abs(rx_signal)**2)
    p_noise = p_sig / (10**(snr_db/10))
    noise = (np.random.randn(len(rx_signal)) + 1j*np.random.randn(len(rx_signal))) * np.sqrt(p_noise/2)
    rx_signal += noise
    print(f"Injected Noise: SNR {snr_db} dB")
    
    # 3. Synchronization (Coarse + CFO)
    print("\n--- Running Synchronization ---")
    synced_signal, sync_metrics = link._synchronize(rx_signal)
    
    if not sync_metrics.get('sync_success'):
        print(f"FAIL: Sync failed to lock. Peak: {sync_metrics.get('peak_val', 0):.2f}")
        return
        
    print(f"Sync Locked! Peak: {sync_metrics['peak_val']:.1f}")
    print(f"Est CFO: {sync_metrics['cfo_est']/1000:.2f} kHz (Error: {abs(sync_metrics['cfo_est'] - cfo_hz):.2f} Hz)")
    
    # Verify CFO accuracy
    if abs(sync_metrics['cfo_est'] - cfo_hz) > 200: # 200Hz tolerance
        print("WARNING: CFO estimation inaccurate!")
    else:
        print("PASS: CFO estimation accurate.")
        
    # 4. Demodulation (Includes Fine Channel Est)
    print("\n--- Demodulation ---")
    rx_bits, metrics = link.transceiver.demodulate(synced_signal)
    
    # Check pilots delay estimate
    # We can't easily access the internal 'est_delay' print from demodulate unless we modified it.
    # But we can call estimate_delay manually on the synced signal.
    
    # Note: synced_signal matches the start of the preamble.
    # But demodulate expects the payload part? 
    # Wait, examine _synchronize return.
    # It returns "corrected" signal. Does it strip preamble?
    # Let's check sdr_video_comm.py logic.
    # Line 2026: return signal_corrected[peak_idx + len(preamble):]
    # So it strips preamble.
    
    # To test estimate_delay, we need the first symbol (with pilots).
    # OFDMTransceiver.estimate_delay expects one symbol or more.
    est_delay = link.transceiver.estimate_delay(synced_signal)
    print(f"Fine Timing Est (Phase Slope): {est_delay:.2f} samples")
    
    # Ideally, if integer sync was perfect, this should be near 0.
    # If modulation worked, BER should be low.
    
    min_len = min(len(bits), len(rx_bits))
    errors = np.sum(bits[:min_len] != rx_bits[:min_len])
    ber = errors / min_len
    print(f"BER: {ber:.5f} ({errors} errors)")
    
    if ber < 0.01:
        print("PASS: Low BER achieved.")
    else:
        print("FAIL: High BER.")

if __name__ == "__main__":
    test_fine_sync()
