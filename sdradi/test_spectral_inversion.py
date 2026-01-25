import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sdr_video_comm import (
    SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, 
    WaveformType, FECType
)

def test_spectral_inversion():
    print("=== Spectral Inversion Diagnostic ===")
    
    # 1. Config
    sdr_config = SDRConfig.load_from_json()
    sdr_config.fs = 1e6 # 1 MSPS
    sdr_config.bandwidth = 1e6
    
    ofdm_config = OFDMConfig(mod_order=4) # QPSK
    fec_config = FECConfig(enabled=True, fec_type=FECType.REPETITION, repetitions=7)
    
    link = SDRVideoLink(
        sdr_config=sdr_config,
        ofdm_config=ofdm_config,
        fec_config=fec_config,
        waveform=WaveformType.OFDM
    )
    
    if not link.connect_sdr():
        return

    # 2. Text Payload
    text = "Diag Test 12345" * 10
    tx_bits = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))
    encoded_bits = link.fec_codec.encode(tx_bits)
    tx_signal = link.transceiver.modulate(encoded_bits)
    preamble = link._generate_preamble()
    full_tx = np.concatenate([preamble, tx_signal, np.zeros(200)])
    
    # 3. Transmit
    print("Transmitting...")
    link.sdr.SDR_TX_send(full_tx, cyclic=True)
    time.sleep(1.0)
    
    # 4. Receive
    print("Receiving...")
    link.sdr.SDR_RX_setup(n_SAMPLES=len(full_tx)*4)
    rx_signal = link.sdr.SDR_RX_receive()
    
    # Double buffer for cyclic safety
    rx_signal_doubled = np.concatenate([rx_signal, rx_signal])
    
    link.sdr.SDR_TX_stop()
    
    # 5. Dual Test
    print("\n--- TEST 1: Normal Signal ---")
    try_demod(link, rx_signal_doubled, tx_bits, label="Normal")
    
    print("\n--- TEST 2: Conjugated (Inverted) Signal ---")
    # Conjugate in time domain = Flip in freq domain (Spectral Inversion fix)
    try_demod(link, np.conj(rx_signal_doubled), tx_bits, label="Inverted")

def try_demod(link, signal, tx_bits, label):
    try:
        # Sync
        payload = link._synchronize(signal)
        
        # Trim/Pad
        expected_len = link.transceiver._tx_bits_count // 2 # Rough est, actually transceiver doesn't know exact symbols now?
        # Actually link.transceiver.modulate stores _tx_bits_count but that was local.
        # We need to trust demodulate to handle whatever length payload is.
        # But _synchronize returns specific length? No, returns rest of buffer.
        
        # Demod
        demod_bits, metrics = link.transceiver.demodulate(payload)
        decoded = link.fec_codec.decode(demod_bits)
        
        # BER
        min_len = min(len(tx_bits), len(decoded))
        if min_len == 0:
            print(f"[{label}] No bits decoded.")
            return
            
        bit_errs = np.sum(tx_bits[:min_len] != decoded[:min_len])
        ber = bit_errs / min_len
        
        print(f"[{label}] BER: {ber:.5f} ({bit_errs} errs)")
        print(f"[{label}] SNR: {metrics.get('snr_est_db', -99):.1f} dB")
        
        if ber < 0.1:
            rx_bytes = np.packbits(decoded[:min_len]).tobytes()
            print(f"[{label}] Text: {rx_bytes[:20]}...")
            print(f"!!! SUCCESS with {label} !!!")
            
    except Exception as e:
        print(f"[{label}] Failed: {e}")

if __name__ == "__main__":
    test_spectral_inversion()
