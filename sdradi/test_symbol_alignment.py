import sys
import time
import numpy as np
from sdr_video_comm import (
    SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, 
    WaveformType, FECType
)

def test_symbol_alignment():
    print("=== Symbol Alignment Search ===")
    
    # Setup
    sdr_config = SDRConfig.load_from_json()
    ofdm_config = OFDMConfig(mod_order=4)
    # Use simpler FEC or None to see raw BER? No, Repetition is fine.
    fec_config = FECConfig(enabled=True, fec_type=FECType.REPETITION, repetitions=7)
    
    link = SDRVideoLink(
        sdr_config=sdr_config,
        ofdm_config=ofdm_config,
        fec_config=fec_config,
        waveform=WaveformType.OFDM
    )
    if not link.connect_sdr(): return

    # Payload
    text = "AlignmentTest" * 5
    tx_bits = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))
    encoded_bits = link.fec_codec.encode(tx_bits)
    tx_signal = link.transceiver.modulate(encoded_bits)
    preamble = link._generate_preamble()
    full_tx = np.concatenate([preamble, tx_signal, np.zeros(200)])
    
    # Transmit
    print("Transmitting...")
    link.sdr.SDR_TX_send(full_tx, cyclic=True)
    time.sleep(0.5)
    
    # Receive
    print("Receiving...")
    # 4x buffer
    link.sdr.SDR_RX_setup(n_SAMPLES=len(full_tx)*4)
    rx_signal = link.sdr.SDR_RX_receive()
    link.sdr.SDR_TX_stop()
    rx_signal = np.concatenate([rx_signal, rx_signal]) # Cyclic safe

    # 1. Base Synchronization
    # Get peak index from standard sync
    # We copy code from _synchronize to get the index directly
    
    corr = np.abs(np.correlate(rx_signal, preamble, mode='valid'))
    peak_idx = np.argmax(corr)
    val = corr[peak_idx]
    print(f"Base Peak: {peak_idx} (Val: {val:.1f})")
    
    # 2. Iterate Offsets
    # Symbol Size = 64 + 16 = 80
    # Try sliding window from -2 symbols to +2 symbols
    symbol_len = link.ofdm_config.fft_size + link.ofdm_config.cp_length
    
    offsets_to_test = range(-symbol_len * 2, symbol_len * 2 + 1, 10) # Steps of 10 samples
    
    best_ber = 1.0
    best_offset = 0
    
    print("\nScanning Offsets (Samples)...")
    for offset in offsets_to_test:
        # Proposed Start
        start = peak_idx + len(preamble) + offset
        if start < 0 or start + len(tx_signal) > len(rx_signal):
            continue
            
        payload = rx_signal[start : start + len(tx_signal)]
        
        # Demod
        try:
            bits, _ = link.transceiver.demodulate(payload)
            decoded = link.fec_codec.decode(bits)
            min_len = min(len(tx_bits), len(decoded))
            if min_len == 0: continue
            
            diff = (tx_bits[:min_len] != decoded[:min_len])
            ber = np.sum(diff) / min_len
            
            if ber < 0.4:
                print(f"Offset {offset}: BER {ber:.4f} !!! FOUND SIGNAL !!!")
            elif ber < 0.48:
                print(f"Offset {offset}: BER {ber:.4f} (Interesting?)")
                
            if ber < best_ber:
                best_ber = ber
                best_offset = offset
                
        except:
            pass
            
    print(f"\nBest Offset: {best_offset} samples (BER {best_ber:.4f})")
    if best_ber > 0.4:
        print("Scanned range failed. Sync Peak might be total garbage or data inverted.")

if __name__ == "__main__":
    test_symbol_alignment()
