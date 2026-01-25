import sys
import time
import numpy as np
from sdr_video_comm import (
    SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, 
    WaveformType, FECType
)

def test_short_packet():
    print("=== Short Packet Reliability Test ===")
    
    # 1. Config
    sdr_config = SDRConfig.load_from_json()
    # Override for stability (Use 3M to satisfy AD9363 min limits)
    sdr_config.fs = 3e6
    sdr_config.bandwidth = 3e6
    
    sdr_config.bandwidth = 1e6
    
    
    # Use Conservatively Reduced Bandwidth to avoid Filter Roll-off
    # Default is 48+8=56 (87.5%). Let's try 20 (31%).
    ofdm_config = OFDMConfig(mod_order=4, num_data_carriers=16, num_pilot_carriers=4)
    # fec_config = FECConfig(enabled=True, fec_type=FECType.REPETITION, repetitions=7)
    fec_config = FECConfig(enabled=False)
    
    link = SDRVideoLink(
        sdr_config=sdr_config,
        ofdm_config=ofdm_config,
        fec_config=fec_config,
        waveform=WaveformType.OFDM
    )
    
    if not link.connect_sdr():
        return

    # 2. Text Payload
    text = "Hello World! This is a short test packet to verify sync." * 5 # ~250 chars
    print(f"Payload: {len(text)} chars")
    
    # Text -> Bits
    import binascii
    tx_bytes = text.encode('utf-8')
    # Pad to ensure byte alignment if needed? Codec methods?
    # Manual bit conversion
    tx_bits = np.unpackbits(np.frombuffer(tx_bytes, dtype=np.uint8))
    
    # Encode
    encoded_bits = link.fec_codec.encode(tx_bits)
    tx_signal = link.transceiver.modulate(encoded_bits)
    print(f"Symbols: {len(tx_signal)}")
    
    # Preamble
    preamble = link._generate_preamble()
    full_tx = np.concatenate([preamble, tx_signal, np.zeros(100)])
    
    # 3. Transmit
    print("Transmitting...")
    link.sdr.SDR_TX_send(full_tx, cyclic=True)
    time.sleep(0.5)
    
    # 4. Receive
    print("Receiving...")
    # FLUSH BUFFERS (Critical for Stale Data check)
    print("Flushing RX buffers...")
    for _ in range(5):
        _ = link.sdr.SDR_RX_receive()
        
    # Real Capture
    link.sdr.SDR_RX_setup(n_SAMPLES=len(full_tx)*4)
    rx_signal = link.sdr.SDR_RX_receive()
    link.sdr.SDR_TX_stop()
    
    # 5. Process
    # Handle cyclic wrap by doubling buffer
    rx_signal = np.concatenate([rx_signal, rx_signal])
    
    payload = link._synchronize(rx_signal)
    
    # Trim
    if len(payload) > len(tx_signal):
        payload = payload[:len(tx_signal)]
    elif len(payload) < len(tx_signal):
        print("Payload too short.")
        # Pad
        payload = np.concatenate([payload, np.zeros(len(tx_signal)-len(payload))])
        
    bits, metrics = link.transceiver.demodulate(payload)
    decoded = link.fec_codec.decode(bits)
    
    # BER
    min_len = min(len(tx_bits), len(decoded))
    bit_errs = np.sum(tx_bits[:min_len] != decoded[:min_len])
    ber = bit_errs / min_len
    
    print(f"BER: {ber:.5f} ({bit_errs} errors)")
    print(f"SNR Est: {metrics['snr_est_db']:.1f} dB")
    
    # Try to decode string
    try:
        rx_bytes = np.packbits(decoded[:min_len]).tobytes()
        print(f"Rx Text: {rx_bytes[:50]}...")
    except:
        print("Decode failed")

if __name__ == "__main__":
    test_short_packet()
