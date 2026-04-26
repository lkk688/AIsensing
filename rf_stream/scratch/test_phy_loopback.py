
import numpy as np
import zlib
import matplotlib.pyplot as plt

# Import from the RX/TX files
import rf_stream_tx_step5phy as tx
import rf_stream_rx_step5phy_v2 as rx

def test_loopback():
    print("Testing PHY Loopback...")
    
    # 1. Create a packet
    payload = b"Hello World Loopback Test " * 10
    seq = 123
    total = 1
    
    # MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC32(4)
    body = rx.MAGIC + seq.to_bytes(2, 'little') + total.to_bytes(2, 'little') + len(payload).to_bytes(2, 'little') + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    pkt_bytes = body + crc.to_bytes(4, 'little')
    
    # 2. Convert to samples
    fs = 3e6
    stf = tx.create_schmidl_cox_stf(6)
    ltf = tx.create_ltf(4)
    pkt_samples = tx.bytes_to_ofdm_samples(
        pkt_bytes, 1, stf, ltf, fs, 0, 0, 0, 0, 1.0
    )
    
    print(f"Generated packet: {len(pkt_samples)} samples")
    
    # 3. Add channel effects
    # CFO
    fs = 3e6
    cfo_hz = 50000.0 # 50 kHz
    t = np.arange(len(pkt_samples)) / fs
    pkt_rx = pkt_samples * np.exp(1j * 2 * np.pi * cfo_hz * t)
    
    # Noise
    pkt_rx += (np.random.randn(len(pkt_rx)) + 1j*np.random.randn(len(pkt_rx))) * 0.05
    
    # Pad with noise
    rx_buf = np.zeros(100000, dtype=np.complex64)
    start_idx = 10000
    rx_buf[start_idx : start_idx+len(pkt_rx)] = pkt_rx
    
    # 4. Receive
    rx_cfg = rx.RxConfig(uri="", fc=2.3e9, fs=fs, stf_repeats=6, ltf_symbols=4, verbose=True)
    
    # Manually trigger detection
    stf_ref = rx.create_schmidl_cox_stf(rx_cfg.stf_repeats).astype(np.complex64)
    ltf_ref, ltf_freq_ref = rx.create_ltf_ref(rx_cfg.ltf_symbols)
    
    # Try detection
    xc = rx.cross_correlate_ncc(rx_buf, stf_ref)
    best_stf = np.argmax(xc)
    print(f"STF detection: peak={xc[best_stf]:.3f} at {best_stf}")
    
    # Process
    ok, reason, seq_rx, total_rx, payload_rx, evm, bb = rx.try_demod_at(
        rx_buf, best_stf, rx_cfg, stf_ref, ltf_ref, ltf_freq_ref
    )
    
    print(f"Demod result: ok={ok} reason={reason} evm={evm:.3f}")
    if ok:
        print(f"Payload match: {payload == payload_rx}")
    else:
        print(f"Fallback hex: {bb[:16].hex()}")

if __name__ == "__main__":
    test_loopback()
