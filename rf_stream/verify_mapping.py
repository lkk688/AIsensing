
import numpy as np
import sys
import os

# Import from the actual files
sys.path.append(os.getcwd())
import rf_stream_tx_step5phy as tx
import rf_stream_rx_step5phy_v2 as rx

def test_mapping():
    print("Testing bit mapping consistency between TX and RX...")
    
    # 1. Generate all possible 2-bit combinations
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            bits = np.array([b0, b1], dtype=np.uint8)
            
            # Map on TX
            syms = tx.qpsk_map(bits)
            
            # Demap on RX
            bits_rx = rx.qpsk_demap(syms)
            
            match = np.all(bits == bits_rx)
            print(f"Bits [{b0}, {b1}] -> Sym {syms[0]:.3f} -> RX Bits {bits_rx} -> Match: {match}")
            if not match:
                print("  ERROR: Mapping mismatch!")

    # 2. Test a full byte
    byte_val = 0xAD
    bits = np.unpackbits(np.array([byte_val], dtype=np.uint8))
    syms = tx.qpsk_map(bits)
    bits_rx = rx.qpsk_demap(syms)
    byte_rx = np.packbits(bits_rx)[0]
    print(f"Byte 0x{byte_val:02X} -> TX bits {bits} -> RX bits {bits_rx} -> Byte 0x{byte_rx:02X}")
    if byte_val != byte_rx:
        print("  ERROR: Byte mismatch!")
    else:
        print("  SUCCESS: Byte match!")

if __name__ == "__main__":
    test_mapping()
