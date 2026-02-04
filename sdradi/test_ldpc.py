import time
import numpy as np
import sys
import os

# Ensure we can import sdr_video_comm and sdr_ldpc
sys.path.append(os.getcwd())

from sdr_video_comm import LDPC5GCoder, FECConfig, FECType

def test_ldpc():
    print("Testing LDPC5GCoder...")
    
    # Config
    cfg = FECConfig(enabled=True, fec_type=FECType.LDPC, code_rate="1/2")
    
    try:
        coder = LDPC5GCoder(cfg)
    except Exception as e:
        print(f"Failed to init: {e}")
        return

    # Random Data
    k = coder.k
    bits = np.random.randint(0, 2, k).astype(int)
    
    print(f"Encoding {k} bits...")
    t0 = time.time()
    encoded = coder.encode(bits)
    t1 = time.time()
    print(f"Encoded {len(encoded)} bits in {t1-t0:.4f}s")
    
    # Noise
    # Flip a few bits
    rx_bits = encoded.copy()
    rx_bits[0] = 1 - rx_bits[0] 
    rx_bits[100] = 1 - rx_bits[100]
    
    print(f"Decoding {len(rx_bits)} bits...")
    t2 = time.time()
    decoded = coder.decode(rx_bits)
    t3 = time.time()
    print(f"Decoded {len(decoded)} bits in {t3-t2:.4f}s")
    
    # Verify
    errors = np.sum(bits != decoded)
    print(f"Errors: {errors}")
    
    if errors == 0:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_ldpc()
