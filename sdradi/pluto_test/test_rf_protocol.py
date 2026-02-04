
import sys
import os
import numpy as np

# Add folder to path to import the script
sys.path.append("/Developer/AIsensing/sdradi")

# Import the module (assuming it's safe to import without running main)
# We need to suppress the argparse behavior if it runs main on import? 
# The script uses `if __name__ == "__main__":` so it should be safe.
import sdradi.pluto_test.rf_image_transfer as rf

def test_packet_chain():
    print("Testing Packet Chain (Build -> Parse)...")
    
    payload = b"Hello World! This is a test payload."
    seq = 42
    total = 100
    
    # Build
    bits, frame_len = rf.build_packet_bits(seq, total, payload, repeat=1)
    print(f"  Encoded Bits: {len(bits)} (Frame Len: {frame_len} bytes)")
    
    # Convert bits back to bytes (simulate perfect channel)
    # The receiver function `parse_packet_data` expects BYTES (from `packbits` result usually?)
    # Wait, `parse_packet_data` inputs `bits_bytes`.
    # `build_packet_bits` returns `bits` (numpy array of 0s and 1s).
    # We need to simulate the "demodulator" converting these bits to packed bytes.
    # rf.bits_to_bytes does this.
    
    rx_bytes = rf.bits_to_bytes(bits)
    print(f"  RX Bytes: {rx_bytes[:20]}...")
    
    # Parse
    valid, r_seq, r_total, r_payload = rf.parse_packet_data(rx_bytes)
    
    if not valid:
        print("  FAIL: Packet validation failed.")
        return False
        
    if r_seq != seq:
        print(f"  FAIL: Seq mismatch ({r_seq} != {seq})")
        return False
        
    if r_payload != payload:
        print(f"  FAIL: Payload mismatch ({r_payload} != {payload})")
        return False
        
    print("  PASS: Packet chain verified.")
    return True

def test_chunking():
    print("Testing Chunking...")
    data = b"1234567890" * 10 # 100 bytes
    chunk_size = 32
    
    chunks = list(rf.create_chunks(data, chunk_size))
    
    if len(chunks) != 4: # 32, 32, 32, 4
        print(f"  FAIL: Expected 4 chunks, got {len(chunks)}")
        return False
        
    # Reassemble
    reassembled = b""
    for s, t, d in chunks:
        reassembled += d
        
    if reassembled != data:
        print("  FAIL: Data mismatch")
        return False
        
    print("  PASS: Chunking verified.")
    return True

if __name__ == "__main__":
    t1 = test_packet_chain()
    t2 = test_chunking()
    
    if t1 and t2:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("TESTS FAILED")
        sys.exit(1)
