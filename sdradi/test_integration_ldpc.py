import sys
sys.path.append('/Developer/AIsensing/sdradi')
from sdr_video_comm import SDRVideoLink, FECConfig, FECType, LDPC_AVAILABLE
import numpy as np

def test_integration():
    print("=== Testing Integration of PyTorch LDPC in SDRVideoLink ===")
    
    if not LDPC_AVAILABLE:
        print("LDPC not available. Test skipped.")
        return

    # Config
    fec_config = FECConfig(
        enabled=True,
        fec_type=FECType.LDPC,
        code_rate="1/2",
        num_bits_per_symbol=4
    )
    
    # Init Link (Simulation Mode)
    try:
        link = SDRVideoLink(fec_config=fec_config, simulation_mode=True)
        print("SDRVideoLink initialized.")
    except Exception as e:
        print(f"Failed to init Link: {e}")
        return
        
    # Check Backend
    try:
        backend = link.fec_codec.backend
        print(f"LDPC Backend: {backend}")
        if backend != 'torch':
            print("WARNING: Not using PyTorch backend!")
    except AttributeError:
        print("Could not determine backend.")

    # Test Encode/Decode flow
    k = link.fec_codec.k
    bits = np.random.randint(0, 2, k)
    
    print("Encoding...")
    encoded = link.fec_codec.encode(bits)
    print(f"Encoded shape: {encoded.shape}")
    
    print("Decoding...")
    decoded = link.fec_codec.decode(encoded)
    
    errors = np.abs(decoded - bits).sum()
    print(f"Errors: {errors}")
    
    if errors == 0:
        print("SUCCESS: Integration Verified.")
    else:
        print("FAILURE: Errors in integration test.")

if __name__ == "__main__":
    test_integration()
