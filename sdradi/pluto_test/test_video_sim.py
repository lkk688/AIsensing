from sdr_video_comm import SDRVideoLink, SDRConfig
import numpy as np

def test_sim():
    print("Initializing SDR Video Link (Sim Mode)...")
    # Force sim mode by invalid IP or letting it fail, but simpler to just use None if supported?
    # The class tries to load 'adi'. If not found, it sets SDR_AVAILABLE=False.
    # On this machine 'adi' might exist or not.
    # Let's use loopback_test() which simulates expected behavior.
    
    cfg = SDRConfig()
    link = SDRVideoLink(cfg)
    
    print("Running Loopback Test...")
    res = link.loopback_test(num_bits=10000)
    print(f"Results: {res}")
    
    if res['ber'] > 0.01:
        print("FAIL: High BER in simulation!")
        exit(1)
    else:
        print("PASS: Low BER in simulation.")

if __name__ == "__main__":
    test_sim()
