import torch
import numpy as np
import sys
import os

# Add paths
sys.path.append('/Developer/AIsensing/sdradi')
sys.path.append('/Developer/AIsensing')

from sdr_ldpc import LDPC5GEncoder, LDPC5GDecoder

def test_ldpc_encoding_decoding():
    print("=== Testing LDPC PyTorch Implementation ===")
    
    # Parameters
    k = 100
    n = 200 # Rate 1/2
    batch_size = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    # Init Encoder
    try:
        encoder = LDPC5GEncoder(k, n, device=device)
    except Exception as e:
        print(f"Failed to init encoder: {e}")
        return

    # Init Decoder
    decoder = LDPC5GDecoder(encoder, max_iter=20, device=device)
    
    # Generate Info Bits
    info_bits = torch.randint(0, 2, (batch_size, k), device=device).float()
    
    # Encode
    print("Encoding...")
    codeword = encoder.encode(info_bits)
    print(f"Codeword shape: {codeword.shape}")
    
    # Check simple properties
    # Since we don't have the TF implementation loaded here easily without complex setup,
    # we first check self-consistency: Encoder -> Decoder (No Noise) -> Original
    
    print("Decoding (No Noise)...")
    # LLRs: 0 -> +Large, 1 -> -Large
    # c=0 -> +10, c=1 -> -10
    llrs = (1 - 2 * codeword) * 10.0
    
    decoded_bits = decoder.decode(llrs)
    
    # BER
    errors = torch.abs(decoded_bits - info_bits).sum()
    ber = errors / (batch_size * k)
    print(f"BER (No Noise): {ber.item()}")
    
    if ber.item() == 0:
        print("SUCCESS: Zero errors on clean channel.")
    else:
        print("FAILURE: Errors found on clean channel.")
        print(f"First 10 decoded: {decoded_bits[0, :10]}")
        print(f"First 10 original: {info_bits[0, :10]}")

    # Add Noise
    print("\nDecoding (With Noise SNR=5dB)...")
    snr_db = 5.0
    snr_lin = 10**(snr_db/10)
    # LLR magnitude approx 2 * channel_val * 4 / N0? 
    # For BPSK/QPSK: y = x + n. LLR = 2y/sigma^2. sigma^2 = 1/(2*R*SNR).
    # Approx LLR = y. (Scale doesn't matter for Min-Sum usually, but sign does)
    
    # Modulate BPSK: 0->+1, 1->-1
    # x = 1 - 2 * c
    x = 1 - 2 * codeword
    
    noise_std = np.sqrt(1 / (2 * (k/n) * snr_lin))
    noise = torch.randn_like(x) * noise_std
    y = x + noise
    
    # LLRs for Decoder
    llrs_noisy = 2 * y / (noise_std**2)
    
    decoded_bits_noisy = decoder.decode(llrs_noisy)
    errors_noisy = torch.abs(decoded_bits_noisy - info_bits).sum()
    ber_noisy = errors_noisy / (batch_size * k)
    print(f"BER (SNR {snr_db}dB): {ber_noisy.item()}")

if __name__ == "__main__":
    test_ldpc_encoding_decoding()
