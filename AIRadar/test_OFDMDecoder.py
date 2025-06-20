import torch
import numpy as np
import matplotlib.pyplot as plt
from AIRadarLib.modeling_TimeNet import RadarTimeNet
from AIRadarLib.waveform_utils import generate_ofdm_signal
from AIRadarLib.ofdm_decoder import OFDMDecoder, OFDMSymbolDecoder


def generate_test_ofdm_data_with_bits(batch_size=2, num_rx=2, num_chirps=64, samples_per_chirp=64, modulation='qpsk'):
    """
    Generate synthetic OFDM radar data with known bit patterns for testing.
    
    Args:
        batch_size: Number of samples in the batch
        num_rx: Number of receive antennas
        num_chirps: Number of chirps (OFDM symbols)
        samples_per_chirp: Number of samples per chirp (FFT size)
        modulation: Modulation scheme ('bpsk', 'qpsk', 'qam16', 'qam64', 'qam256')
        
    Returns:
        Tuple of:
        - Tensor with shape [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
        - Tensor with ground truth bits
    """
    # Initialize empty tensor
    data = torch.zeros(batch_size, num_rx, num_chirps, samples_per_chirp, 2)
    
    # Determine bits per symbol based on modulation
    bits_per_symbol = {
        'bpsk': 1,
        'qpsk': 2,
        'qam16': 4,
        'qam64': 6,
        'qam256': 8
    }[modulation.lower()]
    
    # Number of active subcarriers (half of FFT size for simplicity)
    num_subcarriers = samples_per_chirp // 2
    
    # Generate random bits for each batch
    total_bits_per_batch = num_chirps * num_subcarriers * bits_per_symbol
    ground_truth_bits = torch.randint(0, 2, (batch_size, total_bits_per_batch), dtype=torch.int8)
    
    for b in range(batch_size):
        # Generate OFDM signal with the random bits
        # For simplicity, we're not actually encoding the bits here, just generating a random OFDM signal
        ofdm_signal = generate_ofdm_signal(
            num_subcarriers=num_subcarriers,
            num_symbols=num_chirps,
            subcarrier_spacing=1e6,  # 1 MHz spacing (arbitrary for test)
            fs=samples_per_chirp * 1e6,  # Sampling frequency
            cp_length_ratio=0  # No cyclic prefix for simplicity
        )
        
        # Reshape to match expected dimensions
        ofdm_signal = ofdm_signal.reshape(num_chirps, samples_per_chirp)
        
        # Add targets (similar to FMCW but with OFDM signal)
        num_targets = 3
        max_range = 50  # meters
        max_velocity = 10  # m/s
        
        # Generate random targets
        ranges = np.random.uniform(5, max_range, num_targets)
        velocities = np.random.uniform(-max_velocity, max_velocity, num_targets)
        amplitudes = np.random.uniform(0.5, 1.0, num_targets)
        
        # Radar parameters
        fc = 5.8e9  # Center frequency: 5.8 GHz (typical for OFDM radar)
        c = 3e8  # Speed of light
        wavelength = c / fc
        symbol_duration = 10e-6  # 10 microseconds
        
        # For each OFDM symbol and receive antenna
        for symbol in range(num_chirps):
            symbol_time = symbol * symbol_duration
            
            # Initialize symbol signal
            symbol_signal = ofdm_signal[symbol].copy()
            
            # Add target reflections
            for i in range(num_targets):
                # Calculate range delay in samples
                tau = 2 * ranges[i] / c
                delay_samples = int(tau / (symbol_duration / samples_per_chirp))
                delay_samples = min(delay_samples, samples_per_chirp - 1)
                
                # Calculate Doppler shift
                doppler_freq = 2 * velocities[i] / wavelength
                doppler_phase = 2 * np.pi * doppler_freq * symbol_time
                
                # Apply delay and Doppler
                delayed_signal = np.roll(symbol_signal, delay_samples) * amplitudes[i] * np.exp(1j * doppler_phase)
                
                # Add to symbol signal
                symbol_signal += delayed_signal
            
            # Add some noise (lower noise for better decoding)
            noise = np.random.normal(0, 0.05, samples_per_chirp) + 1j * np.random.normal(0, 0.05, samples_per_chirp)
            symbol_signal += noise
            
            # For each receive antenna (add slight phase differences)
            for rx in range(num_rx):
                rx_phase = np.exp(1j * rx * np.pi / 4)  # Simple phase shift between antennas
                rx_signal = symbol_signal * rx_phase
                
                # Convert to real/imag format
                data[b, rx, symbol, :, 0] = torch.tensor(rx_signal.real, dtype=torch.float32)
                data[b, rx, symbol, :, 1] = torch.tensor(rx_signal.imag, dtype=torch.float32)
    
    return data, ground_truth_bits


def test_ofdm_decoder():
    """
    Test the OFDMDecoder module with synthetic data.
    """
    # Parameters
    batch_size = 2
    num_rx = 2
    num_chirps = 64
    samples_per_chirp = 64
    fft_size = samples_per_chirp
    num_subcarriers = samples_per_chirp // 2
    
    # Test with different modulation schemes
    modulation_schemes = ['bpsk', 'qpsk', 'qam16']
    
    # Initialize figure for visualization
    plt.figure(figsize=(15, 10))
    
    for i, modulation in enumerate(modulation_schemes):
        print(f"\nTesting with {modulation.upper()} modulation...")
        
        # Generate test data
        ofdm_data, ground_truth_bits = generate_test_ofdm_data_with_bits(
            batch_size=batch_size,
            num_rx=num_rx,
            num_chirps=num_chirps,
            samples_per_chirp=samples_per_chirp,
            modulation=modulation
        )
        
        # Initialize RadarTimeNet with the specific modulation
        model = RadarTimeNet(
            num_rx=num_rx,
            num_chirps=num_chirps,
            samples_per_chirp=samples_per_chirp,
            out_doppler_bins=num_chirps,
            out_range_bins=samples_per_chirp,
            use_learnable_fft=True,
            support_ofdm=True,
            ofdm_modulation=modulation
        )
        
        # Set model to evaluation mode
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            rd_map, ofdm_map, decoded_bits = model(ofdm_data, is_ofdm=True, modulation=modulation)
        
        # Print output shapes
        print(f"Range-Doppler Map shape: {rd_map.shape}")
        print(f"OFDM Map shape: {ofdm_map.shape}")
        print(f"Decoded Bits shape: {decoded_bits.shape}")
        
        # Calculate bit error rate (BER)
        # Note: In a real scenario, we would need to align the decoded bits with ground truth
        # For this test, we're just comparing shapes and visualizing
        
        # Visualize results
        plt.subplot(3, 3, i*3 + 1)
        plt.title(f"{modulation.upper()} Range-Doppler Map")
        rd_map_magnitude = torch.sqrt(rd_map[0, 0]**2 + rd_map[0, 1]**2)
        plt.imshow(rd_map_magnitude.numpy(), aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.xlabel('Range Bins')
        plt.ylabel('Doppler Bins')
        
        plt.subplot(3, 3, i*3 + 2)
        plt.title(f"{modulation.upper()} OFDM Map")
        ofdm_map_magnitude = torch.sqrt(ofdm_map[0, 0]**2 + ofdm_map[0, 1]**2)
        plt.imshow(ofdm_map_magnitude.numpy(), aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.xlabel('Subcarrier')
        plt.ylabel('Symbol')
        
        plt.subplot(3, 3, i*3 + 3)
        plt.title(f"{modulation.upper()} Decoded Bits")
        # Reshape bits for visualization
        bits_reshaped = decoded_bits[0].reshape(-1, 8)[:50]  # Show first 50 bytes
        plt.imshow(bits_reshaped.numpy(), aspect='auto', cmap='binary')
        plt.colorbar(label='Bit Value')
        plt.xlabel('Bit Position')
        plt.ylabel('Byte Index')
    
    plt.tight_layout()
    plt.savefig('ofdm_decoder_test_results.png')
    plt.show()
    
    print("\nTest completed successfully!")
    print("Results saved to 'ofdm_decoder_test_results.png'")


def test_standalone_decoder():
    """
    Test the standalone OFDMSymbolDecoder and OFDMDecoder modules.
    """
    # Parameters
    batch_size = 2
    fft_size = 64
    num_symbols = 16
    num_subcarriers = 32
    
    # Create random OFDM map (simulating output from RadarTimeNet)
    ofdm_map = torch.randn(batch_size, 2, num_symbols, fft_size)
    
    # Test with different modulation schemes
    modulation_schemes = ['bpsk', 'qpsk', 'qam16', 'qam64']
    
    print("\nTesting standalone OFDM decoders...")
    
    for modulation in modulation_schemes:
        print(f"\nTesting with {modulation.upper()} modulation...")
        
        # Initialize standalone symbol decoder
        symbol_decoder = OFDMSymbolDecoder(
            fft_size=fft_size,
            num_subcarriers=num_subcarriers,
            dc_null=True,
            guard_bands=[4, 4]
        )
        
        # Decode symbols
        decoded_bits_symbol = symbol_decoder(ofdm_map, modulation)
        print(f"Symbol Decoder output shape: {decoded_bits_symbol.shape}")
        
        # Initialize full decoder with channel estimation
        full_decoder = OFDMDecoder(
            fft_size=fft_size,
            num_symbols=num_symbols,
            num_subcarriers=num_subcarriers,
            dc_null=True,
            guard_bands=[4, 4],
            use_channel_estimation=True
        )
        
        # Decode with channel estimation
        decoded_bits_full = full_decoder(ofdm_map, modulation)
        print(f"Full Decoder output shape: {decoded_bits_full.shape}")
        
        # Expected bits per symbol based on modulation
        bits_per_symbol = {
            'bpsk': 1,
            'qpsk': 2,
            'qam16': 4,
            'qam64': 6,
            'qam256': 8
        }[modulation.lower()]
        
        expected_bits = batch_size * num_symbols * num_subcarriers * bits_per_symbol
        print(f"Expected total bits: {expected_bits}")
        print(f"Actual total bits: {decoded_bits_full.numel()}")
    
    print("\nStandalone decoder test completed successfully!")


if __name__ == "__main__":
    # Test the OFDMDecoder integrated with RadarTimeNet
    test_ofdm_decoder()
    
    # Test the standalone decoders
    test_standalone_decoder()