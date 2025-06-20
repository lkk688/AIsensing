import torch
import numpy as np
import matplotlib.pyplot as plt
from AIRadarLib.modeling_TimeNet import RadarTimeNet
from AIRadarLib.waveform_utils import generate_linear_chirp, generate_ofdm_signal


def generate_test_fmcw_data(batch_size=2, num_rx=2, num_chirps=64, samples_per_chirp=64):
    """
    Generate synthetic FMCW radar data for testing.
    
    Args:
        batch_size: Number of samples in the batch
        num_rx: Number of receive antennas
        num_chirps: Number of chirps
        samples_per_chirp: Number of samples per chirp
        
    Returns:
        Tensor with shape [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
    """
    # Generate random targets
    num_targets = 3
    max_range = 50  # meters
    max_velocity = 10  # m/s
    
    # Initialize empty tensor
    data = torch.zeros(batch_size, num_rx, num_chirps, samples_per_chirp, 2)
    
    for b in range(batch_size):
        # Generate random targets
        ranges = np.random.uniform(5, max_range, num_targets)
        velocities = np.random.uniform(-max_velocity, max_velocity, num_targets)
        amplitudes = np.random.uniform(0.5, 1.0, num_targets)
        
        # Radar parameters
        fc = 77e9  # Center frequency: 77 GHz
        bw = 1e9  # Bandwidth: 1 GHz
        chirp_duration = 50e-6  # 50 microseconds
        prf = 1 / (chirp_duration * 1.1)  # Pulse repetition frequency
        c = 3e8  # Speed of light
        
        # Calculate parameters
        slope = bw / chirp_duration
        wavelength = c / fc
        
        # Generate time samples
        t = np.linspace(0, chirp_duration, samples_per_chirp)
        
        # For each chirp and receive antenna
        for chirp in range(num_chirps):
            chirp_time = chirp / prf
            
            # Initialize chirp signal
            chirp_signal = np.zeros(samples_per_chirp, dtype=np.complex128)
            
            # Add target reflections
            for i in range(num_targets):
                # Calculate range delay
                tau = 2 * ranges[i] / c
                
                # Calculate Doppler shift
                doppler_freq = 2 * velocities[i] / wavelength
                
                # Phase due to Doppler
                doppler_phase = 2 * np.pi * doppler_freq * chirp_time
                
                # Delayed signal with Doppler shift
                delayed_t = t - tau
                valid_indices = delayed_t >= 0
                
                # Beat signal (difference between transmitted and received)
                beat_phase = 2 * np.pi * (slope * tau * delayed_t[valid_indices] - 0.5 * slope * tau**2)
                beat_signal = amplitudes[i] * np.exp(1j * (beat_phase + doppler_phase))
                
                # Add to chirp signal
                chirp_signal[valid_indices] += beat_signal
            
            # Add some noise
            noise = np.random.normal(0, 0.1, samples_per_chirp) + 1j * np.random.normal(0, 0.1, samples_per_chirp)
            chirp_signal += noise
            
            # For each receive antenna (add slight phase differences)
            for rx in range(num_rx):
                rx_phase = np.exp(1j * rx * np.pi / 4)  # Simple phase shift between antennas
                rx_signal = chirp_signal * rx_phase
                
                # Convert to real/imag format
                data[b, rx, chirp, :, 0] = torch.tensor(rx_signal.real, dtype=torch.float32)
                data[b, rx, chirp, :, 1] = torch.tensor(rx_signal.imag, dtype=torch.float32)
    
    return data


def generate_test_ofdm_data(batch_size=2, num_rx=2, num_chirps=64, samples_per_chirp=64):
    """
    Generate synthetic OFDM radar data for testing.
    
    Args:
        batch_size: Number of samples in the batch
        num_rx: Number of receive antennas
        num_chirps: Number of chirps (OFDM symbols)
        samples_per_chirp: Number of samples per chirp (FFT size)
        
    Returns:
        Tensor with shape [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
    """
    # Initialize empty tensor
    data = torch.zeros(batch_size, num_rx, num_chirps, samples_per_chirp, 2)
    
    for b in range(batch_size):
        # Generate OFDM signal
        ofdm_signal = generate_ofdm_signal(
            num_subcarriers=samples_per_chirp//2,  # Use half of the subcarriers
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
            
            # Add some noise
            noise = np.random.normal(0, 0.1, samples_per_chirp) + 1j * np.random.normal(0, 0.1, samples_per_chirp)
            symbol_signal += noise
            
            # For each receive antenna (add slight phase differences)
            for rx in range(num_rx):
                rx_phase = np.exp(1j * rx * np.pi / 4)  # Simple phase shift between antennas
                rx_signal = symbol_signal * rx_phase
                
                # Convert to real/imag format
                data[b, rx, symbol, :, 0] = torch.tensor(rx_signal.real, dtype=torch.float32)
                data[b, rx, symbol, :, 1] = torch.tensor(rx_signal.imag, dtype=torch.float32)
    
    return data


def test_radartime_net():
    """
    Test the RadarTimeNet model with synthetic data.
    """
    # Model parameters
    num_rx = 2
    num_chirps = 64
    samples_per_chirp = 64
    out_doppler_bins = 64
    out_range_bins = 64
    batch_size = 2
    
    # Initialize model
    model = RadarTimeNet(
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp,
        out_doppler_bins=out_doppler_bins,
        out_range_bins=out_range_bins,
        use_learnable_fft=True,
        support_ofdm=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    print("Testing RadarTimeNet with FMCW data...")
    # Generate test data for FMCW
    fmcw_data = generate_test_fmcw_data(
        batch_size=batch_size,
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp
    )
    
    # Forward pass
    with torch.no_grad():
        fmcw_output = model(fmcw_data, is_ofdm=False)
    
    # Print output shape
    print(f"FMCW Output shape: {fmcw_output.shape}")
    
    # Visualize the first sample's range-Doppler map
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("FMCW Range-Doppler Map (Magnitude)")
    rd_map_magnitude = torch.sqrt(fmcw_output[0, 0]**2 + fmcw_output[0, 1]**2)
    plt.imshow(rd_map_magnitude.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Range Bins')
    plt.ylabel('Doppler Bins')
    
    plt.subplot(2, 2, 2)
    plt.title("FMCW Range-Doppler Map (Phase)")
    rd_map_phase = torch.atan2(fmcw_output[0, 1], fmcw_output[0, 0])
    plt.imshow(rd_map_phase.numpy(), aspect='auto', cmap='hsv')
    plt.colorbar(label='Phase (rad)')
    plt.xlabel('Range Bins')
    plt.ylabel('Doppler Bins')
    
    print("\nTesting RadarTimeNet with OFDM data...")
    # Generate test data for OFDM
    ofdm_data = generate_test_ofdm_data(
        batch_size=batch_size,
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp
    )
    
    # Forward pass
    with torch.no_grad():
        ofdm_output, ofdm_demod, decoded_bits = model(ofdm_data, is_ofdm=True)
    
    # Print output shapes
    print(f"OFDM Range-Doppler Output shape: {ofdm_output.shape}")
    print(f"OFDM Demodulation Output shape: {ofdm_demod.shape}")
    print(f"Decoded Bits Output shape: {decoded_bits.shape}")
    
    # Visualize the first sample's OFDM range-Doppler map
    plt.subplot(2, 2, 3)
    plt.title("OFDM Range-Doppler Map (Magnitude)")
    rd_map_magnitude = torch.sqrt(ofdm_output[0, 0]**2 + ofdm_output[0, 1]**2)
    plt.imshow(rd_map_magnitude.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Range Bins')
    plt.ylabel('Doppler Bins')
    
    plt.subplot(2, 2, 4)
    plt.title("OFDM Demodulation Output (Magnitude)")
    ofdm_demod_magnitude = torch.sqrt(ofdm_demod[0, 0]**2 + ofdm_demod[0, 1]**2)
    plt.imshow(ofdm_demod_magnitude.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Range Bins')
    plt.ylabel('Doppler Bins')
    
    # Add a new figure for the decoded bits
    plt.figure(figsize=(10, 5))
    plt.title("OFDM Decoded Bits")
    # Reshape bits for better visualization (show as bytes)
    bits_reshaped = decoded_bits[0].reshape(-1, 8)[:50]  # Show first 50 bytes
    plt.imshow(bits_reshaped.numpy(), aspect='auto', cmap='binary')
    plt.colorbar(label='Bit Value')
    plt.xlabel('Bit Position')
    plt.ylabel('Byte Index')
    plt.tight_layout()
    plt.savefig('ofdm_decoded_bits.png')
    
    plt.tight_layout()
    plt.savefig('radartime_net_test_results.png')
    plt.show()
    
    print("\nTest completed successfully!")
    print("Results saved to 'radartime_net_test_results.png'")
    print("Decoded bits visualization saved to 'ofdm_decoded_bits.png'")


if __name__ == "__main__":
    test_radartime_net()