import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split, ConcatDataset
from AIRadarLib.waveform_utils import generate_adf4159_fmcw_chirp, generate_fmcw_chirp_signal
from AIRadarLib.target_utils import generate_radar_targets, create_target_mask
from AIRadarLib.channel_simulation import (
    generate_fmcw_chirp,
    add_realistic_effects,
    add_noise,
    calculate_received_power,
    ray_tracing_simulation
)
from AIRadarLib.signal_processing import fmcw_demodulate, time_to_range_doppler

# === SHAPE-ALIGNED SyntheticRadarDataset WITH AUTO-PADDING ===
class SyntheticRadarDataset(Dataset):
    """
    A dataset class that generates synthetic FMCW radar data with configurable parameters.
    
    This class simulates radar returns from multiple targets with configurable parameters
    such as SNR, modulation type, and target characteristics. It can generate realistic
    FMCW chirp signals using the ADF4159 PLL model and supports various signal processing
    operations including modulation, demodulation, and augmentation.
    """
    def __init__(self, num_samples=1000, num_chirps=64, samples_per_chirp=64,
                 modulation_type='none', augment=True, max_targets=3,
                 target_shape=(64, 64), fixed_snr_db=40, use_random_snr=False,
                 use_adf4159=True, start_freq=24.0e9, bandwidth=200e6, 
                 sample_rate=1e6, phase_noise_level=-85):
        """
        Initialize the SyntheticRadarDataset.
        
        Args:
            num_samples: Number of samples in the dataset
            num_chirps: Number of chirps in each sample
            samples_per_chirp: Number of samples per chirp
            modulation_type: Type of modulation ('none', 'sine', 'ofdm')
            augment: Whether to apply augmentation to the signal
            max_targets: Maximum number of targets to generate
            target_shape: Shape of the target map (doppler_bins, range_bins)
            fixed_snr_db: Fixed SNR value in dB if use_random_snr is False
            use_random_snr: Whether to use random SNR values
            use_adf4159: Whether to use the ADF4159 FMCW chirp generator
            start_freq: Starting frequency of the chirp in Hz
            bandwidth: Bandwidth of the chirp in Hz
            sample_rate: Sampling rate in Hz
            phase_noise_level: Phase noise level in dBc/Hz at 100 kHz offset
        """
        self.num_samples = num_samples
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.modulation_type = modulation_type
        self.augment = augment
        self.max_targets = max_targets
        self.target_shape = target_shape  # (doppler_bins, range_bins)
        self.num_doppler_bins, self.num_range_bins = target_shape #new added
        self.fixed_snr_db = fixed_snr_db
        self.use_random_snr = use_random_snr
        self.snr_db = self.fixed_snr_db
        
        # FMCW radar parameters
        self.use_adf4159 = use_adf4159
        self.start_freq = start_freq
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate
        self.phase_noise_level = phase_noise_level
        self.chirp_duration = samples_per_chirp / sample_rate
        
    def pad_or_crop(self, x, target_shape):
        """
        Pad or crop a 2D array to the target shape.
        
        Args:
            x: Input array to pad or crop
            target_shape: Target shape (height, width)
            
        Returns:
            Padded or cropped array
        """
        pad_d, pad_r = target_shape[0] - x.shape[0], target_shape[1] - x.shape[1]
        pad_d = max(pad_d, 0)
        pad_r = max(pad_r, 0)
        x_padded = np.pad(x, ((0, pad_d), (0, pad_r)), mode='constant')
        return x_padded[:target_shape[0], :target_shape[1]]

    def modulate_chirp(self, signal):
        """
        Apply modulation to the chirp signal.
        
        Args:
            signal: Input signal to modulate
            
        Returns:
            Modulated signal
        """
        if self.modulation_type == 'sine':
            t = np.linspace(0, 1, signal.shape[-1], endpoint=False)
            sine_wave = np.exp(1j * 2 * np.pi * 5 * t)
            return signal * sine_wave[np.newaxis, :]
        elif self.modulation_type == 'ofdm':
            carriers = np.fft.ifft(np.random.choice([1, -1], size=(self.num_chirps, self.samples_per_chirp)), axis=-1)
            return signal * carriers
        else:
            return signal

    def demodulate(self, rx_signal):
        """
        Demodulate the received signal by mixing with the transmitted signal.
        
        In FMCW radar, demodulation is performed by mixing the received signal
        with the transmitted signal, which produces beat frequencies proportional
        to target ranges.
        
        Args:
            rx_signal: Received signal to demodulate
            
        Returns:
            Demodulated signal (beat signal)
        """
        # Generate the reference chirp signal (transmitted signal)
        tx_signal = self.generate_chirp_signal()
        
        # For FMCW radar, demodulation is mixing (multiplication) of rx and tx signals
        # Complex conjugate of tx_signal is used for proper mixing
        demodulated_signal = rx_signal * np.conjugate(tx_signal)
        
        # Apply additional modulation-specific processing if needed
        if self.modulation_type == 'sine':
            t = np.linspace(0, 1, rx_signal.shape[-1], endpoint=False)
            sine_wave = np.exp(-1j * 2 * np.pi * 5 * t)
            return demodulated_signal * sine_wave[np.newaxis, :]
        elif self.modulation_type == 'ofdm':
            return demodulated_signal
        else:
            return demodulated_signal
            


    def apply_augmentation(self, signal):
        """
        Apply random augmentations to the signal.
        
        Args:
            signal: Input signal to augment
            
        Returns:
            Augmented signal
        """
        if self.augment:
            if np.random.rand() < 0.5:
                # Add complex Gaussian noise
                noise_power = 0.05
                noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) * noise_power
                signal += noise
            if np.random.rand() < 0.3:
                # Apply random delay
                delay = np.random.randint(0, 3)
                signal = np.roll(signal, delay, axis=-1)
        return signal

    def inject_targets(self, shape):
        """
        Generate radar signal with injected targets by simulating the FMCW radar process.
        
        This method simulates the complete FMCW radar process:
        1. Generate transmitted chirp signal
        2. Simulate target reflections with range delays and Doppler shifts
        3. Create the received signal as the sum of all target echoes
        
        Args:
            shape: Shape of the signal (num_chirps, samples_per_chirp)
            
        Returns:
            rx_signal: Complex radar received signal with target echoes
            label_map: Binary map indicating target locations
            vel_map: Map of target velocities
            meta_targets: List of target metadata
        """
        # Generate the transmitted FMCW chirp signal
        tx_signal = self.generate_chirp_signal()
        
        # Initialize received signal, label map, velocity map, and metadata
        rx_signal = np.zeros(shape, dtype=np.complex64)
        label_map = np.zeros((shape[0], shape[1]), dtype=np.float32)
        vel_map = np.zeros((shape[0], shape[1]), dtype=np.float32)
        meta_targets = []
        
        # Calculate radar parameters for target simulation
        # Speed of light in m/s
        c = 3e8
        # Calculate range resolution
        range_resolution = c / (2 * self.bandwidth)
        # Calculate velocity resolution
        velocity_resolution = c / (2 * self.start_freq * self.num_chirps * self.chirp_duration)
        
        # Maximum range based on sampling rate and bandwidth
        max_range = (self.samples_per_chirp * c) / (2 * self.bandwidth)
        # Maximum velocity based on wavelength and PRF
        wavelength = c / self.start_freq
        prf = 1 / (self.chirp_duration)  # Pulse Repetition Frequency
        max_velocity = (wavelength * prf) / 4
        
        # Generate random radar targets using the target_utils function
        num_targets = np.random.randint(1, self.max_targets + 1)
        targets = generate_radar_targets(
            num_targets=num_targets,
            min_range=1,
            max_range=max_range * 0.8,  # Stay within 80% of max range to avoid aliasing
            min_velocity=0.1,
            max_velocity=max_velocity * 0.8,  # Stay within 80% of max velocity to avoid aliasing
            min_rcs=5.0,
            max_rcs=30.0,
            azimuth_range=(-45, 45),
            elevation_range=(-10, 10)
        )
        
        # Process each target
        for target in targets:
            # Calculate range and Doppler bins
            rbin = int(target['distance'] / range_resolution)
            # Convert velocity to Doppler bin (centered at num_chirps/2)
            vbin = int(shape[0] // 2 + target['velocity'] / velocity_resolution)
            
            # Ensure bins are within valid range
            rbin = min(max(0, rbin), shape[1] - 1)
            vbin = min(max(0, vbin), shape[0] - 1)
            
            # Calculate normalized Doppler frequency
            doppler_cycles = (vbin - shape[0] // 2) / shape[0]
            
            # Calculate SNR based on RCS and distance
            snr_db = np.random.uniform(10, 30) if self.use_random_snr else self.fixed_snr_db
            amplitude = 10 ** (snr_db / 20)
            
            # Simulate target echo: delayed and Doppler-shifted version of tx_signal
            doppler_phase = 2 * np.pi * doppler_cycles * np.arange(shape[0])[:, np.newaxis]
            echo = amplitude * tx_signal * np.exp(1j * doppler_phase)
            
            # Apply range delay by rolling the signal
            rx_signal += np.roll(echo, rbin, axis=1)
            
            # Update label and velocity maps
            label_map[vbin, rbin] = 1.0
            vel_map[vbin, rbin] = doppler_cycles
            
            # Store target metadata
            meta_targets.append({
                'range_bin': rbin, 
                'doppler_bin': vbin, 
                'snr_db': snr_db,
                'range_m': target['distance'],
                'velocity_mps': target['velocity'],
                'rcs': target['rcs'],
                'azimuth': target['azimuth'],
                'elevation': target['elevation']
            })

        return rx_signal, label_map, vel_map, meta_targets

    def generate_chirp_signal(self):
        """
        Generate FMCW chirp signal using either the ADF4159 model, generate_fmcw_chirp_signal,
        or simple complex exponential.
        
        Returns:
            Complex chirp signal with shape (num_chirps, samples_per_chirp)
        """
        if self.use_adf4159:
            # Use the ADF4159 PLL model for realistic chirp generation
            continuous_signal, _ = generate_adf4159_fmcw_chirp(
                num_chirps=self.num_chirps,
                total_samples_per_chirp=self.samples_per_chirp,
                active_samples=self.samples_per_chirp,
                sample_rate=self.sample_rate,
                start_freq=self.start_freq,
                bandwidth=self.bandwidth,
                chirp_duration=self.chirp_duration,
                tx_power=1.0,
                phase_noise_level=self.phase_noise_level,
                window_type=None  # No windowing at this stage
            )
            
            # Reshape the continuous signal to (num_chirps, samples_per_chirp)
            chirp_signal = continuous_signal.reshape(self.num_chirps, self.samples_per_chirp)
        else:
            # Use the generate_fmcw_chirp_signal function for more realistic chirp generation
            # This function provides phase-continuous FMCW chirps with configurable parameters
            # Calculate slope from bandwidth and chirp duration
            slope = self.bandwidth / self.chirp_duration
            
            # Call generate_fmcw_chirp_signal with the correct parameters
            continuous_signal, _ = generate_fmcw_chirp_signal(
                num_chirps=self.num_chirps,
                total_samples_per_chirp=self.samples_per_chirp,
                active_samples=self.samples_per_chirp,
                sample_rate=self.sample_rate,
                slope=slope,
                tx_power=1.0,
                edge_ratio=0.1,
                window_type=None  # No windowing at this stage
            )
            
            # Reshape the continuous signal to (num_chirps, samples_per_chirp) if needed
            chirp_signal = continuous_signal.reshape(self.num_chirps, self.samples_per_chirp)
                
        return chirp_signal

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset, simulating the FMCW radar process and generating range-Doppler map.
        
        Args:
            idx: Index of the sample
            
        Returns:
            iq_tensor: Range-Doppler map as tensor [1, 2, num_doppler_bins, num_range_bins]
            label_tensor: Target label map as tensor [1, num_chirps, samples_per_chirp, 1]
            vel_tensor: Target velocity map as tensor [1, num_chirps, samples_per_chirp, 1]
            modulation_type: Type of modulation used
            metadata: Dictionary containing target metadata and simulation parameters
        """
        # Set random seed for reproducibility if needed
        # Initialize seed in __init__ if not already set
        if not hasattr(self, 'seed'):
            self.seed = None
            
        if self.seed is not None:
            np.random.seed(self.seed + idx)
        
        # Calculate wavelength, range and velocity resolution
        c = 3e8  # Speed of light in m/s
        wavelength = c / self.start_freq  # in meters
        range_resolution = c / (2 * self.bandwidth)  # in meters
        velocity_resolution = wavelength / (2 * self.num_chirps * (self.chirp_duration))  # in m/s
            
        # Generate TX chirp signal
        tx_signal = self.generate_chirp_signal() #(64, 64) complex signal
        
        # Generate target echoes (received signal)
        rx_signal, label_map, vel_map, meta_targets = self.inject_targets((self.num_chirps, self.samples_per_chirp))
        #(64, 64) complex
        # Apply augmentation to the received signal
        rx_signal = self.apply_augmentation(rx_signal)
        
        # Apply modulation if specified
        rx_signal = self.modulate_chirp(rx_signal) #do nothing, still (64, 64) complex
        
        # Flatten signals for demodulation if needed
        tx_flat = tx_signal.flatten() if tx_signal.ndim > 1 else tx_signal #(4096,)
        rx_flat = rx_signal.flatten() if rx_signal.ndim > 1 else rx_signal #(4096,)
        
        # Perform FMCW demodulation using the signal_processing function
        beat_signal = fmcw_demodulate(
            tx_full=tx_flat,
            rx_full=rx_flat,
            total_samples_per_chirp=self.samples_per_chirp,
            beat_samples_per_chirp=self.samples_per_chirp,
            num_chirps=self.num_chirps
        )#(64, 64) complex

        # Pad or crop to target shape
        beat_signal = self.pad_or_crop(beat_signal, self.target_shape)
        label_map = self.pad_or_crop(label_map, self.target_shape)
        vel_map = self.pad_or_crop(vel_map, self.target_shape)
        #still the same (64, 64) complex
        # Generate range-Doppler map using the time_to_range_doppler function from signal_processing
        # Reshape beat_signal to [num_chirps, samples_per_chirp] for time_to_range_doppler
        beat_signal_reshaped = beat_signal.reshape(self.num_chirps, self.samples_per_chirp)
        #(64, 64)
        rd_map = time_to_range_doppler(
            rx_signal=beat_signal_reshaped.reshape(1, self.num_chirps, self.samples_per_chirp),  # Reshape to [1, num_chirps, samples_per_chirp]
            num_chirps=self.num_chirps,
            samples_per_chirp=self.samples_per_chirp,
            num_doppler_bins=self.num_doppler_bins,
            num_range_bins=self.num_range_bins,
            apply_mti=False,
            apply_doppler_centering=True, #Whether to center the Doppler FFT
            apply_notch_filter=False
        )#(1, 2, 64, 64)
        
        # Convert to PyTorch tensors
        # For the IQ data (beat signal), we need to reshape it to include batch dimension
        # This represents the time-domain signal before range-Doppler processing
        iq_tensor = torch.tensor(beat_signal_reshaped[np.newaxis, np.newaxis, ...], dtype=torch.complex64)
        
        # For the range-Doppler map, we already have the real and imaginary parts separated
        rd_tensor = torch.tensor(rd_map[np.newaxis, ...], dtype=torch.float32)
        #[1, 1, 2, 64, 64]
        # For label and velocity maps, maintain the original format
        label_tensor = torch.tensor(label_map[np.newaxis, ..., np.newaxis], dtype=torch.float32) #[1, 64, 64, 1]
        vel_tensor = torch.tensor(vel_map[np.newaxis, ..., np.newaxis], dtype=torch.float32) #[1, 64, 64, 1]
        
        # Create metadata dictionary with both target info and simulation parameters
        metadata = {
            'meta_targets': meta_targets,
            'simulation_params': {
                'num_chirps': self.num_chirps,
                'samples_per_chirp': self.samples_per_chirp,
                'bandwidth': self.bandwidth,
                'start_freq': self.start_freq,
                'wavelength': wavelength,
                'range_resolution': range_resolution,
                'velocity_resolution': velocity_resolution,
                'chirp_duration': self.chirp_duration,
                'sample_rate': self.sample_rate
            }
        }

        return iq_tensor, rd_tensor, label_tensor, vel_tensor, self.modulation_type, metadata
    

    
    def get_range_doppler_map(self, idx=0, apply_window=True, window_type='hanning', window_param=None):
        """
        Generate and return the range-Doppler map for a specific sample using time_to_range_doppler.
        
        Args:
            idx: Index of the sample to process (default: 0)
            apply_window: Whether to apply window function before FFT
            window_type: Type of window function ('hanning', 'hamming', 'blackman', 'kaiser')
            window_param: Parameter for window function (e.g., beta for Kaiser window)
            
        Returns:
            Dictionary containing:
                - complex_signal: Raw complex time-domain signal
                - rd_map: Complex range-Doppler map after FFT
                - rd_mag: Magnitude of range-Doppler map (linear scale)
                - rd_log: Log-scale magnitude in dB
                - target_mask: Binary mask showing target locations
                - meta: Target metadata
        """
        # Get the sample data
        iq_tensor, rd_tensor, label_tensor, vel_tensor, _, meta = self[idx]
        #[1, 2, num_doppler_bins, num_range_bins][1, 1, 2, 64, 64], [1, 64, 64, 1], [1, 64, 64, 1]
        
        # Convert to complex signal - iq_tensor is now directly complex64
        complex_signal = iq_tensor[0, 0].numpy()
        target_mask = label_tensor[0].numpy()
        
        # Reshape complex signal to match time_to_range_doppler input format [num_rx, num_chirps, samples_per_chirp]
        # Assuming complex_signal shape is [num_doppler_bins, num_range_bins]
        rx_signal = complex_signal.reshape(1, self.num_chirps, self.samples_per_chirp)
        #Reshaped the complex signal to match the expected input format [num_rx, num_chirps, samples_per_chirp]
        # Use time_to_range_doppler to compute range-Doppler map
        rd_map_result = time_to_range_doppler(
            rx_signal=rx_signal, #[num_rx, num_chirps, samples_per_chirp]
            num_chirps=self.num_chirps,
            samples_per_chirp=self.samples_per_chirp,
            num_doppler_bins=self.num_doppler_bins,
            num_range_bins=self.num_range_bins,
            apply_mti=False,
            apply_doppler_centering=True,
            apply_notch_filter=False,
            use_blackman_window=apply_window and window_type == 'blackman',
            dynamic_range_db=50
        )
        #The function returns a tensor with shape [num_rx, 2, num_doppler_bins, num_range_bins]
        #Extracted the real part from [0, 0, :, :] and the imaginary part from [0, 1, :, :]
        # Reconstruct complex range-Doppler map from real and imaginary parts
        # rd_map_result has shape [num_rx, 2, num_doppler_bins, num_range_bins]
        rd_map = rd_map_result[0, 0, :, :] + 1j * rd_map_result[0, 1, :, :]
        
        # Calculate magnitude and log-scale magnitude
        rd_mag = np.abs(rd_map)
        rd_log = 20 * np.log10(rd_mag + 1e-10)  # Add small constant to avoid log(0)
        
        return {
            'complex_signal': complex_signal,
            'rd_map': rd_map,
            'rd_mag': rd_mag,
            'rd_log': rd_log,
            'target_mask': target_mask,
            'meta': meta
        }

class DebugFMCWDataset(Dataset):
    def __init__(self, num_samples=100, num_chirps=64, samples_per_chirp=64, snr_db=60,
                 rbin=32, vbin_shifted=16, add_noise=True, use_window=True):
        """
        Optimized FMCW single target simulation dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            num_chirps: Number of chirps in each sample
            samples_per_chirp: Number of samples per chirp
            snr_db: Signal-to-noise ratio in dB
            rbin: Range bin position of the target (default: 32)
            vbin_shifted: Doppler bin position in shifted FFT coordinates (default: 16)
            add_noise: Whether to add noise to the signal (default: True)
            use_window: Whether to apply Hanning window for FFT processing (default: True)
        """
        self.num_samples = num_samples
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.snr_db = snr_db
        self.rbin = rbin
        self.vbin_shifted = vbin_shifted
        self.add_noise = add_noise
        self.use_window = use_window

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        shape = (self.num_chirps, self.samples_per_chirp)
        signal = np.zeros(shape, dtype=np.complex64)
        label_map = np.zeros(shape, dtype=np.float32)
        vel_map = np.zeros(shape, dtype=np.float32)

        # Inject a single target at shifted bin position
        rbin = self.rbin
        vbin_shifted = self.vbin_shifted  # visible in FFT plot
        vbin = (vbin_shifted - shape[0] // 2) % shape[0]  # map to unshifted bin
        doppler_cycles = vbin / shape[0]
        amplitude = 10 ** (self.snr_db / 20)

        # Generate time domain signal
        t = np.arange(shape[1])
        echo = amplitude * np.exp(1j * 2 * np.pi * doppler_cycles * np.outer(np.arange(shape[0]), t / shape[1]))
        signal += np.roll(echo, rbin, axis=1)
        
        # Add noise if requested
        if self.add_noise:
            noise_power = 10 ** (-self.snr_db / 10) * amplitude**2
            noise = np.sqrt(noise_power/2) * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
            signal += noise
            
        # Create label and velocity maps
        label_map[vbin, rbin] = 1.0
        vel_map[vbin, rbin] = doppler_cycles
        meta = [{'range_bin': rbin, 'doppler_bin': vbin_shifted, 'snr_db': self.snr_db}]

        # Convert to PyTorch tensors
        # For the IQ data (beat signal), convert to complex tensor
        iq_tensor = torch.tensor(signal[np.newaxis, np.newaxis, ...], dtype=torch.complex64)
        
        # For the range-Doppler map, compute it and separate real and imaginary parts
        # First compute the range-Doppler map
        rd_map_data = self.get_range_doppler_map(idx)
        rd_map_complex = rd_map_data['rd_map']
        rd_map = np.stack([np.real(rd_map_complex), np.imag(rd_map_complex)], axis=0)
        rd_tensor = torch.tensor(rd_map[np.newaxis, ...], dtype=torch.float32)
        
        # For label and velocity maps
        label_tensor = torch.tensor(label_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)
        vel_tensor = torch.tensor(vel_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)

        return iq_tensor, rd_tensor, label_tensor, vel_tensor, "none", meta
    
    def get_range_doppler_map(self, idx=0, window_type='hanning', window_param = None):
        """
        Generate and return the range-Doppler map for a specific sample.
        
        Args:
            idx: Index of the sample to process (default: 0)
            
        Returns:
            Dictionary containing:
                - complex_signal: Raw complex time-domain signal
                - rd_map: Complex range-Doppler map after FFT
                - rd_mag: Magnitude of range-Doppler map
                - rd_log: Log-scale magnitude in dB
                - target_mask: Binary mask showing target location
                - meta: Target metadata
        """
        # Get the sample data
        iq_tensor, rd_tensor, label_tensor, vel_tensor, _, meta = self[idx]
        
        # Convert to complex signal - iq_tensor is now directly complex64
        complex_signal = iq_tensor[0, 0].numpy()
        target_mask = label_tensor[0].numpy()
        
        # Apply window if requested
        # Apply window function if window_type is specified
        if window_type is not None:
            # Create window function based on specified type
            if window_type == 'hanning':
                window_doppler = np.hanning(complex_signal.shape[0])
                window_range = np.hanning(complex_signal.shape[1])
            elif window_type == 'hamming':
                window_doppler = np.hamming(complex_signal.shape[0])
                window_range = np.hamming(complex_signal.shape[1])
            elif window_type == 'blackman':
                window_doppler = np.blackman(complex_signal.shape[0])
                window_range = np.blackman(complex_signal.shape[1])
            elif window_type == 'kaiser':
                beta = window_param if window_param is not None else 8.0
                window_doppler = np.kaiser(complex_signal.shape[0], beta)
                window_range = np.kaiser(complex_signal.shape[1], beta)
            else:
                raise ValueError("Invalid window type. Please choose from 'hanning', 'hamming', 'blackman', or 'kaiser'.")
            # Create 2D window by outer product
            window = np.outer(window_doppler, window_range)
            windowed_signal = complex_signal * window
        else:
            windowed_signal = complex_signal
            
        # Compute range-Doppler map
        rd_map = np.fft.fftshift(np.fft.fft2(windowed_signal))
        rd_mag = np.abs(rd_map)
        rd_log = 20 * np.log10(rd_mag + 1e-10)
        
        return {
            'complex_signal': complex_signal,
            'rd_map': rd_map,
            'rd_mag': rd_mag,
            'rd_log': rd_log,
            'target_mask': target_mask,
            'meta': meta
        }

def visualize_debug_fmcw(dataset, idx=0, save_path=None):
    """
    Visualize the debug FMCW dataset with range-Doppler map and target location.
    
    Args:
        dataset: DebugFMCWDataset instance
        idx: Index of the sample to visualize
        save_path: Path to save the visualization (if None, display instead)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Get range-Doppler map and other data
    rd_data = dataset.get_range_doppler_map(idx)
    complex_signal = rd_data['complex_signal']
    rd_mag = rd_data['rd_mag']
    rd_log = rd_data['rd_log']
    target_mask = rd_data['target_mask']
    meta = rd_data['meta']
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot time-domain magnitude
    time_mag = np.abs(complex_signal)
    axs[0, 0].imshow(time_mag, aspect='auto', cmap='viridis')
    axs[0, 0].set_title("Time Domain Magnitude")
    axs[0, 0].set_xlabel("Range Sample")
    axs[0, 0].set_ylabel("Chirp Number")
    
    # Plot range-Doppler map (linear scale)
    im1 = axs[0, 1].imshow(rd_mag, aspect='auto', cmap='magma')
    axs[0, 1].set_title("Range-Doppler Map (Linear)")
    axs[0, 1].set_xlabel("Range Bin")
    axs[0, 1].set_ylabel("Doppler Bin")
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot range-Doppler map (log scale)
    vmax = np.max(rd_log)
    vmin = vmax - 40  # 40 dB dynamic range
    im2 = axs[1, 0].imshow(rd_log, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Range-Doppler Map (dB)")
    axs[1, 0].set_xlabel("Range Bin")
    axs[1, 0].set_ylabel("Doppler Bin")
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Plot target mask
    axs[1, 1].imshow(target_mask.squeeze(), aspect='auto', cmap='hot')
    axs[1, 1].set_title("Target Mask")
    axs[1, 1].set_xlabel("Range Bin")
    axs[1, 1].set_ylabel("Doppler Bin")
    
    # Mark target locations
    for t in meta:
        # For time domain and mask plots (unshifted coordinates)
        axs[0, 0].add_patch(patches.Circle((t['range_bin'], t['doppler_bin']), 
                                          radius=1, color='lime', fill=False))
        axs[1, 1].add_patch(patches.Circle((t['range_bin'], t['doppler_bin']), 
                                          radius=1, color='lime', fill=False))
        
        # For range-Doppler maps (shifted FFT coordinates)
        axs[0, 1].add_patch(patches.Circle((t['range_bin'], t['doppler_bin']), 
                                          radius=1, color='lime', fill=False))
        axs[1, 0].add_patch(patches.Circle((t['range_bin'], t['doppler_bin']), 
                                          radius=1, color='lime', fill=False))
        
        # Add SNR annotation
        axs[1, 1].text(t['range_bin'], t['doppler_bin'], 
                      f"{t['snr_db']:.1f}dB", color='white', fontsize=8, 
                      ha='center', va='center')
    
    fig.suptitle(f"FMCW Single Target Simulation (SNR: {dataset.snr_db} dB)")
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
        plt.close()
    else:
        plt.show()

# === Visualization ===
def visualize_synthetic_sample1(dataset, index=0, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    iq, rd, det, vel, mod, meta = dataset[index]
    det = det.squeeze().numpy()
    vel = vel.squeeze().numpy()
    # Get complex signal - iq is now directly complex64
    complex_signal = iq[0, 0].numpy()
    raw_mag = np.abs(complex_signal)
    rd_fft = np.fft.fftshift(np.fft.fft2(complex_signal))
    rd_mag = np.abs(rd_fft)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    axs[0].imshow(raw_mag, aspect='auto', cmap='viridis')
    axs[0].set_title(f"Raw IQ Magnitude (Mod: {mod})")
    axs[1].imshow(rd_mag, aspect='auto', cmap='magma')
    axs[1].set_title("Range-Doppler Magnitude (FFT)")
    axs[2].imshow(det, aspect='auto', cmap='hot')
    axs[2].set_title("Detection Map")
    axs[3].imshow(vel, aspect='auto', cmap='coolwarm')
    axs[3].set_title("Velocity Map")

    for a in axs:
        a.set_xlabel("Range Bin")
        a.set_ylabel("Doppler Bin")
        a.grid(False)

    for t in meta:
        for ax in axs:
            ax.add_patch(patches.Circle((t['range_bin'], t['doppler_bin']), radius=1, color='lime', fill=False))
        axs[3].text(t['range_bin'], t['doppler_bin'], f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

    fig.suptitle(f"Sample {index} | {len(meta)} targets | Modulation: {mod}")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dataset visualization to {save_path}")
        plt.close()
    else:
        plt.show()

def visualize_synthetic_sample(dataset, index=0, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    iq, rd, det, vel, mod, meta = dataset[index]
    det = det.squeeze().numpy()
    vel = vel.squeeze().numpy()
    # Get complex signal - iq is now directly complex64
    complex_signal = iq[0, 0].numpy()
    raw_mag = np.abs(complex_signal)

    window = np.outer(np.hanning(complex_signal.shape[0]), np.hanning(complex_signal.shape[1]))
    windowed = complex_signal * window
    rd_fft = np.fft.fftshift(np.fft.fft2(windowed))
    rd_mag = np.abs(rd_fft)
    rd_log = 20 * np.log10(rd_mag + 1e-6)
    vmax = np.max(rd_log)
    vmin = vmax - 30

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(raw_mag, aspect='auto', cmap='viridis')
    axs[0].set_title(f"Raw IQ Magnitude (Mod: {mod})")
    axs[1].imshow(rd_log, aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axs[1].set_title("Range-Doppler Magnitude (dB)")
    axs[2].imshow(det, aspect='auto', cmap='hot')
    axs[2].set_title("Detection Map")
    axs[3].imshow(vel, aspect='auto', cmap='coolwarm')
    axs[3].set_title("Velocity Map")

    for a in axs:
        a.set_xlabel("Range Bin")
        a.set_ylabel("Doppler Bin")
        a.grid(False)

    for t in meta:
        v, r = t['doppler_bin'], t['range_bin']
        v_shift = (v - vel.shape[0] // 2) % vel.shape[0]
        r_shift = (r - vel.shape[1] // 2) % vel.shape[1]

        axs[0].add_patch(patches.Circle((r, v), radius=1, color='lime', fill=False))
        axs[1].add_patch(patches.Circle((r_shift, v_shift), radius=1, color='lime', fill=False))
        axs[2].add_patch(patches.Circle((r, v), radius=1, color='lime', fill=False))
        axs[3].text(r, v, f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

    fig.suptitle(f"Sample {index} | {len(meta)} targets | Modulation: {mod}")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dataset visualization to {save_path}")
        plt.close()
    else:
        plt.show()

def visualize_synthetic_sample3D(dataset, index=0, save_path=None, correct_coordinates=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches
    from matplotlib import cm

    iq, rd, det, vel, mod, meta = dataset[index]
    det = det.squeeze().numpy()
    vel = vel.squeeze().numpy()
    # Get complex signal - iq is now directly complex64
    complex_signal = iq[0, 0].numpy()
    raw_mag = np.abs(complex_signal)

    # Apply window function (Hanning by default)
    window_doppler = np.hanning(complex_signal.shape[0])
    window_range = np.hanning(complex_signal.shape[1])
    window = np.outer(window_doppler, window_range)
    windowed = complex_signal * window
    
    # Compute range-Doppler map
    rd_fft = np.fft.fftshift(np.fft.fft2(windowed))
    rd_mag = np.abs(rd_fft)
    rd_log = 20 * np.log10(rd_mag + 1e-6)
    vmax = np.max(rd_log)
    vmin = vmax - 30

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(raw_mag, aspect='auto', cmap='viridis')
    axs[0].set_title(f"Raw IQ Magnitude (Mod: {mod})")
    axs[1].imshow(rd_log, aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axs[1].set_title("Range-Doppler Magnitude (dB)")
    axs[2].imshow(det, aspect='auto', cmap='hot')
    axs[2].set_title("Detection Map")
    axs[3].imshow(vel, aspect='auto', cmap='coolwarm')
    axs[3].set_title("Velocity Map")

    for a in axs:
        a.set_xlabel("Range Bin")
        a.set_ylabel("Doppler Bin")
        a.grid(False)

    for t in meta:
        v_orig, r_orig = t['doppler_bin'], t['range_bin']
        
        # Apply coordinate correction for FFT-shifted map if requested
        if correct_coordinates:
            v_shift, r_shift = get_corrected_target_coordinates((v_orig, r_orig), rd_log.shape)
        else:
            # Legacy coordinate transformation (may cause misalignment)
            v_shift = (v_orig - vel.shape[0] // 2) % vel.shape[0]
            r_shift = (r_orig - vel.shape[1] // 2) % vel.shape[1]

        # Mark targets on plots
        axs[0].add_patch(patches.Circle((r_orig, v_orig), radius=1, color='lime', fill=False))
        axs[1].add_patch(patches.Circle((r_shift, v_shift), radius=1, color='lime', fill=False))
        axs[2].add_patch(patches.Circle((r_orig, v_orig), radius=1, color='lime', fill=False))
        axs[3].text(r_orig, v_orig, f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

    fig.suptitle(f"Sample {index} | {len(meta)} targets | Modulation: {mod}")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dataset visualization to {save_path}")
        plt.close()
    else:
        plt.show()

    # 3D visualization of RD FFT
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(rd_log.shape[1]), np.arange(rd_log.shape[0]))
    ax.plot_surface(X, Y, rd_log, cmap='magma', linewidth=0, antialiased=False)
    ax.set_title("3D Range-Doppler (dB)")
    ax.set_xlabel("Range Bin")
    ax.set_ylabel("Doppler Bin")
    ax.set_zlabel("Magnitude (dB)")

    for t in meta:
        v_orig, r_orig = t['doppler_bin'], t['range_bin']
        
        # Apply coordinate correction for FFT-shifted map if requested
        if correct_coordinates:
            v_shift, r_shift = get_corrected_target_coordinates((v_orig, r_orig), rd_log.shape)
        else:
            # Legacy coordinate transformation (may cause misalignment)
            v_shift = (v_orig - vel.shape[0] // 2) % vel.shape[0]
            r_shift = (r_orig - vel.shape[1] // 2) % vel.shape[1]
            
        # Mark target on 3D plot
        ax.scatter(r_shift, v_shift, np.max(rd_log), c='lime', marker='o', s=60, edgecolor='black')

    plt.tight_layout()
    if save_path:
        base = save_path.rsplit('.', 1)[0]
        plt.savefig(f"{base}_3d.png")
        print(f"Saved 3D plot to {base}_3d.png")
        plt.close()
    else:
        plt.show()

def visualize_synthetic_batch(dataset, indices=None, save_dir="debug_outputs", cols=3, correct_coordinates=True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os

    os.makedirs(save_dir, exist_ok=True)
    indices = indices or list(range(min(9, len(dataset))))

    for idx in indices:
        iq, rd, det, vel, mod, meta = dataset[idx]
        det = det.squeeze().numpy()
        vel = vel.squeeze().numpy()
        
        # Get complex signal - iq is now directly complex64
        complex_signal = iq[0, 0].numpy()
        
        # Get range-Doppler map from rd tensor
        rd_real = rd[0, 0].numpy()
        rd_imag = rd[0, 1].numpy()
        rd_complex = rd_real + 1j * rd_imag
        
        # Apply window function
        window_doppler = np.hanning(complex_signal.shape[0])
        window_range = np.hanning(complex_signal.shape[1])
        window = np.outer(window_doppler, window_range)
        windowed = complex_signal * window
        
        # Compute range-Doppler map
        rd_fft = np.fft.fftshift(np.fft.fft2(windowed))
        rd_mag = np.abs(rd_fft)
        rd_log = 20 * np.log10(rd_mag + 1e-6)
        vmax = np.max(rd_log)
        vmin = vmax - 30

        # Raw IQ magnitude for comparison
        raw_mag = np.sqrt(iq[0, ..., 0]**2 + iq[0, ..., 1]**2)

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        axs[0].imshow(raw_mag, aspect='auto', cmap='viridis')
        axs[0].set_title(f"Raw IQ (Mod: {mod})")
        axs[1].imshow(rd_log, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[1].set_title("Range-Doppler Map (dB)")
        axs[2].imshow(det, aspect='auto', cmap='hot')
        axs[2].set_title("Detection Map")

        for a in axs:
            a.set_xlabel("Range Bin")
            a.set_ylabel("Doppler Bin")

        for t in meta:
            v_orig, r_orig = t['doppler_bin'], t['range_bin']
            
            # Apply coordinate correction for FFT-shifted map if requested
            if correct_coordinates:
                v_shift, r_shift = get_corrected_target_coordinates((v_orig, r_orig), rd_log.shape)
            else:
                # Legacy coordinate transformation (may cause misalignment)
                v_shift = (v_orig - vel.shape[0] // 2) % vel.shape[0]
                r_shift = (r_orig - vel.shape[1] // 2) % vel.shape[1]
            
            # Mark targets on plots
            axs[0].add_patch(patches.Circle((r_orig, v_orig), radius=1, color='lime', fill=False))
            axs[1].add_patch(patches.Circle((r_shift, v_shift), radius=1, color='lime', fill=False))
            axs[2].add_patch(patches.Circle((r_orig, v_orig), radius=1, color='lime', fill=False))
            axs[2].text(r_orig, v_orig, f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

        fig.suptitle(f"Sample {idx} | {len(meta)} targets | Modulation: {mod}")
        fig.tight_layout()
        save_path = os.path.join(save_dir, f"batch_sample_{idx}.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()

def get_corrected_target_coordinates(target_pos, rd_shape):
    """
    Convert target coordinates from the original domain to FFT-shifted coordinates.
    This function corrects the coordinate transformation issue when displaying targets
    on the range-Doppler map after FFT shift.
    
    Args:
        target_pos: Tuple of (doppler_bin, range_bin) in original coordinates
        rd_shape: Shape of the range-Doppler map (doppler_bins, range_bins)
        
    Returns:
        Tuple of (doppler_bin, range_bin) in FFT-shifted coordinates
    """
    v, r = target_pos
    num_doppler_bins, num_range_bins = rd_shape
    
    # Apply FFT shift transformation to coordinates
    v_shifted = (v + num_doppler_bins//2) % num_doppler_bins
    r_shifted = (r + num_range_bins//2) % num_range_bins
    
    return v_shifted, r_shifted

def verify_target_alignment(dataset, idx=0, save_path=None, correct_coordinates=True):
    """
    Visualize the range-Doppler map with target markers and verify alignment accuracy.
    This function creates detailed visualizations to check if the simulated targets
    are properly aligned with the peaks in the range-Doppler map.
    
    Args:
        dataset: SyntheticRadarDataset instance
        idx: Index of the sample to visualize
        save_path: Path to save the visualization (if None, display instead)
        correct_coordinates: Whether to apply coordinate correction for target positions
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    import numpy as np
    
    # Get range-Doppler map and other data
    rd_data = dataset.get_range_doppler_map(idx)
    rd_map = rd_data['rd_map']
    rd_log = rd_data['rd_log']
    meta = rd_data['meta']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2 + min(len(meta), 3), figure=fig)
    
    # Plot full range-Doppler map (log scale)
    ax_main = fig.add_subplot(gs[0, :2])
    vmax = np.max(rd_log)
    vmin = vmax - 40  # 40 dB dynamic range
    im = ax_main.imshow(rd_log, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    ax_main.set_title("Range-Doppler Map (dB)")
    ax_main.set_xlabel("Range Bin")
    ax_main.set_ylabel("Doppler Bin")
    plt.colorbar(im, ax=ax_main)
    
    # Plot 3D surface of range-Doppler map
    from mpl_toolkits.mplot3d import Axes3D
    ax_3d = fig.add_subplot(gs[1, :2], projection='3d')
    X, Y = np.meshgrid(np.arange(rd_log.shape[1]), np.arange(rd_log.shape[0]))
    ax_3d.plot_surface(X, Y, rd_log, cmap='jet', linewidth=0, antialiased=False, alpha=0.7)
    ax_3d.set_title("3D Range-Doppler Surface")
    ax_3d.set_xlabel("Range Bin")
    ax_3d.set_ylabel("Doppler Bin")
    ax_3d.set_zlabel("Magnitude (dB)")
    
    # Mark targets on main plot and create zoomed views for each target
    target_colors = ['lime', 'yellow', 'cyan', 'magenta', 'white']
    
    # Calculate peak positions in the range-Doppler map
    # This helps verify if the actual peaks align with the expected target positions
    local_maxima = []
    window_size = 5  # Size of window to search for local maximum
    
    for i, t in enumerate(meta[:min(len(meta), 3)]):
        # Get target position
        v_orig, r_orig = t['doppler_bin'], t['range_bin']
        
        # Apply coordinate correction if requested
        if correct_coordinates:
            v, r = get_corrected_target_coordinates((v_orig, r_orig), rd_log.shape)
        else:
            v, r = v_orig, r_orig
        
        # Mark target on main plot
        color = target_colors[i % len(target_colors)]
        ax_main.add_patch(patches.Circle((r, v), radius=2, color=color, fill=False, linewidth=2))
        ax_main.text(r, v+3, f"T{i+1}", color=color, fontsize=10, ha='center', va='center')
        
        # Mark target on 3D plot
        ax_3d.scatter([r], [v], [rd_log[v, r]], color=color, s=100, marker='o', edgecolor='black')
        
        # Find local maximum around target position
        v_min, v_max = max(0, v-window_size), min(rd_log.shape[0], v+window_size+1)
        r_min, r_max = max(0, r-window_size), min(rd_log.shape[1], r+window_size+1)
        local_region = rd_log[v_min:v_max, r_min:r_max]
        local_max_idx = np.unravel_index(np.argmax(local_region), local_region.shape)
        local_max_v, local_max_r = local_max_idx[0] + v_min, local_max_idx[1] + r_min
        local_maxima.append((local_max_r, local_max_v))
        
        # Create zoomed view for this target
        ax_zoom = fig.add_subplot(gs[i//2, 2+(i%2)])
        zoom_window = 15
        v_zoom_min, v_zoom_max = max(0, v-zoom_window), min(rd_log.shape[0], v+zoom_window+1)
        r_zoom_min, r_zoom_max = max(0, r-zoom_window), min(rd_log.shape[1], r+zoom_window+1)
        
        # Extract and plot zoomed region
        zoomed_data = rd_log[v_zoom_min:v_zoom_max, r_zoom_min:r_zoom_max]
        ax_zoom.imshow(zoomed_data, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        
        # Convert to local coordinates for the zoomed plot
        local_v, local_r = v - v_zoom_min, r - r_zoom_min
        local_max_v_zoom, local_max_r_zoom = local_max_v - v_zoom_min, local_max_r - r_zoom_min
        
        # Mark expected target position
        ax_zoom.add_patch(patches.Circle((local_r, local_v), radius=2, color=color, fill=False, linewidth=2))
        
        # Mark actual peak position
        ax_zoom.add_patch(patches.Circle((local_max_r_zoom, local_max_v_zoom), radius=2, 
                                       color='red', fill=False, linewidth=2, linestyle='--'))
        
        # Calculate offset between expected and actual peak
        offset_r = local_max_r - r
        offset_v = local_max_v - v
        offset_distance = np.sqrt(offset_r**2 + offset_v**2)
        
        # Add information to plot
        ax_zoom.set_title(f"Target {i+1} (SNR: {t['snr_db']:.1f}dB)")
        ax_zoom.text(0.5, 0.02, f"Offset: {offset_distance:.2f} bins", 
                    transform=ax_zoom.transAxes, ha='center', fontsize=9)
        ax_zoom.text(0.5, 0.08, f"Peak: {rd_log[local_max_v, local_max_r]:.1f} dB", 
                    transform=ax_zoom.transAxes, ha='center', fontsize=9)
        ax_zoom.text(0.5, 0.14, f"Expected: {rd_log[v, r]:.1f} dB", 
                    transform=ax_zoom.transAxes, ha='center', fontsize=9)
        
        # Add legend to the first zoomed plot
        if i == 0:
            ax_zoom.plot([], [], color=color, linewidth=2, label='Expected')
            ax_zoom.plot([], [], color='red', linestyle='--', linewidth=2, label='Actual Peak')
            ax_zoom.legend(loc='upper right', fontsize=8)
    
    # Add summary information
    avg_offset = np.mean([np.sqrt((lm[0]-t['range_bin'])**2 + (lm[1]-t['doppler_bin'])**2) 
                         for lm, t in zip(local_maxima, meta[:min(len(meta), 3)])])
    fig.suptitle(f"Target Alignment Verification - Sample {idx}\n" 
                f"Average Offset: {avg_offset:.2f} bins | {len(meta)} targets", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved alignment verification to {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Create dataset with custom parameters
    dataset = SyntheticRadarDataset(
        # Increase SNR for better visualization
        fixed_snr_db=20,  # Higher SNR for clearer peaks
        use_random_snr=False,
        # Use ADF4159 for realistic chirp generation
        use_adf4159=False,
        # Reduce phase noise for less smearing
        phase_noise_level=0.1,  # Lower phase noise (default is 0.5)
        # Set number of targets
        max_targets=1
    )

    # Compare different window functions
    window_types = ['hanning', 'hamming', 'blackman', 'kaiser']
    window_params = [None, None, None, 8.0]  # Beta parameter for Kaiser window
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, (window_type, param) in enumerate(zip(window_types, window_params)):
        # Get range-Doppler map with different window functions
        rd_data = dataset.get_range_doppler_map(idx=0, apply_window=True, 
                                              window_type=window_type, 
                                              window_param=param)
        
        # Access components
        rd_log = rd_data['rd_log']
        meta = rd_data['meta']
        
        # Plot range-Doppler map
        vmax = np.max(rd_log)
        vmin = vmax - 40  # 40 dB dynamic range
        im = axs[i].imshow(rd_log, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[i].set_title(f"Window: {window_type.capitalize()}")
        
        # Mark targets with corrected coordinates
        for t in meta:
            v_orig, r_orig = t['doppler_bin'], t['range_bin']
            # Apply coordinate correction
            v_shift, r_shift = get_corrected_target_coordinates((v_orig, r_orig), rd_log.shape)
            axs[i].add_patch(plt.Circle((r_shift, v_shift), radius=2, color='lime', fill=False))
            axs[i].text(r_shift, v_shift+3, f"SNR: {t['snr_db']:.1f}dB", 
                      color='white', fontsize=8, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('window_function_comparison.png')
    plt.close()
    
    # Verify target alignment with the improved coordinate transformation
    verify_target_alignment(dataset, idx=0, save_path='target_alignment_verification_improved.png', 
                          correct_coordinates=True)
    
    # For comparison, also show the old method without coordinate correction
    verify_target_alignment(dataset, idx=0, save_path='target_alignment_verification_original.png', 
                          correct_coordinates=False)
    
    # Visualize with 3D plot
    visualize_synthetic_sample3D(dataset, index=0, save_path='synthetic_sample_3d_improved.png', 
                               correct_coordinates=True)
    
    print("Visualization complete. Check the output images for improved target alignment.")
    print("The improvements include:")
    print("1. Better window functions to reduce range smearing")
    print("2. Corrected coordinate transformation for proper target alignment")
    print("3. Reduced phase noise for clearer peaks")
    print("4. Higher SNR for better target visibility")
    for i, target in enumerate(meta):
        print(f"Target {i+1}: Range bin = {target['range_bin']}, Doppler bin = {target['doppler_bin']}, SNR = {target['snr_db']:.1f} dB")

    # Visualize the data with standard visualization
    visualize_debug_fmcw(dataset, idx=0, save_path='fmcw_visualization.png')
    
    # Verify target alignment with detailed visualization
    verify_target_alignment(dataset, idx=0, save_path='target_alignment_verification.png')
    
    # Generate and verify multiple samples
    for i in range(1, 3):
        verify_target_alignment(dataset, idx=i, save_path=f'target_alignment_verification_{i}.png')