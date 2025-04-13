import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import random
from scipy.signal import chirp
from tqdm import tqdm
IMG_FORMAT=".pdf" #".png"


class RadarDataset(Dataset):
    def __init__(self, 
                 datapath=None,
                 num_samples=10000, 
                 num_range_bins=64, 
                 num_doppler_bins=12, 
                 snr_min=5, 
                 snr_max=30, 
                 max_targets=3,
                 training=False, 
                 drawfig=False, 
                 save_data=True,
                 savedataformat = 'hdf5', #or numpy
                 save_path='./data/radar',
                 # Parameters from real device (radarconfig.yaml)
                 sample_rate=3e6,        # Sample rate in Hz (from radarconfig.yaml)
                 chirp_duration=500e-6,   # Chirp duration in seconds (ramp_time: 500 us)
                 num_chirps=1,           # Number of chirps (from radarconfig.yaml)
                 bandwidth=500e6,        # Bandwidth in Hz (default_chirp_bw from radarconfig.yaml)
                 center_freq=2.1e9,      # Center frequency in Hz (from radarconfig.yaml)
                 num_rx=4,               # Number of RX antennas (typical for Phaser)
                 num_tx=1,               # Number of TX antennas
                 signal_type='OFDM',     # Signal type (from radarconfig.yaml)
                 signal_freq=1e6,        # Signal frequency for FMCW modulation (Hz)
                 use_lazy_loading=False,  # Enable lazy loading for HDF5 files
                 use_memory_mapping=False, # Enable memory mapping for NumPy files
                 cache_size=100,         # Number of samples to cache when using lazy loading
                 apply_realistic_effects=True,  # Apply realistic RF impairments
                 recalculate_rd_map=True):       # Recalculate RD map after applying effects
        """
        Dataset for radar range-Doppler data
        
        Args:
            datapath: Path to load existing data, if None, generate new data
            num_samples: Number of samples to generate
            num_range_bins: Number of range bins (width)
            num_doppler_bins: Number of Doppler bins (height)
            snr_min: Minimum SNR for targets
            snr_max: Maximum SNR for targets
            max_targets: Maximum number of targets in a scene
            training: Whether this dataset is for training
            drawfig: Whether to draw figures for visualization
            save_data: Whether to save generated data
            savedataformat: Format to save data ('hdf5' or 'numpy')
            save_path: Path to save generated data
            sample_rate: Sample rate in Hz
            chirp_duration: Chirp duration in seconds
            num_chirps: Number of chirps
            bandwidth: Bandwidth in Hz
            center_freq: Center frequency in Hz
            num_rx: Number of RX antennas
            num_tx: Number of TX antennas
            signal_type: Type of radar signal ('FMCW', 'OFDM', or 'Sine')
            signal_freq: Signal frequency for FMCW modulation (Hz)
            use_lazy_loading: Whether to use lazy loading for HDF5 files
            use_memory_mapping: Whether to use memory mapping for NumPy files
            cache_size: Number of samples to cache when using lazy loading
            apply_realistic_effects: Whether to apply realistic RF impairments
            recalculate_rd_map: Whether to recalculate RD map after applying effects
        """
        # Store parameters
        self.training = training
        self.drawfig = drawfig
        self.num_samples = num_samples
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.max_targets = max_targets
        self.save_path = save_path
        
        # Store SDR parameters
        self.sample_rate = sample_rate
        self.chirp_duration = chirp_duration
        self.num_chirps = num_chirps
        self.bandwidth = bandwidth
        self.center_freq = center_freq
        self.signal_type = signal_type
        self.signal_freq = signal_freq
        self.num_rx = num_rx
        self.num_tx = num_tx
        
        # Store realistic effects parameters
        self.apply_realistic_effects = apply_realistic_effects
        self.recalculate_rd_map = recalculate_rd_map
        
        # Calculate derived parameters
        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        self.range_resolution = 3e8 / (2 * self.bandwidth)  # c / (2 * B)
        self.max_range = self.range_resolution * self.num_range_bins
        self.velocity_resolution = 3e8 / (2 * self.center_freq * self.chirp_duration * self.num_chirps)
        self.max_velocity = self.velocity_resolution * (self.num_doppler_bins // 2)
        self.wavelength = 3e8 / self.center_freq  # Wavelength in meters
        self.speed_of_light = 3e8  # Speed of light in m/s
        
        # Initialize lazy loading parameters
        self.use_lazy_loading = use_lazy_loading
        self.use_memory_mapping = use_memory_mapping
        self.h5_file = None
        self.data_cache = {} if use_lazy_loading else None
        self.cache_size = cache_size

        if datapath is not None:
            self._load_data(datapath)
        else:
            print("Generating new radar data")
            self.generate_radar_data(save_data, format=savedataformat)

    def _add_target(self, rx_signal, distance, velocity, rcs):
        """Add a target to the received signal with realistic parameters
        
        Args:
            rx_signal: Received signal array [num_rx, num_chirps, samples_per_chirp]
            distance: Target distance in meters
            velocity: Target velocity in m/s
            rcs: Radar cross-section (signal strength)
        """
        # Calculate time delay based on distance (two-way propagation)
        delay_s = 2 * distance / self.speed_of_light
        
        # Calculate delay in samples
        delay_samples = int(delay_s * self.sample_rate)
        
        # Calculate Doppler shift based on velocity
        # Doppler frequency shift = 2 * velocity * carrier_freq / speed_of_light
        doppler_freq = 2 * velocity * self.center_freq / self.speed_of_light
        
        # Calculate phase shift per chirp due to Doppler
        doppler_phase_per_chirp = 2 * np.pi * doppler_freq * self.chirp_duration
        
        # Generate the chirp signal
        tx_chirp = self._generate_chirp()
        
        # Calculate signal power based on radar equation and RCS
        # Using realistic power levels similar to the real device
        # In real radar, power decreases with distance^4
        # Phaser device typically uses power levels around -10 to -30 dBm
        # Convert RCS to power using radar equation approximation
        distance_factor = (self.max_range / max(distance, 1.0))**4  # Prevent division by zero
        signal_power = rcs * distance_factor
        
        # Scale signal power to match real device characteristics
        # Based on typical SDR rx_gain values (30 dB) from myradar4.py
        signal_amplitude = np.sqrt(signal_power)
        
        # Add the target to each receiver and chirp
        for rx in range(self.num_rx):
            for chirp in range(self.num_chirps):
                # Apply Doppler phase shift for this chirp
                doppler_phase = chirp * doppler_phase_per_chirp
                
                # Apply phase shift due to different antenna positions (for beamforming simulation)
                # This simulates the phase difference between RX antennas
                antenna_spacing = 0.014  # 14mm spacing from Phaser device
                rx_phase = 2 * np.pi * rx * antenna_spacing * np.sin(0) * self.center_freq / self.speed_of_light
                
                # Delayed and phase-shifted signal
                if delay_samples < self.samples_per_chirp:
                    # Create delayed signal with proper phase
                    delayed_signal = np.zeros_like(tx_chirp)
                    delayed_signal[delay_samples:] = tx_chirp[:self.samples_per_chirp-delay_samples]
                    
                    # Apply Doppler and antenna phase shifts
                    phase_shifted_signal = delayed_signal * np.exp(1j * (doppler_phase + rx_phase))
                    
                    # Scale by signal amplitude and add to received signal
                    rx_signal[rx, chirp, :] += signal_amplitude * phase_shifted_signal
        
        return rx_signal

    def generate_radar_data(self, save_data=True, format='hdf5'):
        """Generate synthetic radar data with targets at random positions
        
        Args:
            save_data (bool): Whether to save the generated data
            format (str): Format to save data in ('hdf5' or 'numpy')
        """
        # Initialize arrays for data and labels
        # Time domain data: [num_samples, num_rx, num_chirps, samples_per_chirp, 2]
        # where the last dimension is for I/Q data
        self.time_domain_data = np.zeros(
            (self.num_samples, self.num_rx, self.num_chirps, self.samples_per_chirp, 2), 
            dtype=np.float32
        ) #(10, 2, 12, 20, 2)
        
        # Range-Doppler maps: [num_samples, 2, num_doppler_bins, num_range_bins]
        self.range_doppler_maps = np.zeros(
            (self.num_samples, 2, self.num_doppler_bins, self.num_range_bins), 
            dtype=np.float32
        ) #(10, 2, 12, 64)
        
        # Target masks: [num_samples, num_doppler_bins, num_range_bins, 1]
        self.target_masks = np.zeros(
            (self.num_samples, self.num_doppler_bins, self.num_range_bins, 1), 
            dtype=np.float32
        ) #(10, 12, 64, 1)
        
        # Target information for each sample
        self.target_info = []
        
        # Generate data for each sample
        for i in tqdm(range(self.num_samples), desc="Generating radar data", unit="samples"):
            # Determine number of targets for this sample (1 to max_targets)
            num_targets = random.randint(1, self.max_targets)
            
            # Store target information for this sample
            sample_targets = []
            
            # Generate time domain chirp signal (transmitted signal)
            tx_chirp = self._generate_chirp() #(20,) complex
            
            # Generate complex range-Doppler map with noise
            noise_power = 1.0
            
            # Initialize received signal with noise for each RX antenna
            rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), dtype=np.complex64) #(2, 12, 20)
            for rx in range(self.num_rx):
                for chirp in range(self.num_chirps):
                    noise_real = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                    rx_signal[rx, chirp, :] = noise_real + 1j * noise_imag
            
            # Add targets
            for _ in range(num_targets):
                # Random target distance (in meters)
                # Use realistic range based on the radar configuration
                # Max range is determined by the radar parameters
                distance = random.uniform(1.0, self.max_range * 0.9)  # Avoid the very edge
                
                # Random target velocity (in m/s)
                # Typical velocities for indoor/short-range applications
                max_velocity = self.wavelength / (4 * self.chirp_duration)  # Maximum unambiguous velocity
                velocity = random.uniform(-max_velocity * 0.8, max_velocity * 0.8)  # Avoid aliasing
                
                # Random target RCS (radar cross-section)
                # Use SNR values from the real device configuration
                # Convert SNR to linear scale
                snr_db = random.uniform(self.snr_min, self.snr_max)
                rcs = 10**(snr_db/10)  # Convert dB to linear scale
                
                # Add target to the received signal
                rx_signal = self._add_target(rx_signal, distance, velocity, rcs)
                
                # Calculate range and Doppler bin for this target
                range_bin = int(distance / self.range_resolution)
                if range_bin >= self.num_range_bins:
                    range_bin = self.num_range_bins - 1
                
                doppler_bin = int((velocity / self.max_velocity + 0.5) * self.num_doppler_bins)
                if doppler_bin >= self.num_doppler_bins:
                    doppler_bin = self.num_doppler_bins - 1
                elif doppler_bin < 0:
                    doppler_bin = 0
                
                # Store target information
                target_info = {
                    'distance': distance,
                    'velocity': velocity,
                    'rcs': rcs,
                    'snr': snr_db,  # Add SNR in dB to target info
                    'range_bin': range_bin,
                    'doppler_bin': doppler_bin
                }
                sample_targets.append(target_info)
                
                # Add target to mask with a small spread to simulate real radar response
                # The target response typically spans multiple range-Doppler bins
                spread = 1  # Number of bins to spread the target
                for r in range(max(0, range_bin-spread), min(self.num_range_bins, range_bin+spread+1)):
                    for d in range(max(0, doppler_bin-spread), min(self.num_doppler_bins, doppler_bin+spread+1)):
                        # Gaussian-like intensity falloff from the center
                        intensity = np.exp(-0.5 * ((r-range_bin)**2 + (d-doppler_bin)**2) / (spread**2))
                        self.target_masks[i, d, r, 0] = max(self.target_masks[i, d, r, 0], intensity)
            
            # Store target information for this sample
            self.target_info.append(sample_targets)
            
            # Convert complex time domain data to I/Q format
            for rx in range(self.num_rx):
                for chirp in range(self.num_chirps):
                    self.time_domain_data[i, rx, chirp, :, 0] = np.real(rx_signal[rx, chirp, :])
                    self.time_domain_data[i, rx, chirp, :, 1] = np.imag(rx_signal[rx, chirp, :])
            
            # Generate range-Doppler map from time domain data
            rd_map = self._time_to_range_doppler(rx_signal) #(2, 12, 20) => (12, 64)
            
            # Store range-Doppler map
            self.range_doppler_maps[i, 0, :, :] = np.real(rd_map)
            self.range_doppler_maps[i, 1, :, :] = np.imag(rd_map)
            
            # Visualize a few samples
            if self.drawfig and i < 3:
                self._visualize_sample(i)
        
        if save_data:
            self._save_data(format=format)

    def generate_radar_freqdata(self, save_data=True):
        """Generate synthetic radar data with targets at random positions"""
        # Initialize arrays for data and labels, (10000, 2, 12, 64)
        self.range_doppler_maps = np.zeros((self.num_samples, 2, self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        self.target_masks = np.zeros((self.num_samples, self.num_doppler_bins, self.num_range_bins, 1), dtype=np.float32)
        #(10000, 12, 64, 1)
        # Generate data for each sample
        for i in range(self.num_samples):
            # Determine number of targets for this sample (1 to max_targets)
            num_targets = random.randint(1, self.max_targets)
            
            # Generate complex range-Doppler map with noise
            noise_power = 1.0
            noise_real = np.random.normal(0, np.sqrt(noise_power/2), (self.num_doppler_bins, self.num_range_bins))
            noise_imag = np.random.normal(0, np.sqrt(noise_power/2), (self.num_doppler_bins, self.num_range_bins))
            rd_map = noise_real + 1j * noise_imag #(12, 64)
            
            # Add targets
            for _ in range(num_targets):
                # Random target position
                range_idx = random.randint(0, self.num_range_bins-1) #38
                doppler_idx = random.randint(0, self.num_doppler_bins-1) #9
                
                # Random SNR for this target
                snr = random.uniform(self.snr_min, self.snr_max)
                target_power = noise_power * 10**(snr/10)
                
                # Random complex amplitude for target
                amplitude = np.sqrt(target_power) * np.exp(1j * random.uniform(0, 2*np.pi))
                
                # Add target to range-Doppler map with some spread (to simulate real targets)
                spread = 1.0  # Spread factor
                for dr in range(-1, 2):
                    for dd in range(-1, 2):
                        r_idx = range_idx + dr
                        d_idx = doppler_idx + dd
                        if 0 <= r_idx < self.num_range_bins and 0 <= d_idx < self.num_doppler_bins:
                            # Decrease amplitude with distance from center
                            dist = np.sqrt(dr**2 + dd**2)
                            if dist == 0:
                                # Mark the center point in the target mask
                                self.target_masks[i, d_idx, r_idx, 0] = 1.0
                            
                            # Add target with reduced amplitude based on distance
                            rd_map[d_idx, r_idx] += amplitude * np.exp(-dist/spread)
            
            # Split into real and imaginary components
            self.range_doppler_maps[i, 0, :, :] = np.real(rd_map)
            self.range_doppler_maps[i, 1, :, :] = np.imag(rd_map)
            
            # Visualize a few samples
            if self.drawfig and i < 3:
                self._visualize_sample(i)
        
        if save_data:
            self._save_data()
    
    def _generate_chirp(self):
        """Generate a chirp signal with parameters matching the real device"""
        # Create time vector for one chirp
        t = np.linspace(0, self.chirp_duration, self.samples_per_chirp)
        
        # Generate chirp signal using parameters from real device
        # Using scipy.signal.chirp for linear frequency modulation
        # f0 = starting frequency, f1 = ending frequency
        f0 = 0  # Start at baseband
        f1 = self.bandwidth  # End at bandwidth
        
        if self.signal_type.upper() == 'FMCW':
            # Generate FMCW chirp signal
            chirp_signal = chirp(t, f0=f0, f1=f1, t1=self.chirp_duration, method='linear')
            
            # Convert to complex signal (analytical signal)
            chirp_complex = chirp_signal * np.exp(1j * 2 * np.pi * self.signal_freq * t)
        
        elif self.signal_type.upper() == 'OFDM':
            # For OFDM, create a signal similar to the one in AIsim_maindataset3.py
            # Define OFDM parameters based on the real system
            fft_size = 76  # Number of subcarriers (from AIsim_maindataset3)
            num_ofdm_symbols = 14  # Number of OFDM symbols per frame
            num_guard_carriers = [5, 6]  # Guard carriers at edges
            dc_null = True  # Null the DC carrier
            pilot_ofdm_symbol_indices = [2, 11]  # Pilot symbol positions
            
            # Calculate effective subcarriers (excluding guards and DC)
            num_effective_subcarriers = fft_size - sum(num_guard_carriers)
            if dc_null:
                num_effective_subcarriers -= 1
            
            # Create resource grid similar to MyResourceGrid in AIsim_maindataset3
            # Initialize empty OFDM resource grid
            resource_grid = np.zeros((num_ofdm_symbols, fft_size), dtype=complex)
            
            # Generate random QPSK data for data subcarriers
            # QPSK constellation points
            qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            
            # Create a mask to identify data, pilot, and null positions
            mask = np.ones((num_ofdm_symbols, fft_size), dtype=int)
            
            # Set pilot symbols
            for pilot_idx in pilot_ofdm_symbol_indices:
                mask[pilot_idx, :] = 2  # Mark as pilot
            
            # Set DC null
            if dc_null:
                mask[:, fft_size // 2] = 3  # Mark as DC null
            
            # Set guard carriers
            mask[:, :num_guard_carriers[0]] = 0  # Left guard
            mask[:, -num_guard_carriers[1]:] = 0  # Right guard
            
            # Fill resource grid with data and pilots
            for i in range(num_ofdm_symbols):
                for j in range(fft_size):
                    if mask[i, j] == 1:  # Data position
                        # Random QPSK symbol
                        resource_grid[i, j] = np.random.choice(qpsk_symbols)
                    elif mask[i, j] == 2:  # Pilot position
                        # Use deterministic pilots for channel estimation
                        resource_grid[i, j] = qpsk_symbols[0]  # Use first QPSK symbol for pilots
            
            # Perform IFFT to convert to time domain (per OFDM symbol)
            time_signal = np.zeros((num_ofdm_symbols, fft_size), dtype=complex)
            for i in range(num_ofdm_symbols):
                time_signal[i] = np.fft.ifft(resource_grid[i])
            
            # Add cyclic prefix
            cyclic_prefix_length = 6  # From AIsim_maindataset3
            time_signal_with_cp = np.zeros((num_ofdm_symbols, fft_size + cyclic_prefix_length), dtype=complex)
            for i in range(num_ofdm_symbols):
                # Copy the end of the symbol to the beginning (cyclic prefix)
                time_signal_with_cp[i, :cyclic_prefix_length] = time_signal[i, -cyclic_prefix_length:]
                # Copy the original symbol
                time_signal_with_cp[i, cyclic_prefix_length:] = time_signal[i]
            
            # Flatten to create a continuous time signal
            chirp_complex = time_signal_with_cp.flatten()
            
            # Trim or pad to match samples_per_chirp
            if len(chirp_complex) > self.samples_per_chirp:
                chirp_complex = chirp_complex[:self.samples_per_chirp]
            elif len(chirp_complex) < self.samples_per_chirp:
                # Pad with zeros
                padding = np.zeros(self.samples_per_chirp - len(chirp_complex), dtype=complex)
                chirp_complex = np.concatenate([chirp_complex, padding])
        elif self.signal_type.upper() == 'SINE':
            # Generate a simple sine wave
            chirp_complex = np.exp(1j * 2 * np.pi * (self.bandwidth/2) * t)
        else:
            # Default to a simple sine wave if signal type is not recognized
            print(f"Warning: Signal type '{self.signal_type}' not recognized. Using default sine wave.")
            chirp_complex = np.exp(1j * 2 * np.pi * (self.bandwidth/2) * t)
        
        # Normalize power
        chirp_complex = chirp_complex / np.sqrt(np.mean(np.abs(chirp_complex)**2))
        
        return chirp_complex
    
    def _time_to_range_doppler_basic(self, rx_signal):
        """Convert time domain data to range-Doppler map"""
        # rx_signal shape: [num_rx, num_chirps, samples_per_chirp]
        
        # Combine RX antennas (simple averaging for now)
        combined_signal = np.mean(rx_signal, axis=0)
        
        # Apply range FFT (along fast time)
        range_fft = np.fft.fft(combined_signal, n=self.num_range_bins, axis=1)
        
        # Apply Doppler FFT (along slow time)
        range_doppler = np.fft.fftshift(np.fft.fft(range_fft, n=self.num_doppler_bins, axis=0), axes=0)
        
        # Transpose to get [num_doppler_bins, num_range_bins]
        range_doppler = range_doppler[:, :self.num_range_bins]
        
        return range_doppler
    
    def _time_to_range_doppler(self, rx_signal):
        """Convert time domain data to range-Doppler map with realistic signal processing"""
        # rx_signal shape: [num_rx, num_chirps, samples_per_chirp]
        
        # Initialize output array for combined range-Doppler map
        combined_rd_map = None
        
        # Process time domain data to create range-Doppler maps
        # This simulates the signal processing chain in the real radar
        for rx in range(self.num_rx):
            # Apply windowing to reduce sidelobes (similar to real radar processing)
            # Using Blackman window as in the real device
            window_range = np.blackman(self.samples_per_chirp)
            window_doppler = np.blackman(self.num_chirps)
            
            # Reshape for 2D FFT processing
            rx_data_windowed = rx_signal[rx, :, :].copy()
            
            # Apply range window to each chirp
            for chirp in range(self.num_chirps):
                rx_data_windowed[chirp, :] *= window_range
            
            # Perform range FFT (first dimension)
            range_fft = np.fft.fft(rx_data_windowed, n=self.num_range_bins, axis=1)
            
            # Apply Doppler window across chirps
            for bin_idx in range(min(range_fft.shape[1], self.num_range_bins)):
                range_fft[:, bin_idx] *= window_doppler
            
            # Perform Doppler FFT (second dimension)
            range_doppler = np.fft.fftshift(np.fft.fft(range_fft, n=self.num_doppler_bins, axis=0), axes=0)
            
            # Take the magnitude
            range_doppler_mag = np.abs(range_doppler)
            
            # Normalize and convert to dB scale (similar to real radar processing)
            if np.max(range_doppler_mag) > 0:  # Avoid division by zero
                range_doppler_db = 20 * np.log10(range_doppler_mag / np.max(range_doppler_mag) + 1e-10)
            else:
                range_doppler_db = np.zeros_like(range_doppler_mag)
            
            # Clip to reasonable dB range as in real radar systems
            range_doppler_db = np.clip(range_doppler_db, -40, 0)
            
            # Normalize to [0, 1] for neural network input
            range_doppler_norm = (range_doppler_db + 40) / 40
            
            # If this is the first RX, initialize the combined map
            if combined_rd_map is None:
                combined_rd_map = range_doppler_norm
            else:
                # Combine with previous RX data (using maximum value)
                combined_rd_map = np.maximum(combined_rd_map, range_doppler_norm)
        
        # Ensure the output has the correct dimensions
        if combined_rd_map.shape[0] != self.num_doppler_bins or combined_rd_map.shape[1] != self.num_range_bins:
            # Resize to the desired dimensions using interpolation
            from scipy.ndimage import zoom
            zoom_factors = (self.num_doppler_bins / combined_rd_map.shape[0], 
                            self.num_range_bins / combined_rd_map.shape[1])
            combined_rd_map = zoom(combined_rd_map, zoom_factors, order=1)
        
        # Return the processed range-Doppler map
        return combined_rd_map

    def time_to_range_doppler_batch(self, time_data):
        """Convert a batch of time domain data to range-Doppler maps"""
        # time_data shape: [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
        batch_size = time_data.shape[0]
        
        # Convert to complex
        complex_data = time_data[..., 0] + 1j * time_data[..., 1]
        
        # Initialize output
        rd_maps = np.zeros((batch_size, 2, self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        
        # Process each sample
        for i in range(batch_size):
            rd_map = self._time_to_range_doppler(complex_data[i])
            rd_maps[i, 0, :, :] = np.real(rd_map)
            rd_maps[i, 1, :, :] = np.imag(rd_map)
        
        return rd_maps

    def _visualize_sample(self, idx):
        """Visualize a sample range-Doppler map, time domain signal, and target mask"""
        # Check if the required attributes exist
        if not hasattr(self, 'range_doppler_maps') or idx >= len(self.range_doppler_maps):
            print(f"Warning: Cannot visualize sample {idx}. Range-Doppler maps not available.")
            return
            
        rd_map = self.range_doppler_maps[idx]
        
        if not hasattr(self, 'target_masks') or idx >= len(self.target_masks):
            print(f"Warning: Cannot visualize sample {idx}. Target masks not available.")
            return
            
        target_mask = self.target_masks[idx, :, :, 0]
        
        # Check if time domain data is available
        has_time_data = hasattr(self, 'time_domain_data') and self.time_domain_data is not None and idx < len(self.time_domain_data)
        
        # Calculate magnitude from real and imaginary parts
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        plt.figure(figsize=(15, 10))
        
        # Plot range-Doppler magnitude
        plt.subplot(2, 2, 1)
        plt.imshow(20*np.log10(rd_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(f'Range-Doppler Map (Sample {idx})')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Plot target mask
        plt.subplot(2, 2, 2)
        plt.imshow(target_mask, aspect='auto', cmap='gray')
        plt.colorbar(label='Target Presence')
        plt.title(f'Target Mask (Sample {idx})')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Plot time domain signal if available
        plt.subplot(2, 2, 3)
        if has_time_data:
            time_data = self.time_domain_data[idx]
            t = np.arange(self.samples_per_chirp) / self.sample_rate * 1e6  # Convert to microseconds
            plt.plot(t, time_data[0, 0, :, 0], label='I')
            plt.plot(t, time_data[0, 0, :, 1], label='Q')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.title(f'Time Domain Signal (RX 0, Chirp 0)')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Time domain data not available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # Plot target information if available
        plt.subplot(2, 2, 4)
        ax = plt.gca()
        ax.axis('off')
        
        # Check if target_info exists and has data for this index
        has_target_info = (hasattr(self, 'target_info') and 
                          self.target_info is not None and 
                          idx < len(self.target_info))
        
        if has_target_info:
            target_text = "Target Information:\n"
            for i, target in enumerate(self.target_info[idx]):
                target_text += f"Target {i+1}:\n"
                target_text += f"  Distance: {target['distance']:.2f} m\n"
                target_text += f"  Velocity: {target['velocity']:.2f} m/s\n"
                target_text += f"  Range Bin: {target['range_bin']}\n"
                target_text += f"  Doppler Bin: {target['doppler_bin']}\n"
                if 'snr' in target:
                    target_text += f"  SNR: {target['snr']:.2f} dB\n"
                elif 'rcs' in target:
                    target_text += f"  RCS: {target['rcs']:.2f}\n"
            plt.text(0, 1, target_text, fontsize=9, verticalalignment='top')
        else:
            plt.text(0.5, 0.5, 'Target information not available yet', 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        os.makedirs('data/radar', exist_ok=True)
        plt.savefig(f'data/radar/radar_sample_{idx}{IMG_FORMAT}')
        plt.close()
    
    def _save_data(self, format='hdf5'):
        """Save generated data to file
        
        Args:
            format (str): Format to save data in ('hdf5' or 'numpy')
        """
        os.makedirs(self.save_path, exist_ok=True)
        print(f"Saving data to file format: {format}")
        if format.lower() == 'hdf5':
            # Use HDF5 format for efficient chunked storage
            try:
                import h5py
            except ImportError:
                print("h5py not installed. Please install with 'pip install h5py' or use format='numpy'")
                format = 'numpy'
        
            # Define the output file path
            file_ext = ".h5"
            output_file = os.path.join(self.save_path, f'radar_simulation_data_{self.signal_type.lower()}{file_ext}')
            #output_file = 'data/radar/radar_simulation_data.h5'
            
            # Save data in HDF5 format with chunking
            with h5py.File(output_file, 'w') as f:
                # Save large arrays with chunking
                # Time domain data
                if self.time_domain_data is not None:
                    time_chunks = (1, self.num_rx, self.num_chirps, self.samples_per_chirp, 2)
                    f.create_dataset('time_domain_data', data=self.time_domain_data, 
                                    chunks=time_chunks, compression='gzip', compression_opts=4)
                
                # Range-Doppler maps
                rd_chunks = (1, 2, self.num_doppler_bins, self.num_range_bins)
                f.create_dataset('range_doppler_maps', data=self.range_doppler_maps, 
                                chunks=rd_chunks, compression='gzip', compression_opts=4)
                
                # Target masks
                mask_chunks = (1, self.num_doppler_bins, self.num_range_bins, 1)
                f.create_dataset('target_masks', data=self.target_masks, 
                                chunks=mask_chunks, compression='gzip', compression_opts=4)
                
                # Save metadata and smaller arrays directly
                f.create_dataset('num_range_bins', data=self.num_range_bins)
                f.create_dataset('num_doppler_bins', data=self.num_doppler_bins)
                f.create_dataset('snr_range', data=[self.snr_min, self.snr_max])
                f.create_dataset('max_targets', data=self.max_targets)
                f.create_dataset('sample_rate', data=self.sample_rate)
                f.create_dataset('chirp_duration', data=self.chirp_duration)
                f.create_dataset('num_chirps', data=self.num_chirps)
                f.create_dataset('bandwidth', data=self.bandwidth)
                f.create_dataset('center_freq', data=self.center_freq)
                f.create_dataset('range_resolution', data=self.range_resolution)
                f.create_dataset('velocity_resolution', data=self.velocity_resolution)
                f.create_dataset('max_range', data=self.max_range)
                f.create_dataset('max_velocity', data=self.max_velocity)
                
                # Save target info as a JSON string (since it's a list of dictionaries)
                import json
                target_info_json = json.dumps(self.target_info)
                f.create_dataset('target_info_json', data=target_info_json)
            
            print(f"Radar simulation data saved to {output_file}")
        else:
            # Use numpy with pickle protocol 4 for large files
            file_ext = ".npy"
            output_file = os.path.join(self.save_path, f'radar_simulation_data_{self.signal_type.lower()}{file_ext}')
            #output_file = 'data/radar/radar_simulation_data.npy'
            data_dict = {
                'time_domain_data': self.time_domain_data,
                'range_doppler_maps': self.range_doppler_maps,
                'target_masks': self.target_masks,
                'target_info': self.target_info,
                'num_range_bins': self.num_range_bins,
                'num_doppler_bins': self.num_doppler_bins,
                'snr_range': [self.snr_min, self.snr_max],
                'max_targets': self.max_targets,
                'sample_rate': self.sample_rate,
                'chirp_duration': self.chirp_duration,
                'num_chirps': self.num_chirps,
                'bandwidth': self.bandwidth,
                'center_freq': self.center_freq,
                'range_resolution': self.range_resolution,
                'velocity_resolution': self.velocity_resolution,
                'max_range': self.max_range,
                'max_velocity': self.max_velocity
            }
            
            # Use protocol=4 to support data larger than 4GB
            np.save(output_file, data_dict, allow_pickle=True, fix_imports=True)
            print(f"Radar simulation data saved to {output_file} using pickle protocol 4")
    
    def _load_data(self, datapath):
        """Load radar data from file with support for lazy loading"""
        # Check if the file is HDF5 format
        is_hdf5 = datapath.endswith('.h5') or datapath.endswith('.hdf5')
        
        if is_hdf5:
            # HDF5 format
            if self.use_lazy_loading:
                # Lazy loading - keep file open and load data on demand
                self.h5_file = h5py.File(datapath, 'r')
                
                # Load metadata
                self.num_range_bins = self.h5_file['num_range_bins'][()]
                self.num_doppler_bins = self.h5_file['num_doppler_bins'][()]
                snr_range = self.h5_file['snr_range'][:]
                self.snr_min, self.snr_max = snr_range
                self.max_targets = self.h5_file['max_targets'][()]
                
                # Load SDR parameters
                self.sample_rate = self.h5_file['sample_rate'][()]
                self.chirp_duration = self.h5_file['chirp_duration'][()]
                self.num_chirps = self.h5_file['num_chirps'][()]
                self.bandwidth = self.h5_file['bandwidth'][()]
                self.center_freq = self.h5_file['center_freq'][()]
                self.range_resolution = self.h5_file['range_resolution'][()]
                self.velocity_resolution = self.h5_file['velocity_resolution'][()]
                self.max_range = self.h5_file['max_range'][()]
                self.max_velocity = self.h5_file['max_velocity'][()]
                
                # Load target info from JSON
                import json
                if 'target_info_json' in self.h5_file:
                    target_info_json = self.h5_file['target_info_json'][()]
                    if isinstance(target_info_json, bytes):
                        target_info_json = target_info_json.decode('utf-8')
                    self.target_info = json.loads(target_info_json)
                else:
                    self.target_info = []
                
                # Initialize cache
                self.data_cache = {}
                
                # Get number of samples
                self.num_samples = self.h5_file['range_doppler_maps'].shape[0]
            else:
                # Load everything into memory
                with h5py.File(datapath, 'r') as f:
                    # Load data
                    self.range_doppler_maps = f['range_doppler_maps'][:]
                    self.target_masks = f['target_masks'][:]
                    
                    # Load time domain data if available
                    if 'time_domain_data' in f:
                        self.time_domain_data = f['time_domain_data'][:]
                    else:
                        self.time_domain_data = None
                    
                    # Load metadata
                    self.num_range_bins = f['num_range_bins'][()]
                    self.num_doppler_bins = f['num_doppler_bins'][()]
                    snr_range = f['snr_range'][:]
                    self.snr_min, self.snr_max = snr_range
                    self.max_targets = f['max_targets'][()]
                    
                    # Load SDR parameters
                    self.sample_rate = f['sample_rate'][()]
                    self.chirp_duration = f['chirp_duration'][()]
                    self.num_chirps = f['num_chirps'][()]
                    self.bandwidth = f['bandwidth'][()]
                    self.center_freq = f['center_freq'][()]
                    self.range_resolution = f['range_resolution'][()]
                    self.velocity_resolution = f['velocity_resolution'][()]
                    self.max_range = f['max_range'][()]
                    self.max_velocity = f['max_velocity'][()]
                    
                    # Load target info from JSON
                    import json
                    if 'target_info_json' in f:
                        target_info_json = f['target_info_json'][()]
                        if isinstance(target_info_json, bytes):
                            target_info_json = target_info_json.decode('utf-8')
                        self.target_info = json.loads(target_info_json)
                    else:
                        self.target_info = []
                
                self.num_samples = len(self.range_doppler_maps)
        else:
            # NumPy format
            if self.use_memory_mapping:
                # Use memory mapping for large NumPy files
                data_dict = np.load(datapath, allow_pickle=True, mmap_mode='r')
                
                # For memory mapping, we need to handle the item() differently
                if isinstance(data_dict, np.ndarray) and data_dict.dtype == np.dtype('O'):
                    # This is a numpy object array containing a dictionary
                    # We need to load it into memory to access its contents
                    data_dict = np.load(datapath, allow_pickle=True).item()
            else:
                # Load everything into memory
                data_dict = np.load(datapath, allow_pickle=True).item()
            
            self.range_doppler_maps = data_dict.get('range_doppler_maps')
            self.target_masks = data_dict['target_masks']
            self.time_domain_data = data_dict.get('time_domain_data')
            self.target_info = data_dict.get('target_info', [])
            self.num_range_bins = data_dict['num_range_bins']
            self.num_doppler_bins = data_dict['num_doppler_bins']
            self.snr_min, self.snr_max = data_dict['snr_range']
            self.max_targets = data_dict['max_targets']
            
            # Load SDR parameters if available
            self.sample_rate = data_dict.get('sample_rate', 1e6)
            self.chirp_duration = data_dict.get('chirp_duration', 20e-6)
            self.num_chirps = data_dict.get('num_chirps', self.num_doppler_bins)
            self.bandwidth = data_dict.get('bandwidth', 100e6)
            self.center_freq = data_dict.get('center_freq', 2.4e9)
            self.range_resolution = data_dict.get('range_resolution')
            self.velocity_resolution = data_dict.get('velocity_resolution')
            self.max_range = data_dict.get('max_range')
            self.max_velocity = data_dict.get('max_velocity')
            
            self.num_samples = len(self.range_doppler_maps)
        
        print(f"Loaded {self.num_samples} radar samples")
        
        # Visualize a few samples if requested
        if self.drawfig:
            for i in range(min(3, self.num_samples)):
                self._visualize_sample(i)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Check if we're using lazy loading with h5py
        if self.use_lazy_loading and self.h5_file is not None:
            # Check if the sample is already in cache
            if idx in self.data_cache:
                sample = self.data_cache[idx]
            else:
                # Lazy loading from HDF5
                sample = {}
                sample['feature_2d'] = np.array(self.h5_file['range_doppler_maps'][idx])
                sample['labels'] = np.array(self.h5_file['target_masks'][idx])
                
                # Load time_domain data if available
                if 'time_domain_data' in self.h5_file:
                    sample['time_domain'] = np.array(self.h5_file['time_domain_data'][idx])
                
                # Load target info if available
                if hasattr(self, 'target_info') and self.target_info and idx < len(self.target_info):
                    sample['target_info'] = self.target_info[idx]
                
                # Add to cache if we're using caching
                # Manage cache size
                if len(self.data_cache) >= self.cache_size:
                    # Remove oldest item if cache is full
                    self.data_cache.pop(next(iter(self.data_cache)))
                self.data_cache[idx] = sample
        else:
            # Original implementation for in-memory data
            rd_feature = self.range_doppler_maps[idx]
            time_feature = self.time_domain_data[idx] if self.time_domain_data is not None else None
            target_mask = self.target_masks[idx]
            
            # Ensure consistent shapes for all samples
            # For range-Doppler maps
            if rd_feature.shape[1] != self.num_doppler_bins or rd_feature.shape[2] != self.num_range_bins:
                # Resize using interpolation if needed
                from scipy.ndimage import zoom
                
                # Calculate zoom factors for each dimension
                zoom_factors = (1, self.num_doppler_bins / rd_feature.shape[1], 
                               self.num_range_bins / rd_feature.shape[2])
                
                # Apply zoom to resize
                rd_feature = zoom(rd_feature, zoom_factors, order=1)
            
            # For target masks
            if target_mask.shape[0] != self.num_doppler_bins or target_mask.shape[1] != self.num_range_bins:
                # Resize using interpolation
                from scipy.ndimage import zoom
                
                # Calculate zoom factors
                zoom_factors = (self.num_doppler_bins / target_mask.shape[0], 
                               self.num_range_bins / target_mask.shape[1], 1)
                
                # Apply zoom to resize
                target_mask = zoom(target_mask, zoom_factors, order=1)
            
            # For time domain data
            if time_feature is not None:
                if (time_feature.shape[1] != self.num_chirps or 
                    time_feature.shape[2] != self.samples_per_chirp):
                    
                    # Resize using interpolation
                    from scipy.ndimage import zoom
                    
                    # Calculate zoom factors
                    zoom_factors = (1, self.num_chirps / time_feature.shape[1], 
                                   self.samples_per_chirp / time_feature.shape[2], 1)
                    
                    # Apply zoom to resize
                    time_feature = zoom(time_feature, zoom_factors, order=1)
                
            sample = {
                'feature_2d': rd_feature.astype(np.float32),  # [2, num_doppler_bins, num_range_bins]
                'labels': target_mask.astype(np.float32),     # [num_doppler_bins, num_range_bins, 1]
                'target_info': self.target_info[idx] if self.target_info else None
            }
            
            if time_feature is not None:
                sample['time_domain'] = time_feature.astype(np.float32)  # [num_rx, num_chirps, samples_per_chirp, 2]
        
        # Apply training augmentations to the loaded data (for both in-memory and h5 data)
        if self.training:
            # Add random noise to make the model more robust
            noise_level = random.uniform(0.05, 0.2)
            
            # Add noise to range-Doppler maps
            if 'feature_2d' in sample:
                noise = np.random.normal(0, noise_level, sample['feature_2d'].shape)
                sample['feature_2d'] = sample['feature_2d'] + noise
            
            # Add noise to time domain data if available
            if 'time_domain' in sample:
                time_noise = np.random.normal(0, noise_level, sample['time_domain'].shape)
                sample['time_domain'] = sample['time_domain'] + time_noise
        
        # Apply realistic RF impairments to time domain data if available
        if 'time_domain' in sample and hasattr(self, 'apply_realistic_effects') and self.apply_realistic_effects:
            sample['time_domain'] = self._apply_realistic_rf_effects(sample['time_domain'], sample.get('target_info'))
            
            # Recalculate range-Doppler map from the modified time domain data
            if self.recalculate_rd_map:
                # Convert I/Q format back to complex
                complex_data = sample['time_domain'][..., 0] + 1j * sample['time_domain'][..., 1]
                
                # Process using the range-Doppler processing chain
                rd_map = self._time_to_range_doppler(complex_data)
                
                # Update the feature_2d with the new range-Doppler map
                sample['feature_2d'][0, :, :] = np.real(rd_map)
                sample['feature_2d'][1, :, :] = np.imag(rd_map)
        
        return sample
    
    def _apply_realistic_rf_effects(self, time_data, target_info=None):
        """Apply realistic RF impairments to time domain data
        
        Args:
            time_data: Time domain data with shape [num_rx, num_chirps, samples_per_chirp, 2]
            target_info: Optional target information for more accurate path loss modeling
            
        Returns:
            Modified time domain data with realistic impairments
        """
        # Make a copy to avoid modifying the original data
        modified_data = time_data.copy()
        
        # Convert I/Q format to complex for easier processing
        complex_data = modified_data[..., 0] + 1j * modified_data[..., 1]
        
        # 1. Apply frequency-dependent path loss
        # Path loss increases with frequency: PL = 20*log10(4*pi*d/λ)
        if target_info is not None:
            for target in target_info:
                # Extract target distance
                distance = target.get('distance', 10.0)  # Default to 10m if not specified
                
                # Calculate wavelength (λ) from center frequency
                wavelength = 3e8 / self.center_freq
                
                # Calculate path loss in dB
                path_loss_db = 20 * np.log10(4 * np.pi * distance / wavelength)
                
                # Convert to linear scale
                path_loss_factor = 10 ** (-path_loss_db / 20)
                
                # Apply distance-dependent attenuation
                # This is a simplified model - in reality, the attenuation would be
                # applied to specific parts of the signal corresponding to this target
                complex_data *= path_loss_factor
        
        # 2. Add phase noise (common in real oscillators)
        # Phase noise is typically higher at lower frequency offsets
        phase_noise_level = 0.05  # radians
        phase_noise = np.random.normal(0, phase_noise_level, complex_data.shape)
        complex_data *= np.exp(1j * phase_noise)
        
        # 3. Add I/Q imbalance (common in real receivers)
        # I/Q imbalance causes amplitude and phase mismatch between I and Q channels
        amplitude_imbalance = np.random.uniform(0.9, 1.1)  # 10% amplitude imbalance
        phase_imbalance = np.random.uniform(-0.1, 0.1)     # 0.1 radians phase imbalance
        
        # Apply I/Q imbalance
        i_component = np.real(complex_data)
        q_component = np.imag(complex_data)
        
        # Imbalanced I/Q
        i_imbalanced = i_component
        q_imbalanced = amplitude_imbalance * q_component * np.exp(1j * phase_imbalance)
        
        # Recombine
        complex_data = i_imbalanced + 1j * np.imag(q_imbalanced)
        
        # 4. Add antenna crosstalk between RX channels
        if complex_data.shape[0] > 1:  # Only if we have multiple RX antennas
            crosstalk_level = 0.05  # 5% crosstalk between adjacent channels
            
            # Create a copy of the original data for crosstalk calculation
            original_data = complex_data.copy()
            
            # Apply crosstalk between adjacent RX antennas
            for rx_idx in range(complex_data.shape[0]):
                # Left neighbor
                if rx_idx > 0:
                    complex_data[rx_idx] += crosstalk_level * original_data[rx_idx-1]
                
                # Right neighbor
                if rx_idx < complex_data.shape[0] - 1:
                    complex_data[rx_idx] += crosstalk_level * original_data[rx_idx+1]
        
        # 5. Add TX to RX leakage (direct coupling between TX and RX antennas)
        # This creates a strong return at zero range
        tx_leakage_level = 0.1  # 10% of TX signal leaks directly to RX
        
        # Generate a simplified TX leakage signal (strongest at the beginning of the chirp)
        leakage_profile = np.exp(-np.arange(complex_data.shape[2]) / (complex_data.shape[2] / 5))
        
        # Apply to all RX antennas and all chirps
        for rx_idx in range(complex_data.shape[0]):
            for chirp_idx in range(complex_data.shape[1]):
                complex_data[rx_idx, chirp_idx] += tx_leakage_level * leakage_profile
        
        # 6. Add frequency-dependent receiver gain variation
        # Real receivers don't have flat frequency response
        freq_response = 1 + 0.2 * np.sin(np.linspace(0, 2*np.pi, complex_data.shape[2]))
        
        # Apply to all RX antennas and all chirps
        for rx_idx in range(complex_data.shape[0]):
            for chirp_idx in range(complex_data.shape[1]):
                complex_data[rx_idx, chirp_idx] *= freq_response
        
        # 7. Add thermal noise (increases with temperature and bandwidth)
        # Thermal noise power = kTB where k is Boltzmann's constant, T is temperature, B is bandwidth
        # We'll use a simplified model with a temperature-dependent noise level
        temperature_kelvin = 290  # Room temperature in Kelvin
        boltzmann_constant = 1.38e-23  # Boltzmann's constant
        noise_power = boltzmann_constant * temperature_kelvin * self.bandwidth
        
        # Scale to make it visible in the simulation
        noise_scale = 1e10  # Scaling factor to make noise visible
        thermal_noise_level = np.sqrt(noise_power * noise_scale)
        
        # Generate complex Gaussian noise
        thermal_noise = (np.random.normal(0, thermal_noise_level, complex_data.shape) + 
                        1j * np.random.normal(0, thermal_noise_level, complex_data.shape))
        
        # Add thermal noise to the signal
        complex_data += thermal_noise
        
        # 8. Add ADC quantization effects
        # Real ADCs have limited bit depth, causing quantization noise
        adc_bits = 12  # Typical for radar systems
        
        # Find the maximum amplitude for scaling
        max_amplitude = np.max(np.abs(complex_data))
        
        # Calculate the quantization step size
        quant_step = max_amplitude / (2**(adc_bits-1))
        
        # Quantize the real and imaginary parts separately
        real_part = np.real(complex_data)
        imag_part = np.imag(complex_data)
        
        # Apply quantization
        real_part_quantized = np.round(real_part / quant_step) * quant_step
        imag_part_quantized = np.round(imag_part / quant_step) * quant_step
        
        # Recombine
        complex_data = real_part_quantized + 1j * imag_part_quantized
        
        # 9. Add occasional phase jumps (common in real PLLs)
        # With a small probability, add a phase jump to simulate PLL glitches
        if np.random.random() < 0.1:  # 10% chance of a phase jump
            jump_location = np.random.randint(0, complex_data.shape[2])
            jump_angle = np.random.uniform(-np.pi/4, np.pi/4)  # Random phase jump
            
            # Apply the phase jump to all samples after the jump location
            for rx_idx in range(complex_data.shape[0]):
                for chirp_idx in range(complex_data.shape[1]):
                    complex_data[rx_idx, chirp_idx, jump_location:] *= np.exp(1j * jump_angle)
        
        # Convert back to I/Q format
        modified_data[..., 0] = np.real(complex_data)
        modified_data[..., 1] = np.imag(complex_data)
        
        return modified_data

    def visualize_realistic_effects(self, idx=0):
        """Visualize the effects of realistic RF impairments on a sample
        
        Args:
            idx: Index of the sample to visualize
        """
        if not hasattr(self, 'apply_realistic_effects') or not self.apply_realistic_effects:
            print("Realistic effects are not enabled. Please set apply_realistic_effects=True")
            return
        
        # Get the original sample
        original_sample = self[idx]
        
        # Temporarily disable realistic effects
        self.apply_realistic_effects = False
        clean_sample = self[idx]
        
        # Re-enable realistic effects
        self.apply_realistic_effects = True
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # 1. Compare time domain signals (first RX, first chirp)
        plt.subplot(3, 2, 1)
        t = np.arange(self.samples_per_chirp) / self.sample_rate * 1e6  # Convert to microseconds
        plt.plot(t, clean_sample['time_domain'][0, 0, :, 0], 'b-', label='Clean I')
        plt.plot(t, clean_sample['time_domain'][0, 0, :, 1], 'r-', label='Clean Q')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('Clean Time Domain Signal (RX 0, Chirp 0)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 2, 2)
        plt.plot(t, original_sample['time_domain'][0, 0, :, 0], 'b-', label='Realistic I')
        plt.plot(t, original_sample['time_domain'][0, 0, :, 1], 'r-', label='Realistic Q')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('Realistic Time Domain Signal (RX 0, Chirp 0)')
        plt.legend()
        plt.grid(True)
        
        # 2. Compare range-Doppler maps
        # Calculate magnitude from real and imaginary parts
        clean_rd_magnitude = np.sqrt(clean_sample['feature_2d'][0]**2 + clean_sample['feature_2d'][1]**2)
        realistic_rd_magnitude = np.sqrt(original_sample['feature_2d'][0]**2 + original_sample['feature_2d'][1]**2)
        
        plt.subplot(3, 2, 3)
        plt.imshow(20*np.log10(clean_rd_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Clean Range-Doppler Map')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        plt.subplot(3, 2, 4)
        plt.imshow(20*np.log10(realistic_rd_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Realistic Range-Doppler Map')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # 3. Show the difference
        difference = 20*np.log10(realistic_rd_magnitude + 1e-10) - 20*np.log10(clean_rd_magnitude + 1e-10)
        
        plt.subplot(3, 2, 5)
        plt.imshow(difference, aspect='auto', cmap='coolwarm')
        plt.colorbar(label='Difference (dB)')
        plt.title('Difference (Realistic - Clean)')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # 4. Show the target mask
        plt.subplot(3, 2, 6)
        plt.imshow(original_sample['labels'][:, :, 0], aspect='auto', cmap='gray')
        plt.colorbar(label='Target Presence')
        plt.title('Target Mask')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        plt.tight_layout()
        os.makedirs('data/radar/realistic_effects', exist_ok=True)
        plt.savefig(f'data/radar/realistic_effects/comparison_{idx}{IMG_FORMAT}')
        plt.close()
        
        print(f"Visualization saved to data/radar/realistic_effects/comparison_{idx}{IMG_FORMAT}")

    def __del__(self):
        # Close the HDF5 file when the dataset is deleted
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            print("Closed HDF5 file")

# Test function to evaluate the RadarDataset class
def test_radar_dataset():
    print("Testing RadarDataset class...")
    
    # Create a small test dataset
    # Parameters are reduced for faster testing
    # Create a small test dataset with parameters from radarconfig.yaml
    test_dataset = RadarDataset(
        num_samples=10,              # Number of radar scenes to generate
        num_range_bins=64,           # Number of range bins (width dimension)
        num_doppler_bins=12,         # Number of Doppler bins (height dimension)
        snr_min=10,                  # Minimum signal-to-noise ratio in dB
        snr_max=25,                  # Maximum signal-to-noise ratio in dB
        max_targets=3,               # Maximum number of targets per scene
        training=False,              # Not in training mode
        drawfig=True,                # Generate visualization figures
        save_data=True,              # Save the generated data
        # Parameters from real device (radarconfig.yaml)
        sample_rate=3e6,             # 3 MHz sampling rate (from radarconfig.yaml)
        chirp_duration=500e-6,       # 500 microsecond chirp (ramp_time from radarconfig.yaml)
        num_chirps=1,                # 1 chirp per frame (from radarconfig.yaml, would be 128 for TDD mode)
        bandwidth=500e6,             # 500 MHz bandwidth (default_chirp_bw from radarconfig.yaml)
        center_freq=2.1e9,           # 2.1 GHz center frequency (from radarconfig.yaml)
        num_rx=4,                    # 4 receive antennas (typical for Phaser)
        num_tx=1,                    # 1 transmit antenna
        signal_type='OFDM'           # OFDM signal type (from radarconfig.yaml)
    )
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"Number of samples: {len(test_dataset)}") #10
    print(f"Range resolution: {test_dataset.range_resolution:.2f} m")
    print(f"Maximum range: {test_dataset.max_range:.2f} m")
    print(f"Velocity resolution: {test_dataset.velocity_resolution:.2f} m/s")
    print(f"Maximum velocity: {test_dataset.max_velocity:.2f} m/s")
    print(f"Samples per chirp: {test_dataset.samples_per_chirp}")
    
    # Data shapes information
    print("\nData Shapes:")
    print(f"Time domain data: {test_dataset.time_domain_data.shape}")
    # Shape: [num_samples, num_rx, num_chirps, samples_per_chirp, 2]
    # Example: (10, 2, 12, 20, 2) - 10 samples, 2 RX, 12 chirps, 20 samples per chirp, 2 for I/Q
    
    print(f"Range-Doppler maps: {test_dataset.range_doppler_maps.shape}")
    # Shape: [num_samples, 2, num_doppler_bins, num_range_bins]
    # Example: (10, 2, 12, 64) - 10 samples, 2 for real/imag, 12 Doppler bins, 64 range bins
    
    print(f"Target masks: {test_dataset.target_masks.shape}")
    # Shape: [num_samples, num_doppler_bins, num_range_bins, 1]
    # Example: (10, 12, 64, 1) - 10 samples, 12 Doppler bins, 64 range bins, 1 channel
    
    # Test the __getitem__ method
    print("\nTesting __getitem__ method:")
    sample_idx = 0
    sample = test_dataset[sample_idx]
    
    print(f"Sample keys: {sample.keys()}") #['feature_2d', 'labels', 'target_info', 'time_domain']
    print(f"Feature 2D shape: {sample['feature_2d'].shape}")  # [2, num_doppler_bins, num_range_bins] (2, 12, 64)
    print(f"Labels shape: {sample['labels'].shape}")          # [num_doppler_bins, num_range_bins, 1] (12, 64, 1)
    print(f"Time domain shape: {sample['time_domain'].shape}")  # [num_rx, num_chirps, samples_per_chirp, 2] (2, 12, 20, 2)
    
    # Test range-Doppler conversion
    print("\nTesting time to range-Doppler conversion:")
    # Take a small batch of time domain data
    batch_size = 3
    time_batch = test_dataset.time_domain_data[:batch_size] #(3, 2, 12, 20, 2)
    # Convert to range-Doppler
    rd_maps = test_dataset.time_to_range_doppler_batch(time_batch)
    print(f"Input time batch shape: {time_batch.shape}") #(3, 2, 12, 64)
    # Shape: [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
    print(f"Output range-Doppler shape: {rd_maps.shape}") #(3, 2, 12, 64)
    # Shape: [batch_size, 2, num_doppler_bins, num_range_bins]
    
        # Create a visualization of the first target's information
    if len(test_dataset.target_info) > 0 and len(test_dataset.target_info[0]) > 0:
        print("\nFirst sample target information:")
        for i, target in enumerate(test_dataset.target_info[0]):
            print(f"Target {i+1}:")
            print(f"  Distance: {target['distance']:.2f} m")
            print(f"  Velocity: {target['velocity']:.2f} m/s")
            print(f"  Range bin: {target['range_bin']}")
            print(f"  Doppler bin: {target['doppler_bin']}")
            print(f"  SNR: {target['snr']:.2f} dB")
    
    # Create additional visualizations and save to local folder
    print("\nGenerating additional visualizations...")
    os.makedirs('data/radar/visualizations', exist_ok=True)
    
    # 1. Range-Doppler heatmap with targets marked
    for idx in range(min(3, len(test_dataset))):
        plt.figure(figsize=(10, 8))
        
        # Get data for this sample
        rd_map = test_dataset.range_doppler_maps[idx] #(2, 12, 64)
        target_mask = test_dataset.target_masks[idx, :, :, 0]
        
        # Calculate magnitude
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2) #(12, 64)
        
        # Plot range-Doppler heatmap
        plt.imshow(20*np.log10(rd_magnitude + 1e-10), aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude (dB)')
        
        # Mark targets with red circles
        for target in test_dataset.target_info[idx]:
            plt.plot(target['range_bin'], target['doppler_bin'], 'ro', markersize=10, 
                     markerfacecolor='none', markeredgewidth=2)
            # Add text label with distance and velocity
            plt.annotate(f"{target['distance']:.1f}m, {target['velocity']:.1f}m/s", 
                         (target['range_bin'], target['doppler_bin']),
                         xytext=(10, 10), textcoords='offset points',
                         color='white', fontsize=8, 
                         bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7))
        
        plt.title(f'Range-Doppler Map with Targets (Sample {idx})')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'data/radar/visualizations/rd_map_with_targets_{idx}{IMG_FORMAT}')
        plt.close()
    
    # 2. Time domain IQ signals for all chirps (first sample, first RX)
    plt.figure(figsize=(15, 10))
    sample_idx = 0
    rx_idx = 0
    
    # Get time domain data for first sample, first RX
    time_data = test_dataset.time_domain_data[sample_idx, rx_idx] #(12, 20, 2)
    
    # Create time axis in microseconds
    t = np.arange(test_dataset.samples_per_chirp) / test_dataset.sample_rate * 1e6
    
    # Plot I/Q data for each chirp
    num_chirps_to_plot = min(6, test_dataset.num_chirps)  # Plot up to 6 chirps
    for chirp_idx in range(num_chirps_to_plot):
        plt.subplot(num_chirps_to_plot, 2, 2*chirp_idx+1)
        plt.plot(t, time_data[chirp_idx, :, 0], 'b-')
        plt.title(f'I Component - Chirp {chirp_idx}')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.subplot(num_chirps_to_plot, 2, 2*chirp_idx+2)
        plt.plot(t, time_data[chirp_idx, :, 1], 'r-')
        plt.title(f'Q Component - Chirp {chirp_idx}')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'data/radar/visualizations/time_domain_iq_signals{IMG_FORMAT}')
    plt.close()
    
    # 3. Range profiles for multiple chirps
    plt.figure(figsize=(12, 8))
    
    # Get complex data for first sample
    complex_data = test_dataset.time_domain_data[0, 0, :, :, 0] + 1j * test_dataset.time_domain_data[0, 0, :, :, 1]
    
    # Compute range profiles for each chirp
    range_profiles = np.abs(np.fft.fft(complex_data, n=test_dataset.num_range_bins, axis=1))
    
    # Plot range profiles for multiple chirps
    num_chirps_to_plot = min(6, test_dataset.num_chirps)
    for chirp_idx in range(num_chirps_to_plot):
        plt.plot(range_profiles[chirp_idx, :], label=f'Chirp {chirp_idx}')
    
    plt.title('Range Profiles for Multiple Chirps')
    plt.xlabel('Range Bin')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'data/radar/visualizations/range_profiles{IMG_FORMAT}')
    plt.close()
    
    # 4. 3D visualization of Range-Doppler map
    from mpl_toolkits.mplot3d import Axes3D
    
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    
    # Get data for first sample
    rd_map = test_dataset.range_doppler_maps[0]
    rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(np.arange(test_dataset.num_range_bins), np.arange(test_dataset.num_doppler_bins))
    
    # Plot surface
    surf = ax.plot_surface(X, Y, 20*np.log10(rd_magnitude + 1e-10), cmap='plasma', 
                          linewidth=0, antialiased=True)
    
    # Mark targets with red points
    for target in test_dataset.target_info[0]:
        target_magnitude = 20*np.log10(rd_magnitude[target['doppler_bin'], target['range_bin']] + 1e-10)
        ax.scatter([target['range_bin']], [target['doppler_bin']], [target_magnitude], 
                  color='red', s=100, marker='o')
    
    # Set labels and title
    ax.set_xlabel('Range Bin')
    ax.set_ylabel('Doppler Bin')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title('3D Range-Doppler Map')
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Magnitude (dB)')
    
    plt.tight_layout()
    plt.savefig(f'data/radar/visualizations/3d_range_doppler{IMG_FORMAT}')
    plt.close()
    
    # 5. Comparison of multiple RX antennas (if available)
    if test_dataset.num_rx > 1:
        plt.figure(figsize=(15, 8))
        
        # Get time domain data for first sample, first chirp
        chirp_idx = 0
        
        # Plot I component for each RX antenna
        for rx_idx in range(test_dataset.num_rx):
            plt.subplot(test_dataset.num_rx, 2, 2*rx_idx+1)
            plt.plot(t, test_dataset.time_domain_data[0, rx_idx, chirp_idx, :, 0])
            plt.title(f'I Component - RX Antenna {rx_idx}')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.subplot(test_dataset.num_rx, 2, 2*rx_idx+2)
            plt.plot(t, test_dataset.time_domain_data[0, rx_idx, chirp_idx, :, 1])
            plt.title(f'Q Component - RX Antenna {rx_idx}')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'data/radar/visualizations/multi_rx_comparison{IMG_FORMAT}')
        plt.close()
    
    print(f"Additional visualizations saved to data/radar/visualizations/")
    
    print("\nTest completed successfully!")
    return test_dataset

# if __name__ == '__main__':
    
#     # Run the test function
#     test_dataset = test_radar_dataset()
    
#     # Test loading the saved data
#     print("\nTesting data loading functionality:")
#     loaded_dataset = RadarDataset(
#         datapath="/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar/radar_simulation_data.npy",
#         drawfig=True
#     )
    
#     print(f"Loaded dataset size: {len(loaded_dataset)}")
#     print(f"Loaded Range-Doppler maps shape: {loaded_dataset.range_doppler_maps.shape}")
    
#     # Compare a sample from both datasets
#     original_sample = test_dataset[0]['feature_2d']
#     loaded_sample = loaded_dataset[0]['feature_2d']
    
#     print(f"Original sample shape: {original_sample.shape}")
#     print(f"Loaded sample shape: {loaded_sample.shape}")
    
#     print("Data validation complete!")

def test_realistic_effects():
    """Test the realistic RF impairments functionality"""
    print("Testing realistic RF impairments...")
    
    # Create a small test dataset with realistic effects enabled
    test_dataset = RadarDataset(
        num_samples=5,               # Small number for testing
        num_range_bins=64,           
        num_doppler_bins=12,         
        snr_min=10,                  
        snr_max=25,                  
        max_targets=3,               
        training=False,              
        drawfig=True,                
        save_data=False,             
        sample_rate=3e6,             
        chirp_duration=500e-6,       
        num_chirps=12,               
        bandwidth=500e6,             
        center_freq=2.1e9,           
        num_rx=4,                    
        num_tx=1,                    
        signal_type='FMCW',          
        apply_realistic_effects=True,  # Enable realistic effects
        recalculate_rd_map=True       # Recalculate RD map after applying effects
    )
    
    # Visualize the effects on each sample
    for i in range(len(test_dataset)):
        test_dataset.visualize_realistic_effects(i)
    
    print("Realistic effects test completed!")
    return test_dataset

    

if __name__ == '__main__':
    test_realistic_effects()

    # Run the test function with different signal types
    signal_types = ['FMCW', 'OFDM', 'Sine']
    test_datasets = {}
    
    print("\n=== COMPARING DIFFERENT RADAR SIGNAL TYPES ===")
    
    # Create a dataset for each signal type
    for signal_type in signal_types:
        print(f"\n--- Testing RadarDataset with {signal_type} signal type ---")
        test_datasets[signal_type] = RadarDataset(
            num_samples=5,               # Reduced number for comparison
            num_range_bins=64,           
            num_doppler_bins=12,         
            snr_min=10,                  
            snr_max=25,                  
            max_targets=3,               
            training=False,              
            drawfig=True,                
            save_data=False,             # Don't save individual datasets
            sample_rate=3e6,             
            chirp_duration=500e-6,       
            num_chirps=12,               # Match Doppler bins
            bandwidth=500e6,             
            center_freq=2.1e9,           
            num_rx=4,                    
            num_tx=1,                    
            signal_type=signal_type      # Use the current signal type
        )
    
    # Create comparison visualizations
    os.makedirs('data/radar/signal_comparison', exist_ok=True)
    
    # 1. Compare time domain signals (first chirp)
    plt.figure(figsize=(15, 10))
    
    # Time axis in microseconds
    t = np.linspace(0, test_datasets['FMCW'].chirp_duration * 1e6, 
                   test_datasets['FMCW'].samples_per_chirp)
    
    # Plot I/Q components for each signal type
    for i, signal_type in enumerate(signal_types):
        # Get first sample, first RX, first chirp
        time_data = test_datasets[signal_type].time_domain_data[0, 0, 0]
        
        # I component
        plt.subplot(3, 2, 2*i+1)
        plt.plot(t, time_data[:, 0], 'b-')
        plt.title(f'{signal_type} - I Component')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Q component
        plt.subplot(3, 2, 2*i+2)
        plt.plot(t, time_data[:, 1], 'r-')
        plt.title(f'{signal_type} - Q Component')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'data/radar/signal_comparison/time_domain_comparison{IMG_FORMAT}')
    plt.close()
    
    # 2. Compare frequency domain signals (first chirp)
    plt.figure(figsize=(15, 10))
    
    for i, signal_type in enumerate(signal_types):
        # Get first sample, first RX, first chirp
        time_data = test_datasets[signal_type].time_domain_data[0, 0, 0]
        complex_data = time_data[:, 0] + 1j * time_data[:, 1]
        
        # Compute FFT
        freq_data = np.fft.fftshift(np.fft.fft(complex_data))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(len(complex_data), 
                                                  1/test_datasets[signal_type].sample_rate)) / 1e6  # MHz
        
        # Plot magnitude
        plt.subplot(3, 1, i+1)
        plt.plot(freq_axis, 20*np.log10(np.abs(freq_data) + 1e-10))
        plt.title(f'{signal_type} - Frequency Domain')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([-test_datasets[signal_type].sample_rate/2e6, test_datasets[signal_type].sample_rate/2e6])
    
    plt.tight_layout()
    plt.savefig(f'data/radar/signal_comparison/frequency_domain_comparison{IMG_FORMAT}')
    plt.close()
    
    # 3. Compare range profiles
    plt.figure(figsize=(15, 10))
    
    for i, signal_type in enumerate(signal_types):
        # Get first sample, first RX, first chirp
        time_data = test_datasets[signal_type].time_domain_data[0, 0, 0]
        complex_data = time_data[:, 0] + 1j * time_data[:, 1]
        
        # Compute range profile
        range_profile = np.abs(np.fft.fft(complex_data, n=test_datasets[signal_type].num_range_bins))
        range_axis = np.arange(test_datasets[signal_type].num_range_bins) * test_datasets[signal_type].range_resolution
        
        # Plot range profile
        plt.subplot(3, 1, i+1)
        plt.plot(range_axis, 20*np.log10(range_profile + 1e-10))
        plt.title(f'{signal_type} - Range Profile')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([0, test_datasets[signal_type].max_range])
    
    plt.tight_layout()
    plt.savefig(f'data/radar/signal_comparison/range_profile_comparison{IMG_FORMAT}')
    plt.close()
    
    # 4. Compare range-Doppler maps
    plt.figure(figsize=(15, 15))
    
    for i, signal_type in enumerate(signal_types):
        # Get first sample
        rd_map = test_datasets[signal_type].range_doppler_maps[0]
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        # Plot range-Doppler map
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(20*np.log10(rd_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(f'{signal_type} - Range-Doppler Map')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Plot target mask
        plt.subplot(3, 2, 2*i+2)
        plt.imshow(test_datasets[signal_type].target_masks[0, :, :, 0], aspect='auto', cmap='gray')
        plt.colorbar(label='Target Presence')
        plt.title(f'{signal_type} - Target Mask')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
    
    plt.tight_layout()
    plt.savefig(f'data/radar/signal_comparison/range_doppler_comparison{IMG_FORMAT}')
    plt.close()
    
    # 5. Compare target detection performance
    plt.figure(figsize=(15, 10))
    
    # For each signal type, compute detection metrics on the same targets
    detection_results = {}
    
    for signal_type in signal_types:
        # Get first sample
        rd_map = test_datasets[signal_type].range_doppler_maps[0]
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        target_mask = test_datasets[signal_type].target_masks[0, :, :, 0]
        
        # Simple detection threshold (could be more sophisticated)
        threshold = np.mean(rd_magnitude) + 2 * np.std(rd_magnitude)
        detection_mask = (rd_magnitude > threshold).astype(float)
        
        # Calculate metrics
        true_positives = np.sum((detection_mask == 1) & (target_mask == 1))
        false_positives = np.sum((detection_mask == 1) & (target_mask == 0))
        false_negatives = np.sum((detection_mask == 0) & (target_mask == 1))
        true_negatives = np.sum((detection_mask == 0) & (target_mask == 0))
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        detection_results[signal_type] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    # Plot metrics comparison
    metrics = ['precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, signal_type in enumerate(signal_types):
        values = [detection_results[signal_type][metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=signal_type)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Target Detection Performance Comparison')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'data/radar/signal_comparison/detection_performance_comparison{IMG_FORMAT}')
    plt.close()
    
    # 6. Create a comprehensive summary visualization
    plt.figure(figsize=(15, 12))
    
    # Plot time domain signals (first row)
    for i, signal_type in enumerate(signal_types):
        plt.subplot(4, 3, i+1)
        time_data = test_datasets[signal_type].time_domain_data[0, 0, 0]
        plt.plot(t, time_data[:, 0], 'b-', label='I')
        plt.plot(t, time_data[:, 1], 'r-', label='Q')
        plt.title(f'{signal_type} - Time Domain')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
    
    # Plot frequency domain (second row)
    for i, signal_type in enumerate(signal_types):
        plt.subplot(4, 3, i+4)
        time_data = test_datasets[signal_type].time_domain_data[0, 0, 0]
        complex_data = time_data[:, 0] + 1j * time_data[:, 1]
        freq_data = np.fft.fftshift(np.fft.fft(complex_data))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(len(complex_data), 
                                                  1/test_datasets[signal_type].sample_rate)) / 1e6
        plt.plot(freq_axis, 20*np.log10(np.abs(freq_data) + 1e-10))
        plt.title(f'{signal_type} - Frequency Domain')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
    
    # Plot range profiles (third row)
    for i, signal_type in enumerate(signal_types):
        plt.subplot(4, 3, i+7)
        time_data = test_datasets[signal_type].time_domain_data[0, 0, 0]
        complex_data = time_data[:, 0] + 1j * time_data[:, 1]
        range_profile = np.abs(np.fft.fft(complex_data, n=test_datasets[signal_type].num_range_bins))
        range_axis = np.arange(test_datasets[signal_type].num_range_bins) * test_datasets[signal_type].range_resolution
        plt.plot(range_axis, 20*np.log10(range_profile + 1e-10))
        plt.title(f'{signal_type} - Range Profile')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
    
    # Plot range-Doppler maps (fourth row)
    for i, signal_type in enumerate(signal_types):
        plt.subplot(4, 3, i+10)
        rd_map = test_datasets[signal_type].range_doppler_maps[0]
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        plt.imshow(20*np.log10(rd_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='dB')
        plt.title(f'{signal_type} - Range-Doppler')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
    
    plt.tight_layout()
    plt.savefig(f'data/radar/signal_comparison/comprehensive_comparison{IMG_FORMAT}')
    plt.close()
    
    print("\n=== Signal Type Comparison Complete ===")
    print(f"Comparison visualizations saved to data/radar/signal_comparison/")
    
    # Save the FMCW dataset for further testing
    print("\nSaving FMCW dataset for further testing...")
    test_datasets['FMCW'].save_data = True
    test_datasets['FMCW']._save_data()
    
    # Test loading the saved data
    print("\nTesting data loading functionality:")
    loaded_dataset = RadarDataset(
        datapath="/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar/radar_simulation_data.npy",
        drawfig=True
    )
    
    print(f"Loaded dataset size: {len(loaded_dataset)}")
    print(f"Loaded Range-Doppler maps shape: {loaded_dataset.range_doppler_maps.shape}")
    
    # Compare a sample from both datasets
    original_sample = test_datasets['FMCW'][0]['feature_2d']
    loaded_sample = loaded_dataset[0]['feature_2d']
    
    print(f"Original sample shape: {original_sample.shape}")
    print(f"Loaded sample shape: {loaded_sample.shape}")
    
    print("Data validation complete!")