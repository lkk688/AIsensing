import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import os
import random
from scipy.signal import chirp
from tqdm import tqdm
from AIradar_processing import RadarProcessing
IMG_FORMAT=".pdf" #".png"


class RadarDataset(Dataset):
    # In the RadarDataset class initialization, update the default parameters
    def __init__(self, 
                 num_samples=100,
                 num_range_bins=64, 
                 num_doppler_bins=12,
                 sample_rate=60e6,  # 61.44 MSPS is AD9361's maximum sampling rate
                 transceiver_bandwidth=56e6,  # AD9361 bandwidth limitation (~56MHz)
                 chirp_duration=500e-6,  # 500 μs (from radarconfig.yaml ramp_time: 500 us)
                 num_chirps=12,  # 12 chirps to match Doppler bins
                 bandwidth=500e6,  # 500 MHz (from radarconfig.yaml default_chirp_bw)
                 transceiver_center_freq=2.1e9,  # 2.1 GHz (from radarconfig.yaml)
                 center_freq=2.1e9,  # 2.1 GHz (from radarconfig.yaml)
                 num_rx=4,  # 4 RX antennas (typical for Phaser)
                 num_tx=1,  # 1 TX antenna
                 max_targets=3,  # Maximum 3 targets per sample
                 snr_min=10,  # Minimum SNR in dB
                 snr_max=25,  # Maximum SNR in dB
                 signal_type='OFDM',  # OFDM signal type (from radarconfig.yaml)
                 signal_freq=100e3,  # 100 kHz (from radarconfig.yaml)
                 apply_realistic_effects=True,
                 save_path='data/radarv3',
                 savedataformat='hdf5',
                 precision='float32',
                 drawfig=False,
                 datapath=None,
                 use_lazy_loading=False,
                 use_memory_mapping=False,
                 cache_size=100):
        """
        Initialize radar dataset
        
        Args:
            num_samples: Number of samples to generate
            num_range_bins: Number of range bins
            num_doppler_bins: Number of Doppler bins
            sample_rate: Sample rate in Hz
            chirp_duration: Chirp duration in seconds
            num_chirps: Number of chirps
            bandwidth: Bandwidth in Hz
            center_freq: Center frequency in Hz
            num_rx: Number of RX antennas
            num_tx: Number of TX antennas
            max_targets: Maximum number of targets per sample
            snr_min: Minimum SNR in dB
            snr_max: Maximum SNR in dB
            signal_type: Type of radar signal ('FMCW', 'OFDM', 'Sine', etc.)
            signal_freq: Signal frequency for modulation (Hz)
            apply_realistic_effects: Whether to apply realistic RF effects
            save_path: Path to save generated data
            savedataformat: Format to save data ('hdf5' or 'numpy')
            precision: Precision of saved data ('float32' or 'float16')
            drawfig: Whether to draw figures
            datapath: Path to existing dataset (if loading)
            use_lazy_loading: Whether to use lazy loading for HDF5 files
            use_memory_mapping: Whether to use memory mapping for NumPy files
            cache_size: Size of cache for lazy loading
        """
        # Store parameters
        #self.training = training
        self.drawfig = drawfig
        self.num_samples = num_samples
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.max_targets = max_targets
        self.save_path = save_path

        self.transceiver_bandwidth = transceiver_bandwidth  # AD9361 bandwidth limitation (~56MHz)
        self.num_subcarriers = 64
        
        # Set precision for data (MPS framework doesn't support float64)
        self.precision = precision
        if precision not in ['float32', 'float16']:
            print(f"Warning: Unsupported precision '{precision}'. Using 'float32' instead.")
            self.precision = 'float32'
        
        # Store SDR parameters
        self.sample_rate = sample_rate
        self.chirp_duration = chirp_duration
        self.num_chirps = num_chirps
        self.bandwidth = bandwidth
        self.center_freq = center_freq
        self.transceiver_center_freq = transceiver_center_freq
        # Validate signal type
        valid_signal_types = ['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW']
        if signal_type not in valid_signal_types:
            print(f"Warning: Invalid signal type '{signal_type}'. Using 'FMCW' instead.")
            self.signal_type = 'FMCW'
        else:
            self.signal_type = signal_type
        self.signal_freq = signal_freq #used for Sine_FMCW
        self.num_rx = num_rx
        self.num_tx = num_tx
        
        # Store realistic effects parameters
        self.apply_realistic_effects = apply_realistic_effects
        #self.recalculate_rd_map = recalculate_rd_map
        
        # Initialize the radar processor
        self.radar_processor = RadarProcessing(
            num_range_bins=self.num_range_bins,
            num_doppler_bins=self.num_doppler_bins,
            sample_rate=self.sample_rate,
            chirp_duration=self.chirp_duration,
            num_chirps=self.num_chirps,
            bandwidth=self.bandwidth,
            center_freq=self.center_freq,
            signal_type=self.signal_type,
            signal_freq=self.signal_freq
        )
        
        # Calculate derived parameters
        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        self.range_resolution = self.radar_processor.range_resolution
        self.max_range = self.radar_processor.max_range
        self.velocity_resolution = self.radar_processor.velocity_resolution
        self.max_velocity = self.radar_processor.max_velocity
        self.wavelength = 3e8 / self.center_freq  # Wavelength in meters
        self.speed_of_light = 3e8  # Speed of light in m/s
        
        # Initialize lazy loading parameters
        self.use_lazy_loading = use_lazy_loading
        self.use_memory_mapping = use_memory_mapping
        self.h5_file = None
        self.data_cache = {} if use_lazy_loading else None
        self.cache_size = cache_size
        
        # Initialize data containers
        self.time_domain_data = None
        self.range_doppler_maps = None
        self.target_masks = None
        self.target_info = None

        if datapath is not None:
            self._load_data(datapath)
        else:
            print("Generating new radar data")
            self.generate_radar_data(save_data=True, format=savedataformat)

    def generate_radar_data(self, save_data=True, format='hdf5'):
        """Generate synthetic radar data
        
        Args:
            save_data: Whether to save the generated data
            format: Format to save data ('hdf5' or 'numpy')
        """
        # Create directory if it doesn't exist
        if save_data:
            os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize arrays to store data
        self.time_domain_data = np.zeros((self.num_samples, self.num_rx, self.num_chirps, 
                                         self.samples_per_chirp, 2), dtype=np.float32)
        self.range_doppler_maps = np.zeros((self.num_samples, 2, self.num_doppler_bins, 
                                           self.num_range_bins), dtype=np.float32)
        self.target_masks = np.zeros((self.num_samples, self.num_doppler_bins, 
                                     self.num_range_bins, 1), dtype=np.float32)
        self.target_info = []
        
        # Generate data for each sample
        for i in tqdm(range(self.num_samples), desc="Generating radar data"):
            # Initialize complex received signal
            rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), dtype=np.complex64) #(4, 12, 150) complex
            
            # Determine number of targets for this sample
            num_targets = random.randint(1, self.max_targets)
            
            # Store target information for this sample
            sample_targets = []
            
            # Calculate practical max range (use the minimum of theoretical and a reasonable limit)
            practical_max_range = min(self.max_range, 150.0)  # Limit to 150m for automotive radar
            
            # Add targets
            for _ in range(num_targets):
                # Random target parameters with more realistic ranges
                distance = random.uniform(1.0, 0.9 * practical_max_range)  # meters
                velocity = random.uniform(-0.9 * self.max_velocity, 0.9 * self.max_velocity)  # m/s
                rcs = random.uniform(0.1, 1.0)  # Normalized RCS
                
                # Add target to received signal
                #_add_target calls _generate_chirp
                rx_signal = self._add_target(rx_signal, distance, velocity, rcs)
                #(4, 12, 150) complex
                # Store target information
                sample_targets.append({
                    'distance': distance,
                    'velocity': velocity,
                    'rcs': rcs
                })
            
            # Add noise
            snr_db = random.uniform(self.snr_min, self.snr_max)
            rx_signal = self._add_noise(rx_signal, snr_db) #(4, 12, 150)
            
            # Apply realistic RF effects if enabled
            if self.apply_realistic_effects:
                # Convert to I/Q format
                iq_data = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp, 2), dtype=np.float32)
                iq_data[..., 0] = np.real(rx_signal)
                iq_data[..., 1] = np.imag(rx_signal)
                
                # Apply realistic effects
                iq_data = self.radar_processor.apply_realistic_rf_effects(iq_data, sample_targets)
                
                # Convert back to complex for processing
                rx_signal = iq_data[..., 0] + 1j * iq_data[..., 1]
            
            # Store time domain data
            self.time_domain_data[i, :, :, :, 0] = np.real(rx_signal) #(4, 12, 150) complex to 
            self.time_domain_data[i, :, :, :, 1] = np.imag(rx_signal)
            #(100, 4, 12, 150, 2)
            # Process to range-Doppler map using RadarProcessing
            rd_map = self.radar_processor.time_to_range_doppler(rx_signal) #(4, 12, 150) to (12, 64)
            
            # Store range-Doppler map
            self.range_doppler_maps[i, 0, :, :] = np.real(rd_map) #(100, 2, 12, 64)
            self.range_doppler_maps[i, 1, :, :] = np.imag(rd_map)
            
            # Create target mask
            self.target_masks[i] = self._create_target_mask(sample_targets) #(100, 12, 64, 1)
            
            # Store target information
            self.target_info.append(sample_targets)
            
            # Draw figure if enabled
            if self.drawfig and i < 5:  # Only draw first 5 samples
                self._draw_sample(i, rx_signal, rd_map, sample_targets)
        
        # Save data if requested
        if save_data:
            if format.lower() == 'hdf5':
                self._save_hdf5()
            elif format.lower() == 'numpy':
                self._save_numpy()
            else:
                print(f"Warning: Unsupported format '{format}'. Using 'hdf5' instead.")
                self._save_hdf5()

    def _generate_chirp(self):
        """Generate a single chirp based on the signal type
        
        Returns:
            Complex chirp signal with shape [samples_per_chirp]
        """
        # Create time vector
        t = np.arange(self.samples_per_chirp) / self.sample_rate
        
        if self.signal_type == 'FMCW':
            # Linear frequency modulation (FMCW)
            # Frequency increases linearly from f0 to f0 + bandwidth
            # Phase is the integral of frequency: phi(t) = 2*pi*(f0*t + 0.5*k*t^2)
            # where k = bandwidth/chirp_duration
            k = self.bandwidth / self.chirp_duration  # Chirp rate
            phase = 2 * np.pi * (self.center_freq * t + 0.5 * k * t**2)
            chirp = np.exp(1j * phase)
            
        elif self.signal_type == 'OFDM':
            # Orthogonal Frequency Division Multiplexing
            # Use multiple subcarriers with random phases
            subcarrier_spacing = self.bandwidth / self.num_subcarriers
            
            # Initialize chirp
            chirp = np.zeros(self.samples_per_chirp, dtype=np.complex64)
            
            # Generate subcarriers
            for i in range(self.num_subcarriers):
                # Random phase for each subcarrier
                phase_offset = np.random.uniform(0, 2*np.pi)
                
                # Subcarrier frequency
                f_sc = self.center_freq - self.bandwidth/2 + i * subcarrier_spacing
                
                # Generate subcarrier signal
                subcarrier = np.exp(1j * (2 * np.pi * f_sc * t + phase_offset))
                
                # Add to total signal
                chirp += subcarrier
            
            # Normalize
            chirp /= np.sqrt(self.num_subcarriers)
            
        elif self.signal_type == 'Sine':
            # Simple sine wave at center frequency
            phase = 2 * np.pi * self.center_freq * t
            chirp = np.exp(1j * phase)
            
        elif self.signal_type == 'OFDM_FMCW':
            # Two-step process to simulate AD9361 + CN0566 hardware setup:
            # 1. AD9361 generates OFDM signal (limited to ~56MHz) centered at 2.1GHz
            # 2. CN0566 performs frequency sweep to achieve 500MHz bandwidth at ~10GHz
            
            # Step 1: Generate OFDM signal with AD9361 limitations
            # Initialize OFDM component (centered at 2.1GHz)
            ofdm_signal = np.zeros(self.samples_per_chirp, dtype=np.complex64)
            subcarrier_spacing = self.transceiver_bandwidth / self.num_subcarriers

            # Generate subcarriers within AD9361 bandwidth
            for i in range(self.num_subcarriers):
                phase_offset = np.random.uniform(0, 2*np.pi)
                # Subcarrier frequency centered around 2.1GHz (self.transceiver_center_freq)
                f_sc = self.transceiver_center_freq - self.transceiver_bandwidth/2 + i * subcarrier_spacing
                subcarrier = np.exp(1j * (2 * np.pi * f_sc * t + phase_offset))
                ofdm_signal += subcarrier
            
            # Normalize OFDM component
            ofdm_signal /= np.sqrt(self.num_subcarriers)
            
            # Step 2: Apply CN0566 frequency sweep to achieve full bandwidth
            # CN0566 performs frequency sweep centered around 10GHz
            cn0566_center_freq = self.center_freq # 10e9  # 10GHz center frequency for CN0566
            
            # Generate FMCW sweep component (CN0566)
            k = self.bandwidth / self.chirp_duration  # Chirp rate for 500MHz bandwidth
            
            # Phase calculation for the frequency sweep
            # Starting at cn0566_center_freq - bandwidth/2 and sweeping to cn0566_center_freq + bandwidth/2
            phase = 2 * np.pi * (cn0566_center_freq * t + 0.5 * k * t**2)
            fmcw_sweep = np.exp(1j * phase)
            
            # Combine the OFDM signal with the frequency sweep
            # This simulates the OFDM signal being upconverted and swept by CN0566
            chirp = ofdm_signal * fmcw_sweep
            
        elif self.signal_type == 'Sine_FMCW':
            # Two-step process to simulate AD9361 + CN0566 hardware setup:
            # 1. AD9361 generates Sine signal centered at 2.1GHz
            # 2. CN0566 performs frequency sweep to achieve 500MHz bandwidth at ~10GHz
            
            # Step 1: Generate Sine signal with AD9361 (centered at 2.1GHz)
            # Simple sine wave at center frequency (AD9361 output)
            
            #sine_phase = 2 * np.pi * self.transceiver_center_freq * t
            sine_phase = 2 * np.pi * self.signal_freq * t #1KHz
            sine_signal = np.exp(1j * sine_phase)
            
            # Step 2: Apply CN0566 frequency sweep to achieve full bandwidth
            # CN0566 performs frequency sweep centered around 10GHz
            cn0566_center_freq = self.center_freq #10e9  # 10GHz center frequency for CN0566
            
            # Generate FMCW sweep component (CN0566)
            k = self.bandwidth / self.chirp_duration  # Chirp rate for 500MHz bandwidth
            
            # Phase calculation for the frequency sweep
            # Starting at cn0566_center_freq - bandwidth/2 and sweeping to cn0566_center_freq + bandwidth/2
            fmcw_phase = 2 * np.pi * (cn0566_center_freq * t + 0.5 * k * t**2)
            fmcw_sweep = np.exp(1j * fmcw_phase)
            
            # Combine the Sine signal with the frequency sweep
            # This simulates the Sine signal being upconverted and swept by CN0566
            chirp = sine_signal * fmcw_sweep
            
        else:
            # Default to FMCW if unknown signal type
            print(f"Warning: Unknown signal type '{self.signal_type}'. Using FMCW instead.")
            k = self.bandwidth / self.chirp_duration
            phase = 2 * np.pi * (self.center_freq * t + 0.5 * k * t**2)
            chirp = np.exp(1j * phase)
        
        return chirp

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
        tx_chirp = self._generate_chirp() #(150,)
        
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
                doppler_phase = doppler_phase_per_chirp * chirp
                
                # Create delayed and phase-shifted version of the transmit chirp
                if delay_samples < len(tx_chirp):
                    # Delayed signal (with zero-padding)
                    delayed_signal = np.zeros_like(tx_chirp)
                    delayed_signal[delay_samples:] = tx_chirp[:len(tx_chirp)-delay_samples]
                    
                    # Apply Doppler phase shift
                    delayed_signal = delayed_signal * np.exp(1j * doppler_phase)
                    
                    # Scale by amplitude and add to received signal
                    rx_signal[rx, chirp] += signal_amplitude * delayed_signal
        
        return rx_signal

    def _add_noise(self, rx_signal, snr_db):
        """Add complex Gaussian noise to the received signal
        
        Args:
            rx_signal: Received signal array
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy signal
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(rx_signal)**2)
        
        # Calculate noise power based on SNR
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), rx_signal.shape)
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), rx_signal.shape)
        noise = noise_real + 1j * noise_imag
        
        # Add noise to signal
        return rx_signal + noise

    def _create_target_mask(self, target_info):
        """Create a binary mask indicating target locations in the range-Doppler map
        
        Args:
            target_info: List of dictionaries with target information
            
        Returns:
            Binary mask with shape [num_doppler_bins, num_range_bins, 1]
        """
        # Initialize mask
        mask = np.zeros((self.num_doppler_bins, self.num_range_bins, 1), dtype=np.float32)
        
        # For each target, mark its location in the mask
        for target in target_info:
            # Convert distance to range bin
            range_bin = int(target['distance'] / self.range_resolution)
            
            # Convert velocity to Doppler bin (centered)
            doppler_bin = int(self.num_doppler_bins/2 + target['velocity'] / self.velocity_resolution)
            
            # Check if within bounds
            if (0 <= range_bin < self.num_range_bins and 
                0 <= doppler_bin < self.num_doppler_bins):
                
                # Mark target location with a larger region to make it more visible
                mask_width = max(2, int(0.1 * self.num_range_bins))
                mask_height = max(2, int(0.1 * self.num_doppler_bins))
                
                for i in range(max(0, doppler_bin - mask_height), 
                               min(self.num_doppler_bins, doppler_bin + mask_height + 1)):
                    for j in range(max(0, range_bin - mask_width), 
                                   min(self.num_range_bins, range_bin + mask_width + 1)):
                        # Use a Gaussian-like falloff for more natural looking targets
                        dist_sq = ((i - doppler_bin) / mask_height)**2 + ((j - range_bin) / mask_width)**2
                        mask[i, j, 0] = max(mask[i, j, 0], np.exp(-dist_sq))
        
        return mask

    
    def _draw_sample(self, index, rx_signal, rd_map, targets):
        """Draw visualization of a sample
        
        Args:
            index: Sample index
            rx_signal: Complex received signal
            rd_map: Range-Doppler map
            targets: Target information
        """
        # Create directory for figures
        os.makedirs(f"{self.save_path}/figures", exist_ok=True)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot time domain signal (real part)
        axs[0, 0].plot(np.real(rx_signal[0, 0]))
        axs[0, 0].set_title('Time Domain Signal (Real Part)')
        axs[0, 0].set_xlabel('Sample')
        axs[0, 0].set_ylabel('Amplitude')
        
        # Plot time domain signal (imaginary part)
        axs[0, 1].plot(np.imag(rx_signal[0, 0]))
        axs[0, 1].set_title('Time Domain Signal (Imaginary Part)')
        axs[0, 1].set_xlabel('Sample')
        axs[0, 1].set_ylabel('Amplitude')
        
        # Plot range-Doppler map (magnitude)
        rd_magnitude = np.sqrt(np.real(rd_map)**2 + np.imag(rd_map)**2)
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)  # Convert to dB
        im = axs[1, 0].imshow(rd_db, aspect='auto', cmap='jet')
        axs[1, 0].set_title('Range-Doppler Map (Magnitude, dB)')
        axs[1, 0].set_xlabel('Range Bin')
        axs[1, 0].set_ylabel('Doppler Bin')
        plt.colorbar(im, ax=axs[1, 0])
        
        # Plot target mask
        mask = self.target_masks[index, :, :, 0]
        axs[1, 1].imshow(mask, aspect='auto', cmap='hot')  # Changed to 'hot' colormap for better visibility
        axs[1, 1].set_title('Target Mask')
        axs[1, 1].set_xlabel('Range Bin')
        axs[1, 1].set_ylabel('Doppler Bin')
        
        # Add target annotations with clear markers
        for target in targets:
            range_bin = int(target['distance'] / self.range_resolution)
            doppler_bin = int(self.num_doppler_bins/2 + target['velocity'] / self.velocity_resolution)
            
            if (0 <= range_bin < self.num_range_bins and 
                0 <= doppler_bin < self.num_doppler_bins):
                # Add markers to both plots
                axs[1, 0].plot(range_bin, doppler_bin, 'wo', markersize=8, markeredgecolor='black')
                axs[1, 1].plot(range_bin, doppler_bin, 'wo', markersize=8, markeredgecolor='black')
                
                # Add target information text
                axs[1, 0].text(range_bin + 2, doppler_bin, 
                          f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                          color='white', fontsize=8, backgroundcolor='black')
        
        # Add sample information
        plt.suptitle(f'Sample {index}: {len(targets)} Targets, Signal Type: {self.signal_type}')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/figures/sample_{index}{IMG_FORMAT}")
        plt.close()

    def _save_hdf5(self):
        """Save dataset to HDF5 format"""
        file_path = f"{self.save_path}/radar_data.h5"
        print(f"Saving dataset to {file_path}")
        # Close any existing open file handle to this file
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
                print("Closed previously open HDF5 file")
            except Exception as e:
                print(f"Warning when closing existing file: {e}")
        
        # Make sure the file doesn't exist before creating it
        if os.path.exists(file_path):
            # Try alternative approach - create a new filename
            file_path = f"{self.save_path}/radar_data_{int(time.time())}.h5"
            print(f"File already exist, using alternative filename: {file_path}")

        try:
            with h5py.File(file_path, 'w') as f:
                # Create datasets
                f.create_dataset('time_domain_data', data=self.time_domain_data, 
                                dtype=self.precision, compression='gzip')
                f.create_dataset('range_doppler_maps', data=self.range_doppler_maps, 
                                dtype=self.precision, compression='gzip')
                f.create_dataset('target_masks', data=self.target_masks, 
                                dtype=self.precision, compression='gzip')
                
                # Store target information as attributes
                for i, targets in enumerate(self.target_info):
                    target_group = f.create_group(f'target_info/{i}')
                    for j, target in enumerate(targets):
                        target_subgroup = target_group.create_group(f'{j}')
                        for key, value in target.items():
                            target_subgroup.attrs[key] = value
                
                # Store dataset parameters
                params = f.create_group('parameters')
                params.attrs['num_samples'] = self.num_samples
                params.attrs['num_range_bins'] = self.num_range_bins
                params.attrs['num_doppler_bins'] = self.num_doppler_bins
                params.attrs['sample_rate'] = self.sample_rate
                params.attrs['chirp_duration'] = self.chirp_duration
                params.attrs['num_chirps'] = self.num_chirps
                params.attrs['bandwidth'] = self.bandwidth
                params.attrs['center_freq'] = self.center_freq
                params.attrs['num_rx'] = self.num_rx
                params.attrs['num_tx'] = self.num_tx
                params.attrs['signal_type'] = self.signal_type
                params.attrs['range_resolution'] = self.range_resolution
                params.attrs['velocity_resolution'] = self.velocity_resolution
                params.attrs['max_range'] = self.max_range
                params.attrs['max_velocity'] = self.max_velocity
            
            print(f"Successfully saved dataset to {file_path}")
        except Exception as e:
            print(f"Error saving HDF5 file: {e}")

    def _save_numpy(self):
        """Save dataset to NumPy format"""
        print(f"Saving dataset to {self.save_path}/numpy")
        
        # Create directory
        os.makedirs(f"{self.save_path}/numpy", exist_ok=True)
        
        # Save arrays
        np.save(f"{self.save_path}/numpy/time_domain_data.npy", self.time_domain_data.astype(self.precision))
        np.save(f"{self.save_path}/numpy/range_doppler_maps.npy", self.range_doppler_maps.astype(self.precision))
        np.save(f"{self.save_path}/numpy/target_masks.npy", self.target_masks.astype(self.precision))
        
        # Save target information
        with open(f"{self.save_path}/numpy/target_info.npy", 'wb') as f:
            np.save(f, self.target_info)
        
        # Save parameters
        params = {
            'num_samples': self.num_samples,
            'num_range_bins': self.num_range_bins,
            'num_doppler_bins': self.num_doppler_bins,
            'sample_rate': self.sample_rate,
            'chirp_duration': self.chirp_duration,
            'num_chirps': self.num_chirps,
            'bandwidth': self.bandwidth,
            'center_freq': self.center_freq,
            'num_rx': self.num_rx,
            'num_tx': self.num_tx,
            'signal_type': self.signal_type,
            'range_resolution': self.range_resolution,
            'velocity_resolution': self.velocity_resolution,
            'max_range': self.max_range,
            'max_velocity': self.max_velocity
        }
        np.save(f"{self.save_path}/numpy/parameters.npy", params)

    def _load_data(self, datapath):
        """Load dataset from file
        
        Args:
            datapath: Path to dataset file
        """
        print(f"Loading dataset from {datapath}")
        
        if datapath.endswith('.h5'):
            self._load_hdf5(datapath)
        elif os.path.isdir(datapath) and os.path.exists(f"{datapath}/time_domain_data.npy"):
            self._load_numpy(datapath)
        else:
            raise ValueError(f"Unsupported data format: {datapath}")

    def _load_hdf5(self, datapath):
        """Load dataset from HDF5 file
        
        Args:
            datapath: Path to HDF5 file
        """
        if self.use_lazy_loading:
            # Open file for lazy loading
            self.h5_file = h5py.File(datapath, 'r')
            
            # Load parameters
            params = self.h5_file['parameters']
            self.num_samples = params.attrs['num_samples']
            self.num_range_bins = params.attrs['num_range_bins']
            self.num_doppler_bins = params.attrs['num_doppler_bins']
            self.sample_rate = params.attrs['sample_rate']
            self.chirp_duration = params.attrs['chirp_duration']
            self.num_chirps = params.attrs['num_chirps']
            self.bandwidth = params.attrs['bandwidth']
            self.center_freq = params.attrs['center_freq']
            self.num_rx = params.attrs['num_rx']
            self.num_tx = params.attrs['num_tx']
            self.signal_type = params.attrs['signal_type']
            self.range_resolution = params.attrs['range_resolution']
            self.velocity_resolution = params.attrs['velocity_resolution']
            self.max_range = params.attrs['max_range']
            self.max_velocity = params.attrs['max_velocity']
            
            # Initialize radar processor with loaded parameters
            self.radar_processor = RadarProcessing(
                num_range_bins=self.num_range_bins,
                num_doppler_bins=self.num_doppler_bins,
                sample_rate=self.sample_rate,
                chirp_duration=self.chirp_duration,
                num_chirps=self.num_chirps,
                bandwidth=self.bandwidth,
                center_freq=self.center_freq,
                signal_type=self.signal_type
            )
            
            # Get dataset shapes for __getitem__
            self.time_domain_shape = self.h5_file['time_domain_data'].shape
            self.range_doppler_shape = self.h5_file['range_doppler_maps'].shape
            self.target_mask_shape = self.h5_file['target_masks'].shape
            
            # Load target info
            self.target_info = []
            for i in range(self.num_samples):
                sample_targets = []
                if f'target_info/{i}' in self.h5_file:
                    target_group = self.h5_file[f'target_info/{i}']
                    for j in target_group:
                        target = {}
                        for key, value in target_group[j].attrs.items():
                            target[key] = value
                        sample_targets.append(target)
                self.target_info.append(sample_targets)
        else:
            # Load entire dataset into memory
            with h5py.File(datapath, 'r') as f:
                # Load data arrays
                self.time_domain_data = f['time_domain_data'][:]
                self.range_doppler_maps = f['range_doppler_maps'][:]
                self.target_masks = f['target_masks'][:]
                
                # Load parameters
                params = f['parameters']
                self.num_samples = params.attrs['num_samples']
                self.num_range_bins = params.attrs['num_range_bins']
                self.num_doppler_bins = params.attrs['num_doppler_bins']
                self.sample_rate = params.attrs['sample_rate']
                self.chirp_duration = params.attrs['chirp_duration']
                self.num_chirps = params.attrs['num_chirps']
                self.bandwidth = params.attrs['bandwidth']
                self.center_freq = params.attrs['center_freq']
                self.num_rx = params.attrs['num_rx']
                self.num_tx = params.attrs['num_tx']
                self.signal_type = params.attrs['signal_type']
                self.range_resolution = params.attrs['range_resolution']
                self.velocity_resolution = params.attrs['velocity_resolution']
                self.max_range = params.attrs['max_range']
                self.max_velocity = params.attrs['max_velocity']
                
                # Initialize radar processor with loaded parameters
                self.radar_processor = RadarProcessing(
                    num_range_bins=self.num_range_bins,
                    num_doppler_bins=self.num_doppler_bins,
                    sample_rate=self.sample_rate,
                    chirp_duration=self.chirp_duration,
                    num_chirps=self.num_chirps,
                    bandwidth=self.bandwidth,
                    center_freq=self.center_freq,
                    signal_type=self.signal_type
                )
                
                # Load target info
                self.target_info = []
                for i in range(self.num_samples):
                    sample_targets = []
                    if f'target_info/{i}' in f:
                        target_group = f[f'target_info/{i}']
                        for j in target_group:
                            target = {}
                            for key, value in target_group[j].attrs.items():
                                target[key] = value
                            sample_targets.append(target)
                    self.target_info.append(sample_targets)

    def _load_numpy(self, datapath):
        """Load dataset from NumPy files
        
        Args:
            datapath: Path to directory containing NumPy files
        """
        if self.use_memory_mapping:
            # Use memory mapping for large arrays
            self.time_domain_data = np.load(f"{datapath}/time_domain_data.npy", mmap_mode='r')
            self.range_doppler_maps = np.load(f"{datapath}/range_doppler_maps.npy", mmap_mode='r')
            self.target_masks = np.load(f"{datapath}/target_masks.npy", mmap_mode='r')
        else:
            # Load entire dataset into memory
            self.time_domain_data = np.load(f"{datapath}/time_domain_data.npy")
            self.range_doppler_maps = np.load(f"{datapath}/range_doppler_maps.npy")
            self.target_masks = np.load(f"{datapath}/target_masks.npy")
        
        # Load target information
        with open(f"{datapath}/target_info.npy", 'rb') as f:
            self.target_info = np.load(f, allow_pickle=True)
        
        # Load parameters
        params = np.load(f"{datapath}/parameters.npy", allow_pickle=True).item()
        self.num_samples = params['num_samples']
        self.num_range_bins = params['num_range_bins']
        self.num_doppler_bins = params['num_doppler_bins']
        self.sample_rate = params['sample_rate']
        self.chirp_duration = params['chirp_duration']
        self.num_chirps = params['num_chirps']
        self.bandwidth = params['bandwidth']
        self.center_freq = params['center_freq']
        self.num_rx = params['num_rx']
        self.num_tx = params['num_tx']
        self.signal_type = params['signal_type']
        self.range_resolution = params['range_resolution']
        self.velocity_resolution = params['velocity_resolution']
        self.max_range = params['max_range']
        self.max_velocity = params['max_velocity']
        
        # Initialize radar processor with loaded parameters
        self.radar_processor = RadarProcessing(
            num_range_bins=self.num_range_bins,
            num_doppler_bins=self.num_doppler_bins,
            sample_rate=self.sample_rate,
            chirp_duration=self.chirp_duration,
            num_chirps=self.num_chirps,
            bandwidth=self.bandwidth,
            center_freq=self.center_freq,
            signal_type=self.signal_type
        )

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.num_samples

    def __getitem__(self, idx):
        """Get a sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        if self.use_lazy_loading and self.h5_file is not None:
            # Check if sample is in cache
            if idx in list(self.data_cache.keys()):
                return self.data_cache.get(idx)
            
            # Load data from HDF5 file
            time_domain = self.h5_file['time_domain_data'][idx]
            range_doppler = self.h5_file['range_doppler_maps'][idx]
            target_mask = self.h5_file['target_masks'][idx]
            
            # Convert to torch tensors
            time_domain_tensor = torch.from_numpy(time_domain).float()
            range_doppler_tensor = torch.from_numpy(range_doppler).float()
            target_mask_tensor = torch.from_numpy(target_mask).float()
            
            # Create sample dictionary
            sample = {
                'time_domain': time_domain_tensor,
                'range_doppler': range_doppler_tensor,
                'target_mask': target_mask_tensor,
                'target_info': self.target_info[idx]
            }
            
            # Add to cache
            if len(self.data_cache) >= self.cache_size:
                # Remove oldest item if cache is full
                oldest_key = next(iter(self.data_cache))
                self.data_cache.pop(oldest_key, None)  # Use pop() to safely remove item from cache
            
            # Update cache using dictionary update() method which is thread-safe
            self.data_cache.update({idx: sample})
            
            return sample
        else:
            # Convert to torch tensors
            time_domain_tensor = torch.from_numpy(self.time_domain_data[idx]).float()
            range_doppler_tensor = torch.from_numpy(self.range_doppler_maps[idx]).float()
            target_mask_tensor = torch.from_numpy(self.target_masks[idx]).float()
            
            # Create sample dictionary
            sample = {
                'time_domain': time_domain_tensor,
                'range_doppler': range_doppler_tensor,
                'target_mask': target_mask_tensor,
                'target_info': self.target_info[idx]
            }
            
            return sample

    def visualize_sample(self, idx):
        """Visualize a sample from the dataset
        
        Args:
            idx: Sample index
        """
        # Get sample
        sample = self[idx]
        
        # Extract data
        range_doppler = sample['range_doppler'].numpy()
        target_mask = sample['target_mask'].numpy()
        target_info = sample['target_info']
        
        # Create figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot range-Doppler map (magnitude)
        rd_magnitude = np.sqrt(range_doppler[0]**2 + range_doppler[1]**2)
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)  # Convert to dB
        im = axs[0].imshow(rd_db, aspect='auto', cmap='jet')
        axs[0].set_title('Range-Doppler Map (Magnitude, dB)')
        axs[0].set_xlabel('Range Bin')
        axs[0].set_ylabel('Doppler Bin')
        plt.colorbar(im, ax=axs[0])
        
        # Plot target mask
        axs[1].imshow(target_mask[:, :, 0], aspect='auto', cmap='gray')
        axs[1].set_title('Target Mask')
        axs[1].set_xlabel('Range Bin')
        axs[1].set_ylabel('Doppler Bin')
        
        # Add target annotations
        for target in target_info:
            range_bin = int(target['distance'] / self.range_resolution)
            doppler_bin = int(self.num_doppler_bins/2 + target['velocity'] / self.velocity_resolution)
            
            if (0 <= range_bin < self.num_range_bins and 
                0 <= doppler_bin < self.num_doppler_bins):
                axs[0].plot(range_bin, doppler_bin, 'ro')
                axs[1].plot(range_bin, doppler_bin, 'ro')
                
                # Add target information
                axs[0].text(range_bin + 1, doppler_bin + 1, 
                          f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                          color='white', fontsize=8, backgroundcolor='black')
        
        # Add sample information
        plt.suptitle(f'Sample {idx}: {len(target_info)} Targets, Signal Type: {self.signal_type}')
        
        # Adjust layout and show
        plt.tight_layout()
        plt.show()


def print_dataset_info(dataset):
    """Print detailed information about the dataset"""
    print("\n" + "="*80)
    print("RADAR DATASET INFORMATION")
    print("="*80)
    
    # Basic dataset information
    print(f"Number of samples: {dataset.num_samples}")
    print(f"Signal type: {dataset.signal_type}")
    
    # Radar parameters
    print("\nRADAR PARAMETERS:")
    print(f"  Sample rate: {dataset.sample_rate/1e6:.1f} MHz")
    print(f"  Chirp duration: {dataset.chirp_duration*1e6:.1f} μs")
    print(f"  Number of chirps: {dataset.num_chirps}")
    print(f"  Bandwidth: {dataset.bandwidth/1e6:.1f} MHz")
    print(f"  Center frequency: {dataset.center_freq/1e9:.2f} GHz")
    print(f"  Number of RX antennas: {dataset.num_rx}")
    print(f"  Number of TX antennas: {dataset.num_tx}")
    
    # Resolution and range information
    print("\nRESOLUTION AND RANGE:")
    print(f"  Range resolution: {dataset.range_resolution:.2f} m")
    # Calculate unambiguous range based on PRF
    speed_of_light = 3e8  # m/s
    prf = 1.0 / dataset.chirp_duration  # Pulse repetition frequency
    unambiguous_range = speed_of_light / (2 * prf)
    
    print(f"  Maximum range: {dataset.max_range:.2f} m")
    print(f"  Unambiguous range: {unambiguous_range:.2f} m")
    
    # Check if unambiguous range is realistic
    if unambiguous_range > 500:
        print(f"  WARNING: Unambiguous range is very high ({unambiguous_range:.2f}m).")
        print(f"  Consider reducing chirp_duration to get a more realistic unambiguous range.")
    
    print(f"  Velocity resolution: {dataset.velocity_resolution:.2f} m/s")
    print(f"  Maximum velocity: {dataset.max_velocity:.2f} m/s")
    
    # Data dimensions
    print("\nDATA DIMENSIONS:")
    print(f"  Time domain data: {dataset.time_domain_data.shape}")
    print(f"    - Interpretation: [num_samples, num_rx, num_chirps, samples_per_chirp, 2(I/Q)]")
    print(f"  Range-Doppler maps: {dataset.range_doppler_maps.shape}")
    print(f"    - Interpretation: [num_samples, 2(real/imag), num_doppler_bins, num_range_bins]")
    print(f"  Target masks: {dataset.target_masks.shape}")
    print(f"    - Interpretation: [num_samples, num_doppler_bins, num_range_bins, 1]")
    
    # Memory usage estimation
    time_domain_size = np.prod(dataset.time_domain_data.shape) * (2 if dataset.precision == 'float32' else 1)
    rd_maps_size = np.prod(dataset.range_doppler_maps.shape) * (2 if dataset.precision == 'float32' else 1)
    masks_size = np.prod(dataset.target_masks.shape) * (2 if dataset.precision == 'float32' else 1)
    total_size = time_domain_size + rd_maps_size + masks_size
    
    print("\nMEMORY USAGE ESTIMATION:")
    print(f"  Time domain data: {time_domain_size/1e6:.2f} MB")
    print(f"  Range-Doppler maps: {rd_maps_size/1e6:.2f} MB")
    print(f"  Target masks: {masks_size/1e6:.2f} MB")
    print(f"  Total: {total_size/1e6:.2f} MB ({total_size/1e9:.2f} GB)")
    
    print("="*80 + "\n")
    
def visualize_dataset_statistics(dataset, num_samples=5):
    """Visualize statistics of the dataset"""
    # Create directory for statistics figures
    os.makedirs(f"{dataset.save_path}/statistics", exist_ok=True)
    
    # Collect statistics
    num_targets = []
    distances = []
    velocities = []
    rcs_values = []
    
    # Sample indices to visualize
    indices_to_visualize = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Collect data from all samples
    for i in range(len(dataset)):
        targets = dataset.target_info[i]
        num_targets.append(len(targets))
        
        for target in targets:
            distances.append(target['distance'])
            velocities.append(target['velocity'])
            rcs_values.append(target.get('rcs', 0.5))  # Default RCS if not available
    
    # Create figure for statistics
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot number of targets histogram
    axs[0, 0].hist(num_targets, bins=range(1, dataset.max_targets + 2), align='left', rwidth=0.8)
    axs[0, 0].set_title('Number of Targets per Sample')
    axs[0, 0].set_xlabel('Number of Targets')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_xticks(range(1, dataset.max_targets + 1))
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot distance histogram
    axs[0, 1].hist(distances, bins=20)
    axs[0, 1].set_title('Target Distance Distribution')
    axs[0, 1].set_xlabel('Distance (m)')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot velocity histogram
    axs[1, 0].hist(velocities, bins=20)
    axs[1, 0].set_title('Target Velocity Distribution')
    axs[1, 0].set_xlabel('Velocity (m/s)')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot RCS histogram
    axs[1, 1].hist(rcs_values, bins=20)
    axs[1, 1].set_title('Target RCS Distribution')
    axs[1, 1].set_xlabel('RCS (normalized)')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{dataset.save_path}/statistics/dataset_statistics{IMG_FORMAT}")
    plt.close()
    
    # Visualize selected samples
    for idx in indices_to_visualize:
        dataset.visualize_sample(idx)
        plt.savefig(f"{dataset.save_path}/statistics/sample_{idx}_visualization{IMG_FORMAT}")
        plt.close()
    
    # Create a 2D scatter plot of targets in range-velocity space
    plt.figure(figsize=(10, 8))
    plt.scatter(distances, velocities, alpha=0.5)
    plt.title('Targets in Range-Velocity Space')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.savefig(f"{dataset.save_path}/statistics/range_velocity_scatter{IMG_FORMAT}")
    plt.close()
    
    # Create a heatmap of target density in range-velocity space
    plt.figure(figsize=(10, 8))
    h, xedges, yedges, im = plt.hist2d(distances, velocities, bins=[20, 20], cmap='jet')
    plt.colorbar(label='Number of Targets')
    plt.title('Target Density in Range-Velocity Space')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    plt.savefig(f"{dataset.save_path}/statistics/range_velocity_heatmap{IMG_FORMAT}")
    plt.close()
    
    print(f"Dataset statistics visualized and saved to {dataset.save_path}/statistics/")

def visualize_signal_processing_steps(dataset, sample_idx=0):
    """Visualize the signal processing steps for a single sample"""
    # Create directory for processing figures
    os.makedirs(f"{dataset.save_path}/processing", exist_ok=True)
    
    # Get sample data
    sample = dataset[sample_idx]
    time_domain = sample['time_domain'].numpy()
    range_doppler = sample['range_doppler'].numpy()
    target_info = sample['target_info']
    
    # Convert time domain data back to complex
    # Shape: [num_rx, num_chirps, samples_per_chirp]
    complex_data = time_domain[:, :, :, 0] + 1j * time_domain[:, :, :, 1]
    
    # Create figure with multiple subplots for processing steps
    fig, axs = plt.subplots(4, 2, figsize=(15, 24))
    
    # 1. Plot original received time domain signal (real and imaginary parts)
    rx_idx, chirp_idx = 0, 0  # First RX, first chirp
    axs[0, 0].plot(np.real(complex_data[rx_idx, chirp_idx]))
    axs[0, 0].set_title(f'Original Received Signal (Real Part) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[0, 0].set_xlabel('Sample')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].plot(np.imag(complex_data[rx_idx, chirp_idx]))
    axs[0, 1].set_title(f'Original Received Signal (Imaginary Part) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[0, 1].set_xlabel('Sample')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True, alpha=0.3)
    
    # 2. Simulate hardware demodulation (CN0566 to AD9361)
    # Use the radar processor to perform the hardware demodulation
    demodulated_data = dataset.radar_processor.simulate_hardware_demodulation(complex_data)
    
    # Plot demodulated signal
    axs[1, 0].plot(np.real(demodulated_data[rx_idx, chirp_idx]))
    axs[1, 0].set_title(f'Demodulated Signal (Real Part) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[1, 0].set_xlabel('Sample')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].grid(True, alpha=0.3)
    
    axs[1, 1].plot(np.imag(demodulated_data[rx_idx, chirp_idx]))
    axs[1, 1].set_title(f'Demodulated Signal (Imaginary Part) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[1, 1].set_xlabel('Sample')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].grid(True, alpha=0.3)
    
    # 3. Range processing on demodulated data
    # Use the radar processor to perform range FFT
    range_profiles = np.zeros((dataset.num_rx, dataset.num_chirps, dataset.num_range_bins), dtype=np.complex64)
    
    for rx in range(dataset.num_rx):
        for chirp in range(dataset.num_chirps):
            # Apply window function
            window = np.hamming(dataset.samples_per_chirp)
            windowed_data = demodulated_data[rx, chirp] * window
            
            # Perform range FFT
            range_fft = np.fft.fft(windowed_data, n=dataset.num_range_bins)
            range_profiles[rx, chirp] = range_fft
    
    # Plot range profile for the first chirp of the first RX
    range_profile_mag = np.abs(range_profiles[rx_idx, chirp_idx])
    range_profile_db = 20 * np.log10(range_profile_mag + 1e-10)
    
    axs[2, 0].plot(range_profile_db)
    axs[2, 0].set_title(f'Range Profile (dB) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[2, 0].set_xlabel('Range Bin')
    axs[2, 0].set_ylabel('Magnitude (dB)')
    axs[2, 0].grid(True, alpha=0.3)
    
    # Add secondary x-axis for range in meters
    ax_range = axs[2, 0].secondary_xaxis('top', functions=(
        lambda x: x * dataset.range_resolution,  # bin to meters
        lambda x: x / dataset.range_resolution   # meters to bin
    ))
    ax_range.set_xlabel('Range (m)')
    
    # 4. Doppler processing on range profiles
    # Find the range bin with maximum energy
    max_range_bin = np.argmax(np.mean(np.abs(range_profiles[rx_idx])**2, axis=0))
    
    # Extract range bin data across all chirps for the first RX
    range_bin_data = range_profiles[rx_idx, :, max_range_bin]
    
    # Apply Doppler FFT
    doppler_window = np.hamming(dataset.num_chirps)
    windowed_doppler = range_bin_data * doppler_window
    doppler_fft = np.fft.fftshift(np.fft.fft(windowed_doppler, n=dataset.num_doppler_bins))
    doppler_profile_mag = np.abs(doppler_fft)
    doppler_profile_db = 20 * np.log10(doppler_profile_mag + 1e-10)
    
    axs[2, 1].plot(doppler_profile_db)
    axs[2, 1].set_title(f'Doppler Profile (dB) - RX {rx_idx}, Range Bin {max_range_bin}')
    axs[2, 1].set_xlabel('Doppler Bin')
    axs[2, 1].set_ylabel('Magnitude (dB)')
    axs[2, 1].grid(True, alpha=0.3)
    
    # Add secondary x-axis for velocity in m/s
    ax_velocity = axs[2, 1].secondary_xaxis('top', functions=(
        lambda y: (y - dataset.num_doppler_bins/2) * dataset.velocity_resolution,  # bin to m/s
        lambda y: y / dataset.velocity_resolution + dataset.num_doppler_bins/2     # m/s to bin
    ))
    ax_velocity.set_xlabel('Velocity (m/s)')
    
    # 5. Complete Range-Doppler processing using the radar processor
    # This uses the full two-step process implemented in AIradar_processing.py
    rd_map = dataset.radar_processor.time_to_range_doppler(complex_data)
    
    # Plot the final range-Doppler map
    rd_magnitude = np.abs(rd_map)
    rd_db = 20 * np.log10(rd_magnitude + 1e-10)  # Convert to dB
    im = axs[3, 0].imshow(rd_db, aspect='auto', cmap='jet', origin='lower')
    axs[3, 0].set_title('Range-Doppler Map (Full Processing)')
    axs[3, 0].set_xlabel('Range Bin')
    axs[3, 0].set_ylabel('Doppler Bin')
    plt.colorbar(im, ax=axs[3, 0])
    
    # Generate and plot target mask for visualization
    target_mask = np.zeros((dataset.num_doppler_bins, dataset.num_range_bins, 1), dtype=np.float32)
    
    # Create mask based on target information
    for target in target_info:
        range_bin = int(target['distance'] / dataset.range_resolution)
        doppler_bin = int(dataset.num_doppler_bins/2 + target['velocity'] / dataset.velocity_resolution)
        
        if (0 <= range_bin < dataset.num_range_bins and 
            0 <= doppler_bin < dataset.num_doppler_bins):
            # Set mask width based on range resolution
            mask_width = max(2, int(0.1 * dataset.num_range_bins))
            mask_height = max(2, int(0.1 * dataset.num_doppler_bins))
            
            for i in range(max(0, doppler_bin - mask_height), 
                           min(dataset.num_doppler_bins, doppler_bin + mask_height + 1)):
                for j in range(max(0, range_bin - mask_width), 
                               min(dataset.num_range_bins, range_bin + mask_width + 1)):
                    # Use a Gaussian-like falloff for more natural looking targets
                    dist_sq = ((i - doppler_bin) / mask_height)**2 + ((j - range_bin) / mask_width)**2
                    target_mask[i, j, 0] = max(target_mask[i, j, 0], np.exp(-dist_sq))
    
    # Plot the target mask
    mask_im = axs[3, 1].imshow(target_mask[:, :, 0], aspect='auto', cmap='hot', origin='lower')
    axs[3, 1].set_title('Target Mask')
    axs[3, 1].set_xlabel('Range Bin')
    axs[3, 1].set_ylabel('Doppler Bin')
    plt.colorbar(mask_im, ax=axs[3, 1])
    
    # Add target annotations to mask plot
    for target in target_info:
        range_bin = int(target['distance'] / dataset.range_resolution)
        doppler_bin = int(dataset.num_doppler_bins/2 + target['velocity'] / dataset.velocity_resolution)
        
        if (0 <= range_bin < dataset.num_range_bins and 
            0 <= doppler_bin < dataset.num_doppler_bins):
            axs[3, 1].plot(range_bin, doppler_bin, 'wo', markersize=8, markeredgecolor='black')
            
            # Add target information text
            axs[3, 1].text(range_bin + 2, doppler_bin, 
                      f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                      color='white', fontsize=8, backgroundcolor='black')
    
    # 6. Create a new figure for 3D surface of range-Doppler map
    fig_3d = plt.figure(figsize=(10, 8))
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(np.arange(dataset.num_range_bins), np.arange(dataset.num_doppler_bins))
    
    # Plot surface
    surf = ax3d.plot_surface(X, Y, rd_magnitude, cmap='jet', alpha=0.8)
    
    # Add colorbar
    plt.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    # Set labels
    ax3d.set_title('Range-Doppler Map (3D Surface)')
    ax3d.set_xlabel('Range Bin')
    ax3d.set_ylabel('Doppler Bin')
    ax3d.set_zlabel('Magnitude')
    
    # Add target markers in 3D
    for target in target_info:
        range_bin = int(target['distance'] / dataset.range_resolution)
        doppler_bin = int(dataset.num_doppler_bins/2 + target['velocity'] / dataset.velocity_resolution)
        
        if (0 <= range_bin < dataset.num_range_bins and 
            0 <= doppler_bin < dataset.num_doppler_bins):
            # Get magnitude at target location
            magnitude = rd_magnitude[doppler_bin, range_bin]
            ax3d.scatter([range_bin], [doppler_bin], [magnitude], color='r', s=50)
    
    # Save the 3D figure separately
    plt.tight_layout()
    plt.savefig(f"{dataset.save_path}/processing/sample_{sample_idx}_3d_range_doppler{IMG_FORMAT}")
    plt.close(fig_3d)

    
if __name__ == '__main__':
    # Test and visualization of the RadarDataset
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Radar Dataset Generation and Visualization')
    parser.add_argument('--mode', type=str, default='generate', choices=['generate', 'load', 'visualize'],
                        help='Mode: generate new data, load existing data, or visualize existing data')
    parser.add_argument('--datapath', type=str, default=None, 
                        help='Path to existing dataset (for load or visualize modes)')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples to generate')
    parser.add_argument('--save_path', type=str, default='data/radarv3',
                        help='Path to save generated data')
    parser.add_argument('--format', type=str, default='hdf5', choices=['hdf5', 'numpy'],
                        help='Format to save/load data')
    parser.add_argument('--signal_type', type=str, default='FMCW', 
                        choices=['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW'],
                        help='Type of radar signal')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='Number of samples to visualize')
    args = parser.parse_args()
    
    args.save_path = os.path.join(args.save_path, args.signal_type)
    # Main execution based on mode
    if args.mode == 'generate':
        print(f"Generating {args.num_samples} radar samples with signal type: {args.signal_type}")
        
        # Create dataset with specified parameters
        dataset = RadarDataset(
            num_samples=args.num_samples,
            signal_type=args.signal_type,
            save_path=args.save_path,
            savedataformat=args.format,
            drawfig=True  # Enable figure drawing for visualization
        )
        
        # Print dataset information
        print_dataset_info(dataset)
        
        # Visualize dataset statistics
        visualize_dataset_statistics(dataset, args.visualize_samples)
        
        # Visualize signal processing steps for a random sample
        sample_idx = random.randint(0, len(dataset) - 1)
        visualize_signal_processing_steps(dataset, sample_idx)
        
    elif args.mode == 'load':
        if args.datapath is None:
            print("Error: --datapath must be specified in 'load' mode")
            exit(1)
            
        print(f"Loading radar dataset from: {args.datapath}")
        
        # Load dataset from specified path
        dataset = RadarDataset(
            datapath=args.datapath,
            save_path=args.save_path
        )
        
        # Print dataset information
        print_dataset_info(dataset)
        
    elif args.mode == 'visualize':
        if args.datapath is None:
            print("Error: --datapath must be specified in 'visualize' mode")
            exit(1)
            
        print(f"Visualizing radar dataset from: {args.datapath}")
        
        # Load dataset from specified path
        dataset = RadarDataset(
            datapath=args.datapath,
            save_path=args.save_path
        )
        
        # Print dataset information
        print_dataset_info(dataset)
        
        # Visualize dataset statistics
        visualize_dataset_statistics(dataset, args.visualize_samples)
        
        # Visualize signal processing steps for a random sample
        sample_idx = random.randint(0, len(dataset) - 1)
        visualize_signal_processing_steps(dataset, sample_idx)
    
    print("Done!")