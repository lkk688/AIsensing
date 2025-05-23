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
import time

class RadarDataset(Dataset):
    # In the RadarDataset class initialization, update the default parameters
    def __init__(self, 
                 num_samples=100,
                 num_range_bins=128,  # Increased from 64 for better range resolution
                 num_doppler_bins=16,  # Increased from 12 for better velocity resolution
                 sample_rate=1.5e6, #1.5e6,    # reduce from 15MHz to 1.5MHz Adjusted to match hardware constraints
                 transceiver_bandwidth=30e6,  # AD9361 bandwidth limitation
                 #chirp_duration=1e-4,  # reduce from 1ms to 0.1ms Increased to 1ms for better SNR (from 500μs)
                 num_chirps=32,       # Increased from 12 for better integration gain
                 bandwidth=500e6,     # Fixed by CN0566 capabilities
                 transceiver_center_freq=2.1e9,  # AD9361 center frequency
                 center_freq=10e9,    # CN0566 output frequency
                 num_rx=4,            # 4 RX antennas (typical for Phaser)
                 num_tx=1,            # 1 TX antenna
                 max_targets=3,       # Maximum 3 targets per sample
                 snr_min=10,          # Minimum SNR in dB
                 snr_max=25,          # Maximum SNR in dB
                 signal_type='FMCW',  # Changed from OFDM to FMCW for better range performance
                 signal_freq=1e6,     # Increased from 100kHz for better modulation
                 apply_realistic_effects=True,
                 save_path='data/radarv3_100m',  # Updated path to indicate 100m optimization
                 savedataformat='hdf5',
                 precision='float32',
                 drawfig=False,
                 datapath=None,
                 use_lazy_loading=False,
                 use_memory_mapping=False,
                 cache_size=100):
        """
        Initialize radar dataset optimized for 100-meter detection range
        
        Args:
            num_samples: Number of samples to generate
            num_range_bins: Number of range bins that determines range resolution granularity
            num_doppler_bins: Number of Doppler bins that determines velocity resolution granularity
            sample_rate: Sample rate in Hz (15 MHz based on hardware constraints)
            chirp_duration: Chirp duration in seconds (1ms for better SNR)
            num_chirps: Number of chirps per frame (32 for better integration gain)
            bandwidth: Signal bandwidth in Hz (500 MHz fixed by CN0566)
            transceiver_center_freq: Center frequency of AD9361 transceiver (2.1 GHz)
            center_freq: Output frequency of CN0566 (10 GHz)
            num_rx: Number of RX antennas
            num_tx: Number of TX antennas
            max_targets: Maximum number of targets per sample
            snr_min: Minimum SNR in dB
            snr_max: Maximum SNR in dB
            signal_type: Type of radar signal ('FMCW' for best range performance)
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

        self.transceiver_bandwidth = transceiver_bandwidth  # AD9361 bandwidth limitation
        self.num_subcarriers = 128 #for OFDM
        self.subcarrier_spacing = 30e3  # OFDM subcarrier spacing
        
        # Set precision for data (MPS framework doesn't support float64)
        self.precision = precision
        if precision not in ['float32', 'float16']:
            print(f"Warning: Unsupported precision '{precision}'. Using 'float32' instead.")
            self.precision = 'float32'
        
        # Validate signal type
        valid_signal_types = ['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW']
        if signal_type not in valid_signal_types:
            print(f"Warning: Invalid signal type '{signal_type}'. Using 'FMCW' instead.")
            self.signal_type = 'FMCW'
        else:
            self.signal_type = signal_type

        # Store SDR parameters
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.center_freq = center_freq
        self.transceiver_center_freq = transceiver_center_freq
        
        # For FMCW radar, we want to ensure the chirp rate is appropriate for visualization
        # A good rule of thumb is to have the chirp duration be proportional to the ratio of
        # bandwidth to center frequency to avoid too many oscillations in one chirp
        carrier_to_bw_ratio = self.center_freq / self.bandwidth
        
        # Calculate appropriate sample rate based on bandwidth
        # According to Nyquist, we need at least 2x the bandwidth for proper sampling
        if self.signal_type == 'FMCW':
            # For FMCW, we need higher sampling rate to properly capture the chirp
            self.sample_rate = max(2.5 * self.bandwidth, sample_rate)
            print(f"Automatically calculated sample rate for FMCW: {self.sample_rate/1e6:.2f} MHz")
        elif self.signal_type in ['OFDM', 'OFDM_FMCW']:
            # For OFDM, sample rate should be at least num_subcarriers * subcarrier_spacing
            min_sample_rate = self.num_subcarriers * self.subcarrier_spacing * 1.2  # 20% margin
            self.sample_rate = max(min_sample_rate, 2.2 * self.bandwidth, sample_rate)
            print(f"Automatically calculated sample rate for OFDM: {self.sample_rate/1e6:.2f} MHz")
        elif self.signal_type in ['Sine', 'Sine_FMCW']:
            # For sine wave, we need at least 10 samples per period
            min_sample_rate = 10 * self.signal_freq
            self.sample_rate = max(min_sample_rate, 2.2 * self.bandwidth, sample_rate)
            print(f"Automatically calculated sample rate for Sine: {self.sample_rate/1e6:.2f} MHz")
        else:
            # Default case - use at least 2.2x bandwidth (Nyquist + margin)
            self.sample_rate = max(2.2 * self.bandwidth, sample_rate)
            print(f"Using default sample rate: {self.sample_rate/1e6:.2f} MHz")
            
        # Check if sample rate is too high for practical simulation
        max_practical_sample_rate = 2e9  # 2 GHz is a reasonable upper limit for simulation
        if self.sample_rate > max_practical_sample_rate:
            print(f"Warning: Calculated sample rate ({self.sample_rate/1e6:.2f} MHz) is very high.")
            print(f"Limiting to {max_practical_sample_rate/1e6:.2f} MHz for practical simulation.")
            self.sample_rate = max_practical_sample_rate
        
        # Base chirp duration that would work well for visualization
        # Longer duration for higher carrier-to-bandwidth ratio
        self.chirp_duration = (carrier_to_bw_ratio / 20) * 1e-4
        
        # Ensure chirp duration is within reasonable limits
        self.chirp_duration = max(0.8e-4, min(5e-4, self.chirp_duration))
        
        print(f"Automatically calculated chirp duration: {self.chirp_duration:.2e} seconds")

        self.num_chirps = num_chirps
        
        
        self.signal_freq = signal_freq #used for Sine_FMCW
        self.num_rx = num_rx
        self.num_tx = num_tx
        
        # Store realistic effects parameters
        self.apply_realistic_effects = apply_realistic_effects
        #self.recalculate_rd_map = recalculate_rd_map
        
        # Initialize the radar processor
        # Initialize the radar processor
        self.radar_processor = RadarProcessing(
            num_range_bins=self.num_range_bins,
            num_doppler_bins=self.num_doppler_bins,
            sample_rate=self.sample_rate,
            chirp_duration=self.chirp_duration,
            num_chirps=self.num_chirps,
            num_subcarriers=self.num_subcarriers,
            subcarrier_spacing=self.subcarrier_spacing,  # OFDM subcarrier spacing
            bandwidth=self.bandwidth,
            transceiver_bandwidth=self.transceiver_bandwidth,
            transceiver_center_freq=self.transceiver_center_freq,
            output_freq=self.center_freq,  # CN0566 output frequency (10 GHz)
            signal_type=self.signal_type,
            signal_freq=self.signal_freq
        )
        
        # Calculate derived parameters
        #1.5e6 * 1e-4 = 150
        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        self.range_resolution = self.radar_processor.range_resolution
        self.max_range = self.radar_processor.max_range
        self.min_range = self.radar_processor.min_range
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

    def generate_radar_data(self, save_data=True, format='hdf5', do_debug=True):
        """Generate radar dataset with random targets and signals

        Args:
            save_data: Whether to save the dataset to disk
            format: Format to save the dataset ('hdf5' or 'numpy')

        Returns:
            Generated dataset
        """
        print(f"Generating {self.num_samples} radar samples with {self.signal_type} signals...")
        """Generate radar dataset with random targets and signals
        
        Args:
            save_data: Whether to save the dataset to disk
            format: Format to save the dataset ('hdf5' or 'numpy')
            
        Returns:
            Generated dataset
        """
        print(f"Generating {self.num_samples} radar samples with {self.signal_type} signals...")
        
        # Create directory for saving data
        if save_data:
            os.makedirs(self.save_path, exist_ok=True)
        
        # Generate a test TX signal to ensure samples_per_chirp is correctly calculated
        test_tx_signal = self._generate_tx_signal() #(num_chirps, sample_per_chirp) complex
        # Update samples_per_chirp based on the actual signal length
        self.samples_per_chirp = test_tx_signal.shape[1]


        # Initialize arrays to store data
        self.time_domain_data = np.zeros((self.num_samples, self.num_rx, self.num_chirps, 
                                         self.samples_per_chirp, 2), dtype=self.precision) #(100, 4, 32, 150, 2)
        self.range_doppler_maps = np.zeros((self.num_samples, 2, self.num_doppler_bins, 
                                           self.num_range_bins), dtype=self.precision) #(100, 2, 16, 128)
        self.target_masks = np.zeros((self.num_samples, self.num_doppler_bins, 
                                     self.num_range_bins, 1), dtype=self.precision) #(100, 16, 128, 1)
        
        # Initialize lists to store additional information
        self.target_info = []
        self.tx_signals = []
        self.tx_powers = []
        self.channel_models = []
        self.snr_values = []
        self.signal_types = []

        #debug one sample
        if do_debug:
            # Run the debug function to add an ideal target
            rx_signal, rd_map, target_info = self.debug_add_ideal_target(test_tx_signal)

           # Debug the range-Doppler processing
            rd_map, debug_info = self.debug_time_to_range_doppler(rx_signal)

            # Now you can examine the debug_info dictionary and visualizations
            print(f"Peak-to-noise ratio: {debug_info['peak_to_noise_db']:.2f} dB")
        
        # Generate samples
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            # Randomly select signal parameters for this sample
            #self._randomize_signal_parameters()
            
            # Generate random TX power for this sample
            tx_power = np.random.uniform(0.5, 1.5)
            
            # Generate TX signal with the random parameters
            tx_signal = self._generate_tx_signal(tx_power=tx_power) #(32, 125000)
            if i % 100 == 0:
                self._visualize_tx_signal(tx_signal, title=f"TX Signal {i}", save_path=f"{self.save_path}/tx_signal_{i}{IMG_FORMAT}")
            
            # Initialize RX signal (all zeros)
            rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), 
                                dtype=np.complex64)
            
            # Generate random number of targets (0 to max_targets)
            num_targets = random.randint(0, self.max_targets)
            
            # List to store target information for this sample
            sample_targets = []
            
            # Generate target parameters if we have targets
            if num_targets > 0:
                for _ in range(num_targets):
                    # Generate random target parameters
                    distance = random.uniform(self.min_range, self.max_range)
                    velocity = random.uniform(-self.max_velocity, self.max_velocity)
                    rcs = random.uniform(1.0, 20.0)  # Radar Cross Section
                    
                    # Store target information
                    target = {
                        'distance': distance,
                        'velocity': velocity,
                        'rcs': rcs
                    }
                    sample_targets.append(target)
            
            # Simulate radar channel with all effects
            rx_signal = self._simulate_radar_channel(
                rx_signal=rx_signal,
                tx_signal=tx_signal,
                targets=sample_targets,
                add_crosstalk=self.apply_realistic_effects, 
                add_ground_clutter=self.apply_realistic_effects,
                add_system_noise=self.apply_realistic_effects,
                crosstalk_isolation_db=30,
                crosstalk_delay_samples=5,
                system_noise_power=1e-9 if self.apply_realistic_effects else 1e-12,
                clutter_probability=0.1 if num_targets == 0 else 0.05  # More clutter when no targets
            )
            if i % 100 == 0:
                self._visualize_rx_signal(rx_signal, target_info=sample_targets, title=f"RX Signal {i}", save_path=f"{self.save_path}/rx_signal_{i}{IMG_FORMAT}")

            # Generate random SNR for this sample
            snr_db = random.uniform(self.snr_min, self.snr_max)
            
            # Add noise to the received signal
            rx_signal = self._add_noise(rx_signal, snr_db) #(4, 32, 125000)
            
            
            # Process the received signal to generate range-Doppler map
            #rd_map = self.radar_processor.generate_range_doppler_map(rx_signal)
            rd_map = self.radar_processor.time_to_range_doppler(rx_signal) #(2, 16, 128) float64
            # Visualize range-Doppler map# Visualize range-Doppler map for selected samples
            #if i % 100 == 0:
            self._visualize_range_doppler_map(rd_map, sample_targets, 
                                                title=f"Range-Doppler Map {i}", 
                                                save_path=f"{self.save_path}/rd_map_{i}{IMG_FORMAT}")
            self._visualize_range_doppler_map3d(rd_map, sample_targets, 
                                                title=f"Range-Doppler Map {i}", 
                                                save_path=f"{self.save_path}/rd_map_3d_{i}{IMG_FORMAT}")


            # Process radar data with CFAR detection
            results = self.radar_processor.process_radar_data(
                rx_signal, 
                apply_cfar=True,
                guard_cells=(2, 2),
                training_cells=(4, 4),
                pfa=1e-4
            )
            # Access results
            rd_map = results['range_doppler_map'] #(2, 16, 128)
            cfar_map = results['cfar_map'] #(16, 128)
            detected_targets = results['detected_targets']

            # Visualize both maps together
            self._visualize_range_doppler_map(
                rd_map, 
                sample_targets, 
                title=f"Range-Doppler Map with CFAR {i}", 
                save_path=f"{self.save_path}/rd_map_cfar_{i}{IMG_FORMAT}",
                cfar_map=cfar_map
            )

            # Create target mask
            target_mask = self._create_target_mask(sample_targets)
            # Visualize detection comparison
            self._visualize_detection_comparison(
                rd_map,
                detected_targets,
                target_mask,
                sample_targets,
                title=f"Detection Comparison {i}",
                save_path=f"{self.save_path}/detection_comparison_{i}{IMG_FORMAT}"
            )
            
            # Store data (100, 4, 32, 150, 2) (4, 32, 176)
            self.time_domain_data[i, :, :, :, 0] = np.real(rx_signal)
            self.time_domain_data[i, :, :, :, 1] = np.imag(rx_signal)
            # self.range_doppler_maps[i, 0] = np.real(rd_map)
            # self.range_doppler_maps[i, 1] = np.imag(rd_map)
            # Fix the broadcasting issue - rd_map shape is (2, 16, 128)
            # and range_doppler_maps[i] shape is (2, 16, 128)
            self.range_doppler_maps[i] = rd_map  # Direct assignment of the whole array
            self.target_masks[i] = target_mask
            self.target_info.append(sample_targets)
            
            # Store additional information
            self.tx_signals.append(tx_signal)
            self.tx_powers.append(tx_power)
            self.channel_models.append({
                'apply_realistic_effects': self.apply_realistic_effects,
                'isolation_db': 30 if self.apply_realistic_effects else None,
                'system_delay_samples': 5 if self.apply_realistic_effects else None
            })
            self.snr_values.append(snr_db)
            self.signal_types.append(self.signal_type)
            
                        # Draw sample visualization if requested
            if self.drawfig and i % 10 == 0:
                self._draw_sample(i, rx_signal, rd_map, sample_targets)
        
        # Save the dataset
        if save_data:
            if format.lower() == 'hdf5':
                self._save_hdf5()
            elif format.lower() == 'numpy':
                self._save_numpy()
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        return self.time_domain_data, self.range_doppler_maps, self.target_masks, self.target_info
    
    def debug_add_ideal_target(self, tx_signal, visualize=True):
        """
        Add a single ideal target to the transmitted signal for debugging purposes.
        This function creates a perfect target reflection with high RCS and no noise
        to ensure it's clearly visible in the range-Doppler map.
        
        Args:
            tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp]
            visualize: Whether to visualize the signals and range-Doppler map
            
        Returns:
            Tuple of (rx_signal, rd_map, target_info)
        """
        print("=== DEBUG MODE: Adding ideal target ===")
        
        # Initialize RX signal (all zeros)
        rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), 
                            dtype=np.complex64)
        
        # Create a single target with ideal parameters
        # - Place at 30% of max range for clear visibility
        # - Use moderate velocity (30% of max) to create clear Doppler shift
        # - Use very high RCS to ensure strong reflection
        target_distance = self.max_range * 0.3
        target_velocity = self.max_velocity * 0.3
        target_rcs = 100.0  # Very high RCS for clear visibility
        
        # Store target information
        target_info = {
            'distance': target_distance,
            'velocity': target_velocity,
            'rcs': target_rcs
        }
        
        print(f"Ideal target parameters:")
        print(f"  - Distance: {target_distance:.2f} m")
        print(f"  - Velocity: {target_velocity:.2f} m/s")
        print(f"  - RCS: {target_rcs:.2f} m²")
        
        # Calculate target parameters
        # Time delay for the target (round trip)
        delay_seconds = 2 * target_distance / self.speed_of_light
        delay_samples = int(delay_seconds * self.sample_rate)
        
        # Doppler shift due to target velocity
        doppler_freq = 2 * target_velocity * self.center_freq / self.speed_of_light
        
        # Calculate attenuation (using radar equation principles but amplified)
        # We're using a simplified version with very high gain for debugging
        attenuation = np.sqrt(target_rcs) / (target_distance ** 1.5)  # Reduced exponent for stronger signal
        attenuation *= 1e3  # Amplification factor to make target clearly visible
        
        print(f"Target signal parameters:")
        print(f"  - Delay: {delay_samples} samples ({delay_seconds*1e6:.2f} μs)")
        print(f"  - Doppler shift: {doppler_freq:.2f} Hz")
        print(f"  - Attenuation factor: {attenuation:.6f}")
        
        # IMPORTANT FIX: Check if delay is too large for the signal
        if delay_samples >= self.samples_per_chirp:
            print(f"WARNING: Target delay ({delay_samples} samples) exceeds signal length ({self.samples_per_chirp} samples)")
            print(f"Reducing target distance to ensure visibility")
            # Adjust target distance to be visible
            max_visible_distance = (self.samples_per_chirp * self.speed_of_light) / (2 * self.sample_rate)
            target_distance = max_visible_distance * 0.8  # 80% of max visible distance
            delay_seconds = 2 * target_distance / self.speed_of_light
            delay_samples = int(delay_seconds * self.sample_rate)
            print(f"New target distance: {target_distance:.2f} m")
            print(f"New delay: {delay_samples} samples ({delay_seconds*1e6:.2f} μs)")
            
            # Update target info
            target_info['distance'] = target_distance
        
        # For each chirp, add the delayed and phase-shifted version of the TX signal
        for chirp_idx in range(self.num_chirps):
            # Calculate phase shift for this chirp due to Doppler
            # Phase accumulates over time
            phase_shift = 2 * np.pi * doppler_freq * chirp_idx * self.chirp_duration
            
            # IMPROVED APPROACH: Create the delayed signal with phase shift
            delayed_signal = np.zeros(self.samples_per_chirp, dtype=np.complex64)
            
            # Only copy valid samples (avoid index out of bounds)
            samples_to_copy = min(self.samples_per_chirp - delay_samples, self.samples_per_chirp)
            if samples_to_copy > 0:
                # Copy the delayed portion of the TX signal
                delayed_signal[delay_samples:delay_samples+samples_to_copy] = tx_signal[chirp_idx, :samples_to_copy]
                
                # Apply Doppler phase shift and attenuation
                delayed_signal *= attenuation * np.exp(1j * phase_shift)
                
                # Add to all RX channels (with slight variations for realism)
                for rx_idx in range(self.num_rx):
                    # Add small random phase variations between RX channels (for angle estimation)
                    rx_phase_variation = 0.1 * rx_idx  # Small phase difference between RX channels
                    rx_signal[rx_idx, chirp_idx, :] += delayed_signal * np.exp(1j * rx_phase_variation)
        
        # Add a small amount of noise to avoid numerical issues
        noise_power = 1e-6  # Very low noise, just to avoid zeros
        noise = np.random.normal(0, np.sqrt(noise_power/2), rx_signal.shape) + \
                1j * np.random.normal(0, np.sqrt(noise_power/2), rx_signal.shape)
        rx_signal += noise
        
        # Process the received signal to generate range-Doppler map
        rd_map = self.radar_processor.time_to_range_doppler(rx_signal)
        
        if visualize:
            # Visualize the TX signal
            self._visualize_tx_signal(tx_signal, title="DEBUG: TX Signal", 
                                    save_path=f"{self.save_path}/debug_tx_signal{IMG_FORMAT}")
            
            # Visualize the RX signal
            self._visualize_rx_signal(rx_signal, target_info=[target_info], 
                                    title="DEBUG: RX Signal with Ideal Target", 
                                    save_path=f"{self.save_path}/debug_rx_signal{IMG_FORMAT}")
            
            # Visualize the range-Doppler map
            self._visualize_range_doppler_map(rd_map, [target_info], 
                                            title="DEBUG: Range-Doppler Map with Ideal Target", 
                                            save_path=f"{self.save_path}/debug_rd_map{IMG_FORMAT}")
            
            # Calculate expected bin positions
            range_bin = int(target_distance / self.max_range * self.num_range_bins)
            velocity_bin = int((target_velocity + self.max_velocity) / (2 * self.max_velocity) * self.num_doppler_bins)
            
            print(f"Expected target position in range-Doppler map:")
            print(f"  - Range bin: {range_bin} (of {self.num_range_bins})")
            print(f"  - Velocity bin: {velocity_bin} (of {self.num_doppler_bins})")
            
            # Find actual peak in range-Doppler map
            magnitude = np.abs(rd_map[0] + 1j * rd_map[1])
            peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            peak_velocity_bin, peak_range_bin = peak_idx
            
            print(f"Actual peak position in range-Doppler map:")
            print(f"  - Range bin: {peak_range_bin} (of {self.num_range_bins})")
            print(f"  - Velocity bin: {peak_velocity_bin} (of {self.num_doppler_bins})")
            
            # Print peak magnitude for debugging
            peak_magnitude = magnitude[peak_velocity_bin, peak_range_bin]
            print(f"Peak magnitude: {peak_magnitude:.6f}")
            
            # Print magnitude at expected position
            expected_magnitude = magnitude[velocity_bin, range_bin]
            print(f"Magnitude at expected position: {expected_magnitude:.6f}")
            
            # Print statistics about the range-Doppler map
            print(f"Range-Doppler map statistics:")
            print(f"  - Min magnitude: {np.min(magnitude):.6f}")
            print(f"  - Max magnitude: {np.max(magnitude):.6f}")
            print(f"  - Mean magnitude: {np.mean(magnitude):.6f}")
            print(f"  - Median magnitude: {np.median(magnitude):.6f}")
            
            # Process radar data with CFAR detection
            results = self.radar_processor.process_radar_data(
                rx_signal, 
                apply_cfar=True,
                guard_cells=(2, 2),
                training_cells=(4, 4),
                pfa=1e-4
            )
            
            # Access CFAR results
            cfar_map = results['cfar_map']
            detected_targets = results['detected_targets']
            
            # Visualize with CFAR detections
            self._visualize_range_doppler_map(
                rd_map, 
                [target_info], 
                title="DEBUG: Range-Doppler Map with CFAR", 
                save_path=f"{self.save_path}/debug_rd_map_cfar{IMG_FORMAT}",
                cfar_map=cfar_map
            )
            
            print(f"CFAR detection results:")
            print(f"  - Number of detected targets: {len(detected_targets)}")
            for i, target in enumerate(detected_targets):
                print(f"  - Target {i+1}: Range bin {target['range_idx']}, Velocity bin {target['doppler_idx']}")
        
        return rx_signal, rd_map, target_info

    def debug_time_to_range_doppler(self, rx_signal, visualize=True):
        """
        Debug version of time_to_range_doppler function that shows intermediate steps
        and provides detailed visualization of the processing pipeline.
        
        Args:
            rx_signal: Received signal with shape [num_rx, num_chirps, samples_per_chirp]
            visualize: Whether to visualize intermediate steps
            
        Returns:
            Range-Doppler map and dictionary of intermediate results
        """
        print("=== DEBUG: Range-Doppler Processing Pipeline ===")
        
        # Store intermediate results
        debug_info = {}
        
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp = rx_signal.shape
        print(f"Input signal shape: {rx_signal.shape}")
        
        # Step 1: Check for NaN or Inf values in input
        has_nan = np.isnan(rx_signal).any()
        has_inf = np.isinf(rx_signal).any()
        print(f"Input contains NaN: {has_nan}, Inf: {has_inf}")
        if has_nan or has_inf:
            print("WARNING: Input signal contains NaN or Inf values!")
            # Replace with zeros to continue processing
            rx_signal = np.nan_to_num(rx_signal)
        
        # Step 2: Calculate signal power statistics
        signal_power = np.mean(np.abs(rx_signal)**2)
        signal_min = np.min(np.abs(rx_signal))
        signal_max = np.max(np.abs(rx_signal))
        print(f"Signal power: {signal_power:.6e}")
        print(f"Signal magnitude range: {signal_min:.6e} to {signal_max:.6e}")
        debug_info['signal_power'] = signal_power
        
        # Step 3: Apply windowing to reduce sidelobes
        # Use Hanning window for range dimension
        range_window = np.hanning(samples_per_chirp)
        # Use Hanning window for Doppler dimension
        doppler_window = np.hanning(num_chirps)
        
        # Apply windows
        windowed_signal = np.zeros_like(rx_signal)
        for rx_idx in range(num_rx):
            for chirp_idx in range(num_chirps):
                windowed_signal[rx_idx, chirp_idx, :] = rx_signal[rx_idx, chirp_idx, :] * range_window
            
            # Apply Doppler window across chirps
            for sample_idx in range(samples_per_chirp):
                windowed_signal[rx_idx, :, sample_idx] = windowed_signal[rx_idx, :, sample_idx] * doppler_window
        
        # Calculate windowed signal power
        windowed_power = np.mean(np.abs(windowed_signal)**2)
        print(f"Windowed signal power: {windowed_power:.6e}")
        debug_info['windowed_power'] = windowed_power
        
        # Step 4: Perform Range FFT (first dimension)
        range_fft = np.zeros((num_rx, num_chirps, samples_per_chirp), dtype=np.complex64)
        for rx_idx in range(num_rx):
            for chirp_idx in range(num_chirps):
                range_fft[rx_idx, chirp_idx, :] = np.fft.fft(windowed_signal[rx_idx, chirp_idx, :])
        
        # Calculate range FFT power
        range_fft_power = np.mean(np.abs(range_fft)**2)
        print(f"Range FFT power: {range_fft_power:.6e}")
        debug_info['range_fft_power'] = range_fft_power
        
        # Step 5: Perform Doppler FFT (second dimension)
        doppler_fft = np.zeros((num_rx, num_chirps, samples_per_chirp), dtype=np.complex64)
        for rx_idx in range(num_rx):
            for range_bin in range(samples_per_chirp):
                doppler_fft[rx_idx, :, range_bin] = np.fft.fft(range_fft[rx_idx, :, range_bin])
        
        # Calculate Doppler FFT power
        doppler_fft_power = np.mean(np.abs(doppler_fft)**2)
        print(f"Doppler FFT power: {doppler_fft_power:.6e}")
        debug_info['doppler_fft_power'] = doppler_fft_power
        
        # Step 6: Extract the desired number of range and Doppler bins
        # For range, we typically use the first half of FFT output (positive frequencies)
        # For Doppler, we need to rearrange to have zero Doppler in the center
        
        # Determine how many range bins to keep
        range_bins_to_keep = min(self.num_range_bins, samples_per_chirp)
        
        # Determine how many Doppler bins to keep
        doppler_bins_to_keep = min(self.num_doppler_bins, num_chirps)
        
        # Initialize the range-Doppler map
        rd_map = np.zeros((2, doppler_bins_to_keep, range_bins_to_keep), dtype=np.float32)
        
        # Extract and rearrange Doppler bins
        for rx_idx in range(num_rx):
            for range_idx in range(range_bins_to_keep):
                # Rearrange Doppler bins to have zero Doppler in the center
                doppler_data = np.fft.fftshift(doppler_fft[rx_idx, :, range_idx])
                
                # Extract the desired number of Doppler bins
                start_idx = (num_chirps - doppler_bins_to_keep) // 2
                end_idx = start_idx + doppler_bins_to_keep
                doppler_data = doppler_data[start_idx:end_idx]
                
                # Accumulate real and imaginary parts separately across RX channels
                rd_map[0, :, range_idx] += np.real(doppler_data)
                rd_map[1, :, range_idx] += np.imag(doppler_data)
        
        # Normalize by number of RX channels
        rd_map /= num_rx
        
        # Calculate final RD map power
        rd_map_power = np.mean(rd_map[0]**2 + rd_map[1]**2)
        print(f"RD map power: {rd_map_power:.6e}")
        debug_info['rd_map_power'] = rd_map_power
        
        # Step 7: Check for NaN or Inf values in output
        has_nan_output = np.isnan(rd_map).any()
        has_inf_output = np.isinf(rd_map).any()
        print(f"Output contains NaN: {has_nan_output}, Inf: {has_inf_output}")
        if has_nan_output or has_inf_output:
            print("WARNING: Output RD map contains NaN or Inf values!")
            # Replace with zeros
            rd_map = np.nan_to_num(rd_map)
        
        # Step 8: Calculate dynamic range of RD map
        magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        min_val = np.min(magnitude)
        max_val = np.max(magnitude)
        dynamic_range_db = 20 * np.log10(max_val / (min_val + 1e-10))
        print(f"RD map dynamic range: {dynamic_range_db:.2f} dB")
        debug_info['dynamic_range_db'] = dynamic_range_db
        
        # Step 9: Calculate noise floor
        # Sort magnitude values and take the median of the lower half as noise floor
        sorted_magnitude = np.sort(magnitude.flatten())
        noise_floor = np.median(sorted_magnitude[:len(sorted_magnitude)//2])
        print(f"Estimated noise floor: {noise_floor:.6e}")
        debug_info['noise_floor'] = noise_floor
        
        # Step 10: Calculate peak-to-noise ratio
        peak_value = np.max(magnitude)
        peak_to_noise_db = 20 * np.log10(peak_value / (noise_floor + 1e-10))
        print(f"Peak-to-noise ratio: {peak_to_noise_db:.2f} dB")
        debug_info['peak_to_noise_db'] = peak_to_noise_db
        
        if visualize:
            # Visualize intermediate steps
            self._visualize_debug_steps(rx_signal, windowed_signal, range_fft, doppler_fft, rd_map, debug_info)
        
        return rd_map, debug_info

    def _visualize_debug_steps(self, rx_signal, windowed_signal, range_fft, doppler_fft, rd_map, debug_info):
        """
        Visualize intermediate steps in the range-Doppler processing pipeline
        
        Args:
            rx_signal: Original received signal
            windowed_signal: Signal after windowing
            range_fft: Signal after range FFT
            doppler_fft: Signal after Doppler FFT
            rd_map: Final range-Doppler map
            debug_info: Dictionary with debug information
        """
        # Create directory for debug visualizations
        debug_dir = os.path.join(self.save_path, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 1. Visualize original signal (first RX, first chirp)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.real(rx_signal[0, 0, :]))
        plt.title("Original Signal (First RX, First Chirp)")
        plt.ylabel("Real Part")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(np.imag(rx_signal[0, 0, :]))
        plt.ylabel("Imaginary Part")
        plt.xlabel("Sample Index")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_original_signal{IMG_FORMAT}"))
        plt.close()
        
        # 2. Visualize windowed signal
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.real(windowed_signal[0, 0, :]))
        plt.title("Windowed Signal (First RX, First Chirp)")
        plt.ylabel("Real Part")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(np.imag(windowed_signal[0, 0, :]))
        plt.ylabel("Imaginary Part")
        plt.xlabel("Sample Index")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_windowed_signal{IMG_FORMAT}"))
        plt.close()
        
        # 3. Visualize range FFT magnitude (first RX, first chirp)
        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(range_fft[0, 0, :]))
        plt.title("Range FFT Magnitude (First RX, First Chirp)")
        plt.xlabel("Range Bin")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_range_fft{IMG_FORMAT}"))
        plt.close()
        
        # 4. Visualize range FFT magnitude in dB
        plt.figure(figsize=(10, 6))
        range_fft_db = 20 * np.log10(np.abs(range_fft[0, 0, :]) + 1e-10)
        plt.plot(range_fft_db)
        plt.title("Range FFT Magnitude (dB)")
        plt.xlabel("Range Bin")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_range_fft_db{IMG_FORMAT}"))
        plt.close()
        
        # 5. Visualize Doppler FFT magnitude for a specific range bin
        range_bin = np.argmax(np.abs(range_fft[0, 0, :]))  # Use the strongest range bin
        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(doppler_fft[0, :, range_bin]))
        plt.title(f"Doppler FFT Magnitude (Range Bin {range_bin})")
        plt.xlabel("Doppler Bin")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_doppler_fft{IMG_FORMAT}"))
        plt.close()
        
        # 6. Visualize final range-Doppler map
        plt.figure(figsize=(12, 8))
        magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        plt.imshow(magnitude, aspect='auto', cmap='viridis', 
                extent=[0, self.max_range, -self.max_velocity, self.max_velocity])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Range-Doppler Map')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_rd_map{IMG_FORMAT}"))
        plt.close()
        
        # 7. Visualize range-Doppler map in dB scale
        plt.figure(figsize=(12, 8))
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        vmin = np.max(magnitude_db) - debug_info['dynamic_range_db']  # Set min to dynamic range below max
        plt.imshow(magnitude_db, aspect='auto', cmap='viridis', 
                extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                vmin=vmin)
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Range-Doppler Map (dB scale)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_rd_map_db{IMG_FORMAT}"))
        plt.close()
        
        # 8. Visualize 3D surface plot of range-Doppler map
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D plot
        x = np.linspace(0, self.max_range, magnitude.shape[1])
        y = np.linspace(-self.max_velocity, self.max_velocity, magnitude.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, magnitude, cmap='viridis', linewidth=0, antialiased=True)
        
        # Add labels and colorbar
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_zlabel('Magnitude')
        ax.set_title('3D Range-Doppler Map')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_rd_map_3d{IMG_FORMAT}"))
        plt.close()
        
        # 9. Create a summary plot with key metrics
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.text(0.1, 0.9, "Range-Doppler Processing Debug Summary", fontsize=16, weight='bold')
        plt.text(0.1, 0.8, f"Signal Power: {debug_info['signal_power']:.6e}", fontsize=12)
        plt.text(0.1, 0.75, f"Windowed Power: {debug_info['windowed_power']:.6e}", fontsize=12)
        plt.text(0.1, 0.7, f"Range FFT Power: {debug_info['range_fft_power']:.6e}", fontsize=12)
        plt.text(0.1, 0.65, f"Doppler FFT Power: {debug_info['doppler_fft_power']:.6e}", fontsize=12)
        plt.text(0.1, 0.6, f"RD Map Power: {debug_info['rd_map_power']:.6e}", fontsize=12)
        plt.text(0.1, 0.55, f"Dynamic Range: {debug_info['dynamic_range_db']:.2f} dB", fontsize=12)
        plt.text(0.1, 0.5, f"Noise Floor: {debug_info['noise_floor']:.6e}", fontsize=12)
        plt.text(0.1, 0.45, f"Peak-to-Noise Ratio: {debug_info['peak_to_noise_db']:.2f} dB", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"debug_summary{IMG_FORMAT}"))
        plt.close()
        
        print(f"Debug visualizations saved to {debug_dir}")


    def _visualize_debug_comparison(self, rd_map, target_info, expected_pos, actual_pos, 
                                title="Debug Comparison", save_path=None):
        """
        Visualize the range-Doppler map with markers for expected and actual target positions
        
        Args:
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
            target_info: List of target dictionaries
            expected_pos: Tuple of (range_bin, velocity_bin) for expected position
            actual_pos: Tuple of (range_bin, velocity_bin) for actual peak position
            title: Plot title
            save_path: Path to save the figure
        """
        # Convert complex RD map to magnitude
        magnitude = np.abs(rd_map[0] + 1j * rd_map[1])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot the range-Doppler map
        plt.imshow(magnitude, aspect='auto', cmap='viridis', 
                extent=[0, self.max_range, -self.max_velocity, self.max_velocity])
        
        # Mark expected position with a red circle
        expected_range_bin, expected_velocity_bin = expected_pos
        expected_range = expected_range_bin / self.num_range_bins * self.max_range
        expected_velocity = (expected_velocity_bin / self.num_doppler_bins * 2 * self.max_velocity) - self.max_velocity
        plt.scatter(expected_range, expected_velocity, color='red', s=100, marker='o', 
                label=f'Expected: ({expected_range:.1f}m, {expected_velocity:.1f}m/s)')
        
        # Mark actual peak position with a green cross
        actual_range_bin, actual_velocity_bin = actual_pos
        actual_range = actual_range_bin / self.num_range_bins * self.max_range
        actual_velocity = (actual_velocity_bin / self.num_doppler_bins * 2 * self.max_velocity) - self.max_velocity
        plt.scatter(actual_range, actual_velocity, color='green', s=100, marker='x', 
                label=f'Actual: ({actual_range:.1f}m, {actual_velocity:.1f}m/s)')
        
        # Add target ground truth markers
        for i, target in enumerate(target_info):
            plt.scatter(target['distance'], target['velocity'], color='white', s=80, marker='+',
                    label=f'Ground Truth: ({target["distance"]:.1f}m, {target["velocity"]:.1f}m/s)')
        
        # Add labels and title
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title(title)
        plt.colorbar(label='Magnitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved debug comparison to {save_path}")
        
        plt.close()

    def _simulate_radar_channel(self, rx_signal, tx_signal, targets=None, 
                           add_crosstalk=True, add_ground_clutter=True, add_system_noise=True,
                           crosstalk_isolation_db=30, crosstalk_delay_samples=5,
                           system_noise_power=1e-6, clutter_probability=0.3,
                           clutter_distance_range=(2, 10), clutter_rcs_range=(0.01, 0.05)):
        """
        Simulate radar channel effects including targets, crosstalk, ground clutter, and system noise
        
        Args:
            rx_signal: Received signal array with shape [num_rx, num_chirps, samples_per_chirp]
            tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp]
            targets: List of target dictionaries, each with 'distance', 'velocity', and 'rcs' keys
                    If None, no targets will be simulated
            add_crosstalk: Whether to add TX-RX crosstalk
            add_ground_clutter: Whether to add ground clutter
            add_system_noise: Whether to add system noise
            crosstalk_isolation_db: Isolation between TX and RX in dB
            crosstalk_delay_samples: Delay for crosstalk in samples
            system_noise_power: Power of system noise
            clutter_probability: Probability of adding ground clutter
            clutter_distance_range: Range of distances for ground clutter (min, max)
            clutter_rcs_range: Range of RCS values for ground clutter (min, max)
            
        Returns:
            Updated received signal with all channel effects
        """
        # Add targets if provided
        if targets is not None and len(targets) > 0:
            for target in targets:
                distance = target['distance']
                velocity = target['velocity']
                rcs = target['rcs']
                rx_signal = self._add_target_reflection(rx_signal, tx_signal, distance, velocity, rcs)
        else:
            # If no targets, add some environmental reflections to ensure signal content
            # This simulates reflections from the environment that are always present
            # Add a few weak reflections at random distances
            num_env_reflections = random.randint(2, 5)
            for _ in range(num_env_reflections):
                env_distance = random.uniform(5, self.max_range * 0.8)
                env_velocity = random.uniform(-0.5, 0.5)  # Very low velocity
                env_rcs = random.uniform(0.01, 0.1)  # Very small RCS
                rx_signal = self._add_target_reflection(rx_signal, tx_signal, env_distance, env_velocity, env_rcs)
        
        # Add system noise if enabled
        if add_system_noise:
            for rx_idx in range(self.num_rx):
                for chirp_idx in range(self.num_chirps):
                    noise_real = np.random.normal(0, np.sqrt(system_noise_power/2), self.samples_per_chirp)
                    noise_imag = np.random.normal(0, np.sqrt(system_noise_power/2), self.samples_per_chirp)
                    rx_signal[rx_idx, chirp_idx] += noise_real + 1j * noise_imag
        
        # Add crosstalk if enabled
        if add_crosstalk:
            attenuation = 10**(-crosstalk_isolation_db/20)
            for rx_idx in range(self.num_rx):
                for chirp_idx in range(self.num_chirps):
                    if crosstalk_delay_samples < self.samples_per_chirp:
                        rx_signal[rx_idx, chirp_idx, crosstalk_delay_samples:] += attenuation * tx_signal[chirp_idx, :self.samples_per_chirp-crosstalk_delay_samples]
        
        # Add ground clutter if enabled
        if add_ground_clutter:
            # Add random ground clutter with specified probability
            if random.random() < clutter_probability:
                # Generate 1-3 clutter points
                num_clutter_points = random.randint(1, 3)
                for _ in range(num_clutter_points):
                    clutter_distance = random.uniform(*clutter_distance_range)
                    clutter_rcs = random.uniform(*clutter_rcs_range)
                    # Ground clutter has zero velocity
                    rx_signal = self._add_target_reflection(rx_signal, tx_signal, clutter_distance, 0, clutter_rcs)
        
        return rx_signal

    def _add_target_reflection(self, rx_signal, tx_signal, distance, velocity, rcs=1.0, attenuation_scale=1e3):
        """
        Add a target reflection to the received signal
        
        Args:
            rx_signal: Received signal array with shape [num_rx, num_chirps, samples_per_chirp]
            tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp]
            distance: Distance to target in meters
            velocity: Velocity of target in m/s
            rcs: Radar Cross Section of target
            
        Returns:
            Updated received signal with target reflection added
        """
        # Calculate round-trip time delay for the target
        round_trip_time = 2 * distance / self.speed_of_light
        
        # Calculate delay in samples
        delay_samples = int(round_trip_time * self.sample_rate)
        
        # Calculate Doppler shift due to target velocity
        # Doppler frequency shift = 2 * velocity / wavelength
        doppler_freq = 2 * velocity / self.wavelength
        
        # Calculate phase shift per chirp due to Doppler
        # Phase shift = 2π * doppler_freq * chirp_duration
        doppler_phase_per_chirp = 2 * np.pi * doppler_freq * self.chirp_duration
        
        # Calculate signal attenuation due to distance and RCS
        # Using radar equation: P_r = P_t * G^2 * λ^2 * σ / ((4π)^3 * R^4)
        # We simplify this to focus on the R^4 dependency and RCS (σ)
        attenuation = np.sqrt(rcs) / (distance ** 2)  # Amplitude scales with sqrt(power)
        
        # Apply a scaling factor to get reasonable signal levels
        attenuation *= attenuation_scale #1e-2  # Adjust this based on your simulation needs
        
        # Add the target to each RX antenna and each chirp
        for rx_idx in range(self.num_rx):
            for chirp_idx in range(self.num_chirps):
                # Calculate total phase for this chirp (including Doppler)
                doppler_phase = doppler_phase_per_chirp * chirp_idx
                
                # Apply Doppler shift to the reflected signal
                # For each chirp, we need to apply the appropriate Doppler phase
                reflected_signal = tx_signal[chirp_idx] * np.exp(1j * doppler_phase)
                
                # Add the delayed and attenuated signal to the received signal
                if delay_samples < self.samples_per_chirp:
                    rx_signal[rx_idx, chirp_idx, delay_samples:] += attenuation * reflected_signal[:self.samples_per_chirp-delay_samples]
        
        return rx_signal

    def _randomize_signal_parameters(self):
        """Randomize signal parameters for more diverse dataset"""
        # Randomly select signal type from available options
        signal_types = ['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW']
        self.signal_type = random.choice(signal_types)
        
        # Randomize bandwidth within reasonable limits
        if self.signal_type in ['FMCW', 'OFDM']:
            self.bandwidth = random.uniform(300e6, 500e6)
        
        # Randomize chirp duration
        self.chirp_duration = random.uniform(0.8e-4, 1.2e-4)
        
        # Randomize number of subcarriers for OFDM
        if self.signal_type in ['OFDM', 'OFDM_FMCW']:
            self.num_subcarriers = random.randint(64, 128)
        
        # Randomize signal frequency for Sine
        if self.signal_type in ['Sine', 'Sine_FMCW']:
            self.signal_freq = random.uniform(0.5e6, 2e6)
        
        # Update radar processor with new parameters
        self.radar_processor = RadarProcessing(
            num_range_bins=self.num_range_bins,
            num_doppler_bins=self.num_doppler_bins,
            sample_rate=self.sample_rate,
            chirp_duration=self.chirp_duration,
            num_chirps=self.num_chirps,
            num_subcarriers=self.num_subcarriers if hasattr(self, 'num_subcarriers') else 128,
            subcarrier_spacing=self.subcarrier_spacing if hasattr(self, 'subcarrier_spacing') else 30e3,
            bandwidth=self.bandwidth,
            transceiver_bandwidth=self.transceiver_bandwidth,
            transceiver_center_freq=self.transceiver_center_freq,
            output_freq=self.center_freq,
            signal_type=self.signal_type,
            signal_freq=self.signal_freq if hasattr(self, 'signal_freq') else 1e6
        )

    def _visualize_tx_signal(self, tx_signal, title="Transmit Signal Visualization", save_path=None):
        """Visualize the transmit signal and its frequency spectrum
        
        Args:
            tx_signal: Transmit signal array [num_chirps, samples_per_chirp]
            title: Title for the plot
            save_path: Path to save the figure (if None, just display)
        """
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Select first chirp for visualization
        chirp_idx = 0
        
        # Time domain plot
        t = np.linspace(0, self.chirp_duration, self.samples_per_chirp, endpoint=False)
        
        ax1.plot(t * 1e6, np.real(tx_signal[chirp_idx]), label='Real')
        ax1.plot(t * 1e6, np.imag(tx_signal[chirp_idx]), label='Imaginary')
        ax1.set_title(f'Time Domain TX Signal (Chirp {chirp_idx})\nChirp Duration: {self.chirp_duration*1e6:.2f} μs')
        ax1.set_xlabel('Time (μs)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Calculate and plot instantaneous frequency
        # For a chirp, we can extract the instantaneous frequency by taking the derivative of the phase
        phase = np.unwrap(np.angle(tx_signal[chirp_idx]))
        # Calculate the derivative of the phase
        inst_freq = np.diff(phase) / (2 * np.pi * (t[1] - t[0]))
        
        # Plot instantaneous frequency
        ax2.plot(t[1:] * 1e6, inst_freq / 1e6, 'r-')
        ax2.set_title('Instantaneous Frequency (Shows Chirp Characteristic)')
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('Frequency (MHz)')
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal lines showing the bandwidth limits
        ax2.axhline(y=-self.bandwidth/2e6, color='g', linestyle='--', 
                    label=f'Start: {-self.bandwidth/2e6:.1f} MHz')
        ax2.axhline(y=self.bandwidth/2e6, color='b', linestyle='--', 
                    label=f'End: {self.bandwidth/2e6:.1f} MHz')
        ax2.legend()
        
        # Frequency domain plot
        freq = np.fft.fftshift(np.fft.fftfreq(len(tx_signal[chirp_idx]), 1/self.sample_rate))
        
        fft_signal = np.fft.fftshift(np.fft.fft(tx_signal[chirp_idx]))
        fft_mag = np.abs(fft_signal)
        fft_db = 20 * np.log10(fft_mag + 1e-10)
        
        ax3.plot(freq / 1e6, fft_db)
        ax3.set_title('Frequency Domain TX Signal')
        ax3.set_xlabel('Frequency (MHz)')
        ax3.set_ylabel('Magnitude (dB)')
        ax3.grid(True, alpha=0.3)
        
        # Mark the bandwidth region
        bw_start = -self.bandwidth/2
        bw_end = self.bandwidth/2
        ax3.axvspan(bw_start/1e6, bw_end/1e6, alpha=0.2, color='green', 
                    label=f'Signal Bandwidth ({self.bandwidth/1e6:.0f} MHz)')
        ax3.legend()
        
        # Set x-axis limits to focus on the relevant frequency range
        ax3.set_xlim([-self.bandwidth/1e6, self.bandwidth/1e6])  # Focus on the bandwidth region
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def _visualize_rx_signal(self, rx_signal, target_info=None, title="Received Signal Visualization", save_path=None):
        """Visualize the received signal to check correctness
        
        Args:
            rx_signal: Received signal array [num_rx, num_chirps, samples_per_chirp]
            target_info: Optional list of dictionaries with target information
            title: Title for the plot
            save_path: Path to save the figure (if None, just display)
        """
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # Select first RX and chirp for visualization
        rx_idx = 0
        chirp_idx = 0
        
        # Plot time domain signal (real part)
        axs[0, 0].plot(np.real(rx_signal[rx_idx, chirp_idx]))
        axs[0, 0].set_title(f'Time Domain Signal (Real Part) - RX {rx_idx}, Chirp {chirp_idx}')
        axs[0, 0].set_xlabel('Sample')
        axs[0, 0].set_ylabel('Amplitude')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot time domain signal (imaginary part)
        axs[0, 1].plot(np.imag(rx_signal[rx_idx, chirp_idx]))
        axs[0, 1].set_title(f'Time Domain Signal (Imaginary Part) - RX {rx_idx}, Chirp {chirp_idx}')
        axs[0, 1].set_xlabel('Sample')
        axs[0, 1].set_ylabel('Amplitude')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot frequency domain representation
        freq = np.fft.fftshift(np.fft.fftfreq(len(rx_signal[rx_idx, chirp_idx]), 1/self.sample_rate))
        freq = freq + self.center_freq  # Shift to center frequency
        freq_ghz = freq / 1e9  # Convert to GHz
        
        fft_signal = np.fft.fftshift(np.fft.fft(rx_signal[rx_idx, chirp_idx]))
        fft_mag = np.abs(fft_signal)
        fft_db = 20 * np.log10(fft_mag + 1e-10)
        
        axs[1, 0].plot(freq_ghz, fft_db)
        axs[1, 0].set_title(f'Frequency Domain (Magnitude) - RX {rx_idx}, Chirp {chirp_idx}')
        axs[1, 0].set_xlabel('Frequency (GHz)')
        axs[1, 0].set_ylabel('Magnitude (dB)')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Mark the bandwidth region
        bw_start = self.center_freq - self.bandwidth/2
        bw_end = self.center_freq + self.bandwidth/2
        axs[1, 0].axvspan(bw_start/1e9, bw_end/1e9, alpha=0.2, color='green', 
                        label=f'Signal Bandwidth ({self.bandwidth/1e6:.0f} MHz)')
        axs[1, 0].legend()
        
        # Set x-axis limits to focus on the relevant frequency range
        axs[1, 0].set_xlim([self.center_freq/1e9 - 1, self.center_freq/1e9 + 1])  # ±1 GHz around center
        
        # Plot phase of the signal
        phase = np.angle(rx_signal[rx_idx, chirp_idx])
        axs[1, 1].plot(phase)
        axs[1, 1].set_title(f'Signal Phase - RX {rx_idx}, Chirp {chirp_idx}')
        axs[1, 1].set_xlabel('Sample')
        axs[1, 1].set_ylabel('Phase (radians)')
        axs[1, 1].grid(True, alpha=0.3)
        
        # Plot spectrogram to visualize frequency change over time (for FMCW)
                # Plot spectrogram to visualize frequency change over time (for FMCW)
        if self.signal_type == 'FMCW':
            try:
                # Calculate spectrogram using scipy.signal instead of plt.mlab
                from scipy import signal
                f, t, Sxx = signal.spectrogram(rx_signal[rx_idx, chirp_idx], 
                                              fs=self.sample_rate,
                                              nperseg=128,
                                              noverlap=64,
                                              scaling='spectrum')
                
                # Convert to dB
                Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
                
                # Plot spectrogram
                im = axs[2, 0].pcolormesh(t, f/1e6, Sxx_db, shading='gouraud', cmap='viridis')
                axs[2, 0].set_title('Spectrogram (Frequency vs Time)')
                axs[2, 0].set_xlabel('Time (s)')
                axs[2, 0].set_ylabel('Frequency (MHz)')
                plt.colorbar(im, ax=axs[2, 0], label='Power (dB)')
            except Exception as e:
                # Fallback if spectrogram calculation fails
                axs[2, 0].text(0.5, 0.5, f"Spectrogram calculation failed:\n{str(e)}", 
                              ha='center', va='center', transform=axs[2, 0].transAxes)
                axs[2, 0].set_title('Spectrogram (Error)')
        
        # Plot multiple chirps to see Doppler effect
        if rx_signal.shape[1] > 1:  # If we have multiple chirps
            # Plot first 5 chirps or all if less than 5
            num_chirps_to_plot = min(5, rx_signal.shape[1])
            for i in range(num_chirps_to_plot):
                axs[2, 1].plot(np.real(rx_signal[rx_idx, i]), 
                              label=f'Chirp {i}', 
                              alpha=0.7)
            
            axs[2, 1].set_title('Multiple Chirps Comparison (Real Part)')
            axs[2, 1].set_xlabel('Sample')
            axs[2, 1].set_ylabel('Amplitude')
            axs[2, 1].legend()
            axs[2, 1].grid(True, alpha=0.3)
        
        # Add target information if provided
        if target_info:
            target_text = "Target Information:\n"
            for i, target in enumerate(target_info):
                target_text += f"Target {i+1}: Distance={target['distance']:.1f}m, "
                target_text += f"Velocity={target['velocity']:.1f}m/s, RCS={target['rcs']:.2f}\n"
            
            fig.text(0.5, 0.01, target_text, ha='center', 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Add overall title
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save or show the figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def _visualize_range_doppler_map(self, rd_map, target_info, title="Range-Doppler Map", save_path=None, cfar_map=None):
        """
        Visualize range-Doppler map and optionally CFAR detection results
        
        Args:
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins] (2, 16, 128)
            target_info: List of target dictionaries
            title: Title of the plot
            save_path: Path to save the plot
            cfar_map: Optional CFAR detection map to display alongside the range-Doppler map
        """
        # Compute magnitude of complex range-Doppler map
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2) #(16, 128)
        
        # Apply logarithmic scaling to better visualize targets
        # Add a small constant to avoid log(0)
        rd_magnitude_db = 20 * np.log10(rd_magnitude + 1e-10)
        
        # Normalize to 0-1 range for better visualization
        rd_magnitude_norm = (rd_magnitude_db - np.min(rd_magnitude_db)) / (np.max(rd_magnitude_db) - np.min(rd_magnitude_db) + 1e-10)
        
        # Determine if we need to show CFAR map alongside
        if cfar_map is not None:
            # Create figure with two subplots
            plt.figure(figsize=(12, 5))
            
            # Plot Range-Doppler Map
            plt.subplot(121)
            plt.imshow(rd_magnitude_db, aspect='auto', origin='lower', 
                    extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                    cmap='viridis')
            plt.title('Range-Doppler Map (dB)')
            plt.xlabel('Range (m)')
            plt.ylabel('Velocity (m/s)')
            plt.colorbar(label='Magnitude (dB)')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Mark targets on the RD map if available
            if target_info:
                for target in target_info:
                    distance = target['distance']
                    velocity = target['velocity']
                    # Mark target with a red circle
                    plt.plot(distance, velocity, 'ro', markersize=8, markeredgecolor='white')
            
            # Plot CFAR Detection Map
            plt.subplot(122)
            plt.imshow(cfar_map, aspect='auto', origin='lower',
                    extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                    cmap='hot')
            plt.title('CFAR Detection')
            plt.xlabel('Range (m)')
            plt.ylabel('Velocity (m/s)')
            plt.colorbar(label='Detection')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Mark targets on the CFAR map if available
            if target_info:
                for target in target_info:
                    distance = target['distance']
                    velocity = target['velocity']
                    # Mark target with a red circle
                    plt.plot(distance, velocity, 'ro', markersize=8, markeredgecolor='white')
            
            plt.tight_layout()
        else:
            # Create single plot for just the range-Doppler map
            plt.figure(figsize=(10, 8))
            
            # Create range and velocity axes
            range_axis = np.linspace(0, self.max_range, self.num_range_bins)
            velocity_axis = np.linspace(-self.max_velocity, self.max_velocity, self.num_doppler_bins)
            
            # Plot range-Doppler map with improved colormap
            plt.imshow(rd_magnitude_norm, aspect='auto', origin='lower', 
                    extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                    cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Normalized Magnitude (dB)')
            
            # Mark targets on the plot
            if target_info:
                for target in target_info:
                    distance = target['distance']
                    velocity = target['velocity']
                    # Mark target with a red circle and add annotation
                    plt.plot(distance, velocity, 'ro', markersize=10, markeredgecolor='white')
                    plt.annotate(f"Target\nD: {distance:.1f}m\nV: {velocity:.1f}m/s", 
                                (distance, velocity), 
                                xytext=(10, 10), 
                                textcoords='offset points',
                                color='white',
                                bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7))
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Set labels and title
            plt.xlabel('Range (m)')
            plt.ylabel('Velocity (m/s)')
            plt.title(title)
            
            # Tight layout
            plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _visualize_range_doppler_map3d(self, rd_map, target_info=None, title="Range-Doppler Map Visualization", save_path=None):
        """
        Visualize the range-Doppler map with target annotations in both 2D and 3D
        
        Args:
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
            target_info: List of dictionaries containing target information
            title: Title for the plot
            save_path: Path to save the figure
            
        Returns:
            None
        """
        # Create figure with two subplots (2D and 3D)
        fig = plt.figure(figsize=(18, 10))
        
        # Get magnitude of range-Doppler map
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        # Convert to dB scale with clipping
        rd_db = 20 * np.log10(rd_magnitude / np.max(rd_magnitude) + 1e-10)
        rd_db_clipped = np.clip(rd_db, -40, 0)
        
        # Create normalized version for display
        rd_normalized = (rd_db_clipped + 40) / 40
        
        # Create range and velocity axes
        range_axis = np.arange(self.num_range_bins) * self.range_resolution
        velocity_axis = (np.arange(self.num_doppler_bins) - self.num_doppler_bins // 2) * self.velocity_resolution
        
        # 2D Plot (left subplot)
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(rd_normalized, aspect='auto', origin='lower', 
                  extent=[0, range_axis[-1], velocity_axis[0], velocity_axis[-1]],
                  cmap='viridis')
        
        # Add colorbar to 2D plot
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Normalized Power (dB)')
        
        # Add target annotations to 2D plot if provided
        if target_info is not None and len(target_info) > 0:
            for i, target in enumerate(target_info):
                distance = target['distance']
                velocity = target['velocity']
                rcs = target.get('rcs', 1.0)
                
                # Calculate marker size based on RCS
                marker_size = 50 * np.sqrt(rcs)
                
                # Plot target
                ax1.scatter(distance, velocity, s=marker_size, c='red', 
                           marker='x', linewidths=2, label=f'Target {i+1}' if i == 0 else None)
                
                # Add target label
                ax1.annotate(f'T{i+1}', (distance, velocity), 
                            xytext=(5, 5), textcoords='offset points',
                            color='white', fontweight='bold')
        
        # Add plot labels and title for 2D plot
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('2D Range-Doppler Map')
        
        # Add grid to 2D plot
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend if targets are present
        if target_info is not None and len(target_info) > 0:
            ax1.legend(loc='upper right')
        
        # 3D Plot (right subplot)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(range_axis, velocity_axis)
        
        # Plot the 3D surface
        surf = ax2.plot_surface(X, Y, rd_normalized, cmap='plasma', 
                               linewidth=0, antialiased=True, alpha=0.8)
        
        # Add colorbar to 3D plot
        cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.6)
        cbar2.set_label('Normalized Power (dB)')
        
        # Add target markers to 3D plot if provided
        if target_info is not None and len(target_info) > 0:
            for i, target in enumerate(target_info):
                distance = target['distance']
                velocity = target['velocity']
                
                # Find the closest indices to the target location
                range_idx = int(distance / self.range_resolution)
                velocity_idx = int((velocity / self.velocity_resolution) + self.num_doppler_bins // 2)
                
                # Ensure indices are within bounds
                range_idx = min(max(0, range_idx), self.num_range_bins - 1)
                velocity_idx = min(max(0, velocity_idx), self.num_doppler_bins - 1)
                
                # Get the power value at target location
                z_value = rd_normalized[velocity_idx, range_idx]
                
                # Plot target in 3D with a small offset for visibility
                ax2.scatter([distance], [velocity], [z_value + 0.05], 
                           c='red', marker='o', s=80, label=f'Target {i+1}' if i == 0 else None)
                
                # Add target label in 3D
                ax2.text(distance, velocity, z_value + 0.1, f'T{i+1}', 
                        color='white', fontweight='bold', fontsize=10)
        
        # Set labels and title for 3D plot
        ax2.set_xlabel('Range (m)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_zlabel('Normalized Power')
        ax2.set_title('3D Range-Doppler Surface')
        
        # Adjust 3D view angle for better visualization
        ax2.view_init(elev=30, azim=225)
        
        # Add radar parameters as text
        param_text = f"Signal: {self.signal_type}\n"
        param_text += f"Bandwidth: {self.bandwidth/1e6:.1f} MHz\n"
        param_text += f"Range Res: {self.range_resolution:.2f} m\n"
        param_text += f"Velocity Res: {self.velocity_resolution:.2f} m/s"
        
        plt.figtext(0.02, 0.02, param_text, fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Set main title for the entire figure
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if path is provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _visualize_detection_comparison(self, rd_map, detected_targets, target_mask, target_info, 
                                   title="Detection Comparison", save_path=None):
        """
        Visualize detected targets and ground truth target mask together for comparison
        
        Args:
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
            detected_targets: List of detected target dictionaries from CFAR
            target_mask: Ground truth target mask with shape [num_doppler_bins, num_range_bins, 1]
            target_info: List of ground truth target dictionaries
            title: Title of the plot
            save_path: Path to save the plot
        """
        # Compute magnitude of complex range-Doppler map
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        # Apply logarithmic scaling
        rd_magnitude_db = 20 * np.log10(rd_magnitude + 1e-10)
        
        # Normalize to 0-1 range for better visualization
        rd_magnitude_norm = (rd_magnitude_db - np.min(rd_magnitude_db)) / (np.max(rd_magnitude_db) - np.min(rd_magnitude_db) + 1e-10)
        
        # Create figure with two subplots
        plt.figure(figsize=(12, 5))
        
        # Plot Range-Doppler Map with ground truth targets
        plt.subplot(121)
        plt.imshow(rd_magnitude_db, aspect='auto', origin='lower', 
                extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                cmap='viridis')
        plt.title('Ground Truth Targets')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.colorbar(label='Magnitude (dB)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Overlay target mask as contour
        target_mask_2d = target_mask[:, :, 0]
        if np.max(target_mask_2d) > 0:  # Only if there are targets
            plt.contour(target_mask_2d, levels=[0.5], 
                    extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                    colors='r', linewidths=2)
        
        # Mark ground truth targets
        if target_info:
            for target in target_info:
                distance = target['distance']
                velocity = target['velocity']
                plt.plot(distance, velocity, 'ro', markersize=8, markeredgecolor='white')
                plt.annotate(f"GT\nD: {distance:.1f}m\nV: {velocity:.1f}m/s", 
                            (distance, velocity), 
                            xytext=(10, 10), 
                            textcoords='offset points',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7))
        
        # Plot Range-Doppler Map with detected targets
        plt.subplot(122)
        plt.imshow(rd_magnitude_db, aspect='auto', origin='lower', 
                extent=[0, self.max_range, -self.max_velocity, self.max_velocity],
                cmap='viridis')
        plt.title('CFAR Detected Targets')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.colorbar(label='Magnitude (dB)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Mark detected targets
        if detected_targets:
            for target in detected_targets:
                distance = target['range']
                velocity = target['velocity']
                plt.plot(distance, velocity, 'go', markersize=8, markeredgecolor='white')
                plt.annotate(f"DET\nD: {distance:.1f}m\nV: {velocity:.1f}m/s", 
                            (distance, velocity), 
                            xytext=(10, 10), 
                            textcoords='offset points',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.7))
        
        # Add title and adjust layout
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _add_target_old(self, rx_signal, distance, velocity, rcs=1.0, tx_signal=None):
        """
        Add a target to the received signal
        
        Args:
            rx_signal: Received signal array with shape [num_rx, num_chirps, samples_per_chirp]
            distance: Distance to target in meters
            velocity: Velocity of target in m/s
            rcs: Radar Cross Section of target
            tx_signal: Transmitted signal to use for target reflection
            
        Returns:
            Updated received signal with target added
        """
        # If tx_signal is not provided, use the last generated tx_signal
        if tx_signal is None:
            # This is not ideal - we should always pass the tx_signal
            print("Warning: tx_signal not provided to _add_target. Using default signal.")
            tx_signal = self._generate_tx_signal()
        
        # Calculate round-trip time delay for the target
        round_trip_time = 2 * distance / self.speed_of_light
        
        # Calculate delay in samples
        delay_samples = int(round_trip_time * self.sample_rate)
        
        # Calculate Doppler shift due to target velocity
        # Doppler frequency shift = 2 * velocity / wavelength
        doppler_freq = 2 * velocity / self.wavelength
        
        # Calculate phase shift per chirp due to Doppler
        # Phase shift = 2π * doppler_freq * chirp_duration
        doppler_phase_per_chirp = 2 * np.pi * doppler_freq * self.chirp_duration
        
        # Calculate signal attenuation due to distance and RCS
        # Using radar equation: P_r = P_t * G^2 * λ^2 * σ / ((4π)^3 * R^4)
        # We simplify this to focus on the R^4 dependency and RCS (σ)
        attenuation = np.sqrt(rcs) / (distance ** 2)  # Amplitude scales with sqrt(power)
        
        # Apply a scaling factor to get reasonable signal levels
        attenuation *= 1e-2  # Adjust this based on your simulation needs
        
        # Add the target to each RX antenna and each chirp
        for rx_idx in range(self.num_rx):
            for chirp_idx in range(self.num_chirps):
                # Calculate total phase for this chirp (including Doppler)
                doppler_phase = doppler_phase_per_chirp * chirp_idx
                
                # Apply Doppler shift to the reflected signal
                # For each chirp, we need to apply the appropriate Doppler phase
                reflected_signal = tx_signal[chirp_idx] * np.exp(1j * doppler_phase)
                
                # Add the delayed and attenuated signal to the received signal
                if delay_samples < self.samples_per_chirp:
                    rx_signal[rx_idx, chirp_idx, delay_samples:] += attenuation * reflected_signal[:self.samples_per_chirp-delay_samples]
        
        return rx_signal
    
    def _generate_ofdm_signal(self, num_chirps, samples_per_chirp, center_freq, bandwidth, 
                         num_subcarriers, subcarrier_spacing=None, t=None):
        """
        Generate OFDM signal for radar sensing
        
        Args:
            num_chirps: Number of OFDM symbols (chirps)
            samples_per_chirp: Number of samples per chirp
            center_freq: Center frequency in Hz
            bandwidth: Signal bandwidth in Hz
            num_subcarriers: Number of subcarriers
            subcarrier_spacing: Spacing between subcarriers in Hz (optional)
            t: Time vector (optional)
            
        Returns:
            Complex OFDM signal with shape [num_chirps, samples_per_chirp]
        """
        # Calculate subcarrier spacing if not provided
        if subcarrier_spacing is None:
            subcarrier_spacing = bandwidth / num_subcarriers
        
        # Calculate FFT size (typically power of 2)
        fft_size = 2**int(np.ceil(np.log2(num_subcarriers)))
        
        # Calculate symbol duration (without CP)
        symbol_duration = 1 / subcarrier_spacing
        
        # Calculate cyclic prefix length (typically 20-25% of symbol)
        cp_length = int(0.25 * fft_size)
        
        # Calculate total symbol length (with CP)
        total_symbol_length = fft_size + cp_length
        
        # Create time vector if not provided
        if t is None:
            t = np.arange(samples_per_chirp) / self.sample_rate
        
        # Initialize output signal
        ofdm_signal = np.zeros((num_chirps, samples_per_chirp), dtype=np.complex64)
        
        # Generate OFDM symbols (chirps)
        for chirp_idx in range(num_chirps):
            # Generate random data for subcarriers
            # For radar, we can use known symbols for better channel estimation
            data_symbols = np.exp(1j * 2 * np.pi * np.random.random(num_subcarriers))
            
            # Place subcarriers in the frequency domain
            # Center the subcarriers around DC
            freq_domain = np.zeros(fft_size, dtype=np.complex64)
            start_idx = (fft_size - num_subcarriers) // 2
            freq_domain[start_idx:start_idx + num_subcarriers] = data_symbols
            
            # Convert to time domain using IFFT
            time_domain = np.fft.ifft(freq_domain) * np.sqrt(fft_size)
            
            # Add cyclic prefix
            symbol_with_cp = np.concatenate([time_domain[-cp_length:], time_domain])
            
            # Apply windowing to reduce spectral leakage
            window = np.hamming(total_symbol_length)
            windowed_symbol = symbol_with_cp * window
            
            # Ensure the symbol fits within the chirp duration
            if total_symbol_length <= samples_per_chirp:
                ofdm_signal[chirp_idx, :total_symbol_length] = windowed_symbol
            else:
                # If the symbol is too long, truncate it
                ofdm_signal[chirp_idx, :] = windowed_symbol[:samples_per_chirp]
            
            # Apply frequency shift to center the signal at center_freq
            freq_shift = np.exp(1j * 2 * np.pi * center_freq * t[:samples_per_chirp])
            ofdm_signal[chirp_idx, :] *= freq_shift
        
        return ofdm_signal

    def _generate_tx_signal(self, tx_power=1.0):
        """Generate the transmit signal based on signal type
        
        Args:
            tx_power: Transmit power scaling factor (default: 1.0)
            
        Returns:
            Transmit signal array [num_chirps, samples_per_chirp]
        """
        # Create time vector for one chirp
        t = np.arange(0, self.chirp_duration, 1/self.sample_rate) 
        self.samples_per_chirp = len(t)#150 points for one chirp

        # Update samples_per_chirp only if it's the first time (during initialization)
        if not hasattr(self, '_tx_signal_initialized'):
            self.samples_per_chirp = len(t)
            self._tx_signal_initialized = True
        
        # Initialize transmit signal array
        tx_signal = np.zeros((self.num_chirps, self.samples_per_chirp), dtype=np.complex64)
        #(32, 150)
        #For basic FMCW: generates the chirp signal directly at the center frequency; 
        # Generate time vector with exact number of samples
        
        if self.signal_type == 'FMCW':
            # Generate time vector with exact number of samples
            t = np.linspace(0, self.chirp_duration, self.samples_per_chirp, endpoint=False)
            
            # For FMCW radar, we need to generate a baseband chirp that sweeps across the full bandwidth
            # The baseband chirp should sweep from -bandwidth/2 to +bandwidth/2
            start_freq = -self.bandwidth/2
            end_freq = self.bandwidth/2
            
            # Calculate chirp rate (Hz/s)
            chirp_rate = self.bandwidth / self.chirp_duration
            
            print(f"FMCW Chirp Parameters:")
            print(f"  Center Frequency: {self.center_freq/1e9:.2f} GHz")
            print(f"  Bandwidth: {self.bandwidth/1e6:.1f} MHz")
            print(f"  Chirp Duration: {self.chirp_duration*1e6:.2f} μs")
            print(f"  Chirp Rate: {chirp_rate/1e12:.3f} THz/s")
            print(f"  Samples per chirp: {self.samples_per_chirp}")
                
            # Generate FMCW chirp signal for each chirp
            for chirp_idx in range(self.num_chirps):
                # Generate linear frequency sweep
                # The phase is the integral of 2π*f(t): φ(t) = 2π*(start_freq*t + (chirp_rate/2)*t²)
                phase = 2 * np.pi * (start_freq * t + (chirp_rate/2) * t**2)
                
                # Apply a small random phase offset between chirps for more realistic simulation
                chirp_phase_offset = np.random.uniform(0, 2*np.pi)
                
                # Generate the complex baseband signal: A*exp(j*φ(t))
                tx_signal[chirp_idx] = tx_power * np.exp(1j * (phase + chirp_phase_offset))
            
            # Print diagnostic information to verify bandwidth utilization
            # Calculate instantaneous frequency of first chirp to verify sweep
            if self.samples_per_chirp > 1:
                phase = np.unwrap(np.angle(tx_signal[0]))
                inst_freq = np.diff(phase) / (2 * np.pi * (t[1] - t[0]))
                print(f"  Instantaneous frequency range: {np.min(inst_freq)/1e6:.1f} MHz to {np.max(inst_freq)/1e6:.1f} MHz")
                print(f"  Measured bandwidth: {(np.max(inst_freq) - np.min(inst_freq))/1e6:.1f} MHz")
                
        elif self.signal_type == 'OFDM':
            # Generate standard OFDM signal using the dedicated function
            tx_signal = self._generate_ofdm_signal(
                num_chirps=self.num_chirps,
                samples_per_chirp=self.samples_per_chirp,
                center_freq=self.center_freq,
                bandwidth=self.bandwidth,
                num_subcarriers=self.num_subcarriers,
                t=t
            )
            
        elif self.signal_type == 'Sine':
            # Generate sine wave signal
            for chirp_idx in range(self.num_chirps):
                # Simple sine wave at center frequency
                sine_signal = np.exp(1j * 2 * np.pi * self.center_freq * t)
                tx_signal[chirp_idx] = sine_signal
                
        elif self.signal_type == 'OFDM_FMCW' or self.signal_type == 'Sine_FMCW':
            # Two-step process to simulate AD9361 + CN0566 hardware setup
            
            # Step 1: Generate baseband signal at AD9361 frequency (2.1GHz)
            baseband_signal = np.zeros((self.num_chirps, self.samples_per_chirp), dtype=np.complex64)
            
            if self.signal_type == 'OFDM_FMCW':
                # Generate OFDM baseband signal using the dedicated function
                baseband_signal = self._generate_ofdm_signal(
                    num_chirps=self.num_chirps,
                    samples_per_chirp=self.samples_per_chirp,
                    center_freq=self.transceiver_center_freq,
                    bandwidth=self.transceiver_bandwidth,
                    num_subcarriers=self.num_subcarriers,
                    t=t
                )
                    
            elif self.signal_type == 'Sine_FMCW':
                # Generate Sine baseband signal
                for chirp_idx in range(self.num_chirps):
                    # Simple sine wave at signal_freq
                    sine_phase = 2 * np.pi * self.signal_freq * t
                    baseband_signal[chirp_idx] = np.exp(1j * sine_phase)
            
            # Step 2: Use the radar processor to apply hardware modulation
            # This simulates the CN0566 frequency sweep
            # Reshape to match the expected input format [num_rx=1, num_chirps, samples_per_chirp]
            baseband_reshaped = baseband_signal.reshape(1, self.num_chirps, self.samples_per_chirp)
            modulated_signal = self.radar_processor.simulate_hardware_modulation(
                baseband_reshaped, 
                signal_type=self.signal_type
            )
            
            # Extract the modulated signal (remove the dummy RX dimension)
            tx_signal = modulated_signal[0]
            
        else:
            # Default to FMCW if unknown signal type
            print(f"Warning: Unknown signal type '{self.signal_type}'. Using FMCW instead.")
            for chirp_idx in range(self.num_chirps):
                k = self.bandwidth / self.chirp_duration
                f0 = self.center_freq - self.bandwidth/2
                phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
                tx_signal[chirp_idx] = np.exp(1j * phase)
        
        # Apply the transmit power scaling
        tx_signal *= tx_power
        
        return tx_signal
    
    def _apply_delay_to_tx_signal(self, tx_signal, time_delay, doppler_phase, chirp_idx, rcs):
        """Apply delay, attenuation, and phase shift to transmit signal
        
        Args:
            tx_signal: Transmit signal array [num_chirps, samples_per_chirp]
            time_delay: Time delay in seconds
            doppler_phase: Phase shift due to Doppler
            chirp_idx: Current chirp index
            rcs: Target radar cross section (normalized)
            
        Returns:
            Delayed and attenuated signal
        """
        # Calculate delay in samples
        delay_samples = int(time_delay * self.sample_rate)
        
        # Get the transmit signal for this chirp
        chirp_signal = tx_signal[chirp_idx]
        
        # Create delayed signal (zero-padded)
        delayed_signal = np.zeros_like(chirp_signal)
        
        if delay_samples < len(chirp_signal):
            # Copy the delayed portion of the signal
            delayed_signal[delay_samples:] = chirp_signal[:len(chirp_signal)-delay_samples]
        
        # Apply attenuation based on distance (R^4 loss) and RCS
        # We use the time_delay which is proportional to distance
        attenuation = np.sqrt(rcs) / (time_delay**2)  # Simplified radar equation
        
        # Apply attenuation and Doppler phase shift
        delayed_signal = attenuation * delayed_signal * np.exp(1j * doppler_phase)
        
        return delayed_signal

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

    def visualize_sample(self, idx):
        """
        Public wrapper for _draw_sample that visualizes a sample with its range-Doppler map,
        time domain signal, and target information
        
        Args:
            idx: Sample index to visualize
        """
        # Get the data for this sample
        if self.use_lazy_loading:
            rx_signal = self._get_time_domain_data(idx)
            rd_map = self._get_range_doppler_map(idx)
        else:
            rx_signal = self.time_domain_data[idx]
            rd_map = self.range_doppler_maps[idx]
        
        # Get target information
        targets = self.target_info[idx] if hasattr(self, 'target_info') and idx < len(self.target_info) else []
        
        # Call the comprehensive visualization function
        self._draw_sample(idx, rx_signal, rd_map, targets)

    def _draw_sample(self, index, rx_signal, rd_map, targets):
        """
        Draw comprehensive visualization of a sample using existing visualization functions
        
        Args:
            index: Sample index
            rx_signal: Received signal
            rd_map: Range-Doppler map
            targets: List of target information
            
        Returns:
            None
        """
        # Create a directory for this sample's visualizations
        sample_dir = os.path.join(self.save_path, f"sample_{index}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Generate a TX signal for visualization (since we might not have it stored)
        tx_signal = self._generate_tx_signal()
        
        # Create a comprehensive figure title
        title = f"Sample {index}: {len(targets)} Targets, {self.signal_type}"
        
        # Visualize TX signal
        self._visualize_tx_signal(
            tx_signal, 
            title=f"TX Signal - {title}", 
            save_path=f"{sample_dir}/tx_signal{IMG_FORMAT}"
        )
        
        # Visualize RX signal
        self._visualize_rx_signal(
            rx_signal, 
            target_info=targets, 
            title=f"RX Signal - {title}", 
            save_path=f"{sample_dir}/rx_signal{IMG_FORMAT}"
        )
        
        # Visualize Range-Doppler map
        self._visualize_range_doppler_map(
            rd_map, 
            target_info=targets, 
            title=f"Range-Doppler Map - {title}", 
            save_path=f"{sample_dir}/rd_map{IMG_FORMAT}"
        )
        
        # Also create a comprehensive visualization that combines all three
        self._visualize_comprehensive_signal(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            title=title,
            target_info=targets,
            save_path=f"{sample_dir}/comprehensive{IMG_FORMAT}"
        )
        
        print(f"Sample {index} visualizations saved to {sample_dir}")

    def _visualize_comprehensive_signal(self, tx_signal, rx_signal, title="Signal Visualization", target_info=None, save_path=None):
        """Comprehensive visualization focusing on unique insights not covered by other visualization functions
        
        Args:
            tx_signal: Transmitted signal
            rx_signal: Received signal
            title: Title for the visualization
            target_info: List of dictionaries containing target information
            save_path: Path to save the figure
            
        Returns:
            None
        """
        # Create a figure with multiple subplots for unique visualizations
        fig = plt.figure(figsize=(15, 12))
        
        # Create a grid layout for our specialized plots
        gs = fig.add_gridspec(3, 2)
        
        # 1. Signal Spectrum Analysis (not in other visualizations)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Calculate spectrum of TX signal (first chirp)
        tx_spectrum = np.fft.fftshift(np.fft.fft(tx_signal[0]))
        freq = np.fft.fftshift(np.fft.fftfreq(len(tx_spectrum), 1/self.sample_rate))
        
        # Plot TX spectrum
        ax1.plot(freq/1e6, 20*np.log10(np.abs(tx_spectrum) + 1e-10), 'b-', label='TX Spectrum')
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Power (dB)')
        ax1.set_title('Signal Spectrum Analysis')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Phase Analysis (not in other visualizations)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate phase of RX signal for first antenna, first chirp
        rx_phase = np.angle(rx_signal[0, 0])
        time_axis = np.arange(len(rx_phase)) / self.sample_rate * 1e6  # in μs
        
        # Plot phase
        ax2.plot(time_axis, rx_phase, 'r-')
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('Phase (rad)')
        ax2.set_title('RX Signal Phase Analysis')
        ax2.grid(True)
        
        # 3. Multi-Antenna Correlation Analysis (unique visualization)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if self.num_rx > 1:
            # Calculate correlation between antennas
            corr_matrix = np.zeros((self.num_rx, self.num_rx))
            for i in range(self.num_rx):
                for j in range(self.num_rx):
                    # Correlation for first chirp
                    corr = np.abs(np.corrcoef(
                        np.abs(rx_signal[i, 0]), 
                        np.abs(rx_signal[j, 0])
                    )[0, 1])
                    corr_matrix[i, j] = corr
            
            # Plot correlation matrix
            im = ax3.imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1)
            ax3.set_title('RX Antenna Correlation')
            ax3.set_xlabel('Antenna Index')
            ax3.set_ylabel('Antenna Index')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Correlation')
            
            # Add antenna indices
            for i in range(self.num_rx):
                for j in range(self.num_rx):
                    ax3.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                            ha='center', va='center', color='w' if corr_matrix[i, j] < 0.7 else 'k')
        else:
            ax3.text(0.5, 0.5, 'Multiple antennas required for correlation analysis', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Doppler Profile Analysis (unique visualization)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Process the received signal to generate range-Doppler map if not already provided
        rd_map = self.radar_processor.time_to_range_doppler(rx_signal)
        
        # Get magnitude of range-Doppler map
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        # Sum across range bins to get Doppler profile
        doppler_profile = np.sum(rd_magnitude, axis=1)
        doppler_profile = doppler_profile / np.max(doppler_profile)
        
        # Create velocity axis
        velocity_axis = (np.arange(self.num_doppler_bins) - self.num_doppler_bins // 2) * self.velocity_resolution
        
        # Plot Doppler profile
        ax4.plot(velocity_axis, doppler_profile, 'g-', linewidth=2)
        ax4.set_xlabel('Velocity (m/s)')
        ax4.set_ylabel('Normalized Power')
        ax4.set_title('Doppler Profile Analysis')
        ax4.grid(True)
        
        # Add target markers if available
        if target_info is not None and len(target_info) > 0:
            for i, target in enumerate(target_info):
                velocity = target['velocity']
                # Find closest velocity bin
                vel_idx = int((velocity / self.velocity_resolution) + self.num_doppler_bins // 2)
                if 0 <= vel_idx < self.num_doppler_bins:
                    ax4.axvline(x=velocity, color='r', linestyle='--', alpha=0.7)
                    ax4.text(velocity, 0.8, f'T{i+1}', color='r', fontweight='bold')
        
        # 5. Range Profile Analysis (unique visualization)
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Sum across Doppler bins to get range profile
        range_profile = np.sum(rd_magnitude, axis=0)
        range_profile = range_profile / np.max(range_profile)
        
        # Create range axis
        range_axis = np.arange(self.num_range_bins) * self.range_resolution
        
        # Plot range profile
        ax5.plot(range_axis, range_profile, 'b-', linewidth=2)
        ax5.set_xlabel('Range (m)')
        ax5.set_ylabel('Normalized Power')
        ax5.set_title('Range Profile Analysis')
        ax5.grid(True)
        
        # Add target markers if available
        if target_info is not None and len(target_info) > 0:
            for i, target in enumerate(target_info):
                distance = target['distance']
                # Find closest range bin
                range_idx = int(distance / self.range_resolution)
                if 0 <= range_idx < self.num_range_bins:
                    ax5.axvline(x=distance, color='r', linestyle='--', alpha=0.7)
                    ax5.text(distance, 0.8, f'T{i+1}', color='r', fontweight='bold')
        
        # 6. Signal-to-Noise Ratio Analysis (unique visualization)
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Calculate SNR across chirps for first RX antenna
        signal_power = np.zeros(self.num_chirps)
        noise_power = np.zeros(self.num_chirps)
        
        for chirp_idx in range(self.num_chirps):
            # Get chirp data
            chirp_data = np.abs(rx_signal[0, chirp_idx])
            
            # Estimate noise floor (using lower 10% of samples)
            sorted_data = np.sort(chirp_data)
            noise_floor = np.mean(sorted_data[:int(len(sorted_data)*0.1)])
            
            # Calculate signal power (peak)
            signal_peak = np.max(chirp_data)
            
            # Store powers
            signal_power[chirp_idx] = signal_peak
            noise_power[chirp_idx] = noise_floor
        
        # Calculate SNR in dB
        snr_db = 20 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Plot SNR
        chirp_axis = np.arange(self.num_chirps)
        ax6.plot(chirp_axis, snr_db, 'r-o')
        ax6.set_xlabel('Chirp Index')
        ax6.set_ylabel('SNR (dB)')
        ax6.set_title('Signal-to-Noise Ratio Analysis')
        ax6.grid(True)
        
        # Add radar parameters as text
        param_text = f"Signal Type: {self.signal_type}\n"
        param_text += f"Bandwidth: {self.bandwidth/1e6:.1f} MHz\n"
        param_text += f"Sample Rate: {self.sample_rate/1e6:.1f} MHz\n"
        param_text += f"Chirp Duration: {self.chirp_duration*1e6:.1f} μs\n"
        param_text += f"Range Resolution: {self.range_resolution:.2f} m\n"
        param_text += f"Velocity Resolution: {self.velocity_resolution:.2f} m/s\n"
        param_text += f"Number of Targets: {len(target_info) if target_info else 0}"
        
        plt.figtext(0.02, 0.02, param_text, fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Set main title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or show figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _save_hdf5(self):
        """Save dataset to HDF5 format with improved organization for large datasets"""
        print(f"Saving dataset to {self.save_path}/hdf5")
        
        # Create directory
        os.makedirs(f"{self.save_path}/hdf5", exist_ok=True)
        
        # Define maximum samples per file to avoid memory issues
        max_samples_per_file = 500
        num_files = (self.num_samples + max_samples_per_file - 1) // max_samples_per_file
        
        # Close any existing open file handles
        if hasattr(self, 'h5_files') and self.h5_files:
            for f in self.h5_files:
                try:
                    f.close()
                except Exception as e:
                    print(f"Warning when closing existing file: {e}")
        
        # Create a parameters file
        params_file_path = f"{self.save_path}/hdf5/parameters.h5"
        with h5py.File(params_file_path, 'w') as f:
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
            params.attrs['apply_realistic_effects'] = self.apply_realistic_effects
            params.attrs['num_files'] = num_files
            params.attrs['max_samples_per_file'] = max_samples_per_file
            
            # Store additional parameters if they exist
            if hasattr(self, 'num_subcarriers'):
                params.attrs['num_subcarriers'] = self.num_subcarriers
            if hasattr(self, 'subcarrier_spacing'):
                params.attrs['subcarrier_spacing'] = self.subcarrier_spacing
            if hasattr(self, 'transceiver_bandwidth'):
                params.attrs['transceiver_bandwidth'] = self.transceiver_bandwidth
            if hasattr(self, 'transceiver_center_freq'):
                params.attrs['transceiver_center_freq'] = self.transceiver_center_freq
            if hasattr(self, 'signal_freq'):
                params.attrs['signal_freq'] = self.signal_freq
        
        # Save data in chunks
        for file_idx in range(num_files):
            start_idx = file_idx * max_samples_per_file
            end_idx = min((file_idx + 1) * max_samples_per_file, self.num_samples)
            current_samples = end_idx - start_idx
            
            print(f"Saving samples {start_idx} to {end_idx-1} to file {file_idx+1}/{num_files}")
            
            # Create a file for this chunk
            chunk_file_path = f"{self.save_path}/hdf5/chunk_{file_idx}.h5"
            
            try:
                with h5py.File(chunk_file_path, 'w') as f:
                    # Create a group for metadata
                    meta = f.create_group('metadata')
                    meta.attrs['start_idx'] = start_idx
                    meta.attrs['end_idx'] = end_idx
                    meta.attrs['num_samples'] = current_samples
                    
                    # Create datasets for this chunk
                    f.create_dataset('time_domain_data', 
                                    data=self.time_domain_data[start_idx:end_idx],
                                    dtype=self.precision, compression='gzip')
                    f.create_dataset('range_doppler_maps', 
                                    data=self.range_doppler_maps[start_idx:end_idx],
                                    dtype=self.precision, compression='gzip')
                    f.create_dataset('target_masks', 
                                    data=self.target_masks[start_idx:end_idx],
                                    dtype=self.precision, compression='gzip')
                    
                    # Create a group for samples
                    samples_group = f.create_group('samples')
                    
                    # For each sample in this chunk, create a group with all related data
                    for i in range(start_idx, end_idx):
                        sample_idx = i - start_idx
                        sample_group = samples_group.create_group(f'{sample_idx}')
                        
                        # Store target information
                        target_group = sample_group.create_group('target_info')
                        for j, target in enumerate(self.target_info[i]):
                            target_subgroup = target_group.create_group(f'{j}')
                            for key, value in target.items():
                                target_subgroup.attrs[key] = value
                        
                        # Store TX signal if available
                        if hasattr(self, 'tx_signals') and i < len(self.tx_signals):
                            tx_group = sample_group.create_group('tx_signal')
                            tx_group.create_dataset('real', 
                                                  data=np.real(self.tx_signals[i]),
                                                  dtype=self.precision, compression='gzip')
                            tx_group.create_dataset('imag', 
                                                  data=np.imag(self.tx_signals[i]),
                                                  dtype=self.precision, compression='gzip')
                            
                            # Store TX power if available
                            if hasattr(self, 'tx_powers') and i < len(self.tx_powers):
                                tx_group.attrs['power'] = self.tx_powers[i]
                            
                            # Store signal type if available
                            if hasattr(self, 'signal_types') and i < len(self.signal_types):
                                tx_group.attrs['signal_type'] = self.signal_types[i]
                        
                        # Store SNR value if available
                        if hasattr(self, 'snr_values') and i < len(self.snr_values):
                            sample_group.attrs['snr'] = self.snr_values[i]
                        
                        # Store channel model if available
                        if hasattr(self, 'channel_models') and i < len(self.channel_models):
                            channel_group = sample_group.create_group('channel_model')
                            for key, value in self.channel_models[i].items():
                                if value is not None:
                                    channel_group.attrs[key] = value
                
                print(f"Successfully saved chunk {file_idx+1}/{num_files} to {chunk_file_path}")
            except Exception as e:
                print(f"Error saving HDF5 file {chunk_file_path}: {e}")
        
        print(f"Successfully saved dataset to {self.save_path}/hdf5 in {num_files} files")

    def _save_numpy(self):
        """Save dataset to NumPy format with improved organization for large datasets"""
        print(f"Saving dataset to {self.save_path}/numpy")
        
        # Create directory
        os.makedirs(f"{self.save_path}/numpy", exist_ok=True)
        
        # Define maximum samples per file to avoid memory issues
        max_samples_per_file = 500
        num_files = (self.num_samples + max_samples_per_file - 1) // max_samples_per_file
        
        # Save parameters first (common to all files)
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
            'max_velocity': self.max_velocity,
            'apply_realistic_effects': self.apply_realistic_effects,
            'num_files': num_files,
            'max_samples_per_file': max_samples_per_file
        }
        
        # Add additional parameters if they exist
        if hasattr(self, 'num_subcarriers'):
            params['num_subcarriers'] = self.num_subcarriers
        if hasattr(self, 'subcarrier_spacing'):
            params['subcarrier_spacing'] = self.subcarrier_spacing
        if hasattr(self, 'transceiver_bandwidth'):
            params['transceiver_bandwidth'] = self.transceiver_bandwidth
        if hasattr(self, 'transceiver_center_freq'):
            params['transceiver_center_freq'] = self.transceiver_center_freq
        if hasattr(self, 'signal_freq'):
            params['signal_freq'] = self.signal_freq
            
        np.save(f"{self.save_path}/numpy/parameters.npy", params)
        
        # Save data in chunks
        for file_idx in range(num_files):
            start_idx = file_idx * max_samples_per_file
            end_idx = min((file_idx + 1) * max_samples_per_file, self.num_samples)
            current_samples = end_idx - start_idx
            
            print(f"Saving samples {start_idx} to {end_idx-1} to file {file_idx+1}/{num_files}")
            
            # Create a directory for this chunk
            chunk_dir = f"{self.save_path}/numpy/chunk_{file_idx}"
            os.makedirs(chunk_dir, exist_ok=True)
            
            # For each sample in this chunk, save all related data in a single file
            for i in range(start_idx, end_idx):
                sample_idx = i - start_idx
                sample_data = {
                    'time_domain_data': self.time_domain_data[i].astype(self.precision),
                    'range_doppler_map': self.range_doppler_maps[i].astype(self.precision),
                    'target_mask': self.target_masks[i].astype(self.precision),
                    'target_info': self.target_info[i],
                }
                
                # Add TX signal data if available
                if hasattr(self, 'tx_signals') and i < len(self.tx_signals):
                    sample_data['tx_signal_real'] = np.real(self.tx_signals[i]).astype(self.precision)
                    sample_data['tx_signal_imag'] = np.imag(self.tx_signals[i]).astype(self.precision)
                
                # Add TX power if available
                if hasattr(self, 'tx_powers') and i < len(self.tx_powers):
                    sample_data['tx_power'] = self.tx_powers[i]
                
                # Add SNR value if available
                if hasattr(self, 'snr_values') and i < len(self.snr_values):
                    sample_data['snr'] = self.snr_values[i]
                
                # Add signal type if available
                if hasattr(self, 'signal_types') and i < len(self.signal_types):
                    sample_data['signal_type'] = self.signal_types[i]
                
                # Add channel model if available
                if hasattr(self, 'channel_models') and i < len(self.channel_models):
                    sample_data['channel_model'] = self.channel_models[i]
                
                # Save the sample data
                np.save(f"{chunk_dir}/sample_{i}.npy", sample_data)
            
            # Also save a consolidated file for this chunk for faster loading
            chunk_data = {
                'time_domain_data': self.time_domain_data[start_idx:end_idx].astype(self.precision),
                'range_doppler_maps': self.range_doppler_maps[start_idx:end_idx].astype(self.precision),
                'target_masks': self.target_masks[start_idx:end_idx].astype(self.precision),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_samples': current_samples
            }
            np.save(f"{chunk_dir}/chunk_data.npy", chunk_data)
        
        print(f"Successfully saved dataset to {self.save_path}/numpy in {num_files} files")

    def _load_data(self, datapath):
        """Load dataset from file
        
        Args:
            datapath: Path to dataset file or directory
        """
        print(f"Loading dataset from {datapath}")
        
        # Check if it's an HDF5 file
        if datapath.endswith('.h5'):
            self._load_hdf5(datapath)
        
        # Check if it's a directory with HDF5 chunks
        elif os.path.isdir(datapath) and os.path.exists(f"{datapath}/hdf5/parameters.h5"):
            self._load_chunked_hdf5(f"{datapath}/hdf5")
        
        # Check if it's a directory with NumPy files (old format)
        elif os.path.isdir(datapath) and os.path.exists(f"{datapath}/time_domain_data.npy"):
            self._load_numpy(datapath)
        
        # Check if it's a directory with NumPy chunks
        elif os.path.isdir(datapath) and os.path.exists(f"{datapath}/numpy/parameters.npy"):
            self._load_chunked_numpy(f"{datapath}/numpy")
        
        else:
            raise ValueError(f"Unsupported data format or path not found: {datapath}")

    def _load_hdf5(self, datapath):
        """Load dataset from a single HDF5 file (old format)
        
        Args:
            datapath: Path to HDF5 file
        """
        if self.use_lazy_loading:
            # Open file for lazy loading
            self.h5_file = h5py.File(datapath, 'r')
            
            # Load parameters
            if 'parameters' in self.h5_file:
                params = self.h5_file['parameters']
            else:
                # For backward compatibility with old format
                params = self.h5_file
                
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
            self._init_radar_processor_from_params()
            
            # Get dataset shapes for __getitem__
            self.time_domain_shape = self.h5_file['time_domain_data'].shape
            self.range_doppler_shape = self.h5_file['range_doppler_maps'].shape
            self.target_mask_shape = self.h5_file['target_masks'].shape
            
            # Load target info
            self.target_info = []
            for i in range(self.num_samples):
                sample_targets = []
                target_path = f'target_info/{i}'
                
                # Handle different formats of target info storage
                if target_path in self.h5_file:
                    target_group = self.h5_file[target_path]
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
                if 'parameters' in f:
                    params = f['parameters']
                else:
                    # For backward compatibility with old format
                    params = f
                    
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
                self._init_radar_processor_from_params()
                
                # Load target info
                self.target_info = []
                for i in range(self.num_samples):
                    sample_targets = []
                    target_path = f'target_info/{i}'
                    
                    # Handle different formats of target info storage
                    if target_path in f:
                        target_group = f[target_path]
                        for j in target_group:
                            target = {}
                            for key, value in target_group[j].attrs.items():
                                target[key] = value
                            sample_targets.append(target)
                    self.target_info.append(sample_targets)
                
                # Load additional data if available
                self._load_additional_data(f)

    def _load_chunked_hdf5(self, datapath):
        """Load dataset from chunked HDF5 files (new format)
        
        Args:
            datapath: Path to directory containing HDF5 chunks
        """
        # Load parameters first
        params_file_path = f"{datapath}/parameters.h5"
        with h5py.File(params_file_path, 'r') as f:
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
            
            # Get chunking information
            self.num_files = params.attrs['num_files']
            self.max_samples_per_file = params.attrs['max_samples_per_file']
            
            # Load additional parameters if they exist
            for attr_name in ['num_subcarriers', 'subcarrier_spacing', 
                             'transceiver_bandwidth', 'transceiver_center_freq', 
                             'signal_freq', 'apply_realistic_effects']:
                if attr_name in params.attrs:
                    setattr(self, attr_name, params.attrs[attr_name])
        
        # Initialize radar processor with loaded parameters
        self._init_radar_processor_from_params()
        
        if self.use_lazy_loading:
            # For lazy loading, we'll keep track of which chunk each sample belongs to
            self.chunk_map = {}  # Maps sample index to (chunk_idx, local_idx)
            self.h5_files = {}   # Maps chunk_idx to open file handle
            
            # Initialize data structures
            self.target_info = [[] for _ in range(self.num_samples)]
            
            # Scan all chunks to build the chunk map and load target info
            for chunk_idx in range(self.num_files):
                chunk_file_path = f"{datapath}/chunk_{chunk_idx}.h5"
                
                if not os.path.exists(chunk_file_path):
                    print(f"Warning: Chunk file {chunk_file_path} not found, skipping")
                    continue
                
                # Open the chunk file
                chunk_file = h5py.File(chunk_file_path, 'r')
                self.h5_files[chunk_idx] = chunk_file
                
                # Get metadata
                meta = chunk_file['metadata']
                start_idx = meta.attrs['start_idx']
                end_idx = meta.attrs['end_idx']
                
                # Build chunk map
                for i in range(start_idx, end_idx):
                    local_idx = i - start_idx
                    self.chunk_map[i] = (chunk_idx, local_idx)
                
                # Load target info for this chunk
                if 'samples' in chunk_file:
                    samples_group = chunk_file['samples']
                    for local_idx in samples_group:
                        i = start_idx + int(local_idx)
                        if i >= self.num_samples:
                            continue
                            
                        sample_group = samples_group[local_idx]
                        if 'target_info' in sample_group:
                            target_group = sample_group['target_info']
                            sample_targets = []
                            for j in target_group:
                                target = {}
                                for key, value in target_group[j].attrs.items():
                                    target[key] = value
                                sample_targets.append(target)
                            self.target_info[i] = sample_targets
            
            # Get shapes from the first chunk
            if self.h5_files and 0 in self.h5_files:
                first_chunk = self.h5_files[0]
                # Get shapes for a single sample
                time_domain_shape_single = first_chunk['time_domain_data'].shape[1:]
                range_doppler_shape_single = first_chunk['range_doppler_maps'].shape[1:]
                target_mask_shape_single = first_chunk['target_masks'].shape[1:]
                
                # Construct full shapes
                self.time_domain_shape = (self.num_samples,) + time_domain_shape_single
                self.range_doppler_shape = (self.num_samples,) + range_doppler_shape_single
                self.target_mask_shape = (self.num_samples,) + target_mask_shape_single
        else:
            # For full loading, we'll load all chunks into memory
            # Initialize arrays
            # First, determine the shapes from the first chunk
            first_chunk_path = f"{datapath}/chunk_0.h5"
            with h5py.File(first_chunk_path, 'r') as f:
                time_domain_shape_single = f['time_domain_data'].shape[1:]
                range_doppler_shape_single = f['range_doppler_maps'].shape[1:]
                target_mask_shape_single = f['target_masks'].shape[1:]
            
            # Create full arrays
            self.time_domain_data = np.zeros((self.num_samples,) + time_domain_shape_single, 
                                           dtype=self.precision)
            self.range_doppler_maps = np.zeros((self.num_samples,) + range_doppler_shape_single, 
                                             dtype=self.precision)
            self.target_masks = np.zeros((self.num_samples,) + target_mask_shape_single, 
                                       dtype=self.precision)
            
            # Initialize target info
            self.target_info = [[] for _ in range(self.num_samples)]
            
            # Additional data structures
            self.tx_signals = [None] * self.num_samples
            self.tx_powers = [None] * self.num_samples
            self.snr_values = [None] * self.num_samples
            self.signal_types = [None] * self.num_samples
            self.channel_models = [None] * self.num_samples
            
            # Load all chunks
            for chunk_idx in range(self.num_files):
                chunk_file_path = f"{datapath}/chunk_{chunk_idx}.h5"
                
                if not os.path.exists(chunk_file_path):
                    print(f"Warning: Chunk file {chunk_file_path} not found, skipping")
                    continue
                
                with h5py.File(chunk_file_path, 'r') as f:
                    # Get metadata
                    meta = f['metadata']
                    start_idx = meta.attrs['start_idx']
                    end_idx = meta.attrs['end_idx']
                    
                    # Load data arrays
                    self.time_domain_data[start_idx:end_idx] = f['time_domain_data'][:]
                    self.range_doppler_maps[start_idx:end_idx] = f['range_doppler_maps'][:]
                    self.target_masks[start_idx:end_idx] = f['target_masks'][:]
                    
                    # Load sample-specific data
                    if 'samples' in f:
                        samples_group = f['samples']
                        for local_idx_str in samples_group:
                            local_idx = int(local_idx_str)
                            i = start_idx + local_idx
                            if i >= self.num_samples:
                                continue
                                
                            sample_group = samples_group[local_idx_str]
                            
                            # Load target info
                            if 'target_info' in sample_group:
                                target_group = sample_group['target_info']
                                sample_targets = []
                                for j in target_group:
                                    target = {}
                                    for key, value in target_group[j].attrs.items():
                                        target[key] = value
                                    sample_targets.append(target)
                                self.target_info[i] = sample_targets
                            
                            # Load TX signal if available
                            if 'tx_signal' in sample_group:
                                tx_group = sample_group['tx_signal']
                                tx_real = tx_group['real'][:]
                                tx_imag = tx_group['imag'][:]
                                self.tx_signals[i] = tx_real + 1j * tx_imag
                                
                                # Load TX power if available
                                if 'power' in tx_group.attrs:
                                    self.tx_powers[i] = tx_group.attrs['power']
                                
                                # Load signal type if available
                                if 'signal_type' in tx_group.attrs:
                                    self.signal_types[i] = tx_group.attrs['signal_type']
                            
                            # Load SNR value if available
                            if 'snr' in sample_group.attrs:
                                self.snr_values[i] = sample_group.attrs['snr']
                            
                            # Load channel model if available
                            if 'channel_model' in sample_group:
                                channel_group = sample_group['channel_model']
                                channel_model = {}
                                for key, value in channel_group.attrs.items():
                                    channel_model[key] = value
                                self.channel_models[i] = channel_model

    def _load_numpy(self, datapath):
        """Load dataset from NumPy files (old format)
        
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
        
        # Load additional parameters if they exist
        for attr_name in ['num_subcarriers', 'subcarrier_spacing', 
                         'transceiver_bandwidth', 'transceiver_center_freq', 
                         'signal_freq', 'apply_realistic_effects']:
            if attr_name in params:
                setattr(self, attr_name, params[attr_name])
        
        # Initialize radar processor with loaded parameters
        self._init_radar_processor_from_params()
        
        # Load additional data if available
        try:
            # Try to load TX signals
            if os.path.exists(f"{datapath}/tx_signals_real.npy") and os.path.exists(f"{datapath}/tx_signals_imag.npy"):
                tx_signals_real = np.load(f"{datapath}/tx_signals_real.npy")
                tx_signals_imag = np.load(f"{datapath}/tx_signals_imag.npy")
                self.tx_signals = [tx_signals_real[i] + 1j * tx_signals_imag[i] for i in range(len(tx_signals_real))]
            
            # Try to load TX powers
            if os.path.exists(f"{datapath}/tx_powers.npy"):
                self.tx_powers = np.load(f"{datapath}/tx_powers.npy")
            
            # Try to load SNR values
            if os.path.exists(f"{datapath}/snr_values.npy"):
                self.snr_values = np.load(f"{datapath}/snr_values.npy")
            
            # Try to load signal types
            if os.path.exists(f"{datapath}/signal_types.npy"):
                self.signal_types = np.load(f"{datapath}/signal_types.npy", allow_pickle=True)
            
            # Try to load channel models
            if os.path.exists(f"{datapath}/channel_models.npy"):
                self.channel_models = np.load(f"{datapath}/channel_models.npy", allow_pickle=True)
        except Exception as e:
            print(f"Warning: Could not load some additional data: {e}")

    def _load_chunked_numpy(self, datapath):
        """Load dataset from chunked NumPy files (new format)
        
        Args:
            datapath: Path to directory containing NumPy chunks
        """
        # Load parameters first
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
        
        # Get chunking information
        self.num_files = params['num_files']
        self.max_samples_per_file = params['max_samples_per_file']
        
        # Load additional parameters if they exist
        for attr_name in ['num_subcarriers', 'subcarrier_spacing', 
                         'transceiver_bandwidth', 'transceiver_center_freq', 
                         'signal_freq', 'apply_realistic_effects']:
            if attr_name in params:
                setattr(self, attr_name, params[attr_name])
        
        # Initialize radar processor with loaded parameters
        self._init_radar_processor_from_params()
        
        if self.use_memory_mapping or self.use_lazy_loading:
            # For lazy loading, we'll keep track of sample paths
            self.sample_paths = {}  # Maps sample index to file path
            
            # Initialize data structures
            self.target_info = [[] for _ in range(self.num_samples)]
            
            # Scan all chunks to build the sample paths and load target info
            for chunk_idx in range(self.num_files):
                chunk_dir = f"{datapath}/chunk_{chunk_idx}"
                
                if not os.path.exists(chunk_dir):
                    print(f"Warning: Chunk directory {chunk_dir} not found, skipping")
                    continue
                
                # Load chunk metadata
                chunk_data = np.load(f"{chunk_dir}/chunk_data.npy", allow_pickle=True).item()
                start_idx = chunk_data['start_idx']
                end_idx = chunk_data['end_idx']
                
                # Build sample paths
                for i in range(start_idx, end_idx):
                    self.sample_paths[i] = f"{chunk_dir}/sample_{i}.npy"
                    
                    # Try to load target info
                    try:
                        sample_data = np.load(self.sample_paths[i], allow_pickle=True).item()
                        if 'target_info' in sample_data:
                            self.target_info[i] = sample_data['target_info']
                    except Exception as e:
                        print(f"Warning: Could not load target info for sample {i}: {e}")
            
            # Get shapes from the first sample
            if self.sample_paths and 0 in self.sample_paths:
                sample_data = np.load(self.sample_paths[0], allow_pickle=True).item()
                
                # Get shapes for a single sample
                time_domain_shape = sample_data['time_domain_data'].shape
                range_doppler_shape = sample_data['range_doppler_map'].shape
                target_mask_shape = sample_data['target_mask'].shape
                
                # Construct full shapes
                self.time_domain_shape = (self.num_samples,) + time_domain_shape
                self.range_doppler_shape = (self.num_samples,) + range_doppler_shape
                self.target_mask_shape = (self.num_samples,) + target_mask_shape
        else:
            # For full loading, we'll load all chunks into memory
            # First, determine the shapes from the first chunk
            first_chunk_dir = f"{datapath}/chunk_0"
            first_chunk_data = np.load(f"{first_chunk_dir}/chunk_data.npy", allow_pickle=True).item()
            
            # Create full arrays
            self.time_domain_data = np.zeros((self.num_samples,) + first_chunk_data['time_domain_data'].shape[1:], 
                                           dtype=self.precision)
            self.range_doppler_maps = np.zeros((self.num_samples,) + first_chunk_data['range_doppler_maps'].shape[1:], 
                                             dtype=self.precision)
            self.target_masks = np.zeros((self.num_samples,) + first_chunk_data['target_masks'].shape[1:], 
                                       dtype=self.precision)
            
            # Initialize target info and additional data structures
            self.target_info = [[] for _ in range(self.num_samples)]
            self.tx_signals = [None] * self.num_samples
            self.tx_powers = [None] * self.num_samples
            self.snr_values = [None] * self.num_samples
            self.signal_types = [None] * self.num_samples
            self.channel_models = [None] * self.num_samples
            
            # Load all chunks
            for chunk_idx in range(self.num_files):
                chunk_dir = f"{datapath}/chunk_{chunk_idx}"
                
                if not os.path.exists(chunk_dir):
                    print(f"Warning: Chunk directory {chunk_dir} not found, skipping")
                    continue
                
                # Load chunk data
                chunk_data = np.load(f"{chunk_dir}/chunk_data.npy", allow_pickle=True).item()
                start_idx = chunk_data['start_idx']
                end_idx = chunk_data['end_idx']
                
                # Load consolidated data
                self.time_domain_data[start_idx:end_idx] = chunk_data['time_domain_data']
                self.range_doppler_maps[start_idx:end_idx] = chunk_data['range_doppler_maps']
                self.target_masks[start_idx:end_idx] = chunk_data['target_masks']
                
                # Load individual sample data
                for i in range(start_idx, end_idx):
                    sample_path = f"{chunk_dir}/sample_{i}.npy"
                    
                    if not os.path.exists(sample_path):
                        print(f"Warning: Sample file {sample_path} not found, skipping")
                        continue
                    
                    sample_data = np.load(sample_path, allow_pickle=True).item()
                    
                    # Load target info
                    if 'target_info' in sample_data:
                        self.target_info[i] = sample_data['target_info']
                    
                    # Load TX signal if available
                    if 'tx_signal_real' in sample_data and 'tx_signal_imag' in sample_data:
                        self.tx_signals[i] = sample_data['tx_signal_real'] + 1j * sample_data['tx_signal_imag']
                    
                    # Load TX power if available
                    if 'tx_power' in sample_data:
                        self.tx_powers[i] = sample_data['tx_power']
                    
                    # Load SNR value if available
                    if 'snr' in sample_data:
                        self.snr_values[i] = sample_data['snr']
                    
                    # Load signal type if available
                    if 'signal_type' in sample_data:
                        self.signal_types[i] = sample_data['signal_type']
                    
                    # Load channel model if available
                    if 'channel_model' in sample_data:
                        self.channel_models[i] = sample_data['channel_model']

    def _init_radar_processor_from_params(self):
        """Initialize radar processor with loaded parameters"""
        # Get default values for parameters that might not be in the loaded data
        num_subcarriers = getattr(self, 'num_subcarriers', 128)
        subcarrier_spacing = getattr(self, 'subcarrier_spacing', 30e3)
        transceiver_bandwidth = getattr(self, 'transceiver_bandwidth', 30e6)
        transceiver_center_freq = getattr(self, 'transceiver_center_freq', 2.1e9)
        signal_freq = getattr(self, 'signal_freq', 1e6)
        
        self.radar_processor = RadarProcessing(
            num_range_bins=self.num_range_bins,
            num_doppler_bins=self.num_doppler_bins,
            sample_rate=self.sample_rate,
            chirp_duration=self.chirp_duration,
            num_chirps=self.num_chirps,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing=subcarrier_spacing,
            bandwidth=self.bandwidth,
            transceiver_bandwidth=transceiver_bandwidth,
            transceiver_center_freq=transceiver_center_freq,
            output_freq=self.center_freq,
            signal_type=self.signal_type,
            signal_freq=signal_freq
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
        if self.use_lazy_loading:
            # Check if sample is in cache
            if idx in list(self.data_cache.keys()):
                return self.data_cache.get(idx)
            
            # Load data based on storage format
            if hasattr(self, 'h5_file') and self.h5_file is not None:
                # Old format - single HDF5 file
                time_domain = self.h5_file['time_domain_data'][idx]
                range_doppler = self.h5_file['range_doppler_maps'][idx]
                target_mask = self.h5_file['target_masks'][idx]
            elif hasattr(self, 'chunk_map') and idx in self.chunk_map:
                # New format - chunked HDF5 files
                chunk_idx, local_idx = self.chunk_map[idx]
                chunk_file = self.h5_files[chunk_idx]
                time_domain = chunk_file['time_domain_data'][local_idx]
                range_doppler = chunk_file['range_doppler_maps'][local_idx]
                target_mask = chunk_file['target_masks'][local_idx]
            elif hasattr(self, 'sample_paths') and idx in self.sample_paths:
                # New format - chunked NumPy files
                sample_data = np.load(self.sample_paths[idx], allow_pickle=True).item()
                time_domain = sample_data['time_domain_data']
                range_doppler = sample_data['range_doppler_map']
                target_mask = sample_data['target_mask']
            else:
                raise IndexError(f"Sample index {idx} not found in dataset")
            
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
                self.data_cache.pop(oldest_key, None)
            
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

    def calibrate_system_delay_using_crosstalk(self):
        """
        Calibrate the system delay by detecting the crosstalk between TX and RX
        
        This method generates a clean signal with no targets, then detects the
        crosstalk between transmitter and receiver to estimate the system delay.
        
        Returns:
            Estimated system delay in seconds
        """
        print("Calibrating system delay using TX-RX crosstalk...")
        
        # Generate a clean signal with no targets (just crosstalk)
        rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), dtype=np.complex64)
        
        # Add crosstalk (direct coupling between TX and RX)
        # This simulates the leakage from TX to RX with minimal delay
        tx_signal = self._generate_tx_signal()
        
        # Apply crosstalk to all RX antennas
        for rx_idx in range(self.num_rx):
            for chirp_idx in range(self.num_chirps):
                # Add attenuated TX signal as crosstalk
                # The attenuation factor simulates the isolation between TX and RX
                isolation_db = 30  # Typical isolation between TX and RX in dB
                attenuation = 10**(-isolation_db/20)
                
                # Add small random delay to simulate real hardware variations
                delay_samples = np.random.randint(5, 15)  # Small random delay
                
                # Apply the delay and attenuation
                if delay_samples < self.samples_per_chirp:
                    rx_signal[rx_idx, chirp_idx, delay_samples:] = attenuation * tx_signal[chirp_idx, :self.samples_per_chirp-delay_samples]
        
        # Visualize the crosstalk signal (for debugging)
        self._visualize_rx_signal(
            rx_signal, 
            [],  # No targets
            title="TX-RX Crosstalk Signal",
            save_path=f"{self.save_path}/debug/crosstalk_signal.pdf"
        )
        
        # Estimate system delay from crosstalk
        system_delay = self.radar_processor.estimate_system_delay_from_crosstalk(rx_signal)
        
        print(f"Estimated system delay from crosstalk: {system_delay*1e9:.2f} ns")
        
        # Store the system delay
        self.system_delay = system_delay
        self.radar_processor.system_delay = system_delay
        
        return system_delay
    
    def visualize_crosstalk_detection(self):
        """
        Visualize the crosstalk detection process
        
        This function generates a signal with crosstalk, then shows how
        the crosstalk is detected and used to estimate system delay.
        """
        # Create directory for debug figures
        os.makedirs(f"{self.save_path}/debug", exist_ok=True)
        
        # Generate a clean signal with no targets (just crosstalk)
        rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), dtype=np.complex64)
        
        # Add crosstalk (direct coupling between TX and RX)
        tx_signal = self._generate_tx_signal()
        
        # Apply crosstalk to all RX antennas
        for rx_idx in range(self.num_rx):
            for chirp_idx in range(self.num_chirps):
                # Add attenuated TX signal as crosstalk
                isolation_db = 30  # Typical isolation between TX and RX in dB
                attenuation = 10**(-isolation_db/20)
                
                # Add small random delay to simulate real hardware variations
                delay_samples = np.random.randint(5, 15)  # Small random delay
                
                # Apply the delay and attenuation
                if delay_samples < self.samples_per_chirp:
                    rx_signal[rx_idx, chirp_idx, delay_samples:] = attenuation * tx_signal[chirp_idx, :self.samples_per_chirp-delay_samples]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Select RX and chirp to visualize
        rx_idx, chirp_idx = 0, 0
        
        # 1. Plot the signal with crosstalk
        axs[0].plot(np.abs(rx_signal[rx_idx, chirp_idx]))
        axs[0].set_title('Signal Magnitude with TX-RX Crosstalk')
        axs[0].set_xlabel('Sample')
        axs[0].set_ylabel('Magnitude')
        axs[0].grid(True, alpha=0.3)
        
        # 2. Calculate and plot signal energy
        signal_energy = np.abs(rx_signal[rx_idx, chirp_idx])**2
        
        # Apply moving average to smooth
        window_size = int(self.sample_rate * 1e-6)  # 1 μs window
        window_size = max(1, window_size)  # Ensure window size is at least 1
        energy_smooth = np.convolve(signal_energy, np.ones(window_size)/window_size, mode='same')
        
        axs[1].plot(signal_energy, label='Raw Energy')
        axs[1].plot(energy_smooth, label='Smoothed Energy')
        axs[1].set_title('Signal Energy')
        axs[1].set_xlabel('Sample')
        axs[1].set_ylabel('Energy')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # 3. Calculate and plot energy derivative
        energy_diff = np.diff(energy_smooth)
        threshold = 0.3 * np.max(energy_diff)
        
        axs[2].plot(energy_diff)
        axs[2].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2e})')
        
        # Find the first point where energy rises significantly
        rise_points = np.where(energy_diff > threshold)[0]
        
        if len(rise_points) > 0:
            # The first significant rise is likely the crosstalk
            first_rise = rise_points[0]
            axs[2].axvline(x=first_rise, color='g', linestyle='--', 
                        label=f'Detected Crosstalk ({first_rise} samples, {first_rise/self.sample_rate*1e9:.2f} ns)')
            
            # Mark the same point on the other plots
            axs[0].axvline(x=first_rise, color='g', linestyle='--')
            axs[1].axvline(x=first_rise, color='g', linestyle='--')
        
        axs[2].set_title('Energy Derivative (for Crosstalk Detection)')
        axs[2].set_xlabel('Sample')
        axs[2].set_ylabel('Energy Derivative')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle('TX-RX Crosstalk Detection for System Delay Estimation', fontsize=16)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig(f"{self.save_path}/debug/crosstalk_detection.pdf")
        plt.close(fig)


def print_dataset_info(dataset):
    """Print detailed information about the dataset"""
    print("\n" + "="*80)
    print("RADAR DATASET INFORMATION")
    print("="*80)
    
    # Basic dataset information
    print(f"Number of samples: {dataset.num_samples}")
    print(f"Signal type: {dataset.signal_type}")
    
    # Dataset format information
    if hasattr(dataset, 'h5_file') and dataset.h5_file is not None:
        print(f"Dataset format: Single HDF5 file")
    elif hasattr(dataset, 'chunk_map') and dataset.chunk_map:
        print(f"Dataset format: Chunked HDF5 files ({len(set(chunk[0] for chunk in dataset.chunk_map.values()))} chunks)")
    elif hasattr(dataset, 'sample_paths') and dataset.sample_paths:
        print(f"Dataset format: Chunked NumPy files ({len(os.listdir(os.path.dirname(next(iter(dataset.sample_paths.values())))))} chunks)")
    
    # Loading mode
    if hasattr(dataset, 'use_lazy_loading') and dataset.use_lazy_loading:
        print(f"Loading mode: Lazy loading (with cache size: {dataset.cache_size})")
    else:
        print(f"Loading mode: Full loading")
    
    # Radar parameters
    print("\nRADAR PARAMETERS:")
    print(f"  Sample rate: {dataset.sample_rate/1e6:.1f} MHz")
    print(f"  Chirp duration: {dataset.chirp_duration*1e6:.1f} μs")
    print(f"  Number of chirps: {dataset.num_chirps}")
    print(f"  Bandwidth: {dataset.bandwidth/1e6:.1f} MHz")
    print(f"  Center frequency: {dataset.center_freq/1e9:.2f} GHz")
    print(f"  Number of RX antennas: {dataset.num_rx}")
    print(f"  Number of TX antennas: {dataset.num_tx}")
    
    # Hardware parameters (if available)
    if hasattr(dataset, 'transceiver_bandwidth'):
        print("\nHARDWARE PARAMETERS:")
        print(f"  Transceiver bandwidth: {dataset.transceiver_bandwidth/1e6:.1f} MHz")
        print(f"  Transceiver center frequency: {dataset.transceiver_center_freq/1e9:.2f} GHz")
    
    # Signal-specific parameters
    print("\nSIGNAL PARAMETERS:")
    print(f"  Signal type: {dataset.signal_type}")
    if dataset.signal_type in ['OFDM', 'OFDM_FMCW'] and hasattr(dataset, 'num_subcarriers'):
        print(f"  Number of subcarriers: {dataset.num_subcarriers}")
        print(f"  Subcarrier spacing: {dataset.subcarrier_spacing/1e3:.1f} kHz")
    if dataset.signal_type in ['Sine', 'Sine_FMCW'] and hasattr(dataset, 'signal_freq'):
        print(f"  Signal frequency: {dataset.signal_freq/1e6:.1f} MHz")
    
    # Realistic effects
    if hasattr(dataset, 'apply_realistic_effects'):
        print(f"\nREALISTIC EFFECTS: {'Enabled' if dataset.apply_realistic_effects else 'Disabled'}")
    
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
    
    # Handle different dataset formats
    if hasattr(dataset, 'time_domain_data') and not isinstance(dataset.time_domain_data, list):
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
    elif hasattr(dataset, 'time_domain_shape'):
        # For lazy-loaded datasets, we have shapes but not actual arrays
        print(f"  Time domain data shape: {dataset.time_domain_shape}")
        print(f"    - Interpretation: [num_samples, num_rx, num_chirps, samples_per_chirp, 2(I/Q)]")
        print(f"  Range-Doppler maps shape: {dataset.range_doppler_shape}")
        print(f"    - Interpretation: [num_samples, 2(real/imag), num_doppler_bins, num_range_bins]")
        print(f"  Target masks shape: {dataset.target_mask_shape}")
        print(f"    - Interpretation: [num_samples, num_doppler_bins, num_range_bins, 1]")
        
        # Estimate memory usage based on shapes
        time_domain_size = np.prod(dataset.time_domain_shape) * (2 if dataset.precision == 'float32' else 1)
        rd_maps_size = np.prod(dataset.range_doppler_shape) * (2 if dataset.precision == 'float32' else 1)
        masks_size = np.prod(dataset.target_mask_shape) * (2 if dataset.precision == 'float32' else 1)
        total_size = time_domain_size + rd_maps_size + masks_size
        
        print("\nESTIMATED MEMORY USAGE (IF FULLY LOADED):")
        print(f"  Time domain data: {time_domain_size/1e6:.2f} MB")
        print(f"  Range-Doppler maps: {rd_maps_size/1e6:.2f} MB")
        print(f"  Target masks: {masks_size/1e6:.2f} MB")
        print(f"  Total: {total_size/1e6:.2f} MB ({total_size/1e9:.2f} GB)")
        
        # Chunking information
        if hasattr(dataset, 'num_files') and hasattr(dataset, 'max_samples_per_file'):
            print(f"\nCHUNKING INFORMATION:")
            print(f"  Number of chunks: {dataset.num_files}")
            print(f"  Maximum samples per chunk: {dataset.max_samples_per_file}")
    else:
        print("  Data dimensions not available (dataset may be empty or not fully initialized)")
    
    # Additional data information
    if hasattr(dataset, 'tx_signals') and dataset.tx_signals:
        if isinstance(dataset.tx_signals, list):
            print("\nADDITIONAL DATA:")
            print(f"  TX signals: {len(dataset.tx_signals)} samples")
            if len(dataset.tx_signals) > 0 and dataset.tx_signals[0] is not None:
                print(f"    - Shape per sample: {dataset.tx_signals[0].shape}")
        else:
            print(f"  TX signals shape: {dataset.tx_signals.shape}")
    
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
    signal_types = []
    snr_values = []
    
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
        
        # Collect signal type if available
        if hasattr(dataset, 'signal_types') and dataset.signal_types and i < len(dataset.signal_types):
            if dataset.signal_types[i] is not None:
                signal_types.append(dataset.signal_types[i])
        
        # Collect SNR if available
        if hasattr(dataset, 'snr_values') and dataset.snr_values and i < len(dataset.snr_values):
            if dataset.snr_values[i] is not None:
                snr_values.append(dataset.snr_values[i])
    
    # Create figure for statistics
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
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
    
    # Plot signal type distribution if available
    if signal_types:
        unique_types = list(set(signal_types))
        type_counts = [signal_types.count(t) for t in unique_types]
        axs[2, 0].bar(unique_types, type_counts)
        axs[2, 0].set_title('Signal Type Distribution')
        axs[2, 0].set_xlabel('Signal Type')
        axs[2, 0].set_ylabel('Count')
        axs[2, 0].grid(True, alpha=0.3)
    else:
        axs[2, 0].text(0.5, 0.5, 'Signal type data not available', 
                      ha='center', va='center', transform=axs[2, 0].transAxes)
        axs[2, 0].set_title('Signal Type Distribution')
    
    # Plot SNR distribution if available
    if snr_values:
        axs[2, 1].hist(snr_values, bins=20)
        axs[2, 1].set_title('SNR Distribution')
        axs[2, 1].set_xlabel('SNR (dB)')
        axs[2, 1].set_ylabel('Count')
        axs[2, 1].grid(True, alpha=0.3)
    else:
        axs[2, 1].text(0.5, 0.5, 'SNR data not available', 
                      ha='center', va='center', transform=axs[2, 1].transAxes)
        axs[2, 1].set_title('SNR Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{dataset.save_path}/statistics/dataset_statistics{IMG_FORMAT}")
    plt.close()
    
    # Visualize selected samples
    for idx in indices_to_visualize:
        # Use the comprehensive visualization if available
        if hasattr(dataset, '_visualize_comprehensive_signal'):
            # Get sample data
            sample = dataset[idx]
            time_domain = sample['time_domain'].numpy()
            target_info = sample['target_info']
            
            # Convert time domain data back to complex
            complex_data = time_domain[:, :, :, 0] + 1j * time_domain[:, :, :, 1]
            
            # Get TX signal if available
            if hasattr(dataset, 'tx_signals') and dataset.tx_signals and idx < len(dataset.tx_signals):
                tx_signal = dataset.tx_signals[idx]
            else:
                # Generate a TX signal if not available
                tx_signal = dataset._generate_tx_signal()
            
            # Use comprehensive visualization
            dataset._visualize_comprehensive_signal(
                tx_signal=tx_signal,
                rx_signal=complex_data,
                title=f"Sample {idx}: {len(target_info)} Targets, {dataset.signal_type}",
                target_info=target_info,
                save_path=f"{dataset.save_path}/statistics/sample_{idx}_comprehensive{IMG_FORMAT}"
            )
        
        # Also use the standard visualization
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
    """
    Visualize the radar signal processing steps for a given sample
    
    Args:
        dataset: RadarDataset instance
        sample_idx: Index of the sample to visualize
    """
    # Get the sample data
    sample = dataset[sample_idx]
    
    # Extract time domain data (complex format)
    time_data = sample['time_domain_data']  # Shape: [num_rx, num_chirps, samples_per_chirp, 2]
    
    # Convert to complex format for processing
    complex_data = time_data[..., 0] + 1j * time_data[..., 1]
    
    # Create figure for visualization
    plt.figure(figsize=(15, 12))
    
    # 1. Plot raw time domain signal (first RX, first chirp)
    plt.subplot(3, 2, 1)
    plt.plot(np.real(complex_data[0, 0, :]), label='Real')
    plt.plot(np.imag(complex_data[0, 0, :]), label='Imaginary')
    plt.title('Raw Time Domain Signal (RX1, Chirp1)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 2. Apply signal-specific processing based on signal type
    if dataset.signal_type in ['OFDM_FMCW', 'Sine_FMCW']:
        # For hybrid signal types, apply the two-step demodulation process
        demodulated_data = dataset.radar_processor.simulate_hardware_demodulation(complex_data)
        
        # Plot demodulated time domain signal
        plt.subplot(3, 2, 2)
        plt.plot(np.real(demodulated_data[0, 0, :]), label='Real')
        plt.plot(np.imag(demodulated_data[0, 0, :]), label='Imaginary')
        plt.title(f'Demodulated Signal ({dataset.signal_type})')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # Process the demodulated data
        range_profile = dataset.radar_processor.compute_range_profile(demodulated_data)
    else:
        # For standard signal types, follow the normal processing pipeline
        range_profile = dataset.radar_processor.compute_range_profile(complex_data)
    
    # 3. Plot range profile (first RX, first chirp)
    plt.subplot(3, 2, 3)
    range_profile_db = 20 * np.log10(np.abs(range_profile[0, 0, :]) + 1e-10)
    plt.plot(range_profile_db)
    plt.title('Range Profile (dB)')
    plt.xlabel('Range Bin')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    
    # 4. Compute Doppler processing
    range_doppler = dataset.radar_processor.compute_range_doppler(range_profile)
    
    # 5. Plot range-Doppler map
    plt.subplot(3, 2, 4)
    range_doppler_db = 20 * np.log10(np.abs(range_doppler[0, :, :]) + 1e-10)
    plt.imshow(range_doppler_db, aspect='auto', cmap='jet', origin='lower')
    plt.title('Range-Doppler Map (dB)')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.colorbar(label='Magnitude (dB)')
    
    # 6. Apply CFAR detection
    detections = dataset.radar_processor.apply_cfar_detection(range_doppler)
    
    # 7. Plot CFAR detections
    plt.subplot(3, 2, 5)
    plt.imshow(range_doppler_db, aspect='auto', cmap='jet', origin='lower')
    
    # Overlay detections
    detection_indices = np.where(detections[0, :, :] > 0)
    if len(detection_indices[0]) > 0:
        plt.scatter(detection_indices[1], detection_indices[0], 
                   c='red', marker='x', s=40, label='Detections')
    plt.title('CFAR Detections')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.colorbar(label='Magnitude (dB)')
    if len(detection_indices[0]) > 0:
        plt.legend()
    
    # 8. Plot ground truth targets
    plt.subplot(3, 2, 6)
    plt.imshow(range_doppler_db, aspect='auto', cmap='jet', origin='lower')
    
    # Get target information
    target_info = sample['target_info']
    
    # Convert target information to range-Doppler bins
    for target in target_info:
        # Convert distance to range bin
        range_bin = int(target['distance'] / dataset.range_resolution)
        
        # Convert velocity to Doppler bin
        doppler_bin = int(target['velocity'] / dataset.velocity_resolution + dataset.num_doppler_bins // 2)
        
        # Ensure bins are within valid range
        if 0 <= range_bin < dataset.num_range_bins and 0 <= doppler_bin < dataset.num_doppler_bins:
            plt.scatter(range_bin, doppler_bin, c='green', marker='o', s=80, 
                       label='Ground Truth', alpha=0.7)
    
    plt.title('Ground Truth Targets')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.colorbar(label='Magnitude (dB)')
    
    # Add overall title
    plt.suptitle(f'Radar Signal Processing Steps - {dataset.signal_type}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(f'data/radar_results/signal_processing_{dataset.signal_type}.pdf')
    plt.close()
    
    print(f"Signal processing visualization saved for {dataset.signal_type}")

def compare_signal_types(save_path, num_samples=100, signal_types=None, visualize_samples=5, 
                        radar_params=None, apply_realistic_effects=True):
    """Compare the performance of different signal types in signal processing steps
    
    Args:
        save_path: Base path to save comparison results
        num_samples: Number of samples to generate for each signal type
        signal_types: List of signal types to compare. If None, uses default types.
        visualize_samples: Number of samples to visualize for each signal type
        radar_params: Dictionary of radar parameters to use for all signal types
        apply_realistic_effects: Whether to apply realistic radar effects
        
    Returns:
        tuple: (datasets, metrics) containing the generated datasets and performance metrics
    """
    # Create directory for comparison figures
    comparison_dir = os.path.join(save_path, "signal_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Signal types to compare
    if signal_types is None:
        signal_types = ['FMCW', 'OFDM', 'OFDM_FMCW', 'Sine_FMCW']
    
    # Common parameters for all datasets
    if radar_params is None:
        radar_params = {
            'num_samples': num_samples,
            'num_range_bins': 128,
            'num_doppler_bins': 64,
            'sample_rate': 15e6,
            'chirp_duration': 1e-3,
            'num_chirps': 32,
            'bandwidth': 500e6,
            'center_freq': 10e9,
            'num_rx': 4,
            'num_tx': 1,
            'max_targets': 3,
            'snr_min': 15,
            'snr_max': 25,
            'apply_realistic_effects': apply_realistic_effects,
            'drawfig': False,
            'use_lazy_loading': True,  # Enable lazy loading for better memory management
            'cache_size': 20,          # Cache size for lazy loading
            'savedataformat': 'chunked_hdf5',  # Use chunked format for better I/O performance
            'max_samples_per_file': 50  # Number of samples per chunk
        }
    
    # Metrics to track
    metrics = {
        'range_resolution': [],
        'velocity_resolution': [],
        'snr_improvement': [],
        'detection_accuracy': [],
        'processing_time': []
    }
    
    # Generate datasets and collect metrics
    datasets = {}
    for signal_type in signal_types:
        print(f"\nGenerating dataset for {signal_type}...")
        signal_save_path = os.path.join(save_path, signal_type)
        
        # Create dataset with current signal type
        dataset = RadarDataset(
            signal_type=signal_type,
            save_path=signal_save_path,
            **radar_params
        )
        
        datasets[signal_type] = dataset
        
        # Store metrics
        metrics['range_resolution'].append(dataset.range_resolution)
        metrics['velocity_resolution'].append(dataset.velocity_resolution)
        
        # Calculate average SNR improvement (ratio of peak to noise floor)
        snr_improvements = []
        processing_times = []
        detection_accuracies = []
        
        # Process a subset of samples to measure performance
        for i in range(min(visualize_samples, num_samples)):
            sample = dataset[i]
            time_domain = sample['time_domain'].numpy()
            target_info = sample['target_info']
            
            # Convert time domain data to complex
            complex_data = time_domain[:, :, :, 0] + 1j * time_domain[:, :, :, 1]
            
            # Get TX signal if available
            if hasattr(dataset, 'tx_signals') and dataset.tx_signals and i < len(dataset.tx_signals):
                tx_signal = dataset.tx_signals[i]
            else:
                # Generate a TX signal if not available
                tx_signal = dataset._generate_tx_signal()
            
            # Measure processing time
            start_time = time.time()
            rd_map = dataset.radar_processor.time_to_range_doppler(complex_data)
            processing_times.append(time.time() - start_time)
            
            # Calculate SNR improvement
            rd_magnitude = np.abs(rd_map)
            noise_floor = np.median(rd_magnitude)
            
            # Find peaks corresponding to targets
            target_peaks = []
            for target in target_info:
                range_bin = int(target['distance'] / dataset.range_resolution)
                doppler_bin = int(dataset.num_doppler_bins/2 + target['velocity'] / dataset.velocity_resolution)
                
                if (0 <= range_bin < dataset.num_range_bins and 
                    0 <= doppler_bin < dataset.num_doppler_bins):
                    # Get 3x3 region around target
                    region = rd_magnitude[
                        max(0, doppler_bin-1):min(dataset.num_doppler_bins, doppler_bin+2),
                        max(0, range_bin-1):min(dataset.num_range_bins, range_bin+2)
                    ]
                    target_peaks.append(np.max(region))
            
            if target_peaks:
                avg_peak = np.mean(target_peaks)
                snr_improvements.append(20 * np.log10(avg_peak / noise_floor))
            
            # Simple detection accuracy (percentage of targets correctly detected)
            detected = 0
            for target in target_info:
                range_bin = int(target['distance'] / dataset.range_resolution)
                doppler_bin = int(dataset.num_doppler_bins/2 + target['velocity'] / dataset.velocity_resolution)
                
                if (0 <= range_bin < dataset.num_range_bins and 
                    0 <= doppler_bin < dataset.num_doppler_bins):
                    # Check if there's a peak near the target
                    region = rd_magnitude[
                        max(0, doppler_bin-2):min(dataset.num_doppler_bins, doppler_bin+3),
                        max(0, range_bin-2):min(dataset.num_range_bins, range_bin+3)
                    ]
                    if np.max(region) > 3 * noise_floor:  # Simple threshold
                        detected += 1
            
            if target_info:
                detection_accuracies.append(detected / len(target_info))
            
            # Use the comprehensive visualization if available
            if hasattr(dataset, '_visualize_comprehensive_signal'):
                # Use comprehensive visualization
                dataset._visualize_comprehensive_signal(
                    tx_signal=tx_signal,
                    rx_signal=complex_data,
                    title=f"{signal_type} Sample {i}: {len(target_info)} Targets",
                    target_info=target_info,
                    save_path=f"{comparison_dir}/{signal_type}_sample_{i}_comprehensive{IMG_FORMAT}"
                )
        
        metrics['snr_improvement'].append(np.mean(snr_improvements) if snr_improvements else 0)
        metrics['processing_time'].append(np.mean(processing_times))
        metrics['detection_accuracy'].append(np.mean(detection_accuracies) * 100 if detection_accuracies else 0)  # Convert to percentage
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # 1. Range Resolution Comparison
    plt.subplot(2, 2, 1)
    plt.bar(signal_types, metrics['range_resolution'])
    plt.title('Range Resolution Comparison')
    plt.ylabel('Resolution (m)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Velocity Resolution Comparison
    plt.subplot(2, 2, 2)
    plt.bar(signal_types, metrics['velocity_resolution'])
    plt.title('Velocity Resolution Comparison')
    plt.ylabel('Resolution (m/s)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. SNR Improvement Comparison
    plt.subplot(2, 2, 3)
    plt.bar(signal_types, metrics['snr_improvement'])
    plt.title('SNR Improvement Comparison')
    plt.ylabel('SNR Improvement (dB)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Detection Accuracy Comparison
    plt.subplot(2, 2, 4)
    plt.bar(signal_types, metrics['detection_accuracy'])
    plt.title('Detection Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"signal_type_metrics_comparison{IMG_FORMAT}"))
    plt.close()
    
    # Create detailed range profile comparison
    plt.figure(figsize=(15, 10))
    
    # Use the same target scenario for all signal types
    target_scenario = {
        'distance': [30, 60, 90],
        'velocity': [5, -3, 0],
        'rcs': [0.8, 0.6, 0.4]
    }
    
    # Process and plot range profiles
    for i, signal_type in enumerate(signal_types):
        dataset = datasets[signal_type]
        
        # Generate a sample with the same target scenario
        time_data, _, _ = dataset.radar_processor.generate_radar_data(
            target_scenario['distance'],
            target_scenario['velocity'],
            target_scenario['rcs'],
            add_noise=True,
            snr_db=20
        )
        
        # Convert to complex
        complex_data = time_data[:, :, :, 0] + 1j * time_data[:, :, :, 1]
        
        # Generate TX signal for this scenario
        tx_signal = dataset._generate_tx_signal()
        
        # Process to get range profile (first RX, first chirp)
        rx_idx, chirp_idx = 0, 0
        
        # Apply window function
        window = np.hamming(dataset.samples_per_chirp)
        windowed_data = complex_data[rx_idx, chirp_idx] * window
        
        # Perform range FFT
        range_fft = np.fft.fft(windowed_data, n=dataset.num_range_bins)
        range_profile_mag = np.abs(range_fft)
        range_profile_db = 20 * np.log10(range_profile_mag + 1e-10)
        
        # Plot range profile
        plt.subplot(2, 2, i+1)
        plt.plot(np.arange(dataset.num_range_bins) * dataset.range_resolution, range_profile_db)
        plt.title(f'{signal_type} Range Profile')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        
        # Mark true target positions
        for dist in target_scenario['distance']:
            plt.axvline(x=dist, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"range_profile_comparison{IMG_FORMAT}"))
    plt.close()
    
    # Create detailed range-Doppler map comparison
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Process and plot range profiles
    for i, signal_type in enumerate(signal_types):
        dataset = datasets[signal_type]
        
        # Generate a sample with the same target scenario
        # Create targets with the same scenario for all signal types
        targets = []
        for dist, vel, rcs_val in zip(target_scenario['distance'], 
                                     target_scenario['velocity'], 
                                     target_scenario['rcs']):
            targets.append({
                'distance': dist,
                'velocity': vel,
                'rcs': rcs_val
            })
        
        # Generate time domain data using dataset's internal methods
        time_data = dataset._generate_radar_data(targets)
        
        # Convert to complex
        complex_data = time_data[:, :, :, 0] + 1j * time_data[:, :, :, 1]
        
        # Generate TX signal for this scenario
        tx_signal = dataset._generate_tx_signal()
        
        # Process to get range-Doppler map
        rd_map = dataset.radar_processor.time_to_range_doppler(complex_data)
        
        # Plot range-Doppler map
        rd_magnitude = np.abs(rd_map)
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)
        
        row, col = i // 2, i % 2
        im = axs[row, col].imshow(rd_db, aspect='auto', cmap='jet', origin='lower',
                                 extent=[0, dataset.max_range, -dataset.max_velocity, dataset.max_velocity])
        axs[row, col].set_title(f'{signal_type} Range-Doppler Map')
        axs[row, col].set_xlabel('Range (m)')
        axs[row, col].set_ylabel('Velocity (m/s)')
        
        # Mark true target positions
        for dist, vel in zip(target_scenario['distance'], target_scenario['velocity']):
            axs[row, col].plot(dist, vel, 'ro', markersize=8)
        
        plt.colorbar(im, ax=axs[row, col])
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"range_doppler_comparison{IMG_FORMAT}"))
    plt.close()
    
    # Create a summary table
    summary_data = {
        'Signal Type': signal_types,
        'Range Resolution (m)': metrics['range_resolution'],
        'Velocity Resolution (m/s)': metrics['velocity_resolution'],
        'SNR Improvement (dB)': metrics['snr_improvement'],
        'Detection Accuracy (%)': metrics['detection_accuracy'],
        'Processing Time (s)': metrics['processing_time']
    }
    
    # Save summary as CSV
    import csv
    with open(os.path.join(comparison_dir, "signal_type_comparison.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(summary_data.keys())
        writer.writerows(zip(*summary_data.values()))
    
    # Generate a comprehensive comparison report
    with open(os.path.join(comparison_dir, 'signal_comparison_report.md'), 'w') as f:
        f.write('# Radar Signal Type Comparison Report\n\n')
        
        f.write('## Signal Types Compared\n\n')
        for signal_type in signal_types:
            f.write(f'- {signal_type}\n')
        
        f.write('\n## Radar Parameters\n\n')
        for param, value in radar_params.items():
            # Format the value based on its magnitude
            if isinstance(value, (int, float)) and value >= 1e6:
                formatted_value = f"{value/1e6:.2f} MHz" if param.endswith('_rate') or param == 'bandwidth' else f"{value/1e6:.2f} M"
            elif isinstance(value, (int, float)) and value >= 1e3:
                formatted_value = f"{value/1e3:.2f} kHz" if param.endswith('_rate') or param == 'bandwidth' else f"{value/1e3:.2f} k"
            else:
                formatted_value = str(value)
            
            f.write(f'- **{param}**: {formatted_value}\n')
        
        f.write('\n## Detection Performance\n\n')
        f.write('| Signal Type | Accuracy | SNR Improvement | Processing Time |\n')
        f.write('|------------|----------|-----------------|----------------|\n')
        for i, signal_type in enumerate(signal_types):
            f.write(f'| {signal_type} | {metrics["detection_accuracy"][i]:.2f}% | {metrics["snr_improvement"][i]:.2f} dB | {metrics["processing_time"][i]*1000:.2f} ms |\n')
        
        f.write('\n## Signal Characteristics\n\n')
        f.write('### Time-Domain Properties\n\n')
        for signal_type in signal_types:
            f.write(f'#### {signal_type}\n\n')
            f.write(f'- **Waveform**: {signal_type} uses a {"continuous frequency sweep" if "FMCW" in signal_type else "discrete multi-carrier" if "OFDM" in signal_type else "sinusoidal"} approach\n')
            if "FMCW" in signal_type:
                f.write('- **Advantages**: Good range resolution, simple processing\n')
                f.write('- **Disadvantages**: Susceptible to interference, limited flexibility\n')
            elif "OFDM" in signal_type:
                f.write('- **Advantages**: Robust against frequency-selective fading, flexible resource allocation\n')
                f.write('- **Disadvantages**: Higher peak-to-average power ratio, more complex processing\n')
            elif "Sine" in signal_type:
                f.write('- **Advantages**: Simple generation, good for Doppler estimation\n')
                f.write('- **Disadvantages**: Limited range resolution\n')
            
            if "_" in signal_type:
                f.write('- **Hybrid Approach**: Combines multiple waveform types for improved performance\n')
            
            f.write('\n')
        
        f.write('\n## Conclusion\n\n')
        best_signal_type = signal_types[np.argmax(metrics['detection_accuracy'])]
        f.write(f'Based on the detection accuracy, **{best_signal_type}** shows the best overall performance ')
        f.write(f'with an accuracy of {np.max(metrics["detection_accuracy"]):.2f}%.\n\n')
        
        f.write('### Recommendations\n\n')
        f.write('- For applications requiring highest detection accuracy, use **' + best_signal_type + '**\n')
        f.write('- For applications requiring best range resolution, use **' + 
                signal_types[np.argmin(metrics['range_resolution'])] + '**\n')
        f.write('- For applications requiring best velocity resolution, use **' + 
                signal_types[np.argmin(metrics['velocity_resolution'])] + '**\n')
        
        # Add information about TX signal characteristics
        f.write('\n## TX Signal Analysis\n\n')
        f.write('The transmit signal characteristics significantly impact radar performance:\n\n')
        for signal_type in signal_types:
            f.write(f'### {signal_type}\n\n')
            if "FMCW" in signal_type:
                f.write('- **TX Signal**: Linear frequency sweep from f₀ to f₀+B\n')
                f.write('- **Bandwidth Utilization**: Efficiently uses entire allocated bandwidth\n')
                f.write('- **Peak-to-Average Power Ratio**: Low (≈1), enabling efficient power amplifier operation\n\n')
            elif "OFDM" in signal_type:
                f.write('- **TX Signal**: Multiple orthogonal subcarriers with digital modulation\n')
                f.write('- **Bandwidth Utilization**: Highly efficient with flexible subcarrier allocation\n')
                f.write('- **Peak-to-Average Power Ratio**: High, requiring power back-off in amplifiers\n\n')
            elif "Sine" in signal_type:
                f.write('- **TX Signal**: Single-tone sinusoidal signal\n')
                f.write('- **Bandwidth Utilization**: Limited (narrow bandwidth)\n')
                f.write('- **Peak-to-Average Power Ratio**: Optimal (=1)\n\n')
    
    print(f"Comparison complete. Results saved to {comparison_dir}")
    print(f"Generated comparison report: {os.path.join(comparison_dir, 'signal_comparison_report.md')}")
    
    return datasets, metrics


# Update the main execution to include the new comparison function
if __name__ == '__main__':
    # Test and visualization of the RadarDataset
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Radar Dataset Generation and Visualization')
    parser.add_argument('--mode', type=str, default='generate', 
                        choices=['generate', 'load', 'visualize', 'compare'],
                        help='Mode: generate new data, load existing data, visualize existing data, or compare signal types')
    parser.add_argument('--datapath', type=str, default="data/radarv3/FMCW/radar_data.h5", 
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
            snr_min=15,
            snr_max=30,
            save_path=args.save_path,
            savedataformat=args.format,
            apply_realistic_effects = False,
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
    
    elif args.mode == 'compare':
        print("Comparing performance of different signal types...")
        base_save_path = os.path.dirname(args.save_path)  # Use parent directory
        
        # Run comparison with smaller number of samples for efficiency
        compare_samples = min(args.num_samples, 20)
        # Update to use additional parameters
        datasets = compare_signal_types(
            save_path=base_save_path, 
            num_samples=compare_samples,
            signal_types=['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW'],
            visualize_samples=args.visualize_samples,
            apply_realistic_effects=True
        )
    
    print("Done!")