import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from scipy.signal import chirp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
#from scipy.signal import blackmanharris

class RayTracingRadarDataset:
    """
    A radar dataset generator that uses ray-tracing principles to simulate
    radar signal propagation, target interaction, and detection.
    """
    
    def __init__(self, 
                num_samples=20,
                 num_range_bins=256,
                 num_doppler_bins=128,
                 sample_rate=8e6,
                 bandwidth=200e6,
                 center_freq=77e9,
                 chirp_duration=50e-6, #50e-6,
                 num_chirps=128,
                 num_rx=4,
                 num_tx=2,
                 max_targets=5,
                 snr_min=5,
                 snr_max=20,
                 apply_realistic_effects=False,
                 save_path='data/raytracing_radar',
                 precision='float32'):
        """
        Initialize the ray-tracing radar dataset generator.
        
        Args:
            num_samples: Number of samples to generate
            num_range_bins: Number of range bins
            num_doppler_bins: Number of Doppler bins
            sample_rate: Sample rate in Hz
            bandwidth: Signal bandwidth in Hz
            center_freq: Center frequency in Hz
            chirp_duration: Chirp duration in seconds
            num_chirps: Number of chirps per frame
            num_rx: Number of RX antennas
            num_tx: Number of TX antennas
            max_targets: Maximum number of targets per sample
            snr_min: Minimum SNR in dB
            snr_max: Maximum SNR in dB
            apply_realistic_effects: Whether to apply realistic effects
            save_path: Path to save generated data
            precision: Precision of saved data
        """
        # Store basic parameters
        self.num_samples = num_samples
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.center_freq = center_freq
        self.chirp_duration = chirp_duration
        self.num_chirps = num_chirps
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.max_targets = max_targets
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.apply_realistic_effects = apply_realistic_effects
        self.save_path = save_path
        self.precision = precision
        
        # Calculate derived parameters
        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        self.wavelength = 3e8 / self.center_freq  # Wavelength in meters (e.g., ~3.9mm for 77GHz)
        self.speed_of_light = 3e8  # Speed of light in m/s

        # === Range Performance ===
        self.range_resolution = self.speed_of_light / (2 * self.bandwidth)  # ΔR = c/(2*BW)
        self.max_range = (self.sample_rate * self.speed_of_light * self.chirp_duration) / (2 * self.bandwidth)  # R_max = (fs * c * T_chirp)/(2*BW)
        self.min_range = self.range_resolution  # Practical minimum (can be higher due to Tx/Rx coupling)

        # === Velocity/Doppler Performance ===
        self.doppler_resolution = self.wavelength / (2 * self.num_chirps * self.chirp_duration)  # Δv = λ/(2*N*T_chirp)
        self.max_unambiguous_velocity = self.wavelength / (4 * self.chirp_duration)  # v_max = λ/(4*T_chirp)
        self.max_velocity = self.max_unambiguous_velocity  # Practical maximum (can be lower due to Doppler shift)
        self.velocity_resolution = self.wavelength / (2 * self.num_chirps * self.chirp_duration)
        
        # === SNR & Processing Metrics ===
        self.range_fft_size = self.num_range_bins  # Typically matches samples_per_chirp (zero-padded if needed)
        self.doppler_fft_size = self.num_doppler_bins  # Affects velocity resolution
        self.pulse_repetition_interval = self.chirp_duration  # PRI = T_chirp (for FMCW)
        self.pulse_repetition_frequency = 1 / self.chirp_duration  # PRF = 1/T_chirp

        # === Print Key Metrics === (Optional, for debugging)
        print(f"Range Resolution: {self.range_resolution:.2f} m")
        print(f"Max Unambiguous Range: {self.max_range:.2f} m")
        print(f"Doppler Resolution: {self.doppler_resolution:.2f} m/s")
        print(f"Max Unambiguous Velocity: {self.max_unambiguous_velocity:.2f} m/s")
        print(f"PRF: {self.pulse_repetition_frequency/1e3:.2f} kHz")

        # Create directory for saving data
        os.makedirs(self.save_path, exist_ok=True)
        
        # Print radar parameters
        self._print_radar_parameters()
    
    def _print_radar_parameters(self):
        """Print the radar system parameters."""
        print("\n=== Ray-Tracing Radar Simulation Parameters ===")
        print(f"Range Resolution: {self.range_resolution:.2f} m")
        print(f"Maximum Range: {self.max_range:.2f} m")
        print(f"Velocity Resolution: {self.velocity_resolution:.2f} m/s")
        print(f"Maximum Velocity: {self.max_velocity:.2f} m/s")
        print(f"Samples per Chirp: {self.samples_per_chirp}")
        print(f"Wavelength: {self.wavelength:.4f} m")
        print(f"Center Frequency: {self.center_freq/1e9:.2f} GHz")
        print(f"Bandwidth: {self.bandwidth/1e6:.2f} MHz")
        print(f"Chirp Duration: {self.chirp_duration*1e6:.2f} μs")
        print(f"Number of Chirps: {self.num_chirps}")
        print(f"Number of RX Antennas: {self.num_rx}")
        print(f"Number of TX Antennas: {self.num_tx}")
        print("================================================\n")
    
    def generate_dataset(self, visualize=True):
        """
        Generate a radar dataset using ray-tracing simulation.
        
        Args:
            visualize: Whether to visualize the results
            
        Returns:
            Dictionary containing the generated dataset
        """
        print(f"Generating {self.num_samples} radar samples using ray-tracing simulation...")
        
        # Initialize data containers
        dataset = {
            'time_domain_data': np.zeros((self.num_samples, self.num_rx, self.num_chirps, 
                                         self.samples_per_chirp, 2), dtype=self.precision),
            'range_doppler_maps': np.zeros((self.num_samples, 2, self.num_doppler_bins, 
                                           self.num_range_bins), dtype=self.precision),
            'target_masks': np.zeros((self.num_samples, self.num_doppler_bins, 
                                     self.num_range_bins, 1), dtype=self.precision),
            'target_info': [],
            'detection_results': []
        }
        
        # Generate samples
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            # Generate TX signal
            tx_signal = self._generate_tx_signal() #(128, 400) complex
            
            # Generate random targets
            targets = self._generate_random_targets()
            
            # Perform ray-tracing simulation
            rx_signal = self._ray_tracing_simulation(tx_signal, targets, perfect_mode=True)  # Enable perfect mode
            
            # Add noise to the received signal
            snr_db = random.uniform(self.snr_min, self.snr_max)
            rx_signal = self._add_noise(rx_signal, snr_db)
            
            # Process the received signal to generate range-Doppler map
            #rd_map = self._time_to_range_doppler(rx_signal) #(4, 128, 400) complex
            #(2, 128, 256)
            # Process the received signal to generate range-Doppler map
            rd_map = self._time_to_range_doppler(
                rx_signal,
                apply_mti=True,               # Enable MTI for stationary target suppression
                apply_doppler_centering=True,
                apply_notch_filter=True,
                notch_width=5,               # Increase notch width from default 3
                use_blackman_window=True,
                dynamic_range_db=50          # Increase from 40dB
            )

            # Perform target detection using CFAR
            detection_results = self._cfar_detection(rd_map)
            
            # Create target mask (ground truth)
            target_mask = self._create_target_mask(targets)
            
            # Store data
            dataset['time_domain_data'][i, :, :, :, 0] = np.real(rx_signal)
            dataset['time_domain_data'][i, :, :, :, 1] = np.imag(rx_signal)
            dataset['range_doppler_maps'][i] = rd_map
            dataset['target_masks'][i] = target_mask
            dataset['target_info'].append(targets)
            dataset['detection_results'].append(detection_results)
            
            # Visualize if requested
            if visualize: # and (i % 10 == 0 or i == self.num_samples - 1):
                self._visualize_sample(i, tx_signal, rx_signal, rd_map, targets, detection_results)
        
        print("Dataset generation complete!")
        return dataset
    
    def __getitem__(self, idx):
        """
        Fetch a sample from the dataset
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Dictionary containing:
                - time_domain: Time-domain FMCW RX signal with targets [num_rx, num_chirps, samples_per_chirp, 2]
                - feature_2d: Range-Doppler map [2, num_doppler_bins, num_range_bins]
                - labels: Target mask [num_doppler_bins, num_range_bins, 1]
                - target_info: List of dictionaries containing target information
        """
        # Check if we have a cached dataset
        if hasattr(self, 'dataset') and self.dataset is not None:
            # Get data from the cached dataset
            time_domain_data = self.dataset['time_domain_data'][idx]
            range_doppler_map = self.dataset['range_doppler_maps'][idx]
            target_mask = self.dataset['target_masks'][idx]
            target_info = self.dataset['target_info'][idx]
        else:
            # Generate a single sample on-the-fly
            # Set random seed based on idx for reproducibility
            np.random.seed(idx)
            random.seed(idx)
            
            # Generate TX signal
            tx_signal = self._generate_tx_signal()
            
            # Generate random targets
            targets = self._generate_random_targets()
            
            # Perform ray-tracing simulation
            rx_signal = self._ray_tracing_simulation(tx_signal, targets, perfect_mode=True)
            
            # Add noise to the received signal
            snr_db = random.uniform(self.snr_min, self.snr_max)
            rx_signal = self._add_noise(rx_signal, snr_db)
            
            # Process the received signal to generate range-Doppler map
            range_doppler_map = self._time_to_range_doppler(
                rx_signal,
                apply_mti=True,
                apply_doppler_centering=True,
                apply_notch_filter=True,
                notch_width=5,
                use_blackman_window=True,
                dynamic_range_db=50
            )
            
            # Create target mask (ground truth)
            target_mask = self._create_target_mask(targets)
            
            # Convert complex rx_signal to real/imag components
            time_domain_data = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp, 2), dtype=self.precision)
            time_domain_data[:, :, :, 0] = np.real(rx_signal)
            time_domain_data[:, :, :, 1] = np.imag(rx_signal)
            
            # Store target info
            target_info = targets
            
            # Reset random seed
            np.random.seed(None)
            random.seed(None)
        
        # Create sample dictionary
        sample = {
            'time_domain': time_domain_data,  # [num_rx, num_chirps, samples_per_chirp, 2]
            'feature_2d': range_doppler_map,  # [2, num_doppler_bins, num_range_bins]
            'labels': target_mask,            # [num_doppler_bins, num_range_bins, 1]
            'target_info': target_info        # List of dictionaries with target information
        }
        
        return sample
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.num_samples

    def _generate_tx_signal(self, tx_power=1.0):
        """
        Generate phase-continuous FMCW chirp signal with proper Doppler handling.
        - Phase Continuity : Uses a single continuous time vector for all chirps
        - Doppler Consistency : Proper phase accumulation between chirps using t_frame // self.chirp_duration

        Args:
            tx_power: Transmission power scaling factor
            
        Returns:
            Complex TX signal with shape [num_chirps, samples_per_chirp]
        """
        # Create continuous time vector for entire frame
        t_frame = np.arange(self.num_chirps * self.samples_per_chirp) / self.sample_rate
        
        # Calculate sweep rate (Hz/s)
        sweep_rate = self.bandwidth / self.chirp_duration
        
        # Generate phase-continuous signal
        phase = 2 * np.pi * (
            0.5 * sweep_rate * (t_frame % self.chirp_duration)**2 +  # Chirp phase
            sweep_rate * self.chirp_duration * (t_frame // self.chirp_duration) * (t_frame % self.chirp_duration)  # Phase accumulation
        )
        
        # Reshape into individual chirps
        tx_signal = np.exp(1j * phase).reshape(self.num_chirps, self.samples_per_chirp)
        
        # Apply power scaling
        tx_signal *= np.sqrt(tx_power)
        
        return tx_signal
    
    def _generate_random_targets(self):
        """
        Generate random radar targets.
        
        Returns:
            List of dictionaries containing target parameters
        """
        # Generate random number of targets (1 to max_targets) - ensure at least 1 target
        num_targets = random.randint(1, self.max_targets)
        
        # List to store target information
        targets = []
        
        # Generate target parameters
        for _ in range(num_targets):
            # Generate random target parameters with more reasonable ranges
            distance = random.uniform(self.min_range, self.max_range * 0.5)
            velocity = random.uniform(-self.max_velocity * 0.5, self.max_velocity * 0.5)
            
            # Increase RCS range for better visibility
            rcs = random.uniform(5.0, 30.0)  # Increase from (1.0, 20.0)
            
            # Generate random 3D position (for ray-tracing)
            azimuth = random.uniform(-45, 45)  # Narrower azimuth range
            elevation = random.uniform(-10, 10)  # Narrower elevation range
            
            # Convert spherical to Cartesian coordinates
            azimuth_rad = np.deg2rad(azimuth)
            elevation_rad = np.deg2rad(elevation)
            x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            z = distance * np.sin(elevation_rad)
            
            # Store target information
            target = {
                'distance': distance,
                'velocity': velocity,
                'rcs': rcs,
                'azimuth': azimuth,
                'elevation': elevation,
                'position': (x, y, z)
            }
            targets.append(target)
        
        return targets
    
    def _ray_tracing_simulation(self, tx_signal, targets, perfect_mode=False):
        """
        Perform ray-tracing simulation to generate received signals.
        
        Args:
            tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp]
            targets: List of target dictionaries
            perfect_mode: If True, eliminates cross-talk between channels for ideal simulation
            
        Returns:
            Complex RX signal with shape [num_rx, num_chirps, samples_per_chirp]
        """
        # Initialize RX signal (all zeros)
        rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), 
                            dtype=np.complex64)
        
        # Define RX antenna positions (simple linear array along x-axis)
        rx_positions = []
        rx_spacing = self.wavelength / 2  # Half-wavelength spacing
        for rx_idx in range(self.num_rx):
            rx_positions.append((rx_idx * rx_spacing, 0, 0))
        
        # For each target, calculate the reflected signal
        for target in targets:
            # Extract target parameters
            distance = target['distance']
            velocity = target['velocity']
            rcs = target['rcs']
            position = target['position']
            
            # For each RX antenna, calculate the received signal
            for rx_idx, rx_pos in enumerate(rx_positions):
                # Calculate exact distance from target to this RX antenna
                dx = position[0] - rx_pos[0]
                dy = position[1] - rx_pos[1]
                dz = position[2] - rx_pos[2]
                exact_distance = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Time delay for the target (round trip)
                delay_seconds = 2 * exact_distance / self.speed_of_light
                delay_samples = int(delay_seconds * self.sample_rate)
                
                # Doppler shift due to target velocity
                doppler_freq = 2 * velocity * self.center_freq / self.speed_of_light
                
                # Calculate attenuation using radar equation
                # Increase the attenuation factor to make targets more visible
                attenuation = np.sqrt(rcs) / (exact_distance ** 2)
                
                # Scale attenuation to reasonable values - increase by 5x
                attenuation *= 5e6 #5e6  # Increase from 1e6
                
                # For each chirp, add the delayed and phase-shifted version of the TX signal
                for chirp_idx in range(self.num_chirps):
                    # Calculate exact time vector for this chirp's samples
                    t = np.arange(self.samples_per_chirp) / self.sample_rate
                    
                    # Calculate precise phase shift accounting for continuous time
                    phase_shift = 2 * np.pi * doppler_freq * (chirp_idx * self.chirp_duration + t)
                    
                    # Create the delayed signal with phase shift
                    delayed_signal = np.zeros(self.samples_per_chirp, dtype=np.complex64)
                    
                    # Only copy valid samples (avoid index out of bounds)
                    samples_to_copy = min(self.samples_per_chirp - delay_samples, self.samples_per_chirp)
                    if samples_to_copy > 0 and delay_samples < self.samples_per_chirp:
                        # Copy the delayed portion of the TX signal
                        delayed_signal[delay_samples:delay_samples+samples_to_copy] = tx_signal[chirp_idx, :samples_to_copy]
                        
                        # Apply Doppler phase shift and attenuation
                        delayed_signal *= attenuation * np.exp(1j * phase_shift)
                        
                        # Add to RX signal
                        rx_signal[rx_idx, chirp_idx, :] += delayed_signal
        
        # In perfect mode, eliminate any potential cross-talk between channels
        if perfect_mode:
            # Normalize each channel independently to ensure no cross-channel interference
            for rx_idx in range(self.num_rx):
                # Calculate the maximum amplitude in this channel
                max_amp = np.max(np.abs(rx_signal[rx_idx]))
                if max_amp > 0:
                    # Normalize to maintain relative signal strength within channel
                    # but eliminate any cross-channel effects
                    rx_signal[rx_idx] = rx_signal[rx_idx] / max_amp * max_amp
        
        # Add realistic effects if requested
        if self.apply_realistic_effects:
            rx_signal = self._add_realistic_effects(rx_signal, tx_signal)
        
        return rx_signal
    
    def _add_realistic_effects(self, rx_signal, tx_signal):
        """
        Add realistic effects to the received signal.
        
        Args:
            rx_signal: Received signal
            tx_signal: Transmitted signal
            
        Returns:
            Modified received signal
        """
        # Add direct coupling between TX and RX (crosstalk) - reduce the effect
        crosstalk_isolation_db = 60  # Increase from 30 to 40 dB isolation
        crosstalk_delay_samples = 5  # Small delay
        
        # Convert dB to linear scale
        crosstalk_factor = 10 ** (-crosstalk_isolation_db / 20)
        
        # Add crosstalk to all RX channels
        for rx_idx in range(self.num_rx):
            for chirp_idx in range(self.num_chirps):
                # Create delayed version of TX signal
                delayed_tx = np.zeros(self.samples_per_chirp, dtype=np.complex64)
                if crosstalk_delay_samples < self.samples_per_chirp:
                    samples_to_copy = self.samples_per_chirp - crosstalk_delay_samples
                    delayed_tx[crosstalk_delay_samples:] = tx_signal[chirp_idx, :samples_to_copy]
                
                # Add to RX signal with attenuation
                rx_signal[rx_idx, chirp_idx, :] += delayed_tx * crosstalk_factor
        
        # Add ground clutter - reduce the probability and power
        clutter_probability = 0.02  # Reduce from 0.05 to 0.02
        max_clutter_distance = self.max_range * 0.1  # Reduce from 0.2 to 0.1
        
        # Convert distance to samples
        max_clutter_samples = int(2 * max_clutter_distance * self.sample_rate / self.speed_of_light)
        
        # Add clutter reflections
        for sample_idx in range(min(max_clutter_samples, self.samples_per_chirp)):
            # Random chance of clutter at this range
            if random.random() < clutter_probability:
                # Calculate distance for this sample
                distance = sample_idx * self.speed_of_light / (2 * self.sample_rate)
                
                # Random RCS for clutter - reduce power
                clutter_rcs = random.uniform(0.05, 0.5)  # Reduce from (0.1, 1.0)
                
                # Calculate attenuation - reduce power
                attenuation = np.sqrt(clutter_rcs) / (distance ** 2) * 5e4  # Reduce from 1e5
                
                # Random phase
                phase = random.uniform(0, 2 * np.pi)
                
                # Add to all RX channels with random variations
                for rx_idx in range(self.num_rx):
                    rx_phase_variation = random.uniform(0, 0.1)
                    for chirp_idx in range(self.num_chirps):
                        rx_signal[rx_idx, chirp_idx, sample_idx] += attenuation * np.exp(1j * (phase + rx_phase_variation))
        
        # Add system noise (thermal noise, phase noise, etc.)
        system_noise_power = 1e-6
        system_noise = np.random.normal(0, np.sqrt(system_noise_power/2), rx_signal.shape) + \
                    1j * np.random.normal(0, np.sqrt(system_noise_power/2), rx_signal.shape)
        rx_signal += system_noise
        
        return rx_signal
    
    def _add_noise(self, rx_signal, snr_db):
        """
        Add white Gaussian noise to the received signal.
        
        Args:
            rx_signal: Received signal
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy received signal
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(rx_signal)**2)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate complex noise
        noise = np.random.normal(0, np.sqrt(noise_power/2), rx_signal.shape) + \
               1j * np.random.normal(0, np.sqrt(noise_power/2), rx_signal.shape)
        
        # Add noise to signal
        noisy_signal = rx_signal + noise
        
        return noisy_signal
    
    def _time_to_range_doppler(self, rx_signal, 
                          apply_mti=True,  # Changed default
                          apply_doppler_centering=True,
                          apply_notch_filter=True,
                          notch_width=5,  # New parameter
                          use_blackman_window=True,
                          dynamic_range_db=50):  # New parameter
        """
        Convert time domain signal to range-Doppler map.
        
        Args:
            rx_signal: Received signal with shape [num_rx, num_chirps, samples_per_chirp]
            apply_mti: Whether to apply Moving Target Indication filtering
            apply_doppler_centering: Whether to center the Doppler FFT
            apply_notch_filter: Whether to apply a notch filter to suppress zero-Doppler
            use_blackman_window: Whether to use Blackman-Harris window instead of Hamming
            
        Returns:
            Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
        """
        # Initialize range-Doppler map
        rd_map = np.zeros((2, self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        
        # Create window functions for sidelobe suppression
        # Improved windowing with proper normalization
        if use_blackman_window:
            range_window = np.blackman(self.samples_per_chirp)
            range_window /= np.sum(range_window)  # Normalize window
            doppler_window = np.blackman(self.num_chirps)
            doppler_window /= np.sum(doppler_window)  # Normalize window
        else:
            range_window = np.hamming(self.samples_per_chirp)
            range_window /= np.sum(range_window)  # Normalize window
            doppler_window = np.hamming(self.num_chirps)
            doppler_window /= np.sum(doppler_window)  # Normalize window
        
        # Add CFAR-like processing
        def apply_cfar(magnitude):
            # Simple CA-CFAR implementation
            guard = 2
            training = 4
            threshold = 1.5  # Adjust based on noise level
            
            cfar_output = np.zeros_like(magnitude)
            for i in range(guard, magnitude.shape[0]-guard):
                for j in range(guard, magnitude.shape[1]-guard):
                    # Get training cells
                    training_cells = np.concatenate([
                        magnitude[i-guard-training:i-guard, j-guard-training:j+guard+training],
                        magnitude[i+guard:i+guard+training, j-guard-training:j+guard+training]
                    ])
                    # Calculate threshold
                    noise_level = np.mean(training_cells)
                    if magnitude[i,j] > noise_level * threshold:
                        cfar_output[i,j] = magnitude[i,j]
            return cfar_output

        # Process each RX antenna
        for rx_idx in range(self.num_rx):
            # Apply MTI filtering if requested (subtract consecutive chirps)
            if apply_mti:
                mti_signal = np.zeros_like(rx_signal[rx_idx])
                mti_signal[1:] = rx_signal[rx_idx, 1:] - rx_signal[rx_idx, :-1]
                processed_signal = mti_signal
            else:
                processed_signal = rx_signal[rx_idx]
            
            # Apply windowing to each chirp (along fast-time/samples dimension)
            windowed_signal = processed_signal * range_window[np.newaxis, :]
            
            # Apply range FFT (along fast-time/samples dimension)
            range_fft = np.fft.fft(windowed_signal, n=self.num_range_bins, axis=1)
            
            # Apply windowing to each range bin (along slow-time/chirps dimension)
            windowed_range_fft = range_fft * doppler_window[:, np.newaxis]
            
            # Apply Doppler FFT (along slow-time/chirps dimension)
            doppler_fft = np.fft.fft(windowed_range_fft, n=self.num_doppler_bins, axis=0)
            
            # Shift zero-Doppler to center if requested
            if apply_doppler_centering:
                doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
            
            # Apply notch filter to suppress zero-Doppler if requested
            if apply_notch_filter:
                #notch_width = 3  # Width of notch in bins
                center_bin = self.num_doppler_bins // 2  # Center bin
                
                # Create notch filter
                notch_filter = np.ones(self.num_doppler_bins)
                notch_filter[center_bin-notch_width:center_bin+notch_width+1] = 0
                
                # Apply notch filter to each range bin
                for range_bin in range(doppler_fft.shape[1]):
                    doppler_fft[:, range_bin] *= notch_filter
            
            # Calculate magnitude (convert to dB)
            magnitude = 20 * np.log10(np.abs(doppler_fft) + 1e-10)
            magnitude = apply_cfar(magnitude)  # Apply CFAR detection
            
            # Adjust dynamic range based on noise floor
            noise_floor = np.percentile(magnitude, 10)  # Estimate noise floor
            #ynamic_range_db = 60  # Increased from 40 to 60 dB
            magnitude_norm = np.clip(magnitude, noise_floor, noise_floor + dynamic_range_db)
            magnitude_norm = (magnitude_norm - noise_floor) / dynamic_range_db * 100
            
            # Store magnitude in range-Doppler map
            # First channel: magnitude
            rd_map[0] = magnitude_norm
            
            # Second channel: phase (normalized to [0, 1])
            phase = np.angle(doppler_fft) / (2 * np.pi) + 0.5
            rd_map[1] = phase
            
            # Only process the first RX antenna for the range-Doppler map
            break
        
        return rd_map
    
    def _cfar_detection(self, rd_map):
        """
        Perform CFAR detection on range-Doppler map with improved false alarm control.
        
        Args:
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
            
        Returns:
            List of detected targets with range and Doppler information
        """
        # Convert complex RD map to magnitude
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        # Define CFAR parameters - increase guard cells and training cells
        guard_cells = (3, 3)  # Increased from (2, 2)
        training_cells = (6, 6)  # Increased from (4, 4)
        pfa = 1e-5  # Reduced from 1e-4 for stricter detection
        
        # Initialize CFAR detection map
        cfar_map = np.zeros((self.num_doppler_bins, self.num_range_bins), dtype=bool)
        
        # Apply CFAR detection
        for d_idx in range(self.num_doppler_bins):
            for r_idx in range(self.num_range_bins):
                # Define cell under test (CUT)
                cut_value = rd_magnitude[d_idx, r_idx]
                
                # Define guard and training cell ranges
                d_min = max(0, d_idx - guard_cells[0] - training_cells[0])
                d_max = min(self.num_doppler_bins - 1, d_idx + guard_cells[0] + training_cells[0])
                r_min = max(0, r_idx - guard_cells[1] - training_cells[1])
                r_max = min(self.num_range_bins - 1, r_idx + guard_cells[1] + training_cells[1])
                
                # Extract training cells (excluding guard cells)
                training_region = []
                for di in range(d_min, d_max + 1):
                    for ri in range(r_min, r_max + 1):
                        # Skip guard cells and CUT
                        if (abs(di - d_idx) <= guard_cells[0] and abs(ri - r_idx) <= guard_cells[1]):
                            continue
                        training_region.append(rd_magnitude[di, ri])
                
                # Calculate threshold from training cells
                if len(training_region) > 0:
                    # Use ordered statistic CFAR with higher scaling factor
                    training_region.sort()
                    k = int(len(training_region) * (1 - pfa))
                    threshold = training_region[min(k, len(training_region) - 1)]
                    
                    # Add a scaling factor to increase threshold
                    threshold *= 1.5  # Increase threshold by 50%
                    
                    # Apply threshold
                    cfar_map[d_idx, r_idx] = cut_value > threshold
        
        # Apply additional filtering to remove isolated detections (likely false alarms)
        filtered_cfar_map = np.zeros_like(cfar_map)
        for d_idx in range(1, self.num_doppler_bins-1):
            for r_idx in range(1, self.num_range_bins-1):
                if cfar_map[d_idx, r_idx]:
                    # Count neighbors (3x3 window)
                    neighbor_count = np.sum(cfar_map[d_idx-1:d_idx+2, r_idx-1:r_idx+2]) - 1  # -1 to exclude self
                    # Keep detection only if it has at least one neighbor
                    filtered_cfar_map[d_idx, r_idx] = neighbor_count > 0
        
        # Extract detected targets
        detected_targets = []
        for d_idx in range(self.num_doppler_bins):
            for r_idx in range(self.num_range_bins):
                if filtered_cfar_map[d_idx, r_idx]:
                    # Calculate distance and velocity
                    distance = r_idx * self.range_resolution
                    velocity = (d_idx - self.num_doppler_bins // 2) * self.velocity_resolution
                    
                    # Calculate SNR (approximation)
                    noise_floor = np.median(rd_magnitude)
                    snr_db = 20 * np.log10(rd_magnitude[d_idx, r_idx] / (noise_floor + 1e-10))
                    
                    # Only include targets with sufficient SNR
                    if snr_db > 10.0:  # Minimum SNR threshold
                        # Store target information
                        target = {
                            'range_bin': r_idx,
                            'doppler_bin': d_idx,
                            'distance': distance,
                            'velocity': velocity,
                            'snr': snr_db
                        }
                        detected_targets.append(target)
        
        return detected_targets
    
    def _create_target_mask(self, targets):
        """
        Create ground truth mask for targets.
        
        Args:
            targets: List of target dictionaries
            
        Returns:
            Binary mask with shape [num_doppler_bins, num_range_bins, 1]
        """
        # Initialize target mask
        target_mask = np.zeros((self.num_doppler_bins, self.num_range_bins, 1), dtype=self.precision)
        
        # Create Gaussian-shaped targets in the mask
        for target in targets:
            # Calculate range and Doppler bin
            range_bin = int(target['distance'] / self.range_resolution)
            doppler_bin = int(self.num_doppler_bins // 2 + target['velocity'] / self.velocity_resolution)
            
            # Ensure bins are within valid range
            if (0 <= range_bin < self.num_range_bins and 
                0 <= doppler_bin < self.num_doppler_bins):
                
                # Create Gaussian-shaped target (to account for target spread)
                sigma_range = 1.0  # Standard deviation in range dimension
                sigma_doppler = 1.0  # Standard deviation in Doppler dimension
                
                # Define region around target
                r_min = max(0, int(range_bin - 3*sigma_range))
                r_max = min(self.num_range_bins - 1, int(range_bin + 3*sigma_range))
                d_min = max(0, int(doppler_bin - 3*sigma_doppler))
                d_max = min(self.num_doppler_bins - 1, int(doppler_bin + 3*sigma_doppler))
                
                # Fill target mask with Gaussian shape
                for r in range(r_min, r_max + 1):
                    for d in range(d_min, d_max + 1):
                        # Calculate Gaussian value
                        exponent = -0.5 * ((r - range_bin) / sigma_range)**2 - 0.5 * ((d - doppler_bin) / sigma_doppler)**2
                        value = np.exp(exponent)
                        
                        # Update mask (use maximum value in case of overlapping targets)
                        target_mask[d, r, 0] = max(target_mask[d, r, 0], value)
        
        # Threshold mask to create binary target mask
        target_mask = (target_mask > 0.1).astype(self.precision)
        
        return target_mask
    
    def _visualize_sample(self, sample_idx, tx_signal, rx_signal, rd_map, targets, detection_results):
        """
        Visualize radar data for a single sample.
        
        Args:
            sample_idx: Sample index
            tx_signal: Transmitted signal
            rx_signal: Received signal
            rd_map: Range-Doppler map
            targets: List of ground truth targets
            detection_results: List of detected targets
        """
        # Create directory for visualizations
        os.makedirs(os.path.join(self.save_path, 'visualizations'), exist_ok=True)
        
        # Time vector for one chirp (in microseconds)
        t = np.linspace(0, self.chirp_duration * 1e6, self.samples_per_chirp)
        
        # Figure 1: TX/RX Signal Analysis
        fig1 = plt.figure(figsize=(12, 10))
        plt.suptitle(f"TX/RX Signal Analysis - Sample {sample_idx}", fontsize=16)
        
        # TX signal time domain
        plt.subplot(2, 2, 1)
        plt.plot(t, np.real(tx_signal[0]), 'b-', label='Real')
        plt.plot(t, np.imag(tx_signal[0]), 'r-', label='Imag')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('TX Signal (Time Domain)')
        plt.legend()
        plt.grid(True)
        
        # TX signal instantaneous frequency
        plt.subplot(2, 2, 2)
        # Calculate instantaneous frequency by taking the derivative of the phase
        inst_phase_tx = np.unwrap(np.angle(tx_signal[0]))
        inst_freq_tx = np.diff(inst_phase_tx) / (2 * np.pi * (self.chirp_duration / self.samples_per_chirp))
        # Pad with the first value to maintain array size
        inst_freq_tx = np.concatenate(([inst_freq_tx[0]], inst_freq_tx))
        
        plt.plot(t, inst_freq_tx / 1e6, 'g-')  # Convert to MHz for display
        plt.xlabel('Time (μs)')
        plt.ylabel('Frequency (MHz)')
        plt.title('TX FMCW Instantaneous Frequency')
        plt.grid(True)
        
        # RX signal time domain (first RX antenna, first chirp)
        plt.subplot(2, 2, 3)
        plt.plot(t, np.real(rx_signal[0, 0]), 'b-', label='Real')
        plt.plot(t, np.imag(rx_signal[0, 0]), 'r-', label='Imag')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('RX Signal (Time Domain)')
        plt.legend()
        plt.grid(True)
        
        # RX signal instantaneous frequency
        plt.subplot(2, 2, 4)
        # Calculate instantaneous frequency by taking the derivative of the phase
        inst_phase_rx = np.unwrap(np.angle(rx_signal[0, 0]))
        inst_freq_rx = np.diff(inst_phase_rx) / (2 * np.pi * (self.chirp_duration / self.samples_per_chirp))
        # Pad with the first value to maintain array size
        inst_freq_rx = np.concatenate(([inst_freq_rx[0]], inst_freq_rx))
        
        plt.plot(t, inst_freq_rx / 1e6, 'g-')  # Convert to MHz for display
        plt.xlabel('Time (μs)')
        plt.ylabel('Frequency (MHz)')
        plt.title('RX FMCW Instantaneous Frequency')
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(self.save_path, 'visualizations', f'sample_{sample_idx}_signals.png'))
        plt.close(fig1)
        
        # Figure 2: Range-Doppler Map with Ground Truth
        fig2 = plt.figure(figsize=(12, 10))
        plt.suptitle(f"Range-Doppler Map with Ground Truth - Sample {sample_idx}", fontsize=16)
        
                # 2D Range-Doppler map
        plt.subplot(2, 1, 1)
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)
        vmin = np.max(rd_db) - 40  # Dynamic range of 40 dB
        plt.imshow(rd_db, aspect='auto', cmap='jet', vmin=vmin)
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        plt.title('Range-Doppler Map')
        
        # Print target information for debugging
        print(f"\nTarget Information for Sample {sample_idx}:")
        for i, target in enumerate(targets):
            print(f"Target {i+1}:")
            print(f"  Distance: {target['distance']:.2f} m")
            print(f"  Velocity: {target['velocity']:.2f} m/s")
            print(f"  RCS: {target['rcs']:.2f} m²")
            
            # Calculate range and Doppler bins
            range_bin = int(target['distance'] / self.range_resolution)
            doppler_bin = int(self.num_doppler_bins // 2 + target['velocity'] / self.velocity_resolution)
            print(f"  Range Bin: {range_bin}")
            print(f"  Doppler Bin: {doppler_bin}")
            print(f"  In Range? {0 <= range_bin < self.num_range_bins}")
            print(f"  In Doppler? {0 <= doppler_bin < self.num_doppler_bins}")
            
            # Check if target is within valid range and Doppler bins
            if (0 <= range_bin < self.num_range_bins and 
                0 <= doppler_bin < self.num_doppler_bins):
                # Plot target with larger marker and different color for better visibility
                plt.plot(range_bin, doppler_bin, 'ro', markersize=10, markeredgecolor='white')
                
                # Add text with target information
                # Position text to avoid overlap and ensure visibility
                plt.text(range_bin + 2, doppler_bin, 
                      f"Target {i+1}\nR: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s\nRCS: {target['rcs']:.1f}m²", 
                      color='white', fontsize=9, backgroundcolor='black',
                      bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round'))
            else:
                print(f"  WARNING: Target {i+1} is outside the visible range-Doppler map!")
                
        # Add a note if no targets are in range
        if len(targets) == 0:
            plt.text(self.num_range_bins//2, self.num_doppler_bins//2, 
                  "No targets in this scene", 
                  color='white', fontsize=12, backgroundcolor='red',
                  ha='center', va='center')
        
        # 3D Range-Doppler map
        ax = fig2.add_subplot(2, 1, 2, projection='3d')
        X, Y = np.meshgrid(np.arange(self.num_range_bins), np.arange(self.num_doppler_bins))
        surf = ax.plot_surface(X, Y, rd_db, cmap='jet', linewidth=0, antialiased=True)
        ax.set_xlabel('Range Bin')
        ax.set_ylabel('Doppler Bin')
        ax.set_zlabel('Magnitude (dB)')
        ax.set_title('3D Range-Doppler Map with Ground Truth')
        
        # Add ground truth targets to 3D plot
        for target in targets:
            range_bin = int(target['distance'] / self.range_resolution)
            doppler_bin = int(self.num_doppler_bins // 2 + target['velocity'] / self.velocity_resolution)
            
            if (0 <= range_bin < self.num_range_bins and 
                0 <= doppler_bin < self.num_doppler_bins):
                # Get z-value at target location
                z_val = rd_db[doppler_bin, range_bin]
                ax.scatter([range_bin], [doppler_bin], [z_val], color='r', s=50, marker='o')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(self.save_path, 'visualizations', f'sample_{sample_idx}_rd_map.png'))
        plt.close(fig2)
        
        # Figure 3: CFAR Detection vs Ground Truth
        fig3 = plt.figure(figsize=(12, 10))
        plt.suptitle(f"CFAR Detection vs Ground Truth - Sample {sample_idx}", fontsize=16)
        
        # Ground truth target mask
        plt.subplot(2, 1, 1)
        target_mask = self._create_target_mask(targets)
        plt.imshow(target_mask[:, :, 0], aspect='auto', cmap='gray')
        plt.colorbar(label='Target Presence')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        plt.title('Ground Truth Target Mask')
        
        # Add ground truth targets
        for target in targets:
            range_bin = int(target['distance'] / self.range_resolution)
            doppler_bin = int(self.num_doppler_bins // 2 + target['velocity'] / self.velocity_resolution)
            
            if (0 <= range_bin < self.num_range_bins and 
                0 <= doppler_bin < self.num_doppler_bins):
                plt.plot(range_bin, doppler_bin, 'ro', markersize=8)
                plt.text(range_bin + 1, doppler_bin + 1, 
                      f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                      color='white', fontsize=8, backgroundcolor='black')
        
        # CFAR detection results
        plt.subplot(2, 1, 2)
        cfar_map = np.zeros((self.num_doppler_bins, self.num_range_bins))
        for target in detection_results:
            cfar_map[target['doppler_bin'], target['range_bin']] = 1
        plt.imshow(cfar_map, aspect='auto', cmap='gray')
        plt.colorbar(label='Detection')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        plt.title('CFAR Detection Results')
        
        # Add detected targets
        for target in detection_results:
            plt.plot(target['range_bin'], target['doppler_bin'], 'bo', markersize=8)
            plt.text(target['range_bin'] + 1, target['doppler_bin'] + 1, 
                  f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s\nSNR: {target['snr']:.1f}dB", 
                  color='white', fontsize=8, backgroundcolor='blue')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(self.save_path, 'visualizations', f'sample_{sample_idx}_detection.png'))
        plt.close(fig3)
    
    def save_dataset(self, dataset, format='hdf5'):
        """
        Save the generated dataset to disk.
        
        Args:
            dataset: Dictionary containing the dataset
            format: Format to save the data ('hdf5' or 'numpy')
        """
        print(f"Saving dataset to {self.save_path}...")
        
        if format.lower() == 'hdf5':
            # Save as HDF5 file
            import h5py
            
            file_path = os.path.join(self.save_path, 'raytracing_radar_dataset.h5')
            with h5py.File(file_path, 'w') as f:
                # Save time domain data
                f.create_dataset('time_domain_data', data=dataset['time_domain_data'], 
                                compression='gzip', compression_opts=9)
                
                # Save range-Doppler maps
                f.create_dataset('range_doppler_maps', data=dataset['range_doppler_maps'], 
                                compression='gzip', compression_opts=9)
                
                # Save target masks
                f.create_dataset('target_masks', data=dataset['target_masks'], 
                                compression='gzip', compression_opts=9)
                
                # Save metadata
                metadata_grp = f.create_group('metadata')
                metadata_grp.attrs['num_samples'] = self.num_samples
                metadata_grp.attrs['num_range_bins'] = self.num_range_bins
                metadata_grp.attrs['num_doppler_bins'] = self.num_doppler_bins
                metadata_grp.attrs['sample_rate'] = self.sample_rate
                metadata_grp.attrs['bandwidth'] = self.bandwidth
                metadata_grp.attrs['center_freq'] = self.center_freq
                metadata_grp.attrs['chirp_duration'] = self.chirp_duration
                metadata_grp.attrs['num_chirps'] = self.num_chirps
                metadata_grp.attrs['num_rx'] = self.num_rx
                metadata_grp.attrs['num_tx'] = self.num_tx
                metadata_grp.attrs['range_resolution'] = self.range_resolution
                metadata_grp.attrs['velocity_resolution'] = self.velocity_resolution
                
                # Save target information as JSON
                import json
                target_info_json = json.dumps(dataset['target_info'])
                metadata_grp.attrs['target_info'] = target_info_json
                
                # Save detection results as JSON
                detection_results_json = json.dumps(dataset['detection_results'])
                metadata_grp.attrs['detection_results'] = detection_results_json
            
            print(f"Dataset saved to {file_path}")
        
        elif format.lower() == 'numpy':
            # Save as NumPy files
            os.makedirs(os.path.join(self.save_path, 'numpy'), exist_ok=True)
            
            # Save time domain data
            np.save(os.path.join(self.save_path, 'numpy', 'time_domain_data.npy'), 
                   dataset['time_domain_data'])
            
            # Save range-Doppler maps
            np.save(os.path.join(self.save_path, 'numpy', 'range_doppler_maps.npy'), 
                   dataset['range_doppler_maps'])
            
            # Save target masks
            np.save(os.path.join(self.save_path, 'numpy', 'target_masks.npy'), 
                   dataset['target_masks'])
            
            # Save target information
            np.save(os.path.join(self.save_path, 'numpy', 'target_info.npy'), 
                   np.array(dataset['target_info'], dtype=object))
            
            # Save detection results
            np.save(os.path.join(self.save_path, 'numpy', 'detection_results.npy'), 
                   np.array(dataset['detection_results'], dtype=object))
            
            # Save metadata
            metadata = {
                'num_samples': self.num_samples,
                'num_range_bins': self.num_range_bins,
                'num_doppler_bins': self.num_doppler_bins,
                'sample_rate': self.sample_rate,
                'bandwidth': self.bandwidth,
                'center_freq': self.center_freq,
                'chirp_duration': self.chirp_duration,
                'num_chirps': self.num_chirps,
                'num_rx': self.num_rx,
                'num_tx': self.num_tx,
                'range_resolution': self.range_resolution,
                'velocity_resolution': self.velocity_resolution
            }
            np.save(os.path.join(self.save_path, 'numpy', 'metadata.npy'), metadata)
            
            print(f"Dataset saved to {os.path.join(self.save_path, 'numpy')}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'hdf5' or 'numpy'.")
    
    def load_dataset(self, file_path):
        """
        Load a previously saved dataset.
        
        Args:
            file_path: Path to the saved dataset
            
        Returns:
            Dictionary containing the loaded dataset
        """
        print(f"Loading dataset from {file_path}...")
        
        if file_path.endswith('.h5'):
            # Load HDF5 file
            import h5py
            
            with h5py.File(file_path, 'r') as f:
                # Load time domain data
                time_domain_data = f['time_domain_data'][:]
                
                # Load range-Doppler maps
                range_doppler_maps = f['range_doppler_maps'][:]
                
                # Load target masks
                target_masks = f['target_masks'][:]
                
                # Load metadata
                metadata = f['metadata']
                self.num_samples = metadata.attrs['num_samples']
                self.num_range_bins = metadata.attrs['num_range_bins']
                self.num_doppler_bins = metadata.attrs['num_doppler_bins']
                self.sample_rate = metadata.attrs['sample_rate']
                self.bandwidth = metadata.attrs['bandwidth']
                self.center_freq = metadata.attrs['center_freq']
                self.chirp_duration = metadata.attrs['chirp_duration']
                self.num_chirps = metadata.attrs['num_chirps']
                self.num_rx = metadata.attrs['num_rx']
                self.num_tx = metadata.attrs['num_tx']
                self.range_resolution = metadata.attrs['range_resolution']
                self.velocity_resolution = metadata.attrs['velocity_resolution']
                
                # Load target information
                import json
                target_info = json.loads(metadata.attrs['target_info'])
                
                # Load detection results
                detection_results = json.loads(metadata.attrs['detection_results'])
            
            # Create dataset dictionary
            dataset = {
                'time_domain_data': time_domain_data,
                'range_doppler_maps': range_doppler_maps,
                'target_masks': target_masks,
                'target_info': target_info,
                'detection_results': detection_results
            }
            
            print(f"Dataset loaded successfully with {self.num_samples} samples.")
            return dataset
        
        elif os.path.isdir(file_path) and os.path.exists(os.path.join(file_path, 'metadata.npy')):
            # Load NumPy files
            # Load time domain data
            time_domain_data = np.load(os.path.join(file_path, 'time_domain_data.npy'))
            
            # Load range-Doppler maps
            range_doppler_maps = np.load(os.path.join(file_path, 'range_doppler_maps.npy'))
            
            # Load target masks
            target_masks = np.load(os.path.join(file_path, 'target_masks.npy'))
            
            # Load target information
            target_info = np.load(os.path.join(file_path, 'target_info.npy'), allow_pickle=True).tolist()
            
            # Load detection results
            detection_results = np.load(os.path.join(file_path, 'detection_results.npy'), allow_pickle=True).tolist()
            
            # Load metadata
            metadata = np.load(os.path.join(file_path, 'metadata.npy'), allow_pickle=True).item()
            self.num_samples = metadata['num_samples']
            self.num_range_bins = metadata['num_range_bins']
            self.num_doppler_bins = metadata['num_doppler_bins']
            self.sample_rate = metadata['sample_rate']
            self.bandwidth = metadata['bandwidth']
            self.center_freq = metadata['center_freq']
            self.chirp_duration = metadata['chirp_duration']
            self.num_chirps = metadata['num_chirps']
            self.num_rx = metadata['num_rx']
            self.num_tx = metadata['num_tx']
            self.range_resolution = metadata['range_resolution']
            self.velocity_resolution = metadata['velocity_resolution']
            
            # Create dataset dictionary
            dataset = {
                'time_domain_data': time_domain_data,
                'range_doppler_maps': range_doppler_maps,
                'target_masks': target_masks,
                'target_info': target_info,
                'detection_results': detection_results
            }
            
            print(f"Dataset loaded successfully with {self.num_samples} samples.")
            return dataset
        
        else:
            raise ValueError(f"Unsupported file format or directory structure: {file_path}")


# Add a main function to test the ray-tracing radar dataset
if __name__ == "__main__":
    # Create ray-tracing radar dataset
    radar_dataset = RayTracingRadarDataset(
        save_path='data/raytracing_radar',
        precision='float32'
    )
    
    # Generate dataset
    dataset = radar_dataset.generate_dataset(visualize=True)
    
    # Save dataset
    radar_dataset.save_dataset(dataset, format='hdf5')
    
    # Test loading the dataset
    loaded_dataset = radar_dataset.load_dataset(os.path.join('data/raytracing_radar', 'raytracing_radar_dataset.h5'))
    
    print("Ray-tracing radar dataset generation and testing complete!")