import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sympy.sets.sets import false
from tqdm import tqdm
from scipy.signal import chirp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import scipy.signal
#from scipy.signal import blackmanharris
from tqdm import tqdm
from torch.utils.data import Dataset
from AIradar_processing import RadarProcessing
IMG_FORMAT=".pdf" #".png"
import time
from datautil import *

class RadarDataset(Dataset):
    # In the RadarDataset class initialization, update the default parameters
    def __init__(self, 
                 num_samples=100,
                 num_range_bins=128,  # Increased from 64 for better range resolution
                 num_doppler_bins=16,  # Increased from 12 for better velocity resolution
                 sample_rate=50e6, #1.5e6,    # reduce from 15MHz to 1.5MHz Adjusted to match hardware constraints
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
        self.signal_freq = signal_freq

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
        
        perf_params = calculate_radar_parameters(
            sample_rate=self.sample_rate,
            chirp_duration=self.chirp_duration,
            center_freq=self.center_freq,
            bandwidth=self.bandwidth,
            num_chirps=self.num_chirps
        )
        self.samples_per_chirp = perf_params["samples_per_chirp"]
        self.total_samples_per_chirp = self.samples_per_chirp
        self.range_resolution = perf_params["range_resolution"]
        self.min_range = self.range_resolution
        self.max_range = perf_params["max_range"]
        self.velocity_resolution = perf_params["velocity_resolution"]
        self.max_velocity = perf_params["max_velocity"]
        self.wavelength = perf_params["wavelength"]
        self.max_unambiguous_velocity = perf_params["max_unambiguous_velocity"]
        self.fmcw_slope = perf_params["fmcw_slope"]
        self.f_beat_max = perf_params["f_beat_max"]
        self.nyquist_freq = perf_params["nyquist_freq"]
        self.speed_of_light = 3e8  # Speed of light in m/s
        #Set num_range_bins and num_doppler_bins to match the FFT sizes unless you have a specific reason to use fewer bins (e.g., for memory or speed). This ensures that your processed data fully utilizes the frequency resolution provided by the FFT.
        #If you want to limit the number of bins for memory or display reasons, you can set num_range_bins and num_doppler_bins to be less than or equal to the FFT sizes, but you will only use a subset of the FFT output.
        self.num_range_bins = perf_params["range_fft_size"]
        self.num_doppler_bins = perf_params["doppler_fft_size"]

        print("\n=== Radar System Parameters ===")
        print(f"✅ Maximum Range       : {self.max_range:.2f} m")
        print(f"✅ Range Resolution    : {self.range_resolution:.2f} m")
        print(f"✅ Maximum Velocity    : {self.max_velocity:.2f} m/s (Target)")
        print(f"✅ Max Unambig. Velocity: {self.max_unambiguous_velocity:.2f} m/s (Actual)")
        print(f"✅ Velocity Resolution : {self.velocity_resolution:.2f} m/s")
        print(f"✅ Center Frequency    : {self.center_freq/1e9:.2f} GHz")
        print(f"✅ Bandwidth           : {self.bandwidth/1e6:.2f} MHz")
        print(f"✅ Sample Rate         : {self.sample_rate/1e6:.1f} MHz")
        print(f"✅ Chirp Duration      : {self.chirp_duration*1e6:.2f} μs")
        print(f"✅ FMCW Slope          : {self.fmcw_slope/1e12:.2f} THz/s")
        print(f"✅ Actual Sweep        : {(self.fmcw_slope * self.chirp_duration)/1e6:.2f} MHz")
        print(f"✅ Samples per Chirp   : {self.samples_per_chirp}")
        print(f"✅ Number of Chirps    : {self.num_chirps}")
        print(f"✅ Number of RX Antennas: {self.num_rx}")
        print(f"✅ Number of TX Antennas: {self.num_tx}")
        print(f"✅ Beat Frequency Max  : {self.f_beat_max/1e6:.2f} MHz (Nyquist: {self.nyquist_freq/1e6:.2f} MHz)")
        print(f"✅ Frequency Wraparound: {'No' if self.f_beat_max <= self.nyquist_freq else 'Yes'}")
        print("=====================================\n")
        
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
            print(f"Loading radar data from {datapath}")
            #self._load_data(datapath)
        else:
            print("Generating new radar data")
            #self.generate_radar_data(save_data=True, format=savedataformat)
            self.generate_dataset(visualize=True)

    def generate_dataset(self, fixedsnr_db=40, visualize=True):
        """
        Generate a radar dataset using ray-tracing simulation.
        
        Args:
            visualize: Whether to visualize the results
            
        Returns:
            Dictionary containing the generated dataset
        """
        print(f"Generating {self.num_samples} radar samples using ray-tracing simulation...")
        if visualize:
            savevis_path=os.path.join(self.save_path, "visualization")
            os.makedirs(savevis_path, exist_ok=True)

        # Calculate the actual flattened length including idle time
        flattened_length = self.num_chirps * self.total_samples_per_chirp
        
        dataset = {
            'time_domain_data': np.zeros((self.num_samples, self.num_rx, flattened_length, 2), 
                                        dtype=self.precision),
            'range_doppler_maps': np.zeros((self.num_samples, self.num_rx, 2, self.num_doppler_bins, 
                                           self.num_range_bins), dtype=self.precision),
            'target_masks': np.zeros((self.num_samples, self.num_doppler_bins, 
                                     self.num_range_bins, 1), dtype=self.precision),
            'target_info': [],
            'detection_results': []
        }
        
        # Generate samples
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            tx_signal = self._generate_tx_signal(num_chirps=self.num_chirps,
                    total_samples_per_chirp=self.total_samples_per_chirp,
                    active_samples=self.samples_per_chirp,
                    sample_rate=self.sample_rate,
                    slope=self.fmcw_slope, return_full=True, window_type=None) #(128, 400) complex
            #(51200,) 128*400
            if visualize:
                # Extract a single chirp from the full TX signal for visualization
                tx_chirp = tx_signal[:self.samples_per_chirp]  # Get first chirp's active samples
                # Visualize the time and spectrum of the TX chirp
                plot_signal_time_and_spectrum(
                    signal=tx_chirp,
                    sample_rate=self.sample_rate,
                    total_duration=self.chirp_duration,
                    title_prefix="TX Chirp",
                    bandwidth=self.bandwidth,
                    center_freq=self.center_freq,
                    textstr=None,
                    normalize=False,
                    save_path=os.path.join(savevis_path, f"txchirp_{i}IMG_FORMAT"),
                    draw_window = False
                )
            # Generate random targets
            targets = self._generate_random_targets()
            # Perform ray-tracing simulation
            rx_signal = self._ray_tracing_simulation(tx_signal, targets, perfect_mode=True, flatten_output=True)
            #The shape should be (4, 153600) [num_rx, num_chirps*samples_per_chirp]

            # Add noise to the received signal (even in perfect mode, we need some minimal noise)
            if fixedsnr_db is not None:
                snr_db = fixedsnr_db
            else:
                snr_db = random.uniform(self.snr_min, self.snr_max)
            rx_signal = self._add_noise(rx_signal, snr_db)
            if visualize:
                # Visualize the time and spectrum of the received signal
                rx_signal_chirp = rx_signal[0, :self.samples_per_chirp] 
                plot_signal_time_and_spectrum(
                        signal=rx_signal_chirp,
                        sample_rate=self.sample_rate,
                        total_duration=self.chirp_duration,
                        title_prefix="RX Chirp",
                        bandwidth=self.bandwidth,
                        center_freq=self.center_freq,
                        textstr=None,
                        normalize=False,
                        save_path=os.path.join(savevis_path, f"rxchirp_{i}IMG_FORMAT"),
                        draw_window = False
                    )

            # Demodulate the signal to baseband
            beat_signal_list = []
            for rx_idx in range(self.num_rx):
                beat = self.fmcw_demodulate(
                    tx_full=tx_signal,
                    rx_full=rx_signal[rx_idx, :],
                    total_samples_per_chirp=self.total_samples_per_chirp,
                    beat_samples_per_chirp=self.samples_per_chirp,
                    num_chirps=self.num_chirps
                )
                beat_signal_list.append(beat)
            beat_signal = np.stack(beat_signal_list, axis=0)  # Shape: (num_rx, num_chirps, samples_per_chirp)
            #Beat signal with shape (4, 128, 1000) [num_rx, num_chirps, samples_per_chirp]
            if visualize: # and i == 0:  # Only for the first sample to avoid too many plots
                beat_signal_chirp = beat_signal[0, 0, :] 
                plot_signal_time_and_spectrum(
                        signal=beat_signal_chirp,
                        sample_rate=self.sample_rate,
                        total_duration=self.chirp_duration,
                        title_prefix="Beat Chirp",
                        textstr=None,
                        normalize=False,
                        save_path=os.path.join(savevis_path, f"beatchirp_{i}IMG_FORMAT"),
                        draw_window = False
                    )
            
            # Process the received signal to generate range-Doppler map
            self.apply_doppler_centering = True
            rd_map = self._time_to_range_doppler(
                rx_signal=beat_signal,  # [num_rx, num_chirps, samples_per_chirp]
                num_chirps=self.num_chirps,
                samples_per_chirp=self.total_samples_per_chirp,
                num_doppler_bins=self.num_doppler_bins,
                num_range_bins=self.num_range_bins,
                apply_mti=False,               # Enable MTI for stationary target suppression
                apply_doppler_centering=self.apply_doppler_centering,
                apply_notch_filter=False,
                notch_width=5,               # Increase notch width from default 3
                use_blackman_window=False,
                dynamic_range_db=0          # Increase from 40dB
            )
            #rd_map is [num_rx, 2(real+imaginary), num_doppler_bins, num_range_bins]
            if visualize:
                plot_range_doppler_map_with_ground_truth(
                    rd_map=rd_map[0,:],
                    targets=targets,  # Make sure 'targets' is defined in your context
                    range_resolution=self.range_resolution,
                    velocity_resolution=self.velocity_resolution,
                    num_range_bins=self.num_range_bins,
                    num_doppler_bins=self.num_doppler_bins,
                    title_prefix=f"Range-Doppler Map Sample {i}",
                    save_path=os.path.join(savevis_path, f"rdmap_{i}IMG_FORMAT")
                )

            # Perform target detection using CFAR for Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
            #[2, num_doppler_bins, num_range_bins]
            detection_results = self._cfar_detection(rd_map[0,:])#(2, 128, 256)
            
            if targets is not None:
                target_mask = self._create_target_mask(targets) #(128, 256, 1)
                # Store the partially flattened rx_signal
                dataset['time_domain_data'][i, :, :, 0] = np.real(rx_signal)
                dataset['time_domain_data'][i, :, :, 1] = np.imag(rx_signal)
                dataset['range_doppler_maps'][i] = rd_map #[num_rx, 2(real+imaginary), num_doppler_bins, num_range_bins]
                dataset['target_masks'][i] = target_mask
                dataset['target_info'].append(targets)
                dataset['detection_results'].append(detection_results)
                
            
        print("Dataset generation complete!")
        return dataset
    
    def _generate_tx_signal(self, num_chirps, total_samples_per_chirp, active_samples, sample_rate, slope, tx_power=1.0, edge_ratio=0.1, window_type='edge', return_full=False):
        """
        Generate phase-continuous FMCW chirp signal with proper Doppler handling,
        optional edge windowing and idle gaps.

        Args:
            num_chirps: Number of chirps to generate
            total_samples_per_chirp: Total samples per chirp including idle time
            active_samples: Number of active samples in each chirp
            sample_rate: Sampling rate in Hz
            slope: Chirp slope in Hz/s
            tx_power: Transmission power scale (float)
            edge_ratio: Proportion of chirp to taper at each edge (0–0.5)
            window_type: Windowing function ('edge', 'hann', 'hamming', or None)
            return_full: If True, return full TX signal; else per-chirp array

        Returns:
            - If return_full=False: [num_chirps, samples_per_chirp] complex array
            - If return_full=True: 1D waveform with all chirps concatenated
        """
        # Create continuous time vector for entire frame
        t_frame = np.arange(num_chirps * total_samples_per_chirp) / sample_rate
        
        # Calculate chirp duration from active samples
        chirp_duration = active_samples / sample_rate
        
        # Generate phase-continuous signal
        phase = 2 * np.pi * (
            0.5 * slope * (t_frame % chirp_duration)**2 +  # Chirp phase
            slope * chirp_duration * (t_frame // chirp_duration) * (t_frame % chirp_duration)  # Phase accumulation
        )
        
        # Create continuous signal
        continuous_signal = np.exp(1j * phase)
        
        # Apply windowing to active portion of each chirp
        if window_type:
            # Create window function
            if window_type == 'hann':
                window = np.hanning(active_samples)
            elif window_type == 'hamming':
                window = np.hamming(active_samples)
            elif window_type == 'edge':
                # Enforce min edge length
                min_edge_len = 16
                edge_len = max(int(edge_ratio * active_samples), min_edge_len)
                
                # Ensure even length for symmetric hann windowing
                edge_len = edge_len if edge_len % 2 == 0 else edge_len + 1
                total_taper_len = 2 * edge_len
                
                if total_taper_len >= active_samples:
                    # Fall back to full Hann window if taper too large
                    window = np.hanning(active_samples)
                else:
                    hann_win = np.hanning(total_taper_len)
                    rise = hann_win[:edge_len]
                    fall = hann_win[edge_len:]
                    flat = np.ones(active_samples - total_taper_len)
                    window = np.concatenate([rise, flat, fall])
            else:  # No windowing
                window = np.ones(active_samples)
            
            # Apply window to each chirp's active portion
            for i in range(num_chirps):
                start_idx = i * total_samples_per_chirp
                continuous_signal[start_idx:start_idx + active_samples] *= window
        
        # Apply power scaling
        scale = np.sqrt(tx_power)
        continuous_signal *= scale
        
        # Create zero padding for idle time in each chirp
        if total_samples_per_chirp > active_samples:
            # Create output with proper shape including idle time
            if return_full:
                # Single continuous waveform
                tx_full = np.zeros(num_chirps * total_samples_per_chirp, dtype=np.complex128)
                for i in range(num_chirps):
                    start_idx = i * total_samples_per_chirp
                    tx_full[start_idx:start_idx + active_samples] = continuous_signal[i * total_samples_per_chirp:i * total_samples_per_chirp + active_samples]
                return tx_full
            else:
                # Array of individual chirps
                tx_signal = np.zeros((num_chirps, total_samples_per_chirp), dtype=np.complex128)
                for i in range(num_chirps):
                    start_idx = i * total_samples_per_chirp
                    tx_signal[i, :active_samples] = continuous_signal[start_idx:start_idx + active_samples]
                return tx_signal
        else:
            # No idle time case
            if return_full:
                return continuous_signal
            else:
                return continuous_signal.reshape(num_chirps, total_samples_per_chirp)

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
    
    def _ray_tracing_simulation(self, tx_signal, targets, perfect_mode=False, flatten_output=False):
        """
        Perform ray-tracing simulation to generate received signals.
        
        Args:
            tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp] or flattened 1D array
            targets: List of target dictionaries
            perfect_mode: If True, uses a single fixed target for ideal simulation
            flatten_output: If True, returns a flattened 1D array similar to simulate_single_target_echo
            
        Returns:
            Complex RX signal with shape [num_rx, num_chirps, samples_per_chirp] or flattened 1D array
        """
        # Check if tx_signal is flattened and reshape if needed
        tx_is_flattened = tx_signal.ndim == 1
        if tx_is_flattened:
            # Reshape flattened tx_signal to [num_chirps, samples_per_chirp]
            tx_signal_reshaped = tx_signal.reshape(self.num_chirps, -1)
            samples_per_chirp = tx_signal_reshaped.shape[1]
        else:
            tx_signal_reshaped = tx_signal
            samples_per_chirp = tx_signal.shape[1]
        
        # Initialize RX signal (all zeros)
        rx_signal = np.zeros((self.num_rx, self.num_chirps, samples_per_chirp), 
                            dtype=np.complex64)
        
        # Define RX antenna positions (simple linear array along x-axis)
        rx_positions = []
        rx_spacing = self.wavelength / 2  # Half-wavelength spacing
        for rx_idx in range(self.num_rx):
            rx_positions.append((rx_idx * rx_spacing, 0, 0))
        
        # In perfect mode, override targets with a single ideal target if no targets provided
        if perfect_mode and (targets is None or len(targets) == 0):
            # Create a single fixed target at 50m with 10m/s velocity and high RCS
            perfect_target = {
                'distance': 50.0,  # 50 meters
                'velocity': 10.0,  # 10 m/s
                'rcs': 20.0,       # 20 dBsm (high RCS for clear visibility)
                'position': (50.0, 0, 0)  # Position in 3D space (x, y, z)
            }
            targets = [perfect_target]
        
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
                attenuation = np.sqrt(rcs) / (exact_distance ** 2)
                
                # Scale attenuation to reasonable values
                attenuation *= 5e6
                
                # For each chirp, add the delayed and phase-shifted version of the TX signal
                for chirp_idx in range(self.num_chirps):
                    # Calculate exact time vector for this chirp's samples
                    t = np.arange(samples_per_chirp) / self.sample_rate
                    
                    # Calculate precise phase shift accounting for continuous time
                    phase_shift = 2 * np.pi * doppler_freq * (chirp_idx * self.chirp_duration + t)
                    
                    # Create the delayed signal with phase shift
                    delayed_signal = np.zeros(samples_per_chirp, dtype=np.complex64)
                    
                    # Only copy valid samples (avoid index out of bounds)
                    samples_to_copy = min(samples_per_chirp - delay_samples, samples_per_chirp)
                    if samples_to_copy > 0 and delay_samples < samples_per_chirp:
                        # Copy the delayed portion of the TX signal
                        delayed_signal[delay_samples:delay_samples+samples_to_copy] = tx_signal_reshaped[chirp_idx, :samples_to_copy]
                        
                        # Apply Doppler phase shift and attenuation
                        delayed_signal *= attenuation * np.exp(1j * phase_shift)
                        
                        # Add to RX signal
                        rx_signal[rx_idx, chirp_idx, :] += delayed_signal
        
        # Add realistic effects if requested and not in perfect mode
        if self.apply_realistic_effects and not perfect_mode:
            rx_signal = self._add_realistic_effects(rx_signal, tx_signal_reshaped)
        
        # If flatten_output is True, flatten the rx_signal to match simulate_single_target_echo format
        if flatten_output:
            # Create a flattened array that includes all RX antennas
            # The shape should be [num_rx, num_chirps*samples_per_chirp]
            rx_flattened = np.zeros((self.num_rx, self.num_chirps * samples_per_chirp), dtype=np.complex64)
            
            # Reshape each RX antenna's data
            for rx_idx in range(self.num_rx):
                rx_flattened[rx_idx] = rx_signal[rx_idx].flatten()
                
            return rx_flattened
        
        return rx_signal

    def _generate_fmcw_chirp(self, chirp_idx):
        """Generate a single FMCW chirp signal with phase continuity"""
        t = np.linspace(0, self.chirp_duration, self.samples_per_chirp)
        
        # Calculate phase with proper phase continuity between chirps
        freq_sweep = self.bandwidth/self.chirp_duration * t
        phase_accumulation = 2 * np.pi * chirp_idx * self.bandwidth * self.chirp_duration
        phase = 2 * np.pi * (self.center_freq * t + 0.5 * freq_sweep * t) + phase_accumulation
        
        return np.exp(1j * phase)

    def _add_realistic_effects(self, rx_signal, tx_signal):
        """
        Add realistic effects to the received signal. including:

        - Direct coupling (TX leakage)
        - Environmental clutter
        - Crosstalk
        - Ground clutter
        - System noise
        
        Args:
            rx_signal: Received signal with shape [num_rx, num_chirps, samples_per_chirp]
            tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp]
            
        Returns:
            Modified received signal with realistic effects
        """
        # Add direct coupling component (TX leakage)
        direct_coupling_power = 0.01  # Adjust based on desired coupling strength
        for rx_idx in range(self.num_rx):
            # Direct coupling is a delayed and attenuated version of TX signal
            delay_samples = int(0.1 * self.samples_per_chirp)  # Small delay for direct path
            for chirp_idx in range(self.num_chirps):
                # Add attenuated TX signal with small delay
                tx_chirp = self._generate_fmcw_chirp(chirp_idx)
                delayed_tx = np.zeros_like(tx_chirp)
                delayed_tx[delay_samples:] = tx_chirp[:-delay_samples] if delay_samples > 0 else tx_chirp
                rx_signal[rx_idx, chirp_idx] += np.sqrt(direct_coupling_power) * delayed_tx

        # Add environmental clutter (static reflections)
        num_clutter_points = random.randint(5, 15)
        for _ in range(num_clutter_points):
            clutter_range = random.uniform(5, self.max_range)
            clutter_rcs = random.uniform(-40, -20)  # dBsm
            clutter_power = self._calculate_received_power(clutter_range, clutter_rcs)
            
            # Add clutter to all chirps with same range (static)
            for rx_idx in range(self.num_rx):
                for chirp_idx in range(self.num_chirps):
                    delay_samples = int((2 * clutter_range / self.speed_of_light) * self.sample_rate)
                    if delay_samples < self.samples_per_chirp:
                        # Phase randomization for each clutter point
                        phase = random.uniform(0, 2 * np.pi)
                        rx_signal[rx_idx, chirp_idx, delay_samples:] += np.sqrt(clutter_power) * np.exp(1j * phase)
        
        # Add crosstalk between TX and RX (reduced effect)
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

    def _add_noise(self, signal, snr_db):
        """Add realistic noise to the signal"""
        # Calculate signal power
        signal_power = np.mean(np.abs(signal)**2)
        
        # Ensure minimum signal power for noise calculation
        min_power = 1e-10
        signal_power = max(signal_power, min_power)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / (10**(snr_db/10))
        
        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, signal.shape) + 
                                        1j * np.random.normal(0, 1, signal.shape))
        
        # Add noise to signal
        return signal + noise
    
    def _calculate_received_power(self, distance, rcs):
        """
        Calculate received power based on radar equation.
        
        Args:
            distance: Target distance in meters
            rcs: Radar cross-section in dBsm
            
        Returns:
            Received power (linear scale)
        """
        # Convert RCS from dBsm to linear scale
        rcs_linear = 10**(rcs/10)
        
        # Simplified radar equation: P_r = P_t * G^2 * λ^2 * σ / ((4π)^3 * R^4)
        # We're using normalized values, so P_t * G^2 * λ^2 / (4π)^3 = 1
        received_power = rcs_linear / (distance**4)
        
        return received_power
    
    def fmcw_demodulate(self, tx_full, rx_full, total_samples_per_chirp, beat_samples_per_chirp, num_chirps):
        """
        Extract beat signals by dechirping Rx with Tx for each chirp.
        """
        #beat_samples_per_chirp = int(fs * chirp_duration)
        #samples_per_idle = int(fs * chirp_duration * idle_time_ratio)
        #total_samples_per_chirp = samples_per_chirp + samples_per_idle

        beat_signals = np.zeros((num_chirps, beat_samples_per_chirp), dtype=complex)
        for i in range(num_chirps):
            start = i * total_samples_per_chirp
            end = start + beat_samples_per_chirp
            if end > len(tx_full): continue
            beat_signals[i] = rx_full[start:end] * np.conj(tx_full[start:end])

        return beat_signals
    
    def _time_to_range_doppler(self, rx_signal,
                          num_chirps,
                          samples_per_chirp,
                          num_doppler_bins,
                          num_range_bins,
                          apply_mti=False,  # Default to False for simple case
                          apply_doppler_centering=True,  # Default to True to match line 338-345
                          apply_notch_filter=False,  # Default to False for simple case
                          notch_width=5,  # Parameter for notch filter
                          use_blackman_window=False,  # Default to False for simple case
                          dynamic_range_db=50):  # Keep dynamic range parameter
        """
        Convert time domain signal to range-Doppler map.
        
        Args:
            rx_signal: Received signal with shape either:
                      - [num_rx, num_chirps, samples_per_chirp] (standard format)
                      - [num_rx, num_chirps * samples_per_chirp] (flattened format)
            apply_mti: Whether to apply Moving Target Indication filtering
            apply_doppler_centering: Whether to center the Doppler FFT
            apply_notch_filter: Whether to apply a notch filter to suppress zero-Doppler
            notch_width: Width of the notch filter in bins
            use_blackman_window: Whether to use Blackman window instead of Hamming
            dynamic_range_db: Dynamic range in dB for normalization
            
        Returns:
            Range-Doppler map with shape [num_rx, 2, num_doppler_bins, num_range_bins]
        """
        # Check if input is flattened format and reshape if needed
        if rx_signal.ndim == 2 and rx_signal.shape[1] == num_chirps * samples_per_chirp:
            # Reshape from [num_rx, num_chirps * samples_per_chirp] to [num_rx, num_chirps, samples_per_chirp]
            rx_signal = rx_signal.reshape(rx_signal.shape[0], num_chirps, samples_per_chirp)
        
        num_rx = rx_signal.shape[0]
        rd_map = np.zeros((num_rx, 2, num_doppler_bins, num_range_bins), dtype=np.float32)
        
        for rx in range(num_rx):
            processed_signal = rx_signal[rx]
            
            # Apply MTI filtering if requested (subtract consecutive chirps)
            if apply_mti:
                mti_signal = np.zeros_like(processed_signal)
                mti_signal[1:] = processed_signal[1:] - processed_signal[:-1]
                processed_signal = mti_signal
            
            # Apply windowing to each chirp if requested
            if use_blackman_window:
                range_window = np.blackman(samples_per_chirp)
                range_window /= np.sum(range_window)  # Normalize window
                doppler_window = np.blackman(num_chirps)
                doppler_window /= np.sum(doppler_window)  # Normalize window
                
                # Apply windowing to each chirp (along fast-time/samples dimension)
                processed_signal = processed_signal * range_window[np.newaxis, :]
                
                # Apply range FFT
                range_fft = np.fft.fft(processed_signal, n=num_range_bins, axis=1)
                
                # Apply windowing to each range bin (along slow-time/chirps dimension)
                range_fft = range_fft * doppler_window[:, np.newaxis]
            else:
                # Simple range FFT without windowing
                range_fft = np.fft.fft(processed_signal, n=num_range_bins, axis=1)
            
            # Apply range FFT shifting if requested
            if apply_doppler_centering:
                range_fft = np.fft.fftshift(range_fft, axes=1)
            
            # Apply Doppler FFT
            doppler_fft = np.fft.fft(range_fft, n=num_doppler_bins, axis=0)
            
            # Apply Doppler FFT shifting if requested
            if apply_doppler_centering:
                doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
            
            # Store real and imaginary parts
            rd_map[rx, 0, :, :] = np.real(doppler_fft)
            rd_map[rx, 1, :, :] = np.imag(doppler_fft)
        
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
        
        # Define CFAR parameters - increased guard and training cells
        guard_cells = (3, 3)  # Increased from (2, 2)
        training_cells = (6, 6)  # Increased from (4, 4)
        pfa = 1e-5  # Reduced probability of false alarm
        
        # Initialize CFAR detection map
        cfar_map = np.zeros((self.num_doppler_bins, self.num_range_bins), dtype=bool)
        
        # Apply CFAR detection with boundary checks
        for d_idx in range(self.num_doppler_bins):
            for r_idx in range(self.num_range_bins):
                cut_value = rd_magnitude[d_idx, r_idx]
                
                # Calculate window boundaries with safe limits
                d_min = max(0, d_idx - guard_cells[0] - training_cells[0])
                d_max = min(self.num_doppler_bins - 1, d_idx + guard_cells[0] + training_cells[0])
                r_min = max(0, r_idx - guard_cells[1] - training_cells[1])
                r_max = min(self.num_range_bins - 1, r_idx + guard_cells[1] + training_cells[1])
                
                # Extract training cells excluding guard area
                training_region = []
                for di in range(d_min, d_max + 1):
                    for ri in range(r_min, r_max + 1):
                        if abs(di - d_idx) > guard_cells[0] or abs(ri - r_idx) > guard_cells[1]:
                            training_region.append(rd_magnitude[di, ri])
                
                # Ordered statistic CFAR with adaptive threshold
                if len(training_region) > 0:
                    training_region.sort()
                    k = int(len(training_region) * (1 - pfa))
                    threshold = training_region[min(k, len(training_region) - 1)] * 1.5
                    cfar_map[d_idx, r_idx] = cut_value > threshold

        # Post-processing to remove isolated detections
        filtered_cfar_map = np.zeros_like(cfar_map)
        for d_idx in range(1, self.num_doppler_bins-1):
            for r_idx in range(1, self.num_range_bins-1):
                if cfar_map[d_idx, r_idx]:
                    neighbor_count = np.sum(cfar_map[d_idx-1:d_idx+2, r_idx-1:r_idx+2])
                    filtered_cfar_map[d_idx, r_idx] = neighbor_count > 1

        # Extract and validate targets
        detected_targets = []
        for d_idx in range(self.num_doppler_bins):
            for r_idx in range(self.num_range_bins):
                if filtered_cfar_map[d_idx, r_idx]:
                    distance = r_idx * self.range_resolution
                    velocity = (d_idx - self.num_doppler_bins // 2) * self.velocity_resolution
                    noise_floor = np.median(rd_magnitude)
                    snr_db = 20 * np.log10(rd_magnitude[d_idx, r_idx] / (noise_floor + 1e-10))
                    
                    if snr_db > 10.0:  # SNR threshold
                        detected_targets.append({
                            'range_bin': r_idx,
                            'doppler_bin': d_idx,
                            'distance': distance,
                            'velocity': velocity,
                            'snr': snr_db
                        })

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
    parser.add_argument('--save_path', type=str, default='data/radarv4/',
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
    os.makedirs(args.save_path, exist_ok=True)
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