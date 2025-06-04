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
from AIRadarLib.datautil import *
from AIRadarLib.radar_det import cfar_2d_pytorch, cfar_2d_numpy
from AIRadarLib.target_utils import generate_radar_targets, create_target_mask
from AIRadarLib.visualization import plot_detection_results
from AIRadarLib.waveform_utils import generate_fmcw_chirp_signal
from AIRadarLib.channel_simulation import (
    generate_fmcw_chirp,
    add_realistic_effects,
    add_noise,
    calculate_received_power,
    ray_tracing_simulation
)
from AIRadarLib.signal_processing import fmcw_demodulate, time_to_range_doppler

class RadarDataset(Dataset):
    # In the RadarDataset class initialization, update the default parameters
    def __init__(self, 
                 num_samples=100,
                 num_range_bins=128,  # Increased from 64 for better range resolution
                 num_doppler_bins=16,  # Increased from 12 for better velocity resolution
                 sample_rate=500e6, #1.5e6,    # reduce from 15MHz to 1.5MHz Adjusted to match hardware constraints
                 transceiver_bandwidth=30e6,  # AD9361 bandwidth limitation
                 #chirp_duration=1e-4,  # reduce from 1ms to 0.1ms Increased to 1ms for better SNR (from 500μs)
                 num_chirps=32,       # Increased from 12 for better integration gain
                 bandwidth=200e6,     # Fixed by CN0566 capabilities
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
        self.target_max_range = 100  # meters
        self.target_min_range = 1  # meters
        self.target_min_velocity = 0  # m/s
        self.target_max_velocity = 10  # m/s
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
        self.speed_of_light = 3e8  # Speed of light in m/s
        
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
        
        desired_max_unambiguous_range = 100  # meters
        speed_of_light = 3e8  # m/s
        chirp_duration = 2 * desired_max_unambiguous_range / speed_of_light  # seconds
        print(f"Set chirp_duration to {chirp_duration*1e6:.2f} us for max unambiguous range of {desired_max_unambiguous_range} m")
        self.chirp_duration = chirp_duration
        # Calculate slope
        slope = bandwidth / chirp_duration

        # Calculate maximum beat frequency for max_range
        #Max beat frequency: f_beat_max = 2 * slope * max_range / c
        f_beat_max = 2 * slope * desired_max_unambiguous_range / speed_of_light
        # Calculate Nyquist frequency
        nyquist_freq = sample_rate / 2

        print(f"Max beat frequency: {f_beat_max/1e6:.2f} MHz")#equalivent to bandwidth
        print(f"Nyquist frequency: {nyquist_freq/1e6:.2f} MHz")
        if f_beat_max <= nyquist_freq:
            print("✅ No frequency wraparound (aliasing).")
        else:
            print("❌ Frequency wraparound detected! Increase chirp_duration or sample_rate, or reduce bandwidth or max_range.")
            
        
        #velocity resolution is given by: \Delta v = \frac{\lambda}{2 \cdot T_{\text{frame}}}
        #where ( \lambda ) is the wavelength and ( T_{\text{frame}} ) is the total time of all chirps in a frame.
        #To improve velocity resolution: Increase num_chirps, Increase the chirp duration ( chirp_duration), or Decrease the center frequency
        self.num_chirps = 512
        frame_duration = num_chirps * chirp_duration
        refresh_rate = 1.0 / frame_duration if frame_duration > 0 else 0
        print(f"Frame duration: {frame_duration*1e6:.2f} us")
        print(f"Refresh rate: {refresh_rate:.2f} Hz")

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


    def generate_dataset(self, fixedsnr_db=None, visualize=True, simulate_rf_chain=False, analog_sample_rate=None):
        """
        Generate a radar dataset using ray-tracing simulation.
        Args:
            visualize: Whether to visualize the results
            simulate_rf_chain: If True, simulate analog upconversion/downconversion and ADC sampling
            analog_sample_rate: The sample rate for analog processing (if None, defaults to 10x ADC rate)
        Returns:
            Dictionary containing the generated dataset
        """
        print(f"Generating {self.num_samples} radar samples using ray-tracing simulation...")
        if visualize:
            savevis_path = os.path.join(self.save_path, "visualization")
            os.makedirs(savevis_path, exist_ok=True)

        flattened_length = self.num_chirps * self.total_samples_per_chirp
        dataset = {
            'time_domain_data': np.zeros((self.num_samples, self.num_rx, flattened_length, 2), dtype=self.precision),
            'range_doppler_maps': np.zeros((self.num_samples, self.num_rx, 2, self.num_doppler_bins, self.num_range_bins), dtype=self.precision),
            'target_masks': np.zeros((self.num_samples, self.num_doppler_bins, self.num_range_bins, 1), dtype=self.precision),
            'target_info': [],
            'detection_results': []
        }

        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            targets = generate_radar_targets(
                num_targets=self.max_targets,  # or self.max_targets for random number
                min_range=self.target_min_range,    # or self.min_range
                max_range=self.target_max_range,   # or self.max_range
                min_velocity=self.target_min_velocity,
                max_velocity=self.target_max_velocity,
                min_rcs=5.0,
                max_rcs=30.0,
                azimuth_range=(-45, 45),
                elevation_range=(-10, 10),
                range_factor=0.5,
                velocity_factor=0.5
            )
            if simulate_rf_chain:
                # Use a higher analog sample rate for upconversion/downconversion
                if analog_sample_rate is None:
                    analog_sample_rate = self.sample_rate * 10  # e.g., 10x oversampling
                # Generate TX at analog rate
                total_analogsamples_per_chirp = int(self.total_samples_per_chirp * analog_sample_rate / self.sample_rate)
                tx_signal_analog = self._generate_tx_signal(
                    num_chirps=self.num_chirps,
                    total_samples_per_chirp=total_analogsamples_per_chirp,
                    active_samples=total_analogsamples_per_chirp,
                    sample_rate=analog_sample_rate,
                    slope=self.fmcw_slope, return_full=True, window_type=None
                )
                # Upconvert
                tx_signal_rf = upconvert_signal(tx_signal_analog, self.center_freq, analog_sample_rate)
                
                # Channel simulation at analog rate
                rx_signal_rf = self._ray_tracing_simulation(tx_signal_rf, targets, perfect_mode=False, flatten_output=True)
                
                # === Add noise in the analog/RF domain ===
                if fixedsnr_db is not None:
                    snr_db = fixedsnr_db
                else:
                    snr_db = random.uniform(self.snr_min, self.snr_max)
                rx_signal_rf = self._add_noise(rx_signal_rf, snr_db)
                # === End noise addition ===
                
                # Downconvert
                rx_signal_bb = downconvert_signal(rx_signal_rf, self.center_freq, analog_sample_rate)
                # Decimate to ADC sample rate
                decimation_factor = int(analog_sample_rate // self.sample_rate)
                rx_signal = scipy.signal.decimate(rx_signal_bb, decimation_factor, axis=-1, zero_phase=True)
                tx_signal = scipy.signal.decimate(tx_signal_analog, decimation_factor, axis=-1, zero_phase=True)
            else:
                # Baseband-only simulation (no explicit up/downconversion)
                tx_signal = self._generate_tx_signal(
                    num_chirps=self.num_chirps,
                    total_samples_per_chirp=self.total_samples_per_chirp,
                    active_samples=self.samples_per_chirp,
                    sample_rate=self.sample_rate,
                    slope=self.fmcw_slope, return_full=True, window_type=None
                )
                rx_signal = self._ray_tracing_simulation(tx_signal, targets, perfect_mode=False, flatten_output=True)

                # === Add noise in the baseband domain ===
                if fixedsnr_db is not None:
                    snr_db = fixedsnr_db
                else:
                    snr_db = random.uniform(self.snr_min, self.snr_max)
                rx_signal = self._add_noise(rx_signal, snr_db)
                # === End noise addition ===
            
            if visualize:
                # Extract a single chirp from the full TX signal for visualization
                tx_chirp = tx_signal[:self.total_samples_per_chirp]  # Get first chirp's active samples
                # Visualize the time and spectrum of the TX chirp
                plot_signal_time_and_spectrum(
                    signal=tx_chirp,
                    sample_rate=self.sample_rate,
                    total_duration=self.chirp_duration,
                    title_prefix="TX Chirp",
                    #bandwidth=self.bandwidth,
                    center_freq=self.center_freq,
                    textstr=None,
                    normalize=False,
                    save_path=os.path.join(savevis_path, f"tx_rf_chirp_{i}{IMG_FORMAT}"),
                    draw_window = False
                )
                # Add instantaneous frequency plot for the chirp signal
                plot_instantaneous_frequency(
                        signal=tx_chirp,
                        sample_rate=self.sample_rate,
                        total_duration=self.chirp_duration,
                        slope=self.fmcw_slope,
                        bandwidth=self.bandwidth,
                        center_freq=self.center_freq,
                        title_prefix="TX Chirp",
                        textstr=f"Bandwidth: {self.bandwidth/1e6:.1f} MHz\nSlope: {self.fmcw_slope/1e12:.2f} THz/s",
                        save_path=os.path.join(savevis_path, f"tx_rf_chirp_freq_{i}{IMG_FORMAT}")
                    )

                # Visualize the time and spectrum of the received signal
                rx_signal_chirp = rx_signal[0, :self.samples_per_chirp] 
                plot_signal_time_and_spectrum(
                        signal=rx_signal_chirp,
                        sample_rate=self.sample_rate,
                        total_duration=self.chirp_duration,
                        title_prefix="RX Chirp",
                        #bandwidth=self.bandwidth,
                        #center_freq=self.center_freq,
                        textstr=None,
                        normalize=False,
                        save_path=os.path.join(savevis_path, f"rx_chirp_{i}{IMG_FORMAT}"),
                        draw_window = False
                    )
            # Demodulate the FMCW signal to beat signal
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
                        save_path=os.path.join(savevis_path, f"beatchirp_{i}{IMG_FORMAT}"),
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
                    targets=[], #targets,  # Make sure 'targets' is defined in your context
                    range_resolution=self.range_resolution,
                    velocity_resolution=self.velocity_resolution,
                    num_range_bins=self.num_range_bins,
                    num_doppler_bins=self.num_doppler_bins,
                    title_prefix=f"Range-Doppler Map Sample {i}",
                    save_path=os.path.join(savevis_path, f"rdmap_{i}{IMG_FORMAT}")
                )
                plot_3d_range_doppler_map_with_ground_truth(
                    rd_map=rd_map[0,:],
                    targets=[],
                    range_resolution=self.range_resolution,
                    velocity_resolution=self.velocity_resolution,
                    num_range_bins=self.num_range_bins,
                    num_doppler_bins=self.num_doppler_bins,
                    save_path=os.path.join(savevis_path, f"rd3dmap_{i}{IMG_FORMAT}")
                )


            # Perform target detection using CFAR for Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
            #[2, num_doppler_bins, num_range_bins]
            detection_results = cfar_2d_numpy(
                rd_map,
                num_train=8,
                num_guard=4,
                range_res=self.range_resolution,          # meters per range bin
                doppler_res=self.velocity_resolution, #0.25,       # m/s per Doppler bin
                max_range=100,
                max_speed=50,
                method='GO',
                estimate_aoa=False,
            )#list of dicts
            
            #detection_results = self._cfar_detection(rd_map[0,:])#(2, 128, 256)
            #List of Multiple targets in dict
            if targets is not None:
                target_mask = create_target_mask(
                    targets=targets,
                    num_doppler_bins=self.num_doppler_bins,
                    num_range_bins=self.num_range_bins,
                    range_resolution=self.range_resolution,
                    velocity_resolution=self.velocity_resolution,
                    precision=self.precision
                ) #(128, 256, 1)
                # Store the partially flattened rx_signal
                dataset['time_domain_data'][i, :, :, 0] = np.real(rx_signal)
                dataset['time_domain_data'][i, :, :, 1] = np.imag(rx_signal)
                dataset['range_doppler_maps'][i] = rd_map #[num_rx, 2(real+imaginary), num_doppler_bins, num_range_bins]
                dataset['target_masks'][i] = target_mask
                dataset['target_info'].append(targets)
                dataset['detection_results'].append(detection_results)

                # Plot detection results if visualize is True
                if visualize:
                    plot_detection_results(
                        rd_map=rd_map,
                        target_mask=target_mask,
                        targets=targets,
                        detection_results=detection_results,
                        range_resolution=self.range_resolution,
                        velocity_resolution=self.velocity_resolution,
                        num_doppler_bins=self.num_doppler_bins,
                        num_range_bins=self.num_range_bins,
                        save_path=os.path.join(savevis_path, f"detection_results_{i}{IMG_FORMAT}"),
                        title=f"Radar Detection Results - Sample {i+1}",
                        show_plot=False
                    )
                
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
        # Generate the continuous signal using the utility function
        continuous_signal, chirp_duration = generate_fmcw_chirp_signal(
            num_chirps=num_chirps,
            total_samples_per_chirp=total_samples_per_chirp,
            active_samples=active_samples,
            sample_rate=sample_rate,
            slope=slope,
            tx_power=tx_power,
            edge_ratio=edge_ratio,
            window_type=window_type
        )
        
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
    
    def _ray_tracing_simulation(self, tx_signal, targets, perfect_mode=False, flatten_output=False, tx_signal_is_upconverted=False, analog_sample_rate=None):
        """
        Perform ray-tracing simulation to generate received signals.

        Args:
            tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp] or flattened 1D array
            targets: List of target dictionaries
            perfect_mode: If True, uses a single fixed target for ideal simulation
            flatten_output: If True, returns a flattened 1D array similar to simulate_single_target_echo
            tx_signal_is_upconverted: If True, tx_signal is already upconverted to RF (center frequency)
        Returns:
            Complex RX signal with shape [num_rx, num_chirps, samples_per_chirp] or flattened 1D array
        """
        # Create a dictionary of radar parameters to pass to the utility function
        radar_params = {
            'num_rx': self.num_rx,
            'num_chirps': self.num_chirps,
            'samples_per_chirp': self.samples_per_chirp,
            'chirp_duration': self.chirp_duration,
            'bandwidth': self.bandwidth,
            'center_freq': self.center_freq,
            'sample_rate': self.sample_rate,
            'speed_of_light': self.speed_of_light,
            'wavelength': self.wavelength,
            'max_range': self.max_range,
            'apply_realistic_effects': self.apply_realistic_effects
        }
        
        # Call the utility function
        return ray_tracing_simulation(
            tx_signal=tx_signal,
            targets=targets,
            radar_params=radar_params,
            perfect_mode=perfect_mode,
            flatten_output=flatten_output,
            tx_signal_is_upconverted=tx_signal_is_upconverted,
            analog_sample_rate=analog_sample_rate
        )

    def _generate_fmcw_chirp(self, chirp_idx):
        """Generate a single FMCW chirp signal with phase continuity"""
        # Call the utility function
        return generate_fmcw_chirp(
            chirp_idx=chirp_idx,
            chirp_duration=self.chirp_duration,
            samples_per_chirp=self.samples_per_chirp,
            bandwidth=self.bandwidth,
            center_freq=self.center_freq
        )

    def _add_realistic_effects(self, rx_signal, tx_signal):
        """Add realistic effects to the received signal"""
        # Create a dictionary of radar parameters to pass to the utility function
        radar_params = {
            'num_rx': self.num_rx,
            'num_chirps': self.num_chirps,
            'samples_per_chirp': self.samples_per_chirp,
            'chirp_duration': self.chirp_duration,
            'bandwidth': self.bandwidth,
            'center_freq': self.center_freq,
            'sample_rate': self.sample_rate,
            'speed_of_light': self.speed_of_light,
            'max_range': self.max_range
        }
        
        # Call the utility function
        return add_realistic_effects(rx_signal, tx_signal, radar_params)

    def _add_noise(self, signal, snr_db):
        """Add realistic noise to the signal"""
        # Call the utility function
        return add_noise(signal, snr_db)

    def _calculate_received_power(self, distance, rcs):
        """Calculate received power based on radar equation"""
        # Call the utility function
        return calculate_received_power(distance, rcs)
    
    def fmcw_demodulate(self, tx_full, rx_full, total_samples_per_chirp, beat_samples_per_chirp, num_chirps):
        """
        Extract beat signals by dechirping Rx with Tx for each chirp.
        """
        return fmcw_demodulate(tx_full, rx_full, total_samples_per_chirp, beat_samples_per_chirp, num_chirps)
    
    def _time_to_range_doppler(self, rx_signal,
                          num_chirps,
                          samples_per_chirp,
                          num_doppler_bins,
                          num_range_bins,
                          apply_mti=False,
                          apply_doppler_centering=True,
                          apply_notch_filter=False,
                          notch_width=5,
                          use_blackman_window=False,
                          dynamic_range_db=50):
        """
        Convert time domain signal to range-Doppler map.
        """
        return time_to_range_doppler(
            rx_signal,
            num_chirps,
            samples_per_chirp,
            num_doppler_bins,
            num_range_bins,
            apply_mti=apply_mti,
            apply_doppler_centering=apply_doppler_centering,
            apply_notch_filter=apply_notch_filter,
            notch_width=notch_width,
            use_blackman_window=use_blackman_window,
            dynamic_range_db=dynamic_range_db
        )
    
    

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
            apply_realistic_effects = True,
            drawfig=True  # Enable figure drawing for visualization
        )