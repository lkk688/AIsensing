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

class RayTracingRadarDataset:
    """
    A radar dataset generator that uses ray-tracing principles to simulate
    radar signal propagation, target interaction, and detection.
    """
    
    def __init__(self, 
                num_samples=20,
                 num_range_bins=256,
                 num_doppler_bins=128,
                 sample_rate=50e6,
                 bandwidth=200e6,
                 center_freq=77e9,
                 chirp_duration=100e-6, #50e-6,
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
        
        # Configure and validate radar parameters
        #self.range_resolution
        self.max_range = 300
        self._configure_radar_parameters2()
        
        # Create directory for saving data
        os.makedirs(self.save_path, exist_ok=True)
        
        # Print radar parameters
        self._print_radar_parameters()
    
    def _configure_radar_parameters(self):
        """
        Configure and validate radar parameters based on physical constraints.
        """
        # --- Constants and defaults ---
        c = self.speed_of_light = 3e8  # Speed of light (m/s)
        f0 = self.center_freq           # Center frequency (Hz)
        λ = self.wavelength = c / f0
        max_bandwidth = 4e9
        max_sample_rate = 500e6
        max_range_default = 300
        max_velocity = 60.0  # m/s

        # --- Bandwidth and range resolution ---
        if hasattr(self, 'range_resolution'):
            self.bandwidth = c / (2 * self.range_resolution)
        self.bandwidth = min(getattr(self, 'bandwidth', 500e6), max_bandwidth)
        self.range_resolution = c / (2 * self.bandwidth)

        # --- Chirp duration ---
        max_range = getattr(self, 'max_range', max_range_default)
        Tc_min = max(5.5 * (2 * max_range) / c, 20e-6)
        self.chirp_duration = Tc_min

        # Step: Calculate FMCW slope
        self.slope = self.bandwidth / self.chirp_duration  # Hz/s

        # Slope check: Prevent multiple chirp sweeps per chirp cycle
        max_allowed_slope = 100e12  # 100 THz/s typical max for radar chipsets

        if self.slope > max_allowed_slope:
            print(f"Warning: Chirp slope {self.slope/1e12:.2f} THz/s exceeds hardware limit of {max_allowed_slope/1e12:.1f} THz/s")
            # Adjust chirp duration to reduce slope
            self.chirp_duration = self.bandwidth / max_allowed_slope
            self.slope = self.bandwidth / self.chirp_duration
            self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
            print(f"Adjusted chirp duration to {self.chirp_duration*1e6:.1f} µs to stay within slope limit.")

        f_beat_max = (2 * self.max_range * self.bandwidth) / (self.speed_of_light * self.chirp_duration)
        sample_rate_raw = min(2.2 * f_beat_max, max_sample_rate)
        self.sample_rate = np.ceil(sample_rate_raw / 1e6) * 1e6  # Round up to nearest MHz
        # --- Samples per chirp ---
        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)

        # --- Chirp count & velocity resolution ---
        self.num_chirps = min(getattr(self, 'num_chirps', 128), 512)
        self.velocity_resolution = λ / (2 * self.num_chirps * self.chirp_duration)
        self.max_unambiguous_velocity = λ / (4 * self.chirp_duration)
        self.max_velocity = min(self.max_unambiguous_velocity, max_velocity)

        # --- FFT sizes ---
        self.range_fft_size = 2 ** int(np.ceil(np.log2(self.samples_per_chirp)))
        self.doppler_fft_size = 2 ** int(np.ceil(np.log2(self.num_chirps)))

        # --- Max range check and chirp adjustment ---
        achievable_range = (self.sample_rate * c * self.chirp_duration) / (2 * self.bandwidth)
        if achievable_range > max_range_default:
            self.chirp_duration = (max_range_default * 2 * self.bandwidth) / (self.sample_rate * c)
            self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
            self.range_fft_size = 2 ** int(np.ceil(np.log2(self.samples_per_chirp)))
            achievable_range = max_range_default

        self.max_range = achievable_range
        self.min_range = self.range_resolution

        # --- Repetition parameters ---
        self.pulse_repetition_interval = self.chirp_duration
        self.pulse_repetition_frequency = 1 / self.chirp_duration
        self.chirp_repetition_interval = self.chirp_duration

        # --- FMCW configuration check ---
        max_beat_freq = (2 * self.max_range * self.bandwidth) / (c * self.chirp_duration)
        if self.sample_rate < 2 * max_beat_freq:
            self.sample_rate = 2 * max_beat_freq
            print(f"Adjusted sample rate to {self.sample_rate / 1e6:.1f} MHz for max range {self.max_range:.1f} m")

        # --- Slope calculation ---
        self.slope = self.bandwidth / self.chirp_duration
        print(f"New Sample rage: {self.sample_rate / 1e6:.1f} MHz")
        print(f"New FMCW Slope: {self.slope / 1e12:.2f} THz/s")

        self.idle_time_ratio = 0.2 #used for tx signal
        if self.idle_time_ratio>0:
            total_chirp_duration = self.chirp_duration * (1 + self.idle_time_ratio)
            samples_per_chirp = int(self.sample_rate * self.chirp_duration)
            self.total_samples_per_chirp = int(self.sample_rate * total_chirp_duration)
            self.active_samples = int(samples_per_chirp * (1 - self.idle_time_ratio))
            self.idle_time = self.chirp_duration * self.idle_time_ratio
            self.chirp_repetition_interval = total_chirp_duration
            self.total_chirp_duration = total_chirp_duration
        else:
            self.total_samples_per_chirp = self.samples_per_chirp
            self.active_samples = self.samples_per_chirp
            self.total_chirp_duration = self.chirp_duration
            #self.chirp_repetition_interval

    def _configure_radar_parameters2(self):
        """
        Configure and validate radar parameters for FMCW radar.

        Ensures valid bandwidth, chirp duration, slope, sample rate, and avoids sweep wraparound or out-of-band RF issues.
        """

        # --- Constants ---
        c = self.speed_of_light = 3e8                      # Speed of light (m/s)
        f0 = self.center_freq                              # Center frequency (Hz)
        λ = self.wavelength = c / f0
        max_bandwidth = 4e9                                # 4 GHz max
        max_sample_rate = 500e6                            # 500 MSPS
        max_range_default = 300                            # m
        max_velocity = 60.0                                # m/s
        max_rf_slope = 80e12                               # 80 THz/s max slope (hardware limit)
        rf_min_freq = 76e9
        rf_max_freq = 78e9

        # --- Range resolution and bandwidth ---
        if hasattr(self, 'range_resolution'):
            self.bandwidth = c / (2 * self.range_resolution)
        self.bandwidth = min(getattr(self, 'bandwidth', 500e6), max_bandwidth)
        self.range_resolution = c / (2 * self.bandwidth)

        # --- Max target range ---
        max_range = getattr(self, 'max_range', max_range_default)

        # --- Enforce RF range constraints ---
        f_start = f0 - self.bandwidth / 2
        f_end = f0 + self.bandwidth / 2
        if f_start < rf_min_freq or f_end > rf_max_freq:
            raise ValueError(f"RF sweep range {f_start/1e9:.2f}–{f_end/1e9:.2f} GHz exceeds RF hardware limits ({rf_min_freq/1e9:.2f}–{rf_max_freq/1e9:.2f} GHz)")

        # --- Determine safe chirp duration ---
        Tc_range = max(5.5 * (2 * max_range) / c, 20e-6)        # for range
        Tc_slope = self.bandwidth / max_rf_slope                # for slope
        self.chirp_duration = max(Tc_range, Tc_slope)

        # --- Compute slope ---
        self.slope = self.bandwidth / self.chirp_duration
        if self.slope > max_rf_slope:
            raise ValueError(f"Final slope {self.slope / 1e12:.2f} THz/s still exceeds hardware limit ({max_rf_slope / 1e12:.1f} THz/s)")

        # --- Estimate required sample rate ---
        f_beat_max = (2 * max_range * self.bandwidth) / (c * self.chirp_duration)
        self.sample_rate = np.ceil(min(2.2 * f_beat_max, max_sample_rate) / 1e6) * 1e6

        # --- Samples per chirp ---
        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)

        # --- Velocity parameters ---
        self.num_chirps = min(getattr(self, 'num_chirps', 128), 512)
        self.velocity_resolution = λ / (2 * self.num_chirps * self.chirp_duration)
        self.max_unambiguous_velocity = λ / (4 * self.chirp_duration)
        self.max_velocity = min(self.max_unambiguous_velocity, max_velocity)

        # --- FFT sizes ---
        self.range_fft_size = 2 ** int(np.ceil(np.log2(self.samples_per_chirp)))
        self.doppler_fft_size = 2 ** int(np.ceil(np.log2(self.num_chirps)))

        # --- Final achievable max range check ---
        achievable_range = (self.sample_rate * c * self.chirp_duration) / (2 * self.bandwidth)
        if achievable_range > max_range_default:
            self.chirp_duration = (max_range_default * 2 * self.bandwidth) / (self.sample_rate * c)
            self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
            self.range_fft_size = 2 ** int(np.ceil(np.log2(self.samples_per_chirp)))
            achievable_range = max_range_default

        self.max_range = achievable_range
        self.min_range = self.range_resolution

        # --- Chirp repetition timing ---
        self.pulse_repetition_interval = self.chirp_duration
        self.pulse_repetition_frequency = 1 / self.chirp_duration
        self.chirp_repetition_interval = self.chirp_duration

        # --- Final diagnostics ---
        print(f"✅ Sample Rate        : {self.sample_rate / 1e6:.1f} MHz")
        print(f"✅ Chirp Duration     : {self.chirp_duration * 1e6:.1f} µs")
        print(f"✅ FMCW Slope         : {self.slope / 1e12:.2f} THz/s")
        print(f"✅ Max Achievable Range: {self.max_range:.2f} m")
        print(f"✅ RF Sweep Band      : {f_start/1e9:.2f}–{f_end/1e9:.2f} GHz")

        # --- TX signal idle time configuration (optional) ---
        self.idle_time_ratio = 0.2
        if self.idle_time_ratio > 0:
            self.idle_time = self.chirp_duration * self.idle_time_ratio
            self.total_chirp_duration = self.chirp_duration + self.idle_time
            self.chirp_repetition_interval = self.total_chirp_duration
            self.total_samples_per_chirp = int(self.sample_rate * self.total_chirp_duration)
            self.active_samples = int(self.samples_per_chirp * (1 - self.idle_time_ratio))
        else:
            self.total_chirp_duration = self.chirp_duration
            self.total_samples_per_chirp = self.samples_per_chirp
            self.active_samples = self.samples_per_chirp
            
    def _print_radar_parameters(self):
        """Print the radar system parameters."""
        print("\n=== Ray-Tracing Radar Simulation Parameters ===")
        print(f"Range Resolution: {self.range_resolution:.2f} m")
        print(f"Maximum Range: {self.max_range:.2f} m")
        #print(f"Velocity Resolution: {self.velocity_resolution:.2f} m/s")
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
        
        # Calculate the actual flattened length including idle time
        idle_time_ratio = getattr(self, 'idle_time_ratio', 0.2)  # Default to 0.2 if not defined
        samples_per_chirp = self.samples_per_chirp
        samples_per_idle = int(self.sample_rate * self.chirp_duration * idle_time_ratio)
        total_samples_per_chirp = samples_per_chirp + samples_per_idle
        flattened_length = self.num_chirps * total_samples_per_chirp
        
        dataset = {
            'time_domain_data': np.zeros((self.num_samples, self.num_rx, flattened_length, 2), 
                                        dtype=self.precision),
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
            
            #self.new_pipeline()
            testing_newpipe = False
            if testing_newpipe==True:
                tx_signal = self._generate_tx_signal(return_full=True) #(153600,) (4, 128, 400) complex
                rx_signal = self.simulate_single_target_echo(
                    tx_full=tx_signal,
                    fs=self.sample_rate,
                    target_range=50,        # meters
                    target_velocity=15,     # m/s
                    center_freq=self.center_freq,
                )#(153600,)
                self.num_rx = 1
                rx_signal = rx_signal.reshape(1, -1)
                targets = []
            else:
                self.idle_time_ratio = 0.2
                #tx_signal = self._generate_tx_signal(return_full=True, idle_time_ratio=self.idle_time_ratio, edge_ratio=0, window_type=None) #(128, 400) complex
            #(153600,)
                tx_signal = self._generate_tx_signal2(return_full=True) #(128, 400) complex
            #(153600,)
                # Extract a single chirp from the full TX signal for visualization
                tx_chirp = tx_signal[:self.active_samples]  # Get first chirp's active samples
                self.visualize_tx_chirp_with_window(
                    tx_signal=tx_chirp,  # Pass the extracted TX chirp
                    sample_rate=self.sample_rate,
                    active_samples=self.active_samples,
                    bandwidth=self.bandwidth,
                    slope=self.slope,
                    center_freq=self.center_freq,
                    edge_ratio=0.1,
                    window_type='edge'
                )
                # Generate random targets
                self.max_targets =1 
                targets = self._generate_random_targets()
                # Perform ray-tracing simulation
                rx_signal = self._ray_tracing_simulation(tx_signal, targets, perfect_mode=True, flatten_output=True)
                #The shape should be (4, 153600) [num_rx, num_chirps*samples_per_chirp]

            # Add noise to the received signal (even in perfect mode, we need some minimal noise)
            snr_db = 40 #random.uniform(self.snr_min, self.snr_max)
            rx_signal = self._add_noise(rx_signal, snr_db)

            # Demodulate the signal to baseband
            # Demodulate the signal to baseband, reusing the TX signal
            #only for one RX antenna
            beat_signal = self.fmcw_demodulate_new(tx_full=tx_signal, \
                        rx_full=rx_signal[0,:], \
                        total_samples_per_chirp=self.total_samples_per_chirp, \
                        beat_samples_per_chirp=self.active_samples, \
                        num_chirps=self.num_chirps)
            # Reshape to add RX dimension: (num_rx, num_chirps, samples_per_chirp)
            beat_signal = np.expand_dims(beat_signal, axis=0)  # Shape becomes (1, num_chirps, samples_per_chirp)
            self.num_rx = 1
            #beat_signal = self.fmcw_demodulate(rx_signal, tx_signal)
            #Beat signal with shape (4, 128, 1000) [num_rx, num_chirps, samples_per_chirp]
            if visualize: # and i == 0:  # Only for the first sample to avoid too many plots
                self._visualize_beat_signal(
                    tx_signal=tx_signal,  # Pass complete signals
                    rx_signal=rx_signal,               # Complete RX signal
                    beat_signal=beat_signal,             # Complete beat signal
                    total_samples_per_chirp=self.total_samples_per_chirp,
                    activesamples_per_chirp=self.active_samples,
                    total_chirp_duration=self.total_chirp_duration,
                    slope=self.slope,
                    c=self.speed_of_light,
                    sample_rate=self.sample_rate,
                    sample_idx=i,
                    chirp_idx=0,
                    rx_idx=0
                )
            # Process the received signal to generate range-Doppler map
            #rd_map = self._time_to_range_doppler(rx_signal) #(4, 128, 400) complex
            #(2, 128, 256)
            # Process the received signal to generate range-Doppler map
            self.apply_doppler_centering = True
            rd_map = self._time_to_range_doppler(
                rx_signal=beat_signal,  # Use demodulated signal instead of raw RX,
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

            # Perform target detection using CFAR
            detection_results = self._cfar_detection(rd_map)#(2, 128, 256)
            
            # Visualize if requested
            if visualize: # and (i % 10 == 0 or i == self.num_samples - 1):
                self._visualize_sample(i, self.chirp_duration,
                        self.total_samples_per_chirp, tx_signal, rx_signal, rd_map, targets, detection_results, save_path=self.save_path)
                        # Create target mask (ground truth)
            if targets is not None:
                target_mask = self._create_target_mask(targets) #(128, 256, 1)
                # Store the partially flattened rx_signal
                dataset['time_domain_data'][i, :, :, 0] = np.real(rx_signal)
                dataset['time_domain_data'][i, :, :, 1] = np.imag(rx_signal)
                dataset['range_doppler_maps'][i] = rd_map
                dataset['target_masks'][i] = target_mask
                dataset['target_info'].append(targets)
                dataset['detection_results'].append(detection_results)
                
            
        print("Dataset generation complete!")
        return dataset
    
    def fmcw_demodulate_new(self, tx_full, rx_full, total_samples_per_chirp, beat_samples_per_chirp, num_chirps):
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

    def new_pipeline(self):
        
        
        def _time_to_range_doppler(beat_signals, range_fft_size=256, doppler_fft_size=64):
            """
            Convert beat signal to Range-Doppler map using 2D FFT.
            """
            range_fft = np.fft.fftshift(np.fft.fft(beat_signals, n=range_fft_size, axis=1), axes=1)
            doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, n=doppler_fft_size, axis=0), axes=0)
            rd_map = np.abs(doppler_fft)
            return rd_map
        
        def cfar_detect_rd_map(rd_map, guard_cells=2, training_cells=8, threshold_scale=5.0):
            """
            Apply simple 2D Cell-Averaging CFAR to the RD map.
            Returns list of detected (range_idx, doppler_idx).
            """
            detections = []
            doppler_bins, range_bins = rd_map.shape

            for d in range(training_cells + guard_cells, doppler_bins - (training_cells + guard_cells)):
                for r in range(training_cells + guard_cells, range_bins - (training_cells + guard_cells)):
                    # Extract training region
                    window = rd_map[d - training_cells - guard_cells : d + training_cells + guard_cells + 1,
                                    r - training_cells - guard_cells : r + training_cells + guard_cells + 1]

                    cut = rd_map[d, r]
                    guard_slice = slice(training_cells, training_cells + 2 * guard_cells + 1)
                    window[guard_slice, guard_slice] = 0  # Zero guard + CUT

                    noise_level = np.mean(window)
                    threshold = noise_level * threshold_scale

                    if cut > threshold:
                        detections.append((r, d))

            return detections
        
        def visualize_fmcw_processing_with_detections(
            beat_signal,
            rd_map,
            detections=None,
            fs=1e6,
            slope=50e12,
            fc=77e9,
            range_fft_size=256,
            doppler_fft_size=64,
            chirp_duration=40e-6,
            save_path=None
        ):
            c = 3e8  # Speed of light

            # FFT for beat signal spectrum
            beat_spectrum = np.fft.fftshift(np.fft.fft(beat_signal))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(beat_signal), d=1/fs))

            # Range and Doppler axes
            freq_axis = np.fft.fftshift(np.fft.fftfreq(range_fft_size, d=1/fs))
            range_axis = freq_axis * c / (2 * slope)

            prf = 1 / chirp_duration
            doppler_freqs = np.fft.fftshift(np.fft.fftfreq(doppler_fft_size, d=1/prf))
            velocity_axis = doppler_freqs * c / (2 * fc)

            # Plot
            plt.figure(figsize=(15, 4))

            # Time-domain beat signal
            plt.subplot(1, 3, 1)
            plt.plot(np.real(beat_signal), label='Real')
            plt.plot(np.imag(beat_signal), label='Imag', linestyle='--')
            plt.title('Beat Signal (Time Domain)')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend()

            # Frequency-domain
            plt.subplot(1, 3, 2)
            plt.plot(freqs / 1e3, 20 * np.log10(np.abs(beat_spectrum) + 1e-6))
            plt.title('Beat Spectrum')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)

            # Range-Doppler map
            plt.subplot(1, 3, 3)
            plt.imshow(20 * np.log10(rd_map + 1e-6),
                    extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
                    aspect='auto',
                    cmap='jet',
                    origin='lower')
            plt.title('Range-Doppler Map')
            plt.xlabel('Range (m)')
            plt.ylabel('Velocity (m/s)')
            plt.colorbar(label='Magnitude (dB)')

            if detections:
                for r_bin, d_bin in detections:
                    if 0 <= r_bin < len(range_axis) and 0 <= d_bin < len(velocity_axis):
                        r = range_axis[r_bin]
                        v = velocity_axis[d_bin]
                        plt.plot(r, v, 'rx', markersize=8, markeredgewidth=2)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"[INFO] Figure saved to {save_path}")
                plt.close()
            else:
                plt.show()

        tx_full = self._generate_tx_signal(return_full=True) #(153600,) (4, 128, 400) complex
        rx_full = self.simulate_single_target_echo(
            tx_full=tx_full,
            fs=self.sample_rate,
            target_range=50,        # meters
            target_velocity=15,     # m/s
            center_freq=self.center_freq,
        )#(153600,)
        # Step 1: Demodulate
        beat_signals = self.fmcw_demodulate_new(tx_full, rx_full, fs=self.sample_rate, num_chirps=self.num_chirps, chirp_duration=self.chirp_duration)
        #(128, 1000)
        # Step 2: Convert to RD Map
        rd_map = _time_to_range_doppler(beat_signals)
        rd_map += np.random.normal(0, 1e-3, rd_map.shape)
        #(64, 256)
        # Step 3: Detect targets with CFAR
        detections = cfar_detect_rd_map(rd_map.copy())

        # Visualize
        visualize_fmcw_processing_with_detections(
            beat_signals, rd_map, detections=None,
            fs=self.sample_rate, slope=self.slope, fc=self.center_freq,
            save_path='data/rd_output.png'
        )


    def simulate_single_target_echo(self,
        tx_full,
        fs,
        target_range,
        target_velocity,
        center_freq,
        c=3e8,
        reflection_amplitude=1.0
    ):
        """
        Simulate the RX signal caused by a single moving target.

        Args:
            tx_full: 1D numpy array, transmitted signal with idle time
            fs: Sampling frequency (Hz)
            target_range: Target distance in meters
            target_velocity: Target radial velocity in m/s
            center_freq: Radar center frequency (Hz)
            slope: Chirp slope (Hz/s)
            c: Speed of light (m/s)
            reflection_amplitude: Complex amplitude scaling factor

        Returns:
            rx_signal: complex numpy array same shape as tx_full
        """

        # Total time delay
        tau = 2 * target_range / c  # seconds
        fd = 2 * target_velocity * center_freq / c  # Doppler frequency shift (Hz)

        # Create time vector
        t = np.arange(len(tx_full)) / fs

        # Apply delay using interpolation
        delayed_tx = scipy.signal.resample_poly(tx_full, up=1, down=1)
        delay_samples = int(np.round(tau * fs))
        if delay_samples >= len(tx_full):
            return np.zeros_like(tx_full)

        delayed_tx = np.roll(tx_full, delay_samples)
        delayed_tx[:delay_samples] = 0  # zero before delay

        # Apply Doppler shift
        doppler_phase = np.exp(1j * 2 * np.pi * fd * t)

        rx_signal = reflection_amplitude * delayed_tx * doppler_phase

        return rx_signal

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

    def _remove_direct_coupling(self, beat_signal):
        """
        Remove direct coupling/leakage component from beat signal
        
        Args:
            beat_signal: Complex beat signal array (num_rx, num_chirps, samples_per_chirp)
            
        Returns:
            Processed beat signal with direct coupling removed
        """
        # Create a copy of the input signal
        processed_signal = beat_signal.copy()
        
        # For each RX channel
        for rx_idx in range(beat_signal.shape[0]):
            # Compute average spectrum across all chirps
            avg_spectrum = np.zeros(self.samples_per_chirp, dtype=complex)
            for chirp_idx in range(beat_signal.shape[1]):
                avg_spectrum += np.fft.fft(beat_signal[rx_idx, chirp_idx])
            avg_spectrum /= beat_signal.shape[1]
            
            # Find the peak frequency bin
            peak_idx = np.argmax(np.abs(avg_spectrum))
            
            # Create notch filter centered at the peak frequency
            notch_width = 3  # Width of notch in frequency bins
            notch_filter = np.ones(self.samples_per_chirp, dtype=complex)
            for i in range(max(0, peak_idx-notch_width), min(self.samples_per_chirp, peak_idx+notch_width+1)):
                # Apply cosine taper for smooth transition
                notch_filter[i] = 0.5 * (1 - np.cos(np.pi * (i - (peak_idx-notch_width)) / (2*notch_width)))
            
            # Apply filter to each chirp
            for chirp_idx in range(beat_signal.shape[1]):
                spectrum = np.fft.fft(beat_signal[rx_idx, chirp_idx])
                filtered_spectrum = spectrum * notch_filter
                processed_signal[rx_idx, chirp_idx] = np.fft.ifft(filtered_spectrum)
        
        return processed_signal

    def _generate_fmcw_chirp(self, chirp_idx):
        """Generate a single FMCW chirp signal with phase continuity"""
        t = np.linspace(0, self.chirp_duration, self.samples_per_chirp)
        
        # Calculate phase with proper phase continuity between chirps
        freq_sweep = self.bandwidth/self.chirp_duration * t
        phase_accumulation = 2 * np.pi * chirp_idx * self.bandwidth * self.chirp_duration
        phase = 2 * np.pi * (self.center_freq * t + 0.5 * freq_sweep * t) + phase_accumulation
        
        return np.exp(1j * phase)

    def fmcw_demodulate(self, rx_signal, tx_signal=None):
        """
        Demodulate the received signal to baseband, handling flattened signals.
        
        Args:
            rx_signal: Received signal, can be:
                      - 1D array (fully flattened)
                      - 2D array [num_rx, num_chirps*samples_per_chirp] (partially flattened)
                      - 3D array [num_rx, num_chirps, samples_per_chirp] (standard format)
            tx_signal: Optional pre-generated TX signal, either flattened 1D array or 2D array [num_chirps, samples_per_chirp]
                    If None, will generate a new TX signal
            
        Returns:
            Beat signal with shape [num_rx, num_chirps, samples_per_chirp]
        """
        # Calculate timing parameters
        samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        idle_time_ratio = getattr(self, 'idle_time_ratio', 0.2)  # Default to 0.2 if not defined
        samples_per_idle = int(self.sample_rate * self.chirp_duration * idle_time_ratio)
        total_samples_per_chirp = samples_per_chirp + samples_per_idle
        
        # Check signal dimensions
        rx_ndim = rx_signal.ndim
        
        # Use provided TX signal or generate a new one if not provided
        if tx_signal is None:
            # Generate TX signal if not provided
            tx_signal = self._generate_tx_signal(return_full=True)  # Get flattened signal
            tx_is_flat = True
        else:
            tx_is_flat = tx_signal.ndim == 1
        
        # Initialize beat signal container with proper shape
        beat_signal = np.zeros((self.num_rx, self.num_chirps, samples_per_chirp), dtype=complex)
        
        if rx_ndim == 1:
            # Fully flattened RX signal (1D array)
            # Process each RX antenna (assuming flat signal contains all RX data concatenated)
            for rx_idx in range(self.num_rx):
                # Process each chirp
                for chirp_idx in range(self.num_chirps):
                    start = chirp_idx * total_samples_per_chirp
                    end = start + samples_per_chirp
                    
                    # Skip if we're beyond the signal length
                    if end > len(tx_signal) or end > len(rx_signal):
                        continue
                        
                    # Extract the relevant portions of the signals
                    if tx_is_flat:
                        tx_chirp = tx_signal[start:end]
                    else:
                        tx_chirp = tx_signal[chirp_idx]
                        
                    rx_chirp = rx_signal[start:end]
                    
                    # Demodulate by multiplying RX with conjugate of TX
                    beat_signal[rx_idx, chirp_idx] = rx_chirp * np.conj(tx_chirp)
        
        elif rx_ndim == 2:
            # Partially flattened RX signal [num_rx, num_chirps*samples_per_chirp]
            for rx_idx in range(self.num_rx):
                for chirp_idx in range(self.num_chirps):
                    # Calculate start and end indices for this chirp in the flattened array
                    start = chirp_idx * samples_per_chirp
                    end = start + samples_per_chirp
                    
                    # Skip if we're beyond the signal length
                    if end > rx_signal.shape[1]:
                        continue
                    
                    # Extract RX chirp from the flattened array for this antenna
                    rx_chirp = rx_signal[rx_idx, start:end]
                    
                    # Extract TX chirp
                    if tx_is_flat:
                        tx_start = chirp_idx * total_samples_per_chirp
                        tx_end = tx_start + samples_per_chirp
                        if tx_end > len(tx_signal):
                            continue
                        tx_chirp = tx_signal[tx_start:tx_end]
                    else:
                        tx_chirp = tx_signal[chirp_idx]
                    
                    # Demodulate by multiplying RX with conjugate of TX
                    beat_signal[rx_idx, chirp_idx] = rx_chirp * np.conj(tx_chirp)
        
        else:
            # Standard 3D format [num_rx, num_chirps, samples_per_chirp]
            # Process each chirp
            for chirp_idx in range(self.num_chirps):
                # Extract the TX chirp for this index
                if tx_is_flat:
                    start = chirp_idx * total_samples_per_chirp
                    end = start + samples_per_chirp
                    if end > len(tx_signal):
                        continue
                    tx_chirp = tx_signal[start:end]
                else:
                    tx_chirp = tx_signal[chirp_idx]
                
                for rx_idx in range(self.num_rx):
                    # Demodulate by multiplying RX with conjugate of TX
                    beat_signal[rx_idx, chirp_idx] = rx_signal[rx_idx, chirp_idx] * np.conj(tx_chirp)
        
        return beat_signal

    def _demodulate_fmcw_chirp(self, tx_chirp, rx_chirp, chirp_idx=0):
        """
        Perform FMCW de-chirping with proper phase handling, accounting for inter-chirp idle time.
        The function now accounts for phase accumulation between chirps, which is critical for accurate Doppler processing.
        By passing the chirp index, the demodulation can properly match the phase characteristics of the transmitted signal.

        Args:
            tx_chirp: Transmitted chirp signal
            rx_chirp: Received chirp signal
            chirp_idx: Index of current chirp (for phase continuity)
            
        Returns:
            Complex beat signal after mixing and filtering
        """
        # Mix transmitted and received signals (conjugate mixing)
        # This preserves the phase information needed for Doppler processing
        mixed = rx_chirp * np.conj(tx_chirp)
        
        # Apply phase correction to account for phase accumulation between chirps
        # This ensures phase continuity matching the _generate_tx_signal implementation
        # Now includes the effect of idle time in the phase accumulation
        # Use chirp_duration if chirp_repetition_interval is not defined
        repetition_interval = getattr(self, 'chirp_repetition_interval', self.chirp_duration)

        phase_accumulation = 2 * np.pi * chirp_idx * self.bandwidth * repetition_interval
        phase_correction = np.exp(-1j * phase_accumulation)
        
        # Apply phase correction to the mixed signal
        mixed_corrected = mixed * phase_correction
        
        # Apply center-emphasizing window function to the mixed signal
        # This emphasizes the center portion of the chirp where the signal is most stable
        n = len(mixed_corrected)
        center_window = np.blackman(n)  # Blackman window has good sidelobe suppression
        
        # Modify the window to emphasize the center more strongly
        # Create a center-weighted window by taking the window to a power
        center_emphasis = 0.5  # Adjust this value to control center emphasis (higher = more emphasis)
        center_window = center_window ** center_emphasis
        
        # Normalize the window
        center_window = center_window / np.max(center_window)
        
        # Apply the center-emphasizing window
        mixed_windowed = mixed_corrected * center_window
        
        # Apply low-pass filtering with better window function
        # The window size is calculated to prevent aliasing based on bandwidth
        window_size = max(3, int(self.sample_rate / (2 * self.bandwidth)))
        window = np.hamming(window_size) / np.sum(np.hamming(window_size))
        
        # Apply filtering to remove high-frequency components
        filtered_signal = np.convolve(mixed_windowed, window, mode='same')
        
        return filtered_signal

    # def _demodulate_signal(self, tx_chirp, rx_chirp):
    #     """Perform FMCW de-chirping (beat signal generation)"""
    #     # Mix transmitted and received signals (conjugate mixing)
    #     mixed = rx_chirp * np.conj(tx_chirp)
        
    #     # Apply low-pass filtering with better window function
    #     window_size = max(3, int(self.sample_rate / (2 * self.bandwidth)))  # anti-aliasing
    #     window = np.hamming(window_size) / np.sum(np.hamming(window_size))  # Normalized Hamming window
        
    #     # Apply filtering and return
    #     return np.convolve(mixed, window, mode='same')

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
        # Generate a single sample on-the-fly
        # Set random seed based on idx for reproducibility
        np.random.seed(idx)
        random.seed(idx)
        
        # Generate TX signal
        tx_signal = self._generate_tx_signal()
        
        # Generate random targets
        targets = self._generate_random_targets()
        
        # Perform ray-tracing simulation
        rx_signal = self._ray_tracing_simulation(tx_signal, targets, perfect_mode=True, flatten_output=True)
        
        # Add noise to the received signal
        snr_db = random.uniform(self.snr_min, self.snr_max)
        rx_signal = self._add_noise(rx_signal, snr_db)
        
        # Demodulate the signal to baseband
        beat_signal = np.zeros_like(rx_signal)
        for chirp_idx in range(self.num_chirps):
            tx_chirp = self._generate_fmcw_chirp(chirp_idx)
            for rx_idx in range(self.num_rx):
                beat_signal[rx_idx, chirp_idx] = self._demodulate_fmcw_chirp(
                    tx_chirp, rx_signal[rx_idx, chirp_idx], chirp_idx
                )
        
        # Update processing to use beat signal
        range_doppler_map = self._time_to_range_doppler(
            beat_signal,  # Use demodulated signal instead of raw RX
            
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
        
        # Ensure consistent dimensions for all samples
        # Ensure time_domain has shape [num_rx, num_chirps, samples_per_chirp, 2]
        if time_domain_data.shape != (self.num_rx, self.num_chirps, self.samples_per_chirp, 2):
            # Resize or pad to match expected dimensions
            correct_shape = (self.num_rx, self.num_chirps, self.samples_per_chirp, 2)
            temp_data = np.zeros(correct_shape, dtype=self.precision)
            # Copy what we can from the original data
            slices = tuple(slice(0, min(dim, src_dim)) for dim, src_dim in zip(correct_shape, time_domain_data.shape))
            temp_data[slices] = time_domain_data[slices]
            time_domain_data = temp_data
        
        # Ensure feature_2d has shape [2, num_doppler_bins, num_range_bins]
        if range_doppler_map.shape != (2, self.num_doppler_bins, self.num_range_bins):
            correct_shape = (2, self.num_doppler_bins, self.num_range_bins)
            temp_data = np.zeros(correct_shape, dtype=self.precision)
            # Copy what we can from the original data
            slices = tuple(slice(0, min(dim, src_dim)) for dim, src_dim in zip(correct_shape, range_doppler_map.shape))
            temp_data[slices] = range_doppler_map[slices]
            range_doppler_map = temp_data
        
        # Ensure labels has shape [num_doppler_bins, num_range_bins, 1]
        if target_mask.shape != (self.num_doppler_bins, self.num_range_bins, 1):
            correct_shape = (self.num_doppler_bins, self.num_range_bins, 1)
            temp_data = np.zeros(correct_shape, dtype=self.precision)
            # Copy what we can from the original data
            slices = tuple(slice(0, min(dim, src_dim)) for dim, src_dim in zip(correct_shape, target_mask.shape))
            temp_data[slices] = target_mask[slices]
            target_mask = temp_data
        
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

    def _generate_tx_signal2(self, tx_power=1.0, edge_ratio=0.1, window_type='edge', return_full=False):
        """
        Generate realistic FMCW transmit signal with optional edge windowing and idle gaps.

        Args:
            tx_power: Transmission power scale (float)
            idle_time_ratio: Idle time relative to chirp duration (0–1)
            edge_ratio: Proportion of chirp to taper at each edge (0–0.5)
            window_type: Windowing function ('edge', 'hann', 'hamming', or None)
            return_full: If True, return full TX signal; else per-chirp array

        Returns:
            - If return_full=False: [num_chirps, samples_per_chirp] complex array
            - If return_full=True: 1D waveform with all chirps concatenated
        """
        # Time and sweep rate for active portion
        total_samples_per_chirp = int(self.total_samples_per_chirp)
        active_samples = int(self.active_samples)
        t_active = np.arange(active_samples) / self.sample_rate
        sweep_rate = self.slope #self.bandwidth / self.chirp_duration
        
        # Generate base chirp (zero start phase)
        base_phase = 2 * np.pi * (0.5 * sweep_rate * t_active**2)
        active_chirp = np.exp(1j * base_phase)
        
        # Apply windowing to active portion
        if window_type:
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
                
            active_chirp *= window  # Apply window to amplitude
        
        # Create full chirp with idle time
        full_chirp = np.zeros(total_samples_per_chirp, dtype=np.complex128)
        full_chirp[:active_samples] = active_chirp
        
        # Power scaling
        scale = np.sqrt(tx_power)
        full_chirp *= scale
        
        # Allocate outputs
        if return_full:
            # Create a single continuous waveform with all chirps
            tx_full = np.zeros(self.num_chirps * total_samples_per_chirp, dtype=np.complex128)
            for i in range(self.num_chirps):
                start_idx = i * total_samples_per_chirp
                tx_full[start_idx:start_idx + total_samples_per_chirp] = full_chirp
            return tx_full
        else:
            # Create array of individual chirps
            tx_signal = np.zeros((self.num_chirps, total_samples_per_chirp), dtype=np.complex128)
            for i in range(self.num_chirps):
                tx_signal[i] = full_chirp
            return tx_signal

    def _generate_tx_signal(self, tx_power=1.0, idle_time_ratio=0.2, edge_ratio=0.1, window_type='edge', return_full=False):
        """
        Generate realistic FMCW transmit signal with optional edge windowing and idle gaps.

        Args:
            tx_power: Transmission power scale (float)
            idle_time_ratio: Idle time relative to chirp duration (0–1)
            edge_ratio: Proportion of chirp to taper at each edge (0–0.5)
            window_type: Windowing function ('edge', 'hann', 'hamming', or None)
            return_full: If True, return full TX signal with idle gaps; else per-chirp array

        Returns:
            - If return_full=False: [num_chirps, samples_per_chirp] complex array
            - If return_full=True: 1D waveform with idle segments between chirps
        """
        # Timing
        self.idle_time = self.chirp_duration * idle_time_ratio
        self.chirp_repetition_interval = self.chirp_duration + self.idle_time

        samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        samples_per_idle = int(self.sample_rate * self.idle_time)
        total_samples_per_interval = samples_per_chirp + samples_per_idle

        # Time and sweep rate
        t_chirp = np.arange(samples_per_chirp) / self.sample_rate
        sweep_rate = self.bandwidth / self.chirp_duration

        # Generate base chirp (zero start phase)
        base_phase = 2 * np.pi * (0.5 * sweep_rate * t_chirp**2)
        base_chirp = np.exp(1j * base_phase)

        # Apply windowing
        if window_type == 'hann':
            window = np.hanning(samples_per_chirp)
        elif window_type == 'hamming':
            window = np.hamming(samples_per_chirp)
        elif window_type == 'edge':
            # Enforce min edge length
            min_edge_len = 16
            edge_len = max(int(edge_ratio * samples_per_chirp), min_edge_len)

            # Ensure even length for symmetric hann windowing
            edge_len = edge_len if edge_len % 2 == 0 else edge_len + 1
            total_taper_len = 2 * edge_len

            if total_taper_len >= samples_per_chirp:
                # Fall back to full Hann window if taper too large
                window = np.hanning(samples_per_chirp)
            else:
                hann_win = np.hanning(total_taper_len)
                rise = hann_win[:edge_len]
                fall = hann_win[edge_len:]
                flat = np.ones(samples_per_chirp - total_taper_len)
                window = np.concatenate([rise, flat, fall])
        else:  # No windowing
            window = np.ones(samples_per_chirp)

        base_chirp *= window  # Apply window to amplitude

        # Allocate outputs
        tx_signal = np.zeros((self.num_chirps, samples_per_chirp), dtype=np.complex128)
        tx_full = np.zeros(self.num_chirps * total_samples_per_interval, dtype=np.complex128)

        for i in range(self.num_chirps):
            start_idx = i * total_samples_per_interval
            tx_signal[i] = base_chirp
            tx_full[start_idx:start_idx + samples_per_chirp] = base_chirp

        # Power scaling
        scale = np.sqrt(tx_power)
        tx_signal *= scale
        tx_full *= scale

        return tx_full if return_full else tx_signal
    
    def visualize_tx_chirp_with_window(self,
        tx_signal=None,  # Optional pre-generated TX signal
        sample_rate=None,
        active_samples=None,
        bandwidth=None,
        slope=None,
        center_freq=77e9,
        edge_ratio=0.1,
        min_edge_len=16,
        window_type='edge',
        noise_level=1e-6  # Add small noise level parameter
    ):
        """
        Visualize TX chirp generation with optional windowing and save the figure.

        Args:
            tx_signal: Optional pre-generated TX signal to visualize
            sample_rate: Sampling rate (Hz).
            active_samples: Number of active samples in the chirp.
            bandwidth: Sweep bandwidth (Hz).
            slope: Chirp slope (Hz/s).
            center_freq: RF carrier frequency (Hz), e.g., 77e9.
            edge_ratio: Proportion of chirp to taper on each edge.
            min_edge_len: Minimum number of samples in each taper.
            window_type: 'edge', 'hann', 'hamming', or None.
            noise_level: Small noise level to add for numerical stability (default: 1e-6)
        """
        # Use class attributes if parameters not provided
        sample_rate = sample_rate or self.sample_rate
        active_samples = active_samples or self.active_samples
        bandwidth = bandwidth or self.bandwidth
        slope = slope or self.slope
        
        # Create directory for visualizations if it doesn't exist
        os.makedirs(os.path.join(self.save_path, 'visualizations'), exist_ok=True)

        # Signal setup
        samples = active_samples
        t = np.arange(samples) / sample_rate
        k = slope  # Chirp slope

        # If TX signal is provided, use it; otherwise generate one
        if tx_signal is not None:
            # Extract a single chirp if a full signal is provided
            if tx_signal.ndim > 1 or len(tx_signal) > samples:
                # Assume it's a flattened signal with multiple chirps
                tx_chirp_unwindowed = tx_signal[:samples]
            else:
                # Already a single chirp
                tx_chirp_unwindowed = tx_signal
        else:
            # Generate unwindowed base chirp
            phase = 2 * np.pi * (center_freq * t + 0.5 * k * t**2)
            tx_chirp_unwindowed = np.exp(1j * phase)

        # Add small noise for numerical stability in spectrum calculation
        noise_real = np.random.normal(0, noise_level, samples)
        noise_imag = np.random.normal(0, noise_level, samples)
        tx_chirp_unwindowed += noise_real + 1j * noise_imag
        
        # Window creation
        if window_type == 'hann':
            window = np.hanning(samples)
        elif window_type == 'hamming':
            window = np.hamming(samples)
        elif window_type == 'edge':
            edge_len = max(int(edge_ratio * samples), min_edge_len)
            edge_len += edge_len % 2  # Ensure even
            if 2 * edge_len >= samples:
                window = np.hanning(samples)
            else:
                hann_win = np.hanning(2 * edge_len)
                rise = hann_win[:edge_len]
                fall = hann_win[edge_len:]
                flat = np.ones(samples - 2 * edge_len)
                window = np.concatenate([rise, flat, fall])
        else:
            window = np.ones(samples)

        # Apply window
        tx_chirp_windowed = tx_chirp_unwindowed * window

        # FFT parameters
        N_fft = 8192
        freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate)) + center_freq
        freqs = freqs / 1e9  # Convert to GHz

        fft_unwindowed = np.fft.fftshift(np.fft.fft(tx_chirp_unwindowed, n=N_fft))
        fft_windowed = np.fft.fftshift(np.fft.fft(tx_chirp_windowed, n=N_fft))

        # Plotting
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle("TX Chirp Windowing Visualization", fontsize=16)

        # Time-domain original
        axs[0, 0].plot(t * 1e6, np.real(tx_chirp_unwindowed), label='Real', alpha=0.7)
        axs[0, 0].plot(t * 1e6, np.imag(tx_chirp_unwindowed), '--', label='Imag', alpha=0.7)
        axs[0, 0].set_title("Original TX Chirp (Time Domain)")
        axs[0, 0].set_xlabel("Time (µs)")
        axs[0, 0].set_ylabel("Amplitude")
        axs[0, 0].legend()
        axs[0, 0].grid()

        # Window shape
        axs[1, 0].plot(t * 1e6, window, color='orange')
        axs[1, 0].set_title("Window Function")
        axs[1, 0].set_xlabel("Time (µs)")
        axs[1, 0].set_ylabel("Amplitude")
        axs[1, 0].grid()

        # Time-domain windowed
        axs[2, 0].plot(t * 1e6, np.real(tx_chirp_windowed), label='Real', alpha=0.7)
        axs[2, 0].plot(t * 1e6, np.imag(tx_chirp_windowed), '--', label='Imag', alpha=0.7)
        axs[2, 0].set_title("Windowed TX Chirp (Time Domain)")
        axs[2, 0].set_xlabel("Time (µs)")
        axs[2, 0].set_ylabel("Amplitude")
        axs[2, 0].legend()
        axs[2, 0].grid()

        # Spectrum unwindowed
        axs[0, 1].plot(freqs, 20 * np.log10(np.abs(fft_unwindowed) + 1e-10))
        axs[0, 1].set_title("Original TX Chirp Spectrum")
        axs[0, 1].set_xlabel("Frequency (GHz)")
        axs[0, 1].set_ylabel("Magnitude (dB)")
        axs[0, 1].grid()
        
        # Set y-axis limits to focus on the relevant part of the spectrum
        y_min = max(-100, np.min(20 * np.log10(np.abs(fft_unwindowed) + 1e-10)))
        y_max = np.max(20 * np.log10(np.abs(fft_unwindowed) + 1e-10)) + 10
        axs[0, 1].set_ylim([y_min, y_max])
        
        # Zoom x-axis to focus on the bandwidth region with some margin
        f_start = center_freq - bandwidth / 2
        f_end = center_freq + bandwidth / 2
        margin = bandwidth * 0.5  # 50% margin on each side
        axs[0, 1].set_xlim([(f_start - margin) / 1e9, (f_end + margin) / 1e9])

        # Spectrum windowed
        axs[2, 1].plot(freqs, 20 * np.log10(np.abs(fft_windowed) + 1e-10))
        axs[2, 1].set_title("Windowed TX Chirp Spectrum")
        axs[2, 1].set_xlabel("Frequency (GHz)")
        axs[2, 1].set_ylabel("Magnitude (dB)")
        axs[2, 1].grid()
        
        # Apply same y-axis limits to windowed spectrum
        y_min_w = max(-100, np.min(20 * np.log10(np.abs(fft_windowed) + 1e-10)))
        y_max_w = np.max(20 * np.log10(np.abs(fft_windowed) + 1e-10)) + 10
        axs[2, 1].set_ylim([y_min_w, y_max_w])
        
        # Apply same x-axis zoom
        axs[2, 1].set_xlim([(f_start - margin) / 1e9, (f_end + margin) / 1e9])

        # Highlight chirp bandwidth with more visible lines and shaded region
        for ax in [axs[0, 1], axs[2, 1]]:
            # Add vertical lines for bandwidth boundaries
            ax.axvline(f_start / 1e9, color='red', linestyle='--', linewidth=1.5, 
                    label=f'Start: {f_start/1e9:.2f} GHz')
            ax.axvline(f_end / 1e9, color='green', linestyle='--', linewidth=1.5,
                    label=f'End: {f_end/1e9:.2f} GHz')
            
            # Add shaded region to highlight bandwidth
            ax.axvspan(f_start / 1e9, f_end / 1e9, alpha=0.2, color='yellow')
            
            # Add legend
            ax.legend(loc='upper right', fontsize='small')
            
            # Add text annotation for bandwidth
            ax.text(center_freq / 1e9, y_min + (y_max - y_min) * 0.1, 
                f'BW: {bandwidth/1e6:.1f} MHz', 
                bbox=dict(facecolor='white', alpha=0.7),
                horizontalalignment='center')

        # Hide empty plot
        axs[1, 1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(self.save_path, "visualizations", "tx_chirp_window_visualization_rf.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved TX chirp visualization to: {out_path}")
        
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
    
    def _time_to_range_doppler(self, rx_signal,
                          num_chirps,
                          samples_per_chirp,
                          num_doppler_bins,
                          num_range_bins,
                          rx_idx = 0,
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
            Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
        """
        # Check if input is flattened format and reshape if needed
        if rx_signal.ndim == 2 and rx_signal.shape[1] == num_chirps * samples_per_chirp:
            # Reshape from [num_rx, num_chirps * samples_per_chirp] to [num_rx, num_chirps, samples_per_chirp]
            rx_signal = rx_signal.reshape(rx_signal.shape[0], num_chirps, samples_per_chirp)
        
        # Initialize range-Doppler map
        rd_map = np.zeros((2, num_doppler_bins, num_range_bins), dtype=np.float32)
        
        # Process first RX antenna only
        #rx_idx = 0
        processed_signal = rx_signal[rx_idx]
        
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
        
        # Apply notch filter to suppress zero-Doppler if requested
        if apply_notch_filter:
            center_bin = num_doppler_bins // 2 if apply_doppler_centering else 0
            
            # Create notch filter
            notch_filter = np.ones(num_doppler_bins)
            notch_filter[center_bin-notch_width:center_bin+notch_width+1] = 0
            
            # Apply notch filter to each range bin
            for range_bin in range(doppler_fft.shape[1]):
                doppler_fft[:, range_bin] *= notch_filter
        
        # Calculate magnitude
        magnitude = np.abs(doppler_fft)
        
        # Normalize magnitude based on dynamic range if requested
        if dynamic_range_db > 0:
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            noise_floor = np.percentile(magnitude_db, 10)  # Estimate noise floor
            magnitude_norm = np.clip(magnitude_db, noise_floor, noise_floor + dynamic_range_db)
            magnitude_norm = (magnitude_norm - noise_floor) / dynamic_range_db * 100
            rd_map[0] = magnitude_norm
        else:
            # Just use raw magnitude
            rd_map[0] = magnitude
        
        # Store phase information (normalized to [0, 1])
        phase = np.angle(doppler_fft) / (2 * np.pi) + 0.5
        rd_map[1] = phase
        
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
    
    def _visualize_beat_signal(self, tx_signal, rx_signal, beat_signal, total_samples_per_chirp, activesamples_per_chirp, total_chirp_duration, slope, c, sample_rate, sample_idx=0, chirp_idx=0, rx_idx=0):
        """Visualize the beat signal for a specific chirp with improved FFT resolution and frequency accuracy."""
        os.makedirs(os.path.join(self.save_path, 'visualizations'), exist_ok=True)

        # FFT size for improved resolution
        N_fft = 8192

        # TX extraction
        if tx_signal.ndim == 1:
            start = chirp_idx * total_samples_per_chirp
            end = start + total_samples_per_chirp
            tx_chirp = tx_signal[start:end] if end <= len(tx_signal) else np.zeros(activesamples_per_chirp, dtype=complex)
        else:
            tx_chirp = tx_signal[chirp_idx] if chirp_idx < tx_signal.shape[0] else np.zeros(activesamples_per_chirp, dtype=complex)

        # RX extraction
        if rx_signal.ndim == 1:
            start = chirp_idx * total_samples_per_chirp
            end = start + total_samples_per_chirp
            rx_chirp = rx_signal[start:end] if end <= len(rx_signal) else np.zeros(activesamples_per_chirp, dtype=complex)
        elif rx_signal.ndim == 2:
            start = chirp_idx * activesamples_per_chirp
            end = start + total_samples_per_chirp
            rx_chirp = rx_signal[rx_idx, start:end] if rx_idx < rx_signal.shape[0] else np.zeros(activesamples_per_chirp, dtype=complex)
        else:
            rx_chirp = rx_signal[rx_idx, chirp_idx] if rx_idx < rx_signal.shape[0] and chirp_idx < rx_signal.shape[1] else np.zeros(activesamples_per_chirp, dtype=complex)

        # Beat signal
        beat_chirp = beat_signal[rx_idx, chirp_idx] if rx_idx < beat_signal.shape[0] and chirp_idx < beat_signal.shape[1] else np.zeros(activesamples_per_chirp, dtype=complex)

        # Add small noise for numerical stability in spectrum calculation
        noise_level = 1e-6
        for signal in [tx_chirp, rx_chirp, beat_chirp]:
            noise_real = np.random.normal(0, noise_level, len(signal))
            noise_imag = np.random.normal(0, noise_level, len(signal))
            signal += noise_real + 1j * noise_imag

        # Time and frequency axes
        t = np.linspace(0, total_chirp_duration, total_samples_per_chirp)
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate)) / 1e6  # MHz

        # Apply window functions to reduce spectral leakage
        window_blackman = np.blackman(len(tx_chirp))
        
        # Original and windowed signals
        tx_chirp_orig = tx_chirp.copy()
        rx_chirp_orig = rx_chirp.copy()
        beat_chirp_orig = beat_chirp.copy()
        
        # Apply window to copies for FFT
        tx_chirp_windowed = tx_chirp * window_blackman
        rx_chirp_windowed = rx_chirp * window_blackman
        beat_chirp_windowed = beat_chirp * window_blackman

        # FFTs with zero-padding (original and windowed)
        tx_fft_orig = np.fft.fftshift(np.fft.fft(tx_chirp_orig, n=N_fft))
        rx_fft_orig = np.fft.fftshift(np.fft.fft(rx_chirp_orig, n=N_fft))
        beat_fft_orig = np.fft.fftshift(np.fft.fft(beat_chirp_orig, n=N_fft))
        
        tx_fft = np.fft.fftshift(np.fft.fft(tx_chirp_windowed, n=N_fft))
        rx_fft = np.fft.fftshift(np.fft.fft(rx_chirp_windowed, n=N_fft))
        beat_fft = np.fft.fftshift(np.fft.fft(beat_chirp_windowed, n=N_fft))

        # Calculate spectra in dB
        tx_spectrum_orig = 20 * np.log10(np.abs(tx_fft_orig) + 1e-10)
        rx_spectrum_orig = 20 * np.log10(np.abs(rx_fft_orig) + 1e-10)
        beat_spectrum_orig = 20 * np.log10(np.abs(beat_fft_orig) + 1e-10)
        
        tx_spectrum = 20 * np.log10(np.abs(tx_fft) + 1e-10)
        rx_spectrum = 20 * np.log10(np.abs(rx_fft) + 1e-10)
        beat_spectrum = 20 * np.log10(np.abs(beat_fft) + 1e-10)
        
        # Normalize spectra for better comparison
        # First, find the maximum value across both TX and RX spectra
        max_tx_rx = max(np.max(tx_spectrum), np.max(rx_spectrum))
        
        # Normalize both spectra to the same reference
        tx_spectrum_norm = tx_spectrum - np.max(tx_spectrum) + max_tx_rx
        rx_spectrum_norm = rx_spectrum - np.max(rx_spectrum) + max_tx_rx

        # Create figure
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Beat Signal Analysis - Sample {sample_idx}, Chirp {chirp_idx}, RX {rx_idx}', fontsize=16)

        # TX Time Domain
        axs[0, 0].plot(t, np.real(tx_chirp_orig), 'b-', label='Real', alpha=0.7)
        axs[0, 0].plot(t, np.imag(tx_chirp_orig), 'r--', label='Imag', alpha=0.7)
        axs[0, 0].plot(t, window_blackman * np.max(np.abs(tx_chirp_orig)), 'g-', label='Window', alpha=0.3)
        axs[0, 0].set_title('TX Signal (Time Domain)')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Amplitude')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # TX Spectrum - with and without windowing
        axs[0, 1].plot(freq_axis, tx_spectrum_orig, 'b-', label='Original', alpha=0.5)
        axs[0, 1].plot(freq_axis, tx_spectrum, 'r-', label='Blackman Window', alpha=0.8)
        axs[0, 1].set_title('TX Signal Spectrum (Windowed vs Original)')
        axs[0, 1].set_xlabel('Frequency (MHz)')
        axs[0, 1].set_ylabel('Magnitude (dB)')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # Set y-axis limits to focus on the relevant part of the spectrum
        y_min_tx = max(-100, np.min(tx_spectrum))
        y_max_tx = np.max(tx_spectrum) + 10
        axs[0, 1].set_ylim([y_min_tx, y_max_tx])
        
        # Highlight bandwidth region in TX spectrum
        bandwidth_mhz = self.bandwidth / 1e6
        f_start = -bandwidth_mhz / 2  # Convert to MHz
        f_end = bandwidth_mhz / 2
        axs[0, 1].axvline(f_start, color='red', linestyle='--', linewidth=1.5, 
                        label=f'Start: {f_start:.2f} MHz')
        axs[0, 1].axvline(f_end, color='green', linestyle='--', linewidth=1.5,
                        label=f'End: {f_end:.2f} MHz')
        axs[0, 1].axvspan(f_start, f_end, alpha=0.2, color='yellow')
        
        # Zoom in on bandwidth region with margin
        margin = bandwidth_mhz * 0.5  # 50% margin
        axs[0, 1].set_xlim([f_start - margin, f_end + margin])

        # RX Time Domain
        axs[1, 0].plot(t, np.real(rx_chirp_orig), 'g-', label='Real', alpha=0.7)
        axs[1, 0].plot(t, np.imag(rx_chirp_orig), 'g--', label='Imag', alpha=0.7)
        axs[1, 0].plot(t, window_blackman * np.max(np.abs(rx_chirp_orig)), 'b-', label='Window', alpha=0.3)
        axs[1, 0].set_title('RX Signal (Time Domain)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Amplitude')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # RX Spectrum - with and without windowing
        axs[1, 1].plot(freq_axis, rx_spectrum_orig, 'g-', label='Original', alpha=0.5)
        axs[1, 1].plot(freq_axis, rx_spectrum, 'r-', label='Blackman Window', alpha=0.8)
        axs[1, 1].set_title('RX Signal Spectrum (Windowed vs Original)')
        axs[1, 1].set_xlabel('Frequency (MHz)')
        axs[1, 1].set_ylabel('Magnitude (dB)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        # Apply same y-axis limits and zoom as TX for comparison
        axs[1, 1].set_ylim([y_min_tx, y_max_tx])
        axs[1, 1].set_xlim([f_start - margin, f_end + margin])
        
        # Highlight bandwidth region in RX spectrum
        axs[1, 1].axvline(f_start, color='red', linestyle='--', linewidth=1.5)
        axs[1, 1].axvline(f_end, color='green', linestyle='--', linewidth=1.5)
        axs[1, 1].axvspan(f_start, f_end, alpha=0.2, color='yellow')

        # Beat Time Domain
        axs[2, 0].plot(t[0:len(beat_chirp)], np.real(beat_chirp_orig), 'b-', label='Real', alpha=0.7)
        axs[2, 0].plot(t[0:len(beat_chirp)], np.imag(beat_chirp_orig), 'b--', label='Imag', alpha=0.7)
        if len(beat_chirp) == len(window_blackman):
            axs[2, 0].plot(t[0:len(beat_chirp)], window_blackman * np.max(np.abs(beat_chirp_orig)), 'g-', label='Window', alpha=0.3)
        axs[2, 0].set_title('Beat Signal (Time Domain)')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Amplitude')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Beat Spectrum - with and without windowing
        axs[2, 1].plot(freq_axis, beat_spectrum_orig, 'b-', label='Original', alpha=0.5)
        axs[2, 1].plot(freq_axis, beat_spectrum, 'r-', label='Blackman Window', alpha=0.8)
        axs[2, 1].set_title('Beat Signal Spectrum (Windowed vs Original)')
        axs[2, 1].set_xlabel('Frequency (MHz)')
        axs[2, 1].set_ylabel('Magnitude (dB)')
        axs[2, 1].grid(True)
        axs[2, 1].legend()
        
        # Set y-axis limits for beat spectrum
        y_min_beat = max(-100, np.min(beat_spectrum))
        y_max_beat = np.max(beat_spectrum) + 10
        axs[2, 1].set_ylim([y_min_beat, y_max_beat])
        
        # Find beat frequency from windowed spectrum for better accuracy
        beat_freq_idx = np.argmax(np.abs(beat_fft[N_fft // 2:]))  # Peak in positive freq
        beat_freq = np.fft.fftshift(np.fft.fftfreq(N_fft, 1 / sample_rate))[N_fft // 2 + beat_freq_idx]
        beat_freq_mhz = beat_freq / 1e6
        
        # Set x-axis limits to focus on the beat frequency with some margin
        margin_beat = 20  # MHz
        axs[2, 1].set_xlim([beat_freq_mhz - margin_beat, beat_freq_mhz + margin_beat])
        
        # Highlight beat frequency
        axs[2, 1].axvline(beat_freq_mhz, color='red', linestyle='-', linewidth=2, 
                        label=f'Beat: {beat_freq_mhz:.2f} MHz')
        
        # Estimate range from beat frequency
        estimated_range = (beat_freq * c) / (2 * slope)

        # Display radar parameters
        textstr = '\n'.join((
            f'Chirp Duration: {self.total_chirp_duration * 1e6:.1f} μs',
            f'Sample Rate: {sample_rate / 1e6:.1f} MHz',
            f'Slope: {slope / 1e12:.2f} THz/s',
            f'Peak Beat Freq: {beat_freq / 1e3:.2f} kHz',
            f'Est. Range: {estimated_range:.2f} m',
            f'Window: Blackman'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[2, 1].text(0.05, 0.95, textstr, transform=axs[2, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        # Save and clean up
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_name = f'beat_signal_s{sample_idx}_c{chirp_idx}_rx{rx_idx}.png'
        plt.savefig(os.path.join(self.save_path, 'visualizations', save_name))
        plt.close()

    def _visualize_sample(self, sample_idx, 
                        chirp_duration,
                        samples_per_chirp,
                        tx_signal, 
                        rx_signal, 
                        rd_map, 
                        targets, 
                        detection_results,
                        save_path):
        """
        Visualize radar data for a single sample.
        
        Args:
            sample_idx: Sample index
            tx_signal: Transmitted signal, either flattened 1D array or 2D array [num_chirps, samples_per_chirp]
            rx_signal: Received signal, can be:
                      - 2D array [num_rx, num_chirps*samples_per_chirp] (partially flattened)
                      - 3D array [num_rx, num_chirps, samples_per_chirp] (standard format)
            rd_map: Range-Doppler map
            targets: List of ground truth targets
            detection_results: List of detected targets
        """
        # Create directory for visualizations
        os.makedirs(os.path.join(save_path, 'visualizations'), exist_ok=True)
        
        # Time vector for one chirp (in microseconds)
        t = np.linspace(0, chirp_duration * 1e6, samples_per_chirp)
        
        # Extract first chirp from TX signal based on its format
        if tx_signal.ndim == 1:  # Flattened TX signal
            # Extract first chirp from flattened TX signal
            tx_first_chirp = tx_signal[:samples_per_chirp]
        else:  # Already in [num_chirps, samples_per_chirp] format
            tx_first_chirp = tx_signal[0]
        
        # Extract first chirp from RX signal based on its format
        if rx_signal.ndim == 2:  # Partially flattened RX signal [num_rx, num_chirps*samples_per_chirp]
            # Extract first chirp for first RX antenna from flattened format
            rx_first_chirp = rx_signal[0, :samples_per_chirp]
        else:  # Standard 3D format [num_rx, num_chirps, samples_per_chirp]
            rx_first_chirp = rx_signal[0, 0]
        
        # Figure 1: TX/RX Signal Analysis
        fig1 = plt.figure(figsize=(12, 10))
        plt.suptitle(f"TX/RX Signal Analysis - Sample {sample_idx}", fontsize=16)
        
        # TX signal time domain
        plt.subplot(2, 2, 1)
        plt.plot(t, np.real(tx_first_chirp), 'b-', label='Real')
        plt.plot(t, np.imag(tx_first_chirp), 'r-', label='Imag')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('TX Signal (Time Domain)')
        plt.legend()
        plt.grid(True)
        
        # TX signal instantaneous frequency
        plt.subplot(2, 2, 2)
        # Calculate instantaneous frequency by taking the derivative of the phase
        inst_phase_tx = np.unwrap(np.angle(tx_first_chirp))
        inst_freq_tx = np.diff(inst_phase_tx) / (2 * np.pi * (chirp_duration / samples_per_chirp))
        # Pad with the first value to maintain array size
        inst_freq_tx = np.concatenate(([inst_freq_tx[0]], inst_freq_tx))
        
        plt.plot(t, inst_freq_tx / 1e6, 'g-')  # Convert to MHz for display
        plt.xlabel('Time (μs)')
        plt.ylabel('Frequency (MHz)')
        plt.title('TX FMCW Instantaneous Frequency')
        plt.grid(True)
        
        # RX signal time domain (first RX antenna, first chirp)
        plt.subplot(2, 2, 3)
        plt.plot(t, np.real(rx_first_chirp), 'b-', label='Real')
        plt.plot(t, np.imag(rx_first_chirp), 'r-', label='Imag')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('RX Signal (Time Domain)')
        plt.legend()
        plt.grid(True)
        
        # RX signal instantaneous frequency
        plt.subplot(2, 2, 4)
        # Calculate instantaneous frequency by taking the derivative of the phase
        inst_phase_rx = np.unwrap(np.angle(rx_first_chirp))
        inst_freq_rx = np.diff(inst_phase_rx) / (2 * np.pi * (chirp_duration / samples_per_chirp))
        # Pad with the first value to maintain array size
        inst_freq_rx = np.concatenate(([inst_freq_rx[0]], inst_freq_rx))
        
        plt.plot(t, inst_freq_rx / 1e6, 'g-')  # Convert to MHz for display
        plt.xlabel('Time (μs)')
        plt.ylabel('Frequency (MHz)')
        plt.title('RX FMCW Instantaneous Frequency')
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(save_path, 'visualizations', f'sample_{sample_idx}_signals.png'))
        plt.close(fig1)
        
# Calculate distance and velocity axes based on whether FFT shifting was applied
        if hasattr(self, 'apply_doppler_centering') and self.apply_doppler_centering:
            # For shifted FFT (zero frequency at center)
            # For distance, we need to center the axis around max_range/2
            distance_axis = np.linspace(-self.max_range/2, self.max_range/2, self.num_range_bins)
            # For centered velocity axis, we need to go from -max_velocity to +max_velocity
            velocity_axis = np.linspace(-self.max_velocity, self.max_velocity, self.num_doppler_bins)
        else:
            # For non-shifted FFT (zero frequency at start)
            distance_axis = np.linspace(0, self.max_range, self.num_range_bins)
            # For non-centered velocity, positive velocities are in the first half, negative in the second half
            # We need to create the correct ordering: [0 to max_velocity, -max_velocity to 0)
            positive_velocities = np.linspace(0, self.max_velocity, self.num_doppler_bins // 2)
            negative_velocities = np.linspace(-self.max_velocity, 0, self.num_doppler_bins // 2 + 1)[:-1]
            velocity_axis = np.concatenate([positive_velocities, negative_velocities])
        # Figure 1: Magnitude-only Range-Doppler Map
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"Range-Doppler Map (Magnitude) - Sample {sample_idx}", fontsize=16)

        # Get magnitude from first channel
        magnitude_only = rd_map[0]
        magnitude_db = 20 * np.log10(magnitude_only + 1e-10)
        vmin_mag = np.max(magnitude_db) - 40  # Dynamic range of 40 dB
        
        # Plot with physical units on axes
        plt.imshow(magnitude_db, aspect='auto', cmap='jet', vmin=vmin_mag,
                  extent=[distance_axis[0], distance_axis[-1], velocity_axis[0], velocity_axis[-1]])
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Distance (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Range-Doppler Map (Magnitude Only)')
        
        # Add ground truth targets
        for target in targets:
            if target is None:
                continue
            plt.plot(target['distance'], target['velocity'], 'ro', markersize=8, markeredgecolor='white')
            plt.text(target['distance'] + 2, target['velocity'], 
                  f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                  color='white', fontsize=8, backgroundcolor='black')
        
        # Add detections if available
        if detection_results:
            for detection in detection_results:
                # Handle the dictionary format returned by _cfar_detection
                if isinstance(detection, dict) and 'distance' in detection and 'velocity' in detection:
                    # Use the physical values directly
                    r = detection['distance']
                    v = detection['velocity']
                    r_bin = detection['range_bin']
                    d_bin = detection['doppler_bin']
                    snr = detection.get('snr', 0)
                    
                    plt.plot(r, v, 'gx', markersize=8, markeredgewidth=2)
                    plt.text(r + 2, v - 2, 
                          f"Bin: ({r_bin},{d_bin})\nSNR: {snr:.1f}dB", 
                          color='green', fontsize=8, backgroundcolor='black')
                # Fallback for older format (r_bin, d_bin) tuples
                elif isinstance(detection, (list, tuple)) and len(detection) >= 2:
                    r_bin, d_bin = detection[0], detection[1]
                    if 0 <= r_bin < self.num_range_bins and 0 <= d_bin < self.num_doppler_bins:
                        r = distance_axis[r_bin]
                        v = velocity_axis[d_bin]
                        plt.plot(r, v, 'gx', markersize=8, markeredgewidth=2)
                        plt.text(r + 2, v - 2, f"Bin: ({r_bin},{d_bin})", 
                              color='green', fontsize=8, backgroundcolor='black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'visualizations', f'sample_{sample_idx}_rd_magnitude.png'))
        plt.close()
        
        # Figure 2: Phase information
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"Range-Doppler Phase - Sample {sample_idx}", fontsize=16)
        
        # Phase information is in the second channel
        phase_data = rd_map[1]
        
        # Plot with physical units on axes - use same axes as magnitude plot for consistency
        plt.imshow(phase_data, aspect='auto', cmap='hsv', vmin=0, vmax=1,
                  extent=[distance_axis[0], distance_axis[-1], velocity_axis[0], velocity_axis[-1]])
        plt.colorbar(label='Phase (normalized)')
        plt.xlabel('Distance (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Phase Information')
        
        # Add ground truth targets to phase plot
        for target in targets:
            if target is None:
                continue
            plt.plot(target['distance'], target['velocity'], 'ro', markersize=8, markeredgecolor='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'visualizations', f'sample_{sample_idx}_rd_phase.png'))
        plt.close()
        
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