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
        test_tx_signal = self._generate_tx_signal()
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
        
        # Generate samples
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            # Randomly select signal parameters for this sample
            #self._randomize_signal_parameters()
            
            # Generate random TX power for this sample
            tx_power = np.random.uniform(0.5, 1.5)
            
            # Generate TX signal with the random parameters
            tx_signal = self._generate_tx_signal(tx_power=tx_power) #(32, 150)
            self._visualize_tx_signal(tx_signal, title=f"TX Signal {i}", save_path=f"{self.save_path}/tx_signal_{i}{IMG_FORMAT}")
            
            # Initialize RX signal (all zeros)
            rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), 
                                dtype=np.complex64)
            
            # Generate random number of targets (0 to max_targets)
            num_targets = random.randint(0, self.max_targets)
            
            # List to store target information for this sample
            sample_targets = []
            
            # If we have targets, add them to the signal
            if num_targets > 0:
                for _ in range(num_targets):
                    # Generate random target parameters
                    distance = random.uniform(5, self.max_range)
                    velocity = random.uniform(-self.max_velocity, self.max_velocity)
                    rcs = random.uniform(0.1, 10.0)  # Radar Cross Section
                    
                    # Add target to received signal
                    rx_signal = self._add_target(rx_signal, distance, velocity, rcs)
                    
                    # Store target information
                    target = {
                        'distance': distance,
                        'velocity': velocity,
                        'rcs': rcs
                    }
                    sample_targets.append(target)
            else:
                # If no targets, just apply channel model to TX signal
                # This simulates the radar transmitting but not receiving any reflections
                for rx_idx in range(self.num_rx):
                    for chirp_idx in range(self.num_chirps):
                        # Apply channel effects (like noise, interference, etc.)
                        if self.apply_realistic_effects:
                            # Add minimal noise to represent system noise
                            noise_power = 1e-6  # Very low noise power
                            noise_real = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                            noise_imag = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                            rx_signal[rx_idx, chirp_idx] = noise_real + 1j * noise_imag
                            
                            # Add minimal crosstalk from TX to RX
                            isolation_db = 30  # Typical isolation between TX and RX in dB
                            attenuation = 10**(-isolation_db/20)
                            delay_samples = 5  # Small fixed delay
                            
                            if delay_samples < self.samples_per_chirp:
                                rx_signal[rx_idx, chirp_idx, delay_samples:] += attenuation * tx_signal[chirp_idx, :self.samples_per_chirp-delay_samples]
                        else:
                            # Just add minimal noise
                            noise_power = 1e-9  # Extremely low noise power
                            noise_real = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                            noise_imag = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                            rx_signal[rx_idx, chirp_idx] = noise_real + 1j * noise_imag
            
            # Generate random SNR for this sample
            snr_db = random.uniform(self.snr_min, self.snr_max)
            
            # Add noise to the received signal
            rx_signal = self._add_noise(rx_signal, snr_db)
            self._visualize_rx_signal(rx_signal, target_info=sample_targets, title=f"RX Signal {i}", save_path=f"{self.save_path}/rx_signal_{i}{IMG_FORMAT}")
            
            # Process the received signal to generate range-Doppler map
            #rd_map = self.radar_processor.generate_range_doppler_map(rx_signal)
            rd_map = self.radar_processor.time_to_range_doppler(rx_signal)

            # Create target mask
            target_mask = self._create_target_mask(sample_targets)
            
            # Store data (100, 4, 32, 150, 2) (4, 32, 176)
            self.time_domain_data[i, :, :, :, 0] = np.real(rx_signal)
            self.time_domain_data[i, :, :, :, 1] = np.imag(rx_signal)
            self.range_doppler_maps[i, 0] = np.real(rd_map)
            self.range_doppler_maps[i, 1] = np.imag(rd_map)
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
            if self.drawfig:
                self._draw_sample(i, rx_signal, rd_map, sample_targets)
                
            # Visualize the first sample and a few random samples
            if i == 0 or (i % 20 == 0 and i > 0):
                self._visualize_comprehensive_signal(
                    tx_signal=tx_signal,
                    rx_signal=rx_signal,
                    title=f"Sample {i}: {len(sample_targets)} Targets, {self.signal_type}",
                    target_info=sample_targets
                )
        
        # Save the dataset
        if save_data:
            if format.lower() == 'hdf5':
                self._save_hdf5()
            elif format.lower() == 'numpy':
                self._save_numpy()
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        return self.time_domain_data, self.range_doppler_maps, self.target_masks, self.target_info
    
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

    

    def _add_target(self, rx_signal, distance, velocity, rcs=1.0):
        """Add target to received signal
        
        Args:
            rx_signal: Received signal array [num_rx, num_chirps, samples_per_chirp]
            distance: Target distance in meters
            velocity: Target velocity in m/s
            rcs: Target radar cross section (normalized)
            
        Returns:
            Updated received signal array
        """
        # Generate the transmit signal first
        tx_signal = self._generate_tx_signal()
        
        # For each RX antenna and chirp, calculate the received signal
        for rx_idx in range(self.num_rx):
            for chirp_idx in range(self.num_chirps):
                # Calculate time delay for this chirp based on distance and velocity
                # For moving targets, distance changes with each chirp
                current_distance = distance + velocity * chirp_idx * self.chirp_duration
                
                # Calculate time delay (round trip)
                time_delay = 2 * current_distance / 3e8  # Speed of light = 3e8 m/s
                
                # Calculate Doppler shift
                doppler_shift = 2 * velocity * self.center_freq / 3e8  # Hz
                
                # Calculate phase shift due to Doppler
                doppler_phase = 2 * np.pi * doppler_shift * chirp_idx * self.chirp_duration
                
                # Apply delay, attenuation, and phase shift to the transmit signal
                delayed_signal = self._apply_delay_to_tx_signal(
                    tx_signal, time_delay, doppler_phase, chirp_idx, rcs
                )
                
                # Add the delayed signal to the received signal
                rx_signal[rx_idx, chirp_idx] += delayed_signal
                
        return rx_signal
    
    def _generate_ofdm_signal(self, num_chirps, samples_per_chirp, center_freq, bandwidth, 
                             num_subcarriers, subcarrier_spacing=None, t=None):
        """Generate OFDM signal with configurable parameters
        
        Args:
            num_chirps: Number of chirps to generate
            samples_per_chirp: Number of samples per chirp
            center_freq: Center frequency of the OFDM signal in Hz
            bandwidth: Total bandwidth of the OFDM signal in Hz
            num_subcarriers: Number of subcarriers to use
            subcarrier_spacing: Spacing between subcarriers in Hz (if None, calculated from bandwidth)
            t: Time vector (if None, created based on samples_per_chirp)
            
        Returns:
            OFDM signal array [num_chirps, samples_per_chirp]
        """
        # Create time vector if not provided
        if t is None:
            t = np.arange(samples_per_chirp) / self.sample_rate
        
        # Calculate subcarrier spacing if not provided
        if subcarrier_spacing is None:
            subcarrier_spacing = bandwidth / num_subcarriers
        
        # Initialize OFDM signal array
        ofdm_signal = np.zeros((num_chirps, samples_per_chirp), dtype=np.complex64)
        
        # Generate OFDM signal for each chirp
        for chirp_idx in range(num_chirps):
            # Initialize chirp signal
            chirp_signal = np.zeros(samples_per_chirp, dtype=np.complex64)
            
            # Generate subcarriers
            for i in range(num_subcarriers):
                # Random phase for each subcarrier
                phase_offset = np.random.uniform(0, 2*np.pi)
                
                # Subcarrier frequency centered around center_freq
                f_sc = center_freq - bandwidth/2 + i * subcarrier_spacing
                
                # Generate subcarrier signal
                subcarrier = np.exp(1j * (2 * np.pi * f_sc * t + phase_offset))
                
                # Add to total signal
                chirp_signal += subcarrier
            
            # Normalize
            chirp_signal /= np.sqrt(num_subcarriers)
            ofdm_signal[chirp_idx] = chirp_signal
        
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

    def _visualize_comprehensive_signal(self, tx_signal, rx_signal, title="Signal Visualization", target_info=None, save_path=None):
        """Comprehensive visualization of TX and RX signals with hardware effects
        
        Args:
            tx_signal: Transmit signal array [num_chirps, samples_per_chirp]
            rx_signal: Received signal array [num_rx, num_chirps, samples_per_chirp]
            title: Title for the plot
            target_info: Optional list of dictionaries with target information
            save_path: Path to save the figure (if None, just display)
        """
        # Create figure with subplots (3 rows, 4 columns)
        fig, axs = plt.subplots(3, 4, figsize=(20, 12))
        
        # Select first chirp and RX for visualization
        chirp_idx = 0
        rx_idx = 0
        
        # Row 1: TX Signal Analysis
        # --------------------------
        
        # Plot TX time domain signal (real part)
        axs[0, 0].plot(np.real(tx_signal[chirp_idx]))
        axs[0, 0].set_title('TX Signal (Real Part)')
        axs[0, 0].set_xlabel('Sample')
        axs[0, 0].set_ylabel('Amplitude')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot TX time domain signal (imaginary part)
        axs[0, 1].plot(np.imag(tx_signal[chirp_idx]))
        axs[0, 1].set_title('TX Signal (Imaginary Part)')
        axs[0, 1].set_xlabel('Sample')
        axs[0, 1].set_ylabel('Amplitude')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot TX frequency domain
        freq_tx = np.fft.fftshift(np.fft.fftfreq(len(tx_signal[chirp_idx]), 1/self.sample_rate))
        freq_tx_ghz = (freq_tx + self.center_freq) / 1e9  # Convert to GHz
        
        fft_tx = np.fft.fftshift(np.fft.fft(tx_signal[chirp_idx]))
        fft_tx_mag = np.abs(fft_tx)
        fft_tx_db = 20 * np.log10(fft_tx_mag + 1e-10)
        
        axs[0, 2].plot(freq_tx_ghz, fft_tx_db)
        axs[0, 2].set_title('TX Frequency Domain (Magnitude)')
        axs[0, 2].set_xlabel('Frequency (GHz)')
        axs[0, 2].set_ylabel('Magnitude (dB)')
        axs[0, 2].grid(True, alpha=0.3)
        
        # Mark the bandwidth region
        bw_start = self.center_freq - self.bandwidth/2
        bw_end = self.center_freq + self.bandwidth/2
        axs[0, 2].axvspan(bw_start/1e9, bw_end/1e9, alpha=0.2, color='green', 
                        label=f'Signal Bandwidth ({self.bandwidth/1e6:.0f} MHz)')
        axs[0, 2].legend()
        
        # Set x-axis limits to focus on the relevant frequency range
        axs[0, 2].set_xlim([self.center_freq/1e9 - 1, self.center_freq/1e9 + 1])
        
        # Plot TX phase
        phase_tx = np.angle(tx_signal[chirp_idx])
        axs[0, 3].plot(phase_tx)
        axs[0, 3].set_title('TX Signal Phase')
        axs[0, 3].set_xlabel('Sample')
        axs[0, 3].set_ylabel('Phase (radians)')
        axs[0, 3].grid(True, alpha=0.3)
        
        # Row 2: Hardware Modulation Effects (if applicable)
        # -------------------------------------------------
        
        if self.signal_type in ['OFDM_FMCW', 'Sine_FMCW']:
            # For these signal types, we have hardware modulation
            # Generate baseband signal at AD9361 frequency
            t = np.arange(0, self.chirp_duration, 1/self.sample_rate)
            baseband_signal = np.zeros((self.num_chirps, self.samples_per_chirp), dtype=np.complex64)
            
            if self.signal_type == 'OFDM_FMCW':
                # Generate OFDM baseband signal
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
                for c_idx in range(self.num_chirps):
                    sine_phase = 2 * np.pi * self.signal_freq * t
                    baseband_signal[c_idx] = np.exp(1j * sine_phase)
            
            # Plot baseband time domain (real part)
            axs[1, 0].plot(np.real(baseband_signal[chirp_idx]))
            axs[1, 0].set_title('Baseband Signal (Real Part)')
            axs[1, 0].set_xlabel('Sample')
            axs[1, 0].set_ylabel('Amplitude')
            axs[1, 0].grid(True, alpha=0.3)
            
            # Plot baseband frequency domain
            freq_bb = np.fft.fftshift(np.fft.fftfreq(len(baseband_signal[chirp_idx]), 1/self.sample_rate))
            freq_bb_ghz = (freq_bb + self.transceiver_center_freq) / 1e9  # Convert to GHz
            
            fft_bb = np.fft.fftshift(np.fft.fft(baseband_signal[chirp_idx]))
            fft_bb_mag = np.abs(fft_bb)
            fft_bb_db = 20 * np.log10(fft_bb_mag + 1e-10)
            
            axs[1, 1].plot(freq_bb_ghz, fft_bb_db)
            axs[1, 1].set_title('Baseband Frequency Domain')
            axs[1, 1].set_xlabel('Frequency (GHz)')
            axs[1, 1].set_ylabel('Magnitude (dB)')
            axs[1, 1].grid(True, alpha=0.3)
            
            # Mark the transceiver bandwidth region
            bb_bw_start = self.transceiver_center_freq - self.transceiver_bandwidth/2
            bb_bw_end = self.transceiver_center_freq + self.transceiver_bandwidth/2
            axs[1, 1].axvspan(bb_bw_start/1e9, bb_bw_end/1e9, alpha=0.2, color='blue', 
                            label=f'Transceiver BW ({self.transceiver_bandwidth/1e6:.0f} MHz)')
            axs[1, 1].legend()
            
            # Set x-axis limits to focus on the relevant frequency range
            axs[1, 1].set_xlim([self.transceiver_center_freq/1e9 - 0.5, self.transceiver_center_freq/1e9 + 0.5])
            
            # Plot hardware modulation effect (time domain)
            axs[1, 2].plot(np.real(tx_signal[chirp_idx]))
            axs[1, 2].set_title('After Hardware Modulation (Real)')
            axs[1, 2].set_xlabel('Sample')
            axs[1, 2].set_ylabel('Amplitude')
            axs[1, 2].grid(True, alpha=0.3)
            
            # Plot hardware modulation effect (frequency domain)
            axs[1, 3].plot(freq_tx_ghz, fft_tx_db)
            axs[1, 3].set_title('After Hardware Modulation (Freq)')
            axs[1, 3].set_xlabel('Frequency (GHz)')
            axs[1, 3].set_ylabel('Magnitude (dB)')
            axs[1, 3].grid(True, alpha=0.3)
            axs[1, 3].axvspan(bw_start/1e9, bw_end/1e9, alpha=0.2, color='green', 
                            label=f'Output BW ({self.bandwidth/1e6:.0f} MHz)')
            axs[1, 3].legend()
            axs[1, 3].set_xlim([self.center_freq/1e9 - 1, self.center_freq/1e9 + 1])
        else:
            # For direct signal types, show spectrogram and other visualizations
            # Plot spectrogram to visualize frequency change over time (for FMCW)
            try:
                from scipy import signal as sig
                f, t, Sxx = sig.spectrogram(tx_signal[chirp_idx], 
                                          fs=self.sample_rate,
                                          nperseg=128,
                                          noverlap=64,
                                          scaling='spectrum')
                
                # Convert to dB
                Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
                
                # Plot spectrogram
                im = axs[1, 0].pcolormesh(t, f/1e6, Sxx_db, shading='gouraud', cmap='viridis')
                axs[1, 0].set_title('TX Spectrogram (Frequency vs Time)')
                axs[1, 0].set_xlabel('Time (s)')
                axs[1, 0].set_ylabel('Frequency (MHz)')
                plt.colorbar(im, ax=axs[1, 0], label='Power (dB)')
            except Exception as e:
                axs[1, 0].text(0.5, 0.5, f"Spectrogram calculation failed:\n{str(e)}", 
                              ha='center', va='center', transform=axs[1, 0].transAxes)
                axs[1, 0].set_title('TX Spectrogram (Error)')
            
            # Plot multiple chirps to see consistency
            if tx_signal.shape[0] > 1:
                num_chirps_to_plot = min(5, tx_signal.shape[0])
                for i in range(num_chirps_to_plot):
                    axs[1, 1].plot(np.real(tx_signal[i]), 
                                  label=f'Chirp {i}', 
                                  alpha=0.7)
                
                axs[1, 1].set_title('Multiple TX Chirps (Real Part)')
                axs[1, 1].set_xlabel('Sample')
                axs[1, 1].set_ylabel('Amplitude')
                axs[1, 1].legend()
                axs[1, 1].grid(True, alpha=0.3)
            
            # Plot instantaneous frequency (for FMCW)
            if self.signal_type == 'FMCW':
                # Calculate instantaneous frequency
                inst_phase = np.unwrap(np.angle(tx_signal[chirp_idx]))
                inst_freq = np.diff(inst_phase) * self.sample_rate / (2 * np.pi)
                
                axs[1, 2].plot(inst_freq)
                axs[1, 2].set_title('TX Instantaneous Frequency')
                axs[1, 2].set_xlabel('Sample')
                axs[1, 2].set_ylabel('Frequency (Hz)')
                axs[1, 2].grid(True, alpha=0.3)
            else:
                # For non-FMCW, show signal envelope
                envelope = np.abs(tx_signal[chirp_idx])
                axs[1, 2].plot(envelope)
                axs[1, 2].set_title('TX Signal Envelope')
                axs[1, 2].set_xlabel('Sample')
                axs[1, 2].set_ylabel('Amplitude')
                axs[1, 2].grid(True, alpha=0.3)
            
            # Empty plot for consistency
            axs[1, 3].set_visible(False)
        
        # Row 3: RX Signal Analysis
        # --------------------------
        
        # Plot RX time domain signal (real part)
        axs[2, 0].plot(np.real(rx_signal[rx_idx, chirp_idx]))
        axs[2, 0].set_title('RX Signal (Real Part)')
        axs[2, 0].set_xlabel('Sample')
        axs[2, 0].set_ylabel('Amplitude')
        axs[2, 0].grid(True, alpha=0.3)
        
        # Plot RX time domain signal (imaginary part)
        axs[2, 1].plot(np.imag(rx_signal[rx_idx, chirp_idx]))
        axs[2, 1].set_title('RX Signal (Imaginary Part)')
        axs[2, 1].set_xlabel('Sample')
        axs[2, 1].set_ylabel('Amplitude')
        axs[2, 1].grid(True, alpha=0.3)
        
        # Plot RX frequency domain
        freq_rx = np.fft.fftshift(np.fft.fftfreq(len(rx_signal[rx_idx, chirp_idx]), 1/self.sample_rate))
        freq_rx_ghz = (freq_rx + self.center_freq) / 1e9  # Convert to GHz
        
        fft_rx = np.fft.fftshift(np.fft.fft(rx_signal[rx_idx, chirp_idx]))
        fft_rx_mag = np.abs(fft_rx)
        fft_rx_db = 20 * np.log10(fft_rx_mag + 1e-10)
        
        axs[2, 2].plot(freq_rx_ghz, fft_rx_db)
        axs[2, 2].set_title('RX Frequency Domain (Magnitude)')
        axs[2, 2].set_xlabel('Frequency (GHz)')
        axs[2, 2].set_ylabel('Magnitude (dB)')
        axs[2, 2].grid(True, alpha=0.3)
        
        # Mark the bandwidth region
        axs[2, 2].axvspan(bw_start/1e9, bw_end/1e9, alpha=0.2, color='green', 
                        label=f'Signal Bandwidth ({self.bandwidth/1e6:.0f} MHz)')
        axs[2, 2].legend()
        
        # Set x-axis limits to focus on the relevant frequency range
        axs[2, 2].set_xlim([self.center_freq/1e9 - 1, self.center_freq/1e9 + 1])
        
        # Plot multiple RX antennas comparison
        if rx_signal.shape[0] > 1:
            num_rx_to_plot = min(4, rx_signal.shape[0])
            for i in range(num_rx_to_plot):
                axs[2, 3].plot(np.real(rx_signal[i, chirp_idx]), 
                              label=f'RX {i}', 
                              alpha=0.7)
            
            axs[2, 3].set_title('Multiple RX Antennas (Real Part)')
            axs[2, 3].set_xlabel('Sample')
            axs[2, 3].set_ylabel('Amplitude')
            axs[2, 3].legend()
            axs[2, 3].grid(True, alpha=0.3)
        
        # Add target information if provided
        if target_info:
            target_text = "Target Information:\n"
            for i, target in enumerate(target_info):
                target_text += f"Target {i+1}: Distance={target['distance']:.1f}m, "
                target_text += f"Velocity={target['velocity']:.1f}m/s, RCS={target['rcs']:.2f}\n"
            
            fig.text(0.5, 0.01, target_text, ha='center', 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Add signal parameters
        params_text = (f"Signal Type: {self.signal_type}, Bandwidth: {self.bandwidth/1e6:.0f} MHz\n"
                      f"Center Freq: {self.center_freq/1e9:.1f} GHz, Sample Rate: {self.sample_rate/1e6:.1f} MHz\n"
                      f"Chirp Duration: {self.chirp_duration*1e6:.1f} μs, SNR: {self.snr_values[-1]:.1f} dB")
        
        fig.text(0.5, 0.97, params_text, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add overall title
        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or show the figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
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
    """Visualize the signal processing steps for a single sample"""
    # Create directory for processing figures
    os.makedirs(f"{dataset.save_path}/processing", exist_ok=True)

    # Initialize metrics lists
    detection_accuracies = []
    metrics = {
        'snr_improvement': [],
        'detection_accuracy': [],
        'processing_time': []
    }
    processing_times = []
    
    # Get sample data
    sample = dataset[sample_idx]
    time_domain = sample['time_domain'].numpy()
    range_doppler = sample['range_doppler'].numpy()
    target_info = sample['target_info']
    
    # Convert time domain data back to complex
    # Shape: [num_rx, num_chirps, samples_per_chirp]
    complex_data = time_domain[:, :, :, 0] + 1j * time_domain[:, :, :, 1]
    
    # Get TX signal if available
    if hasattr(dataset, 'tx_signals') and dataset.tx_signals and sample_idx < len(dataset.tx_signals):
        tx_signal = dataset.tx_signals[sample_idx]
    else:
        # Generate a TX signal if not available
        tx_signal = dataset._generate_tx_signal()
    
    # Create figure with multiple subplots for processing steps
    fig, axs = plt.subplots(5, 2, figsize=(15, 25))
    
    # 1. Plot TX signal (real and imaginary parts)
    chirp_idx = 0  # First chirp
    
    # Define a subsection of the signal to visualize more clearly
    start_sample = 0
    num_samples_to_show = min(1000, dataset.samples_per_chirp)
    
    # Create sample indices for the subsection
    sample_indices = np.arange(start_sample, start_sample + num_samples_to_show)
    
    # Plot real part of TX signal
    axs[0, 0].plot(sample_indices, np.real(tx_signal[chirp_idx][start_sample:start_sample + num_samples_to_show]))
    axs[0, 0].set_title(f'TX Signal (Real Part) - Chirp {chirp_idx}')
    axs[0, 0].set_xlabel('Sample')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot imaginary part of TX signal
    axs[0, 1].plot(sample_indices, np.imag(tx_signal[chirp_idx][start_sample:start_sample + num_samples_to_show]))
    axs[0, 1].set_title(f'TX Signal (Imaginary Part) - Chirp {chirp_idx}')
    axs[0, 1].set_xlabel('Sample')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Add a note about the subsection being shown
    axs[0, 0].text(0.5, 0.02, f'Showing samples {start_sample}-{start_sample + num_samples_to_show} of {dataset.samples_per_chirp}',
                  transform=axs[0, 0].transAxes, fontsize=8, ha='center', 
                  bbox=dict(facecolor='white', alpha=0.7))
    
    # 2. Plot TX signal frequency domain
    freq_tx = np.fft.fftshift(np.fft.fftfreq(len(tx_signal[chirp_idx]), 1/dataset.sample_rate))
    freq_tx_ghz = (freq_tx + dataset.center_freq) / 1e9  # Convert to GHz
    
    fft_tx = np.fft.fftshift(np.fft.fft(tx_signal[chirp_idx]))
    fft_tx_mag = np.abs(fft_tx)
    fft_tx_db = 20 * np.log10(fft_tx_mag + 1e-10)
    
    axs[1, 0].plot(freq_tx_ghz, fft_tx_db)
    axs[1, 0].set_title('TX Frequency Domain (Magnitude)')
    axs[1, 0].set_xlabel('Frequency (GHz)')
    axs[1, 0].set_ylabel('Magnitude (dB)')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Mark the bandwidth region
    bw_start = dataset.center_freq - dataset.bandwidth/2
    bw_end = dataset.center_freq + dataset.bandwidth/2
    axs[1, 0].axvspan(bw_start/1e9, bw_end/1e9, alpha=0.2, color='green', 
                    label=f'Signal Bandwidth ({dataset.bandwidth/1e6:.0f} MHz)')
    axs[1, 0].legend()
    
    # Set x-axis limits to focus on the relevant frequency range
    axs[1, 0].set_xlim([dataset.center_freq/1e9 - 1, dataset.center_freq/1e9 + 1])
    
    # 3. Plot original received time domain signal (real and imaginary parts)
    rx_idx, chirp_idx = 0, 0  # First RX, first chirp
    
    # Plot real part of the subsection
    axs[1, 1].plot(sample_indices, np.real(complex_data[rx_idx, chirp_idx][start_sample:start_sample + num_samples_to_show]))
    axs[1, 1].set_title(f'Original Received Signal (Real Part) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[1, 1].set_xlabel('Sample')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].grid(True, alpha=0.3)
    
    # 4. Simulate hardware demodulation (CN0566 to AD9361)
    # Use the radar processor to perform the hardware demodulation
    demodulated_data = dataset.radar_processor.simulate_hardware_demodulation(complex_data)
    
    # Plot demodulated signal
    axs[2, 0].plot(np.real(demodulated_data[rx_idx, chirp_idx]))
    axs[2, 0].set_title(f'Demodulated Signal (Real Part) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[2, 0].set_xlabel('Sample')
    axs[2, 0].set_ylabel('Amplitude')
    axs[2, 0].grid(True, alpha=0.3)
    
    axs[2, 1].plot(np.imag(demodulated_data[rx_idx, chirp_idx]))
    axs[2, 1].set_title(f'Demodulated Signal (Imaginary Part) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[2, 1].set_xlabel('Sample')
    axs[2, 1].set_ylabel('Amplitude')
    axs[2, 1].grid(True, alpha=0.3)
    
    # 5. Range processing on demodulated data
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
    
    axs[3, 0].plot(range_profile_db)
    axs[3, 0].set_title(f'Range Profile (dB) - RX {rx_idx}, Chirp {chirp_idx}')
    axs[3, 0].set_xlabel('Range Bin')
    axs[3, 0].set_ylabel('Magnitude (dB)')
    axs[3, 0].grid(True, alpha=0.3)
    
    # Add secondary x-axis for range in meters
    ax_range = axs[3, 0].secondary_xaxis('top', functions=(
        lambda x: x * dataset.range_resolution,  # bin to meters
        lambda x: x / dataset.range_resolution   # meters to bin
    ))
    ax_range.set_xlabel('Range (m)')
    
    # 6. Doppler processing on range profiles
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
    
    axs[3, 1].plot(doppler_profile_db)
    axs[3, 1].set_title(f'Doppler Profile (dB) - RX {rx_idx}, Range Bin {max_range_bin}')
    axs[3, 1].set_xlabel('Doppler Bin')
    axs[3, 1].set_ylabel('Magnitude (dB)')
    axs[3, 1].grid(True, alpha=0.3)
    
    # Add secondary x-axis for velocity in m/s
    ax_velocity = axs[3, 1].secondary_xaxis('top', functions=(
        lambda y: (y - dataset.num_doppler_bins/2) * dataset.velocity_resolution,  # bin to m/s
        lambda y: y / dataset.velocity_resolution + dataset.num_doppler_bins/2     # m/s to bin
    ))
    ax_velocity.set_xlabel('Velocity (m/s)')
    
    # 7. Complete Range-Doppler processing using the radar processor
    # This uses the full two-step process implemented in AIradar_processing.py
    rd_map = dataset.radar_processor.time_to_range_doppler(complex_data)
    
    # Plot the final range-Doppler map
    rd_magnitude = np.abs(rd_map)
    
    # Apply dynamic range compression to enhance visibility
    rd_db = 20 * np.log10(rd_magnitude + 1e-10)  # Convert to dB
    
    # Apply normalization and thresholding for better visualization
    rd_db_norm = rd_db - np.min(rd_db)  # Normalize to start from 0
    dynamic_range = 60  # Set dynamic range in dB
    rd_db_norm = np.clip(rd_db_norm, 0, dynamic_range)  # Clip to dynamic range
    rd_db_norm = rd_db_norm / dynamic_range  # Scale to [0,1]
    
    # Use a better colormap with higher contrast
    im = axs[4, 0].imshow(rd_db_norm, aspect='auto', cmap='viridis', origin='lower',
                         extent=[0, dataset.num_range_bins, 0, dataset.num_doppler_bins])
    axs[4, 0].set_title('Range-Doppler Map (Full Processing)')
    axs[4, 0].set_xlabel('Range Bin')
    axs[4, 0].set_ylabel('Doppler Bin')
    
    # Add secondary axes for physical units
    ax_range = axs[4, 0].secondary_xaxis('top', functions=(
        lambda x: x * dataset.range_resolution,  # bin to meters
        lambda x: x / dataset.range_resolution   # meters to bin
    ))
    ax_range.set_xlabel('Range (m)')
    
    ax_velocity = axs[4, 0].secondary_yaxis('right', functions=(
        lambda y: (y - dataset.num_doppler_bins/2) * dataset.velocity_resolution,  # bin to m/s
        lambda y: y / dataset.velocity_resolution + dataset.num_doppler_bins/2     # m/s to bin
    ))
    ax_velocity.set_xlabel('Velocity (m/s)')
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=axs[4, 0])
    cbar.set_label('Normalized Magnitude (dB)')
    
    # Generate and plot target mask for visualization
    target_mask = np.zeros((dataset.num_doppler_bins, dataset.num_range_bins, 1), dtype=np.float32)
    
        # Initialize metrics dictionary for this function
    metrics = {
        'snr_improvement': [],
        'detection_accuracy': []
    }
    
    # Calculate noise floor
    rd_magnitude = np.abs(rd_map)
    noise_floor = np.median(rd_magnitude)

    detected = 0
    # Create mask based on target information
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
            
            # If the peak is significantly above the noise floor, count as detected
            if np.max(region) > 3 * noise_floor:
                detected += 1
            
        detection_accuracies.append(detected / len(target_info) if target_info else 1.0)
    
    # Average the metrics
    # Calculate detection accuracy
    detection_accuracy = detected / len(target_info) if target_info else 1.0
    metrics['detection_accuracy'].append(detection_accuracy)
    
    # Calculate SNR improvement
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
        snr_improvement = 20 * np.log10(avg_peak / noise_floor)
        metrics['snr_improvement'].append(snr_improvement)
    else:
        metrics['snr_improvement'].append(0)
    # metrics['snr_improvement'].append(np.mean(snr_improvements) if snr_improvements else 0)
    metrics['processing_time'].append(np.mean(processing_times))
    #metrics['detection_accuracy'].append(np.mean(detection_accuracies))

    # Visualize samples for each signal type
    for i in range(min(visualize_samples, num_samples)):
        # Use the comprehensive visualization if available
        if hasattr(dataset, '_visualize_comprehensive_signal'):
            # Get sample data
            sample = dataset[i]
            time_domain = sample['time_domain'].numpy()
            target_info = sample['target_info']
            
            # Convert time domain data back to complex
            complex_data = time_domain[:, :, :, 0] + 1j * time_domain[:, :, :, 1]
            
            # Get TX signal if available
            if hasattr(dataset, 'tx_signals') and dataset.tx_signals and i < len(dataset.tx_signals):
                tx_signal = dataset.tx_signals[i]
            else:
                # Generate a TX signal if not available
                tx_signal = dataset._generate_tx_signal()
            
            # Use comprehensive visualization
            dataset._visualize_comprehensive_signal(
                tx_signal=tx_signal,
                rx_signal=complex_data,
                title=f"{signal_type} Sample {i}: {len(target_info)} Targets",
                target_info=target_info,
                save_path=f"{comparison_dir}/{signal_type}_sample_{i}_comprehensive{IMG_FORMAT}"
            )
        
        # Also use the standard visualization
        dataset.visualize_sample(i)
        plt.savefig(f"{comparison_dir}/{signal_type}_sample_{i}_visualization{IMG_FORMAT}")
        plt.close()
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot range resolution comparison
    plt.subplot(2, 2, 1)
    plt.bar(signal_types, metrics['range_resolution'])
    plt.title('Range Resolution Comparison')
    plt.xlabel('Signal Type')
    plt.ylabel('Range Resolution (m)')
    plt.grid(True, alpha=0.3)
    
    # Plot velocity resolution comparison
    plt.subplot(2, 2, 2)
    plt.bar(signal_types, metrics['velocity_resolution'])
    plt.title('Velocity Resolution Comparison')
    plt.xlabel('Signal Type')
    plt.ylabel('Velocity Resolution (m/s)')
    plt.grid(True, alpha=0.3)
    
    # Plot SNR improvement comparison
    plt.subplot(2, 2, 3)
    plt.bar(signal_types, metrics['snr_improvement'])
    plt.title('SNR Improvement Comparison')
    plt.xlabel('Signal Type')
    plt.ylabel('SNR Improvement (dB)')
    plt.grid(True, alpha=0.3)
    
    # Plot processing time comparison
    plt.subplot(2, 2, 4)
    plt.bar(signal_types, metrics['processing_time'])
    plt.title('Processing Time Comparison')
    plt.xlabel('Signal Type')
    plt.ylabel('Processing Time (s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/signal_type_comparison{IMG_FORMAT}")
    plt.close()
    
    # Create detection accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(signal_types, metrics['detection_accuracy'])
    plt.title('Target Detection Accuracy Comparison')
    plt.xlabel('Signal Type')
    plt.ylabel('Detection Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/detection_accuracy_comparison{IMG_FORMAT}")
    plt.close()
    
    print(f"Signal type comparison completed and saved to {comparison_dir}/")
    
    return datasets, metrics

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