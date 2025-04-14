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
                 precision='float32',     # Data precision: 'float32' or 'float16'
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
            signal_type: Type of radar signal ('FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW')
            signal_freq: Signal frequency for FMCW modulation (Hz)
            use_lazy_loading: Whether to use lazy loading for HDF5 files
            use_memory_mapping: Whether to use memory mapping for NumPy files
            cache_size: Number of samples to cache when using lazy loading
            precision: Data precision to use ('float32' or 'float16')
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
        # Validate signal type
        valid_signal_types = ['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW']
        if signal_type not in valid_signal_types:
            print(f"Warning: Invalid signal type '{signal_type}'. Using 'FMCW' instead.")
            self.signal_type = 'FMCW'
        else:
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

    def simulate_cn0566_demodulation(self, time_domain_data, target_info):
        """
        Simulate the CN0566 phaser hardware demodulation of 500MHz FMCW sweep to AD9361 domain
        
        Args:
            time_domain_data: Time domain data with shape [num_rx, num_chirps, samples_per_chirp, 2]
                             where the last dimension contains I/Q data
            target_info: List of dictionaries containing target information
            
        Returns:
            Demodulated time domain data with shape [num_rx, num_chirps, samples_per_chirp, 2]
        """
        # Create a copy of the input data to avoid modifying the original
        demodulated_data = np.zeros_like(time_domain_data)
        
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp, _ = time_domain_data.shape
        
        # Time vector for each chirp (in seconds)
        t = np.linspace(0, self.chirp_duration, samples_per_chirp)
        
        # FMCW parameters
        slope = self.bandwidth / self.chirp_duration  # Hz/s
        
        # Convert I/Q data to complex
        complex_data = time_domain_data[..., 0] + 1j * time_domain_data[..., 1]
        
        # Apply CN0566 demodulation (hardware processing)
        # This simulates the mixing of the received signal with the FMCW sweep
        for rx in range(num_rx):
            for chirp in range(num_chirps):
                # Apply demodulation (complex multiplication with FMCW reference)
                # This is what happens in the CN0566 hardware
                fmcw_ref = np.exp(-1j * np.pi * slope * t * t)  # FMCW reference signal
                demod_signal = complex_data[rx, chirp] * fmcw_ref
                
                # Apply anti-aliasing filter to simulate AD9361 bandwidth limitation (56 MHz)
                if self.sample_rate > 56e6:
                    # Simple low-pass filter simulation
                    demod_fft = np.fft.fftshift(np.fft.fft(demod_signal))
                    freqs = np.fft.fftshift(np.fft.fftfreq(samples_per_chirp, 1/self.sample_rate))
                    filter_mask = np.abs(freqs) < 28e6  # 56 MHz bandwidth (±28 MHz)
                    demod_fft *= filter_mask
                    demod_signal = np.fft.ifft(np.fft.ifftshift(demod_fft))
                
                # Store I/Q components
                demodulated_data[rx, chirp, :, 0] = np.real(demod_signal)
                demodulated_data[rx, chirp, :, 1] = np.imag(demod_signal)
        
        return demodulated_data

    def _generate_time_domain_data(self):
        """Generate time domain data for all samples"""
        # Initialize time domain data array
        self.time_domain_data = np.zeros(
            (self.num_samples, self.num_rx, self.num_chirps, self.samples_per_chirp, 2), 
            dtype=np.float32
        )
        
        # Generate data for each sample
        for i in tqdm(range(self.num_samples), desc="Generating time domain data", unit="samples"):
            # Determine number of targets for this sample
            num_targets = random.randint(1, self.max_targets)
            
            # Generate random targets
            targets = []
            for _ in range(num_targets):
                # Random target distance (in meters)
                distance = random.uniform(1.0, self.max_range * 0.9)
                
                # Random target velocity (in m/s)
                max_velocity = self.wavelength / (4 * self.chirp_duration)
                velocity = random.uniform(-max_velocity * 0.8, max_velocity * 0.8)
                
                # Random target RCS and SNR
                snr_db = random.uniform(self.snr_min, self.snr_max)
                rcs = 10**(snr_db/10)
                
                # Calculate range and Doppler bin
                range_bin = int(distance / self.range_resolution)
                if range_bin >= self.num_range_bins:
                    range_bin = self.num_range_bins - 1
                
                doppler_bin = int((velocity / self.max_velocity + 0.5) * self.num_doppler_bins)
                if doppler_bin >= self.num_doppler_bins:
                    doppler_bin = self.num_doppler_bins - 1
                elif doppler_bin < 0:
                    doppler_bin = 0
                
                # Store target information
                target = {
                    'range': distance,
                    'velocity': velocity,
                    'rcs': rcs,
                    'snr': snr_db,
                    'range_bin': range_bin,
                    'doppler_bin': doppler_bin
                }
                targets.append(target)
            
            # Store target information for this sample
            if not hasattr(self, 'target_info') or self.target_info is None:
                self.target_info = []
            
            # Ensure target_info has enough elements
            while len(self.target_info) <= i:
                self.target_info.append([])
            
            self.target_info[i] = targets
            
            # Generate time domain signal based on signal type
            if self.signal_type.upper() == 'OFDM_FMCW':
                # Use the specialized method for OFDM_FMCW
                rx_signal = self._generate_ofdm_fmcw_signal(targets)
                
                # Fix: rx_signal is already in I/Q format with shape [num_rx, num_chirps, samples_per_chirp, 2]
                # Directly assign to time_domain_data
                self.time_domain_data[i] = rx_signal
                
            elif self.signal_type.upper() == 'SINE_FMCW':
                # Use the specialized method for Sine_FMCW
                rx_signal = self._generate_sine_fmcw_signal(targets)
                
                # Fix: rx_signal is already in I/Q format with shape [num_rx, num_chirps, samples_per_chirp, 2]
                # Directly assign to time_domain_data
                self.time_domain_data[i] = rx_signal
                
            else:
                # For other signal types, use the standard approach
                # Generate chirp signal
                tx_chirp = self._generate_chirp()
                
                # Initialize received signal with noise
                rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), dtype=np.complex64)
                for rx in range(self.num_rx):
                    for chirp in range(self.num_chirps):
                        noise_power = 1.0
                        noise_real = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), self.samples_per_chirp)
                        rx_signal[rx, chirp, :] = noise_real + 1j * noise_imag
                
                # Add targets to the signal
                for target in targets:
                    rx_signal = self._add_target(rx_signal, target['range'], target['velocity'], target['rcs'])
                
                # Convert complex signal to I/Q format
                for rx in range(self.num_rx):
                    for chirp in range(self.num_chirps):
                        # Fix: Explicitly reshape the real and imaginary parts to match the expected shape
                        self.time_domain_data[i, rx, chirp, :, 0] = np.real(rx_signal[rx, chirp, :])
                        self.time_domain_data[i, rx, chirp, :, 1] = np.imag(rx_signal[rx, chirp, :])
        
        return self.time_domain_data

    def _generate_fmcw_signal(self, targets, apply_noise=True):
        """
        Generate FMCW signal with targets
        
        Args:
            targets: List of target dictionaries with range, velocity, and RCS
            apply_noise: Whether to apply noise to the signal
            
        Returns:
            FMCW signal with shape [num_rx, num_chirps, samples_per_chirp, 2]
        """
        # Initialize signal array
        signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp, 2), dtype=np.float32)
        
        # ... rest of your existing FMCW signal generation code ...
        
        # Apply noise if requested
        if apply_noise:
            for rx in range(self.num_rx):
                for chirp in range(self.num_chirps):
                    noise_power = 1.0  # Normalized noise power
                    noise = np.random.normal(0, np.sqrt(noise_power/2), 
                                           (self.samples_per_chirp, 2))
                    signal[rx, chirp, :, :] += noise
        
        return signal

    def _apply_ofdm_modulation(self, fmcw_signal):
        """
        Apply OFDM modulation to FMCW signal (simulating AD9361 baseband signal)
        
        Args:
            fmcw_signal: FMCW signal with shape [num_rx, num_chirps, samples_per_chirp, 2]
            
        Returns:
            Modulated signal with shape [num_rx, num_chirps, samples_per_chirp, 2]
        """
        # Create a copy of the input signal
        modulated_signal = np.copy(fmcw_signal)
        
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp, _ = fmcw_signal.shape
        
        # OFDM parameters - limited by AD9361 bandwidth (56 MHz max)
        num_subcarriers = 64
        subcarrier_spacing = 50e3  # 50 kHz spacing
        
        # Time vector for one chirp
        t = np.linspace(0, self.chirp_duration, samples_per_chirp)
        
        # Generate OFDM modulation
        for rx in range(num_rx):
            for chirp in range(num_chirps):
                # Convert I/Q to complex
                complex_signal = fmcw_signal[rx, chirp, :, 0] + 1j * fmcw_signal[rx, chirp, :, 1]
                
                # Random OFDM data symbols (QPSK modulation)
                data_symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_subcarriers) / np.sqrt(2)
                
                # Generate OFDM modulation
                ofdm_signal = np.zeros_like(complex_signal)
                
                # Apply each subcarrier
                for k in range(num_subcarriers):
                    # Subcarrier frequency
                    f_k = (k - num_subcarriers//2) * subcarrier_spacing
                    # Add subcarrier
                    ofdm_signal += data_symbols[k] * np.exp(2j * np.pi * f_k * t)
                
                # Normalize OFDM signal
                ofdm_signal = ofdm_signal / np.sqrt(np.mean(np.abs(ofdm_signal)**2))
                
                # Apply OFDM modulation to FMCW signal
                modulated_complex = complex_signal * ofdm_signal
                
                # Store I/Q components
                modulated_signal[rx, chirp, :, 0] = np.real(modulated_complex)
                modulated_signal[rx, chirp, :, 1] = np.imag(modulated_complex)
        
        return modulated_signal
    
    def _apply_sine_modulation(self, fmcw_signal):
        """
        Apply Sine modulation to FMCW signal (simulating AD9361 baseband signal)
        
        Args:
            fmcw_signal: FMCW signal with shape [num_rx, num_chirps, samples_per_chirp, 2]
            
        Returns:
            Modulated signal with shape [num_rx, num_chirps, samples_per_chirp, 2]
        """
        # Create a copy of the input signal
        modulated_signal = np.copy(fmcw_signal)
        
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp, _ = fmcw_signal.shape
        
        # Sine wave parameters - from AD9361
        sine_freq = self.signal_freq  # Typically 100 kHz to 1 MHz
        
        # Time vector for one chirp
        t = np.linspace(0, self.chirp_duration, samples_per_chirp)
        
        # Generate sine wave carrier
        sine_carrier = np.exp(2j * np.pi * sine_freq * t)
        
        # Apply sine modulation
        for rx in range(num_rx):
            for chirp in range(num_chirps):
                # Convert I/Q to complex
                complex_signal = fmcw_signal[rx, chirp, :, 0] + 1j * fmcw_signal[rx, chirp, :, 1]
                
                # Apply sine modulation
                modulated_complex = complex_signal * sine_carrier
                
                # Store I/Q components
                modulated_signal[rx, chirp, :, 0] = np.real(modulated_complex)
                modulated_signal[rx, chirp, :, 1] = np.imag(modulated_complex)
        
        return modulated_signal

    def _generate_ofdm_fmcw_signal(self, targets):
        """
        Generate OFDM signal modulated with FMCW sweep to simulate AD9361 + CN0566 phaser
        
        Args:
            targets: List of target dictionaries with range, velocity, and RCS
            
        Returns:
            Complex time domain signal with shape [num_rx, num_chirps, samples_per_chirp, 2]
        """
        # Initialize signal array
        signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp, 2), dtype=np.float32)
        
        # OFDM parameters - limited by AD9361 bandwidth (56 MHz max)
        num_subcarriers = 64
        subcarrier_spacing = 50e3  # 50 kHz spacing
        ofdm_bandwidth = num_subcarriers * subcarrier_spacing  # ~3.2 MHz
        symbol_duration = 1 / subcarrier_spacing
        cp_duration = symbol_duration / 4  # Cyclic prefix duration
        total_symbol_duration = symbol_duration + cp_duration
        
        # FMCW sweep parameters - from CN0566 phaser
        sweep_bandwidth = 500e6  # 500 MHz sweep
        sweep_duration = self.chirp_duration  # Typically 500 μs
        sweep_rate = sweep_bandwidth / sweep_duration
        
        # Time vector for one chirp
        t_chirp = np.linspace(0, self.chirp_duration, self.samples_per_chirp)
        
        # Generate OFDM symbols
        for chirp_idx in range(self.num_chirps):
            # Random OFDM data symbols (QPSK modulation)
            data_symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_subcarriers) / np.sqrt(2)
            
            # Generate OFDM signal
            ofdm_signal = np.zeros(self.samples_per_chirp, dtype=np.complex64)
            
            # Number of OFDM symbols that fit in one chirp
            num_symbols = int(self.chirp_duration / total_symbol_duration)
            
            # Generate each OFDM symbol
            for sym_idx in range(num_symbols):
                t_start = sym_idx * total_symbol_duration
                t_end = t_start + total_symbol_duration
                t_indices = np.where((t_chirp >= t_start) & (t_chirp < t_end))[0]
                
                if len(t_indices) == 0:
                    continue
                
                # Time within this symbol
                t_sym = t_chirp[t_indices] - t_start
                
                # Generate symbol (excluding cyclic prefix)
                cp_indices = np.where(t_sym < cp_duration)[0]
                sym_indices = np.where(t_sym >= cp_duration)[0]
                
                if len(sym_indices) == 0:
                    continue
                
                # Generate OFDM symbol
                symbol = np.zeros(len(sym_indices), dtype=np.complex64)
                for k in range(num_subcarriers):
                    # Subcarrier frequency
                    f_k = (k - num_subcarriers//2) * subcarrier_spacing
                    # Add subcarrier
                    symbol += data_symbols[k] * np.exp(2j * np.pi * f_k * (t_sym[sym_indices] - cp_duration))
                
                # Add symbol to OFDM signal
                ofdm_signal[t_indices[sym_indices]] = symbol
                
                # Add cyclic prefix (copy from end of symbol)
                if len(cp_indices) > 0 and len(sym_indices) > len(cp_indices):
                    ofdm_signal[t_indices[cp_indices]] = symbol[-len(cp_indices):]
            
            # Apply FMCW sweep from CN0566 phaser
            for rx_idx in range(self.num_rx):
                # Initialize received signal for this RX antenna
                rx_signal = np.zeros(self.samples_per_chirp, dtype=np.complex64)
                
                # Add target returns
                for target in targets:
                    # Target parameters
                    range_m = target['range']
                    velocity = target['velocity']
                    rcs = target['rcs']
                    
                    # Calculate signal parameters
                    delay_s = 2 * range_m / 3e8  # Round-trip delay in seconds
                    doppler_hz = 2 * velocity * self.center_freq / 3e8  # Doppler shift in Hz
                    
                    # Calculate amplitude based on radar equation
                    amplitude = np.sqrt(rcs) / (range_m ** 2)  # Simplified radar equation
                    
                    # Apply SNR scaling
                    snr_linear = 10 ** (target['snr'] / 10)
                    amplitude *= np.sqrt(snr_linear)
                    
                    # Calculate delayed and Doppler-shifted signal
                    delayed_samples = int(delay_s * self.sample_rate)
                    if delayed_samples >= self.samples_per_chirp:
                        continue  # Target too far, skip
                    
                    # Delayed OFDM signal
                    delayed_ofdm = np.zeros_like(ofdm_signal)
                    delayed_ofdm[delayed_samples:] = ofdm_signal[:(self.samples_per_chirp - delayed_samples)]
                    
                    # Apply FMCW phase shift due to target range
                    # Phase shift due to FMCW sweep
                    phase_shift = 2 * np.pi * (
                        self.center_freq * delay_s +  # Carrier phase shift
                        sweep_rate * delay_s * t_chirp -  # Beat frequency
                        0.5 * sweep_rate * delay_s ** 2  # Residual video phase
                    )
                    
                    # Apply Doppler shift
                    doppler_shift = 2 * np.pi * doppler_hz * t_chirp
                    
                    # Combine all phase shifts
                    total_phase = phase_shift + doppler_shift
                    
                    # Add to received signal
                    rx_signal += amplitude * delayed_ofdm * np.exp(1j * total_phase)
                
                # Add noise
                noise_power = 1.0  # Normalized noise power
                noise = np.sqrt(noise_power/2) * (
                    np.random.normal(0, 1, self.samples_per_chirp) + 
                    1j * np.random.normal(0, 1, self.samples_per_chirp)
                )
                rx_signal += noise
                
                # Store I/Q components
                signal[rx_idx, chirp_idx, :, 0] = rx_signal.real
                signal[rx_idx, chirp_idx, :, 1] = rx_signal.imag
        
        return signal

    def _generate_sine_fmcw_signal(self, targets):
        """
        Generate Sine signal modulated with FMCW sweep to simulate AD9361 + CN0566 phaser
        
        Args:
            targets: List of target dictionaries with range, velocity, and RCS
            
        Returns:
            Complex time domain signal with shape [num_rx, num_chirps, samples_per_chirp, 2]
        """
        # Initialize signal array
        signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp, 2), dtype=np.float32)
        
        # Sine wave parameters - from AD9361
        sine_freq = self.signal_freq  # Typically 100 kHz to 1 MHz
        
        # FMCW sweep parameters - from CN0566 phaser
        sweep_bandwidth = 500e6  # 500 MHz sweep
        sweep_duration = self.chirp_duration  # Typically 500 μs
        sweep_rate = sweep_bandwidth / sweep_duration
        
        # Time vector for one chirp
        t_chirp = np.linspace(0, self.chirp_duration, self.samples_per_chirp)
        
        # Generate sine wave carrier
        sine_carrier = np.exp(2j * np.pi * sine_freq * t_chirp)
        
        # Process each chirp
        for chirp_idx in range(self.num_chirps):
            # Apply FMCW sweep from CN0566 phaser
            for rx_idx in range(self.num_rx):
                # Initialize received signal for this RX antenna
                rx_signal = np.zeros(self.samples_per_chirp, dtype=np.complex64)
                
                # Add target returns
                for target in targets:
                    # Target parameters
                    range_m = target['range']
                    velocity = target['velocity']
                    rcs = target['rcs']
                    
                    # Calculate signal parameters
                    delay_s = 2 * range_m / 3e8  # Round-trip delay in seconds
                    doppler_hz = 2 * velocity * self.center_freq / 3e8  # Doppler shift in Hz
                    
                    # Calculate amplitude based on radar equation
                    amplitude = np.sqrt(rcs) / (range_m ** 2)  # Simplified radar equation
                    
                    # Apply SNR scaling
                    snr_linear = 10 ** (target['snr'] / 10)
                    amplitude *= np.sqrt(snr_linear)
                    
                    # Calculate delayed and Doppler-shifted signal
                    delayed_samples = int(delay_s * self.sample_rate)
                    if delayed_samples >= self.samples_per_chirp:
                        continue  # Target too far, skip
                    
                    # Delayed sine carrier
                    delayed_sine = np.zeros_like(sine_carrier)
                    delayed_sine[delayed_samples:] = sine_carrier[:(self.samples_per_chirp - delayed_samples)]
                    
                    # Apply FMCW phase shift due to target range
                    # Phase shift due to FMCW sweep
                    phase_shift = 2 * np.pi * (
                        self.center_freq * delay_s +  # Carrier phase shift
                        sweep_rate * delay_s * t_chirp -  # Beat frequency
                        0.5 * sweep_rate * delay_s ** 2  # Residual video phase
                    )
                    
                    # Apply Doppler shift
                    doppler_shift = 2 * np.pi * doppler_hz * t_chirp
                    
                    # Combine all phase shifts
                    total_phase = phase_shift + doppler_shift
                    
                    # Add to received signal
                    rx_signal += amplitude * delayed_sine * np.exp(1j * total_phase)
                
                # Add noise
                noise_power = 1.0  # Normalized noise power
                noise = np.sqrt(noise_power/2) * (
                    np.random.normal(0, 1, self.samples_per_chirp) + 
                    1j * np.random.normal(0, 1, self.samples_per_chirp)
                )
                rx_signal += noise
                
                # Store I/Q components
                signal[rx_idx, chirp_idx, :, 0] = rx_signal.real
                signal[rx_idx, chirp_idx, :, 1] = rx_signal.imag
        
        return signal

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
            rx_signal = np.zeros((self.num_rx, self.num_chirps, self.samples_per_chirp), dtype=np.complex64) #(4, 12, 1500)
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
                #(4, 12, 1500) complex64
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
            chirp_signal = chirp(t, f0=f0, f1=f1, t1=self.chirp_duration, method='linear') #(1500,) float64
            
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
        elif self.signal_type.upper() == 'OFDM_FMCW':
            # For OFDM_FMCW, we need to handle this in _generate_time_domain_data
            # Return a placeholder that will be replaced
            chirp_complex = np.ones(self.samples_per_chirp, dtype=complex)
        elif self.signal_type.upper() == 'SINE_FMCW':
            # For Sine_FMCW, we need to handle this in _generate_time_domain_data
            # Return a placeholder that will be replaced
            chirp_complex = np.ones(self.samples_per_chirp, dtype=complex)
        else:
            # Default to a simple sine wave if signal type is not recognized
            print(f"Warning: Signal type '{self.signal_type}' not recognized. Using default sine wave.")
            chirp_complex = np.exp(1j * 2 * np.pi * (self.bandwidth/2) * t)
        
        # Normalize power
        chirp_complex = chirp_complex / np.sqrt(np.mean(np.abs(chirp_complex)**2))
        
        return chirp_complex #(1500,) complex128
    
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
    
    def _time_to_range_doppler(self, time_data):
        """
        Convert time domain data to range-Doppler map
        
        Args:
            time_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
            
        Returns:
            Range-Doppler map with shape [num_doppler_bins, num_range_bins]
        """
        # Sum across RX channels (non-coherent combining)
        combined_data = np.sum(time_data, axis=0)
        
        # Apply window function to reduce sidelobes
        window_range = np.hamming(self.samples_per_chirp).reshape(1, -1)
        window_doppler = np.hamming(self.num_chirps).reshape(-1, 1)
        windowed_data = combined_data * window_doppler * window_range
        
        # Special processing for combined signal types
        if self.signal_type in ['OFDM_FMCW', 'Sine_FMCW']:
            # For combined signals, we need to extract the beat frequency from the FMCW sweep
            # This is similar to regular FMCW processing but accounts for the baseband modulation
            
            # Range FFT (along fast-time/samples dimension)
            range_fft = np.fft.fft(windowed_data, n=self.num_range_bins, axis=1)
            range_fft = np.fft.fftshift(range_fft, axes=1)
            
            # Doppler FFT (along slow-time/chirps dimension)
            doppler_fft = np.fft.fft(range_fft, n=self.num_doppler_bins, axis=0)
            doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
            
            # Return the 2D range-Doppler map
            return doppler_fft
        else:
            # Standard processing for other signal types
            # Range FFT (along fast-time/samples dimension)
            range_fft = np.fft.fft(windowed_data, n=self.num_range_bins, axis=1)
            range_fft = np.fft.fftshift(range_fft, axes=1)
            
            # Doppler FFT (along slow-time/chirps dimension)
            doppler_fft = np.fft.fft(range_fft, n=self.num_doppler_bins, axis=0)
            doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
            
            # Return the 2D range-Doppler map
            return doppler_fft

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
        # Check if the file exists
        if not os.path.exists(datapath):
            # Try to find a signal-type specific file instead
            base_dir = os.path.dirname(datapath)
            base_name = os.path.basename(datapath)
            
            # Extract file extension
            file_ext = os.path.splitext(base_name)[1]
            
            # Check for signal-type specific files
            if hasattr(self, 'signal_type') and self.signal_type:
                signal_specific_path = os.path.join(
                    base_dir, 
                    f"radar_simulation_data_{self.signal_type.lower()}{file_ext}"
                )
                
                if os.path.exists(signal_specific_path):
                    print(f"File {datapath} not found, using signal-specific file: {signal_specific_path}")
                    datapath = signal_specific_path
                else:
                    # Try with .h5 extension if .npy was not found
                    if file_ext.lower() == '.npy':
                        h5_path = os.path.join(
                            base_dir, 
                            f"radar_simulation_data_{self.signal_type.lower()}.h5"
                        )
                        if os.path.exists(h5_path):
                            print(f"File {datapath} not found, using signal-specific HDF5 file: {h5_path}")
                            datapath = h5_path
                    
                    # If still not found, raise error
                    if not os.path.exists(datapath):
                        raise FileNotFoundError(
                            f"Data file not found: {datapath}\n"
                            f"Signal-specific file also not found: {signal_specific_path}\n"
                            f"Please make sure the file exists or generate the dataset first."
                        )
            else:
                # List available files in the directory
                available_files = [f for f in os.listdir(base_dir) if f.startswith("radar_simulation_data_")]
                
                if available_files:
                    suggested_file = os.path.join(base_dir, available_files[0])
                    raise FileNotFoundError(
                        f"Data file not found: {datapath}\n"
                        f"Available files: {', '.join(available_files)}\n"
                        f"Try using: {suggested_file}"
                    )
                else:
                    raise FileNotFoundError(
                        f"Data file not found: {datapath}\n"
                        f"No radar simulation data files found in {base_dir}\n"
                        f"Please generate the dataset first."
                    )
        
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
                # Convert to specified precision (float32 or float16)
                dtype = np.float32 if self.precision == 'float32' else np.float16
                
                # Load and convert data to the specified precision
                sample['feature_2d'] = np.array(self.h5_file['range_doppler_maps'][idx], dtype=dtype)
                sample['labels'] = np.array(self.h5_file['target_masks'][idx], dtype=dtype)
                
                # Load time_domain data if available
                if 'time_domain_data' in self.h5_file:
                    sample['time_domain'] = np.array(self.h5_file['time_domain_data'][idx], dtype=dtype)
                
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
                
            # Set the appropriate dtype based on precision setting
            dtype = np.float32 if self.precision == 'float32' else np.float16
            
            sample = {
                'feature_2d': rd_feature.astype(dtype),  # [2, num_doppler_bins, num_range_bins]
                'labels': target_mask.astype(dtype),     # [num_doppler_bins, num_range_bins, 1]
                'target_info': self.target_info[idx] if self.target_info else None
            }
            
            if time_feature is not None:
                sample['time_domain'] = time_feature.astype(dtype)  # [num_rx, num_chirps, samples_per_chirp, 2]
        
        # Apply training augmentations to the loaded data (for both in-memory and h5 data)
        if self.training:
            # Add random noise to make the model more robust
            noise_level = random.uniform(0.05, 0.2)
            
            # Add noise to range-Doppler maps
            if 'feature_2d' in sample:
                noise = np.random.normal(0, noise_level, sample['feature_2d'].shape).astype(dtype)
                sample['feature_2d'] = sample['feature_2d'] + noise
            
            # Add noise to time domain data if available
            if 'time_domain' in sample:
                time_noise = np.random.normal(0, noise_level, sample['time_domain'].shape).astype(dtype)
                sample['time_domain'] = sample['time_domain'] + time_noise
        
        # Apply realistic RF impairments to time domain data if available
        if 'time_domain' in sample and hasattr(self, 'apply_realistic_effects') and self.apply_realistic_effects:
            sample['time_domain'] = self._apply_realistic_rf_effects(sample['time_domain'], sample.get('target_info'))
            
            # Ensure time_domain data is in the correct dtype after applying effects
            dtype = np.float32 if self.precision == 'float32' else np.float16
            sample['time_domain'] = sample['time_domain'].astype(dtype)
            
            # Recalculate range-Doppler map from the modified time domain data
            if self.recalculate_rd_map:
                # Convert I/Q format back to complex
                complex_data = sample['time_domain'][..., 0] + 1j * sample['time_domain'][..., 1]
                
                # Process using the range-Doppler processing chain
                rd_map = self._time_to_range_doppler(complex_data)
                
                # Update the feature_2d with the new range-Doppler map
                sample['feature_2d'][0, :, :] = np.real(rd_map).astype(dtype)
                sample['feature_2d'][1, :, :] = np.imag(rd_map).astype(dtype)
        
        # Final check to ensure all tensors are in the correct dtype
        dtype = np.float32 if self.precision == 'float32' else np.float16
        for key in sample:
            if isinstance(sample[key], np.ndarray):
                sample[key] = sample[key].astype(dtype)
        
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
        clean_sample = self[idx] #dict
        
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
    
    def traditional_radar_processing(self, time_domain_data, signal_type='FMCW'):
        """
        Process radar data using traditional signal processing techniques based on signal type
        
        Args:
            time_domain_data: Time domain data with shape [num_rx, num_chirps, samples_per_chirp, 2]
                            where the last dimension contains I/Q data
            signal_type: Type of radar signal ('FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW')
            
        Returns:
            Range-Doppler map with shape [num_doppler_bins, num_range_bins]
        """
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp, _ = time_domain_data.shape
        
        # Convert I/Q data to complex
        complex_data = time_domain_data[..., 0] + 1j * time_domain_data[..., 1]
        
        # Initialize range-Doppler map
        rd_map = np.zeros((self.num_doppler_bins, self.num_range_bins), dtype=np.complex64)
        
        # Process based on signal type
        if signal_type == 'FMCW':
            # Standard FMCW processing
            # 1. Range FFT for each chirp
            range_profiles = np.zeros((num_rx, num_chirps, self.num_range_bins), dtype=np.complex64)
            
            for rx in range(num_rx):
                for chirp in range(num_chirps):
                    # Apply window function to reduce sidelobes
                    window = np.hamming(samples_per_chirp)
                    windowed_data = complex_data[rx, chirp] * window
                    
                    # Perform range FFT
                    range_fft = np.fft.fft(windowed_data, n=self.num_range_bins)
                    range_profiles[rx, chirp] = range_fft
            
            # 2. Doppler FFT across chirps
            for rx in range(num_rx):
                for r in range(self.num_range_bins):
                    # Apply window function for Doppler processing
                    doppler_window = np.hamming(num_chirps)
                    windowed_doppler = range_profiles[rx, :, r] * doppler_window
                    
                    # Perform Doppler FFT
                    doppler_fft = np.fft.fftshift(np.fft.fft(windowed_doppler, n=self.num_doppler_bins))
                    
                    # Add to range-Doppler map (coherent integration across RX channels)
                    rd_map[:, r] += doppler_fft
            
            # Normalize by number of RX antennas
            rd_map = rd_map / num_rx
            
        elif signal_type == 'OFDM':
            # OFDM radar processing
            # 1. Perform channel estimation for each OFDM symbol
            # 2. Extract range and Doppler information from channel estimates
            
            # For OFDM, we need to reshape the data to match OFDM structure
            # Assuming OFDM parameters
            num_subcarriers = 64
            num_symbols = num_chirps
            
            # Reshape data to match OFDM structure
            ofdm_data = np.zeros((num_rx, num_symbols, num_subcarriers), dtype=np.complex64)
            
            for rx in range(num_rx):
                for symbol in range(num_symbols):
                    # Extract OFDM symbol
                    ofdm_symbol = complex_data[rx, symbol]
                    
                    # Remove cyclic prefix (assuming 25% CP)
                    cp_length = samples_per_chirp // 4
                    symbol_data = ofdm_symbol[cp_length:]
                    
                    # FFT to get frequency domain
                    freq_data = np.fft.fft(symbol_data, n=num_subcarriers)
                    ofdm_data[rx, symbol] = freq_data
            
            # Perform range-Doppler processing on OFDM channel estimates
            for rx in range(num_rx):
                # Range processing (IFFT across subcarriers)
                for symbol in range(num_symbols):
                    range_profile = np.fft.ifft(ofdm_data[rx, symbol], n=self.num_range_bins)
                    
                    # Doppler processing (FFT across symbols)
                    for r in range(self.num_range_bins):
                        doppler_data = np.zeros(num_symbols, dtype=np.complex64)
                        for s in range(num_symbols):
                            if r < len(range_profile):
                                doppler_data[s] = range_profile[r]
                        
                        # Apply window and FFT
                        doppler_window = np.hamming(num_symbols)
                        windowed_doppler = doppler_data * doppler_window
                        doppler_fft = np.fft.fftshift(np.fft.fft(windowed_doppler, n=self.num_doppler_bins))
                        
                        # Add to range-Doppler map
                        rd_map[:, r] += doppler_fft
            
            # Normalize
            rd_map = rd_map / num_rx
            
        elif signal_type == 'Sine':
            # Sine wave processing (similar to CW radar)
            # For sine wave, we mainly look at phase changes across chirps
            
            # Initialize phase data
            phase_data = np.zeros((num_rx, num_chirps), dtype=np.complex64)
            
            for rx in range(num_rx):
                for chirp in range(num_chirps):
                    # Extract average phase from the sine wave
                    phase_data[rx, chirp] = np.mean(complex_data[rx, chirp])
            
            # Process Doppler information from phase changes
            for rx in range(num_rx):
                # Doppler FFT across chirps
                doppler_window = np.hamming(num_chirps)
                windowed_doppler = phase_data[rx] * doppler_window
                doppler_fft = np.fft.fftshift(np.fft.fft(windowed_doppler, n=self.num_doppler_bins))
                
                # Place in range-Doppler map (at center range bin)
                center_range = self.num_range_bins // 2
                rd_map[:, center_range] += doppler_fft
            
            # Normalize
            rd_map = rd_map / num_rx
            
        elif signal_type == 'OFDM_FMCW' or signal_type == 'Sine_FMCW':
            # For combined signals, we need to process the demodulated data
            # The CN0566 hardware has already demodulated the FMCW component
            
            # Process similar to FMCW but with additional filtering
            # 1. Range FFT for each chirp
            range_profiles = np.zeros((num_rx, num_chirps, self.num_range_bins), dtype=np.complex64)
            
            for rx in range(num_rx):
                for chirp in range(num_chirps):
                    # Apply bandpass filter to isolate the signal of interest
                    # For OFDM_FMCW, we need to filter around OFDM subcarriers
                    # For Sine_FMCW, we need to filter around the sine frequency
                    
                    # Apply window function
                    window = np.hamming(samples_per_chirp)
                    
                    if signal_type == 'OFDM_FMCW':
                        # For OFDM, apply filter bank around subcarrier frequencies
                        filtered_data = self._apply_ofdm_filter(complex_data[rx, chirp])
                    else:  # Sine_FMCW
                        # For Sine, apply bandpass filter around sine frequency
                        filtered_data = self._apply_sine_filter(complex_data[rx, chirp])
                    
                    # Apply window and FFT
                    windowed_data = filtered_data * window
                    range_fft = np.fft.fft(windowed_data, n=self.num_range_bins)
                    range_profiles[rx, chirp] = range_fft
            
            # 2. Doppler FFT across chirps
            for rx in range(num_rx):
                for r in range(self.num_range_bins):
                    # Apply window function for Doppler processing
                    doppler_window = np.hamming(num_chirps)
                    windowed_doppler = range_profiles[rx, :, r] * doppler_window
                    
                    # Perform Doppler FFT
                    doppler_fft = np.fft.fftshift(np.fft.fft(windowed_doppler, n=self.num_doppler_bins))
                    
                    # Add to range-Doppler map
                    rd_map[:, r] += doppler_fft
            
            # Normalize
            rd_map = rd_map / num_rx
        
        # Apply CFAR detection to enhance targets
        rd_map_magnitude = np.abs(rd_map)
        rd_map_cfar = self._apply_cfar(rd_map_magnitude)
        
        # Convert back to complex
        rd_map_normalized = rd_map_cfar * np.exp(1j * np.angle(rd_map))
        
        return rd_map_normalized

    def _apply_cfar(self, rd_map, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-4):
        """
        Apply Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection
        
        Args:
            rd_map: Range-Doppler map magnitude
            guard_cells: Tuple of (doppler_guard, range_guard) cells to exclude around CUT
            training_cells: Tuple of (doppler_train, range_train) cells to use for estimation
            pfa: Probability of false alarm
            
        Returns:
            Binary detection map
        """
        doppler_bins, range_bins = rd_map.shape
        
        # Extract guard and training cell sizes
        doppler_guard, range_guard = guard_cells
        doppler_train, range_train = training_cells
        
        # Calculate total number of training cells
        num_training_cells = (2*doppler_train + 2*range_train + 4*doppler_train*range_train)
        
        # Calculate CFAR threshold factor
        # For CA-CFAR: threshold = alpha * noise_level
        # alpha = num_training_cells * (pfa^(-1/num_training_cells) - 1)
        alpha = num_training_cells * (pfa**(-1/num_training_cells) - 1)
        
        # Initialize output map
        cfar_map = np.zeros_like(rd_map)
        
        # Apply CFAR for each cell
        for i in range(doppler_bins):
            for j in range(range_bins):
                # Define cell under test (CUT)
                cut = rd_map[i, j]
                
                # Define guard cell region
                guard_low_doppler = max(0, i - doppler_guard)
                guard_high_doppler = min(doppler_bins, i + doppler_guard + 1)
                guard_low_range = max(0, j - range_guard)
                guard_high_range = min(range_bins, j + range_guard + 1)
                
                # Define training cell region
                train_low_doppler = max(0, i - doppler_guard - doppler_train)
                train_high_doppler = min(doppler_bins, i + doppler_guard + doppler_train + 1)
                train_low_range = max(0, j - range_guard - range_train)
                train_high_range = min(range_bins, j + range_guard + range_train + 1)
                
                # Extract training cells (excluding guard cells and CUT)
                training_region = np.concatenate([
                    rd_map[train_low_doppler:guard_low_doppler, train_low_range:train_high_range].flatten(),
                    rd_map[guard_high_doppler:train_high_doppler, train_low_range:train_high_range].flatten(),
                    rd_map[guard_low_doppler:guard_high_doppler, train_low_range:guard_low_range].flatten(),
                    rd_map[guard_low_doppler:guard_high_doppler, guard_high_range:train_high_range].flatten()
                ])
                
                # Calculate noise level (mean of training cells)
                if len(training_region) > 0:
                    noise_level = np.mean(training_region)
                    
                    # Apply threshold
                    threshold = alpha * noise_level
                    
                    # Compare CUT with threshold
                    if cut > threshold:
                        cfar_map[i, j] = cut
                
        return cfar_map

    def _apply_ofdm_filter(self, signal):
        """
        Apply filter bank for OFDM subcarriers
        
        Args:
            signal: Complex time domain signal
            
        Returns:
            Filtered signal
        """
        # OFDM parameters
        num_subcarriers = 64
        subcarrier_spacing = 50e3  # 50 kHz
        
        # FFT to frequency domain
        signal_fft = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/self.sample_rate))
        
        # Create filter mask for OFDM subcarriers
        mask = np.zeros_like(freq, dtype=bool)
        
        # Mark subcarrier regions
        for k in range(num_subcarriers):
            # Subcarrier frequency
            f_k = (k - num_subcarriers//2) * subcarrier_spacing
            
            # Find closest frequency bin
            idx = np.argmin(np.abs(freq - f_k))
            
            # Mark a small region around this frequency
            width = int(subcarrier_spacing * 0.8 * len(freq) / self.sample_rate)
            low_idx = max(0, idx - width//2)
            high_idx = min(len(freq), idx + width//2 + 1)
            mask[low_idx:high_idx] = True
        
        # Apply mask
        filtered_fft = signal_fft * mask
        
        # Convert back to time domain
        filtered_signal = np.fft.ifft(np.fft.ifftshift(filtered_fft))
        
        return filtered_signal

    def _apply_sine_filter(self, signal):
        """
        Apply bandpass filter around sine frequency
        
        Args:
            signal: Complex time domain signal
            
        Returns:
            Filtered signal
        """
        # Sine wave frequency
        sine_freq = self.signal_freq
        
        # FFT to frequency domain
        signal_fft = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/self.sample_rate))
        
        # Create bandpass filter
        bandwidth = sine_freq * 0.2  # 20% bandwidth
        mask = np.abs(freq - sine_freq) < bandwidth
        
        # Apply filter
        filtered_fft = signal_fft * mask
        
        # Convert back to time domain
        filtered_signal = np.fft.ifft(np.fft.ifftshift(filtered_fft))
        
        return filtered_signal

    def process_data_traditional(self, time_domain_data=None):
        """
        Process radar data using traditional signal processing for all samples
        
        Args:
            time_domain_data: Optional time domain data to process. If None, use self.time_domain_data
            
        Returns:
            Processed range-Doppler maps and detection masks
        """
        if time_domain_data is None:
            time_domain_data = self.time_domain_data
        
        num_samples = time_domain_data.shape[0]
        
        # Initialize output arrays
        rd_maps = np.zeros((num_samples, 2, self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        detection_masks = np.zeros((num_samples, self.num_doppler_bins, self.num_range_bins, 1), dtype=np.float32)
        
        print(f"Processing {num_samples} samples using traditional radar processing...")
        
        for i in tqdm(range(num_samples), desc="Traditional processing"):
            # Process this sample
            rd_map = self.traditional_radar_processing(time_domain_data[i], signal_type=self.signal_type)
            
            # Store real and imaginary parts
            rd_maps[i, 0, :, :] = np.real(rd_map)
            rd_maps[i, 1, :, :] = np.imag(rd_map)
            
            # Create detection mask (using magnitude)
            rd_magnitude = np.abs(rd_map)
            
            # Normalize magnitude to [0, 1]
            if np.max(rd_magnitude) > 0:
                rd_magnitude = rd_magnitude / np.max(rd_magnitude)
            
            # Apply threshold for detection
            threshold = 0.15  # Adjust based on your needs
            detection = (rd_magnitude > threshold).astype(np.float32)
            
            # Store detection mask
            detection_masks[i, :, :, 0] = detection
        
        return rd_maps, detection_masks

    def evaluate_traditional_processing(self, ground_truth_masks=None):
        """
        Evaluate traditional radar processing against ground truth
        
        Args:
            ground_truth_masks: Optional ground truth masks. If None, use self.target_masks
            
        Returns:
            Dictionary of evaluation metrics
        """
        if ground_truth_masks is None:
            ground_truth_masks = self.target_masks
        
        # Process data using traditional methods
        _, detection_masks = self.process_data_traditional()
        
        # Calculate metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range(len(ground_truth_masks)):
            gt = ground_truth_masks[i, :, :, 0]
            pred = detection_masks[i, :, :, 0]
            
            # Count true positives, false positives, and false negatives
            true_positives += np.sum((gt > 0) & (pred > 0))
            false_positives += np.sum((gt == 0) & (pred > 0))
            false_negatives += np.sum((gt > 0) & (pred == 0))
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Return metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        return metrics

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
    signal_types = ['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW']
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
    
    # Compare traditional radar processing performance
    print("\n=== COMPARING TRADITIONAL RADAR PROCESSING PERFORMANCE ===")
    
    # Create directory for traditional processing comparison
    os.makedirs('data/radar/traditional_processing', exist_ok=True)
    
    # Initialize metrics storage
    traditional_metrics = {}
    
    # Process each signal type with traditional methods
    for signal_type in signal_types:
        print(f"\nProcessing {signal_type} with traditional radar techniques...")
        dataset = test_datasets[signal_type]
        
        # Process data using traditional methods
        rd_maps, detection_masks = dataset.process_data_traditional()
        
        # Evaluate against ground truth
        metrics = dataset.evaluate_traditional_processing()
        traditional_metrics[signal_type] = metrics
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # Visualize traditional processing results for first sample
        plt.figure(figsize=(15, 10))
        
        # Original range-Doppler map
        plt.subplot(2, 2, 1)
        rd_map = dataset.range_doppler_maps[0]
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        plt.imshow(20*np.log10(rd_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(f'Original Range-Doppler Map')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Traditional processing result
        plt.subplot(2, 2, 2)
        trad_rd_map = rd_maps[0]
        trad_magnitude = np.sqrt(trad_rd_map[0]**2 + trad_rd_map[1]**2)
        plt.imshow(20*np.log10(trad_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(f'Traditional Processing Result')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Ground truth mask
        plt.subplot(2, 2, 3)
        plt.imshow(dataset.target_masks[0, :, :, 0], aspect='auto', cmap='gray')
        plt.colorbar(label='Target Presence')
        plt.title('Ground Truth Mask')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Detection mask from traditional processing
        plt.subplot(2, 2, 4)
        plt.imshow(detection_masks[0, :, :, 0], aspect='auto', cmap='gray')
        plt.colorbar(label='Detection')
        plt.title('Traditional Detection Mask')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        plt.suptitle(f'Traditional Radar Processing - {signal_type}')
        plt.tight_layout()
        plt.savefig(f'data/radar/traditional_processing/{signal_type}_traditional_processing{IMG_FORMAT}')
        plt.close()
        
        # Visualize CFAR detection for this signal type
        plt.figure(figsize=(15, 5))
        
        # Original range-Doppler magnitude
        plt.subplot(1, 3, 1)
        plt.imshow(20*np.log10(rd_magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Original Range-Doppler')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # CFAR detection result
        plt.subplot(1, 3, 2)
        # Apply CFAR directly for visualization
        #cfar_result = dataset._apply_cfar(rd_magnitude)
        # When calling _apply_cfar, use different parameters
        cfar_result = dataset._apply_cfar(rd_magnitude, 
                                 guard_cells=(1, 1),  # Smaller guard region
                                 training_cells=(6, 6),  # Larger training region
                                 pfa=1e-3)  # Higher probability of false alarm
        plt.imshow(cfar_result, aspect='auto', cmap='jet')
        plt.colorbar(label='CFAR Output')
        plt.title('CFAR Detection')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Overlay of targets on CFAR result
        plt.subplot(1, 3, 3)
        # Create RGB image for overlay
        overlay = np.zeros((rd_magnitude.shape[0], rd_magnitude.shape[1], 3))
        # Normalize CFAR result for red channel
        if np.max(cfar_result) > 0:
            overlay[:, :, 0] = cfar_result / np.max(cfar_result)
        # Use ground truth for green channel
        overlay[:, :, 1] = dataset.target_masks[0, :, :, 0]
        plt.imshow(overlay, aspect='auto')
        plt.title('CFAR (red) vs Ground Truth (green)')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        plt.suptitle(f'CFAR Detection - {signal_type}')
        plt.tight_layout()
        plt.savefig(f'data/radar/traditional_processing/{signal_type}_cfar_detection{IMG_FORMAT}')
        plt.close()
    
    # Create comparative visualizations across signal types
    
    # 1. Performance metrics comparison
    plt.figure(figsize=(12, 8))
    
    # Extract metrics for plotting
    metrics_names = ['precision', 'recall', 'f1_score']
    x = np.arange(len(metrics_names))
    width = 0.15
    
    # Plot bars for each signal type
    for i, signal_type in enumerate(signal_types):
        metrics_values = [traditional_metrics[signal_type][metric] for metric in metrics_names]
        plt.bar(x + i*width, metrics_values, width, label=signal_type)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Traditional Radar Processing Performance Comparison')
    plt.xticks(x + width*2, metrics_names)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'data/radar/traditional_processing/performance_comparison{IMG_FORMAT}')
    plt.close()
    
    # 2. SNR vs Detection Rate
    plt.figure(figsize=(12, 8))
    
    # Create synthetic SNR range for plotting
    snr_range = np.linspace(0, 30, 100)
    
    # Plot theoretical detection curves for each signal type
    # These are approximations based on signal properties
    for i, signal_type in enumerate(signal_types):
        if signal_type == 'FMCW':
            # Standard FMCW has good performance
            detection_prob = 1 / (1 + np.exp(-(snr_range - 10) / 2))
        elif signal_type == 'OFDM':
            # OFDM has better low-SNR performance due to coding gain
            detection_prob = 1 / (1 + np.exp(-(snr_range - 8) / 2))
        elif signal_type == 'Sine':
            # Sine wave has worse range resolution but good Doppler
            detection_prob = 1 / (1 + np.exp(-(snr_range - 12) / 2))
        elif signal_type == 'OFDM_FMCW':
            # Combined signals have better performance
            detection_prob = 1 / (1 + np.exp(-(snr_range - 7) / 2))
        elif signal_type == 'Sine_FMCW':
            # Sine_FMCW is between FMCW and OFDM_FMCW
            detection_prob = 1 / (1 + np.exp(-(snr_range - 9) / 2))
        
        plt.plot(snr_range, detection_prob, label=signal_type, linewidth=2)
    
    # Add measured points from our simulations
    for signal_type in signal_types:
        # Use F1 score as overall detection performance
        f1_score = traditional_metrics[signal_type]['f1_score']
        # Use average SNR from dataset
        # avg_snr = np.mean([target['snr'] for sample in test_datasets[signal_type].target_info 
        #                   for target in sample['targets']])

        # avg_snr = np.mean([target['snr'] for sample_idx in range(len(test_datasets[signal_type].target_info))
        #           for target in test_datasets[signal_type].target_info[sample_idx]['targets']])
        
        # Calculate average SNR across all targets in all samples
        # Calculate average SNR across all targets in all samples
        snr_values = []
        for sample_idx in range(len(test_datasets[signal_type].target_info)):
            # The target_info is a list of lists, not a list of dictionaries with 'targets' key
            # Access the targets directly from the list
            targets = test_datasets[signal_type].target_info[sample_idx] #list of len 1
            for target in targets: #target is dict
                snr_values.append(target['snr'])
        
        avg_snr = np.mean(snr_values) if snr_values else 15  # Default to 15 if no values

        plt.scatter(avg_snr, f1_score, marker='o', s=100, 
                   label=f'{signal_type} (measured)', edgecolors='black')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Detection Probability')
    plt.title('SNR vs Detection Performance for Different Signal Types')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'data/radar/traditional_processing/snr_vs_detection{IMG_FORMAT}')
    plt.close()
    
    # 3. Range-Doppler resolution comparison
    plt.figure(figsize=(15, 10))
    
    # For each signal type, show a point target's response
    for i, signal_type in enumerate(signal_types):
        plt.subplot(2, 3, i+1)
        
        # Create a dataset with a single point target at the center
        point_target_dataset = RadarDataset(
            num_samples=1,
            num_range_bins=64,
            num_doppler_bins=12,
            snr_min=30,  # High SNR for clear visualization
            snr_max=30,
            max_targets=1,  # Just one target
            training=False,
            drawfig=False,
            save_data=False,
            sample_rate=3e6,
            chirp_duration=500e-6,
            num_chirps=12,
            bandwidth=500e6,
            center_freq=2.1e9,
            num_rx=4,
            num_tx=1,
            signal_type=signal_type
        )
        
        # Force target to be at center
        center_range_bin = point_target_dataset.num_range_bins // 2
        center_doppler_bin = point_target_dataset.num_doppler_bins // 2
        
        # Override target info - Fix the structure to match what's expected
        # Instead of point_target_dataset.target_info[0]['targets'] = [{...}]
        # Directly set the targets list
        point_target_dataset.target_info[0] = [{
            'range': point_target_dataset.range_resolution * center_range_bin,
            'velocity': 0,  # Zero velocity for clear range response
            'rcs': 1.0,
            'snr': 30,
            'range_bin': center_range_bin,
            'doppler_bin': center_doppler_bin
        }]
        
        # Regenerate data with this specific target
        point_target_dataset._generate_time_domain_data()
        #point_target_dataset._generate_range_doppler_maps()

        # First, get the time domain data
        time_data = point_target_dataset.time_domain_data
        
        # Then convert it to range-Doppler maps
        point_target_dataset.range_doppler_maps = point_target_dataset.time_to_range_doppler_batch(time_data)
        
        # Process with traditional methods
        rd_maps, _ = point_target_dataset.process_data_traditional()
        
        # Get magnitude
        rd_magnitude = np.sqrt(rd_maps[0, 0]**2 + rd_maps[0, 1]**2)
        
        # Normalize for comparison
        if np.max(rd_magnitude) > 0:
            rd_magnitude = rd_magnitude / np.max(rd_magnitude)
        
        # Plot in dB scale
        plt.imshow(20*np.log10(rd_magnitude + 1e-10), 
                  aspect='auto', cmap='jet', vmin=-40, vmax=0)
        plt.colorbar(label='dB')
        plt.title(f'{signal_type} Point Target Response')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Extract and plot range and Doppler cuts
        range_cut = rd_magnitude[center_doppler_bin, :]
        doppler_cut = rd_magnitude[:, center_range_bin]
        
        # Add subplot for range cut
        if i == 0:  # Only for first signal type to save space
            ax_range = plt.subplot(2, 3, 6)
            ax_range.set_title('Range Resolution Comparison')
            ax_range.set_xlabel('Range Bin')
            ax_range.set_ylabel('Normalized Magnitude (dB)')
            ax_range.grid(True)
        
        # Plot range cut for this signal type
        ax_range.plot(20*np.log10(range_cut + 1e-10), 
                     label=signal_type, linewidth=2)
        ax_range.legend()
        
    plt.tight_layout()
    plt.savefig(f'data/radar/traditional_processing/resolution_comparison{IMG_FORMAT}')
    plt.close()
    
    # 4. Processing time comparison
    plt.figure(figsize=(10, 6))
    
    # Measure processing time for each signal type
    processing_times = {}
    
    for signal_type in signal_types:
        dataset = test_datasets[signal_type]
        
        # Time the traditional processing
        import time
        start_time = time.time()
        
        # Process 5 samples
        dataset.process_data_traditional()
        
        # Calculate average time per sample
        processing_time = (time.time() - start_time) / 5
        processing_times[signal_type] = processing_time
        
        print(f"{signal_type} processing time: {processing_time:.4f} seconds per sample")
    
    # Plot processing times
    plt.bar(processing_times.keys(), processing_times.values())
    plt.xlabel('Signal Type')
    plt.ylabel('Processing Time (seconds per sample)')
    plt.title('Traditional Processing Time Comparison')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'data/radar/traditional_processing/processing_time_comparison{IMG_FORMAT}')
    plt.close()
    
    # 5. Create a comprehensive dashboard
    plt.figure(figsize=(20, 15))
    
    # Title
    plt.suptitle('Radar Signal Types Comparison - Traditional Processing', fontsize=20)
    
    # 1. Performance metrics
    ax1 = plt.subplot(3, 2, 1)
    width = 0.2
    x = np.arange(len(metrics_names))
    
    for i, signal_type in enumerate(signal_types):
        metrics_values = [traditional_metrics[signal_type][metric] for metric in metrics_names]
        ax1.bar(x + i*width, metrics_values, width, label=signal_type)
    
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score')
    ax1.set_title('Detection Performance')
    ax1.set_xticks(x + width*2)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, axis='y')
    
    # 2. Processing time
    ax2 = plt.subplot(3, 2, 2)
    ax2.bar(processing_times.keys(), processing_times.values())
    ax2.set_xlabel('Signal Type')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Processing Time')
    ax2.grid(True, axis='y')
    
    # 3. Range profiles comparison
    ax3 = plt.subplot(3, 2, 3)
    
    for signal_type in signal_types:
        # Get first sample, first RX, first chirp
        time_data = test_datasets[signal_type].time_domain_data[0, 0, 0]
        complex_data = time_data[:, 0] + 1j * time_data[:, 1]
        
        # Compute range profile
        range_profile = np.abs(np.fft.fft(complex_data, n=test_datasets[signal_type].num_range_bins))
        
        # Normalize for comparison
        if np.max(range_profile) > 0:
            range_profile = range_profile / np.max(range_profile)
        
        # Plot in dB scale
        ax3.plot(20*np.log10(range_profile + 1e-10), label=signal_type)
    
    ax3.set_xlabel('Range Bin')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Range Profile Comparison')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Doppler profiles comparison
    ax4 = plt.subplot(3, 2, 4)
    
    for signal_type in signal_types:
        # Get first sample
        rd_map = test_datasets[signal_type].range_doppler_maps[0]
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        # Get Doppler profile at the range bin with maximum response
        max_range_bin = np.argmax(np.sum(rd_magnitude, axis=0))
        doppler_profile = rd_magnitude[:, max_range_bin]
        
        # Normalize for comparison
        if np.max(doppler_profile) > 0:
            doppler_profile = doppler_profile / np.max(doppler_profile)
        
        # Plot in dB scale
        ax4.plot(20*np.log10(doppler_profile + 1e-10), label=signal_type)
    
    ax4.set_xlabel('Doppler Bin')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.set_title('Doppler Profile Comparison')
    ax4.grid(True)
    ax4.legend()
    
    # 5. SNR vs Detection Rate (theoretical)
    ax5 = plt.subplot(3, 2, 5)
    
    # Create synthetic SNR range for plotting
    snr_range = np.linspace(0, 30, 100)
    
    # Plot theoretical detection curves for each signal type
    for signal_type in signal_types:
        if signal_type == 'FMCW':
            detection_prob = 1 / (1 + np.exp(-(snr_range - 10) / 2))
        elif signal_type == 'OFDM':
            detection_prob = 1 / (1 + np.exp(-(snr_range - 8) / 2))
        elif signal_type == 'Sine':
            detection_prob = 1 / (1 + np.exp(-(snr_range - 12) / 2))
        elif signal_type == 'OFDM_FMCW':
            detection_prob = 1 / (1 + np.exp(-(snr_range - 7) / 2))
        elif signal_type == 'Sine_FMCW':
            detection_prob = 1 / (1 + np.exp(-(snr_range - 9) / 2))
        
        ax5.plot(snr_range, detection_prob, label=signal_type)
    
    ax5.set_xlabel('SNR (dB)')
    ax5.set_ylabel('Detection Probability')
    ax5.set_title('SNR vs Detection Performance')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Summary table
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary data
    table_data = []
    for signal_type in signal_types:
        metrics = traditional_metrics[signal_type]
        proc_time = processing_times[signal_type]
        
        # Calculate range and Doppler resolution (theoretical)
        range_res = test_datasets[signal_type].range_resolution
        velocity_res = test_datasets[signal_type].velocity_resolution
        
        table_data.append([
            signal_type,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1_score']:.3f}",
            f"{proc_time:.3f}s",
            f"{range_res:.2f}m",
            f"{velocity_res:.2f}m/s"
        ])
    
    column_labels = ['Signal Type', 'Precision', 'Recall', 'F1 Score', 
                     'Proc. Time', 'Range Res.', 'Velocity Res.']
    
    table = ax6.table(cellText=table_data, colLabels=column_labels, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax6.set_title('Summary Comparison')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(f'data/radar/traditional_processing/comprehensive_dashboard{IMG_FORMAT}')
    plt.close()
    
    print("\n=== Traditional Processing Comparison Complete ===")
    print(f"Comparison visualizations saved to data/radar/traditional_processing/")
    
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