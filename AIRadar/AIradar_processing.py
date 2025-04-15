import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.signal import chirp
import cv2

class RadarProcessing:
    """
    Class for radar signal processing functions
    """
    def __init__(self, 
                 num_range_bins=128, #64, 
                 num_doppler_bins=16, #12,
                 sample_rate=15e6,
                 chirp_duration=1e-3, #500e-6,
                 num_chirps=32, #12,
                 num_subcarriers = 64, #OFDM subcarriers
                 subcarrier_spacing = 50e3, #OFDM subcarrier spacing
                 bandwidth=500e6,
                 transceiver_bandwidth=30e6, #AD9361 bandwidth limitation (~56MHz)
                 transceiver_center_freq=2.1e9,
                 output_freq=10e9,  # Added output frequency (CN0566)
                 signal_type='FMCW',
                 signal_freq=1e6):
        """
        Initialize radar processing parameters
        
        Args:
            num_range_bins: Number of range bins that determines range resolution granularity. Higher values provide finer range discrimination at the cost of increased processing time.
            
            num_doppler_bins: Number of Doppler bins that determines velocity resolution granularity. Higher values enable detection of smaller velocity differences but require more processing.
            
            sample_rate: Sample rate in Hz that defines how frequently the signal is sampled. Higher rates capture more signal details but increase data volume and memory usage. Limited by AD9361 capabilities.
            
            chirp_duration: Chirp duration in seconds that affects both maximum unambiguous range and velocity resolution. Longer durations improve SNR and range but reduce maximum detectable velocity.
            
            num_chirps: Number of chirps per frame that directly impacts Doppler resolution and coherent processing gain. More chirps improve velocity resolution and SNR but increase frame time.
            
            num_subcarriers: Number of OFDM subcarriers used when signal_type is 'OFDM' or 'OFDM_FMCW'. Affects frequency diversity and multipath resilience.
            
            subcarrier_spacing: Spacing between OFDM subcarriers in Hz. Impacts OFDM symbol duration and multipath handling capability.
            
            bandwidth: Signal bandwidth in Hz that directly determines range resolution. Higher bandwidth provides finer range resolution (c/2B). Limited by CN0566 capabilities.
            
            transceiver_bandwidth: AD9361 bandwidth limitation in Hz. Constrains the effective signal bandwidth that can be processed after demodulation.
            
            transceiver_center_freq: Center frequency of the AD9361 transceiver in Hz. Used in the two-step demodulation process.
            
            output_freq: CN0566 output frequency in Hz. Determines the wavelength used for velocity calculations and affects maximum unambiguous velocity.
            
            signal_type: Type of radar signal ('FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW'). Determines the processing pipeline and signal generation method.
            
            signal_freq: Signal frequency for modulation in Hz. Used primarily for sine wave modulation and filtering.
        """
        #Range-Doppler map with shape [num_doppler_bins, num_range_bins]
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.sample_rate = sample_rate
        self.chirp_duration = chirp_duration
        self.num_chirps = num_chirps
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing #OFDM subcarrier spacing
        self.bandwidth = bandwidth
        self.transceiver_center_freq = transceiver_center_freq
        self.output_freq = output_freq  # CN0566 output frequency
        self.signal_type = signal_type
        self.signal_freq = signal_freq
        
        # Calculate derived parameters
        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        self.range_resolution = self._calculate_range_resolution()
        self.velocity_resolution = self._calculate_velocity_resolution()
        self.max_range = self._calculate_max_range()
        self.max_velocity = self._calculate_max_velocity()
        
        # Hardware-specific parameters
        self.ad9361_bandwidth = transceiver_bandwidth  # AD9361 bandwidth limitation (~56MHz)
        self.cn0566_center_freq = output_freq  # CN0566 center frequency (10GHz)
    

    def _calculate_range_resolution(self):
        """Calculate range resolution based on bandwidth"""
        # Range resolution = c / (2 * bandwidth)
        speed_of_light = 3e8  # m/s
        return speed_of_light / (2 * self.bandwidth)
    
    def _calculate_velocity_resolution(self):
        """Calculate velocity resolution based on parameters"""
        # Velocity resolution = wavelength / (2 * total_observation_time)
        speed_of_light = 3e8  # m/s
        wavelength = speed_of_light / self.output_freq
        total_observation_time = self.num_chirps * self.chirp_duration
        return wavelength / (2 * total_observation_time)
    
    def _calculate_max_range(self):
        """Calculate maximum unambiguous range"""
        # Calculate unambiguous range based on pulse repetition frequency (PRF)
        # For FMCW radar, PRF is 1/chirp_duration
        speed_of_light = 3e8  # m/s
        prf = 1.0 / self.chirp_duration  # Pulse repetition frequency
        unambiguous_range = speed_of_light / (2 * prf)
        
        # Calculate maximum range based on sampling parameters
        # Max range = (sample_rate * c) / (2 * bandwidth)
        sampling_max_range = (self.sample_rate * speed_of_light) / (2 * self.bandwidth)
        
        # Take the minimum of the two constraints
        theoretical_max_range = min(unambiguous_range, sampling_max_range)
        
        # Apply practical constraints based on signal power
        # Radar equation: R_max^4 = (P_t * G^2 * λ^2 * σ) / ((4π)^3 * S_min)
        # We'll use a simplified version with reasonable defaults for radar parameters
        wavelength = speed_of_light / self.output_freq
        
        # Typical values for a short-range radar system
        tx_power_watts = 0.1  # 100 mW transmit power
        antenna_gain = 10.0   # Linear gain (10 dB)
        min_detectable_power = 1e-12  # -90 dBm minimum detectable signal
        radar_cross_section = 1.0  # 1 m² (typical for a small vehicle)
        
        # Calculate power-limited max range
        power_max_range = ((tx_power_watts * antenna_gain**2 * wavelength**2 * radar_cross_section) / 
                          ((4 * np.pi)**3 * min_detectable_power))**(1/4)
        
        # Take the minimum of all constraints
        practical_max_range = min(theoretical_max_range, power_max_range, 150.0)
        
        if theoretical_max_range > 500 or power_max_range > 500:
            print(f"Warning: Calculated max ranges exceed practical limits:")
            print(f"  - Unambiguous range: {unambiguous_range:.2f}m")
            print(f"  - Sampling max range: {sampling_max_range:.2f}m")
            print(f"  - Power-limited max range: {power_max_range:.2f}m")
            print(f"Setting max range to {practical_max_range:.2f}m")
            
        return practical_max_range
    
    def _calculate_max_velocity(self):
        """Calculate maximum unambiguous velocity"""
        # Max velocity = wavelength / (4 * chirp_duration)
        speed_of_light = 3e8  # m/s
        wavelength = speed_of_light / self.output_freq
        return wavelength / (4 * self.chirp_duration)
    
    def simulate_hardware_demodulation(self, complex_data):
        """
        Simulate the hardware demodulation process from CN0566 to AD9361
        
        This simulates the CN0566 to AD9361 demodulation that happens in hardware
        before digital processing.
        
        Args:
            complex_data: Complex time domain data [num_rx, num_chirps, samples_per_chirp]
            
        Returns:
            Demodulated complex data [num_rx, num_chirps, samples_per_chirp]
        """
        # Create time vector
        t = np.arange(self.samples_per_chirp) / self.sample_rate
        
        # Generate the FMCW sweep component that was used in CN0566
        k = self.bandwidth / self.chirp_duration  # Chirp rate for 500MHz bandwidth
        
        # For each RX and chirp, demodulate the signal
        demodulated_data = np.zeros_like(complex_data)
        
        for rx in range(complex_data.shape[0]):
            for chirp in range(complex_data.shape[1]):
                # Generate the conjugate of the sweep signal used in CN0566
                # This simulates the downconversion/mixing process
                phase = 2 * np.pi * (self.cn0566_center_freq * t + 0.5 * k * t**2)
                fmcw_sweep_conj = np.exp(-1j * phase)  # Conjugate for demodulation
                
                # Demodulate by multiplying with conjugate of the sweep
                # This brings the signal from 10GHz back to 2.1GHz (AD9361 frequency)
                demodulated_data[rx, chirp] = complex_data[rx, chirp] * fmcw_sweep_conj
                
                # Apply low-pass filtering to simulate hardware filtering
                # Using a simple moving average filter for demonstration
                window_size = 5
                demodulated_data[rx, chirp] = np.convolve(
                    demodulated_data[rx, chirp], 
                    np.ones(window_size)/window_size, 
                    mode='same'
                )
        
        return demodulated_data

    def time_to_range_doppler(self, complex_data):
        """
        Convert time-domain data to range-Doppler map
        
        Args:
            complex_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
            
        Returns:
            Range-Doppler map with shape [num_doppler_bins, num_range_bins]
        """
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp = complex_data.shape
        
        # Initialize range-Doppler map
        rd_map = np.zeros((self.num_doppler_bins, self.num_range_bins), dtype=np.complex64)
        
        # Check if we need to use the two-step hardware processing
        if self.signal_type in ['OFDM_FMCW', 'Sine_FMCW']:
            # Two-step hardware-specific processing for combined signals
            
            # Step 1: Simulate hardware demodulation (CN0566 to AD9361)
            # This step brings the signal from 10GHz back to 2.1GHz
            demodulated_data = self.simulate_hardware_demodulation(complex_data)
            
            # Process the demodulated data based on signal type
            if self.signal_type == 'OFDM_FMCW':
                # Process similar to FMCW but with additional filtering
                # 1. Range FFT for each chirp
                range_profiles = np.zeros((num_rx, num_chirps, self.num_range_bins), dtype=np.complex64)
                
                for rx in range(num_rx):
                    for chirp in range(num_chirps):
                        # Apply bandpass filter to isolate the OFDM signal
                        filtered_data = self._apply_ofdm_filter(demodulated_data[rx, chirp])
                        
                        # Apply window and FFT
                        window = np.hamming(samples_per_chirp)
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
                
            elif self.signal_type == 'Sine_FMCW':
                # Process similar to FMCW but with sine filtering
                # 1. Range FFT for each chirp
                range_profiles = np.zeros((num_rx, num_chirps, self.num_range_bins), dtype=np.complex64)
                
                for rx in range(num_rx):
                    for chirp in range(num_chirps):
                        # Apply bandpass filter around sine frequency
                        filtered_data = self._apply_sine_filter(demodulated_data[rx, chirp])
                        
                        # Apply window and FFT
                        window = np.hamming(samples_per_chirp)
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
        else:
            # Standard processing for normal signal types (FMCW, OFDM, Sine)
            # without hardware-specific demodulation
            
            if self.signal_type == 'FMCW':
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
                
            elif self.signal_type == 'OFDM':
                # OFDM radar processing
                # For OFDM, we need to extract the OFDM signal from the data
                
                # Assuming OFDM parameters
                #num_subcarriers = 64
                
                # Reshape data to match OFDM structure
                ofdm_data = np.zeros((num_rx, num_chirps, self.num_subcarriers), dtype=np.complex64)
                
                for rx in range(num_rx):
                    for chirp in range(num_chirps):
                        # Extract OFDM symbol from data
                        ofdm_symbol = complex_data[rx, chirp]
                        
                        # Remove cyclic prefix (assuming 25% CP)
                        cp_length = samples_per_chirp // 4
                        symbol_data = ofdm_symbol[cp_length:]
                        
                        # FFT to get frequency domain
                        freq_data = np.fft.fft(symbol_data, n=self.num_subcarriers)
                        ofdm_data[rx, chirp] = freq_data
        
        # Apply CFAR detection to enhance targets
        rd_map_magnitude = np.abs(rd_map)
        rd_map_cfar = self._apply_cfar(rd_map_magnitude)
        
        # Convert back to complex
        rd_map_normalized = rd_map_cfar * np.exp(1j * np.angle(rd_map))
        
        return rd_map_normalized
    
    def time_to_range_doppler_batch(self, time_domain_data):
        """
        Process a batch of time domain data to range-Doppler maps
        
        Args:
            time_domain_data: Time domain data with shape [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
                            where the last dimension contains I/Q data
                            
        Returns:
            Range-Doppler maps with shape [batch_size, 2, num_doppler_bins, num_range_bins]
                            where the second dimension contains real/imaginary parts
        """
        batch_size = time_domain_data.shape[0]
        
        # Initialize output array
        rd_maps = np.zeros((batch_size, 2, self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Convert I/Q format to complex
            complex_data = time_domain_data[i, :, :, :, 0] + 1j * time_domain_data[i, :, :, :, 1]
            
            # Process using time_to_range_doppler
            rd_map = self.time_to_range_doppler(complex_data)
            
            # Store real and imaginary parts
            rd_maps[i, 0, :, :] = np.real(rd_map)
            rd_maps[i, 1, :, :] = np.imag(rd_map)
        
        return rd_maps
    
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
        #num_subcarriers = 64
        subcarrier_spacing = self.subcarrier_spacing #50e3  # 50 kHz
        
        # FFT to frequency domain
        signal_fft = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/self.sample_rate))
        
        # Create filter mask for OFDM subcarriers
        mask = np.zeros_like(freq, dtype=bool)
        
        # Mark subcarrier regions
        for k in range(self.num_subcarriers):
            # Subcarrier frequency
            f_k = (k - self.num_subcarriers//2) * subcarrier_spacing
            
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
    
    def apply_realistic_rf_effects(self, time_domain_data, target_info=None):
        """
        Apply realistic RF impairments to time domain data
        
        Args:
            time_domain_data: Time domain data with shape [num_rx, num_chirps, samples_per_chirp, 2]
                            where the last dimension contains I/Q data
            target_info: Optional target information for realistic effects
            
        Returns:
            Modified time domain data with realistic impairments
        """
        # Make a copy to avoid modifying the original data
        modified_data = time_domain_data.copy()
        
        # Convert I/Q data to complex
        complex_data = modified_data[..., 0] + 1j * modified_data[..., 1]
        
        # 1. Add I/Q imbalance
        # Real-world systems have amplitude and phase imbalance between I and Q
        iq_amplitude_imbalance = np.random.uniform(0.9, 1.1)  # 10% amplitude imbalance
        iq_phase_imbalance = np.random.uniform(-np.pi/36, np.pi/36)  # ±5 degrees phase imbalance
        
        # Apply I/Q imbalance
        i_component = np.real(complex_data)
        q_component = np.imag(complex_data) * iq_amplitude_imbalance * np.exp(1j * iq_phase_imbalance)
        complex_data = i_component + 1j * q_component
        
        # 2. Add DC offset
        # Real-world systems have DC offset in I and Q channels
        dc_offset_i = np.random.uniform(-0.05, 0.05)  # 5% DC offset
        dc_offset_q = np.random.uniform(-0.05, 0.05)  # 5% DC offset
        
        # Apply DC offset
        complex_data = complex_data + (dc_offset_i + 1j * dc_offset_q)
        
        # 3. Add phase noise
        # Phase noise is a common impairment in radar systems
        phase_noise_level = np.random.uniform(0.01, 0.05)  # Phase noise level
        
        # Generate phase noise
        phase_noise = np.random.normal(0, phase_noise_level, complex_data.shape)
        
        # Apply phase noise
        complex_data = complex_data * np.exp(1j * phase_noise)
        
        # 4. Add frequency drift
        # Frequency drift due to oscillator instability
        freq_drift_rate = np.random.uniform(-0.01, 0.01)  # Frequency drift rate
        
        # Generate time vector
        t = np.arange(complex_data.shape[2]) / self.sample_rate
        
        # Apply frequency drift to each chirp
        for rx_idx in range(complex_data.shape[0]):
            for chirp_idx in range(complex_data.shape[1]):
                # Calculate frequency drift for this chirp
                chirp_time = chirp_idx * self.chirp_duration
                freq_drift = freq_drift_rate * chirp_time
                
                # Apply frequency drift
                complex_data[rx_idx, chirp_idx] *= np.exp(1j * 2 * np.pi * freq_drift * t)
        
        # 5. Add gain variations between RX channels
        # Real-world systems have gain variations between RX channels
        if complex_data.shape[0] > 1:  # If we have multiple RX channels
            for rx_idx in range(1, complex_data.shape[0]):
                # Generate random gain variation
                gain_variation = np.random.uniform(0.8, 1.2)  # 20% gain variation
                
                # Apply gain variation
                complex_data[rx_idx] *= gain_variation
        
        # 6. Add non-linearity effects
        # Real-world systems have non-linear amplifiers
        # Simulate non-linearity with a simple polynomial model
        nonlinearity_level = np.random.uniform(0.01, 0.1)  # Non-linearity level
        
        # Apply non-linearity
        complex_data = complex_data * (1 + nonlinearity_level * np.abs(complex_data)**2)
        
        # 7. Add thermal noise
        # Thermal noise is always present in real-world systems
        # Calculate thermal noise power using Boltzmann's constant
        boltzmann_constant = 1.38e-23  # J/K
        temperature_kelvin = 290  # Room temperature in Kelvin
        
        # Calculate thermal noise power
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
    
    def process_data_traditional(self, time_domain_data, signal_type=None):
        """
        Process radar data using traditional signal processing for all samples
        
        Args:
            time_domain_data: Time domain data to process with shape [num_samples, num_rx, num_chirps, samples_per_chirp, 2]
            signal_type: Optional signal type to override the default
            
        Returns:
            Processed range-Doppler maps and detection masks
        """
        if signal_type is None:
            signal_type = self.signal_type
            
        num_samples = time_domain_data.shape[0]
        
        # Initialize output arrays
        rd_maps = np.zeros((num_samples, 2, self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        detection_masks = np.zeros((num_samples, self.num_doppler_bins, self.num_range_bins, 1), dtype=np.float32)
        
        print(f"Processing {num_samples} samples using traditional radar processing...")
        
        for i in tqdm(range(num_samples), desc="Traditional processing"):
            # Process this sample
            rd_map = self.time_to_range_doppler(
                time_domain_data[i, :, :, :, 0] + 1j * time_domain_data[i, :, :, :, 1]
            )
            
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

    def detect_targets(self, rd_map, threshold=0.15, min_area=2):
        """
        Detect targets in a range-Doppler map using thresholding and clustering
        
        Args:
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins] or [num_doppler_bins, num_range_bins]
            threshold: Detection threshold (normalized)
            min_area: Minimum area (in cells) for a valid target
            
        Returns:
            List of detected targets with properties
        """
        # Check if rd_map has real/imaginary components
        if len(rd_map.shape) == 3 and rd_map.shape[0] == 2:
            # Convert to magnitude
            rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        else:
            # Already in magnitude format or complex
            rd_magnitude = np.abs(rd_map)
        
        # Normalize magnitude to [0, 1]
        if np.max(rd_magnitude) > 0:
            rd_magnitude = rd_magnitude / np.max(rd_magnitude)
        
        # Apply threshold for detection
        detection = (rd_magnitude > threshold).astype(np.uint8)
        
        # Find connected components (clusters)
        # Use connectedComponents instead since connectedComponentsWithStats is not available
        num_labels, labels = cv2.connectedComponents(detection)
        
        # Calculate stats and centroids manually
        stats = []
        centroids = []
        
        for label in range(num_labels):
            # Get mask for current label
            mask = (labels == label).astype(np.uint8)
            
            # Calculate stats
            area = np.sum(mask)
            indices = np.where(mask)
            x_min = np.min(indices[1]) if len(indices[1]) > 0 else 0
            y_min = np.min(indices[0]) if len(indices[0]) > 0 else 0
            width = np.max(indices[1]) - x_min + 1 if len(indices[1]) > 0 else 0
            height = np.max(indices[0]) - y_min + 1 if len(indices[0]) > 0 else 0
            
            stats.append([x_min, y_min, width, height, area])
            
            # Calculate centroid
            if area > 0:
                centroid_y = np.mean(indices[0])
                centroid_x = np.mean(indices[1])
            else:
                centroid_y = centroid_x = 0
                
            centroids.append([centroid_x, centroid_y])
            
        stats = np.array(stats)
        centroids = np.array(centroids)
        
        # Initialize list to store detected targets
        targets = []
        
        # Process each detected component (skip background label 0)
        for i in range(1, num_labels):
            # Get component statistics
            area = stats[i, 4]  # Area is stored in the 5th column (index 4) of stats array
            
            # Filter small components
            if area >= min_area:
                # Get centroid
                doppler_bin, range_bin = centroids[i]
                
                # Convert to physical units
                range_m = range_bin * self.range_resolution
                
                # Convert Doppler bin to velocity (centered at 0)
                doppler_bin_centered = doppler_bin - self.num_doppler_bins / 2
                velocity = doppler_bin_centered * self.velocity_resolution
                
                # Get peak value
                mask = (labels == i)
                peak_value = np.max(rd_magnitude[mask])
                
                # Store target information
                target = {
                    'range_bin': range_bin,
                    'doppler_bin': doppler_bin,
                    'range_m': range_m,
                    'velocity': velocity,
                    'magnitude': peak_value,
                    'area': area
                }
                
                targets.append(target)
        
        return targets
    
    def estimate_angle(self, complex_data, target_range_bin, target_doppler_bin, method='music'):
        """
        Estimate angle of arrival for a target using antenna array processing
        
        Args:
            complex_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
            target_range_bin: Range bin of the target
            target_doppler_bin: Doppler bin of the target
            method: Angle estimation method ('music', 'mvdr', 'bartlett')
            
        Returns:
            Estimated angle in degrees
        """
        # Get dimensions
        num_rx = complex_data.shape[0]
        
        # Check if we have enough antennas for angle estimation
        if num_rx < 2:
            print("Warning: At least 2 RX antennas are needed for angle estimation")
            return 0.0
        
        # Get range profiles for each chirp and RX
        range_profiles = np.zeros((num_rx, self.num_chirps, self.num_range_bins), dtype=np.complex64)
        
        for rx in range(num_rx):
            for chirp in range(self.num_chirps):
                # Apply window function
                window = np.hamming(self.samples_per_chirp)
                windowed_data = complex_data[rx, chirp] * window
                
                # Perform range FFT
                range_fft = np.fft.fft(windowed_data, n=self.num_range_bins)
                range_profiles[rx, chirp] = range_fft
        
        # Extract the target's range bin data across all RX and chirps
        target_data = range_profiles[:, :, target_range_bin]
        
        # Apply Doppler processing to get the target's Doppler bin
        doppler_bin_centered = target_doppler_bin - self.num_doppler_bins // 2
        
        # If Doppler bin is provided, extract the specific Doppler component
        if target_doppler_bin is not None:
            # Apply Doppler FFT for each RX
            doppler_data = np.zeros(num_rx, dtype=np.complex64)
            
            for rx in range(num_rx):
                # Apply window function
                doppler_window = np.hamming(self.num_chirps)
                windowed_doppler = target_data[rx] * doppler_window
                
                # Perform Doppler FFT
                doppler_fft = np.fft.fftshift(np.fft.fft(windowed_doppler, n=self.num_doppler_bins))
                
                # Extract the target's Doppler bin
                doppler_data[rx] = doppler_fft[target_doppler_bin]
        else:
            # If no Doppler bin is provided, use the average across chirps
            doppler_data = np.mean(target_data, axis=1)
        
        # Calculate wavelength
        wavelength = 3e8 / self.output_freq
        
        # Calculate antenna spacing (typically λ/2)
        d = wavelength / 2
        
        # Define angle grid for scanning
        angle_grid = np.linspace(-90, 90, 181)  # 1-degree resolution
        
        # Convert to radians
        angle_rad = np.deg2rad(angle_grid)
        
        # Calculate steering vectors for each angle
        steering_vectors = np.exp(-1j * 2 * np.pi * d * np.outer(np.arange(num_rx), np.sin(angle_rad)) / wavelength)
        
        # Estimate angle based on selected method
        if method == 'music':
            # MUSIC algorithm
            # Calculate covariance matrix
            R = np.outer(doppler_data, np.conj(doppler_data))
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(R)
            
            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Noise subspace (assuming 1 signal)
            noise_eigenvectors = eigenvectors[:, :-1]
            
            # MUSIC spectrum
            spectrum = np.zeros(len(angle_grid))
            
            for i, a in enumerate(angle_grid):
                v = steering_vectors[:, i]
                spectrum[i] = 1 / np.abs(np.dot(np.dot(v.conj(), noise_eigenvectors), np.dot(noise_eigenvectors.conj().T, v)))
            
            # Find peak
            peak_idx = np.argmax(spectrum)
            estimated_angle = angle_grid[peak_idx]
            
        elif method == 'mvdr':
            # MVDR (Capon) beamformer
            # Calculate covariance matrix
            R = np.outer(doppler_data, np.conj(doppler_data))
            
            # Add small diagonal loading for stability
            R = R + np.eye(num_rx) * 1e-6
            
            # Calculate inverse
            R_inv = np.linalg.inv(R)
            
            # MVDR spectrum
            spectrum = np.zeros(len(angle_grid))
            
            for i, a in enumerate(angle_grid):
                v = steering_vectors[:, i]
                spectrum[i] = 1 / np.abs(np.dot(np.dot(v.conj(), R_inv), v))
            
            # Find peak
            peak_idx = np.argmax(spectrum)
            estimated_angle = angle_grid[peak_idx]
            
        else:  # Default to Bartlett
            # Conventional (Bartlett) beamformer
            spectrum = np.zeros(len(angle_grid))
            
            for i, a in enumerate(angle_grid):
                v = steering_vectors[:, i]
                spectrum[i] = np.abs(np.dot(v.conj(), doppler_data))**2
            
            # Find peak
            peak_idx = np.argmax(spectrum)
            estimated_angle = angle_grid[peak_idx]
        
        return estimated_angle
    
    def apply_doa_processing(self, complex_data, rd_map):
        """
        Apply Direction of Arrival (DOA) processing to estimate angles for all detected targets
        
        Args:
            complex_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins] or [num_doppler_bins, num_range_bins]
            
        Returns:
            List of targets with angle information
        """
        # Detect targets in the range-Doppler map
        targets = self.detect_targets(rd_map)
        
        # For each target, estimate angle
        for target in targets:
            range_bin = int(target['range_bin'])
            doppler_bin = int(target['doppler_bin'])
            
            # Estimate angle
            angle = self.estimate_angle(complex_data, range_bin, doppler_bin)
            
            # Add angle to target information
            target['angle_deg'] = angle
        
        return targets
    
    def generate_point_cloud(self, targets):
        """
        Generate a 3D point cloud from target information
        
        Args:
            targets: List of targets with range, angle, and velocity information
            
        Returns:
            Point cloud as numpy array with shape [num_points, 3] (x, y, z coordinates)
        """
        # Initialize point cloud
        num_targets = len(targets)
        point_cloud = np.zeros((num_targets, 3))
        
        # For each target, calculate 3D position
        for i, target in enumerate(targets):
            # Extract target information
            range_m = target['range_m']
            angle_deg = target.get('angle_deg', 0.0)  # Default to 0 if not available
            
            # Convert angle to radians
            angle_rad = np.deg2rad(angle_deg)
            
            # Calculate 3D position (assuming radar at origin)
            # x: forward, y: right, z: up
            x = range_m * np.cos(angle_rad)
            y = range_m * np.sin(angle_rad)
            z = 0.0  # Assuming targets are at the same height as radar
            
            # Store in point cloud
            point_cloud[i] = [x, y, z]
        
        return point_cloud
    
    def visualize_range_doppler(self, rd_map, targets=None, title='Range-Doppler Map'):
        """
        Visualize range-Doppler map with detected targets
        
        Args:
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins] or [num_doppler_bins, num_range_bins]
            targets: Optional list of detected targets
            title: Plot title
        """
        # Check if rd_map has real/imaginary components
        if len(rd_map.shape) == 3 and rd_map.shape[0] == 2:
            # Convert to magnitude
            rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        else:
            # Already in magnitude format or complex
            rd_magnitude = np.abs(rd_map)
        
        # Convert to dB
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot range-Doppler map
        plt.imshow(rd_db, aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(title)
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Add velocity and range axes
        ax = plt.gca()
        
        # Add secondary x-axis for range in meters
        ax_range = ax.secondary_xaxis('top', functions=(
            lambda x: x * self.range_resolution,  # bin to meters
            lambda x: x / self.range_resolution   # meters to bin
        ))
        ax_range.set_xlabel('Range (m)')
        
        # Add secondary y-axis for velocity in m/s
        ax_velocity = ax.secondary_yaxis('right', functions=(
            lambda y: (y - self.num_doppler_bins/2) * self.velocity_resolution,  # bin to m/s
            lambda y: y / self.velocity_resolution + self.num_doppler_bins/2     # m/s to bin
        ))
        ax_velocity.set_ylabel('Velocity (m/s)')
        
        # Plot detected targets if provided
        if targets is not None:
            for target in targets:
                plt.plot(target['range_bin'], target['doppler_bin'], 'ro', markersize=10)
                
                # Add target information
                plt.text(target['range_bin'] + 1, target['doppler_bin'] + 1, 
                        f"R: {target['range_m']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                        color='white', fontsize=8, backgroundcolor='black')
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
    
    def visualize_point_cloud(self, point_cloud, velocities=None, title='Radar Point Cloud'):
        """
        Visualize 3D point cloud
        
        Args:
            point_cloud: Point cloud as numpy array with shape [num_points, 3]
            velocities: Optional array of velocities for coloring
            title: Plot title
        """
        # Create 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        
        # Plot points
        if velocities is not None:
            # Color by velocity
            scatter = ax.scatter(x, y, z, c=velocities, cmap='jet', s=50)
            plt.colorbar(scatter, label='Velocity (m/s)')
        else:
            # Single color
            ax.scatter(x, y, z, c='b', s=50)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.max([np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))])
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Add grid
        ax.grid(True)
        
        # Add radar position
        ax.scatter(0, 0, 0, c='r', s=100, marker='^')
        ax.text(0, 0, 0, 'Radar', color='r')
        
        plt.tight_layout()
    
    def track_targets(self, targets_history, max_age=5, max_distance=2.0):
        """
        Simple multi-target tracking using Global Nearest Neighbor (GNN) association
        
        Args:
            targets_history: List of target lists from consecutive frames
            max_age: Maximum age of a track before deletion
            max_distance: Maximum distance for association
            
        Returns:
            List of tracks with IDs and states
        """
        # Initialize tracks
        tracks = []
        next_id = 0
        
        # Process each frame
        for frame_idx, frame_targets in enumerate(targets_history):
            # Predict existing tracks to current frame
            for track in tracks:
                if track['active']:
                    # Simple constant velocity prediction
                    dt = 1.0  # Assuming constant time between frames
                    
                    # Update predicted position
                    track['predicted_range'] = track['range'] + track['velocity'] * dt
                    track['predicted_angle'] = track['angle']  # Assuming constant angle
                    
                    # Increment age
                    track['age'] += 1
            
            # Create cost matrix for assignment
            cost_matrix = np.zeros((len(tracks), len(frame_targets)))
            
            for i, track in enumerate(tracks):
                if not track['active']:
                    cost_matrix[i, :] = float('inf')
                    continue
                    
                for j, target in enumerate(frame_targets):
                    # Calculate distance between track prediction and target
                    range_diff = abs(track['predicted_range'] - target['range_m'])
                    angle_diff = abs(track['predicted_angle'] - target.get('angle_deg', 0.0))
                    
                    # Normalize angle difference
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    
                    # Calculate total distance (weighted sum)
                    distance = range_diff + 0.1 * angle_diff
                    
                    if distance > max_distance:
                        cost_matrix[i, j] = float('inf')
                    else:
                        cost_matrix[i, j] = distance
            
            # Solve assignment problem
            if len(tracks) > 0 and len(frame_targets) > 0:
                try:
                    from scipy.optimize import linear_sum_assignment
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    # Update assigned tracks
                    for i, j in zip(row_ind, col_ind):
                        if cost_matrix[i, j] < float('inf'):
                            track = tracks[i]
                            target = frame_targets[j]
                            
                            # Update track with new measurement
                            track['range'] = target['range_m']
                            track['velocity'] = target['velocity']
                            track['angle'] = target.get('angle_deg', 0.0)
                            track['last_update'] = frame_idx
                            track['age'] = 0  # Reset age
                            
                            # Mark target as assigned
                            target['assigned'] = True
                except ImportError:
                    print("Warning: scipy.optimize not available, using simple association")
                    # Simple greedy association
                    for i, track in enumerate(tracks):
                        if not track['active']:
                            continue
                            
                        # Find closest target
                        min_dist = float('inf')
                        best_target = None
                        best_idx = -1
                        
                        for j, target in enumerate(frame_targets):
                            if target.get('assigned', False):
                                continue
                                
                            if cost_matrix[i, j] < min_dist:
                                min_dist = cost_matrix[i, j]
                                best_target = target
                                best_idx = j
                        
                        # Update track if a match is found
                        if best_target is not None:
                            # Update track with new measurement
                            track['range'] = best_target['range_m']
                            track['velocity'] = best_target['velocity']
                            track['angle'] = best_target.get('angle_deg', 0.0)
                            track['last_update'] = frame_idx
                            track['age'] = 0  # Reset age
                            
                            # Mark target as assigned
                            frame_targets[best_idx]['assigned'] = True
            
            # Create new tracks for unassigned targets
            for target in frame_targets:
                if not target.get('assigned', False):
                    # Create new track
                    new_track = {
                        'id': next_id,
                        'range': target['range_m'],
                        'velocity': target['velocity'],
                        'angle': target.get('angle_deg', 0.0),
                        'predicted_range': target['range_m'],
                        'predicted_angle': target.get('angle_deg', 0.0),
                        'first_seen': frame_idx,
                        'last_update': frame_idx,
                        'age': 0,
                        'active': True
                    }
                    
                    tracks.append(new_track)
                    next_id += 1
            
            # Remove old tracks
            for track in tracks:
                if track['age'] > max_age:
                    track['active'] = False
        
        # Return only active tracks
        return [track for track in tracks if track['active']]
    
    def visualize_tracks(self, tracks, title='Target Tracks'):
        """
        Visualize target tracks in a 2D plot
        
        Args:
            tracks: List of tracks with position and velocity information
            title: Plot title
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot each track
        for track in tracks:
            # Convert polar to Cartesian coordinates
            range_m = track['range']
            angle_rad = np.deg2rad(track['angle'])
            
            x = range_m * np.cos(angle_rad)
            y = range_m * np.sin(angle_rad)
            
            # Plot position
            plt.scatter(x, y, s=50, label=f"ID: {track['id']}")
            
            # Plot velocity vector
            velocity = track['velocity']
            vx = velocity * np.cos(angle_rad)
            vy = velocity * np.sin(angle_rad)
            
            plt.arrow(x, y, vx, vy, width=0.1, head_width=0.3, head_length=0.5, fc='red', ec='red')
            
            # Add track ID
            plt.text(x, y, str(track['id']), fontsize=12)
        
        # Set labels and title
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(title)
        
        # Add radar position
        plt.scatter(0, 0, s=100, c='r', marker='^')
        plt.text(0, 0, 'Radar', color='r')
        
        # Set equal aspect ratio
        plt.axis('equal')
        
        # Add grid
        plt.grid(True)
        
        plt.tight_layout()