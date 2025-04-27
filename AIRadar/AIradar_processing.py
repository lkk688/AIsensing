import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.signal import chirp
import cv2

class RadarProcessing:
    def __init__(self, num_range_bins=64, num_doppler_bins=12, sample_rate=3e6, 
                 chirp_duration=500e-6, num_chirps=1, bandwidth=500e6, center_freq=2.1e9,
                 signal_type='FMCW', num_subcarriers=128, subcarrier_spacing=30e3,
                 transceiver_bandwidth=None, transceiver_center_freq=None, output_freq=None,
                 signal_freq=1e6):
        """
        Initialize radar signal processing class
        
        Args:
            num_range_bins: Number of range bins
            num_doppler_bins: Number of Doppler bins
            sample_rate: Sample rate in Hz
            chirp_duration: Chirp duration in seconds
            num_chirps: Number of chirps per frame
            bandwidth: Signal bandwidth in Hz
            center_freq: Center frequency in Hz
            signal_type: Type of radar signal ('FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW')
            num_subcarriers: Number of subcarriers for OFDM
            subcarrier_spacing: Subcarrier spacing for OFDM in Hz
            transceiver_bandwidth: Transceiver bandwidth in Hz (if different from signal bandwidth)
            transceiver_center_freq: Transceiver center frequency in Hz (if different from center_freq)
            output_freq: Output frequency in Hz (for hardware implementation)
            signal_freq: Signal frequency for FMCW modulation (Hz)
        """
        # Store parameters
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.sample_rate = sample_rate
        self.chirp_duration = chirp_duration
        self.num_chirps = num_chirps
        self.bandwidth = bandwidth
        self.center_freq = center_freq
        self.signal_type = signal_type
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.transceiver_bandwidth = transceiver_bandwidth if transceiver_bandwidth else bandwidth
        self.transceiver_center_freq = transceiver_center_freq if transceiver_center_freq else center_freq
        self.output_freq = output_freq if output_freq else center_freq
        self.signal_freq = signal_freq
        
        # Calculate derived parameters
        self.speed_of_light = 3e8  # Speed of light in m/s
        self.wavelength = self.speed_of_light / self.center_freq
        
        # Calculate radar performance metrics
        self.calculate_performance_metrics()
        
        # # Calculate derived parameters
        # self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
        # self.range_resolution = self._calculate_range_resolution()
        # self.velocity_resolution = self._calculate_velocity_resolution()
        # self.max_range = self._calculate_max_range()
        # self.max_velocity = self._calculate_max_velocity()
        
        # # Hardware-specific parameters
        # self.transceiver_bandwidth = transceiver_bandwidth  # AD9361 bandwidth limitation (~56MHz)
        # self.output_freq = output_freq  # CN0566 center frequency (10GHz)
    
    def calculate_performance_metrics(self):
        """
        Calculate radar performance metrics based on signal parameters
        
        This method calculates key radar performance metrics including:
        - Range resolution
        - Maximum range
        - Velocity resolution
        - Maximum velocity
        - Angular resolution (if applicable)
        - Chirp slope (for FMCW)
        - Optimal chirp duration (based on bandwidth and center frequency)
        
        The relationships between parameters and metrics are documented for each calculation.
        """
        # Range resolution (determined by bandwidth)
        # Higher bandwidth provides better range resolution
        self.range_resolution = self.speed_of_light / (2 * self.bandwidth)
        
        # Maximum unambiguous range (determined by chirp duration and sample rate)
        # Longer chirp duration or higher sample rate increases maximum range
        self.max_range = self.range_resolution * self.num_range_bins
        
        # Velocity resolution (determined by center frequency, chirp duration, and number of chirps)
        # Higher center frequency, longer total measurement time (chirp_duration * num_chirps) improves velocity resolution
        self.velocity_resolution = self.wavelength / (2 * self.chirp_duration * self.num_chirps)
        
        # Maximum unambiguous velocity (determined by center frequency and chirp duration)
        # Higher PRF (1/chirp_duration) increases maximum velocity
        self.max_velocity = self.velocity_resolution * (self.num_doppler_bins // 2)
        
        # Calculate chirp slope for FMCW (bandwidth / chirp_duration)
        # Steeper slope improves range resolution but requires higher sampling rate
        if self.signal_type in ['FMCW', 'OFDM_FMCW', 'Sine_FMCW']:
            self.chirp_slope = self.bandwidth / self.chirp_duration
        else:
            self.chirp_slope = None
            
        # Calculate optimal chirp duration based on bandwidth and center frequency
        # This is a theoretical optimal value that balances range and velocity resolution
        # For automotive radar applications, typical values range from 10-100 μs
        self.optimal_chirp_duration = self.calculate_optimal_chirp_duration()
        
        # Calculate SNR requirements for detection at maximum range
        # This is a simplified model based on the radar equation
        self.min_snr_for_max_range = self.calculate_min_snr_for_detection()
        
        # For OFDM signals, calculate additional metrics
        if self.signal_type in ['OFDM', 'OFDM_FMCW']:
            # OFDM symbol duration (without cyclic prefix)
            self.ofdm_symbol_duration = 1 / self.subcarrier_spacing
            # OFDM bandwidth
            self.ofdm_bandwidth = self.num_subcarriers * self.subcarrier_spacing
        
    def calculate_optimal_chirp_duration(self):
        """
        Calculate the optimal chirp duration based on bandwidth and center frequency
        
        The optimal chirp duration balances:
        1. Range resolution (requires high bandwidth)
        2. Velocity resolution (requires longer measurement time)
        3. Hardware limitations (maximum chirp slope)
        
        Returns:
            Optimal chirp duration in seconds
        """
        # For automotive radar (77 GHz), typical chirp durations are 50-100 μs
        # For lower frequency radars, longer chirp durations are often used
        
        # This is a simplified model that scales chirp duration based on bandwidth and frequency
        # Higher bandwidth requires shorter chirps to maintain reasonable slopes
        # Higher frequencies allow for shorter chirps while maintaining velocity resolution
        
        # Base chirp duration for a 77 GHz radar with 150 MHz bandwidth
        base_chirp_duration = 50e-6  # 50 μs
        
        # Scale based on bandwidth (inversely proportional)
        bandwidth_factor = 150e6 / self.bandwidth
        
        # Scale based on center frequency (inversely proportional)
        frequency_factor = 77e9 / self.center_freq
        
        # Calculate optimal duration with constraints
        optimal_duration = base_chirp_duration * bandwidth_factor * frequency_factor
        
        # Apply practical constraints (minimum and maximum values)
        min_duration = 10e-6  # 10 μs minimum for practical hardware
        max_duration = 500e-6  # 500 μs maximum to avoid excessive range-Doppler coupling
        
        return max(min_duration, min(optimal_duration, max_duration))
    
    def calculate_min_snr_for_detection(self, detection_probability=0.9, false_alarm_rate=1e-6):
        """
        Calculate the minimum SNR required for target detection at maximum range
        
        Args:
            detection_probability: Desired probability of detection
            false_alarm_rate: Acceptable false alarm rate
            
        Returns:
            Minimum SNR in dB
        """
        # This is a simplified model based on the Neyman-Pearson detector
        # In practice, this depends on the specific detection algorithm (e.g., CFAR)
        
        # For a typical radar with coherent integration of multiple pulses
        # Higher detection probability or lower false alarm rate requires higher SNR
        
        # Simplified calculation based on detection theory
        # For non-fluctuating target (Swerling 0)
        import numpy as np
        from scipy.stats import norm
        
        # Calculate detection threshold for given false alarm rate
        threshold = -np.log(false_alarm_rate)
        
        # Calculate required SNR for given detection probability
        # This is a simplified approximation
        min_snr_linear = threshold / (1 - detection_probability)
        
        # Convert to dB
        min_snr_db = 10 * np.log10(min_snr_linear)
        
        return min_snr_db
    
    def get_performance_metrics_summary(self):
        """
        Get a summary of radar performance metrics and parameter relationships
        
        Returns:
            Dictionary containing performance metrics and explanations
        """
        metrics = {
            "range_resolution": {
                "value": self.range_resolution,
                "unit": "m",
                "description": "Minimum distance between two targets that can be separated",
                "dependencies": "Inversely proportional to bandwidth. Higher bandwidth gives better resolution.",
                "formula": "c / (2 * bandwidth)"
            },
            "max_range": {
                "value": self.max_range,
                "unit": "m",
                "description": "Maximum unambiguous range that can be measured",
                "dependencies": "Proportional to sample rate and chirp duration. Limited by signal attenuation.",
                "formula": "range_resolution * num_range_bins"
            },
            "velocity_resolution": {
                "value": self.velocity_resolution,
                "unit": "m/s",
                "description": "Minimum velocity difference between targets that can be separated",
                "dependencies": "Inversely proportional to total observation time (chirp_duration * num_chirps) and center frequency",
                "formula": "wavelength / (2 * chirp_duration * num_chirps)"
            },
            "max_velocity": {
                "value": self.max_velocity,
                "unit": "m/s",
                "description": "Maximum unambiguous velocity that can be measured",
                "dependencies": "Proportional to PRF (1/chirp_duration) and wavelength. Higher PRF allows higher velocities.",
                "formula": "velocity_resolution * (num_doppler_bins // 2)"
            }
        }
        
        # Add signal-type specific metrics
        if self.signal_type in ['FMCW', 'OFDM_FMCW', 'Sine_FMCW']:
            metrics["chirp_slope"] = {
                "value": self.chirp_slope,
                "unit": "Hz/s",
                "description": "Rate of frequency change in the FMCW chirp",
                "dependencies": "Ratio of bandwidth to chirp duration. Higher slopes require better hardware.",
                "formula": "bandwidth / chirp_duration"
            }
            
            metrics["optimal_chirp_duration"] = {
                "value": self.optimal_chirp_duration,
                "unit": "s",
                "description": "Calculated optimal chirp duration for this radar configuration",
                "dependencies": "Based on bandwidth and center frequency. Balances range and velocity resolution.",
                "formula": "Complex relationship based on bandwidth and center frequency"
            }
        
        if self.signal_type in ['OFDM', 'OFDM_FMCW']:
            metrics["ofdm_symbol_duration"] = {
                "value": self.ofdm_symbol_duration,
                "unit": "s",
                "description": "Duration of one OFDM symbol (without cyclic prefix)",
                "dependencies": "Inversely proportional to subcarrier spacing",
                "formula": "1 / subcarrier_spacing"
            }
            
            metrics["ofdm_bandwidth"] = {
                "value": self.ofdm_bandwidth,
                "unit": "Hz",
                "description": "Total bandwidth occupied by OFDM signal",
                "dependencies": "Product of number of subcarriers and subcarrier spacing",
                "formula": "num_subcarriers * subcarrier_spacing"
            }
        
        return metrics

    def print_radar_capabilities(self):
        """
        Print radar capabilities and parameter relationships in a human-readable format
        """
        metrics = self.get_performance_metrics_summary()
        
        print("\n===== RADAR CAPABILITIES =====")
        print(f"Signal Type: {self.signal_type}")
        print(f"Center Frequency: {self.center_freq/1e9:.2f} GHz")
        print(f"Bandwidth: {self.bandwidth/1e6:.2f} MHz")
        print(f"Chirp Duration: {self.chirp_duration*1e6:.2f} μs")
        print(f"Number of Chirps: {self.num_chirps}")
        print("\n--- PERFORMANCE METRICS ---")
        
        for name, info in metrics.items():
            # Format value based on magnitude
            if info["value"] < 0.001:
                value_str = f"{info['value']*1e6:.2f} μ{info['unit']}"
            elif info["value"] < 1:
                value_str = f"{info['value']*1e3:.2f} m{info['unit']}"
            else:
                value_str = f"{info['value']:.2f} {info['unit']}"
                
            print(f"{name.replace('_', ' ').title()}: {value_str}")
            print(f"  Description: {info['description']}")
            print(f"  Dependencies: {info['dependencies']}")
            print()
        
        print("--- PARAMETER IMPACT SUMMARY ---")
        print("• Increasing bandwidth improves range resolution but requires higher sampling rate")
        print("• Increasing chirp duration improves velocity resolution but reduces maximum unambiguous velocity")
        print("• Increasing number of chirps improves velocity resolution and SNR through coherent integration")
        print("• Increasing center frequency improves angular resolution but increases atmospheric attenuation")
        print("• For FMCW, the chirp slope (bandwidth/duration) is limited by hardware capabilities")
        print("• For OFDM, the subcarrier spacing affects symbol duration and Doppler resolution")
        
        if self.signal_type in ['FMCW', 'OFDM_FMCW', 'Sine_FMCW']:
            print(f"\nOptimal chirp duration for this configuration: {self.optimal_chirp_duration*1e6:.2f} μs")
            if abs(self.optimal_chirp_duration - self.chirp_duration) > 0.2 * self.chirp_duration:
                print(f"Note: Current chirp duration ({self.chirp_duration*1e6:.2f} μs) differs significantly from optimal.")
                if self.chirp_duration < self.optimal_chirp_duration:
                    print("  Consider increasing chirp duration to improve velocity resolution.")
                else:
                    print("  Consider decreasing chirp duration to improve maximum velocity measurement.")

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
    
    def simulate_hardware_modulation(self, baseband_signal, signal_type='FMCW'):
        """
        Simulate the hardware modulation process (inverse of demodulation)
        This simulates the two-step hardware solution:
        1. AD9361 generates baseband signal (OFDM or Sine) at ~2.1GHz
        2. CN0566 performs frequency sweep to achieve full bandwidth at ~10GHz
        
        Args:
            baseband_signal: Complex baseband signal [num_rx, num_chirps, samples_per_chirp]
            signal_type: Type of signal ('FMCW', 'OFDM_FMCW', 'Sine_FMCW')
            
        Returns:
            Modulated complex signal
        """
        # Create a copy of the input signal to avoid modifying the original
        modulated_signal = np.copy(baseband_signal)
        
        # Create time vector for one chirp
        t = np.arange(0, self.chirp_duration, 1/self.sample_rate)
        samples_per_chirp = len(t)
        
        # Only apply special modulation for two-step hardware solutions
        if signal_type in ['OFDM_FMCW', 'Sine_FMCW']:
            # For each RX and chirp, apply the CN0566 frequency sweep
            for rx_idx in range(modulated_signal.shape[0]):
                for chirp_idx in range(modulated_signal.shape[1]):
                    # Generate CN0566 frequency sweep component
                    k = self.bandwidth / self.chirp_duration  # Chirp rate for bandwidth
                    
                    # Phase calculation for the frequency sweep
                    # Starting at output_freq - bandwidth/2 and sweeping to output_freq + bandwidth/2
                    sweep_phase = 2 * np.pi * (self.output_freq * t + 0.5 * k * t**2)
                    fmcw_sweep = np.exp(1j * sweep_phase)
                    
                    # Apply the frequency sweep to the baseband signal
                    # This simulates the baseband signal being upconverted and swept by CN0566
                    modulated_signal[rx_idx, chirp_idx] = modulated_signal[rx_idx, chirp_idx] * fmcw_sweep
        
        # For standard FMCW, the modulation is already handled in the signal generation
        
        return modulated_signal

    def simulate_hardware_demodulation(self, complex_data):
        """
        Simulate the hardware demodulation process for hybrid signal types
        
        This simulates the two-step demodulation process:
        1. CN0566 (10GHz) sweep demodulation
        2. AD9361 (2.1GHz) baseband processing
        
        Args:
            complex_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
            
        Returns:
            Demodulated complex data with the same shape
        """
        # Create a copy of the input data to avoid modifying the original
        demodulated_data = np.zeros_like(complex_data, dtype=np.complex128)
        
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp = complex_data.shape
        
        # Simulate the CN0566 demodulation (first stage)
        if self.signal_type == 'OFDM_FMCW':
            # For OFDM_FMCW, we need to extract the OFDM component
            # First, apply bandpass filtering around the OFDM carrier frequency
            
            # Create time vector
            t = np.arange(samples_per_chirp) / self.sample_rate
            
            # For each RX and chirp
            for rx in range(num_rx):
                for chirp in range(num_chirps):
                    # 1. Demodulate the FMCW component (CN0566 stage)
                    # Simulate the mixing with the FMCW sweep
                    chirp_slope = self.bandwidth / self.chirp_duration
                    fmcw_phase = 2 * np.pi * (chirp_slope * t**2 / 2)
                    fmcw_demod = complex_data[rx, chirp, :] * np.exp(-1j * fmcw_phase)
                    
                    # 2. Apply bandpass filtering to isolate OFDM component
                    # This would be done by the AD9361 frontend
                    # Simplified simulation using FFT-based filtering
                    fft_signal = np.fft.fft(fmcw_demod)
                    
                    # Create bandpass filter around the OFDM subcarriers
                    # Assuming OFDM is centered at signal_freq
                    filter_mask = np.zeros_like(fft_signal)
                    center_bin = int(self.signal_freq * samples_per_chirp / self.sample_rate)
                    bandwidth_bins = int(self.num_subcarriers * self.subcarrier_spacing * samples_per_chirp / self.sample_rate)
                    
                    # Set filter passband
                    half_bw = bandwidth_bins // 2
                    filter_mask[center_bin-half_bw:center_bin+half_bw+1] = 1
                    
                    # Apply filter
                    filtered_fft = fft_signal * filter_mask
                    
                    # Convert back to time domain
                    demodulated_data[rx, chirp, :] = np.fft.ifft(filtered_fft)
                    
        elif self.signal_type == 'Sine_FMCW':
            # For Sine_FMCW, we need to extract the sine wave component
            
            # Create time vector
            t = np.arange(samples_per_chirp) / self.sample_rate
            
            # For each RX and chirp
            for rx in range(num_rx):
                for chirp in range(num_chirps):
                    # 1. Demodulate the FMCW component (CN0566 stage)
                    # Simulate the mixing with the FMCW sweep
                    chirp_slope = self.bandwidth / self.chirp_duration
                    fmcw_phase = 2 * np.pi * (chirp_slope * t**2 / 2)
                    fmcw_demod = complex_data[rx, chirp, :] * np.exp(-1j * fmcw_phase)
                    
                    # 2. Apply bandpass filtering to isolate sine component
                    # This would be done by the AD9361 frontend
                    # Simplified simulation using FFT-based filtering
                    fft_signal = np.fft.fft(fmcw_demod)
                    
                    # Create bandpass filter around the sine frequency
                    filter_mask = np.zeros_like(fft_signal)
                    center_bin = int(self.signal_freq * samples_per_chirp / self.sample_rate)
                    bandwidth_bins = int(0.1 * self.signal_freq * samples_per_chirp / self.sample_rate)  # 10% bandwidth
                    
                    # Set filter passband
                    half_bw = max(1, bandwidth_bins // 2)
                    filter_mask[center_bin-half_bw:center_bin+half_bw+1] = 1
                    
                    # Apply filter
                    filtered_fft = fft_signal * filter_mask
                    
                    # Convert back to time domain
                    demodulated_data[rx, chirp, :] = np.fft.ifft(filtered_fft)
        
        else:
            # For other signal types, just pass through
            demodulated_data = complex_data.copy()
        
        return demodulated_data

    def _apply_bandpass_filter(self, signal):
        """Apply bandpass filter to simulate AD9361 bandwidth limitation
        
        Args:
            signal: Complex signal array
            
        Returns:
            Filtered complex signal array
        """
        # Convert to frequency domain
        signal_fft = np.fft.fftshift(np.fft.fft(signal))
        
        # Create frequency vector
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/self.sample_rate))
        
        # Create bandpass filter centered at 0 (after downconversion)
        # with bandwidth matching the transceiver_bandwidth
        half_bw = self.transceiver_bandwidth / 2
        bandpass = np.abs(freqs) <= half_bw
        
        # Apply filter
        filtered_fft = signal_fft * bandpass
        
        # Convert back to time domain
        filtered_signal = np.fft.ifft(np.fft.ifftshift(filtered_fft))
        
        return filtered_signal

    def process_radar_data(self, time_domain_data, apply_cfar=True, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-4):
        """
        Process radar data to generate range-Doppler map and detect targets
        
        Args:
            time_domain_data: Time domain radar data with shape [num_rx, num_chirps, samples_per_chirp]
                            which is already in complex format
            apply_cfar: Whether to apply CFAR detection
            guard_cells: Tuple of (doppler_guard, range_guard) cells to exclude around CUT
            training_cells: Tuple of (doppler_train, range_train) cells to use for estimation
            pfa: Probability of false alarm
            
        Returns:
            Dictionary containing processed data including range-Doppler map and detected targets
        """
        # Check if the input data is already complex
        if np.iscomplexobj(time_domain_data):
            complex_data = time_domain_data
        else:
            # If the data has a last dimension of size 2, convert to complex
            if time_domain_data.shape[-1] == 2:
                complex_data = time_domain_data[..., 0] + 1j * time_domain_data[..., 1]
            else:
                raise ValueError("Input data must be either complex or have a last dimension of size 2 for I/Q data")
        
        # Generate range-Doppler map
        rd_map = self.time_to_range_doppler(complex_data) #(2, 16, 128)
        
        # Apply CFAR detection if requested
        if apply_cfar:
            # Apply CFAR detection to identify targets
            cfar_map = self._apply_cfar(rd_map, guard_cells, training_cells, pfa) #(16, 128)
            detected_targets = self.extract_targets_from_cfar(cfar_map)
        else:
            cfar_map = None
            detected_targets = None
        
        # Return results
        return {
            'range_doppler_map': rd_map,
            'cfar_map': cfar_map,
            'detected_targets': detected_targets
        }
    
    def extract_targets_from_cfar(self, cfar_map, min_value=0.5):
        """
        Extract target information from CFAR detection map
        
        Args:
            cfar_map: CFAR detection map with shape [num_doppler_bins, num_range_bins]
            min_value: Minimum value for target detection (typically 0.5 or 1.0 for binary maps)
            
        Returns:
            List of dictionaries containing target information
        """
        # Find peaks in CFAR map
        peaks = np.where(cfar_map >= min_value)
        
        # Extract target information
        targets = []
        for i in range(len(peaks[0])):
            doppler_idx = peaks[0][i]
            range_idx = peaks[1][i]
            
            # Convert indices to physical values
            range_m = range_idx * self.range_resolution
            
            # Convert Doppler index to velocity
            # Adjust for Doppler FFT shift
            doppler_shifted = doppler_idx
            if doppler_shifted >= self.num_doppler_bins // 2:
                doppler_shifted = doppler_shifted - self.num_doppler_bins
            velocity = doppler_shifted * self.velocity_resolution
            
            # Add target to list
            targets.append({
                'range': range_m,
                'velocity': velocity,
                'amplitude': cfar_map[doppler_idx, range_idx],
                'doppler_idx': doppler_idx,
                'range_idx': range_idx
            })
        
        return targets

    def time_to_range_doppler(self, complex_data):
        """
        Convert time domain complex data to range-Doppler map
        
        Args:
            complex_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
                          or [num_chirps, samples_per_chirp] or with I/Q components
                          
        Returns:
            Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
            where the first dimension contains real and imaginary parts
        """
        # Ensure data has the right shape and format
        complex_data = self._prepare_complex_data(complex_data) #(4, 32, 125000)
        
        # Process based on signal type
        if self.signal_type == 'FMCW' or self.signal_type == 'Sine_FMCW':
            # For FMCW, use standard range-Doppler processing
            rd_map = self._process_fmcw_range_doppler(complex_data)
        elif self.signal_type == 'OFDM' or self.signal_type == 'OFDM_FMCW':
            # For OFDM, use OFDM-specific processing
            rd_map = self._process_ofdm_range_doppler(complex_data)
        elif self.signal_type == 'Sine':
            # For Sine wave, use CW processing
            rd_map = self._process_sine_range_doppler(complex_data)
        else:
            # Default to FMCW processing
            rd_map = self._process_fmcw_range_doppler(complex_data)
        
        # Apply any additional processing if needed
        # (e.g., normalization, calibration, etc.)
        rd_map = self._post_process_rd_map(rd_map) #(2, 16, 128)
        
        return rd_map
    
    def _process_fmcw_range_doppler(self, complex_data):
        """
        Process FMCW radar data to create range-Doppler map
        
        Args:
            complex_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
                        
        Returns:
            Range-Doppler map with shape [num_doppler_bins, num_range_bins] or
            [2, num_doppler_bins, num_range_bins] with real and imaginary parts
        """
        # Ensure data has the right shape
        if len(complex_data.shape) == 2:
            # Single receiver case, add dimension
            complex_data = complex_data[np.newaxis, :, :]
        
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp = complex_data.shape
        
        # Initialize range-Doppler map to accumulate results from all receivers
        rd_map_complex = np.zeros((self.num_doppler_bins, self.num_range_bins), dtype=complex)
        
        # Process each receiver and accumulate results for better SNR
        for rx_idx in range(num_rx):
            # Apply windowing for range processing (along fast time)
            # Window function reduces sidelobes in the frequency domain
            range_window = np.hamming(samples_per_chirp)
            windowed_data = complex_data[rx_idx] * range_window[np.newaxis, :]
            
            # Perform range FFT (along fast time)
            # For FMCW, the beat frequency is proportional to range
            range_fft = np.fft.fft(windowed_data, n=self.num_range_bins, axis=1)
            
            # Remove DC component (direct signal leakage)
            range_fft[:, 0] = range_fft[:, 0] * 0.1  # Attenuate DC component
            
            # Apply windowing for Doppler processing (along slow time)
            doppler_window = np.hamming(num_chirps)
            doppler_windowed = range_fft * doppler_window[:, np.newaxis]
            
            # Perform Doppler FFT (along slow time)
            # For FMCW, the phase change between chirps is proportional to velocity
            rd_map_rx = np.fft.fftshift(np.fft.fft(doppler_windowed, n=self.num_doppler_bins, axis=0), axes=0)
            
            # Accumulate results (coherent integration)
            rd_map_complex += rd_map_rx
        
        # Normalize by number of receivers
        rd_map_complex /= num_rx
        
        # Format output based on expected format
        if np.iscomplexobj(rd_map_complex):
            # Split into real and imaginary parts
            real_part = np.real(rd_map_complex)
            imag_part = np.imag(rd_map_complex)
            
            # Stack to create output format [2, num_doppler_bins, num_range_bins]
            return np.stack([real_part, imag_part], axis=0)
        else:
            return rd_map_complex

    def _post_process_rd_map(self, rd_map):
        """
        Apply post-processing to range-Doppler map to enhance target visibility
        
        Args:
            rd_map: Range-Doppler map
            
        Returns:
            Processed range-Doppler map
        """
        # Convert to complex format if needed
        if len(rd_map.shape) == 3 and rd_map.shape[0] == 2:
            # Convert from [2, num_doppler_bins, num_range_bins] to complex
            rd_complex = rd_map[0] + 1j * rd_map[1]
        elif np.iscomplexobj(rd_map):
            rd_complex = rd_map
        else:
            # If we have a magnitude-only array, create zero imaginary part
            rd_complex = rd_map + 0j
        
        # Calculate magnitude
        rd_magnitude = np.abs(rd_complex)
        
        # Apply logarithmic scaling to better visualize targets
        # Add a small constant to avoid log(0)
        rd_magnitude_db = 20 * np.log10(rd_magnitude + 1e-10)
        
        # Apply dynamic range limiting to highlight targets
        dynamic_range_db = 40  # Adjust this value as needed
        rd_magnitude_db_limited = np.maximum(rd_magnitude_db, np.max(rd_magnitude_db) - dynamic_range_db)
        
        # Normalize to 0-1 range for better visualization
        rd_magnitude_norm = (rd_magnitude_db_limited - np.min(rd_magnitude_db_limited)) / (np.max(rd_magnitude_db_limited) - np.min(rd_magnitude_db_limited) + 1e-10)
        
        # Convert back to complex format
        rd_enhanced = rd_magnitude_norm * np.exp(1j * np.angle(rd_complex))
        
        # Format output based on expected format
        real_part = np.real(rd_enhanced)
        imag_part = np.imag(rd_enhanced)
        
        # Stack to create output format [2, num_doppler_bins, num_range_bins]
        return np.stack([real_part, imag_part], axis=0)

    def _post_process_rd_map_old(self, rd_map):
        """
        Apply post-processing to range-Doppler map
        
        Args:
            rd_map: Range-Doppler map
            
        Returns:
            Processed range-Doppler map
        """
        # Ensure the output has the correct shape
        if len(rd_map.shape) == 3 and rd_map.shape[0] == 2:
            # Already in the correct format [2, num_doppler_bins, num_range_bins]
            return rd_map
        
        # If we have a complex array, convert to real/imaginary format
        if np.iscomplexobj(rd_map):
            # Extract real and imaginary parts
            real_part = np.real(rd_map)
            imag_part = np.imag(rd_map)
            
            # Stack to create output format [2, num_doppler_bins, num_range_bins]
            return np.stack([real_part, imag_part], axis=0)
        
        # If we have a magnitude-only array, create zero imaginary part
        if len(rd_map.shape) == 2:
            # Create zero imaginary part
            imag_part = np.zeros_like(rd_map)
            
            # Stack to create output format [2, num_doppler_bins, num_range_bins]
            return np.stack([rd_map, imag_part], axis=0)
        
        # If we have another format, try to convert it
        return self._format_output(rd_map)
    
    def _process_ofdm_range_doppler(self, complex_data):
        """
        Process OFDM radar data to create range-Doppler map
        
        Args:
            complex_data: Complex time domain data
        
        Returns:
            Range-Doppler map
        """
        # Ensure data has the right shape
        complex_data = self._prepare_complex_data(complex_data)
        
        # For OFDM, we first need to perform OFDM demodulation
        # This involves FFT across subcarriers
        ofdm_demod_data = self._perform_ofdm_demodulation(complex_data)
        
        # Extract channel information from demodulated data
        channel_data = self._extract_ofdm_channel(ofdm_demod_data)
        
        # Apply windowing for range processing
        # Use FFT size based on num_subcarriers (typically power of 2)
        fft_size = 2**int(np.ceil(np.log2(self.num_subcarriers)))
        windowed_data = self._apply_window(channel_data, axis=2)
        
        # Perform range FFT (IFFT of channel data gives range profile)
        # For OFDM, range information is encoded in the frequency domain
        range_data = self._perform_ofdm_range_fft(windowed_data)
        
        # Apply windowing for Doppler processing
        # Window across chirps (OFDM symbols) for Doppler processing
        doppler_windowed = self._apply_window(range_data, axis=1)
        
        # Perform Doppler FFT (along slow time)
        rd_map = self._perform_doppler_fft(doppler_windowed)
        
        # Reshape to standard output format [2, num_doppler_bins, num_range_bins]
        return self._format_output(rd_map)
    
    def _perform_ofdm_demodulation(self, complex_data):
        """
        Perform OFDM demodulation on complex time domain data
        
        Args:
            complex_data: Complex time domain data with shape [num_rx, num_chirps, samples_per_chirp]
            
        Returns:
            Demodulated OFDM data with shape [num_rx, num_chirps, num_subcarriers]
        """
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp = complex_data.shape
        
        # Calculate FFT size (typically power of 2)
        fft_size = 2**int(np.ceil(np.log2(self.num_subcarriers)))
        
        # Calculate cyclic prefix length (typically 20-25% of symbol)
        cp_length = int(0.25 * fft_size)
        
        # Initialize output array
        demodulated_data = np.zeros((num_rx, num_chirps, self.num_subcarriers), dtype=np.complex64)
        
        # Process each RX and chirp
        for rx in range(num_rx):
            for chirp in range(num_chirps):
                # Extract time domain signal for this RX and chirp
                signal = complex_data[rx, chirp, :]
                
                # Remove cyclic prefix if signal is long enough
                if samples_per_chirp >= fft_size + cp_length:
                    # Extract the symbol without CP
                    symbol = signal[cp_length:cp_length+fft_size]
                else:
                    # If signal is too short, pad with zeros
                    symbol = np.zeros(fft_size, dtype=np.complex64)
                    symbol[:min(fft_size, samples_per_chirp)] = signal[:min(fft_size, samples_per_chirp)]
                
                # Perform FFT to convert to frequency domain
                freq_domain = np.fft.fft(symbol) / np.sqrt(fft_size)
                
                # Extract the subcarriers
                start_idx = (fft_size - self.num_subcarriers) // 2
                demodulated_data[rx, chirp, :] = freq_domain[start_idx:start_idx+self.num_subcarriers]
        
        return demodulated_data

    def _extract_ofdm_channel(self, ofdm_demod_data):
        """
        Extract channel information from demodulated OFDM data
        
        Args:
            ofdm_demod_data: Demodulated OFDM data with shape [num_rx, num_chirps, num_subcarriers]
            
        Returns:
            Channel data with shape [num_rx, num_chirps, num_subcarriers]
        """
        # In a real system, we would use pilot subcarriers for channel estimation
        # For simulation, we can use the demodulated data directly as channel information
        # In a more advanced implementation, we would divide by the known transmitted symbols
        
        # For now, just return the demodulated data as channel information
        return ofdm_demod_data

    def _perform_ofdm_range_fft(self, channel_data):
        """
        Perform range FFT on OFDM channel data
        
        Args:
            channel_data: Channel data with shape [num_rx, num_chirps, num_subcarriers]
            
        Returns:
            Range data with shape [num_rx, num_chirps, num_range_bins]
        """
        # Get dimensions
        num_rx, num_chirps, num_subcarriers = channel_data.shape
        
        # Initialize output array
        range_data = np.zeros((num_rx, num_chirps, self.num_range_bins), dtype=np.complex64)
        
        # Process each RX and chirp
        for rx in range(num_rx):
            for chirp in range(num_chirps):
                # Extract channel data for this RX and chirp
                subcarrier_data = channel_data[rx, chirp, :]
                
                # Perform IFFT to get range profile
                # For OFDM, range information is in the frequency domain
                # IFFT converts frequency domain to time domain (range)
                range_profile = np.fft.ifft(subcarrier_data, n=self.num_range_bins)
                
                # Store range profile
                range_data[rx, chirp, :] = range_profile
        
        return range_data

    def _process_sine_range_doppler(self, complex_data):
        """
        Process Sine wave radar data to create range-Doppler map
        
        Args:
            complex_data: Complex time domain data
        
        Returns:
            Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
        """
        # Ensure data has the right shape
        complex_data = self._prepare_complex_data(complex_data)
        
        # For sine wave, we primarily focus on phase changes for Doppler
        # First, extract phase information
        phase_data = self._extract_sine_phase(complex_data)
        
        # Apply windowing for range processing (limited range info in sine wave)
        windowed_data = self._apply_window(complex_data, axis=2)
        
        # Perform simplified range processing
        range_data = self._perform_sine_range_processing(windowed_data)
        
        # Apply windowing for Doppler processing
        doppler_windowed = self._apply_window(range_data, axis=1)
        
        # Perform Doppler FFT (along slow time)
        rd_map = self._perform_doppler_fft(doppler_windowed)
        
        # Reshape to standard output format [2, num_doppler_bins, num_range_bins]
        return self._format_output(rd_map)
    
    def _process_hybrid_ofdm_fmcw_range_doppler(self, complex_data):
        """
        Process hybrid OFDM-FMCW radar data to create range-Doppler map
        
        Args:
            complex_data: Complex time domain data
        
        Returns:
            Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
        """
        # Ensure data has the right shape
        complex_data = self._prepare_complex_data(complex_data)
        
        # For hybrid signals, we process both components and combine results
        # First, extract OFDM component
        ofdm_data = self._extract_ofdm_component(complex_data)
        
        # Process OFDM component
        ofdm_rd_map = self._process_ofdm_range_doppler(ofdm_data)
        
        # Extract FMCW component
        fmcw_data = self._extract_fmcw_component(complex_data)
        
        # Process FMCW component
        fmcw_rd_map = self._process_fmcw_range_doppler(fmcw_data)
        
        # Combine the results (weighted average based on SNR)
        rd_map = self._combine_hybrid_results(ofdm_rd_map, fmcw_rd_map)
        
        # Ensure output format is consistent [2, num_doppler_bins, num_range_bins]
        return self._format_output(rd_map)
    
    def _process_hybrid_sine_fmcw_range_doppler(self, complex_data):
        """
        Process hybrid Sine-FMCW radar data to create range-Doppler map
        
        Args:
            complex_data: Complex time domain data
        
        Returns:
            Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
        """
        # Ensure data has the right shape
        complex_data = self._prepare_complex_data(complex_data)
        
        # Extract Sine component
        sine_data = self._extract_sine_component(complex_data)
        
        # Process Sine component
        sine_rd_map = self._process_sine_range_doppler(sine_data)
        
        # Extract FMCW component
        fmcw_data = self._extract_fmcw_component(complex_data)
        
        # Process FMCW component
        fmcw_rd_map = self._process_fmcw_range_doppler(fmcw_data)
        
        # Combine the results (weighted average based on SNR)
        rd_map = self._combine_hybrid_results(sine_rd_map, fmcw_rd_map)
        
        # Ensure output format is consistent [2, num_doppler_bins, num_range_bins]
        return self._format_output(rd_map)
    
    # Shared utility functions for range-Doppler processing
    
    def _prepare_complex_data(self, complex_data):
        """
        Ensure complex data has the right shape for processing
        
        Args:
            complex_data: Complex time domain data
        
        Returns:
            Properly shaped complex data
        """
        # Check if we need to add a dimension for single receiver
        if len(complex_data.shape) == 2:
            # Add dimension for single receiver
            complex_data = complex_data[np.newaxis, :, :]
            
        # Check if we have I/Q components separated
        elif len(complex_data.shape) == 4 and complex_data.shape[-1] == 2:
            # Convert from [num_rx, num_chirps, samples_per_chirp, 2] to complex
            complex_data = complex_data[..., 0] + 1j * complex_data[..., 1]
        
        return complex_data
    
    def _apply_window(self, data, axis, window_type='hann'):
        """
        Apply window function to data along specified axis
        
        Args:
            data: Input data
            axis: Axis to apply window
            window_type: Type of window function
        
        Returns:
            Windowed data
        """
        # Get the size of the dimension to window
        dim_size = data.shape[axis]
        
        # Create window function
        if window_type == 'hann':
            window = np.hanning(dim_size)
        elif window_type == 'hamming':
            window = np.hamming(dim_size)
        elif window_type == 'blackman':
            window = np.blackman(dim_size)
        else:
            window = np.hanning(dim_size)  # Default to Hann window
        
        # Reshape window for broadcasting
        reshape_dims = [1] * len(data.shape)
        reshape_dims[axis] = dim_size
        window = window.reshape(reshape_dims)
        
        # Apply window
        return data * window
    
    def _perform_range_fft(self, data):
        """
        Perform range FFT on windowed data
        
        Args:
            data: Windowed data
        
        Returns:
            Range processed data
        """
        # Perform FFT along the samples_per_chirp dimension (axis=2)
        range_data = np.fft.fft(data, axis=2)
        
        # Optionally truncate to num_range_bins if needed
        if range_data.shape[2] > self.num_range_bins:
            range_data = range_data[:, :, :self.num_range_bins]
        
        return range_data
    
    def _perform_doppler_fft(self, data):
        """
        Perform Doppler FFT on range processed data
        
        Args:
            data: Range processed data
        
        Returns:
            Range-Doppler map
        """
        # Perform FFT along the chirps dimension (axis=1)
        rd_map = np.fft.fft(data, axis=1)
        
        # Optionally truncate to num_doppler_bins if needed
        if rd_map.shape[1] > self.num_doppler_bins:
            rd_map = rd_map[:, :self.num_doppler_bins, :]
        
        # Shift zero-frequency component to center
        rd_map = np.fft.fftshift(rd_map, axes=1)
        
        return rd_map
    
    def _format_output(self, rd_map):
        """
        Format range-Doppler map to standard output format
        
        Args:
            rd_map: Range-Doppler map
        
        Returns:
            Formatted range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
        """
        # Average across receivers if multiple receivers
        if rd_map.shape[0] > 1:
            rd_map = np.mean(rd_map, axis=0, keepdims=True)
        
        # Reshape to [2, num_doppler_bins, num_range_bins]
        # where the first dimension contains real and imaginary parts
        real_part = np.real(rd_map[0])
        imag_part = np.imag(rd_map[0])
        
        return np.stack([real_part, imag_part], axis=0)

    def time_to_range_doppler_old(self, complex_data):
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
            rd_map: Range-Doppler map with shape [2, num_doppler_bins, num_range_bins]
                    where the first dimension contains real and imaginary parts
            guard_cells: Tuple of (doppler_guard, range_guard) cells to exclude around CUT
            training_cells: Tuple of (doppler_train, range_train) cells to use for estimation
            pfa: Probability of false alarm
            
        Returns:
            Binary detection map with shape [num_doppler_bins, num_range_bins]
        """
        # Check if rd_map is in the format [2, num_doppler_bins, num_range_bins]
        if len(rd_map.shape) == 3 and rd_map.shape[0] == 2:
            # Convert complex format [2, num_doppler_bins, num_range_bins] to magnitude
            rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
            doppler_bins, range_bins = rd_magnitude.shape
        else:
            # Assume rd_map is already in magnitude format [num_doppler_bins, num_range_bins]
            doppler_bins, range_bins = rd_map.shape
            rd_magnitude = rd_map
        
        # Extract guard and training cell sizes
        doppler_guard, range_guard = guard_cells
        doppler_train, range_train = training_cells
        
        # Calculate total window size
        window_doppler = 2 * (doppler_guard + doppler_train) + 1
        window_range = 2 * (range_guard + range_train) + 1
        
        # Initialize detection map
        detection_map = np.zeros((doppler_bins, range_bins))
        
        # Apply CFAR for each cell
        for d in range(doppler_bins):
            for r in range(range_bins):
                # Define cell under test
                cell_under_test = rd_magnitude[d, r]
                
                # Define window boundaries with wraparound for Doppler (circular)
                # and clipping for range
                d_min = max(0, d - doppler_guard - doppler_train)
                d_max = min(doppler_bins - 1, d + doppler_guard + doppler_train)
                r_min = max(0, r - range_guard - range_train)
                r_max = min(range_bins - 1, r + range_guard + range_train)
                
                # Extract window
                window = rd_magnitude[d_min:d_max+1, r_min:r_max+1]
                
                # Create guard cell mask (1 for training cells, 0 for guard cells and CUT)
                mask = np.ones_like(window)
                
                # Calculate guard cell region in the window
                gd_min = max(0, doppler_guard + doppler_train - (d - d_min))
                gd_max = min(window.shape[0] - 1, doppler_guard + doppler_train + (d_max - d))
                gr_min = max(0, range_guard + range_train - (r - r_min))
                gr_max = min(window.shape[1] - 1, range_guard + range_train + (r_max - r))
                
                # Set guard cells and CUT to 0 in mask
                mask[gd_min:gd_max+1, gr_min:gr_max+1] = 0
                
                # Calculate noise level from training cells
                if np.sum(mask) > 0:  # Ensure we have training cells
                    noise_level = np.sum(window * mask) / np.sum(mask)
                    
                    # Calculate threshold based on desired PFA
                    # For exponentially distributed noise (magnitude of complex Gaussian)
                    alpha = -np.log(pfa)
                    threshold = alpha * noise_level
                    
                    # Apply threshold
                    if cell_under_test > threshold:
                        detection_map[d, r] = 1
        
        return detection_map
    
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
        Detect targets in a range-Doppler map using signal-type specific processing
        
        Args:
            rd_map: Range-Doppler map
            threshold: Detection threshold (0-1)
            min_area: Minimum area of connected components to be considered a target
            
        Returns:
            List of dictionaries containing target information
        """
        # Determine which detector to use based on signal type
        if self.signal_type == 'FMCW':
            return self._detect_targets_fmcw(rd_map, threshold, min_area)
        elif self.signal_type == 'OFDM':
            return self._detect_targets_ofdm(rd_map, threshold, min_area)
        elif self.signal_type == 'Sine':
            return self._detect_targets_sine(rd_map, threshold, min_area)
        elif self.signal_type == 'OFDM_FMCW' or self.signal_type == 'Sine_FMCW':
            return self._detect_targets_combined(rd_map, threshold, min_area)
        else:
            # Default to FMCW processing if signal type is not recognized
            return self._detect_targets_fmcw(rd_map, threshold, min_area)
    
    def _detect_targets_common(self, binary_map):
        """
        Common target detection logic shared by all signal types
        
        Args:
            binary_map: Binary map of detections after thresholding
            
        Returns:
            List of connected components (potential targets)
        """
        # Find connected components in the binary map
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map.astype(np.uint8), connectivity=8)
        
        # Extract components (skipping the background which is label 0)
        components = []
        for i in range(1, num_labels):
            components.append({
                'label': i,
                'centroid': centroids[i],
                'stats': stats[i],
                'area': stats[i, cv2.CC_STAT_AREA]
            })
            
        return components
    
    def _detect_targets_fmcw(self, rd_map, threshold=0.15, min_area=2):
        """
        Detect targets in FMCW radar data
        
        Args:
            rd_map: Range-Doppler map
            threshold: Detection threshold (0-1)
            min_area: Minimum area of connected components to be considered a target
            
        Returns:
            List of dictionaries containing target information
        """
        # Get magnitude of range-Doppler map
        rd_magnitude = np.abs(rd_map)
        
        # Normalize to 0-1 range
        rd_normalized = rd_magnitude / np.max(rd_magnitude)
        
        # Apply threshold to get binary detection map
        binary_map = (rd_normalized > threshold).astype(np.uint8)
        
        # Get connected components
        components = self._detect_targets_common(binary_map)
        
        # Filter components by minimum area and extract target information
        targets = []
        for comp in components:
            if comp['area'] >= min_area:
                # Get centroid coordinates
                doppler_idx, range_idx = comp['centroid']
                
                # Convert indices to physical values
                range_m = range_idx * self.range_resolution
                velocity_mps = (doppler_idx - self.num_doppler_bins // 2) * self.velocity_resolution
                
                # Calculate SNR (simplified)
                peak_value = rd_normalized[int(doppler_idx), int(range_idx)]
                snr_db = 20 * np.log10(peak_value / (np.mean(rd_normalized) + 1e-10))
                
                # Add target to list
                targets.append({
                    'range': range_m,
                    'velocity': velocity_mps,
                    'snr': snr_db,
                    'doppler_idx': int(doppler_idx),
                    'range_idx': int(range_idx),
                    'area': comp['area']
                })
        
        return targets
    
    def _detect_targets_ofdm(self, rd_map, threshold=0.15, min_area=2):
        """
        Detect targets in OFDM radar data
        
        Args:
            rd_map: Range-Doppler map
            threshold: Detection threshold (0-1)
            min_area: Minimum area of connected components to be considered a target
            
        Returns:
            List of dictionaries containing target information
        """
        # Get magnitude of range-Doppler map
        rd_magnitude = np.abs(rd_map)
        
        # Apply additional OFDM-specific processing
        # OFDM typically has higher sidelobes, so we apply a more aggressive normalization
        rd_normalized = rd_magnitude / (np.max(rd_magnitude) + 1e-10)
        
        # Apply OFDM-specific filtering to reduce sidelobes
        # Simple 3x3 median filter to reduce noise
        rd_filtered = cv2.medianBlur(rd_normalized.astype(np.float32), 3)
        
        # Apply threshold to get binary detection map
        binary_map = (rd_filtered > threshold).astype(np.uint8)
        
        # Get connected components
        components = self._detect_targets_common(binary_map)
        
        # Filter components by minimum area and extract target information
        targets = []
        for comp in components:
            if comp['area'] >= min_area:
                # Get centroid coordinates
                doppler_idx, range_idx = comp['centroid']
                
                # Convert indices to physical values
                range_m = range_idx * self.range_resolution
                velocity_mps = (doppler_idx - self.num_doppler_bins // 2) * self.velocity_resolution
                
                # Calculate SNR (simplified)
                peak_value = rd_normalized[int(doppler_idx), int(range_idx)]
                snr_db = 20 * np.log10(peak_value / (np.mean(rd_normalized) + 1e-10))
                
                # Add target to list with OFDM-specific confidence metric
                # OFDM typically has better range resolution but worse Doppler resolution
                range_confidence = min(1.0, peak_value * 1.2)  # Boost range confidence
                doppler_confidence = min(1.0, peak_value * 0.8)  # Reduce Doppler confidence
                
                targets.append({
                    'range': range_m,
                    'velocity': velocity_mps,
                    'snr': snr_db,
                    'doppler_idx': int(doppler_idx),
                    'range_idx': int(range_idx),
                    'area': comp['area'],
                    'range_confidence': range_confidence,
                    'doppler_confidence': doppler_confidence
                })
        
        return targets
    
    def _detect_targets_sine(self, rd_map, threshold=0.15, min_area=2):
        """
        Detect targets in Sine (CW) radar data
        
        Args:
            rd_map: Range-Doppler map
            threshold: Detection threshold (0-1)
            min_area: Minimum area of connected components to be considered a target
            
        Returns:
            List of dictionaries containing target information
        """
        # Get magnitude of range-Doppler map
        rd_magnitude = np.abs(rd_map)
        
        # For Sine wave radar, we focus primarily on the Doppler dimension
        # since range information is limited
        
        # Sum across range bins to get Doppler profile
        doppler_profile = np.sum(rd_magnitude, axis=1)
        doppler_profile = doppler_profile / np.max(doppler_profile)
        
        # Create a 2D map with enhanced Doppler information
        rd_normalized = rd_magnitude / (np.max(rd_magnitude) + 1e-10)
        
        # Apply threshold to get binary detection map
        binary_map = (rd_normalized > threshold).astype(np.uint8)
        
        # Get connected components
        components = self._detect_targets_common(binary_map)
        
        # Filter components by minimum area and extract target information
        targets = []
        for comp in components:
            if comp['area'] >= min_area:
                # Get centroid coordinates
                doppler_idx, range_idx = comp['centroid']
                
                # For Sine wave, range is less accurate, so we add uncertainty
                range_m = range_idx * self.range_resolution
                velocity_mps = (doppler_idx - self.num_doppler_bins // 2) * self.velocity_resolution
                
                # Calculate SNR (simplified)
                peak_value = rd_normalized[int(doppler_idx), int(range_idx)]
                snr_db = 20 * np.log10(peak_value / (np.mean(rd_normalized) + 1e-10))
                
                # For Sine wave, we have high confidence in velocity but low in range
                targets.append({
                    'range': range_m,
                    'velocity': velocity_mps,
                    'snr': snr_db,
                    'doppler_idx': int(doppler_idx),
                    'range_idx': int(range_idx),
                    'area': comp['area'],
                    'range_confidence': 0.3,  # Low confidence in range for Sine wave
                    'doppler_confidence': 0.9  # High confidence in Doppler for Sine wave
                })
        
        return targets
    
    def _detect_targets_combined(self, rd_map, threshold=0.15, min_area=2):
        """
        Detect targets in combined signal types (OFDM_FMCW or Sine_FMCW)
        
        Args:
            rd_map: Range-Doppler map
            threshold: Detection threshold (0-1)
            min_area: Minimum area of connected components to be considered a target
            
        Returns:
            List of dictionaries containing target information
        """
        # Get magnitude of range-Doppler map
        rd_magnitude = np.abs(rd_map)
        
        # Normalize to 0-1 range
        rd_normalized = rd_magnitude / np.max(rd_magnitude)
        
        # For combined signals, we can leverage the strengths of both signal types
        if self.signal_type == 'OFDM_FMCW':
            # OFDM provides better range resolution, FMCW provides better Doppler
            # Apply a 2D filter that preserves peaks while reducing sidelobes
            kernel = np.ones((3, 3)) / 9
            rd_filtered = cv2.filter2D(rd_normalized.astype(np.float32), -1, kernel)
        else:  # Sine_FMCW
            # Sine provides better Doppler resolution, FMCW provides range information
            # Apply a filter that emphasizes Doppler dimension
            kernel = np.ones((5, 3)) / 15
            rd_filtered = cv2.filter2D(rd_normalized.astype(np.float32), -1, kernel)
        
        # Apply threshold to get binary detection map
        binary_map = (rd_filtered > threshold).astype(np.uint8)
        
        # Get connected components
        components = self._detect_targets_common(binary_map)
        
        # Filter components by minimum area and extract target information
        targets = []
        for comp in components:
            if comp['area'] >= min_area:
                # Get centroid coordinates
                doppler_idx, range_idx = comp['centroid']
                
                # Convert indices to physical values
                range_m = range_idx * self.range_resolution
                velocity_mps = (doppler_idx - self.num_doppler_bins // 2) * self.velocity_resolution
                
                # Calculate SNR (simplified)
                peak_value = rd_normalized[int(doppler_idx), int(range_idx)]
                snr_db = 20 * np.log10(peak_value / (np.mean(rd_normalized) + 1e-10))
                
                # Set confidence based on signal type
                if self.signal_type == 'OFDM_FMCW':
                    range_confidence = 0.9  # High confidence in range
                    doppler_confidence = 0.8  # Good confidence in Doppler
                else:  # Sine_FMCW
                    range_confidence = 0.7  # Good confidence in range
                    doppler_confidence = 0.9  # High confidence in Doppler
                
                targets.append({
                    'range': range_m,
                    'velocity': velocity_mps,
                    'snr': snr_db,
                    'doppler_idx': int(doppler_idx),
                    'range_idx': int(range_idx),
                    'area': comp['area'],
                    'range_confidence': range_confidence,
                    'doppler_confidence': doppler_confidence
                })
        
        return targets

    def detect_targets_old(self, rd_map, threshold=0.15, min_area=2):
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
    
    def detect_crosstalk_timing(self, rx_signal):
        """
        Detect the transmitter-receiver crosstalk to estimate system timing
        
        Args:
            rx_signal: Complex received signal array [num_rx, num_chirps, samples_per_chirp]
            
        Returns:
            Estimated delay in samples and seconds
        """
        # Get dimensions
        num_rx, num_chirps, samples_per_chirp = rx_signal.shape
        
        # Initialize arrays to store detected delays
        delays_samples = np.zeros((num_rx, num_chirps), dtype=int)
        delays_seconds = np.zeros((num_rx, num_chirps), dtype=float)
        
        # For each RX and chirp, detect the crosstalk
        for rx_idx in range(num_rx):
            for chirp_idx in range(num_chirps):
                # Get the signal for this RX and chirp
                signal = rx_signal[rx_idx, chirp_idx]
                
                # Method 1: Energy detection
                # Calculate signal energy
                signal_energy = np.abs(signal)**2
                
                # Apply moving average to smooth
                window_size = int(self.sample_rate * 1e-6)  # 1 μs window
                window_size = max(1, window_size)  # Ensure window size is at least 1
                energy_smooth = np.convolve(signal_energy, np.ones(window_size)/window_size, mode='same')
                
                # Find the first significant rise in energy
                # This is likely the crosstalk from TX to RX
                energy_diff = np.diff(energy_smooth)
                threshold = 0.3 * np.max(energy_diff)  # Adjust threshold as needed
                
                # Find the first point where energy rises significantly
                rise_points = np.where(energy_diff > threshold)[0]
                
                if len(rise_points) > 0:
                    # The first significant rise is likely the crosstalk
                    first_rise = rise_points[0]
                    delays_samples[rx_idx, chirp_idx] = first_rise
                    delays_seconds[rx_idx, chirp_idx] = first_rise / self.sample_rate
                else:
                    # Fallback if no clear rise is detected
                    delays_samples[rx_idx, chirp_idx] = 0
                    delays_seconds[rx_idx, chirp_idx] = 0
        
        # Calculate median delay across all RX and chirps for robustness
        median_delay_samples = np.median(delays_samples)
        median_delay_seconds = np.median(delays_seconds)
        
        return int(median_delay_samples), median_delay_seconds

    def estimate_system_delay_from_crosstalk(self, rx_signal):
        """
        Estimate the system delay using the crosstalk between TX and RX
        
        Args:
            rx_signal: Complex received signal array [num_rx, num_chirps, samples_per_chirp]
            
        Returns:
            Estimated system delay in seconds
        """
        # Detect the crosstalk timing
        crosstalk_samples, crosstalk_seconds = self.detect_crosstalk_timing(rx_signal)
        
        # The crosstalk represents the internal delay of the system
        # Store this as the system delay
        self.system_delay = crosstalk_seconds
        
        return self.system_delay