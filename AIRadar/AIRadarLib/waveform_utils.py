import numpy as np

def generate_linear_chirp(t, start_freq, slope):
    """
    Generate a linear chirp signal.
    
    Args:
        t: Time array
        start_freq: Starting frequency in Hz
        slope: Chirp slope in Hz/s
        
    Returns:
        signal: Complex chirp signal
        inst_freq: Instantaneous frequency
        phase_array: Phase array
    """
    # Linear chirp phase calculation
    phase = 2 * np.pi * (start_freq * t + (slope * t * t) / 2)
    signal = np.cos(phase)
    
    # Calculate instantaneous frequency from phase derivative
    unwrapped_phase = np.unwrap(phase)
    inst_freq = np.gradient(unwrapped_phase, t) / (2 * np.pi)
    
    return signal, inst_freq, phase

def generate_sawtooth_chirp(t, start_freq, bandwidth, chirp_duration_sec, num_saws=3):
    """
    Generate a sawtooth chirp signal.
    
    Args:
        t: Time array
        start_freq: Starting frequency in Hz
        bandwidth: Bandwidth in Hz
        chirp_duration_sec: Chirp duration in seconds
        num_saws: Number of sawtooth segments
        
    Returns:
        signal: Complex chirp signal
        inst_freq: Instantaneous frequency
        phase_array: Phase array
    """
    saw_duration = chirp_duration_sec / num_saws
    signal = np.zeros_like(t)
    inst_freq = np.zeros_like(t)
    phase_array = np.zeros_like(t)
    slope = bandwidth / saw_duration
    
    for i in range(len(t)):
        saw_idx = int(t[i] / saw_duration)
        if saw_idx >= num_saws:
            saw_idx = num_saws - 1
            
        rel_t = t[i] - saw_idx * saw_duration
        norm_t = rel_t / saw_duration
        
        # Instantaneous frequency for this sawtooth
        inst_freq[i] = start_freq + bandwidth * norm_t
        
        # Phase calculation for this sawtooth segment
        phase = 2 * np.pi * (start_freq * rel_t + (slope * norm_t * norm_t * saw_duration) / 2)
        phase_array[i] = phase
        signal[i] = np.cos(phase)
    
    # Calculate instantaneous frequency from phase derivative
    unwrapped_phase = np.unwrap(phase_array)
    inst_freq = np.gradient(unwrapped_phase, t) / (2 * np.pi)
    
    return signal, inst_freq, phase_array

def generate_triangular_chirp(t, start_freq, end_freq, chirp_duration_sec):
    """
    Generate a triangular chirp signal.
    
    Args:
        t: Time array
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        chirp_duration_sec: Chirp duration in seconds
        
    Returns:
        signal: Complex chirp signal
        inst_freq: Instantaneous frequency
        phase_array: Phase array
    """
    half_duration = chirp_duration_sec / 2
    signal = np.zeros_like(t)
    inst_freq = np.zeros_like(t)
    phase_array = np.zeros_like(t)
    slope = (end_freq - start_freq) / half_duration
    
    for i in range(len(t)):
        if t[i] <= half_duration:
            # Up chirp
            norm_t = t[i] / half_duration
            inst_freq[i] = start_freq + (end_freq - start_freq) * norm_t
            phase = 2 * np.pi * (start_freq * t[i] + (slope * t[i] * t[i]) / 2)
        else:
            # Down chirp
            rel_t = t[i] - half_duration
            norm_t = 1 - (rel_t / half_duration)
            inst_freq[i] = start_freq + (end_freq - start_freq) * norm_t
            
            down_chirp_start_freq = end_freq
            down_chirp_slope = -slope
            phase = 2 * np.pi * (down_chirp_start_freq * rel_t + (down_chirp_slope * rel_t * rel_t) / 2)
            
        phase_array[i] = phase
        signal[i] = np.cos(phase)
    
    # Calculate instantaneous frequency from phase derivative
    unwrapped_phase = np.unwrap(phase_array)
    inst_freq = np.gradient(unwrapped_phase, t) / (2 * np.pi)
    
    return signal, inst_freq, phase_array

def generate_spectrum(freq_range, start_freq, end_freq, bandwidth, waveform_type):
    """
    Generate a simulated spectrum for different waveform types.
    
    Args:
        freq_range: Frequency range array
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        bandwidth: Bandwidth in Hz
        waveform_type: Type of waveform ('linear', 'sawtooth', 'triangular')
        
    Returns:
        spectrum: Simulated spectrum
    """
    spectrum = np.zeros_like(freq_range, dtype=float)
    
    for i, freq in enumerate(freq_range):
        # Check if frequency is within the chirp bandwidth
        normalized_freq = (freq - start_freq) / bandwidth
        
        if waveform_type == 'linear':
            # Linear chirp has relatively flat spectrum within bandwidth
            if start_freq <= freq <= end_freq:
                # Main lobe
                magnitude = 0.9
                
                # Add some rolloff at the edges
                if normalized_freq < 0.05 or normalized_freq > 0.95:
                    magnitude *= 0.8
            else:
                # Side lobes
                distance_from_band = min(
                    abs(freq - start_freq),
                    abs(freq - end_freq)
                ) / bandwidth
                
                magnitude = max(0.1, 0.3 * np.exp(-5 * distance_from_band))
                
        elif waveform_type == 'sawtooth':
            # Sawtooth has harmonics and more side lobes
            if start_freq <= freq <= end_freq:
                # Main lobe
                magnitude = 0.85
                
                # Add harmonics
                harmonic_spacing = bandwidth / 3
                if abs((freq - start_freq) % harmonic_spacing) < bandwidth / 200:
                    magnitude = 0.95
            else:
                # Side lobes with harmonics
                distance_from_band = min(
                    abs(freq - start_freq),
                    abs(freq - end_freq)
                ) / bandwidth
                
                magnitude = max(0.15, 0.4 * np.exp(-3 * distance_from_band))
                
                # Add harmonics outside the band
                harmonic_spacing = bandwidth / 3
                dist_from_harmonic = min(
                    abs(((freq - start_freq) % harmonic_spacing) / harmonic_spacing),
                    abs(1 - ((freq - start_freq) % harmonic_spacing) / harmonic_spacing)
                )
                
                if dist_from_harmonic < 0.05:
                    magnitude += 0.2
                    
        elif waveform_type == 'triangular':
            # Triangular has smoother spectrum with less side lobes
            if start_freq <= freq <= end_freq:
                # Main lobe with smooth shape
                magnitude = 0.9 * (1 - 0.3 * np.power(2 * normalized_freq - 1, 2))
            else:
                # Smoother side lobes
                distance_from_band = min(
                    abs(freq - start_freq),
                    abs(freq - end_freq)
                ) / bandwidth
                
                magnitude = max(0.05, 0.25 * np.exp(-4 * distance_from_band))
        
        spectrum[i] = magnitude
    
    return spectrum


def generate_fmcw_chirp_signal(num_chirps, total_samples_per_chirp, active_samples, sample_rate, 
                              slope, tx_power=1.0, edge_ratio=0.1, window_type='edge'):
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

    Returns:
        continuous_signal: The continuous signal before reshaping
        chirp_duration: Duration of each chirp in seconds
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
    
    return continuous_signal, chirp_duration

# The new function models several important characteristics of the ADF4159 PLL hardware:

# 1. PLL Frequency Quantization - Models the 25-bit fractional modulator of the ADF4159, which limits the frequency resolution based on the reference frequency
# 2. PLL Settling Time - Simulates the non-instantaneous frequency transitions at the beginning of each chirp with an exponential settling curve
# 3. Phase Noise - Adds colored phase noise with a -20 dB/decade slope, typical of PLL synthesizers
# 4. Reference Spurs - Simulates reference spurs at multiples of the reference frequency
# 5. Frequency Deviation - Models crystal accuracy variations in parts per million
# 6. Configurable PLL Parameters - Allows configuration of reference frequency, reference doubler, and divider factor
def generate_adf4159_fmcw_chirp(num_chirps, total_samples_per_chirp, active_samples, sample_rate,
                               start_freq, bandwidth, chirp_duration=None, tx_power=1.0,
                               pll_ref_freq=100e6, pll_ref_doubler=False, pll_ref_div_factor=1,
                               phase_noise_level=-85, freq_deviation_ppm=1.5,
                               edge_ratio=0.1, window_type='edge'):
    """
    Generate FMCW chirp signal that simulates the ADF4159 phase-locked loop (PLL) hardware device.
    This function models realistic hardware effects including:
    - Phase noise
    - PLL frequency quantization
    - Non-linear frequency ramp behavior
    - PLL settling time
    - Reference spurs
    
    The ADF4159 is a fractional-N PLL frequency synthesizer commonly used in FMCW radar applications.
    
    Args:
        num_chirps: Number of chirps to generate
        total_samples_per_chirp: Total samples per chirp including idle time
        active_samples: Number of active samples in each chirp
        sample_rate: Sampling rate in Hz
        start_freq: Starting frequency of the chirp in Hz
        bandwidth: Bandwidth of the chirp in Hz
        chirp_duration: Duration of each chirp in seconds (if None, calculated from active_samples)
        tx_power: Transmission power scale (float)
        pll_ref_freq: PLL reference frequency in Hz (typically 100 MHz)
        pll_ref_doubler: Whether to use the reference doubler (doubles the reference frequency)
        pll_ref_div_factor: Reference divider factor (1, 2, 4, 8, 16, 32)
        phase_noise_level: Phase noise level in dBc/Hz at 100 kHz offset
        freq_deviation_ppm: Frequency deviation in parts per million
        edge_ratio: Proportion of chirp to taper at each edge (0–0.5)
        window_type: Windowing function ('edge', 'hann', 'hamming', or None)
        
    Returns:
        continuous_signal: The continuous signal before reshaping
        chirp_duration: Duration of each chirp in seconds
    """
    # Calculate chirp duration if not provided
    if chirp_duration is None:
        chirp_duration = active_samples / sample_rate
    
    # Calculate slope from bandwidth and chirp duration
    slope = bandwidth / chirp_duration
    
    # Create continuous time vector for entire frame
    t_frame = np.arange(num_chirps * total_samples_per_chirp) / sample_rate
    
    # Calculate effective reference frequency based on doubler and divider
    effective_ref_freq = pll_ref_freq
    if pll_ref_doubler:
        effective_ref_freq *= 2
    effective_ref_freq /= pll_ref_div_factor
    
    # ADF4159 has a 25-bit fractional modulator
    # Calculate frequency resolution (minimum frequency step)
    freq_resolution = effective_ref_freq / (2**25)
    
    # Simulate PLL settling time at the beginning of each chirp
    # Typically 5-20 microseconds for ADF4159
    settling_time = 10e-6  # 10 microseconds
    settling_samples = int(settling_time * sample_rate)
    
    # Generate ideal frequency ramp for each chirp
    freq_ramp = np.zeros(num_chirps * total_samples_per_chirp)
    for i in range(num_chirps):
        chirp_start = i * total_samples_per_chirp
        chirp_end = chirp_start + active_samples
        
        # Time vector for this chirp
        t_chirp = t_frame[chirp_start:chirp_end] - t_frame[chirp_start]
        
        # Ideal frequency ramp
        ideal_freq = start_freq + slope * t_chirp
        
        # Apply frequency quantization due to PLL resolution
        #The ideal chirp frequency ramp is quantized to simulate the effect of hardware resolution limits:
        quantized_freq = np.round(ideal_freq / freq_resolution) * freq_resolution
        
        # Apply settling behavior at the beginning of each chirp
        if settling_samples > 0 and settling_samples < active_samples:
            # Create settling curve (exponential approach)
            #PLL Settling Time: Models the initial transient response when switching frequencies:
            #A smooth exponential ramp approximates analog behavior in the first few samples.
            settling_factor = 1 - np.exp(-5 * np.arange(settling_samples) / settling_samples)
            freq_error = quantized_freq[settling_samples] - quantized_freq[0]
            quantized_freq[:settling_samples] = quantized_freq[0] + freq_error * settling_factor
        
        # Add frequency deviation (crystal accuracy)
        if freq_deviation_ppm > 0:
            #Frequency Deviation (Crystal Inaccuracy)
	        #Random multiplicative scaling simulates PPM offset due to imperfect crystal oscillators:
            # Convert ppm to fractional deviation
            deviation_factor = 1 + np.random.normal(0, freq_deviation_ppm / 1e6)
            quantized_freq *= deviation_factor
        
        # Store in the full ramp
        freq_ramp[chirp_start:chirp_end] = quantized_freq
    
    # Calculate phase by integrating frequency
    # Phase is the integral of frequency over time
    phase = 2 * np.pi * np.cumsum(freq_ramp) / sample_rate
    
    # Add phase noise (colored noise with -20 dB/decade slope)
    if phase_noise_level < 0:
        # Convert dBc/Hz to linear scale
        noise_power = 10**(phase_noise_level/10)
        
        # Generate white noise
        white_noise = np.random.normal(0, np.sqrt(noise_power), len(phase))
        
        # Apply filter to create colored noise (-20 dB/decade slope)
        # Simple first-order filter
        colored_noise = np.zeros_like(white_noise)
        alpha = 0.995  # Filter coefficient for -20 dB/decade slope
        for i in range(1, len(colored_noise)):
            colored_noise[i] = alpha * colored_noise[i-1] + (1-alpha) * white_noise[i]
        
        # Scale noise to desired level
        colored_noise *= 0.1  # Scaling factor
        
        # Add phase noise to signal phase
        phase += colored_noise
    
    # Add reference spurs
    # Reference spurs occur at multiples of the reference frequency
    spur_level = -70  # dBc
    spur_amplitude = 10**(spur_level/20)
    
    # Add spurs at reference frequency and harmonics
    for harmonic in [1, 2, 3]:  # First three harmonics
        spur_freq = effective_ref_freq * harmonic
        if spur_freq < sample_rate / 2:  # Only add if below Nyquist
            spur_phase = 2 * np.pi * spur_freq * t_frame
            phase += spur_amplitude * np.sin(spur_phase) / harmonic  # Amplitude decreases with harmonic number
    
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
    
    return continuous_signal, chirp_duration

def generate_sine_wave(freq, duration, fs):
    """
    Generate a complex sine wave
    
    Args:
        freq: Frequency of the sine wave in Hz
        duration: Duration of the signal in seconds
        fs: Sampling frequency in Hz
        
    Returns:
        Complex sine wave signal
    """
    t = np.arange(int(duration * fs)) / fs
    return np.exp(1j * 2 * np.pi * freq * t)


def generate_ofdm_signal(num_subcarriers, num_symbols, subcarrier_spacing, fs, cp_length_ratio=0.25):
    """
    Generate a simple OFDM signal
    
    Args:
        num_subcarriers: Number of subcarriers
        num_symbols: Number of OFDM symbols
        subcarrier_spacing: Spacing between subcarriers in Hz
        fs: Sampling frequency in Hz
        cp_length_ratio: Cyclic prefix length as a ratio of the symbol length
        
    Returns:
        OFDM signal
    """
    # Calculate FFT size and CP length
    fft_size = int(fs / subcarrier_spacing)
    cp_length = int(fft_size * cp_length_ratio)
    symbol_length = fft_size + cp_length
    
    # Generate random data for each subcarrier and symbol
    # Use QPSK modulation (2 bits per symbol)
    bits = np.random.randint(0, 4, size=(num_symbols, num_subcarriers))
    qpsk_symbols = np.exp(1j * bits * np.pi/2)
    
    # Create OFDM signal
    ofdm_signal = np.zeros(num_symbols * symbol_length, dtype=complex)
    
    for i in range(num_symbols):
        # Place subcarriers symmetrically around DC
        subcarrier_data = np.zeros(fft_size, dtype=complex)
        start_idx = (fft_size - num_subcarriers) // 2
        subcarrier_data[start_idx:start_idx + num_subcarriers] = qpsk_symbols[i]
        
        # Perform IFFT
        time_signal = np.fft.ifft(subcarrier_data) * np.sqrt(fft_size)
        
        # Add cyclic prefix
        symbol_with_cp = np.concatenate([time_signal[-cp_length:], time_signal])
        
        # Add to output signal
        ofdm_signal[i * symbol_length:(i + 1) * symbol_length] = symbol_with_cp
    
    return ofdm_signal

