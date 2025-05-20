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