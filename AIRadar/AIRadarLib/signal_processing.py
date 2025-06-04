import numpy as np

def fmcw_demodulate(tx_full, rx_full, total_samples_per_chirp, beat_samples_per_chirp, num_chirps):
    """
    Extract beat signals by dechirping Rx with Tx for each chirp.
    
    Args:
        tx_full: Transmitted signal
        rx_full: Received signal
        total_samples_per_chirp: Total samples per chirp including idle time
        beat_samples_per_chirp: Number of active samples in each chirp
        num_chirps: Number of chirps
        
    Returns:
        beat_signals: Beat signals with shape [num_chirps, beat_samples_per_chirp]
    """
    beat_signals = np.zeros((num_chirps, beat_samples_per_chirp), dtype=complex)
    for i in range(num_chirps):
        start = i * total_samples_per_chirp
        end = start + beat_samples_per_chirp
        if end > len(tx_full): continue
        beat_signals[i] = rx_full[start:end] * np.conj(tx_full[start:end])

    return beat_signals

def time_to_range_doppler(rx_signal,
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
        num_chirps: Number of chirps
        samples_per_chirp: Number of samples per chirp
        num_doppler_bins: Number of Doppler bins for FFT
        num_range_bins: Number of range bins for FFT
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