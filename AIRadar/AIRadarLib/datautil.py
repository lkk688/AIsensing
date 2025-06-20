import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sp_signal

def upconvert_signal(signal, center_freq, sample_rate):
    """
    Upconvert a baseband signal to the specified center frequency.
    """
    n = np.arange(len(signal))
    t = n / sample_rate
    return signal * np.exp(1j * 2 * np.pi * center_freq * t)

def downconvert_signal(signal, center_freq, sample_rate):
    """
    Downconvert an RF signal to baseband.
    """
    # signal shape: (num_rx, num_samples) or (num_samples,)
    if signal.ndim == 1:
        n = np.arange(len(signal))
        t = n / sample_rate
        carrier = np.exp(-1j * 2 * np.pi * center_freq * t)
        return signal * carrier
    elif signal.ndim == 2:
        n = np.arange(signal.shape[1])
        t = n / sample_rate
        carrier = np.exp(-1j * 2 * np.pi * center_freq * t)[None, :]  # shape (1, N)
        return signal * carrier  # broadcasts over first axis
    else:
        raise ValueError("Signal must be 1D or 2D array")

def calculate_radar_parameters(
    sample_rate,
    chirp_duration,
    center_freq,
    bandwidth,
    num_chirps,
):
    """
    Calculate radar performance metrics and validate system parameters.

    Parameters
    ----------
    sample_rate : float
        ADC sample rate in Hz.
        - Impacts: samples_per_chirp, max_range, range_fft_size.
        - Higher sample_rate allows capturing higher beat frequencies (wider unambiguous range).
        - Nyquist limit: $f_{Nyquist} = \\frac{f_s}{2}$

    chirp_duration : float
        Duration of a single FMCW chirp in seconds.
        - Impacts: samples_per_chirp, velocity_resolution, max_unambiguous_velocity.
        - Longer chirp_duration increases velocity resolution but reduces max unambiguous velocity.

    center_freq : float
        Center frequency of the radar in Hz.
        - Impacts: wavelength, velocity calculations.
        - $\\lambda = \\frac{c}{f_c}$

    bandwidth : float
        FMCW chirp bandwidth in Hz.
        - Impacts: range_resolution, max_range.
        - Larger bandwidth improves range resolution.

    num_chirps : int
        Number of chirps per frame.
        - Impacts: velocity_resolution, doppler_fft_size.
        - More chirps improve velocity resolution.

    Returns
    -------
    dict
        Dictionary containing performance metrics:
        - max_range
        - range_resolution
        - max_velocity
        - velocity_resolution
        - max_unambiguous_velocity
        - fmcw_slope
        - samples_per_chirp
        - range_fft_size
        - doppler_fft_size
        - wavelength
    """

    c = 3e8  # Speed of light (m/s)
    λ = c / center_freq  # Wavelength (m), impacts velocity calculations

    # --- Performance Metrics ---
    # Number of samples per chirp
    # $N_{samples/chirp} = f_s \cdot T_{chirp}$
    samples_per_chirp = int(sample_rate * chirp_duration)

    # Range resolution (meters)
    # $\\Delta R = \\frac{c}{2B}$
    range_resolution = c / (2 * bandwidth)

    # Maximum range (meters)
    # $R_{max} = \\frac{f_s \cdot c \cdot T_{chirp}}{2B}$
    max_range = (sample_rate * c * chirp_duration) / (2 * bandwidth)

    # Velocity resolution (m/s)
    # $\\Delta v = \\frac{\\lambda}{2 N_{chirps} T_{chirp}}$
    velocity_resolution = λ / (2 * num_chirps * chirp_duration)

    # Maximum unambiguous velocity (m/s)
    # $v_{max,unamb} = \\frac{\\lambda}{4 T_{chirp}}$
    max_unambiguous_velocity = λ / (4 * chirp_duration)

    # FMCW Slope (Hz/s)
    # $S = \\frac{B}{T_{chirp}}$
    fmcw_slope = bandwidth / chirp_duration

    # FFT sizes (powers of 2 for efficiency)
    range_fft_size = 2 ** int(np.ceil(np.log2(samples_per_chirp)))
    doppler_fft_size = 2 ** int(np.ceil(np.log2(num_chirps)))

    wavelength = λ

    # --- Validation and Warnings ---
    # Nyquist check for beat frequency
    # $f_{beat,max} = \\frac{2 R_{max} B}{c T_{chirp}}$
    f_beat_max = (2 * max_range * bandwidth) / (c * chirp_duration)
    nyquist_freq = sample_rate / 2
    if f_beat_max > nyquist_freq:
        print(f"Warning: Beat frequency ({f_beat_max/1e6:.2f} MHz) exceeds Nyquist ({nyquist_freq/1e6:.2f} MHz). Increase sample_rate or chirp_duration.")
        # If f_beat_max > nyquist, range performance will be aliased.

    # Bandwidth check
    if bandwidth > sample_rate / 2:
        print(f"Warning: Bandwidth ({bandwidth/1e6:.2f} MHz) exceeds Nyquist ({(sample_rate/2)/1e6:.2f} MHz).")
        # If bandwidth > Nyquist, range bins will be aliased.

    # Slope hardware constraint (example: 80 THz/s)
    max_rf_slope = 80e12
    if fmcw_slope > max_rf_slope:
        print(f"Warning: FMCW slope ({fmcw_slope/1e12:.2f} THz/s) exceeds hardware limit ({max_rf_slope/1e12:.2f} THz/s).")
        # If slope > hardware limit, chirp cannot be generated.

    # RF sweep band constraint (example: ±0.5 GHz around center)
    rf_min_freq = center_freq - 0.5e9
    rf_max_freq = center_freq + 0.5e9
    f_start = center_freq - bandwidth / 2
    f_end = center_freq + bandwidth / 2
    if f_start < rf_min_freq or f_end > rf_max_freq:
        print(f"Warning: RF sweep range {f_start/1e9:.2f}–{f_end/1e9:.2f} GHz exceeds hardware limits ({rf_min_freq/1e9:.2f}–{rf_max_freq/1e9:.2f} GHz).")
        # If sweep band exceeds hardware, signal will be clipped or distorted.

    # Chirp duration check
    if chirp_duration < 1e-6 or chirp_duration > 1e-2:
        print(f"Warning: Chirp duration ({chirp_duration*1e6:.2f} us) is outside typical range (1 us – 10 ms).")
        # Too short: not enough samples; too long: slow update rate.

    # Print summary with equations
    print("=== Radar Performance Metrics ===")
    print(f"Range Resolution      : {range_resolution:.3f} m   (ΔR = c/(2B))")
    print(f"Max Range             : {max_range:.2f} m   (R_max = (f_s·c·T_c)/(2B))")
    print(f"Velocity Resolution   : {velocity_resolution:.3f} m/s   (Δv = λ/(2·N·T_c))")
    print(f"Max Unambiguous Vel.  : {max_unambiguous_velocity:.2f} m/s   (v_max,unamb = λ/(4·T_c))")
    print(f"FMCW Slope            : {fmcw_slope/1e12:.2f} THz/s   (S = B/T_c)")
    print(f"Samples per Chirp     : {samples_per_chirp}   (N_samples/chirp = f_s·T_c)")
    print(f"Range FFT Size        : {range_fft_size}")
    print(f"Doppler FFT Size      : {doppler_fft_size}")
    print(f"Wavelength            : {wavelength*1e3:.2f} mm   (λ = c/f_c)")
    print("=================================")

    return {
        "max_range": max_range,
        "range_resolution": range_resolution,
        "max_velocity": max_unambiguous_velocity,
        "velocity_resolution": velocity_resolution,
        "max_unambiguous_velocity": max_unambiguous_velocity,
        "fmcw_slope": fmcw_slope,
        "samples_per_chirp": samples_per_chirp,
        "range_fft_size": range_fft_size,
        "doppler_fft_size": doppler_fft_size,
        "wavelength": wavelength,
        "f_beat_max": f_beat_max,
        "nyquist_freq": nyquist_freq,
    }

def apply_window(signal, window_type="blackman"):
    """Apply a window function to the signal."""
    if window_type == "blackman":
        window = np.blackman(len(signal))
    elif window_type == "hamming":
        window = np.hamming(len(signal))
    elif window_type == "hann":
        window = np.hanning(len(signal))
    elif window_type == "rect":
        window = np.ones(len(signal))
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    return signal * window, window

def calculate_spectrum(signal, N_fft):
    """Compute the FFT and return the magnitude spectrum in dB."""
    fft_data = np.fft.fftshift(np.fft.fft(signal, n=N_fft))
    spectrum = 20 * np.log10(np.abs(fft_data) + 1e-10)
    return spectrum

def normalize_spectrum(spectrum, reference=None):
    """Normalize the spectrum for better comparison."""
    if reference is None:
        reference = np.max(spectrum)
    return spectrum - reference

def find_peak_frequency(spectrum, freq_axis):
    """Find the peak frequency and its value in the spectrum."""
    peak_idx = np.argmax(spectrum)
    peak_freq = freq_axis[peak_idx]
    peak_val = spectrum[peak_idx]
    return peak_freq, peak_val


def extract_chirp(signal, chirp_idx, total_samples_per_chirp, rx_idx=None):
    """
    Extract a single chirp from a signal array.

    Args:
        signal: np.ndarray, can be 1D (single channel), 2D (multi-channel, flattened), or 3D (rx, chirp, sample)
        chirp_idx: int, index of the chirp to extract
        total_samples_per_chirp: int, number of samples per chirp
        rx_idx: int or None, RX channel index (if applicable)

    Returns:
        np.ndarray: 1D complex array of the extracted chirp
    """
    if signal.ndim == 1:
        start = chirp_idx * total_samples_per_chirp
        end = start + total_samples_per_chirp
        return signal[start:end]
    elif signal.ndim == 2 and rx_idx is not None:
        start = chirp_idx * total_samples_per_chirp
        end = start + total_samples_per_chirp
        return signal[rx_idx, start:end]
    elif signal.ndim == 3 and rx_idx is not None:
        return signal[rx_idx, chirp_idx]
    else:
        raise ValueError("Unsupported signal shape for chirp extraction.")

def process_chirp_signal(chirp, N_fft, sample_rate, window_type='blackman'):
    """
    Apply windowing and FFT to a chirp signal.

    Args:
        chirp: np.ndarray, 1D complex array
        N_fft: int, FFT size
        sample_rate: float, sample rate in Hz
        window_type: str, type of window ('blackman' or 'rect')

    Returns:
        dict: {
            'chirp': original chirp,
            'chirp_windowed': windowed chirp,
            'spectrum_orig': dB spectrum (original),
            'spectrum_win': dB spectrum (windowed),
            'freq_axis': frequency axis in MHz,
            'window': window used
        }
    """
    if window_type == 'blackman':
        window = np.blackman(len(chirp))
    else:
        window = np.ones(len(chirp))
    chirp_windowed = chirp * window
    fft_orig = np.fft.fftshift(np.fft.fft(chirp, n=N_fft))
    fft_win = np.fft.fftshift(np.fft.fft(chirp_windowed, n=N_fft))
    spectrum_orig = data2db(fft_orig)
    spectrum_win = data2db(fft_win)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate)) / 1e6  # MHz
    return {
        'chirp': chirp,
        'chirp_windowed': chirp_windowed,
        'spectrum_orig': spectrum_orig,
        'spectrum_win': spectrum_win,
        'freq_axis': freq_axis,
        'window': window
    }

def data2db(fft_data):
    """
    Convert FFT data to dB magnitude spectrum.

    Args:
        fft_data: np.ndarray, FFT output (complex)

    Returns:
        np.ndarray: dB magnitude spectrum
    """
    return 20 * np.log10(np.abs(fft_data) + 1e-10)

def define_autonomous_driving_radar_parameters(radar_type='long_range'):
    """
    Define realistic radar parameters for autonomous driving applications.
    
    Args:
        radar_type: Type of automotive radar ('long_range', 'mid_range', or 'short_range')
        
    Returns:
        Dictionary containing radar parameters and performance metrics
    """
    # Common parameters for automotive radars
    center_freq = 77e9  # 77 GHz is standard for automotive radar
    
    if radar_type == 'long_range':
        # Long-range radar (LRR) for adaptive cruise control, collision warning
        # Typical range: 150-250m, narrow field of view
        sample_rate = 25e6       # 25 MHz
        bandwidth = 500e6        # 500 MHz for ~30cm range resolution
        chirp_duration = 40e-6   # 40 μs
        num_chirps = 128         # More chirps for better velocity resolution
        description = "Long-range radar for highway driving, adaptive cruise control"
        
    elif radar_type == 'mid_range':
        # Mid-range radar (MRR) for cross-traffic alert, lane change assist
        # Typical range: 60-100m, wider field of view
        sample_rate = 25e6       # 25 MHz
        bandwidth = 800e6        # 800 MHz for ~19cm range resolution
        chirp_duration = 30e-6   # 30 μs
        num_chirps = 64          # Balanced velocity resolution
        description = "Mid-range radar for lane change assist, cross-traffic alert"
        
    elif radar_type == 'short_range':
        # Short-range radar (SRR) for parking assist, blind spot detection
        # Typical range: 0.15-30m, very wide field of view
        sample_rate = 40e6       # 40 MHz
        bandwidth = 1.5e9        # 1.5 GHz for ~10cm range resolution
        chirp_duration = 20e-6   # 20 μs
        num_chirps = 32          # Fewer chirps, faster update rate
        description = "Short-range radar for parking assist, blind spot detection"
        
    else:
        raise ValueError(f"Unknown radar type: {radar_type}")
    
    # Calculate radar performance metrics
    print(f"\n=== {radar_type.upper()} RADAR PARAMETERS ===\n")
    print(f"Description: {description}")
    print(f"Center Frequency: {center_freq/1e9:.1f} GHz")
    print(f"Bandwidth: {bandwidth/1e6:.1f} MHz")
    print(f"Sample Rate: {sample_rate/1e6:.1f} MHz")
    print(f"Chirp Duration: {chirp_duration*1e6:.1f} μs")
    print(f"Number of Chirps: {num_chirps}")
    
    # Call the existing calculate_radar_parameters function
    radar_metrics = calculate_radar_parameters(
        sample_rate=sample_rate,
        chirp_duration=chirp_duration,
        center_freq=center_freq,
        bandwidth=bandwidth,
        num_chirps=num_chirps
    )
    
    # Add the input parameters to the results
    radar_metrics.update({
        "radar_type": radar_type,
        "description": description,
        "center_freq": center_freq,
        "bandwidth": bandwidth,
        "sample_rate": sample_rate,
        "chirp_duration": chirp_duration,
        "num_chirps": num_chirps
    })
    
    return radar_metrics

# Example usage
if __name__ == "__main__":
    # Calculate parameters for all radar types
    lrr_params = define_autonomous_driving_radar_parameters('long_range')
    print("\n" + "-"*50 + "\n")
    
    mrr_params = define_autonomous_driving_radar_parameters('mid_range')
    print("\n" + "-"*50 + "\n")
    
    srr_params = define_autonomous_driving_radar_parameters('short_range')