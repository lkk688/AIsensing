import numpy as np
import matplotlib.pyplot as plt
import os


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

def plot_range_doppler_map_with_ground_truth(
    rd_map,
    targets,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    title_prefix='',
    save_path=None
):
    """
    Plot a 2D Range-Doppler map with ground truth target annotations.

    Args:
        rd_map: np.ndarray, shape (2, num_doppler_bins, num_range_bins), real and imaginary parts of RD map
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        title_prefix: str, prefix for plot title/labeling/saving)
        save_path: str, directory to save the figure
    """
    rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
    rd_db = 20 * np.log10(rd_magnitude + 1e-10)
    vmin = np.max(rd_db) - 40  # Dynamic range of 40 dB

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Range-Doppler Map with Ground Truth - {title_prefix}", fontsize=16)
    plt.imshow(rd_db, aspect='auto', cmap='jet', vmin=vmin)
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title('Range-Doppler Map')

    for i, target in enumerate(targets):
        # Calculate range and Doppler bins for each target
        range_bin = int(target['distance'] / range_resolution)
        doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        if (0 <= range_bin < num_range_bins and 0 <= doppler_bin < num_doppler_bins):
            plt.plot(range_bin, doppler_bin, 'ro', markersize=10, markeredgecolor='white')
            plt.text(
                range_bin + 2, doppler_bin,
                f"Target {i+1}\nR: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s\nRCS: {target['rcs']:.1f}m²",
                color='white', fontsize=9, backgroundcolor='black',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round')
            )
    # if len(targets) == 0:
    #     plt.text(
    #         num_range_bins//2, num_doppler_bins//2,
    #         "No targets in this scene",
    #         color='white', fontsize=12, backgroundcolor='red',
    #         ha='center', va='center'
    #     )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(os.path.join(save_path, f'sample_{sample_idx}_rd_map_2d.png'))
    # plt.close()

def plot_3d_range_doppler_map_with_ground_truth(
    rd_map,
    targets,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    title_prefix='',
    save_path=None
):
    """
    Plot a 3D Range-Doppler map with ground truth target annotations.

    Args:
        rd_map: np.ndarray, shape (2, num_doppler_bins, num_range_bins), real and imaginary parts of RD map
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        sample_idx: int, index of the current sample (for labeling/saving)
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        save_path: str, directory to save the figure
    """
    rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
    rd_db = 20 * np.log10(rd_magnitude + 1e-10)
    X, Y = np.meshgrid(np.arange(num_range_bins), np.arange(num_doppler_bins))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, rd_db, cmap='jet', linewidth=0, antialiased=True)
    ax.set_xlabel('Range Bin')
    ax.set_ylabel('Doppler Bin')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title('3D Range-Doppler Map with Ground Truth')
    for target in targets:
        range_bin = int(target['distance'] / range_resolution)
        doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        if (0 <= range_bin < num_range_bins and 0 <= doppler_bin < num_doppler_bins):
            z_val = rd_db[doppler_bin, range_bin]
            ax.scatter([range_bin], [doppler_bin], [z_val], color='r', s=50, marker='o')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(os.path.join(save_path, f'sample_{sample_idx}_rd_map_3d.png'))
    # plt.close(fig)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_cfar_vs_ground_truth(
    targets,
    detection_results,
    sample_idx,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    save_path,
    create_target_mask_func
):
    """
    Plot CFAR detection results versus ground truth target mask.

    Args:
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        detection_results: list of dicts, each with keys 'range_bin', 'doppler_bin', 'distance', 'velocity', 'snr'
        sample_idx: int, index of the current sample (for labeling/saving)
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        save_path: str, directory to save the figure
        create_target_mask_func: function, generates a target mask from targets
    """
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"CFAR Detection vs Ground Truth - Sample {sample_idx}", fontsize=16)
    # Ground truth target mask
    plt.subplot(2, 1, 1)
    target_mask = create_target_mask_func(targets)
    plt.imshow(target_mask[:, :, 0], aspect='auto', cmap='gray')
    plt.colorbar(label='Target Presence')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title('Ground Truth Target Mask')
    for target in targets:
        range_bin = int(target['distance'] / range_resolution)
        doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        if (0 <= range_bin < num_range_bins and 0 <= doppler_bin < num_doppler_bins):
            plt.plot(range_bin, doppler_bin, 'ro', markersize=8)
            plt.text(
                range_bin + 1, doppler_bin + 1,
                f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s",
                color='white', fontsize=8, backgroundcolor='black'
            )
    # CFAR detection results
    plt.subplot(2, 1, 2)
    cfar_map = np.zeros((num_doppler_bins, num_range_bins))
    for target in detection_results:
        cfar_map[target['doppler_bin'], target['range_bin']] = 1
    plt.imshow(cfar_map, aspect='auto', cmap='gray')
    plt.colorbar(label='Detection')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title('CFAR Detection Results')
    for target in detection_results:
        plt.plot(target['range_bin'], target['doppler_bin'], 'bo', markersize=8)
        plt.text(
            target['range_bin'] + 1, target['doppler_bin'] + 1,
            f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s\nSNR: {target['snr']:.1f}dB",
            color='white', fontsize=8, backgroundcolor='blue'
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'sample_{sample_idx}_detection.png'))
    plt.close()

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

def plot_signal_time_and_spectrum(
    signal,
    sample_rate,
    total_duration,
    title_prefix="Signal",
    window_type="blackman",
    N_fft=8192,
    bandwidth=None,
    center_freq=None,
    zoom_margin=0,
    textstr=None,
    highlight_peak=True,
    normalize=True,
    save_path=None,
    draw_window=True
):
    """
    Plot the time-domain and spectrum of a complex signal with advanced options.

    Args:
        signal: np.ndarray, 1D complex array (the signal to plot)
        sample_rate: float, sample rate in Hz
        total_duration: float, total duration of the signal in seconds
        title_prefix: str, prefix for plot titles
        window_type: str, type of window to use for spectrum ('blackman', 'hamming', 'hann', 'rect')
        N_fft: int, FFT size
        bandwidth: float or None, bandwidth in Hz to highlight and zoom
        center_freq: float or None, center frequency in Hz to highlight region (used with bandwidth)
        zoom_margin: float, fraction of bandwidth for zoom margin
        textstr: str or None, text to display on spectrum plot
        highlight_peak: bool, whether to highlight the peak frequency
        normalize: bool, whether to normalize spectra for comparison
        save_path: str or None, if provided, save the figure to this path
        draw_window: bool, whether to draw the window in the time-domain plot
    """
    t = np.linspace(0, total_duration, len(signal))
    signal_windowed, window = apply_window(signal, window_type=window_type)
    if center_freq is not None:
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate)) + center_freq
    else:
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate))
    freq_axis = freq_axis / 1e6  # MHz

    # Compute spectra
    spectrum_orig = calculate_spectrum(signal, N_fft)
    spectrum_win = calculate_spectrum(signal_windowed, N_fft)

    # Normalize if requested
    if normalize:
        ref = max(np.max(spectrum_orig), np.max(spectrum_win))
        spectrum_orig = normalize_spectrum(spectrum_orig, reference=ref)
        spectrum_win = normalize_spectrum(spectrum_win, reference=ref)

    # Find peak frequency
    peak_freq, peak_val = find_peak_frequency(spectrum_win, freq_axis) if highlight_peak else (None, None)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{title_prefix} - Time and Spectrum", fontsize=14)

    # Time domain
    axs[0].plot(t, np.real(signal), 'b-', label='Real', alpha=0.7)
    axs[0].plot(t, np.imag(signal), 'r--', label='Imag', alpha=0.7)
    if draw_window:
        axs[0].plot(t, window * np.max(np.abs(signal)), 'g-', label='Window', alpha=0.3)
    axs[0].set_title(f"{title_prefix} (Time Domain)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True)

    # Spectrum
    axs[1].plot(freq_axis, spectrum_orig, 'b-', label='Original', alpha=0.5)
    if draw_window:
        axs[1].plot(freq_axis, spectrum_win, 'r-', label=f'{window_type.capitalize()} Window', alpha=0.8)
    axs[1].set_title(f"{title_prefix} Spectrum")
    axs[1].set_xlabel("Frequency (MHz)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].grid(True)
    axs[1].legend()

    # Highlight bandwidth region and zoom
    if bandwidth is not None and center_freq is not None:
        bandwidth_mhz = bandwidth / 1e6
        center_freq_mhz = center_freq / 1e6
        f_start = center_freq_mhz - bandwidth_mhz / 2
        f_end = center_freq_mhz + bandwidth_mhz / 2
        axs[1].axvline(f_start, color='red', linestyle='--', linewidth=1.5, label=f'Start: {f_start:.2f} MHz')
        axs[1].axvline(f_end, color='green', linestyle='--', linewidth=1.5, label=f'End: {f_end:.2f} MHz')
        axs[1].axvspan(f_start, f_end, alpha=0.2, color='yellow', label='Highlighted Region')
        margin = bandwidth_mhz * zoom_margin
        axs[1].set_xlim([f_start - margin, f_end + margin])
    elif bandwidth is not None and zoom_margin > 0:
        bandwidth_mhz = bandwidth / 1e6
        f_start = -bandwidth_mhz / 2
        f_end = bandwidth_mhz / 2
        axs[1].axvline(f_start, color='red', linestyle='--', linewidth=1.5, label=f'Start: {f_start:.2f} MHz')
        axs[1].axvline(f_end, color='green', linestyle='--', linewidth=1.5, label=f'End: {f_end:.2f} MHz')
        axs[1].axvspan(f_start, f_end, alpha=0.2, color='yellow', label='Highlighted Region')
        margin = bandwidth_mhz * zoom_margin
        axs[1].set_xlim([f_start - margin, f_end + margin])

    # Highlight peak frequency
    if highlight_peak and peak_freq is not None:
        axs[1].axvline(peak_freq, color='magenta', linestyle='-', linewidth=2, label=f'Peak: {peak_freq:.2f} MHz')
        axs[1].annotate(f'Peak: {peak_freq:.2f} MHz', xy=(peak_freq, peak_val), xytext=(peak_freq, peak_val+5),
                        arrowprops=dict(facecolor='magenta', shrink=0.05), fontsize=10, color='magenta')

    # Display text string
    if textstr is not None:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[1].text(0.05, 0.95, textstr, transform=axs[1].transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


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