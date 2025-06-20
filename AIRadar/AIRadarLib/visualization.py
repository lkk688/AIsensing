import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy import signal
import sys

from AIRadarLib.datautil import normalize_spectrum, find_peak_frequency, apply_window, calculate_spectrum

def plot_detection_results(
    rd_map,
    target_mask,
    targets,
    detection_results,
    range_resolution,
    velocity_resolution,
    num_doppler_bins,
    num_range_bins,
    save_path=None,
    title="Radar Detection Results",
    show_plot=True,
    figsize=(12, 10),
    dpi=100,
    apply_doppler_centering=True
):
    """
    Plot radar detection results, target mask, and ground truth target locations in a single figure.
    
    Args:
        rd_map: Range-Doppler map with shape [num_rx, 2, num_doppler_bins, num_range_bins]
        target_mask: Target mask with shape [num_doppler_bins, num_range_bins, 1]
        targets: List of target dictionaries with 'distance' and 'velocity' keys
        detection_results: List of detection dictionaries with 'range_idx', 'doppler_idx', etc.
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        num_doppler_bins: Number of Doppler bins
        num_range_bins: Number of range bins
        save_path: Path to save the figure (if None, figure is not saved)
        title: Title of the figure
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) in inches
        dpi: DPI for the figure
        apply_doppler_centering: bool, whether Doppler and Range FFT are centered (affects target position calculation)
        
    Returns:
        Figure and axes objects
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Extract magnitude of range-Doppler map (use first RX antenna)
    rd_magnitude = np.sqrt(rd_map[0, 0, :, :]**2 + rd_map[0, 1, :, :]**2)
    
    # Normalize RD map for better visualization
    rd_magnitude_norm = 20 * np.log10(rd_magnitude / np.max(rd_magnitude) + 1e-10)
    rd_magnitude_norm = np.clip(rd_magnitude_norm, -40, 0)  # Clip to dynamic range
    
    # Create range and Doppler axes
    range_axis = np.arange(num_range_bins) * range_resolution
    doppler_axis = (np.arange(num_doppler_bins) - num_doppler_bins // 2) * velocity_resolution
    
    # Plot range-Doppler map
    im = ax.imshow(
        rd_magnitude_norm,
        aspect='auto',
        cmap='jet',
        origin='lower',
        extent=[0, range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        interpolation='none',
        vmin=-40,
        vmax=0
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    # Create a custom colormap for the target mask (transparent to red)
    colors = [(1, 0, 0, 0), (1, 0, 0, 0.7)]  # Red with varying alpha
    target_cmap = LinearSegmentedColormap.from_list('target_mask', colors)
    
    # Plot target mask as overlay with transparency
    if target_mask is not None:
        # Reshape if needed and transpose for correct orientation
        #Binary mask with shape [num_doppler_bins, num_range_bins, 1]
        mask_plot = target_mask.reshape(num_doppler_bins, num_range_bins)
        ax.imshow(
            mask_plot,
            aspect='auto',
            cmap=target_cmap,
            origin='lower',
            extent=[0, range_axis[-1], doppler_axis[0], doppler_axis[-1]],
            interpolation='none'
        )
    
    # Plot ground truth target locations
    if targets:
        # Use the actual distance and velocity values directly
        # No need to adjust for centering as we're plotting in physical units (meters and m/s)
        # not in bin indices
        target_ranges = [target['distance'] for target in targets]
        target_velocities = [target['velocity'] for target in targets]
        ax.scatter(
            target_ranges,
            target_velocities,
            c='lime',
            marker='o',
            s=100,
            edgecolors='black',
            linewidths=1.5,
            label='Ground Truth'
        )
    
    # Plot CFAR detection results
    if detection_results:
        detection_ranges = []
        detection_velocities = []
        
        for detection in detection_results:
            # Check if detection has range_idx and doppler_idx or range and velocity
            if 'range_idx' in detection and 'doppler_idx' in detection:
                range_val = detection['range_idx'] * range_resolution
                doppler_val = (detection['doppler_idx'] - num_doppler_bins // 2) * velocity_resolution
            elif 'range' in detection and 'velocity' in detection:
                range_val = detection['range']
                doppler_val = detection['velocity']
            else:
                continue
                
            detection_ranges.append(range_val)
            detection_velocities.append(doppler_val)
        
        if detection_ranges:
            ax.scatter(
                detection_ranges,
                detection_velocities,
                c='white',
                marker='x',
                s=80,
                linewidths=2,
                label='CFAR Detections'
            )
    
    # Set labels and title
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    # Set axis limits
    ax.set_xlim(0, range_axis[-1])
    ax.set_ylim(doppler_axis[0], doppler_axis[-1])
    
    # Add text with detection statistics
    if targets and detection_results:
        num_targets = len(targets)
        num_detections = len(detection_results)
        
        stats_text = f"Targets: {num_targets}\nDetections: {num_detections}"
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def plot_signal_comparison(original_signal, processed_signal, fs, title, time_domain_ylim=None):
    """
    Plot time and frequency domain comparison of original and processed signals
    
    Args:
        original_signal: The original input signal
        processed_signal: The signal after processing
        fs: Sampling frequency in Hz
        title: Plot title
        time_domain_ylim: Y-axis limits for time domain plot (optional)
    """
    # Create figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Time vector for plotting (only show a portion for clarity)
    plot_samples = min(1000, len(original_signal))
    t = np.arange(plot_samples) / fs
    
    # Time domain plots
    axs[0, 0].plot(t, np.real(original_signal[:plot_samples]), 'b-', label='Real')
    axs[0, 0].plot(t, np.imag(original_signal[:plot_samples]), 'r-', label='Imag')
    axs[0, 0].set_title('Original Signal (Time Domain)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    if time_domain_ylim:
        axs[0, 0].set_ylim(time_domain_ylim)
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(t, np.real(processed_signal[:plot_samples]), 'b-', label='Real')
    axs[0, 1].plot(t, np.imag(processed_signal[:plot_samples]), 'r-', label='Imag')
    axs[0, 1].set_title('Processed Signal (Time Domain)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].legend()
    if time_domain_ylim:
        axs[0, 1].set_ylim(time_domain_ylim)
    axs[0, 1].grid(True)
    
    # Frequency domain plots
    f_orig, Pxx_orig = signal.welch(original_signal, fs, nperseg=1024, return_onesided=False)
    f_orig = np.fft.fftshift(f_orig)
    Pxx_orig = np.fft.fftshift(Pxx_orig)
    
    f_proc, Pxx_proc = signal.welch(processed_signal, fs, nperseg=1024, return_onesided=False)
    f_proc = np.fft.fftshift(f_proc)
    Pxx_proc = np.fft.fftshift(Pxx_proc)
    
    # Convert to dB
    Pxx_orig_db = 10 * np.log10(Pxx_orig + 1e-10)
    Pxx_proc_db = 10 * np.log10(Pxx_proc + 1e-10)
    
    axs[1, 0].plot(f_orig, Pxx_orig_db)
    axs[1, 0].set_title('Original Signal (Frequency Domain)')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('Power Spectral Density (dB/Hz)')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(f_proc, Pxx_proc_db)
    axs[1, 1].set_title('Processed Signal (Frequency Domain)')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel('Power Spectral Density (dB/Hz)')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_range_doppler_map_with_ground_truth(
    rd_map,
    targets,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    title_prefix='',
    save_path=None,
    apply_doppler_centering=True
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
        apply_doppler_centering: bool, whether Doppler FFT is centered (affects target position calculation)
    """
    rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2) #(1024, 1024)
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
        if apply_doppler_centering:
            # When centered, zero range is at num_range_bins/2
            range_bin = int(num_range_bins // 2 + target['distance'] / range_resolution)
            # When centered, zero velocity is at num_doppler_bins/2
            doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        else:
            # When not centered, zero range is at bin 0
            range_bin = int(target['distance'] / range_resolution)
            # When not centered, zero velocity is at bin 0
            doppler_bin = int(target['velocity'] / velocity_resolution) % num_doppler_bins
            
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
    save_path=None,
    apply_doppler_centering=True
):
    """
    Plot a 3D Range-Doppler map with ground truth target annotations.

    Args:
        rd_map: np.ndarray, shape (2, num_doppler_bins, num_range_bins), real and imaginary parts of RD map
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        title_prefix: str, prefix for plot title/labeling/saving
        save_path: str, directory to save the figure
        apply_doppler_centering: bool, whether Doppler FFT is centered (affects target position calculation)
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
        if apply_doppler_centering:
            # When centered, zero range is at num_range_bins/2
            range_bin = int(num_range_bins // 2 + target['distance'] / range_resolution)
            # When centered, zero velocity is at num_doppler_bins/2
            doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        else:
            # When not centered, zero range is at bin 0
            range_bin = int(target['distance'] / range_resolution)
            # When not centered, zero velocity is at bin 0
            doppler_bin = int(target['velocity'] / velocity_resolution) % num_doppler_bins
            
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
    create_target_mask_func,
    apply_doppler_centering=True
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
        apply_doppler_centering: bool, whether Doppler and Range FFT are centered (affects target position calculation)
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
        if apply_doppler_centering:
            # When centered, zero range is at num_range_bins/2
            range_bin = int(num_range_bins // 2 + target['distance'] / range_resolution)
            # When centered, zero velocity is at num_doppler_bins/2
            doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        else:
            # When not centered, zero range is at bin 0
            range_bin = int(target['distance'] / range_resolution)
            # When not centered, zero velocity is at bin 0
            doppler_bin = int(target['velocity'] / velocity_resolution) % num_doppler_bins
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


def plot_instantaneous_frequency(signal, sample_rate, total_duration, slope, bandwidth=None, center_freq=None, 
                                title_prefix="Chirp", textstr=None, save_path=None, draw_window=False):
    """
    Plot the instantaneous frequency of a chirp signal to visualize the frequency sweeping band.
    
    Args:
        signal: Complex chirp signal
        sample_rate: Sample rate in Hz
        total_duration: Total duration of the signal in seconds
        slope: FMCW slope in Hz/s
        bandwidth: Signal bandwidth in Hz (optional)
        center_freq: Center frequency in Hz (optional)
        title_prefix: Prefix for the plot title
        textstr: Additional text to display on the plot
        save_path: Path to save the figure
        draw_window: Whether to draw window function
    """

    
    # Create time axis
    num_samples = len(signal)
    t = np.linspace(0, total_duration, num_samples)
    
    # Calculate instantaneous frequency
    # For FMCW chirp: f(t) = f0 + slope * t
    # where f0 is the starting frequency
    if bandwidth is not None:
        f0 = center_freq - bandwidth/2 if center_freq is not None else 0
        inst_freq = f0 + slope * t
    else:
        # If bandwidth not provided, just show relative frequency change
        inst_freq = slope * t
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot instantaneous frequency
    plt.plot(t * 1e6, inst_freq / 1e6, 'b-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time (μs)')
    plt.ylabel('Frequency (MHz)')
    plt.title(f'{title_prefix} Instantaneous Frequency')
    plt.grid(True)
    
    # Add bandwidth markers if provided
    if bandwidth is not None:
        if center_freq is not None:
            plt.axhline(y=(center_freq - bandwidth/2)/1e6, color='r', linestyle='--', label='Start Frequency')
            plt.axhline(y=(center_freq + bandwidth/2)/1e6, color='g', linestyle='--', label='End Frequency')
        else:
            plt.axhline(y=0, color='r', linestyle='--', label='Start Frequency')
            plt.axhline(y=bandwidth/1e6, color='g', linestyle='--', label='End Frequency')
        plt.legend()
    
    # Add text information if provided
    if textstr is not None:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()