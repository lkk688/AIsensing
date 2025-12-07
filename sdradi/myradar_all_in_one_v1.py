import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from timeit import default_timer as timer

from scipy.constants import c
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import h5py
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm  # noqa: F401

# Visualization imports (optional)
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# #from AIRadar.AIradar_datasetv7 import AIRadarDataset
# try:
#     from AIRadar.AIRadarLib.visualization import (
#         plot_signal_time_and_spectrum,
#         plot_instantaneous_frequency,
#         plot_3d_range_doppler_map_with_ground_truth
#     )
#     VISUALIZATION_AVAILABLE = True
# except ImportError:
#     print("Warning: AIRadarLib.visualization not available. Some visualizations will be skipped.")
#     VISUALIZATION_AVAILABLE = False

VISUALIZATION_AVAILABLE = True
def plot_3d_range_doppler_map_with_ground_truth(
    rd_map,
    targets,
    range_resolution,
    velocity_resolution,
    num_range_bins,
    num_doppler_bins,
    title_prefix='',
    save_path=None,
    apply_doppler_centering=True,
    detections=None,
    view_range_limits=None,
    view_velocity_limits=None,
    is_db=False,
    stride=4
):
    """
    Plot a 3D Range-Doppler map with ground truth target annotations and CFAR detections.
    Uses physical units (Meters and m/s) for axes.

    Args:
        rd_map: np.ndarray, shape (2, num_doppler_bins, num_range_bins) if is_db=False, 
                or (num_doppler_bins, num_range_bins) if is_db=True.
        targets: list of dicts, each with keys 'distance', 'velocity', 'rcs'
        range_resolution: float, range resolution in meters
        velocity_resolution: float, velocity resolution in m/s
        num_range_bins: int, number of range bins
        num_doppler_bins: int, number of Doppler bins
        title_prefix: str, prefix for plot title/labeling/saving
        save_path: str, directory to save the figure
        apply_doppler_centering: bool, whether Doppler FFT is centered (affects target position calculation)
        detections: list of dicts, optional CFAR detection results
        view_range_limits: tuple, (min, max) range in meters to display
        view_velocity_limits: tuple, (min, max) velocity in m/s to display
        is_db: bool, if True, rd_map is treated as already in dB.
        stride: int, stride for surface plot downsampling (higher is faster)
    """
    if is_db:
        rd_db = rd_map
    else:
        rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)
    
    # Create physical axes
    range_axis = np.arange(num_range_bins) * range_resolution
    if apply_doppler_centering:
        velocity_axis = (np.arange(num_doppler_bins) - num_doppler_bins // 2) * velocity_resolution
    else:
        velocity_axis = np.arange(num_doppler_bins) * velocity_resolution
        
    X, Y = np.meshgrid(range_axis, velocity_axis)
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, rd_db, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8, rstride=stride, cstride=stride)
    
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title(f'3D Range-Doppler Map with Ground Truth & Detections - {title_prefix}')
    
    # Set view limits if provided
    if view_range_limits:
        ax.set_xlim(view_range_limits)
    if view_velocity_limits:
        ax.set_ylim(view_velocity_limits)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Magnitude (dB)')

    # Plot Ground Truth Targets
    for i, target in enumerate(targets):
        r_val = target['distance']
        v_val = target['velocity']
        
        # Find indices for z-value lookup
        if apply_doppler_centering:
            r_idx = int(r_val / range_resolution)
            v_idx = int(num_doppler_bins // 2 + v_val / velocity_resolution)
        else:
            r_idx = int(r_val / range_resolution)
            v_idx = int(v_val / velocity_resolution) % num_doppler_bins
            
        if (0 <= r_idx < num_range_bins and 0 <= v_idx < num_doppler_bins):
            z_val = rd_db[v_idx, r_idx]
            # Lift the marker slightly above the surface
            ax.scatter([r_val], [v_val], [z_val + 2], color='red', s=100, marker='o', 
                      label='Ground Truth' if i == 0 else "")
            
            # Add label
            ax.text(r_val, v_val, z_val + 10, 
                   f"T{i+1}", color='black', fontsize=10, fontweight='bold')

    # Plot Detections (TP/FP)
    if detections:
        tp_count = 0
        fp_count = 0
        
        # Helper to check if a detection matches any target
        def is_tp(det_r_idx, det_d_idx, targets_list, r_tol=2, d_tol=2):
            for t in targets_list:
                if apply_doppler_centering:
                    t_r_idx = int(t['distance'] / range_resolution) # Assuming range starts at 0
                    t_d_idx = int(num_doppler_bins // 2 + t['velocity'] / velocity_resolution)
                else:
                    t_r_idx = int(t['distance'] / range_resolution)
                    t_d_idx = int(t['velocity'] / velocity_resolution) % num_doppler_bins
                
                if abs(det_r_idx - t_r_idx) <= r_tol and abs(det_d_idx - t_d_idx) <= d_tol:
                    return True
            return False

        for i, det in enumerate(detections):
            r_idx = det['range_idx']
            d_idx = det['doppler_idx']
            
            if (0 <= r_idx < num_range_bins and 0 <= d_idx < num_doppler_bins):
                z_val = rd_db[d_idx, r_idx]
                
                # Convert indices to physical units for plotting
                r_phys = r_idx * range_resolution
                if apply_doppler_centering:
                    v_phys = (d_idx - num_doppler_bins // 2) * velocity_resolution
                else:
                    v_phys = d_idx * velocity_resolution
                
                if is_tp(r_idx, d_idx, targets):
                    # True Positive
                    ax.scatter([r_phys], [v_phys], [z_val + 2], color='lime', s=80, marker='x', 
                              linewidth=2, label='Correct Detection' if tp_count == 0 else "")
                    tp_count += 1
                else:
                    # False Positive
                    ax.scatter([r_phys], [v_phys], [z_val + 2], color='yellow', s=80, marker='x', 
                              linewidth=2, label='False Positive' if fp_count == 0 else "")
                    fp_count += 1
    
    ax.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

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
    Plot the time-domain and spectrum of a complex signal with advanced options and optimized clarity.

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
    # Enhanced time axis with microsecond units for better readability
    t = np.linspace(0, total_duration, len(signal))
    t_us = t * 1e6  # Convert to microseconds for better readability
    
    signal_windowed, window = apply_window(signal, window_type=window_type)
    
    # Enhanced frequency axis handling
    if center_freq is not None:
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate)) + center_freq
    else:
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / sample_rate))
    freq_axis_mhz = freq_axis / 1e6  # MHz

    # Compute spectra with enhanced dynamic range
    spectrum_orig = calculate_spectrum(signal, N_fft)
    spectrum_win = calculate_spectrum(signal_windowed, N_fft)

    # Enhanced normalization with better dynamic range control
    if normalize:
        ref = max(np.max(spectrum_orig), np.max(spectrum_win))
        spectrum_orig = normalize_spectrum(spectrum_orig, reference=ref)
        spectrum_win = normalize_spectrum(spectrum_win, reference=ref)
        # Limit dynamic range to improve visibility
        spectrum_orig = np.maximum(spectrum_orig, -80)  # 80 dB dynamic range
        spectrum_win = np.maximum(spectrum_win, -80)

    # Find peak frequency
    peak_freq, peak_val = find_peak_frequency(spectrum_win, freq_axis_mhz) if highlight_peak else (None, None)

    # Create figure with optimized size and layout
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{title_prefix} - Time and Spectrum", fontsize=16, fontweight='bold')

    # Enhanced time domain plot
    axs[0].plot(t_us, np.real(signal), 'b-', label='Real', linewidth=1.5, alpha=0.8)
    axs[0].plot(t_us, np.imag(signal), 'r--', label='Imaginary', linewidth=1.5, alpha=0.8)
    if draw_window and len(window) == len(signal):
        # Scale window to signal amplitude for better visualization
        window_scaled = window * np.max(np.abs(signal)) * 0.8
        axs[0].plot(t_us, window_scaled, 'g-', label=f'{window_type.capitalize()} Window', 
                   linewidth=2, alpha=0.6)
    
    axs[0].set_title(f"{title_prefix} (Time Domain)", fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Time (μs)", fontsize=12)
    axs[0].set_ylabel("Amplitude", fontsize=12)
    axs[0].legend(loc='upper right', framealpha=0.9)
    axs[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axs[0].set_facecolor('#f8f9fa')
    
    # Add minor ticks for better precision
    axs[0].minorticks_on()
    axs[0].grid(True, which='minor', alpha=0.1, linestyle=':')

    # Enhanced spectrum plot with better colors and styling
    if draw_window:
        axs[1].plot(freq_axis_mhz, spectrum_orig, 'b-', label='Original', 
                   linewidth=1.2, alpha=0.6)
        axs[1].plot(freq_axis_mhz, spectrum_win, 'r-', 
                   label=f'{window_type.capitalize()} Windowed', 
                   linewidth=2, alpha=0.9)
    else:
        axs[1].plot(freq_axis_mhz, spectrum_orig, 'b-', label='Spectrum', 
                   linewidth=2, alpha=0.9)
    
    axs[1].set_title(f"{title_prefix} Spectrum", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Frequency (MHz)", fontsize=12)
    axs[1].set_ylabel("Magnitude (dB)", fontsize=12)
    axs[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axs[1].set_facecolor('#f8f9fa')
    
    # Add minor ticks for better precision
    axs[1].minorticks_on()
    axs[1].grid(True, which='minor', alpha=0.1, linestyle=':')

    # Enhanced bandwidth highlighting with better visual cues
    if bandwidth is not None and center_freq is not None:
        bandwidth_mhz = bandwidth / 1e6
        center_freq_mhz = center_freq / 1e6
        f_start = center_freq_mhz - bandwidth_mhz / 2
        f_end = center_freq_mhz + bandwidth_mhz / 2
        
        # Enhanced bandwidth visualization
        axs[1].axvspan(f_start, f_end, alpha=0.15, color='orange', 
                      label=f'Bandwidth: {bandwidth_mhz:.1f} MHz')
        axs[1].axvline(f_start, color='red', linestyle='--', linewidth=2, 
                      label=f'Start: {f_start:.1f} MHz')
        axs[1].axvline(f_end, color='green', linestyle='--', linewidth=2, 
                      label=f'End: {f_end:.1f} MHz')
        axs[1].axvline(center_freq_mhz, color='purple', linestyle=':', linewidth=2, 
                      label=f'Center: {center_freq_mhz:.1f} MHz')
        
        # Smart zoom with margin
        if zoom_margin > 0:
            margin = bandwidth_mhz * zoom_margin
            axs[1].set_xlim([f_start - margin, f_end + margin])
    elif bandwidth is not None and zoom_margin > 0:
        bandwidth_mhz = bandwidth / 1e6
        f_start = -bandwidth_mhz / 2
        f_end = bandwidth_mhz / 2
        
        axs[1].axvspan(f_start, f_end, alpha=0.15, color='orange', 
                      label=f'Bandwidth: {bandwidth_mhz:.1f} MHz')
        axs[1].axvline(f_start, color='red', linestyle='--', linewidth=2)
        axs[1].axvline(f_end, color='green', linestyle='--', linewidth=2)
        
        margin = bandwidth_mhz * zoom_margin
        axs[1].set_xlim([f_start - margin, f_end + margin])

    # Enhanced peak highlighting
    if highlight_peak and peak_freq is not None:
        axs[1].axvline(peak_freq, color='magenta', linestyle='-', linewidth=3, 
                      label=f'Peak: {peak_freq:.2f} MHz')
        # Better annotation positioning
        y_range = axs[1].get_ylim()
        annotation_y = peak_val + (y_range[1] - y_range[0]) * 0.1
        axs[1].annotate(f'Peak\n{peak_freq:.2f} MHz\n{peak_val:.1f} dB', 
                       xy=(peak_freq, peak_val), 
                       xytext=(peak_freq, annotation_y),
                       arrowprops=dict(arrowstyle='->', color='magenta', lw=2),
                       fontsize=10, color='magenta', fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='magenta', alpha=0.8))

    # Enhanced legend positioning
    axs[1].legend(loc='upper right', framealpha=0.9, fontsize=10)

    # Enhanced text display
    if textstr is not None:
        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                    edgecolor='navy', alpha=0.8)
        axs[1].text(0.02, 0.98, textstr, transform=axs[1].transAxes, 
                   fontsize=10, verticalalignment='top', bbox=props,
                   fontweight='bold')

    # Enhanced layout and styling
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Improve overall appearance
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

# Hardware imports (optional)
try:
    import adi
    import phaser.mycn0566 as mycn0566
    from myadi.aditddn import tddn
    from myadiclass import SDR
    HARDWARE_AVAILABLE = True
except ImportError:
    print("Warning: Hardware libraries (adi, phaser, myadi) not found. Running in SIMULATION mode only.")
    HARDWARE_AVAILABLE = False

# --- Plot view limits ---
VIEW_RANGE_LIMITS = (0, 100)
VIEW_VELOCITY_LIMITS = (-48, 48)

# --- Radar Configs (central source of truth) ---
RADAR_CONFIGS = {
    'config1': {
        'name': 'Automotive_77GHz_LongRange',
        'signal_type': 'FMCW',  # Default signal type
        'fc': 77e9,             # 77 GHz
        'B': 1.5e9,             # 1.5 GHz Bandwidth
        'T_chirp': 40e-6,       # 40 μs
        'fs': 51.2e6,           # 51.2 MHz Sampling Rate
        'N_chirps': 128,        # Number of chirps
        'R_max': 100.0,         # Max range 100m
        'description': 'Standard automotive long-range radar configuration',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {
            'num_train': 10,
            'num_guard': 4,
            'threshold_offset': 15,
            'nms_kernel_size': 5
        }
    },
    'config2': {
        'name': 'XBand_10GHz_MediumRange',
        'signal_type': 'FMCW',
        'fc': 10e9,             # 10 GHz (X-band)
        'B': 1.0e9,             # 1 GHz Bandwidth
        'T_chirp': 160e-6,      # 160 μs
        'fs': 40e6,             # 40 MHz Sampling Rate
        'N_chirps': 128,        # Number of chirps
        'R_max': 100.0,         # Max range 100m
        'description': 'X-band radar for medium range surveillance or robotics',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {
            'num_train': 24,
            'num_guard': 8,
            'threshold_offset': 18,
            'nms_kernel_size': 7
        }
    },
    'config_otfs': {
        'name': 'OTFS_Automotive_77GHz',
        'signal_type': 'OTFS',
        'fc': 77e9,             # 77 GHz
        'B': 1.536e9,           # ~1.5 GHz
        'T_chirp': 40e-6,       # Symbol duration
        'fs': 51.2e6,           # Sampling rate
        'N_chirps': 128,        # OTFS symbols
        'N_samples': 512,       # subcarriers / delay bins
        'R_max': 100.0,
        'description': 'OTFS Radar configuration (generic automotive)',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {
            'num_train': 16,
            'num_guard': 8,
            'threshold_offset': 25,
            'nms_kernel_size': 9
        }
    },
    'config_phaser': {
        'name': 'Phaser_10GHz_DevKit',
        'signal_type': 'FMCW',
        'fc': 10e9,             # 10 GHz
        'B': 500e6,             # 500 MHz Bandwidth
        'T_chirp': 500e-6,      # 500 μs
        'fs': 2.0e6,            # 2 MHz Sampling Rate
        'N_chirps': 64,
        'R_max': 100.0,
        'description': 'Phaser Dev Kit configuration (simple model)',
        'hardware_model': 'generic',
        'num_rx': 1,
        'use_array_factor': False,
        'cfar_params': {
            'num_train': 10,
            'num_guard': 4,
            'threshold_offset': 15,
            'nms_kernel_size': 5
        }
    },
    'config_cn0566': {
        'name': 'CN0566_Phaser_DevKit',
        'signal_type': 'FMCW',
        'fc': 10.25e9,          # Centered in 10–10.5 GHz band
        'B': 500e6,             # 500 MHz FMCW BW
        'T_chirp': 500e-6,      # 500 µs ramp
        'fs': 2.0e6,            # Pluto baseband sample rate
        'N_chirps': 64,
        'R_max': 150.0,
        'description': 'Analog Devices CN0566 Phaser radar dev kit (Pluto IF chain model)',
        'hardware_model': 'CN0566',
        'num_rx': 2,            # Rx1, Rx2
        'use_array_factor': True,
        'array_N': 8,           # 8-element ULA
        'steering_angles': [0.0, 0.0],
        # Hardware impairments
        'model_dc_offset': True,
        'model_iq_imbalance': True,
        'model_phase_noise': True,
        'quantize_adc': True,
        'adc_bits': 12,
        'dc_scale': 0.05,
        'iq_gain_std': 0.02,
        'iq_phase_std_deg': 3.0,
        'phase_noise_std': 0.02,
        'cfo_std_hz': 200.0,
        # Clutter & coupling
        'static_clutter_velocity_std': 0.02,
        'ground_clutter_velocity_std': 0.1,
        'coupling_rcs_db': 0.0,      # strong TX leakage
        # CN0566-specific CFAR tuned for fewer FAs
        'cfar_params': {
            'num_train': 24,
            'num_guard': 8,
            'threshold_offset': 22,  # stricter threshold
            'nms_kernel_size': 9
        }
    }
}

# ======================================================================
# Shared helpers
# ======================================================================

def suppress_known_artifacts(rdm, range_axis, velocity_axis):
    """
    Suppress obvious artifact regions (e.g., strong TX leakage near 0 m, static clutter band).
    This is applied before CFAR in CN0566 mode.
    """
    rdm_clean = rdm.copy()

    # 1) Strong TX leakage at very near range: suppress < 1 m
    near_mask = range_axis < 1.0
    if np.any(near_mask):
        rdm_clean[:, near_mask] -= 30.0  # push down by 30 dB

    # 2) Very small velocities (|v| < 0.3 m/s) get mild attenuation
    vel_mask = np.abs(velocity_axis) < 0.3
    if np.any(vel_mask):
        rdm_clean[vel_mask, :] -= 10.0

    return rdm_clean


def fmcw_rd_from_beats(beat_rx, fs, T, slope, zero_pad, R_max,
                       num_range_bins=None):
    """
    Compute Range-Doppler map from FMCW multi-RX beat signals.

    Args:
        beat_rx: complex ndarray [Rx, Nc, Ns] or [Nc, Ns]
        fs: sample rate (Hz)
        T: chirp duration (s)
        slope: chirp slope (Hz/s)
        zero_pad: FFT zero-padding size (int)
        R_max: used to crop range bins
        num_range_bins: optionally fix the number of range bins

    Returns:
        rdm: [Nc, num_range_bins] in dB
    """
    if beat_rx.ndim == 3:
        beat = np.sum(beat_rx, axis=0)  # [Nc, Ns]
    else:
        beat = beat_rx

    Nc, Ns = beat.shape

    window_range = np.hanning(Ns)[None, :]
    window_doppler = np.hanning(Nc)[:, None]

    beat = beat * window_range
    beat = beat * window_doppler

    # Range FFT
    range_fft = np.fft.fft(beat, n=zero_pad, axis=1)
    range_fft = range_fft[:, :zero_pad // 2]

    # Doppler FFT
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

    rdm = 20 * np.log10(np.abs(doppler_fft) + 1e-6)

    # Range bin cropping
    if num_range_bins is None:
        range_res_fft = (c * fs) / (2 * slope * zero_pad)
        max_bin_idx = int(R_max / range_res_fft)
        num_range_bins = min(zero_pad // 2, max_bin_idx)

    rdm = rdm[:, :num_range_bins]
    return rdm


def _plot_2d_rdm(dataset_instance, rdm, sample_idx, metrics,
                 matched_pairs, unmatched_targets, unmatched_detections, save_path):
    """
    Plot 2D Range-Doppler Map with annotations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    dr = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
    dv = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]
    extent = [dataset_instance.range_axis[0] - dr/2, dataset_instance.range_axis[-1] + dr/2,
              dataset_instance.velocity_axis[0] - dv/2, dataset_instance.velocity_axis[-1] + dv/2]

    im = ax.imshow(rdm, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Range-Doppler Map with CFAR Detection - Sample {sample_idx}")

    ax.set_xlim(VIEW_RANGE_LIMITS)
    ax.set_ylim(VIEW_VELOCITY_LIMITS)

    legend_elements = []

    for target in matched_pairs:
        t = target[0]
        ax.scatter(t['range'], t['velocity'], facecolors='none', edgecolors='lime',
                   s=150, linewidth=2, label='Matched GT')
        d = target[1]
        ax.plot([t['range'], d['range_m']], [t['velocity'], d['velocity_mps']], 'w--', alpha=0.5)

    for target in unmatched_targets:
        ax.scatter(target['range'], target['velocity'], facecolors='none', edgecolors='red',
                   s=150, linewidth=2, label='Missed GT (FN)')

    for pair in matched_pairs:
        d = pair[1]
        ax.scatter(d['range_m'], d['velocity_mps'], marker='x', color='cyan',
                   s=100, linewidth=2, label='True Positive (TP)')

    for det in unmatched_detections:
        ax.scatter(det['range_m'], det['velocity_mps'], marker='x', color='orange',
                   s=100, linewidth=2, label='False Alarm (FP)')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_elements.extend(by_label.values())

    metrics_text = (
        f"Evaluation Metrics:\n"
        f"-------------------\n"
        f"Targets: {metrics['num_targets']}\n"
        f"Detections: {metrics['num_detections']}\n"
        f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}\n"
        f"Range Error (MAE): {metrics['mean_range_error']:.2f} m\n"
        f"Vel Error (MAE): {metrics['mean_velocity_error']:.2f} m/s"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_3d_rdm(dataset_instance, rdm, sample_idx, targets, detections, save_path):
    """
    Plot 3D Range-Doppler Map using external visualization library (if available).
    """
    if not VISUALIZATION_AVAILABLE:
        return

    converted_targets = []
    for t in targets:
        ct = t.copy()
        ct['distance'] = t['range']
        converted_targets.append(ct)

    range_res = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
    vel_res = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]

    cleaned_detections = []
    if detections:
        for det in detections:
            d_copy = det.copy()
            if 'range_idx' in d_copy:
                d_copy['range_idx'] = int(d_copy['range_idx'])
            if 'doppler_idx' in d_copy:
                d_copy['doppler_idx'] = int(d_copy['doppler_idx'])
            cleaned_detections.append(d_copy)

    plot_3d_range_doppler_map_with_ground_truth(
        rd_map=rdm,
        targets=converted_targets,
        range_resolution=range_res,
        velocity_resolution=vel_res,
        num_range_bins=rdm.shape[1],
        num_doppler_bins=rdm.shape[0],
        save_path=save_path,
        apply_doppler_centering=True,
        detections=cleaned_detections,
        view_range_limits=VIEW_RANGE_LIMITS,
        view_velocity_limits=VIEW_VELOCITY_LIMITS,
        is_db=True,
        stride=8
    )


def evaluate_dataset_metrics(dataset, name):
    """
    Aggregate CFAR detection metrics across all samples in a dataset,
    and optionally generate 2D/3D Range-Doppler visualizations for a few samples.

    Visualization behavior:
      - If `dataset.save_path` is not None, figures for the first few samples
        are saved under: <dataset.save_path>/eval_visualizations
      - Uses _plot_2d_rdm and _plot_3d_rdm defined in this file.
    """
    print(f"\nEvaluating CFAR Metrics for {name}...")
    all_tp, all_fp, all_fn = 0, 0, 0
    all_range_errors = []
    all_vel_errors = []

    # Where to save evaluation visualizations (if possible)
    vis_dir = None
    if hasattr(dataset, "save_path") and dataset.save_path is not None:
        vis_dir = os.path.join(dataset.save_path, "eval_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    # How many samples to visualize
    max_vis_samples = min(3, len(dataset))

    # Slug version of the dataset name for filenames
    name_slug = "".join(
        ch for ch in name.lower().replace(" ", "_")
        if ch.isalnum() or ch in ["_"]
    )

    for i in range(len(dataset)):
        sample = dataset[i]
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']

        # Per-sample metrics + matching (needed for visualization)
        metrics, matched_pairs, unmatched_targets, unmatched_detections = \
            dataset._evaluate_metrics(targets, detections)

        # Aggregate stats
        all_tp += metrics['tp']
        all_fp += metrics['fp']
        all_fn += metrics['fn']
        if metrics['mean_range_error'] > 0:
            all_range_errors.append(metrics['mean_range_error'])
        if metrics['mean_velocity_error'] > 0:
            all_vel_errors.append(metrics['mean_velocity_error'])

        # Optional visualization for the first few samples
        if vis_dir is not None and i < max_vis_samples:
            # RD map tensor -> numpy, normalize for plotting
            rdm = sample['range_doppler_map'].numpy()
            rdm_norm = rdm - np.max(rdm)

            # 2D figure
            save_path_2d = os.path.join(vis_dir, f"{name_slug}_sample_{i}_2d.png")
            _plot_2d_rdm(
                dataset,
                rdm_norm,
                i,
                metrics,
                matched_pairs,
                unmatched_targets,
                unmatched_detections,
                save_path_2d
            )

            # 3D figure (uses external visualization if available)
            save_path_3d = os.path.join(vis_dir, f"{name_slug}_sample_{i}_3d.png")
            _plot_3d_rdm(
                dataset,
                rdm_norm,
                i,
                targets,
                detections,
                save_path_3d
            )

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_range_err = np.mean(all_range_errors) if all_range_errors else 0.0
    mean_vel_err = np.mean(all_vel_errors) if all_vel_errors else 0.0

    print(f"--- {name} CFAR Results ---")
    print(f"Samples: {len(dataset)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print(f"Mean Range Error: {mean_range_err:.4f} m")
    print(f"Mean Vel Error:   {mean_vel_err:.4f} m/s")
    if vis_dir is not None:
        print(f"Eval visualizations saved to: {vis_dir}")
    print("-" * 30)


# ======================================================================
# AIRadarDataset
# ======================================================================

class AIRadarDataset(Dataset):
    def __init__(self,
                 num_samples=100,
                 radar_config=None,
                 config_name='config1',
                 fc=None,
                 B=None,
                 T_chirp=None,
                 fs=None,
                 N_samples=None,
                 N_chirps=None,
                 R_max=None,
                 SNR_dB_min=20,
                 SNR_dB_max=40,
                 zero_pad_factor=2,
                 max_targets=3,
                 save_path='data/radar_dataset',
                 precision='float32',
                 drawfig=False,
                 datapath=None,
                 cfar_params=None,
                 apply_realistic_effects=True,
                 clutter_intensity=1.0,
                 autogen=True):
        """
        Radar dataset simulator + container.
        """
        # Load config
        if radar_config is None:
            if config_name in RADAR_CONFIGS:
                cfg = RADAR_CONFIGS[config_name]
                print(f"Loading Radar Configuration: {config_name} ({cfg['name']})")
            else:
                print(f"Warning: Config '{config_name}' not found. Using 'config1'.")
                cfg = RADAR_CONFIGS['config1']
        else:
            cfg = radar_config

        self.config = cfg
        self.signal_type = cfg.get('signal_type', 'FMCW')
        self.num_samples = num_samples
        self.fc = fc if fc is not None else cfg.get('fc', 77e9)
        self.B = B if B is not None else cfg.get('B', 1.5e9)
        self.T = T_chirp if T_chirp is not None else cfg.get('T_chirp', 40e-6)
        self.Nc = N_chirps if N_chirps is not None else cfg.get('N_chirps', 128)
        self.R_max = R_max if R_max is not None else cfg.get('R_max', 100.0)
        self.hardware_model = cfg.get('hardware_model', 'generic')

        self.num_rx = int(cfg.get('num_rx', 1))

        self.fs = fs if fs is not None else cfg.get('fs', None)
        if self.fs is not None:
            self.Ns = int(self.fs * self.T)
            if N_samples is not None:
                print(f"Info: N_samples ({N_samples}) ignored because fs is provided.")
        else:
            self.Ns = N_samples if N_samples is not None else cfg.get('N_samples', 2048)
            self.fs = self.Ns / self.T
            print(f"Info: fs not provided. Calculated fs = {self.fs/1e6:.2f} MHz from N_samples ({self.Ns})")

        self.SNR_dB_min = SNR_dB_min
        self.SNR_dB_max = SNR_dB_max
        self.max_targets = max_targets
        self.save_path = save_path
        self.drawfig = drawfig
        self.precision = precision
        self.apply_realistic_effects = apply_realistic_effects
        self.clutter_intensity = clutter_intensity

        # CFAR parameters
        default_cfar = {'num_train': 10, 'num_guard': 4, 'threshold_offset': 15, 'nms_kernel_size': 5}
        config_cfar = cfg.get('cfar_params', default_cfar)
        self.cfar_params = cfar_params if cfar_params is not None else config_cfar

        # Derived parameters
        self.lambda_c = c / self.fc
        self.slope = self.B / self.T
        self.zero_pad = zero_pad_factor * self.Ns

        # Hardware modeling parameters
        self.use_array_factor = bool(cfg.get('use_array_factor', False))
        self.array_N = int(cfg.get('array_N', 8))
        steering_angles = cfg.get('steering_angles', [0.0] * self.num_rx)
        if len(steering_angles) < self.num_rx:
            steering_angles = list(steering_angles) + [steering_angles[-1]] * (self.num_rx - len(steering_angles))
        self.steering_angles = np.array(steering_angles, dtype=float)

        self.model_dc_offset = bool(cfg.get('model_dc_offset', False))
        self.model_iq_imbalance = bool(cfg.get('model_iq_imbalance', False))
        self.model_phase_noise = bool(cfg.get('model_phase_noise', False))
        self.quantize_adc = bool(cfg.get('quantize_adc', False))

        self.dc_scale = float(cfg.get('dc_scale', 0.02))
        self.iq_gain_std = float(cfg.get('iq_gain_std', 0.02))
        self.iq_phase_std_deg = float(cfg.get('iq_phase_std_deg', 2.0))
        self.phase_noise_std = float(cfg.get('phase_noise_std', 0.01))
        self.cfo_std_hz = float(cfg.get('cfo_std_hz', 100.0))
        self.adc_bits = int(cfg.get('adc_bits', 12))

        self.static_clutter_velocity_std = float(cfg.get('static_clutter_velocity_std', 0.0))
        self.ground_clutter_velocity_std = float(cfg.get('ground_clutter_velocity_std', 0.3))
        self.coupling_rcs_db = float(cfg.get('coupling_rcs_db', -10.0))

        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T

        # Nyquist-based R_max check for FMCW
        if self.signal_type == 'FMCW':
            f_nyquist = self.fs / 2.0
            R_max_nyquist = f_nyquist * c / (2 * self.slope)
            if self.R_max > R_max_nyquist:
                print(f"WARNING: Requested R_max={self.R_max:.1f}m exceeds Nyquist limit {R_max_nyquist:.1f}m. Clipping.")
                self.R_max = R_max_nyquist
        self.R_max_physical = self.R_max

        # Axes
        if self.signal_type == 'OTFS':
            range_res = c / (2 * self.fs)
            self.range_resolution = range_res
            self.num_range_bins = int(self.R_max / range_res)
            self.range_axis = np.arange(self.num_range_bins) * range_res
            self.num_doppler_bins = self.Nc
        else:
            range_res_fft = (c * self.fs) / (2 * self.slope * self.zero_pad)
            self.range_axis = np.arange(self.zero_pad // 2) * range_res_fft
            max_bin_idx = int(self.R_max / range_res_fft)
            self.num_range_bins = min(self.zero_pad // 2, max_bin_idx)
            self.range_axis = self.range_axis[:self.num_range_bins]
            self.num_doppler_bins = self.Nc

        self.velocity_axis = np.fft.fftshift(np.fft.fftfreq(self.Nc, d=self.T)) * self.lambda_c / 2
        self.range_resolution = c / (2 * self.B)
        self.velocity_resolution = self.lambda_c / (2 * self.Nc * self.T)
        self.max_unambiguous_velocity = self.lambda_c / (4 * self.T)
        self.v_max = self.max_unambiguous_velocity

        self.time_domain_data = None
        self.range_doppler_maps = None
        self.target_masks = None
        self.target_info = None
        self.cfar_detections = None

        print("\n=== Radar System Parameters ===")
        print(f"Signal Type         : {self.signal_type}")
        print(f"Hardware Model      : {self.hardware_model}")
        print(f"Center Frequency    : {self.fc/1e9:.2f} GHz")
        print(f"Bandwidth           : {self.B/1e6:.1f} MHz")
        print(f"Chirp Duration      : {self.T*1e6:.1f} μs")
        print(f"Sample Rate         : {self.fs/1e6:.2f} MHz")
        print(f"R_max (used)        : {self.R_max:.1f} m")
        print(f"Range Resolution    : {self.range_resolution:.2f} m")
        print(f"v_max               : {self.v_max:.1f} m/s")
        print(f"Velocity Resolution : {self.velocity_resolution:.2f} m/s")
        print(f"Ns, Nc              : {self.Ns}, {self.Nc}")
        print(f"Num RX              : {self.num_rx}")
        print(f"Range Bins          : {self.num_range_bins}")
        print(f"Doppler Bins        : {self.num_doppler_bins}")
        print(f"CFAR Params         : {self.cfar_params}")
        print("====================================\n")

        if datapath is not None:
            self._load_data(datapath)
        elif autogen:
            print("Generating new radar dataset...")
            self.generate_dataset()

    # ---- target & clutter ----
    def generate_targets(self, num_targets=None):
        if num_targets is None:
            num_targets = np.random.randint(1, self.max_targets + 1)

        targets = []
        for _ in range(num_targets):
            r = np.random.uniform(10, self.R_max - 10)
            v = np.random.uniform(-self.v_max + 1, self.v_max - 1)
            rcs = np.random.uniform(5.0, 30.0)  # dBsm

            targets.append({
                'range': r,
                'velocity': v,
                'rcs': rcs,
                'azimuth': np.random.uniform(-30, 30),
                'elevation': np.random.uniform(-10, 10)
            })
        return targets

    def _generate_clutter_targets(self):
        clutter_targets = []
        intensity_db = 10 * np.log10(max(self.clutter_intensity, 1e-6))

        if self.hardware_model == 'CN0566':
            num_static = np.random.randint(3, 9)
            num_ground = np.random.randint(10, 21)
        else:
            num_static = np.random.randint(5, 16)
            num_ground = np.random.randint(20, 51)

        # static clutter
        for _ in range(num_static):
            vel = np.random.normal(0.0, self.static_clutter_velocity_std)
            clutter_targets.append({
                'range': np.random.uniform(5, self.R_max),
                'velocity': vel,
                'rcs': np.random.uniform(-40, -20) + intensity_db,
                'azimuth': np.random.uniform(-30, 30),
                'elevation': 0.0
            })

        # ground clutter
        for _ in range(num_ground):
            dist = np.random.uniform(1.0, self.R_max * 0.5)
            vel = np.random.normal(0.0, self.ground_clutter_velocity_std)
            clutter_targets.append({
                'range': dist,
                'velocity': vel,
                'rcs': np.random.uniform(-50, -30) + intensity_db,
                'azimuth': np.random.uniform(-60, 60),
                'elevation': np.random.uniform(-10, 0)
            })

        return clutter_targets

    def _generate_coupling_target(self):
        intensity_db = 10 * np.log10(max(self.clutter_intensity, 1e-6))
        return {
            'range': 0.001,
            'velocity': 0.0,
            'rcs': self.coupling_rcs_db + intensity_db,
            'azimuth': 0.0,
            'elevation': 0.0
        }

    # ---- array factor & impairments ----
    def _array_gain(self, azimuth_deg, rx_idx=0):
        if not self.use_array_factor or self.array_N <= 1:
            return 1.0
        d = self.lambda_c / 2.0
        N = self.array_N
        theta = np.deg2rad(azimuth_deg)
        theta_steer = np.deg2rad(self.steering_angles[rx_idx])
        k = 2 * np.pi / self.lambda_c
        psi = k * d * (np.sin(theta) - np.sin(theta_steer))
        denom = N * np.sin(psi / 2.0) + 1e-12
        num = np.sin(N * psi / 2.0)
        AF = num / denom
        return float(np.abs(AF))

    def _apply_hardware_impairments(self, beat_rx):
        # beat_rx: [Rx, Nc, Ns]
        if self.model_dc_offset:
            power = np.mean(np.abs(beat_rx)**2) + 1e-12
            sigma = np.sqrt(power) * self.dc_scale
            dc = (np.random.randn(self.num_rx, 1, 1) + 1j * np.random.randn(self.num_rx, 1, 1)) * sigma
            beat_rx = beat_rx + dc

        if self.model_iq_imbalance:
            for r in range(self.num_rx):
                I = beat_rx[r].real
                Q = beat_rx[r].imag
                gain_q = 1.0 + np.random.normal(0.0, self.iq_gain_std)
                phase_err = np.deg2rad(np.random.normal(0.0, self.iq_phase_std_deg))
                Q_imb = gain_q * (Q * np.cos(phase_err) + I * np.sin(phase_err))
                beat_rx[r] = I + 1j * Q_imb

        if self.model_phase_noise:
            pn = np.cumsum(np.random.normal(0.0, self.phase_noise_std, size=self.Nc))
            factor_slow = np.exp(1j * pn)[None, :, None]
            f_cfo = np.random.normal(0.0, self.cfo_std_hz)
            factor_fast = np.exp(1j * 2 * np.pi * f_cfo * self.t_fast)[None, None, :]
            beat_rx = beat_rx * factor_slow * factor_fast

        if self.quantize_adc:
            max_abs = np.max(np.abs(beat_rx)) + 1e-12
            norm = beat_rx / max_abs
            q_levels = 2 ** (self.adc_bits - 1) - 1
            Iq = np.round(norm.real * q_levels) / q_levels
            Qq = np.round(norm.imag * q_levels) / q_levels
            beat_rx = (Iq + 1j * Qq) * max_abs

        return beat_rx

    # ---- signal simulation ----
    def simulate_fmcw_signal(self, targets, snr_db=20):
        """
        FMCW multi-RX simulation.
        Returns: beat_rx [Rx, Nc, Ns] (complex)
        """
        beat_rx = np.zeros((self.num_rx, self.Nc, self.Ns), dtype=np.complex128)

        if targets:
            ranges = np.array([t['range'] for t in targets])
            velocities = np.array([t['velocity'] for t in targets])
            rcs = np.array([t['rcs'] for t in targets])
            azimuths = np.array([t['azimuth'] for t in targets])

            K = len(ranges)
            rcs_linear = 10 ** (rcs / 10)
            amplitudes = np.sqrt(rcs_linear)

            fb = 2 * ranges * self.slope / c
            fd = 2 * velocities / self.lambda_c

            fb_grid = fb[:, None, None]
            fd_grid = fd[:, None, None]
            t_fast_grid = self.t_fast[None, None, :]
            t_slow_grid = self.t_slow[None, :, None]

            phase = 2 * np.pi * (fb_grid * t_fast_grid + fd_grid * t_slow_grid)
            signal_k = amplitudes[:, None, None] * np.exp(1j * phase)  # [K, Nc, Ns]

            gains = np.ones((K, self.num_rx), dtype=float)
            if self.use_array_factor:
                for k in range(K):
                    for r in range(self.num_rx):
                        gains[k, r] = self._array_gain(azimuths[k], rx_idx=r)

            beat_rx = np.einsum('kr,kij->rij', gains, signal_k)

        # windows
        beat_rx *= np.hanning(self.Ns)[None, None, :]
        beat_rx *= np.hanning(self.Nc)[None, :, None]

        # AWGN
        signal_power = np.mean(np.abs(beat_rx) ** 2)
        if signal_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_scale = np.sqrt(noise_power / 2)
            noise = (np.random.randn(*beat_rx.shape) + 1j * np.random.randn(*beat_rx.shape)) * noise_scale
            beat_rx += noise

        if self.hardware_model == 'CN0566':
            beat_rx = self._apply_hardware_impairments(beat_rx)

        return beat_rx

    def _otfs_modulate(self, dd_grid):
        tf_grid = np.fft.fft(dd_grid, axis=0)
        tf_grid = np.fft.ifft(tf_grid, axis=1)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        tx_signal = time_domain_grid.flatten(order='F')
        return tx_signal

    def _otfs_demodulate(self, rx_signal):
        time_domain_grid = rx_signal.reshape((self.Ns, self.Nc), order='F')
        tf_grid = np.fft.fft(time_domain_grid, axis=0)
        dd_grid = np.fft.fft(tf_grid, axis=1)
        dd_grid = np.fft.ifft(dd_grid, axis=0)
        return dd_grid

    def simulate_otfs_signal(self, targets, snr_db=20):
        num_symbols = self.Ns * self.Nc
        bits = np.random.randint(0, 4, num_symbols)
        mod_map = {
            0: (1 + 1j) / np.sqrt(2),
            1: (1 - 1j) / np.sqrt(2),
            2: (-1 + 1j) / np.sqrt(2),
            3: (-1 - 1j) / np.sqrt(2)
        }
        symbols = np.array([mod_map[b] for b in bits])
        tx_dd_grid = symbols.reshape((self.Ns, self.Nc))

        tx_signal = self._otfs_modulate(tx_dd_grid)

        n_samples = tx_signal.size
        rx_signal = np.zeros(n_samples, dtype=complex)
        time_vector = np.arange(n_samples) / self.fs

        for t in targets:
            rm = t['range']
            vm = t['velocity']
            rcs = t['rcs']
            amp = np.sqrt(10 ** (rcs / 10))

            delay_sec = 2 * rm / c
            delay_samples = int(round(delay_sec * self.fs))
            if delay_samples < n_samples:
                delayed_signal = np.roll(tx_signal, delay_samples)
                delayed_signal[:delay_samples] = 0
                doppler_hz = 2 * vm * self.fc / c
                doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector)
                rx_signal += amp * delayed_signal * doppler_shift

        signal_power = np.mean(np.abs(rx_signal)**2)
        if signal_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * np.sqrt(noise_power / 2)
            rx_signal += noise

        rx_dd_grid = self._otfs_demodulate(rx_signal)

        rx_dd_fft = np.fft.fft2(rx_dd_grid)
        tx_dd_fft = np.fft.fft2(tx_dd_grid)
        epsilon = 1e-6
        ddm_fft = rx_dd_fft / (tx_dd_fft + epsilon)
        ddm_complex = np.fft.ifft2(ddm_fft)

        ddm_transposed = ddm_complex.T
        ddm_shifted = np.fft.fftshift(ddm_transposed, axes=0)
        ddm_mag = np.abs(ddm_shifted)
        ddm_db = 20 * np.log10(ddm_mag + 1e-6)

        if ddm_db.shape[1] > self.num_range_bins:
            ddm_db = ddm_db[:, :self.num_range_bins]

        rx_time_reshaped = rx_signal.reshape((self.Ns, self.Nc), order='F').T
        beat_rx = rx_time_reshaped[None, :, :]

        return beat_rx, ddm_db

    # ---- mask & CFAR ----
    def create_target_mask(self, targets):
        mask = np.zeros((self.num_doppler_bins, self.num_range_bins))
        for t in targets:
            r_idx = np.argmin(np.abs(self.range_axis - t['range']))
            v_idx = np.argmin(np.abs(self.velocity_axis - t['velocity']))
            for di in range(-1, 2):
                for ri in range(-1, 2):
                    vi = v_idx + di
                    ri_idx = r_idx + ri
                    if 0 <= vi < self.num_doppler_bins and 0 <= ri_idx < self.num_range_bins:
                        mask[vi, ri_idx] = 1.0
        return mask

    def _cfar_2d_custom(self, rd_map_db, num_train=8, num_guard=4,
                        range_res=0.5, doppler_res=0.25,
                        max_range=100, max_speed=50,
                        threshold_offset=4, nms_kernel_size=3,
                        mtd=False):
        from scipy.signal import convolve2d
        from scipy.ndimage import maximum_filter

        rows, cols = rd_map_db.shape
        k = num_guard + num_train
        window_size = 2 * k + 1
        full_kernel = np.ones((window_size, window_size), dtype=np.float32)
        guard_area = np.zeros_like(full_kernel)
        guard_area[num_train:num_train + 2*num_guard + 1,
                   num_train:num_train + 2*num_guard + 1] = 1
        train_kernel = full_kernel - guard_area

        horiz_kernel = train_kernel.copy()
        horiz_kernel[num_train:num_train + 2*num_guard + 1, :] = 0
        vert_kernel = train_kernel.copy()
        vert_kernel[:, num_train:num_train + 2*num_guard + 1] = 0

        noise_h = convolve2d(rd_map_db, horiz_kernel / np.sum(horiz_kernel),
                             mode='same', boundary='symm')
        noise_v = convolve2d(rd_map_db, vert_kernel / np.sum(vert_kernel),
                             mode='same', boundary='symm')
        noise_est = np.maximum(noise_h, noise_v)

        threshold = noise_est + threshold_offset
        detections = rd_map_db > threshold

        if nms_kernel_size > 1:
            local_max = maximum_filter(rd_map_db, size=nms_kernel_size)
            detections &= (rd_map_db == local_max)

        doppler_idxs, range_idxs = np.where(detections)
        results = []

        num_doppler = rows
        for d_idx, r_idx in zip(doppler_idxs, range_idxs):
            range_m = r_idx * range_res
            velocity_mps = (d_idx - num_doppler // 2) * doppler_res
            if not (0.5 < range_m < max_range and abs(velocity_mps) < max_speed):
                continue
            if mtd and abs(velocity_mps) < 1.0:
                continue
            results.append({
                "range_idx": r_idx,
                "doppler_idx": d_idx,
                "range_m": range_m,
                "velocity_mps": velocity_mps,
                "angle_deg": None
            })
        return results

    def cfar_detection(self, rd_map):
        range_res = self.range_axis[1] - self.range_axis[0]
        velocity_res = self.velocity_axis[1] - self.velocity_axis[0]
        mtd_enabled = self.apply_realistic_effects

        rd_map_used = rd_map
        if self.hardware_model == 'CN0566':
            rd_map_used = suppress_known_artifacts(rd_map, self.range_axis, self.velocity_axis)

        cfar_results = self._cfar_2d_custom(
            rd_map_used,
            num_train=self.cfar_params.get('num_train', 10),
            num_guard=self.cfar_params.get('num_guard', 4),
            range_res=range_res,
            doppler_res=velocity_res,
            max_range=self.R_max,
            max_speed=50,
            threshold_offset=self.cfar_params.get('threshold_offset', 15),
            nms_kernel_size=self.cfar_params.get('nms_kernel_size', 5),
            mtd=mtd_enabled
        )

        for det in cfar_results:
            d_idx = det['doppler_idx']
            r_idx = det['range_idx']
            if 0 <= d_idx < rd_map.shape[0] and 0 <= r_idx < rd_map.shape[1]:
                det['magnitude'] = rd_map[d_idx, r_idx]
        return cfar_results

    # ---- dataset generation ----
    def generate_dataset(self):
        print(f"Generating {self.num_samples} samples...")

        self.time_domain_data = np.zeros(
            (self.num_samples, self.num_rx, self.Nc, self.Ns, 2),
            dtype=self.precision
        )
        self.range_doppler_maps = np.zeros(
            (self.num_samples, self.num_doppler_bins, self.num_range_bins),
            dtype=self.precision
        )
        self.target_masks = np.zeros(
            (self.num_samples, self.num_doppler_bins, self.num_range_bins, 1),
            dtype=self.precision
        )
        self.target_info = []
        self.cfar_detections = []

        if self.save_path is not None:
            vis_path = os.path.join(self.save_path, 'visualizations')
            if self.drawfig and not os.path.exists(vis_path):
                os.makedirs(vis_path)
        else:
            vis_path = None

        for i in tqdm(range(self.num_samples)):
            targets = self.generate_targets()
            snr_db = np.random.uniform(self.SNR_dB_min, self.SNR_dB_max)

            sim_targets = list(targets)
            if self.apply_realistic_effects:
                sim_targets.extend(self._generate_clutter_targets())
                sim_targets.append(self._generate_coupling_target())

            if self.signal_type == 'OTFS':
                beat_rx, rdm = self.simulate_otfs_signal(sim_targets, snr_db)
            else:
                beat_rx = self.simulate_fmcw_signal(sim_targets, snr_db)
                rdm = fmcw_rd_from_beats(
                    beat_rx,
                    fs=self.fs,
                    T=self.T,
                    slope=self.slope,
                    zero_pad=self.zero_pad,
                    R_max=self.R_max,
                    num_range_bins=self.num_range_bins
                )

            if self.drawfig and i < 3 and VISUALIZATION_AVAILABLE and vis_path is not None:
                beat_chirp = beat_rx[0, 0, :]
                plot_signal_time_and_spectrum(
                    signal=beat_chirp,
                    sample_rate=self.fs,
                    total_duration=self.T,
                    title_prefix=f"Beat Signal Sample {i}",
                    textstr=None,
                    normalize=False,
                    save_path=os.path.join(vis_path, f"beat_signal_{i}.png"),
                    draw_window=False
                )

            self.time_domain_data[i, :, :, :, 0] = beat_rx.real.astype(self.precision)
            self.time_domain_data[i, :, :, :, 1] = beat_rx.imag.astype(self.precision)
            self.range_doppler_maps[i] = rdm.astype(self.precision)

            mask = self.create_target_mask(targets)
            self.target_masks[i, :, :, 0] = mask.astype(self.precision)

            cfar_results = self.cfar_detection(rdm)
            self.cfar_detections.append(cfar_results)

            self.target_info.append({
                'targets': targets,
                'snr_db': snr_db,
                'sample_idx': i,
                'cfar_detections': cfar_results
            })

            if self.drawfig and i < 5 and vis_path is not None:
                self.plot_sample(i, targets, rdm, vis_path, detections=cfar_results)

        if self.save_path is not None:
            print(f"Dataset generation complete. Saving to {self.save_path}")
            self.save_dataset()
        else:
            print("Dataset generation complete. Skipping save (save_path=None).")

    def _evaluate_metrics(self, targets, detections, match_dist_thresh=3.0):
        tp = 0
        range_errors = []
        vel_errors = []

        unmatched_targets = targets.copy()
        unmatched_detections = detections.copy()
        matched_pairs = []

        for t in targets:
            best_det = None
            best_dist = float('inf')
            best_det_idx = -1
            for i, d in enumerate(unmatched_detections):
                d_r = t['range'] - d['range_m']
                d_v = t['velocity'] - d['velocity_mps']
                dist = np.sqrt(d_r**2 + d_v**2)
                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det = d
                    best_det_idx = i
            if best_det is not None:
                tp += 1
                range_errors.append(abs(t['range'] - best_det['range_m']))
                vel_errors.append(abs(t['velocity'] - best_det['velocity_mps']))
                matched_pairs.append((t, best_det))
                unmatched_targets.remove(t)
                unmatched_detections.pop(best_det_idx)

        fp = len(unmatched_detections)
        fn = len(unmatched_targets)

        mean_range_error = np.mean(range_errors) if range_errors else 0.0
        mean_vel_error = np.mean(vel_errors) if vel_errors else 0.0

        metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'mean_range_error': mean_range_error,
            'mean_velocity_error': mean_vel_error,
            'num_targets': len(targets),
            'num_detections': len(detections)
        }
        return metrics, matched_pairs, unmatched_targets, unmatched_detections

    def plot_sample(self, sample_idx, targets, rdm, save_dir, detections=None):
        rdm_norm = rdm - np.max(rdm)
        detection_results = detections if detections is not None else self.cfar_detection(rdm)
        metrics, matched_pairs, unmatched_targets, unmatched_detections = self._evaluate_metrics(
            targets, detection_results
        )
        save_path_2d = os.path.join(save_dir, f"rdm_sample_{sample_idx}.png")
        _plot_2d_rdm(self, rdm_norm, sample_idx, metrics,
                     matched_pairs, unmatched_targets, unmatched_detections, save_path_2d)

        save_path_3d = os.path.join(save_dir, f"rdm_3d_sample_{sample_idx}.png")
        _plot_3d_rdm(self, rdm_norm, sample_idx, targets, detection_results, save_path_3d)

        print(f"Saved 2D visualization: {save_path_2d}")
        print(f"Saved 3D visualization: {save_path_3d}")
        print(f"CFAR detected {metrics['num_detections']} vs {metrics['num_targets']} ground truth targets")

    def save_dataset(self):
        os.makedirs(self.save_path, exist_ok=True)
        save_file = os.path.join(self.save_path, "radar_dataset.h5")
        with h5py.File(save_file, 'w') as f:
            f.create_dataset('time_domain_data', data=self.time_domain_data, compression='gzip')
            f.create_dataset('range_doppler_maps', data=self.range_doppler_maps, compression='gzip')
            f.create_dataset('target_masks', data=self.target_masks, compression='gzip')
            f.create_dataset('range_axis', data=self.range_axis)
            f.create_dataset('velocity_axis', data=self.velocity_axis)

            f.attrs['fc'] = self.fc
            f.attrs['B'] = self.B
            f.attrs['T_chirp'] = self.T
            f.attrs['N_samples'] = self.Ns
            f.attrs['N_chirps'] = self.Nc
            f.attrs['R_max'] = self.R_max
            f.attrs['fs'] = self.fs
            f.attrs['range_resolution'] = self.range_resolution
            f.attrs['velocity_resolution'] = self.velocity_resolution
            f.attrs['num_rx'] = self.num_rx

            import json
            info_str = [json.dumps(info, default=str) for info in self.target_info]
            f.create_dataset('target_info', data=info_str, dtype=h5py.string_dtype())
        print(f"Dataset saved to: {save_file}")

    def _load_data(self, datapath):
        with h5py.File(datapath, 'r') as f:
            self.time_domain_data = f['time_domain_data'][:]
            self.range_doppler_maps = f['range_doppler_maps'][:]
            self.target_masks = f['target_masks'][:]
            import json
            info_str = f['target_info'][:]
            self.target_info = [json.loads(s) for s in info_str]
            self.cfar_detections = []
            for info in self.target_info:
                if 'cfar_detections' in info:
                    self.cfar_detections.append(info['cfar_detections'])
                else:
                    self.cfar_detections.append([])
        print(f"Loaded dataset with {len(self.target_info)} samples")
        self.num_samples = len(self.target_info)

    def __len__(self):
        return self.num_samples if self.range_doppler_maps is not None else 0

    def __getitem__(self, idx):
        if self.range_doppler_maps is None:
            raise ValueError("Dataset not generated or loaded")

        raw_np = self.time_domain_data[idx]

        if np.iscomplexobj(raw_np):
            raw_tensor = torch.from_numpy(raw_np)
            time_domain_formatted = torch.stack([raw_tensor.real, raw_tensor.imag], dim=-1).float()
        elif getattr(raw_np, 'dtype', None) is not None and raw_np.dtype.names is not None:
            if 'r' in raw_np.dtype.names:
                real = torch.from_numpy(raw_np['r'])
                imag = torch.from_numpy(raw_np['i'])
            elif 'real' in raw_np.dtype.names:
                real = torch.from_numpy(raw_np['real'])
                imag = torch.from_numpy(raw_np['imag'])
            else:
                real = torch.from_numpy(raw_np[raw_np.dtype.names[0]])
                imag = torch.from_numpy(raw_np[raw_np.dtype_names[1]])
            time_domain_formatted = torch.stack([real, imag], dim=-1).float()
        else:
            t = torch.from_numpy(raw_np).float()
            if t.dim() == 4 and t.shape[-1] == 2:
                time_domain_formatted = t
            elif t.dim() == 3:
                if t.shape[-1] == 2:
                    time_domain_formatted = t
                else:
                    time_domain_formatted = torch.stack([t, torch.zeros_like(t)], dim=-1)
            else:
                time_domain_formatted = torch.stack([t, torch.zeros_like(t)], dim=-1)

        return {
            'time_domain': time_domain_formatted,
            'range_doppler_map': torch.from_numpy(self.range_doppler_maps[idx]).float(),
            'target_mask': torch.from_numpy(self.target_masks[idx]).float(),
            'target_info': self.target_info[idx],
            'cfar_detections': self.cfar_detections[idx] if hasattr(self, 'cfar_detections') and idx < len(self.cfar_detections) else [],
            'range_axis': self.range_axis,
            'velocity_axis': self.velocity_axis
        }


# ======================================================================
# Config, Hardware, Simulation, Processor, Main
# ======================================================================

class RadarConfig:
    def __init__(self, config_name='config_cn0566', mode='simulation',
                 sdr_ip="ip:192.168.86.40", phaser_ip="ip:phaser.local:50901"):
        self.mode = mode
        self.config_name = config_name

        if config_name in RADAR_CONFIGS:
            self.params = RADAR_CONFIGS[config_name].copy()
        else:
            print(f"Warning: Config {config_name} not found. Falling back to 'config_phaser'.")
            self.params = RADAR_CONFIGS['config_phaser'].copy()
            self.config_name = 'config_phaser'

        self.hardware_params = {
            'sdr_ip': sdr_ip,
            'phaser_ip': phaser_ip,
            'rx_gain': 30,
            'tx_gain': -10,
            'rx_channels': [0, 1],
            'tx_channels': [0, 1],
            'buffer_size': 1024 * 16,
            'tdd_mode': False,
            'ramp_mode': "continuous_triangular"
        }

    def update_hardware_params(self, **kwargs):
        self.hardware_params.update(kwargs)

    def get_dataset_params(self):
        return {
            'config_name': self.config_name,
            'fc': self.params['fc'],
            'B': self.params['B'],
            'T_chirp': self.params['T_chirp'],
            'fs': self.params['fs'],
            'N_chirps': self.params['N_chirps'],
            'R_max': self.params['R_max']
        }


class RadarHardware:
    def __init__(self, config: RadarConfig):
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Hardware libraries not available.")
        self.config = config
        self.sdr = None
        self.phaser = None
        self.tdd = None

        self._setup_sdr()
        self._setup_phaser()
        if self.config.hardware_params['tdd_mode']:
            self._setup_tdd()

    def _setup_sdr(self):
        print(f"Initializing SDR at {self.config.hardware_params['sdr_ip']}...")
        self.sdr = SDR(
            SDR_IP=self.config.hardware_params['sdr_ip'],
            SDR_FC=int(2.1e9),
            SDR_SAMPLERATE=int(self.config.params['fs']),
            SDR_BANDWIDTH=int(self.config.params['fs']),
            Rx_CHANNEL=len(self.config.hardware_params['rx_channels']),
            Tx_CHANNEL=len(self.config.hardware_params['tx_channels'])
        )
        self.sdr.SDR_RX_setup(
            n_SAMPLES=self.config.hardware_params['buffer_size'],
            controlmode='manual',
            rx1_gain=self.config.hardware_params['rx_gain'],
            rx2_gain=self.config.hardware_params['rx_gain']
        )
        self.sdr.SDR_TX_setup(
            cyclic_buffer=True,
            tx1_gain=self.config.hardware_params['tx_gain'],
            tx2_gain=self.config.hardware_params['tx_gain']
        )

    def _setup_phaser(self):
        if not self.config.hardware_params['phaser_ip']:
            print("Phaser IP not provided. Skipping Phaser setup.")
            return
        print(f"Initializing Phaser at {self.config.hardware_params['phaser_ip']}...")
        self.phaser = mycn0566.CN0566(
            uri=self.config.hardware_params['phaser_ip'],
            sdr=self.sdr.sdr
        )
        self.phaser.configure(device_mode="rx")
        self.phaser.load_gain_cal()
        self.phaser.load_phase_cal()
        for i in range(8):
            self.phaser.set_chan_phase(i, 0)
            self.phaser.set_chan_gain(i, 127, apply_cal=True)

        output_freq = 12.1e9
        BW = self.config.params['B']
        ramp_time_us = self.config.params['T_chirp'] * 1e6

        self.phaser.frequency = int(output_freq / 4)
        self.phaser.freq_dev_range = int(BW / 4)
        self.phaser.freq_dev_step = int((BW / 4) / 1000)
        self.phaser.freq_dev_time = int(ramp_time_us)
        self.phaser.ramp_mode = self.config.hardware_params['ramp_mode']
        self.phaser.enable = 0

    def _setup_tdd(self):
        print("Configuring TDD mode...")
        self.tdd = tddn(self.config.hardware_params['sdr_ip'])
        self.tdd.enable = False
        self.tdd.sync_external = True
        self.tdd.frame_length_ms = (self.config.params['T_chirp'] * 1000) + 1.0
        self.tdd.burst_count = self.config.params['N_chirps']
        self.tdd.channel[0].enable = True
        self.tdd.channel[1].enable = True
        self.tdd.enable = True

    def transmit(self):
        # For CN0566, chirp is generated by Phaser PLL; SDR might only provide IF tone.
        pass

    def receive(self):
        """
        Receive one CPI and reshape to [Rx, Nc, Ns] complex.
        NOTE: This implementation assumes SDR_RX_receive returns [num_rx, total_samples].
        You may need to adapt based on your SDR wrapper.
        """
        raw = self.sdr.SDR_RX_receive(combinerule='none', normalize=False)
        # raw: [num_rx, total_samples]
        num_rx = raw.shape[0]
        total_samples_per_rx = raw.shape[1]
        Nc = self.config.params['N_chirps']
        Ns = int(self.config.params['fs'] * self.config.params['T_chirp'])

        if total_samples_per_rx != Nc * Ns:
            raise RuntimeError(f"Buffer mismatch: got {total_samples_per_rx}, expected {Nc * Ns}")

        data_reshaped = raw.reshape(num_rx, Nc, Ns)
        return data_reshaped

    def stop(self):
        if self.sdr:
            self.sdr.SDR_TX_stop()
        if self.tdd:
            self.tdd.enable = False


class RadarSimulation:
    def __init__(self, config: RadarConfig):
        self.config = config
        print("Initializing Radar Simulation...")
        # autogen=False; we trigger generation in get_data()
        self.dataset = AIRadarDataset(
            num_samples=1,
            autogen=False,
            save_path=None,
            drawfig=False,
            **self.config.get_dataset_params()
        )

    def get_data(self):
        """
        Generate one simulated frame:
        Returns:
            raw_data: complex ndarray [Rx, Nc, Ns]
            target_info: dict with 'targets' list
        """
        self.dataset.num_samples = 1
        self.dataset.generate_dataset()
        sample = self.dataset[0]
        td = sample['time_domain'].numpy()  # [Rx, Nc, Ns, 2]
        raw_complex = td[..., 0] + 1j * td[..., 1]
        return raw_complex, sample['target_info']


class RadarProcessor:
    def __init__(self, config: RadarConfig):
        self.config = config
        # dataset instance just for parameters & CFAR; no autogen
        self.processor = AIRadarDataset(
            num_samples=1,
            autogen=False,
            save_path=None,
            drawfig=False,
            **self.config.get_dataset_params()
        )

    def process_frame(self, raw_data):
        """
        raw_data: complex [Rx, Nc, Ns] or [Nc, Ns]
        returns rdm: [Nc, Nr] in dB
        """
        rdm = fmcw_rd_from_beats(
            raw_data,
            fs=self.processor.fs,
            T=self.processor.T,
            slope=self.processor.slope,
            zero_pad=self.processor.zero_pad,
            R_max=self.processor.R_max,
            num_range_bins=self.processor.num_range_bins
        )
        return rdm

    def detect(self, rdm):
        return self.processor.cfar_detection(rdm)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="MyRadar - Simulation/Hardware Controller")
    parser.add_argument('--mode', type=str, default='simulation',
                        choices=['hardware', 'simulation'], help='Operation mode')
    parser.add_argument('--config', type=str, default='config_cn0566',
                        help='Radar configuration name, config1, config2, config_cn0566')
    parser.add_argument('--save_path', type=str, default='data/cn0566_eval', help='Path to save')
    parser.add_argument('--sdr_ip', type=str, default="ip:192.168.86.40", help='SDR IP address')
    parser.add_argument('--phaser_ip', type=str, default="ip:phaser.local:50901", help='Phaser IP address')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to capture/simulate')
    parser.add_argument('--plot', default=True, help='Enable visualization')
    parser.add_argument('--eval_sim_cn0566', default=True,
                        help='Run full CN0566 simulation dataset (100 samples) and evaluate CFAR')
    args = parser.parse_args() #action='store_true',

    # Optional: run a full CN0566 offline evaluation
    if args.eval_sim_cn0566:
        print(f"Running CN0566 simulation evaluation with {args.config} config...")
        dataset_cn = AIRadarDataset(
            num_samples=100,
            config_name=args.config,
            save_path=args.save_path,
            drawfig=False,
            apply_realistic_effects=True,
            clutter_intensity=0.2
        )
        evaluate_dataset_metrics(dataset_cn, "Config CN0566 (Offline Sim)")
        return

    # 1. Config
    config = RadarConfig(
        config_name=args.config,
        mode=args.mode,
        sdr_ip=args.sdr_ip,
        phaser_ip=args.phaser_ip
    )

    # 2. Source
    if args.mode == 'hardware':
        if not HARDWARE_AVAILABLE:
            print("Error: Hardware mode requested but hardware libs not available.")
            return
        source = RadarHardware(config)
    else:
        source = RadarSimulation(config)

    # 3. Processor
    processor = RadarProcessor(config)

    # 4. Main loop
    try:
        for i in range(args.frames):
            print(f"\n--- Frame {i+1}/{args.frames} ---")
            start_time = timer()
            if args.mode == 'hardware':
                raw_data = source.receive()  # [Rx, Nc, Ns]
                targets = []
            else:
                raw_data, target_info = source.get_data()
                targets = target_info['targets']

            rdm = processor.process_frame(raw_data)
            detections = processor.detect(rdm)
            proc_time = timer() - start_time

            print(f"Processing Time: {proc_time*1000:.2f} ms")
            print(f"Detections: {len(detections)}")

            if args.plot:
                os.makedirs("output", exist_ok=True)
                save_path = f"output/myradar_frame_{i}.png"

                if targets:
                    metrics, matched, unmatched_t, unmatched_d = processor.processor._evaluate_metrics(
                        targets, detections
                    )
                    _plot_2d_rdm(processor.processor, rdm, i, metrics,
                                 matched, unmatched_t, unmatched_d, save_path)
                else:
                    dummy_metrics = {
                        'num_targets': 0,
                        'num_detections': len(detections),
                        'tp': 0,
                        'fp': len(detections),
                        'fn': 0,
                        'mean_range_error': 0.0,
                        'mean_velocity_error': 0.0
                    }
                    _plot_2d_rdm(processor.processor, rdm, i, dummy_metrics,
                                 [], [], detections, save_path)
                print(f"Saved visualization to {save_path}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if args.mode == 'hardware':
            source.stop()


if __name__ == "__main__":
    main()