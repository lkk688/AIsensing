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
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch not found. Running without PyTorch features.")
    TORCH_AVAILABLE = False
    class Dataset: pass  # Dummy class if torch is missing
try:
    import h5py
except ImportError:
    h5py = None

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


# Deep Learning Imports
try:
    import torch
    import torch.nn.functional as F
    
    # Add parent directory to path to find AIRadar
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AIRadar')))
    # Assume script is run from project root or sdradi
    # We need to find the AIRadar module.
    # If this file is in sdradi/, then AIRadar is in ../AIRadar
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.append(base_dir)

    from AIRadar.AIradar_comm_model_g4 import RadarNetG3, ConfigEncoder
    DL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Deep Learning models not available: {e}")
    DL_AVAILABLE = False


def _os_cfar_torch(rdm, num_train, num_guard, percentile=75, device=None):

    """
    Accelerated OS-CFAR using PyTorch Unfold + TopK/Sort.
    dims: rdm [H, W] (real valued)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Torch not available for OS-CFAR acceleration")
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Prepare Input
    H, W = rdm.shape
    inp = torch.from_numpy(rdm).to(device).float() # [H, W]
    inp = inp.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # Kernel parameters
    nt = num_train
    ng = num_guard
    kernel_size = 1 + 2*(nt + ng)
    pad = nt + ng
    
    # Unfold -> [1, K*K, L] where L = H*W
    # padding='same' behavior via manual padding
    # torch.nn.functional.unfold pads explicitly? 
    # unfold expects 4D input.
    # We pad input manually to handle boundary condition 'same'
    # scipy percentile_filter uses 'reflect', 'constant', 'nearest', 'mirror', 'wrap'. Default is 'reflect'.
    # We'll use reflection padding to match roughly.
    inp_padded = torch.nn.functional.pad(inp, (pad, pad, pad, pad), mode='reflect')
    
    # Unfold
    # output: [N, C*K*K, L]
    windows = torch.nn.functional.unfold(inp_padded, kernel_size=kernel_size) 
    # windows: [1, K*K, H*W]
    
    # Masking Guard + CUT
    # We need to ignore the central (2*ng + 1) x (2*ng + 1) region
    # Window size K = 2*(nt+ng) + 1
    # Center is at index K//2
    # We can create a boolean mask of valid indices
    K = kernel_size
    valid_mask = torch.ones((K, K), device=device, dtype=torch.bool)
    
    center = K // 2
    start_g = center - ng
    end_g = center + ng + 1
    valid_mask[start_g:end_g, start_g:end_g] = False
    
    valid_indices = valid_mask.view(-1).nonzero(as_tuple=True)[0]
    # valid_indices size = K*K - (2*ng+1)^2
    
    # Select only valid elements from windows
    # windows [1, K*K, L] -> [1, N_valid, L]
    valid_windows = windows[:, valid_indices, :] 
    
    # Compute K-th value
    # We want the p-th percentile.
    # index k = p/100 * (N_valid - 1)
    N_valid = valid_windows.shape[1]
    k_idx = int(percentile / 100.0 * (N_valid - 1))
    
    # torch.kthvalue finds the k-th smallest element (1-based index)
    # k_idx is 0-based index for sorted array. So we need k_idx + 1
    values, _ = torch.kthvalue(valid_windows, k=k_idx + 1, dim=1) 
    
    # Reshape back to [H, W]
    noise_est = values.view(H, W)
    
    return noise_est.cpu().numpy()

VISUALIZATION_AVAILABLE = True
# Helper: 3D Plotting (Ported from AIRadarLib/visualization.py to fix boundary issues)
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
    """
    from mpl_toolkits.mplot3d import Axes3D
    
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
        
        # Check boundary
        r_ok = 0 <= r_val <= (num_range_bins * range_resolution)
        v_min, v_max = velocity_axis[0], velocity_axis[-1]
        v_ok = v_min <= v_val <= v_max
        
        if r_ok and v_ok:
            # Find indices for z-value lookup (approximate)
            r_idx = int(r_val / range_resolution)
            if apply_doppler_centering:
                v_idx = int(num_doppler_bins // 2 + v_val / velocity_resolution)
            else:
                v_idx = int(v_val / velocity_resolution) % num_doppler_bins
                
            r_idx = np.clip(r_idx, 0, num_range_bins - 1)
            v_idx = np.clip(v_idx, 0, num_doppler_bins - 1)
            
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
# --- Plot view limits ---
VIEW_RANGE_LIMITS = (0, 150) # Updated to match R_max of CN0566
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
        # CN0566 G2 Config
        'fc': 10.25e9,
        'B': 500e6,
        'T_chirp': 500e-6,
        'fs': 2.0e6,
        'N_chirps': 64,
        'R_max': 150.0,
        'R_max': 150.0,
        'hardware_model': 'CN0566',
        'num_rx': 1, # G2 uses 1 Rx for TRADITIONAL mode to simplify correlation
        'use_array_factor': False, # Disabled array factor for 1 Rx
        'array_N': 8,
        'steering_angles': [0.0, 0.0],
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.1,
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.005,
            'k_shape': 2.0,
            'range_exponent': 2.5,
            'weather_clutter': False,
            'weather_intensity': 0.02,
            'doppler_spread': 3.0
        },
        'cfar_params': {
            'num_train': 12, 
            'num_guard': 4, 
            'threshold_offset': 25, 
            'nms_kernel_size': 7
        },

        # Hardware impairments (legacy params kept for compatibility)
        # Use explicit flag to disable them for G2 consistency
        # Hardware impairments (legacy params kept for compatibility)
        # Use explicit flag to disable them for G2 consistency
        'disable_impairments': False,
        'model_dc_offset': True,
        'model_iq_imbalance': True,
        'model_phase_noise': True,
        'quantize_adc': True,
        'adc_bits': 12,
        'dc_scale': 0.02, # Tuned: 2% DC offset (realistic)
        'iq_gain_std': 0.01, # Tuned: 1% imbalance
        'iq_phase_std_deg': 1.0, # Tuned: 1 degree imbalance
        'phase_noise_std': 0.01, # Tuned: 1% RMS (~0.5 deg)
        'cfo_std_hz': 200.0,
        'static_clutter_velocity_std': 0.02,
        'ground_clutter_velocity_std': 0.1,
        'coupling_rcs_db': 0.0,
        
        # Compensation (DSP)
        'enable_compensation': True,
        'enable_phase_noise_tracking': True,
        'cfar_type': 'OS' # Options: 'CA', 'OS'
    },
    'config_loopback': {
        'name': 'Pluto_Digital_Loopback',
        'signal_type': 'FMCW',
        'fc': 10.25e9,
        'B': 500e6,
        'T_chirp': 500e-6,
        'fs': 2.0e6,
        'N_chirps': 64,
        'R_max': 150.0,
        'hardware_model': 'loopback',
        'num_rx': 1,
        'use_array_factor': False,
        'disable_impairments': True,
        'model_dc_offset': False,
        'model_iq_imbalance': False,
        'model_phase_noise': False,
        'quantize_adc': False,
        'cfar_params': {
            'num_train': 12,
            'num_guard': 4,
            'threshold_offset': 20,
            'nms_kernel_size': 7
        },
        'adaptive_cfar': True,
        'enable_compensation': False,
        'enable_phase_noise_tracking': False,
        'cfar_type': 'OS'
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
    max_vis_samples = min(10, len(dataset))

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
            rdm = sample['range_doppler_map']
            if hasattr(rdm, 'numpy'):
                rdm = rdm.numpy()
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

        # G2 Enhancements
        self.adaptive_cfar = cfg.get('adaptive_cfar', True)
        self.clutter_params = cfg.get('clutter_params', {})
        self.csi_error = cfg.get('csi_error', 0.1)

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
            
        # DL Model Init
        self.dl_model = None
        self.dl_device = None
        if DL_AVAILABLE and TORCH_AVAILABLE:
            self.dl_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_dl_model()

    def _load_dl_model(self):
        """Load the pre-trained radar detection model (RobustRadarNetG3 or RadarNetG3)."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            # Search paths in priority order
            search_paths = [
                os.path.join(base_dir, 'data/g3_radar/radar_best_fmcw.pt'),
                os.path.join(base_dir, 'AIRadar/data/AIradar_comm_model_g3b/radar_best_fmcw.pt'),
                os.path.join(base_dir, 'data/AIradar_comm_model_g3b/radar_best_fmcw.pt'),
            ]

            ckpt_path = None
            for p in search_paths:
                if os.path.exists(p):
                    ckpt_path = p
                    break

            if ckpt_path is None:
                print("[DL] No checkpoint found. DL Detection disabled.")
                return

            print(f"[DL] Loading model from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=self.dl_device,
                                    weights_only=False)

            # Detect model type from checkpoint
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                # New RobustRadarNetG3 with saved config
                cfg = checkpoint['config']
                try:
                    sys.path.insert(0, os.path.join(base_dir, 'newsdr'))
                    from train_g3_radar import RobustRadarNetG3
                    self.dl_model = RobustRadarNetG3(**cfg).to(self.dl_device)
                    self.dl_model.load_state_dict(checkpoint['model'])
                    print("[DL] Loaded RobustRadarNetG3 model.")
                except Exception as e_robust:
                    print(f"[DL] RobustRadarNetG3 load failed ({e_robust}), trying RadarNetG3...")
                    self.dl_model = RadarNetG3(cond_dim=64).to(self.dl_device)
                    self.dl_model.load_state_dict(checkpoint['model'])
                    print("[DL] Loaded legacy RadarNetG3 model.")
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.dl_model = RadarNetG3(cond_dim=64).to(self.dl_device)
                self.dl_model.load_state_dict(checkpoint['model'])
                print("[DL] Loaded RadarNetG3 model.")
            else:
                self.dl_model = RadarNetG3(cond_dim=64).to(self.dl_device)
                self.dl_model.load_state_dict(checkpoint)
                print("[DL] Loaded RadarNetG3 model (bare state_dict).")

            self.dl_model.eval()
            print("[DL] Model loaded successfully.")

        except Exception as e:
            print(f"[DL] Error loading model: {e}")
            self.dl_model = None

    def run_dl_detection(self, rdm_db, threshold=0.5):
        """
        Run Deep Learning based detection.
        """
        if self.dl_model is None:
            return []
            
        H, W = rdm_db.shape
        # rdm_db is [Doppler, Range]
        # RadarNet expects [B, 1, R, D] or [B, 1, D, R]?
        # G4 Model: Input: Range-Doppler Map [B, 1, H, W]
        # Let's check G3Dataset: rdm = np.array(sample['range_doppler_map']) [H, W]
        # It passes [1, 1, H, W] to model.
        # So we pass it as is.
        
        # 1. Preprocess
        rdm_min = rdm_db.min()
        rdm_max = rdm_db.max()
        if rdm_max - rdm_min < 1e-6:
            rdm_norm = np.zeros_like(rdm_db)
        else:
            rdm_norm = (rdm_db - rdm_min) / (rdm_max - rdm_min)
            
        # To Tensor [1, 1, H, W]
        inp_tensor = torch.from_numpy(rdm_norm).float().unsqueeze(0).unsqueeze(0).to(self.dl_device)
        
        # 2. Config Embedding
        # [fc, bw, n_sub, mod_order, snr_db, res, max_range, max_vel]
        cfg_tensor = torch.tensor([
            self.fc / 1e9,
            self.B / 1e9,
            64.0 / 64.0,
            16.0 / 64.0,
            30.0 / 30.0,
            self.range_resolution,
            self.R_max / 100.0,
            self.v_max / 50.0
        ], dtype=torch.float32).unsqueeze(0).to(self.dl_device)
        
        # 3. Inference
        with torch.no_grad():
            heatmap_logits = self.dl_model(inp_tensor, cfg_tensor)
            heatmap = torch.sigmoid(heatmap_logits).cpu().numpy().squeeze()
            
        # 4. Post-process
        detections_map = heatmap > threshold
        
        # Find peaks
        doppler_idxs, range_idxs = np.where(detections_map)
        
        results = []
        num_doppler = H
        
        dr = self.range_axis[1] - self.range_axis[0]
        dv = self.velocity_axis[1] - self.velocity_axis[0]
        
        for d_idx, r_idx in zip(doppler_idxs, range_idxs):
            val = heatmap[d_idx, r_idx]
            
            # Simple NMS (3x3)
            s_d = max(0, d_idx-1)
            e_d = min(H, d_idx+2)
            s_r = max(0, r_idx-1)
            e_r = min(W, r_idx+2)
            local_max = np.max(heatmap[s_d:e_d, s_r:e_r])
            if val < local_max:
                continue
                
            range_m = r_idx * dr
            velocity_mps = (d_idx - num_doppler // 2) * dv
            
            results.append({
                "range_idx": int(r_idx),
                "doppler_idx": int(d_idx),
                "range_m": float(range_m),
                "velocity_mps": float(velocity_mps),
                "magnitude": float(val * 100)
            })
            
        return results

    # ------------------------------------------------------------------
    # G2 Enhancement: Helper Methods (SNR, Clutter, etc.)
    # ------------------------------------------------------------------
    def _estimate_snr(self, rdm_db):
        """Estimate SNR from RDM statistics using noise floor estimation"""
        noise_floor = np.percentile(rdm_db, 25)
        peak_power = np.max(rdm_db)
        return peak_power - noise_floor
    
    def _compute_adaptive_threshold(self, rdm_db, base_threshold):
        """Compute adaptive threshold based on estimated SNR"""
        estimated_snr = self._estimate_snr(rdm_db)
        # Adaptive logic from G2
        if estimated_snr > 35:
            return base_threshold - 5
        elif estimated_snr > 30:
            return base_threshold - 3
        elif estimated_snr > 20:
            return base_threshold
        elif estimated_snr > 15:
            return base_threshold + 3
        else:
            return base_threshold + 5

    def _generate_ground_clutter(self, rdm_shape, r_axis):
        if not self.config.get('adaptive_cfar', False) or not self.clutter_params.get('ground_clutter', False):
            return np.zeros(rdm_shape)
        
        intensity = self.clutter_params.get('ground_intensity', 0.05)
        k_shape = self.clutter_params.get('k_shape', 2.0)
        range_exp = self.clutter_params.get('range_exponent', 2.5)
        
        range_profile = (r_axis + 1) ** (-range_exp)
        range_profile = range_profile / np.max(range_profile)
        
        gamma_samples = np.random.gamma(k_shape, 1.0 / k_shape, rdm_shape)
        rayleigh_samples = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        clutter = gamma_samples * rayleigh_samples
        
        if len(r_axis) == rdm_shape[1]:
            clutter = clutter * range_profile[None, :]
        return intensity * clutter

    def _generate_weather_clutter(self, rdm_shape, v_axis):
        if not self.config.get('adaptive_cfar', False) or not self.clutter_params.get('weather_clutter', False):
            return np.zeros(rdm_shape)
        
        intensity = self.clutter_params.get('weather_intensity', 0.03)
        doppler_spread = self.clutter_params.get('doppler_spread', 3.0)
        
        doppler_profile = np.exp(-v_axis**2 / (2 * doppler_spread**2))
        doppler_profile = doppler_profile / np.max(doppler_profile)
        
        weather = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        if len(v_axis) == rdm_shape[0]:
            weather = weather * doppler_profile[:, None]
        return intensity * weather

    def _add_clutter_to_rdm(self, rdm_db, r_axis, v_axis):
        """Add combined clutter to RDM (in dB domain)"""
        rdm_linear = 10 ** (rdm_db / 20)
        
        # Scale clutter relative to signal peak for realistic CNR
        signal_peak = np.percentile(rdm_linear, 99)
        # Clutter intensity from init
        clutter_scale = self.clutter_intensity * signal_peak
        
        ground = self._generate_ground_clutter(rdm_linear.shape, r_axis) * clutter_scale
        weather = self._generate_weather_clutter(rdm_linear.shape, v_axis) * clutter_scale * 0.5
        
        rdm_with_clutter = rdm_linear + ground + weather
        return 20 * np.log10(rdm_with_clutter + 1e-9)

    def compute_rdm(self, beat_input):
        """
        Unified processing: Beat Signal -> RDM
        Input: [Rx, Nc, Ns] or [Nc, Ns]
        Output: [Nc, Nr] RDM in dB
        Output: [Nc, Nr] RDM in dB
        """
        
        # 0. Compensation (DSP)
        if self.config.get('enable_compensation', False):
            beat_input = self._compensate_impairments(beat_input)

        # 1. Collapse Rx if needed (Beamforming or Sum)
        if beat_input.ndim == 3:
             # Simple sum beamforming (can be improved with steering vector)
            beat = np.sum(beat_input, axis=0) 
        else:
            beat = beat_input

        Nc, Ns = beat.shape
        # 2. Windowing
        win_range = np.hanning(Ns)[None, :]
        win_doppler = np.hanning(Nc)[:, None]
        beat_win = beat * win_range * win_doppler

        # 3. FFTs
        range_fft = np.fft.fft(beat_win, n=self.zero_pad, axis=1)
        # Crop to range
        range_fft = range_fft[:, :self.num_range_bins]
        
        # Doppler FFT
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        
        # 4. Magnitude (dB)
        rdm_db = 20 * np.log10(np.abs(doppler_fft) + 1e-9)

        # 5. Add Clutter (Simulation only? Or always if configured?)
        # For hardware data, we don't add clutter usually, unless we want to stress test.
        # But the prompt says "make the processing part code the same".
        # If we are processing hardware data, we might want to skip clutter addition unless explicitly desired.
        # Check if we are in simulation or hardware mode? 
        # Actually, clutter addition is a "Simulation" artifact. 
        # But if we use this for hardware, we shouldn't add clutter.
        # However, the user said "simulation follow the hardware setup". 
        # I'll rely on `self.apply_realistic_effects` flag or check if input is from hardware.
        # For now, I'll add it if enabled (which is default True in simulation).
        # In hardware mode, we should likely disable this or set intensity to 0.
        if self.apply_realistic_effects and self.clutter_intensity > 0:
             rdm_db = self._add_clutter_to_rdm(rdm_db, self.range_axis, self.velocity_axis)
             
        return rdm_db

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
        
        # G2 Consistency: Check for disable flag
        if self.config.get('disable_impairments', False):
            return beat_rx

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

        # windows - MOVED TO compute_rdm to ensure hardware uses it too
        # But we need windows here if we add time-domain clutter before? 
        # G2 adds time-domain clutter BEFORE windowing.
        
        # G2: Add time-domain clutter (Ground reflections etc)
        # Only add significant clutter when clutter_intensity is HIGH
        if self.apply_realistic_effects and self.clutter_intensity > 0.2:
             sig_pow = np.mean(np.abs(beat_rx)**2)
             clutter_power = sig_pow * self.clutter_intensity * 0.01
             
             clutter_time = np.zeros((self.Nc, self.Ns), dtype=np.complex128)
             # Simple ground clutter model for time domain
             num_clutter_gates = max(3, self.Ns // 100)
             for range_idx in range(0, self.Ns, self.Ns // num_clutter_gates):
                 clutter_amp = np.sqrt(clutter_power) * np.random.uniform(0.5, 1.5)
                 low_doppler = np.random.uniform(-2, 2)
                 phase = 2 * np.pi * low_doppler * self.t_slow[:, None]
                 gate_width = min(5, self.Ns // 50)
                 # Apply to all Rx
                 clutter_slice = (clutter_amp * np.exp(1j * phase)[:, :gate_width*2])
                 # Broadcast to Rx
                 # beat_rx is [Rx, Nc, Ns]
                 # We need to add to specific range bins
                 start = max(0, range_idx-gate_width)
                 end = min(self.Ns, range_idx+gate_width)
                 width = end - start
                 if width > 0:
                      beat_rx[:, :, start:end] += clutter_slice[:, :width]

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
            })
        return results

    # ------------------------------------------------------------------
    # CFAR Detection (G2 Exact Logic)
    # ------------------------------------------------------------------
    def cfar_detection(self, rd_map):
        """
        Wrapper for G2-style CFAR.
        Adapter to match the surrounding code structure while using strict G2 logic.
        """
        rdm_used = rd_map.copy()
        # G2 Consistency: G2 does not use suppress_known_artifacts in _run_cfar.
        # Although useful for hardware, it creates a step in noise floor that biases CFAR (low noise est -> high FP).
        # We disable it here to match G2 simulation performance.
        # if self.hardware_model == 'CN0566':
        #    rdm_used = suppress_known_artifacts(rdm_used, self.range_axis, self.velocity_axis)

        # Call the Logic ported from G2 _run_cfar
        dets = self._run_cfar_g2(rdm_used, self.range_axis, self.velocity_axis)
        
        # Format for compatibility with existing evaluation code
        formatted_dets = []
        for d in dets:
            # G2 returns list of dicts with 'range_m', 'velocity_mps', 'range_idx', 'doppler_idx', 'power'
            # creating a compatible dict for legacy visualization if needed, 
            # though new viz expects range/velocity/rcs.
            d['magnitude'] = d['power']
            formatted_dets.append(d)
        
        return formatted_dets

    def _run_cfar_g2(self, rdm_db, r_axis, v_axis):
        """
        Constant False Alarm Rate (CFAR) Detector (CA-CFAR) - Ported from G2.
        Supports CA-CFAR (Cell Averaging) and OS-CFAR (Ordered Statistic).
        """
        from scipy.signal import convolve2d
        from scipy.ndimage import maximum_filter, percentile_filter

        params = self.cfar_params
        nt = params['num_train']
        ng = params['num_guard']
        base_thresh = params['threshold_offset']
        cfar_type = self.config.get('cfar_type', 'CA') # Default to CA

        # G2: Compute adaptive threshold if enabled (only for CA usually)
        if self.adaptive_cfar and cfar_type == 'CA':
            thresh = self._compute_adaptive_threshold(rdm_db, base_thresh)
        else:
            thresh = base_thresh
        
        # Use dB domain for all modes (unified processing)
        norm_rdm = rdm_db.copy()
        gp = params.get('global_percentile', None)
        if gp is not None:
            pval = np.percentile(norm_rdm, gp)
            norm_rdm = np.minimum(norm_rdm, pval)

        # Kernel construction
        kernel_size = 1 + 2*(nt + ng)
        
        if cfar_type == 'OS':
            # OS-CFAR: robust rank-order filtering
            # We use 75th percentile to robustly estimate noise floor, ignoring outliers/targets.
            # Footprint excludes central guard region? 
            # Simplified OS-CFAR: Percentile on full window (including CUT if not too large).
            # To strictly exclude CUT/Guard, we need a custom footprint.
            footprint = np.ones((kernel_size, kernel_size))
            c = nt + ng
            footprint[c-ng:c+ng+1, c-ng:c+ng+1] = 0 # Mask out Guard + CUT
            
            # Use 75th percentile of the reference cells as the "noise level"
            if TORCH_AVAILABLE:
                try:
                    noise_est = _os_cfar_torch(norm_rdm, num_train=nt, num_guard=ng, percentile=75)
                except Exception as e:
                    # Fallback
                    # print(f"Torch OS-CFAR failed: {e}, falling back to scipy")
                    footprint = np.ones((kernel_size, kernel_size))
                    c = nt + ng
                    footprint[c-ng:c+ng+1, c-ng:c+ng+1] = 0
                    noise_est = percentile_filter(norm_rdm, percentile=75, footprint=footprint)
            else:
                noise_est = percentile_filter(norm_rdm, percentile=75, footprint=footprint)
        else:
            # CA-CFAR: estimate noise level by averaging reference cells
            kernel = np.ones((kernel_size, kernel_size))
            guard_region = 1 + 2*ng
            start_g = nt
            end_g = nt + guard_region
            kernel[start_g:end_g, start_g:end_g] = 0
            kernel /= np.sum(kernel)
            
            noise_est = convolve2d(norm_rdm, kernel, mode='same', boundary='symm')

        detections = norm_rdm > (noise_est + thresh)
        
        # Non-Maximum Suppression (NMS)
        if params.get('nms_kernel_size', 1) > 1:
            local_max = maximum_filter(norm_rdm, size=params['nms_kernel_size'])
            detections = detections & (norm_rdm == local_max)
            
        idxs = np.argwhere(detections)
        
        min_r = params.get('min_range_m', 0.0)
        min_v = params.get('min_speed_mps', 0.0)
        notch_k = params.get('notch_doppler_bins', 0)
        center = len(v_axis) // 2
        candidates = []
        
        for idx in idxs:
            d_idx, r_idx = idx
            if d_idx >= len(v_axis) or r_idx >= len(r_axis): continue
            range_m = r_axis[r_idx]
            vel_mps = v_axis[d_idx]
            # Filter artifacts and near-zero clutter
            if range_m < min_r or abs(vel_mps) < min_v: continue
            if notch_k > 0 and abs(d_idx - center) <= notch_k: continue
            candidates.append({
                'range_m': range_m,
                'velocity_mps': vel_mps, # G2 key
                'velocity': vel_mps,     # Legacy key
                'range': range_m,        # Legacy key
                'range_idx': r_idx,
                'doppler_idx': d_idx,
                'power': norm_rdm[d_idx, r_idx]
            })
        
        # Limit number of peaks by score (power)
        max_peaks = params.get('max_peaks', None)
        if max_peaks is not None:
            candidates.sort(key=lambda x: x['power'], reverse=True)
            candidates = candidates[:max_peaks]

        # Connected-component pruning: retain local maxima within neighborhoods
        pruned = []
        taken = set()
        neigh = params.get('nms_kernel_size', 5)
        
        # Sort by power descending to keep strongest in neighborhood
        candidates.sort(key=lambda x: x['power'], reverse=True)

        for det in candidates:
            key = (det['doppler_idx']//neigh, det['range_idx']//neigh)
            if key in taken:
                continue
            pruned.append(det)
            taken.add(key)
        return pruned

    def _compensate_impairments(self, beat_input):
        """
        Apply blind DSP compensation for DC offset and IQ imbalance.
        Input: [Rx, Nc, Ns] or [Nc, Ns]
        """
        beat_comp = beat_input.copy()
        
        # 1. DC Block (Mean Subtraction)
        # Removes static processing artifacts (Range 0 leakage)
        # Apply per-chirp or over whole frame? 
        # Usually per-chirp or average over chirps.
        # Simple mean subtraction specific to Fast-Time (Range) axis is standard for FMCW DC.
        # But hardware DC is constant over time.
        # Mean across ALL samples (Nc * Ns) removes the global DC.
        if beat_comp.ndim == 3:
            # [Rx, Nc, Ns]
            for r in range(beat_comp.shape[0]):
                mean_val = np.mean(beat_comp[r])
                beat_comp[r] -= mean_val
        else:
            # [Nc, Ns]
            mean_val = np.mean(beat_comp)
            beat_comp -= mean_val
            
        # 2. Blind IQ Imbalance Correction (Gram-Schmidt)
        # Only applicable if we have access to I/Q (complex).
        # beat_input is complex.
        # We process the complex signal to orthogonalize I and Q.
        
        def correct_iq(sig):
            # sig: complex array (flattened or arbitrary shape)
            I = sig.real
            Q = sig.imag
            
            # 2a. Remove DC (already done roughly above, but safe to do again on components)
            I = I - np.mean(I)
            Q = Q - np.mean(Q)
            
            # 2b. Power Balancing
            P_I = np.mean(I**2)
            P_Q = np.mean(Q**2)
            gain_factor = np.sqrt(P_I / (P_Q + 1e-12))
            Q_bal = Q * gain_factor
            
            # 2c. Phase Orthogonalization (Gram-Schmidt)
            # Remove projection of Q on I
            # We want Q_orth such that E[I * Q_orth] = 0
            # E[I * (Q_bal - alpha * I)] = 0
            # E[I*Q_bal] - alpha * E[I^2] = 0
            # alpha = E[I*Q_bal] / E[I^2]
            
            rho = np.mean(I * Q_bal) / (np.mean(I**2) + 1e-12)
            Q_orth = Q_bal - rho * I
            
            # 2d. Re-scale Q_orth to match P_I energy?
            # Theoretically Gram-Schmidt changes energy.
            # Let's re-normalize.
            P_Q_orth = np.mean(Q_orth**2)
            Q_final = Q_orth * np.sqrt(P_I / (P_Q_orth + 1e-12))
            
            return I + 1j * Q_final

        if beat_comp.ndim == 3:
            for r in range(beat_comp.shape[0]):
                beat_comp[r] = correct_iq(beat_comp[r])
        else:
            beat_comp = correct_iq(beat_comp)
            
        # 3. Phase Noise Tracking (CPE Correction) - Data Aided
        # Assumption: The strongest target in the scene dominates the phase error.
        # We track its phase over chirps (slow-time) and remove the non-linear part.
        if self.config.get('enable_phase_noise_tracking', False):
            # Process typically on summed signal or primary Rx
            # We apply to all Rx identically based on the strongest signal found in the sum.
            
            # Working on 'beat_comp' which might be [Rx, Nc, Ns] or [Nc, Ns]
            if beat_comp.ndim == 3:
                ref_sig = np.sum(beat_comp, axis=0) # [Nc, Ns]
            else:
                ref_sig = beat_comp
                
            # Find strongest target in rough Range-Doppler domain
            # Windowing helps
            w_ref = ref_sig * np.hanning(ref_sig.shape[1])[None, :]
            r_fft = np.fft.fft(w_ref, axis=1)
            rd_fft = np.fft.fft(r_fft, axis=0)
            
            # Peak index
            idx_flat = np.argmax(np.abs(rd_fft))
            idx_d, idx_r = np.unravel_index(idx_flat, rd_fft.shape)
            
            # Extract phase history of this peak bin along slow-time (Nc)
            # We look at the Range Bin `idx_r` across all chirps
            # But we must be careful: if we just take `r_fft[:, idx_r]`, it contains modulation due to Doppler.
            # The signal at range bin r is: A * exp(j * (phi_noise(t) + w_doppler * t))
            # We want phi_noise(t).
            # So we unwrap phase and remove the linear trend (Doppler).
            
            peak_slow_time = r_fft[:, idx_r] # [Nc] complex
            phase_meas = np.unwrap(np.angle(peak_slow_time))
            
            # Fit linear trend (Doppler frequency)
            x_axis = np.arange(len(phase_meas))
            p = np.polyfit(x_axis, phase_meas, 1) # [slope, intercept]
            linear_trend = np.polyval(p, x_axis)
            
            # Residual is the Phase Error estimate (CPE)
            phase_error = phase_meas - linear_trend
            
            # Subtract this phase error from ALL range bins for each chirp
            correction_phasor = np.exp(-1j * phase_error)[:, None] # [Nc, 1]
            
            if beat_comp.ndim == 3:
                beat_comp *= correction_phasor[None, :, :]
            else:
                beat_comp *= correction_phasor

        return beat_comp

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
            # G2 Consistency: Do NOT add discrete clutter targets or coupling target to the simulation input.
            # Clutter is added as distributed noise in time/RDM domain (if enabled).
            # Adding them here created perfect point targets that counted as False Positives.

            if self.signal_type == 'OTFS':
                beat_rx, rdm = self.simulate_otfs_signal(sim_targets, snr_db)
            else:
                beat_rx = self.simulate_fmcw_signal(sim_targets, snr_db)
                # Unified processing call
                rdm = self.compute_rdm(beat_rx)

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
        if h5py is None:
            print("Warning: h5py not installed. Skipping dataset save.")
            return
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
        if h5py is None:
            raise RuntimeError("h5py not installed. Cannot load dataset.")
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
        
        if not TORCH_AVAILABLE:
            # Return numpy dict if torch is not available
            return {
                'time_domain': raw_np,
                'range_doppler_map': self.range_doppler_maps[idx],
                'target_mask': self.target_masks[idx],
                'target_info': self.target_info[idx],
                'cfar_detections': self.cfar_detections[idx] if hasattr(self, 'cfar_detections') and idx < len(self.cfar_detections) else [],
                'range_axis': self.range_axis,
                'velocity_axis': self.velocity_axis
            }

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

        # Enable FDD mode for simultaneous TX/RX (required for CN0566 FMCW)
        try:
            self.sdr.sdr._ctrl.debug_attrs[
                "adi,frequency-division-duplex-mode-enable"
            ].value = "1"
            self.sdr.sdr._ctrl.debug_attrs[
                "adi,ensm-enable-txnrx-control-enable"
            ].value = "0"
            self.sdr.sdr._ctrl.debug_attrs["initialize"].value = "1"
            import time as _time
            _time.sleep(3)  # AD9361 needs 2-3s to re-initialize
            print("  FDD mode enabled")
        except Exception as e:
            print(f"  FDD mode setup: {e}")

        self.sdr.SDR_RX_setup(
            n_SAMPLES=self.config.hardware_params['buffer_size'],
            controlmode='manual',
            rx1_gain=self.config.hardware_params['rx_gain'],
            rx2_gain=self.config.hardware_params['rx_gain']
        )

        # Set kernel buffers to 1 and enable quadrature tracking
        try:
            self.sdr.sdr._rxadc.set_kernel_buffers_count(1)
            rx_ch = self.sdr.sdr._ctrl.find_channel("voltage0")
            rx_ch.attrs["quadrature_tracking_en"].value = "1"
            print("  Kernel buffers: 1, quadrature tracking: enabled")
        except Exception as e:
            print(f"  SDR fine-tuning: {e}")

        self.sdr.SDR_TX_setup(
            cyclic_buffer=True,
            tx1_gain=self.config.hardware_params['tx_gain'],
            tx2_gain=self.config.hardware_params['tx_gain']
        )

        # TX1 (channel 0) is physically connected to the CN0566 phaser.
        # Turn off TX2 (channel 1) to avoid interference.
        self.sdr.sdr.tx_hardwaregain_chan0 = self.config.hardware_params['tx_gain']
        self.sdr.sdr.tx_hardwaregain_chan1 = -88
        print(f"  TX: ch0={self.config.hardware_params['tx_gain']}dB (TX1, active), ch1=-88dB (TX2, off)")

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
        try:
            self.phaser.load_gain_cal()
            self.phaser.load_phase_cal()
        except Exception:
            print("  Calibration files not found, using defaults")

        # Blackman antenna taper for better sidelobe control
        gain_list = [8, 34, 84, 127, 127, 84, 34, 8]
        for i in range(8):
            self.phaser.set_chan_phase(i, 0)
            self.phaser.set_chan_gain(i, gain_list[i], apply_cal=False)

        # Ensure GPIO routing for LO signal path
        try:
            self.phaser._gpios.gpio_vctrl_1 = 1  # Onboard PLL/LO source
            self.phaser._gpios.gpio_vctrl_2 = 1  # Send LO to TX circuitry
            self.phaser._gpios.gpio_tx_sw = 1     # Route to TX_OUT_1
            print("  GPIO: vctrl_1=1, vctrl_2=1, tx_sw=1")
        except Exception as e:
            print(f"  GPIO setup: {e}")

        output_freq = 12.1e9
        BW = self.config.params['B']
        num_steps = 1000
        ramp_time_us = self.config.params['T_chirp'] * 1e6

        self.phaser.frequency = int(output_freq / 4)
        self.phaser.freq_dev_range = int(BW / 4)
        self.phaser.freq_dev_step = int(BW / num_steps)  # ADI reference: BW/num_steps
        self.phaser.freq_dev_time = int(ramp_time_us)
        self.phaser.delay_word = 4095
        self.phaser.delay_clk = 'PFD'
        self.phaser.delay_start_en = 0
        self.phaser.ramp_delay_en = 0
        self.phaser.trig_delay_en = 0
        self.phaser.ramp_mode = self.config.hardware_params['ramp_mode']
        self.phaser.sing_ful_tri = 0
        self.phaser.tx_trig_en = 0
        self.phaser.enable = 0  # 0 = PLL ON, write last to update all registers

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
        """Send CW IF tone via Pluto TX for CN0566 FMCW radar.
        The PLL/VCO generates the chirp; Pluto TX provides the IF reference tone
        that gets upconverted to RF by the CN0566 mixer.
        """
        fs = int(self.config.params['fs'])
        N = self.config.hardware_params['buffer_size']
        signal_freq = 100e3  # 100 kHz IF tone (ADI reference)
        fc = int(signal_freq / (fs / N)) * (fs / N)  # Snap to FFT bin
        ts = 1 / float(fs)
        t = np.arange(0, N * ts, ts)
        i_sig = np.cos(2 * np.pi * t * fc) * 2 ** 14
        q_sig = np.sin(2 * np.pi * t * fc) * 2 ** 14
        iq = 1.0 * (i_sig + 1j * q_sig)
        self.sdr.sdr._ctx.set_timeout(0)
        self.sdr.sdr.tx([iq, iq * 0.5])  # TX1 (ch0) full, TX2 (ch1) half
        print(f"  TX tone: {fc/1e3:.1f} kHz IF reference sent via TX1")

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
        td = sample['time_domain']
        if hasattr(td, 'numpy'):
            td = td.numpy()  # Convert torch tensor to numpy
        # td shape: [Rx, Nc, Ns, 2] (real/imag)
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
        # delegated to AIRadarDataset.compute_rdm for consistency
        return self.processor.compute_rdm(raw_data)

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
            clutter_intensity=0.1
        )
        evaluate_dataset_metrics(dataset_cn, "Config CN0566 (Offline Sim)")
        
        # Also Evaluate DL if available in this test mode
        if hasattr(dataset_cn, 'dl_model') and dataset_cn.dl_model is not None:
             print("\n--- Evaluating Deep Learning Model (Sim) ---")
             dl_detections = []
             for i in range(len(dataset_cn)):
                 rdm = dataset_cn[i]['range_doppler_map']
                 if torch.is_tensor(rdm): rdm = rdm.numpy()
                 det = dataset_cn.run_dl_detection(rdm)
                 dl_detections.append(det)
             
             # Swap detections to reuse metrics function (hacky but effective)
             dataset_cn.cfar_detections = dl_detections
             evaluate_dataset_metrics(dataset_cn, "Deep Learning (Sim)")
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
    # Initialize processor
    processor = RadarProcessor(config)
    
    # In hardware mode, disable clutter effects in the processor to avoid adding synthetic clutter to real data
    if args.mode == 'hardware':
        processor.processor.apply_realistic_effects = False
        processor.processor.clutter_intensity = 0.0

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


# ======================================================================
# V2 Extensions: Realistic Channel + Robust DSP + Evaluation
# ======================================================================


class RealisticChannelSimulator:
    """
    Simulates the full CN0566 physical device chain:
      TX antenna → free-space channel → targets → return path → RX antenna
    Including TX-RX leakage, multipath reflections, antenna coupling,
    and phase noise propagation through the FMCW chirp.

    Can be toggled on/off via the `enabled` flag.
    """

    def __init__(self, config_params, enabled=True):
        self.enabled = enabled
        self.params = config_params

        # TX-RX leakage (direct coupling between TX and RX antennas)
        self.leakage_power_dB = config_params.get('leakage_power_dB', -25.0)
        self.leakage_delay_ns = config_params.get('leakage_delay_ns', 0.5)

        # Multipath: discrete reflectors near the radar
        self.multipath_enabled = config_params.get('multipath_enabled', True)
        self.num_multipath = config_params.get('num_multipath', 3)
        self.multipath_rcs_range_dB = config_params.get('multipath_rcs_range_dB', (-40, -20))
        self.multipath_range_m = config_params.get('multipath_range_m', (1.0, 30.0))

        # Antenna pattern (simplified cosine taper)
        self.antenna_pattern_enabled = config_params.get('antenna_pattern_enabled', True)
        self.antenna_3dB_beamwidth_deg = config_params.get('antenna_3dB_beamwidth_deg', 120.0)

        # Phase noise propagation (range-dependent decorrelation)
        self.phase_noise_propagation = config_params.get('phase_noise_propagation', True)
        self.oscillator_pn_dBc_Hz = config_params.get('oscillator_pn_dBc_Hz', -80.0)

        # ADF4159 PLL settling artifacts
        self.pll_settling_enabled = config_params.get('pll_settling_enabled', True)
        self.pll_settling_samples = config_params.get('pll_settling_samples', 20)
        self.pll_settling_amplitude = config_params.get('pll_settling_amplitude', 0.05)

        # Thermal noise floor
        self.thermal_noise_figure_dB = config_params.get('thermal_noise_figure_dB', 8.0)

    def apply(self, beat_rx, targets, fs, T_chirp, B, fc, slope):
        """
        Apply realistic channel effects to the beat signal.

        Parameters:
            beat_rx: [Rx, Nc, Ns] complex beat signal (already has target returns + AWGN)
            targets: list of target dicts
            fs, T_chirp, B, fc, slope: radar parameters

        Returns:
            beat_modified: [Rx, Nc, Ns] with channel effects applied
        """
        if not self.enabled:
            return beat_rx

        num_rx, Nc, Ns = beat_rx.shape
        beat = beat_rx.copy()
        t_fast = np.arange(Ns) / fs

        # 1. TX-RX leakage: strong near-zero-range return
        leakage_amp = np.sqrt(10 ** (self.leakage_power_dB / 10))
        leakage_delay = self.leakage_delay_ns * 1e-9
        fb_leak = 2 * (leakage_delay * c / 2) * slope / c  # beat freq from delay
        leakage_sig = leakage_amp * np.exp(2j * np.pi * fb_leak * t_fast)
        # Add random slow phase drift across chirps
        leak_phase = np.cumsum(np.random.randn(Nc) * 0.02)
        for rx in range(num_rx):
            beat[rx] += leakage_sig[None, :] * np.exp(1j * leak_phase)[:, None]

        # 2. Multipath reflections (stationary clutter at short range)
        if self.multipath_enabled:
            rng = np.random.default_rng(12345)
            for _ in range(self.num_multipath):
                mp_range = rng.uniform(*self.multipath_range_m)
                mp_rcs_dB = rng.uniform(*self.multipath_rcs_range_dB)
                mp_amp = np.sqrt(10 ** (mp_rcs_dB / 10))
                fb_mp = 2 * mp_range * slope / c
                # Near-zero Doppler with slight jitter
                fd_mp = rng.normal(0, 0.3) * 2 / (c / fc)
                mp_phase = 2 * np.pi * fb_mp * t_fast[None, :] + \
                           2 * np.pi * fd_mp * np.arange(Nc)[:, None] * T_chirp
                mp_sig = mp_amp * np.exp(1j * mp_phase)
                # Random initial phase
                mp_sig *= np.exp(1j * rng.uniform(0, 2 * np.pi))
                for rx in range(num_rx):
                    beat[rx] += mp_sig

        # 3. Antenna pattern roll-off (attenuate targets at wide angles)
        if self.antenna_pattern_enabled:
            bw_rad = np.radians(self.antenna_3dB_beamwidth_deg / 2)
            for t in targets:
                az = np.radians(t.get('azimuth', 0))
                # cos^2 pattern approximation; not applied to beat directly
                # (already included in RCS/amplitude). Instead, apply a
                # broadband attenuation pattern to the overall signal.
                pass  # Pattern already implicitly in target RCS modeling

        # 4. Phase noise propagation (range-dependent decorrelation)
        if self.phase_noise_propagation:
            # Phase noise power spectral density → time-domain jitter
            pn_lin = 10 ** (self.oscillator_pn_dBc_Hz / 10)
            pn_std = np.sqrt(pn_lin * fs)  # time-domain std
            # Per-chirp random phase walk (slow-time)
            pn_slow = np.cumsum(np.random.randn(Nc) * pn_std * 0.1)
            # Per-sample jitter (fast-time) - correlated
            pn_fast = np.cumsum(np.random.randn(Ns) * pn_std * 0.01)
            pn_2d = pn_slow[:, None] + pn_fast[None, :]
            for rx in range(num_rx):
                beat[rx] *= np.exp(1j * pn_2d)

        # 5. PLL settling transient at start of each chirp
        if self.pll_settling_enabled and self.pll_settling_samples > 0:
            ns = min(self.pll_settling_samples, Ns)
            # Exponential decay transient
            transient = self.pll_settling_amplitude * \
                        np.exp(-np.arange(ns) / max(ns / 3, 1))
            for rx in range(num_rx):
                for chirp in range(Nc):
                    phase_kick = np.random.randn() * np.pi * 0.5
                    beat[rx, chirp, :ns] += transient * np.exp(1j * phase_kick)

        return beat


# Default realistic channel parameters for CN0566
REALISTIC_CHANNEL_PARAMS = {
    'leakage_power_dB': -25.0,
    'leakage_delay_ns': 0.5,
    'multipath_enabled': True,
    'num_multipath': 3,
    'multipath_rcs_range_dB': (-40, -20),
    'multipath_range_m': (1.0, 30.0),
    'antenna_pattern_enabled': True,
    'antenna_3dB_beamwidth_deg': 120.0,
    'phase_noise_propagation': True,
    'oscillator_pn_dBc_Hz': -80.0,
    'pll_settling_enabled': True,
    'pll_settling_samples': 20,
    'pll_settling_amplitude': 0.05,
    'thermal_noise_figure_dB': 8.0,
}


class RobustRadarProcessor:
    """
    Enhanced radar processor with robust DSP techniques to improve
    detection performance under realistic channel conditions.

    Wraps AIRadarDataset but adds:
      1. MTI (Moving Target Indication) clutter cancellation
      2. Adaptive window selection (Chebyshev for sidelobe suppression)
      3. Zero-Doppler notch filter
      4. CFAR with post-detection integration
    """

    def __init__(self, config: RadarConfig, enable_mti=True,
                 enable_chebyshev_window=True, chebyshev_sll_dB=60,
                 enable_zero_doppler_notch=True, notch_width=3,
                 cfar_threshold_boost=5):
        self.config = config
        self.enable_mti = enable_mti
        self.enable_chebyshev_window = enable_chebyshev_window
        self.chebyshev_sll_dB = chebyshev_sll_dB
        self.enable_zero_doppler_notch = enable_zero_doppler_notch
        self.notch_width = notch_width

        # Create base processor
        ds_params = config.get_dataset_params()
        self.processor = AIRadarDataset(
            num_samples=1,
            autogen=False,
            save_path=None,
            drawfig=False,
            **ds_params
        )
        # Boost CFAR threshold for robust operation under channel impairments
        if cfar_threshold_boost > 0:
            self.processor.cfar_params['threshold_offset'] += cfar_threshold_boost

    def process_frame(self, raw_data):
        """
        Enhanced RDM computation with robust techniques.

        Parameters:
            raw_data: [Rx, Nc, Ns] or [Nc, Ns] complex

        Returns:
            rdm_db: [Nc, Nr] range-doppler map in dB
        """
        beat = raw_data.copy()

        # Collapse multi-RX
        if beat.ndim == 3:
            beat = np.sum(beat, axis=0)

        Nc, Ns = beat.shape

        # 1. Compensation (DC removal + IQ correction)
        if hasattr(self.processor, 'config') and \
                self.processor.config.get('enable_compensation', False):
            # Simple DC removal per chirp
            beat = beat - np.mean(beat, axis=1, keepdims=True)

        # 2. MTI filter: 2-pulse canceller removes static clutter
        if self.enable_mti and Nc > 1:
            # Apply 2-pulse canceller along slow-time
            beat_mti = np.diff(beat, n=1, axis=0)
            # Pad back to original size (prepend first row)
            beat = np.concatenate([beat_mti[:1], beat_mti], axis=0)

        # 3. Windowing
        if self.enable_chebyshev_window:
            from scipy.signal.windows import chebwin
            win_range = chebwin(Ns, at=self.chebyshev_sll_dB).astype(np.float64)
            win_doppler = chebwin(Nc, at=self.chebyshev_sll_dB).astype(np.float64)
        else:
            win_range = np.hanning(Ns)
            win_doppler = np.hanning(Nc)

        beat_win = beat * win_range[None, :] * win_doppler[:, None]

        # 4. Range FFT with zero padding
        zero_pad = self.processor.zero_pad
        range_fft = np.fft.fft(beat_win, n=zero_pad, axis=1)
        range_fft = range_fft[:, :self.processor.num_range_bins]

        # 5. Doppler FFT
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

        # 6. Magnitude in dB
        rdm_db = 20 * np.log10(np.abs(doppler_fft) + 1e-9)

        # 7. Zero-Doppler notch filter (suppress strong DC/leakage in center)
        if self.enable_zero_doppler_notch:
            center = Nc // 2
            nw = self.notch_width
            rdm_db[max(0, center - nw):center + nw + 1, :] = \
                np.min(rdm_db)

        return rdm_db

    def detect(self, rdm):
        """Delegate to CFAR detection."""
        return self.processor.cfar_detection(rdm)


def evaluate_radar_performance(engine, num_frames=50, snr_db=25.0,
                               save_path='output/radar_eval',
                               title='Radar Performance Evaluation',
                               use_realistic_channel=False,
                               use_robust_processing=False):
    """
    Comprehensive radar performance evaluation for simulation and loopback modes.

    Runs N frames, collects detection metrics, and generates:
      - Precision, Recall, F1 score
      - Range/velocity RMSE
      - False alarm rate
      - Per-frame detection statistics
      - Summary plots (confusion metrics, error CDFs, RDM examples)

    Parameters:
        engine: RadarEngine instance (must be started)
        num_frames: number of evaluation frames
        snr_db: SNR for simulation/loopback
        save_path: directory for output plots
        title: plot title string
        use_realistic_channel: if True, applies RealisticChannelSimulator
        use_robust_processing: if True, uses RobustRadarProcessor
    """
    os.makedirs(save_path, exist_ok=True)

    # Optionally create realistic channel simulator
    channel_sim = None
    if use_realistic_channel:
        channel_sim = RealisticChannelSimulator(REALISTIC_CHANNEL_PARAMS, enabled=True)

    # Optionally create robust processor
    robust_proc = None
    if use_robust_processing:
        robust_proc = RobustRadarProcessor(
            engine.config,
            enable_mti=True,
            enable_chebyshev_window=True,
            enable_zero_doppler_notch=True
        )

    # Collect metrics across frames
    all_tp, all_fp, all_fn = 0, 0, 0
    range_errors = []
    velocity_errors = []
    per_frame_detections = []
    per_frame_targets = []
    per_frame_tp = []
    per_frame_fp = []
    per_frame_fn = []
    per_frame_proc_time = []
    sample_rdms = []
    sample_data = []

    processor_ds = engine.processor.processor  # AIRadarDataset instance

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Frames: {num_frames}, SNR: {snr_db} dB")
    print(f"  Realistic channel: {use_realistic_channel}")
    print(f"  Robust processing: {use_robust_processing}")
    print(f"{'='*60}\n")

    for i in range(num_frames):
        t0 = timer()

        # Generate data
        if engine.mode == 'simulation':
            raw_data, target_info = engine.source.get_data()
            targets = target_info.get('targets', [])
        elif engine.mode == 'digital_loopback':
            raw_data, target_info, diag = engine.source.get_data(snr_db=snr_db)
            targets = target_info.get('targets', [])
        else:
            raw_data, target_info, diag = engine.source.get_data()
            targets = []

        # Apply realistic channel if enabled
        if channel_sim is not None and engine.mode in ('simulation', 'digital_loopback'):
            # For simulation mode raw_data comes from generate_dataset
            # Need the beat signal directly
            fs = engine.config.params['fs']
            T_chirp = engine.config.params['T_chirp']
            B = engine.config.params['B']
            fc = engine.config.params['fc']
            slope = B / T_chirp
            if raw_data.ndim == 2:
                raw_data = raw_data[None, :, :]  # add RX dim
            raw_data = channel_sim.apply(
                raw_data, targets, fs, T_chirp, B, fc, slope
            )

        # Process with either standard or robust processor
        if robust_proc is not None:
            rdm_db = robust_proc.process_frame(raw_data)
            detections = robust_proc.detect(rdm_db)
        else:
            rdm_db = engine.processor.process_frame(raw_data)
            detections = engine.processor.detect(rdm_db)

        proc_time = timer() - t0

        # Evaluate metrics
        if targets:
            metrics, matched, unmatched_t, unmatched_d = \
                processor_ds._evaluate_metrics(targets, detections)
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            all_tp += tp
            all_fp += fp
            all_fn += fn
            per_frame_tp.append(tp)
            per_frame_fp.append(fp)
            per_frame_fn.append(fn)

            if metrics['mean_range_error'] > 0:
                range_errors.append(metrics['mean_range_error'])
            if metrics['mean_velocity_error'] > 0:
                velocity_errors.append(metrics['mean_velocity_error'])
        else:
            per_frame_tp.append(0)
            per_frame_fp.append(len(detections))
            per_frame_fn.append(0)

        per_frame_detections.append(len(detections))
        per_frame_targets.append(len(targets))
        per_frame_proc_time.append(proc_time * 1000)

        # Save first few RDMs for visualization
        if i < 4:
            sample_rdms.append(rdm_db)
            sample_data.append({
                'targets': targets, 'detections': detections,
                'metrics': metrics if targets else None
            })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i+1}/{num_frames}: "
                  f"TP={per_frame_tp[-1]}, FP={per_frame_fp[-1]}, "
                  f"FN={per_frame_fn[-1]}, "
                  f"time={proc_time*1000:.1f}ms")

    # Compute aggregate metrics
    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    mean_range_err = np.mean(range_errors) if range_errors else 0.0
    std_range_err = np.std(range_errors) if range_errors else 0.0
    mean_vel_err = np.mean(velocity_errors) if velocity_errors else 0.0
    std_vel_err = np.std(velocity_errors) if velocity_errors else 0.0
    mean_proc_time = np.mean(per_frame_proc_time)
    fa_rate = all_fp / max(num_frames, 1)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS: {title}")
    print(f"{'='*60}")
    print(f"  Total frames:      {num_frames}")
    print(f"  Total targets:     {sum(per_frame_targets)}")
    print(f"  True Positives:    {all_tp}")
    print(f"  False Positives:   {all_fp}")
    print(f"  False Negatives:   {all_fn}")
    print(f"  Precision:         {precision:.4f}")
    print(f"  Recall:            {recall:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  FA rate/frame:     {fa_rate:.2f}")
    print(f"  Range RMSE:        {mean_range_err:.3f} +/- {std_range_err:.3f} m")
    print(f"  Velocity RMSE:     {mean_vel_err:.3f} +/- {std_vel_err:.3f} m/s")
    print(f"  Mean proc time:    {mean_proc_time:.1f} ms")
    print(f"{'='*60}\n")

    # Generate evaluation summary plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Per-frame TP/FP/FN bar chart
    ax = axes[0, 0]
    frames_x = np.arange(num_frames)
    ax.bar(frames_x, per_frame_tp, color='green', alpha=0.7, label='TP')
    ax.bar(frames_x, per_frame_fp, bottom=per_frame_tp,
           color='red', alpha=0.5, label='FP')
    ax.bar(frames_x, per_frame_fn,
           bottom=np.array(per_frame_tp) + np.array(per_frame_fp),
           color='orange', alpha=0.5, label='FN')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Count')
    ax.set_title('Detection Results per Frame')
    ax.legend(fontsize=8)

    # Panel 2: Cumulative precision/recall
    ax = axes[0, 1]
    cum_tp = np.cumsum(per_frame_tp)
    cum_fp = np.cumsum(per_frame_fp)
    cum_fn = np.cumsum(per_frame_fn)
    cum_prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    cum_rec = cum_tp / np.maximum(cum_tp + cum_fn, 1)
    cum_f1 = 2 * cum_prec * cum_rec / np.maximum(cum_prec + cum_rec, 1e-9)
    ax.plot(frames_x, cum_prec, 'b-', label=f'Precision ({precision:.3f})')
    ax.plot(frames_x, cum_rec, 'g-', label=f'Recall ({recall:.3f})')
    ax.plot(frames_x, cum_f1, 'r--', label=f'F1 ({f1:.3f})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Score')
    ax.set_title('Cumulative Detection Metrics')
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Panel 3: Range error histogram
    ax = axes[0, 2]
    if range_errors:
        ax.hist(range_errors, bins=30, color='steelblue', alpha=0.7,
                edgecolor='white')
        ax.axvline(mean_range_err, color='red', linestyle='--',
                   label=f'Mean={mean_range_err:.3f}m')
        ax.legend(fontsize=8)
    ax.set_xlabel('Range Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Range Error Distribution')

    # Panel 4: Velocity error histogram
    ax = axes[1, 0]
    if velocity_errors:
        ax.hist(velocity_errors, bins=30, color='darkorange', alpha=0.7,
                edgecolor='white')
        ax.axvline(mean_vel_err, color='red', linestyle='--',
                   label=f'Mean={mean_vel_err:.3f}m/s')
        ax.legend(fontsize=8)
    ax.set_xlabel('Velocity Error (m/s)')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Error Distribution')

    # Panel 5: Processing time
    ax = axes[1, 1]
    ax.plot(frames_x, per_frame_proc_time, 'k-', alpha=0.7)
    ax.axhline(mean_proc_time, color='red', linestyle='--',
               label=f'Mean={mean_proc_time:.1f}ms')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Processing Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"EVALUATION SUMMARY\n"
        f"{'─'*30}\n"
        f"Mode: {engine.mode}\n"
        f"Frames: {num_frames}\n"
        f"SNR: {snr_db} dB\n"
        f"Realistic Channel: {use_realistic_channel}\n"
        f"Robust Processing: {use_robust_processing}\n\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n\n"
        f"Range RMSE:    {mean_range_err:.3f} m\n"
        f"Velocity RMSE: {mean_vel_err:.3f} m/s\n"
        f"FA/frame:      {fa_rate:.2f}\n"
        f"Proc time:     {mean_proc_time:.1f} ms"
    )
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(save_path, 'eval_summary.png')
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"  Saved evaluation summary: {fig_path}")

    # Save sample RDM plots
    for idx, (rdm, sdata) in enumerate(zip(sample_rdms, sample_data)):
        fig_path = os.path.join(save_path, f'eval_rdm_sample_{idx}.png')
        if sdata['metrics']:
            metrics_s, matched_s, ut_s, ud_s = \
                processor_ds._evaluate_metrics(
                    sdata['targets'], sdata['detections'])
            _plot_2d_rdm(processor_ds, rdm, idx, metrics_s,
                         matched_s, ut_s, ud_s, fig_path)
        else:
            dummy = {'num_targets': 0, 'num_detections': len(sdata['detections']),
                     'tp': 0, 'fp': len(sdata['detections']), 'fn': 0,
                     'mean_range_error': 0.0, 'mean_velocity_error': 0.0}
            _plot_2d_rdm(processor_ds, rdm, idx, dummy,
                         [], [], sdata['detections'], fig_path)
        plt.close('all')

    print(f"  Saved {len(sample_rdms)} sample RDM plots to {save_path}/")

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_range_error': mean_range_err,
        'std_range_error': std_range_err,
        'mean_velocity_error': mean_vel_err,
        'std_velocity_error': std_vel_err,
        'fa_rate_per_frame': fa_rate,
        'total_tp': all_tp,
        'total_fp': all_fp,
        'total_fn': all_fn,
        'mean_proc_time_ms': mean_proc_time,
    }
    return results


# ======================================================================
# V2 Extensions: Digital Loopback + Unified RadarEngine
# ======================================================================


class RadarLoopback:
    """
    Digital loopback radar source using Pluto SDR's internal loopback.
    Generates FMCW beat signal via AIRadarDataset, transmits through digital
    loopback, receives, and returns the raw data for processing.
    """

    def __init__(self, config: RadarConfig, sdr_ip="ip:192.168.2.2"):
        self.config = config
        self.sdr_ip = sdr_ip
        self.sdr = None
        self.loopback_enabled = False
        self.connected = False

        self.fs = int(config.params['fs'])
        self.Nc = config.params['N_chirps']
        self.Ns = int(config.params['fs'] * config.params['T_chirp'])
        self.total_samples = self.Nc * self.Ns

        self.frame_count = 0
        self.tx_signal_info = {}

        self.dataset = AIRadarDataset(
            num_samples=1,
            autogen=False,
            save_path=None,
            drawfig=False,
            apply_realistic_effects=False,
            **config.get_dataset_params()
        )

    def connect(self):
        """Connect to Pluto SDR and enable digital loopback."""
        import adi

        print(f"[Loopback] Connecting to Pluto SDR at {self.sdr_ip}...")
        self.sdr = adi.Pluto(self.sdr_ip)

        self.sdr.sample_rate = self.fs
        self.sdr.tx_lo = int(915e6)
        self.sdr.rx_lo = int(915e6)
        self.sdr.tx_rf_bandwidth = self.fs
        self.sdr.rx_rf_bandwidth = self.fs
        self.sdr.tx_hardwaregain_chan0 = -10
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = 40
        self.sdr.rx_buffer_size = self.total_samples

        self.loopback_enabled = self._enable_loopback()
        if not self.loopback_enabled:
            raise RuntimeError("[Loopback] Failed to enable digital loopback")

        self._reset_sdr()
        self._flush_rx(num_flushes=5)

        self.connected = True
        print(f"[Loopback] Connected. Ns={self.Ns}, Nc={self.Nc}, "
              f"buffer={self.total_samples} samples")

    def _enable_loopback(self):
        """Enable digital loopback with verification."""
        try:
            self.sdr._ctrl.debug_attrs['loopback'].value = '1'
            time.sleep(0.1)
            actual = self.sdr._ctrl.debug_attrs['loopback'].value
            if actual != '1':
                self.sdr._ctrl.debug_attrs['loopback'].value = '1'
                time.sleep(0.1)
                actual = self.sdr._ctrl.debug_attrs['loopback'].value
            print(f"[Loopback] Digital loopback mode={actual}")
            return actual == '1'
        except Exception as e:
            print(f"[Loopback] Loopback enable failed: {e}")
            return False

    def _reset_sdr(self):
        """Reset AD9361 state machine by toggling LO frequencies."""
        try:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
            original_lo = self.sdr.rx_lo
            self.sdr.rx_lo = int(original_lo + 1e6)
            time.sleep(0.1)
            self.sdr.rx_lo = int(original_lo)
            time.sleep(0.1)
            original_tx_lo = self.sdr.tx_lo
            self.sdr.tx_lo = int(original_tx_lo + 1e6)
            time.sleep(0.05)
            self.sdr.tx_lo = int(original_tx_lo)
            time.sleep(0.05)
            print("[Loopback] SDR reset completed")
        except Exception as e:
            print(f"[Loopback] Reset warning: {e}")

    def _flush_rx(self, num_flushes=5):
        """Flush stale data from RX buffer."""
        flushed = 0
        for _ in range(num_flushes):
            try:
                rx_data = self.sdr.rx()
                if rx_data is not None and len(rx_data) > 0:
                    flushed += 1
            except Exception:
                pass
            time.sleep(0.01)
        print(f"[Loopback] Flushed RX buffer ({flushed}/{num_flushes})")

    def get_data(self, targets=None, snr_db=25):
        """
        Generate beat signal, TX via loopback, RX, return raw data.

        Parameters:
            targets: list of target dicts, or None for auto-generation
            snr_db: SNR for synthetic beat signal

        Returns:
            raw_data: complex ndarray [1, Nc, Ns]
            target_info: dict with 'targets' list
            diagnostics: dict with loopback metrics
        """
        if not self.connected:
            self.connect()

        # Generate targets if not provided
        if targets is None:
            targets = self.dataset.generate_targets(num_targets=2)

        # Generate FMCW beat signal via simulation
        beat_rx = self.dataset.simulate_fmcw_signal(targets, snr_db=snr_db)
        # beat_rx shape: [num_rx, Nc, Ns] = [1, 64, 1000]

        # Extract single-RX and flatten for TX
        beat_1d = beat_rx[0].flatten()  # [64000] complex

        # Normalize and scale for Pluto DAC (14-bit)
        max_val = np.max(np.abs(beat_1d))
        if max_val > 0:
            beat_normalized = beat_1d / max_val * 0.8
        else:
            beat_normalized = beat_1d
        tx_samples = (beat_normalized * 2**14).astype(np.complex64)

        self.tx_signal_info = {
            'max_amplitude': float(max_val),
            'num_samples': len(tx_samples),
            'num_targets': len(targets)
        }

        # Transmit via cyclic buffer
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(tx_samples)
        time.sleep(0.05)

        # Flush stale RX data that may contain previous frame's signal
        for _ in range(3):
            try:
                self.sdr.rx()
            except Exception:
                pass
            time.sleep(0.01)

        # Receive (retry up to 5 times)
        rx_raw = None
        for attempt in range(5):
            try:
                rx_temp = self.sdr.rx()
                if rx_temp is not None and len(rx_temp) > 0:
                    max_amp = np.max(np.abs(rx_temp))
                    if max_amp > 100:
                        rx_raw = rx_temp
                        break
            except Exception as e:
                print(f"[Loopback] RX attempt {attempt+1} error: {e}")
            time.sleep(0.05)

        # Stop TX
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass

        # Validate
        if rx_raw is None or len(rx_raw) == 0:
            raise RuntimeError("[Loopback] No valid RX data received")

        # Normalize RX back to float
        rx_float = rx_raw.astype(np.complex128) / (2**14)

        # Trim or pad to exact expected length
        if len(rx_float) >= self.total_samples:
            rx_trimmed = rx_float[:self.total_samples]
        else:
            rx_trimmed = np.pad(rx_float, (0, self.total_samples - len(rx_float)))

        # Reshape to [1, Nc, Ns]
        raw_data = rx_trimmed.reshape(1, self.Nc, self.Ns)

        # Compute diagnostics
        tx_power = float(np.mean(np.abs(beat_normalized)**2))
        rx_power = float(np.mean(np.abs(rx_trimmed)**2))

        # Correlation between TX and RX to assess loopback quality
        tx_ref = beat_normalized[:self.total_samples]
        rx_ref = rx_trimmed[:len(tx_ref)]
        if len(tx_ref) == len(rx_ref) and np.std(tx_ref) > 0 and np.std(rx_ref) > 0:
            corr = float(np.abs(np.corrcoef(tx_ref, rx_ref)[0, 1]))
        else:
            corr = 0.0

        diagnostics = {
            'frame_number': self.frame_count,
            'tx_power_dB': 10 * np.log10(tx_power + 1e-12),
            'rx_power_dB': 10 * np.log10(rx_power + 1e-12),
            'rx_max_amplitude': float(np.max(np.abs(rx_raw))),
            'tx_rx_correlation': corr,
            'buffer_fill_ratio': len(rx_raw) / self.total_samples
        }

        self.frame_count += 1
        target_info = {'targets': targets}

        return raw_data, target_info, diagnostics

    def stop(self):
        """Cleanup: disable loopback, release SDR."""
        if self.sdr is not None:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
            try:
                self.sdr._ctrl.debug_attrs['loopback'].value = '0'
                print("[Loopback] Digital loopback disabled")
            except Exception:
                pass
            try:
                del self.sdr
            except Exception:
                pass
            self.sdr = None
        self.connected = False
        self.loopback_enabled = False
        print("[Loopback] Stopped and cleaned up.")


class RadarHardwareV2:
    """
    Full CN0566 hardware mode (Pluto + Phaser).
    Wraps v1's RadarHardware with uniform get_data() interface.
    """

    def __init__(self, config: RadarConfig):
        self.config = config
        self.hw = RadarHardware(config)
        self.frame_count = 0

    def connect(self):
        """Start TX tone (already connected in __init__ via RadarHardware)."""
        self.hw.transmit()

    def get_data(self):
        """
        Receive one CPI from hardware.

        Returns:
            raw_data: [Rx, Nc, Ns] complex
            target_info: {'targets': []} (no ground truth in hw mode)
            diagnostics: dict
        """
        raw_data = self.hw.receive()

        rx_power = float(np.mean(np.abs(raw_data)**2))
        diagnostics = {
            'frame_number': self.frame_count,
            'rx_power_dB': 10 * np.log10(rx_power + 1e-12),
            'rx_max_amplitude': float(np.max(np.abs(raw_data)))
        }
        self.frame_count += 1

        return raw_data, {'targets': []}, diagnostics

    def steer_beam(self, angle_deg):
        """Delegate beam steering to phaser."""
        if self.hw.phaser is not None:
            fc = self.config.params.get('fc', 10.25e9)
            wavelength = c / fc
            d = 0.014  # element spacing (m)
            phase_delta = np.degrees(
                2 * np.pi * d * np.sin(np.radians(angle_deg)) / wavelength
            )
            self.hw.phaser.set_beam_phase_diff(phase_delta)

    def stop(self):
        self.hw.stop()


class RadarEngine:
    """
    Unified radar engine supporting three modes:
      - 'simulation': Pure software FMCW simulation
      - 'digital_loopback': Pluto SDR digital loopback
      - 'hardware': Full CN0566 phaser system
    """

    def __init__(self, config_name='config_cn0566', mode='simulation',
                 sdr_ip="ip:192.168.2.2", phaser_ip="ip:phaser.local",
                 use_realistic_channel=False, use_robust_processing=False,
                 use_dl_detection=False, dl_threshold=0.3):
        self.mode = mode
        self.config_name = config_name
        self.sdr_ip = sdr_ip
        self.phaser_ip = phaser_ip
        self.use_realistic_channel = use_realistic_channel
        self.use_robust_processing = use_robust_processing
        self.use_dl_detection = use_dl_detection
        self.dl_threshold = dl_threshold

        if mode == 'digital_loopback':
            self.config = RadarConfig(
                config_name='config_loopback', mode=mode,
                sdr_ip=sdr_ip, phaser_ip=""
            )
        else:
            self.config = RadarConfig(
                config_name=config_name, mode=mode,
                sdr_ip=sdr_ip, phaser_ip=phaser_ip
            )

        self.source = None

        # Choose processor: robust or standard
        if use_robust_processing:
            self.robust_processor = RobustRadarProcessor(
                self.config,
                enable_mti=True,
                enable_chebyshev_window=True,
                enable_zero_doppler_notch=True
            )
            self.processor = RadarProcessor(self.config)
        else:
            self.robust_processor = None
            self.processor = RadarProcessor(self.config)

        # Load DL model on the processor's dataset if DL detection is enabled
        if use_dl_detection:
            self.processor.processor._load_dl_model()

        # Realistic channel simulator
        if use_realistic_channel:
            self.channel_sim = RealisticChannelSimulator(
                REALISTIC_CHANNEL_PARAMS, enabled=True
            )
        else:
            self.channel_sim = None

        # Disable synthetic clutter for real-device modes
        if mode in ('digital_loopback', 'hardware'):
            self.processor.processor.apply_realistic_effects = False
            self.processor.processor.clutter_intensity = 0.0

        self.last_diagnostics = {}

    def start(self):
        """Initialize the data source for the selected mode."""
        if self.mode == 'simulation':
            self.source = RadarSimulation(self.config)
        elif self.mode == 'digital_loopback':
            self.source = RadarLoopback(self.config, sdr_ip=self.sdr_ip)
            self.source.connect()
        elif self.mode == 'hardware':
            self.source = RadarHardwareV2(self.config)
            self.source.connect()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        print(f"[RadarEngine] Started in {self.mode} mode")

    def get_frame(self, targets=None, snr_db=25):
        """
        Acquire one frame, process it, return everything needed for display.

        Returns:
            rdm_db: [Nc, Nr] range-doppler map in dB
            detections: list of detection dicts
            targets_gt: list of target dicts (ground truth; empty for hardware)
            diagnostics: dict with per-frame metrics
            r_axis: range axis array
            v_axis: velocity axis array
        """
        if self.source is None:
            self.start()

        # Acquire data
        if self.mode == 'simulation':
            raw_data, target_info = self.source.get_data()
            diagnostics = {'frame_number': 0, 'mode': 'simulation'}
            targets_gt = target_info.get('targets', [])
        elif self.mode == 'digital_loopback':
            raw_data, target_info, diagnostics = self.source.get_data(
                targets=targets, snr_db=snr_db
            )
            diagnostics['mode'] = 'digital_loopback'
            targets_gt = target_info.get('targets', [])
        elif self.mode == 'hardware':
            raw_data, target_info, diagnostics = self.source.get_data()
            diagnostics['mode'] = 'hardware'
            targets_gt = []
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply realistic channel effects if enabled
        if self.channel_sim is not None and self.mode in ('simulation', 'digital_loopback'):
            fs = self.config.params['fs']
            T_chirp = self.config.params['T_chirp']
            B = self.config.params['B']
            fc = self.config.params['fc']
            slope = B / T_chirp
            if raw_data.ndim == 2:
                raw_data = raw_data[None, :, :]
            raw_data = self.channel_sim.apply(
                raw_data, targets_gt, fs, T_chirp, B, fc, slope
            )

        # Process with robust or standard processor
        if self.robust_processor is not None:
            rdm_db = self.robust_processor.process_frame(raw_data)
        else:
            rdm_db = self.processor.process_frame(raw_data)

        # Detection: DL or CFAR
        if self.use_dl_detection:
            detections = self.processor.processor.run_dl_detection(
                rdm_db, threshold=self.dl_threshold
            )
            diagnostics['detection_method'] = 'dl'
        elif self.robust_processor is not None:
            detections = self.robust_processor.detect(rdm_db)
            diagnostics['detection_method'] = 'robust_cfar'
        else:
            detections = self.processor.detect(rdm_db)
            diagnostics['detection_method'] = 'cfar'

        diagnostics['num_detections'] = len(detections)
        diagnostics['num_targets'] = len(targets_gt)
        self.last_diagnostics = diagnostics

        r_axis = self.processor.processor.range_axis
        v_axis = self.processor.processor.velocity_axis

        return rdm_db, detections, targets_gt, diagnostics, r_axis, v_axis

    def update_processor_params(self, **kwargs):
        """Update CFAR/DSP parameters at runtime."""
        ds = self.processor.processor
        if 'cfar_threshold' in kwargs:
            ds.cfar_params['threshold_offset'] = kwargs['cfar_threshold']
        if 'cfar_type' in kwargs:
            if hasattr(ds, 'config') and isinstance(ds.config, dict):
                ds.config['cfar_type'] = kwargs['cfar_type']
        if 'enable_compensation' in kwargs:
            if hasattr(ds, 'config') and isinstance(ds.config, dict):
                ds.config['enable_compensation'] = kwargs['enable_compensation']
        if 'enable_phase_tracking' in kwargs:
            if hasattr(ds, 'config') and isinstance(ds.config, dict):
                ds.config['enable_phase_noise_tracking'] = kwargs['enable_phase_tracking']
        if 'min_range' in kwargs:
            ds.cfar_params['min_range_m'] = kwargs['min_range']

    def steer_beam(self, angle_deg):
        """Only valid in hardware mode."""
        if self.mode == 'hardware' and hasattr(self.source, 'steer_beam'):
            self.source.steer_beam(angle_deg)

    def stop(self):
        """Stop and clean up."""
        if self.source is not None:
            if hasattr(self.source, 'stop'):
                self.source.stop()
            self.source = None
        print("[RadarEngine] Stopped")


# ======================================================================
# V2 Main
# ======================================================================

def main_v2():
    parser = argparse.ArgumentParser(
        description="MyRadar V2 - Simulation / Digital Loopback / Hardware"
    )
    parser.add_argument('--mode', type=str, default='simulation',
                        choices=['simulation', 'digital_loopback', 'hardware'],
                        help='Operation mode')
    parser.add_argument('--config', type=str, default='config_cn0566',
                        help='Radar configuration name')
    parser.add_argument('--save_path', type=str, default='output/radar_v2',
                        help='Path to save output')
    parser.add_argument('--sdr_ip', type=str, default="ip:192.168.2.2",
                        help='SDR IP address')
    parser.add_argument('--phaser_ip', type=str, default="ip:phaser.local",
                        help='Phaser IP address')
    parser.add_argument('--frames', type=int, default=5,
                        help='Number of frames to capture/simulate')
    parser.add_argument('--snr', type=float, default=25.0,
                        help='SNR for simulation/loopback (dB)')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Enable visualization')
    parser.add_argument('--eval_sim', action='store_true', default=False,
                        help='Run offline simulation evaluation (100 samples)')
    parser.add_argument('--realistic_channel', action='store_true', default=False,
                        help='Enable realistic channel simulation (leakage, multipath, phase noise)')
    parser.add_argument('--robust', action='store_true', default=False,
                        help='Enable robust DSP (MTI, Chebyshev window, zero-Doppler notch)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Run comprehensive evaluation (N frames with metrics)')
    parser.add_argument('--eval_frames', type=int, default=50,
                        help='Number of frames for evaluation (default 50)')
    parser.add_argument('--dl', action='store_true', default=False,
                        help='Use DL-based detection instead of CFAR')
    parser.add_argument('--dl_threshold', type=float, default=0.3,
                        help='DL detection threshold (default 0.3)')
    args = parser.parse_args()

    # Optional offline evaluation (same as v1)
    if args.eval_sim:
        print(f"Running offline simulation evaluation with {args.config} config...")
        dataset_cn = AIRadarDataset(
            num_samples=100,
            config_name=args.config,
            save_path=args.save_path,
            drawfig=False,
            apply_realistic_effects=True,
            clutter_intensity=0.1
        )
        evaluate_dataset_metrics(dataset_cn, f"Config {args.config} (Offline Sim)")
        return

    # Create engine
    engine = RadarEngine(
        config_name=args.config,
        mode=args.mode,
        sdr_ip=args.sdr_ip,
        phaser_ip=args.phaser_ip,
        use_realistic_channel=args.realistic_channel,
        use_robust_processing=args.robust,
        use_dl_detection=args.dl,
        dl_threshold=args.dl_threshold
    )

    print(f"\nMyRadar V2")
    print(f"  Mode: {args.mode}")
    print(f"  Config: {args.config}")
    print(f"  Detection: {'DL (threshold={})'.format(args.dl_threshold) if args.dl else 'CFAR'}")
    if args.mode == 'digital_loopback':
        print(f"  SDR IP: {args.sdr_ip}")
        print(f"  Buffer: Nc={engine.config.params['N_chirps']}, "
              f"Ns={int(engine.config.params['fs'] * engine.config.params['T_chirp'])}, "
              f"Total={engine.config.params['N_chirps'] * int(engine.config.params['fs'] * engine.config.params['T_chirp'])}")
    print(f"  Realistic channel: {args.realistic_channel}")
    print(f"  Robust processing: {args.robust}")
    print(f"  Frames: {args.frames}")
    print(f"  SNR: {args.snr} dB")
    print("-" * 60)

    engine.start()
    os.makedirs(args.save_path, exist_ok=True)

    # Run evaluation mode if requested
    if args.eval:
        eval_title = f"Radar Eval ({args.mode})"
        if args.realistic_channel:
            eval_title += " + RealisticChannel"
        if args.robust:
            eval_title += " + RobustDSP"

        eval_save = os.path.join(args.save_path, 'evaluation')
        results = evaluate_radar_performance(
            engine,
            num_frames=args.eval_frames,
            snr_db=args.snr,
            save_path=eval_save,
            title=eval_title,
            use_realistic_channel=args.realistic_channel,
            use_robust_processing=args.robust
        )
        engine.stop()
        print(f"\nEvaluation complete. Results saved to {eval_save}/")
        return

    try:
        for i in range(args.frames):
            print(f"\n--- Frame {i+1}/{args.frames} ---")
            start_time = timer()

            rdm, detections, targets, diag, r_axis, v_axis = engine.get_frame(
                snr_db=args.snr
            )

            proc_time = timer() - start_time
            print(f"  Mode: {diag.get('mode', '?')}")
            print(f"  Processing time: {proc_time*1000:.2f} ms")
            print(f"  Detections: {len(detections)}")
            print(f"  Targets (GT): {len(targets)}")
            if 'rx_power_dB' in diag:
                print(f"  RX power: {diag['rx_power_dB']:.1f} dB")
            if 'tx_rx_correlation' in diag:
                print(f"  TX-RX correlation: {diag['tx_rx_correlation']:.4f}")
            if 'buffer_fill_ratio' in diag:
                print(f"  Buffer fill: {diag['buffer_fill_ratio']:.2f}")

            # Print detection details
            for j, det in enumerate(detections[:5]):
                print(f"    Det {j}: R={det['range_m']:.1f}m, "
                      f"V={det['velocity_mps']:.1f}m/s, "
                      f"P={det.get('power', 0):.1f}dB")

            if args.plot:
                save_path = os.path.join(args.save_path, f"radar_v2_frame_{i}.png")

                if targets:
                    metrics, matched, unmatched_t, unmatched_d = \
                        engine.processor.processor._evaluate_metrics(targets, detections)
                    _plot_2d_rdm(engine.processor.processor, rdm, i, metrics,
                                 matched, unmatched_t, unmatched_d, save_path)
                    print(f"  Metrics: TP={metrics['tp']}, FP={metrics['fp']}, "
                          f"FN={metrics['fn']}, "
                          f"Range_err={metrics['mean_range_error']:.2f}m, "
                          f"Vel_err={metrics['mean_velocity_error']:.2f}m/s")
                else:
                    dummy_metrics = {
                        'num_targets': 0,
                        'num_detections': len(detections),
                        'tp': 0, 'fp': len(detections), 'fn': 0,
                        'mean_range_error': 0.0,
                        'mean_velocity_error': 0.0
                    }
                    _plot_2d_rdm(engine.processor.processor, rdm, i, dummy_metrics,
                                 [], [], detections, save_path)
                print(f"  Saved: {save_path}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        engine.stop()

    print(f"\nDone. Output saved to {args.save_path}/")


if __name__ == "__main__":
    main_v2()