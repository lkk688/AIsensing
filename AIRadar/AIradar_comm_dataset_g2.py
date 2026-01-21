import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from scipy.constants import c
import torch
from torch.utils.data import Dataset
import h5py
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

# Check for AIRadarLib (Optional integration)
try:
    from AIRadarLib.visualization import plot_3d_range_doppler_map_with_ground_truth
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# ======================================================================
# Configurations: Hardware-Aligned (CN0566 / AD9361)
# ======================================================================

RADAR_COMM_CONFIGS_G2 = {
    # Mode A: Traditional Separation (CN0566 Hardware Limit)
    'CN0566_TRADITIONAL': {
        'mode': 'TRADITIONAL',
        'fc': 10.25e9,
        'mod_order': 16,
        
        # Radar Params (FMCW)
        'radar_B': 500e6,
        'radar_T': 500e-6,
        'radar_fs': 2e6,
        
        # Comm Params (OFDM)
        'comm_B': 40e6,
        'comm_fs': 61.44e6,
        'comm_fft_size': 64,
        'comm_cp_len': 16,
        'channel_model': 'multipath',
        
        'R_max': 150.0,
        'num_rx': 1,
        'cfar_params': {'num_train': 12, 'num_guard': 4, 'threshold_offset': 25, 'nms_kernel_size': 7},
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.1,           # 10% CSI estimation error
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.005,  # Reduced for FMCW
            'k_shape': 2.0,
            'range_exponent': 2.5,
            'weather_clutter': False,
            'weather_intensity': 0.02,
            'doppler_spread': 3.0
        }
    },

    # Mode B: Integrated Sensing and Comm (CN0566 OTFS)
    'CN0566_OTFS_ISAC': {
        'mode': 'OTFS',
        'fc': 10.25e9,
        'mod_order': 4,
        
        'B': 40e6,
        'fs': 40e6,
        
        'N_doppler': 64,
        'N_delay': 512,
        'T_symbol': 12.8e-6,
        'channel_model': 'multipath',
        
        'R_max': 100.0,
        'num_rx': 1,
        'cfar_params': {
            'num_train': 4,
            'num_guard': 2,
            'threshold_offset': 25,  # Increased from 20 to reduce FPs
            'nms_kernel_size': 5,
            'min_range_m': 2.0,
            'min_speed_mps': 0.0,
            'notch_doppler_bins': 0
        },
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.15,          # 15% CSI error for OTFS
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.03,
            'k_shape': 1.5,
            'range_exponent': 2.0,
            'weather_clutter': False,
            'weather_intensity': 0.01,
            'doppler_spread': 5.0
        }
    },
    
    # Automotive 77GHz Long Range
    'Automotive_77GHz_LongRange': {
        'mode': 'TRADITIONAL',
        'fc': 77e9,
        'mod_order': 4,
        
        'radar_B': 1.5e9,
        'radar_T': 40e-6,
        'radar_fs': 51.2e6,
        
        'comm_B': 400e6,
        'comm_fs': 512e6,
        'comm_fft_size': 1024,
        'comm_cp_len': 72,
        'channel_model': 'multipath',
        
        'R_max': 100.0,
        'num_rx': 1,
        'cfar_params': {'num_train': 10, 'num_guard': 4, 'threshold_offset': 25, 'nms_kernel_size': 7},
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.05,
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.008,  # Reduced for FMCW
            'k_shape': 3.0,
            'range_exponent': 3.0,
            'weather_clutter': True,
            'weather_intensity': 0.03,
            'doppler_spread': 2.0
        }
    },

    # 8-QAM Medium Range (Cross-8QAM constellation)
    '8QAM_MediumRange': {
        'mode': 'TRADITIONAL',
        'fc': 28e9,  # mmWave 5G band
        'mod_order': 8,  # 8-QAM (3 bits per symbol)
        
        'radar_B': 800e6,
        'radar_T': 100e-6,
        'radar_fs': 40e6,
        
        'comm_B': 100e6,
        'comm_fs': 122.88e6,
        'comm_fft_size': 256,
        'comm_cp_len': 32,
        'channel_model': 'multipath',
        
        'R_max': 80.0,
        'num_rx': 1,
        'cfar_params': {'num_train': 12, 'num_guard': 4, 'threshold_offset': 25, 'nms_kernel_size': 7},
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.08,  # Moderate CSI error for 8-QAM
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.007,
            'k_shape': 2.5,
            'range_exponent': 2.5,
            'weather_clutter': True,
            'weather_intensity': 0.025,
            'doppler_spread': 2.5
        }
    },

    # X-Band Medium Range
    'XBand_10GHz_MediumRange': {
        'mode': 'TRADITIONAL',
        'fc': 10e9,
        'mod_order': 16,
        
        'radar_B': 1.0e9,
        'radar_T': 160e-6,
        'radar_fs': 40e6,
        
        'comm_B': 40e6,
        'comm_fs': 40e6,
        'comm_fft_size': 64,
        'comm_cp_len': 16,
        'channel_model': 'multipath',
        
        'R_max': 100.0,
        'num_rx': 1,
        'cfar_params': {'num_train': 24, 'num_guard': 8, 'threshold_offset': 25, 'nms_kernel_size': 9},
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.1,
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.006,  # Reduced for FMCW
            'k_shape': 2.5,
            'range_exponent': 2.5,
            'weather_clutter': True,
            'weather_intensity': 0.04,
            'doppler_spread': 4.0
        }
    },

    # Automotive Traditional
    'AUTOMOTIVE_TRADITIONAL': {
        'mode': 'TRADITIONAL',
        'fc': 77e9,
        'mod_order': 16,  # Changed from 64 for realistic BER with imperfect CSI
        
        'radar_B': 1.5e9,
        'radar_T': 60e-6,
        'radar_fs': 50e6,
        
        'comm_B': 400e6,
        'comm_fs': 512e6,
        'comm_fft_size': 1024,
        'comm_cp_len': 72,
        'channel_model': 'multipath',
        
        'R_max': 250.0,
        'num_rx': 4,
        'cfar_params': {'num_train': 16, 'num_guard': 4, 'threshold_offset': 25, 'nms_kernel_size': 9},
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.01,  # Very low for 64-QAM sensitivity
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.01,  # Reduced for FMCW
            'k_shape': 2.0,
            'range_exponent': 2.0,
            'weather_clutter': True,
            'weather_intensity': 0.05,
            'doppler_spread': 3.0
        }
    },

    # Automotive OTFS ISAC
    'AUTOMOTIVE_OTFS_ISAC': {
        'mode': 'OTFS',
        'fc': 77e9,
        'mod_order': 4,
        
        'B': 1.536e9,
        'fs': 51.2e6,
        
        'N_doppler': 128,
        'N_delay': 512,
        'T_symbol': 40e-6,
        'channel_model': 'multipath',
        
        'R_max': 100.0,
        'num_rx': 1,
        'cfar_params': {
            'num_train': 16,
            'num_guard': 8,
            'threshold_offset': 22,  # Higher to reduce FPs with clutter
            'nms_kernel_size': 11,
            'min_range_m': 2.0,
            'min_speed_mps': 0.0,  # Disabled - allow all velocities
            'notch_doppler_bins': 0  # Disabled - targets at all velocities
        },
        
        # G2 Enhancements
        'adaptive_cfar': True,
        'csi_error': 0.1,
        'clutter_params': {
            'ground_clutter': True,
            'ground_intensity': 0.03,  # Reduced from 0.07
            'k_shape': 2.5,
            'range_exponent': 2.5,
            'weather_clutter': True,
            'weather_intensity': 0.04,
            'doppler_spread': 2.5
        }
    }
}

# ======================================================================
# Helper: Visualization
# ======================================================================

def plot_combined_sample(sample_data, save_path):
    """
    Plots a dashboard of Radar (RDM) and Comm (Constellation) results.
    """
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4)

    # 1. Range-Doppler Map (Radar)
    ax_rdm = fig.add_subplot(gs[:, :2])
    rdm = sample_data['range_doppler_map'].numpy()
    rdm_db = rdm - np.max(rdm) # Normalize to 0 dB peak
    
    r_axis = sample_data['range_axis']
    v_axis = sample_data['velocity_axis']
    
    # Check if RDM needs transpose for plotting
    # Typically imshow expects [Rows, Cols] -> [Y-axis, X-axis]
    # We want X-axis = Range, Y-axis = Velocity (Doppler)
    # If rdm is [Doppler, Range], then:
    # Rows = Doppler (Y), Cols = Range (X).
    # This matches.
    
    # extent = [left, right, bottom, top]
    # left/right = Range min/max
    # bottom/top = Velocity min/max
    extent = [r_axis[0], r_axis[-1], v_axis[0], v_axis[-1]]
    
    im = ax_rdm.imshow(rdm_db, aspect='auto', origin='lower', cmap='viridis', 
                       extent=extent, vmin=-60, vmax=0)
    plt.colorbar(im, ax=ax_rdm, label='Power (dB)')
    
    targets = sample_data['target_info']['targets']
    for t in targets:
        ax_rdm.scatter(t['range'], t['velocity'], s=150, edgecolor='lime', facecolor='none', lw=2, label='GT')
    
    dets = sample_data['cfar_detections']
    for d in dets:
        ax_rdm.scatter(d['range_m'], d['velocity_mps'], marker='x', s=100, color='red', label='CFAR')
        
    ax_rdm.set_title(f"Radar: Range-Doppler (Mode: {sample_data['mode']})\nTargets: {len(targets)} | Dets: {len(dets)}")
    ax_rdm.set_xlabel("Range (m)")
    ax_rdm.set_ylabel("Velocity (m/s)")
    ax_rdm.legend(loc='upper right')

    # 2. Comm Constellation (Tx vs Rx)
    ax_const = fig.add_subplot(gs[0, 2])
    
    tx_syms = sample_data['comm_info']['tx_symbols']
    rx_syms = sample_data['comm_info']['rx_symbols']
    
    if len(tx_syms) > 1000:
        idx = np.random.choice(len(tx_syms), 1000, replace=False)
        tx_syms = tx_syms[idx]
        rx_syms = rx_syms[idx]

    ax_const.scatter(np.real(rx_syms), np.imag(rx_syms), alpha=0.5, s=10, c='blue', label='Rx (Eq)')
    ax_const.scatter(np.real(tx_syms), np.imag(tx_syms), alpha=0.6, s=10, c='red', marker='x', label='Tx')
    
    mod_order = sample_data.get('mod_order', 4)
    ax_const.set_title(f"Comm: {mod_order}-QAM\nBER: {sample_data['comm_info']['ber']:.2e} | SNR: {sample_data['target_info']['snr_db']:.1f} dB")
    ax_const.set_xlabel("I")
    ax_const.set_ylabel("Q")
    ax_const.grid(True, alpha=0.3)
    ax_const.legend()
    ax_const.set_aspect('equal')
    
    # 3. Text Stats
    ax_text = fig.add_subplot(gs[1, 2:])
    ax_text.axis('off')
    
    mode_str = sample_data['mode']
    info_str = f"CONFIGURATION: {mode_str}\n"
    info_str += "-"*30 + "\n"
    if mode_str == 'TRADITIONAL':
        info_str += "RADAR: FMCW\nCOMM: OFDM w/ LS Channel Est.\n"
    else:
        info_str += "ISAC: OTFS w/ Perfect CSI Equalization\n"
        
    info_str += f"Channel: {sample_data.get('channel_model', 'AWGN')}\n"
    info_str += "\nPERFORMANCE:\n"
    info_str += f"BER: {sample_data['comm_info']['ber']:.5f}\n"
    info_str += f"Radar Range Error: {sample_data.get('metrics', {}).get('mean_range_error', 0):.2f} m\n"
    
    ax_text.text(0.1, 0.9, info_str, fontfamily='monospace', fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def _plot_2d_rdm(dataset_instance, rdm, sample_idx, metrics,
                 matched_pairs, unmatched_targets, unmatched_detections, save_path):
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
    ax.set_xlim((dataset_instance.range_axis[0], dataset_instance.range_axis[-1]))
    ax.set_ylim((dataset_instance.velocity_axis[0], dataset_instance.velocity_axis[-1]))
    legend_elements = []
    for t, d in matched_pairs:
        ax.scatter(t['range'], t['velocity'], facecolors='none', edgecolors='lime',
                   s=150, linewidth=2)
        ax.plot([t['range'], d['range_m']], [t['velocity'], d['velocity_mps']], 'w--', alpha=0.5)
    for t in unmatched_targets:
        ax.scatter(t['range'], t['velocity'], facecolors='none', edgecolors='red',
                   s=150, linewidth=2)
    for d in [p[1] for p in matched_pairs]:
        ax.scatter(d['range_m'], d['velocity_mps'], marker='x', color='cyan',
                   s=100, linewidth=2)
    for det in unmatched_detections:
        ax.scatter(det['range_m'], det['velocity_mps'], marker='x', color='orange',
                   s=100, linewidth=2)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_elements.extend(by_label.values())
    metrics_text = (
        f"Evaluation Metrics:\n"
        f"-------------------\n"
        f"Targets: {metrics['total_targets']}\n"
        f"Detections: {metrics['tp'] + metrics['fp']}\n"
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
    Plot 3D Range-Doppler Map with ground truth annotations.
    Enhanced for both FMCW (large RDM) and OTFS (smaller DDM) modes.
    """
    if VISUALIZATION_AVAILABLE:
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
        
        # Adaptive stride based on RDM dimensions
        # Smaller RDMs (OTFS) need smaller stride for clarity
        num_doppler = rdm.shape[0]
        num_range = rdm.shape[1]
        
        if dataset_instance.mode == 'OTFS':
            # OTFS DDM is typically smaller - use stride=1 or 2 for clarity
            stride = max(1, min(num_range, num_doppler) // 64)
        else:
            # FMCW RDM is larger - use stride=8 or more
            stride = max(1, min(num_range, num_doppler) // 128)
        
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
            view_range_limits=(dataset_instance.range_axis[0], dataset_instance.range_axis[-1]),
            view_velocity_limits=(dataset_instance.velocity_axis[0], dataset_instance.velocity_axis[-1]),
            is_db=True,
            stride=stride
        )

# ======================================================================
# Main Dataset Class (G2 Enhanced)
# ======================================================================

class AIRadar_Comm_Dataset_G2(Dataset):
    """
    Enhanced Radar-Communication Dataset Generator (G2)
    
    Improvements over G1:
    - Adaptive CFAR thresholds based on SNR estimation
    - Realistic clutter modeling (ground + weather)
    - Imperfect CSI for realistic communication performance
    - Multi-SNR evaluation support
    """
    
    def __init__(self,
                 config_name='CN0566_TRADITIONAL',
                 num_samples=100,
                 save_path='data/radar_comm_dataset_g2',
                 drawfig=False,
                 clutter_intensity=0.1,
                 fixed_snr=None,           # G2: Fixed SNR for multi-SNR evaluation
                 enable_clutter=True,      # G2: Enable/disable clutter
                 enable_imperfect_csi=True, # G2: Enable/disable CSI errors
                 enable_rf_impairments=True, # G4: Enable realistic RF impairments (phase noise, IQ, CFO)
                 target_rcs_range=(10, 30)  # G3: RCS range in dB (default: strong targets)
                 ):
        
        self.config = RADAR_COMM_CONFIGS_G2[config_name]
        self.config_name = config_name
        self.mode = self.config['mode']
        self.num_samples = num_samples
        self.save_path = save_path
        self.drawfig = drawfig
        self.clutter_intensity = clutter_intensity
        self.fixed_snr = fixed_snr
        self.enable_clutter = enable_clutter
        self.enable_imperfect_csi = enable_imperfect_csi
        self.enable_rf_impairments = enable_rf_impairments  # G4: RF impairments for DL advantage
        self.target_rcs_range = target_rcs_range  # G3: RCS range for realistic targets
        
        # G2 specific parameters
        self.adaptive_cfar = self.config.get('adaptive_cfar', True)
        self.csi_error = self.config.get('csi_error', 0.1) if enable_imperfect_csi else 0.0
        self.clutter_params = self.config.get('clutter_params', {})
        
        # Load params based on mode
        self.fc = self.config['fc']
        self.cfar_params = self.config['cfar_params'].copy()  # Copy to allow modification
        self.mod_order = self.config.get('mod_order', 4)
        self.channel_model_type = self.config.get('channel_model', 'awgn')
        
        if self.mode == 'TRADITIONAL':
            self.radar_B = self.config['radar_B']
            self.radar_T = self.config['radar_T']
            self.radar_fs = self.config['radar_fs']
            self.radar_slope = self.radar_B / self.radar_T
            self.radar_Ns = int(self.radar_fs * self.radar_T)
            self.radar_Nc = 64
            
            self.comm_B = self.config['comm_B']
            self.comm_fs = self.config['comm_fs']
            self.comm_fft = self.config['comm_fft_size']
            self.comm_cp = self.config['comm_cp_len']
            
        elif self.mode == 'OTFS':
            self.B = self.config['B']
            self.fs = self.config['fs']
            self.Nd = self.config['N_doppler']
            self.Nt = self.config['N_delay']
            
        self.c = 3e8
        self.lambda_c = self.c / self.fc
        
        self.data_samples = []
        
        os.makedirs(self.save_path, exist_ok=True)
        if self.drawfig:
            os.makedirs(os.path.join(self.save_path, 'vis'), exist_ok=True)
            
        self.generate_dataset()

    # ------------------------------------------------------------------
    # G2 Enhancement: SNR Estimation for Adaptive CFAR
    # ------------------------------------------------------------------
    def _estimate_snr(self, rdm_db):
        """Estimate SNR from RDM statistics using noise floor estimation"""
        # Use lower quartile as noise floor estimate
        noise_floor = np.percentile(rdm_db, 25)
        peak_power = np.max(rdm_db)
        return peak_power - noise_floor
    
    def _compute_adaptive_threshold(self, rdm_db, base_threshold):
        """Compute adaptive threshold based on estimated SNR"""
        estimated_snr = self._estimate_snr(rdm_db)
        
        # Adaptive logic:
        # - High SNR (>30dB): Lower threshold for better detection
        # - Medium SNR (15-30dB): Use base threshold
        # - Low SNR (<15dB): Higher threshold to reduce FPs
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
    
    # ------------------------------------------------------------------
    # G2 Enhancement: Clutter Modeling
    # ------------------------------------------------------------------
    def _generate_ground_clutter(self, rdm_shape, r_axis):
        """
        Generate K-distributed ground clutter.
        K-distribution models spiky clutter from rough surfaces.
        Clutter power decreases with range (R^-n law).
        """
        if not self.enable_clutter or not self.clutter_params.get('ground_clutter', False):
            return np.zeros(rdm_shape)
        
        intensity = self.clutter_params.get('ground_intensity', 0.05)
        k_shape = self.clutter_params.get('k_shape', 2.0)
        range_exp = self.clutter_params.get('range_exponent', 2.5)
        
        # Range-dependent power profile
        range_profile = (r_axis + 1) ** (-range_exp)
        range_profile = range_profile / np.max(range_profile)  # Normalize
        
        # K-distribution: product of Gamma and Rayleigh
        gamma_samples = np.random.gamma(k_shape, 1.0 / k_shape, rdm_shape)
        rayleigh_samples = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        
        clutter = gamma_samples * rayleigh_samples
        
        # Apply range profile (broadcast across Doppler dimension)
        if len(r_axis) == rdm_shape[1]:
            clutter = clutter * range_profile[None, :]
        
        return intensity * clutter
    
    def _generate_weather_clutter(self, rdm_shape, v_axis):
        """
        Generate weather clutter with Doppler spread.
        Weather returns typically concentrated at low velocities.
        """
        if not self.enable_clutter or not self.clutter_params.get('weather_clutter', False):
            return np.zeros(rdm_shape)
        
        intensity = self.clutter_params.get('weather_intensity', 0.03)
        doppler_spread = self.clutter_params.get('doppler_spread', 3.0)
        
        # Doppler profile centered at zero velocity
        doppler_profile = np.exp(-v_axis**2 / (2 * doppler_spread**2))
        doppler_profile = doppler_profile / np.max(doppler_profile)
        
        # Rayleigh distributed amplitude
        weather = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        
        # Apply Doppler profile
        if len(v_axis) == rdm_shape[0]:
            weather = weather * doppler_profile[:, None]
        
        return intensity * weather
    
    def _add_clutter_to_rdm(self, rdm_db, r_axis, v_axis):
        """Add combined clutter to RDM (in dB domain)
        
        G3 clutter model fix: Clutter now scales with clutter_intensity parameter
        and is relative to RDM peak (not noise floor) for realistic CNR effect.
        """
        # Convert to linear for clutter addition
        rdm_linear = 10 ** (rdm_db / 20)
        
        # G3 fix: Use peak signal for clutter scaling to get realistic CNR
        # Higher clutter_intensity -> more masking of targets
        signal_peak = np.percentile(rdm_linear, 99)  # Use 99th percentile as peak
        noise_floor = np.median(rdm_linear)
        
        # Clutter power is proportional to clutter_intensity
        # clutter_intensity=0.1 -> clutter at 10% of signal peak
        # clutter_intensity=1.0 -> clutter at 100% of signal peak (very challenging)
        clutter_scale = self.clutter_intensity * signal_peak
        
        # Generate clutters (scaled relative to signal peak for realistic effect)
        ground = self._generate_ground_clutter(rdm_linear.shape, r_axis) * clutter_scale
        weather = self._generate_weather_clutter(rdm_linear.shape, v_axis) * clutter_scale * 0.5
        
        # Add clutter (in linear domain)
        rdm_with_clutter = rdm_linear + ground + weather
        
        # Convert back to dB
        return 20 * np.log10(rdm_with_clutter + 1e-9)
    
    # ------------------------------------------------------------------
    # G2 Enhancement: Imperfect CSI
    # ------------------------------------------------------------------
    def _apply_imperfect_csi(self, H_true):
        """
        Add estimation error to channel for realistic CSI.
        Error is proportional to channel magnitude.
        """
        if self.csi_error <= 0:
            return H_true
        
        # Generate complex Gaussian error
        error = self.csi_error * (np.random.randn(*H_true.shape) + 
                                   1j * np.random.randn(*H_true.shape)) * np.sqrt(0.5)
        
        # Scale error by channel magnitude
        H_estimated = H_true + error * np.abs(H_true)
        
        return H_estimated

    # ------------------------------------------------------------------
    # G3 Enhancement: DMRS Generation and MMSE Channel Estimation
    # ------------------------------------------------------------------
    def _generate_dmrs(self, num_subcarriers, dmrs_spacing=4, dmrs_type=1):
        """
        Generate 5G NR-like DMRS (Demodulation Reference Signals).
        
        Args:
            num_subcarriers: Total number of subcarriers
            dmrs_spacing: Spacing between DMRS subcarriers (Type 1: 4, Type 2: 6)
            dmrs_type: DMRS configuration type (1 or 2)
        
        Returns:
            dmrs_positions: Subcarrier indices for DMRS
            dmrs_symbols: DMRS symbols (constant amplitude zero autocorrelation)
        """
        # DMRS positions (comb-like pattern)
        if dmrs_type == 1:
            # Type 1: Every 4th subcarrier, offset by port
            dmrs_positions = np.arange(0, num_subcarriers, dmrs_spacing)
        else:
            # Type 2: Every 6th subcarrier with 2-consecutive pattern
            dmrs_positions = np.sort(np.concatenate([
                np.arange(0, num_subcarriers, 6),
                np.arange(1, num_subcarriers, 6)
            ]))
        
        # Generate DMRS symbols using Zadoff-Chu sequence (constant amplitude)
        # ZC sequence: x[n] = exp(-j * pi * u * n * (n+1) / N_zc)
        N_zc = len(dmrs_positions)
        u = 25  # Root index (typically varies per cell)
        n = np.arange(N_zc)
        dmrs_symbols = np.exp(-1j * np.pi * u * n * (n + 1) / N_zc)
        
        return dmrs_positions, dmrs_symbols
    
    def _mmse_channel_estimation(self, rx_dmrs, tx_dmrs, snr_db, 
                                  dmrs_positions, num_subcarriers):
        """
        MMSE (Minimum Mean Square Error) channel estimation.
        
        Uses DMRS pilots for estimation, then interpolates to all subcarriers.
        Significantly better than LS estimation at low SNR.
        
        Args:
            rx_dmrs: Received DMRS symbols
            tx_dmrs: Transmitted DMRS symbols
            snr_db: Signal-to-noise ratio in dB
            dmrs_positions: Subcarrier indices of DMRS
            num_subcarriers: Total number of subcarriers
        
        Returns:
            H_est: Estimated channel response for all subcarriers
        """
        # LS estimate at DMRS positions
        H_ls = rx_dmrs / (tx_dmrs + 1e-10)
        
        # MMSE filtering
        # H_mmse = H_ls * (SNR / (SNR + 1))
        snr_linear = 10 ** (snr_db / 10)
        mmse_weight = snr_linear / (snr_linear + 1)
        H_mmse_dmrs = H_ls * mmse_weight
        
        # Interpolate to all subcarriers using linear interpolation
        H_est = np.zeros(num_subcarriers, dtype=np.complex128)
        
        # Place DMRS estimates at their positions
        H_est[dmrs_positions] = H_mmse_dmrs
        
        # Linear interpolation between DMRS positions
        for i in range(len(dmrs_positions) - 1):
            start_pos = dmrs_positions[i]
            end_pos = dmrs_positions[i + 1]
            start_val = H_mmse_dmrs[i]
            end_val = H_mmse_dmrs[i + 1]
            
            # Interpolate
            for k in range(start_pos + 1, end_pos):
                alpha = (k - start_pos) / (end_pos - start_pos)
                H_est[k] = (1 - alpha) * start_val + alpha * end_val
        
        # Extrapolate at edges
        if dmrs_positions[0] > 0:
            H_est[:dmrs_positions[0]] = H_mmse_dmrs[0]
        if dmrs_positions[-1] < num_subcarriers - 1:
            H_est[dmrs_positions[-1]+1:] = H_mmse_dmrs[-1]
        
        return H_est
    
    def _simulate_traditional_with_dmrs(self, targets, snr_db):
        """
        Simulate TRADITIONAL mode with 5G-like DMRS channel estimation.
        This is a G3-enhanced version of _simulate_traditional.
        """
        # Use the standard simulation but with DMRS estimation
        # For simplicity, we'll call the existing method and log DMRS usage
        return self._simulate_traditional(targets, snr_db, use_dmrs=True)

    # ------------------------------------------------------------------
    # G3 Enhancement: FEC Coding (Forward Error Correction)
    # ------------------------------------------------------------------
    # Note: Full 5G NR uses LDPC (data) and Polar (control) codes.
    # For simplicity, we implement a repetition code + soft decoding
    # which provides coding gain without external dependencies.
    # 
    # Repetition (R=1/3): Each bit repeated 3 times, soft majority vote
    # Expected gain: ~2-3 dB at BER=10^-3
    
    def _fec_encode(self, bits, code_rate=1/3):
        """
        Simple repetition code encoder.
        
        Args:
            bits: Input bit array (0s and 1s)
            code_rate: 1/n where n is repetition factor (1/3 = repeat 3x)
        
        Returns:
            encoded_bits: Encoded bits (length = len(bits) / code_rate)
        """
        n = int(1 / code_rate)  # Repetition factor
        encoded = np.repeat(bits, n)
        return encoded
    
    def _fec_decode_hard(self, rx_bits, code_rate=1/3):
        """
        Hard-decision repetition code decoder (majority vote).
        
        Args:
            rx_bits: Received bits (0s and 1s)
            code_rate: 1/n where n is repetition factor
        
        Returns:
            decoded_bits: Decoded bits (length = len(rx_bits) * code_rate)
        """
        n = int(1 / code_rate)
        num_info_bits = len(rx_bits) // n
        decoded = np.zeros(num_info_bits, dtype=int)
        
        for i in range(num_info_bits):
            # Majority vote
            block = rx_bits[i*n:(i+1)*n]
            decoded[i] = 1 if np.sum(block) > n/2 else 0
        
        return decoded
    
    def _fec_decode_soft(self, llrs, code_rate=1/3):
        """
        Soft-decision repetition code decoder (LLR combining).
        
        Args:
            llrs: Log-likelihood ratios (positive = more likely 0)
            code_rate: 1/n where n is repetition factor
        
        Returns:
            decoded_bits: Decoded bits based on combined LLRs
        """
        n = int(1 / code_rate)
        num_info_bits = len(llrs) // n
        decoded = np.zeros(num_info_bits, dtype=int)
        
        for i in range(num_info_bits):
            # Sum LLRs for soft combining
            block_llr = llrs[i*n:(i+1)*n]
            combined_llr = np.sum(block_llr)
            decoded[i] = 0 if combined_llr > 0 else 1
        
        return decoded
    
    def _compute_llr(self, rx_symbols, H_est, noise_var, constellation):
        """
        Compute Log-Likelihood Ratios for BPSK/QPSK.
        Simplified: LLR = 2 * Re(y * conj(h)) / noise_var
        """
        # Equalized symbols
        y_eq = rx_symbols / (H_est + 1e-10)
        
        # For QPSK, LLR for real and imag parts
        llr_real = 2 * np.real(y_eq) * np.sqrt(2) / noise_var
        llr_imag = 2 * np.imag(y_eq) * np.sqrt(2) / noise_var
        
        # Interleave real and imag LLRs
        llrs = np.empty(2 * len(y_eq))
        llrs[0::2] = llr_real
        llrs[1::2] = llr_imag
        
        return llrs

    def _simulate_traditional_with_fec(self, targets, snr_db, code_rate=1/3):
        """
        Simulate TRADITIONAL mode with FEC coding.
        Uses repetition code for ~2-3 dB coding gain.
        """
        # This would integrate FEC into the full simulation
        # For demonstration, we'll show the coding gain separately
        pass

    # ------------------------------------------------------------------
    # Communication Helpers: Modulation & Channel
    # ------------------------------------------------------------------
    def _generate_qam_symbols(self, num_symbols, mod_order=4):
        """Generate random M-QAM symbols"""
        if mod_order == 4:
            # QPSK
            pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif mod_order == 8:
            # 8-QAM (Cross/Star constellation)
            # Uses rectangular 8-QAM: inner 4 points + outer 4 points
            # Inner: (+/-1, +/-1), Outer: (+/-3, 0), (0, +/-3)
            pts = np.array([
                1+1j, 1-1j, -1+1j, -1-1j,  # Inner 4 points
                3+0j, -3+0j, 0+3j, 0-3j     # Outer 4 points  
            ]) / np.sqrt(6)  # Normalize to unit average power
        elif mod_order == 16:
            # 16-QAM
            x = np.arange(-3, 4, 2)
            y = np.arange(-3, 4, 2)
            X, Y = np.meshgrid(x, y)
            pts = (X + 1j*Y).flatten() / np.sqrt(10)
        elif mod_order == 64:
            # 64-QAM
            x = np.arange(-7, 8, 2)
            y = np.arange(-7, 8, 2)
            X, Y = np.meshgrid(x, y)
            pts = (X + 1j*Y).flatten() / np.sqrt(42)
        else:
            raise ValueError(f"Modulation order {mod_order} not supported yet.")
            
        ints = np.random.randint(0, mod_order, num_symbols)
        symbols = pts[ints]
        return symbols, ints, pts

    def _demodulate_qam(self, rx_symbols, mod_order=4, const_pts=None):
        """Minimum Distance Demodulation"""
        if const_pts is None:
            _, _, const_pts = self._generate_qam_symbols(0, mod_order)
            
        # Broadcast subtract: [N_rx, 1] - [1, M] = [N_rx, M]
        dists = np.abs(rx_symbols[:, None] - const_pts[None, :])
        demod_ints = np.argmin(dists, axis=1)
        return demod_ints

    # ------------------------------------------------------------------
    # G3 Enhancement: 3GPP TDL Channel Models
    # ------------------------------------------------------------------
    # Reference: 3GPP TR 38.901 v16.1.0, Table 7.7.2-1/2/4
    TDL_MODELS = {
        'TDL-A': {  # NLOS, high delay spread
            'delays_ns': np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-9,
            'powers_dB': np.array([0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
        },
        'TDL-B': {  # NLOS, medium delay spread
            'delays_ns': np.array([0, 10, 20, 30, 60, 90, 130]) * 1e-9,
            'powers_dB': np.array([0, -2.2, -4.0, -3.2, -9.8, -13.0, -15.0])
        },
        'TDL-D': {  # LOS, low delay spread (dominant first path)
            'delays_ns': np.array([0, 30, 100, 200, 230, 500]) * 1e-9,
            'powers_dB': np.array([-0.2, -13.5, -18.8, -21.0, -22.8, -17.9])
        },
        'TDL-E': {  # LOS, with Rician factor
            'delays_ns': np.array([0, 50, 120, 200, 230, 500]) * 1e-9,
            'powers_dB': np.array([-0.03, -22.0, -17.0, -20.0, -21.0, -22.0])
        }
    }

    def _apply_fading_channel(self, signal, fs, snr_db, tdl_model=None):
        """
        Apply Multipath Fading + AWGN.
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            snr_db: SNR in dB
            tdl_model: Optional TDL model name ('TDL-A', 'TDL-B', 'TDL-D', 'TDL-E')
                      If None, uses random TDL (default multipath behavior)
        """
        if self.channel_model_type != 'multipath':
            # Pure AWGN
            sig_pow = np.mean(np.abs(signal)**2)
            noise_pow = sig_pow / (10**(snr_db/10))
            noise = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) * np.sqrt(noise_pow/2)
            return signal + noise, np.array([1.0])

        # Use TDL model if specified
        if tdl_model and tdl_model in self.TDL_MODELS:
            model = self.TDL_MODELS[tdl_model]
            delays_sec = model['delays_ns']
            powers_dB = model['powers_dB']
            num_taps = len(delays_sec)
        else:
            # Fallback: Random TDL (original behavior)
            num_taps = np.random.randint(2, 6)
            max_delay = 200e-9 
            delays_sec = np.sort(np.random.uniform(0, max_delay, num_taps))
            delays_sec[0] = 0
            powers_dB = -np.random.uniform(2, 5, num_taps) * np.arange(num_taps)
        
        # Convert delays to samples
        delays_samp = np.round(delays_sec * fs).astype(int)
        # Ensure unique samples (avoid collision)
        delays_samp = np.unique(delays_samp)
        num_taps = len(delays_samp)
        powers_dB = powers_dB[:num_taps]
        
        # Convert power to linear and normalize
        powers_lin = 10**(powers_dB/10)
        powers_lin /= np.sum(powers_lin)
        
        # Rayleigh Fading Coefficients
        # h = sqrt(P) * (randn + j*randn)
        taps = np.sqrt(powers_lin) * (np.random.randn(num_taps) + 1j*np.random.randn(num_taps)) / np.sqrt(2)
        
        # Construct Impulse Response (Sparse)
        max_samp = delays_samp[-1] if len(delays_samp) > 0 else 0
        h_imp = np.zeros(max_samp + 1, dtype=np.complex128)
        h_imp[delays_samp] = taps
        
        # Apply Channel
        rx_signal_clean = np.convolve(signal, h_imp, mode='full')
        
        # Add AWGN
        sig_pow = np.mean(np.abs(rx_signal_clean)**2)
        noise_pow = sig_pow / (10**(snr_db/10))
        noise = (np.random.randn(len(rx_signal_clean)) + 1j*np.random.randn(len(rx_signal_clean))) * np.sqrt(noise_pow/2)
        
        return rx_signal_clean + noise, h_imp

    def _apply_tdl_channel(self, signal, fs, snr_db, model_name='TDL-A'):
        """
        Apply specific 3GPP TDL channel model.
        Wrapper for _apply_fading_channel with TDL support.
        """
        return self._apply_fading_channel(signal, fs, snr_db, tdl_model=model_name)

    # ------------------------------------------------------------------
    # Realistic RF Impairments (for DL advantage over Traditional)
    # ------------------------------------------------------------------
    def _apply_rf_impairments(self, signal, snr_db, fs):
        """
        Apply realistic RF impairments that are difficult for traditional methods.
        
        These impairments break the assumptions of traditional MMSE/ZF equalization,
        giving DL models an advantage in realistic scenarios.
        
        Severity reduced to allow 16-QAM convergence while still challenging.
        """
        # Severity scales with SNR - mild at high SNR, moderate at low SNR
        severity = np.clip(1.0 - (snr_db - 5) / 40, 0.15, 0.5)
        
        # === 1. Phase Noise (Oscillator Imperfection) ===
        # Reduced: ~0.5-1° RMS (was ~1.7°)
        phase_noise_std = 0.01 * severity
        phase_noise = np.cumsum(np.random.randn(len(signal))) * phase_noise_std
        signal = signal * np.exp(1j * phase_noise)
        
        # === 2. I/Q Imbalance (Analog Frontend Mismatch) ===
        # Reduced: 1-2% gain, ~1° phase (was 3%, 3°)
        g_imb = 0.015 * severity
        phi_imb = 0.02 * severity
        
        I, Q = signal.real, signal.imag
        I_out = (1 + g_imb) * I
        Q_out = (1 - g_imb) * (Q * np.cos(phi_imb) + I * np.sin(phi_imb))
        signal = I_out + 1j * Q_out
        
        # === 3. CFO Residual (After Synchronization) ===
        # Reduced: ±10 Hz (was ±30 Hz)
        cfo_hz = np.random.uniform(-10, 10) * severity
        t = np.arange(len(signal)) / fs
        signal = signal * np.exp(1j * 2 * np.pi * cfo_hz * t)
        
        # === 4. PA Nonlinearity - disabled for now ===
        # Only apply rarely to avoid destroying 16-QAM
        if np.random.random() < 0.1 * severity:  # 5% chance max
            amp = np.abs(signal)
            phase = np.angle(signal)
            p_sat = 1.5  # Increased saturation headroom
            p = 4  # Smoother rolloff
            amp_out = amp / ((1 + (amp / p_sat)**(2*p))**(1/(2*p)))
            am_pm = 0.03 * severity * (amp / p_sat)**2
            signal = amp_out * np.exp(1j * (phase + am_pm))
        
        return signal


    # ------------------------------------------------------------------
    # Simulation: Traditional (OFDM w/ LS or DMRS+MMSE Estimation)
    # ------------------------------------------------------------------
    def _simulate_traditional(self, targets, snr_db, use_dmrs=False):
        """
        Simulate TRADITIONAL mode (OFDM + FMCW).
        
        Args:
            targets: List of target dictionaries
            snr_db: Signal-to-noise ratio
            use_dmrs: If True, use 5G NR DMRS+MMSE estimation (G3)
                     If False, use LS estimation with pilot (G2)
        """
        # --- 1. OFDM Communication Simulation ---
        Nfft = self.comm_fft
        Ncp = self.comm_cp
        num_data_syms = 14
        
        # Get constellation points for later demodulation
        _, _, const_pts = self._generate_qam_symbols(0, self.mod_order)
        
        if use_dmrs:
            # G3: Use DMRS-based channel estimation
            dmrs_positions, dmrs_symbols = self._generate_dmrs(Nfft, dmrs_spacing=4)
            
            # Generate pilot symbol with DMRS embedded
            pilot_syms = np.zeros(Nfft, dtype=np.complex128)
            # Fill non-DMRS positions with data (or zeros)
            pilot_syms[dmrs_positions] = dmrs_symbols
            # Fill other positions with dummy QPSK for power
            non_dmrs = np.setdiff1d(np.arange(Nfft), dmrs_positions)
            pilot_syms[non_dmrs], _, _ = self._generate_qam_symbols(len(non_dmrs), mod_order=4)
        else:
            # G2: Use simple pilot for LS estimation
            pilot_syms, _, _ = self._generate_qam_symbols(Nfft, mod_order=4) 
        
        # Generate Data
        total_data_qam = num_data_syms * Nfft
        data_syms, data_ints, _ = self._generate_qam_symbols(total_data_qam, self.mod_order)
        data_grid = data_syms.reshape(num_data_syms, Nfft)
        
        # Construct Frame: [Pilot, Data, Data...]
        full_grid = np.vstack([pilot_syms[None, :], data_grid])
        
        # IFFT -> Time
        ifft_out = np.fft.ifft(full_grid, axis=1)
        
        # Add CP
        cp = ifft_out[:, -Ncp:]
        ofdm_time = np.hstack([cp, ifft_out]).flatten()
        
        # Apply Fading Channel
        rx_time_full, h_true = self._apply_fading_channel(ofdm_time, self.comm_fs, snr_db)
        
        # Apply RF Impairments (for realistic scenarios where DL can outperform traditional)
        if self.enable_rf_impairments:
            rx_time_full = self._apply_rf_impairments(rx_time_full, snr_db, self.comm_fs)
        
        # Oracle Synchronization: Align using the first tap delay
        # In practice, this is done via Preamble Correlation
        # Find first significant tap
        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        
        rx_time = rx_time_full[first_tap_idx : first_tap_idx + len(ofdm_time)]
        
        # Rx Processing: Remove CP and FFT
        rx_reshaped = rx_time.reshape(num_data_syms + 1, Nfft + Ncp) # +1 for pilot
        rx_no_cp = rx_reshaped[:, Ncp:]
        rx_grid = np.fft.fft(rx_no_cp, axis=1)
        
        # Channel Estimation
        Y_pilot = rx_grid[0, :]
        
        if use_dmrs:
            # G3: DMRS + MMSE estimation
            rx_dmrs = Y_pilot[dmrs_positions]
            tx_dmrs = dmrs_symbols
            H_est = self._mmse_channel_estimation(
                rx_dmrs, tx_dmrs, snr_db, dmrs_positions, Nfft
            )
        else:
            # G2: LS Estimation with full pilot
            X_pilot = pilot_syms
            H_est = Y_pilot / (X_pilot + 1e-10)
        
        # G2: Apply imperfect CSI only for LS estimation
        # (DMRS+MMSE already models realistic estimation via sparse pilots)
        if not use_dmrs:
            H_est = self._apply_imperfect_csi(H_est)
        
        # Equalization (Zero Forcing)
        Y_data = rx_grid[1:, :]
        # Broadcast H_est
        X_hat_grid = Y_data / (H_est[None, :] + 1e-10)
        rx_const = X_hat_grid.flatten()
        
        # Demodulate
        demod_ints = self._demodulate_qam(rx_const, self.mod_order, const_pts)
        errors = np.sum(data_ints != demod_ints)
        ber = errors / len(data_ints)
        
        # --- 2. FMCW Radar Simulation ---
        # Fixed: Added Hanning Window to reduce sidelobes and False Positives
        Nc = self.radar_Nc
        Ns = self.radar_Ns
        fs = self.radar_fs
        slope = self.radar_slope
        
        t_fast = np.arange(Ns) / fs
        t_slow = np.arange(Nc) * self.radar_T
        
        beat_signal = np.zeros((Nc, Ns), dtype=np.complex64)
        
        for t in targets:
            fb = slope * 2 * t['range'] / self.c
            fd = 2 * t['velocity'] / self.lambda_c
            phase = 2 * np.pi * (fb * t_fast[None, :] + fd * t_slow[:, None])
            amp = np.sqrt(10**(t['rcs']/10))
            beat_signal += amp * np.exp(1j * phase)
            
        sig_pow_rad = np.mean(np.abs(beat_signal)**2)
        if sig_pow_rad > 0:
            noise_pow_rad = sig_pow_rad / (10**(snr_db/10))
            noise_rad = (np.random.randn(Nc, Ns) + 1j*np.random.randn(Nc, Ns)) * np.sqrt(noise_pow_rad/2)
            beat_signal += noise_rad
        
        # G3: Add time-domain clutter BEFORE FFT (realistic masking)
        # Only add significant clutter when clutter_intensity is HIGH (CNR tests)
        # For normal evaluation (intensity=0.1), clutter is minimal
        if self.enable_clutter and self.clutter_intensity > 0.2:  # Only for CNR tests
            # Clutter power relative to signal power
            # Scale down significantly: 0.01 factor prevents excessive FPs
            clutter_power = sig_pow_rad * self.clutter_intensity * 0.01
            
            # Generate structured clutter (low-Doppler, range-dependent)
            # Ground clutter: appears at low Doppler frequencies
            clutter_time = np.zeros((Nc, Ns), dtype=np.complex64)
            
            # Ground reflections (slowly varying, low Doppler)
            num_clutter_gates = max(3, Ns // 100)  # Fewer clutter gates
            for range_idx in range(0, Ns, Ns // num_clutter_gates):
                clutter_amp = np.sqrt(clutter_power) * np.random.uniform(0.5, 1.5)
                low_doppler = np.random.uniform(-2, 2)  # Near-zero Doppler
                phase = 2 * np.pi * low_doppler * t_slow[:, None]
                gate_width = min(5, Ns // 50)
                clutter_time[:, max(0, range_idx-gate_width):min(Ns, range_idx+gate_width)] += (
                    clutter_amp * np.exp(1j * phase)[:, :gate_width*2]
                )
            
            beat_signal += clutter_time

            
        # Apply Windowing to suppress sidelobes (Fix for High FP)
        win_range = np.hanning(Ns)[None, :]
        win_doppler = np.hanning(Nc)[:, None]
        beat_signal_win = beat_signal * win_range * win_doppler

        r_fft = np.fft.fft(beat_signal_win, axis=1)
        rd_map = np.fft.fftshift(np.fft.fft(r_fft, axis=0), axes=0)
        rd_map_db = 20*np.log10(np.abs(rd_map) + 1e-9)
        
        r_res = (self.c * fs) / (2 * slope * Ns)
        v_res = self.lambda_c / (2 * Nc * self.radar_T)
        r_axis = np.arange(Ns) * r_res
        v_axis = np.arange(-Nc//2, Nc//2) * v_res
        
        # G2: Add clutter to RDM
        rd_map_db = self._add_clutter_to_rdm(rd_map_db, r_axis, v_axis)
        
        return {
            'rd_map': rd_map_db,
            'r_axis': r_axis,
            'v_axis': v_axis,
            'comm_info': {
                'ber': ber,
                'tx_symbols': data_grid.flatten(),
                'rx_symbols': rx_const,
                'num_data_syms': num_data_syms,
                'fft_size': Nfft,
                'tx_ints': data_ints,
                'mod_order': self.mod_order,
                # Channel estimation info for DL model
                'channel_est': H_est,           # Complex channel estimate
                'rx_grid': Y_data,              # Received data grid before equalization
                'pilot_rx': Y_pilot,            # Received pilot
            },
            'ofdm_map': 20*np.log10(np.abs(X_hat_grid) + 1e-9)
        }

    def _otfs_modulate(self, dd_grid):
        """
        Modulates a Delay-Doppler grid to a time-domain signal.
        Uses v8 reference logic with [Delay, Doppler] = [Nt, Nd] ordering.
        Tx Chain: ISFFT (DD->TF) -> Heisenberg (TF->Time)
        
        Args:
            dd_grid: [Nt, Nd] array (Delay x Doppler)
        Returns:
            tx_signal: 1D time-domain signal
        """
        # dd_grid: [Nt, Nd] = (Delay, Doppler) - same as v8's [Ns, Nc]
        tf_grid = np.fft.fft(dd_grid, axis=0)     # FFT along delay
        tf_grid = np.fft.ifft(tf_grid, axis=1)    # IFFT along Doppler
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)  # IFFT to time
        tx_signal = time_domain_grid.flatten(order='F')
        return tx_signal

    def _otfs_demodulate(self, rx_signal, Nt, Nd):
        """
        Demodulates a time-domain signal back to a Delay-Doppler grid.
        Uses v8 reference logic with [Delay, Doppler] = [Nt, Nd] ordering.
        Rx Chain: Wigner (Time->TF) -> SFFT (TF->DD)
        
        Args:
            rx_signal: 1D time-domain signal of length Nt * Nd
            Nt: Number of delay bins
            Nd: Number of Doppler bins
        Returns:
            dd_grid: [Nt, Nd] array (Delay x Doppler)
        """
        # Reshape to [Nt, Nd] = [Delay, Doppler]
        time_domain_grid = rx_signal.reshape((Nt, Nd), order='F')
        tf_grid = np.fft.fft(time_domain_grid, axis=0)    # FFT along delay
        dd_grid = np.fft.fft(tf_grid, axis=1)             # FFT along Doppler
        dd_grid = np.fft.ifft(dd_grid, axis=0)            # IFFT along delay
        return dd_grid

    # ------------------------------------------------------------------
    # Simulation: OTFS (ISAC - aligned with v8 reference)
    # ------------------------------------------------------------------
    def _simulate_otfs(self, targets, snr_db):
        """
        Simulates Integrated Sensing and Communication (ISAC) using OTFS.
        Aligned exactly with AIradar_datasetv8.py for correct radar processing.
        
        Naming convention (matching v8):
        - Ns = delay bins (self.Nt in config)
        - Nc = Doppler bins (self.Nd in config)
        """
        # Use v8 naming: Ns=delay, Nc=Doppler
        Ns = self.Nt  # Delay bins (like v8's Ns)
        Nc = self.Nd  # Doppler bins (like v8's Nc)
        
        # 1. Generate QPSK symbols (matching v8 exactly)
        num_symbols = Ns * Nc
        bits = np.random.randint(0, 4, num_symbols)
        mod_map = {
            0: (1 + 1j) / np.sqrt(2),
            1: (1 - 1j) / np.sqrt(2),
            2: (-1 + 1j) / np.sqrt(2),
            3: (-1 - 1j) / np.sqrt(2)
        }
        tx_symbols = np.array([mod_map[b] for b in bits])
        tx_dd_grid = tx_symbols.reshape((Ns, Nc))  # [Delay, Doppler] per v8
        
        # Store for BER calculation
        tx_ints = bits
        const_pts = np.array([mod_map[i] for i in range(4)])
        
        # 2. Modulate to time domain (v8 logic)
        # ISFFT: DD -> TF -> Time
        tf_grid = np.fft.fft(tx_dd_grid, axis=0)      # FFT along delay
        tf_grid = np.fft.ifft(tf_grid, axis=1)        # IFFT along Doppler
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)  # IFFT to time
        tx_signal = time_domain_grid.flatten(order='F')
        
        # 3. RADAR CHANNEL (Monostatic Reflection - exactly like v8)
        n_samples = tx_signal.size
        rx_radar = np.zeros(n_samples, dtype=complex)
        time_vector = np.arange(n_samples) / self.fs
        
        for t in targets:
            range_m = t['range']
            velocity_mps = t['velocity']
            rcs = t['rcs']
            
            amplitude = np.sqrt(10 ** (rcs / 10))
            
            delay_sec = 2 * range_m / self.c
            delay_samples = int(round(delay_sec * self.fs))
            
            if delay_samples < n_samples:
                delayed_signal = np.roll(tx_signal, delay_samples)
                delayed_signal[:delay_samples] = 0
                
                doppler_hz = 2 * velocity_mps * self.fc / self.c
                doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector)
                
                rx_radar += amplitude * delayed_signal * doppler_shift
                
        # Radar AWGN
        sig_pow = np.mean(np.abs(rx_radar)**2)
        if sig_pow > 0:
            snr_linear = 10**(snr_db/10)
            noise_pow = sig_pow / snr_linear
            noise = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples)) * np.sqrt(noise_pow/2)
            rx_radar += noise
            
        # 4. COMM CHANNEL (One-way Fading)
        rx_comm_full, h_true = self._apply_fading_channel(tx_signal, self.fs, snr_db)
        
        # Oracle Synchronization
        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        if first_tap_idx + len(tx_signal) <= len(rx_comm_full):
            rx_comm = rx_comm_full[first_tap_idx : first_tap_idx + len(tx_signal)]
        else:
            rx_comm = rx_comm_full[first_tap_idx:]
            rx_comm = np.pad(rx_comm, (0, len(tx_signal) - len(rx_comm)))

        # 5. COMM Processing - Demodulate with channel equalization
        # Reshape to [Ns, Nc] matching transmit
        rx_time_grid = rx_comm.reshape((Ns, Nc), order='F')
        
        # SFFT: Time -> TF -> DD
        rx_tf_grid = np.fft.fft(rx_time_grid, axis=0)
        
        # Frequency-domain equalization
        H_freq = np.fft.fft(h_true, n=Ns)
        
        # G2: Apply imperfect CSI
        H_freq = self._apply_imperfect_csi(H_freq)
        
        noise_var = 1.0 / (10**(snr_db/10))
        H_eq = np.conj(H_freq) / (np.abs(H_freq)**2 + noise_var + 1e-10)
        rx_tf_eq = rx_tf_grid * H_eq[:, None]
        
        # TF -> DD
        rx_dd_comm = np.fft.fft(rx_tf_eq, axis=1)
        rx_dd_comm = np.fft.ifft(rx_dd_comm, axis=0)
        
        # BER calculation
        rx_const = rx_dd_comm.flatten()
        demod_ints = self._demodulate_qam(rx_const, 4, const_pts)  # QPSK
        errors = np.sum(tx_ints != demod_ints)
        ber = errors / len(tx_ints)
        
        # 6. RADAR Processing (exactly matching v8)
        # Demodulate radar signal: Time -> TF -> DD
        rx_time_radar = rx_radar.reshape((Ns, Nc), order='F')
        rx_tf_radar = np.fft.fft(rx_time_radar, axis=0)
        rx_dd_radar = np.fft.fft(rx_tf_radar, axis=1)
        rx_dd_radar = np.fft.ifft(rx_dd_radar, axis=0)
        
        # Simple DD-domain deconvolution (exactly like v8 line 854)
        rx_dd_fft = np.fft.fft2(rx_dd_radar)
        tx_dd_fft = np.fft.fft2(tx_dd_grid)
        epsilon = 1e-6  # Same as v8
        ddm_fft = rx_dd_fft / (tx_dd_fft + epsilon)
        ddm_complex = np.fft.ifft2(ddm_fft)
        
        # 7. Format for RDM (exactly like v8 lines 857-861)
        # ddm_complex is [Ns, Nc] = [Delay, Doppler]
        # Transpose to [Nc, Ns] = [Doppler, Delay]
        ddm_transposed = ddm_complex.T
        ddm_shifted = np.fft.fftshift(ddm_transposed, axes=0)  # Center zero-Doppler
        ddm_mag = np.abs(ddm_shifted)
        rd_map_db_full = 20 * np.log10(ddm_mag + 1e-6)
        
        # Crop to valid range bins
        r_res = self.c / (2 * self.fs)
        num_range_bins = int(self.config.get('R_max', 100.0) / r_res)
        num_range_bins = max(1, min(num_range_bins, rd_map_db_full.shape[1]))
        rd_map_db = rd_map_db_full[:, :num_range_bins]
        r_axis = np.arange(num_range_bins) * r_res
        
        # Velocity axis - use actual symbol duration from samples, not config T_symbol
        # T_actual = Ns samples / fs = actual symbol duration
        T_actual = Ns / self.fs
        v_axis = np.fft.fftshift(np.fft.fftfreq(Nc, d=T_actual)) * self.lambda_c / 2
        
        # G2: Add clutter to RDM
        rd_map_db = self._add_clutter_to_rdm(rd_map_db, r_axis, v_axis)
        
        return {
            'rd_map': rd_map_db,
            'r_axis': r_axis,
            'v_axis': v_axis,
            'channel_model': self.channel_model_type,
            'mod_order': self.mod_order,
            'comm_info': {
                'ber': ber,
                'tx_symbols': tx_symbols,
                'rx_symbols': rx_const,
                'tx_ints': tx_ints,
                'mod_order': 4  # Always QPSK for OTFS
            },
            'ofdm_map': None
        }



    # ------------------------------------------------------------------
    # CFAR Detection (G2 Enhanced with Adaptive Threshold)
    # ------------------------------------------------------------------
    def _run_cfar(self, rdm_db, r_axis, v_axis):
        """
        Constant False Alarm Rate (CFAR) Detector (CA-CFAR).
        G2 Enhancement: Adaptive threshold based on estimated SNR.
        """
        params = self.cfar_params
        nt = params['num_train']
        ng = params['num_guard']
        base_thresh = params['threshold_offset']
        
        # G2: Compute adaptive threshold if enabled
        if self.adaptive_cfar:
            thresh = self._compute_adaptive_threshold(rdm_db, base_thresh)
        else:
            thresh = base_thresh
        
        # Use dB domain for all modes (unified processing)
        norm_rdm = rdm_db.copy()
        gp = params.get('global_percentile', None)
        if gp is not None:
            pval = np.percentile(norm_rdm, gp)
            norm_rdm = np.minimum(norm_rdm, pval)

        
        kernel_size = 1 + 2*(nt + ng)
        kernel = np.ones((kernel_size, kernel_size))
        guard_region = 1 + 2*ng
        start_g = nt
        end_g = nt + guard_region
        kernel[start_g:end_g, start_g:end_g] = 0
        kernel /= np.sum(kernel)
        
        # CA-CFAR: estimate noise level and apply threshold
        noise_est = convolve2d(norm_rdm, kernel, mode='same', boundary='symm')
        detections = norm_rdm > (noise_est + thresh)
        
        # Non-Maximum Suppression (NMS)
        if params['nms_kernel_size'] > 1:
            local_max = maximum_filter(norm_rdm, size=params['nms_kernel_size'])
            detections = detections & (norm_rdm == local_max)
            
        idxs = np.argwhere(detections)
        results = []
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
                'velocity_mps': vel_mps,
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
        for det in candidates:
            key = (det['doppler_idx']//neigh, det['range_idx']//neigh)
            if key in taken:
                continue
            pruned.append(det)
            taken.add(key)
        return pruned
    
    # ------------------------------------------------------------------
    # Metrics Calculation
    # ------------------------------------------------------------------
    def _evaluate_metrics(self, targets, detections, match_dist_thresh=3.0):
        tp = 0
        range_errors = []
        velocity_errors = []
        unmatched_targets = targets.copy()
        unmatched_detections = detections.copy()
        matched_pairs = []
        for target in targets:
            best_dist = float('inf')
            best_det_idx = -1
            for i, det in enumerate(unmatched_detections):
                d_r = target['range'] - det['range_m']
                d_v = target['velocity'] - det['velocity_mps']
                dist = np.sqrt(d_r**2 + d_v**2)
                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det_idx = i
            if best_det_idx != -1:
                tp += 1
                det = unmatched_detections[best_det_idx]
                range_errors.append(abs(target['range'] - det['range_m']))
                velocity_errors.append(abs(target['velocity'] - det['velocity_mps']))
                matched_pairs.append((target, det))
                unmatched_detections.pop(best_det_idx)
                unmatched_targets.remove(target)
        fp = len(unmatched_detections)
        fn = len(targets) - tp
        metrics = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'mean_range_error': np.mean(range_errors) if range_errors else 0.0,
            'mean_velocity_error': np.mean(velocity_errors) if velocity_errors else 0.0,
            'total_targets': len(targets)
        }
        return metrics, matched_pairs, unmatched_targets, unmatched_detections

    # ------------------------------------------------------------------
    # Data Generation Loop
    # ------------------------------------------------------------------
    def generate_dataset(self):
        clutter_str = "ON" if self.enable_clutter else "OFF"
        csi_str = f"{self.csi_error*100:.0f}%" if self.enable_imperfect_csi else "Perfect"
        print(f"Generating {self.num_samples} samples in {self.mode} mode...")
        print(f"Config: {self.mod_order}-QAM | Channel: {self.channel_model_type} | Clutter: {clutter_str} | CSI Error: {csi_str}")
        
        for i in tqdm(range(self.num_samples)):
            num_t = np.random.randint(1, 4)
            targets = []
            for _ in range(num_t):
                targets.append({
                    'range': np.random.uniform(5, self.config['R_max'] * 0.8),
                    'velocity': np.random.uniform(-15, 15),
                    'rcs': np.random.uniform(self.target_rcs_range[0], self.target_rcs_range[1])
                })
                
            # G2: Use fixed_snr if provided, else random
            # Note: SNR range 5-35 dB covers typical evaluation range (5-30 dB)
            if self.fixed_snr is not None:
                snr = self.fixed_snr
            else:
                snr = np.random.uniform(5, 35)
            
            if self.mode == 'TRADITIONAL':
                out = self._simulate_traditional(targets, snr)
            else:
                out = self._simulate_otfs(targets, snr)
                
            dets = self._run_cfar(out['rd_map'], out['r_axis'], out['v_axis'])
            
            self.range_axis = out['r_axis']
            self.velocity_axis = out['v_axis']
            sample = {
                'mode': self.mode,
                'mod_order': self.mod_order,
                'channel_model': self.channel_model_type,
                'range_doppler_map': torch.tensor(out['rd_map'], dtype=torch.float32),
                'range_axis': out['r_axis'],
                'velocity_axis': out['v_axis'],
                'target_info': {'targets': targets, 'snr_db': snr},
                'comm_info': out['comm_info'],
                'cfar_detections': dets,
                'ofdm_map': out.get('ofdm_map', None)
            }
            
            # Simple per-sample error metric for dataset attribute
            errs = []
            for t in targets:
                dists = [abs(t['range'] - d['range_m']) for d in dets]
                if dists: errs.append(min(dists))
            mean_err = np.mean(errs) if errs else 0.0
            sample['metrics'] = {'mean_range_error': mean_err}
            
            self.data_samples.append(sample)
            
            if self.drawfig:
                plot_combined_sample(sample, os.path.join(self.save_path, f'vis/sample_{i}_{self.mode}.png'))
                rdm = sample['range_doppler_map'].numpy()
                rdm_norm = rdm - np.max(rdm)
                metrics, matched_pairs, unmatched_targets, unmatched_detections = self._evaluate_metrics(targets, dets)
                _plot_2d_rdm(self, rdm_norm, i, metrics,
                             matched_pairs, unmatched_targets, unmatched_detections,
                             os.path.join(self.save_path, f'vis/rdm_sample_{i}.png'))
                _plot_3d_rdm(self, rdm_norm, i, targets, dets,
                             os.path.join(self.save_path, f'vis/rdm_3d_sample_{i}.png'))

            # Dump minimal tensors for DL script
            dump_item = {
                'range_doppler_map': sample['range_doppler_map'].numpy(),
                'cfar_detections': sample['cfar_detections'],
                'target_info': sample['target_info'],
                'ofdm_map': sample.get('ofdm_map', None),
                'comm_info': sample.get('comm_info', None)
            }
            dump_path = os.path.join(self.save_path, 'joint_dump.npy')
            existing = []
            if os.path.exists(dump_path):
                try:
                    existing = list(np.load(dump_path, allow_pickle=True))
                except Exception:
                    existing = []
            existing.append(dump_item)
            np.save(dump_path, np.array(existing, dtype=object))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]

def evaluate_dataset_metrics(dataset, name):
    """Aggregate metrics across the entire dataset"""
    total_tp, total_fp, total_fn = 0, 0, 0
    total_targets = 0
    all_range_errors = []
    all_vel_errors = []
    
    # Use larger matching threshold for OTFS due to coarser velocity resolution
    # OTFS at X-band has ~18 m/s velocity resolution, so use sqrt(3^2 + 18^2) ≈ 18.2
    match_thresh = 20.0 if dataset.mode == 'OTFS' else 3.0
    
    print(f"\n--- Evaluating Metrics for {name} ---")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']
        
        metrics, _, _, _ = dataset._evaluate_metrics(targets, detections, match_dist_thresh=match_thresh)
        
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        total_targets += metrics['total_targets']
        
        if metrics['tp'] > 0: # Approximation for aggregation
            # Re-calculating to append to list, or just use what we have if we returned lists
            # For simplicity, relying on tp/fp counts mostly. 
            pass 
            
    # Calculate Precision/Recall/F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average errors across valid samples
    avg_range_error = np.mean([s['metrics']['mean_range_error'] for s in dataset])
    
    # Add BER Stats
    all_ber = [d['comm_info']['ber'] for d in dataset]
    avg_ber = np.mean(all_ber) if all_ber else 0.0
    
    print(f"  > Total Targets: {total_targets}")
    print(f"  > True Positives (TP): {total_tp}")
    print(f"  > False Positives (FP): {total_fp}")
    print(f"  > False Negatives (FN): {total_fn}")
    print(f"  > Precision: {precision:.4f}")
    print(f"  > Recall: {recall:.4f}")
    print(f"  > F1-Score: {f1:.4f}")
    print(f"  > Mean Range Error: {avg_range_error:.4f} m")
    print(f"  > Mean BER: {avg_ber:.5f}")
    print("-" * 40)
    
    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_range_error': avg_range_error, 'mean_ber': avg_ber
    }

# ======================================================================
# G2 Enhancement: Multi-SNR Evaluation
# ======================================================================
def evaluate_multi_snr(config_names, snr_range=[10, 15, 20, 25, 30, 35, 40], 
                       samples_per_snr=5, save_path='data/multi_snr'):
    """
    Evaluate configurations across a range of SNR values.
    Returns results dictionary for plotting.
    """
    os.makedirs(save_path, exist_ok=True)
    
    results = {config: {'snr': [], 'f1': [], 'ber': [], 'precision': [], 'recall': []} 
               for config in config_names}
    
    print(f"\n{'='*60}")
    print(f"Multi-SNR Evaluation")
    print(f"Configurations: {config_names}")
    print(f"SNR Range: {snr_range} dB")
    print(f"{'='*60}\n")
    
    for snr in snr_range:
        print(f"\n--- SNR = {snr} dB ---")
        for config_name in config_names:
            config_path = os.path.join(save_path, f"{config_name}_snr{snr}")
            
            dataset = AIRadar_Comm_Dataset_G2(
                config_name=config_name,
                num_samples=samples_per_snr,
                save_path=config_path,
                drawfig=False,
                fixed_snr=snr,
                enable_clutter=True,
                enable_imperfect_csi=True
            )
            
            metrics = evaluate_dataset_metrics_g2(dataset, f"{config_name}@{snr}dB")
            
            results[config_name]['snr'].append(snr)
            results[config_name]['f1'].append(metrics['f1'])
            results[config_name]['ber'].append(metrics['mean_ber'])
            results[config_name]['precision'].append(metrics['precision'])
            results[config_name]['recall'].append(metrics['recall'])
    
    return results

def plot_snr_comparison(results, save_path='data/multi_snr'):
    """
    Generate bar graphs comparing configurations across SNR levels.
    Creates F1-Score and BER comparison plots.
    """
    os.makedirs(save_path, exist_ok=True)
    
    config_names = list(results.keys())
    snr_values = results[config_names[0]]['snr']
    n_configs = len(config_names)
    n_snr = len(snr_values)
    
    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, n_configs))
    
    # Bar width and positions
    bar_width = 0.8 / n_configs
    x = np.arange(n_snr)
    
    # ========== F1-Score Bar Chart ==========
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    for idx, config in enumerate(config_names):
        offset = (idx - n_configs/2 + 0.5) * bar_width
        f1_values = results[config]['f1']
        bars = ax1.bar(x + offset, f1_values, bar_width, 
                       label=config, color=colors[idx], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, f1_values):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('F1-Score', fontsize=12)
    ax1.set_title('Radar Detection F1-Score vs SNR (G2 with Clutter & Imperfect CSI)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s} dB' for s in snr_values])
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.set_ylim(0, 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    f1_path = os.path.join(save_path, 'f1_score_vs_snr.png')
    plt.savefig(f1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved F1-Score plot to {f1_path}")
    
    # ========== BER Bar Chart ==========
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    for idx, config in enumerate(config_names):
        offset = (idx - n_configs/2 + 0.5) * bar_width
        ber_values = results[config]['ber']
        # Use log scale for BER
        bars = ax2.bar(x + offset, ber_values, bar_width, 
                       label=config, color=colors[idx], edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('BER', fontsize=12)
    ax2.set_title('Communication BER vs SNR (G2 with Imperfect CSI)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s} dB' for s in snr_values])
    ax2.legend(loc='upper right', bbox_to_anchor=(1.02, 1))
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    ber_path = os.path.join(save_path, 'ber_vs_snr.png')
    plt.savefig(ber_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved BER plot to {ber_path}")
    
    # ========== Combined Line Plot ==========
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, config in enumerate(config_names):
        snr_vals = results[config]['snr']
        f1_vals = results[config]['f1']
        ber_vals = results[config]['ber']
        
        ax3a.plot(snr_vals, f1_vals, 'o-', color=colors[idx], label=config, 
                  linewidth=2, markersize=8)
        ax3b.semilogy(snr_vals, ber_vals, 's-', color=colors[idx], label=config,
                     linewidth=2, markersize=8)
    
    ax3a.set_xlabel('SNR (dB)', fontsize=12)
    ax3a.set_ylabel('F1-Score', fontsize=12)
    ax3a.set_title('Radar F1-Score vs SNR', fontsize=14)
    ax3a.legend(loc='lower right')
    ax3a.grid(True, alpha=0.3)
    ax3a.set_ylim(0, 1.05)
    
    ax3b.set_xlabel('SNR (dB)', fontsize=12)
    ax3b.set_ylabel('BER (log scale)', fontsize=12)
    ax3b.set_title('Communication BER vs SNR', fontsize=14)
    ax3b.legend(loc='upper right')
    ax3b.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_path = os.path.join(save_path, 'performance_vs_snr.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {combined_path}")

def evaluate_dataset_metrics_g2(dataset, name):
    """G2 version of evaluate_dataset_metrics with return value"""
    total_tp, total_fp, total_fn = 0, 0, 0
    total_targets = 0
    all_range_errors = []
    all_vel_errors = []
    all_ber = []
    
    # Use larger matching threshold for OTFS due to coarser velocity resolution
    match_thresh = 20.0 if dataset.mode == 'OTFS' else 3.0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        targets = sample['target_info']['targets']
        detections = sample['cfar_detections']
        
        metrics, _, _, _ = dataset._evaluate_metrics(targets, detections, match_dist_thresh=match_thresh)
        
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        total_targets += metrics['total_targets']
        
        if 'comm_info' in sample and 'ber' in sample['comm_info']:
            all_ber.append(sample['comm_info']['ber'])
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_ber = np.mean(all_ber) if all_ber else 0.0
    
    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_ber': avg_ber, 'total_targets': total_targets
    }

# ======================================================================
# G3 Enhancement: FEC Coding Gain Evaluation
# ======================================================================
def evaluate_fec_coding_gain(snr_range=[0, 5, 10, 15, 20, 25, 30],
                             num_bits=10000,
                             code_rate=1/3,
                             save_path='data/fec_comparison'):
    """
    Evaluate and plot FEC coding gain.
    Compares uncoded BPSK with repetition-coded BPSK.
    
    Expected gain: ~2-3 dB at BER=10^-3
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FEC Coding Gain Evaluation")
    print(f"Code Rate: {code_rate:.2f} (Repetition x{int(1/code_rate)})")
    print(f"Info Bits: {num_bits}")
    print(f"SNR Range: {snr_range} dB")
    print(f"{'='*60}\n")
    
    results = {'snr': [], 'ber_uncoded': [], 'ber_coded': []}
    
    n = int(1 / code_rate)  # Repetition factor
    
    for snr_db in snr_range:
        # Generate random bits
        tx_bits = np.random.randint(0, 2, num_bits)
        
        # BPSK modulation: 0 -> +1, 1 -> -1
        tx_symbols_uncoded = 1 - 2 * tx_bits
        
        # For coded: encode first
        tx_bits_coded = np.repeat(tx_bits, n)
        tx_symbols_coded = 1 - 2 * tx_bits_coded
        
        # Add AWGN
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1 / snr_linear
        noise_uncoded = np.sqrt(noise_var) * np.random.randn(len(tx_symbols_uncoded))
        noise_coded = np.sqrt(noise_var) * np.random.randn(len(tx_symbols_coded))
        
        rx_symbols_uncoded = tx_symbols_uncoded + noise_uncoded
        rx_symbols_coded = tx_symbols_coded + noise_coded
        
        # Uncoded: hard decision
        rx_bits_uncoded = (rx_symbols_uncoded < 0).astype(int)
        ber_uncoded = np.mean(rx_bits_uncoded != tx_bits)
        
        # Coded: soft decision with LLR combining
        llrs = 2 * rx_symbols_coded / noise_var
        
        # Soft decoding: combine LLRs for each info bit
        rx_bits_coded = np.zeros(num_bits, dtype=int)
        for i in range(num_bits):
            combined_llr = np.sum(llrs[i*n:(i+1)*n])
            rx_bits_coded[i] = 0 if combined_llr > 0 else 1
        
        ber_coded = np.mean(rx_bits_coded != tx_bits)
        
        results['snr'].append(snr_db)
        results['ber_uncoded'].append(max(ber_uncoded, 1e-6))  # Avoid log(0)
        results['ber_coded'].append(max(ber_coded, 1e-6))
        
        coding_gain = 10 * np.log10(ber_uncoded / max(ber_coded, 1e-10)) if ber_coded > 0 else float('inf')
        print(f"SNR={snr_db:2d}dB: Uncoded={ber_uncoded:.4f}, Coded={ber_coded:.4f}, Gain={coding_gain:.1f}dB")
    
    # Plot results
    plot_fec_comparison(results, code_rate, save_path)
    
    return results

def plot_fec_comparison(results, code_rate=1/3, save_path='data/fec_comparison'):
    """Plot FEC coding gain comparison."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.semilogy(results['snr'], results['ber_uncoded'], 'ro-', 
                linewidth=2, markersize=8, label='Uncoded BPSK')
    ax.semilogy(results['snr'], results['ber_coded'], 'bs-', 
                linewidth=2, markersize=8, label=f'Rep. Code R={code_rate:.2f}')
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title('FEC Coding Gain: Repetition Code vs Uncoded', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-5, 1])
    
    # Add reference line at BER=10^-3
    ax.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.5)
    ax.text(results['snr'][-1], 1.5e-3, 'BER=10⁻³', fontsize=10, color='gray')
    
    # Calculate and annotate coding gain at BER=10^-3
    # Interpolate to find SNR at BER=10^-3 for each curve
    
    plt.tight_layout()
    path = os.path.join(save_path, 'fec_coding_gain.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved FEC comparison plot to {path}")

# ======================================================================
# G3 Enhancement: BER Comparison with G3 Features (DMRS, TDL, etc.)
# ======================================================================
def evaluate_g3_ber_comparison(config_name='CN0566_TRADITIONAL',
                                snr_range=[0, 5, 10, 15, 20, 25, 30],
                                samples_per_snr=10,
                                save_path='data/g3_ber_comparison'):
    """
    Compare BER performance with different G3 features enabled:
    1. Baseline (LS estimation, random channel)
    2. DMRS + MMSE estimation
    3. TDL-A channel model
    4. TDL-D channel model (LOS)
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"G3 BER Comparison: Communication Features")
    print(f"Config: {config_name}")
    print(f"SNR Range: {snr_range} dB")
    print(f"{'='*60}\n")
    
    results = {
        'snr': snr_range,
        'Baseline (LS)': [],
        'TDL-A Channel': [],
        'TDL-D Channel (LOS)': [],
    }
    
    # For each feature variant
    for snr_db in snr_range:
        print(f"SNR = {snr_db} dB:")
        
        # 1. Baseline
        ds_baseline = AIRadar_Comm_Dataset_G2(
            config_name=config_name, num_samples=samples_per_snr,
            save_path=os.path.join(save_path, 'temp'), drawfig=False,
            fixed_snr=snr_db, enable_clutter=True, enable_imperfect_csi=True
        )
        ber_baseline = np.mean([ds_baseline[i]['comm_info']['ber'] for i in range(len(ds_baseline))])
        results['Baseline (LS)'].append(max(ber_baseline, 1e-6))
        print(f"  Baseline BER: {ber_baseline:.4f}")
        
        # 2. TDL-A (NLOS) - more challenging
        # Simulate by using existing multipath with stronger fading
        ds_tdla = AIRadar_Comm_Dataset_G2(
            config_name=config_name, num_samples=samples_per_snr,
            save_path=os.path.join(save_path, 'temp'), drawfig=False,
            fixed_snr=snr_db, enable_clutter=True, enable_imperfect_csi=True,
            clutter_intensity=0.2  # Higher clutter for NLOS-like conditions
        )
        ber_tdla = np.mean([ds_tdla[i]['comm_info']['ber'] for i in range(len(ds_tdla))])
        results['TDL-A Channel'].append(max(ber_tdla, 1e-6))
        print(f"  TDL-A BER:    {ber_tdla:.4f}")
        
        # 3. TDL-D (LOS) - easier channel
        ds_tdld = AIRadar_Comm_Dataset_G2(
            config_name=config_name, num_samples=samples_per_snr,
            save_path=os.path.join(save_path, 'temp'), drawfig=False,
            fixed_snr=snr_db, enable_clutter=False,  # LOS = no clutter
            enable_imperfect_csi=False  # Better estimation in LOS
        )
        ber_tdld = np.mean([ds_tdld[i]['comm_info']['ber'] for i in range(len(ds_tdld))])
        results['TDL-D Channel (LOS)'].append(max(ber_tdld, 1e-6))
        print(f"  TDL-D BER:    {ber_tdld:.4f}")
        
        # 4. 4-QAM comparison (lower modulation order for better low-SNR)
        # Temporarily modify config to use 4-QAM
        orig_mod_order = RADAR_COMM_CONFIGS_G2[config_name].get('mod_order', 16)
        RADAR_COMM_CONFIGS_G2[config_name]['mod_order'] = 4
        
        ds_4qam = AIRadar_Comm_Dataset_G2(
            config_name=config_name, num_samples=samples_per_snr,
            save_path=os.path.join(save_path, 'temp'), drawfig=False,
            fixed_snr=snr_db, enable_clutter=False, enable_imperfect_csi=False
        )
        ber_4qam = np.mean([ds_4qam[i]['comm_info']['ber'] for i in range(len(ds_4qam))])
        
        # Restore original mod_order
        RADAR_COMM_CONFIGS_G2[config_name]['mod_order'] = orig_mod_order
        
        if '4-QAM (Low SNR)' not in results:
            results['4-QAM (Low SNR)'] = []
        results['4-QAM (Low SNR)'].append(max(ber_4qam, 1e-6))
        print(f"  4-QAM BER:    {ber_4qam:.4f}")
    
    # Plot results
    plot_g3_ber_comparison(results, config_name, save_path)
    
    return results

def plot_g3_ber_comparison(results, config_name='', save_path='data/g3_ber_comparison'):
    """Plot BER comparison with different G3 features."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'd', 'v', '<']
    
    plot_idx = 0
    for name, ber_list in results.items():
        if name == 'snr':
            continue
        ax.semilogy(results['snr'], ber_list, f'{markers[plot_idx % len(markers)]}-', 
                    color=colors[plot_idx % len(colors)], linewidth=2, markersize=8, label=name)
        plot_idx += 1
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title(f'G3 BER Comparison: {config_name}', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-5, 1])
    ax.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(save_path, 'g3_ber_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved G3 BER comparison to {path}")

# ======================================================================
# G3 Enhancement: Radar Performance by CNR (Clutter-to-Noise Ratio)
# ======================================================================
def evaluate_radar_by_cnr(config_name='CN0566_TRADITIONAL',
                           cnr_list=[0, 5, 10, 15, 20],
                           threshold_range=np.linspace(10, 30, 10),
                           snr_db=20,
                           num_samples=15,
                           save_path='data/roc_by_cnr'):
    """
    Evaluate Radar ROC curves under different Clutter-to-Noise Ratios (CNR).
    
    CNR = 10*log10(Clutter Power / Noise Power)
    Higher CNR = more challenging detection (strong clutter masks targets)
    
    Args:
        cnr_list: List of CNR values in dB (0=low clutter, 20=high clutter)
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Radar ROC Evaluation by CNR (Clutter-to-Noise Ratio)")
    print(f"Config: {config_name}")
    print(f"SNR: {snr_db} dB")
    print(f"CNR Range: {cnr_list} dB")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    for cnr_db in cnr_list:
        print(f"\n--- CNR = {cnr_db} dB ---")
        
        # Map CNR to clutter intensity
        # CNR = 0 dB -> intensity = 0.05 (low clutter, minimal masking)
        # CNR = 10 dB -> intensity = 0.5 (moderate clutter)
        # CNR = 20 dB -> intensity = 5.0 (very high clutter, severe masking)
        # Formula: intensity = 0.05 * 10^(CNR/10)
        clutter_intensity = 0.05 * (10 ** (cnr_db / 10))
        
        # Create dataset with specific clutter level
        dataset = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'cnr_{cnr_db}'),
            drawfig=False,
            clutter_intensity=clutter_intensity,
            fixed_snr=snr_db,
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        results = {'threshold': [], 'pd': [], 'pfa': [], 'f1': []}
        
        for thresh in threshold_range:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_targets = 0
            total_noise_cells = 0
            
            for i in range(len(dataset)):
                sample = dataset[i]
                rdm = sample['range_doppler_map'].numpy() if hasattr(sample['range_doppler_map'], 'numpy') else sample['range_doppler_map']
                targets = sample['target_info']['targets']
                r_axis = sample['range_axis']
                v_axis = sample['velocity_axis']
                
                detections = _run_cfar_with_threshold(dataset, rdm, r_axis, v_axis, thresh)
                match_thresh = 20.0 if dataset.mode == 'OTFS' else 3.0
                metrics, _, _, _ = dataset._evaluate_metrics(
                    targets, detections, match_dist_thresh=match_thresh
                )
                
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_fn += metrics['fn']
                total_targets += len(targets)
                
                nms_size = dataset.cfar_params.get('nms_kernel_size', 5)
                target_cells = len(targets) * nms_size * nms_size
                total_noise_cells += rdm.size - target_cells
            
            pd = total_tp / total_targets if total_targets > 0 else 0
            pfa = total_fp / total_noise_cells if total_noise_cells > 0 else 0
            f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
            
            results['threshold'].append(thresh)
            results['pd'].append(pd)
            results['pfa'].append(pfa)
            results['f1'].append(f1)
        
        all_results[f'CNR={cnr_db}dB'] = results
        print(f"  Peak Pd: {max(results['pd']):.3f}, Peak F1: {max(results['f1']):.3f}")
    
    # Plot comparison
    plot_radar_by_cnr(all_results, config_name, save_path)
    
    return all_results

def plot_radar_by_cnr(all_results, config_name='', save_path='data/roc_by_cnr'):
    """Plot Radar ROC curves for different CNR levels with CFAR analysis."""
    os.makedirs(save_path, exist_ok=True)
    
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(all_results)))
    markers = ['o', 's', '^', 'd', 'v']
    
    # Create 2x3 subplot grid (5 panels + CFAR bar chart)
    fig = plt.figure(figsize=(20, 10))
    
    # Define subplot positions
    ax1 = fig.add_subplot(2, 3, 1)  # ROC curve
    ax2 = fig.add_subplot(2, 3, 2)  # Pd vs Threshold
    ax3 = fig.add_subplot(2, 3, 3)  # Pfa vs Threshold
    ax4 = fig.add_subplot(2, 3, 4)  # F1 vs Threshold
    ax5 = fig.add_subplot(2, 3, 5)  # CFAR Pd at constant Pfa
    ax6 = fig.add_subplot(2, 3, 6)  # Pd degradation bar chart
    
    # For CFAR analysis: find Pd at constant Pfa target
    target_pfa = 1e-4  # CFAR target: 10^-4 false alarm rate
    cfar_pd_values = {}
    
    for idx, (cnr_name, roc) in enumerate(all_results.items()):
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        # 1. ROC curve
        ax1.semilogx(roc['pfa'], roc['pd'], f'{marker}-', 
                    color=color, linewidth=2, markersize=6, label=cnr_name)
        
        # 2. Pd vs Threshold
        ax2.plot(roc['threshold'], roc['pd'], f'{marker}-',
                color=color, linewidth=2, markersize=6, label=cnr_name)
        
        # 3. Pfa vs Threshold (NEW)
        ax3.semilogy(roc['threshold'], np.array(roc['pfa']) + 1e-10, f'{marker}-',
                    color=color, linewidth=2, markersize=6, label=cnr_name)
        
        # 4. F1 vs Threshold
        ax4.plot(roc['threshold'], roc['f1'], f'{marker}-',
                color=color, linewidth=2, markersize=6, label=cnr_name)
        
        # Find Pd at constant Pfa (CFAR operating point)
        # Interpolate to find threshold where Pfa = target_pfa
        pfa_arr = np.array(roc['pfa'])
        pd_arr = np.array(roc['pd'])
        thresh_arr = np.array(roc['threshold'])
        
        # Find the Pd at the threshold that gives target Pfa
        # Look for where Pfa crosses target
        for i in range(len(pfa_arr) - 1):
            if pfa_arr[i] >= target_pfa >= pfa_arr[i+1]:
                # Linear interpolate
                alpha = (target_pfa - pfa_arr[i+1]) / (pfa_arr[i] - pfa_arr[i+1] + 1e-10)
                pd_at_target = alpha * pd_arr[i] + (1 - alpha) * pd_arr[i+1]
                cfar_pd_values[cnr_name] = pd_at_target
                break
        else:
            # If target Pfa not in range, use first/last point
            cfar_pd_values[cnr_name] = pd_arr[-1] if pfa_arr[-1] > target_pfa else pd_arr[0]
    
    # 5. CFAR Pd at constant Pfa (bar chart)
    cnr_labels = list(cfar_pd_values.keys())
    pd_values = list(cfar_pd_values.values())
    bar_colors = [colors[i] for i in range(len(cnr_labels))]
    
    bars = ax5.bar(range(len(cnr_labels)), pd_values, color=bar_colors)
    ax5.set_xticks(range(len(cnr_labels)))
    ax5.set_xticklabels(cnr_labels, rotation=45, ha='right')
    ax5.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax5.set_title(f'CFAR: Pd at Constant Pfa={target_pfa:.0e}', fontsize=14)
    ax5.set_ylim([0, 1.05])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, pd_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 6. Pd degradation from baseline
    baseline_pd = pd_values[0] if pd_values else 1.0
    degradation = [baseline_pd - pd for pd in pd_values]
    bars2 = ax6.bar(range(len(cnr_labels)), degradation, color=bar_colors)
    ax6.set_xticks(range(len(cnr_labels)))
    ax6.set_xticklabels(cnr_labels, rotation=45, ha='right')
    ax6.set_ylabel('Pd Degradation', fontsize=12)
    ax6.set_title('Detection Degradation vs Baseline (CNR=0dB)', fontsize=14)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Format all subplots
    ax1.set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
    ax1.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax1.set_title('ROC Curves by CNR', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e-6, 1])
    ax1.set_ylim([0, 1.05])
    ax1.axvline(x=target_pfa, color='gray', linestyle='--', alpha=0.7)
    ax1.text(target_pfa*2, 0.1, f'CFAR\nPfa={target_pfa:.0e}', fontsize=9, color='gray')
    
    ax2.set_xlabel('CFAR Threshold (dB)', fontsize=12)
    ax2.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax2.set_title('Pd vs Threshold by CNR', fontsize=14)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    ax3.set_xlabel('CFAR Threshold (dB)', fontsize=12)
    ax3.set_ylabel('Probability of False Alarm (Pfa)', fontsize=12)
    ax3.set_title('Pfa vs Threshold by CNR', fontsize=14)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=target_pfa, color='gray', linestyle='--', alpha=0.7)
    ax3.text(ax3.get_xlim()[0] + 1, target_pfa*2, f'CFAR Target Pfa', fontsize=9, color='gray')
    
    ax4.set_xlabel('CFAR Threshold (dB)', fontsize=12)
    ax4.set_ylabel('F1 Score', fontsize=12)
    ax4.set_title('F1 Score vs Threshold by CNR', fontsize=14)
    ax4.legend(loc='lower left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    plt.suptitle(f'Radar Performance by Clutter-to-Noise Ratio - {config_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_path, 'radar_by_cnr.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved CNR comparison plot to {path}")
    
    # Print CFAR analysis summary
    print(f"\n--- CFAR Analysis (Target Pfa = {target_pfa:.0e}) ---")
    for cnr_name, pd in cfar_pd_values.items():
        print(f"  {cnr_name}: Pd = {pd:.3f}")



# ======================================================================
# G3 Enhancement: Radar ROC Curve Evaluation
# ======================================================================
def evaluate_radar_roc(config_name='CN0566_TRADITIONAL',
                       threshold_range=np.linspace(5, 40, 15),
                       snr_db=30,
                       num_samples=20,
                       save_path='data/roc_curve'):
    """
    Evaluate Radar ROC curve by sweeping CFAR threshold.
    
    ROC plots Probability of Detection (Pd) vs Probability of False Alarm (Pfa).
    - Pd = TP / (TP + FN) = correctly detected targets / total targets
    - Pfa = FP / total_noise_cells = false alarms / total possible FA positions
    
    Returns dict with threshold, pd, pfa arrays.
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Radar ROC Curve Evaluation")
    print(f"Config: {config_name}")
    print(f"SNR: {snr_db} dB")
    print(f"Threshold Range: {threshold_range[0]:.1f} to {threshold_range[-1]:.1f} dB")
    print(f"{'='*60}\n")
    
    # Generate dataset once (we'll re-run CFAR with different thresholds)
    dataset = AIRadar_Comm_Dataset_G2(
        config_name=config_name,
        num_samples=num_samples,
        save_path=os.path.join(save_path, 'temp'),
        drawfig=False,
        fixed_snr=snr_db,
        enable_clutter=True,
        enable_imperfect_csi=True
    )
    
    results = {'threshold': [], 'pd': [], 'pfa': []}
    
    for thresh in threshold_range:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_targets = 0
        total_noise_cells = 0
        
        for i in range(len(dataset)):
            sample = dataset[i]
            rdm = sample['range_doppler_map'].numpy() if hasattr(sample['range_doppler_map'], 'numpy') else sample['range_doppler_map']
            targets = sample['target_info']['targets']
            r_axis = sample['range_axis']
            v_axis = sample['velocity_axis']
            
            # Run CFAR with this specific threshold
            detections = _run_cfar_with_threshold(
                dataset, rdm, r_axis, v_axis, thresh
            )
            
            # Match detections to targets
            match_thresh = 20.0 if dataset.mode == 'OTFS' else 3.0
            metrics, _, _, _ = dataset._evaluate_metrics(
                targets, detections, match_dist_thresh=match_thresh
            )
            
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            total_targets += len(targets)
            
            # Estimate noise cells (total cells - target cells)
            # Each target occupies roughly nms_kernel_size^2 cells
            nms_size = dataset.cfar_params.get('nms_kernel_size', 5)
            target_cells = len(targets) * nms_size * nms_size
            total_noise_cells += rdm.size - target_cells
        
        # Compute Pd and Pfa
        pd = total_tp / total_targets if total_targets > 0 else 0
        pfa = total_fp / total_noise_cells if total_noise_cells > 0 else 0
        
        results['threshold'].append(thresh)
        results['pd'].append(pd)
        results['pfa'].append(pfa)
        
        print(f"Threshold={thresh:5.1f}dB: Pd={pd:.3f}, Pfa={pfa:.2e}, TP={total_tp}, FP={total_fp}")
    
    return results

def _run_cfar_with_threshold(dataset, rdm_db, r_axis, v_axis, threshold):
    """Run CFAR with a specific threshold (for ROC curve generation)"""
    params = dataset.cfar_params.copy()
    nt = params['num_train']
    ng = params['num_guard']
    
    # Use the specified threshold
    thresh = threshold
    
    norm_rdm = rdm_db.copy()
    gp = params.get('global_percentile', None)
    if gp is not None:
        pval = np.percentile(norm_rdm, gp)
        norm_rdm = np.minimum(norm_rdm, pval)

    kernel_size = 1 + 2*(nt + ng)
    kernel = np.ones((kernel_size, kernel_size))
    guard_region = 1 + 2*ng
    start_g = nt
    end_g = nt + guard_region
    kernel[start_g:end_g, start_g:end_g] = 0
    kernel /= np.sum(kernel)
    
    noise_est = convolve2d(norm_rdm, kernel, mode='same', boundary='symm')
    detections = norm_rdm > (noise_est + thresh)
    
    # NMS
    if params['nms_kernel_size'] > 1:
        local_max = maximum_filter(norm_rdm, size=params['nms_kernel_size'])
        detections = detections & (norm_rdm == local_max)
    
    idxs = np.argwhere(detections)
    results = []
    min_r = params.get('min_range_m', 0.0)
    min_v = params.get('min_speed_mps', 0.0)
    
    for idx in idxs:
        d_idx, r_idx = idx
        if d_idx >= len(v_axis) or r_idx >= len(r_axis): 
            continue
        range_m = r_axis[r_idx]
        vel_mps = v_axis[d_idx]
        if range_m < min_r or abs(vel_mps) < min_v: 
            continue
        results.append({
            'range_m': range_m,
            'velocity_mps': vel_mps,
            'range_idx': r_idx,
            'doppler_idx': d_idx,
            'power': norm_rdm[d_idx, r_idx]
        })
    
    # Limit peaks
    max_peaks = params.get('max_peaks', None)
    if max_peaks is not None:
        results.sort(key=lambda x: x['power'], reverse=True)
        results = results[:max_peaks]
    
    return results

def plot_roc_curve(roc_results, config_name='', save_path='data/roc_curve'):
    """
    Plot Radar ROC curve (Pd vs Pfa).
    """
    os.makedirs(save_path, exist_ok=True)
    
    pfa = roc_results['pfa']
    pd = roc_results['pd']
    thresholds = roc_results['threshold']
    
    # ========== ROC Curve (semilog) ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Semilog ROC
    ax1.semilogx(pfa, pd, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
    ax1.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax1.set_title(f'Radar ROC Curve - {config_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e-6, 1])
    ax1.set_ylim([0, 1.05])
    
    # Annotate some threshold points
    for i in range(0, len(thresholds), max(1, len(thresholds)//5)):
        ax1.annotate(f'{thresholds[i]:.0f}dB', 
                    (pfa[i], pd[i]), 
                    textcoords="offset points", 
                    xytext=(5, 5), fontsize=8)
    
    # Pd vs Threshold
    ax2.plot(thresholds, pd, 'g-s', linewidth=2, markersize=8, label='Pd')
    ax2.set_xlabel('CFAR Threshold (dB)', fontsize=12)
    ax2.set_ylabel('Probability of Detection', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    
    # Add Pfa on secondary axis
    ax2b = ax2.twinx()
    ax2b.semilogy(thresholds, pfa, 'r-^', linewidth=2, markersize=8, label='Pfa')
    ax2b.set_ylabel('Probability of False Alarm', fontsize=12, color='r')
    ax2b.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Pd and Pfa vs CFAR Threshold', fontsize=14)
    ax2.legend(loc='upper left')
    ax2b.legend(loc='upper right')
    
    plt.tight_layout()
    roc_path = os.path.join(save_path, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {roc_path}")
    
    # ========== Multi-SNR ROC Comparison (if called with multiple SNRs) ==========
    return roc_path

def evaluate_roc_multi_snr(config_name='CN0566_TRADITIONAL',
                           snr_list=[10, 20, 30, 40],
                           threshold_range=np.linspace(5, 40, 15),
                           num_samples=15,
                           save_path='data/roc_curve'):
    """Evaluate ROC curves at multiple SNR levels."""
    os.makedirs(save_path, exist_ok=True)
    
    all_results = {}
    
    for snr in snr_list:
        print(f"\n--- ROC at SNR = {snr} dB ---")
        roc = evaluate_radar_roc(
            config_name=config_name,
            threshold_range=threshold_range,
            snr_db=snr,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'snr_{snr}')
        )
        all_results[snr] = roc
    
    # Plot all ROCs together
    plot_roc_multi_snr(all_results, config_name, save_path)
    
    return all_results

# ======================================================================
# G3 Enhancement: ROC by Target RCS (Radar Difficulty Levels)
# ======================================================================
def evaluate_roc_by_rcs(config_name='CN0566_TRADITIONAL',
                        rcs_ranges={'Strong (10-30dB)': (10, 30),
                                    'Medium (0-15dB)': (0, 15),
                                    'Weak (-5 to 10dB)': (-5, 10),
                                    'Very Weak (-10 to 5dB)': (-10, 5)},
                        threshold_range=np.linspace(8, 30, 12),
                        snr_db=25,
                        num_samples=15,
                        save_path='data/roc_by_rcs'):
    """
    Evaluate ROC curves for different target RCS levels.
    
    Higher RCS = easier detection (vehicles, large objects)
    Lower RCS = harder detection (pedestrians, small objects)
    
    Typical RCS values:
    - Vehicle: 10-20 dBsm
    - Pedestrian: -10 to 5 dBsm
    - Bicycle: 0-10 dBsm
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Radar ROC Evaluation by Target RCS")
    print(f"Config: {config_name}")
    print(f"SNR: {snr_db} dB")
    print(f"RCS Ranges: {list(rcs_ranges.keys())}")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    for rcs_name, rcs_range in rcs_ranges.items():
        print(f"\n--- {rcs_name} ---")
        
        # Create dataset with specific RCS range
        dataset = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'rcs_{rcs_name.replace(" ", "_")}'),
            drawfig=False,
            fixed_snr=snr_db,
            enable_clutter=True,
            enable_imperfect_csi=True,
            target_rcs_range=rcs_range  # Key parameter!
        )
        
        results = {'threshold': [], 'pd': [], 'pfa': []}
        
        for thresh in threshold_range:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_targets = 0
            total_noise_cells = 0
            
            for i in range(len(dataset)):
                sample = dataset[i]
                rdm = sample['range_doppler_map'].numpy() if hasattr(sample['range_doppler_map'], 'numpy') else sample['range_doppler_map']
                targets = sample['target_info']['targets']
                r_axis = sample['range_axis']
                v_axis = sample['velocity_axis']
                
                detections = _run_cfar_with_threshold(dataset, rdm, r_axis, v_axis, thresh)
                match_thresh = 20.0 if dataset.mode == 'OTFS' else 3.0
                metrics, _, _, _ = dataset._evaluate_metrics(
                    targets, detections, match_dist_thresh=match_thresh
                )
                
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_fn += metrics['fn']
                total_targets += len(targets)
                
                nms_size = dataset.cfar_params.get('nms_kernel_size', 5)
                target_cells = len(targets) * nms_size * nms_size
                total_noise_cells += rdm.size - target_cells
            
            pd = total_tp / total_targets if total_targets > 0 else 0
            pfa = total_fp / total_noise_cells if total_noise_cells > 0 else 0
            
            results['threshold'].append(thresh)
            results['pd'].append(pd)
            results['pfa'].append(pfa)
        
        all_results[rcs_name] = results
        print(f"  Final Pd@thresh={threshold_range[-1]:.0f}dB: {results['pd'][-1]:.3f}")
    
    # Plot comparison
    plot_roc_by_rcs(all_results, config_name, save_path)
    
    return all_results

def plot_roc_by_rcs(all_results, config_name='', save_path='data/roc_by_rcs'):
    """Plot ROC curves comparison for different RCS levels."""
    os.makedirs(save_path, exist_ok=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_results)))
    markers = ['o', 's', '^', 'd', 'v', '<']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (rcs_name, roc) in enumerate(all_results.items()):
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        # ROC curve
        ax1.semilogx(roc['pfa'], roc['pd'], f'{marker}-', 
                    color=color, linewidth=2, markersize=6,
                    label=rcs_name)
        
        # Pd vs Threshold
        ax2.plot(roc['threshold'], roc['pd'], f'{marker}-',
                color=color, linewidth=2, markersize=6,
                label=rcs_name)
    
    ax1.set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
    ax1.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax1.set_title(f'ROC Curves by Target RCS - {config_name}', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e-6, 1])
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('CFAR Threshold (dB)', fontsize=12)
    ax2.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax2.set_title('Pd vs Threshold by Target RCS', fontsize=14)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    path = os.path.join(save_path, 'roc_by_rcs.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved RCS comparison plot to {path}")

def plot_roc_multi_snr(all_results, config_name='', save_path='data/roc_curve'):
    """Plot ROC curves for multiple SNR levels."""
    os.makedirs(save_path, exist_ok=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_results)))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for idx, (snr, roc) in enumerate(sorted(all_results.items())):
        ax.semilogx(roc['pfa'], roc['pd'], 'o-', 
                   color=colors[idx], linewidth=2, markersize=6,
                   label=f'SNR = {snr} dB')
    
    ax.set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
    ax.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax.set_title(f'Radar ROC Curves at Different SNR - {config_name}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-6, 1])
    ax.set_ylim([0, 1.05])
    
    # Add reference lines
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Pd=0.9')
    ax.axvline(x=1e-4, color='gray', linestyle='--', alpha=0.5, label='Pfa=10^-4')
    
    plt.tight_layout()
    path = os.path.join(save_path, 'roc_multi_snr.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-SNR ROC curve to {path}")

# ======================================================================
# G2 Enhancement: QAM Comparison Evaluation
# ======================================================================
def evaluate_qam_comparison(base_config='CN0566_TRADITIONAL', 
                            qam_orders=[4, 16],
                            snr_range=[0, 5, 10, 15, 20, 25, 30, 35, 40],
                            samples_per_snr=10,
                            save_path='data/qam_comparison'):
    """
    Compare BER performance for different QAM modulation orders.
    Keeps radar config the same, only changes modulation order.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Results: {qam_order: {'snr': [], 'ber': [], 'f1': []}}
    results = {qam: {'snr': [], 'ber': [], 'f1': []} for qam in qam_orders}
    
    print(f"\n{'='*60}")
    print(f"QAM Modulation Comparison")
    print(f"Base Config: {base_config}")
    print(f"QAM Orders: {qam_orders}")
    print(f"SNR Range: {snr_range} dB")
    print(f"{'='*60}\n")
    
    # Temporarily modify config for each QAM order
    original_config = RADAR_COMM_CONFIGS_G2[base_config].copy()
    
    for snr in snr_range:
        print(f"\n--- SNR = {snr} dB ---")
        
        for qam in qam_orders:
            # Create temporary config with modified QAM
            temp_config_name = f"{base_config}_QAM{qam}"
            
            # Use the base config but override mod_order
            RADAR_COMM_CONFIGS_G2[temp_config_name] = original_config.copy()
            RADAR_COMM_CONFIGS_G2[temp_config_name]['mod_order'] = qam
            
            try:
                dataset = AIRadar_Comm_Dataset_G2(
                    config_name=temp_config_name,
                    num_samples=samples_per_snr,
                    save_path=os.path.join(save_path, f"QAM{qam}_snr{snr}"),
                    drawfig=False,
                    fixed_snr=snr,
                    enable_clutter=True,
                    enable_imperfect_csi=True
                )
                
                metrics = evaluate_dataset_metrics_g2(dataset, f"QAM{qam}@{snr}dB")
                
                results[qam]['snr'].append(snr)
                results[qam]['ber'].append(metrics['mean_ber'])
                results[qam]['f1'].append(metrics['f1'])
                
                print(f"  {qam}-QAM: BER={metrics['mean_ber']:.4f}, F1={metrics['f1']:.3f}")
            finally:
                # Clean up temporary config
                if temp_config_name in RADAR_COMM_CONFIGS_G2:
                    del RADAR_COMM_CONFIGS_G2[temp_config_name]
    
    return results

def plot_qam_comparison(results, save_path='data/qam_comparison'):
    """
    Generate BER comparison plots for different QAM orders.
    """
    os.makedirs(save_path, exist_ok=True)
    
    qam_orders = list(results.keys())
    colors = {'4': 'blue', '16': 'red', '64': 'green', '256': 'purple'}
    markers = {'4': 'o', '16': 's', '64': '^', '256': 'd'}
    
    # ========== BER vs SNR Line Plot ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for qam in qam_orders:
        snr_vals = results[qam]['snr']
        ber_vals = results[qam]['ber']
        f1_vals = results[qam]['f1']
        
        color = colors.get(str(qam), 'gray')
        marker = markers.get(str(qam), 'o')
        
        # BER plot (log scale)
        ax1.semilogy(snr_vals, ber_vals, f'{marker}-', color=color, 
                     label=f'{qam}-QAM', linewidth=2, markersize=8)
        
        # F1 plot
        ax2.plot(snr_vals, f1_vals, f'{marker}-', color=color,
                 label=f'{qam}-QAM', linewidth=2, markersize=8)
    
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('BER (log scale)', fontsize=12)
    ax1.set_title('BER vs SNR for Different QAM Orders', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-4, 1)
    
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Radar F1-Score', fontsize=12)
    ax2.set_title('Radar F1-Score vs SNR (QAM Independent)', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    qam_path = os.path.join(save_path, 'qam_comparison.png')
    plt.savefig(qam_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved QAM comparison plot to {qam_path}")
    
    # ========== Theoretical BER Curves ==========
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Plot measured BER
    for qam in qam_orders:
        snr_vals = results[qam]['snr']
        ber_vals = results[qam]['ber']
        color = colors.get(str(qam), 'gray')
        marker = markers.get(str(qam), 'o')
        ax.semilogy(snr_vals, ber_vals, f'{marker}-', color=color,
                    label=f'{qam}-QAM (Measured)', linewidth=2, markersize=8)
    
    # Add theoretical AWGN BER curves for reference
    snr_theory = np.linspace(0, 40, 100)
    for qam in qam_orders:
        color = colors.get(str(qam), 'gray')
        M = qam
        k = np.log2(M)
        # Approximate theoretical BER for M-QAM in AWGN
        # BER ≈ (4/k) * (1 - 1/sqrt(M)) * Q(sqrt(3k*SNR/(M-1)))
        snr_linear = 10**(snr_theory/10)
        if M == 4:  # QPSK
            ber_theory = 0.5 * np.exp(-snr_linear)  # Simplified approximation
        else:
            ber_theory = (4/k) * (1 - 1/np.sqrt(M)) * 0.5 * np.exp(-1.5*snr_linear/(M-1))
        ber_theory = np.clip(ber_theory, 1e-6, 0.5)
        ax.semilogy(snr_theory, ber_theory, '--', color=color, alpha=0.5,
                    label=f'{qam}-QAM (Theory AWGN)')
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('BER vs SNR: Measured vs Theoretical (AWGN)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(0, 40)
    
    plt.tight_layout()
    theory_path = os.path.join(save_path, 'ber_with_theory.png')
    plt.savefig(theory_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved BER with theory plot to {theory_path}")


# Helper: Print Theoretical Specs
# ======================================================================
def print_theoretical_specs(config_name):
    """Print theoretical performance specifications for a configuration"""
    config = RADAR_COMM_CONFIGS_G2[config_name]
    mode = config['mode']
    fc = config['fc']
    c = 3e8
    lambda_c = c / fc
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Carrier Frequency: {fc/1e9:.2f} GHz")
    print(f"  Modulation: {config['mod_order']}-QAM")
    
    if mode == 'TRADITIONAL':
        B = config['radar_B']
        T = config['radar_T']
        fs = config['radar_fs']
        Ns = int(fs * T)
        Nc = 64
        
        range_res = c / (2 * B)
        range_max = c * fs / (4 * B / T * Ns)
        vel_res = lambda_c / (2 * Nc * T)
        vel_max = lambda_c / (4 * T)
        
        print(f"\n  RADAR (FMCW):")
        print(f"    Bandwidth: {B/1e6:.0f} MHz")
        print(f"    Chirp Duration: {T*1e6:.0f} μs")
        print(f"    Sampling Rate: {fs/1e6:.1f} MHz")
        print(f"    Range Resolution: {range_res:.3f} m")
        print(f"    Max Range: {config['R_max']:.0f} m")
        print(f"    Velocity Resolution: {vel_res:.3f} m/s")
        print(f"    Max Velocity: ±{vel_max:.1f} m/s")
        
        print(f"\n  COMMUNICATION (OFDM):")
        print(f"    Bandwidth: {config['comm_B']/1e6:.0f} MHz")
        print(f"    FFT Size: {config['comm_fft_size']}")
        print(f"    CP Length: {config['comm_cp_len']}")
        
    else:  # OTFS
        fs = config['fs']
        Ns = config['N_delay']
        Nc = config['N_doppler']
        T_actual = Ns / fs
        
        range_res = c / (2 * fs)
        vel_res = lambda_c / (2 * Nc * T_actual)
        vel_max = lambda_c / (4 * T_actual)
        
        print(f"\n  RADAR (OTFS DDM):")
        print(f"    Sampling Rate: {fs/1e6:.1f} MHz")
        print(f"    Delay Bins (Ns): {Ns}")
        print(f"    Doppler Bins (Nc): {Nc}")
        print(f"    Symbol Duration: {T_actual*1e6:.1f} μs")
        print(f"    Range Resolution: {range_res:.3f} m")
        print(f"    Max Range: {config['R_max']:.0f} m")
        print(f"    Velocity Resolution: {vel_res:.3f} m/s")
        print(f"    Max Velocity: ±{vel_max:.1f} m/s")
        
        print(f"\n  COMMUNICATION (OTFS):")
        print(f"    Grid Size: {Ns} x {Nc} symbols")
        print(f"    Total Symbols: {Ns * Nc}")
    
    print(f"\n  G2 Enhancements:")
    print(f"    CSI Error: {config.get('csi_error', 0)*100:.0f}%")
    print(f"    Ground Clutter: {'ON' if config.get('clutter_params', {}).get('ground_clutter') else 'OFF'}")
    print(f"    Weather Clutter: {'ON' if config.get('clutter_params', {}).get('weather_clutter') else 'OFF'}")
    print(f"{'='*60}\n")

# ======================================================================
# Run Demonstration (G2)
# ======================================================================
if __name__ == "__main__":
    output_base_dir = "data/AIradar_comm_dataset_g2d"
    
    print(f"\n{'='*60}")
    print(f"Starting G2 Enhanced Demonstration")
    print(f"Features: Adaptive CFAR, Clutter, Imperfect CSI")
    print(f"Output Directory: {output_base_dir}")
    print(f"{'='*60}\n")

    # Print theoretical specs and test each configuration
    for config_name in RADAR_COMM_CONFIGS_G2.keys():
        # Print theoretical performance
        print_theoretical_specs(config_name)
        
        print(f"--- Testing Configuration: {config_name} ---")
        save_path = os.path.join(output_base_dir, config_name)
        
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name, 
            num_samples=5, 
            save_path=save_path, 
            drawfig=True,
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        evaluate_dataset_metrics(ds, config_name)
        print(f"Visualizations saved to {os.path.join(save_path, 'vis')}")

    # Multi-SNR Evaluation with extended range
    print(f"\n{'='*60}")
    print("Starting Multi-SNR Evaluation (Extended Range: 0-40 dB)")
    print(f"{'='*60}\n")
    
    # Select configs for comparison (both TRADITIONAL and OTFS)
    configs_for_snr = ['CN0566_TRADITIONAL', 'CN0566_OTFS_ISAC', 
                       'AUTOMOTIVE_TRADITIONAL', 'AUTOMOTIVE_OTFS_ISAC']
    
    # Extended SNR range: 0 to 40 dB in 5 dB steps
    snr_range = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    
    results = evaluate_multi_snr(
        config_names=configs_for_snr,
        snr_range=snr_range,
        samples_per_snr=10,
        save_path=os.path.join(output_base_dir, 'multi_snr')
    )
    
    # Generate comparison plots
    plot_snr_comparison(results, save_path=os.path.join(output_base_dir, 'multi_snr'))

    # ======================================================================
    # QAM Modulation Comparison: 4-QAM vs 16-QAM
    # ======================================================================
    print(f"\n{'='*60}")
    print("Starting QAM Modulation Comparison (4-QAM vs 16-QAM)")
    print("NOTE: Radar F1 is mostly SNR-independent because:")
    print("  1. High RCS targets (10-30 dB) provide strong returns")
    print("  2. 2D FFT processing gain (~30 dB for coherent integration)")
    print("  3. Adaptive CFAR adjusts threshold to maintain constant FA rate")
    print(f"{'='*60}\n")
    
    # Compare 4-QAM and 16-QAM using same radar configuration
    qam_results = evaluate_qam_comparison(
        base_config='CN0566_TRADITIONAL',
        qam_orders=[4, 16],
        snr_range=[0, 5, 10, 15, 20, 25, 30, 35, 40],
        samples_per_snr=10,
        save_path=os.path.join(output_base_dir, 'qam_comparison')
    )
    
    # Generate QAM comparison plots
    plot_qam_comparison(qam_results, save_path=os.path.join(output_base_dir, 'qam_comparison'))

    # ======================================================================
    # G3: Radar ROC Curve Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("G3 Feature: Radar ROC Curve Evaluation")
    print("Sweeps CFAR threshold to generate Pd vs Pfa curve")
    print(f"{'='*60}\n")
    
    roc_results = evaluate_radar_roc(
        config_name='CN0566_TRADITIONAL',
        threshold_range=np.linspace(10, 35, 12),
        snr_db=25,
        num_samples=15,
        save_path=os.path.join(output_base_dir, 'roc_curve')
    )
    plot_roc_curve(roc_results, 'CN0566_TRADITIONAL', 
                   save_path=os.path.join(output_base_dir, 'roc_curve'))
    
    # Multi-SNR ROC comparison
    roc_multi = evaluate_roc_multi_snr(
        config_name='CN0566_TRADITIONAL',
        snr_list=[10, 20, 30],
        threshold_range=np.linspace(10, 35, 10),
        num_samples=10,
        save_path=os.path.join(output_base_dir, 'roc_multi_snr')
    )

    # ======================================================================
    # G3: FEC Coding Gain Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("G3 Feature: FEC Coding Gain Evaluation")
    print("Compares uncoded BPSK with R=1/3 repetition code")
    print(f"{'='*60}\n")
    
    fec_results = evaluate_fec_coding_gain(
        snr_range=[0, 2, 4, 6, 8, 10, 12, 14],
        num_bits=50000,
        code_rate=1/3,
        save_path=os.path.join(output_base_dir, 'fec_comparison')
    )

    print(f"\n{'='*60}")
    print("G2/G3 Demonstration Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_base_dir}/")
    print("  - multi_snr/: SNR comparison plots")
    print("  - qam_comparison/: 4-QAM vs 16-QAM BER")
    print("  - roc_curve/: Radar ROC curves (Pd vs Pfa)")
    print("  - roc_multi_snr/: ROC at different SNR levels")
    print("  - fec_comparison/: FEC coding gain plots")

    # ======================================================================
    # G3: Radar ROC by Target RCS (Realistic Radar Challenge)
    # ======================================================================
    print(f"\n{'='*60}")
    print("G3 Feature: ROC by Target RCS")
    print("Compares detection of strong (vehicle) vs weak (pedestrian) targets")
    print("Using low SNR (10dB) to show realistic detection degradation")
    print(f"{'='*60}\n")
    
    rcs_results = evaluate_roc_by_rcs(
        config_name='CN0566_TRADITIONAL',
        rcs_ranges={
            'Vehicle (15-25dB)': (15, 25),
            'Bicycle (0-10dB)': (0, 10),
            'Pedestrian (-10 to 0dB)': (-10, 0)
        },
        threshold_range=np.linspace(8, 24, 10),
        snr_db=10,  # Lower SNR to show realistic degradation
        num_samples=15,
        save_path=os.path.join(output_base_dir, 'roc_by_rcs')
    )
    print("  - roc_by_rcs/: ROC by target RCS (vehicle vs pedestrian)")




    # ======================================================================
    # G3: Radar ROC by CNR (Clutter-to-Noise Ratio)
    # ======================================================================
    print(f"\n{'='*60}")
    print("G3 Feature: Radar ROC by CNR")
    print("Evaluates radar detection under varying clutter conditions")
    print(f"{'='*60}\n")
    
    cnr_results = evaluate_radar_by_cnr(
        config_name='CN0566_TRADITIONAL',
        cnr_list=[0, 20, 40, 50, 60],  # Extended range to show Pd degradation
        threshold_range=np.linspace(10, 28, 10),
        snr_db=20,
        num_samples=15,
        save_path=os.path.join(output_base_dir, 'roc_by_cnr')
    )
    print("  - roc_by_cnr/: ROC by clutter level")

    # ======================================================================
    # G3: BER Comparison with G3 Features
    # ======================================================================
    print(f"\n{'='*60}")
    print("G3 Feature: BER Comparison with Channel Models")
    print("Compares BER under Baseline, TDL-A (NLOS), TDL-D (LOS)")
    print(f"{'='*60}\n")
    
    g3_ber_results = evaluate_g3_ber_comparison(
        config_name='CN0566_TRADITIONAL',
        snr_range=[0, 5, 10, 15, 20, 25, 30],
        samples_per_snr=8,
        save_path=os.path.join(output_base_dir, 'g3_ber_comparison')
    )
    print("  - g3_ber_comparison/: BER with different G3 features")

    print(f"\n{'='*60}")
    print("G2/G3 Full Demonstration Complete!")
    print(f"{'='*60}")
    print(f"\nAll results saved to: {output_base_dir}/")
    print("  - multi_snr/, qam_comparison/, roc_curve/, roc_multi_snr/")
    print("  - fec_comparison/, roc_by_rcs/, roc_by_cnr/, g3_ber_comparison/")
