"""
Advanced ISAC Simulation (v5 - Merged): OFDM vs. OTFS

This script merges the working ISAC v4 simulation (OFDM/OTFS)
with the advanced visualization suite from the SyntheticRadarDataset.

It performs the following:
1.  (FIXED) Uses the corrected v4 OTFS transforms to ensure
    BER is not 1.0 and sensing maps have hotspots.
2.  (NEW) Creates an `ISACRadarParams` class to "translate"
    OFDM/OTFS parameters (M, N, SCS) into the format
    the visualization functions expect (range_res, vel_res, etc.).
3.  (NEW) Uses the advanced `viz_rd`, `viz_rd_3d`, and `viz_bev_3d`
    functions to generate all plots.
4.  (MODIFIED) The `generate_physical_targets` function now
    creates both the "on-grid" targets for the simulation
    and the `target_gt_list` needed by the visualizers.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import os
import json
import time

# =========================================================================
# 0. Global Parameters & Helper Functions
# =========================================================================

# --- Create directory to save figures ---
SAVE_DIR = "isac_simulation_figures"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")

# --- Grid & Waveform Parameters ---
N_SUBCARRIERS = 512    # Number of subcarriers (Delay bins, M)
N_SYMBOLS = 128        # Number of OFDM symbols (Doppler bins, N)
CP_LEN = 36            # Length of OFDM Cyclic Prefix
QAM_ORDER = 4          # QPSK

# --- Physical System Parameters ---
C = 3e8                # Speed of light (m/s)
FC = 77e9              # Carrier frequency (Hz)
SCS_HZ = 60e3          # Subcarrier Spacing (Hz)
BANDWIDTH_HZ = N_SUBCARRIERS * SCS_HZ # 30.72 MHz
FS_HZ = BANDWIDTH_HZ   # System sampling rate (Hz)

# --- Frame & Symbol Durations ---
T_SYMBOL_SEC = 1 / SCS_HZ  # ~16.7 us
T_SYMBOL_CP_SEC = (N_SUBCARRIERS + CP_LEN) * (1 / FS_HZ)
T_FRAME_SEC = N_SYMBOLS * T_SYMBOL_SEC # ~2.13 ms

# --- Sensing Resolution & Max Values (derived) ---
RANGE_RES_M = C / (2 * BANDWIDTH_HZ) # ~4.88 m
VEL_RES_MPS = C / (2 * FC * T_FRAME_SEC) # ~0.91 m/s

# Calculate max bins for 25m simulation
MAX_RANGE_REQUEST_M = 25.0
MAX_DELAY_BIN = int(MAX_RANGE_REQUEST_M / RANGE_RES_M) # ~5

# Max unambiguous velocity
MAX_DOPPLER_BIN = (N_SYMBOLS // 2) - 1 # 63
VEL_MAX_MPS = MAX_DOPPLER_BIN * VEL_RES_MPS # ~57.3 m/s

# --- Simulation Parameters ---
SNR_DB_LIST = np.arange(0, 21, 4) # SNR range for BER simulation
SIM_SNR_DB = 20        # A single SNR for sensing map visualization
MAX_TARGETS = 3        # Max number of targets to simulate

# --- Communication Receiver Location ---
COMM_RX_LOC = {'x': 20.0, 'y': 10.0}

# --- Separate "On-Grid" Channel for COMM BER Simulation ---
COMM_CHANNEL_BINS = [
    (4, 3, 1.0),   # LoS path (range ~19.5m, vel ~2.7 m/s)
    (8, -8, 0.4),  # NLoS path 1
    (12, 15, 0.2)  # NLoS path 2
]

# =========================================================================
# 1. NEW: RadarParams Class for ISAC Simulation
# =========================================================================

class ISACRadarParams:
    """
    NEW: A "translator" class that wraps our ISAC (OFDM/OTFS)
    parameters to make them compatible with the provided
    visualization functions.
    """
    def __init__(self, n_subcarriers, n_symbols, range_res_m, vel_res_mps,
                 sensor_height_m=1.5, azi_deg=90.0, ele_deg=30.0):
        self.n_chirps = n_symbols       # Map N_SYMBOLS (Doppler) to M
        self.n_samples = n_subcarriers  # Map N_SUBCARRIERS (Delay) to Ns
        
        self.range_res_val = range_res_m
        self.vel_res_val = vel_res_mps
        
        self.sensor_height_m = sensor_height_m
        self.azi_beamwidth_deg = azi_deg
        self.ele_beamwidth_deg = ele_deg
        self.max_range_m = self.unambiguous_range()

    def Ns_fast(self):
        return self.n_samples

    def range_res(self):
        return self.range_res_val

    def unambiguous_range(self):
        return self.range_res_val * self.n_samples

    def velocity_res(self):
        return self.vel_res_val

    def unambiguous_doppler_vel(self):
        """Returns the full unambiguous velocity range (+/- v_max)"""
        return self.vel_res_val * self.n_chirps

# =========================================================================
# 2. ISAC Core Functions (Modulation, Channel, Helpers)
# =========================================================================

# --- QAM Modulation Helpers ---
MOD_MAP = {
    0: (1 + 1j) / np.sqrt(2), 1: (1 - 1j) / np.sqrt(2),
    2: (-1 + 1j) / np.sqrt(2), 3: (-1 - 1j) / np.sqrt(2)
}

def get_qam_symbols(num_symbols):
    bits = np.random.randint(0, QAM_ORDER, num_symbols)
    return np.array([MOD_MAP[b] for b in bits])

def demodulate_qam_symbols(symbols):
    demod_bits = []
    if np.isnan(symbols).any():
        return np.random.randint(0, QAM_ORDER, symbols.size)
    for s in symbols:
        distances = {abs(s - const_s): key for const_s, key in MOD_MAP.items()}
        demod_bits.append(distances[min(distances.keys())])
    return np.array(demod_bits)

def calculate_ber(bits_tx, bits_rx):
    return np.sum(bits_tx != bits_rx) / bits_tx.size

def generate_physical_targets(n_targets):
    """
    MODIFIED: Generates "on-grid" targets for the simulation
    AND formats them as a `target_gt_list` for visualization.
    """
    physical_targets_isac = [] # For ISAC channel model
    target_bins_list = []      # For BER channel model
    target_gt_list_viz = []    # For visualization functions
    
    for _ in range(n_targets):
        # 1. Pick INTEGER bins (FIXED)
        delay_bin = np.random.randint(2, MAX_DELAY_BIN + 1)
        doppler_bin = np.random.randint(-MAX_DOPPLER_BIN, MAX_DOPPLER_BIN)
        
        # 2. Calculate physical values *from* the integer bins
        range_m = delay_bin * RANGE_RES_M
        velocity_mps = doppler_bin * VEL_RES_MPS
        
        # 3. Calculate (x, y) from range_m
        angle = np.random.uniform(-np.pi/6, np.pi/6) # +/- 30 deg FOV
        x_m = range_m * np.cos(angle)
        y_m = range_m * np.sin(angle)
        reflectivity = np.random.uniform(0.7, 1.0)
        
        # List 1: For ISAC time-domain channel
        physical_targets_isac.append({
            'x': x_m, 'y': y_m, 'velocity': velocity_mps,
            'reflectivity': reflectivity
        })
        
        # List 2: For BER DD-domain channel
        target_bins_list.append((delay_bin, doppler_bin, reflectivity))
        
        # List 3: For advanced visualization
        target_gt_list_viz.append({
            'center_xyz': [x_m, y_m, 0.75], # Assume 0.75m z-center
            'size_xyz': [4.0, 1.8, 1.5],   # Dummy car size
            'vel_xyz': [velocity_mps, 0, 0] # Assume all velocity is radial/x
        })
        
    return physical_targets_isac, target_bins_list, target_gt_list_viz

def apply_radar_channel_and_noise(tx_signal, snr_db, physical_targets, fs_hz):
    """ Simulates the MONOSTATIC (Radar) channel. """
    n_samples = tx_signal.size
    rx_signal = np.zeros(n_samples, dtype=complex)
    time_vector_sec = np.arange(n_samples) / fs_hz
    
    for target in physical_targets:
        range_m = np.sqrt(target['x']**2 + target['y']**2)
        delay_sec = 2 * range_m / C
        delay_samples = int(round(delay_sec * fs_hz))
        delayed_signal = np.roll(tx_signal, delay_samples)
        
        velocity_mps = target['velocity']
        doppler_hz = 2 * velocity_mps * FC / C
        doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector_sec)
        
        rx_signal += target['reflectivity'] * delayed_signal * doppler_shift
        
    signal_power = np.mean(np.abs(tx_signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = (signal_power / snr_linear) if signal_power > 0 else 1e-20
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * \
            np.sqrt(noise_power / 2)
    return rx_signal + noise

def apply_comm_channel_and_noise(tx_signal, snr_db, comm_channel_bins, fs_hz):
    """ Simulates the BISTATIC (Comm) channel. """
    n_samples = tx_signal.size
    rx_signal = np.zeros(n_samples, dtype=complex)
    time_vector_sec = np.arange(n_samples) / fs_hz
    delay_res_sec = 1 / BANDWIDTH_HZ
    doppler_res_hz = 1 / T_FRAME_SEC
    
    for (delay_bin, doppler_bin, reflectivity) in comm_channel_bins:
        delay_sec = delay_bin * delay_res_sec
        delay_samples = int(round(delay_sec * fs_hz))
        delayed_signal = np.roll(tx_signal, delay_samples)
        
        doppler_hz = doppler_bin * doppler_res_hz
        doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector_sec)
        rx_signal += reflectivity * delayed_signal * doppler_shift
        
    signal_power = np.mean(np.abs(tx_signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = (signal_power / snr_linear) if signal_power > 0 else 1e-20
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * \
            np.sqrt(noise_power / 2)
    return rx_signal + noise

# --- OFDM Functions ---
def ofdm_modulate(data_grid):
    time_domain_symbols = np.fft.ifft(data_grid, axis=0)
    cp_signal = time_domain_symbols[-CP_LEN:, :]
    with_cp_signal = np.concatenate((cp_signal, time_domain_symbols), axis=0)
    return with_cp_signal.flatten(order='F')

def ofdm_sensing_receiver(rx_signal, tx_signal):
    tx_grid_time = tx_signal.reshape((N_SUBCARRIERS + CP_LEN, N_SYMBOLS), order='F')
    rx_grid_time = rx_signal.reshape((N_SUBCARRIERS + CP_LEN, N_SYMBOLS), order='F')
    tx_grid_no_cp = tx_grid_time[CP_LEN:, :]
    rx_grid_no_cp = rx_grid_time[CP_LEN:, :]
    tx_grid_freq = np.fft.fft(tx_grid_no_cp, axis=0)
    rx_grid_freq = np.fft.fft(rx_grid_no_cp, axis=0)
    correlation_grid = rx_grid_freq * np.conj(tx_grid_freq)
    range_doppler_map = np.fft.ifft2(correlation_grid)
    return np.fft.fftshift(range_doppler_map)

def get_ofdm_tf_channel(target_bins_list):
    H_tf_ofdm = np.zeros((N_SUBCARRIERS, N_SYMBOLS), dtype=complex)
    delay_res_sec = 1 / BANDWIDTH_HZ
    doppler_res_hz = 1 / T_FRAME_SEC
    for m in range(N_SUBCARRIERS):
        for l in range(N_SYMBOLS):
            t_sec = l * T_SYMBOL_CP_SEC
            f_hz = m * SCS_HZ
            for (delay_bin, doppler_bin, reflectivity) in target_bins_list:
                delay_sec = delay_bin * delay_res_sec
                doppler_hz = doppler_bin * doppler_res_hz
                phase = 1j * 2 * np.pi * (doppler_hz * t_sec - f_hz * delay_sec)
                H_tf_ofdm[m, l] += reflectivity * np.exp(phase)
    return H_tf_ofdm

# --- OTFS Functions (Corrected v4) ---
def otfs_modulate_v2(dd_grid):
    tf_grid = np.fft.ifft(dd_grid, axis=0)
    tf_grid = np.fft.fft(tf_grid, axis=1)
    time_domain_grid = np.fft.ifft(tf_grid, axis=0)
    return time_domain_grid.flatten(order='F')

def otfs_demodulate_v2(rx_signal):
    time_domain_grid = rx_signal.reshape((N_SUBCARRIERS, N_SYMBOLS), order='F')
    tf_grid = np.fft.fft(time_domain_grid, axis=0)
    dd_grid = np.fft.ifft(tf_grid, axis=1)
    dd_grid = np.fft.fft(dd_grid, axis=0)
    return dd_grid

def get_otfs_dd_channel(target_bins_list):
    H_ideal_dd = np.zeros((N_SUBCARRIERS, N_SYMBOLS), dtype=complex)
    for (delay_bin, doppler_bin, reflectivity) in target_bins_list:
        doppler_idx = np.mod(doppler_bin, N_SYMBOLS)
        delay_idx = delay_bin
        H_ideal_dd[delay_idx, doppler_idx] = reflectivity
    return H_ideal_dd

# =========================================================================
# 3. IMPORTED: Advanced Visualization Functions
# =========================================================================

def get_target_rd_coords(target_gt_list, params: ISACRadarParams, sensor_origin=np.array([0.0,0.0,0.0])):
    """ Helper to convert ground truth boxes to RD coordinates. """
    gt_coords = []
    R_unamb = params.unambiguous_range()
    v_unamb = params.unambiguous_doppler_vel()

    for gt in target_gt_list:
        if 'center_xyz' in gt and 'vel_xyz' in gt:
            p_abs = np.array(gt['center_xyz'])
            v_abs = np.array(gt['vel_xyz'])
        else:
            continue
        
        p_rel = p_abs - sensor_origin
        range_m = np.linalg.norm(p_rel)
        if range_m < 1e-3: continue
        
        u_vec = p_rel / range_m
        rad_vel_ms = np.dot(u_vec, v_abs) 
        
        if range_m > R_unamb: continue
        if abs(rad_vel_ms) > v_unamb / 2: continue
            
        gt_coords.append({
            'range': range_m, 
            'velocity': rad_vel_ms, 
            'label': f"({range_m:.1f}m, {rad_vel_ms:.1f}m/s)"
        })
    return gt_coords

def viz_rd(RD_dB, out_png, params: ISACRadarParams, target_gt_list=None, sensor_origin=np.array([0.0,0.0,0.0])):
    M, Ns = RD_dB.shape
    
    R_unamb = params.unambiguous_range()
    range_bins = np.linspace(0, R_unamb, Ns)
    
    v_unamb = params.unambiguous_doppler_vel()
    doppler_bins = np.linspace(-v_unamb/2, v_unamb/2, M)

    plt.figure(figsize=(12, 6))
    
    v_max = np.max(RD_dB)
    v_min = v_max - 80 # Show 80dB dynamic range
    
    plt.pcolormesh(range_bins, doppler_bins, RD_dB, shading='auto', vmin=v_min, vmax=v_max, cmap='jet')
    
    if target_gt_list:
        gt_coords = get_target_rd_coords(target_gt_list, params, sensor_origin)
        if len(gt_coords) > 0:
            gt_ranges = [gc['range'] for gc in gt_coords]
            gt_vels = [gc['velocity'] for gc in gt_coords]
            gt_labels = [gc['label'] for gc in gt_coords]
                
            plt.plot(gt_ranges, gt_vels, 'rx', markersize=10, markeredgewidth=2, label='Ground Truth')
            
            for r, v, label in zip(gt_ranges, gt_vels, gt_labels):
                plt.text(r + 0.5, v + 0.5, label, color='red', fontsize=9, ha='left', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
                         
            plt.legend()
        
    plt.title("Range-Doppler Map (dB)")
    plt.xlabel(f"Range (m) [Max: {R_unamb:.1f} m]")
    plt.ylabel(f"Velocity (m/s) [Max: +/- {v_unamb/2:.1f} m/s]")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def viz_rd_3d(RD_dB, out_png, params: ISACRadarParams, target_gt_list=None, sensor_origin=np.array([0.0,0.0,0.0])):
    M, Ns = RD_dB.shape
    R_unamb = params.unambiguous_range()
    range_axis = np.linspace(0, R_unamb, Ns)
    v_unamb = params.unambiguous_doppler_vel()
    doppler_axis = np.linspace(-v_unamb/2, v_unamb/2, M)
    R_grid, V_grid = np.meshgrid(range_axis, doppler_axis)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    v_max = np.max(RD_dB)
    v_min = v_max - 80 
    
    Z_surface = np.maximum(RD_dB, v_min)
    ax.plot_surface(R_grid, V_grid, Z_surface, cmap='jet', 
                    rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.8)
    
    if target_gt_list:
        gt_coords = get_target_rd_coords(target_gt_list, params, sensor_origin)
        if len(gt_coords) > 0:
            gt_ranges = [gc['range'] for gc in gt_coords]
            gt_vels = [gc['velocity'] for gc in gt_coords]
            gt_z = [v_max + 10 for _ in gt_ranges]
            
            ax.scatter(gt_ranges, gt_vels, gt_z, color='red', marker='x', s=100, linewidth=2, label='Ground Truth')
            
            for r, v, z in zip(gt_ranges, gt_vels, gt_z):
                z_surface_val = Z_surface[np.argmin(np.abs(doppler_axis - v)), np.argmin(np.abs(range_axis - r))]
                ax.plot([r, r], [v, v], [z, z_surface_val], 'r--', linewidth=0.8)
            ax.legend()

    ax.set_title("3D Range-Doppler Map (Clipped at floor)")
    ax.set_xlabel("Range (m)"); ax.set_ylabel("Velocity (m/s)"); ax.set_zlabel("Power (dB)")
    ax.view_init(elev=30, azim=-135)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs) if renderer else 0

    def draw(self, renderer):
        self.do_3d_projection(renderer)
        super().draw(renderer)

def plot_cube_wireframe_3d(ax, cube_gt, color='r'):
    center = np.array(cube_gt['center_xyz'])
    size = np.array(cube_gt['size_xyz'])
    half_size = size / 2.0
    corners = np.array([
        [center[0] - half_size[0], center[1] - half_size[1], center[2] - half_size[2]],
        [center[0] + half_size[0], center[1] - half_size[1], center[2] - half_size[2]],
        [center[0] + half_size[0], center[1] + half_size[1], center[2] - half_size[2]],
        [center[0] - half_size[0], center[1] + half_size[1], center[2] - half_size[2]],
        [center[0] - half_size[0], center[1] - half_size[1], center[2] + half_size[2]],
        [center[0] + half_size[0], center[1] - half_size[1], center[2] + half_size[2]],
        [center[0] + half_size[0], center[1] + half_size[1], center[2] + half_size[2]],
        [center[0] - half_size[0], center[1] + half_size[1], center[2] + half_size[2]]
    ])
    edges = [
        [corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]],
        [corners[4], corners[5]], [corners[5], corners[6]], [corners[6], corners[7]], [corners[7], corners[4]],
        [corners[0], corners[4]], [corners[1], corners[5]], [corners[2], corners[6]], [corners[3], corners[7]]
    ]
    lines = Line3DCollection(edges, colors=color, linewidths=1.0)
    ax.add_collection3d(lines)

def draw_radar_fov_3d(ax, azi_deg, ele_deg, max_range, sensor_origin):
    h_min, h_max = np.deg2rad([-azi_deg/2, azi_deg/2])
    v_min, v_max = np.deg2rad([-ele_deg/2, ele_deg/2])
    corners_rel = []
    for h in [h_min, h_max]:
        for v in [v_min, v_max]:
            x_rel = max_range * np.cos(v) * np.cos(h)
            y_rel = max_range * np.cos(v) * np.sin(h)
            z_rel = max_range * np.sin(v)
            corners_rel.append(np.array([x_rel, y_rel, z_rel]))
    corners_abs = [sensor_origin + c for c in corners_rel]
    edges = [[sensor_origin, corners_abs[0]], [sensor_origin, corners_abs[1]],
             [sensor_origin, corners_abs[2]], [sensor_origin, corners_abs[3]],
             [corners_abs[0], corners_abs[1]], [corners_abs[0], corners_abs[2]],
             [corners_abs[1], corners_abs[3]], [corners_abs[2], corners_abs[3]]]
    lines = Line3DCollection(edges, colors='gray', linewidths=0.5, linestyles='--')
    ax.add_collection3d(lines)

def viz_bev_3d(out_png, pts_xyz, target_gt_list, params: ISACRadarParams, lidar_config: dict, sensor_origin=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, params.sensor_height_m])
    
    ax.plot([sensor_origin[0]], [sensor_origin[1]], [sensor_origin[2]], 'ko', markersize=5, label="Sensor Origin")
    
    arrow_prop_x = dict(mutation_scale=15, lw=1, arrowstyle='-|>', color='r')
    arrow_prop_y = dict(mutation_scale=15, lw=1, arrowstyle='-|>', color='g')
    arrow_prop_z = dict(mutation_scale=15, lw=1, arrowstyle='-|>', color='b')
    ax.add_artist(Arrow3D([sensor_origin[0], sensor_origin[0]+3], [sensor_origin[1], sensor_origin[1]], [sensor_origin[2], sensor_origin[2]], **arrow_prop_x))
    ax.text(sensor_origin[0]+3.5, sensor_origin[1], sensor_origin[2], "X (Forward)", color='r', fontsize=8)
    ax.add_artist(Arrow3D([sensor_origin[0], sensor_origin[0]], [sensor_origin[1], sensor_origin[1]+3], [sensor_origin[2], sensor_origin[2]], **arrow_prop_y))
    ax.text(sensor_origin[0], sensor_origin[1]+3.5, sensor_origin[2], "Y (Left)", color='g', fontsize=8)
    ax.add_artist(Arrow3D([sensor_origin[0], sensor_origin[0]], [sensor_origin[1], sensor_origin[1]], [sensor_origin[2], sensor_origin[2]+3], **arrow_prop_z))
    ax.text(sensor_origin[0], sensor_origin[1], sensor_origin[2]+3.5, "Z (Up)", color='b', fontsize=8)

    if pts_xyz is not None and pts_xyz.shape[0] > 0:
        N, max_pts = pts_xyz.shape[0], 20000
        indices = np.random.choice(N, min(N, max_pts), replace=False)
        pts_to_plot = pts_xyz[indices]
        ax.scatter(pts_to_plot[:,0], pts_to_plot[:,1], pts_to_plot[:,2], 
                   s=0.05, c=pts_to_plot[:,2], cmap='viridis', 
                   alpha=0.5, label="Scatterer Hits (if any)")
    
    if target_gt_list:
        for gt in target_gt_list:
            plot_cube_wireframe_3d(ax, gt, color='r')
            target_center = np.array(gt['center_xyz'])
            target_vel = np.array(gt['vel_xyz'])
            vec_to_target = target_center - sensor_origin
            range_m = np.linalg.norm(vec_to_target)
            
            if range_m > 0:
                ax.plot([sensor_origin[0], target_center[0]], 
                        [sensor_origin[1], target_center[1]], 
                        [sensor_origin[2], target_center[2]], 
                        '--', color='purple', linewidth=0.7)
                mid_point = sensor_origin + vec_to_target / 2.0
                ax.text(mid_point[0], mid_point[1], mid_point[2] + 1.0, 
                        f"R:{range_m:.1f}m", color='purple', fontsize=8, ha='center', va='bottom')
                
                unit_vec_to_target = vec_to_target / range_m
                radial_vel_scalar = np.dot(unit_vec_to_target, target_vel)
                
                if abs(radial_vel_scalar) > 0.1:
                    vel_vec_radial = radial_vel_scalar * unit_vec_to_target
                    arrow_len_factor = 0.3
                    draw_vec = vel_vec_radial * arrow_len_factor
                    ax.add_artist(Arrow3D(
                        [target_center[0], target_center[0] + draw_vec[0]],
                        [target_center[1], target_center[1] + draw_vec[1]],
                        [target_center[2], target_center[2] + draw_vec[2]],
                        mutation_scale=10, lw=1, arrowstyle='-|>', color='darkgreen'))
                    ax.text(target_center[0] + draw_vec[0] + 0.5, 
                            target_center[1] + draw_vec[1] + 0.5, 
                            target_center[2] + draw_vec[2] + 0.5,
                            f"V_r:{radial_vel_scalar:.1f}m/s", color='darkgreen', fontsize=8)

        draw_radar_fov_3d(ax, 
                          params.azi_beamwidth_deg, 
                          params.ele_beamwidth_deg, 
                          lidar_config.get('max_range', params.max_range_m), 
                          sensor_origin)
        
    ax.set_xlabel("X (m) [Forward]"); ax.set_ylabel("Y (m) [Left]"); ax.set_zlabel("Z (m) [Up]")
    ax.set_title("3D Bird's-Eye-View with Target Ground Truth and Radar FOV")
    
    max_plot_range = lidar_config.get('max_range', params.max_range_m) * 1.1
    ax.set_xlim([0, max_plot_range]); ax.set_ylim([-max_plot_range/2, max_plot_range/2])
    ax.set_zlim([0, params.sensor_height_m + max_plot_range/4])
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        ax.set_box_aspect([2, 1, 0.66]) # Approximate
    ax.view_init(elev=20, azim=-120)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# =========================================================================
# 4. Main Simulation and Visualization
# =========================================================================

def main():
    print("--- Starting ISAC Simulation (v5 - Merged) ---")
    
    # --- 1. Setup Parameters ---
    isac_params = ISACRadarParams(
        n_subcarriers=N_SUBCARRIERS, n_symbols=N_SYMBOLS,
        range_res_m=RANGE_RES_M, vel_res_mps=VEL_RES_MPS
    )
    # Dummy lidar config for 3D BEV plotting
    lidar_config_dummy = {"max_range": isac_params.unambiguous_range()}
    sensor_origin = np.array([0.0, 0.0, isac_params.sensor_height_m])
    
    print("\n--- System Parameters ---")
    print(f"Grid Size (Delay x Doppler): {N_SUBCARRIERS} x {N_SYMBOLS}")
    print(f"Bandwidth: {BANDWIDTH_HZ/1e6:.2f} MHz")
    print(f"==> Range Resolution: {isac_params.range_res():.2f} m")
    print(f"==> Velocity Resolution: {isac_params.velocity_res():.2f} m/s")
    print(f"==> Max Range: {isac_params.unambiguous_range():.2f} m")
    print(f"==> Max Velocity: +/- {isac_params.unambiguous_doppler_vel()/2.0:.2f} m/s")
    
    start_time = time.time()
    
    # --- 2. Generate Targets ---
    N_TARGETS = np.random.randint(1, MAX_TARGETS + 1)
    # Generate all 3 target list formats
    physical_targets_isac, target_bins_list_radar, target_gt_list_viz = \
        generate_physical_targets(N_TARGETS)
    
    print(f"\n--- Generated {N_TARGETS} Radar Target(s) (ON-GRID) ---")
    for i, t in enumerate(physical_targets_isac):
        range_m = np.sqrt(t['x']**2 + t['y']**2)
        print(f"  Target {i+1}:")
        print(f"    Location (x, y): ({t['x']:.2f} m, {t['y']:.2f} m)")
        print(f"    Range: {range_m:.2f} m, Velocity: {t['velocity']:.2f} m/s")
        print(f"    Mapped Bins (Delay, Doppler): {target_bins_list_radar[i][:2]}")

    # --- 3. Generate 3D BEV Plot ---
    print("\nGenerating 3D BEV plot...")
    bev_3d_png = os.path.join(SAVE_DIR, "bev_3d_scene.png")
    # We pass empty points `np.empty((0,3))` since we didn't run the LiDAR sim
    viz_bev_3d(bev_3d_png, np.empty((0,3)), target_gt_list_viz, 
               isac_params, lidar_config_dummy, sensor_origin)
    print(f"Saved 3D BEV plot to: {bev_3d_png}")

    # --- 4. Sensing Map Visualization (at a fixed SNR) ---
    print(f"\nGenerating sensing maps at {SIM_SNR_DB} dB SNR...")
    
    # --- Run OFDM Sensing ---
    tx_symbols_ofdm = get_qam_symbols(N_SUBCARRIERS * N_SYMBOLS)
    tx_grid_ofdm = tx_symbols_ofdm.reshape((N_SUBCARRIERS, N_SYMBOLS))
    tx_signal_ofdm = ofdm_modulate(tx_grid_ofdm)
    rx_signal_ofdm = apply_radar_channel_and_noise(tx_signal_ofdm, SIM_SNR_DB, 
                                                   physical_targets_isac, FS_HZ)
    rdm_ofdm = ofdm_sensing_receiver(rx_signal_ofdm, tx_signal_ofdm)
    rdm_ofdm_mag_db = 10 * np.log10(np.abs(rdm_ofdm))

    # --- Run OTFS Sensing (using CORRECTED transforms) ---
    dd_grid_pilot = np.zeros((N_SUBCARRIERS, N_SYMBOLS), dtype=complex)
    dd_grid_pilot[0, 0] = 1.0 # Single pilot at (0,0)
    tx_signal_otfs = otfs_modulate_v2(dd_grid_pilot) # Use v2
    rx_signal_otfs = apply_radar_channel_and_noise(tx_signal_otfs, SIM_SNR_DB, 
                                                   physical_targets_isac, FS_HZ)
    ddm_otfs = otfs_demodulate_v2(rx_signal_otfs) # Use v2
    ddm_otfs_mag_db = 10 * np.log10(np.abs(np.fft.fftshift(ddm_otfs)))
 
    # --- Plot Sensing Maps (using new viz functions) ---
    ofdm_rd_png = os.path.join(SAVE_DIR, "ofdm_rd_map.png")
    ofdm_rd_3d_png = os.path.join(SAVE_DIR, "ofdm_rd_map_3d.png")
    otfs_rd_png = os.path.join(SAVE_DIR, "otfs_rd_map.png")
    otfs_rd_3d_png = os.path.join(SAVE_DIR, "otfs_rd_map_3d.png")
    
    print("  Visualizing OFDM RD map (2D)...")
    viz_rd(rdm_ofdm_mag_db, ofdm_rd_png, isac_params, 
           target_gt_list_viz, sensor_origin)
    print("  Visualizing OFDM RD map (3D)...")
    viz_rd_3d(rdm_ofdm_mag_db, ofdm_rd_3d_png, isac_params, 
              target_gt_list_viz, sensor_origin)
    
    print("  Visualizing OTFS RD map (2D)...")
    viz_rd(ddm_otfs_mag_db, otfs_rd_png, isac_params, 
           target_gt_list_viz, sensor_origin)
    print("  Visualizing OTFS RD map (3D)...")
    viz_rd_3d(ddm_otfs_mag_db, otfs_rd_3d_png, isac_params, 
              target_gt_list_viz, sensor_origin)
    
    print(f"Saved all sensing maps to: {SAVE_DIR}")
 
    # --- 5. Communication BER vs. SNR Simulation ---
    print(f"\nRunning BER vs. SNR simulation (Comm Channel)...")
    print(f"Using separate 'on-grid' comm channel: {COMM_CHANNEL_BINS}")
    ber_ofdm = []
    ber_otfs = []
    
    H_ideal_dd_comm = get_otfs_dd_channel(COMM_CHANNEL_BINS)
    H_ideal_tf_ofdm_comm = get_ofdm_tf_channel(COMM_CHANNEL_BINS)
 
    for snr_db in SNR_DB_LIST:
        print(f"  Simulating at {snr_db} dB...")
        
        tx_bits = np.random.randint(0, QAM_ORDER, N_SUBCARRIERS * N_SYMBOLS)
        tx_symbols = np.array([MOD_MAP[b] for b in tx_bits])
        tx_data_grid = tx_symbols.reshape((N_SUBCARRIERS, N_SYMBOLS))
        
        # --- OTFS BER (Simplified DD-Domain Simulation) ---
        tx_tf = np.fft.fft2(tx_data_grid)
        h_tf = np.fft.fft2(H_ideal_dd_comm)
        rx_dd_noiseless = np.fft.ifft2(tx_tf * h_tf)
        
        signal_power = np.mean(np.abs(rx_dd_noiseless)**2)
        if signal_power == 0: signal_power = 1e-20
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_grid = (np.random.randn(N_SUBCARRIERS, N_SYMBOLS) + 
                      1j*np.random.randn(N_SUBCARRIERS, N_SYMBOLS)) * \
                     np.sqrt(noise_power / 2)
        rx_dd_noisy = rx_dd_noiseless + noise_grid
        
        Y_tf = np.fft.fft2(rx_dd_noisy)
        H_tf_reg = np.fft.fft2(H_ideal_dd_comm)
        H_tf_reg[np.abs(H_tf_reg) < 1e-6] = 1e-6
        X_est_tf = Y_tf / H_tf_reg
        rx_symbols_otfs = np.fft.ifft2(X_est_tf).flatten()
        
        rx_bits_otfs = demodulate_qam_symbols(rx_symbols_otfs)
        ber_otfs.append(calculate_ber(tx_bits, rx_bits_otfs))
 
        # --- OFDM BER (Time-Domain Simulation) ---
        tx_signal_ofdm = ofdm_modulate(tx_data_grid)
        rx_signal_ofdm = apply_comm_channel_and_noise(tx_signal_ofdm, snr_db, 
                                                      COMM_CHANNEL_BINS, FS_HZ)
        rx_grid_time = rx_signal_ofdm.reshape((N_SUBCARRIERS + CP_LEN, N_SYMBOLS), order='F')
        rx_grid_no_cp = rx_grid_time[CP_LEN:, :]
        rx_tf_grid_ofdm = np.fft.fft(rx_grid_no_cp, axis=0)
        
        H_ideal_tf_ofdm_comm_reg = H_ideal_tf_ofdm_comm.copy()
        H_ideal_tf_ofdm_comm_reg[np.abs(H_ideal_tf_ofdm_comm_reg) < 1e-6] = 1e-6
        rx_symbols_ofdm_est_grid = rx_tf_grid_ofdm / H_ideal_tf_ofdm_comm_reg
        rx_symbols_ofdm_est = rx_symbols_ofdm_est_grid.flatten()
        
        rx_bits_ofdm = demodulate_qam_symbols(rx_symbols_ofdm_est)
        ber_ofdm.append(calculate_ber(tx_bits, rx_bits_ofdm))
 
    # --- Plot BER Curves ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_DB_LIST, ber_otfs, '-o', linewidth=2, markersize=8, 
                 label='OTFS (with 2D Deconvolution Equalizer)')
    plt.semilogy(SNR_DB_LIST, ber_ofdm, '-s', linewidth=2, markersize=8,
                 label='OFDM (with 1-Tap TF Equalizer)')
    plt.title('Communication Performance (BER vs. SNR) in High-Mobility')
    plt.xlabel('SNR (dB)'); plt.ylabel('Bit Error Rate (BER)')
    plt.legend(); plt.grid(True, which='both'); plt.ylim(1e-4, 1.0)
    
    ber_filename = os.path.join(SAVE_DIR, "ber_comparison.png")
    plt.savefig(ber_filename)
    print(f"Saved BER plot to: {ber_filename}")

    # --- Finished ---
    end_time = time.time()
    print(f"\n--- Simulation Complete in {end_time - start_time:.2f} seconds ---")
    print(f"All figures saved to '{SAVE_DIR}' directory.")
    
    # Show all plots at the end
    # plt.show() # Commented out to run cleanly on servers

# --- Run the main simulation ---
if __name__ == "__main__":
    main()