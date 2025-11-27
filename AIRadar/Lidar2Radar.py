"""
LiDAR -> Radar Dataset Generator (Point-Cloud Ray Tracing Approximation)
=======================================================================

This tool converts LiDAR point clouds (KITTI, NuScenes, Waymo) or synthetic
scenes into simulated radar radio returns using a physically motivated,
computationally efficient model.

Highlights
----------
- NEW: Fixed `rcs_from_intensity` logic to correctly plot the rcs.png.
- NEW: `viz_rd_3d` now plots 3D Ground Truth markers.
- NEW: `run_dataset_demo` now prints radial velocities to explain
         the MTI "blind spot" (i.e., why a target might be missing).
- MTI filter (2-pulse canceler) correctly removes static clutter.
- 3D BEV plot draws coordinate axes, range/velocity vectors, and FOV.
- Sensor height is parameterized, and the ground is at Z=0.
"""

import os, json, math, glob, argparse, warnings, time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# -----------------------------
# Physics & Radar params
# -----------------------------
C0 = 299792458.0

@dataclass
class RadarParams:
    fc: float = 77e9            # carrier [Hz]
    fs: float = 10e6            # ADC sampling rate [Hz]
    T_chirp: float = 50e-6      # chirp duration [s]
    M_chirp: int = 64           # number of chirps (slow-time)
    slope: float = 60e12        # chirp slope k [Hz/s]
    noise_tempK: float = 290.0  # thermal temperature [K]
    noise_figure_dB: float = 5.0 # receiver NF [dB]
    cfo_Hz: float = 0.0         # carrier frequency offset [Hz]
    phase_std: float = 0.0      # per-sample Wiener PN std [rad]
    ground_bounce: bool = True  # add mirror across z=0 plane
    azi_beamwidth_deg: float = 30.0 # Azimuth FOV for raycasting/viz
    ele_beamwidth_deg: float = 10.0 # Elevation FOV for raycasting/viz
    max_range_m: float = 150.0  # used for clipping/sanity
    sensor_height_m: float = 1.8 # Sensor height above ground (Z=0)
    # Derived:
    def lambda_m(self): return C0 / self.fc
    def Ns_fast(self): return int(round(self.fs * self.T_chirp))
    def unambiguous_range(self): return (self.fs * C0) / (2.0 * self.slope)
    def unambiguous_doppler_vel(self): return self.lambda_m() / (2 * self.T_chirp)


# -----------------------------
# Utility: RCS mapping
# -----------------------------
RCS_TABLE_DBSM = {
    "car": 10.0, "truck": 15.0, "bus": 15.0, "motorcycle": -5.0,
    "pedestrian": -10.0, "bicycle": -15.0, "vegetation": -20.0, "background": -25.0
}

# <--- FIX: Corrected logic for RCS plot --->
def rcs_from_intensity(intensity, label=None):
    if intensity is None: intensity = 0.5
    
    # Check for synthetic flags *first*
    if intensity == 255.0: # Target
        return 10.0**(20.0/10.0) # 20 dBsm
    if intensity == 100.0: # Ground
        return 10.0**(-15.0/10.0) # -15 dBsm

    # Fallback for real LiDAR data (0..1 or 0..255)
    intensity_norm = intensity
    if not isinstance(intensity, (float, int)): # Check if it's array-like
        if intensity.dtype != np.float32 and intensity.dtype != np.float64:
            intensity_norm = intensity.astype(np.float32) / 255.0
    elif intensity > 1.0: # Is a float, but > 1.0 (so 0-255 range)
        intensity_norm = intensity / 255.0
        
    base_db = -25.0 + 35.0 * float(np.clip(intensity_norm, 0.0, 1.0))
    if label is not None and label in RCS_TABLE_DBSM:
        base_db = 0.5*base_db + 0.5*RCS_TABLE_DBSM[label]
    return 10.0**(base_db/10.0)

# -----------------------------
# Occlusion & transforms
# -----------------------------
def occlusion_mask(points_xyz, beam_azi_deg=3.0, beam_ele_deg=10.0):
    x, y, z = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]
    rng = np.maximum(1e-3, np.sqrt(x*x + y*y + z*z))
    azi = np.degrees(np.arctan2(y, x))
    ele = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))

    da = max(1.0, beam_azi_deg/2)
    de = max(2.0, beam_ele_deg/2)
    a_bin = np.floor((azi + 180.0)/da).astype(np.int32)
    e_bin = np.floor((ele + 90.0)/de).astype(np.int32)
    key = a_bin * 10000 + e_bin
    order = np.argsort(key)
    mask = np.ones(len(points_xyz), dtype=bool)
    i = 0
    while i < len(points_xyz):
        j = i
        K = key[order[i]]
        while j < len(points_xyz) and key[order[j]] == K:
            j += 1
        idx = order[i:j]
        sub_rng = rng[idx]
        keep = np.argmin(sub_rng)
        mask[idx] = False
        mask[idx[keep]] = True
        i = j
    return mask

def apply_extrinsics(points_xyz, T_radar_lidar):
    if T_radar_lidar is None:
        return points_xyz.copy()
    P = np.concatenate([points_xyz, np.ones((points_xyz.shape[0],1))], axis=1).T
    Q = T_radar_lidar @ P
    return Q[:3,:].T

# -----------------------------
# Radar synthesis
# -----------------------------
def synthesize_fmcw_iq_from_points(points_xyz, intensities, params: RadarParams,
                                   ego_vel_xyz=np.array([0.0,0.0,0.0]),
                                   obj_vel_xyz=None, sensor_origin=np.array([0.0,0.0,0.0])):
    Ns = params.Ns_fast()
    M  = params.M_chirp
    fs = params.fs
    lam = params.lambda_m()

    if points_xyz.shape[0] == 0:
        return np.zeros((M, Ns), dtype=np.complex64), np.zeros((M, Ns)), {}

    points_xyz_relative = points_xyz - sensor_origin
    
    if obj_vel_xyz is None:
        vis = occlusion_mask(points_xyz_relative, params.azi_beamwidth_deg, params.ele_beamwidth_deg)
        P_relative = points_xyz_relative[vis]
        I = intensities[vis] if intensities is not None and len(intensities)==len(points_xyz) else np.ones(P_relative.shape[0])
        v_obj = np.zeros_like(P_relative)
    else:
        P_relative = points_xyz_relative
        I = intensities
        v_obj = obj_vel_xyz

    R = np.linalg.norm(P_relative, axis=1)
    valid_range = R > 1e-3
    P_relative = P_relative[valid_range]
    I = I[valid_range]
    v_obj = v_obj[valid_range]
    R = R[valid_range]
    
    if P_relative.shape[0] == 0:
        return np.zeros((M, Ns), dtype=np.complex64), np.zeros((M, Ns)), {}

    u = P_relative / R[:,None]
    
    v_rel = v_obj - ego_vel_xyz.reshape(1, 3) 
    v_r = np.sum(u * v_rel, axis=1)
    
    rcs = np.array([rcs_from_intensity(i) for i in I])
    A = np.sqrt(rcs) / np.maximum(1.0, R)**2

    fb = 2.0 * params.slope * R / C0
    fd = 2.0 * v_r / lam

    n = np.arange(Ns) / fs
    m = np.arange(M) * params.T_chirp

    phase_fast = 2.0*np.pi * (fb[:,None,None] * n[None,None,:])
    phase_slow = 2.0*np.pi * (fd[:,None,None] * m[None,:,None])
    phi0 = 2.0*np.pi * np.random.rand(len(A))[:,None,None]
    ph = phase_fast + phase_slow + phi0

    contrib = (A[:,None,None] * np.exp(1j*ph)).astype(np.complex64)
    iq = np.sum(contrib, axis=0)

    if abs(params.cfo_Hz) > 0 or params.phase_std > 0:
        tgrid = (m[:,None] + n[None,:])
        ph_cfo = 2.0*np.pi*params.cfo_Hz * tgrid
        ph_pn = 0.0
        if params.phase_std > 0:
            inc = np.random.randn(M, Ns) * params.phase_std
            ph_pn = np.cumsum(inc, axis=1)
        iq *= np.exp(1j*(ph_cfo + ph_pn)).astype(np.complex64)

    kB = 1.38064852e-23
    N0 = kB * params.noise_tempK * params.fs * (10**(params.noise_figure_dB/10.0))
    noise = np.sqrt(N0/2.0) * (np.random.randn(M, Ns) + 1j*np.random.randn(M, Ns))
    iq_noisy = iq + noise.astype(np.complex64)

    meta = {
        "num_scatterers": int(P_relative.shape[0]),
        "ranges_mean_m": float(np.mean(R)) if len(R)>0 else 0.0,
        "rcs_mean": float(np.mean(rcs)) if len(rcs)>0 else 0.0
    }
    return iq_noisy, iq, meta

def rd_map(iq):
    """
    Performs 2D FFT Range-Doppler processing with a 2-pulse canceler MTI.
    """
    M, Ns = iq.shape
    mat = iq.copy()
    
    win_fast = np.hanning(Ns)
    mat = mat * win_fast[None, :]
    
    range_fft = np.fft.fft(mat, axis=1)
    
    mti_shifted = np.roll(range_fft, 1, axis=0)
    range_fft_mti = range_fft - mti_shifted
    range_fft_mti[0, :] = 0
    
    win_slow = np.hanning(M)
    range_fft_mti = range_fft_mti * win_slow[:, None]
    
    rd_mat = np.fft.fft(range_fft_mti, axis=0)
    
    RD = np.fft.fftshift(rd_mat, axes=(0,))
    
    return 20*np.log10(np.abs(RD) + 1e-6)

# -----------------------------
# Dataset adapters (lightweight)
# -----------------------------
class LidarFrame:
    def __init__(self, points_xyz, intensity=None, timestamp=None, 
                 T_radar_lidar=None, ego_vel_xyz=None, target_boxes=None, meta=None):
        self.points_xyz = points_xyz
        self.intensity = intensity
        self.timestamp = timestamp
        self.T_radar_lidar = T_radar_lidar
        self.ego_vel_xyz = ego_vel_xyz if ego_vel_xyz is not None else np.zeros(3)
        self.target_boxes = target_boxes if target_boxes is not None else []
        self.meta = meta or {}

# --- KITTI Helper Functions ---
def read_calib_file(filepath):
    calib = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, *values = line.split(' ')
            key = key.strip(':')
            if key in ['P2', 'R0_rect', 'Tr_velo_to_cam']:
                if key == 'P2':
                    calib[key] = np.array([float(x) for x in values]).reshape(3, 4)
                elif key == 'R0_rect':
                    calib[key] = np.array([float(x) for x in values]).reshape(3, 3)
                elif key == 'Tr_velo_to_cam':
                    calib[key] = np.array([float(x) for x in values]).reshape(3, 4)

    R0 = np.eye(4)
    R0[:3, :3] = calib['R0_rect']
    calib['R0_rect_4x4'] = R0
    
    Tr_v2c = np.eye(4)
    Tr_v2c[:3, :] = calib['Tr_velo_to_cam']
    calib['Tr_velo_to_cam_4x4'] = Tr_v2c
    
    return calib

def read_label_file(filepath):
    objects = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            data = line.split(' ')
            if len(data) < 15:
                continue
            
            obj = {}
            obj['type'] = data[0]
            if obj['type'] not in ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck']:
                continue
            
            obj['dimensions'] = [float(data[8]), float(data[9]), float(data[10])] # h, w, l
            obj['location'] = [float(data[11]), float(data[12]), float(data[13])] # x, y, z in camera
            obj['rotation_y'] = float(data[14])
            objects.append(obj)
    return objects

def read_oxts_file(filepath):
    with open(filepath, 'r') as f:
        line = f.readline().split(' ')
        vf = float(line[8]) # forward
        vl = float(line[9]) # left
        vu = float(line[10]) # up
        return np.array([vf, vl, vu])

class KittiAdapter:
    def __init__(self, root, seq="0000", T_radar_lidar=None):
        self.root = root
        self.seq = seq
        self.T_radar_lidar = T_radar_lidar
        
        self.bin_dir = os.path.join(root, "velodyne", seq)
        self.label_dir = os.path.join(root, "label_2", seq)
        self.calib_dir = os.path.join(root, "calib", seq)
        self.oxts_dir = os.path.join(root, "oxts", seq)
        
        bin_stems = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(self.bin_dir, "*.bin"))}
        label_stems = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(self.label_dir, "*.txt"))}
        calib_stems = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(self.calib_dir, "*.txt"))}
        oxts_stems = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(self.oxts_dir, "*.txt"))}
        
        common_stems = sorted(list(bin_stems & label_stems & calib_stems & oxts_stems))
        
        self.files = [os.path.join(self.bin_dir, f"{s}.bin") for s in common_stems]
        self.label_files = [os.path.join(self.label_dir, f"{s}.txt") for s in common_stems]
        self.calib_files = [os.path.join(self.calib_dir, f"{s}.txt") for s in common_stems]
        self.oxts_files = [os.path.join(self.oxts_dir, f"{s}.txt") for s in common_stems]
        
        if len(self.files) == 0:
            warnings.warn(f"No common frames found for KITTI seq {seq} in {root}.")
        else:
             print(f"Found {len(self.files)} common frames for KITTI seq {seq}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts_file = self.files[idx]
        pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, 0:3]
        intensity = pts[:, 3]
        
        calib = read_calib_file(self.calib_files[idx])
        Tr_v2c_4x4 = calib['Tr_velo_to_cam_4x4']
        R0_rect_4x4 = calib['R0_rect_4x4']
        
        Tr_c2v_4x4 = np.linalg.inv(R0_rect_4x4 @ Tr_v2c_4x4)
        
        raw_labels = read_label_file(self.label_files[idx])
        target_boxes = []
        for obj in raw_labels:
            loc_cam_homo = np.array(obj['location'] + [1.0])
            loc_velo = (Tr_c2v_4x4 @ loc_cam_homo)[:3]
            
            box_gt = {
                "type": obj['type'],
                "center_xyz_lidar": loc_velo.tolist(),
                "dimensions_hwl": obj['dimensions'],
                "rotation_y_cam": obj['rotation_y'],
                "vel_xyz": [0.0, 0.0, 0.0]
            }
            target_boxes.append(box_gt)

        ego_vel_xyz = read_oxts_file(self.oxts_files[idx])
        
        return LidarFrame(points_xyz=xyz, 
                          intensity=intensity, 
                          T_radar_lidar=self.T_radar_lidar,
                          ego_vel_xyz=ego_vel_xyz,
                          target_boxes=target_boxes,
                          meta={"frame_id": os.path.basename(pts_file)})

class NuScenesAdapter:
    def __init__(self, root, version='v1.0-mini', T_radar_lidar=None):
        self.root = root
        self.version = version
        self.T_radar_lidar = T_radar_lidar
        self.samples = []
        try:
            print("NuScenesAdapter: Devkit not imported. Using placeholder structure.")
            if version == 'v1.0-mini':
                self.samples = [i for i in range(404)]
        except ImportError:
            warnings.warn("nuscenes-devkit not found. NuScenesAdapter will not work.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        warnings.warn(f"NuScenesAdapter: Returning placeholder data for sample {idx}.")
        return LidarFrame(points_xyz=np.random.rand(20000, 3) * 50 - 25,
                          intensity=np.random.rand(20000),
                          ego_vel_xyz=np.array([5.0, 0.0, 0.0]),
                          target_boxes=[])

class WaymoAdapter:
    def __init__(self, tfrecord_paths, T_radar_lidar=None):
        self.tfrecord_paths = tfrecord_paths
        self.T_radar_lidar = T_radar_lidar
        self.frames = []
        try:
            print("WaymoAdapter: TF/Waymo devkit not imported. Using placeholder structure.")
            self.frames = [i for i in range(100)]
        except ImportError:
            warnings.warn("tensorflow or waymo-open-dataset not found. WaymoAdapter will not work.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        warnings.warn(f"WaymoAdapter: Returning placeholder data for sample {idx}.")
        return LidarFrame(points_xyz=np.random.rand(50000, 3) * 70 - 35,
                          intensity=np.random.rand(50000),
                          ego_vel_xyz=np.array([10.0, 0.1, 0.0]),
                          target_boxes=[])
                          
# -----------------------------
# Visualization
# -----------------------------
def get_target_rd_coords(target_gt_list, params: RadarParams, sensor_origin=np.array([0.0,0.0,0.0])):
    """ Helper to convert ground truth boxes to RD coordinates. """
    gt_coords = []
    R_unamb = params.unambiguous_range()
    v_unamb = params.unambiguous_doppler_vel()

    for gt in target_gt_list:
        if 'center_xyz' in gt and 'vel_xyz' in gt:
            p_abs = np.array(gt['center_xyz'])
            v_abs = np.array(gt['vel_xyz'])
        elif 'center_xyz_lidar' in gt:
            p_abs = np.array(gt['center_xyz_lidar'])
            v_abs = np.array(gt.get('vel_xyz', [0,0,0]))
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

def viz_rd(RD_dB, out_png, params: RadarParams, target_gt_list=None, sensor_origin=np.array([0.0,0.0,0.0])):
    M, Ns = RD_dB.shape
    
    R_unamb = params.unambiguous_range()
    range_bins = np.linspace(0, R_unamb, Ns)
    
    v_unamb = params.unambiguous_doppler_vel()
    doppler_bins = np.linspace(-v_unamb/2, v_unamb/2, M)

    plt.figure(figsize=(12, 6))
    
    v_max = np.max(RD_dB)
    v_min = -120 
    
    plt.pcolormesh(range_bins, doppler_bins, RD_dB, shading='auto', vmin=v_min, vmax=v_max)
    
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
    plt.ylabel(f"Velocity (m/s) [Max: {v_unamb/2:.1f} m/s]")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def viz_rd_3d(RD_dB, out_png, params: RadarParams, target_gt_list=None, sensor_origin=np.array([0.0,0.0,0.0])):
    M, Ns = RD_dB.shape
    R_unamb = params.unambiguous_range()
    range_axis = np.linspace(0, R_unamb, Ns)
    v_unamb = params.unambiguous_doppler_vel()
    doppler_axis = np.linspace(-v_unamb/2, v_unamb/2, M)
    R_grid, V_grid = np.meshgrid(range_axis, doppler_axis)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    v_max = np.max(RD_dB)
    v_min = -120 
    
    Z_surface = np.maximum(RD_dB, v_min)
    ax.plot_surface(R_grid, V_grid, Z_surface, cmap='viridis', 
                    rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.8)
    
    # <--- NEW: Add Ground Truth to 3D plot --->
    gt_coords = get_target_rd_coords(target_gt_list, params, sensor_origin)
    if len(gt_coords) > 0:
        gt_ranges = [gc['range'] for gc in gt_coords]
        gt_vels = [gc['velocity'] for gc in gt_coords]
        
        # Plot markers just above the surface (or at a fixed high Z)
        gt_z = [v_max + 10 for _ in gt_ranges]
        
        ax.scatter(gt_ranges, gt_vels, gt_z, color='red', marker='x', s=100, linewidth=2, label='Ground Truth')
        
        # Draw lines down to the surface
        for r, v, z in zip(gt_ranges, gt_vels, gt_z):
            z_surface_val = Z_surface[np.argmin(np.abs(doppler_axis - v)), np.argmin(np.abs(range_axis - r))]
            ax.plot([r, r], [v, v], [z, z_surface_val], 'r--', linewidth=0.8)

        ax.legend()

    ax.set_title("3D Range-Doppler Map (Clipped at -120dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_zlabel("Power (dB)")
    ax.view_init(elev=30, azim=-135)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def viz_range_profile(RD_dB, out_png, params: RadarParams):
    prof = np.max(RD_dB, axis=0)
    R_unamb = params.unambiguous_range()
    range_axis = np.linspace(0, R_unamb, len(prof))
    plt.figure()
    plt.plot(range_axis, prof)
    plt.title("Range Profile (max over Doppler)")
    plt.xlabel(f"Range (m) [Max: {R_unamb:.1f} m]")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def viz_bev(points_xyz, out_png, peak_bins=None, params: RadarParams = None):
    # This is the 2D BEV plot
    plt.figure(figsize=(8, 8))
    if points_xyz.shape[0] > 0:
        plt.scatter(points_xyz[:,1], points_xyz[:,0], s=0.5, c=points_xyz[:,2], cmap='viridis', label="Scatterers")
        plt.colorbar(label="Z (height, m)")
        
    if peak_bins is not None and params is not None:
        R_unamb = params.unambiguous_range()
        Ns = params.Ns_fast()
        for rb in peak_bins:
            r = (float(rb) / Ns) * R_unamb
            theta = np.linspace(0, 2*np.pi, 360)
            x_ring = r * np.cos(theta)
            y_ring = r * np.sin(theta)
            plt.plot(y_ring, x_ring, 'r--', linewidth=0.5)
            
    plt.gca().set_aspect('equal', 'box')
    plt.title("2D Bird's-Eye View (BEV)")
    plt.xlabel("Y (m) [Left]")
    plt.ylabel("X (m) [Forward]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def viz_intensity_rcs(intensity, rcs, out_png):
    if intensity is None or len(intensity) == 0:
        return
    
    # Separate ground and target points
    ground_mask = (intensity == 100.0)
    target_mask = (intensity == 255.0)

    plt.figure()
    if np.any(ground_mask):
        plt.scatter(intensity[ground_mask], 10*np.log10(rcs[ground_mask]+1e-12), s=1, label=f"Ground (Intensity=100)")
    if np.any(target_mask):
        plt.scatter(intensity[target_mask], 10*np.log10(rcs[target_mask]+1e-12), s=1, label=f"Target (Intensity=255)")
    
    plt.xlabel("LiDAR intensity (raw)")
    plt.ylabel("Assigned RCS [dBsm]")
    plt.title("RCS vs LiDAR intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# --- IMPROVED 3D BEV VISUALIZATION ---
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        if renderer:
            return np.min(zs)
        else:
            return 0

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
             [sensor_origin, corners_abs[2]], [sensor_origin, corners_abs[3]]]
    
    edges.extend([[corners_abs[0], corners_abs[1]], [corners_abs[0], corners_abs[2]],
                  [corners_abs[1], corners_abs[3]], [corners_abs[2], corners_abs[3]]])
                  
    lines = Line3DCollection(edges, colors='gray', linewidths=0.5, linestyles='--')
    ax.add_collection3d(lines)

def viz_bev_3d(out_png, pts_xyz, target_gt_list, params: RadarParams, lidar_config: dict, sensor_origin=None):
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

    N = pts_xyz.shape[0]
    max_pts = 20000
    if N > max_pts:
        indices = np.random.choice(N, max_pts, replace=False)
        pts_to_plot = pts_xyz[indices]
    else:
        pts_to_plot = pts_xyz
        
    if pts_to_plot.shape[0] > 0:
        ax.scatter(pts_to_plot[:,0], pts_to_plot[:,1], pts_to_plot[:,2], 
                   s=0.05, c=pts_to_plot[:,2], cmap='viridis', 
                   alpha=0.5, label="LiDAR Hits")
    
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
        
    ax.set_xlabel("X (m) [Forward]")
    ax.set_ylabel("Y (m) [Left]")
    ax.set_zlabel("Z (m) [Up]")
    ax.set_title("3D Bird's-Eye-View with Target Ground Truth and Radar FOV")
    
    max_plot_range = lidar_config.get('max_range', params.max_range_m) * 1.1
    ax.set_xlim([0, max_plot_range])
    ax.set_ylim([-max_plot_range/2, max_plot_range/2])
    ax.set_zlim([0, params.sensor_height_m + max_plot_range/4]) # Ground at Z=0
    
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        ax.set_box_aspect([2, 1, 0.66]) # Approximate equal aspect

    ax.view_init(elev=20, azim=-120)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# -----------------------------
# Saver
# -----------------------------
def save_sample(outdir, frame_idx, iq, rd, meta):
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f"sample_{frame_idx:06d}.npz"),
                        iq=iq.astype(np.complex64),
                        rd=rd.astype(np.float32))
    with open(os.path.join(outdir, f"sample_{frame_idx:06d}.json"), "w") as f:
        meta_serializable = meta.copy()
        if "radar_params" in meta_serializable:
            meta_serializable["radar_params"] = meta["radar_params"].__dict__
        json.dump(meta_serializable, f, indent=2, default=lambda o: '<not serializable>')

# -----------------------------
# Synthetic Raycast Scene Generation
# -----------------------------
def intersect_ray_aabb(origin, direction, cube_gt):
    center = np.array(cube_gt['center_xyz'])
    size = np.array(cube_gt['size_xyz'])
    min_bound = center - size / 2.0
    max_bound = center + size / 2.0
    
    t_min = (min_bound - origin) / (direction + 1e-6)
    t_max = (max_bound - origin) / (direction + 1e-6)
    
    t_enter = np.max(np.minimum(t_min, t_max))
    t_exit = np.min(np.maximum(t_min, t_max))
    
    if (t_enter < t_exit) and (t_exit > 0):
        return True, max(0.0, t_enter)
    return False, np.inf

def generate_raycast_lidar(lidar_config, target_gt_list, rng, sensor_origin=np.array([0.0, 0.0, 0.0])):
    v_fov_deg = lidar_config.get("v_fov_deg", 30.0)
    h_fov_deg = lidar_config.get("h_fov_deg", 180.0)
    v_res_beams = lidar_config.get("v_res_beams", 32)
    h_res_steps = lidar_config.get("h_res_steps", 512)
    max_range = lidar_config.get("max_range", 100.0)
    
    ground_plane_z = 0.0 
    
    vertical_angles = np.linspace(-v_fov_deg/2, v_fov_deg/2, v_res_beams)
    horizontal_angles = np.linspace(-h_fov_deg/2, h_fov_deg/2, h_res_steps)
    
    hit_points = []
    hit_intensities = []
    hit_velocities = []
    
    for h_ang in horizontal_angles:
        for v_ang in vertical_angles:
            az = np.deg2rad(h_ang)
            el = np.deg2rad(v_ang)
            
            ray_dir = np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az), # Y-Left
                np.sin(el)
            ])
            
            min_dist = max_range
            hit_vel = np.array([0.0, 0.0, 0.0])
            hit_int = 0.0
            
            if abs(ray_dir[2]) > 1e-6:
                t_ground = (ground_plane_z - sensor_origin[2]) / ray_dir[2]
                if t_ground > 0 and t_ground < min_dist:
                    min_dist = t_ground
                    hit_vel = np.array([0.0, 0.0, 0.0])
                    hit_int = 100.0 # Intensity for ground points
                    
            for gt in target_gt_list:
                is_hit, dist = intersect_ray_aabb(sensor_origin, ray_dir, gt)
                if is_hit and dist < min_dist:
                    min_dist = dist
                    hit_vel = np.array(gt['vel_xyz'])
                    hit_int = 255.0 # Intensity for target points
                    
            if hit_int > 0 and min_dist < max_range:
                hit_point = sensor_origin + min_dist * ray_dir
                hit_points.append(hit_point)
                hit_intensities.append(hit_int)
                hit_velocities.append(hit_vel)
                
    if not hit_points:
        return np.zeros((0,3)), np.zeros(0), np.zeros((0,3))
        
    return (np.array(hit_points).astype(np.float32), 
            np.array(hit_intensities).astype(np.float32), 
            np.array(hit_velocities).astype(np.float32))

# -----------------------------
# Synthetic Dataset Class
# -----------------------------
class SyntheticRadarDataset:
    def __init__(self, num_samples, params: RadarParams, 
                 lidar_config, target_config, max_targets=3, seed=42):
        self.num_samples = num_samples
        self.params = params
        self.lidar_config = lidar_config
        self.target_config = target_config
        self.max_targets = max(1, max_targets)
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        
        num_targets = rng.integers(1, self.max_targets + 1)
        target_gt_list = []
        tc = self.target_config
        
        sensor_origin = np.array([0.0, 0.0, self.params.sensor_height_m])

        # Generate the first target to ensure a visible one
        center_1 = [
            rng.uniform(tc['pos_x'][0], tc['pos_x'][1]),
            rng.uniform(tc['pos_y'][0], tc['pos_y'][1]),
            rng.uniform(tc['pos_z'][0], tc['pos_z'][1])
        ]
        size_1 = [
            rng.uniform(tc['size_x'][0], tc['size_x'][1]),
            rng.uniform(tc['size_y'][0], tc['size_y'][1]),
            rng.uniform(tc['size_z'][0], tc['size_z'][1])
        ]
        # Ensure first target has high radial velocity
        vel_1 = [
            rng.uniform(tc['high_vel_x'][0], tc['high_vel_x'][1]),
            rng.uniform(tc['vel_y'][0], tc['vel_y'][1]),
            rng.uniform(tc['vel_z'][0], tc['vel_z'][1])
        ]
        target_gt_list.append({
            'center_xyz': center_1, 'size_xyz': size_1, 'vel_xyz': vel_1
        })

        # Generate additional targets (up to num_targets-1)
        for _ in range(num_targets - 1):
            center = [
                rng.uniform(tc['pos_x'][0], tc['pos_x'][1]),
                rng.uniform(tc['pos_y'][0], tc['pos_y'][1]),
                rng.uniform(tc['pos_z'][0], tc['pos_z'][1])
            ]
            size = [
                rng.uniform(tc['size_x'][0], tc['size_x'][1]),
                rng.uniform(tc['size_y'][0], tc['size_y'][1]),
                rng.uniform(tc['size_z'][0], tc['size_z'][1])
            ]
            # Randomly pick between high and low radial velocity for other targets
            if rng.random() < 0.5: # 50% chance of slow-moving target (potential MTI blind spot)
                vel = [
                    rng.uniform(tc['low_vel_x'][0], tc['low_vel_x'][1]),
                    rng.uniform(tc['vel_y'][0], tc['vel_y'][1]),
                    rng.uniform(tc['vel_z'][0], tc['vel_z'][1])
                ]
            else: # 50% chance of fast-moving target
                vel = [
                    rng.uniform(tc['high_vel_x'][0], tc['high_vel_x'][1]),
                    rng.uniform(tc['vel_y'][0], tc['vel_y'][1]),
                    rng.uniform(tc['vel_z'][0], tc['vel_z'][1])
                ]
            target_gt_list.append({
                'center_xyz': center, 'size_xyz': size, 'vel_xyz': vel
            })
            
        all_pts, all_int, all_vel = generate_raycast_lidar(
            self.lidar_config, target_gt_list, rng, sensor_origin=sensor_origin
        )
            
        iq_noisy, iq_clean, meta = synthesize_fmcw_iq_from_points(
            all_pts, all_int, self.params, 
            ego_vel_xyz=np.array([0.0, 0.0, 0.0]),
            obj_vel_xyz=all_vel,
            sensor_origin=sensor_origin
        )
        
        RD_dB = rd_map(iq_noisy)
        
        return {
            'rd_map': RD_dB.astype(np.float32),
            'iq_cube': iq_noisy.astype(np.complex64),
            'target_gt': target_gt_list,
            'points_xyz': all_pts,
            'points_int': all_int,
            'sensor_origin': sensor_origin,
            'lidar_config': self.lidar_config,
            'params': self.params
        }

# -----------------------------
# Lidar-to-Radar Dataset Class
# -----------------------------
class Lidar2RadarDataset:
    def __init__(self, adapter: KittiAdapter, params: RadarParams):
        self.adapter = adapter
        self.params = params
        
    def __len__(self):
        return len(self.adapter)
        
    def __getitem__(self, idx):
        try:
            lidar_frame = self.adapter[idx]
        except Exception as e:
            print(f"Error loading frame {idx} from adapter: {e}")
            M, Ns = self.params.M_chirp, self.params.Ns_fast()
            return {
                'rd_map': np.zeros((M, Ns), dtype=np.float32),
                'iq_cube': np.zeros((M, Ns), dtype=np.complex64),
                'target_gt': [],
                'points_xyz': np.zeros((0,3)),
                'points_int': np.zeros(0),
                'sensor_origin': np.array([0.0, 0.0, self.params.sensor_height_m]),
                'params': self.params,
                'meta': {"error": str(e)}
            }
        
        pts_radar = apply_extrinsics(lidar_frame.points_xyz, lidar_frame.T_radar_lidar)
        
        sensor_origin_for_synth = np.array([0.0, 0.0, self.params.sensor_height_m])
        
        iq_noisy, iq_clean, meta_sc = synthesize_fmcw_iq_from_points(
            pts_radar,
            lidar_frame.intensity, 
            self.params, 
            ego_vel_xyz=lidar_frame.ego_vel_xyz,
            obj_vel_xyz=None,
            sensor_origin=np.array([0.0,0.0,0.0])
        )
        
        RD_dB = rd_map(iq_noisy)
        
        target_gt = lidar_frame.target_boxes
        
        return {
            'rd_map': RD_dB.astype(np.float32),
            'iq_cube': iq_noisy.astype(np.complex64),
            'target_gt': target_gt,
            'points_xyz': pts_radar,
            'points_int': lidar_frame.intensity,
            'sensor_origin': sensor_origin_for_synth,
            'params': self.params,
            'meta': {**lidar_frame.meta, "scatter_stats": meta_sc}
        }

# -----------------------------
# Dataset Demo & Processing
# -----------------------------
def run_dataset_demo(outdir):
    print(f"--- Running Synthetic Dataset Demo (Raycast) ---")
    os.makedirs(outdir, exist_ok=True)
    viz_dir = os.path.join(outdir, "figs"); os.makedirs(viz_dir, exist_ok=True)
    
    params = RadarParams(M_chirp=64, sensor_height_m=1.8, azi_beamwidth_deg=60.0, ele_beamwidth_deg=20.0)
    R_unamb = params.unambiguous_range()
    v_unamb_doppler = params.unambiguous_doppler_vel()
    print(f"FMCW Unambiguous Range: {R_unamb:.2f} m")
    print(f"FMCW Unambiguous Doppler Velocity: {v_unamb_doppler:.2f} m/s")

    lidar_config = {
        "v_fov_deg": params.ele_beamwidth_deg,
        "h_fov_deg": params.azi_beamwidth_deg,
        "v_res_beams": 32,
        "h_res_steps": 512,
        "max_range": R_unamb * 0.9, 
    }
    
    # <--- NEW: Define high and low velocity ranges to demo MTI filter --->
    target_config = {
        "pos_x": (5.0, R_unamb * 0.8),
        "pos_y": (-R_unamb * 0.4, R_unamb * 0.4),
        "pos_z": (0.0, 2.0),
        "size_x": (1.5, 4.5), "size_y": (1.5, 2.5), "size_z": (1.0, 2.0),
        "high_vel_x": (7.0, 15.0), # Ensure targets here are well above MTI blind spot
        "low_vel_x": (-1.0, 1.0), # Targets here might be in the MTI blind spot
        "vel_y": (-2.0, 2.0),
        "vel_z": (0.0, 0.0)
    }
    
    dataset = SyntheticRadarDataset(
        num_samples=10, 
        params=params, 
        lidar_config=lidar_config, 
        target_config=target_config,
        max_targets=3
    )
    
    print(f"Created dataset with {len(dataset)} samples.")
    
    for i in range(3):
        if i >= len(dataset):
            break
            
        print(f"\n--- Generating and visualizing sample {i} ---")
        sample = dataset[i]
        
        RD_dB = sample['rd_map']
        gt_list = sample['target_gt']
        
        print(f"--- Ground Truth for Sample {i} ({len(gt_list)} targets) ---")
        sensor_origin = sample['sensor_origin']
        
        for j, gt in enumerate(gt_list):
            p_abs = np.array(gt['center_xyz'])
            v_abs = np.array(gt['vel_xyz'])
            p_rel = p_abs - sensor_origin
            range_m = np.linalg.norm(p_rel)
            u_vec = p_rel / range_m
            rad_vel_ms = np.dot(u_vec, v_abs)
            
            # <--- NEW: Explicitly check for MTI blind spot --->
            is_in_blind_spot = abs(rad_vel_ms) < 2.0 # MTI filter attenuates strongly < 2 m/s
            mti_status = " (WARNING: In MTI blind spot, will be filtered!)" if is_in_blind_spot else ""

            print(f"  Target {j+1}:")
            print(f"    Position (m): {p_abs.round(2)}")
            print(f"    Velocity (m/s): {v_abs.round(2)}")
            print(f"    -> Radial Velocity (m/s): {rad_vel_ms:.2f}{mti_status}")
        print("----------------------------------")
        
        print(f"Saving visualizations to: {viz_dir}/sample_{i:02d}_*")
        
        viz_rd(RD_dB, os.path.join(viz_dir, f"sample_{i:02d}_rd_2d.png"), 
               sample['params'], target_gt_list=gt_list, sensor_origin=sensor_origin)
        
        viz_rd_3d(RD_dB, os.path.join(viz_dir, f"sample_{i:02d}_rd_3d.png"), 
                  sample['params'], target_gt_list=gt_list, sensor_origin=sensor_origin)
        
        viz_range_profile(RD_dB, os.path.join(viz_dir, f"sample_{i:02d}_range.png"), sample['params'])
        
        viz_bev_3d(os.path.join(viz_dir, f"sample_{i:02d}_bev_3d.png"), 
                   sample['points_xyz'], 
                   sample['target_gt'],
                   sample['params'],
                   sample['lidar_config'],
                   sensor_origin=sample['sensor_origin'])
        
        # <--- FIX: This plot will now work --->
        if len(sample['points_int']) > 0:
            rcs_values = np.array([rcs_from_intensity(v) for v in sample['points_int']])
            viz_intensity_rcs(sample['points_int'], rcs_values, os.path.join(viz_dir, f"sample_{i:02d}_rcs.png"))
        else:
            print(f"  ... No points generated for sample {i}, skipping RCS plot.")
    
    print(f"\n--- Synthetic demo complete. {i+1} samples written to: {outdir} ---")
    print("Check 'sample_XX_bev_3d.png' for FOV fan, coordinate axes, range/vel vectors, and 'sample_XX_rd_2d.png' for clear targets and GT labels.")

def process_real_dataset(adapter, params: RadarParams, outdir, viz_every=20):
    print(f"--- Processing Real Dataset from Adapter ---")
    dataset = Lidar2RadarDataset(adapter, params)
    os.makedirs(outdir, exist_ok=True)
    viz_dir = os.path.join(outdir, "figs"); os.makedirs(viz_dir, exist_ok=True)

    for i in range(len(dataset)):
        if i >= viz_every * 5:
             print(f"Stopping real dataset processing after {i} frames.")
             break
        if i % viz_every == 0:
            print(f"Processing and visualizing frame {i}...")
        else:
            print(f"Processing frame {i}...")
            
        sample = dataset[i]
        
        meta = {
            "frame_index": i,
            "radar_params": params,
            **sample.get('meta', {})
        }
        save_sample(outdir, i, sample['iq_cube'], sample['rd_map'], meta)

        if (i % viz_every) == 0:
            RD_dB = sample['rd_map']
            gt_list = sample['target_gt']
            sensor_origin = sample['sensor_origin']
            
            viz_rd(RD_dB, os.path.join(viz_dir, f"rd_2d_{i:06d}.png"), params, target_gt_list=gt_list, sensor_origin=sensor_origin)
            viz_rd_3d(RD_dB, os.path.join(viz_dir, f"rd_3d_{i:06d}.png"), params, target_gt_list=gt_list, sensor_origin=sensor_origin)
            viz_range_profile(RD_dB, os.path.join(viz_dir, f"range_{i:06d}.png"), params)
            # We can't run viz_bev_3d on real data without lidar_config, so we skip it.
            viz_bev(sample['points_xyz'], os.path.join(viz_dir, f"bev_2d_{i:06d}.png"), params=params)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="demo", choices=["demo","kitti","nuscenes","waymo"])
    ap.add_argument("--root", type=str, default="./kitti_tiny/", help="dataset root (for adapters)")
    ap.add_argument("--seq", type=str, default="0000", help="KITTI sequence id")
    ap.add_argument("--outdir", type=str, default="./output/radar_from_lidar")
    ap.add_argument("--viz-every", type=int, default=1)
    # Radar params
    ap.add_argument("--fc", type=float, default=77e9)
    ap.add_argument("--fs", type=float, default=10e6)
    ap.add_argument("--slope", type=float, default=60e12)
    ap.add_argument("--T", type=float, default=50e-6)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--nf", type=float, default=5.0)
    ap.add_argument("--cfo", type=float, default=0.0)
    ap.add_argument("--pn-std", type=float, default=0.0)
    ap.add_argument("--sensor-height", type=float, default=1.8, help="Radar sensor height above ground (Z=0)")
    ap.add_argument("--azi-bw", type=float, default=60.0, help="Radar Azimuth beamwidth in degrees")
    ap.add_argument("--ele-bw", type=float, default=20.0, help="Radar Elevation beamwidth in degrees")

    args = ap.parse_args()
    
    if args.mode == "demo":
        run_dataset_demo(args.outdir)
    else:
        params = RadarParams(fc=args.fc, fs=args.fs, slope=args.slope, T_chirp=args.T,
                             M_chirp=args.M, noise_figure_dB=args.nf, cfo_Hz=args.cfo,
                             phase_std=args.pn_std, sensor_height_m=args.sensor_height,
                             azi_beamwidth_deg=args.azi_bw, ele_beamwidth_deg=args.ele_bw)
        adapter = None
        if args.mode == "kitti":
            adapter = KittiAdapter(args.root, seq=args.seq, T_radar_lidar=None)
        elif args.mode == "nuscenes":
            adapter = NuScenesAdapter(args.root, T_radar_lidar=None)
        elif args.mode == "waymo":
            adapter = WaymoAdapter([args.root], T_radar_lidar=None)
        
        if adapter and len(adapter) > 0:
            process_real_dataset(adapter, params, args.outdir, viz_every=args.viz_every)
        elif adapter:
            print(f"Adapter for mode '{args.mode}' initialized, but found 0 frames. Check --root and --seq.")
        else:
            print(f"Adapter for mode '{args.mode}' not initialized.")

if __name__ == "__main__":
    main()