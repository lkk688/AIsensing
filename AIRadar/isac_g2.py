import os, json, math, glob, argparse, warnings, time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Try importing scipy
try:
    import scipy.ndimage as ndi
    import scipy.signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("Scipy not found. CFAR detection will not work.")
    SCIPY_AVAILABLE = False

C0 = 299792458.0

# -----------------------------
# 1. Tuned Parameters
# -----------------------------
@dataclass
class RadarParams:
    fc: float = 77e9
    fs: float = 10e6
    T_chirp: float = 48e-6 
    M_chirp: int = 128
    slope: float = (10e6 * C0) / (2 * 50.0) # Tuned for exactly 50m max range
    noise_tempK: float = 290.0
    noise_figure_dB: float = 10.0
    azi_beamwidth_deg: float = 30.0
    ele_beamwidth_deg: float = 10.0
    sensor_height_m: float = 1.8
    
    def lambda_m(self): return C0 / self.fc
    def Ns_fast(self): return int(round(self.fs * self.T_chirp))
    def unambiguous_range(self): return (self.fs * C0) / (2.0 * self.slope)
    def unambiguous_doppler_vel(self): return self.lambda_m() / (2 * self.T_chirp)

@dataclass
class OTFSParams:
    fc: float = 77e9
    fs: float = 10e6             # Bandwidth determines Delay resolution
    M_doppler: int = 128         # Number of Doppler bins
    N_delay: int = 128           # Number of Delay bins
    
    def lambda_m(self): return C0 / self.fc
    # Total frame duration determines Doppler resolution. 
    # We set it to match FMCW frame time roughly (~6ms) for comparable velocity resolution.
    def total_duration(self): return 128 * 48e-6 
    
    def max_range(self): return (C0 / 2) * (self.N_delay / self.fs)
    def max_velocity(self): return (self.lambda_m() / 2) * (self.M_doppler / self.total_duration())

# -----------------------------
# 2. Signal Processing (Fixed)
# -----------------------------
def rcs_from_intensity(intensity):
    if intensity is None: return 0.5
    # Boost targets even more to stand out clearly against noise
    if intensity == 255.0: return 10.0**(30.0/10.0) # Target: +30 dBsm
    if intensity == 100.0: return 10.0**(-10.0/10.0) # Ground: -10 dBsm
    return 1.0

def synthesize_fmcw_iq(points, intensity, params: RadarParams, sensor_pos, obj_vel):
    M, Ns = params.M_chirp, params.Ns_fast()
    if len(points) == 0: return np.zeros((M, Ns), dtype=np.complex64)

    P_rel = points - sensor_pos
    R = np.linalg.norm(P_rel, axis=1)
    valid = (R > 0.5) & (R < params.unambiguous_range() * 1.1)
    P_rel, R, I, V = P_rel[valid], R[valid], intensity[valid], obj_vel[valid]
    
    if len(P_rel) == 0: return np.zeros((M, Ns), dtype=np.complex64)

    v_r = np.sum((P_rel/R[:,None]) * V, axis=1)
    amp = np.sqrt([rcs_from_intensity(i) for i in I]) / R**2

    t_n = np.arange(Ns) / params.fs
    t_m = np.arange(M) * params.T_chirp
    
    phase_n = (4 * np.pi * params.slope * R[:,None] / C0) * t_n[None,:]
    phase_m = (4 * np.pi * v_r[:,None] / params.lambda_m()) * t_m[None,:]
    
    sig = amp[:,None,None] * np.exp(1j * (phase_n[:,None,:] + phase_m[:,:,None]))
    iq = np.sum(sig, axis=0)

    # Noise floor
    noise_pow = 1e-14 * 10**(params.noise_figure_dB/10)
    iq += (np.random.randn(M, Ns) + 1j*np.random.randn(M, Ns)) * np.sqrt(noise_pow)
    return iq

def fmcw_process(iq):
    """Applies Blackman window and 2D FFT."""
    # MTI
    iq_mti = iq - np.roll(iq, 1, axis=0); iq_mti[0]=0
    # Blackman window for better sidelobe suppression than Hanning
    win_fast = np.blackman(iq.shape[1])
    win_slow = np.blackman(iq.shape[0])
    iq_win = iq_mti * win_fast[None,:] * win_slow[:,None]
    
    rd = np.fft.fftshift(np.fft.fft2(iq_win, axes=(0,1)), axes=0)
    return normalize_db(rd)

def synthesize_otfs_dd(points, intensity, o_p: OTFSParams, sensor_pos, obj_vel):
    """
    Precisely maps scatterers to Delay-Doppler bins based on physical resolution.
    """
    M, N = o_p.M_doppler, o_p.N_delay
    H_DD = np.zeros((M, N), dtype=np.complex64)
    
    if len(points) > 0:
        P_rel = points - sensor_pos
        R = np.linalg.norm(P_rel, axis=1)
        valid = (R > 0.5) & (R < 70.0) # Allow slightly beyond 50m to avoid hard clipping edge artifacts
        P_rel, R, I, V = P_rel[valid], R[valid], intensity[valid], obj_vel[valid]
        
        v_r = np.sum((P_rel/R[:,None]) * V, axis=1)
        amp = np.sqrt([rcs_from_intensity(i) for i in I]) / R**2

        # --- Precise Mapping ---
        # Delay resolution (s) = 1 / Bandwidth
        res_delay_s = 1.0 / o_p.fs
        # Doppler resolution (Hz) = 1 / Total Duration
        res_doppler_hz = 1.0 / o_p.total_duration()
        
        # Physical values
        tau = 2 * R / C0
        nu = 2 * v_r / o_p.lambda_m()
        
        # Grid indices
        k_idx = np.round(tau / res_delay_s).astype(int)
        l_idx = np.round(nu / res_doppler_hz).astype(int)

        # Accumulate with boundary checks
        for i in range(len(R)):
            if 0 <= k_idx[i] < N and abs(l_idx[i]) < M//2:
                # Shift Doppler so 0 velocity is at center index M//2
                l_shifted = l_idx[i] + M//2
                H_DD[l_shifted, k_idx[i]] += amp[i] * np.exp(1j*np.random.rand()*2*np.pi)

    # Noise matches FMCW level roughly
    noise_pow = 1e-14
    H_DD += (np.random.randn(M,N) + 1j*np.random.randn(M,N)) * np.sqrt(noise_pow)
    return normalize_db(H_DD)

def normalize_db(map_data):
    mag = np.abs(map_data)
    # Normalize to max=0dB for easy consistent visualization
    return 20 * np.log10(mag / np.max(mag) + 1e-12)

# -----------------------------
# 3. Tuned CFAR
# -----------------------------
def cfar_detect_tuned(rd_db, guard=(8,6), train=(12,8), offset_db=20.0):
    """
    Highly tuned CFAR to reject sidelobes.
    Increased guard cells to ignore the spread of the bright target itself.
    """
    if not SCIPY_AVAILABLE: return np.zeros_like(rd_db, dtype=bool)
    
    kh, kw = 2*train[0]+2*guard[0]+1, 2*train[1]+2*guard[1]+1
    kernel = np.ones((kh, kw))
    kernel[train[0]:train[0]+2*guard[0]+1, train[1]:train[1]+2*guard[1]+1] = 0
    kernel /= np.sum(kernel)
    
    noise = ndi.convolve(rd_db, kernel, mode='mirror')
    dets = rd_db > (noise + offset_db)
    
    # Clutter notch
    dc = rd_db.shape[0] // 2
    dets[dc-2:dc+3, :] = False
    
    return dets

def get_detections(mask, r_ax, v_ax):
    if not SCIPY_AVAILABLE: return []
    lbl, n = ndi.label(mask)
    dets = []
    for i in range(1, n+1):
        coords = np.argwhere(lbl == i)
        v_idx, r_idx = np.mean(coords, axis=0).astype(int)
        dets.append({'range': r_ax[r_idx], 'velocity': v_ax[v_idx]})
    return dets

# -----------------------------
# 4. Visualization (Retained & Fixed)
# -----------------------------
# [Keeping Arrow3D and plot_cube_wireframe from previous good version]
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0),(0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)
    def draw(self, renderer):
        self.do_3d_projection(renderer)
        super().draw(renderer)

def plot_cube_wireframe(ax, center, size, color='r'):
    # Ensure inputs are numpy arrays for element-wise arithmetic
    center = np.array(center)
    size = np.array(size)
    
    hl = size / 2.0
    x = [center[0]-hl[0], center[0]+hl[0]]
    y = [center[1]-hl[1], center[1]+hl[1]]
    z = [center[2]-hl[2], center[2]+hl[2]]
    v = np.array([[x[0],y[0],z[0]], [x[1],y[0],z[0]], [x[1],y[1],z[0]], [x[0],y[1],z[0]],
                  [x[0],y[0],z[1]], [x[1],y[0],z[1]], [x[1],y[1],z[1]], [x[0],y[1],z[1]]])
    edges = [[v[0],v[1],v[2],v[3],v[0]], [v[4],v[5],v[6],v[7],v[4]], 
             [v[0],v[4]], [v[1],v[5]], [v[2],v[6]], [v[3],v[7]]]
    for e in edges:
        ax.plot3D(*zip(*e), color=color, linewidth=1.0)

def viz_rd_compare(RD_dB, out_png, r_ax, v_ax, gt_list=None, detections=None, title="", sensor_origin=None):
    plt.figure(figsize=(10, 6))
    # Locked color scale for fair FMCW vs OTFS comparison
    plt.pcolormesh(r_ax, v_ax, RD_dB, shading='auto', cmap='jet', vmin=-60, vmax=0)
    cbar = plt.colorbar(); cbar.set_label("Normalized Power (dB)")

    plt.xlim([0, 50])
    plt.ylim([-20, 20])

    if gt_list:
        gt_r, gt_v = [], []
        for g in gt_list:
             gt_r.append(g['range'])
             gt_v.append(g['velocity'])
             plt.text(g['range']+1, g['velocity']+1, f"GT\n{g['range']:.0f}m\n{g['velocity']:.0f}m/s", 
                      color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
        plt.plot(gt_r, gt_v, 'rx', ms=12, mew=3, label='Ground Truth')

    if detections:
        r_d = [d['range'] for d in detections]
        v_d = [d['velocity'] for d in detections]
        plt.plot(r_d, v_d, 'co', ms=15, markerfacecolor='none', mew=2, label='CFAR Detection')

    plt.xlabel("Range (m)"); plt.ylabel("Radial Velocity (m/s)")
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def viz_bev_3d_retained(out_png, points, gt_list, r_params, lidar_cfg, sensor_pos):
    """Exact same BEV visualization as before, kept as requested."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot([sensor_pos[0]], [sensor_pos[1]], [sensor_pos[2]], 'ko', ms=8)
    # Simple axes
    ax.add_artist(Arrow3D([sensor_pos[0], sensor_pos[0]+5], [sensor_pos[1], sensor_pos[1]], [sensor_pos[2], sensor_pos[2]], mutation_scale=20, lw=2, arrowstyle='-|>', color='r'))
    ax.add_artist(Arrow3D([sensor_pos[0], sensor_pos[0]], [sensor_pos[1], sensor_pos[1]+5], [sensor_pos[2], sensor_pos[2]], mutation_scale=20, lw=2, arrowstyle='-|>', color='g'))
    
    if len(points) > 0:
        # Show only relevant points to avoid clutter
        mask = (points[:,0] < 55) & (np.abs(points[:,1]) < 30)
        pts = points[mask]
        pts = pts[::5] if len(pts) > 10000 else pts
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=0.2, c=pts[:,2], cmap='viridis', alpha=0.5)

    for gt in gt_list:
        plot_cube_wireframe(ax, gt['center_xyz'], gt['size_xyz'], 'r')
        # Draw radial velocity vector from center
        c = gt['center_xyz']
        p_rel = c - sensor_pos
        u_vec = p_rel / np.linalg.norm(p_rel)
        v_rad = np.dot(u_vec, gt['vel_xyz']) * u_vec # Vector component
        # Scale for visualization
        ax.add_artist(Arrow3D([c[0], c[0]+v_rad[0]], [c[1], c[1]+v_rad[1]], [c[2], c[2]+v_rad[2]], 
                              mutation_scale=15, lw=3, arrowstyle='-|>', color='cyan'))

    # Limits
    ax.set_xlim([0, 55]); ax.set_ylim([-27.5, 27.5]); ax.set_zlim([0, 20])
    ax.set_xlabel("X [Fwd]"); ax.set_ylabel("Y [Left]"); ax.set_zlabel("Z [Up]")
    ax.set_title("3D BEV (Cyan arrows = Radial Velocity)")
    ax.view_init(elev=60, azim=-100) # Higher view to see layout better
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# [Re-using previous viz_rd_3d, viz_range_profile, viz_intensity_rcs directly]
# They are unchanged in logic, just ensured they are present in final script.
def viz_rd_3d_retained(RD_dB, out_png, r_axis, v_axis, gt_list, sensor_pos):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    R_g, V_g = np.meshgrid(r_axis, v_axis)
    # Clip for cleaner 3D view
    Z = np.maximum(RD_dB, -60)
    ax.plot_surface(R_g, V_g, Z, cmap='viridis', rstride=2, cstride=2, alpha=0.8, linewidth=0, antialiased=False)
    for gt in gt_list:
        ax.scatter([gt['range']], [gt['velocity']], [0], color='r', marker='x', s=500, linewidth=4, zorder=10)
    ax.set_xlim([0, 50]); ax.set_ylim([-20, 20]); ax.set_zlim([-60, 0])
    ax.set_xlabel('Range (m)'); ax.set_ylabel('Velocity (m/s)')
    ax.view_init(elev=55, azim=-130)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def viz_range_profile_v2(RD_dB, out_png, r_axis):
    plt.figure(figsize=(8,4))
    plt.plot(r_axis, np.max(RD_dB, axis=0))
    plt.xlabel("Range (m)"); plt.ylabel("Power (dB)"); plt.grid(True)
    plt.title("Range Profile (Max over Doppler)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# -----------------------------
# Main Demo Driver
# -----------------------------
def run_demo(outdir):
    os.makedirs(outdir, exist_ok=True)
    viz_dir = os.path.join(outdir, "figs_v4_fixed")
    os.makedirs(viz_dir, exist_ok=True)

    r_par = RadarParams()
    o_par = OTFSParams()
    s_pos = np.array([0,0,r_par.sensor_height_m])

    # --- Scene Setup ---
    rng = np.random.default_rng(42)
    # Ground clutter
    p_g = np.stack([rng.uniform(0,60,20000), rng.uniform(-30,30,20000), np.zeros(20000)], axis=1)
    v_g = np.zeros((20000,3))
    i_g = np.full(20000, 100.0)

    # Target 1: 30m range, +15m/s radial velocity
    # Position it at (30, 0, 1) so radial velocity is exactly +15
    t1_c = np.array([30.0, 0.0, 1.0]); t1_v = np.array([15.0, 0.0, 0.0])
    p_t1 = t1_c + rng.uniform(-1,1,(150,3))
    
    # Target 2: 20m range, -10m/s radial velocity
    t2_c = np.array([20.0, 5.0, 1.0]); 
    # Calculate required velocity vector to get exactly -10m/s radial
    dir_2 = (t2_c - s_pos) / np.linalg.norm(t2_c - s_pos)
    t2_v = dir_2 * (-10.0) 
    p_t2 = t2_c + rng.uniform(-1,1,(150,3))

    all_pts = np.vstack([p_g, p_t1, p_t2]).astype(np.float32)
    all_vel = np.vstack([v_g, np.tile(t1_v,(150,1)), np.tile(t2_v,(150,1))]).astype(np.float32)
    all_int = np.concatenate([i_g, np.full(150,255.0), np.full(150,255.0)]).astype(np.float32)

    # GT List for plotting
    gt_list = []
    for c, v in [(t1_c, t1_v), (t2_c, t2_v)]:
        p_rel = c - s_pos
        r = np.linalg.norm(p_rel)
        vr = np.dot(p_rel/r, v)
        gt_list.append({'range': r, 'velocity': vr, 'center_xyz': c, 'vel_xyz': v, 'size_xyz': [2,2,2]})
        print(f"GT: Range {r:.1f}m, V_rad {vr:.1f}m/s")

    # --- Axes Definition ---
    r_ax_f = np.linspace(0, r_par.unambiguous_range(), r_par.Ns_fast())
    v_ax_f = np.linspace(-r_par.unambiguous_doppler_vel()/2, r_par.unambiguous_doppler_vel()/2, r_par.M_chirp)
    
    # OTFS Axes (derived from physical limits and grid size)
    r_max_o = (C0 / 2) * (o_par.N_delay / o_par.fs)
    v_max_o = (o_par.lambda_m() / 2) * (o_par.M_doppler / o_par.total_duration())
    r_ax_o = np.linspace(0, r_max_o, o_par.N_delay)
    v_ax_o = np.linspace(-v_max_o/2, v_max_o/2, o_par.M_doppler)

    # --- OTFS Debug Prints ---
    print(f"\nOTFS Grid Check:")
    for i, gt in enumerate(gt_list):
        k = int(gt['range'] / (r_max_o / o_par.N_delay))
        l = int(gt['velocity'] / (v_max_o / o_par.M_doppler)) + o_par.M_doppler//2
        print(f"GT {i+1} should appear at Delay Bin [{k}] and Doppler Bin [{l}]")

    # --- Processing & Viz ---
    print("\nGenerating Maps...")
    rd_fmcw = fmcw_process(synthesize_fmcw_iq(all_pts, all_int, r_par, s_pos, all_vel))
    dd_otfs = synthesize_otfs_dd(all_pts, all_int, o_par, s_pos, all_vel)

    mask_fmcw = cfar_detect_tuned(rd_fmcw, offset_db=20.0) # Tuned offset
    dets_fmcw = get_detections(mask_fmcw, r_ax_f, v_ax_f)
    print(f"FMCW CFAR Detections: {len(dets_fmcw)}")

    print(f"Saving to {viz_dir}...")
    viz_rd_compare(rd_fmcw, os.path.join(viz_dir, "demo_fmcw_rd_2d.png"), r_ax_f, v_ax_f, gt_list, dets_fmcw, "FMCW Range-Doppler (Blackman+CFAR)", s_pos)
    viz_rd_compare(dd_otfs, os.path.join(viz_dir, "demo_otfs_dd_2d.png"), r_ax_o, v_ax_o, gt_list, None, "OTFS Delay-Doppler (Sensing)", s_pos)
    
    # Retained 3D visualizations
    viz_bev_3d_retained(os.path.join(viz_dir, "demo_bev_3d.png"), all_pts, gt_list, r_par, {}, s_pos)
    viz_rd_3d_retained(rd_fmcw, os.path.join(viz_dir, "demo_fmcw_rd_3d.png"), r_ax_f, v_ax_f, gt_list, s_pos)

if __name__ == "__main__":
    run_demo("./output/demo_v4")