import os, time, numpy as np, matplotlib.pyplot as plt
import torch
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

try: import scipy.ndimage as ndi; SCIPY = True
except ImportError: SCIPY = False; print("Warning: Scipy not installed. CFAR disabled.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {DEVICE} ---")
C0 = 299792458.0

# ================= PARAMS (UNIFIED & MATCHED) =================
@dataclass
class SystemParams:
    """Common parameters for fair comparison"""
    fc: float = 77e9
    fs: float = 150e6    # 150 MHz bandwidth (~1m resolution)
    M: int = 512         # Doppler bins
    N: int = 512         # Range/Delay bins
    H: float = 1.8       # Sensor height
    az_fov: float = 60.0
    el_fov: float = 20.0

    @property
    def lambda_m(self): return C0 / self.fc

    # --- FMCW Specific ---
    @property
    def fmcw_slope(self): return self.fs**2 * C0 / (2 * self.fs * 100.0) # Tune for ~100m max range
    # Actually, let's just fix T_chirp to get N samples
    @property
    def T_chirp(self): return self.N / self.fs
    @property
    def fmcw_max_r(self): return self.fs * C0 / (2 * (self.fs/self.T_chirp))
    @property
    def fmcw_max_v(self): return self.lambda_m / (4 * self.T_chirp)
    
    # --- OTFS Specific ---
    @property
    def otfs_max_r(self): return (C0 / (2 * self.fs)) * self.N
    @property
    def otfs_max_v(self): return (self.lambda_m / 2) * (self.fs / (self.N * self.M)) * (self.M / 2)

    def fmcw_axes(self):
        r = np.linspace(0, self.fmcw_max_r, self.N)
        v = np.linspace(-self.fmcw_max_v, self.fmcw_max_v, self.M)
        return r, v

    def otfs_axes(self):
        r = np.linspace(0, self.otfs_max_r, self.N)
        v = np.linspace(-self.otfs_max_v, self.otfs_max_v, self.M)
        return r, v

# ================= GPU KERNELS =================
def to_torch(x): return torch.tensor(x, device=DEVICE, dtype=torch.float32)

def raycast_torch(sp: SystemParams, gts):
    az = torch.linspace(np.deg2rad(-sp.az_fov/2), np.deg2rad(sp.az_fov/2), 1024, device=DEVICE)
    el = torch.linspace(np.deg2rad(-sp.el_fov/2), np.deg2rad(sp.el_fov/2), 128, device=DEVICE)
    EL, AZ = torch.meshgrid(el, az, indexing='ij')
    rays = torch.stack([torch.cos(EL)*torch.cos(AZ), torch.cos(EL)*torch.sin(AZ), torch.sin(EL)], dim=-1).reshape(-1, 3)
    pos = torch.tensor([0.,0.,sp.H], device=DEVICE)
    
    # Clip at 100m to keep scene contained
    t_min = torch.full((rays.shape[0],), 100.0, device=DEVICE)
    hits_int = torch.zeros((rays.shape[0],), device=DEVICE)
    hits_vel = torch.zeros((rays.shape[0], 3), device=DEVICE)

    mask_g = (rays[:, 2] < -2e-2) # Slightly stricter ground angle
    t_g = -pos[2] / rays[:, 2]
    mask_valid_g = mask_g & (t_g > 0) & (t_g < t_min)
    t_min[mask_valid_g] = t_g[mask_valid_g]
    hits_int[mask_valid_g] = 100.0

    if gts:
        Cs = torch.stack([to_torch(gt['c']) for gt in gts])
        Ss = torch.stack([to_torch(gt['s']) for gt in gts])
        Vs = torch.stack([to_torch(gt['v']) for gt in gts])
        ro = pos.view(1,1,3); rd = rays.view(-1,1,3)+1e-9
        t1 = (Cs-Ss/2-ro)/rd; t2 = (Cs+Ss/2-ro)/rd
        tn = torch.max(torch.min(t1,t2), dim=-1)[0]
        tf = torch.min(torch.max(t1,t2), dim=-1)[0]
        mask_hit = (tn < tf) & (tn > 0)
        tn[~mask_hit] = np.inf
        min_t, min_idx = torch.min(tn, dim=1)
        mask_t = min_t < t_min
        t_min[mask_t] = min_t[mask_t]
        hits_int[mask_t] = 255.0
        hits_vel[mask_t] = Vs[min_idx[mask_t]]

    mask = hits_int > 0
    return pos + t_min[mask].unsqueeze(1)*rays[mask], hits_int[mask], hits_vel[mask]

def fmcw_torch(pts, its, vels, sp: SystemParams):
    M, N = sp.M, sp.N
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    if len(pts)==0: return iq.cpu().numpy()
    
    P = pts - to_torch([0,0,sp.H]); R = torch.norm(P, dim=1)
    mask = R > 0.1; P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    
    # High target RCS for visibility
    amp = torch.where(its==255, 1e7, 1e-1) / R**2
    vr = torch.sum(P/R.unsqueeze(1)*vels, dim=1)
    
    t_f = torch.arange(N, device=DEVICE)/sp.fs
    t_s = torch.arange(M, device=DEVICE)*sp.T_chirp
    slope = sp.fs / sp.T_chirp
    k_r = 2*slope/C0; k_v = 2/sp.lambda_m
    
    BATCH = 4096
    for i in range(0, len(R), BATCH):
        rb, vrb, ab = R[i:i+BATCH], vr[i:i+BATCH], amp[i:i+BATCH]
        phase = 2j*np.pi*( (k_r*rb[:,None,None])*t_f[None,None,:] + (k_v*vrb[:,None,None])*t_s[None,:,None] )
        iq += torch.sum(ab[:,None,None]*torch.exp(phase), dim=0)
        
    iq += (torch.randn(M,N,device=DEVICE)+1j*torch.randn(M,N,device=DEVICE))*1e-4
    iq[1:] -= iq[:-1].clone(); iq[0]=0 # MTI
    win = torch.hann_window(N,device=DEVICE)*torch.hann_window(M,device=DEVICE)[:,None]
    return (20*torch.log10(torch.abs(torch.fft.fftshift(torch.fft.fft2(iq*win)))+1e-9)).cpu().numpy()

def otfs_torch(pts, its, vels, sp: SystemParams):
    M, N = sp.M, sp.N
    H = torch.zeros((M,N), dtype=torch.complex64, device=DEVICE)
    if len(pts)==0: return H.cpu().numpy()
    
    P = pts - to_torch([0,0,sp.H]); R = torch.norm(P, dim=1)
    mask = R > 0.1; P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    
    amp = torch.where(its==255, 1e7, 1e-1) / R**2
    vr = torch.sum(P/R.unsqueeze(1)*vels, dim=1)
    
    # Exact OTFS grid definitions
    delta_f = sp.fs / M
    T_sym = 1.0 / sp.fs # effectively
    # Wait, standard OTFS: 
    # Bandwidth = N * delta_f_otfs? Or fs is total BW?
    # Let's use: Total BW = fs. Total duration = M * T_sym.
    # Delay resolution = 1/fs. Doppler resolution = 1/(M*N/fs) = fs/(MN).
    
    k_res = C0 / (2 * sp.fs)
    l_res = (sp.lambda_m / 2) * (sp.fs / (M * N))

    k = (R / k_res).long()
    l = (vr / l_res).long() + M//2
    
    valid = (k>=0) & (k<N) & (l>=0) & (l<M)
    H.view(-1).scatter_add_(0, l[valid]*N + k[valid], amp[valid].to(torch.complex64))
    H += (torch.randn(M,N,device=DEVICE)+1j*torch.randn(M,N,device=DEVICE))*1e-4
    return (20*torch.log10(torch.abs(H)+1e-9)).cpu().numpy()

# ================= VIZ =================
def cfar(rd, thresh=15):
    if not SCIPY: return np.zeros_like(rd, bool)
    k = np.ones((5,5)); k[2,2]=0; k/=k.sum()
    noise = ndi.convolve(rd, k, mode='mirror')
    det = rd > (noise + thresh)
    det[rd.shape[0]//2-2:rd.shape[0]//2+3, :] = False
    return det

def viz_rd_2d_compare(path, rd_f, rd_o, gts, sp: SystemParams):
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    pos = np.array([0,0,sp.H])
    
    # Get exact axes for both
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    for i, (rd, ra, va, name) in enumerate([(rd_f, ra_f, va_f, "FMCW"), (rd_o, ra_o, va_o, "OTFS")]):
        # Dynamic vmin for visibility: top 35dB only
        vmin_val = np.max(rd) - 35
        im = ax[i].pcolormesh(ra, va, rd, cmap='jet', vmin=vmin_val, vmax=np.max(rd), shading='auto')
        plt.colorbar(im, ax=ax[i], label='dB')
        
        # Plot GT only if it falls within THIS specific plot's axes
        for gt in gts:
            P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                 ax[i].plot(r, v, 'wx', ms=12, mew=3, label='GT' if gt==gts[0] and i==0 else "")
                 ax[i].text(r+1, v+1, f"{r:.0f}m,{v:.0f}m/s", color='white', fontweight='bold')

        det = cfar(rd, thresh=20) # Slightly higher threshold to reduce clutter
        if np.any(det):
            y,x = np.where(det)
            ax[i].scatter(ra[x], va[y], s=150, facecolors='none', edgecolors='cyan', lw=2, label='CFAR')
        
        ax[i].set_title(f"{name} Range-Doppler")
        ax[i].set_xlabel("Range (m)")
        ax[i].set_ylim(va[0], va[-1])
        ax[i].set_xlim(0, ra[-1])
        ax[i].legend(loc='upper right')
    
    ax[0].set_ylabel("Velocity (m/s)")
    plt.tight_layout(); plt.savefig(path); plt.close()

def viz_rd_3d_compare(path, rd_f, rd_o, gts, sp: SystemParams):
    fig = plt.figure(figsize=(18,8))
    pos = np.array([0,0,sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    for i, (rd, ra, va, name) in enumerate([(rd_f, ra_f, va_f, "FMCW"), (rd_o, ra_o, va_o, "OTFS")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        R, V = np.meshgrid(ra, va)
        surf = np.maximum(rd, np.max(rd)-40) # Clip floor
        ax.plot_surface(R, V, surf, cmap='viridis', rstride=2, cstride=2, alpha=0.8)
        for gt in gts:
            P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax.scatter([r],[v],[np.max(rd)+5], c='r', marker='x', s=300, lw=4, zorder=10)
        ax.set_title(f"{name} 3D"); ax.set_xlabel("Range(m)"); ax.set_ylabel("Vel(m/s)")
        ax.set_xlim(0, ra[-1]); ax.set_ylim(va[0], va[-1]); ax.view_init(45, -110)
    plt.tight_layout(); plt.savefig(path); plt.close()

# ================= RUN =================
if __name__ == '__main__':
    root = "./output/final_matched_viz"
    os.makedirs(root, exist_ok=True)
    sp = SystemParams() # Common params
    
    # Targets positioned to be visible in BOTH systems (approx <75m, <+/-30m/s)
    gts = [{'c':[20, 0, 1], 's':[4,2,2], 'v':[12, 0, 0]}, 
           {'c':[50, -5, 1], 's':[5,3,3], 'v':[-18, 5, 0]}]

    print(f"Simulating..."); pts, its, vels = raycast_torch(sp, gts); torch.cuda.synchronize()
    print(f"Raycast: {len(pts)} points.")
    rd_f = fmcw_torch(pts, its, vels, sp)
    rd_o = otfs_torch(pts, its, vels, sp)
    
    print("Saving visualizations...")
    viz_rd_2d_compare(f"{root}/compare_2d.png", rd_f, rd_o, gts, sp)
    viz_rd_3d_compare(f"{root}/compare_3d.png", rd_f, rd_o, gts, sp)
    print("Done.")