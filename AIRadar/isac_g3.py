import os, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Try importing scipy for CFAR
try: import scipy.ndimage as ndi; SCIPY = True
except ImportError: SCIPY = False; print("Warning: Scipy not installed. CFAR disabled.")

# Setup Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Computation Device: {DEVICE} ---")

C0 = 299792458.0

# ================= PARAMS =================
@dataclass
class RadarParams:
    fc: float = 77e9; fs: float = 15e6; T_chirp: float = 30e-6
    M: int = 256; slope: float = 80e12
    az_fov: float = 60.0; el_fov: float = 20.0; H: float = 1.8
    def ranges(self): return np.linspace(0, self.fs*C0/(2*self.slope), int(self.fs*self.T_chirp))
    def vels(self): v=C0/self.fc/(2*self.T_chirp); return np.linspace(-v/2, v/2, self.M)

@dataclass
class OTFSParams:
    fc: float = 77e9; fs: float = 15e6; M: int = 256; N: int = 256
    def ranges(self): return np.linspace(0, C0*self.N/(2*self.fs), self.N)
    def vels(self): v=C0/self.fc*self.fs/(2*self.N); return np.linspace(-v/2, v/2, self.M)

# ================= TORCH UTILS =================
def to_torch(x, dtype=torch.float32): return torch.tensor(x, device=DEVICE, dtype=dtype)

def rcs_torch(i_tensor):
    # Vectorized RCS map: 255->1e3 (30dBsm), 100->1e-2 (-20dBsm)
    rcs = torch.ones_like(i_tensor) * 0.1
    rcs[i_tensor == 255] = 1e3
    rcs[i_tensor == 100] = 1e-2
    return rcs

# ================= GPU KERNELS =================
def raycast_torch(rp: RadarParams, gts):
    # 1. Generate Rays (Massive parallel batch)
    # High resolution 512x64 = 32,768 rays
    az = torch.linspace(np.deg2rad(-rp.az_fov/2), np.deg2rad(rp.az_fov/2), 512, device=DEVICE)
    el = torch.linspace(np.deg2rad(-rp.el_fov/2), np.deg2rad(rp.el_fov/2), 64, device=DEVICE)
    EL, AZ = torch.meshgrid(el, az, indexing='ij')
    rays = torch.stack([torch.cos(EL)*torch.cos(AZ), torch.cos(EL)*torch.sin(AZ), torch.sin(EL)], dim=-1).reshape(-1, 3)
    
    pos = torch.tensor([0.0, 0.0, rp.H], device=DEVICE, dtype=torch.float32)
    t_min = torch.full((rays.shape[0],), 150.0, device=DEVICE)
    hits_int = torch.zeros((rays.shape[0],), device=DEVICE)
    hits_vel = torch.zeros((rays.shape[0], 3), device=DEVICE)

    # 2. Ground Plane Intersection (Z=0)
    # t = -pos[2] / ray_z
    mask_down = rays[:, 2] < -1e-3
    t_g = -pos[2] / rays[:, 2]
    mask_g = mask_down & (t_g > 0) & (t_g < t_min)
    t_min[mask_g] = t_g[mask_g]
    hits_int[mask_g] = 100.0 # Ground ID

    # 3. Target AABB Intersections (Vectorized across all rays AND targets)
    if gts:
        # Stack all target parameters into tensors
        Cs = torch.stack([to_torch(gt['c']) for gt in gts]) # (N_targ, 3)
        Ss = torch.stack([to_torch(gt['s']) for gt in gts])
        Vs = torch.stack([to_torch(gt['v']) for gt in gts])
        
        # Broadcast: (N_rays, 1, 3) vs (1, N_targets, 3)
        ro = pos.view(1, 1, 3)
        rd = rays.view(-1, 1, 3) + 1e-9 # Avoid div zero
        
        t1 = (Cs - Ss/2 - ro) / rd
        t2 = (Cs + Ss/2 - ro) / rd
        t_near = torch.max(torch.min(t1, t2), dim=-1)[0] # (N_rays, N_targets)
        t_far = torch.min(torch.max(t1, t2), dim=-1)[0]
        
        hit_mask = (t_near < t_far) & (t_near > 0)
        
        # Find closest target for each ray
        # Set non-hits to infinity so they don't interfere with min()
        t_near[~hit_mask] = np.inf
        min_t_val, min_t_idx = torch.min(t_near, dim=1)
        
        # Update global t_min where target is closer than ground
        mask_t = min_t_val < t_min
        t_min[mask_t] = min_t_val[mask_t]
        hits_int[mask_t] = 255.0
        hits_vel[mask_t] = Vs[min_t_idx[mask_t]]

    # 4. Gather valid points
    valid = hits_int > 0
    pts = pos + t_min[valid].unsqueeze(1) * rays[valid]
    return pts, hits_int[valid], hits_vel[valid]

def fmcw_torch(pts, its, vels, rp: RadarParams):
    M, N = rp.M, int(rp.fs*rp.T_chirp)
    pos = torch.tensor([0.,0.,rp.H], device=DEVICE)
    
    # Pre-allocate IQ cube on GPU (careful with VRAM if M/N are huge)
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    if len(pts) == 0: return iq.cpu().numpy()

    P = pts - pos
    R = torch.norm(P, dim=1)
    mask = R > 1e-2
    P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    
    # Radar equation components
    vr = torch.sum(P/R.unsqueeze(1) * vels, dim=1)
    amp = torch.sqrt(rcs_torch(its)) / R**2
    
    # Time vectors
    t_f = torch.arange(N, device=DEVICE) / rp.fs
    t_s = torch.arange(M, device=DEVICE) * rp.T_chirp
    
    # Phase calculation (broadcasting to [N_pts, M, N] might be too big, use batching if needed)
    # Trying full broadcast for speed on decent GPUs:
    # (N_pts, 1, 1) * (1, 1, N) -> (N_pts, 1, N)
    # (N_pts, 1, 1) * (1, M, 1) -> (N_pts, M, 1)
    k_r = 2 * rp.slope / C0
    k_v = 2 / (C0 / rp.fc)
    
    # Memory-safe batching for the massive outer product
    BATCH = 4096
    for i in range(0, len(R), BATCH):
        rb, vrb, ab = R[i:i+BATCH], vr[i:i+BATCH], amp[i:i+BATCH]
        phase = 2j * np.pi * ( (k_r * rb[:,None,None]) * t_f[None,None,:] + 
                               (k_v * vrb[:,None,None]) * t_s[None,:,None] )
        iq += torch.sum(ab[:,None,None] * torch.exp(phase), dim=0)

    # Noise & MTI on GPU
    iq += (torch.randn(M,N, device=DEVICE) + 1j*torch.randn(M,N, device=DEVICE)) * 1e-6
    iq[1:] -= iq[:-1].clone(); iq[0] = 0
    
    # RD Map on GPU
    win = torch.hann_window(N, device=DEVICE) * torch.hann_window(M, device=DEVICE)[:,None]
    rd = torch.fft.fftshift(torch.fft.fft2(iq * win))
    return (20 * torch.log10(torch.abs(rd) + 1e-9)).cpu().numpy()

def otfs_torch(pts, its, vels, op: OTFSParams, rp: RadarParams):
    M, N = op.M, op.N
    H = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    if len(pts) > 0:
        pos = torch.tensor([0.,0.,rp.H], device=DEVICE)
        P = pts - pos; R = torch.norm(P, dim=1)
        vr = torch.sum(P/torch.clamp(R.unsqueeze(1), min=1e-3) * vels, dim=1)
        amp = torch.sqrt(rcs_torch(its)) / torch.clamp(R, min=1.0)**2
        
        # Grid mapping
        k = (R / (C0/(2*op.fs))).long()
        l = (vr / (C0/op.fc * op.fs/(2*N))).long() + M//2
        
        mask = (k>=0) & (k<N) & (l>=0) & (l<M)
        # Fast GPU accumulation using index_add_ or similar if needed, 
        # but simple loop might be okay if few points. Let's use scatter for speed.
        # Flatten indices for scatter: idx = l * N + k
        flat_idx = l[mask] * N + k[mask]
        H.view(-1).scatter_add_(0, flat_idx, amp[mask].to(torch.complex64))

    H += (torch.randn(M,N,device=DEVICE)+1j*torch.randn(M,N,device=DEVICE)) * 1e-5
    return (20*torch.log10(torch.abs(H)+1e-9)).cpu().numpy()

# ================= VIZ =================
def cfar_numpy(rd, thresh=20): # Keep CFAR on CPU for simplicity with scipy
    if not SCIPY: return np.zeros_like(rd, bool)
    k = np.ones((17,17)); k[6:-6,6:-6]=0; k/=k.sum()
    noise = ndi.convolve(rd, k, mode='mirror')
    return (rd > noise + thresh) & (np.abs(np.arange(rd.shape[0])-rd.shape[0]//2) > 2)[:,None]

# [Previous Viz classes Arrow3D, etc. remain the same]
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)

def viz_final(path_prefix, pts, rd_f, rd_o, gts, rp, op):
    pos_np = np.array([0,0,rp.H])
    
    # 1. BEV 3D
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    if len(pts)>0: 
        p_cpu = pts.cpu().numpy()[::5] # Subsample for plotting
        ax.scatter(p_cpu[:,0], p_cpu[:,1], p_cpu[:,2], s=0.5, c=p_cpu[:,2], alpha=0.3)
    
    # Draw complete cubes
    for gt in gts:
        c, s = np.array(gt['c']), np.array(gt['s'])
        # Generate all 8 corners
        dx, dy, dz = s[0]/2, s[1]/2, s[2]/2
        corners = np.array([[c[0]+i*dx, c[1]+j*dy, c[2]+k*dz] for i in [-1,1] for j in [-1,1] for k in [-1,1]])
        # Define the 12 edges by connecting specific corners
        edges = [[corners[0], corners[1]], [corners[0], corners[2]], [corners[0], corners[4]],
                 [corners[7], corners[6]], [corners[7], corners[5]], [corners[7], corners[3]],
                 [corners[2], corners[6]], [corners[2], corners[3]], [corners[1], corners[5]],
                 [corners[1], corners[3]], [corners[4], corners[5]], [corners[4], corners[6]]]
        ax.add_collection3d(Line3DCollection(edges, colors='r', linewidths=2))

    ax.set_xlim(0,70); ax.set_ylim(-35,35); ax.set_zlim(0,20)
    ax.view_init(30, -110); plt.savefig(f"{path_prefix}_bev.png"); plt.close()

    # 2. RD Maps
    for rd, params, name in [(rd_f, rp, "fmcw"), (rd_o, op, "otfs")]:
        plt.figure(figsize=(10,6))
        ra, va = params.ranges(), params.vels()
        plt.pcolormesh(ra, va, rd, cmap='jet', vmin=np.max(rd)-60, shading='auto')
        for gt in gts:
            p = np.array(gt['c'])-pos_np; r = np.linalg.norm(p); v = np.dot(p/r, gt['v'])
            if r < ra[-1] and abs(v) < abs(va[0]):
                plt.plot(r, v, 'wx', mew=2, ms=15) # White 'X' for visibility on jet
                plt.text(r+1, v, f"{r:.0f}m,{v:.0f}m/s", color='w', fontweight='bold')
        plt.colorbar(); plt.title(name.upper()); plt.savefig(f"{path_prefix}_{name}.png"); plt.close()

# ================= RUN =================
if __name__ == '__main__':
    root = "./output/torch_demo"
    os.makedirs(root, exist_ok=True)
    rp, op = RadarParams(), OTFSParams()
    
    # Dynamic scene
    gts = [{'c':[30, -10, 1], 's':[5,2,2], 'v':[12, 2, 0]},   # Car changing lane
           {'c':[50, 5, 1],   's':[3,3,3], 'v':[-15, 0, 0]}]  # Oncoming truck
           
    print("GPU Raycasting...")
    t0 = time.time()
    pts, its, vels = raycast_torch(rp, gts)
    torch.cuda.synchronize()
    print(f"Generated {len(pts)} points in {time.time()-t0:.3f}s")

    print("GPU Signal Processing...")
    rd_f = fmcw_torch(pts, its, vels, rp)
    rd_o = otfs_torch(pts, its, vels, op, rp)
    
    print("Visualizing...")
    viz_final(f"{root}/demo", pts, rd_f, rd_o, gts, rp, op)
    print("Finished.")