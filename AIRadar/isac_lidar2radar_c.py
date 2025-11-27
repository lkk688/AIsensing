import os, numpy as np, matplotlib.pyplot as plt
import torch
from dataclasses import dataclass

try:
    import scipy.ndimage as ndi
    SCIPY = True
except ImportError:
    SCIPY = False
    print("Warning: Scipy not installed. Using NumPy-only ops.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {DEVICE} ---")
C0 = 299_792_458.0

# ================= PARAMS =================
@dataclass
class SystemParams:
    fc: float = 77e9
    B:  float = 150e6     # FMCW bandwidth (sets range resolution)
    fs: float = 150e6     # ADC sample-rate (>= B)
    M:  int   = 512       # chirps (Doppler bins)
    N:  int   = 512       # samples per chirp (Range FFT size)
    H:  float = 1.8
    az_fov: float = 60.0
    el_fov: float = 20.0
    bev_r_max: float = 50.0   # meters (BEV range clamp)

    @property
    def lambda_m(self): return C0 / self.fc
    @property
    def T_chirp(self):  return self.N / self.fs
    @property
    def slope(self):    return self.B / self.T_chirp  # S = B/T

    # Axes for FMCW processing used below (one-sided range, centered Doppler)
    def fmcw_axes(self):
        # Range: bins 0..N/2-1 -> R_k = c * (k*fs/N) / (2*S) == c*k/(2*B)
        ra = (C0 / (2.0 * self.B)) * np.arange(self.N // 2)
        # Doppler: f_d bins via slow-time PRF=1/T, then v = (λ/2) f_d
        f_d = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.T_chirp))
        va = (self.lambda_m / 2.0) * f_d
        return ra, va

    # Keep previous OTFS extents for consistency with your sim
    def otfs_axes(self):
        r = np.linspace(0, (C0 / (2 * self.fs)) * self.N, self.N)
        v = np.linspace(-(self.lambda_m/2)*(self.fs/(self.N*self.M))*(self.M/2),
                         (self.lambda_m/2)*(self.fs/(self.N*self.M))*(self.M/2), self.M)
        return r, v

# ================= Utils =================
def to_torch(x): return torch.tensor(x, device=DEVICE, dtype=torch.float32)

def _moving_sum_2d(a, r, c):
    if r == 0 and c == 0: return a.copy()
    ap = np.pad(a, ((r, r), (c, c)), mode='edge')
    S = ap.cumsum(axis=0).cumsum(axis=1)
    H, W = a.shape
    s22 = S[2*r:2*r+H, 2*c:2*c+W]
    s02 = S[0:H,       2*c:2*c+W]
    s20 = S[2*r:2*r+H, 0:W]
    s00 = S[0:H,       0:W]
    return s22 - s02 - s20 + s00

def nms2d(arr, kernel=3):
    k = max(3, int(kernel) | 1)
    pad = k // 2
    ap = np.pad(arr, ((pad, pad), (pad, pad)), mode='edge')
    max_nb = np.full_like(arr, -np.inf)
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            if di == 0 and dj == 0: continue
            view = ap[pad+di:pad+di+arr.shape[0], pad+dj:pad+dj+arr.shape[1]]
            max_nb = np.maximum(max_nb, view)
    return arr > max_nb

def cfar2d_ca(rd_db,
              train=(10, 8), guard=(2, 2),
              pfa=1e-4, min_snr_db=8.0,
              notch_doppler_bins=2,
              apply_nms=True, max_peaks=60,
              return_stats=False):
    rd_lin = 10.0 ** (rd_db / 10.0)
    H, W = rd_lin.shape
    mid = H // 2
    if notch_doppler_bins > 0:
        k = int(notch_doppler_bins)
        rd_lin[mid - k: mid + k + 1, :] = np.minimum(
            rd_lin[mid - k: mid + k + 1, :],
            np.percentile(rd_lin, 10)
        )
    Tr, Tc = train
    Gr, Gc = guard
    tot = _moving_sum_2d(rd_lin, Tr + Gr, Tc + Gc)
    gpl = _moving_sum_2d(rd_lin, Gr, Gc)
    train_sum = tot - gpl
    n_train = (2*(Tr+Gr)+1)*(2*(Tc+Gc)+1) - (2*Gr+1)*(2*Gc+1)
    noise = np.maximum(train_sum / max(n_train, 1), 1e-12)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    thresh = alpha * noise
    det = rd_lin > thresh
    snr_db = 10.0 * np.log10(np.maximum(rd_lin / noise, 1e-12))
    if min_snr_db and min_snr_db > 0:
        det &= snr_db >= min_snr_db
    if apply_nms:
        det &= nms2d(rd_lin, kernel=3)
    if max_peaks is not None and np.any(det):
        yy, xx = np.where(det)
        vals = rd_lin[yy, xx]
        if len(vals) > max_peaks:
            idx = np.argpartition(-vals, max_peaks - 1)[:max_peaks]
            keep = np.zeros_like(det, dtype=bool)
            keep[yy[idx], xx[idx]] = True
            det = keep
    if return_stats:
        return det, noise, snr_db
    return det

def plot_rd(ax, rd_db, ra, va, title, dynamic_db=35, percentile_clip=99.2, cmap='magma'):
    top = np.percentile(rd_db, percentile_clip)
    vmin = top - dynamic_db
    im = ax.imshow(rd_db, extent=[ra[0], ra[-1], va[0], va[-1]],
                   origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=top)
    ax.set_title(title)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    return im

# ================= Raycast & Sims =================
def raycast_torch(sp: SystemParams, gts):
    az = torch.linspace(np.deg2rad(-sp.az_fov/2), np.deg2rad(sp.az_fov/2), 1024, device=DEVICE)
    el = torch.linspace(np.deg2rad(-sp.el_fov/2), np.deg2rad(sp.el_fov/2), 128, device=DEVICE)
    EL, AZ = torch.meshgrid(el, az, indexing='ij')
    rays = torch.stack([torch.cos(EL)*torch.cos(AZ), torch.cos(EL)*torch.sin(AZ), torch.sin(EL)], dim=-1).reshape(-1, 3)
    pos = torch.tensor([0.,0.,sp.H], device=DEVICE)

    t_min = torch.full((rays.shape[0],), 100.0, device=DEVICE)
    hits_int = torch.zeros((rays.shape[0],), device=DEVICE)
    hits_vel = torch.zeros((rays.shape[0], 3), device=DEVICE)

    # ground plane
    mask_g = (rays[:, 2] < -2e-2)
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

# --------- FMCW with correct RD formation & axes ----------
def fmcw_torch(pts, its, vels, sp: SystemParams):
    M, N = sp.M, sp.N
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    if len(pts)==0:
        return (np.zeros((M, N//2)),)

    P = pts - to_torch([0,0,sp.H]); R = torch.norm(P, dim=1)
    mask = R > 0.1; P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    amp = torch.where(its==255, 1e6, 1e-1) / (R**2 + 1e-6)
    vr  = torch.sum(P/R.unsqueeze(1)*vels, dim=1)

    t_f = torch.arange(N, device=DEVICE)/sp.fs             # fast-time samples within chirp
    t_s = torch.arange(M, device=DEVICE)*sp.T_chirp        # slow-time (chirp index spacing)
    k_r = 2*sp.slope/C0                                    # beat freq = k_r * R
    k_v = 2/sp.lambda_m                                    # Doppler freq = k_v * v

    BATCH = 4096
    for i in range(0, len(R), BATCH):
        rb, vrb, ab = R[i:i+BATCH], vr[i:i+BATCH], amp[i:i+BATCH]
        phase = 2j*np.pi*( (k_r*rb[:,None,None])*t_f[None,None,:] + (k_v*vrb[:,None,None])*t_s[None,:,None] )
        iq += torch.sum(ab[:,None,None]*torch.exp(phase), dim=0)

    # noise + 1st-order MTI
    iq = iq + (torch.randn(M,N,device=DEVICE)+1j*torch.randn(M,N,device=DEVICE)) * 1e-4
    iq[1:] -= iq[:-1].clone(); iq[0]=0

    # window
    w_r = torch.hann_window(N,device=DEVICE)
    w_d = torch.hann_window(M,device=DEVICE)
    iq = iq * (w_d[:,None] * w_r[None,:])

    # Range FFT (one-sided 0..N/2-1), then Doppler FFT with fftshift
    RFFT = torch.fft.fft(iq, dim=1)
    RFFT = RFFT[:, :N//2]
    RD   = torch.fft.fftshift(torch.fft.fft(RFFT, dim=0), dim=0)

    RD_mag = torch.abs(RD).clamp_min(1e-12)
    rd_db = 20*torch.log10(RD_mag).cpu().numpy()
    return (rd_db,)

# --------- OTFS stays as before (toy mapping) ----------
def otfs_torch(pts, its, vels, sp: SystemParams):
    M, N = sp.M, sp.N
    H = torch.zeros((M,N), dtype=torch.complex64, device=DEVICE)
    if len(pts)==0: return (np.zeros((M,N)),)

    P = pts - to_torch([0,0,sp.H]); R = torch.norm(P, dim=1)
    mask = R > 0.1; P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    amp = torch.where(its==255, 1e6, 1e-1) / (R**2 + 1e-6)
    vr  = torch.sum(P/R.unsqueeze(1)*vels, dim=1)

    k_res = C0 / (2 * sp.fs)
    l_res = (sp.lambda_m / 2) * (sp.fs / (sp.M * sp.N))

    k = torch.clamp((R / k_res).long(), 0, N-1)
    l = torch.clamp((vr / l_res).long() + M//2, 0, M-1)
    H.view(-1).scatter_add_(0, (l*N + k).view(-1), amp.to(torch.complex64))
    H += (torch.randn(M,N,device=DEVICE)+1j*torch.randn(M,N,device=DEVICE))*1e-4

    rd_db = (20*torch.log10(torch.abs(H).clamp_min(1e-12))).cpu().numpy()
    return (rd_db,)

# ================= Visualization (2D/3D RD) =================
def viz_rd_2d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams, cfar_cfg=None):
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    pos = np.array([0,0,sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    if cfar_cfg is None:
        cfar_cfg = dict(train=(10, 8), guard=(2, 2), pfa=1e-4,
                        min_snr_db=8.0, notch_doppler_bins=2,
                        apply_nms=True, max_peaks=60)

    im = plot_rd(ax[0], rd_f_db, ra_f, va_f, "FMCW Range–Doppler", dynamic_db=35, percentile_clip=99.2, cmap='magma')
    plt.colorbar(im, ax=ax[0], label='dB')

    det_f, noise_f, snr_f = cfar2d_ca(rd_f_db, **cfar_cfg, return_stats=True)
    fy, fx = np.where(det_f)
    if fy.size:
        ax[0].scatter(ra_f[fx], va_f[fy], s=60, facecolors='none', edgecolors='cyan', linewidths=1.8, label='CFAR')

    im2 = plot_rd(ax[1], rd_o_db, ra_o, va_o, "OTFS Delay–Doppler", dynamic_db=35, percentile_clip=99.2, cmap='magma')
    plt.colorbar(im2, ax=ax[1], label='dB')

    for i, (ra, va, rd) in enumerate([(ra_f, va_f, rd_f_db), (ra_o, va_o, rd_o_db)]):
        for gt in gts:
            P = np.array(gt['c']) - pos
            r = np.linalg.norm(P)
            v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax[i].plot(r, v, 'wx', ms=10, mew=2, label='GT' if i==0 else "")
                ax[i].text(r+1, v+0.3, f"{r:.0f} m, {v:.1f} m/s", color='white', fontsize=9, weight='bold')

    for i in range(2):
        ax[i].grid(alpha=0.25, linestyle=':')
        ax[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

    return (det_f, ra_f, va_f, noise_f, snr_f)

def viz_rd_3d_compare(path, rd_f_db, rd_o_db, gts, sp: SystemParams):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(18,8))
    pos = np.array([0,0,sp.H])
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    for i, (rd, ra, va, name) in enumerate([(rd_f_db, ra_f, va_f, "FMCW"), (rd_o_db, ra_o, va_o, "OTFS")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        R, V = np.meshgrid(ra, va)
        floor = np.percentile(rd, 99.5) - 40
        surf = np.maximum(rd, floor)
        ax.plot_surface(R, V, surf, cmap='viridis', rstride=2, cstride=2, alpha=0.85)
        for gt in gts:
            P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax.scatter([r],[v],[np.max(rd)+5], c='r', marker='x', s=120, linewidths=2, zorder=10)
        ax.set_title(f"{name} 3D")
        ax.set_xlabel("Range (m)"); ax.set_ylabel("Velocity (m/s)")
        ax.set_xlim(0, ra[-1]); ax.set_ylim(va[0], va[-1]); ax.view_init(45, -110)

    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()

# ================= NEW: 3D RD with Detections vs GT =================
def extract_detections(rd_db, det_mask, ra, va, noise_db=None, snr_db=None):
    yy, xx = np.where(det_mask)
    dets = []
    for y, x in zip(yy, xx):
        det = {'r': float(ra[x]), 'v': float(va[y]), 'mag_db': float(rd_db[y, x])}
        if snr_db is not None: det['snr_db'] = float(snr_db[y, x])
        if noise_db is not None: det['noise_db'] = float(noise_db[y, x])
        dets.append(det)
    return dets

def viz_rd_3d_with_dets(path, rd_db, ra, va, det_mask, gts, sp: SystemParams, title="FMCW RD with Detections & GT"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    pos = np.array([0,0,sp.H])
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    R, V = np.meshgrid(ra, va)
    floor = np.percentile(rd_db, 99.5) - 40
    surf = np.maximum(rd_db, floor)
    ax.plot_surface(R, V, surf, cmap='viridis', rstride=2, cstride=2, alpha=0.85)

    # GT markers
    for gt in gts:
        P = np.array(gt['c']) - pos; r = np.linalg.norm(P); v = np.dot(P/r, gt['v'])
        if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
            ax.scatter([r],[v],[np.max(rd_db)+5], c='r', marker='x', s=140, linewidths=2, label='GT')

    # Detections as cyan spheres
    yx = np.where(det_mask)
    if yx[0].size:
        zvals = rd_db[yx]
        ax.scatter(ra[yx[1]], va[yx[0]], zvals, c='c', s=30, depthshade=True, label='CFAR')

    ax.set_title(title)
    ax.set_xlabel("Range (m)"); ax.set_ylabel("Velocity (m/s)"); ax.set_zlabel("Power (dB)")
    ax.set_xlim(0, ra[-1]); ax.set_ylim(va[0], va[-1]); ax.view_init(35, -115)
    if ax.get_legend_handles_labels()[1]: ax.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()

# ================= NEW: BEV scene + BEV overlay of detections =================
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def _gt_rv_az(gts, sp: SystemParams):
    pos = np.array([0.0, 0.0, sp.H])
    out = []
    for gt in gts:
        c = np.array(gt['c']); v = np.array(gt['v'])
        d = c - pos; r = np.linalg.norm(d)
        if r < 1e-6: continue
        u = d / r
        vr = float(np.dot(u, v))
        az = float(np.arctan2(c[1], c[0]))  # ground-plane azimuth
        out.append({'c': c, 'r': float(r), 'v': vr, 'az': az})
    return out

def _match_dets_to_gts(dets, gt_rv, w_r=1.0, w_v=0.5):
    used = set(); pairs = []; 
    for di, d in enumerate(dets):
        best_cost = 1e9; best_g = None
        for gi, g in enumerate(gt_rv):
            if gi in used: continue
            cost = w_r*abs(d['r'] - g['r']) + w_v*abs(d['v'] - g['v'])
            if cost < best_cost: best_cost = cost; best_g = gi
        if best_g is not None:
            used.add(best_g); pairs.append((d, gt_rv[best_g], best_cost))
    unpaired = [d for d in dets if all(d is not p[0] for p in pairs)]
    return pairs, unpaired

def viz_bev_scene(path_prefix, pts, gts, sp: SystemParams):
    """3D BEV-like scene: radar, raycast points, GT boxes, clamped to 0..50 m."""
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    radar = np.array([0,0,sp.H])
    ax.plot([radar[0]],[radar[1]],[radar[2]], 'ko', ms=8, label='Radar')

    if len(pts)>0:
        p = pts.detach().cpu().numpy()[::10]
        ax.scatter(p[:,0], p[:,1], p[:,2], s=0.5, c=p[:,2], alpha=0.3, cmap='viridis')
    for gt in gts:
        c, s = np.array(gt['c']), np.array(gt['s']); dx,dy,dz = s/2
        corn = np.array([[c[0]+i*dx, c[1]+j*dy, c[2]+k*dz] for i in [-1,1] for j in [-1,1] for k in [-1,1]])
        edges = [[corn[0], corn[1]], [corn[0], corn[2]], [corn[0], corn[4]], [corn[7], corn[6]],
                 [corn[7], corn[5]], [corn[7], corn[3]], [corn[2], corn[6]], [corn[2], corn[3]],
                 [corn[1], corn[5]], [corn[1], corn[3]], [corn[4], corn[5]], [corn[4], corn[6]]]
        ax.add_collection3d(Line3DCollection(edges, colors='r', lw=2))

    # Clamp to 0..50 m in X; +/- 25 m in Y
    ax.set_xlim(0, sp.bev_r_max); 
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2); 
    ax.set_zlim(0, 15)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title("3D Scene (Raycast, 0–50 m)")
    ax.view_init(30, -110)
    plt.tight_layout(); plt.savefig(f"{path_prefix}_bev_scene.png"); plt.close()

def viz_bev_dets_vs_gt(path_prefix, dets, gts, sp: SystemParams):
    """BEV comparison (detections placed at matched GT azimuth for visualization)."""
    gtinfo = _gt_rv_az(gts, sp)
    pairs, unpaired = _match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5)

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.scatter([0],[0], marker='*', s=120, c='k', label='Radar (XY)')

    # Range rings up to 50 m
    for rr in np.arange(10, sp.bev_r_max+1e-6, 10):
        circ = plt.Circle((0,0), rr, color='gray', fill=False, alpha=0.25, lw=0.8)
        ax.add_artist(circ)
        ax.text(rr, 0, f"{rr:.0f}m", color='gray', fontsize=8)

    # GT points
    for g in gtinfo:
        xg = g['r']*np.cos(g['az']); yg = g['r']*np.sin(g['az'])
        if 0 <= xg <= sp.bev_r_max and -sp.bev_r_max/2 <= yg <= sp.bev_r_max/2:
            ax.scatter([xg],[yg], c='r', s=60, marker='x', label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else None)

    # Matched detections at GT azimuth; line shows range error
    for det, g, cost in pairs:
        xd = det['r']*np.cos(g['az']); yd = det['r']*np.sin(g['az'])
        if 0 <= xd <= sp.bev_r_max and -sp.bev_r_max/2 <= yd <= sp.bev_r_max/2:
            ax.scatter([xd],[yd], facecolors='none', edgecolors='c', s=80, label='Det (CFAR)' if 'Det (CFAR)' not in ax.get_legend_handles_labels()[1] else None)
            xg = g['r']*np.cos(g['az']); yg = g['r']*np.sin(g['az'])
            ax.plot([xg, xd], [yg, yd], 'c--', linewidth=1)

    # Unpaired detections (project along +X)
    for d in unpaired:
        if 0 <= d['r'] <= sp.bev_r_max:
            ax.scatter([d['r']], [0], facecolors='none', edgecolors='orange', s=60,
                       label='Det (unmatched)' if 'Det (unmatched)' not in ax.get_legend_handles_labels()[1] else None)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, sp.bev_r_max); 
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title("BEV: CFAR Detections vs Ground Truth (0–50 m)")
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(f"{path_prefix}_bev_compare.png", dpi=180); plt.close()

from matplotlib.patches import Rectangle

def _gt_rv_az(gts, sp):
    pos = np.array([0.0, 0.0, sp.H])
    out = []
    for gt in gts:
        c = np.array(gt['c']); v = np.array(gt['v'])
        d = c - pos; r = np.linalg.norm(d)
        if r < 1e-6: continue
        u = d / r
        vr = float(np.dot(u, v))
        az = float(np.arctan2(c[1], c[0]))  # ground-plane azimuth
        out.append({'c': c, 'r': float(r), 'v': vr, 'az': az, 's': np.array(gt['s'])})
    return out

def _match_dets_to_gts(dets, gt_rv, w_r=1.0, w_v=0.5):
    used = []  # 允许多个检测匹配到同一 GT，用列表而不是 set（如果你想“一对一”，把它改回 set 即可）
    pairs = []
    for d in dets:
        best_g = None; best_cost = 1e9
        for gi, g in enumerate(gt_rv):
            cost = w_r*abs(d['r'] - g['r']) + w_v*abs(d['v'] - g['v'])
            if cost < best_cost:
                best_cost = cost; best_g = gi
        if best_g is not None:
            pairs.append((d, best_g, best_cost))
            used.append(best_g)
    return pairs

def _inside_cube_xy(x, y, g):
    cx, cy = g['c'][0], g['c'][1]
    sx, sy = g['s'][0], g['s'][1]
    return (cx - sx/2.0 <= x <= cx + sx/2.0) and (cy - sy/2.0 <= y <= cy + sy/2.0)

def _project_det_xy_using_gt_az(det, g):
    # 用匹配 GT 的方位角把 (range) 投影到 BEV
    az = g['az']
    x = det['r'] * np.cos(az)
    y = det['r'] * np.sin(az)
    return x, y

def _compute_metrics_from_pairs(pairs, gtinfo, sp):
    """
    pairs: list of (det, gi, cost), gi 为 gtinfo 的索引
    规则：检测点投影到匹配 GT 的 az；若 (x,y) 落在该 GT cube 的 XY 范围内 => TP，否则 FP。
    Recall：被至少一个 TP 覆盖的 GT / GT 总数。
    """
    TP, FP = 0, 0
    per_gt_tp = {i:0 for i in range(len(gtinfo))}
    tp_pts, fp_pts = [], []

    for det, gi, _ in pairs:
        g = gtinfo[gi]
        x, y = _project_det_xy_using_gt_az(det, g)
        # 限定 BEV 范围（0..sp.bev_r_max, -sp.bev_r_max/2..+sp.bev_r_max/2）
        if not (0 <= x <= sp.bev_r_max and -sp.bev_r_max/2 <= y <= sp.bev_r_max/2):
            # 画面之外也计为 FP（可按需改成忽略）
            FP += 1
            fp_pts.append((x,y,gi))
            continue
        if _inside_cube_xy(x, y, g):
            TP += 1
            per_gt_tp[gi] += 1
            tp_pts.append((x,y,gi))
        else:
            FP += 1
            fp_pts.append((x,y,gi))

    detected_gts = sum(1 for k,v in per_gt_tp.items() if v > 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = detected_gts / max(1, len(gtinfo))
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    metrics = dict(TP=TP, FP=FP, detected_gts=detected_gts,
                   total_gts=len(gtinfo), precision=precision, recall=recall, f1=f1)
    return metrics, tp_pts, fp_pts

def _draw_bev_panel(ax, dets, gts, sp, title="BEV", ring_step=10):
    gtinfo = _gt_rv_az(gts, sp)
    pairs  = _match_dets_to_gts(dets, gtinfo, w_r=1.0, w_v=0.5)
    metrics, tp_pts, fp_pts = _compute_metrics_from_pairs(pairs, gtinfo, sp)

    # 雷达与量程环
    ax.scatter([0],[0], marker='*', s=140, c='k', label='Radar')
    for rr in np.arange(ring_step, sp.bev_r_max+1e-6, ring_step):
        circ = plt.Circle((0,0), rr, color='gray', fill=False, alpha=0.22, lw=0.8)
        ax.add_artist(circ)

    # 画 GT 的 XY footprint（矩形）
    for g in gtinfo:
        cx, cy = g['c'][0], g['c'][1]
        sx, sy = g['s'][0], g['s'][1]
        rect = Rectangle((cx - sx/2, cy - sy/2), sx, sy, linewidth=1.8,
                         edgecolor='r', facecolor='none', alpha=0.9, label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else None)
        ax.add_patch(rect)
        ax.plot([cx],[cy],'rx',ms=6)

    # 画 TP / FP
    if tp_pts:
        ax.scatter([p[0] for p in tp_pts], [p[1] for p in tp_pts],
                   s=64, facecolors='none', edgecolors='lime', linewidths=2.0,
                   label='TP (in cube)')
    if fp_pts:
        ax.scatter([p[0] for p in fp_pts], [p[1] for p in fp_pts],
                   s=64, facecolors='none', edgecolors='orange', linewidths=2.0,
                   label='FP (out of cube)')

    # 轴与布局
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, sp.bev_r_max)
    ax.set_ylim(-sp.bev_r_max/2, sp.bev_r_max/2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='upper left')

    # 指标文本框
    txt = (f"TP: {metrics['TP']}   FP: {metrics['FP']}\n"
           f"Precision: {metrics['precision']:.2f}\n"
           f"Recall: {metrics['recall']:.2f} ({metrics['detected_gts']}/{metrics['total_gts']})\n"
           f"F1: {metrics['f1']:.2f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, linewidth=0.8))

    return metrics

def viz_scene_bev_compare(path, dets_fmcw, dets_otfs, gts, sp):
    """输出 scene_bev_compare.png：左 FMCW，右 OTFS，并在每个面板上写指标"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    _draw_bev_panel(axes[0], dets_fmcw, gts, sp, title='BEV (FMCW)')
    _draw_bev_panel(axes[1], dets_otfs, gts, sp, title='BEV (OTFS)')

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

# ================= Run =================
if __name__ == '__main__':
    # ============== paths & params ==============
    root = "./output/final_matched_viz"
    os.makedirs(root, exist_ok=True)
    sp = SystemParams()

    # 场景目标（保持与你之前一致；注意修正引号）
    gts = [
        {'c':[20,  0, 1], 's':[4,2,2], 'v':[ 12,  0, 0]},
        {'c':[50, -5, 1], 's':[5,3,3], 'v':[-18,  5, 0]}
    ]

    # ============== pipeline ==============
    print("Simulating raycast...")
    pts, its, vels = raycast_torch(sp, gts)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print(f"Raycast hits: {len(pts)}")

    print("Synthesizing FMCW & OTFS RD maps...")
    (rd_f_db,) = fmcw_torch(pts, its, vels, sp)   # shape (M, N//2)
    (rd_o_db,) = otfs_torch(pts, its, vels, sp)   # shape (M, N)

    print("Saving 2D/3D RD comparisons...")
    # 2D 对比（会返回 FMCW 的 CFAR mask 与坐标轴）
    det_f_mask, ra_f, va_f, noise_f, snr_f = viz_rd_2d_compare(
        f"{root}/compare_2d.png", rd_f_db, rd_o_db, gts, sp
    )
    # 3D 曲面对比（FMCW/OTFS）
    viz_rd_3d_compare(f"{root}/compare_3d.png", rd_f_db, rd_o_db, gts, sp)

    # ============== FMCW: 3D with dets ==============
    viz_rd_3d_with_dets(
        f"{root}/fmcw_3d_with_dets.png",
        rd_f_db, ra_f, va_f, det_f_mask, gts, sp,
        title="FMCW RD with Detections & GT"
    )

    # ============== OTFS: CFAR + 3D with dets ==============
    cfar_otfs_cfg = dict(
        train=(10, 8), guard=(2, 2),
        pfa=1e-4, min_snr_db=6.0,
        notch_doppler_bins=0,   # DD 域无直流杂波
        apply_nms=True, max_peaks=80
    )
    det_o_mask = cfar2d_ca(rd_o_db, **cfar_otfs_cfg)
    ra_o, va_o = sp.otfs_axes()

    viz_rd_3d_with_dets(
        f"{root}/otfs_3d_with_dets.png",
        rd_o_db, ra_o, va_o, det_o_mask, gts, sp,
        title="OTFS Delay–Doppler with Detections & GT"
    )

    # ============== BEV: 场景 & 对比评估 ==============
    print("Saving BEV figures...")
    # 场景 3D 视图（0–50m）
    viz_bev_scene(f"{root}/scene", pts, gts, sp)

    # 提取 FMCW/OTFS 的 (r,v) 检测，生成双面板 BEV（含 TP/FP/Precision/Recall/F1）
    dets_f = extract_detections(rd_f_db, det_f_mask, ra_f, va_f, snr_db=snr_f)
    dets_o = extract_detections(rd_o_db, det_o_mask, ra_o, va_o)

    viz_scene_bev_compare(
        f"{root}/scene_bev_compare.png",
        dets_f, dets_o, gts, sp
    )

    print("Done.")