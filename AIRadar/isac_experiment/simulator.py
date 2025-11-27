import numpy as np
import torch
from dataclasses import dataclass
from .config import DEVICE
from .utils import to_torch

@dataclass
class SceneDist:
    """Distribution parameters for random scene generation."""
    max_targets: int = 3
    r_min: float = 6.0
    r_max: float = 80.0
    az_deg: float = 60.0
    vx_min: float = -25.0
    vx_max: float =  25.0
    vy_min: float =  -6.0
    vy_max: float =   6.0
    cube_min: tuple = (2.0, 1.5, 1.5)  # (x,y,z) min size
    cube_max: tuple = (5.0, 3.0, 3.0)  # (x,y,z) max size

@dataclass
class ClutterDist:
    """Distribution parameters for clutter and artifacts."""
    ground_return_db: float = -25.0    # higher = stronger ground
    speckle_db: float = -35.0          # white-like floor
    ghost_prob: float = 0.15           # “mirror” ghost across Doppler
    rd_jitter_bins: int = 2            # random +/- bin jitter
    drop_stripe_prob: float = 0.1      # vertical/horizontal stripes

def raycast_torch(sp, gts):
    """
    Perform raycasting to simulate radar returns from targets and ground.
    
    Args:
        sp (SystemParams): System parameters.
        gts (list): List of ground truth target dictionaries.
        
    Returns:
        tuple: (hit_positions, hit_intensities, hit_velocities)
    """
    # Generate rays covering the Field of View
    az = torch.linspace(np.deg2rad(-sp.az_fov/2), np.deg2rad(sp.az_fov/2), 1024, device=DEVICE)
    el = torch.linspace(np.deg2rad(-sp.el_fov/2), np.deg2rad(sp.el_fov/2), 128, device=DEVICE)
    EL, AZ = torch.meshgrid(el, az, indexing='ij')
    
    # Ray direction vectors
    rays = torch.stack([torch.cos(EL)*torch.cos(AZ), torch.cos(EL)*torch.sin(AZ), torch.sin(EL)], dim=-1).reshape(-1, 3)
    pos = torch.tensor([0.,0.,sp.H], device=DEVICE)

    t_min = torch.full((rays.shape[0],), 100.0, device=DEVICE)
    hits_int = torch.zeros((rays.shape[0],), device=DEVICE)
    hits_vel = torch.zeros((rays.shape[0], 3), device=DEVICE)

    # 1. Ground plane intersection
    # Ray z-component must be negative to hit ground
    mask_g = (rays[:, 2] < -2e-2)
    t_g = -pos[2] / rays[:, 2]
    mask_valid_g = mask_g & (t_g > 0) & (t_g < t_min)
    t_min[mask_valid_g] = t_g[mask_valid_g]
    hits_int[mask_valid_g] = 100.0  # Base intensity for ground

    # 2. Target intersection (AABB - Axis Aligned Bounding Box)
    if gts:
        Cs = torch.stack([to_torch(gt['c']) for gt in gts])
        Ss = torch.stack([to_torch(gt['s']) for gt in gts])
        Vs = torch.stack([to_torch(gt['v']) for gt in gts])
        
        ro = pos.view(1,1,3)
        rd = rays.view(-1,1,3) + 1e-9 # Avoid div by zero
        
        # Slab method for AABB intersection
        t1 = (Cs - Ss/2 - ro) / rd
        t2 = (Cs + Ss/2 - ro) / rd
        
        tn = torch.max(torch.min(t1, t2), dim=-1)[0]
        tf = torch.min(torch.max(t1, t2), dim=-1)[0]
        
        mask_hit = (tn < tf) & (tn > 0)
        tn[~mask_hit] = np.inf
        
        # Find closest target hit
        min_t, min_idx = torch.min(tn, dim=1)
        mask_t = min_t < t_min
        
        t_min[mask_t] = min_t[mask_t]
        hits_int[mask_t] = 255.0  # Higher intensity for targets
        hits_vel[mask_t] = Vs[min_idx[mask_t]]

    mask = hits_int > 0
    return pos + t_min[mask].unsqueeze(1)*rays[mask], hits_int[mask], hits_vel[mask]

def rand_scene(sp, rng, sd: SceneDist):
    """Generate a random scene with targets."""
    K = rng.integers(1, sd.max_targets+1)
    gts = []
    for _ in range(K):
        r  = float(rng.uniform(sd.r_min, sd.r_max))
        az = float(rng.uniform(-np.deg2rad(sd.az_deg/2), np.deg2rad(sd.az_deg/2)))
        x, y = r*np.cos(az), r*np.sin(az)
        vx   = float(rng.uniform(sd.vx_min, sd.vx_max))
        vy   = float(rng.uniform(sd.vy_min, sd.vy_max))
        
        sx = float(rng.uniform(*sd.cube_min[:1]+sd.cube_max[:1])) if isinstance(sd.cube_min, tuple) else 4.0
        sy = float(rng.uniform(*sd.cube_min[1:2]+sd.cube_max[1:2])) if isinstance(sd.cube_min, tuple) else 2.0
        sz = float(rng.uniform(*sd.cube_min[2:3]+sd.cube_max[2:3])) if isinstance(sd.cube_min, tuple) else 2.0
        
        gts.append({'c':[x, y, 1.0], 's':[sx, sy, sz], 'v':[vx, vy, 0.0]})
    return gts

def apply_artifacts(rd_db, rng, cd: ClutterDist):
    """Apply synthetic radar artifacts (clutter, noise, ghosts) to the RD map."""
    rd = rd_db.copy()
    H, W = rd.shape
    
    # Add speckle noise
    rd += rng.normal(0, 1.0, rd.shape) * 0.0  # amplitude already in dB
    rd = np.maximum(rd, np.max(rd) + cd.speckle_db)
    
    # Add "ghosts" (Doppler wrapping/aliasing simulation)
    if rng.random() < cd.ghost_prob:
        shift = rng.integers(-5, 6)
        rd = np.maximum(rd, np.roll(rd, shift, axis=0) + cd.ground_return_db)
        
    # Stripe dropout/jitter (simulating interference or processing errors)
    if rng.random() < cd.drop_stripe_prob:
        if rng.random() < 0.5:  # vertical stripe
            col = rng.integers(0, W)
            rd[:, col] -= 20
        else:  # horizontal stripe
            row = rng.integers(0, H)
            rd[row, :] -= 20
            
    return rd
