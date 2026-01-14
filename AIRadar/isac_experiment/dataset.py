import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from .config import DEVICE
from .utils import to_torch
from .simulator import raycast_torch, rand_scene, apply_artifacts, SceneDist, ClutterDist
from .methods.fmcw import fmcw_torch
from .methods.otfs import otfs_torch
from .methods.comm_utils import rand_bits, awgn
from .methods.ofdm import ofdm_mod, ofdm_demod
from .methods.otfs import otfs_mod, otfs_demod, qpsk_gray_mod

def _rd_normalize(rd_db, top_p=99.5, dyn_db=40.0):
    """Normalize Range-Doppler map to [0, 1]."""
    top = np.percentile(rd_db, top_p)
    rd = np.clip(rd_db, top-dyn_db, top)
    rd = (rd - (top-dyn_db)) / dyn_db
    return rd.astype(np.float32)

def _heatmap_from_gts(shape, ra, va, gts, sp, sigma_pix=(2.0, 2.0)):
    """Generate Gaussian heatmap from ground truth targets."""
    H, W = shape
    pos = np.array([0, 0, sp.H])
    yy, xx = np.mgrid[0:H, 0:W]
    # Y = va[yy]
    # X = ra[xx]
    hm = np.zeros((H, W), np.float32)
    
    for gt in gts:
        P = np.array(gt['c']) - pos
        r = np.linalg.norm(P)
        v = np.dot(P/r, gt['v'])
        
        # Skip if out of bounds
        if not (0 <= r <= ra[-1] and va[0] <= v <= va[-1]): 
            continue
            
        # Nearest pixel index
        ix = np.searchsorted(ra, r)
        ix = np.clip(ix, 0, W-1)
        iy = np.searchsorted(va, v)
        iy = np.clip(iy, 0, H-1)
        
        # Gaussian splat
        sx, sy = sigma_pix[1], sigma_pix[0]
        g = np.exp(-((xx-ix)**2/(2*sx**2) + (yy-iy)**2/(2*sy**2)))
        hm = np.maximum(hm, g)
        
    return hm

class RadarSimDataset(torch.utils.data.Dataset):
    """On-the-fly Radar Simulation Dataset."""
    def __init__(self, sp, n_items=2000, rng_seed=123, min_targets=1, max_targets=3):
        self.sp = sp
        self.n = n_items
        self.rng = np.random.default_rng(rng_seed)
        self.sd = SceneDist(max_targets=max_targets)
        self.ra, self.va = sp.fmcw_axes()
        
    def __getitem__(self, idx):
        gts = rand_scene(self.sp, self.rng, self.sd)
        pts, its, vels = raycast_torch(self.sp, gts)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        (rd_db,) = fmcw_torch(pts, its, vels, self.sp)
        rd = _rd_normalize(rd_db)
        hm = _heatmap_from_gts(rd.shape, self.ra, self.va, gts, self.sp)
        
        x = torch.from_numpy(rd[None,...])     # (1,H,W)
        y = torch.from_numpy(hm[None,...])     # (1,H,W)
        return x, y
        
    def __len__(self): return self.n

class RadarDiskDataset(torch.utils.data.Dataset):
    """Dataset for loading pre-simulated radar data from disk."""
    def __init__(self, folder, normalize=True):
        self.files = sorted(Path(folder).glob("*.npz"))
        self.normalize = normalize
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        z = np.load(self.files[idx], allow_pickle=True)
        rd = z["rd_f_db"].astype(np.float32)
        hm = z["heatmap_f"].astype(np.float32) if "heatmap_f" in z else z["heatmap"].astype(np.float32)
        
        if self.normalize:
            rd = _rd_normalize(rd)
            
        x = torch.from_numpy(rd)[None,...]  # (1,H,W)
        y = torch.from_numpy(hm)[None,...]
        return x, y

class RadarDiskDatasetModal(torch.utils.data.Dataset):
    """
    Modality-aware Radar Dataset (FMCW or OTFS).
    If OTFS data is missing in .npz, it can be generated on-the-fly.
    """
    def __init__(self, folder, sp, modality="fmcw", normalize=True, generate_otfs_on_the_fly=False):
        self.files = sorted(Path(folder).glob("*.npz"))
        self.sp = sp
        self.modality = modality
        self.normalize = normalize
        self.gen_otfs = generate_otfs_on_the_fly

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx], allow_pickle=True)
        gts = json.loads(str(z["gts"]))

        if self.modality == "fmcw":
            rd = z["rd_f_db"].astype(np.float32)
            hm = (z["heatmap_f"] if "heatmap_f" in z else z["heatmap"]).astype(np.float32)
        else:
            if "rd_o_db" in z:
                rd = z["rd_o_db"].astype(np.float32)
            else:
                # Fallback: rebuild OTFS DD from gts
                pts, its, vels = raycast_torch(self.sp, gts)
                if DEVICE.type == "cuda": torch.cuda.synchronize()
                (rd,) = otfs_torch(pts, its, vels, self.sp)
                rd = rd.astype(np.float32)
                
            if "heatmap_o" in z:
                hm = z["heatmap_o"].astype(np.float32)
            else:
                ra_o, va_o = self.sp.otfs_axes()
                hm = _heatmap_from_gts(rd.shape, ra_o, va_o, gts, self.sp).astype(np.float32)

        if self.normalize:
            rd = _rd_normalize(rd)
            
        x = torch.from_numpy(rd)[None, ...]  # (1,H,W)
        y = torch.from_numpy(hm)[None, ...]
        return x, y

def _synth_one(idx, split, out_dir, sp, ebn0_db, seed, sd, cd, save_otfs=True):
    """Synthesize one radar sample and save to disk."""
    rng = np.random.default_rng(int(seed) + idx)
    gts = rand_scene(sp, rng, sd)

    pts, its, vels = raycast_torch(sp, gts)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    (rd_f_db,) = fmcw_torch(pts, its, vels, sp)
    ra_f, va_f = sp.fmcw_axes()
    heat_f = _heatmap_from_gts(rd_f_db.shape, ra_f, va_f, gts, sp)

    if save_otfs:
        (rd_o_db,) = otfs_torch(pts, its, vels, sp)
        ra_o, va_o = sp.otfs_axes()
        heat_o = _heatmap_from_gts(rd_o_db.shape, ra_o, va_o, gts, sp)
    else:
        rd_o_db = np.zeros((sp.M, sp.N), np.float32)
        heat_o = np.zeros_like(rd_o_db)

    # Apply artifacts
    rd_f_db = apply_artifacts(rd_f_db, rng, cd)
    if save_otfs:
        rd_o_db = apply_artifacts(rd_o_db, rng, cd)

    # Save
    save_path = Path(out_dir)/"radar"/split/f"{idx:07d}.npz"
    np.savez_compressed(
        save_path,
        rd_f_db=rd_f_db.astype(np.float32),
        heatmap_f=heat_f.astype(np.float32),
        rd_o_db=rd_o_db.astype(np.float32),
        heatmap_o=heat_o.astype(np.float32),
        gts=json.dumps(gts),
        ebn0_db=float(ebn0_db)
    )

def build_big_dataset(out_dir, sp, n_train=1000, n_val=200, seed=2025, save_otfs=True, overwrite=False):
    """Build a large dataset of simulated radar scenes."""
    out = Path(out_dir)
    (out/"radar"/"train").mkdir(parents=True, exist_ok=True)
    (out/"radar"/"val").mkdir(parents=True, exist_ok=True)

    if not overwrite:
        if any((out/"radar"/"train").glob("*.npz")):
            print(f"[DATA] Dataset exists at {out}. Skipping generation.")
            return

    sd = SceneDist()
    cd = ClutterDist()
    
    print(f"[DATA] Generating {n_train} train + {n_val} val samples...")
    
    # Simple sequential generation for now (to avoid multiprocessing complexity in this refactor)
    # Can be parallelized if needed.
    for i in tqdm(range(n_train), desc="Train Gen"):
        _synth_one(i, "train", out, sp, 10.0, seed, sd, cd, save_otfs)
        
    for i in tqdm(range(n_val), desc="Val Gen"):
        _synth_one(i, "val", out, sp, 10.0, seed, sd, cd, save_otfs)
        
    # Comm specs
    comm_dir = out/"comm"
    comm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    def _make_specs(n):
        return [{"ebn0_db": float(rng.choice(np.arange(0, 21, 2))), "seed": int(rng.integers(0, 1<<31))} for _ in range(n)]
        
    with open(comm_dir/"train_spec.json", "w") as f: json.dump(_make_specs(n_train), f)
    with open(comm_dir/"val_spec.json", "w") as f: json.dump(_make_specs(n_val), f)
    
    print("[DATA] Done.")

# --- Comm Data Generators ---

def _bits_to_qpsk_grid(bits, H, W):
    bits = bits.reshape(H*W, 2)
    syms = qpsk_gray_mod(bits)
    return syms.reshape(H, W)

def _grid_feats(S, use_mag_phase=False):
    if use_mag_phase:
        mag = np.abs(S); ang = np.angle(S)
        x = np.stack([S.real, S.imag, mag, ang], axis=0).astype(np.float32)
    else:
        x = np.stack([S.real, S.imag], axis=0).astype(np.float32)
    return x

def comm_dl_gen_batch_OFDM(ebn0_db, batch=8, Nfft=256, cp_len=32, n_sym=8, rng=None):
    if rng is None: rng = np.random.default_rng()
    bits_per_sym = 2
    H, W = n_sym, Nfft
    x_list, y_list = [], []
    
    for _ in range(batch):
        bits = rand_bits(H*W*bits_per_sym, rng)
        Xf = _bits_to_qpsk_grid(bits, H, W)
        
        tx = np.fft.ifft(Xf, n=W, axis=1, norm='ortho')
        if cp_len > 0:
            tx = np.concatenate([tx[:, -cp_len:], tx], axis=1)
            
        cp_ratio = cp_len / W
        rx = awgn(tx, ebn0_db, bits_per_sym=2, cp_ratio=cp_ratio, rng=rng)
        
        if cp_len > 0:
            rx = rx[:, cp_len:cp_len+W]
            
        Yf = np.fft.fft(rx, n=W, axis=1, norm='ortho')
        x = _grid_feats(Yf)
        y = bits.reshape(H, W, 2).transpose(2,0,1)
        
        x_list.append(x)
        y_list.append(y.astype(np.float32))
        
    return torch.from_numpy(np.stack(x_list)), torch.from_numpy(np.stack(y_list))

def comm_dl_gen_batch_OTFS(ebn0_db, batch=8, M=64, N=256, cp_len=32, rng=None):
    if rng is None: rng = np.random.default_rng()
    bits_per_sym = 2
    x_list, y_list = [], []
    
    for _ in range(batch):
        bits = rand_bits(M*N*bits_per_sym, rng)
        tx, Xdd = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)
        
        cp_ratio = cp_len / N
        rx = awgn(tx, ebn0_db, bits_per_sym=2, cp_ratio=cp_ratio, rng=rng)
        
        Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)
        
        x = _grid_feats(Ydd)
        y = bits.reshape(M, N, 2).transpose(2,0,1)
        
        x_list.append(x)
        y_list.append(y.astype(np.float32))
        
    return torch.from_numpy(np.stack(x_list)), torch.from_numpy(np.stack(y_list))

# ============================================================================
# G2 Dataset Integration
# ============================================================================

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add AIRadar to path

class G2DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper for AIradar_comm_dataset_g2 to integrate with isac_experiment.
    
    Provides unified interface for multiple radar/comm configurations.
    Returns normalized RDM, target heatmap, comm features, and config embedding.
    """
    
    def __init__(self, config_names=None, samples_per_config=100, snr_range=(10, 30),
                 target_size=(256, 256), include_comm=True):
        """
        Args:
            config_names: List of config names from RADAR_COMM_CONFIGS_G2
            samples_per_config: Number of samples per configuration
            snr_range: (min_snr, max_snr) for random SNR selection
            target_size: Target RDM size for normalization
            include_comm: Whether to include communication features
        """
        super().__init__()
        
        # Import G2 dataset
        try:
            from AIradar_comm_dataset_g2 import AIRadar_Comm_Dataset_G2, RADAR_COMM_CONFIGS_G2
            self.G2Dataset = AIRadar_Comm_Dataset_G2
            self.G2_CONFIGS = RADAR_COMM_CONFIGS_G2
        except ImportError:
            raise ImportError("AIradar_comm_dataset_g2 not found. Ensure it's in AIRadar folder.")
        
        if config_names is None:
            config_names = ['CN0566_TRADITIONAL', 'CN0566_OTFS_ISAC']
        
        self.config_names = config_names
        self.samples_per_config = samples_per_config
        self.snr_range = snr_range
        self.target_size = target_size
        self.include_comm = include_comm
        
        # Pre-generate samples for each config
        self.samples = []
        self.config_indices = []
        
        for config_idx, config_name in enumerate(config_names):
            print(f"[G2Wrapper] Loading {config_name}...")
            ds = self.G2Dataset(
                config_name=config_name,
                num_samples=samples_per_config,
                save_path=f'/tmp/g2_wrapper_{config_name}',
                drawfig=False,
                fixed_snr=None,  # Random SNR in range
                enable_clutter=True
            )
            
            for i in range(len(ds)):
                self.samples.append(ds[i])
                self.config_indices.append(config_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        config_idx = self.config_indices[idx]
        config_name = self.config_names[config_idx]
        config = self.G2_CONFIGS[config_name]
        
        # Get RDM and normalize
        rdm = sample['rd_map']  # (H, W) in dB
        rdm_norm = _rd_normalize(rdm)
        
        # Resize to target size if needed
        if rdm_norm.shape != self.target_size:
            rdm_tensor = torch.from_numpy(rdm_norm)[None, None, ...]
            rdm_tensor = torch.nn.functional.interpolate(
                rdm_tensor, size=self.target_size, mode='bilinear', align_corners=False
            )
            rdm_norm = rdm_tensor.squeeze().numpy()
        
        # Create target heatmap from CFAR detections
        targets = sample['target_info']['targets']
        r_axis = sample['r_axis']
        v_axis = sample['v_axis']
        
        # Simple heatmap from targets
        heatmap = np.zeros(self.target_size, dtype=np.float32)
        for t in targets:
            r_bin = int(t['range'] / r_axis[-1] * self.target_size[1])
            v_bin = int((t['velocity'] - v_axis[0]) / (v_axis[-1] - v_axis[0]) * self.target_size[0])
            r_bin = np.clip(r_bin, 0, self.target_size[1]-1)
            v_bin = np.clip(v_bin, 0, self.target_size[0]-1)
            
            # Gaussian splat
            yy, xx = np.mgrid[0:self.target_size[0], 0:self.target_size[1]]
            g = np.exp(-((xx-r_bin)**2/(2*3**2) + (yy-v_bin)**2/(2*3**2)))
            heatmap = np.maximum(heatmap, g.astype(np.float32))
        
        # Config embedding
        from .models.generalized_radar import ConfigEncoder
        config_tensor = ConfigEncoder.encode_config(config)
        
        result = {
            'rdm': torch.from_numpy(rdm_norm)[None, ...],  # (1, H, W)
            'heatmap': torch.from_numpy(heatmap)[None, ...],  # (1, H, W)
            'config': config_tensor,
            'config_idx': config_idx,
            'config_name': config_name,
        }
        
        # Include comm features if requested
        if self.include_comm and 'comm_info' in sample:
            comm = sample['comm_info']
            result['ber'] = comm.get('ber', 0.0)
            result['mod_order'] = config.get('mod_order', 16)
            result['tx_symbols'] = torch.from_numpy(np.array(comm.get('tx_symbols', [])))
            result['rx_symbols'] = torch.from_numpy(np.array(comm.get('rx_symbols', [])))
        
        return result


def make_g2_loaders(config_names=None, samples_per_config=100, batch_size=8, 
                    val_split=0.2, workers=0):
    """Create train and validation loaders for G2 dataset.
    
    Args:
        config_names: List of G2 config names
        samples_per_config: Samples per config
        batch_size: Batch size
        val_split: Fraction for validation
        workers: DataLoader workers
    
    Returns:
        train_loader, val_loader
    """
    ds = G2DatasetWrapper(
        config_names=config_names,
        samples_per_config=samples_per_config,
        target_size=(256, 256),
        include_comm=True
    )
    
    # Split into train/val
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=_g2_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=_g2_collate_fn
    )
    
    return train_loader, val_loader


def _g2_collate_fn(batch):
    """Custom collate function for G2 dataset."""
    rdm = torch.stack([b['rdm'] for b in batch])
    heatmap = torch.stack([b['heatmap'] for b in batch])
    config = torch.stack([b['config'] for b in batch])
    config_idx = torch.tensor([b['config_idx'] for b in batch])
    
    result = {
        'rdm': rdm,
        'heatmap': heatmap,
        'config': config,
        'config_idx': config_idx,
    }
    
    if 'ber' in batch[0]:
        result['ber'] = torch.tensor([b['ber'] for b in batch])
        result['mod_order'] = torch.tensor([b['mod_order'] for b in batch])
    
    return result
