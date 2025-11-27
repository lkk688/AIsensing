import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .config import DEVICE
from .models.unet import UNetLite
from .models.cnn import CommDemapperCNN
from .models.multitask import RadarCommNet, calib_reg
from .models.losses import focal_bce_with_logits, radar_loss, comm_loss
from .dataset import RadarDiskDataset, RadarDiskDatasetModal, comm_dl_gen_batch_OFDM, comm_dl_gen_batch_OTFS

def make_radar_loaders(root, batch=6, workers=0):
    root = Path(root)
    tr = RadarDiskDataset(root/"radar"/"train")
    va = RadarDiskDataset(root/"radar"/"val")
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers)
    return dl_tr, dl_va

def make_radar_loaders_modal(root, sp, batch=6, workers=0, modality="fmcw", generate_otfs_on_the_fly=False):
    root = Path(root)
    tr = RadarDiskDatasetModal(root/"radar"/"train", sp, modality=modality, generate_otfs_on_the_fly=generate_otfs_on_the_fly)
    va = RadarDiskDatasetModal(root/"radar"/"val", sp, modality=modality, generate_otfs_on_the_fly=generate_otfs_on_the_fly)
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers)
    return dl_tr, dl_va

def train_radar_model(data_root, ckpt_dir, epochs=6, batch=6, lr=1e-3):
    """Train a standalone Radar UNet."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    net = UNetLite().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    
    dl_tr, dl_va = make_radar_loaders(data_root, batch=batch)
    
    best_loss = float('inf')
    
    for ep in range(1, epochs+1):
        net.train()
        loss_tr = 0.0
        pbar = tqdm(dl_tr, desc=f"Radar Ep {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = net(x)
            loss = focal_bce_with_logits(logits, y)
            loss.backward()
            opt.step()
            loss_tr += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())
            
        if len(dl_tr.dataset) > 0:
            loss_tr /= len(dl_tr.dataset)
        else:
            loss_tr = 0.0
        
        net.eval()
        loss_va = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = net(x)
                loss_va += focal_bce_with_logits(logits, y).item() * x.size(0)
        if len(dl_va.dataset) > 0:
            loss_va /= len(dl_va.dataset)
        else:
            loss_va = 0.0
        
        print(f"[Radar] Ep {ep}: Train {loss_tr:.4f}, Val {loss_va:.4f}")
        
        if loss_va < best_loss:
            best_loss = loss_va
            torch.save(net.state_dict(), ckpt_dir / "radar_unet_best.pt")
            
    return net

def train_comm_demap(model, gen_batch_fn, cfg, snr_min=0, snr_max=18, epochs=5, steps_per_epoch=200, lr=3e-4, tag="OFDM"):
    """Train a standalone Communication Demapper."""
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for ep in range(1, epochs+1):
        model.train()
        loss_ep = 0.0
        for _ in range(steps_per_epoch):
            eb = np.random.uniform(snr_min, snr_max)
            X, Y = gen_batch_fn(eb, **cfg)
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            
            opt.zero_grad()
            logits = model(X)
            loss = F.binary_cross_entropy_with_logits(logits, Y)
            loss.backward()
            opt.step()
            loss_ep += loss.item()
            
        print(f"[{tag} Demap] Ep {ep}/{epochs}: Loss {loss_ep/steps_per_epoch:.4f}")
    return model

def train_multidomain_multitask(data_root, sp, epochs=12, batch_radar=6, batch_comm_ofdm=8, batch_comm_otfs=6, lr=3e-4, out_root="./output"):
    """Train the joint RadarCommNet."""
    root = Path(out_root)
    (root/"checkpoints").mkdir(parents=True, exist_ok=True)
    
    dl_fm_tr, _ = make_radar_loaders_modal(data_root, sp, batch=batch_radar, modality="fmcw")
    dl_ot_tr, _ = make_radar_loaders_modal(data_root, sp, batch=batch_radar, modality="otfs")
    
    ofdm_cfg = dict(Nfft=256, cp_len=32, n_sym=8, batch=batch_comm_ofdm)
    otfs_cfg = dict(M=64, N=256, cp_len=32, batch=batch_comm_otfs)
    
    net = RadarCommNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    
    for ep in range(1, epochs+1):
        net.train()
        pbar = tqdm(range(max(len(dl_fm_tr), len(dl_ot_tr), 200)), desc=f"MDMT Ep {ep}/{epochs}")
        it_fm = iter(dl_fm_tr)
        it_ot = iter(dl_ot_tr)
        
        for _ in pbar:
            loss = 0.0
            
            # Radar FMCW
            try:
                x_f, y_f = next(it_fm)
                x_f, y_f = x_f.to(DEVICE), y_f.to(DEVICE)
                logits_f = net.forward_radar(x_f, domain="fmcw")
                loss += radar_loss(logits_f, y_f)
            except StopIteration:
                pass
                
            # Radar OTFS
            try:
                x_o, y_o = next(it_ot)
                x_o, y_o = x_o.to(DEVICE), y_o.to(DEVICE)
                logits_o = net.forward_radar(x_o, domain="otfs")
                loss += radar_loss(logits_o, y_o)
            except StopIteration:
                pass
                
            # Comm OFDM
            Xo, Yo = comm_dl_gen_batch_OFDM(ebn0_db=np.random.uniform(6, 16), **ofdm_cfg)
            Xo, Yo = Xo.to(DEVICE), Yo.to(DEVICE)
            logits_comm_ofdm = net.forward_comm(Xo, domain="ofdm")
            loss += 0.5 * comm_loss(logits_comm_ofdm, Yo)
            
            # Comm OTFS
            Xt, Yt = comm_dl_gen_batch_OTFS(ebn0_db=np.random.uniform(6, 16), **otfs_cfg)
            Xt, Yt = Xt.to(DEVICE), Yt.to(DEVICE)
            logits_comm_otfs = net.forward_comm(Xt, domain="otfs")
            loss += 0.5 * comm_loss(logits_comm_otfs, Yt)
            
            # Regularization
            loss += calib_reg(net)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))
            
        torch.save(net.state_dict(), root/"checkpoints"/"mdmt_latest.pt")
        
    return net
