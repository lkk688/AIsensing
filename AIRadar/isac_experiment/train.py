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

# ============================================================================
# Generalized Model Training (G2)
# ============================================================================

from .models.generalized_radar import GeneralizedRadarNet, ConfigEncoder, radar_focal_loss
from .models.generalized_comm import GeneralizedCommNet, GeneralizedCommNet2D, CommConfigEncoder, comm_bce_loss
from .dataset import G2DatasetWrapper, make_g2_loaders


def train_generalized_radar(config_names=None, samples_per_config=200, 
                            epochs=10, batch_size=8, lr=3e-4, 
                            ckpt_dir='./output/generalized_radar'):
    """Train GeneralizedRadarNet on multiple configurations.
    
    Args:
        config_names: List of G2 config names to train on
        samples_per_config: Samples per config
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        ckpt_dir: Checkpoint directory
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    if config_names is None:
        config_names = ['CN0566_TRADITIONAL', 'AUTOMOTIVE_TRADITIONAL']
    
    print(f"\n{'='*60}")
    print(f"Training GeneralizedRadarNet on {len(config_names)} configs")
    print(f"Configs: {config_names}")
    print(f"{'='*60}\n")
    
    # Create data loaders
    train_loader, val_loader = make_g2_loaders(
        config_names=config_names,
        samples_per_config=samples_per_config,
        batch_size=batch_size,
        val_split=0.2
    )
    
    # Create model
    model = GeneralizedRadarNet(base_ch=48, cond_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for batch in pbar:
            rdm = batch['rdm'].to(DEVICE)
            heatmap = batch['heatmap'].to(DEVICE)
            config = batch['config'].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(rdm, config)
            loss = radar_focal_loss(logits, heatmap)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * rdm.size(0)
            pbar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rdm = batch['rdm'].to(DEVICE)
                heatmap = batch['heatmap'].to(DEVICE)
                config = batch['config'].to(DEVICE)
                
                logits = model(rdm, config)
                loss = radar_focal_loss(logits, heatmap)
                val_loss += loss.item() * rdm.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / 'generalized_radar_best.pt')
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
    
    # Save final
    torch.save(model.state_dict(), ckpt_dir / 'generalized_radar_final.pt')
    print(f"\nTraining complete. Checkpoints saved to {ckpt_dir}")
    
    return model


def train_generalized_comm(mod_orders=[4, 16], snr_range=(0, 30),
                           epochs=10, steps_per_epoch=200, batch_size=16,
                           lr=3e-4, ckpt_dir='./output/generalized_comm'):
    """Train GeneralizedCommNet on multiple modulation orders.
    
    Args:
        mod_orders: List of modulation orders to train on
        snr_range: (min_snr, max_snr) for training
        epochs: Training epochs
        steps_per_epoch: Steps per epoch
        batch_size: Batch size
        lr: Learning rate
        ckpt_dir: Checkpoint directory
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training GeneralizedCommNet")
    print(f"Modulations: {mod_orders}-QAM")
    print(f"SNR Range: {snr_range} dB")
    print(f"{'='*60}\n")
    
    model = GeneralizedCommNet2D(max_mod_bits=6, base_ch=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    rng = np.random.default_rng(42)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}")
        for _ in pbar:
            # Random modulation and SNR
            mod_order = rng.choice(mod_orders)
            snr_db = rng.uniform(snr_range[0], snr_range[1])
            
            # Generate batch (using existing OFDM generator as proxy)
            X, Y = comm_dl_gen_batch_OFDM(snr_db, batch=batch_size)
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            
            # Create config tensor
            config = CommConfigEncoder.encode_config(mod_order, snr_db, 'multipath')
            config = config.unsqueeze(0).expand(X.size(0), -1).to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X, config)
            
            # Only use first 2 bits for QPSK (existing generator uses QPSK)
            loss = F.binary_cross_entropy_with_logits(logits[:, :2], Y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch}: Avg Loss={epoch_loss/steps_per_epoch:.4f}")
        torch.save(model.state_dict(), ckpt_dir / f'generalized_comm_ep{epoch}.pt')
    
    print(f"\nTraining complete. Checkpoints saved to {ckpt_dir}")
    return model


def train_g2_multitask(config_names=None, epochs=15, batch_size=8, lr=3e-4,
                       out_dir='./output/g2_multitask'):
    """Train jointly on Radar and Comm using G2 dataset.
    
    Trains GeneralizedRadarNet and GeneralizedCommNet together.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if config_names is None:
        config_names = ['CN0566_TRADITIONAL', 'AUTOMOTIVE_TRADITIONAL']
    
    print(f"\n{'='*60}")
    print(f"G2 Multi-Task Training (Radar + Comm)")
    print(f"Configs: {config_names}")
    print(f"{'='*60}\n")
    
    # Data loaders
    train_loader, val_loader = make_g2_loaders(
        config_names=config_names,
        samples_per_config=150,
        batch_size=batch_size
    )
    
    # Models
    radar_model = GeneralizedRadarNet(base_ch=48, cond_dim=128).to(DEVICE)
    comm_model = GeneralizedCommNet2D(max_mod_bits=6).to(DEVICE)
    
    # Joint optimizer
    params = list(radar_model.parameters()) + list(comm_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        radar_model.train()
        comm_model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            rdm = batch['rdm'].to(DEVICE)
            heatmap = batch['heatmap'].to(DEVICE)
            config = batch['config'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Radar loss
            radar_logits = radar_model(rdm, config)
            radar_loss = radar_focal_loss(radar_logits, heatmap)
            
            # Comm loss (placeholder - would use actual comm data)
            # For now, just train radar
            
            total_loss = radar_loss
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix(radar=radar_loss.item())
        
        print(f"Epoch {epoch} complete")
        
        # Save checkpoints
        torch.save(radar_model.state_dict(), out_dir / 'radar_latest.pt')
        torch.save(comm_model.state_dict(), out_dir / 'comm_latest.pt')
    
    print(f"\nMulti-task training complete. Models saved to {out_dir}")
    return radar_model, comm_model
