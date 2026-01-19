#!/usr/bin/env python
"""
AIradar_comm_model_g3.py - Separate Radar and Communication Models

Key differences from G2:
1. Radar and Communication are SEPARATE models (no joint training)
2. Independent training pipelines for each task
3. Expanded config support for better generalization
4. Higher capacity CommNet for 16-QAM

Usage:
    # Train radar only
    python AIradar_comm_model_g3.py --mode train_radar --epochs 30
    
    # Train communication only
    python AIradar_comm_model_g3.py --mode train_comm --epochs 50
    
    # Evaluate both
    python AIradar_comm_model_g3.py --mode eval_comprehensive
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Import dataset and configs from g2c
from AIradar_comm_model_g2c import (
    RADAR_COMM_CONFIGS_G2, CONFIG_ID_MAP, MAX_MOD_ORDER,
    ConfigEncoder, FiLMLayer, symbol_to_bits,
    compute_llr_loss,
    # Radar evaluation functions
    postprocess_radar, radar_metrics, cfar_metrics_from_g2,
    # Radar model from G2C (proven to work)
    GeneralizedRadarNet,
    # Use G2C dataset with proper ZF equalization and constellation normalization
    G2DeepDataset, g2_collate_fn
)
from AIradar_comm_dataset_g2 import AIRadar_Comm_Dataset_G2


# ==============================================================================
# RADAR MODEL (Standalone)
# ==============================================================================

class RadarNetG3(nn.Module):
    """
    Standalone Radar Detection Network.
    
    Input: Range-Doppler Map [B, 1, H, W]
    Output: Detection Heatmap [B, 1, H, W]
    """
    
    def __init__(self, base_ch=64, cond_dim=64):
        super().__init__()
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        
        # Encoder path
        self.enc1 = self._conv_block(1, base_ch)
        self.enc2 = self._conv_block(base_ch, base_ch * 2)
        self.enc3 = self._conv_block(base_ch * 2, base_ch * 4)
        
        # FiLM conditioning
        self.film1 = FiLMLayer(base_ch, cond_dim)
        self.film2 = FiLMLayer(base_ch * 2, cond_dim)
        self.film3 = FiLMLayer(base_ch * 4, cond_dim)
        
        # Decoder path
        self.dec3 = self._upconv_block(base_ch * 4, base_ch * 2)
        self.dec2 = self._upconv_block(base_ch * 4, base_ch)  # With skip
        self.dec1 = self._upconv_block(base_ch * 2, base_ch)   # With skip
        
        # Output head
        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, config_tensor):
        cond = self.config_encoder(config_tensor)
        input_size = x.shape[2:]  # Save for final resize
        
        # Encoder
        e1 = self.enc1(x)
        e1 = self.film1(e1, cond)
        
        e2 = F.max_pool2d(e1, 2)
        e2 = self.enc2(e2)
        e2 = self.film2(e2, cond)
        
        e3 = F.max_pool2d(e2, 2)
        e3 = self.enc3(e3)
        e3 = self.film3(e3, cond)
        
        # Decoder with skip connections
        d3 = self.dec3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.dec1(d2)
        out = self.out_conv(d1)
        
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out


# ==============================================================================
# COMMUNICATION MODEL (Standalone)
# ==============================================================================

class CommNetG3(nn.Module):
    """
    Standalone Communication Demapper Network.
    
    Enhanced per-pixel MLP with:
    - All 5 input channels (I, Q, H_mag, H_phase, SNR)
    - Deeper network for 16-QAM
    - Config embedding for multi-config generalization
    
    Input: [B, 5, H, W] equalized symbols + channel info
    Output: [B, 6, H, W] bit LLRs
    """
    MAX_BITS = 6
    
    def __init__(self, hidden_dim=256, in_channels=5, cond_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Config encoder for multi-config support
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        
        # Main demapper MLP (per-pixel)
        # Input: 5 features + cond_dim config embedding = 5 + 64 = 69
        total_in = in_channels + cond_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            
            nn.Linear(hidden_dim // 2, self.MAX_BITS),
        )
        
        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(1) * 5.0)
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Final layer with larger weights for stronger LLR output
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=3.0)
    
    def forward(self, x, config_tensor, mod_order=None, config_id=None):
        """
        Args:
            x: [B, C, H, W] where C >= 5 (I, Q, H_mag, H_phase, snr)
            config_tensor: [B, cond_dim] config embedding
        Returns:
            llr: [B, 6, H, W] bit LLRs
        """
        B, C, H, W = x.shape
        
        # Get config embedding: [B, cond_dim]
        cond = self.config_encoder(config_tensor)
        
        # Extract features: [B, in_channels, H, W]
        in_ch = min(C, self.in_channels)
        features = x[:, :in_ch]
        
        # Reshape for per-pixel processing: [B, H, W, in_ch]
        features = features.permute(0, 2, 3, 1).contiguous()
        
        # Flatten spatial: [B*H*W, in_ch]
        features_flat = features.view(B * H * W, in_ch)
        
        # Pad if needed
        if in_ch < self.in_channels:
            padding = torch.zeros(B * H * W, self.in_channels - in_ch, 
                                  device=features_flat.device)
            features_flat = torch.cat([features_flat, padding], dim=1)
        
        # Expand config to match spatial dims: [B, 1, 1, cond_dim] -> [B*H*W, cond_dim]
        cond_expanded = cond.view(B, 1, 1, -1).expand(B, H, W, -1).contiguous()
        cond_flat = cond_expanded.view(B * H * W, -1)
        
        # Concatenate features and config
        mlp_input = torch.cat([features_flat, cond_flat], dim=1)
        
        # MLP forward: [B*H*W, 6]
        llr_flat = self.mlp(mlp_input)
        
        # Reshape: [B, H, W, 6] -> [B, 6, H, W]
        llr = llr_flat.view(B, H, W, self.MAX_BITS).permute(0, 3, 1, 2)
        
        return llr * self.output_scale
    
    def get_symbol_logits(self, llr_logits, mod_order):
        """Convert LLR to symbol logits."""
        active_bits = int(np.log2(mod_order))
        active_llr = llr_logits[:, :active_bits]
        bit_probs = torch.sigmoid(active_llr)
        
        B, _, H, W = active_llr.shape
        symbol_logits = torch.zeros(B, mod_order, H, W, device=active_llr.device)
        
        for sym in range(mod_order):
            log_prob = 0
            for b in range(active_bits):
                bit_val = (sym >> b) & 1
                if bit_val == 1:
                    log_prob = log_prob + torch.log(bit_probs[:, b] + 1e-8)
                else:
                    log_prob = log_prob + torch.log(1 - bit_probs[:, b] + 1e-8)
            symbol_logits[:, sym] = log_prob
        
        return symbol_logits


# ==============================================================================
# DATASET (Reuse from G2 with minor modifications)
# ==============================================================================

class G3Dataset(Dataset):
    """Dataset for G3 with separate radar and comm outputs."""
    
    def __init__(self, config_name: str, num_samples: int, 
                 save_root: str, split: str = 'train',
                 target_size=(512, 512), radar_sigma=3.0,
                 enable_rf_impairments=True):
        super().__init__()
        self.config_name = config_name
        self.config = RADAR_COMM_CONFIGS_G2[config_name]
        self.target_size = target_size
        self.radar_sigma = radar_sigma
        self.config_id = CONFIG_ID_MAP[config_name]
        self.enable_rf_impairments = enable_rf_impairments
        
        # Cache path includes rf flag to avoid mixing data
        cache_dir = os.path.join(save_root, split, config_name)
        os.makedirs(cache_dir, exist_ok=True)
        rf_suffix = '_rf' if enable_rf_impairments else ''
        cache_file = os.path.join(cache_dir, f'cache_{num_samples}{rf_suffix}.pkl')
        
        if os.path.exists(cache_file):
            print(f"[Cache] Loading {config_name}/{split} from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.g2_ds = pickle.load(f)
        else:
            print(f"[Generate] Creating {config_name}/{split} ({num_samples} samples, RF={enable_rf_impairments})")
            self.g2_ds = AIRadar_Comm_Dataset_G2(
                config_name=config_name,
                num_samples=num_samples,
                save_path=cache_dir,
                drawfig=False,
                enable_clutter=True,
                enable_imperfect_csi=True,
                enable_rf_impairments=enable_rf_impairments
            )
            with open(cache_file, 'wb') as f:
                pickle.dump(self.g2_ds, f)
        
        # Config tensor
        self.config_tensor = self._build_config_tensor()
    
    def _build_config_tensor(self):
        """Build config tensor with physical parameters."""
        cfg = self.config
        return torch.tensor([
            cfg.get('fc', 77e9) / 1e9,
            cfg.get('bandwidth', 4e9) / 1e9,
            cfg.get('num_subcarriers', 64) / 64.0,
            cfg.get('mod_order', 16) / 64.0,
            cfg.get('snr_db', 20) / 30.0,
            cfg.get('range_resolution', 0.5),
            cfg.get('max_range', 100) / 100.0,
            cfg.get('max_velocity', 50) / 50.0,
        ], dtype=torch.float32)
    
    def __len__(self):
        return len(self.g2_ds)
    
    def __getitem__(self, idx):
        sample = self.g2_ds[idx]
        
        # Radar input (Range-Doppler Map)
        rdm = np.array(sample['range_doppler_map'])
        if rdm.ndim == 2:
            rdm = rdm[np.newaxis, :, :]
        rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min() + 1e-8)
        radar_input = torch.tensor(rdm, dtype=torch.float32)
        
        # Radar target (detection heatmap)
        # Targets are nested in target_info.targets in G2 format
        target_info = sample.get('target_info', {})
        targets = target_info.get('targets', sample.get('targets', []))
        r_axis = np.array(sample.get('range_axis', np.linspace(0, 100, rdm.shape[1])))
        v_axis = np.array(sample.get('velocity_axis', np.linspace(-50, 50, rdm.shape[2])))
        radar_target = self._create_heatmap(targets, r_axis, v_axis, rdm.shape[1:])
        
        # Comm input
        comm_info = sample.get('comm_info', {})
        mod_order = self.config.get('mod_order', 16)
        snr_db = comm_info.get('snr_db', self.config.get('snr_db', 20))
        
        # Use correct keys from AIRadar_Comm_Dataset_G2:
        # - rx_symbols: received symbols (equalized)
        # - channel_est: channel estimate
        eq_syms = np.array(comm_info.get('rx_symbols', comm_info.get('equalized_symbols', [])))
        H_est = np.array(comm_info.get('channel_est', comm_info.get('H_est', [])))
        
        fft_size = comm_info.get('fft_size', self.config.get('fft_size', 64))
        n_syms = self.config.get('num_ofdm_symbols', self.config.get('num_symbols', 14))
        
        if eq_syms.size >= n_syms * fft_size:
            eq_real = eq_syms.real.flatten()[:n_syms * fft_size].reshape(n_syms, fft_size)
            eq_imag = eq_syms.imag.flatten()[:n_syms * fft_size].reshape(n_syms, fft_size)
            
            if H_est.size >= fft_size:
                # Tile channel estimate across symbols if needed
                H_flat = np.tile(H_est.flatten()[:fft_size], n_syms).reshape(n_syms, fft_size)
            else:
                H_flat = np.ones((n_syms, fft_size), dtype=complex)
            
            H_mag = np.abs(H_flat)
            H_phase = np.angle(H_flat) / np.pi
            snr_normalized = snr_db / 30.0
            snr_channel = np.full_like(eq_real, snr_normalized)
            
            comm_input = torch.tensor(
                np.stack([eq_real, eq_imag, H_mag, H_phase, snr_channel], axis=0),
                dtype=torch.float32
            )
            
            tx_ints = np.array(comm_info.get('tx_ints', []), dtype=np.int64)
            if len(tx_ints) >= n_syms * fft_size:
                comm_target = torch.tensor(
                    tx_ints[:n_syms * fft_size].reshape(n_syms, fft_size),
                    dtype=torch.long
                )
            else:
                comm_target = torch.zeros(n_syms, fft_size, dtype=torch.long)
        else:
            comm_input = torch.zeros(5, n_syms, fft_size, dtype=torch.float32)
            comm_target = torch.zeros(n_syms, fft_size, dtype=torch.long)
        
        meta = {
            'config_id': self.config_id,
            'config_name': self.config_name,
            'config_tensor': self.config_tensor,
            'mod_order': mod_order,
            'snr_db': snr_db,
        }
        
        return radar_input, radar_target, comm_input, comm_target, meta
    
    def _create_heatmap(self, targets, r_axis, v_axis, shape):
        heatmap = np.zeros(shape, dtype=np.float32)
        for tgt in targets:
            r, v = tgt.get('range', 0), tgt.get('velocity', 0)
            r_idx = np.argmin(np.abs(r_axis - r))
            v_idx = np.argmin(np.abs(v_axis - v))
            if 0 <= r_idx < shape[0] and 0 <= v_idx < shape[1]:
                heatmap[r_idx, v_idx] = 1.0
        if self.radar_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.radar_sigma)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
        return torch.tensor(heatmap[np.newaxis, :, :], dtype=torch.float32)


def g3_collate_fn(batch):
    """Custom collate function for G3 dataset with simplified meta."""
    radar_inputs = torch.stack([b[0] for b in batch])
    radar_targets = torch.stack([b[1] for b in batch])
    comm_inputs = torch.stack([b[2] for b in batch])
    comm_targets = torch.stack([b[3] for b in batch])
    
    meta = {
        'config_id': torch.tensor([b[4]['config_id'] for b in batch]),
        'config_name': [b[4]['config_name'] for b in batch],
        'config_tensor': torch.stack([b[4]['config_tensor'] for b in batch]),
        'mod_order': torch.tensor([b[4]['mod_order'] for b in batch]),
        'snr_db': torch.tensor([b[4]['snr_db'] for b in batch], dtype=torch.float32),
    }
    
    return radar_inputs, radar_targets, comm_inputs, comm_targets, meta


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_radar_epoch(model, loaders, optimizer, device):
    """Train radar model for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pos_weight = torch.tensor([5.0], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for cfg_name, loader in loaders.items():
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            radar_in = radar_in.to(device)
            radar_tgt = radar_tgt.to(device)
            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
            
            optimizer.zero_grad()
            logits = model(radar_in, config_tensors)
            loss = bce(logits, radar_tgt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def train_comm_epoch(model, loaders, optimizer, device):
    """Train communication model for one epoch."""
    model.train()
    total_loss = 0.0
    total_ber = 0.0
    n_batches = 0
    
    per_cfg = {cfg: {'loss': 0, 'ber': 0, 'n': 0} for cfg in loaders}
    
    for cfg_name, loader in loaders.items():
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            comm_in = comm_in.to(device)
            comm_tgt = comm_tgt.to(device)
            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
            mod_order = int(meta['mod_order'][0])
            
            optimizer.zero_grad()
            llr = model(comm_in, config_tensors, mod_order)
            loss = compute_llr_loss(llr, comm_tgt, mod_order, lambda_comm=1.0)
            loss.backward()
            optimizer.step()
            
            # Compute BER
            with torch.no_grad():
                symbol_logits = model.get_symbol_logits(llr, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_tgt, mod_order)
                ber = (pred_bits != gt_bits).float().mean().item()
            
            total_loss += loss.item()
            total_ber += ber
            n_batches += 1
            
            per_cfg[cfg_name]['loss'] += loss.item()
            per_cfg[cfg_name]['ber'] += ber
            per_cfg[cfg_name]['n'] += 1
    
    # Average per-config stats
    for cfg in per_cfg:
        if per_cfg[cfg]['n'] > 0:
            per_cfg[cfg]['loss'] /= per_cfg[cfg]['n']
            per_cfg[cfg]['ber'] /= per_cfg[cfg]['n']
    
    return total_loss / n_batches, total_ber / n_batches, per_cfg


@torch.no_grad()
def evaluate_radar(model, loaders, device):
    """Evaluate radar detection performance."""
    model.eval()
    total_tp = total_fp = total_fn = 0
    
    for cfg_name, loader in loaders.items():
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            radar_in = radar_in.to(device)
            radar_tgt = radar_tgt.to(device)
            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
            
            logits = model(radar_in, config_tensors)
            preds = torch.sigmoid(logits) > 0.5
            targets = radar_tgt > 0.5
            
            total_tp += (preds & targets).sum().item()
            total_fp += (preds & ~targets).sum().item()
            total_fn += (~preds & targets).sum().item()
    
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


@torch.no_grad()
def evaluate_comm(model, loaders, device):
    """Evaluate communication BER."""
    model.eval()
    per_cfg = {}
    
    for cfg_name, loader in loaders.items():
        total_ber = 0.0
        n = 0
        
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            comm_in = comm_in.to(device)
            comm_tgt = comm_tgt.to(device)
            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
            mod_order = int(meta['mod_order'][0])
            
            llr = model(comm_in, config_tensors, mod_order)
            symbol_logits = model.get_symbol_logits(llr, mod_order)
            pred = symbol_logits.argmax(dim=1)
            
            pred_bits = symbol_to_bits(pred, mod_order)
            gt_bits = symbol_to_bits(comm_tgt, mod_order)
            ber = (pred_bits != gt_bits).float().mean().item()
            
            total_ber += ber
            n += 1
        
        per_cfg[cfg_name] = total_ber / n if n > 0 else 0.0
    
    return per_cfg


# ==============================================================================
# COMPREHENSIVE EVALUATION
# ==============================================================================

def plot_radar_comparison(results, xlabel, save_path, filename):
    """Plot DL vs CFAR F1/Precision/Recall comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get x-axis values from any of the possible keys
    x = results.get('snr', results.get('cnr', results.get('rcs', [])))
    
    # F1 Score
    axes[0].plot(x, results['dl_f1'], 'b-o', label='DL', linewidth=2, markersize=8)
    axes[0].plot(x, results['cfar_f1'], 'r--s', label='CFAR', linewidth=2, markersize=8)
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('F1 Score: DL vs CFAR', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Precision
    axes[1].plot(x, results['dl_prec'], 'b-o', label='DL', linewidth=2, markersize=8)
    axes[1].plot(x, results['cfar_prec'], 'r--s', label='CFAR', linewidth=2, markersize=8)
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision: DL vs CFAR', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    # Recall
    axes[2].plot(x, results['dl_rec'], 'b-o', label='DL', linewidth=2, markersize=8)
    axes[2].plot(x, results['cfar_rec'], 'r--s', label='CFAR', linewidth=2, markersize=8)
    axes[2].set_xlabel(xlabel, fontsize=12)
    axes[2].set_ylabel('Recall', fontsize=12)
    axes[2].set_title('Recall: DL vs CFAR', fontsize=14)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(save_path, filename)}")


@torch.no_grad()
def evaluate_radar_by_snr(model, device, config_name='CN0566_TRADITIONAL',
                          snr_list=[5, 10, 15, 20, 25, 30], num_samples=15, 
                          save_path='data/eval_radar'):
    """Evaluate DL radar vs CFAR at different SNR levels with full P/R/F1 metrics."""
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Radar Evaluation: DL vs CFAR by SNR")
    print(f"Config: {config_name}, SNR range: {snr_list} dB")
    print(f"{'='*60}\n")
    
    results = {'snr': [], 'dl_f1': [], 'dl_prec': [], 'dl_rec': [],
               'cfar_f1': [], 'cfar_prec': [], 'cfar_rec': []}
    
    config = RADAR_COMM_CONFIGS_G2[config_name]
    
    for snr_db in snr_list:
        print(f"SNR = {snr_db} dB...", end=" ")
        
        # Create dataset with specific SNR
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'snr_{snr_db}'),
            drawfig=False,
            fixed_snr=snr_db,
            enable_clutter=True,
            enable_imperfect_csi=True,
            enable_rf_impairments=True
        )
        
        dl_total = {'tp': 0, 'fp': 0, 'fn': 0}
        cfar_total = {'tp': 0, 'fp': 0, 'fn': 0}
        
        config_tensor = torch.tensor([
            config.get('fc', 77e9) / 1e9,
            config.get('bandwidth', 4e9) / 1e9,
            config.get('num_subcarriers', 64) / 64.0,
            config.get('mod_order', 16) / 64.0,
            snr_db / 30.0,
            config.get('range_resolution', 0.5),
            config.get('max_range', 100) / 100.0,
            config.get('max_velocity', 50) / 50.0,
        ], dtype=torch.float32).unsqueeze(0).to(device)
        
        for sample in ds:
            # Get RDM and targets
            rdm = np.array(sample['range_doppler_map'])
            if rdm.ndim == 2:
                rdm = rdm[np.newaxis, :, :]
            rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min() + 1e-8)
            radar_in = torch.tensor(rdm, dtype=torch.float32).unsqueeze(0).to(device)
            
            targets = sample.get('target_info', {}).get('targets', [])
            r_axis = np.array(sample.get('range_axis', np.linspace(0, 100, rdm.shape[1])))
            v_axis = np.array(sample.get('velocity_axis', np.linspace(-50, 50, rdm.shape[2])))
            
            # DL detection
            radar_logits = model(radar_in, config_tensor)
            probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
            dl_dets = postprocess_radar(probs, r_axis, v_axis)
            dl_m = radar_metrics(targets, dl_dets)
            dl_total['tp'] += dl_m['tp']
            dl_total['fp'] += dl_m['fp']
            dl_total['fn'] += dl_m['fn']
            
            # CFAR (from sample)
            cfar_m = cfar_metrics_from_g2(sample)
            cfar_total['tp'] += cfar_m['tp']
            cfar_total['fp'] += cfar_m['fp']
            cfar_total['fn'] += cfar_m['fn']
        
        # Compute metrics for this SNR
        dl_prec = dl_total['tp'] / (dl_total['tp'] + dl_total['fp'] + 1e-8)
        dl_rec = dl_total['tp'] / (dl_total['tp'] + dl_total['fn'] + 1e-8)
        dl_f1 = 2 * dl_prec * dl_rec / (dl_prec + dl_rec + 1e-8)
        
        cfar_prec = cfar_total['tp'] / (cfar_total['tp'] + cfar_total['fp'] + 1e-8)
        cfar_rec = cfar_total['tp'] / (cfar_total['tp'] + cfar_total['fn'] + 1e-8)
        cfar_f1 = 2 * cfar_prec * cfar_rec / (cfar_prec + cfar_rec + 1e-8)
        
        results['snr'].append(snr_db)
        results['dl_f1'].append(dl_f1)
        results['dl_prec'].append(dl_prec)
        results['dl_rec'].append(dl_rec)
        results['cfar_f1'].append(cfar_f1)
        results['cfar_prec'].append(cfar_prec)
        results['cfar_rec'].append(cfar_rec)
        
        print(f"DL F1={dl_f1:.3f}, CFAR F1={cfar_f1:.3f}")
    
    # Generate plots
    plot_radar_comparison(results, 'SNR (dB)', save_path, 'radar_snr_comparison.png')
    
    return results


def evaluate_comm_by_snr(model, config_name, device, save_path, snr_list=[5, 10, 15, 20, 25, 30],
                         channel_mode='realistic'):
    """Evaluate communication vs traditional by SNR with fixed SNR per evaluation point.
    
    Args:
        channel_mode: 'awgn' for clean AWGN channel (no clutter/fading/CSI error)
                      'realistic' for multipath + clutter + imperfect CSI
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    
    # Set channel parameters based on mode
    if channel_mode == 'awgn':
        enable_clutter = False
        enable_imperfect_csi = False
        enable_rf_impairments = False
        print(f"  [Channel Mode: AWGN - clean, no impairments]")
    else:  # realistic
        enable_clutter = True
        enable_imperfect_csi = True
        enable_rf_impairments = False  # RF impairments disabled for fair comparison
        print(f"  [Channel Mode: Realistic - multipath + clutter + CSI error]")
    
    results = {'snr': [], 'dl_ber': [], 'trad_ber': []}
    config = RADAR_COMM_CONFIGS_G2[config_name]
    mod_order = config.get('mod_order', 16)
    
    # Constellation scaling factors for normalization (same as G2DeepDataset)
    if mod_order == 4:
        scale_factor = np.sqrt(2)   # QPSK
    elif mod_order == 16:
        scale_factor = np.sqrt(10)  # 16-QAM
    else:
        scale_factor = np.sqrt(42)  # 64-QAM
    
    # Pre-compute config tensor
    config_tensor = ConfigEncoder.encode_config(config).unsqueeze(0).to(device)
    
    for snr_db in snr_list:
        # Generate fresh data for this specific SNR level
        snr_save_path = os.path.join(save_path, f'snr_{snr_db}_{channel_mode}')
        
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=15,
            save_path=snr_save_path,
            drawfig=False,
            fixed_snr=snr_db,
            enable_clutter=enable_clutter,
            enable_imperfect_csi=enable_imperfect_csi,
            enable_rf_impairments=enable_rf_impairments,
        )
        
        dl_bers = []
        trad_bers = []
        
        for sample in ds:
            comm_info = sample.get('comm_info', {})
            
            # Traditional BER from sample
            trad_bers.append(comm_info.get('ber', 0.5))
            
            # Get raw received symbols and channel estimate
            rx_symbols = np.array(comm_info.get('rx_symbols', []), dtype=np.complex64)
            channel_est = np.array(comm_info.get('channel_est', []))
            tx_ints = np.array(comm_info.get('tx_ints', []), dtype=np.int64)
            
            if len(rx_symbols) == 0:
                dl_bers.append(0.5)
                continue
            
            # Get grid dimensions from comm_info (same as G2DeepDataset)
            n_syms = comm_info.get('num_data_syms', 8)
            fft_size = comm_info.get('fft_size', len(rx_symbols) // n_syms) if n_syms > 0 else 256
            
            try:
                rx_grid = rx_symbols.reshape(n_syms, fft_size)
            except:
                total = len(rx_symbols)
                fft_size = min(256, total)
                n_syms = total // fft_size if fft_size > 0 else 1
                rx_grid = rx_symbols[:n_syms * fft_size].reshape(n_syms, fft_size)
            
            # Build channel estimate grid (same as G2DeepDataset)
            if channel_est is not None and len(channel_est) > 0:
                if len(channel_est) != fft_size:
                    from scipy.ndimage import zoom
                    H_est_resized = zoom(channel_est.real, fft_size/len(channel_est)) + \
                                   1j * zoom(channel_est.imag, fft_size/len(channel_est))
                else:
                    H_est_resized = channel_est
                H_grid = np.tile(H_est_resized[None, :], (n_syms, 1))
            else:
                H_grid = np.ones_like(rx_grid)
            
            # CONSTELLATION-AWARE NORMALIZATION (same as G2DeepDataset)
            # ZF equalization
            H_safe = np.where(np.abs(H_grid) > 1e-6, H_grid, 1e-6 + 0j)
            eq_symbols = rx_grid / H_safe
            
            # Scale by constellation normalization factor
            eq_real = eq_symbols.real / scale_factor
            eq_imag = eq_symbols.imag / scale_factor
            
            # Clip to prevent outliers
            eq_real = np.clip(eq_real, -3, 3)
            eq_imag = np.clip(eq_imag, -3, 3)
            
            # Normalize channel info (same as G2DeepDataset)
            H_mag = np.abs(H_grid) / (np.abs(H_grid).max() + 1e-6)
            H_phase = np.angle(H_grid) / np.pi
            
            # SNR channel - NOTE: G2DeepDataset uses snr_db/35.0
            snr_normalized = snr_db / 35.0
            snr_channel = np.full_like(eq_real, snr_normalized)
            
            # Build comm input [5, H, W]
            comm_in = torch.tensor(
                np.stack([eq_real, eq_imag, H_mag, H_phase, snr_channel], axis=0),
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # Target
            if len(tx_ints) >= n_syms * fft_size:
                comm_tgt = torch.tensor(
                    tx_ints[:n_syms * fft_size].reshape(n_syms, fft_size),
                    dtype=torch.long
                ).unsqueeze(0).to(device)
            else:
                comm_tgt = torch.zeros(1, n_syms, fft_size, dtype=torch.long, device=device)
            
            with torch.no_grad():
                llr = model(comm_in, config_tensor, mod_order)
                symbol_logits = model.get_symbol_logits(llr, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_tgt, mod_order)
                dl_bers.append((pred_bits != gt_bits).float().mean().item())
        
        results['snr'].append(snr_db)
        results['dl_ber'].append(np.mean(dl_bers))
        results['trad_ber'].append(np.mean(trad_bers))
        
        print(f"  SNR={snr_db}dB: DL={np.mean(dl_bers):.4e}, Trad={np.mean(trad_bers):.4e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(results['snr'], results['dl_ber'], 'b-o', linewidth=2, markersize=8, label='DL (G3)')
    plt.semilogy(results['snr'], results['trad_ber'], 'r--s', linewidth=2, markersize=8, label='Traditional')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title(f'BER vs SNR - {config_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.ylim([1e-3, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ber_vs_snr.png'), dpi=150)
    plt.close()
    
    return results


@torch.no_grad()
def evaluate_radar_by_cnr(model, device, config_name='CN0566_TRADITIONAL',
                          cnr_list=[0, 5, 10, 15, 20], num_samples=20, 
                          save_path='data/eval_cnr'):
    """Evaluate DL radar vs CFAR at different CNR (Clutter-to-Noise Ratio) levels.
    
    Higher CNR = stronger clutter = harder detection.
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Radar Evaluation: DL vs CFAR by CNR")
    print(f"Config: {config_name}, CNR range: {cnr_list} dB")
    print(f"{'='*60}\n")
    
    results = {'cnr': [], 'dl_f1': [], 'dl_prec': [], 'dl_rec': [],
               'cfar_f1': [], 'cfar_prec': [], 'cfar_rec': []}
    
    config = RADAR_COMM_CONFIGS_G2[config_name]
    
    for cnr_db in cnr_list:
        print(f"CNR = {cnr_db} dB...", end=" ")
        
        # Map CNR to clutter intensity
        clutter_intensity = 0.05 * (10 ** (cnr_db / 10))
        
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'cnr_{cnr_db}'),
            drawfig=False,
            clutter_intensity=clutter_intensity,
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        dl_total = {'tp': 0, 'fp': 0, 'fn': 0}
        cfar_total = {'tp': 0, 'fp': 0, 'fn': 0}
        
        config_tensor = torch.tensor([
            config.get('fc', 77e9) / 1e9,
            config.get('bandwidth', 4e9) / 1e9,
            config.get('num_subcarriers', 64) / 64.0,
            config.get('mod_order', 16) / 64.0,
            config.get('snr_db', 20) / 30.0,
            config.get('range_resolution', 0.5),
            config.get('max_range', 100) / 100.0,
            config.get('max_velocity', 50) / 50.0,
        ], dtype=torch.float32).unsqueeze(0).to(device)
        
        for sample in ds:
            rdm = np.array(sample['range_doppler_map'])
            if rdm.ndim == 2:
                rdm = rdm[np.newaxis, :, :]
            rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min() + 1e-8)
            radar_in = torch.tensor(rdm, dtype=torch.float32).unsqueeze(0).to(device)
            
            targets = sample.get('target_info', {}).get('targets', [])
            r_axis = np.array(sample.get('range_axis', np.linspace(0, 100, rdm.shape[1])))
            v_axis = np.array(sample.get('velocity_axis', np.linspace(-50, 50, rdm.shape[2])))
            
            radar_logits = model(radar_in, config_tensor)
            probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
            dl_dets = postprocess_radar(probs, r_axis, v_axis)
            dl_m = radar_metrics(targets, dl_dets)
            dl_total['tp'] += dl_m['tp']
            dl_total['fp'] += dl_m['fp']
            dl_total['fn'] += dl_m['fn']
            
            cfar_m = cfar_metrics_from_g2(sample)
            cfar_total['tp'] += cfar_m['tp']
            cfar_total['fp'] += cfar_m['fp']
            cfar_total['fn'] += cfar_m['fn']
        
        dl_prec = dl_total['tp'] / (dl_total['tp'] + dl_total['fp'] + 1e-8)
        dl_rec = dl_total['tp'] / (dl_total['tp'] + dl_total['fn'] + 1e-8)
        dl_f1 = 2 * dl_prec * dl_rec / (dl_prec + dl_rec + 1e-8)
        
        cfar_prec = cfar_total['tp'] / (cfar_total['tp'] + cfar_total['fp'] + 1e-8)
        cfar_rec = cfar_total['tp'] / (cfar_total['tp'] + cfar_total['fn'] + 1e-8)
        cfar_f1 = 2 * cfar_prec * cfar_rec / (cfar_prec + cfar_rec + 1e-8)
        
        results['cnr'].append(cnr_db)
        results['dl_f1'].append(dl_f1)
        results['dl_prec'].append(dl_prec)
        results['dl_rec'].append(dl_rec)
        results['cfar_f1'].append(cfar_f1)
        results['cfar_prec'].append(cfar_prec)
        results['cfar_rec'].append(cfar_rec)
        
        print(f"DL F1={dl_f1:.3f}, CFAR F1={cfar_f1:.3f}")
    
    plot_radar_comparison(results, 'CNR (dB)', save_path, 'radar_cnr_comparison.png')
    return results


@torch.no_grad()
def evaluate_radar_by_rcs(model, device, config_name='CN0566_TRADITIONAL',
                          rcs_list=[5, 10, 15, 20, 25], num_samples=20, 
                          save_path='data/eval_rcs'):
    """Evaluate DL radar vs CFAR at different RCS (Radar Cross Section) levels.
    
    Lower RCS = weaker targets = harder detection.
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Radar Evaluation: DL vs CFAR by RCS")
    print(f"Config: {config_name}, RCS range: {rcs_list} dB")
    print(f"{'='*60}\n")
    
    results = {'rcs': [], 'dl_f1': [], 'dl_prec': [], 'dl_rec': [],
               'cfar_f1': [], 'cfar_prec': [], 'cfar_rec': []}
    
    config = RADAR_COMM_CONFIGS_G2[config_name]
    
    for rcs_db in rcs_list:
        print(f"RCS = {rcs_db} dB...", end=" ")
        
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'rcs_{rcs_db}'),
            drawfig=False,
            target_rcs_range=(rcs_db - 2, rcs_db + 2),  # Narrow range around target
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        dl_total = {'tp': 0, 'fp': 0, 'fn': 0}
        cfar_total = {'tp': 0, 'fp': 0, 'fn': 0}
        
        config_tensor = torch.tensor([
            config.get('fc', 77e9) / 1e9,
            config.get('bandwidth', 4e9) / 1e9,
            config.get('num_subcarriers', 64) / 64.0,
            config.get('mod_order', 16) / 64.0,
            config.get('snr_db', 20) / 30.0,
            config.get('range_resolution', 0.5),
            config.get('max_range', 100) / 100.0,
            config.get('max_velocity', 50) / 50.0,
        ], dtype=torch.float32).unsqueeze(0).to(device)
        
        for sample in ds:
            rdm = np.array(sample['range_doppler_map'])
            if rdm.ndim == 2:
                rdm = rdm[np.newaxis, :, :]
            rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min() + 1e-8)
            radar_in = torch.tensor(rdm, dtype=torch.float32).unsqueeze(0).to(device)
            
            targets = sample.get('target_info', {}).get('targets', [])
            r_axis = np.array(sample.get('range_axis', np.linspace(0, 100, rdm.shape[1])))
            v_axis = np.array(sample.get('velocity_axis', np.linspace(-50, 50, rdm.shape[2])))
            
            radar_logits = model(radar_in, config_tensor)
            probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
            dl_dets = postprocess_radar(probs, r_axis, v_axis)
            dl_m = radar_metrics(targets, dl_dets)
            dl_total['tp'] += dl_m['tp']
            dl_total['fp'] += dl_m['fp']
            dl_total['fn'] += dl_m['fn']
            
            cfar_m = cfar_metrics_from_g2(sample)
            cfar_total['tp'] += cfar_m['tp']
            cfar_total['fp'] += cfar_m['fp']
            cfar_total['fn'] += cfar_m['fn']
        
        dl_prec = dl_total['tp'] / (dl_total['tp'] + dl_total['fp'] + 1e-8)
        dl_rec = dl_total['tp'] / (dl_total['tp'] + dl_total['fn'] + 1e-8)
        dl_f1 = 2 * dl_prec * dl_rec / (dl_prec + dl_rec + 1e-8)
        
        cfar_prec = cfar_total['tp'] / (cfar_total['tp'] + cfar_total['fp'] + 1e-8)
        cfar_rec = cfar_total['tp'] / (cfar_total['tp'] + cfar_total['fn'] + 1e-8)
        cfar_f1 = 2 * cfar_prec * cfar_rec / (cfar_prec + cfar_rec + 1e-8)
        
        results['rcs'].append(rcs_db)
        results['dl_f1'].append(dl_f1)
        results['dl_prec'].append(dl_prec)
        results['dl_rec'].append(dl_rec)
        results['cfar_f1'].append(cfar_f1)
        results['cfar_prec'].append(cfar_prec)
        results['cfar_rec'].append(cfar_rec)
        
        print(f"DL F1={dl_f1:.3f}, CFAR F1={cfar_f1:.3f}")
    
    plot_radar_comparison(results, 'RCS (dB)', save_path, 'radar_rcs_comparison.png')
    return results


def plot_ber_by_qam(all_qam_results, out_dir):
    """Plot BER vs SNR for all QAM modulations on single figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'4QAM': 'green', '16QAM': 'blue'}
    markers_dl = {'4QAM': 'o', '16QAM': 's'}
    markers_trad = {'4QAM': '^', '16QAM': 'd'}
    
    for qam_type, data in all_qam_results.items():
        color = colors.get(qam_type, 'gray')
        ax.semilogy(data['snr'], data['dl_ber'], f'-{markers_dl.get(qam_type, "o")}', 
                    color=color, label=f'{qam_type} DL', linewidth=2, markersize=6)
        ax.semilogy(data['snr'], data['trad_ber'], f'--{markers_trad.get(qam_type, "s")}', 
                    color=color, label=f'{qam_type} Traditional', linewidth=2, markersize=6, alpha=0.7)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('BER vs SNR: DL vs Traditional by QAM Order', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-4, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ber_vs_snr_all_qam.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(out_dir, 'ber_vs_snr_all_qam.png')}")


def generate_consolidated_report(all_results, out_dir):
    """Generate consolidated markdown evaluation report."""
    import datetime
    
    report = f"""# G3 Comprehensive Evaluation Report

**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Radar Detection: DL vs CFAR

"""
    radar_data = all_results.get('radar', {})
    if radar_data and 'snr' in radar_data:
        report += """### 1.1 Performance by SNR

| SNR (dB) | DL F1 | CFAR F1 | DL Precision | CFAR Precision | DL Recall | CFAR Recall |
|----------|-------|---------|--------------|----------------|-----------|-------------|
"""
        for i, snr in enumerate(radar_data.get('snr', [])):
            report += f"| {snr} | {radar_data['dl_f1'][i]:.3f} | {radar_data['cfar_f1'][i]:.3f} | {radar_data['dl_prec'][i]:.3f} | {radar_data['cfar_prec'][i]:.3f} | {radar_data['dl_rec'][i]:.3f} | {radar_data['cfar_rec'][i]:.3f} |\n"
        
        report += f"""
![Radar SNR Comparison](radar/radar_snr_comparison.png)

"""
    elif radar_data:
        report += f"""| Metric | DL | CFAR |
|--------|-----|------|
| F1 Score | {radar_data.get('dl_f1', 0):.3f} | {radar_data.get('cfar_f1', 0):.3f} |

"""
    else:
        report += "> [!WARNING]\n> Radar model not trained. Run `--mode train_radar` first.\n\n"
    
    # CNR evaluation
    radar_cnr_data = all_results.get('radar_cnr', {})
    if radar_cnr_data and 'cnr' in radar_cnr_data:
        report += """### 1.2 Performance by CNR (Clutter-to-Noise Ratio)
Higher CNR = stronger clutter = harder detection

| CNR (dB) | DL F1 | CFAR F1 | DL Precision | CFAR Precision | DL Recall | CFAR Recall |
|----------|-------|---------|--------------|----------------|-----------|-------------|
"""
        for i, cnr in enumerate(radar_cnr_data.get('cnr', [])):
            report += f"| {cnr} | {radar_cnr_data['dl_f1'][i]:.3f} | {radar_cnr_data['cfar_f1'][i]:.3f} | {radar_cnr_data['dl_prec'][i]:.3f} | {radar_cnr_data['cfar_prec'][i]:.3f} | {radar_cnr_data['dl_rec'][i]:.3f} | {radar_cnr_data['cfar_rec'][i]:.3f} |\n"
        
        report += f"""
![Radar CNR Comparison](radar_cnr/radar_cnr_comparison.png)

"""
    
    # RCS evaluation
    radar_rcs_data = all_results.get('radar_rcs', {})
    if radar_rcs_data and 'rcs' in radar_rcs_data:
        report += """### 1.3 Performance by RCS (Radar Cross Section)
Lower RCS = weaker targets = harder detection

| RCS (dB) | DL F1 | CFAR F1 | DL Precision | CFAR Precision | DL Recall | CFAR Recall |
|----------|-------|---------|--------------|----------------|-----------|-------------|
"""
        for i, rcs in enumerate(radar_rcs_data.get('rcs', [])):
            report += f"| {rcs} | {radar_rcs_data['dl_f1'][i]:.3f} | {radar_rcs_data['cfar_f1'][i]:.3f} | {radar_rcs_data['dl_prec'][i]:.3f} | {radar_rcs_data['cfar_prec'][i]:.3f} | {radar_rcs_data['dl_rec'][i]:.3f} | {radar_rcs_data['cfar_rec'][i]:.3f} |\n"
        
        report += f"""
![Radar RCS Comparison](radar_rcs/radar_rcs_comparison.png)

"""

    report += """---

## 2. Communication: DL vs Traditional (MMSE)

"""
    
    # 4-QAM results
    qam4_data = all_results.get('4QAM', {})
    if qam4_data:
        report += """### 2.1 4-QAM Performance

| SNR (dB) | DL BER | Traditional BER | Improvement |
|----------|--------|-----------------|-------------|
"""
        for i, snr in enumerate(qam4_data.get('snr', [])):
            dl_ber = qam4_data['dl_ber'][i]
            trad_ber = qam4_data['trad_ber'][i]
            improve = (1 - dl_ber/trad_ber) * 100 if trad_ber > 0 else 0
            report += f"| {snr} | {dl_ber:.4e} | {trad_ber:.4e} | {improve:.1f}% |\n"
        report += f"\n![4-QAM BER vs SNR](4QAM/ber_vs_snr.png)\n\n"

    # 16-QAM results  
    qam16_data = all_results.get('16QAM', {})
    if qam16_data:
        report += """### 2.2 16-QAM Performance

| SNR (dB) | DL BER | Traditional BER | Improvement |
|----------|--------|-----------------|-------------|
"""
        for i, snr in enumerate(qam16_data.get('snr', [])):
            dl_ber = qam16_data['dl_ber'][i]
            trad_ber = qam16_data['trad_ber'][i]
            improve = (1 - dl_ber/trad_ber) * 100 if trad_ber > 0 else 0
            report += f"| {snr} | {dl_ber:.4e} | {trad_ber:.4e} | {improve:.1f}% |\n"
        report += f"\n![16-QAM BER vs SNR](16QAM/ber_vs_snr.png)\n\n"

    report += """### 2.3 Combined QAM Comparison

![All QAM BER vs SNR](ber_vs_snr_all_qam.png)

---

## 3. Summary

"""
    # Calculate average improvement
    improvements = []
    for qam_data in [qam4_data, qam16_data]:
        if qam_data:
            for i in range(len(qam_data.get('snr', []))):
                dl = qam_data['dl_ber'][i]
                trad = qam_data['trad_ber'][i]
                if trad > 0:
                    improvements.append((1 - dl/trad) * 100)
    
    avg_improve = np.mean(improvements) if improvements else 0
    
    report += f"""- **Communication**: DL outperforms Traditional by **{avg_improve:.1f}%** average BER improvement
- **Best DL Performance**: 
"""
    
    if qam4_data and qam4_data.get('dl_ber'):
        best_4qam = min(qam4_data['dl_ber'])
        best_4qam_snr = qam4_data['snr'][qam4_data['dl_ber'].index(best_4qam)]
        report += f"  - 4-QAM: {best_4qam:.4e} at SNR={best_4qam_snr}dB\n"
    
    if qam16_data and qam16_data.get('dl_ber'):
        best_16qam = min(qam16_data['dl_ber'])
        best_16qam_snr = qam16_data['snr'][qam16_data['dl_ber'].index(best_16qam)]
        report += f"  - 16-QAM: {best_16qam:.4e} at SNR={best_16qam_snr}dB\n"
    
    # Save report
    report_path = os.path.join(out_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved evaluation report: {report_path}")
    
    return report


# ==============================================================================
# MAIN
# ==============================================================================

# Training configs - expanded for better generalization
RADAR_TRAIN_CONFIGS = [
    'CN0566_TRADITIONAL',
    'Automotive_77GHz_LongRange',
]

COMM_TRAIN_CONFIGS = {
    '4QAM': ['Automotive_77GHz_LongRange'],
    '16QAM': ['CN0566_TRADITIONAL'],
}


def main():
    parser = argparse.ArgumentParser(description='G3: Separate Radar and Comm Training')
    parser.add_argument('--mode', choices=[
        'train_radar', 'train_comm', 'train_both',
        'eval_radar', 'eval_comm', 'eval_comprehensive'
    ], default='eval_comprehensive')
    parser.add_argument('--train_samples', type=int, default=300)
    parser.add_argument('--val_samples', type=int, default=50)
    parser.add_argument('--data_root', type=str, default='data/AIradar_comm_model_g3')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--radar_ckpt', type=str, default=None)
    parser.add_argument('--comm_ckpt', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='data/AIradar_comm_model_g3')
    parser.add_argument('--channel_mode', choices=['awgn', 'realistic'], default='realistic',
                        help="Channel mode: 'awgn' (clean AWGN) or 'realistic' (multipath + clutter + CSI error)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ========== TRAIN RADAR ==========
    if args.mode in ['train_radar', 'train_both']:
        print("\n" + "="*60)
        print("TRAINING RADAR MODEL (Standalone)")
        print("="*60 + "\n")
        
        radar_model = GeneralizedRadarNet(in_ch=1, base_ch=48, cond_dim=64).to(device)
        print(f"Radar params: {sum(p.numel() for p in radar_model.parameters()):,}")
        
        # Load configs
        train_loaders = {}
        val_loaders = {}
        for cfg_name in RADAR_TRAIN_CONFIGS:
            train_ds = G2DeepDataset(cfg_name, args.train_samples, args.data_root, 'train')
            val_ds = G2DeepDataset(cfg_name, args.val_samples, args.data_root, 'val')
            train_loaders[cfg_name] = DataLoader(train_ds, batch_size=args.batch_size, 
                                                  shuffle=True, collate_fn=g2_collate_fn)
            val_loaders[cfg_name] = DataLoader(val_ds, batch_size=args.batch_size,
                                                shuffle=False, collate_fn=g2_collate_fn)
        
        optimizer = torch.optim.AdamW(radar_model.parameters(), lr=args.lr / 5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_f1 = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_radar_epoch(radar_model, train_loaders, optimizer, device)
            val_metrics = evaluate_radar(radar_model, val_loaders, device)
            scheduler.step()
            
            print(f"[Epoch {epoch:02d}] Loss={train_loss:.4f} | "
                  f"Val F1={val_metrics['f1']:.4f} P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save({
                    'model': radar_model.state_dict(),
                    'f1': best_f1,
                }, os.path.join(args.out_dir, 'radar_best.pt'))
                print(f"  -> Saved best radar model (F1={best_f1:.4f})")
    
    # ========== TRAIN COMM ==========
    if args.mode in ['train_comm', 'train_both']:
        print("\n" + "="*60)
        print("TRAINING COMMUNICATION MODEL (Standalone)")
        print("="*60 + "\n")
        
        comm_model = CommNetG3(hidden_dim=256, in_channels=5, cond_dim=64).to(device)
        print(f"Comm params: {sum(p.numel() for p in comm_model.parameters()):,}")
        
        # Load all comm configs
        all_configs = list(set(
            COMM_TRAIN_CONFIGS.get('4QAM', []) + 
            COMM_TRAIN_CONFIGS.get('16QAM', [])
        ))
        
        train_loaders = {}
        val_loaders = {}
        for cfg_name in all_configs:
            train_ds = G2DeepDataset(cfg_name, args.train_samples, args.data_root, 'train')
            val_ds = G2DeepDataset(cfg_name, args.val_samples, args.data_root, 'val')
            train_loaders[cfg_name] = DataLoader(train_ds, batch_size=args.batch_size,
                                                  shuffle=True, collate_fn=g2_collate_fn)
            val_loaders[cfg_name] = DataLoader(val_ds, batch_size=args.batch_size,
                                                shuffle=False, collate_fn=g2_collate_fn)
        
        optimizer = torch.optim.Adam(comm_model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_ber = float('inf')
        for epoch in range(1, args.epochs + 1):
            train_loss, train_ber, per_cfg = train_comm_epoch(comm_model, train_loaders, optimizer, device)
            val_ber = evaluate_comm(comm_model, val_loaders, device)
            scheduler.step()
            
            avg_val_ber = np.mean(list(val_ber.values()))
            print(f"[Epoch {epoch:02d}] Loss={train_loss:.4f} Train BER={train_ber:.4e} | Val BER={avg_val_ber:.4e}")
            for cfg in per_cfg:
                if per_cfg[cfg]['n'] > 0:
                    print(f"  {cfg}: Train BER={per_cfg[cfg]['ber']:.4e}, Val BER={val_ber.get(cfg, 0):.4e}")
            
            if avg_val_ber < best_ber:
                best_ber = avg_val_ber
                torch.save({
                    'model': comm_model.state_dict(),
                    'ber': best_ber,
                }, os.path.join(args.out_dir, 'comm_best.pt'))
                print(f"  -> Saved best comm model (BER={best_ber:.4e})")
    
    # ========== EVALUATE COMPREHENSIVE ==========
    if args.mode == 'eval_comprehensive':
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION")
        print("="*60 + "\n")
        
        # Load radar model
        radar_model = GeneralizedRadarNet(in_ch=1, base_ch=48, cond_dim=64).to(device)
        radar_ckpt = args.radar_ckpt or os.path.join(args.out_dir, 'radar_best.pt')
        if os.path.exists(radar_ckpt):
            ckpt = torch.load(radar_ckpt, map_location=device, weights_only=False)
            radar_model.load_state_dict(ckpt['model'])
            print(f"Loaded radar model from {radar_ckpt}")
        else:
            print(f"[Warning] Radar model not found at {radar_ckpt}, using untrained model")
        
        # Load comm model
        comm_model = CommNetG3(hidden_dim=256, in_channels=5, cond_dim=64).to(device)
        comm_ckpt = args.comm_ckpt or os.path.join(args.out_dir, 'comm_best.pt')
        if os.path.exists(comm_ckpt):
            ckpt = torch.load(comm_ckpt, map_location=device, weights_only=False)
            comm_model.load_state_dict(ckpt['model'])
            print(f"Loaded comm model from {comm_ckpt}")
        
        # Collect all results for report
        all_results = {}
        eval_dir = os.path.join(args.out_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        
        # ========== RADAR EVALUATION ==========
        print("\n" + "-"*60)
        print("[1/3] Radar Evaluation: DL vs CFAR by SNR")
        print("-"*60)
        radar_save = os.path.join(eval_dir, 'radar')
        radar_results = evaluate_radar_by_snr(
            radar_model, device, 'CN0566_TRADITIONAL',
            snr_list=[5, 10, 15, 20, 25, 30], num_samples=15,
            save_path=radar_save
        )
        all_results['radar'] = radar_results
        
        # ========== RADAR EVALUATION BY CNR ==========
        print("\n" + "-"*60)
        print("[1b/4] Radar Evaluation: DL vs CFAR by CNR")
        print("-"*60)
        radar_cnr_save = os.path.join(eval_dir, 'radar_cnr')
        radar_cnr_results = evaluate_radar_by_cnr(
            radar_model, device, 'CN0566_TRADITIONAL',
            cnr_list=[0, 5, 10, 15, 20], num_samples=15,
            save_path=radar_cnr_save
        )
        all_results['radar_cnr'] = radar_cnr_results
        
        # ========== RADAR EVALUATION BY RCS ==========
        print("\n" + "-"*60)
        print("[1c/4] Radar Evaluation: DL vs CFAR by RCS")
        print("-"*60)
        radar_rcs_save = os.path.join(eval_dir, 'radar_rcs')
        radar_rcs_results = evaluate_radar_by_rcs(
            radar_model, device, 'CN0566_TRADITIONAL',
            rcs_list=[5, 10, 15, 20, 25], num_samples=15,
            save_path=radar_rcs_save
        )
        all_results['radar_rcs'] = radar_rcs_results
        
        # ========== COMM EVALUATION ==========
        print("\n" + "-"*60)
        print("[2/4] Communication Evaluation: BER vs SNR")
        print("-"*60)
        qam_results = {}
        for qam_type, configs in COMM_TRAIN_CONFIGS.items():
            for config_name in configs:
                print(f"\n=== {qam_type}: {config_name} ===")
                save_path = os.path.join(eval_dir, qam_type)
                results = evaluate_comm_by_snr(comm_model, config_name, device, save_path,
                                               channel_mode=args.channel_mode)
                qam_results[qam_type] = results
                all_results[qam_type] = results
        
        # ========== GENERATE PLOTS AND REPORT ==========
        print("\n" + "-"*60)
        print("[3/4] Generating Report and Figures")
        print("-"*60)
        
        # Combined QAM plot
        plot_ber_by_qam(qam_results, eval_dir)
        
        # Generate markdown report
        generate_consolidated_report(all_results, eval_dir)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION COMPLETE")
        print(f"Report: {os.path.join(eval_dir, 'evaluation_report.md')}")
        print("="*60)


if __name__ == '__main__':
    main()
