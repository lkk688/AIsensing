#!/usr/bin/env python3
"""
Trains radar detection models on FMCW radar data from AIRadarDataset augmented
with RealisticChannelSimulator effects (TX-RX leakage, multipath, phase noise,
PLL settling).

Supports multiple model architectures:
  - RobustRadarNetG3: U-Net with residual blocks, SE attention, FiLM conditioning
  - RadarCRNN: 2D CNN encoder + bidirectional GRU across Doppler dimension
  - RadarTransformerNet: Vision Transformer with patch embedding
  - RadarDualPathNet: Dual-path (range + Doppler) processing with SE fusion

After training, evaluates DL vs CFAR across SNR and clutter conditions,
generates comparison plots, reports, and ROC curves (Pd vs Pfa).

The trained model integrates into myradar_all_in_one_v2.py via run_dl_detection().

Usage:
    # Full pipeline: train + compare + ROC
    conda run -n py312 python sdradi/myradar_all_train_v1.py --epochs 20

    # Train specific model
    conda run -n py312 python sdradi/myradar_all_train_v1.py --mode train --model RadarDualPathNet

    # ROC curves only (requires pre-trained model)
    conda run -n py312 python sdradi/myradar_all_train_v1.py --mode roc

    # Compare DL vs CFAR only
    conda run -n py312 python sdradi/myradar_all_train_v1.py --mode compare
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scipy.signal import convolve2d
from scipy.ndimage import percentile_filter

# Add paths - parent directory so 'sdradi' and 'AIRadar' are importable as packages
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'AIRadar'))

from sdradi.myradar_all_in_one_v2 import (
    AIRadarDataset, RealisticChannelSimulator, REALISTIC_CHANNEL_PARAMS
)


# ==============================================================================
# Model Components
# ==============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for config-aware adaptation."""
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)
        nn.init.ones_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.bias.data)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ConfigEncoder(nn.Module):
    """Encode radar configuration parameters to embedding vector."""
    def __init__(self, embed_dim=64, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResConvBlock(nn.Module):
    """Residual convolutional block with optional channel attention."""
    def __init__(self, in_ch, out_ch, use_attention=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.attn = ChannelAttention(out_ch) if use_attention else None

    def forward(self, x):
        out = self.conv(x)
        if self.attn is not None:
            out = self.attn(out)
        out = out + self.skip(x)
        return self.relu(out)


# ==============================================================================
# Model 1: RobustRadarNetG3 (U-Net with SE attention + FiLM)
# ==============================================================================

class RobustRadarNetG3(nn.Module):
    """
    Improved Radar Detection Network with:
    - Residual connections for better gradient flow
    - Channel attention (SE blocks) for feature selection
    - FiLM conditioning for multi-config generalization
    - Dropout for regularization against channel variations

    Input:  [B, 1, H, W] range-doppler map (normalized to [0,1])
    Config: [B, 8] radar configuration vector
    Output: [B, 1, H, W] detection heatmap (logits)
    """

    def __init__(self, base_ch=48, cond_dim=64, dropout=0.1):
        super().__init__()
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)

        # Encoder with residual blocks + attention
        self.enc1 = ResConvBlock(1, base_ch, use_attention=True)
        self.enc2 = ResConvBlock(base_ch, base_ch * 2, use_attention=True)
        self.enc3 = ResConvBlock(base_ch * 2, base_ch * 4, use_attention=True)

        # FiLM conditioning at each level
        self.film1 = FiLMLayer(base_ch, cond_dim)
        self.film2 = FiLMLayer(base_ch * 2, cond_dim)
        self.film3 = FiLMLayer(base_ch * 4, cond_dim)

        # Bottleneck
        self.bottleneck = ResConvBlock(base_ch * 4, base_ch * 4, use_attention=True)
        self.dropout = nn.Dropout2d(dropout)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = ResConvBlock(base_ch * 4, base_ch * 2)  # concat with enc2

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ResConvBlock(base_ch * 2, base_ch)  # concat with enc1

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 1, 1)
        )

    def forward(self, x, config_tensor):
        cond = self.config_encoder(config_tensor)
        input_size = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        e1 = self.film1(e1, cond)

        e2 = F.max_pool2d(e1, 2)
        e2 = self.enc2(e2)
        e2 = self.film2(e2, cond)

        e3 = F.max_pool2d(e2, 2)
        e3 = self.enc3(e3)
        e3 = self.film3(e3, cond)

        # Bottleneck
        b = self.bottleneck(e3)
        b = self.dropout(b)

        # Decoder
        d3 = self.up3(b)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.out_conv(d2)

        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out


# ==============================================================================
# Model 2: RadarCRNN (CNN encoder + Bidirectional GRU across Doppler)
# ==============================================================================

class RadarCRNN(nn.Module):
    """
    CNN + Bidirectional GRU for radar detection.

    Architecture:
    1. 2D CNN encoder reduces spatial dimensions and extracts features
    2. FiLM conditioning from radar config
    3. Bidirectional GRU across Doppler dimension (captures cross-Doppler patterns)
    4. 2D conv decoder reconstructs detection map

    Input:  [B, 1, 64, 1000] RDM
    Output: [B, 1, 64, 1000] detection logits
    """

    def __init__(self, base_ch=64, cond_dim=64, rnn_hidden=64, rnn_layers=2, dropout=0.1):
        super().__init__()
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)

        # 2D CNN encoder (reduces spatial dims by 4x)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, ch, 32, 500]
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, ch*2, 16, 250]
        )

        self.film = FiLMLayer(base_ch * 2, cond_dim)

        # Bidirectional GRU across Doppler dimension
        # At 1/4 resolution: D'=16, R'=250; process [B*R', D', C] sequences
        self.rnn_hidden = rnn_hidden
        self.gru = nn.GRU(
            input_size=base_ch * 2,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        # Decoder (upsamples back to input resolution)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(rnn_hidden * 2, base_ch, 2, stride=2),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, base_ch // 2, 2, stride=2),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_ch // 2, 1, 1)
        )

    def forward(self, x, config_tensor):
        B = x.shape[0]
        input_size = x.shape[2:]
        cond = self.config_encoder(config_tensor)

        # CNN encode
        feat = self.encoder(x)  # [B, C, D', R']
        feat = self.film(feat, cond)

        C, D, R = feat.shape[1], feat.shape[2], feat.shape[3]

        # GRU across Doppler for each range position
        # Reshape: [B, C, D, R] -> [B*R, D, C]
        feat_gru = feat.permute(0, 3, 2, 1).reshape(B * R, D, C)
        gru_out, _ = self.gru(feat_gru)  # [B*R, D, 2*hidden]

        # Reshape back: [B, 2*hidden, D, R]
        gru_out = gru_out.reshape(B, R, D, -1).permute(0, 3, 2, 1)

        # Decode
        out = self.decoder(gru_out)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


# ==============================================================================
# Model 3: RadarTransformerNet (Vision Transformer)
# ==============================================================================

class RadarTransformerNet(nn.Module):
    """
    Vision Transformer for radar detection.

    Architecture:
    1. Patch embedding: (8, 50) patches -> 160 tokens from [64, 1000]
    2. Learned positional encoding + config conditioning
    3. Transformer encoder (multi-head self-attention)
    4. Patch-wise decoder + refinement conv

    Input:  [B, 1, 64, 1000] RDM
    Output: [B, 1, 64, 1000] detection logits
    """

    def __init__(self, patch_size=(8, 50), embed_dim=256, num_heads=8, depth=4,
                 cond_dim=64, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)

        patch_dim = patch_size[0] * patch_size[1]

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Positional encoding for [64, 1000] with (8, 50) patches: 8*20=160 patches
        self.num_patches_d = 64 // patch_size[0]
        self.num_patches_r = 1000 // patch_size[1]
        num_patches = self.num_patches_d * self.num_patches_r
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Config projection for conditioning
        self.config_proj = nn.Linear(cond_dim, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Patch decoder
        self.patch_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_dim)
        )

        # Refinement conv on reassembled image
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, config_tensor):
        B, _, H, W = x.shape
        pH, pW = self.patch_size
        cond = self.config_encoder(config_tensor)

        # Create patches: [B, 1, H, W] -> [B, num_patches, patch_dim]
        nD = H // pH
        nR = W // pW
        # Reshape to extract patches
        patches = x.reshape(B, 1, nD, pH, nR, pW)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, nD * nR, pH * pW)

        # Embed patches + positional encoding
        tokens = self.patch_embed(patches) + self.pos_embed[:, :nD * nR]

        # Add config conditioning (broadcast to all tokens)
        config_token = self.config_proj(cond).unsqueeze(1)  # [B, 1, embed_dim]
        tokens = tokens + config_token

        # Transformer
        tokens = self.transformer(tokens)  # [B, num_patches, embed_dim]

        # Decode patches
        decoded = self.patch_decoder(tokens)  # [B, num_patches, patch_dim]
        decoded = decoded.reshape(B, nD, nR, pH, pW)

        # Reassemble image
        out = decoded.permute(0, 1, 3, 2, 4).reshape(B, nD * pH, nR * pW)
        out = out.unsqueeze(1)  # [B, 1, H', W']

        # Refine
        out = self.refine(out)

        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ==============================================================================
# Model 4: RadarDualPathNet (Dual-path range + Doppler processing)
# ==============================================================================

class RadarDualPathNet(nn.Module):
    """
    Dual-path processing network for radar detection.

    Architecture:
    1. Shared 2D encoder
    2. Range path: Dilated Conv1d along Range dimension
    3. Doppler path: Dilated Conv1d along Doppler dimension
    4. SE attention fusion + residual from encoder
    5. 2D decoder

    Input:  [B, 1, 64, 1000] RDM
    Output: [B, 1, 64, 1000] detection logits
    """

    def __init__(self, base_ch=64, cond_dim=64, num_dilated_blocks=3, dropout=0.1):
        super().__init__()
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)

        # Shared 2D encoder
        self.encoder = nn.Sequential(
            ResConvBlock(1, base_ch // 2),
            nn.MaxPool2d(2),
            ResConvBlock(base_ch // 2, base_ch),
        )

        self.film = FiLMLayer(base_ch, cond_dim)

        # Range path: dilated Conv1d along range dimension
        range_layers = []
        for i in range(num_dilated_blocks):
            dilation = 2 ** i
            range_layers.extend([
                nn.Conv1d(base_ch, base_ch, 7, padding=3 * dilation, dilation=dilation),
                nn.BatchNorm1d(base_ch),
                nn.ReLU(inplace=True),
            ])
        self.range_path = nn.Sequential(*range_layers)

        # Doppler path: dilated Conv1d along Doppler dimension
        doppler_layers = []
        for i in range(num_dilated_blocks):
            dilation = 2 ** i
            doppler_layers.extend([
                nn.Conv1d(base_ch, base_ch, 5, padding=2 * dilation, dilation=dilation),
                nn.BatchNorm1d(base_ch),
                nn.ReLU(inplace=True),
            ])
        self.doppler_path = nn.Sequential(*doppler_layers)

        # SE attention fusion
        self.fusion_attn = ChannelAttention(base_ch * 2)
        self.fusion_conv = nn.Conv2d(base_ch * 2, base_ch, 1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch // 2, 2, stride=2),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_ch // 2, 1, 1)
        )

    def forward(self, x, config_tensor):
        B = x.shape[0]
        input_size = x.shape[2:]
        cond = self.config_encoder(config_tensor)

        # Shared encoder: [B, 1, 64, 1000] -> [B, C, 32, 500]
        feat = self.encoder(x)
        feat = self.film(feat, cond)

        C, D, R = feat.shape[1], feat.shape[2], feat.shape[3]

        # Range path: process each Doppler row
        # [B, C, D, R] -> [B*D, C, R]
        range_in = feat.reshape(B * D, C, R)
        range_feat = self.range_path(range_in)  # [B*D, C, R]
        range_feat = range_feat.reshape(B, D, C, R).permute(0, 2, 1, 3)  # [B, C, D, R]

        # Doppler path: process each range column
        # [B, C, D, R] -> [B*R, C, D]
        doppler_in = feat.permute(0, 3, 1, 2).reshape(B * R, C, D)
        doppler_feat = self.doppler_path(doppler_in)  # [B*R, C, D]
        doppler_feat = doppler_feat.reshape(B, R, C, D).permute(0, 2, 3, 1)  # [B, C, D, R]

        # Fuse: concat + SE attention + residual
        fused = torch.cat([range_feat, doppler_feat], dim=1)  # [B, 2C, D, R]
        fused = self.fusion_attn(fused)
        fused = self.fusion_conv(fused)  # [B, C, D, R]
        fused = fused + feat  # Residual connection

        # Decode
        out = self.decoder(fused)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


# ==============================================================================
# Model Registry
# ==============================================================================

MODEL_REGISTRY = {
    'RobustRadarNetG3': (
        RobustRadarNetG3,
        {'base_ch': 48, 'cond_dim': 64, 'dropout': 0.15}
    ),
    'RadarCRNN': (
        RadarCRNN,
        {'base_ch': 64, 'cond_dim': 64, 'rnn_hidden': 64, 'rnn_layers': 2, 'dropout': 0.1}
    ),
    'RadarTransformerNet': (
        RadarTransformerNet,
        {'patch_size': (8, 50), 'embed_dim': 256, 'num_heads': 8, 'depth': 4,
         'cond_dim': 64, 'dropout': 0.1}
    ),
    'RadarDualPathNet': (
        RadarDualPathNet,
        {'base_ch': 64, 'cond_dim': 64, 'num_dilated_blocks': 3, 'dropout': 0.1}
    ),
}


# ==============================================================================
# Loss Functions
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced detection (mostly background)."""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=10.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            reduction='none'
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss for sparse binary masks.

    Dice loss handles extreme class imbalance (targets are <0.1% of pixels).
    Focal loss provides per-pixel hard-example mining.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=10.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)

        # Dice loss
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        dice_loss = 1.0 - dice

        return focal_loss + self.dice_weight * dice_loss


# ==============================================================================
# Pixel-level Metrics
# ==============================================================================

def compute_pixel_metrics(logits, targets, threshold=0.5):
    """Compute pixel-level TP, FP, FN, TN from model logits and binary targets."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()
        tn = ((1 - preds) * (1 - targets)).sum().item()
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ==============================================================================
# Dataset Wrapper with Realistic Channel Augmentation
# ==============================================================================

class RealisticRadarDataset(Dataset):
    """
    Wraps AIRadarDataset and applies RealisticChannelSimulator during training.
    Each __getitem__ returns normalized RDM + target mask.
    """

    def __init__(self, num_samples, config_name='config_cn0566',
                 apply_realistic_effects=True, clutter_intensity=0.1,
                 use_realistic_channel=True, channel_prob=0.7,
                 snr_min=5, snr_max=35, max_targets=3):
        self.use_realistic_channel = use_realistic_channel
        self.channel_prob = channel_prob

        self.dataset = AIRadarDataset(
            num_samples=num_samples,
            config_name=config_name,
            autogen=True,
            apply_realistic_effects=apply_realistic_effects,
            clutter_intensity=clutter_intensity,
            SNR_dB_min=snr_min,
            SNR_dB_max=snr_max,
            max_targets=max_targets,
            save_path=None,
            drawfig=False
        )

        if use_realistic_channel:
            self.channel_sim = RealisticChannelSimulator(
                REALISTIC_CHANNEL_PARAMS, enabled=True
            )
        else:
            self.channel_sim = None

        # Cache radar params
        self.fc = self.dataset.fc
        self.B = self.dataset.B
        self.T_chirp = self.dataset.T
        self.fs = self.dataset.fs
        self.slope = self.B / self.T_chirp
        self.range_resolution = self.dataset.range_resolution
        self.R_max = self.dataset.R_max
        self.v_max = self.dataset.v_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        rdm = sample['range_doppler_map']
        mask = sample['target_mask']
        targets = sample['target_info']['targets']

        if isinstance(rdm, torch.Tensor):
            rdm = rdm.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Optionally apply realistic channel effects with probability channel_prob
        if (self.channel_sim is not None and
                np.random.random() < self.channel_prob):
            td = sample['time_domain']
            if isinstance(td, torch.Tensor):
                td = td.numpy()
            # Convert to complex if stored as real/imag
            if td.ndim >= 2 and not np.iscomplexobj(td):
                if td.shape[-1] == 2:
                    td = td[..., 0] + 1j * td[..., 1]
            if td.ndim == 2:
                td = td[None, :, :]  # [1, Nc, Ns]
            # Apply channel
            td_aug = self.channel_sim.apply(
                td, targets, self.fs, self.T_chirp, self.B, self.fc, self.slope
            )
            # Recompute RDM from augmented time-domain
            if td_aug.ndim == 3:
                td_aug = td_aug[0]  # [Nc, Ns]
            rdm = self.dataset.compute_rdm(td_aug)

        # Normalize RDM to [0, 1]
        rdm_min = rdm.min()
        rdm_max = rdm.max()
        if rdm_max - rdm_min > 1e-6:
            rdm_norm = (rdm - rdm_min) / (rdm_max - rdm_min)
        else:
            rdm_norm = np.zeros_like(rdm)

        rdm_tensor = torch.from_numpy(rdm_norm).float().unsqueeze(0)  # [1, H, W]

        # Target mask: [H, W, 1] -> [1, H, W]
        if mask.ndim == 3:
            mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1)
        else:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # Config vector (8-dim, matching run_dl_detection)
        cfg = torch.tensor([
            self.fc / 1e9,
            self.B / 1e9,
            64.0 / 64.0,
            16.0 / 64.0,
            30.0 / 30.0,
            self.range_resolution,
            self.R_max / 100.0,
            self.v_max / 50.0
        ], dtype=torch.float32)

        return {
            'rdm': rdm_tensor,
            'mask': mask_tensor,
            'config': cfg,
            'targets': targets
        }

    def regenerate(self, clutter_intensity=None, snr_min=None, snr_max=None):
        """Regenerate the underlying dataset with new parameters."""
        if clutter_intensity is not None:
            self.dataset.clutter_intensity = clutter_intensity
        if snr_min is not None:
            self.dataset.SNR_dB_min = snr_min
        if snr_max is not None:
            self.dataset.SNR_dB_max = snr_max
        self.dataset.generate_dataset()


def collate_fn(batch):
    """Custom collate that handles variable-length target lists."""
    return {
        'rdm': torch.stack([b['rdm'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'config': torch.stack([b['config'] for b in batch]),
        'targets': [b['targets'] for b in batch]
    }


# ==============================================================================
# Training Pipeline
# ==============================================================================

def _save_val_visualization(model, val_loader, device, save_dir, epoch):
    """Save sample validation predictions as image."""
    model.eval()
    batch = next(iter(val_loader))
    rdm = batch['rdm'].to(device)
    mask = batch['mask'].to(device)
    cfg = batch['config'].to(device)

    with torch.no_grad():
        logits = model(rdm, cfg)
        heatmap = torch.sigmoid(logits)

    n = min(4, rdm.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(rdm[i, 0].cpu().numpy(), aspect='auto', cmap='viridis')
        axes[i, 0].set_title('RDM (normalized)')
        axes[i, 1].imshow(mask[i, 0].cpu().numpy(), aspect='auto', cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 2].imshow(heatmap[i, 0].cpu().numpy(), aspect='auto', cmap='hot')
        axes[i, 2].set_title('DL Prediction')

    fig.suptitle(f'Validation Samples - Epoch {epoch}', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, f'val_epoch_{epoch:03d}.png'), dpi=100)
    plt.close(fig)


def _plot_training_curves(history, best_epoch, save_dir):
    """Plot training curves including Pd/Pfa metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.5,
               label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history['lr'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    ax.set_title('LR Schedule'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history['pd'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Pd (Detection Probability)')
    ax.set_title('Validation Pd (Recall)'); ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = axes[1, 1]
    pfa_vals = [max(v, 1e-10) for v in history['pfa']]
    ax.semilogy(pfa_vals, 'r-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Pfa (False Alarm Rate)')
    ax.set_title('Validation Pfa'); ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=130)
    plt.close(fig)


def train_model(args):
    """Train radar detection model with improved pipeline.

    Features: model selection, loss selection, CosineAnnealingWarmRestarts,
    early stopping, per-epoch Pd/Pfa metrics, mixed precision, validation viz.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Model selection
    model_name = args.model
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        return None

    model_cls, default_cfg = MODEL_REGISTRY[model_name]
    model = model_cls(**default_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name} ({param_count:,} parameters)")

    # Loss selection
    if args.loss == 'dicefocal':
        criterion = DiceFocalLoss(alpha=0.25, gamma=2.0, pos_weight=15.0, dice_weight=0.5)
    else:
        criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=15.0)
    print(f"Loss: {args.loss}")

    # Create training dataset
    print("\nGenerating training dataset...")
    train_dataset = RealisticRadarDataset(
        num_samples=args.num_train,
        config_name='config_cn0566',
        apply_realistic_effects=True,
        clutter_intensity=0.15,
        use_realistic_channel=True,
        channel_prob=0.7,
        snr_min=5,
        snr_max=35,
        max_targets=3
    )

    # Create validation dataset (always with channel effects for fair eval)
    print("Generating validation dataset...")
    val_dataset = RealisticRadarDataset(
        num_samples=args.num_val,
        config_name='config_cn0566',
        apply_realistic_effects=True,
        clutter_intensity=0.2,
        use_realistic_channel=True,
        channel_prob=1.0,
        snr_min=10,
        snr_max=30,
        max_targets=3
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'pd': [], 'pfa': []}

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Batch: {args.batch_size}")
    print(f"  LR: {args.lr}, AMP: {use_amp}")
    print("-" * 60)

    for epoch in range(args.epochs):
        # Curriculum: increase difficulty every 5 epochs
        if epoch > 0 and epoch % 5 == 0:
            new_clutter = min(0.15 + epoch * 0.03, 0.5)
            new_snr_min = max(5 - epoch, -5)
            print(f"\n  [Curriculum] clutter={new_clutter:.2f}, snr_min={new_snr_min}dB")
            train_dataset.regenerate(
                clutter_intensity=new_clutter,
                snr_min=new_snr_min,
                snr_max=35
            )
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, collate_fn=collate_fn,
                num_workers=0, pin_memory=True
            )

        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            rdm = batch['rdm'].to(device)
            mask = batch['mask'].to(device)
            cfg = batch['config'].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(rdm, cfg)
                    loss = criterion(logits, mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(rdm, cfg)
                loss = criterion(logits, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validate with pixel metrics
        model.eval()
        val_loss = 0
        epoch_tp, epoch_fp, epoch_fn, epoch_tn = 0, 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                rdm = batch['rdm'].to(device)
                mask = batch['mask'].to(device)
                cfg = batch['config'].to(device)
                logits = model(rdm, cfg)
                loss = criterion(logits, mask)
                val_loss += loss.item()

                pm = compute_pixel_metrics(logits, mask)
                epoch_tp += pm['tp']
                epoch_fp += pm['fp']
                epoch_fn += pm['fn']
                epoch_tn += pm['tn']

        val_loss /= len(val_loader)
        pd = epoch_tp / max(epoch_tp + epoch_fn, 1)
        pfa = epoch_fp / max(epoch_fp + epoch_tn, 1)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)
        history['pd'].append(pd)
        history['pfa'].append(pfa)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'model_class': model_name,
                'config': default_cfg,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'pd': pd, 'pfa': pfa
            }, os.path.join(save_dir, 'radar_best_fmcw.pt'))
            improved = " *best*"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, "
              f"Pd={pd:.3f}, Pfa={pfa:.2e}, lr={lr:.2e}{improved}")

        # Validation visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            _save_val_visualization(model, val_loader, device, save_dir, epoch + 1)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # Save final model
    torch.save({
        'model': model.state_dict(),
        'model_class': model_name,
        'config': default_cfg,
        'epoch': epoch + 1,
        'val_loss': val_loss,
    }, os.path.join(save_dir, 'radar_final_fmcw.pt'))

    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    _plot_training_curves(history, best_epoch, save_dir)

    print(f"\nTraining complete. Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_dir}/radar_best_fmcw.pt")
    return model


# ==============================================================================
# DL Inference Helper
# ==============================================================================

def load_model(save_dir, device):
    """Load the best trained model (supports multiple architectures)."""
    ckpt_path = os.path.join(save_dir, 'radar_best_fmcw.pt')
    if not os.path.exists(ckpt_path):
        print(f"Model not found at {ckpt_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine model class from checkpoint
    model_name = checkpoint.get('model_class', 'RobustRadarNetG3')
    cfg = checkpoint.get('config', {'base_ch': 48, 'cond_dim': 64, 'dropout': 0.15})

    if model_name in MODEL_REGISTRY:
        model_cls, _ = MODEL_REGISTRY[model_name]
    else:
        model_cls = RobustRadarNetG3

    model = model_cls(**cfg).to(device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[DL] Loaded {model_name} from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def dl_detect(model, rdm_db, dataset, device, threshold=0.3):
    """Run DL detection on a single RDM (numpy [H, W] in dB)."""
    H, W = rdm_db.shape
    rdm_min, rdm_max = rdm_db.min(), rdm_db.max()
    if rdm_max - rdm_min > 1e-6:
        rdm_norm = (rdm_db - rdm_min) / (rdm_max - rdm_min)
    else:
        rdm_norm = np.zeros_like(rdm_db)

    inp = torch.from_numpy(rdm_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    cfg = torch.tensor([
        dataset.fc / 1e9, dataset.B / 1e9,
        64.0 / 64.0, 16.0 / 64.0, 30.0 / 30.0,
        dataset.range_resolution, dataset.R_max / 100.0, dataset.v_max / 50.0
    ], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp, cfg)
        heatmap = torch.sigmoid(logits).cpu().numpy().squeeze()

    # Peak detection with 3x3 NMS
    det_map = heatmap > threshold
    d_idxs, r_idxs = np.where(det_map)

    dr = dataset.range_axis[1] - dataset.range_axis[0] if len(dataset.range_axis) > 1 else 1.0
    dv = dataset.velocity_axis[1] - dataset.velocity_axis[0] if len(dataset.velocity_axis) > 1 else 1.0

    results = []
    for d_idx, r_idx in zip(d_idxs, r_idxs):
        val = heatmap[d_idx, r_idx]
        s_d, e_d = max(0, d_idx - 1), min(H, d_idx + 2)
        s_r, e_r = max(0, r_idx - 1), min(W, r_idx + 2)
        if val < np.max(heatmap[s_d:e_d, s_r:e_r]):
            continue
        results.append({
            'range_m': float(r_idx * dr),
            'velocity_mps': float((d_idx - H // 2) * dv),
            'range_idx': int(r_idx),
            'doppler_idx': int(d_idx),
            'magnitude': float(val * 100),
            'power': float(val * 100)
        })
    return results


# ==============================================================================
# Evaluation Helpers
# ==============================================================================

def match_detections(targets, detections, r_thresh=2.0, v_thresh=1.0):
    """Match detections to ground-truth targets."""
    tp = 0
    range_errors, vel_errors = [], []
    matched_det = set()

    for t in targets:
        best_dist = float('inf')
        best_j = -1
        for j, d in enumerate(detections):
            if j in matched_det:
                continue
            dr = abs(t['range'] - d['range_m'])
            dv = abs(t['velocity'] - d['velocity_mps'])
            if dr < r_thresh and dv < v_thresh:
                dist = dr + dv
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
        if best_j != -1:
            tp += 1
            matched_det.add(best_j)
            range_errors.append(abs(t['range'] - detections[best_j]['range_m']))
            vel_errors.append(abs(t['velocity'] - detections[best_j]['velocity_mps']))

    fp = len(detections) - len(matched_det)
    fn = len(targets) - tp

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'range_errors': range_errors, 'vel_errors': vel_errors
    }


def evaluate_method(dataset, detect_fn, num_samples=None):
    """Evaluate a detection method across all dataset samples."""
    n = min(num_samples or len(dataset), len(dataset))
    all_tp, all_fp, all_fn = 0, 0, 0
    all_range_err, all_vel_err = [], []

    for i in range(n):
        sample = dataset[i]
        rdm = sample['range_doppler_map']
        if isinstance(rdm, torch.Tensor):
            rdm = rdm.numpy()
        targets = sample['target_info']['targets']

        detections = detect_fn(rdm)
        m = match_detections(targets, detections)

        all_tp += m['tp']
        all_fp += m['fp']
        all_fn += m['fn']
        all_range_err.extend(m['range_errors'])
        all_vel_err.extend(m['vel_errors'])

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': all_tp, 'fp': all_fp, 'fn': all_fn,
        'range_rmse': float(np.mean(all_range_err)) if all_range_err else 0.0,
        'vel_rmse': float(np.mean(all_vel_err)) if all_vel_err else 0.0,
    }


# ==============================================================================
# ROC Curve System (Pixel-level Pd vs Pfa)
# ==============================================================================

def _cfar_noise_estimate(rdm_db, cfar_type='CA', num_train=12, num_guard=4):
    """Estimate per-cell CFAR noise level across the entire RDM.

    CA-CFAR: average of training cells (excluding guard cells) via 2D convolution.
    OS-CFAR: 75th percentile of training cells using vectorized sliding window.
    """
    H, W = rdm_db.shape
    half_win = num_guard + num_train

    if cfar_type == 'CA':
        # Build 2D kernel with 0s in guard+CUT region
        ks = 2 * half_win + 1
        kernel = np.ones((ks, ks))
        center = half_win
        kernel[center - num_guard:center + num_guard + 1,
               center - num_guard:center + num_guard + 1] = 0
        kernel /= kernel.sum()
        noise_est = convolve2d(rdm_db, kernel, mode='same', boundary='wrap')
    else:
        # OS-CFAR: 75th percentile along range dimension for each Doppler row
        # Vectorized using sliding_window_view
        from numpy.lib.stride_tricks import sliding_window_view
        noise_est = np.zeros((H, W))
        win_total = 2 * num_train  # number of training cells per CUT
        k_os = int(0.75 * win_total)  # 75th percentile index

        for d in range(H):
            row = np.pad(rdm_db[d], half_win, mode='wrap')
            windows = sliding_window_view(row, 2 * half_win + 1)  # [W, 2*half_win+1]
            # Extract training cells: left and right, excluding guard+CUT
            left = windows[:, :half_win - num_guard]
            right = windows[:, half_win + num_guard + 1:]
            training = np.concatenate([left, right], axis=1)  # [W, 2*num_train]
            sorted_training = np.sort(training, axis=1)
            noise_est[d] = sorted_training[:, k_os]

    return noise_est


def compute_cfar_pixel_roc(dataset, cfar_type='CA', num_train=12, num_guard=4,
                           threshold_offsets=None, num_samples=None):
    """Compute pixel-level ROC for CFAR detector by sweeping threshold offsets.

    Returns dict with 'thresholds', 'pd' (Pd array), 'pfa' (Pfa array).
    """
    if threshold_offsets is None:
        threshold_offsets = np.linspace(5, 40, 36)

    n = min(num_samples or len(dataset), len(dataset))

    tp_acc = np.zeros(len(threshold_offsets))
    fp_acc = np.zeros(len(threshold_offsets))
    fn_acc = np.zeros(len(threshold_offsets))
    tn_acc = np.zeros(len(threshold_offsets))

    for i in range(n):
        sample = dataset[i]
        rdm = sample['range_doppler_map']
        mask = sample['target_mask']

        if isinstance(rdm, torch.Tensor):
            rdm = rdm.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Convert to dB
        rdm_db = 10 * np.log10(np.abs(rdm) + 1e-12)
        noise_est = _cfar_noise_estimate(rdm_db, cfar_type, num_train, num_guard)

        gt = (mask > 0.5).astype(float)

        for j, offset in enumerate(threshold_offsets):
            det = (rdm_db > noise_est + offset).astype(float)
            tp_acc[j] += (det * gt).sum()
            fp_acc[j] += (det * (1 - gt)).sum()
            fn_acc[j] += ((1 - det) * gt).sum()
            tn_acc[j] += ((1 - det) * (1 - gt)).sum()

    pd = tp_acc / np.maximum(tp_acc + fn_acc, 1)
    pfa = fp_acc / np.maximum(fp_acc + tn_acc, 1)

    return {'thresholds': threshold_offsets, 'pd': pd, 'pfa': pfa}


def compute_dl_pixel_roc(model, ai_dataset, device, thresholds=None, num_samples=None):
    """Compute pixel-level ROC for DL model by sweeping detection thresholds.

    Args:
        model: Trained DL model
        ai_dataset: AIRadarDataset instance (raw, not wrapped)
        device: torch device
        thresholds: detection threshold values to sweep
        num_samples: number of samples to evaluate

    Returns dict with 'thresholds', 'pd', 'pfa'.
    """
    if thresholds is None:
        # Dense near extremes for better ROC curve resolution
        thresholds = np.concatenate([
            np.linspace(0.01, 0.1, 10),
            np.linspace(0.15, 0.85, 15),
            np.linspace(0.9, 0.99, 10)
        ])

    n = min(num_samples or len(ai_dataset), len(ai_dataset))

    # Build config tensor from dataset params
    cfg = torch.tensor([
        ai_dataset.fc / 1e9, ai_dataset.B / 1e9,
        64.0 / 64.0, 16.0 / 64.0, 30.0 / 30.0,
        ai_dataset.range_resolution, ai_dataset.R_max / 100.0,
        ai_dataset.v_max / 50.0
    ], dtype=torch.float32).unsqueeze(0).to(device)

    tp_acc = np.zeros(len(thresholds))
    fp_acc = np.zeros(len(thresholds))
    fn_acc = np.zeros(len(thresholds))
    tn_acc = np.zeros(len(thresholds))

    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = ai_dataset[i]
            rdm = sample['range_doppler_map']
            mask = sample['target_mask']

            if isinstance(rdm, torch.Tensor):
                rdm = rdm.numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            # Normalize RDM
            rdm_min, rdm_max = rdm.min(), rdm.max()
            if rdm_max - rdm_min > 1e-6:
                rdm_norm = (rdm - rdm_min) / (rdm_max - rdm_min)
            else:
                rdm_norm = np.zeros_like(rdm)

            inp = torch.from_numpy(rdm_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            logits = model(inp, cfg)
            heatmap = torch.sigmoid(logits).cpu().numpy().squeeze()

            gt = (mask > 0.5).astype(float)

            for j, thr in enumerate(thresholds):
                det = (heatmap > thr).astype(float)
                tp_acc[j] += (det * gt).sum()
                fp_acc[j] += (det * (1 - gt)).sum()
                fn_acc[j] += ((1 - det) * gt).sum()
                tn_acc[j] += ((1 - det) * (1 - gt)).sum()

    pd = tp_acc / np.maximum(tp_acc + fn_acc, 1)
    pfa = fp_acc / np.maximum(fp_acc + tn_acc, 1)

    return {'thresholds': thresholds, 'pd': pd, 'pfa': pfa}


def generate_roc_curves(args):
    """Generate ROC curves (Pd vs Pfa) at different clutter levels.

    Compares DL model vs CA-CFAR vs OS-CFAR at clutter levels [0.0, 0.2, 0.5, 1.0]
    with fixed SNR=20dB. Produces combined ROC plot, per-clutter subplots, and AUC
    bar chart.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.save_dir, device)
    if model is None:
        print("Cannot generate ROC curves without trained model.")
        return None

    output_dir = os.path.join(args.save_dir, 'roc')
    os.makedirs(output_dir, exist_ok=True)

    clutter_levels = [0.0, 0.2, 0.5, 1.0]
    snr_db = 20
    num_eval = args.num_eval

    results = {}

    for clut in clutter_levels:
        print(f"\n  Clutter={clut:.1f}, SNR={snr_db}dB:")
        ds = AIRadarDataset(
            num_samples=num_eval, config_name='config_cn0566',
            autogen=True, apply_realistic_effects=True,
            clutter_intensity=clut,
            SNR_dB_min=snr_db, SNR_dB_max=snr_db,
            save_path=None, drawfig=False
        )

        print("    Computing CA-CFAR ROC...")
        ca_roc = compute_cfar_pixel_roc(
            ds, 'CA', num_train=12, num_guard=4, num_samples=num_eval
        )

        print("    Computing OS-CFAR ROC...")
        os_roc = compute_cfar_pixel_roc(
            ds, 'OS', num_train=12, num_guard=4, num_samples=num_eval
        )

        print("    Computing DL ROC...")
        dl_roc = compute_dl_pixel_roc(model, ds, device, num_samples=num_eval)

        results[clut] = {
            'ca_cfar': ca_roc,
            'os_cfar': os_roc,
            'dl': dl_roc
        }

    _plot_roc_curves(results, output_dir)
    print(f"\n  ROC curves saved to {output_dir}/")
    return results


def _plot_roc_curves(results, output_dir):
    """Generate ROC curve plots (combined, per-clutter, AUC comparison)."""
    clutter_levels = sorted(results.keys())
    line_styles = {0.0: '-', 0.2: '--', 0.5: '-.', 1.0: ':'}
    colors = {'dl': 'red', 'ca_cfar': 'blue', 'os_cfar': 'green'}
    labels = {'dl': 'DL (Ours)', 'ca_cfar': 'CA-CFAR', 'os_cfar': 'OS-CFAR'}

    # ---- Plot 1: Combined ROC ----
    fig, ax = plt.subplots(figsize=(10, 8))
    for clut in clutter_levels:
        ls = line_styles.get(clut, '-')
        for method in ['dl', 'ca_cfar', 'os_cfar']:
            roc = results[clut][method]
            pfa = np.clip(roc['pfa'], 1e-8, 1.0)
            pd = roc['pd']
            order = np.argsort(pfa)
            label = f"{labels[method]}, clut={clut:.1f}"
            ax.semilogx(pfa[order], pd[order], color=colors[method],
                        linestyle=ls, linewidth=1.5, label=label)

    ax.set_xlabel('Probability of False Alarm (Pfa)', fontsize=12)
    ax.set_ylabel('Probability of Detection (Pd)', fontsize=12)
    ax.set_title('ROC Curves: DL vs CFAR at Different Clutter Levels', fontsize=14)
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([1e-6, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_combined.png'), dpi=150)
    plt.close(fig)

    # ---- Plot 2: 2x2 subplots per clutter ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, clut in enumerate(clutter_levels[:4]):
        ax = axes[idx // 2, idx % 2]
        for method in ['dl', 'ca_cfar', 'os_cfar']:
            roc = results[clut][method]
            pfa = np.clip(roc['pfa'], 1e-8, 1.0)
            pd = roc['pd']
            order = np.argsort(pfa)
            ax.semilogx(pfa[order], pd[order], color=colors[method],
                        linewidth=2, label=labels[method])
        ax.set_xlabel('Pfa')
        ax.set_ylabel('Pd')
        ax.set_title(f'Clutter = {clut:.1f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([1e-6, 1])
        ax.set_ylim([0, 1.05])

    fig.suptitle('ROC Curves per Clutter Level (SNR=20dB)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'roc_per_clutter.png'), dpi=150)
    plt.close(fig)

    # ---- Plot 3: AUC bar chart ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(clutter_levels))
    width = 0.25

    for i, method in enumerate(['dl', 'ca_cfar', 'os_cfar']):
        aucs = []
        for clut in clutter_levels:
            roc = results[clut][method]
            pfa = np.clip(roc['pfa'], 1e-8, 1.0)
            pd = roc['pd']
            order = np.argsort(pfa)
            auc = np.abs(np.trapz(pd[order], pfa[order]))
            aucs.append(auc)
        ax.bar(x + (i - 1) * width, aucs, width,
               label=labels[method], color=colors[method], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{c:.1f}' for c in clutter_levels])
    ax.set_xlabel('Clutter Intensity')
    ax.set_ylabel('AUC (Area Under ROC)')
    ax.set_title('AUC Comparison: DL vs CFAR')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_auc_comparison.png'), dpi=150)
    plt.close(fig)

    print(f"  Saved ROC plots: roc_combined.png, roc_per_clutter.png, roc_auc_comparison.png")


# ==============================================================================
# DL vs CFAR Comparison
# ==============================================================================

def compare_dl_vs_cfar(args):
    """Comprehensive comparison of DL vs CA-CFAR vs OS-CFAR."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.save_dir, device)
    if model is None:
        print("Cannot run comparison without trained model.")
        return

    output_dir = os.path.join(args.save_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    num_eval = args.num_eval

    # ---- 1. SNR Sweep ----
    print("\n" + "=" * 60)
    print("  SNR Sweep Comparison")
    print("=" * 60)

    snr_levels = list(range(0, 36, 5))
    snr_res = {k: [] for k in ['snr', 'ca_f1', 'os_f1', 'dl_f1',
                                 'ca_prec', 'os_prec', 'dl_prec',
                                 'ca_rec', 'os_rec', 'dl_rec',
                                 'ca_range', 'dl_range']}

    for snr in snr_levels:
        print(f"\n  SNR={snr}dB:")
        ds = AIRadarDataset(
            num_samples=num_eval, config_name='config_cn0566',
            autogen=True, apply_realistic_effects=True,
            clutter_intensity=0.15,
            SNR_dB_min=snr, SNR_dB_max=snr,
            save_path=None, drawfig=False
        )

        ds.config['cfar_type'] = 'CA'
        ds.cfar_params['threshold_offset'] = 25
        ca = evaluate_method(ds, lambda rdm, _ds=ds: _ds.cfar_detection(rdm), num_eval)
        print(f"    CA-CFAR: P={ca['precision']:.3f} R={ca['recall']:.3f} F1={ca['f1']:.3f}")

        ds.config['cfar_type'] = 'OS'
        os_m = evaluate_method(ds, lambda rdm, _ds=ds: _ds.cfar_detection(rdm), num_eval)
        print(f"    OS-CFAR: P={os_m['precision']:.3f} R={os_m['recall']:.3f} F1={os_m['f1']:.3f}")

        dl = evaluate_method(
            ds,
            lambda rdm, _m=model, _ds=ds, _dev=device: dl_detect(_m, rdm, _ds, _dev, threshold=0.3),
            num_eval
        )
        print(f"    DL:      P={dl['precision']:.3f} R={dl['recall']:.3f} F1={dl['f1']:.3f}")

        snr_res['snr'].append(snr)
        for key, val in [('ca_f1', ca['f1']), ('os_f1', os_m['f1']), ('dl_f1', dl['f1']),
                         ('ca_prec', ca['precision']), ('os_prec', os_m['precision']),
                         ('dl_prec', dl['precision']),
                         ('ca_rec', ca['recall']), ('os_rec', os_m['recall']),
                         ('dl_rec', dl['recall']),
                         ('ca_range', ca['range_rmse']), ('dl_range', dl['range_rmse'])]:
            snr_res[key].append(val)

    # ---- 2. Clutter Sweep ----
    print("\n" + "=" * 60)
    print("  Clutter Intensity Sweep (SNR=15dB)")
    print("=" * 60)

    clutter_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    clut_res = {k: [] for k in ['clutter', 'ca_f1', 'os_f1', 'dl_f1']}

    for clut in clutter_levels:
        print(f"\n  Clutter={clut:.1f}:")
        ds = AIRadarDataset(
            num_samples=num_eval, config_name='config_cn0566',
            autogen=True, apply_realistic_effects=True,
            clutter_intensity=clut,
            SNR_dB_min=15, SNR_dB_max=15,
            save_path=None, drawfig=False
        )

        ds.config['cfar_type'] = 'CA'
        ds.cfar_params['threshold_offset'] = 25
        ca = evaluate_method(ds, lambda rdm, _ds=ds: _ds.cfar_detection(rdm), num_eval)

        ds.config['cfar_type'] = 'OS'
        os_m = evaluate_method(ds, lambda rdm, _ds=ds: _ds.cfar_detection(rdm), num_eval)

        dl = evaluate_method(
            ds,
            lambda rdm, _m=model, _ds=ds, _dev=device: dl_detect(_m, rdm, _ds, _dev, threshold=0.3),
            num_eval
        )
        print(f"    CA={ca['f1']:.3f}, OS={os_m['f1']:.3f}, DL={dl['f1']:.3f}")

        clut_res['clutter'].append(clut)
        clut_res['ca_f1'].append(ca['f1'])
        clut_res['os_f1'].append(os_m['f1'])
        clut_res['dl_f1'].append(dl['f1'])

    # ---- 3. Realistic Channel Sweep ----
    print("\n" + "=" * 60)
    print("  Realistic Channel Effects (SNR=20dB)")
    print("=" * 60)

    channel_configs = [
        ("Clean", False, 0.0),
        ("Mild effects", True, 0.1),
        ("Moderate effects", True, 0.3),
        ("Heavy effects", True, 0.5),
    ]
    channel_sim = RealisticChannelSimulator(REALISTIC_CHANNEL_PARAMS, enabled=True)
    chan_res = {k: [] for k in ['config', 'ca_f1', 'os_f1', 'dl_f1',
                                 'ca_prec', 'dl_prec', 'ca_rec', 'dl_rec']}

    for name, use_chan, clut in channel_configs:
        print(f"\n  {name}:")
        ds = AIRadarDataset(
            num_samples=num_eval, config_name='config_cn0566',
            autogen=True, apply_realistic_effects=True,
            clutter_intensity=clut,
            SNR_dB_min=20, SNR_dB_max=20,
            save_path=None, drawfig=False
        )

        if use_chan:
            for i in range(len(ds)):
                td = ds.time_domain_data[i]
                if np.iscomplexobj(td):
                    td_c = td.copy()
                else:
                    td_c = (td[..., 0] + 1j * td[..., 1]) if td.shape[-1] == 2 else td.copy()
                if td_c.ndim == 2:
                    td_c = td_c[None, :, :]
                targets = ds.target_info[i]['targets']
                td_aug = channel_sim.apply(
                    td_c, targets, ds.fs, ds.T, ds.B, ds.fc, ds.B / ds.T
                )
                if td_aug.ndim == 3:
                    td_aug = td_aug[0]
                ds.range_doppler_maps[i] = ds.compute_rdm(td_aug)
                ds.cfar_detections[i] = ds.cfar_detection(ds.range_doppler_maps[i])

        ds.config['cfar_type'] = 'CA'
        ds.cfar_params['threshold_offset'] = 25
        ca = evaluate_method(ds, lambda rdm, _ds=ds: _ds.cfar_detection(rdm), num_eval)

        ds.config['cfar_type'] = 'OS'
        os_m = evaluate_method(ds, lambda rdm, _ds=ds: _ds.cfar_detection(rdm), num_eval)

        dl = evaluate_method(
            ds,
            lambda rdm, _m=model, _ds=ds, _dev=device: dl_detect(_m, rdm, _ds, _dev, threshold=0.3),
            num_eval
        )

        print(f"    CA: P={ca['precision']:.3f} R={ca['recall']:.3f} F1={ca['f1']:.3f}")
        print(f"    OS: P={os_m['precision']:.3f} R={os_m['recall']:.3f} F1={os_m['f1']:.3f}")
        print(f"    DL: P={dl['precision']:.3f} R={dl['recall']:.3f} F1={dl['f1']:.3f}")

        chan_res['config'].append(name)
        chan_res['ca_f1'].append(ca['f1'])
        chan_res['os_f1'].append(os_m['f1'])
        chan_res['dl_f1'].append(dl['f1'])
        chan_res['ca_prec'].append(ca['precision'])
        chan_res['dl_prec'].append(dl['precision'])
        chan_res['ca_rec'].append(ca['recall'])
        chan_res['dl_rec'].append(dl['recall'])

    # Generate comparison plots
    _plot_comparison(snr_res, clut_res, chan_res, output_dir)

    all_results = {
        'snr_sweep': snr_res,
        'clutter_sweep': clut_res,
        'channel_sweep': chan_res
    }
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nComparison complete. Results saved to {output_dir}/")


def _plot_comparison(snr_res, clut_res, chan_res, output_dir):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(snr_res['snr'], snr_res['ca_f1'], 'b--o', label='CA-CFAR', linewidth=2)
    ax.plot(snr_res['snr'], snr_res['os_f1'], 'g-.^', label='OS-CFAR', linewidth=2)
    ax.plot(snr_res['snr'], snr_res['dl_f1'], 'r-s', label='DL (Ours)', linewidth=2)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs SNR'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

    ax = axes[0, 1]
    ax.plot(snr_res['snr'], snr_res['ca_prec'], 'b--o', label='CA-CFAR', linewidth=2)
    ax.plot(snr_res['snr'], snr_res['os_prec'], 'g-.^', label='OS-CFAR', linewidth=2)
    ax.plot(snr_res['snr'], snr_res['dl_prec'], 'r-s', label='DL (Ours)', linewidth=2)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Precision')
    ax.set_title('Precision vs SNR'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

    ax = axes[0, 2]
    ax.plot(snr_res['snr'], snr_res['ca_rec'], 'b--o', label='CA-CFAR', linewidth=2)
    ax.plot(snr_res['snr'], snr_res['os_rec'], 'g-.^', label='OS-CFAR', linewidth=2)
    ax.plot(snr_res['snr'], snr_res['dl_rec'], 'r-s', label='DL (Ours)', linewidth=2)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Recall')
    ax.set_title('Recall vs SNR'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

    ax = axes[1, 0]
    ax.plot(clut_res['clutter'], clut_res['ca_f1'], 'b--o', label='CA-CFAR', linewidth=2)
    ax.plot(clut_res['clutter'], clut_res['os_f1'], 'g-.^', label='OS-CFAR', linewidth=2)
    ax.plot(clut_res['clutter'], clut_res['dl_f1'], 'r-s', label='DL (Ours)', linewidth=2)
    ax.set_xlabel('Clutter Intensity'); ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Clutter (SNR=15dB)'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

    ax = axes[1, 1]
    x = np.arange(len(chan_res['config']))
    w = 0.25
    ax.bar(x - w, chan_res['ca_f1'], w, label='CA-CFAR', color='steelblue')
    ax.bar(x, chan_res['os_f1'], w, label='OS-CFAR', color='green', alpha=0.7)
    ax.bar(x + w, chan_res['dl_f1'], w, label='DL (Ours)', color='orangered')
    ax.set_xticks(x); ax.set_xticklabels(chan_res['config'], rotation=20, fontsize=8)
    ax.set_ylabel('F1 Score'); ax.set_title('F1 under Realistic Channel')
    ax.legend(fontsize=8); ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 2]
    ax.plot(snr_res['snr'], snr_res['ca_range'], 'b--o', label='CA-CFAR')
    ax.plot(snr_res['snr'], snr_res['dl_range'], 'r-s', label='DL (Ours)')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Range RMSE (m)')
    ax.set_title('Range Accuracy vs SNR'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Radar Detection: DL vs CFAR Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'dl_vs_cfar_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved comparison plot: {output_dir}/dl_vs_cfar_comparison.png")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train & Evaluate Robust Radar DL Detection Models"
    )
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'compare', 'roc', 'all'],
                        help='Operation mode')
    parser.add_argument('--model', type=str, default='RobustRadarNetG3',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--loss', type=str, default='dicefocal',
                        choices=['focal', 'dicefocal'],
                        help='Loss function')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_train', type=int, default=800,
                        help='Training samples')
    parser.add_argument('--num_val', type=int, default=100,
                        help='Validation samples')
    parser.add_argument('--num_eval', type=int, default=50,
                        help='Samples per evaluation point')
    parser.add_argument('--save_dir', type=str, default='data/g3_radar',
                        help='Directory for model + results')
    args = parser.parse_args()

    print("=" * 60)
    print("  Robust Radar DL Detection - Train & Evaluate")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Model: {args.model}")
    print(f"  Loss: {args.loss}")
    print(f"  Save dir: {args.save_dir}")

    if args.mode in ('train', 'all'):
        train_model(args)

    if args.mode in ('compare', 'all'):
        compare_dl_vs_cfar(args)

    if args.mode in ('roc', 'all'):
        generate_roc_curves(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
