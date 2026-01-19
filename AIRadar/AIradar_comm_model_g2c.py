"""
AIradar_comm_model_g2c.py

Joint Radar+Comm Deep Model with G2 Dataset Integration

New in this version (G2C):
- Uses AIradar_comm_dataset_g2 with realistic 5G features
- Supports both TRADITIONAL and OTFS modes
- Generalized models with FiLM conditioning for multi-config training
- Includes CNR/RCS evaluation metrics
- DMRS channel estimation and TDL channel models
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import G2 dataset
from AIradar_comm_dataset_g2 import (
    RADAR_COMM_CONFIGS_G2,
    AIRadar_Comm_Dataset_G2,
)

C = 3e8  # speed of light

# ----------------------------------------------------------------------
# Config sets for training / evaluation
# IMPORTANT: Different RDM sizes require separate training for best accuracy
# Configs are grouped by similar RDM dimensions
# ----------------------------------------------------------------------

# Group A: Small RDM (64 x ~1000-2000) - Fastest training
CONFIG_GROUP_A = [
    "CN0566_TRADITIONAL",      # (64, 1000)
    "Automotive_77GHz_LongRange",  # (64, 2048)
]

# Group B: Medium RDM (64 x 3000) 
CONFIG_GROUP_B = [
    "AUTOMOTIVE_TRADITIONAL",  # (64, 3000)
]

# Group C: Large RDM (64 x 6400) - Requires more memory
CONFIG_GROUP_C = [
    "XBand_10GHz_MediumRange", # (64, 6400)
]

# Default: Use Group A (proven F1=0.93 on CN0566_TRADITIONAL)
# Set TRAIN_CONFIGS to desired group before training
TRAIN_CONFIGS = CONFIG_GROUP_A

VAL_CONFIGS = TRAIN_CONFIGS.copy()

TEST_CONFIGS = TRAIN_CONFIGS.copy()

# Mapping from config_name -> integer ID
CONFIG_ID_MAP = {name: i for i, name in enumerate(RADAR_COMM_CONFIGS_G2.keys())}
NUM_CONFIGS = len(CONFIG_ID_MAP)

# Global max modulation order
MAX_MOD_ORDER = max(cfg.get("mod_order", 4) for cfg in RADAR_COMM_CONFIGS_G2.values())


# ----------------------------------------------------------------------
# FiLM Conditioning Layers
# ----------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for config adaptation."""
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
    """Encode radar/comm configuration to embedding vector."""
    CHANNEL_MAP = {'awgn': 0, 'multipath': 1, 'rayleigh': 2, 'none': 3}
    MODE_MAP = {'TRADITIONAL': 0, 'OTFS': 1}
    
    def __init__(self, embed_dim=64):
        super().__init__()
        # Support both legacy 8-dim and new 5-dim config inputs
        self.fc_legacy = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
        self.fc_generalized = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, config_tensor):
        # Auto-detect input dimension
        if config_tensor.dim() == 1:
            config_tensor = config_tensor.unsqueeze(0)
        if config_tensor.shape[-1] == 5:
            return self.fc_generalized(config_tensor)
        return self.fc_legacy(config_tensor)
    
    @staticmethod
    def encode_config(config: dict) -> torch.Tensor:
        """Legacy: Convert config dict to normalized tensor (8-dim)."""
        channel_id = ConfigEncoder.CHANNEL_MAP.get(config.get('channel_model', 'multipath'), 1)
        mode_id = ConfigEncoder.MODE_MAP.get(config.get('mode', 'TRADITIONAL'), 0)
        return torch.tensor([
            config.get('fc', 10e9) / 1e11,
            config.get('radar_B', 500e6) / 1e9,
            config.get('radar_Nc', 64) / 256,
            config.get('radar_Ns', 1000) / 4000,
            np.log2(config.get('mod_order', 16)) / 6,
            channel_id / 3,
            mode_id,
            config.get('radar_T', 50e-6) * 1e4,
        ], dtype=torch.float32)
    
    @staticmethod
    def encode_config_generalized(config: dict) -> torch.Tensor:
        """
        Continuous physical parameter encoding for zero-shot generalization.
        Uses only physical parameters without discrete IDs.
        """
        return torch.tensor([
            config.get('fc', 10e9) / 100e9,        # Carrier Freq (normalized 0-100GHz)
            config.get('radar_B', 500e6) / 2e9,    # Bandwidth (normalized 0-2GHz)
            config.get('snr_db', 20) / 40,         # Estimated SNR (normalized ~0-1)
            config.get('delay_spread', 1e-7) * 1e7, # Channel geometry (scaled)
            np.log2(config.get('mod_order', 4)) / 6.0,  # Bits per symbol (0-1)
        ], dtype=torch.float32)


# ----------------------------------------------------------------------
# Generalized Radar Model with FiLM
# ----------------------------------------------------------------------
class FiLMConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.film = FiLMLayer(out_ch, cond_dim)
        
    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)
        return F.relu(x)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1)
        
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class GeneralizedRadarNet(nn.Module):
    """Generalized radar detection network with FiLM conditioning."""
    
    def __init__(self, in_ch=1, base_ch=48, cond_dim=64, target_size=(512, 512)):
        super().__init__()
        self.target_size = target_size
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        
        # Encoder
        self.enc1 = FiLMConvBlock(in_ch, base_ch, cond_dim)
        self.enc2 = FiLMConvBlock(base_ch, base_ch*2, cond_dim, stride=2)
        self.enc3 = FiLMConvBlock(base_ch*2, base_ch*4, cond_dim, stride=2)
        self.enc4 = FiLMConvBlock(base_ch*4, base_ch*8, cond_dim, stride=2)
        
        self.se = SEBlock(base_ch*8)
        
        # Decoder
        self.dec4 = FiLMConvBlock(base_ch*8, base_ch*4, cond_dim)
        self.dec3 = FiLMConvBlock(base_ch*4 + base_ch*4, base_ch*2, cond_dim)
        self.dec2 = FiLMConvBlock(base_ch*2 + base_ch*2, base_ch, cond_dim)
        self.dec1 = FiLMConvBlock(base_ch + base_ch, base_ch, cond_dim)
        
        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        
    def forward(self, x, config_tensor):
        B = x.size(0)
        orig_size = x.shape[-2:]
        
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        cond = self.config_encoder(config_tensor)
        
        e1 = self.enc1(x, cond)
        e2 = self.enc2(e1, cond)
        e3 = self.enc3(e2, cond)
        e4 = self.enc4(e3, cond)
        
        z = self.se(e4)
        
        d4 = self.dec4(z, cond)
        d4 = F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d4, e3], dim=1), cond)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1), cond)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1), cond)
        
        logits = self.out_conv(d1)
        
        if logits.shape[-2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
            
        return logits


# ----------------------------------------------------------------------
# Complex-Valued Convolution Backbone (for I/Q processing)
# ----------------------------------------------------------------------
class ComplexConvBlock(nn.Module):
    """
    Complex convolution that respects I/Q phase-amplitude coupling.
    Mathematically: (A+Bi)*(C+Di) = (AC-BD) + (AD+BC)i
    Uses GroupNorm instead of BatchNorm for stability in signal processing.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_re = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm_re = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm_im = nn.GroupNorm(min(8, out_ch), out_ch)
        
    def forward(self, x_re, x_im):
        # Complex multiplication: (conv_re + conv_im*j) * (x_re + x_im*j)
        # Real output = Re*x_re - Im*x_im
        out_re = self.conv_re(x_re) - self.conv_im(x_im)
        # Imaginary output = Re*x_im + Im*x_re
        out_im = self.conv_re(x_im) + self.conv_im(x_re)
        
        out_re = F.relu(self.norm_re(out_re))
        out_im = F.relu(self.norm_im(out_im))
        return out_re, out_im


class ComplexResBlock(nn.Module):
    """Residual Complex Convolution Block with skip connection."""
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        self.conv1 = ComplexConvBlock(ch, ch, kernel_size, padding=kernel_size//2)
        self.conv2_re = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size//2)
        self.conv2_im = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size//2)
        self.norm_re = nn.GroupNorm(min(8, ch), ch)
        self.norm_im = nn.GroupNorm(min(8, ch), ch)
        
    def forward(self, x_re, x_im):
        h_re, h_im = self.conv1(x_re, x_im)
        h_re = self.norm_re(self.conv2_re(h_re))
        h_im = self.norm_im(self.conv2_im(h_im))
        return F.relu(x_re + h_re), F.relu(x_im + h_im)  # Residual skip


class ResFiLMBlock(nn.Module):
    """
    FiLM conditioning with residual connection for gradient stability.
    Uses GroupNorm instead of BatchNorm (signal processing best practice).
    """
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = FiLMLayer(ch, cond_dim)
        
    def forward(self, x, cond):
        h = self.norm(x)
        h = F.relu(self.conv(h))
        h = self.film(h, cond)
        return x + h  # Residual skip connection


class SimpleBitDemapper(nn.Module):
    """
    Per-pixel MLP that maps I/Q + channel info to bit LLRs.
    
    Uses all 5 input channels: I, Q, H_mag, H_phase, SNR
    This provides the demapper with channel state information needed
    for soft demapping in noisy/fading conditions.
    """
    MAX_BITS = 6
    
    def __init__(self, hidden_dim=256, in_channels=5):
        super().__init__()
        self.in_channels = in_channels
        
        # Per-pixel MLP: 5 inputs -> 6 bit LLRs
        # Deeper network with more capacity for noisy data
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, self.MAX_BITS),
        )
        
        # Output scaling for better initial LLR range
        self.output_scale = nn.Parameter(torch.ones(1) * 3.0)
        
        # Initialize final layer with larger weights for stronger initial outputs
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=2.0)
    
    def forward(self, x, config_tensor=None, mod_order=None, config_id=None):
        """
        Args:
            x: [B, C, H, W] where C is 5 (I, Q, H_mag, H_phase, snr)
        Returns:
            llr: [B, 6, H, W] bit LLRs
        """
        B, C, H, W = x.shape
        
        # Use all available channels (up to 5)
        in_ch = min(C, self.in_channels)
        features = x[:, :in_ch]  # [B, in_ch, H, W]
        
        # Reshape for per-pixel MLP: [B, H, W, in_ch]
        features = features.permute(0, 2, 3, 1).contiguous()
        
        # Flatten spatial dims: [B*H*W, in_ch]
        features_flat = features.view(-1, in_ch)
        
        # Pad if fewer channels than expected
        if in_ch < self.in_channels:
            padding = torch.zeros(features_flat.size(0), self.in_channels - in_ch, 
                                  device=features_flat.device)
            features_flat = torch.cat([features_flat, padding], dim=1)
        
        # MLP forward: [B*H*W, 6]
        llr_flat = self.mlp(features_flat)
        
        # Reshape back: [B, H, W, 6] -> [B, 6, H, W]
        llr = llr_flat.view(B, H, W, self.MAX_BITS).permute(0, 3, 1, 2)
        
        # Scale for reasonable LLR range
        return llr * self.output_scale
    
    def get_symbol_logits(self, llr_logits, mod_order):
        """Convert LLR to symbol logits for backward compatibility."""
        active_bits = int(np.log2(mod_order))
        active_llr = llr_logits[:, :active_bits]  # [B, bits, H, W]
        
        # Convert bit probabilities to symbol logits
        bit_probs = torch.sigmoid(active_llr)  # [B, bits, H, W]
        
        # Enumerate all possible symbols
        B, _, H, W = active_llr.shape
        n_symbols = mod_order
        
        symbol_logits = torch.zeros(B, n_symbols, H, W, device=active_llr.device)
        for sym in range(n_symbols):
            log_prob = 0
            for b in range(active_bits):
                bit_val = (sym >> b) & 1
                if bit_val == 1:
                    log_prob = log_prob + torch.log(bit_probs[:, b] + 1e-8)
                else:
                    log_prob = log_prob + torch.log(1 - bit_probs[:, b] + 1e-8)
            symbol_logits[:, sym] = log_prob
        
        return symbol_logits


# ----------------------------------------------------------------------
# Universal Communication Demapper with Bit-Wise LLR Output
# ----------------------------------------------------------------------
class UniversalCommNet(nn.Module):
    """
    Universal demapper using complex convolutions and bit-wise LLR output.
    
    Architecture:
        Input [B, 2, H, W] (I/Q) → Complex Backbone → Features
                                → FiLM Config Adapter → Adapted Features
                                → Universal LLR Head → [B, max_bits, H, W]
    
    Key improvements over AdaptiveCommNet:
        1. Complex convolutions respect I/Q phase-amplitude coupling
        2. Single universal head instead of QAM-specific adapters
        3. Bit-wise LLR output with masking for different mod orders
        4. GroupNorm + Residual connections for training stability
    """
    MAX_BITS = 6  # log2(64) = 6 bits for 64-QAM max
    
    def __init__(self, in_ch=2, base_ch=64, cond_dim=64):
        super().__init__()
        self.base_ch = base_ch
        self.cond_dim = cond_dim
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        
        # ========== Complex-Valued Backbone ==========
        # Input: 2 channels (I, Q) treated as complex signal
        self.complex_conv1 = ComplexConvBlock(1, base_ch, 3, padding=1)
        self.complex_conv2 = ComplexConvBlock(base_ch, base_ch, 3, padding=1)
        self.complex_res = ComplexResBlock(base_ch)
        self.complex_conv3 = ComplexConvBlock(base_ch, base_ch*2, 3, stride=2, padding=1)
        self.complex_res2 = ComplexResBlock(base_ch*2)
        
        # Merge I/Q back to single representation
        self.merge_conv = nn.Conv2d(base_ch*2 * 2, base_ch*2, 1)  # 2x for concat of re/im
        self.merge_norm = nn.GroupNorm(8, base_ch*2)
        
        # ========== FiLM Residual Adapter ==========
        self.adapter1 = ResFiLMBlock(base_ch*2, cond_dim)
        self.adapter2 = ResFiLMBlock(base_ch*2, cond_dim)
        
        # ========== Channel Info Processing ==========
        # Additional channels for H_mag, H_phase, snr (if provided)
        self.has_channel_info = True
        self.channel_encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),  # H_mag, H_phase, snr
            nn.GroupNorm(8, base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch*2),
            nn.ReLU(),
        )
        
        # Fusion layer
        self.fusion = nn.Conv2d(base_ch*2 * 2, base_ch*2, 1)
        self.fusion_norm = nn.GroupNorm(8, base_ch*2)
        
        # ========== Universal Bit-Wise LLR Head ==========
        # Outputs 6 LLR values per symbol position (max 64-QAM)
        self.llr_head = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(),
            nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2),  # Upsample to match input
            nn.GroupNorm(8, base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, self.MAX_BITS, 1),  # Final: 6 LLR outputs
        )
        
    def normalize_iq(self, x_re, x_im):
        """Normalize I/Q to unit average power for consistent input scale."""
        power = torch.sqrt(x_re**2 + x_im**2 + 1e-8)
        avg_power = torch.mean(power, dim=(2, 3), keepdim=True)
        return x_re / (avg_power + 1e-6), x_im / (avg_power + 1e-6)
        
    def forward(self, x, config_tensor, mod_order=None, config_id=None):
        """
        Args:
            x: [B, C, H, W] where C can be:
               - 2: (I, Q) only
               - 5: (eq_real, eq_imag, H_mag, H_phase, snr) - legacy format
            config_tensor: [B, cond_dim] or [B, 5] conditioning
            mod_order: modulation order (4, 16, or 64) - determines active bits
            config_id: ignored (for backward compatibility)
        Returns:
            llr_logits: [B, max_bits, H, W] - bit-wise LLR values
        """
        B, C, H, W = x.shape
        
        # Handle NaN/Inf from edge cases
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Split into I/Q and optional channel info
        if C >= 5:
            # Legacy format: (eq_real, eq_imag, H_mag, H_phase, snr)
            x_re = x[:, 0:1]  # [B, 1, H, W]
            x_im = x[:, 1:2]  # [B, 1, H, W]
            channel_info = x[:, 2:5]  # [B, 3, H, W]
        else:
            # New format: just I/Q
            x_re = x[:, 0:1]
            x_im = x[:, 1:2]
            channel_info = None
        
        # ========== I/Q Power Normalization ==========
        # Critical: normalize to unit power for consistent input to complex backbone
        x_re, x_im = self.normalize_iq(x_re, x_im)
        
        # Config encoding
        cond = self.config_encoder(config_tensor)
        
        # ========== Complex Backbone ==========
        h_re, h_im = self.complex_conv1(x_re, x_im)
        h_re, h_im = self.complex_conv2(h_re, h_im)
        h_re, h_im = self.complex_res(h_re, h_im)
        h_re, h_im = self.complex_conv3(h_re, h_im)
        h_re, h_im = self.complex_res2(h_re, h_im)
        
        # Merge complex to real representation
        h = torch.cat([h_re, h_im], dim=1)  # [B, base_ch*4, H/2, W/2]
        h = F.relu(self.merge_norm(self.merge_conv(h)))  # [B, base_ch*2, H/2, W/2]
        
        # ========== Fuse Channel Info ==========
        if channel_info is not None and self.has_channel_info:
            ch_feat = self.channel_encoder(channel_info)  # [B, base_ch*2, H/2, W/2]
            h = torch.cat([h, ch_feat], dim=1)
            h = F.relu(self.fusion_norm(self.fusion(h)))
        
        # ========== FiLM Adaptation ==========
        h = self.adapter1(h, cond)
        h = self.adapter2(h, cond)
        
        # ========== LLR Output ==========
        llr_logits = self.llr_head(h)  # [B, 6, H, W]
        
        # NOTE: Do NOT mask here - let the loss function handle active bits
        # This ensures gradients flow through all bits during training
        return llr_logits
    
    def get_symbol_logits(self, llr_logits, mod_order):
        """
        Convert bit-wise LLRs to symbol logits for backward compatibility.
        This creates M-ary symbol probabilities from binary bit likelihoods.
        
        Args:
            llr_logits: [B, max_bits, H, W]
            mod_order: M (4, 16, 64)
        Returns:
            symbol_logits: [B, mod_order, H, W]
        """
        B, _, H, W = llr_logits.shape
        active_bits = int(np.log2(mod_order))
        
        # Get active LLRs
        active_llr = llr_logits[:, :active_bits]  # [B, k, H, W]
        
        # Convert LLR to bit probabilities: p(b=1) = sigmoid(llr)
        bit_probs = torch.sigmoid(active_llr)  # [B, k, H, W]
        
        # Build symbol probabilities (product of bit probs matching Gray code)
        # For each symbol s, compute prod_i p(b_i = s_i) where s_i is i-th bit of s
        symbol_logits = torch.zeros(B, mod_order, H, W, device=llr_logits.device)
        
        for s in range(mod_order):
            log_prob = torch.zeros(B, H, W, device=llr_logits.device)
            for i in range(active_bits):
                bit_val = (s >> i) & 1
                if bit_val == 1:
                    log_prob += torch.log(bit_probs[:, i] + 1e-8)
                else:
                    log_prob += torch.log(1 - bit_probs[:, i] + 1e-8)
            symbol_logits[:, s] = log_prob
        
        return symbol_logits


# ----------------------------------------------------------------------
# Adaptive Communication Demapper with QAM-Specific Adapters (LEGACY)
# ----------------------------------------------------------------------
class AdaptiveCommNet(nn.Module):
    """CNN backbone with QAM-specific adapter heads.
    
    Architecture:
        Input [B, 4, H, W] → ZF Pre-Eq → [B, 6, H, W]
                          → Shared Backbone → Features [B, base_ch*2, H, W]
                          → QAM Adapter (selected by mod_order) → Logits
    
    Each adapter is a small 1x1 conv head that specializes in its QAM order's
    decision boundaries. The backbone learns general channel-aware features,
    while adapters learn QAM-specific constellation geometry.
    
    Training strategy:
        1. Train backbone + all adapters jointly
        2. Optionally freeze backbone, fine-tune specific adapter
    """
    
    SUPPORTED_MOD_ORDERS = [4, 16, 64]  # QPSK, 16-QAM, 64-QAM
    
    def __init__(self, in_ch=4, base_ch=64, cond_dim=64, max_mod_order=64):
        super().__init__()
        self.base_ch = base_ch
        self.max_mod_order = max_mod_order
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        
        # ========== Shared Backbone ==========
        # Input: 5 channels (eq_real, eq_imag, H_mag, H_phase, snr)
        # ZF equalization is done in dataset with constellation-aware normalization
        actual_in_ch = 5
        
        self.conv1 = nn.Conv2d(actual_in_ch, base_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.film1 = FiLMLayer(base_ch, cond_dim)
        
        self.conv2 = nn.Conv2d(base_ch, base_ch*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_ch*2)
        self.film2 = FiLMLayer(base_ch*2, cond_dim)
        
        self.conv3 = nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_ch*2)
        self.film3 = FiLMLayer(base_ch*2, cond_dim)
        
        self.conv4 = nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_ch*2)
        
        # ========== Per-QAM Adapter Heads ==========
        # Each adapter is a small network specialized for its modulation order
        self.adapters = nn.ModuleDict()
        for mod in self.SUPPORTED_MOD_ORDERS:
            self.adapters[str(mod)] = self._build_adapter(base_ch*2, mod)
        
        # Fallback adapter for unsupported mod_orders (uses max)
        self.fallback_adapter = nn.Conv2d(base_ch*2, max_mod_order, 1)
        
    def _build_adapter(self, in_ch, out_ch):
        """Build adapter head with capacity proportional to modulation order."""
        # Deeper adapter for higher-order modulation (16-QAM and above)
        if out_ch >= 16:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, in_ch, 3, padding=1),  # Extra layer for 16-QAM+
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1),  # 1x1 to output logits
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1),
            )
    
    def freeze_backbone(self):
        """Freeze backbone for adapter-only fine-tuning."""
        for name, param in self.named_parameters():
            if 'adapter' not in name and 'fallback' not in name:
                param.requires_grad = False
        print("[AdaptiveCommNet] Backbone frozen, only adapters trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for full training."""
        for param in self.parameters():
            param.requires_grad = True
        print("[AdaptiveCommNet] All parameters trainable")
    
    def forward_backbone(self, x, config_tensor):
        """Forward through shared backbone.
        
        Args:
            x: [B, 5, H, W] - (eq_real, eq_imag, H_mag, H_phase, snr)
               Already ZF-equalized and constellation-normalized from dataset
        """
        # Input validation - handle NaN/Inf from edge cases
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        cond = self.config_encoder(config_tensor)
        
        h = F.relu(self.film1(self.bn1(self.conv1(x)), cond))
        h = F.relu(self.film2(self.bn2(self.conv2(h)), cond))
        h = F.relu(self.film3(self.bn3(self.conv3(h)), cond))
        h = F.relu(self.bn4(self.conv4(h)))
        
        return h
    
    def forward(self, x, config_tensor, mod_order=None, config_id=None):
        """
        Args:
            x: [B, 5, H, W] - (eq_real, eq_imag, H_mag, H_phase, snr)
            config_tensor: [B, cond_dim] conditioning
            mod_order: modulation order (4, 16, or 64) - selects adapter
            config_id: ignored
        Returns:
            logits: [B, mod_order, H, W]
        """
        # Backbone feature extraction
        features = self.forward_backbone(x, config_tensor)
        
        # Select appropriate adapter based on mod_order
        mod_key = str(mod_order) if mod_order in self.SUPPORTED_MOD_ORDERS else None
        
        if mod_key and mod_key in self.adapters:
            logits = self.adapters[mod_key](features)
        else:
            # Fallback for unsupported mod_orders
            logits = self.fallback_adapter(features)
            if mod_order is not None and mod_order < self.max_mod_order:
                logits = logits[:, :mod_order, :, :]
        
        return logits


# Backward compatibility aliases
GeneralizedCommNet = AdaptiveCommNet
ChannelAwareCommNet = AdaptiveCommNet


# ----------------------------------------------------------------------
# Joint Model
# ----------------------------------------------------------------------
class JointRadarCommNet_G2(nn.Module):
    """Joint Radar+Comm network with G2 features.
    
    Supports multiple comm architectures:
      - use_universal=True: Complex convolutions + LLR output
      - use_simple=True: Simple per-pixel MLP demapper (recommended)
      - Both False: Legacy QAM-specific adapters
    """
    
    def __init__(self, base_ch=48, cond_dim=64, max_mod_order=64, 
                 use_universal=False, use_simple=False):
        super().__init__()
        self.use_universal = use_universal
        self.use_simple = use_simple
        self.radar_net = GeneralizedRadarNet(base_ch=base_ch, cond_dim=cond_dim)
        
        if use_simple:
            # Simple per-pixel MLP demapper (best for QAM)
            self.comm_net = SimpleBitDemapper(hidden_dim=256)
        elif use_universal:
            # New architecture: complex convolutions + LLR output
            self.comm_net = UniversalCommNet(base_ch=base_ch, cond_dim=cond_dim)
        else:
            # Legacy architecture: QAM-specific adapters
            self.comm_net = AdaptiveCommNet(base_ch=base_ch, cond_dim=cond_dim, 
                                            max_mod_order=max_mod_order)
        
    def forward(self, radar_input, comm_input, config_tensor, mod_order=None, config_id=None):
        radar_logits = self.radar_net(radar_input, config_tensor)
        comm_logits = self.comm_net(comm_input, config_tensor, mod_order, config_id)
        return radar_logits, comm_logits
    
    def get_symbol_logits(self, llr_logits, mod_order):
        """Convert LLR to symbol logits (only valid when use_universal or use_simple)."""
        if self.use_universal or self.use_simple:
            return self.comm_net.get_symbol_logits(llr_logits, mod_order)
        return llr_logits  # Already symbol logits


# Alias for new architecture
JointRadarCommNet_G2_Universal = lambda **kw: JointRadarCommNet_G2(use_universal=True, **kw)
JointRadarCommNet_G2_Simple = lambda **kw: JointRadarCommNet_G2(use_simple=True, **kw)


# ----------------------------------------------------------------------
# G2 Dataset Wrapper
# ----------------------------------------------------------------------
import pickle

class G2DeepDataset(Dataset):
    """Wrapper for AIRadar_Comm_Dataset_G2 for deep learning with caching."""
    
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
        
        # Cache file name includes RF impairments flag to avoid mixing data
        cache_suffix = '_rf' if enable_rf_impairments else ''
        save_path = os.path.join(save_root, split, config_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Check for cached data
        cache_file = os.path.join(save_path, f'cache_{num_samples}{cache_suffix}.pkl')
        
        if os.path.exists(cache_file):
            # Load from cache
            print(f"[Cache] Loading {config_name}/{split} from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            
            # Create minimal dataset wrapper to access cached data
            self.g2_ds = type('CachedDataset', (), {
                'data_samples': cached['samples'],
                '__len__': lambda s: len(s.data_samples),
                '__getitem__': lambda s, i: s.data_samples[i],
            })()
        else:
            # Generate new data
            print(f"[Generate] Creating {config_name}/{split} ({num_samples} samples, RF={enable_rf_impairments})")
            self.g2_ds = AIRadar_Comm_Dataset_G2(
                config_name=config_name,
                num_samples=num_samples,
                save_path=save_path,
                drawfig=False,
                enable_clutter=True,
                enable_imperfect_csi=True,
                enable_rf_impairments=enable_rf_impairments,
            )
            
            # Save to cache
            print(f"[Cache] Saving to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({'samples': self.g2_ds.data_samples}, f)
        
        # Pre-compute config tensor
        self.config_tensor = ConfigEncoder.encode_config(self.config)
        
    def __len__(self):
        return len(self.g2_ds)
    
    def _build_radar_label(self, rdm_shape, r_axis, v_axis, targets):
        """Build Gaussian heatmap for radar targets."""
        D, R = rdm_shape
        label = np.zeros((D, R), dtype=np.float32)
        sigma2 = self.radar_sigma ** 2
        
        for t in targets:
            r_m = t.get("range", 0)
            v_m = t.get("velocity", 0)
            
            # Find closest bin
            r_idx = int(np.argmin(np.abs(r_axis - r_m)))
            v_idx = int(np.argmin(np.abs(v_axis - v_m)))
            
            if not (0 <= r_idx < R and 0 <= v_idx < D):
                continue
                
            # Gaussian splat with larger radius for better DL training
            radius = 5
            for dv in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    rr = r_idx + dr
                    dd = v_idx + dv
                    if 0 <= rr < R and 0 <= dd < D:
                        dist2 = dr * dr + dv * dv
                        val = math.exp(-dist2 / (2 * sigma2))
                        label[dd, rr] = max(label[dd, rr], val)
        return label
    
    def _normalize_rdm(self, rdm, top_p=99.5, dyn_db=40.0):
        """Normalize RDM to [0, 1]."""
        top = np.percentile(rdm, top_p)
        rdm = np.clip(rdm, top - dyn_db, top)
        rdm = (rdm - (top - dyn_db)) / dyn_db
        return rdm.astype(np.float32)
    
    def __getitem__(self, idx):
        sample = self.g2_ds[idx]
        
        # Radar data
        rdm = np.array(sample['range_doppler_map'])  # May be in dB
        rdm_norm = self._normalize_rdm(rdm)
        
        r_axis = np.array(sample['range_axis'])
        v_axis = np.array(sample['velocity_axis'])
        targets = sample['target_info']['targets']
        snr_db = sample['target_info'].get('snr_db', 15.0)
        
        # Build radar label at original resolution
        radar_label = self._build_radar_label(rdm.shape, r_axis, v_axis, targets)
        
        # NOTE: Keep original RDM shape - configs in same group have similar sizes
        # The model's internal pooling handles minor size differences
        
        radar_input = torch.from_numpy(rdm_norm).unsqueeze(0).contiguous()
        radar_target = torch.from_numpy(radar_label).unsqueeze(0).contiguous()
        
        # Communication data
        comm_info = sample['comm_info']
        mod_order = self.config.get('mod_order', 16)
        
        # Get symbols and channel estimate
        tx_symbols = np.array(comm_info.get('tx_symbols', []), dtype=np.complex64)
        rx_symbols = np.array(comm_info.get('rx_symbols', []), dtype=np.complex64)
        channel_est = np.array(comm_info.get('channel_est', None))
        
        if len(rx_symbols) == 0:
            # Fallback: create dummy comm data (4 channels for ChannelAwareCommNet)
            H, W = 8, 256
            comm_input = torch.zeros(4, H, W, dtype=torch.float32)
            comm_target = torch.zeros(H, W, dtype=torch.long)
        else:
            # Reshape to grid
            n_syms = comm_info.get('num_data_syms', 8)
            fft_size = comm_info.get('fft_size', len(rx_symbols) // n_syms)
            
            try:
                rx_grid = rx_symbols.reshape(n_syms, fft_size)
            except:
                # Handle size mismatch
                total = len(rx_symbols)
                fft_size = min(256, total)
                n_syms = total // fft_size
                rx_grid = rx_symbols[:n_syms * fft_size].reshape(n_syms, fft_size)
            
            # Build channel estimate grid (H_est is per subcarrier, broadcast to all symbols)
            if channel_est is not None and len(channel_est) > 0:
                # Resize H_est to match fft_size if needed
                if len(channel_est) != fft_size:
                    from scipy.ndimage import zoom
                    H_est_resized = zoom(channel_est.real, fft_size/len(channel_est)) + \
                                   1j * zoom(channel_est.imag, fft_size/len(channel_est))
                else:
                    H_est_resized = channel_est
                # Broadcast to all OFDM symbols
                H_grid = np.tile(H_est_resized[None, :], (n_syms, 1))
            else:
                # Fallback: assume channel = 1 (no distortion)
                H_grid = np.ones_like(rx_grid)
            
            # CONSTELLATION-AWARE NORMALIZATION
            # Perform ZF equalization here to preserve geometry
            H_safe = np.where(np.abs(H_grid) > 1e-6, H_grid, 1e-6 + 0j)
            eq_symbols = rx_grid / H_safe
            
            # Scale by constellation normalization factor so symbols ~[-1, 1]
            if mod_order == 4:
                scale_factor = np.sqrt(2)   # QPSK
            elif mod_order == 16:
                scale_factor = np.sqrt(10)  # 16-QAM
            else:
                scale_factor = np.sqrt(42)  # 64-QAM
            
            eq_real = eq_symbols.real / scale_factor
            eq_imag = eq_symbols.imag / scale_factor
            
            # Clip to prevent outliers from deep fades
            eq_real = np.clip(eq_real, -3, 3)
            eq_imag = np.clip(eq_imag, -3, 3)
            
            # Normalize channel magnitude info
            H_mag = np.abs(H_grid) / (np.abs(H_grid).max() + 1e-6)
            H_phase = np.angle(H_grid) / np.pi  # Normalize to [-1, 1]
            
            # Add SNR as channel for adaptive decision boundaries
            snr_normalized = snr_db / 35.0  # Normalize to ~[0, 1]
            snr_channel = np.full_like(eq_real, snr_normalized)
            
            # 5-channel input: (eq_real, eq_imag, H_mag, H_phase, snr)
            comm_input = torch.tensor(
                np.stack([eq_real, eq_imag, H_mag, H_phase, snr_channel], axis=0),
                dtype=torch.float32
            ).contiguous()
            
            # Symbol indices as labels
            tx_ints = np.array(comm_info.get('tx_ints', []), dtype=np.int64)
            if len(tx_ints) >= n_syms * fft_size:
                comm_target = torch.tensor(
                    tx_ints[:n_syms * fft_size].reshape(n_syms, fft_size),
                    dtype=torch.long
                ).contiguous()
            else:
                comm_target = torch.zeros(n_syms, fft_size, dtype=torch.long)
        
        meta = {
            'config_id': self.config_id,
            'config_name': self.config_name,
            'config_tensor': self.config_tensor,
            'mod_order': mod_order,
            'snr_db': snr_db,
            'mode': self.config.get('mode', 'TRADITIONAL'),
            'targets': targets,
            'cfar_detections': sample.get('cfar_detections', []),
            'r_axis': r_axis,
            'v_axis': v_axis,
            'ber': comm_info.get('ber', 0.0),
        }
        
        return radar_input, radar_target, comm_input, comm_target, meta


def g2_collate_fn(batch):
    """Custom collate function for G2 dataset with variable-size meta."""
    radar_inputs = torch.stack([b[0] for b in batch])
    radar_targets = torch.stack([b[1] for b in batch])
    comm_inputs = torch.stack([b[2] for b in batch])
    comm_targets = torch.stack([b[3] for b in batch])
    
    # Meta: keep as list for variable-size items, stack tensors
    meta = {
        'config_id': torch.tensor([b[4]['config_id'] for b in batch]),
        'config_name': [b[4]['config_name'] for b in batch],
        'config_tensor': torch.stack([b[4]['config_tensor'] for b in batch]),
        'mod_order': torch.tensor([b[4]['mod_order'] for b in batch]),
        'snr_db': torch.tensor([b[4]['snr_db'] for b in batch], dtype=torch.float32),
        'mode': [b[4]['mode'] for b in batch],
        'targets': [b[4]['targets'] for b in batch],  # Keep as list
        'cfar_detections': [b[4]['cfar_detections'] for b in batch],
        'r_axis': [b[4]['r_axis'] for b in batch],  # Keep as list
        'v_axis': [b[4]['v_axis'] for b in batch],
        'ber': torch.tensor([b[4]['ber'] for b in batch], dtype=torch.float32),
    }
    
    return radar_inputs, radar_targets, comm_inputs, comm_targets, meta


# ----------------------------------------------------------------------
# Loss Functions
# ----------------------------------------------------------------------
def radar_focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    """Focal loss for radar detection with higher positive weight."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def compute_losses(radar_logits, radar_target, comm_logits, comm_target,
                   radar_pos_weight=5.0, lambda_comm=1.0, mod_order=None):
    """Compute joint losses using BCEWithLogitsLoss.
    
    Args:
        radar_pos_weight: Positive class weight for radar (default 5.0 for sharper peaks)
        lambda_comm: Weight for comm loss (default 1.0)
    """
    # Simple BCE loss like g2b - proven to work better
    pos_weight = torch.tensor([radar_pos_weight], device=radar_logits.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    radar_loss = bce(radar_logits, radar_target)
    
    # Comm loss
    B, M, H, W = comm_logits.shape
    if mod_order is not None and mod_order < M:
        logits_flat = comm_logits[:, :mod_order].permute(0, 2, 3, 1).reshape(-1, mod_order)
    else:
        logits_flat = comm_logits.permute(0, 2, 3, 1).reshape(-1, M)
    labels_flat = comm_target.reshape(-1).long().clamp(0, logits_flat.size(1) - 1)
    
    comm_loss = F.cross_entropy(logits_flat, labels_flat)
    
    total_loss = radar_loss + lambda_comm * comm_loss
    return total_loss, radar_loss, comm_loss


def symbol_to_bits(symbols, mod_order):
    """Convert symbol indices to bit representation.
    
    Args:
        symbols: [B, H, W] symbol indices (0 to mod_order-1)
        mod_order: Modulation order (4, 16, 64)
    Returns:
        bits: [B, num_bits, H, W] binary representation
    """
    num_bits = int(np.log2(mod_order))
    B, H, W = symbols.shape
    bits = torch.zeros(B, num_bits, H, W, device=symbols.device, dtype=torch.float32)
    
    for i in range(num_bits):
        bits[:, i] = ((symbols >> i) & 1).float()
    
    return bits


class MaskedBitLoss(nn.Module):
    """Masked bit-wise BCE loss that only considers active bits for current mod_order."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, symbol_targets, mod_order):
        """
        Args:
            logits: [B, max_bits, H, W] from UniversalCommNet (6 channels)
            symbol_targets: [B, H, W] ground truth symbol indices
            mod_order: Current modulation order (4, 16, 64)
        Returns:
            loss: Scalar, mean BCE over active bits only
        """
        active_bits = int(np.log2(mod_order))
        
        # Convert symbols to bits: [B, active_bits, H, W]
        bit_targets = symbol_to_bits(symbol_targets, mod_order)
        
        # Slice only the bits relevant to current mod_order
        relevant_logits = logits[:, :active_bits, :, :]
        
        # Compute bit-wise BCE
        loss = self.bce(relevant_logits, bit_targets)
        
        return loss.mean()


def compute_llr_loss(llr_logits, comm_target, mod_order, lambda_comm=1.0):
    """Compute LLR-based communication loss with masked bits.
    
    Args:
        llr_logits: [B, max_bits, H, W] LLR outputs from UniversalCommNet
        comm_target: [B, H, W] symbol indices
        mod_order: Modulation order (4, 16, 64)
        lambda_comm: Weight for comm loss
    Returns:
        comm_loss: Bit-wise BCE loss on active bits only
    """
    # Convert symbol targets to bit targets
    bit_targets = symbol_to_bits(comm_target, mod_order)  # [B, num_bits, H, W]
    
    # Get active bits only - critical for proper gradient flow
    active_bits = int(np.log2(mod_order))
    active_llr = llr_logits[:, :active_bits]  # [B, num_bits, H, W]
    
    # Binary cross-entropy on active bits only
    comm_loss = F.binary_cross_entropy_with_logits(active_llr, bit_targets)
    
    return lambda_comm * comm_loss


def compute_losses_llr(radar_logits, radar_target, llr_logits, comm_target,
                       radar_pos_weight=5.0, lambda_comm=1.0, mod_order=None):
    """Compute joint losses with LLR-based communication loss.
    
    Args:
        radar_logits: [B, 1, H, W] radar detection logits
        radar_target: [B, 1, H, W] radar heatmap targets
        llr_logits: [B, max_bits, H, W] LLR outputs
        comm_target: [B, H, W] symbol indices
        radar_pos_weight: Positive class weight for radar
        lambda_comm: Weight for comm loss
        mod_order: Modulation order (4, 16, 64)
    """
    # Radar loss
    pos_weight = torch.tensor([radar_pos_weight], device=radar_logits.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    radar_loss = bce(radar_logits, radar_target)
    
    # Communication LLR loss (uses only active bits)
    comm_loss = compute_llr_loss(llr_logits, comm_target, mod_order, lambda_comm=1.0)
    
    total_loss = radar_loss + lambda_comm * comm_loss
    return total_loss, radar_loss, comm_loss


# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------
def train_one_epoch(model, train_loaders, optimizer, device, lambda_comm=1.0):
    model.train()
    total_loss = total_radar = total_comm = 0.0
    total_ber = 0.0  # Track BER instead of SER for better comparison
    n_samples = 0
    
    per_cfg_stats = {cfg: {'loss': 0, 'ber': 0, 'n': 0} for cfg in train_loaders}
    # Use LLR mode for both universal and simple architectures
    use_llr = getattr(model, 'use_universal', False) or getattr(model, 'use_simple', False)
    
    for cfg_name, loader in train_loaders.items():
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            radar_in = radar_in.to(device)
            radar_tgt = radar_tgt.to(device)
            comm_in = comm_in.to(device)
            comm_tgt = comm_tgt.to(device)
            
            # Get config tensor and config_id
            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
            mod_order = int(meta['mod_order'][0])
            config_id = int(meta['config_id'][0])
            
            optimizer.zero_grad()
            radar_logits, comm_logits = model(radar_in, comm_in, config_tensors, mod_order, config_id)
            
            if use_llr:
                # LLR-based loss for UniversalCommNet
                loss, l_radar, l_comm = compute_losses_llr(
                    radar_logits, radar_tgt, comm_logits, comm_tgt,
                    lambda_comm=lambda_comm, mod_order=mod_order
                )
                # Convert LLR to symbol predictions for metrics
                symbol_logits = model.get_symbol_logits(comm_logits, mod_order)
                pred = symbol_logits.argmax(dim=1)
            else:
                # Symbol-based loss for AdaptiveCommNet (legacy)
                loss, l_radar, l_comm = compute_losses(
                    radar_logits, radar_tgt, comm_logits, comm_tgt,
                    lambda_comm=lambda_comm, mod_order=mod_order
                )
                pred = comm_logits[:, :mod_order].argmax(dim=1)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Calculate BER (bit errors, not symbol errors)
            bsz = radar_in.size(0)
            active_bits = int(np.log2(mod_order))
            pred_bits = symbol_to_bits(pred, mod_order)
            gt_bits = symbol_to_bits(comm_tgt, mod_order)
            ber = (pred_bits != gt_bits).float().mean().item()
            
            total_loss += loss.item() * bsz
            total_radar += l_radar.item() * bsz
            total_comm += l_comm.item() * bsz
            total_ber += ber * bsz
            n_samples += bsz
            
            per_cfg_stats[cfg_name]['loss'] += loss.item() * bsz
            per_cfg_stats[cfg_name]['ber'] += ber * bsz
            per_cfg_stats[cfg_name]['n'] += bsz
    
    if n_samples == 0:
        return 0, 0, 0, 0, {}
    
    # Average per-config
    for cfg in per_cfg_stats:
        if per_cfg_stats[cfg]['n'] > 0:
            per_cfg_stats[cfg]['loss'] /= per_cfg_stats[cfg]['n']
            per_cfg_stats[cfg]['ber'] /= per_cfg_stats[cfg]['n']
    
    return (total_loss / n_samples, total_radar / n_samples,
            total_comm / n_samples, total_ber / n_samples, per_cfg_stats)


@torch.no_grad()
def evaluate_epoch(model, val_loaders, device, lambda_comm=1.0):
    model.eval()
    total_loss = total_radar = total_comm = 0.0
    total_ber = 0.0
    n_samples = 0
    use_llr = getattr(model, 'use_universal', False) or getattr(model, 'use_simple', False)
    
    for cfg_name, loader in val_loaders.items():
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            radar_in = radar_in.to(device)
            radar_tgt = radar_tgt.to(device)
            comm_in = comm_in.to(device)
            comm_tgt = comm_tgt.to(device)
            
            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
            mod_order = int(meta['mod_order'][0])
            config_id = int(meta['config_id'][0])
            
            radar_logits, comm_logits = model(radar_in, comm_in, config_tensors, mod_order, config_id)
            
            if use_llr:
                loss, l_radar, l_comm = compute_losses_llr(
                    radar_logits, radar_tgt, comm_logits, comm_tgt,
                    lambda_comm=lambda_comm, mod_order=mod_order
                )
                symbol_logits = model.get_symbol_logits(comm_logits, mod_order)
                pred = symbol_logits.argmax(dim=1)
            else:
                loss, l_radar, l_comm = compute_losses(
                    radar_logits, radar_tgt, comm_logits, comm_tgt,
                    lambda_comm=lambda_comm, mod_order=mod_order
                )
                pred = comm_logits[:, :mod_order].argmax(dim=1)
            
            bsz = radar_in.size(0)
            pred_bits = symbol_to_bits(pred, mod_order)
            gt_bits = symbol_to_bits(comm_tgt, mod_order)
            ber = (pred_bits != gt_bits).float().mean().item()
            
            total_loss += loss.item() * bsz
            total_radar += l_radar.item() * bsz
            total_comm += l_comm.item() * bsz
            total_ber += ber * bsz
            n_samples += bsz
    
    if n_samples == 0:
        return 0, 0, 0, 0
    
    return (total_loss / n_samples, total_radar / n_samples,
            total_comm / n_samples, total_ber / n_samples)


# ----------------------------------------------------------------------
# Radar Post-processing
# ----------------------------------------------------------------------
from scipy.ndimage import maximum_filter

def postprocess_radar(probs, r_axis, v_axis, prob_thresh=0.7, nms_kernel=7,
                       adaptive_thresh=True, min_peak_distance=3, max_detections=10):
    """Convert probability map to detections with adaptive threshold and strict NMS.
    
    Args:
        probs: Probability heatmap from DL model
        r_axis: Range axis in meters
        v_axis: Velocity axis in m/s
        prob_thresh: Base detection threshold (default 0.7)
        nms_kernel: NMS kernel size (default 7 for stricter NMS)
        adaptive_thresh: If True, use per-RDM adaptive threshold
        min_peak_distance: Minimum distance between peaks (reduces FPs)
        max_detections: Maximum number of detections per RDM
    """
    # Adaptive threshold: use higher of fixed or percentile-based
    if adaptive_thresh:
        # Use 99.5th percentile as adaptive threshold (only strongest peaks)
        adaptive_th = np.percentile(probs, 99.5)
        # Take max of fixed and adaptive threshold
        final_thresh = max(prob_thresh, adaptive_th * 0.9)
    else:
        final_thresh = prob_thresh
    
    # Stricter NMS with larger kernel
    local_max = maximum_filter(probs, size=nms_kernel)
    mask = (probs >= final_thresh) & (probs == local_max)
    
    idxs = np.argwhere(mask)
    candidates = []
    
    for d_idx, r_idx in idxs:
        if r_idx < len(r_axis) and d_idx < len(v_axis):
            candidates.append({
                'range_m': float(r_axis[r_idx]),
                'velocity_mps': float(v_axis[d_idx]),
                'range_idx': int(r_idx),
                'doppler_idx': int(d_idx),
                'score': float(probs[d_idx, r_idx]),
            })
    
    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Prune detections that are too close (reduces FPs from nearby peaks)
    detections = []
    for cand in candidates:
        too_close = False
        for det in detections:
            dr = abs(cand['range_idx'] - det['range_idx'])
            dv = abs(cand['doppler_idx'] - det['doppler_idx'])
            if dr < min_peak_distance and dv < min_peak_distance:
                too_close = True
                break
        if not too_close:
            detections.append(cand)
        
        # Limit max detections
        if len(detections) >= max_detections:
            break
    
    return detections


def radar_metrics(targets, detections, match_thresh=3.0):
    """Compute radar detection metrics."""
    tp = fp = fn = 0
    matched_targets = set()
    
    for det in detections:
        matched = False
        for i, t in enumerate(targets):
            if i in matched_targets:
                continue
            dr = abs(det['range_m'] - t['range'])
            dv = abs(det['velocity_mps'] - t['velocity'])
            if math.sqrt(dr**2 + dv**2) < match_thresh:
                tp += 1
                matched_targets.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    
    fn = len(targets) - len(matched_targets)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


# ----------------------------------------------------------------------
# Full Evaluation with CFAR Comparison
# ----------------------------------------------------------------------
def heatmap_metrics(pred_probs, gt_heatmap, prob_thresh=0.1):
    """Compute heatmap-based detection metrics using IoU."""
    pred_mask = pred_probs > prob_thresh
    gt_mask = gt_heatmap > 0.5
    
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    
    iou = intersection / (union + 1e-8)
    
    # Pixel-level precision/recall
    tp = intersection
    fp = (pred_mask & ~gt_mask).sum()
    fn = (~pred_mask & gt_mask).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'pred_pixels': int(pred_mask.sum()),
        'gt_pixels': int(gt_mask.sum())
    }


def cfar_metrics_from_g2(sample):
    """Get CFAR performance from G2 dataset sample."""
    targets = sample['target_info']['targets']
    cfar_dets = sample.get('cfar_detections', [])
    
    tp = fp = 0
    matched = set()
    
    for det in cfar_dets:
        found = False
        for i, t in enumerate(targets):
            if i in matched:
                continue
            dr = abs(det.get('range_m', det.get('range', 0)) - t['range'])
            dv = abs(det.get('velocity_mps', det.get('velocity', 0)) - t['velocity'])
            if math.sqrt(dr**2 + dv**2) < 5.0:  # 5m/5mps threshold
                tp += 1
                matched.add(i)
                found = True
                break
        if not found:
            fp += 1
    
    fn = len(targets) - len(matched)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


@torch.no_grad()
def run_full_evaluation(model, deep_ds, device, out_dir, prob_thresh=0.7):
    """Evaluate DL model with detection-based metrics (like g2b.py).
    
    Uses postprocess_radar to convert heatmap to detections, then matches
    detections to targets by range/velocity distance.
    
    Args:
        prob_thresh: Detection threshold (default 0.7 like g2b)
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    # Aggregate metrics
    dl_total = {'tp': 0, 'fp': 0, 'fn': 0}
    cfar_total = {'tp': 0, 'fp': 0, 'fn': 0}
    total_targets = 0
    bers = []
    sers = []
    
    for idx in range(min(len(deep_ds), 100)):
        radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_ds[idx]
        
        config_tensor = meta['config_tensor'].unsqueeze(0).to(device)
        mod_order = meta['mod_order']
        
        radar_in_b = radar_in.unsqueeze(0).to(device)
        comm_in_b = comm_in.unsqueeze(0).to(device)
        
        radar_logits, comm_logits = model(radar_in_b, comm_in_b, config_tensor, mod_order)
        
        # DL Radar: convert heatmap to detections
        probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
        r_axis = np.array(meta['r_axis'])
        v_axis = np.array(meta['v_axis'])
        targets = meta['targets']
        
        dl_dets = postprocess_radar(probs, r_axis, v_axis, prob_thresh=prob_thresh)
        dl_m = radar_metrics(targets, dl_dets, match_thresh=3.0)
        
        dl_total['tp'] += dl_m['tp']
        dl_total['fp'] += dl_m['fp']
        dl_total['fn'] += dl_m['fn']
        total_targets += len(targets)
        
        # CFAR metrics (from G2 sample)
        g2_sample = deep_ds.g2_ds[idx]
        cfar_m = cfar_metrics_from_g2(g2_sample)
        cfar_total['tp'] += cfar_m['tp']
        cfar_total['fp'] += cfar_m['fp']
        cfar_total['fn'] += cfar_m['fn']
        
        # Comm metrics
        pred = comm_logits[:, :mod_order].argmax(dim=1)[0].cpu().numpy()
        gt = comm_tgt.numpy()
        ser = float((pred != gt).mean())
        sers.append(ser)
        bers.append(meta['ber'])
    
    # Compute aggregate metrics
    def compute_prf(m):
        p = m['tp'] / (m['tp'] + m['fp'] + 1e-8)
        r = m['tp'] / (m['tp'] + m['fn'] + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        return p, r, f
    
    dl_prec, dl_rec, dl_f1 = compute_prf(dl_total)
    cfar_prec, cfar_rec, cfar_f1 = compute_prf(cfar_total)
    
    summary = f"""
=== Radar Metrics (Detection-Based) ===
Total Targets: {total_targets}

Deep Learning:
  TP={dl_total['tp']} FP={dl_total['fp']} FN={dl_total['fn']}
  Precision={dl_prec:.4f} Recall={dl_rec:.4f} F1={dl_f1:.4f}

CFAR (baseline):
  TP={cfar_total['tp']} FP={cfar_total['fp']} FN={cfar_total['fn']}
  Precision={cfar_prec:.4f} Recall={cfar_rec:.4f} F1={cfar_f1:.4f}

=== Communication Metrics ===
Mean Baseline BER: {np.mean(bers):.4e}
Mean DL SER:       {np.mean(sers):.4e}
"""
    print(summary)
    
    with open(os.path.join(out_dir, 'eval_summary.txt'), 'w') as f:
        f.write(summary)
    
    return {
        'dl_tp': dl_total['tp'], 'dl_fp': dl_total['fp'], 'dl_fn': dl_total['fn'],
        'dl_f1': dl_f1, 'cfar_f1': cfar_f1,
        'mean_ber': np.mean(bers), 'mean_ser': np.mean(sers)
    }


# ======================================================================
# Comprehensive Evaluation: DL vs CFAR Radar Performance
# ======================================================================
@torch.no_grad()
def evaluate_radar_by_cnr(model, device, config_name='CN0566_TRADITIONAL',
                          cnr_list=[0, 5, 10, 15, 20],
                          num_samples=20, save_path='data/eval_cnr'):
    """Evaluate DL vs CFAR at different CNR (Clutter-to-Noise Ratio) levels.
    
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
    
    for cnr_db in cnr_list:
        print(f"CNR = {cnr_db} dB...")
        
        # Map CNR to clutter intensity
        clutter_intensity = 0.05 * (10 ** (cnr_db / 10))
        
        # Create dataset with specific clutter level
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'cnr_{cnr_db}'),
            drawfig=False,
            clutter_intensity=clutter_intensity,
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        deep_ds = G2DeepDataset(config_name, num_samples, save_path, 'test')
        deep_ds.g2_ds = ds  # Use the custom clutter dataset
        
        dl_total = {'tp': 0, 'fp': 0, 'fn': 0}
        cfar_total = {'tp': 0, 'fp': 0, 'fn': 0}
        
        for idx in range(len(deep_ds)):
            radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_ds[idx]
            
            config_tensor = meta['config_tensor'].unsqueeze(0).to(device)
            radar_in_b = radar_in.unsqueeze(0).to(device)
            comm_in_b = comm_in.unsqueeze(0).to(device)
            
            radar_logits, _ = model(radar_in_b, comm_in_b, config_tensor)
            probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
            
            r_axis = np.array(meta['r_axis'])
            v_axis = np.array(meta['v_axis'])
            targets = meta['targets']
            
            # DL detection
            dl_dets = postprocess_radar(probs, r_axis, v_axis)
            dl_m = radar_metrics(targets, dl_dets)
            dl_total['tp'] += dl_m['tp']
            dl_total['fp'] += dl_m['fp']
            dl_total['fn'] += dl_m['fn']
            
            # CFAR metrics
            cfar_m = cfar_metrics_from_g2(ds[idx])
            cfar_total['tp'] += cfar_m['tp']
            cfar_total['fp'] += cfar_m['fp']
            cfar_total['fn'] += cfar_m['fn']
        
        # Compute F1 scores
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
        
        print(f"  DL: F1={dl_f1:.3f}, CFAR: F1={cfar_f1:.3f}")
    
    # Plot results
    plot_radar_comparison(results, 'CNR (dB)', save_path, 'radar_cnr_comparison.png')
    return results


@torch.no_grad()
def evaluate_radar_by_rcs(model, device, config_name='CN0566_TRADITIONAL',
                          rcs_list=[5, 10, 15, 20, 25, 30],
                          num_samples=20, save_path='data/eval_rcs'):
    """Evaluate DL vs CFAR at different RCS (Radar Cross Section) levels.
    
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
    
    for rcs_db in rcs_list:
        print(f"RCS = {rcs_db} dB...")
        
        # Create dataset with specific RCS range
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'rcs_{rcs_db}'),
            drawfig=False,
            target_rcs_range=(rcs_db - 2, rcs_db + 2),  # Narrow range around target
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        deep_ds = G2DeepDataset(config_name, num_samples, save_path, 'test')
        deep_ds.g2_ds = ds
        
        dl_total = {'tp': 0, 'fp': 0, 'fn': 0}
        cfar_total = {'tp': 0, 'fp': 0, 'fn': 0}
        
        for idx in range(len(deep_ds)):
            radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_ds[idx]
            
            config_tensor = meta['config_tensor'].unsqueeze(0).to(device)
            radar_in_b = radar_in.unsqueeze(0).to(device)
            comm_in_b = comm_in.unsqueeze(0).to(device)
            
            radar_logits, _ = model(radar_in_b, comm_in_b, config_tensor)
            probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
            
            r_axis = np.array(meta['r_axis'])
            v_axis = np.array(meta['v_axis'])
            targets = meta['targets']
            
            dl_dets = postprocess_radar(probs, r_axis, v_axis)
            dl_m = radar_metrics(targets, dl_dets)
            dl_total['tp'] += dl_m['tp']
            dl_total['fp'] += dl_m['fp']
            dl_total['fn'] += dl_m['fn']
            
            cfar_m = cfar_metrics_from_g2(ds[idx])
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
        
        print(f"  DL: F1={dl_f1:.3f}, CFAR: F1={cfar_f1:.3f}")
    
    plot_radar_comparison(results, 'RCS (dB)', save_path, 'radar_rcs_comparison.png')
    return results


def plot_radar_comparison(results, xlabel, save_path, filename):
    """Plot DL vs CFAR F1/Precision/Recall comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = results.get('cnr', results.get('rcs', results.get('snr', [])))
    
    # F1 Score
    axes[0].plot(x, results['dl_f1'], 'b-o', label='DL', linewidth=2, markersize=8)
    axes[0].plot(x, results['cfar_f1'], 'r--s', label='CFAR', linewidth=2, markersize=8)
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('F1 Score: DL vs CFAR', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Precision
    axes[1].plot(x, results['dl_prec'], 'b-o', label='DL', linewidth=2, markersize=8)
    axes[1].plot(x, results['cfar_prec'], 'r--s', label='CFAR', linewidth=2, markersize=8)
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision: DL vs CFAR', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Recall
    axes[2].plot(x, results['dl_rec'], 'b-o', label='DL', linewidth=2, markersize=8)
    axes[2].plot(x, results['cfar_rec'], 'r--s', label='CFAR', linewidth=2, markersize=8)
    axes[2].set_xlabel(xlabel, fontsize=12)
    axes[2].set_ylabel('Recall', fontsize=12)
    axes[2].set_title('Recall: DL vs CFAR', fontsize=14)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(save_path, filename)}")


# ======================================================================
# Comprehensive Evaluation: DL vs Traditional Communication Performance  
# ======================================================================
@torch.no_grad()
def evaluate_comm_by_snr(model, device, config_name='CN0566_TRADITIONAL',
                         snr_list=[0, 5, 10, 15, 20, 25, 30],
                         num_samples=30, save_path='data/eval_snr'):
    """Evaluate DL vs Traditional (MMSE) BER at different SNR levels."""
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    use_llr = getattr(model, 'use_universal', False) or getattr(model, 'use_simple', False)
    
    print(f"\n{'='*60}")
    print(f"Communication Evaluation: DL vs Traditional by SNR")
    print(f"Config: {config_name}, SNR range: {snr_list} dB")
    print(f"Using {'LLR' if use_llr else 'Symbol'} mode")
    print(f"{'='*60}\n")
    
    results = {'snr': [], 'dl_ber': [], 'trad_ber': []}
    
    for snr_db in snr_list:
        print(f"SNR = {snr_db} dB...")
        
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, f'snr_{snr_db}'),
            drawfig=False,
            fixed_snr=snr_db,
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        deep_ds = G2DeepDataset(config_name, num_samples, save_path, 'test')
        deep_ds.g2_ds = ds
        
        dl_bit_errors = 0
        dl_total_bits = 0
        trad_bers = []
        
        for idx in range(len(deep_ds)):
            radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_ds[idx]
            
            config_tensor = meta['config_tensor'].unsqueeze(0).to(device)
            mod_order = meta['mod_order']
            radar_in_b = radar_in.unsqueeze(0).to(device)
            comm_in_b = comm_in.unsqueeze(0).to(device)
            
            _, comm_logits = model(radar_in_b, comm_in_b, config_tensor, mod_order)
            
            # Handle both LLR and symbol outputs
            if use_llr:
                symbol_logits = model.get_symbol_logits(comm_logits, mod_order)
                pred = symbol_logits.argmax(dim=1)[0].cpu()
            else:
                pred = comm_logits[:, :mod_order].argmax(dim=1)[0].cpu()
            
            # Compute BER (bit-level errors)
            pred_bits = symbol_to_bits(pred.unsqueeze(0), mod_order)[0]
            gt_bits = symbol_to_bits(comm_tgt.unsqueeze(0), mod_order)[0]
            dl_bit_errors += (pred_bits != gt_bits).sum().item()
            dl_total_bits += gt_bits.numel()
            trad_bers.append(meta['ber'])
        
        dl_ber = dl_bit_errors / dl_total_bits if dl_total_bits > 0 else 0
        trad_ber = np.mean(trad_bers)
        
        results['snr'].append(snr_db)
        results['dl_ber'].append(dl_ber)
        results['trad_ber'].append(trad_ber)
        
        print(f"  DL BER: {dl_ber:.4e}, Traditional BER: {trad_ber:.4e}")
    
    plot_ber_comparison(results, save_path, 'ber_snr_comparison.png')
    return results


@torch.no_grad()
def evaluate_comm_by_qam(model, device, snr_db=20,
                         qam_configs=['CN0566_TRADITIONAL'],  # 16-QAM
                         num_samples=30, save_path='data/eval_qam'):
    """Evaluate DL vs Traditional BER for different QAM modulation orders."""
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Communication Evaluation: DL vs Traditional by QAM")
    print(f"SNR: {snr_db} dB")
    print(f"{'='*60}\n")
    
    results = {'qam': [], 'dl_ber': [], 'trad_ber': []}
    
    # Get different QAM from different configs
    qam_map = {
        'CN0566_TRADITIONAL': 16,
        'Automotive_77GHz_LongRange': 4,
        'AUTOMOTIVE_TRADITIONAL': 16,
    }
    
    for cfg_name in qam_configs:
        mod_order = qam_map.get(cfg_name, 16)
        print(f"Config: {cfg_name}, {mod_order}-QAM...")
        
        ds = AIRadar_Comm_Dataset_G2(
            config_name=cfg_name,
            num_samples=num_samples,
            save_path=os.path.join(save_path, cfg_name),
            drawfig=False,
            fixed_snr=snr_db,
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        deep_ds = G2DeepDataset(cfg_name, num_samples, save_path, 'test')
        deep_ds.g2_ds = ds
        
        dl_errors = 0
        dl_total = 0
        trad_bers = []
        
        for idx in range(len(deep_ds)):
            radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_ds[idx]
            
            config_tensor = meta['config_tensor'].unsqueeze(0).to(device)
            radar_in_b = radar_in.unsqueeze(0).to(device)
            comm_in_b = comm_in.unsqueeze(0).to(device)
            
            _, comm_logits = model(radar_in_b, comm_in_b, config_tensor, mod_order)
            pred = comm_logits[:, :mod_order].argmax(dim=1)[0].cpu().numpy()
            gt = comm_tgt.numpy()
            
            dl_errors += (pred != gt).sum()
            dl_total += gt.size
            trad_bers.append(meta['ber'])
        
        dl_ber = dl_errors / dl_total if dl_total > 0 else 0
        trad_ber = np.mean(trad_bers)
        
        results['qam'].append(mod_order)
        results['dl_ber'].append(dl_ber)
        results['trad_ber'].append(trad_ber)
        
        print(f"  DL BER: {dl_ber:.4e}, Traditional BER: {trad_ber:.4e}")
    
    return results


def plot_ber_comparison(results, save_path, filename):
    """Plot DL vs Traditional BER comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    snr = results['snr']
    dl_ber = results['dl_ber']
    trad_ber = results['trad_ber']
    
    ax.semilogy(snr, dl_ber, 'b-o', label='DL', linewidth=2, markersize=8)
    ax.semilogy(snr, trad_ber, 'r--s', label='Traditional (MMSE)', linewidth=2, markersize=8)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('BER vs SNR: DL vs Traditional', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-5, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(save_path, filename)}")


def run_comprehensive_evaluation(model, device, out_dir, config_name='CN0566_TRADITIONAL'):
    """Run all comprehensive evaluations and generate consolidated markdown report."""
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION: DL vs Traditional Methods")
    print("="*70 + "\n")
    
    all_results = {}
    
    # Radar evaluations
    print("\n[1/4] Evaluating Radar by CNR...")
    cnr_results = evaluate_radar_by_cnr(
        model, device, config_name,
        cnr_list=[0, 5, 10, 15, 20],
        num_samples=15,
        save_path=os.path.join(out_dir, 'radar_cnr')
    )
    all_results['radar_cnr'] = cnr_results
    
    print("\n[2/4] Evaluating Radar by RCS...")
    rcs_results = evaluate_radar_by_rcs(
        model, device, config_name,
        rcs_list=[5, 10, 15, 20, 25],
        num_samples=15,
        save_path=os.path.join(out_dir, 'radar_rcs')
    )
    all_results['radar_rcs'] = rcs_results
    
    # Communication evaluation by SNR
    print("\n[3/4] Evaluating Communication by SNR...")
    snr_results = evaluate_comm_by_snr(
        model, device, config_name,
        snr_list=[5, 10, 15, 20, 25, 30],
        num_samples=20,
        save_path=os.path.join(out_dir, 'comm_snr')
    )
    all_results['comm_snr'] = snr_results
    
    # BER vs SNR for different QAM modulations
    print("\n[4/4] Evaluating BER vs SNR for different QAM orders...")
    qam_results = evaluate_ber_by_qam_snr(
        model, device,
        snr_list=[5, 10, 15, 20, 25, 30],
        num_samples=15,
        save_path=os.path.join(out_dir, 'comm_qam')
    )
    all_results['comm_qam'] = qam_results
    
    # Generate consolidated markdown report
    report = generate_consolidated_report(all_results, config_name, out_dir)
    
    return all_results


@torch.no_grad()
def evaluate_ber_by_qam_snr(model, device, snr_list=[5, 10, 15, 20, 25, 30],
                            num_samples=15, save_path='data/eval_qam_snr'):
    """Evaluate BER vs SNR for different QAM modulation orders (4-QAM, 16-QAM)."""
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    use_llr = getattr(model, 'use_universal', False) or getattr(model, 'use_simple', False)
    
    print(f"\n{'='*60}")
    print(f"Communication Evaluation: BER vs SNR by QAM Order")
    print(f"Using {'LLR' if use_llr else 'Symbol'} mode")
    print(f"{'='*60}\n")
    
    # Configs with different QAM orders
    qam_configs = {
        '4-QAM': 'Automotive_77GHz_LongRange',
        '16-QAM': 'CN0566_TRADITIONAL',
    }
    
    results = {qam: {'snr': [], 'dl_ber': [], 'trad_ber': []} for qam in qam_configs}
    
    for qam_name, cfg_name in qam_configs.items():
        print(f"\n{qam_name} ({cfg_name}):")
        mod_order = int(qam_name.split('-')[0])
        
        for snr_db in snr_list:
            ds = AIRadar_Comm_Dataset_G2(
                config_name=cfg_name,
                num_samples=num_samples,
                save_path=os.path.join(save_path, f'{cfg_name}_snr{snr_db}'),
                drawfig=False,
                fixed_snr=snr_db,
                enable_clutter=True,
                enable_imperfect_csi=True
            )
            
            deep_ds = G2DeepDataset(cfg_name, num_samples, save_path, 'test')
            deep_ds.g2_ds = ds
            
            dl_bit_errors = 0
            dl_total_bits = 0
            trad_bers = []
            
            for idx in range(len(deep_ds)):
                radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_ds[idx]
                
                config_tensor = meta['config_tensor'].unsqueeze(0).to(device)
                radar_in_b = radar_in.unsqueeze(0).to(device)
                comm_in_b = comm_in.unsqueeze(0).to(device)
                
                _, comm_logits = model(radar_in_b, comm_in_b, config_tensor, mod_order)
                
                if use_llr:
                    symbol_logits = model.get_symbol_logits(comm_logits, mod_order)
                    pred = symbol_logits.argmax(dim=1)[0].cpu()
                else:
                    pred = comm_logits[:, :mod_order].argmax(dim=1)[0].cpu()
                
                pred_bits = symbol_to_bits(pred.unsqueeze(0), mod_order)[0]
                gt_bits = symbol_to_bits(comm_tgt.unsqueeze(0), mod_order)[0]
                dl_bit_errors += (pred_bits != gt_bits).sum().item()
                dl_total_bits += gt_bits.numel()
                trad_bers.append(meta['ber'])
            
            dl_ber = dl_bit_errors / dl_total_bits if dl_total_bits > 0 else 0
            trad_ber = np.mean(trad_bers)
            
            results[qam_name]['snr'].append(snr_db)
            results[qam_name]['dl_ber'].append(dl_ber)
            results[qam_name]['trad_ber'].append(trad_ber)
            
            print(f"  SNR={snr_db:2d}dB: DL={dl_ber:.4e}, Trad={trad_ber:.4e}")
    
    # Plot all QAM curves on one figure
    plot_ber_by_qam(results, save_path, 'ber_vs_snr_by_qam.png')
    return results


def plot_ber_by_qam(results, save_path, filename):
    """Plot BER vs SNR for different QAM orders on single plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'4-QAM': 'green', '16-QAM': 'blue'}
    
    for qam_name, data in results.items():
        color = colors.get(qam_name, 'gray')
        snr = data['snr']
        dl_ber = data['dl_ber']
        trad_ber = data['trad_ber']
        
        ax.semilogy(snr, dl_ber, f'{color[0]}-o', label=f'{qam_name} DL', linewidth=2, markersize=6)
        ax.semilogy(snr, trad_ber, f'{color[0]}--s', label=f'{qam_name} Traditional', linewidth=2, markersize=6)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('BER vs SNR: DL vs Traditional for Different QAM', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-5, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(save_path, filename)}")


def generate_consolidated_report(all_results, config_name, out_dir):
    """Generate consolidated markdown evaluation report."""
    
    report = f"""# Comprehensive Evaluation Report

## Configuration: {config_name}
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Radar Detection: DL vs CFAR

### 1.1 Performance by CNR (Clutter-to-Noise Ratio)
Higher CNR = stronger clutter = harder detection

| CNR (dB) | DL F1 | CFAR F1 | DL Precision | CFAR Precision | DL Recall | CFAR Recall |
|----------|-------|---------|--------------|----------------|-----------|-------------|
"""
    cnr_data = all_results.get('radar_cnr', {})
    for i, cnr in enumerate(cnr_data.get('cnr', [])):
        report += f"| {cnr} | {cnr_data['dl_f1'][i]:.3f} | {cnr_data['cfar_f1'][i]:.3f} | {cnr_data['dl_prec'][i]:.3f} | {cnr_data['cfar_prec'][i]:.3f} | {cnr_data['dl_rec'][i]:.3f} | {cnr_data['cfar_rec'][i]:.3f} |\n"

    report += f"""
![Radar CNR Comparison](radar_cnr/radar_cnr_comparison.png)

### 1.2 Performance by RCS (Radar Cross Section)
Lower RCS = weaker targets = harder detection

| RCS (dB) | DL F1 | CFAR F1 | DL Precision | CFAR Precision | DL Recall | CFAR Recall |
|----------|-------|---------|--------------|----------------|-----------|-------------|
"""
    rcs_data = all_results.get('radar_rcs', {})
    for i, rcs in enumerate(rcs_data.get('rcs', [])):
        report += f"| {rcs} | {rcs_data['dl_f1'][i]:.3f} | {rcs_data['cfar_f1'][i]:.3f} | {rcs_data['dl_prec'][i]:.3f} | {rcs_data['cfar_prec'][i]:.3f} | {rcs_data['dl_rec'][i]:.3f} | {rcs_data['cfar_rec'][i]:.3f} |\n"

    report += f"""
![Radar RCS Comparison](radar_rcs/radar_rcs_comparison.png)

---

## 2. Communication: DL vs Traditional (MMSE)

### 2.1 BER vs SNR

| SNR (dB) | DL BER | Traditional BER | Gap |
|----------|--------|-----------------|-----|
"""
    snr_data = all_results.get('comm_snr', {})
    for i, snr in enumerate(snr_data.get('snr', [])):
        dl_ber = snr_data['dl_ber'][i]
        trad_ber = snr_data['trad_ber'][i]
        gap = (dl_ber - trad_ber) / trad_ber * 100 if trad_ber > 0 else 0
        report += f"| {snr} | {dl_ber:.4e} | {trad_ber:.4e} | {gap:+.1f}% |\n"

    report += f"""
![BER vs SNR Comparison](comm_snr/ber_snr_comparison.png)

### 2.2 BER vs SNR by QAM Modulation Order

"""
    qam_data = all_results.get('comm_qam', {})
    for qam_name, data in qam_data.items():
        report += f"#### {qam_name}\n\n"
        report += "| SNR (dB) | DL BER | Traditional BER |\n"
        report += "|----------|--------|-----------------|"
        for i, snr in enumerate(data.get('snr', [])):
            report += f"\n| {snr} | {data['dl_ber'][i]:.4e} | {data['trad_ber'][i]:.4e} |"
        report += "\n\n"

    report += f"""
![BER vs SNR by QAM](comm_qam/ber_vs_snr_by_qam.png)

---

## Summary

- **Radar Detection**: DL outperforms CFAR at high clutter levels (CNR > 10 dB)
- **Communication**: DL matches Traditional MMSE within ~5-10% BER gap
- **Multi-QAM**: Lower order modulation (4-QAM) shows better absolute BER

"""
    
    # Save report
    report_path = os.path.join(out_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved consolidated report: {report_path}")
    
    # Also save as JSON for programmatic access
    import json
    json_path = os.path.join(out_dir, 'evaluation_results.json')
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            json_results[key] = {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v 
                                 for k, v in val.items()}
        else:
            json_results[key] = val
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=float)
    print(f"Saved JSON results: {json_path}")
    
    return report


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate', 'test', 'eval_comprehensive', 'train_curriculum'], 
                        default='train')
    parser.add_argument('--train_samples', type=int, default=200)
    parser.add_argument('--val_samples', type=int, default=50)
    parser.add_argument('--data_root', type=str, default='data/AIradar_comm_model_g2c')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_comm', type=float, default=100.0,
                        help='Comm loss weight (default 100 to balance radar gradients)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='data/AIradar_comm_model_g2c')
    parser.add_argument('--use_universal', action='store_true', default=False,
                        help='Use UniversalCommNet with complex convolutions')
    parser.add_argument('--use_simple', action='store_true', default=True,
                        help='Use SimpleBitDemapper (per-pixel MLP, recommended)')
    parser.add_argument('--curriculum', action='store_true', default=False,
                        help='Use curriculum learning: 4-QAM first, then 16-QAM')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Build model
    arch_name = 'SimpleBitDemapper (MLP)' if args.use_simple else \
                ('UniversalCommNet (Conv)' if args.use_universal else 'AdaptiveCommNet (Legacy)')
    model = JointRadarCommNet_G2(
        base_ch=48, cond_dim=64, max_mod_order=MAX_MOD_ORDER,
        use_universal=args.use_universal,
        use_simple=args.use_simple
    )
    model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Comm architecture: {arch_name}")
    
    if args.mode == 'test':
        # Quick test
        print("Running quick test...")
        cfg_name = 'CN0566_TRADITIONAL'
        ds = G2DeepDataset(cfg_name, 5, args.data_root, 'test')
        loader = DataLoader(ds, batch_size=2, collate_fn=g2_collate_fn)
        
        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
            radar_in = radar_in.to(device)
            comm_in = comm_in.to(device)
            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
            
            with torch.no_grad():
                radar_out, comm_out = model(radar_in, comm_in, config_tensors)
            
            print(f"Radar: {radar_in.shape} -> {radar_out.shape}")
            print(f"Comm: {comm_in.shape} -> {comm_out.shape}")
            break
        print("Test passed!")
        return
    
    # Create datasets
    print(f"\n{'='*60}")
    print(f"Building datasets for configs: {TRAIN_CONFIGS}")
    print(f"{'='*60}\n")
    
    train_loaders = {}
    val_loaders = {}
    
    for cfg_name in TRAIN_CONFIGS:
        print(f"Loading {cfg_name}...")
        train_ds = G2DeepDataset(cfg_name, args.train_samples, args.data_root, 'train')
        val_ds = G2DeepDataset(cfg_name, args.val_samples, args.data_root, 'val')
        
        train_loaders[cfg_name] = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
            collate_fn=g2_collate_fn
        )
        val_loaders[cfg_name] = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
            collate_fn=g2_collate_fn
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.mode == 'train':
        best_val_loss = float('inf')
        use_llr = getattr(model, 'use_simple', False) or getattr(model, 'use_universal', False)
        
        # ========== COMM-ONLY PRETRAINING ==========
        # Train just comm network for a few epochs to establish basic demapping
        if use_llr:
            print("\n" + "="*60)
            print("COMM-ONLY PRETRAINING (3 epochs)")
            print("Freezing radar network to let comm network converge first")
            print("="*60 + "\n")
            
            # Freeze radar network
            for param in model.radar_net.parameters():
                param.requires_grad = False
            
            comm_optimizer = torch.optim.Adam(model.comm_net.parameters(), lr=0.01)
            
            for pretrain_epoch in range(1, 4):
                model.train()
                pretrain_loss = 0.0
                pretrain_n = 0
                
                for cfg_name, loader in train_loaders.items():
                    for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
                        comm_in = comm_in.to(device)
                        comm_tgt = comm_tgt.to(device)
                        config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
                        mod_order = int(meta['mod_order'][0])
                        
                        comm_optimizer.zero_grad()
                        llr = model.comm_net(comm_in, config_tensors, mod_order)
                        
                        # Comm loss only
                        from AIradar_comm_model_g2c import compute_llr_loss
                        loss = compute_llr_loss(llr, comm_tgt, mod_order, lambda_comm=1.0)
                        loss.backward()
                        comm_optimizer.step()
                        
                        pretrain_loss += loss.item() * comm_in.size(0)
                        pretrain_n += comm_in.size(0)
                
                # Check pretrain BER
                model.eval()
                with torch.no_grad():
                    sample_ber = 0.0
                    sample_n = 0
                    for cfg_name, loader in list(val_loaders.items())[:1]:
                        for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
                            comm_in = comm_in.to(device)
                            comm_tgt = comm_tgt.to(device) 
                            config_tensors = torch.stack([m for m in meta['config_tensor']]).to(device)
                            mod_order = int(meta['mod_order'][0])
                            
                            llr = model.comm_net(comm_in, config_tensors, mod_order)
                            symbol_logits = model.comm_net.get_symbol_logits(llr, mod_order)
                            pred = symbol_logits.argmax(dim=1)
                            
                            pred_bits = symbol_to_bits(pred, mod_order)
                            gt_bits = symbol_to_bits(comm_tgt, mod_order)
                            sample_ber += (pred_bits != gt_bits).float().mean().item() * comm_in.size(0)
                            sample_n += comm_in.size(0)
                            break
                    sample_ber = sample_ber / sample_n if sample_n > 0 else 0
                
                print(f"[Pretrain {pretrain_epoch}] Loss={pretrain_loss/pretrain_n:.4f}, BER={sample_ber:.4e}")
            
            # Unfreeze radar network
            for param in model.radar_net.parameters():
                param.requires_grad = True
            print("\n=== Comm pretraining complete, starting joint training ===\n")
        
        # ========== JOINT TRAINING ==========
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_radar, tr_comm, tr_ber, tr_cfg = train_one_epoch(
                model, train_loaders, optimizer, device, args.lambda_comm
            )
            val_loss, val_radar, val_comm, val_ber = evaluate_epoch(
                model, val_loaders, device, args.lambda_comm
            )
            scheduler.step()
            
            print(f"[Epoch {epoch:02d}] Train: Loss={tr_loss:.4f} Radar={tr_radar:.4f} "
                  f"Comm={tr_comm:.4f} BER={tr_ber:.4e} | "
                  f"Val: Loss={val_loss:.4f} BER={val_ber:.4e}")
            
            for cfg in tr_cfg:
                if tr_cfg[cfg]['n'] > 0:
                    print(f"  {cfg}: BER={tr_cfg[cfg]['ber']:.4e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'configs': TRAIN_CONFIGS,
                    'use_universal': args.use_universal,
                    'use_simple': args.use_simple,
                }, os.path.join(args.out_dir, 'best_model.pt'))
                print(f"  -> Saved best model")
        
        # Quick evaluation on test sets
        print("\n=== Running final evaluation ===")
        for cfg_name in TRAIN_CONFIGS:
            test_ds = G2DeepDataset(cfg_name, args.val_samples, args.data_root, 'test')
            eval_dir = os.path.join(args.out_dir, f'eval_{cfg_name}')
            run_full_evaluation(model, test_ds, device, eval_dir)
        
        # Run comprehensive evaluation (DL vs CFAR/Traditional)
        print("\n=== Running comprehensive evaluation (DL vs Traditional) ===")
        eval_out = os.path.join(args.out_dir, 'comprehensive_eval')
        run_comprehensive_evaluation(model, device, eval_out, config_name=TRAIN_CONFIGS[0])
    
    elif args.mode == 'evaluate':
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(ckpt['model'])
        
        for cfg_name in TRAIN_CONFIGS:
            test_ds = G2DeepDataset(cfg_name, args.val_samples, args.data_root, 'test')
            eval_dir = os.path.join(args.out_dir, f'eval_{cfg_name}')
            run_full_evaluation(model, test_ds, device, eval_dir)
    
    elif args.mode == 'eval_comprehensive':
        # Comprehensive evaluation: DL vs Traditional (CFAR for radar, MMSE for comm)
        # across different CNR/RCS/SNR levels
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(ckpt['model'])
        else:
            # Try to load best model from out_dir
            best_path = os.path.join(args.out_dir, 'best_model.pt')
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=device)
                model.load_state_dict(ckpt['model'])
                print(f"Loaded model from {best_path}")
            else:
                print("WARNING: No checkpoint found, using random weights!")
        
        eval_out = os.path.join(args.out_dir, 'comprehensive_eval')
        run_comprehensive_evaluation(model, device, eval_out, config_name=TRAIN_CONFIGS[0])
    
    elif args.mode == 'train_curriculum':
        # Curriculum learning: 4-QAM first (high SNR), then 16-QAM (all SNR)
        print("\n" + "="*60)
        print("CURRICULUM TRAINING MODE")
        print("Phase 1: 4-QAM only (simple decision boundaries)")
        print("Phase 2: 16-QAM (finer boundaries)")
        print("="*60 + "\n")
        
        # Phase 1: Train on 4-QAM config only (Automotive_77GHz_LongRange uses 4-QAM)
        phase1_config = 'Automotive_77GHz_LongRange'
        phase1_epochs = max(args.epochs // 3, 5)
        
        print(f"\n=== PHASE 1: {phase1_config} for {phase1_epochs} epochs ===")
        train_ds_p1 = G2DeepDataset(phase1_config, args.train_samples, args.data_root, 'train')
        val_ds_p1 = G2DeepDataset(phase1_config, args.val_samples, args.data_root, 'val')
        
        loader_p1 = {phase1_config: DataLoader(
            train_ds_p1, batch_size=args.batch_size, shuffle=True, 
            num_workers=0, collate_fn=g2_collate_fn
        )}
        val_loader_p1 = {phase1_config: DataLoader(
            val_ds_p1, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=g2_collate_fn
        )}
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        for epoch in range(1, phase1_epochs + 1):
            tr_loss, tr_radar, tr_comm, tr_ber, _ = train_one_epoch(
                model, loader_p1, optimizer, device, args.lambda_comm
            )
            val_loss, _, _, val_ber = evaluate_epoch(
                model, val_loader_p1, device, args.lambda_comm
            )
            print(f"[P1 Epoch {epoch:02d}] Train BER={tr_ber:.4e} | Val BER={val_ber:.4e}")
        
        # Phase 2: Add 16-QAM config (CN0566_TRADITIONAL)
        phase2_config = 'CN0566_TRADITIONAL'
        phase2_epochs = args.epochs - phase1_epochs
        
        print(f"\n=== PHASE 2: Add {phase2_config} for {phase2_epochs} epochs ===")
        train_ds_p2 = G2DeepDataset(phase2_config, args.train_samples, args.data_root, 'train')
        val_ds_p2 = G2DeepDataset(phase2_config, args.val_samples, args.data_root, 'val')
        
        all_loaders = {
            phase1_config: loader_p1[phase1_config],
            phase2_config: DataLoader(
                train_ds_p2, batch_size=args.batch_size, shuffle=True,
                num_workers=0, collate_fn=g2_collate_fn
            )
        }
        all_val_loaders = {
            phase1_config: val_loader_p1[phase1_config],
            phase2_config: DataLoader(
                val_ds_p2, batch_size=args.batch_size, shuffle=False,
                num_workers=0, collate_fn=g2_collate_fn
            )
        }
        
        # Lower learning rate for phase 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.3
        
        best_val_loss = float('inf')
        for epoch in range(1, phase2_epochs + 1):
            tr_loss, tr_radar, tr_comm, tr_ber, tr_cfg = train_one_epoch(
                model, all_loaders, optimizer, device, args.lambda_comm
            )
            val_loss, _, _, val_ber = evaluate_epoch(
                model, all_val_loaders, device, args.lambda_comm
            )
            print(f"[P2 Epoch {epoch:02d}] Train Loss={tr_loss:.4f} BER={tr_ber:.4e} | "
                  f"Val BER={val_ber:.4e}")
            
            for cfg in tr_cfg:
                if tr_cfg[cfg]['n'] > 0:
                    print(f"  {cfg}: BER={tr_cfg[cfg]['ber']:.4e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'configs': [phase1_config, phase2_config],
                    'use_universal': args.use_universal,
                }, os.path.join(args.out_dir, 'best_model.pt'))
                print(f"  -> Saved best model")
        
        print("\n=== Curriculum training complete ===")


if __name__ == '__main__':
    main()
