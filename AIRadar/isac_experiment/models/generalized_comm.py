"""
Generalized Communication Model with Config Conditioning

This model handles multiple modulation orders and channel conditions:
1. Adaptive demapper for 4-QAM, 16-QAM, 64-QAM
2. SNR-aware processing for noise estimation
3. Channel model conditioning for fading compensation

Designed to train on multiple configs and generalize to unseen conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CommConfigEncoder(nn.Module):
    """Encode communication configuration parameters.
    
    Input features:
        - mod_order: modulation order (4, 16, 64)
        - snr_db: signal-to-noise ratio
        - channel_type: 'awgn', 'multipath', 'rayleigh'
        - with_fec: whether FEC is enabled (0/1)
    """
    CHANNEL_MAP = {'awgn': 0, 'multipath': 1, 'rayleigh': 2, 'tdl_a': 3, 'tdl_d': 4}
    
    def __init__(self, embed_dim=64):
        super().__init__()
        # 4 input features
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, config_tensor):
        """
        Args:
            config_tensor: (B, 4) normalized config features
        Returns:
            Config embedding (B, embed_dim)
        """
        return self.fc(config_tensor)
    
    @staticmethod
    def encode_config(mod_order: int, snr_db: float, channel_model: str = 'multipath', with_fec: bool = False) -> torch.Tensor:
        """Convert config to normalized tensor."""
        channel_id = CommConfigEncoder.CHANNEL_MAP.get(channel_model, 1)
        return torch.tensor([
            np.log2(mod_order) / 6,       # 0.33 for 4-QAM, 0.67 for 16-QAM, 1.0 for 64-QAM
            snr_db / 40,                  # Normalize to ~0-1
            channel_id / 4,               # 0-1
            float(with_fec)               # 0 or 1
        ], dtype=torch.float32)


class FiLMBlock1D(nn.Module):
    """FiLM conditioning for 1D conv layers."""
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)
        
        nn.init.ones_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.bias.data)
        
    def forward(self, x, cond):
        # x: (B, C, L) or (B, C, H, W)
        gamma = self.gamma_fc(cond)
        beta = self.beta_fc(cond)
        
        # Expand to match x dimensions
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            
        return gamma * x + beta


class GeneralizedCommNet(nn.Module):
    """Generalized Communication Demapper Network.
    
    Uses config conditioning to adapt to different modulations and channels.
    Outputs per-bit log-likelihood ratios (LLRs) for soft decoding.
    
    Architecture:
        Input: (B, 2, L) - Complex received symbols (I/Q channels)
        Config: (B, 4) - Comm config embedding
        Output: (B, max_bits, L) - LLRs for each bit position
    
    Args:
        max_mod_bits: Maximum bits per symbol (6 for 64-QAM)
        base_ch: Base channel width
        cond_dim: Config embedding dimension
    """
    
    def __init__(self, max_mod_bits=6, base_ch=64, cond_dim=64):
        super().__init__()
        self.max_mod_bits = max_mod_bits
        
        # Config encoder
        self.config_encoder = CommConfigEncoder(embed_dim=cond_dim)
        
        # Input projection (2 channels: I/Q)
        self.input_conv = nn.Conv1d(2, base_ch, kernel_size=1)
        
        # Encoder with FiLM conditioning
        self.enc1 = nn.Conv1d(base_ch, base_ch, kernel_size=3, padding=1)
        self.film1 = FiLMBlock1D(base_ch, cond_dim)
        
        self.enc2 = nn.Conv1d(base_ch, base_ch*2, kernel_size=3, padding=1)
        self.film2 = FiLMBlock1D(base_ch*2, cond_dim)
        
        self.enc3 = nn.Conv1d(base_ch*2, base_ch*2, kernel_size=3, padding=1)
        self.film3 = FiLMBlock1D(base_ch*2, cond_dim)
        
        # Output head - produces LLRs for each bit
        self.out_conv = nn.Conv1d(base_ch*2, max_mod_bits, kernel_size=1)
        
        # Modulation-aware masking (for variable output bits)
        # 4-QAM: 2 bits, 16-QAM: 4 bits, 64-QAM: 6 bits
        self.register_buffer('bit_mask', torch.ones(max_mod_bits))
        
    def forward(self, x, config_tensor, return_mask=False):
        """
        Args:
            x: Complex symbols (B, 2, L) where channel 0=I, 1=Q
            config_tensor: (B, 4) normalized config
            return_mask: If True, also return valid bit mask
        Returns:
            LLRs: (B, max_bits, L) log-likelihood ratios
            mask: (B, max_bits) binary mask for valid bits (if return_mask=True)
        """
        B, _, L = x.shape
        
        # Get config embedding
        cond = self.config_encoder(config_tensor)  # (B, cond_dim)
        
        # Determine actual bits per symbol from config (mod_order encoded in config_tensor[:, 0])
        # config_tensor[:, 0] = log2(mod_order) / 6
        bits_per_symbol = (config_tensor[:, 0] * 6).round().long().clamp(2, 6)  # (B,)
        
        # Forward pass with FiLM conditioning
        h = self.input_conv(x)
        h = F.relu(self.film1(self.enc1(h), cond))
        h = F.relu(self.film2(self.enc2(h), cond))
        h = F.relu(self.film3(self.enc3(h), cond))
        
        # Output LLRs
        llrs = self.out_conv(h)  # (B, max_bits, L)
        
        if return_mask:
            # Create mask for valid bits based on modulation order
            mask = torch.zeros(B, self.max_mod_bits, device=x.device)
            for i in range(B):
                mask[i, :bits_per_symbol[i]] = 1.0
            return llrs, mask
        
        return llrs
    
    def forward_with_config(self, x, mod_order: int, snr_db: float, channel_model: str = 'multipath'):
        """Convenience method with config dict."""
        config_tensor = CommConfigEncoder.encode_config(mod_order, snr_db, channel_model)
        config_tensor = config_tensor.unsqueeze(0).expand(x.size(0), -1).to(x.device)
        return self.forward(x, config_tensor)


class GeneralizedCommNet2D(nn.Module):
    """2D version for grid-based demapping (OFDM/OTFS grids).
    
    Input: (B, 2, H, W) - Received OFDM/OTFS grid (I/Q channels)
    Output: (B, max_bits, H, W) - LLRs per subcarrier/delay-Doppler bin
    """
    
    def __init__(self, max_mod_bits=6, base_ch=64, cond_dim=64):
        super().__init__()
        self.max_mod_bits = max_mod_bits
        
        # Config encoder
        self.config_encoder = CommConfigEncoder(embed_dim=cond_dim)
        
        # 2D CNN with FiLM
        self.conv1 = nn.Conv2d(2, base_ch, 3, padding=1)
        self.film1 = FiLMBlock1D(base_ch, cond_dim)
        
        self.conv2 = nn.Conv2d(base_ch, base_ch*2, 3, padding=1)
        self.film2 = FiLMBlock1D(base_ch*2, cond_dim)
        
        self.conv3 = nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1)
        self.film3 = FiLMBlock1D(base_ch*2, cond_dim)
        
        self.out_conv = nn.Conv2d(base_ch*2, max_mod_bits, 1)
        
    def forward(self, x, config_tensor):
        """
        Args:
            x: (B, 2, H, W) received grid
            config_tensor: (B, 4) config
        Returns:
            (B, max_bits, H, W) LLRs
        """
        cond = self.config_encoder(config_tensor)
        
        h = F.relu(self.film1(self.conv1(x), cond))
        h = F.relu(self.film2(self.conv2(h), cond))
        h = F.relu(self.film3(self.conv3(h), cond))
        
        return self.out_conv(h)


def comm_bce_loss(llrs, bits, mask=None):
    """Binary cross-entropy loss for LLR prediction.
    
    Args:
        llrs: Predicted LLRs (B, max_bits, L)
        bits: Ground truth bits (B, max_bits, L)
        mask: Valid bit mask (B, max_bits) optional
    """
    loss = F.binary_cross_entropy_with_logits(llrs, bits, reduction='none')
    
    if mask is not None:
        # Apply mask to ignore invalid bits
        mask = mask.unsqueeze(-1)  # (B, max_bits, 1)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    
    return loss.mean()


# Test code
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test 1D model
    model_1d = GeneralizedCommNet(max_mod_bits=6, base_ch=64).to(device)
    print(f"GeneralizedCommNet (1D) parameters: {sum(p.numel() for p in model_1d.parameters()):,}")
    
    # Test with different configs
    for mod_order, snr in [(4, 5), (16, 15), (64, 25)]:
        x = torch.randn(4, 2, 100).to(device)  # 100 symbols
        config = CommConfigEncoder.encode_config(mod_order, snr).unsqueeze(0).expand(4, -1).to(device)
        
        llrs, mask = model_1d(x, config, return_mask=True)
        print(f"Mod={mod_order}-QAM, SNR={snr}dB: Input {x.shape} -> LLRs {llrs.shape}, Valid bits: {mask[0].sum().int()}")
    
    # Test 2D model
    model_2d = GeneralizedCommNet2D(max_mod_bits=6).to(device)
    print(f"\nGeneralizedCommNet (2D) parameters: {sum(p.numel() for p in model_2d.parameters()):,}")
    
    x_grid = torch.randn(2, 2, 64, 256).to(device)  # OFDM grid
    config = CommConfigEncoder.encode_config(16, 20).unsqueeze(0).expand(2, -1).to(device)
    out = model_2d(x_grid, config)
    print(f"2D Model: Input {x_grid.shape} -> Output {out.shape}")
