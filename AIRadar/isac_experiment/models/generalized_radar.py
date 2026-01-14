"""
Generalized Radar Model with FiLM Conditioning

This model can handle multiple radar configurations by using:
1. Config embeddings to encode radar parameters (fc, B, N, M, etc.)
2. FiLM (Feature-wise Linear Modulation) for config-adaptive processing
3. Adaptive pooling to handle variable input sizes

Designed to train on multiple configs and generalize to unseen configs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.
    
    Applies learned affine transformation conditioned on config embedding:
    y = gamma(cond) * x + beta(cond)
    """
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)
        
        # Initialize to identity transform
        nn.init.ones_(self.gamma_fc.weight.data[:, 0])
        nn.init.zeros_(self.gamma_fc.weight.data[:, 1:])
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)
        
    def forward(self, x, cond):
        """
        Args:
            x: Feature tensor (B, C, H, W)
            cond: Condition vector (B, cond_dim)
        Returns:
            Modulated features (B, C, H, W)
        """
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return gamma * x + beta


class FiLMConvBlock(nn.Module):
    """Convolution block with FiLM conditioning."""
    def __init__(self, in_ch, out_ch, cond_dim, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.norm = nn.GroupNorm(8, out_ch)
        self.film = FiLMLayer(out_ch, cond_dim)
        
    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)
        return F.relu(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1)
        
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConfigEncoder(nn.Module):
    """Encode radar configuration parameters to embedding vector.
    
    Input config dict should contain:
        - fc: carrier frequency (Hz)
        - radar_B: bandwidth (Hz)
        - radar_Nc: number of chirps
        - radar_Ns: samples per chirp
        - mod_order: modulation order (4, 16, 64)
        - channel_model: 'awgn', 'multipath', 'rayleigh'
    """
    CHANNEL_MAP = {'awgn': 0, 'multipath': 1, 'rayleigh': 2, 'none': 3}
    
    def __init__(self, embed_dim=128):
        super().__init__()
        # 6 input features: fc, B, Nc, Ns, mod_order, channel
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, config_tensor):
        """
        Args:
            config_tensor: (B, 6) normalized config features
        Returns:
            Config embedding (B, embed_dim)
        """
        return self.fc(config_tensor)
    
    @staticmethod
    def encode_config(config: dict) -> torch.Tensor:
        """Convert config dict to normalized tensor."""
        channel_id = ConfigEncoder.CHANNEL_MAP.get(config.get('channel_model', 'multipath'), 1)
        return torch.tensor([
            config.get('fc', 10e9) / 1e11,              # ~0.1-0.8
            config.get('radar_B', 500e6) / 1e9,         # ~0.1-1.0
            config.get('radar_Nc', 64) / 256,           # ~0.25-1.0
            config.get('radar_Ns', 1000) / 4000,        # ~0.25-1.0
            np.log2(config.get('mod_order', 16)) / 6,   # ~0.33-1.0
            channel_id / 3                               # 0-1
        ], dtype=torch.float32)


class GeneralizedRadarNet(nn.Module):
    """Generalized Radar Detection Network.
    
    Uses FiLM conditioning to adapt to different radar configurations.
    Single model handles multiple fc, bandwidth, FFT sizes, etc.
    
    Architecture:
        Input: (B, 1, H, W) - Range-Doppler map
        Config: (B, 6) - Radar config embedding
        Output: (B, 1, H, W) - Target detection heatmap
    """
    
    def __init__(self, in_ch=1, base_ch=48, cond_dim=128, target_size=(256, 256)):
        super().__init__()
        self.target_size = target_size
        
        # Config encoder
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        
        # Encoder with FiLM conditioning
        self.enc1 = FiLMConvBlock(in_ch, base_ch, cond_dim)
        self.enc2 = FiLMConvBlock(base_ch, base_ch*2, cond_dim, stride=2)
        self.enc3 = FiLMConvBlock(base_ch*2, base_ch*4, cond_dim, stride=2)
        self.enc4 = FiLMConvBlock(base_ch*4, base_ch*8, cond_dim, stride=2)
        
        # SE attention at bottleneck
        self.se = SEBlock(base_ch*8)
        
        # Decoder path
        self.dec4 = FiLMConvBlock(base_ch*8, base_ch*4, cond_dim)
        self.dec3 = FiLMConvBlock(base_ch*4 + base_ch*4, base_ch*2, cond_dim)  # Skip connection
        self.dec2 = FiLMConvBlock(base_ch*2 + base_ch*2, base_ch, cond_dim)
        self.dec1 = FiLMConvBlock(base_ch + base_ch, base_ch, cond_dim)
        
        # Output head
        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        
    def forward(self, x, config_tensor):
        """
        Args:
            x: Input RDM (B, 1, H, W)
            config_tensor: Config features (B, 6)
        Returns:
            Detection heatmap logits (B, 1, H, W)
        """
        B = x.size(0)
        orig_size = x.shape[-2:]
        
        # Normalize input size for consistent processing
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Get config embedding
        cond = self.config_encoder(config_tensor)  # (B, cond_dim)
        
        # Encoder with skip connections
        e1 = self.enc1(x, cond)    # (B, 48, H, W)
        e2 = self.enc2(e1, cond)   # (B, 96, H/2, W/2)
        e3 = self.enc3(e2, cond)   # (B, 192, H/4, W/4)
        e4 = self.enc4(e3, cond)   # (B, 384, H/8, W/8)
        
        # Bottleneck with SE attention
        z = self.se(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(z, cond)
        d4 = F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d4, e3], dim=1), cond)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1), cond)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1), cond)
        
        # Output
        logits = self.out_conv(d1)
        
        # Restore original size if needed
        if logits.shape[-2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
            
        return logits
    
    def forward_with_config(self, x, config: dict):
        """Convenience method that encodes config dict automatically."""
        config_tensor = ConfigEncoder.encode_config(config).unsqueeze(0).to(x.device)
        config_tensor = config_tensor.expand(x.size(0), -1)
        return self.forward(x, config_tensor)


def radar_focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Focal loss for radar target detection.
    
    Handles class imbalance (few targets, many background).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    return (focal_weight * bce).mean()


# Test code
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = GeneralizedRadarNet(base_ch=48, cond_dim=128).to(device)
    print(f"GeneralizedRadarNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different input sizes and configs
    configs = [
        {'fc': 10.5e9, 'radar_B': 500e6, 'radar_Nc': 64, 'radar_Ns': 1000, 'mod_order': 16, 'channel_model': 'multipath'},
        {'fc': 77e9, 'radar_B': 2.5e9, 'radar_Nc': 128, 'radar_Ns': 512, 'mod_order': 4, 'channel_model': 'awgn'},
    ]
    
    for i, cfg in enumerate(configs):
        x = torch.randn(2, 1, 64, 256).to(device)
        config_tensor = ConfigEncoder.encode_config(cfg).unsqueeze(0).expand(2, -1).to(device)
        
        out = model(x, config_tensor)
        print(f"Config {i+1}: Input {x.shape} -> Output {out.shape}")
