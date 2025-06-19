import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

# - A transformer-based model that processes time-domain radar data directly
# - Specialized attention mechanisms for range and Doppler dimensions
# - Optional learnable FFT for range-Doppler processing
# - CNN backbone for initial feature extraction

class RadarTransformerNet(nn.Module):
    """
    Transformer-based model for radar detection that processes time-domain radar data directly.
    This model combines CNN layers for initial feature extraction with transformer blocks for
    capturing long-range dependencies in both range and Doppler dimensions.
    """
    def __init__(self, 
                 num_rx=4,                # Number of receiver antennas
                 num_chirps=128,          # Number of chirps per frame
                 samples_per_chirp=1000,  # Number of samples per chirp
                 out_doppler_bins=128,    # Output Doppler bins
                 out_range_bins=256,      # Output range bins
                 dim=256,                 # Model dimension
                 depth=6,                 # Number of transformer blocks
                 heads=8,                 # Number of attention heads
                 mlp_dim=512,            # MLP hidden dimension
                 dropout=0.1,
                 use_learnable_fft=True,  # Whether to use learnable FFT
                 use_cnn_backbone=True):  # Whether to use CNN backbone
        super().__init__()
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.out_doppler_bins = out_doppler_bins
        self.out_range_bins = out_range_bins
        self.use_learnable_fft = use_learnable_fft
        self.use_cnn_backbone = use_cnn_backbone
        
        # Initial 3D convolutional layers for processing time-domain data
        # Input shape: [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
        # 2 channels for real and imaginary parts
        self.time_preprocess = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(1, 3, 5), padding=(0, 1, 2), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 5), padding=(0, 1, 2), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 5), padding=(0, 1, 2), stride=(1, 1, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        
        # CNN backbone for feature extraction
        if use_cnn_backbone:
            self.cnn_backbone = nn.Sequential(
                # Process across receivers
                nn.Conv3d(64, 64, kernel_size=(num_rx, 1, 1), stride=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                # Further reduce time dimension
                nn.Conv3d(64, 128, kernel_size=(1, 1, 5), stride=(1, 1, 2), padding=(0, 0, 2)),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                # Final feature extraction
                nn.Conv3d(128, dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(dim),
                nn.ReLU(),
            )
        
        # Calculate the size after CNN processing
        # This will depend on the exact architecture and stride/padding values
        self.reduced_chirps = num_chirps // 2 if use_cnn_backbone else num_chirps
        self.reduced_samples = samples_per_chirp // 4 if use_cnn_backbone else samples_per_chirp // 2
        
        # Learnable FFT weights for range dimension
        if use_learnable_fft:
            self.fft_weights_range = nn.Parameter(
                torch.randn(self.reduced_samples, out_range_bins, 2)  # 2 for real and imaginary parts
            )
            # Learnable FFT weights for Doppler dimension
            self.fft_weights_doppler = nn.Parameter(
                torch.randn(self.reduced_chirps, out_doppler_bins, 2)  # 2 for real and imaginary parts
            )
        
        # Positional encoding with learnable parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, 1))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Output processing
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Sigmoid()  # For binary detection
        )
    
    def complex_multiply(self, x, y):
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # x, y shape: [..., 2] where last dim is [real, imag]
        real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
        imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        return torch.stack([real, imag], dim=-1)
    
    def learnable_fft(self, x):
        """
        Apply learnable FFT to convert time-domain data to range-Doppler map
        x shape: [batch_size, dim, reduced_chirps, reduced_samples]
        """
        batch_size, dim, chirps, samples = x.shape
        
        # Reshape for processing
        x = x.permute(0, 1, 2, 3).reshape(batch_size * dim, chirps, samples)
        
        # Add complex dimension
        x = torch.stack([x, torch.zeros_like(x)], dim=-1)  # [batch_size*dim, chirps, samples, 2]
        
        # Apply range FFT (time to range)
        range_fft = torch.zeros(batch_size * dim, chirps, self.out_range_bins, 2, device=x.device)
        for i in range(chirps):
            for j in range(self.out_range_bins):
                # Weighted sum for this range bin
                weighted_sum = self.complex_multiply(x[:, i, :, :], self.fft_weights_range[:, j, :].unsqueeze(0))
                range_fft[:, i, j, 0] = weighted_sum[..., 0].sum(dim=1)  # Real part
                range_fft[:, i, j, 1] = weighted_sum[..., 1].sum(dim=1)  # Imaginary part
        
        # Apply Doppler FFT (chirp to velocity)
        doppler_fft = torch.zeros(batch_size * dim, self.out_doppler_bins, self.out_range_bins, 2, device=x.device)
        for i in range(self.out_doppler_bins):
            for j in range(self.out_range_bins):
                # Weighted sum for this Doppler bin
                weighted_sum = self.complex_multiply(range_fft[:, :, j, :], self.fft_weights_doppler[:, i, :].unsqueeze(0))
                doppler_fft[:, i, j, 0] = weighted_sum[..., 0].sum(dim=1)  # Real part
                doppler_fft[:, i, j, 1] = weighted_sum[..., 1].sum(dim=1)  # Imaginary part
        
        # Calculate magnitude
        magnitude = torch.sqrt(doppler_fft[..., 0]**2 + doppler_fft[..., 1]**2)
        
        # Normalize
        magnitude = magnitude / (samples * chirps)
        
        # Reshape back
        magnitude = magnitude.reshape(batch_size, dim, self.out_doppler_bins, self.out_range_bins)
        
        return magnitude
    
    def forward(self, x):
        """
        Forward pass for the model
        x shape: [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
        where the last dimension contains real and imaginary parts
        """
        batch_size = x.shape[0]
        
        # Initial 3D convolutional processing
        # Permute to [batch_size, 2, num_rx, num_chirps, samples_per_chirp]
        x = x.permute(0, 4, 1, 2, 3)
        x = self.time_preprocess(x)  # [batch_size, 64, num_rx, num_chirps, samples_per_chirp/2]
        
        # CNN backbone if enabled
        if self.use_cnn_backbone:
            x = self.cnn_backbone(x)  # [batch_size, dim, 1, reduced_chirps, reduced_samples]
            # Remove receiver dimension which is now 1
            x = x.squeeze(2)  # [batch_size, dim, reduced_chirps, reduced_samples]
        else:
            # If no CNN backbone, just reshape after initial processing
            x = x.mean(dim=2)  # Average across receivers
        
        # Apply learnable FFT if enabled
        if self.use_learnable_fft:
            fft_output = self.learnable_fft(x)  # [batch_size, dim, out_doppler_bins, out_range_bins]
        
        # Reshape for transformer processing
        # We need to reshape to [batch_size, dim, height, width]
        # where height=out_doppler_bins and width=out_range_bins
        if self.use_cnn_backbone:
            # Reshape CNN output to match transformer input
            x = F.adaptive_avg_pool3d(x.unsqueeze(2), (1, self.out_doppler_bins, self.out_range_bins))
            x = x.squeeze(2)  # [batch_size, dim, out_doppler_bins, out_range_bins]
        else:
            # If no CNN backbone, use a projection layer
            x = x.permute(0, 1, 2, 3)  # [batch_size, dim, reduced_chirps, reduced_samples]
            x = F.adaptive_avg_pool2d(x, (self.out_doppler_bins, self.out_range_bins))
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Combine with FFT output if enabled
        if self.use_learnable_fft:
            x = x + fft_output
        
        # Output processing
        x = self.output_proj(x)  # [batch_size, 1, out_doppler_bins, out_range_bins]
        
        # Reshape to match expected output format [batch_size, out_doppler_bins, out_range_bins, 1]
        x = x.permute(0, 2, 3, 1)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.GroupNorm(1, dim)  # GroupNorm with 1 group is equivalent to LayerNorm for 4D tensors
        self.norm2 = nn.GroupNorm(1, dim)
        self.norm3 = nn.GroupNorm(1, dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(dim, heads, dropout)
        
        # Specialized attention for range and Doppler dimensions
        self.range_attention = RangeAttention(dim, heads // 2, dropout)
        self.doppler_attention = DopplerAttention(dim, heads // 2, dropout)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_dim, dim, 1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Regular self-attention
        norm_x = self.norm1(x)
        x = x + self.attention(norm_x)
        
        # Range and Doppler attention
        norm_x = self.norm2(x)
        x_range = self.range_attention(norm_x)
        x_doppler = self.doppler_attention(norm_x)
        x = x + x_range + x_doppler
        
        # MLP
        norm_x = self.norm3(x)
        x = x + self.mlp(norm_x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V with convolutions
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.heads, C // self.heads, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Reshape for attention computation
        q = q.reshape(B, self.heads, C // self.heads, H * W)
        k = k.reshape(B, self.heads, C // self.heads, H * W)
        v = v.reshape(B, self.heads, C // self.heads, H * W)
        
        # Compute attention
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, C, H, W)
        
        # Project back to original dimension
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class RangeAttention(nn.Module):
    """Special attention mechanism that focuses on the range dimension"""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.heads, C // self.heads, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Reshape to focus on range dimension (W)
        q = q.permute(0, 1, 2, 4, 3).reshape(B * self.heads * H, C // self.heads, W)
        k = k.permute(0, 1, 2, 4, 3).reshape(B * self.heads * H, C // self.heads, W)
        v = v.permute(0, 1, 2, 4, 3).reshape(B * self.heads * H, C // self.heads, W)
        
        # Compute attention along range dimension
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, self.heads, H, C // self.heads, W).permute(0, 3, 1, 4, 2).reshape(B, C, H, W)
        
        # Project back
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class DopplerAttention(nn.Module):
    """Special attention mechanism that focuses on the Doppler dimension"""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.heads, C // self.heads, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Reshape to focus on Doppler dimension (H)
        q = q.permute(0, 1, 3, 2, 4).reshape(B * self.heads * W, C // self.heads, H)
        k = k.permute(0, 1, 3, 2, 4).reshape(B * self.heads * W, C // self.heads, H)
        v = v.permute(0, 1, 3, 2, 4).reshape(B * self.heads * W, C // self.heads, H)
        
        # Compute attention along Doppler dimension
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, self.heads, W, C // self.heads, H).permute(0, 3, 4, 1, 2).reshape(B, C, H, W)
        
        # Project back
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

def test_model():
    # Test the model with random input
    batch_size = 2
    num_rx = 4
    num_chirps = 128
    samples_per_chirp = 1000
    out_doppler_bins = 128
    out_range_bins = 256
    
    # Create random input tensor
    x = torch.randn(batch_size, num_rx, num_chirps, samples_per_chirp, 2)
    
    # Initialize model
    model = RadarTransformerNet(
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp,
        out_doppler_bins=out_doppler_bins,
        out_range_bins=out_range_bins,
        dim=128,
        depth=4,
        heads=8,
        mlp_dim=256,
        dropout=0.1,
        use_learnable_fft=True,
        use_cnn_backbone=True
    )
    
    # Forward pass
    output = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output has the expected shape
    expected_shape = (batch_size, out_doppler_bins, out_range_bins, 1)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("Model test passed!")

if __name__ == "__main__":
    test_model()