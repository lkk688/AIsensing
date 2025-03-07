import torch
import torch.nn as nn
import torch.nn.functional as F
#add the einops package for the Rearrange layer. install it with pip install einops
from einops.layers.torch import Rearrange

class DualPurposeTransformer(nn.Module):
    def __init__(self, 
                 in_channels=2,          # Real and imaginary channels
                 out_channels=2,         # Output channels (2 for complex values)
                 dim=256,                # Model dimension
                 depth=6,                # Number of transformer blocks
                 heads=8,                # Number of attention heads
                 mlp_dim=512,            # MLP hidden dimension
                 dropout=0.1,
                 mode='comm',           # 'comm' or 'radar'
                 num_rx=1,               # Number of receivers
                 num_rx_ant=16):         # Number of receiver antennas
        super().__init__()
        self.mode = mode
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant
        
        # MIMO preprocessing - combine receiver and antenna dimensions
        #2 is (real and imaginary)
        #[B, 2, R*A, H, W] as the input to mimo_preprocessing.
        self.mimo_preprocessing = nn.Sequential(
            # First reshape and process antenna dimension with real/imag channels
            nn.Conv3d(2, dim//2, kernel_size=(num_rx_ant, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(dim//2),
            nn.GELU(),
            # Process across receivers
            nn.Conv3d(dim//2, dim, kernel_size=(num_rx, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.GELU()
        )
        
        # Initial processing layers - shared between both modes
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Mode-specific initial processing
        if mode == 'radar':
            # Special processing for radar signals - larger kernel for better range-Doppler resolution
            self.mode_specific_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        else:  # comm mode
            # Special processing for communication signals - focused on symbol recovery
            self.mode_specific_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        
        # Positional encoding with learnable parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, 1))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                mode=mode
            ) for _ in range(depth)
        ])
        
        # Output processing
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=1)
        )
        
        # Mode-specific output activation
        if mode == 'comm':
            # For OFDM symbol detection - values between 0 and 1
            self.output_activation = nn.Sigmoid()
        else:
            # For radar target detection - values between -1 and 1
            self.output_activation = nn.Tanh()
    
    def forward(self, x):
        # x shape: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        # x is complex data
        B, R, A, H, W = x.shape
        
        # Convert complex data to real representation [B, R, A, H, W, 2]
        # where the last dimension contains real and imaginary parts
        if torch.is_complex(x):
            x_real = x.real
            x_imag = x.imag
            x = torch.stack([x_real, x_imag], dim=-1)  # [B, R, A, H, W, 2]
        
        # Process MIMO data
        if R > 1 or A > 1:
            # Reshape for MIMO processing
            # Combine real/imag into channels first
            x = x.permute(0, 5, 1, 2, 3, 4).reshape(B, 2, R*A, H, W)
            
            # Apply a convolutional layer to reduce R*A dimension
            x = self.mimo_preprocessing(x)  # [B, dim, H, W]
        else:
            # If single receiver/antenna, reshape to [B, 2, H, W]
            x = x.permute(0, 5, 3, 4).reshape(B, 2, H, W)
        
        # Initial embedding
        x = self.embedding(x)  # [B, dim, H, W]
        
        # Mode-specific processing
        x = self.mode_specific_conv(x)
        
        # Add learnable positional encoding
        x = x + self.pos_embedding
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output processing
        x = self.output_proj(x)
        x = self.output_activation(x)
        
        return x
    
    def forward_2d(self, x):
        # x shape: [batch_size, 2, height, width]
        B, C, H, W = x.shape #[4, 2, 12, 64]
        
        # Initial embedding
        x = self.embedding(x) #[4, 256, 12, 64]
        
        # Mode-specific processing
        x = self.mode_specific_conv(x) #[4, 256, 12, 64]
        
        # Add learnable positional encoding
        x = x + self.pos_embedding #[1, 256, 1, 1]
        #[4, 256, 12, 64]
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x) #[4, 256, 12, 64]
        
        # Output processing
        x = self.output_proj(x) #[4, 2, 12, 64]
        x = self.output_activation(x) #[4, 2, 12, 64]
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, mode='comm'):
        super().__init__()
        self.mode = mode
        
        # Layer normalization - properly handling 4D tensors
        self.norm1 = nn.GroupNorm(1, dim)  # GroupNorm with 1 group is equivalent to LayerNorm for 4D tensors
        self.norm2 = nn.GroupNorm(1, dim)
        if mode == 'radar':
            self.norm3 = nn.GroupNorm(1, dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(dim, heads, dropout)
        
        # Mode-specific processing
        if mode == 'radar':
            # Separate attention mechanisms for range and Doppler dimensions
            self.range_attention = RangeAttention(dim, heads // 2, dropout)
            self.doppler_attention = DopplerAttention(dim, heads // 2, dropout)
        
        # MLP block - replace sequential with individual layers for better control
        self.rearrange_in = Rearrange('b c h w -> b (h w) c')
        self.mlp_fc1 = nn.Linear(dim, mlp_dim)
        self.mlp_act = nn.GELU()
        self.mlp_drop1 = nn.Dropout(dropout)
        self.mlp_fc2 = nn.Linear(mlp_dim, dim)
        self.mlp_drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Regular self-attention with proper normalization
        norm_x = self.norm1(x)
        x = x + self.attention(norm_x)
        
        # Mode-specific processing
        if self.mode == 'radar':
            # Range and Doppler attention for radar mode
            norm_x = self.norm2(x)
            x_range = self.range_attention(norm_x)
            x_doppler = self.doppler_attention(norm_x)
            x = x + x_range + x_doppler
            
            # MLP with proper reshaping
            norm_x = self.norm3(x)
            # Reshape to [B, H*W, C]
            mlp_in = norm_x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            # Apply MLP layers
            mlp_out = self.mlp_fc1(mlp_in)
            mlp_out = self.mlp_act(mlp_out)
            mlp_out = self.mlp_drop1(mlp_out)
            mlp_out = self.mlp_fc2(mlp_out)
            mlp_out = self.mlp_drop2(mlp_out)
            # Reshape back to [B, C, H, W]
            mlp_out = mlp_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = x + mlp_out
        else:
            # Standard processing for communication mode
            norm_x = self.norm2(x)
            # Reshape to [B, H*W, C]
            mlp_in = norm_x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            # Apply MLP layers
            mlp_out = self.mlp_fc1(mlp_in)
            mlp_out = self.mlp_act(mlp_out)
            mlp_out = self.mlp_drop1(mlp_out)
            mlp_out = self.mlp_fc2(mlp_out)
            mlp_out = self.mlp_drop2(mlp_out)
            # Reshape back to [B, C, H, W]
            mlp_out = mlp_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = x + mlp_out
        
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
    """Special attention mechanism that focuses on the range dimension for radar processing"""
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
        
        # Reshape to focus on range dimension (H)
        q = q.permute(0, 1, 3, 2, 4).reshape(B * self.heads * W, C // self.heads, H)
        k = k.permute(0, 1, 3, 2, 4).reshape(B * self.heads * W, C // self.heads, H)
        v = v.permute(0, 1, 3, 2, 4).reshape(B * self.heads * W, C // self.heads, H)
        
        # Compute attention along range dimension
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

class DopplerAttention(nn.Module):
    """Special attention mechanism that focuses on the Doppler dimension for radar processing"""
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
        
        # Reshape to focus on Doppler dimension (W)
        q = q.permute(0, 1, 2, 4, 3).reshape(B * self.heads * H, C // self.heads, W)
        k = k.permute(0, 1, 2, 4, 3).reshape(B * self.heads * H, C // self.heads, W)
        v = v.permute(0, 1, 2, 4, 3).reshape(B * self.heads * H, C // self.heads, W)
        
        # Compute attention along Doppler dimension
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

def test1():
    # Example usage:
    # Test both OFDM and Radar modes
    batch_size = 4
    height = 12
    width = 64
    
    # Test OFDM communication mode
    comm_model = DualPurposeTransformer(mode='comm')
    comm_input = torch.randn(batch_size, 2, height, width) #[4, 2, 12, 64]
    comm_output = comm_model(comm_input)
    print(f"OFDM output shape: {comm_output.shape}") #[4, 2, 12, 64]
    
    # Test Radar mode
    #Dual-purpose architecture that can handle both OFDM and radar signals
    #Standard self-attention for OFDM
    #Range and Doppler attention for radar processing
    #Separate output activations for each mode: Sigmoid for OFDM symbol detection; Tanh for radar target detection
    #Learnable positional embeddings
    #For OFDM communication: Initialize with mode='comm'
    #For Radar processing: Initialize with mode='radar'
    radar_model = DualPurposeTransformer(mode='radar')
    radar_input = torch.randn(batch_size, 2, height, width) #[4, 2, 12, 64]
    radar_output = radar_model(radar_input) #[4, 2, 12, 64]
    print(f"Radar output shape: {radar_output.shape}")
    
if __name__ == "__main__":
    test1()