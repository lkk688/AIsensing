
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualPurposeTransformer(nn.Module):
    def __init__(self, 
                 in_channels=2,          # Real and imaginary channels
                 out_channels=2,         # Output channels (2 for complex values)
                 dim=256,                # Model dimension
                 depth=6,                # Number of transformer blocks
                 heads=8,                # Number of attention heads
                 mlp_dim=512,           # MLP hidden dimension
                 dropout=0.1,
                 mode='comm'):          # 'comm' or 'radar'
        super().__init__()
        self.mode = mode
        
        # Initial processing layers
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Mode-specific processing
        if mode == 'radar':
            self.range_doppler_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=5, padding=2),
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
        self.output_layers = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, padding=1)
        )
        
        if mode == 'comm':
            self.output_activation = nn.Sigmoid()  # For OFDM symbol detection
        else:
            self.output_activation = nn.Tanh()     # For radar target detection
    
    def forward(self, x):
        # x shape: [batch_size, 2, height, width]
        
        # Initial embedding
        x = self.embedding(x)
        
        # Add learnable positional encoding
        x = x + self.pos_embedding
        
        # Mode-specific processing
        if self.mode == 'radar':
            x = self.range_doppler_conv(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output processing
        x = self.output_layers(x)
        x = self.output_activation(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, mode='comm'):
        super().__init__()
        self.mode = mode
        
        # Multi-head attention
        self.attention = MultiHeadAttention(dim, heads, dropout)
        
        # Mode-specific processing
        if mode == 'radar':
            self.range_attention = MultiHeadAttention(dim, heads // 2, dropout)
            self.doppler_attention = MultiHeadAttention(dim, heads // 2, dropout)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if mode == 'radar':
            self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Regular self-attention
        x = x + self.attention(self.norm1(x))
        
        # Mode-specific processing
        if self.mode == 'radar':
            # Range and Doppler attention for radar mode
            x_range = self.range_attention(self.norm2(x))
            x_doppler = self.doppler_attention(self.norm2(x))
            x = x + x_range + x_doppler
            x = x + self.mlp(self.norm3(x))
        else:
            # Standard processing for communication mode
            x = x + self.mlp(self.norm2(x))
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape input for attention
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, H * W, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention
        x = (attn @ v).reshape(B, H * W, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        # Reshape back to original format
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return x

class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ofdm_output_dim, radar_output_dim):
        """
        Transformer model for both OFDM and Radar receivers.
        
        Args:
            input_dim (int): Input feature dimension (e.g., 2 for complex signals represented as real and imaginary parts).
            model_dim (int): Transformer model dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            ofdm_output_dim (int): Output dimension for OFDM task (e.g., number of subcarriers or symbols).
            radar_output_dim (int): Output dimension for Radar task (e.g., number of targets or detection parameters).
        """
        super(MultiTaskTransformer, self).__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Task-specific output heads
        self.ofdm_head = nn.Linear(model_dim, ofdm_output_dim)
        self.radar_head = nn.Linear(model_dim, radar_output_dim)
    
    def forward(self, x, task_type):
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            task_type (str): Task type, either "ofdm" or "radar".
        
        Returns:
            torch.Tensor: Output tensor for the specified task.
        """
        # Input embedding
        x = self.embedding(x)  # (batch_size, seq_len, model_dim)
        
        # Transformer encoder
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, model_dim) for Transformer
        x = self.transformer(x)  # (seq_len, batch_size, model_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, model_dim)
        
        # Task-specific output
        if task_type == "ofdm":
            x = self.ofdm_head(x)  # (batch_size, seq_len, ofdm_output_dim)
        elif task_type == "radar":
            x = self.radar_head(x)  # (batch_size, seq_len, radar_output_dim)
        else:
            raise ValueError("Invalid task type. Use 'ofdm' or 'radar'.")
        
        return x

# Example usage
def test1():
    # Parameters
    batch_size = 32
    seq_len = 64  # Sequence length (e.g., number of OFDM symbols or radar samples)
    input_dim = 2  # Input dimension (real and imaginary parts)
    model_dim = 128  # Transformer model dimension
    num_heads = 8  # Number of attention heads
    num_layers = 4  # Number of Transformer layers
    ofdm_output_dim = 64  # Output dimension for OFDM task (e.g., number of subcarriers)
    radar_output_dim = 3  # Output dimension for Radar task (e.g., range, velocity, angle)

    # Initialize model
    model = MultiTaskTransformer(input_dim, model_dim, num_heads, num_layers, ofdm_output_dim, radar_output_dim)

    # Example input (batch_size, seq_len, input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)

    # OFDM task
    ofdm_output = model(x, task_type="ofdm")
    print("OFDM output shape:", ofdm_output.shape)  # Expected: (batch_size, seq_len, ofdm_output_dim)

    # Radar task
    radar_output = model(x, task_type="radar")
    print("Radar output shape:", radar_output.shape)  # Expected: (batch_size, seq_len, radar_output_dim)

def test2():
    # Example usage:
    # Test both OFDM and Radar modes
    batch_size = 4
    height = 12
    width = 64
    
    # Test OFDM communication mode
    comm_model = DualPurposeTransformer(mode='comm')
    comm_input = torch.randn(batch_size, 2, height, width)
    comm_output = comm_model(comm_input)
    print(f"OFDM output shape: {comm_output.shape}")
    
    # Test Radar mode
    #Dual-purpose architecture that can handle both OFDM and radar signals
    #Standard self-attention for OFDM
    #Range and Doppler attention for radar processing
    #Separate output activations for each mode: Sigmoid for OFDM symbol detection; Tanh for radar target detection
    #Learnable positional embeddings
    #For OFDM communication: Initialize with mode='comm'
    #For Radar processing: Initialize with mode='radar'
    radar_model = DualPurposeTransformer(mode='radar')
    radar_input = torch.randn(batch_size, 2, height, width)
    radar_output = radar_model(radar_input)
    print(f"Radar output shape: {radar_output.shape}")
    
if __name__ == "__main__":
    test1()