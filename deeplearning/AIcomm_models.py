import unittest
import torch
import torch.nn as nn
import sys
import os
from unittest.mock import patch, MagicMock


# Define the OFDM neural network model with transformer architecture
class OFDMNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, dim=128, depth=4, heads=8, mlp_dim=256, dropout=0.1):
        super(OFDMNet, self).__init__()
        
        # Initial convolutional embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(dim)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input shape: [batch_size, 2, 12, 64] (complex I/Q channels)
        
        # Initial embedding
        x = self.embedding(x)  # [batch_size, dim, 12, 64]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Output projection
        x = self.output_projection(x)  # [batch_size, out_channels, 12, 64]
        
        # Reshape to match target dimensions [batch_size, 12, 64, 2]
        x = x.permute(0, 2, 3, 1)
        
        return x


# 2D Positional Encoding for the transformer
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Create positional encodings
        y_pos = torch.arange(height, device=x.device).unsqueeze(1).expand(height, width).float()
        x_pos = torch.arange(width, device=x.device).unsqueeze(0).expand(height, width).float()
        
        # Scale positions
        y_pos = y_pos / height * 2 - 1
        x_pos = x_pos / width * 2 - 1
        
        # Create positional channels
        pos_encoding = torch.stack([y_pos, x_pos], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Expand positional encoding to match channel dimension
        pos_encoding_expanded = torch.zeros(batch_size, channels, height, width, device=x.device)
        pos_encoding_expanded[:, 0:2, :, :] = pos_encoding
        
        # Add positional encoding to input
        return x + 0.1 * pos_encoding_expanded

# Multi-head Self-Attention module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, 3, self.heads, channels // self.heads, height, width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Reshape for attention computation
        q = q.permute(0, 1, 3, 4, 2).reshape(batch_size, self.heads, height * width, channels // self.heads)
        k = k.permute(0, 1, 3, 4, 2).reshape(batch_size, self.heads, height * width, channels // self.heads)
        v = v.permute(0, 1, 3, 4, 2).reshape(batch_size, self.heads, height * width, channels // self.heads)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).reshape(batch_size, self.heads, height, width, channels // self.heads)
        x = x.permute(0, 1, 4, 2, 3).reshape(batch_size, channels, height, width)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

# MLP block for transformer
class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# Transformer block combining attention and MLP
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([dim, None, None])
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm([dim, None, None])
        self.mlp = MLP(dim, mlp_dim, dropout)
    
    def forward(self, x):
        # Apply layer normalization and attention with residual connection
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm)
        
        # Apply layer normalization and MLP with residual connection
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.mlp(x_norm)
        
        return x

if __name__ == '__main__':
    #This line runs all the unit tests defined in the file.
    #unittest.main()
    in_channels = 2
    out_channels = 2
    dim = 16
    depth = 2
    heads = 4
    mlp_dim = 32
    model = OFDMNet(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.0) 
    
    model.eval()
        
        # Create sample input
    batch_size = 2
    height = 12
    width = 64
    input_tensor = torch.rand(batch_size, in_channels, height, width)
    
    output = model(input_tensor)
        
    # Check output shape
    print(output.shape)