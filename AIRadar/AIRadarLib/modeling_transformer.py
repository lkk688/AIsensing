import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops.layers.torch import Rearrange
from einops import rearrange
from AIRadarLib.ofdm_decoder import OFDMDecoder

# === RadarTransformerNet: transformer-based model for radar detection and OFDM communication ===
class RadarTransformerNet(nn.Module):
    """
    Transformer-based model for radar detection and OFDM communication that processes time-domain radar data directly.
    This model combines CNN layers for initial feature extraction with transformer blocks for
    capturing long-range dependencies in both range and Doppler dimensions.
    
    The model can process raw IQ time-domain signals, perform demodulation (including OFDM
    demodulation if applicable), and output both a range-Doppler map and target detection results.
    It combines the functionality of RadarTimeNet and RadarEndToEnd in a single end-to-end model.
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
    """
    Test the RadarTransformerNet model with synthetic data for both radar detection and OFDM communication.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from AIRadarLib.waveform_utils import generate_linear_chirp, generate_ofdm_signal
    
    # Model parameters
    num_rx = 2                  # Number of receive antennas
    num_chirps = 64             # Number of chirps in the input signal
    samples_per_chirp = 64      # Number of samples per chirp
    out_doppler_bins = 64       # Number of Doppler bins in the output
    out_range_bins = 64         # Number of range bins in the output
    batch_size = 2              # Batch size for testing
    
    print("Initializing RadarTransformerNet model...")
    # Initialize model with OFDM support and detection capabilities
    model = RadarTransformerNet(
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp,
        out_doppler_bins=out_doppler_bins,
        out_range_bins=out_range_bins,
        dim=128,                # Model dimension
        depth=4,                # Number of transformer blocks
        heads=8,                # Number of attention heads
        mlp_dim=256,            # MLP hidden dimension
        dropout=0.1,
        use_learnable_fft=True, # Use learnable FFT for range-Doppler processing
        use_cnn_backbone=True,  # Use CNN backbone for feature extraction
        support_ofdm=True,      # Enable OFDM support
        detect_threshold=0.5,   # Detection threshold
        max_targets=10          # Maximum number of targets to detect
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate test FMCW data
    print("\nTesting RadarTransformerNet with FMCW data...")
    fmcw_data = generate_test_fmcw_data(
        batch_size=batch_size,
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp
    )  # Shape: [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
    
    # Forward pass with FMCW data
    with torch.no_grad():
        fmcw_output = model(fmcw_data, is_ofdm=False)  # Shape: dict with detection_results and rd_map
    
    # Print output shapes
    print(f"FMCW Output - RD Map shape: {fmcw_output['rd_map'].shape}")  # [batch_size, 2, out_doppler_bins, out_range_bins]
    print(f"FMCW Output - Detection Map shape: {fmcw_output['detection_results']['detection_map'].shape}")  # [batch_size, 1, out_doppler_bins, out_range_bins]
    print(f"FMCW Output - Number of detected targets: {len(fmcw_output['detection_results']['target_list'][0])}")
    
    # Generate test OFDM data
    print("\nTesting RadarTransformerNet with OFDM data...")
    ofdm_data = generate_test_ofdm_data(
        batch_size=batch_size,
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp
    )  # Shape: [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
    
    # Forward pass with OFDM data
    with torch.no_grad():
        ofdm_output = model(ofdm_data, is_ofdm=True)  # Shape: dict with detection_results, rd_map, ofdm_map, decoded_bits
    
    # Print output shapes
    print(f"OFDM Output - RD Map shape: {ofdm_output['rd_map'].shape}")  # [batch_size, 2, out_doppler_bins, out_range_bins]
    print(f"OFDM Output - Detection Map shape: {ofdm_output['detection_results']['detection_map'].shape}")  # [batch_size, 1, out_doppler_bins, out_range_bins]
    print(f"OFDM Output - OFDM Map shape: {ofdm_output['ofdm_map'].shape}")  # [batch_size, 2, num_chirps, samples_per_chirp]
    print(f"OFDM Output - Decoded Bits shape: {ofdm_output['decoded_bits'].shape}")  # [batch_size, num_bits]
    
    # Visualize results
    plt.figure(figsize=(12, 10))
    
    # Plot FMCW Range-Doppler Map
    plt.subplot(2, 2, 1)
    plt.title("FMCW Range-Doppler Map (Magnitude)")
    rd_map_magnitude = torch.sqrt(fmcw_output['rd_map'][0, 0]**2 + fmcw_output['rd_map'][0, 1]**2)  # Magnitude of complex RD map
    plt.imshow(rd_map_magnitude.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Range Bins')
    plt.ylabel('Doppler Bins')
    
    # Plot FMCW Detection Map
    plt.subplot(2, 2, 2)
    plt.title("FMCW Detection Map")
    plt.imshow(fmcw_output['detection_results']['detection_map'][0, 0].numpy(), aspect='auto', cmap='plasma')
    plt.colorbar(label='Detection Probability')
    plt.xlabel('Range Bins')
    plt.ylabel('Doppler Bins')
    
    # Plot OFDM Range-Doppler Map
    plt.subplot(2, 2, 3)
    plt.title("OFDM Range-Doppler Map (Magnitude)")
    rd_map_magnitude = torch.sqrt(ofdm_output['rd_map'][0, 0]**2 + ofdm_output['rd_map'][0, 1]**2)  # Magnitude of complex RD map
    plt.imshow(rd_map_magnitude.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Range Bins')
    plt.ylabel('Doppler Bins')
    
    # Plot OFDM Map
    plt.subplot(2, 2, 4)
    plt.title("OFDM Demodulation Output (Magnitude)")
    ofdm_magnitude = torch.sqrt(ofdm_output['ofdm_map'][0, 0]**2 + ofdm_output['ofdm_map'][0, 1]**2)  # Magnitude of complex OFDM map
    plt.imshow(ofdm_magnitude.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Symbol Index')
    
    # Add a new figure for the decoded bits
    plt.figure(figsize=(10, 5))
    plt.title("OFDM Decoded Bits")
    # Reshape bits for better visualization (show as bytes)
    bits_reshaped = ofdm_output['decoded_bits'][0].reshape(-1, 8)[:50]  # Show first 50 bytes
    plt.imshow(bits_reshaped.numpy(), aspect='auto', cmap='binary')
    plt.colorbar(label='Bit Value')
    plt.xlabel('Bit Position')
    plt.ylabel('Byte Index')
    
    plt.tight_layout()
    plt.savefig('transformer_net_test_results.png')
    plt.show()
    
    print("\nTest completed successfully!")
    print("Results saved to 'transformer_net_test_results.png'")


def generate_test_fmcw_data(batch_size=2, num_rx=2, num_chirps=64, samples_per_chirp=64):
    """
    Generate synthetic FMCW radar data for testing.
    
    Args:
        batch_size: Number of samples in the batch
        num_rx: Number of receive antennas
        num_chirps: Number of chirps
        samples_per_chirp: Number of samples per chirp
        
    Returns:
        Tensor with shape [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
    """
    import torch
    import numpy as np
    
    # Initialize empty tensor
    data = torch.zeros(batch_size, num_rx, num_chirps, samples_per_chirp, 2)
    
    for b in range(batch_size):
        # Generate random targets
        num_targets = 3
        max_range = 50  # meters
        max_velocity = 10  # m/s
        
        ranges = np.random.uniform(5, max_range, num_targets)
        velocities = np.random.uniform(-max_velocity, max_velocity, num_targets)
        amplitudes = np.random.uniform(0.5, 1.0, num_targets)
        
        # Radar parameters
        fc = 77e9  # Center frequency: 77 GHz
        bw = 1e9  # Bandwidth: 1 GHz
        chirp_duration = 50e-6  # 50 microseconds
        prf = 1 / (chirp_duration * 1.1)  # Pulse repetition frequency
        c = 3e8  # Speed of light
        
        # Calculate parameters
        slope = bw / chirp_duration
        wavelength = c / fc
        
        # Generate time samples
        t = np.linspace(0, chirp_duration, samples_per_chirp)
        
        # For each chirp and receive antenna
        for chirp in range(num_chirps):
            chirp_time = chirp / prf
            
            # Initialize chirp signal
            chirp_signal = np.zeros(samples_per_chirp, dtype=np.complex128)
            
            # Add target reflections
            for i in range(num_targets):
                # Calculate range delay
                tau = 2 * ranges[i] / c
                
                # Calculate Doppler shift
                doppler_freq = 2 * velocities[i] / wavelength
                
                # Phase due to Doppler
                doppler_phase = 2 * np.pi * doppler_freq * chirp_time
                
                # Delayed signal with Doppler shift
                delayed_t = t - tau
                valid_indices = delayed_t >= 0
                
                # Beat signal (difference between transmitted and received)
                beat_phase = 2 * np.pi * (slope * tau * delayed_t[valid_indices] - 0.5 * slope * tau**2)
                beat_signal = amplitudes[i] * np.exp(1j * (beat_phase + doppler_phase))
                
                # Add to chirp signal
                chirp_signal[valid_indices] += beat_signal
            
            # Add some noise
            noise = np.random.normal(0, 0.1, samples_per_chirp) + 1j * np.random.normal(0, 0.1, samples_per_chirp)
            chirp_signal += noise
            
            # For each receive antenna (add slight phase differences)
            for rx in range(num_rx):
                rx_phase = np.exp(1j * rx * np.pi / 4)  # Simple phase shift between antennas
                rx_signal = chirp_signal * rx_phase
                
                # Convert to real/imag format
                data[b, rx, chirp, :, 0] = torch.tensor(rx_signal.real, dtype=torch.float32)
                data[b, rx, chirp, :, 1] = torch.tensor(rx_signal.imag, dtype=torch.float32)
    
    return data


def generate_test_ofdm_data(batch_size=2, num_rx=2, num_chirps=64, samples_per_chirp=64):
    """
    Generate synthetic OFDM radar data for testing.
    
    Args:
        batch_size: Number of samples in the batch
        num_rx: Number of receive antennas
        num_chirps: Number of chirps (OFDM symbols)
        samples_per_chirp: Number of samples per chirp (FFT size)
        
    Returns:
        Tensor with shape [batch_size, num_rx, num_chirps, samples_per_chirp, 2]
    """
    import torch
    import numpy as np
    from AIRadarLib.waveform_utils import generate_ofdm_signal
    
    # Initialize empty tensor
    data = torch.zeros(batch_size, num_rx, num_chirps, samples_per_chirp, 2)
    
    for b in range(batch_size):
        # Generate OFDM signal
        ofdm_signal = generate_ofdm_signal(
            num_subcarriers=samples_per_chirp//2,  # Use half of the subcarriers
            num_symbols=num_chirps,
            subcarrier_spacing=1e6,  # 1 MHz spacing (arbitrary for test)
            fs=samples_per_chirp * 1e6,  # Sampling frequency
            cp_length_ratio=0  # No cyclic prefix for simplicity
        )
        
        # Reshape to match expected dimensions
        ofdm_signal = ofdm_signal.reshape(num_chirps, samples_per_chirp)
        
        # Add targets (similar to FMCW but with OFDM signal)
        num_targets = 3
        max_range = 50  # meters
        max_velocity = 10  # m/s
        
        # Generate random targets
        ranges = np.random.uniform(5, max_range, num_targets)
        velocities = np.random.uniform(-max_velocity, max_velocity, num_targets)
        amplitudes = np.random.uniform(0.5, 1.0, num_targets)
        
        # Radar parameters
        fc = 5.8e9  # Center frequency: 5.8 GHz (typical for OFDM radar)
        c = 3e8  # Speed of light
        wavelength = c / fc
        symbol_duration = 10e-6  # 10 microseconds
        
        # For each OFDM symbol and receive antenna
        for symbol in range(num_chirps):
            symbol_time = symbol * symbol_duration
            
            # Initialize symbol signal
            symbol_signal = ofdm_signal[symbol].copy()
            
            # Add target reflections
            for i in range(num_targets):
                # Calculate range delay in samples
                tau = 2 * ranges[i] / c
                delay_samples = int(tau / (symbol_duration / samples_per_chirp))
                delay_samples = min(delay_samples, samples_per_chirp - 1)
                
                # Calculate Doppler shift
                doppler_freq = 2 * velocities[i] / wavelength
                doppler_phase = 2 * np.pi * doppler_freq * symbol_time
                
                # Apply delay and Doppler
                delayed_signal = np.roll(symbol_signal, delay_samples) * amplitudes[i] * np.exp(1j * doppler_phase)
                
                # Add to symbol signal
                symbol_signal += delayed_signal
            
            # Add some noise
            noise = np.random.normal(0, 0.1, samples_per_chirp) + 1j * np.random.normal(0, 0.1, samples_per_chirp)
            symbol_signal += noise
            
            # For each receive antenna (add slight phase differences)
            for rx in range(num_rx):
                rx_phase = np.exp(1j * rx * np.pi / 4)  # Simple phase shift between antennas
                rx_signal = symbol_signal * rx_phase
                
                # Convert to real/imag format
                data[b, rx, symbol, :, 0] = torch.tensor(rx_signal.real, dtype=torch.float32)
                data[b, rx, symbol, :, 1] = torch.tensor(rx_signal.imag, dtype=torch.float32)
    
    return data


if __name__ == "__main__":
    test_model()