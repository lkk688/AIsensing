import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import random
from scipy.signal import chirp

IMG_FORMAT=".pdf" #".png"

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RadarNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(RadarNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 32) #(real,image)2->32
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        
        # Decoder
        self.dec3 = ConvBlock(128, 64)
        self.dec2 = ConvBlock(64, 32)
        self.dec1 = ConvBlock(32, 16)
        
        # Output layer
        self.output = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)#x: [32, 2, 12, 64]=>[32, 32, 12, 64]
        p1 = self.pool(e1) #[32, 32, 6, 32]
        
        e2 = self.enc2(p1) #[32, 64, 6, 32]
        p2 = self.pool(e2) #[32, 64, 3, 16]
        
        e3 = self.enc3(p2) #[32, 128, 3, 16]
        
        # Decoder with skip connections
        d3 = self.dec3(e3) #[32, 64, 3, 16]
        u2 = self.upsample(d3) #[32, 64, 6, 32]
        u2 = u2 + e2  # Skip connection
        
        d2 = self.dec2(u2) #[32, 32, 6, 32]
        u1 = self.upsample(d2) #[32, 32, 12, 64]
        u1 = u1 + e1  # Skip connection
        
        d1 = self.dec1(u1) #[32, 16, 12, 64]
        
        # Output
        out = self.output(d1) #[32, 1, 12, 64]
        out = self.sigmoid(out)
        
        # Reshape to match the expected output format [batch, H, W, 1]
        out = out.permute(0, 2, 3, 1)
        
        return out #[32, 12, 64, 1]

class RadarDataset(Dataset):
    def __init__(self, 
                 datapath=None,
                 num_samples=10000, 
                 num_range_bins=64, 
                 num_doppler_bins=12, 
                 snr_min=5, 
                 snr_max=30, 
                 max_targets=3,
                 training=False, 
                 drawfig=False, 
                 save_data=True):
        """
        Dataset for radar range-Doppler data
        
        Args:
            datapath: Path to load existing data, if None, generate new data
            num_samples: Number of samples to generate
            num_range_bins: Number of range bins (width)
            num_doppler_bins: Number of Doppler bins (height)
            snr_min: Minimum SNR for targets
            snr_max: Maximum SNR for targets
            max_targets: Maximum number of targets in a scene
            training: Whether this dataset is for training
            drawfig: Whether to draw figures for visualization
            save_data: Whether to save generated data
        """
        self.num_samples = num_samples
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.max_targets = max_targets
        self.training = training
        self.drawfig = drawfig
        
        if datapath is not None and os.path.exists(datapath):
            print(f"Loading radar data from {datapath}")
            self.load_data(datapath)
        else:
            print("Generating new radar data")
            self.generate_radar_data(save_data)
            
    def generate_radar_data(self, save_data=True):
        """Generate synthetic radar data with targets at random positions"""
        # Initialize arrays for data and labels, (10000, 2, 12, 64)
        self.range_doppler_maps = np.zeros((self.num_samples, 2, self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        self.target_masks = np.zeros((self.num_samples, self.num_doppler_bins, self.num_range_bins, 1), dtype=np.float32)
        #(10000, 12, 64, 1)
        # Generate data for each sample
        for i in range(self.num_samples):
            # Determine number of targets for this sample (1 to max_targets)
            num_targets = random.randint(1, self.max_targets)
            
            # Generate complex range-Doppler map with noise
            noise_power = 1.0
            noise_real = np.random.normal(0, np.sqrt(noise_power/2), (self.num_doppler_bins, self.num_range_bins))
            noise_imag = np.random.normal(0, np.sqrt(noise_power/2), (self.num_doppler_bins, self.num_range_bins))
            rd_map = noise_real + 1j * noise_imag #(12, 64)
            
            # Add targets
            for _ in range(num_targets):
                # Random target position
                range_idx = random.randint(0, self.num_range_bins-1) #38
                doppler_idx = random.randint(0, self.num_doppler_bins-1) #9
                
                # Random SNR for this target
                snr = random.uniform(self.snr_min, self.snr_max)
                target_power = noise_power * 10**(snr/10)
                
                # Random complex amplitude for target
                amplitude = np.sqrt(target_power) * np.exp(1j * random.uniform(0, 2*np.pi))
                
                # Add target to range-Doppler map with some spread (to simulate real targets)
                spread = 1.0  # Spread factor
                for dr in range(-1, 2):
                    for dd in range(-1, 2):
                        r_idx = range_idx + dr
                        d_idx = doppler_idx + dd
                        if 0 <= r_idx < self.num_range_bins and 0 <= d_idx < self.num_doppler_bins:
                            # Decrease amplitude with distance from center
                            dist = np.sqrt(dr**2 + dd**2)
                            if dist == 0:
                                # Mark the center point in the target mask
                                self.target_masks[i, d_idx, r_idx, 0] = 1.0
                            
                            # Add target with reduced amplitude based on distance
                            rd_map[d_idx, r_idx] += amplitude * np.exp(-dist/spread)
            
            # Split into real and imaginary components
            self.range_doppler_maps[i, 0, :, :] = np.real(rd_map)
            self.range_doppler_maps[i, 1, :, :] = np.imag(rd_map)
            
            # Visualize a few samples
            if self.drawfig and i < 3:
                self._visualize_sample(i)
        
        if save_data:
            self._save_data()
    
    def _visualize_sample(self, idx):
        """Visualize a sample range-Doppler map and its target mask"""
        rd_map = self.range_doppler_maps[idx] #one example (2, 12, 64)
        target_mask = self.target_masks[idx, :, :, 0] #(12, 64)
        
        # Calculate magnitude from real and imaginary parts
        magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
        
        plt.figure(figsize=(12, 5))
        
        # Plot range-Doppler magnitude
        plt.subplot(1, 2, 1)
        plt.imshow(20*np.log10(magnitude + 1e-10), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(f'Range-Doppler Map (Sample {idx})')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Plot target mask
        plt.subplot(1, 2, 2)
        plt.imshow(target_mask, aspect='auto', cmap='gray')
        plt.colorbar(label='Target Presence')
        plt.title(f'Target Mask (Sample {idx})')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        plt.tight_layout()
        plt.savefig(f'data/radar_sample_{idx}{IMG_FORMAT}')
        plt.close()
    
    def _save_data(self):
        """Save generated data to file"""
        os.makedirs('data/radar', exist_ok=True)
        
        data_dict = {
            'range_doppler_maps': self.range_doppler_maps, #(10000, 2, 12, 64)
            'target_masks': self.target_masks, #(10000, 12, 64, 1)
            'num_range_bins': self.num_range_bins, #64
            'num_doppler_bins': self.num_doppler_bins, #12
            'snr_range': [self.snr_min, self.snr_max],
            'max_targets': self.max_targets
        }
        
        np.save('data/radar/radar_simulation_data.npy', data_dict)
        print("Radar simulation data saved to data/radar/radar_simulation_data.npy")
    
    def load_data(self, datapath):
        """Load data from file"""
        data_dict = np.load(datapath, allow_pickle=True).item()
        
        self.range_doppler_maps = data_dict['range_doppler_maps']
        self.target_masks = data_dict['target_masks']
        self.num_range_bins = data_dict['num_range_bins']
        self.num_doppler_bins = data_dict['num_doppler_bins']
        self.snr_min, self.snr_max = data_dict['snr_range']
        self.max_targets = data_dict['max_targets']
        self.num_samples = len(self.range_doppler_maps)
        
        print(f"Loaded {self.num_samples} radar samples")
        
        # Visualize a few samples if requested
        if self.drawfig:
            for i in range(min(3, self.num_samples)):
                self._visualize_sample(i)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Add some noise variation for training robustness
        if self.training:
            # Add random noise to make the model more robust
            noise_level = random.uniform(0.05, 0.2)
            noise = np.random.normal(0, noise_level, self.range_doppler_maps[idx].shape)
            feature = self.range_doppler_maps[idx] + noise
        else:
            feature = self.range_doppler_maps[idx]
            
        batch = {
            'feature_2d': feature.astype(np.float32),  # [2, num_doppler_bins, num_range_bins]
            'labels': self.target_masks[idx]           # [num_doppler_bins, num_range_bins, 1]
        }
        
        return batch

