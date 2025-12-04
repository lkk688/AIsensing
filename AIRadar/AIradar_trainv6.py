#!/usr/bin/env python3
"""
RadarTimeNet Training Script with Simulation Data (v6)

This script trains the RadarTimeNet model using simulated FMCW radar data from AIradar_datasetv5.
It features:
1. Advanced Loss Functions: BCE + Focal + Dice Loss
2. Improved Model Architecture: U-Net based Range-Doppler processing
3. Enhanced Evaluation: Comparison with CFAR, detailed metrics
4. Visualization: Side-by-side comparison of DL vs CFAR

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import time
import argparse
from scipy import signal
import time
import math
from typing import Tuple, List, Dict, Optional
import os
import json
from tqdm import tqdm

# Import the dataset
# Ensure AIRadar directory is in python path or use relative import if running from root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AIRadar.AIradar_datasetv6 import AIRadarDataset, _plot_2d_rdm, _plot_3d_rdm, VIEW_RANGE_LIMITS, VIEW_VELOCITY_LIMITS

# Import visualization functions
try:
    from AIRadarLib.visualization import (
        plot_signal_time_and_spectrum,
        plot_instantaneous_frequency,
        plot_range_doppler_map_with_ground_truth,
        plot_3d_range_doppler_map_with_ground_truth
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: AIRadarLib.visualization not available. Some visualizations will be skipped.")
    VISUALIZATION_AVAILABLE = False

# Try to import torch.fft for modern PyTorch versions
try:
    import torch.fft
    HAS_TORCH_FFT = True
except ImportError:
    HAS_TORCH_FFT = False

# ==========================================
# Global Constants
# ==========================================
VIEW_RANGE_LIMITS = (0, 100)
VIEW_VELOCITY_LIMITS = (-48, 48)

# ==========================================
# Advanced Loss Functions
# ==========================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, H, W] (logits)
        # targets: [B, H, W] (0 or 1)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks, good for shape alignment and class imbalance.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, H, W] (logits)
        # targets: [B, H, W] (0 or 1)
        
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combines BCE, Focal, and Dice losses.
    """
    def __init__(self, w_bce=0.5, w_focal=1.0, w_dice=1.0):
        super(CombinedLoss, self).__init__()
        # Increase pos_weight to handle extreme sparsity (approx 1:5000)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2000.0]).to('cuda' if torch.cuda.is_available() else 'cpu'))
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_dice = 10.0 # Increase Dice weight significantly

    def forward(self, inputs, targets):
        loss_bce = self.bce(inputs, targets)
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        
        return self.w_bce * loss_bce + self.w_focal * loss_focal + self.w_dice * loss_dice

# ==========================================
# Advanced Model Architecture (U-Net based)
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetRadar2D(nn.Module):
    """
    U-Net adapted for Range-Doppler Map segmentation/detection.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetRadar2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        # Initialize bias of the last layer to reflect class imbalance
        # Probability of target is very low (sparse)
        # bias = -log((1-p)/p) where p is prior probability
        # e.g. p=0.001 => bias approx -6.9
        # self.outc.conv.bias.data.fill_(-5.0) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Remove the custom bias initialization which forces very low initial probabilities
        # This might be too aggressive if the training data distribution is not extremely sparse
        # or if the loss function (Focal Loss) already handles class imbalance.
        # Let the model learn the bias.
        
        return logits

class RadarTimeNetV5(nn.Module):
    """
    Deep learning model for processing time-domain IQ signals to extract range-Doppler maps
    and perform object detection.
    Supports both FMCW and OTFS signals by adapting to input dimensions.
    """
    def __init__(self, num_rx=1, num_chirps=None, samples_per_chirp=None, 
                 fft_size=None, out_range_bins=None, out_doppler_bins=None,
                 use_learnable_fft=False, target_unet_size=(128, 128)):
        super().__init__()
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.fft_size = fft_size
        self.out_range_bins = out_range_bins
        self.out_doppler_bins = out_doppler_bins
        self.use_learnable_fft = use_learnable_fft
        self.target_unet_size = target_unet_size # (Doppler, Range)
        
        # === Time-domain preprocessing ===
        # Input shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        self.time_conv = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1, 1, 3), padding=(0, 0, 1)), # Increased filters for robustness
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 1), padding=(0, 1, 0)), # Convolve across chirps
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 2, kernel_size=(1, 1, 1)), # Reduce back to 2 channels
            nn.BatchNorm3d(2),
        )
        
        # === Demodulation module ===
        self.demod_weights = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))
        
        # === U-Net Post-processing ===
        # Input to UNet will be 1 channel: Log-Magnitude (Normalized)
        self.unet = UNetRadar2D(n_channels=1, n_classes=1, bilinear=True)
        
        self._init_weights()

    def _init_weights(self):
        self.demod_weights.data = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
        
    def demodulate(self, rx_signal):
        rx_signal_flat = rx_signal.reshape(-1, 2)
        demod_signal_flat = torch.matmul(rx_signal_flat, self.demod_weights)
        return demod_signal_flat.reshape(*rx_signal.shape)
    
    def apply_range_fft(self, x):
        # Memory optimization: chunked processing for Range FFT
        # x shape: [B, Rx, Chirps, Samples, 2]
        B, Rx, Chirps, Samples, _ = x.shape
        
        # Convert to complex for FFT
        # Instead of creating a huge complex tensor, we can process in chunks if B*Rx is large
        
        # If we can fit in memory, just do it:
        # complex_input = torch.view_as_complex(x)
        
        # But given OOM, let's process batch-wise or rx-wise if needed.
        # Since the error is 16GB, and we have B=2, maybe Rx=128 is too much.
        
        output_list = []
        
        # Process each batch item separately to save memory
        for b in range(B):
            # Extract one batch item: [Rx, Chirps, Samples, 2]
            x_b = x[b] 
            complex_input_b = torch.view_as_complex(x_b) # [Rx, Chirps, Samples]
            
            # Perform Range FFT on last dimension (Samples)
            # Output shape: [Rx, Chirps, Samples] (complex)
            # We want to keep it in frequency domain.
            # Typically Range FFT is along 'Samples' dimension.
            
            # Note: User code had dim=2 (Samples).
            # complex_input_b is [Rx, Chirps, Samples]
            # FFT on dim=2
            fft_out_b = torch.fft.fft(complex_input_b, dim=2)
            
            # Stack back to [Rx, Chirps, Range, 2]
            # View as real: [Rx, Chirps, Range, 2]
            output_list.append(torch.view_as_real(fft_out_b))
            
        # Stack back to [B, Rx, Chirps, Range, 2]
        x = torch.stack(output_list, dim=0)
        
        return x

    def apply_doppler_fft(self, x):
        """
        Apply Doppler FFT across Chirps dimension.
        """
        # x shape: [B, Rx, Chirps, Range, 2]
        B, Rx, Chirps, Range, _ = x.shape
        
        output_list = []
        
        for b in range(B):
            x_b = x[b]
            complex_input_b = torch.view_as_complex(x_b) # [Rx, Chirps, Range]
            
            # FFT on Chirps dimension (dim=1)
            fft_out_b = torch.fft.fft(complex_input_b, dim=1)
            
            # FFT Shift (optional but common for Doppler)
            fft_out_b = torch.fft.fftshift(fft_out_b, dim=1)
            
            output_list.append(torch.view_as_real(fft_out_b))
            
        x = torch.stack(output_list, dim=0)
        return x

    def forward(self, x):
        # x: [B, Rx, Chirps, Samples, 2]
        
        # 1. Time-domain preprocessing
        # Bypass time_conv to preserve phase information for FFT
        # x = x.permute(0, 4, 1, 2, 3) 
        # x = self.time_conv(x)
        # x = x.permute(0, 2, 3, 4, 1) # Back to [B, Rx, Chirps, Samples, 2]
        
        # 2. Demodulation (optional learnable mixer)
        # x = self.demodulate(x) 
        
        # 3. Range Processing (FFT)
        x = self.apply_range_fft(x) # [B, Rx, Chirps, Range, 2]
        
        # 4. Doppler Processing (FFT)
        x = self.apply_doppler_fft(x) # [B, Rx, Doppler, Range, 2]
        
        # 5. Coherent Integration (Sum over Rx)
        # Sum complex values
        x_complex = torch.view_as_complex(x) # [B, Rx, Doppler, Range]
        x_combined = torch.mean(x_complex, dim=1) # [B, Doppler, Range]
        
        # 6. Log-Magnitude for Input to U-Net
        x_mag = torch.abs(x_combined)
        x_log = 20 * torch.log10(x_mag + 1e-6)
        
        # Normalize x_log to roughly [0, 1] range
        # Assumes floor around -100dB and peak around 0-30dB
        x_norm = (x_log + 100.0) / 100.0
        
        # Prepare for U-Net: [B, 1, Doppler, Range]
        unet_input = x_norm.unsqueeze(1)
        
        # Resize to standard U-Net input size if needed
        # This handles varying RDM sizes (e.g. OTFS vs FMCW)
        # Using bilinear interpolation
        unet_input_resized = F.interpolate(unet_input, size=self.target_unet_size, 
                                          mode='bilinear', align_corners=False)
        
        # 7. U-Net Segmentation/Detection
        logits_resized = self.unet(unet_input_resized) # [B, 1, D_target, R_target]
        
        # Resize back to original RDM size
        original_size = x_log.shape[-2:] # (Doppler, Range)
        logits = F.interpolate(logits_resized, size=original_size, 
                              mode='bilinear', align_corners=False)
        
        return {
            'detection_logits': logits,
            'rd_map_db': x_log
        }

# ==========================================
# Helper Functions
# ==========================================

def collate_fn_custom(batch):
    """
    Custom collate function to handle variable size inputs if necessary,
    or just efficient stacking.
    """
    # Assume all items have same shape for now
    elem = batch[0]
    return {
        'time_domain': torch.stack([item['time_domain'] for item in batch]),
        'target_mask': torch.stack([item['target_mask'] for item in batch]),
        'range_doppler_map': torch.stack([item['range_doppler_map'] for item in batch]),
        'cfar_detections': [item['cfar_detections'] for item in batch], # List of lists
        'target_info': [item['target_info'] for item in batch] # List of dicts
    }

def cfar_to_mask(cfar_detections, mask_shape):
    """
    Convert CFAR detection list to binary mask.
    """
    mask = np.zeros(mask_shape, dtype=np.float32)
    for det in cfar_detections:
        r_idx = int(det['range_idx'])
        d_idx = int(det['doppler_idx'])
        if 0 <= r_idx < mask_shape[1] and 0 <= d_idx < mask_shape[0]:
            mask[d_idx, r_idx] = 1.0
    return torch.from_numpy(mask)

def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculate Precision, Recall, F1.
    """
    preds_bin = (preds > threshold).astype(int)
    targets_bin = (targets > 0.5).astype(int)
    
    tp = np.sum((preds_bin == 1) & (targets_bin == 1))
    fp = np.sum((preds_bin == 1) & (targets_bin == 0))
    fn = np.sum((preds_bin == 0) & (targets_bin == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1

# ==========================================
# Training and Evaluation Loop
# ==========================================

BATCH_SIZE = 2 # Small batch size for large radar tensors
NUM_EPOCHS = 20

def train_model(config_name='config1', save_dir='results/radar_config1'):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Dataset
    # Create a directory for the specific config
    save_dir = os.path.join(save_dir, f'radar_{config_name}')
    dataset_path = os.path.join(save_dir, 'radar_dataset.h5')
    
    # Check if dataset exists, else generate
    if not os.path.exists(dataset_path) or not os.path.exists(save_dir):
        print(f"Dataset for {config_name} not found at {dataset_path}. Generating new one...")
        ds = AIRadarDataset(
            num_samples=100, 
            config_name=config_name, 
            save_path=save_dir, 
            drawfig=False,
            apply_realistic_effects=False, # Enable realistic effects
            clutter_intensity=0.1         # Set moderate clutter intensity
        )
        # Note: AIRadarDataset generates data in __init__ if save_path provided? 
        # No, it generates if we call generate_dataset or if we rely on __init__ to load/generate.
        # Let's check AIradar_datasetv6.py again. 
        # It generates in __init__ if datapath is None.
        # So the above line already generated it.
        
    full_dataset = AIRadarDataset(datapath=dataset_path, config_name=config_name)
    
    # Verify dataset size
    if len(full_dataset) == 0:
        print("Error: Dataset is empty.")
        return
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_custom)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_custom)
    
    # 2. Initialize Model
    # Extract shapes from a sample to configure the model dynamically
    sample_0 = full_dataset[0]
    # sample_0['time_domain'] might be [Rx, Chirps, Samples, 2] or [Chirps, Samples, 2]
    # Dataset __getitem__ returns stacked real/imag.
    # Let's check tensor shape.
    td_shape = sample_0['time_domain'].shape
    if len(td_shape) == 4:
        n_rx, n_chirps, n_samples, _ = td_shape
    elif len(td_shape) == 3:
        n_rx = 1
        n_chirps, n_samples, _ = td_shape
    else:
        raise ValueError(f"Unexpected time domain shape: {td_shape}")
        
    n_doppler, n_range = sample_0['range_doppler_map'].shape
    
    print(f"Configuring model for {config_name}:")
    print(f"  Input: Rx={n_rx}, Chirps={n_chirps}, Samples={n_samples}")
    print(f"  Output: Doppler={n_doppler}, Range={n_range}")
    
    # Set FFT sizes based on config parameters if available, or derive from data
    # We want the model's FFT output to match the target RDM size
    
    model = RadarTimeNetV5(
        num_rx=n_rx,
        num_chirps=n_chirps,
        samples_per_chirp=n_samples,
        fft_size=n_range if n_range >= n_samples else None, # If target range bins > samples, pad. Else default.
        out_range_bins=n_range,
        out_doppler_bins=n_doppler,
        target_unet_size=(128, 2048) # Increased from (128, 128) to preserve range resolution
    ).to(DEVICE)
    
    # 3. Loss and Optimizer
    criterion = CombinedLoss(w_bce=10.0, w_focal=10.0, w_dice=5.0) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Fix: Remove 'verbose' argument for ReduceLROnPlateau if using newer torch or just default behavior
    # (In some torch versions verbose is deprecated or default)
    # Actually, verbose=True is valid in most versions, but maybe user environment is different?
    # Let's remove it to be safe, or use logging.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # 4. Training Loop
    print(f"Starting training for {config_name}...")
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'cfar_f1': []}
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            # [B, Rx, Chirps, Samples, 2] -> [B, Rx, Chirps, Samples, 2]
            # No need to unsqueeze if dataset returns [Rx, ...]. 
            # Dataset returns [Rx, Chirps, Samples, 2].
            # Model expects [B, Rx, Chirps, Samples, 2].
            iq_data = batch['time_domain'].to(DEVICE)
            
            # Fix dimensions if necessary
            # If iq_data is [B, Chirps, Samples, 2], unsqueeze dim 1 for Rx
            if iq_data.dim() == 4:
                iq_data = iq_data.unsqueeze(1) 
            
            # [B, Doppler, Range] -> [B, 1, Doppler, Range]
            # Check if target_mask already has an extra dimension at the end?
            # The error says target size is [2, 1, 128, 1334, 1]
            # This means batch['target_mask'] was [2, 128, 1334, 1] or similar.
            # Let's check batch['target_mask'] shape.
            tm = batch['target_mask']
            if tm.dim() == 3: # [B, D, R]
                 target_mask = tm.unsqueeze(1).to(DEVICE) # [B, 1, D, R]
            elif tm.dim() == 4 and tm.shape[-1] == 1: # [B, D, R, 1]
                 target_mask = tm.permute(0, 3, 1, 2).to(DEVICE) # [B, 1, D, R]
            else:
                 # Fallback or unexpected shape
                 target_mask = tm.unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(iq_data)
            logits = outputs['detection_logits']
            
            # Resize logits if needed to match target_mask (e.g. if UNet output size mismatch due to padding/resize)
            if logits.shape[-2:] != target_mask.shape[-2:]:
                logits = F.interpolate(logits, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(logits, target_mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation (Compare DL vs CFAR)
        val_results = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_results['loss']:.4f}")
        print(f"  DL Metrics: F1={val_results['dl_f1']:.4f}, Prec={val_results['dl_prec']:.4f}, Rec={val_results['dl_rec']:.4f}")
        print(f"  CFAR Metrics: F1={val_results['cfar_f1']:.4f}, Prec={val_results['cfar_prec']:.4f}, Rec={val_results['cfar_rec']:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_f1'].append(val_results['dl_f1'])
        history['cfar_f1'].append(val_results['cfar_f1'])
        
        scheduler.step(val_results['loss'])
        
        # 5. Save Model
        model_path = os.path.join(save_dir, f'radar_timenet_{config_name}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # 6. Plot Metrics
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss History')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_f1'], label='DL F1')
        plt.plot(history['cfar_f1'], label='CFAR F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('F1 Score History')
        
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        plt.close()
        print(f"Training metrics saved to {os.path.join(save_dir, 'training_metrics.png')}")
        
        # 7. Final Visualization
        visualize_comparison(model, test_dataset, DEVICE, config_name=config_name, save_dir=save_dir)
        
    return history

def evaluate(model, dataloader, criterion, device, max_samples_vis=5, save_dir=None):
    model.eval()
    total_loss = 0.0
    
    dl_preds, dl_targets = [], []
    cfar_preds, cfar_targets = [], []
    
    vis_count = 0
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Use tqdm for progress bar
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            iq_data = batch['time_domain'].to(device)
            
            # Fix dimensions if necessary
            # If iq_data is [B, Chirps, Samples, 2], unsqueeze dim 1 for Rx
            if iq_data.dim() == 4:
                iq_data = iq_data.unsqueeze(1)
            
            # [B, Doppler, Range] -> [B, 1, Doppler, Range]
            # target_mask = batch['target_mask'].unsqueeze(1).to(DEVICE)
            # Check if target_mask already has an extra dimension at the end?
            tm = batch['target_mask']
            if tm.dim() == 3: # [B, D, R]
                 target_mask = tm.unsqueeze(1).to(device) # [B, 1, D, R]
            elif tm.dim() == 4 and tm.shape[-1] == 1: # [B, D, R, 1]
                 target_mask = tm.permute(0, 3, 1, 2).to(device) # [B, 1, D, R]
            else:
                 # Fallback or unexpected shape
                 target_mask = tm.unsqueeze(1).to(device)
            
            outputs = model(iq_data)
            # outputs['rd_map_db'] shape: [B, Doppler, Range] (from forward method)
            # We need to check dimensions.
            # In forward: x_log = ... [B, Doppler, Range]
            # But wait, `x` was averaged across Rx.
            
            rd_map_db_tensor = outputs['rd_map_db']
            # Handle rd_map_db extraction for visualization later
                 
            logits = outputs['detection_logits'] # [B, 1, D, R]
            
            # Resize logits if needed to match target_mask
            if logits.shape[-2:] != target_mask.shape[-2:]:
                logits = F.interpolate(logits, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(logits, target_mask)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # DL Predictions
            dl_probs = torch.sigmoid(logits)
            dl_mask = (dl_probs > 0.1).float().cpu().numpy() # Lower threshold to 0.1 for high recall
            
            # Debug: Check if dl_mask has any positive predictions
            # if len(dl_preds) == 0: # Print once
            #      print(f"Debug: dl_mask unique values: {np.unique(dl_mask)}")
            #      print(f"Debug: target_mask unique values: {np.unique(target_mask.cpu().numpy())}")
            
            dl_preds.append(dl_mask.flatten())
            dl_targets.append(target_mask.cpu().numpy().flatten())
            
            # CFAR Predictions
            # batch_cfar = batch['cfar_detections'] # List of lists of dicts
            # batch_targets = batch['target_mask'].numpy() # [B, D, R]
            
            if 'cfar_detections' in batch:
                batch_cfar = batch['cfar_detections']
                batch_targets = batch['target_mask'].numpy()
                
                for i, cfar_list in enumerate(batch_cfar):
                     mask_shape = batch_targets[i].shape
                     cfar_mask = cfar_to_mask(cfar_list, mask_shape)
                     cfar_preds.append(cfar_mask.flatten())
                     cfar_targets.append(batch_targets[i].flatten())

            # --- Visualization Logic Merged Here ---
            if save_dir and vis_count < max_samples_vis:
                batch_size = iq_data.size(0)
                for b in range(batch_size):
                    if vis_count >= max_samples_vis:
                        break
                    
                    # Prepare data for visualization
                    # GT Mask
                    gt_mask_np = target_mask[b, 0].cpu().numpy() # [D, R]

                    # Input RDM
                    # Need to handle different tensor shapes for rd_map_db
                    if rd_map_db_tensor.dim() == 3: # [B, D, R]
                        gt_rdm_np = rd_map_db_tensor[b].cpu().numpy()
                    elif rd_map_db_tensor.dim() == 4: # [B, C, D, R]
                        gt_rdm_np = rd_map_db_tensor[b, 0].cpu().numpy()
                    else:
                        # Fallback if shape is unexpected
                        gt_rdm_np = np.zeros_like(gt_mask_np) 
        
                    # CFAR Detections
                    cfar_detections_sample = batch['cfar_detections'][b] if 'cfar_detections' in batch else []
                    
                    # --- Extract DL Detections for Visualization ---
                    # We need to extract 'detections' from pred_mask similar to how CFAR does it
                    
                    # Reconstruct targets from GT mask (simplified)
                    # This is an approximation for visualization
                    gt_rows, gt_cols = np.where(gt_mask_np > 0)
                    targets_viz = []
                    # We need range/velocity axis information to convert indices to physical units
                    # We can get this from the dataset object if we had access to it.
                    # But here we are in `evaluate` which takes `dataloader`.
                    # `dataloader.dataset` should give us access.
                    dataset_ref = dataloader.dataset
                    # Handle Subset
                    if isinstance(dataset_ref, torch.utils.data.Subset):
                        dataset_ref = dataset_ref.dataset
                        
                    r_axis = dataset_ref.range_axis
                    v_axis = dataset_ref.velocity_axis
                    
                    for r_idx, d_idx in zip(gt_cols, gt_rows):
                        t = {
                            'range': r_axis[r_idx],
                            'velocity': v_axis[d_idx],
                            'range_idx': r_idx,
                            'doppler_idx': d_idx
                        }
                        targets_viz.append(t)
                        
                    # Reconstruct detections from DL Mask
                    pred_mask_np = dl_mask[b, 0]
                    # Use connected components to find centroids for cleaner detection list
                    from scipy.ndimage import label, center_of_mass
                    labeled_array, num_features = label(pred_mask_np)
                    detections_viz = []
                    if num_features > 0:
                        centers = center_of_mass(pred_mask_np, labeled_array, range(1, num_features+1))
                        for center in centers:
                            d_idx, r_idx = center
                            d_idx_int = int(round(d_idx))
                            r_idx_int = int(round(r_idx))
                            
                            # Ensure within bounds
                            d_idx_int = max(0, min(d_idx_int, len(v_axis)-1))
                            r_idx_int = max(0, min(r_idx_int, len(r_axis)-1))
                            
                            d = {
                                'range_m': r_axis[r_idx_int],
                                'velocity_mps': v_axis[d_idx_int],
                                'range_idx': r_idx_int,
                                'doppler_idx': d_idx_int,
                                'power': 0 # Dummy
                            }
                            detections_viz.append(d)
                        
                    # Run per-sample evaluation for visualization
                    # We can use a helper or method from dataset if available.
                    # dataset_ref._evaluate_metrics is an instance method.
                    metrics_viz, matched_pairs, unmatched_targets, unmatched_detections = dataset_ref._evaluate_metrics(targets_viz, detections_viz)
                    
                    # --- Call Visualization Functions ---
                    
                    # 2D Plot for DL
                    save_path_2d_dl = f"{save_dir}/detailed_batch{batch_idx}_sample{b}_dl_2d.png"
                    # We call the imported standalone function
                    
                    # Ensure rdm is a float numpy array
                    if torch.is_tensor(gt_rdm_np):
                        gt_rdm_np = gt_rdm_np.cpu().numpy()
                    
                    gt_rdm_np = gt_rdm_np.astype(float)
                    
                    # Normalize RDM for visualization
                    rdm_norm = gt_rdm_np - np.max(gt_rdm_np)

                    _plot_2d_rdm(
                        dataset_ref, # Pass dataset instance
                        rdm_norm, 
                        f"{batch_idx}_{b}_DL",
                        metrics_viz,
                        matched_pairs,
                        unmatched_targets,
                        unmatched_detections,
                        save_path_2d_dl
                    )
                    
                    # 3D Plot for DL
                    save_path_3d_dl = f"{save_dir}/detailed_batch{batch_idx}_sample{b}_dl_3d.png"
                    _plot_3d_rdm(
                        dataset_ref,
                        rdm_norm,
                        f"{batch_idx}_{b}_DL",
                        targets_viz,
                        detections_viz,
                        save_path_3d_dl
                    )
                    
                    # CFAR comparison
                    metrics_cfar, matched_cfar, unmatched_targets_cfar, unmatched_detections_cfar = dataset_ref._evaluate_metrics(targets_viz, cfar_detections_sample)
                    
                    save_path_2d_cfar = f"{save_dir}/detailed_batch{batch_idx}_sample{b}_cfar_2d.png"
                    _plot_2d_rdm(
                        dataset_ref,
                        rdm_norm,
                        f"{batch_idx}_{b}_CFAR",
                        metrics_cfar,
                        matched_cfar,
                        unmatched_targets_cfar,
                        unmatched_detections_cfar,
                        save_path_2d_cfar
                    )
                    
                    save_path_3d_cfar = f"{save_dir}/detailed_batch{batch_idx}_sample{b}_cfar_3d.png"
                    _plot_3d_rdm(
                        dataset_ref,
                        rdm_norm,
                        f"{batch_idx}_{b}_CFAR",
                        targets_viz,
                        cfar_detections_sample,
                        save_path_3d_cfar
                    )
                    
                    vis_count += 1

    # DL Metrics
    dl_all_preds = np.concatenate(dl_preds)
    dl_all_targets = np.concatenate(dl_targets)
    dl_prec, dl_rec, dl_f1 = calculate_metrics(dl_all_preds, dl_all_targets)
    
    # CFAR Metrics
    if cfar_preds:
        cfar_all_preds = np.concatenate(cfar_preds)
        cfar_all_targets = np.concatenate(cfar_targets)
        cfar_prec, cfar_rec, cfar_f1 = calculate_metrics(cfar_all_preds, cfar_all_targets)
    else:
        cfar_prec, cfar_rec, cfar_f1 = 0, 0, 0
    
    return {
        'loss': total_loss / len(dataloader),
        'dl_f1': dl_f1, 'dl_prec': dl_prec, 'dl_rec': dl_rec,
        'cfar_f1': cfar_f1, 'cfar_prec': cfar_prec, 'cfar_rec': cfar_rec
    }

def visualize_comparison(model, dataset, device, num_samples=3, config_name='default', save_dir='results'):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    save_dir = os.path.join(save_dir, f'results_{config_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Retrieve dataset axes for plotting
    # Handle Subset case
    if hasattr(dataset, 'dataset'):
        dataset_instance = dataset.dataset
    else:
        dataset_instance = dataset
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        iq_data = sample['time_domain'].unsqueeze(0).to(device) # [1, Rx, Chirps, Samples, 2]
        
        # Fix dimensions if necessary
        # If iq_data is [1, Chirps, Samples, 2], unsqueeze dim 1 for Rx
        if iq_data.dim() == 4:
            iq_data = iq_data.unsqueeze(1)
            
        target_mask = sample['target_mask'].numpy()
        
        # Retrieve targets from original dataset using dataset index (if it's a subset, we need to be careful)
        if 'sample_idx' in sample:
            sample_idx = sample['sample_idx']
        elif 'target_info' in sample and 'sample_idx' in sample['target_info']:
            sample_idx = sample['target_info']['sample_idx']
        else:
            if hasattr(dataset, 'indices'):
                sample_idx = dataset.indices[idx]
            else:
                sample_idx = idx
        
        if hasattr(dataset, 'dataset'):
             targets = dataset.dataset.target_info[sample_idx]['targets']
        else:
             targets = dataset.target_info[sample_idx]['targets']
        
        with torch.no_grad():
            outputs = model(iq_data)
            rd_map_db_tensor = outputs['rd_map_db']
            
            if rd_map_db_tensor.dim() == 3: # [B, D, R]
                 rd_map_db = rd_map_db_tensor[0].cpu().numpy()
            elif rd_map_db_tensor.dim() == 4: 
                 rd_map_db = rd_map_db_tensor[0, 0].cpu().numpy()
            else:
                 rd_map_db = rd_map_db_tensor[0].cpu().numpy()
                 
            logits = outputs['detection_logits'][0, 0].cpu().numpy()
            
        # DL Mask and Detections
        dl_mask = (1 / (1 + np.exp(-logits))) > 0.5 # Or 0.1 based on previous tuning
        
        # Convert DL mask to detection list for metrics and plotting
        from scipy.ndimage import label, center_of_mass
        labeled_array, num_features = label(dl_mask)
        dl_detections = []
        
        # Get axes
        range_axis = dataset_instance.range_axis
        velocity_axis = dataset_instance.velocity_axis
        
        if num_features > 0:
            centers = center_of_mass(dl_mask, labeled_array, range(1, num_features+1))
            for center in centers:
                d_idx, r_idx = center
                d_idx_int = int(round(d_idx))
                r_idx_int = int(round(r_idx))
                
                # Ensure within bounds
                d_idx_int = max(0, min(d_idx_int, len(velocity_axis)-1))
                r_idx_int = max(0, min(r_idx_int, len(range_axis)-1))
                
                dl_detections.append({
                    'range_m': range_axis[r_idx_int],
                    'velocity_mps': velocity_axis[d_idx_int],
                    'range_idx': r_idx_int,
                    'doppler_idx': d_idx_int
                })
        
        # CFAR Detections
        cfar_detections = sample['cfar_detections']
        
        # Normalize RDM
        rdm_norm = rd_map_db - np.max(rd_map_db)
        
        # --- 1. DL Evaluation and Plotting ---
        metrics_dl, matched_dl, unmatched_targets_dl, unmatched_detections_dl = \
            dataset_instance._evaluate_metrics(targets, dl_detections)
            
        save_path_2d_dl = os.path.join(save_dir, f"sample_{idx}_dl_2d.png")
        _plot_2d_rdm(dataset_instance, rdm_norm, idx, metrics_dl, matched_dl, unmatched_targets_dl, unmatched_detections_dl, 
                     save_path_2d_dl)
        
        save_path_3d_dl = os.path.join(save_dir, f"sample_{idx}_dl_3d.png")
        _plot_3d_rdm(dataset_instance, rdm_norm, idx, targets, dl_detections, save_path_3d_dl)
        
        # --- 2. CFAR Evaluation and Plotting ---
        metrics_cfar, matched_cfar, unmatched_targets_cfar, unmatched_detections_cfar = \
            dataset_instance._evaluate_metrics(targets, cfar_detections)
            
        save_path_2d_cfar = os.path.join(save_dir, f"sample_{idx}_cfar_2d.png")
        _plot_2d_rdm(dataset_instance, rdm_norm, idx, metrics_cfar, matched_cfar, unmatched_targets_cfar, unmatched_detections_cfar, 
                     save_path_2d_cfar)
        
        save_path_3d_cfar = os.path.join(save_dir, f"sample_{idx}_cfar_3d.png")
        _plot_3d_rdm(dataset_instance, rdm_norm, idx, targets, cfar_detections, save_path_3d_cfar)
        
        print(f"Sample {idx}: Saved DL and CFAR visualizations (2D & 3D).")

    print(f"All visualizations saved to {save_dir}")

def load_and_evaluate(model_path='radar_timenet_v5.pth', dataset_path='data/radar_config2/radar_dataset.h5', save_dir='data/radar_corrected_test5/evaluation'):
    """
    Load a pretrained model, evaluate it on the test dataset, and generate detailed visualizations.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Setup Device and Data
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please generate it first.")
        return

    # Load full dataset
    full_dataset = AIRadarDataset(datapath=dataset_path, save_path='data/radar_config2')
    
    # Use a deterministic split for reproducibility (or just use the whole dataset for eval if preferred)
    # Here we mimic the train split to get the "test" portion, but ideally we should have saved indices.
    # For now, let's just use the last 20% as test to be consistent with training.
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42) # Use seed to try to get same split if possible, though random_split without seed in train_model makes this hard.
        # actually, train_model didn't use a seed. So we can't guarantee same split.
        # Let's just evaluate on the *entire* dataset for validation purposes.
    )
    
    # Use full dataset for comprehensive evaluation
    test_loader = DataLoader(full_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_custom)
    
    # 2. Initialize Model
    sample_0 = full_dataset[0]
    n_chirps, n_samples, _ = sample_0['time_domain'].shape
    n_doppler, n_range = sample_0['range_doppler_map'].shape
    
    model = RadarTimeNetV5(
        num_rx=1,
        num_chirps=n_chirps,
        samples_per_chirp=n_samples,
        fft_size=4096,
        out_range_bins=n_range,
        out_doppler_bins=n_doppler
    ).to(DEVICE)
    
    # Load Weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"Model file {model_path} not found!")
        return

    # 3. Evaluate
    criterion = CombinedLoss(w_bce=1.0, w_focal=0.5, w_dice=0.5)
    print("Starting evaluation...")
    
    # Pass save_dir and max_samples_vis to evaluate
    # Ensure we use the visualization logic consistent with dataset script
    metrics = evaluate(model, test_loader, criterion, DEVICE, max_samples_vis=5, save_dir=save_dir)
    
    print("\n=== Evaluation Results ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"DL Metrics:   F1={metrics['dl_f1']:.4f}, Precision={metrics['dl_prec']:.4f}, Recall={metrics['dl_rec']:.4f}")
    print(f"CFAR Metrics: F1={metrics['cfar_f1']:.4f}, Precision={metrics['cfar_prec']:.4f}, Recall={metrics['cfar_rec']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train or Evaluate RadarTimeNet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode: train or eval')
    parser.add_argument('--model_path', type=str, default='results/radar_config1/radar_timenet_config1.pth', help='Path to model weights for eval')
    parser.add_argument('--config', type=str, default='config1', help='Radar configuration to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Update global constants based on args if needed (though globals are tricky, we pass config_name)
    NUM_EPOCHS = args.epochs
    
    if args.mode == 'train':
        train_model(config_name=args.config, save_dir=f'results/radar_{args.config}')
    else:
        dataset_path = f'results/radar_{args.config}/radar_dataset.h5'
        save_dir = f'results/radar_{args.config}/evaluation'
        load_and_evaluate(model_path=args.model_path, dataset_path=dataset_path, save_dir=save_dir)
