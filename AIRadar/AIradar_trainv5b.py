#!/usr/bin/env python3
"""
RadarTimeNet Training Script with Simulation Data (v5)

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
from AIRadar.AIradar_datasetv7 import AIRadarDataset

# Try to import torch.fft for modern PyTorch versions
try:
    import torch.fft
    HAS_TORCH_FFT = True
except ImportError:
    HAS_TORCH_FFT = False

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
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        loss = 0
        if self.w_bce > 0:
            loss += self.w_bce * self.bce(inputs, targets)
        if self.w_focal > 0:
            loss += self.w_focal * self.focal(inputs, targets)
        if self.w_dice > 0:
            loss += self.w_dice * self.dice(inputs, targets)
        return loss

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
    U-Net architecture for 2D Range-Doppler map processing.
    Input: [B, 2, Doppler, Range] (Real/Imag parts of RD map)
    Output: [B, 1, Doppler, Range] (Detection probability)
    """
    def __init__(self, n_channels=2, n_classes=1, bilinear=True):
        super(UNetRadar2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        # Encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)
        
        # Decoder
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        # Initialize the final convolution bias for sparse targets
        # p = 0.01 (expected probability of target) -> log(p/(1-p)) = log(0.01/0.99) = -4.595
        self.outc.conv.bias.data.fill_(-4.6)

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
        return logits

class RadarTimeNetV5(nn.Module):
    """
    Deep learning model for processing time-domain IQ signals to extract range-Doppler maps
    and perform object detection.
    """
    def __init__(self, num_rx=1, num_chirps=128, samples_per_chirp=2048, 
                 fft_size=4096, out_range_bins=2048, out_doppler_bins=128,
                 use_learnable_fft=False, target_unet_size=(128, 2048)):
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
        # REMOVED ReLU to preserve phase information for FFT
        # Input shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        self.time_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(16),
            # nn.ReLU(inplace=True), # REMOVED: ReLU destroys phase info (negative values)
            nn.Conv3d(16, 2, kernel_size=(1, 1, 3), padding=(0, 0, 1)), # Reduce back to 2 channels
            nn.BatchNorm3d(2),
            # nn.ReLU(inplace=True) # REMOVED
        )
        
        # === Demodulation module ===
        self.demod_weights = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))
        
        # === U-Net Post-processing ===
        # Input to UNet will be 2 channels (Real/Imag) or 3 (Real/Imag/Mag)
        # Let's use 3 channels: Real, Imag, and Log-Magnitude for better features
        self.unet = UNetRadar2D(n_channels=3, n_classes=1, bilinear=True)
        
        self._init_weights()

    def _init_weights(self):
        self.demod_weights.data = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
        
    def demodulate(self, rx_signal):
        rx_signal_flat = rx_signal.reshape(-1, 2)
        demod_signal_flat = torch.matmul(rx_signal_flat, self.demod_weights)
        return demod_signal_flat.reshape(*rx_signal.shape)
    
    def apply_range_fft(self, x):
        batch_size, num_rx, num_chirps, samples_per_chirp, _ = x.shape
        x_reshaped = x.reshape(batch_size * num_rx * num_chirps, samples_per_chirp, 2)
        real_part, imag_part = x_reshaped[..., 0], x_reshaped[..., 1]
        
        complex_input = torch.complex(real_part, imag_part)
        
        # Dynamic FFT size: Use configured fft_size or at least the input size
        # If input is smaller, zero-padding happens (interpolation)
        # If input is larger, we might want to increase fft size or truncate
        current_fft_size = max(self.fft_size, samples_per_chirp) if self.fft_size else samples_per_chirp
        
        complex_output = torch.fft.fft(complex_input, n=current_fft_size, dim=1)
        
        # Determine output bins
        # If out_range_bins is set, use it (cropping or padding handled by slicing/indexing)
        # But slicing [:2048] on a 1024 array will fail if we didn't pad.
        # current_fft_size is >= samples_per_chirp.
        # We typically want the first half or a specific portion.
        
        out_bins = self.out_range_bins if self.out_range_bins else current_fft_size
        
        # Ensure we don't slice out of bounds
        valid_bins = min(out_bins, current_fft_size)
        complex_output = complex_output[:, :valid_bins]
        
        range_spectrum = torch.stack([complex_output.real, complex_output.imag], dim=-1)
        return range_spectrum.reshape(batch_size, num_rx, num_chirps, valid_bins, 2)
    
    def apply_doppler_fft(self, x):
        batch_size, num_rx, num_chirps, range_bins, _ = x.shape
        x_transposed = x.permute(0, 1, 3, 2, 4)
        x_reshaped = x_transposed.reshape(batch_size * num_rx * range_bins, num_chirps, 2)
        real_part, imag_part = x_reshaped[..., 0], x_reshaped[..., 1]
        
        complex_input = torch.complex(real_part, imag_part)
        
        # Dynamic Doppler FFT
        current_doppler_fft = max(self.out_doppler_bins, num_chirps) if self.out_doppler_bins else num_chirps
        
        complex_output = torch.fft.fft(complex_input, n=current_doppler_fft, dim=1)
        complex_output = torch.fft.fftshift(complex_output, dim=1)
        
        doppler_spectrum = torch.stack([complex_output.real, complex_output.imag], dim=-1)
        return doppler_spectrum.reshape(batch_size, num_rx, range_bins, current_doppler_fft, 2).permute(0, 1, 3, 2, 4)

    def forward(self, x):
        # x: [B, num_rx, num_chirps, samples_per_chirp, 2]
        
        # 1. Time-domain preprocessing
        x = x.permute(0, 4, 1, 2, 3) # [B, 2, Rx, Chirps, Samples]
        
        # Optional: Skip time_conv for now to establish baseline, or use the linear version
        # x = self.time_conv(x) 
        
        x = x.permute(0, 2, 3, 4, 1) # Back to [B, Rx, Chirps, Samples, 2]
        
        # 2. Demodulation
        x = self.demodulate(x)
        
        # 3. Range FFT
        x = self.apply_range_fft(x) # [B, Rx, Chirps, Range, 2]
        
        # 4. Doppler FFT
        x = self.apply_doppler_fft(x) # [B, Rx, Doppler, Range, 2]
        
        # 5. Post-processing
        # Average across receive antennas (if Rx > 1)
        x = x.mean(dim=1) # [B, Doppler, Range, 2]
        
        # Save original RD shape
        original_rd_shape = x.shape[1:3] # (Doppler, Range)
        
        # Compute features for U-Net
        # 1. Real and Imag parts
        x_real = x[..., 0]
        x_imag = x[..., 1]
        
        # 2. Log-Magnitude (Standard Radar View)
        x_mag = torch.norm(x, dim=-1)
        x_log = 20 * torch.log10(x_mag + 1e-6)
        
        # Normalize Log-Magnitude roughly to [-1, 1] or [0, 1] range for NN
        x_log_norm = (x_log + 50.0) / 100.0 # Approx normalization
        
        # Stack for U-Net: [B, 3, Doppler, Range]
        x_in = torch.stack([x_real, x_imag, x_log_norm], dim=1)
        
        # --- RESIZING FOR ROBUSTNESS ---
        # Resize to fixed target size for U-Net if defined
        if self.target_unet_size is not None and (x_in.shape[2:] != self.target_unet_size):
            x_in_resized = F.interpolate(x_in, size=self.target_unet_size, mode='bilinear', align_corners=False)
        else:
            x_in_resized = x_in
            
        # Detection Map (Logits)
        detection_logits_resized = self.unet(x_in_resized) # [B, 1, Target_Doppler, Target_Range]
        
        # Resize back to original RD map resolution
        if self.target_unet_size is not None and (detection_logits_resized.shape[2:] != original_rd_shape):
            detection_logits = F.interpolate(detection_logits_resized, size=original_rd_shape, mode='bilinear', align_corners=False)
        else:
            detection_logits = detection_logits_resized
            
        return {
            'rd_map_db': x_log, # Return unnormalized dB for visualization (Original Scale)
            'detection_logits': detection_logits # (Original Scale)
        }

# ==========================================
# Helper Functions
# ==========================================

def collate_fn_custom(batch):
    """
    Custom collate function to handle variable size cfar_detections list
    """
    elem = batch[0]
    collated = {}
    for key in elem:
        if key == 'cfar_detections':
            collated[key] = [d[key] for d in batch]
        elif key == 'target_info':
            collated[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], np.ndarray):
            collated[key] = torch.stack([torch.from_numpy(d[key]) for d in batch])
        else:
            collated[key] = [d[key] for d in batch]
    return collated

def calculate_metrics(preds, targets):
    from sklearn.metrics import precision_score, recall_score, f1_score
    flat_preds = preds.flatten()
    flat_targets = targets.flatten()
    
    precision = precision_score(flat_targets, flat_preds, zero_division=0)
    recall = recall_score(flat_targets, flat_preds, zero_division=0)
    f1 = f1_score(flat_targets, flat_preds, zero_division=0)
    return precision, recall, f1

def cfar_to_mask(cfar_list, shape):
    """Convert CFAR detection list to binary mask"""
    mask = np.zeros(shape, dtype=np.float32)
    for det in cfar_list:
        try:
            r_idx = int(det['range_idx'])
            d_idx = int(det['doppler_idx'])
            if 0 <= r_idx < shape[1] and 0 <= d_idx < shape[0]:
                mask[d_idx, r_idx] = 1.0
        except Exception as e:
            print(f"Error processing CFAR detection: {det}, Error: {e}")
    return mask

# ==========================================
# Training & Evaluation
def train_model(
    output_path: str = 'data/',
    model_name: str = 'radar_timenet_v5',
    data_path: str = 'data/radar_corrected_test5/radar_dataset.h5',
    device: str = 'cuda',
    batch_size: int = 8,
    num_epochs: int = 30,
    learning_rate: float = 0.001
) -> dict:
    """
    Train the RadarTimeNetV5 model.

    Args:
        output_path (str): Path to save the trained model weights.
        data_path (str): Path to the HDF5 dataset file.
        device (str): Device to use for training ('cuda' or 'cpu').
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        dict: Training history dictionary containing loss and metrics.
    """
    # Parameters
    DEVICE = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Dataset
    save_dir = os.path.dirname(data_path)
    
    # Check if dataset exists, else generate
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating new one...")
        ds = AIRadarDataset(num_samples=50, save_path=save_dir, drawfig=False)
        ds.generate_dataset()
        # Re-verify path (ds.generate_dataset might save to a specific name)
        if not os.path.exists(data_path):
             data_path = os.path.join(save_dir, 'radar_dataset.h5')

    full_dataset = AIRadarDataset(datapath=data_path, save_path=save_dir)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_custom)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_custom)
    
    # 2. Initialize Model
    # Extract shapes from a sample
    sample_0 = full_dataset[0]
    # sample_0['time_domain'] is [Chirps, Samples, 2]
    n_chirps, n_samples, _ = sample_0['time_domain'].shape
    n_doppler, n_range = sample_0['range_doppler_map'].shape
    
    model = RadarTimeNetV5(
        num_rx=1,
        num_chirps=n_chirps,
        samples_per_chirp=n_samples,
        fft_size=4096, # Assumed, adjust if needed
        out_range_bins=n_range,
        out_doppler_bins=n_doppler
    ).to(DEVICE)
    
    # 3. Loss and Optimizer
    # Adjusted weights to penalize false positives more (since we had Prec~0)
    # w_bce=0.5, w_focal=1.0, w_dice=1.0 -> The model was over-predicting.
    # Let's reduce Focal/Dice slightly or rely more on BCE with pos_weight?
    # Actually, with the architecture fix (removing ReLU), the model might just converge naturally.
    # But let's keep weights balanced.
    criterion = CombinedLoss(w_bce=1.0, w_focal=0.5, w_dice=0.5) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Note: In newer PyTorch versions, verbose is deprecated or moved. We remove it to be safe.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 4. Training Loop
    print("Starting training...")
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'cfar_f1': []}
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # [B, Chirps, Samples, 2] -> [B, 1, Chirps, Samples, 2]
            iq_data = batch['time_domain'].unsqueeze(1).to(DEVICE) 
            # [B, Doppler, Range, 1] -> [B, 1, Doppler, Range]
            target_mask = batch['target_mask'].permute(0, 3, 1, 2).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(iq_data)
            logits = outputs['detection_logits']
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
        
        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save(model.state_dict(), os.path.join(output_path, f'{model_name}.pth'))
            print(f"Model saved to {os.path.join(output_path, f'{model_name}.pth')} (Best Val Loss: {best_val_loss:.4f})")
        
    # 5. Final Visualization
    # We use the best model for visualization
    model.load_state_dict(torch.load(os.path.join(output_path, f'{model_name}.pth')))
    visualize_results(model, test_dataset, DEVICE, save_dir=output_path, max_vis_samples=5)
    
    return history

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    dl_preds, dl_targets = [], []
    cfar_preds, cfar_targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            iq_data = batch['time_domain'].unsqueeze(1).to(device)
            target_mask = batch['target_mask'].permute(0, 3, 1, 2).to(device)
            
            outputs = model(iq_data)
            logits = outputs['detection_logits']
            loss = criterion(logits, target_mask)
            total_loss += loss.item()
            
            # DL Predictions
            preds = torch.sigmoid(logits) > 0.5
            dl_preds.append(preds.cpu().numpy())
            dl_targets.append(target_mask.cpu().numpy())
            
            # CFAR Predictions
            batch_cfar = batch['cfar_detections'] # List of lists of dicts
            batch_targets = batch['target_mask'].squeeze(-1).numpy() # [B, D, R]
            
            for i, cfar_list in enumerate(batch_cfar):
                mask_shape = batch_targets[i].shape
                cfar_mask = cfar_to_mask(cfar_list, mask_shape)
                cfar_preds.append(cfar_mask.flatten())
                cfar_targets.append(batch_targets[i].flatten())
            
    # DL Metrics
    dl_all_preds = np.concatenate(dl_preds).flatten()
    dl_all_targets = np.concatenate(dl_targets).flatten()
    dl_prec, dl_rec, dl_f1 = calculate_metrics(dl_all_preds, dl_all_targets)
    
    # CFAR Metrics
    cfar_all_preds = np.concatenate(cfar_preds)
    cfar_all_targets = np.concatenate(cfar_targets)
    cfar_prec, cfar_rec, cfar_f1 = calculate_metrics(cfar_all_preds, cfar_all_targets)
    
    return {
        'loss': total_loss / len(dataloader),
        'dl_f1': dl_f1, 'dl_prec': dl_prec, 'dl_rec': dl_rec,
        'cfar_f1': cfar_f1, 'cfar_prec': cfar_prec, 'cfar_rec': cfar_rec
    }

def visualize_results(model, dataset, device, max_vis_samples=5, save_dir='data/radar_corrected_test5/model_comparison'):
    """
    Visualize comparison between Ground Truth, DL Prediction, and CFAR Detection.
    Includes detailed 3x3 grid analysis and 2D/3D RD Map visualizations using dataset tools.
    """
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from AIradar_datasetv7 import _plot_2d_rdm, _plot_3d_rdm
        HAS_VIZ_TOOLS = True
    except ImportError:
        print("Warning: Could not import visualization tools from AIradar_datasetv7.py")
        HAS_VIZ_TOOLS = False

    model.eval()
    actual_samples = min(max_vis_samples, len(dataset))
    if actual_samples == 0:
        print("No samples to visualize.")
        return
        
    indices = np.random.choice(len(dataset), actual_samples, replace=False)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in indices:
        sample = dataset[idx]
        iq_data = sample['time_domain'].unsqueeze(0).unsqueeze(1).to(device)
        gt_mask = sample['target_mask'].squeeze().numpy() # [D, R]
        gt_rdm = sample['range_doppler_map'].numpy()      # [D, R]
        cfar_detections = sample['cfar_detections']
        target_info = sample['target_info']
        
        # Generate CFAR mask
        cfar_mask = cfar_to_mask(cfar_detections, gt_mask.shape)
        
        with torch.no_grad():
            outputs = model(iq_data)
            pred_logits = outputs['detection_logits'].squeeze().cpu().numpy() # [D, R]
            pred_prob = 1 / (1 + np.exp(-pred_logits))
            pred_mask = pred_prob > 0.5
            pred_rdm_db = outputs['rd_map_db'].squeeze().cpu().numpy()
            
        # --- 1. Comprehensive 3x3 Grid Visualization ---
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        plt.suptitle(f"Detailed Analysis - Sample {idx}", fontsize=16)
        
        # Row 1: Inputs and Ground Truth
        im0 = axes[0, 0].imshow(gt_rdm, aspect='auto', origin='lower', cmap='jet')
        axes[0, 0].set_title("Input Range-Doppler Map (dB)")
        plt.colorbar(im0, ax=axes[0, 0])
        
        axes[0, 1].imshow(gt_mask, aspect='auto', origin='lower', cmap='gray')
        axes[0, 1].set_title("Ground Truth Mask")
        
        axes[0, 2].imshow(gt_rdm, aspect='auto', origin='lower', cmap='gray')
        axes[0, 2].imshow(gt_mask, aspect='auto', origin='lower', cmap='spring', alpha=0.5)
        axes[0, 2].set_title("GT Overlay (Pink)")
        
        # Row 2: DL Model Outputs
        im3 = axes[1, 0].imshow(pred_rdm_db, aspect='auto', origin='lower', cmap='jet')
        axes[1, 0].set_title("DL Internal Feature (Log-Mag)")
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(pred_prob, aspect='auto', origin='lower', cmap='inferno')
        axes[1, 1].set_title("DL Prediction Probability")
        plt.colorbar(im4, ax=axes[1, 1])
        
        axes[1, 2].imshow(pred_mask, aspect='auto', origin='lower', cmap='gray')
        axes[1, 2].set_title("DL Binary Mask (Thresh=0.5)")
        
        # Row 3: Comparisons and CFAR
        axes[2, 0].imshow(cfar_mask, aspect='auto', origin='lower', cmap='gray')
        axes[2, 0].set_title(f"CFAR Detection ({len(cfar_detections)} pts)")
        
        # DL Errors (Green: TP, Red: FP, Blue: FN)
        error_map = np.zeros((*gt_mask.shape, 3))
        tp = np.logical_and(pred_mask, gt_mask)
        fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
        
        error_map[tp] = [0, 1, 0] # Green
        error_map[fp] = [1, 0, 0] # Red
        error_map[fn] = [0, 0, 1] # Blue
        
        axes[2, 1].imshow(error_map, aspect='auto', origin='lower')
        axes[2, 1].set_title("DL Error Analysis (G=TP, R=FP, B=FN)")
        
        # 1D Range Profile (Cut at max Doppler)
        if np.sum(gt_mask) > 0:
            d_idx, r_idx = np.unravel_index(np.argmax(gt_mask), gt_mask.shape)
        else:
            d_idx = gt_mask.shape[0] // 2
            
        axes[2, 2].plot(gt_rdm[d_idx, :], label='Input RDM')
        axes[2, 2].plot(pred_prob[d_idx, :] * np.max(gt_rdm), label='DL Prob (Scaled)', linestyle='--')
        axes[2, 2].set_title(f"Range Profile at Doppler Bin {d_idx}")
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/detailed_grid_{idx}.png")
        plt.close()

        # --- 2. Advanced Visualization (2D & 3D) using dataset tools ---
        if HAS_VIZ_TOOLS:
            targets = target_info['targets']
            
            # Convert DL mask to detections list
            dl_detections = []
            rows, cols = np.where(pred_mask)
            range_axis = sample['range_axis']
            velocity_axis = sample['velocity_axis']
            
            def get_val(axis, idx):
                if idx >= len(axis): return axis[-1]
                return axis[idx]

            for r_idx, d_idx in zip(rows, cols):
                 doppler_idx = r_idx
                 range_idx = d_idx
                 
                 dl_detections.append({
                     'range_m': get_val(range_axis, range_idx),
                     'velocity_mps': get_val(velocity_axis, doppler_idx),
                     'range_idx': range_idx,
                     'doppler_idx': doppler_idx
                 })

            class MockDataset:
                def __init__(self, r_axis, v_axis):
                    self.range_axis = r_axis
                    self.velocity_axis = v_axis
            
            mock_ds = MockDataset(range_axis, velocity_axis)
            
            # 3D Plots
            _plot_3d_rdm(
                dataset_instance=mock_ds,
                rdm=gt_rdm,
                sample_idx=f"{idx}_DL",
                targets=targets,
                detections=dl_detections,
                save_path=f"{save_dir}/3d_dl_sample_{idx}.png"
            )
            
            _plot_3d_rdm(
                dataset_instance=mock_ds,
                rdm=gt_rdm,
                sample_idx=f"{idx}_CFAR",
                targets=targets,
                detections=cfar_detections,
                save_path=f"{save_dir}/3d_cfar_sample_{idx}.png"
            )
            
            # Matching Logic for 2D Plots
            def simple_match(targets, detections, threshold=5.0): # Using updated threshold
                matched = []
                unmatched_t = list(targets)
                unmatched_d = list(detections)
                
                for t in targets:
                    best_d = None
                    best_dist = float('inf')
                    for d in detections:
                        dist = np.sqrt((t['range'] - d['range_m'])**2 + (t['velocity'] - d['velocity_mps'])**2)
                        if dist < threshold and dist < best_dist:
                            best_dist = dist
                            best_d = d
                    
                    if best_d:
                        matched.append((t, best_d))
                        if t in unmatched_t: unmatched_t.remove(t)
                        if best_d in unmatched_d: unmatched_d.remove(best_d)
                        
                tp = len(matched)
                fp = len(unmatched_d)
                fn = len(unmatched_t)
                
                metrics = {
                    'num_targets': len(targets),
                    'num_detections': len(detections),
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'mean_range_error': np.mean([abs(t['range'] - d['range_m']) for t, d in matched]) if matched else 0,
                    'mean_velocity_error': np.mean([abs(t['velocity'] - d['velocity_mps']) for t, d in matched]) if matched else 0
                }
                return metrics, matched, unmatched_t, unmatched_d

            # 2D Plots
            m_dl, pairs_dl, miss_dl, fa_dl = simple_match(targets, dl_detections)
            _plot_2d_rdm(mock_ds, gt_rdm, f"{idx}_DL", m_dl, pairs_dl, miss_dl, fa_dl, f"{save_dir}/2d_dl_sample_{idx}.png")

            m_cfar, pairs_cfar, miss_cfar, fa_cfar = simple_match(targets, cfar_detections)
            _plot_2d_rdm(mock_ds, gt_rdm, f"{idx}_CFAR", m_cfar, pairs_cfar, miss_cfar, fa_cfar, f"{save_dir}/2d_cfar_sample_{idx}.png")
            
    print(f"Visualizations saved to {save_dir}")

def load_and_evaluate(model_path='radar_timenet_v5.pth', dataset_path = 'data/radar_corrected_test5/radar_dataset.h5', save_dir='data/radar_corrected_test5/evaluation'):
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
    full_dataset = AIRadarDataset(datapath=dataset_path, save_path='data/radar_corrected_test5')
    
    # Use a deterministic split for reproducibility (or just use the whole dataset for eval if preferred)
    # Here we mimic the train split to get the "test" portion, but ideally we should have saved indices.
    # For now, let's just use the last 20% as test to be consistent with training.
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42) # Use seed to try to get same split if possible, though random_split without seed in train_model makes this hard.
        # actually, train_model didn't use a seed. So we can't guarantee same split.
        # Let's just evaluate on the *entire* dataset for validation purposes, 
        # or just random sample. Evaluating on full dataset is safer to see overall performance.
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
    metrics = evaluate(model, test_loader, criterion, DEVICE)
    
    print("\n=== Evaluation Results ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"DL Metrics:   F1={metrics['dl_f1']:.4f}, Precision={metrics['dl_prec']:.4f}, Recall={metrics['dl_rec']:.4f}")
    print(f"CFAR Metrics: F1={metrics['cfar_f1']:.4f}, Precision={metrics['cfar_prec']:.4f}, Recall={metrics['cfar_rec']:.4f}")
    
    # 4. Detailed Visualization
    visualize_results(model, full_dataset, DEVICE, save_dir=save_dir, max_vis_samples=5)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train or Evaluate RadarTimeNet')
    
    # Common Arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode: train or eval')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda or cpu')
    parser.add_argument('--data_path', type=str, default='data/radar_trainv5b/radar_dataset.h5', help='Path to dataset')
    parser.add_argument('--output_path', type=str, default='data/radar_trainv5b/', help='Path to save/load model weights')
    
    # Training Specific Arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(
            output_path=args.output_path,
            model_name='radar_timenet_v5',
            data_path=args.data_path,
            device=args.device,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr
        )
    else:
        # Note: load_and_evaluate might need updates to accept device/data_path if we want it fully configurable too.
        # For now, we pass model_path as output_path from args.
        # Let's quickly update load_and_evaluate signature in a separate step if needed, 
        # but for now the user only asked to put parameters here.
        # However, load_and_evaluate in current file doesn't take data_path/device as args in its definition yet (it hardcodes them inside).
        # I should check load_and_evaluate definition again.
        # It is defined as: def load_and_evaluate(model_path='radar_timenet_v5.pth', save_dir='data/radar_corrected_test5/evaluation'):
        # So we can pass model_path.
        load_and_evaluate(model_path=args.output_path)
