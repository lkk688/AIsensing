#!/usr/bin/env python3
"""
RadarTimeNet Training Script (v7)

This script trains the RadarTimeNet model using the robust dataset generation from AIradar_datasetv7.
It incorporates improvements for high-performance detection:
1. Optimized Model Architecture: RadarTimeNetV7 (Resolution preserving, Normalized inputs)
2. Advanced Loss Functions: Weighted BCE + Focal + Dice (Tuned for extreme sparsity)
3. Comprehensive Visualization: Side-by-side comparison with CFAR using dataset's plotting tools
4. Robust Training Loop: Metrics plotting and validation

Usage:
    python3 AIradar_trainv7.py --mode train --config config1 --epochs 20
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
import math

# Add AIRadar directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from AIradar_datasetv7 (The robust version)
from AIRadar.AIradar_datasetv7 import AIRadarDataset, _plot_2d_rdm, _plot_3d_rdm, VIEW_RANGE_LIMITS, VIEW_VELOCITY_LIMITS

# ==========================================
# Loss Functions (Robust to Class Imbalance)
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
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
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_focal=1.0, w_dice=1.0):
        super(CombinedLoss, self).__init__()
        # Reset pos_weight to reasonable value as we use "Start High" init
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000.0]).to('cuda' if torch.cuda.is_available() else 'cpu'))
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_dice = w_dice

    def forward(self, inputs, targets):
        loss_bce = self.bce(inputs, targets)
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        return self.w_bce * loss_bce + self.w_focal * loss_focal + self.w_dice * loss_dice

# ==========================================
# U-Net Architecture
# ==========================================

class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetRadar2D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetRadar2D, self).__init__()
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
        
        # Initialize bias to positive value to start with High Recall (detect everything)
        self.outc.conv.bias.data.fill_(2.0)

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

# ==========================================
# RadarTimeNet Model
# ==========================================

class RadarTimeNetV7(nn.Module):
    def __init__(self, num_rx=1, num_chirps=None, samples_per_chirp=None, 
                 fft_size=None, out_range_bins=None, out_doppler_bins=None,
                 target_unet_size=(128, 2048)): # Preserves range resolution
        super().__init__()
        self.num_rx = num_rx
        self.target_unet_size = target_unet_size
        
        # UNet takes 1 channel: Normalized Log-Magnitude
        self.unet = UNetRadar2D(n_channels=1, n_classes=1, bilinear=True)

    def apply_range_fft(self, x):
        # x: [B, Rx, Chirps, Samples, 2]
        B, Rx, Chirps, Samples, _ = x.shape
        output_list = []
        for b in range(B):
            x_b = x[b] 
            complex_input_b = torch.view_as_complex(x_b) # [Rx, Chirps, Samples]
            # FFT on Samples dim (2)
            fft_out_b = torch.fft.fft(complex_input_b, dim=2)
            output_list.append(torch.view_as_real(fft_out_b))
        return torch.stack(output_list, dim=0)

    def apply_doppler_fft(self, x):
        # x: [B, Rx, Chirps, Range, 2]
        B, Rx, Chirps, Range, _ = x.shape
        output_list = []
        for b in range(B):
            x_b = x[b]
            complex_input_b = torch.view_as_complex(x_b) # [Rx, Chirps, Range]
            # FFT on Chirps dim (1) + Shift
            fft_out_b = torch.fft.fft(complex_input_b, dim=1)
            fft_out_b = torch.fft.fftshift(fft_out_b, dim=1)
            output_list.append(torch.view_as_real(fft_out_b))
        return torch.stack(output_list, dim=0)

    def forward(self, x):
        # x: [B, Rx, Chirps, Samples, 2]
        
        # 1. Range Processing (FFT) - Direct on raw IQ
        x = self.apply_range_fft(x)
        
        # 2. Doppler Processing (FFT)
        x = self.apply_doppler_fft(x)
        
        # 3. Coherent Integration (Sum over Rx)
        x_complex = torch.view_as_complex(x) # [B, Rx, Doppler, Range]
        x_combined = torch.mean(x_complex, dim=1) # [B, Doppler, Range]
        
        # 4. Log-Magnitude Calculation & Normalization
        x_mag = torch.abs(x_combined)
        x_log = 20 * torch.log10(x_mag + 1e-6)
        
        # Normalize: Assume floor ~-100dB, peak ~0dB. Scale to ~[0, 1]
        x_norm = (x_log + 100.0) / 100.0
        
        # Prepare for UNet: [B, 1, Doppler, Range]
        unet_input = x_norm.unsqueeze(1)
        
        # Resize to standard UNet input size (preserves resolution)
        unet_input_resized = F.interpolate(unet_input, size=self.target_unet_size, 
                                          mode='bilinear', align_corners=False)
        
        # 5. Detection
        logits_resized = self.unet(unet_input_resized)
        
        # Resize back to original RDM size
        original_size = x_log.shape[-2:]
        logits = F.interpolate(logits_resized, size=original_size, 
                              mode='bilinear', align_corners=False)
        
        return {
            'detection_logits': logits,
            'rd_map_db': x_log
        }

# ==========================================
# Helpers & Training Loop
# ==========================================

def collate_fn_custom(batch):
    return {
        'time_domain': torch.stack([item['time_domain'] for item in batch]),
        'target_mask': torch.stack([item['target_mask'] for item in batch]),
        'cfar_detections': [item['cfar_detections'] for item in batch],
        'target_info': [item['target_info'] for item in batch]
    }

def calculate_metrics(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).astype(int)
    targets_bin = (targets > 0.5).astype(int)
    tp = np.sum((preds_bin == 1) & (targets_bin == 1))
    fp = np.sum((preds_bin == 1) & (targets_bin == 0))
    fn = np.sum((preds_bin == 0) & (targets_bin == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def visualize_comparison(model, dataset, device, num_samples=3, config_name='default', save_dir='results'):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    os.makedirs(save_dir, exist_ok=True)
    
    # Retrieve dataset axes
    dataset_instance = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    range_axis = dataset_instance.range_axis
    velocity_axis = dataset_instance.velocity_axis
    
    for idx in indices:
        sample = dataset[idx]
        iq_data = sample['time_domain'].unsqueeze(0).to(device)
        if iq_data.dim() == 4: iq_data = iq_data.unsqueeze(1)
        
        # Get sample index
        if 'sample_idx' in sample: sample_idx = sample['sample_idx']
        elif hasattr(dataset, 'indices'): sample_idx = dataset.indices[idx]
        else: sample_idx = idx
            
        targets = dataset_instance.target_info[sample_idx]['targets']
        
        with torch.no_grad():
            outputs = model(iq_data)
            rd_map_db = outputs['rd_map_db'][0].cpu().numpy()
            logits = outputs['detection_logits'][0, 0].cpu().numpy()
            
        # DL Detections
        dl_mask = (1 / (1 + np.exp(-logits))) > 0.5
        from scipy.ndimage import label, center_of_mass
        labeled_array, num_features = label(dl_mask)
        dl_detections = []
        if num_features > 0:
            centers = center_of_mass(dl_mask, labeled_array, range(1, num_features+1))
            for center in centers:
                d_idx, r_idx = center
                d_idx_int = max(0, min(int(round(d_idx)), len(velocity_axis)-1))
                r_idx_int = max(0, min(int(round(r_idx)), len(range_axis)-1))
                dl_detections.append({
                    'range_m': range_axis[r_idx_int],
                    'velocity_mps': velocity_axis[d_idx_int],
                    'range_idx': r_idx_int,
                    'doppler_idx': d_idx_int
                })
        
        # CFAR Detections
        cfar_detections = sample['cfar_detections']
        
        rdm_norm = rd_map_db - np.max(rd_map_db)
        
        # Plot DL
        metrics_dl, matched_dl, unmatched_t_dl, unmatched_d_dl = dataset_instance._evaluate_metrics(targets, dl_detections)
        _plot_2d_rdm(dataset_instance, rdm_norm, idx, metrics_dl, matched_dl, unmatched_t_dl, unmatched_d_dl, 
                     os.path.join(save_dir, f"sample_{idx}_dl_2d.png"))
        _plot_3d_rdm(dataset_instance, rdm_norm, idx, targets, dl_detections, os.path.join(save_dir, f"sample_{idx}_dl_3d.png"))
        
        # Plot CFAR
        metrics_cfar, matched_cfar, unmatched_t_cfar, unmatched_d_cfar = dataset_instance._evaluate_metrics(targets, cfar_detections)
        _plot_2d_rdm(dataset_instance, rdm_norm, idx, metrics_cfar, matched_cfar, unmatched_t_cfar, unmatched_d_cfar, 
                     os.path.join(save_dir, f"sample_{idx}_cfar_2d.png"))
        _plot_3d_rdm(dataset_instance, rdm_norm, idx, targets, cfar_detections, os.path.join(save_dir, f"sample_{idx}_cfar_3d.png"))

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    dl_preds, dl_targets = [], []
    cfar_preds, cfar_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            iq_data = batch['time_domain'].to(device)
            if iq_data.dim() == 4: iq_data = iq_data.unsqueeze(1)
            
            tm = batch['target_mask']
            if tm.dim() == 3: target_mask = tm.unsqueeze(1).to(device)
            elif tm.dim() == 4 and tm.shape[-1] == 1: target_mask = tm.permute(0, 3, 1, 2).to(device)
            else: target_mask = tm.unsqueeze(1).to(device)
            
            outputs = model(iq_data)
            logits = outputs['detection_logits']
            if logits.shape[-2:] != target_mask.shape[-2:]:
                logits = F.interpolate(logits, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
                
            loss = criterion(logits, target_mask)
            total_loss += loss.item()
            
            dl_probs = torch.sigmoid(logits)
            dl_mask = (dl_probs > 0.5).float().cpu().numpy()
            dl_preds.append(dl_mask.flatten())
            dl_targets.append(target_mask.cpu().numpy().flatten())
            
            if 'cfar_detections' in batch:
                # Calculate CFAR metrics for comparison log
                cfar_preds_batch, cfar_targets_batch = [], []
                batch_cfar = batch['cfar_detections']
                batch_targets = batch['target_mask'].numpy()
                
                for i, cfar_list in enumerate(batch_cfar):
                     # Create mask from CFAR detections
                     mask_shape = batch_targets[i].shape
                     cfar_mask = np.zeros(mask_shape, dtype=np.float32)
                     for det in cfar_list:
                         r_idx = int(det['range_idx'])
                         d_idx = int(det['doppler_idx'])
                         if 0 <= r_idx < mask_shape[1] and 0 <= d_idx < mask_shape[0]:
                             cfar_mask[d_idx, r_idx] = 1.0
                     
                     cfar_preds_batch.append(cfar_mask.flatten())
                     cfar_targets_batch.append(batch_targets[i].flatten())
                
                cfar_preds.extend(cfar_preds_batch)
                cfar_targets.extend(cfar_targets_batch)

    dl_all_preds = np.concatenate(dl_preds)
    dl_all_targets = np.concatenate(dl_targets)
    dl_prec, dl_rec, dl_f1 = calculate_metrics(dl_all_preds, dl_all_targets)
    
    if cfar_preds:
        cfar_all_preds = np.concatenate(cfar_preds)
        cfar_all_targets = np.concatenate(cfar_targets)
        cfar_prec, cfar_rec, cfar_f1 = calculate_metrics(cfar_all_preds, cfar_all_targets)
    else:
        cfar_prec, cfar_rec, cfar_f1 = 0, 0, 0
    
    return {
        'loss': total_loss/len(dataloader), 
        'dl_f1': dl_f1, 'dl_prec': dl_prec, 'dl_rec': dl_rec,
        'cfar_f1': cfar_f1, 'cfar_prec': cfar_prec, 'cfar_rec': cfar_rec
    }

def train_model(config_name='config1', epochs=20, save_base_dir='results_v7'):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    save_dir = os.path.join(save_base_dir, f'radar_{config_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load Dataset (Generate if needed)
    print("Initializing Dataset...")
    # We generate new data to ensure V7 standards
    dataset = AIRadarDataset(
        num_samples=100, # Can increase for real training
        config_name=config_name,
        save_path=save_dir,
        drawfig=False,
        apply_realistic_effects=True,
        clutter_intensity=0.1
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_custom)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_custom)
    
    # Model
    sample_0 = dataset[0]
    n_chirps, n_samples, _ = sample_0['time_domain'].shape if sample_0['time_domain'].dim()==3 else sample_0['time_domain'].shape[1:]
    n_doppler, n_range = sample_0['range_doppler_map'].shape
    
    print(f"Model Config: Input(Rx=1, C={n_chirps}, S={n_samples}) -> Output(D={n_doppler}, R={n_range})")
    
    model = RadarTimeNetV7(
        num_rx=1,
        num_chirps=n_chirps,
        samples_per_chirp=n_samples,
        out_range_bins=n_range,
        out_doppler_bins=n_doppler,
        target_unet_size=(128, 2048) # Optimized for 2000 range bins
    ).to(DEVICE)
    
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    history = {'train_loss': [], 'val_loss': [], 'dl_f1': []}
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            iq_data = batch['time_domain'].to(DEVICE)
            if iq_data.dim() == 4: iq_data = iq_data.unsqueeze(1)
            
            tm = batch['target_mask']
            if tm.dim() == 3: target_mask = tm.unsqueeze(1).to(DEVICE)
            elif tm.dim() == 4 and tm.shape[-1] == 1: target_mask = tm.permute(0, 3, 1, 2).to(DEVICE)
            else: target_mask = tm.unsqueeze(1).to(DEVICE)
            
            # Dilate target mask to make training easier
            # Using MaxPool2d as a morphological dilation (3x3 kernel)
            target_mask = F.max_pool2d(target_mask, kernel_size=3, stride=1, padding=1)
            
            optimizer.zero_grad()
            outputs = model(iq_data)
            logits = outputs['detection_logits']
            
            if logits.shape[-2:] != target_mask.shape[-2:]:
                logits = F.interpolate(logits, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
                
            loss = criterion(logits, target_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        val_res = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_res['loss']:.4f}")
        print(f"  DL Metrics:   F1={val_res['dl_f1']:.4f}, Prec={val_res['dl_prec']:.4f}, Rec={val_res['dl_rec']:.4f}")
        print(f"  CFAR Metrics: F1={val_res['cfar_f1']:.4f}, Prec={val_res['cfar_prec']:.4f}, Rec={val_res['cfar_rec']:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_res['loss'])
        history['dl_f1'].append(val_res['dl_f1'])
        
        scheduler.step(val_res['loss'])
        
        # Save & Visualize
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_latest.pth'))
        visualize_comparison(model, test_dataset, DEVICE, config_name=config_name, save_dir=os.path.join(save_dir, 'viz'))
        
        # Plot metrics
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['dl_f1'], label='DL F1')
        plt.title('F1 Score')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'metrics.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    train_model(config_name=args.config, epochs=args.epochs)
