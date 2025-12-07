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

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class CombinedLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_focal=0.5, w_dice=0.5, w_tversky=0.0, w_area=0.0, w_l1=0.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3)
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.w_tversky = w_tversky
        self.w_area = w_area
        self.w_l1 = w_l1
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        with torch.no_grad():
            t = targets.view(-1)
            pos = torch.sum(t)
            neg = t.numel() - pos
            # Use a balanced pos_weight, v5 didn't use this explicitly but had balanced weights.
            # Given v7 dataset is cleaner but potentially sparser, a moderate weight helps.
            pos_w = (neg / (pos + 1e-6)).clamp(min=1.0, max=50.0)
            
        # Standard BCE with pos_weight
        loss_bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_w)
        
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        loss_tversky = self.tversky(inputs, targets)
        
        # Sparsity penalties (Disabled for now)
        # probs = torch.sigmoid(inputs)
        # area_penalty = probs.mean()
        # l1_penalty = torch.mean(torch.abs(probs))
        
        total_loss = (
            self.w_bce * loss_bce +
            self.w_focal * loss_focal +
            self.w_dice * loss_dice +
            self.w_tversky * loss_tversky
        )
        return total_loss

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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w

class UNetRadar2D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetRadar2D, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.se = SEBlock(1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        # Initialize bias for small initial probability (e.g., 0.01)
        # log(0.01 / 0.99) ~= -4.6
        self.outc.conv.bias.data.fill_(-4.6)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.se(x5)
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
        
        # UNet takes 3 channels: Real, Imag, Normalized Log-Magnitude (Matched to v5)
        self.unet = UNetRadar2D(n_channels=3, n_classes=1, bilinear=True)

    def apply_range_fft(self, x):
        # x: [B, Rx, Chirps, Samples, 2]
        # Vectorized complex conversion and FFT for speed
        x_c = torch.view_as_complex(x)  # [B, Rx, Chirps, Samples]
        fft_out = torch.fft.fft(x_c, dim=3)
        return torch.view_as_real(fft_out)

    def apply_doppler_fft(self, x):
        # x: [B, Rx, Chirps, Range, 2]
        x_c = torch.view_as_complex(x)  # [B, Rx, Chirps, Range]
        fft_out = torch.fft.fft(x_c, dim=2)
        fft_out = torch.fft.fftshift(fft_out, dim=2)
        return torch.view_as_real(fft_out)

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
        
        # Debug: Check range of x_log
        # print(f"x_log range: {x_log.min().item():.2f} to {x_log.max().item():.2f}")
        
        # Normalization (Matched to v5 for stability)
        # v5: (x + 50) / 100. 
        # If x_log is in dB, usually noise is around 80-100dB if ADC is large, or -100 if normalized 1.
        # Let's assume raw ADC values -> large numbers.
        # In v5, maybe data was different?
        # Let's use dynamic normalization per sample to be safe?
        # Or just stick to v5 but check values.
        
        x_norm = (x_log + 50.0) / 100.0
        
        # Prepare 3-channel input for UNet
        x_real = x_combined.real
        x_imag = x_combined.imag
        # Normalize real/imag to be roughly in sensible range? 
        # v5 didn't normalize them, but NN usually likes small numbers. 
        # FFT output can be large. Let's keep it raw as v5 did or maybe log?
        # v5 code: x_real = x[..., 0], x_imag = x[..., 1]. It used them directly.
        
        x_in = torch.stack([x_real, x_imag, x_norm], dim=1)
        
        # Resize to standard UNet input size (preserves resolution)
        # if self.target_unet_size is not None and (x_in.shape[2:] != self.target_unet_size):
        #     x_in_resized = F.interpolate(x_in, size=self.target_unet_size, 
        #                                   mode='bilinear', align_corners=False)
        # else:
        x_in_resized = x_in
        
        # 5. Detection
        logits_resized = self.unet(x_in_resized)
        
        # Resize back to original RDM size
        original_size = x_log.shape[-2:]
        if logits_resized.shape[2:] != original_size:
            logits = F.interpolate(logits_resized, size=original_size, 
                                  mode='bilinear', align_corners=False)
        else:
            logits = logits_resized
        
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

def calculate_metrics(preds, targets):
    from sklearn.metrics import precision_score, recall_score, f1_score
    flat_preds = preds.flatten()
    flat_targets = targets.flatten()
    
    # Ensure binary
    flat_preds = (flat_preds > 0.5).astype(int)
    flat_targets = (flat_targets > 0.5).astype(int)
    
    precision = precision_score(flat_targets, flat_preds, zero_division=0)
    recall = recall_score(flat_targets, flat_preds, zero_division=0)
    f1 = f1_score(flat_targets, flat_preds, zero_division=0)
    return precision, recall, f1

def build_curriculum_loader(train_subset, snr_min, batch_size, collate_fn):
    dataset_instance = train_subset.dataset if hasattr(train_subset, 'dataset') else train_subset
    indices = train_subset.indices if hasattr(train_subset, 'indices') else list(range(len(dataset_instance)))
    selected = []
    for idx in indices:
        try:
            info = dataset_instance.target_info[idx]
            snr = info.get('snr_db', None)
            if snr is None:
                sample = dataset_instance[idx]
                snr = sample['target_info'].get('snr_db', 0)
            if snr >= snr_min:
                selected.append(idx)
        except Exception:
            selected.append(idx)
    if len(selected) == 0:
        selected = indices
    subset = torch.utils.data.Subset(dataset_instance, selected)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def visualize_comparison(model, dataset, device, num_samples=3, config_name='default', save_dir='results', dl_threshold=0.5):
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
            
        # DL Detections with min-area CC filter
        from scipy.special import expit
        dl_probs = expit(logits)
        dl_mask = dl_probs > dl_threshold
        from scipy.ndimage import label, center_of_mass
        labeled_array, num_features = label(dl_mask)
        dl_detections = []
        if num_features > 0:
            centers = center_of_mass(dl_mask, labeled_array, range(1, num_features+1))
            # Remove small components
            for comp_id in range(1, num_features+1):
                comp_size = np.sum(labeled_array == comp_id)
                if comp_size < 6:
                    dl_mask[labeled_array == comp_id] = 0
            labeled_array, num_features = label(dl_mask)
            centers = center_of_mass(dl_mask, labeled_array, range(1, num_features+1)) if num_features>0 else []
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
            
            # DL Predictions
            preds = torch.sigmoid(logits) > 0.5
            dl_preds.append(preds.cpu().numpy().flatten())
            dl_targets.append(target_mask.cpu().numpy().flatten())
            
            if 'cfar_detections' in batch:
                # CFAR Predictions
                batch_cfar = batch['cfar_detections']
                batch_targets = batch['target_mask'].numpy()
                
                for i, cfar_list in enumerate(batch_cfar):
                     mask_shape = batch_targets[i].shape
                     cfar_mask = np.zeros(mask_shape, dtype=np.float32)
                     for det in cfar_list:
                         try:
                             r_idx = int(det['range_idx'])
                             d_idx = int(det['doppler_idx'])
                             if 0 <= r_idx < mask_shape[1] and 0 <= d_idx < mask_shape[0]:
                                 cfar_mask[d_idx, r_idx] = 1.0
                         except Exception:
                             pass
                     
                     cfar_preds.append(cfar_mask.flatten())
                     cfar_targets.append(batch_targets[i].flatten())

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
        'loss': total_loss/len(dataloader), 
        'dl_f1': dl_f1, 'dl_prec': dl_prec, 'dl_rec': dl_rec,
        'cfar_f1': cfar_f1, 'cfar_prec': cfar_prec, 'cfar_rec': cfar_rec
    }

def evaluate_detection_metrics(model, dataset, device):
    model.eval()
    dl_tp = dl_fp = dl_fn = 0
    cf_tp = cf_fp = cf_fn = 0
    dataset_instance = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    range_axis = dataset_instance.range_axis
    velocity_axis = dataset_instance.velocity_axis
    import numpy as np
    from scipy.ndimage import label, center_of_mass
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            iq_data = sample['time_domain'].unsqueeze(0).to(device)
            if iq_data.dim() == 4: iq_data = iq_data.unsqueeze(1)
            outputs = model(iq_data)
            logits = outputs['detection_logits'][0, 0].cpu().numpy()
            from scipy.special import expit
            dl_mask = expit(logits) > 0.5
            labeled, nfeat = label(dl_mask)
            dl_detections = []
            if nfeat > 0:
                centers = center_of_mass(dl_mask, labeled, range(1, nfeat+1))
                for c in centers:
                    d_idx, r_idx = c
                    d_idx = max(0, min(int(round(d_idx)), len(velocity_axis)-1))
                    r_idx = max(0, min(int(round(r_idx)), len(range_axis)-1))
                    dl_detections.append({'range_m': range_axis[r_idx], 'velocity_mps': velocity_axis[d_idx], 'range_idx': r_idx, 'doppler_idx': d_idx})
            if 'sample_idx' in sample:
                sidx = sample['sample_idx']
            elif hasattr(dataset, 'indices'):
                sidx = dataset.indices[i]
            else:
                sidx = i
            targets = dataset_instance.target_info[sidx]['targets']
            metrics_dl, _, _, _ = dataset_instance._evaluate_metrics(targets, dl_detections)
            dl_tp += metrics_dl['tp']; dl_fp += metrics_dl['fp']; dl_fn += metrics_dl['fn']
            cfar_list = sample['cfar_detections']
            metrics_cf, _, _, _ = dataset_instance._evaluate_metrics(targets, cfar_list)
            cf_tp += metrics_cf['tp']; cf_fp += metrics_cf['fp']; cf_fn += metrics_cf['fn']
    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0
        return prec, rec, f1
    dl_prec, dl_rec, dl_f1 = prf(dl_tp, dl_fp, dl_fn)
    cf_prec, cf_rec, cf_f1 = prf(cf_tp, cf_fp, cf_fn)
    return {
        'dl_prec': dl_prec, 'dl_rec': dl_rec, 'dl_f1': dl_f1,
        'cfar_prec': cf_prec, 'cfar_rec': cf_rec, 'cfar_f1': cf_f1
    }

def plot_metric_summary(val_res, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ['Precision', 'Recall', 'F1']
    dl = [val_res['dl_prec'], val_res['dl_rec'], val_res['dl_f1']]
    cf = [val_res['cfar_prec'], val_res['cfar_rec'], val_res['cfar_f1']]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(x - w/2, dl, w, label='Model')
    plt.bar(x + w/2, cf, w, label='CFAR')
    plt.xticks(x, labels)
    plt.ylim(0,1)
    plt.ylabel('Score')
    plt.title('Detection Metrics Comparison')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close()

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
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    history = {'train_loss': [], 'val_loss': [], 'dl_f1': [], 'cfar_f1': []}
    
    print(f"Starting training for {epochs} epochs...")
        # Curriculum learning can sometimes hinder initial convergence if not tuned.
        # Let's use the full dataset for now to match v5 behavior and ensure we see all targets.
        # thresholds = [40.0, 35.0, 30.0, 25.0, 20.0]
        # stage_epochs = max(1, epochs // len(thresholds))
        # current_thr = None
        # dilation_epochs = min(3, max(2, epochs // 6))
        
    # Use standard loader
    # train_loader is already defined above with full dataset
    
    for epoch in range(epochs):
        # stage = min(len(thresholds)-1, epoch // stage_epochs)
        # thr = thresholds[stage]
        # if thr != current_thr:
        #     train_loader = build_curriculum_loader(train_dataset, snr_min=thr, batch_size=2, collate_fn=collate_fn_custom)
        #     current_thr = thr
        #     print(f"Curriculum: using SNR >= {thr:.1f} dB for epoch {epoch+1}")
        
        model.train()
        train_loss = 0.0
        
        # Debug: Check max values of predictions
        max_pred_prob = 0.0
        min_x_log = 0.0
        max_x_log = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            iq_data = batch['time_domain'].to(DEVICE)
            if iq_data.dim() == 4: iq_data = iq_data.unsqueeze(1)
            
            tm = batch['target_mask']
            if tm.dim() == 3: target_mask = tm.unsqueeze(1).to(DEVICE)
            elif tm.dim() == 4 and tm.shape[-1] == 1: target_mask = tm.permute(0, 3, 1, 2).to(DEVICE)
            else: target_mask = tm.unsqueeze(1).to(DEVICE)
            
            if epoch < 3:
                # Slight dilation for early epochs
                target_mask = F.max_pool2d(target_mask, kernel_size=3, stride=1, padding=1)
            
            optimizer.zero_grad()
            outputs = model(iq_data)
            logits = outputs['detection_logits']
            
            if logits.shape[-2:] != target_mask.shape[-2:]:
                logits = F.interpolate(logits, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
                
            # Apply weighted BCE through criterion
            loss = criterion(logits, target_mask)
            
            # Monitor probabilities
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                max_pred_prob = max(max_pred_prob, probs.max().item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        # print(f"  Max Predicted Prob: {max_pred_prob:.4f}, RDM dB Range: {min_x_log:.1f} to {max_x_log:.1f}")
        val_res = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_res['loss']:.4f}")
        print(f"  DL Metrics:   F1={val_res['dl_f1']:.4f}, Prec={val_res['dl_prec']:.4f}, Rec={val_res['dl_rec']:.4f}")
        print(f"  CFAR Metrics: F1={val_res['cfar_f1']:.4f}, Prec={val_res['cfar_prec']:.4f}, Rec={val_res['cfar_rec']:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_res['loss'])
        history['dl_f1'].append(val_res['dl_f1'])
        history['cfar_f1'].append(val_res['cfar_f1'])
        
        scheduler.step()
        
        # Save & Visualize
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_latest.pth'))
        visualize_comparison(model, test_dataset, DEVICE, config_name=config_name, save_dir=os.path.join(save_dir, 'viz'), dl_threshold=0.5)
        plot_metric_summary(val_res, save_dir)
        
        # Plot metrics
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['dl_f1'], label='Model F1')
        plt.plot(history['cfar_f1'], label='CFAR F1')
        plt.title('F1 Score')
        plt.legend()
        
        det_res = evaluate_detection_metrics(model, test_dataset, DEVICE)
        plot_metric_summary(det_res, save_dir)
        plt.savefig(os.path.join(save_dir, 'metrics.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    train_model(config_name=args.config, epochs=args.epochs)
