#!/usr/bin/env python3
"""
RadarTimeNet Training Script with Simulation Data

This script demonstrates training the RadarTimeNet model using simulated FMCW radar data
and compares its performance against traditional signal processing methods.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import signal
import time
import math
from typing import Tuple, List, Dict
import os
from AIradar_datasetv5 import AIRadarDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pickle

# Try to import torch.fft for modern PyTorch versions
try:
    import torch.fft
    HAS_TORCH_FFT = True
except ImportError:
    HAS_TORCH_FFT = False

# Define placeholder classes for required modules
class LearnableFFT(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.real = nn.Parameter(torch.randn(input_size, output_size))
        self.imag = nn.Parameter(torch.randn(input_size, output_size))
    
    def forward(self, real_part, imag_part):
        real_out = torch.matmul(real_part, self.real) - torch.matmul(imag_part, self.imag)
        imag_out = torch.matmul(real_part, self.imag) + torch.matmul(imag_part, self.real)
        return torch.stack([real_out, imag_out], dim=-1)

# === RadarTimeNet: processes time-domain IQ signals for radar detection ===
class RadarTimeNet(nn.Module):
    """
    Deep learning model for processing time-domain IQ signals to extract range-Doppler maps
    and perform object detection.
    
    The processing pipeline includes:
    1. Time-domain preprocessing with 3D convolutions
    2. Demodulation (mixing) with reference signal
    3. Range FFT processing
    4. Doppler FFT processing
    5. Post-processing with 2D convolutions for RD map estimation
    6. Object detection head for target detection
    """
    def __init__(self, num_rx=1, num_chirps=128, samples_per_chirp=1024, 
                 out_doppler_bins=128, out_range_bins=4096, use_learnable_fft=False):
        """
        Initialize the RadarTimeNet module.
        
        Args:
            num_rx: Number of receive antennas
            num_chirps: Number of chirps in the input signal
            samples_per_chirp: Number of samples per chirp
            out_doppler_bins: Number of Doppler bins in the output
            out_range_bins: Number of range bins in the output
            use_learnable_fft: Whether to use learnable FFT or standard FFT
        """
        super().__init__()
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.out_doppler_bins = out_doppler_bins
        self.out_range_bins = out_range_bins
        self.use_learnable_fft = use_learnable_fft
        
        # === Time-domain preprocessing ===
        # Input shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        # Output shape: [B, 32, num_rx, num_chirps, samples_per_chirp]
        self.time_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # === Demodulation module (mixing with reference) ===
        # Learnable complex multiplication for demodulation
        # Implements y = x * conj(ref) where x is the received signal and ref is the reference signal
        self.demod_weights = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))
        
        # === Range FFT processing ===
        # Process each chirp with range FFT
        if use_learnable_fft:
            self.range_fft = LearnableFFT(samples_per_chirp, out_range_bins)
        else:
            self.range_fft = None
            
        # === Doppler FFT processing ===
        # Process each range bin with Doppler FFT
        if use_learnable_fft:
            self.doppler_fft = LearnableFFT(num_chirps, out_doppler_bins)
        else:
            self.doppler_fft = None
            
        # === Post-processing for range-Doppler map ===
        # Process the range-Doppler map with 2D convolutions
        self.rd_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # === Object detection head ===
        self.detection_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
            # No sigmoid - BCEWithLogitsLoss applies it internally
        )
        
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )
        
        # Initialize with FFT-like weights
        self._init_fft_weights()
        
    def _init_fft_weights(self):
        """
        Initialize the learnable FFT weights to mimic the standard FFT.
        This helps the model converge faster during training.
        """
        if self.use_learnable_fft and self.range_fft is not None:
            # Initialize range FFT weights
            N = self.samples_per_chirp
            for k in range(self.out_range_bins):
                for n in range(N):
                    angle = -2 * np.pi * k * n / N
                    self.range_fft.real.data[n, k] = np.cos(angle) / np.sqrt(N)
                    self.range_fft.imag.data[n, k] = np.sin(angle) / np.sqrt(N)
            
            # Initialize Doppler FFT weights
            N = self.num_chirps
            for k in range(self.out_doppler_bins):
                for n in range(N):
                    angle = -2 * np.pi * k * n / N
                    self.doppler_fft.real.data[n, k] = np.cos(angle) / np.sqrt(N)
                    self.doppler_fft.imag.data[n, k] = np.sin(angle) / np.sqrt(N)
        
        # Initialize demodulation weights for complex conjugate multiplication
        self.demod_weights.data = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)

    def complex_multiply(self, x, y):
        """
        Perform complex multiplication between two tensors.
        
        Args:
            x: First tensor with shape [..., 2] (real, imag)
            y: Second tensor with shape [..., 2] (real, imag)
            
        Returns:
            Complex product with shape [..., 2]
        """
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
        imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        return torch.stack([real, imag], dim=-1)
    
    def complex_conjugate(self, x):
        """
        Compute the complex conjugate of a tensor.
        
        Args:
            x: Input tensor with shape [..., 2] (real, imag)
            
        Returns:
            Complex conjugate with shape [..., 2]
        """
        return torch.stack([x[..., 0], -x[..., 1]], dim=-1)
    
    def demodulate(self, rx_signal, ref_signal=None):
        """
        Demodulate the received signal by mixing with the reference signal.
        
        Args:
            rx_signal: Received signal with shape [..., 2]
            ref_signal: Reference signal with shape [..., 2], if None, use learnable demodulation
            
        Returns:
            Demodulated signal with shape [..., 2]
        """
        if ref_signal is not None:
            # Use provided reference signal
            # y = x * conj(ref)
            return self.complex_multiply(rx_signal, self.complex_conjugate(ref_signal))
        else:
            # Use learnable demodulation
            # Apply the demodulation weights to the input
            # [B, num_rx, num_chirps, samples_per_chirp, 2]
            batch_size = rx_signal.shape[0]
            rx_signal_flat = rx_signal.reshape(-1, 2)  # Flatten all dimensions except the last
            demod_signal_flat = torch.matmul(rx_signal_flat, self.demod_weights)
            return demod_signal_flat.reshape(*rx_signal.shape)
    
    def apply_range_fft(self, x):
        """
        Apply range FFT to the input signal.
        
        The range FFT converts the time-domain signal to the range domain.
        For FMCW radar, the frequency after mixing is proportional to the target range.
        
        Mathematical formulation:
        Range FFT: X[k] = ∑_{n=0}^{N-1} x[n] * e^{-j2πkn/N}
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, samples_per_chirp, 2]
            
        Returns:
            Range spectrum with shape [B, num_rx, num_chirps, out_range_bins, 2]
        """
        batch_size, num_rx, num_chirps, samples_per_chirp, _ = x.shape
        
        # Reshape for processing
        x_reshaped = x.reshape(batch_size * num_rx * num_chirps, samples_per_chirp, 2)
        real_part, imag_part = x_reshaped[..., 0], x_reshaped[..., 1]
        
        if self.use_learnable_fft and self.range_fft is not None:
            # Use learnable FFT
            range_spectrum = self.range_fft(real_part, imag_part)
        else:
            # Use standard FFT - compatible with different PyTorch versions
            if hasattr(torch, 'complex'):
                complex_input = torch.complex(real_part, imag_part)
                if HAS_TORCH_FFT:
                    # Use torch.fft.fft directly - it's available in modern PyTorch
                    import torch.fft as fft_module
                    complex_output = fft_module.fft(complex_input, n=self.out_range_bins, dim=1)
                else:
                    # Fallback for older PyTorch versions
                    complex_output = torch.rfft(torch.stack([real_part, imag_part], dim=-1), 1, onesided=False)
                    complex_output = torch.view_as_complex(complex_output)
                range_spectrum = torch.stack([complex_output.real, complex_output.imag], dim=-1)
            else:
                # Fallback for very old PyTorch versions
                complex_input = torch.stack([real_part, imag_part], dim=-1)
                complex_output = torch.rfft(complex_input, 1, onesided=False)
                range_spectrum = complex_output
        
        # Reshape back to original dimensions
        return range_spectrum.reshape(batch_size, num_rx, num_chirps, self.out_range_bins, 2)
    
    def apply_doppler_fft(self, x):
        """
        Apply Doppler FFT to the input signal.
        
        The Doppler FFT converts the chirp-domain signal to the Doppler domain.
        For FMCW radar, the phase change across chirps is proportional to the target velocity.
        
        Mathematical formulation:
        Doppler FFT: X[k] = ∑_{n=0}^{N-1} x[n] * e^{-j2πkn/N}
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, out_range_bins, 2]
            
        Returns:
            Range-Doppler map with shape [B, num_rx, out_doppler_bins, out_range_bins, 2]
        """
        batch_size, num_rx, num_chirps, range_bins, _ = x.shape
        
        # Transpose to put chirps in the right dimension for FFT
        x_transposed = x.permute(0, 1, 3, 2, 4)  # [B, num_rx, range_bins, num_chirps, 2]
        
        # Reshape for processing
        x_reshaped = x_transposed.reshape(batch_size * num_rx * range_bins, num_chirps, 2)
        real_part, imag_part = x_reshaped[..., 0], x_reshaped[..., 1]
        
        if self.use_learnable_fft and self.doppler_fft is not None:
            # Use learnable FFT
            doppler_spectrum = self.doppler_fft(real_part, imag_part)
        else:
            # Use standard FFT - compatible with different PyTorch versions
            if hasattr(torch, 'complex'):
                complex_input = torch.complex(real_part, imag_part)
                if HAS_TORCH_FFT:
                    # Use torch.fft.fft directly - it's available in modern PyTorch
                    import torch.fft as fft_module
                    complex_output = fft_module.fft(complex_input, n=self.out_doppler_bins, dim=1)
                    # Apply FFT shift to center the Doppler spectrum
                    complex_output = fft_module.fftshift(complex_output, dim=1)
                else:
                    # Fallback for older PyTorch versions
                    complex_output = torch.rfft(torch.stack([real_part, imag_part], dim=-1), 1, onesided=False)
                    complex_output = torch.view_as_complex(complex_output)
                    # Manual fftshift implementation
                    n = complex_output.shape[1]
                    indices = torch.cat([torch.arange(n//2, n), torch.arange(0, n//2)])
                    complex_output = complex_output[:, indices]
                doppler_spectrum = torch.stack([complex_output.real, complex_output.imag], dim=-1)
            else:
                # Fallback for very old PyTorch versions
                complex_input = torch.stack([real_part, imag_part], dim=-1)
                complex_output = torch.rfft(complex_input, 1, onesided=False)
                # Manual fftshift
                n = complex_output.shape[1]
                indices = torch.cat([torch.arange(n//2, n), torch.arange(0, n//2)])
                doppler_spectrum = complex_output[:, indices]
        
        # Reshape back to original dimensions
        return doppler_spectrum.reshape(batch_size, num_rx, range_bins, self.out_doppler_bins, 2).permute(0, 1, 3, 2, 4)
    


    def forward(self, x, ref_signal=None, return_intermediate=False):
        """
        Forward pass of the RadarTimeNet module.
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, samples_per_chirp, 2]
            ref_signal: Optional reference signal for demodulation
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
            - 'rd_map': Range-Doppler map [B, 1, out_doppler_bins, out_range_bins]
            - 'detection_map': Object detection probability map [B, 1, out_doppler_bins, out_range_bins]
        """
        # Input shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        batch_size = x.shape[0]
        intermediate_results = {}
        
        # === Step 1: Time-domain preprocessing ===
        # Permute to [B, 2, num_rx, num_chirps, samples_per_chirp]
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply 3D convolution for time-domain preprocessing
        # Output shape: [B, 32, num_rx, num_chirps, samples_per_chirp]
        x = self.time_conv(x)
        
        if return_intermediate:
            intermediate_results['time_processed'] = x.clone()
        
        # Permute back to [B, num_rx, num_chirps, samples_per_chirp, 2]
        # Take only the first 2 channels and permute to correct shape
        x = x[:, :2].permute(0, 2, 3, 4, 1)
        
        # === Step 2: Demodulation (mixing with reference) ===
        # Output shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        x = self.demodulate(x, ref_signal)
        
        # === Step 3: Range FFT processing ===
        # Output shape: [B, num_rx, num_chirps, out_range_bins, 2]
        x = self.apply_range_fft(x)
        
        if return_intermediate:
            intermediate_results['range_fft'] = x.clone()
        
        # === Step 4: Doppler FFT processing ===
        # Output shape: [B, num_rx, out_doppler_bins, out_range_bins, 2]
        x = self.apply_doppler_fft(x)
        
        if return_intermediate:
            intermediate_results['doppler_fft'] = x.clone()
        
        # === Step 5: Post-processing ===
        # Average across receive antennas
        # Output shape: [B, out_doppler_bins, out_range_bins, 2]
        x = x.mean(dim=1)
        
        # Permute to [B, 2, out_doppler_bins, out_range_bins] for 2D convolution
        x = x.permute(0, 3, 1, 2)
        
        # Apply 2D convolution for post-processing
        # Output shape: [B, 64, out_doppler_bins, out_range_bins]
        rd_features = self.rd_conv(x)
        
        # Final output layer for range-Doppler map
        # Output shape: [B, 1, out_doppler_bins, out_range_bins]
        rd_map = self.output(rd_features)
        
        # Object detection head (raw logits for BCEWithLogitsLoss)
        # Output shape: [B, 1, out_doppler_bins, out_range_bins]
        detection_map = self.detection_conv(rd_features)
        
        results = {
            'rd_map': rd_map,
            'detection_map': detection_map
        }
        
        if return_intermediate:
            results['intermediate'] = intermediate_results
            
        return results


# === Custom Dataset for RadarTimeNet Training ===
class RadarDataset(Dataset):
    """
    Custom dataset class for RadarTimeNet training using AIRadarDataset.
    """
    def __init__(self, radar_data, targets, rd_maps):
        self.radar_data = radar_data  # Raw IQ data
        self.targets = targets        # Target detection labels
        self.rd_maps = rd_maps       # Ground truth RD maps
        
    def __len__(self):
        return len(self.radar_data)
    
    def __getitem__(self, idx):
        return {
            'iq_data': torch.FloatTensor(self.radar_data[idx]),
            'rd_map': torch.FloatTensor(self.rd_maps[idx]),
            'targets': torch.FloatTensor(self.targets[idx])
        }


# === Training Functions ===
def train_epoch(model, dataloader, optimizer, criterion_rd, criterion_det, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    rd_loss_total = 0.0
    det_loss_total = 0.0
    
    for batch in dataloader:
        iq_data = batch['iq_data'].to(device)
        rd_map_gt = batch['rd_map'].to(device)
        targets_gt = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(iq_data)
        rd_map_pred = outputs['rd_map']
        detection_pred = outputs['detection_map']
        
        # Calculate losses
        rd_loss = criterion_rd(rd_map_pred, rd_map_gt)
        # Squeeze channel dimension from detection prediction to match targets
        detection_pred_squeezed = detection_pred.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        det_loss = criterion_det(detection_pred_squeezed, targets_gt)
        total_loss_batch = rd_loss + det_loss
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        rd_loss_total += rd_loss.item()
        det_loss_total += det_loss.item()
    
    return total_loss / len(dataloader), rd_loss_total / len(dataloader), det_loss_total / len(dataloader)


def evaluate_model(model, dataloader, criterion_rd, criterion_det, device):
    """
    Evaluate the model on validation/test data.
    """
    model.eval()
    total_loss = 0.0
    rd_loss_total = 0.0
    det_loss_total = 0.0
    
    all_rd_preds = []
    all_rd_targets = []
    all_det_preds = []
    all_det_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            iq_data = batch['iq_data'].to(device)
            rd_map_gt = batch['rd_map'].to(device)
            targets_gt = batch['targets'].to(device)
            
            # Forward pass
            outputs = model(iq_data)
            rd_map_pred = outputs['rd_map']
            detection_pred = outputs['detection_map']
            
            # Calculate losses
            rd_loss = criterion_rd(rd_map_pred, rd_map_gt)
            # Squeeze channel dimension from detection prediction to match targets
            detection_pred_squeezed = detection_pred.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            # BCEWithLogitsLoss expects raw logits, not sigmoid output
            det_loss = criterion_det(detection_pred_squeezed, targets_gt)
            total_loss += (rd_loss + det_loss).item()
            rd_loss_total += rd_loss.item()
            det_loss_total += det_loss.item()
            
            # Store predictions for metrics calculation
            all_rd_preds.append(rd_map_pred.cpu())
            all_rd_targets.append(rd_map_gt.cpu())
            # Use same adaptive threshold as in comparison
            detection_probs = torch.sigmoid(detection_pred_squeezed)
            all_det_preds.append((detection_probs > 0.95).float().cpu())
            all_det_targets.append(targets_gt.cpu())
    
    # Calculate metrics
    rd_preds = torch.cat(all_rd_preds, dim=0)
    rd_targets = torch.cat(all_rd_targets, dim=0)
    det_preds = torch.cat(all_det_preds, dim=0)
    det_targets = torch.cat(all_det_targets, dim=0)
    
    # RD map accuracy (MSE)
    rd_mse = torch.mean((rd_preds - rd_targets) ** 2).item()
    
    # Detection metrics
    det_preds_flat = det_preds.flatten().numpy()
    det_targets_flat = det_targets.flatten().numpy()
    
    precision = precision_score(det_targets_flat, det_preds_flat, zero_division=0)
    recall = recall_score(det_targets_flat, det_preds_flat, zero_division=0)
    f1 = f1_score(det_targets_flat, det_preds_flat, zero_division=0)
    
    return {
        'total_loss': total_loss / len(dataloader),
        'rd_loss': rd_loss_total / len(dataloader),
        'det_loss': det_loss_total / len(dataloader),
        'rd_mse': rd_mse,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def compare_with_cfar(model, ai_radar_dataset, device, num_samples=100):
    """
    Compare RadarTimeNet performance with traditional CFAR detection.
    """
    print("\n=== Comparing RadarTimeNet with Traditional CFAR ===")
    
    model.eval()
    nn_detections = []
    cfar_detections = []
    ground_truth = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(ai_radar_dataset))):
            sample = ai_radar_dataset[i]
            
            # Prepare input data
            time_domain = sample['time_domain']
            # time_domain shape: [num_chirps, samples_per_chirp, 2] -> [1, num_chirps, samples_per_chirp, 2]
            iq_data = time_domain[np.newaxis, :, :, :]  # [1, num_chirps, samples_per_chirp, 2]
            iq_data = torch.FloatTensor(iq_data).unsqueeze(0).to(device)
            
            # Neural network prediction
            outputs = model(iq_data)
            # detection_map shape: [1, 1, doppler_bins, range_bins] -> squeeze to [doppler_bins, range_bins]
            detection_logits = outputs['detection_map']
            detection_probs = torch.sigmoid(detection_logits).cpu().numpy().squeeze()
            
            # Use adaptive threshold based on ground truth sparsity
            # Since ground truth has ~0.003% positive pixels, use higher threshold
            detection_threshold = 0.95  # Much higher threshold for sparse targets
            detection_pred = (detection_probs > detection_threshold).astype(float)
            nn_detections.append(detection_pred)
            
            # CFAR detection (traditional method)
            rdm = sample['range_doppler_map']
            cfar_results = ai_radar_dataset.cfar_detection(rdm)
            
            # Convert CFAR results to detection map
            cfar_map = np.zeros_like(rdm)
            for detection in cfar_results:
                r_idx = detection['range_idx']
                d_idx = detection['doppler_idx']
                # RDM shape is [doppler_bins, range_bins], so index as [d_idx, r_idx]
                if 0 <= d_idx < cfar_map.shape[0] and 0 <= r_idx < cfar_map.shape[1]:
                    cfar_map[d_idx, r_idx] = 1
            cfar_detections.append(cfar_map)
            
            # Ground truth (use target mask)
            gt_map = sample['target_mask']
            ground_truth.append(gt_map)
    
    # Calculate metrics for both methods
    nn_detections = np.array(nn_detections).flatten()
    cfar_detections = np.array(cfar_detections).flatten()
    ground_truth = np.array(ground_truth).flatten()
    
    # Debug information
    print(f"\nDebug Information:")
    print(f"  Ground truth positives: {np.sum(ground_truth)} / {len(ground_truth)} ({100*np.sum(ground_truth)/len(ground_truth):.3f}%)")
    print(f"  NN detections: {np.sum(nn_detections)} / {len(nn_detections)} ({100*np.sum(nn_detections)/len(nn_detections):.3f}%)")
    print(f"  CFAR detections: {np.sum(cfar_detections)} / {len(cfar_detections)} ({100*np.sum(cfar_detections)/len(cfar_detections):.3f}%)")
    
    # Neural Network metrics
    nn_precision = precision_score(ground_truth, nn_detections, zero_division=0)
    nn_recall = recall_score(ground_truth, nn_detections, zero_division=0)
    nn_f1 = f1_score(ground_truth, nn_detections, zero_division=0)
    
    # CFAR metrics
    cfar_precision = precision_score(ground_truth, cfar_detections, zero_division=0)
    cfar_recall = recall_score(ground_truth, cfar_detections, zero_division=0)
    cfar_f1 = f1_score(ground_truth, cfar_detections, zero_division=0)
    
    print(f"\nNeural Network Performance:")
    print(f"  Precision: {nn_precision:.4f}")
    print(f"  Recall: {nn_recall:.4f}")
    print(f"  F1-Score: {nn_f1:.4f}")
    
    print(f"\nTraditional CFAR Performance:")
    print(f"  Precision: {cfar_precision:.4f}")
    print(f"  Recall: {cfar_recall:.4f}")
    print(f"  F1-Score: {cfar_f1:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Precision: {((nn_precision - cfar_precision) / max(cfar_precision, 1e-6)) * 100:.2f}%")
    print(f"  Recall: {((nn_recall - cfar_recall) / max(cfar_recall, 1e-6)) * 100:.2f}%")
    print(f"  F1-Score: {((nn_f1 - cfar_f1) / max(cfar_f1, 1e-6)) * 100:.2f}%")
    
    return {
        'nn_metrics': {'precision': nn_precision, 'recall': nn_recall, 'f1': nn_f1},
        'cfar_metrics': {'precision': cfar_precision, 'recall': cfar_recall, 'f1': cfar_f1}
    }


# === Main Training Script ===
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate training data using AIRadarDataset
    print("Generating training data...")
    ai_radar = AIRadarDataset(
        num_samples=1000,  # Reduced for faster testing
        N_chirps=128,
        R_max=200.0,
        max_targets=3,
        save_path='outputs/radar_training_data_1000',
        #datapath='output/radar_training_data_1000/radar_dataset.h5'
    )
    
    # Generate dataset
    #ai_radar.generate_dataset()
    
    # Prepare data for training
    radar_data = []
    rd_maps = []
    targets = []
    
    for i in range(len(ai_radar)):
        sample = ai_radar[i]
        
        # Get time domain data and convert to proper format [num_rx, num_chirps, samples_per_chirp, 2]
        time_domain = sample['time_domain']  # Shape: [num_chirps, samples_per_chirp, 2]
        # Reshape to [1, num_chirps, samples_per_chirp, 2] for model input (num_rx=1)
        iq_data = time_domain[np.newaxis, :, :, :]  # [1, num_chirps, samples_per_chirp, 2]
        radar_data.append(iq_data)
        
        # Get RD map and normalize
        rdm = sample['range_doppler_map']
        rdm_normalized = (rdm - rdm.min()) / (rdm.max() - rdm.min() + 1e-8)
        rd_maps.append(rdm_normalized[np.newaxis, :, :])  # Add channel dimension
        
        # Get target mask and squeeze extra dimension
        target_mask = sample['target_mask']  # Shape: [128, 4096, 1]
        target_mask_squeezed = target_mask.squeeze(-1)  # Remove last dimension -> [128, 4096]
        targets.append(target_mask_squeezed)  # Keep 2D dimensions
    
    # Split data
    split_idx = int(0.8 * len(radar_data))
    train_data = RadarDataset(radar_data[:split_idx], targets[:split_idx], rd_maps[:split_idx])
    val_data = RadarDataset(radar_data[split_idx:], targets[split_idx:], rd_maps[split_idx:])
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    
    # Create model
    model = RadarTimeNet(
        num_rx=1,
        num_chirps=128,
        samples_per_chirp=1024,
        out_doppler_bins=128,
        out_range_bins=4096,  # Match AIRadarDataset output (zero_pad_factor * N_samples // 2 = 8 * 1024 // 2 = 4096)
        use_learnable_fft=False
    ).to(device)
    
    # Define loss functions and# Loss functions with class balancing
    criterion_rd = nn.MSELoss()
    
    # Calculate positive weight for balanced BCE loss
    total_pixels = 0
    positive_pixels = 0
    for target in targets:
        # Handle both numpy arrays and torch tensors
        if hasattr(target, 'numel'):  # PyTorch tensor
            total_pixels += target.numel()
            positive_pixels += torch.sum(target).item()
        elif hasattr(target, 'size') and not callable(target.size):  # NumPy array
            total_pixels += target.size
            positive_pixels += np.sum(target)
        else:  # Other array-like objects
            total_pixels += np.prod(target.shape)
            positive_pixels += np.sum(target)
    
    pos_weight = (total_pixels - positive_pixels) / positive_pixels if positive_pixels > 0 else 1.0
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    print(f"\nClass balance info:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Positive pixels: {positive_pixels} ({100*positive_pixels/total_pixels:.4f}%)")
    print(f"  Positive weight: {pos_weight.item():.2f}")
    
    criterion_det = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 50
    best_val_f1 = 0.0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_rd_loss, train_det_loss = train_epoch(
            model, train_loader, optimizer, criterion_rd, criterion_det, device
        )
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion_rd, criterion_det, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (RD: {train_rd_loss:.4f}, Det: {train_det_loss:.4f})")
        print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"  Val RD MSE: {val_metrics['rd_mse']:.4f}")
        print(f"  Val Detection - P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        
        # Save best model (save at least once)
        if val_metrics['f1_score'] > best_val_f1 or epoch == 0:
            best_val_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), 'best_radar_model.pth')
            if epoch == 0:
                print(f"  Initial model saved! F1: {best_val_f1:.4f}")
            else:
                print(f"  New best model saved! F1: {best_val_f1:.4f}")
    
    # Load best model for comparison
    model.load_state_dict(torch.load('best_radar_model.pth'))
    
    # Compare with traditional CFAR
    comparison_results = compare_with_cfar(model, ai_radar, device, num_samples=200)
    
    print("\n=== Training Complete ===")
    print(f"Best validation F1-score: {best_val_f1:.4f}")
    print("Model saved as 'best_radar_model.pth'")
    print("Training data saved as 'radar_training_data.pkl'")
