#!/usr/bin/env python3
"""
Training script for RadarEndToEnd model using FMCW radar dataset.

This script trains the RadarEndToEnd model (RadarTimeNet + RadarNet) using the
FMCWRadarDataset for realistic FMCW radar signal processing and target detection.

"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Add AIRadarLib to path
sys.path.append('/Users/kaikailiu/Documents/MyRepo/radarsensing/AIRadar')
from AIRadarLib.modeling_RadarNet import RadarEndToEnd, RadarNet, MultiTaskLoss
from AIRadarLib.modeling_TimeNet import RadarTimeNet
from AIRadarLib.modeling_RadarNet import get_device
from AIradar_fmcw_dataset import FMCWRadarDataset


class FMCWRadarDatasetWrapper(Dataset):
    """
    Wrapper for FMCWRadarDataset to match the expected format for RadarEndToEnd training.
    
    The RadarEndToEnd model expects:
    - Input: [B, num_rx, num_chirps, samples_per_chirp, 2] (complex time-domain signals)
    - Detection target: [B, num_doppler_bins, num_range_bins] (binary detection map)
    - Velocity target: [B, num_doppler_bins, num_range_bins] (velocity values)
    - Metadata: List of target dictionaries
    """
    
    def __init__(self, fmcw_dataset):
        """
        Initialize wrapper around FMCWRadarDataset.
        
        Args:
            fmcw_dataset: FMCWRadarDataset instance
        """
        self.fmcw_dataset = fmcw_dataset
        self.num_rx = fmcw_dataset.num_rx
        self.num_chirps = fmcw_dataset.num_chirps
        self.samples_per_chirp = fmcw_dataset.samples_per_chirp
        self.num_doppler_bins = fmcw_dataset.num_doppler_bins
        self.num_range_bins = fmcw_dataset.num_range_bins
        
    def __len__(self):
        return len(self.fmcw_dataset)
    
    def __getitem__(self, idx):
        """
        Get dataset item in format expected by RadarEndToEnd training.
        
        Returns:
            x: Input signal [num_rx, num_chirps, samples_per_chirp, 2]
            y_det: Detection target [num_doppler_bins, num_range_bins]
            y_vel: Velocity target [num_doppler_bins, num_range_bins]
            mod_type: Modulation type (string)
            meta: Target metadata (list of target dictionaries)
        """
        sample = self.fmcw_dataset[idx]
        
        # Extract beat signals as input (already processed by FMCW mixing)
        # Shape: [num_rx, num_chirps, samples_per_chirp] (complex)
        beat_signals = sample['beat_signal']  # Complex array
        
        # Convert complex signals to [num_rx, num_chirps, samples_per_chirp, 2] format
        x = np.stack([np.real(beat_signals), np.imag(beat_signals)], axis=-1)
        x = torch.from_numpy(x).float()
        
        # Detection target from target mask
        # Shape: [num_doppler_bins, num_range_bins]
        y_det = torch.from_numpy(sample['target_mask']).float()
        
        # Create velocity target map from target information
        y_vel = torch.zeros(self.num_doppler_bins, self.num_range_bins, dtype=torch.float32)
        
        # Convert target info to expected metadata format
        meta = []
        for target in sample['target_info']:
            # Convert range to range bin
            range_bin = int(target['range'] / self.fmcw_dataset.range_resolution)
            range_bin = min(range_bin, self.num_range_bins - 1)
            
            # Convert velocity to Doppler bin (centered)
            doppler_bin = int(target['velocity'] / self.fmcw_dataset.velocity_resolution) + self.num_doppler_bins // 2
            doppler_bin = np.clip(doppler_bin, 0, self.num_doppler_bins - 1)
            
            # Set velocity value in velocity target map
            # Normalize velocity to [-1, 1] range for training
            normalized_velocity = target['velocity'] / self.fmcw_dataset.max_velocity
            y_vel[doppler_bin, range_bin] = normalized_velocity
            
            # Create metadata entry
            target_meta = {
                'range_bin': range_bin,
                'doppler_bin': doppler_bin,
                'range': target['range'],
                'velocity': target['velocity'],
                'rcs': target['rcs'],
                'snr_db': sample['snr_db']
            }
            meta.append(target_meta)
        
        # Modulation type (FMCW)
        mod_type = 'fmcw'
        
        return x, y_det, y_vel, mod_type, meta


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length metadata lists.
    """
    x_batch = []
    y_det_batch = []
    y_vel_batch = []
    mod_type_batch = []
    meta_batch = []
    
    for x, y_det, y_vel, mod_type, meta in batch:
        x_batch.append(x)
        y_det_batch.append(y_det)
        y_vel_batch.append(y_vel)
        mod_type_batch.append(mod_type)
        meta_batch.append(meta)  # Keep as list, don't stack
    
    # Stack tensors
    x_batch = torch.stack(x_batch)
    y_det_batch = torch.stack(y_det_batch)
    y_vel_batch = torch.stack(y_vel_batch)
    
    return x_batch, y_det_batch, y_vel_batch, mod_type_batch, meta_batch


def create_fmcw_data_loaders(num_samples=1000, batch_size=8, train_split=0.8, 
                             save_path='data/fmcw_radar', **dataset_kwargs):
    """
    Create training and validation data loaders for FMCW radar dataset.
    
    Args:
        num_samples: Number of samples to generate
        batch_size: Batch size for training
        train_split: Fraction of data for training
        save_path: Path to save/load dataset
        **dataset_kwargs: Additional arguments for FMCWRadarDataset
    
    Returns:
        train_loader, val_loader: DataLoader instances
    """
    print(f"Creating FMCW radar dataset with {num_samples} samples...")
    
    # Create FMCW dataset with optimized parameters for training
    fmcw_dataset = FMCWRadarDataset(
        num_samples=num_samples,
        save_path=save_path,
        num_chirps=32,         # Reduced from 150 to save memory
        num_range_bins=64,     # Reasonable size for training
        num_doppler_bins=32,   # Reasonable size for training
        chirp_duration=1e-3,   # Reduced to 1ms to get fewer samples per chirp
        **dataset_kwargs
    )
    
    # Wrap dataset for training compatibility
    wrapped_dataset = FMCWRadarDatasetWrapper(fmcw_dataset)
    
    # Split into train/validation
    train_size = int(train_split * len(wrapped_dataset))
    val_size = len(wrapped_dataset) - train_size
    train_dataset, val_dataset = random_split(wrapped_dataset, [train_size, val_size])
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader, fmcw_dataset


def train_fmcw_radar_model(train_loader, val_loader, fmcw_dataset, 
                           epochs=10, learning_rate=1e-3, device=None,
                           save_model_path='models/fmcw_radar_model.pth'):
    """
    Train RadarEndToEnd model on FMCW radar data.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        fmcw_dataset: Original FMCW dataset for parameter reference
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (cuda/cpu)
        save_model_path: Path to save trained model
    
    Returns:
        model: Trained RadarEndToEnd model
        training_history: Dictionary with training metrics
    """
    if device is None:
        device = get_device()
    
    print(f"Training on device: {device}")
    
    # Initialize model components
    time_net = RadarTimeNet(
        num_rx=fmcw_dataset.num_rx,
        num_chirps=fmcw_dataset.num_chirps,
        samples_per_chirp=fmcw_dataset.samples_per_chirp,
        out_doppler_bins=fmcw_dataset.num_doppler_bins,
        out_range_bins=fmcw_dataset.num_range_bins,
        use_learnable_fft=True,
        support_ofdm=False  # FMCW doesn't use OFDM
    )
    
    detect_net = RadarNet(
        in_channels=2,  # Complex (real, imag)
        num_classes=1,  # Binary detection
        detect_threshold=0.5,
        max_targets=fmcw_dataset.max_targets
    )
    
    # Create end-to-end model
    model = RadarEndToEnd(
        time_net=time_net,
        detect_net=detect_net,
        detect_threshold=0.5,
        max_targets=fmcw_dataset.max_targets
    )
    
    model = model.to(device)
    
    # Loss function and optimizer
    multitask_loss = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_det_loss': [],
        'train_vel_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_total = 0
        train_det_loss_total = 0
        train_vel_loss_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (x, y_det, y_vel, mod_type, meta) in enumerate(train_pbar):
            x = x.to(device)
            y_det = y_det.to(device)
            y_vel = y_vel.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model outputs
            outputs = model(x)
            
            # Extract detection and velocity predictions
            if isinstance(outputs, dict):
                det_pred = outputs['detection_map'].squeeze(1)  # [B, D, R]
                vel_pred = outputs['velocity_map'][:, 0].squeeze()  # [B, D, R] - use first velocity component
            else:
                det_pred = outputs.squeeze(1)  # Assume detection output
                vel_pred = torch.zeros_like(det_pred)  # Placeholder
            
            # Compute loss
            loss, det_loss, vel_loss = multitask_loss(
                det_pred.unsqueeze(1), y_det.unsqueeze(1),  # Add channel dimension
                vel_pred.unsqueeze(-1), y_vel.unsqueeze(-1)  # Add feature dimension
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss_total += loss.item()
            train_det_loss_total += det_loss.item()
            train_vel_loss_total += vel_loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Det': f'{det_loss.item():.4f}',
                'Vel': f'{vel_loss.item():.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss_total = 0
        all_preds = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for x, y_det, y_vel, mod_type, meta in val_pbar:
                x = x.to(device)
                y_det = y_det.to(device)
                y_vel = y_vel.to(device)
                
                # Forward pass
                outputs = model(x)
                
                # Extract predictions
                if isinstance(outputs, dict):
                    det_pred = outputs['detection_map'].squeeze(1)
                    vel_pred = outputs['velocity_map'][:, 0].squeeze()
                else:
                    det_pred = outputs.squeeze(1)
                    vel_pred = torch.zeros_like(det_pred)
                
                # Compute loss
                loss, _, _ = multitask_loss(
                    det_pred.unsqueeze(1), y_det.unsqueeze(1),
                    vel_pred.unsqueeze(-1), y_vel.unsqueeze(-1)
                )
                
                val_loss_total += loss.item()
                
                # Collect predictions for metrics
                preds = (det_pred > 0.5).cpu().numpy().flatten()
                targets = y_det.cpu().numpy().flatten()
                all_preds.extend(preds)
                all_targets.extend(targets)
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        avg_train_det_loss = train_det_loss_total / len(train_loader)
        avg_train_vel_loss = train_vel_loss_total / len(train_loader)
        
        # Validation metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_det_loss'].append(avg_train_det_loss)
        history['train_vel_loss'].append(avg_train_vel_loss)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Det: {avg_train_det_loss:.4f}, Vel: {avg_train_vel_loss:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_model_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_rx': fmcw_dataset.num_rx,
            'num_chirps': fmcw_dataset.num_chirps,
            'samples_per_chirp': fmcw_dataset.samples_per_chirp,
            'num_doppler_bins': fmcw_dataset.num_doppler_bins,
            'num_range_bins': fmcw_dataset.num_range_bins,
            'max_targets': fmcw_dataset.max_targets
        },
        'training_history': history
    }, save_model_path)
    
    print(f"\nTraining completed! Model saved to: {save_model_path}")
    return model, history


def plot_training_history(history, save_path='plots/training_history.png'):
    """
    Plot training history metrics.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Detection and velocity loss
    axes[0, 1].plot(history['train_det_loss'], label='Detection Loss', color='green')
    axes[0, 1].plot(history['train_vel_loss'], label='Velocity Loss', color='orange')
    axes[0, 1].set_title('Training Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Validation metrics
    axes[1, 0].plot(history['val_precision'], label='Precision', color='purple')
    axes[1, 0].plot(history['val_recall'], label='Recall', color='brown')
    axes[1, 0].plot(history['val_f1'], label='F1 Score', color='pink')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate (if available)
    axes[1, 1].text(0.5, 0.5, 'Training\nCompleted\nSuccessfully!', 
                     ha='center', va='center', fontsize=16, 
                     transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Training Status')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train RadarEndToEnd model on FMCW radar data')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='data/fmcw_radar', help='Dataset save path')
    parser.add_argument('--model_path', type=str, default='models/fmcw_radar_model.pth', help='Model save path')
    parser.add_argument('--max_range', type=float, default=100, help='Maximum detection range (m)')
    parser.add_argument('--max_velocity', type=float, default=50, help='Maximum velocity (m/s)')
    parser.add_argument('--num_targets', type=int, default=3, help='Maximum number of targets')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FMCW Radar Model Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max range: {args.max_range}m")
    print(f"  Max velocity: {args.max_velocity}m/s")
    print(f"  Max targets: {args.num_targets}")
    print("=" * 60)
    
    # Create data loaders
    train_loader, val_loader, fmcw_dataset = create_fmcw_data_loaders(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_path=args.save_path,
        max_range=args.max_range,
        max_velocity=args.max_velocity,
        max_targets=args.num_targets,
        drawfig=True  # Generate visualization plots
    )
    
    # Train model
    model, history = train_fmcw_radar_model(
        train_loader=train_loader,
        val_loader=val_loader,
        fmcw_dataset=fmcw_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_model_path=args.model_path
    )
    
    # Plot training history
    plot_training_history(history, 'plots/fmcw_training_history.png')
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {args.model_path}")
    print(f"Training plots saved to: plots/")
    print("=" * 60)


if __name__ == '__main__':
    main()