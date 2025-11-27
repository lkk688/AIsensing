import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
import logging
from torch.cuda.amp import autocast, GradScaler

# Import custom modules
from AIradar_datasetv4 import RadarDataset
from AIRadarLib.modeling_transformer import RadarTransformerNet
from AIradar_processing import RadarProcessing

def train_transformer_model(args):
    """
    Train the transformer-based radar detection model
    """
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training.log')
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logging.info(f"Random seed set to {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create datasets
    print("\nStep 1: Creating datasets...")
    
    # Training dataset
    train_dataset = RadarDataset(
        num_samples=args.train_samples,
        sample_rate=args.sample_rate,
        transceiver_bandwidth=args.transceiver_bandwidth,
        num_chirps=args.num_chirps,
        bandwidth=args.bandwidth,
        center_freq=args.center_freq,
        num_rx=args.num_rx,
        max_targets=args.max_targets,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        apply_realistic_effects=args.realistic_effects,
        signal_type=args.signal_type
    )
    
    # Validation dataset
    val_dataset = RadarDataset(
        num_samples=args.val_samples,
        sample_rate=args.sample_rate,
        transceiver_bandwidth=args.transceiver_bandwidth,
        num_chirps=args.num_chirps,
        bandwidth=args.bandwidth,
        center_freq=args.center_freq,
        num_rx=args.num_rx,
        max_targets=args.max_targets,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        add_realistic_effects=args.realistic_effects,
        signal_type=args.signal_type
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Step 2: Create model
    print("\nStep 2: Creating transformer model...")
    
    # Get sample data to determine dimensions
    sample_data = train_dataset[0]
    time_data = sample_data['time_data']
    num_rx, num_chirps, samples_per_chirp, _ = time_data.shape
    
    # Create model
    model = RadarTransformerNet(
        num_rx=num_rx,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp,
        out_doppler_bins=args.out_doppler_bins,
        out_range_bins=args.out_range_bins,
        dim=args.model_dim,
        depth=args.model_depth,
        heads=args.model_heads,
        mlp_dim=args.model_mlp_dim,
        dropout=args.dropout,
        use_learnable_fft=args.use_learnable_fft,
        use_cnn_backbone=args.use_cnn_backbone
    )
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Step 3: Define loss function and optimizer
    logging.info("\nStep 3: Setting up training...")
    
    # Binary cross entropy loss for detection
    criterion = nn.BCELoss()
    
    # Optimizer with learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.learning_rate, 
            steps_per_epoch=len(train_loader), epochs=args.epochs
        )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Step 4: Training loop
    logging.info("\nStep 4: Starting training...")
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping parameters
    patience = args.early_stopping_patience
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in train_bar:
            # Get data
            time_data = batch['time_data'].to(device)  # [batch, num_rx, num_chirps, samples_per_chirp, 2]
            target_mask = batch['target_mask'].to(device)  # [batch, doppler_bins, range_bins, 1]
            
            # Resize target mask to match model output dimensions if needed
            if target_mask.shape[1] != args.out_doppler_bins or target_mask.shape[2] != args.out_range_bins:
                target_mask = torch.nn.functional.interpolate(
                    target_mask.permute(0, 3, 1, 2),  # [batch, 1, doppler_bins, range_bins]
                    size=(args.out_doppler_bins, args.out_range_bins),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # [batch, out_doppler_bins, out_range_bins, 1]
            
            # Forward pass with mixed precision if enabled
            optimizer.zero_grad()
            
            if args.mixed_precision and torch.cuda.is_available():
                with autocast():
                    outputs = model(time_data)  # [batch, out_doppler_bins, out_range_bins, 1]
                    loss = criterion(outputs, target_mask)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(time_data)  # [batch, out_doppler_bins, out_range_bins, 1]
                loss = criterion(outputs, target_mask)
                
                # Backward pass and optimize
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * time_data.size(0)
            
            # Calculate accuracy (detection accuracy)
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == target_mask).sum().item()
            train_total += target_mask.numel()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': (predictions == target_mask).float().mean().item()
            })
        
        # Calculate average training loss and accuracy
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in val_bar:
                # Get data
                time_data = batch['time_data'].to(device)
                target_mask = batch['target_mask'].to(device)
                
                # Resize target mask to match model output dimensions if needed
                if target_mask.shape[1] != args.out_doppler_bins or target_mask.shape[2] != args.out_range_bins:
                    target_mask = torch.nn.functional.interpolate(
                        target_mask.permute(0, 3, 1, 2),
                        size=(args.out_doppler_bins, args.out_range_bins),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Forward pass
                outputs = model(time_data)
                
                # Calculate loss
                loss = criterion(outputs, target_mask)
                
                # Update metrics
                val_loss += loss.item() * time_data.size(0)
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == target_mask).sum().item()
                val_total += target_mask.numel()
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': (predictions == target_mask).float().mean().item()
                })
        
        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.scheduler in ['cosine', 'onecycle']:
            scheduler.step()
        
        # Print epoch summary
        logging.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if args.scheduler != 'plateau' else None,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'args': args,
            }, os.path.join(args.output_dir, 'best_transformer_model.pth'))
            logging.info(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if args.scheduler != 'plateau' else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'args': args,
            }, os.path.join(args.output_dir, f'transformer_checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if args.early_stopping and patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            logging.info(f"Best model was at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f} and accuracy: {best_val_accuracy:.4f}")
            break
    
    # Step 5: Plot training curves
    logging.info("\nStep 5: Plotting training curves...")
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'transformer_training_curves.png'))
    
    # Step 6: Evaluate on test data
    logging.info("\nStep 6: Evaluating on test data...")
    
    # Create test dataset with realistic parameters
    test_dataset = RadarDataset(
        num_samples=args.test_samples,
        sample_rate=args.sample_rate,
        transceiver_bandwidth=args.transceiver_bandwidth,
        num_chirps=args.num_chirps,
        bandwidth=args.bandwidth,
        center_freq=args.center_freq,
        num_rx=args.num_rx,
        max_targets=args.max_targets,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        apply_realistic_effects=True,  # Always use realistic effects for testing
        signal_type=args.signal_type
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_transformer_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize radar processing for conventional detection
    radar_processor = RadarProcessing(
        num_range_bins=args.out_range_bins,
        num_doppler_bins=args.out_doppler_bins,
        num_rx=args.num_rx,
        cfar_window_params={
            'num_train': 8,
            'num_guard': 4,
            'pfa': 1e-5
        }
    )
    
    # Evaluate and visualize results
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # Metrics for detection performance
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Conventional detection metrics
    conv_true_positives = 0
    conv_false_positives = 0
    conv_false_negatives = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Get data
            time_data = batch['time_data'].to(device)
            target_mask = batch['target_mask'].to(device)
            range_doppler_map = batch['range_doppler_map'].to(device)
            target_info = batch['target_info']
            
            # Resize target mask if needed
            if target_mask.shape[1] != args.out_doppler_bins or target_mask.shape[2] != args.out_range_bins:
                target_mask = torch.nn.functional.interpolate(
                    target_mask.permute(0, 3, 1, 2),
                    size=(args.out_doppler_bins, args.out_range_bins),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Forward pass
            outputs = model(time_data)
            
            # Calculate loss
            loss = criterion(outputs, target_mask)
            test_loss += loss.item() * time_data.size(0)
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            test_correct += (predictions == target_mask).sum().item()
            test_total += target_mask.numel()
            
            # Calculate detection metrics
            for b in range(time_data.size(0)):
                # ML detection
                pred = predictions[b].cpu().numpy()
                gt = target_mask[b].cpu().numpy()
                
                # Count true positives, false positives, false negatives
                true_positives += np.sum(np.logical_and(pred > 0.5, gt > 0.5))
                false_positives += np.sum(np.logical_and(pred > 0.5, gt < 0.5))
                false_negatives += np.sum(np.logical_and(pred < 0.5, gt > 0.5))
                
                # Conventional CFAR detection
                rd_map = range_doppler_map[b].cpu().numpy()
                cfar_detections = radar_processor.cfar_2d_detection(rd_map)
                
                # Count conventional detection metrics
                conv_true_positives += np.sum(np.logical_and(cfar_detections > 0.5, gt > 0.5))
                conv_false_positives += np.sum(np.logical_and(cfar_detections > 0.5, gt < 0.5))
                conv_false_negatives += np.sum(np.logical_and(cfar_detections < 0.5, gt > 0.5))
            
            # Visualize a few samples
            if i < args.num_visualizations:
                for b in range(min(time_data.size(0), 4)):
                    plt.figure(figsize=(15, 10))
                    
                    # Plot range-Doppler map
                    plt.subplot(2, 2, 1)
                    rd_map = range_doppler_map[b].cpu().numpy()
                    plt.imshow(20 * np.log10(np.abs(rd_map) + 1e-10), aspect='auto', cmap='viridis')
                    plt.colorbar(label='Magnitude (dB)')
                    plt.title('Range-Doppler Map')
                    plt.xlabel('Range Bin')
                    plt.ylabel('Doppler Bin')
                    
                    # Plot ground truth
                    plt.subplot(2, 2, 2)
                    gt = target_mask[b].cpu().numpy()
                    plt.imshow(gt[:, :, 0], aspect='auto', cmap='binary')
                    plt.colorbar(label='Target Presence')
                    plt.title('Ground Truth')
                    plt.xlabel('Range Bin')
                    plt.ylabel('Doppler Bin')
                    
                    # Annotate targets
                    for t in range(len(target_info['range_bin'][b])):
                        if target_info['range_bin'][b][t] >= 0:  # Valid target
                            r_bin = target_info['range_bin'][b][t]
                            d_bin = target_info['doppler_bin'][b][t]
                            plt.plot(r_bin, d_bin, 'rx', markersize=10)
                    
                    # Plot ML detection
                    plt.subplot(2, 2, 3)
                    pred = outputs[b].cpu().numpy()
                    plt.imshow(pred[:, :, 0], aspect='auto', cmap='plasma')
                    plt.colorbar(label='Detection Confidence')
                    plt.title('ML Detection (Transformer)')
                    plt.xlabel('Range Bin')
                    plt.ylabel('Doppler Bin')
                    
                    # Plot CFAR detection
                    plt.subplot(2, 2, 4)
                    cfar_detections = radar_processor.cfar_2d_detection(rd_map)
                    plt.imshow(cfar_detections, aspect='auto', cmap='plasma')
                    plt.colorbar(label='Detection')
                    plt.title('Conventional CFAR Detection')
                    plt.xlabel('Range Bin')
                    plt.ylabel('Doppler Bin')
                    
                    # Detected targets
                    detected_ranges, detected_dopplers = np.where(cfar_detections > 0.5)
                    for r, d in zip(detected_ranges, detected_dopplers):
                        plt.plot(d, r, 'gx', markersize=8)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, f'test_sample_{i}_{b}.png'))
                    plt.close()
    
    # Calculate test metrics
    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / test_total
    
    # Calculate precision, recall, F1 score for ML detection
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate precision, recall, F1 score for conventional detection
    conv_precision = conv_true_positives / (conv_true_positives + conv_false_positives) if (conv_true_positives + conv_false_positives) > 0 else 0
    conv_recall = conv_true_positives / (conv_true_positives + conv_false_negatives) if (conv_true_positives + conv_false_negatives) > 0 else 0
    conv_f1_score = 2 * conv_precision * conv_recall / (conv_precision + conv_recall) if (conv_precision + conv_recall) > 0 else 0
    
    # Print test results
    logging.info(f"\nTest Results:")
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"\nML Detection (Transformer):")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1_score:.4f}")
    logging.info(f"\nConventional CFAR Detection:")
    logging.info(f"Precision: {conv_precision:.4f}")
    logging.info(f"Recall: {conv_recall:.4f}")
    logging.info(f"F1 Score: {conv_f1_score:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"\nML Detection (Transformer):\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
        f.write(f"\nConventional CFAR Detection:\n")
        f.write(f"Precision: {conv_precision:.4f}\n")
        f.write(f"Recall: {conv_recall:.4f}\n")
        f.write(f"F1 Score: {conv_f1_score:.4f}\n")
    
    logging.info(f"\nTraining and evaluation complete. Results saved to {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer-based radar detection model')
    
    # Dataset parameters
    parser.add_argument('--train_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--test_samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--sample_rate', type=float, default=500e6, help='Sample rate in Hz')
    parser.add_argument('--transceiver_bandwidth', type=float, default=30e6, help='Transceiver bandwidth in Hz')
    parser.add_argument('--num_chirps', type=int, default=128, help='Number of chirps per frame')
    parser.add_argument('--bandwidth', type=float, default=200e6, help='Bandwidth in Hz')
    parser.add_argument('--center_freq', type=float, default=10e9, help='Center frequency in Hz')
    parser.add_argument('--num_rx', type=int, default=4, help='Number of receiver antennas')
    parser.add_argument('--max_targets', type=int, default=3, help='Maximum number of targets')
    parser.add_argument('--snr_min', type=float, default=5.0, help='Minimum SNR in dB')
    parser.add_argument('--snr_max', type=float, default=20.0, help='Maximum SNR in dB')
    parser.add_argument('--apply_realistic_effects', dest='realistic_effects', action='store_true', help='Apply realistic effects to data')
    parser.add_argument('--signal_type', type=str, default='FMCW', help='Signal type (FMCW)')
    
    # Model parameters
    parser.add_argument('--out_doppler_bins', type=int, default=128, help='Output Doppler bins')
    parser.add_argument('--out_range_bins', type=int, default=256, help='Output range bins')
    parser.add_argument('--model_dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--model_depth', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--model_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--model_mlp_dim', type=int, default=256, help='MLP hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_learnable_fft', action='store_true', help='Use learnable FFT')
    parser.add_argument('--use_cnn_backbone', action='store_true', help='Use CNN backbone')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--num_visualizations', type=int, default=5, help='Number of test samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'onecycle'], help='Learning rate scheduler')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./transformer_results', help='Output directory')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    train_transformer_model(args)
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")