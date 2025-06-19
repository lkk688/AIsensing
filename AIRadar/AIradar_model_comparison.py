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
import json
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

# Import custom modules
from AIradar_datasetv4 import RadarDataset
from AIradar_processing import RadarProcessing
from AIradar_transformer import RadarTransformerNet

# Import other models for comparison
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AImodels_joint import RadarNet, RadarTimeToFreqNet

def compare_models(args):
    """
    Compare the performance of different radar detection models
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create test dataset
    print("\nStep 1: Creating test dataset...")
    
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
        add_realistic_effects=True,  # Always use realistic effects for testing
        signal_type=args.signal_type
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Get sample data to determine dimensions
    sample_data = test_dataset[0]
    time_data = sample_data['time_data']
    range_doppler_map = sample_data['range_doppler_map']
    num_rx, num_chirps, samples_per_chirp, _ = time_data.shape
    doppler_bins, range_bins = range_doppler_map.shape
    
    # Step 2: Load models
    print("\nStep 2: Loading models...")
    
    # Initialize radar processing for conventional detection
    radar_processor = RadarProcessing(
        num_range_bins=range_bins,
        num_doppler_bins=doppler_bins,
        num_rx=num_rx,
        cfar_window_params={
            'num_train': 8,
            'num_guard': 4,
            'pfa': 1e-5
        }
    )
    
    # Dictionary to store models
    models = {}
    model_types = {}
    
    # Load Transformer model if path is provided
    if args.transformer_model_path:
        print(f"Loading Transformer model from {args.transformer_model_path}")
        transformer_model = RadarTransformerNet(
            num_rx=num_rx,
            num_chirps=num_chirps,
            samples_per_chirp=samples_per_chirp,
            out_doppler_bins=doppler_bins,
            out_range_bins=range_bins,
            dim=args.model_dim,
            depth=args.model_depth,
            heads=args.model_heads,
            mlp_dim=args.model_mlp_dim,
            dropout=0.0,  # No dropout during evaluation
            use_learnable_fft=args.use_learnable_fft,
            use_cnn_backbone=args.use_cnn_backbone
        )
        
        checkpoint = torch.load(args.transformer_model_path, map_location=device)
        transformer_model.load_state_dict(checkpoint['model_state_dict'])
        transformer_model = transformer_model.to(device)
        transformer_model.eval()
        
        models['Transformer'] = transformer_model
        model_types['Transformer'] = 'time_domain'
    
    # Load RadarNet model if path is provided
    if args.radarnet_model_path:
        print(f"Loading RadarNet model from {args.radarnet_model_path}")
        radarnet_model = RadarNet(
            input_channels=1,  # Magnitude of range-Doppler map
            output_channels=1  # Detection mask
        )
        
        checkpoint = torch.load(args.radarnet_model_path, map_location=device)
        radarnet_model.load_state_dict(checkpoint['model_state_dict'])
        radarnet_model = radarnet_model.to(device)
        radarnet_model.eval()
        
        models['RadarNet'] = radarnet_model
        model_types['RadarNet'] = 'range_doppler'
    
    # Load RadarTimeToFreqNet model if path is provided
    if args.time_to_freq_model_path:
        print(f"Loading RadarTimeToFreqNet model from {args.time_to_freq_model_path}")
        time_to_freq_model = RadarTimeToFreqNet(
            num_rx=num_rx,
            num_chirps=num_chirps,
            samples_per_chirp=samples_per_chirp,
            output_channels=1  # Detection mask
        )
        
        checkpoint = torch.load(args.time_to_freq_model_path, map_location=device)
        time_to_freq_model.load_state_dict(checkpoint['model_state_dict'])
        time_to_freq_model = time_to_freq_model.to(device)
        time_to_freq_model.eval()
        
        models['TimeToFreqNet'] = time_to_freq_model
        model_types['TimeToFreqNet'] = 'time_domain'
    
    # Step 3: Evaluate models
    print("\nStep 3: Evaluating models...")
    
    # Dictionary to store metrics
    metrics = {
        'CFAR': {'precision': 0, 'recall': 0, 'f1': 0, 'ap': 0, 'auc': 0, 
                'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    }
    
    for model_name in models.keys():
        metrics[model_name] = {'precision': 0, 'recall': 0, 'f1': 0, 'ap': 0, 'auc': 0,
                              'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    
    # Lists to store predictions and ground truth for ROC and PR curves
    all_predictions = {model_name: [] for model_name in models.keys()}
    all_predictions['CFAR'] = []
    all_ground_truth = []
    
    # Process test data
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Get data
            time_data = batch['time_data'].to(device)
            target_mask = batch['target_mask'].to(device)
            range_doppler_map = batch['range_doppler_map'].to(device)
            target_info = batch['target_info']
            
            # Store ground truth for ROC and PR curves
            all_ground_truth.append(target_mask.cpu().numpy())
            
            # Evaluate each model
            for model_name, model in models.items():
                # Forward pass based on model type
                if model_types[model_name] == 'time_domain':
                    outputs = model(time_data)
                else:  # range_doppler
                    # Prepare input for RadarNet (magnitude of range-Doppler map)
                    rd_magnitude = torch.abs(range_doppler_map).unsqueeze(1)  # Add channel dimension
                    outputs = model(rd_magnitude)
                    # Reshape output if needed
                    if outputs.shape != target_mask.shape:
                        outputs = outputs.permute(0, 2, 3, 1)  # [batch, H, W, C]
                
                # Store predictions for ROC and PR curves
                all_predictions[model_name].append(outputs.cpu().numpy())
                
                # Calculate detection metrics
                predictions = (outputs > 0.5).float()
                for b in range(time_data.size(0)):
                    pred = predictions[b].cpu().numpy()
                    gt = target_mask[b].cpu().numpy()
                    
                    # Count true positives, false positives, false negatives
                    metrics[model_name]['true_positives'] += np.sum(np.logical_and(pred > 0.5, gt > 0.5))
                    metrics[model_name]['false_positives'] += np.sum(np.logical_and(pred > 0.5, gt < 0.5))
                    metrics[model_name]['false_negatives'] += np.sum(np.logical_and(pred < 0.5, gt > 0.5))
            
            # Conventional CFAR detection
            for b in range(time_data.size(0)):
                rd_map = range_doppler_map[b].cpu().numpy()
                cfar_detections = radar_processor.cfar_2d_detection(rd_map)
                gt = target_mask[b].cpu().numpy()
                
                # Store CFAR predictions
                if b == 0:
                    batch_cfar = np.expand_dims(cfar_detections, axis=(0, -1))
                else:
                    batch_cfar = np.concatenate([batch_cfar, np.expand_dims(cfar_detections, axis=(0, -1))], axis=0)
                
                # Count conventional detection metrics
                metrics['CFAR']['true_positives'] += np.sum(np.logical_and(cfar_detections > 0.5, gt > 0.5))
                metrics['CFAR']['false_positives'] += np.sum(np.logical_and(cfar_detections > 0.5, gt < 0.5))
                metrics['CFAR']['false_negatives'] += np.sum(np.logical_and(cfar_detections < 0.5, gt > 0.5))
            
            all_predictions['CFAR'].append(batch_cfar)
            
            # Visualize a few samples
            if i < args.num_visualizations:
                for b in range(min(time_data.size(0), 2)):
                    plt.figure(figsize=(15, 10))
                    
                    # Plot range-Doppler map
                    plt.subplot(2, 3, 1)
                    rd_map = range_doppler_map[b].cpu().numpy()
                    plt.imshow(20 * np.log10(np.abs(rd_map) + 1e-10), aspect='auto', cmap='viridis')
                    plt.colorbar(label='Magnitude (dB)')
                    plt.title('Range-Doppler Map')
                    plt.xlabel('Range Bin')
                    plt.ylabel('Doppler Bin')
                    
                    # Plot ground truth
                    plt.subplot(2, 3, 2)
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
                    
                    # Plot CFAR detection
                    plt.subplot(2, 3, 3)
                    cfar_detections = radar_processor.cfar_2d_detection(rd_map)
                    plt.imshow(cfar_detections, aspect='auto', cmap='plasma')
                    plt.colorbar(label='Detection')
                    plt.title('Conventional CFAR Detection')
                    plt.xlabel('Range Bin')
                    plt.ylabel('Doppler Bin')
                    
                    # Plot model detections
                    plot_idx = 4
                    for model_name, model in models.items():
                        if plot_idx <= 6:  # Only show up to 3 models in the plot
                            plt.subplot(2, 3, plot_idx)
                            
                            # Get model output
                            if model_types[model_name] == 'time_domain':
                                output = model(time_data[b:b+1])[0].cpu().numpy()
                            else:  # range_doppler
                                rd_magnitude = torch.abs(range_doppler_map[b:b+1]).unsqueeze(1)
                                output = model(rd_magnitude)[0].cpu().numpy()
                                if output.shape[0] == 1:  # [C, H, W]
                                    output = output.transpose(1, 2, 0)  # [H, W, C]
                            
                            plt.imshow(output[:, :, 0], aspect='auto', cmap='plasma')
                            plt.colorbar(label='Detection Confidence')
                            plt.title(f'{model_name} Detection')
                            plt.xlabel('Range Bin')
                            plt.ylabel('Doppler Bin')
                            
                            plot_idx += 1
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, f'comparison_sample_{i}_{b}.png'))
                    plt.close()
    
    # Step 4: Calculate and save metrics
    print("\nStep 4: Calculating metrics...")
    
    # Concatenate all predictions and ground truth
    for model_name in all_predictions.keys():
        all_predictions[model_name] = np.concatenate(all_predictions[model_name], axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    # Calculate precision, recall, F1 score for each model
    for model_name in metrics.keys():
        tp = metrics[model_name]['true_positives']
        fp = metrics[model_name]['false_positives']
        fn = metrics[model_name]['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[model_name]['precision'] = precision
        metrics[model_name]['recall'] = recall
        metrics[model_name]['f1'] = f1_score
        
        # Calculate Average Precision and AUC
        if model_name in all_predictions:
            # Flatten predictions and ground truth for sklearn metrics
            y_pred = all_predictions[model_name].flatten()
            y_true = all_ground_truth.flatten()
            
            # Average Precision
            metrics[model_name]['ap'] = average_precision_score(y_true, y_pred)
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            metrics[model_name]['auc'] = auc(fpr, tpr)
    
    # Print metrics
    print("\nDetection Metrics:")
    print("-" * 80)
    print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'AP':<10} {'AUC':<10}")
    print("-" * 80)
    
    for model_name, model_metrics in metrics.items():
        print(f"{model_name:<15} {model_metrics['precision']:.4f}     {model_metrics['recall']:.4f}     "
              f"{model_metrics['f1']:.4f}      {model_metrics['ap']:.4f}     {model_metrics['auc']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'comparison_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Step 5: Plot ROC and Precision-Recall curves
    print("\nStep 5: Plotting ROC and PR curves...")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for model_name in all_predictions.keys():
        y_pred = all_predictions[model_name].flatten()
        y_true = all_ground_truth.flatten()
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'roc_curves.png'))
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    
    for model_name in all_predictions.keys():
        y_pred = all_predictions[model_name].flatten()
        y_true = all_ground_truth.flatten()
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'pr_curves.png'))
    
    print(f"\nComparison complete. Results saved to {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Compare radar detection models')
    
    # Dataset parameters
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
    parser.add_argument('--signal_type', type=str, default='FMCW', help='Signal type (FMCW)')
    
    # Model parameters
    parser.add_argument('--model_dim', type=int, default=128, help='Model dimension for transformer')
    parser.add_argument('--model_depth', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--model_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--model_mlp_dim', type=int, default=256, help='MLP hidden dimension')
    parser.add_argument('--use_learnable_fft', action='store_true', help='Use learnable FFT')
    parser.add_argument('--use_cnn_backbone', action='store_true', help='Use CNN backbone')
    
    # Model paths
    parser.add_argument('--transformer_model_path', type=str, default='', help='Path to transformer model checkpoint')
    parser.add_argument('--radarnet_model_path', type=str, default='', help='Path to RadarNet model checkpoint')
    parser.add_argument('--time_to_freq_model_path', type=str, default='', help='Path to RadarTimeToFreqNet model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--num_visualizations', type=int, default=5, help='Number of test samples to visualize')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./model_comparison_results', help='Output directory')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    compare_models(args)
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")