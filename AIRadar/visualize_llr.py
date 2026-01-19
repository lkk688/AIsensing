#!/usr/bin/env python
"""
visualize_llr.py - Visualize LLR bit heatmaps to debug demapper learning

This script shows:
1. Input I/Q constellation
2. LLR bit predictions for each bit plane
3. Ground truth bit planes
4. Bit error locations

Run after training to check if model is learning correct bit boundaries.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from AIradar_comm_model_g2c import (
    JointRadarCommNet_G2, G2DeepDataset, ConfigEncoder, 
    symbol_to_bits, RADAR_COMM_CONFIGS_G2, MAX_MOD_ORDER
)


def visualize_llr_heatmaps(model, sample, device, save_path=None):
    """
    Visualize LLR outputs for a single sample.
    
    Args:
        model: Trained JointRadarCommNet_G2 model
        sample: Tuple from G2DeepDataset
        device: torch device
        save_path: Optional path to save figure
    """
    radar_in, radar_tgt, comm_in, comm_tgt, meta = sample
    mod_order = meta['mod_order']
    active_bits = int(np.log2(mod_order))
    
    # Move to device
    comm_in_b = comm_in.unsqueeze(0).to(device)
    config_tensor = meta['config_tensor'].unsqueeze(0).to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        _, llr_logits = model(
            radar_in.unsqueeze(0).to(device),
            comm_in_b, 
            config_tensor, 
            mod_order
        )
    
    # Convert to numpy
    llr = llr_logits[0].cpu().numpy()  # [6, H, W]
    bit_probs = torch.sigmoid(llr_logits[0]).cpu().numpy()  # [6, H, W]
    
    # Ground truth bits
    gt_bits = symbol_to_bits(comm_tgt.unsqueeze(0), mod_order)[0].numpy()  # [active_bits, H, W]
    
    # Predicted bits (hard decision)
    pred_bits = (bit_probs[:active_bits] > 0.5).astype(float)
    
    # Bit errors
    bit_errors = (pred_bits != gt_bits[:active_bits]).astype(float)
    ber_per_bit = bit_errors.mean(axis=(1, 2))
    total_ber = bit_errors.mean()
    
    # I/Q data
    iq_real = comm_in[0].numpy()  # [H, W]
    iq_imag = comm_in[1].numpy()  # [H, W]
    
    # Create visualization
    n_cols = active_bits + 1
    fig, axes = plt.subplots(4, n_cols, figsize=(3*n_cols, 12))
    
    # Row 0: I/Q constellation + LLR heatmaps
    ax = axes[0, 0]
    ax.scatter(iq_real.flatten(), iq_imag.flatten(), s=1, alpha=0.3, c='blue')
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title(f'I/Q Constellation\n{mod_order}-QAM')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    for i in range(active_bits):
        ax = axes[0, i+1]
        im = ax.imshow(llr[i], cmap='RdBu', aspect='auto', vmin=-5, vmax=5)
        ax.set_title(f'LLR Bit {i}\nmean={llr[i].mean():.2f}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 1: Bit probabilities
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, f'Total BER:\n{total_ber:.4f}', 
                     fontsize=14, ha='center', va='center', 
                     transform=axes[1, 0].transAxes)
    
    for i in range(active_bits):
        ax = axes[1, i+1]
        im = ax.imshow(bit_probs[i], cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'P(bit{i}=1)\nBER={ber_per_bit[i]:.4f}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Ground truth bits
    axes[2, 0].axis('off')
    axes[2, 0].text(0.5, 0.5, 'Ground Truth\nBits', 
                     fontsize=12, ha='center', va='center',
                     transform=axes[2, 0].transAxes)
    
    for i in range(active_bits):
        ax = axes[2, i+1]
        im = ax.imshow(gt_bits[i], cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'GT Bit {i}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Bit errors
    axes[3, 0].axis('off')
    axes[3, 0].text(0.5, 0.5, 'Bit Errors\n(Red = Error)', 
                     fontsize=12, ha='center', va='center',
                     transform=axes[3, 0].transAxes)
    
    for i in range(active_bits):
        ax = axes[3, i+1]
        im = ax.imshow(bit_errors[i], cmap='Reds', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Errors Bit {i}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f"LLR Visualization - {meta['config_name']}\n"
                 f"SNR={meta['snr_db']:.1f}dB, {mod_order}-QAM, BER={total_ber:.4f}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'ber_per_bit': ber_per_bit,
        'total_ber': total_ber,
        'llr_mean': llr[:active_bits].mean(axis=(1, 2)),
        'llr_std': llr[:active_bits].std(axis=(1, 2)),
    }


def visualize_multiple_snr(model, config_name, device, out_dir, 
                           snr_list=[5, 15, 25], num_samples=3):
    """Visualize LLR heatmaps across different SNR levels."""
    os.makedirs(out_dir, exist_ok=True)
    
    from AIradar_comm_dataset_g2 import AIRadar_Comm_Dataset_G2
    
    results = []
    
    for snr_db in snr_list:
        print(f"Generating samples at SNR={snr_db}dB...")
        
        # Generate fresh samples at specific SNR
        ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=os.path.join(out_dir, f'snr_{snr_db}'),
            drawfig=False,
            fixed_snr=snr_db,
            enable_clutter=True,
            enable_imperfect_csi=True
        )
        
        deep_ds = G2DeepDataset(config_name, num_samples, out_dir, 'viz')
        deep_ds.g2_ds = ds
        
        for idx in range(min(len(deep_ds), num_samples)):
            sample = deep_ds[idx]
            save_path = os.path.join(out_dir, f'llr_snr{snr_db}_sample{idx}.png')
            stats = visualize_llr_heatmaps(model, sample, device, save_path)
            stats['snr_db'] = snr_db
            stats['sample_idx'] = idx
            results.append(stats)
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    snr_bers = {}
    for r in results:
        snr = r['snr_db']
        if snr not in snr_bers:
            snr_bers[snr] = []
        snr_bers[snr].append(r['total_ber'])
    
    snrs = sorted(snr_bers.keys())
    mean_bers = [np.mean(snr_bers[s]) for s in snrs]
    
    ax.semilogy(snrs, mean_bers, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title(f'DL BER vs SNR - {config_name}', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-4, 1])
    
    plt.tight_layout()
    summary_path = os.path.join(out_dir, 'ber_vs_snr_summary.png')
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved summary: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Visualize LLR heatmaps')
    parser.add_argument('--ckpt', type=str, default='data/AIradar_comm_model_g2c/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='CN0566_TRADITIONAL',
                        help='Config name to visualize')
    parser.add_argument('--out_dir', type=str, default='data/llr_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--snr_list', type=str, default='5,10,15,20,25,30',
                        help='Comma-separated SNR values')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    use_universal = ckpt.get('use_universal', True)
    
    model = JointRadarCommNet_G2(
        base_ch=48, cond_dim=64, max_mod_order=MAX_MOD_ORDER,
        use_universal=use_universal
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print(f"Model loaded (use_universal={use_universal})")
    
    # Parse SNR list
    snr_list = [int(s) for s in args.snr_list.split(',')]
    
    # Generate visualizations
    visualize_multiple_snr(model, args.config, device, args.out_dir, snr_list)
    print(f"\nVisualization complete! Check {args.out_dir}")


if __name__ == '__main__':
    main()
