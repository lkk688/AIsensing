
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path to import AIRadar modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AIRadar.AIradar_datasetv6 import AIRadarDataset
from AIRadar.AIradar_trainv6 import RadarTimeNetV5

def check_rdm_difference():
    # Load a small dataset
    ds = AIRadarDataset(num_samples=1, config_name='config1', drawfig=False, save_path='data/test_repro')
    sample = ds[0]
    
    # Get input time domain data
    # [Rx, Chirps, Samples, 2]
    iq_data = sample['time_domain'].unsqueeze(0) # Add batch dim -> [1, ... ]
    if iq_data.dim() == 4:
        iq_data = iq_data.unsqueeze(1) # [1, 1, Chirps, Samples, 2]
    
    # Get GT RDM from dataset (which uses windowing)
    gt_rdm = sample['range_doppler_map'].numpy()
    
    # Initialize Model
    n_rx = iq_data.shape[1]
    n_chirps = iq_data.shape[2]
    n_samples = iq_data.shape[3]
    n_doppler, n_range = gt_rdm.shape
    
    model = RadarTimeNetV5(
        num_rx=n_rx,
        num_chirps=n_chirps,
        samples_per_chirp=n_samples,
        fft_size=n_range if n_range >= n_samples else None,
        out_range_bins=n_range,
        out_doppler_bins=n_doppler
    )
    model.eval()
    
    # Run model
    with torch.no_grad():
        outputs = model(iq_data)
        model_rdm = outputs['rd_map_db'][0].numpy() # [Doppler, Range]
        
    # Plot comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normalize for plotting
    gt_norm = gt_rdm - np.max(gt_rdm)
    model_norm = model_rdm - np.max(model_rdm)
    
    im1 = axs[0].imshow(gt_norm, aspect='auto', vmin=-60, vmax=0, cmap='jet')
    axs[0].set_title("Dataset RDM (Windowed)")
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(model_norm, aspect='auto', vmin=-60, vmax=0, cmap='jet')
    axs[1].set_title("Model RDM (No Window)")
    plt.colorbar(im2, ax=axs[1])
    
    plt.savefig('rdm_comparison.png')
    print("Saved rdm_comparison.png")

if __name__ == "__main__":
    check_rdm_difference()
