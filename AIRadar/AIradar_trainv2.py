import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import csv
import time
from matplotlib.ticker import FormatStrFormatter
IMG_FORMAT=".pdf" #".png"
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    import matplotlib.pyplot as plt
    # Create a basic replacement for seaborn's heatmap
    def heatmap(data, ax=None, cmap='viridis', annot=False, fmt='.2f', **kwargs):
        if ax is None:
            ax = plt.gca()
        im = ax.imshow(data, cmap=cmap)
        plt.colorbar(im)
        if annot:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, fmt.format(data[i, j]), ha='center', va='center')
        return im
    sns = type('Sns', (), {'heatmap': heatmap})()
#from tqdm import tqdm

from AIradar_datasetv3 import RadarDataset, print_dataset_info
from AIradar_processing import RadarProcessing

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RadarNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(RadarNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 32) #(real,image)2->32
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        
        # Decoder
        self.dec3 = ConvBlock(128, 64)
        self.dec2 = ConvBlock(64, 32)
        self.dec1 = ConvBlock(32, 16)
        
        # Output layer
        self.output = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)#x: [32, 2, 12, 64]=>[32, 32, 12, 64]
        p1 = self.pool(e1) #[32, 32, 6, 32]
        
        e2 = self.enc2(p1) #[32, 64, 6, 32]
        p2 = self.pool(e2) #[32, 64, 3, 16]
        
        e3 = self.enc3(p2) #[32, 128, 3, 16]
        
        # Decoder with skip connections
        d3 = self.dec3(e3) #[32, 64, 3, 16]
        u2 = self.upsample(d3) #[32, 64, 6, 32]
        u2 = u2 + e2  # Skip connection
        
        d2 = self.dec2(u2) #[32, 32, 6, 32]
        u1 = self.upsample(d2) #[32, 32, 12, 64]
        u1 = u1 + e1  # Skip connection
        
        d1 = self.dec1(u1) #[32, 16, 12, 64]
        
        # Output
        out = self.output(d1) #[32, 1, 12, 64]
        out = self.sigmoid(out)
        
        # Reshape to match the expected output format [batch, H, W, 1]
        out = out.permute(0, 2, 3, 1)
        
        return out #[32, 12, 64, 1]


class RadarTimeNet(nn.Module):
    def __init__(self, num_rx=2, num_chirps=12, samples_per_chirp=20, out_doppler_bins=12, out_range_bins=64):
        """
        Neural network for processing time-domain radar data
        
        Args:
            num_rx: Number of receiver antennas
            num_chirps: Number of chirps in a frame
            samples_per_chirp: Number of time samples per chirp
            out_doppler_bins: Number of Doppler bins in output
            out_range_bins: Number of range bins in output
        """
        super(RadarTimeNet, self).__init__()
        
        # Store dimensions
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.out_doppler_bins = out_doppler_bins
        self.out_range_bins = out_range_bins
        
        # Initial processing of time-domain data
        # Input shape: [batch, num_rx, num_chirps, samples_per_chirp, 2]
        self.time_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Process across receivers
        self.rx_conv = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(num_rx, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Process across chirps (for Doppler information)
        self.chirp_conv = nn.Sequential(
            nn.Conv2d(32 * num_chirps, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Range-Doppler processing
        self.rd_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling to match output dimensions
        self.upsample = nn.Sequential(
            nn.Upsample(size=(out_doppler_bins, out_range_bins), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Optional: Add a learnable FFT-like layer
        self.fft_weights_range = nn.Parameter(torch.randn(samples_per_chirp, out_range_bins, 2))  # 2 for real/imag
        self.fft_weights_doppler = nn.Parameter(torch.randn(num_chirps, out_doppler_bins, 2))  # 2 for real/imag
        
    def complex_multiply(self, a_real, a_imag, b_real, b_imag):
        """Complex multiplication: (a_real + j*a_imag) * (b_real + j*b_imag)"""
        real_part = a_real * b_real - a_imag * b_imag
        imag_part = a_real * b_imag + a_imag * b_real
        return real_part, imag_part
    
    def learnable_fft(self, x):
        """Apply learnable FFT-like transform to time domain data"""
        # x shape: [batch, 1, num_chirps, samples_per_chirp, 2] or [batch, 1, num_chirps, samples_per_chirp, 1]
        batch_size = x.shape[0] #[32, 1, 12, 64, 1]
        
        # Check if we have complex data (last dimension size 2) or just real data (last dimension size 1)
        has_complex = x.shape[-1] == 2
        
        # Extract real and imaginary parts
        if has_complex:
            x_real = x[:, 0, :, :, 0]  # [batch, num_chirps, samples_per_chirp]
            x_imag = x[:, 0, :, :, 1]  # [batch, num_chirps, samples_per_chirp]
        else:
            # If we only have real data, set imaginary part to zeros
            x_real = x[:, 0, :, :, 0]  # [batch, num_chirps, samples_per_chirp]
            x_imag = torch.zeros_like(x_real)  # [batch, num_chirps, samples_per_chirp]
        
        # Rest of the method remains the same
        # Apply range FFT weights
        range_real = torch.zeros(batch_size, self.num_chirps, self.out_range_bins, device=x.device)
        range_imag = torch.zeros(batch_size, self.num_chirps, self.out_range_bins, device=x.device)

        for b in range(batch_size):
            for c in range(self.num_chirps):
                for r in range(self.out_range_bins):
                    # Compute weighted sum for range bin r
                    w_real = self.fft_weights_range[:, r, 0]
                    w_imag = self.fft_weights_range[:, r, 1]
                    
                    # Complex multiplication and summation
                    real_sum, imag_sum = self.complex_multiply(
                        x_real[b, c, :], x_imag[b, c, :], w_real, w_imag
                    )
                    range_real[b, c, r] = real_sum.sum()
                    range_imag[b, c, r] = imag_sum.sum()
        
        # Apply Doppler FFT weights
        rd_real = torch.zeros(batch_size, self.out_doppler_bins, self.out_range_bins, device=x.device)
        rd_imag = torch.zeros(batch_size, self.out_doppler_bins, self.out_range_bins, device=x.device)
        
        for b in range(batch_size):
            for d in range(self.out_doppler_bins):
                for r in range(self.out_range_bins):
                    # Compute weighted sum for Doppler bin d
                    w_real = self.fft_weights_doppler[:, d, 0]
                    w_imag = self.fft_weights_doppler[:, d, 1]
                    
                    # Complex multiplication and summation
                    real_sum, imag_sum = self.complex_multiply(
                        range_real[b, :, r], range_imag[b, :, r], w_real, w_imag
                    )
                    rd_real[b, d, r] = real_sum.sum()
                    rd_imag[b, d, r] = imag_sum.sum()
        
        # Compute magnitude
        rd_magnitude = torch.sqrt(rd_real**2 + rd_imag**2)
        
        # Normalize
        rd_magnitude = rd_magnitude / (self.num_chirps * self.samples_per_chirp)
        
        return rd_magnitude.unsqueeze(1)  # [batch, 1, out_doppler_bins, out_range_bins]
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch, num_rx, num_chirps, samples_per_chirp, 2]
                where the last dimension contains I/Q data
        
        Returns:
            Detection map of shape [batch, out_doppler_bins, out_range_bins, 1]
        """
        batch_size = x.shape[0] #torch.Size([32, 4, 12, 1500, 2]
        
        # Permute input to put channels (I/Q) first
        # [batch, num_rx, num_chirps, samples_per_chirp, 2] -> [batch, 2, num_rx, num_chirps, samples_per_chirp]
        x = x.permute(0, 4, 1, 2, 3) #torch.Size([32, 2, 4, 12, 1500])
        
        # Apply 3D convolutions to process time samples
        x = self.time_conv(x)  # [batch, 32, num_rx, num_chirps, samples_per_chirp]
        #[32, 32, 4, 12, 1500]
        # Process across receivers
        x = self.rx_conv(x)  # [batch, 32, 1, num_chirps, samples_per_chirp] [32, 32, 1, 12, 1500]
        x = x.squeeze(2)  # [batch, 32, num_chirps, samples_per_chirp] [32, 32, 12, 1500]
        
        # Reshape to process chirps
        x = x.reshape(batch_size, -1, self.samples_per_chirp)  # [batch, 32*num_chirps, samples_per_chirp] [32, 384, 1500]
        x = x.unsqueeze(-1)  # [batch, 32*num_chirps, samples_per_chirp, 1] [32, 384, 1500, 1]
        x = x.permute(0, 1, 3, 2)  # [batch, 32*num_chirps, 1, samples_per_chirp] [32, 384, 1, 1500]
        
        # Apply 2D convolutions for further processing
        x = self.chirp_conv(x)  # [batch, 64, 1, samples_per_chirp] [32, 64, 1, 1500]
        
        # Process for range-Doppler information
        x = self.rd_conv(x)  # [batch, 128, 1, samples_per_chirp] [32, 128, 1, 1500]
        
        # Upsample to match output dimensions
        x = self.upsample(x)  # [batch, 1, out_doppler_bins, out_range_bins] [32, 1, 12, 64]
        
        # Optional: Add learnable FFT output
        if hasattr(self, 'fft_weights_range') and hasattr(self, 'fft_weights_doppler'):
            # Apply learnable FFT
            fft_output = self.learnable_fft(x.permute(0, 2, 3, 1).unsqueeze(1))
            
            # Combine with CNN output
            x = x + 0.1 * fft_output
            x = torch.clamp(x, 0, 1)
        
        # Reshape to match expected output format [batch, H, W, 1]
        x = x.permute(0, 2, 3, 1)  # [batch, out_doppler_bins, out_range_bins, 1]
        
        return x

class RadarTimeToFreqNet(nn.Module):
    def __init__(self, num_rx=2, num_chirps=12, samples_per_chirp=20, out_doppler_bins=12, out_range_bins=64):
        """
        Neural network that converts time-domain radar data to frequency domain
        and then uses RadarNet for detection
        
        Args:
            num_rx: Number of receiver antennas
            num_chirps: Number of chirps in a frame
            samples_per_chirp: Number of time samples per chirp
            out_doppler_bins: Number of Doppler bins in output
            out_range_bins: Number of range bins in output
        """
        super(RadarTimeToFreqNet, self).__init__()
        
        # Store dimensions
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.out_doppler_bins = out_doppler_bins
        self.out_range_bins = out_range_bins
        
        # Create the RadarNet model for processing range-Doppler maps
        self.radar_net = RadarNet(in_channels=2, out_channels=1)
        
        # Learnable FFT weights for range dimension
        self.range_fft_weights = nn.Parameter(
            torch.randn(samples_per_chirp, out_range_bins, 2)
        )
        
        # Learnable FFT weights for Doppler dimension
        self.doppler_fft_weights = nn.Parameter(
            torch.randn(num_chirps, out_doppler_bins, 2)
        )
        
        # Optional: Add convolutional layers to preprocess time-domain data
        self.time_preprocess = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 2, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True)
        )
    
    def complex_multiply(self, a_real, a_imag, b_real, b_imag):
        """Complex multiplication: (a_real + j*a_imag) * (b_real + j*b_imag)"""
        real_part = a_real * b_real - a_imag * b_imag
        imag_part = a_real * b_imag + a_imag * b_real
        return real_part, imag_part
    
    def time_to_frequency(self, x):
        """
        Convert time-domain data to frequency domain (range-Doppler map)
        using shared processing code from AIradar_datasetv2
        
        Args:
            x: Time-domain data [batch, num_rx, num_chirps, samples_per_chirp, 2]
                where the last dimension contains I/Q data
        
        Returns:
            Range-Doppler map [batch, 2, out_doppler_bins, out_range_bins]
                where the first dimension contains real/imaginary parts
        """
        # Import the dataset module to use its processing functions
        from AIradar_datasetv2 import RadarDataset
        
        batch_size = x.shape[0]
        device = x.device
        
        # Create a temporary dataset object to access its methods
        # We don't need to initialize with actual data since we're just using the processing methods
        radar_dataset = RadarDataset(num_samples=1, training=False)
        
        # Process each sample in the batch
        rd_maps = []
        for i in range(batch_size):
            # Get the current sample
            sample = x[i].cpu().numpy()  # Move to CPU and convert to numpy
            
            # Convert I/Q format to complex
            # sample shape: [num_rx, num_chirps, samples_per_chirp, 2]
            complex_data = sample[..., 0] + 1j * sample[..., 1]
            
            # Use the dataset's time_to_range_doppler method
            # This ensures consistent processing between dataset generation and model inference
            rd_map = radar_dataset._time_to_range_doppler(complex_data)
            
            # Convert complex output to real/imaginary format
            rd_real = np.real(rd_map).astype(np.float32)
            rd_imag = np.imag(rd_map).astype(np.float32)
            
            # Stack real and imaginary parts
            rd_stacked = np.stack([rd_real, rd_imag], axis=0)
            rd_maps.append(rd_stacked)
        
        # Stack all processed samples
        rd_maps = np.stack(rd_maps, axis=0)
        
        # Convert back to tensor and move to the original device
        rd_maps_tensor = torch.tensor(rd_maps, device=device)
        
        return rd_maps_tensor
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Time-domain data [batch, num_rx, num_chirps, samples_per_chirp, 2]
        
        Returns:
            Detection map [batch, out_doppler_bins, out_range_bins, 1]
        """
        # Convert time-domain data to frequency domain
        rd_map = self.time_to_frequency(x)
        
        # Apply RadarNet for detection
        detection = self.radar_net(rd_map)
        
        return detection

def get_device(gpuid='0', useamp=False):
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpuid))  # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        useamp = False
    else:
        device = torch.device("cpu")
        useamp = False
    print("Using device:", device)
    return device, useamp

def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for binary segmentation
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def train_radar_model(output_dir, num_samples=10000, batch_size=32, num_epochs=50, saved_model_path=None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device, useamp = get_device(gpuid='0', useamp=False)
    print(f"Using device: {device}")
    
    # Create or load dataset
    data_path = os.path.join(output_dir, 'radar_simulation_data.npy')
    if os.path.exists(data_path):
        train_data = RadarDataset(datapath=data_path, training=True, drawfig=True)
    else:
        train_data = RadarDataset(num_samples=num_samples, training=True, drawfig=True, save_data=True)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = RadarNet(in_channels=2, out_channels=1).to(device)
    
    # Check if we're loading a saved model
    start_epoch = 0
    if saved_model_path and os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model from {saved_model_path}, starting from epoch {start_epoch}")
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Define loss function - combination of BCE and Dice loss
    def combined_loss(pred, target, alpha=0.5):
        bce = nn.BCELoss()(pred, target)
        dice = dice_loss(pred, target)
        return alpha * bce + (1 - alpha) * dice
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training phase
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = batch['feature_2d'].to(device) #[32, 2, 12, 64]
            targets = batch['labels'].to(device) #[32, 12, 64, 1]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        detection_accuracy = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['feature_2d'].to(device)
                targets = batch['labels'].to(device)
                
                outputs = model(inputs)
                loss = combined_loss(outputs, targets)
                val_loss += loss.item()
                
                # Calculate detection accuracy
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == targets).float().mean()
                detection_accuracy += accuracy.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_detection_accuracy = detection_accuracy / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(output_dir, 'best_radar_model.pth'))
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Detection Accuracy: {avg_detection_accuracy:.4f}')
        print('-' * 60)
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_detection_accuracy])

# Define a custom collate function to handle samples of different sizes
def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return {}
        
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    result = {}
    for key in keys:
        # Check if all samples have this key and the tensors have the same shape
        if all(key in sample and isinstance(sample[key], np.ndarray) and 
              sample[key].shape == batch[0][key].shape for sample in batch):
            # Stack tensors with the same shape
            result[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
        else:
            # For tensors with different shapes or non-tensor items, keep them in a list
            result[key] = [sample.get(key) for sample in batch if key in sample]
    
    return result


#Support for time-domain data with the use_time_domain parameter
def train_radar_modelv2(output_dir, 
                     num_samples=10000,
                     batch_size=32,
                     num_epochs=50,
                     saved_model_path=None,
                     use_time_domain=False,
                     visualize_progress=True,
                     learning_rate=0.001,
                     signal_type='OFDM',
                     data_format='hdf5',
                     data_dir='data/radar',
                     num_workers=4,
                     prefetch_factor=2,
                     use_lazy_loading=True,  # New parameter to control lazy loading
                     cache_size=100):        # Cache size for lazy loading
    """
    Train a radar target detection model
    
    Args:
        output_dir: Directory to save model and results
        num_samples: Number of samples to generate if no dataset exists
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        saved_model_path: Path to a saved model to continue training
        use_time_domain: Whether to use time-domain data instead of range-Doppler maps
        visualize_progress: Whether to visualize training progress
        learning_rate: Initial learning rate
        signal_type: Type of radar signal to use ('FMCW', 'OFDM', or 'Sine')
        data_format: Format to save/load data ('numpy' or 'hdf5')
        data_dir: Directory containing radar data
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches to prefetch
        use_lazy_loading: Whether to use lazy loading for HDF5 files
        cache_size: Number of samples to keep in memory cache for lazy loading
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, signal_type.lower()), exist_ok=True)
    os.makedirs(os.path.join(output_dir, signal_type.lower(), 'visualizations'), exist_ok=True)
    
    # Set device
    device, useamp = get_device(gpuid='0', useamp=False)
    print(f"Using device: {device}")
    
    # Determine file extension based on data format
    file_ext = '.h5' if data_format.lower() == 'hdf5' else '.npy'
    
    # Create or load dataset
    data_path = os.path.join(data_dir, f'radar_simulation_data_{signal_type.lower()}{file_ext}')
    numpy_path = os.path.join(data_dir, f'radar_simulation_data_{signal_type.lower()}.npy')
    hdf5_path = os.path.join(data_dir, f'radar_simulation_data_{signal_type.lower()}.h5')
    
        # Load dataset with appropriate method based on file type
    if os.path.exists(hdf5_path) or os.path.exists(numpy_path) or os.path.exists(data_path):
        # Determine which file exists and use the appropriate loading method
        if os.path.exists(hdf5_path) and use_lazy_loading:
            data_file = hdf5_path
            loading_method = "lazy loading"
            dataset_args = {
                "use_lazy_loading": True,
                "cache_size": cache_size
            }
        elif os.path.exists(numpy_path):
            data_file = numpy_path
            loading_method = "memory mapping"
            dataset_args = {
                "use_memory_mapping": True
            }
        elif os.path.exists(data_path):
            data_file = data_path
            if data_path.endswith('.h5') and use_lazy_loading:
                loading_method = "lazy loading"
                dataset_args = {
                    "use_lazy_loading": True,
                    "cache_size": cache_size
                }
            elif data_path.endswith('.npy'):
                loading_method = "memory mapping"
                dataset_args = {
                    "use_memory_mapping": True
                }
            else:
                loading_method = "standard loading"
                dataset_args = {}
        
        print(f"Loading existing radar dataset from {data_file} with {loading_method}")
        train_data = RadarDataset(
            datapath=data_file,
            training=True,
            drawfig=visualize_progress,
            **dataset_args
        )
    else:
        print(f"Generating new radar dataset with {num_samples} samples using {signal_type} signal type")
        print(f"Data will be saved in {data_format} format")
        
        # Check if dataset_params was provided as a parameter
        if 'dataset_params' in locals() and dataset_params is not None:
            # Use the provided dataset parameters
            train_data = RadarDataset(
                num_samples=num_samples, 
                training=True, 
                drawfig=visualize_progress, 
                save_data=True,
                savedataformat=data_format,
                signal_type=signal_type,
                **dataset_params  # Unpack the dataset parameters
            )
        else:
            # Use default parameters
            train_data = RadarDataset(
                num_samples=num_samples, 
                training=True, 
                drawfig=visualize_progress, 
                save_data=True,
                savedataformat=data_format,
                # Updated SDR parameters to match real device
                sample_rate=3e6,
                chirp_duration=500e-6,
                num_chirps=12,
                bandwidth=500e6,
                center_freq=2.1e9,
                num_rx=4,
                num_tx=1,
                signal_type=signal_type
            )

    # Check if time-domain data is available
    has_time_domain = 'time_domain' in train_data[0]
    if use_time_domain and not has_time_domain:
        print("Warning: Time-domain data not available. Falling back to range-Doppler maps.")
        use_time_domain = False
    
    # Print dataset information
    print(f"Dataset size: {len(train_data)}") #10000
    print(f"Range-Doppler map shape: {train_data[0]['feature_2d'].shape}")
    if has_time_domain:
        print(f"Time-domain data shape: {train_data[0]['time_domain'].shape}")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = torch.utils.data.random_split(train_data, [train_size, val_size])

    # Create data loaders
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # Create model based on input type
    if use_time_domain:
        # For time-domain data, we need a different model architecture
        # Get the shape of time-domain data
        time_shape = train_data[0]['time_domain'].shape  # [num_rx, num_chirps, samples_per_chirp, 2]
        # model = RadarTimeNet(
        #     num_rx=time_shape[0],
        #     num_chirps=time_shape[1],
        #     samples_per_chirp=time_shape[2],
        #     out_doppler_bins=train_data[0]['labels'].shape[0],
        #     out_range_bins=train_data[0]['labels'].shape[1]
        # ).to(device)
        # print("Using RadarTimeNet model for time-domain data")
        model = RadarTimeToFreqNet(
            num_rx=time_shape[0],
            num_chirps=time_shape[1],
            samples_per_chirp=time_shape[2],
            out_doppler_bins=train_data[0]['labels'].shape[0],
            out_range_bins=train_data[0]['labels'].shape[1]
        ).to(device)
        print("Using RadarTimeToFreqNet model for time-domain data")
    else:
        # Use the standard RadarNet for range-Doppler maps
        model = RadarNet(in_channels=2, out_channels=1).to(device)
        print("Using RadarNet model for range-Doppler maps")
    
    # Check if we're loading a saved model
    start_epoch = 0
    if saved_model_path and os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model from {saved_model_path}, starting from epoch {start_epoch}")
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Define loss function - combination of BCE and Dice loss
    def combined_loss(pred, target, alpha=0.5):
        bce = nn.BCELoss()(pred, target)
        dice = dice_loss(pred, target)
        return alpha * bce + (1 - alpha) * dice
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'detection_accuracy': [],
        'learning_rate': []
    }
    
    # Create CSV file for training history
    history_file = os.path.join(output_dir, 'training_history.csv')
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'detection_accuracy', 'learning_rate'])
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training phase
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch+num_epochs}'):
            try:
                # Select input data based on mode
                if use_time_domain:
                    if isinstance(batch['time_domain'], list):
                        # Skip this batch if time_domain data is inconsistent
                        continue
                    inputs = batch['time_domain'].to(device)  # [batch, num_rx, num_chirps, samples_per_chirp, 2]
                else:
                    if isinstance(batch['feature_2d'], list):
                        # Skip this batch if feature_2d data is inconsistent
                        continue
                    inputs = batch['feature_2d'].to(device)  # [batch, 2, num_doppler_bins, num_range_bins]
                    
                if isinstance(batch['labels'], list):
                    # Skip this batch if labels are inconsistent
                    continue
                targets = batch['labels'].to(device)  # [batch, num_doppler_bins, num_range_bins, 1]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = combined_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            except RuntimeError as e:
                if "each element in list of batch should be of equal size" in str(e):
                    # Skip this batch due to inconsistent sizes
                    print("Skipping batch with inconsistent sizes")
                    continue
                else:
                    # Re-raise other runtime errors
                    raise e
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        detection_accuracy = 0.0
        false_alarm_rate = 0.0
        missed_detection_rate = 0.0
        
        # Store predictions for visualization
        val_inputs = []
        val_targets = []
        val_outputs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation'):
                try:
                    # Select input data based on mode
                    if use_time_domain:
                        inputs = batch['time_domain'].to(device)
                    else:
                        inputs = batch['feature_2d'].to(device)
                        
                    targets = batch['labels'].to(device)
                    
                    outputs = model(inputs)
                    loss = combined_loss(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calculate detection metrics
                    predictions = (outputs > 0.5).float()
                    accuracy = (predictions == targets).float().mean()
                    detection_accuracy += accuracy.item()
                    
                    # Calculate false alarms and missed detections
                    false_alarms = ((predictions == 1) & (targets == 0)).float().mean().item()
                    missed_detections = ((predictions == 0) & (targets == 1)).float().mean().item()
                    
                    false_alarm_rate += false_alarms
                    missed_detection_rate += missed_detections
                    
                    # Store first batch for visualization
                    if len(val_inputs) == 0 and visualize_progress:
                        if use_time_domain:
                            # For time domain, store the corresponding range-Doppler map for visualization
                            val_inputs.append(batch['feature_2d'][:4].cpu().numpy())
                        else:
                            val_inputs.append(inputs[:4].cpu().numpy())
                        val_targets.append(targets[:4].cpu().numpy())
                        val_outputs.append(outputs[:4].cpu().numpy())
                except RuntimeError as e:
                    if "each element in list of batch should be of equal size" in str(e):
                        # Skip this batch due to inconsistent sizes
                        print(f"Skipping validation batch with inconsistent sizes")
                        continue
                    else:
                        # Re-raise other runtime errors
                        raise e
        
        # Calculate average metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_detection_accuracy = detection_accuracy / len(val_loader)
        avg_false_alarm_rate = false_alarm_rate / len(val_loader)
        avg_missed_detection_rate = missed_detection_rate / len(val_loader)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['detection_accuracy'].append(avg_detection_accuracy)
        history['learning_rate'].append(current_lr)
        
        # Save history to CSV
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_detection_accuracy, current_lr])
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'use_time_domain': use_time_domain,
            }, os.path.join(output_dir, 'best_radar_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{start_epoch+num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Detection Accuracy: {avg_detection_accuracy:.4f}')
        print(f'False Alarm Rate: {avg_false_alarm_rate:.4f}')
        print(f'Missed Detection Rate: {avg_missed_detection_rate:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        print('-' * 60)
        
        # Visualize training progress
        if visualize_progress: # and epoch % 5 == 0:
            # Plot loss curves
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(132)
            plt.plot(history['detection_accuracy'], label='Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Detection Accuracy')
            plt.grid(True)
            
            plt.subplot(133)
            plt.plot(history['learning_rate'], label='LR')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            
            plt.tight_layout()
            # Ensure visualization directory exists
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'visualizations', f'training_progress_epoch_{epoch+1}.pdf'))
            plt.close()
            
            # Visualize predictions
            if len(val_inputs) > 0:
                for i in range(min(4, len(val_inputs[0]))):
                    visualize_detection(
                        val_inputs[0][i],
                        val_targets[0][i],
                        val_outputs[0][i],
                        os.path.join(output_dir, 'visualizations', f'prediction_epoch_{epoch+1}_sample_{i}.pdf')
                    )
    
    # Save final model
    torch.save({
        'epoch': start_epoch + num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'use_time_domain': use_time_domain,
    }, os.path.join(output_dir, 'final_radar_model.pth'))
    
    print(f"Training completed. Final model saved to {os.path.join(output_dir, 'final_radar_model.pth')}")
    
    # Plot final training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(132)
    plt.plot(history['detection_accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Detection Accuracy')
    plt.grid(True)
    
    plt.subplot(133)
    plt.plot(history['learning_rate'], label='LR')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.pdf'))
    plt.close()
    
    return model, history

def test_radar_model(model_path=None, test_data_path=None, output_dir=None, signal_type='OFDM'):
    device, useamp = get_device(gpuid='0', useamp=False)
    
    # Load test data or create new test data with real device parameters
    if test_data_path and os.path.exists(test_data_path):
        test_dataset = RadarDataset(
            datapath=test_data_path,
            training=False,
            drawfig=True
        )
    else:
        print(f"Creating new test dataset with {signal_type} signal type")
        test_dataset = RadarDataset(
            num_samples=1000,
            training=False,
            drawfig=True,
            # Updated SDR parameters to match real device
            sample_rate=3e6,             # 3 MHz sampling rate
            chirp_duration=500e-6,       # 500 microsecond chirp
            num_chirps=1,                # 1 chirp per frame (or 128 for TDD mode)
            bandwidth=500e6,             # 500 MHz bandwidth
            center_freq=2.1e9,           # 2.1 GHz center frequency
            num_rx=4,                    # 4 receive antennas
            num_tx=1,                    # 1 transmit antenna
            signal_type=signal_type      # Use the specified signal type
        )
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the model was trained on time-domain data
    use_time_domain = checkpoint.get('use_time_domain', False)
    
    if use_time_domain and 'time_domain' in test_dataset[0]:
        # Get the shape of time-domain data
        time_shape = test_dataset[0]['time_domain'].shape
        model = RadarTimeNet(
            num_rx=time_shape[0],
            num_chirps=time_shape[1],
            samples_per_chirp=time_shape[2],
            out_doppler_bins=test_dataset[0]['labels'].shape[0],
            out_range_bins=test_dataset[0]['labels'].shape[1]
        ).to(device)
        print("Using RadarTimeNet model for time-domain data")
    else:
        model = RadarNet().to(device)
        print("Using RadarNet model for range-Doppler maps")
        use_time_domain = False
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test metrics
    detection_accuracy = 0
    false_alarm_rate = 0
    missed_detection_rate = 0
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            
            # Select input data based on model type
            if use_time_domain:
                input_data = torch.from_numpy(sample['time_domain']).unsqueeze(0).to(device)
            else:
                input_data = torch.from_numpy(sample['feature_2d']).unsqueeze(0).to(device)
                
            target = torch.from_numpy(sample['labels']).to(device)
            
            # Forward pass
            output = model(input_data)
            predictions = (output > 0.5).float()
            
            # Calculate metrics
            accuracy = (predictions == target).float().mean().item()
            detection_accuracy += accuracy
            
            # Calculate false alarms and missed detections
            false_alarms = ((predictions == 1) & (target == 0)).float().mean().item()
            missed_detections = ((predictions == 0) & (target == 1)).float().mean().item()
            
            false_alarm_rate += false_alarms
            missed_detection_rate += missed_detections
            
            # Visualize results for a few samples
            if i < 5 and output_dir is not None:
                visualize_detection(
                    sample['feature_2d'],  # Always use range-Doppler map for visualization
                    target.cpu().numpy(),
                    predictions[0].cpu().numpy(),
                    os.path.join(output_dir, f'detection_result_{i}.pdf')
                )
    
    # Average metrics
    num_samples = len(test_dataset)
    detection_accuracy /= num_samples
    false_alarm_rate /= num_samples
    missed_detection_rate /= num_samples
    
    print(f'Test Results:')
    print(f'Detection Accuracy: {detection_accuracy:.4f}')
    print(f'False Alarm Rate: {false_alarm_rate:.4f}')
    print(f'Missed Detection Rate: {missed_detection_rate:.4f}')
    
    return detection_accuracy, false_alarm_rate, missed_detection_rate

def visualize_detection(input_data, target, prediction, save_path):
    """Visualize radar detection results with enhanced visualization"""
    # Calculate magnitude from real and imaginary parts
    magnitude = np.sqrt(input_data[0]**2 + input_data[1]**2)
    
    # Create a more informative visualization
    plt.figure(figsize=(15, 10))
    
    # Plot input range-Doppler map with improved dynamic range
    plt.subplot(221)
    rd_db = 20*np.log10(magnitude + 1e-10)
    vmin = np.max(rd_db) - 40  # Dynamic range of 40dB
    plt.imshow(rd_db, aspect='auto', cmap='jet', vmin=vmin)
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Range-Doppler Map')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Plot phase information
    plt.subplot(222)
    phase = np.angle(input_data[0] + 1j*input_data[1]) / np.pi
    plt.imshow(phase, aspect='auto', cmap='hsv', vmin=-1, vmax=1)
    plt.colorbar(label='Phase (π rad)')
    plt.title('Phase Information')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Plot ground truth
    plt.subplot(223)
    plt.imshow(target[:,:,0], aspect='auto', cmap='gray')
    plt.colorbar(label='Target Presence')
    plt.title('Ground Truth')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Plot prediction
    plt.subplot(224)
    plt.imshow(prediction[:,:,0], aspect='auto', cmap='gray')
    plt.colorbar(label='Detection')
    plt.title('Model Prediction')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Add overlay of prediction contours on ground truth
    ax = plt.subplot(223)
    contour = ax.contour(prediction[:,:,0], levels=[0.5], colors='red', alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Add overlay of ground truth contours on prediction
    ax = plt.subplot(224)
    contour = ax.contour(target[:,:,0], levels=[0.5], colors='green', alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# if __name__ == '__main__':
#     output_dir = '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar_training'
#     os.makedirs(output_dir, exist_ok=True)
#     # Generate or load radar data
#     # train_radar_model(
#     #     output_dir=output_dir,
#     #     num_samples=10000,
#     #     batch_size=32,
#     #     num_epochs=50
#     # )
    
#     model_path = '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar_training/best_radar_model.pth'
#     output_dir = '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar_results'
#     os.makedirs(output_dir, exist_ok=True)
    
#     test_radar_model(
#         model_path=model_path,
#         output_dir=output_dir
#     )

def train_val():
    """
    Train and validate radar detection models with updated dataset and processing classes
    """
    # Set output directories for training and results
    output_dir = 'data/radar_training2'
    results_dir = 'data/radar_results2'
    data_dir = 'data/radar'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Define which signal types to use and compare
    signal_types = ['FMCW', 'OFDM']  # Options: 'FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW'
    
    # Step 1: Compare different radar signal types
    print("=" * 80)
    print(f"STEP 1: COMPARING DIFFERENT RADAR SIGNAL TYPES")
    print("=" * 80)
    
    # Use the updated compare_signal_types function
    datasets = compare_signal_types(
        base_path='data/radar_comparison',
        signal_types=signal_types,
        num_samples=50,
        visualize_samples=5
    )
    
    # Step 2: Train models for each signal type
    print("\n" + "=" * 80)
    print(f"STEP 2: TRAINING RADAR DETECTION MODELS")
    print("=" * 80)
    
    models = {}
    histories = {}
    
    for signal_type in signal_types:
        print(f"\nTraining model for {signal_type} signal type...")
        
        # Create specific output directory for this signal type
        signal_output_dir = os.path.join(output_dir, signal_type.lower())
        os.makedirs(signal_output_dir, exist_ok=True)
        
        # Train model with the specific signal type
        model, history = train_radar_modelv2(
            output_dir=signal_output_dir,
            num_samples=5000,          # Reduced sample count for faster training
            batch_size=32,             # Mini-batch size for SGD
            num_epochs=10,             # Number of training epochs
            learning_rate=0.001,       # Initial learning rate
            use_time_domain=True,      # Use time domain data for more accurate detection
            visualize_progress=True,   # Generate visualizations during training
            signal_type=signal_type,   # Use the specified signal type
            data_dir=data_dir,
            # Remove the radar parameters that aren't accepted by train_radar_modelv2
            dataset_params={
                'sample_rate': 3e6,           # 3 MHz sampling rate
                'chirp_duration': 50e-6,      # 50 μs chirp duration
                'num_chirps': 12,             # 12 chirps per frame
                'bandwidth': 150e6,           # 150 MHz bandwidth
                'center_freq': 77e9           # 77 GHz center frequency (automotive radar)
            }
        )
        
        # Store model and history for later comparison
        models[signal_type] = model
        histories[signal_type] = history
    
    # Step 3: Evaluate and compare models
    print("\n" + "=" * 80)
    print(f"STEP 3: EVALUATING AND COMPARING RADAR DETECTION MODELS")
    print("=" * 80)
    
    # Initialize metrics for comparison
    detection_accuracies = {}
    false_alarm_rates = {}
    missed_detection_rates = {}
    
    for signal_type in signal_types:
        print(f"\nEvaluating {signal_type} model...")
        
        # Load the best model for this signal type
        model_path = os.path.join(output_dir, signal_type.lower(), 'best_radar_model.pth')
        signal_results_dir = os.path.join(results_dir, signal_type.lower())
        os.makedirs(signal_results_dir, exist_ok=True)
        
        # Test the model
        accuracy, false_alarm, missed_detection = test_radar_model(
            model_path=model_path,
            output_dir=signal_results_dir,
            # Create test data with the same signal type
            test_data_path=None  # Generate new test data
        )
        
        # Store metrics
        detection_accuracies[signal_type] = accuracy
        false_alarm_rates[signal_type] = false_alarm
        missed_detection_rates[signal_type] = missed_detection
    
    # Step 4: Generate comparison visualizations
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # Create comparison directory
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compare training histories
    plt.figure(figsize=(15, 10))
    
    # Plot training loss comparison
    plt.subplot(221)
    for signal_type in signal_types:
        plt.plot(histories[signal_type]['train_loss'], label=f'{signal_type} Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot validation loss comparison
    plt.subplot(222)
    for signal_type in signal_types:
        plt.plot(histories[signal_type]['val_loss'], label=f'{signal_type} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot detection accuracy comparison
    plt.subplot(223)
    for signal_type in signal_types:
        plt.plot(histories[signal_type]['detection_accuracy'], label=signal_type)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Detection Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot final metrics comparison
    plt.subplot(224)
    x = np.arange(len(signal_types))
    width = 0.25
    
    plt.bar(x - width, [detection_accuracies[st] for st in signal_types], width, label='Accuracy')
    plt.bar(x, [false_alarm_rates[st] for st in signal_types], width, label='False Alarm Rate')
    plt.bar(x + width, [missed_detection_rates[st] for st in signal_types], width, label='Missed Detection Rate')
    
    plt.xlabel('Signal Type')
    plt.ylabel('Rate')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, signal_types)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'model_comparison.pdf'))
    plt.close()
    
    # Step 5: Generate real-world test case with the best model
    print("\n" + "=" * 80)
    print("STEP 5: TESTING WITH REALISTIC RADAR PARAMETERS")
    print("=" * 80)
    
    # Determine best model based on accuracy
    best_signal_type = max(detection_accuracies, key=detection_accuracies.get)
    best_model_path = os.path.join(output_dir, best_signal_type.lower(), 'best_radar_model.pth')
    
    print(f"Best model is {best_signal_type} with accuracy: {detection_accuracies[best_signal_type]:.4f}")
    
    # Create a test dataset with realistic automotive radar parameters
    realistic_test_dataset = RadarDataset(
        num_samples=20,
        training=False,
        drawfig=True,
        # Realistic automotive radar parameters
        sample_rate=3e6,             # 3 MHz sampling rate
        chirp_duration=50e-6,        # 50 μs chirp duration
        num_chirps=12,               # 12 chirps per frame
        bandwidth=150e6,             # 150 MHz bandwidth
        center_freq=77e9,            # 77 GHz center frequency
        num_rx=4,                    # 4 receive antennas
        num_tx=2,                    # 2 transmit antennas
        max_targets=3,               # Maximum 3 targets per frame
        snr_min=0,                   # Minimum SNR in dB (more challenging)
        snr_max=15,                  # Maximum SNR in dB
        signal_type=best_signal_type,
        apply_realistic_effects=True  # Apply realistic radar effects
    )
    
    # Create radar processor for the best signal type
    processor = RadarProcessing(
        num_range_bins=realistic_test_dataset.num_range_bins,
        num_doppler_bins=realistic_test_dataset.num_doppler_bins,
        sample_rate=realistic_test_dataset.sample_rate,
        chirp_duration=realistic_test_dataset.chirp_duration,
        num_chirps=realistic_test_dataset.num_chirps,
        bandwidth=realistic_test_dataset.bandwidth,
        center_freq=realistic_test_dataset.center_freq,
        signal_type=best_signal_type
    )
    
    # Load the best model
    device, _ = get_device()
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # Check if the model was trained on time-domain data
    use_time_domain = checkpoint.get('use_time_domain', False)
    
    if use_time_domain:
        # Get the shape of time-domain data
        time_shape = realistic_test_dataset[0]['time_domain'].shape
        model = RadarTimeToFreqNet(
            num_rx=time_shape[0],
            num_chirps=time_shape[1],
            samples_per_chirp=time_shape[2],
            out_doppler_bins=realistic_test_dataset[0]['labels'].shape[0],
            out_range_bins=realistic_test_dataset[0]['labels'].shape[1]
        ).to(device)
    else:
        model = RadarNet().to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create directory for realistic test results
    realistic_dir = os.path.join(results_dir, 'realistic_test')
    os.makedirs(realistic_dir, exist_ok=True)
    
    # Process and visualize each sample
    for i in range(min(10, len(realistic_test_dataset))):
        sample = realistic_test_dataset[i]
        
        # Get model input based on model type
        if use_time_domain:
            model_input = torch.from_numpy(sample['time_domain']).unsqueeze(0).to(device)
        else:
            model_input = torch.from_numpy(sample['feature_2d']).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = model(model_input)
            prediction = (output > 0.5).float()[0].cpu().numpy()
        
        # Get conventional radar processing result
        complex_data = sample['time_domain'][:, :, :, 0] + 1j * sample['time_domain'][:, :, :, 1]
        rd_map = processor.time_to_range_doppler(complex_data)
        detected_targets = processor.detect_targets(rd_map, threshold=0.15, min_area=2)
        
        # Create visualization comparing ML vs conventional detection
        plt.figure(figsize=(15, 10))
        
        # Plot range-Doppler map
        plt.subplot(221)
        rd_magnitude = np.sqrt(sample['feature_2d'][0]**2 + sample['feature_2d'][1]**2)
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)
        plt.imshow(rd_db, aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Range-Doppler Map')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Plot ground truth
        plt.subplot(222)
        plt.imshow(sample['labels'][:,:,0], aspect='auto', cmap='gray')
        plt.colorbar(label='Target Presence')
        plt.title('Ground Truth')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Add target markers
        for target in realistic_test_dataset.target_info[i]:
            range_bin = int(target['distance'] / realistic_test_dataset.range_resolution)
            doppler_bin = int(realistic_test_dataset.num_doppler_bins/2 + target['velocity'] / realistic_test_dataset.velocity_resolution)
            
            if (0 <= range_bin < realistic_test_dataset.num_range_bins and 
                0 <= doppler_bin < realistic_test_dataset.num_doppler_bins):
                plt.plot(range_bin, doppler_bin, 'ro', markersize=8)
                plt.text(range_bin + 1, doppler_bin + 1, 
                      f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                      color='white', fontsize=8, backgroundcolor='black')
        
        # Plot ML detection
        plt.subplot(223)
        plt.imshow(prediction[:,:,0], aspect='auto', cmap='gray')
        plt.colorbar(label='ML Detection')
        plt.title(f'ML Detection ({best_signal_type} Model)')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Plot conventional detection
        plt.subplot(224)
        conventional_mask = np.zeros((realistic_test_dataset.num_doppler_bins, realistic_test_dataset.num_range_bins))
        for target in detected_targets:
            conventional_mask[target['doppler_bin'], target['range_bin']] = 1
        plt.imshow(conventional_mask, aspect='auto', cmap='gray')
        plt.colorbar(label='Conventional Detection')
        plt.title('Conventional CFAR Detection')
        plt.xlabel('Range Bin')
        plt.ylabel('Doppler Bin')
        
        # Add detected targets
        for target in detected_targets:
            plt.plot(target['range_bin'], target['doppler_bin'], 'bo', markersize=8)
            plt.text(target['range_bin'] + 1, target['doppler_bin'] + 1, 
                  f"R: {target['range_bin'] * realistic_test_dataset.range_resolution:.1f}m\nV: {(target['doppler_bin'] - realistic_test_dataset.num_doppler_bins/2) * realistic_test_dataset.velocity_resolution:.1f}m/s", 
                  color='white', fontsize=8, backgroundcolor='black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(realistic_dir, f'realistic_test_sample_{i}.pdf'))
        plt.close()
    
    print(f"Realistic test results saved to {realistic_dir}")
    print("\nRadar detection model training, evaluation, and comparison complete!")

def compare_signal_types(base_path='data/radar_comparison', 
                         signal_types=['FMCW', 'OFDM', 'Sine'],
                         num_samples=50,
                         visualize_samples=5):
    """
    Generate and compare different radar signal types
    
    Args:
        base_path: Base path to save comparison data
        signal_types: List of signal types to compare
        num_samples: Number of samples to generate for each type
        visualize_samples: Number of samples to visualize
    """
    
    print(f"Comparing {len(signal_types)} different radar signal types...")
    
    # Create base directory
    os.makedirs(base_path, exist_ok=True)
    
    # Common parameters for all datasets
    common_params = {
        'num_samples': num_samples,
        'num_range_bins': 64,
        'num_doppler_bins': 12,
        'sample_rate': 3e6,
        'chirp_duration': 50e-6,  # 50 μs
        'num_chirps': 12,
        'bandwidth': 150e6,  # 150 MHz
        'center_freq': 77e9,  # 77 GHz (automotive radar)
        'num_rx': 4,
        'num_tx': 2,
        'max_targets': 5,
        'snr_min': 5,
        'snr_max': 20,
        'apply_realistic_effects': True,
        'drawfig': True
    }
    
    # Generate datasets for each signal type
    datasets = {}
    for signal_type in signal_types:
        print(f"\nGenerating dataset for {signal_type} radar...")
        
        # Create dataset with this signal type
        save_path = f"{base_path}/{signal_type.lower()}"
        dataset = RadarDataset(
            **common_params,
            signal_type=signal_type,
            signal_freq=1e6,
            save_path=save_path
        )
        
        # Store dataset
        datasets[signal_type] = dataset
        
        # Print dataset information
        print_dataset_info(dataset)
    
    # Compare range-Doppler maps
    compare_range_doppler_maps(datasets, base_path, visualize_samples)
    
    # Compare detection performance
    compare_detection_performance(datasets, base_path)
    
    # Compare processing time
    compare_processing_time(datasets, base_path)
    
    print(f"\nComparison complete. Results saved to {base_path}")
    
    return datasets

def compare_range_doppler_maps(datasets, base_path, num_samples=5):
    """Compare range-Doppler maps from different signal types"""
    print("\nComparing range-Doppler maps...")
    
    # Create directory for comparison figures
    os.makedirs(f"{base_path}/comparison", exist_ok=True)
    
    # Get signal types
    signal_types = list(datasets.keys())
    
    # Select random samples to visualize
    sample_indices = np.random.choice(datasets[signal_types[0]].num_samples, 
                                     size=min(num_samples, datasets[signal_types[0]].num_samples), 
                                     replace=False)
    
    # Compare each sample
    for idx in sample_indices:
        # Create figure
        fig, axs = plt.subplots(len(signal_types), 2, figsize=(15, 5*len(signal_types)))
        
        # Process each signal type
        for i, signal_type in enumerate(signal_types):
            dataset = datasets[signal_type]
            
            # Get range-Doppler map
            rd_map = dataset.range_doppler_maps[idx]
            rd_magnitude = np.sqrt(rd_map[0]**2 + rd_map[1]**2)
            
            # Get target mask
            target_mask = dataset.target_masks[idx, :, :, 0]
            
            # Get target info
            target_info = dataset.target_info[idx]
            
            # Plot range-Doppler map
            if len(signal_types) == 1:
                ax1, ax2 = axs
            else:
                ax1, ax2 = axs[i]
                
            # Plot range-Doppler map
            rd_db = 20 * np.log10(rd_magnitude + 1e-10)
            im1 = ax1.imshow(rd_db, aspect='auto', cmap='jet')
            ax1.set_title(f'{signal_type} Range-Doppler Map')
            ax1.set_xlabel('Range Bin')
            ax1.set_ylabel('Doppler Bin')
            plt.colorbar(im1, ax=ax1)
            
            # Plot target mask
            im2 = ax2.imshow(target_mask, aspect='auto', cmap='hot')
            ax2.set_title(f'{signal_type} Target Mask')
            ax2.set_xlabel('Range Bin')
            ax2.set_ylabel('Doppler Bin')
            plt.colorbar(im2, ax=ax2)
            
            # Add target annotations
            for target in target_info:
                range_bin = int(target['distance'] / dataset.range_resolution)
                doppler_bin = int(dataset.num_doppler_bins/2 + target['velocity'] / dataset.velocity_resolution)
                
                if (0 <= range_bin < dataset.num_range_bins and 
                    0 <= doppler_bin < dataset.num_doppler_bins):
                    # Add markers to both plots
                    ax1.plot(range_bin, doppler_bin, 'wo', markersize=8, markeredgecolor='black')
                    ax2.plot(range_bin, doppler_bin, 'wo', markersize=8, markeredgecolor='black')
                    
                    # Add target information text
                    ax1.text(range_bin + 2, doppler_bin, 
                          f"R: {target['distance']:.1f}m\nV: {target['velocity']:.1f}m/s", 
                          color='white', fontsize=8, backgroundcolor='black')
        
        # Add sample information
        plt.suptitle(f'Sample {idx}: Comparison of Different Radar Signal Types')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{base_path}/comparison/sample_{idx}_comparison{IMG_FORMAT}")
        plt.close()
    
    print(f"Range-Doppler map comparison saved to {base_path}/comparison/")

def compare_detection_performance(datasets, base_path):
    """Compare target detection performance for different signal types"""
    print("\nComparing target detection performance...")
    
    # Create directory for results
    os.makedirs(f"{base_path}/results", exist_ok=True)
    
    # Get signal types
    signal_types = list(datasets.keys())
    
    # Initialize metrics
    detection_rates = {}
    false_alarms = {}
    
    # Process each signal type
    for signal_type in signal_types:
        dataset = datasets[signal_type]
        
        # Create radar processor with same parameters
        processor = RadarProcessing(
            num_range_bins=dataset.num_range_bins,
            num_doppler_bins=dataset.num_doppler_bins,
            sample_rate=dataset.sample_rate,
            chirp_duration=dataset.chirp_duration,
            num_chirps=dataset.num_chirps,
            bandwidth=dataset.bandwidth,
            center_freq=dataset.center_freq,
            signal_type=signal_type
        )
        
        # Process all samples
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range(dataset.num_samples):
            # Get complex data
            complex_data = dataset.time_domain_data[i, :, :, :, 0] + 1j * dataset.time_domain_data[i, :, :, :, 1]
            
            # Process to range-Doppler map
            rd_map = processor.time_to_range_doppler(complex_data)
            
            # Detect targets
            detected_targets = processor.detect_targets(rd_map, threshold=0.15, min_area=2)
            
            # Get ground truth targets
            true_targets = dataset.target_info[i]
            
            # Count matches (using a distance threshold)
            matched_targets = 0
            for true_target in true_targets:
                # Convert to range/doppler bins
                true_range_bin = int(true_target['distance'] / dataset.range_resolution)
                true_doppler_bin = int(dataset.num_doppler_bins/2 + true_target['velocity'] / dataset.velocity_resolution)
                
                # Check if any detected target matches
                found_match = False
                for detected_target in detected_targets:
                    # Calculate distance in bin space
                    range_diff = abs(detected_target['range_bin'] - true_range_bin)
                    doppler_diff = abs(detected_target['doppler_bin'] - true_doppler_bin)
                    bin_distance = np.sqrt(range_diff**2 + doppler_diff**2)
                    
                    # If close enough, count as match
                    if bin_distance < 3:  # Threshold of 3 bins
                        found_match = True
                        break
                
                if found_match:
                    matched_targets += 1
                else:
                    false_negatives += 1
            
            # Count false positives
            false_positives += max(0, len(detected_targets) - matched_targets)
            true_positives += matched_targets
        
        # Calculate metrics
        if true_positives + false_negatives > 0:
            detection_rate = true_positives / (true_positives + false_negatives)
        else:
            detection_rate = 0
            
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
            
        # Store metrics
        detection_rates[signal_type] = detection_rate
        false_alarms[signal_type] = false_positives / dataset.num_samples
        
        # Print results
        print(f"{signal_type} Detection Rate: {detection_rate:.2f}")
        print(f"{signal_type} Precision: {precision:.2f}")
        print(f"{signal_type} False Alarms per Frame: {false_alarms[signal_type]:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Plot detection rates
    plt.subplot(1, 2, 1)
    plt.bar(signal_types, [detection_rates[st] for st in signal_types])
    plt.title('Detection Rate Comparison')
    plt.ylabel('Detection Rate')
    plt.ylim(0, 1)
    
    # Plot false alarm rates
    plt.subplot(1, 2, 2)
    plt.bar(signal_types, [false_alarms[st] for st in signal_types])
    plt.title('False Alarms per Frame')
    plt.ylabel('False Alarms')
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/detection_performance{IMG_FORMAT}")
    plt.close()
    
    print(f"Detection performance comparison saved to {base_path}/results/")

def compare_processing_time(datasets, base_path):
    """Compare processing time for different signal types"""
    print("\nComparing processing time...")
    
    # Get signal types
    signal_types = list(datasets.keys())
    
    # Initialize timing results
    processing_times = {}
    
    # Process each signal type
    for signal_type in signal_types:
        dataset = datasets[signal_type]
        
        # Create radar processor with same parameters
        processor = RadarProcessing(
            num_range_bins=dataset.num_range_bins,
            num_doppler_bins=dataset.num_doppler_bins,
            sample_rate=dataset.sample_rate,
            chirp_duration=dataset.chirp_duration,
            num_chirps=dataset.num_chirps,
            bandwidth=dataset.bandwidth,
            center_freq=dataset.center_freq,
            signal_type=signal_type
        )
        
        # Measure processing time for 10 samples
        num_test_samples = min(10, dataset.num_samples)
        total_time = 0
        
        for i in range(num_test_samples):
            # Get complex data
            complex_data = dataset.time_domain_data[i, :, :, :, 0] + 1j * dataset.time_domain_data[i, :, :, :, 1]
            
            # Measure processing time
            start_time = time.time()
            _ = processor.time_to_range_doppler(complex_data)
            end_time = time.time()
            
            # Add to total
            total_time += (end_time - start_time)
        
        # Calculate average
        avg_time = total_time / num_test_samples
        processing_times[signal_type] = avg_time
        
        print(f"{signal_type} Average Processing Time: {avg_time*1000:.2f} ms per frame")
    
    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.bar(signal_types, [processing_times[st]*1000 for st in signal_types])
    plt.title('Processing Time Comparison')
    plt.ylabel('Processing Time (ms)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/processing_time{IMG_FORMAT}")
    plt.close()
    
    print(f"Processing time comparison saved to {base_path}/results/")

# Add this to the main function to run the comparison
if __name__ == '__main__':
    train_val()


    # compare_signal_types(base_path='data/radar_comparison', 
    #                      signal_types=['FMCW', 'OFDM', 'Sine', 'OFDM_FMCW', 'Sine_FMCW'],
    #                      num_samples=50,
    #                      visualize_samples=5)

# if __name__ == '__main__':
#     train_val()