import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import csv

from AIradar_dataset import RadarDataset

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
        
        Args:
            x: Time-domain data [batch, num_rx, num_chirps, samples_per_chirp, 2]
                where the last dimension contains I/Q data
        
        Returns:
            Range-Doppler map [batch, 2, out_doppler_bins, out_range_bins]
                where the first dimension contains real/imaginary parts
        """
        batch_size = x.shape[0]
        
        # Optional: Preprocess time-domain data
        # Permute to [batch, 2, num_rx, num_chirps, samples_per_chirp]
        x_processed = x.permute(0, 4, 1, 2, 3)
        x_processed = x_processed.float()
        x_processed = self.time_preprocess(x_processed)
        # Permute back to [batch, num_rx, num_chirps, samples_per_chirp, 2]
        x_processed = x_processed.permute(0, 2, 3, 4, 1)
        
        # Sum across receivers for simplicity (could be more sophisticated)
        x_summed = x_processed.sum(dim=1)  # [batch, num_chirps, samples_per_chirp, 2]
        
        # Extract real and imaginary parts
        x_real = x_summed[:, :, :, 0]  # [batch, num_chirps, samples_per_chirp]
        x_imag = x_summed[:, :, :, 1]  # [batch, num_chirps, samples_per_chirp]
        
        # Apply range FFT (first dimension)
        range_real = torch.zeros(batch_size, self.num_chirps, self.out_range_bins, device=x.device)
        range_imag = torch.zeros(batch_size, self.num_chirps, self.out_range_bins, device=x.device)
        
        # For each batch and chirp, compute range FFT
        for b in range(batch_size):
            for c in range(self.num_chirps):
                for r in range(self.out_range_bins):
                    # Get FFT weights for this range bin
                    w_real = self.range_fft_weights[:, r, 0]
                    w_imag = self.range_fft_weights[:, r, 1]
                    
                    # Apply weights (complex multiplication)
                    real_sum, imag_sum = self.complex_multiply(
                        x_real[b, c], x_imag[b, c], w_real, w_imag
                    )
                    
                    # Sum to get FFT result
                    range_real[b, c, r] = real_sum.sum()
                    range_imag[b, c, r] = imag_sum.sum()
        
        # Apply Doppler FFT (second dimension)
        rd_real = torch.zeros(batch_size, self.out_doppler_bins, self.out_range_bins, device=x.device)
        rd_imag = torch.zeros(batch_size, self.out_doppler_bins, self.out_range_bins, device=x.device)
        
        # For each batch, compute Doppler FFT
        for b in range(batch_size):
            for d in range(self.out_doppler_bins):
                for r in range(self.out_range_bins):
                    # Get FFT weights for this Doppler bin
                    w_real = self.doppler_fft_weights[:, d, 0]
                    w_imag = self.doppler_fft_weights[:, d, 1]
                    
                    # Apply weights (complex multiplication)
                    real_sum, imag_sum = self.complex_multiply(
                        range_real[b, :, r], range_imag[b, :, r], w_real, w_imag
                    )
                    
                    # Sum to get FFT result
                    rd_real[b, d, r] = real_sum.sum()
                    rd_imag[b, d, r] = imag_sum.sum()
        
        # Stack real and imaginary parts
        rd_map = torch.stack([rd_real, rd_imag], dim=1)  # [batch, 2, out_doppler_bins, out_range_bins]
        
        return rd_map
    
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
    if os.path.exists(hdf5_path) and use_lazy_loading:
        print(f"Loading existing radar dataset from {hdf5_path} with lazy loading")
        train_data = RadarDataset(
            datapath=hdf5_path, 
            training=True, 
            drawfig=visualize_progress,
            use_lazy_loading=True,  # Enable lazy loading
            cache_size=cache_size    # Set cache size
        )
    elif os.path.exists(numpy_path):
        print(f"Loading existing radar dataset from {numpy_path} with memory mapping")
        train_data = RadarDataset(
            datapath=numpy_path, 
            training=True, 
            drawfig=visualize_progress,
            use_memory_mapping=True  # Enable memory mapping
        )
    elif os.path.exists(data_path):
        # Determine loading method based on file extension
        if data_path.endswith('.h5') and use_lazy_loading:
            print(f"Loading existing radar dataset from {data_path} with lazy loading")
            train_data = RadarDataset(
                datapath=data_path, 
                training=True, 
                drawfig=visualize_progress,
                use_lazy_loading=True,
                cache_size=cache_size
            )
        elif data_path.endswith('.npy'):
            print(f"Loading existing radar dataset from {data_path} with memory mapping")
            train_data = RadarDataset(
                datapath=data_path, 
                training=True, 
                drawfig=visualize_progress,
                use_memory_mapping=True
            )
        else:
            print(f"Loading existing radar dataset from {data_path} with standard loading")
            train_data = RadarDataset(
                datapath=data_path, 
                training=True, 
                drawfig=visualize_progress
            )
    else:
        print(f"Generating new radar dataset with {num_samples} samples using {signal_type} signal type")
        print(f"Data will be saved in {data_format} format")
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
        if visualize_progress and epoch % 5 == 0:
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

def test_radar_model(model_path=None, test_data_path=None, output_dir=None):
    device, useamp = get_device(gpuid='0', useamp=False)
    
    # Load test data or create new test data with real device parameters
    if test_data_path and os.path.exists(test_data_path):
        test_dataset = RadarDataset(
            datapath=test_data_path,
            training=False,
            drawfig=True
        )
    else:
        print("Creating new test dataset with real device parameters")
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
            signal_type='OFDM'           # OFDM signal type
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
    # Set output directories for training and results
    output_dir = 'data/radar_training'
    results_dir = 'data/radar_results'
    data_dir = 'data/radar'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Define which signal type to use
    signal_type = 'OFDM'  # Options: 'FMCW', 'OFDM', 'Sine'
    
    # Step 1: Training the radar detection model
    # The model learns to detect targets in range-Doppler maps
    # $P(target|x) = \sigma(f_\theta(x))$ where $f_\theta$ is our neural network
    print("=" * 80)
    print(f"STEP 1: TRAINING RADAR DETECTION MODEL WITH {signal_type} SIGNAL")
    print("=" * 80)
    model, history = train_radar_modelv2(
        output_dir=output_dir, #os.path.join(output_dir, signal_type.lower()),
        num_samples=10000,          # Number of synthetic samples to generate
        batch_size=32,              # Mini-batch size for SGD
        num_epochs=10,              # Number of training epochs
        learning_rate=0.001,        # Initial learning rate
        use_time_domain=True,      # Use range-Doppler maps instead of time domain
        visualize_progress=True,    # Generate visualizations during training
        signal_type=signal_type,     # Use the specified signal type
        data_dir=data_dir
    )
    
    # Step 2: Evaluate the trained model on test data
    # Compute metrics: accuracy, false alarm rate, missed detection rate
    # Accuracy = $\frac{TP + TN}{TP + TN + FP + FN}$
    # False Alarm Rate = $\frac{FP}{FP + TN}$
    # Missed Detection Rate = $\frac{FN}{TP + FN}$
    print("\n" + "=" * 80)
    print(f"STEP 2: EVALUATING RADAR DETECTION MODEL WITH {signal_type} SIGNAL")
    print("=" * 80)
    model_path = os.path.join(output_dir, signal_type.lower(), 'best_radar_model.pth')
    detection_accuracy, false_alarm_rate, missed_detection_rate = test_radar_model(
        model_path=model_path,
        output_dir=os.path.join(results_dir, signal_type.lower())
    )
    
    # Step 3: Generate additional visualizations for analysis
    # Create range-Doppler maps with target overlays and detection results
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING ADDITIONAL VISUALIZATIONS")
    print("=" * 80)
    
    # Create a test dataset for visualization
    test_dataset = RadarDataset(
        num_samples=10,
        training=False,
        drawfig=True,
        # Real device parameters
        sample_rate=3e6,             # 3 MHz sampling rate
        chirp_duration=500e-6,       # 500 microsecond chirp
        num_chirps=128,              # 128 chirps per frame for TDD mode
        bandwidth=500e6,             # 500 MHz bandwidth (matches myradar4.py)
        center_freq=2.1e9,           # 2.1 GHz center frequency
        num_rx=4,                    # 4 receive antennas
        num_tx=1,                    # 1 transmit antenna
        #range_max=30,                # Maximum detection range in meters
        snr_min=5,                   # Minimum SNR in dB
        snr_max=20                   # Maximum SNR in dB
    )
    
    # Load the trained model
    device, _ = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model = RadarNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate detailed visualizations for each sample
    for i, sample in enumerate(tqdm(test_dataset, desc="Generating visualizations")):
        # Prepare input data
        input_data = torch.from_numpy(sample['feature_2d']).unsqueeze(0).to(device)
        target = sample['labels']
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_data)
            prediction = (output > 0.5).float()[0].cpu().numpy()
        
        # Create visualization filename
        viz_filename = os.path.join(results_dir, f'detailed_visualization_{i}.pdf')
        
        # Use the existing visualization function
        visualize_detection(
            sample['feature_2d'],
            target,
            prediction,
            viz_filename
        )
        
        # Create enhanced 3D visualization
        plt.figure(figsize=(15, 10))
        
        # Calculate magnitude from real and imaginary parts
        magnitude = np.sqrt(sample['feature_2d'][0]**2 + sample['feature_2d'][1]**2)
        rd_db = 20*np.log10(magnitude + 1e-10)
        
        # 3D visualization of range-Doppler map with targets
        ax = plt.subplot(111, projection='3d')
        x, y = np.meshgrid(range(magnitude.shape[1]), range(magnitude.shape[0]))
        ax.plot_surface(x, y, rd_db, cmap='jet', alpha=0.8)
        
        # Add target markers
        for y_idx in range(target.shape[0]):
            for x_idx in range(target.shape[1]):
                if target[y_idx, x_idx, 0] > 0.5:
                    ax.scatter([x_idx], [y_idx], [rd_db[y_idx, x_idx] + 5], 
                              color='green', s=50, marker='o', label='Ground Truth')
                if prediction[y_idx, x_idx, 0] > 0.5:
                    ax.scatter([x_idx], [y_idx], [rd_db[y_idx, x_idx] + 2], 
                              color='red', s=30, marker='x', label='Prediction')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        ax.set_title('3D Range-Doppler Map with Targets')
        ax.set_xlabel('Range Bin')
        ax.set_ylabel('Doppler Bin')
        ax.set_zlabel('Magnitude (dB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'3d_visualization_{i}.pdf'))
        plt.close()
        
        print(f"Generated detailed visualization {i+1}/{len(test_dataset)}")
    
    # Step 4: Generate performance summary
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Create summary plot of training history
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    # Loss = $\alpha \cdot BCE(y, \hat{y}) + (1-\alpha) \cdot Dice(y, \hat{y})$
    plt.subplot(221)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot detection accuracy
    # Accuracy = $\frac{TP + TN}{TP + TN + FP + FN}$
    plt.subplot(222)
    plt.plot(history['detection_accuracy'], label='Validation Accuracy')
    plt.axhline(y=detection_accuracy, color='r', linestyle='--', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Detection Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    # $\eta_t = \eta_0 \cdot \text{factor}^{n}$ where n is number of reductions
    plt.subplot(223)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    
    # Plot error rates
    plt.subplot(224)
    metrics = ['False Alarm Rate', 'Missed Detection Rate']
    values = [false_alarm_rate, missed_detection_rate]
    plt.bar(metrics, values, color=['red', 'blue'])
    plt.ylabel('Rate')
    plt.title('Error Metrics')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_summary.pdf'))
    plt.close()
    
    print(f"Performance summary saved to {os.path.join(results_dir, 'performance_summary.pdf')}")
    print("\nRadar detection model training and evaluation complete!")

def compare_signal_types(
    output_dir='data/radar_comparison',
    data_dir='data/radar',
    num_samples=10000,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    use_time_domain=True,
    visualize_progress=True,
    snr_test_levels=None
):
    """
    Train and compare radar detection models using different signal types (FMCW, OFDM, Sine)
    
    Args:
        output_dir: Directory to save comparison results
        data_dir: Directory containing radar data
        num_samples: Number of samples to generate if no dataset exists
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        use_time_domain: Whether to use time-domain data
        visualize_progress: Whether to visualize training progress
        snr_test_levels: List of SNR levels to test (in dB), defaults to [0, 5, 10, 15, 20, 25]
    """
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Define signal types to compare
    signal_types = ['FMCW', 'OFDM', 'Sine']
    
    # Define SNR test levels if not provided
    if snr_test_levels is None:
        snr_test_levels = [0, 5, 10, 15, 20, 25]
    
    # Dictionary to store results
    results = {
        'signal_type': [],
        'snr': [],
        'detection_accuracy': [],
        'false_alarm_rate': [],
        'missed_detection_rate': [],
        'training_time': [],
        'val_loss': []
    }
    
    # Dictionary to store training histories
    histories = {}
    
    # Train models for each signal type
    for signal_type in signal_types:
        print("\n" + "=" * 80)
        print(f"TRAINING MODEL FOR {signal_type} SIGNAL")
        print("=" * 80)
        
        # Record training start time
        start_time = time.time()
        
        # Train model
        model_dir = os.path.join(output_dir, 'models', signal_type.lower())
        os.makedirs(model_dir, exist_ok=True)
        
        model, history = train_radar_modelv2(
            output_dir=model_dir,
            num_samples=num_samples,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_time_domain=use_time_domain,
            visualize_progress=visualize_progress,
            signal_type=signal_type,
            data_dir=data_dir
        )
        
        # Record training end time
        training_time = time.time() - start_time
        
        # Store training history
        histories[signal_type] = history
        
        # Save training time and final validation loss
        results['signal_type'].append(signal_type)
        results['snr'].append('Overall')  # Placeholder for overall metrics
        results['training_time'].append(training_time)
        results['val_loss'].append(history['val_loss'][-1])
        results['detection_accuracy'].append(history['detection_accuracy'][-1])
        results['false_alarm_rate'].append(None)  # Will be filled in during testing
        results['missed_detection_rate'].append(None)  # Will be filled in during testing
        
        print(f"Training completed for {signal_type} in {training_time:.2f} seconds")
    
    # Test models at different SNR levels
    print("\n" + "=" * 80)
    print("TESTING MODELS AT DIFFERENT SNR LEVELS")
    print("=" * 80)
    
    # Dictionary to store performance metrics by SNR
    snr_performance = {
        signal_type: {
            'snr': [],
            'detection_accuracy': [],
            'false_alarm_rate': [],
            'missed_detection_rate': []
        } for signal_type in signal_types
    }
    
    # Test each model at different SNR levels
    for signal_type in signal_types:
        print(f"\nTesting {signal_type} model at different SNR levels...")
        
        # Load the trained model
        model_path = os.path.join(output_dir, 'models', signal_type.lower(), 'best_radar_model.pth')
        
        for snr in tqdm(snr_test_levels, desc=f"Testing {signal_type}"):
            # Create test dataset with specific SNR
            test_dataset = RadarDataset(
                num_samples=100,  # Smaller test set for each SNR level
                training=False,
                drawfig=False,
                # Real device parameters
                sample_rate=3e6,
                chirp_duration=500e-6,
                num_chirps=128,
                bandwidth=500e6,
                center_freq=2.1e9,
                num_rx=4,
                num_tx=1,
                snr_min=snr,
                snr_max=snr,  # Fixed SNR for this test
                signal_type=signal_type
            )
            
            # Test the model
            detection_accuracy, false_alarm_rate, missed_detection_rate = test_radar_model(
                model_path=model_path,
                test_data_path=None,  # Use the dataset we just created
                output_dir=None  # Don't save visualizations for each SNR test
            )
            
            # Store results
            results['signal_type'].append(signal_type)
            results['snr'].append(snr)
            results['detection_accuracy'].append(detection_accuracy)
            results['false_alarm_rate'].append(false_alarm_rate)
            results['missed_detection_rate'].append(missed_detection_rate)
            results['training_time'].append(None)  # Only relevant for overall results
            results['val_loss'].append(None)  # Only relevant for overall results
            
            # Store in SNR performance dictionary
            snr_performance[signal_type]['snr'].append(snr)
            snr_performance[signal_type]['detection_accuracy'].append(detection_accuracy)
            snr_performance[signal_type]['false_alarm_rate'].append(false_alarm_rate)
            snr_performance[signal_type]['missed_detection_rate'].append(missed_detection_rate)
    
    # Create DataFrame for results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'signal_type_comparison_results.csv'), index=False)
    
    # Generate comparison visualizations
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Training Loss Comparison
    plt.figure(figsize=(12, 8))
    for signal_type in signal_types:
        plt.plot(histories[signal_type]['train_loss'], linestyle='-', label=f'{signal_type} Train')
        plt.plot(histories[signal_type]['val_loss'], linestyle='--', label=f'{signal_type} Val')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'visualizations', 'loss_comparison.pdf'))
    plt.close()
    
    # 2. Detection Accuracy vs SNR
    plt.figure(figsize=(12, 8))
    for signal_type in signal_types:
        plt.plot(
            snr_performance[signal_type]['snr'],
            snr_performance[signal_type]['detection_accuracy'],
            marker='o',
            label=signal_type
        )
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Detection Accuracy')
    plt.title('Detection Accuracy vs SNR')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'visualizations', 'accuracy_vs_snr.pdf'))
    plt.close()
    
    # 3. False Alarm Rate vs SNR
    plt.figure(figsize=(12, 8))
    for signal_type in signal_types:
        plt.plot(
            snr_performance[signal_type]['snr'],
            snr_performance[signal_type]['false_alarm_rate'],
            marker='o',
            label=signal_type
        )
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('False Alarm Rate')
    plt.title('False Alarm Rate vs SNR')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'visualizations', 'false_alarm_vs_snr.pdf'))
    plt.close()
    
    # 4. Missed Detection Rate vs SNR
    plt.figure(figsize=(12, 8))
    for signal_type in signal_types:
        plt.plot(
            snr_performance[signal_type]['snr'],
            snr_performance[signal_type]['missed_detection_rate'],
            marker='o',
            label=signal_type
        )
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Missed Detection Rate')
    plt.title('Missed Detection Rate vs SNR')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'visualizations', 'missed_detection_vs_snr.pdf'))
    plt.close()
    
    # 5. Combined Performance Metrics
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Detection Accuracy
    for signal_type in signal_types:
        axes[0].plot(
            snr_performance[signal_type]['snr'],
            snr_performance[signal_type]['detection_accuracy'],
            marker='o',
            label=signal_type
        )
    
    axes[0].set_ylabel('Detection Accuracy')
    axes[0].set_title('Detection Performance vs SNR')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(0, 1)
    
    # False Alarm Rate
    for signal_type in signal_types:
        axes[1].plot(
            snr_performance[signal_type]['snr'],
            snr_performance[signal_type]['false_alarm_rate'],
            marker='o',
            label=signal_type
        )
    
    axes[1].set_ylabel('False Alarm Rate')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(0, 1)
    
    # Missed Detection Rate
    for signal_type in signal_types:
        axes[2].plot(
            snr_performance[signal_type]['snr'],
            snr_performance[signal_type]['missed_detection_rate'],
            marker='o',
            label=signal_type
        )
    
    axes[2].set_xlabel('SNR (dB)')
    axes[2].set_ylabel('Missed Detection Rate')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'combined_performance_vs_snr.pdf'))
    plt.close()
    
    # 6. Create a radar chart for overall performance comparison
    # Filter for overall metrics
    overall_df = results_df[results_df['snr'] == 'Overall'].copy()
    
    # Create radar chart
    plt.figure(figsize=(10, 10))
    
    # Number of variables
    categories = ['Detection Accuracy', 'Training Time (normalized)', 'Validation Loss (normalized)']
    N = len(categories)
    
    # Normalize training time (lower is better)
    max_time = overall_df['training_time'].max()
    overall_df['training_time_normalized'] = 1 - (overall_df['training_time'] / max_time)
    
    # Normalize validation loss (lower is better)
    max_loss = overall_df['val_loss'].max()
    overall_df['val_loss_normalized'] = 1 - (overall_df['val_loss'] / max_loss)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each signal type
    for i, signal_type in enumerate(signal_types):
        values = overall_df[overall_df['signal_type'] == signal_type]
        stats = [
            values['detection_accuracy'].values[0],
            values['training_time_normalized'].values[0],
            values['val_loss_normalized'].values[0]
        ]
        stats += stats[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, stats, linewidth=2, linestyle='solid', label=signal_type)
        ax.fill(angles, stats, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Overall Performance Comparison', size=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'radar_chart_comparison.pdf'))
    plt.close()
    
    # 7. Create detailed SNR vs Performance heatmaps
    # Prepare data for heatmaps
    snr_metrics = results_df[results_df['snr'] != 'Overall'].copy()
    
    # Pivot tables for each metric
    accuracy_pivot = snr_metrics.pivot(index='signal_type', columns='snr', values='detection_accuracy')
    false_alarm_pivot = snr_metrics.pivot(index='signal_type', columns='snr', values='false_alarm_rate')
    missed_detection_pivot = snr_metrics.pivot(index='signal_type', columns='snr', values='missed_detection_rate')
    
    # Create heatmaps
    plt.figure(figsize=(15, 15))
    
    # Detection Accuracy Heatmap
    plt.subplot(3, 1, 1)
    sns.heatmap(accuracy_pivot, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
    plt.title('Detection Accuracy by Signal Type and SNR')
    plt.ylabel('Signal Type')
    
    # False Alarm Rate Heatmap
    plt.subplot(3, 1, 2)
    sns.heatmap(false_alarm_pivot, annot=True, cmap='coolwarm_r', vmin=0, vmax=1, fmt='.3f')
    plt.title('False Alarm Rate by Signal Type and SNR')
    plt.ylabel('Signal Type')
    
    # Missed Detection Rate Heatmap
    plt.subplot(3, 1, 3)
    sns.heatmap(missed_detection_pivot, annot=True, cmap='coolwarm_r', vmin=0, vmax=1, fmt='.3f')
    plt.title('Missed Detection Rate by Signal Type and SNR')
    plt.ylabel('Signal Type')
    plt.xlabel('SNR (dB)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'performance_heatmaps.pdf'))
    plt.close()
    
    # 8. Create a 3D visualization comparing all three signal types
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each signal type
    markers = ['o', '^', 's']
    for i, signal_type in enumerate(signal_types):
        snr = snr_performance[signal_type]['snr']
        accuracy = snr_performance[signal_type]['detection_accuracy']
        false_alarm = snr_performance[signal_type]['false_alarm_rate']
        
        ax.scatter(
            snr, 
            accuracy, 
            false_alarm, 
            marker=markers[i], 
            s=100, 
            label=signal_type,
            alpha=0.7
        )
        
        # Add connecting lines
        ax.plot(
            snr, 
            accuracy, 
            false_alarm, 
            linestyle='-', 
            alpha=0.5
        )
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Detection Accuracy')
    ax.set_zlabel('False Alarm Rate')
    ax.set_title('3D Performance Comparison')
    
    # Set axis limits
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', '3d_performance_comparison.pdf'))
    plt.close()
    
    # 9. Create ROC-like curves (Detection Accuracy vs False Alarm Rate)
    plt.figure(figsize=(12, 8))
    
    for signal_type in signal_types:
        plt.plot(
            snr_performance[signal_type]['false_alarm_rate'],
            snr_performance[signal_type]['detection_accuracy'],
            marker='o',
            label=signal_type
        )
        
        # Add SNR annotations
        for i, snr in enumerate(snr_performance[signal_type]['snr']):
            plt.annotate(
                f"{snr} dB",
                (snr_performance[signal_type]['false_alarm_rate'][i], 
                 snr_performance[signal_type]['detection_accuracy'][i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
    
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Detection Accuracy')
    plt.title('Detection Accuracy vs False Alarm Rate (ROC-like curve)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'visualizations', 'roc_like_curves.pdf'))
    plt.close()
    
    # 10. Create a summary table visualization
    # Prepare summary data
    summary_data = []
    
    for signal_type in signal_types:
        # Get best SNR performance
        best_snr_idx = np.argmax(snr_performance[signal_type]['detection_accuracy'])
        best_snr = snr_performance[signal_type]['snr'][best_snr_idx]
        best_accuracy = snr_performance[signal_type]['detection_accuracy'][best_snr_idx]
        
        # Get worst SNR performance
        worst_snr_idx = np.argmin(snr_performance[signal_type]['detection_accuracy'])
        worst_snr = snr_performance[signal_type]['snr'][worst_snr_idx]
        worst_accuracy = snr_performance[signal_type]['detection_accuracy'][worst_snr_idx]
        
        # Get training time
        training_time = overall_df[overall_df['signal_type'] == signal_type]['training_time'].values[0]
        
        summary_data.append([
            signal_type,
            best_accuracy,
            best_snr,
            worst_accuracy,
            worst_snr,
            training_time / 60  # Convert to minutes
        ])
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=summary_data,
        colLabels=['Signal Type', 'Best Accuracy', 'Best SNR (dB)', 'Worst Accuracy', 'Worst SNR (dB)', 'Training Time (min)'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Performance Summary by Signal Type', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'summary_table.pdf'))
    plt.close()
    
    print(f"Comparison completed. Results saved to {output_dir}")
    return results_df, histories, snr_performance

# Add this to the main function to run the comparison
if __name__ == '__main__':
    train_val()
    # Uncomment to run the signal type comparison
    # compare_signal_types(
    #     output_dir='data/radar_comparison',
    #     data_dir='data/radar',
    #     num_samples=10000,
    #     batch_size=32,
    #     num_epochs=20,  # Reduced for faster comparison
    #     snr_test_levels=[0, 5, 10, 15, 20, 25]  # Test at these SNR levels
    # )

# if __name__ == '__main__':
#     train_val()