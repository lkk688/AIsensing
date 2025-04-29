import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import math
import torch.nn.functional as F

# Import custom modules
from AIradar_datasetraytracing import RayTracingRadarDataset
from AIradar_processing import RadarProcessing
from AIradar_trainv3 import RadarNet #RadarTimeToFreqNet 

class RadarTimeToFreqNet(nn.Module):
    def __init__(self, num_rx=2, num_chirps=12, samples_per_chirp=20, out_doppler_bins=12, out_range_bins=64):
        super(RadarTimeToFreqNet, self).__init__()
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.out_doppler_bins = out_doppler_bins
        self.out_range_bins = out_range_bins

        # Learnable CNN-based time-to-frequency transformation
        self.time_encoder = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, out_doppler_bins, out_range_bins))
        )

        # Attention mechanism for temporal relationships
        self.chirp_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            kdim=self.samples_per_chirp,
            vdim=self.samples_per_chirp
        )
        
        # Final projection layers
        self.projection = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.radar_net = RadarNet(in_channels=2, out_channels=1)

    def time_to_frequency(self, x):
        # x shape: [batch, num_rx, num_chirps, samples_per_chirp, 2]
        batch_size = x.shape[0]
        
        # Combine RX and chirp dimensions
        x = x.view(batch_size * self.num_rx, self.num_chirps, self.samples_per_chirp, 2)
        x = x.permute(0, 3, 1, 2)  # [batch*rx, 2, chirps, samples]
        
        # Add channel dimension for 3D conv
        x = x.unsqueeze(2)  # [batch*rx, 2, 1, chirps, samples]
        
        # Process through encoder
        x = self.time_encoder(x)  # [batch*rx, 128, 1, doppler_bins, range_bins]
        x = x.squeeze(2)  # [batch*rx, 128, doppler_bins, range_bins]
        
        # Apply attention across chirps
        x_attn = x.view(batch_size, self.num_rx, 128, self.out_doppler_bins, self.out_range_bins)
        x_attn = x_attn.permute(0, 3, 1, 2, 4)  # [batch, doppler, rx, 128, range]
        x_attn = x_attn.reshape(-1, self.num_rx, 128, self.out_range_bins)
        attn_out, _ = self.chirp_attention(x_attn, x_attn, x_attn)
        x = attn_out.reshape(batch_size, self.out_doppler_bins, self.num_rx, 128, self.out_range_bins)
        x = x.permute(0, 2, 3, 1, 4)  # [batch, rx, 128, doppler, range]
        
        # Project to final dimensions
        x = self.projection(x.reshape(-1, 128, self.out_doppler_bins, self.out_range_bins))
        x = x.reshape(batch_size, self.num_rx, 2, self.out_doppler_bins, self.out_range_bins)
        x = x.mean(dim=1)  # Combine RX channels
        
        return x.permute(0, 1, 3, 2)  # [batch, 2, doppler, range]

    def forward(self, x):
        rd_map = self.time_to_frequency(x)
        detection = self.radar_net(rd_map)
        return detection

def radar_collate_fn(batch):
    """
    Custom collate function for radar data with variable dimensions
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched tensors with consistent dimensions
    """
    # Extract data from batch
    time_domain_data = [item['time_domain'] for item in batch]
    feature_2d_data = [item['feature_2d'] for item in batch]
    labels_data = [item['labels'] for item in batch]
    target_info = [item['target_info'] for item in batch]
    
    # Get maximum dimensions
    max_rx = max(data.shape[0] for data in time_domain_data)
    max_chirps = max(data.shape[1] for data in time_domain_data)
    max_samples = max(data.shape[2] for data in time_domain_data)
    
    max_feature_dim1 = max(data.shape[0] for data in feature_2d_data)
    max_feature_dim2 = max(data.shape[1] for data in feature_2d_data)
    max_feature_dim3 = max(data.shape[2] for data in feature_2d_data)
    
    max_label_dim1 = max(data.shape[0] for data in labels_data)
    max_label_dim2 = max(data.shape[1] for data in labels_data)
    max_label_dim3 = max(data.shape[2] for data in labels_data)
    
    # Create padded batches
    batch_size = len(batch)
    
    # Pad time domain data
    time_domain_batch = torch.zeros(batch_size, max_rx, max_chirps, max_samples, 2)
    for i, data in enumerate(time_domain_data):
        rx, chirps, samples, channels = data.shape
        time_domain_batch[i, :rx, :chirps, :samples, :] = torch.tensor(data, dtype=torch.float32)
    
    # Pad feature 2D data
    feature_2d_batch = torch.zeros(batch_size, max_feature_dim1, max_feature_dim2, max_feature_dim3)
    for i, data in enumerate(feature_2d_data):
        dim1, dim2, dim3 = data.shape
        feature_2d_batch[i, :dim1, :dim2, :dim3] = torch.tensor(data, dtype=torch.float32)
    
    # Pad labels data
    labels_batch = torch.zeros(batch_size, max_label_dim1, max_label_dim2, max_label_dim3)
    for i, data in enumerate(labels_data):
        dim1, dim2, dim3 = data.shape
        labels_batch[i, :dim1, :dim2, :dim3] = torch.tensor(data, dtype=torch.float32)
    
    # Create batched dictionary
    batched_sample = {
        'time_domain': time_domain_batch,
        'feature_2d': feature_2d_batch,
        'labels': labels_batch,
        'target_info': target_info
    }
    
    return batched_sample

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class RadarTransformer_basic(nn.Module):
    def __init__(self, num_rx=4, num_chirps=32, samples_per_chirp=400, 
                 out_doppler_bins=128, out_range_bins=256, num_heads=8, 
                 dim_feedforward=2048, num_encoder_layers=6, dropout=0.1):
        """
        Transformer-based neural network for radar signal processing
        
        Args:
            num_rx: Number of receiver antennas
            num_chirps: Number of chirps in a frame
            samples_per_chirp: Number of time samples per chirp
            out_doppler_bins: Number of Doppler bins in output
            out_range_bins: Number of range bins in output
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            num_encoder_layers: Number of encoder layers
            dropout: Dropout rate
        """
        super(RadarTransformer, self).__init__()
        
        # Store dimensions
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.out_doppler_bins = out_doppler_bins
        self.out_range_bins = out_range_bins
        
        # Calculate input sequence length and embedding dimension
        self.seq_len = num_rx * num_chirps
        self.embed_dim = samples_per_chirp * 2  # *2 for I/Q data
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding1(self.embed_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Output projection to range-Doppler map
        self.output_projection = nn.Sequential(
            nn.Linear(self.embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_doppler_bins * out_range_bins),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Time-domain data [batch, num_rx, num_chirps, samples_per_chirp, 2]
        
        Returns:
            Detection map [batch, out_doppler_bins, out_range_bins, 1]
        """
        batch_size = x.shape[0]
        
        # Reshape input: [batch, num_rx, num_chirps, samples_per_chirp, 2] -> [batch, seq_len, embed_dim]
        # Combine I/Q data and flatten
        x = x.reshape(batch_size, self.num_rx * self.num_chirps, -1)
        
        # Project input to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling across sequence dimension
        x = torch.mean(x, dim=1)
        
        # Project to output detection map
        x = self.output_projection(x)
        
        # Reshape to detection map format [batch, out_doppler_bins, out_range_bins, 1]
        x = x.reshape(batch_size, self.out_doppler_bins, self.out_range_bins, 1)
        
        return x


class PositionalEncoding1(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding1, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class RadarTransformer(nn.Module):
    """Transformer-based FMCW Radar Target Detection Network with dimension checks"""
    def __init__(self, num_rx=4, num_chirps=128, samples_per_chirp=400):
        super().__init__()
        d_model = 32
        nhead =4
        multiheadAttention_head=2 #4
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        
        # Input processing (I/Q channels)
        self.input_conv = nn.Conv2d(num_rx * 2, d_model, kernel_size=(3,3), padding=1)
        self.position_enc = PositionalEncoding2D(d_model)
        
        # Cross-Talk Suppression
        self.crosstalk_attn = nn.MultiheadAttention(d_model, multiheadAttention_head)
        
        # Transformer Encoder with dimension preservation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, multiheadAttention_head)
        
        # Detection Head with dimension validation
        self.detection_head = nn.Sequential(
            nn.Conv2d(d_model, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input: [batch, rx, chirps, samples, 2]
        batch_size, num_rx, num_chirps, samples_per_chirp, channels = x.shape
        
        # Validate input dimensions
        assert num_rx == self.num_rx, f"Expected {self.num_rx} RX channels, got {num_rx}"
        assert num_chirps == self.num_chirps, f"Expected {self.num_chirps} chirps, got {num_chirps}"
        assert samples_per_chirp == self.samples_per_chirp, f"Expected {self.samples_per_chirp} samples/chirp, got {samples_per_chirp}"

        # Reshape for convolution
        x = x.permute(0, 1, 4, 2, 3)  # [batch, rx, 2, chirps, samples]
        x = x.reshape(batch_size, num_rx * 2, num_chirps, samples_per_chirp)
        x = self.input_conv(x)  # [batch, 64, chirps, samples]
        x = self.position_enc(x)

        # Attention processing
        x_flat = x.flatten(2).permute(0, 2, 1)  # [batch, seq_len, features]
        attn_out, _ = self.crosstalk_attn(x_flat, x_flat, x_flat)
        x = attn_out.permute(0, 2, 1).view_as(x)  # Restore original shape

        # Transformer processing with dimension preservation
        x = x.permute(0, 2, 3, 1)  # [batch, chirps, samples, features]
        orig_shape = x.shape
        x = x.reshape(orig_shape[0], -1, orig_shape[3])  # [batch, chirps*samples, 64]
        x = self.transformer(x)
        x = x.reshape(orig_shape)  # Restore original dimensions
        x = x.permute(0, 3, 1, 2)  # [batch, features, chirps, samples]

        # Final detection mask
        detection_mask = self.detection_head(x)
        return detection_mask  # [batch, 1, chirps, samples]


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x):
        batch, channels, height, width = x.size()
        pos_enc = torch.zeros((batch, channels, height, width), device=x.device)
        
        # Create positional encoding components
        y_pos = torch.arange(height, device=x.device).float().view(-1, 1)
        x_pos = torch.arange(width, device=x.device).float().view(1, -1)
        
        # Calculate frequency terms for both dimensions
        max_dim = max(height, width)
        div_term = torch.exp(torch.arange(0, max_dim, device=x.device).float() *
                           (-math.log(10000.0) / max_dim))
        
        # Create separate terms for height and width
        div_term_y = div_term[:height].view(-1, 1)
        div_term_x = div_term[:width].view(1, -1)
        
        # Apply positional encoding to each channel group
        if channels >= 4:
            for i in range(0, channels, 4):
                # Height encodings
                if i < channels:
                    pos_enc[:, i, :, :] = torch.sin(y_pos * div_term_y)
                if i+1 < channels:
                    pos_enc[:, i+1, :, :] = torch.cos(y_pos * div_term_y)
                # Width encodings
                if i+2 < channels:
                    pos_enc[:, i+2, :, :] = torch.sin(x_pos * div_term_x)
                if i+3 < channels:
                    pos_enc[:, i+3, :, :] = torch.cos(x_pos * div_term_x)
        
        return x + pos_enc

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

def train_radar_transformer(output_dir='data/radar_transformer', 
                           num_samples=1000,
                           batch_size=16,
                           num_epochs=50,
                           learning_rate=0.0001,
                           visualize_progress=True,
                           dataset_params=None):
    """
    Train the RadarTransformer model
    
    Args:
        output_dir: Directory to save model and results
        num_samples: Number of samples to generate for training
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        visualize_progress: Whether to generate visualizations during training
        dataset_params: Dictionary of parameters for the dataset
    
    Returns:
        Trained model and training history
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device, useamp = get_device()
    print(f"Using device: {device}")
    
    # Default dataset parameters
    if dataset_params is None:
        dataset_params = {
            'num_range_bins': 256,
            'num_doppler_bins': 128, #128,
            'sample_rate': 3e6,
            'chirp_duration': 50e-6,
            'num_chirps': 32,
            'bandwidth': 150e6,
            'center_freq': 77e9,
            'num_rx': 4,
            'max_targets': 3,
            'snr_min': 5,
            'snr_max': 25
        }
    
    # Create dataset
    print("Creating radar dataset...")
    dataset = RayTracingRadarDataset(
        num_samples=num_samples,
        **dataset_params
    )
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=4, collate_fn=radar_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=4, collate_fn=radar_collate_fn)

    # Get sample dimensions
    sample = dataset[0]
    time_domain_shape = sample['time_domain'].shape
    label_shape = sample['labels'].shape
    
    # Create model
    model_type = "time_to_freq"  # Can make this configurable via command line args
    if model_type == "transformer":
        model = RadarTransformer(
            num_rx=time_domain_shape[0],
            num_chirps=time_domain_shape[1],
            samples_per_chirp=time_domain_shape[2]
        ).to(device)
    elif model_type == "time_to_freq":
        model = RadarTimeToFreqNet(
            num_rx=time_domain_shape[0],
            num_chirps=time_domain_shape[1],
            samples_per_chirp=time_domain_shape[2],
            out_doppler_bins=128,  # Match your dataset's Doppler bin count
            out_range_bins=256     # Match your dataset's range bin count
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Define loss function and optimizer
    # Define loss functions for multi-task learning
    detection_criterion = nn.BCELoss()
    rd_map_criterion = nn.MSELoss()
    
    # Combined loss weights
    detection_weight = 0.7
    rd_map_weight = 0.3

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'detection_accuracy': [],
        'rd_map_loss': [],
        'detection_loss': [],
        'learning_rate': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_rd_loss = 0.0
        train_detection_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Get data
                time_domain = torch.tensor(batch['time_domain'], dtype=torch.float32).to(device)
                labels = torch.tensor(batch['labels'], dtype=torch.float32).to(device)
                feature_2d = torch.tensor(batch['feature_2d'], dtype=torch.float32).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                #rd_map, detection_mask = model(time_domain)
                detection_mask = model(time_domain)
                
                # Calculate losses for each task
                #rd_loss = rd_map_criterion(rd_map, feature_2d)
                detection_loss = detection_criterion(detection_mask, labels)
                
                # Combined loss with weights
                loss = detection_loss #rd_map_weight * rd_loss + detection_weight * detection_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                #train_rd_loss += rd_loss.item()
                train_detection_loss += detection_loss.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    #'rd_loss': rd_loss.item(),
                    'det_loss': detection_loss.item()
                })
        
        # Calculate average training losses
        train_loss /= len(train_loader)
        #train_rd_loss /= len(train_loader)
        train_detection_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rd_loss = 0.0
        val_detection_loss = 0.0
        correct_detections = 0
        total_detections = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Get data
                    time_domain = torch.tensor(batch['time_domain'], dtype=torch.float32).to(device)
                    labels = torch.tensor(batch['labels'], dtype=torch.float32).to(device)
                    feature_2d = torch.tensor(batch['feature_2d'], dtype=torch.float32).to(device)
                    
                    # Forward pass
                    #rd_map, detection_mask = model(time_domain)
                    detection_mask = model(time_domain)
                    
                    # Calculate losses for each task
                    #rd_loss = rd_map_criterion(rd_map, feature_2d)
                    detection_loss = detection_criterion(detection_mask, labels)
                    
                    # Combined loss with weights
                    loss = detection_loss #rd_map_weight * rd_loss + detection_weight * detection_loss
                    
                    # Update statistics
                    val_loss += loss.item()
                    #val_rd_loss += rd_loss.item()
                    val_detection_loss += detection_loss.item()
                    
                    # Calculate detection accuracy
                    predictions = (detection_mask > 0.5).float()
                    correct_detections += (predictions == labels).sum().item()
                    total_detections += labels.numel()
                    
                    pbar.set_postfix({
                        'loss': loss.item(),
                        #'rd_loss': rd_loss.item(),
                        'det_loss': detection_loss.item()
                    })
        
        # Calculate average validation loss and detection accuracy
        val_loss /= len(val_loader)
        #val_rd_loss /= len(val_loader)
        val_detection_loss /= len(val_loader)
        detection_accuracy = correct_detections / total_detections
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['detection_accuracy'].append(detection_accuracy)
        #history['rd_map_loss'].append(val_rd_loss)
        history['detection_loss'].append(val_detection_loss)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              #f"RD Loss: {val_rd_loss:.6f}, "
              f"Det Loss: {val_detection_loss:.6f}, "
              f"Detection Accuracy: {detection_accuracy:.4f}, "
              f"LR: {current_lr:.8f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'detection_accuracy': detection_accuracy,
                #'rd_map_loss': val_rd_loss,
                'detection_loss': val_detection_loss,
                'history': history
            }, os.path.join(output_dir, 'best_radar_transformer.pth'))
            print(f"Saved best model with validation loss: {val_loss:.6f}")
        
        # Generate visualizations
        if visualize_progress and (epoch % 5 == 0 or epoch == num_epochs - 1):
            #visualize_results(model, val_dataset, device, output_dir, epoch)
            visualize_radar_results(model, val_dataset, device, output_dir, epoch=epoch, is_test=False)
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'detection_accuracy': detection_accuracy,
        'rd_map_loss': val_rd_loss,
        'detection_loss': val_detection_loss,
        'history': history,
        'training_time': training_time
    }, os.path.join(output_dir, 'final_radar_transformer.pth'))
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    return model, history


def visualize_radar_results(model, dataset, device, output_dir, epoch=None, is_test=False):
    """
    Generate visualizations of model predictions
    
    Args:
        model: Trained model
        dataset: Dataset (validation or test)
        device: Device to run inference on
        output_dir: Directory to save visualizations
        epoch: Current epoch number (only used for training visualizations)
        is_test: Whether this is for test data visualization
    """
    # Create visualization directory
    if is_test:
        vis_dir = os.path.join(output_dir, 'test_visualizations')
    else:
        vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Select random samples for visualization
    num_samples = min(10 if is_test else 5, len(dataset))
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get sample
            sample = dataset[idx]
            
            # Prepare input
            time_domain = torch.tensor(sample['time_domain'], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get ground truth
            ground_truth_mask = sample['labels']
            ground_truth_rd = sample['feature_2d']
            
            # Get model prediction
            #rd_map_pred, detection_mask_pred = model(time_domain)
            detection_mask_pred = model(time_domain)
            #rd_map_pred = rd_map_pred.cpu().numpy()[0]
            detection_mask_pred = detection_mask_pred.cpu().numpy()[0]
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            # Plot ground truth detection mask
            plt.subplot(2, 3, 1)
            plt.imshow(ground_truth_mask[:, :, 0], aspect='auto', cmap='viridis')
            plt.colorbar(label='Target Presence')
            plt.title('Ground Truth Detection Mask')
            plt.xlabel('Range Bin')
            plt.ylabel('Doppler Bin')
            
            # Plot predicted detection mask
            plt.subplot(2, 3, 2)
            plt.imshow(detection_mask_pred[0], aspect='auto', cmap='viridis')
            plt.colorbar(label='Predicted Probability')
            plt.title('Predicted Detection Mask')
            plt.xlabel('Range Bin')
            plt.ylabel('Doppler Bin')
            
            # Plot thresholded detection mask
            plt.subplot(2, 3, 3)
            plt.imshow((detection_mask_pred[0] > 0.5).astype(float), aspect='auto', cmap='viridis')
            plt.colorbar(label='Thresholded Prediction')
            plt.title('Thresholded Detection Mask')
            plt.xlabel('Range Bin')
            plt.ylabel('Doppler Bin')
            
            # Plot ground truth range-Doppler map (magnitude)
            plt.subplot(2, 3, 4)
            # Compute magnitude from I/Q components
            gt_magnitude = np.sqrt(ground_truth_rd[0]**2 + ground_truth_rd[1]**2)
            plt.imshow(gt_magnitude, aspect='auto', cmap='jet')
            plt.colorbar(label='Magnitude')
            plt.title('Ground Truth Range-Doppler Map')
            plt.xlabel('Range Bin')
            plt.ylabel('Doppler Bin')
            
            # Plot predicted range-Doppler map (magnitude)
            # plt.subplot(2, 3, 5)
            # # Compute magnitude from I/Q components
            # pred_magnitude = np.sqrt(rd_map_pred[0]**2 + rd_map_pred[1]**2)
            # plt.imshow(pred_magnitude, aspect='auto', cmap='jet')
            # plt.colorbar(label='Magnitude')
            # plt.title('Predicted Range-Doppler Map')
            # plt.xlabel('Range Bin')
            # plt.ylabel('Doppler Bin')
            
            # Plot error between ground truth and predicted range-Doppler maps
            plt.subplot(2, 3, 6)
            error_map = np.abs(gt_magnitude - pred_magnitude)
            plt.imshow(error_map, aspect='auto', cmap='hot')
            plt.colorbar(label='Absolute Error')
            plt.title('Range-Doppler Map Error')
            plt.xlabel('Range Bin')
            plt.ylabel('Doppler Bin')
            
            # Add target markers if available
            if 'target_info' in sample:
                for target in sample['target_info']:
                    for ax_idx in range(1, 7):
                        plt.subplot(2, 3, ax_idx)
                        plt.plot(target['range_bin'], target['doppler_bin'], 'ro', markersize=5)
            
            plt.tight_layout()
            
            # Save with appropriate filename
            if is_test:
                filename = f'test_sample_{i}.png'
            else:
                filename = f'epoch_{epoch}_sample_{i}.png'
                
            plt.savefig(os.path.join(vis_dir, filename))
            plt.close()


def plot_training_history(history, output_dir):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot detection accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['detection_accuracy'], label='Detection Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Detection Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def test_radar_transformer(model_path, output_dir, test_dataset=None, num_test_samples=100):
    """
    Test the RadarTransformer model
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to save test results
        test_dataset: Test dataset (if None, create a new one)
        num_test_samples: Number of test samples to generate if test_dataset is None
    
    Returns:
        Dictionary containing test metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create test dataset if not provided
    if test_dataset is None:
        print("Creating test dataset...")
        test_dataset = RayTracingRadarDataset(
            num_samples=num_test_samples,
            drawfig=False
        )
    
    # Get sample dimensions
    sample = test_dataset[0]
    time_domain_shape = sample['time_domain'].shape
    label_shape = sample['labels'].shape
    
    # Create model
    model = RadarTransformer(
        num_rx=time_domain_shape[0],
        num_chirps=time_domain_shape[1],
        samples_per_chirp=time_domain_shape[2]
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create data loader
    #test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    # In your test_radar_transformer function
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                        num_workers=4, collate_fn=radar_collate_fn)

    # Initialize metrics
    test_loss = 0.0
    rd_map_loss = 0.0
    detection_loss = 0.0
    correct_detections = 0
    total_detections = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Define loss functions
    detection_criterion = nn.BCELoss()
    rd_map_criterion = nn.MSELoss()
    
    # Combined loss weights (should match training weights)
    detection_weight = 0.7
    rd_map_weight = 0.3
    
    # Test loop
    print("Testing model...")
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Get data
                time_domain = torch.tensor(batch['time_domain'], dtype=torch.float32).to(device)
                labels = torch.tensor(batch['labels'], dtype=torch.float32).to(device)
                feature_2d = torch.tensor(batch['feature_2d'], dtype=torch.float32).to(device)
                
                # Forward pass - now returns two outputs
                rd_map, detection_mask = model(time_domain)
                
                # Calculate losses for each task
                rd_loss = rd_map_criterion(rd_map, feature_2d)
                detection_loss_val = detection_criterion(detection_mask, labels)
                
                # Combined loss with weights
                loss = rd_map_weight * rd_loss + detection_weight * detection_loss_val
                
                # Update statistics
                test_loss += loss.item()
                rd_map_loss += rd_loss.item()
                detection_loss += detection_loss_val.item()
                
                # Calculate detection metrics
                predictions = (detection_mask > 0.5).float()
                correct_detections += (predictions == labels).sum().item()
                total_detections += labels.numel()
                
                # Calculate precision and recall metrics
                true_positives += ((predictions == 1) & (labels == 1)).sum().item()
                false_positives += ((predictions == 1) & (labels == 0)).sum().item()
                false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'rd_loss': rd_loss.item(),
                    'det_loss': detection_loss_val.item()
                })
    
    # Calculate average test loss and metrics
    test_loss /= len(test_loader)
    rd_map_loss /= len(test_loader)
    detection_loss /= len(test_loader)
    detection_accuracy = correct_detections / total_detections
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Print test summary
    print(f"Test Loss: {test_loss:.6f}")
    print(f"RD Map Loss: {rd_map_loss:.6f}")
    print(f"Detection Loss: {detection_loss:.6f}")
    print(f"Detection Accuracy: {detection_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Save test metrics
    test_metrics = {
        'test_loss': test_loss,
        'rd_map_loss': rd_map_loss,
        'detection_loss': detection_loss,
        'detection_accuracy': detection_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    # Generate visualizations for a few test samples
    #visualize_test_results(model, test_dataset, device, output_dir)
    visualize_radar_results(model, test_dataset, device, output_dir, is_test=True)
    
    return test_metrics




if __name__ == "__main__":
    import math  # Required for PositionalEncoding
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'data/radar_transformer_{timestamp}'
    
    num_range_bins = 256
    num_doppler_bins = 128
    # Train model
    model, history = train_radar_transformer(
        output_dir=output_dir,
        num_samples=1000,
        batch_size=16,
        num_epochs=50,
        learning_rate=0.0001,
        visualize_progress=True,
        dataset_params={
            'num_range_bins': 256,
            'num_doppler_bins': 128,
            'sample_rate': 3e6,
            'chirp_duration': 50e-6,
            'num_chirps': 32,
            'bandwidth': 150e6,
            'center_freq': 77e9,
            'num_rx': 4,
            'max_targets': 3,
            'snr_min': 5,
            'snr_max': 25
        }
    )
    
    # Test model
    # test_metrics = test_radar_transformer(
    #     model_path=os.path.join(output_dir, 'best_radar_transformer.pth'),
    #     output_dir=os.path.join(output_dir, 'test_results'),
    #     num_test_samples=200
    # )
    
    print("Training and testing completed successfully!")