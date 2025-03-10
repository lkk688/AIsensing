
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error
import time

# Import the model
from AImodels_joint import DualPurposeTransformer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom dataset for OFDM and Radar data
class OFDMRadarDataset(Dataset):
    def __init__(self, data_path, mode='train', split_ratio=0.8):
        """
        Dataset for OFDM and Radar data.
        
        Args:
            data_path (str): Path to the dataset pickle file.
            mode (str): 'train', 'val', or 'test'.
            split_ratio (float): Ratio of training data.
        """
        # Load the dataset
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Get dataset size (128, 1, 2, 14, 76)
        self.total_samples = len(self.data['ofdm_symbols']) #128
        
        # Split indices
        indices = np.arange(self.total_samples)
        np.random.shuffle(indices)
        
        train_size = int(self.total_samples * split_ratio)
        val_size = int(self.total_samples * (1 - split_ratio) / 2)
        
        if mode == 'train':
            self.indices = indices[:train_size]
        elif mode == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
            
        print(f"Loaded {len(self.indices)} samples for {mode}")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index
        actual_idx = self.indices[idx]
        
        # Get OFDM symbols and radar reflections
        ofdm_symbols = self.data['ofdm_symbols'][actual_idx]  # (128, 1, 2, 14, 76) [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        radar_reflection = self.data['radar_reflection'][actual_idx]  #(128, 1, 2, 14, 76)
        #Radar reflection signals (shape: [batch_size, num_ofdm_symbols, fft_size]).
        # (1, 2, 14, 76)
        
        # Convert to real/imaginary representation
        ofdm_real = np.real(ofdm_symbols)
        ofdm_imag = np.imag(ofdm_symbols)
        ofdm_features = np.stack([ofdm_real, ofdm_imag], axis=0)  # (2, 1, 2, 14, 76) Shape: [2, num_ofdm_symbols, fft_size]
        
        radar_real = np.real(radar_reflection)
        radar_imag = np.imag(radar_reflection)
        radar_features = np.stack([radar_real, radar_imag], axis=0)  # (2, 1, 2, 14, 76) Shape: [2, num_ofdm_symbols, fft_size]
        
        # Get targets
        # For OFDM: the received symbols (for channel estimation/equalization)
        received_symbols = self.data['received_symbols'][actual_idx] #(1, 16, 14, 76)
        received_real = np.real(received_symbols)
        received_imag = np.imag(received_symbols)
        ofdm_target = np.stack([received_real, received_imag], axis=0)
        
        # For Radar: the range-Doppler map
        # Compute range-Doppler map if not already in the dataset
        if 'range_doppler' in self.data:
            range_doppler = self.data['range_doppler'][actual_idx]
        else:
            # Compute 2D FFT for range-Doppler response
            range_doppler = np.fft.fftshift(np.fft.fft2(radar_reflection))
        
        range_doppler_real = np.real(range_doppler)
        range_doppler_imag = np.imag(range_doppler)
        radar_target = np.stack([range_doppler_real, range_doppler_imag], axis=0)
        
        # Convert to tensors
        ofdm_features = torch.from_numpy(ofdm_features).float()
        radar_features = torch.from_numpy(radar_features).float()
        ofdm_target = torch.from_numpy(ofdm_target).float()
        radar_target = torch.from_numpy(radar_target).float()
        
        return {
            'ofdm_features': ofdm_features,
            'radar_features': radar_features,
            'ofdm_target': ofdm_target,
            'radar_target': radar_target,
            'target_info': self.data.get('targets', None)
        }

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

# Training function
def train_model(config):
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set device
    device, useamp = get_device(gpuid='0', useamp=False)
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = OFDMRadarDataset(config['data_path'], mode='train')
    val_dataset = OFDMRadarDataset(config['data_path'], mode='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # Initialize models
    comm_model = DualPurposeTransformer(
        in_channels=2,
        out_channels=2,
        dim=config['model_dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        dropout=config['dropout'],
        mode='comm'
    ).to(device)
    
    radar_model = DualPurposeTransformer(
        in_channels=2,
        out_channels=2,
        dim=config['model_dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        dropout=config['dropout'],
        mode='radar'
    ).to(device)
    
    # Define optimizers
    comm_optimizer = optim.Adam(comm_model.parameters(), lr=config['learning_rate'])
    radar_optimizer = optim.Adam(radar_model.parameters(), lr=config['learning_rate'])
    
    # Define learning rate schedulers
    comm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        comm_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    radar_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        radar_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Define loss functions
    comm_criterion = nn.MSELoss()
    radar_criterion = nn.MSELoss()
    
    # Training history
    history = {
        'comm_train_loss': [],
        'comm_val_loss': [],
        'radar_train_loss': [],
        'radar_val_loss': [],
        'comm_val_nmse': [],
        'radar_val_nmse': []
    }
    
    # Training loop
    best_comm_loss = float('inf')
    best_radar_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Training phase
        comm_model.train()
        radar_model.train()
        
        comm_train_loss = 0.0
        radar_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Training"):
            # Get data 
            ofdm_features = batch['ofdm_features'].to(device) #[32, 2, 1, 2, 14, 76]
            radar_features = batch['radar_features'].to(device) #[32, 2, 1, 2, 14, 76])
            ofdm_target = batch['ofdm_target'].to(device) #[32, 2, 1, 2, 14, 76]
            radar_target = batch['radar_target'].to(device) #[32, 2, 1, 2, 14, 76]
            
            # Train communication model
            comm_optimizer.zero_grad()
            ofdm_output = comm_model(ofdm_features)
            comm_loss = comm_criterion(ofdm_output, ofdm_target)
            comm_loss.backward()
            comm_optimizer.step()
            
            # Train radar model
            radar_optimizer.zero_grad()
            radar_output = radar_model(radar_features)
            radar_loss = radar_criterion(radar_output, radar_target)
            radar_loss.backward()
            radar_optimizer.step()
            
            # Accumulate losses
            comm_train_loss += comm_loss.item()
            radar_train_loss += radar_loss.item()
        
        # Calculate average training losses
        comm_train_loss /= len(train_loader)
        radar_train_loss /= len(train_loader)
        
        # Validation phase
        comm_model.eval()
        radar_model.eval()
        
        comm_val_loss = 0.0
        radar_val_loss = 0.0
        comm_val_nmse = 0.0
        radar_val_nmse = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Validation"):
                # Get data
                ofdm_features = batch['ofdm_features'].to(device)
                radar_features = batch['radar_features'].to(device)
                ofdm_target = batch['ofdm_target'].to(device)
                radar_target = batch['radar_target'].to(device)
                
                # Forward pass for communication model
                ofdm_output = comm_model(ofdm_features)
                comm_loss = comm_criterion(ofdm_output, ofdm_target)
                
                # Forward pass for radar model
                radar_output = radar_model(radar_features)
                radar_loss = radar_criterion(radar_output, radar_target)
                
                # Accumulate losses
                comm_val_loss += comm_loss.item()
                radar_val_loss += radar_loss.item()
                
                # Calculate NMSE (Normalized Mean Squared Error)
                comm_nmse = torch.sum((ofdm_output - ofdm_target)**2) / torch.sum(ofdm_target**2)
                radar_nmse = torch.sum((radar_output - radar_target)**2) / torch.sum(radar_target**2)
                
                comm_val_nmse += comm_nmse.item()
                radar_val_nmse += radar_nmse.item()
                
                # Visualize results for the first batch in the last epoch
                if epoch == config['num_epochs'] - 1 and batch == next(iter(val_loader)):
                    visualize_results(
                        ofdm_features.cpu().numpy(),
                        ofdm_target.cpu().numpy(),
                        ofdm_output.cpu().numpy(),
                        radar_features.cpu().numpy(),
                        radar_target.cpu().numpy(),
                        radar_output.cpu().numpy(),
                        batch['target_info'],
                        config['output_dir']
                    )
        
        # Calculate average validation losses and metrics
        comm_val_loss /= len(val_loader)
        radar_val_loss /= len(val_loader)
        comm_val_nmse /= len(val_loader)
        radar_val_nmse /= len(val_loader)
        
        # Update learning rate schedulers
        comm_scheduler.step(comm_val_loss)
        radar_scheduler.step(radar_val_loss)
        
        # Save best models
        if comm_val_loss < best_comm_loss:
            best_comm_loss = comm_val_loss
            torch.save(comm_model.state_dict(), os.path.join(config['output_dir'], 'best_comm_model.pth'))
            print(f"Saved best communication model with loss: {best_comm_loss:.6f}")
        
        if radar_val_loss < best_radar_loss:
            best_radar_loss = radar_val_loss
            torch.save(radar_model.state_dict(), os.path.join(config['output_dir'], 'best_radar_model.pth'))
            print(f"Saved best radar model with loss: {best_radar_loss:.6f}")
        
        # Update history
        history['comm_train_loss'].append(comm_train_loss)
        history['comm_val_loss'].append(comm_val_loss)
        history['radar_train_loss'].append(radar_train_loss)
        history['radar_val_loss'].append(radar_val_loss)
        history['comm_val_nmse'].append(comm_val_nmse)
        history['radar_val_nmse'].append(radar_val_nmse)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Time: {epoch_time:.2f}s")
        print(f"Comm - Train Loss: {comm_train_loss:.6f}, Val Loss: {comm_val_loss:.6f}, NMSE: {comm_val_nmse:.6f}")
        print(f"Radar - Train Loss: {radar_train_loss:.6f}, Val Loss: {radar_val_loss:.6f}, NMSE: {radar_val_nmse:.6f}")
        print("-" * 80)
    
    # Plot training history
    plot_training_history(history, config['output_dir'])
    
    # Save final models
    torch.save(comm_model.state_dict(), os.path.join(config['output_dir'], 'final_comm_model.pth'))
    torch.save(radar_model.state_dict(), os.path.join(config['output_dir'], 'final_radar_model.pth'))
    
    return comm_model, radar_model, history

# Function to visualize results
#Visualization functions to display:
# - Input, target, and output for both OFDM and radar tasks
# - Error maps to show where the models make mistakes
# - Training history plots for losses and NMSE metrics
def visualize_results(ofdm_features, ofdm_target, ofdm_output, 
                     radar_features, radar_target, radar_output, 
                     target_info, output_dir):
    """
    Visualize the model outputs for both OFDM and Radar tasks.
    """
    # Create output directory for figures
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Take the first sample for visualization
    sample_idx = 0
    
    # OFDM Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot OFDM input
    plt.subplot(3, 2, 1)
    ofdm_input_mag = np.sqrt(ofdm_features[sample_idx, 0]**2 + ofdm_features[sample_idx, 1]**2)
    plt.imshow(ofdm_input_mag, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('OFDM Input Magnitude')
    plt.xlabel('Subcarrier')
    plt.ylabel('OFDM Symbol')
    
    # Plot OFDM target
    plt.subplot(3, 2, 3)
    ofdm_target_mag = np.sqrt(ofdm_target[sample_idx, 0]**2 + ofdm_target[sample_idx, 1]**2)
    plt.imshow(ofdm_target_mag, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('OFDM Target Magnitude')
    plt.xlabel('Subcarrier')
    plt.ylabel('OFDM Symbol')
    
    # Plot OFDM output
    plt.subplot(3, 2, 5)
    ofdm_output_mag = np.sqrt(ofdm_output[sample_idx, 0]**2 + ofdm_output[sample_idx, 1]**2)
    plt.imshow(ofdm_output_mag, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('OFDM Output Magnitude')
    plt.xlabel('Subcarrier')
    plt.ylabel('OFDM Symbol')
    
    # Plot Radar input
    plt.subplot(3, 2, 2)
    radar_input_mag = np.sqrt(radar_features[sample_idx, 0]**2 + radar_features[sample_idx, 1]**2)
    plt.imshow(20*np.log10(radar_input_mag + 1e-10), aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Radar Input Magnitude')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Plot Radar target
    plt.subplot(3, 2, 4)
    radar_target_mag = np.sqrt(radar_target[sample_idx, 0]**2 + radar_target[sample_idx, 1]**2)
    plt.imshow(20*np.log10(radar_target_mag + 1e-10), aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Radar Target (Range-Doppler Map)')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Plot Radar output
    plt.subplot(3, 2, 6)
    radar_output_mag = np.sqrt(radar_output[sample_idx, 0]**2 + radar_output[sample_idx, 1]**2)
    plt.imshow(20*np.log10(radar_output_mag + 1e-10), aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Radar Output (Range-Doppler Map)')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Add target markers if available
    if target_info is not None:
        for i, target in enumerate(target_info):
            distance = target['distance']
            velocity = target['velocity']
            # Convert to bin indices (simplified)
            # This would need proper conversion based on your radar parameters
            plt.subplot(3, 2, 4)  # Target ground truth
            plt.scatter(distance//10, velocity+num_ofdm_symbols//2, c='red', marker='x', s=100)
            plt.subplot(3, 2, 6)  # Model output
            plt.scatter(distance//10, velocity+num_ofdm_symbols//2, c='red', marker='x', s=100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_outputs.png'), dpi=300)
    plt.close()
    
    # Plot error maps
    plt.figure(figsize=(15, 10))
    
    # OFDM error
    plt.subplot(1, 2, 1)
    ofdm_error = np.abs(ofdm_output_mag - ofdm_target_mag)
    plt.imshow(ofdm_error, aspect='auto', cmap='hot')
    plt.colorbar(label='Error Magnitude')
    plt.title('OFDM Error Map')
    plt.xlabel('Subcarrier')
    plt.ylabel('OFDM Symbol')
    
    # Radar error
    plt.subplot(1, 2, 2)
    radar_error = np.abs(radar_output_mag - radar_target_mag)
    plt.imshow(20*np.log10(radar_error + 1e-10), aspect='auto', cmap='hot')
    plt.colorbar(label='Error Magnitude (dB)')
    plt.title('Radar Error Map')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'error_maps.png'), dpi=300)
    plt.close()

# Function to plot training history
def plot_training_history(history, output_dir):
    """
    Plot training and validation losses and metrics.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot communication losses
    plt.subplot(2, 2, 1)
    plt.plot(history['comm_train_loss'], label='Train Loss')
    plt.plot(history['comm_val_loss'], label='Val Loss')
    plt.title('Communication Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot radar losses
    plt.subplot(2, 2, 2)
    plt.plot(history['radar_train_loss'], label='Train Loss')
    plt.plot(history['radar_val_loss'], label='Val Loss')
    plt.title('Radar Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot communication NMSE
    plt.subplot(2, 2, 3)
    plt.plot(history['comm_val_nmse'], label='NMSE')
    plt.title('Communication NMSE')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.legend()
    plt.grid(True)
    
    # Plot radar NMSE
    plt.subplot(2, 2, 4)
    plt.plot(history['radar_val_nmse'], label='NMSE')
    plt.title('Radar NMSE')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()

# Function to evaluate models on test data
# - Calculates MSE and NMSE metrics on test data
# - Saves sample visualizations for qualitative assessment
# - Outputs results to a text file
def evaluate_models(comm_model, radar_model, data_path, output_dir, batch_size=32, num_workers=4):
    """
    Evaluate trained models on test data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset and dataloader
    test_dataset = OFDMRadarDataset(data_path, mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Set models to evaluation mode
    comm_model.eval()
    radar_model.eval()
    
    # Metrics
    comm_mse = 0.0
    comm_nmse = 0.0
    radar_mse = 0.0
    radar_nmse = 0.0
    
    # For visualization
    sample_inputs = []
    sample_targets = []
    sample_outputs = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Get data
            ofdm_features = batch['ofdm_features'].to(device)
            radar_features = batch['radar_features'].to(device)
            ofdm_target = batch['ofdm_target'].to(device)
            radar_target = batch['radar_target'].to(device)
            
            # Forward pass
            ofdm_output = comm_model(ofdm_features)
            radar_output = radar_model(radar_features)
            
            # Calculate metrics
            comm_batch_mse = torch.mean((ofdm_output - ofdm_target)**2).item()
            comm_batch_nmse = torch.sum((ofdm_output - ofdm_target)**2) / torch.sum(ofdm_target**2)
            comm_batch_nmse = comm_batch_nmse.item()
            
            radar_batch_mse = torch.mean((radar_output - radar_target)**2).item()
            radar_batch_nmse = torch.sum((radar_output - radar_target)**2) / torch.sum(radar_target**2)
            radar_batch_nmse = radar_batch_nmse.item()
            
            # Accumulate metrics
            comm_mse += comm_batch_mse
            comm_nmse += comm_batch_nmse
            radar_mse += radar_batch_mse
            radar_nmse += radar_batch_nmse
            
            # Save samples for visualization
            if i < 5:  # Save first 5 batches
                sample_inputs.append({
                    'ofdm': ofdm_features.cpu().numpy(),
                    'radar': radar_features.cpu().numpy()
                })
                sample_targets.append({
                    'ofdm': ofdm_target.cpu().numpy(),
                    'radar': radar_target.cpu().numpy()
                })
                sample_outputs.append({
                    'ofdm': ofdm_output.cpu().numpy(),
                    'radar': radar_output.cpu().numpy()
                })
    
    # Calculate average metrics
    comm_mse /= len(test_loader)
    comm_nmse /= len(test_loader)
    radar_mse /= len(test_loader)
    radar_nmse /= len(test_loader)
    
    # Print results
    print("\nTest Results:")
    print(f"Communication - MSE: {comm_mse:.6f}, NMSE: {comm_nmse:.6f}")
    print(f"Radar - MSE: {radar_mse:.6f}, NMSE: {radar_nmse:.6f}")
    
    # Save results to file
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Communication - MSE: {comm_mse:.6f}, NMSE: {comm_nmse:.6f}\n")
        f.write(f"Radar - MSE: {radar_mse:.6f}, NMSE: {radar_nmse:.6f}\n")
    
    # Visualize sample results
    for i in range(min(5, len(sample_inputs))):
        visualize_results(
            sample_inputs[i]['ofdm'],
            sample_targets[i]['ofdm'],
            sample_outputs[i]['ofdm'],
            sample_inputs[i]['radar'],
            sample_targets[i]['radar'],
            sample_outputs[i]['radar'],
            test_dataset.data.get('targets', None),
            os.path.join(output_dir, f'test_sample_{i+1}')
        )
    
    return {
        'comm_mse': comm_mse,
        'comm_nmse': comm_nmse,
        'radar_mse': radar_mse,
        'radar_nmse': radar_nmse
    }

# Main function
def main():
    # Configuration
    config = {
        'data_path': '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar_results/ofdm_radar_dataset.pkl',
        'output_dir': '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/multitask_results',
        'batch_size': 32,
        'num_workers': 4,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'model_dim': 256,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 512,
        'dropout': 0.1
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Train models
    print("Training models...")
    comm_model, radar_model, history = train_model(config)
    
    # Evaluate models
    print("\nEvaluating models...")
    test_results = evaluate_models(
        comm_model, 
        radar_model, 
        config['data_path'], 
        config['output_dir'], 
        config['batch_size'], 
        config['num_workers']
    )
    
    print("\nTraining and evaluation completed!")
    print(f"Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
    