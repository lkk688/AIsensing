import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

class TransformerChannelEstimator(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerChannelEstimator, self).__init__()
        self.input_embedding = nn.Linear(input_dim, model_dim)
        
        # Position encoding for better sequence modeling
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, 
                                                   dim_feedforward=4*model_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim*2, output_dim)
        )
    
    def forward(self, x):
        # x: (batch_size, num_rx_antennas, num_ofdm_symbols, fft_size, input_dim)
        batch_size, num_rx_antennas, num_ofdm_symbols, fft_size, input_dim = x.shape
        
        # Reshape for transformer: (seq_len, batch_size * num_rx_antennas, input_dim)
        x = x.view(batch_size * num_rx_antennas, num_ofdm_symbols * fft_size, input_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size * num_rx_antennas, input_dim)
        
        # Input embedding and positional encoding
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Reshape back to original dimensions
        x = x.permute(1, 0, 2)  # (batch_size * num_rx_antennas, seq_len, output_dim)
        x = x.view(batch_size, num_rx_antennas, num_ofdm_symbols, fft_size, -1)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ChannelSimulator:
    def __init__(self):
        pass
    
    def generate_rayleigh_channel(self, batch_size, num_rx_antennas, num_tx_antennas, 
                                 num_ofdm_symbols, fft_size, max_delay_spread=10):
        """Generate Rayleigh fading channel with realistic delay spread"""
        # Time-domain channel impulse response
        h_time = (np.random.randn(batch_size, num_rx_antennas, num_tx_antennas, max_delay_spread) + 
                 1j * np.random.randn(batch_size, num_rx_antennas, num_tx_antennas, max_delay_spread)) / np.sqrt(2)
        
        # Apply exponential power delay profile
        delay_profile = np.exp(-np.arange(max_delay_spread) / (max_delay_spread/3))
        h_time *= delay_profile[np.newaxis, np.newaxis, np.newaxis, :]
        
        # Normalize power
        h_time /= np.sqrt(np.sum(np.abs(h_time)**2, axis=-1, keepdims=True))
        
        # Convert to frequency domain for each OFDM symbol
        h_freq = np.zeros((batch_size, num_rx_antennas, num_tx_antennas, num_ofdm_symbols, fft_size), dtype=np.complex128)
        
        for i in range(num_ofdm_symbols):
            # Add small time variation between OFDM symbols
            h_time_symbol = h_time * np.exp(1j * 0.01 * i * np.random.randn(batch_size, num_rx_antennas, num_tx_antennas, max_delay_spread))
            
            # FFT to get frequency domain channel
            h_freq_symbol = np.fft.fft(h_time_symbol, fft_size, axis=-1)
            h_freq[:, :, :, i, :] = h_freq_symbol
            
        return h_freq
    
    def generate_mimo_ofdm_data(self, batch_size, num_rx_antennas, num_tx_antennas, 
                               num_ofdm_symbols, fft_size, snr_db, pilot_pattern='block'):
        """
        Generate MIMO-OFDM data with realistic channel and pilot patterns
        
        Parameters:
        - pilot_pattern: 'block', 'comb', or 'scattered'
        """
        # Generate channel
        h = self.generate_rayleigh_channel(batch_size, num_rx_antennas, num_tx_antennas, 
                                          num_ofdm_symbols, fft_size)
        
        # Create pilot pattern mask (1 for pilot, 0 for data)
        pilot_mask = np.zeros((num_ofdm_symbols, fft_size), dtype=bool)
        
        if pilot_pattern == 'block':
            # Block-type pilots (entire OFDM symbols)
            pilot_symbols = [0, num_ofdm_symbols//2]  # First and middle OFDM symbols as pilots
            pilot_mask[pilot_symbols, :] = 1
            
        elif pilot_pattern == 'comb':
            # Comb-type pilots (specific subcarriers in all OFDM symbols)
            pilot_carriers = np.arange(0, fft_size, 8)  # Every 8th subcarrier
            pilot_mask[:, pilot_carriers] = 1
            
        elif pilot_pattern == 'scattered':
            # Scattered pilots
            for i in range(0, num_ofdm_symbols, 4):
                for j in range(0, fft_size, 4):
                    pilot_mask[i, j] = 1
                    if i+2 < num_ofdm_symbols and j+2 < fft_size:
                        pilot_mask[i+2, j+2] = 1
        
        # Generate transmitted symbols (QPSK)
        x = np.zeros((batch_size, num_tx_antennas, num_ofdm_symbols, fft_size), dtype=np.complex128)
        
        # Generate random data symbols
        data_symbols = (np.random.randint(0, 4, size=(batch_size, num_tx_antennas, num_ofdm_symbols, fft_size)))
        data_symbols = np.exp(1j * data_symbols * np.pi/2 + 1j * np.pi/4)  # QPSK constellation
        
        # Generate pilot symbols (known sequence)
        pilot_symbols = np.exp(1j * np.pi/4) * np.ones((batch_size, num_tx_antennas, num_ofdm_symbols, fft_size))
        
        # Combine data and pilots
        for b in range(batch_size):
            for t in range(num_tx_antennas):
                x[b, t, pilot_mask] = pilot_symbols[b, t, pilot_mask]
                x[b, t, ~pilot_mask] = data_symbols[b, t, ~pilot_mask]
        
        # Pass through channel
        y = np.zeros((batch_size, num_rx_antennas, num_ofdm_symbols, fft_size), dtype=np.complex128)
        for b in range(batch_size):
            for r in range(num_rx_antennas):
                for t in range(num_tx_antennas):
                    y[b, r] += x[b, t] * h[b, r, t]
        
        # Add noise
        noise_power = 10 ** (-snr_db / 10)
        noise = (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape)) * np.sqrt(noise_power / 2)
        y_noisy = y + noise
        
        # Convert to PyTorch tensors with real and imaginary parts
        y_tensor = torch.tensor(np.stack([np.real(y_noisy), np.imag(y_noisy)], axis=-1), dtype=torch.float32)
        x_tensor = torch.tensor(np.stack([np.real(x), np.imag(x)], axis=-1), dtype=torch.float32)
        h_tensor = torch.tensor(np.stack([np.real(h), np.imag(h)], axis=-1), dtype=torch.float32)
        pilot_mask_tensor = torch.tensor(pilot_mask, dtype=torch.bool)
        
        return y_tensor, x_tensor, h_tensor, pilot_mask_tensor

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    """Train the model with validation"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (y, x, h, pilot_mask) in enumerate(train_loader):
            y, x, h = y.to(device), x.to(device), h.to(device)
            
            # Create input by concatenating received signal and pilot information
            # Expand pilot information to match dimensions
            pilot_info = x[:, :, None, :, :, :].expand(-1, -1, y.shape[1], -1, -1, -1)
            inputs = torch.cat([y, pilot_info], dim=-1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, h)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y, x, h, pilot_mask in val_loader:
                y, x, h = y.to(device), x.to(device), h.to(device)
                
                # Create input
                pilot_info = x[:, :, None, :, :, :].expand(-1, -1, y.shape[1], -1, -1, -1)
                inputs = torch.cat([y, pilot_info], dim=-1)
                
                outputs = model(inputs)
                loss = criterion(outputs, h)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device, snr_range=None):
    """Evaluate model performance across different SNRs"""
    model.eval()
    
    if snr_range is None:
        # Evaluate on test set
        test_loss = 0.0
        nmse_values = []
        
        with torch.no_grad():
            for y, x, h, pilot_mask in test_loader:
                y, x, h = y.to(device), x.to(device), h.to(device)
                
                # Create input
                pilot_info = x[:, :, None, :, :, :].expand(-1, -1, y.shape[1], -1, -1, -1)
                inputs = torch.cat([y, pilot_info], dim=-1)
                
                h_est = model(inputs)
                
                # Calculate NMSE (Normalized Mean Square Error)
                error = torch.sum((h_est - h)**2, dim=-1)
                power = torch.sum(h**2, dim=-1)
                nmse = torch.mean(error / power).item()
                nmse_values.append(nmse)
                
        avg_nmse = np.mean(nmse_values)
        print(f'Test NMSE: {avg_nmse:.6f}')
        return avg_nmse
    
    else:
        # Evaluate across SNR range
        simulator = ChannelSimulator()
        nmse_per_snr = []
        
        for snr in snr_range:
            print(f'Evaluating at SNR = {snr} dB')
            batch_size = 32
            num_rx_antennas = 2
            num_tx_antennas = 2
            num_ofdm_symbols = 14
            fft_size = 64
            
            # Generate test data at this SNR
            y, x, h, pilot_mask = simulator.generate_mimo_ofdm_data(
                batch_size, num_rx_antennas, num_tx_antennas, 
                num_ofdm_symbols, fft_size, snr, pilot_pattern='block'
            )
            
            y, x, h = y.to(device), x.to(device), h.to(device)
            
            # Create input
            pilot_info = x[:, :, None, :, :, :].expand(-1, -1, y.shape[1], -1, -1, -1)
            inputs = torch.cat([y, pilot_info], dim=-1)
            
            with torch.no_grad():
                h_est = model(inputs)
                
                # Calculate NMSE
                error = torch.sum((h_est - h)**2, dim=-1)
                power = torch.sum(h**2, dim=-1)
                nmse = torch.mean(error / power).item()
            
            nmse_per_snr.append(nmse)
            print(f'SNR = {snr} dB, NMSE = {nmse:.6f}')
        
        return nmse_per_snr

def plot_results(train_losses, val_losses, nmse_per_snr=None, snr_range=None):
    """Plot training curves and NMSE vs SNR if available"""
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2 if nmse_per_snr is not None else 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot NMSE vs SNR if available
    if nmse_per_snr is not None and snr_range is not None:
        plt.subplot(1, 2, 2)
        plt.plot(snr_range, nmse_per_snr, 'o-')
        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.title('NMSE vs SNR')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/kaikailiu/Documents/MyRepo/radarsensing/deeplearning/channel_estimation_results.png')
    plt.show()

def compare_with_ls_estimator(test_loader, model, device, snr_range):
    """Compare neural network estimator with traditional LS estimator"""
    model.eval()
    simulator = ChannelSimulator()
    
    nn_nmse_per_snr = []
    ls_nmse_per_snr = []
    
    for snr in snr_range:
        print(f'Evaluating at SNR = {snr} dB')
        batch_size = 32
        num_rx_antennas = 2
        num_tx_antennas = 2
        num_ofdm_symbols = 14
        fft_size = 64
        
        # Generate test data at this SNR
        y, x, h, pilot_mask = simulator.generate_mimo_ofdm_data(
            batch_size, num_rx_antennas, num_tx_antennas, 
            num_ofdm_symbols, fft_size, snr, pilot_pattern='block'
        )
        
        # Convert to numpy for LS estimation
        y_np = y.numpy()
        x_np = x.numpy()
        h_np = h.numpy()
        pilot_mask_np = pilot_mask.numpy()
        
        # LS estimation at pilot locations
        h_ls = np.zeros_like(h_np)
        
        # Perform LS estimation
        for b in range(batch_size):
            for r in range(num_rx_antennas):
                for t in range(num_tx_antennas):
                    for i in range(num_ofdm_symbols):
                        for j in range(fft_size):
                            if pilot_mask_np[i, j]:
                                # Extract real and imaginary parts
                                y_real = y_np[b, r, i, j, 0]
                                y_imag = y_np[b, r, i, j, 1]
                                x_real = x_np[b, t, i, j, 0]
                                x_imag = x_np[b, t, i, j, 1]
                                
                                # Convert to complex
                                y_complex = y_real + 1j * y_imag
                                x_complex = x_real + 1j * x_imag
                                
                                # LS estimation: h = y/x
                                h_complex = y_complex / x_complex if abs(x_complex) > 1e-10 else 0
                                
                                # Store back
                                h_ls[b, r, t, i, j, 0] = np.real(h_complex)
                                h_ls[b, r, t, i, j, 1] = np.imag(h_complex)
        
        # Interpolate LS estimates (simple linear interpolation for non-pilot locations)
        # This is a simplified interpolation - in practice, more sophisticated methods would be used
        for b in range(batch_size):
            for r in range(num_rx_antennas):
                for t in range(num_tx_antennas):
                    for i in range(num_ofdm_symbols):
                        for j in range(fft_size):
                            if not pilot_mask_np[i, j]:
                                # Find nearest pilot
                                nearest_pilot_i = i
                                nearest_pilot_j = j
                                min_dist = float('inf')
                                
                                for pi in range(num_ofdm_symbols):
                                    for pj in range(fft_size):
                                        if pilot_mask_np[pi, pj]:
                                            dist = (pi - i)**2 + (pj - j)**2
                                            if dist < min_dist:
                                                min_dist = dist
                                                nearest_pilot_i = pi
                                                nearest_pilot_j = pj
                                
                                # Copy from nearest pilot
                                h_ls[b, r, t, i, j, 0] = h_ls[b, r, t, nearest_pilot_i, nearest_pilot_j, 0]
                                h_ls[b, r, t, i, j, 1] = h_ls[b, r, t, nearest_pilot_i, nearest_pilot_j, 1]
        
        # Calculate NMSE for LS estimator
        error_ls = np.sum((h_ls - h_np)**2)
        power_ls = np.sum(h_np**2)
        nmse_ls = error_ls / power_ls
        ls_nmse_per_snr.append(nmse_ls)
        
        # Neural network estimation
        y, x, h = y.to(device), x.to(device), h.to(device)
        
        # Create input
        pilot_info = x[:, :, None, :, :, :].expand(-1, -1, y.shape[1], -1, -1, -1)
        inputs = torch.cat([y, pilot_info], dim=-1)
        
        with torch.no_grad():
            h_est = model(inputs)
            
            # Calculate NMSE
            error = torch.sum((h_est - h)**2, dim=-1)
            power = torch.sum(h**2, dim=-1)
            nmse = torch.mean(error / power).item()
        
        nn_nmse_per_snr.append(nmse)
        
        print(f'SNR = {snr} dB, NN NMSE = {nmse:.6f}, LS NMSE = {nmse_ls:.6f}')
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, nn_nmse_per_snr, 'o-', label='Neural Network Estimator')
    plt.plot(snr_range, ls_nmse_per_snr, 's-', label='LS Estimator')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE')
    plt.title('Performance Comparison: Neural Network vs LS Estimator')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/kaikailiu/Documents/MyRepo/radarsensing/deeplearning/estimator_comparison.png')
    plt.show()
    
    return nn_nmse_per_snr, ls_nmse_per_snr

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Parameters
    batch_size = 32
    num_rx_antennas = 2
    num_tx_antennas = 2
    num_ofdm_symbols = 14
    fft_size = 64
    input_dim = 4  # Real and imaginary parts of received signal and pilots
    model_dim = 256
    num_heads = 8
    num_layers = 4
    output_dim = 2 * num_tx_antennas  # Real and imaginary parts of channel for each TX antenna
    
    # Generate dataset
    print("Generating dataset...")
    simulator = ChannelSimulator()
    
    # Generate training data with varying SNR
    train_data = []
    val_data = []
    test_data = []
    
    # Training data: varying SNR from 0 to 30 dB
    for snr in range(0, 31, 5):
        num_samples = 100 if snr <= 20 else 50  # More samples at lower SNR
        for _ in range(num_samples):
            y, x, h, pilot_mask = simulator.generate_mimo_ofdm_data(
                batch_size, num_rx_antennas, num_tx_antennas, 
                num_ofdm_symbols, fft_size, snr, pilot_pattern='block'
            )
            train_data.append((y, x, h, pilot_mask))
    
    # Validation data: SNR from 5 to 25 dB
    for snr in range(5, 26, 5):
        for _ in range(20):
            y, x, h, pilot_mask = simulator.generate_mimo_ofdm_data(
                batch_size, num_rx_antennas, num_tx_antennas, 
                num_ofdm_symbols, fft_size, snr, pilot_pattern='block'
            )
            val_data.append((y, x, h, pilot_mask))
    
    # Test data: SNR from 0 to 30 dB
    for snr in range(0, 31, 5):
        for _ in range(10):
            y, x, h, pilot_mask = simulator.generate_mimo_ofdm_data(
                batch_size, num_rx_antennas, num_tx_antennas, 
                num_ofdm_symbols, fft_size, snr, pilot_pattern='block'
            )
            test_data.append((y, x, h, pilot_mask))
    
    print(f"Dataset generated: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test batches")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Initialize model
    model = TransformerChannelEstimator(input_dim, model_dim, num_heads, num_layers, output_dim)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train model
    print("Training model...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, device=device, scheduler=scheduler
    )
    
    # Save model
    torch.save(model.state_dict(), '/Users/kaikailiu/Documents/MyRepo/radarsensing/deeplearning/channel_estimator_model.pth')
    
    # Evaluate model
    print("Evaluating model...")
    test_nmse = evaluate_model(model, test_loader, device)
    
    # Evaluate across SNR range
    snr_range = list(range(0, 31, 5))
    nmse_per_snr = evaluate_model(model, None, device, snr_range)
    
    # Plot results
    plot_results(train_losses, val_losses, nmse_per_snr, snr_range)
    
    # Compare with LS estimator
    nn_nmse, ls_nmse = compare_with_ls_estimator(test_loader, model, device, snr_range)
    
    print("Done!")

if __name__ == "__main__":
    main()