import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import csv

from AIradar_dataset import RadarDataset, RadarNet

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


def test_radar_model(model_path=None, test_data_path=None, output_dir=None):
    device, useamp = get_device(gpuid='0', useamp=False)
    
    # Load test data
    test_dataset = RadarDataset(
        datapath=test_data_path,
        num_samples=1000,
        training=False,
        drawfig=True
    )
    
    # Load trained model
    model = RadarNet().to(device)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test metrics
    detection_accuracy = 0
    false_alarm_rate = 0
    missed_detection_rate = 0
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            input_data = torch.from_numpy(sample['feature_2d']).unsqueeze(0).to(device)
            target = torch.from_numpy(sample['labels']).to(device) #[12, 64, 1]
            
            # Forward pass
            output = model(input_data) #[1, 2, 12, 64]=>[1, 12, 64, 1]
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
                # visualize_detection(
                #     input_data[0].cpu().numpy(),
                #     target[0].cpu().numpy(),
                #     predictions[0].cpu().numpy(),
                #     os.path.join(output_dir, f'detection_result_{i}.pdf')
                # )
                visualize_detection(
                    input_data[0].cpu().numpy(),
                    target.cpu().numpy(),#does not contain bathch dimension
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
    """Visualize radar detection results"""
    magnitude = np.sqrt(input_data[0]**2 + input_data[1]**2) #(12, 64)
    
    plt.figure(figsize=(15, 5))
    
    # Plot input range-Doppler map
    plt.subplot(131)
    plt.imshow(20*np.log10(magnitude + 1e-10), aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Range-Doppler Map')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Plot ground truth
    plt.subplot(132)
    plt.imshow(target[:,:,0], aspect='auto', cmap='gray')
    plt.colorbar(label='Target Presence')
    plt.title('Ground Truth')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    # Plot prediction
    plt.subplot(133)
    plt.imshow(prediction[:,:,0], aspect='auto', cmap='gray')#(1, 12, 64, 1)
    plt.colorbar(label='Detection')
    plt.title('Model Prediction')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    output_dir = '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar_training'
    os.makedirs(output_dir, exist_ok=True)
    # Generate or load radar data
    # train_radar_model(
    #     output_dir=output_dir,
    #     num_samples=10000,
    #     batch_size=32,
    #     num_epochs=50
    # )
    
    model_path = '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar_training/best_radar_model.pth'
    output_dir = '/Users/kaikailiu/Documents/MyRepo/radarsensing/data/radar_results'
    os.makedirs(output_dir, exist_ok=True)
    
    test_radar_model(
        model_path=model_path,
        output_dir=output_dir
    )
    
