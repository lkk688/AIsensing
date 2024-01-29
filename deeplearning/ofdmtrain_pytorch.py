import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as tFunc # usually F, but that is reserved for other use
import csv
import os
from ofdmsim_pytorchlib import *
from signalmodels import RXModel_2
import math
from tqdm.auto import tqdm

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
    # Test tensor creation on the selected device
    if device.type != "cpu":
        x = torch.ones(1, device=device)
        print(x)
    if device.type == "mps":
        device = "cpu" # Force CPU for now, trouble with converting complex tensors to mps with macos M1

    return device, useamp

def trainmain():
    device, useamp=get_device(gpuid='0', useamp=False)

    dataset = CustomDataset()
    dataset = torch.load('output/ofdm_dataset.pth')
    batch_size = 16

    # train, validation and test split
    train_size = int(0.8 * len(dataset)) #8000
    val_size = len(dataset) - train_size
    train_set, val_set= torch.utils.data.random_split(dataset, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # OFDM Parameters
    #Qm (int): Modulation order
    Qm = 6  # bits per symbol
    S = 14  # Number of symbols
    Sp = 2  # Pilot symbol, 0 for none
    F = 72  # Number of subcarriers, including DC
    model = RXModel_2(Qm, S=S, F=F).to(device)

    initial_lr = 0.001 # Initial learning rate
    final_lr = 0.0003 # Final learning rate at the end
    num_epochs = 100 # epochs for learning rate scheduler decay

    # Define the model's optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Lambda function for learning rate decay
    lambda_lr = lambda epoch: final_lr / initial_lr + (1 - epoch / num_epochs) * (1 - final_lr / initial_lr)

    # Define the learning rate scheduler and loss function
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    criterion = nn.BCELoss()

    saved_model_path = "" #'data/rx_model_168.pth'
    performance_csv_path = 'output/performance_details.csv'

    # Check if a saved model exists
    if os.path.exists(saved_model_path):
        # Load the existing model and epoch
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Existing model loaded from {saved_model_path}, Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No saved model found. Training from scratch.")
    
    # Lists to store performance details for plotting
    train_losses = []
    val_losses = []
    val_BERs = []

    # Check if a performance CSV file exists
    if not os.path.exists(performance_csv_path):
        # Create a new CSV file and write headers
        with open(performance_csv_path, mode='w', newline='') as csv_file:
            fieldnames = ['Epoch', 'Training_Loss', 'Validation_Loss', 'Validation_BER']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    
    TTI_mask_RE = TTI_mask(S=S,F=F, Fp=Fp, Sp=Sp, FFT_offset=FFT_offset, plotTTI=False).to(device) # TTI mask [14, 128]
    pilot_symbols = pilot_set(TTI_mask_RE, Pilot_Power).to(device) # pilot symbols [36]

    # remove DC and FFT offsets from TTI mask_RE and add third dimension size of Qm, and expand TTI mask values into the third dimension
    TTI_mask_RE_small = TTI_mask_RE[:, FFT_offset:-FFT_offset] #FFT_offset=28 [14, 72]
    middle_index = TTI_mask_RE_small.size(1) // 2
    TTI_mask_RE_small = torch.cat((TTI_mask_RE_small[:, :middle_index], TTI_mask_RE_small[:, middle_index + 1:]), dim=1) #[14, 71]
    TTI_mask_RE_3d = TTI_mask_RE_small.unsqueeze(-1).expand(batch_size, S, F-1, Qm) #[16, 14, 71, 6]

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        model.train()  # Set the model to training mode

        #num_update_steps_per_epoch = math.ceil(len(train_loader))
        #progress_bar = tqdm(range(num_update_steps_per_epoch))

        for pdsch_iq,  labels in enumerate(tqdm(train_loader)):
            pdsch_iq,  labels = pdsch_iq.to(device), labels.to(device) #[16, 14, 71], [16, 14, 71, 6]
            optimizer.zero_grad()  # Zero the gradients
            outputs = model((pdsch_iq))  # forward pass [16, 14, 71, 6]
            loss = criterion(outputs, labels)
            loss.backward()  # backward pass
            optimizer.step()  # update the weights
            total_loss += loss.item()  # accumulate the loss
            #progress_bar.update(1)

        # Update the learning rate
        scheduler.step()

        # Print average loss for the epoch
        average_loss = total_loss / len(train_loader)

        # Validation
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for val_pdsch_iq, val_labels in enumerate(tqdm(val_loader)):
                val_pdsch_iq, val_labels = val_pdsch_iq.to(device), val_labels.to(device) #[16, 14, 71], [16, 14, 71, 6]
                val_outputs = model((val_pdsch_iq)) #[16, 14, 71, 6]
                val_loss = criterion(val_outputs, val_labels)

                # Convert probabilities to binary predictions (0 or 1)
                binary_predictions = torch.round(val_outputs) #[16, 14, 71, 6]

                # Calculate Bit Error Rate (BER)
                error_count = torch.sum(binary_predictions != val_labels).float()  # Count of unequal bits
                error_rate = error_count / len(val_labels.flatten())  # Error rate calculation
                BER = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places

        # Save performance details
        train_losses.append(average_loss)
        val_losses.append(val_loss.item())
        val_BERs.append(BER.item())

        # Print or log validation loss after each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Val Loss: {val_loss:.4f}, Val BER: {BER:.4f}, learning rate: {scheduler.get_last_lr()[0]:.4f}")

        # Save performance details in the CSV file
        with open(performance_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, average_loss, val_loss.item(), BER.item()])

        if (epoch + 1) % 2 == 0:
            # Save model along with the current epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            torch.save(checkpoint, f'output/rx_model_{epoch + 1}.pth')
            print(f"Model saved at epoch {epoch + 1}")
    
    # Save the final trained model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, 'output/rx_model.pth')

if __name__ == '__main__':
    trainmain()