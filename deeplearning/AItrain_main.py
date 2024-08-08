import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import os
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm.auto import tqdm

from signalmodels import ResModel_2D
from AIsim_maindataset import OFDMDataset

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
    #if device.type == "mps":
        #device = "cpu" # Force CPU for now, trouble with converting complex tensors to mps with macos M1

    return device, useamp

def trainmain(trainoutput, batch_size = 16, saved_model_path = ""):
    device, useamp=get_device(gpuid='0', useamp=False)

    # OFDM Dataset
    #train_data = OFDMDataset(Qm=Qm, S=S, Sp=Sp, F=F, ch_SINR_min=-10, ch_SINR_max=40, training=True)
    
    train_data = OFDMDataset(training=True, ch_SINR_min=-20, ch_SINR_max=30, maxdatalen=10000, testing=False, compare=False, drawfig=False)
    onebatch = train_data[0]

    # train, validation and test split
    train_size = int(0.8 * len(train_data)) #8000
    val_size = len(train_data) - train_size
    train_set, val_set= torch.utils.data.random_split(train_data, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)

    onebatch = next(iter(train_loader))
    #rx_samples = onebatch['samples']
    feature_2d = onebatch['feature_2d']
    data_labels = onebatch['labels']
    #print(f"Sample batch shape: {rx_samples.size()}")
    print(f"Feature batch shape: {feature_2d.size()}") #[16, 2, 12, 64]
    print(f"Labels batch shape: {data_labels.size()}") #[16, 12, 64, 2]

    model =  ResModel_2D(num_bits_per_symbol=train_data.num_bits_per_symbol, \
                         num_ch=train_data.feature_2d_channel, \
                            S=train_data.effectiveofdmsymbols, \
                                F=train_data.effectivesubcarrier+1).to(device)
    #model = ResModel_simple1_2D(num_bits_per_symbol=Qm, num_ch=4, S=S, F=F).to(device)
    #model = MyWave2vec(num_bits_per_symbol=Qm, num_ch=4, S=S, F=F).to(device)

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

    performance_csv_path = os.path.join(trainoutput, 'performance.csv')#'output/performance_res2d2.csv'

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
            fieldnames = ['Epoch', 'Training_Loss', 'Validation_Loss', 'Validation_BER', 'LS_BER']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        model.train()  # Set the model to training mode

        for index, data_batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in data_batch.items()}
            #rx_samples = batch['samples']
            feature_2d = batch['feature_2d'] #[16, 2, 12, 64]
            labels = batch['labels'] #[16, 12, 64, 2]
            outputs = model(feature_2d)  # forward pass [16, 14, 71, 6]
            loss = criterion(outputs, labels) #[16, 14, 71, 6]
            loss.backward()  # backward pass
            optimizer.step()  # update the weights
            total_loss += loss.item()  # accumulate the loss
            #progress_bar.update(1)
            optimizer.zero_grad()  # Zero the gradients

        # Update the learning rate
        scheduler.step()

        # Print average loss for the epoch
        average_loss = total_loss / len(train_loader)

        # Validation
        model.eval()  # Set the model to evaluation mode
        BER_batch=[]
        LSBER_batch=[]
        with torch.no_grad():
            for index, data_batch in enumerate(tqdm(val_loader)):
                batch = {k: v.to(device) for k, v in data_batch.items()}
                feature_2d = batch['feature_2d']
                labels = batch['labels']
                #rx_samples = batch['samples'] #[1, 2078]
                val_outputs = model((feature_2d)) #[1, 14, 71, 6]
                val_loss = criterion(val_outputs, labels)

                # Convert probabilities to binary predictions (0 or 1)
                binary_predictions = torch.round(val_outputs) #[1, 14, 71, 6]

                # Calculate Bit Error Rate (BER)
                labels = labels.cpu().squeeze() #[1, 14, 71, 6]
                #BER, NN_wrongs = multiprocessor.NNevaluate(binary_predictions.cpu(), labels)
                BER = train_data.calculate_BER(binary_predictions.cpu(), labels)
                BER_batch.append(BER)

                # rx_samples=rx_samples.squeeze().cpu()
                # #back into the frequency domain
                # OFDM_demod = multiprocessor.receiver_preprocessing(rx_samples) #[14, 128]
                # ZHLS_binary_predictions = multiprocessor.ZHLSreceiver(OFDM_demod)
                # ZHLS_BER, ZHLS_wrongs = multiprocessor.evaluate(ZHLS_binary_predictions, labels)
                ZHLS_BER = -1
                LSBER_batch.append(ZHLS_BER)
                
        # Save performance details
        train_losses.append(average_loss)
        val_losses.append(val_loss.item())
        BER_batch_mean=np.mean(BER_batch)
        val_BERs.append(BER_batch_mean)#(BER.item())
        LSBER_batch_mean=np.mean(LSBER_batch)

        # Print or log validation loss after each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Val Loss: {val_loss:.4f}, Val BER: {BER_batch_mean:.4f}, LS BER: {LSBER_batch_mean:.4f},learning rate: {scheduler.get_last_lr()[0]:.4f}")

        # Save performance details in the CSV file
        with open(performance_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, average_loss, val_loss.item(), BER_batch_mean, LSBER_batch_mean])

        if (epoch + 1) % 2 == 0:
            # Save model along with the current epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            modelsave_path = os.path.join(trainoutput, f'res2d_model_{epoch + 1}.pth')
            torch.save(checkpoint, modelsave_path)
            print(f"Model saved at epoch {epoch + 1}")
    
    # Save the final trained model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    #torch.save(checkpoint, 'output/res2d_model.pth')
    modelsave_path = os.path.join(trainoutput, 'res2d_model.pth')
    torch.save(checkpoint, modelsave_path)

import pandas as pd
import matplotlib.pyplot as plt
def draw_trainresults(csv_path = '../output/performance_details.csv', save_plots =  True):
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Plot Training Loss and Validation Loss
    plt.figure(figsize=(7, 3))
    plt.plot(df['Epoch'], df['Training_Loss'], label='Training Loss')
    plt.plot(df['Epoch'], df['Validation_Loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.grid(True)
    if save_plots:
        plt.savefig('training_loss.png')
    plt.show()

    # Plot Validation BER
    plt.figure(figsize=(7, 3))
    plt.plot(df['Epoch'], df['Validation_BER'], label='Validation BER')
    #plt.plot(df['Epoch'], df['LS_BER'], label='LS BER')
    plt.xlabel('Epochs')
    plt.ylabel('BER')
    plt.legend()
    plt.title('Bit Error Rate (BER) on validation set')
    plt.grid(True)
    #plt.ylim(0,0.06)
    if save_plots:
        plt.savefig('training_ber.png')
    plt.show(block=True)

import json
def savedict2file(data_dict, filename):
    # save vocab dict to be loaded into tokenizer
    with open(filename, "w") as file:
        json.dump(data_dict, file)

def saveargs2file(args, trainoutput):
    args_dict={}
    args_str=' '
    for k, v in vars(args).items():
        args_dict[k]=v
        args_str.join(f'{k}={v}, ')
    print(args_str)
    savedict2file(data_dict=args_dict, filename=os.path.join(trainoutput,'args.json'))

def testdataset():
    train_data = OFDMDataset(training=True, testing=False, compare=False)
    onebatch = train_data[0]
    print(onebatch['feature_2d'].shape) #(2, 12, 64)
    print(onebatch['labels'].shape) #(12, 64, 2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OFDM training job')
    #data related arguments
    parser.add_argument('--mode', default="Train", choices=['Train','Evaluate', 'Visualization'], help='Running mode')
    parser.add_argument('--traintag', type=str, default='exp0809b',
                    help='Train rag name, used for output folder')
    parser.add_argument('--data_type', type=str, default="OFDMsim",
                    help='data type name')
    parser.add_argument('--data_name', type=str, default="",
                    help='data name')
    parser.add_argument('--data_path', type=str, default="./data", help='data folder')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='outputpath') #r"E:\output"
    
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    trainoutput=os.path.join(args.outputdir, args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    #testdataset()
    if args.mode == "Evaluate":
        draw_trainresults(csv_path=os.path.join(trainoutput, 'performance.csv'), save_plots=True)
    elif args.mode == "Train":
        trainmain(trainoutput)