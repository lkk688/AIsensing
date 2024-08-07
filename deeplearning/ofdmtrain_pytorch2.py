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
from signalmodels import ResModel_2D, ResModel_simple1_2D #, MyWave2vec
import math
from tqdm.auto import tqdm
import pickle

# custom dataset
class OFDMDataset(Dataset):
    def __init__(self, Qm=6, S=14, Sp=2, F=72, Fp=2, FFT_size=128, CP=20, ch_SINR_min=25, ch_SINR_max=50, maxdatalen=10000, training=False):
        self.maxdatalen = maxdatalen
        self.training = training
        #Signal-to-Interference-plus-Noise Ratio (SINR) for the CDL-C channel emulation
        # in case SDR not available, for channel simulation
        self.ch_SINR_min = ch_SINR_min # channel emulation min SINR
        self.ch_SINR_max = ch_SINR_max # channel emulation max SINR
        #Qm (int): Modulation order
        self.Qm = Qm  # bits per symbol
        # channel simulation
        self.n_taps = 2 
        self.max_delay = 6 #samples
        self.leading_zeros = 0 #80  # For SDR, Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.
        # OFDM Parameters
        self.S = S  # Number of symbols
        self.Sp = Sp  # Pilot symbol, 0 for none
        self.F = F  # Number of subcarriers, including DC
        self.Fp = Fp  # Pilot subcarrier spacing
        self.FFT_size = FFT_size  # FFT size
        self.FFT_offset = int((self.FFT_size - self.F) / 2)  # FFT offset
        self.CP = CP  # Cyclic prefix

        mapping_table_QPSK, de_mapping_table_QPSK = mapping_table(2) # mapping table QPSK (e.g. for pilot symbols)
        self.mapping_table_Qm, self.de_mapping_table_Qm = mapping_table(Qm, plot=False) # mapping table for Qm

        self.TTI_mask_RE = TTI_mask(S=self.S,F=self.F, Fp=self.Fp, Sp=self.Sp, FFT_offset=self.FFT_offset, plotTTI=False) #[14, 128]
        #among TTI_mask, 958 places are 1 (means data)
        self.pilot_symbols = pilot_set(self.TTI_mask_RE, Pilot_Power) #[36]
        
    def __len__(self):
        return self.maxdatalen
    
    def __getitem__(self, index):
        ch_SINR = int(random.uniform(self.ch_SINR_min, self.ch_SINR_max)) # SINR generation for adding noise to the channel
        pdsch_bits, TX_Samples = create_OFDM_data(self.TTI_mask_RE, self.Qm, self.mapping_table_Qm, self.pilot_symbols)
        #pdsch_bits: return bits: [958, 6], symbol: [958] complex, each symbol is 6 bits, 958 is the effective data slots in mask
        #TX_Samples:(S, CP 20+ FFT_size 128) 14*148 flatten to torch.Size([2072])
        #[2072]->[2078]
        RX_Samples = apply_multipath_channel(TX_Samples, n_taps=self.n_taps, max_delay=self.max_delay, random_start=False, repeats=0, SINR_s=ch_SINR, leading_zeros=self.leading_zeros)

        #Create groundtruth labels:
        # create a bit stream matrix, the same shape as TTI mask RE
        TTI_mask_indices = torch.where(self.TTI_mask_RE==1) #[14, 128]
        TTI_3d = torch.zeros((self.TTI_mask_RE.shape[0], self.TTI_mask_RE.shape[1], pdsch_bits.shape[1]), dtype=pdsch_bits.dtype) #[14, 128, 6]
        row_indices, col_indices = TTI_mask_indices #[958]
        #TTI_3d: [14, 128, 6]
        TTI_3d[row_indices, col_indices, :] = pdsch_bits.clone().detach() #[958, 6]->[14, 128, 6]
        TTI_3d = remove_fft_Offests(TTI_3d, F, FFT_offset) #[14, 72, 6]
        TTI_3d = torch.cat((TTI_3d[:, :F//2,:], TTI_3d[:, F//2 + 1:,:]), dim=1)  # remove DC, [14, 71, 6]
        
        batch={}
        if self.training:
            OFDM_demod = self.receiver_preprocessing(RX_Samples) #[2078]->[14, 128]
            pdsch_symbols_map = self.NNpreprocessing(OFDM_demod) #[14, 128] ->[14, 71]
            feature_2d = self.create_2Dfeature(pdsch_symbols_map, simple_stack=False)
            batch['feature_2d']= feature_2d #[4, 14, 71]
        batch['samples']=RX_Samples #[2078]
        batch['labels']=TTI_3d #[14, 71, 6]
        return batch #RX_Samples, TTI_3d
    
        #for all receivers
    def receiver_preprocessing(self, RX_Samples):
        #RX_Samples = batch['rx_samples']
        #step1: CP remove
        symbol_index = 1 #starting place
        RX_NO_CP = CP_removal(RX_Samples, symbol_index, self.S, self.FFT_size, self.CP, plotsig=False)# remove cyclic prefix and other symbols created by convolution
        RX_NO_CP = RX_NO_CP / torch.max(torch.abs(RX_NO_CP)) # normalize
        #torch.Size([14, 128])

        #back into the frequency domain
        OFDM_demod = DFT(RX_NO_CP, plotDFT=False) # DFT
        #batch['OFDM_demod'] = OFDM_demod
        return OFDM_demod

    def NNpreprocessing(self, OFDM_demod):
        OFDM_demod = OFDM_demod / torch.max(torch.abs(OFDM_demod)) # normalize DFT'd signal for NN input
        #torch.Size([14, 128])
        #F is number of carriers
        pdsch_symbols_map = remove_fft_Offests(OFDM_demod, self.F, self.FFT_offset) # remove FFT offsets
        #[14, 72]
        # remove DC
        pdsch_symbols_map = torch.cat((pdsch_symbols_map[:, :self.F//2], pdsch_symbols_map[:, self.F//2 + 1:]), dim=1) 
        # [14, 71]
        return pdsch_symbols_map
    
    def create_2Dfeature(self, pdsch_symbols_map, simple_stack=True):
        y_real = pdsch_symbols_map.real #[14, 71]
        y_imag = pdsch_symbols_map.imag
        if simple_stack:
            # Stack the tensors along a new dimension (axis 0)
            z = torch.stack([y_real, y_imag], dim=0) #[2, 14, 71]
        else:
            y_mag = y_real.pow(2) + y_imag.pow(2)
            y_phase = torch.atan2(
                    -y_imag + 0.0, y_real
                )  # +0.0 removes -0.0 elements, which leads to error in calculating phase
            y_complex = torch.stack(
                    (y_real, -y_imag), -1
                )  # Remember the minus sign for imaginary part
            z = torch.stack([y_real, y_imag, y_mag, y_phase], dim=0)
            #z = z.permute(1, 0, 2, 3) #[16, 2, 14, 71]
        return z

class MultiReceiver():
    def __init__(self, Qm=6, S=14, Sp=2, F=72, Fp=2, FFT_size=128, CP=20):
        #Qm (int): Modulation order
        self.Qm = Qm  # bits per symbol
        # OFDM Parameters
        self.S = S  # Number of symbols
        self.Sp = Sp  # Pilot symbol, 0 for none
        self.F = F  # Number of subcarriers, including DC
        self.Fp = Fp  # Pilot subcarrier spacing
        self.FFT_size = FFT_size  # FFT size
        self.FFT_offset = int((self.FFT_size - self.F) / 2)  # FFT offset
        self.CP = CP  # Cyclic prefix

        self.TTI_mask_RE = TTI_mask(S=self.S,F=self.F, Fp=self.Fp, Sp=self.Sp, FFT_offset=self.FFT_offset, plotTTI=False) #[14, 128]
        #among TTI_mask, 958 places are 1 (means data)
        self.pilot_symbols = pilot_set(self.TTI_mask_RE, Pilot_Power) #[36]

        mapping_table_QPSK, de_mapping_table_QPSK = mapping_table(2) # mapping table QPSK (e.g. for pilot symbols)
        self.mapping_table_Qm, self.de_mapping_table_Qm = mapping_table(Qm, plot=False) # mapping table for Qm

        #TTI_mask_RE_3d is the payload mapping used for inference
        # remove DC and FFT offsets from TTI mask_RE and add third dimension size of Qm, and expand TTI mask values into the third dimension
        TTI_mask_RE_small = self.TTI_mask_RE[:, self.FFT_offset:-self.FFT_offset] #FFT_offset=28 [14, 128]->[14, 72]
        middle_index = TTI_mask_RE_small.size(1) // 2 #36
        TTI_mask_RE_small = torch.cat((TTI_mask_RE_small[:, :middle_index], TTI_mask_RE_small[:, middle_index + 1:]), dim=1) #[14, 71]
        #TTI_mask_RE_3d = TTI_mask_RE_small.unsqueeze(-1).expand(val_batch_size, S, F-1, Qm) #[1, 14, 71, 6]
        TTI_mask_RE_3d = TTI_mask_RE_small.unsqueeze(-1) #Returns a new tensor with a dimension of size one inserted at the specified position. [14, 71]->[14, 71, 1]
        self.TTI_mask_RE_3d = TTI_mask_RE_3d.expand(self.S, self.F-1, self.Qm) #[14, 71, 6]
        self.index_one =  self.TTI_mask_RE_3d==1 #[14, 71, 6]

    #for all receivers
    def receiver_preprocessing(self, RX_Samples):
        #RX_Samples = batch['rx_samples']
        #step1: CP remove
        symbol_index = 1 #starting place
        RX_NO_CP = CP_removal(RX_Samples, symbol_index, self.S, self.FFT_size, self.CP, plotsig=False)# remove cyclic prefix and other symbols created by convolution
        RX_NO_CP = RX_NO_CP / torch.max(torch.abs(RX_NO_CP)) # normalize
        #torch.Size([14, 128])

        #back into the frequency domain
        OFDM_demod = DFT(RX_NO_CP, plotDFT=False) # DFT
        #batch['OFDM_demod'] = OFDM_demod
        return OFDM_demod

    def ZHLSreceiver(self, OFDM_demod):
        #OFDM_demod [14, 128]
        H_estim = channelEstimate_LS(self.TTI_mask_RE, self.pilot_symbols, self.F, self.FFT_offset, self.Sp, OFDM_demod, plotEst=False) # estimate the channel using least squares and plot

        OFDM_demod_no_offsets = remove_fft_Offests(OFDM_demod, self.F, self.FFT_offset) # remove the FFT offsets and DC carrier from the received signal
        #[14, 72]
        #[14, 128]->[14, 72] (28:28+72)

        # equalize the received signal
        equalized_H_estim = equalize_ZF(OFDM_demod_no_offsets, H_estim, self.F, self.S) # equalize the channel using ZF
        #[14, 72]

        #Payload Symbols extraction
        QAM_est = get_payload_symbols(self.TTI_mask_RE, equalized_H_estim, self.FFT_offset, self.F, plotQAM=False) # get the payload symbols from
        #[958]

        #Converting OFDM Symbols to Data
        PS_est, hardDecision = Demapping(QAM_est, self.de_mapping_table_Qm) # demap the symbols back to codewords
        #PS_est[958, 6] bits
        #hardDecision[958] mapped complex value
        #[958, 6] bits =>[5748]
        binary_predictions = PS(PS_est) # convert the codewords to the bitstream
        #0 1 bits [5748]
        # convert to bits
        #binary_predictions = torch.tensor(PS(PS_est).flatten().cpu(), dtype=torch.int8) #[5748]
        return binary_predictions


    def NNpreprocessing(self, OFDM_demod):
        OFDM_demod = OFDM_demod / torch.max(torch.abs(OFDM_demod)) # normalize DFT'd signal for NN input
        #torch.Size([14, 128])
        #F is number of carriers
        pdsch_symbols_map = remove_fft_Offests(OFDM_demod, self.F, self.FFT_offset) # remove FFT offsets
        #[14, 72]
        # remove DC
        pdsch_symbols_map = torch.cat((pdsch_symbols_map[:, :self.F//2], pdsch_symbols_map[:, self.F//2 + 1:]), dim=1) 
        # [14, 71]
        return pdsch_symbols_map

    def NNinference(self, model, pdsch_symbols_map, device):
        pdsch_symbols_map=torch.unsqueeze(pdsch_symbols_map, dim=0).to(device) #[14, 71]=>[1, 14, 71]
        test_outputs = model(pdsch_symbols_map) #[1, 14, 71]->[1, 14, 71, 6]
        #binary_predictions = test_outputs.squeeze()[TTI_mask_RE_3d==1] #[91968]
        binary_predictions = test_outputs.squeeze() #[14, 71, 6]
        
        #Fetch the payload
        binary_predictions = binary_predictions[self.index_one] #[5748]
        binary_predictions = torch.round(binary_predictions)
        return binary_predictions.cpu()
    
    def NNevaluate(self, binary_predictions, test_labels):
        binary_predictions = binary_predictions.squeeze() #[14, 71, 6]
        test_labels = test_labels.squeeze() #[14, 71, 6]
        #index_one =  TTI_mask_RE_3d==1
        test_labels = test_labels[self.index_one] #[5748]
        binary_predictions = binary_predictions[self.index_one] #[5748]
        
        # Calculate Bit Error Rate (BER) for the NN-receiver
        error_count = torch.sum(binary_predictions != test_labels).float()  # Count of unequal bits

        new_wrongs = (binary_predictions.flatten() != test_labels.flatten()).float().tolist() #each element is [5748]
        error_rate = error_count / len(test_labels.flatten())  # Error rate calculation
        BER_val = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
        return BER_val.item(), new_wrongs
    
    def evaluate(self, binary_predictions, test_labels):
        #binary_predictions: [5748]
        test_labels = test_labels.squeeze() #[14, 71, 6]
        #index_one =  TTI_mask_RE_3d==1
        test_labels = test_labels[self.index_one] #[5748]
        
        # Calculate Bit Error Rate (BER) for the NN-receiver
        error_count = torch.sum(binary_predictions != test_labels).float()  # Count of unequal bits

        new_wrongs = (binary_predictions.flatten() != test_labels.flatten()).float().tolist() #each element is [5748]
        error_rate = error_count / len(test_labels.flatten())  # Error rate calculation
        BER_val = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
        return BER_val.item(), new_wrongs


def trainmain(trainoutput, saved_model_path = ""):
    device, useamp=get_device(gpuid='0', useamp=False)

    # OFDM Parameters
    #Qm (int): Modulation order
    Qm = 6  # bits per symbol
    S = 14  # Number of symbols
    Sp = 2  # Pilot symbol, 0 for none
    F = 72  # Number of subcarriers, including DC
    #5-50 0331exp
    #-10 20 exp0201
    #-10 40 exp0201b
    # exp0201c ResModel_simple1_2D
    train_data = OFDMDataset(Qm=Qm, S=S, Sp=Sp, F=F, ch_SINR_min=-10, ch_SINR_max=40, training=True)
    onebatch = train_data[0]

    batch_size = 16

    # train, validation and test split
    train_size = int(0.8 * len(train_data)) #8000
    val_size = len(train_data) - train_size
    train_set, val_set= torch.utils.data.random_split(train_data, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

    onebatch = next(iter(train_loader))
    rx_samples = onebatch['samples']
    feature_2d = onebatch['feature_2d']
    data_labels = onebatch['labels']
    print(f"Sample batch shape: {rx_samples.size()}")
    print(f"Feature batch shape: {feature_2d.size()}") #[16, 4, 14, 71]
    print(f"Labels batch shape: {data_labels.size()}") #[16, 14, 71, 6]

    model =  ResModel_2D(num_bits_per_symbol=Qm, num_ch=4, S=S, F=F).to(device)
    #model = ResModel_simple1_2D(num_bits_per_symbol=Qm, num_ch=4, S=S, F=F).to(device)
    #model = MyWave2vec(num_bits_per_symbol=Qm, num_ch=4, S=S, F=F).to(device)

    multiprocessor = MultiReceiver(Qm=Qm, S=S, Sp=Sp, F=F)

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

     #'data/rx_model_168.pth'
    # trainoutput=os.path.join('output','exp0202b')
    # os.makedirs(trainoutput, exist_ok=True)
    # print("Trainoutput folder:", trainoutput)
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
            feature_2d = batch['feature_2d'] #[16, 4, 14, 71]
            labels = batch['labels'] #[16, 14, 71, 6]
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
                rx_samples = batch['samples'] #[1, 2078]
                val_outputs = model((feature_2d)) #[1, 14, 71, 6]
                val_loss = criterion(val_outputs, labels)

                # Convert probabilities to binary predictions (0 or 1)
                binary_predictions = torch.round(val_outputs) #[1, 14, 71, 6]

                # Calculate Bit Error Rate (BER)
                labels = labels.cpu().squeeze() #[1, 14, 71, 6]
                BER, NN_wrongs = multiprocessor.NNevaluate(binary_predictions.cpu(), labels)
                BER_batch.append(BER)

                rx_samples=rx_samples.squeeze().cpu()
                #back into the frequency domain
                OFDM_demod = multiprocessor.receiver_preprocessing(rx_samples) #[14, 128]
                ZHLS_binary_predictions = multiprocessor.ZHLSreceiver(OFDM_demod)
                ZHLS_BER, ZHLS_wrongs = multiprocessor.evaluate(ZHLS_binary_predictions, labels)
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
    plt.plot(df['Epoch'], df['LS_BER'], label='LS BER')
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OFDM training job')
    #data related arguments
    parser.add_argument('--mode', default="Train", choices=['Train','Evaluate', 'Visualization'], help='Running mode')
    parser.add_argument('--traintag', type=str, default='exp806',
                    help='Train rag name, used for output folder')
    parser.add_argument('--data_type', type=str, default="OFDMsim",
                    help='data type name')
    parser.add_argument('--data_name', type=str, default="",
                    help='data name')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/Huggingfacecache", help='Huggingface data cache folder') #r"D:\Cache\huggingface", "/data/cmpe249-fa23/Huggingfacecache" "/DATA10T/Cache"
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='outputpath') #r"E:\output"
    
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    trainoutput=os.path.join(args.outputdir, args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    if args.mode == "Evaluate":
        draw_trainresults(csv_path=os.path.join(trainoutput, 'performance.csv'), save_plots=True)
    elif args.mode == "Train":
        trainmain(trainoutput)
