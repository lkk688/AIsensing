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
import pickle

def evalmain():
    device, useamp=get_device(gpuid='0', useamp=False)

    dataset = CustomDataset()
    dataset = torch.load('output/ofdm_dataset.pth')
    batch_size = 16
    val_batch_size = 1

    # train, validation and test split
    train_size = int(0.8 * len(dataset)) #8000
    val_size = len(dataset) - train_size
    train_set, val_set= torch.utils.data.random_split(dataset, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, batch_size=val_batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # OFDM Parameters
    #Qm (int): Modulation order
    Qm = 6  # bits per symbol
    S = 14  # Number of symbols
    Sp = 2  # Pilot symbol, 0 for none
    F = 72  # Number of subcarriers, including DC
    model = RXModel_2(Qm, S=S, F=F).to(device)
    # Load the model architecture and weights
    checkpoint_path = 'output/rx_model_50.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    TTI_mask_RE = TTI_mask(S=S,F=F, Fp=Fp, Sp=Sp, FFT_offset=FFT_offset, plotTTI=False).to(device) # TTI mask [14, 128]
    pilot_symbols = pilot_set(TTI_mask_RE, Pilot_Power).to(device) # pilot symbols [36]

    mapping_table_QPSK, de_mapping_table_QPSK = mapping_table(2) # mapping table QPSK (e.g. for pilot symbols) len=4
    mapping_table_Qm, de_mapping_table_Qm = mapping_table(Qm, plot=False) # mapping table for Qm, len=64

    # Evaluate the model on the test set
    test_losses_NN = []
    test_BERs_NN = []
    test_BERs_ZFLS = []

    # remove DC and FFT offsets from TTI mask_RE and add third dimension size of Qm, and expand TTI mask values into the third dimension
    TTI_mask_RE_small = TTI_mask_RE[:, FFT_offset:-FFT_offset] #FFT_offset=28 [14, 72]
    middle_index = TTI_mask_RE_small.size(1) // 2 #36
    TTI_mask_RE_small = torch.cat((TTI_mask_RE_small[:, :middle_index], TTI_mask_RE_small[:, middle_index + 1:]), dim=1) #[14, 71]
    #TTI_mask_RE_3d = TTI_mask_RE_small.unsqueeze(-1).expand(val_batch_size, S, F-1, Qm) #[1, 14, 71, 6]
    TTI_mask_RE_3d = TTI_mask_RE_small.unsqueeze(-1) #Returns a new tensor with a dimension of size one inserted at the specified position. [14, 71]->[14, 71, 1]
    TTI_mask_RE_3d = TTI_mask_RE_3d.expand(S, F-1, Qm)
    #[14, 71, 6]
    wrongs=[]
    wrongs_ls=[]
    with torch.no_grad():
        for test_pdsch_iq, test_labels in val_loader: #[1, 14, 71]
            # NN receiver ###################################################
            test_pdsch_iq,  test_labels = test_pdsch_iq.to(device), test_labels.to(device)
            test_outputs = model(test_pdsch_iq) #[1, 14, 71]->[1, 14, 71, 6]
            #binary_predictions = test_outputs.squeeze()[TTI_mask_RE_3d==1] #[91968]
            binary_predictions = test_outputs.squeeze() #[14, 71, 6]
            index_one =  TTI_mask_RE_3d==1 #[14, 71, 6]
            binary_predictions = binary_predictions[index_one] #[5748]
            binary_predictions = torch.round(binary_predictions)
            print(binary_predictions.shape) #[5748]
            #test_labels = test_labels.squeeze()[TTI_mask_RE_3d==1] #[91968]
            test_labels = test_labels.squeeze() #[14, 71, 6]
            #index_one =  TTI_mask_RE_3d==1
            test_labels = test_labels[index_one] #[5748]

            # load a batch of data to the device
            test_pdsch_iq, test_labels = test_pdsch_iq.squeeze().to(device), test_labels.squeeze().to(device) #[14, 71]

            # Calculate Bit Error Rate (BER) for the NN-receiver
            error_count = torch.sum(binary_predictions != test_labels).float()  # Count of unequal bits

            new_wrongs = (binary_predictions.flatten() != test_labels.flatten()).float().tolist() #each element is [5748]
            wrongs.append(new_wrongs)

            error_rate = error_count / len(test_labels.flatten())  # Error rate calculation
            BER_NN = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
            test_BERs_NN.append(BER_NN.item())

            # ZF-LS receiver ####################################################
            # add DC and FFT offsets, as channel estimation expects those
            offsets = torch.zeros(S, FFT_offset, dtype=test_pdsch_iq.dtype, device=device) #[14, 28]
            test_pdsch_iq_w = torch.cat((offsets, test_pdsch_iq, offsets), dim=1) # add FFT offset =>[14, 127]

            middle_index = test_pdsch_iq_w.size(1) // 2+1 #64
            test_pdsch_iq_w = torch.cat((test_pdsch_iq_w[:, :middle_index], torch.zeros(S, 1, device=device), test_pdsch_iq_w[:, middle_index:]), dim=1) # add DC [14, 128]
            
            test_pdsch_iq_w_t = test_pdsch_iq_w
            
            # calculate channel estimate
            H_estim = channelEstimate_LS(TTI_mask_RE.to('cpu'), pilot_symbols.to('cpu'), F, FFT_offset, Sp, test_pdsch_iq_w.to('cpu'), plotEst=False) #[72]

            # remove FFT offsets
            test_pdsch_iq_w = remove_fft_Offests(test_pdsch_iq_w, F, FFT_offset) #[14, 128]->[14, 72] (28:28+72)
            
            # equalize the received signal
            equalized_H_estim = equalize_ZF(test_pdsch_iq_w, H_estim, F, S) #[14, 72]

            # get the payload symbols 
            QAM_est = get_payload_symbols(TTI_mask_RE, equalized_H_estim, FFT_offset, F, plotQAM=False) #mask=1 [958]

            # demap the symbols
            PS_est, hardDecision = Demapping(QAM_est.to('cpu'), de_mapping_table_Qm) #PS_est [958, 6] bits, [958] mapped complex value

            # convert to bits
            bits_est = torch.tensor(PS(PS_est).flatten().cpu(), dtype=torch.int8) #[5748]

            # Calculate Bit Error Rate (BER) for the ZF-LS receiver
            test_labels = torch.tensor(test_labels.flatten().cpu(), dtype=torch.int8) #[5748]
            
            new_wrongs_ls = (bits_est != test_labels).float().tolist()
            wrongs_ls.append(new_wrongs_ls) #5748 len
            
            error_count = torch.sum(bits_est != test_labels).float()  # Count of unequal bits
            error_rate = error_count / bits_est.numel()  # Error rate calculation
            BER_ZFLS = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
            test_BERs_ZFLS.append(BER_ZFLS.item())
    
    result_dict={}
    result_dict['wrongs']=wrongs
    result_dict['wrongs_ls']=wrongs_ls
    result_dict['BERs_NN']=test_BERs_NN
    result_dict['BERs_ZFLS']=test_BERs_ZFLS #2000
    with open('output/eval_data.pkl', 'wb') as fp:
        pickle.dump(result_dict, fp)
        print('dictionary saved successfully to file')

if __name__ == '__main__':
    evalmain()