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

# custom dataset
class OFDMEvalDataset(Dataset):
    def __init__(self, Qm=6, S=14, Sp=2, F=72, Fp=2, FFT_size=128, CP=20, ch_SINR_min=25, ch_SINR_max=50, maxdatalen=10000):
        self.maxdatalen = maxdatalen
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
        
        return RX_Samples, TTI_3d

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
        #step1: CP remove
        symbol_index = 1 #starting place
        RX_NO_CP = CP_removal(RX_Samples, symbol_index, self.S, self.FFT_size, self.CP, plotsig=False)# remove cyclic prefix and other symbols created by convolution
        RX_NO_CP = RX_NO_CP / torch.max(torch.abs(RX_NO_CP)) # normalize
        #torch.Size([14, 128])

        #back into the frequency domain
        OFDM_demod = DFT(RX_NO_CP, plotDFT=False) # DFT
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

    def NNinference(self, model, pdsch_symbols_map):
        test_outputs = model(pdsch_symbols_map) #[1, 14, 71]->[1, 14, 71, 6]
        #binary_predictions = test_outputs.squeeze()[TTI_mask_RE_3d==1] #[91968]
        binary_predictions = test_outputs.squeeze() #[14, 71, 6]
        
        #Fetch the payload
        binary_predictions = binary_predictions[self.index_one] #[5748]
        binary_predictions = torch.round(binary_predictions)
        return binary_predictions
    
    def evaluate(self, binary_predictions, test_labels):
        test_labels = test_labels.squeeze() #[14, 71, 6]
        #index_one =  TTI_mask_RE_3d==1
        test_labels = test_labels[self.index_one] #[5748]
        
        # Calculate Bit Error Rate (BER) for the NN-receiver
        error_count = torch.sum(binary_predictions != test_labels).float()  # Count of unequal bits

        new_wrongs = (binary_predictions.flatten() != test_labels.flatten()).float().tolist() #each element is [5748]
        error_rate = error_count / len(test_labels.flatten())  # Error rate calculation
        BER_val = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
        return BER_val.item(), new_wrongs

def test():
    device, useamp=get_device(gpuid='0', useamp=False)

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

    eval_data = OFDMEvalDataset(Qm=Qm, S=S, Sp=Sp, F=F)
    RX_Samples, TTI_3d = eval_data[0]
    test_dataloader = DataLoader(eval_data, batch_size=1, shuffle=True)
    rx_samples, data_labels = next(iter(test_dataloader))
    print(f"Feature batch shape: {rx_samples.size()}") #[Batch_size, 2078]
    print(f"Labels batch shape: {data_labels.size()}") #[Batch_size, 14, 71, 6]
    rx_samples=rx_samples.squeeze() #[2078]
    data_labels=data_labels.squeeze() #[14, 71, 6]
    

    multiprocessor = MultiReceiver(Qm=Qm, S=S, Sp=Sp, F=F)
    #back into the frequency domain
    OFDM_demod = multiprocessor.receiver_preprocessing(rx_samples) #[14, 128]

    ZHLS_binary_predictions = multiprocessor.ZHLSreceiver(OFDM_demod)
    ZHLS_BER, ZHLS_wrongs = multiprocessor.evaluate(ZHLS_binary_predictions, data_labels)

    pdsch_symbols_map = multiprocessor.NNpreprocessing(OFDM_demod) #[14, 71]
    NN_binary_predictions = multiprocessor.NNinference(model, pdsch_symbols_map)
    NN_BER, NN_wrongs = multiprocessor.evaluate(ZHLS_binary_predictions, data_labels)
    print(NN_BER)

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
    test()
    evalmain()