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

# custom dataset
class OFDMDataset(Dataset):
    def __init__(self, datapath='data/AISimData/cdl_ofdm_ebno25.npy', ch_SINR_min=25, ch_SINR_max=50, maxdatalen=10000, training=False, drawfig=True):
        self.maxdatalen = maxdatalen
        self.training = training
        self.drawfig = drawfig
        #Signal-to-Interference-plus-Noise Ratio (SINR) for the CDL-C channel emulation
        # in case SDR not available, for channel simulation
        self.ch_SINR_min = ch_SINR_min # channel emulation min SINR
        self.ch_SINR_max = ch_SINR_max # channel emulation max SINR
        saved_data = np.load(datapath, allow_pickle=True).item()
        for k, v in saved_data.items():
            if isinstance(v, np.ndarray):
                print(f"{k}'s shape: {v.shape}")
            else:
                print(f"{k}: {v}")
        self.channeltype = saved_data['channeltype']# ofdm
        self.channeldataset = saved_data['channeldataset'] #cdl
        self.fft_size = saved_data['fft_size'] #76
        self.batch_size = saved_data['batch_size']
        self.num_ofdm_symbols = saved_data['num_ofdm_symbols'] #14
        self.num_bits_per_symbol = saved_data['num_bits_per_symbol'] #2
        self.num_ut = saved_data['num_ut'] #1
        self.num_bs = saved_data['num_bs'] #1
        self.num_ut_ant = saved_data['num_ut_ant'] #2
        self.num_bs_ant = saved_data['num_bs_ant'] #16
        self.direction = saved_data['direction'] #uplink
        self.num_streams_per_tx = saved_data['num_streams_per_tx'] #2
        self.k = saved_data['k'] #1536
        self.n = saved_data['n'] #1536
        self.b = saved_data['b'] # b's shape: (128, 1, 2, 1536) [self.batch_size, 1, self.num_streams_per_tx, self.k]
        # After mapper ([batch_size, num_tx, num_streams_per_tx, num_data_symbols]) 
        self.x = saved_data['x'] #x's shape: (128, 1, 2, 768)
        #after rg_mapper
        self.x_rg = saved_data['x_rg']
        #x_rg's shape: (128, 1, 2, 14, 76) [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]

        #Channel IR get_channelcir
        self.h_b = saved_data['h_b'] #h_b's shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps] (128, 1, 16, 1, 2, 23, 14)
        self.tau_b = saved_data['tau_b'] # tau_b's shape: [batch, num_rx, num_tx, num_paths] (128, 1, 1, 23)
        #generated OFDM or time channel
        self.h_out = saved_data['h_out'] # h_out's shape: (128, 1, 16, 1, 2, 14, 76)
        #h_freq.shape #[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers] (64, 1, 16, 1, 2, 14, 76)
        #[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size](2, 1, 1, 1, 16, 1, 76)
        
        
        #channel output, y = self.applychannel([x_rg, h_out, no])
        self.y = saved_data['y']
        # y's shape: (128, 1, 16, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        #channelest_equ
        self.h_hat = saved_data['h_hat'] #h_hat shape: (128, 1, 16, 1, 2, 14, 64)
        self.err_var = saved_data['err_var'] #(1, 1, 1, 1, 2, 14, 64)
        self.h_perfect = saved_data['h_perfect'] #(128, 1, 16, 1, 2, 14, 64)
        self.err_var_perfect = saved_data['err_var_perfect'] #0.0

        #after channel equalization
        self.x_hat = saved_data['x_hat'] # x_hat's shape: (128, 1, 2, 768)
        self.no_eff = saved_data['no_eff']

        self.b_hat = saved_data['b_hat'] #(128, 1, 2, 1536)
        self.llr_est = saved_data['llr_est'] #(128, 1, 2, 1536)
        self.BER = saved_data['BER']  
        self.frequencies = saved_data['frequencies']
        self.num_time_steps = saved_data['num_time_steps']
        self.sampling_frequency = saved_data['sampling_frequency']
        
        self.check_channel()

        
    def check_channel(self, compare=True):
        from deepMIMO5 import time_lag_discrete_time_channel, cir_to_time_channel, cir_to_ofdm_channel, subcarrier_frequencies
        if self.drawfig:
             #eval_transceiver.RESOURCE_GRID.num_ofdm_symbols
            #sampling_frequency= saved_data['sampling_frequency'] #1/eval_transceiver.RESOURCE_GRID.ofdm_symbol_duration
            plt.figure()
            plt.title("Channel impulse response realization")
            plt.stem(self.tau_b[0,0,0,:]/1e-9, np.abs(self.h_b)[0,0,0,0,0,:,0])#10 different pathes
            plt.xlabel(r"$\tau$ [ns]")
            plt.ylabel(r"$|a|$")

            plt.figure()
            plt.title("Time evolution of path gain")
            #x_timesteps = np.arange(num_time_steps)*self.RESOURCE_GRID.ofdm_symbol_duration/1e-6
            x_timesteps = np.arange(self.num_time_steps)/self.sampling_frequency/1e-6
            plt.plot(x_timesteps, np.real(self.h_b)[0,0,0,0,0,0,:])
            plt.plot(x_timesteps, np.imag(self.h_b)[0,0,0,0,0,0,:])
            plt.legend(["Real part", "Imaginary part"])
            plt.xlabel(r"$t$ [us]")
            plt.ylabel(r"$a$");
        h_freq_np = cir_to_ofdm_channel(self.frequencies, self.h_b, self.tau_b, normalize=True)
        #(128, 1, 16, 1, 2, 14, 76)
        print(np.allclose(h_freq_np, self.h_out)) #True False
        print(np.allclose(h_freq_np[:,:,:,0,0,:,:], self.h_out[:,:,:,0,0,:,:])) #True
        print(np.allclose(h_freq_np[:,0,0,0,0,:,:], self.h_out[:,0,0,0,0,:,:])) #True

        if compare == True:
            from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
            h_freq_tf = cir_to_ofdm_channel(self.frequencies, self.h_b, self.tau_b, normalize=True)
            print(np.allclose(h_freq_np, h_freq_tf.numpy())) #True

    def check_channelestimation(self):
        if self.drawfig:
            h_perfect = self.h_perfect[0,0,0,0,0,0]
            h_hat = self.h_hat[0,0,0,0,0,0]
            plt.figure()
            plt.plot(np.real(h_perfect))
            plt.plot(np.imag(h_perfect))
            plt.plot(np.real(h_hat), "--")
            plt.plot(np.imag(h_hat), "--")
            plt.xlabel("Subcarrier index")
            plt.ylabel("Channel frequency response")
            plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
            plt.title("Comparison of channel frequency responses");

    
    def __len__(self):
        return self.maxdatalen
    
    def __getitem__(self, index, rx_id=0, tx_id=0, tx_streams_id=0, rx_antenna_id=0):
        batch={}
        if self.training:
            data = self.b[:,tx_id, tx_streams_id, :] #[batch_size, num_tx, num_streams_per_tx, num_data_bits] #[self.batch_size, 1, self.num_streams_per_tx, self.k]
            batch['labels']= data
        #(128, 1, 16, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        rx_samples = self.y[:,rx_id, rx_antenna_id, :, :] #(128, 14, 76)
        batch['samples']= rx_samples
        return batch
    
def trainmain(args):
    train_data = OFDMDataset(training=True)
    onebatch = train_data[0]
    print(onebatch['samples'].shape)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OFDM training job')
    #data related arguments
    parser.add_argument('--mode', default="Evaluate", choices=['Train','Evaluate', 'Visualization'], help='Running mode')
    parser.add_argument('--traintag', type=str, default='exp0712',
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

    trainmain(args)