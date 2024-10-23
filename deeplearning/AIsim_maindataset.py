import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from deepMIMO5 import hard_decisions, calculate_BER
import random

def compare_allclose(arry1, arry2, threshold=1e-6, figname="data/compare_allclose.png"):
    is_complex=False
    if any(np.iscomplex(arry1)) and any(np.iscomplex(arry2)):
        print("complex data")
        is_complex = True
        print("np allclose for real", np.allclose(np.real(arry1), np.real(arry2))) #False
        print("np allclose for img", np.allclose(np.imag(arry1), np.imag(arry2))) #False
    differences = np.abs(arry1 - arry2)
    
    num_differences = np.sum(differences > threshold)
    print("Percent of differences:", num_differences/len(arry1))
    print("np allclose", np.allclose(arry1, arry2, atol=threshold))
    print("Demodulation error (L2 norm):", np.linalg.norm(arry1 - arry2))
    if figname is not None:
        plt.figure()
        if is_complex:
            plt.plot(np.real(arry1))
            plt.plot(np.imag(arry1))
            plt.plot(np.real(arry2), "--")
            plt.plot(np.imag(arry2), "--")
        else:
            plt.plot(arry1)
            plt.plot(arry2, "--")
        plt.savefig(figname)

    
# custom dataset
class OFDMDataset(Dataset):
    def __init__(self, datapath='data/cdl_ofdm_ebno25.npy', ch_SINR_min=25, ch_SINR_max=50, maxdatalen=10000, training=False, drawfig=False, testing=False, compare=False):
        self.maxdatalen = maxdatalen
        self.training = training
        self.drawfig = drawfig
        #Signal-to-Interference-plus-Noise Ratio (SINR) for the CDL-C channel emulation
        # in case SDR not available, for channel simulation
        self.ch_SINR_min = ch_SINR_min # channel emulation min SINR
        self.ch_SINR_max = ch_SINR_max # channel emulation max SINR
        saved_data = np.load(datapath, allow_pickle=True)
        saved_data = saved_data.item()
        # for k, v in saved_data.items():
        #     if isinstance(v, np.ndarray):
        #         print(f"{k}'s shape: {v.shape}")
        #     else:
        #         print(f"{k}: {v}")
        print("Dataset time:", saved_data['currenttime'])
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
        if self.direction=="uplink": #the UT is transmitting.
            self.num_tx = self.num_ut
            self.num_rx = self.num_bs
            #num_streams_per_tx = num_ut_ant #num_rx ##1
        else:#downlink
            self.num_tx = self.num_bs
            self.num_rx = self.num_ut
            #num_streams_per_tx = num_bs_ant #num_rx ##1
        self.num_streams_per_tx = saved_data['num_streams_per_tx'] #2
        self.no=saved_data['no']
        self.k = saved_data['k'] #1536
        self.n = saved_data['n'] #1536
        self.b = saved_data['b'] # b's shape: (128, 1, 2, 1536) [self.batch_size, 1, self.num_streams_per_tx, self.k]
        # After mapper ([batch_size, num_tx, num_streams_per_tx, num_data_symbols]) 
        self.x = saved_data['x'] #x's shape: (128, 1, 2, 768) data_symbols 1536/2=768
        #after rg_mapper
        self.x_rg = saved_data['x_rg']
        #x_rg's shape: (128, 1, 2, 14, 76) [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #RESOURCE_GRID.num_data_symbols=14(OFDM symbol)*76(subcarrier) array=1064
        #among 76 subcarriers, 5 and 6 are guard carriers, effective subcarrier is 76-6-5-1(DC carrier)=64
        #among 14 symbols, 2,11 is the pilot, effective symbol is 12
        #1064 grids contains the data, DC and pilot, effective grid=12*64=768
        
        num_guard_carriers=[5,6]
        pilot_ofdm_symbol_indices=[2,11]
        colofpilots=len(pilot_ofdm_symbol_indices) #2
        self.totalsymbols = int(self.k/self.num_bits_per_symbol)
        self.effectiveofdmsymbols=self.num_ofdm_symbols-colofpilots #12
        self.effectivesubcarrier=int(self.totalsymbols/(self.effectiveofdmsymbols)) #768/12=64
        self.TTI_mask_RE = self.TTI_mask(S=14, F=64, num_guard_carriers=num_guard_carriers, \
                                         pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices, dc_null=True, plotTTI=drawfig)#(14, 76)
        # TTI_mask_RE_3d = TTI_mask_RE_small.unsqueeze(-1) #Returns a new tensor with a dimension of size one inserted at the specified position. [14, 71]->[14, 71, 1]
        # self.TTI_mask_RE_3d = TTI_mask_RE_3d.expand(self.S, self.F-1, self.Qm) #[14, 71, 6]
        # self.index_one =  self.TTI_mask_RE_3d==1 #[14, 71, 6]

        #RESOURCE_GRID related data
        self.pilot_pattern = saved_data['pilot_pattern']
        self.pilots = saved_data['pilots'] #self.RESOURCE_GRID.pilot_pattern.pilots
        self.num_data_symbols = saved_data['num_data_symbols']
        self.cyclic_prefix_length = saved_data['cyclic_prefix_length']
        self.ofdm_symbol_duration = saved_data['ofdm_symbol_duration']
        self.num_time_samples = saved_data['num_time_samples']
        self.bandwidth = saved_data['bandwidth']

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
        self.frequencies = saved_data['frequencies'] #(76,)
        self.num_time_steps = saved_data['num_time_steps'] #(14,)
        self.sampling_frequency = saved_data['sampling_frequency'] #55609
        
        self.index = 0
        rx_id=0
        tx_id=0
        tx_streams_id=0
        rx_antenna_id=0
        returnbatch=False

        if self.training:
            data = self.b[:,tx_id, tx_streams_id, :] #[batch_size, num_tx, num_streams_per_tx, num_data_bits] #[self.batch_size, 1, self.num_streams_per_tx, self.k]
            #(128, 1536) [batch_size, num_data_bits]
            self.labels_data = data.reshape(-1, self.effectiveofdmsymbols, self.effectivesubcarrier, self.num_bits_per_symbol) #(128, 12, 64, 2)
            #(128, 12, 64, 2)
            #labelsize = (self.num_ofdm_symbols, self.fft_size, self.num_bits_per_symbol) #14, 76, 2
        #(128, 1, 16, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        rx_samples = self.y[:,rx_id, rx_antenna_id, :, :] #(128, 14, 76) [batch_size, num_ofdm_symbols, fft_size]

        #self.TTI_mask_RE #(14, 76)
        TTI_mask_indices = np.where(self.TTI_mask_RE==1)
        rx_samples_eff = rx_samples[:, TTI_mask_indices[0], TTI_mask_indices[1]]
        #print(rx_samples_eff.shape) #(128, 768)
        self.rx_samples_eff= rx_samples_eff.reshape(-1, self.effectiveofdmsymbols, self.effectivesubcarrier) #(128, 12, 64)

        if testing:
            from deepMIMO5 import StreamManagement, MyResourceGrid, MyResourceGridMapper, MyDemapper, RemoveNulledSubcarriers
            from sionna_tf import MyLMMSEEqualizer
            from channel import MyLSChannelEstimator, LSChannelEstimator
            self.RESOURCE_GRID = MyResourceGrid(num_ofdm_symbols=self.num_ofdm_symbols,
                    fft_size=self.fft_size,
                    subcarrier_spacing=60e3, #15e3,
                    num_tx=1,
                    num_streams_per_tx=self.num_streams_per_tx,
                    cyclic_prefix_length=6,
                    num_guard_carriers=[5,6],
                    dc_null=True,
                    pilot_pattern="kronecker",
                    pilot_ofdm_symbol_indices=[2,11])
            pilots = self.RESOURCE_GRID.pilot_pattern.pilots #(1, 2, 128)
            print(self.RESOURCE_GRID.num_effective_subcarriers)
            print(self.RESOURCE_GRID.effective_subcarrier_ind)
            RESOURCE_GRID2 = MyResourceGrid(num_ofdm_symbols=self.num_ofdm_symbols,
                    fft_size=self.fft_size,
                    subcarrier_spacing=60e3, #15e3,
                    num_tx=1,
                    num_streams_per_tx=self.num_streams_per_tx,
                    cyclic_prefix_length=6,
                    num_guard_carriers=[5,6],
                    dc_null=True,
                    pilot_pattern="kronecker",
                    pilot_ofdm_symbol_indices=[2,11])
            pilots2 = RESOURCE_GRID2.pilot_pattern.pilots #(1, 2, 128)
            print(pilots[0,0,:])
            print(pilots2[0,0,:])
            #The pilots are different, Each pilot sequence is constructed from randomly drawn QPSK constellation points.
            print(np.allclose(pilots, pilots2)) #False
            self.RESOURCE_GRID.pilot_pattern.pilots = self.pilots #changed here

            self.myrg_mapper = MyResourceGridMapper(self.RESOURCE_GRID)
            self.remove_nulled_scs = RemoveNulledSubcarriers(self.RESOURCE_GRID)
            self.mydemapper = MyDemapper("app", constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol)
            RX_TX_ASSOCIATION = np.ones([self.num_rx, self.num_tx], int) #[[1]]
            self.STREAM_MANAGEMENT = StreamManagement(RX_TX_ASSOCIATION, self.num_streams_per_tx)
            self.lmmse_equ = MyLMMSEEqualizer(self.RESOURCE_GRID, self.STREAM_MANAGEMENT)
            self.ls_est = MyLSChannelEstimator(self.RESOURCE_GRID, interpolation_type="nn")#"lin_time_avg")
            #self.ls_est = LSChannelEstimator(self.RESOURCE_GRID, interpolation_type="nn")

            self.compare_channelestimationdata()
            self.checkchannelestimate()
            self.receive()
            self.check_uplinktransmission(compare=compare)
            self.check_channel(compare=compare)

    def check_uplinktransmission(self, compare=True):
        c = self.b
        x = self.x
        x_rg = self.x_rg
        #x_rg's shape: (128, 1, 2, 14, 76) 
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]

        if compare:
            #from sionna.mapping import Mapper, Demapper
            #from sionna.ofdm import ResourceGrid, ResourceGridMapper
            from sionna_tf import Mapper #, ResourceGrid, ResourceGridMapper
            #from deepMIMO5 import MyResourceGrid, MyResourceGridMapper #StreamManagement, MyResourceGrid, Mapper, MyResourceGridMapper, MyDemapper

            # The mapper maps blocks of information bits to constellation symbols
            mapper = Mapper("qam", self.num_bits_per_symbol)

            # The resource grid mapper maps symbols onto an OFDM resource grid

            x_tf = mapper(c) #np.array[64,1,1,896] if empty np.array[64,1,1,1064] 1064*4=4256 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
            x_rg_tf = self.myrg_mapper(x_tf) ##complex array[64,1,1,14,76] 14*76=1064
            #output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76] (64, 1, 2, 14, 76)
            
            myx_rg = self.myrg_mapper(x)

            print(np.allclose(x, x_tf.numpy())) #True
            print(np.allclose(x_rg, x_rg_tf)) #False->True
            print(np.allclose(x_rg[0,:,:,:,:], x_rg_tf[0,:,:,:,:])) #False->True
            print(np.allclose(x_rg[0,0,:,:,:], x_rg_tf[0,0,:,:,:])) #False->True
            print(np.allclose(x_rg[0,0,0,:,:], x_rg_tf[0,0,0,:,:])) #False->True

            print(np.allclose(x_rg, myx_rg)) #False->True
            print(np.allclose(x_rg[0,0,0,:,:], myx_rg[0,0,0,:,:])) #False->True

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
        print(np.allclose(h_freq_np, self.h_out)) #True
        print(np.allclose(h_freq_np[:,:,:,0,0,:,:], self.h_out[:,:,:,0,0,:,:])) #True
        print(np.allclose(h_freq_np[:,0,0,0,0,:,:], self.h_out[:,0,0,0,0,:,:])) #True

        if compare == True:
            #from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
            from channel import cir_to_ofdm_channel
            import tensorflow as tf
            h_b_tf = tf.convert_to_tensor(self.h_b, dtype=tf.complex64)
            #tau_b_tf = tf.convert_to_tensor(self.tau_b, dtype=tf.float)
            h_freq_tf = cir_to_ofdm_channel(self.frequencies, h_b_tf, self.tau_b, normalize=True) #(128, 1, 16, 1, 2, 14, 76)
            h_freq_tfnp = h_freq_tf.numpy()
            print(np.allclose(self.h_out, h_freq_tfnp, atol=1e-06)) #False->True
            print(np.allclose(self.h_out[:,:,:,0,0,:,:], h_freq_tfnp[:,:,:,0,0,:,:])) #False
            print(np.allclose(self.h_out[:,0,0,0,0,:,:], h_freq_tfnp[:,0,0,0,0,:,:])) #False
            print(np.allclose(self.h_out[0,0,0,0,0,:,:], h_freq_tfnp[0,0,0,0,0,:,:])) #False
            print(np.allclose(self.h_out[0,0,0,0,0,0], h_freq_tfnp[0,0,0,0,0,0])) #False
            compare_allclose(self.h_out[0,0,0,0,0,0], h_freq_tfnp[0,0,0,0,0,0], figname="data/compare_allclose.png")


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
    
    def comparefigure(self, h_perfect, h_hat, savefile=None):
        h_est = h_hat[0,0,0,0,0,0] #(64, 1, 1, 1, 1, 14, 44)
        h_perfect  = h_perfect[0,0,0,0,0,0]
        plt.figure()
        plt.plot(np.real(h_perfect))
        plt.plot(np.imag(h_perfect))
        plt.plot(np.real(h_est), "--")
        plt.plot(np.imag(h_est), "--")
        plt.xlabel("Subcarrier index")
        plt.ylabel("Channel frequency response")
        plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
        plt.title("Comparison of channel frequency responses");
        if savefile is not None:
            plt.savefig(savefile)
    
    def compare_channelestimationdata(self):
        y_eff_tf = np.load('data/y_eff_tf.npy')
        y_eff_tf2 = np.load('data/y_eff_tf2.npy') #(128, 1, 16, 14, 64)
        print(np.allclose(y_eff_tf, y_eff_tf2)) #True

        y_eff_flat_tf = np.load('data/y_eff_flat_tf.npy')
        y_eff_flat_tf2 = np.load('data/y_eff_flat_tf2.npy') #(128, 1, 16, 896)
        print(np.allclose(y_eff_flat_tf, y_eff_flat_tf2)) #True

        pilot_ind_tf = np.load('data/pilot_ind_tf.npy') #(1, 2, 128)
        pilot_ind_tf2 = np.load('data/pilot_ind_tf2.npy')
        print(np.allclose(pilot_ind_tf, pilot_ind_tf2)) #True

        y_pilots_tf = np.load('data/y_pilots_tf.npy') #(128, 1, 16, 1, 2, 128)
        y_pilots_tf2 = np.load('data/y_pilots_tf2.npy')
        print(np.allclose(y_pilots_tf, y_pilots_tf2)) #True

        #after self.estimate_at_pilot_locations
        h_hat_beforeinter = np.load('data/h_hat_beforeinter.npy') #(128, 1, 16, 1, 2, 128)
        h_hat_beforeinter2 = np.load('data/h_hat_beforeinter2.npy')
        print(np.allclose(h_hat_beforeinter, h_hat_beforeinter2)) #False->True

        h_ls, err_var = self.estimate_at_pilot_locations(y_pilots=y_pilots_tf, no=self.no, resource_grid=self.RESOURCE_GRID)
        print(np.allclose(h_hat_beforeinter, h_ls)) #False->True

        err_var_beforeinter = np.load('data/err_var_beforeinter.npy')
        err_var_beforeinter2 = np.load('data/err_var_beforeinter2.npy')
        print(np.allclose(err_var_beforeinter, err_var_beforeinter2)) #True

        h_hat_inter = np.load('data/h_hat_inter.npy')
        h_hat_inter2 = np.load('data/h_hat_inter2.npy')
        print(np.allclose(h_hat_inter, h_hat_inter2)) #False->True

        err_var_inter = np.load('data/err_var_inter.npy')
        err_var_inter2 = np.load('data/err_var_inter2.npy')
        print(np.allclose(err_var_inter, err_var_inter2)) #True

        self.comparefigure(h_perfect=self.h_perfect, h_hat=self.h_hat, savefile='data/h_hatcompare.png')

        self.comparefigure(h_perfect=self.h_perfect, h_hat=h_hat_inter, savefile='data/h_hat_intercompare.png')

        self.comparefigure(h_perfect=self.h_perfect, h_hat=h_hat_inter2, savefile='data/h_hat_inter2compare.png')

    def estimate_at_pilot_locations(self, y_pilots, no, resource_grid):
        from channel import expand_to_rank
        import tensorflow as tf
        pilots = resource_grid.pilot_pattern.pilots
        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], tf.complex (b, 1, 16, 1, 2, 128)
        #     The observed signals for the pilot-carrying resource elements.

        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   tf.float
        #     The variance of the AWGN.

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_ls = tf.math.divide_no_nan(y_pilots, pilots) #pilots: (1, 2, 128)=>(2, 1, 16, 1, 2, 128)

        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, tf.rank(h_ls), -1) #float=>(1, 1, 1, 1, 1, 1)

        # Expand rank of pilots for broadcasting
        pilots_md = expand_to_rank(pilots, tf.rank(h_ls), 0) #(1, 2, 128)=>(1, 1, 1, 1, 2, 128)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots_md)**2) #(1, 1, 1, 1, 2, 128)

        return h_ls, err_var #h_ls: (2, 1, 16, 1, 2, 128), err_var: (1, 1, 1, 1, 2, 128)

    def checkchannelestimate(self):
        #perform channel estimation via pilots
        print("h_hat after channel estimation via pilots:", self.h_hat.shape)
        import tensorflow as tf
        y_tf = tf.convert_to_tensor(self.y, dtype=tf.complex64)
        h_hat, err_var = self.ls_est([y_tf, self.no])
        self.comparefigure(h_perfect=self.h_hat, h_hat=h_hat.numpy())
        #self.h_hat shape: (128, 1, 16, 1, 2, 14, 64), self.err_var: (1, 1, 1, 1, 2, 14, 64)
        #[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        print(np.allclose(h_hat, self.h_hat)) #False->fixed to True
        print(np.allclose(err_var, self.err_var)) #True

    def receive(self):
        # Channel output y's shape: (128, 1, 16, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        #h_out's shape: (128, 1, 16, 1, 2, 14, 76) [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_perfect, err_var_perfect = self.remove_nulled_scs(self.h_out), 0.
        print(np.allclose(h_perfect, self.h_perfect)) #True
        print("h_out after remove nulled shape:", h_perfect.shape) #(128, 1, 16, 1, 2, 14, 64)
        #among 76 subcarriers, 5 and 6 are guard carriers, effective subcarrier is 76-6-5-1(DC carrier)=64
        self.comparefigure(h_perfect, self.h_hat)

        x_hat, no_eff = self.lmmse_equ([self.y, self.h_hat, self.err_var, self.no]) 
        #x_hat, no_eff = self.lmmse_equ([y_tf, h_hat, err_var, self.no]) 
        #Estimated symbols x_hat : [batch_size, num_tx, num_streams, num_data_symbols], complex
        #Effective noise variance for each estimated symbol no_eff : [batch_size, num_tx, num_streams, num_data_symbols], float
        #64*(14-2pilots)=768
        print("x_hat after channel equalization:", self.x_hat.shape) #(128, 1, 2, 768)
        no_eff=np.mean(no_eff)
        print(self.no_eff)
        print(np.allclose(x_hat, self.x_hat)) #True

        #num_data_symbols=768, 768*2bit=1536bits
        #llr_est = self.mydemapper([self.x_hat, self.no_eff]) #(128, 1, 2, 1536)
        llr_est = self.mydemapper([x_hat, no_eff])
        #output: [batch size, num_rx, num_rx_ant, n * num_bits_per_symbol]
        print(np.allclose(llr_est, self.llr_est)) #True

        b_hat = hard_decisions(llr_est, np.int32)  #(128, 1, 2, 1536)
        BER=calculate_BER(self.b, b_hat) #0
        print("BER Value:", BER)
    
    def calculate_BER(self, binary_predictions, test_labels):
        binary_predictions = binary_predictions.squeeze() #[14, 71, 6]
        test_labels = test_labels.squeeze() #[14, 71, 6]
        BER=calculate_BER(binary_predictions, test_labels) #0
        #print("BER Value:", BER)
        return BER

    #S: num_ofdm_symbols, F: fft_size/num_subcarriers
    #RESOURCE_GRID.num_data_symbols=14(OFDM symbol)*76(subcarrier) array=1064
        #among 76 subcarriers, 5 and 6 are guard carriers, effective subcarrier is 76-6-5-1(DC carrier)=64
        #among 14 symbols, 2,11 is the pilot, effective symbol is 12
        #1064 grids contains the data, DC and pilot, effective grid=12*64=768
    #The meaning of the mask:
    # 0: The PRB is null power.
    # 1: The PRB is PDSCH (Physical Downlink Shared Channel).
    # 2: Pilot symbols
    # 3: (yellow bar) DC
    
    def TTI_mask(self, S=14, F=64, num_guard_carriers=[5,6], pilot_ofdm_symbol_indices=[2,11], dc_null=True, plotTTI=False):

        if dc_null:
            F=F+1
        # Create a mask with all ones
        TTI_mask = torch.ones((S, F), dtype=torch.int8) # all ones (14, 65)

        # Set symbol Sp for pilots
        TTI_mask[pilot_ofdm_symbol_indices[0], :] = 2 # for pilots
        TTI_mask[pilot_ofdm_symbol_indices[1], :] = 2 # for pilots

        # DC
        TTI_mask[:, F // 2] = 3 # DC to non-allocable power (oscillator phase noise): 3 in the middle

        TTI_1 = torch.zeros(S, num_guard_carriers[0], dtype=torch.int8) #empty space (14, 28) add before and after the TTI_mask
        TTI_2 = torch.zeros(S, num_guard_carriers[1], dtype=torch.int8)
        # Add FFT offsets
        TTI_mask = torch.cat((TTI_1, TTI_mask, TTI_2), dim=1) #cat in the dim=1, 65+5+6=76 subcarriers

        # Plotting the TTI mask
        if plotTTI:
            plt.figure(figsize=(8, 1.5))
            plt.imshow(TTI_mask.numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
            plt.title('TTI mask')
            plt.xlabel('Subcarrier index')
            plt.ylabel('Symbol')
            plt.savefig('output/TTImask.png')
            plt.tight_layout()
            plt.show()

        return TTI_mask #(14, 76)

    def __len__(self):
        return self.maxdatalen
    
    def __getitem__(self, item_id=0):
        ch_SINR = int(random.uniform(self.ch_SINR_min, self.ch_SINR_max)) # SINR generation for adding noise to the channel
        
        batch={}
        returnbatch=False
        if self.training:
            if returnbatch:
                labels_data= self.labels_data #(128, 12, 64, 2)
            else:
                labels_data= self.labels_data[self.index,:] #(12, 64, 2)
            batch['labels'] = labels_data
        if returnbatch:
            self.feature_2d_data = self.rx_samples_eff #(128, 12, 64)
        else:
            rx_samples_eff = self.rx_samples_eff[self.index,:] #(12, 64)
            y_real = rx_samples_eff.real #(12, 64)
            y_imag = rx_samples_eff.imag #(12, 64)
            # Stack the tensors along a new dimension (axis 0)
            z = np.stack([y_real, y_imag], axis=0) #[2, 12, 64]
            self.feature_2d_data = z
            self.feature_2d_channel = 2
        
        signal_power = np.mean(np.abs(self.feature_2d_data)**2)
        noise_power = signal_power / (10 ** (ch_SINR / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*self.feature_2d_data.shape))
        feature_2d_data_noise = self.feature_2d_data + noise.astype(np.float32)
        
        batch['feature_2d'] = feature_2d_data_noise.astype(np.float32) #self.feature_2d_data
        self.index = (self.index +1) % self.batch_size
        return batch
    
def testdataset():
    train_data = OFDMDataset(training=True, testing=False, compare=False)
    onebatch = train_data[0]
    print(onebatch['feature_2d'].shape) #(2, 12, 64)
    print(onebatch['labels'].shape) #(12, 64, 2)


if __name__ == '__main__':
    testdataset()