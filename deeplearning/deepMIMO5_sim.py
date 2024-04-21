#ref:https://github.com/NVlabs/sionna/blob/main/examples/Sionna_tutorial_part4.ipynb

import os
# gpu_num = 0 # Use "" to use the CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
import numpy as np

# For saving complex Python data structures efficiently
import pickle

# For plotting
#%matplotlib inline
import matplotlib.pyplot as plt

# For the implementation of the neural receiver
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
# from tensorflow.nn import relu

from deepMIMO5 import Transmitter, BinarySource, count_errors, count_block_errors

def ber_plot(ebno_dbs, bers, legend="", ylabel="BER", title="Bit Error Rate", ebno=True, xlim=None,
             ylim=None, is_bler=False, savefigpath='./data/ber.jpg'):
    if not isinstance(legend, list):
        assert isinstance(legend, str)
        legend = [legend]
    
    assert isinstance(title, str), "title must be str."

    fig, ax = plt.subplots(figsize=(16,10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.title(title, fontsize=25)
    # return figure handle
    if isinstance(bers, list):
        for idx, b in enumerate(bers):
            if is_bler:
                line_style = "--"
            else:
                line_style = ""
            plt.semilogy(ebno_dbs, b, line_style, linewidth=2)
    else:
        if is_bler:
            line_style = "--"
        else:
            line_style = ""
        plt.semilogy(ebno_dbs, bers, line_style, linewidth=2)

    plt.grid(which="both")
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(legend, fontsize=20)
    if savefigpath is not None:
        plt.savefig(savefigpath)
        plt.close(fig)
    else:
        #plt.close(fig)
        pass
    return fig, ax

    

def ber_plot_single(ebno_dbs, bers, title = "BER Simulation", savefigpath='./data/ber.jpg'):

    fig, ax = plt.subplots(figsize=(16,10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    #A tuple of two floats defining x-axis limits.
    # if xlim is not None:
    #     plt.xlim(xlim)
    # if ylim is not None:
    #     plt.ylim(ylim)

    is_bler= False
    plt.title(title, fontsize=25)
    # return figure handle
    if is_bler:
        line_style = "--"
    else:
        line_style = ""
    plt.semilogy(ebno_dbs, bers, line_style, linewidth=2)

    plt.grid(which="both")
    ebno = True
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
    ylabel="BER"
    plt.ylabel(ylabel, fontsize=25)
    # legend=""
    # plt.legend(legend, fontsize=20)
    if savefigpath is not None:
        plt.savefig(savefigpath)
        plt.close(fig)

def sim_ber(ebno_dbs, eval_transceiver, b, batch_size,  channeltype='awgn'):
    #num_points = 100  # Example value, replace with the actual value
    ebno_dbs_np = np.array(ebno_dbs, dtype=np.float64)  # Cast to the desired data type
    batch_size_np = np.array(batch_size, dtype=np.int32)  # Cast to the desired data type
    num_points = ebno_dbs_np.shape[0] #20

    bit_n = np.size(b) #272384
    block_n = np.size(b[..., -1]) #128, b: (128, 1, 1, 2128)

    bers = []
    blers = []
    BERs= []

    for ebno_db in ebno_dbs:
        b_hat, BER = eval_transceiver(b=b, ebno_db = ebno_db, channeltype=channeltype)
        BERs.append(BER)
        # count errors
        bit_e = count_errors(b, b_hat)
        block_e = count_block_errors(b, b_hat)

        # Initialize NumPy arrays bit_errors, block_errors, nb_bits, and nb_blocks (if not already initialized)
        nb_bits = np.zeros_like(bit_n, dtype=np.int64)
        nb_blocks = np.zeros_like(block_n, dtype=np.int64)
        #bit_errors = 0
        #block_errors = 0
        bit_errors = np.zeros_like(bit_e, dtype=np.int64)
        block_errors = np.zeros_like(block_e, dtype=np.int64)

        # Update variables
        bit_errors = bit_errors + np.int64(bit_e)
        block_errors = block_errors + np.int64(block_e)
        nb_bits = nb_bits+ np.int64(bit_n)
        nb_blocks = nb_blocks+ np.int64(block_n)

        ber = np.divide(bit_errors.astype(np.float64), nb_bits.astype(np.float64))
        bler = np.divide(block_errors.astype(np.float64), nb_blocks.astype(np.float64))

        # Replace NaN values with zeros
        ber = np.where(np.isnan(ber), np.zeros_like(ber), ber)
        bler = np.where(np.isnan(bler), np.zeros_like(bler), bler)

        bers.append(ber)
        blers.append(bler)

    return bers, blers, BERs

def simulationloop(ebno_dbs, eval_transceiver, b=None, channeltype='awgn'):
    bers = []
    for ebno_db in ebno_dbs:
        b_hat, BER = eval_transceiver(b=b, ebno_db = ebno_db, channeltype=channeltype)
        bers.append(BER)
    bers_np=np.array(bers)
    return bers_np

if __name__ == '__main__':
    scenario='O1_60'
    dataset_folder='data'

    # Bit per channel use
    NUM_BITS_PER_SYMBOL = 2 # QPSK

    # Minimum value of Eb/N0 [dB] for simulations
    EBN0_DB_MIN = 2.0 #-3.0

    # Maximum value of Eb/N0 [dB] for simulations
    EBN0_DB_MAX = 5.0

    # How many examples are processed by Sionna in parallel
    BATCH_SIZE = 128

    # Coding rate
    CODERATE = 0.5

    # Define the number of UT and BS antennas
    NUM_UT = 1
    NUM_BS = 1
    NUM_UT_ANT = 1
    NUM_BS_ANT = 2

    # The number of transmitted streams is equal to the number of UT antennas
    # in both uplink and downlink
    NUM_STREAMS_PER_TX = NUM_UT_ANT

    ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)

    eval_transceiver = Transmitter(scenario, dataset_folder, num_rx = 1, num_tx = 1, \
                batch_size =BATCH_SIZE, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = NUM_BITS_PER_SYMBOL,  \
                USE_LDPC = False, pilot_pattern = "empty", guards=False, showfig=False) #"kronecker"
        #channeltype="perfect", "awgn", "ofdm", "time"
    #Number of information bits per codeword
    k=eval_transceiver.k
    binary_source = BinarySource()
    # Start Transmitter self.k Number of information bits per codeword
    b = binary_source([BATCH_SIZE, 1, NUM_STREAMS_PER_TX, k]) #[batch_size, num_tx, num_streams_per_tx, num_databits]
    
    bers, blers, BERs = sim_ber(ebno_dbs, eval_transceiver, b, BATCH_SIZE,  channeltype='awgn')

    BER_list = []
    bers=simulationloop(ebno_dbs, eval_transceiver, b, channeltype='awgn')
    print(bers)
    BER_list.append(bers)

    eval_transceiver = Transmitter(scenario, dataset_folder, num_rx = 1, num_tx = 1, \
                batch_size =BATCH_SIZE, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = NUM_BITS_PER_SYMBOL,  \
                USE_LDPC = True, pilot_pattern = "empty", guards=False, showfig=False) #"kronecker"
    bers=simulationloop(ebno_dbs, eval_transceiver, channeltype='awgn')
    print(bers)
    BER_list.append(bers)

    eval_transceiver = Transmitter(scenario, dataset_folder, num_rx = 1, num_tx = 1, \
            batch_size =BATCH_SIZE, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = NUM_BITS_PER_SYMBOL,  \
            USE_LDPC = False, pilot_pattern = "kronecker", guards=True, showfig=False) #"kronecker"
    #channeltype="perfect", "awgn", "ofdm", "time"
    bers=simulationloop(ebno_dbs, eval_transceiver, b, channeltype='ofdm')
    print(bers)
    BER_list.append(bers)

    ber_plot_single(ebno_dbs, bers, title = "BER Simulation", savefigpath='./data/ber.jpg')

    ber_plot(ebno_dbs, BER_list, legend=['awgn','awgn+ldpc', 'OFDM channel'], ylabel="BER", title="Bit Error Rate", ebno=True, xlim=None,
             ylim=None, is_bler=None, savefigpath='./data/berlist.jpg')