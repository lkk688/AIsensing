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

from deepMIMO5 import Transmitter

if __name__ == '__main__':
    scenario='O1_60'
    dataset_folder='data'

    # Bit per channel use
    NUM_BITS_PER_SYMBOL = 2 # QPSK

    # Minimum value of Eb/N0 [dB] for simulations
    EBN0_DB_MIN = -3.0

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

    transmit = Transmitter(scenario, dataset_folder, num_rx = 1, num_tx = 1, \
                batch_size =BATCH_SIZE, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = NUM_BITS_PER_SYMBOL,  \
                USE_LDPC = False, pilot_pattern = "empty", guards=False, showfig=False) #"kronecker"
        #channeltype="perfect", "awgn", "ofdm", "time"
    ber = []
    snr_db = []
    for ebno_db in ebno_dbs:
        snr_db.append(ebno_db)
        b_hat, BER = transmit(ebno_db = ebno_db, channeltype='awgn')
        ber.append(BER)
    print(snr_db)
    print(ber)
    fig, ax = plt.subplots(figsize=(16,10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    #A tuple of two floats defining x-axis limits.
    # if xlim is not None:
    #     plt.xlim(xlim)
    # if ylim is not None:
    #     plt.ylim(ylim)

    title = "BER Simulation"
    is_bler= False
    plt.title(title, fontsize=25)
    # return figure handle
    if is_bler:
        line_style = "--"
    else:
        line_style = ""
    plt.semilogy(snr_db, ber, line_style, linewidth=2)

    plt.grid(which="both")
    ebno = True
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
    ylabel="BER"
    plt.ylabel(ylabel, fontsize=25)
    legend=""
    plt.legend(legend, fontsize=20)
    save_fig = True
    if save_fig:
        plt.savefig('./data/ber.jpg')
        plt.close(fig)