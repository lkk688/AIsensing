import DeepMIMO
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from deepMIMO5 import count_errors, count_block_errors, BinarySource
#New add
from AIsim_main2 import Transmitter, ber_plot_single2
import os
IMG_FORMAT=".pdf" #".png"

def sim_ber(ebno_dbs, eval_transceiver, b, batch_size):
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
        #b_hat, BER = eval_transceiver(b=b, ebno_db = ebno_db, channeltype=channeltype)
        b_hat, BER = eval_transceiver(b=b, ebno_db = ebno_db)
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

def sim_bersingle2(channeldataset='deepmimo', channeltype='ofdm', NUM_BITS_PER_SYMBOL = 2, EBN0_DB_MIN = -5.0, EBN0_DB_MAX = 25.0, \
                   BATCH_SIZE = 128, NUM_UT = 1, NUM_BS = 1, NUM_UT_ANT = 1, NUM_BS_ANT = 16, showfigure = False, datapathbase='data/'):
        # Bit per channel use
    # NUM_BITS_PER_SYMBOL = 2 # QPSK

    # # Minimum value of Eb/N0 [dB] for simulations
    # EBN0_DB_MIN = -5.0 #-3.0

    # # Maximum value of Eb/N0 [dB] for simulations
    # EBN0_DB_MAX = 25.0 #5.0

    # # How many examples are processed by Sionna in parallel
    # BATCH_SIZE = 128 #64

    # # Define the number of UT and BS antennas
    # NUM_UT = 1
    # NUM_BS = 1
    # NUM_UT_ANT = 1 #2 is not working
    # NUM_BS_ANT = 16

    if not os.path.exists(datapathbase):
        os.makedirs(datapathbase)

    ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)

    datapath = datapathbase+channeldataset+'_'+channeltype
    eval_transceiver = Transmitter(channeldataset=channeldataset, channeltype=channeltype, scenario=scenario, dataset_folder=dataset_folder, direction='uplink', \
                    num_ut = NUM_UT, num_ut_ant=NUM_UT_ANT, num_bs = NUM_BS, num_bs_ant=NUM_BS_ANT, \
                    batch_size =BATCH_SIZE, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = NUM_BITS_PER_SYMBOL,  \
                    subcarrier_spacing=60e3, \
                    USE_LDPC = False, pilot_pattern = "kronecker", guards=True, showfig=showfigure, savedata=True, outputpath=datapathbase)
    
    b_hat, BER = eval_transceiver(ebno_db = 5.0, perfect_csi=False, datapath=datapath+"_ebno5.npy")

    #channeltype="perfect", "awgn", "ofdm", "time"
    #Number of information bits per codeword
    k=eval_transceiver.k
    binary_source = BinarySource()
    NUM_STREAMS_PER_TX=eval_transceiver.num_streams_per_tx
    # Start Transmitter self.k Number of information bits per codeword
    b = binary_source([BATCH_SIZE, 1, NUM_STREAMS_PER_TX, k]) #[batch_size, num_tx, num_streams_per_tx, num_databits]

    b_hat, BER = eval_transceiver(ebno_db = 25.0, perfect_csi=False, datapath=datapath+"_ebno25.npy")
    
    bers, blers, BERs = sim_ber(ebno_dbs, eval_transceiver, b, BATCH_SIZE)
    #ber_plot_single(ebno_dbs, bers, title = "BER Simulation", savefigpath='./data/bernew.jpg')
    ber_plot_single2(ebno_dbs, bers, is_bler= False, title = "BER Simulation", savefigpath=datapath+'_ber.pdf')
    ber_plot_single2(ebno_dbs, blers, is_bler=True, title = "BLER Simulation", savefigpath=datapath+'_blers.pdf')
    return bers, blers, BERs

if __name__ == '__main__':

    
    scenario='O1_60'
    dataset_folder='data/DeepMIMO'
    cdltest = False
    bertest = True
    showfigure = False
    
    bers, blers, BERs = sim_bersingle2(channeldataset='cdl', channeltype='ofdm', NUM_BITS_PER_SYMBOL = 2, EBN0_DB_MIN = -5.0, EBN0_DB_MAX = 25.0, \
                   BATCH_SIZE = 128, NUM_UT = 1, NUM_BS = 1, NUM_UT_ANT = 2, NUM_BS_ANT = 16, showfigure = showfigure, datapathbase='data/cdldatagen/')
    # bers, blers, BERs = sim_bersingle2(channeldataset='deepmimo', channeltype='ofdm', NUM_BITS_PER_SYMBOL = 2, EBN0_DB_MIN = -5.0, EBN0_DB_MAX = 25.0, \
    #                BATCH_SIZE = 128, NUM_UT = 1, NUM_BS = 1, NUM_UT_ANT = 1, NUM_BS_ANT = 16, showfigure = showfigure, datapathbase='data/')