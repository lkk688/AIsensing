#https://www.deepmimo.net/versions/v2-python/
#pip install DeepMIMO

import DeepMIMO
import numpy as np
import matplotlib.pyplot as plt
from DeepMIMO import DeepMIMOSionnaAdapter

# Load the default parameters
parameters = DeepMIMO.default_params()

# Set scenario name
parameters['scenario'] = 'O1_60' #https://deepmimo.net/scenarios/o1-scenario/

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'D:\Dataset\CommunicationDataset\O1_60'

# To only include 10 strongest paths in the channel computation, set
parameters['num_paths'] = 10

# To activate only the first basestation, set
#parameters['active_BS'] = np.array([1])
# To activate the basestations 6, set
parameters['active_BS'] = np.array([6])

# To activate the user rows 1-5, set
parameters['user_row_first'] = 400 # First user row to be included in the dataset
parameters['user_row_last'] = 450 # Last user row to be included in the dataset

# Consider 3 active basestations
#parameters['active_BS'] = np.array([1, 5, 8])
# Configuration of the antenna arrays
parameters['bs_antenna']['shape'] = np.array([16, 1, 1]) # BS antenna shape through [x, y, z] axes
parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # UE antenna shape through [x, y, z] axes

# The OFDM_channels parameter allows choosing between the generation of channel impulse
# responses (if set to 0) or frequency domain channels (if set to 1).
# It is set to 0 for this simulation, as the channel responses in frequency domain
# will be generated using Sionna.
parameters['OFDM_channels'] = 0

# Generate data
DeepMIMO_dataset = DeepMIMO.generate_data(parameters)

plt.figure(figsize=(12,8))

## User locations
active_bs_idx = 0 # Select the first active basestation in the dataset
print(DeepMIMO_dataset[active_bs_idx]['user'].keys()) #['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
print(DeepMIMO_dataset[active_bs_idx]['user']['location'].shape) #(9231, 3)  num_ue_locations: 9231
print(DeepMIMO_dataset[active_bs_idx]['user']['channel'].shape) #(9231, 1, 16, 10)
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 1], # y-axis location of the users
         DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 0], # x-axis location of the users
         s=1, marker='x', c='C0', label='The users located on the rows %i to %i (R%i to R%i)'%
           (parameters['user_row_first'], parameters['user_row_last'],
           parameters['user_row_first'], parameters['user_row_last']))
# First 181 users correspond to the first row
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 1],
         DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 0],
         s=1, marker='x', c='C1', label='First row of users (R%i)'% (parameters['user_row_first']))

## Basestation location
plt.scatter(DeepMIMO_dataset[active_bs_idx]['location'][1],
         DeepMIMO_dataset[active_bs_idx]['location'][0],
         s=50.0, marker='o', c='C2', label='Basestation')

plt.gca().invert_xaxis() # Invert the x-axis to align the figure with the figure above
plt.ylabel('x-axis')
plt.xlabel('y-axis')
plt.grid()
plt.legend();

# Number of receivers for the Sionna model.
# MISO is considered here.
num_rx = 1

# The number of UE locations in the generated DeepMIMO dataset
num_ue_locations = len(DeepMIMO_dataset[0]['user']['channel']) # 9231
# Pick the largest possible number of user locations that is a multiple of ``num_rx``
ue_idx = np.arange(num_rx*(num_ue_locations//num_rx)) #(9231,) 0~9230
# Optionally shuffle the dataset to not select only users that are near each others
np.random.shuffle(ue_idx)
# Reshape to fit the requested number of users
ue_idx = np.reshape(ue_idx, [-1, num_rx]) # In the shape of (floor(9231/num_rx) x num_rx) (9231,1)

DeepMIMO_Sionna_adapter = DeepMIMOSionnaAdapter(DeepMIMO_dataset, ue_idx=ue_idx)

# CIRDataset to parse the dataset
batch_size =64
from sionna_tf import CIRDataset, StreamManagement, ResourceGrid, BinarySource, Mapper, ResourceGridMapper, \
    ZFPrecoder, ebnodb2no, LMMSEEqualizer, Demapper, RemoveNulledSubcarriers
from channel import GenerateOFDMChannel, ApplyOFDMChannel, LSChannelEstimator
from ldpc.encoding import LDPC5GEncoder
from ldpc.decoding import LDPC5GDecoder
CIR = CIRDataset(DeepMIMO_Sionna_adapter,
                                batch_size,
                                DeepMIMO_Sionna_adapter.num_rx, #1
                                DeepMIMO_Sionna_adapter.num_rx_ant, #1
                                DeepMIMO_Sionna_adapter.num_tx, #1
                                DeepMIMO_Sionna_adapter.num_tx_ant, #16
                                DeepMIMO_Sionna_adapter.num_paths, #10
                                DeepMIMO_Sionna_adapter.num_time_steps) #1

num_streams_per_tx = DeepMIMO_Sionna_adapter.num_rx ##1
STREAM_MANAGEMENT = StreamManagement(np.ones([DeepMIMO_Sionna_adapter.num_rx, 1], int), num_streams_per_tx) #RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX

cyclic_prefix_length = 0 #6
num_guard_carriers = [0, 0]
dc_null=False
pilot_ofdm_symbol_indices=[2,11]
pilot_pattern = "kronecker"
RESOURCE_GRID = ResourceGrid( num_ofdm_symbols=14,
                                    fft_size=76,
                                    subcarrier_spacing=60e3, #30e3,
                                    num_tx=DeepMIMO_Sionna_adapter.num_tx, #1
                                    num_streams_per_tx=num_streams_per_tx, #1
                                    cyclic_prefix_length=cyclic_prefix_length,
                                    num_guard_carriers=num_guard_carriers,
                                    dc_null=dc_null,
                                    pilot_pattern=pilot_pattern,
                                    pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
RESOURCE_GRID.show() #14(OFDM symbol)*76(subcarrier) array=1064

num_bits_per_symbol = 4
coderate = 0.5
# Codeword length
n = int(RESOURCE_GRID.num_data_symbols * num_bits_per_symbol) #912*4=3648
# Number of information bits per codeword
k = int(n * coderate) #1824

# OFDM channel, 
#generate Channel frequency responses
ofdm_channel = GenerateOFDMChannel(CIR, RESOURCE_GRID, normalize_channel=True)
        
#Apply single-tap channel frequency responses to channel inputs.
channel_freq = ApplyOFDMChannel(add_awgn=True)

# Transmitter
binary_source = BinarySource()
encoder = LDPC5GEncoder(k, n) #1824, 3648
mapper = Mapper("qam", num_bits_per_symbol)
rg_mapper = ResourceGridMapper(RESOURCE_GRID)
#Zero-forcing precoding for multi-antenna transmissions.
zf_precoder = ZFPrecoder(RESOURCE_GRID, STREAM_MANAGEMENT, return_effective_channel=True)

# Receiver
ls_est = LSChannelEstimator(RESOURCE_GRID, interpolation_type="lin_time_avg")
lmmse_equ = LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)
demapper = Demapper("app", "qam", num_bits_per_symbol)
decoder = LDPC5GDecoder(encoder, hard_out=True)
remove_nulled_scs = RemoveNulledSubcarriers(RESOURCE_GRID)

# Start Transmitter
b = binary_source([batch_size, 1, num_streams_per_tx, k]) #[64,1,1,1824]
c = encoder(b) #[64,1,1,3648]
x = mapper(c) #[64,1,1,912]
x_rg = rg_mapper(x) ##[64,1,1,14,76] 14*76=1064
# Generate the OFDM channel
h_freq = ofdm_channel() #(64, 1, 1, 1, 16, 1, 76)
#h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex

# Precoding
#Input: Tensor containing the resource grid to be precoded. x : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
# h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, fft_size], tf.complex
#Tensor containing the channel knowledge based on which the precoding is computed.
x_rg, g = zf_precoder([x_rg, h_freq]) #[64, 1, 16, 14, 76] , [64, 1, 1, 1, 1, 1, 76]
#Output: The precoded resource grids. x_precoded : [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
#h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers]

# Apply OFDM channel
ebnorange=np.linspace(-7, -5.25, 10)
ebno_db = 5.0
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, RESOURCE_GRID)

#input: (x, h_freq, no) or (x, h_freq):
# Channel inputs x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
# h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex Channel frequency responses
y = channel_freq([x_rg, h_freq, no])
#Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex    
print(y.shape) #[64, 1, 1, 14, 76]

# Start Receiver
#Observed resource grid y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
#no : [batch_size, num_rx, num_rx_ant] 
h_hat, err_var = ls_est([y, no]) #(64, 1, 1, 1, 1, 14, 76), (1, 1, 1, 1, 1, 14, 76)
#h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
#Channel estimates accross the entire resource grid for all transmitters and streams

#input (y, h_hat, err_var, no)
#Received OFDM resource grid after cyclic prefix removal and FFT y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
#Channel estimates for all streams from all transmitters h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no]) #(64, 1, 1, 912), (64, 1, 1, 912)
#Estimated symbols x_hat : [batch_size, num_tx, num_streams, num_data_symbols], tf.complex
#Effective noise variance for each estimated symbol no_eff : [batch_size, num_tx, num_streams, num_data_symbols], tf.float


llr = demapper([x_hat, no_eff])  #[64, 1, 1, 3648] 912*4
#output: [...,n*num_bits_per_symbol]

b_hat = decoder(llr) #[64, 1, 1, 1824]
print(b_hat)