#https://www.deepmimo.net/versions/v2-python/
#pip install DeepMIMO

import DeepMIMO
import numpy as np
import matplotlib.pyplot as plt
#from DeepMIMO import DeepMIMOSionnaAdapter

# The DeepMIMO dataset adapter for Sionna
#
# Input: DeepMIMO dataset, UE and BS indices to be included
#
# For a given 1D vector of BS or UE indices, the generated dataset will be stacked as different samples
#
# By default, the adapter will only select the first BS of the DeepMIMO dataset and all UEs
# The adapter assumes BSs are transmitters and users are receivers. 
# Uplink channels could be generated using (transpose) reciprocity.
#
# For multi-user channels, provide a 2D numpy matrix of size (num_samples x num_rx)
#
# Examples:
# ue_idx = np.array([[0, 1 ,2], [1, 2, 3]]) generates (num_bs x 3 UEs) channels
# with 2 data samples from the BSs to the UEs [0, 1, 2] and [1, 2, 3], respectively.
#
# For single-basestation channels with the data from different basestations stacked,
# provide a 1D array of basestation indices
#
# For multi-BS channels, provide a 2D numpy matrix of (num_samples x num_tx)
#
# Examples:
# bs_idx = np.array([[0, 1], [2, 3], [4, 5]]) generates (2 BSs x num_rx) channels
# by stacking the data of channels from the basestations (0 and 1), (2 and 3), 
# and (4 and 5) to the UEs.
#
class DeepMIMOSionnaAdapter:
    def __init__(self, DeepMIMO_dataset, bs_idx = None, ue_idx = None):
        self.dataset = DeepMIMO_dataset
        
        # Set bs_idx based on given parameters
        # If no input is given, choose the first basestation
        if bs_idx is None:
            bs_idx = np.array([[0]])
        self.bs_idx = self._verify_idx(bs_idx)
        
        # Set ue_idx based on given parameters
        # If no input is given, set all user indices
        if ue_idx is None:
            ue_idx = np.arange(DeepMIMO_dataset[0]['user']['channel'].shape[0])
        self.ue_idx = self._verify_idx(ue_idx) #(9231, 1)
        
        # Extract number of antennas from the DeepMIMO dataset
        self.num_rx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[1] #1 
        self.num_tx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[2] #16
        
        # Determine the number of samples based on the given indices
        self.num_samples_bs = self.bs_idx.shape[0] #1
        self.num_samples_ue = self.ue_idx.shape[0] #9231
        self.num_samples = self.num_samples_bs * self.num_samples_ue #9231
        
        # Determine the number of tx and rx elements in each channel sample based on the given indices
        self.num_rx = self.ue_idx.shape[1] #1
        self.num_tx = self.bs_idx.shape[1] #1
        
        # Determine the number of available paths in the DeepMIMO dataset
        self.num_paths = DeepMIMO_dataset[0]['user']['channel'].shape[-1] #10
        self.num_time_steps = 1 # Time step = 1 for static scenarios
        
        # The required path power shape for Sionna
        self.ch_shape = (self.num_rx, 
                         self.num_rx_ant, 
                         self.num_tx, 
                         self.num_tx_ant, 
                         self.num_paths, 
                         self.num_time_steps) #(rx=1, rx_ant=1, tx=1, tx_ant=16, paths=10, timestesp=1)
        
        # The required path delay shape for Sionna
        self.t_shape = (self.num_rx, self.num_tx, self.num_paths) #(rx=1,tx=1,paths=10)
    
    # Verify the index values given as input
    def _verify_idx(self, idx):
        idx = self._idx_to_numpy(idx)
        idx = self._numpy_size_check(idx)
        return idx
    
    # Convert the possible input types to numpy (integer - range - list)
    def _idx_to_numpy(self, idx):
        if isinstance(idx, int): # If the input is an integer - a single ID - convert it to 2D numpy array
            idx = np.array([[idx]])
        elif isinstance(idx, list) or isinstance(idx, range): # If the input is a list or range - convert it to a numpy array
            idx = np.array(idx)
        elif isinstance(idx, np.ndarray):
            pass
        else:
            raise TypeError('The index input type must be an integer, list, or numpy array!') 
        return idx
    
    # Check the size of the given input and convert it to a 2D matrix of proper shape (num_tx x num_samples) or (num_rx x num_samples)
    def _numpy_size_check(self, idx):
        if len(idx.shape) == 1:
            idx = idx.reshape((-1, 1))
        elif len(idx.shape) == 2:
            pass
        else:
            raise ValueError('The index input must be integer, vector or 2D matrix!')
        return idx
    
    # Override length of the generator to provide the available number of samples
    def __len__(self):
        return self.num_samples
        
    # Provide samples each time the generator is called
    def __call__(self):
        for i in range(self.num_samples_ue): # For each UE sample
            for j in range(self.num_samples_bs): # For each BS sample
                # Generate zero vectors for the Sionna sample
                a = np.zeros(self.ch_shape, dtype=np.csingle)
                tau = np.zeros(self.t_shape, dtype=np.single)
                
                # Place the DeepMIMO dataset power and delays into the channel sample for Sionna
                for i_ch in range(self.num_rx): # for each receiver in the sample
                    for j_ch in range(self.num_tx): # for each transmitter in the sample
                        i_ue = self.ue_idx[i][i_ch] # UE channel sample i - channel RX i_ch
                        i_bs = self.bs_idx[j][j_ch] # BS channel sample i - channel TX j_ch
                        a[i_ch, :, j_ch, :, :, 0] = self.dataset[i_bs]['user']['channel'][i_ue]
                        tau[i_ch, j_ch, :self.dataset[i_bs]['user']['paths'][i_ue]['num_paths']] = self.dataset[i_bs]['user']['paths'][i_ue]['ToA'] 
                #(9231, 1, 16, 10)
                yield (a, tau) # yield this sample h=(num_rx=1, 1, num_tx=1, 16, 10, 1), tau=(num_rx=1,num_tx=1,ToA=10)



if __name__ == '__main__':

    # Load the default parameters
    parameters = DeepMIMO.default_params()

    # Set scenario name
    parameters['scenario'] = 'O1_60' #https://deepmimo.net/scenarios/o1-scenario/

    # Set the main folder containing extracted scenarios
    parameters['dataset_folder'] = r'D:\Dataset\CommunicationDataset\O1_60'

    # To only include 10 strongest paths in the channel computation, set
    parameters['num_paths'] = 10

    # To activate only the first basestation, set
    parameters['active_BS'] = np.array([1])
    # To activate the basestations 6, set
    #parameters['active_BS'] = np.array([6])

    parameters['OFDM']['bandwidth'] = 0.05 # 50 MHz
    print(parameters['OFDM']['subcarriers']) #512
    #parameters['OFDM']['subcarriers'] = 512 # OFDM with 512 subcarriers
    #parameters['OFDM']['subcarriers_limit'] = 64 # Keep only first 64 subcarriers

    # To activate the user rows 1-5, set
    parameters['user_row_first'] = 1 #400 # First user row to be included in the dataset
    parameters['user_row_last'] = 100 #450 # Last user row to be included in the dataset

    # Consider 3 active basestations
    #parameters['active_BS'] = np.array([1, 5, 8])
    # Configuration of the antenna arrays
    parameters['bs_antenna']['shape'] = np.array([16, 1, 1]) # BS antenna shape through [x, y, z] axes
    parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # UE antenna shape through [x, y, z] axes, single antenna

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
    j=0 #user j
    print(DeepMIMO_dataset[active_bs_idx]['user']['location'][j]) #The Euclidian location of the user in the form of [x, y, z].

    # Number of basestations
    print(len(DeepMIMO_dataset)) #1
    # Keys of a basestation dictionary
    print(DeepMIMO_dataset[0].keys()) #['user', 'basestation', 'location']
    # Keys of a channel
    print(DeepMIMO_dataset[0]['user'].keys()) #['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
    # Number of UEs
    print(len(DeepMIMO_dataset[0]['user']['channel'])) #9231
    print(DeepMIMO_dataset[active_bs_idx]['user']['channel'].shape) #(num_ue_locations=9231, 1, bs_antenna=16, strongest_path=10) 
    # Shape of the channel matrix
    print(DeepMIMO_dataset[0]['user']['channel'].shape) #(9231, 1, 16, 10)

    i=0
    j=0
    #The channel matrix between basestation i and user j
    DeepMIMO_dataset[i]['user']['channel'][j]
    #Float matrix of size (number of RX antennas) x (number of TX antennas) x (number of OFDM subcarriers)

    # Shape of BS 0 - UE 0 channel
    print(DeepMIMO_dataset[i]['user']['channel'][0].shape) #(1, 16, 10)
    
    # Path properties of BS 0 - UE 0
    print(DeepMIMO_dataset[i]['user']['paths'][j]) #Ray-tracing Path Parameters in dictionary
    #Azimuth and zenith angle-of-arrivals – degrees (DoA_phi, DoA_theta)
    # Azimuth and zenith angle-of-departure – degrees (DoD_phi, DoD_theta)
    # Time of arrival – seconds (ToA)
    # Phase – degrees (phase)
    # Power – watts (power)
    # Number of paths (num_paths)

    print(DeepMIMO_dataset[i]['user']['LoS'][j]) #Integer of values {-1, 0, 1} indicates the existence of the LOS path in the channel.
    # (1): The LoS path exists.
    # (0): Only NLoS paths exist. The LoS path is blocked (LoS blockage).
    # (-1): No paths exist between the transmitter and the receiver (Full blockage).

    print(DeepMIMO_dataset[i]['user']['distance'][j])
    #The Euclidian distance between the RX and TX locations in meters.

    print(DeepMIMO_dataset[i]['user']['pathloss'][j])
    #The combined path-loss of the channel between the RX and TX in dB.


    print(DeepMIMO_dataset[i]['location'])
    #Basestation Location [x, y, z].
    print(DeepMIMO_dataset[i]['user']['location'][j])
    #The Euclidian location of the user in the form of [x, y, z].


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

    dataset = DeepMIMO_dataset
    ## Visualization of a channel matrix
    plt.figure()
    # Visualize channel magnitude response
    # First, select indices of a user and bs
    ue_idx = 0
    bs_idx = 0
    # Import channel
    channel = dataset[bs_idx]['user']['channel'][ue_idx]
    # Take only the first antenna pair
    plt.imshow(np.abs(np.squeeze(channel).T))
    plt.title('Channel Magnitude Response')
    plt.xlabel('TX Antennas')
    plt.ylabel('Subcarriers')

    ## Visualization of the UE positions and path-losses
    loc_x = dataset[bs_idx]['user']['location'][:, 0] #(9231,)
    loc_y = dataset[bs_idx]['user']['location'][:, 1]
    loc_z = dataset[bs_idx]['user']['location'][:, 2]
    pathloss = dataset[bs_idx]['user']['pathloss'] #(9231,
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(loc_x, loc_y, loc_z, c=pathloss)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    bs_loc_x = dataset[bs_idx]['basestation']['location'][:, 0]
    bs_loc_y = dataset[bs_idx]['basestation']['location'][:, 1]
    bs_loc_z = dataset[bs_idx]['basestation']['location'][:, 2]
    ax.scatter(bs_loc_x, bs_loc_y, bs_loc_z, c='r')
    ttl = plt.title('UE and BS Positions')

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.scatter(loc_x, loc_y, c=pathloss)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.colorbar(im, ax=ax)
    ttl = plt.title('UE Grid Path-loss (dB)')

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

    DeepMIMO_Sionna_adapter = DeepMIMOSionnaAdapter(DeepMIMO_dataset, ue_idx=ue_idx) #ch shape: (1, 1, 1, 16, 10, 1)

    # CIRDataset to parse the dataset
    batch_size =64
    from sionna_tf import CIRDataset, StreamManagement, ResourceGrid, BinarySource, Mapper, ResourceGridMapper, \
        ZFPrecoder, ebnodb2no, LMMSEEqualizer, Demapper, RemoveNulledSubcarriers
    from channel import GenerateOFDMChannel, ApplyOFDMChannel, LSChannelEstimator
    from ldpc.encoding import LDPC5GEncoder
    from ldpc.decoding import LDPC5GDecoder

    #DeepMIMO_Sionna_adapter: Generator that returns channel impulse responses ``(a, tau)`` where
            # ``a`` is the tensor of channel coefficients of shape
            # `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
            # and dtype ``dtype``, and ``tau`` the tensor of path delays
            # of shape  `[num_rx, num_tx, num_paths]` 
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
    h_freq = ofdm_channel() #h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10] => (64, 1, 1, 1, 16, 1, 76) 
    #h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex

    # Precoding: (num_rx, num_rx_ant merged to tx?)
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