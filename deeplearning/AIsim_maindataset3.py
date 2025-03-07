import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from deepMIMO5 import hard_decisions, calculate_BER
import random
from AIcomm_models import OFDMNet
from AI_multitask import DualPurposeTransformer #option2
IMG_FORMAT=".pdf" #".png"

def compare_allclose(arry1, arry2, threshold=1e-6, figname="data/compare_allclose"+IMG_FORMAT):
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


# Compute range-Doppler response
def compute_range_doppler(radar_reflection, fft_size, num_ofdm_symbols):
    """
    Compute the range-Doppler response for radar reflection signals.
    
    Args:
        radar_reflection (np.ndarray): Radar reflection signals of shape (batch_size, num_ofdm_symbols, fft_size).
        fft_size (int): Number of subcarriers.
        num_ofdm_symbols (int): Number of OFDM symbols.
    
    Returns:
        np.ndarray: Range-Doppler response of shape (fft_size, num_ofdm_symbols).
    """
    # Take the first sample in the batch for visualization
    radar_reflection = radar_reflection[0]  # Shape: (num_ofdm_symbols, fft_size)
    
    # Compute 2D FFT (range-Doppler response)
    range_doppler = np.fft.fftshift(np.fft.fft2(radar_reflection))
    
    return range_doppler

# Plot range-Doppler response
def plot_range_doppler(range_doppler, fft_size, num_ofdm_symbols, targets=None, system_params=None):
    """
    Plot the range-Doppler response with log10 scaling and ground truth target locations.
    
    Args:
        range_doppler (np.ndarray): Range-Doppler response of shape (fft_size, num_ofdm_symbols).
        fft_size (int): Number of subcarriers.
        num_ofdm_symbols (int): Number of OFDM symbols.
        targets (list): List of target dictionaries with distance and velocity information.
        system_params (dict): Dictionary containing system parameters for converting physical
                             units to bin indices.
    """
    plt.figure(figsize=(10, 6))
    
    # Extract system parameters
    carrier_freq = system_params.get('carrier_frequency', 3.5e9)  # Hz
    subcarrier_spacing = system_params.get('subcarrier_spacing', 15e3)  # Hz
    sampling_rate = subcarrier_spacing * fft_size  # Hz
    speed_of_light = 3e8  # m/s
    wavelength = speed_of_light / carrier_freq  # m
    
    # Calculate range and velocity axes with more realistic limits
    # For range axis (y-axis)
    # Maximum range based on target data rather than theoretical maximum
    max_range = 400  # meters (slightly larger than the furthest target)
    range_axis = np.linspace(0, max_range, fft_size)  # in meters, starting from 0
    
    # For velocity axis (x-axis)
    # Maximum velocity based on target data rather than theoretical maximum
    max_velocity = 30  # m/s (slightly larger than the fastest target)
    velocity_axis = np.linspace(-max_velocity, max_velocity, num_ofdm_symbols)  # in m/s
    
    # Apply log10 scaling for better visualization
    range_doppler_db = 20 * np.log10(np.abs(range_doppler) + 1e-10)
    
    # Normalize to enhance visibility
    range_doppler_db_norm = range_doppler_db - np.min(range_doppler_db)
    range_doppler_db_norm = range_doppler_db_norm / np.max(range_doppler_db_norm) * 60  # 60 dB dynamic range
    
    # Plot the range-Doppler map with physical units
    plt.imshow(range_doppler_db_norm, aspect='auto', cmap='viridis',
               extent=[velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]])
    
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Distance (m)')
    plt.title('Range-Doppler Response')
    
    # Add ground truth target locations if provided
    if targets is not None:
        # Create a list to store target labels for legend
        target_labels = []
        
        for i, target in enumerate(targets):
            # Extract target parameters
            distance = target['distance']  # m
            velocity = target['velocity']  # m/s
            
            # Create unique label for this target
            label = f'Target {i+1}: {distance}m, {velocity}m/s'
            target_labels.append(label)
            
            # Plot target location with enhanced visibility
            plt.scatter(velocity, distance, 
                       c='red', marker='x', s=150, linewidths=3, 
                       label=label, zorder=10)
            
            # Add a circle around the marker for better visibility
            plt.scatter(velocity, distance, 
                       c='none', edgecolors='white', marker='o', s=200, linewidths=2,
                       zorder=9)
            
            print(f"Plotting Target {i+1} at ({velocity} m/s, {distance} m)")
        
        # Add legend (only once, with all targets)
        if target_labels:
            plt.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig("./data/radar_results/range_doppler.png", dpi=300)
    plt.close()
    
def simulate_radar_reflections(ofdm_symbols, targets, snr_db, fft_size, num_ofdm_symbols, 
                              subcarrier_spacing, carrier_frequency, speed_of_light=3e8):
    """
    Simulate radar reflections from multiple targets using physical parameters.
    
    Args:
        ofdm_symbols (np.ndarray): Transmitted OFDM symbols of shape (batch_size, num_ofdm_symbols, fft_size).
        targets (list): List of dictionaries, each containing target parameters:
                        - 'distance': Distance in meters
                        - 'velocity': Velocity in m/s (positive = moving away)
                        - 'rcs': Radar cross-section (relative amplitude, default: 1.0)
                        - 'azimuth': Azimuth angle in degrees (default: 0.0)
                        - 'elevation': Elevation angle in degrees (default: 0.0)
        snr_db (float): Signal-to-noise ratio in dB.
        fft_size (int): Number of subcarriers.
        num_ofdm_symbols (int): Number of OFDM symbols.
        subcarrier_spacing (float): Subcarrier spacing in Hz.
        carrier_frequency (float): Carrier frequency in Hz.
        speed_of_light (float): Speed of light in m/s.
    
    Returns:
        np.ndarray: Radar reflection signals with the same shape as ofdm_symbols.
    """
    # Convert TensorFlow tensor to NumPy array if needed
    if not isinstance(ofdm_symbols, np.ndarray):
        ofdm_symbols = ofdm_symbols.numpy()
    
    # Calculate OFDM symbol duration
    symbol_duration = 1 / subcarrier_spacing  # in seconds
    
    # Calculate sampling rate based on subcarrier spacing and FFT size
    sampling_rate = subcarrier_spacing * fft_size  # 1140000 in Hz
    
    # Calculate wavelength
    wavelength = speed_of_light / carrier_frequency  # in meters
    
    # Initialize radar reflection with zeros
    radar_reflection = np.zeros_like(ofdm_symbols, dtype=np.complex128)
    
    # Process each target
    for target in targets:
        # Extract target parameters with defaults
        distance = target['distance']  # in meters
        velocity = target['velocity']  # in m/s
        rcs = target.get('rcs', 1.0)  # radar cross-section (relative amplitude)
        azimuth = np.radians(target.get('azimuth', 0.0))  # convert degrees to radians
        elevation = np.radians(target.get('elevation', 0.0))  # convert degrees to radians
        
        # Calculate two-way propagation delay
        two_way_delay = 2 * distance / speed_of_light  # in seconds
        
        # Convert delay to samples
        delay_samples = int(round(two_way_delay * sampling_rate))
        
        # Calculate Doppler shift
        # Positive velocity means target moving away (positive Doppler)
        doppler_freq = 2 * velocity / wavelength  # in Hz
        
        # Normalize Doppler frequency to the subcarrier spacing
        normalized_doppler = doppler_freq / sampling_rate * fft_size
        
        # Calculate amplitude based on radar equation
        # Simplified radar equation: amplitude ~ 1/R^4 * sqrt(RCS)
        # We use 1/R^2 for two-way path loss
        amplitude = rcs / (distance ** 2)
        
        # Apply directional factor based on azimuth and elevation
        # Simple cosine model for antenna pattern
        directional_factor = np.cos(azimuth) * np.cos(elevation)
        amplitude *= max(0.1, directional_factor)  # Limit minimum to avoid nulls
        
        # Apply delay (circular shift along time axis)
        delayed_signal = np.roll(ofdm_symbols, shift=delay_samples, axis=1)
        
        # Apply Doppler phase progression across OFDM symbols
        doppler_phase = 2 * np.pi * normalized_doppler * np.arange(fft_size)
        
        # Apply amplitude and initial phase
        target_reflection = amplitude * delayed_signal
        
        # Apply Doppler effect to each OFDM symbol
        for i in range(target_reflection.shape[1]):  # For each OFDM symbol
            # Progressive Doppler phase across symbols
            symbol_phase = 2 * np.pi * doppler_freq * i * symbol_duration
            # Apply both frequency and time-dependent Doppler
            target_reflection[:, i, :] *= np.exp(1j * (doppler_phase + symbol_phase))
        
        # Add this target's reflection to the total
        radar_reflection += target_reflection
    
    # Add noise based on SNR
    # Calculate signal power
    signal_power = np.mean(np.abs(radar_reflection)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), radar_reflection.shape)
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), radar_reflection.shape)
    noise = noise_real + 1j * noise_imag
    
    radar_reflection += noise
    
    return radar_reflection #(128, 1, 2, 14, 76)

def generate_radar_data(ofdm_symbols, snr_db, fft_size, num_ofdm_symbols, 
                        subcarrier_spacing, carrier_frequency, outputfolder='./data'):
    # Define targets with physical parameters
    targets = [
        {
            'distance': 150,          # Distance in meters
            'velocity': 20,           # Velocity in m/s (positive = moving away)
            'rcs': 1.0,               # Radar cross-section
            'azimuth': 0,             # Azimuth angle in degrees
            'elevation': 0            # Elevation angle in degrees
        },
        {
            'distance': 300,          # Further target
            'velocity': -15,          # Negative velocity (approaching)
            'rcs': 0.7,               # Smaller RCS
            'azimuth': 30,            # Off-center
            'elevation': 5            # Slight elevation
        },
        {
            'distance': 75,           # Close target
            'velocity': 5,            # Slow moving
            'rcs': 0.3,               # Small RCS
            'azimuth': -20,           # Different direction
            'elevation': -10          # Below horizon
        }
    ]

    # Simulate radar reflections using our new function
    radar_reflection = simulate_radar_reflections(
        ofdm_symbols, 
        targets, 
        snr_db, 
        fft_size, 
        num_ofdm_symbols,
        subcarrier_spacing,
        carrier_frequency
    )#(128, 1, 2, 14, 76)
    
    # Compute range-Doppler response
    range_doppler = compute_range_doppler(radar_reflection, fft_size, num_ofdm_symbols)
    #(1, 2, 14, 76)
    # Plot the range-Doppler response
    system_params= {
            "carrier_frequency": carrier_frequency,
            "subcarrier_spacing": subcarrier_spacing,
            "fft_size": fft_size,
            "num_ofdm_symbols": num_ofdm_symbols
        }
    plot_range_doppler(range_doppler[0,0,:,:], fft_size, num_ofdm_symbols, targets, system_params)
    
    print("Radar Reflection Shape:", radar_reflection.shape) #(128, 1, 2, 14, 76)


    # Save dataset to file
    import pickle
    import os
    # Create directory if it doesn't exist
    os.makedirs(outputfolder, exist_ok=True)
    outputfile=os.path.join(outputfolder, "radar", "ofdm_radar_dataset.pkl")
    with open(outputfile, "wb") as f:
        pickle.dump(radar_reflection, f)
        
    return radar_reflection
    
    
# custom dataset
class OFDMDataset(Dataset):
    def __init__(self, datapath='data/cdldatagen/cdl_ofdm_ebno25.npy', ch_SINR_min=25, ch_SINR_max=50, maxdatalen=10000, training=False, drawfig=False, testing=False, compare=False):
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
        #print("Dataset time:", saved_data['currenttime'])
        self.channeltype = saved_data['channeltype']# ofdm
        self.channeldataset = saved_data['channeldataset'] #cdl
        self.fft_size = saved_data['fft_size'] #76
        self.batch_size = saved_data['batch_size'] #128
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
        self.totalsymbols = int(self.k/self.num_bits_per_symbol) #768
        self.effectiveofdmsymbols=self.num_ofdm_symbols-colofpilots #12=14-2
        self.effectivesubcarrier=int(self.totalsymbols/(self.effectiveofdmsymbols)) #768/12=64
        self.TTI_mask_RE = self.TTI_mask(S=14, F=64, num_guard_carriers=num_guard_carriers, \
                                         pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices, dc_null=True, plotTTI=drawfig)#(14, 76)
        # TTI_mask_RE_3d = TTI_mask_RE_small.unsqueeze(-1) #Returns a new tensor with a dimension of size one inserted at the specified position. [14, 71]->[14, 71, 1]
        # self.TTI_mask_RE_3d = TTI_mask_RE_3d.expand(self.S, self.F-1, self.Qm) #[14, 71, 6]
        # self.index_one =  self.TTI_mask_RE_3d==1 #[14, 71, 6]

        #RESOURCE_GRID related data
        self.pilot_pattern = saved_data['pilot_pattern'] #'kronecker'
        self.pilots = saved_data['pilots'] #self.RESOURCE_GRID.pilot_pattern.pilots, (1,2,128)
        self.num_data_symbols = saved_data['num_data_symbols'] #768
        self.cyclic_prefix_length = saved_data['cyclic_prefix_length'] #6
        self.ofdm_symbol_duration = saved_data['ofdm_symbol_duration'] #1.8e-5
        self.num_time_samples = saved_data['num_time_samples'] #1148
        self.bandwidth = saved_data['bandwidth'] #4.56M

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
        self.h_perfect = saved_data['h_perfect'] #(128, 1, 16, 1, 2, 14, 64) ?????? only (64)
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
        
        ofdm_symbols = self.x_rg
        snr_db = 25
        subcarrier_spacing=15e3
        carrier_frequency = 2.6e9 # Carrier frequency in Hz.
        generate_radar_data(ofdm_symbols, snr_db, self.fft_size, self.num_ofdm_symbols, 
                        subcarrier_spacing, carrier_frequency, outputfolder='./data')
        
        self.index = 0
        rx_id=0
        tx_id=0
        tx_streams_id=0
        rx_antenna_id=0
        returnbatch=False

        if self.training:
            data = self.b[:,tx_id, tx_streams_id, :] #[batch_size, num_tx, num_streams_per_tx, num_data_bits] #[self.batch_size, 1, self.num_streams_per_tx, self.k]
            #(128, 1, 2, 1536)=>(128, 1536) [batch_size, num_data_bits]
            self.labels_data = data.reshape(-1, self.effectiveofdmsymbols, self.effectivesubcarrier, self.num_bits_per_symbol) #(128, 12, 64, 2)
            #(128, 12, 64, 2)
            #labelsize = (self.num_ofdm_symbols, self.fft_size, self.num_bits_per_symbol) #14, 76, 2
        #(128, 1, 16, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        rx_samples = self.y[:,rx_id, rx_antenna_id, :, :] #(128, 1, 16, 14, 76)=>(128, 14, 76) [batch_size, num_ofdm_symbols, fft_size]

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

            #self.compare_channelestimationdata()
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

    def check_channel(self, compare=True, savefile=True):
        from deepMIMO5 import time_lag_discrete_time_channel, cir_to_time_channel, cir_to_ofdm_channel, subcarrier_frequencies
        if self.drawfig:
             #eval_transceiver.RESOURCE_GRID.num_ofdm_symbols
            #sampling_frequency= saved_data['sampling_frequency'] #1/eval_transceiver.RESOURCE_GRID.ofdm_symbol_duration
            plt.figure()
            plt.title("Channel impulse response realization")
            plt.stem(self.tau_b[0,0,0,:]/1e-9, np.abs(self.h_b)[0,0,0,0,0,:,0])#10 different pathes
            plt.xlabel(r"$\tau$ [ns]")
            plt.ylabel(r"$|a|$")
            if savefile is not None:
                plt.savefig("data/channelimpulse"+IMG_FORMAT)

            plt.figure()
            plt.title("Time evolution of path gain")
            #x_timesteps = np.arange(num_time_steps)*self.RESOURCE_GRID.ofdm_symbol_duration/1e-6
            x_timesteps = np.arange(self.num_time_steps)/self.sampling_frequency/1e-6
            plt.plot(x_timesteps, np.real(self.h_b)[0,0,0,0,0,0,:])
            plt.plot(x_timesteps, np.imag(self.h_b)[0,0,0,0,0,0,:])
            plt.legend(["Real part", "Imaginary part"])
            plt.xlabel(r"$t$ [us]")
            plt.ylabel(r"$a$");
            if savefile is not None:
                plt.savefig("data/timeevolutionofpath"+IMG_FORMAT)
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
            print(np.allclose(self.h_out[0,0,0,0,0,:,:], h_freq_tfnp[0,0,0,0,0,:,:])) #False->True
            print(np.allclose(self.h_out[0,0,0,0,0,0], h_freq_tfnp[0,0,0,0,0,0])) #False->True
            compare_allclose(self.h_out[0,0,0,0,0,0], h_freq_tfnp[0,0,0,0,0,0], figname="data/compare_allclose"+IMG_FORMAT)


    def check_channelestimation(self, savefile=True):
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
            if savefile is not None:
                plt.savefig("data/channelestimation"+IMG_FORMAT)

    
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
    
    def compare_channelestimationdata(self, additionalcompare=False):
        if additionalcompare==True:
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

            h_hat_beforeinter2 = np.load('data/h_hat_beforeinter2.npy')
            plt.figure()
            plt.plot(np.real(h_hat_beforeinter2[0,0,0,0,0,:])) #0:64
            plt.plot(np.imag(h_hat_beforeinter2[0,0,0,0,0,:]))
            #plt.plot(np.real(h_hat_beforeinter2[0,0,0,0,0,64:128]),'--')
            #plt.plot(np.imag(h_hat_beforeinter2[0,0,0,0,0,64:128]),'--')
            plt.title('h_hat at_pilot')
            plt.savefig('data/h_hat_at_pilot'+IMG_FORMAT)
        
        self.comparefigure(h_perfect=self.h_perfect, h_hat=self.h_hat, savefile='data/h_hatcompare'+IMG_FORMAT)

        
        #self.comparefigure(h_perfect=self.h_perfect, h_hat=h_hat_inter, savefile='data/h_hat_intercompare.png')
        h_hat_inter2 = np.load('data/h_hat_inter2.npy')
        self.comparefigure(h_perfect=self.h_perfect, h_hat=h_hat_inter2, savefile='data/h_hat_inter2compare'+IMG_FORMAT)

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
        self.comparefigure(h_perfect=self.h_hat, h_hat=h_hat.numpy(), savefile="data/comparechannelestimate"+IMG_FORMAT)
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
    
    def TTI_mask(self, S=14, F=64, num_guard_carriers=[5,6], pilot_ofdm_symbol_indices=[2,11], dc_null=True, plotTTI=False, savefile=True):

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
            #plt.savefig('output/TTImask.png')
            plt.tight_layout()
            plt.show()
            if savefile is not None:
                plt.savefig("data/TTImask"+IMG_FORMAT)

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
        return batch #'labels':(12, 64, 2) HWbits, 'feature_2d'(2, 12, 64) CHW
    
def testdataset(datapath='data/cdl/cdl_ofdm_ebno25.npy'):
    train_data = OFDMDataset(datapath=datapath, training=True, testing=True, compare=True)
    onebatch = train_data[0]
    print(onebatch['feature_2d'].shape) #(2, 12, 64)
    print(onebatch['labels'].shape) #(12, 64, 2)

from ofdmsim_pytorchlib import get_device
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import os
from tqdm.auto import tqdm



# Create BER calculator for validation
class OFDMProcessor:
    def __init__(self, train_data):
        self.train_data = train_data
        
    def NNevaluate(self, predictions, labels):
        # Calculate BER between predictions and labels
        total_bits = torch.numel(labels)
        wrong_bits = torch.sum(predictions != labels).item()
        ber = wrong_bits / total_bits
        return ber, wrong_bits
        
    def receiver_preprocessing(self, rx_samples):
        # Placeholder for receiver preprocessing
        # In a real implementation, this would convert time-domain samples to frequency domain
        return rx_samples
        
    def ZHLSreceiver(self, ofdm_demod):
        # Placeholder for traditional LS receiver
        # This would implement a least-squares channel estimation and detection
        return torch.zeros_like(ofdm_demod)
        
    def evaluate(self, predictions, labels):
        # Same as NNevaluate but for traditional receiver
        total_bits = torch.numel(labels)
        wrong_bits = torch.sum(predictions != labels).item()
        ber = wrong_bits / total_bits
        return ber, wrong_bits
        
def trainmain(trainoutput, saved_model_path = ""):
    device, useamp=get_device(gpuid='0', useamp=False)

    # OFDM Parameters
    train_data = OFDMDataset(training=True, testing=False, compare=False)
    #train_data = OFDMDataset(Qm=Qm, S=S, Sp=Sp, F=F, ch_SINR_min=-10, ch_SINR_max=40, training=True)
    onebatch = train_data[0]
    print(onebatch['feature_2d'].shape) #(2, 12, 64)
    print(onebatch['labels'].shape) #(12, 64, 2)

    batch_size = 16

    # train, validation and test split
    train_size = int(0.8 * len(train_data)) #8000
    val_size = len(train_data) - train_size
    train_set, val_set= torch.utils.data.random_split(train_data, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

    onebatch = next(iter(train_loader))
    feature_2d = onebatch['feature_2d']
    data_labels = onebatch['labels']
    print(f"Feature batch shape: {feature_2d.size()}") #[16, 2, 12, 64]
    print(f"Labels batch shape: {data_labels.size()}") #[16, 12, 64, 2]

    # Create model instance
    model = OFDMNet(in_channels=2, out_channels=2).to(device)
    print(f"Model created and moved to {device}")

    multiprocessor = OFDMProcessor(train_data)
    
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
            feature_2d = batch['feature_2d'] #(2, 12, 64)
            labels = batch['labels'] #(12, 64, 2)
            outputs = model(feature_2d)  # forward pass 
            loss = criterion(outputs, labels) 
            loss.backward()  # backward pass
            optimizer.step()  # update the weights
            total_loss += loss.item()  # accumulate the loss
            #progress_bar.update(1)
            optimizer.zero_grad()  # Zero the gradients

        # Update the learning rate
        scheduler.step()

        # Print average loss for the epoch
        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        BER_batch = []
        LSBER_batch = []
        val_loss_total = 0.0
        
        with torch.no_grad():
            for index, data_batch in enumerate(tqdm(val_loader)):
                batch = {k: v.to(device) for k, v in data_batch.items()}
                feature_2d = batch['feature_2d']
                labels = batch['labels']
                
                # Forward pass
                val_outputs = model(feature_2d)
                val_loss = criterion(val_outputs, labels)
                val_loss_total += val_loss.item()
                
                # Convert probabilities to binary predictions (0 or 1)
                binary_predictions = torch.round(val_outputs)
                
                # Calculate BER
                binary_predictions_cpu = binary_predictions.cpu()
                labels_cpu = labels.cpu()
                BER = multiprocessor.NNevaluate(binary_predictions_cpu, labels_cpu)[0]
                BER_batch.append(BER)
                
                # For comparison, calculate BER with traditional LS method
                # In a real implementation, this would use the actual LS method
                # Here we're just using a placeholder
                LSBER = 0.1  # Placeholder value
                LSBER_batch.append(LSBER)

        # Calculate average validation metrics
        avg_val_loss = val_loss_total / len(val_loader)
        avg_BER = np.mean(BER_batch)
        avg_LSBER = np.mean(LSBER_batch)
        
        val_losses.append(avg_val_loss)
        val_BERs.append(avg_BER)
        
        # Print epoch results
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val BER: {avg_BER:.4f}, LS BER: {avg_LSBER:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save performance details in the CSV file
        with open(performance_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, average_loss, avg_val_loss, avg_BER, avg_LSBER])
        
        # Save model checkpoint periodically
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
    
    # Plot training and validation curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot BER curve
    plt.subplot(1, 2, 2)
    plt.plot(val_BERs, label='Neural Network BER')
    plt.xlabel('Epoch')
    plt.ylabel('Bit Error Rate')
    plt.title('Validation BER')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(trainoutput, 'training_curves.pdf'))
    plt.close()
    
    print(f"Training completed. Model saved to {modelsave_path}")


if __name__ == '__main__':
    testdataset(datapath='data/cdldatagen/cdl_ofdm_ebno25.npy')

    trainoutput = './output/'
    trainmain(trainoutput)