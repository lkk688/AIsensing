#pip install sionna
import tensorflow as tf
import sionna
from sionna.channel import OFDMChannel
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
from sionna.mapping import Mapper, Demapper
from sionna.utils import QAMSource, ebnodb2no
import numpy as np
import matplotlib.pyplot as plt

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
def plot_range_doppler(range_doppler, fft_size, num_ofdm_symbols):
    """
    Plot the range-Doppler response.
    
    Args:
        range_doppler (np.ndarray): Range-Doppler response of shape (fft_size, num_ofdm_symbols).
        fft_size (int): Number of subcarriers.
        num_ofdm_symbols (int): Number of OFDM symbols.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(range_doppler), aspect='auto', cmap='viridis',
               extent=[-num_ofdm_symbols//2, num_ofdm_symbols//2, -fft_size//2, fft_size//2])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Doppler Frequency (bins)')
    plt.ylabel('Range (bins)')
    plt.title('Range-Doppler Response')
    plt.show()
    plt.savefig("./data/radar_results/range_doppler.png")
    plt.close()

def generate_radar_data():
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
    )
    
    # Compute range-Doppler response
    range_doppler = compute_range_doppler(radar_reflection, fft_size, num_ofdm_symbols)

    # Plot the range-Doppler response
    plot_range_doppler(range_doppler, fft_size, num_ofdm_symbols)
    return radar_reflection
    
def generate_ofdm_data(outputfolder="./data/radar_results"):
    # Parameters
    batch_size = 64  # Number of samples in the dataset
    num_tx_antennas = 1  # Number of transmit antennas
    num_rx_antennas = 1  # Number of receive antennas
    fft_size = 76 #64  # Number of OFDM subcarriers
    num_ofdm_symbols = 14  # Number of OFDM symbols per frame
    num_bits_per_symbol = 2 #4  # Bits per symbol (16-QAM)
    snr_db = 20  # Signal-to-noise ratio in dB
    carrier_frequency = 3.5e9  # Carrier frequency in Hz (5G NR frequency range)
    delay_spread = 100e-9  # Delay spread in seconds (5G urban microcell scenario)
    subcarrier_spacing = 15e3  # Subcarrier spacing in Hz (15 kHz for 5G NR)
    cyclic_prefix_length = 6 #16  # Length of cyclic prefix
    pilot_ofdm_symbol_indices = [2, 11]  # OFDM symbols containing pilots
    dc_null = True  # Null the DC carrier
    num_guard_carriers = [5, 6]  # Number of guard carriers [left, right]
    
    # Define the OFDM resource grid
    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=num_tx_antennas,
        num_streams_per_tx=1,
        cyclic_prefix_length=cyclic_prefix_length,
        num_guard_carriers=num_guard_carriers,
        dc_null=dc_null,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices
    )

    # Generate QAM symbols for communication
    qam_source = QAMSource(num_bits_per_symbol=num_bits_per_symbol)
    mapper = ResourceGridMapper(resource_grid=rg)

    # Generate random bits and map to QAM symbols
    bits = tf.random.uniform(shape=[batch_size, rg.num_data_symbols, num_bits_per_symbol], minval=0, maxval=2, dtype=tf.int32)
    #(64, 768, 2)
    # Fix the QAMSource call - reshape bits to match expected input format
    bits_reshaped = tf.reshape(bits, [-1, num_bits_per_symbol]) #(49152, 2)
    qam_symbols = qam_source(bits_reshaped)
    qam_symbols = tf.reshape(qam_symbols, [batch_size, rg.num_data_symbols])
    #
    ofdm_symbols = mapper(qam_symbols)  # Map to OFDM resource grid

    # Simulate 5G channel
    channel = OFDMChannel(
        num_tx_antennas=num_tx_antennas,
        num_rx_antennas=num_rx_antennas,
        resource_grid=rg,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread,
        add_awgn=True,
        return_channel=True
    )

    # Pass OFDM symbols through the channel
    y, h = channel(ofdm_symbols, snr_db)  # y: received signal, h: channel response

    radar_reflection = generate_ofdm_data()
    
    # Save dataset
    dataset = {
        "ofdm_symbols": ofdm_symbols.numpy(),  # Transmitted OFDM symbols
        "received_symbols": y.numpy(),  # Received OFDM symbols (communication)
        "channel_response": h.numpy(),  # Channel response (communication)
        "radar_reflection": radar_reflection,  # Radar reflection (sensing)
        "targets": targets,  # Target parameters
        "system_params": {
            "carrier_frequency": carrier_frequency,
            "subcarrier_spacing": subcarrier_spacing,
            "fft_size": fft_size,
            "num_ofdm_symbols": num_ofdm_symbols
        }
    }

    # Print dataset shapes
    print("OFDM Symbols Shape:", dataset["ofdm_symbols"].shape)
    #Transmitted OFDM symbols (shape: [batch_size, num_ofdm_symbols, fft_size]).
    print("Received Symbols Shape:", dataset["received_symbols"].shape)
    #Received OFDM symbols after passing through the 5G channel (shape: [batch_size, num_ofdm_symbols, fft_size]).
    print("Channel Response Shape:", dataset["channel_response"].shape)
    #Channel response (shape: [batch_size, num_rx_antennas, num_tx_antennas, num_ofdm_symbols, fft_size]).
    print("Radar Reflection Shape:", dataset["radar_reflection"].shape)
    #Radar reflection signals (shape: [batch_size, num_ofdm_symbols, fft_size]).

    # Save dataset to file
    import pickle
    import os
    # Create directory if it doesn't exist
    os.makedirs(outputfolder, exist_ok=True)
    outputfile=os.path.join(outputfolder, "ofdm_radar_dataset.pkl")
    with open(outputfile, "wb") as f:
        pickle.dump(dataset, f)

    print("Dataset saved to ofdm_radar_dataset.pkl")

if __name__ == '__main__':
    #Simulate radar reflections by introducing a delay and Doppler shift to the transmitted OFDM signal.
    generate_ofdm_data()