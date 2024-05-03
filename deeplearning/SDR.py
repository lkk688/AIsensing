import adi
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
#from pyinstrument import Profiler # for code optimisation:
plt.rcParams['font.size'] = 8.0
class SDR:
    def __init__(self, SDR_TX_IP, SDR_TX_FREQ, SDR_TX_GAIN=-80, SDR_RX_GAIN=0, SDR_TX_SAMPLERATE=1e6, SDR_TX_BANDWIDTH=1e6):
    
        self.SDR_TX_IP = SDR_TX_IP # IP address of the TX SDR device
        self.SDR_TX_FREQ = int(SDR_TX_FREQ) # TX center frequency in Hz
        self.SDR_RX_FREQ = int(SDR_TX_FREQ) # RX center frequency in Hz
        self.SDR_TX_GAIN = int(SDR_TX_GAIN) # TX gain in dB
        self.SDR_RX_GAIN = int(SDR_RX_GAIN) # RX gain in dB
        self.SDR_TX_SAMPLERATE = int(SDR_TX_SAMPLERATE) # TX sample rate (samples/second)
        self.SDR_TX_BANDWIDTH = int(SDR_TX_BANDWIDTH) # TX bandwidth (Hz)
        self.num_samples=0
        
    def SDR_TX_start(self):
        '''
        Initialize and start the SDR transmitter.

        This method sets up the SDR with the specified configuration parameters. It includes:
        - Initializing the SDR device using the ad9364 driver and the provided IP address.
        - Destroying any existing TX buffers to prepare for new transmissions.
        - Setting the local oscillator (LO) frequencies for both TX and RX.
        - Configuring the gain control mode for the TX channel to manual.
        - Setting the sample rate, RF bandwidth for both TX and RX, and hardware gains.

        The method ensures that the SDR transmitter is ready to send data with the configured settings.
        '''
        # Initialize the SDR device using the Analog Devices ad9364 driver
        self.sdr_tx = adi.ad9364(self.SDR_TX_IP)

        # Clear any existing transmit buffers
        self.sdr_tx.tx_destroy_buffer()

        # Set the local oscillator (LO) frequencies for transmit and receive
        self.sdr_tx.tx_lo = self.SDR_TX_FREQ  # Transmit frequency
        self.sdr_tx.rx_lo = self.SDR_TX_FREQ  # Receive frequency (set to the same as TX)

        # Set gain control mode to manual for the TX channel
        self.sdr_tx.gain_control_mode_chan0 = 'manual'

        # Configure the sample rate for both TX and RX
        self.sdr_tx.sample_rate = self.SDR_TX_SAMPLERATE

        # Set the RF bandwidth for both TX and RX
        self.sdr_tx.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH
        self.sdr_tx.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH

        # Set the hardware gain for both TX and RX
        self.sdr_tx.tx_hardwaregain_chan0 = self.SDR_TX_GAIN  # TX gain
        self.sdr_tx.rx_hardwaregain_chan0 = self.SDR_RX_GAIN  # RX gain

        # Clear transmit buffer again to ensure a clean start
        self.sdr_tx.tx_destroy_buffer()

    def SDR_gain_set(self, tx_gain, rx_gain):
        '''
        Set the TX and RX gain
        '''
        self.sdr_tx.tx_hardwaregain_chan0 = tx_gain
        self.sdr_tx.rx_hardwaregain_chan0 = rx_gain

    def SDR_TX_send(self, SAMPLES, max_scale=1, cyclic=False):
        '''
        Transmit the given signal samples through the SDR transmitter.

        Parameters:
        SAMPLES (Tensor or ndarray): The signal samples to be transmitted. 
                                     Can be a NumPy array or a PyTorch tensor.
        max_scale (float): Scaling factor to adjust the amplitude of the signal.
        cyclic (bool): If True, the transmitted signal will be repeated in a cyclic manner.

        This method prepares the samples for transmission by performing several steps:
        - Converts PyTorch tensors to NumPy arrays, if necessary.
        - Normalizes the signal to remove any DC offset.
        - Scales the signal's amplitude.
        - Adjusts the sample values to match the expected range of the SDR hardware.
        - Configures the SDR to either transmit the signal cyclically or just once.
        - Finally, transmits the prepared samples.
        '''

        self.sdr_tx.tx_destroy_buffer()  # Clear any existing buffer

        # Determine the number of samples based on whether SAMPLES is a NumPy array or a PyTorch tensor
        if isinstance(SAMPLES, np.ndarray):
            self.num_samples = SAMPLES.size
        elif isinstance(SAMPLES, torch.Tensor):
            self.num_samples = SAMPLES.numel()
            SAMPLES = SAMPLES.numpy()  # Convert to NumPy array if it's a PyTorch tensor

        # Normalize the signal to remove DC offset
        samples = SAMPLES - np.mean(SAMPLES)

        # Scale the samples to their maximum amplitude and adjust according to max_scale
        samples = (samples / np.max(np.abs(samples))) * max_scale

        # Scale the signal to the dynamic range expected by the SDR hardware
        samples *= 2**14  # PlutoSDR, for example, expects sample values in the range -2^14 to +2^14

        # Set cyclic buffer mode if required
        self.sdr_tx.tx_cyclic_buffer = cyclic

        # Transmit the prepared samples
        self.sdr_tx.tx(samples)
    
    def SDR_TX_stop(self):
        '''
        Stop the SDR TX
        '''
        self.sdr_tx.tx_destroy_buffer() 
        self.sdr_tx.rx_destroy_buffer()

    def SDR_RX_receive(self, n_SAMPLES=None, normalize=True):
        '''
        Receive signal samples from the SDR receiver.

        Parameters:
        n_SAMPLES (int, optional): The number of samples to receive. 
                                   If not specified, defaults to four times the number of transmitted samples.
        normalize (bool): If True, normalizes the received signal to a maximum amplitude of 1.

        This method handles the reception of signal samples from the SDR hardware by:
        - Determining the number of samples to receive.
        - Clearing any existing buffer in the receiver to prepare for new data.
        - Setting the buffer size to the specified number of samples.
        - Receiving the samples from the SDR hardware.
        - Optionally normalizing the amplitude of the received signal.
        - Converting the received samples to a PyTorch tensor for further processing.

        Returns:
        Tensor: The received signal samples as a PyTorch tensor.
        '''
        # Determine the number of samples to receive
        if n_SAMPLES is None:
            n_SAMPLES = self.num_samples * 4  # Default to four times the number of transmitted samples
        if n_SAMPLES <= 0:
            n_SAMPLES = 1  # Ensure at least one sample is received

        # Clear the receiver buffer to prepare for new data
        self.sdr_tx.rx_destroy_buffer()

        # Set the buffer size for receiving data
        self.sdr_tx.rx_buffer_size = n_SAMPLES

        # Receive the samples from the SDR hardware
        a = self.sdr_tx.rx()

        # Normalize the signal amplitude if required
        if normalize:
            a = a / np.max(np.abs(a))

        # Convert the received samples to a PyTorch tensor
        return torch.tensor(a, dtype=torch.complex64)
    
    def find_good_max_TX_gain_value(self, SDR_RX_GAIN=-30, Max_SINR=20, make_plot=True):
        '''
        Find the maximum transmit gain value that does not exceed a specified SINR.

        Parameters:
        SDR_RX_GAIN (int): The receive gain setting for the SDR.
        Max_SINR (float): The maximum desired SINR (Signal-to-Noise Ratio).

        This method operates by:
        - Generating a QPSK signal to be used as a test transmission.
        - Measuring the noise power of the received signal at a low TX gain.
        - Iteratively increasing the TX gain and measuring the SINR until the desired SINR is reached.
        - Plotting SINR versus TX gain for analysis.

        Returns:
        int: The transmit gain value that achieves the desired SINR without exceeding it.
        '''
        # Generate a QPSK signal
        qpsk_symbols = np.tile(np.array([0.7+0.7j, -0.7+0.7j, -0.7-0.7j, 0.7-0.7j]), 10)
        qpsk_symbols = np.fft.ifft(np.fft.ifftshift(qpsk_symbols))

        # Initial setup: stop TX, set gains, and measure noise power
        self.SDR_TX_stop()
        self.SDR_gain_set(tx_gain=-80, rx_gain=SDR_RX_GAIN)
        time.sleep(0.3)  # Wait for settings to take effect
        noise_sample = self.SDR_RX_receive(10000, normalize=False)
        noise_power = torch.mean(torch.abs(noise_sample)**2)
        # Prepare to record SINR and TX gain values
        SINR = []
        TX_GAIN_t = []

        # Iteratively test different TX gain values
        for SDR_TX_GAIN in range(-70, 0):
            # Transmit the QPSK signal and set the current TX gain
            self.SDR_TX_send(qpsk_symbols, max_scale=1, cyclic=True)
            self.SDR_gain_set(tx_gain=SDR_TX_GAIN, rx_gain=SDR_RX_GAIN)
            TX_GAIN_t.append(SDR_TX_GAIN)

            # Measure the signal power and calculate SINR
            sample = self.SDR_RX_receive(10000, normalize=False)
            signal_power = torch.mean(torch.abs(sample)**2)
            SINR.append(10 * torch.log10(signal_power / noise_power).item())

            # Stop if the maximum desired SINR is reached
            if SINR[-1] > Max_SINR:
                self.SDR_TX_stop()
                self.SDR_gain_set(tx_gain=-80, rx_gain=SDR_RX_GAIN)
                break

        # Output and plot the results
        self.SDR_gain_set(tx_gain=-80, rx_gain=SDR_RX_GAIN)
        self.SDR_TX_stop()
        if make_plot:
            plt.figure(figsize=(10, 4))
            plt.scatter(TX_GAIN_t, SINR)
            plt.title(f'SINR vs TX Gain. RX Gain set to {SDR_RX_GAIN}\nFreq={self.SDR_TX_FREQ/1e6} MHz, BW={self.SDR_TX_BANDWIDTH/1e6} MHz, SR={self.SDR_TX_SAMPLERATE/1e6} MHz')
            plt.grid()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('PlutoSDR TX Gain')
            plt.ylabel('Measured SINR')
            plt.tight_layout()
            plt.show()
            plt.savefig('pics/SINR_vs_TX_Gain.png')

        return SDR_TX_GAIN

        
    def show_params(self):
        print(f'RX LO: {self.sdr_tx.rx_lo}')
        print(f'TX LO: {self.sdr_tx.tx_lo}')
        print(f'RX RF Bandwidth: {self.sdr_tx.rx_rf_bandwidth}')
        print(f'TX RF Bandwidth: {self.sdr_tx.tx_rf_bandwidth}')
        print(f'Sample Rate: {self.sdr_tx.sample_rate}')
        print(f'RX Buffer Size: {self.sdr_tx.rx_buffer_size}')
        print(f'Gain Control Mode: {self.sdr_tx.gain_control_mode_chan0}')
        print(f'RX Gain: {self.sdr_tx.rx_hardwaregain_chan0}')
        print(f'TX Gain: {self.sdr_tx.tx_hardwaregain_chan0}')
        print(f'Filter: {self.sdr_tx.filter}')
        print(f'TX Cyclic Buffer: {self.sdr_tx.tx_cyclic_buffer}')



import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
import time
from scipy import signal
from scipy.stats import pearsonr


class SDR1t1r(Layer):
    r"""
    Layer for implementing 1T1R SDR radio over-the-air connection for Sionna, utilising PlutoSDR radio.
    Inputs time domain IQ to be transmitted and and outputs the received time domain IQ data, readily synchronized.

    Args:
        SDR_TX_IP (str): IP address of the TX SDR device
        SDR_TX_FREQ (int): TX center frequency in Hz
        SDR_TX_BANDWIDTH (int): SDR filter cutoff
        SampleRate (int): Sample rate of the SDR

    Returns:
        out (tf.tensor): received time domain IQ data, readily synchronized
        SINR (float): SINR of the received signal
        SDR_TX_GAIN (int): TX gain used in the transmission
        SDR_RX_GAIN (int): RX gain used in the reception
        fails (int): how many times the process failed to reach pearson r > self.corr_threshold
        corr (float): pearson correlation between TX and RX signal
        sdr_time (float): how long the SDR process took in seconds
    """
    def __init__(self, SDR_TX_IP, SDR_TX_FREQ, SDR_TX_BANDWIDTH, SampleRate):        
        super().__init__()

        # class variables from inputs
        self.SDR_TX_IP = SDR_TX_IP # IP address of the TX SDR device
        self.SDR_TX_FREQ = int(SDR_TX_FREQ) # TX center frequency in Hz
        self.SDR_TX_BANDWIDTH = int(SDR_TX_BANDWIDTH) # SDR filter cutoff
        self.SampleRate = SampleRate # Sample rate of the SDR

        # setup the SDR
        self.sdr_pluto = adi.Pluto(self.SDR_TX_IP) # from which IP address the PlutoSDR can be found, default 192.168.2.1
        self.sdr_pluto.sample_rate = int(self.SampleRate) # set the samplerate

        # setup SDR TX
        self.sdr_pluto.tx_lo = self.SDR_TX_FREQ # set the frequency
        self.sdr_pluto.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH # set the SDR tx filter cutoff
        self.sdr_pluto.tx_destroy_buffer() # empty the tx buffer

        # SETUP sdr rx
        self.sdr_pluto.rx_lo = self.SDR_TX_FREQ # set the frequency
        self.sdr_pluto.gain_control_mode_chan0 = 'manual' # don't use AGC
        self.sdr_pluto.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH # rx filter cutoff 
        self.sdr_pluto.rx_destroy_buffer() # clear any data from rx buffer
        
        # further variables
        self.corr_threshold = 0.3 # correlation threshold for TTI detection. Below 0.2 correlation sync probably not right
        self.min_attempts=10 # how many retries before giving up if above thresholds are not met (while increasing TX power each time)
      
    def call(self, SAMPLES, SDR_TX_GAIN=0, SDR_RX_GAIN=30, add_td_samples = 0, debug=False):
        now = time.time() # for measuing the duration of the process

        out_shape = list(SAMPLES.shape) # store the input tensor shape
        num_samples = SAMPLES.shape[-1] # number of samples in the input

        # remove offsets
        flat_samples = tf.reshape(SAMPLES, [-1]) # flatten the input samples
        tx_std =  tf.math.reduce_std(flat_samples) # standard deviation of the input samples
        tx_mean = np.complex64(tf.math.reduce_mean(flat_samples)) # mean of the input samples
        tx_samples = tf.math.subtract(flat_samples, tx_mean) # remove DC offset

        # scale for SDR input
        tx_samples_abs = tf.math.abs(tx_samples) # absolute values of the samples
        tx_samples_abs_max = tf.reduce_max(tx_samples_abs,0) # take the maximum value of the samples
        tx_samples_max_sample = np.float32(tx_samples_abs_max) # convert to float32
        tx_samples = tf.math.divide(tx_samples , tx_samples_max_sample) # scale the tx_samples to max 1
        tx_samples = tf.math.multiply(tx_samples, 2**14) # = 2**14 # scale the samples to 16-bit

        # create the final IQ data for transmission
        leading_zeroes = tf.zeros(500, dtype=tf.dtypes.complex64) # leading 500 zeroes for noise floor measurement
        samples_with_leading_zeros = tf.concat([leading_zeroes, tx_samples], axis=0) # add the quiet for noise mesurements
        
        # internal counters
        fails = 0 # how many times the process failed to reach pearson r > self.corr_threshold
        success = 0 #  how many times the process reached pearson r > self.corr_threshold
        
        # send some parameters to SDR
        self.sdr_pluto.tx_cyclic_buffer = True # enable cyclic buffer for TX
        self.sdr_pluto.tx_hardwaregain_chan0 = int(SDR_TX_GAIN)     
        self.sdr_pluto.rx_hardwaregain_chan0 = int(SDR_RX_GAIN)
        self.sdr_pluto.rx_buffer_size = (num_samples+500)*3 # set the RX buffer size to 3 times the number of samples
        
        self.sdr_pluto.tx_destroy_buffer() # empty TX buffer
        self.sdr_pluto.tx(samples_with_leading_zeros) # start transmitting the samples in cyclic manner

        while success == 0:        
            
            # RX samples
            self.sdr_pluto.rx_destroy_buffer() # clear the RX buffer
            rx_samples = self.sdr_pluto.rx() # receive samples from the SDR

            # convert received IQ samples to tf tensor
            rx_samples_tf = tf.convert_to_tensor(rx_samples, dtype=tf.complex64)

            # remove any offset
            rx_mean = np.complex64(tf.math.reduce_mean(rx_samples_tf)) 
            rx_samples_tf = tf.math.subtract(rx_samples_tf, rx_mean)

            # set the same stdev as in the input samples
            rx_std = tf.math.reduce_std(rx_samples_tf)
            std_multiplier = np.float16(tx_std/ rx_std)*0.9 # for calculating new multiplier for same stdev in TX and RX
            rx_samples_tf = tf.math.multiply(rx_samples_tf, std_multiplier) # set the stdev

            # calculate the correlation between TX and RX signal and find the start symbol of the first full TTI with 500 samples of noise measurements in front
            TTI_corr = signal.correlate(rx_samples_tf, flat_samples,mode='full',method='fft')
            TTI_offset = tf.math.argmax(tf.math.abs(TTI_corr[0:int(len(rx_samples_tf)/2)]))-len(flat_samples)+1 
            if TTI_offset < 500+num_samples:
                TTI_offset = TTI_offset + 500 + num_samples

            # RX TTI symbols + the additional symbols
            rx_TTI = rx_samples_tf[TTI_offset:TTI_offset+num_samples+add_td_samples] 
            
            # RX noise for SINR calculation
            rx_noise =  rx_samples_tf[TTI_offset-450:TTI_offset-50]

            # calculate the pearson correlation between complex samples_orig and rx_TTI as acceptance metric
            corr = pearsonr(tf.math.abs(flat_samples), tf.math.abs(rx_samples_tf)[TTI_offset:TTI_offset+num_samples])[0]
            
            # calculate TX power, RX power & noise power
            tx_TTI_p = tf.math.reduce_variance(flat_samples) # TX power
            noise_p = tf.math.reduce_variance(rx_noise) # noise power
            rx_TTI_p = tf.math.reduce_variance(rx_TTI) # RX signal power
            SINR = 10*tf.experimental.numpy.log10(rx_TTI_p/noise_p) # calculate SINR from received powers
            
            if debug:
                titletext = f'SINR ={SINR:1.1f}, attempt={fails+1}, TTI start index = {TTI_offset}, correlation = {corr:1.2f}, TX_p/RX_p = {tx_TTI_p/rx_TTI_p:1.2f}'
                fig, axs = plt.subplots(3, 2)
                fig.set_size_inches(16, 7)
                fig.suptitle(titletext)
                axs[0,0].plot(10*np.log10(abs(rx_samples)/max(abs(rx_samples))), label='RX_dB')
                axs[0,0].legend()
                axs[0,0].set_title('TTI received 3 times, starting at random time')
                axs[0,1].plot((abs(rx_samples_tf)), label='abs(RXsample)')
                axs[0,1].axvline(x=TTI_offset, c='r', lw=3, label='TTI start')
                axs[0,1].plot(abs(abs(TTI_corr)/np.max(abs(TTI_corr))), label='Pearson R')
                axs[0,1].legend()
                axs[0,1].set_title('Correlator for syncing the start of the second received TTI')
                
                axs[1,0].plot(np.abs(flat_samples), label='abs(TX samples)')
                axs[1,0].set_ylim(0,tx_samples_max_sample)
                axs[1,0].legend()
                axs[1,0].set_title('Transmitted signal, one TTI')
                axs[1,1].plot((abs(rx_TTI)), label='abs(RX samples)')
                axs[1,1].set_ylim(0,tx_samples_max_sample)
                axs[1,1].legend()
                axs[1,1].set_title('Received signal, one TTI, syncronized')
                
                axs[2,0].psd(flat_samples, label='TX Signal')
                axs[2,0].legend()
                axs[2,0].set_title('Transmitted signal PSD')
                axs[2,1].psd(rx_TTI, label='RX signal')
                axs[2,1].psd(rx_noise, label='noise')
                axs[2,1].legend()
                axs[2,1].set_title('Received noise PSD and signal PSD')
                plt.tight_layout()
                plt.show()
                                        
            if fails > self.min_attempts: 
                print(f"Too many sync failures_1, {fails, self.sdr_pluto.rx_hardwaregain_chan0, self.sdr_pluto.tx_hardwaregain_chan0}")
                sys.exit(1)

            # check if the correlation is reasonable to assume sync is right, if not increase power and/or rx sensitivity
            if (corr >= self.corr_threshold):
                success = 1

            else:    
                fails+=1

                if self.sdr_pluto.tx_hardwaregain_chan0 <= -5:
                    self.sdr_pluto.tx_hardwaregain_chan0  = self.sdr_pluto.tx_hardwaregain_chan0 + 5
                    SDR_TX_GAIN = self.sdr_pluto.tx_hardwaregain_chan0                     
                elif self.sdr_pluto.rx_hardwaregain_chan0 <= 40:
                    self.sdr_pluto.tx_hardwaregain_chan0 = 0
                    self.sdr_pluto.rx_hardwaregain_chan0  = self.sdr_pluto.rx_hardwaregain_chan0 + 5
                    SDR_RX_GAIN = self.sdr_pluto.rx_hardwaregain_chan0
                else :
                    self.sdr_pluto.tx_hardwaregain_chan0 = 0
                    self.sdr_pluto.rx_hardwaregain_chan0  = 40
                    SDR_TX_GAIN = self.sdr_pluto.tx_hardwaregain_chan0
                    SDR_RX_GAIN = self.sdr_pluto.rx_hardwaregain_chan0
                
        self.sdr_pluto.tx_destroy_buffer() # shut the transmitter down
                
        try :
            out_shape[-1] = out_shape[-1]+add_td_samples
            out = tf.reshape(rx_TTI, out_shape)

        except:
            print("Something failed!")
            sys.exit(1)
 
        sdr_time=time.time()-now

        return out, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails + 1, corr, sdr_time
