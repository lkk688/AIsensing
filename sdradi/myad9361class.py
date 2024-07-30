#0604/2024 modified based on myad9361.py, add SDR class
#ref: https://ez.analog.com/ez-blogs/b/engineerzone-spotlight/posts/using-python-to-control-the-pluto-radio-and-plot-data
import time

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from timeit import default_timer as timer
import sys
from processing import createcomplexsinusoid, calculate_spectrum, normalize_complexsignal, detect_signaloffset, plot_noisesignalPSD, plot_offsetdetection, detect_signaloffsetv2, check_corrcondition

def printSDRproperties(sdr):
    print("Bandwidth of TX path:", sdr.tx_rf_bandwidth) #Bandwidth of front-end analog filter of TX path
    print("Loopback:", sdr.loopback) #0: disable, 1: Digital TX → Digital RX, 2: RF RX → RF TX
    print("TX gain chan0:", sdr.tx_hardwaregain_chan0) #Attenuation applied to TX path, -30
    print("TX gain chan1:", sdr.tx_hardwaregain_chan1) #Attenuation applied to TX path, -10
    print("TX enabled channels:", sdr.tx_enabled_channels) 
    print("RX enabled channels:", sdr.rx_enabled_channels)
    print("RX buffer size:", sdr.rx_buffer_size) #1024
    print(f'TX Cyclic Buffer: {sdr.tx_cyclic_buffer}')

    print("TX LO: %s" % (sdr.tx_lo)) #Carrier frequency of TX path
    print("RX LO: %s" % (sdr.rx_lo)) #Carrier frequency of RX path
    print("AGC Mode chan0:", sdr.gain_control_mode_chan0)
    print("AGC Mode chan1:", sdr.gain_control_mode_chan1)#Mode of receive path AGC. Options are: slow_attack, fast_attack, manual
    print("RX gain chan0:", sdr.rx_hardwaregain_chan0)#71
    print("RX gain chan1:", sdr.rx_hardwaregain_chan1)#71 Gain applied to RX path. Only applicable when gain_control_mode is set to ‘manual’
    print("rx_rf_bandwidth:", sdr.rx_rf_bandwidth) #4Mhz Bandwidth of front-end analog filter of RX path
    print("TX gain chan0:", sdr.tx_hardwaregain_chan0)
    print("TX gain chan1:", sdr.tx_hardwaregain_chan1)#71
    print("rx_rf_bandwidth:", sdr.tx_rf_bandwidth)
    print("Sample rate:", sdr.sample_rate) #Sample rate RX and TX paths in samples per second
    print("DDS scales:", sdr.dds_scales)
    print(f'Filter: {sdr.filter}')

def testlibiioaccess(urladdress="ip:pluto.local"):
    import iio
    sdr = adi.Pluto(urladdress)
    ctx = iio.Context(urladdress)

    for d1, d2 in zip(sdr.ctx.devices, ctx.devices):
        print(d1.name, "|", d2.name)
    
    phy = sdr.ctx.find_device("ad9361-phy")
    # View current mode
    print(phy.attrs["ensm_mode"].value) #fdd
    # View options
    print(phy.attrs["ensm_mode_available"].value) #sleep wait alert fdd pinctrl pinctrl_fdd_indep

def readiio(sdr):
    phy = sdr.ctx.find_device("ad9361-phy")
    # Read product ID register
    pi = phy.reg_read(0x37)
    #print(f"ID: {hex(pi)}")
    r = 0x80000088
    status = phy.reg_read(r)
    if status & 0b0100:
        print("Overflow")

#For FPGA based systems ADI reference designs include direct digital synthesizers (DDS) which can generate tones with arbitrary phase, frequency, and amplitude. 
# For each individual DAC channel there are two DDSs which can have a unique phase, frequency, and phase.
#ref: https://pyadi-iio.readthedocs.io/en/stable/fpga/index.html
def ddstone(sdr, dualtune=True, dds_freq_hz = 10000, dds_scale = 0.9):
    n = len(sdr.dds_scales)
    # Generate a single complex tone
    #dds_freq_hz = 10000 #must be less than 1/2 the sample rate.
    #dds_scale = 0.9 #range [0,1]
    # Enable all DDSs
    dds_freq_hz = int(dds_freq_hz)
    if dualtune == False:
        #option1:
        sdr.dds_single_tone(dds_freq_hz, dds_scale)
        #option2:
        # sdr.dds_enabled = [True] * n
        # # Set all DDSs to same frequency, scale, and phase
        # sdr.dds_phases = [0] * n
        # sdr.dds_frequencies = [dds_freq_hz] * n
        # sdr.dds_scales = [dds_scale] * n
    else:
        #allows for two complex tones to be generated per complex channel.
        frequency1 = dds_freq_hz
        frequency2 = dds_freq_hz *2
        sdr.dds_dual_tone(frequency1, dds_scale, frequency2, dds_scale, channel=0)


#modified based on https://github.com/lkk688/AIsensing/blob/main/deeplearning/SDR.py
class SDR:
    def __init__(self, SDR_IP, device='ad9361', SDR_FC=2000000000, SDR_SAMPLERATE=1e6, SDR_BANDWIDTH=1e6, \
                    Rx_CHANNEL=2, Tx_CHANNEL=1):
        #device='ad9361', 'ad9364', 'Pluto'
    
        self.SDR_IP = SDR_IP # IP address of the TX SDR device
        self.SDR_TX_FREQ = int(SDR_FC) # TX center frequency in Hz,  #2Ghz 2000000000
        self.SDR_RX_FREQ = int(SDR_FC) # RX center frequency in Hz
        #self.SDR_TX_GAIN = int(SDR_TX_GAIN) # TX gain in dB
        #self.SDR_RX_GAIN = int(SDR_RX_GAIN) # RX gain in dB
        self.SDR_SAMPLERATE = int(SDR_SAMPLERATE) # TX sample rate (samples/second)
        self.SDR_TX_BANDWIDTH = int(SDR_BANDWIDTH) # TX bandwidth (Hz)
        self.SDR_RX_BANDWIDTH = int(SDR_BANDWIDTH) # RX bandwidth (Hz)
        self.num_samples=int(SDR_SAMPLERATE/10) #default save 0.1s data
        self.sdr = self.setupSDR(fs=SDR_SAMPLERATE, device=device, Rx_CHANNEL=Rx_CHANNEL, Tx_CHANNEL=Tx_CHANNEL)

    #new added
    def setupSDR(self, fs= 6000000, device='ad9361', Rx_CHANNEL=2, Tx_CHANNEL=1): #default fs=6Mhz
        # Initialize the SDR device using the Analog Devices driver
        if device=='ad9361':
            sdr = adi.ad9361(uri=self.SDR_IP)
        elif device=='ad9364':
            sdr = adi.ad9364(self.SDR_IP)
        elif device.lower()=='pluto':
            sdr = adi.Pluto(uri=self.SDR_IP)
        else:
            print('device not supported')
            sdr = None
        
        # Configure the sample rate for both TX and RX
        sdr.sample_rate = fs
        # sdr.rx_lo = 2000000000 #2Ghz 2000000000
        # sdr.tx_lo = 2000000000
        # sdr.tx_cyclic_buffer = True
        # sdr.tx_hardwaregain_chan0 = -30
        #sdr.gain_control_mode_chan0 = "slow_attack" #'manual'
        # num_samps = 1024*100#10000 # number of samples returned per call to rx()
        # sdr.rx_buffer_size = num_samps

        # Configuration data channels
        if Rx_CHANNEL==2:
            sdr.rx_enabled_channels = [0,1] #enable two rx channel, # enable Rx1 (voltage0) and Rx2 (voltage1)
        elif Rx_CHANNEL==1:
            sdr.rx_enabled_channels = [0] #enables Rx0
        if Tx_CHANNEL==2:
            sdr.tx_enabled_channels = [0,1] #enables Tx0, tx1
        elif Tx_CHANNEL==1:
            sdr.tx_enabled_channels = [0] #enables Tx0
        self.Rx_CHANNEL = Rx_CHANNEL
        self.Tx_CHANNEL = Tx_CHANNEL
        return sdr
        

    def SDR_TX_setup(self, cyclic_buffer=True, tx_bandwidth=None, tx1_gain = None, tx2_gain = None):
        '''
        Initialize and start the SDR transmitter.

        This method sets up the SDR with the specified configuration parameters. It includes:
        - (removed)Initializing the SDR device using the ad9364 driver and the provided IP address.
        - Destroying any existing TX buffers to prepare for new transmissions.
        - Setting the local oscillator (LO) frequencies for both TX and RX.
        - Configuring the gain control mode for the TX channel to manual.
        - Setting the sample rate, RF bandwidth for both TX and RX, and hardware gains.

        The method ensures that the SDR transmitter is ready to send data with the configured settings.
        '''
        # Initialize the SDR device using the Analog Devices driver
        #self.sdr_tx = self.sdr

        # Clear any existing transmit buffers
        self.sdr.tx_destroy_buffer()

        # Set the local oscillator (LO) frequencies for transmit and receive
        self.sdr.tx_lo = self.SDR_TX_FREQ  # Transmit frequency
        #self.sdr.rx_lo = self.SDR_TX_FREQ  # Receive frequency (set to the same as TX)

        if cyclic_buffer:
            self.sdr.tx_cyclic_buffer = True

        # Configure the sample rate for both TX and RX
        #self.sdr.sample_rate = self.SDR_TX_SAMPLERATE

        # Set the RF bandwidth for both TX and RX
        if tx_bandwidth is not None:
            self.sdr.tx_rf_bandwidth = tx_bandwidth
            self.SDR_TX_BANDWIDTH = tx_bandwidth
        else:
            self.sdr.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH
        #self.sdr.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH

        # Set the hardware gain for both TX and RX
        if tx1_gain is not None:
            self.sdr.tx_hardwaregain_chan0 = tx1_gain  # TX gain
        if self.Tx_CHANNEL==2 and tx2_gain is not None:
            self.sdr.tx_hardwaregain_chan1 = tx2_gain
        #self.sdr.rx_hardwaregain_chan0 = self.SDR_RX_GAIN  # RX gain -30

        # Clear transmit buffer again to ensure a clean start
        self.sdr.tx_destroy_buffer()

    def SDR_gain_set(self, tx_gain, rx_gain):
        '''
        Set the TX and RX gain
        '''
        self.sdr.tx_hardwaregain_chan0 = tx_gain
        if self.Tx_CHANNEL==2:
            self.sdr.tx_hardwaregain_chan1 = tx_gain
        self.sdr.rx_hardwaregain_chan0 = rx_gain
        if self.Rx_CHANNEL==2:
            self.sdr.rx_hardwaregain_chan1 = rx_gain


    def SDR_TX_stop(self):
        '''
        Stop the SDR TX
        '''
        self.sdr.tx_destroy_buffer() 
        self.sdr.rx_destroy_buffer()

    def SDR_RX_setup(self, n_SAMPLES=None, controlmode='manual', sample_rate=None, rx_bandwidth=None, rx1_gain = None, rx2_gain = None):
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
        '''
        # Determine the number of samples to receive
        if n_SAMPLES is None:
            n_SAMPLES = self.num_samples * 4  # Default to four times the number of transmitted samples
        if n_SAMPLES <= 0:
            n_SAMPLES = 1  # Ensure at least one sample is received
        
        # Set the buffer size for receiving data
        self.sdr.rx_buffer_size = n_SAMPLES

        self.sdr.rx_lo = self.SDR_RX_FREQ #SDR_TX_FREQ  # Receive frequency (set to the same as TX)
        # Set gain control mode to manual for the RX channel
        self.sdr.gain_control_mode_chan0 = controlmode #"slow_attack" #'manual'
        if self.Rx_CHANNEL ==2:
            self.sdr.gain_control_mode_chan1 = controlmode
        
        # Set the hardware gain for both TX and RX
        if rx_bandwidth is not None:
            self.sdr.rx_rf_bandwidth = rx_bandwidth
            self.SDR_RX_BANDWIDTH = rx_bandwidth
        else:
            self.sdr.rx_rf_bandwidth = self.SDR_RX_BANDWIDTH # rx filter cutoff 

        if sample_rate is not None:
        # update the sample rate for both TX and RX
            self.sdr.sample_rate = sample_rate
            self.SDR_SAMPLERATE = sample_rate

        if rx1_gain is not None:
            self.sdr.rx_hardwaregain_chan0 = int(rx1_gain)
        if rx2_gain is not None and self.Rx_CHANNEL ==2:
            self.sdr.rx_hardwaregain_chan1 = int(rx2_gain)
        # Clear the receiver buffer to prepare for new data
        self.sdr.rx_destroy_buffer()# clear any data from rx buffer

    def SDR_RX_receive(self, combinerule='drop', normalize=True, remove_dc=True):
        # Receive the samples from the SDR hardware
        x = self.sdr.rx()

        if self.Rx_CHANNEL==2:
            data0=x[0]
            data1=x[1]
            if combinerule == 'drop':
                a = data0
            elif combinerule == 'plus':
                a = data0+data1
        else:
            a=x

        # Normalize the signal amplitude if required
        if normalize:
            #a = a / np.max(np.abs(a))
            a = normalize_complexsignal(a, remove_dc=remove_dc, max_scale=1)

        # Convert the received samples to a PyTorch tensor
        #return torch.tensor(a, dtype=torch.complex64)
        return a
    
    def SDR_RX_receive_continuous(self, T_len = 2, spectrum=False, delay=0.5, plot_flag = False):
        #T_len = 2  #2second
        # Collect data
        #alldata0 = np.empty(0) #Default is numpy.float64.
        alldata0 = np.empty(0, dtype=np.complex_) #Default is numpy.float64.
        rxtime=[]
        processtime=[]
        
        fs = self.SDR_SAMPLERATE
        num_samps = self.sdr.rx_buffer_size
        Nperiod=int(T_len*fs/num_samps) #total time 10s *fs=total samples /fft_size = Number of frames

        if plot_flag:
            #plt.figure(figsize=(10,6))
            fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(12,6))
        
        for r in range(Nperiod):
            start = timer()
            data0 = self.SDR_RX_receive()
            rxt = timer()
            timedelta=rxt-start
            rxtime.append(timedelta)
            
            datarate=len(data0.real)*4/timedelta/1e6 #Mbps, complex data is 4bytes
            print("Data rate at ", datarate, "Mbps.") #7-8Mbps in 10240 points, 10Mbps in 102400points, single channel in 19-20Mbps
            alldata0 = np.concatenate((alldata0, data0))
            if spectrum:
                f, Pxx_den, peak_freq = calculate_spectrum(data0, fs, find_peak=True)
            

            if plot_flag and spectrum:
                Npoints=len(data0.real) #min(len(data1.real), len(data0.real))
                ts = 1/fs
                t = (ts*1000)*np.arange(Npoints) #second to ms
                #t = np.arange(0, N, ts)
                #plt.clf()
                updatefigure(axs, t, data0[0:Npoints], data1=None, f=f, Pxx_den=Pxx_den)
                #plt.draw()
                #plt.show()
                plt.pause(0.01)
                time.sleep(delay)
            endtime = timer()
            processtime.append(endtime-start)
        return alldata0, processtime


    def SDR_TX_send(self, SAMPLES, SAMPLES2=None, max_scale=1, normalize=False, leadingzeros=0, cyclic=False, scale4sdr=True):
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

        self.sdr.tx_destroy_buffer()  # Clear any existing buffer

        # Determine the number of samples based on whether SAMPLES is a NumPy array or a PyTorch tensor
        if isinstance(SAMPLES, np.ndarray):
            self.num_samples = SAMPLES.size
        # elif isinstance(SAMPLES, torch.Tensor):
        #     self.num_samples = SAMPLES.numel()
        #     SAMPLES = SAMPLES.numpy()  # Convert to NumPy array if it's a PyTorch tensor
        elif isinstance(SAMPLES, list):
            self.num_samples = len(SAMPLES) #"Input data is a Python list."
            SAMPLES = np.array(SAMPLES)
            if SAMPLES2 is not None:
                SAMPLES2=np.arrya(SAMPLES2)
        else:
            print("Input data is neither a NumPy array nor a Python list.")

        # # Normalize the signal to remove DC offset
        # samples = SAMPLES - np.mean(SAMPLES)

        # # Scale the samples to their maximum amplitude and adjust according to max_scale
        # samples = (samples / np.max(np.abs(samples))) * max_scale

        # # Scale the signal to the dynamic range expected by the SDR hardware
        # samples *= 2**14  # scale the samples to 16-bit PlutoSDR, for example, expects sample values in the range -2^14 to +2^14
        if normalize:
            SAMPLES = normalize_complexsignal(SAMPLES, max_scale=max_scale)
            if SAMPLES2 is not None:
                SAMPLES2 = normalize_complexsignal(SAMPLES2, max_scale=max_scale)
        if scale4sdr:
            # Scale the signal to the dynamic range expected by the SDR hardware
            SAMPLES *= 2**14  # scale the samples to 16-bit PlutoSDR, for example, expects sample values in the range -2^14 to +2^14
            if SAMPLES2 is not None:
                SAMPLES2 *= 2**14

        if leadingzeros >0:
            leading_zeroes = np.zeros(leadingzeros, dtype=np.complex64)  # Leading 500 zeroes for noise floor measurement
            SAMPLES = np.concatenate([leading_zeroes, SAMPLES], axis=0)  # Add the quiet for noise measurements
            if SAMPLES2 is not None:
                SAMPLES2 = np.concatenate([leading_zeroes, SAMPLES2], axis=0)
        # Set cyclic buffer mode if required
        self.sdr.tx_cyclic_buffer = cyclic

        # Transmit the prepared samples
        if self.Tx_CHANNEL ==1:
            self.sdr.tx(SAMPLES)
        elif self.Tx_CHANNEL ==2 and SAMPLES2 is not None:
            self.sdr.tx([SAMPLES, SAMPLES2])
        else:
            self.sdr.tx([SAMPLES, SAMPLES])

    def SDR_TX_signalgen(self, signal_type='sinusoid', f_signal=int(1000000), N=1024, leadingzeros=0, cyclic=True):
        self.sdr.tx_destroy_buffer()  # Clear any existing buffer

        #get SDR sample rate
        fs = int(self.sdr.sample_rate) #6MHz
    
        if signal_type == 'sinusoid': # Create a sinewave waveform
            tx_samples = createcomplexsinusoid(fs, f_signal, N)
            if leadingzeros >0:
                leading_zeroes = np.zeros(leadingzeros, dtype=np.complex64)  # Leading 500 zeroes for noise floor measurement
                tx_samples = np.concatenate([leading_zeroes, tx_samples], axis=0)  # Add the quiet for noise measurements
            # Send data
            # Since sdr.tx_cyclic_buffer was set to True, this data will just keep repeating.  There’s no need to send it again.   
            # Set cyclic buffer mode if required
            self.sdr.tx_cyclic_buffer = cyclic
            self.sdr.tx(tx_samples)
        elif signal_type == 'dds':
            ddstone(self.sdr, dualtune=False, dds_freq_hz = f_signal, dds_scale = 0.9)
    
    def SDR_RXTX_offset(self, SAMPLES, leadingzeros=500, add_td_samples=16, tx_gain=-10, rx_gain=10, make_plot=True):
        #add_td_samples: number of additional symbols to cater fordelay spread

        out_shape = list(SAMPLES.shape) # store the input tensor shape, [80]
        num_samples = SAMPLES.shape[-1] # number of samples in the input 80
        SAMPLES = SAMPLES.flatten()

        #x_sdr = mysdr(SAMPLES = x_time, SDR_TX_GAIN=-10, SDR_RX_GAIN = 10, add_td_samples = 16, debug=True) # transmit
        self.SDR_TX_stop()
        self.SDR_TX_setup(cyclic_buffer=True, tx1_gain=tx_gain, tx2_gain=tx_gain)
        self.SDR_RX_setup(n_SAMPLES=(num_samples+leadingzeros)*3, controlmode='manual', rx1_gain=rx_gain, rx2_gain=rx_gain) ## set the RX buffer size to 3 times the number of samples
        #self.SDR_gain_set(tx_gain=0, rx_gain=30)
        time.sleep(0.3)  # Wait for settings to take effect

        now = time.time() # for measuing the duration of the process
        self.SDR_TX_send(SAMPLES, leadingzeros=leadingzeros, cyclic=True)
        # internal counters
        corr_threshold = 0.3
        fails = 0 # how many times the process failed to reach pearson r > self.corr_threshold
        success = 0 #  how many times the process reached pearson r > self.corr_threshold
        timeout = 5
        #rx_samples = np.ones(SAMPLES.shape[0] + add_td_samples, dtype=np.complex64)
        while success == 0 and fails < timeout:       
            # RX samples 
            rx_samples = self.SDR_RX_receive(combinerule='drop', normalize=False)

            #the use of tx_SAMPLES is for adjust_stdev and perform correlation
            #rx_samples_normalized, rx_TTI, rx_noise, TTI_offset, TTI_corr, corr, SINR = detect_signaloffset(rx_samples, tx_samples=SAMPLES, num_samples=num_samples, leadingzeros=leadingzeros, add_td_samples=add_td_samples)

            rx_samples_normalized, rx_TTI, rx_noise, TTI_offset, TTI_corr, corr, SINR = detect_signaloffsetv2(rx_samples, tx_samples=SAMPLES, num_samples=num_samples, leadingzeros=leadingzeros, add_td_samples=add_td_samples)

            if make_plot:
                plot_noisesignalPSD(rx_samples, rx_samples_normalized, tx_SAMPLES=SAMPLES, \
                                    rx_TTI=rx_TTI, rx_noise=rx_noise, TTI_offset=TTI_offset, TTI_corr=TTI_corr, corr=corr, SINR=SINR)
                plot_offsetdetection(tx_samples=SAMPLES, all_rx_samples=rx_samples_normalized, onetti_rx_samples=rx_TTI, rx_noise=rx_noise, TTI_offset=TTI_offset, TTI_correlation=TTI_corr, save=False, savefolder = 'output', save_path_prefix = "offset")

            if fails > timeout:
                print("Too many errors, timeout")
                sys.exit(1)
            # check if the correlation is reasonable to assume sync is right, if not increase power and/or rx sensitivity
            # if (corr >= corr_threshold):
            #     success = 1
            # else: 
            #     fails=fails+1
            success, needmorepower = check_corrcondition(corr, SINR, corr_threshold, minSINR=5, maxSINR=30)
            if success != 1:
                fails=fails+1
        
        SDR_TX_GAIN = self.sdr.tx_hardwaregain_chan0
        SDR_RX_GAIN = self.sdr.rx_hardwaregain_chan0
        self.SDR_TX_stop()
        try:
            out_shape[-1] += add_td_samples
            out = np.reshape(rx_TTI, out_shape) #(1892,)
        except Exception as e:
            print("Something failed:", e)
            sys.exit(1)
        sdr_time=time.time()-now
        return out, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails+1, corr, sdr_time
    
    # Read back properties from hardware https://analogdevicesinc.github.io/pyadi-iio/devices/adi.ad936x.html
    def show_params(self):
        sdr = self.sdr
        printSDRproperties(sdr)

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
        self.SDR_TX_setup()
        self.SDR_gain_set(tx_gain=-80, rx_gain=SDR_RX_GAIN)
        self.SDR_RX_setup(n_SAMPLES=10000)
        time.sleep(0.3)  # Wait for settings to take effect
        noise_sample = self.SDR_RX_receive(normalize=False)
        #noise_power = torch.mean(torch.abs(noise_sample)**2)
        noise_power = np.mean(np.abs(noise_sample)**2)
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
            sample = self.SDR_RX_receive(normalize=False)
            signal_power = np.mean(np.abs(noise_sample)**2) #torch.mean(torch.abs(sample)**2)
            #SINR.append(10 * torch.log10(signal_power / noise_power).item())
            SINR.append(10 * np.log10(signal_power / noise_power))

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
            plt.title(f'SINR vs TX Gain. RX Gain set to {SDR_RX_GAIN}\nFreq={self.SDR_TX_FREQ/1e6} MHz, BW={self.SDR_TX_BANDWIDTH/1e6} MHz, SR={self.SDR_SAMPLERATE/1e6} MHz')
            plt.grid()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('PlutoSDR TX Gain')
            plt.ylabel('Measured SINR')
            plt.tight_layout()
            plt.show()
            plt.savefig('pics/SINR_vs_TX_Gain.png')

        return SDR_TX_GAIN

    
def sdr_test(urladdress, signal_type='sinusoid', Rx_CHANNEL=2, plot_flag = True):
    # Create radio
    sdr = adi.ad9361(uri=urladdress)
    
    # Configure properties
    fs= 6000000 #6MHz
    sdr.rx_rf_bandwidth = 4000000 #4MHz
    sdr.sample_rate = fs
    sdr.rx_lo = 2000000000 #2Ghz 2000000000
    sdr.tx_lo = 2000000000
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = -30
    sdr.gain_control_mode_chan0 = "slow_attack" #'manual'

    num_samps = 1024*100#10000 # number of samples returned per call to rx()
    sdr.rx_buffer_size = num_samps

    # Configuration data channels
    if Rx_CHANNEL==2:
        sdr.rx_enabled_channels = [0,1] #enable two rx channel
    else:
        sdr.rx_enabled_channels = [0] #enables Rx0
    sdr.tx_enabled_channels = [0] #enables Tx0

    # Read properties
    print("RX LO %s" % (sdr.rx_lo)) #2Ghz
    printSDRproperties(sdr)


    # Create a sinewave waveform
    fs = int(sdr.sample_rate) #6MHz
    ts = 1/float(fs)
    N = 1024
    fc = int(1000000) #int(1000000 / (fs / N)) * (fs / N) #996093~1MHz
    if signal_type == 'sinusoid':
        iq = createcomplexsinusoid(fs, fc, N)
        # Send data
        # Since sdr.tx_cyclic_buffer was set to True, this data will just keep repeating.  There’s no need to send it again.   
        sdr.tx(iq)
    elif signal_type == 'dds':
        ddstone(sdr, dualtune=False, dds_freq_hz = fc, dds_scale = 0.9)

    if plot_flag:
        #plt.figure(figsize=(10,6))
        fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(12,6))
    # Collect data
    #alldata0 = np.empty(0) #Default is numpy.float64.
    alldata0 = np.empty(0, dtype=np.complex_) #Default is numpy.float64.
    rxtime=[]
    processtime=[]
    T_len = 2  #2second
    Nperiod=int(T_len*fs/num_samps) #total time 10s *fs=total samples /fft_size = Number of frames
    print("Total period for 2s:", Nperiod)
    for r in range(Nperiod):
        start = timer()
        x = sdr.rx() #1024 size array of complex
        rxt = timer()
        timedelta=rxt-start
        rxtime.append(timedelta)
        if Rx_CHANNEL==2:
            data0=x[0]
            data1=x[1]
        else:
            data0=x
        datarate=len(data0.real)*4/timedelta/1e6 #Mbps, complex data is 4bytes
        print("Data rate at ", datarate, "Mbps.") #7-8Mbps in 10240 points, 10Mbps in 102400points, single channel in 19-20Mbps
        alldata0 = np.concatenate((alldata0, data0))
        f, Pxx_den = signal.periodogram(data0.real, fs) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
        #returns f (ndarray): Array of sample frequencies.
        #returns Pxx_den (ndarray): Power spectral density or power spectrum of x.
        f /= 1e6  # Hz -> MHz
        peak_index = np.argmax(Pxx_den) #(71680,) -> 22526
        peak_freq = f[peak_index]
        print("Peak frequency found at ", peak_freq, "MHz.")

        readiio(sdr)

        if plot_flag:
            Npoints=min(len(data1.real), len(data0.real))
            t = (ts*1000)*np.arange(Npoints) #second to ms
            #t = np.arange(0, N, ts)
            #plt.clf()
            updatefigure(axs, t, data0[0:Npoints], data1[0:Npoints], f, Pxx_den)
            #plt.draw()
            #plt.show()
            plt.pause(0.01)
            time.sleep(0.2)
        endtime = timer()
        processtime.append(endtime-start)
        

    # Stop transmitting
    sdr.tx_destroy_buffer() #Clears TX buffer
    sdr.rx_destroy_buffer() #Clears RX buffer
    print(len(alldata0))
    with open('./data/test9361data.npy', 'wb') as f:
        np.save(f, alldata0)
    plotfigure(ts, alldata0.real[0:num_samps*2])
    print(np.mean(rxtime))
    print(np.mean(processtime))

def test_SDRclass(urladdress, signal_type='dds'):
    fc=2.4*1e9 #2Ghz 2000000000
    fs = 6000000 #6MHz
    bandwidth = 4000000 #4MHz
    mysdr = SDR(SDR_IP=urladdress, SDR_FC=fc, SDR_SAMPLERATE=fs, SDR_BANDWIDTH=bandwidth)
    mysdr.SDR_TX_stop()
    mysdr.SDR_TX_setup(tx1_gain=-10, tx2_gain=-10)
    mysdr.SDR_RX_setup(n_SAMPLES=10000, rx1_gain=10, rx2_gain=10)
    
    time.sleep(0.3)  # Wait for settings to take effect

    f_signal = 2000000 #1MHz
    N_len = 10000 #data length, only needed for sin
    mysdr.SDR_TX_signalgen(signal_type=signal_type, f_signal=f_signal, N=N_len, cyclic=True)
    rx_sample = mysdr.SDR_RX_receive(normalize=False)
    num_samps = mysdr.sdr.rx_buffer_size

    alldata0, processtime = mysdr.SDR_RX_receive_continuous(T_len = 0.2, spectrum=False, delay=0.5, plot_flag = False)
    # Stop transmitting
    mysdr.SDR_TX_stop()
    with open('./data/test9361data.npy', 'wb') as f:
        np.save(f, alldata0)
    plotfigure(1/fs, alldata0.real[0:num_samps*2])
    #print(np.mean(rxtime))
    print(np.mean(processtime))

from myadi.tddn import tddn
def test_SDRTDD(urladdress="ip:pluto.local", signal_type='dds'):
    fc=2.4*1e9 #2Ghz 2000000000
    fs = 6000000 #6MHz
    bandwidth = 4000000 #4MHz
    mysdr = SDR(SDR_IP=urladdress, device='Pluto', SDR_FC=fc, SDR_SAMPLERATE=fs, SDR_BANDWIDTH=bandwidth, Rx_CHANNEL=1, Tx_CHANNEL=1)
    mysdr.SDR_TX_stop()
    #mysdr.SDR_TX_setup()
    #mysdr.SDR_RX_setup(n_SAMPLES=10000)

    # Configure TDD properties
    tdd = tddn(urladdress)#"ip:pluto.local")
    tdd.frame_length_ms = 4         # each GPIO toggle is spaced 4ms apart
    tdd.startup_delay_ms = 0        # do not set a startup delay 
    tdd.burst_count = 3             # there is a burst of 3 toggles, then off for a long time
    tdd.out_channel0_on_ms = 0.5    # each GPIO pulse will be 100us (0.6ms - 0.5ms).  And the first trigger will happen 0.5ms into the buffer
    tdd.out_channel0_off_ms = 0.6
    tdd.out_channel0_enable = True  # Enable CH0 output
    tdd.sync_external = True
    tdd.enable = True
    
    time.sleep(0.3)  # Wait for settings to take effect


def main():
    args = parser.parse_args()
    urladdress = args.urladdress #"ip:pluto.local"
    Rx_CHANNEL = args.rxch
    signal_type = args.signal
    plot_flag = args.plot

    #testlibiioaccess(urladdress)
    #sdr_test(urladdress, signal_type=signal_type, Rx_CHANNEL=Rx_CHANNEL, plot_flag = plot_flag)

    test_SDRclass(urladdress)
    #test_SDRTDD(urladdress)
    fs=1000000
    #test_ofdm_SDR(urladdress=urladdress, SampleRate=fs)
    #test_ofdmmimo_SDR(urladdress=urladdress)

    


# def animationfig(axs, data0, data1):
#     line1=axs[0].plot(data0.real, marker="o", ms=2, color="red")  # Only plot real part
#     line2=axs[0].plot(data1.real, marker="o", ms=2, color="blue")
    
#     def update(data0, data1):
#         # for each frame, update the data stored on each artist.
#         #scat.set_offsets(data)
#         # update the line plot:
#         #line2.set_xdata(t[:frame])
#         line1.set_ydata(data0.real)
#         line2.set_ydata(data1.real)
#         return (line1, line2)
def plotfigure(ts, data0):
    f, Pxx_den = signal.periodogram(data0, int(1/ts))
    Npoints=len(data0)
    t = (ts*1000)*np.arange(Npoints) #second to ms
    fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(12,6))
    axs[0].cla()  
    axs[0].plot(t, data0.real, marker="o", ms=2, color="red")  # Only plot real part
    #axs[0].plot(t, data1.real, marker="o", ms=2, color="blue")
    #axs[0].set_xlim(0, 2)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('data0')
    axs[0].set_title("Time Domain Dual Ch Data")
    axs[0].grid(True)
    axs[1].cla()  
    axs[1].semilogy(f/1e6, Pxx_den)
    axs[1].set_ylim([1e-8, 1e2])
    axs[1].set_xlabel("frequency [MHz]") #-3e^6 3e^6
    axs[1].set_ylabel("PSD [V**2/Hz]")
    axs[1].set_title("Spectrum")
    plt.show()

def updatefigure(axs, t, data0, data1, specf,specp):
    #axs[0].plot(t, data0.real, t, data1.real)
    axs[0].cla()  
    axs[0].plot(t, data0.real, marker="o", ms=2, color="red")  # Only plot real part
    if data1 is not None:
        axs[0].plot(t, data1.real, marker="o", ms=2, color="blue")
    #axs[0].set_xlim(0, 2)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('data0 and data1')
    axs[0].set_title("Time Domain Dual Ch Data")
    axs[0].grid(True)
    axs[1].cla()  
    axs[1].semilogy(specf, specp)
    axs[1].set_ylim([1e-7, 1e2])
    axs[1].set_xlabel("frequency [Hz]") #-3e^6 3e^6
    axs[1].set_ylabel("PSD [V**2/Hz]")
    axs[1].set_title("Spectrum")

    # plt.subplot(2, 1, 1)
    # plt.title("Time Domain I/Q Data")
    # plt.plot(data0.real, marker="o", ms=2, color="red")  # Only plot real part
    # plt.plot(data1.real, marker="o", ms=2, color="blue")
    # plt.xlabel("Data Point")
    # plt.ylabel("ADC output")
    # plt.subplot(2, 1, 2)
    # plt.semilogy(specf, specp)
    # plt.ylim([1e-7, 1e2])
    # plt.xlabel("frequency [Hz]") #-3e^6 3e^6
    # plt.ylabel("PSD [V**2/Hz]")
    # plt.title("Spectrum, peak at " + str(freqs[peak_index]) + " MHz.")
    # plt.plot(freqs, ampl, marker="o", ms=2)
    # plt.xlabel("Frequency [MHz]")
    # plt.ylabel("Signal Strength")
    # plt.tight_layout()
    # plt.show()


# piuri="ip:phaser.local:50901"
# localuri="ip:analog.local"
# antsdruri="ip:192.168.1.10"#connected via Ethernet with static IP
# plutodruri="ip:192.168.2.1" "ip:192.168.2.16"#connected via USB
#PoE: "ip:192.168.1.67:50901"
import argparse
parser = argparse.ArgumentParser(description='MyAD9361')
parser.add_argument('--urladdress', default="ip:192.168.1.67:50901", type=str,
                    help='urladdress of the device, e.g., ip:pluto.local, ip:192.168.2.1') 
parser.add_argument('--rxch', default=1, type=int, 
                    help='number of rx channels')
parser.add_argument('--signal', default="dds", type=str,
                    help='signal type: sinusoid, dds')
parser.add_argument('--plot', default=False, type=bool,
                    help='plot figure')

if __name__ == '__main__':
    main()