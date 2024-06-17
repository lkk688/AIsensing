import numpy as np
from scipy import ndimage
from timeit import default_timer as timer
from scipy import signal
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 8.0

def createcomplexsinusoid(fs, fc = 3000000, N = 1024):
    # Create a complex sinusoid
    #fc = 3000000
    #N = 1024
    #fs = int(sdr.tx_sample_rate)
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq = i + 1j * q
    return iq


def calculate_spectrum(data0, fs, find_peak=True):
    f, Pxx_den = signal.periodogram(data0.real, fs) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
    #returns f (ndarray): Array of sample frequencies.
    #returns Pxx_den (ndarray): Power spectral density or power spectrum of x.
    peak_freq = 0
    if find_peak ==True:
        f /= 1e6  # Hz -> MHz
        peak_index = np.argmax(Pxx_den) #(71680,) -> 22526
        peak_freq = f[peak_index]
        print("Peak frequency found at ", peak_freq, "MHz.")
    return f, Pxx_den, peak_freq

def normalize_complexsignal(SAMPLES, max_scale=1, scale4sdr=True):
    # Determine the number of samples based on whether SAMPLES is a NumPy array or a PyTorch tensor
    if isinstance(SAMPLES, np.ndarray):
        num_samples = SAMPLES.size
    # elif isinstance(SAMPLES, torch.Tensor):
    #     self.num_samples = SAMPLES.numel()
    #     SAMPLES = SAMPLES.numpy()  # Convert to NumPy array if it's a PyTorch tensor
    elif isinstance(SAMPLES, list):
        num_samples = len(SAMPLES) #"Input data is a Python list."
        SAMPLES = np.array(SAMPLES)
    else:
        print("Input data is neither a NumPy array nor a Python list.")

    # Assuming SAMPLES is a NumPy array
    flat_samples = SAMPLES.flatten()  # Flatten the input samples
    tx_std = np.std(flat_samples)  # Standard deviation of the input samples
    tx_mean = np.mean(flat_samples)  # Mean of the input samples
    tx_samples = flat_samples - tx_mean  # Remove DC offset

    # Scale for SDR input
    tx_samples_abs = np.abs(tx_samples)  # Absolute values of the samples
    tx_samples_max = np.max(tx_samples_abs)  # Take the maximum value of the samples
    tx_samples_scaled = tx_samples / tx_samples_max  # Scale the tx_samples to max 1

    # Scale the samples to their maximum amplitude and adjust according to max_scale
    samples = tx_samples_scaled * max_scale

    if scale4sdr:
        # Scale the signal to the dynamic range expected by the SDR hardware
        samples *= 2**14  # scale the samples to 16-bit PlutoSDR, for example, expects sample values in the range -2^14 to +2^14

    print("Standard deviation:", tx_std)
    print("Mean:", tx_mean)
    print("Scaled samples (max 1):", tx_samples_scaled)
    return samples

def detect_signaloffset(rx_samples, tx_SAMPLES, num_samples, leadingzeros=500, add_td_samples=0, tx_std=None):
    #add_td_samples: number of additional symbols to cater fordelay spread
    flat_samples = tx_SAMPLES.flatten()  # Flatten the input samples
    tx_std = np.std(flat_samples)  # Standard deviation of the input samples

    # Assuming rx_samples is a NumPy array and other variables are defined
    rx_samples = rx_samples.astype(np.complex64)  # Convert received IQ samples to NumPy complex64

    # Remove any offset
    rx_mean = np.mean(rx_samples)
    rx_samples -= rx_mean

    # Set the same standard deviation as in the input samples
    rx_std = np.std(rx_samples)
    if tx_std is not None:
        std_multiplier = np.float16(tx_std / rx_std) * 0.9  # Calculate new multiplier for same stdev in TX and RX
        rx_samples *= std_multiplier  # Set the stdev
    rx_samples_normalized = rx_samples

    # Calculate the correlation between TX and RX signal
    #find the start symbol of the first full TTI with 500 samples of noise measurements in front
    #TTI_corr is a correlation signal obtained by cross-correlating the received samples (rx_samples) with the transmitted samples (flat_samples). 
    #The goal is to find the alignment (offset) between the two signals.
    TTI_corr = signal.correlate(rx_samples, flat_samples, mode='full', method='fft')
    #find the index of the max value of absolute values of the first half of the correlation signal. 
    #TTI_offset is initially set to the alignment position (index). we subtract the length of flat_samples and add 1 to get the correct offset.
    TTI_offset = np.argmax(np.abs(TTI_corr[0:int(len(rx_samples) / 2)])) - len(flat_samples) + 1
    if TTI_offset < leadingzeros + num_samples: #ensure that it points to the start of the first full Transmission Time Interval (TTI) with noise measurements in front.
        TTI_offset += leadingzeros + num_samples

    # RX TTI symbols + the additional symbols
    rx_TTI = rx_samples[TTI_offset:TTI_offset + num_samples + add_td_samples]

    # RX noise for SINR calculation
    guardsize=50
    rx_noise = rx_samples[TTI_offset - (leadingzeros-guardsize):TTI_offset - guardsize]

    # Calculate the Pearson correlation between complex samples_orig and rx_TTI as acceptance metric
    #extracts a portion of the received samples (rx_samples) starting from the TTI_offset and spanning num_samples elements.
    received=np.abs(rx_samples)[TTI_offset:TTI_offset + num_samples]
    #np.corrcoef(...) computes the Pearson correlation coefficient between the two sets of absolute values 
    #The result of np.corrcoef(...) is a 2x2 matrix. The value at position [0, 1] (or equivalently, [1, 0]) represents the correlation coefficient between the two sets of data.
    corr = np.corrcoef(np.abs(flat_samples), received)[0, 1]
    print("Corr:", corr)

    # Calculate TX power, RX power & noise power
    tx_TTI_p = np.var(flat_samples)  # TX power
    noise_p = np.var(rx_noise)  # Noise power
    rx_TTI_p = np.var(rx_TTI)  # RX signal power
    SINR = 10 * np.log10(rx_TTI_p / noise_p)  # Calculate SINR from received powers
    resulttext = f'SINR ={SINR:1.1f}, TTI start index = {TTI_offset}, correlation = {corr:1.2f}, TX_p/RX_p = {tx_TTI_p/rx_TTI_p:1.2f}'
    print(resulttext)
    return rx_samples_normalized, rx_TTI, rx_noise, TTI_offset, TTI_corr, corr, SINR

def plot_noisesignalPSD(rx_samples, rx_samples_normalized, tx_SAMPLES, rx_TTI, rx_noise, TTI_offset, TTI_corr, corr, SINR):
    #titletext = f'SINR ={SINR:1.1f}, attempt={fails+1}, TTI start index = {TTI_offset}, correlation = {corr:1.2f}, TX_p/RX_p = {tx_TTI_p/rx_TTI_p:1.2f}'
    tx_flat_samples = tx_SAMPLES.flatten()  # Flatten the input samples
    titletext = f'SINR ={SINR:1.1f}, TTI start index = {TTI_offset}, correlation = {corr:1.2f}'
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(16, 7)
    fig.suptitle(titletext)
    axs[0,0].plot(10*np.log10(abs(rx_samples)/max(abs(rx_samples))), label='RX_dB')
    axs[0,0].legend()
    axs[0,0].set_title('TTI received 3 times, starting at random time')
    axs[0,1].plot((abs(rx_samples_normalized)), label='abs(RXsample)')
    axs[0,1].axvline(x=TTI_offset, c='r', lw=3, label='TTI start')
    axs[0,1].plot(abs(abs(TTI_corr)/np.max(abs(TTI_corr))), label='Pearson R')
    axs[0,1].legend()
    axs[0,1].set_title('Correlator for syncing the start of the second received TTI')
    
    axs[1,0].plot(np.abs(tx_flat_samples), label='abs(TX samples)')
    #axs[1,0].set_ylim(0,tx_samples_max_sample)
    axs[1,0].legend()
    axs[1,0].set_title('Transmitted signal, one TTI')
    axs[1,1].plot((abs(rx_TTI)), label='abs(RX samples)')
    #axs[1,1].set_ylim(0,tx_samples_max_sample)
    axs[1,1].legend()
    axs[1,1].set_title('Received signal, one TTI, syncronized')
    
    axs[2,0].psd(tx_flat_samples, label='TX Signal')
    axs[2,0].legend()
    axs[2,0].set_title('Transmitted signal PSD')
    axs[2,1].psd(rx_TTI, label='RX signal')
    axs[2,1].psd(rx_noise, label='noise')
    axs[2,1].legend()
    axs[2,1].set_title('Received noise PSD and signal PSD')
    plt.tight_layout()
    plt.show()

def extenddata(data, zoom=(20,5)):
    resampled_data=ndimage.zoom(data, zoom=(20,5))
    return resampled_data

def select_chirp(sum_data, num_chirps, good_ramp_samples, start_offset_samples, num_samples_frame, fft_size):
    # select just the linear portion of the last chirp
    rx_bursts = np.zeros((num_chirps, good_ramp_samples), dtype=complex)
    for burst in range(num_chirps):
        start_index = start_offset_samples + burst*num_samples_frame
        stop_index = start_index + good_ramp_samples
        rx_bursts[burst] = sum_data[start_index:stop_index]
        burst_data = np.ones(fft_size, dtype=complex)*1e-10
        #win_funct = np.blackman(len(rx_bursts[burst]))
        win_funct = np.ones(len(rx_bursts[burst]))
        burst_data[start_offset_samples:(start_offset_samples+good_ramp_samples)] = rx_bursts[burst]*win_funct
    return burst_data, win_funct

def get_spectrum(data, fft_size, win_funct=None):
    if win_funct is None:
        win_funct = np.blackman(len(data))
        y = data * win_funct
    else:
        y = data
    data_fft = np.fft.fft(y, n=fft_size)
    sp = np.absolute(data_fft)
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
    return s_dbfs

def showspectrum(data, fs):
    # fc = int(100e3 / (fs / N_frame)) * (fs / N_frame) #300KHz
    # ts =1.0/fs
    # t = np.arange(0, N_frame * ts, ts)
    # i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    # q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    # iq_100k = 1 * (i + 1j * q)
    #data=data[0:N_frame] #* iq_100k
    N_frame = len(data)
    win_funct = np.blackman(len(data))
    y = data * win_funct
    sp = np.absolute(np.fft.fft(y))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))

    """there's a scaling issue on the y-axis of the waterfallcthe data is off by 300kHz.  To fix, I'm just shifting the freq"""
    fc = int(300e3 / (fs / N_frame)) * (fs / N_frame) #300KHz
    ts =1.0/fs
    t = np.arange(0, N_frame * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq_300k = 1 * (i + 1j * q)
    data_shift = data * iq_300k
    y = data_shift * win_funct
    sp = np.absolute(np.fft.fft(y))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs_shift = 20 * np.log10(s_mag / (2 ** 11))
    return s_dbfs, s_dbfs_shift


def rangedoppler(data, n_c=150, n_s=600, showdb=True):
    #n_s = 600
    #n_r = int(len(data)/n_s)-1
    table = np.zeros((n_c, n_s), dtype=np.complex_) #150 chirps,1000 samples/chirp
    for chirp_nr in range(n_c):
        table[chirp_nr, :] = data[(chirp_nr*n_s):(n_s*(chirp_nr+1))]
    #fft_output = np.fft.fft2(table)
    #2D FFT and Velocity-Distance Relationship
    Z_fft2 = abs(np.fft.fft2(table)) #
    if showdb:
        s_mag = np.abs(Z_fft2)/len(Z_fft2) #power spectrum
        Z_fft2 = 20 * np.log10(s_mag)
    #Data_fft2 = Z_fft2[0:int(n_r/2),0:int(n_s/2)] #get half
    Data_fft2 = Z_fft2[0:int(n_c/2),0:int(n_s/2)]#70:120] #get half 0:int(n_s/2)
    return Data_fft2, table

from scipy.interpolate import interp1d
#interp1d is used for 1-D interpolation (linear or cubic) of data points.

#ref: https://github.com/brunerm99/ADI_Radar_DSP
def cfar(X_k, num_guard_cells, num_ref_cells, bias=1, cfar_method='average',
    fa_rate=0.2):
    #X_k: An array of input data (presumably radar signal values).
    #num_guard_cells: The number of guard cells around the center cell.
    #num_ref_cells: The number of reference cells around the center cell.
    #bias: An optional bias value (default is 1).
    #cfar_method: The CFAR (Constant False Alarm Rate) method to use (default is ‘average’).
    #fa_rate: The desired false alarm rate (default is 0.2).

    N = X_k.size
    #cfar_values initialized as a masked array with the same shape as X_k.
    #Return an empty masked array of the given shape and dtype, where all the data are masked.
    cfar_values = np.ma.masked_all(X_k.shape)

    #iterates over the center cells (excluding guard and reference cells) in the input array.
    for center_index in range(num_guard_cells + num_ref_cells, N - (num_guard_cells + num_ref_cells)):
        min_index = center_index - (num_guard_cells + num_ref_cells)
        min_guard = center_index - num_guard_cells 
        max_index = center_index + (num_guard_cells + num_ref_cells) + 1
        max_guard = center_index + num_guard_cells + 1

        lower_nearby = X_k[min_index:min_guard] #ref cells
        upper_nearby = X_k[max_guard:max_index] #ref cells

        #The mean values of the nearby/ref cells are computed.
        lower_mean = np.mean(lower_nearby)
        upper_mean = np.mean(upper_nearby)

        if (cfar_method == 'average'):
            mean = np.mean(np.concatenate((lower_nearby, upper_nearby)))
            output = mean + bias
        elif (cfar_method == 'greatest'):
            mean = max(lower_mean, upper_mean)
            output = mean + bias
        elif (cfar_method == 'smallest'):
            mean = min(lower_mean, upper_mean)
            output = mean + bias
        elif (cfar_method == 'false_alarm'):
            refs = np.concatenate((lower_nearby, upper_nearby))
            noise_variance = np.sum(refs**2 / refs.size)
            output = (noise_variance * -2 * np.log(fa_rate))**0.5
        else:
            raise Exception('No CFAR method received')

        #The computed output value is assigned to the corresponding position in cfar_values.
        cfar_values[center_index] = output

    #Any masked (invalid) values in cfar_values are replaced with the minimum value in the array.
    cfar_values[np.where(cfar_values == np.ma.masked)] = np.min(cfar_values)

    #A masked array targets_only is created from a copy of X_k.
    targets_only = np.ma.masked_array(np.copy(X_k))
    #If the absolute value of a signal in X_k exceeds the corresponding value in cfar_values, it is masked in targets_only.
    targets_only[np.where(abs(X_k) > abs(cfar_values))] = np.ma.masked

    if (cfar_method == 'false_alarm'):
        return cfar_values, targets_only, noise_variance
    else:
        return cfar_values, targets_only
    
