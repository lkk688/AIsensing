import numpy as np
from scipy import ndimage
from timeit import default_timer as timer
from scipy import signal

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
#ref: https://github.com/brunerm99/ADI_Radar_DSP
def cfar(X_k, num_guard_cells, num_ref_cells, bias=1, cfar_method='average',
    fa_rate=0.2):
    N = X_k.size
    cfar_values = np.ma.masked_all(X_k.shape)
    for center_index in range(num_guard_cells + num_ref_cells, N - (num_guard_cells + num_ref_cells)):
        min_index = center_index - (num_guard_cells + num_ref_cells)
        min_guard = center_index - num_guard_cells 
        max_index = center_index + (num_guard_cells + num_ref_cells) + 1
        max_guard = center_index + num_guard_cells + 1

        lower_nearby = X_k[min_index:min_guard]
        upper_nearby = X_k[max_guard:max_index]

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

        cfar_values[center_index] = output

    cfar_values[np.where(cfar_values == np.ma.masked)] = np.min(cfar_values)

    targets_only = np.ma.masked_array(np.copy(X_k))
    targets_only[np.where(abs(X_k) > abs(cfar_values))] = np.ma.masked

    if (cfar_method == 'false_alarm'):
        return cfar_values, targets_only, noise_variance
    else:
        return cfar_values, targets_only
    