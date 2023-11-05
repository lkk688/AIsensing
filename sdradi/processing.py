import numpy as np
from scipy import ndimage
from timeit import default_timer as timer
from scipy import signal

def extenddata(data, zoom=(20,5)):
    resampled_data=ndimage.zoom(data, zoom=(20,5))
    return resampled_data

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