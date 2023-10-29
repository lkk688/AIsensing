import time

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from timeit import default_timer as timer


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
    axs[1].set_ylim([1e-7, 1e2])
    axs[1].set_xlabel("frequency [MHz]") #-3e^6 3e^6
    axs[1].set_ylabel("PSD [V**2/Hz]")
    axs[1].set_title("Spectrum")
    plt.show()

def loadad9361data():
    with open('./data/ad9361data.npy', 'rb') as f:
        alldata = np.load(f)
    print(len(alldata))
    fs= 6000000 #6MHz
    ts = 1/float(fs)
    num_samps = 1024*100
    plotfigure(ts, alldata.real[0:num_samps*2])
    Nperiod=int(2*fs/num_samps)
    print(len(alldata)/num_samps)

def showspectrum(data, fs, N_frame):
    # fc = int(100e3 / (fs / N_frame)) * (fs / N_frame) #300KHz
    # ts =1.0/fs
    # t = np.arange(0, N_frame * ts, ts)
    # i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    # q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    # iq_100k = 1 * (i + 1j * q)
    data=data[0:N_frame] #* iq_100k
    win_funct = np.blackman(len(data))
    y = data * win_funct
    sp = np.absolute(np.fft.fft(y))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
    """there's a scaling issue on the y-axis of the waterfallcthe data is off by 300kHz.  To fix, I'm just shifting the freq"""
    # fc = int(300e3 / (fs / N_frame)) * (fs / N_frame) #300KHz
    # ts =1.0/fs
    # t = np.arange(0, N_frame * ts, ts)
    # i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    # q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    # iq_300k = 1 * (i + 1j * q)
    # data_shift = data * iq_300k
    # y = data_shift * win_funct
    # sp = np.absolute(np.fft.fft(y))
    # sp = np.fft.fftshift(sp)
    # s_mag = np.abs(sp) / np.sum(win_funct)
    # s_mag = np.maximum(s_mag, 10 ** (-15))
    # s_dbfs_shift = 20 * np.log10(s_mag / (2 ** 11))
    #dist, s_dbfs
    #freq, s_dbfs
    freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
    plt.figure(figsize=(10,6))
    plt.plot(freq,s_dbfs,color="red")
    #plt.plot(freq,s_dbfs_shift) #s_dbfs)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Spectrum')
    plt.show()

def dynamicspectrum(dataall, fs, N_frame):
    totallen=len(dataall)
    Ntotalframe=int(totallen/N_frame)-1
    plt.figure(figsize=(10,6))
    for i in range(Ntotalframe):
        data = dataall[i*N_frame:(i+1)*N_frame]
        #data=data[0:N_frame]
        win_funct = np.blackman(len(data))
        y = data * win_funct
        sp = np.absolute(np.fft.fft(y))
        sp = np.fft.fftshift(sp)
        s_mag = np.abs(sp) / np.sum(win_funct)
        s_mag = np.maximum(s_mag, 10 ** (-15))
        s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
        freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
        plt.clf()
        plt.plot(freq,s_dbfs)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        #plt.ylim(-140, -40)
        plt.title('Spectrum')
        #plt.show()
        plt.draw()
        plt.pause(0.1)

def dynamicspectrum2(dataall, fs, N_frame):
    totallen=len(dataall)
    Ntotalframe=int(totallen/N_frame)-1
    plt.figure(figsize=(10,6))
    for i in range(Ntotalframe):
        data = dataall[i*N_frame:(i+1)*N_frame]
        f, Pxx_den = signal.periodogram(data.real, fs)
        
        #data=data[0:N_frame]
        # win_funct = np.blackman(len(data))
        # y = data * win_funct
        # sp = np.absolute(np.fft.fft(y))
        # sp = np.fft.fftshift(sp)
        # s_mag = np.abs(sp) / np.sum(win_funct)
        # s_mag = np.maximum(s_mag, 10 ** (-15))
        # s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
        # freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
        plt.clf()
        plt.semilogy(f, Pxx_den)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.ylim([1e-9, 1e2])
        plt.title('Spectrum')
        #plt.show()
        plt.draw()
        plt.pause(0.1)

def dynamicRD(dataall, fs, N_frame):
    totallen=len(dataall)
    Ntotalframe=int(totallen/N_frame)-1
    plt.figure(figsize=(18,18))
    for i in range(Ntotalframe):
        data = dataall[i*N_frame:(i+1)*N_frame]
        n_s = 600
        n_r = int(len(data)/n_s)-1
        table = np.zeros((n_r, n_s)) #150 chirps,1000 samples/chirp
        for chirp_nr in range(n_r):
            table[chirp_nr, :] = data[(chirp_nr*n_s):(n_s*(chirp_nr+1))]
        #fft_output = np.fft.fft2(table)
        #2D FFT and Velocity-Distance Relationship
        Z_fft2 = abs(np.fft.fft2(table)) #
        #Data_fft2 = Z_fft2[0:int(n_r/2),0:int(n_s/2)] #get half
        Data_fft2 = Z_fft2[0:n_r,70:120] #get half 0:int(n_s/2)
        #plt.subplot(4,2,8)
        plt.imshow(Data_fft2) 
        plt.xlabel("Range")
        plt.ylabel("Velocity")
        plt.title('Velocity-Range 2D FFT')
        plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
        plt.draw()
        plt.pause(0.1)

def matplotlibspectrogram(dataall, fs, N_frame):
    # plt.figure(figsize=(10,6))
    totallen=len(dataall)
    Ntotalframe=int(totallen/N_frame)-1
    # plt.specgram(dataall[100:100+fft_size],Fs=fs)
    # plt.show()
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 15))
    #t = np.arange(0.0, N_frame, 1.0/fs)
    for i in range(0, Ntotalframe, 10):
        data = dataall[i*N_frame:(i+1)*N_frame]
        n_s = 600
        n_r = int(len(data)/n_s)-1
        table = np.zeros((n_r, n_s)) #150 chirps,1000 samples/chirp
        for chirp_nr in range(n_r):
            table[chirp_nr, :] = data[(chirp_nr*n_s):(n_s*(chirp_nr+1))]
        Z_fft2 = abs(np.fft.fft2(table))
        Data_fft2 = Z_fft2[0:n_r,0:300]
        ax1.cla()  
        #ax1.plot(data)
        ax1.imshow(Data_fft2) 
        ax2.cla()  
        Pxx, freqs, bins, im = ax2.specgram(data, NFFT=N_frame, Fs=fs)
        # The `specgram` method returns 4 objects. They are:
        # - Pxx: the periodogram
        # - freqs: the frequency vector
        # - bins: the centers of the time bins
        # - im: the matplotlib.image.AxesImage instance representing the data in the plot
        #plt.show()
        plt.draw()
        plt.pause(0.1)


def main():
    with open('./data/radardata5s-indoor2.npy', 'rb') as f:
        alldata = np.load(f)
    print(len(alldata))
    sample_rate = 0.6e6 #0.6M
    fs = sample_rate
    center_freq = 2.1e9 #2.1G
    signal_freq = 100e3 #100K
    num_slices = 200
    fft_size = 1024 * 16 #16384
    ts = 1/float(fs)
    #plotfigure(ts, alldata.real[0:fft_size])
    Nperiod=int(2*fs/fft_size) 
    print("Total period:", Nperiod)
    N_frame = fft_size
    c = 3e8
    BW = 500e6
    num_steps = 1000
    ramp_time = 1e3  # us
    ramp_time_s = ramp_time / 1e6
    slope = BW / ramp_time_s
    Nr = int(ramp_time_s * fs) #Number ADC sampling points in each chirp

    fc = int(100e3 / (fs / N_frame)) * (fs / N_frame) #300KHz
    ts =1.0/fs
    t = np.arange(0, len(alldata) * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq_100k = 1 * (i + 1j * q)
    alldata=alldata * iq_100k

    showspectrum(alldata.real, fs, N_frame)

    dynamicspectrum(alldata.real, fs, N_frame)

    dynamicRD(alldata.real, fs, fft_size)

    

    

    matplotlibspectrogram(alldata.real, fs, N_frame)

    #dynamicspectrum2(alldata, fs, fft_size)

    

    

    #2D plot
    newfft_size=Nr
    Nd=int(len(alldata)/newfft_size) #70
    mat2D = np.zeros((Nd, newfft_size)) #128 chirps * 1024 samples/chirp
    i = 0
    while(i<Nd):
        mat2D[i, :] = alldata.real[i*newfft_size:(i+1)*newfft_size]
        i = i + 1
    plt.figure(figsize=(10,6))
    plt.matshow(mat2D)
    plt.title('2D original data')

    #2D FFT and Velocity-Distance Relationship
    Z_fft2 = abs(np.fft.fft2(mat2D)) #(150, 1000)
    Data_fft2 = Z_fft2#[0:75,0:500] #get half
    #plt.subplot(4,2,8)
    plt.figure(figsize=(10,6))
    plt.imshow(Data_fft2) 
    plt.xlabel("Range")
    plt.ylabel("Velocity")
    plt.title('Velocity-Range 2D FFT')

    plt.tight_layout(pad=3, w_pad=0.05, h_pad=0.05)
    plt.show()
    
    

    #Range FFT
    doppler = 10*np.log10(np.abs(np.fft.fft(alldata.real[0:fft_size]))) #(1024,)
    frequency = np.fft.fftfreq(fft_size, 1/fs)
    range = frequency*c/(2*slope)
    plt.figure(figsize=(10,6))
    plt.plot(range[0:int(fft_size/2)],doppler[0:int(fft_size/2)])
    plt.xlabel('Frequency->Distance')
    plt.ylabel('Amplitude')
    plt.title('IF Signal FFT')
    
    
    #dist = (freq - signal_freq) * c / (4 * slope)

    


if __name__ == '__main__':
    main()