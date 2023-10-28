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

def main():
    with open('./data/ad9361data.npy', 'rb') as f:
        alldata = np.load(f)
    print(len(alldata))
    fs= 6000000 #6MHz
    ts = 1/float(fs)
    num_samps = 1024*100
    plotfigure(ts, alldata.real[0:num_samps*2])
    Nperiod=int(2*fs/num_samps)
    print(len(alldata)/num_samps)


if __name__ == '__main__':
    main()