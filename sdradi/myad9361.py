#ref: https://ez.analog.com/ez-blogs/b/engineerzone-spotlight/posts/using-python-to-control-the-pluto-radio-and-plot-data
import time

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from timeit import default_timer as timer

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

# Read back properties from hardware https://analogdevicesinc.github.io/pyadi-iio/devices/adi.ad936x.html
def printSDRproperties(sdr):
    print("Bandwidth of TX path:", sdr.tx_rf_bandwidth) #Bandwidth of front-end analog filter of TX path
    print("Loopback:", sdr.loopback) #0: disable, 1: Digital TX → Digital RX, 2: RF RX → RF TX
    print("TX gain chan0:", sdr.tx_hardwaregain_chan0) #Attenuation applied to TX path, -30
    print("TX gain chan1:", sdr.tx_hardwaregain_chan1) #Attenuation applied to TX path, -10
    print("TX enabled channels:", sdr.tx_enabled_channels) 
    print("RX enabled channels:", sdr.rx_enabled_channels)
    print("RX buffer size:", sdr.rx_buffer_size) #1024

    print("TX LO: %s" % (sdr.tx_lo)) #Carrier frequency of TX path
    print("RX LO: %s" % (sdr.rx_lo)) #Carrier frequency of RX path
    print("AGC Mode chan0:", sdr.gain_control_mode_chan0)
    print("AGC Mode chan1:", sdr.gain_control_mode_chan1)#Mode of receive path AGC. Options are: slow_attack, fast_attack, manual
    print("RX gain chan0:", sdr.rx_hardwaregain_chan0)#71
    print("RX gain chan1:", sdr.rx_hardwaregain_chan1)#71 Gain applied to RX path. Only applicable when gain_control_mode is set to ‘manual’
    print("rx_rf_bandwidth:", sdr.rx_rf_bandwidth) #4Mhz Bandwidth of front-end analog filter of RX path
    print("Sample rate:", sdr.sample_rate) #Sample rate RX and TX paths in samples per second
    print("DDS scales:", sdr.dds_scales)

#For FPGA based systems ADI reference designs include direct digital synthesizers (DDS) which can generate tones with arbitrary phase, frequency, and amplitude. 
# For each individual DAC channel there are two DDSs which can have a unique phase, frequency, and phase.
def ddstone(sdr, dualtune=True, dds_freq_hz = 10000, dds_scale = 0.9):
    n = len(sdr.dds_scales)
    # Generate a single complex tone
    #dds_freq_hz = 10000 #must be less than 1/2 the sample rate.
    #dds_scale = 0.9 #range [0,1]
    # Enable all DDSs
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

def main():
    args = parser.parse_args()
    urladdress = args.urladdress #"ip:pluto.local"
    Rx_CHANNEL = args.rxch
    signal_type = args.signal
    plot_flag = args.plot

    #testlibiioaccess(urladdress)

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
    Nperiod=int(2*fs/num_samps) #total time 10s *fs=total samples /fft_size = Number of frames
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
    with open('./data/ad9361data.npy', 'wb') as f:
        np.save(f, alldata0)
    plotfigure(ts, alldata0.real[0:num_samps*2])
    print(np.mean(rxtime))
    print(np.mean(processtime))

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
    axs[1].set_ylim([1e-7, 1e2])
    axs[1].set_xlabel("frequency [MHz]") #-3e^6 3e^6
    axs[1].set_ylabel("PSD [V**2/Hz]")
    axs[1].set_title("Spectrum")
    plt.show()

def updatefigure(axs, t, data0, data1, specf,specp):
    #axs[0].plot(t, data0.real, t, data1.real)
    axs[0].cla()  
    axs[0].plot(t, data0.real, marker="o", ms=2, color="red")  # Only plot real part
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
# plutodruri="ip:192.168.2.16"#connected via USB
import argparse
parser = argparse.ArgumentParser(description='MyAD9361')
parser.add_argument('--urladdress', default="ip:pluto.local", type=str,
                    help='urladdress of the device')
parser.add_argument('--rxch', default=1, type=int, 
                    help='number of rx channels')
parser.add_argument('--signal', default="dds", type=str,
                    help='signal type: sinusoid, dds')
parser.add_argument('--plot', default=False, type=bool,
                    help='plot figure')

if __name__ == '__main__':
    main()