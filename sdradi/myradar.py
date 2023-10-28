import time

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from timeit import default_timer as timer

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

def initPhaser(urladdress, my_sdr):
    my_phaser = adi.CN0566(uri=urladdress, sdr=my_sdr)
    print("Phaser url: ", my_phaser.uri)
    print("Phaser already connected")

    # Initialize both ADAR1000s, set gains to max, and all phases to 0
    my_phaser.configure(device_mode="rx")
    my_phaser.load_gain_cal()
    my_phaser.load_phase_cal()
    for i in range(0, 8):
        my_phaser.set_chan_phase(i, 0)

    gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
    for i in range(0, len(gain_list)):
        my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)
    
    return my_phaser

def configureADF4159(my_phaser, output_freq= 12.1e9, BW= 500e6, num_steps= 1000, ramp_time= 1e3):
    # Configure the ADF4159 Rampling PLL
    #final output is 12.1GHz-LO(2.1GHz)=10GHz, Ramp range is 10GHz~10.5Ghz(10GHz+500MHz)
    # output_freq = 12.1e9 
    # BW = 500e6
    # num_steps = 1000
    # ramp_time = 1e3  # us
    ramp_time_s = ramp_time / 1e6
    my_phaser.frequency = int(output_freq / 4)  # Output frequency divided by 4, there is /4 ahead of the ADF4159 RFIN
    my_phaser.freq_dev_range = int(
        BW / 4
    )  # frequency deviation range in Hz.  This is the total freq deviation of the complete freq ramp
    my_phaser.freq_dev_step = int(
        BW / num_steps
    )  # frequency deviation step in Hz.  This is fDEV, in Hz.  Can be positive or negative
    my_phaser.freq_dev_time = int(
        ramp_time
    )  # total time (in us) of the complete frequency ramp
    my_phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
    my_phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
    my_phaser.delay_start_en = 0  # delay start
    my_phaser.ramp_delay_en = 0  # delay between ramps.
    my_phaser.trig_delay_en = 0  # triangle delay
    my_phaser.ramp_mode = "continuous_triangular"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
    my_phaser.sing_ful_tri = (
        0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
    )
    my_phaser.tx_trig_en = 0  # start a ramp with TXdata
    my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers
    return my_phaser

def initAD9361(urladdress, fs, center_freq, fft_size, Rx_CH=2, Tx_CH=2):
    # Create radio
    sdr = adi.ad9361(uri=urladdress)
    sdr.rx_rf_bandwidth = 4000000 #4MHz
    sdr.sample_rate = int(fs)

    # Configure Rx
    sdr.rx_lo = int(center_freq)  # set this to output_freq - (the freq of the HB100)
    #sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
    # Configuration data channels
    if Rx_CH==2:
        sdr.rx_enabled_channels = [0,1] #enable two rx channel
        sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
        sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
        sdr.rx_hardwaregain_chan0 = int(30)  # must be between -3 and 70
        sdr.rx_hardwaregain_chan1 = int(30)  # must be between -3 and 70
    else:
        sdr.rx_enabled_channels = [0] #enables Rx0
        sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
        sdr.rx_hardwaregain_chan0 = int(30)  # must be between -3 and 70
    sdr.rx_buffer_size = int(fft_size)
    #num_samps = 1024*100#10000 # number of samples returned per call to rx()
    #sdr.rx_buffer_size = num_samps
    

    # Configure Tx
    sdr.tx_lo = int(center_freq)
    if Tx_CH==2:
        sdr.tx_enabled_channels = [0, 1]
        sdr.tx_hardwaregain_chan0 = -88  # must be between 0 and -88
        sdr.tx_hardwaregain_chan1 = -0  # must be between 0 and -88
    else:
        sdr.tx_enabled_channels = [0] #enables Tx0
        sdr.tx_hardwaregain_chan0 = -0  # must be between 0 and -88
    sdr.tx_cyclic_buffer = True  # must set cyclic buffer to true for the tdd burst mode.  Otherwise Tx will turn on and off randomly
    
    # Enable TDD logic in pluto (this is for synchronizing Rx Buffer to ADF4159 TX input)
    # gpio = adi.one_bit_adc_dac(sdr_ip)
    # gpio.gpio_phaser_enable = True

    return sdr
    
def createcomplexsinusoid(fs, signal_freq = 100000, N = 1024):
    # Create a complex sinusoid
    #fc = 3000000
    #N = 1024 #rx buffer size
    #fs = int(sdr.tx_sample_rate)
    ts = 1 / float(fs)
    fc = int(signal_freq / (fs / N)) * (fs / N) #100KHz
    t = np.arange(0, N * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq = i + 1j * q
    return iq

def readiio(sdr):
    phy = sdr.ctx.find_device("ad9361-phy")
    # Read product ID register
    pi = phy.reg_read(0x37)
    #print(f"ID: {hex(pi)}")
    r = 0x80000088
    status = phy.reg_read(r)
    if status & 0b0100:
        print("Overflow")

def main():
    args = parser.parse_args()
    urladdress = args.urladdress #"ip:pluto.local"
    Rx_CHANNEL = args.rxch
    signal_type = args.signal
    plot_flag = args.plot
    
    # Configure properties
    #fs= 6000000 #6MHz
    sample_rate = 0.6e6 #0.6M
    center_freq = 2.1e9 #2.1G
    signal_freq = 100e3 #100K
    num_slices = 200
    fft_size = 1024 * 16
    img_array = np.zeros((num_slices, fft_size))

    # Configure the ADF4159 Rampling PLL
    #final output is 12.1GHz-LO(2.1GHz)=10GHz, Ramp range is 10GHz~10.5Ghz(10GHz+500MHz)
    output_freq = 12.1e9 
    BW = 500e6
    num_steps = 1000
    ramp_time = 1e3  # us
    ramp_time_s = ramp_time / 1e6

    sdr=initAD9361(urladdress, sample_rate, center_freq, fft_size, Rx_CH=2, Tx_CH=2)
    
    my_phaser=initPhaser(urladdress, sdr)

    my_phaser=configureADF4159(my_phaser, output_freq, BW, num_steps, ramp_time)

    # Read properties
    print("RX LO %s" % (sdr.rx_lo)) #2Ghz
    printSDRproperties(sdr)

    # Print config
    print(
        """
    CONFIG:
    Sample rate: {sample_rate}MHz
    Num samples: 2^{Nlog2}
    Bandwidth: {BW}MHz
    Ramp time: {ramp_time}ms
    Output frequency: {output_freq}MHz
    IF: {signal_freq}kHz
    """.format(
            sample_rate=sample_rate / 1e6,
            Nlog2=int(np.log2(fft_size)),
            BW=BW / 1e6,
            ramp_time=ramp_time / 1e3,
            output_freq=output_freq / 1e6,
            signal_freq=signal_freq / 1e3,
        )
    )

    fs = int(sdr.sample_rate) #0.6MHz
    print("sample_rate:", fs)
    N = int(sdr.rx_buffer_size)
    iq = createcomplexsinusoid(fs, signal_freq, N)
    iq_300k = createcomplexsinusoid(fs, signal_freq*3, N)

    # Send data
    sdr._ctx.set_timeout(0)
    sdr.tx([iq * 0.5, iq])  # only send data to the 2nd channel (that's all we need)

    c = 3e8
    default_rf_bw = BW #500e6
    N_frame = fft_size
    freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
    slope = BW / ramp_time_s
    dist = (freq - signal_freq) * c / (4 * slope)

    xdata = freq
    plot_dist = False

    print("Slope: %0.2fMHz/s" % (slope / 1e6))
    range_resolution = c / (2 * default_rf_bw)
    range_x = (100e3) * c / (4 * slope)
    #0, range_x or frequency 100e3, 200e3

    # Collect data
    alldata0 = np.empty(0, dtype=np.complex_) #Default is numpy.float64.
    rxtime=[]
    processtime=[]
    Nperiod=int(10*fs/fft_size) #total time 10s *fs=total samples /fft_size = Number of frames
    print("Total period for 10s:", Nperiod)
    for r in range(Nperiod):
        start = timer()
        x = sdr.rx() #1024 size array of complex
        rxt = timer()
        timedelta=rxt-start
        rxtime.append(timedelta)
        if Rx_CHANNEL==2:
            data0=x[0]
            data1=x[1]
            data = data0 + data1
        else:
            data=x
        datarate=len(data.real)*4/timedelta/1e6 #Mbps, complex data is 4bytes
        print("Data rate at ", datarate, "Mbps.") #7-8Mbps in 10240 points, 10Mbps in 102400points, single channel in 19-20Mbps
        alldata0 = np.concatenate((alldata0, data))

        readiio(sdr)
        endtime = timer()
        processtime.append(endtime-start)
    
        # Stop transmitting
    sdr.tx_destroy_buffer() #Clears TX buffer
    sdr.rx_destroy_buffer() #Clears RX buffer
    with open('./data/radardata.npy', 'wb') as f:
        np.save(f, alldata0)
# piuri="ip:phaser.local:50901"
# localuri="ip:analog.local"
# antsdruri="ip:192.168.1.10"#connected via Ethernet with static IP
# plutodruri="ip:192.168.2.16"#connected via USB

import argparse
parser = argparse.ArgumentParser(description='MyRadar')
parser.add_argument('--urladdress', default="ip:phaser.local:50901", type=str,
                    help='urladdress of the device')
parser.add_argument('--rxch', default=1, type=int, 
                    help='number of rx channels')
parser.add_argument('--signal', default="dds", type=str,
                    help='signal type: sinusoid, dds')
parser.add_argument('--plot', default=False, type=bool,
                    help='plot figure')

if __name__ == '__main__':
    main()