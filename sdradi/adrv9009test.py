# Copyright (C) 2019 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import matplotlib.pyplot as plt
from scipy import signal
from argparse import ArgumentParser
import numpy as np

def spectrogram_test():
    sample_rate = 1e6

    # Generate tone plus noise
    t = np.arange(1024*1000)/sample_rate # time vector
    f = 50e3 # freq of tone
    x = np.sin(2*np.pi*f*t) + 0.2*np.random.randn(len(t))
    plt.figure(0)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal Value")
    plt.plot(t[0:1024], x[0:1024],'.-')

    fft_size = 1024
    num_rows = len(x) // fft_size # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    plt.figure(1)
    plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, 0, len(x)/sample_rate])
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()
    print('done')

def psd_test():
    #Power Spectral Density (PSD)
    Fs = 1e6 # lets say we sampled at 1 MHz
    Ts = 1/Fs # sample period
    # assume x contains your array of IQ samples
    N = 2048 # number of samples to simulate

    t = Ts*np.arange(N)
    signal_freq=50000 #50K
    x = np.exp(1j*2*np.pi*signal_freq*t) # simulates sinusoid at 50 KHz

    #simulate the channel
    n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # complex noise with unity power
    noise_power = 2
    r = x + n * np.sqrt(noise_power)

    Nr=2048 # number of samples to receive
    x=r[0:Nr] ## we will only take the FFT of the first Nr samples
    x = x * np.hamming(len(x)) # apply a Hamming window
    PSD = np.abs(np.fft.fft(x))**2 / (N*Fs)
    PSD_log = 10.0*np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)

    center_freq = 0 #2.4e9 # frequency we tuned our SDR to
    f = np.arange(Fs/-2.0, Fs/2.0, Fs/N) # start, stop, step centered around 0 Hz
    f += center_freq # now add center frequency

    plt.plot(f, PSD_shifted)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    plt.show()

def QPSKtest():
    num_symbols = 1000

    x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
    x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
    fig = plt.figure()
    plt.plot(np.real(x_symbols), np.imag(x_symbols), '.')
    plt.grid(True)
    plt.show()

    #additive white Gaussian noise (AGWN)
    n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN with unity power
    noise_power = 0.01
    r = x_symbols + n * np.sqrt(noise_power)
    fig = plt.figure()
    plt.plot(np.real(r), np.imag(r), '.')
    plt.grid(True)
    plt.show()

    #simulating phase noise, which could result from phase jitter within the local oscillator (LO), replace the r with:
    phase_noise = np.random.randn(len(x_symbols)) * 0.1 # adjust multiplier for "strength" of phase noise
    r = r * np.exp(1j*phase_noise)
    fig = plt.figure()
    plt.plot(np.real(r), np.imag(r), '.')
    plt.grid(True)
    plt.show()


def DDStest(sdr):
    # Configure properties
    sdr.rx_enabled_channels = [0, 1]
    sdr.tx_enabled_channels = [0, 1]
    sdr.trx_lo = 2000000000
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = -10
    sdr.tx_hardwaregain_chan1 = -10
    print(sdr.tx_hardwaregain_chan0)
    print(sdr.tx_hardwaregain_chan1)
    sdr.gain_control_mode_chan0 = "slow_attack" #"fast_attack"
    sdr.gain_control_mode_chan1 = "slow_attack"

    # Read properties
    print("TRX LO %s" % (sdr.trx_lo))

    # Send data
    sdr.dds_enabled = [1, 1, 1, 1, 1, 1, 1, 1]
    sdr.dds_frequencies = [2000000, 0, 2000000, 0, 2000000, 0, 2000000, 0]
    sdr.dds_scales = [1, 0, 1, 0, 1, 0, 1, 0]
    sdr.dds_phases = [0, 0, 90000, 0, 0, 0, 90000, 0]


    # Collect data
    fsr = int(sdr.rx_sample_rate)
    for r in range(20):
        x = sdr.rx()
        f, Pxx_den = signal.periodogram(x[0], fsr)
        f2, Pxx_den2 = signal.periodogram(x[1], fsr)
        plt.clf()
        plt.semilogy(f, Pxx_den)
        plt.semilogy(f2, Pxx_den2)
        plt.ylim([1e-7, 1e4])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("PSD [V**2/Hz]")
        plt.draw()
        plt.pause(0.05)
        time.sleep(0.1)

    plt.show()

def txtest(sdr, tx_cyclic=True):
    #generate a sinusoid at +100 kHz, then transmit the complex signal at a carrier frequency of 915 MHz, 
    # causing the receiver to see a carrier at 915.1 MHz.
    sample_rate = 1e6 # Hz
    center_freq = 915e6 # Hz

    sdr.sample_rate = int(sample_rate)
    #sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq)
    #sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

    N = 10000 # number of samples to transmit at once
    t = np.arange(N)/sample_rate
    samples = 0.5*np.exp(2.0j*np.pi*100e3*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

    dualchtx=[samples,samples]
    
    if tx_cyclic==True:
        sdr.tx_cyclic_buffer = True # Enable cyclic buffers
        sdr.tx(dualchtx)
    else:
        # Transmit our batch of samples 100 times, so it should be 1 second worth of samples total, if USB can keep up
        for i in range(100):
            sdr.tx(dualchtx) # transmit the batch of samples once
    #sdr.tx_destroy_buffer(), then call sdr.tx(samples)

def rxtest(sdr):
    #sets the sample rate to 1 MHz, sets the center frequency to 100 MHz, 
    # sets the gain to 70 dB with automatic gain control turned off.
    sample_rate = 1e6 # Hz
    center_freq = 100e6 # Hz
    num_samps = 10000 # number of samples returned per call to rx()

    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70.0 # dB
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    #sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
    sdr.rx_buffer_size = num_samps

    dualchsamples = sdr.rx() # receive samples, two array for two channels
    
    samples=dualchsamples[1] #get the first channel
    sample_len=len(samples)
    print(samples[0:1000])
    plt.figure(0)
    plt.plot(np.arange(sample_len), samples,'.-')

    avg_pwr = np.var(samples) # (signal should have roughly zero mean)
    print("Average power:", avg_pwr)

    gainlevel=sdr._get_iio_attr('voltage0','hardwaregain', False)
    print("Current gain level:", gainlevel)

def txrxcyclic(sdr):
    sample_rate = 1e6 # 1MHz
    center_freq = 915e6 # 915MHz
    num_samps = 100000 # number of samples per call to rx()
    #sdr = adi.Pluto("ip:192.168.2.1")
    sdr.sample_rate = int(sample_rate)

    # Config Tx
    #sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq) #sdr.trx_lo?
    sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

    # Config Rx
    sdr.rx_lo = int(center_freq)
    #sdr.rx_rf_bandwidth = int(sample_rate)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

    # Create transmit waveform (QPSK, 16 samples per symbol)
    num_symbols = 1000
    x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
    x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
    samples = np.repeat(x_symbols, 16) # 16 samples per symbol (rectangular pulses)
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

    dualchtx=[samples,samples]

    # Start the transmitter
    sdr.tx_cyclic_buffer = True # Enable cyclic buffers
    sdr.tx(dualchtx) # start transmitting

    # Clear buffer just to be safe
    for i in range (0, 10):
        raw_data = sdr.rx()

    # Receive samples
    dualrx_samples = sdr.rx()
    #print(rx_samples)
    rx_samples=dualrx_samples[0] #100000 size

    # Stop transmitting
    sdr.tx_destroy_buffer()

    # Calculate power spectral density (frequency domain version of signal)
    psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
    psd_dB = 10*np.log10(psd)
    f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))

    # Plot time domain
    plt.figure(0)
    plt.plot(np.real(rx_samples[::100]))
    plt.plot(np.imag(rx_samples[::100]))
    plt.xlabel("Time")

    # Plot freq domain
    plt.figure(1)
    plt.plot(f/1e6, psd_dB)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD")
    plt.show()

def main():
    import adi
    parser = ArgumentParser()
    parser.add_argument('--devicename',  type=str, default='adrv9009', help='SDR name')#
    parser.add_argument('--ip', type=str, default='192.168.86.40', help='ip address') #192.168.86.29
    args = parser.parse_args()

    # Create radio
    uri='ip:'+args.ip
    sdr = adi.adrv9009(uri=uri, jesd_monitor=True)#"ip:192.168.86.25"
    #sdr.trx_lo = 2400000000

    #basic rx test
    sdr.sample_rate = int(2.5e6)
    data = sdr.rx()

    #rxtest(sdr)

    txtest(sdr)

    #txrxcyclic(sdr)

    DDStest(sdr)
    print("Done")


def simulation():
    #spectrogram_test()
    #psd_test()
    QPSKtest()
    print('finished simulation')

if __name__ == '__main__':
    main()
    #simulation()