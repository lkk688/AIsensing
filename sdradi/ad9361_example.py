#ref: https://ez.analog.com/ez-blogs/b/engineerzone-spotlight/posts/using-python-to-control-the-pluto-radio-and-plot-data
import time

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Create radio
piuri="ip:phaser.local:50901"
localuri="ip:analog.local"
antsdruri="ip:192.168.1.10"#connected via Ethernet with static IP
plutodruri="ip:192.168.2.16"#connected via USB
sdr = adi.ad9361(uri="ip:pluto.local")

Rx_CHANNEL =2

# Configure properties
sdr.rx_rf_bandwidth = 4000000 #4MHz
sdr.sample_rate = 6000000 #6MHz
sdr.rx_lo = 2000000000 #2Ghz 2000000000
sdr.tx_lo = 2000000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -30
sdr.gain_control_mode_chan0 = "slow_attack" #'manual'

# num_samps = 10000 # number of samples returned per call to rx()
# sdr.rx_buffer_size = num_samps

# Configuration data channels
if Rx_CHANNEL==2:
    sdr.rx_enabled_channels = [0,1] #enable two rx channel
else:
    sdr.rx_enabled_channels = [0] #enables Rx0
sdr.tx_enabled_channels = [0] #enables Tx0

# Read properties
print("RX LO %s" % (sdr.rx_lo)) #2Ghz

# Create a sinewave waveform
fs = int(sdr.sample_rate) #6MHz
N = 1024
fc = int(1000000 / (fs / N)) * (fs / N) #996093
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = i + 1j * q

# Send data
# Since sdr.tx_cyclic_buffer was set to True, this data will just keep repeating.  There’s no need to send it again.   
sdr.tx(iq)

plt.figure(figsize=(10,6))
# Collect data
for r in range(20):
    x = sdr.rx() #1024 size array of complex
    if Rx_CHANNEL==2:
        Rx_0=x[0]
        Rx_1=x[1]
    else:
        Rx_0=x
    f, Pxx_den = signal.periodogram(Rx_0, fs) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
    #returns f (ndarray): Array of sample frequencies.
    #returns Pxx_den (ndarray): Power spectral density or power spectrum of x.
    plt.clf()
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel("frequency [Hz]") #-3e^6 3e^6
    plt.ylabel("PSD [V**2/Hz]")
    plt.draw()
    plt.pause(0.05)
    time.sleep(0.1)

plt.show()
