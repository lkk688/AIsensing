
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
sdr = adi.ad9361(uri=antsdruri)

# Configure properties
sdr.rx_rf_bandwidth = 4000000 #4MHz
sdr.sample_rate = 6000000
sdr.rx_lo = 2000000000 #2Ghz 2000000000
sdr.tx_lo = 2000000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -30
sdr.gain_control_mode_chan0 = "slow_attack"

# Configuration data channels
sdr.rx_enabled_channels = [0]
sdr.tx_enabled_channels = [0]

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
sdr.tx(iq)

plt.figure(figsize=(10,6))
# Collect data
for r in range(20):
    x = sdr.rx()
    f, Pxx_den = signal.periodogram(x, fs) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
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
