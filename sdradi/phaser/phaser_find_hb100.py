import os
import pickle
import socket
import sys
import time
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from adi import ad9361
import adi
#from adi.cn0566 import CN0566
import mycn0566 as mycn0566
from phaser_functions import save_hb100_cal, spec_est
from scipy import signal

CN0566=mycn0566.CN0566

# First try to connect to a locally connected CN0566. On success, connect,
# on failure, connect to remote CN0566

# try:
#     print("Attempting to connect to CN0566 via ip:localhost...")
#     my_phaser = CN0566(uri="ip:localhost")
#     print("Found CN0566. Connecting to PlutoSDR via default IP address...")
#     my_sdr = ad9361(uri="ip:192.168.2.1")
#     print("PlutoSDR connected.")

# except:
#     print("CN0566 on ip.localhost not found, connecting via ip:phaser.local...")
#     my_phaser = CN0566(uri="ip:phaser.local")
#     print("Found CN0566. Connecting to PlutoSDR via shared context...")
#     my_sdr = ad9361(uri="ip:phaser.local:50901")
#     print("Found SDR on shared phaser.local.")

#test=adi.one_bit_adc_dac(uri="ip:phaser.local")

my_sdr = ad9361(uri="ip:phaser.local:50901")
print("Found SDR on shared phaser.local.")
sleep(1.5)
my_phaser = CN0566(uri="ip:phaser.local")
print("Found CN0566. Connecting to PlutoSDR via shared context...")
#my_sdr = ad9361(uri="ip:phaser.local:50901")
#print("Found SDR on shared phaser.local.")

my_phaser.sdr = my_sdr  # Set my_phaser.sdr

time.sleep(0.5)

# By default device_mode is "rx"
my_phaser.configure(device_mode="rx")

#  Configure SDR parameters.

my_sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
my_sdr._ctrl.debug_attrs[
    "adi,ensm-enable-txnrx-control-enable"
].value = "0"  # Disable pin control so spi can move the states
my_sdr._ctrl.debug_attrs["initialize"].value = "1"

my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
my_sdr._rxadc.set_kernel_buffers_count(1)  # No stale buffers to flush
rx = my_sdr._ctrl.find_channel("voltage0")
rx.attrs["quadrature_tracking_en"].value = "1"  # enable quadrature tracking
my_sdr.sample_rate = int(30000000)  # Sampling rate 30MHz
my_sdr.rx_buffer_size = int(4 * 256)
my_sdr.rx_rf_bandwidth = int(10e6)
# We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
my_sdr.gain_control_mode_chan0 = "manual"  # DISable AGC
my_sdr.gain_control_mode_chan1 = "manual"
my_sdr.rx_hardwaregain_chan0 = 0  # dB
my_sdr.rx_hardwaregain_chan1 = 0  # dB

my_sdr.rx_lo = int(2.0e9)  # Downconvert by 2GHz  # Receive Freq

#my_sdr.filter = "LTE20_MHz.ftr"  # Handy filter for fairly widdeband measurements
my_sdr.filter = "sdradi\phaser\LTE20_MHz.ftr"  # Handy filter for fairly widdeband measurements

# Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
# this is a negative number between 0 and -88
my_sdr.tx_hardwaregain_chan0 = int(-80)
my_sdr.tx_hardwaregain_chan1 = int(-80)


# Configure CN0566 parameters.
#     ADF4159 and ADAR1000 array attributes are exposed directly, although normally
#     accessed through other methods.


# Set initial PLL frequency to HB100 nominal

my_phaser.SignalFreq = 10.525e9
my_phaser.lo = int(my_phaser.SignalFreq) + my_sdr.rx_lo


gain_list = [64] * 8
for i in range(0, len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i], apply_cal=False)

# Aim the beam at boresight (zero degrees). Place HB100 right in front of array.
my_phaser.set_beam_phase_diff(0.0)

# Averages decide number of time samples are taken to plot and/or calibrate system. By default it is 1.
my_phaser.Averages = 8

# Initialize arrays for amplitudes, frequencies
full_ampl = np.empty(0)
full_freqs = np.empty(0)

# Set up range of frequencies to sweep. Sample rate is set to 30Msps,
# for a total of 30MHz of bandwidth (quadrature sampling)
# Filter is 20MHz LTE, so you get a bit less than 20MHz of usable
# bandwidth. Set step size to something less than 20MHz to ensure
# complete coverage.
f_start = 10.0e9
f_stop = 10.7e9
f_step = 10e6

for freq in range(int(f_start), int(f_stop), int(f_step)):
    #    print("frequency: ", freq)
    my_phaser.SignalFreq = freq
    my_phaser.frequency = (
        int(my_phaser.SignalFreq) + my_sdr.rx_lo
    ) // 4  # PLL feedback via /4 VCO output

    data = my_sdr.rx() #two item array, each iteam is (1024,) complex
    data_sum = data[0] + data[1]
    #    max0 = np.max(abs(data[0]))
    #    max1 = np.max(abs(data[1]))
    #    print("max signals: ", max0, max1)
    ampl, freqs = spec_est(data_sum, 30000000, ref=2 ^ 12, plot=False)
    ampl = np.fft.fftshift(ampl)
    ampl = np.flip(ampl)  # Just an experiment...
    freqs = np.fft.fftshift(freqs)
    freqs += freq
    full_freqs = np.concatenate((full_freqs, freqs))
    full_ampl = np.concatenate((full_ampl, ampl))
    sleep(0.1)
full_freqs /= 1e9  # Hz -> GHz

peak_index = np.argmax(full_ampl) #(71680,) -> 22526
peak_freq = full_freqs[peak_index]
print("Peak frequency found at ", full_freqs[peak_index], " GHz.")

plt.figure(figsize=(10,6))
#plt.figure(2)
plt.title("Full Spectrum, peak at " + str(full_freqs[peak_index]) + " GHz.")
plt.plot(full_freqs, full_ampl, linestyle="", marker="o", ms=2)
plt.xlabel("Frequency [GHz]")
plt.ylabel("Signal Strength")
plt.show()
print("You may need to close plot to continue...")

prompt = input("Save cal file? (y or n)")
if prompt.upper() == "Y":
    save_hb100_cal(peak_freq * 1e9)

del my_sdr
del my_phaser
