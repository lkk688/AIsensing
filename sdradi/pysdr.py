import numpy as np
#import adi
import matplotlib.pyplot as plt
from myadi.ad936x import ad9361
# transmitting a QPSK signal in the 915 MHz band, receiving it, and plotting the PSD

Rx_CHANNEL =1

sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz
num_samps = 100000 # number of samples per call to rx()

#sdr = adi.Pluto("ip:192.168.2.1")
#sdr = adi.ad9361(uri="ip:phaser:50901")
sdr = ad9361(uri="ip:192.168.1.69:50901")
#sdr = adi.ad9361(uri="ip:192.168.1.69:50901")
#sdr = adi.ad9361(uri="ip:pluto.local")
# Read back properties from hardware
print(sdr.tx_rf_bandwidth) #18,000,000
sdr.sample_rate = int(sample_rate) #1Mhz

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

# Configuration data channels
if Rx_CHANNEL==2:
    sdr.rx_enabled_channels = [0,1] #enable two rx channel
else:
    sdr.rx_enabled_channels = [0] #enables Rx0
sdr.tx_enabled_channels = [0] #enables Tx0

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

# Create transmit waveform (QPSK, 16 samples per symbol)
num_symbols = 1000
x_int = np.random.randint(0, 4, num_symbols) # (1000,) 0 to 3
x_degrees = x_int*360/4.0 + 45 # (1000,) 45, 135, 225, 315 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
samples = np.repeat(x_symbols, 16) # (16000,) 16 samples per symbol (rectangular pulses), each data repated 16 times
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
#IQ samples between -1 and 1, before transmitting them scale by 2^14  size: 16000
print(np.min(samples), np.max(samples))

# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(samples) # start transmitting
#To change the samples that are being continuously transmitted, you cannot simply call sdr.tx(samples) again with a new set of samples, 
# you have to first call sdr.tx_destroy_buffer(), then call sdr.tx(samples)

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr.rx()

# Receive samples
rx_samples = sdr.rx() #(100000,)
print(rx_samples)

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