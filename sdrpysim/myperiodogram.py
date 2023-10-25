#ref:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
rng = np.random.default_rng()

#Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by 0.001 V**2/Hz of white noise sampled at 10 kHz.
fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

f, Pxx_den = signal.periodogram(x, fs)
plt.figure()
plt.semilogy(f, Pxx_den)
plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
#plt.show()

#If we average the last half of the spectral density, to exclude the peak, we can recover the noise power on the signal.
np.mean(Pxx_den[25000:])
#compute and plot the power spectrum.
f, Pxx_spec = signal.periodogram(x, fs, 'flattop', scaling='spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.ylim([1e-4, 1e1])
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()