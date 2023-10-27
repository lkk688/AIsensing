import numpy as np
import matplotlib.pyplot as plt

Fs = 300 # sample rate
Ts = 1/Fs # sample period
N = 2048 # number of samples to simulate

t = Ts*np.arange(N)
x = np.exp(1j*2*np.pi*50*t) # simulates sinusoid at 50 Hz

n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # complex noise with unity power
noise_power = 2
r = x + n * np.sqrt(noise_power)

PSD = np.abs(np.fft.fft(r))**2 / (N*Fs)
PSD_log = 10.0*np.log10(PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.arange(Fs/-2.0, Fs/2.0, Fs/N) # start, stop, step

fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(10,6))
axs[0].plot(t, r.real)
#axs[0].set_xlim(0, 2)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('data0 and data1')
axs[0].grid(True)

#axs[1].semilogy(f, PSD_shifted)
axs[1].plot(f, PSD_shifted)
#axs[1].set_ylim([1e-7, 1e2])
axs[1].set_xlabel("frequency [Hz]") #-3e^6 3e^6
axs[1].set_ylabel("Magnitude [dB]")

# plt.plot(f, PSD_shifted)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.show()