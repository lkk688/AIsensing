import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

class RadarParameterDesigner:
    def __init__(self, center_freq=77e9, target_max_range=200, target_range_resolution=0.5,
                 target_max_velocity=100, target_velocity_resolution=1,
                 max_bandwidth=4e9, max_sample_rate=50e6, freq_range=(76e9, 81e9)):
        self.center_freq = center_freq
        self.speed_of_light = 3e8
        self.wavelength = self.speed_of_light / self.center_freq

        self.max_bandwidth = max_bandwidth
        self.max_sample_rate = max_sample_rate
        self.freq_range = freq_range

        self.target_max_range = target_max_range
        self.target_range_resolution = target_range_resolution
        self.target_max_velocity = target_max_velocity
        self.target_velocity_resolution = target_velocity_resolution

        self.bandwidth = None
        self.sample_rate = None
        self.chirp_duration = None
        self.num_chirps = None
        self.range_fft_size = None
        self.doppler_fft_size = None

    def design_parameters(self):
        self.bandwidth = self.speed_of_light / (2 * self.target_range_resolution)
        if self.bandwidth > self.max_bandwidth:
            self.bandwidth = self.max_bandwidth
            self.target_range_resolution = self.speed_of_light / (2 * self.bandwidth)

        basic_chirp_time = (2 * self.target_max_range) / self.speed_of_light
        self.chirp_duration = max(5.5 * basic_chirp_time, 20e-6)

        estimated_sample_rate = max(2 * self.bandwidth, 4 * self.bandwidth)
        if estimated_sample_rate > self.max_sample_rate:
            self.sample_rate = self.max_sample_rate
        else:
            self.sample_rate = estimated_sample_rate

        phase_step = (2 * np.pi * self.bandwidth) / self.sample_rate
        while phase_step >= np.pi/2:
            if self.sample_rate < self.max_sample_rate:
                self.sample_rate = min(self.sample_rate * 1.5, self.max_sample_rate)
            else:
                self.bandwidth *= 0.9
            phase_step = (2 * np.pi * self.bandwidth) / self.sample_rate

        self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)

        self.num_chirps = int(np.ceil(self.wavelength / (2 * self.chirp_duration * self.target_velocity_resolution)))
        if self.num_chirps > 512:
            self.num_chirps = 512
            self.target_velocity_resolution = self.wavelength / (2 * self.num_chirps * self.chirp_duration)

        self.range_fft_size = 2 ** int(np.ceil(np.log2(self.samples_per_chirp)))
        self.doppler_fft_size = 2 ** int(np.ceil(np.log2(self.num_chirps)))

        achievable_max_range = (self.sample_rate * self.speed_of_light * self.chirp_duration) / (2 * self.bandwidth)
        if achievable_max_range > 300:
            self.chirp_duration = (300 * 2 * self.bandwidth) / (self.sample_rate * self.speed_of_light)
            self.samples_per_chirp = int(self.sample_rate * self.chirp_duration)
            self.range_fft_size = 2 ** int(np.ceil(np.log2(self.samples_per_chirp)))

        achievable_max_velocity = self.wavelength / (4 * self.chirp_duration)
        if achievable_max_velocity > 60:
            self.target_max_velocity = 60

    def get_parameters(self):
        return {
            'center_frequency': self.center_freq,
            'bandwidth': self.bandwidth,
            'sample_rate': self.sample_rate,
            'chirp_duration': self.chirp_duration,
            'num_chirps': self.num_chirps,
            'range_fft_size': self.range_fft_size,
            'doppler_fft_size': self.doppler_fft_size
        }

class RadarSimulator:
    def __init__(self, designer):
        self.designer = designer
        self.fs = designer.sample_rate
        self.B = designer.bandwidth
        self.T_chirp = designer.chirp_duration
        self.N_chirps = designer.num_chirps
        self.Ns = int(self.fs * self.T_chirp)
        self.slope = self.B / self.T_chirp
        self.center_freq = designer.center_freq
        self.speed_of_light = 3e8
        self.wavelength = self.speed_of_light / self.center_freq

    def ca_cfar_2d(self, magnitude, guard=2, train=8, scale=3):
        detections = np.zeros_like(magnitude, dtype=bool)
        rows, cols = magnitude.shape
        for i in range(train+guard, rows-train-guard):
            for j in range(train+guard, cols-train-guard):
                ref_cells = magnitude[i-train-guard:i+train+guard+1, j-train-guard:j+train+guard+1].copy()
                ref_cells[train:train+2*guard+1, train:train+2*guard+1] = 0
                threshold = scale * np.mean(ref_cells[ref_cells > 0])
                if magnitude[i, j] > threshold:
                    detections[i, j] = True
        return detections

    def leakage_cancellation(self, mix):
        leakage = np.mean(mix, axis=0)
        return mix - leakage

    def doppler_nulling(self, doppler_fft):
        mid = self.N_chirps // 2
        doppler_fft[mid-2:mid+3, :] = 0
        return doppler_fft

    def simulate_radar(self, targets, snr_db=30):
        t = np.arange(self.Ns) / self.fs
        tx_chirp = np.exp(1j * np.pi * self.slope * t**2)

        rx_signal = np.zeros((self.N_chirps, self.Ns), dtype=complex)
        for k in range(self.N_chirps):
            echo = np.zeros(self.Ns, dtype=complex)
            for target in targets:
                R = target['R'] + target['v'] * k * self.T_chirp
                tau = 2 * R / self.speed_of_light
                delayed_t = t - tau
                valid_idx = delayed_t >= 0
                delayed_chirp = np.zeros_like(t)
                delayed_chirp[valid_idx] = np.exp(1j * np.pi * self.slope * delayed_t[valid_idx]**2)
                phase_shift = np.exp(-1j * 2 * np.pi * self.center_freq * tau)
                echo += delayed_chirp * phase_shift
            rx_signal[k, :] = echo

        mix = rx_signal * np.conj(tx_chirp)
        mix = self.leakage_cancellation(mix)

        noise_power = np.mean(np.abs(mix)**2) / (10**(snr_db/10))
        mix += np.sqrt(noise_power/2)*(np.random.randn(*mix.shape)+1j*np.random.randn(*mix.shape))

        range_fft = np.fft.fftshift(np.fft.fft(mix, axis=1), axes=1)
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        doppler_fft = self.doppler_nulling(doppler_fft)

        range_axis = np.linspace(-self.fs/2, self.fs/2, self.Ns) * self.speed_of_light / (2 * self.slope)
        PRF = 1 / self.T_chirp
        velocity_axis = np.linspace(-PRF/2, PRF/2, self.N_chirps) * self.wavelength / 2

        magnitude = np.abs(doppler_fft)
        detections = self.ca_cfar_2d(magnitude)

        plt.figure(figsize=(10,6))
        plt.imshow(20*np.log10(magnitude), extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]], aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Range-Doppler Map with CA-CFAR Detection (SNR={snr_db}dB)')

        for v_idx, r_idx in zip(*np.where(detections)):
            plt.plot(range_axis[r_idx], velocity_axis[v_idx], 'go', markersize=6, markeredgewidth=1)

        for target in targets:
            plt.plot(target['R'], target['v'], 'rx', markersize=12, markeredgewidth=3)
        plt.grid()
        plt.show()

    def snr_detection_simulation(self, targets, snr_range=np.linspace(-10, 20, 7)):
        detection_rates = []
        for snr_db in snr_range:
            detected = 0
            trials = 30
            for _ in range(trials):
                t = np.arange(self.Ns) / self.fs
                tx_chirp = np.exp(1j * np.pi * self.slope * t**2)

                rx_signal = np.zeros((self.N_chirps, self.Ns), dtype=complex)
                for k in range(self.N_chirps):
                    echo = np.zeros(self.Ns, dtype=complex)
                    for target in targets:
                        R = target['R'] + target['v'] * k * self.T_chirp
                        tau = 2 * R / self.speed_of_light
                        delayed_t = t - tau
                        valid_idx = delayed_t >= 0
                        delayed_chirp = np.zeros_like(t)
                        delayed_chirp[valid_idx] = np.exp(1j * np.pi * self.slope * delayed_t[valid_idx]**2)
                        phase_shift = np.exp(-1j * 2 * np.pi * self.center_freq * tau)
                        echo += delayed_chirp * phase_shift
                    rx_signal[k, :] = echo

                mix = rx_signal * np.conj(tx_chirp)
                mix = self.leakage_cancellation(mix)

                noise_power = np.mean(np.abs(mix)**2) / (10**(snr_db/10))
                mix += np.sqrt(noise_power/2)*(np.random.randn(*mix.shape)+1j*np.random.randn(*mix.shape))

                range_fft = np.fft.fftshift(np.fft.fft(mix, axis=1), axes=1)
                doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
                doppler_fft = self.doppler_nulling(doppler_fft)

                magnitude = np.abs(doppler_fft)
                detections = self.ca_cfar_2d(magnitude)

                success = False
                for target in targets:
                    R_idx = np.argmin(np.abs(np.linspace(-self.fs/2, self.fs/2, self.Ns) * self.speed_of_light / (2 * self.slope) - target['R']))
                    V_idx = np.argmin(np.abs(np.linspace(-1/(2*self.T_chirp), 1/(2*self.T_chirp), self.N_chirps) * self.wavelength / 2 - target['v']))
                    window = detections[max(0, V_idx-4):V_idx+5, max(0, R_idx-4):R_idx+5]
                    if np.any(window):
                        success = True
                        break
                if success:
                    detected += 1
            detection_rates.append(detected / trials)

        plt.figure(figsize=(8,5))
        plt.plot(snr_range, detection_rates, 'o-', label='Detection Probability')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Detection Probability')
        plt.title('Detection Probability vs SNR (with Cancellation)')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    designer = RadarParameterDesigner(
        center_freq=77e9,
        target_max_range=250,
        target_range_resolution=0.5,
        target_max_velocity=80,
        target_velocity_resolution=1
    )
    designer.design_parameters()

    simulator = RadarSimulator(designer)

    targets = [
        {'R': 50, 'v': 20},
        {'R': 100, 'v': -10},
    ]

    simulator.simulate_radar(targets, snr_db=10)
    simulator.snr_detection_simulation(targets)
