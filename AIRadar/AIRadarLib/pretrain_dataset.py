import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split, ConcatDataset
from AIRadarLib.waveform_utils import generate_adf4159_fmcw_chirp

# === SHAPE-ALIGNED SyntheticRadarDataset WITH AUTO-PADDING ===
class SyntheticRadarDataset(Dataset):
    def __init__(self, num_samples=1000, num_chirps=64, samples_per_chirp=64,
                 modulation_type='none', augment=True, max_targets=3,
                 target_shape=(64, 64), fixed_snr_db=40, use_random_snr=False):
        self.num_samples = num_samples
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.modulation_type = modulation_type
        self.augment = augment
        self.max_targets = max_targets
        self.target_shape = target_shape # (doppler_bins, range_bins)
        self.fixed_snr_db = fixed_snr_db
        self.use_random_snr = use_random_snr
        

    def pad_or_crop(self, x, target_shape):
        pad_d, pad_r = target_shape[0] - x.shape[0], target_shape[1] - x.shape[1]
        pad_d = max(pad_d, 0)
        pad_r = max(pad_r, 0)
        x_padded = np.pad(x, ((0, pad_d), (0, pad_r)), mode='constant')
        return x_padded[:target_shape[0], :target_shape[1]]

    def modulate_chirp(self, signal):
        if self.modulation_type == 'sine':
            t = np.linspace(0, 1, signal.shape[-1], endpoint=False)
            sine_wave = np.exp(1j * 2 * np.pi * 5 * t)
            return signal * sine_wave[np.newaxis, :]
        elif self.modulation_type == 'ofdm':
            carriers = np.fft.ifft(np.random.choice([1, -1], size=(self.num_chirps, self.samples_per_chirp)), axis=-1)
            return signal * carriers
        else:
            return signal

    def demodulate(self, signal):
        if self.modulation_type == 'sine':
            t = np.linspace(0, 1, signal.shape[-1], endpoint=False)
            sine_wave = np.exp(-1j * 2 * np.pi * 5 * t)
            return signal * sine_wave[np.newaxis, :]
        elif self.modulation_type == 'ofdm':
            return signal
        else:
            return signal

    def apply_augmentation(self, signal):
        if self.augment:
            if np.random.rand() < 0.5:
                noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) * 0.05
                signal += noise
            if np.random.rand() < 0.3:
                delay = np.random.randint(0, 3)
                signal = np.roll(signal, delay, axis=-1)
        return signal

    def inject_targets(self, shape):
        signal = np.zeros(shape, dtype=np.complex64)
        label_map = np.zeros((shape[0], shape[1]), dtype=np.float32)
        vel_map = np.zeros((shape[0], shape[1]), dtype=np.float32)
        meta_targets = []
        num_targets = np.random.randint(1, self.max_targets + 1)

        for _ in range(num_targets):
            rbin = np.random.randint(0, shape[1])
            vbin = np.random.randint(0, shape[0])
            #doppler_cycles = (vbin / shape[0]) - 0.5
            doppler_cycles = vbin / shape[0]
            snr_db = np.random.uniform(10, 30) if self.use_random_snr else self.fixed_snr_db
            amplitude = 10 ** (snr_db / 20)

            t = np.arange(shape[1])
            echo = amplitude * np.exp(1j * 2 * np.pi * doppler_cycles * np.outer(np.arange(shape[0]), t / shape[1]))
            signal += np.roll(echo, rbin, axis=1)
            label_map[vbin, rbin] = 1.0
            vel_map[vbin, rbin] = doppler_cycles
            meta_targets.append({'range_bin': rbin, 'doppler_bin': vbin, 'snr_db': snr_db})

        return signal, label_map, vel_map, meta_targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        signal, label_map, vel_map, metadata = self.inject_targets((self.num_chirps, self.samples_per_chirp))
        signal = self.apply_augmentation(signal)
        signal = self.modulate_chirp(signal)
        signal = self.demodulate(signal)

        signal = self.pad_or_crop(signal, self.target_shape)
        label_map = self.pad_or_crop(label_map, self.target_shape)
        vel_map = self.pad_or_crop(vel_map, self.target_shape)

        iq = np.stack([np.real(signal), np.imag(signal)], axis=-1)
        iq_tensor = torch.tensor(iq[np.newaxis, ...], dtype=torch.float32)
        label_tensor = torch.tensor(label_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)
        vel_tensor = torch.tensor(vel_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)

        return iq_tensor, label_tensor, vel_tensor, self.modulation_type, metadata

class DebugFMCWDataset(Dataset):
    def __init__(self, num_samples=100, num_chirps=64, samples_per_chirp=64, snr_db=60):
        self.num_samples = num_samples
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.snr_db = snr_db

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        shape = (self.num_chirps, self.samples_per_chirp)
        signal = np.zeros(shape, dtype=np.complex64)
        label_map = np.zeros(shape, dtype=np.float32)
        vel_map = np.zeros(shape, dtype=np.float32)

        # Inject a single target aligned with FFT bin
        rbin = 32
        vbin = 16
        doppler_cycles = vbin / shape[0]  # aligned with FFT bin
        amplitude = 10 ** (self.snr_db / 20)

        t = np.arange(shape[1])
        echo = amplitude * np.exp(1j * 2 * np.pi * doppler_cycles * np.outer(np.arange(shape[0]), t / shape[1]))
        signal += np.roll(echo, rbin, axis=1)
        label_map[vbin, rbin] = 1.0
        vel_map[vbin, rbin] = doppler_cycles
        meta = [{'range_bin': rbin, 'doppler_bin': vbin, 'snr_db': self.snr_db}]

        iq = np.stack([np.real(signal), np.imag(signal)], axis=-1)
        iq_tensor = torch.tensor(iq[np.newaxis, ...], dtype=torch.float32)
        label_tensor = torch.tensor(label_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)
        vel_tensor = torch.tensor(vel_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)

        return iq_tensor, label_tensor, vel_tensor, "none", meta

# === Visualization ===
def visualize_synthetic_sample1(dataset, index=0, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    iq, det, vel, mod, meta = dataset[index]
    det = det.squeeze().numpy()
    vel = vel.squeeze().numpy()
    raw_mag = np.sqrt(iq[0, ..., 0]**2 + iq[0, ..., 1]**2)
    complex_signal = iq[0, ..., 0] + 1j * iq[0, ..., 1]
    rd_fft = np.fft.fftshift(np.fft.fft2(complex_signal))
    rd_mag = np.abs(rd_fft)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    axs[0].imshow(raw_mag, aspect='auto', cmap='viridis')
    axs[0].set_title(f"Raw IQ Magnitude (Mod: {mod})")
    axs[1].imshow(rd_mag, aspect='auto', cmap='magma')
    axs[1].set_title("Range-Doppler Magnitude (FFT)")
    axs[2].imshow(det, aspect='auto', cmap='hot')
    axs[2].set_title("Detection Map")
    axs[3].imshow(vel, aspect='auto', cmap='coolwarm')
    axs[3].set_title("Velocity Map")

    for a in axs:
        a.set_xlabel("Range Bin")
        a.set_ylabel("Doppler Bin")
        a.grid(False)

    for t in meta:
        for ax in axs:
            ax.add_patch(patches.Circle((t['range_bin'], t['doppler_bin']), radius=1, color='lime', fill=False))
        axs[3].text(t['range_bin'], t['doppler_bin'], f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

    fig.suptitle(f"Sample {index} | {len(meta)} targets | Modulation: {mod}")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dataset visualization to {save_path}")
        plt.close()
    else:
        plt.show()

def visualize_synthetic_sample(dataset, index=0, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    iq, det, vel, mod, meta = dataset[index]
    det = det.squeeze().numpy()
    vel = vel.squeeze().numpy()
    raw_mag = np.sqrt(iq[0, ..., 0]**2 + iq[0, ..., 1]**2)
    complex_signal = iq[0, ..., 0] + 1j * iq[0, ..., 1]

    window = np.outer(np.hanning(complex_signal.shape[0]), np.hanning(complex_signal.shape[1]))
    windowed = complex_signal * window
    rd_fft = np.fft.fftshift(np.fft.fft2(windowed))
    rd_mag = np.abs(rd_fft)
    rd_log = 20 * np.log10(rd_mag + 1e-6)
    vmax = np.max(rd_log)
    vmin = vmax - 30

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(raw_mag, aspect='auto', cmap='viridis')
    axs[0].set_title(f"Raw IQ Magnitude (Mod: {mod})")
    axs[1].imshow(rd_log, aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axs[1].set_title("Range-Doppler Magnitude (dB)")
    axs[2].imshow(det, aspect='auto', cmap='hot')
    axs[2].set_title("Detection Map")
    axs[3].imshow(vel, aspect='auto', cmap='coolwarm')
    axs[3].set_title("Velocity Map")

    for a in axs:
        a.set_xlabel("Range Bin")
        a.set_ylabel("Doppler Bin")
        a.grid(False)

    for t in meta:
        v, r = t['doppler_bin'], t['range_bin']
        v_shift = (v - vel.shape[0] // 2) % vel.shape[0]
        r_shift = (r - vel.shape[1] // 2) % vel.shape[1]

        axs[0].add_patch(patches.Circle((r, v), radius=1, color='lime', fill=False))
        axs[1].add_patch(patches.Circle((r_shift, v_shift), radius=1, color='lime', fill=False))
        axs[2].add_patch(patches.Circle((r, v), radius=1, color='lime', fill=False))
        axs[3].text(r, v, f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

    fig.suptitle(f"Sample {index} | {len(meta)} targets | Modulation: {mod}")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dataset visualization to {save_path}")
        plt.close()
    else:
        plt.show()

def visualize_synthetic_sample3D(dataset, index=0, save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches
    from matplotlib import cm

    iq, det, vel, mod, meta = dataset[index]
    det = det.squeeze().numpy()
    vel = vel.squeeze().numpy()
    raw_mag = np.sqrt(iq[0, ..., 0]**2 + iq[0, ..., 1]**2)
    complex_signal = iq[0, ..., 0] + 1j * iq[0, ..., 1]

    window = np.outer(np.hanning(complex_signal.shape[0]), np.hanning(complex_signal.shape[1]))
    windowed = complex_signal * window
    rd_fft = np.fft.fftshift(np.fft.fft2(windowed))
    rd_mag = np.abs(rd_fft)
    rd_log = 20 * np.log10(rd_mag + 1e-6)
    vmax = np.max(rd_log)
    vmin = vmax - 30

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(raw_mag, aspect='auto', cmap='viridis')
    axs[0].set_title(f"Raw IQ Magnitude (Mod: {mod})")
    axs[1].imshow(rd_log, aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axs[1].set_title("Range-Doppler Magnitude (dB)")
    axs[2].imshow(det, aspect='auto', cmap='hot')
    axs[2].set_title("Detection Map")
    axs[3].imshow(vel, aspect='auto', cmap='coolwarm')
    axs[3].set_title("Velocity Map")

    for a in axs:
        a.set_xlabel("Range Bin")
        a.set_ylabel("Doppler Bin")
        a.grid(False)

    for t in meta:
        v, r = t['doppler_bin'], t['range_bin']
        v_shift = (v - vel.shape[0] // 2) % vel.shape[0]
        r_shift = (r - vel.shape[1] // 2) % vel.shape[1]

        axs[0].add_patch(patches.Circle((r, v), radius=1, color='lime', fill=False))
        axs[1].add_patch(patches.Circle((r_shift, v_shift), radius=1, color='lime', fill=False))
        axs[2].add_patch(patches.Circle((r, v), radius=1, color='lime', fill=False))
        axs[3].text(r, v, f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

    fig.suptitle(f"Sample {index} | {len(meta)} targets | Modulation: {mod}")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dataset visualization to {save_path}")
        plt.close()
    else:
        plt.show()

    # 3D visualization of RD FFT
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(rd_log.shape[1]), np.arange(rd_log.shape[0]))
    ax.plot_surface(X, Y, rd_log, cmap=cm.magma, linewidth=0, antialiased=False)
    ax.set_title("3D Range-Doppler (dB)")
    ax.set_xlabel("Range Bin")
    ax.set_ylabel("Doppler Bin")
    ax.set_zlabel("Magnitude (dB)")

    for t in meta:
        v, r = t['doppler_bin'], t['range_bin']
        v_shift = (v - vel.shape[0] // 2) % vel.shape[0]
        r_shift = (r - vel.shape[1] // 2) % vel.shape[1]
        ax.scatter(r_shift, v_shift, np.max(rd_log), c='lime', marker='o', s=60, edgecolor='black')

    plt.tight_layout()
    if save_path:
        base = save_path.rsplit('.', 1)[0]
        plt.savefig(f"{base}_3d.png")
        print(f"Saved 3D plot to {base}_3d.png")
        plt.close()
    else:
        plt.show()

def visualize_synthetic_batch(dataset, indices=None, save_dir="debug_outputs", cols=3):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os

    os.makedirs(save_dir, exist_ok=True)
    indices = indices or list(range(min(9, len(dataset))))

    for idx in indices:
        iq, det, vel, mod, meta = dataset[idx]
        det = det.squeeze().numpy()
        vel = vel.squeeze().numpy()
        rd_mag = np.sqrt(iq[0, ..., 0]**2 + iq[0, ..., 1]**2)

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        axs[0].imshow(rd_mag, aspect='auto', cmap='viridis')
        axs[0].set_title(f"Raw (Mod: {mod})")
        axs[1].imshow(det, aspect='auto', cmap='hot')
        axs[1].set_title("Detection Map")
        axs[2].imshow(vel, aspect='auto', cmap='coolwarm')
        axs[2].set_title("Velocity Map")

        for a in axs:
            a.set_xlabel("Range Bin")
            a.set_ylabel("Doppler Bin")

        for t in meta:
            for ax in axs:
                ax.add_patch(patches.Circle((t['range_bin'], t['doppler_bin']), radius=1, color='lime', fill=False))
            axs[2].text(t['range_bin'], t['doppler_bin'], f"{t['snr_db']:.1f}dB", color='black', fontsize=6, ha='center', va='center')

        fig.suptitle(f"Sample {idx} | {len(meta)} targets | Modulation: {mod}")
        fig.tight_layout()
        save_path = os.path.join(save_dir, f"batch_sample_{idx}.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()

if __name__ == "__main__":
    dataset = DebugFMCWDataset()#SyntheticRadarDataset()
    #visualize_synthetic_sample(dataset=dataset, index=0, save_path="data/debug_outputs_sample.png")
    visualize_synthetic_sample3D(dataset=dataset, index=0, save_path="data/debug_outputs_sample3D.png")
