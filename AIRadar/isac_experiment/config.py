import torch
from dataclasses import dataclass
import numpy as np

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {DEVICE} ---")

# Speed of light in m/s
C0 = 299_792_458.0

@dataclass
class SystemParams:
    """
    System parameters for the ISAC (Integrated Sensing and Communication) system.
    """
    fc: float = 77e9          # Carrier frequency (Hz)
    B:  float = 150e6         # FMCW bandwidth (Hz), sets range resolution
    fs: float = 150e6         # ADC sample-rate (Hz), must be >= B
    M:  int   = 512           # Number of chirps (Doppler bins)
    N:  int   = 512           # Samples per chirp (Range FFT size)
    H:  float = 1.8           # Radar height (meters)
    az_fov: float = 60.0      # Azimuth Field of View (degrees)
    el_fov: float = 20.0      # Elevation Field of View (degrees)
    bev_r_max: float = 50.0   # Maximum range for Bird's Eye View (meters)

    @property
    def lambda_m(self):
        """Wavelength (meters)."""
        return C0 / self.fc

    @property
    def T_chirp(self):
        """Chirp duration (seconds)."""
        return self.N / self.fs

    @property
    def slope(self):
        """Frequency slope (Hz/s)."""
        return self.B / self.T_chirp

    def fmcw_axes(self):
        """
        Calculate Range and Velocity axes for FMCW processing.
        
        Returns:
            ra (np.ndarray): Range axis (meters).
            va (np.ndarray): Velocity axis (m/s).
        """
        # Range: bins 0..N/2-1 -> R_k = c * (k*fs/N) / (2*S) == c*k/(2*B)
        # Note: We use N // 2 because the range FFT is usually one-sided for real input, 
        # or we only care about positive range.
        ra = (C0 / (2.0 * self.B)) * np.arange(self.N // 2)
        
        # Doppler: f_d bins via slow-time PRF=1/T, then v = (λ/2) f_d
        # fftshift is used to center 0 velocity.
        f_d = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.T_chirp))
        va = (self.lambda_m / 2.0) * f_d
        return ra, va

    def otfs_axes(self):
        """
        Calculate Range (Delay) and Velocity (Doppler) axes for OTFS processing.
        
        Returns:
            r (np.ndarray): Range axis (meters).
            v (np.ndarray): Velocity axis (m/s).
        """
        # Range axis based on sampling rate and N
        r = np.linspace(0, (C0 / (2 * self.fs)) * self.N, self.N)
        
        # Velocity axis based on Doppler resolution
        # Max velocity corresponds to max Doppler shift
        v_max = (self.lambda_m / 2) * (self.fs / (self.N * self.M)) * (self.M / 2)
        v = np.linspace(-v_max, v_max, self.M)
        return r, v
