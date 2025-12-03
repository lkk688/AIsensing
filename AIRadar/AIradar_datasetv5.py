#modified based on AIRadar/AIradar_dataset_corrected.py
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from scipy.constants import c
import torch
from torch.utils.data import Dataset
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Import visualization functions
try:
    from AIRadarLib.visualization import (
        plot_signal_time_and_spectrum,
        plot_instantaneous_frequency,
        plot_range_doppler_map_with_ground_truth,
        plot_3d_range_doppler_map_with_ground_truth
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: AIRadarLib.visualization not available. Some visualizations will be skipped.")
    VISUALIZATION_AVAILABLE = False

# Import CFAR detection function
try:
    from AIRadarLib.radar_det import cfar_2d_numpy
    CFAR_AVAILABLE = True
except ImportError:
    print("Warning: AIRadarLib.radar_det not available. CFAR detection will be skipped.")
    CFAR_AVAILABLE = False

# --- Radar Configurations ---
RADAR_CONFIGS = {
    'config1': {
        'name': 'Automotive_77GHz_LongRange',
        'signal_type': 'FMCW',  # Default signal type
        'fc': 77e9,             # 77 GHz
        'B': 1.5e9,             # 1.5 GHz Bandwidth
        'T_chirp': 40e-6,       # 40 μs
        'fs': 51.2e6,           # 51.2 MHz Sampling Rate
        'N_chirps': 128,        # Number of chirps
        'R_max': 100.0,         # Max range 100m
        'description': 'Standard automotive long-range radar configuration',
        'cfar_params': {        # Tuned for high resolution
            'num_train': 10,
            'num_guard': 4,
            'threshold_offset': 15,
            'nms_kernel_size': 5
        }
    },
    'config2': {
        'name': 'XBand_10GHz_MediumRange',
        'signal_type': 'FMCW',
        'fc': 10e9,             # 10 GHz (X-band)
        'B': 1.0e9,             # 1 GHz Bandwidth
        'T_chirp': 40e-6,       # 40 μs
        'fs': 40e6,             # 40 MHz Sampling Rate (Lower ADC requirement)
        'N_chirps': 128,        # Number of chirps
        'R_max': 100.0,         # Max range 100m
        'description': 'X-band radar for medium range surveillance or robotics',
        'cfar_params': {        # Tuned for lower resolution/SNR
            'num_train': 20,      # Increased training cells for more stable noise estimate
            'num_guard': 4,       # Increased guard cells to avoid target self-masking
            'threshold_offset': 12, # Adjusted to balance Precision/Recall
            'nms_kernel_size': 9  # Increased NMS kernel to merge clustered detections
        }
    },
    'config_otfs': {
        'name': 'OTFS_Automotive_77GHz',
        'signal_type': 'OTFS',
        'fc': 77e9,             # 77 GHz
        'B': 1.536e9,           # ~1.5 GHz (Aligned for 512 subcarriers * 3MHz spacing or similar)
                                # Let's align with otfs_ofdm.py or reasonable values
                                # otfs_ofdm.py: 512 subcarriers, 60kHz SCS -> 30.72 MHz BW. That's small.
                                # Automotive radar usually has >1GHz BW.
                                # Let's use similar to config1 but adjusted for OTFS structure.
                                # If N_chirps(Doppler bins)=128, N_samples(Delay bins)=512.
                                # fs = 51.2MHz. B = fs? No, B <= fs/2 usually for complex? Or B=fs for complex sampling.
                                # Let's keep config1-like physical params.
        'T_chirp': 40e-6,       # Symbol duration
        'fs': 51.2e6,           # Sampling rate
        'N_chirps': 128,        # Number of OTFS symbols (Doppler bins)
        'N_samples': 512,       # Number of subcarriers (Delay bins) - Explicitly set for OTFS
        'R_max': 100.0,
        'description': 'OTFS Radar configuration',
        'cfar_params': {
            'num_train': 10,
            'num_guard': 4,
            'threshold_offset': 15,
            'nms_kernel_size': 5
        }
    }
}

class AIRadarDataset(Dataset):
    def __init__(self, 
                 num_samples=100,
                 radar_config=None,          # New argument for config selection
                 config_name='config1',      # Default config name
                 fc=None,                    # Overrides config if provided
                 B=None,
                 T_chirp=None,
                 fs=None,                    # Sampling Rate Override
                 N_samples=None,             # Deprecated: Calculated from fs * T_chirp
                 N_chirps=None,
                 R_max=None,
                 SNR_dB_min=20,              # Minimum SNR
                 SNR_dB_max=40,              # Maximum SNR
                 zero_pad_factor=2,          # Zero padding factor - Reduced to avoid visual smearing
                 max_targets=3,              # Maximum targets per sample
                 save_path='data/radar_corrected_test5',
                 precision='float32',
                 drawfig=False,
                 datapath=None,
                 cfar_params=None):          # Explicit CFAR parameters override
        """
        Initialize corrected radar dataset with proper FMCW parameters
        
        Args:
            num_samples: Number of samples to generate
            radar_config: Dictionary containing radar parameters (optional)
            config_name: Name of predefined config to use if radar_config is None ('config1', 'config2')
            fc, B, T_chirp, fs, N_chirps, R_max: Overrides for specific parameters
            N_samples: (Deprecated) Can still be used if fs is not provided
            SNR_dB_min: Minimum SNR in dB
            SNR_dB_max: Maximum SNR in dB
            zero_pad_factor: Zero padding factor for FFT
            max_targets: Maximum number of targets per sample
            save_path: Path to save generated data
            precision: Data precision ('float32' or 'float16')
            drawfig: Whether to generate visualization plots
            datapath: Path to existing dataset (if loading)
            cfar_params: Dictionary to override CFAR parameters
        """
        
        # Load Base Configuration
        if radar_config is None:
            if config_name in RADAR_CONFIGS:
                cfg = RADAR_CONFIGS[config_name]
                print(f"Loading Radar Configuration: {config_name} ({cfg['name']})")
            else:
                print(f"Warning: Config '{config_name}' not found. Using defaults (config1).")
                cfg = RADAR_CONFIGS['config1']
        else:
            cfg = radar_config
            
        # Store parameters, allowing overrides
        self.config = cfg # Store full config
        self.signal_type = cfg.get('signal_type', 'FMCW')
        self.num_samples = num_samples
        self.fc = fc if fc is not None else cfg.get('fc', 77e9)
        self.B = B if B is not None else cfg.get('B', 1.5e9)
        self.T = T_chirp if T_chirp is not None else cfg.get('T_chirp', 40e-6)
        self.Nc = N_chirps if N_chirps is not None else cfg.get('N_chirps', 128)
        self.R_max = R_max if R_max is not None else cfg.get('R_max', 100)
        
        # Handle Sampling Rate (fs) and Samples per Chirp (Ns)
        # Priority: 1. Explicit fs arg, 2. Config fs, 3. Explicit N_samples arg, 4. Config N_samples (Legacy)
        
        self.fs = fs if fs is not None else cfg.get('fs', None)
        
        if self.fs is not None:
            # Calculate N_samples from fs
            self.Ns = int(self.fs * self.T)
            if N_samples is not None:
                print(f"Info: N_samples argument ({N_samples}) ignored because fs ({self.fs/1e6} MHz) is provided.")
        else:
            # Fallback to N_samples
            self.Ns = N_samples if N_samples is not None else cfg.get('N_samples', 2048)
            self.fs = self.Ns / self.T
            print(f"Info: fs not provided. Calculated fs = {self.fs/1e6:.2f} MHz from N_samples ({self.Ns})")
        
        self.SNR_dB_min = SNR_dB_min
        self.SNR_dB_max = SNR_dB_max
        self.max_targets = max_targets
        self.save_path = save_path
        self.drawfig = drawfig
        self.precision = precision
        
        # CFAR Parameters
        default_cfar = {'num_train': 10, 'num_guard': 4, 'threshold_offset': 15, 'nms_kernel_size': 5}
        config_cfar = cfg.get('cfar_params', default_cfar)
        self.cfar_params = cfar_params if cfar_params is not None else config_cfar
        
        # Calculate derived parameters
        self.lambda_c = c / self.fc
        self.slope = self.B / self.T
        self.zero_pad = zero_pad_factor * self.Ns
        
        # ✅ Sampling rate: Ensure it matches Ns/T and satisfies Nyquist for R_max
        # Theoretical max beat freq = Slope * 2 * R_max / c
        f_beat_max = self.slope * 2 * self.R_max / c
        fs_required = 2 * f_beat_max
        
        # Verify Nyquist
        if self.fs < fs_required:
            print(f"WARNING: Sampling rate {self.fs/1e6:.2f} MHz is below Nyquist {fs_required/1e6:.2f} MHz for R_max={self.R_max}m")
            print("Consider increasing fs/N_samples or reducing T_chirp/R_max.")
    
        # ✅ Doppler max velocity
        self.v_max = self.lambda_c / (4 * self.T)
        
        # Time vectors
        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T
        
        if self.signal_type == 'OTFS':
            # ✅ OTFS Range Axis
            # Delay resolution = 1/fs
            # Range resolution = c / (2 * fs)
            range_res = c / (2 * self.fs)
            self.range_axis = np.arange(self.Ns) * range_res
            
            # Set dimensions for dataset
            self.num_range_bins = self.Ns
            self.num_doppler_bins = self.Nc
            
        else:
            # ✅ FMCW Range Axis using correct beat-to-range conversion
            range_res = (c * self.fs) / (2 * self.slope * self.zero_pad)
            self.range_axis = np.arange(self.zero_pad // 2) * range_res
            
            # Set dimensions for dataset
            self.num_range_bins = self.zero_pad // 2
            self.num_doppler_bins = self.Nc
        
        # ✅ Velocity axis from Doppler FFT
        self.velocity_axis = np.fft.fftshift(np.fft.fftfreq(self.Nc, d=self.T)) * self.lambda_c / 2
        
        # Calculate system parameters
        self.range_resolution = c / (2 * self.B)
        self.velocity_resolution = self.lambda_c / (2 * self.Nc * self.T)
        self.max_unambiguous_velocity = self.lambda_c / (4 * self.T)
        
        # Initialize data containers
        self.time_domain_data = None
        self.range_doppler_maps = None
        self.target_masks = None
        self.target_info = None
        self.cfar_detections = None
        
        print("\n=== Corrected Radar System Parameters ===")
        print(f"✅ Signal Type         : {self.signal_type}")
        print(f"✅ Center Frequency    : {self.fc/1e9:.1f} GHz")
        print(f"✅ Bandwidth           : {self.B/1e6:.1f} MHz")
        print(f"✅ Chirp/Symbol Duration: {self.T*1e6:.1f} μs")
        print(f"✅ Sample Rate         : {self.fs/1e6:.1f} MHz")
        print(f"✅ Maximum Range       : {self.R_max:.1f} m")
        print(f"✅ Range Resolution    : {self.range_resolution:.2f} m")
        print(f"✅ Maximum Velocity    : {self.v_max:.1f} m/s")
        print(f"✅ Velocity Resolution : {self.velocity_resolution:.2f} m/s")
        print(f"✅ Samples per Chirp/Sym: {self.Ns}")
        print(f"✅ Number of Chirps/Sym: {self.Nc}")
        print(f"✅ Range Bins          : {self.num_range_bins}")
        print(f"✅ Doppler Bins        : {self.num_doppler_bins}")
        print(f"✅ CFAR Parameters     : {self.cfar_params}")
        print("========================================\n")
        
        # Initialize storage for generated data
        self.time_domain_data = None      # Shape: [num_samples, num_rx, num_chirps, samples_per_chirp] - Complex time domain signals
        self.range_doppler_maps = None    # Shape: [num_samples, num_doppler_bins, num_range_bins] - Range-Doppler magnitude maps
        self.target_masks = None          # Shape: [num_samples, num_doppler_bins, num_range_bins, 1] - Binary target masks
        self.target_info = None           # List of dictionaries containing target metadata
        self.cfar_detections = None       # Shape: [num_samples] - List of CFAR detection results
        
        if datapath is not None:
            print(f"Loading radar data from {datapath}")
            self._load_data(datapath)
        else:
            print("Generating new radar data")
            self.generate_dataset()
    
    def generate_targets(self, num_targets=None):
        """
        Generate random targets within valid range and velocity limits
        
        Returns:
            List of target dictionaries with range, velocity, and RCS
        """
        if num_targets is None:
            num_targets = random.randint(1, self.max_targets)
        
        targets = []
        for _ in range(num_targets):
            # ✅ Ground truth within valid range and velocity
            target_range = np.random.uniform(10, self.R_max - 10)
            target_velocity = np.random.uniform(-self.v_max + 1, self.v_max - 1)
            target_rcs = np.random.uniform(5.0, 30.0)  # RCS in dBsm
            
            targets.append({
                'range': target_range,
                'velocity': target_velocity,
                'rcs': target_rcs,
                'azimuth': np.random.uniform(-30, 30),  # degrees
                'elevation': np.random.uniform(-10, 10)  # degrees
            })
        
        return targets
    
    def simulate_fmcw_signal(self, targets, snr_db=20):
        """
        Simulate FMCW radar signal for multiple targets using vectorized operations.
        
        The received signal is a mix of reflected signals from all targets.
        For each target, the beat signal is:
        s(t) = A * exp(j * 2*pi * (fb * t_fast + fd * t_slow))
        where:
        - fb = 2 * R * slope / c  (Beat frequency due to range)
        - fd = 2 * v / lambda     (Doppler frequency due to velocity)
        
        Args:
            targets: List of target dictionaries
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            beat: Time domain beat signal (complex)
            rdm: Range-Doppler map (magnitude in dB)
        """
        # Initialize beat signal
        beat = np.zeros((self.Nc, self.Ns), dtype=np.complex128)
        
        if targets:
            # Vectorized target parameter extraction
            ranges = np.array([t['range'] for t in targets])
            velocities = np.array([t['velocity'] for t in targets])
            rcs = np.array([t['rcs'] for t in targets])
            
            # Convert RCS to linear amplitude
            rcs_linear = 10 ** (rcs / 10)
            amplitudes = np.sqrt(rcs_linear)
            
            # Calculate frequencies
            # fb: Beat frequency (Hz) = 2 * Range * Slope / c
            # fd: Doppler frequency (Hz) = 2 * Velocity / Lambda
            fb = 2 * ranges * self.slope / c
            fd = 2 * velocities / self.lambda_c
            
            # Prepare for broadcasting: (num_targets, num_chirps, num_samples)
            # fb, fd, amp: (K,) -> (K, 1, 1)
            fb_grid = fb[:, None, None]
            fd_grid = fd[:, None, None]
            amp_grid = amplitudes[:, None, None]
            
            # t_fast: (1, 1, Ns)
            # t_slow: (1, Nc, 1)
            t_fast_grid = self.t_fast[None, None, :]
            t_slow_grid = self.t_slow[None, :, None]
            
            # Calculate phase and signal
            phase = 2 * np.pi * (fb_grid * t_fast_grid + fd_grid * t_slow_grid)
            signal = amp_grid * np.exp(1j * phase)
            
            # Sum over all targets
            beat = np.sum(signal, axis=0)
        
        # Apply Hann window to reduce spectral leakage (Fast-time / Range)
        window_range = np.hanning(self.Ns)
        beat *= window_range[None, :]
        
        # Apply Hann window to reduce spectral leakage (Slow-time / Doppler)
        window_doppler = np.hanning(self.Nc)
        beat *= window_doppler[:, None]
        
        # Add Additive White Gaussian Noise (AWGN)
        signal_power = np.mean(np.abs(beat)**2)
        if signal_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_scale = np.sqrt(noise_power / 2) # Divide by 2 for complex noise
            noise = (np.random.randn(*beat.shape) + 1j * np.random.randn(*beat.shape)) * noise_scale
            beat += noise
        
        # Perform FFTs to generate Range-Doppler Map
        # Range FFT (Fast-time)
        range_fft = np.fft.fft(beat, n=self.zero_pad, axis=1)
        range_fft = range_fft[:, :self.zero_pad // 2] # Keep positive frequencies
        
        # Doppler FFT (Slow-time)
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        
        # Convert to magnitude in dB
        rdm = 20 * np.log10(np.abs(doppler_fft) + 1e-6)
        
        return beat, rdm
    
    def _otfs_modulate(self, dd_grid):
        """
        Modulates a Delay-Doppler (M x N) grid to a time-domain signal.
        Tx Chain: ISFFT (DD->TF) -> Heisenberg (TF->Time)
        """
        # dd_grid shape: (Ns, Nc) = (Delay, Doppler)
        # ISFFT over Delay (axis 0) and FFT over Doppler (axis 1)
        tf_grid = np.fft.ifft(dd_grid, axis=0)
        tf_grid = np.fft.fft(tf_grid, axis=1)

        # Heisenberg Transform: IFFT over subcarriers (axis 0)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        
        # Serialize (column-major)
        tx_signal = time_domain_grid.flatten(order='F')
        return tx_signal

    def _otfs_demodulate(self, rx_signal):
        """
        Demodulates a time-domain signal back to a Delay-Doppler grid.
        Rx Chain: Wigner (Time->TF) -> SFFT (TF->DD)
        """
        # Deserialize
        time_domain_grid = rx_signal.reshape((self.Ns, self.Nc), order='F')
        
        # Wigner Transform: FFT over time samples (axis 0)
        tf_grid = np.fft.fft(time_domain_grid, axis=0)
        
        # SFFT: IFFT over Time (axis 1) and FFT over Freq (axis 0)
        dd_grid = np.fft.ifft(tf_grid, axis=1)
        dd_grid = np.fft.fft(dd_grid, axis=0)
        
        return dd_grid

    def simulate_otfs_signal(self, targets, snr_db=20):
        """
        Simulate OTFS radar signal.
        """
        # 1. Generate QAM Symbols (QPSK)
        num_symbols = self.Ns * self.Nc
        bits = np.random.randint(0, 4, num_symbols)
        # QPSK mapping
        mod_map = {
            0: (1 + 1j) / np.sqrt(2),
            1: (1 - 1j) / np.sqrt(2),
            2: (-1 + 1j) / np.sqrt(2),
            3: (-1 - 1j) / np.sqrt(2)
        }
        symbols = np.array([mod_map[b] for b in bits])
        tx_dd_grid = symbols.reshape((self.Ns, self.Nc))

        # 2. Modulate to Time Domain
        tx_signal = self._otfs_modulate(tx_dd_grid)

        # 3. Apply Channel (Delay, Doppler, Noise)
        n_samples = tx_signal.size
        rx_signal = np.zeros(n_samples, dtype=complex)
        time_vector = np.arange(n_samples) / self.fs

        for target in targets:
            # Calculate delay and doppler
            range_m = target['range']
            velocity_mps = target['velocity']
            rcs = target['rcs']
            
            # RCS to amplitude
            amplitude = np.sqrt(10**(rcs/10))

            # 2-way delay
            delay_sec = 2 * range_m / c
            delay_samples = int(round(delay_sec * self.fs))

            if delay_samples < n_samples:
                delayed_signal = np.roll(tx_signal, delay_samples)
                # Zero out wrapped around part (optional, but cleaner for radar)
                delayed_signal[:delay_samples] = 0 
                
                # Doppler shift (2-way)
                doppler_hz = 2 * velocity_mps * self.fc / c
                doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector)
                
                rx_signal += amplitude * delayed_signal * doppler_shift

        # 4. Add Noise
        signal_power = np.mean(np.abs(rx_signal)**2)
        if signal_power > 0:
            snr_linear = 10**(snr_db/10)
            noise_power = signal_power / snr_linear
            noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * np.sqrt(noise_power/2)
            rx_signal += noise

        # 5. Demodulate
        rx_dd_grid = self._otfs_demodulate(rx_signal)

        # 6. Channel Estimation (Deconvolution in DD domain)
        # H_est = IFFT2( FFT2(RX) / FFT2(TX) )
        # This assumes the channel acts as a circular convolution in the DD domain.
        
        rx_dd_fft = np.fft.fft2(rx_dd_grid)
        tx_dd_fft = np.fft.fft2(tx_dd_grid)
        
        # Regularized division
        epsilon = 1e-6
        # tx_dd_fft_conj = np.conj(tx_dd_fft)
        # ddm_fft = (rx_dd_fft * tx_dd_fft_conj) / (np.abs(tx_dd_fft)**2 + epsilon)
        # Or simple ZF with regularization
        ddm_fft = rx_dd_fft / (tx_dd_fft + epsilon)
        
        ddm_complex = np.fft.ifft2(ddm_fft)
        
        # 7. Format Output
        # ddm_complex is (Ns, Nc) -> (Delay, Doppler)
        # We want (Doppler, Delay) for consistency with RDM
        ddm_transposed = ddm_complex.T # (Nc, Ns)
        
        # Shift zero frequency to center
        # Delay is usually 0 to max, Doppler is +/-
        # FFT shift only Doppler (axis 0 of transposed)
        # Also need to check if Delay needs shifting. 
        # In OTFS, delay is usually [0, max].
        # Doppler is [-max, max].
        # So we shift Doppler.
        ddm_shifted = np.fft.fftshift(ddm_transposed, axes=0)
        
        # Magnitude in dB
        ddm_mag = np.abs(ddm_shifted)
        ddm_db = 20 * np.log10(ddm_mag + 1e-6)
        
        # Crop to match range bins if needed (FMCW uses zero_pad//2)
        # Here we just take the first Ns/2 delay bins if we want to match RDM structure or keep full?
        # FMCW RDM has num_range_bins = zero_pad // 2.
        # Here we have Ns bins.
        # If we want to match self.num_range_bins = self.zero_pad // 2 = Ns (since zero_pad_factor=2).
        # Actually self.zero_pad = zero_pad_factor * Ns.
        # self.num_range_bins = zero_pad // 2 = Ns.
        # So the shape matches (Nc, Ns).
        
        # Reshape time signal for storage (Nc, Ns)
        rx_time_reshaped = rx_signal.reshape((self.Ns, self.Nc), order='F').T # (Nc, Ns)
        
        return rx_time_reshaped, ddm_db

    def create_target_mask(self, targets):
        """
        Create binary mask for target locations in range-doppler map
        
        Args:
            targets: List of target dictionaries
            
        Returns:
            Binary mask array - Shape: [doppler_bins, range_bins] (2D)
        """
        mask = np.zeros((self.num_doppler_bins, self.num_range_bins))
        
        for target in targets:
            # Find closest range and velocity bins
            range_idx = np.argmin(np.abs(self.range_axis - target['range']))
            velocity_idx = np.argmin(np.abs(self.velocity_axis - target['velocity']))
            
            # Set mask (with some tolerance for nearby bins)
            for di in range(-1, 2):
                for ri in range(-1, 2):
                    v_idx = velocity_idx + di
                    r_idx = range_idx + ri
                    if 0 <= v_idx < self.num_doppler_bins and 0 <= r_idx < self.num_range_bins:
                        mask[v_idx, r_idx] = 1.0
        
        return mask
    
    def _cfar_2d_custom(self, rd_map_db, num_train=8, num_guard=4, range_res=0.5, 
                       doppler_res=0.25, max_range=100, max_speed=50, 
                       threshold_offset=4, nms_kernel_size=3):
        """
        Custom CFAR implementation optimized for 2D dB maps.
        
        Args:
            rd_map_db: Input range-Doppler map in dB [num_doppler, num_range]
            num_train: Number of training cells per side
            num_guard: Number of guard cells around CUT
            range_res: Range resolution in meters per bin
            doppler_res: Doppler resolution in m/s per bin
            max_range: Maximum detection range in meters
            max_speed: Maximum absolute speed in m/s
            threshold_offset: Threshold offset in dB
            nms_kernel_size: Non-maximum suppression kernel size
            
        Returns:
            List of detection dictionaries
        """
        from scipy.signal import convolve2d
        from scipy.ndimage import maximum_filter
        
        rows, cols = rd_map_db.shape
        
        # CFAR window setup
        k = num_guard + num_train
        window_size = 2 * k + 1
        full_kernel = np.ones((window_size, window_size), dtype=np.float32)
        guard_area = np.zeros_like(full_kernel)
        guard_area[num_train:num_train + 2*num_guard + 1,
                   num_train:num_train + 2*num_guard + 1] = 1
        train_kernel = full_kernel - guard_area
        
        # Greatest-Of CFAR kernels
        horiz_kernel = train_kernel.copy()
        horiz_kernel[num_train:num_train + 2*num_guard + 1, :] = 0
        vert_kernel = train_kernel.copy()
        vert_kernel[:, num_train:num_train + 2*num_guard + 1] = 0
        
        # Estimate noise level using convolution
        # Note: rd_map_db is already in dB, so we are averaging dB values (Log-CFAR approximation)
        noise_h = convolve2d(rd_map_db, horiz_kernel / np.sum(horiz_kernel), 
                            mode='same', boundary='symm')
        noise_v = convolve2d(rd_map_db, vert_kernel / np.sum(vert_kernel), 
                            mode='same', boundary='symm')
        noise_est = np.maximum(noise_h, noise_v)
        
        # Apply threshold
        threshold = noise_est + threshold_offset
        detections = rd_map_db > threshold
        
        # Non-maximum suppression
        if nms_kernel_size > 1:
            local_max = maximum_filter(rd_map_db, size=nms_kernel_size)
            detections &= (rd_map_db == local_max)
        
        doppler_idxs, range_idxs = np.where(detections)
        results = []
        
        num_doppler = rows
        
        for d_idx, r_idx in zip(doppler_idxs, range_idxs):
            range_m = r_idx * range_res
            velocity_mps = (d_idx - num_doppler // 2) * doppler_res
            
            # Filter by range and velocity limits
            if not (0.5 < range_m < max_range and abs(velocity_mps) < max_speed):
                continue
                
            results.append({
                "range_idx": r_idx,
                "doppler_idx": d_idx,
                "range_m": range_m,
                "velocity_mps": velocity_mps,
                "angle_deg": None
            })
        
        return results

    def cfar_detection(self, rd_map):
        """
        Perform CFAR (Constant False Alarm Rate) detection on Range-Doppler map
        
        Args:
            rd_map: Range-Doppler magnitude map in dB
                   Shape: [doppler_bins, range_bins]
        
        Returns:
            detection_results: List of detection dictionaries
        """
        # Calculate resolution parameters from axes
        range_res = self.range_axis[1] - self.range_axis[0]
        velocity_res = self.velocity_axis[1] - self.velocity_axis[0]
            
        # Perform CFAR detection
        # We use the internal custom CFAR which is optimized for this dataset
        cfar_results = self._cfar_2d_custom(
            rd_map,
            num_train=self.cfar_params.get('num_train', 10),
            num_guard=self.cfar_params.get('num_guard', 4),
            range_res=range_res,
            doppler_res=velocity_res,
            max_range=self.R_max,
            max_speed=50,       # Maximum speed in m/s
            threshold_offset=self.cfar_params.get('threshold_offset', 15),
            nms_kernel_size=self.cfar_params.get('nms_kernel_size', 5)
        )

        # Add magnitude information to CFAR results
        for detection in cfar_results:
            d_idx = detection['doppler_idx']
            r_idx = detection['range_idx']
            if 0 <= d_idx < rd_map.shape[0] and 0 <= r_idx < rd_map.shape[1]:
                detection['magnitude'] = rd_map[d_idx, r_idx]
                
        return cfar_results


    def generate_dataset(self):
        """
        Generate the radar dataset including:
        1. Target generation
        2. FMCW signal simulation (beat signal)
        3. Range-Doppler Map generation
        4. CFAR detection
        5. Data storage and visualization
        """
        print(f"Generating {self.num_samples} samples...")
        
        # Initialize data arrays with detailed shape documentation
        self.time_domain_data = np.zeros((self.num_samples, self.Nc, self.Ns, 2), dtype=self.precision)
        # Shape: [num_samples, num_chirps, samples_per_chirp, 2] - Complex time domain signals (real/imag)
        
        self.range_doppler_maps = np.zeros((self.num_samples, self.num_doppler_bins, self.num_range_bins), dtype=self.precision)
        # Shape: [num_samples, doppler_bins, range_bins] - Range-Doppler magnitude maps in dB
        
        self.target_masks = np.zeros((self.num_samples, self.num_doppler_bins, self.num_range_bins, 1), dtype=self.precision)
        # Shape: [num_samples, doppler_bins, range_bins, 1] - Binary ground truth target masks
        
        self.target_info = []
        # List of dictionaries containing target metadata for each sample
        
        self.cfar_detections = []  # Store CFAR detection results for each sample
        
        # Create directory for visualizations if needed
        vis_path = os.path.join(self.save_path, 'visualizations')
        if self.drawfig and not os.path.exists(vis_path):
            os.makedirs(vis_path)
            
        for i in tqdm(range(self.num_samples)):
            # Generate random targets for this sample
            # Each target has range, velocity, and RCS
            targets = self.generate_targets()
            
            # Calculate SNR for this sample (randomized between min and max)
            snr_db = random.uniform(self.SNR_dB_min, self.SNR_dB_max)
            
            # Visualize Transmit (TX) Signal for the first sample
            if self.drawfig and i == 0 and VISUALIZATION_AVAILABLE and self.signal_type == 'FMCW':
                # Generate ideal TX signal for visualization
                t = np.linspace(0, self.T, int(self.fs * self.T))
                tx_signal = np.cos(2 * np.pi * (self.fc * t + 0.5 * self.slope * t**2))
                
                # 1. Visualize time domain and spectrum
                plot_signal_time_and_spectrum(
                    signal=tx_signal,
                    sample_rate=self.fs,
                    total_duration=self.T,
                    title_prefix="TX Chirp",
                    textstr=f"fc={self.fc/1e9:.1f}GHz, B={self.B/1e6:.1f}MHz",
                    normalize=False,
                    save_path=os.path.join(vis_path, f"tx_chirp_{i}.png"),
                    draw_window=False
                )
                
                # 2. Visualize instantaneous frequency
                plot_instantaneous_frequency(
                    signal=tx_signal,
                    sample_rate=self.fs,
                    total_duration=self.T,
                    slope=self.slope,
                    bandwidth=self.B,
                    center_freq=self.fc,
                    title_prefix="TX Chirp",
                    textstr=f"Bandwidth: {self.B/1e6:.1f} MHz\nSlope: {self.slope/1e12:.2f} THz/s",
                    save_path=os.path.join(vis_path, f"tx_chirp_freq_{i}.png")
                )
            
            # Simulate Signal based on type
            if self.signal_type == 'OTFS':
                beat_signal, rdm = self.simulate_otfs_signal(targets, snr_db)
            else:
                # FMCW
                beat_signal, rdm = self.simulate_fmcw_signal(targets, snr_db)
            
            # Additional visualizations for beat signal
            if self.drawfig and i < 3 and VISUALIZATION_AVAILABLE:
                # 3. Visualize beat signal (first chirp/symbol)
                beat_chirp = beat_signal[0, :]
                plot_signal_time_and_spectrum(
                     signal=beat_chirp,
                     sample_rate=self.fs,
                     total_duration=self.T,
                     title_prefix="Beat Signal",
                     textstr=None,
                     normalize=False,
                     save_path=os.path.join(vis_path, f"beat_signal_{i}.png"),
                     draw_window=False
                 )
            
            # Store time domain data (real and imaginary parts)
            self.time_domain_data[i, :, :, 0] = beat_signal.real.astype(self.precision)
            self.time_domain_data[i, :, :, 1] = beat_signal.imag.astype(self.precision)
            
            # Store range-doppler map
            self.range_doppler_maps[i] = rdm.astype(self.precision)
            
            # Create and store target mask
            mask = self.create_target_mask(targets)
            self.target_masks[i, :, :, 0] = mask.astype(self.precision)
            
            # Perform CFAR detection and store results
            cfar_results = self.cfar_detection(rdm)
            self.cfar_detections.append(cfar_results)
            
            # Store target information
            self.target_info.append({
                'targets': targets,
                'snr_db': snr_db,
                'sample_idx': i,
                'cfar_detections': cfar_results
            })
            
            # Generate comprehensive visualization for first few samples
            if self.drawfig and i < 5:
                # Pass pre-computed detections to avoid re-calculation
                self.plot_sample(i, targets, rdm, vis_path, detections=cfar_results)
                
                # Add 3D visualization using external function
                if VISUALIZATION_AVAILABLE:
                    # Convert RDM to format expected by visualization function
                    rdm_complex = rdm + 0j  # Convert to complex for compatibility
                    rdm_formatted = np.stack([rdm_complex.real, rdm_complex.imag], axis=0)
                    
                    # Convert target format for compatibility with external function
                    converted_targets = []
                    for target in targets:
                        converted_target = target.copy()
                        converted_target['distance'] = target['range']  # Add 'distance' key
                        converted_targets.append(converted_target)
                    
                    plot_3d_range_doppler_map_with_ground_truth(
                        rd_map=rdm_formatted,
                        targets=converted_targets,
                        range_resolution=self.range_resolution,
                        velocity_resolution=self.velocity_resolution,
                        num_range_bins=self.num_range_bins,
                        num_doppler_bins=self.num_doppler_bins,
                        save_path=os.path.join(vis_path, f"rd3d_map_{i}.png"),
                        apply_doppler_centering=True
                    )
        
        print(f"Dataset generation complete. Saving to {self.save_path}")
        self.save_dataset()
    
    def plot_sample(self, sample_idx, targets, rdm, save_dir, detections=None):
        """
        Plot range-doppler map with target annotations (2D) using CFAR detection results.
        Includes comprehensive metric evaluation (TP/FP, Position/Velocity Accuracy).
        
        Args:
            sample_idx: Index of the sample
            targets: Ground truth targets
            rdm: Range-Doppler Map (dB)
            save_dir: Directory to save plot
            detections: Optional pre-computed CFAR detections
        """
        # 2D Range-Doppler Map with more space for legend
        fig, ax = plt.subplots(figsize=(20, 10))  # Increased figure size
        
        # Plot range-doppler map
        dr = self.range_axis[1] - self.range_axis[0]
        dv = self.velocity_axis[1] - self.velocity_axis[0]
        extent = [self.range_axis[0] - dr/2, self.range_axis[-1] + dr/2,
                  self.velocity_axis[0] - dv/2, self.velocity_axis[-1] + dv/2]
        
        im = ax.imshow(rdm, extent=extent, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label="Magnitude (dB)")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title(f"Range-Doppler Map with CFAR Detection - Sample {sample_idx}")
        
        # CFAR Detection for multiple targets (use provided or compute)
        detection_results = detections if detections is not None else self.cfar_detection(rdm)
        
        # Define colors for different targets and detections (needed for 3D plot)
        target_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        detection_colors = ['yellow', 'cyan', 'magenta', 'lime', 'gold', 'coral', 'lightblue', 'lightgreen']
        
        # --- Metric Evaluation Logic ---
        tp = 0
        fp = 0
        fn = 0
        range_errors = []
        velocity_errors = []
        
        # Matching thresholds
        match_dist_thresh = 3.0  # meters (Euclidean distance in R-V space might need scaling, but simple separate checks work)
        
        # Clone lists to keep track of unmatched
        unmatched_targets = targets.copy()
        unmatched_detections = detection_results.copy()
        matched_pairs = []
        
        # Greedy matching
        for target in targets:
            best_det = None
            best_dist = float('inf')
            best_det_idx = -1
            
            for i, det in enumerate(unmatched_detections):
                # Calculate distance (normalized or raw)
                # Here using raw Euclidean distance, but could weigh range/doppler differently
                d_r = target['range'] - det['range_m']
                d_v = target['velocity'] - det['velocity_mps']
                dist = np.sqrt(d_r**2 + d_v**2)
                
                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det = det
                    best_det_idx = i
            
            if best_det:
                # Found a match (TP)
                tp += 1
                range_errors.append(abs(target['range'] - best_det['range_m']))
                velocity_errors.append(abs(target['velocity'] - best_det['velocity_mps']))
                matched_pairs.append((target, best_det))
                
                # Remove from unmatched lists
                unmatched_targets.remove(target)
                unmatched_detections.pop(best_det_idx)
            else:
                # No match found (FN)
                fn += 1
                
        # Remaining detections are False Positives
        fp = len(unmatched_detections)
        
        # Calculate aggregate metrics
        mean_range_error = np.mean(range_errors) if range_errors else 0.0
        mean_velocity_error = np.mean(velocity_errors) if velocity_errors else 0.0
        
        # --- Visualization ---
        
        legend_elements = []
        
        # 1. Plot Ground Truth (Matched vs Missed)
        for target in matched_pairs:
            t = target[0]
            ax.scatter(t['range'], t['velocity'], facecolors='none', edgecolors='lime', s=150, linewidth=2, label='Matched GT')
            # Draw line to detection
            d = target[1]
            ax.plot([t['range'], d['range_m']], [t['velocity'], d['velocity_mps']], 'w--', alpha=0.5)
            
        for target in unmatched_targets:
            ax.scatter(target['range'], target['velocity'], facecolors='none', edgecolors='red', s=150, linewidth=2, label='Missed GT (FN)')
            
        # 2. Plot Detections (TP vs FP)
        for pair in matched_pairs:
            d = pair[1]
            ax.scatter(d['range_m'], d['velocity_mps'], marker='x', color='cyan', s=100, linewidth=2, label='True Positive (TP)')
            
        for det in unmatched_detections:
            ax.scatter(det['range_m'], det['velocity_mps'], marker='x', color='orange', s=100, linewidth=2, label='False Alarm (FP)')
            
        # Deduplicate legend labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # Add Metrics to Legend
        legend_elements.extend(by_label.values())
        
        # Create a metrics summary text
        metrics_text = (
            f"Evaluation Metrics:\n"
            f"-------------------\n"
            f"Targets: {len(targets)}\n"
            f"Detections: {len(detection_results)}\n"
            f"TP: {tp} | FP: {fp} | FN: {len(unmatched_targets)}\n"
            f"Range Error (MAE): {mean_range_error:.2f} m\n"
            f"Vel Error (MAE): {mean_velocity_error:.2f} m/s"
        )
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Move metrics text to top left
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Place legend
        # Move legend to top right, inside the plot if possible, or outside
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # Adjust layout
        # plt.subplots_adjust(right=0.75) # No longer needed if we don't put things outside to the right
        save_path_2d = os.path.join(save_dir, f"rdm_sample_{sample_idx}.png")
        plt.savefig(save_path_2d, dpi=150, bbox_inches='tight')
        plt.close()

        # 3D Range-Doppler Map with CFAR detections
        fig = plt.figure(figsize=(22, 12))  # Increased figure size
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D plot
        R, V = np.meshgrid(self.range_axis, self.velocity_axis)
        
        # Plot 3D surface with high resolution
        # rstride=1, cstride=1 ensures we plot every point
        surf = ax.plot_surface(R, V, rdm, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True, rstride=1, cstride=1)
        
        # Add target markers in 3D with different colors
        legend_elements_3d = []
        for j, target in enumerate(targets):
            color = target_colors[j % len(target_colors)]
            # Find the closest indices for target position
            r_idx = np.argmin(np.abs(self.range_axis - target['range']))
            v_idx = np.argmin(np.abs(self.velocity_axis - target['velocity']))
            z_val = rdm[v_idx, r_idx] + 5  # Slightly above the surface
            
            ax.scatter(target['range'], target['velocity'], z_val,
                      color=color, s=100, marker='o', edgecolors='black', linewidth=1)
            
            # Add to 3D legend
            legend_elements_3d.append(
                plt.Line2D([0], [0], marker='o', color=color, markersize=8,
                          label=f'🎯 Target {j+1}: R={target["range"]:.1f}m, V={target["velocity"]:.1f}m/s')
            )
        
        # Add CFAR detection markers in 3D
        for j, detection in enumerate(detection_results[:5]):  # Limit to 5 detections for clarity
            color = detection_colors[j % len(detection_colors)]
            # Find the closest indices for detection position
            r_idx_det = np.argmin(np.abs(self.range_axis - detection['range_m']))
            v_idx_det = np.argmin(np.abs(self.velocity_axis - detection['velocity_mps']))
            z_val_det = rdm[v_idx_det, r_idx_det] + 8  # Higher above the surface
            
            ax.scatter(detection['range_m'], detection['velocity_mps'], z_val_det, 
                      color=color, s=200, marker='x', edgecolors='black', linewidth=2)
            
            legend_elements_3d.append(
                plt.Line2D([0], [0], marker='x', color=color, markersize=12, linewidth=3,
                          label=f'✅ CFAR Det {j+1}: R={detection["range_m"]:.1f}m, V={detection["velocity_mps"]:.1f}m/s')
            )
        
        # Add detection performance metrics to 3D legend
        if targets and detection_results:
            for i, target in enumerate(targets):
                # Find closest detection to this target
                min_error = float('inf')
                closest_detection = None
                for detection in detection_results:
                    error = np.sqrt((target['range'] - detection['range_m'])**2 + 
                                   (target['velocity'] - detection['velocity_mps'])**2)
                    if error < min_error:
                        min_error = error
                        closest_detection = detection
                
                if closest_detection and min_error < 10:  # Within reasonable range
                    range_error = abs(target['range'] - closest_detection['range_m'])
                    velocity_error = abs(target['velocity'] - closest_detection['velocity_mps'])
                    legend_elements_3d.append(
                        plt.Line2D([0], [0], color='none',
                                  label=f'📏 T{i+1} Errors: ΔR={range_error:.2f}m, ΔV={velocity_error:.2f}m/s')
                    )
        
        # Add detection statistics to 3D legend
        legend_elements_3d.append(
            plt.Line2D([0], [0], color='none',
                      label=f'📊 Detections: {len(detection_results)}, Targets: {len(targets)}')
        )
        
        # Customize 3D plot
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_zlabel('Magnitude (dB)')
        ax.set_title(f'3D Range-Doppler Map with CFAR Detection - Sample {sample_idx}')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Magnitude (dB)')
        
        # Add legend to 3D plot with more space
        if legend_elements_3d:
            ax.legend(handles=legend_elements_3d, bbox_to_anchor=(1.3, 1), loc='upper left', fontsize=9)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        save_path_3d = os.path.join(save_dir, f"rdm_3d_sample_{sample_idx}.png")
        plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved 2D visualization with CFAR: {save_path_2d}")
        print(f"Saved 3D visualization with CFAR: {save_path_3d}")
        print(f"CFAR detected {len(detection_results)} targets vs {len(targets)} ground truth targets")
    
    def save_dataset(self):
        """
        Save the generated dataset to HDF5 format
        """
        os.makedirs(self.save_path, exist_ok=True)
        
        save_file = os.path.join(self.save_path, "radar_dataset.h5")
        
        with h5py.File(save_file, 'w') as f:
            # Save data arrays
            f.create_dataset('time_domain_data', data=self.time_domain_data, compression='gzip')
            f.create_dataset('range_doppler_maps', data=self.range_doppler_maps, compression='gzip')
            f.create_dataset('target_masks', data=self.target_masks, compression='gzip')
            
            # Save axes
            f.create_dataset('range_axis', data=self.range_axis)
            f.create_dataset('velocity_axis', data=self.velocity_axis)
            
            # Save parameters
            f.attrs['fc'] = self.fc
            f.attrs['B'] = self.B
            f.attrs['T_chirp'] = self.T
            f.attrs['N_samples'] = self.Ns
            f.attrs['N_chirps'] = self.Nc
            f.attrs['R_max'] = self.R_max
            f.attrs['fs'] = self.fs
            f.attrs['range_resolution'] = self.range_resolution
            f.attrs['velocity_resolution'] = self.velocity_resolution
            
            # Save target info as JSON strings
            import json
            target_info_str = [json.dumps(info, default=str) for info in self.target_info]
            f.create_dataset('target_info', data=target_info_str, dtype=h5py.string_dtype())
        
        print(f"Dataset saved to: {save_file}")
    
    def _load_data(self, datapath):
        """
        Load existing dataset from HDF5 file
        """
        with h5py.File(datapath, 'r') as f:
            self.time_domain_data = f['time_domain_data'][:]
            self.range_doppler_maps = f['range_doppler_maps'][:]
            self.target_masks = f['target_masks'][:]
            
            # Load target info
            import json
            target_info_str = f['target_info'][:]
            self.target_info = [json.loads(info) for info in target_info_str]
            
            # Extract CFAR detections from target_info if available
            self.cfar_detections = []
            for info in self.target_info:
                if 'cfar_detections' in info:
                    self.cfar_detections.append(info['cfar_detections'])
                else:
                    self.cfar_detections.append([])
        
        print(f"Loaded dataset with {len(self.target_info)} samples")
        self.num_samples = len(self.target_info)
    
    def __len__(self):
        return self.num_samples if self.range_doppler_maps is not None else 0
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Sample index (int)
        
        Returns:
            Dictionary containing:
                - time_domain: Complex time domain radar signals
                  Shape: [num_rx, num_chirps, samples_per_chirp] (torch.Tensor)
                - range_doppler_map: Range-Doppler magnitude map in dB
                  Shape: [doppler_bins, range_bins] (torch.Tensor)
                - target_mask: Binary ground truth target mask
                  Shape: [doppler_bins, range_bins, 1] (torch.Tensor)
                - target_info: Dictionary with target metadata including:
                  * targets: List of target dictionaries with range, velocity, RCS
                  * snr_db: Signal-to-noise ratio in dB
                  * sample_idx: Sample index
                  * cfar_detections: List of CFAR detection results
                - cfar_detections: List of CFAR detection dictionaries, each containing:
                  * range_idx: Range bin index (int)
                  * doppler_idx: Doppler bin index (int)
                  * range_m: Range in meters (float)
                  * velocity_mps: Velocity in m/s (float)
                  * angle_deg: Angle of arrival in degrees (float or None)
                  * magnitude: Detection magnitude (float)
                - range_axis: Range axis values in meters
                  Shape: [range_bins] (numpy.ndarray)
                - velocity_axis: Velocity axis values in m/s
                  Shape: [doppler_bins] (numpy.ndarray)
        """
        if self.range_doppler_maps is None:
            raise ValueError("Dataset not generated or loaded")
        
        return {
            'time_domain': torch.from_numpy(self.time_domain_data[idx]).float(),
            'range_doppler_map': torch.from_numpy(self.range_doppler_maps[idx]).float(),
            'target_mask': torch.from_numpy(self.target_masks[idx]).float(),
            'target_info': self.target_info[idx],
            'cfar_detections': self.cfar_detections[idx] if hasattr(self, 'cfar_detections') and idx < len(self.cfar_detections) else [],
            'range_axis': self.range_axis,
            'velocity_axis': self.velocity_axis
        }


if __name__ == "__main__":
    # Test the corrected dataset with multiple configurations
    
    # --- 1. Generate Data for Config 1 (77 GHz) ---
    print("\n" + "="*50)
    print("Generating Data for Config 1 (77 GHz Automotive)")
    print("="*50)
    
    dataset_c1 = AIRadarDataset(
        num_samples=2,
        config_name='config1',
        drawfig=True,
        save_path='data/radar_config1_test'
    )
    
    print(f"\nConfig 1 Dataset created with {len(dataset_c1)} samples")
    sample_c1 = dataset_c1[0]
    print(f"Config 1 Sample 0 shapes:")
    print(f"  Time domain: {sample_c1['time_domain'].shape}")
    print(f"  Range-Doppler map: {sample_c1['range_doppler_map'].shape}")
    print(f"  Target mask: {sample_c1['target_mask'].shape}")
    print(f"  Number of targets: {len(sample_c1['target_info']['targets'])}")
    
    # --- 2. Generate Data for Config 2 (10 GHz) ---
    print("\n" + "="*50)
    print("Generating Data for Config 2 (10 GHz X-Band)")
    print("="*50)
    
    dataset_c2 = AIRadarDataset(
        num_samples=2,
        config_name='config2',
        drawfig=True,
        save_path='data/radar_config2_test'
    )
    
    print(f"\nConfig 2 Dataset created with {len(dataset_c2)} samples")
    sample_c2 = dataset_c2[0]
    print(f"Config 2 Sample 0 shapes:")
    print(f"  Time domain: {sample_c2['time_domain'].shape}")
    print(f"  Range-Doppler map: {sample_c2['range_doppler_map'].shape}")
    print(f"  Target mask: {sample_c2['target_mask'].shape}")
    print(f"  Number of targets: {len(sample_c2['target_info']['targets'])}")
    print(f"  CFAR Detections: {len(sample_c2['cfar_detections'])}")
    
    # --- 3. Generate Data for Config OTFS ---
    print("\n" + "="*50)
    print("Generating Data for Config OTFS (77 GHz)")
    print("="*50)
    
    dataset_otfs = AIRadarDataset(
        num_samples=2,
        config_name='config_otfs',
        drawfig=True,
        save_path='data/radar_otfs_test'
    )
    
    print(f"\nConfig OTFS Dataset created with {len(dataset_otfs)} samples")
    sample_otfs = dataset_otfs[0]
    print(f"Config OTFS Sample 0 shapes:")
    print(f"  Time domain: {sample_otfs['time_domain'].shape}")
    print(f"  Range-Doppler map: {sample_otfs['range_doppler_map'].shape}")
    print(f"  Target mask: {sample_otfs['target_mask'].shape}")
    print(f"  Number of targets: {len(sample_otfs['target_info']['targets'])}")
    print(f"  CFAR Detections: {len(sample_otfs['cfar_detections'])}")

    print("\n" + "="*50)
    print("Generation Complete. Visualizations saved to:")
    print("  - data/radar_config1_test/")
    print("  - data/radar_config2_test/")
    print("  - data/radar_otfs_test/")
    print("==================================================")
