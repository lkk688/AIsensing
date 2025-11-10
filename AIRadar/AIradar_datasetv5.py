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


class AIRadarDataset(Dataset):
    def __init__(self, 
                 num_samples=100,
                 fc=77e9,                    # Center frequency (77 GHz)
                 B=150e6,                    # Bandwidth (150 MHz)
                 T_chirp=20e-6,              # Chirp duration (20 μs)
                 N_samples=1024,             # Samples per chirp
                 N_chirps=128,               # Number of chirps
                 R_max=100,                  # Maximum range (100 m)
                 SNR_dB_min=20,              # Minimum SNR
                 SNR_dB_max=40,              # Maximum SNR
                 zero_pad_factor=8,          # Zero padding factor
                 max_targets=3,              # Maximum targets per sample
                 save_path='data/radar_corrected',
                 precision='float32',
                 drawfig=False,
                 datapath=None):
        """
        Initialize corrected radar dataset with proper FMCW parameters
        
        Args:
            num_samples: Number of samples to generate
            fc: Center frequency in Hz (77 GHz for automotive radar)
            B: Bandwidth in Hz (150 MHz)
            T_chirp: Chirp duration in seconds (20 μs)
            N_samples: Number of samples per chirp (1024)
            N_chirps: Number of chirps per frame (128)
            R_max: Maximum detection range in meters (100 m)
            SNR_dB_min: Minimum SNR in dB
            SNR_dB_max: Maximum SNR in dB
            zero_pad_factor: Zero padding factor for FFT
            max_targets: Maximum number of targets per sample
            save_path: Path to save generated data
            precision: Data precision ('float32' or 'float16')
            drawfig: Whether to generate visualization plots
            datapath: Path to existing dataset (if loading)
        """
        
        # Store basic parameters
        self.num_samples = num_samples
        self.fc = fc
        self.B = B
        self.T = T_chirp
        self.Ns = N_samples
        self.Nc = N_chirps
        self.R_max = R_max
        self.SNR_dB_min = SNR_dB_min
        self.SNR_dB_max = SNR_dB_max
        self.max_targets = max_targets
        self.save_path = save_path
        self.drawfig = drawfig
        self.precision = precision
        
        # Calculate derived parameters
        self.lambda_c = c / fc
        self.slope = B / T_chirp
        self.zero_pad = zero_pad_factor * N_samples
        
        # ✅ Sampling rate from max range (anti-aliasing)
        self.fs = int(np.ceil((4 * B * R_max) / (T_chirp * c)))
        
        # ✅ Doppler max velocity
        self.v_max = self.lambda_c / (4 * self.T)
        
        # Time vectors
        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T
        
        # ✅ Range axis using correct beat-to-range conversion
        range_res = (c * self.fs) / (2 * self.slope * self.zero_pad)
        self.range_axis = np.arange(self.zero_pad // 2) * range_res
        
        # ✅ Velocity axis from Doppler FFT
        self.velocity_axis = np.fft.fftshift(np.fft.fftfreq(self.Nc, d=self.T)) * self.lambda_c / 2
        
        # Calculate system parameters
        self.range_resolution = c / (2 * B)
        self.velocity_resolution = self.lambda_c / (2 * self.Nc * self.T)
        self.max_unambiguous_velocity = self.lambda_c / (4 * self.T)
        
        # Set dimensions for dataset
        self.num_range_bins = self.zero_pad // 2
        self.num_doppler_bins = self.Nc
        
        print("\n=== Corrected Radar System Parameters ===")
        print(f"✅ Center Frequency    : {self.fc/1e9:.1f} GHz")
        print(f"✅ Bandwidth           : {self.B/1e6:.1f} MHz")
        print(f"✅ Chirp Duration      : {self.T*1e6:.1f} μs")
        print(f"✅ Sample Rate         : {self.fs/1e6:.1f} MHz")
        print(f"✅ Maximum Range       : {self.R_max:.1f} m")
        print(f"✅ Range Resolution    : {self.range_resolution:.2f} m")
        print(f"✅ Maximum Velocity    : {self.v_max:.1f} m/s")
        print(f"✅ Velocity Resolution : {self.velocity_resolution:.2f} m/s")
        print(f"✅ Samples per Chirp   : {self.Ns}")
        print(f"✅ Number of Chirps    : {self.Nc}")
        print(f"✅ Range Bins          : {self.num_range_bins}")
        print(f"✅ Doppler Bins        : {self.num_doppler_bins}")
        print("========================================\n")
        
        # Initialize storage for generated data
        self.time_domain_data = None      # Shape: [num_samples, num_rx, num_chirps, samples_per_chirp] - Complex time domain signals
        self.range_doppler_maps = None    # Shape: [num_samples, num_doppler_bins, num_range_bins] - Range-Doppler magnitude maps
        self.target_masks = None          # Shape: [num_samples, num_doppler_bins, num_range_bins, 1] - Binary target masks
        self.target_info = None           # List of dictionaries containing target metadata
        self.cfar_detections = None       # Shape: [num_samples] - List of CFAR detection results per sample
        
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
    
    def simulate_fmcw_signal(self, targets, snr_db):
        """
        Simulate FMCW radar signal with multiple targets
        
        Args:
            targets: List of target dictionaries
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Tuple of (beat_signal, range_doppler_map)
        """
        # Initialize beat signal
        beat = np.zeros((self.Nc, self.Ns), dtype=complex)
        
        # Add each target's contribution
        for target in targets:
            R_true = target['range']
            v_true = target['velocity']
            rcs_linear = 10 ** (target['rcs'] / 10)
            
            # Calculate beat frequencies
            fb = 2 * R_true * self.slope / c
            fd = 2 * v_true / self.lambda_c
            
            # Generate target signal with amplitude based on RCS
            amplitude = np.sqrt(rcs_linear)
            target_signal = amplitude * np.exp(1j * 2 * np.pi * (
                fb * self.t_fast[None, :] + fd * self.t_slow[:, None]
            ))
            
            beat += target_signal
        
        # Apply Hann window
        window = np.hanning(self.Ns)
        beat *= window[None, :]
        
        # Add AWGN
        power = np.mean(np.abs(beat)**2)
        snr_linear = 10 ** (snr_db / 10)
        noise = (np.random.randn(*beat.shape) + 1j * np.random.randn(*beat.shape)) * np.sqrt(power / (2 * snr_linear))
        beat += noise
        
        # Perform FFTs to generate Range-Doppler Map
        range_fft = np.fft.fft(beat, n=self.zero_pad, axis=1)
        range_fft = range_fft[:, :self.zero_pad // 2]
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        
        # Convert to magnitude in dB
        rdm = 20 * np.log10(np.abs(doppler_fft) + 1e-6)
        
        return beat, rdm
    
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
    
    def _cfar_2d_custom(self, rd_map, num_train=8, num_guard=4, range_res=0.5, 
                       doppler_res=0.25, max_range=100, max_speed=50, 
                       threshold_offset=4, nms_kernel_size=3):
        """
        Custom CFAR implementation with adjustable threshold offset
        
        Args:
            rd_map: Input range-Doppler map [num_rx, 2, num_doppler, num_range]
            num_train: Number of training cells per side
            num_guard: Number of guard cells around CUT
            range_res: Range resolution in meters per bin
            doppler_res: Doppler resolution in m/s per bin
            max_range: Maximum detection range in meters
            max_speed: Maximum absolute speed in m/s
            threshold_offset: Threshold offset in dB (lower = more sensitive)
            nms_kernel_size: Non-maximum suppression kernel size
            
        Returns:
            List of detection dictionaries
        """
        from scipy.signal import convolve2d
        from scipy.ndimage import maximum_filter
        
        num_rx, _, num_doppler, num_range = rd_map.shape
        
        # Convert to magnitude in dB
        real = rd_map[:, 0]
        imag = rd_map[:, 1]
        complex_map = real + 1j * imag
        mag = np.abs(complex_map).mean(axis=0)
        mag_db = 20 * np.log10(mag + 1e-12)
        
        # CFAR window setup
        k = num_guard + num_train
        window_size = 2 * k + 1
        full_kernel = np.ones((window_size, window_size), dtype=np.float32)
        guard_area = np.zeros_like(full_kernel)
        guard_area[num_train:num_train + 2*num_guard + 1,
                   num_train:num_train + 2*num_guard + 1] = 1
        train_kernel = full_kernel - guard_area
        
        # Greatest-Of CFAR
        horiz_kernel = train_kernel.copy()
        horiz_kernel[num_train:num_train + 2*num_guard + 1, :] = 0
        vert_kernel = train_kernel.copy()
        vert_kernel[:, num_train:num_train + 2*num_guard + 1] = 0
        
        noise_h = convolve2d(mag_db, horiz_kernel / np.sum(horiz_kernel), 
                            mode='same', boundary='symm')
        noise_v = convolve2d(mag_db, vert_kernel / np.sum(vert_kernel), 
                            mode='same', boundary='symm')
        noise_est = np.maximum(noise_h, noise_v)
        
        # Apply adjustable threshold offset
        threshold = noise_est + threshold_offset
        detections = mag_db > threshold
        
        # Non-maximum suppression
        if nms_kernel_size > 1:
            local_max = maximum_filter(mag_db, size=nms_kernel_size)
            detections &= (mag_db == local_max)
        
        doppler_idxs, range_idxs = np.where(detections)
        results = []
        
        for d_idx, r_idx in zip(doppler_idxs, range_idxs):
            range_m = r_idx * range_res
            velocity_mps = (d_idx - num_doppler // 2) * doppler_res
            
            # Relaxed range filtering - allow closer targets
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
            rd_map: Range-Doppler magnitude map
                   Shape: [doppler_bins, range_bins] - 2D magnitude array
        
        Returns:
            detection_results: List of detection dictionaries, each containing:
                - range_idx: int - Range bin index
                - doppler_idx: int - Doppler bin index  
                - range_m: float - Range in meters
                - velocity_mps: float - Velocity in m/s
                - angle_deg: float or None - Angle of arrival in degrees (if available)
                - magnitude: float - Detection magnitude
        """
        if not CFAR_AVAILABLE:
            # Fallback to simple peak detection
            i_peak, j_peak = np.unravel_index(np.argmax(rd_map), rd_map.shape)
            R_det = self.range_axis[j_peak]
            v_det = self.velocity_axis[i_peak]
            detection_results = [{
                'range_idx': j_peak,
                'doppler_idx': i_peak,
                'range_m': R_det, 
                'velocity_mps': v_det, 
                'angle_deg': None,
                'magnitude': rd_map[i_peak, j_peak]
            }]
            return detection_results
        
        try:
            # Prepare RD map for CFAR (add receive antenna and channel dimensions)
            # CFAR expects shape: [num_rx, 2, num_doppler, num_range]
            # We simulate single RX with real channel only
            rd_map_cfar = np.zeros((1, 2, rd_map.shape[0], rd_map.shape[1]), dtype=np.float32)
            rd_map_cfar[0, 0, :, :] = rd_map  # Real channel
            rd_map_cfar[0, 1, :, :] = 0       # Imaginary channel (zeros)
            
            # Calculate resolution parameters
            range_res = self.range_axis[1] - self.range_axis[0]  # meters per range bin
            velocity_res = self.velocity_axis[1] - self.velocity_axis[0]  # m/s per velocity bin
            
            # Perform CFAR detection with more sensitive parameters
            cfar_results = self._cfar_2d_custom(
                rd_map_cfar,
                num_train=8,        # Number of training cells per side
                num_guard=4,        # Number of guard cells around CUT
                range_res=range_res,
                doppler_res=velocity_res,
                max_range=self.R_max,
                max_speed=50,       # Maximum speed in m/s
                threshold_offset=4, # More sensitive threshold (was 12 dB)
                nms_kernel_size=3   # Non-maximum suppression kernel
            )
            
            # Add magnitude information to CFAR results
            for detection in cfar_results:
                d_idx = detection['doppler_idx']
                r_idx = detection['range_idx']
                if 0 <= d_idx < rd_map.shape[0] and 0 <= r_idx < rd_map.shape[1]:
                    detection['magnitude'] = rd_map[d_idx, r_idx]
                else:
                    detection['magnitude'] = 0.0
            
            return cfar_results
            
        except Exception as e:
            print(f"CFAR detection failed: {e}")
            # Fallback to simple peak detection
            i_peak, j_peak = np.unravel_index(np.argmax(rd_map), rd_map.shape)
            R_det = self.range_axis[j_peak]
            v_det = self.velocity_axis[i_peak]
            detection_results = [{
                'range_idx': j_peak,
                'doppler_idx': i_peak,
                'range_m': R_det, 
                'velocity_mps': v_det, 
                'angle_deg': None,
                'magnitude': rd_map[i_peak, j_peak]
            }]
            return detection_results
    
    def generate_dataset(self):
        """
        Generate the complete radar dataset with step-by-step visualizations
        """
        print(f"Generating {self.num_samples} radar samples...")
        
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
        
        if self.drawfig:
            vis_path = os.path.join(self.save_path, "visualization")
            os.makedirs(vis_path, exist_ok=True)
        
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            # Generate targets for this sample
            targets = self.generate_targets()
            
            # Generate SNR for this sample
            snr_db = random.uniform(self.SNR_dB_min, self.SNR_dB_max)
            
            # Generate TX signal for visualization
            t_chirp = np.linspace(0, self.T, self.Ns)
            tx_signal = np.exp(1j * 2 * np.pi * (self.fc * t_chirp + 0.5 * self.slope * t_chirp**2))
            
            # Step-by-step visualizations for first few samples
            if self.drawfig and i < 3 and VISUALIZATION_AVAILABLE:
                # 1. Visualize TX chirp signal
                plot_signal_time_and_spectrum(
                    signal=tx_signal,
                    sample_rate=self.fs,
                    total_duration=self.T,
                    title_prefix="TX Chirp",
                    center_freq=self.fc,
                    textstr=None,
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
            
            # Simulate FMCW signal
            beat_signal, rdm = self.simulate_fmcw_signal(targets, snr_db)
            
            # Additional visualizations for beat signal
            if self.drawfig and i < 3 and VISUALIZATION_AVAILABLE:
                # 3. Visualize beat signal (first chirp)
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
                self.plot_sample(i, targets, rdm, vis_path)
                
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
    
    def plot_sample(self, sample_idx, targets, rdm, save_dir):
        """
        Plot range-doppler map with target annotations (2D and 3D) using CFAR detection
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
        
        # CFAR Detection for multiple targets using the new member function
        detection_results = self.cfar_detection(rdm)
        
        # Define colors for different targets and detections
        target_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        detection_colors = ['yellow', 'cyan', 'magenta', 'lime', 'gold', 'coral', 'lightblue', 'lightgreen']
        
        # Plot target ground truth with different colors
        legend_elements = []
        for j, target in enumerate(targets):
            color = target_colors[j % len(target_colors)]
            ax.scatter(target['range'], target['velocity'], 
                      facecolors='none', edgecolors=color, s=120, 
                      linewidth=2)
            
            # Add to legend with target info
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                          markeredgecolor=color, markersize=10, linewidth=2,
                          label=f'🎯 Target {j+1}: R={target["range"]:.1f}m, V={target["velocity"]:.1f}m/s')
            )
        
        # Plot CFAR detection results
        for j, detection in enumerate(detection_results[:5]):  # Limit to 5 detections for clarity
            color = detection_colors[j % len(detection_colors)]
            ax.scatter(detection['range_m'], detection['velocity_mps'], 
                      marker='x', color=color, s=200, linewidth=3)
            
            legend_elements.append(
                plt.Line2D([0], [0], marker='x', color=color, markersize=12, linewidth=3,
                          label=f'✅ CFAR Det {j+1}: R={detection["range_m"]:.1f}m, V={detection["velocity_mps"]:.1f}m/s')
            )
        
        # Calculate detection performance metrics
        if targets and detection_results:
            # Calculate detection errors for each target
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
                    legend_elements.append(
                        plt.Line2D([0], [0], color='none',
                                  label=f'📏 T{i+1} Errors: ΔR={range_error:.2f}m, ΔV={velocity_error:.2f}m/s')
                    )
        
        # Add detection statistics
        legend_elements.append(
            plt.Line2D([0], [0], color='none',
                      label=f'📊 Detections: {len(detection_results)}, Targets: {len(targets)}')
        )
        
        # Place legend on the right side with more space
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.25, 1), loc='upper left', fontsize=10)
        
        # Adjust layout to prevent legend overlap
        plt.subplots_adjust(right=0.75)  # Make room for legend
        save_path_2d = os.path.join(save_dir, f"rdm_sample_{sample_idx}.png")
        plt.savefig(save_path_2d, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3D Range-Doppler Map with CFAR detections
        fig = plt.figure(figsize=(22, 12))  # Increased figure size
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D plot
        R, V = np.meshgrid(self.range_axis, self.velocity_axis)
        
        # Plot 3D surface
        surf = ax.plot_surface(R, V, rdm, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
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
        
        print(f"Loaded dataset with {len(self.target_info)} samples")
    
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
    # Test the corrected dataset
    dataset = AIRadarDataset(
        num_samples=10,
        drawfig=True,
        save_path='data/radar_corrected_test3'
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample 0 shapes:")
    print(f"  Time domain: {sample['time_domain'].shape}")
    print(f"  Range-Doppler map: {sample['range_doppler_map'].shape}")
    print(f"  Target mask: {sample['target_mask'].shape}")
    print(f"  Number of targets: {len(sample['target_info']['targets'])}")