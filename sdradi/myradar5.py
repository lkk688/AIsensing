import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from timeit import default_timer as timer

# Add AIRadar directory to path to import dataset and visualization tools
sys.path.append('/Developer/AIsensing/AIRadar')
try:
    from AIradar_datasetv6 import AIRadarDataset, RADAR_CONFIGS, _plot_2d_rdm, _plot_3d_rdm
    from AIRadarLib.visualization import plot_3d_range_doppler_map_with_ground_truth
except ImportError:
    print("Error: Could not import AIRadarDataset. Make sure /Developer/AIsensing/AIRadar is accessible.")
    sys.exit(1)

# Hardware Imports (only if needed/available)
try:
    import adi
    import phaser.mycn0566 as mycn0566
    from myadi.aditddn import tddn
    from myadiclass import SDR
    HARDWARE_AVAILABLE = True
except ImportError:
    print("Warning: Hardware libraries (adi, phaser, myadi) not found. Running in SIMULATION mode only.")
    HARDWARE_AVAILABLE = False

# --- Configuration Management ---
class RadarConfig:
    def __init__(self, config_name='config_phaser', mode='simulation', sdr_ip="ip:192.168.86.40", phaser_ip="ip:phaser.local:50901"):
        self.mode = mode
        self.config_name = config_name
        
        # Load base config from AIRadarDataset definitions
        if config_name in RADAR_CONFIGS:
            self.params = RADAR_CONFIGS[config_name].copy()
        else:
            print(f"Warning: Config {config_name} not found. Using config_phaser.")
            self.params = RADAR_CONFIGS['config_phaser'].copy()
            
        # Hardware specific defaults (can be overridden)
        self.hardware_params = {
            'sdr_ip': sdr_ip,
            'phaser_ip': phaser_ip,
            'rx_gain': 30,
            'tx_gain': -10,
            'rx_channels': [0, 1],
            'tx_channels': [0, 1],
            'buffer_size': 1024 * 16, # Default buffer size
            'tdd_mode': False,
            'ramp_mode': "continuous_triangular"
        }
        
        # Update params with hardware constraints if needed
        # For example, AD9361 might have limits on fs or bw
        
    def update_hardware_params(self, **kwargs):
        self.hardware_params.update(kwargs)

    def get_dataset_params(self):
        """Returns parameters suitable for AIRadarDataset init"""
        return {
            'config_name': self.config_name,
            'fc': self.params['fc'],
            'B': self.params['B'],
            'T_chirp': self.params['T_chirp'],
            'fs': self.params['fs'],
            'N_chirps': self.params['N_chirps'],
            'R_max': self.params['R_max']
        }

# --- Hardware Interface ---
class RadarHardware:
    def __init__(self, config):
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Hardware libraries not available.")
        
        self.config = config
        self.sdr = None
        self.phaser = None
        self.tdd = None
        
        self._setup_sdr()
        self._setup_phaser()
        if self.config.hardware_params['tdd_mode']:
            self._setup_tdd()
            
    def _setup_sdr(self):
        print(f"Initializing SDR at {self.config.hardware_params['sdr_ip']}...")
        self.sdr = SDR(
            SDR_IP=self.config.hardware_params['sdr_ip'],
            SDR_FC=int(2.1e9), # AD9361 operates at IF (e.g. 2.1 GHz)
            SDR_SAMPLERATE=int(self.config.params['fs']),
            SDR_BANDWIDTH=int(self.config.params['fs']), # Baseband BW
            Rx_CHANNEL=len(self.config.hardware_params['rx_channels']),
            Tx_CHANNEL=len(self.config.hardware_params['tx_channels'])
        )
        
        # Setup Gains
        self.sdr.SDR_RX_setup(
            n_SAMPLES=self.config.hardware_params['buffer_size'],
            controlmode='manual',
            rx1_gain=self.config.hardware_params['rx_gain'],
            rx2_gain=self.config.hardware_params['rx_gain']
        )
        self.sdr.SDR_TX_setup(
            cyclic_buffer=True, 
            tx1_gain=self.config.hardware_params['tx_gain'], 
            tx2_gain=self.config.hardware_params['tx_gain']
        )
        
    def _setup_phaser(self):
        if not self.config.hardware_params['phaser_ip']:
            print("Phaser IP not provided. Skipping Phaser setup.")
            return

        print(f"Initializing Phaser at {self.config.hardware_params['phaser_ip']}...")
        self.phaser = mycn0566.CN0566(
            uri=self.config.hardware_params['phaser_ip'], 
            sdr=self.sdr.sdr
        )
        
        # Configure Phaser (ADAR1000s)
        self.phaser.configure(device_mode="rx") # Start in RX mode
        self.phaser.load_gain_cal()
        self.phaser.load_phase_cal()
        
        # Set Beam to Boresight (0 degrees)
        for i in range(8):
            self.phaser.set_chan_phase(i, 0)
            self.phaser.set_chan_gain(i, 127, apply_cal=True) # Max gain
            
        # Configure PLL (ADF4159)
        # Output Freq = 10 GHz + IF (2.1 GHz) = 12.1 GHz (approx)
        # But we want Radar Operation at ~10GHz.
        # Let's follow myradar4 logic:
        # output_freq = 12.1e9 (PLL out) -> Mix with LO (2.1e9) -> 10 GHz RF
        
        output_freq = 12.1e9 
        BW = self.config.params['B']
        ramp_time = self.config.params['T_chirp'] * 1e6 # in us
        
        self.phaser.frequency = int(output_freq / 4)
        self.phaser.freq_dev_range = int(BW / 4)
        self.phaser.freq_dev_step = int((BW/4) / 1000) # 1000 steps
        self.phaser.freq_dev_time = int(ramp_time)
        self.phaser.ramp_mode = self.config.hardware_params['ramp_mode']
        self.phaser.enable = 0 # Update registers
        
    def _setup_tdd(self):
        print("Configuring TDD mode...")
        self.tdd = tddn(self.config.hardware_params['sdr_ip'])
        self.tdd.enable = False
        self.tdd.sync_external = True
        self.tdd.frame_length_ms = (self.config.params['T_chirp'] * 1000) + 1.0 
        self.tdd.burst_count = self.config.params['N_chirps']
        self.tdd.channel[0].enable = True # TX
        self.tdd.channel[1].enable = True # RX
        self.tdd.enable = True

    def transmit(self):
        # Generate Waveform (if not using internal PLL ramp only)
        # For FMCW with Phaser, the PLL generates the chirp. 
        # The SDR just needs to be active or sending a tone if used as IF.
        # Here we assume Phaser PLL generates the chirp.
        pass

    def receive(self):
        """Receive a block of data"""
        data = self.sdr.SDR_RX_receive(combinerule='plus', normalize=False)
        return data

    def stop(self):
        if self.sdr:
            self.sdr.SDR_TX_stop()
        if self.tdd:
            self.tdd.enable = False

# --- Simulation Interface ---
class RadarSimulation:
    def __init__(self, config):
        self.config = config
        print("Initializing Radar Simulation...")
        self.dataset = AIRadarDataset(
            num_samples=1,
            **self.config.get_dataset_params(),
            drawfig=False
        )
        
    def get_data(self):
        """Generate a single sample and return it in a format compatible with processing"""
        # AIRadarDataset generates data internally. We can trigger a generation or use existing.
        # Let's force generation of 1 sample.
        self.dataset.generate_dataset()
        sample = self.dataset[0]
        
        # Return Time Domain Data and Ground Truth
        # Format: (Rx, Chirps, Samples) or similar.
        # AIRadarDataset returns: [Rx, Nc, Ns, 2] (Real/Imag)
        # We need to convert to complex numpy array for consistency
        td_tensor = sample['time_domain'] # [Rx, Nc, Ns, 2]
        td_numpy = td_tensor.numpy()
        td_complex = td_numpy[..., 0] + 1j * td_numpy[..., 1]
        
        # Flatten or reshape as needed by processor
        # If processor expects [Nc, Ns], we can return that.
        # Let's return the raw complex data and the target info
        return td_complex, sample['target_info']

# --- Processing ---
class RadarProcessor:
    def __init__(self, config):
        self.config = config
        # Initialize AIRadarDataset instance just to access its processing methods
        # We won't use it to generate data, just for its helper functions if needed
        # Or we can implement processing here using the same logic.
        # Reusing AIRadarDataset logic is better.
        self.processor = AIRadarDataset(
            num_samples=0, # No generation
            **self.config.get_dataset_params(),
            drawfig=False
        )
        
    def process_frame(self, raw_data):
        """
        Process raw radar data to Range-Doppler Map.
        Args:
            raw_data: Complex data [Rx, Nc, Ns] or [Nc, Ns]
        """
        # If multiple Rx, take the first one or beamform
        if raw_data.ndim == 3:
            beat_signal = raw_data[0] # Take first Rx channel
        else:
            beat_signal = raw_data
            
        # beat_signal shape should be [Nc, Ns]
        
        # 1. Range FFT
        # Apply Window
        window_range = np.hanning(self.processor.Ns)
        beat_windowed = beat_signal * window_range[None, :]
        
        range_fft = np.fft.fft(beat_windowed, n=self.processor.zero_pad, axis=1)
        range_fft = range_fft[:, :self.processor.zero_pad // 2]
        
        # 2. Doppler FFT
        window_doppler = np.hanning(self.processor.Nc)
        range_fft_windowed = range_fft * window_doppler[:, None]
        
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft_windowed, axis=0), axes=0)
        
        # 3. Magnitude (dB)
        rdm = 20 * np.log10(np.abs(doppler_fft) + 1e-6)
        
        # Crop to R_max
        if rdm.shape[1] > self.processor.num_range_bins:
            rdm = rdm[:, :self.processor.num_range_bins]
            
        return rdm

    def detect(self, rdm):
        """Perform CFAR detection"""
        return self.processor.cfar_detection(rdm)

# --- Main Application ---
def main():
    parser = argparse.ArgumentParser(description="MyRadar 5 - Modular Radar Controller")
    parser.add_argument('--mode', type=str, default='simulation', choices=['hardware', 'simulation'], help='Operation mode')
    parser.add_argument('--config', type=str, default='config_phaser', help='Radar configuration name')
    parser.add_argument('--sdr_ip', type=str, default="ip:192.168.86.40", help='SDR IP address')
    parser.add_argument('--phaser_ip', type=str, default="ip:phaser.local:50901", help='Phaser IP address')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to capture/simulate')
    parser.add_argument('--plot', action='store_true', help='Enable visualization')
    
    args = parser.parse_args()
    
    # 1. Initialize Configuration
    config = RadarConfig(
        config_name=args.config, 
        mode=args.mode,
        sdr_ip=args.sdr_ip,
        phaser_ip=args.phaser_ip
    )
    
    # 2. Initialize Source
    if args.mode == 'hardware':
        if not HARDWARE_AVAILABLE:
            print("Error: Hardware mode requested but libraries missing.")
            return
        source = RadarHardware(config)
    else:
        source = RadarSimulation(config)
        
    # 3. Initialize Processor
    processor = RadarProcessor(config)
    
    # 4. Main Loop
    try:
        for i in range(args.frames):
            print(f"\n--- Frame {i+1}/{args.frames} ---")
            
            # Capture Data
            start_time = timer()
            if args.mode == 'hardware':
                raw_data = source.receive()
                # Hardware data might need reshaping to [Nc, Ns]
                # Assuming receive() returns 1D or 2D stream. 
                # Need to reshape based on N_chirps and N_samples.
                # This part depends heavily on exact hardware buffer format.
                # For now, we assume simulation returns correct shape.
                targets = [] # No ground truth in hardware
            else:
                raw_data, target_info = source.get_data()
                targets = target_info['targets']
                # raw_data is [Rx, Nc, Ns] complex
                
            # Process
            rdm = processor.process_frame(raw_data)
            detections = processor.detect(rdm)
            
            proc_time = timer() - start_time
            print(f"Processing Time: {proc_time*1000:.2f} ms")
            print(f"Detections: {len(detections)}")
            
            # Visualize
            if args.plot:
                save_path = f"output/myradar5_frame_{i}.png"
                os.makedirs("output", exist_ok=True)
                
                # Use AIRadarDataset's plotting logic via processor instance
                # We need to construct metrics/matches if we have ground truth
                if targets:
                    metrics, matched, unmatched_t, unmatched_d = processor.processor._evaluate_metrics(targets, detections)
                    _plot_2d_rdm(processor.processor, rdm, i, metrics, matched, unmatched_t, unmatched_d, save_path)
                else:
                    # Hardware mode (no GT) - Just plot RDM and Detections
                    # We can create dummy metrics/lists
                    dummy_metrics = {'num_targets': 0, 'num_detections': len(detections), 'tp':0, 'fp':len(detections), 'fn':0, 'mean_range_error':0, 'mean_velocity_error':0}
                    _plot_2d_rdm(processor.processor, rdm, i, dummy_metrics, [], [], detections, save_path)
                
                print(f"Saved visualization to {save_path}")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if args.mode == 'hardware':
            source.stop()
            
if __name__ == "__main__":
    main()
