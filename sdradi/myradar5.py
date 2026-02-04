import time
import sys
import os
import numpy as np
from datetime import datetime
from timeit import default_timer as timer
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try Imports
try:
    import adi
    from phaser.mycn0566 import CN0566
    from myadi.aditddn import tddn
    from myadiclass import SDR
    from processing import createcomplexsinusoid
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Hardware libraries missing: {e}")
    HARDWARE_AVAILABLE = False

class RadarDevice:
    """
    Robust Radar Device Driver v5.
    Handles AD9361 reconfiguration, Phaser setup, and TDD framing.
    """
    def __init__(self, 
                 sdrurl, 
                 phaserurl, 
                 sdrdevice_name='ad9361', 
                 sample_rate=2e6, 
                 center_freq=2.1e9, 
                 rxbuffersize=1024*8, 
                 sdr_bandwidth=1e6, 
                 rx_gain=30, 
                 Rx_CHANNEL=2, 
                 Tx_CHANNEL=2,
                 signal_freq=100e3, 
                 chirp_bandwidth=500e6, 
                 output_freq=10.25e9, 
                 ramp_time=500, # us
                 ramp_mode="disabled", 
                 num_chirps=32, 
                 tddmode=False, 
                 savedata=False, 
                 savefolder="output", 
                 savefilename=None):
        
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Cannot initialize RadarDevice: Hardware libraries missing.")

        self.sdrurl = sdrurl
        self.phaserurl = phaserurl
        self.tddmode = tddmode
        self.sample_rate = int(sample_rate)
        self.rxbuffersize = int(rxbuffersize)
        self.savedata = savedata
        
        # Radar Params
        self.center_freq = center_freq
        self.bandwidth = chirp_bandwidth
        self.output_freq = output_freq
        self.ramp_time = ramp_time # us
        self.ramp_time_s = ramp_time / 1e6
        self.num_chirps = num_chirps
        self.num_steps = int(ramp_time) # 1 step per us rule of thumb from myradar4
        
        # Store Data
        self.allrxdata = np.empty(0, dtype=np.complex_)
        self.savefolder = savefolder
        self.savefilename = savefilename if savefilename else f"RadarData_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        
        # --- 1. Initialize SDR ---
        logger.info(f"Initializing SDR at {sdrurl}...")
        try:
            self.mysdr = SDR(SDR_IP=sdrurl, 
                             SDR_FC=int(self.center_freq), 
                             SDR_SAMPLERATE=int(self.sample_rate), 
                             SDR_BANDWIDTH=int(sdr_bandwidth), 
                             Rx_CHANNEL=Rx_CHANNEL, 
                             Tx_CHANNEL=Tx_CHANNEL, 
                             device_name=sdrdevice_name)
            
            # Setup TX (Cyclic Buffer for TDD/Tone)
            # Default to -88 dB (Off) for TX1, 0 dB for TX2 (or as needed)
            # myradar4 used tx1_gain=-88, tx2_gain=0
            self.mysdr.SDR_TX_setup(cyclic_buffer=True, tx1_gain=-88, tx2_gain=0) 
            self.mysdr.SDR_RX_setup(n_SAMPLES=self.rxbuffersize, controlmode='manual', rx1_gain=rx_gain, rx2_gain=rx_gain)
            self.mysdr.sdr._ctx.set_timeout(30000) # 30s timeout
        except Exception as e:
            logger.error(f"Failed to initialize SDR: {e}")
            raise
            
        # --- 2. Initialize Phaser ---
        self.myphaser = None
        if phaserurl:
            logger.info(f"Initializing Phaser at {phaserurl}...")
            try:
                self._init_phaser(phaserurl, tddmode, ramp_mode)
            except Exception as e:
                logger.error(f"Failed to config Phaser: {e}")
                raise
        else:
            logger.warning("No Phaser URL provided. Running in SDR-only mode.")

        # --- 3. TDD Setup ---
        self.tdd = None
        self.good_ramp_samples = 0
        if self.tddmode:
            logger.info("Configuring TDD Framing...")
            self._setup_tdd()
            
        # --- 4. Prepare Transmit Signal ---
        # Generate a tone for the SDR to upconvert (if needed by Phaser approach)
        # myradar4 Line 639: createcomplexsinusoid(fs, self.signal_freq, N)
        self.iq_sig = createcomplexsinusoid(self.sample_rate, signal_freq, self.rxbuffersize)
        # Transmit 0.5 amplitude on CH2, 0 on CH1 (handled by gain=-88)
        self.transmit() 
        
        logger.info("RadarDevice Init Complete.")

    def _init_phaser(self, url, tddmode, ramp_mode):
        # We assume PhaserDevice class logic is embedded here for simplicity or we use CN0566 directly
        # myradar4 uses a helper class PhaserDevice. Let's use CN0566 directly to be cleaner.
        
        vco_freq = int(self.output_freq + 100e3 + self.center_freq) # logic from myradar4
        
        self.cn0566 = CN0566(uri=url, sdr=self.mysdr.sdr)
        self.cn0566.configure(device_mode="rx")
        self.cn0566.load_gain_cal()
        self.cn0566.load_phase_cal()
        
        # Set Beam to 0
        for i in range(8):
            self.cn0566.set_chan_phase(i, 0)
            self.cn0566.set_chan_gain(i, 64) # Half scale
            
        # GPIO Setup
        try:
            self.cn0566._gpios.gpio_tx_sw = 0
            self.cn0566._gpios.gpio_vctrl_1 = 1
            self.cn0566._gpios.gpio_vctrl_2 = 1
        except:
            if hasattr(self.cn0566, 'gpios'):
                self.cn0566.gpios.gpio_tx_sw = 0
                self.cn0566.gpios.gpio_vctrl_1 = 1
                self.cn0566.gpios.gpio_vctrl_2 = 1

        # PLL Setup (ADF4159)
        self.cn0566.frequency = int(vco_freq / 4)
        self.cn0566.freq_dev_range = int(self.bandwidth / 4)
        self.cn0566.freq_dev_step = int((self.bandwidth/4) / self.num_steps)
        self.cn0566.freq_dev_time = int(self.ramp_time)
        self.cn0566.delay_word = 4095
        self.cn0566.delay_clk = "PFD"
        self.cn0566.delay_start_en = 0
        self.cn0566.ramp_delay_en = 0
        self.cn0566.trig_delay_en = 0
        
        if tddmode:
            self.cn0566.ramp_mode = "single_sawtooth_burst"
            self.cn0566.tx_trig_en = 1
        else:
            self.cn0566.ramp_mode = ramp_mode
            self.cn0566.tx_trig_en = 0
            
        self.cn0566.sing_ful_tri = 0
        self.cn0566.enable = 0 
        self.cn0566.element_spacing = 0.014 # Default spacing
        self.myphaser = self.cn0566 # Alias for compatibility

    def steer_angle(self, angle):
        """Steer the beam to the specified angle (degrees)"""
        if not self.myphaser:
            return
            
        # Calculation from myradar4
        # phase_delta = 2 * pi * f * d * sin(theta) / c
        phase_delta = (2 * np.pi * self.output_freq * self.myphaser.element_spacing
            * np.sin(np.radians(angle))
            / 3e8
        )
        self.myphaser.set_beam_phase_diff(np.degrees(phase_delta))

    def _setup_tdd(self):
        # Configure TDD Logic
        # Frame Length = Ramp Time + Overhead
        # myradar4 used 1.2ms overhead. We'll stick to clear math.
        # Overhead allows for retracing and buffer alignment.
        
        overhead_ms = 1.2 # ms
        self.frame_length_ms = (self.ramp_time_s * 1000) + overhead_ms
        
        self.tdd_hw = tddn(self.sdrurl)
        self.sdr_pins = adi.one_bit_adc_dac(self.sdrurl)
        
        # Setup Pins
        self.sdr_pins.gpio_tdd_ext_sync = True
        self.sdr_pins.gpio_phaser_enable = True
        
        # Setup TDD
        self.tdd_hw.enable = False
        self.tdd_hw.sync_external = True
        self.tdd_hw.startup_delay_ms = 1
        self.tdd_hw.frame_length_ms = self.frame_length_ms
        self.tdd_hw.burst_count = self.num_chirps
        
        # Channel Config
        # CH0: RX Activation? CH1/2 control switches?
        # myradar4: 
        # CH0: Enable=True, Polarity=False (Active High?) 
        # CH1: Enable=True
        # CH2: Enable=False
        self.tdd_hw.channel[0].enable = True
        self.tdd_hw.channel[0].polarity = False
        self.tdd_hw.channel[0].on_ms = 0.01
        self.tdd_hw.channel[0].off_ms = 0.1
        
        self.tdd_hw.channel[1].enable = True
        self.tdd_hw.channel[1].polarity = False
        self.tdd_hw.channel[1].on_ms = 0.01
        self.tdd_hw.channel[1].off_ms = 0.1
        
        self.tdd_hw.channel[2].enable = False
        self.tdd_hw.enable = True
        
        # Calculate Buffer Sizes
        # We need to capture ALL chirps in one buffer.
        # Total Time = FrameLength * NumChirps
        total_time_ms = self.frame_length_ms * self.num_chirps
        
        # Required buffer size (power of 2)
        target_size = int((total_time_ms / 1000.0) * self.sample_rate * 1.2) # 20% safety margin
        # Find next power of 2
        power = 12
        while (2**power) < target_size:
            power += 1
            if power >= 23: break # Max limit
            
        self.rxbuffersize = 2**power
        logger.info(f"Refined Buffer Size: {self.rxbuffersize} samples (Target Time: {total_time_ms}ms)")
        
        # Update SDR Buffer
        self.mysdr.sdr.rx_buffer_size = self.rxbuffersize
        
        # Calculate Parsing Parameters
        begin_offset_time = 0.10 * self.ramp_time_s
        self.good_ramp_samples = int((self.ramp_time_s - begin_offset_time) * self.sample_rate)
        
        start_offset_time = self.tdd_hw.channel[0].on_ms/1e3 + begin_offset_time
        self.start_offset_samples = int(start_offset_time * self.sample_rate)
        
        self.num_samples_frame = int(self.frame_length_ms / 1000 * self.sample_rate)
        
        logger.info(f"TDD Params: Samples/Frame={self.num_samples_frame}, GoodSamples={self.good_ramp_samples}, StartOffset={self.start_offset_samples}")

    def transmit(self):
        # Continuous Transmit of IQ signal (Tone)
        if self.iq_sig is not None:
             self.mysdr.SDR_TX_send(SAMPLES=self.iq_sig*0.5, SAMPLES2=self.iq_sig, normalize=True, leadingzeros=0, cyclic=True)

    def receive(self):
        """
        Receive data block.
        Returns: 
             data (complex64), length
        """
        # Ensure TDD is running if needed
        # In burst mode, we might need to trigger? 
        # myradar4 TDD is external sync.
        
        start = timer()
        data = self.mysdr.SDR_RX_receive(combinerule='plus', normalize=False)
        dt = timer() - start
        
        if self.savedata:
            self.allrxdata = np.concatenate((self.allrxdata, data))
            
        return data, len(data)

    def transceiversetup(self, signaltype='sinusoid'):
        # Included for compatibility, but logic moved to init/transmit
        pass

    def stop_device(self):
        logger.info("Stopping RadarDevice...")
        try:
            self.mysdr.SDR_TX_stop()
        except: pass
        
        if self.tddmode and self.tdd_hw:
            self.tdd_hw.enable = False
            
        if self.savedata and len(self.allrxdata) > 0:
            if not os.path.exists(self.savefolder):
                os.makedirs(self.savefolder)
            savepath = os.path.join(self.savefolder, self.savefilename)
            np.save(savepath, self.allrxdata)
            logger.info(f"Saved {len(self.allrxdata)} samples to {savepath}")
