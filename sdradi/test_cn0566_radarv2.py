#!/usr/bin/env python3
"""
CN0566 Phaser FMCW Radar Test Script
=====================================
Tests the CN0566 phaser dev kit for FMCW radar operation.
Uses standard adi library (pyadi-iio) - no custom wrappers needed.

Hardware setup:
  - PlutoSDR (AD9361) connected via USB to host (ip:192.168.2.1)
  - CN0566 Phaser board accessible at ip:phaser.local
  - ADF4159 PLL generates FMCW chirps (×4 multiplier → 10-12 GHz)
  - ADAR1000 beamformers (8-element phased array)
  - PlutoSDR captures beat frequency (IF) at 2 MHz sample rate

Usage:
  python test_cn0566_radar.py --test connectivity     # Just test device connectivity
  python test_cn0566_radar.py --test basic_rx          # Basic RX capture
  python test_cn0566_radar.py --test fmcw              # Full FMCW radar capture + RDM
  python test_cn0566_radar.py --test all               # Run all tests
"""

import sys
import os
import argparse
import time
import numpy as np
import gc
import shutil

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ======================================================================
# Constants
# ======================================================================
c = 3e8  # Speed of light

# CN0566 FMCW Radar Parameters
# Reference: RADAR_FFT_Waterfall.py (ADI reference - EXACT match)
RADAR_PARAMS = {
    'output_freq': 12.1e9,      # VCO frequency (Hz) - PLL input is this ÷ 4
    'fc': 10.0e9,               # RF center frequency (Hz) = output_freq - sdr_lo
    'B': 500e6,                 # Chirp bandwidth (Hz)
    'T_chirp': 1e-3,            # Chirp duration (s) = 1000 us (ADI ref: ramp_time=1e3 us)
    'fs': 0.6e6,                # PlutoSDR sample rate (Hz)
    'N_chirps': 1,              # Chirps per capture (single ramp for basic test)
    'sdr_lo': int(2.1e9),       # PlutoSDR LO frequency
    'rx_gain': 70,              # RX gain (dB) - max for weak indoor signal
    'tx_gain': 0,               # TX gain (dB) for ch1 - 0 dB = max power
    'signal_freq': 100e3,       # IF/TX tone frequency (Hz)
    'num_steps': 1000,          # PLL ramp steps (ADI reference: 1000)
    'fft_size': 1024 * 16,      # Buffer size (16384, ADI reference)
}

# Derived parameters
p = RADAR_PARAMS
p['Ns'] = int(p['fs'] * p['T_chirp'])  # 600 samples per chirp
p['total_samples'] = p['fft_size']      # 16384 for single-frame mode
p['slope'] = p['B'] / p['T_chirp']      # 5e11 Hz/s
p['range_resolution'] = c / (2 * p['B'])  # 0.3 m
p['R_max'] = p['fs'] * c / (4 * p['slope'])  # max unambiguous range
p['v_max'] = c / (4 * p['fc'] * p['T_chirp'])
p['velocity_resolution'] = 2 * p['v_max'] / max(p['N_chirps'], 1)

# Phaser PLL settings
# freq and freq_dev_range are divided by 4 (pre-multiplier)
# freq_dev_step uses BW/num_steps (ADI reference - NOT divided by 4)
p['pll_freq'] = int(p['output_freq'] / 4)              # 3.025 GHz (÷4 multiplier)
p['pll_bw'] = int(p['B'] / 4)                          # 125 MHz (÷4)
p['pll_step'] = int(p['B'] / p['num_steps'])            # 500 kHz (ADI ref: BW/num_steps)
p['pll_time'] = int(p['T_chirp'] * 1e6)                # 1000 microseconds


def print_radar_params():
    """Print computed radar parameters."""
    p = RADAR_PARAMS
    print("\n=== CN0566 FMCW Radar Parameters ===")
    print(f"  Center Frequency    : {p['fc']/1e9:.2f} GHz")
    print(f"  Bandwidth           : {p['B']/1e6:.0f} MHz")
    print(f"  Chirp Duration      : {p['T_chirp']*1e6:.0f} us")
    print(f"  Sample Rate         : {p['fs']/1e6:.1f} MHz")
    print(f"  Samples/Chirp (Ns)  : {p['Ns']}")
    print(f"  Chirps/CPI (Nc)     : {p['N_chirps']}")
    print(f"  Total Samples       : {p['total_samples']}")
    print(f"  Range Resolution    : {p['range_resolution']:.2f} m")
    print(f"  Max Range           : {p['R_max']:.1f} m")
    print(f"  Max Velocity        : {p['v_max']:.2f} m/s")
    print(f"  Velocity Resolution : {p['velocity_resolution']:.2f} m/s")
    print(f"  PLL Frequency       : {p['pll_freq']/1e9:.4f} GHz (×4 = {p['pll_freq']*4/1e9:.4f} GHz)")
    print(f"  PLL BW (÷4)         : {p['pll_bw']/1e6:.1f} MHz")
    print(f"  PLL Step            : {p['pll_step']} Hz")
    print(f"  PLL Ramp Time       : {p['pll_time']} us")
    print("=" * 40)


# ======================================================================
# Test 1: Connectivity
# ======================================================================
def test_connectivity(sdr_ip, phaser_ip):
    """Test basic connectivity to PlutoSDR and CN0566 Phaser."""
    print("\n" + "=" * 60)
    print("  TEST 1: Device Connectivity")
    print("=" * 60)

    # Test PlutoSDR
    print(f"\n[1a] Connecting to PlutoSDR at {sdr_ip}...")
    try:
        import adi
        sdr = adi.ad9361(sdr_ip)
        print(f"  Connected!")
        print(f"  RX LO: {sdr.rx_lo/1e6:.0f} MHz")
        print(f"  TX LO: {sdr.tx_lo/1e6:.0f} MHz")
        print(f"  Sample Rate: {sdr.sample_rate/1e6:.2f} MHz")

        # Check both RX channels
        try:
            sdr.rx_enabled_channels = [0, 1]
            print(f"  RX Channels: 2 (dual channel)")
        except:
            sdr.rx_enabled_channels = [0]
            print(f"  RX Channels: 1 (single channel)")

        sdr_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        sdr_ok = False
        sdr = None

    # Test CN0566 Phaser
    print(f"\n[1b] Connecting to CN0566 Phaser at {phaser_ip}...")
    try:
        from adi.cn0566 import CN0566
        cn0566 = CN0566(uri=phaser_ip, sdr=sdr)
        cn0566.configure(device_mode='rx')

        print(f"  Connected!")
        print(f"  Elements: {cn0566.num_elements}")
        print(f"  Element Spacing: {cn0566.element_spacing*1000:.1f} mm")
        print(f"  Current Frequency: {cn0566.frequency/1e9:.3f} GHz")
        print(f"  Temperatures: {cn0566.temperatures}")

        # Load calibration
        try:
            cn0566.load_gain_cal()
            cn0566.load_phase_cal()
            print(f"  Calibration: Loaded")
        except:
            print(f"  Calibration: Using defaults")

        phaser_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        phaser_ok = False
        cn0566 = None

    if sdr_ok and phaser_ok:
        print(f"\n  RESULT: Both devices connected successfully")
    else:
        print(f"\n  RESULT: Connection issues detected")
        if not sdr_ok:
            print(f"    - PlutoSDR not reachable at {sdr_ip}")
        if not phaser_ok:
            print(f"    - CN0566 Phaser not reachable at {phaser_ip}")

    # Cleanup
    try:
        if cn0566:
            cn0566.enable = 0
    except:
        pass

    return sdr_ok and phaser_ok


# ======================================================================
# Test 2: Basic RX
# ======================================================================
def setup_cn0566_radar(sdr_ip, phaser_ip):
    """
    Set up PlutoSDR + CN0566 Phaser for FMCW radar operation.
    Follows RADAR_FFT_Waterfall.py (ADI reference) exactly.
    Also explicitly sets GPIO pins for LO routing (critical for signal path).

    Returns: (sdr, cn0566) tuple
    """
    import adi
    from adi.cn0566 import CN0566

    p = RADAR_PARAMS
    import time as _time

    # 1. Connect to PlutoSDR and perform hardware initialization/reset
    #    This is critical: setting "initialize" resets the AD9361, which kills the IIO context.
    #    We must do this first, wait, and then re-connect.
    print(f"  [Setup] Connecting to PlutoSDR at {sdr_ip} for initialization...")
    try:
        sdr = adi.ad9361(sdr_ip)
        
        # Set FDD mode (Frequency Division Duplex) - required for simultaneous TX/RX
        sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
        sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"  # Disable pin control
        sdr._ctrl.debug_attrs["initialize"].value = "1"
        print("  [Setup] Initializing AD9361 (FDD mode)...")
        
    except Exception as e:
        print(f"  [Setup] WARNING: Initial reset trigger captured (often expected): {e}")

    # Always wait for the reset to complete, even if the command threw an error
    print("  [Setup] Waiting 5s for re-initialization...")
    _time.sleep(5)
    
    # Destroy the old (broken) object
    try:
        del sdr
    except UnboundLocalError:
        pass
    gc.collect()

    # 1b. Re-connect to PlutoSDR with fresh context
    print(f"  [Setup] Re-connecting to PlutoSDR...")
    try:
        sdr = adi.ad9361(sdr_ip)
    except Exception as e:
        print(f"  [Setup] Connection failed: {e}. Retrying in 2s...")
        _time.sleep(2)
        sdr = adi.ad9361(sdr_ip)

    # 2. Connect to CN0566 Phaser
    cn0566 = CN0566(uri=phaser_ip, sdr=sdr)
    cn0566.configure(device_mode='rx')
    cn0566.load_gain_cal()
    cn0566.load_phase_cal()

    # 3. CRITICAL: Ensure GPIO pins are set for LO routing
    #    These route the onboard PLL/LO to the TX mixer circuitry.
    #    Without these, the PLL chirp has no path to the antenna.
    #    (mycn0566.py sets these in __init__, but older pyadi-iio may not)
    try:
        cn0566._gpios.gpio_vctrl_1 = 1  # Onboard PLL/LO source
        cn0566._gpios.gpio_vctrl_2 = 1  # Send LO to TX circuitry
        cn0566._gpios.gpio_tx_sw = 1    # Route to TX_OUT_1
        print("  GPIO: vctrl_1=1 (PLL source), vctrl_2=1 (LO to TX), tx_sw=1 (TX_OUT_1)")
    except AttributeError:
        # If _gpios not available, create it explicitly
        try:
            gpios = adi.one_bit_adc_dac(phaser_ip)
            gpios.gpio_vctrl_1 = 1
            gpios.gpio_vctrl_2 = 1
            gpios.gpio_tx_sw = 1
            cn0566._gpios = gpios
            print("  GPIO: Created one_bit_adc_dac, vctrl_1=1, vctrl_2=1")
        except Exception as e:
            print(f"  GPIO WARNING: Could not set vctrl GPIOs: {e}")

    # Verify GPIO state
    try:
        v1 = cn0566._gpios.gpio_vctrl_1
        v2 = cn0566._gpios.gpio_vctrl_2
        tr = cn0566._gpios.gpio_tr
        print(f"  GPIO verify: vctrl_1={v1}, vctrl_2={v2}, tr={tr}")
    except Exception as e:
        print(f"  GPIO verify: {e}")

    # 4. Enable TDD phaser synchronization
    #    This synchronizes the Pluto RX buffer capture to the ADF4159 ramp.
    #    Reference: PhaserRadarLabs/FMCW_RADAR_Waterfall_ChirpSync.py
    try:
        sdr_pins = adi.one_bit_adc_dac(sdr_ip)
        sdr_pins.gpio_phaser_enable = True
        print("  TDD: gpio_phaser_enable = True (chirp-RX sync enabled)")
    except Exception as e:
        print(f"  TDD: gpio_phaser_enable not available: {e}")
        print("       (non-critical for continuous_triangular ramp mode)")

    # 5. Configure PlutoSDR parameters
    print("  FDD mode enabled, ENSM pin control disabled")

    # RX configuration
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(p['fs'])
    sdr.rx_lo = int(p['sdr_lo'])
    sdr.rx_buffer_size = int(p['total_samples'])
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = int(p['rx_gain'])
    sdr.rx_hardwaregain_chan1 = int(p['rx_gain'])

    # Set kernel buffers to 1 (no stale data - measure immediately after change)
    sdr._rxadc.set_kernel_buffers_count(1)
    print("  Kernel buffers: 1 (no stale buffers)")

    # Enable quadrature tracking for better IQ balance
    rx_chan = sdr._ctrl.find_channel("voltage0")
    rx_chan.attrs["quadrature_tracking_en"].value = "1"
    print("  Quadrature tracking: enabled")

    # TX configuration
    # NOTE: TX1 (channel 0) is physically connected to the CN0566 phaser on this board.
    # The ADI reference uses TX2 (channel 1), but our hardware uses TX1.
    sdr.tx_lo = int(p['sdr_lo'])
    sdr.tx_enabled_channels = [0, 1]
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = int(p['tx_gain'])  # TX1 ON (connected to phaser)
    sdr.tx_hardwaregain_chan1 = -88                 # TX2 OFF

    # Generate TX CW tone (reference signal for FMCW mixer)
    # Following RADAR_FFT_Waterfall.py exactly
    fs = int(sdr.sample_rate)
    N = int(sdr.rx_buffer_size)
    signal_freq = p['signal_freq']  # 100 kHz
    fc = int(signal_freq / (fs / N)) * (fs / N)  # Snap to FFT bin
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i_sig = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q_sig = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq = 1.0 * (i_sig + 1j * q_sig)

    sdr._ctx.set_timeout(0)
    sdr.tx([iq, iq * 0.5])  # Ch0 (TX1) at full, Ch1 (TX2) at half

    print(f"  TX tone at {fc/1e3:.1f} kHz, TX gain ch0={p['tx_gain']} dB (TX1 active)")

    # 5. Set up all 8 antenna elements
    # Use Blackman taper like the ADI reference for better sidelobe control
    gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
    for i in range(8):
        cn0566.set_chan_phase(i, 0)
        cn0566.set_chan_gain(i, gain_list[i], apply_cal=True)

    # 6. Configure ADF4159 PLL for FMCW chirp generation
    # Following ADI reference: RADAR_FFT_Waterfall.py
    cn0566.frequency = p['pll_freq']
    cn0566.freq_dev_range = p['pll_bw']
    cn0566.freq_dev_step = p['pll_step']
    cn0566.freq_dev_time = p['pll_time']
    cn0566.delay_word = 4095         # 12-bit delay word
    cn0566.delay_clk = "PFD"         # Delay clock source
    cn0566.delay_start_en = 0        # No delay start
    cn0566.ramp_delay_en = 0         # No delay between ramps
    cn0566.trig_delay_en = 0         # No triangle delay
    cn0566.ramp_mode = "continuous_triangular"
    cn0566.sing_ful_tri = 0
    cn0566.tx_trig_en = 0            # No TX trigger
    cn0566.enable = 0                # 0 = PLL ON. Write last to update all registers

    # 7. Read back VTune to verify PLL is locked
    try:
        monitor = cn0566.read_monitor(verbose=False)
        vtune = monitor[8] if len(monitor) > 8 else monitor[-1]
        print(f"  PLL VTune: {vtune:.3f} V (HMC739: 0-13V, ~10V at 12.1GHz)")
    except Exception as e:
        print(f"  PLL VTune read: {e}")

    return sdr, cn0566


def test_basic_rx(sdr_ip, phaser_ip, output_dir):
    """Configure SDR + Phaser and test basic RX capture."""
    print("\n" + "=" * 60)
    print("  TEST 2: Basic RX Capture")
    print("=" * 60)

    p = RADAR_PARAMS
    print_radar_params()

    # -- Configure hardware using proper SDR_init --
    print("\n[2a] Configuring PlutoSDR + CN0566 Phaser...")
    sdr, cn0566 = setup_cn0566_radar(sdr_ip, phaser_ip)

    print(f"  Sample Rate: {sdr.sample_rate/1e6:.2f} MHz")
    print(f"  RX LO: {sdr.rx_lo/1e6:.0f} MHz")
    print(f"  TX LO: {sdr.tx_lo/1e6:.0f} MHz")
    print(f"  Buffer Size: {sdr.rx_buffer_size}")
    print(f"  PLL Frequency: {cn0566.frequency/1e9:.4f} GHz")
    print(f"  TX tone: {p['signal_freq']/1e3:.0f} kHz")

    # -- Flush RX buffer --
    print("\n[2b] Flushing RX buffer...")
    for _ in range(5):
        sdr.rx()

    # -- Capture with PLL ON vs OFF to verify signal path --
    print("\n[2c] Comparing PLL ON vs OFF...")

    # First capture with PLL OFF (powerdown=1) as baseline
    cn0566.enable = 1  # 1 = PLL OFF (powerdown)
    time.sleep(0.3)
    for _ in range(3):
        sdr.rx()  # Flush
    raw_off = sdr.rx()
    if isinstance(raw_off, list):
        data_off = np.array(raw_off)
    else:
        data_off = raw_off[np.newaxis, :]
    pwr_off = 10 * np.log10(np.mean(np.abs(data_off[0])**2) + 1e-12)
    max_off = np.max(np.abs(data_off[0]))
    print(f"  PLL OFF: power={pwr_off:.1f} dB, max_amp={max_off:.0f}")

    # Now enable PLL (powerdown=0) and capture
    cn0566.enable = 0  # 0 = PLL ON
    time.sleep(0.5)  # Extra settle time for PLL lock
    for _ in range(3):
        sdr.rx()  # Flush stale data

    all_data = []
    for i in range(5):
        raw = sdr.rx()
        if isinstance(raw, list):
            data = np.array(raw)  # [num_rx, samples]
        else:
            data = raw[np.newaxis, :]  # [1, samples]

        power_db = 10 * np.log10(np.mean(np.abs(data[0])**2) + 1e-12)
        max_amp = np.max(np.abs(data[0]))
        print(f"  PLL ON Frame {i}: shape={data.shape}, "
              f"power={power_db:.1f} dB, max_amp={max_amp:.0f}")
        all_data.append(data)

    pwr_on = 10 * np.log10(np.mean(np.abs(all_data[-1][0])**2) + 1e-12)
    pwr_diff = pwr_on - pwr_off
    print(f"\n  Power difference (ON-OFF): {pwr_diff:.1f} dB")
    if abs(pwr_diff) < 3.0:
        print("  WARNING: PLL ON/OFF makes little difference!")
        print("  This suggests the FMCW chirp is NOT reaching the antenna.")
        print("  Check: GPIO routing, RF cables, board connections.")
    else:
        print(f"  PLL is affecting signal ({pwr_diff:+.1f} dB) - signal path OK")

    # -- Analyze captured data --
    print("\n[2d] Analyzing captured data...")
    data = all_data[-1]  # Use last frame (most stable)
    ch0 = data[0]  # First RX channel
    ch1 = data[1] if data.shape[0] > 1 else data[0]
    # Sum both channels (phased array coherent sum)
    data_sum = ch0 + ch1

    # Frequency analysis (following ADI reference)
    N_frame = len(data_sum)
    fs = p['fs']
    signal_freq = p['signal_freq']
    slope = p['slope']
    win = np.blackman(N_frame)
    fft_data = np.fft.fft(data_sum * win)
    fft_data = np.fft.fftshift(fft_data)
    magnitude_db = 20 * np.log10(np.abs(fft_data) + 1e-12)

    freqs = np.linspace(-fs / 2, fs / 2, N_frame)
    # Range calculation: beat freq maps to range (factor of 4 for triangular ramp)
    dist = (freqs - signal_freq) * c / (4 * slope)

    # Find peak (in positive frequency region only)
    pos_mask = freqs > 0
    peak_idx = np.argmax(magnitude_db[pos_mask])
    peak_freq = freqs[pos_mask][peak_idx]
    peak_power = magnitude_db[pos_mask][peak_idx]
    peak_range = dist[pos_mask][peak_idx]
    print(f"  Peak frequency: {peak_freq/1e3:.1f} kHz")
    print(f"  Peak power: {peak_power:.1f} dB")
    print(f"  Beat freq offset from signal: {(peak_freq - signal_freq)/1e3:.1f} kHz")
    print(f"  Corresponding range: {peak_range:.1f} m")

    # Plot
    if HAS_MATPLOTLIB:
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('CN0566 Basic RX Test', fontsize=14)

        # Time domain
        t_ms = np.arange(len(ch0)) / p['fs'] * 1000
        axes[0, 0].plot(t_ms[:2000], np.real(ch0[:2000]), 'b-', linewidth=0.5, label='I')
        axes[0, 0].plot(t_ms[:2000], np.imag(ch0[:2000]), 'r-', linewidth=0.5, label='Q')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Time Domain (first 2 chirps)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Range profile (spectrum mapped to distance)
        pos_mask = dist >= 0
        axes[0, 1].plot(dist[pos_mask], magnitude_db[pos_mask], 'b-', linewidth=0.5)
        axes[0, 1].set_xlabel('Range (m)')
        axes[0, 1].set_ylabel('Magnitude (dB)')
        axes[0, 1].set_title('Range Profile')
        axes[0, 1].set_xlim(0, 30)  # Focus on indoor range
        axes[0, 1].grid(True)

        # Power over frames
        frame_powers = [10 * np.log10(np.mean(np.abs(d[0])**2) + 1e-12) for d in all_data]
        axes[1, 0].bar(range(len(frame_powers)), frame_powers)
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Power (dB)')
        axes[1, 0].set_title('RX Power per Frame')
        axes[1, 0].grid(True)

        # Amplitude histogram
        amplitudes = np.abs(ch0)
        axes[1, 1].hist(amplitudes, bins=100, density=True)
        axes[1, 1].set_xlabel('Amplitude')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Amplitude Distribution')
        axes[1, 1].grid(True)

        plt.tight_layout()
        path = os.path.join(output_dir, 'basic_rx_test.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Plot saved: {path}")

    # Cleanup
    del cn0566, sdr
    return True


# ======================================================================
# Test 2b: Spectral Analysis with Multi-Frame Averaging
# ======================================================================
def test_spectral(sdr_ip, phaser_ip, output_dir, num_avg=100):
    """
    Detailed spectral analysis with multi-frame averaging and PLL ON-OFF subtraction.
    This test gives the clearest view of the radar returns by averaging out noise.
    """
    print("\n" + "=" * 60)
    print("  TEST 2b: Spectral Analysis (Multi-Frame Averaged)")
    print("=" * 60)

    p = RADAR_PARAMS
    print_radar_params()

    # -- Setup hardware --
    print(f"\n[2b.1] Setting up hardware...")
    sdr, cn0566 = setup_cn0566_radar(sdr_ip, phaser_ip)

    fs = int(sdr.sample_rate)
    N_frame = int(sdr.rx_buffer_size)
    signal_freq = p['signal_freq']
    slope = p['slope']

    # Frequency and range axes
    freqs = np.linspace(-fs / 2, fs / 2, N_frame)
    dist = (freqs - signal_freq) * c / (4 * slope)

    os.makedirs(output_dir, exist_ok=True)

    # -- Capture with PLL OFF (baseline) --
    print(f"\n[2b.2] Capturing {num_avg} frames with PLL OFF (baseline)...")
    cn0566.enable = 1  # PLL OFF (powerdown=1)
    time.sleep(0.3)
    for _ in range(5):
        sdr.rx()
    avg_off = np.zeros(N_frame)
    for i in range(num_avg):
        raw = sdr.rx()
        data = np.array(raw) if isinstance(raw, list) else raw[np.newaxis, :]
        data_sum = data[0] + (data[1] if data.shape[0] > 1 else data[0])
        win = np.blackman(N_frame)
        fft_data = np.fft.fftshift(np.fft.fft(data_sum * win))
        avg_off += np.abs(fft_data) ** 2  # Power averaging
    avg_off /= num_avg
    avg_off_db = 10 * np.log10(avg_off + 1e-12)
    print(f"  PLL OFF avg power: {avg_off_db.max():.1f} dB peak")

    # -- Capture with PLL ON --
    print(f"\n[2b.3] Capturing {num_avg} frames with PLL ON...")
    cn0566.enable = 0  # PLL ON (powerdown=0)
    time.sleep(0.5)
    for _ in range(5):
        sdr.rx()
    avg_on = np.zeros(N_frame)
    for i in range(num_avg):
        raw = sdr.rx()
        data = np.array(raw) if isinstance(raw, list) else raw[np.newaxis, :]
        data_sum = data[0] + (data[1] if data.shape[0] > 1 else data[0])
        win = np.blackman(N_frame)
        fft_data = np.fft.fftshift(np.fft.fft(data_sum * win))
        avg_on += np.abs(fft_data) ** 2
    avg_on /= num_avg
    avg_on_db = 10 * np.log10(avg_on + 1e-12)
    print(f"  PLL ON avg power: {avg_on_db.max():.1f} dB peak")

    # -- ON - OFF difference (linear domain subtraction) --
    diff_power = np.maximum(avg_on - avg_off, 1e-12)
    diff_db = 10 * np.log10(diff_power + 1e-12)

    # Find peaks in positive-range region
    pos_mask = dist > 0.5  # Skip very close range
    range_mask = dist < 30  # Indoor range limit
    valid = pos_mask & range_mask
    if np.any(valid):
        peak_idx_valid = np.argmax(diff_db[valid])
        valid_indices = np.where(valid)[0]
        peak_idx = valid_indices[peak_idx_valid]
        peak_range = dist[peak_idx]
        peak_power = diff_db[peak_idx]
        peak_freq = freqs[peak_idx]
        print(f"\n  Strongest return (ON-OFF):")
        print(f"    Range: {peak_range:.1f} m")
        print(f"    Beat freq: {peak_freq/1e3:.1f} kHz")
        print(f"    Power: {peak_power:.1f} dB")
        print(f"    ON-OFF margin: {avg_on_db[peak_idx] - avg_off_db[peak_idx]:.1f} dB")

    # Find all peaks above noise floor
    noise_floor = np.median(diff_db[valid])
    peak_threshold = noise_floor + 6  # 6 dB above noise
    peaks_above = np.where((diff_db > peak_threshold) & valid)[0]
    if len(peaks_above) > 0:
        # Group nearby peaks (within 1m)
        grouped = []
        current_group = [peaks_above[0]]
        for idx in peaks_above[1:]:
            if dist[idx] - dist[current_group[-1]] < 1.0:
                current_group.append(idx)
            else:
                grouped.append(current_group)
                current_group = [idx]
        grouped.append(current_group)

        print(f"\n  Detected returns ({len(grouped)} targets, >{peak_threshold:.0f} dB):")
        for gi, group in enumerate(grouped[:10]):
            best = group[np.argmax(diff_db[group])]
            print(f"    Target {gi+1}: R={dist[best]:.1f}m, "
                  f"P={diff_db[best]:.1f}dB, "
                  f"margin={avg_on_db[best]-avg_off_db[best]:.1f}dB")

    # -- Plot --
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'CN0566 Spectral Analysis ({num_avg}-frame avg)', fontsize=14)

        # Range profile: ON, OFF, and difference
        axes[0, 0].plot(dist[valid], avg_on_db[valid], 'b-', linewidth=0.8,
                        alpha=0.7, label='PLL ON')
        axes[0, 0].plot(dist[valid], avg_off_db[valid], 'r-', linewidth=0.8,
                        alpha=0.7, label='PLL OFF')
        axes[0, 0].set_xlabel('Range (m)')
        axes[0, 0].set_ylabel('Power (dB)')
        axes[0, 0].set_title('Range Profile: PLL ON vs OFF')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # ON - OFF difference (radar-only returns)
        axes[0, 1].plot(dist[valid], diff_db[valid], 'g-', linewidth=0.8)
        axes[0, 1].axhline(y=peak_threshold, color='r', linestyle='--',
                           label=f'Threshold ({peak_threshold:.0f} dB)')
        if len(peaks_above) > 0:
            for group in grouped[:10]:
                best = group[np.argmax(diff_db[group])]
                axes[0, 1].plot(dist[best], diff_db[best], 'rv', ms=10)
                axes[0, 1].annotate(f'{dist[best]:.1f}m',
                                    (dist[best], diff_db[best]),
                                    textcoords="offset points", xytext=(5, 10),
                                    fontsize=8)
        axes[0, 1].set_xlabel('Range (m)')
        axes[0, 1].set_ylabel('Power (dB)')
        axes[0, 1].set_title('ON - OFF Difference (Radar Returns)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Full spectrum (frequency domain)
        axes[1, 0].plot(freqs / 1e3, avg_on_db, 'b-', linewidth=0.5,
                        alpha=0.7, label='PLL ON')
        axes[1, 0].plot(freqs / 1e3, avg_off_db, 'r-', linewidth=0.5,
                        alpha=0.7, label='PLL OFF')
        axes[1, 0].set_xlabel('Frequency (kHz)')
        axes[1, 0].set_ylabel('Power (dB)')
        axes[1, 0].set_title('Full Spectrum')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # ON-OFF margin per frequency bin
        margin = avg_on_db - avg_off_db
        axes[1, 1].plot(dist[valid], margin[valid], 'k-', linewidth=0.5)
        axes[1, 1].axhline(y=3, color='r', linestyle='--', label='3 dB')
        axes[1, 1].set_xlabel('Range (m)')
        axes[1, 1].set_ylabel('ON-OFF Margin (dB)')
        axes[1, 1].set_title('PLL ON/OFF Power Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        path = os.path.join(output_dir, 'cn0566_spectral_analysis.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\n  Plot saved: {path}")

    # Save data
    np.savez(
        os.path.join(output_dir, 'cn0566_spectral_data.npz'),
        avg_on_db=avg_on_db,
        avg_off_db=avg_off_db,
        diff_db=diff_db,
        freqs=freqs,
        dist=dist,
        num_avg=num_avg
    )
    print(f"  Data saved: {os.path.join(output_dir, 'cn0566_spectral_data.npz')}")

    cn0566.enable = 1  # PLL OFF
    del cn0566, sdr
    return True


# ======================================================================
# Test 3: Full FMCW Radar (Range-Doppler Map)
# ======================================================================
def test_fmcw_radar(sdr_ip, phaser_ip, output_dir, num_frames=10):
    """
    Full FMCW radar test: capture CPI, compute Range-Doppler Map,
    apply CFAR detection. Static indoor environment.
    """
    print("\n" + "=" * 60)
    print("  TEST 3: FMCW Radar - Range-Doppler Map")
    print("=" * 60)

    p = RADAR_PARAMS
    Ns = p['Ns']  # Samples per chirp (600)
    # For RDM, we need multiple chirps. Compute Nc from buffer size.
    # total_samples = Nc * Ns => Nc = total_samples // Ns
    Nc = max(p['total_samples'] // Ns, 8)  # At least 8 chirps for Doppler
    print_radar_params()
    print(f"  [FMCW] Using Nc={Nc} chirps (buffer={p['total_samples']}, Ns={Ns})")

    # -- Setup hardware --
    print("\n[3a] Setting up hardware...")
    sdr, cn0566 = setup_cn0566_radar(sdr_ip, phaser_ip)

    # Adjust buffer size to match Nc * Ns
    buffer_size = Nc * Ns
    sdr.rx_buffer_size = buffer_size
    print(f"  Sample Rate: {sdr.sample_rate/1e6:.2f} MHz")
    print(f"  Buffer Size: {sdr.rx_buffer_size} (adjusted for {Nc} chirps)")
    print(f"  DDS tone: {sdr.sample_rate/8/1e3:.1f} kHz")

    # Compute axes
    range_axis = np.arange(Ns) * p['range_resolution']
    velocity_axis = np.linspace(-p['v_max'], p['v_max'], Nc, endpoint=False)

    print(f"  Range axis: 0 to {range_axis[-1]:.1f} m ({Ns} bins)")
    print(f"  Velocity axis: {velocity_axis[0]:.2f} to {velocity_axis[-1]:.2f} m/s ({Nc} bins)")

    # -- Flush and enable PLL --
    print("\n[3b] Flushing buffer and enabling PLL...")
    for _ in range(5):
        sdr.rx()
    cn0566.enable = 0  # 0 = PLL ON (not powered down)
    time.sleep(0.5)  # Let PLL lock
    for _ in range(3):
        sdr.rx()  # Flush stale data after PLL enable

    os.makedirs(output_dir, exist_ok=True)

    # -- Capture and process frames --
    print(f"\n[3c] Capturing {num_frames} frames...")

    all_rdms = []
    all_detections = []

    for frame_idx in range(num_frames):
        t0 = time.time()

        # Capture one CPI
        raw = sdr.rx()
        if isinstance(raw, list):
            data = np.array(raw)
        else:
            data = raw[np.newaxis, :]

        ch0 = data[0]  # Use first RX channel
        rx_power_db = 10 * np.log10(np.mean(np.abs(ch0)**2) + 1e-12)

        # Reshape to [Nc, Ns]
        if len(ch0) < Nc * Ns:
            print(f"  Frame {frame_idx}: WARNING - got {len(ch0)} samples, "
                  f"need {Nc*Ns}. Padding.")
            ch0 = np.pad(ch0, (0, Nc * Ns - len(ch0)))
        elif len(ch0) > Nc * Ns:
            ch0 = ch0[:Nc * Ns]

        chirp_matrix = ch0.reshape(Nc, Ns)

        # -- Range-Doppler Processing --
        # 1. Windowing (Hanning on fast-time)
        win = np.hanning(Ns)
        chirp_matrix = chirp_matrix * win[np.newaxis, :]

        # 2. Range FFT (fast-time, along columns)
        range_fft = np.fft.fft(chirp_matrix, axis=1)

        # 3. MTI (2-pulse canceller) to suppress DC / static clutter
        range_fft_mti = np.diff(range_fft, axis=0)  # [Nc-1, Ns]
        # Pad back to Nc
        range_fft_mti = np.vstack([range_fft_mti, np.zeros((1, Ns), dtype=complex)])

        # 4. Doppler windowing
        doppler_win = np.hanning(Nc)
        range_fft_mti = range_fft_mti * doppler_win[:, np.newaxis]

        # 5. Doppler FFT (slow-time, along rows)
        rdm = np.fft.fftshift(np.fft.fft(range_fft_mti, axis=0), axes=0)

        # 6. Convert to dB
        rdm_mag = np.abs(rdm)
        rdm_db = 20 * np.log10(rdm_mag + 1e-12)

        # Also compute RDM without MTI for reference
        rdm_nomti = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        rdm_nomti_db = 20 * np.log10(np.abs(rdm_nomti) + 1e-12)

        all_rdms.append(rdm_db)

        # -- CFAR Detection (2D CA-CFAR) --
        detections = cfar_2d(rdm_db, range_axis, velocity_axis,
                             num_train=12, num_guard=4,
                             threshold_offset=15, min_range=1.0)
        all_detections.append(detections)

        elapsed = time.time() - t0
        print(f"  Frame {frame_idx:2d}: RX_pwr={rx_power_db:.1f}dB, "
              f"RDM_peak={rdm_db.max():.1f}dB, "
              f"dets={len(detections)}, "
              f"t={elapsed*1000:.0f}ms")

        for d in detections[:5]:
            print(f"           R={d['range_m']:.1f}m, "
                  f"V={d['velocity_mps']:.2f}m/s, "
                  f"P={d['power']:.1f}dB")

    cn0566.enable = 1  # 1 = PLL OFF (powerdown) - cleanup

    # -- Process and Save Frame Images --
    if HAS_MATPLOTLIB and len(all_rdms) > 0:
        print(f"\n[3d] Generating plots {num_frames} frames...")
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Determine global min/max for consistent colormap across frames
        # Use a robust range from all data
        all_rdm_stack = np.array(all_rdms)
        vmin_global = np.percentile(all_rdm_stack, 30)
        vmax_global = all_rdm_stack.max()

        for i, rdm_frame in enumerate(all_rdms):
             fig_f, ax_f = plt.subplots(figsize=(10, 8))
             im = ax_f.imshow(
                rdm_frame, aspect='auto', origin='lower',
                extent=[0, range_axis[-1], velocity_axis[0], velocity_axis[-1]],
                cmap='jet', vmin=vmin_global, vmax=vmax_global
             )
             ax_f.set_xlabel('Range (m)')
             ax_f.set_ylabel('Velocity (m/s)')
             ax_f.set_title(f'Frame {i:03d}')
             plt.colorbar(im, ax=ax_f, label='dB')
             
             # Overlay detections for this frame
             current_dets = all_detections[i]
             if current_dets:
                 det_r = [d['range_m'] for d in current_dets]
                 det_v = [d['velocity_mps'] for d in current_dets]
                 ax_f.scatter(det_r, det_v, s=80, facecolors='none', 
                              edgecolors='white', linewidths=2)
            
             frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
             plt.savefig(frame_path, dpi=100)
             plt.close(fig_f)

    # -- Generate plots --
    if HAS_MATPLOTLIB and len(all_rdms) > 0:
        print(f"[3e] Generating summary plots...")

        # Plot last frame's RDM
        rdm_plot = all_rdms[-1]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CN0566 FMCW Radar - Indoor Static Environment', fontsize=14)

        # RDM (with MTI)
        vmin = np.percentile(rdm_plot, 30)
        vmax = rdm_plot.max()
        im0 = axes[0, 0].imshow(
            rdm_plot, aspect='auto', origin='lower',
            extent=[0, range_axis[-1], velocity_axis[0], velocity_axis[-1]],
            cmap='jet', vmin=vmin, vmax=vmax
        )
        axes[0, 0].set_xlabel('Range (m)')
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].set_title('Range-Doppler Map (with MTI)')
        plt.colorbar(im0, ax=axes[0, 0], label='dB')

        # Mark detections
        if all_detections[-1]:
            det_r = [d['range_m'] for d in all_detections[-1]]
            det_v = [d['velocity_mps'] for d in all_detections[-1]]
            axes[0, 0].scatter(det_r, det_v, s=80, facecolors='none',
                               edgecolors='white', linewidths=2)

        # Range profile (zero-Doppler cut without MTI)
        rdm_nomti_plot = rdm_nomti_db
        zero_dop_idx = Nc // 2
        range_profile = rdm_nomti_plot[zero_dop_idx, :]
        axes[0, 1].plot(range_axis, range_profile, 'b-', linewidth=0.8)
        axes[0, 1].set_xlabel('Range (m)')
        axes[0, 1].set_ylabel('Power (dB)')
        axes[0, 1].set_title('Range Profile (Zero-Doppler, no MTI)')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(0, 50)  # Focus on near range for indoor

        # RDM without MTI
        im2 = axes[1, 0].imshow(
            rdm_nomti_plot, aspect='auto', origin='lower',
            extent=[0, range_axis[-1], velocity_axis[0], velocity_axis[-1]],
            cmap='jet',
            vmin=np.percentile(rdm_nomti_plot, 30),
            vmax=rdm_nomti_plot.max()
        )
        axes[1, 0].set_xlabel('Range (m)')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].set_title('Range-Doppler Map (no MTI)')
        plt.colorbar(im2, ax=axes[1, 0], label='dB')

        # Average RDM across frames
        if len(all_rdms) > 1:
            avg_rdm = np.mean(all_rdms, axis=0)
            im3 = axes[1, 1].imshow(
                avg_rdm, aspect='auto', origin='lower',
                extent=[0, range_axis[-1], velocity_axis[0], velocity_axis[-1]],
                cmap='jet',
                vmin=np.percentile(avg_rdm, 30),
                vmax=avg_rdm.max()
            )
            axes[1, 1].set_xlabel('Range (m)')
            axes[1, 1].set_ylabel('Velocity (m/s)')
            axes[1, 1].set_title(f'Average RDM ({len(all_rdms)} frames, with MTI)')
            plt.colorbar(im3, ax=axes[1, 1], label='dB')
        else:
            axes[1, 1].text(0.5, 0.5, 'Single frame only',
                            transform=axes[1, 1].transAxes,
                            ha='center', va='center')

        plt.tight_layout()
        path = os.path.join(output_dir, 'cn0566_fmcw_radar.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  RDM plot saved: {path}")

        # Detection summary plot
        if any(len(d) > 0 for d in all_detections):
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for fi, dets in enumerate(all_detections):
                for d in dets:
                    ax2.scatter(d['range_m'], d['velocity_mps'],
                                c='blue', alpha=0.5, s=30)
            ax2.set_xlabel('Range (m)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title(f'All Detections ({num_frames} frames)')
            ax2.grid(True)
            ax2.set_xlim(0, 50)
            path2 = os.path.join(output_dir, 'cn0566_detections.png')
            plt.savefig(path2, dpi=150)
            plt.close()
            print(f"  Detections plot saved: {path2}")

    # Save raw data for later analysis
    np.savez(
        os.path.join(output_dir, 'cn0566_radar_data.npz'),
        rdms=np.array(all_rdms),
        range_axis=range_axis,
        velocity_axis=velocity_axis,
        params=RADAR_PARAMS,
        num_frames=num_frames
    )
    print(f"  Data saved: {os.path.join(output_dir, 'cn0566_radar_data.npz')}")

    # Cleanup
    del cn0566, sdr

    # Summary
    total_dets = sum(len(d) for d in all_detections)
    print(f"\n  SUMMARY:")
    print(f"    Frames captured: {num_frames}")
    print(f"    Total detections: {total_dets}")
    print(f"    Avg detections/frame: {total_dets/num_frames:.1f}")

    return True


def cfar_2d(rdm_db, range_axis, velocity_axis,
            num_train=12, num_guard=4, threshold_offset=15,
            min_range=1.0, nms_kernel=7):
    """
    Simple 2D CA-CFAR detection on Range-Doppler Map.
    """
    Nd, Nr = rdm_db.shape
    detections = []
    half_t = num_train
    half_g = num_guard

    # Min range index
    dr = range_axis[1] - range_axis[0] if len(range_axis) > 1 else 1.0
    min_r_idx = int(min_range / dr)

    for d in range(half_t + half_g, Nd - half_t - half_g):
        for r in range(max(half_t + half_g, min_r_idx), Nr - half_t - half_g):
            cell = rdm_db[d, r]

            # Extract training cells (ring around CUT, excluding guard)
            train_sum = 0.0
            train_count = 0
            for dd in range(-half_t - half_g, half_t + half_g + 1):
                for rr in range(-half_t - half_g, half_t + half_g + 1):
                    if abs(dd) <= half_g and abs(rr) <= half_g:
                        continue  # Guard region
                    di = d + dd
                    ri = r + rr
                    if 0 <= di < Nd and 0 <= ri < Nr:
                        train_sum += rdm_db[di, ri]
                        train_count += 1

            if train_count == 0:
                continue

            threshold = train_sum / train_count + threshold_offset

            if cell > threshold:
                detections.append({
                    'range_idx': r,
                    'doppler_idx': d,
                    'range_m': float(range_axis[r]),
                    'velocity_mps': float(velocity_axis[d]),
                    'power': float(cell),
                    'threshold': float(threshold),
                    'magnitude': float(cell)
                })

    # NMS
    if nms_kernel > 1 and len(detections) > 0:
        detections = nms_detections(detections, rdm_db, nms_kernel)

    return detections


def nms_detections(detections, rdm_db, kernel_size=7):
    """Non-Maximum Suppression on detection list."""
    half_k = kernel_size // 2
    Nd, Nr = rdm_db.shape
    filtered = []

    for det in detections:
        d, r = det['doppler_idx'], det['range_idx']
        val = rdm_db[d, r]

        is_max = True
        for dd in range(-half_k, half_k + 1):
            for rr in range(-half_k, half_k + 1):
                if dd == 0 and rr == 0:
                    continue
                di, ri = d + dd, r + rr
                if 0 <= di < Nd and 0 <= ri < Nr:
                    if rdm_db[di, ri] > val:
                        is_max = False
                        break
            if not is_max:
                break

        if is_max:
            filtered.append(det)

    return filtered


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='CN0566 Phaser FMCW Radar Test')
    parser.add_argument('--test', type=str, default='all',
                        choices=['connectivity', 'basic_rx', 'spectral', 'fmcw', 'all'],
                        help='Test to run')
    parser.add_argument('--sdr_ip', type=str, default='ip:192.168.2.1',
                        help='PlutoSDR IP address')
    parser.add_argument('--phaser_ip', type=str, default='ip:phaser.local',
                        help='CN0566 Phaser IP address')
    parser.add_argument('--output_dir', type=str, default='output/cn0566_test',
                        help='Output directory for plots and data')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames for FMCW test')
    parser.add_argument('--avg', type=int, default=100,
                        help='Number of frames to average for spectral test')
    args = parser.parse_args()

    print("=" * 60)
    print("  CN0566 Phaser FMCW Radar Test")
    print(f"  SDR: {args.sdr_ip}")
    print(f"  Phaser: {args.phaser_ip}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.test in ('connectivity', 'all'):
        ok = test_connectivity(args.sdr_ip, args.phaser_ip)
        if not ok and args.test == 'all':
            print("\nConnectivity test failed. Aborting remaining tests.")
            return
        gc.collect()

    if args.test in ('basic_rx', 'all'):
        test_basic_rx(args.sdr_ip, args.phaser_ip, args.output_dir)
        gc.collect()

    if args.test in ('spectral', 'all'):
        test_spectral(args.sdr_ip, args.phaser_ip, args.output_dir,
                       num_avg=args.avg)
        gc.collect()

    if args.test in ('fmcw', 'all'):
        test_fmcw_radar(args.sdr_ip, args.phaser_ip, args.output_dir,
                         num_frames=args.frames)

    print("\nAll tests complete.")


if __name__ == '__main__':
    main()
