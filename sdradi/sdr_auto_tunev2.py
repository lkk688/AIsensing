#!/usr/bin/env python3
"""
SDR Auto Tune - PlutoSDR TX/RX Link Testing Tool
Supports 1R1T mode with DC offset and crosstalk detection.
"""
import argparse
import time
import numpy as np
import adi
import sys

# === Configuration ===
CONFIG = {
    "ip": "ip:192.168.3.2",  # Default Pluto IP
    "fc": 2405e6,
    "fs": 3e6,
    "tx_gain": 0,
    "rx_gain": 70
}

# Link test thresholds
LINK_THRESHOLDS = {
    "min_snr_db": 15.0,           # Minimum acceptable SNR
    "min_signal_peak": 500,        # Minimum signal peak amplitude
    "max_dc_offset_db": -20.0,     # Maximum acceptable DC offset relative to signal
    "min_tone_detection_db": 10.0, # Tone must be this much above noise floor
    "max_freq_error_hz": 1000,     # Maximum frequency error tolerance
}

# Presets for different test scenarios
TEST_PRESETS = {
    "cable_30db": {
        "description": "Cable loopback with 30dB attenuator",
        "tx_gain": -10,
        "rx_gain": 40,
        "tx_amp": 0.5,
    },
    "cable_direct": {
        "description": "Direct cable loopback (no attenuator) - USE CAUTION!",
        "tx_gain": -30,
        "rx_gain": 20,
        "tx_amp": 0.1,
    },
    "antenna_close": {
        "description": "Antennas close together (<1m)",
        "tx_gain": 0,  # Full TX power
        "rx_gain": 50,
        "tx_amp": 0.9,
    },
    "antenna_far": {
        "description": "Antennas further apart (>1m)",
        "tx_gain": 0,  # Full TX power
        "rx_gain": 70,  # Max gain
        "tx_amp": 0.9,
    },
}

def generate_beacon(fs, freq_hz=100e3, amplitude=0.5):
    """Generates a continuous complex tone."""
    N = 1024 * 16
    t = np.arange(N) / fs
    beacon = amplitude * np.exp(1j * 2 * np.pi * freq_hz * t)
    return (beacon * (2**14)).astype(np.complex64)

def measure_channel(sdr, channel_idx):
    try:
        # Enable specific channel
        sdr.rx_enabled_channels = [channel_idx]
        sdr.rx_buffer_size = 1024 * 16
        
        # Robustness: Increase kernel buffers
        if hasattr(sdr, "_rxadc") and hasattr(sdr._rxadc, "set_kernel_buffers_count"):
            sdr._rxadc.set_kernel_buffers_count(4)
        
        # Clear buffer
        for _ in range(2):
            sdr.rx()
            
        data = sdr.rx()
        
        # Stats
        peak = np.max(np.abs(data))
        peak_db = 20 * np.log10(peak + 1)
        
        # SNR
        fft = np.abs(np.fft.fftshift(np.fft.fft(data)))
        fft_db = 20 * np.log10(fft + 1e-12)
        sig = np.max(fft_db)
        noise = np.median(fft_db)
        snr = sig - noise
        
        return peak, peak_db, snr, "OK"
    except Exception as e:
        return 0, -100, 0, str(e)

def restart_device(ip):
    print(f"=== RESTARTING DEVICE ({ip}) ===")
    try:
        import iio
        ctx = iio.Context(ip)
        phy = ctx.find_device("ad9361-phy")
        if phy and "device_reboot" in phy.debug_attrs:
            phy.debug_attrs["device_reboot"].value = "1"
            print("Command sent. Device is rebooting...")
            print("Please wait 15-20 seconds before reconnecting.")
        else:
            print("Error: 'device_reboot' debug attribute not found.")
    except Exception as e:
        print(f"Restart Failed: {e}")


def run_device_diagnostic(ip):
    """
    Comprehensive device diagnostic to check hardware state and connectivity.
    """
    import iio

    print("\n" + "=" * 70)
    print("  PLUTO SDR DEVICE DIAGNOSTIC")
    print("=" * 70)
    print(f"Target: {ip}")
    print("-" * 70)

    issues = []

    try:
        # Step 1: Basic connectivity
        print("\n[1] CONNECTIVITY")
        ctx = iio.Context(ip)
        print(f"    Context: {ctx.name} - OK")

        # List devices
        print(f"    Devices found: {len(ctx.devices)}")
        for dev in ctx.devices:
            print(f"      - {dev.name}")

        # Step 2: PHY device status
        print("\n[2] AD9361-PHY STATUS")
        phy = ctx.find_device("ad9361-phy")
        if not phy:
            print("    ERROR: ad9361-phy not found!")
            issues.append("PHY device not found")
        else:
            # ENSM mode
            ensm = phy.attrs.get('ensm_mode')
            if ensm:
                mode = ensm.value
                print(f"    ENSM Mode: {mode}")
                if mode not in ['fdd', 'tdd']:
                    print(f"    WARNING: ENSM mode '{mode}' may prevent TX/RX")
                    issues.append(f"ENSM mode is '{mode}' (should be 'fdd' or 'tdd')")

            # Check RX channel
            rx_ch = None
            for ch in phy.channels:
                if ch.id == 'voltage0' and not ch.output:
                    rx_ch = ch
                    break

            if rx_ch:
                print(f"    RX Channel 0:")
                print(f"      - Gain mode: {rx_ch.attrs.get('gain_control_mode', {}).value if 'gain_control_mode' in rx_ch.attrs else 'N/A'}")
                print(f"      - Hardware gain: {rx_ch.attrs.get('hardwaregain', {}).value if 'hardwaregain' in rx_ch.attrs else 'N/A'}")
                print(f"      - Sample rate: {rx_ch.attrs.get('sampling_frequency', {}).value if 'sampling_frequency' in rx_ch.attrs else 'N/A'}")

            # Check TX channel
            tx_ch = None
            for ch in phy.channels:
                if ch.id == 'voltage0' and ch.output:
                    tx_ch = ch
                    break

            if tx_ch:
                print(f"    TX Channel 0:")
                print(f"      - Hardware gain: {tx_ch.attrs.get('hardwaregain', {}).value if 'hardwaregain' in tx_ch.attrs else 'N/A'}")

            # Check BIST loopback
            if 'loopback' in phy.debug_attrs:
                loopback = phy.debug_attrs['loopback'].value
                print(f"    BIST Loopback: {loopback}")
                if loopback != '0':
                    print("    WARNING: Digital loopback is enabled!")
                    issues.append("BIST loopback enabled - may interfere with RF")

        # Step 3: RX ADC status
        print("\n[3] RX ADC (cf-ad9361-lpc) STATUS")
        rxadc = ctx.find_device("cf-ad9361-lpc")
        if not rxadc:
            print("    ERROR: RX ADC not found!")
            issues.append("RX ADC device not found")
        else:
            print(f"    Device: {rxadc.name}")

            # Check sync status
            if 'sync_start_enable' in rxadc.attrs:
                sync = rxadc.attrs['sync_start_enable'].value
                print(f"    Sync start: {sync}")

            if 'waiting_for_supplier' in rxadc.attrs:
                waiting = rxadc.attrs['waiting_for_supplier'].value
                print(f"    Waiting for supplier: {waiting}")
                if waiting != '0':
                    issues.append("RX ADC waiting for supplier")

            # Check channels
            scan_channels = [ch for ch in rxadc.channels if ch.scan_element]
            print(f"    Scan element channels: {[ch.id for ch in scan_channels]}")

            # Note: pseudorandom_err_check is only meaningful when BIST/PRBS is enabled
            # In normal mode, "Out of Sync" is expected and NOT an error
            if 'pseudorandom_err_check' in rxadc.debug_attrs:
                pn_status = rxadc.debug_attrs['pseudorandom_err_check'].value
                print(f"    PN Status (BIST only, ignore in normal mode):")
                for line in pn_status.strip().split('\n'):
                    print(f"      {line}")
                print("    (Note: 'Out of Sync' is normal when BIST/PRBS is not active)")

        # Step 4: TX DAC status
        print("\n[4] TX DAC (cf-ad9361-dds-core-lpc) STATUS")
        txdac = ctx.find_device("cf-ad9361-dds-core-lpc")
        if not txdac:
            print("    ERROR: TX DAC not found!")
            issues.append("TX DAC device not found")
        else:
            print(f"    Device: {txdac.name}")

            if 'sync_start_enable' in txdac.attrs:
                sync = txdac.attrs['sync_start_enable'].value
                print(f"    Sync start: {sync}")

        # Step 5: Try to fix common issues
        print("\n[5] ATTEMPTING FIXES")

        # Ensure FDD mode
        if phy:
            current_ensm = phy.attrs['ensm_mode'].value
            if current_ensm not in ['fdd', 'tdd']:
                print("    Setting ENSM to FDD...")
                try:
                    phy.attrs['ensm_mode'].value = 'fdd'
                    print(f"    ENSM now: {phy.attrs['ensm_mode'].value}")
                except Exception as e:
                    print(f"    Failed to set FDD: {e}")

            # Disable BIST loopback
            if 'loopback' in phy.debug_attrs:
                if phy.debug_attrs['loopback'].value != '0':
                    print("    Disabling BIST loopback...")
                    try:
                        phy.debug_attrs['loopback'].value = '0'
                        print("    Loopback disabled")
                    except Exception as e:
                        print(f"    Failed: {e}")

            # Reinitialize
            print("    Reinitializing AD9361...")
            try:
                phy.debug_attrs['initialize'].value = '1'
                import time
                time.sleep(1)
                print("    Reinitialized")
            except Exception as e:
                print(f"    Failed: {e}")

        # Step 6: Quick RX test
        print("\n[6] QUICK RX TEST")
        try:
            import adi
            import numpy as np
            import time

            sdr = adi.Pluto(uri=ip)
            sdr.sample_rate = int(3e6)
            sdr.rx_lo = int(2405e6)
            sdr.rx_rf_bandwidth = int(3e6)
            sdr.gain_control_mode_chan0 = 'manual'
            sdr.rx_hardwaregain_chan0 = 40
            sdr.rx_enabled_channels = [0]
            sdr.rx_buffer_size = 1024

            print("    Attempting to receive 1024 samples...")
            start = time.time()
            try:
                data = sdr.rx()
                elapsed = time.time() - start
                print(f"    SUCCESS: Received {len(data)} samples in {elapsed:.2f}s")
                print(f"    Peak amplitude: {np.max(np.abs(data)):.0f}")
            except Exception as e:
                elapsed = time.time() - start
                print(f"    FAILED after {elapsed:.2f}s: {e}")
                issues.append(f"RX timeout: {e}")

        except Exception as e:
            print(f"    ERROR: {e}")
            issues.append(f"pyadi-iio error: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("  DIAGNOSTIC SUMMARY")
        print("=" * 70)

        if issues:
            print(f"\n  Found {len(issues)} issue(s):")
            for i, issue in enumerate(issues, 1):
                print(f"    {i}. {issue}")

            # Check for critical issues (timeout = real problem)
            has_timeout = any('timeout' in issue.lower() for issue in issues)

            print("\n  RECOMMENDED ACTIONS:")
            if has_timeout:
                print("    1. Power cycle the PlutoSDR (unplug USB/Ethernet, wait 5s, replug)")
                print("    2. Check if firmware needs update")
                print("    3. Try: python sdr_auto_tune.py --mode restart --ip " + ip)
                print("    4. If using Ethernet, ensure no firewall blocking UDP ports")
            else:
                print("    Review the issues above and address as needed.")
        else:
            print("\n  All checks passed!")

        print("=" * 70)

    except Exception as e:
        print(f"\n  DIAGNOSTIC ERROR: {e}")
        import traceback
        traceback.print_exc()

def run_rx_scanner(ip):
    print(f"\n=== RX SCANNER MODE ===")
    print(f"Target IP: {ip}")
    
    try:
        try:
            sdr = adi.ad9361(uri=ip)
        except:
            sdr = adi.Pluto(uri=ip)
            
        sdr.sample_rate = int(CONFIG["fs"])
        sdr.rx_lo = int(CONFIG["fc"])
        sdr.rx_rf_bandwidth = int(CONFIG["fs"])
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = int(CONFIG["rx_gain"])
        
        # Check if Ch1 exists (try setting gain)
        has_ch1 = False
        try:
            sdr.gain_control_mode_chan1 = "manual"
            sdr.rx_hardwaregain_chan1 = int(CONFIG["rx_gain"])
            has_ch1 = True
        except:
            print("[Info] Channel 1 not detected/enabled on this device (Standard Pluto?)")

        print("\nScanning Channels...")
        print(f"{'CH':<3} | {'PEAK':>8} | {'dBFS':>8} | {'SNR (dB)':>10} | {'STATUS':<20}")
        print("-" * 60)
        
        while True:
            # Measure Ch 0
            p0, db0, snr0, stat0 = measure_channel(sdr, 0)
            
            status0_txt = ""
            if p0 > 2000: status0_txt = "\033[92mSTRONG\033[0m"
            elif snr0 > 10: status0_txt = "\033[91mGHOST?\033[0m"
            else: status0_txt = "."
            
            print(f" 0  | {p0:8.0f} | {db0:8.1f} | {snr0:10.1f} | {status0_txt}")
            
            if has_ch1:
                p1, db1, snr1, stat1 = measure_channel(sdr, 1)
                status1_txt = ""
                if p1 > 2000: status1_txt = "\033[92mSTRONG\033[0m"
                elif snr1 > 10: status1_txt = "\033[91mGHOST?\033[0m"
                else: status1_txt = "."
                
                print(f" 1  | {p1:8.0f} | {db1:8.1f} | {snr1:10.1f} | {status1_txt}")
            
            print("-" * 60)
            time.sleep(0.5)

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"RX Error: {e}")

def run_loopback_test(ip):
    print(f"\n=== SELF-LOOPBACK DIAGNOSTIC (2x2 Matrix) ===")
    print(f"Target IP: {ip}")
    print(f"Config: FC={CONFIG['fc']/1e6}MHz, FS={CONFIG['fs']/1e6}MHz")
    print("-" * 60)
    print(f"{'TX':<5} -> {'RX':<5} | {'PEAK':>8} | {'dBFS':>8} | {'SNR (dB)':>10} | {'STATUS':<20}")
    print("-" * 60)
    
    try:
        try:
            sdr = adi.ad9361(uri=ip)
            print("Using Driver: adi.ad9361 (2R2T Support)")
        except:
            sdr = adi.Pluto(uri=ip)
            print("Using Driver: adi.Pluto")
            
        sdr.sample_rate = int(CONFIG["fs"])
        sdr.tx_lo = int(CONFIG["fc"])
        sdr.rx_lo = int(CONFIG["fc"])
        sdr.tx_rf_bandwidth = int(CONFIG["fs"])
        sdr.rx_rf_bandwidth = int(CONFIG["fs"])
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = 30 # Moderate gain for loopback
        sdr.tx_hardwaregain_chan0 = -20 # Low power for loopback (start safe)
        
        # Check available channels
        channels = [0]
        try:
            sdr.rx_hardwaregain_chan1 = 30
            channels.append(1)
        except:
            pass # Only 1 channel
            
        beacon = generate_beacon(CONFIG["fs"])
        
        for tx_ch in channels:
            for rx_ch in channels:
                try:
                    # Setup TX
                    if tx_ch == 0:
                        sdr.tx_hardwaregain_chan0 = -10
                        sdr.tx_enabled_channels = [0]
                    else:
                        sdr.tx_hardwaregain_chan1 = -10
                        sdr.tx_enabled_channels = [1]
                        
                    sdr.tx_cyclic_buffer = True
                    sdr.tx(beacon)
                    time.sleep(0.5) # Allow TX to settle
                    
                    # Setup RX
                    sdr.rx_enabled_channels = [rx_ch]
                    if rx_ch == 0: sdr.rx_hardwaregain_chan0 = 40
                    else: sdr.rx_hardwaregain_chan1 = 40
                    
                    sdr.rx_buffer_size = 1024 * 16
                    
                    # Clear & Measure
                    for _ in range(3): sdr.rx() # Flush
                    data = sdr.rx()
                    sdr.tx_destroy_buffer() # Stop TX for this step
                    
                    # Metrics
                    peak = np.max(np.abs(data))
                    peak_db = 20 * np.log10(peak + 1)
                    fft = np.abs(np.fft.fftshift(np.fft.fft(data)))
                    fft_db = 20 * np.log10(fft + 1e-12)
                    snr = np.max(fft_db) - np.median(fft_db)
                    
                    # Status
                    status = "."
                    if peak > 2000: status = "\033[92mPASSED (Strong)\033[0m"
                    elif snr > 15: status = "\033[93mWEAK LINK\033[0m"
                    else: status = "\033[91mNO SIGNAL\033[0m"
                    
                    print(f" CH{tx_ch}  ->  CH{rx_ch}  | {peak:8.0f} | {peak_db:8.1f} | {snr:10.1f} | {status}")
                    
                except Exception as e:
                    print(f" CH{tx_ch}  ->  CH{rx_ch}  |   ERROR  |          |            | {str(e)}")
                    try: sdr.tx_destroy_buffer()
                    except: pass
                    
    except Exception as e:
        print(f"Device Error: {e}")

def run_tx(ip, channel):
    print(f"\n=== TX BEACON MODE ===")
    print(f"Target IP: {ip}")
    print(f"Freq: {CONFIG['fc']/1e6} MHz | Gain: {CONFIG['tx_gain']} dB")
    
    try:
        try:
            sdr = adi.ad9361(uri=ip)
        except:
            sdr = adi.Pluto(uri=ip)
            
        sdr.sample_rate = int(CONFIG["fs"])
        sdr.tx_lo = int(CONFIG["fc"])
        sdr.tx_rf_bandwidth = int(CONFIG["fs"])
        sdr.tx_cyclic_buffer = True
        
        # Set Gain
        if channel == 0:
            sdr.tx_hardwaregain_chan0 = int(CONFIG["tx_gain"])
            sdr.tx_enabled_channels = [0]
        else:
            try:
                sdr.tx_hardwaregain_chan1 = int(CONFIG["tx_gain"])
                sdr.tx_enabled_channels = [1]
            except:
                print("FATAL: Channel 1 not enabled on this device.")
                return

        data = generate_beacon(CONFIG["fs"])
        sdr.tx(data)
        
        print(f"--> TRANSMITTING on CH {channel}... (Press Ctrl+C to Stop)")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        try: sdr.tx_destroy_buffer()
        except: pass
        sys.exit(0)
    except Exception as e:
        print(f"TX Error: {e}")

def set_phy_mode(sdr, mode):
    """Sets the ENSM mode of the AD9361."""
    try:
        import iio
        ctx = sdr.ctx
        phy = ctx.find_device('ad9361-phy')
        if phy:
            if "ensm_mode" in phy.attrs:
                phy.attrs["ensm_mode"].value = mode
                return True
    except Exception as e:
        print(f"[Mode] Failed to set {mode}: {e}")
    return False

def run_mode_test(ip):
    """Test switching between FDD and TDD modes."""
    print(f"\n=== ENSM MODE SWITCHING TEST ===")
    print(f"Target IP: {ip}")
    
    try:
        sdr = adi.Pluto(uri=ip)
        print("Connected to PlutoSDR.")
        
        # Test FDD
        print("Switching to FDD mode...")
        if set_phy_mode(sdr, "fdd"):
            print("  -> FDD Mode Set: OK")
        else:
            print("  -> FDD Mode Set: FAILED")
            
        time.sleep(1)
        
        # Test TDD (if applicable, but usually we stay in FDD for loopback unless using TDD engine)
        # Note: Pluto default firmware might only support FDD or pinctrl.
        # Let's check available modes first.
        try:
            ctx = sdr.ctx
            phy = ctx.find_device('ad9361-phy')
            available = phy.attrs["ensm_mode_available"].value
            print(f"Available Modes: {available}")
        except:
            print("Could not read available modes.")

    except Exception as e:
        print(f"Mode Test Error: {e}")

def analyze_spectrum(data, fs, tone_freq):
    """
    Analyze received signal spectrum for link quality metrics.
    Returns dict with DC offset, SNR, tone power, noise floor, frequency error.
    """
    N = len(data)

    # Remove DC for analysis (but measure it first)
    dc_component = np.mean(data)
    dc_power = np.abs(dc_component) ** 2

    # FFT analysis
    fft_data = np.fft.fftshift(np.fft.fft(data))
    fft_mag = np.abs(fft_data)
    fft_power = fft_mag ** 2
    freq_axis = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

    # Find DC bin (center)
    dc_bin = N // 2

    # Find expected tone bin
    expected_tone_bin = dc_bin + int(tone_freq * N / fs)

    # Search for actual peak near expected tone (within 5% bandwidth)
    search_width = max(10, int(N * 0.05))
    search_start = max(0, expected_tone_bin - search_width)
    search_end = min(N, expected_tone_bin + search_width)

    tone_region = fft_mag[search_start:search_end]
    peak_offset = np.argmax(tone_region)
    actual_tone_bin = search_start + peak_offset
    actual_tone_freq = freq_axis[actual_tone_bin]

    tone_power = fft_power[actual_tone_bin]

    # Calculate noise floor (exclude DC and tone regions)
    noise_mask = np.ones(N, dtype=bool)
    # Exclude DC region (center 5 bins)
    noise_mask[dc_bin-5:dc_bin+6] = False
    # Exclude tone region (11 bins around peak)
    noise_mask[max(0, actual_tone_bin-5):min(N, actual_tone_bin+6)] = False

    if np.sum(noise_mask) > 0:
        noise_power = np.mean(fft_power[noise_mask])
    else:
        noise_power = np.min(fft_power)

    # Calculate metrics
    dc_power_at_bin = fft_power[dc_bin]

    # SNR in dB
    if noise_power > 0:
        snr_db = 10 * np.log10(tone_power / noise_power)
    else:
        snr_db = 100.0  # Very high SNR if no noise

    # DC offset relative to signal (in dB)
    if tone_power > 0:
        dc_offset_db = 10 * np.log10(dc_power_at_bin / tone_power + 1e-12)
    else:
        dc_offset_db = 0.0

    # Frequency error
    freq_error_hz = actual_tone_freq - tone_freq

    # Signal peak in time domain
    signal_peak = np.max(np.abs(data))

    # EVM estimation (simple - compare to ideal tone after DC removal)
    data_no_dc = data - dc_component
    # Reconstruct ideal tone at detected frequency
    t = np.arange(N) / fs
    # Estimate amplitude and phase from peak bin
    amplitude = np.sqrt(tone_power) * 2 / N
    phase = np.angle(fft_data[actual_tone_bin])
    ideal_tone = amplitude * np.exp(1j * (2 * np.pi * actual_tone_freq * t + phase))

    error_signal = data_no_dc - ideal_tone
    error_power = np.mean(np.abs(error_signal) ** 2)
    signal_power = np.mean(np.abs(ideal_tone) ** 2)
    if signal_power > 0:
        evm_percent = 100 * np.sqrt(error_power / signal_power)
    else:
        evm_percent = 100.0

    return {
        "snr_db": snr_db,
        "dc_offset_db": dc_offset_db,
        "dc_magnitude": np.abs(dc_component),
        "tone_power_db": 10 * np.log10(tone_power + 1e-12),
        "noise_floor_db": 10 * np.log10(noise_power + 1e-12),
        "freq_error_hz": freq_error_hz,
        "detected_freq_hz": actual_tone_freq,
        "signal_peak": signal_peak,
        "evm_percent": evm_percent,
        "fft_mag": fft_mag,
        "freq_axis": freq_axis,
    }


def run_link_test(ip, save_plot=False, preset="cable_30db", tx_gain=None, rx_gain=None,
                  freq=None, tone_freq=100e3):
    """
    Comprehensive TX/RX link test for 1R1T PlutoSDR.
    Tests signal integrity, DC offset, crosstalk, and link quality.
    Includes TX ON/OFF comparison to verify signal source.

    Presets: cable_30db, cable_direct, antenna_close, antenna_far
    """
    # Get preset or use defaults
    if preset in TEST_PRESETS:
        test_config = TEST_PRESETS[preset].copy()
    else:
        test_config = TEST_PRESETS["cable_30db"].copy()

    # Override with explicit parameters if provided
    if tx_gain is not None:
        test_config["tx_gain"] = tx_gain
    if rx_gain is not None:
        test_config["rx_gain"] = rx_gain

    # Use custom frequency or default
    center_freq = freq if freq is not None else CONFIG["fc"]

    print("\n" + "=" * 70)
    print("  PLUTO SDR TX/RX LINK QUALITY TEST (1R1T Mode)")
    print("=" * 70)
    print(f"Target IP: {ip}")
    print(f"Preset: {preset} - {TEST_PRESETS.get(preset, {}).get('description', 'custom')}")
    print(f"Center Frequency: {center_freq/1e6:.1f} MHz")
    print(f"Tone Offset: {tone_freq/1e3:.1f} kHz")
    print(f"Sample Rate: {CONFIG['fs']/1e6:.1f} MHz")
    print(f"TX Gain: {test_config['tx_gain']} dB, RX Gain: {test_config['rx_gain']} dB")
    print("-" * 70)

    results = {
        "connection": False,
        "tx_setup": False,
        "rx_setup": False,
        "signal_detected": False,
        "snr_pass": False,
        "dc_offset_pass": False,
        "freq_accuracy_pass": False,
        "overall_pass": False,
    }

    sdr = None

    try:
        # Step 1: Connect to device
        print("\n[1/7] Connecting to PlutoSDR...")
        try:
            sdr = adi.Pluto(uri=ip)
            print(f"      Connected: adi.Pluto")
            results["connection"] = True
        except Exception as e:
            print(f"      FAILED: {e}")
            return results

        # Step 2: Configure TX
        print("\n[2/7] Configuring TX (Channel 0)...")
        try:
            sdr.sample_rate = int(CONFIG["fs"])
            sdr.tx_lo = int(center_freq)
            sdr.tx_rf_bandwidth = int(CONFIG["fs"])
            sdr.tx_hardwaregain_chan0 = test_config["tx_gain"]
            sdr.tx_enabled_channels = [0]
            print(f"      TX LO: {sdr.tx_lo/1e6:.1f} MHz")
            print(f"      TX Gain: {test_config['tx_gain']} dB")
            results["tx_setup"] = True
        except Exception as e:
            print(f"      FAILED: {e}")
            return results

        # Step 3: Configure RX
        print("\n[3/7] Configuring RX (Channel 0)...")
        try:
            sdr.rx_lo = int(center_freq)
            sdr.rx_rf_bandwidth = int(CONFIG["fs"])
            sdr.gain_control_mode_chan0 = "manual"
            sdr.rx_hardwaregain_chan0 = test_config["rx_gain"]
            sdr.rx_enabled_channels = [0]
            sdr.rx_buffer_size = 2**14  # Use power of 2 buffer size

            # Increase kernel buffers for stability
            if hasattr(sdr, "_rxadc") and sdr._rxadc is not None:
                try:
                    sdr._rxadc.set_kernel_buffers_count(4)
                    print("      Kernel buffers set to 4")
                except:
                    pass

            print(f"      RX LO: {sdr.rx_lo/1e6:.1f} MHz")
            print(f"      RX Gain: {test_config['rx_gain']} dB")
            results["rx_setup"] = True
        except Exception as e:
            print(f"      FAILED: {e}")
            return results

        # Step 4: Baseline measurement (TX OFF)
        print("\n[4/7] Measuring baseline (TX OFF)...")
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        #new add
        # Force TX amplitude effectively zero by sending zeros cyclic (then destroy)
        zeros = np.zeros(2**14, dtype=np.complex64)
        sdr.tx_cyclic_buffer = True
        sdr.tx(zeros)
        time.sleep(0.1)
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        # Measure baseline noise/interference at tone frequency
        time.sleep(0.2)
        baseline_metrics = []
        for _ in range(3):
            try:
                for _ in range(2):
                    sdr.rx()
                rx_data = sdr.rx()
                metrics = analyze_spectrum(rx_data, CONFIG["fs"], tone_freq)
                baseline_metrics.append(metrics)
            except:
                pass

        if baseline_metrics:
            baseline_tone_power = np.mean([m["tone_power_db"] for m in baseline_metrics])
            baseline_snr = np.mean([m["snr_db"] for m in baseline_metrics])
            print(f"      Baseline tone power: {baseline_tone_power:.1f} dB")
            print(f"      Baseline SNR: {baseline_snr:.1f} dB")
        else:
            baseline_tone_power = -100
            baseline_snr = 0
            print("      Baseline measurement failed, continuing...")

        # Step 5: Generate and transmit test tone
        print("\n[5/7] Transmitting test tone...")
        N = 2**14  # Power of 2 for better FFT
        t = np.arange(N) / CONFIG["fs"]
        # Generate clean tone with amplitude from preset
        tx_amplitude = test_config["tx_amp"]
        tx_tone = tx_amplitude * np.exp(1j * 2 * np.pi * tone_freq * t)
        tx_data = (tx_tone * (2**14)).astype(np.complex64)
        print(f"      TX Amplitude: {tx_amplitude}")

        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_data)
        print(f"      Tone Frequency: {tone_freq/1e3:.1f} kHz offset")
        print(f"      Transmitting...")

        # Let TX stabilize
        time.sleep(0.5)

        # Step 6: Receive and analyze
        print("\n[6/7] Receiving and analyzing signal (TX ON)...")

        # Initial RX to prime the buffer - with retry logic
        max_retries = 3
        rx_success = False

        for retry in range(max_retries):
            try:
                # Clear any stale buffers
                try:
                    sdr.rx_destroy_buffer()
                except:
                    pass

                # Re-initialize RX
                sdr.rx_buffer_size = 2**14
                time.sleep(0.2)

                # Flush RX buffer with timeout handling
                for _ in range(3):
                    try:
                        _ = sdr.rx()
                    except Exception as flush_err:
                        print(f"      Flush attempt: {flush_err}")
                        time.sleep(0.1)

                rx_success = True
                break
            except Exception as e:
                print(f"      Retry {retry + 1}/{max_retries}: {e}")
                time.sleep(0.5)

        if not rx_success:
            print("      Failed to initialize RX after retries")
            return results

        # Capture multiple frames for averaging
        num_captures = 3
        all_metrics = []

        for i in range(num_captures):
            try:
                rx_data = sdr.rx()
                metrics = analyze_spectrum(rx_data, CONFIG["fs"], tone_freq)
                all_metrics.append(metrics)
                time.sleep(0.05)
            except Exception as e:
                print(f"      Capture {i+1} failed: {e}")
                continue

        if len(all_metrics) == 0:
            print("      No successful captures!")
            return results

        # Average the metrics
        avg_snr = np.mean([m["snr_db"] for m in all_metrics])
        avg_dc_offset = np.mean([m["dc_offset_db"] for m in all_metrics])
        avg_freq_error = np.mean([m["freq_error_hz"] for m in all_metrics])
        avg_signal_peak = np.mean([m["signal_peak"] for m in all_metrics])
        avg_evm = np.mean([m["evm_percent"] for m in all_metrics])
        avg_tone_power = np.mean([m["tone_power_db"] for m in all_metrics])
        avg_noise_floor = np.mean([m["noise_floor_db"] for m in all_metrics])

        # Stop TX
        sdr.tx_destroy_buffer()

        # Step 7: Evaluate results
        print("\n[7/7] Link Quality Assessment")
        print("-" * 70)

        # TX ON/OFF comparison (key verification!)
        tx_on_tone_power = avg_tone_power
        delta_power_db = tx_on_tone_power - baseline_tone_power
        print(f"      TX OFF Baseline:  {baseline_tone_power:8.1f} dB")
        print(f"      TX ON Power:      {tx_on_tone_power:8.1f} dB")
        if delta_power_db > 10:
            delta_status = "\033[92mVERIFIED\033[0m"
            results["tx_verified"] = True
        elif delta_power_db > 3:
            delta_status = "\033[93mWEAK\033[0m"
            results["tx_verified"] = True
        else:
            delta_status = "\033[91mNOT VERIFIED\033[0m"
            results["tx_verified"] = False
        print(f"      Delta (ON-OFF):   {delta_power_db:8.1f} dB [{delta_status}]")
        print("-" * 70)

        # Signal Detection - use TX verification as primary indicator for antenna tests
        # For antenna tests, delta_power is more reliable than absolute peak
        min_peak = LINK_THRESHOLDS["min_signal_peak"]
        if "antenna" in preset:
            min_peak = 100  # Lower threshold for antenna testing

        if avg_signal_peak > min_peak or delta_power_db > 10:
            results["signal_detected"] = True
            print(f"      Signal Peak:     {avg_signal_peak:8.1f}  [DETECTED]")
        else:
            print(f"      Signal Peak:     {avg_signal_peak:8.1f}  [WEAK/NO SIGNAL]")

        # SNR Check
        if avg_snr >= LINK_THRESHOLDS["min_snr_db"]:
            results["snr_pass"] = True
            snr_status = "\033[92mPASS\033[0m"
        else:
            snr_status = "\033[91mFAIL\033[0m"
        print(f"      SNR:             {avg_snr:8.1f} dB (min: {LINK_THRESHOLDS['min_snr_db']:.1f} dB) [{snr_status}]")

        # DC Offset Check
        if avg_dc_offset <= LINK_THRESHOLDS["max_dc_offset_db"]:
            results["dc_offset_pass"] = True
            dc_status = "\033[92mPASS\033[0m"
        else:
            dc_status = "\033[93mWARN\033[0m"
        print(f"      DC Offset:       {avg_dc_offset:8.1f} dB (max: {LINK_THRESHOLDS['max_dc_offset_db']:.1f} dB) [{dc_status}]")

        # Frequency Accuracy Check
        if abs(avg_freq_error) <= LINK_THRESHOLDS["max_freq_error_hz"]:
            results["freq_accuracy_pass"] = True
            freq_status = "\033[92mPASS\033[0m"
        else:
            freq_status = "\033[91mFAIL\033[0m"
        print(f"      Freq Error:      {avg_freq_error:8.1f} Hz (max: {LINK_THRESHOLDS['max_freq_error_hz']:.0f} Hz) [{freq_status}]")

        # Additional metrics
        print(f"      Tone Power:      {avg_tone_power:8.1f} dB")
        print(f"      Noise Floor:     {avg_noise_floor:8.1f} dB")
        print(f"      EVM:             {avg_evm:8.1f} %")

        # Overall assessment
        print("-" * 70)
        results["overall_pass"] = (
            results["signal_detected"] and
            results["snr_pass"] and
            results["freq_accuracy_pass"] and
            results.get("tx_verified", False)
        )

        if results["overall_pass"]:
            print("\n  \033[92m[LINK TEST PASSED]\033[0m - TX/RX link verified and operational")
            if not results["dc_offset_pass"]:
                print("  \033[93m[WARNING]\033[0m DC offset is elevated - may cause issues with some modulations")
        else:
            print("\n  \033[91m[LINK TEST FAILED]\033[0m - Issues detected")
            if not results["signal_detected"]:
                print("  -> No signal detected. Check antenna/cable connections.")
            if not results["snr_pass"]:
                print("  -> SNR too low. Try increasing TX/RX gain or reducing distance.")
            if not results["freq_accuracy_pass"]:
                print("  -> Frequency error too high. May indicate clock issues.")
            if not results.get("tx_verified", False):
                print("  -> TX signal not verified! The received signal may be interference, not your TX.")
                print("     Try a different frequency (e.g., --freq 2.3e9) to avoid WiFi.")

        # Optional: Save spectrum plot
        if save_plot:
            try:
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                # Time domain
                ax1.plot(np.real(rx_data[:1000]), label='I', alpha=0.7)
                ax1.plot(np.imag(rx_data[:1000]), label='Q', alpha=0.7)
                ax1.set_xlabel('Sample')
                ax1.set_ylabel('Amplitude')
                ax1.set_title('Time Domain (First 1000 Samples)')
                ax1.legend()
                ax1.grid(True)

                # Frequency domain
                fft_db = 20 * np.log10(all_metrics[-1]["fft_mag"] + 1e-12)
                ax2.plot(all_metrics[-1]["freq_axis"] / 1e3, fft_db)
                ax2.axvline(x=tone_freq/1e3, color='r', linestyle='--', label=f'Expected Tone ({tone_freq/1e3:.0f} kHz)')
                ax2.axvline(x=0, color='g', linestyle='--', label='DC', alpha=0.5)
                ax2.set_xlabel('Frequency (kHz)')
                ax2.set_ylabel('Power (dB)')
                ax2.set_title(f'Spectrum (SNR: {avg_snr:.1f} dB)')
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()
                plot_path = '/tmp/link_test_spectrum.png'
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"\n  Spectrum plot saved to: {plot_path}")
            except ImportError:
                print("\n  [Note] matplotlib not available - skipping plot")

        print("=" * 70)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sdr is not None:
            try:
                sdr.tx_destroy_buffer()
            except:
                pass
            try:
                sdr.rx_destroy_buffer()
            except:
                pass

    return results


def run_crosstalk_test(ip):
    """
    Test for TX-RX crosstalk/leakage without cable connection.
    Useful for diagnosing internal leakage issues.
    """
    print("\n" + "=" * 70)
    print("  CROSSTALK / ISOLATION TEST")
    print("=" * 70)
    print(f"Target IP: {ip}")
    print("\n[NOTE] For this test, DISCONNECT the TX-RX cable to measure isolation.")
    print("-" * 70)

    try:
        sdr = adi.Pluto(uri=ip)

        sdr.sample_rate = int(CONFIG["fs"])
        sdr.tx_lo = int(CONFIG["fc"])
        sdr.rx_lo = int(CONFIG["fc"])
        sdr.tx_rf_bandwidth = int(CONFIG["fs"])
        sdr.rx_rf_bandwidth = int(CONFIG["fs"])
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = 70  # Max gain to detect leakage
        sdr.tx_hardwaregain_chan0 = 0   # Full TX power
        sdr.rx_buffer_size = 1024 * 16

        # Measure baseline (TX off)
        print("\n[1/3] Measuring baseline noise (TX off)...")
        sdr.tx_enabled_channels = [0]
        sdr.tx_cyclic_buffer = True
        # Send zeros
        zeros = np.zeros(1024 * 16, dtype=np.complex64)
        sdr.tx(zeros)
        time.sleep(0.2)

        for _ in range(3):
            sdr.rx()
        baseline_data = sdr.rx()
        baseline_power = np.mean(np.abs(baseline_data) ** 2)
        baseline_peak = np.max(np.abs(baseline_data))
        print(f"      Baseline Peak: {baseline_peak:.1f}")
        print(f"      Baseline Power: {10*np.log10(baseline_power+1e-12):.1f} dB")

        sdr.tx_destroy_buffer()

        # Measure with TX tone
        print("\n[2/3] Measuring with TX tone active...")
        tone_freq = 100e3
        N = 1024 * 16
        t = np.arange(N) / CONFIG["fs"]
        tx_tone = 0.9 * np.exp(1j * 2 * np.pi * tone_freq * t)
        tx_data = (tx_tone * (2**14)).astype(np.complex64)

        sdr.tx(tx_data)
        time.sleep(0.2)

        for _ in range(3):
            sdr.rx()
        leakage_data = sdr.rx()
        leakage_peak = np.max(np.abs(leakage_data))

        # Analyze spectrum for tone
        metrics = analyze_spectrum(leakage_data, CONFIG["fs"], tone_freq)

        print(f"      Leakage Peak: {leakage_peak:.1f}")
        print(f"      Tone Power: {metrics['tone_power_db']:.1f} dB")

        sdr.tx_destroy_buffer()

        # Calculate isolation
        print("\n[3/3] Isolation Analysis")
        print("-" * 70)

        if leakage_peak > baseline_peak * 2:
            isolation_db = 20 * np.log10(baseline_peak / leakage_peak + 1e-12)
            print(f"      TX-RX Isolation: {abs(isolation_db):.1f} dB")

            if abs(isolation_db) < 20:
                print("      \033[91m[WARNING]\033[0m Low isolation - significant crosstalk detected!")
                print("      This may cause issues with loopback tests.")
            else:
                print("      \033[92m[OK]\033[0m Isolation is acceptable")
        else:
            print("      \033[92m[EXCELLENT]\033[0m No significant crosstalk detected")
            print("      Leakage is at or below noise floor.")

        print("=" * 70)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()


def run_gain_sweep(ip):
    """
    Sweep RX gain to find optimal operating point for the cable loopback setup.
    """
    print("\n" + "=" * 70)
    print("  RX GAIN SWEEP TEST")
    print("=" * 70)
    print(f"Target IP: {ip}")
    print("-" * 70)

    try:
        sdr = adi.Pluto(uri=ip)

        sdr.sample_rate = int(CONFIG["fs"])
        sdr.tx_lo = int(CONFIG["fc"])
        sdr.rx_lo = int(CONFIG["fc"])
        sdr.tx_rf_bandwidth = int(CONFIG["fs"])
        sdr.rx_rf_bandwidth = int(CONFIG["fs"])
        sdr.gain_control_mode_chan0 = "manual"
        sdr.tx_hardwaregain_chan0 = -10
        sdr.rx_buffer_size = 1024 * 16

        # Generate tone
        tone_freq = 100e3
        N = 1024 * 16
        t = np.arange(N) / CONFIG["fs"]
        tx_tone = 0.5 * np.exp(1j * 2 * np.pi * tone_freq * t)
        tx_data = (tx_tone * (2**14)).astype(np.complex64)

        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_data)
        time.sleep(0.3)

        print(f"\n{'RX Gain (dB)':<15} | {'Peak':>10} | {'SNR (dB)':>10} | {'Status':<20}")
        print("-" * 60)

        best_gain = 0
        best_snr = -100

        # Sweep from low to high gain
        for rx_gain in range(0, 75, 5):
            sdr.rx_hardwaregain_chan0 = rx_gain
            time.sleep(0.1)

            # Flush and capture
            for _ in range(2):
                sdr.rx()
            rx_data = sdr.rx()

            metrics = analyze_spectrum(rx_data, CONFIG["fs"], tone_freq)
            peak = metrics["signal_peak"]
            snr = metrics["snr_db"]

            # Determine status
            if peak > 30000:
                status = "\033[91mSATURATED\033[0m"
            elif peak < 500:
                status = "\033[93mWEAK\033[0m"
            elif snr > best_snr:
                best_snr = snr
                best_gain = rx_gain
                status = "\033[92mGOOD\033[0m"
            else:
                status = "OK"

            print(f"{rx_gain:<15} | {peak:>10.0f} | {snr:>10.1f} | {status}")

        sdr.tx_destroy_buffer()

        print("-" * 60)
        print(f"\n  Recommended RX Gain: {best_gain} dB (SNR: {best_snr:.1f} dB)")
        print("=" * 70)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()


def run_digital_loopback_test(ip):
    """Run a packet test in digital loopback mode."""
    print(f"\n=== DIGITAL LOOPBACK PACKET TEST ===")
    print(f"Target IP: {ip}")
    
    try:
        sdr = adi.Pluto(uri=ip)
        
        # Configure for Digital Loopback
        sdr.sample_rate = int(3e6)
        sdr.rx_lo = int(915e6)
        sdr.tx_lo = int(915e6)
        sdr.rx_buffer_size = 32768
        
        # Robustness
        if hasattr(sdr, "_rxadc") and hasattr(sdr._rxadc, "set_kernel_buffers_count"):
            sdr._rxadc.set_kernel_buffers_count(4)
        
        # Enable Digital Loopback
        print("Enabling Digital Loopback (BIST)...")
        ctx = sdr.ctx
        phy = ctx.find_device('ad9361-phy')
        if phy:
            phy.debug_attrs['loopback'].value = '1'
        
        # Generate Packet
        print("Generating Test Packet...")
        # Simple preamble + payload
        preamble = np.array([1, 1, 1, 1, -1, -1, 1, 1] * 10, dtype=np.complex64)
        payload = np.random.choice([1, -1], 1024).astype(np.complex64)
        tx_data = np.concatenate([preamble, payload])
        tx_data *= 2**14 * 0.5 # Scale
        
        # Transmit
        sdr.tx_cyclic_buffer = True
        sdr.tx(tx_data)
        print("Transmitting (Cyclic)...")
        
        # Receive
        print("Receiving...")
        # Clear buffer
        for _ in range(5):
            sdr.rx()
            
        rx_data = sdr.rx()
        
        # Verify
        peak = np.max(np.abs(rx_data))
        print(f"RX Peak Amplitude: {peak:.1f}")
        
        # Cross Correlation
        corr = np.correlate(rx_data, preamble, mode='valid')
        peak_corr = np.max(np.abs(corr))
        print(f"Correlation Peak: {peak_corr:.1f}")
        
        if peak_corr > 1000:
            print("\033[92mPACKET RECEIVED SUCCESSFULLY\033[0m")
        else:
            print("\033[91mPACKET LOSS / FAILURE\033[0m")
            
        # Disable Loopback
        if phy:
            phy.debug_attrs['loopback'].value = '0'
            
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        
    except Exception as e:
        print(f"Digital Loopback Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PlutoSDR TX/RX Link Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  diagnostic    - Full device diagnostic (run this first if having issues!)
  link_test     - Comprehensive TX/RX link quality test (recommended for cable loopback)
  crosstalk     - Test TX-RX isolation (run with cable disconnected)
  gain_sweep    - Find optimal RX gain for your setup
  loopback      - Legacy 2x2 matrix loopback test
  tx            - Continuous TX beacon mode
  rx            - RX scanner mode
  digital_test  - Digital loopback test (internal BIST)
  modes         - Test ENSM mode switching
  restart       - Reboot the PlutoSDR device

Test Presets (for link_test mode):
  cable_30db     - Cable with 30dB attenuator (default)
  cable_direct   - Direct cable, no attenuator (low power!)
  antenna_close  - Antennas close together (<1m)
  antenna_far    - Antennas further apart (>1m, max power)

Example:
  python sdr_auto_tune.py --mode link_test --ip ip:192.168.3.2
  python sdr_auto_tune.py --mode link_test --ip ip:192.168.3.2 --preset antenna_close
  python sdr_auto_tune.py --mode link_test --ip ip:192.168.3.2 --preset antenna_far --rx_gain 60
  python sdr_auto_tune.py --mode gain_sweep --ip ip:192.168.3.2 --save_plot
        """
    )
    parser.add_argument("--mode", required=True,
                        choices=['tx', 'rx', 'loopback', 'modes', 'digital_test', 'restart',
                                 'link_test', 'crosstalk', 'gain_sweep', 'diagnostic'])
    parser.add_argument("--ip", default="ip:192.168.3.2", help="PlutoSDR IP address (default: ip:192.168.3.2)")
    parser.add_argument("--channel", type=int, default=0, help="TX Channel (0 or 1)")
    parser.add_argument("--save_plot", action="store_true", help="Save spectrum plot (requires matplotlib)")
    parser.add_argument("--preset", default="cable_30db",
                        choices=["cable_30db", "cable_direct", "antenna_close", "antenna_far"],
                        help="Test preset: cable_30db, cable_direct, antenna_close, antenna_far")
    parser.add_argument("--tx_gain", type=float, default=None, help="Override TX gain (dB)")
    parser.add_argument("--rx_gain", type=float, default=None, help="Override RX gain (dB)")
    parser.add_argument("--freq", type=float, default=None, help="Override center frequency (Hz), e.g., 2.3e9 for 2.3 GHz")
    parser.add_argument("--tone_freq", type=float, default=100e3, help="Tone offset frequency (Hz), default 100kHz")

    args = parser.parse_args()

    if args.mode == 'diagnostic':
        run_device_diagnostic(args.ip)
    elif args.mode == 'link_test':
        run_link_test(args.ip, save_plot=args.save_plot, preset=args.preset,
                      tx_gain=args.tx_gain, rx_gain=args.rx_gain,
                      freq=args.freq, tone_freq=args.tone_freq)
    elif args.mode == 'crosstalk':
        run_crosstalk_test(args.ip)
    elif args.mode == 'gain_sweep':
        run_gain_sweep(args.ip)
    elif args.mode == 'tx':
        run_tx(args.ip, args.channel)
    elif args.mode == 'rx':
        run_rx_scanner(args.ip)
    elif args.mode == 'loopback':
        run_loopback_test(args.ip)
    elif args.mode == 'modes':
        run_mode_test(args.ip)
    elif args.mode == 'digital_test':
        run_digital_loopback_test(args.ip)
    elif args.mode == 'restart':
        restart_device(args.ip)
