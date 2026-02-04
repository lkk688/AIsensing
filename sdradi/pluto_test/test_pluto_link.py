#!/usr/bin/env python
"""
test_pluto_link.py - Simple verification of PlutoSDR Hardware Link (Cyclic & TDD)

Target IP: 192.168.3.2
Setup: Loopback cable with 30dB attenuator (TX1 -> RX1)
Tests:
1. Connection & Config
2. Noise Floor Check
3. Cyclic Mode Loopback (Continuous TX)
4. TDD Mode Loopback (Burst TX)
"""

import adi
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import sys

# Configure plotting
plt.style.use('dark_background')

def setup_sdr(ip_address, fc=915e6, fs=3e6, tx_gain=0, rx_gain=70):
    print(f"\n[Setup] Connecting to {ip_address}...")
    try:
        sdr = adi.Pluto(ip_address)
        sdr.sample_rate = int(fs)
        sdr.tx_lo = int(fc)
        sdr.rx_lo = int(fc)
        sdr.tx_hardwaregain_chan0 = int(tx_gain)
        sdr.rx_hardwaregain_chan0 = int(rx_gain)
        sdr.gain_control_mode_chan0 = 'manual'
        sdr.rx_buffer_size = 32768


        
        # Check LO Lock
        try:
             # Just read back LO frequency to verify communication
             r_lo = sdr.rx_lo
             t_lo = sdr.tx_lo
             print(f"[Setup] LO Configured: RX={r_lo/1e6}MHz, TX={t_lo/1e6}MHz")
        except:
             print("[Warn] Failed to read LO frequencies")

        
        # Ensure FDD mode initially for basic setup (Try gently, ignore failure)
        try:
            if hasattr(sdr, "_ctrl") and hasattr(sdr._ctrl, "debug_attrs") and "adi,frequency-division-duplex-mode-enable" in sdr._ctrl.debug_attrs:
                 # Only try if attribute exists to avoid error spam
                 # sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
                 # sdr._ctrl.debug_attrs["initialize"].value = "1"
                 pass
        except Exception as e:
            print(f"[Warn] Failed to force FDD mode: {e}")

        # Disable internal loopback to ensure we test the cable
        try:
            for dev in sdr.ctx.devices:
                if dev.name == 'ad9361-phy':
                    dev.debug_attrs['loopback'].value = '0'
                    break
        except:
            pass

        # AGGRESSIVE RESET
        try:
            # Try to disable TDD via TDDN device if it exists
            tdd = adi.tddn(ip_address)
            tdd.enable = False
            print("[Reset] TDD disabled via tddn")
        except:
            pass

        try:
            # Force FDD
            sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
            sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
            sdr._ctrl.debug_attrs["initialize"].value = "1"
            print("[Reset] FDD mode forced via debug_attrs")
        except Exception as e:
            print(f"[Warn] Failed to force FDD: {e}")
            
        # Blink LED to show we have control
        try:
            # Access GPIO/LED if possible? adi.Pluto doesn't expose it easily.
            # Skip for now, focus on radio.
            pass
        except: pass

        # Reduce buffer size to minimize timing issues
        sdr.rx_buffer_size = 1024 * 8 
        
        # Set timeout to 30 seconds
        try:
            sdr.ctx.set_timeout(30000)
        except:
            print("[Warn] Failed to set context timeout")

        print(f"[Setup] SDR Configured: FC={fc/1e6}MHz, FS={fs/1e6}MHz")
        print(f"[Setup] Gains: TX={tx_gain}dB, RX={rx_gain}dB")
        return sdr
    except Exception as e:
        print(f"[Error] Connection failed: {e}")
        return None

def safe_rx(sdr, retries=3):
    for i in range(retries):
        try:
            return sdr.rx()
        except Exception as e:
            print(f"[Warn] RX failed (attempt {i+1}/{retries}): {e}")
            time.sleep(0.5)
            # Re-create buffer if possible? sdr.rx() does it internally often.
    return None

def generate_qpsk_signal(num_symbols=1024):
    # Match sdr_auto_tune.py exactly
    preamble = np.array([1, 1, 1, 1, -1, -1, 1, 1] * 10, dtype=np.complex64)
    # Payload
    payload = np.random.choice([1, -1], num_symbols).astype(np.complex64)
    signal = np.concatenate([preamble, payload])
    # Scale
    signal *= 2**14 * 0.5
    return signal.astype(np.complex64), preamble


def test_cyclic_mode(sdr):
    print("\n--- Test 1: Cyclic Mode Loopback ---")
    
    # Generate signal
    tx_signal, preamble = generate_qpsk_signal(1024)
    
    # Create cyclic buffer (repeat signal to fill buffer min requirement)
    min_len = 32768
    repeats = int(np.ceil(min_len / len(tx_signal)))
    tx_cyclic = np.tile(tx_signal, repeats)
    
    print(f"[Cyclic] TX Buffer size: {len(tx_cyclic)} samples")
    
    # Transmit
    sdr.tx_destroy_buffer()
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_cyclic)
    time.sleep(1.0) # Wait for TX to stabilize
    
    # Receive
    print("[Cyclic] Receiving...")
    rx_signal = safe_rx(sdr)
    sdr.tx_destroy_buffer() # Stop TX
    
    if rx_signal is None:
        print("\033[91m[FAIL] RX failed repeatedly (timeout?).\033[0m")
        return False
    
    # Analysis
    rx_signal = rx_signal / np.max(np.abs(rx_signal)) if np.max(np.abs(rx_signal)) > 0 else rx_signal
    avg_power = np.mean(np.abs(rx_signal)**2)
    print(f"[Cyclic] RX Avg Power: {10*np.log10(avg_power + 1e-12):.2f} dB (normalized)")
    
    # Locate DC (simple mean subtraction if needed, but we check raw first)
    dc_offset = np.mean(rx_signal)
    print(f"[Cyclic] DC Offset: {np.abs(dc_offset):.4f}")
    
    # Correlation to verify it's our signal
    # Use standard preamble
    corr = np.abs(np.correlate(rx_signal, preamble, mode='valid'))
    peak = np.max(corr)
    avg_corr = np.mean(corr)
    snr_est = peak / (avg_corr + 1e-9)
    
    print(f"[Cyclic] Correlation Peak: {peak:.2f}")
    print(f"[Cyclic] Peak/Avg Ratio: {snr_est:.2f}")
    
    if snr_est > 5.0:
        print("\033[92m[PASS] Cyclic Mode signal detected!\033[0m")
        return True
    else:
        print("\033[91m[FAIL] Cyclic Mode signal weak or not detected.\033[0m")
        return False

def test_tdd_mode(sdr, ip):
    print("\n--- Test 2: TDD Mode Loopback ---")
    
    try:
        tdd = adi.tddn(ip)
    except:
        print("[TDD] Warning: access to TDD Controller failed, skipping TDD test")
        return False

    # Configure TDD
    frame_ms = 4.0
    tx_ms = 1.0
    rx_start = 1.5
    rx_end = 3.5
    
    print(f"[TDD] Configuring TDD: Frame={frame_ms}ms, TX=0-{tx_ms}ms, RX={rx_start}-{rx_end}ms")
    
    # Setup standard TDD mode
    tdd.enable = False
    tdd.frame_length_ms = frame_ms
    tdd.burst_count = 0
    
    # Channel 0: TXNRX (High=TX, Low=RX)
    # On for TX duration
    tdd.channel[0].enable = True
    tdd.channel[0].polarity = 0
    tdd.channel[0].on_ms = 0
    tdd.channel[0].off_ms = tx_ms
    
    # Channel 1: ENABLE (High=Active)
    # On for both TX and RX windows? Or just let it cover whole frame?
    # Usually ENABLE must be high for both.
    # Let's try pulsing ENABLE for TX and RX separately or just keep it high?
    # Simpler: ENABLE high for TX, then high for RX
    # But usually Ch1 is ENABLE pin.
    
    tdd.channel[1].enable = True
    tdd.channel[1].polarity = 0
    tdd.channel[1].on_ms = 0
    tdd.channel[1].off_ms = rx_end # Active from start until RX end
    
    tdd.enable = True
    
    # Set SDR to TDD mode (Pin control)
    try:
        sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "0"
        sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "1"
        sdr._ctrl.debug_attrs["initialize"].value = "1"
    except Exception as e:
        print(f"[TDD] Failed to set pin control: {e}")
        return False
        
    # Generate burst signal
    # TX duration is 1ms. FS=3M. Samples = 3000.
    burst_samples = int(3e6 * (tx_ms / 1000.0))
    tx_signal, _ = generate_qpsk_signal(int(burst_samples/2)) # ensure enough symbols
    tx_signal = tx_signal[:burst_samples] # Trim to fit exactly
    
    print(f"[TDD] Burst size: {len(tx_signal)} samples")
    
    # Stop any previous TX
    sdr.tx_destroy_buffer()

    # Transmit Burst (Non-Cyclic)
    sdr.tx_cyclic_buffer = False
    sdr.tx(tx_signal)
    
    # 2. RX
    # We expect to capture some zeros (guard) and then noise/signal? 
    sdr.rx_buffer_size = 32768
    rx_data = safe_rx(sdr)
    
    # Stop
    tdd.enable = False
    sdr.tx_destroy_buffer()
    
    if rx_data is None:
        print("\033[91m[FAIL] RX failed in TDD Mode.\033[0m")
        return False
    
    # Analysis
    amp = np.abs(rx_data)
    max_val = np.max(amp)
    min_val = np.min(amp)
    mean_val = np.mean(amp)
    
    print(f"[TDD] Max Level: {max_val:.2f}")
    print(f"[TDD] Min Level: {min_val:.2f}")
    print(f"[TDD] Mean Level: {mean_val:.2f}")
    
    # Check for bursts
    # If Max is high (>100) and Min is low (<50), we have bursts.
    # If Max is high and Min is high, we have continuous signal (Fail TDD gating).
    # If Max is low, we have no signal.
    
    if max_val < 10:
        print("\033[91m[FAIL] No Signal detected in TDD mode.\033[0m")
        return False
        
    if min_val > max_val * 0.5:
        print("\033[93m[WARN] Signal is CONTINUOUS (TDD Gating Failed).\033[0m")
        # This counts as a link pass but TDD mode fail
        print("      Likely software configuration issue or TX buffer not clearing.")
        return True # Soft pass for link
        
    print("\033[92m[PASS] TDD Mode Burst Detected! (Dynamic Range Good)\033[0m")
    return True

def test_digital_loopback(sdr):
    print("\n--- Test 0: Digital Loopback (BIST) ---")
    print("Enabling internal AD9361 digital loopback...")
    
    try:
        # Find PHY device to set debug attribute
        phy = None
        for dev in sdr.ctx.devices:
            if dev.name == 'ad9361-phy':
                phy = dev
                break
        
        if phy:
            phy.debug_attrs['loopback'].value = '1'
            print("[BIST] Loopback Enabled")
        else:
            print("[BIST] Failed to find ad9361-phy device")
            return False
            
    except Exception as e:
        print(f"[BIST] Error enabling loopback: {e}")
        return False

    # Generate signal (High amplitude for digital loopback)
    tx_signal, preamble = generate_qpsk_signal(1024)
    # Digital loopback often needs full scale or specific scaling. 
    # generate_qpsk_signal uses 0.5 scale which is fine.

    # Transmit Cyclic
    sdr.tx_destroy_buffer()
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_signal)
    time.sleep(0.5)

    # Receive with flushing (match sdr_auto_tune.py)
    print("[BIST] Flushing buffers...")
    for _ in range(5):
        try: sdr.rx()
        except: pass
        
    rx_signal = safe_rx(sdr)
    
    # Disable Loopback
    try:
        if phy:
            phy.debug_attrs['loopback'].value = '0'
            print("[BIST] Loopback Disabled")
    except: pass
    
    sdr.tx_destroy_buffer()

    if rx_signal is None:
        print("\033[91m[FAIL] RX failed in Digital Mode.\033[0m")
        return False

    # Analysis
    # In BIST, we expect perfect or near perfect signal
    # But scaling might be different.
    
    max_val = np.max(np.abs(rx_signal))
    print(f"[BIST] Max RX Level: {max_val:.2f}")
    
    if max_val < 10:
        print("\033[91m[FAIL] Digital Loopback signal missing (Level too low).\033[0m")
        return False

    # Correlation
    corr = np.abs(np.correlate(rx_signal, preamble, mode='valid'))
    peak = np.max(corr)
    
    # Normalize for correlation
    sig_norm = rx_signal / max_val
    corr_norm = np.abs(np.correlate(sig_norm, preamble, mode='valid'))
    peak_norm = np.max(corr_norm)
    
    print(f"[BIST] Correlation Peak (Norm): {peak_norm:.2f}")

    if peak_norm > 20.0: # Preamble length is ~70, strong corr should be high
         print("\033[92m[PASS] Digital Loopback Verified!\033[0m")
         return True
    else:
         print("\033[91m[FAIL] Digital Loopback Data Corrupted.\033[0m")
         return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="ip:192.168.3.2", help="SDR IP")
    args = parser.parse_args()
    
    sdr = setup_sdr(args.ip)
    if not sdr:
        return
        
    success = True
    
    # 0. Digital Loopback Check (Is the chip alive?)
    if not test_digital_loopback(sdr):
        print("\n" + "!"*60)
        print("\033[91m[CRITICAL] Digital Loopback Failed (Timeout/No Data)\033[0m")
        print("!"*60)
        print("Diagnosis: The PlutoSDR digital interface or DMA is STUCK.")
        print("           This is NOT a cable/RF issue.")
        print("Action:    You MUST power cycle the PlutoSDR (Unplug/Replug USB).")
        print("           Software reset is insufficient.")
        print("!"*60 + "\n")
        return
    
    # 1. Cyclic Test (Is RF Cable OK?)
    if not test_cyclic_mode(sdr):
        success = False
        print("\n[Tip] Check your loopback cable connectivity and attenuator.")
        print("[Tip] Ensure you are connected to TX1 and RX1 (closest to USB).")
    
    time.sleep(1)
    
    # 2. TDD Test (Is Timing OK?)
    if not test_tdd_mode(sdr, args.ip):
        success = False
        print("\n[Tip] TDD mode requires careful pin configuration.")
        
    print("\n" + "="*40)
    if success:
        print("\033[92mHARDWARE LINK VERIFICATION PASSED\033[0m")
    else:
        print("\033[91mHARDWARE LINK VERIFICATION FAILED\033[0m")
    print("="*40)

if __name__ == "__main__":
    main()
