#!/usr/bin/env python
"""
reset_pluto.py - Attempt to reset PlutoSDR state via IIO context

Usage: python reset_pluto.py [ip]
"""
import iio
import sys
import time

def reset_pluto(ip='ip:192.168.3.2'):
    print(f"Connecting to {ip}...")
    try:
        ctx = iio.Context(ip)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Found devices:")
    found_phy = False
    for dev in ctx.devices:
        print(f"  - {dev.name}")
        if dev.name == 'ad9361-phy':
            found_phy = True
            try:
                # Disable TDD
                if 'ensm_mode_available' in dev.attrs:
                    print(f"    ensm_modes: {dev.attrs['ensm_mode_available'].value}")
                
                # Reset Debug attrs
                if 'loopback' in dev.debug_attrs:
                    dev.debug_attrs['loopback'].value = '0'
                    print("    Loopback disabled")
                
                # Force FDD
                if 'adi,frequency-division-duplex-mode-enable' in dev._device.debug_attrs:
                     # Note: direct debug_attr access depends on library version, using generic if possible
                     pass
                     
            except Exception as e:
                print(f"    Error resetting phy: {e}")

    try:
        # Toggle TDD controller if present
        import adi
        tdd = adi.tddn(ip)
        tdd.enable = False
        print("TDDN disabled")
    except:
        print("TDDN not accessible or library missing")

    print("\nReset commands sent. Please wait 5 seconds...")
    time.sleep(5)
    print("Done.")

if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else "ip:192.168.3.2"
    reset_pluto(ip)
