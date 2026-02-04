import adi
import sys
import time

def check_status(uri="ip:192.168.3.2"):
    print(f"Connecting to {uri}...")
    try:
        sdr = adi.Pluto(uri=uri)
        print("Connected.")
        
        print(f"Sample Rate: {sdr.sample_rate}")
        print(f"RX LO: {sdr.rx_lo}")
        print(f"TX LO: {sdr.tx_lo}")
        
        # Check PHY
        ctx = sdr.ctx
        phy = ctx.find_device('ad9361-phy')
        if phy:
            print("\n--- PHY Settings ---")
            for attr in ['ensm_mode', 'ensm_mode_available', 'rssi_chan0', 'gain_control_mode_chan0', 'voltage0_hardwaregain']:
                if attr in phy.attrs:
                    print(f"{attr}: {phy.attrs[attr].value}")
                elif attr in phy.debug_attrs:
                     print(f"{attr}: {phy.debug_attrs[attr].value}")
            
            # Check Loopback
            if 'loopback' in phy.debug_attrs:
                print(f"Loopback Trigger: {phy.debug_attrs['loopback'].value}")
                
            # Check BIST status if possible (registers)
            # Register 0x3F5 is BIST Config, 0x3F6 is BIST Control... 
            # We can read reg via debug_attrs 'direct_reg_access' usually?
            # PyADI-IIO: sdr._ctrl.reg_read(addr)
            
            try:
                # Reg 0x000 - SPI Check (Should be 0x00 or similar depending on sil rev)
                # 0x037 is ENSM Status
                val = sdr._ctrl.reg_read(0x037)
                print(f"Reg 0x037 (ENSM Status): {hex(val)}")
            except Exception as e:
                print(f"Cannot read registers: {e}")

        else:
            print("PHY device not found!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_status()
