import argparse
import numpy as np
import time
from myadiclass import SDR

# Replicate Probe Tone Generation
def generate_probe_frame(fs):
    t = np.arange(512)/fs
    tone_freq = 50e3
    tone = 0.8 * np.exp(1j*2*np.pi*tone_freq*t)
    tone = tone * (2**14)
    return np.tile(tone, 100) # Buffer > 50k samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='ip:192.168.3.2')
    args = parser.parse_args()

    print(f"Testing SDR Class with Probe Tone on {args.ip}...")
    
    # Initialize Class
    # Note: myadiclass defaults fs=6M. We force 2M.
    # We use explicit channel configs to match probe_tx.py
    sdr_obj = SDR(
        SDR_IP=args.ip, 
        SDR_SAMPLERATE=2e6, 
        SDR_BANDWIDTH=1e6, # Match probe_tx 1MHz
        device_name='pluto',
        Rx_CHANNEL=1, 
        Tx_CHANNEL=1
    )
    
    # Setup TX
    sdr_obj.SDR_TX_setup(
        cyclic_buffer=True, 
        tx_bandwidth=1e6, 
        tx1_gain=-5 # LOUD
    )
    # FORCE CHAN 1 GAIN (Match probe_tx.py)
    sdr_obj.sdr.tx_hardwaregain_chan1 = -5
    
    # Generate Signal
    data = generate_probe_frame(2e6)
    
    # Transmit using simple method (bypassing SDR_TX_send complexity for now)
    print(f"Transmitting... Peak: {np.max(np.abs(data))}")
    sdr_obj.sdr.tx(data)
    
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
