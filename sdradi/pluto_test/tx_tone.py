import adi
import numpy as np
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='ip:192.168.2.2')
    parser.add_argument('--fc', type=float, default=2550e6)
    parser.add_argument('--gain', type=int, default=0)
    args = parser.parse_args()

    print(f"Connecting to SDR {args.ip}...")
    sdr = adi.Pluto(uri=args.ip)
    sdr.sample_rate = 2000000
    sdr.tx_lo = int(args.fc)
    sdr.tx_hardwaregain_chan0 = args.gain
    sdr.tx_cyclic_buffer = True

    print(f"Generating Tone at {args.fc/1e6} MHz, Gain {args.gain} dB...")
    fs = int(sdr.sample_rate)
    N = 2048
    t = np.arange(N) / fs
    f_tone = 100e3 # 100 kHz offset
    ts = 0.5 * np.exp(1j * 2 * np.pi * f_tone * t) # 0.5 -6dBFS

    sdr.tx(ts) # Cyclic trasmit
    
    print("Transmitting Continuous Tone. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
