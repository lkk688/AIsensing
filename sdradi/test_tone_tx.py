import adi
import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Simple Sine Wave Transmitter")
    parser.add_argument("--ip", required=True, help="SDR IP Address")
    parser.add_argument("--fc", type=float, default=915e6, help="Center Frequency")
    parser.add_argument("--fs", type=float, default=2e6, help="Sample Rate") # Lower FS for easier debug
    parser.add_argument("--tone_freq", type=float, default=100e3, help="Tone Offset Frequency (Hz)")
    args = parser.parse_args()

    print(f"Connecting to SDR at {args.ip}...")
    try:
        sdr = adi.Pluto(uri=args.ip)
    except Exception as e:
        print(f"Error connecting: {e}")
        sys.exit(1)

    # Configure SDR
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_cyclic_buffer = True # Hardware repeat
    sdr.tx_rf_bandwidth = int(args.fs)
    
    # Max Gain
    sdr.tx_hardwaregain_chan0 = 0 
    
    print(f"Config: FC={args.fc/1e6}MHz, FS={args.fs/1e6}MHz, Gain=0dB")
    print(f"Generating Tone at +{args.tone_freq/1e3} kHz...")

    # Generate Sine Wave
    fs = int(args.fs)
    fc = int(args.tone_freq)
    N = 1024 * 16 # Buffer size
    t = np.arange(N) / fs
    samples = np.exp(1j * 2 * np.pi * fc * t) * 0.9 # 90% Amplitude (Scale factor)
    samples = samples.astype(np.complex64)
    # Scale to 14-bit integer range if needed? ADI python takes 0-1 complex or int?
    # ADI python bindings usually take complex64 (scaled 2^14 internally) or 0-1?
    # Let's use large amplitude to be safe. 
    # Actually adi handles 2^14 scaling often. Let's try 2^14 scale just in case.
    samples *= 2**14 

    print("Uploading Cyclic Buffer...")
    sdr.tx(samples)
    print("Transmitting... Press Ctrl+C to stop.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping...")
        sdr.tx_destroy_buffer()

if __name__ == "__main__":
    main()
