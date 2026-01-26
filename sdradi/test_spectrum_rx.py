import adi
import numpy as np
import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Simple Spectrum Analyzer (Text Based)")
    parser.add_argument("--ip", required=True, help="SDR IP Address")
    parser.add_argument("--fc", type=float, default=915e6, help="Center Frequency")
    parser.add_argument("--fs", type=float, default=2e6, help="Sample Rate")
    args = parser.parse_args()

    print(f"Connecting to SDR at {args.ip}...")
    try:
        sdr = adi.Pluto(uri=args.ip)
    except Exception as e:
        print(f"Error connecting: {e}")
        sys.exit(1)

    # Configure SDR
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.fs)
    sdr.rx_buffer_size = 1024 * 8
    
    # Manual High Gain
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 60 # Max sensitivity
    
    print(f"Config: FC={args.fc/1e6}MHz, FS={args.fs/1e6}MHz, Gain=60dB")
    print("Scanning for Strong Signals...")

    try:
        while True:
            # Receive
            x = sdr.rx()
            
            # FFT
            xf = np.fft.fft(x)
            xf_mag = np.abs(xf)
            xf_mag = np.fft.fftshift(xf_mag) # Shift zero freq to center
            freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1/args.fs))
            
            # Find Peak
            peak_idx = np.argmax(xf_mag)
            peak_val = xf_mag[peak_idx]
            peak_freq = freqs[peak_idx]
            
            # Calculate total energy (to detect if antenna is unplugged/dead)
            total_energy = np.sum(xf_mag)
            
            peak_db = 20 * np.log10(peak_val + 1e-9)
            
            # Visual Bar
            bar = "#" * int(peak_db / 5)
            
            print(f"Peak Freq: {peak_freq/1e3:+.1f} kHz | Mag: {peak_val:.0f} ({peak_db:.1f} dB) | {bar}")
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping...")
        sdr.rx_destroy_buffer()

if __name__ == "__main__":
    main()
