import adi
import sys

def test_connection(uri="ip:192.168.2.2"):
    print(f"Testing connection to {uri} with adi.Pluto...")
    try:
        sdr = adi.Pluto(uri=uri)
        print("Success! Connected to Pluto.")
        print(f"Sample Rate: {sdr.sample_rate}")
        print(f"Center Freq: {sdr.rx_lo}")
        print("Drivers working correctly.")
        del sdr
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else "ip:192.168.2.2"
    test_connection(uri)
