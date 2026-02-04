import adi
import sys

uris = ["ip:192.168.2.2", "ip:pluto.local", "usb:1.4.5", "ip:192.168.2.1"]

print("Testing SDR Connection...")
for uri in uris:
    try:
        print(f"Trying {uri}...", end=" ")
        sdr = adi.Pluto(uri=uri)
        print("SUCCESS")
        print(f"Sample Rate: {sdr.sample_rate}")
    except Exception as e:
        print(f"FAILED ({e})")
print("Done.")
