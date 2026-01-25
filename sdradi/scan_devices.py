import iio

print("Scanning for IIO contexts...")
try:
    ctxs = iio.scan_contexts()
    print(f"Found {len(ctxs)} contexts:")
    for key, val in ctxs.items():
        print(f"  URI: {key} -> {val}")
except Exception as e:
    print(f"Scan failed: {e}")

# Try direct connect to what we think exists
uris = ["usb:9.7.5", "usb:1.11.5"]
print("\nTesting direct connection:")
for u in uris:
    try:
        c = iio.Context(u)
        print(f"  [SUCCESS] Connected to {u}: {c.name}")
    except Exception as e:
        print(f"  [FAIL] Could not connect to {u}: {e}")
