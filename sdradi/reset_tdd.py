import adi
import sys

uri = "ip:192.168.2.2"
print(f"Connecting to {uri}...")
try:
    tdd = adi.tddn(uri)
    print("Found TDD controller")
    tdd.enable = False
    print("Disabled TDD global")
    
    # Try to disable channels if they exist
    # pyadi-iio tddn class structure might vary, let's try generic approach
    if hasattr(tdd, 'channel'):
        for i, ch in enumerate(tdd.channel):
            try:
                ch.enable = False
                print(f"Disabled channel {i}")
            except Exception as e:
                print(f"Failed to disable channel {i}: {e}")
    
    try:
        tdd.sync_external = False
        print("Disabled sync_external")
    except: pass
    
    try:
        tdd.sync_internal = False
        print("Disabled sync_internal")
    except: pass
    
    print("TDD Reset Complete")
    
except Exception as e:
    print(f"Error: {e}")
