
import sys
import os
import numpy as np
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from radarappwdevice5 import RadarWorker

def test_playback():
    print("Creating dummy application...")
    # QApplication is needed for QThread/Signals
    app = QApplication(sys.argv) 

    print("Creating dummy data...")
    # Shape: [Frames, Rx, Nc, Ns] -> [5, 1, 64, 256]
    # Use simple beat signal data
    frames = 5
    rx = 1
    nc = 64
    ns = 256
    dummy_data = np.random.randn(frames, rx, nc, ns) + 1j * np.random.randn(frames, rx, nc, ns)
    
    # Save to temp file
    fname = "dummy_radar_data.npy"
    np.save(fname, dummy_data)
    print(f"Saved {fname}")

    print("Initializing RadarWorker...")
    worker = RadarWorker()
    worker.mode = 'playback'
    
    print("Loading data...")
    success = worker.load_data(fname)
    if not success:
        print("FAIL: load_data returned False")
        sys.exit(1)

    if worker.playback_data is None:
        print("FAIL: playback_data is None")
        sys.exit(1)
        
    print(f"Data loaded: {worker.playback_data.shape}")

    # Set up signal capture
    received_frames = 0
    
    def handle_data_ready(data):
        nonlocal received_frames
        rdm, dets, tgts, fas, ra, va = data
        received_frames += 1
        print(f"Received Frame {received_frames}: RDM shape {rdm.shape}, Dets {len(dets)}")
        if received_frames >= 3:
            print("Received enough frames. Stopping.")
            worker.stop()
            app.quit()

    worker.data_ready.connect(handle_data_ready)
    
    print("Starting worker...")
    worker.start()

    # Run event loop (blocking until app.quit())
    app.exec()

    print("Test finished.")
    
    # Cleanup
    if os.path.exists(fname):
        os.remove(fname)

    if received_frames >= 3:
        print("PASS: Playback verified.")
        sys.exit(0)
    else:
        print("FAIL: Did not receive enough frames.")
        sys.exit(1)

if __name__ == "__main__":
    test_playback()
