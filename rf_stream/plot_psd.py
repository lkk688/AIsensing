
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_spectrum(run_dir):
    csv_path = os.path.join(run_dir, "captures.csv")
    if not os.path.exists(csv_path):
        print("No CSV found")
        return
    
    # We don't need the CSV, just look for .npz if they exist
    # Oh, I disabled save_npz.
    # I'll just use the captures in the current buffer? No.
    
    # I'll modify the RX script to save one raw capture as .npz
    pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plot_spectrum(sys.argv[1])
