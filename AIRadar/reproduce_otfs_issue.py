
import numpy as np
import matplotlib.pyplot as plt
from AIradar_datasetv5 import AIRadarDataset

def reproduce_issue():
    # Initialize dataset with OTFS config
    dataset = AIRadarDataset(
        num_samples=1,
        config_name='config1',
        drawfig=True,
        save_path='data/radar_config1_verification'
    )
    
    # Define a single target
    # Range 20m, Velocity 0m/s (to isolate delay)
    target_range = 20.0
    target_velocity = 0.0
    targets = [{
        'range': target_range,
        'velocity': target_velocity,
        'rcs': 10.0,
        'azimuth': 0,
        'elevation': 0
    }]
    
    print(f"Simulating target at Range={target_range}m, Velocity={target_velocity}m/s")
    
    # Simulate signal
    beat_signal, rdm_db = dataset.simulate_otfs_signal(targets, snr_db=100)
    
    # Find peak in RDM
    # rdm_db shape: (Doppler, Range)
    # Note: The code returns (Nc, Ns) transposed and shifted.
    # Let's check the peak index.
    
    peak_idx = np.unravel_index(np.argmax(rdm_db), rdm_db.shape)
    peak_doppler_idx, peak_range_idx = peak_idx
    
    print(f"Peak found at indices: Doppler={peak_doppler_idx}, Range={peak_range_idx}")
    
    # Calculate expected indices
    # Range resolution
    range_res = dataset.range_resolution
    # Velocity resolution
    velocity_res = dataset.velocity_resolution
    
    # Expected Range Index
    # In OTFS code:
    # range_res = c / (2 * fs)
    # self.range_axis = np.arange(self.Ns) * range_res
    # So index = range / range_res
    
    expected_range_idx = int(round(target_range / (c / (2 * dataset.fs))))
    
    # Expected Doppler Index
    # Doppler bins are shifted. Center is 0 velocity.
    # Nc bins. Center is Nc//2.
    expected_doppler_idx = dataset.Nc // 2 # Since velocity is 0
    
    print(f"Expected indices: Doppler={expected_doppler_idx}, Range={expected_range_idx}")
    
    # Check if they match
    if peak_range_idx == expected_range_idx:
        print("Range Index MATCHES.")
    else:
        print(f"Range Index MISMATCH. Diff: {peak_range_idx - expected_range_idx}")
        # Check if it's wrapped (negative delay)
        # If modulation is inverted, we might see index Ns - expected_range_idx
        if peak_range_idx == dataset.Ns - expected_range_idx:
            print("Range Index is WRAPPED/INVERTED (Ns - expected).")
            
    if peak_doppler_idx == expected_doppler_idx:
        print("Doppler Index MATCHES.")
    else:
        print(f"Doppler Index MISMATCH. Diff: {peak_doppler_idx - expected_doppler_idx}")

    # Test with Velocity
    target_velocity = 10.0
    targets[0]['velocity'] = target_velocity
    print(f"\nSimulating target at Range={target_range}m, Velocity={target_velocity}m/s")
    
    beat_signal, rdm_db = dataset.simulate_otfs_signal(targets, snr_db=100)
    peak_idx = np.unravel_index(np.argmax(rdm_db), rdm_db.shape)
    peak_doppler_idx, peak_range_idx = peak_idx
    
    # Expected Doppler shift
    # v_max = lambda / (4T)
    # velocity_axis goes from -v_max to v_max
    # index = (v / v_res) + Nc/2
    expected_doppler_idx = int(round(target_velocity / dataset.velocity_resolution)) + dataset.Nc // 2
    
    print(f"Peak found at indices: Doppler={peak_doppler_idx}, Range={peak_range_idx}")
    print(f"Expected indices: Doppler={expected_doppler_idx}, Range={expected_range_idx}")
    
    if peak_doppler_idx == expected_doppler_idx:
        print("Doppler Index MATCHES.")
    else:
        print(f"Doppler Index MISMATCH. Diff: {peak_doppler_idx - expected_doppler_idx}")
        # Check for inversion
        # If inverted, v -> -v. Index -> (-v / v_res) + Nc/2
        inverted_doppler_idx = int(round(-target_velocity / dataset.velocity_resolution)) + dataset.Nc // 2
        if peak_doppler_idx == inverted_doppler_idx:
             print("Doppler Index is INVERTED (matches -velocity).")

# Constants needed if not imported from dataset (but they are class members)
c = 3e8

if __name__ == "__main__":
    reproduce_issue()
