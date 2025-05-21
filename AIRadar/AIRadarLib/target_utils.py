import numpy as np
import random

def generate_radar_targets(
    num_targets=1,
    min_range=1,
    max_range=30,
    min_velocity=0.1,
    max_velocity=30,
    min_rcs=5.0,
    max_rcs=30.0,
    azimuth_range=(-45, 45),
    elevation_range=(-10, 10),
    range_factor=0.5,
    velocity_factor=0.5
):
    """
    Generate random radar targets with configurable parameters.
    
    Args:
        num_targets: Number of targets to generate
        min_range: Minimum target range in meters
        max_range: Maximum target range in meters
        min_velocity: Minimum target velocity in m/s
        max_velocity: Maximum target velocity in m/s
        min_rcs: Minimum radar cross-section value
        max_rcs: Maximum radar cross-section value
        azimuth_range: Tuple of (min, max) azimuth angles in degrees
        elevation_range: Tuple of (min, max) elevation angles in degrees
        range_factor: Factor to multiply max_range by for actual range distribution (0-1)
        velocity_factor: Factor to determine velocity distribution (0-1)
        
    Returns:
        List of dictionaries containing target parameters
    """
    # List to store target information
    targets = []
    
    # Generate target parameters
    for _ in range(num_targets):
        # Generate random target parameters with configurable ranges
        distance = random.uniform(min_range, max_range * range_factor)
        velocity = random.uniform(max_velocity * (1-velocity_factor), max_velocity * velocity_factor)
        
        # Generate RCS within specified range
        rcs = random.uniform(min_rcs, max_rcs)
        
        # Generate random 3D position (for ray-tracing)
        azimuth = random.uniform(azimuth_range[0], azimuth_range[1])
        elevation = random.uniform(elevation_range[0], elevation_range[1])
        
        # Convert spherical to Cartesian coordinates
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        z = distance * np.sin(elevation_rad)
        
        # Store target information
        target = {
            'distance': distance,
            'velocity': velocity,
            'rcs': rcs,
            'azimuth': azimuth,
            'elevation': elevation,
            'position': (x, y, z)
        }
        targets.append(target)
    
    return targets


def create_target_mask(targets, num_doppler_bins, num_range_bins, range_resolution, 
                      velocity_resolution, precision='float32', sigma_range=1.0, 
                      sigma_doppler=1.0, threshold=0.1):
    """
    Create ground truth mask for radar targets.
    
    Args:
        targets: List of target dictionaries with 'distance' and 'velocity' keys
        num_doppler_bins: Number of Doppler bins in the range-Doppler map
        num_range_bins: Number of range bins in the range-Doppler map
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        precision: Data type precision ('float32' or 'float16')
        sigma_range: Standard deviation in range dimension
        sigma_doppler: Standard deviation in Doppler dimension
        threshold: Threshold value for creating binary mask
        
    Returns:
        Binary mask with shape [num_doppler_bins, num_range_bins, 1]
    """
    # Initialize target mask
    target_mask = np.zeros((num_doppler_bins, num_range_bins, 1), dtype=precision)
    
    # Create Gaussian-shaped targets in the mask
    for target in targets:
        # Calculate range and Doppler bin
        range_bin = int(target['distance'] / range_resolution)
        doppler_bin = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        
        # Ensure bins are within valid range
        if (0 <= range_bin < num_range_bins and 
            0 <= doppler_bin < num_doppler_bins):
            
            # Define region around target
            r_min = max(0, int(range_bin - 3*sigma_range))
            r_max = min(num_range_bins - 1, int(range_bin + 3*sigma_range))
            d_min = max(0, int(doppler_bin - 3*sigma_doppler))
            d_max = min(num_doppler_bins - 1, int(doppler_bin + 3*sigma_doppler))
            
            # Fill target mask with Gaussian shape
            for r in range(r_min, r_max + 1):
                for d in range(d_min, d_max + 1):
                    # Calculate Gaussian value
                    exponent = -0.5 * ((r - range_bin) / sigma_range)**2 - 0.5 * ((d - doppler_bin) / sigma_doppler)**2
                    value = np.exp(exponent)
                    
                    # Update mask (use maximum value in case of overlapping targets)
                    target_mask[d, r, 0] = max(target_mask[d, r, 0], value)
    
    # Threshold mask to create binary target mask
    target_mask = (target_mask > threshold).astype(precision)
    
    return target_mask

def test_radar_detection_visualization(
    num_targets=3,
    num_range_bins=256,
    num_doppler_bins=128,
    range_resolution=0.5,
    velocity_resolution=0.2,
    save_path=None,
    show_plot=True
):
    """
    Test function to demonstrate radar detection visualization with simulated targets.
    
    Args:
        num_targets: Number of targets to generate
        num_range_bins: Number of range bins in the range-Doppler map
        num_doppler_bins: Number of Doppler bins in the range-Doppler map
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
        
    Returns:
        Tuple of (rd_map, target_mask, targets, detection_results)
    """
    import numpy as np
    from AIRadarLib.visualization import plot_detection_results
    
    # Generate random radar targets
    targets = generate_radar_targets(
        num_targets=num_targets,
        min_range=5,
        max_range=100,
        min_velocity=1,
        max_velocity=20,
        min_rcs=5.0,
        max_rcs=30.0,
        azimuth_range=(-45, 45),
        elevation_range=(-10, 10)
    )
    
    # Create target mask
    target_mask = create_target_mask(
        targets=targets,
        num_doppler_bins=num_doppler_bins,
        num_range_bins=num_range_bins,
        range_resolution=range_resolution,
        velocity_resolution=velocity_resolution
    )
    
    # Generate synthetic range-Doppler map
    rd_map = generate_clear_synthetic_rd_map(
        num_rx=2,
        num_doppler=num_doppler_bins,
        num_range=num_range_bins,
        targets=targets,
        range_resolution=range_resolution,
        velocity_resolution=velocity_resolution,
        noise_floor=-60,
        target_snr=30
    )
    
    # Simulate detection results (normally from CFAR)
    detection_results = []
    for target in targets:
        # Convert target parameters to bin indices
        range_idx = int(target['distance'] / range_resolution)
        doppler_idx = int(num_doppler_bins // 2 + target['velocity'] / velocity_resolution)
        
        # Add some random offset to simulate detection errors
        range_idx += np.random.randint(-2, 3)
        doppler_idx += np.random.randint(-2, 3)
        
        # Ensure indices are within valid range
        range_idx = max(0, min(range_idx, num_range_bins - 1))
        doppler_idx = max(0, min(doppler_idx, num_doppler_bins - 1))
        
        # Create detection result
        detection = {
            'range_idx': range_idx,
            'doppler_idx': doppler_idx,
            'magnitude': float(np.random.uniform(0.7, 1.0)),
            'snr': float(np.random.uniform(15, 25))
        }
        detection_results.append(detection)
    
    # Visualize detection results
    plot_detection_results(
        rd_map=rd_map,
        target_mask=target_mask,
        targets=targets,
        detection_results=detection_results,
        range_resolution=range_resolution,
        velocity_resolution=velocity_resolution,
        num_doppler_bins=num_doppler_bins,
        num_range_bins=num_range_bins,
        save_path=save_path,
        title="Radar Detection Results - Simulated Data",
        show_plot=show_plot
    )
    
    return rd_map, target_mask, targets, detection_results


def generate_clear_synthetic_rd_map(
    num_rx=2,
    num_doppler=128,
    num_range=256,
    targets=None,
    range_resolution=0.5,
    velocity_resolution=0.2,
    noise_floor=-60,
    target_snr=30
):
    """
    Generate a synthetic range-Doppler map with clear target peaks.
    
    Args:
        num_rx: Number of RX antennas
        num_doppler: Number of Doppler bins
        num_range: Number of range bins
        targets: List of target dictionaries with 'distance' and 'velocity' keys
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        noise_floor: Noise floor in dB
        target_snr: Target SNR in dB
        
    Returns:
        Synthetic range-Doppler map with shape [num_rx, 2, num_doppler, num_range]
    """
    import numpy as np
    
    # Initialize range-Doppler map with noise
    noise_power = 10**(noise_floor/10)
    rd_map = np.zeros((num_rx, 2, num_doppler, num_range), dtype=np.float32)
    
    # Add noise to the map
    for rx in range(num_rx):
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), (num_doppler, num_range))
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), (num_doppler, num_range))
        rd_map[rx, 0, :, :] = noise_real
        rd_map[rx, 1, :, :] = noise_imag
    
    # Add targets to the map
    if targets:
        target_power = 10**(target_snr/10) * noise_power
        
        for target in targets:
            # Calculate range and Doppler bin
            range_bin = int(target['distance'] / range_resolution)
            doppler_bin = int(num_doppler // 2 + target['velocity'] / velocity_resolution)
            
            # Ensure bins are within valid range
            if (0 <= range_bin < num_range and 0 <= doppler_bin < num_doppler):
                # Scale target power by RCS
                scaled_power = target_power * (target['rcs'] / 10.0)
                amplitude = np.sqrt(scaled_power)
                
                # Add target to each RX antenna with slightly different phase
                for rx in range(num_rx):
                    # Add random phase for each RX to simulate spatial diversity
                    phase = np.random.uniform(0, 2*np.pi)
                    
                    # Create 2D Gaussian shape around target
                    sigma_range = 1.0
                    sigma_doppler = 1.0
                    
                    # Define region around target
                    r_min = max(0, int(range_bin - 3*sigma_range))
                    r_max = min(num_range - 1, int(range_bin + 3*sigma_range))
                    d_min = max(0, int(doppler_bin - 3*sigma_doppler))
                    d_max = min(num_doppler - 1, int(doppler_bin + 3*sigma_doppler))
                    
                    # Add Gaussian-shaped target
                    for r in range(r_min, r_max + 1):
                        for d in range(d_min, d_max + 1):
                            # Calculate Gaussian value
                            exponent = -0.5 * ((r - range_bin) / sigma_range)**2 - 0.5 * ((d - doppler_bin) / sigma_doppler)**2
                            value = amplitude * np.exp(exponent)
                            
                            # Add to real and imaginary components with phase
                            rd_map[rx, 0, d, r] += value * np.cos(phase)
                            rd_map[rx, 1, d, r] += value * np.sin(phase)
    
    return rd_map


if __name__ == "__main__":
    # Run the test function when the script is executed directly
    test_radar_detection_visualization(
        num_targets=5,
        save_path="data/radarv4/radar_detection_test.png",
        show_plot=True
    )