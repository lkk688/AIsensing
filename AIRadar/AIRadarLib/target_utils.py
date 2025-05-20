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