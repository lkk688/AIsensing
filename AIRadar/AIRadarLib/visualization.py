import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def plot_detection_results(
    rd_map,
    target_mask,
    targets,
    detection_results,
    range_resolution,
    velocity_resolution,
    num_doppler_bins,
    num_range_bins,
    save_path=None,
    title="Radar Detection Results",
    show_plot=True,
    figsize=(12, 10),
    dpi=100
):
    """
    Plot radar detection results, target mask, and ground truth target locations in a single figure.
    
    Args:
        rd_map: Range-Doppler map with shape [num_rx, 2, num_doppler_bins, num_range_bins]
        target_mask: Target mask with shape [num_doppler_bins, num_range_bins, 1]
        targets: List of target dictionaries with 'distance' and 'velocity' keys
        detection_results: List of detection dictionaries with 'range_idx', 'doppler_idx', etc.
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        num_doppler_bins: Number of Doppler bins
        num_range_bins: Number of range bins
        save_path: Path to save the figure (if None, figure is not saved)
        title: Title of the figure
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) in inches
        dpi: DPI for the figure
        
    Returns:
        Figure and axes objects
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Extract magnitude of range-Doppler map (use first RX antenna)
    rd_magnitude = np.sqrt(rd_map[0, 0, :, :]**2 + rd_map[0, 1, :, :]**2)
    
    # Normalize RD map for better visualization
    rd_magnitude_norm = 20 * np.log10(rd_magnitude / np.max(rd_magnitude) + 1e-10)
    rd_magnitude_norm = np.clip(rd_magnitude_norm, -40, 0)  # Clip to dynamic range
    
    # Create range and Doppler axes
    range_axis = np.arange(num_range_bins) * range_resolution
    doppler_axis = (np.arange(num_doppler_bins) - num_doppler_bins // 2) * velocity_resolution
    
    # Plot range-Doppler map
    im = ax.imshow(
        rd_magnitude_norm,
        aspect='auto',
        cmap='jet',
        origin='lower',
        extent=[0, range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        interpolation='none',
        vmin=-40,
        vmax=0
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    # Create a custom colormap for the target mask (transparent to red)
    colors = [(1, 0, 0, 0), (1, 0, 0, 0.7)]  # Red with varying alpha
    target_cmap = LinearSegmentedColormap.from_list('target_mask', colors)
    
    # Plot target mask as overlay with transparency
    if target_mask is not None:
        # Reshape if needed and transpose for correct orientation
        #Binary mask with shape [num_doppler_bins, num_range_bins, 1]
        mask_plot = target_mask.reshape(num_doppler_bins, num_range_bins)
        ax.imshow(
            mask_plot,
            aspect='auto',
            cmap=target_cmap,
            origin='lower',
            extent=[0, range_axis[-1], doppler_axis[0], doppler_axis[-1]],
            interpolation='none'
        )
    
    # Plot ground truth target locations
    if targets:
        target_ranges = [target['distance'] for target in targets]
        target_velocities = [target['velocity'] for target in targets]
        ax.scatter(
            target_ranges,
            target_velocities,
            c='lime',
            marker='o',
            s=100,
            edgecolors='black',
            linewidths=1.5,
            label='Ground Truth'
        )
    
    # Plot CFAR detection results
    if detection_results:
        detection_ranges = []
        detection_velocities = []
        
        for detection in detection_results:
            # Check if detection has range_idx and doppler_idx or range and velocity
            if 'range_idx' in detection and 'doppler_idx' in detection:
                range_val = detection['range_idx'] * range_resolution
                doppler_val = (detection['doppler_idx'] - num_doppler_bins // 2) * velocity_resolution
            elif 'range' in detection and 'velocity' in detection:
                range_val = detection['range']
                doppler_val = detection['velocity']
            else:
                continue
                
            detection_ranges.append(range_val)
            detection_velocities.append(doppler_val)
        
        if detection_ranges:
            ax.scatter(
                detection_ranges,
                detection_velocities,
                c='white',
                marker='x',
                s=80,
                linewidths=2,
                label='CFAR Detections'
            )
    
    # Set labels and title
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    # Set axis limits
    ax.set_xlim(0, range_axis[-1])
    ax.set_ylim(doppler_axis[0], doppler_axis[-1])
    
    # Add text with detection statistics
    if targets and detection_results:
        num_targets = len(targets)
        num_detections = len(detection_results)
        
        stats_text = f"Targets: {num_targets}\nDetections: {num_detections}"
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax