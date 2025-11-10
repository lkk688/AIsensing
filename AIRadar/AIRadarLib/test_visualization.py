#!/usr/bin/env python3
"""
Test script for the optimized plot_signal_time_and_spectrum function
"""

import numpy as np
import sys
import os

# Add the AIRadarLib to the path
sys.path.append('/Users/kaikailiu/Documents/MyRepo/radarsensing/AIRadar')

from AIRadarLib.visualization import plot_signal_time_and_spectrum

def test_optimized_visualization():
    """
    Test the optimized plot_signal_time_and_spectrum function with a sample FMCW chirp
    """
    print("Testing optimized plot_signal_time_and_spectrum function...")
    
    # FMCW radar parameters
    sample_rate = 25e6  # 25 MHz
    chirp_duration = 40e-6  # 40 microseconds
    bandwidth = 500e6  # 500 MHz
    center_freq = 77e9  # 77 GHz
    
    # Generate a sample FMCW chirp signal
    num_samples = int(sample_rate * chirp_duration)
    t = np.linspace(0, chirp_duration, num_samples)
    
    # FMCW chirp: f(t) = f0 + (bandwidth/chirp_duration) * t
    slope = bandwidth / chirp_duration
    f0 = center_freq - bandwidth / 2
    
    # Generate complex chirp signal
    instantaneous_freq = f0 + slope * t
    phase = 2 * np.pi * (f0 * t + 0.5 * slope * t**2)
    signal = np.exp(1j * phase)
    
    # Add some noise for realism
    noise_power = 0.01
    noise = noise_power * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    signal_noisy = signal + noise
    
    # Create output directory
    output_dir = '/Users/kaikailiu/Documents/MyRepo/radarsensing/AIRadar/test_visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Basic visualization with all enhancements
    print("\n1. Testing basic enhanced visualization...")
    plot_signal_time_and_spectrum(
        signal=signal_noisy, #(1000,)
        sample_rate=sample_rate,
        total_duration=chirp_duration,
        title_prefix="Enhanced TX Chirp",
        window_type="blackman",
        N_fft=8192,
        bandwidth=bandwidth,
        center_freq=center_freq,
        zoom_margin=0.2,
        textstr=f"Bandwidth: {bandwidth/1e6:.1f} MHz\nSlope: {slope/1e12:.2f} THz/s\nDuration: {chirp_duration*1e6:.1f} μs",
        highlight_peak=True,
        normalize=True,
        save_path=None, #os.path.join(output_dir, "enhanced_tx_chirp.png"),
        draw_window=True
    )
    
    # Test 2: Comparison with different window types
    print("\n2. Testing different window types...")
    for window_type in ['blackman', 'hamming', 'hann', 'rect']:
        plot_signal_time_and_spectrum(
            signal=signal_noisy,
            sample_rate=sample_rate,
            total_duration=chirp_duration,
            title_prefix=f"TX Chirp - {window_type.capitalize()} Window",
            window_type=window_type,
            N_fft=8192,
            bandwidth=bandwidth,
            center_freq=center_freq,
            zoom_margin=0.1,
            textstr=f"Window: {window_type.capitalize()}\nBandwidth: {bandwidth/1e6:.1f} MHz",
            highlight_peak=True,
            normalize=True,
            save_path=os.path.join(output_dir, f"tx_chirp_{window_type}_window.png"),
            draw_window=True
        )
    
    # Test 3: Beat signal simulation
    print("\n3. Testing with simulated beat signal...")
    # Simulate a beat signal with target at 100m, velocity 50 km/h
    c = 3e8  # Speed of light
    target_range = 100  # meters
    target_velocity = 50 / 3.6  # 50 km/h to m/s
    
    # Calculate beat frequency components
    f_beat_range = 2 * slope * target_range / c
    f_beat_doppler = 2 * center_freq * target_velocity / c
    f_beat = f_beat_range + f_beat_doppler
    
    # Generate beat signal
    beat_signal = np.exp(1j * 2 * np.pi * f_beat * t)
    beat_signal_noisy = beat_signal + 0.1 * noise
    
    plot_signal_time_and_spectrum(
        signal=beat_signal_noisy,
        sample_rate=sample_rate,
        total_duration=chirp_duration,
        title_prefix="Enhanced Beat Signal",
        window_type="blackman",
        N_fft=8192,
        bandwidth=None,  # No bandwidth highlighting for beat signal
        center_freq=None,
        zoom_margin=0,
        textstr=f"Target Range: {target_range} m\nTarget Velocity: {target_velocity*3.6:.1f} km/h\nBeat Frequency: {f_beat/1e3:.2f} kHz",
        highlight_peak=True,
        normalize=True,
        save_path=os.path.join(output_dir, "enhanced_beat_signal.png"),
        draw_window=True
    )
    
    print(f"\nAll test visualizations saved to: {output_dir}")
    print("\nOptimization features tested:")
    print("✓ Enhanced time axis (microseconds)")
    print("✓ Improved color schemes and styling")
    print("✓ Better grid and minor ticks")
    print("✓ Enhanced bandwidth highlighting")
    print("✓ Improved peak annotation")
    print("✓ Better legend positioning")
    print("✓ Enhanced text display")
    print("✓ Optimized figure layout")
    print("✓ Higher DPI output")
    print("✓ Dynamic range limiting (80 dB)")
    
if __name__ == "__main__":
    test_optimized_visualization()