import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

def generate_waveform(
    bandwidth: float,  # MHz
    chirp_duration: float,  # μs
    center_freq: float,  # GHz
    sample_rate: float,  # MHz
    waveform_type: str  # 'linear', 'sawtooth', or 'triangular'
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Generate FMCW radar waveform and calculate derived parameters
    
    Args:
        bandwidth: Bandwidth in MHz
        chirp_duration: Chirp duration in μs
        center_freq: Center frequency in GHz
        sample_rate: Sample rate in MHz
        waveform_type: Type of waveform ('linear', 'sawtooth', or 'triangular')
        
    Returns:
        Tuple containing time domain plot data, frequency domain plot data, and derived parameters
    """
    # Convert to base units
    bandwidth_hz = bandwidth * 1e6
    chirp_duration_sec = chirp_duration * 1e-6
    center_freq_hz = center_freq * 1e9
    sample_rate_hz = sample_rate * 1e6
    
    # Calculate derived parameters
    c = 3e8  # Speed of light (m/s)
    slope = bandwidth_hz / chirp_duration_sec  # Hz/s
    
    # Range resolution
    range_resolution = c / (2 * bandwidth_hz)
    
    # Maximum unambiguous range
    max_range = (c * sample_rate_hz) / (2 * bandwidth_hz * bandwidth_hz)
    
    # Velocity resolution (for a frame with 100 chirps)
    velocity_resolution = (c * 1000) / (4 * center_freq_hz * chirp_duration_sec)
    
    # Maximum unambiguous velocity (assuming 100 chirps per frame)
    max_velocity = (c * 1000) / (4 * center_freq_hz * (chirp_duration_sec / 100))
    
    # Wavelength
    wavelength = (c / center_freq_hz) * 1000  # mm
    
    # Max beat frequency (assuming 100m max range)
    max_range_m = 100
    max_beat_freq = (2 * slope * max_range_m) / c
    
    # Generate time domain data
    num_samples = 1000
    t = np.linspace(0, chirp_duration_sec, num_samples)
    
    # Start and end frequencies
    start_freq = center_freq_hz - bandwidth_hz / 2
    end_freq = center_freq_hz + bandwidth_hz / 2
    
    # Generate signal based on waveform type
    if waveform_type == 'linear':
        # Linear chirp
        phase = 2 * np.pi * (start_freq * t + (slope * t * t) / 2)
        signal = np.cos(phase)
        
        # Instantaneous frequency
        #inst_freq = start_freq + slope * t
        # Calculate instantaneous frequency from phase derivative
        # Unwrap phase to handle 2π jumps
        unwrapped_phase = np.unwrap(phase)
        # Calculate derivative of phase with respect to time
        inst_freq = np.gradient(unwrapped_phase, t) / (2 * np.pi)
        
    elif waveform_type == 'sawtooth':
        # Sawtooth chirp
        num_saws = 3
        saw_duration = chirp_duration_sec / num_saws
        signal = np.zeros_like(t)
        inst_freq = np.zeros_like(t)
        phase_array = np.zeros_like(t)
        
        for i in range(num_samples):
            saw_idx = int(t[i] / saw_duration)
            if saw_idx >= num_saws:
                saw_idx = num_saws - 1
                
            rel_t = t[i] - saw_idx * saw_duration
            norm_t = rel_t / saw_duration
            
            # Instantaneous frequency for this sawtooth
            inst_freq[i] = start_freq + bandwidth_hz * norm_t
            
            # Phase calculation for this sawtooth segment
            # phase = 2 * np.pi * (start_freq * rel_t + (slope * norm_t * norm_t * saw_duration) / 2)
            # signal[i] = np.cos(phase)
            # Phase calculation for this sawtooth segment
            phase = 2 * np.pi * (start_freq * rel_t + (slope * norm_t * norm_t * saw_duration) / 2)
            phase_array[i] = phase
            signal[i] = np.cos(phase)
        
        # Calculate instantaneous frequency from phase derivative
        # Unwrap phase to handle 2π jumps and sawtooth discontinuities
        unwrapped_phase = np.unwrap(phase_array)
        # Calculate derivative of phase with respect to time
        inst_freq = np.gradient(unwrapped_phase, t) / (2 * np.pi)
            
    elif waveform_type == 'triangular':
        # Triangular chirp
        half_duration = chirp_duration_sec / 2
        signal = np.zeros_like(t)
        inst_freq = np.zeros_like(t)
        phase_array = np.zeros_like(t)
        
        for i in range(num_samples):
            if t[i] <= half_duration:
                # Up chirp
                norm_t = t[i] / half_duration
                inst_freq[i] = start_freq + bandwidth_hz * norm_t
                phase = 2 * np.pi * (start_freq * t[i] + (slope * t[i] * t[i]) / 2)
            else:
                # Down chirp
                rel_t = t[i] - half_duration
                norm_t = 1 - (rel_t / half_duration)
                inst_freq[i] = start_freq + bandwidth_hz * norm_t
                
                down_chirp_start_freq = end_freq
                down_chirp_slope = -slope
                phase = 2 * np.pi * (down_chirp_start_freq * rel_t + (down_chirp_slope * rel_t * rel_t) / 2)
                
            phase_array[i] = phase
            signal[i] = np.cos(phase)
            #signal[i] = np.cos(phase)
        
        # Calculate instantaneous frequency from phase derivative
        # Unwrap phase to handle 2π jumps and triangular discontinuities
        unwrapped_phase = np.unwrap(phase_array)
        # Calculate derivative of phase with respect to time
        inst_freq = np.gradient(unwrapped_phase, t) / (2 * np.pi)
    else:
        raise ValueError("Invalid waveform type")
    
    # Generate frequency domain data
    # For simplicity, we'll use a wider frequency range than the bandwidth
    display_bandwidth = bandwidth_hz * 1.5
    display_start_freq = center_freq_hz - display_bandwidth / 2
    display_end_freq = center_freq_hz + display_bandwidth / 2
    
    freq_range = np.linspace(display_start_freq, display_end_freq, 500)
    spectrum = np.zeros_like(freq_range, dtype=float)
    
    # Simulate spectrum based on waveform type
    for i, freq in enumerate(freq_range):
        # Check if frequency is within the chirp bandwidth
        normalized_freq = (freq - start_freq) / bandwidth_hz
        
        if waveform_type == 'linear':
            # Linear chirp has relatively flat spectrum within bandwidth
            if start_freq <= freq <= end_freq:
                # Main lobe
                magnitude = 0.9
                
                # Add some rolloff at the edges
                if normalized_freq < 0.05 or normalized_freq > 0.95:
                    magnitude *= 0.8
            else:
                # Side lobes
                distance_from_band = min(
                    abs(freq - start_freq),
                    abs(freq - end_freq)
                ) / bandwidth_hz
                
                magnitude = max(0.1, 0.3 * np.exp(-5 * distance_from_band))
                
        elif waveform_type == 'sawtooth':
            # Sawtooth has harmonics and more side lobes
            if start_freq <= freq <= end_freq:
                # Main lobe
                magnitude = 0.85
                
                # Add harmonics
                harmonic_spacing = bandwidth_hz / 3
                if abs((freq - start_freq) % harmonic_spacing) < bandwidth_hz / 200:
                    magnitude = 0.95
            else:
                # Side lobes with harmonics
                distance_from_band = min(
                    abs(freq - start_freq),
                    abs(freq - end_freq)
                ) / bandwidth_hz
                
                magnitude = max(0.15, 0.4 * np.exp(-3 * distance_from_band))
                
                # Add harmonics outside the band
                harmonic_spacing = bandwidth_hz / 3
                dist_from_harmonic = min(
                    abs(((freq - start_freq) % harmonic_spacing) / harmonic_spacing),
                    abs(1 - ((freq - start_freq) % harmonic_spacing) / harmonic_spacing)
                )
                
                if dist_from_harmonic < 0.05:
                    magnitude += 0.2
                    
        elif waveform_type == 'triangular':
            # Triangular has smoother spectrum with less side lobes
            if start_freq <= freq <= end_freq:
                # Main lobe with smooth shape
                magnitude = 0.9 * (1 - 0.3 * np.power(2 * normalized_freq - 1, 2))
            else:
                # Smoother side lobes
                distance_from_band = min(
                    abs(freq - start_freq),
                    abs(freq - end_freq)
                ) / bandwidth_hz
                
                magnitude = max(0.05, 0.25 * np.exp(-4 * distance_from_band))
        
        spectrum[i] = magnitude
    
    # Create time domain plot
    time_domain_fig = go.Figure()
    time_domain_fig.add_trace(go.Scatter(
        x=t * 1e6,  # Convert to μs
        y=signal,
        mode='lines',
        name='Signal'
    ))
    
    time_domain_fig.add_trace(go.Scatter(
        x=t * 1e6,  # Convert to μs
        y=inst_freq / 1e9,  # Convert to GHz
        mode='lines',
        name='Instantaneous Frequency',
        yaxis='y2'
    ))
    
    time_domain_fig.update_layout(
        title='FMCW Radar Waveform',
        xaxis_title='Time (μs)',
        yaxis_title='Amplitude',
        yaxis2=dict(
            title='Frequency (GHz)',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor='Black',
            borderwidth=1
        )
    )
    
    # Create frequency domain plot
    freq_domain_fig = go.Figure()
    freq_domain_fig.add_trace(go.Scatter(
        x=(freq_range - center_freq_hz) / 1e6,  # Centered around 0, in MHz
        y=spectrum,
        mode='lines',
        name='Spectrum'
    ))
    
    freq_domain_fig.update_layout(
        title='Frequency Domain Representation',
        xaxis_title='Frequency Offset from Center (MHz)',
        yaxis_title='Magnitude',
        xaxis=dict(
            range=[
                (display_start_freq - center_freq_hz) / 1e6,
                (display_end_freq - center_freq_hz) / 1e6
            ]
        ),
        yaxis=dict(
            range=[0, 1]
        )
    )
    
    # Add vertical lines for bandwidth
    freq_domain_fig.add_shape(
        type="line",
        x0=-bandwidth / 2,
        y0=0,
        x1=-bandwidth / 2,
        y1=1,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        )
    )
    
    freq_domain_fig.add_shape(
        type="line",
        x0=bandwidth / 2,
        y0=0,
        x1=bandwidth / 2,
        y1=1,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        )
    )
    
    # Add annotation for bandwidth
    freq_domain_fig.add_annotation(
        x=0,
        y=0.95,
        text=f"Bandwidth: {bandwidth} MHz",
        showarrow=False,
        font=dict(
            size=12,
            color="red"
        )
    )
    
    # Prepare derived parameters for display
    derived_params = {
        "rangeResolution": {
            "value": range_resolution,
            "unit": "m",
            "description": "Minimum distance between distinguishable targets"
        },
        "maxRange": {
            "value": max_range,
            "unit": "m",
            "description": "Maximum detectable range"
        },
        "velocityResolution": {
            "value": velocity_resolution,
            "unit": "m/s",
            "description": "Minimum velocity difference between distinguishable targets"
        },
        "maxVelocity": {
            "value": max_velocity,
            "unit": "m/s",
            "description": "Maximum detectable velocity"
        },
        "wavelength": {
            "value": wavelength,
            "unit": "mm",
            "description": "Wavelength of the radar signal"
        },
        "chirpSlope": {
            "value": slope / 1e12,
            "unit": "THz/s",
            "description": "Rate of frequency change during chirp"
        },
        "maxBeatFrequency": {
            "value": max_beat_freq / 1e6,
            "unit": "MHz",
            "description": "Maximum beat frequency for 100m range"
        },
        "nyquistFrequency": {
            "value": sample_rate_hz / 2e6,
            "unit": "MHz",
            "description": "Maximum frequency that can be sampled without aliasing"
        }
    }
    
    # Convert Plotly figures to JSON
    time_domain_plot = time_domain_fig.to_dict()
    freq_domain_plot = freq_domain_fig.to_dict()
    
    return time_domain_plot, freq_domain_plot, derived_params

#returning Plotly figure objects converted to dictionaries
# Plotly figure structure (simplified)
# {
#   'data': [
#     {
#       'x': [...],  # Time values
#       'y': [...],  # Signal values
#       'mode': 'lines',
#       'name': 'Signal'
#     },
#     {
#       'x': [...],  # Time values again
#       'y': [...],  # Frequency values
#       'mode': 'lines',
#       'name': 'Instantaneous Frequency',
#       'yaxis': 'y2'
#     }
#   ],
#   'layout': {...}
# }