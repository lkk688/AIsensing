import numpy as np
try:
    import plotly.graph_objects as go
except ImportError:
    # If plotly is not installed, install it using pip
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go
from typing import Dict, Any, Tuple
from AIRadar.AIRadarLib.datautil import calculate_radar_parameters
from AIRadar.AIRadarLib.waveform_utils import (
    generate_linear_chirp,
    generate_sawtooth_chirp,
    generate_triangular_chirp,
    generate_spectrum
)

def generate_waveform(
    bandwidth=500,  # MHz
    chirp_duration=500,  # μs
    center_freq=10,  # GHz
    sample_rate=1,  # MHz
    waveform_type='linear'
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Generate radar waveform and calculate radar parameters.
    
    Args:
        bandwidth: Bandwidth in MHz
        chirp_duration: Chirp duration in μs
        center_freq: Center frequency in GHz
        sample_rate: Sample rate in MHz
        waveform_type: Type of waveform ('linear', 'sawtooth', 'triangular')
        
    Returns:
        Tuple containing:
        - time_domain_plot: Dictionary with time domain plot data
        - freq_domain_plot: Dictionary with frequency domain plot data
        - derived_params: Dictionary with derived radar parameters
    """
    # Convert to base units
    bandwidth_hz = bandwidth * 1e6
    chirp_duration_sec = chirp_duration * 1e-6
    center_freq_hz = center_freq * 1e9
    sample_rate_hz = sample_rate * 1e6
    
    # Calculate radar parameters using the utility function
    radar_params = calculate_radar_parameters(
        sample_rate=sample_rate_hz,
        chirp_duration=chirp_duration_sec,
        center_freq=center_freq_hz,
        bandwidth=bandwidth_hz,
        num_chirps=100  # Assuming 100 chirps for velocity calculations
    )
    
    # Extract parameters from the result
    range_resolution = radar_params["range_resolution"]
    max_range = radar_params["max_range"]
    velocity_resolution = radar_params["velocity_resolution"]
    max_velocity = radar_params["max_unambiguous_velocity"]
    wavelength = radar_params["wavelength"]
    slope = radar_params["fmcw_slope"]
    
    # Generate time domain data
    num_samples = 1000
    t = np.linspace(0, chirp_duration_sec, num_samples)
    
    # Start and end frequencies
    start_freq = center_freq_hz - bandwidth_hz / 2
    end_freq = center_freq_hz + bandwidth_hz / 2
    
    # Generate signal based on waveform type
    if waveform_type == 'linear':
        signal, inst_freq, phase = generate_linear_chirp(t, start_freq, slope)
    elif waveform_type == 'sawtooth':
        signal, inst_freq, phase = generate_sawtooth_chirp(t, start_freq, bandwidth_hz, chirp_duration_sec)
    elif waveform_type == 'triangular':
        signal, inst_freq, phase = generate_triangular_chirp(t, start_freq, end_freq, chirp_duration_sec)
    else:
        raise ValueError("Invalid waveform type")
    
    # Generate frequency domain data
    # For simplicity, we'll use a wider frequency range than the bandwidth
    display_bandwidth = bandwidth_hz * 1.5
    display_start_freq = center_freq_hz - display_bandwidth / 2
    display_end_freq = center_freq_hz + display_bandwidth / 2
    
    freq_range = np.linspace(display_start_freq, display_end_freq, 500)
    spectrum = generate_spectrum(freq_range, start_freq, end_freq, bandwidth_hz, waveform_type)
    
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
    
    # Calculate max beat frequency for 100m range
    max_beat_freq = 2 * 100 * slope / 3e8  # 2*R*slope/c
    
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
            "value": wavelength * 1000,  # Convert to mm
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