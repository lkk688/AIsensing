# AIRadar: Advanced Radar Sensing and Simulation Framework

AIRadar is a comprehensive framework for radar signal processing, simulation, and dataset generation. This project provides tools for generating synthetic radar data with realistic effects, processing radar signals, and training machine learning models for radar applications.

## Overview

The AIRadar framework implements various radar signal processing techniques, with a focus on Frequency-Modulated Continuous Wave (FMCW) radar systems. It provides capabilities for:

- Generating synthetic radar datasets with configurable parameters
- Simulating realistic radar signal propagation using ray-tracing
- Processing radar signals to extract range and velocity information
- Visualizing radar data through various plots and heatmaps
- Training machine learning models for radar applications

## Technical Details

### FMCW Radar Principles

FMCW (Frequency-Modulated Continuous Wave) radar operates by transmitting a chirp signal - a signal whose frequency changes linearly with time. The mathematical representation of a chirp signal is:

$$s_{tx}(t) = \exp\left(j2\pi \left(f_c t + \frac{1}{2}\alpha t^2\right)\right)$$

Where:
- $f_c$ is the carrier frequency
- $\alpha$ is the chirp rate or slope (bandwidth/chirp duration)
- $t$ is time

When this signal reflects off a target at distance $R$ moving with velocity $v$, the received signal is:

$$s_{rx}(t) = \exp\left(j2\pi \left(f_c (t-\tau) + \frac{1}{2}\alpha (t-\tau)^2\right)\right)$$

Where $\tau = \frac{2(R + vt)}{c}$ is the round-trip delay time, with $c$ being the speed of light.

The beat signal (mixer output) is obtained by multiplying the transmitted and received signals:

$$s_{beat}(t) = s_{tx}(t) \cdot s_{rx}^*(t) = \exp\left(j2\pi \left(f_c\tau + \alpha t \tau - \frac{1}{2}\alpha \tau^2\right)\right)$$

For small $\tau$ compared to $t$, this simplifies to:

$$s_{beat}(t) \approx \exp\left(j2\pi \left(f_c\tau + \alpha t \tau\right)\right)$$

The beat frequency $f_{beat} = \alpha \tau = \frac{2R\alpha}{c}$ is proportional to the target range, while the phase change across chirps is related to the target velocity.

### Range and Velocity Resolution

The range resolution of an FMCW radar is determined by the bandwidth $B$ of the chirp:

$$\Delta R = \frac{c}{2B}$$

The maximum unambiguous range is determined by the sampling rate $f_s$:

$$R_{max} = \frac{f_s c}{2\alpha}$$

The velocity resolution depends on the wavelength $\lambda$ and the total observation time $T_{obs} = N_{chirps} \cdot T_{chirp}$:

$$\Delta v = \frac{\lambda}{2 T_{obs}} = \frac{\lambda}{2 N_{chirps} T_{chirp}}$$

The maximum unambiguous velocity (Nyquist limit) is:

$$v_{max} = \frac{\lambda}{4 T_{chirp}}$$

### Signal Processing Pipeline

The radar signal processing pipeline in AIRadar consists of the following steps:

1. **Chirp Generation**: Generate FMCW chirp signals with specified parameters
2. **Target Simulation**: Simulate target reflections with realistic delay, Doppler shift, and attenuation
3. **Signal Mixing**: Mix transmitted and received signals to obtain the beat signal
4. **Range Processing**: Apply FFT along the fast-time dimension to obtain range profiles
5. **Doppler Processing**: Apply FFT along the slow-time dimension to obtain velocity information
6. **Detection**: Apply CFAR (Constant False Alarm Rate) detection to identify targets
7. **Parameter Estimation**: Estimate target parameters (range, velocity, RCS)

### Ray-Tracing Simulation

The framework implements a ray-tracing approach to simulate realistic radar signal propagation. This includes:

- Accurate time delays based on target distance
- Doppler shifts based on target velocity
- Phase shifts due to antenna array geometry
- Signal attenuation based on radar cross-section and distance
- Realistic RF chain effects (optional)

The ray-tracing simulation is implemented in the `_ray_tracing_simulation` method, which calculates the received signal for each target based on physical principles.

### Range-Doppler Processing

Range-Doppler processing is a key technique in radar signal processing that involves:

1. **Range FFT**: Applying FFT along the fast-time dimension to obtain range information
   $$X[m,k] = \sum_{n=0}^{N-1} x[m,n] e^{-j2\pi nk/N}$$

2. **Doppler FFT**: Applying FFT along the slow-time dimension to obtain velocity information
   $$Y[l,k] = \sum_{m=0}^{M-1} X[m,k] e^{-j2\pi ml/M}$$

Where:
- $x[m,n]$ is the beat signal for the $m$-th chirp and $n$-th sample
- $X[m,k]$ is the range profile for the $m$-th chirp
- $Y[l,k]$ is the range-Doppler map
- $N$ is the number of samples per chirp
- $M$ is the number of chirps

## Usage

### Basic Usage

```python
from AIRadar.AIradar_datasetv4 import RadarDataset

# Create a radar dataset with default parameters
radar_dataset = RadarDataset(
    num_samples=100,
    num_range_bins=128,
    num_doppler_bins=16,
    sample_rate=500e6,
    bandwidth=200e6,
    center_freq=10e9,
    num_chirps=512,
    max_targets=3,
    snr_min=10,
    snr_max=25,
    signal_type='FMCW',
    drawfig=True
)

# Generate the dataset
radar_dataset.generate_dataset(visualize=True)
```

### Customizing Radar Parameters

You can customize various radar parameters to match your specific requirements:

```python
# Create a radar dataset with custom parameters
radar_dataset = RadarDataset(
    num_samples=1000,
    num_range_bins=256,
    num_doppler_bins=64,
    sample_rate=1e9,
    bandwidth=500e6,
    center_freq=77e9,  # Automotive radar frequency
    num_chirps=256,
    max_targets=5,
    snr_min=5,
    snr_max=30,
    signal_type='FMCW',
    apply_realistic_effects=True,
    save_path='data/custom_radar_dataset',
    precision='float32'
)
```

## Key Parameters

The `RadarDataset` class accepts the following key parameters:

- `num_samples`: Number of radar scenes to generate
- `num_range_bins`: Number of range bins (determines range resolution granularity)
- `num_doppler_bins`: Number of Doppler bins (determines velocity resolution granularity)
- `sample_rate`: ADC sample rate in Hz
- `bandwidth`: Signal bandwidth in Hz (determines range resolution)
- `center_freq`: Carrier frequency in Hz
- `num_chirps`: Number of chirps per frame (determines velocity resolution)
- `chirp_duration`: Duration of each chirp in seconds
- `max_targets`: Maximum number of targets per scene
- `snr_min/snr_max`: Signal-to-noise ratio range in dB
- `signal_type`: Type of radar signal ('FMCW', 'OFDM', 'Sine', etc.)
- `apply_realistic_effects`: Whether to apply realistic RF effects

## Advanced Features

### Realistic RF Chain Simulation

The framework can simulate realistic RF chain effects including:

- Upconversion and downconversion
- Analog-to-digital conversion
- Phase noise
- I/Q imbalance
- Frequency-dependent attenuation

To enable RF chain simulation:

```python
radar_dataset.generate_dataset(simulate_rf_chain=True, analog_sample_rate=5e9)
```

### Multi-Antenna Support

The framework supports multiple transmit and receive antennas, enabling MIMO radar simulations:

```python
radar_dataset = RadarDataset(
    num_rx=4,  # 4 receive antennas
    num_tx=2,  # 2 transmit antennas
    # Other parameters...
)
```

## Dependencies

- NumPy
- SciPy
- Matplotlib
- tqdm
- PyTorch (optional, for dataset loading)

## References

1. Richards, M. A. (2014). Fundamentals of Radar Signal Processing. McGraw-Hill Education.
2. Skolnik, M. I. (2008). Radar Handbook. McGraw-Hill Education.
3. Winkler, V. (2007). Range Doppler Detection for Automotive FMCW Radars. European Radar Conference.
4. Patole, S. M., et al. (2017). Automotive Radars: A Review of Signal Processing Techniques. IEEE Signal Processing Magazine.

