# AI Radar Processing and Dataset Generation

This repository contains tools for radar signal processing and synthetic dataset generation for deep learning applications in radar sensing.

## Overview

The project consists of two main components:

1. **AIradar_processing.py**: A radar signal processing module that implements various signal processing techniques for different radar signal types.
2. **AIradar_datasetv3.py**: A dataset generation and management module that creates synthetic radar data with configurable parameters.

## Signal Types

The system supports several radar signal types:

### Standard Signal Types

These signal types use standard processing without hardware-specific considerations:

1. **FMCW (Frequency Modulated Continuous Wave)**
   - Uses linear frequency chirps
   - Standard for automotive and industrial radar applications
   - Processing: Range FFT followed by Doppler FFT
   - [View FMCW processing code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L150-L200)

2. **OFDM (Orthogonal Frequency Division Multiplexing)**
   - Uses multiple orthogonal subcarriers
   - Advantages in communication and sensing integration
   - Processing: FFT across subcarriers for range, FFT across symbols for Doppler
   - [View OFDM processing code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L250-L300)

3. **Sine (Continuous Wave)**
   - Uses single-frequency continuous wave
   - Simple Doppler measurements
   - Processing: Phase change analysis across chirps
   - [View Sine processing code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L350-L400)

### Hardware-Based Two-Step Signal Types

These signal types model real hardware systems with a two-step demodulation process:

1. **OFDM_FMCW**
   - Combines OFDM and FMCW techniques
   - Two-step hardware processing:
     - Step 1: CN0566 (10GHz) sweep demodulation
     - Step 2: AD9361 (2.1GHz) baseband processing
   - Requires special filtering to isolate OFDM components
   - [View OFDM_FMCW processing code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L450-L500)
   - [View hardware demodulation steps](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L100-L130)

2. **Sine_FMCW**
   - Combines CW and FMCW techniques
   - Two-step hardware processing similar to OFDM_FMCW
   - Requires bandpass filtering around sine frequency
   - [View Sine_FMCW processing code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L550-L600)
   - [View hardware demodulation steps](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L100-L130)
## Mathematical Background

### FMCW Processing

The FMCW signal can be represented as:

$$s(t) = \exp\left(j2\pi \left(f_c t + \frac{B}{2T}t^2\right)\right)$$

where:
- $f_c$ is the center frequency
- $B$ is the bandwidth
- $T$ is the chirp duration

For a target at range $R$ with velocity $v$, the received signal is:

$$s_r(t) = \alpha \exp\left(j2\pi \left(f_c (t-\tau) + \frac{B}{2T}(t-\tau)^2\right)\right)$$

where $\tau = \frac{2R}{c}$ is the round-trip delay and $\alpha$ is the reflection coefficient.

After mixing with the transmitted signal, the beat signal is:

$$s_b(t) = \alpha \exp\left(j2\pi \left(-f_c\tau - \frac{B}{T}t\tau + \frac{B}{2T}\tau^2\right)\right)$$

The range FFT of this signal produces peaks at frequencies proportional to the target range:

$$f_b = \frac{B}{T}\tau = \frac{2RB}{cT}$$

### Doppler Processing

For moving targets, the Doppler frequency shift is:

$$f_d = \frac{2v}{\lambda} = \frac{2vf_c}{c}$$

This is extracted by performing an FFT across multiple chirps at the same range bin.

### Two-Step Processing

For hardware-based two-step processing:

1. **First demodulation (CN0566)**: 
   $$s_1(t) = s_r(t) \cdot s^*(t) = \alpha \exp\left(j2\pi \left(-f_c\tau - \frac{B}{T}t\tau + \frac{B}{2T}\tau^2\right)\right)$$

2. **Second demodulation (AD9361)**:
   $$s_2(t) = \text{Filter}\{s_1(t)\}$$

## Dataset Generation

The dataset generation process includes:

1. **Target Scenario Generation**:
   - Random placement of targets with configurable parameters
   - Realistic radar cross-section (RCS) modeling

2. **Signal Generation**:
   - Creation of transmitted waveforms based on signal type
   - Simulation of target reflections with appropriate delays and Doppler shifts

3. **Signal Processing**:
   - Range-Doppler processing
   - Target mask generation for deep learning applications

4. **Data Storage**:
   - HDF5 or NumPy formats
   - Comprehensive metadata

## Usage

### Dataset Generation

```bash
python AIradar_datasetv3.py --mode generate --num_samples 100 --signal_type FMCW --save_path data/radar --format hdf5
```

#### Internal Dataset Generation Steps:

1. **Parameter Initialization** - [View code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_datasetv3.py#L50-L100)
   - Sets up radar parameters (bandwidth, sample rate, etc.)
   - Initializes data structures for storing generated samples

2. **Target Generation** - [View code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_datasetv3.py#L200-L250)
   - Creates random target scenarios with configurable parameters
   - Generates target information (distance, velocity, RCS)
   - Function: `_generate_random_targets()`

3. **Signal Generation** - [View code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_datasetv3.py#L300-L350)
   - Generates transmitted waveforms based on signal type
   - Simulates target reflections with appropriate delays
   - Function: `_generate_time_domain_data()`

4. **Range-Doppler Processing** - [View code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_processing.py#L150-L200)
   - Processes time domain data into range-Doppler maps
   - Applies appropriate signal processing based on signal type
   - Function: `time_to_range_doppler()`

5. **Target Mask Generation** - [View code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_datasetv3.py#L400-L450)
   - Creates binary masks indicating target locations
   - Applies Gaussian spreading for more realistic masks
   - Function: `_generate_target_masks()`

6. **Data Storage** - [View code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_datasetv3.py#L500-L550)
   - Saves dataset to HDF5 or NumPy format
   - Stores comprehensive metadata
   - Functions: `_save_hdf5()` and `_save_numpy()`

7. **Visualization** - [View code](https://github.com/kaikailiu/radarsensing/blob/main/AIRadar/AIradar_datasetv3.py#L600-L650)
   - Generates sample visualizations
   - Creates dataset statistics plots
   - Function: `visualize_sample()`
   
### Dataset Visualization
```bash
python AIradar_datasetv3.py --mode visualize --signal_type FMCW --data_path data/radar/radar_data.hdf5
```