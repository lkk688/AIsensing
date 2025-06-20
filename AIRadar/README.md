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

## Simulate the ADF4159 (included in CN0566) for FMCW Modulation
The function generate_adf4159_fmcw_chirp in `AIRadarLib/waveform_utils.py` is a sophisticated simulation of a real-world FMCW (Frequency-Modulated Continuous Wave) radar chirp signal that emulates characteristics of the Analog Devices ADF4159 fractional-N PLL synthesizer. 

To generate a time-domain complex-valued FMCW radar signal with hardware-realistic behavior, accounting for:
	•	PLL quantization
	•	Non-instantaneous frequency ramp transitions
	•	Phase noise
	•	Reference spurs
	•	Frequency deviation (crystal inaccuracy)
	•	Edge windowing
	•	Configurable PLL parameters

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

# Transformer-Based Radar Detection Model

This repository contains a transformer-based deep learning model for radar target detection. The model is designed to process time-domain radar data directly and can be compared with conventional radar processing techniques and other deep learning models.

## Model Architecture

The `RadarTransformerNet` is a novel architecture that combines convolutional neural networks (CNNs) with transformer blocks to process radar data. Key features include:

- **Time-Domain Processing**: Directly processes raw time-domain radar data from multiple receivers
- **Learnable FFT**: Optional learnable Fourier transform for range and Doppler processing
- **CNN Backbone**: Initial feature extraction using 3D convolutional layers
- **Transformer Blocks**: Capture long-range dependencies in both range and Doppler dimensions
- **Specialized Attention Mechanisms**: Dedicated attention modules for range and Doppler dimensions

## Files

- `AIradar_transformer.py`: Contains the transformer model architecture
    - RadarTransformerNet : A transformer-based model that processes time-domain radar data directly
    - Specialized attention mechanisms for range and Doppler dimensions
    - Optional learnable FFT for range-Doppler processing
    - CNN backbone for initial feature extraction
- `AIradar_transformer_train.py`: Script for training and evaluating the transformer model
    - Creates and trains the transformer model
    - Evaluates performance on validation and test data
    - Visualizes results and compares with conventional CFAR detection
    - Saves model checkpoints and performance metrics
- `AIradar_model_comparison.py`: Script for comparing different radar detection models
    - Evaluates multiple radar detection models on the same test data
    - Compares the transformer model with RadarNet, RadarTimeToFreqNet, and conventional CFAR
    - Generates ROC curves, precision-recall curves, and performance metrics
    - Visualizes detection results for qualitative comparison

## Usage

### Training the Transformer Model

```bash
python AIradar_transformer_train.py \
    --train_samples 1000 \
    --val_samples 200 \
    --test_samples 100 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 0.001 \
    --model_dim 128 \
    --model_depth 4 \
    --model_heads 8 \
    --use_learnable_fft \
    --use_cnn_backbone \
    --realistic_effects \
    --output_dir ./transformer_results
```

### Comparing Models

```bash
python AIradar_model_comparison.py \
    --test_samples 100 \
    --transformer_model_path ./transformer_results/best_transformer_model.pth \
    --radarnet_model_path ./model_results/best_radarnet_model.pth \
    --time_to_freq_model_path ./model_results/best_time_to_freq_model.pth \
    --use_learnable_fft \
    --use_cnn_backbone \
    --output_dir ./comparison_results
```

## Model Parameters

### RadarTransformerNet

- `num_rx`: Number of receiver antennas (default: 4)
- `num_chirps`: Number of chirps per frame (default: 128)
- `samples_per_chirp`: Number of samples per chirp (default: 1000)
- `out_doppler_bins`: Output Doppler bins (default: 128)
- `out_range_bins`: Output range bins (default: 256)
- `dim`: Model dimension (default: 128)
- `depth`: Number of transformer blocks (default: 4)
- `heads`: Number of attention heads (default: 8)
- `mlp_dim`: MLP hidden dimension (default: 256)
- `dropout`: Dropout rate (default: 0.1)
- `use_learnable_fft`: Whether to use learnable FFT (default: True)
- `use_cnn_backbone`: Whether to use CNN backbone (default: True)

## Training Parameters

- `train_samples`: Number of training samples (default: 1000)
- `val_samples`: Number of validation samples (default: 200)
- `test_samples`: Number of test samples (default: 100)
- `batch_size`: Batch size (default: 8)
- `epochs`: Number of epochs (default: 50)
- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay (default: 1e-5)

## Dataset Parameters

- `sample_rate`: Sample rate in Hz (default: 500e6)
- `transceiver_bandwidth`: Transceiver bandwidth in Hz (default: 30e6)
- `num_chirps`: Number of chirps per frame (default: 128)
- `bandwidth`: Bandwidth in Hz (default: 200e6)
- `center_freq`: Center frequency in Hz (default: 10e9)
- `num_rx`: Number of receiver antennas (default: 4)
- `max_targets`: Maximum number of targets (default: 3)
- `snr_min`: Minimum SNR in dB (default: 5.0)
- `snr_max`: Maximum SNR in dB (default: 20.0)
- `realistic_effects`: Add realistic effects to data (default: False)
- `signal_type`: Signal type (default: 'FMCW')

## Results

The training script will save the following results in the specified output directory:

- Best model checkpoint (`best_transformer_model.pth`)
- Training curves (`transformer_training_curves.png`)
- Test sample visualizations
- Test metrics (`test_results.txt`)

The comparison script will save:

- Comparison metrics (`comparison_metrics.json`)
- ROC curves (`roc_curves.png`)
- Precision-Recall curves (`pr_curves.png`)
- Sample visualizations

## Advantages of the Transformer Model

1. **End-to-End Learning**: Processes raw time-domain data directly, eliminating the need for manual feature engineering
2. **Attention Mechanisms**: Captures long-range dependencies in both range and Doppler dimensions
3. **Learnable FFT**: Can learn optimal transformations from time to frequency domain
4. **Specialized Processing**: Dedicated attention modules for range and Doppler dimensions
5. **Robust to Noise**: Can learn to filter out noise and interference

# New Versions of the Model
### RadarTransformerNet
The transformer model processes time-domain data directly and includes:

- 3D convolutional layers for initial feature extraction
- Optional CNN backbone
- Learnable FFT for range and Doppler dimensions
- Transformer blocks with specialized attention mechanisms
- Output projection for detection
### RadarNet
This model processes range-Doppler maps and includes:

- Encoder-decoder architecture with skip connections
- Multiple output heads for detection, velocity estimation, and SNR estimation
- Target extraction functionality
### RadarTimeToFreqNet
This hybrid model includes:

- Time-domain preprocessing with 3D convolutions
- Conversion from time to frequency domain
- RadarNet for detection from the generated range-Doppler maps