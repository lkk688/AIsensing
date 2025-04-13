# Deep Learning-Based AI Processing Framework for Wireless Communication and Radar Sensing

## Introduction

Deep learning has revolutionized various application scenarios by significantly improving performance across domains. In the context of wireless communication, researchers have explored the potential of deep learning techniques to enhance system efficiency and reliability. In this work, we present our novel AI backend processing framework, designed to address critical challenges in wireless communication and radar sensing.

## Existing Solutions and Their Limitations
One notable solution in this field is [NVIDIA SIONNA](https://developer.nvidia.com/sionna), which is open sourced at [sionna](https://github.com/NVlabs/sionna). SIONNA leverages the power of Tensorflow to accelerate AI physical-layer research. However, it has limitations:

1. **Simulation-Only Approach:** SIONNA operates solely on simulation data, lacking a real radio interface. This restricts its applicability to practical scenarios.

2. **Tensorflow Dependency:** SIONNA relies exclusively on the Tensorflow framework, limiting flexibility for researchers who prefer other deep learning libraries. Sionna also does not support for Tensorflow versions `>2.14` due to the stopped support of the `complex` data type in Tensorflow Layers.

3. **Basic Neural Networks:** While effective, SIONNA's neural network architecture remains basic, missing out on advanced transformer models.

## Our Proposed AI Backend Processing Framework

Our new AI processing framework aims to overcome these limitations. It offers the following features:


1. **Hybrid Data Sources: Real Hardware Radio and Simulation Data**
   - Our framework interfaces seamlessly with both real hardware radio systems (support Linux Industry IO and Analog's tranceiver chips) and simulation data (e.g., 5G CDL Channel dataset and DeepMIMO dataset). Researchers can seamlessly interface our framework with physical software-defined radio (SDR) hardware. This dual approach ensures robustness and practical relevance. We also integrate with the DeepMIMO raytracing dataset, enabling comprehensive performance evaluation.

2. **Flexible Libraries: Numpy, Pytorch, and Huggingface Transformers**
   - We leverage Numpy for efficient data preprocessing and simulation data preparation.
   - Pytorch serves as our primary deep learning framework, allowing researchers to build complex neural architectures.
   - Huggingface Transformers enhance our capabilities with advanced transformer models.

3. **Dual Capability: Communication and Radar Sensing**
   - Our framework provides AI processing capabilities for both communication tasks (e.g., OFDM symbol detection, demodulation, channel estimation) and radar sensing (target detection and tracking).
   - By combining these functionalities, we create a unified solution for diverse wireless applications.

4. **Empowering Students via Pythonic Architecture**
   - Our backend processing framework is designed in Python, promoting readability, extensibility, and collaboration.
   - It offers a clear modular distinction between domain-specific components (e.g., OFDM communication, signal processing) and general-purpose deep learning models.
   - Our open environment encourages Computer Science and Software Engineering students to innovate. Students can develop software and deep learning models using a specified general-purpose dataset format, without requiring deep domain-specific knowledge in wireless communication. 
   - Our AI processing framework bridges the gap between theory and practice, empowering researchers and students alike. As we refine our implementation, we anticipate further breakthroughs in wireless communication and radar sensing. By fostering collaboration and creativity, we build upon the solid foundation we've established.

The overall system architecture is shown here:

<img width="1066" alt="image" src="https://github.com/lkk688/AIsensing/assets/6676586/7817a076-66cd-49a3-aeba-c960fde4ef86">


## Detailed Documents for AIsensing

1. [AIprocessing](deeplearning/AIprocessing.md) contains the setup and implementation details of the AI processing framework for Radar and Communication based on Numpy, Pytorch and Transformers.
   - [AIsim_main2.py](deeplearning/AIsim_main2.py) is the created main code to perform complete OFDM transmission over CDL or DeepMIMO channel dataset.
   - [deepMIMO5.py](deeplearning/deepMIMO5.py) contains the major code related to OFDM basic modules and DeepMIMO Channel dataset
   - [ofdmtrain_pytorch2.py](deeplearning/ofdmtrain_pytorch2.py) contains the training code of Pytorch models for OFDM communication simulation
   - [ofdmeval_pytorch.py](deeplearning/ofdmeval_pytorch.py) contains the inference and evaluation code of Pytorch models for OFDM communication simulation
   - [wave2vec_ofdm.py](deeplearning/wave2vec_ofdm.py) contains the Wave2Vec transformer models for OFDM communication

2. [MATLAB](matlab/matlabsim.md) contains the details of the MATLAB Interface to the SDR Device and Communication Simulation.
   - [simpleQAM](matlab/simpleQAM.mlx): test the basic QAM modulation, draw the Constellation Diagram
   - [simpleofdm](matlab/simpleofdm.mlx): simulates basic ofdm connection, test the BER
   - [80211ofdm](matlab/ofdm_communication.mlx): simulate the IEEE802.11 OFDM communication
   - [dfts_ofdm](matlab/dfts_ofdm.mlx): simulate the DFT-S OFDM to minimize PAPR in UL via DFT
   - [DFT-spread-OFDM Radar](matlab/periodogram_radar_dfts_one.mlx): Periodogram-based OFDM Radar with DFT-spread Single Target
   - [DFT-spread-OFDM Radar](matlab/periodogram_radar.mlx): DFT-spread-OFDM Radar simulation (two targets)


3. [SDR Radios](sdradi/sdr_radios.md) contains the details of the interface to SDR Radio Devices.
   - [myad9361.py](sdradi/myad9361.py) Test and run the AD9361 transceiver
   - [myad9361class.py](sdradi/myad9361class.py) Put all ADI transeiver related code into one class
   - [myadiclass.py](sdradi/myadiclass.py) extends the `myad9361class.py`
   - [myofdm.py](sdradi/myofdm.py) OFDM related code in one library (subset of `deepMIMO5.py`), used for radio device
   - [myofdmwithsdr.py](sdradi/myofdmwithsdr.py) Integrated OFDM MIMO transmission with SDR radio

4. [Joint Communication and Radar Hardware Systems](sdradi/sdr.md) contains the implementation details and software framework to the software-defined radio devices for communication and radar sensing.
   - [myradar3.py](sdradi/myradar3.py) Radar device control related code
   - [radar_fmcw3.py](sdradi/radar_fmcw3.py) Implements FMCW Radar
   - [radarappwdevice3.py](sdradi/radarappwdevice3.py) Latest main entrance file for Radar device


## Current work in progress
### Radar Dataset and Training

- [x] radar_dataset.py: create class RadarDataset, generate radar simulation data and use it for training. 
   - [x] Update the RadarDataset class to generate time-domain data suitable for software-defined radio devices (new parameters for SDR configuration), and add functions to convert this time-domain data to the range-doppler domain.
      - Generates FMCW chirp signals
      - Simulates target reflections with appropriate time delays and Doppler shifts
      - Handles multiple RX antennas with spatial diversity
      - Stores I/Q data in the format [batch, num_rx, num_chirps, samples_per_chirp, 2]
      - Range-Doppler processing: Converts time-domain data to range-Doppler maps using FFT; Provides both single-sample and batch processing functions
      - Shows time-domain signals alongside range-Doppler maps for visualization, displays detailed target information, visualizes both I and Q components.
   - [ ] Generate more realistic radar data with moving targets: class RealisticRadarDataset
- [] create a training script for the radar target detection model: radar_train.py; create a script to test the trained radar model: radar_test.py

- [ ] Added class Transmiter() and class NNChannelEstimator based on Pytorch inside the AIsim_main2.py to support multiple transmitters
- [ ] Add class TransformerChannelEstimator in AI_Channel.py and add more comprehensive simulation data-based training and evaluation
- [ ] complete the trainmain function by implementing a neural network model for OFDM signal processing in AIsim_maindataset2.py
- [ ] create a new OFDMNet class in AIsim_maindataset3.py that uses transformer architecture and provide better feature extraction and modeling of complex relationships in OFDM signals. Test the Range_doppler plot function.
- [ ] add a flexible transformer model in AIcomm_radar_models.py that can handle both OFDM communication and radar signal processing: Standard self-attention for OFDM, Range and Doppler attention for radar processing, Learnable positional embeddings, Separate output activations for each mode: 1) Sigmoid for OFDM symbol detection; 2) Tanh for radar target detection. For OFDM communication: Initialize with mode='comm'. For Radar processing: Initialize with mode='radar'

- [ ] AIsionna_radar.py: modify the radar reflection simulation code to use NumPy instead of TensorFlow and create a function that allows users to set up multiple targets with custom parameters.
- [ ] train_multitask.py: a training script for the DualPurposeTransformer model that handles both OFDM communication and radar sensing tasks. The script will include training, evaluation, and visualization components.
- [ ] AImodels_joint.py