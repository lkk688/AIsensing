
# Deep Learning-Based AI Processing Framework for Wireless Communication and Radar Sensing

## Introduction

Deep learning has revolutionized various application scenarios by significantly improving performance across domains. In the context of wireless communication, researchers have explored the potential of deep learning techniques to enhance system efficiency and reliability. In this work, we present our novel AI backend processing framework, designed to address critical challenges in wireless communication and radar sensing.

## Existing Solutions and Their Limitations

### NVIDIA SIONNA

One notable solution in this field is [NVIDIA SIONNA](https://developer.nvidia.com/sionna), which is open sourced at [sionna](https://github.com/NVlabs/sionna). SIONNA leverages the power of Tensorflow to accelerate AI physical-layer research. However, it has limitations:

1. **Simulation-Only Approach:** SIONNA operates solely on simulation data, lacking a real radio interface. This restricts its applicability to practical scenarios.

2. **Tensorflow Dependency:** SIONNA relies exclusively on the Tensorflow framework, limiting flexibility for researchers who prefer other deep learning libraries.

3. **Basic Neural Networks:** While effective, SIONNA's neural network architecture remains basic, missing out on advanced transformer models.

## Our Proposed AI Backend Processing Framework

### Key Features

Our new AI processing framework aims to overcome these limitations. It offers the following features:

1. **Hybrid Data Sources: Real Hardware Radio and Simulation Data**
   - Our framework interfaces seamlessly with both real hardware radio systems and simulation data. This dual approach ensures robustness and practical relevance.

2. **Flexible Libraries: Numpy, Pytorch, and Huggingface Transformers**
   - We leverage Numpy for efficient data preprocessing and simulation data preparation.
   - Pytorch serves as our primary deep learning framework, allowing researchers to build complex neural architectures.
   - Huggingface Transformers enhance our capabilities with advanced transformer models.

3. **Dual Capability: Communication and Radar Sensing**
   - Our framework provides AI processing capabilities for both communication tasks (e.g., OFDM symbol detection, demodulation, channel estimation) and radar sensing (target detection and tracking).
   - By combining these functionalities, we create a unified solution for diverse wireless applications.

### Implementation Details

1. **Pythonic Architecture**
   - Our backend processing framework is designed in Python, promoting readability, extensibility, and collaboration.
   - It offers a clear modular distinction between domain-specific components (e.g., OFDM communication, signal processing) and general-purpose deep learning models.

2. **Integration with Physical Hardware and DeepMIMO Dataset**
   - Researchers can seamlessly interface our framework with physical software-defined radio (SDR) hardware.
   - Additionally, we integrate with the DeepMIMO raytracing dataset, enabling comprehensive performance evaluation.

3. **Empowering Students**
   - Our open environment encourages Computer Science and Software Engineering students to innovate. Students can develop software and deep learning models using a specified general-purpose dataset format, without requiring deep domain-specific knowledge in wireless communication. 
   - Our AI processing framework bridges the gap between theory and practice, empowering researchers and students alike. As we refine our implementation, we anticipate further breakthroughs in wireless communication and radar sensing. By fostering collaboration and creativity, we build upon the solid foundation we've established.

## DeepMIMO
[DeepMIMO](https://deepmimo.net/) is a generic dataset that enables a wide range of machine/deep learning applications for MIMO systems. It takes as input a set of parameters (such as antenna array configurations and time-domain/OFDM parameters) and generates MIMO channel realizations, corresponding locations, angles of arrival/departure, etc., based on these parameters and on a ray-tracing scenario selected from those available in DeepMIMO.

DeepMIMO provides multiple scenarios that one can select from. We use the O1 scenario with the carrier frequency set to 60 GHz (O1_60). We need to download the "O1_60" data files from this [page](https://deepmimo.net/scenarios/o1-scenario/). The downloaded zip file should be extracted into a folder, and the parameter DeepMIMO_params['dataset_folder'] should be set to point to this folder. To use DeepMIMO with Sionna, the DeepMIMO dataset first needs to be generated. In our `deepMIMO5.py` file, we need to setup the `dataset_folder='data' #Windows part: r'D:\Dataset\CommunicationDataset\O1_60'` in the main file, and it will use the following function to get the DeepMIMO dataset:
```bash
DeepMIMO_dataset = get_deepMIMOdata(scenario=scenario, dataset_folder=dataset_folder, showfig=showfig)
```
The generated DeepMIMO dataset contains channels for different locations of the users and basestations. In our example, the users located on the rows `user_row_first` to `user_row_first`. Each of these rows consists of 181 user locations, resulting in `181*100=18100` basestation-user channels. The antenna arrays in the DeepMIMO dataset are defined through the x-y-z axes. In the following example, a single-user MISO downlink is considered. The basestation is equipped with a uniform linear array of 16 elements spread along the x-axis. The users are each equipped with a single antenna.
```bash
# Number of basestations
print(len(DeepMIMO_dataset)) #1
# Keys of a basestation dictionary
print(DeepMIMO_dataset[0].keys()) #['user', 'basestation', 'location']
# Keys of a channel
print(DeepMIMO_dataset[0]['user'].keys()) #['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
# Shape of the channel matrix
print(DeepMIMO_dataset[active_bs_idx]['user']['channel'].shape) #(num_ue_locations=18100, 1, bs_antenna=16, strongest_path=10) 
# The channel matrix between basestation i=0 and user j=0, Shape of BS 0 - UE 0 channel
print(DeepMIMO_dataset[active_bs_idx]['user']['channel'][j].shape) #(1, 16, 10)
```

Ray-tracing Path Parameters are saved in dictionary, number of path is 9, and each key is a size of 9 array.
```bash
# Path properties of BS 0 - UE 0
print(DeepMIMO_dataset[active_bs_idx]['user']['paths'][j]) #Ray-tracing Path Parameters in dictionary
#'num_paths': 9, Azimuth and zenith angle-of-arrivals – degrees (DoA_phi, DoA_theta), size of 9 array
# Azimuth and zenith angle-of-departure – degrees (DoD_phi, DoD_theta)
# Time of arrival – seconds (ToA)
# Phase – degrees (phase)
# Power – watts (power)
# Number of paths (num_paths)
print(DeepMIMO_dataset[active_bs_idx]['user']['LoS'][j]) #Integer of values {-1, 0, 1} indicates the existence of the LOS path in the channel.
# (1): The LoS path exists.
# (0): Only NLoS paths exist. The LoS path is blocked (LoS blockage).
# (-1): No paths exist between the transmitter and the receiver (Full blockage).

print(DeepMIMO_dataset[active_bs_idx]['user']['distance'][j])
#The Euclidian distance between the RX and TX locations in meters.

print(DeepMIMO_dataset[active_bs_idx]['user']['pathloss'][j])
#The combined path-loss of the channel between the RX and TX in dB.
```

[UserStation Location](../imgs/ddeepmimo_userstationlocation.png)
[Channel Response](../imgs/deepmimo_channelresponse.png)
[UE BS positions with path loss](../imgs/deepmimo-uebspositions.png)
[UEGrid Path Loss](../imgs/deepmimo_uegridpathloss.png)