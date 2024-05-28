
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