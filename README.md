# AIsensing for Radar and Communication

[AIprocessing](deeplearning/AIprocessing.md) contains the details of the AI processing framework based on Numpy, Pytorch and Transformers.

[MATLAB](matlab/matlabsim.md) contains the details of the MATLAB Interface to the SDR Device and Communication Simulation.

[SDR Radios](sdradi/sdr_radios.md) contains the details of the interface to SDR Radio Devices.

[Joint Communication and Radar Hardware Systems](sdradi/sdr.md) contains the implementation details and software framework to the software-defined radio devices for communication and radar sensing.

# Deep Learning with DeepMIMO Dataset
Install Pytorch and TensorFlow (some package needs Tensorflow). Following [Tensorflow Pip](https://www.tensorflow.org/install/pip) page to install Tensorflow:
```bash
(mypy310) lkk@Alienware-LKKi7G8:~/Developer/AIsensing$ python3 -m pip install tensorflow[and-cuda]
# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Download [DeepMIMO](https://www.deepmimo.net/) dataset.

Follow [link](https://www.deepmimo.net/versions/v2-python/), install DeepMIMO python package:
```bash
pip install DeepMIMO
```

Select and download a scenario from the scenarios [page](https://www.deepmimo.net/scenarios/), for example, select Outdoor scenario1 (O1). Download 'O1_60' and 'O1_3p5' to the 'data' folder.

Run the DeepMIMO simulation and obtain the BER curve for various configurations:
```bash
python deeplearning/deepMIMO5_sim.py
```
[BER Curve](imgs/berlist.jpg)