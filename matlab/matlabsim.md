# MATLAB Interface to the SDR Device and Communication Simulation
## MATLAB Simulation

### OFDM Communication
One of the problem of OFDM based communication is the high peak-to-average power ratio (PAPR) caused by IFFT. The peak power is proportional to IFFT length (L). We can minimize PAPR in UL via DFT. 

The DFT-S technique involves first passing the set of transmit symbols - got from a Q-ary alphabet like QAM or QPSK through a Discrete Fourier Transform (DFT) block, before they are mapped to the inputs of an Inverse Discrete Fourier Transform (IDFT) block. The DFT and IDFT operations are computed very efficiently using FFT and IFFT algorithms. The size of the IDFT block (L) is chosen to be an integer (K) multiple of the size of the DFT block.

The comparison of OFDM and DFTS-OFDM is shown here:
![DFTS-OFDM compare](imgs/figure-dft-precoded-ofdm.png)

The DFTS-OFDM communication diagram is shown here:
![DFTS-OFDM Diagram](imgs/dftsofdm.png)

Communication Sample code:
  * [simpleQAM](matlab/simpleQAM.mlx): test the basic QAM modulation, draw the Constellation Diagram
  * [simpleofdm](matlab/simpleofdm.mlx): simulates basic ofdm connection, test the BER
  * [80211ofdm](matlab/ofdm_communication.mlx): simulate the IEEE802.11 OFDM communication
  * [dfts_ofdm](matlab/dfts_ofdm.mlx): simulate the DFT-S OFDM to minimize PAPR in UL via DFT

The Constellation Diagram of the communication simulation:
![64qamcommunication](64QAM.png)

### OFDM Radar
OFDM Radar diagram is shown here:
![OFDMRadardiagram](imgs/ofdmradardiagram.png)

OFDM Radar Sample code:
  * [DFT-spread-OFDM Radar](matlab/periodogram_radar_dfts_one.mlx): Periodogram-based OFDM Radar with DFT-spread Single Target
  * [DFT-spread-OFDM Radar](matlab/periodogram_radar.mlx): DFT-spread-OFDM Radar simulation (two targets)

The simulation results of the DFTS-OFDM radar for two targets:
![ofdmradarheatmap](imgs/ofdmradarheatmap.png)