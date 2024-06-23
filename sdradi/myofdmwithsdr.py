#Combination of myad9361class.py and myofdm.py

from myofdm import OFDMSymbol, OFDMAMIMO
from myad9361class import SDR

# https://github.com/rikluost/sionna-with-PlutoSDR/tree/main/test
# https://github.com/rikluost/sionna-PlutoSDR

# https://github.com/Repo4Sub/NSDI2024

def test_ofdm_SDR(urladdress, SampleRate, fc=921.1e6, leadingzeros=500, add_td_samples = 16):
    myofdm = OFDMSymbol()
    SAMPLES = myofdm.createOFDMsignal() #(80,) complex128
    #SampleRate = rg.fft_size*rg.subcarrier_spacing # sample 

    bandwidth = SampleRate *1.1
    mysdr = SDR(SDR_IP=urladdress, SDR_FC=fc, SDR_SAMPLERATE=SampleRate, SDR_BANDWIDTH=bandwidth)

    # SINR, SDR_TX_GAIN, SDR_RX_GAIN, Attempts, Pearson R
    x_sdr = mysdr.SDR_RXTX_offset(SAMPLES, leadingzeros=leadingzeros, add_td_samples=add_td_samples)
    #out, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails+1, corr, sdr_time
    rx_samples = x_sdr[0]

def test_ofdmmimo_SDR(urladdress, fc=921.1e6, leadingzeros=500, add_td_samples = 16):
    myofdm = OFDMAMIMO(num_rx = 1, num_tx = 1, \
                batch_size =1, fft_size = 128, num_ofdm_symbols=14, num_bits_per_symbol = 4,  \
                USE_LDPC = False, pilot_pattern = "kronecker", guards=True, showfig=False) #pilot_pattern= "kronecker" "empty"
    #channeltype="perfect", "awgn", "ofdm", "time"
    SAMPLES, x_rg = myofdm.transmit(b=None) #samples: (1, 1, 1, 1876)
    #output: complex Time-domain OFDM signal [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols*(fft_size+cyclic_prefix_length)]
    #SampleRate = rg.fft_size*rg.subcarrier_spacing # sample 
    SampleRate = myofdm.RESOURCE_GRID.fft_size * myofdm.RESOURCE_GRID.subcarrier_spacing #1920000

    bandwidth = SampleRate *1.1
    mysdr = SDR(SDR_IP=urladdress, SDR_FC=fc, SDR_SAMPLERATE=SampleRate, SDR_BANDWIDTH=bandwidth)

    # SINR, SDR_TX_GAIN, SDR_RX_GAIN, Attempts, Pearson R
    x_sdr = mysdr.SDR_RXTX_offset(SAMPLES.flatten(), leadingzeros=leadingzeros, add_td_samples=add_td_samples, tx_gain=-10, rx_gain=10)
    #out, SINR, SDR_TX_GAIN, SDR_RX_GAIN, fails+1, corr, sdr_time
    rx_samples = x_sdr[0]


# piuri="ip:phaser.local:50901"
# localuri="ip:analog.local"
# antsdruri="ip:192.168.1.10"#connected via Ethernet with static IP
# plutodruri="ip:192.168.2.1" "ip:192.168.2.16"#connected via USB
#PoE device: ip:192.168.1.67:50901
import argparse
parser = argparse.ArgumentParser(description='MyAD9361')
parser.add_argument('--urladdress', default="ip:192.168.2.1", type=str,
                    help='urladdress of the device, e.g., ip:pluto.local') 
parser.add_argument('--rxch', default=1, type=int, 
                    help='number of rx channels')
parser.add_argument('--signal', default="dds", type=str,
                    help='signal type: sinusoid, dds')
parser.add_argument('--plot', default=False, type=bool,
                    help='plot figure')

def main():
    args = parser.parse_args()
    urladdress = args.urladdress #"ip:pluto.local"
    Rx_CHANNEL = args.rxch
    signal_type = args.signal
    plot_flag = args.plot

    #testlibiioaccess(urladdress)
    #sdr_test(urladdress, signal_type=signal_type, Rx_CHANNEL=Rx_CHANNEL, plot_flag = plot_flag)

    #test_SDRclass(urladdress)
    fs=1000000
    test_ofdm_SDR(urladdress=urladdress, SampleRate=fs)
    test_ofdmmimo_SDR(urladdress=urladdress, leadingzeros=500)

if __name__ == '__main__':
    main()