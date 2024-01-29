import torch
import torch.nn.functional as tFunc
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

save_plots = False

# OFDM Parameters
S = 14  # Number of symbols
Sp = 2  # Pilot symbol, 0 for none
F = 72  # Number of subcarriers, including DC
Fp = 2  # Pilot subcarrier spacing
FFT_size = 128  # FFT size

CP = 20  # Cyclic prefix
SCS = 15000  # Subcarrier spacing
P = F // Fp  # Number of pilot subcarriers
FFT_offset = int((FFT_size - F) / 2)  # FFT offset=28
SampleRate = FFT_size * SCS  # Sample rate
Ts = 1 / (SCS * FFT_size)  # Sample duration
TTI_duration = Ts * (FFT_size + CP) * S * 1000  # TTI duration in ms
Pilot_Power = 1  # Pilot power
PDSCH_power = 1  # PDSCH power

def visualize_constellation(pts, Qm):
    plt.scatter(pts.numpy().real, pts.numpy().imag)
    plt.title(f'Modulation Order {Qm} Constellation')
    plt.ylabel('Imaginary'); plt.xlabel('Real')
    plt.tight_layout(); 
    if save_plots:
        plt.savefig(f'./const.png')

#Modulation mapping tables translate binary data sequences into complex symbols for transmission
#Qm = 6  # bits per symbol
#return results: mapping: 6-bit data to complex, demapping: complex to 6-bit data
#e.g, mapping_table_Qm[(0, 0, 0, 0, 0, 0)] => tensor(-1.0801-1.0801j)
def mapping_table(Qm, plot=False):
    # Size of the constellation
    size = int(torch.sqrt(torch.tensor(2**Qm))) #6^2=36
    # The constellation
    a = torch.arange(size, dtype=torch.float32)
    # Shift the constellation to the center
    b = a - torch.mean(a)
    # Use broadcasting to create the complex constellation grid
    C = (b.unsqueeze(1) + 1j * b).flatten()
    # Normalize the constellation
    C /= torch.sqrt(torch.mean(torch.abs(C)**2))
    # Function to convert index to binary
    def index_to_binary(i, Qm):
        return tuple(map(int, '{:0{}b}'.format(int(i), Qm)))
    
    # Create the mapping dictionary
    mapping = {index_to_binary(i, Qm): val for i, val in enumerate(C)}
    
    # Create the demapping table
    demapping = {v: k for k, v in mapping.items()}

    if plot:
        visualize_constellation(C, Qm)    
    return mapping, demapping


#A TTI mask represents the allocation of the complex symbols over a specific Transmission Time Interval (TTI). 
#2D array: rows represent the time slots, columns: available symbols in frequency domain. 
#S: number of symbols = 14
#Sp: pilot symbol spacing
#F: number of subcarriers = 72
#Fp: pilot subcarrier spacing (step)
#The meaning of the mask:
# 0: The PRB is null power.
# 1: The PRB is PDSCH (Physical Downlink Shared Channel).
# 2: Pilot symbols
# 3: (yellow bar) DC
#output: first 28 all zero, last 28 all zero, mid F=72 all 1, Sp=2 (2,1,3), yellow bar =3
def TTI_mask(S, F, Fp, Sp, FFT_offset, plotTTI=False):

    # Create a mask with all ones
    TTI_mask = torch.ones((S, F), dtype=torch.int8) # all ones (14, 72)

    # Set symbol Sp for pilots
    TTI_mask[Sp, torch.arange(0, F, Fp)] = 2 # for pilots TX1, #0-70, 2step
    # Ensure the first subcarrier is a pilot
    TTI_mask[Sp, 0] = 2 #2 is Pilot symbols

    # Ensure the last subcarrier is a pilot
    TTI_mask[Sp, F - 1] = 2 #2 is Pilot symbols

    # DC
    TTI_mask[:, F // 2] = 3 # DC to non-allocable power (oscillator phase noise): 3 in the middle

    TTI_1 = torch.zeros(S, FFT_offset, dtype=torch.int8) #empty space (14, 28) add before and after the TTI_mask
    # Add FFT offsets
    TTI_mask = torch.cat((TTI_1, TTI_mask, TTI_1), dim=1) #cat in the dim=1, 28+72+28=128

    # Plotting the TTI mask
    if plotTTI:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(TTI_mask.numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.title('TTI mask')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('TTImask.png')
        plt.tight_layout()
        plt.show()

    return TTI_mask #(14, 128)

#Creating and Inserting Pilot Symbols in OFDM TTI_mask
def pilot_set(TTI_mask, power_scaling=1.0):

    # Define QPSK pilot values, size of 4
    pilot_values = torch.tensor([-0.7 - 0.7j, -0.7 + 0.7j, 0.7 - 0.7j, 0.7 + 0.7j]) * power_scaling

    # Count the number of pilot elements in the TTI mask, Pilot symbols is 2
    num_pilots = TTI_mask[TTI_mask == 2].numel() #36

    # Create a list of pilot values repeated to match the number of pilots
    pilots = pilot_values.repeat(num_pilots//4+1)[:num_pilots] #4 number repeat 10 times=40, select 36 of them

    return pilots #torch.Size([36])

#TTI_mask: [14, 128]
#Qm = 6  # bits per symbol
#mapping_table_Qm  64 len dict
#return bits: [958, 6], symbol: [958] complex, each symbol is 6 bits
def create_PDSCH_data(TTI_mask, Qm, mapping_table, power=1.):
    # Count PDSCH elements =1
    pdsch_elems = TTI_mask.eq(1).sum() #958 is the effective data slots
    
    # Generate random bits
    bits = torch.randint(0, 2, (pdsch_elems, Qm), dtype=torch.float32) #0 or 1 random bits for (958,6)

    # Flatten and reshape bits for symbol lookup
    flattened_bits = bits.reshape(-1, Qm)
   
    # Convert bits to tuples and lookup symbols
    symbs_list = [mapping_table[tuple(row.tolist())] for row in flattened_bits]
    symbs = torch.tensor(symbs_list, dtype=torch.complex64)

    # Reshape symbols back to original shape
    symbs = symbs.view(pdsch_elems, -1) #958
    symbs = symbs.flatten()
    
    # Apply power scaling 
    symbs *= power
    
    return bits, symbs

#TTI_mask: [14, 128]
#pilot_set/pilot_symbols: 36 complex symbols
#pdsch_symbols 958 symbols
def RE_mapping(TTI_mask, pilot_set, pdsch_symbols, plotTTI=False): 

    # Create a zero tensor for the overall F (72) subcarriers * S symbols (S = 14 Number of symbols)
    TTI = torch.zeros(TTI_mask.shape, dtype=torch.complex64) #14,128

    # Allocate the payload and pilot
    TTI[TTI_mask==1] = pdsch_symbols.clone().detach()
    TTI[TTI_mask==2] = pilot_set.clone().detach()
    # Plotting the TTI modulated symbols
    if plotTTI:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(torch.abs(TTI).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.title('TTI modulated symbols')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('TTImod.png')
        plt.show()

    return TTI #[14, 128]

#flip the CP(20) data symbols and add in front of the 128 symbols
#OFDM_data/TD_TTI_IQ: (14, 128)
#S = 14 Number of symbols
#FFT_size = 128
#CP:20
def CP_addition(OFDM_data, S, FFT_size, CP):

    # Initialize output tensor
    out = torch.zeros((S, FFT_size + CP), dtype=torch.complex64) #(14, 148)

    # Add cyclic prefix to each symbol
    #flip the CP(20) symbols and add in front of the 128 symbols
    for symbol in range(S): #14
        out[symbol, :] = torch.cat((OFDM_data[symbol, -CP:], OFDM_data[symbol, :]), dim=0)

    return out.flatten()


def FFT(TTI):
    return torch.fft.ifft(torch.fft.ifftshift(TTI, dim=1))

#leading_zeros (int): Number of symbols with zero value used for SINR measurement
def create_OFDM_data(TTI_mask_RE, Qm, mapping_table_Qm, pilot_symbols, leading_zeros=80, use_sdr=False, plot=False):
    pdsch_bits, pdsch_symbols = create_PDSCH_data(TTI_mask_RE, Qm, mapping_table_Qm, power=PDSCH_power) # create PDSCH data and modulate it
    #pdsch_bits: [958, 6], pdsch_symbols:[958]
    Modulated_TTI = RE_mapping(TTI_mask_RE, pilot_symbols, pdsch_symbols, plotTTI=plot) # map the PDSCH and pilot symbols to the TTI
    #Modulated_TTI: [14, 128]
    TD_TTI_IQ = FFT(Modulated_TTI) # perform the FFT
    #TD_TTI_IQ: [14, 128]
    TX_Samples = CP_addition(TD_TTI_IQ, S, FFT_size, CP) # add the CP
    #TX_Samples: 14*(128+20)=2072
    if use_sdr:
        zeros = torch.zeros(leading_zeros, dtype=TX_Samples.dtype) # create leading zeros for estimating noise floor power
        TX_Samples = torch.cat((zeros, TX_Samples), dim=0) # add leading zeros to TX samples
    return pdsch_bits, TX_Samples #14*148=2072


#signal/TX_Samples 2072 (14*148) complex
def apply_multipath_channel(signal, n_taps, max_delay, repeats=0, random_start=True, SINR_s=30, leading_zeros=500):
    # note that the output is max_delay longer than input, due to the delayed symbols of some of the taps

    h = torch.zeros(max_delay+1, dtype=torch.complex64) #7

    for _ in range(n_taps): #2
        delay = torch.randint(1, max_delay, (1,)).item()  # Avoid delay=0 for remaining taps
        magnitude = torch.abs(torch.randn(1)) * 0.5
        phase = torch.randn(1)
        complex_gain = magnitude * torch.exp(1j * phase)
        h[delay] = complex_gain

    h[0] = torch.complex(torch.randn(1), torch.randn(1))

    # Normalize the channel response
    h = h / torch.norm(h)

    # Convolve the signal with the channel
    #https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html
    convolved_signal = torch.nn.functional.conv1d(signal.unsqueeze(0).unsqueeze(0), 
                                               h.unsqueeze(0).unsqueeze(0), 
                                               padding=max_delay).squeeze()
    
    # add leading zeros
    zeros = torch.zeros(leading_zeros, dtype=convolved_signal.dtype)
    convolved_signal = torch.cat((zeros, convolved_signal), dim=0)

    # Add noise based on SINR
    if SINR_s !=0:
        signal_power = torch.mean(torch.abs(convolved_signal)**2)
        noise_power = signal_power / (10 ** (SINR_s / 10))
        noise = torch.sqrt(noise_power / 2) * (torch.randn_like(convolved_signal) + 1j * torch.randn_like(convolved_signal))
        convolved_signal = convolved_signal + noise

    # Add random start if required
    if random_start:
        start_index = torch.randint(0, len(convolved_signal), (1,)).item()
        convolved_signal = torch.roll(convolved_signal, shifts=start_index)    
    
    # Repeat the signal if required
    if repeats > 0:
        convolved_signal = convolved_signal.repeat(repeats)

    return convolved_signal

def radio_channel(tx_signal, ch_SINR, n_taps, max_delay, leading_zeros, tx_gain=0, rx_gain = 0, use_sdr=False):
    if use_sdr:
        print("TODO: user SDR")
    else:
        rx_signal = apply_multipath_channel(tx_signal, n_taps=n_taps, max_delay=max_delay, random_start=True, repeats=3, SINR_s=ch_SINR, leading_zeros=leading_zeros)
    return rx_signal

def PSD_plot(signal, Fs, f, info='TX'):
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    plt.figure(figsize=(8, 3))
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    plt.title(f'Power Spectral Density, {info}')
    if save_plots:
        plt.savefig(f'PSD_{info}.png')
    plt.show()

#Synchronization is achieved through correlation. The beginning of a Transmission Time Interval (TTI) us selected based on the first value exceeding an adjustable threshold. 
#Cyclic prefix covers ISI for CP symbols from that point onwards.
def sync_TTI(tx_signal, rx_signal, leading_zeros, threshold=6, plot=False):

    # time sync using correlation

    tx_len = tx_signal.numel()
    rx_len = rx_signal.numel()
    end_point=rx_len-tx_len

    rx_signal = rx_signal[leading_zeros:end_point]

    # Calculate the cross-correlation using conv1d
    corr_result_real = tFunc.conv1d(rx_signal.real.view(1, 1, -1), tx_signal.real.view(1, 1, -1)).view(-1)
    corr_result_imag = tFunc.conv1d(rx_signal.imag.view(1, 1, -1), tx_signal.imag.view(1, 1, -1)).view(-1)
    correlation =  torch.complex(corr_result_real, corr_result_imag)
    correlation = correlation.abs()
    threshold = correlation.mean()*threshold

    # Find the first peak that exceeds the threshold
    for i, value in enumerate(correlation):
        if value > threshold:
            if plot:
                plt.figure(figsize=(8, 4))
                plt.plot(correlation.abs()[i-10:i+50]) #only select -10~50 space
                plt.grid()
                plt.xlabel("Samples from selected index")
                plt.ylabel("Complex conjugate correlation")
                plt.axvline(x=10, color = 'r', linewidth=3)
                if save_plots:
                    plt.savefig('corr.png')
                plt.show()

            return i + leading_zeros
        
    i = torch.argmax(correlation).item() + leading_zeros

    return i

#rx_signal/RX_Samples 6474
#TTI_start/symbol_index 1149
#S=14 Number of symbols
def CP_removal(rx_signal, TTI_start, S, FFT_size, CP, plotsig=False):

    # Initialize a payload mask
    b_payload = torch.zeros(len(rx_signal), dtype=torch.bool)

    # Mark the payload parts of the signal
    for s in range(S):
        start_idx = TTI_start + (s + 1) * CP + s * FFT_size
        end_idx = start_idx + FFT_size
        b_payload[start_idx:end_idx] = 1

    # Plotting the received signal and payload mask
    if plotsig:
        # Assuming rx_signal is a PyTorch tensor
        # Convert to NumPy array and ensure it's in a float format to avoid overflow
        rx_signal_numpy = rx_signal.cpu().numpy()  # Use .cpu() if rx_signal is on GPU
        rx_signal_normalized = rx_signal_numpy / np.max(np.abs(rx_signal_numpy))

        plt.figure(0, figsize=(8, 4))
        plt.plot(rx_signal_normalized)
        plt.plot(b_payload, label='Payload Mask')
        plt.xlabel('Sample index')
        plt.ylabel('Amplitude')
        plt.title('Received signal and payload mask')
        plt.legend()
        if save_plots:
            plt.savefig('RXsignal_sync.png')
        plt.show()

    # Remove the cyclic prefix
    rx_signal_no_CP = rx_signal[b_payload]

    return rx_signal_no_CP.view(S, FFT_size)


#Null symbols were added into the beginning of each transmission to allow measuring the noise level at the receiver. 
def SINR(rx_signal, index, leading_zeros):
    # Calculate noise power
    rx_noise_0 = rx_signal[index - leading_zeros + 20 : index - 20]
    rx_noise_power_0 = torch.mean(torch.abs(rx_noise_0) ** 2)
    if rx_noise_power_0==0:
        rx_noise_power_0 = 0.00001

    # Calculate signal power
    rx_signal_0 = rx_signal[index:index + (14 * 72)]
    rx_signal_power_0 = torch.mean(torch.abs(rx_signal_0) ** 2)

    # Compute SINR
    SINR = 10 * torch.log10(rx_signal_power_0 / rx_noise_power_0)

    # Round the SINR value
    SINR = round(SINR.item(), 1)

    return SINR, rx_noise_power_0, rx_signal_power_0

#Fast Fourier Transform (FFT) transforms the received time-domain signal back into the frequency domain, 
#where data on individual subcarriers can be independently demodulated.
def DFT(rxsignal, plotDFT=False):
    # Calculate DFT
    OFDM_RX_DFT = torch.fft.fftshift(torch.fft.fft(rxsignal, dim=1), dim=1)

    # Plot the DFT if required
    if plotDFT:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(torch.abs(OFDM_RX_DFT).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('TTI_RX.png')
        plt.show()

    return OFDM_RX_DFT

def torch_interp(x, xp, fp):
    # Ensure xp and fp are sorted
    sorted_indices = torch.argsort(xp)
    xp = xp[sorted_indices].to(device=x.device)
    fp = fp[sorted_indices].to(device=x.device)

    # Find the indices to the left and right of x
    indices_left = torch.searchsorted(xp, x, right=True)
    indices_left = torch.clamp(indices_left, 0, len(xp) - 1)

    indices_right = torch.clamp(indices_left - 1, 0, len(xp) - 1)

    # Perform linear interpolation
    x_left, x_right = xp[indices_left], xp[indices_right]
    f_left, f_right = fp[indices_left], fp[indices_right]
    interp_values = f_left + (f_right - f_left) * (x - x_left) / (x_right - x_left)

    return interp_values

def channelEstimate_LS(TTI_mask_RE, pilot_symbols, F, FFT_offset, Sp, OFDM_demod, plotEst=False):
    # Pilot extraction
    pilots = OFDM_demod[TTI_mask_RE == 2]

    # Divide the pilots by the set pilot values
    H_estim_at_pilots = pilots / pilot_symbols

    # Interpolation indices
    pilot_indices = torch.nonzero(TTI_mask_RE[Sp] == 2, as_tuple=False).squeeze()

    # Interpolation for magnitude and phase
    all_indices = torch.arange(FFT_offset, FFT_offset + F)
    
    # Linear interpolation for magnitude and phase
    H_estim_abs = torch_interp(all_indices, pilot_indices, torch.abs(H_estim_at_pilots))
    H_estim_phase = torch_interp(all_indices, pilot_indices, torch.angle(H_estim_at_pilots))

    # Convert magnitude and phase to complex numbers
    H_estim_real = H_estim_abs * torch.exp(1j * H_estim_phase).real
    H_estim_imag = H_estim_abs * torch.exp(1j * H_estim_phase).imag

    # Combine real and imaginary parts to form complex numbers
    H_estim = torch.view_as_complex(torch.stack([H_estim_real, H_estim_imag], dim=-1))


    def dB(x):
        return 10 * torch.log10(torch.abs(x))

    if plotEst:
        plt.figure(figsize=(8, 4))
        plt.plot(pilot_indices.cpu().numpy(), dB(H_estim_at_pilots).cpu().numpy(), 'ro-', label='Pilot estimates', markersize=8)
        plt.plot(all_indices.cpu().numpy(), dB(torch.tensor(H_estim)).cpu().numpy(), 'b-', label='Estimated channel', linewidth=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel('Subcarrier Index', fontsize=12)
        plt.ylabel('Magnitude (dB)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.title('Pilot Estimates and Interpolated Channel in dB', fontsize=14)
        if save_plots:
            plt.savefig('ChannelEstimate.png')
        plt.show()

    return H_estim

def remove_fft_Offests(RX_NO_CP, F, FFT_offset):

    # Calculate indices for the remaining subcarriers after removing offsets
    remaining_indices = torch.arange(FFT_offset, F + FFT_offset)

    # Remove the FFT offsets using slicing
    OFDM_demod = RX_NO_CP[:, remaining_indices]

    return OFDM_demod

#Equalization aims to mitigate the phase and amplitude variations introduced by the communication channel, ensuring accurate data recovery.
def equalize_ZF(OFDM_demod, H_estim, F, S):

    # Reshape the OFDM data and perform equalization
    equalized = (OFDM_demod.view(S, F) / H_estim.to(device=OFDM_demod.device))
    return equalized

#The payload in an OFDM system refers to the actual data transmitted, excluding overheads like cyclic prefixes, pilot symbols, and any additional signaling or control information.
def get_payload_symbols(TTI_mask_RE, equalized, FFT_offset, F, plotQAM=False):
    # Extract payload symbols
    mask = TTI_mask_RE[:, FFT_offset:FFT_offset + F] == 1
    out = equalized[mask]

    # Plotting the QAM symbols
    if plotQAM:
        plt.figure(figsize=(8, 8))
        plt.scatter(out.cpu().real, out.cpu().imag, label='QAM Symbols')
        plt.axis('equal')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.xlabel('Real Part', fontsize=12)
        plt.ylabel('Imaginary Part', fontsize=12)
        plt.title('Received QAM Constellation Diagram', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        if save_plots:
            plt.savefig('RXdSymbols.png')
        plt.show()
    return out

def Demapping(QAM, de_mapping_table):
    # Convert the demapping table keys (constellation points) to a tensor
    constellation = torch.tensor(list(de_mapping_table.keys())).to(QAM.device)
    dists = torch.abs(QAM.view(-1, 1) - constellation.view(1, -1))
    const_index = torch.argmin(dists, dim=1).to(QAM.device)
    hardDecision = constellation[const_index].to(QAM.device)
    string_key_table = {str(key.item()): value for key, value in de_mapping_table.items()}
    demapped_symbols = torch.tensor([string_key_table[str(c.item())] for c in hardDecision], dtype=torch.int32)
    return demapped_symbols, hardDecision

def PS(bits): # parallel to serial
    return bits.view(-1)

def calculate_BER(pdsch_bits, bits_est):
    error_count = torch.sum(bits_est != pdsch_bits.flatten()).float()  # Count of unequal bits
    error_rate = error_count / bits_est.numel()  # Error rate calculation
    BER = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
    #SINR_m = SINR(RX_Samples, symbol_index, leading_zeros=leading_zeros)[0] # calculate the SINR
    return BER #, SINR_m

    
def vis_errorbits(TTI_mask_RE, pdsch_bits, bits_est):
    # Create a zero tensor for the overall F subcarriers * S symbols
    TTI_t = np.zeros(TTI_mask_RE.shape)

    # Allocate the payload and pilot
    errors = (bits_est != pdsch_bits.flatten()).reshape(-1, 6).sum(axis=1)

    TTI_t[TTI_mask_RE==1] = errors
    plt.imshow(TTI_t)

def receiver_process(TX_Samples, RX_Samples, TTI_mask_RE, pilot_symbols, de_mapping_table_Qm, leading_zeros, use_sdr=False, plot=False):
    #Receiver process
    symbol_index=sync_TTI(TX_Samples, RX_Samples, leading_zeros=leading_zeros, threshold= 6, plot=plot)
    #symbol_index: 1149
    
    SINR_m, noise_power, signal_power = SINR(RX_Samples, symbol_index, leading_zeros) # calculate the SINR

    RX_NO_CP = CP_removal(RX_Samples, symbol_index, S, FFT_size, CP, plotsig=plot) # remove the cyclic prefix
    RX_NO_CP = RX_NO_CP / torch.max(torch.abs(RX_NO_CP)) # normalize the signal
    #torch.Size([14, 128])

    #Convert the the time domain OFDM signal into frequency domain
    OFDM_demod = DFT(RX_NO_CP, plotDFT=plot) # perform the DFT on the received signal and plot the result
    #[14, 128]

    H_estim = channelEstimate_LS(TTI_mask_RE, pilot_symbols, F, FFT_offset, Sp, OFDM_demod, plotEst=plot) # estimate the channel using least squares and plot

    OFDM_demod_no_offsets = remove_fft_Offests(OFDM_demod, F, FFT_offset) # remove the FFT offsets and DC carrier from the received signal
    #[14, 72]

    equalized_H_estim = equalize_ZF(OFDM_demod_no_offsets, H_estim, F, S) # equalize the channel using ZF
    #[14, 72]

    #Payload Symbols extraction
    QAM_est = get_payload_symbols(TTI_mask_RE, equalized_H_estim, FFT_offset, F, plotQAM=plot) # get the payload symbols from
    #[958]

    #Converting OFDM Symbols to Data
    PS_est, hardDecision = Demapping(QAM_est, de_mapping_table_Qm) # demap the symbols back to codewords
    #PS_est[958, 6]
    #hardDecision[958]

    bits_est = PS(PS_est) # convert the codewords to the bitstream
    #0 1 bits [5748]
    return bits_est, SINR_m

###########################################


# custom dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.pdsch_iq = [] # pdsch symbols
        #self.pilot_iq = [] # pilot symbols
        self.labels = [] # original bitstream labels
        
    def __len__(self):
        return len(self.pdsch_iq)
    
    def __getitem__(self, index):
        x1 = self.pdsch_iq[index]
        #x2 = self.pilot_iq[index] 
        y = self.labels[index]
        return x1, y
    
    #def add_item(self, new_pdsch_iq, new_pilot_iq, new_label):
    def add_item(self, new_pdsch_iq,  new_label):
        self.pdsch_iq.append(new_pdsch_iq) 
        #self.pilot_iq.append(new_pilot_iq) 
        self.labels.append(new_label) 

def create_dataset():
    # SDR Configuration
    use_sdr = False # True for SDR or False for CDL-C channel emulation
    # for SDR
    SDR_TX_Frequency = int(436e6) # SDR TX frequency
    tx_gain_min = -50 # sdr tx max gain
    tx_gain_max = 0 # sdr tx min gain
    rx_gain = 20 # sdr rx gain

    #Signal-to-Interference-plus-Noise Ratio (SINR) for the CDL-C channel emulation
    # in case SDR not available, for channel simulation
    ch_SINR_min = 0 # channel emulation min SINR
    ch_SINR_max = 50 # channel emulation max SINR

    #Qm (int): Modulation order
    Qm = 6  # bits per symbol

    mapping_table_QPSK, de_mapping_table_QPSK = mapping_table(2) # mapping table QPSK (e.g. for pilot symbols)
    mapping_table_Qm, de_mapping_table_Qm = mapping_table(Qm, plot=False) # mapping table for Qm

    TTI_mask_RE = TTI_mask(S=S,F=F, Fp=Fp, Sp=Sp, FFT_offset=FFT_offset, plotTTI=False) #[14, 128]
    pilot_symbols = pilot_set(TTI_mask_RE, Pilot_Power) #[36]

    leading_zeros = 80  # For SDR, Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.
    

    dataset = CustomDataset()

    number_of_training_items = 10 #10000

    ch_SINR_min = 25 # channel emulation min SINR
    ch_SINR_max = 50 # channel emulation max SINR
    use_random_timing_offset = False

    # channel simulation
    n_taps = 2 
    max_delay = 6 #samples

    for i in range(number_of_training_items):
        if i % 1000 == 0:
            print(i)
        ch_SINR = int(random.uniform(ch_SINR_min, ch_SINR_max)) # SINR generation for adding noise to the channel
        #pdsch_bits, TX_Samples = create_OFDM_data() # data stream
        pdsch_bits, TX_Samples = create_OFDM_data(TTI_mask_RE, Qm, mapping_table_Qm, pilot_symbols, leading_zeros=leading_zeros, use_sdr=use_sdr, plot=False)
        #pdsch_bits: return bits: [958, 6], symbol: [958] complex, each symbol is 6 bits, 958 is the effective data slots in mask
        #TX_Samples:(S, CP 20+ FFT_size 128) 14*148 flatten to torch.Size([2072])

        
        RX_Samples = apply_multipath_channel(TX_Samples, n_taps=n_taps, max_delay=max_delay, random_start=False, repeats=0, SINR_s=ch_SINR, leading_zeros=0)
        symbol_index = 1
        RX_NO_CP = CP_removal(RX_Samples, symbol_index, S, FFT_size, CP, plotsig=False)# remove cyclic prefix and other symbols created by convolution
        RX_NO_CP = RX_NO_CP / torch.max(torch.abs(RX_NO_CP)) # normalize
        #torch.Size([14, 128])

        OFDM_demod = DFT(RX_NO_CP, plotDFT=False) # DFT
        OFDM_demod = OFDM_demod / torch.max(torch.abs(OFDM_demod)) # normalize DFT'd signal for NN input
        #torch.Size([14, 128])
        #F is number of carriers
        pdsch_symbols_map = remove_fft_Offests(OFDM_demod, F, FFT_offset) # remove FFT offsets
        #[14, 72]

        pdsch_symbols_map = torch.cat((pdsch_symbols_map[:, :F//2], pdsch_symbols_map[:, F//2 + 1:]), dim=1) # remove DC
        # [14, 71]
        # create a bit stream matrix, the same shape as TTI mask RE
        TTI_mask_indices = torch.where(TTI_mask_RE==1) #[14, 128]
        TTI_3d = torch.zeros((TTI_mask_RE.shape[0], TTI_mask_RE.shape[1], pdsch_bits.shape[1]), dtype=pdsch_bits.dtype)
        row_indices, col_indices = TTI_mask_indices
        #TTI_3d: [14, 128, 6]
        TTI_3d[row_indices, col_indices, :] = pdsch_bits.clone().detach()
        TTI_3d = remove_fft_Offests(TTI_3d, F, FFT_offset) #[14, 72, 6]
        TTI_3d = torch.cat((TTI_3d[:, :F//2,:], TTI_3d[:, F//2 + 1:,:]), dim=1)  # remove DC, [14, 71, 6]
        
        #dataset.add_item(pdsch_symbols_map, pilot_symbols_map, TTI_3d) # add to dataset
        dataset.add_item(pdsch_symbols_map, TTI_3d) 
        # add to dataset pdsch_symbols_map:[14, 71] complex, TTI_3d:[14, 71, 6] bits

    torch.save(dataset, 'output/ofdm_dataset.pth') # save dataset on disk
    print('Dataset saved')



###########################################

def oneround_test():
    print("Test ofdmsim_pytorchlib")

    use_sdr=False

    # OFDM Parameters
    #Qm (int): Modulation order
    Qm = 6  # bits per symbol

    mapping_table_Qm, de_mapping_table_Qm = mapping_table(Qm, plot=True) # mapping table for Qm

    #S=14 symbols, first 28 all zero, last 28 all zero, mid F=72 all 1, Sp=2 (2,1,3), yellow bar =3
    TTI_mask_RE = TTI_mask(S=S,F=F, Fp=Fp, Sp=Sp, FFT_offset=FFT_offset, plotTTI=True)

    pilot_symbols = pilot_set(TTI_mask_RE, Pilot_Power)

    if use_sdr:
        leading_zeros = 80  # For SDR, Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.
    else:
        leading_zeros = 80
    pdsch_bits, TX_Samples = create_OFDM_data(TTI_mask_RE, Qm, mapping_table_Qm, pilot_symbols, leading_zeros=leading_zeros, use_sdr=use_sdr)
    #pdsch_bits: return bits: [958, 6], symbol: [958] complex, each symbol is 6 bits, 958 is the effective data slots in mask
    #TX_Samples:(S, CP 20+ FFT_size 128) 14*148 flatten to torch.Size([2072])

    # channel simulation
    n_taps = 2 
    max_delay = 6 #samples
    #ch_SINR (int): Signal-to-Interference-plus-Noise Ratio (SINR) for the CDL-C channel emulation
    # 3GPP CDL-C Channel Simulation Parameters in case no sdr
    ch_SINR = 40  # SINR for channel emulation CDL-C
    tx_gain = 0  # Transmission gain in dB for SDR
    rx_gain = 10  # Reception gain in dB for SDR
    RX_Samples = radio_channel(tx_signal= TX_Samples, ch_SINR=ch_SINR, n_taps=n_taps, max_delay=max_delay, \
                  leading_zeros=leading_zeros, tx_gain=0, rx_gain = 0, use_sdr=False)
    #RX_Samples: torch.Size([6474]), TX_Samples: torch.Size([2072])

    Fc = int(920e6)  # SDR TX frequency in Hz
    PSD_plot(TX_Samples, SampleRate, Fc, 'TX')
    PSD_plot(RX_Samples, SampleRate, Fc, 'RX')

    bits_est, SINR_m = receiver_process(TX_Samples=TX_Samples, RX_Samples=RX_Samples, \
                                        TTI_mask_RE=TTI_mask_RE, pilot_symbols=pilot_symbols, \
                                            de_mapping_table_Qm=de_mapping_table_Qm, \
                                                leading_zeros=leading_zeros, use_sdr=use_sdr, plot=True)

    BER = calculate_BER(pdsch_bits, bits_est)
    print(f"BER: {BER}, SINR: {SINR_m}dB") # print the BER and SINR

    vis_errorbits(TTI_mask_RE, pdsch_bits, bits_est)

def performance_test():
    # SDR Configuration
    use_sdr = False # True for SDR or False for CDL-C channel emulation
    # for SDR
    SDR_TX_Frequency = int(436e6) # SDR TX frequency
    tx_gain_min = -50 # sdr tx max gain
    tx_gain_max = 0 # sdr tx min gain
    rx_gain = 20 # sdr rx gain

    #Signal-to-Interference-plus-Noise Ratio (SINR) for the CDL-C channel emulation
    # in case SDR not available, for channel simulation
    ch_SINR_min = 0 # channel emulation min SINR
    ch_SINR_max = 50 # channel emulation max SINR

    #Qm (int): Modulation order
    Qm = 6  # bits per symbol

    mapping_table_QPSK, de_mapping_table_QPSK = mapping_table(2) # mapping table QPSK (e.g. for pilot symbols)
    mapping_table_Qm, de_mapping_table_Qm = mapping_table(Qm, plot=False) # mapping table for Qm

    TTI_mask_RE = TTI_mask(S=S,F=F, Fp=Fp, Sp=Sp, FFT_offset=FFT_offset, plotTTI=False) #[14, 128]
    pilot_symbols = pilot_set(TTI_mask_RE, Pilot_Power) #[36]

    # start the SDR
    # if use_sdr: 
    #     SDR_1 = SDR_Pluto.SDR(SDR_TX_IP="ip:192.168.1.10", SDR_TX_FREQ=SDR_TX_Frequency, SDR_TX_GAIN=-80, SDR_RX_GAIN = 0, SDR_TX_SAMPLERATE=SampleRate, SDR_TX_BANDWIDTH=F*SCS*2)
    #     SDR_1.SDR_TX_start()

    leading_zeros = 80  # For SDR, Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.
    pdsch_bits, TX_Samples = create_OFDM_data(TTI_mask_RE, Qm, mapping_table_Qm, pilot_symbols, leading_zeros=leading_zeros, use_sdr=use_sdr)
    #pdsch_bits: return bits: [958, 6], symbol: [958] complex, each symbol is 6 bits, 958 is the effective data slots in mask
    #TX_Samples:(S, CP 20+ FFT_size 128) 14*148 flatten to torch.Size([2072])

    # channel simulation
    n_taps = 2 
    max_delay = 6 #samples
    #ch_SINR (int): Signal-to-Interference-plus-Noise Ratio (SINR) for the CDL-C channel emulation
    # 3GPP CDL-C Channel Simulation Parameters in case no sdr
    ch_SINR = 40  # SINR for channel emulation CDL-C
    tx_gain = 0  # Transmission gain in dB for SDR
    rx_gain = 10  # Reception gain in dB for SDR
    #RX_Samples = radio_channel(tx_signal= TX_Samples, ch_SINR=ch_SINR, n_taps=n_taps, max_delay=max_delay, \
    #              leading_zeros=leading_zeros, tx_gain=0, rx_gain = 0, use_sdr=False)
    #RX_Samples: torch.Size([6474]), TX_Samples: torch.Size([2072])

    number_of_testcases = 1000
    SINR2BER_table = np.zeros((number_of_testcases,4))

    for i in range(number_of_testcases):
        ch_SINR = int(random.uniform(ch_SINR_min, ch_SINR_max))
        tx_gain_i = int(random.uniform(tx_gain_min, tx_gain_max))
        #RX_Samples = radio_channel(use_sdr=use_sdr, tx_signal = TX_Samples, tx_gain = tx_gain_i, rx_gain = rx_gain, ch_SINR = ch_SINR)
        RX_Samples = radio_channel(tx_signal= TX_Samples, ch_SINR=ch_SINR, n_taps=n_taps, max_delay=max_delay, \
                  leading_zeros=leading_zeros, tx_gain=0, rx_gain = 0, use_sdr=False)
        
        bits_est, SINR_m = receiver_process(TX_Samples=TX_Samples, RX_Samples=RX_Samples, \
                                        TTI_mask_RE=TTI_mask_RE, pilot_symbols=pilot_symbols, \
                                            de_mapping_table_Qm=de_mapping_table_Qm, \
                                                leading_zeros=leading_zeros, use_sdr=use_sdr, plot=False)

        BER = calculate_BER(pdsch_bits, bits_est)
        print(f"BER: {BER}, SINR: {SINR_m}dB") # print the BER and SINR

        SINR2BER_table[i,0] = round(tx_gain_i,1)
        SINR2BER_table[i,1] = round(rx_gain,1)
        SINR2BER_table[i,2] = round(SINR_m,1)
        SINR2BER_table[i,3] = BER
    
    df = pd.DataFrame(SINR2BER_table, columns=['txg', 'rxg', 'SINR', 'BER'])
    df['SINR_binned'] = pd.cut(df['SINR'], bins=range(-10, 40, 2), labels=range(-9, 39, 2))
    df_grouped = df.groupby('SINR_binned', observed=False).mean()
    df_grouped.dropna(subset=['txg', 'rxg', 'SINR', 'BER'], inplace=True)

    df.to_pickle("./output/df.pkl")




if __name__ == '__main__':
    #oneround_test()

    #performance_test()

    create_dataset()