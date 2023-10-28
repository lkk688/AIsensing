#https://varun19299.github.io/ID4100-Wireless-Lab-IITM/posts/4-ofdm-python/
import numpy as np
import matplotlib.pyplot as plt
import scipy

K = 64 # number of OFDM subcarriers
CP = K//4  # length of the cyclic prefix: 25% of the block
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier. [ 0,  8, 16, 24, 32, 40, 48, 56]
# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])]) #[ 0,  8, 16, 24, 32, 40, 48, 56, 63] 63 is the last carrier
P = P+1 #9
# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers) #(55,)

print ("allCarriers:   %s" % allCarriers)
print ("pilotCarriers: %s" % pilotCarriers)
print ("dataCarriers:  %s" % dataCarriers)
plt.figure(figsize=(8,0.8))
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
plt.legend(fontsize=10, ncol=2)
plt.xlim((-1,K)); plt.ylim((-0.1, 0.3))
plt.xlabel('Carrier index')
plt.yticks([])
plt.grid(True)

#We consider 16QAM transmission, i.e. we have μ=4 bits per symbol
mu = 4 # bits per symbol (i.e. 16QAM)
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol, 220

#mapping from groups of 4 bits to a 16QAM constellation symbol
mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
plt.figure(figsize=(12,8))
for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
                plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
plt.grid(True)
plt.xlim((-4, 4)); plt.ylim((-4,4)); plt.xlabel('Real part (I)'); plt.ylabel('Imaginary part (Q)')
plt.title('16 QAM Constellation with Gray-Mapping')


#The demapping table is simply the inverse mapping of the mapping table:
demapping_table = {v : k for k, v in mapping_table.items()}

# define the wireless channel between transmitter and receiver. Here, we use a two-tap multipath channel with given impulse response channelResponse
channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
H_exact = np.fft.fft(channelResponse, K)
plt.figure(figsize=(12,8))
plt.plot(allCarriers, abs(H_exact))
plt.xlabel('Subcarrier index'); plt.ylabel('$|H(f)|$'); plt.grid(True); plt.xlim(0, K-1)

SNRdb = 25  # signal to noise-ratio in dB at the receiver 

#starts with a random bit sequence b, generate the according bits by a random generator that draws from a Bernoulli distribution with p=0.5
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
print ("Bits count: ", len(bits))
print ("First 20 bits: ", bits[:20])
print ("Mean of bits (should be around 0.5): ", np.mean(bits))

#The bits are now sent to a serial-to-parallel converter, which groups the bits for the OFDM frame into a groups of mu bits (i.e. one group for each subcarrier):
def SP(bits):
    return bits.reshape((len(dataCarriers), mu))
bits_SP = SP(bits) #(55, 4)
print ("First 5 bit groups")
print (bits_SP[:5,:])

#the bits groups are sent to the mapper. The mapper converts the groups into complex-valued constellation symbols according to the mapping_table.
def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])
QAM = Mapping(bits_SP) #(55,)
print ("First 5 QAM symbols and bits:")
print (bits_SP[:5,:])
print (QAM[:5])

#create the overall OFDM data, we need to put the data and pilots into the OFDM carriers:
def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers 64
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol
OFDM_data = OFDM_symbol(QAM)
print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))

#Now, the OFDM carriers contained in OFDM_data can be transformed to the time-domain by means of the IDFT operation.
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
OFDM_time = IDFT(OFDM_data) #(64,)
print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))

#Subsequently, we add a cyclic prefix to the symbol. This operation concatenates a copy of the last CP samples of the OFDM time domain signal to the beginning. This way, a cyclic extension is achieved. 
def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ... (16,)
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning
OFDM_withCP = addCP(OFDM_time) #(80,)
print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))

#Now, the signal is sent to the antenna and sent over the air to the receiver.
def channel(signal):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise
OFDM_TX = OFDM_withCP #(80,)
OFDM_RX = channel(OFDM_TX) #(82,)
plt.figure(figsize=(10,4))
plt.plot(abs(OFDM_TX), label='TX signal')
plt.plot(abs(OFDM_RX), label='RX signal')
plt.legend(fontsize=10)
plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
plt.grid(True)

def removeCP(signal):
    return signal[CP:(CP+K)]
OFDM_RX_noCP = removeCP(OFDM_RX) #(64,)
#Afterwards, the signal is transformed back to the frequency domain, in order to have the received value on each subcarrier available.
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
OFDM_demod = DFT(OFDM_RX_noCP)#(64,)

def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal (9,)
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase) #(64,)
    plt.figure(figsize=(12,8))
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.ylim(0,2)
    
    return Hest
Hest = channelEstimate(OFDM_demod) #(64,)

#Now that the channel is estimated at all carriers, we can use this information in the channel equalizer step. Here, for each subcarrier, the influence of the channel is removed such that we get the clear (only noisy) constellation symbols back.
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
equalized_Hest = equalize(OFDM_demod, Hest) ##(64,)

#The next step is to extract the data carriers from the equalized symbol. Here, we throw away the pilot carriers, as they do not provide any information, but were used for the channel estimation process.
def get_payload(equalized):
    return equalized[dataCarriers]
QAM_est = get_payload(equalized_Hest) #(55,)
plt.figure(figsize=(12,8))
plt.plot(QAM_est.real, QAM_est.imag, 'bo')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary Part'); plt.title("Received constellation");

#Now, that the constellation is obtained back, we need to send the complex values to the demapper, to transform the constellation points to the bit groups. In order to do this, we compare each received constellation point against each possible constellation point and choose the constellation point which is closest to the received point. Then, we return the bit-group that belongs to this point.
def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()]) #(16,)
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1))) #(55, 16)
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1) #(55,)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

PS_est, hardDecision = Demapping(QAM_est) #(55,) -> (55, 4) (55,)
plt.figure(figsize=(12,8))
for qam, hard in zip(QAM_est, hardDecision):
    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
    plt.plot(hardDecision.real, hardDecision.imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Hard Decision demapping');

#Finally, the bit groups need to be converted to a serial stream of bits, by means of parallel to serial conversion.
def PS(bits):
    return bits.reshape((-1,))
bits_est = PS(PS_est) #(55, 4) -> (220,)
print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))

#https://varun19299.github.io/ID4100-Wireless-Lab-IITM/posts/6-schmidl-cox/