#https://varun19299.github.io/ID4100-Wireless-Lab-IITM/posts/4-ofdm-python/
#https://varun19299.github.io/ID4100-Wireless-Lab-IITM/posts/6-schmidl-cox/
#https://github.com/rishhabhnaik/End-to-end-OFDM-system-with-Deep-Learning-Driven-Approaches/blob/main/Deep%20Learning%20Model%20for%20Symbol%20Detection/ofdm_dnn_block.py
#https://github.com/haoyye/OFDM_DNN/blob/master/DNN_Detection/Train.py
#https://github.com/haoyye/OFDM_DNN

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate 
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import time

LastCarrierPilot = True
K = 64 # number of OFDM subcarriers
CP = K//4  # 16 length of the cyclic prefix: 25% of the block
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier. [ 0,  8, 16, 24, 32, 40, 48, 56], shape (8,)
if LastCarrierPilot:
    # For convenience of channel estimation, let's make the last carriers also be a pilot
    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])]) #[ 0,  8, 16, 24, 32, 40, 48, 56, 63] 63 is the last carrier
    P = P+1 #9
# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers) #(55,)

print ("allCarriers:   %s" % allCarriers) #(64,)
print ("pilotCarriers: %s" % pilotCarriers) #(9,)
print ("dataCarriers:  %s" % dataCarriers) #(55,)
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
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol, 55*4=220
print(K*mu) #256

# SNRdb = 12  # signal to noise-ratio in dB at the receiver 
SNRdb_list = [5, 10, 15, 20, 25]

Clipping_Flag = False 

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
plt.figure(figsize=(12,8)) #16 QAM Constellation
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



def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL*sigma
    x_clipped = x  
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))

    return x_clipped

def PAPR(x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB

def map_bits(bits):
    key = tuple(bits.tolist())
    return mapping_table[key]

def Modulation(bits):                                        
    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    # return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation
    return np.apply_along_axis(map_bits, axis=1, arr=bit_r) 

#serial-to-parallel converter
def SP(bits):
    return bits.reshape((len(dataCarriers), mu))

#The mapper converts the groups into complex-valued constellation symbols according to the mapping_table
def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers 64
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def addCP(OFDM_time): #CP: cyclic prefix: 25% of the block = 16
    cp = OFDM_time[-CP:]               # take the last CP samples ... (16,)
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def removeCP(signal):
    return signal[CP:(CP+K)]

def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal (9,)
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values (9,)
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers) #(64,)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers) #(64,)
    Hest = Hest_abs * np.exp(1j*Hest_phase) #(64,) complex
    
    return Hest, Hest_at_pilots

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]

#parallel to serial conversion
def PS(bits):
    return bits.reshape((-1,))

def Demapping(QAM): #(55,)
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()]) #(16,)
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1))) #(55, 16)
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1) #(55,)
    
    # get back the real constellation point
    hardDecision = constellation[const_index] #(55,)
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag, CR = 1):   
    payloadBits_per_OFDM = mu*len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    #OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX,CR)                            # add clipping 
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,K)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    #if len(codeword_qam) != K:
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    #OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword,CR) # add clipping 
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)

    #OFDM_RX_noCP_codeword = DFT(OFDM_RX_noCP_codeword)
    catresults = np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword)))))
    return catresults, abs(channelResponse)

def ofdm_simulate_single_without_CP(codeword, channelResponse):  
    
    codeword_qam = Modulation(codeword)
    OFDM_data_codeword = OFDM_symbol(codeword_qam)
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)

    # using a new ofdm symbol for the prefix
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    codeword_noise = Modulation(codeword)
    OFDM_data_nosie = OFDM_symbol(codeword_noise)
    OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)   
    cp = OFDM_time_noise[-CP:]               # take the last CP samples ...
    OFDM_withCP_cordword = np.hstack([cp,OFDM_time_codeword])
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    
    #return np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))) , abs(channelResponse) #sparse_mask
    return np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))), abs(channelResponse)

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       

        n_hidden_1 = 500
        n_hidden_2 = 250 # 1st layer num features
        n_hidden_3 = 120 # 2nd layer num features
        n_input = 256  
        n_output = 16 # every 16 bit are predicted by a model

        self.fc1 = torch.nn.Linear(n_input, n_hidden_1)   #256->500 
        self.fc2 = torch.nn.Linear(n_hidden_1, n_hidden_2) #500->250
        self.fc3 = torch.nn.Linear(n_hidden_2, n_hidden_3) #250->120
        self.fc4 = torch.nn.Linear(n_hidden_3, n_output)  #120->16


    def forward(self, x): #256->16
        # Encoder Hidden layer with sigmoid activation #1

        layer_1 = F.relu(self.fc1(x))
        layer_2 = F.relu(self.fc2(layer_1))
        layer_3 = F.relu(self.fc3(layer_2))
        layer_4 = F.sigmoid(self.fc4(layer_3))
        return layer_4

def training(SNRdb, device='cuda', H_folder="D:\\Dataset\\CommunicationDataset\\H_dataset"):     
    # Training parameters
    training_epochs = 20
    batch_size = 256
    display_step = 5
    test_step = 1000
    examples_to_show = 10   
    # Network Parameters
    n_output = 16 # every 16 bit are predicted by a model

    encoder = Encoder()
    encoder.to(device)
    # Targets (Labels) are the input data.
    # print(list(encoder.parameters())) # For debug purpose remove later!!
    # Define loss and optimizer, minimize the squared error
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion2 = torch.nn.L1Loss()
    optimizer = torch.optim.RMSprop(encoder.parameters(), lr=1e-3) # Check default RMSProp parameters later if not working

    # The H information set
    H_folder_train = H_folder
    H_folder_test = H_folder

    train_idx_low = 1
    train_idx_high = 301
    test_idx_low = 301
    test_idx_high = 401
    # Saving Channel conditions to a large matrix
    channel_response_set_train = []
    for train_idx in range(train_idx_low,train_idx_high):
        print("Processing the ", train_idx, "th document")
        #H_file = H_folder_train + str(train_idx) + '.txt'
        H_file = os.path.join(H_folder_train, str(train_idx) + '.txt')
        with open(H_file) as f:
            for line in f:
                try:
                    numbers_str = line.split() #32
                    numbers_float = [float(x) for x in numbers_str]
                    h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)]) + 1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)]) #(16,)
                    channel_response_set_train.append(h_response)
                except ValueError as V:
                    continue
    
    channel_response_set_test = []
    for test_idx in range(test_idx_low,test_idx_high):
        print("Processing the ", test_idx, "th document")
        #H_file = H_folder_test + str(test_idx) + '.txt'
        H_file = os.path.join(H_folder_test, str(test_idx) + '.txt')
        with open(H_file) as f:
            for line in f:
                try:
                    numbers_str = line.split()
                    numbers_float = [float(x) for x in numbers_str]
                    h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                    channel_response_set_test.append(h_response)
                except ValueError as V:
                    continue
    #len: 3000000, 1000000
    print ('length of training channel response', len(channel_response_set_train), 'length of testing channel response', len(channel_response_set_test))
    training_epochs = 250
    learning_rate_current = 0.001
    for epoch in range(training_epochs):
        print(epoch)
        if epoch > 0 and epoch%100 ==0:
            learning_rate_current = learning_rate_current / 5                    

        avg_cost = 0.
        total_batch = 50 
        for g in optimizer.param_groups: 
            g['lr'] = learning_rate_current     # Changing the learning rate with epochs

        for index_m in range(total_batch): #50 batch
            input_samples = []
            input_labels = []
            for index_k in range(0, 1000):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) #(220,)
                channel_response = channel_response_set_train[np.random.randint(0,len(channel_response_set_train))] #(16,)
                try:
                    signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)   
                except ValueError as V:
                    continue
                input_labels.append(bits[16:32])
                input_samples.append(signal_output)  
            batch_x = np.asarray(input_samples)
            batch_y = np.asarray(input_labels)            

            y_pred = encoder(torch.from_numpy(batch_x).float().to(device))
            loss = criterion(y_pred, torch.from_numpy(batch_y.astype(np.float32)).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c = loss.item()

            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            input_samples_test = []
            input_labels_test = []
            test_number = 1000
            # set test channel response for this epoch                    
            if epoch % test_step == 0:
                print ("Big Test Set ")
                test_number = 10000

            for i in range(0, test_number):
                #generate bits
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) 
                try:                       
                    channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                    signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)
                except ValueError as V:
                    continue
                input_labels_test.append(bits[16:32])
                input_samples_test.append(signal_output)
            batch_x = np.asarray(input_samples_test)
            batch_y = np.asarray(input_labels_test)
            y_pred = encoder(torch.from_numpy(batch_x).float().detach().to(device)).to(device)
            loss1_L1 = criterion2(y_pred, torch.from_numpy(batch_y.astype(np.float32)).to(device))
            mean_error = loss1_L1.item()
            # mean_error = torch.mean(abs(y_pred - torch.from_numpy(batch_y).detach()), keepdim=True)
            # mean_error_rate = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))               
            # mean_error_rate = 1 - torch.mean(torch.mean(), keepdim=True)
            mean_error_rate = 1 - np.mean(np.mean(np.equal(np.sign(y_pred.detach().cpu().numpy()-0.5), np.sign(batch_y-0.5)),axis=1))
            print("OFDM Detection QAM output number is", n_output, ",SNR = ", SNRdb, ",Num Pilot = ", P,", prediction and the mean error on test set are:", mean_error, mean_error_rate)

            batch_x = np.asarray(input_samples)
            batch_y = np.asarray(input_labels)
            y_pred = encoder(torch.from_numpy(batch_x).float().detach().to(device)).to(device)
            # mean_error = torch.mean(abs(y_pred - batch_y), keepdim=True)
            loss2_L1 = criterion2(y_pred, torch.from_numpy(batch_y.astype(np.float32)).to(device))
            mean_error = loss2_L1.item()

            # mean_error_rate = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))
            mean_error_rate = 1 - np.mean(np.mean(np.equal(np.sign(y_pred.detach().cpu().numpy()-0.5), np.sign(batch_y-0.5)),axis=1))
            print("Prediction and the mean error on train set are:", mean_error, mean_error_rate)



    print("optimization finished")
    return encoder, mean_error_rate

def mainfunc():
    bits = np.random.binomial(n=1, p=0.5, size=(K*mu, )) #(256,)
    #np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = Modulation(bits) #(64,)
    #QAM = Mapping(bits_SP) #(55,)
    CR = 1 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder_list = []
    ber_list = []
    for SNRdb in SNRdb_list:
        encoder, mean_error_rate = training(SNRdb, device)
        encoder_list.append(encoder)
        ber_list.append(mean_error_rate)

    # with open("./encoder_file_{0}.pkl".format(int(time.time())), "wb") as encFile:
    #     pickle.dump(encoder_list, encFile)

    # with open("./ber_list_file_{0}.pkl".format(int(time.time())), "wb") as berFile:
    #     pickle.dump(ber_list, berFile)


def myofdmsim():
    # define the wireless channel between transmitter and receiver. Here, we use a two-tap multipath channel with given impulse response channelResponse
    channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel, (3,)
    H_exact = np.fft.fft(channelResponse, K) #K=64 (number of OFDM subcarriers), shape (64,) K-point discrete Fourier Transform (DFT)
    plt.figure(figsize=(12,8))
    plt.plot(allCarriers, abs(H_exact)) #0-64 sub carrier, sin shape
    plt.xlabel('Subcarrier index'); plt.ylabel('$|H(f)|$'); plt.grid(True); plt.xlim(0, K-1)

    SNRdb = 25  # signal to noise-ratio in dB at the receiver 

    #starts with a random bit sequence b, generate the according bits by a random generator that draws from a Bernoulli distribution with p=0.5
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) #(220,)
    print ("Bits count: ", len(bits))
    print ("First 20 bits: ", bits[:20])
    print ("Mean of bits (should be around 0.5): ", np.mean(bits))

    #The bits are now sent to a serial-to-parallel converter, which groups the bits for the OFDM frame into a groups of mu bits (i.e. one group for each subcarrier):
    bits_SP = SP(bits) #(55, 4)
    print ("First 5 bit groups")
    print (bits_SP[:5,:])

    #the bits groups are sent to the mapper. The mapper converts the groups into complex-valued constellation symbols according to the mapping_table.
    QAM = Mapping(bits_SP) #(55,)
    print ("First 5 QAM symbols and bits:")
    print (bits_SP[:5,:])
    print (QAM[:5])

    #create the overall OFDM data, we need to put the data and pilots into the OFDM carriers:
    OFDM_data = OFDM_symbol(QAM) #(64,) complex
    print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))

    #Now, the OFDM carriers contained in OFDM_data can be transformed to the time-domain by means of the IDFT operation.
    OFDM_time = IDFT(OFDM_data) #(64,) complex
    print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))

    #Subsequently, we add a cyclic prefix to the symbol. This operation concatenates a copy of the last CP samples of the OFDM time domain signal to the beginning. This way, a cyclic extension is achieved. 
    OFDM_withCP = addCP(OFDM_time) #(80,) 16 (CP)+64
    print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))

    #Now, the signal is sent to the antenna and sent over the air to the receiver.
    OFDM_TX = OFDM_withCP #(80,)
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb) #(82,)
    plt.figure(figsize=(10,4))
    plt.plot(abs(OFDM_TX), label='TX signal')
    plt.plot(abs(OFDM_RX), label='RX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
    plt.grid(True)

    OFDM_RX_noCP = removeCP(OFDM_RX) #(64,)
    #Afterwards, the signal is transformed back to the frequency domain, in order to have the received value on each subcarrier available.
    OFDM_demod = DFT(OFDM_RX_noCP)#(64,)


    Hest, Hest_at_pilots = channelEstimate(OFDM_demod) #(64,)
    plt.figure(figsize=(12,8))
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.ylim(0,2)

    #Now that the channel is estimated at all carriers, we can use this information in the channel equalizer step. Here, for each subcarrier, the influence of the channel is removed such that we get the clear (only noisy) constellation symbols back.
    equalized_Hest = equalize(OFDM_demod, Hest) ##(64,) complex

    #The next step is to extract the data carriers from the equalized symbol. Here, we throw away the pilot carriers, as they do not provide any information, but were used for the channel estimation process.
    QAM_est = get_payload(equalized_Hest) #fetch dataCarriers (55,)
    plt.figure(figsize=(12,8))
    plt.plot(QAM_est.real, QAM_est.imag, 'bo')
    plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary Part'); plt.title("Received constellation")

    #Now, that the constellation is obtained back, we need to send the complex values to the demapper, to transform the constellation points to the bit groups. In order to do this, we compare each received constellation point against each possible constellation point and choose the constellation point which is closest to the received point. Then, we return the bit-group that belongs to this point.
    PS_est, hardDecision = Demapping(QAM_est) #(55,) -> (55, 4), (55,)
    plt.figure(figsize=(12,8))
    for qam, hard in zip(QAM_est, hardDecision):
        plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
        plt.plot(hardDecision.real, hardDecision.imag, 'ro')
    plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Hard Decision demapping')

    #Finally, the bit groups need to be converted to a serial stream of bits, by means of parallel to serial conversion.
    bits_est = PS(PS_est) #(55, 4) -> (220,)
    print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))



if __name__ == '__main__':
    #myofdmsim()
    mainfunc()