#created 6/5/2024 to host all OFDM related code for the SDR radio
import numpy as np
from scipy import ndimage
from timeit import default_timer as timer
from scipy import signal
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 8.0
from matplotlib import colors

#Ref: deeplearning\ofdmsim2.py, create simple OFDM signal
class OFDMSymbol():
    #K: number of OFDM subcarriers
    #P: number of pilot carriers per OFDM block
    #mu:bits per symbol (i.e. 16QAM)
    def __init__(self, K = 64, P=8, mu = 4, plotfig=False) -> None:
        self.plotfig = plotfig
        #K = 64 # number of OFDM subcarriers
        CP = K//4  # 16 length of the cyclic prefix: 25% of the block
        #P = 8 # number of pilot carriers per OFDM block
        #We consider 16QAM transmission, i.e. we have μ=4 bits per symbol
        #mu = 4 # bits per symbol (i.e. 16QAM)
        
        print(K*mu) #256 total number of bits
        self.mu = mu
        self.K = K
        self.P = P
        self.CP = CP
        self.pilotValue = 3+3j # The known value each pilot transmits
        self.allCarriers, self.dataCarriers, self.pilotCarriers = self.createdatacarrier()
        #allCarriers: indices of all subcarriers ([0, 1, ... K-1])
        #pilotCarriers: Pilots is every (K/P)th carrier. [ 0,  8, 16, 24, 32, 40, 48, 56, 63], shape (9,)
        #data carriers are all remaining carriers

        self.payloadBits_per_OFDM = len(self.dataCarriers)*mu  # number of payload bits per OFDM symbol, 55*4=220
        self.mapping_table, self.demapping_table = self.create_mappingtable()

    def create_mappingtable(self):
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
        #The demapping table is simply the inverse mapping of the mapping table:
        demapping_table = {v : k for k, v in mapping_table.items()}

        if self.plotfig:
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

        return mapping_table, demapping_table

    def createdatacarrier(self, LastCarrierPilot = True):
        K= self.K
        P= self.P
        
        allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier. [ 0,  8, 16, 24, 32, 40, 48, 56], shape (8,)
        if LastCarrierPilot:
            # For convenience of channel estimation, let's make the last carriers also be a pilot
            pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])]) #[ 0,  8, 16, 24, 32, 40, 48, 56, 63] 63 is the last carrier
            P = P+1 #9
        # data carriers are all remaining carriers
        dataCarriers = np.delete(allCarriers, pilotCarriers) #(55,)

        if self.plotfig:
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
        return allCarriers, dataCarriers, pilotCarriers
        #allCarriers: indices of all subcarriers ([0, 1, ... K-1])
        #pilotCarriers: Pilots is every (K/P)th carrier. [ 0,  8, 16, 24, 32, 40, 48, 56], shape (8,)
        #data carriers are all remaining carriers

    def generatebits(self, payloadBits_per_OFDM):
        #starts with a random bit sequence b, generate the according bits by a random generator that draws from a Bernoulli distribution with p=0.5
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) #(220,)
        print ("Bits count: ", len(bits))
        print ("First 20 bits: ", bits[:20])
        print ("Mean of bits (should be around 0.5): ", np.mean(bits))
        return bits

    #serial-to-parallel converter
    def SP(self, bits):
        return bits.reshape((len(self.dataCarriers), self.mu))
    
    #The mapper converts the groups into complex-valued constellation symbols according to the mapping_table
    def Mapping(self, bits):
        return np.array([self.mapping_table[tuple(b)] for b in bits])
    
    def OFDM_symbol(self, QAM_payload):
        symbol = np.zeros(self.K, dtype=complex) # the overall K subcarriers 64
        symbol[self.pilotCarriers] = self.pilotValue  # allocate the pilot subcarriers 
        symbol[self.dataCarriers] = QAM_payload  # allocate the pilot subcarriers
        return symbol

    def IDFT(self, OFDM_data):
        return np.fft.ifft(OFDM_data)

    def DFT(self, OFDM_RX):
        return np.fft.fft(OFDM_RX)

    def addCP(self, OFDM_time): #CP: cyclic prefix: 25% of the block = 16
        cp = OFDM_time[-self.CP:]               # take the last CP samples ... (16,)
        return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

    def removeCP(self, signal):
        return signal[self.CP:(self.CP+self.K)]
    
    def createOFDMsignal(self):
        bits = self.generatebits(payloadBits_per_OFDM=self.payloadBits_per_OFDM)#220 information bits (size of payloadBits_per_OFDM)
        #The bits are now sent to a serial-to-parallel converter, which groups the bits for the OFDM frame into a groups of mu bits (i.e. one group for each subcarrier):
        bits_SP = self.SP(bits) #(55, 4) mu=4bits
        print ("First 5 bit groups")
        print (bits_SP[:5,:])

        #the bits groups are sent to the mapper. The mapper converts the groups into complex-valued constellation symbols according to the mapping_table.
        QAM = self.Mapping(bits_SP) #(55,) complex128
        print ("First 5 QAM symbols and bits:")
        #print (bits_SP[:5,:])
        print (QAM[:5])

        #create the overall OFDM data, we need to put the data and pilots into the OFDM carriers:
        OFDM_data = self.OFDM_symbol(QAM) #54 data symbol to 64 OFDM carrier, other places are pilot symbol (64,) complex
        print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data)) #64

        #Now, the OFDM carriers contained in OFDM_data can be transformed to the time-domain by means of the IDFT operation.
        OFDM_time = self.IDFT(OFDM_data) #idft to time domain (64,) complex128
        print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))

        #Subsequently, we add a cyclic prefix to the symbol. This operation concatenates a copy of the last CP samples of the OFDM time domain signal to the beginning. This way, a cyclic extension is achieved. 
        OFDM_withCP = self.addCP(OFDM_time) #(80,) last 16 (CP)+64
        print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP)) #80

        #Now, the signal is sent to the antenna and sent over the air to the receiver.
        OFDM_TX = OFDM_withCP #(80,)

        if self.plotfig:
            plt.figure(figsize=(10,4))
            plt.plot(abs(OFDM_TX), label='TX signal')
            #plt.plot(abs(OFDM_RX), label='RX signal')
            plt.legend(fontsize=10)
            plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
            plt.grid(True)
        return OFDM_TX

#copy code from /deeplearning/deepMIMO5.py
class StreamManagement():
    """Class for management of streams in multi-cell MIMO networks.

    Parameters
    ----------
    rx_tx_association : [num_rx, num_tx], np.int
        A binary NumPy array where ``rx_tx_association[i,j]=1`` means
        that receiver `i` gets one or multiple streams from
        transmitter `j`.

    num_streams_per_tx : int
        Indicates the number of streams that are transmitted by each
        transmitter.

    Note
    ----
    Several symmetry constraints on ``rx_tx_association`` are imposed
    to ensure efficient processing. All row sums and all column sums
    must be equal, i.e., all receivers have the same number of associated
    transmitters and all transmitters have the same number of associated
    receivers. It is also assumed that all transmitters send the same
    number of streams ``num_streams_per_tx``.
    """
    def __init__(self,
                 rx_tx_association,
                 num_streams_per_tx):

        super().__init__()
        self._num_streams_per_tx = int(num_streams_per_tx) #1
        self.rx_tx_association = rx_tx_association #(1,1)

    @property
    def rx_tx_association(self):
        """Association between receivers and transmitters.

        A binary NumPy array of shape `[num_rx, num_tx]`,
        where ``rx_tx_association[i,j]=1`` means that receiver `i`
        gets one ore multiple streams from transmitter `j`.
        """
        return self._rx_tx_association

    @property
    def num_rx(self):
        "Number of receivers."
        return self._num_rx

    @property
    def num_tx(self):
        "Number of transmitters."
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        "Number of streams per transmitter."
        return self._num_streams_per_tx

    @property
    def num_streams_per_rx(self):
        "Number of streams transmitted to each receiver."
        return int(self.num_tx*self.num_streams_per_tx/self.num_rx)

    @property
    def num_interfering_streams_per_rx(self):
        "Number of interfering streams received at each eceiver."
        return int(self.num_tx*self.num_streams_per_tx
                   - self.num_streams_per_rx)

    @property
    def num_tx_per_rx(self):
        "Number of transmitters communicating with a receiver."
        return self._num_tx_per_rx

    @property
    def num_rx_per_tx(self):
        "Number of receivers communicating with a transmitter."
        return self._num_rx_per_tx

    @property
    def precoding_ind(self):
        """Indices needed to gather channels for precoding.

        A NumPy array of shape `[num_tx, num_rx_per_tx]`,
        where ``precoding_ind[i,:]`` contains the indices of the
        receivers to which transmitter `i` is sending streams.
        """
        return self._precoding_ind

    @property
    def stream_association(self):
        """Association between receivers, transmitters, and streams.

        A binary NumPy array of shape
        `[num_rx, num_tx, num_streams_per_tx]`, where
        ``stream_association[i,j,k]=1`` means that receiver `i` gets
        the `k` th stream from transmitter `j`.
        """
        return self._stream_association

    @property
    def detection_desired_ind(self):
        """Indices needed to gather desired channels for receive processing.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that
        can be used to gather desired channels from the flattened
        channel tensor of shape
        `[...,num_rx, num_tx, num_streams_per_tx,...]`.
        The result of the gather operation can be reshaped to
        `[...,num_rx, num_streams_per_rx,...]`.
        """
        return self._detection_desired_ind

    @property
    def detection_undesired_ind(self):
        """Indices needed to gather undesired channels for receive processing.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that
        can be used to gather undesired channels from the flattened
        channel tensor of shape `[...,num_rx, num_tx, num_streams_per_tx,...]`.
        The result of the gather operation can be reshaped to
        `[...,num_rx, num_interfering_streams_per_rx,...]`.
        """
        return self._detection_undesired_ind

    @property
    def tx_stream_ids(self):
        """Mapping of streams to transmitters.

        A NumPy array of shape `[num_tx, num_streams_per_tx]`.
        Streams are numbered from 0,1,... and assiged to transmitters in
        increasing order, i.e., transmitter 0 gets the first
        `num_streams_per_tx` and so on.
        """
        return self._tx_stream_ids

    @property
    def rx_stream_ids(self):
        """Mapping of streams to receivers.

        A Numpy array of shape `[num_rx, num_streams_per_rx]`.
        This array is obtained from ``tx_stream_ids`` together with
        the ``rx_tx_association``. ``rx_stream_ids[i,:]`` contains
        the indices of streams that are supposed to be decoded by receiver `i`.
        """
        return self._rx_stream_ids

    @property
    def stream_ind(self):
        """Indices needed to gather received streams in the correct order.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that can be
        used to gather streams from the flattened tensor of received streams
        of shape `[...,num_rx, num_streams_per_rx,...]`. The result of the
        gather operation is then reshaped to
        `[...,num_tx, num_streams_per_tx,...]`.
        """
        return self._stream_ind

    @rx_tx_association.setter
    def rx_tx_association(self, rx_tx_association):
        """Sets the rx_tx_association and derives related properties. """

        # Make sure that rx_tx_association is a binary NumPy array
        rx_tx_association = np.array(rx_tx_association, np.int32)
        assert all(x in [0,1] for x in np.nditer(rx_tx_association)), \
            "All elements of `stream_association` must be 0 or 1."

        # Obtain num_rx, num_tx from stream_association shape
        self._num_rx, self._num_tx = np.shape(rx_tx_association)

        # Each receiver must be associated with the same number of transmitters
        num_tx_per_rx = np.sum(rx_tx_association, 1)
        assert np.min(num_tx_per_rx) == np.max(num_tx_per_rx), \
            """Each receiver needs to be associated with the same number
               of transmitters."""
        self._num_tx_per_rx = num_tx_per_rx[0]

        # Each transmitter must be associated with the same number of receivers
        num_rx_per_tx = np.sum(rx_tx_association, 0)
        assert np.min(num_rx_per_tx) == np.max(num_rx_per_tx), \
            """Each transmitter needs to be associated with the same number
               of receivers."""
        self._num_rx_per_tx = num_rx_per_tx[0]

        self._rx_tx_association = rx_tx_association

        # Compute indices for precoding
        self._precoding_ind = np.zeros([self.num_tx, self.num_rx_per_tx],
                                        np.int32)
        for i in range(self.num_tx):
            self._precoding_ind[i,:] = np.where(self.rx_tx_association[:,i])[0]

        # Construct the stream association matrix
        # The element [i,j,k]=1 indicates that receiver i, get the kth stream
        # from transmitter j.
        stream_association = np.zeros(
            [self.num_rx, self.num_tx, self.num_streams_per_tx], np.int32)
        n_streams = np.min([self.num_streams_per_rx, self.num_streams_per_tx])
        tmp = np.ones([n_streams])
        for j in range(self.num_tx):
            c = 0
            for i in range(self.num_rx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i,j]:
                    stream_association[i,j,c:c+self.num_streams_per_rx] = tmp
                    c += self.num_streams_per_rx
        self._stream_association = stream_association

        # Get indices of desired and undesired channel coefficients from
        # the flattened stream_association. These indices can be used by
        # a receiver to gather channels of desired and undesired streams.
        self._detection_desired_ind = \
                 np.where(np.reshape(stream_association, [-1])==1)[0]

        self._detection_undesired_ind = \
                 np.where(np.reshape(stream_association, [-1])==0)[0]

        # We number streams from 0,1,... and assign them to the TX
        # TX 0 gets the first num_streams_per_tx and so on:
        self._tx_stream_ids = np.reshape(
                    np.arange(0, self.num_tx*self.num_streams_per_tx),
                    [self.num_tx, self.num_streams_per_tx])

        # We now compute the stream_ids for each receiver
        self._rx_stream_ids = np.zeros([self.num_rx, self.num_streams_per_rx],
                                        np.int32)
        for i in range(self.num_rx):
            c = []
            for j in range(self.num_tx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i,j]:
                    tmp = np.where(stream_association[i,j])[0]
                    tmp += j*self.num_streams_per_tx
                    c += list(tmp)
            self._rx_stream_ids[i,:] = c

        # Get indices to bring received streams back to the right order in
        # which they were transmitted.
        self._stream_ind = np.argsort(np.reshape(self._rx_stream_ids, [-1]))

#https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint
def complex_normal(shape, var=1.0):
    r"""Generates a tensor of complex normal random variables.

    Input
    -----
    shape : tf.shape, or list
        The desired shape.

    var : float
        The total variance., i.e., each complex dimension has
        variance ``var/2``.

    dtype: tf.complex
        The desired dtype. Defaults to `tf.complex64`.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor of complex normal random variables.
    """
    # Half the variance for each dimension
    #var_dim = np.complex64(var/2)
    #var_dim = tf.cast(var, dtype.real_dtype)/tf.cast(2, dtype.real_dtype)
    #stddev = np.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    stddev = np.sqrt(var/2)
    xr = np.random.normal(loc=0.0, scale=stddev, size=shape)
    xi = np.random.normal(loc=0.0, scale=stddev, size=shape)
    x = xr + 1j*xi
    # xr = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    # xi = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    # x = tf.complex(xr, xi)

    return x

def pam_gray(b):
    # pylint: disable=line-too-long
    r"""Maps a vector of bits to a PAM constellation points with Gray labeling.

    This recursive function maps a binary vector to Gray-labelled PAM
    constellation points. It can be used to generated QAM constellations.
    The constellation is not normalized.

    Input
    -----
    b : [n], NumPy array
        Tensor with with binary entries.

    Output
    ------
    : signed int
        The PAM constellation point taking values in
        :math:`\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}`.

    Note
    ----
    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    if len(b)>1:
        return (1-2*b[0])*(2**len(b[1:]) - pam_gray(b[1:]))
    return 1-2*b[0]

def qam(num_bits_per_symbol, normalize=True):
    r"""Generates a QAM constellation.

    This function generates a complex-valued vector, where each element is
    a constellation point of an M-ary QAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be a multiple of two, e.g., 2, 4, 6, 8, etc.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.complex64
        The QAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a QAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}/2` is the number of bits
    per dimension.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol % 2 == 0 # is even
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be a multiple of 2") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.complex64)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b[0::2]) + 1j*pam_gray(b[1::2]) # PAM in each dimension

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol/2)
        qam_var = 1/(2**(n-2))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(qam_var)
    return c

def pam(num_bits_per_symbol, normalize=True):
    r"""Generates a PAM constellation.

    This function generates a real-valued vector, where each element is
    a constellation point of an M-ary PAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be positive.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.float32
        The PAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a PAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}` is the number of bits
    per symbol.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be positive") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.float32)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b)

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol)
        pam_var = 1/(2**(n-1))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(pam_var)
    return c

#ref from class Constellation
def CreateConstellation(constellation_type, num_bits_per_symbol,normalize=True):
    r"""
    Constellation(constellation_type, num_bits_per_symbol, initial_value=None, normalize=True, center=False, trainable=False, dtype=tf.complex64, **kwargs)

    Constellation that can be used by a (de)mapper.

    This class defines a constellation, i.e., a complex-valued vector of
    constellation points. A constellation can be trainable. The binary
    representation of the index of an element of this vector corresponds
    to the bit label of the constellation point. This implicit bit
    labeling is used by the ``Mapper`` and ``Demapper`` classes.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", the constellation points are randomly initialized
        if no ``initial_value`` is provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    initial_value : :math:`[2^\text{num_bits_per_symbol}]`, NumPy array or Tensor
        Initial values of the constellation points. If ``normalize`` or
        ``center`` are `True`, the initial constellation might be changed.

    normalize : bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    center : bool
        If `True`, the constellation is ensured to have zero mean.
        Defaults to `False`.

    trainable : bool
        If `True`, the constellation points are trainable variables.
        Defaults to `False`.

    dtype : [complex64, complex128], DType
        The dtype of the constellation.

    Output
    ------
    : :math:`[2^\text{num_bits_per_symbol}]`, ``dtype``
        The constellation.

    Note
    ----
    One can create a trainable PAM/QAM constellation. This is
    equivalent to creating a custom trainable constellation which is
    initialized with PAM/QAM constellation points.
    """
    num_bits_per_symbol = int(num_bits_per_symbol)
    if constellation_type=="qam":
        assert num_bits_per_symbol%2 == 0 and num_bits_per_symbol>0,\
            "num_bits_per_symbol must be a multiple of 2"
        num_bits_per_symbol = int(num_bits_per_symbol)

        points = qam(num_bits_per_symbol, normalize=normalize)
    if constellation_type=="pam":
        assert num_bits_per_symbol>0,\
            "num_bits_per_symbol must be integer"
        num_bits_per_symbol = int(num_bits_per_symbol)

        points = pam(num_bits_per_symbol, normalize=normalize)
    return points

class Mapper:
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 dtype=np.complex64,
                 #**kwargs
                ):
          self.num_bits_per_symbol = num_bits_per_symbol
          self.binary_base = 2**np.arange(num_bits_per_symbol-1, -1, -1, dtype=int) #array([2, 1], dtype=int32) [8, 4, 2, 1]
          self.points = CreateConstellation(constellation_type, num_bits_per_symbol) #(4,)
          self._return_indices = return_indices
    
    def create_symbol(self, inputs):
        #inputs: (64, 1024) #batch_size, bits len
        new_shape = [-1] + [int(inputs.shape[-1] / self.num_bits_per_symbol), self.num_bits_per_symbol] #[-1, 512, 2]
        reinputs_reshaped = np.reshape(inputs, new_shape) #(64, 512, 2)
        # Convert the last dimension to an integer
        int_rep = reinputs_reshaped * self.binary_base #(64, 512, 2)
        int_rep = np.sum(int_rep, axis=-1) #(64, 512)
        int_rep = int_rep.astype(np.int32)
        print(int_rep.shape)
        # Map integers to constellation symbols
        #x = tf.gather(self.points, int_rep, axis=0)
        symbs_list = [self.points[val_int] for val_int in int_rep]
        symbols=np.array(symbs_list) #(64, 512) complex64
        print(symbols.dtype)
        return symbols
    
    def __call__(self, inputs): #(64, 1, 1, 2128)
        #convert inputs.shape to a python list
        input_shape = list(inputs.shape) #[64, 1, 1, 2128]
        # Reshape inputs to the desired format
        new_shape = [-1] + input_shape[1:-1] + \
           [int(input_shape[-1] / self.num_bits_per_symbol),
            self.num_bits_per_symbol] #[-1, 1, 1, 532, 4]
        #inputs_reshaped = tf.cast(tf.reshape(inputs, new_shape), tf.int32)
        inputs_reshaped = np.reshape(inputs, new_shape).astype(np.int32) #(64, 1, 1, 532, 4)

        # Convert the last dimension to an integer
        #int_rep = tf.reduce_sum(inputs_reshaped * self._binary_base, axis=-1)
        int_rep = inputs_reshaped * self.binary_base #(64, 1, 1, 532, 4)
        int_rep = np.sum(int_rep, axis=-1) #(64, 1, 1, 532)
        int_rep = int_rep.astype(np.int32) #(64, 1, 1, 532)

        # Map integers to constellation symbols
        #x = tf.gather(self.constellation.points, int_rep, axis=0)
        symbs_list = [self.points[val_int] for val_int in int_rep]
        x=np.array(symbs_list) #(64, 1, 1, 532)

        if self._return_indices:
            return x, int_rep
        else:
            return x

class BinarySource:
    """BinarySource(dtype=float32, seed=None, **kwargs)

    Layer generating random binary tensors.

    Parameters
    ----------
    dtype : DType
        Defines the output datatype of the layer.
        Defaults to `float32`.

    seed : int or None
        Set the seed for the random generator used to generate the bits.
        Set to `None` for random initialization of the RNG.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor filled with random binary values.
    """
    def __init__(self, dtype=np.float32, seed=None, **kwargs):
        #super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        if self._seed is not None:
            self._rng = np.random.RandomState(self._seed)

    def __call__(self, inputs): #inputs is shape
        if self._seed is not None:
            return self._rng.randint(low=0, high=2, size=inputs).astype(np.float32)
            # return tf.cast(self._rng.uniform(inputs, 0, 2, tf.int32),
            #                dtype=super().dtype)
        else:
            return np.random.randint(low=0, high=2, size=inputs).astype(np.float32)
            # return tf.cast(tf.random.uniform(inputs, 0, 2, tf.int32),
            #                dtype=super().dtype)

class SymbolSource():
    r"""SymbolSource(constellation_type=None, num_bits_per_symbol=None, constellation=None, return_indices=False, return_bits=False, seed=None, dtype=tf.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random constellation symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [complex64, complex128], DType
        The output dtype. Defaults to complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random symbols of the chosen ``constellation_type``.

    symbol_indices : ``shape``, int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=np.complex64,
                 **kwargs
                ):
        self.points = CreateConstellation(constellation_type, num_bits_per_symbol) #(4,)
        self._num_bits_per_symbol = num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        #self._binary_source = BinarySource(seed=seed, dtype=dtype.real_dtype)
        self._binary_source = BinarySource()
        self._mapper = Mapper(constellation_type=constellation_type,
                              return_indices=return_indices,
                              num_bits_per_symbol=num_bits_per_symbol,
                              dtype=dtype)

    def __call__(self, inputs):
        #shape = tf.concat([inputs, [self._num_bits_per_symbol]], axis=-1)
        shape = np.concatenate([inputs, [self._num_bits_per_symbol]], axis=-1)
        shape = shape.astype(np.int32)
        #b = self._binary_source(tf.cast(shape, tf.int32))
        b = self._binary_source(shape)
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)

        #result = tf.squeeze(x, -1)
        result = np.squeeze(x, -1)
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            #result.append(tf.squeeze(ind, -1))
            result.append(np.squeeze(ind, -1))
        if self._return_bits:
            result.append(b)

        return result
    

class PilotPattern():
    r"""Class defining a pilot pattern for an OFDM ResourceGrid.

    Parameters
    ----------
    mask : [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], bool
        Tensor indicating resource elements that are reserved for pilot transmissions.

    pilots : [num_tx, num_streams_per_tx, num_pilots], complex
        The pilot symbols to be mapped onto the ``mask``.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension. This can be useful to
        ensure that trainable ``pilots`` have a finite energy.
        Defaults to `False`.

    dtype : Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `complex64`.
    """
    def __init__(self, mask, pilots, trainable=False, normalize=False,
                 dtype=np.complex64):
        super().__init__()
        self._dtype = dtype
        #self._mask = tf.cast(mask, tf.int32)
        self._mask = mask.astype(np.int32) #(1, 1, 14, 76)
        #self._pilots = tf.Variable(tf.cast(pilots, self._dtype), trainable)
        self._pilots = pilots.astype(self._dtype) #(1, 1, 0) complex
        self.normalize = normalize
        #self._check_settings()

    @property
    def num_tx(self):
        """Number of transmitters"""
        return self._mask.shape[0]

    @property
    def num_streams_per_tx(self):
        """Number of streams per transmitter"""
        return self._mask.shape[1]

    @ property
    def num_ofdm_symbols(self):
        """Number of OFDM symbols"""
        return self._mask.shape[2]

    @ property
    def num_effective_subcarriers(self):
        """Number of effectvie subcarriers"""
        return self._mask.shape[3]

    @property
    def num_pilot_symbols(self):
        """Number of pilot symbols per transmit stream."""
        #return tf.shape(self._pilots)[-1]
        return np.shape(self._pilots)[-1]


    @property
    def num_data_symbols(self):
        """ Number of data symbols per transmit stream."""
        # return tf.shape(self._mask)[-1]*tf.shape(self._mask)[-2] - \
        #        self.num_pilot_symbols
        return np.shape(self._mask)[-1]*np.shape(self._mask)[-2] - \
               self.num_pilot_symbols

    @property
    def normalize(self):
        """Returns or sets the flag indicating if the pilots
           are normalized or not
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        #self._normalize = tf.cast(value, tf.bool)
        self._normalize =value


    @property
    def mask(self):
        """Mask of the pilot pattern"""
        return self._mask

    @property
    def pilots(self):
        """Returns or sets the possibly normalized tensor of pilot symbols.
           If pilots are normalized, the normalization will be applied
           after new values for pilots have been set. If this is
           not the desired behavior, turn normalization off.
        """
        def norm_pilots():
            #scale = tf.abs(self._pilots)**2
            scale = np.abs(self._pilots)**2
            #scale = 1/tf.sqrt(tf.reduce_mean(scale, axis=-1, keepdims=True))
            scale = 1/np.sqrt(np.mean(scale, axis=-1, keepdims=True))
            #scale = tf.cast(scale, self._dtype)
            scale = scale.astype(self._dtype)

            return scale*self._pilots

        #conditionally execute different operations based on the value of a boolean tensor
        #return tf.cond(self.normalize, norm_pilots, lambda: self._pilots)
        if self.normalize:
            return norm_pilots()
        else:
            return self._pilots

    @pilots.setter
    def pilots(self, value):
        self._pilots.assign(value)


    def show(self, tx_ind=None, stream_ind=None, show_pilot_ind=False):
        """Visualizes the pilot patterns for some transmitters and streams.

        Input
        -----
        tx_ind : list, int
            Indicates the indices of transmitters to be included.
            Defaults to `None`, i.e., all transmitters included.

        stream_ind : list, int
            Indicates the indices of streams to be included.
            Defaults to `None`, i.e., all streams included.

        show_pilot_ind : bool
            Indicates if the indices of the pilot symbols should be shown.

        Output
        ------
        list : matplotlib.figure.Figure
            List of matplot figure objects showing each the pilot pattern
            from a specific transmitter and stream.
        """
        # mask = self.mask.numpy() #(1, 1, 14, 76)
        # pilots = self.pilots.numpy() #(1, 1, 152)
        mask = self.mask
        pilots = self.pilots

        if tx_ind is None:
            tx_ind = range(0, self.num_tx) #range(0,1)
        elif not isinstance(tx_ind, list):
            tx_ind = [tx_ind]

        if stream_ind is None:
            stream_ind = range(0, self.num_streams_per_tx) #range(0,1)
        elif not isinstance(stream_ind, list):
            stream_ind = [stream_ind]

        figs = []
        for i in tx_ind: #range(0,1)
            for j in stream_ind: #range(0,1)
                q = np.zeros_like(mask[0,0]) #(14, 76)
                q[np.where(mask[i,j])] = (np.abs(pilots[i,j])==0) + 1
                legend = ["Data", "Pilots", "Masked"]
                fig = plt.figure()
                plt.title(f"TX {i} - Stream {j}")
                plt.xlabel("OFDM Symbol")
                plt.ylabel("Subcarrier Index")
                plt.xticks(range(0, q.shape[1]))
                cmap = plt.cm.tab20c
                b = np.arange(0, 4)
                norm = colors.BoundaryNorm(b, cmap.N)
                im = plt.imshow(np.transpose(q), origin="lower", aspect="auto", norm=norm, cmap=cmap)
                cbar = plt.colorbar(im)
                cbar.set_ticks(b[:-1]+0.5)
                cbar.set_ticklabels(legend)

                if show_pilot_ind:
                    c = 0
                    for t in range(self.num_ofdm_symbols):
                        for k in range(self.num_effective_subcarriers):
                            if mask[i,j][t,k]:
                                if np.abs(pilots[i,j,c])>0:
                                    plt.annotate(c, [t, k])
                                c+=1
                figs.append(fig)

        return figs

class EmptyPilotPattern(PilotPattern):
    """Creates an empty pilot pattern.

    Generates a instance of :class:`PilotPattern` with
    an empty ``mask`` and ``pilots``.

    Parameters
    ----------
    num_tx : int
        Number of transmitters.

    num_streams_per_tx : int
        Number of streams per transmitter.

    num_ofdm_symbols : int
        Number of OFDM symbols.

    num_effective_subcarriers : int
        Number of effective subcarriers
        that are available for the transmission of data and pilots.
        Note that this number is generally smaller than the ``fft_size``
        due to nulled subcarriers.

    dtype : Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `complex64`.
    """
    def __init__(self,
                 num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 dtype=np.complex64):

        assert num_tx > 0, \
            "`num_tx` must be positive`."
        assert num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert num_effective_subcarriers > 0, \
            "`num_effective_subcarriers` must be positive`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers] #[1, 1, 14, 76]
        #mask = tf.zeros(shape, tf.bool)
        mask = np.zeros(shape, np.bool_)
        #pilots = tf.zeros(shape[:2]+[0], dtype)
        pilots = np.zeros(shape[:2]+[0], np.bool_) #(1, 1, 0)
        super().__init__(mask, pilots, trainable=False, normalize=False,
                         dtype=dtype)
class KroneckerPilotPattern(PilotPattern):
    """Simple orthogonal pilot pattern with Kronecker structure.

    This function generates an instance of :class:`~sionna.ofdm.PilotPattern`
    that allocates non-overlapping pilot sequences for all transmitters and
    streams on specified OFDM symbols. As the same pilot sequences are reused
    across those OFDM symbols, the resulting pilot pattern has a frequency-time
    Kronecker structure. This structure enables a very efficient implementation
    of the LMMSE channel estimator. Each pilot sequence is constructed from
    randomly drawn QPSK constellation points.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of a :class:`~sionna.ofdm.ResourceGrid`.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension.
        Defaults to `True`.

    seed : int
        Seed for the generation of the pilot sequence. Different seed values
        lead to different sequences. Defaults to 0.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Note
    ----
    It is required that the ``resource_grid``'s property
    ``num_effective_subcarriers`` is an
    integer multiple of ``num_tx * num_streams_per_tx``. This condition is
    required to ensure that all transmitters and streams get
    non-overlapping pilot sequences. For a large number of streams and/or
    transmitters, the pilot pattern becomes very sparse in the frequency
    domain.

    Examples
    --------
    >>> rg = ResourceGrid(num_ofdm_symbols=14,
    ...                   fft_size=64,
    ...                   subcarrier_spacing = 30e3,
    ...                   num_tx=4,
    ...                   num_streams_per_tx=2,
    ...                   pilot_pattern = "kronecker",
    ...                   pilot_ofdm_symbol_indices = [2, 11])
    >>> rg.pilot_pattern.show();

    .. image:: ../figures/kronecker_pilot_pattern.png

    """
    def __init__(self,
                 resource_grid,
                 pilot_ofdm_symbol_indices,
                 normalize=True,
                 seed=0,
                 dtype=np.complex64):

        num_tx = resource_grid.num_tx #1
        num_streams_per_tx = resource_grid.num_streams_per_tx #1
        num_ofdm_symbols = resource_grid.num_ofdm_symbols #14
        num_effective_subcarriers = resource_grid.num_effective_subcarriers #76
        self._dtype = dtype

        # Number of OFDM symbols carrying pilots
        num_pilot_symbols = len(pilot_ofdm_symbol_indices) #2

        # Compute the total number of required orthogonal sequences
        num_seq = num_tx*num_streams_per_tx #1

        # Compute the length of a pilot sequence
        num_pilots = num_pilot_symbols*num_effective_subcarriers/num_seq #2*76=152
        assert num_pilots%1==0, \
            """`num_effective_subcarriers` must be an integer multiple of
            `num_tx`*`num_streams_per_tx`."""

        # Number of pilots per OFDM symbol
        num_pilots_per_symbol = int(num_pilots/num_pilot_symbols) #76

        # Prepare empty mask and pilots
        shape = [num_tx, num_streams_per_tx,
                 num_ofdm_symbols,num_effective_subcarriers] #kronecker
        mask = np.zeros(shape, bool)
        shape[2] = num_pilot_symbols #[1, 1, 14, 76]->[1, 1, 2, 76]
        pilots = np.zeros(shape, np.complex64) #(1, 1, 2, 76)

        # Populate all selected OFDM symbols in the mask
        mask[..., pilot_ofdm_symbol_indices, :] = True #[1, 1, 14, 76], [2, 11] col(14) set to True

        # Populate the pilots with random QPSK symbols
        #qam_source = QAMSource(2, seed=seed, dtype=self._dtype)
        qam_source = SymbolSource(constellation_type="qam", num_bits_per_symbol=2, dtype=self._dtype)
        for i in range(num_tx):
            for j in range(num_streams_per_tx):
                # Generate random QPSK symbols
                p = qam_source([1,1,num_pilot_symbols,num_pilots_per_symbol])

                # Place pilots spaced by num_seq to avoid overlap
                pilots[i,j,:,i*num_streams_per_tx+j::num_seq] = p

        # Reshape the pilots tensor
        pilots = np.reshape(pilots, [num_tx, num_streams_per_tx, -1])

        super().__init__(mask, pilots, trainable=False,
                         normalize=normalize, dtype=self._dtype)
        
class MyResourceGrid():
    r"""Defines a `ResourceGrid` spanning multiple OFDM symbols and subcarriers.

    Parameters
    ----------
        num_ofdm_symbols : int
            Number of OFDM symbols.

        fft_size : int
            FFT size (, i.e., the number of subcarriers).

        subcarrier_spacing : float
            The subcarrier spacing in Hz.

        num_tx : int
            Number of transmitters.

        num_streams_per_tx : int
            Number of streams per transmitter.

        cyclic_prefix_length : int
            Length of the cyclic prefix.

        num_guard_carriers : int
            List of two integers defining the number of guardcarriers at the
            left and right side of the resource grid.

        dc_null : bool
            Indicates if the DC carrier is nulled or not.

        pilot_pattern : One of [None, "kronecker", "empty", PilotPattern]
            Defaults to `None` which is equivalent to `"empty"`.

        pilot_ofdm_symbol_indices : List, int
            List of indices of OFDM symbols reserved for pilot transmissions.
            Only needed if ``pilot_pattern="kronecker"``. Defaults to `None`.

        dtype : 
            Defines the datatype for internal calculations and the output
    """
    def __init__(self,
                 num_ofdm_symbols,
                 fft_size,
                 subcarrier_spacing,
                 num_tx=1,
                 num_streams_per_tx=1,
                 cyclic_prefix_length=0,
                 num_guard_carriers=(0,0),
                 dc_null=False,
                 pilot_pattern=None,
                 pilot_ofdm_symbol_indices=None,
                 dtype=np.complex64):
        super().__init__()
        self._dtype = dtype
        self._num_ofdm_symbols = num_ofdm_symbols #14
        self._fft_size = fft_size #76
        self._subcarrier_spacing = subcarrier_spacing #30000
        self._cyclic_prefix_length = int(cyclic_prefix_length) #6
        self._num_tx = num_tx #1
        self._num_streams_per_tx = num_streams_per_tx #1
        self._num_guard_carriers = np.array(num_guard_carriers) #(0,0)
        self._dc_null = dc_null #False
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices #[2,11]
        self.pilot_pattern = pilot_pattern #'kronecker'
        self._check_settings()

    @property
    def cyclic_prefix_length(self):
        """Length of the cyclic prefix."""
        return self._cyclic_prefix_length

    @property
    def num_tx(self):
        """Number of transmitters."""
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        """Number of streams  per transmitter."""
        return self._num_streams_per_tx

    @property
    def num_ofdm_symbols(self):
        """The number of OFDM symbols of the resource grid."""
        return self._num_ofdm_symbols

    @property
    def num_resource_elements(self):
        """Number of resource elements."""
        return self._fft_size*self._num_ofdm_symbols

    @property
    def num_effective_subcarriers(self):
        """Number of subcarriers used for data and pilot transmissions."""
        n = self._fft_size - self._dc_null - np.sum(self._num_guard_carriers) #no change 76
        return n

    @property
    def effective_subcarrier_ind(self):
        """Returns the indices of the effective subcarriers."""
        num_gc = self._num_guard_carriers
        sc_ind = range(num_gc[0], self.fft_size-num_gc[1])
        if self.dc_null:
            sc_ind = np.delete(sc_ind, self.dc_ind-num_gc[0])
        return sc_ind

    @property
    def num_data_symbols(self):
        """Number of resource elements used for data transmissions."""
        n = self.num_effective_subcarriers * self._num_ofdm_symbols - \
               self.num_pilot_symbols
        #return tf.cast(n, tf.int32)
        return n.astype(np.int32)


    @property
    def num_pilot_symbols(self):
        """Number of resource elements used for pilot symbols."""
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self):
        """Number of empty resource elements."""
        n = (self._fft_size-self.num_effective_subcarriers) * \
               self._num_ofdm_symbols
        #return tf.cast(n, tf.int32)
        return n.astype(np.int32)

    @property
    def num_guard_carriers(self):
        """Number of left and right guard carriers."""
        return self._num_guard_carriers

    @property
    def dc_ind(self):
        """Index of the DC subcarrier.

        If ``fft_size`` is odd, the index is (``fft_size``-1)/2.
        If ``fft_size`` is even, the index is ``fft_size``/2.
        """
        return int(self._fft_size/2 - (self._fft_size%2==1)/2)

    @property
    def fft_size(self):
        """The FFT size."""
        return self._fft_size

    @property
    def subcarrier_spacing(self):
        """The subcarrier spacing [Hz]."""
        return self._subcarrier_spacing

    @property
    def ofdm_symbol_duration(self):
        """Duration of an OFDM symbol with cyclic prefix [s]."""
        return (1. + self.cyclic_prefix_length/self.fft_size) \
                / self.subcarrier_spacing

    @property
    def bandwidth(self):
        """The occupied bandwidth [Hz]: ``fft_size*subcarrier_spacing``."""
        return self.fft_size*self.subcarrier_spacing

    @property
    def num_time_samples(self):
        """The number of time-domain samples occupied by the resource grid."""
        return (self.fft_size + self.cyclic_prefix_length) \
                * self._num_ofdm_symbols #(76+6)*14

    @property
    def dc_null(self):
        """Indicates if the DC carriers is nulled or not."""
        return self._dc_null

    @property
    def pilot_pattern(self):
        """The used PilotPattern."""
        return self._pilot_pattern

    @pilot_pattern.setter
    def pilot_pattern(self, value):
        if value is None:
            value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
        elif isinstance(value, PilotPattern):
            pass
        elif isinstance(value, str):
            assert value in ["kronecker", "empty"],\
                "Unknown pilot pattern"
            if value=="empty":
                value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
            elif value=="kronecker":
                assert self._pilot_ofdm_symbol_indices is not None,\
                    "You must provide pilot_ofdm_symbol_indices."
                #Kronecker not implemented
                value = KroneckerPilotPattern(self,
                        self._pilot_ofdm_symbol_indices, dtype=self._dtype)
        else:
            raise ValueError("Unsupported pilot_pattern")
        self._pilot_pattern = value

    def _check_settings(self):
        """Validate that all properties define a valid resource grid"""
        assert self._num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert self._fft_size > 0, \
            "`fft_size` must be positive`."
        assert self._cyclic_prefix_length>=0, \
            "`cyclic_prefix_length must be nonnegative."
        assert self._cyclic_prefix_length<=self._fft_size, \
            "`cyclic_prefix_length cannot be longer than `fft_size`."
        assert self._num_tx > 0, \
            "`num_tx` must be positive`."
        assert self._num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert len(self._num_guard_carriers)==2, \
            "`num_guard_carriers` must have two elements."
        assert np.all(np.greater_equal(self._num_guard_carriers, 0)), \
            "`num_guard_carriers` must have nonnegative entries."
        assert np.sum(self._num_guard_carriers)<=self._fft_size-self._dc_null,\
            "Total number of guardcarriers cannot be larger than `fft_size`."
        assert self._dtype in [np.complex64, np.complex128], \
            "dtype must be complex64 or complex128"
        return True

    def build_type_grid(self):
        """Returns a tensor indicating the type of each resource element.

        Resource elements can be one of

        - 0 : Data symbol
        - 1 : Pilot symbol
        - 2 : Guard carrier symbol
        - 3 : DC carrier symbol

        Output
        ------
        : [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.int32
            Tensor indicating for each transmitter and stream the type of
            the resource elements of the corresponding resource grid.
            The type can be one of [0,1,2,3] as explained above.
        """
        shape = [self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols] #[1,1,14]
        #gc_l = 2*tf.ones(shape+[self._num_guard_carriers[0]], tf.int32) #(1, 1, 14, 0)
        gc_l = 2*np.ones(shape+[self._num_guard_carriers[0]], np.int32)
        #gc_r = 2*tf.ones(shape+[self._num_guard_carriers[1]], tf.int32) #(1, 1, 14, 0)
        gc_r = 2*np.ones(shape+[self._num_guard_carriers[1]], np.int32) #(1, 1, 14, 0)
        #dc   = 3*tf.ones(shape + [tf.cast(self._dc_null, tf.int32)], tf.int32) #(1, 1, 14, 0)
        dc   = 3*np.ones(shape + [int(self._dc_null)], np.int32) #(1, 1, 14, 0)
        mask = self.pilot_pattern.mask #(1, 1, 14, 76)
        split_ind = self.dc_ind-self._num_guard_carriers[0] #38-0=38
        # rg_type = tf.concat([gc_l,                 # Left Guards
        #                      mask[...,:split_ind], # Data & pilots
        #                      dc,                   # DC
        #                      mask[...,split_ind:], # Data & pilots
        #                      gc_r], -1)            # Right guards
        rg_type = np.concatenate([gc_l,                 # Left Guards (1, 1, 14, 0)
                             mask[...,:split_ind], # Data & pilots
                             dc,                   # DC (1, 1, 14, 0)
                             mask[...,split_ind:], # Data & pilots
                             gc_r], -1)            # Right guards (1, 1, 14, 0)
        return rg_type #(1, 1, 14, 76) #38+38

    def show(self, tx_ind=0, tx_stream_ind=0):
        """Visualizes the resource grid for a specific transmitter and stream.

        Input
        -----
        tx_ind : int
            Indicates the transmitter index.

        tx_stream_ind : int
            Indicates the index of the stream.

        Output
        ------
        : `matplotlib.figure`
            A handle to a matplot figure object.
        """
        fig = plt.figure()
        data = self.build_type_grid()[tx_ind, tx_stream_ind] #0,0 =>[14,76]
        cmap = colors.ListedColormap([[60/256,8/256,72/256],
                              [45/256,91/256,128/256],
                              [45/256,172/256,111/256],
                              [250/256,228/256,62/256]])
        bounds=[0,1,2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(np.transpose(data), interpolation="nearest",
                         origin="lower", cmap=cmap, norm=norm,
                         aspect="auto")
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5,3.5],
                            orientation="vertical", shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot", "Guard carrier", "DC carrier"])
        plt.title("OFDM Resource Grid")
        plt.ylabel("Subcarrier Index")
        plt.xlabel("OFDM Symbol")
        plt.xticks(range(0, data.shape[0]))

        return fig


def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor.

    This operation flattens the last ``num_dims`` dimensions of a ``tensor``.
    It is a simplified version of the function ``flatten_dims``.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be greater than or equal to two and less or equal
            than the rank of ``tensor``.

    Returns:
        A tensor of the same type as ``tensor`` with ``num_dims``-1 lesser
        dimensions, but the same number of elements.
    """

    if num_dims==len(np.shape(tensor)): #len(tensor.shape):
        new_shape = [-1]
    else:
        shape = np.shape(tensor) #tf.shape(tensor)
        flipshape = shape[-num_dims:]
        last_dim = np.prod(flipshape)
        #last_dim = tf.reduce_prod(tensor.shape[-num_dims:])
        #new_shape = tf.concat([shape[:-num_dims], [last_dim]], 0)
        new_shape = np.concatenate([shape[:-num_dims], [last_dim]], 0)

    return np.reshape(tensor, new_shape) #tf.reshape(tensor, new_shape)

##Scatters updates into a tensor of shape shape according to indices
def scatter_nd_numpy(indices, updates, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
    updates = updates.ravel()
    np.add.at(target, indices, updates)
    return target

def tensor_scatter_nd_update(tensor, indices, updates):
    """
    Updates the `tensor` by scattering `updates` into it at the specified `indices`.

    :param tensor: Existing tensor to update
    :param indices: Array of indices where updates should be placed
    :param updates: Array of values to scatter
    :return: Updated tensor
    """
    # Create a tuple of indices for advanced indexing
    index_tuple = tuple(indices.T) #(1064, 4) = > 4 tuple array (1064,) each, same to np.where output, means all places

    # Scatter values from updates into tensor
    tensornew = tensor.copy() #(1, 1, 14, 76, 64)
    tensornew[index_tuple] = updates #updates(1064,64) to all 1064 locations
    #(1, 1, 14, 76, 64) updates(1064, 64)

    #print the last dimension data of tensornew
    #print(tensornew[0,0,0,0,:]) #(64,)
    #print(updates[0,:]) #(64,)
    return tensornew #(1, 1, 14, 76, 64)

def scatter_numpy(tensor, indices, values):
    """
    Scatters values into a tensor at specified indices.
    
    :param tensor: The target tensor to scatter values into.
    :param indices: Indices where values should be placed.
    :param values: Values to scatter.
    :return: Updated tensor after scattering.
    """
    # Scatter values
    tensor[tuple(indices.T)] = values

    return tensor

class MyResourceGridMapper:
    r"""ResourceGridMapper(resource_grid, dtype=complex64, **kwargs)

    Maps a tensor of modulated data symbols to a ResourceGrid.

    Takes as input a tensor of modulated data symbols
    and maps them together with pilot symbols onto an
    OFDM `ResourceGrid`. 

    Parameters
    ----------
    resource_grid : ResourceGrid
    dtype

    x = mapper(b) #[64,1,1,912] 912 symbols
    x_rg = rg_mapper(x) ##[64,1,1,14,76] 14*76=1064

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_data_symbols], complex
        The modulated data symbols to be mapped onto the resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols=14, fft_size=76], complex
        The full OFDM resource grid in the frequency domain.
    """
    def __init__(self, resource_grid, dtype=np.complex64, **kwargs):
        self._resource_grid = resource_grid
        """Precompute a tensor of shape
        [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        which is prefilled with pilots and stores indices
        to scatter data symbols.
        """
        self._rg_type = self._resource_grid.build_type_grid() #(1, 1, 14, 76)
        #[num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #        - 0 : Data symbol
        # - 1 : Pilot symbol
        # - 2 : Guard carrier symbol
        # - 3 : DC carrier symbol

        #Return the indices of non-zero elements in _rg_type via pytorch
        tupleindex = np.where(self._rg_type==1)#result is a tuple with first all the row indices, then all the column indices.
        self._pilot_ind=np.stack(tupleindex, axis=1) #shape=(0,4)
        #_rg_type array(1, 1, 14, 76)
        datatupleindex = np.where(self._rg_type==0) 
        #0 (all 0),1(all 0),2 (0-13),3 (0-75) tuple, (1064,) each, index for each dimension of _rg_type(1, 1, 14, 76)
        self._data_ind=np.stack(datatupleindex, axis=1) #(1064, 4)
        #self._pilot_ind = tf.where(self._rg_type==1) #shape=(0, 4)
        #self._data_ind = tf.where(self._rg_type==0) #[1064, 4]

        #test
        # test=self._rg_type.copy() #(1, 1, 14, 76)
        # data_test=test[datatupleindex]  #(1064,) 1064=14*76
        # print(data_test.shape)
        # pilot_test=test[tupleindex] #empty
        # print(pilot_test.shape)

    def __call__(self, inputs):#inputs: (64, 1, 1, 1064)
        #inputs: [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        # Map pilots on empty resource grid
        pilots = flatten_last_dims(self._resource_grid.pilot_pattern.pilots, 3) #empty

        #the indices tensor is the _pilot_ind tensor, which is a2D tensor that contains the indices of the pilot symbols in the resource grid. 
        #The values tensor is the pilots tensor, which is a1D tensor that contains the pilot symbols. 
        #The shape tensor is the _rg_type.shape tensor, which is a1D tensor that specifies the shape of the resource grid.
        ##Scatters pilots into a tensor of shape _rg_type.shape according to _pilot_ind
        # template = tf.scatter_nd(self._pilot_ind,
        #                          pilots,
        #                          self._rg_type.shape)
        template = scatter_nd_numpy(self._pilot_ind, pilots, self._rg_type.shape) #(1, 1, 14, 76) all 0 complex?

        # Expand the template to batch_size)
        # expand the last dimension for template via numpy
        template = np.expand_dims(template, axis=-1) ##[1, 1, 14, 76, 1]
        #template = tf.expand_dims(template, -1) #[1, 1, 14, 76, 1]

        # Broadcast the resource grid template to batch_size
        #batch_size = tf.shape(inputs)[0]
        batch_size = np.shape(inputs)[0]
        shapelist=list(np.shape(template)) ##[1, 1, 14, 76, 1]
        new_shape = np.concatenate([shapelist[:-1], [batch_size]], 0) #shape 5: array([ 1,  1, 14, 76, 64]
        #new_shape = tf.concat([tf.shape(template)[:-1], [batch_size]], 0) #shape 5: array([ 1,  1, 14, 76, 64]
        #template = tf.broadcast_to(template, new_shape)
        template = np.broadcast_to(template, new_shape) #(1, 1, 14, 76, 64)

        # Flatten the inputs and put batch_dim last for scatter update
        newflatten=flatten_last_dims(inputs, 3) #inputs:(64, 1, 1, 1064) =>(64, 1064)
        inputs = np.transpose(newflatten) #(1064, 64)
        #inputs = tf.transpose(newflatten)
        #The tf.tensor_scatter_nd_update function is a more efficient version of the scatter_nd function. 
        #update the resource grid with the data symbols. The input tensor is the resource grid, the values tensor is the data symbols, 
        #and the shape tensor is the _rg_type.shape tensor. The output tensor is the resource grid with the data symbols scattered in.
        #Scatter inputs into an existing template tensor according to _data_ind indices 
        #rg = tf.tensor_scatter_nd_update(template, self._data_ind, inputs)

        #Scatter inputs(1064, 64) into an existing template tensor(1, 1, 14, 76, 64) according to _data_ind indices (tuple from rg_type(1, 1, 14, 76)) 
        rg = tensor_scatter_nd_update(template, self._data_ind, inputs)
        #rg = scatter_nd_numpy(template, self._data_ind, inputs) #(1, 1, 14, 76, 64), (1064, 4), (1064, 64)
        #rg = tf.transpose(rg, [4, 0, 1, 2, 3])
        rg = np.transpose(rg, [4, 0, 1, 2, 3]) #(64, 1, 1, 14, 76)

        return rg

class OFDMModulator(): #Computes the frequency-domain representation of an OFDM waveform with cyclic prefix removal.
    def __init__(self, fft_size, l_min, cyclic_prefix_length=0) -> None:
        #(L_\text{min}) is the largest negative time lag of the discrete-time channel impulse response.
        self.fft_size = fft_size #int "`fft_size` must be positive."
        self.l_min = l_min #int "l_min must be nonpositive."
        self.cyclic_prefix_length = cyclic_prefix_length #"`cyclic_prefix_length` must be nonnegative."

    #The code calculates a phase compensation factor for an OFDM (Orthogonal Frequency Division Multiplexing) system.
    #It computes the phase shift needed to remove the timing offset introduced by the cyclic prefix in the OFDM signal.
    #The phase compensation factor is stored in self._phase_compensation.
    #The formula for the phase compensation factor is: [ \text{phase_compensation} = e^{j2\pi k L_\text{min} / N} ] 
        #where: (k) is the subcarrier index, N is fft_size
    #The code computes the phase compensation factor for all subcarriers using a range of values from 0 to self.fft_size
    def compute_phase_compensation(self):
        fft_size = self.fft_size
        l_min = self.l_min
        k_values = np.arange(fft_size, dtype=np.float32) #0-63
        tmp = -2 * np.pi * l_min / fft_size *  (64,)
        self.phase_compensation = np.exp(1j * tmp)

    #Truncation and OFDM Symbol Calculation:
    #It determines the number of elements that will be truncated from the input shape.
    #The truncation occurs due to the cyclic prefix and the FFT size.
    #The remaining elements after truncation correspond to full OFDM symbols.
    #The number of full OFDM symbols is stored in self._num_ofdm_symbols.
    def calculate_num_ofdm_symbols(self, input_shape):
        fft_size = self.fft_size
        cyclic_prefix_length = self.cyclic_prefix_length
        rest = np.mod(input_shape[-1], fft_size + cyclic_prefix_length)
        self.num_ofdm_symbols = np.floor_divide(input_shape[-1] - rest, fft_size + cyclic_prefix_length)

    def ofdm_modulator(self, inputs):
        # Shift DC subcarrier to first position
        inputs = np.fft.ifftshift(inputs, axes=-1)

        # Compute IFFT along the last dimension
        x = np.fft.ifft(inputs, axis=-1)

        # Obtain cyclic prefix
        last_dimension = np.shape(inputs)[-1]
        cp = x[..., last_dimension-self.cyclic_prefix_length:]

        # Prepend cyclic prefix
        x = np.concatenate([cp, x], axis=-1)

        # Serialize last two dimensions
        x = x.reshape(x.shape[:-2] + (-1,))

        return x

def testOFDMModulator(batch_size=64, num_elements=1024):
    # Example usage:
    input_shape = (batch_size, num_elements)  # Replace with actual input shape
    fft_size = 64  # Replace with the desired FFT size
    cyclic_prefix_length = 16  # Replace with the cyclic prefix length
    l_min = -6  # Replace with the largest negative time lag
    myofdm = OFDMModulator(fft_size=fft_size, l_min=l_min, cyclic_prefix_length=cyclic_prefix_length)
    phase_compensation = myofdm.compute_phase_compensation()
    num_ofdm_symbols = myofdm.calculate_num_ofdm_symbols(input_shape)
    print(num_ofdm_symbols)

class OFDMAMIMO():
    def __init__(self, num_rx = 1, num_tx = 1, \
                 batch_size =64, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = 4,  \
                USE_LDPC = True, pilot_pattern = "kronecker", guards = True, showfig = True) -> None:
        self.fft_size = fft_size
        self.batch_size = batch_size
        self.num_bits_per_symbol = num_bits_per_symbol
        self.showfig = showfig
        self.pilot_pattern = pilot_pattern

        # The number of transmitted streams is equal to the number of UT antennas
        # in both uplink and downlink
        #NUM_STREAMS_PER_TX = NUM_UT_ANT
        #NUM_UT_ANT = num_rx
        num_streams_per_tx = num_rx ##1
        # Create an RX-TX association matrix.
        # RX_TX_ASSOCIATION[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        # For example, considering a system with 2 RX and 4 TX, the RX-TX
        # association matrix could be
        # [ [1 , 1, 0, 0],
        #   [0 , 0, 1, 1] ]
        # which indicates that the RX 0 receives from TX 0 and 1, and RX 1 receives from
        # TX 2 and 3.
        #
        # we have only a single transmitter and receiver,
        # the RX-TX association matrix is simply:
        #RX_TX_ASSOCIATION = np.array([[1]]) #np.ones([num_rx, 1], int)
        RX_TX_ASSOCIATION = np.ones([num_rx, num_tx], int)
        self.STREAM_MANAGEMENT = StreamManagement(RX_TX_ASSOCIATION, num_streams_per_tx) #RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX

        if guards:
            cyclic_prefix_length = 6 #0 #6 Length of the cyclic prefix
            num_guard_carriers = [5,6] #[0, 0] #List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null=True #False
            pilot_ofdm_symbol_indices=[2,11]
        else:
            cyclic_prefix_length = 0 #0 #6 Length of the cyclic prefix
            num_guard_carriers = [0, 0] #List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null=False
            pilot_ofdm_symbol_indices=[0,0]
        #pilot_pattern = "kronecker" #"kronecker", "empty"
        #fft_size = 76
        #num_ofdm_symbols=14
        RESOURCE_GRID = MyResourceGrid( num_ofdm_symbols=num_ofdm_symbols,
                                            fft_size=fft_size,
                                            subcarrier_spacing=60e3, #30e3,
                                            num_tx=num_tx, #1
                                            num_streams_per_tx=num_streams_per_tx, #1
                                            cyclic_prefix_length=cyclic_prefix_length,
                                            num_guard_carriers=num_guard_carriers,
                                            dc_null=dc_null,
                                            pilot_pattern=pilot_pattern,
                                            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        self.cyclic_prefix_length = cyclic_prefix_length
        if showfig:
            RESOURCE_GRID.show() #14(OFDM symbol)*76(subcarrier) array=1064
            RESOURCE_GRID.pilot_pattern.show();
            #The pilot patterns are defined over the resource grid of *effective subcarriers* from which the nulled DC and guard carriers have been removed. 
            #This leaves us in our case with 76 - 1 (DC) - 5 (left guards) - 6 (right guards) = 64 effective subcarriers.

        if showfig and pilot_pattern == "kronecker":
            #actual pilot sequences for all streams which consists of random QPSK symbols.
            #By default, the pilot sequences are normalized, such that the average power per pilot symbol is
            #equal to one. As only every fourth pilot symbol in the sequence is used, their amplitude is scaled by a factor of two.
            plt.figure()
            plt.title("Real Part of the Pilot Sequences")
            for i in range(num_streams_per_tx):
                plt.stem(np.real(RESOURCE_GRID.pilot_pattern.pilots[0, i]),
                        markerfmt="C{}.".format(i), linefmt="C{}-".format(i),
                        label="Stream {}".format(i))
            plt.legend()
        print("Average energy per pilot symbol: {:1.2f}".format(np.mean(np.abs(RESOURCE_GRID.pilot_pattern.pilots[0,0])**2)))
        self.num_streams_per_tx = num_streams_per_tx
        self.RESOURCE_GRID = RESOURCE_GRID

        #num_bits_per_symbol = 4
        # Codeword length
        n = int(RESOURCE_GRID.num_data_symbols * num_bits_per_symbol) #num_data_symbols:64*14=896 896*4=3584, if empty 1064*4=4256

        #USE_LDPC = True
        if USE_LDPC:
            coderate = 0.5
            # Number of information bits per codeword
            k = int(n * coderate)  
            encoder = None; #LDPC5GEncoder(k, n) #1824, 3648
            decoder = None; #LDPC5GDecoder(encoder, hard_out=True)
            self.decoder = decoder
            self.encoder = encoder
        else:
            coderate = 1
            # Number of information bits per codeword
            k = int(n * coderate)  
        self.k = k # Number of information bits per codeword
        self.USE_LDPC = USE_LDPC
        self.coderate = coderate
        
        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.rg_mapper = MyResourceGridMapper(RESOURCE_GRID) #ResourceGridMapper(RESOURCE_GRID)

        #receiver part
        #self.mydemapper = MyDemapper("app", constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)
        # OFDM modulator and demodulator
        #self.modulator = OFDMModulator(self.RESOURCE_GRID.cyclic_prefix_length)
        #l_min = -6
        #self.demodulator = OFDMDemodulator(self.RESOURCE_GRID.fft_size, l_min, self.RESOURCE_GRID.cyclic_prefix_length)


    def __call__(self, b=None):
        # Transmitter
        if b is None:
            binary_source = BinarySource()
            # Start Transmitter self.k Number of information bits per codeword
            b = binary_source([self.batch_size, 1, self.num_streams_per_tx, self.k]) #[64,1,1,3584] if empty [64,1,1,1536] [batch_size, num_tx, num_streams_per_tx, num_databits]
        if self.USE_LDPC:
            c = self.encoder(b) #tf.tensor[64,1,1,3072] [batch_size, num_tx, num_streams_per_tx, num_codewords]
        else:
            c = b
        x = self.mapper(c) #np.array[64,1,1,896] if empty np.array[64,1,1,768] 768*4=3072 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x_rg = self.rg_mapper(x) ##array[64,1,1,14,76] 14*76=1064
        #output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]

        # OFDM modulation with cyclic prefix insertion
        x_time = self.modulator(x_rg) #output: [64, 1, 1, 1148]
        return x_time


if __name__ == '__main__':
    #Test basic OFDM
    # myofdm = OFDMSymbol()
    # ofdmsignal = myofdm.createOFDMsignal()
    # print(ofdmsignal)

    testOFDMModulator()

    #Test OFDMMIMO
    transmit = OFDMAMIMO(num_rx = 1, num_tx = 1, \
                batch_size =64, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = 4,  \
                USE_LDPC = False, pilot_pattern = "empty", guards=False, showfig=True) #"kronecker"
    #channeltype="perfect", "awgn", "ofdm", "time"
    x_rg = transmit(b=None)##array[64,1,1,14,76] 14*76=1064
    #output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]