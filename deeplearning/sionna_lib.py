#based on Tensorflow
# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
import numpy as np

# For plotting
#%matplotlib inline 
# also try %matplotlib widget
import matplotlib.pyplot as plt

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

    dtype : [tf.complex64, tf.complex128], tf.DType
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

def show(points, num_bits_per_symbol, labels=True, figsize=(7,7)):
    """Generate a scatter-plot of the constellation.

    Input
    -----
    labels : bool
        If `True`, the bit labels will be drawn next to each constellation
        point. Defaults to `True`.

    figsize : Two-element Tuple, float
        Width and height in inches. Defaults to `(7,7)`.

    Output
    ------
    : matplotlib.figure.Figure
        A handle to a matplot figure object.
    """
    maxval = np.max(np.abs(points))*1.05
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xlim(-maxval, maxval)
    plt.ylim(-maxval, maxval)
    plt.scatter(np.real(points), np.imag(points))
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True, which="both", axis="both")
    plt.title("Constellation Plot")
    if labels is True:
        for j, p in enumerate(points):
            plt.annotate(
                np.binary_repr(j, num_bits_per_symbol),
                (np.real(p), np.imag(p))
            )
    return fig

def plotcomplex(y):
    plt.figure(figsize=(8,8))
    plt.axes().set_aspect(1)
    plt.grid(True)
    plt.title('Channel output')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    #plt.scatter(tf.math.real(y), tf.math.imag(y))
    plt.scatter(np.real(y), np.imag(y))
    plt.tight_layout()

def BinarySource(shape, backend='numpy'):
    """BinarySource(dtype=tf.float32, seed=None, **kwargs)

    Layer generating random binary tensors.

    Parameters
    ----------
    dtype : tf.DType
        Defines the output datatype of the layer.
        Defaults to `tf.float32`.

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
    if backend == "numpy":
        
        return np.random.randint(2, size=shape).astype(np.float32)
    elif backend == "tf":
        return tf.cast(tf.random.uniform(shape, 0, 2, tf.int32), tf.float32)
    
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

class Mapper:
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 #dtype=tf.complex64,
                 #**kwargs
                ):
          self.num_bits_per_symbol = num_bits_per_symbol
          self.binary_base = 2**np.arange(num_bits_per_symbol-1, -1, -1, dtype=int) #array([2, 1], dtype=int32)
          self.points = CreateConstellation(constellation_type, num_bits_per_symbol) #(4,)
    
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

def ebnodb2no(ebno_db, num_bits_per_symbol, coderate):
    r"""Compute the noise variance `No` for a given `Eb/No` in dB.
    Input
    -----
    ebno_db : float
        The `Eb/No` value in dB.

    num_bits_per_symbol : int
        The number of bits per symbol.

    coderate : float
        The coderate used.

    resource_grid : ResourceGrid
        An (optional) instance of :class:`~sionna.ofdm.ResourceGrid`
        for OFDM transmissions.

    Output
    ------
    : float
        The value of :math:`N_o` in linear scale.
    """
    #ebno = tf.math.pow(tf.cast(10., dtype), ebno_db/10.)
    ebno = np.power(10, ebno_db/10.0)
    energy_per_symbol = 1
    tmp= (ebno * coderate * float(num_bits_per_symbol)) / float(energy_per_symbol)
    n0 = 1/tmp
    return n0

class MyDemapper:
    def __init__(self,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 hard_out=False,
                 with_prior=False,
                 #dtype=tf.complex64,
                 #**kwargs
                ):
        self.points = CreateConstellation(constellation_type, num_bits_per_symbol) #(4,)
        self.num_bits_per_symbol = num_bits_per_symbol
        self.with_prior = with_prior
        self.hard_out = hard_out
        self._logits2llrs = SymbolLogits2LLRs(demapping_method,
                                              num_bits_per_symbol,
                                              hard_out,
                                              with_prior)
    
    def demap(self, inputs):
        if self.with_prior:
            y, prior, no = inputs
        else:
            y, no = inputs #(64, 512)
        
        # Reshape constellation points to [1,...1,num_points]
        #points_shape = [1]*y.shape.rank + self.points.shape
        points_shape = [1]*len(y.shape) +list(self.points.shape) #[1,1]+[4] = [1, 1, 4]
        #points = tf.reshape(self.constellation.points, points_shape)
        points_reshape =np.reshape(self.points, points_shape) #(1, 1, 4)

        # Compute squared distances from y to all points
        # shape [...,n,num_points]
        #squared_dist = tf.pow(tf.abs(tf.expand_dims(y, axis=-1) - points_reshape), 2)
        ynew=np.expand_dims(y, axis=-1) #(64, 512, 1)
        #squared_dist=((ynew-points_reshape)**2) #(64, 512, 4)
        dist = np.abs(ynew-points_reshape) #(64, 512, 4) float32
        squared_dist = dist **2
        
        # Compute exponents
        exponents = -squared_dist/no

        if self.with_prior:
            llr = self._logits2llrs([exponents, prior])
        else:
            exponents = tf.convert_to_tensor(exponents)
            llr = self._logits2llrs(exponents) #(64, 512, 2)
        
        # Reshape LLRs to [...,n*num_bits_per_symbol]
        print(tf.shape(y)) #[64, 512]
        print(tf.shape(y)[:-1]) # [64]
        print(y.shape[-1]) #512
        out_shape = tf.concat([tf.shape(y)[:-1],
                               [y.shape[-1] * \
                                self.num_bits_per_symbol]], 0)
        llr_reshaped = tf.reshape(llr, out_shape) #(64, 1024)

        return llr_reshaped

def calculate_BER(bits, bits_est):
    errors = (bits != bits_est).sum()
    N = len(bits.flatten())
    BER = 1.0 * errors / N
    # error_count = torch.sum(bits_est != bits.flatten()).float()  # Count of unequal bits
    # error_rate = error_count / bits_est.numel()  # Error rate calculation
    # BER = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
    return BER

class UncodedSystemAWGN:
    def __init__(self, num_bits_per_symbol, BATCH_SIZE=64, Blocklength = 1024, constellation_type="qam", demapping_method="app", data_type=np.complex64):
        self.points = CreateConstellation(constellation_type, num_bits_per_symbol)
        print(self.points.shape) #(4,) complex64
        self.shape = ([BATCH_SIZE, Blocklength])# Blocklength [64, 1024]
        self.constellation_type = constellation_type
        self.num_bits_per_symbol = num_bits_per_symbol
        self.data_type = data_type
        self.demapper = MyDemapper(demapping_method, constellation_type=constellation_type, num_bits_per_symbol=num_bits_per_symbol)
        self.mapper=Mapper(constellation_type=constellation_type, num_bits_per_symbol=num_bits_per_symbol)
    
    def process(self, ebno_db=10.0):
        bits = BinarySource(self.shape)
        print("Shape of bits: ", bits.shape) #(64, 1024)
    
        
        x=self.mapper.create_symbol(bits) #(64, 512) complex64

        n0=ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=1.0) #scalar 0.05

        noise=complex_normal(x.shape, 1.0) #(64, 512) complex128
        #print(noise.dtype)
        noise = noise.astype(self.data_type)
        noise *= np.sqrt(n0) #tf.cast(tf.sqrt(n0), noise.dtype)
        y=x+noise #(64, 512)

        llr = self.demapper.demap([y, n0])
        #print("Shape of llr: ", llr.shape) #(64, 1024)
        b_hat = hard_decisions(llr, tf.int32) #(64, 1024) 0,1
        ber = calculate_BER(bits, b_hat.numpy())
        print(ber)
        return ber
    
    def simulation(self, EBN0_DB_MIN=0, EBN0_DB_MAX=20):
        ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
        BER=[]
        for enbo in ebno_dbs:
            ber=self.process(enbo)
            BER.append(ber)
        
        plt.plot(ebno_dbs, BER, 'bo', ebno_dbs, BER, 'k')
        plt.axis([0, 10, 1e-6, 0.1])
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.grid(True)
        plt.title('BPSK Modulation')
        plt.show()

class CodedSystemAWGN:
    def __init__(self, num_bits_per_symbol, BATCH_SIZE=64, n = 1024, constellation_type="qam", demapping_method="app", coderate=0.5, data_type=np.complex64):
        self.points = CreateConstellation(constellation_type, num_bits_per_symbol)
        print(self.points.shape) #(4,) complex64
        self.constellation_type = constellation_type
        self.num_bits_per_symbol = num_bits_per_symbol
        self.data_type = data_type
        self.mapper = Mapper(constellation_type=constellation_type, num_bits_per_symbol=num_bits_per_symbol)
        self.demapper = MyDemapper(demapping_method, constellation_type=constellation_type, num_bits_per_symbol=num_bits_per_symbol)
        self.n = n
        self.k = int(n*coderate)
        self.coderate = coderate
        self.shape = ([BATCH_SIZE, self.k])# Blocklength [64, 1024]
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
    
    def process(self, ebno_db=10.0, test_flag=True):
        bits = BinarySource(self.shape)
        print("Shape of bits: ", bits.shape) #(64, 1024)
        codewords = self.encoder(bits)#numpy to tf.tensor float32
    
        x=self.mapper.create_symbol(codewords) #(64, 512) complex64

        n0=ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=1.0) #scalar 0.05

        noise=complex_normal(x.shape, 1.0) #(64, 512) complex128
        #print(noise.dtype)
        noise = noise.astype(self.data_type)
        noise *= np.sqrt(n0) #tf.cast(tf.sqrt(n0), noise.dtype)
        y=x+noise #(64, 512)

        llr = self.demapper.demap([y, n0])
        #print("Shape of llr: ", llr.shape) #(64, 1024)
        #b_hat = hard_decisions(llr, tf.int32) #(64, 1024) 0,1

        b_hat = self.decoder(llr)
        if test_flag == True:
            num_samples = 8 # how many samples shall be printed
            num_symbols = int(num_samples/self.num_bits_per_symbol)
            print(f"First {num_samples} original bits: {bits[0,:num_samples]}")
            print(f"First {num_samples} transmitted symbol: {x[0,:num_samples]}")
            print(f"First {num_samples} demapped llrs: {np.round(llr[0,:num_samples], 2)}") #positive is 1, negative is 0
            print(f"First {num_samples} decoded bits: {b_hat[0,:num_samples]}")
        
        ber = calculate_BER(bits, b_hat.numpy())
        print(ber)
        return ber
    
    def simulation(self, EBN0_DB_MIN=-20, EBN0_DB_MAX=20):
        ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
        BER=[]
        for enbo in ebno_dbs:
            ber=self.process(enbo, test_flag=False)
            BER.append(ber)
        
        plt.plot(ebno_dbs, BER, 'bo', ebno_dbs, BER, 'k')
        #plt.axis([0, 10, 1e-6, 0.1])
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.grid(True)
        plt.title('BPSK Modulation')
        plt.show()
        return ebno_dbs, BER

from sionna_tf import Demapper, SymbolLogits2LLRs, hard_decisions, count_errors, count_block_errors

def test():
    data_type = np.complex64# Complex64 number (real and imaginary parts are float32)
    NUM_BITS_PER_SYMBOL = 2 # QPSK
    #constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    points = CreateConstellation("qam", NUM_BITS_PER_SYMBOL)
    print(points.shape) #(4,) complex64

    show(points, NUM_BITS_PER_SYMBOL)

    BATCH_SIZE = 64 # How many examples are processed in parallel
    Blocklength = 1024
    shape = ([BATCH_SIZE, Blocklength])# Blocklength [64, 1024]
    bits = BinarySource(shape)
    print("Shape of bits: ", bits.shape) #(64, 1024)

    mapper=Mapper(constellation_type="qam", num_bits_per_symbol=2)
    x=mapper.create_symbol(bits) #(64, 512) complex64

    n0=ebnodb2no(ebno_db=10.0, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=1.0) #scalar 0.05

    noise=complex_normal(x.shape, 1.0) #(64, 512) complex128
    print(noise.dtype)
    noise = noise.astype(data_type)
    noise *= np.sqrt(n0) #tf.cast(tf.sqrt(n0), noise.dtype)
    y=x+noise #(64, 512)

    # The demapper uses the same constellation object as the mapper
    #demapper = Demapper("app", num_bits_per_symbol=NUM_BITS_PER_SYMBOL, points=points) 
    demapper = MyDemapper("app", constellation_type="qam", num_bits_per_symbol=NUM_BITS_PER_SYMBOL )
    llr = demapper.demap([y, n0])

    #y_tensor = tf.convert_to_tensor(y)
    #llr = demapper([y, n0])
    print("Shape of llr: ", llr.shape) #(64, 1024)

    num_samples = 8 # how many samples shall be printed
    num_symbols = int(num_samples/NUM_BITS_PER_SYMBOL)
    print(f"First {num_samples} transmitted bits: {bits[0,:num_samples]}")
    print(f"First {num_symbols} transmitted symbols: {np.round(x[0,:num_symbols], 2)}")
    print(f"First {num_symbols} received symbols: {np.round(y[0,:num_symbols], 2)}")
    print(f"First {num_samples} demapped llrs: {np.round(llr[0,:num_samples], 2)}") #positive is 1, negative is 0

    plotcomplex(y)
    b_hat = hard_decisions(llr, tf.int32) #(64, 1024) 0,1


    ber = calculate_BER(bits, b_hat.numpy())
    print(ber)

    # count errors
    b = tf.convert_to_tensor(bits)  #(64, 1024)
    bit_e = count_errors(b, b_hat)
    print(bit_e)
    block_e = count_block_errors(b, b_hat)
    print(block_e)

    llr_np = llr.numpy() 

from ldpc.encoding import LDPC5GEncoder
from ldpc.decoding import LDPC5GDecoder
def testencoding():
    #k = 12
    #n = 20
    BATCH_SIZE = 10 # samples per scenario
    num_basestations = 4 
    num_users = 5 # users per basestation
    n = 1000 # codeword length per transmitted codeword
    coderate = 0.5 # coderate
    k = int(coderate * n) # number of info bits per codeword

    # instantiate a new encoder for codewords of length n
    encoder = LDPC5GEncoder(k, n)

    decoder = LDPC5GDecoder(encoder, hard_out=True)
    # the decoder must be linked to the encoder (to know the exact code parameters used for encoding)
    # decoder = LDPC5GDecoder(encoder, hard_out=True, # binary output or provide soft-estimates
    #                                 return_infobits=True, # or also return (decoded) parity bits
    #                                 num_iter=20, # number of decoding iterations
    #                                 cn_type="boxplus-phi") # also try "minsum" decoding

    #BATCH_SIZE = 1 # one codeword in parallel
    #u = binary_source([BATCH_SIZE, k])
    #u = BinarySource([BATCH_SIZE, k], backend='tf')
    # draw random bits to encode
    u = BinarySource([BATCH_SIZE, num_basestations, num_users, k], backend='tf')
    print("Shape of u: ", u.shape) #(10, 4, 5, 500)
    #print("Input bits are: \n", u.numpy())

    # We can immediately encode u for all users, basetation and samples 
    # This all happens with a single line of code
    c = encoder(u)
    print("Shape of c: ", c.shape) #(10, 4, 5, 1000)
    #print("Encoded bits are: \n", c.numpy())

    print("Total number of processed bits: ", np.prod(c.shape)) #200000

    d = decoder(c)
    print("Shape of d: ", d.shape) #(10, 4, 5, 500)

    ber = calculate_BER(u.numpy(), d.numpy())
    print(ber)

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class ComplexDataset(Dataset):
    def __init__(self, num_bits_per_symbol, BATCH_SIZE=64, Blocklength = 1024, DB_MIN=-10, DB_MAX=20, totaldbs=2000, constellation_type="qam", data_type=np.complex64):
        self.points = CreateConstellation(constellation_type, num_bits_per_symbol)
        print(self.points.shape) #(4,) complex64
        self.shape = ([BATCH_SIZE, Blocklength])# Blocklength [64, 1024]
        self.constellation_type = constellation_type
        self.num_bits_per_symbol = num_bits_per_symbol
        self.data_type = data_type
        self.mapper=Mapper(constellation_type=constellation_type, num_bits_per_symbol=num_bits_per_symbol)

        ebno_dbs=np.linspace(DB_MIN, DB_MAX, totaldbs)
        np.random.shuffle(ebno_dbs)
        self.ebno_dbs = ebno_dbs
    
    def __getitem__(self, index):
        ebno_db = self.ebno_dbs[index]

        bits = BinarySource(self.shape)
        print("Shape of bits: ", bits.shape) #(64, 1024)

        x=self.mapper.create_symbol(bits) #(64, 512) complex64

        n0=ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=1.0) #scalar 0.05
        noise=complex_normal(x.shape, 1.0) #(64, 512) complex128
        #print(noise.dtype)
        noise = noise.astype(self.data_type)
        noise *= np.sqrt(n0) 
        y=x+noise #(64, 512)
        signal_complex = torch.from_numpy(y)
        return signal_complex
    
    def __len__(self):
        return len(self.ebno_dbs)

def test_dataset():
    NUM_BITS_PER_SYMBOL = 2
    BATCH_SIZE = 64
    Blocklength = 1024
    dataset = ComplexDataset(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, BATCH_SIZE=BATCH_SIZE, Blocklength=Blocklength, DB_MIN=-10, DB_MAX=20, totaldbs=2000)
    onesample = dataset[0] #
    # train, validation and test split
    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size
    train_set, val_set= torch.utils.data.random_split(dataset, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

    onebatch = next(iter(train_loader))
    print(onebatch.shape)#torch.Size([64, 64, 512])

    #print the size and shape of the data
    print('Train set size:', len(train_set))
    print('Validation set size:', len(val_set))
    print('Test set size:', len(dataset) - len(train_set) - len(val_set))

    #print the first 10 elements of the train set
    print('First 10 elements of the train set:')
    for i in range(10):
        print(train_set[i])

if __name__ == '__main__':

    test_dataset()

    testencoding()

    #simulate = UncodedSystemAWGN(num_bits_per_symbol=2)
    simulate = CodedSystemAWGN(num_bits_per_symbol=2, BATCH_SIZE=2000, n = 2048)
    ber = simulate.process(ebno_db=-10)
    ebno_dbs, BER = simulate.simulation()
    print(ebno_dbs)
    print(BER)
