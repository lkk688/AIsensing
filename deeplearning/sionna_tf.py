#sionna\mapping.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt

def insert_dims(tensor, num_dims, axis=-1):
    """Adds multiple length-one dimensions to a tensor.

    This operation is an extension to TensorFlow`s ``expand_dims`` function.
    It inserts ``num_dims`` dimensions of length one starting from the
    dimension ``axis`` of a ``tensor``. The dimension
    index follows Python indexing rules, i.e., zero-based, where a negative
    index is counted backward from the end.

    Args:
        tensor : A tensor.
        num_dims (int) : The number of dimensions to add.
        axis : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with ``num_dims`` additional
        dimensions inserted at the index specified by ``axis``.
    """
    msg = "`num_dims` must be nonnegative."
    tf.debugging.assert_greater_equal(num_dims, 0, msg)

    rank = tf.rank(tensor)
    msg = "`axis` is out of range `[-(D+1), D]`)"
    tf.debugging.assert_less_equal(axis, rank, msg)
    tf.debugging.assert_greater_equal(axis, -(rank+1), msg)

    axis = axis if axis>=0 else rank+axis+1
    shape = tf.shape(tensor)
    new_shape = tf.concat([shape[:axis],
                           tf.ones([num_dims], tf.int32),
                           shape[axis:]], 0)
    output = tf.reshape(tensor, new_shape)

    return output

def expand_to_rank(tensor, target_rank, axis=-1):
    """Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a ``tensor`` starting at
    ``axis``, so that so that the rank of the resulting tensor has rank
    ``target_rank``. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    Args:
        tensor : A tensor.
        target_rank (int) : The rank of the output tensor.
            If ``target_rank`` is smaller than the rank of ``tensor``,
            the function does nothing.
        axis (int) : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with
        ``target_rank``- rank(``tensor``) additional dimensions inserted at the
        index specified by ``axis``.
        If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.
    """
    num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    output = insert_dims(tensor, num_dims, axis)

    return output

def hard_decisions(llr):
    """Transforms LLRs into hard decisions.

    Positive values are mapped to :math:`1`.
    Nonpositive values are mapped to :math:`0`.

    Input
    -----
    llr : any non-complex tf.DType
        Tensor of LLRs.

    Output
    ------
    : Same shape and dtype as ``llr``
        The hard decisions.
    """
    zero = tf.constant(0, dtype=llr.dtype)

    return tf.cast(tf.math.greater(llr, zero), dtype=llr.dtype)


class SymbolLogits2LLRs(Layer):
    # pylint: disable=line-too-long
    r"""
    SymbolLogits2LLRs(method, num_bits_per_symbol, hard_out=False, with_prior=False, dtype=tf.float32, **kwargs)

    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points.
    If the flag ``with_prior`` is set, prior knowledge on the bits is assumed to be available.

    Parameters
    ----------
    method : One of ["app", "maxlog"], str
        The method used for computing the LLRs.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    hard_out : bool
        If `True`, the layer provides hard-decided bits instead of soft-values.
        Defaults to `False`.

    with_prior : bool
        If `True`, it is assumed that prior knowledge on the bits is available.
        This prior information is given as LLRs as an additional input to the layer.
        Defaults to `False`.

    dtype : One of [tf.float32, tf.float64] tf.DType (dtype)
        The dtype for the input and output.
        Defaults to `tf.float32`.

    Input
    -----
    logits or (logits, prior):
        Tuple:

    logits : [...,n, num_points], tf.float
        Logits on constellation points.

    prior : [num_bits_per_symbol] or [...n, num_bits_per_symbol], tf.float
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]`
        for the entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.
        Only required if the ``with_prior`` flag is set.

    Output
    ------
    : [...,n, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit.

    Note
    ----
    With the "app" method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of :math:`2^K` constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]` is the vector of logits on the constellation points, :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point and is set to :math:`\mathbf{0}` if no prior knowledge is assumed to be available,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
        = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }\right)
                .
        \end{align}
    """
    def __init__(self,
                 method,
                 num_bits_per_symbol,
                 hard_out=False,
                 with_prior=False,
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert method in ("app","maxlog"), "Unknown demapping method"
        self._method = method
        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol
        self._with_prior = with_prior
        num_points = int(2**num_bits_per_symbol) #4

        # Array composed of binary representations of all symbols indices
        a = np.zeros([num_points, num_bits_per_symbol]) #4,2, each symbol map to 2 bits, total 4 symbols
        for i in range(0, num_points):
            a[i,:] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                              dtype=np.int16)

        # Compute symbol indices for which the bits are 0 or 1
        c0 = np.zeros([int(num_points/2), num_bits_per_symbol]) #2,2
        c1 = np.zeros([int(num_points/2), num_bits_per_symbol])
        for i in range(num_bits_per_symbol-1,-1,-1):
            c0[:,i] = np.where(a[:,i]==0)[0]
            c1[:,i] = np.where(a[:,i]==1)[0]
        self._c0 = tf.constant(c0, dtype=tf.int32) # Symbols with ith bit=0
        self._c1 = tf.constant(c1, dtype=tf.int32) # Symbols with ith bit=1

        if with_prior:
            # Array of labels from {-1, 1} of all symbols
            # [num_points, num_bits_per_symbol]
            a = 2*a-1
            self._a = tf.constant(a, dtype=dtype)

        # Determine the reduce function for LLR computation
        if self._method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

    @property
    def num_bits_per_symbol(self):
        return self._num_bits_per_symbol

    def call(self, inputs):
        if self._with_prior:
            logits, prior = inputs
        else:
            logits = inputs

        # Compute exponents
        exponents = logits #(64, 512, 4) tf float32

        # Gather exponents for all bits
        # shape [...,n,num_points/2,num_bits_per_symbol]
        exp0 = tf.gather(exponents, self._c0, axis=-1, batch_dims=0) #c0 (2, 2) =>(64, 512, 2, 2)
        exp1 = tf.gather(exponents, self._c1, axis=-1, batch_dims=0) #(64, 512, 2, 2)

        # Process the prior information
        if self._with_prior:
            # Expanding `prior` such that it is broadcastable with
            # shape [..., n or 1, 1, num_bits_per_symbol]
            prior = expand_to_rank(prior, tf.rank(logits), axis=0)
            prior = tf.expand_dims(prior, axis=-2)

            # Expand the symbol labeling to be broadcastable with prior
            # shape [..., 1, num_points, num_bits_per_symbol]
            a = expand_to_rank(self._a, tf.rank(prior), axis=0)

            # Compute the prior probabilities on symbols exponents
            # shape [..., n or 1, num_points]
            exp_ps = tf.reduce_sum(tf.math.log_sigmoid(a*prior), axis=-1)

            # Gather prior probability symbol for all bits
            # shape [..., n or 1, num_points/2, num_bits_per_symbol]
            exp_ps0 = tf.gather(exp_ps, self._c0, axis=-1)
            exp_ps1 = tf.gather(exp_ps, self._c1, axis=-1)

        # Compute LLRs using the definition log( Pr(b=1)/Pr(b=0) )
        # shape [..., n, num_bits_per_symbol]
        if self._with_prior:
            llr = self._reduce(exp_ps1 + exp1, axis=-2)\
                    - self._reduce(exp_ps0 + exp0, axis=-2)
        else:
            #llr = self._reduce(exp1, axis=-2) - self._reduce(exp0, axis=-2)
            reduce1=self._reduce(exp1, axis=-2) #(64, 512, 2)
            reduce2=self._reduce(exp0, axis=-2) #(64, 512, 2)
            llr = reduce1-reduce2

        if self._hard_out:
            return hard_decisions(llr)
        else:
            return llr
        
class Demapper(Layer):
    # pylint: disable=line-too-long
    r"""
    Demapper(demapping_method, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, with_prior=False, dtype=tf.complex64, **kwargs)

    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    for a tensor of received symbols.
    If the flag ``with_prior`` is set, prior knowledge on the bits is assumed to be available.

    This class defines a layer implementing different demapping
    functions. All demapping functions are fully differentiable when soft-decisions
    are computed.

    Parameters
    ----------
    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        An instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the demapper provides hard-decided bits instead of soft-values.
        Defaults to `False`.

    with_prior : bool
        If `True`, it is assumed that prior knowledge on the bits is available.
        This prior information is given as LLRs as an additional input to the layer.
        Defaults to `False`.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of `y`. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    -----
    (y,no) or (y, prior, no) :
        Tuple:

    y : [...,n], tf.complex
        The received symbols.

    prior : [num_bits_per_symbol] or [...,num_bits_per_symbol], tf.float
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.
        Only required if the ``with_prior`` flag is set.

    no : Scalar or [...,n], tf.float
        The noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    Output
    ------
    : [...,n*num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit.

    Note
    ----
    With the "app" demapping method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point and is set to :math:`\mathbf{0}` if no prior knowledge is assumed to be available,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" demapping method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)\\
                &= \max_{c\in\mathcal{C}_{i,0}}
                    \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
                 \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
                .
        \end{align}
    """
    def __init__(self,
                 demapping_method,
                 points=None, #constellation
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 with_prior=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._with_prior = with_prior


        # Create constellation object
        # self._constellation = Constellation.create_or_check_constellation(
        #                                                 constellation_type,
        #                                                 num_bits_per_symbol,
        #                                                 constellation,
        #                                                 dtype=dtype)
        #num_bits_per_symbol = self._constellation.num_bits_per_symbol
        self.num_bits_per_symbol = num_bits_per_symbol
        self.points = points

        self._logits2llrs = SymbolLogits2LLRs(demapping_method,
                                              num_bits_per_symbol,
                                              hard_out,
                                              with_prior,
                                              dtype.real_dtype,
                                              **kwargs)

    # @property
    # def constellation(self):
    #     return self._constellation

    def call(self, inputs):
        if self._with_prior:
            y, prior, no = inputs
        else:
            y, no = inputs

        # Reshape constellation points to [1,...1,num_points]
        points_shape = [1]*y.shape.rank + self.points.shape
        points = tf.reshape(self.points, points_shape)

        # Compute squared distances from y to all points
        # shape [...,n,num_points]
        squared_dist = tf.pow(tf.abs(tf.expand_dims(y, axis=-1) - points), 2)

        # Add a dummy dimension for broadcasting. This is not needed when no
        # is a scalar, but also does not do any harm.
        no = tf.expand_dims(no, axis=-1)

        # Compute exponents
        exponents = -squared_dist/no

        if self._with_prior:
            llr = self._logits2llrs([exponents, prior])
        else:
            llr = self._logits2llrs(exponents)

        # Reshape LLRs to [...,n*num_bits_per_symbol]
        out_shape = tf.concat([tf.shape(y)[:-1],
                               [y.shape[-1] * \
                                self.constellation.num_bits_per_symbol]], 0)
        llr_reshaped = tf.reshape(llr, out_shape)

        return llr_reshaped
    
def hard_decisions(llr, datatype):
    """Transforms LLRs into hard decisions.

    Positive values are mapped to :math:`1`.
    Nonpositive values are mapped to :math:`0`.

    Input
    -----
    llr : any non-complex tf.DType
        Tensor of LLRs.

    Output
    ------
    : Same shape and dtype as ``llr``
        The hard decisions.
    """
    zero = tf.constant(0, dtype=llr.dtype)

    #return tf.cast(tf.math.greater(llr, zero), dtype=llr.dtype)
    return tf.cast(tf.math.greater(llr, zero), dtype=datatype)

from tensorflow.keras.metrics import Metric
class BitErrorRate(Metric):
    """BitErrorRate(name="bit_error_rate", **kwargs)

    Computes the average bit error rate (BER) between two binary tensors.

    This class implements a Keras metric for the bit error rate
    between two tensors of bits.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float32
            A scalar, the BER.
    """
    def __init__(self, name="bit_error_rate", **kwargs):
        super().__init__(name, **kwargs)
        self.ber = self.add_weight(name="ber",
                                   initializer="zeros",
                                   dtype=tf.float64)
        self.counter = self.add_weight(name="counter",
                                       initializer="zeros",
                                       dtype=tf.float64)

    def update_state(self, b, b_hat):
        self.counter.assign_add(1)
        self.ber.assign_add(compute_ber(b, b_hat))

    def result(self):
        #cast results of computer_ber for compatibility with tf.float32
        return tf.cast(tf.math.divide_no_nan(self.ber, self.counter),
                       dtype=tf.float32)

    def reset_state(self):
        self.ber.assign(0.0)
        self.counter.assign(0.0)

def compute_ber(b, b_hat):
    """Computes the bit error rate (BER) between two binary tensors.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float64
            A scalar, the BER.
    """
    ber = tf.not_equal(b, b_hat)
    ber = tf.cast(ber, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(ber)

def compute_ser(s, s_hat):
    """Computes the symbol error rate (SER) between two integer tensors.

    Input
    -----
        s : tf.int
            A tensor of arbitrary shape filled with integers indicating
            the symbol indices.

        s_hat : tf.int
            A tensor of the same shape as ``s`` filled with integers indicating
            the estimated symbol indices.

    Output
    ------
        : tf.float64
            A scalar, the SER.
    """
    ser = tf.not_equal(s, s_hat)
    ser = tf.cast(ser, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(ser)

def compute_bler(b, b_hat):
    """Computes the block error rate (BLER) between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i. e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float64
            A scalar, the BLER.
    """
    #first computes the boolean tensor errors by comparing b and b_hat.
    #The tf.reduce_any function is used to reduce errors over the last dimension, 
    #which returns a scalar tensor indicating whether there was at least one error in each block.
    # (W, H) array=> (W,)
    bler = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
    bler = tf.cast(bler, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(bler) #takes the mean of the BLER tensor, which returns a scalar tensor with the average BLER

def count_errors(b, b_hat, backend="numpy"):
    """Counts the number of bit errors between two binary tensors.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.int64
            A scalar, the number of bit errors.
    """
    if backend == "numpy":
        errors = np.not_equal(b, b_hat)
        errors = np.cast(errors, np.int64)
        return np.sum(errors)
    elif backend == "tf":
        errors = tf.not_equal(b,b_hat) #(64, 1024) bool find the elements that are different.
        errors = tf.cast(errors, tf.int64) #convert the boolean values to integers
        return tf.reduce_sum(errors) #add up the number of errors

def count_block_errors(b, b_hat):
    """Counts the number of block errors between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i. e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.int64
            A scalar, the number of block errors.
    """
    errors = tf.reduce_any(tf.not_equal(b,b_hat), axis=-1) #(64,)
    errors = tf.cast(errors, tf.int64)
    return tf.reduce_sum(errors)

# For the implementation of the neural receiver
NUM_BITS_PER_SYMBOL = 6
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
class NeuralDemapper(Layer): # Inherits from Keras Layer

    def __init__(self):
        super().__init__()

        # The three dense layers that form the custom trainable neural network-based demapper
        self.dense_1 = Dense(64, 'relu')
        self.dense_2 = Dense(64, 'relu')
        self.dense_3 = Dense(NUM_BITS_PER_SYMBOL, None) # The last layer has no activation and therefore outputs logits, i.e., LLRs

    def call(self, y):

        # y : complex-valued with shape [batch size, block length]
        # y is first mapped to a real-valued tensor with shape
        #  [batch size, block length, 2]
        # where the last dimension consists of the real and imaginary components
        # The dense layers operate on the last dimension, and treat the inner dimensions as batch dimensions, i.e.,
        # all the received symbols are independently processed.
        nn_input = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        z = self.dense_1(nn_input)
        z = self.dense_2(z)
        z = self.dense_3(z) # [batch size, number of symbols per block, number of bits per symbol]
        llr = tf.reshape(z, [tf.shape(y)[0], -1]) # [batch size, number of bits per block]
        return llr


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

class Constellation(Layer):
    # pylint: disable=line-too-long
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
    # pylint: enable=C0301

    def __init__(self,
                 constellation_type,
                 num_bits_per_symbol,
                 initial_value=None,
                 normalize=True,
                 center=False,
                 trainable=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(**kwargs)
        assert dtype in [tf.complex64, tf.complex128],\
            "dtype must be tf.complex64 or tf.complex128"
        self._dtype = dtype

        assert constellation_type in ("qam", "pam", "custom"),\
            "Wrong constellation type"
        self._constellation_type = constellation_type

        assert isinstance(normalize, bool), "normalize must be boolean"
        self._normalize = normalize

        assert isinstance(center, bool), "center must be boolean"
        self._center = center

        assert isinstance(trainable, bool), "trainable must be boolean"
        self._trainable = trainable

        # allow float inputs that represent int
        assert isinstance(num_bits_per_symbol, (float,int)),\
            "num_bits_per_symbol must be integer"
        assert (num_bits_per_symbol%1==0),\
            "num_bits_per_symbol must be integer"
        num_bits_per_symbol = int(num_bits_per_symbol)

        if self._constellation_type=="qam":
            assert num_bits_per_symbol%2 == 0 and num_bits_per_symbol>0,\
                "num_bits_per_symbol must be a multiple of 2"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            assert initial_value is None, "QAM must not have an initial value"
            points = qam(self._num_bits_per_symbol, normalize=self.normalize)
            points = tf.cast(points, self._dtype)

        if self._constellation_type=="pam":
            assert num_bits_per_symbol>0,\
                "num_bits_per_symbol must be integer"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            assert initial_value is None, "PAM must not have an initial value"
            points = pam(self._num_bits_per_symbol, normalize=self.normalize)
            points = tf.cast(points, self._dtype)

        if self._constellation_type=="custom":
            assert num_bits_per_symbol>0,\
                "num_bits_per_symbol must be integer"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            # Randomly initialize points if no initial_value is provided
            if initial_value is None:
                points = tf.random.uniform(  # pylint: disable=E1123
                                        [2, 2**self._num_bits_per_symbol],
                                        minval=-0.05, maxval=0.05,
                                    dtype=tf.as_dtype(self._dtype).real_dtype)
                points  = tf.complex(points[0], points[1])
            else:
                assert tf.rank(initial_value).numpy() == 1
                assert tf.shape(initial_value)[0] == 2**num_bits_per_symbol,\
                    "initial_value must have shape [2**num_bits_per_symbol]"
                points = tf.cast(initial_value, self._dtype)
        self._points = points

    def build(self, input_shape): #pylint: disable=unused-argument
        points = self._points
        points = tf.stack([tf.math.real(points),
                           tf.math.imag(points)], axis=0)
        if self._trainable:
            self._points = tf.Variable(points,
                                       trainable=self._trainable,
                                    dtype=tf.as_dtype(self._dtype).real_dtype)
        else:
            self._points = tf.constant(points,
                                    dtype=tf.as_dtype(self._dtype).real_dtype)

    # pylint: disable=no-self-argument
    def create_or_check_constellation(  constellation_type=None,
                                        num_bits_per_symbol=None,
                                        constellation=None,
                                        dtype=tf.complex64):
        # pylint: disable=line-too-long
        r"""Static method for conviently creating a constellation object or checking that an existing one
        is consistent with requested settings.

        If ``constellation`` is `None`, then this method creates a :class:`~sionna.mapping.Constellation`
        object of type ``constellation_type`` and with ``num_bits_per_symbol`` bits per symbol.
        Otherwise, this method checks that `constellation` is consistent with ``constellation_type`` and
        ``num_bits_per_symbol``. If it is, ``constellation`` is returned. Otherwise, an assertion is raised.

        Input
        ------
        constellation_type : One of ["qam", "pam", "custom"], str
            For "custom", an instance of :class:`~sionna.mapping.Constellation`
            must be provided.

        num_bits_per_symbol : int
            The number of bits per constellation symbol, e.g., 4 for QAM16.
            Only required for ``constellation_type`` in ["qam", "pam"].

        constellation :  Constellation
            An instance of :class:`~sionna.mapping.Constellation` or
            `None`. In the latter case, ``constellation_type``
            and ``num_bits_per_symbol`` must be provided.

        Output
        -------
        : :class:`~sionna.mapping.Constellation`
            A constellation object.
        """
        constellation_object = None
        if constellation is not None:
            assert constellation_type in [None, "custom"], \
                """`constellation_type` must be "custom"."""
            assert num_bits_per_symbol in \
                     [None, constellation.num_bits_per_symbol], \
                """`Wrong value of `num_bits_per_symbol.`"""
            assert constellation.dtype==dtype, \
                "Constellation has wrong dtype."
            constellation_object = constellation
        else:
            assert constellation_type in ["qam", "pam"], \
                "Wrong constellation type."
            assert num_bits_per_symbol is not None, \
                "`num_bits_per_symbol` must be provided."
            constellation_object = Constellation(   constellation_type,
                                                    num_bits_per_symbol,
                                                    dtype=dtype)
        return constellation_object

    def call(self, inputs): #pylint: disable=unused-argument
        x = self._points
        x = tf.complex(x[0], x[1])
        if self._center:
            x = x - tf.reduce_mean(x)
        if self._normalize:
            energy = tf.reduce_mean(tf.square(tf.abs(x)))
            energy_sqrt = tf.cast(tf.sqrt(energy), self._dtype)
            x = x / energy_sqrt
        return x

    @property
    def normalize(self):
        """Indicates if the constellation is normalized or not."""
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        assert isinstance(value, bool), "`normalize` must be boolean"
        self._normalize = value

    @property
    def center(self):
        """Indicates if the constellation is centered."""
        return self._center

    @center.setter
    def center(self, value):
        assert isinstance(value, bool), "`center` must be boolean"
        self._center = value

    @property
    def num_bits_per_symbol(self):
        """The number of bits per constellation symbol."""
        return self._num_bits_per_symbol

    @property
    def points(self):
        """The (possibly) centered and normalized constellation points."""
        return self(None)

    def show(self, labels=True, figsize=(7,7)):
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
        maxval = np.max(np.abs(self.points))*1.05
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.xlim(-maxval, maxval)
        plt.ylim(-maxval, maxval)
        plt.scatter(np.real(self.points), np.imag(self.points))
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.grid(True, which="both", axis="both")
        plt.title("Constellation Plot")
        if labels is True:
            for j, p in enumerate(self.points.numpy()):
                plt.annotate(
                    np.binary_repr(j, self.num_bits_per_symbol),
                    (np.real(p), np.imag(p))
                )
        return fig

class Mapper(Layer):
    # pylint: disable=line-too-long
    r"""
    Mapper(constellation_type=None, num_bits_per_symbol=None, constellation=None, return_indices=False, dtype=tf.complex64, **kwargs)

    Maps binary tensors to points of a constellation.

    This class defines a layer that maps a tensor of binary values
    to a tensor of points from a provided constellation.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    return_indices : bool
        If enabled, symbol indices are additionally returned.
        Defaults to `False`.

    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.

    Input
    -----
    : [..., n], tf.float or tf.int
        Tensor with with binary entries.

    Output
    ------
    : [...,n/Constellation.num_bits_per_symbol], tf.complex
        The mapped constellation symbols.

    : [...,n/Constellation.num_bits_per_symbol], tf.int32
        The symbol indices corresponding to the constellation symbols.
        Only returned if ``return_indices`` is set to True.


    Note
    ----
    The last input dimension must be an integer multiple of the
    number of bits per constellation symbol.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(dtype=dtype, **kwargs)
        assert dtype in [tf.complex64, tf.complex128],\
            "dtype must be tf.complex64 or tf.complex128"

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)

        self._return_indices = return_indices

        self._binary_base = 2**tf.constant(
                        range(self.constellation.num_bits_per_symbol-1,-1,-1))

    @property
    def constellation(self):
        """The Constellation used by the Mapper."""
        return self._constellation

    def call(self, inputs):
        tf.debugging.assert_greater_equal(tf.rank(inputs), 2,
            message="The input must have at least rank 2")

        # Reshape inputs to the desired format
        new_shape = [-1] + inputs.shape[1:-1].as_list() + \
           [int(inputs.shape[-1] / self.constellation.num_bits_per_symbol),
            self.constellation.num_bits_per_symbol]
        inputs_reshaped = tf.cast(tf.reshape(inputs, new_shape), tf.int32)

        # Convert the last dimension to an integer
        int_rep = tf.reduce_sum(inputs_reshaped * self._binary_base, axis=-1)

        # Map integers to constellation symbols
        x = tf.gather(self.constellation.points, int_rep, axis=0)

        if self._return_indices:
            return x, int_rep
        else:
            return x


#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
#from sionna.utils import QAMSource

class BinarySource(Layer):
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
    def __init__(self, dtype=tf.float32, seed=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        if self._seed is not None:
            self._rng = tf.random.Generator.from_seed(self._seed)

    def call(self, inputs):
        if self._seed is not None:
            return tf.cast(self._rng.uniform(inputs, 0, 2, tf.int32),
                           dtype=super().dtype)
        else:
            return tf.cast(tf.random.uniform(inputs, 0, 2, tf.int32),
                           dtype=super().dtype)

class SymbolSource(Layer):
    # pylint: disable=line-too-long
    r"""SymbolSource(constellation_type=None, num_bits_per_symbol=None, constellation=None, return_indices=False, return_bits=False, seed=None, dtype=tf.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random constellation symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

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

    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random symbols of the chosen ``constellation_type``.

    symbol_indices : ``shape``, tf.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], tf.int32
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
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(dtype=dtype, **kwargs)
        constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype)
        self._num_bits_per_symbol = constellation.num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        self._binary_source = BinarySource(seed=seed, dtype=dtype.real_dtype)
        self._mapper = Mapper(constellation=constellation,
                              return_indices=return_indices,
                              dtype=dtype)

    def call(self, inputs):
        shape = tf.concat([inputs, [self._num_bits_per_symbol]], axis=-1)
        b = self._binary_source(tf.cast(shape, tf.int32))
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)

        result = tf.squeeze(x, -1)
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            result.append(tf.squeeze(ind, -1))
        if self._return_bits:
            result.append(b)

        return result


class QAMSource(SymbolSource):
    # pylint: disable=line-too-long
    r"""QAMSource(num_bits_per_symbol=None, return_indices=False, return_bits=False, seed=None, dtype=tf.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random QAM symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

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

    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random QAM symbols.

    symbol_indices : ``shape``, tf.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], tf.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(constellation_type="qam",
                         num_bits_per_symbol=num_bits_per_symbol,
                         return_indices=return_indices,
                         return_bits=return_bits,
                         seed=seed,
                         dtype=dtype,
                         **kwargs)



class PilotPattern():
    # pylint: disable=line-too-long
    r"""Class defining a pilot pattern for an OFDM ResourceGrid.

    This class defines a pilot pattern object that is used to configure
    an OFDM :class:`~sionna.ofdm.ResourceGrid`.

    Parameters
    ----------
    mask : [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], bool
        Tensor indicating resource elements that are reserved for pilot transmissions.

    pilots : [num_tx, num_streams_per_tx, num_pilots], tf.complex
        The pilot symbols to be mapped onto the ``mask``.

    trainable : bool
        Indicates if ``pilots`` is a trainable `Variable`.
        Defaults to `False`.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension. This can be useful to
        ensure that trainable ``pilots`` have a finite energy.
        Defaults to `False`.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self, mask, pilots, trainable=False, normalize=False,
                 dtype=tf.complex64):
        super().__init__()
        self._dtype = dtype
        self._mask = tf.cast(mask, tf.int32)
        self._pilots = tf.Variable(tf.cast(pilots, self._dtype), trainable)
        self.normalize = normalize
        self._check_settings()

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
        return tf.shape(self._pilots)[-1]

    @property
    def num_data_symbols(self):
        """ Number of data symbols per transmit stream."""
        return tf.shape(self._mask)[-1]*tf.shape(self._mask)[-2] - \
               self.num_pilot_symbols

    @property
    def normalize(self):
        """Returns or sets the flag indicating if the pilots
           are normalized or not
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        self._normalize = tf.cast(value, tf.bool)

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
            scale = tf.abs(self._pilots)**2
            scale = 1/tf.sqrt(tf.reduce_mean(scale, axis=-1, keepdims=True))
            scale = tf.cast(scale, self._dtype)
            return scale*self._pilots

        return tf.cond(self.normalize, norm_pilots, lambda: self._pilots)

    @pilots.setter
    def pilots(self, value):
        self._pilots.assign(value)

    def _check_settings(self):
        """Validate that all properties define a valid pilot pattern."""

        assert tf.rank(self._mask)==4, "`mask` must have four dimensions."
        assert tf.rank(self._pilots)==3, "`pilots` must have three dimensions."
        assert np.array_equal(self._mask.shape[:2], self._pilots.shape[:2]), \
            "The first two dimensions of `mask` and `pilots` must be equal."

        num_pilots = tf.reduce_sum(self._mask, axis=(-2,-1))
        assert tf.reduce_min(num_pilots)==tf.reduce_max(num_pilots), \
            """The number of nonzero elements in the masks for all transmitters
            and streams must be identical."""

        assert self.num_pilot_symbols==tf.reduce_max(num_pilots), \
            """The shape of the last dimension of `pilots` must equal
            the number of non-zero entries within the last two
            dimensions of `mask`."""

        return True

    @property
    def trainable(self):
        """Returns if pilots are trainable or not"""
        return self._pilots.trainable


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
        mask = self.mask.numpy() #(1, 1, 14, 76)
        pilots = self.pilots.numpy() #(1, 1, 152)

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

    Generates a instance of :class:`~sionna.ofdm.PilotPattern` with
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

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self,
                 num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 dtype=tf.complex64):

        assert num_tx > 0, \
            "`num_tx` must be positive`."
        assert num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert num_effective_subcarriers > 0, \
            "`num_effective_subcarriers` must be positive`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers]
        mask = tf.zeros(shape, tf.bool)
        pilots = tf.zeros(shape[:2]+[0], dtype)
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
                 dtype=tf.complex64):

        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_effective_subcarriers = resource_grid.num_effective_subcarriers
        self._dtype = dtype

        # Number of OFDM symbols carrying pilots
        num_pilot_symbols = len(pilot_ofdm_symbol_indices)

        # Compute the total number of required orthogonal sequences
        num_seq = num_tx*num_streams_per_tx

        # Compute the length of a pilot sequence
        num_pilots = num_pilot_symbols*num_effective_subcarriers/num_seq
        assert num_pilots%1==0, \
            """`num_effective_subcarriers` must be an integer multiple of
            `num_tx`*`num_streams_per_tx`."""

        # Number of pilots per OFDM symbol
        num_pilots_per_symbol = int(num_pilots/num_pilot_symbols)

        # Prepare empty mask and pilots
        shape = [num_tx, num_streams_per_tx,
                 num_ofdm_symbols,num_effective_subcarriers]
        mask = np.zeros(shape, bool)
        shape[2] = num_pilot_symbols
        pilots = np.zeros(shape, np.complex64)

        # Populate all selected OFDM symbols in the mask
        mask[..., pilot_ofdm_symbol_indices, :] = True

        # Populate the pilots with random QPSK symbols
        qam_source = QAMSource(2, seed=seed, dtype=self._dtype)
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
        
#https://github.com/NVlabs/sionna/blob/main/sionna/ofdm/resource_grid.py
class ResourceGrid():
    # pylint: disable=line-too-long
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
            An instance of :class:`~sionna.ofdm.PilotPattern`, a string
            shorthand for the :class:`~sionna.ofdm.KroneckerPilotPattern`
            or :class:`~sionna.ofdm.EmptyPilotPattern`, or `None`.
            Defaults to `None` which is equivalent to `"empty"`.

        pilot_ofdm_symbol_indices : List, int
            List of indices of OFDM symbols reserved for pilot transmissions.
            Only needed if ``pilot_pattern="kronecker"``. Defaults to `None`.

        dtype : tf.Dtype
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.
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
                 dtype=tf.complex64):
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
        n = self._fft_size - self._dc_null - np.sum(self._num_guard_carriers)
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
        return tf.cast(n, tf.int32)

    @property
    def num_pilot_symbols(self):
        """Number of resource elements used for pilot symbols."""
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self):
        """Number of empty resource elements."""
        n = (self._fft_size-self.num_effective_subcarriers) * \
               self._num_ofdm_symbols
        return tf.cast(n, tf.int32)

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
                * self._num_ofdm_symbols

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
        assert self._dtype in [tf.complex64, tf.complex128], \
            "dtype must be tf.complex64 or tf.complex128"
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
        gc_l = 2*tf.ones(shape+[self._num_guard_carriers[0]], tf.int32) #(1, 1, 14, 0)
        gc_r = 2*tf.ones(shape+[self._num_guard_carriers[1]], tf.int32) #(1, 1, 14, 0)
        dc   = 3*tf.ones(shape + [tf.cast(self._dc_null, tf.int32)], tf.int32) #(1, 1, 14, 0)
        mask = self.pilot_pattern.mask #(1, 1, 14, 76)
        split_ind = self.dc_ind-self._num_guard_carriers[0] #38-0=38
        rg_type = tf.concat([gc_l,                 # Left Guards
                             mask[...,:split_ind], # Data & pilots
                             dc,                   # DC
                             mask[...,split_ind:], # Data & pilots
                             gc_r], -1)            # Right guards
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

class ResourceGridMapper(Layer):
    # pylint: disable=line-too-long
    r"""ResourceGridMapper(resource_grid, dtype=tf.complex64, **kwargs)

    Maps a tensor of modulated data symbols to a ResourceGrid.

    This layer takes as input a tensor of modulated data symbols
    and maps them together with pilot symbols onto an
    OFDM :class:`~sionna.ofdm.ResourceGrid`. The output can be
    converted to a time-domain signal with the
    :class:`~sionna.ofdm.Modulator` or further processed in the
    frequency domain.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_data_symbols], tf.complex
        The modulated data symbols to be mapped onto the resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        The full OFDM resource grid in the frequency domain.
    """
    def __init__(self, resource_grid, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._resource_grid = resource_grid

    def build(self, input_shape): # pylint: disable=unused-argument
        """Precompute a tensor of shape
        [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        which is prefilled with pilots and stores indices
        to scatter data symbols.
        """
        self._rg_type = self._resource_grid.build_type_grid()
        self._pilot_ind = tf.where(self._rg_type==1)
        self._data_ind = tf.where(self._rg_type==0)

    def call(self, inputs):
        # Map pilots on empty resource grid
        pilots = flatten_last_dims(self._resource_grid.pilot_pattern.pilots, 3)
        template = tf.scatter_nd(self._pilot_ind,
                                 pilots,
                                 self._rg_type.shape)
        template = tf.expand_dims(template, -1)

        # Broadcast the resource grid template to batch_size
        batch_size = tf.shape(inputs)[0]
        new_shape = tf.concat([tf.shape(template)[:-1], [batch_size]], 0)
        template = tf.broadcast_to(template, new_shape)

        # Flatten the inputs and put batch_dim last for scatter update
        inputs = tf.transpose(flatten_last_dims(inputs, 3))
        rg = tf.tensor_scatter_nd_update(template, self._data_ind, inputs)
        rg = tf.transpose(rg, [4, 0, 1, 2, 3])

        return rg

#https://github.com/NVlabs/sionna/blob/main/sionna/mimo/stream_management.py
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

from ldpc.encoding import LDPC5GEncoder
from ldpc.decoding import LDPC5GDecoder
if __name__ == '__main__':
    # Define the number of UT and BS antennas
    NUM_UT = 1
    NUM_BS = 1
    NUM_UT_ANT = 1
    NUM_BS_ANT = 1 #4

    # The number of transmitted streams is equal to the number of UT antennas
    # in both uplink and downlink
    NUM_STREAMS_PER_TX = NUM_UT_ANT

    RX_TX_ASSOCIATION = np.array([[1]])

    # Instantiate a StreamManagement object
    # This determines which data streams are determined for which receiver.
    # In this simple setup, this is fairly easy. However, it can get more involved
    # for simulations with many transmitters and receivers.
    STREAM_MANAGEMENT = StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

    RESOURCE_GRID = ResourceGrid( num_ofdm_symbols=14,
                                      fft_size=76,
                                      subcarrier_spacing=30e3,
                                      num_tx=NUM_UT,
                                      num_streams_per_tx=NUM_STREAMS_PER_TX,
                                      cyclic_prefix_length=6,
                                      pilot_pattern="kronecker",
                                      pilot_ofdm_symbol_indices=[2,11])
    RESOURCE_GRID.show();
    RESOURCE_GRID.pilot_pattern.show();

    rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=128,
                  subcarrier_spacing=15e3,
                  num_tx=1,
                  num_streams_per_tx=1,
                  cyclic_prefix_length=6,
                  num_guard_carriers=[15,16],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2])
    rg.show()

    NUM_BITS_PER_SYMBOL = 2 # QPSK
    CODERATE = 0.5

    # Number of coded bits in a resource grid
    n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL)
    # Number of information bits in a resource groud
    k = int(n*CODERATE)

    # The binary source will create batches of information bits
    binary_source = BinarySource()

    # The encoder maps information bits to coded bits
    encoder = LDPC5GEncoder(k, n)

    # The mapper maps blocks of information bits to constellation symbols
    mapper = Mapper("qam", NUM_BITS_PER_SYMBOL)

    # The resource grid mapper maps symbols onto an OFDM resource grid
    rg_mapper = ResourceGridMapper(RESOURCE_GRID)

    # Frequency domain channel
    channel = OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)

#https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html