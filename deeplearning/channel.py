import tensorflow as tf
from tensorflow.keras.layers import Layer
from abc import ABC, abstractmethod
import numpy as np

PI = 3.141592653589793
SPEED_OF_LIGHT = 299792458

from sionna_tf import config
def matrix_sqrt_inv(tensor):
    r""" Computes the inverse square root of a Hermitian matrix.

    Given a batch of Hermitian positive definite matrices
    :math:`\mathbf{A}`, with square root matrices :math:`\mathbf{B}`,
    such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`, the function
    returns :math:`\mathbf{B}^{-1}`, such that
    :math:`\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as ``tensor`` containing
        the inverse matrix square root of its last two dimensions.

    Note:
        If you want to use this function in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.Config.xla_compat=true``.
        See :py:attr:`~sionna.Config.xla_compat`.
    """
    if config.xla_compat and not tf.executing_eagerly():
        s, u = tf.linalg.eigh(tensor)

        # Compute 1/sqrt of eigenvalues
        s = tf.abs(s)
        tf.debugging.assert_positive(s, "Input must be positive definite.")
        s = 1/tf.sqrt(s)
        s = tf.cast(s, u.dtype)

        # Matrix multiplication
        s = tf.expand_dims(s, -2)
        return tf.matmul(u*s, u, adjoint_b=True)
    else:
        return tf.linalg.inv(tf.linalg.sqrtm(tensor))
    
def whiten_channel(y, h, s, return_s=True):
    # pylint: disable=line-too-long
    r"""Whitens a canonical MIMO channel.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M(\mathbb{R}^M)` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K(\mathbb{R}^K)` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}(\mathbb{R}^{M\times K})` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M(\mathbb{R}^M)` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}(\mathbb{R}^{M\times M})`.

    This function whitens this channel by multiplying :math:`\mathbf{y}` and
    :math:`\mathbf{H}` from the left by :math:`\mathbf{S}^{-\frac{1}{2}}`.
    Optionally, the whitened noise covariance matrix :math:`\mathbf{I}_M`
    can be returned.

    Input
    -----
    y : [...,M], tf.float or tf.complex
        1+D tensor containing the received signals.

    h : [...,M,K], tf.float or tf.complex
        2+D tensor containing the  channel matrices.

    s : [...,M,M], tf.float or complex
        2+D tensor containing the noise covariance matrices.

    return_s : bool
        If `True`, the whitened covariance matrix is returned.
        Defaults to `True`.

    Output
    ------
    : [...,M], tf.float or tf.complex
        1+D tensor containing the whitened received signals.

    : [...,M,K], tf.float or tf.complex
        2+D tensor containing the whitened channel matrices.

    : [...,M,M], tf.float or tf.complex
        2+D tensor containing the whitened noise covariance matrices.
        Only returned if ``return_s`` is `True`.
    """
    # Compute whitening matrix
    s_inv_1_2 = matrix_sqrt_inv(s)
    s_inv_1_2 = expand_to_rank(s_inv_1_2, tf.rank(h), 0)

    # Whiten obervation and channel matrix
    yw = tf.expand_dims(y, -1)
    yw = tf.matmul(s_inv_1_2, yw)
    yw = tf.squeeze(yw, axis=-1)

    hw = tf.matmul(s_inv_1_2, h)

    if return_s:
        # Ideal interference covariance matrix after whitening
        sw = tf.eye(tf.shape(s)[-2], dtype=s.dtype)
        sw = expand_to_rank(sw, tf.rank(s), 0)
        return yw, hw, sw
    else:
        return yw, hw


# An ABC (Abstract Base Class) is a class that cannot be instantiated, but can be used as a base class for other classes. 
#This allows you to define common functionality that can be inherited by multiple classes.
from _py_abc import ABCMeta
class ABC(metaclass=ABCMeta):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()

class ChannelModel(ABC):
    # pylint: disable=line-too-long
    r"""ChannelModel()

    Abstract class that defines an interface for channel models.

    Any channel model which generates channel impulse responses must implement this interface.
    All the channel models available in Sionna, such as :class:`~sionna.channel.RayleighBlockFading` or :class:`~sionna.channel.tr38901.TDL`, implement this interface.

    *Remark:* Some channel models only require a subset of the input parameters.

    Input
    -----

    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    @abstractmethod
    def __call__(self,  batch_size, num_time_steps, sampling_frequency):

        return NotImplemented


def complex_normal(shape, var=1.0, dtype=tf.complex64):
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
    var_dim = tf.cast(var, dtype.real_dtype)/tf.cast(2, dtype.real_dtype)
    stddev = tf.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    xr = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    xi = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    x = tf.complex(xr, xi)

    return x

class AWGN(Layer):
    r"""AWGN(dtype=tf.complex64, **kwargs)

    Add complex AWGN to the inputs with a certain variance.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    This layer adds complex AWGN noise with variance ``no`` to the input.
    The noise has variance ``no/2`` per real dimension.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    Example
    --------

    Setting-up:

    >>> awgn_channel = AWGN()

    Running:

    >>> # x is the channel input
    >>> # no is the noise variance
    >>> y = awgn_channel((x, no))

    Parameters
    ----------
        dtype : Complex tf.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----

        (x, no) :
            Tuple:

        x :  Tensor, tf.complex
            Channel input

        no : Scalar or Tensor, tf.float
            Scalar or tensor whose shape can be broadcast to the shape of ``x``.
            The noise power ``no`` is per complex dimension. If ``no`` is a
            scalar, noise of the same variance will be added to the input.
            If ``no`` is a tensor, it must have a shape that can be broadcast to
            the shape of ``x``. This allows, e.g., adding noise of different
            variance to each example in a batch. If ``no`` has a lower rank than
            ``x``, then ``no`` will be broadcast to the shape of ``x`` by adding
            dummy dimensions after the last axis.

    Output
    -------
        y : Tensor with same shape as ``x``, tf.complex
            Channel output
    """

    def __init__(self, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._real_dtype = tf.dtypes.as_dtype(self._dtype).real_dtype

    def call(self, inputs):

        x, no = inputs

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(tf.shape(x), dtype=x.dtype)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, tf.rank(x), axis=-1)

        # Apply variance scaling
        no = tf.cast(no, self._real_dtype)
        noise *= tf.cast(tf.sqrt(no), noise.dtype)

        # Add noise to input
        y = x + noise

        return y


class ApplyOFDMChannel(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""ApplyOFDMChannel(add_awgn=True, dtype=tf.complex64, **kwargs)

    Apply single-tap channel frequency responses to channel inputs.

    This class inherits from the Keras `Layer` class and can be used as layer
    in a Keras model.

    For each OFDM symbol :math:`s` and subcarrier :math:`n`, the single-tap channel
    is applied as follows:

    .. math::
        y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}

    where :math:`y_{s,n}` is the channel output computed by this layer,
    :math:`\widehat{h}_{s, n}` the frequency channel response (``h_freq``),
    :math:`x_{s,n}` the channel input ``x``, and :math:`w_{s,n}` the additive noise.

    For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
    of each receiver and by summing over all the antennas of all transmitters.

    Parameters
    ----------

    add_awgn : bool
        If set to `False`, no white Gaussian noise is added.
        Defaults to `True`.

    dtype : tf.DType
        Complex datatype to use for internal processing and output. Defaults to
        `tf.complex64`.

    Input
    -----

    (x, h_freq, no) or (x, h_freq):
        Tuple:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel inputs

    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel frequency responses

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs:
        [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
        Only required if ``add_awgn`` is set to `True`.
        The noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the outputs.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of the channel outputs. This allows, e.g., adding noise of
        different variance to each example in a batch. If ``no`` has a lower
        rank than the channel outputs, then ``no`` will be broadcast to the
        shape of the channel outputs by adding dummy dimensions after the
        last axis.

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    """

    def __init__(self, add_awgn=True, dtype=tf.complex64, **kwargs):

        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self._add_awgn = add_awgn

    def build(self, input_shape): #pylint: disable=unused-argument

        if self._add_awgn:
            self._awgn = AWGN(dtype=self.dtype)

    def call(self, inputs):

        if self._add_awgn:
            x, h_freq, no = inputs #x: [64, 1, 16, 14, 76], h_freq: [64, 1, 1, 1, 16, 1, 76]
        else:
            x, h_freq = inputs 

        # Apply the channel response
        x = expand_to_rank(x, h_freq.shape.rank, axis=1) #[64, 1(added), 1(added), 1, 1, 14, 76]
        y = tf.reduce_sum(tf.reduce_sum(h_freq*x, axis=4), axis=3) #[64, 1, 1, 14, 76] (3,4 removed)

        # Add AWGN if requested
        if self._add_awgn:
            y = self._awgn((y, no))

        return y


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


def cir_to_ofdm_channel(frequencies, a, tau, normalize=False):
    # pylint: disable=line-too-long
    r"""
    Compute the frequency response of the channel at ``frequencies``.

    Given a channel impulse response
    :math:`(a_{m}, \tau_{m}), 0 \leq m \leq M-1` (inputs ``a`` and ``tau``),
    the channel frequency response for the frequency :math:`f`
    is computed as follows:

    .. math::
        \widehat{h}(f) = \sum_{m=0}^{M-1} a_{m} e^{-j2\pi f \tau_{m}}

    Input
    ------
    frequencies : [fft_size], tf.float
        Frequencies at which to compute the channel response

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float
        Path delays

    normalize : bool
        If set to `True`, the channel is normalized over the resource grid
        to ensure unit average energy per resource element. Defaults to `False`.

    Output
    -------
    h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex
        Channel frequency responses at ``frequencies``
    """

    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        # Broadcast is not supported yet by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1])

    # Add a time samples dimension for broadcasting
    tau = tf.expand_dims(tau, axis=6)

    # Bring all tensors to broadcastable shapes
    tau = tf.expand_dims(tau, axis=-1)
    h = tf.expand_dims(a, axis=-1)
    frequencies = expand_to_rank(frequencies, tf.rank(tau), axis=0)

    ## Compute the Fourier transforms of all cluster taps
    # Exponential component
    e = tf.exp(tf.complex(tf.constant(0, real_dtype),
        -2*PI*frequencies*tau))
    h_f = h*e
    # Sum over all clusters to get the channel frequency responses
    h_f = tf.reduce_sum(h_f, axis=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and
        # subcarriers.
        c = tf.reduce_mean( tf.square(tf.abs(h_f)), axis=(2,4,5,6),
                            keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        h_f = tf.math.divide_no_nan(h_f, c)

    return h_f

def subcarrier_frequencies(num_subcarriers, subcarrier_spacing,
                           dtype=tf.complex64):
    # pylint: disable=line-too-long
    r"""
    Compute the baseband frequencies of ``num_subcarrier`` subcarriers spaced by
    ``subcarrier_spacing``, i.e.,

    >>> # If num_subcarrier is even:
    >>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
    >>>
    >>> # If num_subcarrier is odd:
    >>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing


    Input
    ------
    num_subcarriers : int
        Number of subcarriers

    subcarrier_spacing : float
        Subcarrier spacing [Hz]

    dtype : tf.DType
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `tf.complex64` (`tf.float32`).

    Output
    ------
        frequencies : [``num_subcarrier``], tf.float
            Baseband frequencies of subcarriers
    """

    if dtype.is_complex:
        real_dtype = dtype.real_dtype
    elif dtype.if_floating:
        real_dtype = dtype
    else:
        raise AssertionError("dtype must be a complex or floating datatype")

    if tf.equal(tf.math.floormod(num_subcarriers, 2), 0):
        start=-num_subcarriers/2
        limit=num_subcarriers/2
    else:
        start=-(num_subcarriers-1)/2
        limit=(num_subcarriers-1)/2+1

    frequencies = tf.range( start=start,
                            limit=limit,
                            dtype=real_dtype)
    frequencies = frequencies*subcarrier_spacing
    return frequencies

def cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=False):
    # pylint: disable=line-too-long
    r"""
    Compute the channel taps forming the discrete complex-baseband
    representation of the channel from the channel impulse response
    (``a``, ``tau``).

    This function assumes that a sinc filter is used for pulse shaping and receive
    filtering. Therefore, given a channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, the channel taps
    are computed as follows:

    .. math::
        \bar{h}_{b, \ell}
        = \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
            \text{sinc}\left( \ell - W\tau_{m} \right)

    for :math:`\ell` ranging from ``l_min`` to ``l_max``, and where :math:`W` is
    the ``bandwidth``.

    Input
    ------
    bandwidth : float
        Bandwidth [Hz]

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float
        Path delays [s]

    l_min : int
        Smallest time-lag for the discrete complex baseband channel (:math:`L_{\text{min}}`)

    l_max : int
        Largest time-lag for the discrete complex baseband channel (:math:`L_{\text{max}}`)

    normalize : bool
        If set to `True`, the channel is normalized over the block size
        to ensure unit average energy per time step. Defaults to `False`.

    Output
    -------
    hm :  [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], tf.complex
        Channel taps coefficients
    """

    real_dtype = tau.dtype #(64, 1, 1, 10)

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4) #[64, 1, num_rx_antenna=1(added), 1, num_tx_antenna=1(added), 10]
        # Broadcast is not supported by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1]) #(64, 1, 1, 1, 16(added num_tx_ant), 10)

    # Add a time samples dimension for broadcasting
    tau = tf.expand_dims(tau, axis=6) #[64, 1, 1, 1, 16, 10, 1(time)]

    # Time lags for which to compute the channel taps
    l = tf.range(l_min, l_max+1, dtype=real_dtype) #(27,)

    # Bring tau and l to broadcastable shapes
    tau = tf.expand_dims(tau, axis=-1) #(64, 1, 1, 1, 16, 10, 1, 1(added for time lag))
    l = expand_to_rank(l, tau.shape.rank, axis=0) #[1, 1, 1, 1, 1, 1, 1, 27]

    # sinc pulse shaping
    g = tf.experimental.numpy.sinc(l-tau*bandwidth)#[64, 1, 1, 1, 16, 10, 1, 27]
    g = tf.complex(g, tf.constant(0., real_dtype))
    a = tf.expand_dims(a, axis=-1) #[64, 1, 1, 1, 16, 10, 1, 1(added for time lag)]

    # For every tap, sum the sinc-weighted coefficients
    hm = tf.reduce_sum(a*g, axis=-3) #[64, 1, 1, 1, 16, 10(pathes reduced), 1, 27]=>[64, 1, 1, 1, 16, 1, 27]

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per block is one.
        # The total energy of a channel response is the sum of the squared
        # norm over the channel taps.
        # Average over block size, RX antennas, and TX antennas
        c = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(hm)),
                                         axis=6, keepdims=True), #sum for time
                           axis=(2,4,5), keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        hm = tf.math.divide_no_nan(hm, c) #c:[64, 1, 1, 1, 1, 1, 1]

    return hm #[64, 1, 1, 1, 16, 1, 27]

def time_lag_discrete_time_channel(bandwidth, maximum_delay_spread=3e-6):
    # pylint: disable=line-too-long
    r"""
    Compute the smallest and largest time-lag for the descrete complex baseband
    channel, i.e., :math:`L_{\text{min}}` and :math:`L_{\text{max}}`.

    The smallest time-lag (:math:`L_{\text{min}}`) returned is always -6, as this value
    was found small enough for all models included in Sionna.

    The largest time-lag (:math:`L_{\text{max}}`) is computed from the ``bandwidth``
    and ``maximum_delay_spread`` as follows:

    .. math::
        L_{\text{max}} = \lceil W \tau_{\text{max}} \rceil + 6

    where :math:`L_{\text{max}}` is the largest time-lag, :math:`W` the ``bandwidth``,
    and :math:`\tau_{\text{max}}` the ``maximum_delay_spread``.

    The default value for the ``maximum_delay_spread`` is 3us, which was found
    to be large enough to include most significant paths with all channel models
    included in Sionna assuming a nominal delay spread of 100ns.

    Note
    ----
    The values of :math:`L_{\text{min}}` and :math:`L_{\text{max}}` computed
    by this function are only recommended values.
    :math:`L_{\text{min}}` and :math:`L_{\text{max}}` should be set according to
    the considered channel model. For OFDM systems, one also needs to be careful
    that the effective length of the complex baseband channel is not larger than
    the cyclic prefix length.

    Input
    ------
    bandwidth : float
        Bandwith (:math:`W`) [Hz]

    maximum_delay_spread : float
        Maximum delay spread [s]. Defaults to 3us.

    Output
    -------
    l_min : int
        Smallest time-lag (:math:`L_{\text{min}}`) for the descrete complex baseband
        channel. Set to -6, , as this value was found small enough for all models
        included in Sionna.

    l_max : int
        Largest time-lag (:math:`L_{\text{max}}`) for the descrete complex baseband
        channel
    """
    l_min = tf.cast(-6, tf.int32)
    l_max = tf.math.ceil(maximum_delay_spread*bandwidth) + 6
    l_max = tf.cast(l_max, tf.int32)
    return l_min, l_max #-6, 20


def time_to_ofdm_channel(h_t, rg, l_min):
    # pylint: disable=line-too-long
    r"""
    Compute the channel frequency response from the discrete complex-baseband
    channel impulse response.

    Given a discrete complex-baseband channel impulse response
    :math:`\bar{h}_{b,\ell}`, for :math:`\ell` ranging from :math:`L_\text{min}\le 0`
    to :math:`L_\text{max}`, the discrete channel frequency response is computed as

    .. math::

        \hat{h}_{b,n} = \sum_{k=0}^{L_\text{max}} \bar{h}_{b,k} e^{-j \frac{2\pi kn}{N}} + \sum_{k=L_\text{min}}^{-1} \bar{h}_{b,k} e^{-j \frac{2\pi n(N+k)}{N}}, \quad n=0,\dots,N-1

    where :math:`N` is the FFT size and :math:`b` is the time step.

    This function only produces one channel frequency response per OFDM symbol, i.e.,
    only values of :math:`b` corresponding to the start of an OFDM symbol (after
    cyclic prefix removal) are considered.

    Input
    ------
    h_t : [...num_time_steps,l_max-l_min+1], tf.complex
        Tensor of discrete complex-baseband channel impulse responses

    resource_grid : :class:`~sionna.ofdm.ResourceGrid`
        Resource grid

    l_min : int
        Smallest time-lag for the discrete complex baseband
        channel impulse response (:math:`L_{\text{min}}`)

    Output
    ------
    h_f : [...,num_ofdm_symbols,fft_size], tf.complex
        Tensor of discrete complex-baseband channel frequency responses

    Note
    ----
    Note that the result of this function is generally different from the
    output of :meth:`~sionna.channel.utils.cir_to_ofdm_channel` because
    the discrete complex-baseband channel impulse response is truncated
    (see :meth:`~sionna.channel.utils.cir_to_time_channel`). This effect
    can be observed in the example below.

    Examples
    --------
    .. code-block:: Python

        # Setup resource grid and channel model
        tf.random.set_seed(4)
        sm = StreamManagement(np.array([[1]]), 1)
        rg = ResourceGrid(num_ofdm_symbols=1,
                          fft_size=1024,
                          subcarrier_spacing=15e3)
        tdl = TDL("A", 100e-9, 3.5e9)

        # Generate CIR
        cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)

        # Generate OFDM channel from CIR
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = tf.squeeze(cir_to_ofdm_channel(frequencies, *cir, normalize=True))

        # Generate time channel from CIR
        l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
        h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)

        # Generate OFDM channel from time channel
        h_freq_hat = tf.squeeze(time_to_ofdm_channel(h_time, rg, l_min))

        # Visualize results
        plt.figure()
        plt.plot(np.real(h_freq), "-")
        plt.plot(np.real(h_freq_hat), "--")
        plt.plot(np.imag(h_freq), "-")
        plt.plot(np.imag(h_freq_hat), "--")
        plt.xlabel("Subcarrier index")
        plt.ylabel(r"Channel frequency response")
        plt.legend(["OFDM Channel (real)", "OFDM Channel from time (real)", "OFDM Channel (imag)", "OFDM Channel from time (imag)"])

    .. image:: ../figures/time_to_ofdm_channel.png
    """
    # Totla length of an OFDM symbol including cyclic prefix
    ofdm_length = rg.fft_size + rg.cyclic_prefix_length

    # Downsample the impulse respons to one sample per OFDM symbol
    h_t = h_t[...,rg.cyclic_prefix_length:rg.num_time_samples:ofdm_length, :]

    # Pad channel impulse response with zeros to the FFT size
    pad_dims = rg.fft_size - tf.shape(h_t)[-1]
    pad_shape = tf.concat([tf.shape(h_t)[:-1], [pad_dims]], axis=-1)
    h_t = tf.concat([h_t, tf.zeros(pad_shape, dtype=h_t.dtype)], axis=-1)

    # Circular shift of negative time lags so that the channel impulse reponse
    # starts with h_{b,0}
    h_t = tf.roll(h_t, l_min, axis=-1)

    # Compute FFT
    h_f = tf.signal.fft(h_t)

    # Move the zero subcarrier to the center of the spectrum
    h_f = tf.signal.fftshift(h_f, axes=-1)

    return h_f


def deg_2_rad(x):
    r"""
    Convert degree to radian

    Input
    ------
        x : Tensor
            Angles in degree

    Output
    -------
        y : Tensor
            Angles ``x`` converted to radian
    """
    return x*tf.constant(PI/180.0, x.dtype)

def rad_2_deg(x):
    r"""
    Convert radian to degree

    Input
    ------
        x : Tensor
            Angles in radian

    Output
    -------
        y : Tensor
            Angles ``x`` converted to degree
    """
    return x*tf.constant(180.0/PI, x.dtype)

#ref: https://github.com/NVlabs/sionna/blob/main/sionna/channel/apply_time_channel.py
import scipy
class ApplyTimeChannel(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""ApplyTimeChannel(num_time_samples, l_tot, add_awgn=True, dtype=tf.complex64, **kwargs)

    Apply time domain channel responses ``h_time`` to channel inputs ``x``,
    by filtering the channel inputs with time-variant channel responses.

    This class inherits from the Keras `Layer` class and can be used as layer
    in a Keras model.

    For each batch example, ``num_time_samples`` + ``l_tot`` - 1 time steps of a
    channel realization are required to filter the channel inputs.

    The channel output consists of ``num_time_samples`` + ``l_tot`` - 1
    time samples, as it is the result of filtering the channel input of length
    ``num_time_samples`` with the time-variant channel filter  of length
    ``l_tot``. In the case of a single-input single-output link and given a sequence of channel
    inputs :math:`x_0,\cdots,x_{N_B}`, where :math:`N_B` is ``num_time_samples``, this
    layer outputs

    .. math::
        y_b = \sum_{\ell = 0}^{L_{\text{tot}}} x_{b-\ell} \bar{h}_{b,\ell} + w_b

    where :math:`L_{\text{tot}}` corresponds ``l_tot``, :math:`w_b` to the additive noise, and
    :math:`\bar{h}_{b,\ell}` to the :math:`\ell^{th}` tap of the :math:`b^{th}` channel sample.
    This layer outputs :math:`y_b` for :math:`b` ranging from 0 to
    :math:`N_B + L_{\text{tot}} - 1`, and :math:`x_{b}` is set to 0 for :math:`b \geq N_B`.

    For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
    of each receiver and by summing over all the antennas of all transmitters.

    Parameters
    ----------

    num_time_samples : int
        Number of time samples forming the channel input (:math:`N_B`)

    l_tot : int
        Length of the channel filter (:math:`L_{\text{tot}} = L_{\text{max}} - L_{\text{min}} + 1`)

    add_awgn : bool
        If set to `False`, no white Gaussian noise is added.
        Defaults to `True`.

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.

    Input
    -----

    (x, h_time, no) or (x, h_time):
        Tuple:

    x :  [batch size, num_tx, num_tx_ant, num_time_samples], tf.complex
        Channel inputs

    h_time : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1, l_tot], tf.complex
        Channel responses.
        For each batch example, ``num_time_samples`` + ``l_tot`` - 1 time steps of a
        channel realization are required to filter the channel inputs.

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the channel outputs: [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1].
        Only required if ``add_awgn`` is set to `True`.
        The noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the outputs.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of the channel outputs. This allows, e.g., adding noise of
        different variance to each example in a batch. If ``no`` has a lower
        rank than the channel outputs, then ``no`` will be broadcast to the
        shape of the channel outputs by adding dummy dimensions after the
        last axis.

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1], tf.complex
        Channel outputs.
        The channel output consists of ``num_time_samples`` + ``l_tot`` - 1
        time samples, as it is the result of filtering the channel input of length
        ``num_time_samples`` with the time-variant channel filter  of length
        ``l_tot``.
    """

    def __init__(self, num_time_samples, l_tot, add_awgn=True,
                 dtype=tf.complex64, **kwargs):

        #super().__init__(trainable=False, dtype=dtype, **kwargs)
        super().__init__(trainable=False, **kwargs)

        self._add_awgn = add_awgn
        #self.currenttype = dtype

        # The channel transfert function is implemented by first gathering from
        # the vector of transmitted baseband symbols
        # x = [x_0,...,x_{num_time_samples-1}]^T  the symbols that are then
        # multiplied by the channel tap coefficients.
        # We build here the matrix of indices G, with size
        # `num_time_samples + l_tot - 1` x `l_tot` that is used to perform this
        # gathering.
        # For example, if there are 4 channel taps
        # h = [h_0, h_1, h_2, h_3]^T
        # and `num_time_samples` = 10 time steps then G  would be
        #       [[0, 10, 10, 10]
        #        [1,  0, 10, 10]
        #        [2,  1,  0, 10]
        #        [3,  2,  1,  0]
        #        [4,  3,  2,  1]
        #        [5,  4,  3,  2]
        #        [6,  5,  4,  3]
        #        [7,  6,  5,  4]
        #        [8,  7,  6,  5]
        #        [9,  8,  7,  6]
        #        [10, 9,  8,  7]
        #        [10,10,  9,  8]
        #        [10,10, 10,  9]
        # Note that G is a Toeplitz matrix.
        # In this example, the index `num_time_samples`=10 corresponds to the
        # zero symbol. The vector of transmitted symbols is padded with one
        # zero at the end.
        first_colum = np.concatenate([  np.arange(0, num_time_samples),
                                        np.full([l_tot-1], num_time_samples)]) #(1090,)
        #The first_column (1D array) is created by concatenating two arrays: 
            #The first part generates an array of integers from 0 to num_time_samples - 1
            #the second part creates an array of length l_tot - 1 filled with the value num_time_samples

        first_row = np.concatenate([[0], np.full([l_tot-1], num_time_samples)]) #1D array (27,)
            #The first part is [0], which is a single-element array containing just the value 0.
            #The second part is np.full([l_tot-1], num_time_samples), which is the same as in the first_column.

        #Toeplitz Matrix Creation
        self._g = scipy.linalg.toeplitz(first_colum, first_row)
        #A Toeplitz matrix has constant diagonals, where the first column (first_column) 
        #serves as the diagonal and the first row (first_row) serves as the first row of the matrix.
        #The resulting matrix _g is a square matrix with dimensions (l_tot, l_tot).

    def build(self, input_shape): #pylint: disable=unused-argument

        if self._add_awgn:
            self._awgn = AWGN(dtype=self.dtype)
            #self._awgn = AWGN(dtype=self.currenttype)

    def call(self, inputs):

        if self._add_awgn:
            x, h_time, no = inputs
        else:
            x, h_time = inputs
    
        #x :  [batch size, num_tx, num_tx_ant, num_time_samples]
        # Preparing the channel input for broadcasting and matrix multiplication
        x = tf.pad(x, [[0,0], [0,0], [0,0], [0,1]]) #(64, 1, 1, 1065)
        #The first dimension [0,0] indicates no padding along the batch size dimension.
        #The second dimension [0,0] indicates no padding along the num_tx dimension.
        #The third dimension [0,0] indicates no padding along the num_tx_ant dimension.
        #The fourth dimension [0,1] adds one zero-padding element along the num_time_samples dimension.
        #The purpose of this padding is to ensure that the dimensions of x are compatible for broadcasting and matrix multiplication.

        #It inserts ``num_dims`` dimensions of length one starting from the dimension ``axis``
        #output = insert_dims(tensor, num_dims, axis)
        x = insert_dims(x, 2, axis=1) #inserts a new dimension into the x tensor (64, 1, 1, 1065)
        #The inserted dimension is placed at index 2 (third dimension) along the axis 1.

        #tf.gather(params, indices, axis) is a TensorFlow function that slices the input tensor params along the specified axis using the indices provided in the tensor indices
        #gather slices from the input tensor x based on indices specified by the matrix self._g along the last axis (axis=-1).
        x = tf.gather(x, self._g, axis=-1) #(64, 1, 1, 1, 1, 1090, 27)
        #self._g is a matrix of shape [num_time_samples, num_time_samples], the operation will gather slices from x along the last axis (i.e., num_time_samples) using the indices specified by self._g.
        #The resulting tensor will have the same shape as x, but with values rearranged based on the indices in self._g
        #x :  [batch size, num_tx, num_tx_ant, num_time_samples, l_tot]

        # Apply the channel response
        #h_time : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1, l_tot]
        #The first line computes the element-wise product (*) between h_time and another tensor x. The result is a new tensor y.
        #The tf.reduce_sum(y, axis=-1) computes the sum along the last axis (axis with index -1). 
        #This reduces the dimensionality of y by one, resulting in a tensor with shape [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1].
        y = tf.reduce_sum(h_time*x, axis=-1) #(64, 1, 1, 1, 16, 1090)

        #The inner tf.reduce_sum(y, axis=4) computes the sum along the fourth axis (assuming y has the shape described above)
        #The result of the inner reduction will have shape [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1].
        y = tf.reduce_sum(tf.reduce_sum(y, axis=4), axis=3) #(64, 1, 1, 1090) float32
        #The outer tf.reduce_sum(..., axis=3) computes the sum along the third axis of the intermediate result.
        #The final result is assigned back to the variable y.
        #The resulting tensor y will have shape [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1].

        # Add AWGN if requested
        if self._add_awgn:
            y = self._awgn((y, no))

        return y


class GenerateOFDMChannel:
    # pylint: disable=line-too-long
    r"""GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=False)

    Generate channel frequency responses.
    The channel impulse response is constant over the duration of an OFDM symbol.

    Given a channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, generated by the ``channel_model``,
    the channel frequency response for the :math:`s^{th}` OFDM symbol and
    :math:`n^{th}` subcarrier is computed as follows:

    .. math::
        \widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}

    where :math:`\Delta_f` is the subcarrier spacing, and :math:`s` is used as time
    step to indicate that the channel impulse response can change from one OFDM symbol to the
    next in the event of mobility, even if it is assumed static over the duration
    of an OFDM symbol.

    Parameters
    ----------
    channel_model : :class:`~sionna.channel.ChannelModel` object
        An instance of a :class:`~sionna.channel.ChannelModel` object, such as
        :class:`~sionna.channel.RayleighBlockFading` or
        :class:`~sionna.channel.tr38901.UMi`.

    resource_grid : :class:`~sionna.ofdm.ResourceGrid`
        Resource grid

    normalize_channel : bool
        If set to `True`, the channel is normalized over the resource grid
        to ensure unit average energy per resource element. Defaults to `False`.

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.

    Input
    -----

    batch_size : int
        Batch size. Defaults to `None` for channel models that do not require this paranmeter.

    Output
    -------
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
        Channel frequency responses
    """

    def __init__(self, channel_model, resource_grid, normalize_channel=False,
                 dtype=tf.complex64):

        # Callable used to sample channel input responses
        self._cir_sampler = channel_model

        # We need those in call()
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols #14
        self._subcarrier_spacing = resource_grid.subcarrier_spacing #60000
        self._num_subcarriers = resource_grid.fft_size #76
        self._normalize_channel = normalize_channel #True
        self._sampling_frequency = 1./resource_grid.ofdm_symbol_duration #60000

        # Frequencies of the subcarriers
        self._frequencies = subcarrier_frequencies(self._num_subcarriers,
                                                   self._subcarrier_spacing,
                                                   dtype) #[76]

    def __call__(self, batch_size=None):

        # Sample channel impulse responses
        h, tau = self._cir_sampler( batch_size,
                                    self._num_ofdm_symbols, #14
                                    self._sampling_frequency)#60000
        #h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10]
        h_freq = cir_to_ofdm_channel(self._frequencies, h, tau,
                                     self._normalize_channel)
        #Channel frequency responses at ``frequencies`` 
        return h_freq #[64, 1, 1, 1, 16, 1, 76]
    
class OFDMChannel(Layer):
    # pylint: disable=line-too-long
    r"""OFDMChannel(channel_model, resource_grid, add_awgn=True, normalize_channel=False, return_channel=False, dtype=tf.complex64, **kwargs)

    Generate channel frequency responses and apply them to channel inputs
    assuming an OFDM waveform with no ICI nor ISI.

    This class inherits from the Keras `Layer` class and can be used as layer
    in a Keras model.

    For each OFDM symbol :math:`s` and subcarrier :math:`n`, the channel output is computed as follows:

    .. math::
        y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}

    where :math:`y_{s,n}` is the channel output computed by this layer,
    :math:`\widehat{h}_{s, n}` the frequency channel response,
    :math:`x_{s,n}` the channel input ``x``, and :math:`w_{s,n}` the additive noise.

    For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
    of each receiver and by summing over all the antennas of all transmitters.

    The channel frequency response for the :math:`s^{th}` OFDM symbol and
    :math:`n^{th}` subcarrier is computed from a given channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1` generated by the ``channel_model``
    as follows:

    .. math::
        \widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}

    where :math:`\Delta_f` is the subcarrier spacing, and :math:`s` is used as time
    step to indicate that the channel impulse response can change from one OFDM symbol to the
    next in the event of mobility, even if it is assumed static over the duration
    of an OFDM symbol.

    Parameters
    ----------
    channel_model : :class:`~sionna.channel.ChannelModel` object
        An instance of a :class:`~sionna.channel.ChannelModel` object, such as
        :class:`~sionna.channel.RayleighBlockFading` or
        :class:`~sionna.channel.tr38901.UMi`.

    resource_grid : :class:`~sionna.ofdm.ResourceGrid`
        Resource grid

    add_awgn : bool
        If set to `False`, no white Gaussian noise is added.
        Defaults to `True`.

    normalize_channel : bool
        If set to `True`, the channel is normalized over the resource grid
        to ensure unit average energy per resource element. Defaults to `False`.

    return_channel : bool
        If set to `True`, the channel response is returned in addition to the
        channel output. Defaults to `False`.

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to tf.complex64.

    Input
    -----

    (x, no) or x:
        Tuple or Tensor:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel inputs

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs:
        [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
        Only required if ``add_awgn`` is set to `True`.
        The noise power ``no`` is per complex dimension. If ``no`` is a scalar,
        noise of the same variance will be added to the outputs.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of the channel outputs. This allows, e.g., adding noise of
        different variance to each example in a batch. If ``no`` has a lower
        rank than the channel outputs, then ``no`` will be broadcast to the
        shape of the channel outputs by adding dummy dimensions after the last
        axis.

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        (Optional) Channel frequency responses. Returned only if
        ``return_channel`` is set to `True`.
    """

    def __init__(self, channel_model, resource_grid, add_awgn=True,
                normalize_channel=False, return_channel=False,
                dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self._cir_sampler = channel_model
        self._rg = resource_grid
        self._add_awgn = add_awgn
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel

    def build(self, input_shape): #pylint: disable=unused-argument

        self._generate_channel = GenerateOFDMChannel(self._cir_sampler,
                                                     self._rg,
                                                     self._normalize_channel,
                                                     tf.as_dtype(self.dtype))
        self._apply_channel = ApplyOFDMChannel( self._add_awgn,
                                                tf.as_dtype(self.dtype))

    def call(self, inputs):

        if self._add_awgn:
            x, no = inputs
        else:
            x = inputs

        h_freq = self._generate_channel(tf.shape(x)[0])
        if self._add_awgn:
            y = self._apply_channel([x, h_freq, no])
        else:
            y = self._apply_channel([x, h_freq])

        if self._return_channel:
            return y, h_freq
        else:
            return y

from sionna_tf import ResourceGrid, MyRemoveNulledSubcarriers, RemoveNulledSubcarriers
class BaseChannelInterpolator(ABC):
    # pylint: disable=line-too-long
    r"""BaseChannelInterpolator()

    Abstract layer for implementing an OFDM channel interpolator.

    Any layer that implements an OFDM channel interpolator must implement this
    callable class.

    A channel interpolator is used by an OFDM channel estimator
    (:class:`~sionna.ofdm.BaseChannelEstimator`) to compute channel estimates
    for the data-carrying resource elements from the channel estimates for the
    pilot-carrying resource elements.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    @abstractmethod
    def __call__(self, h_hat, err_var):
        pass


class NearestNeighborInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""NearestNeighborInterpolator(pilot_pattern)

    Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided channel estimates and error
    variances according to the nearest neighbor method. It is assumed
    that the measurements were taken at the nonzero positions of a
    :class:`~sionna.ofdm.PilotPattern`.

    The figure below shows how four channel estimates are interpolated
    accross a resource grid. Grey fields indicate measurement positions
    while the colored regions show which resource elements are assigned
    to the same measurement value.

    .. image:: ../figures/nearest_neighbor_interpolation.png

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols>0),\
            """The pilot pattern cannot be empty"""

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)
        mask_shape = mask.shape # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots)==0, -1))
        assert max_num_zero_pilots<pilots.shape[-1],\
            """Each pilot sequence must have at least one nonzero entry"""

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int32)
        for a in range(gather_ind.shape[0]): # For each pilot pattern...
            i_p, j_p = np.where(mask[a]) # ...determine the pilot indices

            for i in range(mask_shape[-2]): # Iterate over...
                for j in range(mask_shape[-1]): # ... all resource elements

                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i-i_p) + np.abs(j-j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a])==0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance...
                    ind = np.argmin(d)

                    # ... and store it in the index tensor
                    gather_ind[a, i, j] = ind

        # Reshape to the original shape of the mask, i.e.:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers]
        self._gather_ind = tf.reshape(gather_ind, mask_shape) #(1, 2, 14, 64)

    def _interpolate(self, inputs): #h_ls: (2, 1, 16, 1, 2, 128), err_var: (1, 1, 1, 1, 2, 128)
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_pilots, k, l, m]
        perm = tf.roll(tf.range(tf.rank(inputs)), -3, 0) #[3, 4, 5, 0, 1, 2]
        inputs = tf.transpose(inputs, perm) #(1, 2, 128, 2, 1, 16)

        # Interpolate through gather. Shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  ..., num_effective_subcarriers, k, l, m]
        outputs = tf.gather(inputs, self._gather_ind, 2, batch_dims=2) #inputs: (1, 2, 128, 2, 1, 16) _gather_ind: (1, 2, 14, 64)
        #outputs: (1, 2, 14, 64, 2, 1, 16)
        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0) #[4, 5, 6, 0, 1, 2, 3]
        outputs = tf.transpose(outputs, perm)

        return outputs #(2, 1, 16, 1, 2, 14, 64)

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var

class LinearInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""LinearInterpolator(pilot_pattern, time_avg=False)

    Linear channel estimate interpolation on a resource grid.

    This class computes for each element of an OFDM resource grid
    a channel estimate based on ``num_pilots`` provided channel estimates and
    error variances through linear interpolation.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.ofdm.PilotPattern`.

    The interpolation is done first across sub-carriers and then
    across OFDM symbols.

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    time_avg : bool
        If enabled, measurements will be averaged across OFDM symbols
        (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame. Defaults to `False`.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern, time_avg=False):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols>0),\
            """The pilot pattern cannot be empty"""

        self._time_avg = time_avg

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask) #(1, 2, 14, 64)
        mask_shape = mask.shape # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:])) #(2, 14, 64)

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots #(1, 2, 128)
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]]) #(2, 128)

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots)==0, -1)) #64
        assert max_num_zero_pilots<pilots.shape[-1],\
            """Each pilot sequence must have at least one nonzero entry"""

        # Create actual pilot patterns for each stream over the resource grid
        z = np.zeros_like(mask, dtype=pilots.dtype)
        for a in range(z.shape[0]):
            z[a][np.where(mask[a])] = pilots[a]

        # Linear interpolation works as follows:
        # We compute for each resource element (RE)
        # x_0 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the first channel measurement was taken
        # x_1 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the second channel measurement was taken
        # y_0 : The first channel estimate
        # y_1 : The second channel estimate
        # x   : The x-value (i.e., sub-carrier index or OFDM symbol)
        #
        # The linearly interpolated value y is then given as:
        # y = (x-x_0) * (y_1-y_0) / (x_1-x_0) + y_0
        #
        # The following code pre-computes various quantities and indices
        # that are needed to compute x_0, x_1, y_0, y_1, x for frequency- and
        # time-domain interpolation.

        ##
        ## Frequency-domain interpolation
        ##
        self._x_freq = tf.cast(expand_to_rank(tf.range(0, mask.shape[-1]),
                                              7,
                                              axis=0),
                               pilots.dtype)

        # Permutation indices to shift batch_dims last during gather
        self._perm_fwd_freq = tf.roll(tf.range(6), -3, 0)

        x_0_freq = np.zeros_like(mask, np.int32)
        x_1_freq = np.zeros_like(mask, np.int32)

        # Set REs of OFDM symbols without any pilot equal to -1 (dummy value)
        x_0_freq[np.sum(np.abs(z), axis=-1)==0] = -1
        x_1_freq[np.sum(np.abs(z), axis=-1)==0] = -1

        y_0_freq_ind = np.copy(x_0_freq) # Indices used to gather estimates
        y_1_freq_ind = np.copy(x_1_freq) # Indices used to gather estimates

        # For each stream
        for a in range(z.shape[0]):

            pilot_count = 0 # Counts the number of non-zero pilots

            # Indices of non-zero pilots within the pilots vector
            pilot_ind = np.where(np.abs(pilots[a]))[0]

            # Go through all OFDM symbols
            for i in range(x_0_freq.shape[1]):

                # Indices of non-zero pilots within the OFDM symbol
                pilot_ind_ofdm = np.where(np.abs(z[a][i]))[0]

                # If OFDM symbol contains only one non-zero pilot
                if len(pilot_ind_ofdm)==1:
                    # Set the indices of the first and second pilot to the same
                    # value for all REs of the OFDM symbol
                    x_0_freq[a][i] = pilot_ind_ofdm[0]
                    x_1_freq[a][i] = pilot_ind_ofdm[0]
                    y_0_freq_ind[a,i] = pilot_ind[pilot_count]
                    y_1_freq_ind[a,i] = pilot_ind[pilot_count]

                # If OFDM symbol contains two or more pilots
                elif len(pilot_ind_ofdm)>=2:
                    x0 = 0
                    x1 = 1

                    # Go through all resource elements of this OFDM symbol
                    for j in range(x_0_freq.shape[2]):
                        x_0_freq[a,i,j] = pilot_ind_ofdm[x0]
                        x_1_freq[a,i,j] = pilot_ind_ofdm[x1]
                        y_0_freq_ind[a,i,j] = pilot_ind[pilot_count + x0]
                        y_1_freq_ind[a,i,j] = pilot_ind[pilot_count + x1]
                        if j==pilot_ind_ofdm[x1] and x1<len(pilot_ind_ofdm)-1:
                            x0 = x1
                            x1 += 1

                pilot_count += len(pilot_ind_ofdm)

        x_0_freq = np.reshape(x_0_freq, mask_shape)
        x_1_freq = np.reshape(x_1_freq, mask_shape)
        x_0_freq = expand_to_rank(x_0_freq, 7, axis=0)
        x_1_freq = expand_to_rank(x_1_freq, 7, axis=0)
        self._x_0_freq = tf.cast(x_0_freq, pilots.dtype)
        self._x_1_freq = tf.cast(x_1_freq, pilots.dtype)

        # We add +1 here to shift all indices as the input will be padded
        # at the beginning with 0, (i.e., the dummy index -1 will become 0).
        self._y_0_freq_ind = np.reshape(y_0_freq_ind, mask_shape)+1
        self._y_1_freq_ind = np.reshape(y_1_freq_ind, mask_shape)+1

        ##
        ## Time-domain interpolation
        ##
        self._x_time = tf.expand_dims(tf.range(0, mask.shape[-2]), -1)
        self._x_time = tf.cast(expand_to_rank(self._x_time, 7, axis=0),
                               dtype=pilots.dtype)

        # Indices used to gather estimates
        self._perm_fwd_time = tf.roll(tf.range(7), -3, 0)

        y_0_time_ind = np.zeros(z.shape[:2], np.int32) # Gather indices
        y_1_time_ind = np.zeros(z.shape[:2], np.int32) # Gather indices

        # For each stream
        for a in range(z.shape[0]):

            # Indices of OFDM symbols for which channel estimates were computed
            ofdm_ind = np.where(np.sum(np.abs(z[a]), axis=-1))[0]

            # Only one OFDM symbol with pilots
            if len(ofdm_ind)==1:
                y_0_time_ind[a] = ofdm_ind[0]
                y_1_time_ind[a] = ofdm_ind[0]

            # Two or more OFDM symbols with pilots
            elif len(ofdm_ind)>=2:
                x0 = 0
                x1 = 1
                for i in range(z.shape[1]):
                    y_0_time_ind[a,i] = ofdm_ind[x0]
                    y_1_time_ind[a,i] = ofdm_ind[x1]
                    if i==ofdm_ind[x1] and x1<len(ofdm_ind)-1:
                        x0 = x1
                        x1 += 1

        self._y_0_time_ind = np.reshape(y_0_time_ind, mask_shape[:-1])
        self._y_1_time_ind = np.reshape(y_1_time_ind, mask_shape[:-1])

        self._x_0_time = expand_to_rank(tf.expand_dims(self._y_0_time_ind, -1),
                                                       7, axis=0)
        self._x_0_time = tf.cast(self._x_0_time, dtype=pilots.dtype)
        self._x_1_time = expand_to_rank(tf.expand_dims(self._y_1_time_ind, -1),
                                                       7, axis=0)
        self._x_1_time = tf.cast(self._x_1_time, dtype=pilots.dtype)

        #
        # Other precomputed values
        #
        # Undo permutation of batch_dims for gather
        self._perm_bwd = tf.roll(tf.range(7), 3, 0)

        # Padding for the inputs
        pad = np.zeros([6, 2], np.int32)
        pad[-1, 0] = 1
        self._pad = pad

        # Number of ofdm symbols carrying at least one pilot.
        # Used for time-averaging (optional)
        n = np.sum(np.abs(np.reshape(z, mask_shape)), axis=-1, keepdims=True)
        n = np.sum(n>0, axis=-2, keepdims=True)
        self._num_pilot_ofdm_symbols = expand_to_rank(n, 7, axis=0)


    def _interpolate_1d(self, inputs, x, x0, x1, y0_ind, y1_ind):
        # Gather the right values for y0 and y1
        y0 = tf.gather(inputs, y0_ind, axis=2, batch_dims=2)
        y1 = tf.gather(inputs, y1_ind, axis=2, batch_dims=2)

        # Undo the permutation of the inputs
        y0 = tf.transpose(y0, self._perm_bwd)
        y1 = tf.transpose(y1, self._perm_bwd)

        # Compute linear interpolation
        slope = tf.math.divide_no_nan(y1-y0, tf.cast(x1-x0, dtype=y0.dtype))
        return tf.cast(x-x0, dtype=y0.dtype)*slope + y0

    def _interpolate(self, inputs):
        #
        # Prepare inputs
        #
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Pad the inputs with a leading 0.
        # All undefined channel estimates will get this value.
        inputs = tf.pad(inputs, self._pad, constant_values=0)

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, 1+num_pilots, k, l, m]
        inputs = tf.transpose(inputs, self._perm_fwd_freq)

        #
        # Frequency-domain interpolation
        #
        # h_hat_freq has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers]
        h_hat_freq = self._interpolate_1d(inputs,
                                          self._x_freq,
                                          self._x_0_freq,
                                          self._x_1_freq,
                                          self._y_0_freq_ind,
                                          self._y_1_freq_ind)
        #
        # Time-domain interpolation
        #

        # Time-domain averaging (optional)
        if self._time_avg:
            num_ofdm_symbols = h_hat_freq.shape[-2]
            h_hat_freq = tf.reduce_sum(h_hat_freq, axis=-2, keepdims=True)
            h_hat_freq /= tf.cast(self._num_pilot_ofdm_symbols,h_hat_freq.dtype)
            h_hat_freq = tf.repeat(h_hat_freq, [num_ofdm_symbols], axis=-2)

        # Transpose h_hat_freq to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers, k, l, m]
        h_hat_time = tf.transpose(h_hat_freq, self._perm_fwd_time)

        # h_hat_time has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers]
        h_hat_time = self._interpolate_1d(h_hat_time,
                                          self._x_time,
                                          self._x_0_time,
                                          self._x_1_time,
                                          self._y_0_time_ind,
                                          self._y_1_time_ind)

        return h_hat_time

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)

        # the interpolator requires complex-valued inputs
        err_var = tf.cast(err_var, tf.complex64)
        err_var = self._interpolate(err_var)
        err_var = tf.math.real(err_var)

        return h_hat, err_var
    
class BaseChannelInterpolator(ABC):
    # pylint: disable=line-too-long
    r"""BaseChannelInterpolator()

    Abstract layer for implementing an OFDM channel interpolator.

    Any layer that implements an OFDM channel interpolator must implement this
    callable class.

    A channel interpolator is used by an OFDM channel estimator
    (:class:`~sionna.ofdm.BaseChannelEstimator`) to compute channel estimates
    for the data-carrying resource elements from the channel estimates for the
    pilot-carrying resource elements.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    @abstractmethod
    def __call__(self, h_hat, err_var):
        pass

from sionna_tf import flatten_last_dims
class BaseChannelEstimator(ABC, Layer):
    # pylint: disable=line-too-long
    r"""BaseChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Abstract layer for implementing an OFDM channel estimator.

    Any layer that implements an OFDM channel estimator must implement this
    class and its
    :meth:`~sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    abstract method.

    This class extracts the pilots from the received resource grid ``y``, calls
    the :meth:`~sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    method to estimate the channel for the pilot-carrying resource elements,
    and then interpolates the channel to compute channel estimates for the
    data-carrying resouce elements using the interpolation method specified by
    ``interpolation_type`` or the ``interpolator`` object.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    interpolation_type : One of ["nn", "lin", "lin_time_avg"], string
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).
        Defaults to "nn".

    interpolator : BaseChannelInterpolator
        An instance of :class:`~sionna.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specfied
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        # assert isinstance(resource_grid, ResourceGrid),\
        #     "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern
        self._removed_nulled_scs = RemoveNulledSubcarriers(resource_grid)

        assert interpolation_type in ["nn","lin","lin_time_avg",None], \
            "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type

        if interpolator is not None:
            assert isinstance(interpolator, BaseChannelInterpolator), \
        "`interpolator` must implement the BaseChannelInterpolator interface"
            self._interpol = interpolator
        elif self._interpolation_type == "nn":
            self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin":
            self._interpol = LinearInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin_time_avg":
            self._interpol = LinearInterpolator(self._pilot_pattern,
                                                time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        mask = flatten_last_dims(self._pilot_pattern.mask)
        pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING")
        self._pilot_ind = pilot_ind[...,:num_pilot_symbols]

    @abstractmethod
    def estimate_at_pilot_locations(self, y_pilots, no):
        """
        Estimates the channel for the pilot-carrying resource elements.

        This is an abstract method that must be implemented by a concrete
        OFDM channel estimator that implement this class.

        Input
        -----
        y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex
            Observed signals for the pilot-carrying resource elements

        no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
            Variance of the AWGN

        Output
        ------
        h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex
            Channel estimates for the pilot-carrying resource elements

        err_var : Same shape as ``h_hat``, tf.float
            Channel estimation error variance for the pilot-carrying
            resource elements
        """
        pass

    def call(self, inputs):

        y, no = inputs

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Removed nulled subcarriers (guards, dc)
        y_eff = self._removed_nulled_scs(y)

        # Flatten the resource grid for pilot extraction
        # New shape: [...,num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)

        # Gather pilots along the last dimensions
        # Resulting shape: y_eff_flat.shape[:-1] + pilot_ind.shape, i.e.:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_pilot_symbols]
        y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)

        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_hat, err_var = self._interpol(h_hat, err_var)
            err_var = tf.maximum(err_var, tf.cast(0, err_var.dtype))

        return h_hat, err_var


class LSChannelEstimator(BaseChannelEstimator, Layer):
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized observation,
    where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated accross
    the entire resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    interpolation_type : One of ["nn", "lin", "lin_time_avg"], string
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).
        Defaults to "nn".

    interpolator : BaseChannelInterpolator
        An instance of :class:`~sionna.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specfied
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    def estimate_at_pilot_locations(self, y_pilots, no):

        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], tf.complex
        #     The observed signals for the pilot-carrying resource elements.

        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   tf.float
        #     The variance of the AWGN.

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots)

        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, tf.rank(h_ls), -1)

        # Expand rank of pilots for broadcasting
        pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2)

        return h_ls, err_var


class MyLSChannelEstimator():
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized observation,
    where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated accross
    the entire resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    interpolation_type : One of ["nn", "lin", "lin_time_avg"], string
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).
        Defaults to "nn".

    interpolator : BaseChannelInterpolator
        An instance of :class:`~sionna.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specfied
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    def __init__(self, resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs):
        #super().__init__(dtype=dtype, **kwargs)

        # assert isinstance(resource_grid, ResourceGrid),\
        #     "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern

        #added test code
        mask = np.array(self._pilot_pattern.mask) #(1, 2, 14, 64)
        mask_shape = mask.shape # Store to reconstruct the original shape
        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = self._pilot_pattern.pilots #(1, 2, 128)
        #print(mask_shape) #(1, 2, 14, 64)
        #print('mask:', mask[0,0,0,:]) #all 0
        #print('pilots:', pilots[0,0,:]) #0.99999994-0.99999994j  0.        +0.j         -0.99999994-0.99999994j
        #0.        +0.j         -0.99999994-0.99999994j  0.        +0.j
        self._removed_nulled_scs = MyRemoveNulledSubcarriers(resource_grid)

        assert interpolation_type in ["nn","lin","lin_time_avg",None], \
            "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type

        if interpolator is not None:
            assert isinstance(interpolator, BaseChannelInterpolator), \
        "`interpolator` must implement the BaseChannelInterpolator interface"
            self._interpol = interpolator
        elif self._interpolation_type == "nn":
            self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin":
            self._interpol = LinearInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin_time_avg":
            self._interpol = LinearInterpolator(self._pilot_pattern,
                                                time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols #128
        mask = flatten_last_dims(self._pilot_pattern.mask) #(1, 2, 896)
        np.save('mask_tf.npy', mask)
        pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING") #(1, 2, 896)
        self._pilot_ind = pilot_ind[...,:num_pilot_symbols]

        print(self._pilot_ind) #(1, 2, 128) int32


    def estimate_at_pilot_locations(self, y_pilots, no):

        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], tf.complex (2, 1, 16, 1, 2, 128)
        #     The observed signals for the pilot-carrying resource elements.

        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   tf.float
        #     The variance of the AWGN.

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots) #pilots: (1, 2, 128)=>(2, 1, 16, 1, 2, 128)

        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, tf.rank(h_ls), -1) #(1, 1, 1, 1, 1, 1)

        # Expand rank of pilots for broadcasting
        pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0) #(1, 1, 1, 1, 2, 128)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2) #(1, 1, 1, 1, 2, 128)

        return h_ls, err_var #h_ls: (2, 1, 16, 1, 2, 128), err_var: (1, 1, 1, 1, 2, 128)

    #def call(self, inputs):
    def __call__(self, inputs):

        y, no = inputs #y: (2, 1, 16, 14, 76) tf complex64

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Removed nulled subcarriers (guards, dc)
        y_eff = self._removed_nulled_scs(y) #(2, 1, 16, 14, 64) complex64

        # Flatten the resource grid for pilot extraction
        # New shape: [...,num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff) #(2, 1, 16, 896)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.real(y_eff_flat[0,0,0,:]))
        plt.plot(np.imag(y_eff_flat[0,0,0,:]))
        plt.title('y_eff_flat')

        # Gather pilots along the last dimensions, pilot_ind: (1, 2, 128)
        # Resulting shape: y_eff_flat.shape[:-1] + pilot_ind.shape, i.e.:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_pilot_symbols]
        y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1) #y_eff_flat:(2, 1, 16, 896) pilot_ind: (1, 2, 128) =>(2, 1, 16, 1, 2, 128)
        
        plt.figure()
        plt.plot(np.real(y_pilots[0,0,0,0,0,:]))
        plt.plot(np.imag(y_pilots[0,0,0,0,0,:]))
        plt.title('y_pilots')
        np.save('y_eff_flat_tf.npy', y_eff_flat.numpy())
        np.save('pilot_ind_tf.npy', self._pilot_ind)
        np.save('y_pilots_tf.npy', y_pilots.numpy())

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)#h_hat: (2, 1, 16, 1, 2, 128), err_var: (1, 1, 1, 1, 2, 128)
        #h_ls: (2, 1, 16, 1, 2, 128), err_var: (1, 1, 1, 1, 2, 128)
        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_hat, err_var = self._interpol(h_hat, err_var) #h_hat: (2, 1, 16, 1, 2, 14, 64)=>(2, 1, 16, 1, 2, 14, 64)
            err_var = tf.maximum(err_var, tf.cast(0, err_var.dtype))

        return h_hat, err_var

class GenerateFlatFadingChannel():
    # pylint: disable=line-too-long
    r"""Generates tensors of flat-fading channel realizations.

    This class generates batches of random flat-fading channel matrices.
    A spatial correlation can be applied.

    Parameters
    ----------
    num_tx_ant : int
        Number of transmit antennas.

    num_rx_ant : int
        Number of receive antennas.

    spatial_corr : SpatialCorrelation, None
        An instance of :class:`~sionna.channel.SpatialCorrelation` or `None`.
        Defaults to `None`.

    dtype : tf.complex64, tf.complex128
        The dtype of the output. Defaults to `tf.complex64`.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of channel matrices to generate.

    Output
    ------
    h : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Batch of random flat fading channel matrices.

    """
    def __init__(self, num_tx_ant, num_rx_ant, spatial_corr=None, dtype=tf.complex64, **kwargs):
        super().__init__(**kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._dtype = dtype
        self.spatial_corr = spatial_corr

    @property
    def spatial_corr(self):
        """The :class:`~sionna.channel.SpatialCorrelation` to be used."""
        return self._spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._spatial_corr = value

    def __call__(self, batch_size):
        # Generate standard complex Gaussian matrices
        shape = [batch_size, self._num_rx_ant, self._num_tx_ant]
        h = complex_normal(shape, dtype=self._dtype)

        # Apply spatial correlation
        if self.spatial_corr is not None:
            h = self.spatial_corr(h)

        return h

class ApplyFlatFadingChannel(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""ApplyFlatFadingChannel(add_awgn=True, dtype=tf.complex64, **kwargs)

    Applies given channel matrices to a vector input and adds AWGN.

    This class applies a given tensor of flat-fading channel matrices
    to an input tensor. AWGN noise can be optionally added.
    Mathematically, for channel matrices
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}`
    and input :math:`\mathbf{x}\in\mathbb{C}^{K}`, the output is

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})`
    is an AWGN vector that is optionally added.


    Parameters
    ----------
    add_awgn: bool
        Indicates if AWGN noise should be added to the output.
        Defaults to `True`.

    dtype : tf.complex64, tf.complex128
        The dtype of the output. Defaults to `tf.complex64`.

    Input
    -----
    (x, h, no) :
        Tuple:

    x : [batch_size, num_tx_ant], tf.complex
        Tensor of transmit vectors.

    h : [batch_size, num_rx_ant, num_tx_ant], tf.complex
        Tensor of channel realizations. Will be broadcast to the
        dimensions of ``x`` if needed.

    no : Scalar or Tensor, tf.float
        The noise power ``no`` is per complex dimension.
        Only required if ``add_awgn==True``.
        Will be broadcast to the shape of ``y``.
        For more details, see :class:`~sionna.channel.AWGN`.

    Output
    ------
    y : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Channel output.
    """
    def __init__(self, add_awgn=True, dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._add_awgn = add_awgn

    def build(self, input_shape): #pylint: disable=unused-argument
        if self._add_awgn:
            self._awgn = AWGN(dtype=self.dtype)

    def call(self, inputs):
        if self._add_awgn:
            x, h, no = inputs
        else:
            x, h = inputs

        x = tf.expand_dims(x, axis=-1)
        y = tf.matmul(h, x)
        y = tf.squeeze(y, axis=-1)

        if self._add_awgn:
            y = self._awgn((y, no))

        return y

class FlatFadingChannel(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""FlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=None, add_awgn=True, return_channel=False, dtype=tf.complex64, **kwargs)

    Applies random channel matrices to a vector input and adds AWGN.

    This class combines :class:`~sionna.channel.GenerateFlatFadingChannel` and
    :class:`~sionna.channel.ApplyFlatFadingChannel` and computes the output of
    a flat-fading channel with AWGN.

    For a given batch of input vectors :math:`\mathbf{x}\in\mathbb{C}^{K}`,
    the output is

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` are randomly generated
    flat-fading channel matrices and
    :math:`\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})`
    is an AWGN vector that is optionally added.

    A :class:`~sionna.channel.SpatialCorrelation` can be configured and the
    channel realizations optionally returned. This is useful to simulate
    receiver algorithms with perfect channel knowledge.

    Parameters
    ----------
    num_tx_ant : int
        Number of transmit antennas.

    num_rx_ant : int
        Number of receive antennas.

    spatial_corr : SpatialCorrelation, None
        An instance of :class:`~sionna.channel.SpatialCorrelation` or `None`.
        Defaults to `None`.

    add_awgn: bool
        Indicates if AWGN noise should be added to the output.
        Defaults to `True`.

    return_channel: bool
        Indicates if the channel realizations should be returned.
        Defaults  to `False`.

    dtype : tf.complex64, tf.complex128
        The dtype of the output. Defaults to `tf.complex64`.

    Input
    -----
    (x, no) :
        Tuple or Tensor:

    x : [batch_size, num_tx_ant], tf.complex
        Tensor of transmit vectors.

    no : Scalar of Tensor, tf.float
        The noise power ``no`` is per complex dimension.
        Only required if ``add_awgn==True``.
        Will be broadcast to the dimensions of the channel output if needed.
        For more details, see :class:`~sionna.channel.AWGN`.

    Output
    ------
    (y, h) :
        Tuple or Tensor:

    y : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Channel output.

    h : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Channel realizations. Will only be returned if
        ``return_channel==True``.
    """
    def __init__(self,
                 num_tx_ant,
                 num_rx_ant,
                 spatial_corr=None,
                 add_awgn=True,
                 return_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._add_awgn = add_awgn
        self._return_channel = return_channel
        self._gen_chn = GenerateFlatFadingChannel(self._num_tx_ant,
                                                  self._num_rx_ant,
                                                  spatial_corr,
                                                  dtype=dtype)
        self._app_chn = ApplyFlatFadingChannel(add_awgn=add_awgn, dtype=dtype)

    @property
    def spatial_corr(self):
        """The :class:`~sionna.channel.SpatialCorrelation` to be used."""
        return self._gen_chn.spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._gen_chn.spatial_corr = value

    @property
    def generate(self):
        """Calls the internal :class:`GenerateFlatFadingChannel`."""
        return self._gen_chn

    @property
    def apply(self):
        """Calls the internal :class:`ApplyFlatFadingChannel`."""
        return self._app_chn

    def call(self, inputs):
        if self._add_awgn:
            x, no = inputs
        else:
            x = inputs

        # Generate a batch of channel realizations
        batch_size = tf.shape(x)[0]
        h = self._gen_chn(batch_size)

        # Apply the channel to the input
        if self._add_awgn:
            y = self._app_chn([x, h, no])
        else:
            y = self._app_chn([x, h])

        if self._return_channel:
            return y, h
        else:
            return y
