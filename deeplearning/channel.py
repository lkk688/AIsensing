import tensorflow as tf
from tensorflow.keras.layers import Layer

PI = 3.141592653589793
SPEED_OF_LIGHT = 299792458


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
            x, h_freq, no = inputs
        else:
            x, h_freq = inputs

        # Apply the channel response
        x = expand_to_rank(x, h_freq.shape.rank, axis=1)
        y = tf.reduce_sum(tf.reduce_sum(h_freq*x, axis=4), axis=3)

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
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        self._subcarrier_spacing = resource_grid.subcarrier_spacing
        self._num_subcarriers = resource_grid.fft_size
        self._normalize_channel = normalize_channel
        self._sampling_frequency = 1./resource_grid.ofdm_symbol_duration

        # Frequencies of the subcarriers
        self._frequencies = subcarrier_frequencies(self._num_subcarriers,
                                                   self._subcarrier_spacing,
                                                   dtype)

    def __call__(self, batch_size=None):

        # Sample channel impulse responses
        h, tau = self._cir_sampler( batch_size,
                                    self._num_ofdm_symbols,
                                    self._sampling_frequency)

        h_freq = cir_to_ofdm_channel(self._frequencies, h, tau,
                                     self._normalize_channel)

        return h_freq
    
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
