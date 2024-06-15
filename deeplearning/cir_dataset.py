#ref: https://github.com/NVlabs/sionna/blob/main/sionna/channel/cir_dataset.py

import torch
from torch.utils.data import Dataset

from DeepMIMO import DeepMIMOSionnaAdapter

# class CIRDataset(Dataset):
#     # pylint: disable=line-too-long
#     r"""CIRDataset(cir_generator, batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps, dtype=tf.complex64)

#     Creates a channel model from a dataset that can be used with classes such as
#     :class:`~sionna.channel.TimeChannel` and :class:`~sionna.channel.OFDMChannel`.
#     The dataset is defined by a `generator <https://wiki.python.org/moin/Generators>`_.

#     The batch size is configured when instantiating the dataset or through the :attr:`~sionna.channel.CIRDataset.batch_size` property.
#     The number of time steps (`num_time_steps`) and sampling frequency (`sampling_frequency`) can only be set when instantiating the dataset.
#     The specified values must be in accordance with the data.

#     Example
#     --------

#     The following code snippet shows how to use this class as a channel model.

#     >>> my_generator = MyGenerator(...)
#     >>> channel_model = sionna.channel.CIRDataset(my_generator,
#     ...                                           batch_size,
#     ...                                           num_rx,
#     ...                                           num_rx_ant,
#     ...                                           num_tx,
#     ...                                           num_tx_ant,
#     ...                                           num_paths,
#     ...                                           num_time_steps+l_tot-1)
#     >>> channel = sionna.channel.TimeChannel(channel_model, bandwidth, num_time_steps)

#     where ``MyGenerator`` is a generator

#     >>> class MyGenerator:
#     ...
#     ...     def __call__(self):
#     ...         ...
#     ...         yield a, tau

#     that returns complex-valued path coefficients ``a`` with shape
#     `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
#     and real-valued path delays ``tau`` (in second)
#     `[num_rx, num_tx, num_paths]`.

#     Parameters
#     ----------
#     cir_generator : `generator <https://wiki.python.org/moin/Generators>`_
#         Generator that returns channel impulse responses ``(a, tau)`` where
#         ``a`` is the tensor of channel coefficients of shape
#         `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
#         and dtype ``dtype``, and ``tau`` the tensor of path delays
#         of shape  `[num_rx, num_tx, num_paths]` and dtype ``dtype.
#         real_dtype``.

#     batch_size : int
#         Batch size

#     num_rx : int
#         Number of receivers (:math:`N_R`)

#     num_rx_ant : int
#         Number of antennas per receiver (:math:`N_{RA}`)

#     num_tx : int
#         Number of transmitters (:math:`N_T`)

#     num_tx_ant : int
#         Number of antennas per transmitter (:math:`N_{TA}`)

#     num_paths : int
#         Number of paths (:math:`M`)

#     num_time_steps : int
#         Number of time steps

#     dtype : tf.DType
#         Complex datatype to use for internal processing and output.
#         Defaults to `tf.complex64`.

#     Output
#     -------
#     a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
#         Path coefficients

#     tau : [batch size, num_rx, num_tx, num_paths], tf.float
#         Path delays [s]
#     """
#     def __init__(self, cir_generator, batch_size, num_rx, num_rx_ant, num_tx,
#                  num_tx_ant, num_paths, num_time_steps, dtype=torch.complex64):
#         self._cir_generator = cir_generator
#         self._batch_size = batch_size
#         self._num_time_steps = num_time_steps

#     def __len__(self):
#         # Implement the length method (number of samples in the dataset)
#         # Return an appropriate value based on your data

#     def __getitem__(self, idx):
#         # Implement the item retrieval logic
#         # Return a tuple (path_coefficients, path_delays)

# Example usage:
# Instantiate your CIRDataset with appropriate parameters
# my_generator = MyGenerator()  # Replace with your actual generator
# channel_dataset = CIRDataset(my_generator, batch_size, num_rx, num_rx_ant,
#                              num_tx, num_tx_ant, num_paths, num_time_steps)

# Now you can use 'channel_dataset' as needed
# Note: You'll need to implement '__len__' and '__getitem__' methods based on your data
