import numpy as np

def collapse_last_two_dimensions(arr):
    """
    Collapses the last two dimensions of a NumPy array.

    :param arr: Input array with at least two dimensions
    :return: Collapsed array
    """
    if arr.ndim < 2:
        raise ValueError("Input array must have at least two dimensions")

    # Reshape the array by collapsing the last two dimensions
    collapsed_array = arr.reshape(arr.shape[:-2] + (-1,))

    return collapsed_array

# Example usage:
input_array = np.random.rand(3, 4, 5, 6)  # Example input array with shape (3, 4, 5, 6)
collapsed_result = collapse_last_two_dimensions(input_array)
print("Collapsed array shape:", collapsed_result.shape) #(3, 4, 30)



#template_tensor is the existing tensor where you want to scatter the inputs.
#inputs contains the values you want to scatter.
def scatter_numpy(template_tensor, inputs, _data_ind):
    """
    Scatters values from the `inputs` array into the `template_tensor`
    at the indices specified by `_data_ind`.

    :param template_tensor: Existing tensor (template) to scatter into
    :param inputs: Array of values to scatter
    :param _data_ind: Indices where values from `inputs` should be placed
    :return: Updated template tensor with scattered values
    """
    if _data_ind.dtype != np.dtype('int_'):
        raise TypeError("The values of _data_ind must be integers")

    if template_tensor.ndim != _data_ind.ndim:
        raise ValueError("The number of dimensions in _data_ind should match the template tensor")

    # Ensure positive indices
    _data_ind = np.clip(_data_ind, 0, template_tensor.shape[0] - 1)

    # Scatter values from inputs into template_tensor
    template_tensor[_data_ind] = inputs

    return template_tensor

def tensor_scatter_nd_update(tensor, indices, updates):
    """
    Updates the `tensor` by scattering `updates` into it at the specified `indices`.

    :param tensor: Existing tensor to update
    :param indices: Array of indices where updates should be placed
    :param updates: Array of values to scatter
    :return: Updated tensor
    """
    # Ensure positive indices
    indices = np.clip(indices, 0, np.array(tensor.shape) - 1)

    # Scatter values from updates into tensor
    tensor[tuple(indices.T)] = updates #indices(3, 2)=>2,3 tensor(3,3)

    return tensor

# Example usage:
template = np.zeros((3, 3))  # Example template tensor
indices = np.array([[0, 1], [2, 0], [1, 1]])  # Example indices (3,2)
values = np.array([10, 20, 30])  # Example values (3,)

result = tensor_scatter_nd_update(template, indices, values)
print("Updated template tensor:")
print(result)

# Example usage:
template = np.zeros((10,))  # Example template tensor
inputs = np.array([1, 2, 3])  # Example input values
_data_ind = np.array([2, 5, 8])  # Example indices

result = scatter_numpy(template, inputs, _data_ind)
print("Updated template tensor:")
print(result)
