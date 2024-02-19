import numpy as np

#Scatter Operation (Equivalent to tf.scatter_nd): The scatter_nd operation writes values from a source tensor (src) into 
#a destination tensor (self) at specified indices. Below is a custom implementation of scatter_nd in NumPy:
def scatter_numpy(self, dim, index, src):
    """
    Writes all values from the Tensor src into self at the indices specified in the index Tensor.
    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: self
    """
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    if self.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= self.ndim or dim < -self.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        dim = self.ndim + dim

    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]

    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(f"Except for dimension {dim}, all dimensions of index and output should be the same size")

    if (index >= self.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] - 1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return slc

    idx = [[*np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1),
            index[make_slice(index, dim, i)].reshape(1, -1)[0]] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError(f"Dimension {dim} of index cannot be bigger than that of src")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError(f"Except for dimension {dim}, all dimensions of index and src should be the same size")

        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        self[idx] = src[src_idx]
    else:
        self[idx] = src

    return self


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
