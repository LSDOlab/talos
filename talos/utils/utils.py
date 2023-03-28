import numpy as np


def get_array_indices(*shape):
    return np.arange(np.prod(shape)).reshape(shape)


def get_val(shape, random_indices=None):
    if isinstance(shape, int):
        shape = (shape,)
    if random_indices is None:
        random_indices = np.arange(len(shape))

    alphabet = 'abcdefghijk'
    random_shape = []
    ones_shape = []
    random_str = ''
    ones_str = ''
    for ind in range(len(shape)):
        if ind in random_indices:
            random_shape.append(shape[ind])
            random_str += alphabet[ind]
        else:
            ones_shape.append(shape[ind])
            ones_str += alphabet[ind]

    all_str = alphabet[:len(shape)]

    return np.einsum(
        '{},{}->{}'.format(random_str, ones_str, all_str),
        np.random.rand(*random_shape),
        np.ones(ones_shape),
    )


def get_array_expansion_data(shape, expand_indices):
    alphabet = 'abcdefghij'

    in_string = ''
    out_string = ''
    ones_string = ''
    in_shape = []
    out_shape = []
    ones_shape = []
    for index in range(len(shape)):
        if index not in expand_indices:
            in_string += alphabet[index]
            in_shape.append(shape[index])
        else:
            ones_string += alphabet[index]
            ones_shape.append(shape[index])
        out_string += alphabet[index]
        out_shape.append(shape[index])

    einsum_string = '{},{}->{}'.format(in_string, ones_string, out_string)
    in_shape = tuple(in_shape)
    out_shape = tuple(out_shape)
    ones_shape = tuple(ones_shape)

    return einsum_string, in_shape, out_shape, ones_shape