import numpy as np


def padding(input_arr, block_size):
    assert len(input_arr) <= block_size
    n = block_size - len(input_arr)
    return np.pad(input_arr, (n, 0), 'constant', constant_values=(0, ))