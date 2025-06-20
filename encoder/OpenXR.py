import numpy as np
from encoder.bch.bchcodegenerator import BchCodeGenerator
from encoder.bch.bchcoder import BchCoder
from encoder.bch.padding import *
from encoder.bch.mathutils import *

def iFEC_encoder(bits, bch_encoder, M=None):
    '''

    :return:
    '''
    # initialized parameters
    indexes = [[(0, 2, 21), (2, 1, 21), (3, 0, 21), (1, 3, 21)],
               [(0, 1, 5), (2, 3, 5), (1, 2, 5), (3, 0, 5)],
               [(3, 3, 2), (0, 2, 2), (1, 1, 2), (2, 0, 2)],
               [(2, 0, 15), (0, 2, 15), (3, 3, 15), (1, 1, 15)],
               [(3, 0, 10), (0, 1, 10), (1, 2, 10), (2, 3, 10)],
               [(2, 1, 29), (0, 2, 29), (1, 0, 29), (3, 3, 29)],
               [(1, 2, 6), (0, 0, 6), (3, 3, 6), (2, 1, 6)],
               [(3, 0, 27), (1, 1, 27), (2, 2, 27), (0, 3, 27)],
               [(3, 1, 16), (0, 3, 16), (2, 0, 16), (1, 2, 16)],
               [(1, 3, 13), (3, 0, 13), (0, 1, 13), (2, 2, 13)],
               [(0, 3, 7), (3, 1, 7), (1, 0, 7), (2, 2, 7)],
               [(3, 3, 4), (1, 0, 4), (0, 1, 4), (2, 2, 4)],
               [(1, 3, 28), (0, 1, 28), (3, 2, 28), (2, 0, 28)],
               [(0, 0, 20), (3, 3, 20), (2, 2, 20), (1, 1, 20)],
               [(2, 0, 24), (3, 1, 24), (1, 2, 24), (0, 3, 24)],
               [(0, 1, 30), (3, 3, 30), (1, 0, 30), (2, 2, 30)],
               [(0, 1, 26), (3, 2, 26), (2, 0, 26), (1, 3, 26)],
               [(2, 2, 25), (3, 3, 25), (1, 0, 25), (0, 1, 25)],
               [(0, 2, 18), (3, 0, 18), (1, 1, 18), (2, 3, 18)],
               [(2, 2, 14), (0, 0, 14), (3, 1, 14), (1, 3, 14)],
               [(2, 0, 0), (3, 1, 0), (1, 2, 0), (0, 3, 0)],
               [(2, 1, 22), (1, 0, 22), (0, 3, 22), (3, 2, 22)],
               [(3, 2, 1), (0, 0, 1), (2, 1, 1), (1, 3, 1)],
               [(2, 1, 3), (0, 3, 3), (1, 0, 3), (3, 2, 3)],
               [(0, 0, 17), (3, 1, 17), (1, 2, 17), (2, 3, 17)],
               [(3, 0, 23), (0, 1, 23), (1, 2, 23), (2, 3, 23)],
               [(1, 0, 12), (2, 3, 12), (0, 2, 12), (3, 1, 12)],
               [(1, 2, 8), (2, 3, 8), (0, 1, 8), (3, 0, 8)],
               [(3, 1, 19), (2, 2, 19), (0, 3, 19), (1, 0, 19)],
               [(1, 3, 9), (2, 1, 9), (3, 0, 9), (0, 2, 9)],
               [(1, 2, 11), (0, 3, 11), (3, 1, 11), (2, 0, 11)]]
    # bits padding
    if M is None:
        M = np.zeros(shape=(9, 4, 31, 31), dtype='bool')
    if len(bits) % 3317 == 0:
        Niter = int(len(bits) // 3317)
    else:
        Npad = 3317 - int(len(bits) % 3317)
        bits = np.pad(bits, (0, Npad), 'constant', constant_values=(False, False))
        Niter = int(len(bits) // 3317)
    bits = bits.reshape((3317, Niter))
    c = np.array([0,1,2,3], dtype='int')
    x = np.arange(31)

    for i in range(Niter):
        bits_iter = bits[:, i]
        L_1_i = bits_iter.reshape((31, 107))
        L_2_i = np.zeros(shape=(31, 124), dtype='bool')
        # initialization
        if i==0:
            for v, row_indexes in enumerate(indexes):
                for k, column_indexes in enumerate(row_indexes):
                    L_2_i[v, k * 31:(k+1) * 31] = M[column_indexes[0], column_indexes[1], :, column_indexes[2]]
        else:
            X_i = np.random.permutation(x)
            r = [(c_j + i) % 9 for c_j in c]
            R_i = np.random.permutation(np.array(r, dtype='int'))
            C_i = np.random.permutation(c)
            pairs_i = [(R_i_j, C_i_j) for R_i_j, C_i_j in zip(R_i, C_i)]
            # TODO: optimization as a matrix operation
            v = 0
            while v < 31:
                for k, pair in enumerate(pairs_i):
                    for l in range(31):
                        L_2_i[v, k*31:] = M[pair[0], pair[1], :, X_i[l]]
                    if k==3:
                        v += 1
        L_i = np.concatenate((L_1_i, L_2_i),axis=1)
        LL_1_i = BCH_248_231_encoder(L_i)
        row_block = int((i+8)%9)
        M[row_block,:,:,:] = LL_1_i

    return M

def BCH_248_231_encoder(L, encoder):
    L_star = np.zeros((31, 124))
    for i in range(L.shape[0]):
        input_arr = L[i,:]
        padded_tx = padding(input_arr, 239)
        encoded = encoder(msg_poly=Poly(padded_tx, x))
        encoded = np.array(encoded)
        padded_encoded = padding(encoded, 248)
        L_star[i, :] = padded_encoded[247:124]
    return L_star

def iFEC_encoder_fast(bits, M=None):
    pass

def short_interleaver(LL_1_4arr):
    '''

    Parameters
    ----------
    LL_1_4arr: size [4, 31, 124]

    Returns short_interleaver
    -------
    '''
    # short interleaving
    LL_1_short_interleaver = np.zeros(124, 124)
    LL_1_fed = LL_1_4arr.reshape(124,124)
    for i in range (0,124):
        shift_value = int(i % 31)
        LL_1_short_interleaver[i,:] = np.roll(LL_1_fed[i,:], shift_value)
    return LL_1_short_interleaver

