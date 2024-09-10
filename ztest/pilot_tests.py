
import numpy as np
from qampy import theory
from qampy.core import equalisation,  phaserecovery, pilotbased_receiver,pilotbased_transmitter,filter,\
    resample
from qampy import signals, impairments, helpers, phaserec
from qampy.equalisation import pilot_equaliser
import matplotlib.pylab as plt
from optic.comm.modulation import modulateGray, demodulateGray, grayMapping, detector
from qampy.core.signal_quality import make_decision, generate_bitmapping_mtx




M = 4
dtype = np.complex128

npols = 2
frame_len = 2**16
pilot_seq_len = 512
pilot_ins_rat = 32
pilot_scale = 1
out_symbs = np.empty((npols, frame_len), dtype=dtype)

idx = np.arange(frame_len)
idx_pil_seq = idx < pilot_seq_len
# print(idx_pil_seq)

if pilot_ins_rat == 0 or pilot_ins_rat is None:
    idx_pil = idx_pil_seq
else:
    if (frame_len - pilot_seq_len) % pilot_ins_rat != 0:
        raise ValueError("Frame without pilot sequence divided by pilot rate needs to be an integer")
    N_ph_frames = (frame_len - pilot_seq_len) // pilot_ins_rat
    idx_ph_pil = ((idx - pilot_seq_len) % pilot_ins_rat != 0) & (idx - pilot_seq_len > 0)
    idx_ph_pil[pilot_seq_len] =~ idx_ph_pil[pilot_seq_len]
    idx_pil = ~idx_ph_pil  # ^ idx_pil_seq
idx_dat = ~idx_pil
# print(np.count_nonzero(idx_pil))
req_symb = np.count_nonzero(idx_pil)


scale = np.sqrt(theory.cal_scaling_factor_qam(M))
# Nbits = int(np.log2(M))
symbols = theory.cal_symbols_qam(M).astype(dtype)
# check if this gives the correct mapping
symbols /= scale
_graycode = theory.gray_code_qam(M)
# print(_graycode.shape)
u = np.zeros_like(_graycode)
u[_graycode] = np.arange(u.size)
coded_symbols = symbols[u]
encoding = np.zeros((_graycode.size, int(np.log2(M))), bool)

for i in range(_graycode.size):
    encoding[i] = np.fromstring(np.binary_repr(i, width=int(np.log2(M))), dtype="S1").astype(bool)
# print(encoding.shape)

symbol_idx = np.arange(_graycode.size)
bits = encoding[symbol_idx]

# print(bits)
# print(symbol_idx.ndim)
if symbol_idx.ndim > 1:
    bits.reshape(symbol_idx.shape[0], -1)
else:
    bits.flatten()

# print(bits.shape)

bitmap_mtx = generate_bitmapping_mtx(coded_symbols, bits, M, dtype=dtype)
print(bitmap_mtx.shape)
num_bits = req_symb * int(np.log2(M))
seed = 0
R = np.random.RandomState(seed)
data = R.randint(0, high=2, size=(npols, num_bits)).astype(bool)
data = np.atleast_2d(data)
print(data.shape)
nmodes = data.shape[0]
M = coded_symbols.shape[0]
bitspsym = int(np.log2(M))
Nsym = data.shape[1] // bitspsym
print(Nsym)
out = np.empty((nmodes, Nsym), dtype=dtype)
N = data.shape[1] - data.shape[-1] % bitspsym
print(N)
cov = 2**np.arange(bitspsym-1, -1, -1)
# cov = [2, 1]
for i in range(nmodes):
    datab = data[i, :N].reshape(-1, bitspsym)
    idx = datab.dot(cov)
    out[i, :] = coded_symbols[idx]

out_symbs[:, idx_pil] = out
symbs = signals.SignalQAMGrayCoded(256, np.count_nonzero(idx_dat), nmodes=nmodes, dtype=dtype)
out_symbs[:, idx_dat] = symbs
print(out_symbs.shape)
print(out_symbs[0, 511], out_symbs[0, 512])
# out_symbs = np.tile(out_symbs, nframes)
# pilots = signals.SignalQAMGrayCoded(4, np.count_nonzero(idx_pil), nmodes=npols, dtype=dtype) * pilot_scale

# print(pilots._code)
# print(pilots._encoding)
# print(pilots._bitmap_mtx)
# print(pilots._coded_symbols)

#signal = signals.SignalWithPilots(M=16,
#                                  frame_len=2**16,
#                                  pilot_seq_len=512,
#                                  pilot_ins_rat=32,
#                                  nframes=3, nmodes=npols,
#                                  fb=20e9)
