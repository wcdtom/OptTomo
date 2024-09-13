import numpy as np
from qampy import theory
from qampy.core import equalisation,  phaserecovery, pilotbased_receiver,pilotbased_transmitter,filter,\
    resample
from qampy import signals, impairments, helpers, phaserec
from qampy.equalisation import pilot_equaliser
import matplotlib.pylab as plt
from optic.comm.modulation import modulateGray, demodulateGray, grayMapping, detector
from qampy.core.signal_quality import make_decision, generate_bitmapping_mtx
from optic.models.devices import mzm, photodiode, edfa
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import grayMapping, modulateGray, demodulateGray
from optic.comm.metrics import theoryBER
from optic.dsp.core import upsample, pulseShape, lowPassFIR, pnorm, signal_power, firFilter
from optic.utils import parameters, dBm2W
from optic.plot import eyediagram, pconst
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm.notebook import tqdm
import scipy as sp
from optic.models.tx import simpleWDMTx

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
print(symbols)
# check if this gives the correct mapping
symbols /= scale
print(symbols)
print(pnorm(symbols))
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

# print(out_symbs[0, 512:530])
  # order of the modulation format
# constType = 'qam' # 'qam', 'psk', 'apsk', 'pam' or 'ook'

# generate random bits
# bit_series = np.random.randint(2, size=int(2**16))
# print(bit_series.shape)
# Map bits to constellation symbols
# symbTx = modulateGray(bit_series, 256, constType)
# print(symbTx)
# normalize symbols energy to 1
# symbTx = pnorm(symbTx)
# print(symbTx[512:530])
# out_symbs = np.tile(out_symbs, nframes)
# pilots = signals.SignalQAMGrayCoded(4, np.count_nonzero(idx_pil), nmodes=npols, dtype=dtype) * pilot_scale

# print(pilots._code)
# print(pilots._encoding)
# print(pilots._bitmap_mtx)
# print(pilots._coded_symbols)

signal_class = signals.SignalWithPilots(M=256,
                                        frame_len=2**16,
                                        pilot_seq_len=512,
                                        pilot_ins_rat=32,
                                        nframes=3, nmodes=npols,
                                        fb=20e9)
# signal2 = signal.resample(signal.fb*2, beta=0.1, renormalise=True)
signal = signal_class.symbTx()
print(signal.shape)
# print(signal2.shape)


# simulation parameters
SpS = 16            # samples per symbol
# M = 4              # order of the modulation format
Rs = 10e9          # Symbol rate (for OOK case Rs = Rb)
Fs = SpS*Rs        # Sampling frequency in samples/second
Ts = 1/Fs          # Sampling period

# Laser power
Pi_dBm = 0         # laser optical power at the input of the MZM in dBm
Pi = dBm2W(Pi_dBm) # convert from dBm to W
# upsampling
symbolsUp = upsample(signal, SpS)
print(symbolsUp.shape)
# typical NRZ pulse
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

# pulse shaping
sigTx = firFilter(pulse, symbolsUp)

print(sigTx.shape)
print(signal_class.pilot_seq)


# # If synchronization fails, then change sync_bool to 'False'
# sync_bool = True
#
# FRAME_SYNC_THRS = 120 # this is somewhat arbitrary but seems to work well
# rx_signal = np.atleast_2d(rx_signal)
# ref_symbs = np.atleast_2d(ref_symbs)
# pilot_seq_len = ref_symbs.shape[-1]
# nmodes = rx_signal.shape[0]
# assert rx_signal.shape[-1] >= (frame_len + 2*pilot_seq_len)*os, "Signal must be at least as long as frame"
#
# mode_sync_order = np.zeros(nmodes, dtype=int)
# not_found_modes = np.arange(0, nmodes)
# search_overlap = 2 # fraction of pilot_sequence to overlap
# search_window = pilot_seq_len * os
# step = search_window // search_overlap
# # we only need to search the length of one frame*os plus some buffer (the extra step)
# num_steps = (frame_len*os)//step + 1
# # Now search for every mode independent
# shift_factor = np.zeros(nmodes, dtype=int)
# # Search based on equalizer error. Avoid one pilot_seq_len part in the beginning and
# # end to ensure that sufficient symbols can be used for the search
# sub_vars = np.ones((nmodes, num_steps)) * 1e2
# wxys = np.zeros((num_steps, nmodes, nmodes, Ntaps), dtype=rx_signal.dtype)
# for i in np.arange(search_overlap, num_steps): # we avoid one step at the beginning
#     tmp = rx_signal[:, i*step:i*step+search_window]
#     wxy, err_out = equalisation.equalise_signal(tmp, os, mu, M_pilot, Ntaps=Ntaps, **eqargs)
#     wxys[i] = wxy
#     sub_vars[:,i] = np.var(err_out, axis=-1)
# # Lowest variance of the CMA error for each pol
# min_range = np.argmin(sub_vars, axis=-1)
# wxy = wxys[min_range]
# for l in range(nmodes):
#     idx_min = min_range[l]
#     # Extract a longer sequence to ensure that the complete pilot sequence is found
#     longSeq = rx_signal[:, (idx_min)*step-search_window: (idx_min )*step+search_window]
#     # Apply filter taps to the long sequence and remove coarse FO
#     wx1 = wxy[l]
#     symbs_out = equalisation.apply_filter(longSeq,os,wx1)
#     foe_corse = phaserecovery.find_freq_offset(symbs_out)
#     symbs_out = phaserecovery.comp_freq_offset(symbs_out, foe_corse)
#     # Check for pi/2 ambiguties and verify all
#     max_phase_rot = np.zeros(nmodes, dtype=np.float64)
#     found_delay = np.zeros(nmodes, dtype=np.int32)
#     for ref_pol in not_found_modes:
#         ix, dat, ii, ac = ber_functions.find_sequence_offset_complex(ref_symbs[ref_pol], symbs_out[l])
#         found_delay[ref_pol] = -ix
#         max_phase_rot[ref_pol] = ac
#     # Check for which mode found and extract the reference delay
#     max_sync_pol = np.argmax(max_phase_rot)
#     if max_phase_rot[max_sync_pol] < FRAME_SYNC_THRS: #
#         warnings.warn("Very low autocorrelation, likely the frame-sync failed")
#         sync_bool = False
#     mode_sync_order[l] = max_sync_pol
#     symb_delay = found_delay[max_sync_pol]
#     # Remove the found reference mode
#     not_found_modes = not_found_modes[not_found_modes != max_sync_pol]
#     # New starting sample
#     shift_factor[l] = (idx_min)*step + os*symb_delay - search_window

# Transmitter parameters:
paramTx = parameters()
paramTx.M   = 16           # order of the modulation format
paramTx.Rs  = 32e9         # symbol rate [baud]
paramTx.SpS = 16           # samples per symbol
paramTx.pulse = 'rrc'      # pulse shaping filter
paramTx.Ntaps = 4096     # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01    # RRC rolloff
paramTx.Pch_dBm = -2        # power per WDM channel [dBm]
paramTx.Nch     = 8       # number of WDM channels
paramTx.Fc      = 193.1e12 # central optical frequency of the WDM spectrum
paramTx.lw      = 100e3    # laser linewidth in Hz
paramTx.freqSpac = 37.5e9  # WDM grid spacing
paramTx.Nmodes = 2         # number of signal modes [2 for polarization multiplexed signals]
paramTx.Nbits = int(np.log2(paramTx.M)*1e3) # total number of bits per polarization

# generate WDM signal
sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)

print(sigWDM_Tx.shape)
