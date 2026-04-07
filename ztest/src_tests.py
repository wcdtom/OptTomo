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
from optic_plus.model_plus import tx_plus

param = parameters()
TX = tx_plus.pilotWDMTx(param)
print(TX.pulseTxWDM)




# # If synchronization fails, then change sync_bool to 'False'
sync_bool = True
#
FRAME_SYNC_THRS = 120 # this is somewhat arbitrary but seems to work well
rx_signal = np.atleast_2d(rx_signal)
ref_symbs = np.atleast_2d(ref_symbs)
pilot_seq_len = ref_symbs.shape[-1]
nmodes = rx_signal.shape[0]
assert rx_signal.shape[-1] >= (frame_len + 2*pilot_seq_len) * SpS, "Signal must be at least as long as frame"
#
mode_sync_order = np.zeros(nmodes, dtype=int)
not_found_modes = np.arange(0, nmodes)
search_overlap = 2 # fraction of pilot_sequence to overlap
search_window = pilot_seq_len * SpS
step = search_window // search_overlap
# we only need to search the length of one frame*os plus some buffer (the extra step)
num_steps = (frame_len * SpS)//step + 1
# Now search for every mode independent
shift_factor = np.zeros(nmodes, dtype=int)
# Search based on equalizer error. Avoid one pilot_seq_len part in the beginning and
# end to ensure that sufficient symbols can be used for the search
sub_vars = np.ones((nmodes, num_steps)) * 1e2
wxys = np.zeros((num_steps, nmodes, nmodes, Ntaps), dtype=rx_signal.dtype)
for i in np.arange(search_overlap, num_steps): # we avoid one step at the beginning
    tmp = rx_signal[:, i*step:i*step+search_window]
    wxy, err_out = equalisation.equalise_signal(tmp, os, mu, M_pilot, Ntaps=Ntaps, **eqargs)
    wxys[i] = wxy
    sub_vars[:,i] = np.var(err_out, axis=-1)

# Lowest variance of the CMA error for each pol
min_range = np.argmin(sub_vars, axis=-1)
wxy = wxys[min_range]
for l in range(nmodes):
    idx_min = min_range[l]
    # Extract a longer sequence to ensure that the complete pilot sequence is found
    longSeq = rx_signal[:, (idx_min)*step-search_window: (idx_min )*step+search_window]
    # Apply filter taps to the long sequence and remove coarse FO
    wx1 = wxy[l]
    symbs_out = equalisation.apply_filter(longSeq,os,wx1)
    foe_corse = phaserecovery.find_freq_offset(symbs_out)
    symbs_out = phaserecovery.comp_freq_offset(symbs_out, foe_corse)
    # Check for pi/2 ambiguties and verify all
    max_phase_rot = np.zeros(nmodes, dtype=np.float64)
    found_delay = np.zeros(nmodes, dtype=np.int32)
    for ref_pol in not_found_modes:
        ix, dat, ii, ac = ber_functions.find_sequence_offset_complex(ref_symbs[ref_pol], symbs_out[l])
        found_delay[ref_pol] = -ix
        max_phase_rot[ref_pol] = ac
    # Check for which mode found and extract the reference delay
    max_sync_pol = np.argmax(max_phase_rot)
    if max_phase_rot[max_sync_pol] < FRAME_SYNC_THRS: #
        warnings.warn("Very low autocorrelation, likely the frame-sync failed")
        sync_bool = False
    mode_sync_order[l] = max_sync_pol
    symb_delay = found_delay[max_sync_pol]
    # Remove the found reference mode
    not_found_modes = not_found_modes[not_found_modes != max_sync_pol]
    # New starting sample
    shift_factor[l] = (idx_min)*step + os*symb_delay - search_windoww



