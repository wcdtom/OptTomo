import numpy
from optic.models.tx import simpleWDMTx
from optic.utils import parameters, dBm2W
import numpy as np
from matplotlib import pyplot as plt

seed_num = 55  # necessary condition
np.random.seed(seed=seed_num)  # fixing the seed to get reproducible results

# simulation parameters
SpS = 16            # samples per symbol
M = 4              # order of the modulation format
Rs = 10e9          # Symbol rate (for OOK case Rs = Rb)
Fs = SpS*Rs        # Sampling frequency in samples/second  # necessary condition
Ts = 1/Fs          # Sampling period  # necessary condition
signal_length = 1e4  # <-- 4.2e6 sample

# Transmitter parameters:
paramTx = parameters()
paramTx.M = M           # order of the modulation format
paramTx.Rs = Rs         # symbol rate [baud]
paramTx.SpS = SpS           # samples per symbol
paramTx.pulse = 'rrc'      # pulse shaping filter
paramTx.Ntaps = 4096     # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01    # RRC rolloff
paramTx.Pch_dBm = 10        # power per WDM channel [dBm]
paramTx.Nch = 1       # number of WDM channels
paramTx.Fc = 193.1e12  # central optical frequency of the WDM spectrum
paramTx.lw = 0  # 100e3    # laser linewidth in Hz
paramTx.freqSpac = 37.5e9  # WDM grid spacing
paramTx.Nmodes = 1         # number of signal modes [2 for polarization multiplexed signals]
paramTx.Nbits = int(np.log2(paramTx.M)*signal_length)  # total number of bits per polarization

# # generate WDM signal
# sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)
# sigTxo = numpy.squeeze(sigWDM_Tx)

# print('Generator')

if __name__ == '__main__':
    print('Generating signal...')
    # generate WDM signal
    sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)
    sigTxo = numpy.squeeze(sigWDM_Tx)

    interval = np.arange(16 * 20, 16 * 50)
    t = interval * Ts / 1e-9
    fig, axs = plt.subplots()
    axs.plot(t, sigTxo[interval].real, 'r-')
    axs.plot(t, sigTxo[interval].imag, 'r--')
    plt.show()
