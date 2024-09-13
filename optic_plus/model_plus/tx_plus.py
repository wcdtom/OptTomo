"""
=================================================================
Advanced models for optical transmitters (:mod:`optic.models.tx`)
=================================================================

.. autosummary::
   :toctree: generated/

   simpleWDMTx          -- Implement a simple WDM transmitter.
"""
# 3rd party library
import numpy as np
from tqdm.notebook import tqdm

from optic.dsp.core import pnorm, pulseShape, signal_power, upsample, phaseNoise
from optic.models.devices import iqm
from optic.comm.modulation import grayMapping, modulateGray

try:
    from optic.dsp.coreGPU import firFilter
except ImportError:
    from optic.dsp.core import firFilter

# local library
from qampy.signals import SignalWithPilots

# sys library
import logging as logg



def pilotWDMTx(param):
    """
    Implement a simple WDM transmitter.

    Generates a complex baseband waveform representing a WDM signal with
    arbitrary number of carriers

    Parameters
    ----------
    param : system parameters of the WDM transmitter.
        optic.core.parameter object.

        - param.M: modulation order [default: 16].

        - param.constType: 'qam' or 'psk' [default: 'qam'].

        - param.Rs: carrier baud rate [baud][default: 32e9].

        - param.SpS: samples per symbol [default: 16].

        - param.Nbits: total number of bits per carrier [default: 60000].

        - param.pulse: pulse shape ['nrz', 'rrc'][default: 'rrc'].

        - param.Ntaps: number of coefficients of the rrc filter [default: 4096].

        - param.alphaRRC: rolloff do rrc filter [default: 0.01].

        - param.Pch_dBm: launched power per WDM channel [dBm][default:-3 dBm].

        - param.Nch: number of WDM channels [default: 5].

        - param.Fc: central frequency of the WDM spectrum [Hz][default: 193.1e12 Hz].

        - param.lw: laser linewidth [Hz][default: 100 kHz].

        - param.freqSpac: frequency spacing of the WDM grid [Hz][default: 40e9 Hz].

        - param.Nmodes: number of polarization modes [default: 1].

    Returns
    -------
    sigTxWDM : np.array
        WDM signal.
    symbTxWDM : np.array
        Array of symbols per WDM carrier.
    param : optic.core.parameter object
        System parameters for the WDM transmitter.

    """
    # check input parameters
    param.M = getattr(param, "M", 16)
    param.constType = getattr(param, "constType", "qam")
    param.Rs = getattr(param, "Rs", 32e9)
    param.SpS = getattr(param, "SpS", 16)
    # Cen: number bits per frames
    param.NBpF = getattr(param, "NBpF", 2**18)
    param.Nframes = getattr(param, "Nframes", 3)
    param.pulse = getattr(param, "pulse", "rrc")
    param.Ntaps = getattr(param, "Ntaps", 4096)
    param.alphaRRC = getattr(param, "alphaRRC", 0.01)
    param.Pch_dBm = getattr(param, "Pch_dBm", -3)
    param.Nch = getattr(param, "Nch", 5)
    param.Fc = getattr(param, "Fc", 193.1e12)
    param.lw = getattr(param, "lw", 0)
    param.freqSpac = getattr(param, "freqSpac", 50e9)
    param.Nmodes = getattr(param, "Nmodes", 1)
    param.prgsBar = getattr(param, "prgsBar", True)

    # Cen: pilot related parameters
    param.pilotSeq = getattr(param, "pilotSeq", 512)
    # Cen: if phasePilot is 0, there is no phase pilot; otherwise, it is the interval of every phase pilot
    # the first pilot is inserted at the (pilotSeq + phasePilot - 1)-th symbol (symbol index starts from 0)
    param.phasePilot = getattr(param, "phasePilot", 32)

    # transmitter parameters
    Ts = 1 / param.Rs  # symbol period [s]
    Fs = 1 / (Ts / param.SpS)  # sampling frequency [samples/s]

    # central frequencies of the WDM channels
    freqGrid = (
        np.arange(-np.floor(param.Nch / 2), np.floor(param.Nch / 2) + 1, 1)
        * param.freqSpac
    )

    if (param.Nch % 2) == 0:
        freqGrid += param.freqSpac / 2

    if type(param.Pch_dBm) == list:
        assert (
            len(param.Pch_dBm) == param.Nch
        ), "list length of power per channel does not match number of channels."
        Pch = (
            10 ** (np.array(param.Pch_dBm) / 10) * 1e-3
        )  # optical signal power per WDM channel
    else:
        Pch = 10 ** (param.Pch_dBm / 10) * 1e-3
        Pch = Pch * np.ones(param.Nch)

    π = np.pi
    # time array
    Nbits = int(param.Nframes * param.NBpF)
    Nsymb = int(Nbits / np.log2(param.M))
    NSpF = int(param.NBpF/np.log2(param.M))

    t = np.arange(0, Nsymb * param.SpS)

    # allocate array
    sigTxWDM = np.zeros((len(t), param.Nmodes), dtype="complex")
    symbTxWDM = np.zeros(
        (len(t) // param.SpS, param.Nmodes, param.Nch), dtype="complex"
    )

    Psig = 0

    # constellation symbols info
    const = grayMapping(param.M, param.constType)
    Es = np.mean(np.abs(const) ** 2)

    # pulse shaping filter
    if param.pulse == "nrz":
        pulse = pulseShape("nrz", param.SpS)
    elif param.pulse == "rrc":
        pulse = pulseShape("rrc", param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=Ts)

    pulse = pulse / np.max(np.abs(pulse))

    for indCh in tqdm(range(param.Nch), disable=not (param.prgsBar)):
        logg.info(
            "channel %d\t fc : %3.4f THz" % (indCh, (param.Fc + freqGrid[indCh]) / 1e12)
        )

        Pmode = 0
        for indMode in range(param.Nmodes):
            logg.info(
                "  mode #%d\t power: %.2f dBm"
                % (indMode, 10 * np.log10((Pch[indCh] / param.Nmodes) / 1e-3))
            )

            # generate random bits
            # bitsTx = np.random.randint(2, size=param.Nbits)

            # map bits to constellation symbols
            # symbTx = modulateGray(bitsTx, param.M, param.constType)
            signals = SignalWithPilots(M=param.M,
                                       frame_len=NSpF,
                                       pilot_seq_len=param.pilotSeq,
                                       pilot_ins_rat=param.phasePilot,
                                       nframes=param.Nframes,
                                       nmodes=param.Nmodes,
                                       fb=20e9)
            symbTx = signals.symbTx()
            # normalize symbols energy to 1
            symbTx = symbTx / np.sqrt(Es)

            symbTxWDM[:, indMode, indCh] = symbTx

            # upsampling
            symbolsUp = upsample(symbTx, param.SpS)

            # pulse shaping
            sigTx = firFilter(pulse, symbolsUp)

            # optical modulation
            if indMode == 0:  # generate LO field with phase noise
                ϕ_pn_lo = phaseNoise(param.lw, len(sigTx), 1 / Fs)
                sigLO = np.exp(1j * ϕ_pn_lo)

            sigTxCh = iqm(sigLO, 0.5 * sigTx)
            sigTxCh = np.sqrt(Pch[indCh] / param.Nmodes) * pnorm(sigTxCh)

            sigTxWDM[:, indMode] += sigTxCh * np.exp(
                1j * 2 * π * (freqGrid[indCh] / Fs) * t
            )

            Pmode += signal_power(sigTxCh)

        Psig += Pmode

        logg.info(
            "channel %d\t power: %.2f dBm\n" % (indCh, 10 * np.log10(Pmode / 1e-3))
        )

    logg.info("total WDM signal power: %.2f dBm" % (10 * np.log10(Psig / 1e-3)))

    param.freqGrid = freqGrid