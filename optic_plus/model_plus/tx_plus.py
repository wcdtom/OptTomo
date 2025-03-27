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


class pilotWDMTx:
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
    def __init__(self, param, modulated=True):
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
        self.Ts = 1 / param.Rs  # symbol period [s]
        self.Fs = 1 / (self.Ts / param.SpS)  # sampling frequency [samples/s]

        # central frequencies of the WDM channels
        self.freqGrid = (
            np.arange(-np.floor(param.Nch / 2), np.floor(param.Nch / 2) + 1, 1)
            * param.freqSpac
        )

        if (param.Nch % 2) == 0:
            self.freqGrid += param.freqSpac / 2

        if type(param.Pch_dBm) == list:
            assert (
                len(param.Pch_dBm) == param.Nch
            ), "list length of power per channel does not match number of channels."
            self.Pch = (
                10 ** (np.array(param.Pch_dBm) / 10) * 1e-3
            )  # optical signal power per WDM channel
        else:
            self.Pch = 10 ** (param.Pch_dBm / 10) * 1e-3
            self.Pch = self.Pch * np.ones(param.Nch)
        self.Psig = 0.0
        self.Pch_launch = np.zeros(param.Nch)

        # time array
        self.Nbits = int(param.Nframes * param.NBpF)
        self.Nsymb = int(self.Nbits / np.log2(param.M))
        self.NSpF = int(param.NBpF/np.log2(param.M))

        t = np.arange(0, self.Nsymb * param.SpS)

        # allocate array
        self.pulseTxWDM = np.zeros(
            (len(t), param.Nmodes, param.Nch), dtype="complex"
        )
        self.symbTxWDM = np.zeros(
        (len(t) // param.SpS, param.Nmodes, param.Nch), dtype="complex"
        )
        self.sigTxWDM = np.zeros((len(t), param.Nmodes), dtype="complex")
        self.Psig = 0

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
                "channel %d\t fc : %3.4f THz" % (indCh, (param.Fc + self.freqGrid[indCh]) / 1e12)
            )

            self.Pmode = 0
            for indMode in range(param.Nmodes):
                logg.info(
                    "  mode #%d\t power: %.2f dBm"
                    % (indMode, 10 * np.log10((self.Pch[indCh] / param.Nmodes) / 1e-3))
                )

                # generate random bits
                signals = SignalWithPilots(M=param.M,
                                           frame_len=self.NSpF,
                                           pilot_seq_len=param.pilotSeq,
                                           pilot_ins_rat=param.phasePilot,
                                           nframes=param.Nframes,
                                           nmodes=param.Nmodes,
                                           fb=20e9)
                symbTx = signals.symbTx()
                # normalize symbols energy to 1
                symbTx = symbTx / np.sqrt(Es)

                self.symbTxWDM[:, indMode, indCh] = symbTx

                # upsampling
                symbolsUp = upsample(symbTx, param.SpS)

                # pulse shaping
                sigTx = firFilter(pulse, symbolsUp)

                self.pulseTxWDM[:, indMode, indCh] = sigTx

        if modulated:
            # optical modulation
            self._modulate(param)

    def _modulate(self, param):
        # generate LO field with phase noise
        (length, modes, channels) = self.pulseTxWDM.shape
        ϕ_pn_lo = phaseNoise(param.lw, length, 1 / self.Fs)
        sigLO = np.exp(1j * ϕ_pn_lo)

        for indCh in range(channels):
            Pmode = 0
            for indMode in range(modes):
                sigTx = self.pulseTxWDM[:, indMode, indCh]

                sigTxCh = iqm(sigLO, 0.5 * sigTx)
                sigTxCh = np.sqrt(self.Pch[indCh] / param.Nmodes) * pnorm(sigTxCh)

                self.sigTxWDM[:, indMode] += sigTxCh * np.exp(
                   1j * 2 * np.pi * (self.freqGrid[indCh] / self.Fs) * np.arange(0, self.Nsymb * param.SpS)
                )

            Pmode += signal_power(sigTxCh)
            self.Pch_launch[indCh] = 10 * np.log10(Pmode / 1e-3)
            logg.info(
                "channel %d\t power: %.2f dBm\n" % (indCh, 10 * np.log10(Pmode / 1e-3))
            )
        self.Psig += Pmode
        logg.info("total WDM signal power: %.2f dBm" % (10 * np.log10(self.Psig / 1e-3)))


    def _digital_subcarrier_modulate(self, param):
        pass


    def reinput_pulse(self):
        pass
