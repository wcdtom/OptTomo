# 3rd party library
import numpy as np
from tqdm.notebook import tqdm

from optic.dsp.core import (phaseNoise, pnorm, pulseShape,
                            signalPower, upsample)
from optic.models.devices import iqm
from optic.comm.modulation import grayMapping, modulateGray

try:
    from optic.dsp.coreGPU import firFilter
except ImportError:
    from optic.dsp.core import firFilter

from optic.comm.sources import symbolSource
from optic.utils import parameters

# sys library
import logging as logg


class pilotWDMTx:
    """
     Implement a pilot-inserted WDM transmitter.

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
        ## Others
        param.seed = getattr(param, "seed", 42)
        param.prgsBar = getattr(param, "prgsBar", True)

        ## Symbols related
        param.probDist = getattr(param, "probDist", "uniform")
        param.shapingFactor = getattr(param, "shapingFactor", 0)
        param.M = getattr(param, "M", 16)
        param.constType = getattr(param, "constType", "qam")
        param.Rs = getattr(param, "Rs", 32e9)
        param.nBitsPerFrame = getattr(param, "nBitsPerFrame", 2 ** 18)  # Cen: number bits per frames
        param.nFrames = getattr(param, "nFrames", 5)

        ## Pulse related
        param.pulseType = getattr(param, "pulseType", "rrc")
        param.nFilterTaps = getattr(param, "nFilterTaps", 1024)
        param.pulseRollOff = getattr(param, "pulseRollOff", 0.01)
        param.alphaRRC = getattr(param, "alphaRRC", 0.01)
        param.SpS = getattr(param, "SpS", 16)

        ## Channel related
        param.powerPerChannel = getattr(param, "powerPerChannel", -3)
        param.nChannels = getattr(param, "nChannels", 5)
        param.Fc = getattr(param, "Fc", 193.1e12)
        param.laserLinewidth = getattr(param, "laserLinewidth", 100e3)
        param.wdmGridSpacing = getattr(param, "wdmGridSpacing", 50e9)
        param.nPolModes = getattr(param, "nPolModes", 1)

        ## Modulator related
        param.mzmScale = getattr(param, "mzmScale", 0.5)

        ## Cen: pilot related parameters
        param.pilotSeq = getattr(param, "pilotSeq", 512)
        # Cen: if phasePilot is 0, there is no phase pilot; otherwise, it is the interval of every phase pilot
        # the first pilot is inserted at the (pilotSeq + phasePilot - 1)-th symbol (symbol index starts from 0)
        param.phasePilot = getattr(param, "phasePilot", 32)
        param.pilotM = getattr(param, "pilotM", 4)

        # store param object for downstream access (e.g. PilotWDMReceiver)
        self.param = param

        # transmitter parameters
        self.Ts = 1 / param.Rs  # symbol period [s]
        self.Fs = 1 / (self.Ts / param.SpS)  # sampling frequency [samples/s]

        # central frequencies of the WDM channels
        if param.nChannels % 2 == 1:
            self.freqGrid = (
                    np.arange(-np.floor(param.nChannels / 2), np.floor(param.nChannels / 2) + 1)
                    * param.wdmGridSpacing
            )
        else:
            self.freqGrid = (
                    np.arange(-(param.nChannels / 2 - 0.5), param.nChannels / 2)
                    * param.wdmGridSpacing
            )

        if type(param.powerPerChannel) == list:
            assert (
                    len(param.powerPerChannel) == param.nChannels
            ), "list length of power per channel does not match number of channels."
            self.Pch = (
                    10 ** (np.array(param.powerPerChannel) / 10) * 1e-3
            )  # optical signal power per WDM channel
        else:
            self.Pch = 10 ** (param.powerPerChannel / 10) * 1e-3
            self.Pch = self.Pch * np.ones(param.nChannels)
        self.Psig = 0.0
        self.Pch_launch = np.zeros(param.nChannels)

        # time array
        self.Nbits = int(param.nFrames * param.nBitsPerFrame)
        self.NSpF, self.Nsymb = self._cal_nSymbols(param)

        t = np.arange(0, self.Nsymb * param.SpS)

        # allocate array
        self.pulseTxWDM = np.zeros(
            (len(t), param.nPolModes, param.nChannels), dtype="complex"
        )
        self.symbTxWDM = np.zeros(
        (self.Nsymb, param.nPolModes, param.nChannels), dtype="complex"
        )
        self.sigTxWDM = np.zeros((len(t), param.nPolModes), dtype="complex")

        self.Psig = 0

        # constellation symbols info
        const = grayMapping(param.M, param.constType)
        Es = np.mean(np.abs(const) ** 2)

        paramPulse = parameters()
        paramPulse.pulseType = param.pulseType
        paramPulse.nFilterTaps = param.nFilterTaps
        paramPulse.rollOff = param.pulseRollOff
        paramPulse.SpS = param.SpS

        # pulse shaping filter
        pulse = pulseShape(paramPulse)

        pulse = pulse / np.max(np.abs(pulse))

        idx, idx_data, idx_pilot = self._cal_pilot_idx(param)
        paramPilot = parameters()
        paramPilot.nSymbols = np.count_nonzero(idx_pilot) #nSymbolsPerFrame
        paramPilot.M = param.pilotM
        paramPilot.constType = param.constType
        paramPilot.dist = param.probDist
        paramPilot.shapingFactor = param.shapingFactor
        self.pilot = np.zeros(
            (paramPilot.nSymbols, param.nPolModes, param.nChannels), dtype="complex"
        )
        self.pilotOnPosition = np.zeros(
            (self.Nsymb, param.nPolModes, param.nChannels), dtype="complex"
        )
        self.dataOnPosition = np.zeros(
            (self.Nsymb, param.nPolModes, param.nChannels), dtype="complex"
        )
        self.pulseSampledPilot = np.zeros(
            (len(t), param.nPolModes, param.nChannels), dtype="complex"
        )
        self.pulseSampledData = np.zeros(
            (len(t), param.nPolModes, param.nChannels), dtype="complex"
        )


        paramSymb = parameters()
        paramSymb.nSymbols = np.count_nonzero(idx_data) # nSymbolsPerFrame
        paramSymb.M = param.M
        paramSymb.constType = param.constType
        paramSymb.dist = param.probDist
        paramSymb.shapingFactor = param.shapingFactor


        for indCh in tqdm(range(param.nChannels), disable=not (param.prgsBar)):
            logg.info(
                "channel %d\t fc : %3.4f THz" % (indCh, (param.Fc + self.freqGrid[indCh]) / 1e12)
            )

            self.Pmode = 0
            for indMode in range(param.nPolModes):
                paramPilot.seed = param.seed + 100000 + 1000 * indCh + indMode
                paramSymb.seed = param.seed + 200000 + 1000 * indCh + indMode
                logg.info(
                    "  mode #%d\t power: %.2f dBm"
                    % (indMode, 10 * np.log10((self.Pch[indCh] / param.nPolModes) / 1e-3))
                )
                symbTx = np.zeros(self.Nsymb, dtype="complex")
                singleCirclePilots = symbolSource(paramPilot)
                self.pilot[:, indMode, indCh] = singleCirclePilots
                _idx_pilot_frames = np.tile(idx_pilot, param.nFrames)
                symbTx[_idx_pilot_frames] = np.tile(singleCirclePilots, param.nFrames)
                _symbDataTx = []
                for indFrame in range(param.nFrames):
                    _symbDataTx.append(symbolSource(paramSymb))
                _idx_data_frames = np.tile(idx_data, param.nFrames)
                symbTx[_idx_data_frames] = np.concatenate(_symbDataTx)

                # normalize symbols energy to 1
                symbTx = symbTx / np.sqrt(Es)
                symbTx = symbTx.reshape(-1)

                self.pilotOnPosition[_idx_pilot_frames, indMode, indCh] = symbTx[_idx_pilot_frames]
                self.dataOnPosition[_idx_data_frames, indMode, indCh] = symbTx[_idx_data_frames]
                self.symbTxWDM[:, indMode, indCh] = symbTx

                # upsampling
                symbolsUp = upsample(symbTx, param.SpS)

                # pulse shaping
                sigTx = firFilter(pulse, symbolsUp)

                _idx_pilot_pulse_frames = np.repeat(_idx_pilot_frames, param.SpS)
                _idx_data_pulse_frames = np.repeat(_idx_data_frames, param.SpS)

                self.pulseTxWDM[:, indMode, indCh] = sigTx
                self.pulseSampledPilot[_idx_pilot_pulse_frames, indMode, indCh] = sigTx[_idx_pilot_pulse_frames]
                self.pulseSampledData[_idx_data_pulse_frames, indMode, indCh] = sigTx[_idx_data_pulse_frames]

        if modulated:
            # optical modulation
            self._modulate(param)

    def _modulate(self, param):
        # generate LO field with phase noise
        (length, modes, channels) = self.pulseTxWDM.shape
        ϕ_pn_lo = phaseNoise(param.laserLinewidth, length, 1 / self.Fs)
        sigLO = np.exp(1j * ϕ_pn_lo)

        for indCh in range(channels):
            Pmode = 0
            for indMode in range(modes):
                sigTx = self.pulseTxWDM[:, indMode, indCh]

                sigTxCh = iqm(sigLO, param.mzmScale * sigTx)
                sigTxCh = np.sqrt(self.Pch[indCh] / param.nPolModes) * pnorm(sigTxCh)

                self.sigTxWDM[:, indMode] += sigTxCh * np.exp(
                   1j * 2 * np.pi * (self.freqGrid[indCh] / self.Fs) * np.arange(0, self.Nsymb * param.SpS)
                )

            Pmode += signalPower(sigTxCh)
            self.Pch_launch[indCh] = 10 * np.log10(Pmode / 1e-3)
            logg.info(
                "channel %d\t power: %.2f dBm\n" % (indCh, 10 * np.log10(Pmode / 1e-3))
            )
        self.Psig += Pmode
        logg.info("total WDM signal power: %.2f dBm" % (10 * np.log10(self.Psig / 1e-3)))

    def _cal_pilot_idx(self, param):
        idx = np.arange(self.NSpF)
        idx_pil_seq = idx < param.pilotSeq
        if param.phasePilot == 0 or param.phasePilot is None:
            idx_pil = idx_pil_seq
        else:
            if (self.NSpF - param.pilotSeq) % param.phasePilot != 0:
                raise ValueError("Frame without pilot sequence divided by pilot rate needs to be an integer")
            N_ph_frames = (self.NSpF - param.pilotSeq) // param.phasePilot
            idx_ph_pil = ((idx - param.pilotSeq) % param.phasePilot != 0) & (idx - param.pilotSeq > 0)
            idx_ph_pil[param.pilotSeq] = ~ idx_ph_pil[param.pilotSeq]
            idx_pil = ~idx_ph_pil  # ^ idx_pil_seq
        idx_dat = ~idx_pil
        return idx, idx_dat, idx_pil

    def _cal_nSymbols(self, param):
        dataSymbPerFrame = int(param.nBitsPerFrame / np.log2(param.M))
        if param.phasePilot == 0 or param.phasePilot is None:
            phasePilotPerFrame = 0
        else:
            phasePilotPerFrame = int(dataSymbPerFrame // param.phasePilot)
        nSymbPerFrame = param.pilotSeq + dataSymbPerFrame + phasePilotPerFrame
        nSymbols = nSymbPerFrame * param.nFrames
        return nSymbPerFrame, nSymbols

    def _digital_subcarrier_modulate(self, param):
        pass



