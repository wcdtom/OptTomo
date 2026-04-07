import numpy as np
from optic.comm.modulation import qamConst
from scipy.signal import fftconvolve
import logging as logg

class PilotWDMRX:
    def __init__(self, param, sigRX, refSymbs):
        self.sigRx = sigRX
        self.pilot = refSymbs

        # others
        param.seed = getattr(param, "seed", 42)
        param.probDist = getattr(param, "probDist", "uniform")
        param.shapingFactor = getattr(param, "shapingFactor", 0)
        param.mzmScale = getattr(param, "mzmScale", 0.5)
        param.laserLinewidth = getattr(param, "laserLinewidth", 100e3)
        # check input parameters
        param.M = getattr(param, "M", 16)
        param.constType = getattr(param, "constType", "qam")
        param.Rs = getattr(param, "Rs", 32e9)
        param.SpS = getattr(param, "SpS", 16)
        # Cen: number bits per frames
        param.NBpF = getattr(param, "NBpF", 2 ** 18)
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
        param.pilotM = getattr(param, "pilotM", 4)

        # Cen: synchronization training related parameters
        param.mu = getattr(param, "mu", 1e-3)
        param.initialWxy = getattr(param, "initialWxy", None)
        param.trainingSymb = getattr(param, "trainingSymb", None)
        param.eqSymb = getattr(param, "eqSymb", None)
        param.Niter = getattr(param, "Niter", 1)
        param.adaptiveOn = getattr(param, "adaptiveOn", True)
        param.avgoverMode = getattr(param, "avgoverMode", False)
        param.fftSize = getattr(param, "fftSize", 2 ** 16)
        param.syncTaps = getattr(param, "syncTaps", 4096)

        self.Nsymb = int(self.Nbits / np.log2(param.M))
        self.syncBool = True
        self.mu=param.mu

        if param.initialWxy is not None:
            init_W_xy = np.zeros((param.Ntaps, param.Nmodes, param.Nch))
            for i in range(param.Nmodes):
                for j in range(param.Nch):
                    init_W_xy[param.Ntaps // 2, i, j] = 1
            param.initialWxy = init_W_xy

    def _frame_sync(self, param):
        """
        Locate the pilot sequence frame

        Uses a CMA-based search scheme to located the initiial pilot sequence in
        the long data frame.

        Returns
        -------
        shift_factor: array_like
            location of frame start index per polarization
        foe_corse:  array_like
            corse frequency offset
        mode_sync_order: array_like
            Synced descrambled reference pattern order
        """
        # If synchronization fails, then change sync_bool to 'False'


        FRAME_SYNC_THRS = 120 # this is somewhat arbitrary but seems to work well

        ch_sync_order = np.zeros(param.Nch, dtype=int)
        not_found_modes = np.arange(0, param.Nmodes)
        search_overlap = 2 # fraction of pilot_sequence to overlap
        search_window = param.pilotSeq * param.SpS
        step = search_window // search_overlap
        # we only need to search the length of one frame*os plus some buffer (the extra step)
        num_steps = (self.Nsymb * param.SpS)//step + 1
        # Now search for every mode independent
        shift_factor = np.zeros(param.Nch, dtype=int)
        # Search based on equalizer error. Avoid one pilot_seq_len part in the beginning and
        # end to ensure that sufficient symbols can be used for the search
        sub_vars = np.ones((num_steps, param.Nch)) * 1e2
        W_xy = np.zeros((num_steps, param.Ntaps, param.Nmodes, param.ch), dtype=self.sigRx.dtype)
        w_xy = param.initialWxy
        foe_coarse = np.zeros((1,param.Nch))
        for i in np.arange(search_overlap, num_steps): # we avoid one step at the beginning
            tmp = self.sigRx[(i * step) : (i * step + search_window), :, :]
            w_xy, err_out = self._equalize_signal(tmp, w_xy, param)
            W_xy[i] = w_xy
            sub_vars[i, :] = np.var(err_out, axis=-1)
        # Lowest variance of the CMA error for each pol
        min_range = np.argmin(sub_vars, axis=-1)
        w_xy = W_xy[min_range]
        for c in range(param.Nch):
            idx_min = min_range[c]
            # Extract a longer sequence to ensure that the complete pilot sequence is found
            longSeq = self.sigRx[(idx_min*step-search_window): (idx_min * step + search_window),:,c]
            # Apply filter taps to the long sequence and remove coarse FO
            w_x1 = w_xy[:, :, c]
            symbs_out = self._apply_filter(longSeq, w_x1, param)
            symbs_out, foe_coarse[c] = self._compensate_freqOffset(symbs_out)
            # Check for pi/2 ambiguties and verify all
            max_phase_rot = np.zeros(param.Nch, dtype=np.float64)
            found_delay = np.zeros(param.Nch, dtype=np.int32)
            for ref_pol in not_found_modes:
                ix, dat, ii, ac = self._find_seqOffset(self.pilot[ref_pol], symbs_out[ref_pol], param)
                found_delay[ref_pol] = -ix
                max_phase_rot[ref_pol] = ac
            # Check for which mode found and extract the reference delay
            max_sync_pol = np.argmax(max_phase_rot)
            if max_phase_rot[max_sync_pol] < FRAME_SYNC_THRS: #
                logg.warning("Very low autocorrelation, likely the frame-sync failed")
                self.syncBool = False
            ch_sync_order[c] = max_sync_pol
            symb_delay = found_delay[max_sync_pol]
            # Remove the found reference mode
            not_found_modes = not_found_modes[not_found_modes != max_sync_pol]
            # New starting sample
            shift_factor[c] = (idx_min) * step + param.SpS * symb_delay - search_window
        # Important: the shift factors are arranged in the order of the signal modes, but
        # the mode_sync_order specifies how the signal modes need to be rearranged to match the pilots
        # therefore shift factors also need to be "mode_aligned"
        self.sigRx[:, :, :] = self.sigRx[ch_sync_order, :, :, ]
        shift_factor[shift_factor < 0] += self.Nsymb * param.SpS  # we don't really want negative shift factors
        self.shiftFactor = shift_factor[ch_sync_order]
        self.FOE = foe_coarse
        # add for compensate

    def _equalize_through_pilot(self, param):
        eq_shiftfctrs = np.array(self.shiftFactor, dtype=int)
        mu = np.atleast_1d(param.mu)
        if len(mu) == 1:  # use the same mu for both equaliser steps
            mu = np.repeat(mu, 2)
        if (abs(param.Ntaps - param.syncTaps) % 2) != 0:
            raise ValueError("Tap difference need to be an integer of the oversampling")
        elif param.Ntaps != param.syncTaps:
            eq_shiftfctrs = eq_shiftfctrs - (param.Ntaps - param.syncTaps) // 2 + param.SpS * self.Nsymb
        assert self.sigRx.shape[-1] - eq_shiftfctrs.max() > param.SpS * self.Nsymb, "You are trying to equalize an incomplete frame which does not work"

        taps_all, foe_all = self._equalize_pilotSeq(eq_shiftfctrs, param)

        out_sig = self._compensate_freqOffset(self.sigRx, foe_all)

        eq_mode_sig = self._apply_filter(out_sig, taps_all, param)


    def _equalize_pilotSeq(self, shift_fctrs, param):
                           # rx_signal, ref_symbs, , os, foe_comp=False, mu=(1e-4, 1e-4), M_pilot=4,
                           #     Ntaps=45, Niter=30,
                           #     adaptive_stepsize=True, methods=('cma', 'cma'), wxinit=None):
        """
        Equalise a pilot signal using the pilot sequence, with a two step equalisation.
        Parameters
        ----------
        rx_signal : array_like
            The received signal containing the pilots
        ref_symbs : array_like
            The reference symbols or pilot sequence
        shift_fctrs : array_like
            The indices where the pilot_sequence starts, typically this would come from the framesync.
        os : int
            Oversampling ratio
        foe_comp : bool, optional
            Whether to perform a foe inside the pilot equalisation. If yes we will first perform equalisation
            using methods[0], then do a pilot frequency recovery and then perform equalisation again. This can
            yield slightly higher performance. Currently this uses the average offset frequency of all modes.
        mu : tuple(float,float), optional
            Equalisaer steps sizes for methods[0] and methods[1]
        M_pilot : int, optional
            The QAM order of the pilots. By default we assume QPSK symbols
        Ntaps : int, optional
            The number of equalisation taps
        Niter : int, optional
            The number of iterations to do over the pilot sequence when training the filter
        adaptive_stepsize : bool, optional
            Whether to use an adapative step-size algorithm in the equaliser. Generally, you want to leave
            this one, because it allows for much shorter sequences.
        methods : tuple(string,string)
            The two methods to use in the equaliser
        wxinit : array_like, optional
            Filtertaps for initialisation of the filter. By default we generate typical filter taps.

        Returns
        -------
        out_taps : array_like
            Trained filter taps
        foe : array_like
            Offset frequency. Has the same number of modes as the signal, however is a single value only. If
            foe_comp was false, this are simply ones.
        """

        syms_out = np.zeros_like((self.pilot[0], param.Nch))
        for i in range(param.Nch):
            rx_sig_ch = self.sigRx[shift_fctrs[i]: shift_fctrs[i] + param.pilotSeq * param.SpS + param.Ntaps - 1, :, i]
            syms_out[:, i], w_x, err = self._equalize_signal(rx_sig_ch, param)
        # Run FOE and shift spectrum
        pilots = np.tile(self.pilot, (1, param.Nmodes))
        foe, foePerMode, cond = self._calculate_pilotFOE(syms_out, pilots)
        foe_all = np.ones(foePerMode.shape) * foe

        out_taps = w_x.copy()
        for i in range(param.Nch):
            rx_sig_ch = self.sigRx[shift_fctrs[i]: shift_fctrs[i] + param.pilotSeq * param.SpS + param.Ntaps - 1,:,i]
            rx_sig_ch = self._compensate_freqOffset(rx_sig_ch, param)
            # np.ones(foePerMode.shape) * foe, os=os)
            out_taps, err = self._equalize_signal(rx_sig_ch, param)

        return np.array(out_taps), foe_all

    def _calculate_pilotFOE(self, rec_symbs, pilot_symbs):
        """
        Frequency offset estimation for pilot-based DSP. Uses a transmitted pilot
        sequence to find the frequency offset from the corresponding aligned symbols.

        Gives higher accuracy than blind power of 4 based FFT for noisy signals.
        Calculates the phase variations between the batches and does a linear fit
        to find the corresponding frequency offset.

        Input:
            rec_symbs:  Complex symbols after initial Rx DSP
            pilot_symbs: Complex pilot symbols transmitted


        Output:
            foe:    Estimated FO in terms of complex phase. Average over all modes
            foePerMode: FO estimate for each mode
            condNum:   Condition number of linear fit. Gives accuracy of estimation

        """

        rec_symbs = np.atleast_2d(rec_symbs)
        pilot_symbs = np.atleast_2d(pilot_symbs)
        npols = rec_symbs.shape[0]

        condNum = np.zeros([1, npols])
        foePerMode = np.zeros([1, npols])

        # Search over all polarization
        for l in range(npols):
            phaseEvolution = np.unwrap(np.angle(pilot_symbs[:, l].conj() * rec_symbs[:, l]))

            # fit a first order polynomial to the unwrapped phase evolution
            freqFit = np.polyfit(np.arange(0, len(phaseEvolution)), phaseEvolution, 1)

            foePerMode[0, l] = freqFit[0] / (2 * np.pi)
            condNum[0, l] = freqFit[1]

        # Average over all modes used
        foe = np.mean(foePerMode)

        return foe, foePerMode, condNum


    def _equalize_signal(self, E, param):
     #os, mu, M, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma",
     #                   adaptive_stepsize=False, symbols=None, modes=None, apply=False, **kwargs):
        """
        Blind equalisation of PMD and residual dispersion, using a chosen equalisation method. The method can be any of the keys in the TRAINING_FCTS dictionary.

        Parameters
        ----------
        E    : array_like
            single or dual polarisation signal field (2D complex array first dim is the polarisation)

        os      : int
            oversampling factor

        mu      : float
            step size parameter

        M       : integer
            QAM order

        wxy     : array_like optional
            the wx and wy filter taps. Either this or Ntaps has to be given.

        Ntaps   : int
            number of filter taps. Either this or wxy need to be given. If given taps are initialised as [00100]

        TrSyms  : int, optional
            number of symbols to use for filter estimation. Default is None which means use all symbols.

        Niter   : int, optional
            number of iterations. Default is one single iteration

        method  : string, optional
            equaliser method has to be one of cma, mcma, rde, mrde, sbd, sbd_data, mddma, dd

        adaptive_stepsize : bool, optional
            whether to use an adaptive stepsize or a fixed

        symbols : array_like, optional
            array of coded symbols to decide on for dd-based equalisation functions (default=None, generate symbols for this
            QAM format)

        modes: array_like, optional
            array or list  of modes to  equalise over (default=None  equalise over all modes of the input signal)

        apply: Bool, optional
            whether to apply the filter taps and return the equalised signal

        Returns
        -------
        if apply:
            E : array_like
            equalised signal

        (wx, wy)    : tuple(array_like, array_like)
           equaliser taps for the x and y polarisation

        err       : array_like
           estimation error for x and y polarisation

        """
        param.trainingSymb = int(E.shape[0]//param.SpS//param.Ntaps-1)*int(param.Ntaps)
        symb_for_eq = self._recalculate_symbols(symb_for_eq=None)
        err, w_xy = self._train_equalizer(E, symb_for_eq.copy(), param.initialWxy, param)
        # copies are needed because pythran has problems with reshaped arrays
        E_est = self._apply_filter(E, w_xy, param)
        return err, w_xy, E_est


    def _calculate_pilotCPE(self):
        pass


    def _recalculate_symbols(self, symb_for_eq, param):
        #TODO: add support for cross-Qam
        if symb_for_eq is None:  # This code currently prevents passing "symbol arrays for RDE or CMA algorithms
            symb_for_eq = qamConst(param.M).flatten()

        symb_for_eq = np.tile(symb_for_eq, (param.Nmodes, 1))
        return symb_for_eq

    def _train_equalizer(self, E, symb_for_eq, w_xy, param):
        err = np.zeros((param.trainingSymb * param.Ntaps, param.Nmodes, param.Nch), dtype=E.dtype)
        # omp parallel for
        for ch in range(param.Nch):
            for iter in range(param.Niter):
                for i in range(param.trainingSymb):
                    X = E[i * param.SpS:i * param.SpS + param.Ntaps, :, ch]
                    Xest = self._apply_filter(X, w_xy[:,:, ch], param)
                    err[iter * param.trainingSymb + i, :, ch] = (symb_for_eq[0].real - abs(Xest)) ** 2 * Xest
                    w_xy[ch] += param.mu * np.conj(err[iter * param.trainingSymb + i, :, ch]) * X
                    if param.adaptiveOn and i > 0:
                        param.mu = self._adapt_step(param.mu, err[iter * param.trainingSymb + i, :, ch], err[iter * param.trainingSymb + i - 1, :, ch])
        return err, w_xy

    def _apply_filter(self, E, w_x, param):
        X_est = E.dtype.type(0)
        for k in range(param.Ntaps):
            for i in range(param.Nmodes):
                X_est += E[k, i] * np.conj(w_x[k, i])
        return X_est

    def _adapt_step(self, mu, err_p, err):
        if err.real * err_p.real > 0 and err.imag * err_p.imag > 0:
            return mu
        else:
            return mu / (1 + mu * (err.real * err.real + err.imag * err.imag))

    def _compensate_freqOffset(self, sig, param, freq_offset=None):
        """
        Find the frequency offset by searching in the spectrum of the signal
        raised to 4. Doing so eliminates the modulation for QPSK but the method also
        works for higher order M-QAM.

        Parameters
        ----------
            sig : array_line
                signal array with N modes
            param.avgoverModes : bool
                Using the field in all modes for estimation
            param.fftSize: array
                Size of FFT used to estimate. Should be power of 2, otherwise the
                next higher power of 2 will be used.

        Returns
        -------
            freq_offset : int
                found frequency offset

        """
        if freq_offset is None:
            if not ((np.log2(param.fftSize) % 2 == 0) | (np.log2(param.fftSize) % 2 == 1)):
                fft_size = 2 ** (int(np.ceil(np.log2(param.fftSize))))

            # Fix number of stuff
            sig = np.atleast_2d(sig)
            L, npols = sig.shape

            # Find offset for all modes
            freq_sig = np.zeros([fft_size, npols])
            for l in range(npols):
                freq_sig[:, l] = np.abs(np.fft.fft(sig[:, l] ** 4, fft_size)) ** 2

            # Extract corresponding FO
            freq_offset = np.zeros([2, npols])
            freq_vector = np.fft.fftfreq(fft_size, 1 / param.SpS) / 4
            for k in range(npols):
                max_freq_bin = np.argmax(np.abs(freq_sig[:, k]))
                freq_offset[0, k] = freq_vector[max_freq_bin]

            if param.avgoverMode:
                freq_offset = np.mean(freq_offset) * np.ones(freq_offset.shape)

        comp_signal = np.zeros([np.shape(sig)[0], param.Nmodes], dtype=sig.dtype)
        # Fix output
        sig_len = len(sig[:, 0])
        time_vec = np.arange(1, sig_len + 1, dtype=float)
        for l in range(param.Nmodes):
            lin_phase = 2 * np.pi * time_vec * freq_offset[l] / param.SpS
            comp_signal[:, l] = sig[:, l] * np.exp(-1j * lin_phase)

        return comp_signal, freq_offset

    def _find_seqOffset(self, x, y):
        """
        Find the offset of one sequence in the other even if both sequences are complex.

        Parameters
        ----------
        x : array_like
            transmitted data sequence
        y : array_like
            received data sequence

        Returns
        -------
        idx : integer
            offset index
        y : array_like
            y array possibly rotated to correct 1.j**i for complex arrays
        ii : integer
            power for complex rotation angle 1.j**ii
        """
        acm = 0.
        for i in range(4):
            rx = y * 1.j ** i
            if np.issubdtype(x.dtype, np.complexfloating):
                ac = fftconvolve(x, rx.conj()[::-1], 'full')
            else:
                ac = fftconvolve(x, rx[::-1], 'full')
            idx = abs(ac).argmax() - (rx.shape[0] - 1)  # this is necessary to find the correct position (size of full is N_X+N_Y-1)
            act = ac.real.max()
            if act > acm:
                ii = i
                ix = idx
                acm = act
        return ix, y * 1.j ** ii, ii, acm

    def _correct_FOE(self):
        pass