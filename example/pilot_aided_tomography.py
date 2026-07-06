"""
pilot_aided_tomography.py
=============
Same Tx/channel/conventional DSP flow as ref_aided_tomography.py, but the tomography
stage is driven by PilotAidedeRP1 with phase-pilot CPR.

PilotAidedeRP1 starts from pilotWDMRx rxSyncFO, trains the FSE with known 16QAM Tx
symbols, uses phase pilots for CPR, removes pilot/phase-pilot samples through
dataSymbolMask, and then reuses the RefAidedeRP1 G-matrix and gamma
solver.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from optic.models.modelsGPU import manakovSSF
except Exception:
    from optic.models.channels import manakovSSF

import ref_aided_tomography as base
from optic.utils import parameters
from optic_plus.models_plus.rx import PilotAidedeRP1, pilotWDMRx
from optic_plus.models_plus.tx_plus import pilotWDMTx


OUT_DIR = Path("/Users/yamaima/PycharmProjects/OptTomo/Results/pilot_aided_tomography")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def no_reference_alignment(ppe, sig_tx, sig_rx):
    return sig_tx, sig_rx


def make_tomo_test_run_params():
    paramRun = parameters()
    paramRun.lambdaReg = 1e-2
    paramRun.normalize = base.NORMALIZE_LIKE_LEGACY
    paramRun.verbose = True
    paramRun.plot = False
    paramRun.allModes = True
    paramRun.FsTx = base.FS_TOMO
    paramRun.FsRx = base.FS_TOMO
    paramRun.tomoSpS = base.SPS_OUT
    paramRun.tomoStartSymbol = base.paramTx.pilotSeq
    paramRun.tomoTrainSymbols = int(1e5)
    paramRun.tomoNormalize = True
    paramRun.tomoDataMode = "compact"
    paramRun.tomoRequireFullDataRun = False
    paramRun.tomoEqSpSout = base.SPS_OUT
    paramRun.tomoEqNTaps = base.paramRx.nTaps
    paramRun.tomoEqMu = base.paramRx.mu
    paramRun.tomoEqTrainSymbols = int(5e4)
    paramRun.tomoEqDD = False
    paramRun.tomoEqM = base.paramTx.M
    paramRun.tomoEqVerbose = True
    paramRun.tomoPilotSeq = base.paramTx.pilotSeq
    paramRun.tomoPhasePilot = base.paramTx.phasePilot
    paramRun.tomoCprUseInitialPilot = False
    paramRun.diagnoseReference = True
    return paramRun


def main():
    base.OUT_DIR = OUT_DIR
    base.paramTx.phasePilot = 100
    base.prepare_reference_signals = no_reference_alignment

    t0 = time.time()
    print("[pilot_aided_tomography] Single-channel dual-pol Manakov RP1 using PilotAidedeRP1")
    print(f"[pilot_aided_tomography] phasePilot={base.paramTx.phasePilot}, "
          "PilotAidedeRP1 CPR uses phase pilots only")
    print(f"[pilot_aided_tomography] Tx optical SpS={base.SPS}, DSP/Tomo SpS={base.SPS_OUT}, "
          f"Fs_tomo={base.FS_TOMO:.6e} Hz")

    print("\n[pilot_aided_tomography] Generating Tx")
    transmitter = pilotWDMTx(base.paramTx)
    sig_tx_wdm = np.asarray(transmitter.sigTxWDM)
    print(f"[pilot_aided_tomography] sigTxWDM shape = {sig_tx_wdm.shape}")

    print("\n[pilot_aided_tomography] Running manakovSSF")
    sig_wdm = manakovSSF(sig_tx_wdm, base.paramCh)
    print(f"[pilot_aided_tomography] sigWDM after fiber shape = {sig_wdm.shape}")

    print("\n[pilot_aided_tomography] Running pilotWDMRx DSP")
    receiver = pilotWDMRx(transmitter, sig_wdm, base.paramRx)
    ch_index = int(np.floor(base.paramTx.nChannels / 2))
    results = receiver.run(channels=[ch_index], plot=False, verbose=True)
    dsp = results[ch_index]
    dsp["NSpF"] = transmitter.NSpF

    tomo = PilotAidedeRP1(base.make_tomo_params())

    print(f"[pilot_aided_tomography] DSP metrics: BER={dsp['BER']}, SNR={dsp['SNR']}, "
          f"delay={dsp['delay']}, FO={dsp['fo_est'] / 1e6:.3f} MHz")

    print("\n[pilot_aided_tomography] Running PilotAidedeRP1 on rxSyncFO with known-data FSE")
    gamma_est, g_mat = tomo.run(dsp, param=make_tomo_test_run_params())
    sig_tx_run = tomo._lastRunSigTx
    sig_rx_run = tomo._lastRunSigRx
    p_tx = tomo._lastRunPowerTx

    print(f"\n[pilot_aided_tomography] Tomography Tx shape = {sig_tx_run.shape}")
    print(f"[pilot_aided_tomography] Tomography Rx shape = {sig_rx_run.shape}")
    fit_err = base.waveform_fit_error(tomo, sig_tx_run, sig_rx_run,
                                      gamma_est, g_mat)
    print(f"[pilot_aided_tomography] Waveform fit error = {fit_err:.4e}")

    base.save_plots(tomo, gamma_est, g_mat, sig_tx_run, sig_rx_run, p_tx, dsp)

    print(f"\n[pilot_aided_tomography] elapsed: {time.time() - t0:.2f} s")
    print(f"[pilot_aided_tomography] results saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
