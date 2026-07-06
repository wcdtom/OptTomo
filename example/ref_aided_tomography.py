"""
ref_aided_tomography.py
===========
Single-channel, dual-polarization Manakov RP1 validation after receiver DSP.

This script follows the main flow of test0518.py, but inserts the full
pilotWDMRx DSP chain before RefAidedeRP1:

    DSP diagnostics:
        pilotWDMTx -> manakovSSF -> pilotWDMRx traditional DSP
        (pilot sync, mimoLMSFSE, CPR, metrics and constellation)

    Tomography:
        pilotWDMRx rxSyncFO -> matched Tx reference waveform regression with
        mimoAdaptEqualizer(SpS=1), then RefAidedeRP1(afterDSP=True)
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from optic.models.modelsGPU import manakovSSF
except Exception:
    from optic.models.channels import manakovSSF

from optic.utils import parameters
from optic.dsp.core import pnorm
from optic_plus.models_plus.rx import RefAidedeRP1, pilotWDMRx
from optic_plus.models_plus.tx_plus import pilotWDMTx


# ─── Experiment switches ────────────────────────────────────────────────────
NORMALIZE_LIKE_LEGACY = True
SHOW_PLOTS = False

OUT_DIR = Path("/Users/yamaima/PycharmProjects/OptTomo/Results/ref_aided_tomography")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Physical / simulation parameters ───────────────────────────────────────
LTOTAL_KM = 300
LSPAN_KM = 50
DELTA_Z_KM = 1

D = 16
ALPHA_DB = 0.20
GAMMA0 = 1.3
FC = 193.1e12

RS = 100e9
SPS = 4
SPS_OUT = 2
FS_TX = RS * SPS
FS_TOMO = RS * SPS_OUT


# ─── Tx parameters: single channel, dual polarization ───────────────────────
paramTx = parameters()
paramTx.pilotSeq = 2048
paramTx.phasePilot = 0
paramTx.pilotM = 4
paramTx.M = 16
paramTx.Rs = RS
paramTx.SpS = SPS
paramTx.pulseType = "rrc"
paramTx.nFilterTaps = 1024
paramTx.pulseRollOff = 0.1
paramTx.powerPerChannel = 1.5
paramTx.nChannels = 3
paramTx.Fc = FC
paramTx.laserLinewidth = 0
paramTx.wdmGridSpacing = 125e9
paramTx.nPolModes = 2
paramTx.nBitsPerFrame = int(np.log2(paramTx.M) * 1e5)
paramTx.nFrames = 1
paramTx.seed = 123
paramTx.prgsBar = True


# ─── Fiber/channel parameters ───────────────────────────────────────────────
paramCh = parameters()
paramCh.Ltotal = LTOTAL_KM
paramCh.Lspan = LSPAN_KM
paramCh.alpha = ALPHA_DB
paramCh.D = D
paramCh.gamma = GAMMA0
paramCh.Fc = FC
paramCh.amp = "edfa"
paramCh.hz = 0.05
paramCh.maxIter = 5
paramCh.tol = 1e-5
paramCh.nlprMethod = True
paramCh.maxNlinPhaseRot = 2e-2
paramCh.prgsBar = True
paramCh.Fs = FS_TX
paramCh.seed = 456


# ─── Receiver DSP parameters ────────────────────────────────────────────────
paramRx = parameters()
paramRx.paramCh = paramCh
paramRx.channels = "center"
paramRx.FO = 0
paramRx.polRotation = 0
paramRx.nTaps = 61
paramRx.mu = 1e-3
paramRx.nTrain = paramTx.pilotSeq
paramRx.dd = False
paramRx.SpSout = SPS_OUT
paramRx.eqSpSout = SPS_OUT
paramRx.cprSpS = SPS_OUT
paramRx.syncSearchMode = "best"
paramRx.pdIdeal = True
paramRx.loLw = 0
paramRx.loP = 10
paramRx.eqVerbose = True
paramRx.evalDiscard = 500


def make_tomo_params():
    paramTomo = parameters()
    paramTomo.Fs = FS_TOMO
    paramTomo.Rs = RS
    paramTomo.Ltotal = LTOTAL_KM
    paramTomo.Lspan = LSPAN_KM
    paramTomo.D = D
    paramTomo.alpha = ALPHA_DB
    paramTomo.gamma = GAMMA0
    paramTomo.Fc = FC
    paramTomo.afterDSP = True
    paramTomo.deltaZ = DELTA_Z_KM
    return paramTomo


def make_run_params():
    paramRun = parameters()
    paramRun.lambdaReg = 1e-2
    paramRun.normalize = NORMALIZE_LIKE_LEGACY
    paramRun.verbose = True
    paramRun.plot = False
    paramRun.allModes = True
    paramRun.FsTx = FS_TOMO
    paramRun.FsRx = FS_TOMO
    paramRun.waveformRegression = True
    paramRun.regressionSpS = SPS_OUT
    paramRun.regressionStartSymbol = paramTx.pilotSeq
    paramRun.regressionTrainSymbols = int(1e5)
    paramRun.regressionNTaps = 15
    paramRun.regressionNumIter = 2
    paramRun.regressionMu = 1e-3
    paramRun.regressionM = paramTx.M
    paramRun.regressionNormalize = True
    paramRun.regressionRequireFullDataRun = True
    return paramRun


def gamma_theory(ppe):
    return np.array([
        np.exp(-ppe.alphaTomo * z +
               int(np.floor(z / ppe.Lspan)) * ppe.Lspan * ppe.alphaTomo)
        for z in ppe.zTomoBank
    ])


def optcommpy_gamma_postprocess(z, gamma_norm, theory):
    gamma_abs = np.abs(gamma_norm)
    window_size = 3
    gamma_smooth = np.convolve(
        gamma_abs, np.ones(window_size) / window_size, mode="same"
    )
    gamma_smooth[0] = gamma_abs[0]
    gamma_smooth[-1] = gamma_abs[-1]

    eval_idx = int(10.0 / DELTA_Z_KM)
    est_vec = gamma_smooth[eval_idx:]
    theo_vec = theory[eval_idx:]
    denom = np.dot(est_vec, est_vec)
    optimal_scale = np.dot(est_vec, theo_vec) / denom if denom > 0 else 1.0
    gamma_safe = np.maximum(gamma_smooth * optimal_scale, 1e-10)

    for i in range(eval_idx):
        weight = i / eval_idx
        gamma_safe[i] = (1 - weight) * theory[i] + weight * gamma_safe[i]

    rms_error = np.sqrt(np.mean(
        (10 * np.log10(gamma_safe[eval_idx:])
         - 10 * np.log10(theory[eval_idx:])) ** 2
    ))
    return gamma_safe, gamma_smooth, optimal_scale, rms_error


def prepare_reference_signals(ppe, sig_tx, sig_rx):
    return ppe._alignSignals(sig_tx, sig_rx, phaseAlign=False, verbose=False)


def waveform_fit_error(ppe, sig_tx, sig_rx, gamma_est, g_mat):
    sig_tx, sig_rx = prepare_reference_signals(ppe, sig_tx, sig_rx)
    a0 = ppe._linearCD(sig_tx, LTOTAL_KM)
    al = ppe.cdReload(sig_rx)
    if getattr(ppe, "_lastAugmentedX", None) is not None:
        x_vec = ppe._lastAugmentedX
        c_factor = ppe._lastAugmentedCFactor
        recon = c_factor * a0
        for m in range(sig_tx.shape[1]):
            recon[:, m] += g_mat[m].T @ x_vec[:-1]
        return np.linalg.norm(al - recon) / max(np.linalg.norm(al), 1e-30)

    a1 = al - a0
    recon = np.zeros_like(a1)
    for m in range(sig_tx.shape[1]):
        recon[:, m] = g_mat[m].T @ gamma_est
    return np.linalg.norm(a1 - recon) / max(np.linalg.norm(a1), 1e-30)


def save_plots(ppe, gamma_est, g_mat, sig_tx, sig_rx, p_tx, dsp_result):
    sig_tx, sig_rx = prepare_reference_signals(ppe, sig_tx, sig_rx)
    theory = gamma_theory(ppe)
    gamma_norm = gamma_est / GAMMA0
    if NORMALIZE_LIKE_LEGACY:
        gamma_norm = gamma_norm / p_tx
    gamma_safe, gamma_smooth, optimal_scale, rms_error = \
        optcommpy_gamma_postprocess(ppe.zTomoBank, gamma_norm, theory)

    a0 = ppe._linearCD(sig_tx, LTOTAL_KM)
    al = ppe.cdReload(sig_rx)
    a1 = al - a0
    if getattr(ppe, "_lastAugmentedX", None) is not None:
        x_vec = ppe._lastAugmentedX
        c_factor = ppe._lastAugmentedCFactor
        recon = c_factor * a0
        nonlinear_recon = np.zeros_like(a1)
        for m in range(sig_tx.shape[1]):
            nonlinear_recon[:, m] = g_mat[m].T @ x_vec[:-1]
            recon[:, m] += nonlinear_recon[:, m]
    else:
        nonlinear_recon = np.zeros_like(a1)
        for m in range(sig_tx.shape[1]):
            nonlinear_recon[:, m] = g_mat[m].T @ gamma_est
        recon = a0 + nonlinear_recon

    np.savez(
        OUT_DIR / "ref_aided_tomography_dsp_cdreload_tomography.npz",
        z=ppe.zTomoBank,
        gammaEst=gamma_est,
        gammaNorm=gamma_norm,
        gammaSmooth=gamma_smooth,
        gammaOptcommpyStyle=gamma_safe,
        gammaTheory=theory,
        optimalScale=optimal_scale,
        rmsErrorDb=rms_error,
        sigTxPower=p_tx,
        sigTx=sig_tx,
        sigRxDsp=sig_rx,
        A0=a0,
        AL=al,
        A1=a1,
        recon=recon,
        delay=dsp_result["delay"],
        foEst=dsp_result["fo_est"],
        BER=dsp_result["BER"],
        SER=dsp_result["SER"],
        SNR=dsp_result["SNR"],
        GMI=dsp_result["GMI"],
        NGMI=dsp_result["NGMI"],
        Ltotal=LTOTAL_KM,
        Lspan=LSPAN_KM,
        deltaZ=DELTA_Z_KM,
        Fs=FS_TOMO,
        SpSout=SPS_OUT,
    )

    path_a = np.asarray(dsp_result.get("y_Eval", dsp_result["y_CPR"]))
    if path_a.shape[0] != dsp_result["y_CPR"].shape[0]:
        path_a = path_a
    elif SPS_OUT > 1:
        path_a = path_a[::SPS_OUT, :]

    fig0, axs = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 2]}
    )
    axs[0].set_title(
        "Rx Constellation (pilotWDMRx DSP diagnostics)\n"
        f"SNR_X: {dsp_result['SNR'][0]:.1f}dB, "
        f"SNR_Y: {dsp_result['SNR'][1]:.1f}dB"
    )
    plot_pts = min(10000, len(path_a))
    axs[0].hist2d(path_a[:plot_pts, 0].real, path_a[:plot_pts, 0].imag,
                  bins=100, cmap="inferno", density=True)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("In-Phase (I)")
    axs[0].set_ylabel("Quadrature (Q)")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    axs[1].plot(ppe.zTomoBank, theory, "k--", linewidth=2,
                label=r"Theory $\gamma(z)$")
    axs[1].plot(ppe.zTomoBank, gamma_safe, "r-", linewidth=1.5,
                label=r"Estimated $\gamma(z)$")
    axs[1].set_xlabel("Distance (km)")
    axs[1].set_ylabel("Normalized Power")
    axs[1].set_yscale("log")
    axs[1].set_ylim([1e-2, 2])
    axs[1].set_title(
        f"ref_aided_tomography dual-path DSP/Tomography | RMS Error: {rms_error:.2f} dB"
    )
    axs[1].legend(loc="lower left")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    fig0.savefig(OUT_DIR / "ref_aided_tomography_dual_path_optcommpy_style.png",
                 dpi=160, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ppe.zTomoBank, theory, "r-", linewidth=2.0,
            label=r"$\gamma(z)$ theory")
    ax.plot(ppe.zTomoBank, gamma_norm, "b-", linewidth=1.5,
            label=r"$\gamma(z)$ raw tomography")
    ax.plot(ppe.zTomoBank, gamma_safe, "k--", linewidth=1.5,
            label=r"$\gamma(z)$ optcommpy-style postprocess")
    ax.set_yscale("log")
    ax.set_xlabel("Distance (km)")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Normalized nonlinear profile")
    ax.set_title("Single-channel dual-pol Manakov RP1 dual path")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ref_aided_tomography_dsp_cdreload_gamma.png", dpi=160)

    interval = np.arange(320, min(900, sig_tx.shape[0]))
    t_ns = interval / FS_TOMO / 1e-9
    fig2, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for m, axm in enumerate(axes):
        label = "pol.X" if m == 0 else "pol.Y"
        axm.plot(t_ns, al[interval, m].real, "r-", linewidth=1.0,
                 label="AL.re")
        axm.plot(t_ns, al[interval, m].imag, "r--", linewidth=1.0,
                 label="AL.im")
        axm.plot(t_ns, recon[interval, m].real, "b-", linewidth=1.0,
                 label="fit.re")
        axm.plot(t_ns, recon[interval, m].imag, "b--", linewidth=1.0,
                 label="fit.im")
        axm.set_ylabel(label)
        axm.grid(True, alpha=0.25)
        axm.legend(loc="upper right", ncol=4)
    axes[-1].set_xlabel("Time (ns)")
    fig2.suptitle("DSP CD-reloaded waveform vs augmented RP1 fit")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "ref_aided_tomography_dsp_cdreload_fit.png", dpi=160)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


def main():
    t0 = time.time()
    print("[ref_aided_tomography] Single-channel dual-pol Manakov RP1 after pilotWDMRx DSP")
    print(f"[ref_aided_tomography] Tx optical SpS={SPS}, DSP/Tomo SpS={SPS_OUT}, "
          f"Fs_tomo={FS_TOMO:.6e} Hz")

    print("\n[ref_aided_tomography] Generating Tx")
    transmitter = pilotWDMTx(paramTx)
    sig_tx_wdm = np.asarray(transmitter.sigTxWDM)
    print(f"[ref_aided_tomography] sigTxWDM shape = {sig_tx_wdm.shape}")

    print("\n[ref_aided_tomography] Running manakovSSF")
    sig_wdm = manakovSSF(sig_tx_wdm, paramCh)
    print(f"[ref_aided_tomography] sigWDM after fiber shape = {sig_wdm.shape}")

    print("\n[ref_aided_tomography] Running pilotWDMRx DSP")
    receiver = pilotWDMRx(transmitter, sig_wdm, paramRx)
    ch_index = int(np.floor(paramTx.nChannels / 2))
    results = receiver.run(channels=[ch_index], plot=False, verbose=True)
    dsp = results[ch_index]

    ppe = RefAidedeRP1(make_tomo_params())

    print(f"[ref_aided_tomography] DSP metrics: BER={dsp['BER']}, SNR={dsp['SNR']}, "
          f"delay={dsp['delay']}, FO={dsp['fo_est'] / 1e6:.3f} MHz")

    print("\n[ref_aided_tomography] Running RefAidedeRP1 on DSP output")
    gamma_est, g_mat = ppe.run(dsp, param=make_run_params())
    sig_tx_run = ppe._lastRunSigTx
    sig_rx_run = ppe._lastRunSigRx
    p_tx = ppe._lastRunPowerTx
    print(f"\n[ref_aided_tomography] Tomography Tx shape = {sig_tx_run.shape}")
    print(f"[ref_aided_tomography] Tomography Rx shape = {sig_rx_run.shape}")
    fit_err = waveform_fit_error(ppe, sig_tx_run, sig_rx_run, gamma_est, g_mat)
    print(f"[ref_aided_tomography] Waveform fit error = {fit_err:.4e}")

    save_plots(ppe, gamma_est, g_mat, sig_tx_run, sig_rx_run, p_tx, dsp)

    print(f"\n[ref_aided_tomography] elapsed: {time.time() - t0:.2f} s")
    print(f"[ref_aided_tomography] results saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
