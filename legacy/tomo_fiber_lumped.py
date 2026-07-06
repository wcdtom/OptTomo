from numpy.fft import fft, fftfreq, ifft
import numpy as np
from numpy.typing import NDArray
import scipy.constants as const
from signal_generator_coherent import paramTx, Fs, Ts, signal_length, SpS
from matplotlib import pyplot as plt
from optic.utils import parameters
from optic.models.channels import ssfm
from optic.models.tx import simpleWDMTx
from datetime import datetime
import argparse
from tqdm import tqdm
from collections import defaultdict

seed_num = 55
try:
    parser = argparse.ArgumentParser(description='Example script with random seed.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    args = parser.parse_args()
    seed_num = args.seed
except SystemExit as e:
    if e.code != 0:
        print(f"Argument parsing failed. Using default seed: {seed_num}")
    else:
        raise
np.random.seed(seed=seed_num)

D = 16          # ps/nm/km
alpha = 0.20    # dB/km
Fc = 193.1e12   # Hz
c_kms = const.c / 1e3
wavelength = c_kms / Fc
gamma = 1.3     # 1/W/km

alpha_np = alpha / (10 * np.log10(np.exp(1)))   # Np/km
beta_2 = -(D * wavelength ** 2) / (2 * np.pi * c_kms)

Nfft = int(signal_length * SpS)
omega = 2 * np.pi * Fs * fftfreq(Nfft)
omega = omega.reshape(omega.size, 1)

l_total = 100   # km
l_span  = 50    # km
delta_z = 0.5   # km

computing_ssfm  = True
if_save_ssfm    = False
if_save_g       = False
if_save_gamma   = True
computing_G     = True

if_normalize_power = True
if_plot = True

# =============================================================================
# LUMPED LOSSES  (connectors, couplers, splices ...)
#   Each entry: (position_km, loss_dB)
# =============================================================================
lumped_losses = [
    (25.0, 5.0),   # 5 dB at 25 km
    (75.0, 5.0),   # 5 dB at 75 km
]
# =============================================================================


def tomo_cd(length, signal_input):
    try:
        Nmodes = signal_input.shape[1]
    except IndexError:
        Nmodes = 1
    signal_input = signal_input.reshape(signal_input.size, Nmodes)
    omega_tomo = np.tile(omega, (1, Nmodes))
    signal_output = ifft(
        fft(signal_input, axis=0) * np.exp(1j * (beta_2 / 2) * (omega_tomo ** 2) * length),
        axis=0,
    )
    if Nmodes == 1:
        signal_output = signal_output.reshape(signal_output.size)
    return signal_output


# =============================================================================
# SSFM with lumped losses
#
# The link is split at every lumped-loss position and every span end.
# Each segment uses amp=None so ONLY distributed alpha is applied.
# At each span end: manual EDFA gain = exp(alpha_np/2 * l_span)
#   -> always compensates exactly ONE full span of distributed loss.
# At each lumped loss: field *= 10^(-loss_dB/20)
#   -> loss is NOT compensated by the EDFA.
#
# Result: P_ssfm = L_fwd_total^2 * P_tx
#   where L_fwd_total = prod(10^(-loss_dB_k/20)) over all lumped elements.
# =============================================================================
def nonlinear_fiber(signal_input):
    event_dict = defaultdict(list)
    N_spans = int(np.round(l_total / l_span))
    for i in range(1, N_spans + 1):
        event_dict[float(i * l_span)].append(('edfa',))
    for z_loss, loss_dB in lumped_losses:
        event_dict[float(z_loss)].append(('loss', loss_dB))

    positions = sorted(event_dict.keys())
    signal    = signal_input.copy()
    z_current = 0.0

    for z_event in positions:
        seg_length = z_event - z_current
        if seg_length > 1e-9:
            paramSeg            = parameters()
            paramSeg.Ltotal     = seg_length
            paramSeg.Lspan      = seg_length
            paramSeg.hz         = min(0.05, seg_length / 2.0)
            paramSeg.alpha      = alpha
            paramSeg.D          = D
            paramSeg.gamma      = gamma
            paramSeg.Fc         = Fc
            paramSeg.Fs         = Fs
            paramSeg.prgsBar    = False
            paramSeg.amp        = None   # distributed loss only, no automatic EDFA
            signal = ssfm(signal, paramSeg)

        # Apply events: lumped loss first, then EDFA
        events_here = sorted(event_dict[z_event],
                             key=lambda e: 0 if e[0] == 'loss' else 1)
        for event in events_here:
            if event[0] == 'loss':
                field_factor = 10.0 ** (-event[1] / 20.0)
                signal *= field_factor
                print(f"  [lumped loss] {event[1]:.2f} dB at z={z_event:.1f} km")
            elif event[0] == 'edfa':
                # Compensate distributed loss over this span
                gain_field = np.exp(alpha_np / 2.0 * l_span)
                # Also compensate any lumped losses that occurred within this span
                span_start = z_event - l_span
                for z_k, loss_dB_k in lumped_losses:
                    if span_start <= z_k < z_event:
                        gain_field *= 10.0 ** (loss_dB_k / 20.0)
                signal *= gain_field
                print(f"  [edfa] span {z_event/l_span:.0f}: total gain = {20*np.log10(gain_field):.2f} dB")

        z_current = z_event

    return signal


# =============================================================================
# G matrix  --  IDENTICAL TO THE ORIGINAL LOSSLESS VERSION
#
# The G matrix does NOT need to include any lumped-loss factor.
#
# Proof:
#   After normalising signal_ssfm by sqrt(P_ssfm) = L_fwd_total * sqrt(P_tx):
#
#     A1 = signal_ssfm_norm - A0
#        = sum_z  [gamma * exp(-alpha_np*zeta) * L_fwd(z)^2]  *  G_lossless[z]
#
#   The L_fwd(z)^2 arises from:
#     forward loss cubic:  L_fwd(z)^3
#     back-propagation:    L_bk(z)  = L_fwd_total / L_fwd(z)
#     normalisation denom: 1 / L_fwd_total
#     net:  L_fwd(z)^3 * L_bk(z) / L_fwd_total = L_fwd(z)^2
#
#   With the LOSSLESS G, solve_gamma recovers:
#     gamma_vec[z]  ~  gamma * P_tx * exp(-alpha_np*zeta) * L_fwd(z)^2
#
#   Display  gamma_g / gamma / P_average_tx  then gives:
#     exp(-alpha_np*zeta) * L_fwd(z)^2
#     = sawtooth with step-DOWN drops at each lumped-loss position  [correct]
#
#   If L_fwd(z)^2 were added INTO G, the solve would divide it back out
#   and the drops would disappear from the recovered profile.  WRONG.
# =============================================================================
def generate_matrix_g(save_g=False):
    G = np.zeros([len(z_tomo_bank), Nfft], dtype=complex)
    for z_index, z_tomo in enumerate(tqdm(z_tomo_bank, desc="Processing")):
        signal_before = tomo_cd(length=z_tomo, signal_input=sigTxo)
        P_average = 0
        nonlinear_operator = (signal_before * np.conj(signal_before) - 2 * P_average) * signal_before
        # Lossless kernel -- unchanged from original
        G[z_index] = 1j * delta_z * tomo_cd(length=l_total - z_tomo,
                                             signal_input=nonlinear_operator)
    if save_g:
        np.savez('./Result/seed=' + str(seed_num) + 'G.npz', G=G)
        print('Have saved _G.npz')
    return G


# =============================================================================
# solve_gamma  --  UNCHANGED from original
#
# A0 = tomo_cd(L, sigTxo_norm):
#   The lossless linear output. After normalisation, L_fwd_total cancels in
#   numerator and denominator, so this IS the correct linear reference.
# A1 = signal_ssfm_norm - A0:
#   Pure first-order nonlinear perturbation (with lumped losses encoded in it).
# =============================================================================
def solve_gamma(matrix_g, lambda_i=0, save_gamma=True):
    A_0 = tomo_cd(length=l_total, signal_input=sigTxo)
    A_1 = signal_ssfm - A_0

    G_dagger_G = np.dot(np.conjugate(matrix_g), matrix_g.T).real
    I = np.eye(G_dagger_G.shape[0])
    G_dagger_G = G_dagger_G + I * lambda_i
    inverse_G_dagger_G_plus_I = np.linalg.inv(G_dagger_G)
    G_dagger_A = np.dot(np.conjugate(matrix_g), A_1).real

    gamma_total = np.dot(inverse_G_dagger_G_plus_I, G_dagger_A)
    if save_gamma:
        np.savez('./Results/seed=' + str(seed_num) + 'gamma.npz', gamma_total=gamma_total)
        print('Have saved seed=' + str(seed_num) + 'gamma.npz')
    return gamma_total


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    if computing_ssfm:
        print('Generating signal...')
        sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)
        sigTxo = np.squeeze(sigWDM_Tx)

        print('Performing SSFM simulation with lumped losses...')
        print(datetime.now())
        signal_ssfm = nonlinear_fiber(signal_input=sigTxo)
        print(datetime.now())

        if if_save_ssfm:
            np.savez('./Results/seed=' + str(seed_num) + '.npz',
                     signal_ssfm=signal_ssfm, sigTxo=sigTxo)
    else:
        signal_ssfm = np.load('./Results/seed=' + str(seed_num) + '.npz')['signal_ssfm']
        sigTxo      = np.load('./Results/seed=' + str(seed_num) + '.npz')['sigTxo']

    signal_fiber = tomo_cd(length=l_total, signal_input=sigTxo)

    P_average_tx   = np.average(np.conjugate(sigTxo)       * sigTxo).real
    P_average_ssfm = np.average(np.conjugate(signal_ssfm)  * signal_ssfm).real
    P_average_cd   = np.average(np.conjugate(signal_fiber) * signal_fiber).real
    print('Average tx:', P_average_tx, 'Average ssfm:', P_average_ssfm)

    if if_normalize_power:
        signal_ssfm  = signal_ssfm  / np.sqrt(P_average_ssfm)
        sigTxo       = sigTxo       / np.sqrt(P_average_tx)
        signal_fiber = signal_fiber / np.sqrt(P_average_cd)
        np.savez('./Results/seed=' + str(seed_num) + 'P0.npz', P_aver=P_average_tx)

    z_tomo_bank = np.arange(0, l_total, delta_z)

    if computing_G:
        G = generate_matrix_g(save_g=if_save_g)
    else:
        G = np.load('./Results/seed=' + str(seed_num) + 'G.npz')['G']

    gamma_g = solve_gamma(matrix_g=G, save_gamma=if_save_gamma)

    if if_plot:
        fig, ax = plt.subplots(4, 1, figsize=(9, 11))

        # ------------------------------------------------------------------
        # gamma_theory(z) = exp(-alpha_np * local_z)   [distributed, resets at EDFA]
        #                 * prod(10^(-loss_dB/10) for z_k < z)  [lumped power drops]
        #
        # Scaled by gamma*delta_z to match the gamma_g / gamma / P_tx display.
        # This shows a SAWTOOTH with step-DOWN drops at each lumped-loss element.
        # ------------------------------------------------------------------
        gamma_theory = []
        for z_tomo in z_tomo_bank:
            span_num  = int(np.floor(z_tomo / l_span))
            local_z   = z_tomo - span_num * l_span
            span_start = span_num * l_span
            # Distributed loss resets to 1.0 at each span start (EDFA fully restores)
            g_z = np.exp(-alpha_np * local_z)
            # Only lumped losses within the CURRENT span affect gamma_theory
            # (losses in previous spans were compensated by the EDFA at each span end)
            for z_k, loss_dB_k in lumped_losses:
                if span_start <= z_k < z_tomo:
                    g_z *= 10.0 ** (-loss_dB_k / 10.0)  # power drop at each lumped loss
            gamma_theory.append(g_z)
        gamma_theory = np.array(gamma_theory)

        ax[0].plot(z_tomo_bank, gamma_g / gamma / P_average_tx, label=r'$\gamma$(z) tomo')
        ax[0].plot(z_tomo_bank, gamma_theory,                    label=r'$\gamma$(z) theory')
        ax[0].legend(loc='upper right')
        ax[0].set_xlabel('Distance (km)')
        ax[0].xaxis.set_label_position('top')
        ax[0].set_yscale('log')

        recover_RP1 = np.dot(G.T, gamma_theory)
        interval    = np.arange(16 * 20, 16 * 50)
        t           = interval * Ts / 1e-9

        ax[1].plot(t, sigTxo[interval].real, 'r-',  label='Tx.re')
        ax[1].plot(t, sigTxo[interval].imag, 'r--', label='Tx.im')
        ax[1].legend(loc='upper right')

        signal_fiber = tomo_cd(length=l_total, signal_input=sigTxo)
        ax[2].plot(t, signal_fiber[interval].real, 'b-',  label='A0.re')
        ax[2].plot(t, signal_fiber[interval].imag, 'b--', label='A0.im')
        ax[2].plot(t, signal_ssfm[interval].real,  'r-',  label='SSFM.re')
        ax[2].plot(t, signal_ssfm[interval].imag,  'r--', label='SSFM.im')
        ax[2].legend(loc='upper right')

        res         = signal_ssfm - signal_fiber
        solved_wave = np.dot(G.T, gamma_g)
        ax[3].plot(t, res[interval].real,         'r-',  label='A1.re')
        ax[3].plot(t, res[interval].imag,         'r--', label='A1.im')
        ax[3].plot(t, solved_wave[interval].real, 'b-',  label='solve.re')
        ax[3].plot(t, solved_wave[interval].imag, 'b--', label='solve.im')
        ax[3].legend(loc='upper right')
        ax[3].set_xlabel('Time (ns)')

        _, ax2 = plt.subplots(figsize=(10, 4))
        plot_line_period = 20
        for line_index, g_z in enumerate(G):
            if line_index % plot_line_period == 0:
                ax2.plot(t, g_z[interval].real, alpha=0.5)

        plt.show()
