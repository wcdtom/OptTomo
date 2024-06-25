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

seed_num = 55  # necessary condition
try:
    parser = argparse.ArgumentParser(description='Example script with random seed.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    args = parser.parse_args()
    seed_num = args.seed
except SystemExit as e:
    if e.code != 0:  # 非正常退出
        print(f"Argument parsing failed. Using default seed: {seed_num}")
    else:
        raise
np.random.seed(seed=seed_num)  # fixing the seed to get reproducible results

D = 16  # ps/nm/km
alpha = 0.20  # dB/km
Fc = 193.1e12  # Hz
c_kms = const.c / 1e3  # km/s <-- speed of light (vacuum)
wavelength = c_kms / Fc  # km
gamma = 1.3  # 1/W/km

alpha_tomo = alpha / (10 * np.log10(np.exp(1)))
beta_2 = -(D * wavelength ** 2) / (2 * np.pi * c_kms)  # ps*s/nm --> *1e24 --> ps**2/km (-2.047e-23)
# beta_2 = -21.6  # ps**2/km

Nfft = int(signal_length * SpS)
omega = 2 * np.pi * Fs * fftfreq(Nfft)
omega = omega.reshape(omega.size, 1)

l_total = 150
l_span = 50
delta_z = 1

computing_ssfm = False
if_save_ssfm = False
if_save_g = False
if_save_gamma = True
computing_G = True

if_normalize_power = False

if_plot = False


def tomo_cd(length: float, signal_input: NDArray) -> NDArray:
    try:
        Nmodes = signal_input.shape[1]
    except IndexError:
        Nmodes = 1
        signal_input = signal_input.reshape(signal_input.size, Nmodes)

    omega_tomo = np.tile(omega, (1, Nmodes))
    signal_output = ifft(
        fft(signal_input, axis=0) * np.exp(1j * (beta_2 / 2) * (omega_tomo ** 2) * length), axis=0
    )

    if Nmodes == 1:
        signal_output = signal_output.reshape(
            signal_output.size,
        )

    return signal_output


def nonlinear_fiber(signal_input: NDArray) -> NDArray:
    # optical channel parameters
    paramCh = parameters()
    paramCh.Ltotal = l_total  # total link distance [km]
    paramCh.Lspan = l_span  # span length [km]
    paramCh.hz = 0.05  # step-size of the split-step Fourier method [km]
    paramCh.alpha = alpha  # fiber loss parameter [dB/km]
    paramCh.D = D  # fiber dispersion parameter [ps/nm/km]
    paramCh.gamma = gamma  # fiber nonlinear parameter [1/(W.km)]
    paramCh.Fc = Fc  # central optical frequency of the WDM spectrum
    paramCh.Fs = Fs  # sampling rate
    paramCh.prgsBar = True  # show progress bar
    paramCh.amp = 'ideal'
    # paramCh.amp = None

    # nonlinear signal propagation
    signal_output = ssfm(signal_input, paramCh)
    return signal_output


def generate_matrix_g(save_g=False):
    G = np.zeros([len(z_tomo_bank), Nfft], dtype=complex)
    # for z_tomo in z_tomo_bank:
    for z_index, z_tomo in enumerate(tqdm(z_tomo_bank, desc="Processing")):
        # A(z)
        signal_before = tomo_cd(length=z_tomo, signal_input=sigTxo)

        # N(z) = |A(z)|^2 * A(z)
        nonlinear_operator = (signal_before * np.conj(signal_before)) * signal_before  # - 2 * P_average

        # g(z) = j * D{zl}{N(z)}
        g_bias = 1j * delta_z * tomo_cd(length=l_total - z_tomo, signal_input=nonlinear_operator)

        G[z_index] = g_bias
    if save_g:
        np.savez('./Result/seed='+str(seed_num)+'G.npz', G=G)
        print('Have saved _G.npz')
    return G


def solve_gamma(matrix_g: np.ndarray, lambda_i=0, save_gamma=True):
    # A1 = A - A0
    A_1 = signal_ssfm - tomo_cd(length=l_total, signal_input=sigTxo)

    # gamma = Re[G*G + lambda*I]^-1 * Re[G*A_1]
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

if __name__ == '__main__':
    if computing_ssfm:
        print('Generating signal...')
        sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)
        sigTxo = np.squeeze(sigWDM_Tx)

        print('Generating chromatic dispersion signal...')
        signal_fiber = tomo_cd(length=l_total, signal_input=sigTxo)

        print('Performing SSFM simulation...')
        print(datetime.now())
        signal_ssfm = nonlinear_fiber(signal_input=sigTxo)
        print(datetime.now())

        if if_save_ssfm:
            np.savez('./Results/seed=' + str(seed_num) + '.npz', signal_ssfm=signal_ssfm, sigTxo=sigTxo)
            print('Have saved the seed=' + str(seed_num) + ' ssfm waveform .npz')
    else:
        signal_ssfm = np.load('./Results/seed=' + str(seed_num) + '.npz')['signal_ssfm']
        sigTxo = np.load('./Results/seed=' + str(seed_num) + '.npz')['sigTxo']

    # # average_power
    # P_average_tx = np.average(np.conjugate(sigTxo) * sigTxo)
    # P_average_ssfm = np.average(np.conjugate(signal_ssfm) * signal_ssfm)
    # P_average_cd = np.average(np.conjugate(signal_fiber) * signal_fiber)
    # print('Average tx:', P_average_tx, 'Average ssfm:', P_average_ssfm, 'Average cd:', P_average_cd)

    # if if_normalize_power:
    #     # signal_ssfm = signal_ssfm * np.sqrt(P_average_tx/P_average_ssfm)
    #     signal_ssfm = signal_ssfm * np.sqrt(1 / P_average_ssfm)
    #     sigTxo = sigTxo * np.sqrt(1 / P_average_tx)
    #     signal_fiber = signal_fiber * np.sqrt(1 / P_average_cd)
    #
    #     P_average_tx = np.average(np.conjugate(sigTxo) * sigTxo)
    #     P_average_ssfm = np.average(np.conjugate(signal_ssfm) * signal_ssfm)
    #     P_average_cd = np.average(np.conjugate(signal_fiber) * signal_fiber)
    #     print('After Power Normalization:')
    #     print('Average tx:', P_average_tx, 'Average ssfm:', P_average_ssfm, 'Average cd:', P_average_cd)

    z_tomo_bank = np.arange(0, l_total, delta_z)

    if computing_G:
        G = generate_matrix_g(save_g=if_save_g)
    else:
        G = np.load('./Results/seed=' + str(seed_num) + 'G.npz')['G']

    gamma_g = solve_gamma(matrix_g=G, save_gamma=if_save_gamma)

    if if_plot:

        fig, ax = plt.subplots(4, 1, figsize=(9, 11))

        N_spans = int(np.floor(l_total / l_span))
        gamma_theory = []
        for z_index in range(z_tomo_bank.shape[0]):
            z_tomo = z_tomo_bank[z_index]
            span_num = np.floor(z_tomo / l_span)
            gamma_theory_z = np.exp(-alpha_tomo / 2 * z_tomo + span_num * l_span * alpha_tomo / 2)
            gamma_theory.append(gamma_theory_z)
        gamma_theory = np.array(gamma_theory)

        ax[0].plot(z_tomo_bank, gamma_g, label=r'$\gamma$(z) tomo')
        ax[0].plot(z_tomo_bank, gamma_theory, label=r'$\gamma$(z) theory')
        ax[0].legend(loc='upper right')
        ax[0].set_xlabel('Distance(km)')
        ax[0].xaxis.set_label_position('top')

        recover_RP1 = np.dot(G.T, gamma_theory)
        interval = np.arange(16 * 20, 16 * 50)
        t = interval * Ts / 1e-9

        ax[1].plot(t, sigTxo[interval].real, 'r-', label='Tx.re')
        ax[1].plot(t, sigTxo[interval].imag, 'r--', label='Tx.im')
        ax[1].legend(loc='upper right')

        signal_fiber = tomo_cd(length=l_total, signal_input=sigTxo)
        ax[2].plot(t, signal_fiber[interval].real, 'b-', label='A0.re')
        ax[2].plot(t, signal_fiber[interval].imag, 'b--', label='A0.im')
        ax[2].plot(t, signal_ssfm[interval].real, 'r-', label='SSFM.re')
        ax[2].plot(t, signal_ssfm[interval].imag, 'r--', label='SSFM.im')
        ax[2].legend(loc='upper right')

        res = signal_ssfm - signal_fiber
        solved_wave = np.dot(G.T, gamma_g)
        ax[3].plot(t, res[interval].real, 'r-', label='A1.re')
        ax[3].plot(t, res[interval].imag, 'r--', label='A1.im')
        ax[3].plot(t, solved_wave[interval].real, 'b-', label='solve.re')
        ax[3].plot(t, solved_wave[interval].imag, 'b--', label='solve.im')
        ax[3].legend(loc='upper right')
        ax[3].set_xlabel('Time(ns)')

        _, ax = plt.subplots(figsize=(10, 4))
        plot_line_period = 20
        line_index = 0
        for g_z in G:
            if line_index % plot_line_period == 0:
                ax.plot(t, g_z[interval].real, alpha=0.5)
            line_index += 1

        plt.show()
