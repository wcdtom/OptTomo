import numpy as np
from signal_generator_coherent import Fs, Ts
from matplotlib import pyplot as plt
from tomo_fiber import tomo_cd, l_total, l_span, gamma, Nfft, alpha_tomo, seed_num

delta_z = 1
# z_tomo_bank = delta_z + np.arange(0, l_total, delta_z)
z_tomo_bank = np.arange(0, l_total, delta_z)

N_spans = int(np.floor(l_total / l_span))
gamma_theory = []
for z_index in range(z_tomo_bank.shape[0]):
    z_tomo = z_tomo_bank[z_index]
    span_num = np.floor(z_tomo/l_span)
    gamma_theory_z = np.exp(-alpha_tomo/2 * z_tomo + span_num * l_span * alpha_tomo/2)
    gamma_theory.append(gamma_theory_z)
gamma_theory = np.array(gamma_theory)

signal_ssfm = np.load('seed='+str(seed_num)+'.npz')['signal_ssfm']
sigTxo = np.load('seed='+str(seed_num)+'.npz')['sigTxo']

# average_power
P_average = np.average(np.conjugate(sigTxo) * sigTxo)

# Generating matrx G
computing_G = True
if_save_G = False  # G is the largest Object in this simulation program (~1GB)
if computing_G:
    G = np.zeros([len(z_tomo_bank), Nfft], dtype=complex)
    z_index = 0
    for z_tomo in z_tomo_bank:
        # A(z)
        signal_before = tomo_cd(length=z_tomo, signal_input=sigTxo)

        # N(z) = |A(z)|^2 * A(z)
        nonlinear_operator = (signal_before * np.conj(signal_before)) * signal_before  # - 2 * P_average

        # g(z) = j * D{zl}{N(z)}
        g_bias = 1j * delta_z * tomo_cd(length=l_total - z_tomo, signal_input=nonlinear_operator)

        G[z_index] = g_bias
        z_index += 1
        print(z_tomo)
    if if_save_G:
        np.savez('seed='+str(seed_num)+'G.npz', G=G)
        print('Have saved _G.npz')
else:
    G = np.load('seed='+str(seed_num)+'G.npz')['G']

computing_gamma = True
if computing_gamma:
    A_1 = signal_ssfm - tomo_cd(length=l_total, signal_input=sigTxo)
    G_dagger_G = np.dot(np.conjugate(G), G.T).real

    I = np.eye(G_dagger_G.shape[0])
    lambda_I = 0
    G_dagger_G = G_dagger_G + I * lambda_I

    inverse_G_dagger_G_plus_I = np.linalg.inv(G_dagger_G)

    G_dagger_A = np.dot(np.conjugate(G), A_1).real

    gamma_total = np.dot(inverse_G_dagger_G_plus_I, G_dagger_A)

    np.savez('seed='+str(seed_num)+'gamma.npz', gamma_total=gamma_total)
    print('Have saved seed=' + str(seed_num) + 'gamma.npz')
else:
    gamma_total = np.load('seed='+str(seed_num)+'gamma.npz')['gamma_total']

# gamma_total, residuals, rank, s = np.linalg.lstsq(a=G.T, b=A_1, rcond=0.1)

fig, ax = plt.subplots(4, 1, figsize=(9, 11))

ax[0].plot(z_tomo_bank, gamma_total, label=r'$\gamma$(z) tomo')
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
solved_wave = np.dot(G.T, gamma_total)
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
