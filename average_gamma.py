import numpy as np
from matplotlib import pyplot as plt
from tomo_fiber import tomo_cd, l_total, l_span, gamma, Nfft, alpha_tomo

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

seeds_count = 64
seeds = range(seeds_count)


_, ax = plt.subplots()

gamma_average = np.zeros_like(gamma_theory)
for seed_value in seeds:
    print('seed_value: ', seed_value)
    gamma = np.load('seed=' + str(seed_value) + 'gamma.npz')['gamma_total']
    ax.plot(z_tomo_bank, gamma, '--', alpha=0.2)
    gamma_average = gamma_average + gamma


ax.plot(z_tomo_bank, gamma_theory, 'r-', label=r'$\gamma$(z) theory')
ax.plot(z_tomo_bank, gamma_average/len(seeds), 'b-', label=r'$\gamma$(z) Average='+str(len(seeds)))

ax.legend(loc='upper right')
ax.set_xlabel('Distance(km)')
ax.xaxis.set_label_position('top')
ax.set_ylim([-1, 1.5])

plt.show()

