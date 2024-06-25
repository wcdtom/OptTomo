import numpy as np
from matplotlib import pyplot as plt
from tomo_fiber import tomo_cd, l_total, l_span, Nfft, alpha_tomo, delta_z, gamma

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

seeds_count = 16
seeds = range(seeds_count)


_, ax = plt.subplots()

gamma_average = np.zeros_like(gamma_theory)
for seed_value in seeds:
    print('seed_value: ', seed_value)
    gamma_t = np.load('./Results/seed=' + str(seed_value) + 'gamma.npz')['gamma_total']
    ax.plot(z_tomo_bank, gamma_t/gamma, '--', alpha=0.2)
    gamma_average = gamma_average + gamma_t


ax.plot(z_tomo_bank, gamma_theory ** 2, 'r-', label=r'$\gamma$(z) theory')  # in Power <-- ** 2
ax.plot(z_tomo_bank, gamma_average/len(seeds)/gamma, 'b-', label=r'$\gamma$(z) Average='+str(len(seeds)))

ax.legend(loc='upper right')
ax.set_xlabel('Distance(km)')
ax.xaxis.set_label_position('top')
ax.set_yscale('log')
ax.set_ylabel('Loss (dB)')
ax.set_xlabel('Distance(km)')
# ax.set_ylim([-1, 1.5])

plt.show()

