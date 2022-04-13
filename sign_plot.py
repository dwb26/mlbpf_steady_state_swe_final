import numpy as np
import matplotlib.pyplot as plt

signs_f = open("signs.txt", "r")
N0, N1, N_tot = list(map(int, signs_f.readline().split()))
particles = np.empty(N_tot)
signs = np.empty(N_tot)
abs_weights = np.empty(N_tot)
for i in range(N_tot):
	particles[i], signs[i], abs_weights[i] = list(map(float, signs_f.readline().split()))
ind_sort = np.argsort(particles)
particles = particles[ind_sort]
signs = signs[ind_sort]
abs_weights = abs_weights[ind_sort]
F_signs = np.empty(N_tot)
F_signs[0] = signs[0] * abs_weights[0]
for i in range(1, N_tot):
	F_signs[i] = signs[i] * abs_weights[i] + F_signs[i - 1]

fig, axs = plt.subplots(1, 1, figsize=(12, 9))
# axs[0].plot(particles, signs * abs_weights)
axs.plot(particles, F_signs)
# axs[0].set_xticks([])
# axs[0].set_title(r"$|w^i|\cdot$sgn$(\theta^i)$", fontsize=16)
axs.set_title(r"Cumulative sum of $\psi$", fontsize=16)
axs.set_xlabel(r"$\theta$", fontsize=14)
plt.subplots_adjust(hspace=0.8)
plt.tight_layout()
plt.show()