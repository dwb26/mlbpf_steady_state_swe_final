import numpy as np
import matplotlib.pyplot as plt

l1_fine_particles_f = open("level1_fine_particles.txt", "r")
l1_coarse_particles_f = open("level1_coarse_particles.txt", "r")
l0_particles_f = open("level0_particles.txt", "r")
uncorrected_particles_f = open("uncorrected_particles.txt", "r")
full_ref_data_f = open("full_ref_data.txt", "r")
normalisers_f = open("normalisers.txt", "r")

N0 = 2772; N1 = 50; N_tot = N0 + N1; length = 8
N_ref = 1000

l1_fine_particles_x = []
l1_coarse_particles_x = []
l0_particles_x = []
uncorrected_particles_x = []
l1_fine_particles_w = []
l1_coarse_particles_w = []
l0_particles_w = []
uncorrected_particles_w = []
ref_particles_x = []
ref_particles_w = []
normalisers = []

for line in l1_fine_particles_f:
	x, w = list(map(float, line.split()))
	l1_fine_particles_x.append(x)
	l1_fine_particles_w.append(w)
l1_fine_particles_x = np.array(l1_fine_particles_x).reshape((length, N1))
l1_fine_particles_w = np.array(l1_fine_particles_w).reshape((length, N1))

for line in l1_coarse_particles_f:
	x, w = list(map(float, line.split()))
	l1_coarse_particles_x.append(x)
	l1_coarse_particles_w.append(w)
l1_coarse_particles_x = np.array(l1_coarse_particles_x).reshape((length, N1))
l1_coarse_particles_w = np.array(l1_coarse_particles_w).reshape((length, N1))

for line in l0_particles_f:
	x, w = list(map(float, line.split()))
	l0_particles_x.append(x)
	l0_particles_w.append(w)
l0_particles_x = np.array(l0_particles_x).reshape((length, N0))
l0_particles_w = np.array(l0_particles_w).reshape((length, N0))

# for line in uncorrected_particles_f:
# 	x, w = list(map(float, line.split()))
# 	uncorrected_particles_x.append(x)
# 	uncorrected_particles_w.append(w)
# uncorrected_particles_x = np.array(uncorrected_particles_x).reshape((length, N0))
# uncorrected_particles_w = np.array(uncorrected_particles_w).reshape((length, N0))

for line in full_ref_data_f:
	x, w = list(map(float, line.split()))
	ref_particles_x.append(x)
	ref_particles_w.append(w)
normlisers = np.array(list(map(float, (normalisers_f.readline().split()))))
ref_particles_x = np.array(ref_particles_x).reshape((length, N_ref))
ref_particles_w = np.array(ref_particles_w).reshape((length, N_ref))

l1_w = l1_fine_particles_w - l1_coarse_particles_w
total_mlbpf_x = np.empty((length, N_tot))
total_mlbpf_w = np.empty((length, N_tot))
for n in range(length):
	total_mlbpf_x[n] = np.concatenate([l0_particles_x[n], l1_fine_particles_x[n]])
	total_mlbpf_w[n] = np.concatenate([l0_particles_w[n], l1_fine_particles_w[n] - l1_coarse_particles_w[n]])




# -------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# -------------------------------------------------------------------------------------------------------------------- #
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 7))
m = 0
for n in range(length):
	axs[m, n % 4].vlines(l0_particles_x[n], 0, l0_particles_w[n] * N0, lw=0.25, color="green", alpha=0.3, label="Level 0")
	# axs[m, n % 4].vlines(ref_particles_x[n], 0, ref_particles_w[n] * normlisers[n], lw=0.25, color="blue", alpha=0.1, label="Reference soln")
	if (n + 1) % 4 == 0:
		m += 1
	# if n == 0:
		# axs[m, n % 4].legend(markerscale=18)
plt.suptitle("nx0 = 25, N1 = 50, positive results")
plt.show()
































