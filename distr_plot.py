import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

hmm_data_f = open("hmm_data.txt", "r")
l1_fine_f = open("level1_fine.txt", "r")
l1_coarse_f = open("level1_coarse.txt", "r")
l0_coarse_f = open("level0_coarse.txt", "r")
bpf_centiles_mse_f = open("bpf_centile_mse.txt", "r")
mlbpf_centiles_mse_f = open("mlbpf_centile_mse.txt", "r")
# bpf_distr_f = open("bpf_distr.txt", "r")

# bpf_centiles_mse = list(map(float, bpf_centiles_mse_f.readline().split()))
# mlbpf_centiles_mse = list(map(float, mlbpf_centiles_mse_f.readline().split()))


# Read in the HMM data #
# -------------------- #
length = int(hmm_data_f.readline())
sig_sd, obs_sd = list(map(float, hmm_data_f.readline().split()))
space_left, space_right = list(map(float, hmm_data_f.readline().split()))
nx = int(hmm_data_f.readline())
k_parameter = float(hmm_data_f.readline())
h0, q0 = list(map(float, hmm_data_f.readline().split()))
lb, up = list(map(float, hmm_data_f.readline().split()))
signal = np.empty(length); obs = np.empty(length)
xs, ws = [], []
N_bpf = 2000
# for n in range(length * N_bpf):
# 	x, w = list(map(float, bpf_distr_f.readline().split()))
# 	xs.append(x); ws.append(w)
# bpf_distr_x = np.array(xs).reshape((length, N_bpf))
# bpf_distr_w = np.array(ws).reshape((length, N_bpf))

length = 1
# Read in the likelihood data #
# --------------------------- #
l1_fine = []; l1_coarse = []; l0_coarse = []
k1 = 0
for line in l1_fine_f:
	l1_fine.extend(list(map(float, line.split())))
	k1 += 1
N1 = int(k1 / length)
l1_fine = np.reshape(np.array(l1_fine), (length * N1, 2))
for line in l1_coarse_f:
	l1_coarse.extend(list(map(float, line.split())))
l1_coarse = np.reshape(np.array(l1_coarse), (length * N1, 2))
k0 = 0
for line in l0_coarse_f:
	l0_coarse.extend(list(map(float, line.split())))
	k0 += 1
N0 = int(k0 / length)
l0_coarse = np.reshape(np.array(l0_coarse), (length * N0, 2))


# Reshape the data #
# ---------------- #
l1_h1 = np.empty((length, N1))
l1_h0 = np.empty((length, N1))
l0_h0 = np.empty((length, N0))
l1_g1 = np.empty((length, N1))
l1_g0 = np.empty((length, N1))
l0_g0 = np.empty((length, N0))
for n in range(length):
	l1_h1[n] = l1_fine[n * N1:(n + 1) * N1, 0]
	l1_g1[n] = l1_fine[n * N1:(n + 1) * N1, 1]
	l1_h0[n] = l1_coarse[n * N1:(n + 1) * N1, 0]
	l1_g0[n] = l1_coarse[n * N1:(n + 1) * N1, 1]
	l0_h0[n] = l0_coarse[n * N0:(n + 1) * N0, 0]
	l0_g0[n] = l0_coarse[n * N0:(n + 1) * N0, 1]


# Plot the data #
# ------------- #
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
for n in range(length):
	# axs.vlines(l0_h0, 0, l0_g0, lw=1, label=r"$g^0$", alpha=0.15, color="mediumseagreen")
	# axs.vlines(l1_h0, 0, l1_g0, lw=1, alpha=0.15, color="mediumseagreen")
	# axs.vlines(l1_h1, 0, l1_g1, lw=1, label=r"$g^1$", alpha=0.2, color="cornflowerblue")

	# These are the unnormalised measures
	# axs[n].vlines(l0_h0[n], 0, l0_g0[n], lw=1, label=r"$\tilde{\pi}^0$", alpha=0.15, color="mediumseagreen")
	# axs[n].vlines(l1_h1[n], 0, l1_g0[n], lw=1, alpha=0.75, color="mediumseagreen")
	# axs[n].vlines(l1_h1[n], 0, l1_g1[n], lw=1, label=r"$\tilde{\pi}^1$", alpha=0.75, color="cornflowerblue")
	# axs[n].vlines(l1_h1[n], 0, l1_g1[n] - l1_g0[n], lw=1, label=r"$\tilde{\pi}^1$", alpha=0.75, color="cornflowerblue")
	axs.vlines(l0_h0[n], 0, l0_g0[n], lw=1, label=r"$\tilde{\pi}^0$", alpha=0.15, color="mediumseagreen")
	axs.vlines(l1_h1[n], 0, l1_g0[n], lw=1, alpha=0.75, color="mediumseagreen")
	axs.vlines(l1_h1[n], 0, l1_g1[n], lw=1, label=r"$\tilde{\pi}^1$", alpha=0.75, color="cornflowerblue")
	axs.vlines(l1_h1[n], 0, l1_g1[n] - l1_g0[n], lw=1, label=r"$\tilde{\pi}^1$", alpha=0.75, color="cornflowerblue")

	# axs[n].vlines(bpf_distr_x[n], 0, bpf_distr_w[n], lw=1, alpha=0.75)
	# axs[n].vlines(bpf_centiles_mse[n], 0, 0.0006, lw=1.5, color="red")

	# axs.vlines(l0_h0, 0, l0_g0, lw=1, label=r"$\tilde{\pi}^0$", alpha=0.15, color="mediumseagreen")
	# axs.vlines(l1_h1, 0, l1_g0, lw=1, alpha=0.15, color="mediumseagreen")
	# axs.vlines(l1_h1, 0, l1_g1 - l1_g0, lw=1, label=r"$\tilde{\pi}^1$", alpha=0.75, color="cornflowerblue")

# axs.set_xlabel(r"$\varphi(\theta)$", fontsize=14)
# axs.set_xlabel(r"$\theta$", fontsize=14)
# plt.suptitle(r"Raw likelihood evaluations for $\varphi^1$ and $\varphi^0$", fontsize=18))
plt.suptitle("Unscaled and unnormalised level 0 and level 1 measures", fontsize=18)
# plt.suptitle("MLBPF posterior", fontsize=18)
# handles, labels = axs[1].get_legend_handles_labels()
handles, labels = axs.get_legend_handles_labels()
handles = []
a = 0; b = 1.0
line1 = np.zeros((1000, 2))
line2 = np.zeros((1000, 2))
line1[:, 0] = np.linspace(a, b, 1000); line1[:, 0] = np.linspace(a, b, 1000)
line2[:, 0] = np.linspace(a, b, 1000); line2[:, 0] = np.linspace(a, b, 1000)
handles.append(LineCollection(segments=[line1], linewidths=4, color="mediumseagreen"))
handles.append(LineCollection(segments=[line2], linewidths=4, color="cornflowerblue"))
# axs[1].legend(handles, labels, prop={'size': 12})
axs.legend(handles, labels, prop={'size': 12})
plt.tight_layout()
plt.show()



