import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

hmm_data_f = open("hmm_data.txt", "r")
mlbpf_prior_f = open("mlbpf_prior.txt", "r")
l1_fine_lhoods_f = open("level1_fine_lhoods.txt", "r")
l1_coarse_lhoods_f = open("level1_coarse_lhoods.txt", "r")
l0_coarse_lhoods_f = open("level0_coarse_lhoods.txt", "r")
l1_fine_f = open("level1_fine.txt", "r")
l1_coarse_f = open("level1_coarse.txt", "r")
l0_coarse_f = open("level0_coarse.txt", "r")
ml_distr_f = open("ml_distr.txt", "r")
bpf_lhoods_f = open("bpf_lhoods.txt", "r")
# bpf_posterior_f = open("bpf_posterior.txt", "r")
ref_posterior_f = open("ref_particles.txt", "r")
mlbpf_mutated_f = open("mlbpf_mutated.txt", "r")
N0, N1, N_tot = list(map(int, ml_distr_f.readline().split()))

# HMM data #
# -------- #
length = int(hmm_data_f.readline())
sig_sd, obs_sd = list(map(float, hmm_data_f.readline().split()))
space_left, space_right = list(map(float, hmm_data_f.readline().split()))
nx = int(hmm_data_f.readline())
k_parameter = float(hmm_data_f.readline())
h0, q0 = list(map(float, hmm_data_f.readline().split()))
lb, up = list(map(float, hmm_data_f.readline().split()))
signal = np.empty(length); obs = np.empty(length)
for n in range(length):
	signal[n], obs[n] = list(map(float, hmm_data_f.readline().split()))
length, N = list(map(int, bpf_lhoods_f.readline().split()))


# Posterior #
# --------- #
N_ref = 1000000
ref_posterior = np.empty((length, N_ref))
for n in range(length):
	ref_posterior[n] = list(map(float, ref_posterior_f.readline().split()))


# Prior #
# ----- #
mlbpf_prior = np.empty((length, N_tot))
for n in range(length):
	mlbpf_prior[n] = list(map(float, mlbpf_prior_f.readline().split()))


# Raw likelihoods #
# --------------- #
h0_coarse_solns = np.empty(length * N0)
h1_coarse_solns = np.empty(length * N1)
h1_fine_solns = np.empty(length * N1)
h0_coarse_lhoods = np.empty(length * N0)
h1_coarse_lhoods = np.empty(length * N1)
h1_fine_lhoods = np.empty(length * N1)
for n in range(length * N0):
	h0_coarse_solns[n], h0_coarse_lhoods[n] = list(map(float, l0_coarse_lhoods_f.readline().split()))
for n in range(length * N1):
	h1_coarse_solns[n], h1_coarse_lhoods[n] = list(map(float, l1_coarse_lhoods_f.readline().split()))
	h1_fine_solns[n], h1_fine_lhoods[n] = list(map(float, l1_fine_lhoods_f.readline().split()))
h0_coarse_solns = h0_coarse_solns.reshape((length, N0)); h0_coarse_lhoods = h0_coarse_lhoods.reshape((length, N0))
h1_coarse_solns = h1_coarse_solns.reshape((length, N1)); h1_coarse_lhoods = h1_coarse_lhoods.reshape((length, N1))
h1_fine_solns = h1_fine_solns.reshape((length, N1)); h1_fine_lhoods = h1_fine_lhoods.reshape((length, N1))


# signed measures #
# --------------- #
l0_coarse_thetas = np.empty(length * N0)
l1_coarse_thetas = np.empty(length * N1)
l1_fine_thetas = np.empty(length * N1)
l0_coarse_weights = np.empty(length * N0)
l1_coarse_weights = np.empty(length * N1)
l1_fine_weights = np.empty(length * N1)
for n in range(length * N0):
	l0_coarse_thetas[n], l0_coarse_weights[n] = list(map(float, l0_coarse_f.readline().split()))
for n in range(length * N1):
	l1_coarse_thetas[n], l1_coarse_weights[n] = list(map(float, l1_coarse_f.readline().split()))
	l1_fine_thetas[n], l1_fine_weights[n] = list(map(float, l1_fine_f.readline().split()))
l0_coarse_thetas = l0_coarse_thetas.reshape((length, N0)); l0_coarse_weights = l0_coarse_weights.reshape((length, N0))
l1_coarse_thetas = l1_coarse_thetas.reshape((length, N1)); l1_coarse_weights = l1_coarse_weights.reshape((length, N1))
l1_fine_thetas = l1_fine_thetas.reshape((length, N1)); l1_fine_weights = l1_fine_weights.reshape((length, N1))


# Posterior #
# --------- #
mlbpf_posterior_x = np.empty(length * N_tot)
mlbpf_posterior_w = np.empty(length * N_tot)
# mlbpf_posterior = np.empty((length, N_tot))
for n in range(length * N_tot):
	mlbpf_posterior_x[n], mlbpf_posterior_w[n] = list(map(float, ml_distr_f.readline().split()))
mlbpf_posterior_w = mlbpf_posterior_w.reshape((length, N_tot))
mlbpf_posterior_x = mlbpf_posterior_x.reshape((length, N_tot))
# mlbpf_posterior = np.empty((length, N_tot))
# for n in range(length):
# 	mlbpf_posterior[n] = list(map(float, ml_distr_f.readline().split()))


# Mutation #
# -------- #
mlbpf_mutated = np.empty((length, N_tot))
for n in range(length):
	mlbpf_mutated[n] = list(map(float, mlbpf_mutated_f.readline().split()))


# Plotting #
# -------- #
# print(raw_weights)
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))
fig.subplots_adjust(wspace=0.4)
axs[0].hist(x=mlbpf_prior[1], bins=50, density=True, color="crimson")
# axs[1].vlines(obs[1], 0, np.max(h0_coarse_lhoods[1]), lw=1, color="black", label=r"$y_n$")
# axs[1].vlines(h0_coarse_solns[1], 0, h0_coarse_lhoods[1], lw=0.1, color="seagreen")
# axs[1].vlines(h1_coarse_solns[1], 0, h1_coarse_lhoods[1], lw=0.1, color="dodgerblue")
# axs[1].vlines(h1_fine_solns[1], 0, h1_fine_lhoods[1], lw=0.1, color="dodgerblue")
axs[1].vlines(l0_coarse_thetas[1], 0, l0_coarse_weights[1], lw=0.1, color="seagreen", label=r"l0 $g^0$")
axs[1].vlines(l1_coarse_thetas[1], 0, -l1_coarse_weights[1], lw=0.1, color="dodgerblue", label=r"l1 $g^0$")
axs[1].vlines(l1_fine_thetas[1], 0, l1_fine_weights[1], lw=0.1, color="crimson", label=r"l1 $g^1$")
# axs[1].set(ylim=(0, 4.5))
# axs[2].hist(x=l0_coarse_thetas[1], bins=50, density=True, alpha=0.6, weights=l0_coarse_weights[1])
axs[2].hist(x=ref_posterior[1], bins=500, density=True, alpha=0.5)
axs[3].hist(x=mlbpf_mutated[1], bins=50, density=True, color="crimson")
# axs[2].hist(x=mlbpf_posterior_x[1], bins=250, density=True, weights=mlbpf_posterior_w[1], alpha=0.5)
# axs[2].hist(x=mlbpf_posterior[1], bins=100, density=True, alpha=0.5)
# axs[3].hist(x=bpf_mutated[1], bins=50, density=True, color="crimson")
axs[0].set_xlabel(r"$\theta_n$", fontsize=18)
axs[1].set_xlabel(r"$\theta_n$", fontsize=18)
axs[2].set_xlabel(r"$\theta_n$", fontsize=18)
axs[3].set_xlabel(r"$\theta_{n + 1}$", fontsize=18)
axs[0].set_title(r"$\pi(\theta_n|y_{1:n - 1})$", fontsize=20)
axs[1].set_title(r"$g_n(\theta)$", fontsize=20)
axs[2].set_title(r"$\widehat{\pi}(\theta_n|y_{1:n})$", fontsize=20)
axs[3].set_title(r"$\pi(\theta_{n + 1}|y_{1:n})$", fontsize=20)
handles, labels = axs[1].get_legend_handles_labels()
handles = []
a = 0; b = 1.0
line1 = np.zeros((1000, 2))
line2 = np.zeros((1000, 2))
line3 = np.zeros((1000, 2))
line1[:, 0] = np.linspace(a, b, 1000); line1[:, 0] = np.linspace(a, b, 1000)
line2[:, 0] = np.linspace(a, b, 1000); line2[:, 0] = np.linspace(a, b, 1000)
line3[:, 0] = np.linspace(a, b, 1000); line3[:, 0] = np.linspace(a, b, 1000)
handles.append(LineCollection(segments=[line1], linewidths=4, color="seagreen"))
handles.append(LineCollection(segments=[line2], linewidths=4, color="dodgerblue"))
handles.append(LineCollection(segments=[line3], linewidths=4, color="crimson"))
axs[1].legend(handles, labels, prop={'size': 12})
# axs[1].legend(prop={'size': 16})
plt.tight_layout()
plt.show()






























