import numpy as np
import matplotlib.pyplot as plt

hmm_data_f = open("hmm_data.txt", "r")
bpf_prior_f = open("bpf_prior.txt", "r")
bpf_lhoods_f = open("bpf_lhoods.txt", "r")
bpf_posterior_f = open("bpf_posterior.txt", "r")
ref_posterior_f = open("ref_particles.txt", "r")
bpf_mutated_f = open("bpf_mutated.txt", "r")

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
bpf_prior = np.empty((length, N))
for n in range(length):
	bpf_prior[n] = list(map(float, bpf_prior_f.readline().split()))


# Raw likelihoods #
# --------------- #
h_solns = np.empty(length * N)
raw_weights = np.empty(length * N)
for n in range(length * N):
	h_solns[n], raw_weights[n] = list(map(float, bpf_lhoods_f.readline().split()))
	# print(len(h_solns[n]))
	# print(len(raw_weights[n]))
h_solns = h_solns.reshape((length, N))
raw_weights = raw_weights.reshape((length, N))
# print(h_solns.shape)


# Posterior #
# --------- #
bpf_posterior = np.empty((length, N))
for n in range(length):
	bpf_posterior[n] = list(map(float, bpf_posterior_f.readline().split()))


# Mutation #
# -------- #
bpf_mutated = np.empty((length, N))
for n in range(length):
	bpf_mutated[n] = list(map(float, bpf_mutated_f.readline().split()))


# Plotting #
# -------- #
# print(raw_weights)
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))
fig.subplots_adjust(wspace=0.4)
axs[0].hist(x=bpf_prior[1], bins=50, density=True, color="crimson")
axs[1].vlines(h_solns[1], 0, raw_weights[1], lw=0.1, color="seagreen")
axs[1].vlines(obs[1], 0, np.max(raw_weights[1]), lw=1, color="black", label=r"$y_n$")
axs[1].set(ylim=(0, 4.5))
axs[2].hist(x=ref_posterior[1], bins=500, density=True, alpha=0.6)
axs[2].hist(x=bpf_posterior[1], bins=50, density=True, alpha=0.5)
axs[3].hist(x=bpf_mutated[1], bins=50, density=True, color="crimson")
axs[0].set_xlabel(r"$\theta_n$", fontsize=18)
axs[1].set_xlabel(r"$\varphi(\theta_n)$", fontsize=18)
axs[2].set_xlabel(r"$\theta_n$", fontsize=18)
axs[3].set_xlabel(r"$\theta_{n + 1}$", fontsize=18)
axs[0].set_title(r"$\pi(\theta_n|y_{1:n - 1})$", fontsize=20)
axs[1].set_title(r"$g_n(\theta_n)$", fontsize=20)
axs[2].set_title(r"$\widehat{\pi}(\theta_n|y_{1:n})$", fontsize=20)
axs[3].set_title(r"$\pi(\theta_{n + 1}|y_{1:n})$", fontsize=20)
axs[1].legend(prop={'size': 16})
plt.tight_layout()
plt.show()






























