import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
plt.style.use("ggplot")

hmm_data = open("hmm_data.txt", "r")
raw_bpf_times_f = open("raw_bpf_times.txt", "r")
raw_bpf_mse_f = open("raw_bpf_mse.txt", "r")
raw_bpf_ks_f = open("raw_bpf_ks.txt", "r")


# --------------------------------------------------- HMM data ------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
length = int(hmm_data.readline())
sig_sd, obs_sd = list(map(float, hmm_data.readline().split()))
hmm_data.readline()
nx = int(hmm_data.readline())
for n in range(2):
    hmm_data.readline()


# -------------------------------------------------- BPF results ---------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
N_nxs = 60; nx = 500; nx_incr = 5
nx_bpfs = np.empty(N_nxs + 1)
nx_bpfs[0] = nx
for n in range(1, N_nxs + 1):
	nx_bpfs[n] = nx - n * nx_incr
res = nx_bpfs
bpf_rmse_means = np.empty(N_nxs + 1)
bpf_rmse_medians = np.empty(N_nxs + 1)
bpf_ks_means = np.empty(N_nxs + 1)
bpf_ks_medians = np.empty(N_nxs + 1)
k = 0
for line in raw_bpf_mse_f:
	data = np.array(list(map(float, line.split())))
	bpf_rmse_means[k] = np.mean(np.log10(np.sqrt(data)))
	bpf_rmse_medians[k] = np.median(np.log10(np.sqrt(data)))
	k += 1
k = 0
for line in raw_bpf_ks_f:
	data = np.array(list(map(float, line.split())))
	bpf_ks_means[k] = np.mean(np.log10(data))
	bpf_ks_medians[k] = np.median(np.log10(data))
	k += 1
times = []
target_time = np.mean(np.array(list(map(float, raw_bpf_times_f.readline().split()))))
for line in raw_bpf_times_f:
	times.extend(list(map(float, line.split())))



# ------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
fig_width = 8; fig_height = 7
hspace = 0.9
fig1, axs1 = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))
fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))
fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height))
axs1[0].plot(res, bpf_rmse_means)
axs1[0].plot(res, bpf_rmse_means[0] * np.ones(N_nxs + 1), color="black", label="True nx")
axs1[0].invert_xaxis()
axs1[0].set_xticks([])
axs1[1].plot(res, bpf_ks_means)
axs1[1].plot(res, bpf_ks_means[0] * np.ones(N_nxs + 1), color="black", label="True nx")
axs1[1].invert_xaxis()
fig1.suptitle("Mean data")
axs1[0].set_title("Mean(log10(RMSE))")
axs1[1].set_title("Mean(log10(KS))")
axs1[1].set_xlabel("nx")
axs1[0].legend()

axs2[0].plot(res, bpf_rmse_medians)
axs2[0].plot(res, bpf_rmse_medians[0] * np.ones(N_nxs + 1), color="black", label="True nx")
axs2[0].invert_xaxis()
axs2[0].set_xticks([])
axs2[1].plot(res, bpf_ks_medians)
axs2[1].plot(res, bpf_ks_medians[0] * np.ones(N_nxs + 1), color="black", label="True nx")
axs2[1].invert_xaxis()
fig2.suptitle("Median data")
axs2[0].set_title("Median(log10(RMSE))")
axs2[1].set_title("Median(log10(KS))")
axs2[1].set_xlabel("nx")
axs2[0].legend()

axs3.plot(range(len(times)), times)
axs3.plot(range(len(times)), target_time * np.ones(len(times)), color="black", label="Target time")
axs3.set_title("BPF experiment times")
axs3.set_xlabel("Iterate")
axs3.legend()

plt.tight_layout()
plt.show()






























