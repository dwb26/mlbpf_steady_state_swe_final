import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns

hmm_data = open("hmm_data.txt", "r")
ml_parameters = open("ml_parameters.txt", "r")
raw_bpf_times = open("raw_bpf_times.txt", "r")
raw_bpf_mse = open("raw_bpf_mse.txt", "r")
raw_bpf_ks = open("raw_bpf_ks.txt", "r")
raw_mse = open("raw_mse.txt", "r")
raw_ks = open("raw_ks.txt", "r")
raw_srs = open("raw_srs.txt", "r")
raw_times = open("raw_times.txt", "r")
N1s_data = open("N1s_data.txt", "r")
alloc_counters_f = open("alloc_counters.txt", "r")
ref_stds_f = open("ref_stds.txt", "r")
bpf_centile_mse_f = open("bpf_centile_mse.txt", "r")
mlbpf_centile_mse_f = open("mlbpf_centile_mse.txt", "r")



# --------------------------------------------------- HMM data ------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf = list(map(int, ml_parameters.readline().split()))
level0s = list(map(int, ml_parameters.readline().split()))
length = int(hmm_data.readline())
sig_sd, obs_sd = list(map(float, hmm_data.readline().split()))
hmm_data.readline()
nx = int(hmm_data.readline())
for n in range(2):
    hmm_data.readline()



# -------------------------------------------------- BPF results ---------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
bpf_times = []; bpf_ks = []; bpf_mse = []; bpf_ks = []
for n_data in range(N_data):
	bpf_times.extend(list(map(float, raw_bpf_times.readline().split())))
	bpf_mse.extend(list(map(float, raw_bpf_mse.readline().split())))
	bpf_ks.extend(list(map(float, raw_bpf_ks.readline().split())))

# bpf_mse_new = []; bpf_ks_new = []
# for x in bpf_mse:
# 	if not pd.isna(x):
# 		bpf_mse_new.append(x)
# for x in bpf_ks:
# 	if not pd.isna(x):
# 		bpf_ks_new.append(x)
# bpf_mse = bpf_mse_new
# bpf_ks = bpf_ks_new
bpf_rmse = np.sqrt(bpf_mse)
bpf_mean_mse_log10 = np.mean(np.log10(bpf_mse))
bpf_median_mse_log10 = np.median(np.log10(bpf_mse))
bpf_mean_rmse_log10 = np.mean(np.log10(bpf_rmse))
bpf_median_rmse_log10 = np.median(np.log10(bpf_rmse))
bpf_mean_ks_log10 = np.mean(np.log10(bpf_ks))
bpf_median_ks_log10 = np.median(np.log10(bpf_ks))
ref_stds = np.array(list(map(float, ref_stds_f.readline().split())))
bpf_centile_mse = np.array(list(map(float, bpf_centile_mse_f.readline().split())))



# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)
mse_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
rmse_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
ks_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
srs_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
mlbpf_centile_mse_arr = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
N_NaNs = np.zeros((N_MESHES, N_ALLOCS))
ks_glob_min = 100000
ks_glob_max = -100000
eps = 1e-06

# Data read #
# --------- #
for i_mesh in range(N_MESHES):
	for n_alloc in range(N_ALLOCS):
		mse_arr[i_mesh, n_alloc, :] = list(map(float, raw_mse.readline().split()))
		ks_arr[i_mesh, n_alloc, :] = list(map(float, raw_ks.readline().split()))
		srs_arr[i_mesh, n_alloc, :] = list(map(float, raw_srs.readline().split()))
		mlbpf_centile_mse_arr[i_mesh, n_alloc, :] = list(map(float, mlbpf_centile_mse_f.readline().split()))

		# KS data read
		for x in ks_arr[i_mesh, n_alloc, :]:
			if not np.abs(x - -1) < eps and not np.abs(x - -2) < eps:
				if x < ks_glob_min:
					ks_glob_min = x
				if x > ks_glob_max:
					ks_glob_max = x

# NaN handling #
# ------------ #
ks_glob_min = np.log10(ks_glob_min)
ks_glob_max = np.log10(ks_glob_max)
for i_mesh in range(N_MESHES):
	for n_alloc in range(N_ALLOCS):
		for n_trial in range(N_trials * N_data):

			# KS NaN handling
			# Set these to NaNs because of the N1 allocation expiration
			if np.abs(ks_arr[i_mesh, n_alloc, n_trial] - -1) < eps:
				ks_arr[i_mesh, n_alloc, n_trial] = float("NaN")

			# Set these to NaNs because of the results
			if np.abs(ks_arr[i_mesh, n_alloc, n_trial] - -2) < eps:
				ks_arr[i_mesh, n_alloc, n_trial] = float("NaN")
				N_NaNs[i_mesh, n_alloc] += 1

			# MSE and RMSE NaN handling
			# Set these to NaNs because of the N1 allocation expiration
			if np.abs(mse_arr[i_mesh, n_alloc, n_trial] - -1) < eps:
				mse_arr[i_mesh, n_alloc, n_trial] = float("NaN")
				rmse_arr[i_mesh, n_alloc, n_trial] = float("NaN")

			# Set these to NaNs because of the results
			if np.abs(mse_arr[i_mesh, n_alloc, n_trial] - -2) < eps:
				mse_arr[i_mesh, n_alloc, n_trial] = float("NaN")
				rmse_arr[i_mesh, n_alloc, n_trial] = float("NaN")

			if np.abs(mlbpf_centile_mse_arr[i_mesh, n_alloc, n_trial] - -1) < eps:
				mlbpf_centile_mse_arr[i_mesh, n_alloc, n_trial] = float("NaN")

for i_mesh in range(N_MESHES):
	for n_alloc in range(N_ALLOCS):
		c = 0
		for x in mse_arr[i_mesh, n_alloc, :]:
			rmse_arr[i_mesh, n_alloc, c] = np.sqrt(x)
			c += 1


# ------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
colors = ["orchid", "mediumpurple", "royalblue", "powderblue", "mediumseagreen", "greenyellow", "orange", "tomato", "firebrick"]
fig_width = 8; fig_height = 7
hspace = 0.9
fig1, axs = plt.subplots(nrows=N_MESHES, ncols=1, figsize=(fig_width, fig_height))
fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))
fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height))
fig1.subplots_adjust(hspace=hspace)
fig1.suptitle(r"N_data = {}, N_trials = {}, nx = {}, N_bpf = {}, $\sigma_{} = {}$, $\sigma_{} = {}$, len = {}".format(N_data, N_trials, nx, N_bpf, "s", sig_sd, "o", obs_sd, length))
fig2.subplots_adjust(hspace=0.4)
fig2.suptitle(r"N_data = {}, N_trials = {}, nx = {}, N_bpf = {}, $\sigma_{} = {}$, $\sigma_{} = {}$, len = {}".format(N_data, N_trials, nx, N_bpf, "s", sig_sd, "o", obs_sd, length))

ks_boxplot = True
# ks_boxplot = False
# mse_boxplot = True
mse_boxplot = False
mean_rmse = True
# mean_rmse = False



# ------------------------------------------------------------------------------------------------------------------- #
#
# Boxplots
#
# ------------------------------------------------------------------------------------------------------------------- #
if ks_boxplot:
	if N_MESHES > 1:
		for i_mesh in range(N_MESHES):
			ax = sns.boxplot(data=pd.DataFrame(np.log10(ks_arr[i_mesh].T), columns=N1s), ax=axs[i_mesh], color=colors[i_mesh], whis=1000)
			ax.plot(range(max_allocs), bpf_median_ks_log10 * np.ones(max_allocs), color="limegreen", label="BPF KS")
			ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)
			ax.set(ylim=(ks_glob_min, ks_glob_max))
			if i_mesh == N_MESHES - 1:
				ax.set_xlabel(r"$N_1$")
			if i_mesh < N_MESHES - 1:
				ax.set_xticks([])
	else:
		for i_mesh in range(N_MESHES):
			ax = sns.boxplot(data=pd.DataFrame(np.log10(ks_arr[i_mesh].T), columns=N1s), ax=axs, color=colors[i_mesh], whis=1000)
			ax.plot(range(max_allocs), bpf_median_ks_log10 * np.ones(max_allocs), color="limegreen", label="BPF KS")
			ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)
			ax.set(ylim=(ks_glob_min, ks_glob_max))
			if i_mesh == N_MESHES - 1:
				ax.set_xlabel(r"$N_1$")

if mse_boxplot:
	if N_MESHES > 1:
		for i_mesh in range(N_MESHES):
			ax = sns.boxplot(data=pd.DataFrame(np.log10(mse_arr[i_mesh].T), columns=N1s), ax=axs[i_mesh], color=colors[i_mesh], whis=1000)
			ax.plot(range(max_allocs), bpf_median_mse_log10 * np.ones(max_allocs), color="limegreen", label="BPF")
			ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)
			if i_mesh == N_MESHES - 1:
				ax.set_xlabel(r"$N_1$")
			if i_mesh < N_MESHES - 1:
				ax.set_xticks([])
	else:
		for i_mesh in range(N_MESHES):
			ax = sns.boxplot(data=pd.DataFrame(np.log10(mse_arr[i_mesh].T), columns=N1s), ax=axs, color=colors[i_mesh])
			ax.plot(range(max_allocs), bpf_median_mse_log10 * np.ones(max_allocs), color="limegreen", label="BPF")
			ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)
			if i_mesh == N_MESHES - 1:
				ax.set_xlabel(r"$N_1$")



# ------------------------------------------------------------------------------------------------------------------- #
#
# Mean RMSE from reference point estimates
#
# ------------------------------------------------------------------------------------------------------------------- #
mins = []
if mean_rmse:
	axs2[0].plot(N1s[:max_allocs], bpf_mean_rmse_log10 * np.ones(max_allocs), color="black", label="BPF")
	for i_mesh in range(N_MESHES):
		means = np.empty(alloc_counters[i_mesh])
		for n_alloc in range(alloc_counters[i_mesh]):
			means[n_alloc] = np.mean(np.log10(rmse_arr[i_mesh, n_alloc, :]))
		axs2[0].plot(N1s[:alloc_counters[i_mesh]], means, label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
		mins.append(np.min(means))
	print(np.min(mins))
	axs2[0].set_title("Mean(log10(RMSE))", fontsize=9)
	axs2[0].legend()
	# axs2[0].set_xticks([])
print(bpf_mean_rmse_log10)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Median RMSE from reference point estimates
#
# ------------------------------------------------------------------------------------------------------------------- #
# axs2[0].set_title("log10(Median RMSE)", fontsize=9)
# axs2[0].plot(N1s[:max_allocs], bpf_median_rmse_log10 * np.ones(max_allocs), color="black", label="BPF")
# for i_mesh in range(N_MESHES):
# 	rmses = rmse_arr[i_mesh].T
# 	medians = []
# 	for n_alloc in range(alloc_counters[i_mesh]):
# 		rmses_new = []
# 		for x in rmses[:, n_alloc]:
# 			if not pd.isna(x):
# 				rmses_new.append(x)
# 		medians.append(np.median(np.log10(rmses_new)))
# 	axs2[0].plot(N1s[:alloc_counters[i_mesh]], medians, label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
# axs2[0].legend(loc=1, prop={'size': 7})
# axs2[0].set_xlabel(r"$N_1$")



# ------------------------------------------------------------------------------------------------------------------- #
#
# KS statistics from the reference distribution
#
# ------------------------------------------------------------------------------------------------------------------- #
axs2[1].set_title("Median(log10(KS statistics))", fontsize=9)
axs2[1].plot(N1s[:max_allocs], bpf_median_ks_log10 * np.ones(max_allocs), color="black", label="BPF")
for i_mesh in range(N_MESHES):
	axs2[1].plot(N1s[:alloc_counters[i_mesh]], np.median(np.log10(ks_arr[i_mesh, :alloc_counters[i_mesh], :].T), axis=0), label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
# axs2[1].set_xticks([])
# axs2[1].legend()



# ------------------------------------------------------------------------------------------------------------------- #
#
# Quantile RMSE estimation
#
# ------------------------------------------------------------------------------------------------------------------- #
# axs2[2].set_title("Quantile RMSE estimation", fontsize=9)
# axs2[2].plot(N1s[:max_allocs], np.mean(np.log10(bpf_centile_mse)) * np.ones(max_allocs), color="black", label="BPF")
# for i_mesh in range(N_MESHES):
# 	axs2[2].plot(N1s[:alloc_counters[i_mesh]], np.mean(np.log10(mlbpf_centile_mse_arr[i_mesh, :alloc_counters[i_mesh], :].T), axis=0), label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
# # axs2[0].set_xticks([])
# # axs2[2].legend()



# ------------------------------------------------------------------------------------------------------------------- #
#
# Reference standard deviations
#
# ------------------------------------------------------------------------------------------------------------------- #
# axs2[2].set_title("Ref stds", fontsize=9)
# for n in range(N_data):
# 	axs2[2].vlines(n * length, np.min(ref_stds), np.max(ref_stds), color="black")
# axs2[2].scatter(range(N_data * length), ref_stds, s=2)
# axs2[2].set_xlabel("Total iterate")



# ------------------------------------------------------------------------------------------------------------------- #
#
# Trial times
#
# ------------------------------------------------------------------------------------------------------------------- #
times_list = np.array(list(map(float, raw_times.readline().split())))
total_time_length = len(times_list)
axs3.set_title("Trial times", fontsize=9)
axs3.plot(range(total_time_length), np.array(times_list).flatten(), linewidth=0.5)
axs3.plot(range(total_time_length), np.mean(bpf_times) * np.ones(total_time_length), label="bpf mean", color="black")
axs3.set_xlabel("Trial")

plt.tight_layout()
plt.show()



