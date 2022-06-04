import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

ref_xhats_f = open("ref_xhats.txt", "r")
bpf_xhats_f = open("bpf_xhats.txt", "r")
hmm_data = open("hmm_data.txt", "r")
ml_parameters = open("ml_parameters.txt", "r")
N1s_data = open("N1s_data.txt", "r")
alloc_counters_f = open("alloc_counters.txt", "r")


# --------------------------------------------------- HMM data ------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf = list(map(int, ml_parameters.readline().split()))
length = int(hmm_data.readline())
sig_sd, obs_sd = list(map(float, hmm_data.readline().split()))
hmm_data.readline()
nx = int(hmm_data.readline())
for n in range(2):
    hmm_data.readline()



# -------------------------------------------------- BPF results ---------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
custm_length = length
ref_xhats = np.empty((N_data, length))
bpf_xhats = np.empty((N_data * N_trials, length))
bpf_rmse = np.zeros(N_data * N_trials)
var_bpf_rmse = np.zeros((N_data * N_trials, length))
for n_data in range(N_data):
	ref_xhats[n_data] = np.array(list(map(float, ref_xhats_f.readline().split())))
for n_trial in range(N_data * N_trials):
	bpf_xhats[n_trial] = list(map(float, bpf_xhats_f.readline().split()))
if custm_length > length:
	raise Exception("Custom length mustn't exceed full length.")

# Take the RMSE at each trial over the custom length data set
data_set = 0
for n_trial in range(N_data * N_trials):
	rmse = np.sqrt(np.mean((ref_xhats[data_set, :custm_length] - bpf_xhats[n_trial, :custm_length]) ** 2))
	bpf_rmse[n_trial] = rmse
	for n in range(length):
		rmse = np.sqrt(np.mean((ref_xhats[data_set, :n + 1] - bpf_xhats[n_trial, :n + 1]) ** 2))
		var_bpf_rmse[n_trial, n] = rmse
	if (n_trial + 1) % N_trials == 0:
		data_set += 1

# Take the mean of the RMSE over N_data * N_trials
mean_bpf_rmse = np.mean(np.log10(bpf_rmse))
var_mean_bpf_rmse = np.mean(np.log10(var_bpf_rmse), axis=0)



# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
level0s = list(map(int, ml_parameters.readline().split()))
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)
ml_xhats = np.zeros((N_data * N_trials, length))
ml_rmse = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
var_ml_rmse = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials, length))
mean_ml_rmse = np.zeros((N_MESHES, N_ALLOCS))
var_mean_ml_rmse = np.zeros((N_MESHES, N_ALLOCS, length))
min_mean_ml_rmse = np.empty(length)
max_mean_ml_rmse = np.empty(length)

for i_mesh in range(N_MESHES):
	nx0 = level0s[i_mesh]
	for n_alloc in range(alloc_counters[i_mesh]):
		N1 = N1s[n_alloc]
		for n_data in range(N_data):
			raw_ml_xhats_f = open("raw_ml_xhats_nx0={}_N1={}_n_data={}.txt".format(nx0, N1, n_data), "r")
			for n_trial in range(N_trials):
				ml_xhats[n_data * N_trials + n_trial] = list(map(float, raw_ml_xhats_f.readline().split()))
		data_set = 0
		for n_trial in range(N_data * N_trials):
			rmse = np.sqrt(np.mean((ref_xhats[data_set, :custm_length] - ml_xhats[n_trial, :custm_length]) ** 2))
			ml_rmse[i_mesh, n_alloc, n_trial] = rmse
			for n in range(length):
				rmse = np.sqrt(np.mean((ref_xhats[data_set, :n + 1] - ml_xhats[n_trial, :n + 1]) ** 2))
				var_ml_rmse[i_mesh, n_alloc, n_trial, n] = rmse
			if (n_trial + 1) % N_trials == 0:
				data_set += 1

for i_mesh in range(N_MESHES):
	for n_alloc in range(alloc_counters[i_mesh]):
		mean_ml_rmse[i_mesh, n_alloc] = np.mean(np.log10(ml_rmse[i_mesh, n_alloc]))
		for n in range(length):
			var_mean_ml_rmse[i_mesh, n_alloc, n] = np.mean(np.log10(var_ml_rmse[i_mesh, n_alloc, :, n]))

for n in range(length):
	mins = []; maxs = []
	for i_mesh in range(N_MESHES):
		mins.append(np.min(var_mean_ml_rmse[i_mesh, :alloc_counters[i_mesh], n]))
		maxs.append(np.max(var_mean_ml_rmse[i_mesh, :alloc_counters[i_mesh], n]))
	min_mean_ml_rmse[n] = np.min(mins)
	max_mean_ml_rmse[n] = np.max(maxs)
glob_min = np.min((np.min(min_mean_ml_rmse), np.min(var_mean_bpf_rmse)))
glob_max = np.max((np.max(max_mean_ml_rmse), np.max(var_mean_bpf_rmse)))
# print((glob_min, glob_max))
# print(var_mean_bpf_rmse)
# print(min_mean_ml_rmse)



# ---------------------------------------------------- Plotting ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #

# This is the final iterate RMSE plot #
# ----------------------------------- #
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
ax1.plot(N1s[:max_allocs], mean_bpf_rmse * np.ones(max_allocs), color="black", label="BPF")
for i_mesh in range(N_MESHES):
	ax1.plot(N1s[:alloc_counters[i_mesh]], mean_ml_rmse[i_mesh, :alloc_counters[i_mesh]], label=level0s[i_mesh], marker="o", markersize=3)
ax1.set_title("Mean log10(RMSE) at n = {}".format(custm_length - 1))
ax1.legend()


# These are the min RMSE plots #
# ---------------------------- #
fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
ax2[0].plot(range(length), var_mean_bpf_rmse, color="black", label="BPF RMSE")
ax2[0].plot(range(length), min_mean_ml_rmse, color="red", label="Optimal MLBPF RMSE")
ax2[0].set_title("N(BPF) = {}".format(N_bpf))
ax2[0].set_xticks([])
ax2[0].legend()

ratio_min_rmses = var_mean_bpf_rmse / min_mean_ml_rmse
ax2[1].plot(range(length), ratio_min_rmses, label="ML / BPF")
ax2[1].plot(range(length), np.ones(length), color="black")
ax2[1].set_xlabel("Iterate")
ax2[1].set_title("Ratio")
ax2[1].legend()


# These are the sequential RMSE plots #
# ----------------------------------- #
for n in range(length):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
	ax.plot(N1s[:max_allocs], var_mean_bpf_rmse[n] * np.ones(max_allocs), color="black", label="BPF")
	for i_mesh in range(N_MESHES):
		ax.plot(N1s[:alloc_counters[i_mesh]], var_mean_ml_rmse[i_mesh, :alloc_counters[i_mesh], n], label=level0s[i_mesh], marker="o", markersize=3)
	ax.set(ylim=(glob_min, glob_max))
	ax.set_title("N_BPF = {}, n = {}".format(N_bpf, n))
	ax.set_xlabel("N1")
	ax.legend()
	plt.savefig("seq_RMSE_n={}.png".format(n))

plt.show()











