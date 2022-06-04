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
# bpf_rmse = np.zeros((N_data, N_trials))
# We don't care about distinguishing between different data sets so collect all trials as one
bpf_rmse = np.zeros(N_data * N_trials)
for n_data in range(N_data):
	ref_xhats[n_data] = np.array(list(map(float, ref_xhats_f.readline().split())))
for n_trial in range(N_data * N_trials):
	bpf_xhats[n_trial] = list(map(float, bpf_xhats_f.readline().split()))
if custm_length > length:
	raise Exception("Custom length mustn't exceed full length.")
else: 	# Truncate the data to the desired length on which we analyse the RMSE
	ref_xhats = ref_xhats[:, :custm_length]
	bpf_xhats = bpf_xhats[:, :custm_length]

data_set = 0
for n_trial in range(N_data * N_trials):
	bpf_rmse[n_trial] += np.sqrt(np.mean((ref_xhats[data_set] - bpf_xhats[n_trial]) ** 2))
	if (n_trial + 1) % N_trials == 0:
		data_set += 1
bpf_rmse = np.log10(bpf_rmse)
mean_bpf_rmse = np.mean(bpf_rmse)



# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
level0s = list(map(int, ml_parameters.readline().split()))
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)
print(N1s)
print(alloc_counters)
l = []
ml_xhats = np.empty((N_data * N_trials, length))
for n in alloc_counters:
	l.append(N1s[:n])
for i_mesh in range(N_MESHES):
	nx0 = level0s[i_mesh]
	for N1 in l[i_mesh]:
		for n_data in range(N_data):
			raw_ml_xhats_f = open("raw_ml_xhats_nx0={}_N1={}_n_data={}.txt".format(nx0, N1, n_data), "r")
			for n_trial in range(N_trials):
				ml_xhats[n_data * N_trials + n_trial] = list(map(float, raw_ml_xhats_f.readline().split()))
		print("nx0 = {}, N1 = {}:::\n".format(nx0, N1))
		print(ml_xhats)
		print("\n")























print(mean_bpf_rmse)
# ---------------------------------------------------- Plotting ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
# ax.plot(N1s[:max_allocs], mean_bpf_rmse * np.ones(max_allocs), color="black", label="BPF")
# ax.set_title("Mean log10(RMSE) at n = {}".format(custm_length - 1))
# ax.legend()
# plt.show()