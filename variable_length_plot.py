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
level0s = list(map(int, ml_parameters.readline().split()))
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
bpf_rmse = np.zeros((N_data, N_trials))
for n_data in range(N_data):
	ref_xhats[n_data] = np.array(list(map(float, ref_xhats_f.readline().split())))
for n_trial in range(N_data * N_trials):
	bpf_xhats[n_trial] = list(map(float, bpf_xhats_f.readline().split()))
if custm_length > length:
	raise Exception("Custom length mustn't exceed full length.")
else:
	ref_xhats = ref_xhats[:, :custm_length]
	bpf_xhats = bpf_xhats[:, :custm_length]

k = 0
for n_trial in range(N_data * N_trials):
	bpf_rmse[k, n_trial % N_trials] += np.sqrt(np.mean((ref_xhats[k] - bpf_xhats[n_trial]) ** 2))
	if (n_trial + 1) % N_trials == 0:
		k += 1
bpf_rmse = np.log10(bpf_rmse)
mean_bpf_rmse = np.mean(bpf_rmse)



# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)


print(mean_bpf_rmse)
# ---------------------------------------------------- Plotting ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
ax.plot(N1s[:max_allocs], mean_bpf_rmse * np.ones(max_allocs), color="black", label="BPF")
ax.set_title("Mean log10(RMSE) at n = {}".format(custm_length - 1))
ax.legend()
plt.show()