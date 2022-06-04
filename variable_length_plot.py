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
for n_data in range(N_data):
	ref_xhats[n_data] = np.array(list(map(float, ref_xhats_f.readline().split())))
for n_trial in range(N_data * N_trials):
	bpf_xhats[n_trial] = list(map(float, bpf_xhats_f.readline().split()))
if custm_length > length:
	raise Exception("Custom length mustn't exceed full length.")

data_set = 0
for n_trial in range(N_data * N_trials):
	rmse = np.sqrt(np.mean((ref_xhats[data_set, :custm_length] - bpf_xhats[n_trial, :custm_length]) ** 2))
	bpf_rmse[n_trial] += rmse
	if (n_trial + 1) % N_trials == 0:
		data_set += 1

# Take the mean of the RMSE over N_data * N_trials
mean_bpf_rmse = np.mean(np.log10(bpf_rmse))



# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
level0s = list(map(int, ml_parameters.readline().split()))
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)
ml_xhats = np.zeros((N_data * N_trials, length))
ml_rmse = np.zeros((N_MESHES, N_ALLOCS, N_data * N_trials))
l = []
for n in alloc_counters:
	l.append(N1s[:n])
for i_mesh in range(N_MESHES):
	nx0 = level0s[i_mesh]
	n_alloc = 0
	for N1 in l[i_mesh]:
		for n_data in range(N_data):
			raw_ml_xhats_f = open("raw_ml_xhats_nx0={}_N1={}_n_data={}.txt".format(nx0, N1, n_data), "r")
			for n_trial in range(N_trials):
				ml_xhats[n_data * N_trials + n_trial] = list(map(float, raw_ml_xhats_f.readline().split()))
		data_set = 0
		for n_trial in range(N_data * N_trials):
			rmse = np.sqrt(np.mean((ref_xhats[data_set, :custm_length] - ml_xhats[n_trial, :custm_length]) ** 2))
			ml_rmse[i_mesh, n_alloc, n_trial] += rmse
			if (n_trial + 1) % N_trials == 0:
				data_set += 1
		n_alloc += 1
mean_ml_rmse = np.zeros((N_MESHES, N_ALLOCS))
for i_mesh in range(N_MESHES):
	for n_alloc in range(alloc_counters[i_mesh]):
		mean_ml_rmse[i_mesh, n_alloc] = np.mean(np.log10(ml_rmse[i_mesh, n_alloc]))

# print(mean_bpf_rmse)



# ---------------------------------------------------- Plotting ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
ax.plot(N1s[:max_allocs], mean_bpf_rmse * np.ones(max_allocs), color="black", label="BPF")
for i_mesh in range(N_MESHES):
	ax.plot(N1s[:alloc_counters[i_mesh]], mean_ml_rmse[i_mesh, :alloc_counters[i_mesh]], label=level0s[i_mesh], marker="o", markersize=3)
	print(mean_ml_rmse[i_mesh, :alloc_counters[i_mesh]])
ax.set_title("Mean log10(RMSE) at n = {}".format(custm_length - 1))
ax.legend()
plt.show()


















# for i_mesh in range(N_MESHES):
# 	print("nx0 = {}\n".format(level0s[i_mesh]))
# 	for n_alloc in range(alloc_counters[i_mesh]):
# 		print("N1 = {}".format(N1s[n_alloc]))
# 		for n_trial in range(N_data * N_trials):
# 			print("RMSE = {}".format(ml_rmse[i_mesh, n_alloc, n_trial]))
# 		print("\n")
