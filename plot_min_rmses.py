import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Could maybe rewrite this with the xhats instead

min_rmses_f = open("min_rmses.txt", "r")
stepwise_bpf_rmse_f = open("stepwise_bpf_rmse.txt", "r")
ml_parameters = open("ml_parameters.txt", "r")

N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf = list(map(int, ml_parameters.readline().split()))
min_rmses = np.array(list(map(float, min_rmses_f.readline().split())))
stepwise_bpf_rmse = np.array(list(map(float, stepwise_bpf_rmse_f.readline().split())))
# ratio_min_rmses = min_rmses / stepwise_bpf_rmse
ratio_min_rmses = stepwise_bpf_rmse / min_rmses
print(stepwise_bpf_rmse)

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
axs[0].plot(range(len(min_rmses)), min_rmses, label="Optimal ML RMSE")
axs[0].plot(range(len(min_rmses)), stepwise_bpf_rmse, label="BPF RMSE", color="black")
axs[0].set_xticks([])
axs[0].set_title("N(BPF) = {}".format(N_bpf))
axs[0].legend()

axs[1].plot(range(len(min_rmses)), ratio_min_rmses, label="ML / BPF")
axs[1].plot(range(len(min_rmses)), np.ones(len(min_rmses)), color="black")
axs[1].set_xlabel("Iterate")
axs[1].set_title("Ratio")
axs[1].legend()
plt.show()