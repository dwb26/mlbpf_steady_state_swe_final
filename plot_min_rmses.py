import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

min_rmses_f = open("min_rmses.txt", "r")
stepwise_bpf_rmse_f = open("stepwise_bpf_rmse.txt", "r")
ml_parameters = open("ml_parameters.txt", "r")

N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf = list(map(int, ml_parameters.readline().split()))
min_rmses = np.array(list(map(float, min_rmses_f.readline().split())))
stepwise_bpf_rmse = np.array(list(map(float, stepwise_bpf_rmse_f.readline().split())))
plt.plot(range(len(min_rmses)), min_rmses, label="Optimal ML RMSE")
plt.plot(range(len(min_rmses)), stepwise_bpf_rmse, label="BPF RMSE", color="black")
plt.xlabel("Iterate")
print(stepwise_bpf_rmse[-1])
plt.title("N(BPF) = {}".format(N_bpf))
plt.legend()
plt.show()