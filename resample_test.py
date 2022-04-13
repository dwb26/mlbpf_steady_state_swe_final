import numpy as np
import matplotlib.pyplot as plt

tv_measure_f = open("tv_measure.txt", "r")
res_distr_f = open("res_distr.txt", "r")

tv_x = []; tv_w = []
N0, N1, N_tot, length = list(map(int, tv_measure_f.readline().split()))
for line in tv_measure_f:
	x, w = list(map(float, line.split()))
	tv_x.append(x); tv_w.append(w)
tv_x = np.array(tv_x).reshape((length, N1))
tv_w = np.array(tv_w).reshape((length, N1))

res_x = []
for line in res_distr_f:
	res_x.extend(list(map(float, line.split())))
res_x = np.array(res_x)

plt.suptitle("Histograms of level 1 TV measure and the resampled distribution", fontsize=16)
plt.hist(tv_x[0], weights=tv_w[0], bins=80, density=True, alpha=0.3, label="Level 1 TV measure")
plt.hist(res_x, bins=80, density=True, color="yellow", alpha=0.3, label="Resample")
plt.legend()
plt.show()