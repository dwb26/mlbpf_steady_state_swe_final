import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

bpf_particles_f = open("bpf_particles.txt", "r")
hmm_data_f = open("hmm_data.txt", "r")

N_bins = 50
length, N = list(map(int, bpf_particles_f.readline().split()))
A = np.zeros((N_bins, length))
data = np.empty((length, N))
for n in range(length):
	data[n] = np.array(list(map(float, bpf_particles_f.readline().split())))
x_min = np.min(data)
x_max = np.max(data)
bins = np.linspace(x_min, x_max, N_bins)
for n in range(length):
	bin_indices = np.digitize(data[n], bins) - 1
	for b in bin_indices:
		A[b, n] += 1
A /= N

bins_str = []
for i in bins:
	bins_str.append("%1.1f" % (i))

m = (x_max - x_min) / (N_bins - 1)
c = x_min

def f_map(x, m, c):
	return m * x + c

def f_inv(x, m, c):
	y = (x - c) / m
	return y

fig, ax = plt.subplots(figsize=(16, 8))
# im = plt.imshow(A)
sns.heatmap(data=A)
ax.set_xlabel("$n$-th iterate", fontsize=16)
ax.set_ylabel(r"$\theta_n$", fontsize=16)
plt.suptitle("Sequential empirical posterior histograms for $N = {}$ particles".format(N), fontsize=16)

N_ticks = 10
ticks = np.linspace(0, N_bins - 1, N_ticks)
labels = np.linspace(x_min, x_max, N_ticks)
labels_str = []
for l in labels:
	labels_str.append("%1.2f" % (l))
ax.set_yticks(ticks=ticks, labels=labels_str)

for i in range(6):
	hmm_data_f.readline()
signal = np.empty(length)
for n in range(length):
	signal[n] = list(map(float, hmm_data_f.readline().split()))[0]
	ax.plot(n, f_inv(signal[n], m, c), "gx", markersize=7, label="True signal")
	if n == 0:
		ax.legend()

plt.tight_layout()
plt.show()



