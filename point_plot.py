import numpy as np
import matplotlib.pyplot as plt

x_hats_f = open("x_hats.txt", "r")
ml_xhats_f = open("ml_xhats.txt", "r")
hmm_data_f = open("hmm_data.txt", "r")
bpf_particles_f = open("bpf_particles.txt", "r")

length, N = list(map(int, bpf_particles_f.readline().split()))
length = int(hmm_data_f.readline())
for i in range(6):
	hmm_data_f.readline()
signal = np.empty(length); observations = np.empty(length)
for n in range(length):
	signal[n], observations[n] = list(map(float, hmm_data_f.readline().split()))
	
x_hats = np.array(list(map(float, x_hats_f.readline().split())))
# ml_xhats = np.array(list(map(float, ml_xhats_f.readline().split())))

fig = plt.figure(figsize=(16, 8))
ax = plt.subplot(111)
ax.scatter(range(length), signal, s=3, label="signal")
ax.scatter(range(length), x_hats, s=3, label="bpf")
# ax.scatter(range(length), ml_xhats, s=3, label="mlbpf")
ax.set_xlabel("$n$-th iterate", fontsize=16)
ax.set_ylabel(r"$\widehat{\theta}_n$", fontsize=16)
ax.legend()
plt.suptitle("Mean estimates for $N = {}$ BPF particles, nx0 = {}".format(1000, 250), fontsize=16)
plt.tight_layout()
plt.show()