import numpy as np
import matplotlib.pyplot as plt

curve_data_f = open("curve_data.txt", "r")
top_data_f = open("top_data.txt", "r")
hmm_data_f = open("hmm_data.txt", "r")
length = int(hmm_data_f.readline())
hmm_data_f.readline()
space_left, space_right = list(map(float, hmm_data_f.readline().split()))
nx = int(hmm_data_f.readline())
k = float(hmm_data_f.readline())
h0, q0 = list(map(float, hmm_data_f.readline().split()))
low_bd, upp_bd = list(map(float, hmm_data_f.readline().split()))
thetas = np.empty(length)
obs = np.empty(length)
m = 0
for line in hmm_data_f:
	thetas[m], obs[m] = list(map(float, line.split()))
	m += 1
xs = np.linspace(space_left, space_right, nx)
curves = np.empty((length, nx))
Z = np.empty((length, nx))

n = 0
for line in curve_data_f:
	curves[n] = list(map(float, line.split()))
	n += 1

n = 0
for line in top_data_f:
	Z[n] = list(map(float, line.split()))
	n += 1
surface_level = curves + Z

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
fig.subplots_adjust(hspace=0.4)
m = 0
surface_max = np.max(surface_level)
curve_max = np.max(curves)
Z_min = np.min(Z)
custm_length = 8
start_point = 0
for k in range(start_point, start_point + custm_length):
	n = k - start_point
	axs[m, n % 4].plot(xs, Z[k], color="black", label="Z")
	axs[m, n % 4].plot(xs, curves[k] + Z[k], color="blue", label="h")
	# axs[m, n % 4].plot(xs, surface_level[k], color="blue", label="h + Z")
	# axs[m, n % 4].set(ylim=(Z_min - 0.1, curve_max + -Z_min / 10))
	axs[m, n % 4].set_title(r"$\theta_{} = {:.2f}$, $y_{} = {:.2f}$".format(k, thetas[k], k, obs[k]))
	axs[m, n % 4].set_xlabel("x")
	axs[m, n % 4].set(ylim=(np.min(Z) - 1, np.max(curves + Z) + 1))
	if (n + 1) %  4 == 0:
		m += 1
	if n == 0:
		axs[0, 0].legend()
plt.suptitle(r"Steady state SWE ODE solutions", fontsize=18)
plt.show()