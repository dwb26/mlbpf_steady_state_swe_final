import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

N0 = 1000; N1 = 500
length = 8
lhood_obs_f = open("lhood_obs.txt", "r")
lhood_obs0_f = open("lhood_obs0.txt", "r")
sig_thetas = []; g1s = []; g0s = []
for line in	lhood_obs_f:
	s, a, b = list(map(float, line.split()))
	sig_thetas.append(s)
	g1s.append(a)
	g0s.append(b)
l0_sig_thetas = []; l0_g0s = []
for line in lhood_obs0_f:
	s, g = list(map(float, line.split()))
	# l0_sig_thetas.append(s)
	# l0_g0s.append(g)

sig_thetas = np.array(sig_thetas).reshape((length, N1))
g1s = np.array(g1s).reshape((length, N1))
g0s = np.array(g0s).reshape((length, N1))
# l0_sig_thetas = np.array(l0_sig_thetas).reshape((length, N0))
# l0_g0s = np.array(l0_g0s).reshape((length, N0))

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 7))
k = 0
for n in range(length):
	# axs[k, n % 4].scatter(sig_thetas[n], g1s[n] - g0s[n], s=2)
	axs[k, n % 4].scatter(sig_thetas[n], g1s[n], s=2, label="g1")
	axs[k, n % 4].scatter(sig_thetas[n], g0s[n], s=2, label="g0")
	# axs[k, n % 4].scatter(l0_sig_thetas[n], l0_g0s[n], s=2, label="l0 g0")
	if (n + 1) % 4 == 0:
		k += 1
axs[0, 0].legend()
plt.show()