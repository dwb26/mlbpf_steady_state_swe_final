import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

nx1 = 300
nx0 = 25
length = 8
g = 9.81
k = 10.0; gamma_of_k = gamma(k)
h_init = 2.0; q0 = 1.0
q0_sq = q0 * q0
space_left = 0.0; space_right = 50.0
dx1 = (space_right - space_left) / (nx1 - 1)
dx0 = (space_right - space_left) / (nx0 - 1)
xs1 = np.linspace(space_left, space_right, nx1)
xs0 = np.linspace(space_left, space_right, nx0)

def gen_Z_topography(xs, nx, k, theta, gamma_of_k):
	return -10 * xs ** (k - 1) * np.exp(-xs / theta) / (gamma_of_k * theta ** k)

def gen_Z_x_topography(xs, nx, k, theta, gamma_of_k):
	Z_x = -10 * np.exp(-xs / theta) / (gamma_of_k * theta ** k) * ((k - 1) * xs ** (k - 2) - xs ** (k - 1) / theta)
	return Z_x

def target(h, Z_x, q0_sq):
	return q0_sq / (g * h ** 3) - Z_x

def RK4(hn, Z_xn, dx, q0_sq):
	k1 = target(hn, Z_xn, q0_sq)
	k2 = target(hn + 0.5 * dx * k1, Z_xn + 0.5 * dx * k1, q0_sq);
	k3 = target(hn + 0.5 * dx * k2, Z_xn + 0.5 * dx * k2, q0_sq);
	k4 = target(hn + dx * k3, Z_xn + dx * k3, q0_sq);
	return hn + 1 / 6.0 * dx * (k1 + 2 * k2 + 2 * k3 + k4);

def solve(k, theta, nx, xs, h, dx, q0_sq, gamma_of_k):
	Z_x = gen_Z_x_topography(xs, nx, k, theta, gamma_of_k)
	for j in range(1, nx):
		h[j] = RK4(h[j - 1], Z_x[j - 1], dx, q0_sq)
	return h


hmm_data_f = open("hmm_data.txt", "r")
for n in range(7):
	hmm_data_f.readline()

signal = np.empty(length)
for n in range(length):
	x, y = list(map(float, hmm_data_f.readline().split()))
	signal[n] = x

h1s = np.empty((length, nx1))
h0s = np.empty((length, nx0))
Z1s = np.empty((length, nx1))
Z0s = np.empty((length, nx0))
h1s[:, 0] = h_init
h0s[:, 0] = h_init
for n in range(length):
	Z1s[n] = gen_Z_topography(xs1, nx1, k, signal[n], gamma_of_k)
	Z0s[n] = gen_Z_topography(xs0, nx0, k, signal[n], gamma_of_k)
	h1s[n] = solve(k, signal[n], nx1, xs1, h1s[n], dx1, q0_sq, gamma_of_k)
	h0s[n] = solve(k, signal[n], nx0, xs0, h0s[n], dx0, q0_sq, gamma_of_k)

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
plt.subplots_adjust(hspace=0.6)
m = 0
for n in range(length):
	axs[m, n % 4].plot(xs1, Z1s[n])
	axs[m, n % 4].plot(xs1, h1s[n])
	axs[m, n % 4].plot(xs0, Z0s[n])
	axs[m, n % 4].plot(xs0, h0s[n])
	diff = h1s[n, -1] - h0s[n, -1]
	axs[m, n % 4].set_title("{}".format(diff))
	if (n + 1) % 4 == 0:
		m += 1
plt.show()































