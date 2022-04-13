import numpy as np
import matplotlib.pyplot as plt

true_curve_f = open("true_curve.txt", "r")
degree0_f = open("true_curve0_M=0.txt", "r")
degree1_f = open("true_curve0_M=1.txt", "r")
degree2_f = open("true_curve0_M=2.txt", "r")
regression_curve_f = open("regression_curve.txt", "r")
hmm_data_f = open("hmm_data.txt", "r")

# Read in the HMM data
length = int(hmm_data_f.readline())
sig_sd, obs_sd = list(map(float, hmm_data_f.readline().split()))
space_left, space_right = list(map(float, hmm_data_f.readline().split()))
nx = int(hmm_data_f.readline())
k_parameter = float(hmm_data_f.readline())
h0, q0 = list(map(float, hmm_data_f.readline().split()))
lb, up = list(map(float, hmm_data_f.readline().split()))
signal = np.empty(length); obs = np.empty(length)
for n in range(length):
	signal[n], obs[n] = list(map(float, hmm_data_f.readline().split()))

# Read in the mesh to plot the solutions on
mesh_size = int(regression_curve_f.readline())
sig_theta_mesh = []
for line in regression_curve_f:
	x, y = list(map(float, line.split()))
	sig_theta_mesh.append(x)
sig_theta_mesh = np.array(sig_theta_mesh).reshape((length, mesh_size))

# Read in the level 1 true solution
phi1 = np.empty((length, mesh_size))
n = 0
for line in true_curve_f:
	phi1[n] = np.array(list(map(float, line.split())))
	n += 1

# Read in the transformed level 0 solutions
phi0_M0 = np.empty((length, mesh_size))
phi0_M1 = np.empty((length, mesh_size))
phi0_M2 = np.empty((length, mesh_size))
for n in range(length):
	phi0_M0[n] = list(map(float, degree0_f.readline().split()))
	phi0_M1[n] = list(map(float, degree1_f.readline().split()))
	phi0_M2[n] = list(map(float, degree2_f.readline().split()))
phi0s = [phi0_M0[0], phi0_M1[0], phi0_M2[0]]

# Plot the lot
ext_obs = obs[0] * np.ones(mesh_size)
obs_disc = 50
dx_obs = 3 * obs_sd / (obs_disc - 1)
alpha_vals = lambda x: np.exp(-(x - obs[0]) ** 2 / (2.0 * obs_sd ** 2))
fig, ax = plt.subplots(3, 1, figsize=(12, 9))
for n in range(3):
	ax[n].plot(sig_theta_mesh[0], phi1[0], label=r"$\varphi^1$")
	ax[n].plot(sig_theta_mesh[0], phi0s[n], label=r"$\varphi^0$, M={}".format(n), linestyle="--", color="orange")
	ax[n].vlines(signal[0], phi1[0][500], obs[0], color="black")
	ax[n].plot(signal[0], obs[0], "ko", markersize=3)
	ax[n].plot(sig_theta_mesh[0], ext_obs, "k", label="obs")
	for k in range(obs_disc):
		ax[n].fill_between(sig_theta_mesh[0], ext_obs - k * dx_obs, ext_obs - (k + 1) * dx_obs, color="gray", alpha=alpha_vals(obs[0] - k * dx_obs))
		ax[n].fill_between(sig_theta_mesh[0], ext_obs + k * dx_obs, ext_obs + (k + 1) * dx_obs, color="gray", alpha=alpha_vals(obs[0] + k * dx_obs))
	ax[n].legend()
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[2].set_xlabel(r"$\theta$", fontsize=14)
plt.suptitle("Level 1 solution and translated level 0 solution for \n varying regression approximations, for (nx0, nx1) = (50, 250)", fontsize=18)
plt.tight_layout()
plt.show()