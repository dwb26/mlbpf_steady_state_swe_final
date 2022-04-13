import numpy as np
import matplotlib.pyplot as plt

corrections_f = open("corrections.txt", "r")
regression_f = open("regression_curve.txt", "r")
true_curve_f = open("true_curve.txt", "r")
true_curve_f0 = open("true_curve0.txt", "r")
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

# Read in the true correction samples that are used for the regression
N1 = 600
thetas = []; corrections = []
for line in corrections_f:
	theta, correction = list(map(float, line.split()))
	thetas.append(theta)
	corrections.append(correction)
thetas = np.array(thetas).reshape((length, N1))
corrections = np.array(corrections).reshape((length, N1))

# Read in the approximating polynomial
mesh_size = int(regression_f.readline())
mesh = []; points = []
for line in regression_f:
	m, p = list(map(float, line.split()))
	mesh.append(m)
	points.append(p)
sig_theta_mesh = np.array(mesh).reshape((length, mesh_size))
regression_points = np.array(points).reshape((length, mesh_size))

# Read in the artificial true correction values
true_curve = np.empty((length, mesh_size))
true_curve0 = np.empty((length, mesh_size))
k = 0
for line in true_curve_f:
	true_curve[k] = list(map(float, line.split()))
	k += 1
k = 0
for line in true_curve_f0:
	true_curve0[k] = list(map(float, line.split()))
	k += 1

# fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 7))
# m = 0
# custm_length = 8
# start_point = 0
# for k in range(start_point, start_point + custm_length):
# 	n = k - start_point
# 	# axs[m, n % 4].scatter(thetas[k], corrections[k], s=2, color="black", label="Samples")
# 	# axs[m, n % 4].plot(sig_theta_mesh[k], regression_points[k], label="Approximant")
# 	axs[m, n % 4].plot(sig_theta_mesh[k], true_curve[k], label="True soln")
# 	axs[m, n % 4].plot(sig_theta_mesh[k], true_curve0[k], label="0 soln")
# 	if (n + 1) % 4 == 0:
# 		m += 1
# 	if n == 0:
# 		axs[m, n % 4].legend()

# obs_disc = 250
# dx_obs = 3 * obs_sd / (obs_disc - 1)
# fig, axs = plt.subplots(2, 1, figsize=(12, 9))
# for k in range(length):

# 	# Plot the solution curves
# 	axs[k].plot(sig_theta_mesh[k], true_curve[k], label=r"$\varphi^1$")
# 	axs[k].plot(sig_theta_mesh[k], true_curve0[k], label=r"$\varphi^0$")

# 	# Plot the observation
# 	ext_obs = obs[k] * np.ones(mesh_size)
# 	alpha_vals = lambda x: np.exp(-(x - obs[k]) ** 2 / (2.0 * obs_sd ** 2))
# 	if k == 0:
# 		axs[k].vlines(signal[k], true_curve[k][500], obs[k], color="black", linestyle="--")
# 	axs[1].vlines(signal[1], true_curve[1][550], obs[1], color="black", linestyle="--")
# 	axs[k].plot(signal[k], obs[k], "ko", markersize=3)
# 	axs[k].plot(sig_theta_mesh[k], ext_obs, "k", label="obs")

# 	# Fill in the alpha noise representation
# 	for m in range(obs_disc):
# 		axs[k].fill_between(sig_theta_mesh[k], ext_obs - m * dx_obs, ext_obs - (m + 1) * dx_obs, color="gray", alpha=alpha_vals(obs[k] - m * dx_obs))
# 		axs[k].fill_between(sig_theta_mesh[k], ext_obs + m * dx_obs, ext_obs + (m + 1) * dx_obs, color="gray", alpha=alpha_vals(obs[k] + m * dx_obs))

# axs[1].set_xlabel(r"$\theta$", fontsize=14)
# plt.suptitle(r"Solution curves for $(J^0, J^1) = ({}, {})$".format(50, 500), fontsize=20)
# plt.tight_layout()
# axs[1].legend(prop={'size': 15})
# plt.show()


i = 0
obs_disc = 250
dx_obs = 3 * obs_sd / (obs_disc - 1)
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for k in range(length):

	# Plot the solution curves
	axs[i, k % 2].plot(sig_theta_mesh[k], true_curve[k], label=r"$\varphi^1$")
	axs[i, k % 2].plot(sig_theta_mesh[k], true_curve0[k], label=r"$\varphi^0$")

	# Plot the observation
	ext_obs = obs[k] * np.ones(mesh_size)
	alpha_vals = lambda x: np.exp(-(x - obs[k]) ** 2 / (2.0 * obs_sd ** 2))
	# axs[i, k % 2].vlines(signal[k], true_curve[k][500], obs[k], color="black", linestyle="--")
	axs[i, k % 2].plot(signal[k], obs[k], "ko", markersize=3)
	axs[i, k % 2].plot(sig_theta_mesh[k], ext_obs, "k", label="obs")

	# Fill in the alpha noise representation
	for m in range(obs_disc):
		axs[i, k % 2].fill_between(sig_theta_mesh[k], ext_obs - m * dx_obs, ext_obs - (m + 1) * dx_obs, color="gray", alpha=alpha_vals(obs[k] - m * dx_obs))
		axs[i, k % 2].fill_between(sig_theta_mesh[k], ext_obs + m * dx_obs, ext_obs + (m + 1) * dx_obs, color="gray", alpha=alpha_vals(obs[k] + m * dx_obs))

	if (k + 1) % 2 == 0:
		i += 1

# axs[1].set_xlabel(r"$\theta$", fontsize=14)
# plt.suptitle(r"Solution curves for $(J^0, J^1) = ({}, {})$".format(50, 500), fontsize=20)
plt.tight_layout()
axs[0, 0].legend(prop={'size': 15})
plt.show()





















