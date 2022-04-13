import numpy as np
import matplotlib.pyplot as plt

hmm_data_f = open("hmm_data.txt", "r")
full_ref_data_f = open("full_ref_data.txt", "r")
bpf_distr_f = open("bpf_distr.txt", "r");
ml_distr_f = open("ml_distr.txt", "r");
raw_bpf_ks_f = open("raw_bpf_ks.txt", "r")
raw_ks_f = open("raw_ks.txt", "r")

bpf_ks = []
bpf_ks.extend(list(map(float, raw_bpf_ks_f.readline().split())))
bpf_median_ks = np.median(bpf_ks)
bpf_median_ks_log10 = np.median(np.log10(bpf_ks))
raw_ks = list(map(float, raw_ks_f.readline().split()))
median_raw_ks = np.median(raw_ks)
median_raw_ks_log10 = np.median(np.log10(raw_ks))
length = int(hmm_data_f.readline())

# Read in the reference distribution and sort by the particle value
N_ref = 1000000
ref_x = np.empty((length, N_ref))
ref_w = np.empty((length, N_ref))
for n in range(length):
	for i in range(N_ref):
		ref_x[n, i], ref_w[n, i] = list(map(float, full_ref_data_f.readline().split()))
for n in range(length):
	ind_sort = np.argsort(ref_x[n])
	ref_x[n] = ref_x[n][ind_sort]
	ref_w[n] = ref_w[n][ind_sort]
	# print(np.sum(ref_w[n]))


# Read in the BPF distribution and sort by the particle value
N_bpf = int(bpf_distr_f.readline())
bpf_x = np.empty((length, N_bpf))
bpf_w = np.empty((length, N_bpf))
for n in range(length):
	for i in range(N_bpf):
		bpf_x[n, i], bpf_w[n, i] = list(map(float, bpf_distr_f.readline().split()))
for n in range(length):
	ind_sort = np.argsort(bpf_x[n])
	bpf_x[n] = bpf_x[n][ind_sort]
	bpf_w[n] = bpf_w[n][ind_sort]


# Read in the multilevel distribution and sort by the particle value
N0, N1, N_tot = list(map(int, ml_distr_f.readline().split()))
ml_x = np.empty((length, N_tot))
ml_w = np.empty((length, N_tot))
for n in range(length):
	for i in range(N_tot):
		ml_x[n, i], ml_w[n, i] = list(map(float, ml_distr_f.readline().split()))
for n in range(length):
	ind_sort = np.argsort(ml_x[n])
	ml_x[n] = ml_x[n][ind_sort]
	ml_w[n] = ml_w[n][ind_sort]


# Compute the CDF for each distribution
def gen_cdf(x, w):
	F = np.empty(len(x))
	F[0] = w[0]
	for i in range(1, len(x)):
		F[i] = w[i] + F[i - 1]
	return F

F_ref = np.empty((length, N_ref))
F_bpf = np.empty((length, N_bpf))
F_ml = np.empty((length, N_tot))
for n in range(length):
	F_ref[n] = gen_cdf(ref_x[n], ref_w[n])
	F_bpf[n] = gen_cdf(bpf_x[n], bpf_w[n])
	F_ml[n] = gen_cdf(ml_x[n], ml_w[n])


# Plot the CDFs
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
n = 0
axs.plot(ref_x[n], F_ref[n], label="Ref")
axs.plot(bpf_x[n], F_bpf[n], label="BPF")
axs.plot(ml_x[n], F_ml[n], label="ML")
axs.legend(prop={'size': 16})
axs.set_xlabel(r"$\theta$", fontsize=16)
# plt.suptitle("BPF log10 KS = {:.5}, ML log10 KS = {:.5}".format(bpf_median_ks_log10, median_raw_ks_log10))
plt.suptitle("BPF KS = {:.5}, ML KS = {:.5}".format(bpf_median_ks, median_raw_ks), fontsize=18)
plt.tight_layout()
plt.show()














