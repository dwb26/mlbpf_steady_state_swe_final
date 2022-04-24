void generate_hmm(gsl_rng * rng, HMM * hmm, int n_data, int length, int nx);

void equal_runtimes_model(gsl_rng * rng, HMM * hmm, int ** N0s, int * N1s, w_double ** weighted_ref, int N_ref, int N_trials, int N_bpf, int * level0_meshes, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, w_double ** ml_weighted, FILE * BPF_CENTILE_MSE);

void run_reference_filter(gsl_rng * rng, HMM * hmm, int N_ref, w_double ** weighted_ref, int n_data);

void output_cdf(w_double ** w_particles, HMM * hmm, int N, char file_name[200]);

void read_cdf(w_double ** w_particles, HMM * hmm, int n_data);

double perform_BPF_trials(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, FILE * BPF_CENTILE_MSE);

void compute_sample_sizes(HMM * hmm, gsl_rng * rng, int * level0_meshes, double T, int ** N0s, int * N1s, int N_bpf, int N_trials, w_double ** ml_weighted);

double read_sample_sizes(HMM * hmm, int ** N0s, int * N1s, int N_trials);

double ks_statistic(int N_ref, w_double * weighted_ref, int N, w_double * weighted);

double compute_mse(w_double ** weighted1, w_double ** weighted2, int length, int N1, int N2);

void compute_nth_percentile(w_double ** distr, int N, double centile, int length, double * centiles);