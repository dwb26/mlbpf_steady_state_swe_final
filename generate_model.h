void generate_hmm(gsl_rng * rng, HMM * hmm, int n_data, int length, int nx);

double equal_runtimes_model(gsl_rng * rng, HMM * hmm, int ** N0s, int * N1s, w_double ** weighted_ref, int N_ref, int N_trials, int N_bpf, int * level0_meshes, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, w_double ** ml_weighted, FILE * BPF_CENTILE_MSE, FILE * REF_XHATS, FILE * BPF_XHATS, int rng_counter);

void output_hmm_parameters(FILE * DATA_OUT, int length, double sig_sd, double obs_sd, double space_left, double space_right, int nx, double k, double h_init, double q0, double lower_bound, double upper_bound);

void output_curve_solution(double * xs, double * Z, int nx, double k, double sig_theta, double * h, FILE * CURVE_DATA, FILE * TOP_DATA);

void read_hmm_data(char hmm_file_name[200], HMM * hmm, int n_data);

void run_reference_filter(gsl_rng * rng, HMM * hmm, int N_ref, w_double ** weighted_ref, int n_data);

void output_cdf(w_double ** w_particles, HMM * hmm, int N, char file_name[200]);

void read_cdf(w_double ** w_particles, HMM * hmm, int n_data);

double perform_BPF_trials(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, FILE * BPF_CENTILE_MSE, FILE * REF_XHATS, FILE * BPF_XHATS, int rng_counter);

double perform_BPF_trials_var_nx(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, FILE * BPF_CENTILE_MSE, int nx);

void compute_sample_sizes(HMM * hmm, gsl_rng * rng, int * level0_meshes, double T, int ** N0s, int * N1s, int N_bpf, int N_trials, w_double ** ml_weighted);

int compute_sample_sizes_bpf(HMM * hmm, gsl_rng * rng, double T, int nx, w_double ** weighted);

double read_sample_sizes(HMM * hmm, int ** N0s, int * N1s, int N_trials);

double ks_statistic(int N_ref, w_double * weighted_ref, int N, w_double * weighted);

double compute_mse(w_double ** weighted1, w_double ** weighted2, int length, int N1, int N2);

void compute_nth_percentile(w_double ** distr, int N, double centile, int length, double * centiles);