	typedef struct {
	double * signal;
	double * observations;
	int length;
	int nx;
	double sig_sd;
	double obs_sd;
	double space_left;
	double space_right;
	double k;
	double h_init;
	double q0;
	int N_LEVELS;
	int N_MESHES;
	int N_ALLOCS;
	double low_bd;
	double upp_bd;
} HMM;

typedef struct weighted_double {
	double x;
	double w;
} w_double;

int weighted_double_cmp(const void * a, const void * b);
void bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted);
void ml_bootstrap_particle_filter_res(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios);
void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios);
void ml_bootstrap_particle_filter_debug(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, FILE * L2_ERR_DATA);
void ml_bootstrap_particle_filter_timed(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios);
double sigmoid(double x, double a, double b);
double sigmoid_inv(double x, double a, double b);
void mutate(gsl_rng * rng, int N_tot, double * thetas, double * res_thetas, double sig_sd);
void ref_bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted);
void bootstrap_particle_filter_var_nx(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted, int nx);