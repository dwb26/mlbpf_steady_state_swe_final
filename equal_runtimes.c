// gcc -o equal_runtimes -lm -lgsl -lgslcblas equal_runtimes.c particle_filters.c solvers.c generate_model.c
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "particle_filters.h"
#include "solvers.h"
#include "generate_model.h"

const int N_TOTAL_MAX = 100000000;
const int N_LEVELS = 2;
const int N_ALLOCS = 7;
const int N_MESHES = 8;
// const int N_ALLOCS = 1;
// const int N_MESHES = 1;

void output_parameters(int N_trials, int * level0_meshes, int nx, int * N1s, int N_data, int N_bpf);
void record_reference_data(HMM * hmm, w_double ** weighted_ref, int N_ref, FILE * FULL_HMM_DATA, FILE * FULL_REF_DATA, FILE * REF_STDS);
void output_ml_data(HMM * hmm, int N_trials, double *** raw_times, double *** raw_ks, double *** raw_mse, double *** raw_srs, int * level0_meshes, int * N1s, int * alloc_counters, FILE * ALLOC_COUNTERS, FILE * RAW_TIMES, FILE * RAW_KS, FILE * RAW_MSE, FILE * RAW_SRS, int N_data, FILE * MLBPF_CENTILE_MSE, double *** raw_qmses);

static int compare (const void * a, const void * b)
{
  if (*(double*)a > *(double*)b) return 1;
  else if (*(double*)a < *(double*)b) return -1;
  else return 0;  
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// This is the 1d steady state shallow water equations model
int main(void) {

	clock_t timer = clock();
	gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);


	/* Main experiment parameters */
	/* -------------------------- */
	int N_data = 1;
	int N_trials = 10;
	int length = 50, nx = 500;
	int N_ref = 500000;
	int N_bpf = 1000;
	// int N_bpf = 2500;
	// int N_bpf = 5000;
	// int N_bpf = 10000;
	// int N_bpf = 15000;	
	// int level0_meshes[N_MESHES] = { 300, 200, 150, 100, 50, 25 };
	// int level0_meshes[N_MESHES] = { 200, 150, 100, 50 };
	// int level0_meshes[N_MESHES] = { 80, 60, 40, 25 };
	// int level0_meshes[N_MESHES] = { 750, 500, 250, 200, 150, 100, 50 };
	// int level0_meshes[N_MESHES] = { 175, 125, 100, 75, 50 };
	int level0_meshes[N_MESHES] = { 350, 250, 200, 150, 125, 100, 75, 50 }; ///
	// int level0_meshes[N_MESHES] = { 650, 450, 250, 125, 75 };
	// int level0_meshes[N_MESHES] = { 650 };
	int N1s[N_ALLOCS] = { 0, 50, 120, 250, 500, 750, 900 }; ///
	// int N1s[N_ALLOCS] = { 0, 50 };
	// int N1s[N_ALLOCS] = { 0, 125, 250, 500, 1000 };
	// int N1s[N_ALLOCS] = { 0, 125, 250, 500, 1000, 1500, 2000 };
	// int N1s[N_ALLOCS] = { 0, 100, 200, 300, 400, 500, 600, 700, 800, 900 };
	// int N1s[N_ALLOCS] = { 0, 100, 250, 500, 1000, 2500, 5000 };
	// int N1s[N_ALLOCS] = { 0, 25, 50, 100 };
	// int level0_meshes[N_MESHES] = { 100 };
	// int N1s[N_ALLOCS] = { 500 };
	int nxs[N_LEVELS] = { 0, nx };
	int ** N0s = (int **) malloc(N_MESHES * sizeof(int *));	
	int * sample_sizes = (int *) malloc(N_LEVELS * sizeof(int));
	int * alloc_counters = (int *) malloc(N_MESHES * sizeof(int));
	double * sign_ratios = (double *) calloc(length, sizeof(double));
	double * mlbpf_centiles = (double *) malloc(length * sizeof(double));
	double *** raw_ks = (double ***) malloc(N_MESHES * sizeof(double **));
	double *** raw_mse = (double ***) malloc(N_MESHES * sizeof(double **));
	double *** raw_times = (double ***) malloc(N_MESHES * sizeof(double **));
	double *** raw_srs = (double ***) malloc(N_MESHES * sizeof(double **));
	double *** raw_qmses = (double ***) malloc(N_MESHES * sizeof(double **));
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		N0s[i_mesh] = (int *) malloc(N_ALLOCS * sizeof(int));
		raw_ks[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		raw_mse[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		raw_times[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		raw_srs[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		raw_qmses[i_mesh] = (double **) malloc(N_ALLOCS * sizeof(double *));
		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {
			raw_ks[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
			raw_mse[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
			raw_times[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
			raw_srs[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
			raw_qmses[i_mesh][n_alloc] = (double *) calloc(N_data * N_trials, sizeof(double));
		}
	}
	w_double ** weighted_ref = (w_double **) malloc(length * sizeof(w_double *));
	w_double ** weighted_long = (w_double **) malloc(length * sizeof(w_double *));
	w_double ** ml_weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++) {
		weighted_ref[n] = (w_double *) malloc(N_ref * sizeof(w_double));
		weighted_long[n] = (w_double *) malloc(N_ref * sizeof(w_double));
		ml_weighted[n] = (w_double *) malloc(N_TOTAL_MAX * sizeof(w_double));
	}
	FILE * RAW_BPF_TIMES = fopen("raw_bpf_times.txt", "w");
	FILE * RAW_BPF_KS = fopen("raw_bpf_ks.txt", "w");
	FILE * RAW_BPF_MSE = fopen("raw_bpf_mse.txt", "w");
	FILE * RAW_TIMES = fopen("raw_times.txt", "w");
	FILE * RAW_KS = fopen("raw_ks.txt", "w");
	FILE * RAW_MSE = fopen("raw_mse.txt", "w");
	FILE * RAW_SRS = fopen("raw_srs.txt", "w");
	FILE * ALLOC_COUNTERS = fopen("alloc_counters.txt", "w");
	FILE * REF_STDS = fopen("ref_stds.txt", "w");
	FILE * FULL_HMM_DATA = fopen("full_hmm_data.txt", "w");
	FILE * FULL_REF_DATA = fopen("full_ref_data.txt", "w");
	FILE * BPF_CENTILE_MSE = fopen("bpf_centile_mse.txt", "w");
	FILE * MLBPF_CENTILE_MSE = fopen("mlbpf_centile_mse.txt", "w");
	output_parameters(N_trials, level0_meshes, nx, N1s, N_data, N_bpf);

	int ML_experiment, BPF_experiment;
	ML_experiment = 1;
	// ML_experiment = 0;
	// BPF_experiment = 1;
	BPF_experiment = 0;


	/* ----------------------------------------------------------------------------------------------------- */
	/*																										 																									 */
	/* Accuracy trials 																				 																							 */
	/* 																																																			 */
	/* ----------------------------------------------------------------------------------------------------- */
	int N0, N1, N_tot;
	double ks, sr, ref_xhat, ml_xhat, q_mse, centile = 0.95;
	double * ref_centiles = (double *) malloc(length * sizeof(double));
	HMM * hmm = (HMM *) malloc(sizeof(HMM));
	hmm->N_LEVELS = N_LEVELS, hmm->N_MESHES = N_MESHES, hmm->N_ALLOCS = N_ALLOCS;
	if (ML_experiment == 1) {

		for (int n_data = 0; n_data < N_data; n_data++) {

			/* Generate the HMM data and run the BPF on it */
			/* ------------------------------------------- */
			generate_hmm(rng, hmm, n_data, length, nx);
			sr = equal_runtimes_model(rng, hmm, N0s, N1s, weighted_ref, N_ref, N_trials, N_bpf, level0_meshes, n_data, RAW_BPF_TIMES, RAW_BPF_KS, RAW_BPF_MSE, ml_weighted, BPF_CENTILE_MSE);
			record_reference_data(hmm, weighted_ref, N_ref, FULL_HMM_DATA, FULL_REF_DATA, REF_STDS);
			compute_nth_percentile(weighted_ref, N_ref, centile, length, ref_centiles);


			/* Run the MLBPF for each nx0/particle allocation for the same time as the BPF and test its accuracy */
			/* ------------------------------------------------------------------------------------------------- */
			for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {

				printf("--------------\n");
				printf("|  N1 = %d  |\n", N1s[n_alloc]);
				printf("--------------\n");

				for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {

					printf("nx0 = %d\n", level0_meshes[i_mesh]);
					printf("**********************************************************\n");

					nxs[0] = level0_meshes[i_mesh];
					N1 = N1s[n_alloc], N0 = N0s[i_mesh][n_alloc];
					// N0 = 25; N1 = 15;
					N_tot = N0 + N1;
					sample_sizes[0] = N0, sample_sizes[1] = N1;
					alloc_counters[i_mesh] = N_ALLOCS;

					ref_xhat = 0.0, ml_xhat = 0.0;
					if (N0 == 0) {
						alloc_counters[i_mesh] = n_alloc;
						for (int n_trial = 0; n_trial < N_trials; n_trial++) {
							raw_mse[i_mesh][n_alloc][n_data * N_trials + n_trial] = -1;
							raw_ks[i_mesh][n_alloc][n_data * N_trials + n_trial] = -1;
							raw_qmses[i_mesh][n_alloc][n_data * N_trials + n_trial] = -1;
						}
					}
					else {
						for (int n_trial = 0; n_trial < N_trials; n_trial++) {
							clock_t trial_timer = clock();
							ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
							double elapsed = (double) (clock() - trial_timer) / (double) CLOCKS_PER_SEC;

							ks = 0.0, sr = 0.0, q_mse = 0.0;
							for (int n = 0; n < length; n++) {
								qsort(ml_weighted[n], N_tot, sizeof(w_double), weighted_double_cmp);
								ks += ks_statistic(N_ref, weighted_ref[n], N_tot, ml_weighted[n]) / (double) length;
							}
							compute_nth_percentile(ml_weighted, N_tot, centile, length, mlbpf_centiles);
							for (int n = 0; n < length; n++)
								q_mse += (ref_centiles[n] - mlbpf_centiles[n]) * (ref_centiles[n] - mlbpf_centiles[n]);
							raw_qmses[i_mesh][n_alloc][n_data * N_trials + n_trial] = sqrt(q_mse / (double) N_trials);
							raw_mse[i_mesh][n_alloc][n_data * N_trials + n_trial] = compute_mse(weighted_ref, ml_weighted, length, N_ref, N_tot);
							raw_ks[i_mesh][n_alloc][n_data * N_trials + n_trial] = ks;
							raw_times[i_mesh][n_alloc][n_data * N_trials + n_trial] = elapsed;
						}
					}
					printf("\n");
				}
			}
		}
		output_ml_data(hmm, N_trials, raw_times, raw_ks, raw_mse, raw_srs, level0_meshes, N1s, alloc_counters, ALLOC_COUNTERS, RAW_TIMES, RAW_KS, RAW_MSE, RAW_SRS, N_data, MLBPF_CENTILE_MSE, raw_qmses);
	}

	int N_nxs = 60, nx_incr = 5, N;
	int * nx_bpfs = (int *) malloc(N_nxs * sizeof(int));
	double T, T_bin;
	for (int n = 1; n <= N_nxs; n++)
		nx_bpfs[n - 1] = nx - n * nx_incr;
	if (BPF_experiment == 1) {

		for (int n_data = 0; n_data < N_data; n_data++) {

			/* Generate the HMM data and run the BPF with the full nx on it */
			/* ------------------------------------------------------------ */
			generate_hmm(rng, hmm, n_data, length, nx);
			T = equal_runtimes_model(rng, hmm, N0s, N1s, weighted_ref, N_ref, N_trials, N_bpf, level0_meshes, n_data, RAW_BPF_TIMES, RAW_BPF_KS, RAW_BPF_MSE, ml_weighted, BPF_CENTILE_MSE);

			for (int n = 0; n < N_nxs; n++) {

				/* Compute the sample size to run for the same time as the full nx */
				printf("Computing N for nx = %d...\n", nx_bpfs[n]);
				N = compute_sample_sizes_bpf(hmm, rng, T, nx_bpfs[n], weighted_long);
				printf("N = %d\n", N);

				/* Find the accuracy of the BPF run on the lower nx against the reference solution */
				printf("Running trials for nx = %d...\n", nx_bpfs[n]);
				perform_BPF_trials_var_nx(hmm, N, rng, N_trials, N_ref, weighted_ref, n_data, RAW_BPF_TIMES, RAW_BPF_KS, RAW_BPF_MSE, BPF_CENTILE_MSE, nx_bpfs[n]);
			}
		}
	}

	fclose(RAW_BPF_TIMES);
	fclose(RAW_BPF_KS);
	fclose(RAW_BPF_MSE);
	fclose(RAW_TIMES);
	fclose(RAW_KS);
	fclose(RAW_MSE);
	fclose(RAW_SRS);
	fclose(ALLOC_COUNTERS);
	fclose(REF_STDS);
	fclose(FULL_HMM_DATA);
	fclose(FULL_REF_DATA);

	double total_elapsed = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
	int hours = (int) floor(total_elapsed / 3600.0);
	int minutes = (int) floor((total_elapsed - hours * 3600) / 60.0);
	int seconds = (int) (total_elapsed - hours * 3600 - minutes * 60);
	printf("Total time for experiment = %d hours, %d minutes and %d seconds\n", hours, minutes, seconds);

	return 0;
}


/* --------------------------------------------------------------------------------------------------------------------
 *
 * Functions
 *
 * ----------------------------------------------------------------------------------------------------------------- */
void output_parameters(int N_trials, int * level0_meshes, int nx, int * N1s, int N_data, int N_bpf) {

	FILE * ML_PARAMETERS = fopen("ml_parameters.txt", "w");
	FILE * N1s_DATA = fopen("N1s_data.txt", "w");

	fprintf(ML_PARAMETERS, "%d %d %d %d %d \n", N_data, N_trials, N_ALLOCS, N_MESHES, N_bpf);
	for (int m = 0; m < N_MESHES; m++)
		fprintf(ML_PARAMETERS, "%d ", level0_meshes[m]);
	fprintf(ML_PARAMETERS, "\n");
	fprintf(ML_PARAMETERS, "%d\n", nx);
	for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++)
		fprintf(N1s_DATA, "%d ", N1s[n_alloc]);
	fprintf(N1s_DATA, "\n");

	fclose(ML_PARAMETERS);
	fclose(N1s_DATA);

}


void record_reference_data(HMM * hmm, w_double ** weighted_ref, int N_ref, FILE * FULL_HMM_DATA, FILE * FULL_REF_DATA, FILE * REF_STDS) {

	int length = hmm->length;
	double std = 0.0, EX = 0.0, EX2 = 0.0;

	for (int n = 0; n < length; n++) {
		fprintf(FULL_HMM_DATA, "%e %e\n", hmm->signal[n], hmm->observations[n]);
			EX = 0.0, EX2 = 0.0;
		for (int i = 0; i < N_ref; i++) {
			fprintf(FULL_REF_DATA, "%e %e\n", weighted_ref[n][i].x, weighted_ref[n][i].w);
			EX += weighted_ref[n][i].x * weighted_ref[n][i].w;
			EX2 += weighted_ref[n][i].x * weighted_ref[n][i].x * weighted_ref[n][i].w;
		}
		std = sqrt(EX2 - EX * EX);
		fprintf(REF_STDS, "%e ", std);
	}

}


void output_ml_data(HMM * hmm, int N_trials, double *** raw_times, double *** raw_ks, double *** raw_mse, double *** raw_srs, int * level0_meshes, int * N1s, int * alloc_counters, FILE * ALLOC_COUNTERS, FILE * RAW_TIMES, FILE * RAW_KS, FILE * RAW_MSE, FILE * RAW_SRS, int N_data, FILE * MLBPF_CENTILE_MSE, double *** raw_qmses) {

	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		fprintf(ALLOC_COUNTERS, "%d ", alloc_counters[i_mesh]);
		printf("Alloc counters for nx0 = %d = %d\n", level0_meshes[i_mesh], alloc_counters[i_mesh]);
	}
	fprintf(ALLOC_COUNTERS, "\n");

	/* Work horizontally from top left to bottom right, writing the result from each trial, new line when finished */
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {
			for (int n_trial = 0; n_trial < N_data * N_trials; n_trial++) {
				if (raw_times[i_mesh][n_alloc][n_trial] == 0)
					;
				else
					fprintf(RAW_TIMES, "%e ", raw_times[i_mesh][n_alloc][n_trial]);
				if (isnan(raw_mse[i_mesh][n_alloc][n_trial]))
					fprintf(RAW_MSE, "%d ", -2);
				else
					fprintf(RAW_MSE, "%e ", raw_mse[i_mesh][n_alloc][n_trial]);
				if (isnan(raw_ks[i_mesh][n_alloc][n_trial]))
					fprintf(RAW_KS, "%d ", -2);
				else
					fprintf(RAW_KS, "%e ", raw_ks[i_mesh][n_alloc][n_trial]);
				fprintf(RAW_SRS, "%e ", raw_srs[i_mesh][n_alloc][n_trial]);
				fprintf(MLBPF_CENTILE_MSE, "%e ", raw_qmses[i_mesh][n_alloc][n_trial]);
			}

			fprintf(RAW_KS, "\n");
			fprintf(RAW_MSE, "\n");
			fprintf(RAW_SRS, "\n");
			fprintf(MLBPF_CENTILE_MSE, "\n");
		}
	}
}


							// sr += sign_ratios[n] / (double) length;
							// for (int i = 0; i < N_ref; i++)
								// ref_xhat += weighted_ref[n][i].x * weighted_ref[n][i].w;
							// ref_xhat /= (double) N_ref;
							// for (int i = 0; i < N_tot; i++)
								// ml_xhat += ml_weighted[n][i].x * ml_weighted[n][i].w;
							// ml_xhat /= (double) N_tot;
							// double rmse = sqrt((ref_xhat - ml_xhat) * (ref_xhat - ml_xhat));
							// printf("RAW RMSE FOR NX0 = %d = %e\n", level0_meshes[i_mesh], rmse);
							// printf("RAW KS FOR NX0 = %d = %e\n", level0_meshes[i_mesh], ks);