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
#include <assert.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include "particle_filters.h"
#include "solvers.h"
#include "generate_model.h"

// This is the 1d steady state shallow water equations model

double equal_runtimes_model(gsl_rng * rng, HMM * hmm, int ** N0s, int * N1s, w_double ** weighted_ref, int N_ref, int N_trials, int N_bpf, int * level0_meshes, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, w_double ** ml_weighted, FILE * BPF_CENTILE_MSE, FILE * REF_XHATS, FILE * BPF_XHATS, double * ref_xhats, double * bpf_rmses, int rng_counter) {

	// int run_ref = 1;		// REF ON
	int run_ref = 0;		// REF OFF

	/* Reference distribution */
	/* ---------------------- */
	/* Produce the benchmark reference distribution with the BPF. Set run_ref to 0 if the reference data already esists */
	if (run_ref == 1) {
		run_reference_filter(rng, hmm, N_ref, weighted_ref, n_data);
		rng_counter++;
		gsl_rng_set(rng, rng_counter);

		for (int n = 0; n < 3; n++) {
			for (int i = 0; i < 25; i++)
				printf("(w, x)[%d, %d] = %e, %e\n", n, i, weighted_ref[n][i].w, weighted_ref[n][i].x);
		}
		printf("\n");
	}
	else
		read_cdf(weighted_ref, hmm, n_data);


	/* Sample allocation */
	/* ----------------- */
	/* Run the BPF with a set number of particles N_bpf < N_ref and record the accuracy and the mean time taken. Then for each mesh configuration, increment the level 1 particle allocation and compute the level 0 particle allocation so that the time taken for the MLBPF is roughly the same as the BPF */
	double T, T_temp;
	T = perform_BPF_trials(hmm, N_bpf, rng, N_trials, N_ref, weighted_ref, n_data, RAW_BPF_TIMES, RAW_BPF_KS, RAW_BPF_MSE, BPF_CENTILE_MSE, REF_XHATS, BPF_XHATS, ref_xhats, bpf_rmses, rng_counter);
	// if (n_data == 0)
		// compute_sample_sizes(hmm, rng, level0_meshes, T, N0s, N1s, N_bpf, N_trials, ml_weighted);
	T_temp = read_sample_sizes(hmm, N0s, N1s, N_trials);

	return T;

}


void generate_hmm(gsl_rng * rng, HMM * hmm, int n_data, int length, int nx) {

	/** 
	Generates the HMM data and outputs to file to be read in by read_hmm.
	*/
	int obs_pos = nx - 1;
	double sig_sd = 4.0;
	// double sig_sd = 5.5;
	// double obs_sd = 0.1 * 0.037881;
	// double obs_sd = 0.025;
	// double obs_sd = 0.075;
	double obs_sd = 0.1;
	// double obs_sd = 0.15;
	// double obs_sd = 0.25;
	double space_left = 0.0, space_right = 50.0;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double k = 3.5, theta, obs;
	double h_init = 2.0, q0 = 3.0, q0_sq = q0 * q0;	
	double lower_bound = 1.5, upper_bound = 8.0 + lower_bound;
	double Z_obs, gamma_of_k = tgamma(k), gamma_theta, Z_obs_ind, sig_theta = 4.5;
	double * h = (double *) malloc(nx * sizeof(double));
	double * Z = (double *) malloc(nx * sizeof(double));
	double * Z_x = (double *) malloc(nx * sizeof(double));
	double * xs = (double *) malloc(nx * sizeof(double));
	for (int j = 0; j < nx; j++)
		xs[j] = space_left + j * dx;
	h[0] = h_init;
	theta = sigmoid_inv(sig_theta, upper_bound, lower_bound);

	/* Write the available parameters */
	FILE * DATA_OUT = fopen("hmm_data.txt", "w");
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * TOP_DATA = fopen("top_data.txt", "w");
	fprintf(DATA_OUT, "%d\n", length);
	fprintf(DATA_OUT, "%lf %lf\n", sig_sd, obs_sd);
	fprintf(DATA_OUT, "%lf %lf\n", space_left, space_right);
	fprintf(DATA_OUT, "%d\n", nx);
	fprintf(DATA_OUT, "%lf\n", k);
	fprintf(DATA_OUT, "%lf %lf\n", h_init, q0);
	fprintf(DATA_OUT, "%lf %lf\n", lower_bound, upper_bound);

	int N = 1000;
	double EY, varY, top_varY = 0.0;
	double * thetas = (double *) malloc(N * sizeof(double));
	double * solns = (double *) malloc(N * sizeof(double));
	double lmbda = 1.0 * M_PI;

	/* Generate the data */
	for (int n = 0; n < length; n++) {

		sig_theta = sigmoid(theta, upper_bound, lower_bound);
		gamma_theta = gamma_of_k * pow(sig_theta, k);
		solve(k, sig_theta, nx, xs, Z_x, h, dx, q0_sq, gamma_theta);
		obs = h[obs_pos] + gsl_ran_gaussian(rng, obs_sd);
		fprintf(DATA_OUT, "%e %e\n", sig_theta, obs);

		double true_soln = h[obs_pos];
		gen_Z_topography(xs, Z, nx, k, sig_theta);
		for (int j = 0; j < nx; j++) {
			fprintf(CURVE_DATA, "%e ", h[j]);
			fprintf(TOP_DATA, "%e ", Z[j]);
		}
		fprintf(CURVE_DATA, "\n");
		fprintf(TOP_DATA, "\n");

		/* Evolve the signal with the mutation model */
		theta = 0.9999 * theta + gsl_ran_gaussian(rng, sig_sd);

	}
	printf("Average observation stds = %lf\n", top_varY / (double) length);

	fclose(DATA_OUT);
	fclose(CURVE_DATA);	
	fclose(TOP_DATA);

	free(h);
	free(Z);
	free(Z_x);
	free(xs);
	free(thetas);
	free(solns);

	/* Read in the data from the file */
	FILE * DATA_IN = fopen("hmm_data.txt", "r");
	fscanf(DATA_IN, "%d\n", &hmm->length);
	fscanf(DATA_IN, "%lf %lf\n", &hmm->sig_sd, &hmm->obs_sd);
	fscanf(DATA_IN, "%lf %lf\n", &hmm->space_left, &hmm->space_right);
	fscanf(DATA_IN, "%d\n", &hmm->nx);
	fscanf(DATA_IN, "%lf\n", &hmm->k);
	fscanf(DATA_IN, "%lf %lf\n", &hmm->h_init, &hmm->q0);
	fscanf(DATA_IN, "%lf %lf\n", &hmm->low_bd, &hmm->upp_bd);
	hmm->signal = (double *) malloc(hmm->length * sizeof(double));
	hmm->observations = (double *) malloc(hmm->length * sizeof(double));
	for (int n = 0; n < hmm->length; n++)
		fscanf(DATA_IN, "%lf %lf\n", &hmm->signal[n], &hmm->observations[n]);
	fclose(DATA_IN);

	printf("DATA SET %d\n", n_data + 1);
	printf("Data length 	             = %d\n", hmm->length);
	printf("sig_sd      	             = %lf\n", hmm->sig_sd);
	printf("obs_sd      	             = %lf\n", hmm->obs_sd);
	printf("nx          	             = %d\n", hmm->nx);
	printf("k           	             = %lf\n", hmm->k);
	printf("h_init, q0         	         = %lf %lf\n", hmm->h_init, hmm->q0);
	printf("low_bd, upp_bd         	     = %lf %lf\n", hmm->low_bd, hmm->upp_bd);
	for (int n = 0; n < hmm->length; n++)
		printf("n = %d: signal = %lf, observation = %lf\n", n, hmm->signal[n], hmm->observations[n]);

}


void run_reference_filter(gsl_rng * rng, HMM * hmm, int N_ref, w_double ** weighted_ref, int n_data) {

	double ref_elapsed;
	char sig_sd_str[50], obs_sd_str[50], len_str[50], s0_str[50], n_data_str[50], ref_name[200];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(s0_str, 50, "%lf", hmm->signal[0]);
	snprintf(n_data_str, 50, "%d", n_data);
	sprintf(ref_name, "ref_particles_sig_sd=%s_obs_sd=%s_len=%s_s0=%s_n_data=%s.txt", sig_sd_str, obs_sd_str, len_str, s0_str, n_data_str);
	puts(ref_name);

	/* Run the BPF with the reference number of particles */
	printf("Running reference BPF...\n");
	clock_t ref_timer = clock();
	ref_bootstrap_particle_filter(hmm, N_ref, rng, weighted_ref);
	ref_elapsed = (double) (clock() - ref_timer) / (double) CLOCKS_PER_SEC;
	printf("Reference BPF for %d particles completed in %f seconds\n", N_ref, ref_elapsed);

	/* Sort and output the weighted particles for the KS tests */
	for (int n = 0; n < hmm->length; n++)
		qsort(weighted_ref[n], N_ref, sizeof(w_double), weighted_double_cmp);
	output_cdf(weighted_ref, hmm, N_ref, ref_name);
}


void output_cdf(w_double ** w_particles, HMM * hmm, int N, char file_name[200]) {

	FILE * data = fopen(file_name, "w");
	fprintf(data, "%d %d\n", N, hmm->length);

	for (int n = 0; n < hmm->length; n++) {
		for (int i = 0; i < N; i++)
			fprintf(data, "%.16e ", w_particles[n][i].x);
		fprintf(data, "\n");
		for (int i = 0; i < N; i++)
			fprintf(data, "%.16e ", w_particles[n][i].w);
		fprintf(data, "\n");
	}
	fclose(data);
}


void read_cdf(w_double ** w_particles, HMM * hmm, int n_data) {

	int N, length;
	char sig_sd_str[50], obs_sd_str[50], len_str[50], s0_str[50], n_data_str[50], ref_name[200];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(s0_str, 50, "%lf", hmm->signal[0]);
	snprintf(n_data_str, 50, "%d", n_data);
	sprintf(ref_name, "ref_particles_sig_sd=%s_obs_sd=%s_len=%s_s0=%s_n_data=%s.txt", sig_sd_str, obs_sd_str, len_str, s0_str, n_data_str);
	FILE * data = fopen(ref_name, "r");
	fscanf(data, "%d %d\n", &N, &length);
	puts(ref_name);

	for (int n = 0; n < length; n++) {
		for (int i = 0; i < N; i++)
			fscanf(data, "%le ", &w_particles[n][i].x);
		for (int i = 0; i < N; i++)
			fscanf(data, "%le ", &w_particles[n][i].w);
	}
	fclose(data);
	for (int n = 0; n < 3; n++) {
		for (int i = 0; i < 25; i++)
			printf("(w, x)[%d, %d] = %e, %e\n", n, i, w_particles[n][i].w, w_particles[n][i].x);
	}
	printf("\n");
}


double perform_BPF_trials(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, FILE * BPF_CENTILE_MSE, FILE * REF_XHATS, FILE * BPF_XHATS, double * ref_xhats, double * bpf_rmses, int rng_counter) {

	int length = hmm->length;
	double ks = 0.0, elapsed = 0.0, mse = 0.0, mean_elapsed = 0.0, q_mse = 0.0, bpf_xhat = 0.0, centile = 0.95;
	double * ref_centiles = (double *) malloc(length * sizeof(double));
	compute_nth_percentile(weighted_ref, N_ref, centile, length, ref_centiles);
	double * bpf_centiles = (double *) malloc(length * sizeof(double));
	w_double ** weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		weighted[n] = (w_double *) malloc(N_bpf * sizeof(w_double));

	/* Write out the current data set reference x_hats */
	for (int n = 0; n < length; n++) {
		for (int i = 0; i < N_ref; i++)
			ref_xhats[n] += weighted_ref[n][i].w * weighted_ref[n][i].x;
		fprintf(REF_XHATS, "%.16e ", ref_xhats[n]);
	}
	fprintf(REF_XHATS, "\n");

	printf("Running BPF trials...\n");
	for (int n_trial = 0; n_trial < N_trials; n_trial++) {

		/* Run the simulation for the BPF */
		clock_t bpf_timer = clock();
		bootstrap_particle_filter(hmm, N_bpf, rng, weighted);
		elapsed = (double) (clock() - bpf_timer) / (double) CLOCKS_PER_SEC;
		mean_elapsed += elapsed;
		rng_counter++;
		gsl_rng_set(rng, rng_counter);
		

		/* Trial analysis */
		/* -------------- */
		ks = 0.0;
		for (int n = 0; n < length; n++) {

			/* Compute the BPF KS statistic for the trial */
			qsort(weighted[n], N_bpf, sizeof(w_double), weighted_double_cmp);
			ks += ks_statistic(N_ref, weighted_ref[n], N_bpf, weighted[n]) / (double) length;

			/* Compute the BPF mean estimate for the trial */
			bpf_xhat = 0.0;
			for (int i = 0; i < N_bpf; i++)
				bpf_xhat += weighted[n][i].w * weighted[n][i].x;
			fprintf(BPF_XHATS, "%.16e ", bpf_xhat);
		}
		fprintf(BPF_XHATS, "\n");

		mse = compute_mse(weighted_ref, weighted, length, N_ref, N_bpf);
		compute_nth_percentile(weighted, N_bpf, centile, length, bpf_centiles);
		q_mse = 0.0;
		for (int n = 0; n < length; n++)
			q_mse += (ref_centiles[n] - bpf_centiles[n]) * (ref_centiles[n] - bpf_centiles[n]);
		fprintf(BPF_CENTILE_MSE, "%.16e ", sqrt(q_mse / (double) length));
		fprintf(RAW_BPF_TIMES, "%.16e ", elapsed);
		fprintf(RAW_BPF_KS, "%.16e ", ks);
		fprintf(RAW_BPF_MSE, "%.16e ", mse);

		for (int n = 0; n < length; n++) /////
			bpf_rmses[n] += log10(sqrt(compute_mse(weighted_ref, weighted, n + 1, N_ref, N_bpf))) / (double) N_trials;

	}

	fprintf(RAW_BPF_TIMES, "\n");
	fprintf(RAW_BPF_KS, "\n");
	fprintf(RAW_BPF_MSE, "\n");
	
	free(weighted);
	free(ref_centiles);
	free(bpf_centiles);

	return mean_elapsed / (double) N_trials;

}


double perform_BPF_trials_var_nx(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, FILE * BPF_CENTILE_MSE, int nx) {

	int length = hmm->length;
	double centile = 0.95;
	double ks = 0.0, elapsed = 0.0, mse = 0.0, mean_elapsed = 0.0, q_mse = 0.0;
	double * ref_centiles = (double *) malloc(length * sizeof(double));
	compute_nth_percentile(weighted_ref, N_ref, centile, length, ref_centiles);	
	double * bpf_centiles = (double *) malloc(length * sizeof(double));
	w_double ** weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		weighted[n] = (w_double *) malloc(N_ref * sizeof(w_double));

	for (int n_trial = 0; n_trial < N_trials; n_trial++) {

		/* Run the simulation for the BPF */
		clock_t bpf_timer = clock();
		bootstrap_particle_filter_var_nx(hmm, N_bpf, rng, weighted, nx);
		elapsed = (double) (clock() - bpf_timer) / (double) CLOCKS_PER_SEC;
		mean_elapsed += elapsed;

		/* Compute the KS statistic for the run */
		ks = 0.0;
		for (int n = 0; n < length; n++) {
			qsort(weighted[n], N_bpf, sizeof(w_double), weighted_double_cmp);
			ks += ks_statistic(N_ref, weighted_ref[n], N_bpf, weighted[n]) / (double) length;
		}

		mse = compute_mse(weighted_ref, weighted, length, N_ref, N_bpf);
		compute_nth_percentile(weighted, N_bpf, centile, length, bpf_centiles);
		q_mse = 0.0;
		for (int n = 0; n < length; n++)
			q_mse += (ref_centiles[n] - bpf_centiles[n]) * (ref_centiles[n] - bpf_centiles[n]);
		fprintf(BPF_CENTILE_MSE, "%.16e ", sqrt(q_mse / (double) length));
		fprintf(RAW_BPF_TIMES, "%.16e ", elapsed);
		fprintf(RAW_BPF_KS, "%.16e ", ks);
		fprintf(RAW_BPF_MSE, "%.16e ", mse);

	}

	fprintf(RAW_BPF_TIMES, "\n");
	fprintf(RAW_BPF_KS, "\n");
	fprintf(RAW_BPF_MSE, "\n");
	
	free(weighted);
	free(ref_centiles);
	free(bpf_centiles);

	return mean_elapsed / (double) N_trials;

}


void compute_sample_sizes(HMM * hmm, gsl_rng * rng, int * level0_meshes, double T, int ** N0s, int * N1s, int N_bpf, int N_trials, w_double ** ml_weighted) {

	/* Variables to compute the sample sizes */
	/* ------------------------------------- */
	int N0, N0_lo, dist;
	int N_LEVELS = hmm->N_LEVELS, N_MESHES = hmm->N_MESHES, N_ALLOCS = hmm->N_ALLOCS;
	double T_mlbpf, diff;
	clock_t timer;


	/* Variables to run the MLBPF */
	/* -------------------------- */
	int length = hmm->length;
	int * nxs = (int *) calloc(N_LEVELS, sizeof(int));
	int * sample_sizes = (int *) malloc(N_LEVELS * sizeof(int));
	double * sign_ratios = (double *) calloc(length, sizeof(double));
	nxs[N_LEVELS - 1] = hmm->nx;


	/* Variables for printing to file */
	/* ------------------------------ */
	FILE * N0s_f = fopen("N0s_data.txt", "w");
	fprintf(N0s_f, "%d %.16e\n", N_bpf, T);


	/* Compute the particle allocations */
	/* -------------------------------- */
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {

		nxs[0] = level0_meshes[i_mesh];
		printf("Computing the level 0 allocations for nx0 = %d\n", nxs[0]);

		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++)
			N0s[i_mesh][n_alloc] = (int) (nxs[1] / (double) nxs[0] * N_bpf);
		printf("Starting guess for N0:\n");
		printf("%d\n", N0s[i_mesh][0]);
		printf("\n");

		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {

			sample_sizes[1] = N1s[n_alloc];
			printf("N1 = %d\n", N1s[n_alloc]);

			N0 = N_bpf;
			N0 = N0s[i_mesh][n_alloc];
			sample_sizes[0] = N0;
			N0_lo = N0;

			/* Find a value for N0_init that exceeds the required time */
			clock_t timer = clock();
			ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff < 0) {
				N0 *= 2;
				sample_sizes[0] = N0;

				timer = clock();
				ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;
			}

			/* Find a value for N0_lo that does not meet the required time */
			sample_sizes[0] = N0_lo;
			timer = clock();
			ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff > 0) {
				N0_lo = (int) (N0_lo / 2.0);
				sample_sizes[0] = N0_lo;

				timer = clock();
				ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;

				if (N0_lo == 0)
					diff = 0;
			}

			/* Run with the N0 we know exceeds the required time */
			sample_sizes[0] = N0;
			timer = clock();
			ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;

			if (N0_lo == 0)
				sample_sizes[0] = 0;

			else {

				/* Halve the interval until a sufficiently accurate root is found */
				while (fabs(diff) >= 0.1) {
					if (diff > 0)
						N0 = (int) (0.5 * (N0_lo + N0));
					else {
						dist = N0 - N0_lo;
						N0_lo = N0;
						N0 += dist;
					}
					sample_sizes[0] = N0;

					timer = clock();
					for (int i = 0; i < 1; i++)
						ml_bootstrap_particle_filter(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
					T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
					diff = (T_mlbpf - T) / T;

					if (N0_lo == N0)
						diff = 0.0;
				}
			}

			N0s[i_mesh][n_alloc] = sample_sizes[0];
			printf("N0 = %d for N1 = %d and nx0 = %d, timed diff = %.10lf\n", sample_sizes[0], N1s[n_alloc], nxs[0], diff);
			printf("\n");
			fprintf(N0s_f, "%d ", sample_sizes[0]);

		}

		fprintf(N0s_f, "\n");

	}

	fclose(N0s_f);

	free(nxs);
	free(sign_ratios);
	free(sample_sizes);

}


int compute_sample_sizes_bpf(HMM * hmm, gsl_rng * rng, double T, int nx, w_double ** weighted) {

	/**
	 * 
	 * In this function we compute the N_bpf required to run the BPF for the same as T(nx1), given nx < nx1.
	 * 
	 * */

	int N_lo = 500, N_hi = 3000, N_bpf;
	clock_t timer;
	int length = hmm->length;
	double T_lo, T_hi, T_bpf, diff;

	/* Make sure we find a low sample size for which we know the root is greater than */
	timer = clock();
	bootstrap_particle_filter_var_nx(hmm, N_lo, rng, weighted, nx);
	T_lo = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
	diff = (T_lo - T) / T;
	while (diff > 0) {
		T_lo = (int) (0.5 * T_lo);
		timer = clock();
		bootstrap_particle_filter_var_nx(hmm, N_lo, rng, weighted, nx);
		T_lo = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
		diff = (T_lo - T) / T;
	}

	/* Make sure we find a low sample size for which we know the root is less than */
	timer = clock();
	bootstrap_particle_filter_var_nx(hmm, N_hi, rng, weighted, nx);
	T_hi = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
	diff = (T_hi - T) / T;
	while (diff < 0) {
		T_hi = (int) (2 * T_hi);
		timer = clock();
		bootstrap_particle_filter_var_nx(hmm, N_hi, rng, weighted, nx);
		T_hi = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
		diff = (T_hi - T) / T;
	}

	/* Find the root N_bpf in between [N_lo, N_hi] */
	N_bpf = (int) (0.5 * (N_lo + N_hi));
	timer = clock();
	bootstrap_particle_filter_var_nx(hmm, N_bpf, rng, weighted, nx);
	T_bpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
	diff = (T_bpf - T) / T;
	while (fabs(diff) >= 0.01) {
		if (diff > 0) {
			N_hi = N_bpf;
			N_bpf = (int) (0.5 * (N_lo + N_hi));
			timer = clock();
			bootstrap_particle_filter_var_nx(hmm, N_bpf, rng, weighted, nx);
			T_bpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_bpf - T) / T;
		}			
		else {
			N_lo = N_bpf;
			N_bpf = (int) (0.5 * (N_lo + N_hi));
			timer = clock();
			bootstrap_particle_filter_var_nx(hmm, N_bpf, rng, weighted, nx);
			T_bpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_bpf - T) / T;
		}
		if (N_lo == N_hi)
			diff = 0.0;
	}

	return N_bpf;
}


double read_sample_sizes(HMM * hmm, int ** N0s, int * N1s, int N_trials) {

	int N_bpf;
	double T;
	int N_MESHES = hmm->N_MESHES, N_ALLOCS = hmm->N_ALLOCS;
	FILE * N0s_f = fopen("N0s_data.txt", "r");
	fscanf(N0s_f, "%d %le\n", &N_bpf, &T);
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++)
			fscanf(N0s_f, "%d ", &N0s[i_mesh][n_alloc]);
	}

	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		printf("Level 0 index = %d allocation sizes:\n", i_mesh);
		printf("************************************\n");
		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++)
			printf("(N0, N1) = (%d, %d)\n", N0s[i_mesh][n_alloc], N1s[n_alloc]);
		printf("\n");
	}

	fclose(N0s_f);
	return T;

}


double ks_statistic(int N_ref, w_double * weighted_ref, int N, w_double * weighted) {

	double record, diff;
	double cum1 = 0, cum2 = 0;
	int j = 0, lim1, lim2;
	w_double * a1;
	w_double * a2;

	if (weighted_ref[0].x < weighted[0].x) {
		a1 = weighted_ref;
		a2 = weighted;
		lim1 = N_ref;
		lim2 = N;
	}
	else {
		a1 = weighted;
		a2 = weighted_ref;
		lim1 = N;
		lim2 = N_ref;
	}

	cum1 = a1[0].w;
	record = cum1;
	for (int i = 1; i < lim1; i++) {
		while (a2[j].x < a1[i].x && j < lim2) {
			cum2 += a2[j].w;
			diff = fabs(cum2 - cum1);
			record = diff > record ? diff : record;
			j++;
		}
		cum1 += a1[i].w;
		diff = fabs(cum2 - cum1);
		record = diff > record ? diff : record;
	}
	return record;
}


double compute_mse(w_double ** weighted1, w_double ** weighted2, int length, int N1, int N2) {

	double mse = 0.0, x1_hat, x2_hat, w1_sum, w2_sum;

	for (int n = 0; n < length; n++) {
		x1_hat = 0.0, x2_hat = 0.0, w1_sum = 0.0, w2_sum = 0.0;
		for (int i = 0; i < N1; i++) {
			x1_hat += weighted1[n][i].w * weighted1[n][i].x;
			w1_sum += weighted1[n][i].w;
		}
		for (int i = 0; i < N2; i++) {
			x2_hat += weighted2[n][i].w * weighted2[n][i].x;
			w2_sum += weighted2[n][i].w;
		}
		mse = mse + (x1_hat - x2_hat) * (x1_hat - x2_hat);
	}
	return mse / (double) length;
}


void compute_nth_percentile(w_double ** distr, int N, double centile, int length, double * centiles) {

	/* We first need to ascending sort the distribution by particle value */
	// for (int n = 0; n < length; n++)
		// quicksort(distr[n], 0, N - 1);

	/* Now we can find the desired percentile */
	int i;
	double x_centile, cum_prob;
	for (int n = 0; n < length; n++) {
		i = 0;
		cum_prob = 0.0;
		while (cum_prob < centile) {
			x_centile = distr[n][i].x;
			cum_prob += distr[n][i].w;
			i++;
		}
		centiles[n] = x_centile;
	}
}


		// double true_soln = h[obs_pos];
		// gen_Z_topography(xs, Z, nx, k, sig_theta);
		// for (int j = 0; j < nx; j++) {
		// 	fprintf(CURVE_DATA, "%e ", h[j]);
		// 	fprintf(TOP_DATA, "%e ", Z[j]);
		// }
		// fprintf(CURVE_DATA, "\n");
		// fprintf(TOP_DATA, "\n");

		// EY = 0.0, varY = 0.0;
		// for (int i = 0; i < N; i++) {
		// 	thetas[i] = 0.9999 * theta + gsl_ran_gaussian(rng0, sig_sd);
		// 	sig_theta = sigmoid(thetas[i], upper_bound, lower_bound);
		// 	gamma_theta = gamma_of_k * pow(sig_theta, k);
		// 	solve(k, sig_theta, nx, xs, Z_x, h, dx, q0_sq, gamma_theta);
		// 	solns[i] = h[obs_pos];
		// 	EY += solns[i] / (double) N;
		// }
		// for (int i = 0; i < N; i++)
		// 	varY += (solns[i] - EY) * (solns[i] - EY);
		// varY = sqrt(varY / (double) (N - 1));
		// top_varY += varY;
		// printf("(signal, true obs, sample_mean, sample_obs_sd) = (%lf, %lf, %lf, %lf)\n", theta, true_soln, EY, varY);


// double partition(w_double * distr, int lo, int hi) {

// 	/** 
// 	 * Divides array into two partitions
// 	 * */
// 	double pivot = distr[hi].x;	// Choose the last element as the pivot
// 	double x_temp, w_temp;

// 	// Temporary pivot index
// 	int i = lo - 1;

// 	for (int j = lo; j < hi; j++) {
// 		// If the current element is less than or equal to the pivot
// 		if (distr[j].x <= pivot) {
// 			// Move the temporary pivot index forward
// 			i++;

// 			// Swap the current element with the element at the temporary pivot index
// 			x_temp = distr[i].x, w_temp = distr[i].w;
// 			distr[i].x = distr[j].x, distr[i].w = distr[j].w;
// 			distr[j].x = x_temp, distr[j].w = w_temp;
// 		}
// 	}
// 	// Move the pivot element to the correct pivot position (between the smaller and larger elements)
// 	i++;
// 	x_temp = distr[i].x, w_temp = distr[i].w;
// 	distr[i].x = distr[hi].x, distr[i].w = distr[hi].w;
// 	distr[hi].x = x_temp, distr[hi].w = w_temp;

// 	return i; // the pivot index
// }


// int quicksort(w_double * distr, int lo, int hi) {

// 	/**
// 	 * Sorts a (portion of an) array, divides it into portions, then sorts those
// 	*/

// 	// Ensure indices are in correct order
// 	if (lo >= hi || lo < 0)
// 		return 0;

// 	else {

// 		// Partition the array and get the pivot index
// 		double p = partition(distr, lo, hi);

// 		// Sort the two partitions
// 		quicksort(distr, lo, p - 1); // Left side of pivot
// 		quicksort(distr, p + 1, hi); // Right side of pivot

// 		return 1;
// 	}

// }






