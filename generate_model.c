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

void equal_runtimes_model(gsl_rng * rng, HMM * hmm, int ** N0s, int * N1s, w_double ** weighted_ref, int N_ref, int N_trials, int N_bpf, int * level0_meshes, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE, w_double ** ml_weighted)  {

	// int run_ref = 1;		// REF ON
	int run_ref = 0;		// REF OFF

	/* Reference distribution */
	/* ---------------------- */
	/* Produce the benchmark reference distribution with the BPF. Set run_ref to 0 if the reference data already esists */
	if (run_ref == 1)
		run_reference_filter(rng, hmm, N_ref, weighted_ref, n_data);
	else
		read_cdf(weighted_ref, hmm, n_data);


	/* Sample allocation */
	/* ----------------- */
	/* Run the BPF with a set number of particles N_bpf < N_ref and record the accuracy and the mean time taken. Then for each mesh configuration, increment the level 1 particle allocation and compute the level 0 particle allocation so that the time taken for the MLBPF is roughly the same as the BPF */
	double T, T_temp;
	T = perform_BPF_trials(hmm, N_bpf, rng, N_trials, N_ref, weighted_ref, n_data, RAW_BPF_TIMES, RAW_BPF_KS, RAW_BPF_MSE);
	// if (n_data == 0)
		// compute_sample_sizes(hmm, rng, level0_meshes, 2 * T / (double) hmm->length, N0s, N1s, N_bpf, N_trials, ml_weighted);
	T_temp = read_sample_sizes(hmm, N0s, N1s, N_trials);

}


void generate_hmm(gsl_rng * rng, HMM * hmm, int n_data, int length, int nx) {

	/** 
	Generates the HMM data and outputs to file to be read in by read_hmm.
	*/
	int obs_pos = nx - 1;
	// double sig_sd = 5.5;
	double sig_sd = 2.5;
	// double obs_sd = 0.1 * 0.037881;
	double obs_sd = 0.25 * 0.037881;
	// double obs_sd = 0.5 * 0.037881;
	// double obs_sd = 1.0 * 0.037881;
	// double obs_sd = 1.5 * 0.037881;	
	double space_left = 0.0, space_right = 50.0;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double k = 3.5, theta, obs;
	double h_init = 2.0, q0 = 3.0, q0_sq = q0 * q0;	
	double lower_bound = 1.5, upper_bound = 6.0 + lower_bound;
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
	gsl_rng * rng0 = gsl_rng_alloc(gsl_rng_taus);

	/* Generate the data */
	for (int n = 0; n < length; n++) {

		sig_theta = sigmoid(theta, upper_bound, lower_bound);
		gamma_theta = gamma_of_k * pow(sig_theta, k);
		solve(k, sig_theta, nx, xs, Z_x, h, dx, q0_sq, gamma_theta);
		for (int i = 0; i < 10; i++)
			printf("%lf %lf\n", h[obs_pos], gsl_ran_gaussian(rng, obs_sd));
		obs = h[obs_pos] + gsl_ran_gaussian(rng, obs_sd);
		fprintf(DATA_OUT, "%e %e\n", sig_theta, obs);

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

		/* Evolve the signal with the mutation model */
		theta = 0.9999 * theta + gsl_ran_gaussian(rng, sig_sd);

	}
	printf("Average observation stds = %lf\n", top_varY / (double) length);

	fclose(DATA_OUT);
	fclose(CURVE_DATA);	
	fclose(TOP_DATA);
	fflush(stdout);
	gsl_rng_free(rng0);

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
			fprintf(data, "%e ", w_particles[n][i].x);
		fprintf(data, "\n");
		for (int i = 0; i < N; i++)
			fprintf(data, "%e ", w_particles[n][i].w);
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

	for (int n = 0; n < length; n++) {
		for (int i = 0; i < N; i++)
			fscanf(data, "%lf ", &w_particles[n][i].x);
		for (int i = 0; i < N; i++)
			fscanf(data, "%lf ", &w_particles[n][i].w);
	}
	fclose(data);
}


double perform_BPF_trials(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, int n_data, FILE * RAW_BPF_TIMES, FILE * RAW_BPF_KS, FILE * RAW_BPF_MSE) {

	int length = hmm->length;
	double ks = 0.0, elapsed = 0.0, mse = 0.0, mean_elapsed = 0.0;
	w_double ** weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		weighted[n] = (w_double *) malloc(N_bpf * sizeof(w_double));

	printf("Running BPF trials...\n");
	for (int n_trial = 0; n_trial < N_trials; n_trial++) {

		/* Run the simulation for the BPF */
		clock_t bpf_timer = clock();
		bootstrap_particle_filter(hmm, N_bpf, rng, weighted);
		elapsed = (double) (clock() - bpf_timer) / (double) CLOCKS_PER_SEC;
		mean_elapsed += elapsed;

		/* Compute the KS statistic for the run */
		ks = 0.0;
		for (int n = 0; n < length; n++) {
			qsort(weighted[n], N_bpf, sizeof(w_double), weighted_double_cmp);
			ks += ks_statistic(N_ref, weighted_ref[n], N_bpf, weighted[n]) / (double) length;
		}

		mse = compute_mse(weighted_ref, weighted, length, N_ref, N_bpf);
		fprintf(RAW_BPF_TIMES, "%e ", elapsed);
		fprintf(RAW_BPF_KS, "%e ", ks);
		fprintf(RAW_BPF_MSE, "%e ", mse);

	}

	fprintf(RAW_BPF_TIMES, "\n");
	fprintf(RAW_BPF_KS, "\n");
	fprintf(RAW_BPF_MSE, "\n");
	
	free(weighted);

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
	fprintf(N0s_f, "%d %e\n", N_bpf, T);


	/* Compute the particle allocations */
	/* -------------------------------- */
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {

		nxs[0] = level0_meshes[i_mesh];
		printf("Computing the level 0 allocations for nx0 = %d\n", nxs[0]);

		for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {

			sample_sizes[1] = N1s[n_alloc];
			printf("N1 = %d\n", N1s[n_alloc]);

			N0 = N_bpf;
			sample_sizes[0] = N0;
			N0_lo = N0;

			/* Find a value for N0_init that exceeds the required time */
			clock_t timer = clock();
			ml_bootstrap_particle_filter_timed(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff < 0) {
				N0 *= 2;
				sample_sizes[0] = N0;

				timer = clock();
				ml_bootstrap_particle_filter_timed(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;
			}

			/* Find a value for N0_lo that does not meet the required time */
			sample_sizes[0] = N0_lo;
			timer = clock();
			ml_bootstrap_particle_filter_timed(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff > 0) {
				N0_lo = (int) (N0_lo / 2.0);
				sample_sizes[0] = N0_lo;

				timer = clock();
				ml_bootstrap_particle_filter_timed(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;

				if (N0_lo == 0)
					diff = 0;
			}

			/* Run with the N0 we know exceeds the required time */
			sample_sizes[0] = N0;
			timer = clock();
			ml_bootstrap_particle_filter_timed(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;

			if (N0_lo == 0)
				sample_sizes[0] = 0;

			else {

				/* Halve the interval until a sufficiently accurate root is found */
				while (fabs(diff) >= 0.0001) {
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
						ml_bootstrap_particle_filter_timed(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios);
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


double read_sample_sizes(HMM * hmm, int ** N0s, int * N1s, int N_trials) {

	int N_bpf;
	double T;
	int N_MESHES = hmm->N_MESHES, N_ALLOCS = hmm->N_ALLOCS;
	FILE * N0s_f = fopen("N0s_data.txt", "r");
	fscanf(N0s_f, "%d %lf\n", &N_bpf, &T);
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





		// EY = 0.0, varY = 0.0;
		// for (int i = 0; i < N; i++) {
		// 	thetas[i] = 0.9999 * theta + gsl_ran_gaussian(rng, sig_sd);
		// 	solve(k, sigmoid(thetas[i], upper_bound, lower_bound), nx, xs, Z_x, h, dx, q0_sq, gamma_of_k);
		// 	solns[i] = h[obs_pos];
		// 	EY += solns[i] / (double) N;
		// }
		// for (int i = 0; i < N; i++)
		// 	varY += (solns[i] - EY) * (solns[i] - EY);
		// varY = sqrt(varY / (double) (N - 1));
		// top_varY += varY;
		// printf("(signal, true obs, sample_mean, sample_obs_sd) = (%lf, %lf, %lf, %lf)\n", theta, true_soln, EY, varY);


		// for (int j = 0; j < nx; j++) {
			// fprintf(CURVE_DATA, "%e ", h[j]);
			// fprintf(TOP_DATA, "%e ", Z[j]);
		// }
		// fprintf(CURVE_DATA, "\n");
		// fprintf(TOP_DATA, "\n");

