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
#include <gsl/gsl_interp.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <assert.h>
#include "solvers.h"
#include "particle_filters.h"

// This is the 1d steady state shallow water equations model

int weighted_double_cmp(const void * a, const void * b) {

	struct weighted_double d1 = * (struct weighted_double *) a;
	struct weighted_double d2 = * (struct weighted_double *) b;

	if (d1.x < d2.x)
		return -1;
	if (d2.x < d1.x)
		return 1;
	return 0;
}


void regression_fit(double * s, double * corrections, int N0, int N1, int M_poly, double * poly_weights, double * PHI, double * C, double * C_inv, double * MP, gsl_matrix * C_gsl, gsl_permutation * p, gsl_matrix * C_inv_gsl) {

	/* Set the values of the design matrix */
	int counter = 0, sg;
	for (int n = 0; n < N1; n++) {
		for (int m = 0; m < M_poly; m++)
			PHI[n * M_poly + m] = pow(s[N0 + counter], m);
		counter++;
	}

	/* Do C = PHI.T * PHI */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < M_poly; k++) {
			C[j * M_poly + k] = 0.0;
			for (int n = 0; n < N1; n++)
				C[j * M_poly + k] += PHI[n * M_poly + j] * PHI[n * M_poly + k];
		}
	}

	/* Invert C */
	for (int m = 0; m < M_poly * M_poly; m++)
		C_gsl->data[m] = C[m];
	gsl_linalg_LU_decomp(C_gsl, p, &sg);
	gsl_linalg_LU_invert(C_gsl, p, C_inv_gsl);
	counter = 0;
	for (int m = 0; m < M_poly; m++) {
		for (int n = 0; n < M_poly; n++) {
			C_inv[counter] = gsl_matrix_get(C_inv_gsl, m, n);
			counter++;
		}
	}

	/* Do C_inv * PHI.T */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < N1; k++) {
			MP[j * N1 + k] = 0.0;
			for (int n = 0; n < M_poly; n++)
				MP[j * N1 + k] += C_inv[j * M_poly + n] * PHI[k * M_poly + n];
		}
	}

	/* Compute the polynomial weights */
	for (int j = 0; j < M_poly; j++) {
		poly_weights[j] = 0.0;
		for (int n = 0; n < N1; n++)
			poly_weights[j] += MP[j * N1 + n] * corrections[n];
	}

}


void gauss_regression_fit(double * s, double * corrections, int N0, int N1, int M_poly, double * poly_weights, double * PHI, double * C, double * C_inv, double * MP, gsl_matrix * C_gsl, gsl_permutation * p, gsl_matrix * C_inv_gsl, double * mu, double sig_sd) {

	/* Set the values of the design matrix */
	int counter = 0, sg;
	double denom = 2.0 * sig_sd * sig_sd;
	for (int n = 0; n < N1; n++) {
		for (int m = 0; m < M_poly; m++)
			PHI[n * M_poly + m] = exp(-(s[N0 + counter] - mu[m]) * (s[N0 + counter] - mu[m]) / denom);
		counter++;
	}

	/* Do C = PHI.T * PHI */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < M_poly; k++) {
			C[j * M_poly + k] = 0.0;
			for (int n = 0; n < N1; n++)
				C[j * M_poly + k] += PHI[n * M_poly + j] * PHI[n * M_poly + k];
		}
	}

	/* Invert C */
	for (int m = 0; m < M_poly * M_poly; m++)
		C_gsl->data[m] = C[m];
	gsl_linalg_LU_decomp(C_gsl, p, &sg);
	gsl_linalg_LU_invert(C_gsl, p, C_inv_gsl);
	counter = 0;
	for (int m = 0; m < M_poly; m++) {
		for (int n = 0; n < M_poly; n++) {
			C_inv[counter] = gsl_matrix_get(C_inv_gsl, m, n);
			counter++;
		}
	}

	/* Do C_inv * PHI.T */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < N1; k++) {
			MP[j * N1 + k] = 0.0;
			for (int n = 0; n < M_poly; n++)
				MP[j * N1 + k] += C_inv[j * M_poly + n] * PHI[k * M_poly + n];
		}
	}

	/* Compute the polynomial weights */
	for (int j = 0; j < M_poly; j++) {
		poly_weights[j] = 0.0;
		for (int n = 0; n < N1; n++)
			poly_weights[j] += MP[j * N1 + n] * corrections[n];
	}

}


double poly_eval(double x, double * poly_weights, int poly_degree) {
	double y_hat = 0.0;
	for (int m = 0; m < poly_degree + 1; m++)
		y_hat += poly_weights[m] * pow(x, m);
	return y_hat;
}


double gauss_eval(double x, double * poly_weights, int poly_degree, double * mu, double sig_sd) {
	double y_hat = 0.0, denom = 2.0 * sig_sd * sig_sd;
	for (int m = 0; m < poly_degree + 1; m++)
		y_hat += poly_weights[m] * exp(-(x - mu[m]) * (x - mu[m]) / denom);
	return y_hat;
}


void resample(long size, double * w, long * ind, gsl_rng * r) {

	/* Generate the exponentials */
	double * e = (double *) malloc((size + 1) * sizeof(double));
	double g = 0;
	for (long i = 0; i <= size; i++) {
		e[i] = gsl_ran_exponential(r, 1.0);
		g += e[i];
	}
	/* Generate the uniform order statistics */
	double * u = (double *) malloc((size + 1) * sizeof(double));
	u[0] = 0;
	for (long i = 1; i <= size; i++)
		u[i] = u[i - 1] + e[i - 1] / g;

	/* Do the actual sampling with C_inv_gsl cdf */
	double cdf = w[0];
	long j = 0;
	for (long i = 0; i < size; i++) {
		while (cdf < u[i + 1]) {
			j++;
			cdf += w[j];
		}
		ind[i] = j;
	}

	free(e);
	free(u);
}


void random_permuter(int *permutation, int N, gsl_rng *r) {
  
  for (int i = 0; i < N; i++)
    permutation[i] = i;
  
  int j;
  int tmp;
  for (int i = N - 1; i > 0; i--) {
    j = (int)gsl_rng_uniform_int(r, i + 1);
    tmp = permutation[j];
    permutation[j] = permutation[i];
    permutation[i] = tmp;
  }
  
}


double sigmoid(double x, double a, double b) {
	return a / (1.0 + exp(-0.01 * M_PI * x)) + b;
}


double sigmoid_inv(double x, double a, double b) {
	return log((x - b) / (a + b - x)) / (0.01 * M_PI);
}


void mutate(gsl_rng * rng, int N_tot, double * thetas, double * res_thetas, double sig_sd) {
	for (int i = 0; i < N_tot; i++)
		thetas[i] = 0.9999 * res_thetas[i] + gsl_ran_gaussian(rng, sig_sd);
}


void generate_adaptive_artificial_mesh(int N_tot, double * sig_thetas, int mesh_size, double * sig_theta_mesh) {

	double theta_lo = 10.0, theta_hi = 0.0;
	double mesh_incr;
	for (int i = 0; i < N_tot; i++) {
		theta_lo = theta_lo < sig_thetas[i] ? theta_lo : sig_thetas[i];
		theta_hi = theta_hi > sig_thetas[i] ? theta_hi : sig_thetas[i];
	}
	mesh_incr = (theta_hi - theta_lo) / (double) (mesh_size - 1);
	for (int l = 0; l < mesh_size; l++)
		sig_theta_mesh[l] = theta_lo + l * mesh_incr;
}


void RSR(long N, long M, double * w, long * ind, gsl_rng * r) {

	int counter = 0, n_reps;
	double dU, M_inv = 1.0 / (double) M;

	/* Generate an initial random number dU from Unif(0, 1 / M) */
	dU = gsl_ran_flat(r, 0, M_inv);

	for (int m = 0; m < N; m++) {

		/* Compute the number of replicates for particle m */
		n_reps = floor((w[m] - dU) * M) + 1;

		/* Distribute the computed number of index replicates */
		for (int i = 0; i < n_reps; i++)
			ind[counter + i] = m;
		counter += n_reps;

		/* Compute the next dU */
		dU += n_reps / (double) M - w[m];

	}

}


void ml_bootstrap_particle_filter_res(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx1 = hmm->nx;
	int nx0 = nxs[0];
	int obs_pos1 = nx1 - 1;
	int obs_pos0 = nx0 - 1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	int N_tot_long = 2 * N0;
	int poly_degree = 0, M_poly = poly_degree + 1, mesh_size = 1000;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right, k = hmm->k;
	double dx1 = (space_right - space_left) / (double) (nx1 - 1);
	double dx0 = (space_right - space_left) / (double) (nx0 - 1);
	double low_bd = hmm->low_bd, upp_bd = hmm->upp_bd;
	double obs, normaliser, abs_normaliser, x_hat, g0, g1, sign_rat;
	double level_1_abs_normaliser;
	double Z_obs, gamma_of_k = tgamma(k), gamma_theta;
	short * signs = (short *) malloc(N_tot * sizeof(short));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	long * level_1_res_ind = (long *) malloc(N0 * sizeof(long));
	double * level_1_abs_weights = (double *) malloc(N1 * sizeof(double));
	double * ext_sig_thetas = (double *) malloc(N0 * sizeof(double));
	double * ext_signs = (double *) malloc(N0 * sizeof(double));
	double * ext_weights = (double *) malloc(N0 * sizeof(double));
	int * permutation = (int *) malloc(N_tot * sizeof(int));
	double * thetas = (double *) calloc(N_tot, sizeof(double));
	double * sig_thetas = (double *) calloc(N_tot, sizeof(double));
	double * res_thetas = (double *) malloc(N_tot * sizeof(double));
	double * weights = (double *) malloc(N_tot * sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * solns1 = (double *) malloc(N1 * sizeof(double));
	double * solns0 = (double *) malloc(N_tot * sizeof(double));
	double * g0s = (double *) malloc(N1 * sizeof(double));
	double * g1s = (double *) malloc(N1 * sizeof(double));
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * poly_weights = (double *) malloc(M_poly * sizeof(double));
	double * sig_theta_mesh = (double *) malloc(mesh_size * sizeof(double));


	/* Solver arrays */
	/* ------------- */
	double * h1 = (double *) malloc(nx1 * sizeof(double));
	double * Z_x1 = (double *) malloc(nx1 * sizeof(double));
	double * xs1 = (double *) malloc(nx1 * sizeof(double));
	double * h0 = (double *) malloc(nx0 * sizeof(double));
	double * Z_x0 = (double *) malloc(nx0 * sizeof(double));
	double * xs0 = (double *) malloc(nx0 * sizeof(double));
	for (int j = 0; j < nx1; j++)
		xs1[j] = space_left + j * dx1;
	for (int j = 0; j < nx0; j++)
		xs0[j] = space_left + j * dx0;


	/* Regression matrices */
	/* ------------------- */
	double * PHI = (double *) malloc(N1 * M_poly * sizeof(double));
	double * C = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * C_inv = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * MP = (double *) malloc(N1 * M_poly * sizeof(double));
	gsl_matrix * C_gsl = gsl_matrix_alloc(M_poly, M_poly);
	gsl_permutation * p = gsl_permutation_alloc(M_poly);
	gsl_matrix * C_inv_gsl = gsl_matrix_alloc(M_poly, M_poly);


	/* Initial conditions */
	/* ------------------ */
	double theta_init = sigmoid_inv(hmm->signal[0], upp_bd - low_bd, low_bd);
	double h_init = hmm->h_init, q_init = hmm->q0, q_init_sq = q_init * q_init;
	h1[0] = h_init, h0[0] = h_init;
	for (int i = 0; i < N_tot; i++) {
		thetas[i] = theta_init + gsl_ran_gaussian(rng, sig_sd);
		res_signs[i] = 1;
	}	


	/* Files */
	/* ----- */
	FILE * X_HATS = fopen("ml_xhats.txt", "w");
	FILE * CORRECTIONS = fopen("corrections.txt", "w");
	FILE * REGRESSION_CURVE = fopen("regression_curve.txt", "w");
	FILE * TRUE_CURVE = fopen("true_curve.txt", "w");
	FILE * TRUE_CURVE0 = fopen("true_curve0.txt", "w");
	FILE * LEVEL1_FINE = fopen("level1_fine.txt", "w");
	FILE * LEVEL1_COARSE = fopen("level1_coarse.txt", "w");
	FILE * LEVEL0_COARSE = fopen("level0_coarse.txt", "w");
	FILE * ML_DISTR = fopen("ml_distr.txt", "w");
	FILE * SIGNS = fopen("signs.txt", "w");
	FILE * TV_MEASURE = fopen("tv_measure.txt", "w");
	FILE * RES_DISTR = fopen("res_distr.txt", "w");
	fprintf(REGRESSION_CURVE, "%d\n", mesh_size);
	fprintf(ML_DISTR, "%d %d %d\n", N0, N1, N_tot);
	fprintf(SIGNS, "%d %d %d\n", N0, N1, N_tot);
	fprintf(TV_MEASURE, "%d %d %d %d\n", N0, N1, N_tot, length);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0, abs_normaliser = 0.0, x_hat = 0.0;
		level_1_abs_normaliser = 0.0;
		for (int i = 0; i < N_tot; i++)
			sig_thetas[i] = sigmoid(thetas[i], upp_bd - low_bd, low_bd);

		

		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 1 solutions																						 																					   */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);

			/* Fine solution */
			solve(k, sig_thetas[i], nx1, xs1, Z_x1, h1, dx1, q_init_sq, gamma_theta);
			solns1[i - N0] = h1[obs_pos1];

			/* Coarse solution */
			solve(k, sig_thetas[i], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			solns0[i] = h0[obs_pos0];

			/* Record the corrections samples for the regression approximation to the true correction curve */
			corrections[i - N0] = solns1[i - N0] - solns0[i];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 0 solutions																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N0; i++) {

			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);
			solve(k, sig_thetas[i], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			solns0[i] = h0[obs_pos0];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Regresion corrections																			 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		if (N1 > 0) {
			regression_fit(sig_thetas, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);
			for (int i = 0; i < N_tot; i++)
				solns0[i] += poly_eval(sig_thetas[i], poly_weights, poly_degree);
		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight assignment																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */

		/* Level 1 */
		/* ------- */
		for (int i = N0; i < N_tot; i++) {

			g1s[i - N0] = gsl_ran_gaussian_pdf(solns1[i - N0] - obs, obs_sd);
			g0s[i - N0] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd);

		}
		for (int i = N0; i < N_tot; i++) {
			weights[i] = (g1s[i - N0] - g0s[i - N0]) * res_signs[i] / (double) N1;
			absolute_weights[i] = fabs(weights[i]);

			abs_normaliser += absolute_weights[i];
			signs[i] = weights[i] < 0 ? -1 : 1;
		}
		level_1_abs_normaliser = abs_normaliser;


		/* Level 0 */
		/* ------- */
		for (int i = 0; i < N0; i++) {
			weights[i] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd) * res_signs[i] / (double) N0;
			absolute_weights[i] = fabs(weights[i]);

			abs_normaliser += absolute_weights[i];
			signs[i] = weights[i] < 0 ? -1 : 1;
		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 1 local resampling																		 																						 	 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */

		/* Normalise the level 1 total variation measure */
		for (int i = N0; i < N_tot; i++)
			level_1_abs_weights[i - N0] = absolute_weights[i] / level_1_abs_normaliser;

		/* Resample N0 times from [1, N1] wrt the level 1 total variation measure */
		if (N1 > 0) {
			RSR(N1, N0, level_1_abs_weights, level_1_res_ind, rng);
			for (int i = 0; i < N0; i++) {
				ext_sig_thetas[i] = sig_thetas[N0 + level_1_res_ind[i]];
				ext_signs[i] = signs[N0 + level_1_res_ind[i]];
				ext_weights[i] = ext_signs[i] / (double) N0;
			}
		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weighted filter generation																	 																						 	 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */

		/* Level 0 */
		/* ------- */
		for (int i = 0; i < N0; i++)
			normaliser += weights[i];

		/* Extended level 1 */
		/* ---------------- */
		if (N1 > 0) {
			for (int i = 0; i < N0; i++)
				normaliser += ext_weights[i];
		}


		/* Level 0 */
		/* ------- */
		for (int i = 0; i < N0; i++) {
			ml_weighted[n][i].x = sig_thetas[i];
			ml_weighted[n][i].w = weights[i] / normaliser;
			x_hat += ml_weighted[n][i].x * ml_weighted[n][i].w;
		}

		/* Extended level 1 */
		/* ---------------- */
		if (N1 > 0) {
			for (int i = N0; i < N_tot_long; i++) {
				ml_weighted[n][i].x = ext_sig_thetas[i - N0];
				ml_weighted[n][i].w = ext_weights[i - N0] / normaliser;
				x_hat += ml_weighted[n][i].x * ml_weighted[n][i].w;
			}
		}
		// printf("%lf\n", x_hat);
		fprintf(X_HATS, "%e ", x_hat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		/* Normalise now the original joint total variation measure */
		for (int i = 0; i < N_tot; i++)
			absolute_weights[i] /= abs_normaliser;

		/* Resample N_tot times from [1, N_tot] wrt this measure */
		resample(N_tot, absolute_weights, ind, rng);
		random_permuter(permutation, N_tot, rng);
		for (int i = 0; i < N_tot; i++) {
			res_thetas[permutation[i]] = thetas[ind[i]];
			res_signs[permutation[i]] = signs[ind[i]];
		}
		mutate(rng, N_tot, thetas, res_thetas, sig_sd);

	}

	fclose(X_HATS);
	fclose(CORRECTIONS);
	fclose(REGRESSION_CURVE);
	fclose(TRUE_CURVE);
	fclose(TRUE_CURVE0);
	fclose(LEVEL1_FINE);
	fclose(LEVEL1_COARSE);
	fclose(LEVEL0_COARSE);
	fclose(ML_DISTR);
	fclose(SIGNS);
	fclose(TV_MEASURE);
	fclose(RES_DISTR);

	free(signs);
	free(res_signs);
	free(ind);
	free(level_1_res_ind);
	free(level_1_abs_weights);
	free(ext_sig_thetas);
	free(ext_signs);
	free(ext_weights);
	free(permutation);
	free(thetas);
	free(sig_thetas);
	free(res_thetas);
	free(weights);
	free(absolute_weights);
	free(solns1);
	free(solns0);
	free(g0s);
	free(g1s);
	free(corrections);
	free(poly_weights);
	free(h1);
	free(Z_x1);
	free(xs1);
	free(h0);
	free(Z_x0);
	free(xs0);
	free(PHI);
	free(C);
	free(C_inv);
	free(MP);
	free(sig_theta_mesh);

	gsl_matrix_free(C_gsl);
	gsl_permutation_free(p);
	gsl_matrix_free(C_inv_gsl);

}


void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx1 = hmm->nx;
	int nx0 = nxs[0];
	int obs_pos1 = nx1 - 1;
	int obs_pos0 = nx0 - 1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	int poly_degree = 1, M_poly = poly_degree + 1, mesh_size = 1000;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right, k = hmm->k;
	double dx1 = (space_right - space_left) / (double) (nx1 - 1);
	double dx0 = (space_right - space_left) / (double) (nx0 - 1);
	double low_bd = hmm->low_bd, upp_bd = hmm->upp_bd;
	double obs, normaliser, abs_normaliser, x_hat, g0, g1, sign_rat;
	double Z_obs, gamma_of_k = tgamma(k), gamma_theta;
	short * signs = (short *) malloc(N_tot * sizeof(short));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	int * permutation = (int *) malloc(N_tot * sizeof(int));
	double * thetas = (double *) calloc(N_tot, sizeof(double));
	double * sig_thetas = (double *) calloc(N_tot, sizeof(double));
	double * res_thetas = (double *) malloc(N_tot * sizeof(double));
	double * weights = (double *) calloc(N_tot, sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * solns1 = (double *) malloc(N1 * sizeof(double));
	double * solns0 = (double *) malloc(N_tot * sizeof(double));
	double * g0s = (double *) malloc(N1 * sizeof(double));
	double * g1s = (double *) malloc(N1 * sizeof(double));
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * poly_weights = (double *) malloc(M_poly * sizeof(double));
	double * sig_theta_mesh = (double *) malloc(mesh_size * sizeof(double));


	/* Solver arrays */
	/* ------------- */
	double * h1 = (double *) malloc(nx1 * sizeof(double));
	double * Z_x1 = (double *) malloc(nx1 * sizeof(double));
	double * xs1 = (double *) malloc(nx1 * sizeof(double));
	double * h0 = (double *) malloc(nx0 * sizeof(double));
	double * Z_x0 = (double *) malloc(nx0 * sizeof(double));
	double * xs0 = (double *) malloc(nx0 * sizeof(double));
	for (int j = 0; j < nx1; j++)
		xs1[j] = space_left + j * dx1;
	for (int j = 0; j < nx0; j++)
		xs0[j] = space_left + j * dx0;


	/* Regression matrices */
	/* ------------------- */
	double * PHI = (double *) malloc(N1 * M_poly * sizeof(double));
	double * C = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * C_inv = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * MP = (double *) malloc(N1 * M_poly * sizeof(double));
	gsl_matrix * C_gsl = gsl_matrix_alloc(M_poly, M_poly);
	gsl_permutation * p = gsl_permutation_alloc(M_poly);
	gsl_matrix * C_inv_gsl = gsl_matrix_alloc(M_poly, M_poly);


	/* Initial conditions */
	/* ------------------ */
	double theta_init = sigmoid_inv(hmm->signal[0], upp_bd - low_bd, low_bd);
	double h_init = hmm->h_init, q_init = hmm->q0, q_init_sq = q_init * q_init;
	h1[0] = h_init, h0[0] = h_init;
	for (int i = 0; i < N_tot; i++) {
		thetas[i] = theta_init + gsl_ran_gaussian(rng, sig_sd);
		res_signs[i] = 1;
	}	


	/* Files */
	/* ----- */
	FILE * X_HATS = fopen("ml_xhats.txt", "w");
	FILE * CORRECTIONS = fopen("corrections.txt", "w");
	FILE * REGRESSION_CURVE = fopen("regression_curve.txt", "w");
	FILE * TRUE_CURVE = fopen("true_curve.txt", "w");
	FILE * TRUE_CURVE0 = fopen("true_curve0.txt", "w");
	FILE * LEVEL1_FINE = fopen("level1_fine.txt", "w");
	FILE * LEVEL1_COARSE = fopen("level1_coarse.txt", "w");
	FILE * LEVEL0_COARSE = fopen("level0_coarse.txt", "w");
	FILE * ML_DISTR = fopen("ml_distr.txt", "w");
	FILE * SIGNS = fopen("signs.txt", "w");
	fprintf(REGRESSION_CURVE, "%d\n", mesh_size);
	fprintf(ML_DISTR, "%d %d %d\n", N0, N1, N_tot);
	fprintf(SIGNS, "%d %d %d\n", N0, N1, N_tot);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0, abs_normaliser = 0.0;
		for (int i = 0; i < N_tot; i++)
			sig_thetas[i] = sigmoid(thetas[i], upp_bd - low_bd, low_bd);

		

		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 1 solutions																						 																					   */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);

			/* Fine solution */
			solve(k, sig_thetas[i], nx1, xs1, Z_x1, h1, dx1, q_init_sq, gamma_theta);
			solns1[i - N0] = h1[obs_pos1];

			/* Coarse solution */
			solve(k, sig_thetas[i], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			solns0[i] = h0[obs_pos0];

			/* Record the corrections samples for the regression approximation to the true correction curve */
			corrections[i - N0] = solns1[i - N0] - solns0[i];
			// fprintf(CORRECTIONS, "%e %e\n", sig_thetas[i], corrections[i - N0]);

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 0 solutions																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N0; i++) {

			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);
			solve(k, sig_thetas[i], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			solns0[i] = h0[obs_pos0];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Regresion corrections																			 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		if (N1 > 0) {
			;
			regression_fit(sig_thetas, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);
			for (int i = 0; i < N_tot; i++)
				solns0[i] += poly_eval(sig_thetas[i], poly_weights, poly_degree);

			// generate_adaptive_artificial_mesh(N_tot, sig_thetas, mesh_size, sig_theta_mesh);
			// for (int l = 0; l < mesh_size; l++) {

				/* Output the regressed correction curve approximation over the artificial particle mesh */
				// fprintf(REGRESSION_CURVE, "%e %e\n", sig_theta_mesh[l], poly_eval(sig_theta_mesh[l], poly_weights, poly_degree));

				/* Output the true correction curve */
			// 	gamma_theta = gamma_of_k * pow(sig_theta_mesh[l], k);
			// 	solve(k, sig_theta_mesh[l], nx1, xs1, Z_x1, h1, dx1, q_init_sq, gamma_theta);
			// 	// g1 = h1[obs_pos1];
			// 	g1 = sin(lmbda * h1[obs_pos1]) + h1[obs_pos1];
			// 	solve(k, sig_theta_mesh[l], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			// 	// g0 = h0[obs_pos0];
			// 	g0 = sin(lmbda * h0[obs_pos0]) + h0[obs_pos0];
			// 	g0 += poly_eval(sig_theta_mesh[l], poly_weights, poly_degree);
			// 	// fprintf(TRUE_CURVE, "%e ", g1 - g0);
			// 	fprintf(TRUE_CURVE, "%e ", g1);
			// 	fprintf(TRUE_CURVE0, "%e ", g0);
			// }
			// fprintf(TRUE_CURVE, "\n");
			// fprintf(TRUE_CURVE0, "\n");

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight assignment																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */

		/* Level 1 */
		/* ------- */
		// for (int i = N0; i < N_tot; i++) {

		// 	g1s[i - N0] = gsl_ran_gaussian_pdf(solns1[i - N0] - obs, obs_sd);
		// 	g0s[i - N0] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd);

		// 	// fprintf(LEVEL1_FINE, "%e %e\n", solns1[i - N0], g1s[i - N0]);
		// 	// fprintf(LEVEL1_COARSE, "%e %e\n", solns0[i], g0s[i - N0]);

		// 	// fprintf(LEVEL1_FINE, "%e %e\n", sig_thetas[i], g1s[i - N0]);
		// 	// fprintf(LEVEL1_COARSE, "%e %e\n", sig_thetas[i], g0s[i - N0]);

		// }
		// for (int i = N0; i < N_tot; i++)
		// 	weights[i] = (g1s[i - N0] - g0s[i - N0]) / (double) N1;


		/* Level 0 */
		/* ------- */
		for (int i = 0; i < N_tot; i++) {
			weights[i] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd) / (double) N0;

			// fprintf(LEVEL0_COARSE, "%e %e\n", solns0[i], weights[i] * N0);
			// fprintf(LEVEL0_COARSE, "%e %e\n", sig_thetas[i], weights[i] * N0);

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Normalisation 																							 																						 	 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N_tot; i++) {

			/* Scale the weights by the previous sign and compute the new sign */
			weights[i] *= res_signs[i];
			signs[i] = weights[i] < 0 ? -1 : 1;
			absolute_weights[i] = fabs(weights[i]);
			
			/* Compute the normalisation terms */
			normaliser += weights[i];
			abs_normaliser += absolute_weights[i];

		}		
		// for (int i = N0; i < N_tot; i++) {
		// 	fprintf(LEVEL1_FINE, "%e %e\n", sig_thetas[i], g1s[i - N0] / (double) N1 / (double) normaliser);
		// 	fprintf(LEVEL1_COARSE, "%e %e\n", sig_thetas[i], g0s[i - N0] / (double) N1 / (double) normaliser);
		// }
		// for (int i = 0; i < N0; i++)
		// 	fprintf(LEVEL0_COARSE, "%e %e\n", sig_thetas[i], weights[i] / (double) normaliser);

		x_hat = 0.0, sign_rat = 0.0;
		for (int i = 0; i < N_tot; i++) {
			weights[i] /= normaliser;
			absolute_weights[i] /= abs_normaliser;
			ml_weighted[n][i].x = sig_thetas[i];
			ml_weighted[n][i].w = weights[i];
			x_hat += sig_thetas[i] * weights[i];
			// sign_rat += absolute_weights[i] * signs[i];
			// fprintf(SIGNS, "%e %hd %e\n", sig_thetas[i], signs[i], absolute_weights[i]);
		}
		fprintf(X_HATS, "%e ", x_hat);
		// sign_ratios[n] = sign_rat;
		// fprintf(SIGNS, "\n");
		// for (int i = 0; i < N_tot; i++)
		// 	fprintf(ML_DISTR, "%e %e\n", ml_weighted[n][i].x, ml_weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N_tot, absolute_weights, ind, rng);
		random_permuter(permutation, N_tot, rng);
		for (int i = 0; i < N_tot; i++) {
			res_thetas[permutation[i]] = thetas[ind[i]];
			res_signs[permutation[i]] = signs[ind[i]];
		}
		mutate(rng, N_tot, thetas, res_thetas, sig_sd);

	}

	fclose(X_HATS);
	fclose(CORRECTIONS);
	fclose(REGRESSION_CURVE);
	fclose(TRUE_CURVE);
	fclose(TRUE_CURVE0);
	fclose(LEVEL1_FINE);
	fclose(LEVEL1_COARSE);
	fclose(LEVEL0_COARSE);
	fclose(ML_DISTR);
	fclose(SIGNS);

	free(signs);
	free(res_signs);
	free(ind);
	free(permutation);
	free(thetas);
	free(sig_thetas);
	free(res_thetas);
	free(weights);
	free(absolute_weights);
	free(solns1);
	free(solns0);
	free(g0s);
	free(g1s);
	free(corrections);
	free(poly_weights);
	free(h1);
	free(Z_x1);
	free(xs1);
	free(h0);
	free(Z_x0);
	free(xs0);
	free(PHI);
	free(C);
	free(C_inv);
	free(MP);
	free(sig_theta_mesh);
	gsl_matrix_free(C_gsl);
	gsl_permutation_free(p);
	gsl_matrix_free(C_inv_gsl);

}


void bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx = hmm->nx;
	int obs_pos = nx - 1;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd, obs_sd0 = 2.0 * obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right, k = hmm->k;
	double depth_scaler = 10.0;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double obs, normaliser, x_hat, Z_obs, gamma_of_k = tgamma(k), Z_obs_ind, gamma_theta;
	double low_bd = hmm->low_bd, upp_bd = hmm->upp_bd;
	long * ind = (long *) malloc(N * sizeof(long));
	double * thetas = (double *) malloc(N * sizeof(double));
	double * sig_thetas = (double *) malloc(N * sizeof(double));
	double * res_thetas = (double *) malloc(N * sizeof(double));
	double * weights = (double *) malloc(N * sizeof(double));
	double * h = (double *) malloc(nx * sizeof(double));
	double * Z_x = (double *) malloc(nx * sizeof(double));
	double * xs = (double *) malloc(nx * sizeof(double));
	for (int j = 0; j < nx; j++)
		xs[j] = space_left + j * dx;
	Z_obs_ind = -depth_scaler * pow(xs[obs_pos], k - 1) / gamma_of_k;


	/* Initial conditions */
	/* ------------------ */
	double theta_init = sigmoid_inv(hmm->signal[0], upp_bd - low_bd, low_bd);
	double h_init = hmm->h_init, q0 = hmm->q0, q0_sq = q0 * q0;
	h[0] = h_init;
	for (int i = 0; i < N; i++)
		thetas[i] = theta_init + gsl_ran_gaussian(rng, sig_sd);

	FILE * X_HATS = fopen("x_hats.txt", "w");
	FILE * BPF_DISTR = fopen("bpf_distr.txt", "w");
	fprintf(BPF_DISTR, "%d\n", N);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0;


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight generation																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N; i++) {

			sig_thetas[i] = sigmoid(thetas[i], upp_bd - low_bd, low_bd);
			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);
			solve(k, sig_thetas[i], nx, xs, Z_x, h, dx, q0_sq, gamma_theta);
			// weights[i] = gsl_ran_gaussian_pdf(h[obs_pos] + gsl_ran_gaussian(rng, obs_sd0) - obs, obs_sd);
			weights[i] = gsl_ran_gaussian_pdf(h[obs_pos] - obs, obs_sd);
			normaliser += weights[i];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Normalisation 																							 																						   */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = sig_thetas[i];
			weighted[n][i].w = weights[i];
			x_hat += sig_thetas[i] * weights[i];
		}
		fprintf(X_HATS, "%e ", x_hat);
		// for (int i = 0; i < N; i++)
			// fprintf(BPF_DISTR, "%e %e\n", weighted[n][i].x, weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			res_thetas[i] = thetas[ind[i]];
		mutate(rng, N, thetas, res_thetas, sig_sd);

	}

	fclose(X_HATS);
	fclose(BPF_DISTR);

	free(ind);
	free(thetas);
	free(sig_thetas);
	free(res_thetas);
	free(weights);
	free(h);
	free(Z_x);
	free(xs);

}


void ref_bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx = hmm->nx;
	int obs_pos = nx - 1;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd, obs_sd0 = 2.0 * obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right, k = hmm->k;
	double depth_scaler = 10.0;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double obs, normaliser, x_hat, Z_obs, gamma_of_k = tgamma(k), Z_obs_ind, gamma_theta;
	double low_bd = hmm->low_bd, upp_bd = hmm->upp_bd;
	long * ind = (long *) malloc(N * sizeof(long));
	double * thetas = (double *) malloc(N * sizeof(double));
	double * sig_thetas = (double *) malloc(N * sizeof(double));
	double * res_thetas = (double *) malloc(N * sizeof(double));
	double * weights = (double *) malloc(N * sizeof(double));
	double * h = (double *) malloc(nx * sizeof(double));
	double * Z_x = (double *) malloc(nx * sizeof(double));
	double * xs = (double *) malloc(nx * sizeof(double));
	for (int j = 0; j < nx; j++)
		xs[j] = space_left + j * dx;
	Z_obs_ind = -depth_scaler * pow(xs[obs_pos], k - 1) / gamma_of_k;


	/* Initial conditions */
	/* ------------------ */
	double theta_init = sigmoid_inv(hmm->signal[0], upp_bd - low_bd, low_bd);
	double h_init = hmm->h_init, q0 = hmm->q0, q0_sq = q0 * q0;
	h[0] = h_init;
	for (int i = 0; i < N; i++)
		thetas[i] = theta_init + gsl_ran_gaussian(rng, sig_sd);

	FILE * BPF_PARTICLES = fopen("bpf_particles.txt", "w");
	FILE * NORMALISERS = fopen("normalisers.txt", "w");
	FILE * X_HATS = fopen("x_hats.txt", "w");
	fprintf(BPF_PARTICLES, "%d %d\n", length, N);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		for (int i = 0; i < N; i++)
			sig_thetas[i] = sigmoid(thetas[i], upp_bd - low_bd, low_bd);
		


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight generation																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		normaliser = 0.0;
		for (int i = 0; i < N; i++) {
			
			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);
			solve(k, sig_thetas[i], nx, xs, Z_x, h, dx, q0_sq, gamma_theta);
			weights[i] = gsl_ran_gaussian_pdf(h[obs_pos] - obs, obs_sd);
			normaliser += weights[i];

		}
		fprintf(NORMALISERS, "%e ", normaliser);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Normalisation 																							 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = sig_thetas[i];
			weighted[n][i].w = weights[i];
			x_hat += sig_thetas[i] * weights[i];
		}
		fprintf(X_HATS, "%e ", x_hat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			res_thetas[i] = thetas[ind[i]];
		for (int i = 0; i < N; i++)
			fprintf(BPF_PARTICLES, "%e ", sigmoid(res_thetas[i], upp_bd - low_bd, low_bd));
		fprintf(BPF_PARTICLES, "\n");
		mutate(rng, N, thetas, res_thetas, sig_sd);

	}

	fclose(BPF_PARTICLES);
	fclose(X_HATS);
	fclose(NORMALISERS);

	free(ind);
	free(thetas);
	free(sig_thetas);
	free(res_thetas);
	free(weights);
	free(h);
	free(Z_x);
	free(xs);

}




			/* Output the particle and its corresponding correction value */
			// fprintf(CORRECTIONS, "%e %e\n", sig_thetas[i], corrections[i - N0]);


			// generate_adaptive_artificial_mesh(N_tot, sig_thetas, mesh_size, sig_theta_mesh);
			// for (int l = 0; l < mesh_size; l++) {

			// 	/* Output the regressed correction curve approximation over the artificial particle mesh */
			// 	fprintf(REGRESSION_CURVE, "%e %e\n", sig_theta_mesh[l], poly_eval(sig_theta_mesh[l], poly_weights, poly_degree));

			// 	/* Output the true correction curve */
			// 	solve(k, sig_theta_mesh[l], nx1, xs1, Z_x1, h1, dx1, q_init_sq, gamma_of_k);
			// 	g1 = h1[obs_pos1];
			// 	solve(k, sig_theta_mesh[l], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_of_k);
			// 	g0 = h0[obs_pos0];
			// 	fprintf(TRUE_CURVE, "%e ", g1 - g0);
			// }
			// fprintf(TRUE_CURVE, "\n");






	// FILE * LEVEL1_FINE_PARTICLES = fopen("level1_fine_particles.txt", "w");
	// FILE * LEVEL1_COARSE_PARTICLES = fopen("level1_coarse_particles.txt", "w");
	// FILE * LEVEL0_PARTICLES = fopen("level0_particles.txt", "w");
	// FILE * UNCORRECTED_PARTICLES = fopen("uncorrected_particles.txt", "w");

			// fprintf(LEVEL1_FINE_PARTICLES, "%e %e\n", sig_thetas[i], g1 * (double) res_signs[i] / (double) N1);
			// fprintf(LEVEL1_COARSE_PARTICLES, "%e %e\n", sig_thetas[i], g0 * (double) res_signs[i] / (double) N1);



			// fprintf(LEVEL0_PARTICLES, "%e %e\n", sig_thetas[i], g0 * (double) res_signs[i] / (double) N0);


		// for (int i = N0; i < N_tot; i++)
		// 	fprintf(LEVEL1_FINE_PARTICLES, "%e ", sigmoid(res_thetas[i], upp_bd, low_bd));
		// fprintf(LEVEL1_FINE_PARTICLES, "\n");
		// for (int i = 0; i < N0; i++)
		// 	fprintf(LEVEL0_PARTICLES, "%e ", sigmoid(res_thetas[i], upp_bd, low_bd));
		// fprintf(LEVEL0_PARTICLES, "\n");



	// fclose(TRUE_CURVE);
	// fclose(LEVEL1_FINE_PARTICLES);
	// fclose(LEVEL1_COARSE_PARTICLES);
	// fclose(LEVEL0_PARTICLES);
	// fclose(UNCORRECTED_PARTICLES);




		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 1 weighting																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		// for (int i = N0; i < N_tot; i++) {
		// 	g1s[i - N0] = gsl_ran_gaussian_pdf(solns1[i - N0] - obs, obs_sd);
		// 	g0s[i] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd);
		// }


		// for (int i = 0; i < N0; i++) {

		// 	weights[i] = g0s[i] * (double) res_signs[i] / (double) N0;
		// 	absolute_weights[i] = fabs(weights[i]);
		// 	normaliser += weights[i];
		// 	abs_normaliser += absolute_weights[i];

		// }

		// // """ The weight of your particle here needs to be corrected by this """
		// for (int i = N0; i < N_tot; i++) {

		// 	weights[i] = (g1s[i - N0] - g0s[i]) * (double) res_signs[i] / (double) N1;
		// 	absolute_weights[i] = fabs(weights[i]);
		// 	normaliser += weights[i];
		// 	abs_normaliser += absolute_weights[i];

		// }





			// fprintf(TV_MEASURE, "%e %e\n", sig_thetas[i], level_1_abs_weights[i - N0]);  // The normalised TV measure we resample from

		// for (int i = 0; i < N0; i++)
			// fprintf(RES_DISTR, "%e ", ext_sig_thetas[i]);  // The resampled particles (we don't need the weights)
		// fprintf(RES_DISTR, "\n");


void ml_bootstrap_particle_filter_timed(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	if (length > 2)
		length = 2;
	int nx1 = hmm->nx;
	int nx0 = nxs[0];
	int obs_pos1 = nx1 - 1;
	int obs_pos0 = nx0 - 1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	int poly_degree = 1, M_poly = poly_degree + 1, mesh_size = 1000;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right, k = hmm->k;
	double dx1 = (space_right - space_left) / (double) (nx1 - 1);
	double dx0 = (space_right - space_left) / (double) (nx0 - 1);
	double low_bd = hmm->low_bd, upp_bd = hmm->upp_bd;
	double obs, normaliser, abs_normaliser, x_hat, g0, g1, sign_rat;
	double Z_obs, gamma_of_k = tgamma(k), gamma_theta;
	short * signs = (short *) malloc(N_tot * sizeof(short));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	int * permutation = (int *) malloc(N_tot * sizeof(int));
	double * thetas = (double *) calloc(N_tot, sizeof(double));
	double * sig_thetas = (double *) calloc(N_tot, sizeof(double));
	double * res_thetas = (double *) malloc(N_tot * sizeof(double));
	double * weights = (double *) calloc(N_tot, sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * solns1 = (double *) malloc(N1 * sizeof(double));
	double * solns0 = (double *) malloc(N_tot * sizeof(double));
	double * g0s = (double *) malloc(N1 * sizeof(double));
	double * g1s = (double *) malloc(N1 * sizeof(double));
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * poly_weights = (double *) malloc(M_poly * sizeof(double));
	double * sig_theta_mesh = (double *) malloc(mesh_size * sizeof(double));


	/* Solver arrays */
	/* ------------- */
	double * h1 = (double *) malloc(nx1 * sizeof(double));
	double * Z_x1 = (double *) malloc(nx1 * sizeof(double));
	double * xs1 = (double *) malloc(nx1 * sizeof(double));
	double * h0 = (double *) malloc(nx0 * sizeof(double));
	double * Z_x0 = (double *) malloc(nx0 * sizeof(double));
	double * xs0 = (double *) malloc(nx0 * sizeof(double));
	for (int j = 0; j < nx1; j++)
		xs1[j] = space_left + j * dx1;
	for (int j = 0; j < nx0; j++)
		xs0[j] = space_left + j * dx0;


	/* Regression matrices */
	/* ------------------- */
	double * PHI = (double *) malloc(N1 * M_poly * sizeof(double));
	double * C = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * C_inv = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * MP = (double *) malloc(N1 * M_poly * sizeof(double));
	gsl_matrix * C_gsl = gsl_matrix_alloc(M_poly, M_poly);
	gsl_permutation * p = gsl_permutation_alloc(M_poly);
	gsl_matrix * C_inv_gsl = gsl_matrix_alloc(M_poly, M_poly);


	/* Initial conditions */
	/* ------------------ */
	double theta_init = sigmoid_inv(hmm->signal[0], upp_bd - low_bd, low_bd);
	double h_init = hmm->h_init, q_init = hmm->q0, q_init_sq = q_init * q_init;
	h1[0] = h_init, h0[0] = h_init;
	for (int i = 0; i < N_tot; i++) {
		thetas[i] = theta_init + gsl_ran_gaussian(rng, sig_sd);
		res_signs[i] = 1;
	}	


	/* Files */
	/* ----- */
	FILE * X_HATS = fopen("ml_xhats.txt", "w");
	FILE * CORRECTIONS = fopen("corrections.txt", "w");
	FILE * REGRESSION_CURVE = fopen("regression_curve.txt", "w");
	FILE * TRUE_CURVE = fopen("true_curve.txt", "w");
	FILE * TRUE_CURVE0 = fopen("true_curve0.txt", "w");
	FILE * LEVEL1_FINE = fopen("level1_fine.txt", "w");
	FILE * LEVEL1_COARSE = fopen("level1_coarse.txt", "w");
	FILE * LEVEL0_COARSE = fopen("level0_coarse.txt", "w");
	FILE * ML_DISTR = fopen("ml_distr.txt", "w");
	FILE * SIGNS = fopen("signs.txt", "w");
	fprintf(REGRESSION_CURVE, "%d\n", mesh_size);
	fprintf(ML_DISTR, "%d %d %d\n", N0, N1, N_tot);
	fprintf(SIGNS, "%d %d %d\n", N0, N1, N_tot);
	double lmbda = 1.0 * M_PI;


	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0, abs_normaliser = 0.0;
		for (int i = 0; i < N_tot; i++)
			sig_thetas[i] = sigmoid(thetas[i], upp_bd - low_bd, low_bd);

		

		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 1 solutions																						 																					   */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);

			/* Fine solution */
			solve(k, sig_thetas[i], nx1, xs1, Z_x1, h1, dx1, q_init_sq, gamma_theta);
			// solns1[i - N0] = h1[obs_pos1];
			solns1[i - N0] = sin(lmbda * h1[obs_pos1]) + h1[obs_pos1];

			/* Coarse solution */
			solve(k, sig_thetas[i], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			// solns0[i] = h0[obs_pos0];
			solns0[i] = sin(lmbda * h0[obs_pos0]) + h0[obs_pos0];

			/* Record the corrections samples for the regression approximation to the true correction curve */
			corrections[i - N0] = solns1[i - N0] - solns0[i];
			// fprintf(CORRECTIONS, "%e %e\n", sig_thetas[i], corrections[i - N0]);

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 0 solutions																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N0; i++) {

			gamma_theta = gamma_of_k * pow(sig_thetas[i], k);
			solve(k, sig_thetas[i], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			// solns0[i] = h0[obs_pos0];
			solns0[i] = sin(lmbda * h0[obs_pos0]) + h0[obs_pos0];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Regresion corrections																			 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		if (N1 > 0) {
			;
			regression_fit(sig_thetas, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);
			for (int i = 0; i < N_tot; i++)
				solns0[i] += poly_eval(sig_thetas[i], poly_weights, poly_degree);

			// generate_adaptive_artificial_mesh(N_tot, sig_thetas, mesh_size, sig_theta_mesh);
			// for (int l = 0; l < mesh_size; l++) {

				/* Output the regressed correction curve approximation over the artificial particle mesh */
				// fprintf(REGRESSION_CURVE, "%e %e\n", sig_theta_mesh[l], poly_eval(sig_theta_mesh[l], poly_weights, poly_degree));

				/* Output the true correction curve */
			// 	gamma_theta = gamma_of_k * pow(sig_theta_mesh[l], k);
			// 	solve(k, sig_theta_mesh[l], nx1, xs1, Z_x1, h1, dx1, q_init_sq, gamma_theta);
			// 	g1 = h1[obs_pos1];
			// 	solve(k, sig_theta_mesh[l], nx0, xs0, Z_x0, h0, dx0, q_init_sq, gamma_theta);
			// 	g0 = h0[obs_pos0];
			// 	g0 += poly_eval(sig_theta_mesh[l], poly_weights, poly_degree);
			// //	// fprintf(TRUE_CURVE, "%e ", g1 - g0);
			// 	fprintf(TRUE_CURVE, "%e ", g1);
			// 	fprintf(TRUE_CURVE0, "%e ", g0);
			// }
			// fprintf(TRUE_CURVE, "\n");
			// fprintf(TRUE_CURVE0, "\n");

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight assignment																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */

		/* Level 1 */
		/* ------- */
		for (int i = N0; i < N_tot; i++) {

			g1s[i - N0] = gsl_ran_gaussian_pdf(solns1[i - N0] - obs, obs_sd);
			g0s[i - N0] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd);

			// fprintf(LEVEL1_FINE, "%e %e\n", solns1[i - N0], g1s[i - N0]);
			// fprintf(LEVEL1_COARSE, "%e %e\n", solns0[i], g0s[i - N0]);

			// fprintf(LEVEL1_FINE, "%e %e\n", sig_thetas[i], g1s[i - N0]);
			// fprintf(LEVEL1_COARSE, "%e %e\n", sig_thetas[i], g0s[i - N0]);

		}
		for (int i = N0; i < N_tot; i++)
			weights[i] = (g1s[i - N0] - g0s[i - N0]) / (double) N1;


		/* Level 0 */
		/* ------- */
		for (int i = 0; i < N0; i++) {
			weights[i] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd) / (double) N0;

			// fprintf(LEVEL0_COARSE, "%e %e\n", solns0[i], weights[i] * N0);
			// fprintf(LEVEL0_COARSE, "%e %e\n", sig_thetas[i], weights[i] * N0);

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Normalisation 																							 																						 	 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N_tot; i++) {

			/* Scale the weights by the previous sign and compute the new sign */
			weights[i] *= res_signs[i];
			signs[i] = weights[i] < 0 ? -1 : 1;
			absolute_weights[i] = fabs(weights[i]);
			
			/* Compute the normalisation terms */
			normaliser += weights[i];
			abs_normaliser += absolute_weights[i];

		}		
		// for (int i = N0; i < N_tot; i++) {
		// 	fprintf(LEVEL1_FINE, "%e %e\n", sig_thetas[i], g1s[i - N0] / (double) N1 / (double) normaliser);
		// 	fprintf(LEVEL1_COARSE, "%e %e\n", sig_thetas[i], g0s[i - N0] / (double) N1 / (double) normaliser);
		// }
		// for (int i = 0; i < N0; i++)
		// 	fprintf(LEVEL0_COARSE, "%e %e\n", sig_thetas[i], weights[i] / (double) normaliser);

		x_hat = 0.0, sign_rat = 0.0;
		for (int i = 0; i < N_tot; i++) {
			weights[i] /= normaliser;
			absolute_weights[i] /= abs_normaliser;
			ml_weighted[n][i].x = sig_thetas[i];
			ml_weighted[n][i].w = weights[i];
			x_hat += sig_thetas[i] * weights[i];
			// sign_rat += absolute_weights[i] * signs[i];
			// fprintf(SIGNS, "%e %hd %e\n", sig_thetas[i], signs[i], absolute_weights[i]);
		}
		fprintf(X_HATS, "%e ", x_hat);
		// sign_ratios[n] = sign_rat;
		// fprintf(SIGNS, "\n");
		// for (int i = 0; i < N_tot; i++)
		// 	fprintf(ML_DISTR, "%e %e\n", ml_weighted[n][i].x, ml_weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N_tot, absolute_weights, ind, rng);
		random_permuter(permutation, N_tot, rng);
		for (int i = 0; i < N_tot; i++) {
			res_thetas[permutation[i]] = thetas[ind[i]];
			res_signs[permutation[i]] = signs[ind[i]];
		}
		mutate(rng, N_tot, thetas, res_thetas, sig_sd);

	}

	fclose(X_HATS);
	fclose(CORRECTIONS);
	fclose(REGRESSION_CURVE);
	fclose(TRUE_CURVE);
	fclose(TRUE_CURVE0);
	fclose(LEVEL1_FINE);
	fclose(LEVEL1_COARSE);
	fclose(LEVEL0_COARSE);
	fclose(ML_DISTR);
	fclose(SIGNS);

	free(signs);
	free(res_signs);
	free(ind);
	free(permutation);
	free(thetas);
	free(sig_thetas);
	free(res_thetas);
	free(weights);
	free(absolute_weights);
	free(solns1);
	free(solns0);
	free(g0s);
	free(g1s);
	free(corrections);
	free(poly_weights);
	free(h1);
	free(Z_x1);
	free(xs1);
	free(h0);
	free(Z_x0);
	free(xs0);
	free(PHI);
	free(C);
	free(C_inv);
	free(MP);
	free(sig_theta_mesh);
	gsl_matrix_free(C_gsl);
	gsl_permutation_free(p);
	gsl_matrix_free(C_inv_gsl);

}




