// SPDX-License-Identifier: BSD-3-Clause
#include "talg.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Computes the product of two polynomials.
 *
 * Multiplies two polynomials A and B to produce polynomial C, where the degree of C is
 * the sum of the degrees of A and B minus 1. The result is stored in C.
 *
 * @param poly_a First polynomial coefficients (length lA).
 * @param poly_b Second polynomial coefficients (length lB).
 * @param poly_result Output polynomial coefficients (length lA + lB - 1).
 * @param len_a Length of poly_a.
 * @param len_b Length of poly_b.
 * @return Length of the resulting polynomial (lA + lB - 1).
 */
int poly(double *poly_a, double *poly_b, double *poly_result, int len_a, int len_b) {
    // Step 1: Calculate the length of the resulting polynomial
    int len_result = len_a + len_b - 1; // lC = lA + lB - 1

    // Step 2: Initialize result array to zero
    for (int i = 0; i < len_result; ++i) {
        poly_result[i] = 0.0; // Clear output array
    }

    // Step 3: Compute polynomial multiplication
    for (int i = 0; i < len_result; ++i) {
        double temp_sum = 0.0; // Accumulator for coefficient
        for (int j = 0; j < len_a; ++j) {
            for (int k = 0; k < len_b; ++k) {
                if (j + k == i) { // Only add terms where degree matches i
                    temp_sum += poly_a[j] * poly_b[k];
                }
            }
        }
        poly_result[i] = temp_sum; // Store coefficient
    }

    return len_result; // Return length of result
}

/**
 * @brief Computes coefficients of the differencing operator [1, -1]^d.
 *
 * Generates the coefficients of the polynomial (1 - z)^d, used for non-seasonal differencing.
 * For example, d=2 yields [1, -2, 1], d=3 yields [1, -3, 3, -1].
 *
 * @param diff_order Differencing order (d).
 * @param diff_coeffs Output coefficients (length d + 1).
 */
void deld(int diff_order, double *diff_coeffs) {
    // Step 1: Allocate temporary arrays
    double *base_vec = (double*)malloc(sizeof(double) * 2); // Base polynomial [1, -1]
    double *temp_result = (double*)malloc(sizeof(double) * (diff_order + 1)); // Temporary result
    base_vec[0] = 1.0; // [1,
    base_vec[1] = -1.0; // -1]
    temp_result[0] = 1.0; // Initialize result with [1]

    // Step 2: Iteratively compute (1 - z)^d by polynomial multiplication
    for (int i = 0; i < diff_order; ++i) {
        poly(temp_result, base_vec, diff_coeffs, i + 1, 2); // Multiply current result by [1, -1]
        for (int j = 0; j < i + 2; ++j) {
            temp_result[j] = diff_coeffs[j]; // Update temp_result for next iteration
        }
    }

    // Step 3: Free allocated memory
    free(base_vec);
    free(temp_result);
}

/**
 * @brief Computes coefficients of the seasonal differencing operator [1, 0, ..., -1]^D.
 *
 * Generates the coefficients of the polynomial (1 - z^s)^D, used for seasonal differencing.
 * For example, D=1, s=4 yields [1, 0, 0, 0, -1]; D=2, s=4 yields [1, 0, 0, 0, -2, 0, 0, 0, 1].
 *
 * @param seasonal_diff_order Seasonal differencing order (D).
 * @param seasonal_period Seasonal period (s, e.g., 4 for quarterly, 12 for yearly).
 * @param seasonal_diff_coeffs Output coefficients (length D * s + 1).
 */
void delds(int seasonal_diff_order, int seasonal_period, double *seasonal_diff_coeffs) {
    // Step 1: Allocate temporary arrays
    double *base_vec = (double*)malloc(sizeof(double) * (seasonal_period + 1)); // Base polynomial [1, 0, ..., -1]
    double *temp_result = (double*)malloc(sizeof(double) * (seasonal_diff_order * seasonal_period + 1)); // Temporary result
    for (int i = 0; i < seasonal_period + 1; ++i) {
        base_vec[i] = 0.0; // Initialize to zero
    }
    base_vec[0] = 1.0; // [1,
    base_vec[seasonal_period] = -1.0; // 0, ..., -1]
    temp_result[0] = 1.0; // Initialize result with [1]

    // Step 2: Iteratively compute (1 - z^s)^D by polynomial multiplication
    for (int i = 0; i < seasonal_diff_order; ++i) {
        poly(temp_result, base_vec, seasonal_diff_coeffs, i * seasonal_period + 1, seasonal_period + 1); // Multiply
        for (int j = 0; j < seasonal_period * (i + 1) + 1; ++j) {
            temp_result[j] = seasonal_diff_coeffs[j]; // Update temp_result
        }
    }

    // Step 3: Free allocated memory
    free(base_vec);
    free(temp_result);
}

/**
 * @brief Checks if an AR polynomial is stationary by examining its roots.
 *
 * Computes the roots of the AR polynomial 1 - phi_1 z - phi_2 z^2 - ... - phi_p z^p
 * and checks if all roots have magnitude greater than 1 (outside the unit circle).
 *
 * @param ar_order Order of the AR polynomial (p).
 * @param ar_coeffs AR coefficients (length p).
 * @return 1 if stationary, 0 if non-stationary.
 */
int archeck(int ar_order, double *ar_coeffs) {
    // Step 1: Allocate arrays for polynomial and roots
    int poly_degree = ar_order + 1; // Degree of polynomial (p + 1)
    int check_stationary = 1; // Default to stationary
    double *poly_coeffs = (double*)malloc(sizeof(double) * poly_degree); // Polynomial [1, -phi_1, ..., -phi_p]
    double *real_roots = (double*)malloc(sizeof(double) * ar_order); // Real parts of roots
    double *imag_roots = (double*)malloc(sizeof(double) * ar_order); // Imaginary parts of roots
    int *nonzero_indices = (int*)malloc(sizeof(int) * poly_degree); // Indices of nonzero coefficients

    // Step 2: Form polynomial 1 - phi_1 z - ... - phi_p z^p
    poly_coeffs[0] = 1.0; // Constant term
    for (int i = 0; i < ar_order; ++i) {
        poly_coeffs[i + 1] = -ar_coeffs[i]; // Negative coefficients
    }

    // Step 3: Identify nonzero coefficients
    int max_nonzero_index = -1;
    for (int i = 0; i < poly_degree; ++i) {
        if (poly_coeffs[i] != 0.0) {
            nonzero_indices[i] = i;
            if (max_nonzero_index < i) {
                max_nonzero_index = i; // Track highest nonzero index
            }
        } else {
            nonzero_indices[i] = -1;
        }
    }

    // Step 4: If no nonzero coefficients (except constant), return stationary
    if (max_nonzero_index <= 0) {
        free(poly_coeffs);
        free(real_roots);
        free(imag_roots);
        free(nonzero_indices);
        return 1;
    }

    // Step 5: Compute roots of the polynomial
    polyroot(poly_coeffs, ar_order, real_roots, imag_roots); // External call to find roots

    // Step 6: Check if all roots have magnitude > 1
    for (int i = 0; i < ar_order; ++i) {
        double root_magnitude = sqrt(pow(real_roots[i], 2.0) + pow(imag_roots[i], 2.0));
        if (root_magnitude <= 1.0) { // Non-stationary if any root is inside or on unit circle
            check_stationary = 0;
            break;
        }
    }

    // Step 7: Free allocated memory
    free(poly_coeffs);
    free(real_roots);
    free(imag_roots);
    free(nonzero_indices);

    return check_stationary;
}

/**
 * @brief Inverts MA polynomial roots to ensure invertibility.
 *
 * Computes the roots of the MA polynomial 1 + theta_1 z + ... + theta_q z^q and inverts
 * any roots with magnitude <= 1 to ensure they lie outside the unit circle. Updates the
 * MA coefficients in-place.
 *
 * @param ma_order Order of the MA polynomial (q).
 * @param ma_coeffs MA coefficients (length q, modified in-place).
 * @return 0 if no inversion needed, 1 if inversion was performed.
 */
int invertroot(int ma_order, double *ma_coeffs) {
    // Step 1: Find the highest index with nonzero coefficient
    int nonzero_index = -1;
    for (int i = 0; i < ma_order; ++i) {
        if (ma_coeffs[i] != 0.0) {
            nonzero_index = i; // Track last nonzero coefficient
        }
    }

    // Step 2: If all coefficients are zero, return
    if (nonzero_index == -1) {
        return 0;
    }

    // Step 3: Allocate arrays for polynomial and roots
    nonzero_index++; // Effective polynomial degree
    double *poly_coeffs = (double*)malloc(sizeof(double) * (nonzero_index + 1)); // Polynomial [1, theta_1, ..., theta_q]
    double *real_roots = (double*)malloc(sizeof(double) * nonzero_index); // Real parts
    double *imag_roots = (double*)malloc(sizeof(double) * nonzero_index); // Imaginary parts
    int *invert_flags = (int*)malloc(sizeof(int) * nonzero_index); // Flags for roots to invert
    double *real_temp = (double*)malloc(sizeof(double) * (nonzero_index + 1)); // Temp real parts
    double *imag_temp = (double*)malloc(sizeof(double) * (nonzero_index + 1)); // Temp imaginary parts
    double *real_result = (double*)malloc(sizeof(double) * (nonzero_index + 1)); // Result real parts
    double *imag_result = (double*)malloc(sizeof(double) * (nonzero_index + 1)); // Result imaginary parts

    // Step 4: Form polynomial 1 + theta_1 z + ... + theta_q z^q
    poly_coeffs[0] = 1.0;
    for (int i = 1; i <= nonzero_index; ++i) {
        poly_coeffs[i] = ma_coeffs[i - 1];
    }

    // Step 5: Compute roots
    int root_error = polyroot(poly_coeffs, nonzero_index, real_roots, imag_roots);
    if (root_error == 1) { // Check for root-finding failure
        free(real_roots);
        free(imag_roots);
        free(poly_coeffs);
        free(invert_flags);
        free(real_temp);
        free(imag_temp);
        free(real_result);
        free(imag_result);
        return 0;
    }

    // Step 6: Check for roots needing inversion
    int num_roots_to_invert = 0;
    for (int i = 0; i < nonzero_index; ++i) {
        double root_magnitude = real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i];
        invert_flags[i] = (root_magnitude < 1.0) ? 1 : 0; // Flag roots inside unit circle
        if (invert_flags[i] == 1) {
            num_roots_to_invert++;
        }
    }

    // Step 7: If no roots need inversion, return
    if (num_roots_to_invert == 0) {
        free(real_roots);
        free(imag_roots);
        free(poly_coeffs);
        free(invert_flags);
        free(real_temp);
        free(imag_temp);
        free(real_result);
        free(imag_result);
        return 0;
    }

    // Step 8: Handle special case for q=1
    if (nonzero_index == 1) {
        ma_coeffs[0] = 1.0 / ma_coeffs[0]; // Invert single coefficient
        for (int i = 1; i < ma_order; ++i) {
            ma_coeffs[i] = 0.0; // Clear remaining coefficients
        }
        free(real_roots);
        free(imag_roots);
        free(poly_coeffs);
        free(invert_flags);
        free(real_temp);
        free(imag_temp);
        free(real_result);
        free(imag_result);
        return 1;
    }

    // Step 9: Invert roots with magnitude < 1
    for (int i = 0; i < nonzero_index; ++i) {
        if (invert_flags[i] == 1) {
            double root_magnitude = real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i];
            real_roots[i] = real_roots[i] / root_magnitude; // Invert real part
            imag_roots[i] = -imag_roots[i] / root_magnitude; // Invert imaginary part
        }
    }

    // Step 10: Convert roots back to polynomial coefficients
    real_temp[0] = 1.0; imag_temp[0] = 0.0; // Initialize product
    real_result[0] = 1.0; imag_result[0] = 0.0;
    for (int i = 0; i < nonzero_index; ++i) {
        int i_plus_1 = i + 1;
        double root_magnitude = real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i];
        double temp_real = real_roots[i] / root_magnitude;
        double temp_imag = -imag_roots[i] / root_magnitude;
        real_temp[i_plus_1] = imag_temp[i_plus_1] = 0.0;
        for (int j = 1; j <= i_plus_1; ++j) {
            real_result[j] = temp_real * real_temp[j - 1] - temp_imag * imag_temp[j - 1];
            imag_result[j] = temp_real * imag_temp[j - 1] + temp_imag * real_temp[j - 1];
            real_result[j] = real_temp[j] - real_result[j];
            imag_result[j] = imag_temp[j] - imag_result[j];
        }
        for (int j = 1; j <= i_plus_1; ++j) {
            real_temp[j] = real_result[j];
            imag_temp[j] = imag_result[j];
        }
    }

    // Step 11: Update MA coefficients (real parts only, assuming real polynomial)
    for (int i = 0; i < nonzero_index; ++i) {
        ma_coeffs[i] = real_temp[i + 1];
    }
    for (int i = nonzero_index; i < ma_order; ++i) {
        ma_coeffs[i] = 0.0; // Clear remaining coefficients
    }

    // Step 12: Free allocated memory
    free(real_roots);
    free(imag_roots);
    free(poly_coeffs);
    free(invert_flags);
    free(real_temp);
    free(imag_temp);
    free(real_result);
    free(imag_result);

    return 1; // Inversion performed
}

/**
 * @brief Computes the autocovariance function for an ARMA model.
 *
 * Calculates the autocovariances of an ARMA(p, q) model, storing results in ACF, CVLI, and ALPHA.
 * Used to initialize the state-space representation for the Kalman filter in flikam.
 *
 * @param ar_coeffs AR coefficients (length MP).
 * @param ar_order AR order (MP).
 * @param ma_coeffs MA coefficients (length MQ).
 * @param ma_order MA order (MQ).
 * @param autocov Output autocovariance function (length MA).
 * @param autocov_len Length of autocov (MA).
 * @param cvli_output Output CVLI vector (length MXPQ1).
 * @param cvli_len Length of cvli_output (MXPQ1).
 * @param alpha_output Output ALPHA vector (length MXPQ).
 * @param alpha_len Length of alpha_output (MXPQ).
 * @return Error code (0 for success, non-zero for errors).
 */
int twacf(double *ar_coeffs, int ar_order, double *ma_coeffs, int ma_order, double *autocov, int autocov_len, double *cvli_output, int cvli_len, double *alpha_output, int alpha_len) {
    // Step 1: Initialize variables and constants
    int error_code = 0;
    double epsilon = 1.0e-10; // Numerical stability threshold
    double zero = 0.0, half = 0.5, one = 1.0, two = 2.0;
    int max_pq = imax(ar_order, ma_order); // Maximum of p and q
    int max_pq_plus_1 = max_pq + 1;

    // Step 2: Validate inputs
    if (ar_order < 0 || ma_order < 0) {
        error_code = 1; // Invalid order
    }
    if (alpha_len != max_pq) {
        error_code = 2; // Mismatch in ALPHA length
    }
    if (cvli_len != max_pq_plus_1) {
        error_code = 3; // Mismatch in CVLI length
    }
    if (autocov_len < max_pq_plus_1) {
        error_code = 4; // Insufficient ACF length
    }
    if (error_code > 0) {
        return error_code;
    }

    // Step 3: Initialize output arrays
    autocov[0] = one;
    cvli_output[0] = one;
    if (autocov_len == 1) {
        return error_code; // Early return for single lag
    }
    for (int i = 1; i < autocov_len; ++i) {
        autocov[i] = zero; // Clear ACF
    }
    if (max_pq_plus_1 == 1) {
        return error_code; // Early return for trivial case
    }
    for (int i = 1; i < cvli_len; ++i) {
        cvli_output[i] = zero; // Clear CVLI
    }
    for (int k = 0; k < alpha_len; ++k) {
        alpha_output[k] = 0.0; // Clear ALPHA
    }

    // Step 4: Compute ACF for MA part
    if (ma_order != 0) {
        for (int k = 1; k <= ma_order; ++k) {
            cvli_output[k] = -ma_coeffs[k - 1]; // CVLI[k] = -Q[k-1]
            autocov[k] = -ma_coeffs[k - 1]; // ACF[k] = -Q[k-1]
            int remaining_lags = ma_order - k;
            if (remaining_lags != 0) {
                for (int j = 1; j <= remaining_lags; ++j) {
                    int j_plus_k = j + k;
                    autocov[k] += (ma_coeffs[j - 1] * ma_coeffs[j_plus_k - 1]); // ACF[k] += Q[j-1] * Q[j+k-1]
                }
            }
            autocov[0] += (ma_coeffs[k - 1] * ma_coeffs[k - 1]); // ACF[0] += Q[k-1]^2
        }
    }

    // Step 5: Early return if no AR component
    if (ar_order == 0) {
        return error_code;
    }

    // Step 6: Initialize CVLI and ALPHA for AR part
    for (int k = 0; k < ar_order; ++k) {
        alpha_output[k] = ar_coeffs[k]; // ALPHA[k] = P[k]
        cvli_output[k] = ar_coeffs[k]; // CVLI[k] = P[k]
    }

    // Step 7: Compute T.W.-S ALPHA and DELTA
    for (int k = 1; k <= max_pq; ++k) {
        int remaining_lags = max_pq - k;
        if (remaining_lags < ar_order) {
            double divisor = one - alpha_output[remaining_lags] * alpha_output[remaining_lags];
            if (divisor <= epsilon) {
                return 5; // Numerical instability
            }
            if (remaining_lags == 0) {
                break;
            }
            for (int j = 1; j <= remaining_lags; ++j) {
                int k_minus_j = remaining_lags - j;
                alpha_output[j - 1] = (cvli_output[j - 1] + alpha_output[remaining_lags] * cvli_output[k_minus_j]) / divisor;
            }
        }
        if (remaining_lags < ma_order) {
            int j_start = imax(remaining_lags + 1 - ar_order, 1);
            for (int j = j_start; j <= remaining_lags; ++j) {
                int k_minus_j = remaining_lags - j;
                autocov[j] += autocov[remaining_lags + 1] * alpha_output[k_minus_j];
            }
        }
        if (remaining_lags < ar_order) {
            for (int j = 1; j <= remaining_lags; ++j) {
                cvli_output[j - 1] = alpha_output[j - 1];
            }
        }
    }

    // Step 8: Compute T.W.-S NU
    autocov[0] *= half; // Scale ACF[0]
    for (int k = 1; k <= max_pq; ++k) {
        if (k <= ar_order) {
            int k_plus_1 = k + 1;
            double divisor = one - alpha_output[k - 1] * alpha_output[k - 1];
            for (int j = 1; j <= k_plus_1; ++j) {
                int k_plus_2_minus_j = k + 2 - j;
                cvli_output[j - 1] = (autocov[j - 1] + alpha_output[k - 1] * autocov[k_plus_2_minus_j - 1]) / divisor;
            }
            for (int j = 1; j <= k_plus_1; ++j) {
                autocov[j - 1] = cvli_output[j - 1];
            }
        }
    }

    // Step 9: Finalize ACF computation
    for (int i = 1; i <= autocov_len; ++i) {
        int max_lags = imin(i - 1, ar_order);
        if (max_lags != 0) {
            for (int j = 1; j <= max_lags; ++j) {
                int i_minus_j = i - j;
                autocov[i - 1] += ar_coeffs[j - 1] * autocov[i_minus_j - 1];
            }
        }
    }
    autocov[0] *= two; // Restore ACF[0] scaling

    // Step 10: Compute CVLI for MA part
    cvli_output[0] = one;
    if (ma_order > 0) {
        for (int k = 1; k <= ma_order; ++k) {
            cvli_output[k] = -ma_coeffs[k - 1];
            if (ar_order != 0) {
                int max_lags = imin(k, ar_order);
                for (int j = 1; j <= max_lags; ++j) {
                    int k_plus_1_minus_j = k + 1 - j;
                    cvli_output[k] += ar_coeffs[j - 1] * cvli_output[k_plus_1_minus_j - 1];
                }
            }
        }
    }

    return error_code;
}