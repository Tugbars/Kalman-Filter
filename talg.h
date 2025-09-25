/*
 * talg_refactored.h
 *
 * Header file for refactored time series analysis functions from talg.c,
 * used as dependencies for emle.c Kalman filter functions.
 */

#ifndef TALG_REFACTORED_H_
#define TALG_REFACTORED_H_

#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the product of two polynomials.
 *
 * Multiplies two polynomials to produce a result polynomial.
 *
 * @param poly_a First polynomial coefficients (length len_a).
 * @param poly_b Second polynomial coefficients (length len_b).
 * @param poly_result Output polynomial coefficients (length len_a + len_b - 1).
 * @param len_a Length of poly_a.
 * @param len_b Length of poly_b.
 * @return Length of the resulting polynomial (len_a + len_b - 1).
 */
int poly(double *poly_a, double *poly_b, double *poly_result, int len_a, int len_b);

/**
 * @brief Computes coefficients of the differencing operator [1, -1]^d.
 *
 * Generates coefficients for (1 - z)^d used in non-seasonal differencing.
 *
 * @param diff_order Differencing order (d).
 * @param diff_coeffs Output coefficients (length d + 1).
 */
void deld(int diff_order, double *diff_coeffs);

/**
 * @brief Computes coefficients of the seasonal differencing operator [1, 0, ..., -1]^D.
 *
 * Generates coefficients for (1 - z^s)^D used in seasonal differencing.
 *
 * @param seasonal_diff_order Seasonal differencing order (D).
 * @param seasonal_period Seasonal period (s).
 * @param seasonal_diff_coeffs Output coefficients (length D * s + 1).
 */
void delds(int seasonal_diff_order, int seasonal_period, double *seasonal_diff_coeffs);

/**
 * @brief Checks if an AR polynomial is stationary.
 *
 * Verifies if all roots of the AR polynomial lie outside the unit circle.
 *
 * @param ar_order Order of the AR polynomial (p).
 * @param ar_coeffs AR coefficients (length p).
 * @return 1 if stationary, 0 if non-stationary.
 */
int archeck(int ar_order, double *ar_coeffs);

/**
 * @brief Inverts MA polynomial roots to ensure invertibility.
 *
 * Adjusts MA coefficients if roots are inside or on the unit circle.
 *
 * @param ma_order Order of the MA polynomial (q).
 * @param ma_coeffs MA coefficients (length q, modified in-place).
 * @return 0 if no inversion needed, 1 if inversion performed.
 */
int invertroot(int ma_order, double *ma_coeffs);

/**
 * @brief Computes the autocovariance function for an ARMA model.
 *
 * Calculates autocovariances for an ARMA(p, q) model for Kalman filter initialization.
 *
 * @param ar_coeffs AR coefficients (length ar_order).
 * @param ar_order AR order (p).
 * @param ma_coeffs MA coefficients (length ma_order).
 * @param ma_order MA order (q).
 * @param autocov Output autocovariance function (length autocov_len).
 * @param autocov_len Length of autocov.
 * @param cvli_output Output CVLI vector (length cvli_len).
 * @param cvli_len Length of cvli_output.
 * @param alpha_output Output ALPHA vector (length alpha_len).
 * @param alpha_len Length of alpha_output.
 * @return 0 on success, non-zero on error.
 */
int twacf(double *ar_coeffs, int ar_order, double *ma_coeffs, int ma_order, double *autocov, int autocov_len, double *cvli_output, int cvli_len, double *alpha_output, int alpha_len);

/**
 * @brief Computes roots of a polynomial.
 *
 * Finds the real and imaginary parts of the roots of a polynomial.
 *
 * @param coeffs Polynomial coefficients (length degree + 1, highest degree first).
 * @param degree Polynomial degree.
 * @param real_roots Real parts of roots (output, length degree).
 * @param imag_roots Imaginary parts of roots (output, length degree).
 * @return 0 on success, -1 on failure.
 */
int polyroot(double *coeffs, int degree, double *real_roots, double *imag_roots);

#ifdef __cplusplus
}
#endif

#endif /* TALG_REFACTORED_H_ */