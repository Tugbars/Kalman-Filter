/*
 * erfunc_refactored.h
 *
 * Header file for refactored error function calculations from erfunc.c,
 * used as dependencies for pdist.c and indirectly for emle.c via initest.h.
 */

#ifndef ERFUNC_REFACTORED_H_
#define ERFUNC_REFACTORED_H_

#include <math.h>
#include <stdio.h>
#include <float.h>

#define PIVAL 3.1415926535897932384626434
#define XINFVAL 1.79e308
#define XNINFVAL 2.2251e-308

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the largest positive or negative exponent for exp(x).
 *
 * @param l Flag (0 for positive, non-zero for negative).
 * @return Largest exponent value.
 */
double compute_exp_limit(int l);

/**
 * @brief Computes the error function erf(x).
 *
 * @param x Input value.
 * @return erf(x), or ±1 for large |x|, or NAN if invalid.
 */
double erf(double x);

/**
 * @brief Computes the complementary error function erfc(x) or exp(x^2)*erfc(x).
 *
 * @param ind Flag (0 for erfc, non-zero for scaled erfc).
 * @param x Input value.
 * @return erfc(x) or exp(x^2)*erfc(x), or NAN if invalid.
 */
double compute_erfc(int ind, double x);

/**
 * @brief Computes the complementary error function erfc(x).
 *
 * @param x Input value.
 * @return erfc(x), or NAN if invalid.
 */
double erfc(double x);

/**
 * @brief Computes the scaled complementary error function exp(x^2)*erfc(x).
 *
 * @param x Input value.
 * @return exp(x^2)*erfc(x), or NAN if invalid.
 */
double erfc_scaled(double x);

/**
 * @brief Computes the inverse error function erfinv(x).
 *
 * @param x Input value (-1 <= x <= 1).
 * @return erfinv(x), or NAN if invalid.
 */
double erf_inv(double x);

/**
 * @brief Computes the inverse complementary error function erfcinv(x).
 *
 * @param x Input value (0 <= x <= 2).
 * @return erfcinv(x), or NAN if invalid.
 */
double erfc_inv(double x);

#ifdef __cplusplus
}
#endif

#endif /* ERFUNC_REFACTORED_H_ */