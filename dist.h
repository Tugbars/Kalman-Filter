/*
 * dist_refactored.h
 *
 * Header file for refactored gamma and beta distribution functions from dist.c,
 * used as dependencies for pdist.c and indirectly for emle.c via initest.h.
 */

#ifndef DIST_REFACTORED_H_
#define DIST_REFACTORED_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the maximum of two doubles.
 *
 * @param a First value.
 * @param b Second value.
 * @return Maximum of a and b.
 */
double r8_max(double a, double b);

/**
 * @brief Rounds a double to the integer nearest to zero.
 *
 * @param x Input value.
 * @return Rounded integer value.
 */
double fix_number(double x);

/**
 * @brief Computes the absolute value of a double.
 *
 * @param x Input value.
 * @return Absolute value |x|.
 */
double absolute_value(double x);

/**
 * @brief Computes the machine epsilon scaled by the input value.
 *
 * @param x Input value for scaling.
 * @return Scaled machine epsilon.
 */
double machine_epsilon(double x);

/**
 * @brief Computes log(1 + x) with high precision for small x.
 *
 * @param x Input value.
 * @return log(1 + x), or x if x is very small, or NAN if invalid.
 */
double log_one_plus(double x);

/**
 * @brief Recursively computes the factorial of a number.
 *
 * @param n Input value (assumed integer-like).
 * @return Factorial of n, or NAN if invalid.
 */
double recursive_factorial(double n);

/**
 * @brief Computes the factorial of an integer.
 *
 * @param n Integer input.
 * @return n!, or NAN if invalid.
 */
double compute_factorial(int n);

/**
 * @brief Computes the gamma function for a positive real number.
 *
 * @param x Input value (positive real).
 * @return Gamma(x), or NAN if invalid or overflow.
 */
double gamma_function(double x);

/**
 * @brief Computes the logarithm of the gamma function.
 *
 * @param x Input value (positive real).
 * @return log(Gamma(x)), or NAN if invalid or overflow.
 */
double gamma_log(double x);

/**
 * @brief Computes the beta function.
 *
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Beta(a, b), or NAN if invalid.
 */
double beta_function(double shape_a, double shape_b);

/**
 * @brief Computes the logarithm of the beta function.
 *
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return log(Beta(a, b)), or NAN if invalid.
 */
double beta_log(double shape_a, double shape_b);

/**
 * @brief Computes the regularized incomplete gamma function (lower tail).
 *
 * @param x Upper limit of integration (x >= 0).
 * @param shape Shape parameter (a >= 0).
 * @return P(x, a), or NAN if invalid.
 */
double gamma_cdf_lower(double x, double shape);

/**
 * @brief Computes the regularized incomplete gamma function (upper tail).
 *
 * @param x Upper limit of integration (x >= 0).
 * @param shape Shape parameter (a >= 0).
 * @return Q(x, a), or NAN if invalid.
 */
double gamma_cdf_upper(double x, double shape);

/**
 * @brief Computes the continued fraction for the incomplete beta function.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @param iteration_count Output number of iterations used.
 * @return Continued fraction value, or NAN if invalid.
 */
double beta_continued_fraction(double x, double shape_a, double shape_b, int* iteration_count);

/**
 * @brief Computes an approximation to the incomplete beta function.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Approximated incomplete beta value, or NAN if invalid.
 */
double beta_incomplete_approx(double x, double shape_a, double shape_b);

/**
 * @brief Computes the regularized incomplete beta function.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return I_x(a, b), or NAN if invalid.
 */
double beta_incomplete(double x, double shape_a, double shape_b);

/**
 * @brief Computes the complementary incomplete beta function.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return I_{1-x}(b, a), or NAN if invalid.
 */
double beta_incomplete_complement(double x, double shape_a, double shape_b);

/**
 * @brief Computes the PDF of the beta distribution.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return PDF value, or NAN if invalid.
 */
double beta_pdf(double x, double shape_a, double shape_b);

/**
 * @brief Computes the CDF of the beta distribution.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return CDF value, or NAN if invalid.
 */
double beta_cdf(double x, double shape_a, double shape_b);

/**
 * @brief Computes the derivative of the incomplete beta function.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Derivative value, or NAN if invalid.
 */
double beta_incomplete_derivative(double x, double shape_a, double shape_b);

/**
 * @brief Computes the inverse CDF of the beta distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Quantile value, or NAN if invalid or convergence fails.
 */
double beta_inv(double probability, double shape_a, double shape_b);

#ifdef __cplusplus
}
#endif

#endif /* DIST_REFACTORED_H_ */