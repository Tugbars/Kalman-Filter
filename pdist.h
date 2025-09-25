/*
 * pdist_refactored.h
 *
 * Header file for refactored statistical distribution functions from pdist.c,
 * used as dependencies for regression functions in initest.h, supporting emle.c.
 */

#ifndef PDIST_REFACTORED_H_
#define PDIST_REFACTORED_H_

#include "dist.h"
#include "erfunc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PIVAL 3.1415926535897932384626434
#define XINFVAL 1.79e308

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the probability density function (PDF) of the normal distribution.
 *
 * @param x Input value.
 * @param mean Mean of the distribution (mu).
 * @param std_dev Standard deviation of the distribution (sigma).
 * @return PDF value, or NAN if sigma is invalid.
 */
double normal_pdf(double x, double mean, double std_dev);

/**
 * @brief Computes the cumulative distribution function (CDF) of the normal distribution.
 *
 * @param x Input value.
 * @param mean Mean of the distribution (mu).
 * @param std_dev Standard deviation of the distribution (sigma).
 * @return CDF value, or NAN if sigma is invalid.
 */
double normal_cdf(double x, double mean, double std_dev);

/**
 * @brief Computes the inverse CDF of the normal distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param mean Mean of the distribution (mu).
 * @param std_dev Standard deviation of the distribution (sigma).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double normal_inv(double probability, double mean, double std_dev);

/**
 * @brief Computes the probability density function (PDF) of the t-distribution.
 *
 * @param t_value t-statistic value.
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return PDF value, or NAN if df is invalid.
 */
double t_pdf(double t_value, int degrees_freedom);

/**
 * @brief Computes the cumulative distribution function (CDF) of the t-distribution.
 *
 * @param t_value t-statistic value.
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return CDF value, or NAN if inputs are invalid.
 */
double t_cdf(double t_value, int degrees_freedom);

/**
 * @brief Computes the inverse CDF of the t-distribution (approximation for large df).
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return t-critical value, or NAN if inputs are invalid.
 */
double t_inv_approx(double probability, int degrees_freedom);

/**
 * @brief Computes the inverse CDF of the t-distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return t-critical value, or NAN if inputs are invalid.
 */
double t_inv(double probability, int degrees_freedom);

/**
 * @brief Computes the probability density function (PDF) of the F-distribution.
 *
 * @param x Input value (non-negative).
 * @param df_numerator Numerator degrees of freedom (positive integer).
 * @param df_denominator Denominator degrees of freedom (positive integer).
 * @return PDF value, or NAN if inputs are invalid.
 */
double f_pdf(double x, int df_numerator, int df_denominator);

/**
 * @brief Computes the cumulative distribution function (CDF) of the F-distribution.
 *
 * @param x Input value (non-negative).
 * @param df_numerator Numerator degrees of freedom (positive integer).
 * @param df_denominator Denominator degrees of freedom (positive integer).
 * @return CDF value, or NAN if inputs are invalid.
 */
double f_cdf(double x, int df_numerator, int df_denominator);

/**
 * @brief Computes the inverse CDF of the F-distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param df_numerator Numerator degrees of freedom (positive integer).
 * @param df_denominator Denominator degrees of freedom (positive integer).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double f_inv(double probability, int df_numerator, int df_denominator);

/**
 * @brief Computes the probability density function (PDF) of the gamma distribution.
 *
 * @param x Input value (non-negative).
 * @param shape Shape parameter (k > 0).
 * @param scale Scale parameter (theta > 0).
 * @return PDF value, or NAN if inputs are invalid.
 */
double gamma_pdf(double x, double shape, double scale);

/**
 * @brief Computes the cumulative distribution function (CDF) of the gamma distribution.
 *
 * @param x Input value (non-negative).
 * @param shape Shape parameter (k > 0).
 * @param scale Scale parameter (theta > 0).
 * @return CDF value, or NAN if inputs are invalid.
 */
double gamma_cdf(double x, double shape, double scale);

/**
 * @brief Computes the inverse CDF of the gamma distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param shape Shape parameter (k > 0).
 * @param scale Scale parameter (theta > 0).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double gamma_inv(double probability, double shape, double scale);

/**
 * @brief Computes the probability density function (PDF) of the chi-squared distribution.
 *
 * @param x Input value (non-negative).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return PDF value, or NAN if inputs are invalid.
 */
double chi_pdf(double x, int degrees_freedom);

/**
 * @brief Computes the cumulative distribution function (CDF) of the chi-squared distribution.
 *
 * @param x Input value (non-negative).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return CDF value, or NAN if inputs are invalid.
 */
double chi_cdf(double x, int degrees_freedom);

/**
 * @brief Computes the inverse CDF of the chi-squared distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double chi_inv(double probability, int degrees_freedom);

#ifdef __cplusplus
}
#endif

#endif /* PDIST_REFACTORED_H_ */