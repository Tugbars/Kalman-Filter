// SPDX-License-Identifier: BSD-3-Clause
#include "pdist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Computes the probability density function (PDF) of the normal distribution.
 *
 * Calculates the PDF for a normal distribution with mean mu and standard deviation sigma.
 *
 * @param x Input value.
 * @param mean Mean of the distribution (mu).
 * @param std_dev Standard deviation of the distribution (sigma).
 * @return PDF value, or NAN if sigma is invalid.
 */
double normal_pdf(double x, double mean, double std_dev) {
    // Step 1: Validate standard deviation
    if (std_dev < 0.0) {
        printf("Error: Standard deviation must be non-negative in normal_pdf\n");
        return NAN;
    }

    // Step 2: Compute normalized difference
    double normalized_diff = x - mean;
    double squared_diff = normalized_diff * normalized_diff;

    // Step 3: Compute PDF: exp(-((x-mu)^2)/(2*sigma^2)) / (sqrt(2*pi)*sigma)
    double exponent = -squared_diff / (2.0 * std_dev * std_dev);
    double denominator = sqrt(2.0 * PIVAL) * std_dev;
    if (denominator == 0.0) {
        printf("Error: Division by zero in normal_pdf\n");
        return NAN;
    }
    double pdf_value = exp(exponent) / denominator;

    return pdf_value;
}

/**
 * @brief Computes the cumulative distribution function (CDF) of the normal distribution.
 *
 * Calculates the CDF for a normal distribution with mean mu and standard deviation sigma.
 *
 * @param x Input value.
 * @param mean Mean of the distribution (mu).
 * @param std_dev Standard deviation of the distribution (sigma).
 * @return CDF value, or NAN if sigma is invalid.
 */
double normal_cdf(double x, double mean, double std_dev) {
    // Step 1: Validate standard deviation
    if (std_dev < 0.0) {
        printf("Error: Standard deviation must be non-negative in normal_cdf\n");
        return NAN;
    }

    // Step 2: Compute normalized value for erf
    double normalized_value = (x - mean) / (sqrt(2.0) * std_dev);
    if (std_dev == 0.0) {
        printf("Error: Division by zero in normal_cdf\n");
        return NAN;
    }

    // Step 3: Compute CDF: 0.5 * (1 + erf((x-mu)/(sqrt(2)*sigma)))
    double cdf_value = 0.5 * (1.0 + erf(normalized_value));

    return cdf_value;
}

/**
 * @brief Computes the inverse CDF of the normal distribution.
 *
 * Calculates the quantile for a given probability p in a normal distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param mean Mean of the distribution (mu).
 * @param std_dev Standard deviation of the distribution (sigma).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double normal_inv(double probability, double mean, double std_dev) {
    // Step 1: Validate inputs
    if (std_dev < 0.0) {
        printf("Error: Standard deviation must be non-negative in normal_inv\n");
        return NAN;
    }
    if (probability < 0.0 || probability > 1.0) {
        printf("Error: Probability must be between 0.0 and 1.0 in normal_inv\n");
        return NAN;
    }

    // Step 2: Compute inverse CDF using erfinv
    double normalized_quantile = sqrt(2.0) * erfinv(2.0 * probability - 1.0);
    if (isnan(normalized_quantile)) {
        printf("Error: erfinv failed in normal_inv\n");
        return NAN;
    }

    // Step 3: Scale and shift quantile
    double quantile = normalized_quantile * std_dev + mean;

    return quantile;
}

/**
 * @brief Computes the probability density function (PDF) of the t-distribution.
 *
 * Calculates the PDF for a t-distribution with df degrees of freedom.
 *
 * @param t_value t-statistic value.
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return PDF value, or NAN if df is invalid.
 */
double t_pdf(double t_value, int degrees_freedom) {
    // Step 1: Validate degrees of freedom
    if (degrees_freedom <= 0) {
        printf("Error: Degrees of freedom must be a positive integer in t_pdf\n");
        return NAN;
    }

    // Step 2: Compute log-gamma terms
    double log_gamma_diff = gamma_log((double)(degrees_freedom + 1.0) / 2.0) - gamma_log((double)(degrees_freedom / 2.0));
    double scale_factor = exp(log_gamma_diff);

    // Step 3: Compute power term: (1 + t^2/df)^(-(df+1)/2)
    double denominator = 1.0 + (t_value * t_value / (double)degrees_freedom);
    double power_term = pow(denominator, -(degrees_freedom + 1.0) / 2.0);

    // Step 4: Compute PDF: scale_factor * power_term / sqrt(pi * df)
    double pi = 3.1415926535897932384626434;
    double pdf_value = scale_factor * power_term / sqrt(pi * degrees_freedom);
    if (isnan(pdf_value) || isinf(pdf_value)) {
        printf("Error: Numerical instability in t_pdf\n");
        return NAN;
    }

    return pdf_value;
}

/**
 * @brief Computes the cumulative distribution function (CDF) of the t-distribution.
 *
 * Used in zerohyp_multi for p-value calculations in regression.
 *
 * @param t_value t-statistic value.
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return CDF value, or NAN if inputs are invalid.
 */
double t_cdf(double t_value, int degrees_freedom) {
    // Step 1: Validate degrees of freedom
    if (degrees_freedom <= 0) {
        printf("Error: Degrees of freedom must be a positive integer in t_cdf\n");
        return NAN;
    }

    // Step 2: Compute x = df / (df + t^2)
    double t_squared = t_value * t_value;
    double x = (double)degrees_freedom / (degrees_freedom + t_squared);
    if (isnan(x) || isinf(x)) {
        printf("Error: Numerical instability in t_cdf\n");
        return NAN;
    }

    // Step 3: Compute incomplete beta integral
    double incomplete_beta = 0.5 * ibeta(x, degrees_freedom / 2.0, 0.5);
    if (isnan(incomplete_beta)) {
        printf("Error: ibeta failed in t_cdf\n");
        return NAN;
    }

    // Step 4: Compute CDF: 1 - I for t >= 0, I for t < 0
    double cdf_value = 1.0 - incomplete_beta;
    if (t_value < 0.0) {
        cdf_value = incomplete_beta;
    }
    if (x == 0.0) {
        cdf_value = 0.5; // Handle edge case
    }

    return cdf_value;
}

/**
 * @brief Computes the inverse CDF of the t-distribution (approximation for large df).
 *
 * Uses Abramowitz and Stegun approximation for large degrees of freedom.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return t-critical value, or NAN if inputs are invalid.
 */
double t_inv_approx(double probability, int degrees_freedom) {
    // Step 1: Validate inputs
    if (degrees_freedom <= 0) {
        printf("Error: Degrees of freedom must be a positive integer in t_inv_approx\n");
        return NAN;
    }
    if (probability < 0.0 || probability > 1.0) {
        printf("Error: Probability must be between 0.0 and 1.0 in t_inv_approx\n");
        return NAN;
    }

    // Step 2: Handle boundary cases
    if (probability == 0.0) {
        return -XINFVAL;
    }
    if (probability == 1.0) {
        return XINFVAL;
    }

    // Step 3: Compute normal quantile
    double normal_quantile = normal_inv(probability, 0.0, 1.0);
    if (isnan(normal_quantile)) {
        printf("Error: normal_inv failed in t_inv_approx\n");
        return NAN;
    }

    // Step 4: Compute correction terms (Abramowitz and Stegun 26.7.5)
    double df = (double)degrees_freedom;
    double x = normal_quantile;
    double g1 = (pow(x, 3.0) + x) / (4.0 * df);
    double g2 = (5.0 * pow(x, 5.0) + 16.0 * pow(x, 3.0) + 3.0 * x) / (96.0 * pow(df, 2.0));
    double g3 = (3.0 * pow(x, 7.0) + 19.0 * pow(x, 5.0) + 17.0 * pow(x, 3.0) - 15.0 * x) / (384.0 * pow(df, 3.0));
    double g4 = (79.0 * pow(x, 9.0) + 776.0 * pow(x, 7.0) + 1482.0 * pow(x, 5.0) - 1920.0 * pow(x, 3.0) - 945.0 * x) / (92160.0 * pow(df, 4.0));

    // Step 5: Combine terms
    double t_critical = x + g1 + g2 + g3 + g4;
    if (isnan(t_critical) || isinf(t_critical)) {
        printf("Error: Numerical instability in t_inv_approx\n");
        return NAN;
    }

    return t_critical;
}

/**
 * @brief Computes the inverse CDF of the t-distribution.
 *
 * Uses exact computation for df <= 1000 and approximation for larger df.
 * Used in linreg_multi for confidence intervals.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return t-critical value, or NAN if inputs are invalid.
 */
double t_inv(double probability, int degrees_freedom) {
    // Step 1: Validate inputs
    if (degrees_freedom <= 0) {
        printf("Error: Degrees of freedom must be a positive integer in t_inv\n");
        return NAN;
    }
    if (probability < 0.0 || probability > 1.0) {
        printf("Error: Probability must be between 0.0 and 1.0 in t_inv\n");
        return NAN;
    }

    // Step 2: Handle boundary cases
    if (probability == 0.0) {
        return -XINFVAL;
    }
    if (probability == 1.0) {
        return XINFVAL;
    }

    // Step 3: Determine probability for beta function
    double q = 1.0 - probability;
    double p_min = (probability <= q) ? 2.0 * probability : 2.0 * q;

    // Step 4: Compute t-critical value
    double t_critical;
    if (degrees_freedom <= 1000) {
        // Exact computation using incomplete beta inverse
        double sign = (probability < 0.5) ? -1.0 : 1.0;
        double df_half = (double)degrees_freedom / 2.0;
        double beta_inv = betainv(p_min, df_half, 0.5);
        if (isnan(beta_inv)) {
            printf("Error: betainv failed in t_inv\n");
            return NAN;
        }
        double y = 1.0 - beta_inv;
        if (y == 0.0) {
            printf("Error: Division by zero in t_inv\n");
            return NAN;
        }
        t_critical = sign * sqrt((double)degrees_freedom * y / beta_inv);
    } else {
        // Use approximation for large df
        t_critical = t_inv_approx(probability, degrees_freedom);
        if (isnan(t_critical)) {
            printf("Error: t_inv_approx failed in t_inv\n");
            return NAN;
        }
    }

    return t_critical;
}

/**
 * @brief Computes the probability density function (PDF) of the F-distribution.
 *
 * Calculates the PDF for an F-distribution with num and den degrees of freedom.
 *
 * @param x Input value (non-negative).
 * @param df_numerator Numerator degrees of freedom (positive integer).
 * @param df_denominator Denominator degrees of freedom (positive integer).
 * @return PDF value, or NAN if inputs are invalid.
 */
double f_pdf(double x, int df_numerator, int df_denominator) {
    // Step 1: Validate inputs
    if (df_numerator <= 0 || df_denominator <= 0) {
        printf("Error: Degrees of freedom must be positive integers in f_pdf\n");
        return NAN;
    }
    if (x < 0.0) {
        printf("Error: F-distribution input must be non-negative in f_pdf\n");
        return NAN;
    }

    // Step 2: Compute constants
    double k1 = (double)df_numerator;
    double k2 = (double)df_denominator;
    double z = k2 + k1 * x;
    if (z == 0.0) {
        printf("Error: Division by zero in f_pdf\n");
        return NAN;
    }

    // Step 3: Compute PDF using beta function
    double y, pdf_value;
    if (k1 * x > k2) {
        y = (k2 * k1) / (z * z);
        pdf_value = y * betapdf(k2 / z, k2 / 2.0, k1 / 2.0);
    } else {
        y = (z * k1 - x * k1 * k1) / (z * z);
        pdf_value = y * betapdf(k1 * x / z, k1 / 2.0, k2 / 2.0);
    }

    if (isnan(pdf_value) || isinf(pdf_value)) {
        printf("Error: Numerical instability in f_pdf\n");
        return NAN;
    }

    return pdf_value;
}

/**
 * @brief Computes the cumulative distribution function (CDF) of the F-distribution.
 *
 * Used in linreg_multi for ANOVA p-value calculations.
 *
 * @param x Input value (non-negative).
 * @param df_numerator Numerator degrees of freedom (positive integer).
 * @param df_denominator Denominator degrees of freedom (positive integer).
 * @return CDF value, or NAN if inputs are invalid.
 */
double f_cdf(double x, int df_numerator, int df_denominator) {
    // Step 1: Validate inputs
    if (df_numerator <= 0 || df_denominator <= 0) {
        printf("Error: Degrees of freedom must be positive integers in f_cdf\n");
        return NAN;
    }
    if (x < 0.0) {
        printf("Error: F-distribution input must be non-negative in f_cdf\n");
        return NAN;
    }

    // Step 2: Compute constants
    double k1 = (double)df_numerator;
    double k2 = (double)df_denominator;
    double z = k2 + k1 * x;
    if (z == 0.0) {
        printf("Error: Division by zero in f_cdf\n");
        return NAN;
    }

    // Step 3: Compute CDF using incomplete beta function
    double cdf_value;
    if (k1 * x < k2) {
        double y = k1 * x / z;
        cdf_value = ibeta(y, k1 / 2.0, k2 / 2.0);
    } else {
        double y = k2 / z;
        cdf_value = ibetac(y, k2 / 2.0, k1 / 2.0);
    }

    if (isnan(cdf_value)) {
        printf("Error: ibeta/ibetac failed in f_cdf\n");
        return NAN;
    }

    return cdf_value;
}

/**
 * @brief Computes the inverse CDF of the F-distribution.
 *
 * Calculates the quantile for a given probability in an F-distribution.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param df_numerator Numerator degrees of freedom (positive integer).
 * @param df_denominator Denominator degrees of freedom (positive integer).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double f_inv(double probability, int df_numerator, int df_denominator) {
    // Step 1: Validate inputs
    if (probability < 0.0 || probability > 1.0) {
        printf("Error: Probability must be between 0.0 and 1.0 in f_inv\n");
        return NAN;
    }
    if (df_numerator <= 0 || df_denominator <= 0) {
        printf("Error: Degrees of freedom must be positive integers in f_inv\n");
        return NAN;
    }

    // Step 2: Handle boundary case
    if (probability == 1.0) {
        return XINFVAL;
    }

    // Step 3: Compute inverse CDF using beta inverse
    double k1 = (double)df_numerator;
    double k2 = (double)df_denominator;
    double beta_inv = betainv(probability, k1 / 2.0, k2 / 2.0);
    if (isnan(beta_inv)) {
        printf("Error: betainv failed in f_inv\n");
        return NAN;
    }
    double b = 1.0 - beta_inv;
    if (b == 0.0) {
        printf("Error: Division by zero in f_inv\n");
        return NAN;
    }

    // Step 4: Compute quantile
    double quantile = (k2 * beta_inv) / (k1 * b);
    if (isnan(quantile) || isinf(quantile)) {
        printf("Error: Numerical instability in f_inv\n");
        return NAN;
    }

    return quantile;
}

/**
 * @brief Computes the probability density function (PDF) of the gamma distribution.
 *
 * Calculates the PDF for a gamma distribution with shape k and scale theta.
 *
 * @param x Input value (non-negative).
 * @param shape Shape parameter (k > 0).
 * @param scale Scale parameter (theta > 0).
 * @return PDF value, or NAN if inputs are invalid.
 */
double gamma_pdf(double x, double shape, double scale) {
    // Step 1: Validate inputs
    if (shape <= 0.0 || scale <= 0.0) {
        printf("Error: Gamma distribution parameters must be positive in gamma_pdf\n");
        return NAN;
    }

    // Step 2: Compute normalized value
    double t = x / scale;
    if (t < 0.0) {
        return 0.0; // PDF is zero for negative x
    }
    if (t == 0.0 && shape == 1.0) {
        return 1.0 / scale; // Special case
    }
    if (t == 0.0) {
        printf("Error: Undefined PDF for x=0 and shape!=1 in gamma_pdf\n");
        return NAN;
    }

    // Step 3: Compute PDF: exp((k-1)*log(t) - t - gamma_log(k)) / theta
    double exponent = (shape - 1.0) * log(t) - t - gamma_log(shape);
    double pdf_value = exp(exponent) / scale;
    if (isnan(pdf_value) || isinf(pdf_value)) {
        printf("Error: Numerical instability in gamma_pdf\n");
        return NAN;
    }

    return pdf_value;
}

/**
 * @brief Computes the cumulative distribution function (CDF) of the gamma distribution.
 *
 * Calculates the CDF for a gamma distribution with shape k and scale theta.
 *
 * @param x Input value (non-negative).
 * @param shape Shape parameter (k > 0).
 * @param scale Scale parameter (theta > 0).
 * @return CDF value, or NAN if inputs are invalid.
 */
double gamma_cdf(double x, double shape, double scale) {
    // Step 1: Validate inputs
    if (shape <= 0.0 || scale <= 0.0) {
        printf("Error: Gamma distribution parameters must be positive in gamma_cdf\n");
        return NAN;
    }
    if (x < 0.0) {
        return 0.0; // CDF is zero for negative x
    }

    // Step 2: Compute normalized value
    double t = x / scale;

    // Step 3: Compute CDF using pgamma
    double cdf_value = pgamma(t, shape);
    if (isnan(cdf_value)) {
        printf("Error: pgamma failed in gamma_cdf\n");
        return NAN;
    }

    return cdf_value;
}

/**
 * @brief Computes the inverse CDF of the gamma distribution.
 *
 * Uses Newton's method to find the quantile for a given probability.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param shape Shape parameter (k > 0).
 * @param scale Scale parameter (theta > 0).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double gamma_inv(double probability, double shape, double scale) {
    // Step 1: Validate inputs
    if (probability < 0.0 || probability > 1.0) {
        printf("Error: Probability must be between 0.0 and 1.0 in gamma_inv\n");
        return NAN;
    }
    if (shape <= 0.0 || scale <= 0.0) {
        printf("Error: Gamma distribution parameters must be positive in gamma_inv\n");
        return NAN;
    }

    // Step 2: Handle boundary cases
    if (probability == 0.0) {
        return 0.0;
    }
    if (probability == 1.0) {
        return XINFVAL;
    }

    // Step 3: Initialize using lognormal approximation
    double variance = log(1.0 + shape) - log(shape);
    double mean = log(shape) - 0.5 * variance;
    double initial_guess = exp(mean - sqrt(2.0 * variance) * erfcinv(2.0 * probability));
    if (isnan(initial_guess)) {
        printf("Error: erfcinv failed in gamma_inv\n");
        return NAN;
    }

    // Step 4: Newton's method iteration
    double current_guess = initial_guess;
    double delta;
    double convergence_threshold = eps(XNINFVAL);
    int max_iterations = 1000;
    int iteration = 0;

    do {
        delta = (gamma_cdf(current_guess, shape, 1.0) - probability) / r8_max(gamma_pdf(current_guess, shape, 1.0), XNINFVAL);
        if (isnan(delta)) {
            printf("Error: Numerical instability in gamma_inv iteration\n");
            return NAN;
        }
        iteration++;
        double next_guess = current_guess - delta;
        if (next_guess <= 0.0) {
            next_guess = current_guess / 10.0; // Prevent negative values
            delta = current_guess - next_guess;
        }
        current_guess = next_guess;
    } while (fabs(delta) > convergence_threshold * current_guess && iteration < max_iterations);

    if (iteration >= max_iterations) {
        printf("Error: Newton's method did not converge in gamma_inv\n");
        return NAN;
    }

    // Step 5: Scale result
    double quantile = current_guess * scale;
    return quantile;
}

/**
 * @brief Computes the probability density function (PDF) of the chi-squared distribution.
 *
 * Calculates the PDF using the gamma distribution with shape df/2 and scale 2.
 *
 * @param x Input value (non-negative).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return PDF value, or NAN if inputs are invalid.
 */
double chi_pdf(double x, int degrees_freedom) {
    // Step 1: Validate inputs
    if (degrees_freedom <= 0) {
        printf("Error: Degrees of freedom must be a positive integer in chi_pdf\n");
        return NAN;
    }
    if (x < 0.0) {
        printf("Error: Chi-squared input must be non-negative in chi_pdf\n");
        return NAN;
    }

    // Step 2: Compute PDF using gamma_pdf
    double shape = (double)degrees_freedom / 2.0;
    double pdf_value = gamma_pdf(x, shape, 2.0);
    if (isnan(pdf_value)) {
        printf("Error: gamma_pdf failed in chi_pdf\n");
        return NAN;
    }

    return pdf_value;
}

/**
 * @brief Computes the cumulative distribution function (CDF) of the chi-squared distribution.
 *
 * Calculates the CDF using the gamma distribution with shape df/2 and scale 2.
 *
 * @param x Input value (non-negative).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return CDF value, or NAN if inputs are invalid.
 */
double chi_cdf(double x, int degrees_freedom) {
    // Step 1: Validate inputs
    if (degrees_freedom <= 0) {
        printf("Error: Degrees of freedom must be a positive integer in chi_cdf\n");
        return NAN;
    }
    if (x < 0.0) {
        printf("Error: Chi-squared input must be non-negative in chi_cdf\n");
        return NAN;
    }

    // Step 2: Compute CDF using gamma_cdf
    double shape = (double)degrees_freedom / 2.0;
    double cdf_value = pgamma(x / 2.0, shape);
    if (isnan(cdf_value)) {
        printf("Error: pgamma failed in chi_cdf\n");
        return NAN;
    }

    return cdf_value;
}

/**
 * @brief Computes the inverse CDF of the chi-squared distribution.
 *
 * Used in linreg_multi for variance confidence intervals.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param degrees_freedom Degrees of freedom (positive integer).
 * @return Quantile value, or NAN if inputs are invalid.
 */
double chi_inv(double probability, int degrees_freedom) {
    // Step 1: Validate inputs
    if (degrees_freedom <= 0) {
        printf("Error: Degrees of freedom must be a positive integer in chi_inv\n");
        return NAN;
    }
    if (probability < 0.0 || probability > 1.0) {
        printf("Error: Probability must be between 0.0 and 1.0 in chi_inv\n");
        return NAN;
    }

    // Step 2: Compute quantile using gamma_inv
    double shape = (double)degrees_freedom / 2.0;
    double quantile = gamma_inv(probability, shape, 2.0);
    if (isnan(quantile)) {
        printf("Error: gamma_inv failed in chi_inv\n");
        return NAN;
    }

    return quantile;
}