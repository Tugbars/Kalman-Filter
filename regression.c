// SPDX-License-Identifier: BSD-3-Clause
#include "regression.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Initializes a regression object for linear regression analysis.
 *
 * Allocates memory for a regression object, sets default values for parameters,
 * and initializes statistical metrics. The object is used to store regression results,
 * including coefficients, variance, and goodness-of-fit metrics.
 *
 * @param num_observations Number of observation samples (N).
 * @param num_predictors Total number of variables (p, including intercept if present).
 * @return Pointer to the initialized regression object.
 */
reg_object reg_init(int num_observations, int num_predictors) {
    // Step 1: Validate input
    if (num_predictors < 0) {
        printf("The base case requires p >= 1 (e.g., y = b0 + u or y = b1*x + u).\n");
        printf("p = Number of independent variables (p-1) + dependent variable (1).\n");
        exit(1);
    }

    // Step 2: Allocate memory for regression object and coefficients
    reg_object reg_obj = (reg_object)malloc(sizeof(struct reg_set) + sizeof(bparam) * (num_predictors + 1));
    reg_obj->N = num_observations; // Number of observations
    reg_obj->p = num_predictors; // Number of predictors (including intercept)
    reg_obj->alpha = 0.05; // Default confidence level (95%)
    reg_obj->sigma = 0.0; // Residual variance
    reg_obj->sigma_lower = 0.0; // Lower confidence bound for sigma
    reg_obj->sigma_upper = 0.0; // Upper confidence bound for sigma
    reg_obj->r2 = reg_obj->R2[0] = 0.0; // R-squared
    reg_obj->r2adj = reg_obj->R2[1] = 0.0; // Adjusted R-squared
    reg_obj->df = 0; // Degrees of freedom
    reg_obj->intercept = (num_predictors == 0) ? 0 : 1; // Include intercept by default unless p=0
    strcpy(reg_obj->lls, "qr"); // Default least squares method (QR decomposition)

    // Step 3: Initialize ANOVA and model fit metrics
    reg_obj->TSS = 0.0; // Total Sum of Squares
    reg_obj->ESS = 0.0; // Explained Sum of Squares
    reg_obj->RSS = 0.0; // Residual Sum of Squares
    reg_obj->df_ESS = 0; // Degrees of freedom for ESS
    reg_obj->df_RSS = 0; // Degrees of freedom for RSS
    reg_obj->FStat = 0.0; // F-statistic
    reg_obj->PVal = 0.0; // P-value for F-statistic
    reg_obj->loglik = 0.0; // Log-likelihood
    reg_obj->aic = 0.0; // Akaike Information Criterion
    reg_obj->bic = 0.0; // Bayesian Information Criterion
    reg_obj->aicc = 0.0; // Corrected AIC

    // Step 4: Initialize regression coefficients
    for (int i = 0; i < num_predictors + 1; ++i) {
        (reg_obj->beta + i)->value = 0.0; // Coefficient value
        (reg_obj->beta + i)->lower = 0.0; // Lower confidence bound
        (reg_obj->beta + i)->upper = 0.0; // Upper confidence bound
        (reg_obj->beta + i)->stdErr = 0.0; // Standard error
    }

    return reg_obj; // Return initialized object
}

/**
 * @brief Performs multiple linear regression using matrix operations.
 *
 * Computes regression coefficients, residual variance, variance-covariance matrix,
 * R-squared metrics, residuals, confidence intervals, and ANOVA statistics for a
 * multiple linear regression model. Supports models with or without an intercept.
 *
 * @param num_predictors Number of predictors (p, including intercept if present).
 * @param predictors Matrix of independent variables (p-1 columns, N rows; or p if no intercept).
 * @param response Vector of dependent variable values (length N).
 * @param num_observations Number of observations (N).
 * @param coeffs Output regression coefficients (length p).
 * @param residual_variance Output variance of residuals.
 * @param var_covar Output variance-covariance matrix (p x p).
 * @param r_squared Output array for R-squared and adjusted R-squared.
 * @param residuals Output residuals (length N).
 * @param alpha Confidence level for intervals (e.g., 0.05 for 95%).
 * @param anova_stats Output ANOVA statistics (TSS, ESS, RSS, etc.).
 * @param ci_lower Output lower confidence bounds for coefficients and variance.
 * @param ci_upper Output upper confidence bounds for coefficients and variance.
 * @param rank Output rank of the regression matrix.
 * @param ls_method Least squares method ("qr", "normal", or "svd").
 * @param intercept Flag to include intercept (1 = include, 0 = exclude).
 */
void linreg_multi(int num_predictors, double *predictors, double *response, int num_observations, double *coeffs, double *residual_variance,
    double *var_covar, double *r_squared, double *residuals, double alpha, double *anova_stats, double *ci_lower, double *ci_upper, int *rank, char *ls_method, int intercept) {
    // Step 1: Allocate temporary arrays
    double *x_matrix = (double*)malloc(sizeof(double) * num_predictors * num_observations); // X matrix (N x p)
    double *x_transpose = (double*)malloc(sizeof(double) * num_predictors * num_observations); // X' matrix (p x N)
    double *x_beta = (double*)malloc(sizeof(double) * num_observations); // X * beta (N x 1)
    double *xxt_matrix = (double*)malloc(sizeof(double) * num_predictors * num_predictors); // X'X matrix
    double *xxt_copy = (double*)malloc(sizeof(double) * num_predictors * num_predictors); // Copy of X'X for inversion
    double *x_transpose_y = (double*)malloc(sizeof(double) * num_predictors); // X'y vector
    int *ipiv = (int*)malloc(sizeof(int) * num_predictors); // Pivot indices for LU decomposition
    double *temp_scalar = (double*)calloc(1, sizeof(double)); // Temporary scalar
    double *temp_vector = (double*)malloc(sizeof(double) * num_predictors); // Temporary vector
    double *std_errors = (double*)malloc(sizeof(double) * num_predictors); // Standard errors of coefficients
    double *y_transpose_y = (double*)calloc(1, sizeof(double)); // y'y scalar

    // Step 2: Initialize X' matrix (transpose first due to column-major order)
    if (intercept == 1) {
        for (int i = 0; i < num_observations; ++i) {
            x_transpose[i] = 1.0; // First column is ones for intercept
        }
        for (int i = num_observations; i < num_predictors * num_observations; ++i) {
            x_transpose[i] = predictors[i - num_observations]; // Copy predictors
        }
    } else {
        for (int i = 0; i < num_predictors * num_observations; ++i) {
            x_transpose[i] = predictors[i]; // No intercept, copy directly
        }
    }

    // Step 3: Compute X matrix (transpose of X')
    mtranspose(x_transpose, num_predictors, num_observations, x_matrix);

    // Step 4: Compute X'X and X'y
    mmult(x_transpose, x_matrix, xxt_matrix, num_predictors, num_observations, num_predictors); // X'X
    memcpy(xxt_copy, xxt_matrix, sizeof(double) * num_predictors * num_predictors); // Copy for inversion
    mmult(x_transpose, response, x_transpose_y, num_predictors, num_observations, 1); // X'y

    // Step 5: Solve for coefficients using specified least squares method
    if (num_predictors > 0) {
        if (!strcmp(ls_method, "qr")) {
            *rank = lls_qr(xxt_matrix, x_transpose_y, num_predictors, num_predictors, coeffs); // QR decomposition
        } else if (!strcmp(ls_method, "normal")) {
            lls_normal(xxt_matrix, x_transpose_y, num_predictors, num_predictors, coeffs); // Normal equations
            printf("Warning: 'normal' method does not calculate rank. Use 'qr' instead.\n");
        } else if (!strcmp(ls_method, "svd")) {
            lls_svd2(xxt_matrix, x_transpose_y, num_predictors, num_predictors, coeffs); // SVD
            printf("Warning: 'svd' method does not calculate rank. Use 'qr' instead.\n");
        } else {
            printf("Invalid least squares method. Use 'qr', 'normal', or 'svd'.\n");
            exit(-1);
        }
    }

    // Step 6: Compute residual variance
    int degrees_freedom = num_observations - num_predictors; // N - p
    mmult(response, response, y_transpose_y, 1, num_observations, 1); // y'y
    mmult(x_transpose, response, temp_vector, num_predictors, num_observations, 1); // X'y
    mmult(coeffs, temp_vector, temp_scalar, 1, num_predictors, 1); // beta' * (X'y)
    *residual_variance = (*y_transpose_y - *temp_scalar) / (double)degrees_freedom; // sigma^2 = (y'y - beta'X'y) / (N-p)
    double variance = *residual_variance;

    // Step 7: Compute variance-covariance matrix
    ludecomp(xxt_copy, num_predictors, ipiv); // LU decomposition of X'X
    minverse(xxt_copy, num_predictors, ipiv, var_covar); // Inverse of X'X
    for (int i = 0; i < num_predictors * num_predictors; ++i) {
        var_covar[i] *= variance; // Scale by residual variance
    }

    // Step 8: Compute R-squared and adjusted R-squared
    double sum_response = 0.0;
    for (int i = 0; i < num_observations; ++i) {
        sum_response += response[i];
    }
    double mean_response = sum_response / num_observations; // Mean of y
    double mean_response_squared = mean_response * mean_response;
    r_squared[0] = (*temp_scalar - num_observations * mean_response_squared) / (*y_transpose_y - num_observations * mean_response_squared); // R^2
    double df_intercept = intercept == 1 ? 1.0 : 0.0; // Adjust for intercept
    r_squared[1] = 1.0 - (1.0 - r_squared[0]) * ((double)num_observations - df_intercept) / ((double)degrees_freedom); // Adjusted R^2

    // Step 9: Compute confidence intervals for coefficients
    double alpha_half = alpha / 2.0;
    for (int i = 0; i < num_predictors; ++i) {
        std_errors[i] = sqrt(var_covar[i * (num_predictors + 1)]); // Standard error from diagonal
        double t_critical = tinv(alpha_half, degrees_freedom); // t-value for confidence interval
        double interval = t_critical * std_errors[i];
        ci_lower[i] = coeffs[i] - interval; // Lower bound
        ci_upper[i] = coeffs[i] + interval; // Upper bound
    }

    // Step 10: Compute confidence interval for residual variance
    double chi_lower = chiinv(alpha_half, degrees_freedom); // Chi-squared lower critical value
    double chi_upper = chiinv(1.0 - alpha_half, degrees_freedom); // Chi-squared upper critical value
    ci_lower[num_predictors] = (double)degrees_freedom * variance / chi_upper; // Lower bound for sigma^2
    ci_upper[num_predictors] = (double)degrees_freedom * variance / chi_lower; // Upper bound for sigma^2

    // Step 11: Compute residuals (y - X*beta)
    mmult(x_matrix, coeffs, x_beta, num_observations, num_predictors, 1); // X*beta
    for (int i = 0; i < num_observations; ++i) {
        residuals[i] = response[i] - x_beta[i]; // y - X*beta
    }

    // Step 12: Compute ANOVA statistics
    double sum_squares_model = 0.0;
    double sum_squares_residual = 0.0;
    for (int i = 0; i < num_observations; ++i) {
        sum_squares_model += (x_beta[i] * x_beta[i]); // Sum of squares for model
        sum_squares_residual += (residuals[i] * residuals[i]); // Sum of squares for residuals
    }

    if (intercept == 1) {
        if (num_predictors == 1) {
            anova_stats[1] = r_squared[0] * (*y_transpose_y - num_observations * mean_response_squared); // ESS
            anova_stats[2] = (1.0 - r_squared[0]) * (*y_transpose_y - num_observations * mean_response_squared); // RSS
            anova_stats[0] = anova_stats[1] + anova_stats[2]; // TSS
            anova_stats[3] = (double)num_predictors - df_intercept; // df for ESS
            anova_stats[4] = (double)degrees_freedom; // df for RSS
        } else if (num_predictors > 1) {
            anova_stats[1] = r_squared[0] * (*y_transpose_y - num_observations * mean_response_squared);
            anova_stats[2] = (1.0 - r_squared[0]) * (*y_transpose_y - num_observations * mean_response_squared);
            anova_stats[0] = anova_stats[1] + anova_stats[2];
            anova_stats[3] = (double)num_predictors - df_intercept;
            anova_stats[4] = (double)degrees_freedom;
            anova_stats[5] = (r_squared[0] / anova_stats[3]) / ((1.0 - r_squared[0]) / anova_stats[4]); // F-statistic
            anova_stats[6] = 1.0 - fcdf(anova_stats[5], num_predictors - 1, degrees_freedom); // P-value
        }
    } else {
        r_squared[0] = sum_squares_model / (sum_squares_model + sum_squares_residual); // R^2 without intercept
        r_squared[1] = 1.0 - (1.0 - r_squared[0]) * ((double)num_observations - df_intercept) / ((double)degrees_freedom); // Adjusted R^2
        if (num_predictors == 1) {
            anova_stats[1] = sum_squares_model;
            anova_stats[2] = sum_squares_residual;
            anova_stats[0] = anova_stats[1] + anova_stats[2];
            anova_stats[3] = (double)num_predictors;
            anova_stats[4] = (double)degrees_freedom;
            anova_stats[5] = (r_squared[0] / anova_stats[3]) / ((1.0 - r_squared[0]) / anova_stats[4]);
            anova_stats[6] = 1.0 - fcdf(anova_stats[5], num_predictors, degrees_freedom);
        } else if (num_predictors > 1) {
            anova_stats[1] = sum_squares_model;
            anova_stats[2] = sum_squares_residual;
            anova_stats[0] = anova_stats[1] + anova_stats[2];
            anova_stats[3] = (double)num_predictors;
            anova_stats[4] = (double)degrees_freedom;
            anova_stats[5] = (r_squared[0] / anova_stats[3]) / ((1.0 - r_squared[0]) / anova_stats[4]);
            anova_stats[6] = 1.0 - fcdf(anova_stats[5], num_predictors - 1, degrees_freedom);
        } else if (num_predictors == 0) {
            anova_stats[1] = sum_squares_model;
            anova_stats[2] = sum_squares_residual;
            anova_stats[0] = anova_stats[1] + anova_stats[2];
            anova_stats[3] = (double)num_predictors;
            anova_stats[4] = (double)degrees_freedom;
            anova_stats[5] = NAN;
            anova_stats[6] = NAN;
        }
    }

    // Step 13: Free allocated memory
    free(ipiv);
    free(x_matrix);
    free(x_transpose);
    free(xxt_matrix);
    free(xxt_copy);
    free(x_transpose_y);
    free(temp_scalar);
    free(temp_vector);
    free(std_errors);
    free(x_beta);
    free(y_transpose_y);
}

/**
 * @brief Performs linear regression and updates the regression object with results.
 *
 * Executes multiple linear regression using the specified least squares method (QR, normal, or SVD)
 * to estimate coefficients, variance-covariance matrix, residuals, and statistical metrics
 * (e.g., R-squared, AIC, BIC). Handles models with or without an intercept and computes
 * confidence intervals for parameters and residual variance.
 *
 * @param reg_obj Regression object containing model specifications and storage for results.
 * @param predictors Matrix of independent variables (p-1 columns, N rows; NULL if intercept-only).
 * @param response Vector of dependent variable values (length N).
 * @param residuals Output vector for residuals (length N).
 * @param var_covar Output variance-covariance matrix of regression coefficients (p x p).
 * @param alpha Confidence level for intervals (e.g., 0.05 for 95% confidence).
 */
void regress(reg_object reg_obj, double *predictors, double *response, double *residuals, double *var_covar, double alpha) {
    // Step 1: Initialize variables and allocate memory
    int num_predictors = reg_obj->p; // Number of predictors (including intercept if present)
    reg_obj->alpha = alpha; // Store confidence level
    double *anova_stats = (double*)malloc(sizeof(double) * 7); // ANOVA statistics
    double *coeffs = num_predictors == 0 ? NULL : (double*)malloc(sizeof(double) * num_predictors); // Regression coefficients
    double *ci_lower = (double*)malloc(sizeof(double) * (num_predictors + 1)); // Lower confidence bounds
    double *ci_upper = (double*)malloc(sizeof(double) * (num_predictors + 1)); // Upper confidence bounds
    double *residual_variance = (double*)malloc(sizeof(double) * 1); // Residual variance
    double pi = 3.141592653589793; // Constant for log-likelihood
    double sum_squared_residuals = 0.0; // Sum of squared residuals
    int degrees_freedom_model; // Degrees of freedom for model
    double sample_size_half = (double)(reg_obj->N) / 2.0; // N/2 for log-likelihood
    int i; // Loop index

    // Step 2: Perform multiple linear regression
    linreg_multi(
        num_predictors, predictors, response, reg_obj->N, coeffs, residual_variance,
        var_covar, reg_obj->R2, residuals, alpha, anova_stats, ci_lower, ci_upper,
        &reg_obj->rank, reg_obj->lls, reg_obj->intercept
    );

    // Step 3: Update degrees of freedom
    reg_obj->df = reg_obj->N - num_predictors; // Total degrees of freedom (N - p)

    // Step 4: Store variance and confidence intervals
    reg_obj->sigma = residual_variance[0]; // Residual variance
    reg_obj->sigma_lower = ci_lower[num_predictors]; // Lower bound for sigma^2
    reg_obj->sigma_upper = ci_upper[num_predictors]; // Upper bound for sigma^2

    // Step 5: Store ANOVA statistics
    reg_obj->TSS = anova_stats[0]; // Total Sum of Squares
    reg_obj->ESS = anova_stats[1]; // Explained Sum of Squares
    reg_obj->RSS = anova_stats[2]; // Residual Sum of Squares
    reg_obj->df_ESS = (int)anova_stats[3]; // Degrees of freedom for ESS
    reg_obj->df_RSS = (int)anova_stats[4]; // Degrees of freedom for RSS
    reg_obj->FStat = anova_stats[5]; // F-statistic
    reg_obj->PVal = anova_stats[6]; // P-value for F-statistic
    reg_obj->r2 = reg_obj->R2[0]; // R-squared
    reg_obj->r2adj = reg_obj->R2[1]; // Adjusted R-squared

    // Step 6: Update regression coefficients
    for (i = 0; i < num_predictors; ++i) {
        (reg_obj->beta + i)->value = coeffs[i]; // Coefficient value
        (reg_obj->beta + i)->lower = ci_lower[i]; // Lower confidence bound
        (reg_obj->beta + i)->upper = ci_upper[i]; // Upper confidence bound
        (reg_obj->beta + i)->stdErr = sqrt(var_covar[(num_predictors + 1) * i]); // Standard error
    }

    // Step 7: Compute log-likelihood
    for (i = 0; i < reg_obj->N; ++i) {
        sum_squared_residuals += (residuals[i] * residuals[i]); // Sum of squared residuals
    }
    reg_obj->loglik = -sample_size_half * log(2 * pi) - sample_size_half * log(sum_squared_residuals / (double)reg_obj->N) - sample_size_half;

    // Step 8: Compute information criteria
    int intercept_adjust = reg_obj->intercept == 1 ? 1 : 0; // Adjust for intercept
    degrees_freedom_model = reg_obj->intercept == 1 ? num_predictors - 1 : num_predictors; // Model parameters
    reg_obj->aic = -2.0 * reg_obj->loglik + 2.0 * (double)(intercept_adjust + degrees_freedom_model); // AIC
    reg_obj->bic = -2.0 * reg_obj->loglik + log((double)reg_obj->N) * (double)(intercept_adjust + degrees_freedom_model); // BIC
    reg_obj->aicc = reg_obj->aic + 2.0 * degrees_freedom_model * ((double)reg_obj->N / ((double)reg_obj->N - degrees_freedom_model - 1.0) - 1.0); // Corrected AIC

    // Step 9: Free allocated memory
    free(anova_stats);
    free(coeffs);
    free(ci_lower);
    free(ci_upper);
    free(residual_variance);
}

/**
 * @brief Performs hypothesis tests for regression coefficients.
 *
 * Computes t-statistics and p-values for testing the null hypothesis that each
 * regression coefficient is zero, using the variance-covariance matrix.
 *
 * @param num_observations Number of observations (N).
 * @param coeffs Regression coefficients (length p).
 * @param num_predictors Number of predictors (p).
 * @param var_covar Variance-covariance matrix (p x p).
 * @param t_stats Output t-statistics for each coefficient.
 * @param p_values Output p-values for each coefficient.
 */
void zerohyp_multi(int num_observations, double *coeffs, int num_predictors, double *var_covar, double *t_stats, double *p_values) {
    // Step 1: Compute degrees of freedom
    int degrees_freedom = num_observations - num_predictors; // N - p

    // Step 2: Calculate t-statistics and p-values
    for (int i = 0; i < num_predictors; ++i) {
        t_stats[i] = fabs(coeffs[i]) / sqrt(var_covar[(num_predictors + 1) * i]); // t = |beta| / SE
        p_values[i] = 1.0 - tcdf(fabs(t_stats[i]), degrees_freedom); // One-tailed p-value
    }
}

/**
 * @brief Sets the intercept flag for the regression object.
 *
 * Configures whether the regression model includes an intercept term.
 * Ensures valid input and adjusts the intercept flag accordingly.
 *
 * @param reg_obj Regression object to modify.
 * @param intercept Flag (1 = include intercept, 0 = exclude).
 */
void setIntercept(reg_object reg_obj, int intercept) {
    // Step 1: Validate input
    if (intercept == 1 && reg_obj->p > 0) {
        reg_obj->intercept = 1; // Include intercept
    } else if (intercept == 0) {
        reg_obj->intercept = 0; // Exclude intercept
    } else {
        printf("Invalid intercept value. Use 0 or 1. Intercept is 0 when p == 0.\n");
        exit(-1);
    }
}

/**
 * @brief Prints a summary of regression results.
 *
 * Displays the regression coefficients, their standard errors, residual variance,
 * and R-squared metrics in a formatted table.
 *
 * @param reg_obj Regression object containing results.
 */
void summary(reg_object reg_obj) {
    // Step 1: Print header
    printf("\n");
    printf("%-25s \n", "Regression Summary");
    printf("%-25s%-20s%-20s \n", "Coefficients", "Value", "Standard Error");

    // Step 2: Print coefficients and standard errors
    for (int i = 0; i < reg_obj->p; ++i) {
        printf("B%-25d%-20lf%-20g \n", i, (reg_obj->beta + i)->value, (reg_obj->beta + i)->stdErr);
    }

    // Step 3: Print residual variance and R-squared
    printf("\n");
    printf("Residual Variance = %lf \n", reg_obj->sigma);
    printf("R-Squared = %lf , Adjusted R-Squared = %lf \n", reg_obj->R2[0], reg_obj->R2[1]);
}

/**
 * @brief Prints ANOVA table for the regression model.
 *
 * Displays the analysis of variance (ANOVA) table, including degrees of freedom,
 * sum of squares, mean sum of squares, F-statistic, and p-value.
 *
 * @param reg_obj Regression object containing ANOVA statistics.
 */
void anova(reg_object reg_obj) {
    // Step 1: Print header
    printf("\n");
    printf("ANOVA : \n");
    printf("%-25s%-20s%-20s%-20s \n", "Source of Variation", "df", "SS", "MSS");

    // Step 2: Print regression, residual, and total statistics
    printf("%-25s%-20d%-20lf%-20lf \n", "Due to Regression",
           reg_obj->df_ESS, reg_obj->ESS, reg_obj->ESS / ((double)reg_obj->df_ESS));
    printf("%-25s%-20d%-20lf%-20lf \n", "Due to Residual",
           reg_obj->df_RSS, reg_obj->RSS, reg_obj->RSS / ((double)reg_obj->df_RSS));
    printf("%-25s%-20d%-20lf \n", "Total",
           reg_obj->df_RSS + reg_obj->df_ESS, reg_obj->TSS);

    // Step 3: Print F-statistic and p-value
    printf("\n F Statistics = %g \n", reg_obj->FStat);
    printf("\n P Value (F) = %g \n", reg_obj->PVal);
}

/**
 * @brief Prints confidence intervals for regression parameters and residual variance.
 *
 * Displays a table of regression coefficients, their values, and their confidence
 * intervals, along with the confidence interval for the residual variance.
 *
 * @param reg_obj Regression object containing confidence intervals.
 */
void confint(reg_object reg_obj) {
    // Step 1: Print header with confidence level
    printf("\n");
    printf("%-10lf%% Confidence Interval For Parameters And Residual Variance : \n", (1.0 - reg_obj->alpha) * 100);
    printf("%-25s%-20s%-20s%-20s \n", "Parameters", "Value", "Lower Limit", "Upper Limit");

    // Step 2: Print confidence intervals for coefficients
    for (int i = 0; i < reg_obj->p; ++i) {
        printf("B%-25d%-20lf%-20lf%-20lf \n",
               i, (reg_obj->beta + i)->value, (reg_obj->beta + i)->lower, (reg_obj->beta + i)->upper);
    }

    // Step 3: Print confidence interval for residual variance
    printf("%-25s%-20lf%-20lf%-20lf \n", "Residual Variance",
           reg_obj->sigma, reg_obj->sigma_lower, reg_obj->sigma_upper);
}

/**
 * @brief Predicts the response value for a given input and computes prediction variances.
 *
 * Computes the predicted response for a given set of predictor values, along with
 * the variance of the mean prediction and the variance of an individual prediction.
 *
 * @param reg_obj Regression object containing model parameters.
 * @param input_predictors Input predictor values (length p-1 if intercept, p otherwise).
 * @param var_covar Variance-covariance matrix of regression coefficients (p x p).
 * @param prediction_vars Output array for mean and individual prediction variances.
 * @return Predicted response value.
 */
double fitted(reg_object reg_obj, double *input_predictors, double *var_covar, double *prediction_vars) {
    // Step 1: Allocate temporary arrays
    int num_predictors = reg_obj->p;
    double *predicted_value = (double*)malloc(sizeof(double) * 1); // Predicted y
    double *x_input = (double*)malloc(sizeof(double) * num_predictors); // Input vector
    double *coeffs = (double*)malloc(sizeof(double) * num_predictors); // Coefficients
    double *temp_vector = (double*)malloc(sizeof(double) * num_predictors); // Temporary vector
    double output_value; // Final predicted value

    // Step 2: Prepare input vector and coefficients
    if (reg_obj->intercept == 1) {
        x_input[0] = 1.0; // Intercept term
        coeffs[0] = (reg_obj->beta)->value;
        for (int i = 1; i < num_predictors; ++i) {
            x_input[i] = input_predictors[i - 1]; // Copy predictors
            coeffs[i] = (reg_obj->beta + i)->value;
        }
    } else {
        for (int i = 0; i < num_predictors; ++i) {
            x_input[i] = input_predictors[i];
            coeffs[i] = (reg_obj->beta + i)->value;
        }
    }

    // Step 3: Compute predicted value
    mmult(x_input, coeffs, predicted_value, 1, num_predictors, 1); // y = x * beta
    output_value = *predicted_value;

    // Step 4: Compute variance of mean prediction
    mmult(x_input, var_covar, temp_vector, 1, num_predictors, num_predictors); // x * var_covar
    mmult(temp_vector, x_input, predicted_value, 1, num_predictors, 1); // x * var_covar * x'
    prediction_vars[0] = *predicted_value; // Variance of mean prediction

    // Step 5: Compute variance of individual prediction
    prediction_vars[1] = reg_obj->sigma + prediction_vars[0]; // sigma^2 + var(mean)

    // Step 6: Free allocated memory
    free(temp_vector);
    free(predicted_value);
    free(x_input);
    free(coeffs);

    return output_value; // Return predicted value
}

/**
 * @brief Frees memory allocated for the regression object.
 *
 * Releases the memory used by the regression object to prevent memory leaks.
 *
 * @param reg_obj Regression object to free.
 */
void free_reg(reg_object reg_obj) {
    free(reg_obj);
}