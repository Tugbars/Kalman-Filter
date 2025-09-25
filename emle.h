/*
 * emle_refactored.h
 *
 * Header file for refactored Kalman filter functions from emle.c and their direct
 * dependencies from talg.c and matrix.c, including structures from emle.h.
 */

#ifndef EMLE_REFACTORED_H_
#define EMLE_REFACTORED_H_

#include "initest.h"
#include "matrix.h"
#include "talg.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure for ARIMA model with CSS optimization.
 */
typedef struct alik_css_set* alik_css_object;

struct alik_css_set {
    int p;        // Size of phi (AR coefficients)
    int d;        // Number of times the series is to be differenced
    int q;        // Size of theta (MA coefficients)
    int r;        // max(p, q+1)
    int pq;       // p + q
    int length;   // Length of the original time series
    int N;        // Length of time series after differencing
    int M;        // 1 if mean needs to be calculated, else 0
    double eps;   // Convergence threshold
    double mean;  // Estimated mean
    double ssq;   // Sum of squares value
    double loglik;// Log-likelihood
    double phi[100];   // AR coefficients
    double theta[100]; // MA coefficients
    double x[1];       // Flexible array for time series data
};

/**
 * @brief Structure for ARIMA model with exact likelihood.
 */
typedef struct alik_set* alik_object;

struct alik_set {
    int p;        // Size of phi (AR coefficients)
    int d;        // Number of times the series is to be differenced
    int q;        // Size of theta (MA coefficients)
    int r;        // max(p, q+1)
    int pq;       // p + q
    int length;   // Length of the original time series
    int N;        // Length of time series after differencing
    int M;        // 1 if mean needs to be calculated, else 0
    double eps;   // Convergence threshold
    double mean;  // Estimated mean
    double ssq;   // Sum of squares value
    double loglik;// Log-likelihood
    double phi[100];   // AR coefficients
    double theta[100]; // MA coefficients
    double x[1];       // Flexible array for time series data
};

/**
 * @brief Structure for seasonal ARIMA model with CSS optimization.
 */
typedef struct alik_css_seas_set* alik_css_seas_object;

struct alik_css_seas_set {
    int p;        // Size of phi (non-seasonal AR coefficients)
    int d;        // Number of times the series is to be differenced
    int q;        // Size of theta (non-seasonal MA coefficients)
    int s;        // Frequency of seasonal components
    int P;        // Size of seasonal phi (seasonal AR coefficients)
    int D;        // Number of times the seasonal series is to be differenced
    int Q;        // Size of seasonal theta (seasonal MA coefficients)
    int r;        // max(p + s*P, q + s*Q + 1)
    int pq;       // p + q + s*P + s*Q
    int length;   // Length of the original time series
    int N;        // Length of time series after differencing
    int M;        // 1 if mean needs to be calculated, else 0
    double eps;   // Convergence threshold
    double mean;  // Estimated mean
    double ssq;   // Sum of squares value
    double loglik;// Log-likelihood
    int offset;   // Seasonal offset
    double x[1];  // Flexible array for time series data
};

/**
 * @brief Structure for ARIMA model with exogenous variables and CSS optimization.
 */
typedef struct xlik_css_set* xlik_css_object;

struct xlik_css_set {
    int p;        // Size of phi (non-seasonal AR coefficients)
    int d;        // Number of times the series is to be differenced
    int q;        // Size of theta (non-seasonal MA coefficients)
    int s;        // Frequency of seasonal components
    int P;        // Size of seasonal phi (seasonal AR coefficients)
    int D;        // Number of times the seasonal series is to be differenced
    int Q;        // Size of seasonal theta (seasonal MA coefficients)
    int r;        // max(p + s*P, q + s*Q + 1)
    int pq;       // p + q + s*P + s*Q
    int length;   // Length of the original time series
    int N;        // Length of time series after differencing
    int M;        // Total number of exogenous variables + 1 (if mean is calculated)
    int Nmncond;  // N - ncond
    double eps;   // Convergence threshold
    double mean;  // Estimated mean
    double ssq;   // Sum of squares value
    double loglik;// Log-likelihood
    int offset;   // Seasonal offset
    double x[1];  // Flexible array for time series data
};

/**
 * @brief Structure for seasonal ARIMA model with exact likelihood.
 */
typedef struct alik_seas_set* alik_seas_object;

struct alik_seas_set {
    int p;        // Size of phi (non-seasonal AR coefficients)
    int d;        // Number of times the series is to be differenced
    int q;        // Size of theta (non-seasonal MA coefficients)
    int s;        // Frequency of seasonal components
    int P;        // Size of seasonal phi (seasonal AR coefficients)
    int D;        // Number of times the seasonal series is to be differenced
    int Q;        // Size of seasonal theta (seasonal MA coefficients)
    int r;        // max(p + s*P, q + s*Q + 1)
    int pq;       // p + q + s*P + s*Q
    int length;   // Length of the original time series
    int N;        // Length of time series after differencing
    int M;        // 1 if mean needs to be calculated, else 0
    double eps;   // Convergence threshold
    double mean;  // Estimated mean
    double ssq;   // Sum of squares value
    double loglik;// Log-likelihood
    int offset;   // Seasonal offset
    double x[1];  // Flexible array for time series data
};

/**
 * @brief Structure for ARIMA model with exogenous variables and exact likelihood.
 */
typedef struct xlik_set* xlik_object;

struct xlik_set {
    int p;        // Size of phi (non-seasonal AR coefficients)
    int d;        // Number of times the series is to be differenced
    int q;        // Size of theta (non-seasonal MA coefficients)
    int s;        // Frequency of seasonal components
    int P;        // Size of seasonal phi (seasonal AR coefficients)
    int D;        // Number of times the seasonal series is to be differenced
    int Q;        // Size of seasonal theta (seasonal MA coefficients)
    int r;        // max(p + s*P, q + s*Q + 1)
    int pq;       // p + q + s*P + s*Q + M
    int length;   // Length of the original time series
    int N;        // Length of time series after differencing
    int M;        // Total number of exogenous variables + 1 (if mean is calculated)
    double eps;   // Convergence threshold
    double mean;  // Estimated mean
    double ssq;   // Sum of squares value
    double loglik;// Log-likelihood
    int offset;   // Seasonal offset
    double x[1];  // Flexible array for time series data
};

/* Refactored Kalman Filter Functions from emle.c */

/**
 * @brief Checks the roots of AR and MA polynomials for stationarity and invertibility.
 *
 * Exits the program if the AR part is non-stationary.
 *
 * @param ar_coeffs Non-seasonal AR coefficients.
 * @param ar_order Pointer to the non-seasonal AR order.
 * @param ma_coeffs Non-seasonal MA coefficients.
 * @param ma_order Pointer to the non-seasonal MA order.
 * @param sar_coeffs Seasonal AR coefficients.
 * @param sar_order Pointer to the seasonal AR order.
 * @param sma_coeffs Seasonal MA coefficients.
 * @param sma_order Pointer to the seasonal MA order.
 */
void checkroots(double *ar_coeffs, int *ar_order, double *ma_coeffs, int *ma_order, double *sar_coeffs, int *sar_order, double *sma_coeffs, int *sma_order);

/**
 * @brief Checks the roots of AR and MA polynomials, returning an error code.
 *
 * Returns 10 for non-seasonal AR non-stationarity, 12 for seasonal AR non-stationarity.
 *
 * @param ar_coeffs Non-seasonal AR coefficients.
 * @param ar_order Pointer to the non-seasonal AR order.
 * @param ma_coeffs Non-seasonal MA coefficients.
 * @param ma_order Pointer to the non-seasonal MA order.
 * @param sar_coeffs Seasonal AR coefficients.
 * @param sar_order Pointer to the seasonal AR order.
 * @param sma_coeffs Seasonal MA coefficients.
 * @param sma_order Pointer to the seasonal MA order.
 * @return 1 if successful, 10 or 12 on non-stationarity errors.
 */
int checkroots_cerr(double *ar_coeffs, int *ar_order, double *ma_coeffs, int *ma_order, double *sar_coeffs, int *sar_order, double *sma_coeffs, int *sma_order);

/**
 * @brief Updates least squares decomposition with a new observation.
 *
 * Performs a rank-one update to the triangular decomposition.
 *
 * @param num_vars Number of variables/parameters.
 * @param num_triangular_elems Number of elements in the triangular matrix.
 * @param obs_weight Weight of the new observation.
 * @param next_predictors Array of predictor values for the new observation.
 * @param current_row_predictors Temporary array for row predictors.
 * @param next_response Response value for the new observation.
 * @param diagonal_elements Diagonal elements of the decomposition.
 * @param triangular_matrix Triangular part of the decomposition.
 * @param theta_coeffs Theta coefficients.
 * @param sum_sq_err Pointer to sum of squared errors.
 * @param reciprocal_resid Pointer to reciprocal residual.
 * @param rank Pointer to current rank.
 * @return 0 on success, 1 if weight <= 0.
 */
int inclu2(int num_vars, int num_triangular_elems, double obs_weight, double *next_predictors, double *current_row_predictors, double next_response,
           double *diagonal_elements, double *triangular_matrix, double *theta_coeffs, double *sum_sq_err, double *reciprocal_resid, int *rank);

/**
 * @brief Solves for beta coefficients in a triangular system.
 *
 * Performs back-substitution to compute beta coefficients.
 *
 * @param num_vars Number of variables/parameters.
 * @param num_triangular_elems Number of elements in the triangular matrix.
 * @param triangular_matrix Triangular part of the decomposition.
 * @param theta_coeffs Theta coefficients.
 * @param beta_coeffs Output beta coefficients.
 */
void regres(int num_vars, int num_triangular_elems, double *triangular_matrix, double *theta_coeffs, double *beta_coeffs);

/**
 * @brief Initializes state-space representation for an ARMA model.
 *
 * Sets up state vector, covariance matrix, and innovation covariance.
 *
 * @param ar_order AR order.
 * @param ma_order MA order.
 * @param ar_coeffs AR coefficients.
 * @param ma_coeffs MA coefficients.
 * @param state_vector Initial state vector (output).
 * @param cov_matrix Initial covariance matrix (output).
 * @param innov_cov Innovation covariance (output).
 * @return 0 on success, non-zero on error.
 */
int starma(int ar_order, int ma_order, double *ar_coeffs, double *ma_coeffs, double *state_vector, double *cov_matrix, double *innov_cov);

/**
 * @brief Applies Kalman filter to an ARMA model for likelihood computation.
 *
 * Computes residuals, sum of squared errors, and log-likelihood contributions.
 *
 * @param ar_order AR order.
 * @param ma_order MA order.
 * @param ar_coeffs AR coefficients.
 * @param ma_coeffs MA coefficients.
 * @param state_vector State vector.
 * @param cov_matrix Covariance matrix.
 * @param innov_cov Innovation covariance.
 * @param num_obs Number of observations.
 * @param observations Time series data.
 * @param residuals Output residuals.
 * @param sum_log_det Sum of log determinants (output).
 * @param sum_sq_resid Sum of squared residuals (output).
 * @param init_update Initialization/update flag.
 * @param tolerance Delta tolerance.
 * @param iter_count Iteration count (output).
 * @param num_processed Number of processed observations (output).
 */
void karma(int ar_order, int ma_order, double *ar_coeffs, double *ma_coeffs, double *state_vector, double *cov_matrix, double *innov_cov, int num_obs,
           double *observations, double *residuals, double *sum_log_det, double *sum_sq_resid, int init_update, double tolerance, int *iter_count, int *num_processed);

/**
 * @brief Performs finite-sample prediction for ARIMA models using Kalman filter.
 *
 * Computes forecasts and mean squared errors for an ARIMA model.
 *
 * @param ar_order AR order.
 * @param ma_order MA order.
 * @param diff_order Differencing order.
 * @param ar_coeffs AR coefficients.
 * @param ma_coeffs MA coefficients.
 * @param diff_coeffs Differencing coefficients.
 * @param num_obs Number of observations.
 * @param observations Time series data.
 * @param residuals Output residuals.
 * @param forecast_horizon Forecast horizon.
 * @param forecasts Output forecasts.
 * @param forecast_mse Output mean squared errors.
 * @return 0 on success, non-zero on error.
 */
int forkal(int ar_order, int ma_order, int diff_order, double *ar_coeffs, double *ma_coeffs, double *diff_coeffs, int num_obs, double *observations,
           double *residuals, int forecast_horizon, double *forecasts, double *forecast_mse);

/* Supporting Functions from talg.c */

/**
 * @brief Checks stationarity of AR polynomial.
 *
 * @param order Number of AR coefficients.
 * @param coeffs AR coefficients (length order).
 * @return 1 if stationary, 0 if non-stationary, -1 on error.
 */
int archeck(int order, double *coeffs);

/**
 * @brief Ensures invertibility of MA polynomial.
 *
 * @param order Number of MA coefficients.
 * @param coeffs MA coefficients (input/output, length order).
 * @return 0 on success, -1 on failure.
 */
int invertroot(int order, double *coeffs);

/**
 * @brief Computes autocovariances for ARMA model.
 *
 * @param phi AR coefficients (length p).
 * @param p Number of AR coefficients.
 * @param theta MA coefficients (length q).
 * @param q Number of MA coefficients.
 * @param cov Autocovariances (output, length lag + 1).
 * @param lag Maximum lag.
 * @param n Length of time series.
 */
void twacf(double *phi, int p, double *theta, int q, double *cov, int lag, int n);

/* Supporting Functions from matrix.c */

/**
 * @brief Returns the maximum of two integers.
 *
 * @param value_a First integer.
 * @param value_b Second integer.
 * @return Maximum value.
 */
int imax(int value_a, int value_b);

/**
 * @brief Returns the minimum of two integers.
 *
 * @param value_a First integer.
 * @param value_b Second integer.
 * @return Minimum value.
 */
int imin(int value_a, int value_b);

#ifdef __cplusplus
}
#endif

#endif /* EMLE_REFACTORED_H_ */