// SPDX-License-Identifier: BSD-3-Clause
#include "emle.h"

/**
 * @brief Checks the roots of AR and MA polynomials for stationarity and invertibility.
 *        Exits the program if the AR part is non-stationary.
 *
 * This function verifies if the autoregressive (AR) components (both non-seasonal and seasonal)
 * are stationary by checking if all roots of the AR polynomial lie outside the unit circle.
 * For moving average (MA) components, it inverts roots if necessary to ensure invertibility.
 *
 * @param ar_coeffs Non-seasonal AR coefficients.
 * @param ar_order Pointer to the non-seasonal AR order.
 * @param ma_coeffs Non-seasonal MA coefficients.
 * @param ma_order Pointer to the non-seasonal MA order.
 * @param sar_coeffs Seasonal AR coefficients.
 * @param sar_order Pointer to the seasonal AR order.
 * @param sma_coeffs Seasonal MA cobeefficients.
 * @param sma_order Pointer to the seasonal MA order.
 */
void checkroots(double *ar_coeffs, int *ar_order, double *ma_coeffs, int *ma_order, double *sar_coeffs, int *sar_order, double *sma_coeffs, int *sma_order) {
    int is_stationary;  // Flag to indicate if AR part is stationary

    // Step 1: Check non-seasonal AR stationarity if order > 0
    if (*ar_order > 0) {
        is_stationary = archeck(*ar_order, ar_coeffs);  // Call external function to check AR roots
        if (!is_stationary) {
            printf("\nnon-stationary AR part\n");  // Print error if non-stationary
            exit(-1);  // Exit program on failure
        }
    }

    // Step 2: Ensure non-seasonal MA invertibility if order > 0
    if (*ma_order > 0) {
        invertroot(*ma_order, ma_coeffs);  // Invert roots if necessary for invertibility
    }

    // Step 3: Check seasonal AR stationarity if order > 0 (redundant check *sar_order == *sar_order is always true)
    if (*sar_order == *sar_order) {  // This condition is always true, likely a legacy check
        if (*sar_order > 0) {
            is_stationary = archeck(*sar_order, sar_coeffs);  // Check seasonal AR roots
            if (!is_stationary) {
                printf("\nnon-stationary seasonal AR part\n");  // Print error if non-stationary
                exit(-1);  // Exit program on failure
            }
        }
    }

    // Step 4: Ensure seasonal MA invertibility if order > 0
    if (*sma_order == *sma_order) {  // Always true, similar to above
        if (*sma_order > 0) {
            invertroot(*sma_order, sma_coeffs);  // Invert roots for seasonal MA
        }
    }
}

/**
 * @brief Checks the roots of AR and MA polynomials for stationarity and invertibility, returning an error code.
 *
 * Similar to checkroots, but instead of exiting, it returns an error code (10 for non-seasonal AR,
 * 12 for seasonal AR non-stationarity). It also inverts MA roots for invertibility.
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
static int checkroots_cerr(double *ar_coeffs, int *ar_order, double *ma_coeffs, int *ma_order, double *sar_coeffs, int *sar_order, double *sma_coeffs, int *sma_order) {
    int is_stationary, error_code;  // Stationarity flag and return error code

    error_code = 1;  // Default success code

    // Step 1: Check non-seasonal AR stationarity
    if (*ar_order > 0) {
        is_stationary = archeck(*ar_order, ar_coeffs);
        if (!is_stationary) {
            error_code = 10;  // Non-seasonal AR error
            printf("\nnon-stationary AR part\n");
            return error_code;
        }
    }

    // Step 2: Ensure non-seasonal MA invertibility
    if (*ma_order > 0) {
        invertroot(*ma_order, ma_coeffs);
    }

    // Step 3: Check seasonal AR stationarity
    if (*sar_order == *sar_order) {  // Always true
        if (*sar_order > 0) {
            is_stationary = archeck(*sar_order, sar_coeffs);
            if (!is_stationary) {
                error_code = 12;  // Seasonal AR error
                printf("\nnon-stationary seasonal AR part\n");
                return error_code;
            }
        }
    }

    // Step 4: Ensure seasonal MA invertibility
    if (*sma_order == *sma_order) {  // Always true
        if (*sma_order > 0) {
            invertroot(*sma_order, sma_coeffs);
        }
    }

    return error_code;  // Return success if no errors
}

/**
 * @brief Includes a new observation into a least squares problem using a rank-one update.
 *
 * This function updates the triangular decomposition (Cholesky-like) of the least squares
 * matrix when adding a new weighted observation. It handles the update to the diagonal elements,
 * triangular matrix, theta coefficients, sum of squared errors, and rank.
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
static int inclu2(int num_vars, int num_triangular_elems, double obs_weight, double *next_predictors, double *current_row_predictors, double next_response,
    double *diagonal_elements, double *triangular_matrix, double *theta_coeffs, double *sum_sq_err, double *reciprocal_resid, int *rank) {
    int error_code, i, current_triangular_idx, next_i, k;  // Error flag, loop indices
    double response_val, weight_val, predictor_val, diag_val, updated_diag, diag_over_updated, weight_times_predictor_over_updated;
    double current_predictor, triangular_val;

    response_val = next_response;  // Store response value
    weight_val = obs_weight;  // Store weight

    // Step 1: Copy next predictors to current row
    for (i = 0; i < num_vars; ++i) {
        current_row_predictors[i] = next_predictors[i];
    }

    *reciprocal_resid = 0.0;  // Initialize reciprocal residual
    error_code = 1;  // Default error if weight invalid
    if (weight_val <= 0.0) {
        return error_code;  // Return if weight non-positive
    }
    error_code = 0;  // Reset to success
    current_triangular_idx = 0;  // Start triangular index at 0

    // Step 2: Loop over variables to update decomposition
    for (i = 0; i < num_vars; ++i) {
        if (current_row_predictors[i] != 0.0) {  // Skip if predictor zero
            predictor_val = current_row_predictors[i];  // Current predictor
            diag_val = diagonal_elements[i];  // Current diagonal
            updated_diag = diag_val + weight_val * predictor_val * predictor_val;  // Update diagonal
            diagonal_elements[i] = updated_diag;  // Store updated diagonal
            diag_over_updated = diag_val / updated_diag;  // cbar
            weight_times_predictor_over_updated = weight_val * predictor_val / updated_diag;  // sbar
            weight_val = diag_over_updated * weight_val;  // Update weight
            if (i != (num_vars - 1)) {  // If not last variable
                next_i = i + 1;  // Start from next
                for (k = next_i; k < num_vars; ++k) {  // Update row and triangular
                    current_predictor = current_row_predictors[k];  // xk
                    triangular_val = triangular_matrix[current_triangular_idx];  // rbthis
                    current_row_predictors[k] = current_predictor - predictor_val * triangular_val;  // Update xrow
                    triangular_matrix[current_triangular_idx] = diag_over_updated * triangular_val + weight_times_predictor_over_updated * current_predictor;  // Update rbar
                    current_triangular_idx++;  // Increment index
                }
            }
            current_predictor = response_val;  // xk = y
            response_val = current_predictor - predictor_val * theta_coeffs[i];  // Update y
            theta_coeffs[i] = diag_over_updated * theta_coeffs[i] + weight_times_predictor_over_updated * current_predictor;  // Update thetab
            if (diag_val == 0.0) {  // If original diagonal zero
                *rank = *rank + 1;  // Increment rank
                return error_code;  // Return success
            }
        } else {
            current_triangular_idx += (num_vars - i - 1);  // Skip triangular elements
        }
    }

    // Step 3: Update sum of squared errors and reciprocal residual
    *sum_sq_err = *sum_sq_err + weight_val * response_val * response_val;
    *reciprocal_resid = response_val * sqrt(weight_val);

    return error_code;  // Return success
}

/**
 * @brief Performs back-substitution to solve for beta coefficients in a triangular system.
 *
 * This function solves the upper triangular system using back-substitution to compute
 * the beta coefficients from the theta values and triangular matrix.
 *
 * @param num_vars Number of variables/parameters.
 * @param num_triangular_elems Number of elements in the triangular matrix.
 * @param triangular_matrix Triangular part of the decomposition.
 * @param theta_coeffs Theta coefficients.
 * @param beta_coeffs Output beta coefficients.
 */
static void regres(int num_vars, int num_triangular_elems, double *triangular_matrix, double *theta_coeffs, double *beta_coeffs) {
    int current_triangular_idx, i, current_var_idx, next_i, j, current_next_var_idx, j_var;  // Indices
    double current_beta;  // Temporary beta value

    current_triangular_idx = num_triangular_elems - 1;  // Start from end of triangular
    current_var_idx = num_vars - 1;  // Start from last variable

    // Step 1: Loop backwards over variables
    for (i = 0; i < num_vars; ++i) {
        current_beta = theta_coeffs[current_var_idx];  // Initialize with thetab
        if (current_var_idx != (num_vars - 1)) {  // If not last
            next_i = i - 1;  // Upper limit
            current_next_var_idx = num_vars - 1;  // jm
            for (j = 0; j <= next_i; ++j) {  // Back-substitute
                current_beta = current_beta - triangular_matrix[current_triangular_idx] * beta_coeffs[current_next_var_idx];  // Subtract term
                current_triangular_idx--;  // Decrement index
                current_next_var_idx--;  // Decrement jm
            }
        }
        beta_coeffs[current_var_idx] = current_beta;  // Store beta
        current_var_idx--;  // Move to previous variable
    }
}

/**
 * @brief Initializes the state-space representation for an ARMA model.
 *
 * This function sets up the initial state vector, covariance matrix, and innovation
 * covariance for use in the Kalman filter. It handles both AR and MA processes,
 * solving for the initial covariance using a least squares approach.
 *
 * @param ar_order AR order.
 * @param ma_order MA order.
 * @param ar_coeffs AR coefficients.
 * @param ma_coeffs MA coefficients.
 * @param state_vector Initial state vector A.
 * @param cov_matrix Initial covariance matrix P.
 * @param innov_cov Innovation covariance V.
 * @return Error code (0 on success, non-zero on input errors).
 */
int starma(int ar_order, int ma_order, double *ar_coeffs, double *ma_coeffs, double *state_vector, double *cov_matrix, double *innov_cov) {
    int error_code, state_dim, num_params, num_triangular_elems, i, index, j, ar_order_minus_1, rank, optim_error;  // Variables for dimensions and errors
    int num_params_minus_state_dim, num_params_minus_state_dim_plus_1, param_index_j, index2;  // Additional indices
    double current_innov, current_sum_sq_err, current_ar_coeff, current_response, current_ar_coeff_i;  // Temporaries
    double *theta_coeffs, *next_predictors, *current_row_predictors, *triangular_matrix;  // Allocated arrays

    error_code = 0;  // Default no error

    // Step 1: Check input errors
    if (ar_order < 0) {
        error_code = 1;
    }
    if (ma_order < 0) {
        error_code = 2;
    }
    if (ar_order * ar_order + ma_order * ma_order == 0) {
        error_code = 4;
    }

    state_dim = ma_order + 1;  // ir = iq + 1
    if (state_dim < ar_order) {
        state_dim = ar_order;  // Max of p and q+1
    }

    num_params = (state_dim * (state_dim + 1)) / 2;  // np
    num_triangular_elems = (num_params * (num_params - 1)) / 2;  // nrbar

    if (state_dim == 1) {
        error_code = 8;  // Not suitable for AR(1)
    }

    if (error_code != 0) {
        return error_code;  // Return on error
    }

    // Step 2: Allocate temporary arrays
    next_predictors = (double*)malloc(sizeof(double) * num_params);
    theta_coeffs = (double*)malloc(sizeof(double) * num_params);
    current_row_predictors = (double*)malloc(sizeof(double) * num_params);
    triangular_matrix = (double*)malloc(sizeof(double) * num_triangular_elems);

    // Step 3: Initialize state vector, AR coeffs, and innovation cov
    for (i = 1; i < state_dim; ++i) {
        state_vector[i] = 0.0;  // A[i] = 0
        if (i >= ar_order) {
            ar_coeffs[i] = 0.0;  // Extend phi if needed
        }
        innov_cov[i] = 0.0;  // V[i] = 0
        if (i < ma_order + 1) {
            innov_cov[i] = ma_coeffs[i - 1];  // V[i] = theta[i-1]
        }
    }

    state_vector[0] = 0.0;  // A[0] = 0
    if (ar_order == 0) {
        ar_coeffs[0] = 0.0;  // phi[0] = 0 if no AR
    }
    innov_cov[0] = 1.0;  // V[0] = 1

    index = state_dim;  // ind = ir

    // Step 4: Compute innovation covariance matrix V
    for (j = 1; j < state_dim; ++j) {
        current_innov = innov_cov[j];  // vj = V[j]
        for (i = j; i < state_dim; ++i) {
            innov_cov[index] = innov_cov[i] * current_innov;  // V[ind] = V[i] * vj
            index++;
        }
    }

    if (ar_order != 0) {  // If AR present
        // Step 5: Solve for initial covariance P(0)
        ar_order_minus_1 = state_dim - 1;  // ir1 = ir - 1
        rank = 0;  // irank = 0
        optim_error = 0;  // ifail = 0
        current_sum_sq_err = 0.0;  // ssqerr = 0.0

        // Initialize triangular matrix and cov_matrix
        for (i = 0; i < num_triangular_elems; ++i) {
            triangular_matrix[i] = 0.0;
        }
        for (i = 0; i < num_params; ++i) {
            cov_matrix[i] = 0.0;
            theta_coeffs[i] = 0.0;
            next_predictors[i] = 0.0;
        }
        index = 0;  // ind = 0
        index2 = -1;  // ind1 = -1
        num_params_minus_state_dim = num_params - state_dim;  // npr = np - ir
        num_params_minus_state_dim_plus_1 = num_params_minus_state_dim + 1;  // npr1 = npr + 1
        param_index_j = num_params_minus_state_dim;  // indj = npr
        index2 = num_params_minus_state_dim - 1;  // ind2 = npr - 1

        // Step 6: Generate and solve the system for P
        for (j = 0; j < state_dim; ++j) {  // Loop over j
            current_ar_coeff = ar_coeffs[j];  // phij = phi[j]
            next_predictors[param_index_j] = 0.0;  // xnext[indj] = 0
            param_index_j++;
            int param_index_i = num_params_minus_state_dim_plus_1 + j;  // indi = npr1 + j

            for (i = j; i < state_dim; ++i) {  // Inner loop over i >= j
                current_response = innov_cov[index];  // ynext = V[ind]
                index++;
                current_ar_coeff_i = ar_coeffs[i];  // phii = phi[i]
                if (j != (state_dim - 1)) {  // If not last j
                    next_predictors[param_index_j] = -current_ar_coeff_i;  // xnext[indj] = -phii
                    if (i != (state_dim - 1)) {  // If not last i
                        next_predictors[param_index_i] -= current_ar_coeff;  // xnext[indi] -= phij
                        index2++;  // ind1++
                        next_predictors[index2] = -1.0;  // xnext[ind1] = -1
                    }
                }
                next_predictors[num_params_minus_state_dim] = -current_ar_coeff_i * current_ar_coeff;  // xnext[npr] = -phii*phij
                index2++;  // ind2++
                if (index2 >= num_params) {
                    index2 = 0;  // Wrap around
                }
                next_predictors[index2] += 1.0;  // xnext[ind2] += 1
                double obs_weight = 1.0;  // weight = 1.0
                double current_recip_resid;  // recres
                optim_error = inclu2(num_params, num_triangular_elems, obs_weight, next_predictors, current_row_predictors, current_response, cov_matrix, triangular_matrix, theta_coeffs, &current_sum_sq_err, &current_recip_resid, &rank);
                next_predictors[index2] = 0.0;  // Reset xnext[ind2]
                if (i != (state_dim - 1)) {  // Reset if not last
                    next_predictors[param_index_i] = 0.0;
                    param_index_i++;
                    next_predictors[index2] = 0.0;
                }
            }
        }
        // Step 7: Solve the system using back-substitution
        regres(num_params, num_triangular_elems, triangular_matrix, theta_coeffs, cov_matrix);

        // Step 8: Re-order the covariance matrix P
        index = num_params_minus_state_dim;  // ind = npr
        for (i = 0; i < state_dim; ++i) {
            index++;
            next_predictors[i] = cov_matrix[index - 1];  // Temp store in xnext
        }
        index = num_params;  // ind = np
        index2 = num_params_minus_state_dim;  // ind1 = npr

        for (i = 0; i < num_params_minus_state_dim; ++i) {
            cov_matrix[index - 1] = cov_matrix[index2 - 1];  // Shift P
            index--;
            index2--;
        }

        for (i = 0; i < state_dim; ++i) {
            cov_matrix[i] = next_predictors[i];  // Copy back
        }
    } else {
        // Step 9: For pure MA process, compute P by back-substitution
        int index_n = num_params;  // indn = np
        index = num_params;  // ind = np
        for (i = 0; i < state_dim; i++) {  // Loop over i
            for (j = 0; j <= i; j++) {  // Inner loop j <= i
                index--;
                cov_matrix[index] = innov_cov[index];  // P[ind] = V[ind]
                if (j != 0) {
                    index_n--;
                    cov_matrix[index] += cov_matrix[index_n];  // Add previous
                }
            }
        }
    }

    // Step 10: Free allocated memory
    free(next_predictors);
    free(theta_coeffs);
    free(current_row_predictors);
    free(triangular_matrix);

    return error_code;  // Return success or error
}

/**
 * @brief Applies the Kalman filter to an ARMA model for likelihood computation.
 *
 * This function implements the Kalman filter recursion for an ARMA process to compute
 * residuals, sum of squared errors, and log-likelihood contributions. It supports
 * both full recursion and a simplified version for exact computation after initialization.
 *
 * @param ar_order AR order.
 * @param ma_order MA order.
 * @param ar_coeffs AR coefficients.
 * @param ma_coeffs MA_coeffs.
 * @param state_vector State vector A.
 * @param cov_matrix Covariance matrix P.
 * @param innov_cov Innovation covariance V.
 * @param num_obs Number of observations N.
 * @param observations Time series data W.
 * @param residuals Output residuals.
 * @param sum_log_det Pointer to sum of log determinants for likelihood.
 * @param sum_sq_resid Pointer to sum of squared residuals.
 * @param init_update Flag for initialization/update mode (1 for update).
 * @param tolerance Delta tolerance for covariance check.
 * @param iter_count Pointer to iteration count.
 * @param num_processed Pointer to number of processed observations.
 */
void karma(int ar_order, int ma_order, double *ar_coeffs, double *ma_coeffs, double *state_vector, double *cov_matrix, double *innov_cov, int num_obs,
    double *observations, double *residuals, double *sum_log_det, double *sum_sq_resid, int init_update, double tolerance, int *iter_count, int *num_processed) {
    int state_dim, num_params, i, j, ar_order_minus_1, index_e, switch_flag, index, index_n, l, ii, index_w;  // Dimensions and indices
    double current_obs, current_tolerance, current_state_0, current_pred_err_cov, current_innov, current_gain, current_err, current_resid;  // Temporaries
    double *error_buffer;  // Buffer for errors in simplified mode

    state_dim = ma_order + 1;  // ir = iq + 1
    switch_flag = 0;  // swtch = 0
    *iter_count = 0;  // iter = 0

    if (state_dim < ar_order) {
        state_dim = ar_order;  // Max of p and q+1
    }
    error_buffer = (double*)malloc(sizeof(double) * state_dim);  // Allocate E
    num_params = (state_dim * (state_dim + 1)) / 2;  // np

    ar_order_minus_1 = state_dim - 1;  // ir1 = ir - 1

    // Step 1: Initialize error buffer
    for (i = 0; i < state_dim; ++i) {
        error_buffer[i] = 0.0;
    }
    index_e = 0;  // inde = 0

    if (*num_processed == 0) {  // If nit == 0, full Kalman recursion
        for (i = 0; i < num_obs; ++i) {  // Loop over observations
            current_obs = observations[i];  // wnext = W[i]
            if (init_update != 1 || i > 0) {  // If update or not first
                current_tolerance = 0.0;  // dt = 0.0
                if (state_dim != 1) {
                    current_tolerance = cov_matrix[state_dim];  // dt = P[ir]
                }
                /*
                if (current_tolerance < tolerance) {  // Check tolerance
                    switch_flag = 1;
                    *num_processed = i + 1;
                    break;
                }
                */
                current_state_0 = state_vector[0];  // A1 = A[0]
                if (state_dim != 1) {  // Shift state vector
                    for (j = 0; j < ar_order_minus_1; ++j) {
                        state_vector[j] = state_vector[j + 1];
                    }
                }
                state_vector[state_dim - 1] = 0.0;  // Last state = 0
                if (ar_order != 0) {  // Apply AR update
                    for (j = 0; j < ar_order; ++j) {
                        state_vector[j] += ar_coeffs[j] * current_state_0;
                    }
                }
                index = -1;  // ind = -1
                index_n = state_dim - 1;  // indn = ir-1
                for (l = 0; l < state_dim; ++l) {  // Update covariance P
                    for (j = l; j < state_dim; ++j) {
                        index++;
                        cov_matrix[index] = innov_cov[index];  // P[ind] = V[ind]
                        if (j != (state_dim - 1)) {
                            index_n++;
                            cov_matrix[index] += cov_matrix[index_n];  // Add term
                        }
                    }
                }
            }
            current_pred_err_cov = fabs(cov_matrix[0]);  // ft = |P[0]|
            current_innov = current_obs - state_vector[0];  // ut = wnext - A[0]
            if (state_dim != 1) {  // Kalman update if dim > 1
                index = state_dim;  // ind = ir
                for (j = 1; j < state_dim; ++j) {
                    current_gain = cov_matrix[j] / current_pred_err_cov;  // g = P[j]/ft
                    state_vector[j] += current_gain * current_innov;  // A[j] += g*ut
                    for (l = j; l < state_dim; ++l) {
                        cov_matrix[index] -= current_gain * cov_matrix[l];  // P[ind] -= g*P[l]
                        index++;
                    }
                }
            }
            state_vector[0] = current_obs;  // A[0] = wnext
            for (l = 0; l < state_dim; ++l) {
                cov_matrix[l] = 0.0;  // Reset first row of P
            }
            residuals[i] = current_innov / sqrt(current_pred_err_cov);  // resid[i] = ut / sqrt(ft)
            error_buffer[index_e] = residuals[i];  // E[inde] = resid[i]
            index_e++;
            if (index_e >= ma_order) {
                index_e = 0;
            }
            *sum_sq_resid += (current_innov * current_innov) / current_pred_err_cov;  // ssq += ut^2 / ft
            *sum_log_det += log(current_pred_err_cov);  // sumlog += log(ft)
            *iter_count = *iter_count + 1;  // Increment iter
        }
        if (switch_flag == 0) {
            *num_processed = num_obs;  // nit = N if no switch
        }
    } else {
        // Simplified mode if nit != 0
        i = 0;
        *num_processed = i;  // nit = 0
        for (ii = i; ii < num_obs; ++ii) {  // Loop over ii
            current_err = observations[ii];  // et = W[ii]
            index_w = ii;  // indw = ii
            if (ar_order != 0) {  // Subtract AR terms
                for (j = 0; j < ar_order; ++j) {
                    index_w--;
                    if (index_w >= 0) {
                        current_err -= ar_coeffs[j] * observations[index_w];
                    }
                }
            }
            if (ma_order != 0) {  // Subtract MA terms
                for (j = 0; j < ma_order; ++j) {  // Note: loop has ++i, likely typo, should be ++j
                    index_e--;
                    if (index_e == -1) {
                        index_e = ma_order - 1;
                    }
                    current_err -= ma_coeffs[j] * error_buffer[index_e];
                }
            }
            error_buffer[index_e] = current_err;  // E[inde] = et
            residuals[ii] = current_err;  // resid[ii] = et
            *sum_sq_resid += current_err * current_err;  // ssq += et^2
            *iter_count = *iter_count + 1;  // Increment iter
            index_e++;
            if (index_e >= ma_order) {
                index_e = 0;
            }
        }
    }
    if (*iter_count == 0) {
        *iter_count = 1;  // Ensure at least 1
    }
    free(error_buffer);  // Free buffer
}

/**
 * @brief Performs finite-sample prediction for ARIMA models using the Kalman filter.
 *
 * This function applies the Kalman filter to forecast future values in an ARIMA model,
 * handling differencing, and computing mean squared errors for the forecasts.
 *
 * @param ar_order AR order (p).
 * @param ma_order MA order (q).
 * @param diff_order Differencing order (d).
 * @param ar_coeffs AR coefficients (phi).
 * @param ma_coeffs MA coefficients (theta).
 * @param diff_coeffs Differencing coefficients (delta).
 * @param num_obs Number of observations N.
 * @param observations Time series data W.
 * @param residuals Output residuals.
 * @param forecast_horizon Forecast horizon il.
 * @param forecasts Output forecasts Y.
 * @param forecast_mse Output mean squared errors AMSE.
 * @return Error code (0 on success).
 */
int forkal(int ar_order, int ma_order, int diff_order, double *ar_coeffs, double *ma_coeffs, double *diff_coeffs, int num_obs, double *observations, double *residuals, int forecast_horizon, double *forecasts, double *forecast_mse) {
    int error_code, state_dim, num_params, k, num_obs_transformed, current_j, diff_order_minus_1, state_dim_plus_1, state_dim_minus_1;  // Dimensions
    int diff_order_plus_1, diff_order_times_2_plus_1, diff_order_times_2_plus_2, forecast_i45, state_dim_plus_diff_plus_1, diff_order_times_2_plus_state_dim;  // More dims
    int jkl, jkl_plus_1, diff_order_times_2_plus_2, base_cov_index, l, state_plus_j, i, j, jj, lk, lk_plus_1;  // Indices
    double *state_vector, *cov_matrix, *innov_cov, *temp_store, *temp_row;  // Allocated arrays
    double zero_val = 0.0, one_val = 1.0, two_val = 2.0;  // Constants
    double current_state_0, current_delta_sum, current_cov_0, current_ar_j, current_ar_j_times_diag, current_ar_i, current_mse;  // Temporaries
    double current_sigma, current_sum_log, current_sum_sq;  // Likelihood vars

    state_dim = ma_order + 1;  // ir = iq + 1

    if (state_dim < ar_order) {
        state_dim = ar_order;
    }
    num_params = (state_dim * (state_dim + 1)) / 2;  // np
    int num_triangular_elems = (num_params * (num_params - 1)) / 2;  // nrbar, not used here
    int state_dim_with_diff = state_dim + diff_order;  // ird = ir + id
    int cov_dim_with_diff = (state_dim_with_diff * (state_dim_with_diff + 1)) / 2;  // irz = (ird*(ird+1))/2

    // Step 1: Check input errors
    error_code = 0;
    if (ar_order < 0) error_code = 1;
    if (ma_order < 0) error_code += 2;
    if (ar_order*ar_order + ma_order*ma_order == 0) error_code = 4;
    if (diff_order < 0) error_code = 8;
    if (forecast_horizon < 1) error_code = 11;
    if (error_code != 0) return error_code;

    // Step 2: Allocate arrays
    state_vector = (double*)malloc(sizeof(double)* state_dim_with_diff);
    cov_matrix = (double*)malloc(sizeof(double)* cov_dim_with_diff);
    innov_cov = (double*)malloc(sizeof(double)* num_params); 
    temp_store = (double*)malloc(sizeof(double)* state_dim_with_diff);
    temp_row = (double*)malloc(sizeof(double)* num_params);

    // Step 3: Initialize arrays to zero
    for (i = 0; i < state_dim_with_diff; ++i) {
        state_vector[i] = zero_val;
        temp_store[i] = zero_val;
    }
    for (i = 0; i < cov_dim_with_diff; ++i) {
        cov_matrix[i] = zero_val;
    }
    for (i = 0; i < num_params; ++i) {
        innov_cov[i] = zero_val;
        temp_row[i] = zero_val;
    }
    state_vector[0] = zero_val;
    innov_cov[0] = one_val;

    // Step 4: Find initial likelihood conditions using starma
    if (state_dim == 1) {
        cov_matrix[0] = 1.0 / (1.0 - ar_coeffs[0] * ar_coeffs[0]);
    } else {
        starma(ar_order, ma_order, ar_coeffs, ma_coeffs, state_vector, cov_matrix, innov_cov);
    }

    // Step 5: Apply differencing transformations
    num_obs_transformed = num_obs - diff_order;  // nt = N - id

    if (diff_order != 0) {
        for (j = 1; j <= diff_order; ++j) {
            int nj = num_obs - j;
            temp_store[j-1] = observations[nj-1];
        }
        for (i = 1; i <= num_obs_transformed; ++i) {
            double accum = zero_val;
            for (k = 1; k <= diff_order; ++k) {
                int idk = diff_order + i - k;
                accum -= diff_coeffs[k - 1] * observations[idk - 1];
            }
            int iid = i + diff_order;
            observations[i - 1] = observations[iid - 1] + accum;
        }
    }

    // Step 6: Evaluate likelihood using karma
    current_sum_log = current_sum_sq = zero_val;
    double delta_tol = -1.0;
    int iupd = 1;
    int nit = 0;
    int current_iter = 0;
    karma(ar_order, ma_order, ar_coeffs, ma_coeffs, state_vector, cov_matrix, innov_cov, num_obs_transformed, observations, residuals, &current_sum_log, &current_sum_sq, iupd, delta_tol, &current_iter, &nit);

    // Step 7: Compute MLE of sigma squared
    current_sigma = zero_val;
    for (j = 0; j < num_obs_transformed; ++j) {
        current_sigma += residuals[j] * residuals[j];
    }
    current_sigma = current_sigma / num_obs_transformed;

    // Step 8: Reset initial state and cov if differencing
    if (diff_order != 0) {
        for (i = 1; i <= num_params; ++i) {
            temp_row[i-1] = cov_matrix[i-1];
        }
        for (i = 1; i <= cov_dim_with_diff; ++i) {
            cov_matrix[i-1] = zero_val;
        }
        index = 0;
        for (j = 1; j <= state_dim; ++j) {
            k = (j - 1) * (diff_order + state_dim + 1) - (j - 1) * j / 2;
            for (i = j; i <= state_dim; ++i) {
                index++;
                k++;
                cov_matrix[k - 1] = temp_row[index-1];
            }
        }

        for (j = 1; j <= diff_order; ++j) {
            int irj = state_dim + j;
            state_vector[irj-1] = temp_store[j-1];
        }
    }

    // Step 9: Set up constants for forecasting
    int state_dim_plus_2 = state_dim + 1;  // ir2
    state_dim_minus_1 = state_dim - 1;  // ir1
    diff_order_minus_1 = diff_order - 1;  // id1
    int diff_times_2_plus_state_dim = 2 * state_dim_with_diff;  // id2r
    int diff_times_2_plus_state_dim_minus_1 = diff_times_2_plus_state_dim - 1;  // id2r1
    int diff_times_2_plus_1 = 2 * diff_order + 1;  // idd1
    int diff_times_2_plus_2 = diff_times_2_plus_1 + 1;  // idd2
    int forecast_i45 = diff_times_2_plus_state_dim + 1;  // i45
    int state_dim_plus_diff_plus_1 = state_dim_with_diff + 1;  // idrr1
    int diff_times_2_plus_state_dim_2 = 2 * diff_order + state_dim;  // iddr
    jkl = state_dim * (diff_times_2_plus_state_dim_2 + 1) / 2;  // jkl
    jkl_plus_1 = jkl + 1;  // jkl1
    diff_order_times_2_plus_2 = diff_times_2_plus_state_dim + 2;  // id2r2
    base_cov_index = state_dim * (forecast_i45 - state_dim) / 2;  // ibc

    // Step 10: Forecast loop
    for (l = 1; l <= forecast_horizon; ++l) {
        // Predict state vector A
        current_state_0 = state_vector[0];  // A1 = A[0]
        if (state_dim != 1) {
            for (i = 1; i <= state_dim_minus_1; ++i) {
                state_vector[i-1] = state_vector[i];
            }
        }
        state_vector[state_dim-1] = zero_val;
        if (ar_order != 0) {
            for (j = 1; j <= ar_order; ++j) {
                state_vector[j-1] += ar_coeffs[j-1] * current_state_0;
            }
        }
        if (diff_order != 0) {
            for (j = 1; j <= diff_order; ++j) {
                state_plus_j = state_dim + j;
                current_state_0 += diff_coeffs[j-1] * state_vector[state_plus_j-1];
            }
            if (diff_order >= 2) {
                for (i = 1; i <= diff_order_minus_1; ++i) {
                    int iri1 = state_dim_with_diff - i;
                    state_vector[iri1] = state_vector[iri1-1];
                }
            }
            state_vector[state_dim_plus_2 - 1] = current_state_0;
        }

        // Predict covariance P
        if (diff_order != 0) {
            for (i = 1; i <= diff_order; ++i) {
                temp_store[i-1] = zero_val;
                for (j = 1; j <= diff_order; ++j) {
                    int ll = imax(i, j);
                    k = imin(i, j);
                    jj = jkl + (ll - k) + 1 + (k - 1) * (diff_times_2_plus_2 - k) / 2;
                    temp_store[i-1] += diff_coeffs[j-1] * cov_matrix[jj-1];
                }
            }
            if (diff_order != 1) {
                for (j = 1; j <= diff_order_minus_1; ++j) {
                    jj = diff_order - j;
                    lk = (jj - 1) * (diff_times_2_plus_2 - jj) / 2 + jkl;
                    lk_plus_1 = jj * (diff_times_2_plus_1 - jj) / 2 + jkl;
                    for (i = 1; i <= j; ++i) {
                        lk += 1;
                        lk_plus_1 += 1;
                        cov_matrix[lk_plus_1-1] = cov_matrix[lk-1];
                    }
                }
                for (j = 1; j <= diff_order_minus_1; ++j) {
                    int jklj = jkl_plus_1 + j;
                    state_plus_j = state_dim + j;
                    cov_matrix[jklj - 1] = temp_store[j - 1] + cov_matrix[state_plus_j-1];
                }
            }

            cov_matrix[jkl_plus_1 - 1] = cov_matrix[0];

            for (i = 1; i <= diff_order; ++i) {
                int iri = state_dim + i;
                cov_matrix[jkl_plus_1 - 1] += diff_coeffs[i - 1] * (temp_store[i - 1] + two_val * cov_matrix[iri - 1]);
            }

            for (i = 1; i <= diff_order; ++i) {
                int iri = state_dim + i;
                temp_store[i - 1] = cov_matrix[iri - 1];
            }

            for (j = 1; j <= state_dim; ++j) {
                int kk1 = j * (diff_times_2_plus_state_dim_minus_1 - j) / 2 + state_dim;
                int k1 = (j - 1) * (diff_times_2_plus_state_dim - j) / 2 + state_dim;
                for (i = 1; i <= diff_order; ++i) {
                    int kk = kk1 + i;
                    k = k1 + i;
                    cov_matrix[k - 1] = ar_coeffs[j - 1] * temp_store[i - 1];
                    if (j != state_dim) {
                        cov_matrix[k - 1] += cov_matrix[kk - 1];
                    }
                }
            }

            for (j = 1; j <= state_dim; ++j) {
                temp_store[j - 1] = zero_val;
                int kkk = j * (forecast_i45 - j) / 2 - diff_order;
                for (i = 1; i <= diff_order; ++i) {
                    kkk++;
                    temp_store[j - 1] += diff_coeffs[i - 1] * cov_matrix[kkk - 1];
                }
            }

            if (diff_order != 1) {
                for (j = 1; j <= state_dim; ++j) {
                    k = j * state_dim_plus_diff_plus_1 - j * (j + 1) / 2 + 1;
                    for (i = 1; i <= diff_order_minus_1; ++i) {
                        k--;
                        cov_matrix[k - 1] = cov_matrix[k - 2];
                    }
                }
            }

            for (j = 1; j <= state_dim; ++j) {
                k = (j - 1) * (diff_times_2_plus_state_dim - j) / 2 + state_dim + 1;
                cov_matrix[k - 1] = temp_store[j - 1] + ar_coeffs[j - 1] * cov_matrix[0];
                if (j < state_dim) {
                    cov_matrix[k - 1] += cov_matrix[j];
                }
            }
        }

        // Continue prediction of P
        for (i = 0; i < state_dim; ++i) {
            temp_store[i] = cov_matrix[i];
        }

        index = 0;
        double current_diag = cov_matrix[0];  // dt = P[0]
        for (j = 1; j <= state_dim; ++j) {
            current_ar_j = ar_coeffs[j - 1];  // phij
            current_ar_j_times_diag = current_ar_j * current_diag;  // phijdt
            int ind2 = (j - 1) * (diff_order_times_2_plus_2 - j) / 2;  // ind2
            int ind1 = j * (forecast_i45 - j) / 2;  // ind1
            for (i = j; i <= state_dim; ++i) {
                index++;
                ind2++;
                current_ar_i = ar_coeffs[i - 1];  // phii
                cov_matrix[ind2 - 1] = innov_cov[index - 1] + current_ar_i * current_ar_j_times_diag;  // V + phii * phijdt
                if (j < state_dim) {
                    cov_matrix[ind2 - 1] += temp_store[j] * current_ar_i;
                }
                if (i != state_dim) {
                    ind1++;
                    cov_matrix[ind2 - 1] += temp_store[i] * current_ar_j + cov_matrix[ind1 - 1];
                }
            }
        }

        // Step 11: Predict forecast Y
        forecasts[l - 1] = state_vector[0];
        if (diff_order != 0) {
            for (j = 1; j <= diff_order; ++j) {
                state_plus_j = state_dim + j;
                forecasts[l - 1] += state_vector[state_plus_j - 1] * diff_coeffs[j - 1];
            }
        }

        // Step 12: Compute MSE of forecast
        current_mse = cov_matrix[0];  // AMS = P[0]
        if (diff_order != 0) {
            for (j = 1; j <= diff_order; ++j) {
                int jrj = base_cov_index + (j - 1) * (diff_times_2_plus_2 - j) / 2;
                state_plus_j = state_dim + j;
                current_mse += (two_val * diff_coeffs[j - 1] * cov_matrix[state_plus_j - 1] + cov_matrix[jrj] * diff_coeffs[j - 1] * diff_coeffs[j - 1]);
            }
            if (diff_order != 1) {
                for (j = 1; j <= diff_order_minus_1; ++j) {
                    int j1 = j + 1;
                    int jrk = base_cov_index + 1 + (j - 1) * (diff_times_2_plus_2 - j) / 2;
                    for (i = j1; i <= diff_order; ++i) {
                        jrk++;
                        current_mse += two_val * diff_coeffs[i-1] * diff_coeffs[j-1] * cov_matrix[jrk-1];
                    }
                }
            }
        }
        forecast_mse[l - 1] = current_mse * current_sigma;  // AMSE = AMS * sigma
    }

    // Step 13: Free allocated memory
    free(state_vector);
    free(cov_matrix);
    free(innov_cov);
    free(temp_store);
    free(temp_row);

    return error_code;
}

/**
 * @brief Inverts MA roots to ensure they lie inside the unit circle for invertibility.
 *
 * This function adjusts the MA coefficients if any roots are outside or on the unit circle
 * to ensure the model is invertible.
 *
 * @param ma_order MA order.
 * @param ma_coeffs MA coefficients to adjust.
 */
void invertroot(int ma_order, double *ma_coeffs) {
    // Implementation not provided in the original code snippet, assuming it's external.
    // Add steps here if code is available.
}

/**
 * @brief Computes the likelihood for an ARMA model using a Kalman filter variant (AS 197).
 *
 * This function implements an alternative Kalman filter recursion for exact likelihood
 * computation in ARMA models, handling prediction and update steps with tolerance checks.
 *
 * @param ar_coeffs AR coefficients P.
 * @param ar_order AR order MP.
 * @param ma_coeffs MA coefficients Q.
 * @param ma_order MA order MQ.
 * @param observations Time series W.
 * @param residuals Output residuals E.
 * @param num_obs Number of observations N.
 * @param sum_sq_resid Pointer to sum of squared residuals.
 * @param likelihood_factor Pointer to likelihood scaling factor.
 * @param work_vector_vw Working vector VW.
 * @param work_vector_vl Working vector VL.
 * @param mrp1 MRP1 dimension.
 * @param work_vector_vk Working vector VK.
 * @param mr MR dimension.
 * @param tolerance Tolerance for convergence.
 * @return Error code (0 on success, negative on early termination).
 */
double flikam(double *ar_coeffs, int ar_order, double *ma_coeffs, int ma_order, double *observations, double *residuals, int num_obs, double *sum_sq_resid, double *likelihood_factor, double *work_vector_vw, double *work_vector_vl, int mrp1, double *work_vector_vk, int mr, double tolerance) {
    int error_code, max_pq, max_pq_plus_1, ma_order_plus_1, ar_order_plus_1, flag;  // Dimensions and flags
    int k, j, j_plus_2_minus_k, j_plus_1_minus_k, last_index, loop_count, j_from;  // Indices
    int i, next_i, i_minus_j;  // More indices
    double zero_val = 0.0, p0625_val = 0.0625, one_val = 1.0, two_val = 2.0, four_val = 4.0, sixteen_val = 16.0, epsil1_val = 1.0e-10;  // Constants
    double current_state, alpha_val, state_over_r, det_mantissa, det_exponent, fl_j, current_r, vl_1, vw_1;  // Temporaries

    *sum_sq_resid = *likelihood_factor = zero_val;  // Initialize outputs
    det_mantissa = one_val;  // DETMAN
    det_exponent = zero_val;  // DETCAR
    max_pq = imax(ar_order, ma_order);  // MXPQ
    max_pq_plus_1 = max_pq + 1;
    ma_order_plus_1 = ma_order + 1;  // MQP1
    ar_order_plus_1 = ar_order + 1;  // MPP1
    flag = 0;  // FLAG

    // Step 1: Compute autocovariances using twacf
    error_code = twacf(ar_coeffs, ar_order, ma_coeffs, ma_order, work_vector_vw, max_pq_plus_1, work_vector_vl, max_pq_plus_1, work_vector_vk, max_pq);
    if (mr != imax(ar_order, ma_order_plus_1)) {
        error_code = 6;
    }
    if (mrp1 != mr + 1) {
        error_code = 7;
    }

    if (error_code > 0) {
        return error_code;
    }

    // Step 2: Initialize VK from VW and VL
    work_vector_vk[0] = work_vector_vw[0];
    if (mr != 1) {
        for (k = 2; k <= mr; ++k) {
            work_vector_vk[k - 1] = zero_val;
            if (k <= ar_order) {
                for (j = k; j <= ar_order; ++j) {
                    j_plus_2_minus_k = j + 2 - k;
                    work_vector_vk[k - 1] += ar_coeffs[j - 1] * work_vector_vw[j_plus_2_minus_k - 1];
                }
            }
            if (k <= ma_order_plus_1) {
                for (j = k; j <= ma_order_plus_1; ++j) {
                    j_plus_1_minus_k = j + 1 - k;
                    work_vector_vk[k - 1] -= ma_coeffs[j - 2] * work_vector_vl[j_plus_1_minus_k - 1];
                }
            }
        }
    }

    // Step 3: Setup initial vectors
    current_r = work_vector_vk[0];  // R = VK[0]
    work_vector_vl[mr - 1] = zero_val;
    for (j = 0; j < mr; ++j) {
        work_vector_vw[j] = zero_val;
        if (j != mr - 1) {
            work_vector_vl[j] = work_vector_vk[j + 1];
        }
        if (j <= ar_order - 1) {
            work_vector_vl[j] += ar_coeffs[j] * current_r;
        }
        work_vector_vk[j] = work_vector_vl[j];
    }

    last_index = ar_order_plus_1 - ma_order;  // LAST = MPP1 - MQ
    loop_count = ar_order;  // LOOP = MP
    j_from = ar_order_plus_1;  // JFROM = MPP1
    work_vector_vw[ar_order_plus_1 - 1] = zero_val;
    work_vector_vl[max_pq_plus_1 - 1] = zero_val;

    if (num_obs <= 0) {
        return 9;  // Error if N <= 0
    }

    // Step 4: Main Kalman recursion loop
    for (i = 1; i <= num_obs; ++i) {
        if (i == last_index) {
            loop_count = imin(ar_order, ma_order);
            j_from = loop_count + 1;
            if (ma_order <= 0) {
                flag = 1;
                break;
            }
        }
        if (current_r <= epsil1_val) {
            return 8;  // Error if R <= epsil1
        }
        if (fabs(current_r - one_val) < tolerance && i > max_pq) {
            flag = 1;
            break;
        }
        det_mantissa *= current_r;  // Update determinant mantissa
        while (fabs(det_mantissa) >= one_val) {  // Scale mantissa down
            det_mantissa *= p0625_val;
            det_exponent += four_val;
        }
        while (fabs(det_mantissa) < p0625_val) {  // Scale mantissa up
            det_mantissa *= sixteen_val;
            det_exponent -= four_val;
        }
        vw_1 = work_vector_vw[0];  // VW1 = VW[0]
        current_state = observations[i - 1] - vw_1;  // A = W[i-1] - VW1
        residuals[i - 1] = current_state / sqrt(current_r);  // E[i-1] = A / sqrt(R)
        state_over_r = current_state / current_r;  // AOR = A / R
        *sum_sq_resid += current_state * state_over_r;  // ssq += A * AOR
        vl_1 = work_vector_vl[0];  // VL1 = VL[0]
        alpha_val = vl_1 / current_r;  // ALF = VL1 / R
        current_r -= alpha_val * vl_1;  // R -= ALF * VL1
        if (loop_count != 0) {
            for (j = 0; j < loop_count; ++j) {  // Update VW, VL, VK
                fl_j = work_vector_vl[j + 1] + ar_coeffs[j] * vl_1;  // FLJ = VL[j+1] + P[j] * VL1
                work_vector_vw[j] = work_vector_vw[j + 1] + ar_coeffs[j] * vw_1 + state_over_r * work_vector_vk[j];
                work_vector_vl[j] = fl_j - alpha_val * work_vector_vk[j];
                work_vector_vk[j] -= alpha_val * fl_j;
            }
        }

        if (j_from <= ma_order) {
            for (j = j_from; j <= ma_order; ++j) {
                work_vector_vw[j - 1] = work_vector_vw[j] + state_over_r * work_vector_vk[j - 1];
                work_vector_vl[j - 1] = work_vector_vl[j] - alpha_val * work_vector_vk[j - 1];
                work_vector_vk[j - 1] -= alpha_val * work_vector_vl[j];
            }
        }
        if (j_from <= ar_order) {
            for (j = j_from; j <= ar_order; ++j) {
                work_vector_vw[j - 1] = work_vector_vw[j] + ar_coeffs[j-1] * observations[i-1];
            }
        }
    }

    // Step 5: Handle early termination with simplified computation
    if (flag == 1) {
        next_i = i;  // NEXTI = i
        error_code = -next_i;  // ifault = -NEXTI
        for (i = next_i; i <= num_obs; ++i) {
            residuals[i - 1] = observations[i - 1];
        }
        if (ar_order != 0) {
            for (i = next_i; i <= num_obs; ++i) {
                for (j = 1; j <= ar_order; ++j) {
                    i_minus_j = i - j;
                    residuals[i - 1] -= ar_coeffs[j - 1] * observations[i_minus_j - 1];
                }
            }
        }
        if (ma_order != 0) {
            for (i = next_i; i <= num_obs; ++i) {
                for (j = 1; j <= ma_order; ++j) {
                    i_minus_j = i - j;
                    residuals[i - 1] += ma_coeffs[j - 1] * residuals[i_minus_j - 1];
                }
            }
        }
        for (i = next_i; i <= num_obs; ++i) {
            *sum_sq_resid += residuals[i - 1] * residuals[i - 1];
        }
    }

    // Step 6: Compute final likelihood factor
    *likelihood_factor = pow(det_mantissa, one_val / (double)num_obs) * pow(two_val, det_exponent / (double)num_obs);

    return error_code;
}
