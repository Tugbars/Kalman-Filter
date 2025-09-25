```c
// SPDX-License-Identifier: BSD-3-Clause
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

/**
 * @brief Performs LU decomposition for rectangular matrices with partial pivoting.
 *
 * Decomposes a matrix A (M x N) into P*L*U, handling both tall (M > N) and wide (M < N)
 * matrices. Used in regression functions like linreg_multi for solving linear systems.
 *
 * @param matrix Input/output matrix (size M x N, modified in-place).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param pivot_indices Output pivot indices (length min(M, N)).
 * @return 0 on success.
 */
int rludecomp(double *matrix, int num_rows, int num_cols, int *pivot_indices) {
    // Step 1: Determine minimum dimension for pivoting
    int min_dimension = (num_rows < num_cols) ? num_rows : num_cols;

    // Step 2: Initialize pivot indices
    for (int k = 0; k < min_dimension; ++k) {
        pivot_indices[k] = k; // Initial row index
    }

    // Step 3: Handle different matrix shapes
    if (num_rows > num_cols) {
        // Tall matrix (M > N)
        for (int k = 0; k < num_cols; ++k) {
            // Find pivot
            double max_pivot_value = fabs(matrix[k * num_cols + k]);
            int max_pivot_index = k;
            for (int j = k + 1; j < num_rows; ++j) {
                if (max_pivot_value < fabs(matrix[j * num_cols + k])) {
                    max_pivot_value = fabs(matrix[j * num_cols + k]);
                    max_pivot_index = j;
                }
            }

            // Swap rows if necessary
            if (max_pivot_index != k) {
                int source_row = k * num_cols;
                int target_row = max_pivot_index * num_cols;
                int temp_pivot = pivot_indices[max_pivot_index];
                pivot_indices[max_pivot_index] = pivot_indices[k];
                pivot_indices[k] = temp_pivot;
                for (int j = 0; j < num_cols; j++) {
                    double temp = matrix[source_row + j];
                    matrix[source_row + j] = matrix[target_row + j];
                    matrix[target_row + j] = temp;
                }
            }

            // Eliminate below pivot
            int pivot_col = k * num_cols;
            double pivot_element = matrix[pivot_col + k];
            if (pivot_element != 0.0 && k < num_cols) {
                for (int j = k + 1; j < num_rows; ++j) {
                    int target_col = j * num_cols;
                    double multiplier = matrix[target_col + k] / pivot_element;
                    matrix[target_col + k] = multiplier; // Store in L
                    for (int l = k + 1; l < num_cols; ++l) {
                        matrix[target_col + l] -= multiplier * matrix[pivot_col + l];
                    }
                }
            }
        }
    } else if (num_rows < num_cols) {
        // Wide matrix (M < N)
        for (int k = 0; k < num_rows - 1; ++k) {
            // Find pivot
            double max_pivot_value = fabs(matrix[k * num_cols + k]);
            int max_pivot_index = k;
            for (int j = k + 1; j < num_rows; ++j) {
                if (max_pivot_value < fabs(matrix[j * num_cols + k])) {
                    max_pivot_value = fabs(matrix[j * num_cols + k]);
                    max_pivot_index = j;
                }
            }

            // Swap rows
            if (max_pivot_index != k) {
                int source_row = k * num_cols;
                int target_row = max_pivot_index * num_cols;
                int temp_pivot = pivot_indices[max_pivot_index];
                pivot_indices[max_pivot_index] = pivot_indices[k];
                pivot_indices[k] = temp_pivot;
                for (int j = 0; j < num_cols; j++) {
                    double temp = matrix[source_row + j];
                    matrix[source_row + j] = matrix[target_row + j];
                    matrix[target_row + j] = temp;
                }
            }

            // Eliminate
            int pivot_col = k * num_cols;
            double pivot_element = matrix[pivot_col + k];
            if (pivot_element != 0.0) {
                for (int j = k + 1; j < num_rows; ++j) {
                    int target_col = j * num_cols;
                    double multiplier = matrix[target_col + k] / pivot_element;
                    matrix[target_col + k] = multiplier;
                    for (int l = k + 1; l < num_cols; ++l) {
                        matrix[target_col + l] -= multiplier * matrix[pivot_col + l];
                    }
                }
            }
        }
    } else {
        // Square matrix (M == N)
        pludecomp(matrix, num_rows, pivot_indices); // Call square LU decomposition
    }

    return 0;
}

/**
 * @brief Solves a linear system Ax = b using LU decomposition.
 *
 * Solves for x in Ax = b using forward and backward substitution after LU decomposition.
 * Used in regression (linreg_multi) to compute coefficients.
 *
 * @param matrix LU-decomposed matrix (size N x N).
 * @param dimension Matrix dimension (N).
 * @param right_hand_side Right-hand side vector b (length N).
 * @param pivot_indices Pivot indices from ludecomp.
 * @param solution Output solution vector x (length N).
 */
void linsolve(double *matrix, int dimension, double *right_hand_side, int *pivot_indices, double *solution) {
    // Step 1: Allocate temporary array for intermediate solution
    double *intermediate_solution = (double*)malloc(sizeof(double) * dimension);

    // Step 2: Initialize arrays
    for (int i = 0; i < dimension; ++i) {
        intermediate_solution[i] = 0.0;
        solution[i] = 0.0;
        if (matrix[i * dimension + i] == 0.0) {
            printf("Warning: Matrix system may not have a unique solution\n");
        }
    }

    // Step 3: Forward substitution (L * y = Pb)
    intermediate_solution[0] = right_hand_side[pivot_indices[0]];
    for (int i = 1; i < dimension; ++i) {
        double sum = 0.0;
        int row_offset = i * dimension;
        for (int j = 0; j < i; ++j) {
            sum += intermediate_solution[j] * matrix[row_offset + j];
        }
        intermediate_solution[i] = right_hand_side[pivot_indices[i]] - sum;
    }

    // Step 4: Backward substitution (U * x = y)
    solution[dimension - 1] = intermediate_solution[dimension - 1] / matrix[dimension * dimension - 1];
    for (int i = dimension - 2; i >= 0; i--) {
        double sum = 0.0;
        int diag_offset = i * (dimension + 1);
        for (int j = i + 1; j < dimension; j++) {
            sum += matrix[diag_offset + (j - i)] * solution[j];
        }
        solution[i] = (intermediate_solution[i] - sum) / matrix[diag_offset];
    }

    // Step 5: Free temporary array
    free(intermediate_solution);
}

/**
 * @brief Computes the inverse of a square matrix using LU decomposition.
 *
 * Inverts matrix A by solving Ax_i = e_i for each unit vector e_i.
 *
 * @param matrix LU-decomposed matrix (size N x N).
 * @param dimension Matrix dimension (N).
 * @param pivot_indices Pivot indices from ludecomp.
 * @param inverse_matrix Output inverse matrix (size N x N).
 */
void minverse(double *matrix, int dimension, int *pivot_indices, double *inverse_matrix) {
    // Step 1: Allocate temporary arrays
    double *unit_vector = (double*)malloc(sizeof(double) * dimension); // Unit vector e_i
    double *column_solution = (double*)malloc(sizeof(double) * dimension); // Solution x_i

    // Step 2: Initialize temporary arrays
    for (int i = 0; i < dimension; ++i) {
        unit_vector[i] = 0.0;
        column_solution[i] = 0.0;
    }

    // Step 3: Solve for each column of inverse
    for (int i = 0; i < dimension; ++i) {
        unit_vector[i] = 1.0; // Set e_i
        linsolve(matrix, dimension, unit_vector, pivot_indices, column_solution); // Solve Ax_i = e_i
        int column_offset = i;
        for (int j = 0; j < dimension; ++j) {
            inverse_matrix[column_offset] = column_solution[j]; // Store column
            column_offset += dimension;
        }
        unit_vector[i] = 0.0; // Reset e_i
    }

    // Step 4: Free temporary arrays
    free(unit_vector);
    free(column_solution);
}

/**
 * @brief Performs QR decomposition of a matrix.
 *
 * Decomposes A (M x N, M >= N) into Q and R using Householder transformations.
 * Used in lls_qr for regression.
 *
 * @param matrix Input/output matrix (M x N, modified).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param beta_vector Output Householder scalars (length N).
 */
void qrdecomp(double *matrix, int num_rows, int num_cols, double *beta_vector) {
    // Step 1: Validate input
    if (num_rows < num_cols) {
        printf("Error: Rows (M) must be >= Columns (N) for QR decomposition\n");
        exit(1);
    }

    // Step 2: Allocate temporary arrays
    double *householder_vector = (double*)malloc(sizeof(double) * num_rows); // Householder vector
    double *temp_column = (double*)malloc(sizeof(double) * num_rows); // Temporary column
    double *temp_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols); // Temporary matrix
    double *temp_result = (double*)malloc(sizeof(double) * num_rows * num_cols); // Intermediate result

    // Step 3: Householder transformations
    for (int j = 0; j < num_cols; ++j) {
        // Extract column j starting from row j
        for (int i = j; i < num_rows; ++i) {
            householder_vector[i - j] = matrix[i * num_cols + j];
        }

        // Compute Householder vector and beta
        double beta = house_1(householder_vector, num_rows - j, householder_vector);
        beta_vector[j] = beta;

        // Apply transformation to submatrix
        int submatrix_rows = num_cols - j;
        for (int i = j; i < num_rows; i++) {
            int row_offset = i * num_cols;
            int temp_offset = 0;
            for (int k = j; k < num_cols; k++) {
                temp_matrix[temp_offset + i - j] = matrix[row_offset + k];
                temp_offset += (num_rows - j);
            }
        }

        mmult(temp_matrix, householder_vector, temp_result, submatrix_rows, num_rows - j, 1);
        scale(temp_result, submatrix_rows, 1, beta);
        mmult(householder_vector, temp_result, temp_matrix, num_rows - j, 1, submatrix_rows);

        for (int i = j; i < num_rows; i++) {
            int row_offset = i * num_cols;
            for (int k = j; k < num_cols; k++) {
                matrix[row_offset + k] -= temp_matrix[(i - j) * submatrix_rows + k - j];
            }
        }

        // Store Householder vector in lower part of matrix
        if (j < num_rows) {
            for (int i = j + 1; i < num_rows; ++i) {
                matrix[i * num_cols + j] = householder_vector[i - j];
            }
        }
    }

    // Step 4: Free temporary arrays
    free(householder_vector);
    free(temp_column);
    free(temp_matrix);
    free(temp_result);
}

/**
 * @brief Extracts Q and R matrices from QR decomposition.
 *
 * Reconstructs Q (M x N) and R (N x N) from the decomposed matrix and beta vector.
 *
 * @param matrix Decomposed matrix from qrdecomp.
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param beta_vector Householder scalars from qrdecomp.
 * @param q_matrix Output orthogonal matrix Q (M x N).
 * @param r_matrix Output upper triangular matrix R (N x N).
 */
void getQR(double *matrix, int num_rows, int num_cols, double *beta_vector, double *q_matrix, double *r_matrix) {
    // Step 1: Allocate temporary arrays
    double *householder_vector = (double*)malloc(sizeof(double) * num_rows);
    double *temp_column = (double*)malloc(sizeof(double) * num_rows);
    double *temp_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols);
    double *temp_result = (double*)malloc(sizeof(double) * num_rows * num_cols);

    // Step 2: Extract R (upper triangular part of A)
    for (int i = 0; i < num_cols; ++i) {
        int row_offset = i * num_cols;
        for (int j = 0; j < num_cols; ++j) {
            if (i > j) {
                r_matrix[row_offset + j] = 0.0;
            } else {
                r_matrix[row_offset + j] = matrix[row_offset + j];
            }
        }
    }

    // Step 3: Initialize Q as identity
    for (int i = 0; i < num_rows; ++i) {
        int row_offset = i * num_cols;
        for (int j = 0; j < num_cols; ++j) {
            q_matrix[row_offset + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Step 4: Apply Householder reflections to construct Q
    for (int j = num_cols - 1; j >= 0; --j) {
        householder_vector[0] = 1.0;
        for (int i = j + 1; i < num_rows; ++i) {
            householder_vector[i - j] = matrix[i * num_cols + j];
        }

        int submatrix_rows = num_cols - j;
        for (int i = j; i < num_rows; i++) {
            int row_offset = i * num_cols;
            int temp_offset = 0;
            for (int k = j; k < num_cols; k++) {
                temp_matrix[temp_offset + i - j] = q_matrix[row_offset + k];
                temp_offset += (num_rows - j);
            }
        }

        mmult(temp_matrix, householder_vector, temp_result, submatrix_rows, num_rows - j, 1);
        scale(temp_result, submatrix_rows, 1, beta_vector[j]);
        mmult(householder_vector, temp_result, temp_matrix, num_rows - j, 1, submatrix_rows);

        for (int i = j; i < num_rows; i++) {
            int row_offset = i * num_cols;
            for (int k = j; k < num_cols; k++) {
                q_matrix[row_offset + k] -= temp_matrix[(i - j) * submatrix_rows + k - j];
            }
        }
    }

    // Step 5: Free temporary arrays
    free(householder_vector);
    free(temp_column);
    free(temp_matrix);
    free(temp_result);
}

/**
 * @brief Computes the singular value decomposition (SVD) of a matrix.
 *
 * Decomposes A (M x N, M >= N) into U * Sigma * V^T, used in lls_svd2 for regression.
 *
 * @param matrix Input matrix (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param u_matrix Output left singular vectors (M x N).
 * @param v_matrix Output right singular vectors (N x N).
 * @param singular_values Output singular values (length N).
 * @return 0 on success, -1 if M < N, 15 if convergence fails.
 */
int svd(double *matrix, int num_rows, int num_cols, double *u_matrix, double *v_matrix, double *singular_values) {
    // Step 1: Validate input
    if (num_rows < num_cols) {
        printf("Error: Rows (M) must be >= Columns (N) for SVD\n");
        return -1;
    }

    // Step 2: Allocate temporary array
    double *off_diagonal = (double*)malloc(sizeof(double) * num_cols);
    int error_code = 0;
    double machine_epsilon = macheps();
    double max_norm = 0.0;

    // Step 3: Copy input matrix to U
    for (int i = 0; i < num_rows * num_cols; ++i) {
        u_matrix[i] = matrix[i];
    }

    // Step 4: Bidiagonalization using Householder transformations
    for (int i = 0; i < num_cols; ++i) {
        int l = i + 1;
        off_diagonal[i] = max_norm;
        double column_norm = 0.0;
        double scale = 0.0;

        if (i < num_rows) {
            // Householder transformation on column i
            for (int k = i; k < num_rows; ++k) {
                scale += fabs(u_matrix[k * num_cols + i]);
            }
            if (scale != 0.0) {
                double sum_squares = 0.0;
                for (int k = i; k < num_rows; ++k) {
                    int offset = k * num_cols;
                    u_matrix[offset + i] /= scale;
                    sum_squares += u_matrix[offset + i] * u_matrix[offset + i];
                }
                double pivot = u_matrix[i * num_cols + i];
                double norm_sign = (pivot < 0) ? sqrt(sum_squares) : -sqrt(sum_squares);
                double householder_scalar = pivot * norm_sign - sum_squares;
                u_matrix[i * num_cols + i] = pivot - norm_sign;

                if (i < num_cols - 1) {
                    for (int j = l; j < num_cols; ++j) {
                        double sum = 0.0;
                        for (int k = i; k < num_rows; ++k) {
                            sum += u_matrix[k * num_cols + i] * u_matrix[k * num_cols + j];
                        }
                        double factor = sum / householder_scalar;
                        for (int k = i; k < num_rows; ++k) {
                            u_matrix[k * num_cols + j] += factor * u_matrix[k * num_cols + i];
                        }
                    }
                }
                for (int k = i; k < num_rows; ++k) {
                    u_matrix[k * num_cols + i] *= scale;
                }
            }
        }
        singular_values[i] = scale * norm_sign;
        column_norm = 0.0;
        scale = 0.0;

        if (i < num_rows && i != num_cols - 1) {
            // Householder transformation on row i
            int row_offset = i * num_cols;
            for (int k = l; k < num_cols; ++k) {
                scale += fabs(u_matrix[row_offset + k]);
            }
            if (scale != 0.0) {
                double sum_squares = 0.0;
                for (int k = l; k < num_cols; ++k) {
                    u_matrix[row_offset + k] /= scale;
                    sum_squares += u_matrix[row_offset + k] * u_matrix[row_offset + k];
                }
                double pivot = u_matrix[row_offset + l];
                double norm_sign = (pivot < 0) ? sqrt(sum_squares) : -sqrt(sum_squares);
                double householder_scalar = pivot * norm_sign - sum_squares;
                u_matrix[row_offset + l] = pivot - norm_sign;
                for (int k = l; k < num_cols; ++k) {
                    off_diagonal[k] = u_matrix[row_offset + k] / householder_scalar;
                }

                for (int j = l; j < num_rows; j++) {
                    double sum = 0.0;
                    int offset = j * num_cols;
                    for (int k = l; k < num_cols; k++) {
                        sum += u_matrix[offset + k] * u_matrix[row_offset + k];
                    }
                    for (int k = l; k < num_cols; k++) {
                        u_matrix[offset + k] += sum * off_diagonal[k];
                    }
                }
                for (int k = l; k < num_cols; k++) {
                    u_matrix[row_offset + k] *= scale;
                }
            }
        }

        double temp_norm = fabs(singular_values[i]) + fabs(off_diagonal[i]);
        if (max_norm < temp_norm) {
            max_norm = temp_norm;
        }
    }

    // Step 5: Accumulate right-hand transformations for V
    for (int i = num_cols - 1; i >= 0; --i) {
        int row_offset = i * num_cols;
        if (i < num_cols - 1) {
            if (norm_sign != 0.0) {
                double householder_scalar = u_matrix[row_offset + i + 1] * norm_sign;
                for (int j = l; j < num_cols; ++j) {
                    v_matrix[j * num_cols + i] = u_matrix[row_offset + j] / householder_scalar;
                }
                for (int j = l; j < num_cols; ++j) {
                    double sum = 0.0;
                    for (int k = l; k < num_cols; ++k) {
                        sum += u_matrix[row_offset + k] * v_matrix[k * num_cols + j];
                    }
                    for (int k = l; k < num_cols; ++k) {
                        v_matrix[k * num_cols + j] += sum * v_matrix[k * num_cols + i];
                    }
                }
            }
            for (int j = l; j < num_cols; ++j) {
                v_matrix[row_offset + j] = v_matrix[j * num_cols + i] = 0.0;
            }
        }
        v_matrix[row_offset + i] = 1.0;
        norm_sign = off_diagonal[i];
        l = i;
    }

    // Step 6: Accumulate left-hand transformations for U
    for (int i = num_cols - 1; i >= 0; --i) {
        int row_offset = i * num_cols;
        l = i + 1;
        norm_sign = singular_values[i];

        if (i < num_cols - 1) {
            for (int j = l; j < num_cols; ++j) {
                u_matrix[row_offset + j] = 0.0;
            }
        }

        if (norm_sign != 0.0) {
            if (i != num_cols - 1) {
                for (int j = l; j < num_cols; ++j) {
                    double sum = 0.0;
                    for (int k = l; k < num_rows; ++k) {
                        sum += u_matrix[k * num_cols + i] * u_matrix[k * num_cols + j];
                    }
                    double factor = (sum / u_matrix[row_offset + i]) / norm_sign;
                    for (int k = i; k < num_rows; ++k) {
                        u_matrix[k * num_cols + j] += factor * u_matrix[k * num_cols + i];
                    }
                }
            }
            for (int j = i; j < num_rows; ++j) {
                u_matrix[j * num_cols + i] /= norm_sign;
            }
        } else {
            for (int j = i; j < num_rows; ++j) {
                u_matrix[j * num_cols + i] = 0.0;
            }
        }
        u_matrix[row_offset + i] += 1.0;
    }

    // Step 7: Diagonalize bidiagonal matrix
    double epsilon = machine_epsilon * max_norm;
    for (int k = num_cols - 1; k >= 0; --k) {
        int iteration = 0;
        while (1) {
            iteration++;
            if (iteration > SVDMAXITER) {
                printf("Error: SVD convergence not achieved\n");
                free(off_diagonal);
                return 15;
            }

            int cancel = 1;
            int l;
            for (l = k; l >= 0; --l) {
                if (fabs(off_diagonal[l]) <= epsilon) {
                    cancel = 0; // Convergence
                    break;
                }
                if (fabs(singular_values[l - 1]) <= epsilon) {
                    break;
                }
            }
            if (cancel) {
                double cos_theta = 0.0, sin_theta = 1.0;
                int l_minus_1 = l - 1;
                for (int i = l; i <= k; ++i) {
                    double f = sin_theta * off_diagonal[i];
                    off_diagonal[i] *= cos_theta;
                    if (fabs(f) <= epsilon) {
                        break;
                    }
                    double g = singular_values[i];
                    double h = hypot(f, g);
                    singular_values[i] = h;
                    cos_theta = g / h;
                    sin_theta = -f / h;
                    for (int j = 0; j < num_rows; ++j) {
                        int offset = j * num_cols;
                        double y = u_matrix[offset + l_minus_1];
                        double z = u_matrix[offset + i];
                        u_matrix[offset + l_minus_1] = y * cos_theta + z * sin_theta;
                        u_matrix[offset + i] = z * cos_theta - y * sin_theta;
                    }
                }
            }
            double z = singular_values[k];
            if (l != k) {
                double x = singular_values[l];
                double y = singular_values[k - 1];
                double g = off_diagonal[k - 1];
                double h = off_diagonal[k];
                double f = 0.5 * (((g + z) / h) * ((g - z) / y) + y / h - h / y);
                g = hypot(f, 1.0);
                double temp = (f < 0.0) ? f - g : f + g;
                f = x - (z / x) * z + (h / x) * (y / temp - h);

                // QR transformation
                cos_theta = sin_theta = 1.0;
                for (int i = l + 1; i <= k; ++i) {
                    g = off_diagonal[i];
                    y = singular_values[i];
                    h = sin_theta * g;
                    g = cos_theta * g;
                    off_diagonal[i - 1] = z = hypot(f, h);
                    cos_theta = f / z;
                    sin_theta = h / z;
                    f = x * cos_theta + g * sin_theta;
                    g = g * cos_theta - x * sin_theta;
                    h = y * sin_theta;
                    y *= cos_theta;
                    for (int j = 0; j < num_cols; ++j) {
                        int offset = j * num_cols;
                        x = v_matrix[offset + i - 1];
                        z = v_matrix[offset + i];
                        v_matrix[offset + i - 1] = x * cos_theta + z * sin_theta;
                        v_matrix[offset + i] = z * cos_theta - x * sin_theta;
                    }
                    singular_values[i - 1] = z = hypot(f, h);
                    if (z != 0.0) {
                        cos_theta = f / z;
                        sin_theta = h / z;
                    }
                    f = cos_theta * g + sin_theta * y;
                    x = cos_theta * y - sin_theta * g;
                    for (int j = 0; j < num_rows; ++j) {
                        int offset = j * num_cols;
                        y = u_matrix[offset + i - 1];
                        z = u_matrix[offset + i];
                        u_matrix[offset + i - 1] = y * cos_theta + z * sin_theta;
                        u_matrix[offset + i] = z * cos_theta - y * sin_theta;
                    }
                }
                off_diagonal[l] = 0.0;
                off_diagonal[k] = f;
                singular_values[k] = x;
            } else {
                // Convergence
                if (z < 0.0) {
                    singular_values[k] = -z;
                    for (int j = 0; j < num_cols; j++) {
                        v_matrix[j * num_cols + k] = -v_matrix[j * num_cols + k];
                    }
                }
                break;
            }
        }
    }

    // Step 8: Sort singular values and vectors
    svd_sort(u_matrix, num_rows, num_cols, v_matrix, singular_values);

    // Step 9: Free temporary array
    free(off_diagonal);

    return error_code;
}

/**
 * @brief Computes the rank of a matrix using SVD.
 *
 * Determines the numerical rank by counting singular values above a tolerance.
 *
 * @param matrix Input matrix (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @return Matrix rank, or -1 if SVD fails.
 */
int rank(double *matrix, int num_rows, int num_cols) {
    // Step 1: Allocate temporary arrays for SVD
    double *u_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols);
    double *v_matrix = (double*)malloc(sizeof(double) * num_cols * num_cols);
    double *singular_values = (double*)malloc(sizeof(double) * num_cols);

    // Step 2: Determine matrix orientation
    int computed_rank;
    double max_dimension = (double)(num_rows > num_cols ? num_rows : num_cols);
    double machine_epsilon = macheps();

    // Step 3: Compute SVD
    if (num_rows < num_cols) {
        double *transposed_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols);
        mtranspose(matrix, num_rows, num_cols, transposed_matrix);
        computed_rank = rank_c(transposed_matrix, num_cols, num_rows);
        free(transposed_matrix);
    } else {
        computed_rank = rank_c(matrix, num_rows, num_cols);
    }

    // Step 4: Free temporary arrays
    free(u_matrix);
    free(v_matrix);
    free(singular_values);

    return computed_rank;
}

/**
 * @brief Internal function to compute matrix rank using SVD.
 *
 * @param matrix Input matrix (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @return Matrix rank, or -1 if SVD fails.
 */
static int rank_c(double *matrix, int num_rows, int num_cols) {
    // Step 1: Allocate temporary arrays
    double *u_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols);
    double *v_matrix = (double*)malloc(sizeof(double) * num_cols * num_cols);
    double *singular_values = (double*)malloc(sizeof(double) * num_cols);

    // Step 2: Compute SVD
    int error_code = svd(matrix, num_rows, num_cols, u_matrix, v_matrix, singular_values);
    if (error_code != 0) {
        printf("Error: Failed to compute SVD\n");
        free(u_matrix);
        free(v_matrix);
        free(singular_values);
        return -1;
    }

    // Step 3: Calculate rank based on singular values
    double machine_epsilon = macheps();
    double max_dimension = (double)(num_rows > num_cols ? num_rows : num_cols);
    double tolerance = singular_values[0] * max_dimension * machine_epsilon;
    int computed_rank = 0;
    for (int i = 0; i < num_cols; ++i) {
        if (singular_values[i] > tolerance) {
            computed_rank++;
        }
    }

    // Step 4: Free temporary arrays
    free(u_matrix);
    free(v_matrix);
    free(singular_values);

    return computed_rank;
}

/**
 * @brief Solves a least squares problem using QR decomposition.
 *
 * Solves Ax = b for x (M x N matrix A, M >= N) using QR decomposition.
 * Used in linreg_multi for regression coefficients.
 *
 * @param matrix Matrix A (size M x N, typically X'X).
 * @param right_hand_side Vector b (size M x 1, typically X'y).
 * @param num_rows Rows of A (M).
 * @param num_cols Columns of A (N).
 * @param solution Output solution x (size N x 1).
 * @return Rank of the matrix A.
 */
int lls_qr(double *matrix, double *right_hand_side, int num_rows, int num_cols, double *solution) {
    // Step 1: Validate input
    if (num_rows < num_cols) {
        printf("Error: Rows (M) must be >= Columns (N) for lls_qr\n");
        exit(1);
    }

    // Step 2: Allocate temporary arrays
    double *beta_vector = (double*)malloc(sizeof(double) * num_cols);
    double *q_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols);
    double *r_matrix = (double*)malloc(sizeof(double) * num_cols * num_cols);
    double *temp_b = (double*)malloc(sizeof(double) * num_rows);

    // Step 3: Copy input matrix and right-hand side
    double *work_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols);
    memcpy(work_matrix, matrix, sizeof(double) * num_rows * num_cols);
    memcpy(temp_b, right_hand_side, sizeof(double) * num_rows);

    // Step 4: Perform QR decomposition
    qrdecomp(work_matrix, num_rows, num_cols, beta_vector);
    getQR(work_matrix, num_rows, num_cols, beta_vector, q_matrix, r_matrix);

    // Step 5: Solve QRx = b by computing Q^T b and solving Rx = Q^T b
    double *q_transpose_b = (double*)malloc(sizeof(double) * num_cols);
    mmult(q_matrix, temp_b, q_transpose_b, num_cols, num_rows, 1);

    // Step 6: Backward substitution on R
    for (int i = num_cols - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < num_cols; ++j) {
            sum += r_matrix[i * num_cols + j] * solution[j];
        }
        if (fabs(r_matrix[i * num_cols + i]) < 1e-10) {
            solution[i] = 0.0; // Handle near-zero diagonal
        } else {
            solution[i] = (q_transpose_b[i] - sum) / r_matrix[i * num_cols + i];
        }
    }

    // Step 7: Compute rank using SVD
    int matrix_rank = rank(work_matrix, num_rows, num_cols);

    // Step 8: Free temporary arrays
    free(beta_vector);
    free(q_matrix);
    free(r_matrix);
    free(temp_b);
    free(work_matrix);
    free(q_transpose_b);

    return matrix_rank;
}

/**
 * @brief Solves a least squares problem using normal equations.
 *
 * Solves (X'X)x = X'y using LU decomposition of X'X.
 *
 * @param matrix Matrix X'X (size M x N, typically N x N).
 * @param right_hand_side Vector X'y (size M x 1).
 * @param num_rows Rows of matrix (M).
 * @param num_cols Columns of matrix (N).
 * @param solution Output solution x (size N x 1).
 */
void lls_normal(double *matrix, double *right_hand_side, int num_rows, int num_cols, double *solution) {
    // Step 1: Allocate pivot array
    int *pivot_indices = (int*)malloc(sizeof(int) * num_cols);

    // Step 2: Perform LU decomposition
    ludecomp(matrix, num_cols, pivot_indices);

    // Step 3: Solve linear system
    linsolve(matrix, num_cols, right_hand_side, pivot_indices, solution);

    // Step 4: Free pivot array
    free(pivot_indices);
}

/**
 * @brief Solves a least squares problem using SVD.
 *
 * Solves Ax = b using singular value decomposition for numerical stability.
 *
 * @param matrix Matrix A (size M x N).
 * @param right_hand_side Vector b (size M x 1).
 * @param num_rows Rows of A (M).
 * @param num_cols Columns of A (N).
 * @param solution Output solution x (size N x 1).
 */
void lls_svd2(double *matrix, double *right_hand_side, int num_rows, int num_cols, double *solution) {
    // Step 1: Allocate temporary arrays for SVD
    double *u_matrix = (double*)malloc(sizeof(double) * num_rows * num_cols);
    double *v_matrix = (double*)malloc(sizeof(double) * num_cols * num_cols);
    double *singular_values = (double*)malloc(sizeof(double) * num_cols);
    double *temp_b = (double*)malloc(sizeof(double) * num_rows);

    // Step 2: Copy input
    memcpy(u_matrix, matrix, sizeof(double) * num_rows * num_cols);
    memcpy(temp_b, right_hand_side, sizeof(double) * num_rows);

    // Step 3: Compute SVD
    int error_code = svd(u_matrix, num_rows, num_cols, u_matrix, v_matrix, singular_values);
    if (error_code != 0) {
        printf("Error: SVD failed in lls_svd2\n");
        free(u_matrix);
        free(v_matrix);
        free(singular_values);
        free(temp_b);
        exit(1);
    }

    // Step 4: Compute U^T b
    double *u_transpose_b = (double*)malloc(sizeof(double) * num_cols);
    mmult(u_matrix, temp_b, u_transpose_b, num_cols, num_rows, 1);

    // Step 5: Solve for x = V * (Sigma^-1 * U^T b)
    for (int i = 0; i < num_cols; ++i) {
        if (fabs(singular_values[i]) > 1e-10) {
            u_transpose_b[i] /= singular_values[i]; // Divide by non-zero singular values
        } else {
            u_transpose_b[i] = 0.0; // Handle zero singular values
        }
    }
    mmult(v_matrix, u_transpose_b, solution, num_cols, num_cols, 1);

    // Step 6: Free temporary arrays
    free(u_matrix);
    free(v_matrix);
    free(singular_values);
    free(temp_b);
    free(u_transpose_b);
}

/**
 * @brief Computes Householder vector and beta for a vector.
 *
 * Used in QR decomposition for Householder transformations.
 *
 * @param input_vector Input vector (length N).
 * @param vector_length Length of the vector (N).
 * @param householder_vector Output Householder vector (length N).
 * @return Beta scalar for Householder transformation.
 */
static double house_1(double *input_vector, int vector_length, double *householder_vector) {
    // Step 1: Compute norm of subvector
    double *sigma = (double*)malloc(sizeof(double));
    if (vector_length > 1) {
        mmult(input_vector + 1, input_vector + 1, sigma, 1, vector_length - 1, 1);
    } else {
        *sigma = 0.0;
    }

    // Step 2: Initialize Householder vector
    householder_vector[0] = 1.0;
    for (int i = 1; i < vector_length; ++i) {
        householder_vector[i] = input_vector[i];
    }

    // Step 3: Compute beta
    double beta;
    if (*sigma == 0.0 && input_vector[0] >= 0.0) {
        beta = 0.0;
    } else if (*sigma == 0.0 && input_vector[0] < 0.0) {
        beta = -2.0;
    } else {
        double mu = sqrt(*sigma + input_vector[0] * input_vector[0]);
        if (input_vector[0] <= 0.0) {
            householder_vector[0] = input_vector[0] - mu;
        } else {
            householder_vector[0] = -(*sigma) / (input_vector[0] + mu);
        }
        double temp = householder_vector[0];
        beta = (2.0 * householder_vector[0] * householder_vector[0]) / (*sigma + householder_vector[0] * householder_vector[0]);
        for (int i = 0; i < vector_length; ++i) {
            householder_vector[i] /= temp;
        }
    }

    // Step 4: Free temporary array
    free(sigma);
    return beta;
}

/**
 * @brief Sorts SVD components (U, V, singular values) in descending order.
 *
 * Ensures singular values are sorted and corresponding vectors are reordered.
 *
 * @param u_matrix Left singular vectors (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param v_matrix Right singular vectors (N x N).
 * @param singular_values Singular values (length N).
 */
void svd_sort(double *u_matrix, int num_rows, int num_cols, double *v_matrix, double *singular_values) {
    // Step 1: Allocate temporary arrays
    double *temp_u = (double*)malloc(sizeof(double) * num_rows * num_cols);
    double *temp_v = (double*)malloc(sizeof(double) * num_cols * num_cols);
    double *temp_singular = (double*)malloc(sizeof(double) * num_cols);
    int *sorted_indices = (int*)malloc(sizeof(int) * num_cols);

    // Step 2: Copy inputs
    memcpy(temp_u, u_matrix, sizeof(double) * num_rows * num_cols);
    memcpy(temp_v, v_matrix, sizeof(double) * num_cols * num_cols);
    memcpy(temp_singular, singular_values, sizeof(double) * num_cols);

    // Step 3: Sort singular values in descending order
    sort1d(singular_values, num_cols, sorted_indices);

    // Step 4: Reorder U and V matrices
    for (int i = 0; i < num_cols; ++i) {
        singular_values[i] = temp_singular[sorted_indices[i]];
        for (int j = 0; j < num_rows; ++j) {
            u_matrix[j * num_cols + i] = temp_u[j * num_cols + sorted_indices[i]];
        }
        for (int j = 0; j < num_cols; ++j) {
            v_matrix[j * num_cols + i] = temp_v[j * num_cols + sorted_indices[i]];
        }
    }

    // Step 5: Free temporary arrays
    free(temp_u);
    free(temp_v);
    free(temp_singular);
    free(sorted_indices);
}

/**
 * @brief Generates random matrix using Marsaglia's method.
 *
 * Generates a matrix with normally distributed entries (used in rsvd).
 *
 * @param matrix Output matrix (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 */
void random_matrix(double *matrix, int num_rows, int num_cols) {
    // Step 1: Compute total elements
    int total_elements = num_rows * num_cols;
    // Step 2: Generate random values using Marsaglia's method
    marsaglia_generate(matrix, total_elements, 0.0, 1.0);
}

/**
 * @brief Generates normally distributed random numbers using Marsaglia's polar method.
 *
 * @param values Output array (length N).
 * @param array_length Length of the array (N).
 * @param mean Mean of the distribution.
 * @param std_dev Standard deviation of the distribution.
 * @return Pointer to values array.
 */
double* marsaglia_generate(double *values, int array_length, double mean, double std_dev) {
    // Step 1: Ensure even number for paired generation
    int paired_length = array_length + (array_length % 2);

    // Step 2: Generate pairs of normal random variables
    for (int i = 0; i < array_length - 1; i += 2) {
        double x, y, radius_squared, factor;
        do {
            x = 2.0 * rand() / (double)RAND_MAX - 1.0;
            y = 2.0 * rand() / (double)RAND_MAX - 1.0;
            radius_squared = x * x + y * y;
        } while (radius_squared >= 1.0 || radius_squared == 0.0);
        factor = sqrt(-2.0 * log(radius_squared) / radius_squared);
        values[i] = x * factor;
        values[i + 1] = y * factor;
    }

    // Step 3: Handle odd length
    if (paired_length != array_length) {
        double x, y, radius_squared, factor;
        do {
            x = 2.0 * rand() / (double)RAND_MAX - 1.0;
            y = 2.0 * rand() / (double)RAND_MAX - 1.0;
            radius_squared = x * x + y * y;
        } while (radius_squared >= 1.0 || radius_squared == 0.0);
        factor = sqrt(-2.0 * log(radius_squared) / radius_squared);
        values[array_length - 1] = x * factor;
    }

    // Step 4: Scale and shift values
    for (int i = 0; i < array_length; ++i) {
        values[i] = (values[i] * std_dev + mean);
    }
    return values;
}

/**
 * @brief Computes eigenvalues and eigenvectors of a symmetric matrix.
 *
 * Uses Householder reduction and QL iteration (used indirectly via svd).
 *
 * @param matrix Input symmetric matrix (N x N).
 * @param dimension Matrix dimension (N).
 * @param eigenvalues Output eigenvalues (length N).
 * @param eigenvectors Output eigenvectors (N x N).
 */
void eigensystem(double *matrix, int dimension, double *eigenvalues, double *eigenvectors) {
    // Step 1: Allocate temporary array
    double *off_diagonal = (double*)calloc(dimension, sizeof(double));

    // Step 2: Copy input matrix
    memcpy(eigenvectors, matrix, sizeof(double) * dimension * dimension);

    // Step 3: Householder reduction to tridiagonal form
    tred2(eigenvectors, dimension, eigenvalues, off_diagonal);

    // Step 4: QL iteration to compute eigenvalues and eigenvectors
    tqli(eigenvalues, dimension, off_diagonal, eigenvectors);

    // Step 5: Free temporary array
    free(off_diagonal);
}

/**
 * @brief Householder reduction to tridiagonal form.
 *
 * Reduces a symmetric matrix to tridiagonal form for eigenvalue computation.
 *
 * @param matrix Input/output matrix (N x N).
 * @param dimension Matrix dimension (N).
 * @param diagonal Output diagonal elements.
 * @param off_diagonal Output off-diagonal elements.
 */
static void tred2(double *matrix, int dimension, double *diagonal, double *off_diagonal) {
    // Step 1: Iterate from last row to second
    for (int i = dimension - 1; i > 0; --i) {
        int l = i - 1;
        double scale = 0.0;
        double h = 0.0;

        if (l > 0) {
            // Compute scale of row i
            for (int k = 0; k <= l; ++k) {
                scale += fabs(matrix[i * dimension + k]);
            }
            if (scale == 0.0) {
                off_diagonal[i] = matrix[i * dimension + l];
            } else {
                // Normalize row
                for (int k = 0; k <= l; ++k) {
                    matrix[i * dimension + k] /= scale;
                    h += matrix[i * dimension + k] * matrix[i * dimension + k];
                }
                double f = matrix[i * dimension + l];
                double g = (f >= 0.0) ? -sqrt(h) : sqrt(h);
                off_diagonal[i] = scale * g;
                h -= f * g;
                matrix[i * dimension + l] = f - g;
                f = 0.0;

                // Apply Householder transformation
                for (int j = 0; j <= l; ++j) {
                    matrix[j * dimension + i] = matrix[i * dimension + j] / h;
                    double g_local = 0.0;
                    for (int k = 0; k <= j; ++k) {
                        g_local += matrix[j * dimension + k] * matrix[i * dimension + k];
                    }
                    for (int k = j + 1; k <= l; ++k) {
                        g_local += matrix[k * dimension + j] * matrix[i * dimension + k];
                    }
                    off_diagonal[j] = g_local / h;
                    f += off_diagonal[j] * matrix[i * dimension + j];
                }
                double hh = f / (h + h);
                for (int j = 0; j <= l; ++j) {
                    f = matrix[i * dimension + j];
                    off_diagonal[j] = g_local = off_diagonal[j] - hh * f;
                    for (int k = 0; k <= j; ++k) {
                        matrix[j * dimension + k] -= (f * off_diagonal[k] + g_local * matrix[i * dimension + k]);
                    }
                }
            }
        } else {
            off_diagonal[i] = matrix[i * dimension + l];
        }
        diagonal[i] = h;
    }

    // Step 2: Handle first row
    diagonal[0] = 0.0;
    off_diagonal[0] = 0.0;
    for (int i = 0; i < dimension; ++i) {
        int l = i - 1;
        if (diagonal[i]) {
            for (int j = 0; j <= l; ++j) {
                double g = 0.0;
                for (int k = 0; k <= l; ++k) {
                    g += matrix[i * dimension + k] * matrix[k * dimension + j];
                }
                for (int k = 0; k <= l; ++k) {
                    matrix[k * dimension + j] -= g * matrix[k * dimension + i];
                }
            }
        }
        diagonal[i] = matrix[i * dimension + i];
        matrix[i * dimension + i] = 1.0;
        for (int j = 0; j <= l; ++j) {
            matrix[j * dimension + i] = matrix[i * dimension + j] = 0.0;
        }
    }
}

/**
 * @brief QL iteration for eigenvalue computation.
 *
 * Computes eigenvalues and eigenvectors of a tridiagonal matrix.
 *
 * @param diagonal Diagonal elements (length N).
 * @param dimension Matrix dimension (N).
 * @param off_diagonal Off-diagonal elements (length N).
 * @param eigenvectors Output eigenvectors (N x N).
 */
static void tqli(double *diagonal, int dimension, double *off_diagonal, double *eigenvectors) {
    // Step 1: Shift off-diagonal elements
    for (int i = 1; i < dimension; ++i) {
        off_diagonal[i - 1] = off_diagonal[i];
    }
    off_diagonal[dimension - 1] = 0.0;

    // Step 2: QL iteration
    for (int l = 0; l < dimension; ++l) {
        int iteration = 0;
        do {
            int m;
            for (m = l; m < dimension - 1; ++m) {
                double dd = fabs(diagonal[m]) + fabs(diagonal[m + 1]);
                if (fabs(off_diagonal[m]) + dd == dd) {
                    break;
                }
            }
            if (m != l) {
                if (iteration++ == 30) {
                    printf("Error: Too many iterations in tqli\n");
                    return;
                }
                double g = (diagonal[l + 1] - diagonal[l]) / (2.0 * off_diagonal[l]);
                double r = pythag(g, 1.0);
                g = diagonal[m] - diagonal[l] + off_diagonal[l] / (g + signx(r) * r);
                double sin_theta = 1.0, cos_theta = 1.0, p = 0.0;
                for (int i = m - 1; i >= l; --i) {
                    double f = sin_theta * off_diagonal[i];
                    double b = cos_theta * off_diagonal[i];
                    off_diagonal[i + 1] = r = pythag(f, g);
                    if (r == 0.0) {
                        diagonal[i + 1] -= p;
                        off_diagonal[m] = 0.0;
                        break;
                    }
                    sin_theta = f / r;
                    cos_theta = g / r;
                    g = diagonal[i + 1] - p;
                    r = (diagonal[i] - g) * sin_theta + 2.0 * cos_theta * b;
                    diagonal[i + 1] = g + (p = sin_theta * r);
                    g = cos_theta * r - b;
                    for (int k = 0; k < dimension; ++k) {
                        f = eigenvectors[k * dimension + i + 1];
                        eigenvectors[k * dimension + i + 1] = sin_theta * eigenvectors[k * dimension + i] + cos_theta * f;
                        eigenvectors[k * dimension + i] = cos_theta * eigenvectors[k * dimension + i] - sin_theta * f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                diagonal[l] -= p;
                off_diagonal[l] = g;
                off_diagonal[m] = 0.0;
            }
        } while (m != l);
    }
}

/**
 * @brief Computes the Pythagorean sum of two values.
 *
 * Avoids overflow in computing sqrt(a^2 + b^2).
 *
 * @param a First value.
 * @param b Second value.
 * @return Pythagorean sum.
 */
static double pythag(double a, double b) {
    double abs_a = fabs(a);
    double abs_b = fabs(b);
    if (abs_a > abs_b) {
        return abs_a * sqrt(1.0 + (abs_b / abs_a) * (abs_b / abs_a));
    } else {
        return (abs_b == 0.0) ? 0.0 : abs_b * sqrt(1.0 + (abs_a / abs_b) * (abs_a / abs_b));
    }
}

/**
 * @brief Creates an identity matrix.
 *
 * @param matrix Output matrix (N x N).
 * @param dimension Matrix dimension (N).
 */
void eye(double *matrix, int dimension) {
    // Step 1: Fill matrix with zeros, set diagonal to 1
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            matrix[i * dimension + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

/**
 * @brief Creates a scaled identity matrix.
 *
 * @param matrix Output matrix (N x N).
 * @param dimension Matrix dimension (N).
 * @param scalar Scaling factor for diagonal.
 */
void eye_scale(double *matrix, int dimension, double scalar) {
    // Step 1: Fill matrix with zeros, set diagonal to scalar
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            matrix[i * dimension + j] = (i == j) ? scalar : 0.0;
        }
    }
}

/**
 * @brief Adds zero padding to a matrix.
 *
 * Extends matrix with zeros for recursive operations.
 *
 * @param input_matrix Input matrix (rows x cols).
 * @param num_rows Number of rows.
 * @param num_cols Number of columns.
 * @param zero_rows Number of zero rows to add.
 * @param zero_cols Number of zero columns to add.
 * @param output_matrix Output padded matrix.
 */
void add_zero_pad(double *input_matrix, int num_rows, int num_cols, int zero_rows, int zero_cols, double *output_matrix) {
    // Step 1: Compute new dimensions
    int padded_rows = num_rows + zero_rows;
    int padded_cols = num_cols + zero_cols;

    // Step 2: Copy input and add zeros
    for (int i = 0; i < num_rows; ++i) {
        int output_row_offset = i * padded_cols;
        int input_row_offset = i * num_cols;
        for (int j = 0; j < num_cols; ++j) {
            output_matrix[output_row_offset + j] = input_matrix[input_row_offset + j];
        }
        for (int j = num_cols; j < padded_cols; ++j) {
            output_matrix[output_row_offset + j] = 0.0;
        }
    }
    for (int i = num_rows; i < padded_rows; ++i) {
        int output_row_offset = i * padded_cols;
        for (int j = 0; j < padded_cols; ++j) {
            output_matrix[output_row_offset + j] = 0.0;
        }
    }
}

/**
 * @brief Removes zero padding from a matrix.
 *
 * Extracts original matrix from padded matrix.
 *
 * @param input_matrix Padded matrix.
 * @param num_rows Number of rows in padded matrix.
 * @param num_cols Number of columns in padded matrix.
 * @param zero_rows Number of zero rows to remove.
 * @param zero_cols Number of zero columns to remove.
 * @param output_matrix Output unpadded matrix.
 */
void remove_zero_pad(double *input_matrix, int num_rows, int num_cols, int zero_rows, int zero_cols, double *output_matrix) {
    // Step 1: Compute original dimensions
    int original_rows = num_rows - zero_rows;
    int original_cols = num_cols - zero_cols;

    // Step 2: Copy non-padded elements
    for (int i = 0; i < original_rows; ++i) {
        int output_row_offset = i * original_cols;
        int input_row_offset = i * num_cols;
        for (int j = 0; j < original_cols; ++j) {
            output_matrix[output_row_offset + j] = input_matrix[input_row_offset + j];
        }
    }
}

