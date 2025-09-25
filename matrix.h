/*
 * matrix_refactored.h
 *
 * Header file for refactored matrix operation functions from matrix.c,
 * used as dependencies for emle.c and talg.c.
 */

#ifndef MATRIX_REFACTORED_H_
#define MATRIX_REFACTORED_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

#define SVDMAXITER 30

#ifdef __cplusplus
extern "C" {
#endif

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

/**
 * @brief Transposes a matrix.
 *
 * @param input_matrix Input matrix (size rows x cols).
 * @param num_rows Number of rows.
 * @param num_cols Number of columns.
 * @param output_matrix Transposed matrix (output, size cols x rows).
 */
void mtranspose(double *input_matrix, int num_rows, int num_cols, double *output_matrix);

/**
 * @brief Multiplies two matrices.
 *
 * @param matrix_a First matrix (m x n).
 * @param matrix_b Second matrix (n x p).
 * @param matrix_result Result matrix (m x p).
 * @param num_rows_a Rows of matrix_a.
 * @param num_cols_a Columns of matrix_a.
 * @param num_cols_b Columns of matrix_b.
 */
void mmult(double *matrix_a, double *matrix_b, double *matrix_result, int num_rows_a, int num_cols_a, int num_cols_b);

/**
 * @brief Scales a matrix by a scalar.
 *
 * @param matrix Input/output matrix (m x n).
 * @param num_rows Number of rows.
 * @param num_cols Number of columns.
 * @param scalar Scaling factor.
 */
void scale(double *matrix, int num_rows, int num_cols, double scalar);

/**
 * @brief Sorts a 1D array in descending order and returns sorted indices.
 *
 * @param array Input array (length N).
 * @param length Array length.
 * @param indices Output sorted indices (length N).
 */
void sort1d(double *array, int length, int *indices);

/**
 * @brief Computes machine epsilon.
 *
 * @return Machine epsilon value.
 */
double macheps(void);

/**
 * @brief Computes the sign of a value for QL iteration.
 *
 * @param val Input value.
 * @return Sign of the value.
 */
double signx(double val);

/**
 * @brief Performs LU decomposition for square matrices with partial pivoting.
 *
 * @param matrix Input/output matrix (size N x N, modified).
 * @param dimension Matrix dimension.
 * @param pivot_indices Pivot indices (output, length N).
 */
void ludecomp(double *matrix, int dimension, int *pivot_indices);

/**
 * @brief Performs LU decomposition for rectangular matrices with partial pivoting.
 *
 * @param matrix Input/output matrix (size M x N, modified).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param pivot_indices Output pivot indices (length min(M, N)).
 * @return 0 on success.
 */
int rludecomp(double *matrix, int num_rows, int num_cols, int *pivot_indices);

/**
 * @brief Solves a linear system Ax = b using LU decomposition.
 *
 * @param matrix LU-decomposed matrix (size N x N).
 * @param dimension Matrix dimension (N).
 * @param right_hand_side Right-hand side vector b (length N).
 * @param pivot_indices Pivot indices from ludecomp.
 * @param solution Output solution vector x (length N).
 */
void linsolve(double *matrix, int dimension, double *right_hand_side, int *pivot_indices, double *solution);

/**
 * @brief Computes the inverse of a square matrix using LU decomposition.
 *
 * @param matrix LU-decomposed matrix (size N x N).
 * @param dimension Matrix dimension (N).
 * @param pivot_indices Pivot indices from ludecomp.
 * @param inverse_matrix Output inverse matrix (size N x N).
 */
void minverse(double *matrix, int dimension, int *pivot_indices, double *inverse_matrix);

/**
 * @brief Performs QR decomposition of a matrix.
 *
 * @param matrix Input/output matrix (M x N, M >= N, modified).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param beta_vector Output Householder scalars (length N).
 */
void qrdecomp(double *matrix, int num_rows, int num_cols, double *beta_vector);

/**
 * @brief Extracts Q and R matrices from QR decomposition.
 *
 * @param matrix Decomposed matrix from qrdecomp.
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param beta_vector Householder scalars from qrdecomp.
 * @param q_matrix Output orthogonal matrix Q (M x N).
 * @param r_matrix Output upper triangular matrix R (N x N).
 */
void getQR(double *matrix, int num_rows, int num_cols, double *beta_vector, double *q_matrix, double *r_matrix);

/**
 * @brief Computes the singular value decomposition (SVD) of a matrix.
 *
 * @param matrix Input matrix (M x N, M >= N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param u_matrix Output left singular vectors (M x N).
 * @param v_matrix Output right singular vectors (N x N).
 * @param singular_values Output singular values (length N).
 * @return 0 on success, -1 if M < N, 15 if convergence fails.
 */
int svd(double *matrix, int num_rows, int num_cols, double *u_matrix, double *v_matrix, double *singular_values);

/**
 * @brief Computes the rank of a matrix using SVD.
 *
 * @param matrix Input matrix (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @return Matrix rank, or -1 if SVD fails.
 */
int rank(double *matrix, int num_rows, int num_cols);

/**
 * @brief Solves a least squares problem using QR decomposition.
 *
 * @param matrix Matrix A (size M x N, M >= N).
 * @param right_hand_side Vector b (size M x 1).
 * @param num_rows Rows of A (M).
 * @param num_cols Columns of A (N).
 * @param solution Output solution x (size N x 1).
 * @return Rank of the matrix A.
 */
int lls_qr(double *matrix, double *right_hand_side, int num_rows, int num_cols, double *solution);

/**
 * @brief Solves a least squares problem using normal equations.
 *
 * @param matrix Matrix X'X (size M x N).
 * @param right_hand_side Vector X'y (size M x 1).
 * @param num_rows Rows of matrix (M).
 * @param num_cols Columns of matrix (N).
 * @param solution Output solution x (size N x 1).
 */
void lls_normal(double *matrix, double *right_hand_side, int num_rows, int num_cols, double *solution);

/**
 * @brief Solves a least squares problem using SVD.
 *
 * @param matrix Matrix A (size M x N).
 * @param right_hand_side Vector b (size M x 1).
 * @param num_rows Rows of A (M).
 * @param num_cols Columns of A (N).
 * @param solution Output solution x (size N x 1).
 */
void lls_svd2(double *matrix, double *right_hand_side, int num_rows, int num_cols, double *solution);

/**
 * @brief Computes Householder vector and beta for a vector.
 *
 * @param input_vector Input vector (length N).
 * @param vector_length Length of the vector (N).
 * @param householder_vector Output Householder vector (length N).
 * @return Beta scalar for Householder transformation.
 */
double house_1(double *input_vector, int vector_length, double *householder_vector);

/**
 * @brief Sorts SVD components in descending order.
 *
 * @param u_matrix Left singular vectors (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 * @param v_matrix Right singular vectors (N x N).
 * @param singular_values Singular values (length N).
 */
void svd_sort(double *u_matrix, int num_rows, int num_cols, double *v_matrix, double *singular_values);

/**
 * @brief Generates random matrix using Marsaglia's method.
 *
 * @param matrix Output matrix (M x N).
 * @param num_rows Number of rows (M).
 * @param num_cols Number of columns (N).
 */
void random_matrix(double *matrix, int num_rows, int num_cols);

/**
 * @brief Generates normally distributed random numbers using Marsaglia's polar method.
 *
 * @param values Output array (length N).
 * @param array_length Length of the array (N).
 * @param mean Mean of the distribution.
 * @param std_dev Standard deviation of the distribution.
 * @return Pointer to values array.
 */
double* marsaglia_generate(double *values, int array_length, double mean, double std_dev);

/**
 * @brief Computes eigenvalues and eigenvectors of a symmetric matrix.
 *
 * @param matrix Input symmetric matrix (N x N).
 * @param dimension Matrix dimension (N).
 * @param eigenvalues Output eigenvalues (length N).
 * @param eigenvectors Output eigenvectors (N x N).
 */
void eigensystem(double *matrix, int dimension, double *eigenvalues, double *eigenvectors);

/**
 * @brief Performs Householder reduction to tridiagonal form.
 *
 * @param matrix Input/output matrix (N x N).
 * @param dimension Matrix dimension (N).
 * @param diagonal Output diagonal elements.
 * @param off_diagonal Output off-diagonal elements.
 */
void tred2(double *matrix, int dimension, double *diagonal, double *off_diagonal);

/**
 * @brief Performs QL iteration for eigenvalue computation.
 *
 * @param diagonal Diagonal elements (length N).
 * @param dimension Matrix dimension (N).
 * @param off_diagonal Off-diagonal elements (length N).
 * @param eigenvectors Output eigenvectors (N x N).
 */
void tqli(double *diagonal, int dimension, double *off_diagonal, double *eigenvectors);

/**
 * @brief Computes the Pythagorean sum of two values.
 *
 * @param a First value.
 * @param b Second value.
 * @return Pythagorean sum.
 */
double pythag(double a, double b);

/**
 * @brief Creates an identity matrix.
 *
 * @param matrix Output matrix (N x N).
 * @param dimension Matrix dimension (N).
 */
void eye(double *matrix, int dimension);

/**
 * @brief Creates a scaled identity matrix.
 *
 * @param matrix Output matrix (N x N).
 * @param dimension Matrix dimension (N).
 * @param scalar Scaling factor for diagonal.
 */
void eye_scale(double *matrix, int dimension, double scalar);

/**
 * @brief Adds zero padding to a matrix.
 *
 * @param input_matrix Input matrix (rows x cols).
 * @param num_rows Number of rows.
 * @param num_cols Number of columns.
 * @param zero_rows Number of zero rows to add.
 * @param zero_cols Number of zero columns to add.
 * @param output_matrix Output padded matrix.
 */
void add_zero_pad(double *input_matrix, int num_rows, int num_cols, int zero_rows, int zero_cols, double *output_matrix);

/**
 * @brief Removes zero padding from a matrix.
 *
 * @param input_matrix Padded matrix.
 * @param num_rows Number of rows in padded matrix.
 * @param num_cols Number of columns in padded matrix.
 * @param zero_rows Number of zero rows to remove.
 * @param zero_cols Number of zero columns to remove.
 * @param output_matrix Output unpadded matrix.
 */
void remove_zero_pad(double *input_matrix, int num_rows, int num_cols, int zero_rows, int zero_cols, double *output_matrix);

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_REFACTORED_H_ */