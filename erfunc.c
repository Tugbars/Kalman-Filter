```c
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "erfunc.h"
#include <math.h>
#include <stdio.h>
#include <float.h>

/**
 * @brief Computes the largest positive or negative exponent for exp(x).
 *
 * Returns the largest W such that exp(W) is computable (if l=0) or the largest
 * negative W where exp(W) is non-zero (if l!=0).
 *
 * @param l Flag (0 for positive, non-zero for negative).
 * @return Largest exponent value.
 */
static double compute_exp_limit(int l) {
    // Step 1: Define natural log of 2
    const double ln_2 = 0.69314718055995;

    // Step 2: Select exponent based on l
    int max_exp = (l == 0) ? DBL_MAX_EXP : DBL_MIN_EXP - 1;

    // Step 3: Compute approximate limit
    return max_exp * ln_2 * 0.99999;
}

/**
 * @brief Computes the error function erf(x).
 *
 * Evaluates the real error function using rational approximations for different ranges.
 *
 * @param x Input value.
 * @return erf(x), or ±1 for large |x|.
 */
double erf(double x) {
    // Step 1: Define constants
    const double c = 0.564189583547756; // sqrt(pi)/2
    const double a[5] = {
        7.7105849500132e-5, -0.00133733772997339, 0.0323076579225834,
        0.0479137145607681, 0.128379167095513
    };
    const double b[3] = { 0.00301048631703895, 0.0538971687740286, 0.375795757275549 };
    const double p[8] = {
        -1.36864857382717e-7, 0.564195517478974, 7.21175825088309,
        43.1622272220567, 152.98928504694, 339.320816734344,
        451.918953711873, 300.459261020162
    };
    const double q[8] = {
        1.0, 12.7827273196294, 77.0001529352295, 277.585444743988,
        638.980264465631, 931.35409485061, 790.950925327898, 300.459260956983
    };
    const double r[5] = { 2.10144126479064, 26.2370141675169, 21.3688200555087, 4.6580782871847, 0.282094791773523 };
    const double s[4] = { 94.153775055546, 187.11481179959, 99.0191814623914, 18.0124575948747 };

    // Step 2: Compute absolute value
    double abs_x = fabs(x);

    // Step 3: Handle |x| <= 0.5
    if (abs_x <= 0.5) {
        double t = x * x;
        double numerator = ((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) + 1.0;
        double denominator = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        if (denominator == 0.0) {
            printf("Error: Division by zero in erf\n");
            return NAN;
        }
        return x * (numerator / denominator);
    }

    // Step 4: Handle 0.5 < |x| <= 4
    if (abs_x <= 4.0) {
        double numerator = ((((((p[0] * abs_x + p[1]) * abs_x + p[2]) * abs_x + p[3]) * abs_x + p[4]) * abs_x
                            + p[5]) * abs_x + p[6]) * abs_x + p[7];
        double denominator = ((((((q[0] * abs_x + q[1]) * abs_x + q[2]) * abs_x + q[3]) * abs_x + q[4]) * abs_x
                              + q[5]) * abs_x + q[6]) * abs_x + q[7];
        if (denominator == 0.0) {
            printf("Error: Division by zero in erf\n");
            return NAN;
        }
        double result = 0.5 - exp(-x * x) * numerator / denominator + 0.5;
        return (x < 0.0) ? -result : result;
    }

    // Step 5: Handle |x| >= 5.8
    if (abs_x >= 5.8) {
        return (x > 0.0) ? 1.0 : -1.0;
    }

    // Step 6: Handle 4 < |x| < 5.8
    double x_squared = x * x;
    double t = 1.0 / x_squared;
    double numerator = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
    double denominator = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
    if (denominator == 0.0 || x_squared == 0.0) {
        printf("Error: Division by zero in erf\n");
        return NAN;
    }
    double result = 0.5 - exp(-x_squared) * (c - numerator / (x_squared * denominator)) / abs_x + 0.5;
    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in erf\n");
        return NAN;
    }
    return (x < 0.0) ? -result : result;
}

/**
 * @brief Computes the complementary error function erfc(x) or exp(x^2)*erfc(x).
 *
 * Evaluates erfc(x) if ind=0, or exp(x^2)*erfc(x) if ind!=0.
 *
 * @param ind Flag (0 for erfc, non-zero for scaled erfc).
 * @param x Input value.
 * @return erfc(x) or exp(x^2)*erfc(x), or NAN if invalid.
 */
static double compute_erfc(int ind, double x) {
    // Step 1: Define constants
    const double c = 0.564189583547756;
    const double a[5] = {
        7.7105849500132e-5, -0.00133733772997339, 0.0323076579225834,
        0.0479137145607681, 0.128379167095513
    };
    const double b[3] = { 0.00301048631703895, 0.0538971687740286, 0.375795757275549 };
    const double p[8] = {
        -1.36864857382717e-7, 0.564195517478974, 7.21175825088309,
        43.1622272220567, 152.98928504694, 339.320816734344,
        451.918953711873, 300.459261020162
    };
    const double q[8] = {
        1.0, 12.7827273196294, 77.0001529352295, 277.585444743988,
        638.980264465631, 931.35409485061, 790.950925327898, 300.459260956983
    };
    const double r[5] = { 2.10144126479064, 26.2370141675169, 21.3688200555087, 4.6580782871847, 0.282094791773523 };
    const double s[4] = { 94.153775055546, 187.11481179959, 99.0191814623914, 18.0124575948747 };

    // Step 2: Compute absolute value
    double abs_x = fabs(x);

    // Step 3: Handle |x| <= 0.5
    if (abs_x <= 0.5) {
        double t = x * x;
        double numerator = ((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) + 1.0;
        double denominator = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        if (denominator == 0.0) {
            printf("Error: Division by zero in compute_erfc\n");
            return NAN;
        }
        double result = 0.5 - x * (numerator / denominator) + 0.5;
        if (ind != 0) {
            result *= exp(t);
        }
        if (isnan(result) || isinf(result)) {
            printf("Error: Numerical instability in compute_erfc\n");
            return NAN;
        }
        return result;
    }

    // Step 4: Handle 0.5 < |x| <= 4
    double result;
    if (abs_x <= 4.0) {
        double numerator = ((((((p[0] * abs_x + p[1]) * abs_x + p[2]) * abs_x + p[3]) * abs_x + p[4]) * abs_x
                            + p[5]) * abs_x + p[6]) * abs_x + p[7];
        double denominator = ((((((q[0] * abs_x + q[1]) * abs_x + q[2]) * abs_x + q[3]) * abs_x + q[4]) * abs_x
                              + q[5]) * abs_x + q[6]) * abs_x + q[7];
        if (denominator == 0.0) {
            printf("Error: Division by zero in compute_erfc\n");
            return NAN;
        }
        result = numerator / denominator;
    } else {
        // Step 5: Handle |x| > 4
        if (x <= -5.6) {
            result = 2.0;
            if (ind != 0) {
                result *= exp(x * x);
            }
            return result;
        }
        if (ind == 0 && (x > 100.0 || x * x > -compute_exp_limit(1))) {
            return 0.0;
        }
        double t = 1.0 / (x * x);
        double numerator = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
        double denominator = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
        if (denominator == 0.0 || abs_x == 0.0) {
            printf("Error: Division by zero in compute_erfc\n");
            return NAN;
        }
        result = (c - t * numerator / denominator) / abs_x;
    }

    // Step 6: Final assembly
    if (ind != 0) {
        if (x < 0.0) {
            result = exp(x * x) * 2.0 - result;
        }
    } else {
        double w = x * x;
        double t = w;
        double e = w - t;
        result = (0.5 - e + 0.5) * exp(-t) * result;
        if (x < 0.0) {
            result = 2.0 - result;
        }
    }

    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in compute_erfc\n");
        return NAN;
    }
    return result;
}

/**
 * @brief Computes the complementary error function erfc(x).
 *
 * Wrapper for compute_erfc with ind=0.
 *
 * @param x Input value.
 * @return erfc(x), or NAN if invalid.
 */
double erfc(double x) {
    // Step 1: Call compute_erfc with ind=0
    return compute_erfc(0, x);
}

/**
 * @brief Computes the scaled complementary error function exp(x^2)*erfc(x).
 *
 * Wrapper for compute_erfc with ind=2.
 *
 * @param x Input value.
 * @return exp(x^2)*erfc(x), or NAN if invalid.
 */
double erfc_scaled(double x) {
    // Step 1: Call compute_erfc with ind=2
    return compute_erfc(2, x);
}

/**
 * @brief Computes the inverse error function erfinv(x).
 *
 * Uses rational Chebyshev approximations for different ranges.
 *
 * @param x Input value (-1 <= x <= 1).
 * @return erfinv(x), or NAN if invalid.
 */
double erf_inv(double x) {
    // Step 1: Validate input
    if (fabs(x) > 1.0) {
        printf("Error: Input must be between -1 and 1 in erf_inv\n");
        return NAN;
    }

    // Step 2: Handle sign
    double sign = (x >= 0.0) ? 1.0 : -1.0;
    double abs_x = fabs(x);
    const double xinf = XINFVAL;
    const double pi = PIVAL;
    double result;

    // Step 3: Handle |x| <= 0.75 (Table 13)
    if (abs_x <= 0.75) {
        const double p[5] = { 4.62680202125696, -16.6805947126248, 17.6230176190819, -5.4309342907266, 0.236997019142 };
        const double q[5] = { 4.26606447606664, -17.5930726990431, 22.7331464544494, -9.9016863476727, 1.0 };
        double t = x * x - 0.75 * 0.75;
        double numerator = p[0] + t * (p[1] + t * (p[2] + t * (p[3] + t * p[4])));
        double denominator = q[0] + t * (q[1] + t * (q[2] + t * (q[3] + t * q[4])));
        if (denominator == 0.0) {
            printf("Error: Division by zero in erf_inv\n");
            return NAN;
        }
        result = x * numerator / denominator;
    }
    // Step 4: Handle 0.75 < |x| <= 0.9375 (Table 33)
    else if (abs_x <= 0.9375) {
        const double p[6] = { -0.041199817067782, 0.643729854003468, -3.28902674093993, 6.24518431579026, -3.65953596495117, 0.30260114943200 };
        const double q[6] = { -0.029324540620124, 0.501148500527886, -2.90144687299145, 6.65393051963183, -5.40640580412825, 1.0 };
        double t = x * x - 0.9375 * 0.9375;
        double numerator = p[0] + t * (p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5]))));
        double denominator = q[0] + t * (q[1] + t * (q[2] + t * (q[3] + t * (q[4] + t * q[5]))));
        if (denominator == 0.0) {
            printf("Error: Division by zero in erf_inv\n");
            return NAN;
        }
        result = x * numerator / denominator;
    }
    // Step 5: Handle 0.9375 < |x| < 1 (Table 50)
    else if (abs_x < 1.0) {
        const double p[6] = { 0.1550470003116, 1.382719649631, 0.690969348887, -1.128081391617, 0.680544246825, -0.16444156791 };
        const double q[3] = { 0.155024849822, 1.385228141995, 1.0 };
        double t = 1.0 / sqrt(-log(1.0 - abs_x));
        double numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])));
        double denominator = q[0] + t * (q[1] + t * q[2]);
        if (denominator == 0.0) {
            printf("Error: Division by zero in erf_inv\n");
            return NAN;
        }
        result = sign * numerator / denominator;
    }
    // Step 6: Handle |x| = 1
    else {
        result = sign * xinf;
    }

    // Step 7: Apply Newton's correction for numerical accuracy
    if (abs_x < 1.0) {
        double temp = (erf(result) - x) / (2.0 / sqrt(pi) * exp(-result * result));
        if (isnan(temp)) {
            printf("Error: Numerical instability in erf_inv correction\n");
            return NAN;
        }
        result -= temp / (1.0 + temp * result);
    }

    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in erf_inv\n");
        return NAN;
    }
    return result;
}

/**
 * @brief Computes the inverse complementary error function erfcinv(x).
 *
 * Uses rational Chebyshev approximations for different ranges.
 *
 * @param x Input value (0 <= x <= 2).
 * @return erfcinv(x), or NAN if invalid.
 */
double erfc_inv(double x) {
    // Step 1: Validate input
    if (x < 0.0 || x > 2.0) {
        printf("Error: Input must be between 0.0 and 2.0 in erfc_inv\n");
        return NAN;
    }

    // Step 2: Define constants
    const double xinf = XINFVAL;
    const double pi = PIVAL;
    double result;

    // Step 3: Handle x >= 0.0625 and x < 2
    if (x >= 0.0625 && x < 2.0) {
        result = erf_inv(1.0 - x);
    }
    // Step 4: Handle 1e-100 <= x < 0.0625
    else if (x >= XNINFVAL && x < 0.0625) {
        const double p[6] = { 0.1550470003116, 1.382719649631, 0.690969348887, -1.128081391617, 0.680544246825, -0.16444156791 };
        const double q[3] = { 0.155024849822, 1.385228141995, 1.0 };
        double t = 1.0 / sqrt(-log(x));
        double numerator = p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])));
        double denominator = q[0] + t * (q[1] + t * q[2]);
        if (denominator == 0.0) {
            printf("Error: Division by zero in erfc_inv\n");
            return NAN;
        }
        result = numerator / denominator;
    }
    // Step 5: Handle x < XNINFVAL
    else if (x < XNINFVAL) {
        const double p[4] = { 0.00980456202915, 0.363667889171, 0.97302949837, -0.5374947401 };
        const double q[3] = { 0.00980451277802, 0.363699971544, 1.0 };
        double t = 1.0 / sqrt(-log(x));
        double numerator = p[0] / t + p[1] + t * (p[2] + t * p[3]);
        double denominator = q[0] + t * (q[1] + t * q[2]);
        if (denominator == 0.0) {
            printf("Error: Division by zero in erfc_inv\n");
            return NAN;
        }
        result = numerator / denominator;
    }
    // Step 6: Handle x = XNINFVAL
    else if (x <= XNINFVAL) {
        result = xinf;
    }
    // Step 7: Handle x = 2
    else {
        result = -xinf;
    }

    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in erfc_inv\n");
        return NAN;
    }
    return result;
}