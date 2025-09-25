// SPDX-License-Identifier: BSD-3-Clause
#include "dist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Rounds a double to the integer nearest to zero.
 *
 * Rounds positive numbers down and negative numbers up.
 *
 * @param x Input value.
 * @return Rounded integer value.
 */
double fix_number(double x) {
    // Step 1: Round towards zero
    return (x >= 0.0) ? floor(x) : ceil(x);
}

/**
 * @brief Computes the absolute value of a double.
 *
 * Returns the absolute value of the input.
 *
 * @param x Input value.
 * @return Absolute value |x|.
 */
double absolute_value(double x) {
    // Step 1: Return absolute value
    return (x < 0.0) ? -x : x;
}

/**
 * @brief Computes the machine epsilon scaled by the input value.
 *
 * Returns a precision threshold for numerical computations.
 *
 * @param x Input value for scaling.
 * @return Scaled machine epsilon.
 */
double machine_epsilon(double x) {
    // Step 1: Define base machine epsilon (2^-52)
    double epsilon = pow(2.0, -52.0);
    double abs_x = absolute_value(x);

    // Step 2: Scale epsilon based on x
    if (abs_x > 0.0 && abs_x < 1.0) {
        abs_x /= 2.0;
    }
    int exponent = (int)(log(abs_x) / log(2.0));
    double scale = pow(2.0, (double)exponent);

    // Step 3: Return scaled epsilon
    return epsilon * scale;
}

/**
 * @brief Computes log(1 + x) with high precision for small x.
 *
 * Uses a numerically stable algorithm to avoid cancellation errors.
 *
 * @param x Input value.
 * @return log(1 + x), or x if x is very small.
 */
double log_one_plus(double x) {
    // Step 1: Handle small x to avoid cancellation
    volatile double y = 1.0 + x;
    volatile double z = y - 1.0;
    if (z == 0.0) {
        return x; // For very small x
    }

    // Step 2: Compute log(1 + x) = x * log(y) / (y - 1)
    double result = x * log(y) / z;
    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in log_one_plus\n");
        return NAN;
    }
    return result;
}

/**
 * @brief Recursively computes the factorial of a number.
 *
 * Computes N! recursively for a double input.
 *
 * @param n Input value (assumed integer-like).
 * @return Factorial of n, or NAN if invalid.
 */
double recursive_factorial(double n) {
    // Step 1: Validate input
    int int_n = (int)n;
    if (n != (double)int_n || int_n < 0) {
        printf("Error: Invalid input for recursive_factorial\n");
        return NAN;
    }

    // Step 2: Base case
    if (int_n == 1 || int_n == 0) {
        return 1.0;
    }

    // Step 3: Recursive case
    return n * recursive_factorial(n - 1.0);
}

/**
 * @brief Computes the factorial of an integer.
 *
 * Wrapper for recursive_factorial.
 *
 * @param n Integer input.
 * @return n!, or NAN if invalid.
 */
double compute_factorial(int n) {
    // Step 1: Validate input
    if (n < 0) {
        printf("Error: Negative input for compute_factorial\n");
        return NAN;
    }

    // Step 2: Call recursive factorial
    return recursive_factorial((double)n);
}

/**
 * @brief Computes the gamma function for a positive real number.
 *
 * Uses Cody's algorithm with rational approximations for different ranges.
 *
 * @param x Input value (positive real).
 * @return Gamma(x), or NAN if invalid or overflow.
 */
double gamma_function(double x) {
    // Step 1: Define constants
    const double spi = 0.9189385332046727417803297;
    const double pi = 3.1415926535897932384626434;
    const double xmax = 171.624;
    const double xinf = 1.79e308;
    const double epsilon = 2.22e-16;
    const double xninf = 1.79e-308;

    // Coefficients for 1 <= x <= 2
    const double num[8] = {
        -1.71618513886549492533811e+0, 2.47656508055759199108314e+1,
        -3.79804256470945635097577e+2, 6.29331155312818442661052e+2,
        8.66966202790413211295064e+2, -3.14512729688483675254357e+4,
        -3.61444134186911729807069e+4, 6.64561438202405440627855e+4
    };
    const double den[8] = {
        -3.08402300119738975254353e+1, 3.15350626979604161529144e+2,
        -1.01515636749021914166146e+3, -3.10777167157231109440444e+3,
        2.25381184209801510330112e+4, 4.75584627752788110767815e+3,
        -1.34659959864969306392456e+5, -1.15132259675553483497211e+5
    };

    // Coefficients for x >= 12 (Hart's Minimax)
    const double c[7] = {
        -1.910444077728e-03, 8.4171387781295e-04, -5.952379913043012e-04,
        7.93650793500350248e-04, -2.777777777777681622553e-03,
        8.333333333333333331554247e-02, 5.7083835261e-03
    };

    // Step 2: Validate input
    if (x < 0.0) {
        printf("Error: Input must be non-negative in gamma_function\n");
        return NAN;
    }
    if (x > xinf) {
        printf("Error: Overflow in gamma_function\n");
        return NAN;
    }

    double y = x;
    int switch_sign = 0;
    double factor = 1.0;
    int n_iterations = 0;
    double result;

    // Step 3: Handle negative x using reflection formula
    if (y < 0.0) {
        double integer_part = fix_number(y);
        double fractional_part = y - integer_part;
        if (fractional_part != 0.0) {
            if (integer_part != fix_number(integer_part * 0.5) * 2.0) {
                switch_sign = 1;
            }
            factor = -pi / sin(pi * fractional_part);
            y += 1.0;
        } else {
            return xinf;
        }
    }

    // Step 4: Handle small x
    if (y < epsilon) {
        if (y >= xninf) {
            result = 1.0 / y;
        } else {
            return xinf;
        }
    }
    // Step 5: Handle x < 12
    else if (y < 12.0) {
        double y_initial = y;
        double z;
        if (y < 1.0) {
            z = y;
            y += 1.0;
        } else {
            n_iterations = (int)y - 1;
            y -= (double)n_iterations;
            z = y - 1.0;
        }
        double numerator_sum = 0.0;
        double denominator_sum = 1.0;
        for (int i = 0; i < 8; ++i) {
            numerator_sum = (numerator_sum + num[i]) * z;
            denominator_sum = denominator_sum * z + den[i];
        }
        result = numerator_sum / denominator_sum + 1.0;

        if (y_initial < y) {
            result /= y_initial;
        } else if (y_initial > y) {
            for (int i = 0; i < n_iterations; ++i) {
                result *= y;
                y += 1.0;
            }
        }
    }
    // Step 6: Handle x >= 12
    else {
        if (y <= xmax) {
            double y_squared = y * y;
            double sum = c[6];
            for (int i = 0; i < 6; ++i) {
                sum = sum / y_squared + c[i];
            }
            sum = sum / y - y + spi;
            sum += (y - 0.5) * log(y);
            result = exp(sum);
        } else {
            return xinf;
        }
    }

    // Step 7: Apply sign and factor corrections
    if (switch_sign) {
        result = -result;
    }
    if (factor != 1.0) {
        result = factor / result;
    }

    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in gamma_function\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the logarithm of the gamma function.
 *
 * Uses Cody's algorithm with rational approximations for different ranges.
 *
 * @param x Input value (positive real).
 * @return log(Gamma(x)), or NAN if invalid or overflow.
 */
double gamma_log(double x) {
    // Step 1: Define constants
    const double spi = 0.9189385332046727417803297;
    const double pnt = 0.6796875e+0;
    const double d1 = -5.772156649015328605195174e-1;
    const double d2 = 4.227843350984671393993777e-1;
    const double d4 = 1.791759469228055000094023e+0;
    const double xinf = 1.79e308;
    const double xinf4 = 1.1567e+077;
    const double epsilon = 2.22e-16;

    // Coefficients for 0.5 < x <= 1.5
    const double p1[8] = {
        4.945235359296727046734888e+0, 2.018112620856775083915565e+2,
        2.290838373831346393026739e+3, 1.131967205903380828685045e+4,
        2.855724635671635335736389e+4, 3.848496228443793359990269e+4,
        2.637748787624195437963534e+4, 7.225813979700288197698961e+3
    };
    const double q1[8] = {
        6.748212550303777196073036e+1, 1.113332393857199323513008e+3,
        7.738757056935398733233834e+3, 2.763987074403340708898585e+4,
        5.499310206226157329794414e+4, 6.161122180066002127833352e+4,
        3.635127591501940507276287e+4, 8.785536302431013170870835e+3
    };

    // Coefficients for 1.5 < x <= 4.0
    const double p2[8] = {
        4.974607845568932035012064e+0, 5.424138599891070494101986e+2,
        1.550693864978364947665077e+4, 1.847932904445632425417223e+5,
        1.088204769468828767498470e+6, 3.338152967987029735917223e+6,
        5.106661678927352456275255e+6, 3.074109054850539556250927e+6
    };
    const double q2[8] = {
        1.830328399370592604055942e+2, 7.765049321445005871323047e+3,
        1.331903827966074194402448e+5, 1.136705821321969608938755e+6,
        5.267964117437946917577538e+6, 1.346701454311101692290052e+7,
        1.782736530353274213975932e+7, 9.533095591844353613395747e+6
    };

    // Coefficients for 4.0 < x <= 12.0
    const double p4[8] = {
        1.474502166059939948905062e+04, 2.426813369486704502836312e+06,
        1.214755574045093227939592e+08, 2.663432449630976949898078e+09,
        2.940378956634553899906876e+10, 1.702665737765398868392998e+11,
        4.926125793377430887588120e+11, 5.606251856223951465078242e+11
    };
    const double q4[8] = {
        2.690530175870899333379843e+03, 6.393885654300092398984238e+05,
        4.135599930241388052042842e+07, 1.120872109616147941376570e+09,
        1.488613728678813811542398e+10, 1.016803586272438228077304e+11,
        3.417476345507377132798597e+11, 4.463158187419713286462081e+11
    };

    // Coefficients for x > 12.0 (Hart's Minimax)
    const double c[7] = {
        -1.910444077728e-03, 8.4171387781295e-04, -5.952379913043012e-04,
        7.93650793500350248e-04, -2.777777777777681622553e-03,
        8.333333333333333331554247e-02, 5.7083835261e-03
    };

    // Step 2: Validate input
    if (x <= 0.0) {
        printf("Error: Input must be positive in gamma_log\n");
        return NAN;
    }
    if (x > xinf) {
        printf("Error: Overflow in gamma_log\n");
        return NAN;
    }

    double result;
    // Step 3: Handle small x
    if (x <= epsilon) {
        result = -log(x);
    }
    // Step 4: Handle 0 < x <= 0.5
    else if (x <= 0.5) {
        double y = x;
        double numerator_sum = 0.0;
        double denominator_sum = 1.0;
        for (int i = 0; i < 8; ++i) {
            numerator_sum = numerator_sum * y + p1[i];
            denominator_sum = denominator_sum * y + q1[i];
        }
        result = -log(y) + (y * (d1 + y * (numerator_sum / denominator_sum)));
    }
    // Step 5: Handle 0.5 < x <= 0.6796875
    else if (x <= pnt) {
        double y = (x - 0.5) - 0.5;
        double numerator_sum = 0.0;
        double denominator_sum = 1.0;
        for (int i = 0; i < 8; ++i) {
            numerator_sum = numerator_sum * y + p2[i];
            denominator_sum = denominator_sum * y + q2[i];
        }
        result = -log(x) + (y * (d2 + y * (numerator_sum / denominator_sum)));
    }
    // Step 6: Handle 0.6796875 < x <= 1.5
    else if (x <= 1.5) {
        double y = (x - 0.5) - 0.5;
        double numerator_sum = 0.0;
        double denominator_sum = 1.0;
        for (int i = 0; i < 8; ++i) {
            numerator_sum = numerator_sum * y + p1[i];
            denominator_sum = denominator_sum * y + q1[i];
        }
        result = (y * (d1 + y * (numerator_sum / denominator_sum)));
    }
    // Step 7: Handle 1.5 < x <= 4.0
    else if (x <= 4.0) {
        double y = x - 2.0;
        double numerator_sum = 0.0;
        double denominator_sum = 1.0;
        for (int i = 0; i < 8; ++i) {
            numerator_sum = numerator_sum * y + p2[i];
            denominator_sum = denominator_sum * y + q2[i];
        }
        result = (y * (d2 + y * (numerator_sum / denominator_sum)));
    }
    // Step 8: Handle 4.0 < x <= 12.0
    else if (x <= 12.0) {
        double y = x - 4.0;
        double numerator_sum = 0.0;
        double denominator_sum = -1.0;
        for (int i = 0; i < 8; ++i) {
            numerator_sum = numerator_sum * y + p4[i];
            denominator_sum = denominator_sum * y + q4[i];
        }
        result = d4 + y * (numerator_sum / denominator_sum);
    }
    // Step 9: Handle x > 12.0
    else {
        double y = x;
        result = c[6];
        double y_squared = y * y;
        for (int i = 0; i < 6; ++i) {
            result = result / y_squared + c[i];
        }
        result /= y;
        double log_y = log(y);
        result += spi - 0.5 * log_y + y * (log_y - 1.0);
        if (y > xinf4) {
            printf("Error: Overflow in gamma_log\n");
            return NAN;
        }
    }

    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in gamma_log\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the beta function.
 *
 * Calculates B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b).
 *
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Beta(a, b), or NAN if invalid.
 */
double beta_function(double shape_a, double shape_b) {
    // Step 1: Validate inputs
    if (shape_a <= 0.0 || shape_b <= 0.0) {
        printf("Error: Shape parameters must be positive in beta_function\n");
        return NAN;
    }

    // Step 2: Compute beta using gamma_log
    double result = exp(gamma_log(shape_a) + gamma_log(shape_b) - gamma_log(shape_a + shape_b));
    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in beta_function\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the logarithm of the beta function.
 *
 * Calculates log(B(a, b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a + b)).
 *
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return log(Beta(a, b)), or NAN if invalid.
 */
double beta_log(double shape_a, double shape_b) {
    // Step 1: Validate inputs
    if (shape_a <= 0.0 || shape_b <= 0.0) {
        printf("Error: Shape parameters must be positive in beta_log\n");
        return NAN;
    }

    // Step 2: Compute log beta
    double result = gamma_log(shape_a) + gamma_log(shape_b) - gamma_log(shape_a + shape_b);
    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in beta_log\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the regularized incomplete gamma function (lower tail).
 *
 * Calculates P(x, a) = integral from 0 to x of t^(a-1) * exp(-t) / Gamma(a).
 *
 * @param x Upper limit of integration (x >= 0).
 * @param shape Shape parameter (a >= 0).
 * @return P(x, a), or NAN if invalid.
 */
double gamma_cdf_lower(double x, double shape) {
    // Step 1: Validate inputs
    if (shape < 0.0) {
        printf("Error: Shape parameter must be non-negative in gamma_cdf_lower\n");
        return NAN;
    }
    if (x < 0.0) {
        return 0.0; // CDF is zero for negative x
    }

    double result;
    // Step 2: Handle special cases
    if (shape == 0.0) {
        return 1.0;
    }
    if (x == 0.0) {
        return 0.0;
    }

    // Step 3: Handle large shape parameter
    const double large_shape = 2.0e20;
    double adjusted_x = x;
    if (shape > large_shape) {
        adjusted_x = large_shape - 1.0 / 3.0 + sqrt(large_shape / shape) * (x - (shape - 1.0 / 3.0));
        shape = large_shape;
        if (adjusted_x < 0.0) {
            adjusted_x = 0.0;
        }
    }

    // Step 4: Compute for x < shape + 1 (series expansion)
    if (adjusted_x < shape + 1.0) {
        double term = 1.0;
        double sum = 1.0;
        double counter = shape;
        while (absolute_value(term) >= 100.0 * machine_epsilon(sum)) {
            counter += 1.0;
            term = adjusted_x * term / counter;
            sum += term;
        }
        result = sum * exp(-adjusted_x + shape * log(adjusted_x) - gamma_log(shape + 1.0));
        if (adjusted_x > 0.0 && result > 1.0) {
            result = 1.0;
        }
    }
    // Step 5: Compute for x >= shape + 1 (continued fraction)
    else {
        double a0 = 1.0, a1 = adjusted_x, b0 = 0.0, b1 = 1.0;
        double factor = 1.0 / a1;
        double counter = 1.0;
        double g = b1 * factor;
        double g_prev = 0.0;
        double delta = absolute_value(g - g_prev);
        while (delta >= 100.0 * machine_epsilon(absolute_value(g))) {
            g_prev = g;
            double d = counter - shape;
            a0 = (a1 + a0 * d) * factor;
            b0 = (b1 + b0 * d) * factor;
            double s = counter * factor;
            a1 = adjusted_x * a0 + s * a1;
            b1 = adjusted_x * b0 + s * b1;
            factor = 1.0 / a1;
            counter += 1.0;
            g = b1 * factor;
            delta = absolute_value(g - g_prev);
        }
        result = 1.0 - g * exp(-adjusted_x + shape * log(adjusted_x) - gamma_log(shape));
    }

    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in gamma_cdf_lower\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the regularized incomplete gamma function (upper tail).
 *
 * Calculates Q(x, a) = 1 - P(x, a).
 *
 * @param x Upper limit of integration (x >= 0).
 * @param shape Shape parameter (a >= 0).
 * @return Q(x, a), or NAN if invalid.
 */
double gamma_cdf_upper(double x, double shape) {
    // Step 1: Compute lower tail and subtract
    double result = gamma_cdf_lower(x, shape);
    if (isnan(result)) {
        return NAN;
    }
    return 1.0 - result;
}

/**
 * @brief Computes the continued fraction for the incomplete beta function.
 *
 * Uses Abramowitz and Stegun equation 26.5.8.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @param iteration_count Output number of iterations used.
 * @return Continued fraction value, or NAN if invalid.
 */
double beta_continued_fraction(double x, double shape_a, double shape_b, int* iteration_count) {
    // Step 1: Validate inputs
    if (x < 0.0 || x > 1.0) {
        printf("Error: x must be between 0.0 and 1.0 in beta_continued_fraction\n");
        *iteration_count = 0;
        return NAN;
    }
    if (shape_a <= 0.0 || shape_b <= 0.0) {
        printf("Error: Shape parameters must be positive in beta_continued_fraction\n");
        *iteration_count = 0;
        return NAN;
    }

    // Step 2: Initialize continued fraction
    int iteration = 1;
    double a_m = 1.0, b_m = 1.0, g = 1.0;
    double a0 = 0.0, b0 = 0.0, a1 = 0.0, b1 = 0.0, g_prev = 0.0;
    double delta = 1.0 - (shape_a + shape_b) * x / (shape_a + 1.0);
    double difference = absolute_value(g - g_prev);
    const int max_iterations = 1000;

    // Step 3: Iterate continued fraction
    while (difference > 1000.0 * machine_epsilon(g) && iteration < max_iterations) {
        double a_m_plus = (double)(shape_a + iteration);
        double a_m_plus_two = a_m_plus + iteration;
        g_prev = g;
        double d_2m = iteration * (shape_b - iteration) * x / (a_m_plus_two * (a_m_plus_two - 1.0));
        double d_2m1 = -a_m_plus * (a_m_plus + shape_b) * x / (a_m_plus_two * (a_m_plus_two + 1.0));

        a0 = g + d_2m * a_m;
        b0 = delta + d_2m * b_m;
        a1 = a0 + d_2m1 * g;
        b1 = b0 + d_2m1 * delta;
        if (iteration == 1) {
            delta = 1.0;
        }
        if (b1 == 0.0) {
            printf("Error: Division by zero in beta_continued_fraction\n");
            *iteration_count = iteration;
            return NAN;
        }

        a_m = a0 / b1;
        b_m = b0 / b1;
        g = a1 / b1;

        iteration++;
        difference = absolute_value(g - g_prev);
    }

    *iteration_count = iteration;
    if (iteration >= max_iterations) {
        printf("Warning: beta_continued_fraction did not converge\n");
    }

    return g;
}

/**
 * @brief Computes an approximation to the incomplete beta function.
 *
 * Uses equations 26.5.20 and 26.5.21 from Abramowitz and Stegun.
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Approximated incomplete beta value, or NAN if invalid.
 */
double beta_incomplete_approx(double x, double shape_a, double shape_b) {
    // Step 1: Validate inputs
    if (x < 0.0 || x > 1.0) {
        printf("Error: x must be between 0.0 and 1.0 in beta_incomplete_approx\n");
        return NAN;
    }
    if (shape_a <= 0.0 || shape_b <= 0.0) {
        printf("Error: Shape parameters must be positive in beta_incomplete_approx\n");
        return NAN;
    }

    // Step 2: Compute approximation
    double abx = (shape_a + shape_b - 1.0) * (1.0 - x);
    double result;
    if (abx <= 0.8) {
        double c2 = (abx * (3.0 - x) - (shape_b - 1.0) * (1.0 - x)) / 2.0;
        result = gamma_cdf_upper(c2, shape_b);
    } else {
        double w1 = pow(shape_b * x, 1.0 / 3.0);
        double w2 = pow(shape_a * (1.0 - x), 1.0 / 3.0);
        double numerator = -3.0 * (w1 * (1.0 - 1.0 / (9.0 * shape_b)) - w2 * (1.0 - 1.0 / (9.0 * shape_a)));
        double denominator = sqrt((w1 * w1 / shape_b) + (w2 * w2 / shape_a));
        if (denominator == 0.0) {
            printf("Error: Division by zero in beta_incomplete_approx\n");
            return NAN;
        }
        result = erfc(numerator / denominator) / 2.0;
    }

    if (isnan(result)) {
        printf("Error: Numerical instability in beta_incomplete_approx\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the regularized incomplete beta function.
 *
 * Uses continued fraction (method 1) or approximation (method 2 if convergence fails).
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return I_x(a, b), or NAN if invalid.
 */
double beta_incomplete(double x, double shape_a, double shape_b) {
    // Step 1: Validate inputs
    if (x < 0.0 || x > 1.0) {
        printf("Error: x must be between 0.0 and 1.0 in beta_incomplete\n");
        return NAN;
    }
    if (shape_a <= 0.0 || shape_b <= 0.0) {
        printf("Error: Shape parameters must be positive in beta_incomplete\n");
        return NAN;
    }

    // Step 2: Compute continued fraction
    int iteration_count = 0;
    double result;
    double threshold = (shape_a + 1.0) / (shape_a + shape_b + 2.0);
    double temp;
    if (x < threshold) {
        temp = exp(gamma_log(shape_a + shape_b) - gamma_log(shape_a + 1.0) - gamma_log(shape_b) + shape_a * log(x) + shape_b * log_one_plus(-x));
        result = temp * beta_continued_fraction(x, shape_a, shape_b, &iteration_count);
    } else {
        temp = exp(gamma_log(shape_a + shape_b) - gamma_log(shape_b + 1.0) - gamma_log(shape_a) + shape_a * log(x) + shape_b * log_one_plus(-x));
        result = 1.0 - temp * beta_continued_fraction(1.0 - x, shape_b, shape_a, &iteration_count);
    }

    // Step 3: Fall back to approximation if continued fraction fails
    if (iteration_count >= 1000) {
        result = beta_incomplete_approx(x, shape_a, shape_b);
    }

    if (isnan(result)) {
        printf("Error: Numerical instability in beta_incomplete\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the complementary incomplete beta function.
 *
 * Calculates I_{1-x}(b, a) = 1 - I_x(a, b).
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return I_{1-x}(b, a), or NAN if invalid.
 */
double beta_incomplete_complement(double x, double shape_a, double shape_b) {
    // Step 1: Compute I_{1-x}(b, a)
    double result = beta_incomplete(1.0 - x, shape_b, shape_a);
    if (isnan(result)) {
        printf("Error: beta_incomplete failed in beta_incomplete_complement\n");
        return NAN;
    }
    return result;
}

/**
 * @brief Computes the PDF of the beta distribution.
 *
 * Calculates x^(a-1) * (1-x)^(b-1) / B(a, b).
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return PDF value, or NAN if invalid.
 */
double beta_pdf(double x, double shape_a, double shape_b) {
    // Step 1: Validate inputs
    if (x < 0.0 || x > 1.0) {
        printf("Error: x must be between 0.0 and 1.0 in beta_pdf\n");
        return NAN;
    }
    if (shape_a <= 0.0 || shape_b <= 0.0) {
        printf("Error: Shape parameters must be positive in beta_pdf\n");
        return NAN;
    }

    // Step 2: Compute log terms
    double log_a = (shape_a == 1.0 || x == 0.0) ? 0.0 : (shape_a - 1.0) * log(x);
    double log_b = (shape_b == 1.0 || x == 1.0) ? 0.0 : (shape_b - 1.0) * log(1.0 - x);

    // Step 3: Compute PDF
    double result = exp(log_a + log_b - beta_log(shape_a, shape_b));
    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in beta_pdf\n");
        return NAN;
    }

    return result;
}

/**
 * @brief Computes the CDF of the beta distribution.
 *
 * Calculates the regularized incomplete beta function I_x(a, b).
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return CDF value, or NAN if invalid.
 */
double beta_cdf(double x, double shape_a, double shape_b) {
    // Step 1: Compute incomplete beta
    double result = beta_incomplete(x, shape_a, shape_b);
    if (isnan(result)) {
        return NAN;
    }

    // Step 2: Correct overshoot
    if (result > 1.0) {
        result = 1.0;
    }

    return result;
}

/**
 * @brief Computes the derivative of the incomplete beta function.
 *
 * Returns the beta PDF as the derivative of I_x(a, b).
 *
 * @param x Input value (0 <= x <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Derivative value, or NAN if invalid.
 */
double beta_incomplete_derivative(double x, double shape_a, double shape_b) {
    // Step 1: Compute beta PDF
    double result = beta_pdf(x, shape_a, shape_b);
    if (isnan(result)) {
        printf("Error: beta_pdf failed in beta_incomplete_derivative\n");
        return NAN;
    }
    return result;
}

/**
 * @brief Computes the inverse CDF of the beta distribution.
 *
 * Uses modified Newton-Raphson method for quantile computation.
 *
 * @param probability Probability value (0 <= p <= 1).
 * @param shape_a First shape parameter (a > 0).
 * @param shape_b Second shape parameter (b > 0).
 * @return Quantile value, or NAN if invalid or convergence fails.
 */
double beta_inv(double probability, double shape_a, double shape_b) {
    // Step 1: Validate inputs
    if (probability < 0.0 || probability > 1.0) {
        printf("Error: Probability must be between 0.0 and 1.0 in beta_inv\n");
        return NAN;
    }
    if (shape_a <= 0.0 || shape_b <= 0.0) {
        printf("Error: Shape parameters must be positive in beta_inv\n");
        return NAN;
    }

    // Step 2: Handle boundary cases
    if (probability == 0.0) {
        return 0.0;
    }
    if (probability == 1.0) {
        return 1.0;
    }

    // Step 3: Initialize variables
    double a, pp, qq;
    int index;
    if (probability > 0.5) {
        a = 1.0 - probability;
        pp = shape_b;
        qq = shape_a;
        index = 1;
    } else {
        a = probability;
        pp = shape_a;
        qq = shape_b;
        index = 0;
    }

    // Step 4: Initial approximation
    double beta_val = beta_log(pp, qq);
    double result = a;
    double r = sqrt(-log(a * a));
    double y = r - (2.30753 + 0.27061 * r) / (1.0 + (0.99229 + 0.04481 * r) * r);

    if (pp > 1.0 && qq > 1.0) {
        r = (y * y - 3.0) / 6.0;
        double s = 1.0 / (pp + pp - 1.0);
        double t = 1.0 / (qq + qq - 1.0);
        double h = 2.0 / (s + t);
        double w = y * sqrt(h + r) / h - (t - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
        result = pp / (pp + qq * exp(w + w));
    } else {
        r = qq + qq;
        double t = 1.0 / (9.0 * qq);
        t = r * pow(1.0 - t + y * sqrt(t), 3.0);
        if (t <= 0.0) {
            result = 1.0 - exp((log((1.0 - a) * qq) + beta_val) / qq);
        } else {
            t = (4.0 * pp + r - 2.0) / t;
            if (t <= 1.0) {
                result = exp((log(a * pp) + beta_val) / pp);
            } else {
                result = 1.0 - 2.0 / (t + 1.0);
            }
        }
    }

    // Step 5: Modified Newton-Raphson iteration
    double fpu = pow(10.0, -37.0);
    double r_val = 1.0 - pp;
    double t_val = 1.0 - qq;
    double y_prev = 0.0;
    double sq = 1.0;
    double prev = 1.0;
    int iex = (int)r8_max(-5.0 / pp / pp - 1.0 / pow(a, 0.2) - 13.0, -37.0);
    double acu = pow(10.0, iex);

    if (result < 0.0001) {
        result = 0.0001;
    }
    if (result > 0.9999) {
        result = 0.9999;
    }

    while (1) {
        y = beta_cdf(result, pp, qq);
        if (isnan(y)) {
            printf("Error: beta_cdf failed in beta_inv\n");
            return NAN;
        }
        double xin = result;
        y = (y - a) * exp(beta_val + r_val * log(xin) + t_val * log(1.0 - xin));
        if (y * y_prev <= 0.0) {
            prev = r8_max(sq, fpu);
        }

        double g = 1.0;
        while (1) {
            double adj = g * y;
            sq = adj * adj;
            if (sq < prev) {
                double tx = result - adj;
                if (tx >= 0.0 && tx <= 1.0) {
                    result = tx;
                    break;
                }
            }
            g /= 3.0;
        }

        if (prev <= acu || y * y <= acu) {
            if (index) {
                result = 1.0 - result;
            }
            if (isnan(result) || isinf(result)) {
                printf("Error: Numerical instability in beta_inv\n");
                return NAN;
            }
            return result;
        }

        if (result == tx) {
            break;
        }

        y_prev = y;
    }

    if (index) {
        result = 1.0 - result;
    }

    if (isnan(result) || isinf(result)) {
        printf("Error: Numerical instability in beta_inv\n");
        return NAN;
    }

    return result;
}
