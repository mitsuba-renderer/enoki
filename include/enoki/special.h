/*
    enoki/special.h -- Special functions: Bessel functions, Elliptic
    and exponential integrals, etc. (still incomplete)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array.h"

NAMESPACE_BEGIN(enoki)

/// Evaluates a series of Chebyshev polynomials at argument x/2.
template <typename T, typename T2, size_t Size,
          typename Expr = expr_t<T>> Expr chbevl(T x, T2 (&coeffs)[Size]) {
    using Scalar = scalar_t<Expr>;

    Expr b0 = Scalar(coeffs[0]);
    Expr b1 = Scalar(0);
    Expr b2;

    ENOKI_UNROLL for (size_t i = 0; i < Size; ++i) {
        b2 = b1;
        b1 = b0;
        b0 = fmsub(x, b1, b2 - Scalar(coeffs[i]));
    }

    return (b0 - b2) * Scalar(0.5f);
}

template <typename T, enable_if_not_array_t<T> = 0> T erf(T x) {
    return std::erf(x);
}

template <typename T, typename Expr = expr_t<T>, enable_if_array_t<T> = 0> Expr erf(T x_) {
    using Scalar = scalar_t<T>;

    Expr r;
    if (Expr::Approx) {
        /*
         - erf (in [-1, 1]):
           * avg abs. err = 1.02865e-07
           * avg rel. err = 3.29449e-07
              -> in ULPs  = 3.96707
           * max abs. err = 5.58794e-07
             (at x=-0.0803231)
           * max rel. err = 6.17859e-06
             -> in ULPs   = 75
             (at x=-0.0803231)
         */

        // A&S formula 7.1.26
        Expr x = abs(x_), x2 = x_ * x_;
        Expr t = rcp(fmadd(Expr(Scalar(0.3275911)), x, Expr(Scalar(1))));

        Expr y = poly4(t, 0.254829592, -0.284496736,
                          1.421413741, -1.453152027,
                          1.061405429);

        y *= t * exp(-x2);

        /* Switch between the A&S approximation and a Taylor series
           expansion around the origin */
        r = select(
            x > Expr(Scalar(0.08)),
            (Scalar(1) - y) | detail::sign_mask(x_),
            x_ * fmadd(x2, Expr(Scalar(-M_2_SQRTPI/3)),
                           Expr(Scalar( M_2_SQRTPI)))
        );
    } else {
        for (size_t i = 0; i < Expr::Size; ++i)
            r.coeff(i) = enoki::erf(x_.coeff(i));
    }
    return r;
}

// Inverse real error function approximation based on on "Approximating the
// erfinv function" by Mark Giles
template <typename T, typename Expr = expr_t<T>> Expr erfinv(T x_) {
    using Scalar = scalar_t<T>;

    Expr x(x_);
    Expr w = -log((Expr(Scalar(1)) - x) * (Expr(Scalar(1)) + x));

    Expr w1 = w - Scalar(2.5);
    Expr w2 = sqrt(w) - Scalar(3);

    Expr p1 = poly8(w1,
         1.50140941,     0.246640727,
        -0.00417768164, -0.00125372503,
         0.00021858087, -4.39150654e-06,
        -3.5233877e-06,  3.43273939e-07,
         2.81022636e-08);

    Expr p2 = poly8(w2,
         2.83297682,     1.00167406,
         0.00943887047, -0.0076224613,
         0.00573950773, -0.00367342844,
         0.00134934322,  0.000100950558,
        -0.000200214257);

    return select(w < Scalar(5), p1, p2) * x;
}

/// Evaluates Dawson's integral (e^(-x^2) \int_0^x e^(y^2) dy)
template <typename T, typename Expr = expr_t<T>> Expr dawson(T x) {
    // Rational minimax approximation to Dawson's integral with relative
    // error < 1e-6 on the real number line. July 2017, Wenzel Jakob

    Expr x2 = x*x;
    Expr num = poly6(x2, 1.00000080272429,9.18170212243285e-2,
                         4.25835373536124e-2, 6.0536496345054e-3,
                         9.88555033724111e-4, 3.64943550840577e-5,
                         1.55942290996993e-5);

    Expr denom = poly7(x2, 1.0, 7.58517175815194e-1,
                           2.81364355593059e-1, 6.81783097841267e-2,
                           1.13586116798019e-2, 1.92020805811771e-3,
                           5.74217664074868e-5, 3.11884331363595e-5);

    return num / denom * x;
}

/// Imaginary component of the error function
template <typename T, typename Expr = expr_t<T>> Expr erfi(T x) {
    using Scalar = scalar_t<T>;

    return Scalar(M_2_SQRTPI) * dawson(x) * exp(x*x);
}

/// Modified Bessel function of the first kind, order zero (exponentially scaled)
template <typename T, typename Expr = expr_t<T>> Expr i0e(T x) {
    using Scalar = scalar_t<T>;

    /* Chebyshev coefficients for exp(-x) I0(x)
     * in the interval [0,8].
     *
     * lim(x->0) { exp(-x) I0(x) } = 1.
     */

    static float A[] = {
        -1.30002500998624804212E-8f, 6.04699502254191894932E-8f,
        -2.67079385394061173391E-7f, 1.11738753912010371815E-6f,
        -4.41673835845875056359E-6f, 1.64484480707288970893E-5f,
        -5.75419501008210370398E-5f, 1.88502885095841655729E-4f,
        -5.76375574538582365885E-4f, 1.63947561694133579842E-3f,
        -4.32430999505057594430E-3f, 1.05464603945949983183E-2f,
        -2.37374148058994688156E-2f, 4.93052842396707084878E-2f,
        -9.49010970480476444210E-2f, 1.71620901522208775349E-1f,
        -3.04682672343198398683E-1f, 6.76795274409476084995E-1f
    };


    /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
     * in the inverted interval [8,infinity].
     *
     * lim(x->inf) { exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
     */

    static float B[] = {
        3.39623202570838634515E-9f, 2.26666899049817806459E-8f,
        2.04891858946906374183E-7f, 2.89137052083475648297E-6f,
        6.88975834691682398426E-5f, 3.36911647825569408990E-3f,
        8.04490411014108831608E-1f
    };


    x = abs(x);

    auto mask_big = x > Scalar(8);

    Expr r_big, r_small;

    if (!all_nested(mask_big))
        r_small = chbevl(fmsub(x, Expr(Scalar(0.5)), Expr(Scalar(2))), A);

    if (any_nested(mask_big))
        r_big = chbevl(fmsub(Expr(Scalar(32)), rcp(x), Expr(Scalar(2))), B) *
                rsqrt(x);

    return select(mask_big, r_big, r_small);
}

NAMESPACE_END(enoki)
