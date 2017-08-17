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
          typename Expr = expr_t<T>> Expr chbevl(const T &x, T2 (&coeffs)[Size]) {
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

template <typename T, enable_if_not_array_t<T> = 0> T erf(const T &x) {
    return std::erf(x);
}


template <typename T, enable_if_not_array_t<T> = 0> T erfc(const T &x) {
    return std::erfc(x);
}

template <typename T, bool Recurse = true, typename Expr = expr_t<T>, enable_if_array_t<T> = 0> Expr erfc(const T &x);
template <typename T, bool Recurse = true, typename Expr = expr_t<T>, enable_if_array_t<T> = 0> Expr erf(const T &x);

template <typename T, bool Recurse, typename Expr, enable_if_array_t<T>>
Expr erfc(const T &x) {
    constexpr bool Single = std::is_same<scalar_t<T>, float>::value;
    using Scalar = scalar_t<T>;

    Expr r;
    if (Expr::Approx) {
        Expr xa = abs(x),
             z  = exp(-x*x);

        auto erf_mask   = xa < Scalar(1),
             large_mask = xa > Scalar(Single ? 2 : 8);

        if (Single) {
            Expr q  = rcp(xa),
                 y  = q*q, p_small, p_large;

            if (!all_nested(large_mask))
                p_small = poly8(y, 5.638259427386472e-1, -2.741127028184656e-1,
                                   3.404879937665872e-1, -4.944515323274145e-1,
                                   6.210004621745983e-1, -5.824733027278666e-1,
                                   3.687424674597105e-1, -1.387039388740657e-1,
                                   2.326819970068386e-2);

            if (any_nested(large_mask))
                p_large = poly7(y, 5.641895067754075e-1, -2.820767439740514e-1,
                                   4.218463358204948e-1, -1.015265279202700e+0,
                                   2.921019019210786e+0, -7.495518717768503e+0,
                                   1.297719955372516e+1, -1.047766399936249e+1);
            r = z * q * select(large_mask, p_large, p_small);
        } else {
            Expr p_small, p_large, q_small, q_large;

            if (!all_nested(large_mask)) {
                p_small = poly8(xa, 5.57535335369399327526e2, 1.02755188689515710272e3,
                                    9.34528527171957607540e2, 5.26445194995477358631e2,
                                    1.96520832956077098242e2, 4.86371970985681366614e1,
                                    7.46321056442269912687e0, 5.64189564831068821977e-1,
                                    2.46196981473530512524e-10);

                q_small = poly8(xa, 5.57535340817727675546e2, 1.65666309194161350182e3,
                                    2.24633760818710981792e3, 1.82390916687909736289e3,
                                    9.75708501743205489753e2, 3.54937778887819891062e2,
                                    8.67072140885989742329e1, 1.32281951154744992508e1,
                                    1.00000000000000000000e0);
            }


            if (any_nested(large_mask)) {
                p_large = poly5(xa, 2.97886665372100240670e0, 7.40974269950448939160e0,
                                    6.16021097993053585195e0, 5.01905042251180477414e0,
                                    1.27536670759978104416e0, 5.64189583547755073984e-1);

                q_large = poly6(xa, 3.36907645100081516050e0, 9.60896809063285878198e0,
                                    1.70814450747565897222e1, 1.20489539808096656605e1,
                                    9.39603524938001434673e0, 2.26052863220117276590e0,
                                    1.00000000000000000000e0);
            }

            r = (z * select(large_mask, p_large, p_small)) /
                     select(large_mask, q_large, q_small);

            r &= neq(z, zero<Expr>());
        }

        r[x < Scalar(0)] = Scalar(2) - r;

        if (ENOKI_UNLIKELY(Recurse && any_nested(erf_mask)))
            r[erf_mask] = Scalar(1) - erf<T, false>(x);
    } else {
        for (size_t i = 0; i < Expr::Size; ++i)
            r.coeff(i) = enoki::erfc(x.coeff(i));
    }
    return r;
}

template <typename T, bool Recurse, typename Expr, enable_if_array_t<T>>
Expr erf(const T &x) {
    constexpr bool Single = std::is_same<scalar_t<T>, float>::value;
    using Scalar = scalar_t<T>;

    Expr r;
    if (Expr::Approx) {
        auto erfc_mask = abs(x) > Scalar(1);

        Expr z = x * x;

        if (Single) {
            r = poly6(z, 1.128379165726710e+0, -3.761262582423300e-1,
                         1.128358514861418e-1, -2.685381193529856e-2,
                         5.188327685732524e-3, -8.010193625184903e-4,
                         7.853861353153693e-5);
        } else {
            r = poly4(z, 5.55923013010394962768e4, 7.00332514112805075473e3,
                         2.23200534594684319226e3, 9.00260197203842689217e1,
                         9.60497373987051638749e0) /
                poly5(z, 4.92673942608635921086e4, 2.26290000613890934246e4,
                         4.59432382970980127987e3, 5.21357949780152679795e2,
                         3.35617141647503099647e1, 1.00000000000000000000e0);
        }

        r *= x;

        if (ENOKI_UNLIKELY(Recurse && any_nested(erfc_mask)))
            r[erfc_mask] = Scalar(1) - erfc<T, false>(x);
    } else {
        for (size_t i = 0; i < Expr::Size; ++i)
            r.coeff(i) = enoki::erf(x.coeff(i));
    }
    return r;
}

// Inverse real error function approximation based on on "Approximating the
// erfinv function" by Mark Giles
template <typename T, typename Expr = expr_t<T>> Expr erfinv(const T &x_) {
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
template <typename T, typename Expr = expr_t<T>> Expr dawson(const T &x) {
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
template <typename T, typename Expr = expr_t<T>> Expr erfi(const T &x) {
    using Scalar = scalar_t<T>;

    return Scalar(M_2_SQRTPI) * dawson(x) * exp(x*x);
}

/// Modified Bessel function of the first kind, order zero (exponentially scaled)
template <typename T, typename Expr = expr_t<T>> Expr i0e(const T &x_) {
    using Scalar = scalar_t<T>;

    /* Chebyshev coefficients for exp(-x) I0(x)
     * in the interval [0,8].
     *
     * lim(x->0) { exp(-x) I0(x) } = 1.
     */

    static double A[] = {
        -1.30002500998624804212E-8, 6.04699502254191894932E-8,
        -2.67079385394061173391E-7, 1.11738753912010371815E-6,
        -4.41673835845875056359E-6, 1.64484480707288970893E-5,
        -5.75419501008210370398E-5, 1.88502885095841655729E-4,
        -5.76375574538582365885E-4, 1.63947561694133579842E-3,
        -4.32430999505057594430E-3, 1.05464603945949983183E-2,
        -2.37374148058994688156E-2, 4.93052842396707084878E-2,
        -9.49010970480476444210E-2, 1.71620901522208775349E-1,
        -3.04682672343198398683E-1, 6.76795274409476084995E-1
    };


    /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
     * in the inverted interval [8,infinity].
     *
     * lim(x->inf) { exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
     */

    static double B[] = {
        3.39623202570838634515E-9, 2.26666899049817806459E-8,
        2.04891858946906374183E-7, 2.89137052083475648297E-6,
        6.88975834691682398426E-5, 3.36911647825569408990E-3,
        8.04490411014108831608E-1
    };


    Expr x = abs(x_);

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
