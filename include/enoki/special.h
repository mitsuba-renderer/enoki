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
