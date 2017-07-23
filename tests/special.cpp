/*
    tests/conv.cpp -- tests special functions

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <enoki/special.h>

ENOKI_TEST_FLOAT(test01_i0e)  {
    using Scalar = scalar_t<T>;

    double results[] = {
        1.000000000,  0.4657596076, 0.3085083226, 0.2430003542,
        0.2070019212, 0.1835408126, 0.1666574326, 0.1537377447,
        0.1434317819, 0.1349595246, 0.1278333372, 0.1217301682,
        0.1164262212, 0.1117608338, 0.1076152517, 0.1038995314
    };

    for (int i = 0; i < 16; ++i)
        assert(hmax(abs(i0e(T(Scalar(i))) - T(Scalar(results[i])))) < 1e-6);
}

ENOKI_TEST_FLOAT(test02_erf) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return erf(a); },
        [](double a) { return std::erf(a); },
        Value(-1), Value(1), 6
    );

    Array<T, 4> x((Value) 0.5);
    Array<T&, 4> y(x);
    assert(erf(x) == erf(y));
}

ENOKI_TEST_FLOAT(test02_erfc) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return erfc(a); },
        [](double a) { return std::erfc(a); },
        Value(-1), Value(1), 17
    );

    Array<T, 4> x((Value) 0.5);
    Array<T&, 4> y(x);
    assert(erfc(x) == erfc(y));
}

ENOKI_TEST_FLOAT(test04_erfinv) {
    for (int i = 0; i < 1000; ++i) {
        auto f = T((float) i / 1000.0f * 2 - 1 + 1e-6f);
        auto inv = erfinv(f);
        auto f2 = erf(inv);
        assert(std::abs(T(f-f2)[0]) < 1e-6f);
    }
}

ENOKI_TEST_FLOAT(test05_dawson)  {
    using Scalar = scalar_t<T>;

    double results[] = { 0.0,
        0.09933599239785286, 0.1947510333680280, 0.2826316650213119,
        0.3599434819348881, 0.4244363835020223, 0.4747632036629779,
        0.5105040575592318, 0.5321017070563654, 0.5407243187262987,
        0.5380795069127684, 0.5262066799705525, 0.5072734964077396,
        0.4833975173848241, 0.4565072375268973, 0.4282490710853986,
        0.3999398943230814, 0.3725593489740788, 0.3467727691148722,
        0.3229743193228178, 0.3013403889237920, 0.2818849389255278,
        0.2645107599508320, 0.2490529568377667, 0.2353130556638426,
        0.2230837221674355, 0.2121651242424990, 0.2023745109105140,
        0.1935507238593668, 0.1855552345354998, 0.1782710306105583 };


    for (int i = 0; i <= 30; ++i) {
        assert(hmax(abs(dawson(T(Scalar(i * 0.1)))  - T(Scalar( results[i])))) < 1e-6);
        assert(hmax(abs(dawson(T(Scalar(i * -0.1))) - T(Scalar(-results[i])))) < 1e-6);
    }
}
