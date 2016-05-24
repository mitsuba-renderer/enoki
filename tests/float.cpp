/*
    tests/basic.cpp -- tests basic floating point operations

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include "test.h"
#include <enoki/half.h>

ENOKI_TEST_FLOAT(test01_div_fp) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a / b; },
        [](Scalar a, Scalar b) -> Scalar { return a / b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x /= b; return x; },
        [](Scalar a, Scalar b) -> Scalar { return a / b; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return a / Scalar(3); },
        [](Scalar a) -> Scalar { return a / Scalar(3); }, 1e-6f
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return Scalar(3) / a; },
        [](Scalar a) -> Scalar { return Scalar(3) / a; }, 1e-6f
    );

    if (std::is_same<Scalar, float>::value && T::Approx && has_sse42) {
        using T2 = Array<float, T::Size, false>;
        // Make sure that division optimization is used in approximate mode
        assert(T (3.f) / 3.f != T (1.f));
        assert(T2(3.f) / 3.f == T2(1.f));
    }
}

ENOKI_TEST_FLOAT(test02_ceil) {
    auto sample = test::sample_values<Scalar>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return ceil(a); },
        [](Scalar a) -> Scalar { return std::ceil(a); }
    );

    Array<T, 4> x(3.4f); Array<T&, 4> y(x);
    assert(ceil(x) == ceil(y));
}

ENOKI_TEST_FLOAT(test03_floor) {
    auto sample = test::sample_values<Scalar>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return floor(a); },
        [](Scalar a) -> Scalar { return std::floor(a); }
    );

    Array<T, 4> x(3.4f); Array<T&, 4> y(x);
    assert(floor(x) == floor(y));
}

ENOKI_TEST_FLOAT(test04_round) {
    auto sample = test::sample_values<Scalar>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return round(a); },
        [](Scalar a) -> Scalar { return std::rint(a); }
    );

    Array<T, 4> x(3.4f); Array<T&, 4> y(x);
    assert(round(x) == round(y));
}

ENOKI_TEST_FLOAT(test05_sqrt) {
    auto sample = test::sample_values<Scalar>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return sqrt(a); },
        [](Scalar a) -> Scalar { return std::sqrt(a); }
    );

    Array<T, 4> x(3.4f); Array<T&, 4> y(x);
    assert(sqrt(x) == sqrt(y));
}

ENOKI_TEST_FLOAT(test06_rsqrt) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return rsqrt(a); },
        [](double a) { return 1/std::sqrt(a); },
        Scalar(1e-6), Scalar(1024), 3, false
    );
}

ENOKI_TEST_FLOAT(test07_rcp) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return rcp(a); },
        [](double a) { return 1 / a; },
        Scalar(1e-6), Scalar(1024), 2, false
    );
}

ENOKI_TEST_FLOAT(test08_sign) {
    auto sample = test::sample_values<Scalar>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return enoki::sign(a); },
        [](Scalar a) -> Scalar { return std::copysign(1.f, a); }
    );

    Array<T, 4> x(3.4f); Array<T&, 4> y(x);
    assert(sign(x) == sign(y));
}

ENOKI_TEST_FLOAT(test09_isinf) {
    auto sample = test::sample_values<Scalar>();

    using enoki::isinf;
    test::validate_unary<T>(sample,
        [](const T &a) -> T { return select(enoki::isinf(a), T(1), T(0)); },
        [](Scalar a) -> Scalar { return Scalar(std::isinf(a) ? 1 : 0); }
    );
}

ENOKI_TEST_FLOAT(test10_isnan) {
    auto sample = test::sample_values<Scalar>();

    using enoki::isnan;
    test::validate_unary<T>(sample,
        [](const T &a) -> T { return select(enoki::isnan(a), T(1), T(0)); },
        [](Scalar a) -> Scalar { return Scalar(std::isnan(a) ? 1 : 0); }
    );
}

ENOKI_TEST_FLOAT(test11_isfinite) {
    auto sample = test::sample_values<Scalar>();

    using enoki::isfinite;
    test::validate_unary<T>(sample,
        [](const T &a) -> T { return select(enoki::isfinite(a), T(1), T(0)); },
        [](Scalar a) -> Scalar { return Scalar(std::isfinite(a) ? 1 : 0); }
    );
}

ENOKI_TEST_FLOAT(test12_nan_initialization) {
    T x;
    for (size_t i = 0; i < Size; ++i)
        assert(std::isnan(x[i]));
}

ENOKI_TEST(test13_half) {
    using T = Array<float, 4>;
    using THalf = Array<half, 4>;

    for (uint32_t i = 0; i < 0xFFFF; ++i) {
        uint16_t data[8]  = { (uint16_t) i };
        uint16_t data3[8] = { (uint16_t) i };

        float f1 = T(load<THalf>((const half *) data))[0];
        float f2 = (float) half::from_binary(data[0]);

        bool both_nan = std::isnan(f1) && std::isnan(f2);

        assert((memcpy_cast<uint32_t>(f1) ==
                memcpy_cast<uint32_t>(f2)) || both_nan);

        half data2[8];
        store(data2, THalf(T(f1)));
        data3[0] = half(f2).value;

        assert((data2[0].value == data3[0]) || both_nan);
    }
}

ENOKI_TEST_FLOAT(test14_round) {
    using T1 = Array<Scalar, Size, T::Approx, RoundingMode::Up>;
    using T2 = Array<Scalar, Size, T::Approx, RoundingMode::Down>;

    T1 a = T1(Scalar(M_PI)) * T1(Scalar(M_PI));
    T2 b = T2(Scalar(M_PI)) * T2(Scalar(M_PI));

    assert(a[0] > b[0]);

    if (std::is_same<Scalar, float>::value) {
        using T3 = Array<double, Size>;
        a = T1(T3(M_PI));
        b = T2(T3(M_PI));
        assert(a[0] > b[0]);
    }
}
