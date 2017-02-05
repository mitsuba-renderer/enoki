/*
    tests/horiz.cpp -- tests horizontal operators involving different types

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include "test.h"

ENOKI_TEST_ALL(test01_hsum) {
    auto sample = test::sample_values<Scalar>();

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return hsum(a); },
        [](const T &a) -> Scalar {
            Scalar result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result += a[i];
            return result;
        },
        Scalar(1e-5f));
}

ENOKI_TEST_ALL(test02_hprod) {
    auto sample = test::sample_values<Scalar>();

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return hprod(a); },
        [](const T &a) -> Scalar {
            Scalar result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result *= a[i];
            return result;
        },
        Scalar(1e-5f));
}

ENOKI_TEST_ALL(test03_hmin) {
    auto sample = test::sample_values<Scalar>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return hmin(a); },
        [](const T &a) -> Scalar {
            Scalar result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result = std::min(result, a[i]);
            return result;
        }
    );
}

ENOKI_TEST_ALL(test04_hmax) {
    auto sample = test::sample_values<Scalar>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return hmax(a); },
        [](const T &a) -> Scalar {
            Scalar result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result = std::max(result, a[i]);
            return result;
        }
    );
}

ENOKI_TEST_ALL(test05_all) {
    auto sample = test::sample_values<Scalar>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return Scalar(all(a >= a[0]) ? 1 : 0); },
        [](const T &a) -> Scalar {
            bool result = true;
            for (size_t i = 0; i < Size; ++i)
                result &= a[i] >= a[0];
            return Scalar(result ? 1 : 0);
        }
    );

    assert(all(typename T::Mask(true)));
    assert(!all(typename T::Mask(false)));
}

ENOKI_TEST_ALL(test06_none) {
    auto sample = test::sample_values<Scalar>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return Scalar(none(a > a[0]) ? 1 : 0); },
        [](const T &a) -> Scalar {
            bool result = false;
            for (size_t i = 0; i < Size; ++i)
                result |= a[i] > a[0];
            return Scalar(result ? 0 : 1);
        }
    );

    assert(!none(typename T::Mask(true)));
    assert(none(typename T::Mask(false)));
}

ENOKI_TEST_ALL(test07_any) {
    auto sample = test::sample_values<Scalar>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return Scalar(any(a > a[0]) ? 1 : 0); },
        [](const T &a) -> Scalar {
            bool result = false;
            for (size_t i = 0; i < Size; ++i)
                result |= a[i] > a[0];
            return Scalar(result ? 1 : 0);
        }
    );
    assert(any(typename T::Mask(true)));
    assert(!any(typename T::Mask(false)));
}

ENOKI_TEST_ALL(test08_count) {
    auto sample = test::sample_values<Scalar>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Scalar { return Scalar(count(a > a[0])); },
        [](const T &a) -> Scalar {
            int result = 0;
            for (size_t i = 0; i < Size; ++i)
                result += (a[i] > a[0]) ? 1 : 0;
            return Scalar(result);
        }
    );
    assert(Size == count(typename T::Mask(true)));
    assert(0 == count(typename T::Mask(false)));
}


ENOKI_TEST_ALL(test09_dot) {
    auto sample = test::sample_values<Scalar>();
    T value1, value2;
    Scalar ref = 0;

    size_t idx = 0;

    for (size_t i = 0; i < sample.size(); ++i) {
        for (size_t j = 0; j < sample.size(); ++j) {
            Scalar arg_i = sample[i], arg_j = sample[j];
            value1[idx] = arg_i; value2[idx] = arg_j;
            ref += arg_i * arg_j;
            idx++;

            if (idx == value1.size()) {
                Scalar result = dot(value1, value2);
                test::assert_close(result, ref, Scalar(1e-6f));
                idx = 0; ref = 0;
            }
        }
    }

    if (idx > 0) {
        while (idx < Size) {
            value1[idx] = 0;
            value2[idx] = 0;
            idx++;
        }
        Scalar result = dot(value1, value2);
        test::assert_close(result, ref, Scalar(1e-6f));
    }
}
