/*
    tests/horiz.cpp -- tests horizontal operators involving different types

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"

ENOKI_TEST_ALL(test01_hsum) {
    auto sample = test::sample_values<Value>();

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return hsum(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result += a[i];
            return result;
        },
        Value(1e-5f));
}

ENOKI_TEST_ALL(test02_hprod) {
    auto sample = test::sample_values<Value>();

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return hprod(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result *= a[i];
            return result;
        },
        Value(1e-5f));
}

ENOKI_TEST_ALL(test03_hmin) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return hmin(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result = std::min(result, a[i]);
            return result;
        }
    );
}

ENOKI_TEST_ALL(test04_hmax) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return hmax(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result = std::max(result, a[i]);
            return result;
        }
    );
}

ENOKI_TEST_ALL(test05_all) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(all(a >= a[0]) ? 1 : 0); },
        [](const T &a) -> Value {
            bool result = true;
            for (size_t i = 0; i < Size; ++i)
                result &= a[i] >= a[0];
            return Value(result ? 1 : 0);
        }
    );

    assert(all(mask_t<T>(true)));
    assert(!all(mask_t<T>(false)));
}

ENOKI_TEST_ALL(test06_none) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(none(a > a[0]) ? 1 : 0); },
        [](const T &a) -> Value {
            bool result = false;
            for (size_t i = 0; i < Size; ++i)
                result |= a[i] > a[0];
            return Value(result ? 0 : 1);
        }
    );

    assert(!none(mask_t<T>(true)));
    assert(none(mask_t<T>(false)));
}

ENOKI_TEST_ALL(test07_any) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(any(a > a[0]) ? 1 : 0); },
        [](const T &a) -> Value {
            bool result = false;
            for (size_t i = 0; i < Size; ++i)
                result |= a[i] > a[0];
            return Value(result ? 1 : 0);
        }
    );
    assert(any(mask_t<T>(true)));
    assert(!any(mask_t<T>(false)));
}

ENOKI_TEST_ALL(test08_count) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(count(a > a[0])); },
        [](const T &a) -> Value {
            int result = 0;
            for (size_t i = 0; i < Size; ++i)
                result += (a[i] > a[0]) ? 1 : 0;
            return Value(result);
        }
    );
    assert(Size == count(mask_t<T>(true)));
    assert(0 == count(mask_t<T>(false)));
}


ENOKI_TEST_ALL(test09_dot) {
    auto sample = test::sample_values<Value>();
    T value1, value2;
    Value ref = 0;

    size_t idx = 0;

    for (size_t i = 0; i < sample.size(); ++i) {
        for (size_t j = 0; j < sample.size(); ++j) {
            Value arg_i = sample[i], arg_j = sample[j];
            value1[idx] = arg_i; value2[idx] = arg_j;
            ref += arg_i * arg_j;
            idx++;

            if (idx == value1.size()) {
                Value result = dot(value1, value2);
                test::assert_close(result, ref, Value(1e-6f));
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
        Value result = dot(value1, value2);
        test::assert_close(result, ref, Value(1e-6f));
    }
}
