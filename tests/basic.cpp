/*
    tests/basic.cpp -- tests basic operators involving different types

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include "test.h"

#if defined(_MSC_VER)
#  pragma warning(disable: 4146) //  warning C4146: unary minus operator applied to unsigned type, result still unsigned
#endif

ENOKI_TEST_ALL(test00_align) {
#if defined(__SSE4_2__)
    if (sizeof(Scalar)*Size == 16) {
        assert(sizeof(T) == 16);
        assert(alignof(T) == 16);
    }
#elif defined(__AVX__)
    if (sizeof(Scalar)*Size == 32) {
        assert(sizeof(T) == 32);
        assert(alignof(T) == 32);
    }
#elif defined(__AVX512__)
    if (sizeof(Scalar)*Size == 64) {
        assert(sizeof(T) == 64);
        assert(alignof(T) == 64);
    }
#endif
}

ENOKI_TEST_ALL(test01_add) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a + b; },
        [](Scalar a, Scalar b) -> Scalar { return a + b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x += b; return x; },
        [](Scalar a, Scalar b) -> Scalar { return a + b; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return a + Scalar(3); },
        [](Scalar a) -> Scalar { return a + Scalar(3); }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return Scalar(3) + a; },
        [](Scalar a) -> Scalar { return Scalar(3) + a; }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert((x + x == y + y) && (x + y == y + x));
}

ENOKI_TEST_ALL(test02_sub) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a - b; },
        [](Scalar a, Scalar b) -> Scalar { return a - b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x -= b; return x; },
        [](Scalar a, Scalar b) -> Scalar { return a - b; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return a - Scalar(3); },
        [](Scalar a) -> Scalar { return a - Scalar(3); }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return Scalar(3) - a; },
        [](Scalar a) -> Scalar { return Scalar(3) - a; }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert((x - x == y - y) && (x - y == y - x));
}

ENOKI_TEST_ALL(test03_mul) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a * b; },
        [](Scalar a, Scalar b) -> Scalar { return a * b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x *= b; return x; },
        [](Scalar a, Scalar b) -> Scalar { return a * b; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return a * Scalar(3); },
        [](Scalar a) -> Scalar { return a * Scalar(3); }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return Scalar(3) * a; },
        [](Scalar a) -> Scalar { return Scalar(3) * a; }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert((x * x == y * y) && (x * y == y * x));
}

ENOKI_TEST_ALL(test05_neg) {
    auto sample = test::sample_values<Scalar>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return -a; },
        [](Scalar a) -> Scalar { return -a; }
    );
}

ENOKI_TEST_ALL(test06_lt) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a < b, T(0), T(1)); },
        [](Scalar a, Scalar b) -> Scalar { return Scalar(a < b ? 0 : 1); }
    );

    assert(select(T(1) < T(2), T(1), T(2)) == T(1));
    assert(select(T(1) > T(2), T(1), T(2)) == T(2));

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(none(x < y));
}

ENOKI_TEST_ALL(test07_le) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a <= b, T(0), T(1)); },
        [](Scalar a, Scalar b) -> Scalar { return Scalar(a <= b ? 0 : 1); }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(all(x <= y));
}

ENOKI_TEST_ALL(test08_gt) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a > b, T(0), T(1)); },
        [](Scalar a, Scalar b) -> Scalar { return Scalar(a > b ? 0 : 1); }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(none(x > y));
}

ENOKI_TEST_ALL(test09_ge) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a >= b, T(0), T(1)); },
        [](Scalar a, Scalar b) -> Scalar { return Scalar(a >= b ? 0 : 1); }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(all(x >= y));
}

ENOKI_TEST_ALL(test10_eq) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(eq(a, b), T(0), T(1)); },
        [](Scalar a, Scalar b) -> Scalar { return Scalar(a == b ? 0 : 1); }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(all(eq(x, x)) && all(eq(y, y)) && all(eq(x, y)) &&
           all(eq(y, x)) && all(eq(y, Scalar(5))) && y == Scalar(5));
}

ENOKI_TEST_ALL(test11_neq) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(neq(a, b), T(0), T(1)); },
        [](Scalar a, Scalar b) -> Scalar { return Scalar(a != b ? 0 : 1); }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(none(neq(x, x)) && none(neq(y, y)) && none(neq(x, y)) &&
           none(neq(y, x)) && none(neq(y, Scalar(5))) && y != Scalar(1));
}

ENOKI_TEST_ALL(test12_min) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return min(a, b); },
        [](Scalar a, Scalar b) -> Scalar { return std::min(a, b); }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(min(x, y) == y);
}

ENOKI_TEST_ALL(test13_max) {
    auto sample = test::sample_values<Scalar>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return max(a, b); },
        [](Scalar a, Scalar b) -> Scalar { return std::max(a, b); }
    );

    Array<T, 4> x(5); Array<T&, 4> y(x);
    assert(max(x, y) == y);
}

ENOKI_TEST_ALL(test14_abs) {
    auto sample = test::sample_values<Scalar>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return abs(a); },
        [](Scalar a) -> Scalar { return enoki::abs(a); }
    );
}

ENOKI_TEST_ALL(test15_fmadd) {
    auto sample = test::sample_values<Scalar>();

    test::validate_ternary<T>(sample,
        [](const T &a, const T &b, const T& c) -> T { return fmadd(a, b, c); },
        [](Scalar a, Scalar b, Scalar c) -> Scalar {
            return a*b + c;
        },
        Scalar(1e-6)
    );
}

ENOKI_TEST_ALL(test16_fmsub) {
    auto sample = test::sample_values<Scalar>();

    test::validate_ternary<T>(sample,
        [](const T &a, const T &b, const T& c) -> T { return fmsub(a, b, c); },
        [](Scalar a, Scalar b, Scalar c) -> Scalar {
            return a*b - c;
        },
        Scalar(1e-6)
    );
}

ENOKI_TEST_ALL(test17_select) {
    auto sample = test::sample_values<Scalar>();

    test::validate_ternary<T>(sample,
        [](const T &a, const T &b, const T& c) -> T { return select(a > Scalar(5), b, c); },
        [](Scalar a, Scalar b, Scalar c) -> Scalar {
            return a > 5 ? b : c;
        }
    );
}

template <typename T, std::enable_if_t<T::Size == 1, int> = 0> void test18_shuffle_impl() {
    assert((enoki::shuffle<0>(T(1)) == T(1)));
}

template <typename T, std::enable_if_t<T::Size == 2, int> = 0> void test18_shuffle_impl() {
    assert((enoki::shuffle<0, 1>(T(1, 2)) == T(1, 2)));
    assert((enoki::shuffle<1, 0>(T(1, 2)) == T(2, 1)));
}

template <typename T, std::enable_if_t<T::Size == 3, int> = 0> void test18_shuffle_impl() {
    assert((enoki::shuffle<0, 1, 2>(T(1, 2, 3)) == T(1, 2, 3)));
    assert((enoki::shuffle<2, 1, 0>(T(1, 2, 3)) == T(3, 2, 1)));
}

template <typename T, std::enable_if_t<T::Size == 4, int> = 0> void test18_shuffle_impl() {
    assert((enoki::shuffle<0, 1, 2, 3>(T(1, 2, 3, 4)) == T(1, 2, 3, 4)));
    assert((enoki::shuffle<3, 2, 1, 0>(T(1, 2, 3, 4)) == T(4, 3, 2, 1)));
}

template <typename T, std::enable_if_t<T::Size == 8, int> = 0> void test18_shuffle_impl() {
    auto shuf1 = shuffle<0, 1, 2, 3, 4, 5, 6, 7>(T(1, 2, 3, 4, 5, 6, 7, 8));
    auto shuf2 = shuffle<7, 6, 5, 4, 3, 2, 1, 0>(T(1, 2, 3, 4, 5, 6, 7, 8));

    assert(shuf1 == T(1, 2, 3, 4, 5, 6, 7, 8));
    assert(shuf2 == T(8, 7, 6, 5, 4, 3, 2, 1));
}

template <typename T, std::enable_if_t<T::Size == 16, int> = 0> void test18_shuffle_impl() {
    auto shuf1 = shuffle<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>(
        T(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    auto shuf2 = shuffle<15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0>(
        T(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));

    assert(shuf1 == T(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    assert(shuf2 == T(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
}

template <typename T, std::enable_if_t<(T::Size > 16), int> = 0> void test18_shuffle_impl() {
    std::cout << "[skipped] ";
}

ENOKI_TEST_ALL(test18_shuffle) {
    test18_shuffle_impl<T>();
}

template <typename T,
          std::enable_if_t<T::Size != (T::Size / 2) * 2, int> = 0>
void test19_lowhi_impl() {
    std::cout << "[skipped] ";
}
template <typename T,
          std::enable_if_t<T::Size == (T::Size / 2) * 2, int> = 0>
void test19_lowhi_impl() {
    auto value = index_sequence<T>();
    assert(T(low(value), high(value)) == value);
}
ENOKI_TEST_ALL(test19_lowhi) { test19_lowhi_impl<T>(); }

ENOKI_TEST_ALL(test20_iterator) {
    Scalar j(0);
    for (Scalar i : index_sequence<T>()) {
        assert(i == j);
        j += 1;
    }
}

ENOKI_TEST_ALL(test21_mask_assign) {
    T x = index_sequence<T>();
    x[x > Scalar(0)] = x + Scalar(1);
    if (Size >= 2) {
        assert(x.coeff(0) == Scalar(0));
        assert(x.coeff(1) == Scalar(2));
    }
    x[x > Scalar(0)] = Scalar(-1);
    if (Size >= 2) {
        assert(x.coeff(0) == Scalar(0));
        assert(x.coeff(1) == Scalar(-1));
    }
}

