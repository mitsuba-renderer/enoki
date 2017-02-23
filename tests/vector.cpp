/*
    tests/basic.cpp -- tests linear algebra-related routines

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include "test.h"

template <typename T, std::enable_if_t<std::is_signed<typename T::Scalar>::value, int> = 0>
void test01_dot_signed() {
    using Scalar = typename T::Scalar;
    constexpr size_t Size = T::Size;
    Scalar expected1 = Scalar((Size * (2 * Size - 1) * (Size - 1)) / 6);
    Scalar expected2 = Scalar((Size * (Size - 1)) / 2);
    T value = index_sequence<T>();
    T value2(Scalar(1));
    assert(dot(value, -value) == -expected1);
    assert(dot(value, -value2) == -expected2);
    assert(abs_dot(value, -value) == expected1);
    assert(abs_dot(value, -value2) == expected2);
}

template <typename T, std::enable_if_t<!std::is_signed<typename T::Scalar>::value, int> = 0>
void test01_dot_signed() { }

ENOKI_TEST_ALL(test01_dot) {
    Scalar expected1 = Scalar((Size * (2 * Size - 1) * (Size - 1)) / 6);
    Scalar expected2 = Scalar((Size * (Size - 1)) / 2);
    T value = index_sequence<T>();
    T value2(Scalar(1));
    assert(dot(value, value)  == expected1);
    assert(dot(value, value2) == expected2);
    test01_dot_signed<T>();
}

template <typename T> void test02_vecops_float() {
    typedef typename T::Scalar Scalar;

    /* Extra tests for horizontal operations */
    T v(1.f, 2.f, 3.f);
    assert(v.x() == 1);
    assert(v.y() == 2);
    assert(v.z() == 3);
    assert(v.x() == Scalar(1));
    assert(v.y() == Scalar(2));
    assert(v.z() == Scalar(3));
    assert(std::abs(norm(v) - 3.74165738677394f) < 1e-5f);
    assert(std::abs(squared_norm(v) - std::pow(3.74165738677394f, 2.f)) < 1e-5f);
    assert(hsum(abs(normalize(v) - T(0.26726f, 0.53452f, 0.80178f))) < 1e-5f);
    assert(hsum(v) == 6.f);
    assert(hprod(v) == 6.f);
    assert(hmax(v) == 3.f);
    if (T::ActualSize == 4)
        v.coeff(3) = -4;
    assert(hmin(v) == 1.f);
    assert(dot(v, v) == 14.f);

    v = T(0);
    if (T::ActualSize == 4)
        v.coeff(3) = -1;
    assert(!any(v < Scalar(0)));

    v = T(-1);
    if (T::ActualSize == 4)
        v.coeff(3) = 0;
    assert(all(v < Scalar(0)));

    /* Test cross product */
    assert(cross(T(1, 2, 3), T(1, 0, 0)) == T(0, 3, -2));
    assert(cross(T(1, 2, 3), T(0, 1, 0)) == T(-3, 0, 1));
    assert(cross(T(1, 2, 3), T(0, 0, 1)) == T(2, -1, 0));
    assert(cross(T(1, 1, 1), T(1, 2, 3)) == T(1, -2, 1));

    v.x() = 3; v.y() = 4; v.z() = 5;
    if (T::ActualSize == 4)
        v.coeff(3) = -1;
    assert(v == T(3, 4, 5));
    assert(v != T(3, 2, 5));
}

ENOKI_TEST(test02_vecops_float) {
    test02_vecops_float<Array<float, 3>>();
}

ENOKI_TEST(test02_vecops_double) {
    test02_vecops_float<Array<double, 3>>();
}

ENOKI_TEST(array_float_04_transpose) {
    using T  = Array<float, 4>;
    using T2 = Array<T, 4>;

    assert(
        transpose(T2(T(1, 2, 3, 4), T(5, 6, 7, 8), T(9, 10, 11, 12), T(13, 14, 15, 16))) ==
        T2(T(1, 5, 9, 13), T(2, 6, 10, 14), T(3, 7, 11, 15), T(4, 8, 12, 16)));
}

ENOKI_TEST(array_double_04_transpose) {
    using T  = Array<double, 4>;
    using T2 = Array<T, 4>;

    assert(
        transpose(T2(T(1, 2, 3, 4), T(5, 6, 7, 8), T(9, 10, 11, 12), T(13, 14, 15, 16))) ==
        T2(T(1, 5, 9, 13), T(2, 6, 10, 14), T(3, 7, 11, 15), T(4, 8, 12, 16)));
}

ENOKI_TEST(array_float_04_outer_product) {
    using Vector4f = Array<float, 4>;
    using Vector3f = Array<float, 3>;

    assert(to_string(outer_product(Vector4f(1, 2, 3, 4), Vector3f(0, 1, 0))) == "[[0, 0, 0, 0],\n [1, 2, 3, 4],\n [0, 0, 0, 0]]");
    assert(to_string(outer_product(Vector4f(1, 2, 3, 4), 3.f)) == "[3, 6, 9, 12]");
    assert(to_string(outer_product(3.f, Vector3f(1, 2, 3))) == "[3, 6, 9]");
}
