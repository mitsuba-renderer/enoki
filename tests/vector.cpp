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
#include <enoki/matrix.h>

template <typename T, std::enable_if_t<std::is_signed<typename T::Value>::value, int> = 0>
void test01_dot_signed() {
    using Value = typename T::Value;
    constexpr size_t Size = T::Size;
    Value expected1 = Value((Size * (2 * Size - 1) * (Size - 1)) / 6);
    Value expected2 = Value((Size * (Size - 1)) / 2);
    T value = index_sequence<T>();
    T value2(Value(1));
    assert(dot(value, -value) == -expected1);
    assert(dot(value, -value2) == -expected2);
    assert(abs_dot(value, -value) == expected1);
    assert(abs_dot(value, -value2) == expected2);
}

template <typename T, std::enable_if_t<!std::is_signed<typename T::Value>::value, int> = 0>
void test01_dot_signed() { }

ENOKI_TEST_ALL(test01_dot) {
    Value expected1 = Value((Size * (2 * Size - 1) * (Size - 1)) / 6);
    Value expected2 = Value((Size * (Size - 1)) / 2);
    T value = index_sequence<T>();
    T value2(Value(1));
    assert(dot(value, value)  == expected1);
    assert(dot(value, value2) == expected2);
    test01_dot_signed<T>();
}

template <typename T> void test02_vecops_float() {
    typedef typename T::Value Value;

    /* Extra tests for horizontal operations */
    T v(1.f, 2.f, 3.f);
    assert(v.x() == 1);
    assert(v.y() == 2);
    assert(v.z() == 3);
    assert(v.x() == Value(1));
    assert(v.y() == Value(2));
    assert(v.z() == Value(3));
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
    assert(!any(v < Value(0)));

    v = T(-1);
    if (T::ActualSize == 4)
        v.coeff(3) = 0;
    assert(all(v < Value(0)));

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

ENOKI_TEST(array_float_05_outer_product) {
    using Vector4f = Array<float, 4>;
    using Vector3f = Array<float, 3>;

    assert(to_string(broadcast<Vector3f>(Vector4f(1, 2, 3, 4)) * Vector3f(0, 1, 0)) == "[[0, 1, 0],\n [0, 2, 0],\n [0, 3, 0],\n [0, 4, 0]]");
    assert(to_string(broadcast<float>(Vector4f(1, 2, 3, 4)) * 3.f) == "[3, 6, 9, 12]");
    assert(to_string(broadcast<Vector3f>(3.f) * Vector3f(1, 2, 3)) == "[3, 6, 9]");
}

ENOKI_TEST(array_float_06_head_tail) {
    using T  = Array<float, 4>;

    auto t = T(1, 2, 3, 4);
    assert(to_string(head<2>(t)) == "[1, 2]");
    assert(to_string(tail<2>(t)) == "[3, 4]");
}

ENOKI_TEST(array_float_07_matrix) {
    using M2f = Matrix<float, 2>;
    using V2f = Array<float, 2>;

    auto a = M2f(1, 2, 3, 4);
    auto b = V2f(1, 1);
    assert(to_string(a) == "[[1, 2],\n [3, 4]]");
    assert(to_string(transpose(a)) == "[[1, 3],\n [2, 4]]");
    assert(a(0, 1) == 2);
    assert(to_string(a.coeff(0)) == "[1, 3]");
    assert(to_string(a.coeff(1)) == "[2, 4]");
    assert(to_string(a*a) == "[[7, 10],\n [15, 22]]");
    assert(to_string(a*a) == "[[7, 10],\n [15, 22]]");
    assert(to_string(a*b) == "[3, 7]");
}
