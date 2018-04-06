/*
    tests/custom.cpp -- tests operations involving custom data structures

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <enoki/array.h>

template <typename Value_> struct Custom {
    using Value = Value_;
    using FloatVector = Array<Value, 3>;
    using DoubleVector = float64_array_t<FloatVector>;

    FloatVector o;
    DoubleVector d;

    template <typename T>
    bool operator==(const Custom<T> &other) const {
        return other.o == o && other.d == d;
    }

    ENOKI_STRUCT(Custom, o, d)
};

ENOKI_STRUCT_DYNAMIC(Custom, o, d)

ENOKI_TEST(test01_mask_slice_custom) {
    using FloatP = Packet<float>;
    using Vector3f = Array<float, 3>;
    using Vector3d = Array<double, 3>;
    using Custom3f = Custom<float>;
    using Custom3fP = Custom<FloatP>;

    Custom3fP x = zero<Custom3fP>();
    Custom3fP y;
    y.o = Vector3f(1, 2, 3);
    y.d = Vector3d(4, 5, 6);
    auto mask = index_sequence<FloatP>() > 0.f;

    masked(x, mask) = y;

    assert((slice(x, 0) == Custom3f(Vector3f(0, 0, 0), Vector3f(0, 0, 0))));
    if (FloatP::Size > 1)
        assert((slice(x, 1) == Custom3f(Vector3f(1, 2, 3), Vector3f(4, 5, 6))));
}
