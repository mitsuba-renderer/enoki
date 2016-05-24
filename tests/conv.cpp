/*
    tests/conv.cpp -- tests value and mask conversion routines

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include "test.h"
#include <enoki/half.h>


template <typename T, typename Scalar2> void convtest() {
    using T2 = like_t<T, Scalar2>;
    auto value1 = index_sequence<T>();
    auto value2 = T2(value1);
    auto value3 = T(value2);
    assert(value1 == value3);
}

template <typename T, typename Scalar2> void masktest() {
    using Scalar = typename T::Scalar;
    using T2 = like_t<T, Scalar2>;
    for (size_t i = 0; i < T::Size; ++i) {
        typename T::Mask mask = eq(index_sequence<T>() - T(Scalar(i)), T(0));
        typename T2::Mask mask2 = reinterpret_array<typename T2::Mask>(mask);
        T2 result = select(mask2, T2(Scalar2(1)), T2(Scalar2(0)));
        Scalar2 out[T::Size];
        store_unaligned(out, result);
        for (size_t j = 0; j < T::Size; ++j)
            assert(out[j] == ((j == i) ? Scalar2(1) : Scalar2(0)));
    }
}

ENOKI_TEST_ALL(test01_conv_int32_t)  { convtest<T, int32_t>();  }
ENOKI_TEST_ALL(test02_conv_uint32_t) { convtest<T, uint32_t>(); }
ENOKI_TEST_ALL(test03_conv_int64_t)  { convtest<T, int64_t>();  }
ENOKI_TEST_ALL(test04_conv_uint64_t) { convtest<T, uint64_t>(); }
ENOKI_TEST_ALL(test05_conv_half)     { convtest<T, half>();     }
ENOKI_TEST_ALL(test06_conv_float)    { convtest<T, float>();    }
ENOKI_TEST_ALL(test07_conv_double)   { convtest<T, double>();   }

ENOKI_TEST_ALL(test08_mask_int32_t)  { masktest<T, int32_t>();  }
ENOKI_TEST_ALL(test09_mask_uint32_t) { masktest<T, uint32_t>(); }
ENOKI_TEST_ALL(test10_mask_int64_t)  { masktest<T, int64_t>();  }
ENOKI_TEST_ALL(test11_mask_uint64_t) { masktest<T, uint64_t>(); }
ENOKI_TEST_ALL(test12_mask_float)    { masktest<T, float>();    }
ENOKI_TEST_ALL(test13_mask_double)   { masktest<T, double>();   }
ENOKI_TEST_ALL(test14_mask_half)     { masktest<T, half>();     }
