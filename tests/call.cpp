/*
    tests/complex.cpp -- tests vectorized function calls

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"

struct Test;

using Int32P = Array<int, 4>;
using TestPtrP = Array<const Test*, 4>;

struct Test {
    Test(int32_t value) : value(value) { }
    virtual ~Test() { }

    // A vector function call that accepts a mask
    using Mask = mask_t<Array<Test *, 4>>;

    virtual Int32P my_function(Mask /* unused */, Int32P i) const { return i + value; }

    int32_t value;
};

/* Allow Enoki arrays containing pointers to transparently forward function
   calls (with the appropriate masks) */
NAMESPACE_BEGIN(enoki)
ENOKI_CALL_HELPER_BEGIN(TestPtrP)
ENOKI_CALL_HELPER_FUNCTION(my_function)
ENOKI_CALL_HELPER_END(Test)
NAMESPACE_END(enoki)

ENOKI_TEST(test00_complex_str) {
    Test *a = new Test(10);
    Test *b = new Test(20);

    TestPtrP pointers(a);
    pointers.coeff(2) = b;

    auto result = pointers->my_function(index_sequence<Int32P>());
    assert(result == Int32P(10, 11, 22, 13));

    delete a;
    delete b;
}

