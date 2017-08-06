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
#include <enoki/stl.h>

struct Test;

using Int32P = Array<int>;
using TestP = Array<const Test*, Int32P::Size>;

using TestPMask = mask_t<TestP>;

using Int32X = DynamicArray<Int32P>;
using TestX = DynamicArray<TestP>;


struct Test {
    Test(int32_t value) : value(value) { }
    virtual ~Test() { }

    // Vectorized function (accepts a mask, which is ignored here)
    virtual Int32P func1(Int32P i, TestPMask /* unused */) const { return i + value; }

    // Vectorized function (accepts a mask, which is ignored here)
    virtual void func2(Int32P &i, TestPMask mask) const { i[mask] += value; }

    bool func3() const { return value == 20; }

    std::pair<Int32P, Int32P> func4(TestPMask) const { return std::make_pair(value, value+1); }

    int32_t value;
};

/* Allow Enoki arrays containing pointers to transparently forward function
   calls (with the appropriate masks) */
ENOKI_CALL_SUPPORT_BEGIN(TestP)
ENOKI_CALL_SUPPORT(func1)
ENOKI_CALL_SUPPORT(func2)
ENOKI_CALL_SUPPORT_SCALAR(func3)
ENOKI_CALL_SUPPORT(func4)
ENOKI_CALL_SUPPORT_END(TestP)

ENOKI_TEST(test00_call) {
    size_t offset = std::min((size_t) 2, TestP::Size-1);
    Test *a = new Test(10);
    Test *b = new Test(20);

    TestP pointers(a);
    pointers.coeff(offset) = b;

    Int32P index = index_sequence<Int32P>();
    Int32P result = pointers->func1(index);
    Int32P ref = index_sequence<Int32P>() + 10;
    if (offset < Int32P::Size)
        ref.coeff(offset) += 10;
    assert(result == ref);

    Int32P ref2 = 10;
    if (offset < Int32P::Size)
        ref2.coeff(offset) += 10;
    std::pair<Int32P, Int32P> result2 = pointers->func4();
    assert(result2.first == ref2);
    assert(result2.second == ref2+1);

    TestX pointers_x;
    Int32X index_x;
    set_slices(pointers_x, TestP::Size);
    set_slices(index_x, TestP::Size);
    packet(pointers_x, 0) = pointers;
    packet(index_x, 0) = index;
    assert(result == ref);
    Int32X result_x = pointers_x->func1(index_x);
    assert(packet(result_x, 0) == ref);

    pointers->func2(index);
    assert(index == ref);

    auto mask = mask_t<TestP>(pointers->func3());
    assert(mask == eq(pointers, b));

    delete a;
    delete b;
}

