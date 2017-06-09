/*
    tests/call.cpp -- tests vectorized function calls

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"

struct Test;
struct Test2;

using Int32P = Array<int, 4>;
using TestPtrP = Array<const Test*, 4>;
using Test2PtrP = Array<Test2*, 4>;

struct Test {
    Test(int32_t value) : value(value) { }
    virtual ~Test() { }

    // A vector function call that accepts a mask
    using Mask = mask_t<Array<Test *, 4>>;

    virtual Int32P my_function(Mask /* unused */, Int32P i) const { return i + value; }

    int32_t value;
};

struct Test2 {

    Test2(bool working) : working(working) { }
    virtual ~Test2() { }

    // A normal (non-vectorized) function without arguments
    virtual bool is_working() const { return working; }
    virtual void toggle_working() { working = !working; }
    virtual void set_working(bool w) { working = w; }
    // A normal (non-vectorized) function with arguments
    virtual int add(int a, int b) const {
        if (working) return 10;
        return a + b;
    }

    bool working;
};

/* Allow Enoki arrays containing pointers to transparently forward function
   calls (with the appropriate masks) */
NAMESPACE_BEGIN(enoki)
ENOKI_CALL_HELPER_BEGIN(TestPtrP)
ENOKI_CALL_HELPER_FUNCTION(my_function)
ENOKI_CALL_HELPER_END(TestPtrP)

ENOKI_CALL_HELPER_BEGIN(Test2PtrP)
ENOKI_CALL_HELPER_FUNCTION(is_working)
ENOKI_CALL_HELPER_FUNCTION(set_working)
ENOKI_CALL_HELPER_FUNCTION(toggle_working)
ENOKI_CALL_HELPER_FUNCTION(add)
ENOKI_CALL_HELPER_END(Test2PtrP)
NAMESPACE_END(enoki)

ENOKI_TEST(test00_pointer_forward_vectorized_call) {
    Test *a = new Test(10);
    Test *b = new Test(20);

    TestPtrP pointers(a);
    pointers.coeff(2) = b;

    auto result = pointers->my_function(index_sequence<Int32P>());
    assert(result == Int32P(10, 11, 22, 13));

    delete a;
    delete b;
}

ENOKI_TEST(test01_pointer_forward_simple_call) {
    // Not a mask, just an array of bool
    using BoolP = Array<bool, Test2PtrP::Size>;
    using IntP = Array<int, Test2PtrP::Size>;

    Test2 *a = new Test2(false);
    Test2 *b = new Test2(true);
    Test2 *c = new Test2(false);
    Test2 *d = new Test2(true);

    Test2PtrP pointers(a);
    pointers.coeff(1) = b;
    pointers.coeff(2) = c;
    pointers.coeff(3) = d;

    assert(pointers->is_working() == BoolP(false, true, false, true));

    pointers->toggle_working();
    assert(pointers->is_working() == BoolP(true, false, true, false));
    assert(a->is_working() == true);
    assert(b->is_working() == false);

    pointers->set_working(true);
    assert(pointers->is_working() == BoolP(true, true, true, true));
    assert(a->is_working() == true);
    assert(b->is_working() == true);

    b->set_working(false);

    const auto add_result = pointers->add(5, 1);
    assert(add_result == IntP(10, 6, 10, 10));

    delete a;
    delete b;
}

