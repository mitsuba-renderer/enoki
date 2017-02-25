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

ENOKI_TEST_INT(test01_or) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a | b; },
        [](Value a, Value b) -> Value { return a | b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a || b; },
        [](Value a, Value b) -> Value { return a | b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x |= b; return x; },
        [](Value a, Value b) -> Value { return a | b; }
    );
}

ENOKI_TEST_INT(test02_and) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a & b; },
        [](Value a, Value b) -> Value { return a & b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a && b; },
        [](Value a, Value b) -> Value { return a & b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x &= b; return x; },
        [](Value a, Value b) -> Value { return a & b; }
    );
}

ENOKI_TEST_INT(test03_xor) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a ^ b; },
        [](Value a, Value b) -> Value { return a ^ b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x ^= b; return x; },
        [](Value a, Value b) -> Value { return a ^ b; }
    );
}

ENOKI_TEST_INT(test04_not) {
    auto sample = test::sample_values<Value>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return ~a; },
        [](Value a) -> Value { return ~a; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return !a; },
        [](Value a) -> Value { return ~a; }
    );
}

ENOKI_TEST_INT(test05_sign) {
    auto sample = test::sample_values<Value>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return enoki::sign(a); },
        [](Value a) -> Value { return a >= 0 ? Value(1) : Value(-1); }
    );
}

ENOKI_TEST_TYPE(test06_shiftrot, uint32_t) {
    assert((T(0xDEADBEEFu) >> 4) == T(0x0DEADBEEu));
    assert((T(0xDEADBEEFu) << 4) == T(0xEADBEEF0u));
    assert((sri<4>(T(0xDEADBEEFu)) == T(0x0DEADBEEu)));
    assert((sli<4>(T(0xDEADBEEFu)) == T(0xEADBEEF0u)));
    assert(rol(T(0xDEADBEEFu), T(4)) == T(0xEADBEEFDu));
    assert(ror(T(0xDEADBEEFu), T(4)) == T(0xFDEADBEEu));
    assert(rol(T(0xDEADBEEFu), 4) == T(0xEADBEEFDu));
    assert(ror(T(0xDEADBEEFu), 4) == T(0xFDEADBEEu));
    assert(roli<4>(T(0xDEADBEEFu)) == T(0xEADBEEFDu));
    assert(rori<4>(T(0xDEADBEEFu)) == T(0xFDEADBEEu));
}

ENOKI_TEST_TYPE(test06_shiftrot, int32_t) {
    assert((T((int32_t) 0xDEADBEEF) >> 4) == T((int32_t) 0xFDEADBEE));
    assert((T((int32_t) 0xDEADBEEF) << 4) == T((int32_t) 0xEADBEEF0));
    assert((sri<4>(T((int32_t) 0xDEADBEEF)) == T((int32_t) 0xFDEADBEE)));
    assert((sli<4>(T((int32_t) 0xDEADBEEF)) == T((int32_t) 0xEADBEEF0)));
    assert(rol(T((int32_t) 0xDEADBEEFu), T(4)) == T((int32_t) 0xEADBEEFDu));
    assert(ror(T((int32_t) 0xDEADBEEFu), T(4)) == T((int32_t) 0xFDEADBEEu));
    assert(rol(T((int32_t) 0xDEADBEEFu), 4) == T((int32_t) 0xEADBEEFDu));
    assert(ror(T((int32_t) 0xDEADBEEFu), 4) == T((int32_t) 0xFDEADBEEu));
    assert(roli<4>(T((int32_t) 0xDEADBEEFu)) == T((int32_t) 0xEADBEEFDu));
    assert(rori<4>(T((int32_t) 0xDEADBEEFu)) == T((int32_t) 0xFDEADBEEu));
}

ENOKI_TEST_TYPE(test06_shiftrot, uint64_t) {
    assert((T(0xCAFEBABEDEADBEEFull) >> 4) == T(0x0CAFEBABEDEADBEEull));
    assert((T(0xCAFEBABEDEADBEEFull) << 4) == T(0xAFEBABEDEADBEEF0ull));
    assert((sri<4>(T(0xCAFEBABEDEADBEEFull)) == T(0x0CAFEBABEDEADBEEull)));
    assert((sli<4>(T(0xCAFEBABEDEADBEEFull)) == T(0xAFEBABEDEADBEEF0ull)));
    assert(rol(T(0xCAFEBABEDEADBEEFull), T(4)) == T(0xAFEBABEDEADBEEFCull));
    assert(ror(T(0xCAFEBABEDEADBEEFull), T(4)) == T(0xFCAFEBABEDEADBEEull));
    assert(rol(T(0xCAFEBABEDEADBEEFull), 4) == T(0xAFEBABEDEADBEEFCull));
    assert(ror(T(0xCAFEBABEDEADBEEFull), 4) == T(0xFCAFEBABEDEADBEEull));
    assert(roli<4>(T(0xCAFEBABEDEADBEEFull)) == T(0xAFEBABEDEADBEEFCull));
    assert(rori<4>(T(0xCAFEBABEDEADBEEFull)) == T(0xFCAFEBABEDEADBEEull));
}

ENOKI_TEST_TYPE(test06_shiftrot, int64_t) {
    assert((T((int64_t) 0xDEADBEEFCAFEBABEll) >> 4) == T((int64_t) 0xFDEADBEEFCAFEBABll));
    assert((T((int64_t) 0xDEADBEEFCAFEBABEll) << 4) == T((int64_t) 0xEADBEEFCAFEBABE0ll));
    assert((sri<4>(T((int64_t) 0xDEADBEEFCAFEBABEll)) == T((int64_t) 0xFDEADBEEFCAFEBABll)));
    assert((sli<4>(T((int64_t) 0xDEADBEEFCAFEBABEll)) == T((int64_t) 0xEADBEEFCAFEBABE0ll)));
    assert(rol(T((int64_t) 0xCAFEBABEDEADBEEFull), T(4)) == T((int64_t) 0xAFEBABEDEADBEEFCull));
    assert(ror(T((int64_t) 0xCAFEBABEDEADBEEFull), T(4)) == T((int64_t) 0xFCAFEBABEDEADBEEull));
    assert(rol(T((int64_t) 0xCAFEBABEDEADBEEFull), 4) == T((int64_t) 0xAFEBABEDEADBEEFCull));
    assert(ror(T((int64_t) 0xCAFEBABEDEADBEEFull), 4) == T((int64_t) 0xFCAFEBABEDEADBEEull));
    assert(roli<4>(T((int64_t) 0xCAFEBABEDEADBEEFll)) == T((int64_t) 0xAFEBABEDEADBEEFCull));
    assert(rori<4>(T((int64_t) 0xCAFEBABEDEADBEEFll)) == T((int64_t) 0xFCAFEBABEDEADBEEull));
}
