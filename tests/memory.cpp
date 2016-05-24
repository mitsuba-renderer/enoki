/*
    tests/loadstore.cpp -- tests for load/store/gather/scatter, etc.

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include "test.h"


ENOKI_TEST_ALL(test01_load) {
    alignas(alignof(T)) Scalar mem[Size];
    for (size_t i = 0; i < Size; ++i)
        mem[i] = (Scalar) i;
    assert(load<T>(mem) == index_sequence<T>());
}

ENOKI_TEST_ALL(test02_load_unaligned) {
    Scalar mem[Size];
    for (size_t i = 0; i < Size; ++i)
        mem[i] = (Scalar) i;
    assert(load_unaligned<T>(mem) == index_sequence<T>());
}

ENOKI_TEST_ALL(test03_store) {
    alignas(alignof(T)) Scalar mem[Size];
    store(mem, index_sequence<T>());
    for (size_t i = 0; i < Size; ++i)
        assert(mem[i] == (Scalar) i);
}

ENOKI_TEST_ALL(test04_store_unaligned) {
    Scalar mem[Size];
    store_unaligned(mem, index_sequence<T>());
    for (size_t i = 0; i < Size; ++i)
        assert(mem[i] == (Scalar) i);
}

ENOKI_TEST_ALL(test05_gather) {
    alignas(alignof(T)) Scalar mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Scalar) i;
    }

    auto id32 = load_unaligned<Array<uint32_t, Size>>(indices32);
    auto id64 = load_unaligned<Array<uint64_t, Size>>(indices64);

    store(dst, gather<T>(mem, id32));
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Scalar(Size - 1 - i));
    memset(dst, 0, sizeof(Scalar) * Size);

    store(dst, gather<T>(mem, id64));
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Scalar(Size - 1 - i));
    memset(dst, 0, sizeof(Scalar) * Size);
}

ENOKI_TEST_ALL(test06_gather_mask) {
    alignas(alignof(T)) Scalar mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Scalar) i;
    }

    auto id32 = load_unaligned<Array<uint32_t, Size>>(indices32);
    auto id64 = load_unaligned<Array<uint64_t, Size>>(indices64);
    auto idx = index_sequence<UIntArray<T>>();
    auto even_mask = reinterpret_array<typename T::Mask>(eq(sli<1>(sri<1>(idx)), idx));

    memset(dst, 0, sizeof(Scalar) * Size);
    store(dst, gather<T>(mem, id32, even_mask));
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == ((i % 2 == 0) ? Scalar(Size - 1 - i) : 0));
    memset(dst, 0, sizeof(Scalar) * Size);

    store(dst, gather<T>(mem, id64, even_mask));
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == ((i % 2 == 0) ? Scalar(Size - 1 - i) : 0));
}

ENOKI_TEST_ALL(test07_scatter) {
    alignas(alignof(T)) Scalar mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Scalar) i;
    }

    auto id32 = load_unaligned<Array<uint32_t, Size>>(indices32);
    auto id64 = load_unaligned<Array<uint64_t, Size>>(indices64);

    memset(dst, 0, sizeof(Scalar) * Size);
    scatter(dst, load<T>(mem), id32);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Scalar(Size - 1 - i));
    memset(dst, 0, sizeof(Scalar) * Size);

    scatter(dst, load<T>(mem), id64);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == Scalar(Size - 1 - i));
    memset(dst, 0, sizeof(Scalar) * Size);
}

ENOKI_TEST_ALL(test08_scatter_mask) {
    alignas(alignof(T)) Scalar mem[Size], dst[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Scalar) i;
    }
    auto id32 = load_unaligned<Array<uint32_t, Size>>(indices32);
    auto id64 = load_unaligned<Array<uint64_t, Size>>(indices64);

    auto idx = index_sequence<UIntArray<T>>();
    auto even_mask = reinterpret_array<typename T::Mask>(eq(sli<1>(sri<1>(idx)), idx));

    memset(dst, 0, sizeof(Scalar) * Size);
    scatter(dst, load<T>(mem), id32, even_mask);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == (((Size-1-i) % 2 == 0) ? Scalar(Size - 1 - i) : 0));
    memset(dst, 0, sizeof(Scalar) * Size);

    scatter(dst, load<T>(mem), id64, even_mask);
    for (size_t i = 0; i < Size; ++i)
        assert(dst[i] == (((Size-1-i) % 2 == 0) ? Scalar(Size - 1 - i) : 0));
}

ENOKI_TEST_ALL(test09_prefetch) {
    alignas(alignof(T)) Scalar mem[Size];
    uint32_t indices32[Size];
    uint64_t indices64[Size];
    for (size_t i = 0; i < Size; ++i) {
        indices32[i] = uint32_t(Size - 1 - i);
        indices64[i] = uint64_t(Size - 1 - i);
        mem[i] = (Scalar) i;
    }
    auto id32 = load_unaligned<Array<uint32_t, Size>>(indices32);
    auto id64 = load_unaligned<Array<uint64_t, Size>>(indices64);
    auto idx = index_sequence<UIntArray<T>>();
    auto even_mask = reinterpret_array<typename T::Mask>(eq(sli<1>(sri<1>(idx)), idx));

    /* Hard to test these, let's at least make sure that it compiles
       and does not crash .. */
    prefetch<T>(mem, id32);
    prefetch<T>(mem, id64);
    prefetch<T>(mem, id32, even_mask);
    prefetch<T>(mem, id64, even_mask);
}

ENOKI_TEST_ALL(test09_store_compress) {
    alignas(alignof(T)) Scalar tmp[T::ActualSize];
    auto value = index_sequence<T>();
    Scalar *tmp2 = tmp;
    store_compress((void *&) tmp2, value, value >= Scalar(2));
    for (int i = 0; i < int(Size) - 2; ++i)
        assert(tmp[i] == Scalar(2 + i));
    assert(int(tmp2 - tmp) == std::max(0, int(Size) - 2));
}
