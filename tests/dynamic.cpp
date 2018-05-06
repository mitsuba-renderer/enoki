/*
    tests/basic.cpp -- tests dynamic heap-allocated arrays

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"

ENOKI_TEST(test01_alloc)  {
    using T = Array<float, 4>;
    using D = DynamicArray<T>;

    D x;
    set_slices(x, 10);

    assert(x.size() == 10);
    assert(x.capacity() == 12);
    assert(packets(x) == 3);
    assert(!x.is_mapped());
    x.coeff(1) = 1.f;

    auto y = std::move(x);
    assert(x.size() == 0);
    assert(x.capacity() == 0);
    assert(packets(x) == 0);
    assert(!x.is_mapped());

    assert(y.size() == 10);
    assert(y.capacity() == 12);
    assert(packets(y) == 3);
    assert(!y.is_mapped());

    assert(!all(enoki::isnan(packet(y, 0))));
    assert( any(enoki::isnan(packet(y, 0))));
    assert( all(enoki::isnan(packet(y, 1))));
    assert(!all(enoki::isnan(packet(y, 2))));
    assert( any(enoki::isnan(packet(y, 2))));

    y.coeff(2) = 2.f;
    assert(to_string(y) == "[-nan, 1, 2, -nan, -nan, -nan, -nan, -nan, -nan, -nan]" ||
           to_string(y) == "[nan, 1, 2, nan, nan, nan, nan, nan, nan, nan]");
    resize(y, {{ 3 }});
    assert(to_string(y) == "[-nan, 1, 2]" || to_string(y) == "[nan, 1, 2]");
}

ENOKI_TEST(test02_map)  {
    alignas(16) float f[8];
    for (int i = 0; i < 8; ++i)
        f[i] = float(i);
    using T = Array<float, 4>;
    using D = DynamicArray<T>;

    auto x = D(f, 6);

    assert(x.size() == 6);
    assert(x.capacity() == 0);
    assert(packets(x) == 2);
    assert(x.is_mapped());
    assert(to_string(x) == "[0, 1, 2, 3, 4, 5]");
}

ENOKI_TEST(test03_alloc_nested)  {
    using Float     = float;
    using FloatP    = Array<Float, 4>;
    using FloatX    = DynamicArray<FloatP>;
    using Vector3fP = Array<FloatP, 3>;
    using Vector3fX = Array<FloatX, 3>;
    using Vector3f  = Array<float, 3>;

    Vector3fX x;
    resize(x, {{ 3, 2 }});
    packet(x[0], 0) = FloatP(1.f, 2.f, 3.f, 4.f);
    packet(x[1], 0) = FloatP(5.f, 6.f, 7.f, 8.f);
    packet(x[2], 0) = FloatP(9.f,10.f,11.f,12.f);

    Vector3fX y = std::move(x);
    Vector3fX z = y;
    packet(z[2], 0) = FloatP(0.f,11.f,12.f,13.f);

    assert(to_string(y) == "[[1, 5, 9],\n [2, 6, 10]]");
    assert(to_string(z) == "[[1, 5, 0],\n [2, 6, 11]]");

    assert(!is_dynamic_nested<Float>::value && !is_dynamic_nested<FloatP>::value &&
           is_dynamic_nested<FloatX>::value && !is_dynamic_nested<Vector3fP>::value &&
           is_dynamic_nested<Vector3fX>::value);

    assert(packets(123) == 0);
    assert(packet(123, 0) == 123);
    assert(packets(z) == 1);
    assert(packets(z[0]) == 1);
    assert(packets(z[0][0]) == 0);
    assert(to_string(packet(z, 0)) == "[[1, 5, 0],\n [2, 6, 11],\n [3, 7, 12],\n [4, 8, 13]]");
    assert((std::is_reference<decltype(packet(z, 0))::Value>::value));

    vectorize([](auto &&z) { z = z + Vector3f(1.f, 2.f, 3.f); }, z);
    assert(to_string(z) == "[[2, 7, 3],\n [3, 8, 14]]");

    vectorize([](auto &&z) { z = z + Vector3fP(1.f, 2.f, 3.f); }, z);
    assert(to_string(z) == "[[3, 9, 6],\n [4, 10, 17]]");
}

ENOKI_TEST(test04_init)  {
    using FloatP = Array<float, 4>;
    using FloatX = DynamicArray<FloatP>;

    auto v0 = zero<FloatX>(11);
    auto v1 = index_sequence<FloatX>(11);
    auto v2 = linspace<FloatX>(11, 0.f, 1.f);
    int ctr = 0;
    assert(v0.size() == 11);
    assert(v1.size() == 11);
    assert(v2.size() == 11);

    for (float v : v0) { assert(v == 0.f); }
    for (float v : v1) { assert(v == (float) ctr); ctr++; }
    ctr = 0;
    for (float v : v2) {
        assert(std::abs(v - (float) ctr / 10.f) < 1e-6);
        ctr++;
    }
}

ENOKI_TEST(test05_meshgrid) {
    using FloatP = Array<float, 4>;
    using FloatX = DynamicArray<FloatP>;

    auto ac = test::alloc_count;
    auto dc = test::dealloc_count;

    /* Nested scope */ {
        auto x = linspace<FloatX>(3, 0.f, 2.f);
        auto y = linspace<FloatX>(4, 1.f, 4.f);
        auto xy = meshgrid(x, y);

        assert(slices(xy) == 12);
        assert(to_string(xy) == "[[0, 1],\n [1, 1],\n [2, 1],\n [0, 2],\n [1, 2],\n [2, 2],\n [0, 3],\n [1, 3],\n [2, 3],\n [0, 4],\n [1, 4],\n [2, 4]]");

        Array<FloatX, 2> yz = std::move(xy);
        assert(to_string(yz) == "[[0, 1],\n [1, 1],\n [2, 1],\n [0, 2],\n [1, 2],\n [2, 2],\n [0, 3],\n [1, 3],\n [2, 3],\n [0, 4],\n [1, 4],\n [2, 4]]");
        assert(to_string(xy) == "[]");
    }

    assert(test::alloc_count - ac == 4);
    assert(test::dealloc_count - dc == 4);
}

template <typename Value> struct GPSCoord2 {
    using Vector2 = Array<Value, 2>;
    using UInt64  = uint64_array_t<Value>;
    using Bool    = bool_array_t<Value>;

    UInt64 time;
    Vector2 pos;
    Bool reliable;

    ENOKI_STRUCT(GPSCoord2, time, pos, reliable)
};

ENOKI_STRUCT_SUPPORT(GPSCoord2, time, pos, reliable)


/// Calculate the distance in kilometers between 'r1' and 'r2' using the haversine formula
template <typename Value_, typename Value = expr_t<Value_>>
ENOKI_INLINE Value distance(const GPSCoord2<Value_> &r1, const GPSCoord2<Value_> &r2) {
    using Scalar = scalar_t<Value>;
    using Mask = mask_t<Value>;

    const Value deg_to_rad = Scalar(M_PI / 180.0);

    auto sin_diff_h = sin(deg_to_rad * Scalar(.5) * (r2.pos - r1.pos));
    sin_diff_h *= sin_diff_h;

    Value a = sin_diff_h.x() + sin_diff_h.y() *
              cos(r1.pos.x() * deg_to_rad) *
              cos(r2.pos.x() * deg_to_rad);

    return select(
        Mask(r1.reliable) & Mask(r2.reliable),
        Scalar(6371.0 * 2.0) * atan2(sqrt(a), sqrt(Scalar(1.0) - a)),
        Value(std::numeric_limits<Scalar>::quiet_NaN())
    );
}

template <size_t PacketSize> void test06_haversine() {
    using FloatP       = Array<float, PacketSize>;
    using FloatX       = DynamicArray<FloatP>;
    using GPSCoord2fX  = GPSCoord2<FloatX>;
    using GPSCoord2f   = GPSCoord2<float>;
    using Vector2f     = Array<float, 2>;

    GPSCoord2fX coord1;
    GPSCoord2fX coord2;
    FloatX result;

    size_t size = 100;
    set_slices(coord1, size);
    set_slices(coord2, size);
    set_slices(result, size);

    slice(coord1, 0) = GPSCoord2f(
        0ull, Vector2f(51.5f, 0.0f), true);

    slice(coord2, 0) = GPSCoord2f(
        0ull, Vector2f(38.8f, -77.1f), true
    );

    vectorize([](auto &&result, auto &&coord1, auto &&coord2) {
                  result = distance<FloatP>(coord1, coord2);
              }, result, coord1, coord2);

    assert(std::abs(slice(result, 0) - 5918.18f) < 1e-2f);
}

ENOKI_TEST(array_float_04_test06_haversine) { test06_haversine<4>();  }
ENOKI_TEST(array_float_08_test06_haversine) { test06_haversine<8>();  }
ENOKI_TEST(array_float_16_test06_haversine) { test06_haversine<16>(); }
ENOKI_TEST(array_float_32_test06_haversine) { test06_haversine<32>(); }

template <size_t PacketSize> void test07_compress() {
    using FloatP       = Array<float, PacketSize>;
    using FloatX       = DynamicArray<FloatP>;
    using GPSCoord2fX  = GPSCoord2<FloatX>;
    using GPSCoord2f   = GPSCoord2<float>;
    using Vector2f     = Array<float, 2>;

    GPSCoord2fX coord1, coord2;

    set_slices(coord1, 2*PacketSize);
    set_slices(coord2, 3*PacketSize);

    for (size_t i = 0; i < 2 * PacketSize; ++i)
        slice(coord1, i) =
            GPSCoord2f(uint64_t(i), Vector2f((float) i, (float) (i * 100)), ((i) % 3) == 0);

    auto ptr = slice_ptr(coord2, 0);
    for (size_t i = 0; i < 2; ++i) {
        auto idx = index_sequence<uint_array_t<FloatP>>();
        auto even_mask = mask_t<FloatP>(eq(sli<1>(sri<1>(idx)), idx));
        compress(ptr, packet(coord1, i), even_mask);
    }

    for (size_t i = 0; i < 2; ++i) {
        auto idx = index_sequence<uint_array_t<FloatP>>();
        auto odd_mask = mask_t<FloatP>(neq(sli<1>(sri<1>(idx)), idx));
        compress(ptr, packet(coord1, i), odd_mask);
    }
    set_slices(coord2, 2*PacketSize);

    for (size_t i = 0; i < PacketSize; ++i) {
        size_t even = 2*i;
        size_t odd = 2*i+1;
        slice(coord2, i) =
            GPSCoord2f(uint64_t(even), Vector2f((float) even, (float) (even * 100)),
                       ((even) % 3) == 0);
        slice(coord2, i + PacketSize) =
            GPSCoord2f(uint64_t(odd), Vector2f((float) odd, (float) (odd * 100)),
                       ((odd) % 3) == 0);
    }
}

ENOKI_TEST(array_float_04_test07_compress) { test07_compress<4>();  }
ENOKI_TEST(array_float_08_test07_compress) { test07_compress<8>();  }
ENOKI_TEST(array_float_16_test07_compress) { test07_compress<16>(); }
ENOKI_TEST(array_float_32_test07_compress) { test07_compress<32>(); }

ENOKI_TEST(test07_realloc) {
    using FloatP = Array<uint32_t>;
    using FloatX = DynamicArray<FloatP>;

    FloatX array;
    for (size_t i = 0; i < 1024; ++i) {
        set_slices(array, (i + 1) * 1024);
        for (size_t j = 0; j < 1024; ++j)
            slice(array, i * 1024 + j) = (uint32_t) (i * 1024 + j);
    }
    for (size_t j = 0; j < 1024 * 1024; ++j)
        assert(slice(array, j) == j);

    set_slices(array, 1024 * 512);
    for (size_t j = 0; j < 1024 * 512; ++j)
        assert(slice(array, j) == j);
}
