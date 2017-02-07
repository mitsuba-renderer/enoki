/*
    enoki/array_misc.h -- Miscellaneous useful vectorization routines

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include <enoki/array_generic.h>

NAMESPACE_BEGIN(enoki)

#if defined(__SSE4_2__)
/// Flush denormalized numbers to zero
inline void set_flush_denormals(bool value) {
    _MM_SET_FLUSH_ZERO_MODE(value ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(value ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF);
}

// Optimized 4x4 dot product
template <typename T, std::enable_if_t<std::is_same<typename T::Type, float>::value, int> = 0>
ENOKI_INLINE T dot4x4(T r0, T r1, T r2, T r3, T v) {
    __m128 d0 = _mm_dp_ps(r0.m, v.m, 0b11110001);
    __m128 d1 = _mm_dp_ps(r1.m, v.m, 0b11110010);
    __m128 d2 = _mm_dp_ps(r2.m, v.m, 0b11110100);
    __m128 d3 = _mm_dp_ps(r3.m, v.m, 0b11111000);
    return T(_mm_or_ps(_mm_or_ps(_mm_or_ps(d0, d1), d2), d3));
}

// Optimized 4x4 matrix transpose
template <typename T, std::enable_if_t<std::is_same<typename T::Type, float>::value, int> = 0>
ENOKI_INLINE void transpose4x4(T &r0, T &r1, T &r2, T &r3) {
    __m128 t0 = _mm_unpacklo_ps(r0.m, r1.m);
    __m128 t1 = _mm_unpacklo_ps(r2.m, r3.m);
    __m128 t2 = _mm_unpackhi_ps(r0.m, r1.m);
    __m128 t3 = _mm_unpackhi_ps(r2.m, r3.m);
    r0.m = _mm_movelh_ps(t0, t1);
    r1.m = _mm_movehl_ps(t1, t0);
    r2.m = _mm_movelh_ps(t2, t3);
    r3.m = _mm_movehl_ps(t3, t2);
}
template <typename T, std::enable_if_t<!std::is_same<typename T::Type, float>::value, int> = 0>
ENOKI_INLINE T dot4x4(T r0, T r1, T r2, T r3, T v) {
    return T(dot(v, r0), dot(v, r1), dot(v, r2), dot(v, r3));
}
template <typename T, std::enable_if_t<!std::is_same<typename T::Type, float>::value, int> = 0>
ENOKI_INLINE void transpose4x4(T &r0, T &r1, T &r2, T &r3) {
    std::swap(r0.coeff(1), r1.coeff(0));
    std::swap(r0.coeff(2), r2.coeff(0));
    std::swap(r0.coeff(3), r3.coeff(0));
    std::swap(r1.coeff(2), r2.coeff(1));
    std::swap(r1.coeff(3), r3.coeff(1));
    std::swap(r2.coeff(3), r3.coeff(2));
}
#else
template <typename T>
ENOKI_INLINE T dot4x4(T r0, T r1, T r2, T r3, T v) {
    return T(dot(v, r0), dot(v, r1), dot(v, r2), dot(v, r3));
}
template <typename T>
ENOKI_INLINE void transpose4x4(T &r0, T &r1, T &r2, T &r3) {
    std::swap(r0.coeff(1), r1.coeff(0));
    std::swap(r0.coeff(2), r2.coeff(0));
    std::swap(r0.coeff(3), r3.coeff(0));
    std::swap(r1.coeff(2), r2.coeff(1));
    std::swap(r1.coeff(3), r3.coeff(1));
    std::swap(r2.coeff(3), r3.coeff(2));
}
inline void set_flush_denormals(bool) { }
#endif

/// Analagous to meshgrid() in NumPy or MATLAB; for dynamic arrays
template <typename Array>
std::pair<Array, Array> meshgrid(const Array &x, const Array &y) {
    Array X(x.size() * y.size()), Y(x.size() * y.size());

    size_t pos = 0;

    if (x.size() % Array::PacketSize == 0) {
        /* Fast path */

        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < x.packets(); ++j) {
                X.packet(pos) = x.packet(j);
                Y.packet(pos) = typename Array::Packet(y.coeff(i));
                pos++;
            }
        }
    } else {
        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < x.size(); ++j) {
                X.coeff(pos) = x.coeff(j);
                Y.coeff(pos) = y.coeff(i);
                pos++;
            }
        }
    }

    return std::make_pair<Array, Array>(std::move(X), std::move(Y));
}


NAMESPACE_END(enoki)
