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
template <typename Type, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Type::Size == 4 &&
                           std::is_same<typename Type::Type, float>::value, int> = 0>
ENOKI_INLINE auto
transpose(const StaticArrayBase<Type, 4, Approx, Mode, Derived> &a) {
    __m128 c0 = a.derived().coeff(0).m, c1 = a.derived().coeff(1).m,
           c2 = a.derived().coeff(2).m, c3 = a.derived().coeff(3).m;

    __m128 t0 = _mm_unpacklo_ps(c0, c1);
    __m128 t1 = _mm_unpacklo_ps(c2, c3);
    __m128 t2 = _mm_unpackhi_ps(c0, c1);
    __m128 t3 = _mm_unpackhi_ps(c2, c3);

    return Derived(
        _mm_movelh_ps(t0, t1),
        _mm_movehl_ps(t1, t0),
        _mm_movelh_ps(t2, t3),
        _mm_movehl_ps(t3, t2)
    );
}

template <typename Type1, bool Approx1, bool Approx2, RoundingMode Mode1,
          RoundingMode Mode2, typename Derived1, typename Derived2,
          std::enable_if_t<Type1::Size == 4 &&
                           std::is_same<typename Type1::Type, float>::value, int> = 0>
ENOKI_INLINE auto
mvprod(const StaticArrayBase<Type1, 4, Approx1, Mode1, Derived1> &a,
       const StaticArrayBase<float, 4, Approx2, Mode2, Derived2> &b) {
    __m128 c0 = a.derived().coeff(0).m, c1 = a.derived().coeff(1).m,
           c2 = a.derived().coeff(2).m, c3 = a.derived().coeff(3).m,
           v = b.derived().m;

    __m128 t1 = _mm_mul_ps(c0, v), t2 = _mm_mul_ps(c1, v),
           t3 = _mm_mul_ps(c2, v), t4 = _mm_mul_ps(c3, v);

    return Derived2(_mm_hadd_ps(_mm_hadd_ps(t1, t2),
                                _mm_hadd_ps(t3, t4)));
}
#else
inline void set_flush_denormals(bool) { }
#endif

#if defined(__AVX__)
template <typename Type, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Type::Size == 4 &&
                           std::is_same<typename Type::Type, double>::value, int> = 0>
ENOKI_INLINE auto
transpose(const StaticArrayBase<Type, 4, Approx, Mode, Derived> &a) {
    __m256d c0 = a.derived().coeff(0).m, c1 = a.derived().coeff(1).m,
            c2 = a.derived().coeff(2).m, c3 = a.derived().coeff(3).m;

    __m256d t3 = _mm256_shuffle_pd(c2, c3, 0b0000),
            t2 = _mm256_shuffle_pd(c2, c3, 0b1111),
            t1 = _mm256_shuffle_pd(c0, c1, 0b0000),
            t0 = _mm256_shuffle_pd(c0, c1, 0b1111);

    return Derived(
        _mm256_permute2f128_pd(t1, t3, 0b0010'0000),
        _mm256_permute2f128_pd(t0, t2, 0b0010'0000),
        _mm256_permute2f128_pd(t1, t3, 0b0011'0001),
        _mm256_permute2f128_pd(t0, t2, 0b0011'0001)
    );
}

template <typename Type1, bool Approx1, bool Approx2, RoundingMode Mode1,
          RoundingMode Mode2, typename Derived1, typename Derived2,
          std::enable_if_t<Type1::Size == 4 &&
                           std::is_same<typename Type1::Type, double>::value, int> = 0>
ENOKI_INLINE auto
mvprod(const StaticArrayBase<Type1, 4, Approx1, Mode1, Derived1> &a,
       const StaticArrayBase<double, 4, Approx2, Mode2, Derived2> &b) {
    __m256d c0 = a.derived().coeff(0).m, c1 = a.derived().coeff(1).m,
            c2 = a.derived().coeff(2).m, c3 = a.derived().coeff(3).m,
            v = b.derived().m;

    __m256d t0 = _mm256_mul_pd(c0, v);
    __m256d t1 = _mm256_mul_pd(c1, v);
    __m256d t2 = _mm256_mul_pd(c2, v);
    __m256d t3 = _mm256_mul_pd(c3, v);

    __m256d s0 = _mm256_hadd_pd(t0, t1);
    __m256d s1 = _mm256_hadd_pd(t2, t3);

    return Derived2(
        _mm256_add_pd(_mm256_permute2f128_pd(s0, s1, 0b0010'0001),
                      _mm256_blend_pd(s0, s1, 0b1100)));
}
#endif

template <typename Type, size_t Size, bool Approx, RoundingMode Mode, typename Derived>
ENOKI_INLINE auto
transpose(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a) {
    static_assert(Type::Size == Size && array_depth<Derived>::value == 2,
                  "Array must be a square matrix!");
    Derived result;
    using Scalar = typename Type::Scalar;
    ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
        for (size_t j = 0; j < Size; ++j)
            result.coeff(i).coeff(j) = a.derived().coeff(j).coeff(i);
    return result;
}

template <typename Type1, typename Type2, size_t Size, bool Approx1,
          bool Approx2, RoundingMode Mode1, RoundingMode Mode2,
          typename Derived1, typename Derived2>
ENOKI_INLINE auto
mvprod(const StaticArrayBase<Type1, Size, Approx1, Mode1, Derived1> &a,
       const StaticArrayBase<Type2, Size, Approx2, Mode2, Derived2> &b) {
    static_assert(Type1::Size == Size && array_depth<Derived1>::value == 2,
                  "First argument must be a square matrix!");
    static_assert(array_depth<Derived2>::value == 1,
                  "First argument must be a vector!");
    Derived2 result;
    for (size_t i = 0; i< Size; ++i)
        result.coeff(i) = dot(a.derived().coeff(i), b.derived());
    return result;
}


/// Analagous to meshgrid() in NumPy or MATLAB; for dynamic arrays
template <typename Arr>
Array<Arr, 2> meshgrid(const Arr &x, const Arr &y) {
    Arr X(x.size() * y.size()), Y(x.size() * y.size());

    size_t pos = 0;

    if (x.size() % Arr::PacketSize == 0) {
        /* Fast path */

        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < x.packets(); ++j) {
                X.packet(pos) = x.packet(j);
                Y.packet(pos) = typename Arr::Packet(y.coeff(i));
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

    return Array<Arr, 2>(std::move(X), std::move(Y));
}

NAMESPACE_END(enoki)
