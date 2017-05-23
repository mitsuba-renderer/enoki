/*
    enoki/array_misc.h -- Miscellaneous useful vectorization routines

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
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
#endif

template <typename Type, size_t Size, bool Approx, RoundingMode Mode, typename Derived>
ENOKI_INLINE auto
transpose(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a) {
    static_assert(Type::Size == Size && array_depth<Derived>::value == 2,
                  "Array must be a square matrix!");
    Derived result;
    using Value = typename Type::Value;
    ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
        for (size_t j = 0; j < Size; ++j)
            result.coeff(i).coeff(j) = a.derived().coeff(j).coeff(i);
    return result;
}

/// Analagous to meshgrid() in NumPy or MATLAB; for dynamic arrays
template <typename Arr>
Array<Arr, 2> meshgrid(const Arr &x, const Arr &y) {
    Arr X, Y;
    dynamic_resize(X, x.size() * y.size());
    dynamic_resize(Y, x.size() * y.size());

    size_t pos = 0;

    if (x.size() % Arr::PacketSize == 0) {
        /* Fast path */

        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < packets(x); ++j) {
                packet(X, pos) = packet(x, j);
                packet(Y, pos) = typename Arr::Packet(y.coeff(i));
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

NAMESPACE_BEGIN(detail)

template <typename Array, size_t... Index, typename Value = value_t<Array>>
ENOKI_INLINE Array sample_shifted(Value sample, std::index_sequence<Index...>) {
    const Array shift((Value(Index) / Value(Array::Size))...);

    Array value = Array(sample) + shift;
    value[value > Value(1)] -= Value(1);

    return value;
}

NAMESPACE_END(detail)

/**
 * \brief Map a uniformly distributed sample to an array of samples with shifts
 *
 * Given a floating point value \c x on the interval <tt>[0, 1]</tt> return a
 * floating point array with values <tt>[x, x+offset, x+2*offset, ...]</tt>,
 * where \c offset is the reciprocal of the array size. Entries that become
 * greater than 1.0 wrap around to the other side of the unit inteval.
 *
 * This operation is useful to implement a type of correlated stratification in
 * the context of Monte Carlo integration.
 */
template <typename Array> Array sample_shifted(value_t<Array> sample) {
    return detail::sample_shifted<Array>(sample, std::make_index_sequence<Array::Size>());
}

/// Vectorized 'range' iteratable with automatic mask computation
template <typename T> struct range {
    using Scalar = scalar_t<T>;

    struct iterator {
        iterator(size_t index) : index(index) { }
        iterator(size_t index, T value, Scalar range_end)
            : index(index), value(value), range_end(range_end) { }

        bool operator==(const iterator &it) const { return it.index == index; }
        bool operator!=(const iterator &it) const { return it.index != index; }

        iterator &operator++() {
            index += 1;
            value += Scalar(T::Size);
            return *this;
        }

        std::pair<T, typename T::Mask> operator*() const {
            return { value, value < Scalar(range_end) };
        }

    private:
        size_t index;
        T value, value_end;
        Scalar range_end;
    };

    range(size_t range_end) : range_begin(0), range_end(range_end) { }
    range(size_t range_begin, size_t range_end)
        : range_begin(range_begin), range_end(range_end) { }

    iterator begin() {
        return iterator{ 0, index_sequence<T>() + Scalar(range_begin),
                         Scalar(range_end) };
    }

    iterator end() {
        return iterator{ (range_end - range_begin + T::Size - 1) / T::Size };
    }

private:
    size_t range_begin, range_end;
};

NAMESPACE_END(enoki)
