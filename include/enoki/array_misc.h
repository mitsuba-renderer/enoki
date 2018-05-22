/*
    enoki/array_misc.h -- Miscellaneous useful vectorization routines

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array_generic.h>

NAMESPACE_BEGIN(enoki)

#if defined(ENOKI_X86_SSE42)
/// Flush denormalized numbers to zero
inline void set_flush_denormals(bool value) {
    _MM_SET_FLUSH_ZERO_MODE(value ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(value ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF);
}

inline bool flush_denormals() {
    return _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
}

#else
inline void set_flush_denormals(bool) { }
#endif

template <typename Value, size_t Size1, size_t Size2, bool Approx1, bool Approx2,
          RoundingMode Mode1, RoundingMode Mode2, typename Derived1,
          typename Derived2, typename Result = Array<Value, Derived1::Size + Derived2::Size>,
          std::enable_if_t<(Size1 + Size2) / 2 != Size1 || (Size1 + Size2) / 2 != Size2, int> = 0>
Result
concat(const StaticArrayBase<Value, Size1, Approx1, Mode1, Derived1> &a1,
       const StaticArrayBase<Value, Size2, Approx2, Mode2, Derived2> &a2) {
    Result result;
    for (size_t i = 0; i < Derived1::Size; ++i)
        result.coeff(i) = a1.derived().coeff(i);
    for (size_t i = 0; i < Derived2::Size; ++i)
        result.coeff(i + Derived1::Size) = a2.derived().coeff(i);
    return result;
}

template <typename Value, size_t Size1, size_t Size2, bool Approx1, bool Approx2,
          RoundingMode Mode1, RoundingMode Mode2, typename Derived1,
          typename Derived2, typename Result = Array<Value, Derived1::Size + Derived2::Size>,
          std::enable_if_t<(Size1 + Size2) / 2 == Size1 && (Size1 + Size2) / 2 == Size2, int> = 0>
Result
concat(const StaticArrayBase<Value, Size1, Approx1, Mode1, Derived1> &a1,
       const StaticArrayBase<Value, Size2, Approx2, Mode2, Derived2> &a2) {
    return Result(a1, a2);
}

template <typename Value, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename Result = Array<Value, Derived::Size + 1, Approx, Mode>>
Result concat(const StaticArrayBase<Value, Size, Approx, Mode, Derived> &a, expr_t<value_t<Value>> value) {
    Result result;
    for (size_t i = 0; i < Derived::Size; ++i)
        result.coeff(i) = a.derived().coeff(i);
    result.coeff(Derived::Size) = value;
    return result;
}

template <typename Value, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename Result = Array<Value, Derived::Size + 1, Approx, Mode>,
          std::enable_if_t<!std::is_same<scalar_t<Value>, value_t<Value>>::value, int> = 0>
Result concat(const StaticArrayBase<Value, Size, Approx, Mode, Derived> &a,
              scalar_t<Value> value) {
    Result result;
    for (size_t i = 0; i < Derived::Size; ++i)
        result.coeff(i) = a.derived().coeff(i);
    result.coeff(Derived::Size) = value;
    return result;
}

template <typename Value, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Derived::Size == 3, int> = 0>
Array<Value, 4> concat(const StaticArrayBase<Value, 4, Approx, Mode, Derived> &a, value_t<Value> value) {
    Array<Value, 4, Approx, Mode> result = a.derived();
    result.w() = value;
    return result;
}

#if defined(ENOKI_X86_SSE42)
template <bool Approx, RoundingMode Mode, typename Derived, std::enable_if_t<Derived::Size == 3, int> = 0>
Array<float, 4, Approx, Mode> concat(const StaticArrayBase<float, 4, Approx, Mode, Derived> &a, float value) {
    return _mm_insert_ps(a.derived().m, _mm_set_ss(value), 0b00110000);
}
#endif

/// Analagous to meshgrid() in NumPy or MATLAB; for dynamic arrays
template <typename Arr>
Array<Arr, 2> meshgrid(const Arr &x, const Arr &y) {
    Arr X, Y;
    set_slices(X, x.size() * y.size());
    set_slices(Y, x.size() * y.size());

    size_t pos = 0;

    if (x.size() % Arr::PacketSize == 0) {
        /* Fast path */

        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < packets(x); ++j) {
                packet(X, pos) = packet(x, j);
                packet(Y, pos) = packet_t<Arr>(y.coeff(i));
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
    const Array shift(Index / scalar_t<Array>(Array::Size)...);

    Array value = Array(sample) + shift;
    value[value > Value(1)] -= Value(1);

    return value;
}

template <typename Return, size_t Offset, typename T, size_t... Index>
static ENOKI_INLINE Return extract(const T &value, std::index_sequence<Index...>) {
    return Return(value.coeff(Index + Offset)...);
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
    return detail::sample_shifted<Array>(
        sample, std::make_index_sequence<Array::Size>());
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

        std::pair<T, mask_t<T>> operator*() const {
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


template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>,
          std::enable_if_t<T::ActualSize != Return::ActualSize && T::ActualSize / 2 == Return::ActualSize, int> = 0>
static ENOKI_INLINE Return head(const T &value) { return low(value); }

template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>,
          std::enable_if_t<T::ActualSize / 2 == Return::ActualSize, int> = 0>
static ENOKI_INLINE Return tail(const T &value) { return high(value); }

template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>,
          std::enable_if_t<T::ActualSize == Return::ActualSize, int> = 0>
static ENOKI_INLINE Return head(const T &value) { return value; }

template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>,
          std::enable_if_t<T::ActualSize != Return::ActualSize && T::ActualSize / 2 != Return::ActualSize, int> = 0>
static ENOKI_INLINE Return head(const T &value) {
    static_assert(Size <= array_size<T>::value, "Array size mismatch");
    return detail::extract<Return, 0>(value, std::make_index_sequence<Size>());
}

template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>,
          std::enable_if_t<T::ActualSize / 2 != Return::ActualSize, int> = 0>
static ENOKI_INLINE Return tail(const T &value) {
    static_assert(Size <= array_size<T>::value, "Array size mismatch");
    return detail::extract<Return, T::Size - Size>(value, std::make_index_sequence<Size>());
}

/**
 * \brief Numerically well-behaved routine for computing the angle
 * between two unit direction vectors
 *
 * This should be used wherever one is tempted to compute the
 * arc cosine of a dot product.
 *
 * Proposed by Don Hatch at http://www.plunk.org/~hatch/rightway.php
 */
template <typename Array, typename Expr = expr_t<value_t<Array>>> Expr unit_angle(const Array &a, const Array &b) {
    Expr dot_uv = dot(a, b),
         temp = 2.f * asin(.5f * norm(b - mulsign(a, dot_uv)));
    return select(dot_uv >= 0, temp, scalar_t<Expr>(M_PI) - temp);
}

NAMESPACE_END(enoki)
