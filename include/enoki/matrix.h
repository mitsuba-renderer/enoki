/*
    enoki/matrix.h -- Convenience wrapper for square matrixes

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "array.h"

NAMESPACE_BEGIN(enoki)

/**
 * \brief Basic dense square matrix data structure
 * \remark Uses column-major storage order to permit efficient vectorization
 */
template <typename Type_, size_t Size_>
struct Matrix
    : StaticArrayImpl<Array<Type_, Size_>, Size_,
                      detail::approx_default<Type_>::value,
                      RoundingMode::Default, Matrix<Type_, Size_>> {

    using Type = Type_;
    using Column = Array<Type_, Size_>;
    using Entry = value_t<Column>;

    using Base = StaticArrayImpl<Column, Size_,
                                 detail::approx_default<Type_>::value,
                                 RoundingMode::Default, Matrix<Type_, Size_>>;

    template <typename T> using ReplaceType = Matrix<T, Size_>;

    using Base::Base;
    using Base::operator=;
    using Base::coeff;
    using Base::Size;

    /// Initialize the matrix from a set of coefficients
    template <typename... Args, std::enable_if_t<
              detail::all_of<std::is_constructible<Entry, Args>::value...,
                             sizeof...(Args) == Size * Size>::value, int> = 0>
    ENOKI_INLINE Matrix(const Args&... args) {
        alignas(alignof(Column)) Entry values[sizeof...(Args)] = { Entry(args)... };
        for (size_t j = 0; j < Size; ++j)
            for (size_t i = 0; i < Size; ++i)
                coeff(j).coeff(i) = values[i * Size + j];
    }

    /// Return a reference to the (i, j) element
    ENOKI_INLINE Entry& operator()(size_t i, size_t j) { return coeff(j, i); }

    /// Return a reference to the (i, j) element (const)
    ENOKI_INLINE const Entry& operator()(size_t i, size_t j) const { return coeff(j, i); }
};

template <typename Type1, typename Type2, size_t Size,
          typename Scalar = decltype(std::declval<Type1>() + std::declval<Type2>()),
          typename Return = Matrix<expr_t<Scalar>, Size>,
          typename Column = typename Return::Column>
ENOKI_INLINE Return operator*(const Matrix<Type1, Size> &m1,
                              const Matrix<Type2, Size> &m2) {
    Return result;
    /* 4x4 case reduced to 4 multiplications, 12 fused multiply-adds,
       and 16 broadcasts (also fused on AVX512VL) */
    for (size_t j = 0; j < Size; ++j) {
        Column sum = m1.coeff(0) * m2(0, j);
        for (size_t i = 1; i < Size; ++i)
            sum = fmadd(m1.coeff(i), Column(m2(i, j)), sum);
        result.coeff(j) = sum;
    }

    return result;
}

template <typename Type1, typename Type2, size_t Size, size_t Size2,
          bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Size == Derived::Size, int> = 0,
          typename Type = decltype(std::declval<Type1>() + std::declval<Type2>()),
          typename Return = Array<Type, Size>>
ENOKI_INLINE Return
operator*(const Matrix<Type1, Size> &m1,
          const StaticArrayBase<Type2, Size2, Approx, Mode, Derived> &m2) {
    Return sum = m1.coeff(0) * m2.derived().coeff(0);
    for (size_t i = 1; i < Size; ++i)
        sum = fmadd(m1.coeff(i), Return(m2.derived().coeff(i)), sum);
    return sum;
}

template <typename Type, size_t Size>
ENOKI_INLINE expr_t<Type> trace(const Matrix<Type, Size> &m) {
    expr_t<Type> result = m.coeff(0, 0);
    for (size_t i = 1; i < Size; ++i)
        result += m(i, i);
    return result;
}

template <typename Matrix> ENOKI_INLINE Matrix identity() {
    Matrix result = zero<Matrix>();
    for (size_t i = 0; i < Matrix::Size; ++i)
        result(i, i) = 1;
    return result;
}

// =======================================================================
//! @{ \name Enoki accessors for static & dynamic vectorization
// =======================================================================

template <typename T, size_t Size> struct dynamic_support<Matrix<T, Size>, enable_if_static_array_t<Matrix<T, Size>>> {
    static constexpr bool is_dynamic_nested = enoki::is_dynamic_nested<T>::value;
    using dynamic_t = Matrix<enoki::make_dynamic_t<T>, Size>;
    using Value = Matrix<T, Size>;

    static ENOKI_INLINE size_t dynamic_size(const Value &value) {
        return enoki::dynamic_size(value.coeff(0, 0));
    }

    static ENOKI_INLINE size_t packets(const Value &value) {
        return enoki::packets(value.coeff(0, 0));
    }

    static ENOKI_INLINE void dynamic_resize(Value &value, size_t size) {
        for (size_t i = 0; i < Size; ++i)
            enoki::dynamic_resize(value.coeff(i), size);
    }

    template <typename T2>
    static ENOKI_INLINE auto packet(T2&& value, size_t i) {
        return packet(value, i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto slice(T2&& value, size_t i) {
        return slice(value, i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto ref_wrap(T2&& value) {
        return ref_wrap(value, std::make_index_sequence<Size>());
    }

private:
    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto packet(T2&& value, size_t i, std::index_sequence<Index...>) {
        return Matrix<decltype(enoki::packet(value.coeff(0, 0), i)), Size>(
            enoki::packet(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice(T2&& value, size_t i, std::index_sequence<Index...>) {
        return Matrix<decltype(enoki::slice(value.coeff(0, 0), i)), Size>(
            enoki::slice(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto ref_wrap(T2&& value, std::index_sequence<Index...>) {
        return Matrix<decltype(enoki::ref_wrap(value.coeff(0, 0))), Size>(
            enoki::ref_wrap(value.coeff(Index))...);
    }
};

//! @}
// =======================================================================

NAMESPACE_END(enoki)
