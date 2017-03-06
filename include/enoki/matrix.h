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

/* Is this type dynamic? */
template <typename T, size_t Size> struct is_dynamic_impl<Matrix<T, Size>> {
    static constexpr bool value = is_dynamic<T>::value;
};

/* Create a dynamic version of this type on demand */
template <typename T, size_t Size> struct dynamic_impl<Matrix<T, Size>> {
    using type = Matrix<dynamic_t<T>, Size>;
};

/* How many packets are stored in this instance? */
template <typename T, size_t Size>
size_t packets(const Matrix<T, Size> &v) {
    return packets(v.coeff(0));
}

/* What is the size of the dynamic dimension of this instance? */
template <typename T, size_t Size>
size_t dynamic_size(const Matrix<T, Size> &v) {
    return dynamic_size(v.coeff(0));
}

/* Resize the dynamic dimension of this instance */
template <typename T, size_t Size>
void dynamic_resize(Matrix<T, Size> &v, size_t size) {
    for (size_t i = 0; i < Size; ++i)
        dynamic_resize(v.coeff(i), size);
}

template <typename T, size_t Size, size_t... Index>
auto ref_wrap(Matrix<T, Size> &v, std::index_sequence<Index...>) {
    using T2 = decltype(ref_wrap(v.coeff(0, 0)));
    return Matrix<T2, Size>{ ref_wrap(v.coeff(Index))... };
}

template <typename T, size_t Size, size_t... Index>
auto ref_wrap(const Matrix<T, Size> &v, std::index_sequence<Index...>) {
    using T2 = decltype(ref_wrap(v.coeff(0, 0)));
    return Matrix<T2, Size>{ ref_wrap(v.coeff(Index))... };
}

/* Construct a wrapper that references the data of this instance */
template <typename T, size_t Size> auto ref_wrap(Matrix<T, Size> &v) {
    return ref_wrap(v, std::make_index_sequence<Size>());
}

/* Construct a wrapper that references the data of this instance (const) */
template <typename T, size_t Size> auto ref_wrap(const Matrix<T, Size> &v) {
    return ref_wrap(v, std::make_index_sequence<Size>());
}

template <typename T, size_t Size, size_t... Index>
auto packet(Matrix<T, Size> &v, size_t i, std::index_sequence<Index...>) {
    using T2 = decltype(packet(v.coeff(0, 0), i));
    return Matrix<T2, Size>{ packet(v.coeff(Index), i)... };
}

template <typename T, size_t Size, size_t... Index>
auto packet(const Matrix<T, Size> &v, size_t i,
            std::index_sequence<Index...>) {
    using T2 = decltype(packet(v.coeff(0, 0), i));
    return Matrix<T2, Size>{ packet(v.coeff(Index), i)... };
}

/* Return the i-th packet */
template <typename T, size_t Size>
auto packet(Matrix<T, Size> &v, size_t i) {
    return packet(v, i, std::make_index_sequence<Size>());
}

/* Return the i-th packet (const) */
template <typename T, size_t Size>
auto packet(const Matrix<T, Size> &v, size_t i) {
    return packet(v, i, std::make_index_sequence<Size>());
}

/* Return the i-th slice */
template <typename T, size_t Size>
auto slice(Matrix<T, Size> &v, size_t i) {
    return slice(v, i, std::make_index_sequence<Size>());
}

/* Return the i-th slice (const) */
template <typename T, size_t Size>
auto slice(const Matrix<T, Size> &v, size_t i) {
    return slice(v, i, std::make_index_sequence<Size>());
}

//! @}
// =======================================================================

NAMESPACE_END(enoki)
