/*
    enoki/matrix.h -- Convenience wrapper for square matrixes

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array.h"

NAMESPACE_BEGIN(enoki)

/// Value trait to access the column type of a matrix
template <typename T> using column_t = typename std::decay_t<T>::Column;

/// Value trait to access the entry type of a matrix
template <typename T> using entry_t = typename std::decay_t<T>::Entry;

/**
 * \brief Dense square matrix data structure of static size
 * \remark Uses column-major storage order to permit efficient vectorization
 */
template <typename Value_, size_t Size_, bool Approx_ = detail::approx_default<Value_>::value>
struct Matrix
    : StaticArrayImpl<Array<Value_, Size_, Approx_>, Size_, Approx_,
                      RoundingMode::Default, Matrix<Value_, Size_, Approx_>> {

    using Column = Array<Value_, Size_, Approx_>;

    using Base = StaticArrayImpl<Column, Size_, Approx_, RoundingMode::Default,
                                 Matrix<Value_, Size_, Approx_>>;

    using typename Base::Value;
    using typename Base::Scalar;
    using Entry = value_t<Column>;

    template <typename T, typename T2 = Matrix>
    using ReplaceType = Matrix<
        value_t<T>, Size_,
        detail::is_std_float<scalar_t<T>>::value ? T2::Approx
                                                 : detail::approx_default<T>::value>;

    ENOKI_DECLARE_CUSTOM_ARRAY(Base, Matrix)

    using Base::coeff;
    using Base::Size;
    using Base::data;

    /// Initialize the matrix from a set of coefficients
    template <typename... Args, std::enable_if_t<
              detail::all_of<std::is_constructible<Entry, Args>::value...,
                             sizeof...(Args) == Size_ * Size_>::value, int> = 0>
    ENOKI_INLINE Matrix(const Args&... args) {
        alignas(alignof(Column)) Entry values[sizeof...(Args)] = { Entry(args)... };
        for (size_t j = 0; j < Size; ++j)
            for (size_t i = 0; i < Size; ++i)
                coeff(j).coeff(i) = values[i * Size + j];
    }

    /// Create a diagonal matrix
    ENOKI_INLINE Matrix(const Entry &f) : Base(zero<Entry>()) {
        for (size_t i = 0; i < Matrix::Size; ++i)
            operator()(i, i) = f;
    }

    /// Create a diagonal matrix
    template <typename T = Entry, std::enable_if_t<!std::is_same<T, Scalar>::value, int> = 0>
    ENOKI_INLINE Matrix(const Scalar &f) : Base(zero<Entry>()) {
        for (size_t i = 0; i < Matrix::Size; ++i)
            operator()(i, i) = f;
    }

    /// Initialize from a bigger matrix: retains the top left part
    template <size_t Size2, bool Approx2, std::enable_if_t<(Size2 > Size_), int> = 0>
    explicit ENOKI_INLINE Matrix(const Matrix<Value_, Size2, Approx2> &m) {
        for (size_t i = 0; i < Size; ++i)
            coeff(i) = head<Size>(m.coeff(i));
    }

    /// Initialize from a smaller matrix: copies to the top left part and resets
    /// the remainder to the identity matrix
    template <size_t Size2, bool Approx2, std::enable_if_t<(Size2 < Size_), int> = 0>
    explicit ENOKI_INLINE Matrix(const Matrix<Value_, Size2, Approx2> &m) {
        using Remainder = Array<Value_, Size - Size2>;
        for (size_t i = 0; i < Size2; ++i)
            coeff(i) = concat(m.coeff(i), zero<Remainder>());
        for (size_t i = Size2; i < Size; ++i) {
            auto col = zero<Column>();
            col.coeff(i) = 1;
            coeff(i) = col;
        }
    }

    template <typename... Column>
    ENOKI_INLINE static Matrix from_cols(const Column&... cols) {
        return Matrix(cols...);
    }

    template <typename... Row>
    ENOKI_INLINE static Matrix from_rows(const Row&... rows) {
        return transpose(Matrix(rows...));
    }

    ENOKI_INLINE Column& col(size_t index) { return coeff(index); }
    ENOKI_INLINE const Column& col(size_t index) const { return coeff(index); }

    ENOKI_INLINE Column row(size_t index) const {
        using Index = Array<uint32_t, Size>;
        return gather<Column>(coeff(0).data() + index,
                              index_sequence<Index>() * uint32_t(Size));
    }

    /// Return a reference to the (i, j) element
    ENOKI_INLINE Entry& operator()(size_t i, size_t j) { return coeff(j, i); }

    /// Return a reference to the (i, j) element (const)
    ENOKI_INLINE const Entry& operator()(size_t i, size_t j) const { return coeff(j, i); }

    template <typename T>
    ENOKI_INLINE static Matrix fill_(const T &value) { return Array<Column, Size>::fill_(value); }

    ENOKI_ALIGNED_OPERATOR_NEW()
};


NAMESPACE_BEGIN(detail)

template <typename T0, typename T1, size_t Size, bool Approx0, bool Approx1,
          typename Return = Matrix<expr_t<T0, T1>, Size, Approx0 && Approx1>,
          typename Column = column_t<Return>>
ENOKI_INLINE Return matrix_mul(const Matrix<T0, Size, Approx0> &m0, const Matrix<T1, Size, Approx1> &m1) {
    Return result;
    /* 4x4 case reduced to 4 multiplications, 12 fused multiply-adds,
       and 16 broadcasts (also fused on AVX512VL) */
    for (size_t j = 0; j < Size; ++j) {
        Column sum = m0.coeff(0) * m1(0, j);
        for (size_t i = 1; i < Size; ++i)
            sum = fmadd(m0.coeff(i), Column(m1(i, j)), sum);
        result.coeff(j) = sum;
    }

    return result;
}

NAMESPACE_END(detail)

template <typename T0, typename T1, size_t Size, bool Approx0, bool Approx1>
ENOKI_INLINE auto operator*(const Matrix<T0, Size, Approx0> &m0, const Matrix<T1, Size, Approx1> &m1) {
    return detail::matrix_mul(m0, m1);
}

template <typename T0, size_t Size, bool Approx>
ENOKI_INLINE auto operator*(const Matrix<T0, Size, Approx> &m0, const Matrix<T0, Size, Approx> &m1) {
    return detail::matrix_mul(m0, m1);
}

template <typename T0, typename T1, size_t Size, bool Approx, std::enable_if_t<array_size<T1>::value == Size, int> = 0,
          typename Return = column_t<Matrix<expr_t<T0, value_t<T1>>, Size, Approx>>>
ENOKI_INLINE Return operator*(const Matrix<T0, Size, Approx> &m, const T1 &v) {
    Return sum = m.coeff(0) * v.derived().coeff(0);
    for (size_t i = 1; i < Size; ++i)
        sum = fmadd(m.coeff(i), v.derived().coeff(i), sum);
    return sum;
}

template <typename T0, typename T1, size_t Size, bool Approx, std::enable_if_t<array_size<T1>::value != Size, int> = 0>
ENOKI_INLINE Matrix<expr_t<T0, T1>, Size, Approx> operator*(const Matrix<T0, Size, Approx> &m, const T1 &s) {
    return Array<Array<expr_t<T0>, Size>, Size>(m) *
           fill<Array<Array<scalar_t<T1>, Size>, Size>>(s);
}

template <typename T0, typename T1, size_t Size, bool Approx>
ENOKI_INLINE Matrix<expr_t<T0, T1>, Size, Approx> operator*(const T0 &s, const Matrix<T1, Size, Approx> &m) {
    return fill<Array<Array<scalar_t<T0>, Size>, Size>>(s) *
           Array<Array<expr_t<T1>, Size>, Size>(m);
}

template <typename Value, size_t Size, bool Approx>
ENOKI_INLINE expr_t<Value> trace(const Matrix<Value, Size, Approx> &m) {
    expr_t<Value> result = m.coeff(0, 0);
    for (size_t i = 1; i < Size; ++i)
        result += m(i, i);
    return result;
}

template <typename Value, size_t Size, bool Approx> ENOKI_INLINE expr_t<Value> frob(const Matrix<Value, Size, Approx> &matrix) {
    expr_t<column_t<Matrix<Value, Size, Approx>>> result = matrix.coeff(0) * matrix.coeff(0);
    for (size_t i = 1; i < Size; ++i)
        result = fmadd(matrix.coeff(i), matrix.coeff(i), result);
    return hsum(result);
}

template <typename Matrix, std::enable_if_t<Matrix::IsMatrix, int> = 0> ENOKI_INLINE Matrix identity() {
    Matrix result = zero<Matrix>();
    for (size_t i = 0; i < Matrix::Size; ++i)
        result(i, i) = 1;
    return result;
}

template <typename Matrix, std::enable_if_t<Matrix::IsMatrix, int> = 0>
ENOKI_INLINE Matrix diag(const column_t<Matrix> &value) {
    Matrix result = zero<Matrix>();
    for (size_t i = 0; i < Matrix::Size; ++i)
        result(i, i) = value.coeff(i);
    return result;
}

template <typename Matrix>
ENOKI_INLINE column_t<expr_t<Matrix>> diag(const Matrix &value) {
    column_t<expr_t<Matrix>> result;
    for (size_t i = 0; i < Matrix::Size; ++i)
        result.coeff(i) = value(i, i);
    return result;
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 1, Approx> inverse(const Matrix<T, 1, Approx> &m) {
    return rcp<Array<T, 1>::Approx>(m(0, 0));
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 1, Approx> inverse_transpose(const Matrix<T, 1, Approx> &m) {
    return rcp<Array<T, 1>::Approx>(m(0, 0));
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE E det(const Matrix<T, 1, Approx> &m) {
    return m(0, 0);
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 2, Approx> inverse(const Matrix<T, 2, Approx> &m) {
    E inv_det = rcp<Approx>(fmsub(m(0, 0), m(1, 1), m(0, 1) * m(1, 0)));
    return Matrix<E, 2, Approx>(
        m(1, 1) * inv_det, -m(0, 1) * inv_det,
       -m(1, 0) * inv_det,  m(0, 0) * inv_det
    );
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE E det(const Matrix<T, 2, Approx> &m) {
    return fmsub(m(0, 0), m(1, 1), m(0, 1) * m(1, 0));
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 2, Approx> inverse_transpose(const Matrix<T, 2, Approx> &m) {
    E inv_det = rcp<Approx>(fmsub(m(0, 0), m(1, 1), m(0, 1) * m(1, 0)));
    return Matrix<E, 2, Approx>(
        m(1, 1) * inv_det, -m(1, 0) * inv_det,
       -m(0, 1) * inv_det,  m(0, 0) * inv_det
    );
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 3, Approx> inverse_transpose(const Matrix<T, 3, Approx> &m) {
    using Vector = Array<E, 3>;

    Vector col0 = m.coeff(0),
           col1 = m.coeff(1),
           col2 = m.coeff(2);

    Vector row0 = cross(col1, col2);
    Vector row1 = cross(col2, col0);
    Vector row2 = cross(col0, col1);

    Vector inv_det = Vector(rcp<Vector::Approx>(dot(col0, row0)));

    return Matrix<E, 3, Approx>(
        row0 * inv_det,
        row1 * inv_det,
        row2 * inv_det
    );
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 3, Approx> inverse(const Matrix<T, 3, Approx> &m) {
    return transpose(inverse_transpose(m));
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE E det(const Matrix<T, 3, Approx> &m) {
    return dot(m.coeff(0), cross(m.coeff(1), m.coeff(2)));
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 4, Approx> inverse_transpose(const Matrix<T, 4, Approx> &m) {
    using Vector = Array<E, 4>;

    Vector col0 = m.coeff(0), col1 = m.coeff(1),
           col2 = m.coeff(2), col3 = m.coeff(3);

    col1 = shuffle<2, 3, 0, 1>(col1);
    col3 = shuffle<2, 3, 0, 1>(col3);

    Vector tmp, row0, row1, row2, row3;

    tmp = shuffle<1, 0, 3, 2>(col2 * col3);
    row0 = col1 * tmp;
    row1 = col0 * tmp;
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row0 = fmsub(col1, tmp, row0);
    row1 = shuffle<2, 3, 0, 1>(fmsub(col0, tmp, row1));

    tmp = shuffle<1, 0, 3, 2>(col1 * col2);
    row0 = fmadd(col3, tmp, row0);
    row3 = col0 * tmp;
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row0 = fnmadd(col3, tmp, row0);
    row3 = shuffle<2, 3, 0, 1>(fmsub(col0, tmp, row3));

    tmp = shuffle<1, 0, 3, 2>(shuffle<2, 3, 0, 1>(col1) * col3);
    col2 = shuffle<2, 3, 0, 1>(col2);
    row0 = fmadd(col2, tmp, row0);
    row2 = col0 * tmp;
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row0 = fnmadd(col2, tmp, row0);
    row2 = shuffle<2, 3, 0, 1>(fmsub(col0, tmp, row2));

    tmp = shuffle<1, 0, 3, 2>(col0 * col1);
    row2 = fmadd(col3, tmp, row2);
    row3 = fmsub(col2, tmp, row3);
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row2 = fmsub(col3, tmp, row2);
    row3 = fnmadd(col2, tmp, row3);

    tmp = shuffle<1, 0, 3, 2>(col0 * col3);
    row1 = fnmadd(col2, tmp, row1);
    row2 = fmadd(col1, tmp, row2);
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row1 = fmadd(col2, tmp, row1);
    row2 = fnmadd(col1, tmp, row2);

    tmp = shuffle<1, 0, 3, 2>(col0 * col2);
    row1 = fmadd(col3, tmp, row1);
    row3 = fnmadd(col1, tmp, row3);
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row1 = fnmadd(col3, tmp, row1);
    row3 = fmadd(col1, tmp, row3);

    Vector inv_det = Vector(rcp<Vector::Approx>(dot(col0, row0)));

    return Matrix<E, 4, Approx>(
        row0 * inv_det, row1 * inv_det,
        row2 * inv_det, row3 * inv_det
    );
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE Matrix<E, 4, Approx> inverse(const Matrix<T, 4, Approx> &m) {
    return transpose(inverse_transpose(m));
}

template <typename T, bool Approx, typename E = expr_t<T>> ENOKI_INLINE E det(const Matrix<T, 4, Approx> &m) {
    using Vector = Array<E, 4>;

    Vector col0 = m.coeff(0), col1 = m.coeff(1),
           col2 = m.coeff(2), col3 = m.coeff(3);

    col1 = shuffle<2, 3, 0, 1>(col1);
    col3 = shuffle<2, 3, 0, 1>(col3);

    Vector tmp, row0;

    tmp = shuffle<1, 0, 3, 2>(col2 * col3);
    row0 = col1 * tmp;
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row0 = fmsub(col1, tmp, row0);

    tmp = shuffle<1, 0, 3, 2>(col1 * col2);
    row0 = fmadd(col3, tmp, row0);
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row0 = fnmadd(col3, tmp, row0);

    col1 = shuffle<2, 3, 0, 1>(col1);
    col2 = shuffle<2, 3, 0, 1>(col2);
    tmp = shuffle<1, 0, 3, 2>(col1 * col3);
    row0 = fmadd(col2, tmp, row0);
    tmp = shuffle<2, 3, 0, 1>(tmp);
    row0 = fnmadd(col2, tmp, row0);

    return dot(col0, row0);
}

#if defined(__SSE4_2__)
// Optimized 3x3 transpose (single precision)
template <typename Value, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Value::Size == 3 && std::is_same<value_t<Value>, float>::value, int> = 0>
ENOKI_INLINE auto transpose(const StaticArrayBase<Value, 3, Approx, Mode, Derived> &a) {
    __m128 c0 = a.derived().coeff(0).m,
           c1 = a.derived().coeff(1).m,
           c2 = a.derived().coeff(2).m;

    __m128 t0 = _mm_unpacklo_ps(c0, c1);
    __m128 t1 = _mm_unpacklo_ps(c2, c2);
    __m128 t2 = _mm_unpackhi_ps(c0, c1);
    __m128 t3 = _mm_unpackhi_ps(c2, c2);

    return Derived(
        _mm_movelh_ps(t0, t1),
        _mm_movehl_ps(t1, t0),
        _mm_movelh_ps(t2, t3)
    );
}

// Optimized 4x4 transpose (single precision)
template <typename Value, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Value::Size == 4 && std::is_same<value_t<Value>, float>::value, int> = 0>
ENOKI_INLINE auto
transpose(const StaticArrayBase<Value, 4, Approx, Mode, Derived> &a) {
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
#endif

#if defined(__AVX__)
// Optimized 3x3 transpose (double precision)
template <typename Value, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Value::Size == 3 && std::is_same<value_t<Value>, double>::value, int> = 0>
ENOKI_INLINE auto
transpose(const StaticArrayBase<Value, 3, Approx, Mode, Derived> &a) {
    __m256d c0 = a.derived().coeff(0).m,
            c1 = a.derived().coeff(1).m,
            c2 = a.derived().coeff(2).m;

    __m256d t3 = _mm256_shuffle_pd(c2, c2, 0b0000),
            t2 = _mm256_shuffle_pd(c2, c2, 0b1111),
            t1 = _mm256_shuffle_pd(c0, c1, 0b0000),
            t0 = _mm256_shuffle_pd(c0, c1, 0b1111);

    return Derived(
        _mm256_permute2f128_pd(t1, t3, 0b0010'0000),
        _mm256_permute2f128_pd(t0, t2, 0b0010'0000),
        _mm256_permute2f128_pd(t1, t3, 0b0011'0001)
    );
}

// Optimized 4x4 transpose (double precision)
template <typename Value, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<Value::Size == 4 && std::is_same<value_t<Value>, double>::value, int> = 0>
ENOKI_INLINE auto transpose(const StaticArrayBase<Value, 4, Approx, Mode, Derived> &a) {
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

template <typename Value, size_t Size, bool Approx, RoundingMode Mode, typename Derived>
ENOKI_INLINE auto transpose(const StaticArrayBase<Value, Size, Approx, Mode, Derived> &a) {
    static_assert(Value::Size == Size && array_depth<Derived>::value >= 2,
                  "Array must be a square matrix!");
    Derived result;
    ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
        for (size_t j = 0; j < Size; ++j)
            result.coeff(i).coeff(j) = a.derived().coeff(j).coeff(i);
    return result;
}

template <typename T, size_t Size, bool Approx, typename Expr = expr_t<T>, typename Matrix = Matrix<Expr, Size, Approx>>
std::pair<Matrix, Matrix> ENOKI_INLINE polar_decomp(const enoki::Matrix<T, Size, Approx> &A, size_t it = 10) {
    using Arr = Array<Array<Expr, Size>, Size>;
    Matrix Q = A;
    for (size_t i = 0; i < it; ++i) {
        Matrix Qi = inverse_transpose(Q);
        Expr gamma = sqrt(frob(Qi) / frob(Q));
        Q = fmadd(Arr(Q), gamma * .5f, Arr(Qi) * (rcp(gamma) * 0.5f));
    }
    return std::make_pair(Q, transpose(Q) * A);
}

// =======================================================================
//! @{ \name Enoki accessors for static & dynamic vectorization
// =======================================================================

template <typename T, size_t Size, bool Approx>
struct struct_support<Matrix<T, Size, Approx>,
                      enable_if_static_array_t<Matrix<T, Size, Approx>>> {
    static constexpr bool is_dynamic_nested = enoki::is_dynamic_nested<T>::value;
    using dynamic_t = Matrix<enoki::make_dynamic_t<T>, Size, Approx>;
    using Value = Matrix<T, Size, Approx>;

    static ENOKI_INLINE size_t slices(const Value &value) {
        return enoki::slices(value.coeff(0, 0));
    }

    static ENOKI_INLINE size_t packets(const Value &value) {
        return enoki::packets(value.coeff(0, 0));
    }

    static ENOKI_INLINE void set_slices(Value &value, size_t size) {
        for (size_t i = 0; i < Size; ++i)
            enoki::set_slices(value.coeff(i), size);
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
    static ENOKI_INLINE auto slice_ptr(T2&& value, size_t i) {
        return slice_ptr(value, i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto ref_wrap(T2&& value) {
        return ref_wrap(value, std::make_index_sequence<Size>());
    }

private:
    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto packet(T2&& value, size_t i, std::index_sequence<Index...>) {
        return Matrix<decltype(enoki::packet(value.coeff(0, 0), i)), Size, Approx>(
            enoki::packet(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice(T2&& value, size_t i, std::index_sequence<Index...>) {
        return Matrix<decltype(enoki::slice(value.coeff(0, 0), i)), Size, Approx>(
            enoki::slice(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice_ptr(T2&& value, size_t i, std::index_sequence<Index...>) {
        return Matrix<decltype(enoki::slice_ptr(value.coeff(0, 0), i)), Size, Approx>(
            enoki::slice_ptr(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto ref_wrap(T2&& value, std::index_sequence<Index...>) {
        return Matrix<decltype(enoki::ref_wrap(value.coeff(0, 0))), Size, Approx>(
            enoki::ref_wrap(value.coeff(Index))...);
    }
};

//! @}
// =======================================================================

NAMESPACE_END(enoki)
