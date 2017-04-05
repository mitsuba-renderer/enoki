/*
    enoki/morton.h -- Morton/Z-order curve encoding and decoding routines

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>
    Includes contributions by Sebastien Speierer

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array.h"

#if defined(_MSC_VER)
#  pragma warning (push)
#  pragma warning (disable: 4310) // cast truncates constant value
#endif

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

/// Generate bit masks for the functions \ref scatter_bits() and \ref gather_bits()
template <typename Type> constexpr Type morton_magic(size_t dim, size_t level) {
    size_t n_bits = sizeof(Type) * 8;
    size_t max_block_size = n_bits / dim;
    size_t block_size = std::min(size_t(1) << (level - 1), max_block_size);
    size_t count = 0;

    Type mask  = Type(1) << (n_bits - 1),
         value = Type(0);

    for (size_t i = 0; i < n_bits; ++i) {
        value >>= 1;

        if (count < max_block_size && (i / block_size) % dim == 0) {
            count++;
            value |= mask;
        }
    }

    return value;
}

constexpr size_t clog2i(size_t value) {
    return (value > 1) ? 1 + clog2i(value >> 1) : 0;
}

/// Bit scatter function. \c Dimension defines the final distance between two output bits
template <size_t, typename Type, size_t Level,
          std::enable_if_t<Level == 0, int> = 0>
ENOKI_INLINE Type scatter_bits(Type x) { return x; }

template <size_t Dimension, typename Type,
          size_t Level = clog2i(sizeof(Type) * 8),
          std::enable_if_t<Level != 0 && (!has_avx2 || !std::is_integral<Type>::value), int> = 0>
ENOKI_INLINE Type scatter_bits(Type x) {
    using Scalar = scalar_t<Type>;

    constexpr Scalar magic = morton_magic<Scalar>(Dimension, Level);
    constexpr size_t shift_maybe = (1 << (Level - 1)) * (Dimension - 1);
    constexpr size_t shift = (shift_maybe < sizeof(Scalar) * 8) ? shift_maybe : 0;

    if (shift)
        x |= sli<shift>(x);

    x &= magic;

    return scatter_bits<Dimension, Type, Level - 1>(x);
}

template <size_t, typename Type, size_t Level,
          std::enable_if_t<Level == 0, int> = 0>
ENOKI_INLINE Type gather_bits(Type x) { return x; }

/// Bit gather function. \c Dimension defines the final distance between two input bits
template <size_t Dimension, typename Type,
          size_t Level = clog2i(sizeof(Type) * 8),
          std::enable_if_t<Level != 0 && (!has_avx2 || !std::is_integral<Type>::value), int> = 0>
ENOKI_INLINE Type gather_bits(Type x) {
    using Scalar = scalar_t<Type>;

    constexpr size_t ilevel = clog2i(sizeof(Type) * 8) - Level + 1;
    constexpr Scalar magic = morton_magic<Scalar>(Dimension, ilevel);
    constexpr size_t shift_maybe = (1 << (ilevel - 1)) * (Dimension - 1);
    constexpr size_t shift = (shift_maybe < sizeof(Scalar) * 8) ? shift_maybe : 0;

    x &= magic;

    if (shift)
        x |= sri<shift>(x);

    return gather_bits<Dimension, Type, Level - 1>(x);
}

#if defined(__AVX2__)
template <size_t Dimension, typename Type,
          std::enable_if_t<std::is_integral<Type>::value, int> = 0>
ENOKI_INLINE Type scatter_bits(Type x) {
    constexpr Type magic = detail::morton_magic<Type>(Dimension, 1);
    if (sizeof(Type) <= 4)
        return Type(_pdep_u32((uint32_t) x, (uint32_t) magic));
    else
        return Type(_pdep_u64((uint64_t) x, (uint64_t) magic));
}

template <size_t Dimension, typename Type,
          std::enable_if_t<std::is_integral<Type>::value, int> = 0>
ENOKI_INLINE Type gather_bits(Type x) {
    constexpr Type magic = detail::morton_magic<Type>(Dimension, 1);
    if (sizeof(Type) <= 4)
        return Type(_pext_u32((uint32_t) x, (uint32_t) magic));
    else
        return Type(_pext_u64((uint64_t) x, (uint64_t) magic));
}
#endif

template <typename Array, size_t Index,
          std::enable_if_t<Index == 0, int> = 0>
ENOKI_INLINE void morton_decode_helper(value_t<Array> value, Array &out) {
    out.coeff(0) = gather_bits<Array::Size>(value);
}

template <typename Array, size_t Index = array_size<Array>::value - 1,
          std::enable_if_t<Index != 0, int> = 0>
ENOKI_INLINE void morton_decode_helper(value_t<Array> value, Array &out) {
    out.coeff(Index) = gather_bits<Array::Size>(sri<Index>(value));
    morton_decode_helper<Array, Index - 1>(value, out);
}

NAMESPACE_END(detail)

/// Convert a N-dimensional integer array into the Morton/Z-order curve encoding
template <typename Array, size_t Index, typename Return = value_t<Array>,
          std::enable_if_t<Index == 0, int> = 0>
ENOKI_INLINE Return morton_encode(Array a) {
    return detail::scatter_bits<Array::Size>(a.coeff(0));
}

/// Convert a N-dimensional integer array into the Morton/Z-order curve encoding
template <typename Array, size_t Index = array_size<Array>::value - 1,
          typename Return = value_t<Array>, std::enable_if_t<Index != 0, int> = 0>
ENOKI_INLINE Return morton_encode(Array a) {
    static_assert(std::is_unsigned<scalar_t<Array>>::value, "morton_encode() requires unsigned arguments");
    return sli<Index>(detail::scatter_bits<Array::Size>(a.coeff(Index))) |
           morton_encode<Array, Index - 1>(a);
}

/// Convert Morton/Z-order curve encoding into a N-dimensional integer array
template <typename Array, typename Value = value_t<Array>>
ENOKI_INLINE Array morton_decode(Value value) {
    static_assert(std::is_unsigned<scalar_t<Array>>::value, "morton_decode() requires unsigned arguments");
    Array result;
    detail::morton_decode_helper(value, result);
    return result;
}

NAMESPACE_END(enoki)

#if defined(_MSC_VER)
#  pragma warning (pop)
#endif