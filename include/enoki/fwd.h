/*
    enoki/fwd.h -- Preprocessor definitions and forward declarations

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  if !defined(_USE_MATH_DEFINES)
#    define _USE_MATH_DEFINES
#  endif
#endif

#include <cstddef>
#include <type_traits>

#if defined(_MSC_VER)
#  define ENOKI_INLINE                 __forceinline
#  define ENOKI_NOINLINE               __declspec(noinline)
#  define ENOKI_MALLOC                 __declspec(restrict)
#  define ENOKI_MAY_ALIAS
#  define ENOKI_ASSUME_ALIGNED(x)      x
#  define ENOKI_ASSUME_ALIGNED_S(x, s) x
#  define ENOKI_UNROLL
#  define ENOKI_NOUNROLL
#  define ENOKI_IVDEP                  __pragma(loop(ivdep))
#  define ENOKI_PACK
#  define ENOKI_LIKELY(x)              x
#  define ENOKI_UNLIKELY(x)            x
#else
#  define ENOKI_NOINLINE               __attribute__ ((noinline))
#  define ENOKI_INLINE                 __attribute__ ((always_inline)) inline
#  define ENOKI_MALLOC                 __attribute__ ((malloc))
#  define ENOKI_ASSUME_ALIGNED(x)      __builtin_assume_aligned(x, ::enoki::max_packet_size)
#  define ENOKI_ASSUME_ALIGNED_S(x, s) __builtin_assume_aligned(x, s)
#  define ENOKI_LIKELY(x)              __builtin_expect(!!(x), 1)
#  define ENOKI_UNLIKELY(x)            __builtin_expect(!!(x), 0)
#  define ENOKI_PACK                   __attribute__ ((packed))
#  if defined(__clang__)
#    define ENOKI_UNROLL               _Pragma("unroll")
#    define ENOKI_NOUNROLL             _Pragma("nounroll")
#    define ENOKI_IVDEP
#    define ENOKI_MAY_ALIAS            __attribute__ ((may_alias))
#  elif defined(__INTEL_COMPILER)
#    define ENOKI_MAY_ALIAS
#    define ENOKI_UNROLL               _Pragma("unroll")
#    define ENOKI_NOUNROLL             _Pragma("nounroll")
#    define ENOKI_IVDEP                _Pragma("ivdep")
#  else
#    define ENOKI_MAY_ALIAS
#    define ENOKI_UNROLL
#    define ENOKI_NOUNROLL
#    if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))
#      define ENOKI_IVDEP              _Pragma("GCC ivdep")
#    else
#      define ENOKI_IVDEP
#    endif
#  endif
#endif

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

#define ENOKI_VERSION_MAJOR 0
#define ENOKI_VERSION_MINOR 1
#define ENOKI_VERSION_PATCH 0

#define ENOKI_STRINGIFY(x) #x
#define ENOKI_TOSTRING(x)  ENOKI_STRINGIFY(x)
#define ENOKI_VERSION                                                          \
    (ENOKI_TOSTRING(ENOKI_VERSION_MAJOR) "."                                   \
     ENOKI_TOSTRING(ENOKI_VERSION_MINOR) "."                                   \
     ENOKI_TOSTRING(ENOKI_VERSION_PATCH))

/* The following macro is used by the test suite to detect
   unimplemented methods in vectorized backends */
#if !defined(ENOKI_TRACK_SCALAR)
#  define ENOKI_TRACK_SCALAR
#endif
#if !defined(ENOKI_TRACK_ALLOC)
#  define ENOKI_TRACK_ALLOC
#endif
#if !defined(ENOKI_TRACK_DEALLOC)
#  define ENOKI_TRACK_DEALLOC
#endif

#define ENOKI_CHKSCALAR if (std::is_arithmetic<Value>::value) { ENOKI_TRACK_SCALAR }

NAMESPACE_BEGIN(enoki)

using ssize_t = std::make_signed_t<size_t>;

/// Maximum hardware-supported packet size in bytes
#if defined(__AVX512F__)
    static constexpr size_t max_packet_size = 64;
#elif defined(__AVX__)
    static constexpr size_t max_packet_size = 32;
#elif defined(__SSE4_2__)
    static constexpr size_t max_packet_size = 16;
#else
    static constexpr size_t max_packet_size = 4;
#endif

/// Choice of rounding modes for floating point operations
enum class RoundingMode {
    /// Default rounding mode configured in the hardware's status register
    Default = 4,

    /// Round to the nearest representable value (tie-breaking method is hardware dependent)
    Nearest = 8,

    /// Always round to negative infinity
    Down = 9,

    /// Always round to positive infinity
    Up = 10,

    /// Always round to zero
    Zero = 11
};

NAMESPACE_BEGIN(detail)

template <typename T>
using is_std_float =
    std::integral_constant<bool, std::is_same<T, float>::value ||
                                 std::is_same<T, double>::value>;

/// Type trait to determine if a type should be handled using approximate mode by default
template <typename T, typename = int> struct approx_default {
    static constexpr bool value = is_std_float<std::decay_t<T>>::value;
};

NAMESPACE_END()

// -----------------------------------------------------------------------
//! @{ \name Forward declarations
// -----------------------------------------------------------------------

template <typename Type, typename Derived> struct ArrayBase;

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
struct StaticArrayBase;

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename SFINAE = void>
struct StaticArrayImpl;

template <typename Type, typename Derived> struct DynamicArrayBase;

template <typename Type> struct DynamicArray;

struct half;

/// Array type
template <typename Type_,
          size_t Size_ = (max_packet_size / sizeof(Type_) > 1)
                        ? max_packet_size / sizeof(Type_) : 1,
          bool Approx_ = detail::approx_default<Type_>::value,
          RoundingMode Mode_ = RoundingMode::Default>
struct Array;

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
