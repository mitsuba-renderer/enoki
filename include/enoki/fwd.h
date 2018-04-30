/*
    enoki/fwd.h -- Preprocessor definitions and forward declarations

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

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
#  define ENOKI_INLINE_LAMBDA
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
#  define ENOKI_INLINE_LAMBDA          __attribute__ ((always_inline))
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

#if defined(__x86_64__) || defined(_M_X64)
#  define ENOKI_X86_64 1
#endif

#if (defined(__i386__) || defined(_M_IX86)) && !defined(ENOKI_X86_64)
#  define ENOKI_X86_32 1
#endif

#if defined(__aarch64__)
#  define ENOKI_ARM_64 1
#elif defined(__arm__)
#  define ENOKI_ARM_32 1
#endif

#if (defined(_MSC_VER) && defined(ENOKI_X86_32)) && !defined(ENOKI_DISABLE_VECTORIZATION)
// Enoki does not support vectorization on 32-bit Windows due to various
// platform limitations (unaligned stack, calling conventions don't allow
// passing vector registers, etc.).
# define ENOKI_DISABLE_VECTORIZATION 1
#endif

# if !defined(ENOKI_DISABLE_VECTORIZATION)
#  if defined(__AVX512F__)
#    define ENOKI_X86_AVX512F 1
#  endif
#  if defined(__AVX512CD__)
#    define ENOKI_X86_AVX512CD 1
#  endif
#  if defined(__AVX512DQ__)
#    define ENOKI_X86_AVX512DQ 1
#  endif
#  if defined(__AVX512VL__)
#    define ENOKI_X86_AVX512VL 1
#  endif
#  if defined(__AVX512BW__)
#    define ENOKI_X86_AVX512BW 1
#  endif
#  if defined(__AVX512PF__)
#    define ENOKI_X86_AVX512PF 1
#  endif
#  if defined(__AVX512ER__)
#    define ENOKI_X86_AVX512ER 1
#  endif
#  if defined(__AVX512VBMI__)
#    define ENOKI_X86_AVX512VBMI 1
#  endif
#  if defined(__AVX512VPOPCNTDQ__)
#    define ENOKI_X86_AVX512VPOPCNTDQ 1
#  endif
#  if defined(__AVX2__)
#    define ENOKI_X86_AVX2 1
#  endif
#  if defined(__FMA__)
#    define ENOKI_X86_FMA 1
#  endif
#  if defined(__F16C__)
#    define ENOKI_X86_F16C 1
#  endif
#  if defined(__AVX__)
#    define ENOKI_X86_AVX 1
#  endif
#  if defined(__SSE4_2__)
#    define ENOKI_X86_SSE42 1
#  endif
#  if defined(__ARM_NEON)
#    define ENOKI_ARM_NEON
#  endif
#  if defined(__ARM_FEATURE_FMA)
#    define ENOKI_ARM_FMA
#  endif
#endif

/* Fix missing/inconsistent preprocessor flags */
#if defined(ENOKI_X86_AVX512F) && !defined(ENOKI_X86_AVX2)
#  define ENOKI_X86_AVX2
#endif

#if defined(ENOKI_X86_AVX2) && !defined(ENOKI_X86_F16C)
#  define ENOKI_X86_F16C
#endif

#if defined(ENOKI_X86_AVX2) && !defined(ENOKI_X86_FMA)
#  define ENOKI_X86_FMA
#endif

#if defined(ENOKI_X86_AVX2) && !defined(ENOKI_X86_AVX)
#  define ENOKI_X86_AVX
#endif

#if defined(ENOKI_X86_AVX) && !defined(ENOKI_X86_SSE42)
#  define ENOKI_X86_SSE42
#endif

/* The following macro is used by the test suite to detect
   unimplemented methods in vectorized backends */

#if !defined(ENOKI_TRACK_SCALAR)
#  define ENOKI_TRACK_SCALAR
#endif

#if defined(ENOKI_ALLOC_VERBOSE)
#  define ENOKI_TRACK_ALLOC   printf("Enoki: %p: alloc(%llu)\n", ptr, (uint64_t) size);
#  define ENOKI_TRACK_REALLOC printf("Enoki: %p -> %p: realloc(%llu)\n", cur, ptr, (uint64_t) size);
#  define ENOKI_TRACK_DEALLOC printf("Enoki: %p: dealloc()\n", ptr);
#endif

#if !defined(ENOKI_TRACK_ALLOC)
#  define ENOKI_TRACK_ALLOC
#endif
#if !defined(ENOKI_TRACK_REALLOC)
#  define ENOKI_TRACK_REALLOC
#endif
#if !defined(ENOKI_TRACK_DEALLOC)
#  define ENOKI_TRACK_DEALLOC
#endif

#define ENOKI_CHKSCALAR if (std::is_arithmetic<std::decay_t<Value>>::value) { ENOKI_TRACK_SCALAR }

#if !defined(ENOKI_APPROX_DEFAULT)
#  define ENOKI_APPROX_DEFAULT 1
#endif

NAMESPACE_BEGIN(enoki)

using ssize_t = std::make_signed_t<size_t>;

/// Maximum hardware-supported packet size in bytes
#if defined(ENOKI_X86_AVX512F)
    static constexpr size_t max_packet_size = 64;
#elif defined(ENOKI_X86_AVX)
    static constexpr size_t max_packet_size = 32;
#elif defined(ENOKI_X86_SSE42) || defined(ENOKI_ARM_NEON)
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

template <typename T>
using is_std_int =
    std::integral_constant<bool, std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8)>;

/// Value trait to determine if a type should be handled using approximate mode by default
template <typename T, typename = int> struct approx_default {
#if ENOKI_APPROX_DEFAULT == 1
    static constexpr bool value = is_std_float<std::decay_t<T>>::value;
#else
    static constexpr bool value = false;
#endif
};

NAMESPACE_END(detail)

// -----------------------------------------------------------------------
//! @{ \name Forward declarations
// -----------------------------------------------------------------------

template <typename Value, typename Derived> struct ArrayBase;

template <typename Value, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
struct StaticArrayBase;

template <typename Value, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename SFINAE = void>
struct StaticArrayImpl;

template <typename Value, typename Derived> struct DynamicArrayBase;

template <typename Value> struct DynamicArray;

struct half;

template <typename Value_,
          size_t Size_ = (max_packet_size / sizeof(Value_) > 1)
                        ? max_packet_size / sizeof(Value_) : 1,
          bool Approx_ = detail::approx_default<Value_>::value,
          RoundingMode Mode_ = RoundingMode::Default>
struct Array;

template <typename Value_,
          size_t Size_ = (max_packet_size / sizeof(Value_) > 1)
                        ? max_packet_size / sizeof(Value_) : 1,
          bool Approx_ = detail::approx_default<Value_>::value,
          RoundingMode Mode_ = RoundingMode::Default>
struct Packet;

template <typename Value_, size_t Size_, bool Approx_ = detail::approx_default<Value_>::value, RoundingMode Mode_ = RoundingMode::Default> struct Mask;
template <typename Value_, size_t Size_, bool Approx_ = detail::approx_default<Value_>::value, RoundingMode Mode_ = RoundingMode::Default> struct PacketMask;

template<typename T, typename U> T memcpy_cast(const U &);

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
