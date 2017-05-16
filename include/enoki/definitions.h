/*
    enoki/definitions.h -- Preprocessor definitions

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(_MSC_VER)
#  if !defined(_USE_MATH_DEFINES)
#    define _USE_MATH_DEFINES
#  endif
#endif

#if defined(_MSC_VER)
#  define ENOKI_INLINE            __forceinline
#  define ENOKI_NOINLINE          __declspec(noinline)
#  define ENOKI_MALLOC            __declspec(restrict)
#  define ENOKI_ASSUME_ALIGNED(x) x
#  define ENOKI_UNROLL
#  define ENOKI_NOUNROLL
#  define ENOKI_IVDEP             __pragma(loop(ivdep))
#  define ENOKI_PACK
#  define ENOKI_LIKELY(x)         x
#  define ENOKI_UNLIKELY(x)       x
#else
#  define ENOKI_NOINLINE          __attribute__ ((noinline))
#  define ENOKI_INLINE            __attribute__ ((always_inline)) inline
#  define ENOKI_MALLOC            __attribute__ ((malloc))
#  define ENOKI_ASSUME_ALIGNED(x) __builtin_assume_aligned(x, ENOKI_MAX_PACKET_SIZE)
#  define ENOKI_LIKELY(x)         __builtin_expect(!!(x), 1)
#  define ENOKI_UNLIKELY(x)       __builtin_expect(!!(x), 0)
#  define ENOKI_PACK              __attribute__ ((packed))
#  if defined(__clang__)
#    define ENOKI_UNROLL          _Pragma("unroll")
#    define ENOKI_NOUNROLL        _Pragma("nounroll")
#    define ENOKI_IVDEP
#  elif defined(__INTEL_COMPILER)
#    define ENOKI_UNROLL          _Pragma("unroll")
#    define ENOKI_NOUNROLL        _Pragma("nounroll")
#    define ENOKI_IVDEP           _Pragma("ivdep")
#  else
#    define ENOKI_UNROLL
#    define ENOKI_NOUNROLL
#    if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))
#      define ENOKI_IVDEP     _Pragma("GCC ivdep")
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

/// Maximum hardware-supported packet size in bytes
#if defined(__AVX512F__)
#  define ENOKI_MAX_PACKET_SIZE 64
#elif defined(__AVX__)
#  define ENOKI_MAX_PACKET_SIZE 32
#elif defined(__SSE4_2__)
#  define ENOKI_MAX_PACKET_SIZE 16
#else
#  define ENOKI_MAX_PACKET_SIZE 4
#endif
