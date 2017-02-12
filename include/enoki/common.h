/*
    enoki/common.h -- Common definitions and template helpers

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#if defined(_MSC_VER)
#  if !defined(_USE_MATH_DEFINES)
#    define _USE_MATH_DEFINES
#  endif
#endif

#include <cmath>
#include <type_traits>
#include <array>
#include <functional>
#include <ostream>
#include <sstream>
#include <cstring>
#include <string>
#include <limits>
#include <cassert>
#include <immintrin.h>

#if defined(_MSC_VER)
#  include <intrin.h>
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
#  define ENOKI_LIKELY(x)
#  define ENOKI_UNLIKELY(x)
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

#define ENOKI_CHKSCALAR if (std::is_arithmetic<Scalar>::value) { ENOKI_TRACK_SCALAR }

NAMESPACE_BEGIN(enoki)
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

/* Fix missing/inconsistent preprocessor flags */
#if defined(__AVX512F__) && !defined(__AVX2__)
#  define __AVX2__
#endif

#if defined(__AVX2__) && !defined(__F16C__)
#  define __F16C__
#endif

#if defined(__AVX2__) && !defined(__FMA__)
#  define __FMA__
#endif

#if defined(__AVX2__) && !defined(__AVX__)
#  define __AVX__
#endif

#if defined(__AVX__) && !defined(__SSE4_2__)
#  define __SSE4_2__
#endif

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

// -----------------------------------------------------------------------
//! @{ \name Available instruction sets
// -----------------------------------------------------------------------

#if defined(__AVX512DQ__)
    static constexpr bool has_avx512dq = true;
#else
    static constexpr bool has_avx512dq = false;
#endif

#if defined(__AVX512VL__)
    static constexpr bool has_avx512vl = true;
#else
    static constexpr bool has_avx512vl = false;
#endif

#if defined(__AVX512BW__)
    static constexpr bool has_avx512bw = true;
#else
    static constexpr bool has_avx512bw = false;
#endif

#if defined(__AVX512CD__)
    static constexpr bool has_avx512cd = true;
#else
    static constexpr bool has_avx512cd = false;
#endif

#if defined(__AVX512PF__)
    static constexpr bool has_avx512pf = true;
#else
    static constexpr bool has_avx512pf = false;
#endif

#if defined(__AVX512ER__)
    static constexpr bool has_avx512er = true;
#else
    static constexpr bool has_avx512er = false;
#endif

#if defined(__AVX512F__)
    static constexpr bool has_avx512f = true;
#else
    static constexpr bool has_avx512f = false;
#endif

#if defined(__AVX2__)
    static constexpr bool has_avx2 = true;
#else
    static constexpr bool has_avx2 = false;
#endif

#if defined(__FMA__)
    static constexpr bool has_fma = true;
#else
    static constexpr bool has_fma = false;
#endif

#if defined(__F16C__)
    static constexpr bool has_f16c = true;
#else
    static constexpr bool has_f16c = false;
#endif

#if defined(__AVX__)
    static constexpr bool has_avx = true;
#else
    static constexpr bool has_avx = false;
#endif

#if defined(__SSE4_2__)
    static constexpr bool has_sse42 = true;
#else
    static constexpr bool has_sse42 = false;
#endif

static constexpr bool has_vectorization = has_sse42;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Forward declarations
// -----------------------------------------------------------------------

template <typename Type, typename Derived> struct ArrayBase;
template <typename Type, size_t Size, bool Approx, RoundingMode Mode, typename Derived> struct StaticArrayBase;
template <typename Type, size_t Size, bool Approx, RoundingMode Mode, typename Derived, typename SFINAE = void> struct StaticArrayImpl;
template <typename Type, typename Derived> struct DynamicArrayBase;
struct half;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Type traits
// -----------------------------------------------------------------------

/// SFINAE helper to test whether a class is a static or dynamic array type
template <typename T> struct is_array {
private:
    static constexpr std::false_type check(void *);
    template <typename Type, typename Derived>
    static constexpr std::true_type check(ArrayBase<Type, Derived> *);
public:
    using type = decltype(check((T *) nullptr));
    static constexpr bool value = type::value;
};

/// SFINAE helper to test whether a class is a static array type
template <typename T> struct is_sarray {
private:
    static constexpr std::false_type check(void *);
    template <typename Type, size_t Size, bool Approx, RoundingMode Mode, typename Derived>
    static constexpr std::true_type check(StaticArrayBase<Type, Size, Approx, Mode, Derived> *);
public:
    using type = decltype(check((T *) nullptr));
    static constexpr bool value = type::value;
};

/// SFINAE helper to test whether a class is a dynamic array type
template <typename T> struct is_darray {
private:
    static constexpr std::false_type check(void *);
    template <typename Type, typename Derived>
    static constexpr std::true_type check(DynamicArrayBase<Type, Derived> *);
public:
    using type = decltype(check((T *) nullptr));
    static constexpr bool value = type::value;
};

template <typename T>
using enable_if_array_t = std::enable_if_t<is_array<T>::value, int>;

template <typename T>
using enable_if_sarray_t = std::enable_if_t<is_sarray<T>::value, int>;

template <typename T>
using enable_if_darray_t = std::enable_if_t<is_darray<T>::value, int>;

template <typename T>
using enable_if_notarray_t = std::enable_if_t<!is_array<T>::value, int>;

/// Type trait to access the mask type underlying an array
template <typename T, typename = void> struct mask { using type = bool; };

template <typename T>
struct mask<T, std::enable_if_t<is_sarray<std::decay_t<T>>::value>> {
    using type = typename std::decay_t<T>::Mask;
};

/// Type trait to access the scalar type underlying an array
template <typename T, typename = void>
struct scalar {
    using type = T;
};

template <typename T>
struct scalar<T, std::enable_if_t<is_array<std::decay_t<T>>::value>> {
    using type = typename std::decay_t<T>::Scalar;
};

/// Type trait to access the base scalar type underlying a potentially nested array
template <typename T, typename = void>
struct base_scalar {
    using type = T;
};

template <typename T>
struct base_scalar<T, std::enable_if_t<is_array<std::decay_t<T>>::value>> {
    using type = typename base_scalar<typename std::decay_t<T>::Scalar>::type;
};

/// Type trait to access the type that would result from an unary expression involving another type
template <typename T, typename = void>
struct expr {
    using type = std::decay_t<T>;
};

template <typename T>
struct expr<T, std::enable_if_t<is_sarray<std::decay_t<T>>::value>> {
    using type = typename std::decay_t<T>::Expr;
};

template <typename T> using scalar_t = typename scalar<T>::type;
template <typename T> using base_scalar_t = typename base_scalar<T>::type;
template <typename T> using mask_t = typename mask<T>::type;
template <typename T> using expr_t = typename expr<T>::type;

template <typename T, typename = void> struct is_dynamic : std::false_type { };

template <typename T> struct is_dynamic<T, std::enable_if_t<is_sarray<std::decay_t<T>>::value>> {
    static constexpr bool value = is_dynamic<typename std::decay_t<T>::Scalar>::value;
};

template <typename T>
struct is_dynamic<T, std::enable_if_t<is_darray<std::decay_t<T>>::value>> : std::true_type { };

/// Determine the nesting level of an array
template <typename T, typename = void> struct array_depth {
    static constexpr size_t value = 0;
};

template <typename T> struct array_depth<T, std::enable_if_t<enoki::is_array<std::decay_t<T>>::value>> {
    static constexpr size_t value = array_depth<typename std::decay_t<T>::Scalar>::value + 1;
};

NAMESPACE_BEGIN(detail)

/// Type trait to determine if a type should be handled using approximate mode by default
template <typename T, typename = void> struct approx_default {
    static constexpr bool value = std::is_same<std::decay_t<T>, float>::value;
};

template <typename T>
struct approx_default<T, std::enable_if_t<is_sarray<std::decay_t<T>>::value>> {
    static constexpr bool value = std::decay_t<T>::Approx;
};

/// Determines when an operand of a binary operation requires a prior bcast step
template <typename Target, typename Source, typename = void>
struct bcast {
    static constexpr bool value = is_array<Target>::value;
};

template <typename Target, typename Source>
struct bcast<Target, Source,
                 std::enable_if_t<is_array<Target>::value &&
                                  is_array<Source>::value>> {
    static constexpr bool value = Target::Derived::Size != Source::Derived::Size &&
                                  array_depth<Target>::value > array_depth<Source>::value;
};

/// Type equivalence between arithmetic type to work around subtle issues between 'long' vs 'long long' on OSX
template <typename T0, typename T1>
struct is_same {
    static constexpr bool value =
        sizeof(T0) == sizeof(T1) &&
        std::is_floating_point<T0>::value == std::is_floating_point<T1>::value &&
        std::is_signed<T0>::value == std::is_signed<T1>::value &&
        std::is_arithmetic<T0>::value == std::is_arithmetic<T1>::value;
};

struct KMaskBit;

NAMESPACE_END(detail)

//! @}
// -----------------------------------------------------------------------

/// Array type
template <typename Type_,
          size_t Size_ = (ENOKI_MAX_PACKET_SIZE / sizeof(Type_) > 1)
                        ? ENOKI_MAX_PACKET_SIZE / sizeof(Type_) : 1,
          bool Approx_ = detail::approx_default<Type_>::value,
          RoundingMode Mode_ = RoundingMode::Default>
struct Array;

/// Replace the base scalar type of an array (potentially nested)
template <typename T, typename Scalar, typename = void>
struct like { };

template <typename T, typename Scalar>
struct like<T, Scalar, std::enable_if_t<is_sarray<std::decay_t<T>>::value>> {
    using type = Array<typename like<typename std::decay_t<T>::Scalar, Scalar>::type,
                       std::decay_t<T>::Size>;
};

template <typename T, typename Scalar>
struct like<T, Scalar, std::enable_if_t<!is_sarray<std::decay_t<T>>::value>> {
    using type = Scalar;
};

template <typename T, typename Scalar> using like_t = typename like<T, Scalar>::type;

/// Reinterpret the binary represesentation of a data type
template<typename T, typename U> ENOKI_INLINE T memcpy_cast(const U &val) {
    static_assert(sizeof(T) == sizeof(U), "memcpy_cast: sizes did not match!");
    T result;
    std::memcpy(&result, &val, sizeof(T));
    return result;
}

/// Implementation details
NAMESPACE_BEGIN(detail)

template <typename T>
using is_int32_t = std::enable_if_t<sizeof(T) == 4 && std::is_integral<T>::value>;

template <typename T>
using is_int64_t = std::enable_if_t<sizeof(T) == 8 && std::is_integral<T>::value>;

template <typename Scalar, size_t Size, typename = void> struct is_native {
    static constexpr bool value = false;
};

/// Determines when the special fallback in array_round.h is needed
template <typename Scalar, size_t Size, RoundingMode Mode, typename = void>
struct rounding_fallback : std::true_type { };

template <typename Scalar, size_t Size>
struct rounding_fallback<Scalar, Size, RoundingMode::Default, void>
    : std::false_type { };

#if defined(__AVX512F__)

/* Custom rounding modes can also be realized using AVX512 instructions, but
   this requires the array size to be an exact multiple of what's supported by
   the underlying hardware */

template <size_t Size, RoundingMode Mode>
struct rounding_fallback<
    float, Size, Mode,
    std::enable_if_t<Size / 16 * 16 == Size && Mode != RoundingMode::Default>>
    : std::false_type {};

template <size_t Size, RoundingMode Mode>
struct rounding_fallback<
    double, Size, Mode,
    std::enable_if_t<Size / 8 * 8 == Size && Mode != RoundingMode::Default>>
    : std::false_type {};

#endif

/// Compute binary OR of 'i' with right-shifted versions
static constexpr size_t fill(size_t i) {
    return i != 0 ? i | fill(i >> 1) : 0;
}

/// Find the largest power of two smaller than 'i'
static constexpr size_t lpow2(size_t i) {
    return i != 0 ? (fill(i-1) >> 1) + 1 : 0;
}

template <typename Scalar, size_t Size, RoundingMode Mode> struct is_recursive {
    /// Use the recursive array in array_recursive.h?
    static constexpr bool value = !is_native<Scalar, Size>::value &&
                                  has_vectorization && Size > 3 &&
                                  (std::is_same<Scalar, float>::value ||
                                   std::is_same<Scalar, double>::value ||
                                   (std::is_integral<Scalar>::value &&
                                    (sizeof(Scalar) == 4 || sizeof(Scalar) == 8))) &&
                                  !rounding_fallback<Scalar, Size, Mode>::value;
};

struct reinterpret_flag { };

template <bool...> struct bools { };

/// C++14 substitute for std::conjunction
template <bool... value> using all_of = std::is_same<
    bools<value..., true>, bools<true, value...>>;

/// Convenience class to choose an arithmetic type based on its size and flavor
template <size_t Size> struct type_chooser { };

template <> struct type_chooser<1> {
    using Int = int8_t;
    using UInt = uint8_t;
};

template <> struct type_chooser<2> {
    using Int = int16_t;
    using UInt = uint16_t;
    using Float = half;
};

template <> struct type_chooser<4> {
    using Int = int32_t;
    using UInt = uint32_t;
    using Float = float;
};

template <> struct type_chooser<8> {
    using Int = int64_t;
    using UInt = uint64_t;
    using Float = double;
};

template <typename T> ENOKI_INLINE bool mask_active(const T &value) {
    using T2 = typename detail::type_chooser<sizeof(T)>::UInt;
    return memcpy_cast<T2>(value) != 0;
}

ENOKI_INLINE bool mask_active(const bool &value) {
    return value;
}

NAMESPACE_END(detail)

/// Integer-based version of a given array class
template <typename T>
using int_array_t = like_t<T, typename detail::type_chooser<sizeof(typename T::BaseScalar)>::Int>;

/// Unsigned integer-based version of a given array class
template <typename T>
using uint_array_t = like_t<T, typename detail::type_chooser<sizeof(typename T::BaseScalar)>::UInt>;

/// Floating point-based version of a given array class
template <typename T>
using float_array_t = like_t<T, typename detail::type_chooser<sizeof(typename T::BaseScalar)>::Float>;

template <typename T> using int32_array_t   = like_t<T, int32_t>;
template <typename T> using uint32_array_t  = like_t<T, uint32_t>;
template <typename T> using int64_array_t   = like_t<T, int64_t>;
template <typename T> using uint64_array_t  = like_t<T, uint64_t>;
template <typename T> using float16_array_t = like_t<T, half>;
template <typename T> using float32_array_t = like_t<T, float>;
template <typename T> using float64_array_t = like_t<T, double>;

/// Generic string conversion routine
template <typename T> inline std::string to_string(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

/// Fast implementation for computing the base 2 log of an integer.
template <typename T> ENOKI_INLINE T log2i(T value) {
    assert(value >= 0);
#if defined(__GNUC__) && defined(__x86_64__)
    return T(sizeof(T) * 8 - 1) - (sizeof(T) <= 4 ? T(__builtin_clz((unsigned int) value))
                                                  : T(__builtin_clzll((unsigned long long) value)));
#elif defined(_WIN32)
    unsigned long result;
    if (sizeof(T) <= 4)
        _BitScanReverse(&result, static_cast<unsigned long>(value));
    else
        _BitScanReverse64(&result, static_cast<unsigned long long>(value));
    return T(result);
#else
    T r = 0;
    while ((value >> r) != 0)
        r++;
    return r - 1;
#endif
}

// -----------------------------------------------------------------------
//! @{ \name Fallbacks for high 32/64 bit integer multiplication
// -----------------------------------------------------------------------

ENOKI_INLINE int32_t mulhi(int32_t x, int32_t y) {
    int64_t rl = (int64_t) x * (int64_t) y;
    return (int32_t) (rl >> 32);
}

ENOKI_INLINE uint32_t mulhi(uint32_t x, uint32_t y) {
    uint64_t rl = (uint64_t) x * (uint64_t) y;
    return (uint32_t) (rl >> 32);
}

ENOKI_INLINE uint64_t mulhi(uint64_t x, uint64_t y) {
#if defined(_MSC_VER) && defined(_M_X64)
    return __umulh(x, y);
#elif defined(__SIZEOF_INT128__)
    __uint128_t rl = (__uint128_t) x * (__uint128_t) y;
    return (uint64_t)(rl >> 64);
#else
    // full 128 bits are x0 * y0 + (x0 * y1 << 32) + (x1 * y0 << 32) + (x1 * y1 << 64)
    const uint32_t mask = 0xFFFFFFFF;
    const uint32_t x0 = (uint32_t)(x & mask), x1 = (uint32_t)(x >> 32);
    const uint32_t y0 = (uint32_t)(y & mask), y1 = (uint32_t)(y >> 32);
    const uint32_t x0y0_hi = mulhi(x0, y0);
    const uint64_t x0y1 = x0 * (uint64_t)y1;
    const uint64_t x1y0 = x1 * (uint64_t)y0;
    const uint64_t x1y1 = x1 * (uint64_t)y1;
    const uint64_t temp = x1y0 + x0y0_hi;
    const uint64_t temp_lo = temp & mask, temp_hi = temp >> 32;

    return x1y1 + temp_hi + ((temp_lo + x0y1) >> 32);
#endif
}

ENOKI_INLINE int64_t mulhi(int64_t x, int64_t y) {
#if defined(_MSC_VER) && defined(_M_X64)
    return __mulh(x, y);
#elif defined(__SIZEOF_INT128__)
    __int128_t rl = (__int128_t) x * (__int128_t) y;
    return (int64_t)(rl >> 64);
#else
    // full 128 bits are x0 * y0 + (x0 * y1 << 32) + (x1 * y0 << 32) + (x1 * y1 << 64)
    const uint32_t mask = 0xFFFFFFFF;
    const uint32_t x0 = (uint32_t)(x & mask), y0 = (uint32_t)(y & mask);
    const int32_t x1 = (int32_t)(x >> 32), y1 = (int32_t)(y >> 32);
    const uint32_t x0y0_hi = mulhi(x0, y0);
    const int64_t t = x1 * (int64_t) y0 + x0y0_hi;
    const int64_t w1 = x0 * (int64_t) y1 + (t & mask);

    return x1 * (int64_t) y1 + (t >> 32) + (w1 >> 32);
#endif
}

//! @}
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)

// -----------------------------------------------------------------------
//! @{ \name Bitwise arithmetic involving floating point values
// -----------------------------------------------------------------------

template <typename T, std::enable_if_t<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, int> = 0>
ENOKI_INLINE T or_(T a, bool b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) | (b ? memcpy_cast<Int>(Int(-1))
                                                   : memcpy_cast<Int>(Int(0))));
}

template <typename T, std::enable_if_t<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, int> = 0>
ENOKI_INLINE T and_(T a, bool b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) & (b ? memcpy_cast<Int>(Int(-1))
                                                   : memcpy_cast<Int>(Int(0))));
}

template <typename T, std::enable_if_t<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, int> = 0>
ENOKI_INLINE T xor_(T a, bool b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) ^ (b ? memcpy_cast<Int>(Int(-1))
                                                   : memcpy_cast<Int>(Int(0))));
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T not_(T a) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(~memcpy_cast<Int>(a));
}

ENOKI_INLINE bool not_(bool a) { return !a; }

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T or_(T a, T b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) | memcpy_cast<Int>(b));
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T and_(T a, T b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) & memcpy_cast<Int>(b));
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T xor_(T a, T b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) ^ memcpy_cast<Int>(b));
}

template <typename T1, typename T2, std::enable_if_t<!std::is_floating_point<T1>::value, int> = 0>
ENOKI_INLINE auto or_ (const T1 &a, const T2 &b) { return a | b; }

template <typename T1, typename T2, std::enable_if_t<!std::is_floating_point<T1>::value, int> = 0>
ENOKI_INLINE auto and_(const T1 &a, const T2 &b) { return a & b; }

template <typename T1, typename T2, std::enable_if_t<!std::is_floating_point<T1>::value, int> = 0>
ENOKI_INLINE auto xor_(const T1 &a, const T2 &b) { return a ^ b; }

template <typename T, std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T not_(const T &a) { return ~a; }

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Helper routines to merge smaller arrays into larger ones
// -----------------------------------------------------------------------

#if defined(__AVX__)
ENOKI_INLINE __m256 concat(__m128 l, __m128 h) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(l), h, 1);
}

ENOKI_INLINE __m256d concat(__m128d l, __m128d h) {
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(l), h, 1);
}

ENOKI_INLINE __m256i concat(__m128i l, __m128i h) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(l), h, 1);
}
#endif

#if defined(__AVX512F__)
ENOKI_INLINE __m512 concat(__m256 l, __m256 h) {
    #if defined(__AVX512DQ__)
        return _mm512_insertf32x8(_mm512_castps256_ps512(l), h, 1);
    #else
        return _mm512_castpd_ps(
            _mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(l)),
                               _mm256_castps_pd(h), 1));
    #endif
}

ENOKI_INLINE __m512d concat(__m256d l, __m256d h) {
    return _mm512_insertf64x4(_mm512_castpd256_pd512(l), h, 1);
}

ENOKI_INLINE __m512i concat(__m256i l, __m256i h) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(l), h, 1);
}
#endif

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Mask conversion routines for various platforms
// -----------------------------------------------------------------------

#if defined(__AVX__)
ENOKI_INLINE __m256i mm256_cvtepi32_epi64(__m128i x) {
#if defined(__AVX2__)
    return _mm256_cvtepi32_epi64(x);
#else
    /* This version is only suitable for mask conversions */
    __m128i xl = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 1, 0, 0));
    __m128i xh = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 2, 2));
    return detail::concat(xl, xh);
#endif
}

ENOKI_INLINE __m128i mm256_cvtepi64_epi32(__m256i x) {
#if defined(__AVX512VL__)
    return _mm256_cvtepi64_epi32(x);
#else
    __m128i x0 = _mm256_castsi256_si128(x);
    __m128i x1 = _mm256_extractf128_si256(x, 1);
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x0), _mm_castsi128_ps(x1), _MM_SHUFFLE(2, 0, 2, 0)));
#endif
}

ENOKI_INLINE __m256i mm512_cvtepi64_epi32(__m128i x0, __m128i x1, __m128i x2, __m128i x3) {
    __m128i y0 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x0), _mm_castsi128_ps(x1), _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i y1 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x2), _mm_castsi128_ps(x3), _MM_SHUFFLE(2, 0, 2, 0)));
    return detail::concat(y0, y1);
}

ENOKI_INLINE __m256i mm512_cvtepi64_epi32(__m256i x0, __m256i x1) {
    __m128i y0 = _mm256_castsi256_si128(x0);
    __m128i y1 = _mm256_extractf128_si256(x0, 1);
    __m128i y2 = _mm256_castsi256_si128(x1);
    __m128i y3 = _mm256_extractf128_si256(x1, 1);
    return mm512_cvtepi64_epi32(y0, y1, y2, y3);
}
#endif

ENOKI_INLINE __m128i mm256_cvtepi64_epi32(__m128i x0, __m128i x1) {
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x0), _mm_castsi128_ps(x1), _MM_SHUFFLE(2, 0, 2, 0)));
}

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(detail)

NAMESPACE_END(enoki)
