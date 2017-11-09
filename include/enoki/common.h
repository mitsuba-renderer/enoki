/*
    enoki/common.h -- Common definitions and template helpers

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "fwd.h"

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

#if defined(ENOKI_X86_64) || defined(ENOKI_X86_32)
#  include <immintrin.h>
#endif

#if defined(ENOKI_ARM_NEON)
#  include <arm_neon.h>
#endif

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

#include "alloc.h"

NAMESPACE_BEGIN(enoki)

// -----------------------------------------------------------------------
//! @{ \name Available instruction sets
// -----------------------------------------------------------------------

#if defined(ENOKI_X86_AVX512F)
    static constexpr bool has_avx512f = true;
#else
    static constexpr bool has_avx512f = false;
#endif

#if defined(ENOKI_X86_AVX512CD)
    static constexpr bool has_avx512cd = true;
#else
    static constexpr bool has_avx512cd = false;
#endif

#if defined(ENOKI_X86_AVX512DQ)
    static constexpr bool has_avx512dq = true;
#else
    static constexpr bool has_avx512dq = false;
#endif

#if defined(ENOKI_X86_AVX512VL)
    static constexpr bool has_avx512vl = true;
#else
    static constexpr bool has_avx512vl = false;
#endif

#if defined(ENOKI_X86_AVX512BW)
    static constexpr bool has_avx512bw = true;
#else
    static constexpr bool has_avx512bw = false;
#endif

#if defined(ENOKI_X86_AVX512PF)
    static constexpr bool has_avx512pf = true;
#else
    static constexpr bool has_avx512pf = false;
#endif

#if defined(ENOKI_X86_AVX512ER)
    static constexpr bool has_avx512er = true;
#else
    static constexpr bool has_avx512er = false;
#endif

#if defined(__AVX512VBMI__)
    static constexpr bool has_avx512vbmi = true;
#else
    static constexpr bool has_avx512vbmi = false;
#endif

#if defined(ENOKI_X86_AVX512VPOPCNTDQ)
    static constexpr bool has_avx512vpopcntdq = true;
#else
    static constexpr bool has_avx512vpopcntdq = false;
#endif

#if defined(ENOKI_X86_AVX2)
    static constexpr bool has_avx2 = true;
#else
    static constexpr bool has_avx2 = false;
#endif

#if defined(ENOKI_X86_FMA) || defined(ENOKI_ARM_FMA)
    static constexpr bool has_fma = true;
#else
    static constexpr bool has_fma = false;
#endif

#if defined(ENOKI_X86_F16C)
    static constexpr bool has_f16c = true;
#else
    static constexpr bool has_f16c = false;
#endif

#if defined(ENOKI_X86_AVX)
    static constexpr bool has_avx = true;
#else
    static constexpr bool has_avx = false;
#endif

#if defined(ENOKI_X86_SSE42)
    static constexpr bool has_sse42 = true;
#else
    static constexpr bool has_sse42 = false;
#endif

#if defined(ENOKI_X86_32)
    static constexpr bool has_x86_32 = true;
#else
    static constexpr bool has_x86_32 = false;
#endif

#if defined(ENOKI_X86_64)
    static constexpr bool has_x86_64 = true;
#else
    static constexpr bool has_x86_64 = false;
#endif

#if defined(ENOKI_ARM_NEON)
    static constexpr bool has_neon = true;
#else
    static constexpr bool has_neon = false;
#endif

#if defined(ENOKI_ARM_32)
    static constexpr bool has_arm_32 = true;
#else
    static constexpr bool has_arm_32 = false;
#endif

#if defined(ENOKI_ARM_64)
    static constexpr bool has_arm_64 = true;
#else
    static constexpr bool has_arm_64 = false;
#endif

static constexpr bool has_vectorization = has_sse42 || has_neon;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Forward declarations
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)

template <typename... Args> struct extract_array;
template <typename... Args> struct expr;
template <typename     Arg, typename SFINAE = int> struct value_;
template <typename     Arg, typename SFINAE = int> struct scalar_;
template <typename     Arg, typename SFINAE = int> struct mask_;
template <typename     Arg, typename SFINAE = int> struct array_;
template <typename     Arg, typename SFINAE = int> struct packet_;
template <typename     Arg, typename SFINAE = int> struct array_shape_;

NAMESPACE_END(detail)

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Value traits
// -----------------------------------------------------------------------

/// SFINAE helper to test whether a class is a static or dynamic array type
template <typename T> struct is_array {
private:
    static constexpr std::false_type check(void *);
    template <typename Value, typename Derived>
    static constexpr std::true_type check(ArrayBase<Value, Derived> *);
public:
    using type = decltype(check((std::decay_t<T> *) nullptr));
    static constexpr bool value = type::value;
};

/// SFINAE helper to test whether a class is a static or dynamic mask type
template <typename T> struct is_mask {
private:
    static constexpr std::false_type check(void *);
    template <typename T2 = T>
    static constexpr std::integral_constant<bool, T2::Derived::IsMask> check(T2 *);
public:
    using T2 = std::decay_t<T>;
    using type = decltype(check((T2 *) nullptr));
    static constexpr bool value = type::value || std::is_same<T2, bool>::value;
};

/// SFINAE helper to test whether a class is a static array type
template <typename T> struct is_static_array {
private:
    static constexpr std::false_type check(void *);
    template <typename Value, size_t Size, bool Approx, RoundingMode Mode, typename Derived>
    static constexpr std::true_type check(StaticArrayBase<Value, Size, Approx, Mode, Derived> *);
public:
    using type = decltype(check((std::decay_t<T> *) nullptr));
    static constexpr bool value = type::value;
};

/// SFINAE helper to test whether a class is a dynamic array type
template <typename T> struct is_dynamic_array {
private:
    static constexpr std::false_type check(void *);
    template <typename Value, typename Derived>
    static constexpr std::true_type check(DynamicArrayBase<Value, Derived> *);
public:
    using type = decltype(check((std::decay_t<T> *) nullptr));
    static constexpr bool value = type::value;
};

/// SFINAE helper to test whether a initializing an array with a given type should broadcast if possible
template <typename T> struct broadcast {
private:
    static constexpr std::true_type check(void *);
    template <typename T2 = T>
    static constexpr std::integral_constant<bool, T2::Derived::PrefersBroadcast> check(T2 *);
public:
    using T2 = std::decay_t<T>;
    using type = decltype(check((T2 *) nullptr));
    static constexpr bool value = type::value;
};

template <typename T> using enable_if_array_t         = std::enable_if_t<is_array<T>::value, int>;
template <typename T> using enable_if_not_array_t     = std::enable_if_t<!is_array<T>::value, int>;
template <typename T> using enable_if_mask_t          = std::enable_if_t<is_mask<T>::value, int>;
template <typename T> using enable_if_not_mask_t      = std::enable_if_t<!is_mask<T>::value, int>;
template <typename T> using enable_if_static_array_t  = std::enable_if_t<is_static_array<T>::value, int>;
template <typename T> using enable_if_dynamic_array_t = std::enable_if_t<is_dynamic_array<T>::value, int>;

/// Determine the nesting level of an array
template <typename T, size_t Static = 1, size_t Dynamic = 1, typename = int> struct array_depth {
    static constexpr size_t value = 0;
};

template <typename T, size_t Static, size_t Dynamic> struct array_depth<T, Static, Dynamic, enable_if_static_array_t<T>> {
    static constexpr size_t value = array_depth<typename std::decay_t<T>::Value>::value + Static;
};

template <typename T, size_t Static, size_t Dynamic> struct array_depth<T, Static, Dynamic, enable_if_dynamic_array_t<T>> {
    static constexpr size_t value = array_depth<typename std::decay_t<T>::Value>::value + Dynamic;
};

/// Determine the size of an array
template <typename T, typename = int> struct array_size {
    static constexpr size_t value = 1;
};

template <typename T> struct array_size<T, enable_if_static_array_t<T>> {
    static constexpr size_t value = std::decay_t<T>::Derived::Size;
};

template <typename     Arg> using array_shape_t = typename detail::array_shape_<Arg>::type;
template <typename     Arg> using value_t       = typename detail::value_<Arg>::type;
template <typename     Arg> using scalar_t      = typename detail::scalar_<Arg>::type;
template <typename     Arg> using mask_t        = typename detail::mask_<Arg>::type;
template <typename     Arg> using array_t       = typename detail::array_<Arg>::type;
template <typename     Arg> using packet_t      = typename detail::packet_<Arg>::type;
template <typename... Args> using expr_t        = typename detail::expr<Args...>::type;

NAMESPACE_BEGIN(detail)

template <typename T, size_t>
struct prepend_index { };

template <size_t... Index, size_t Value>
struct prepend_index<std::index_sequence<Index...>, Value> {
    using type = std::index_sequence<Value, Index...>;
};

/// Determine the shape of an array
template <typename T, typename> struct array_shape_ {
    using type = std::index_sequence<>;
};

template <typename T> struct array_shape_<T, enable_if_static_array_t<T>> {
    using Td = std::decay_t<T>;
    using type = typename prepend_index<array_shape_t<typename Td::Value>, Td::Size>::type;
};

/// Value trait to extract the first Enoki array from a list of arguments
template <typename... Args>
using extract_array_t = typename extract_array<Args...>::type;

template <typename Arg, typename... Args>
struct extract_array<Arg, Args...> {
private:
    using T0 = Arg;
    using T1 = extract_array_t<Args...>;

    static constexpr size_t D0 = array_depth<T0, 1, 2>::value;
    static constexpr size_t D1 = array_depth<T1, 1, 2>::value;

public:
    using type = std::conditional_t<(D1 > D0 || D0 == 0), T1, T0>;
};

template <> struct extract_array<> { using type = void; };

/// Value trait to compute the result of a unary expression involving a type T
template <typename Array, typename T> struct expr_1;

template <typename T>
struct expr_1<T, T> {
private:
    using Td        = std::decay_t<T>;
    using Entry     = value_t<T>;
    using EntryExpr = expr_t<Entry>;
public:
    using type = std::conditional_t<
        std::is_same<Entry, EntryExpr>::value,
        Td, typename Td::Derived::template ReplaceType<EntryExpr>
    >;
};

template <typename T>
struct expr_1<void, T> { using type = std::decay_t<T>; };

/// Value trait to compute the result of a n-ary expression involving types (Arg, Args...)
template <typename Array, typename Arg, typename... Args>
struct expr_n {
private:
    using Value = expr_t<packet_t<Arg>, packet_t<Args>...>;
public:
    using type  = typename std::decay_t<Array>::Derived::template ReplaceType<Value>;
};

template <typename Arg, typename... Args>
struct expr_n<void, Arg, Args...> {
    using type = decltype(std::declval<Arg>() + std::declval<expr_t<Args...>>());
};

template <typename T> struct expr_n<void, T*, T*> { using type = T*; };
template <typename T> struct expr_n<void, T*, const T*> { using type = const T*; };
template <typename T> struct expr_n<void, const T*, T*> { using type = const T*; };
template <typename T> struct expr_n<void, T*, std::nullptr_t> { using type = T*; };
template <typename T> struct expr_n<void, std::nullptr_t, T*> { using type = T*; };

/// Value trait to compute the result of arbitrary expressions
template <typename... Args> struct expr      : detail::expr_n<detail::extract_array_t<Args...>, Args...> { };
template <typename Arg>     struct expr<Arg> : detail::expr_1<detail::extract_array_t<Arg>,     Arg>     { };

/// Value trait to access the mask type underlying an array
template <typename T, typename> struct mask_ { using type = bool; };

template <typename T> struct mask_<T, enable_if_array_t<T>> {
    using type = typename std::decay_t<T>::MaskType;
};

/// Value trait to access the array type underlying an array
template <typename T, typename> struct array_ { /* Don't know */ };

template <typename T> struct array_<T, enable_if_array_t<T>> {
    using type = typename std::decay_t<T>::ArrayType;
};

/// Value trait to access the component type of an array
template <typename T, typename> struct value_ { using type = T; };

template <typename T> struct value_<T, enable_if_array_t<T>> {
    using type = typename std::decay_t<T>::Value;
};

/// Value trait to access the base scalar type underlying a potentially nested array
template <typename T, typename> struct scalar_ { using type = std::decay_t<T>; };

template <typename T> struct scalar_<T, enable_if_array_t<T>> {
    using type = scalar_t<typename std::decay_t<T>::Value>;
};

/// Value trait to access the packet/component type of an array
template <typename T, typename> struct packet_ { using type = T; };

template <typename T> struct packet_<T, std::enable_if_t<is_array<T>::value && !is_dynamic_array<T>::value, int>> {
    using type = typename std::decay_t<T>::Value;
};

template <typename T> struct packet_<T, std::enable_if_t<is_array<T>::value && is_dynamic_array<T>::value, int>> {
    using type = typename std::decay_t<T>::Packet;
};

template <typename S, typename T> struct copy_flags {
private:
    using R = std::remove_reference_t<S>;
    using T1 = std::conditional_t<std::is_const<R>::value, std::add_const_t<T>, T>;
    using T2 = std::conditional_t<std::is_pointer<S>::value,
                                  std::add_pointer_t<T1>, T1>;
    using T3 = std::conditional_t<std::is_lvalue_reference<S>::value,
                                  std::add_lvalue_reference_t<T2>, T2>;
    using T4 = std::conditional_t<std::is_rvalue_reference<S>::value,
                                  std::add_rvalue_reference_t<T3>, T3>;

public:
    using type = T4;
};

template <typename S, typename T> using copy_flags_t = typename detail::copy_flags<S, T>::type;

template <typename Input, typename Output>
using ref_cast_t = std::conditional_t<std::is_same<Input, Output>::value,
                                      const Input &, Output>;

template <typename T> struct approx_default<T, enable_if_array_t<T>> {
    static constexpr bool value = std::decay_t<T>::Approx;
};

/// Value equivalence between arithmetic type to work around subtle issues between 'long' vs 'long long' on OSX
template <typename T0, typename T1>
struct is_same {
    static constexpr bool value =
        sizeof(T0) == sizeof(T1) &&
        std::is_floating_point<T0>::value == std::is_floating_point<T1>::value &&
        std::is_signed<T0>::value == std::is_signed<T1>::value &&
        std::is_arithmetic<T0>::value == std::is_arithmetic<T1>::value;
};

NAMESPACE_END(detail)

//! @}
// -----------------------------------------------------------------------

/// Replace the base scalar type of a (potentially nested) array
template <typename T, typename Value, typename = int>
struct like { };

template <typename T, typename Value> using like_t = typename like<T, Value>::type;

template <typename T, typename Value> struct like<T, Value, enable_if_not_array_t<T>> {
    using type = detail::copy_flags_t<T, Value>;
};

template <typename T, typename Value> struct like<T, Value, enable_if_array_t<T>> {
private:
    using Td = typename std::decay_t<T>::Derived;
    using Entry = like_t<packet_t<Td>, Value>;
public:
    using type = detail::copy_flags_t<T, typename Td::Derived::template ReplaceType<Entry>>;
};

/// Replace the base scalar type of a (potentially nested) array -- does not copy any flags
template <typename T, typename Value, typename = int>
struct replace { };

template <typename T, typename Value> using replace_t = typename replace<T, Value>::type;

template <typename T, typename Value> struct replace<T, Value, enable_if_not_array_t<T>> {
    using type = Value;
};

template <typename T, typename Value> struct replace<T, Value, enable_if_array_t<T>> {
private:
    using Td = typename std::decay_t<T>::Derived;
    using Entry = replace_t<packet_t<Td>, Value>;
public:
    using type = typename Td::Derived::template ReplaceType<Entry>;
};

/// Reinterpret the binary represesentation of a data type
template<typename T, typename U> ENOKI_INLINE T memcpy_cast(const U &val) {
    static_assert(sizeof(T) == sizeof(U), "memcpy_cast: sizes did not match!");
    T result;
    std::memcpy(&result, &val, sizeof(T));
    return result;
}

/// Implementation details
NAMESPACE_BEGIN(detail)

/// Compile-time integer logarithm
constexpr size_t clog2i(size_t value) {
    return (value > 1) ? 1 + clog2i(value >> 1) : 0;
}

template <typename T>
using is_int32_t = std::enable_if_t<sizeof(T) == 4 && std::is_integral<T>::value>;

template <typename T>
using is_int64_t = std::enable_if_t<sizeof(T) == 8 && std::is_integral<T>::value>;

template <typename Value, size_t Size, typename = void> struct is_native {
    static constexpr bool value = false;
};

/// Determines when the special fallback in array_round.h is needed
template <typename Value, size_t Size, RoundingMode Mode, typename = void>
struct rounding_fallback : std::true_type { };

template <typename Value, size_t Size>
struct rounding_fallback<Value, Size, RoundingMode::Default, void>
    : std::false_type { };

#if defined(ENOKI_X86_AVX512F)

/* Custom rounding modes can also be realized using AVX512 instructions, but
   this requires the array size to be an exact multiple of what's supported by
   the underlying hardware */

template <size_t Size, RoundingMode Mode>
struct rounding_fallback<
    float, Size, Mode,
    std::enable_if_t<Size / 16 * 16 == Size && Mode != RoundingMode::Default>>
    : std::false_type { };

template <size_t Size, RoundingMode Mode>
struct rounding_fallback<
    double, Size, Mode,
    std::enable_if_t<Size / 8 * 8 == Size && Mode != RoundingMode::Default>>
    : std::false_type { };

#endif

/// Compute binary OR of 'i' with right-shifted versions
static constexpr size_t fill(size_t i) {
    return i != 0 ? i | fill(i >> 1) : 0;
}

/// Find the largest power of two smaller than 'i'
static constexpr size_t lpow2(size_t i) {
    return i != 0 ? (fill(i-1) >> 1) + 1 : 0;
}

template <typename Value, size_t Size, RoundingMode Mode> struct is_recursive {
    /// Use the recursive array in array_recursive.h?
    static constexpr bool value = !is_native<Value, Size>::value &&
                                  has_vectorization && Size > 3 &&
                                  (std::is_same<Value, float>::value ||
                                   std::is_same<Value, double>::value ||
                                   (std::is_integral<Value>::value &&
                                    (sizeof(Value) == 4 || sizeof(Value) == 8))) &&
                                  !rounding_fallback<Value, Size, Mode>::value;
};

struct reinterpret_flag { };

template <bool...> struct bools { };

/// C++14 substitute for std::conjunction
template <bool... value>
using all_of = std::is_same<bools<value..., true>, bools<true, value...>>;

template <bool... value>
using any_of = std::integral_constant<bool, !all_of<!value...>::value>;

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

template <typename T, typename UInt = typename detail::type_chooser<sizeof(T)>::UInt>
ENOKI_INLINE bool mask_active(const T &value) {
    return memcpy_cast<UInt>(value) != 0;
}

ENOKI_INLINE bool mask_active(const bool &value) {
    return value;
}

/* Extract the nth entry of a parameter pack */
template <typename T>
struct tuple_tail { using type = std::nullptr_t; };

template <typename Arg, typename... Args>
struct tuple_tail<std::tuple<Arg, Args...>> {
    using type =
        std::tuple_element_t<sizeof...(Args), std::tuple<Arg, Args...>>;
};

template <typename T> using tuple_tail_t = typename tuple_tail<T>::type;

/* Type trait to extract a boolean constant from a type (or return false) */
template<typename... Ts> struct make_void { using type = void; };
template<typename... Ts> using void_t = typename make_void<Ts...>::type;
template <typename T, template<class> class Tester, typename = void> struct get_flag : std::false_type { };
template <typename T, template<class> class Tester> struct get_flag<T, Tester, void_t<Tester<T>>> : Tester<T> { };

NAMESPACE_END(detail)

template <typename Target, typename Source> struct is_reinterpretable {
    static constexpr bool value = std::is_same<Target, Source>::value ||
                                  std::is_constructible<Target, const Source &, detail::reinterpret_flag>::value ||
                                  (std::is_scalar<Source>::value && std::is_scalar<Target>::value);
};

/// Integer-based version of a given array class
template <typename T>
using int_array_t = like_t<T, typename detail::type_chooser<sizeof(scalar_t<T>)>::Int>;

/// Unsigned integer-based version of a given array class
template <typename T>
using uint_array_t = like_t<T, typename detail::type_chooser<sizeof(scalar_t<T>)>::UInt>;

/// Floating point-based version of a given array class
template <typename T>
using float_array_t = like_t<T, typename detail::type_chooser<sizeof(scalar_t<T>)>::Float>;

template <typename T> using int32_array_t   = like_t<T, int32_t>;
template <typename T> using uint32_array_t  = like_t<T, uint32_t>;
template <typename T> using int64_array_t   = like_t<T, int64_t>;
template <typename T> using uint64_array_t  = like_t<T, uint64_t>;
template <typename T> using float16_array_t = like_t<T, half>;
template <typename T> using float32_array_t = like_t<T, float>;
template <typename T> using float64_array_t = like_t<T, double>;
template <typename T> using bool_array_t    = like_t<T, bool>;
template <typename T> using size_array_t    = like_t<T, size_t>;
template <typename T> using ssize_array_t   = like_t<T, ssize_t>;

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
#if defined(_MSC_VER) && defined(ENOKI_X86_64)
    return __umulh(x, y);
#elif defined(__SIZEOF_INT128__)
    __uint128_t rl = (__uint128_t) x * (__uint128_t) y;
    return (uint64_t)(rl >> 64);
#else
    // full 128 bits are x0 * y0 + (x0 * y1 << 32) + (x1 * y0 << 32) + (x1 * y1 << 64)
    const uint32_t mask = 0xFFFFFFFF;
    const uint32_t x0 = (uint32_t) (x & mask), x1 = (uint32_t) (x >> 32);
    const uint32_t y0 = (uint32_t) (y & mask), y1 = (uint32_t) (y >> 32);
    const uint32_t x0y0_hi = mulhi(x0, y0);
    const uint64_t x0y1 = x0 * (uint64_t) y1;
    const uint64_t x1y0 = x1 * (uint64_t) y0;
    const uint64_t x1y1 = x1 * (uint64_t) y1;
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
    const uint32_t x0 = (uint32_t) (x & mask), y0 = (uint32_t) (y & mask);
    const int32_t x1 = (int32_t) (x >> 32), y1 = (int32_t) (y >> 32);
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

template <typename, typename, typename = void>
struct supports_bit_op : std::false_type { };

template <typename T1, typename T2>
struct supports_bit_op<T1, T2, void_t<decltype(std::declval<T1>() | std::declval<T2>())>>
    : std::true_type { };


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
ENOKI_INLINE T andnot_(T a, bool b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) & (b ? memcpy_cast<Int>(Int(0))
                                                   : memcpy_cast<Int>(Int(-1))));
}

template <typename T, std::enable_if_t<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, int> = 0>
ENOKI_INLINE T xor_(T a, bool b) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(memcpy_cast<Int>(a) ^ (b ? memcpy_cast<Int>(Int(-1))
                                                   : memcpy_cast<Int>(Int(0))));
}

ENOKI_INLINE bool not_(bool a) { return !a; }
ENOKI_INLINE bool andnot_(bool a, bool b) { return a & !b; }

template <typename T1, typename T2, std::enable_if_t<supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE auto or_(const T1 &a, const T2 &b) { return a | b; }

template <typename T1, typename T2, std::enable_if_t<supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE auto and_(const T1 &a, const T2 &b) { return a & b; }

template <typename T1, typename T2, std::enable_if_t<supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE auto andnot_(const T1 &a, const T2 &b) { return a & ~b; }

template <typename T1, typename T2, std::enable_if_t<supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE auto xor_(const T1 &a, const T2 &b) { return a ^ b; }

template <typename T, std::enable_if_t<supports_bit_op<T, T>::value, int> = 0>
ENOKI_INLINE auto not_(const T &a) { return ~a; }

template <typename T1, typename T2, std::enable_if_t<!supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE T1 or_(const T1 &a, const T2 &b) {
    using Int = typename type_chooser<sizeof(T1)>::Int;
    return memcpy_cast<T1>(memcpy_cast<Int>(a) | memcpy_cast<Int>(b));
}

template <typename T1, typename T2, std::enable_if_t<!supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE T1 and_(const T1 &a, const T2 &b) {
    using Int = typename type_chooser<sizeof(T1)>::Int;
    return memcpy_cast<T1>(memcpy_cast<Int>(a) & memcpy_cast<Int>(b));
}

template <typename T1, typename T2, std::enable_if_t<!supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE T1 andnot_(const T1 &a, const T2 &b) {
    using Int = typename type_chooser<sizeof(T1)>::Int;
    return memcpy_cast<T1>(memcpy_cast<Int>(a) & ~memcpy_cast<Int>(b));
}

template <typename T1, typename T2, std::enable_if_t<!supports_bit_op<T1, T2>::value, int> = 0>
ENOKI_INLINE T1 xor_(const T1 &a, const T2 &b) {
    using Int = typename type_chooser<sizeof(T1)>::Int;
    return memcpy_cast<T1>(memcpy_cast<Int>(a) ^ memcpy_cast<Int>(b));
}

template <typename T, std::enable_if_t<!supports_bit_op<T, T>::value, int> = 0>
ENOKI_INLINE T not_(const T &a) {
    using Int = typename type_chooser<sizeof(T)>::Int;
    return memcpy_cast<T>(~memcpy_cast<Int>(a));
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Helper routines to merge smaller arrays into larger ones
// -----------------------------------------------------------------------

#if defined(ENOKI_X86_AVX)
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

#if defined(ENOKI_X86_AVX512F)
ENOKI_INLINE __m512 concat(__m256 l, __m256 h) {
    #if defined(ENOKI_X86_AVX512DQ)
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

#if defined(ENOKI_X86_AVX)
ENOKI_INLINE __m256i mm256_cvtepi32_epi64(__m128i x) {
#if defined(ENOKI_X86_AVX2)
    return _mm256_cvtepi32_epi64(x);
#else
    /* This version is only suitable for mask conversions */
    __m128i xl = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 1, 0, 0));
    __m128i xh = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 2, 2));
    return detail::concat(xl, xh);
#endif
}

ENOKI_INLINE __m128i mm256_cvtepi64_epi32(__m256i x) {
#if defined(ENOKI_X86_AVX512VL)
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

#if defined(ENOKI_X86_SSE42)

ENOKI_INLINE __m128i mm256_cvtepi64_epi32(__m128i x0, __m128i x1) {
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x0), _mm_castsi128_ps(x1), _MM_SHUFFLE(2, 0, 2, 0)));
}

ENOKI_INLINE __m128i mm_cvtsi64_si128(long long a)  {
    #if defined(ENOKI_X86_64)
        return _mm_cvtsi64_si128(a);
    #else
        alignas(16) long long x[2] = { a, 0ll };
        return _mm_load_si128((__m128i *) x);
    #endif
}

ENOKI_INLINE long long mm_cvtsi128_si64(__m128i m)  {
    #if defined(ENOKI_X86_64)
        return _mm_cvtsi128_si64(m);
    #else
        alignas(16) long long x[2];
        _mm_store_si128((__m128i *) x, m);
        return x[0];
    #endif
}

template <int Imm8>
ENOKI_INLINE long long mm_extract_epi64(__m128i m)  {
    #if defined(ENOKI_X86_64)
        return _mm_extract_epi64(m, Imm8);
    #else
        alignas(16) long long x[2];
        _mm_store_si128((__m128i *) x, m);
        return x[Imm8];
    #endif
}

#endif

//! @}
// -----------------------------------------------------------------------

template <typename Array>
ENOKI_INLINE Array *alloca_helper(uint8_t *ptr, size_t size, bool clear) {
    (uintptr_t &) ptr +=
        ((max_packet_size - (uintptr_t) ptr) % max_packet_size);
    if (clear)
        memset(ptr, 0, size);
    return (Array *) ptr;
}

NAMESPACE_END(detail)

template <typename Packet, typename Storage> struct call_support {
    call_support(const Storage &) { }
};


// -----------------------------------------------------------------------
//! @{ \name Memory allocation (stack)
// -----------------------------------------------------------------------

/**
 * \brief Wrapper around alloca(), which returns aligned (and potentially
 * zero-initialized) memory
 */
#define ENOKI_ALIGNED_ALLOCA(Array, Count, Clear)                             \
    enoki::detail::alloca_helper<Array>((uint8_t *) alloca(                   \
        sizeof(Array) * (Count) + enoki::max_packet_size - 4),                \
        sizeof(Array) * (Count), Clear)

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
