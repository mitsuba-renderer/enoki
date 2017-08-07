/*
    enoki/array_router.h -- Helper functions which route function calls
    in the enoki namespace to the intended recipients

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <iostream>

NAMESPACE_BEGIN(enoki)

#define ENOKI_ROUTE_UNARY(name, func)                                          \
    /* Case 1: use array-specific implementation of operation */               \
    template <typename T,                                                      \
              std::enable_if_t<is_array<T>::value &&                           \
                               std::is_same<expr_t<T>, T>::value, int> = 0>    \
    ENOKI_INLINE auto name(const T &a) {                                       \
        return a.derived().func##_();                                          \
    }                                                                          \
    /* Case 2: broadcast/evaluate input arrays and try again */                \
    template <typename T,                                                      \
              std::enable_if_t<is_array<T>::value &&                           \
                               !std::is_same<expr_t<T>, T>::value, int> = 0>   \
    ENOKI_INLINE auto name(const T &a) {                                       \
        return name((expr_t<T>) a);                                            \
    }

#define ENOKI_ROUTE_UNARY_SCALAR(name, func, expr)                             \
    /* Case 3: scalar input */                                                 \
    template <typename T, enable_if_not_array_t<T> = 0>                        \
    ENOKI_INLINE auto name(const T &a) {                                       \
        return expr;                                                           \
    }                                                                          \
    ENOKI_ROUTE_UNARY(name, func)


#define ENOKI_ROUTE_UNARY_IMM(name, func)                                      \
    /* Case 1: use array-specific implementation of operation */               \
    template <size_t Imm, typename T,                                          \
              std::enable_if_t<is_array<T>::value &&                           \
                               std::is_same<expr_t<T>, T>::value, int> = 0>    \
    ENOKI_INLINE auto name(const T &a) {                                       \
        return a.template func##_<Imm>();                                      \
    }                                                                          \
    /* Case 2: broadcast/evaluate input arrays and try again */                \
    template <size_t Imm, typename T,                                          \
              std::enable_if_t<is_array<T>::value &&                           \
                               !std::is_same<expr_t<T>, T>::value, int> = 0>   \
    ENOKI_INLINE auto name(const T &a) {                                       \
        return name<Imm>((expr_t<T>) a);                                       \
    }

#define ENOKI_ROUTE_UNARY_SCALAR_IMM(name, func, expr)                         \
    /* Case 3: scalar input */                                                 \
    template <size_t Imm, typename T, enable_if_not_array_t<T> = 0>            \
    ENOKI_INLINE auto name(const T &a) {                                       \
        return expr;                                                           \
    }                                                                          \
    ENOKI_ROUTE_UNARY_IMM(name, func)

#define ENOKI_ROUTE_BINARY(name, func)                                         \
    /* Case 1: use array-specific implementation of operation */               \
    template <typename T,                                                      \
              std::enable_if_t<is_array<T>::value &&                           \
                               std::is_same<expr_t<T>, T>::value, int> = 0>    \
    ENOKI_INLINE auto name(const T &a1, const T &a2) {                         \
        return a1.derived().func##_(a2.derived());                             \
    }                                                                          \
    /* Case 2: broadcast/evaluate input arrays and try again */                \
    template <typename T1, typename T2,                                        \
              typename Array = detail::extract_array_t<T1, T2>,                \
              std::enable_if_t<!std::is_void<Array>::value, int> = 0>          \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using Output = expr_t<T1, T2>;                                         \
        return name((detail::ref_cast_t<T1, Output>) a1,                       \
                    (detail::ref_cast_t<T2, Output>) a2);                      \
    }

#define ENOKI_ROUTE_BINARY_SCALAR(name, func, expr)                            \
    /* Case 3: scalar input */                                                 \
    template <typename T1, typename T2,                                        \
              typename Void = detail::extract_array_t<T1, T2>,                 \
              std::enable_if_t<std::is_void<Void>::value, int> = 0>            \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) -> decltype(expr) {     \
        return expr;                                                           \
    }                                                                          \
    ENOKI_ROUTE_BINARY(name, func)

#define ENOKI_ROUTE_BINARY_BIT(name, func)                                     \
    /* Case 1: use array-specific implementation of operation */               \
    template <typename T,                                                      \
              std::enable_if_t<is_array<T>::value &&                           \
                               std::is_same<expr_t<T>, T>::value, int> = 0>    \
    ENOKI_INLINE auto name(const T &a1, const T &a2) {                         \
        return a1.derived().func##_(a2.derived());                             \
    }                                                                          \
    /* Case 2: broadcast/evaluate input arrays and try again */                \
    template <typename T1, typename T2,                                        \
              typename Array = detail::extract_array_t<T1, T2>,                \
              std::enable_if_t<!std::is_void<Array>::value &&                  \
                               !is_mask<T2>::value, int> = 0>                  \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using Output = expr_t<T1, T2>;                                         \
        return name((detail::ref_cast_t<T1, Output>) a1,                       \
                    (detail::ref_cast_t<T2, Output>) a2);                      \
    }                                                                          \
    /* Case 2: broadcast/evaluate input arrays and try again */                \
    template <typename T1, typename T2,                                        \
              std::enable_if_t<is_array<T1>::value &&                          \
                               is_mask<T2>::value, int> = 0>                   \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        return a1.derived().func##_(mask_t<expr_t<T1>>(a2));                   \
    }

#define ENOKI_ROUTE_SHIFT(name, func)                                          \
    /* Case 1: use array-specific implementation of operation */               \
    template <typename T,                                                      \
              std::enable_if_t<is_array<T>::value &&                           \
                               std::is_same<expr_t<T>, T>::value, int> = 0>    \
    ENOKI_INLINE auto name(const T &a1, const T &a2) {                         \
        return a1.derived().func##_(a2.derived());                             \
    }                                                                          \
    /* Case 2: broadcast/evaluate input arrays and try again */                \
    template <typename T1, typename T2,                                        \
              std::enable_if_t<is_array<T1>::value, int> = 0>                  \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using Output = expr_t<T1, T2>;                                         \
        return name((detail::ref_cast_t<T1, Output>) a1,                       \
                    (detail::ref_cast_t<T2, Output>) a2);                      \
    }

#define ENOKI_ROUTE_TERNARY(name, func)                                        \
    /* Case 1: use array-specific implementation of operation */               \
    template <typename T,                                                      \
              std::enable_if_t<is_array<T>::value &&                           \
                               std::is_same<expr_t<T>, T>::value, int> = 0>    \
    ENOKI_INLINE auto name(const T &a1, const T &a2, const T &a3) {            \
        return a1.derived().func##_(a2.derived(), a3.derived());               \
    }                                                                          \
    /* Case 2: broadcast/evaluate input arrays and try again */                \
    template <typename T1, typename T2, typename T3,                           \
              typename Array = detail::extract_array_t<T1, T2, T3>,            \
              std::enable_if_t<!std::is_void<Array>::value, int> = 0>          \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2, const T3 &a3) {         \
        using Output = expr_t<T1, T2, T3>;                                     \
        return name((detail::ref_cast_t<T1, Output>) a1,                       \
                    (detail::ref_cast_t<T2, Output>) a2,                       \
                    (detail::ref_cast_t<T3, Output>) a3);                      \
    }

/// Macro for compound assignment operators (operator+=, etc.)
#define ENOKI_ROUTE_COMPOUND_OPERATOR(op)                                      \
    template <typename T1, enable_if_array_t<T1> = 0, typename T2>             \
    ENOKI_INLINE T1 &operator op##=(T1 &a1, const T2 &a2) {                    \
        a1 = a1 op a2;                                                         \
        return a1;                                                             \
    }

// -----------------------------------------------------------------------
//! @{ \name Vertical and horizontal operations
// -----------------------------------------------------------------------


ENOKI_ROUTE_UNARY(operator-, neg)
ENOKI_ROUTE_UNARY(operator~, not)
ENOKI_ROUTE_UNARY(operator!, not)

ENOKI_ROUTE_BINARY(operator+, add)
ENOKI_ROUTE_BINARY(operator-, sub)
ENOKI_ROUTE_BINARY(operator*, mul)

ENOKI_ROUTE_BINARY_BIT(operator&,  and)
ENOKI_ROUTE_BINARY_BIT(operator&&, and)
ENOKI_ROUTE_BINARY_BIT(operator|,  or)
ENOKI_ROUTE_BINARY_BIT(operator||, or)
ENOKI_ROUTE_BINARY_BIT(operator^,  xor)

ENOKI_ROUTE_SHIFT(operator<<, slv)
ENOKI_ROUTE_SHIFT(operator>>, srv)

ENOKI_ROUTE_BINARY(operator<,  lt)
ENOKI_ROUTE_BINARY(operator<=, le)
ENOKI_ROUTE_BINARY(operator>,  gt)
ENOKI_ROUTE_BINARY(operator>=, ge)

ENOKI_ROUTE_BINARY_SCALAR(eq,  eq,  a1 == a2)
ENOKI_ROUTE_BINARY_SCALAR(neq, neq, a1 != a2)

ENOKI_ROUTE_COMPOUND_OPERATOR(+)
ENOKI_ROUTE_COMPOUND_OPERATOR(-)
ENOKI_ROUTE_COMPOUND_OPERATOR(*)
ENOKI_ROUTE_COMPOUND_OPERATOR(/)
ENOKI_ROUTE_COMPOUND_OPERATOR(^)
ENOKI_ROUTE_COMPOUND_OPERATOR(|)
ENOKI_ROUTE_COMPOUND_OPERATOR(&)
ENOKI_ROUTE_COMPOUND_OPERATOR(<<)
ENOKI_ROUTE_COMPOUND_OPERATOR(>>)

ENOKI_ROUTE_BINARY_SCALAR(max,   max,  (std::decay_t<decltype(a1+a2)>) std::max((decltype(a1+a2)) a1, (decltype(a1+a2)) a2))
ENOKI_ROUTE_BINARY_SCALAR(min,   min,  (std::decay_t<decltype(a1+a2)>) std::min((decltype(a1+a2)) a1, (decltype(a1+a2)) a2))
ENOKI_ROUTE_BINARY_SCALAR(pow,   pow,   std::pow  ((decltype(a1+a2)) a1, (decltype(a1+a2)) a2))
ENOKI_ROUTE_BINARY_SCALAR(atan2, atan2, std::atan2((decltype(a1+a2)) a1, (decltype(a1+a2)) a2))
ENOKI_ROUTE_BINARY_SCALAR(ldexp, ldexp, std::ldexp((decltype(a1+a2)) a1, int(a2)))
ENOKI_ROUTE_BINARY_SCALAR(dot,   dot,   a1*a2)

ENOKI_ROUTE_BINARY(mulhi, mulhi)

ENOKI_ROUTE_UNARY(abs, abs)

ENOKI_ROUTE_TERNARY(fmadd,    fmadd)
ENOKI_ROUTE_TERNARY(fmsub,    fmsub)
ENOKI_ROUTE_TERNARY(fnmadd,   fnmadd)
ENOKI_ROUTE_TERNARY(fnmsub,   fnmsub)
ENOKI_ROUTE_TERNARY(fmsubadd, fmsubadd)
ENOKI_ROUTE_TERNARY(fmaddsub, fmaddsub)

ENOKI_ROUTE_UNARY_SCALAR(sincos,  sincos,  std::make_pair(std::sin(a),  std::cos(a)))
ENOKI_ROUTE_UNARY_SCALAR(sincosh, sincosh, std::make_pair(std::sinh(a), std::cosh(a)))

ENOKI_ROUTE_UNARY(popcnt, popcnt)
ENOKI_ROUTE_UNARY(lzcnt, lzcnt)
ENOKI_ROUTE_UNARY(tzcnt, tzcnt)

ENOKI_ROUTE_UNARY_SCALAR(all,          all,           (bool) a)
ENOKI_ROUTE_UNARY_SCALAR(all_nested,   all_nested,    (bool) a)
ENOKI_ROUTE_UNARY_SCALAR(any,          any,           (bool) a)
ENOKI_ROUTE_UNARY_SCALAR(any_nested,   any_nested,    (bool) a)
ENOKI_ROUTE_UNARY_SCALAR(none,         none,         !(bool) a)
ENOKI_ROUTE_UNARY_SCALAR(none_nested,  none_nested,  !(bool) a)
ENOKI_ROUTE_UNARY_SCALAR(count,        count,         (size_t) ((bool) a ? 1 : 0))
ENOKI_ROUTE_UNARY_SCALAR(count_nested, count_nested,  (size_t) ((bool) a ? 1 : 0))

ENOKI_ROUTE_UNARY_SCALAR(hmin,          hmin,          a)
ENOKI_ROUTE_UNARY_SCALAR(hmin_nested,   hmin_nested,   a)
ENOKI_ROUTE_UNARY_SCALAR(hmax,          hmax,          a)
ENOKI_ROUTE_UNARY_SCALAR(hmax_nested,   hmax_nested,   a)
ENOKI_ROUTE_UNARY_SCALAR(hsum,          hsum,          a)
ENOKI_ROUTE_UNARY_SCALAR(hsum_nested,   hsum_nested,   a)
ENOKI_ROUTE_UNARY_SCALAR(hprod,         hprod,         a)
ENOKI_ROUTE_UNARY_SCALAR(hprod_nested,  hprod_nested,  a)

ENOKI_ROUTE_UNARY_SCALAR(sqrt,  sqrt,  std::sqrt(a))
ENOKI_ROUTE_UNARY_SCALAR(floor, floor, std::floor(a))
ENOKI_ROUTE_UNARY_SCALAR(ceil,  ceil,  std::ceil(a))
ENOKI_ROUTE_UNARY_SCALAR(round, round, std::rint(a))

ENOKI_ROUTE_UNARY_SCALAR(log,   log,   std::log(a))

ENOKI_ROUTE_UNARY_SCALAR(sin,   sin,   std::sin(a))
ENOKI_ROUTE_UNARY_SCALAR(cos,   cos,   std::cos(a))
ENOKI_ROUTE_UNARY_SCALAR(tan,   tan,   std::tan(a))

ENOKI_ROUTE_UNARY_SCALAR(csc,   csc,   T(1) / std::sin(a))
ENOKI_ROUTE_UNARY_SCALAR(sec,   sec,   T(1) / std::cos(a))
ENOKI_ROUTE_UNARY_SCALAR(cot,   cot,   T(1) / std::tan(a))

ENOKI_ROUTE_UNARY_SCALAR(asin,  asin,  std::asin(a))
ENOKI_ROUTE_UNARY_SCALAR(acos,  acos,  std::acos(a))
ENOKI_ROUTE_UNARY_SCALAR(atan,  atan,  std::atan(a))

ENOKI_ROUTE_UNARY_SCALAR(sinh,  sinh,  std::sinh(a))
ENOKI_ROUTE_UNARY_SCALAR(cosh,  cosh,  std::cosh(a))
ENOKI_ROUTE_UNARY_SCALAR(tanh,  tanh,  std::tanh(a))

ENOKI_ROUTE_UNARY_SCALAR(csch,  csch,  T(1) / std::sinh(a))
ENOKI_ROUTE_UNARY_SCALAR(sech,  sech,  T(1) / std::cosh(a))
ENOKI_ROUTE_UNARY_SCALAR(coth,  coth,  T(1) / std::tanh(a))

ENOKI_ROUTE_UNARY_SCALAR(asinh, asinh, std::asinh(a))
ENOKI_ROUTE_UNARY_SCALAR(acosh, acosh, std::acosh(a))
ENOKI_ROUTE_UNARY_SCALAR(atanh, atanh, std::atanh(a))

ENOKI_ROUTE_UNARY_SCALAR(isnan, isnan, std::isnan(a))
ENOKI_ROUTE_UNARY_SCALAR(isinf, isinf, std::isinf(a))
ENOKI_ROUTE_UNARY_SCALAR(isfinite, isfinite, std::isfinite(a))

/// Break floating-point number into normalized fraction and power of 2 (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE std::pair<Arg, Arg> frexp(const Arg &a) {
    int tmp;
    Arg result = std::frexp(a, &tmp);
    return std::make_pair(result, Arg(tmp));
}

ENOKI_ROUTE_UNARY(frexp, frexp)

template <typename T, std::enable_if_t<!is_array<T>::value && std::is_signed<T>::value, int> = 0>
ENOKI_INLINE T abs(const T &a) {
    return std::abs(a);
}

template <typename T, std::enable_if_t<!is_array<T>::value && std::is_unsigned<T>::value, int> = 0>
ENOKI_INLINE T abs(const T &a) {
    return a;
}

/* select(): case 1: use array-specific implementation of operation */
template <typename T,
          std::enable_if_t<is_array<T>::value &&
                           std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto select(mask_t<T> m, const T &t, const T &f) {
    return T::Derived::select_(m.derived(), t.derived(), f.derived());
}

/* select(): case 2: scalar input */
template <typename T1, typename T2, typename T3,
          typename Void = detail::extract_array_t<T1, T2, T3>,
          std::enable_if_t<std::is_void<Void>::value, int> = 0>
ENOKI_INLINE auto select(const T1 &m, const T2 &t, const T3 &f) {
    using T = expr_t<T2, T3>;
    return (bool) m ? (T) t : (T) f;
}

/* select(): case 3: broadcast/evaluate input arrays and try again */
template <typename T1, typename T2, typename T3,
          typename Array = detail::extract_array_t<T2, T3>,
          std::enable_if_t<!std::is_void<Array>::value, int> = 0>
ENOKI_INLINE auto select(const T1 &m, const T2 &t, const T3 &f) {
    using Output = expr_t<T2, T3>;
    return select((detail::ref_cast_t<T1, mask_t<Output>>) m,
                  (detail::ref_cast_t<T2, Output>) t,
                  (detail::ref_cast_t<T3, Output>) f);
}

template <typename Target, typename Array,
          std::enable_if_t<is_array<Target>::value &&
                           !std::is_same<Array, Target>::value, int> = 0>
ENOKI_INLINE Target reinterpret_array(const Array &a) {
    return Target(a, detail::reinterpret_flag());
}

template <typename Target, typename Array,
          std::enable_if_t<is_array<Target>::value &&
                           std::is_same<Array, Target>::value, int> = 0>
ENOKI_INLINE Target reinterpret_array(const Array &a) {
    return a;
}

template <typename Target, typename Arg, enable_if_not_array_t<Target> = 0,
          std::enable_if_t<sizeof(Arg) == sizeof(Target), int> = 0>
ENOKI_INLINE Target reinterpret_array(const Arg &a) {
    return memcpy_cast<Target>(a);
}

template <typename Target, typename Arg, enable_if_not_array_t<Target> = 0,
          std::enable_if_t<sizeof(Arg) != sizeof(Target) &&
                           std::is_same<Target, bool>::value, int> = 0>
ENOKI_INLINE Target reinterpret_array(const Arg &a) {
    using Int = typename detail::type_chooser<sizeof(Arg)>::Int;
    return memcpy_cast<Int>(a) != 0;
}

template <typename Target, typename Arg, enable_if_not_array_t<Target> = 0,
          std::enable_if_t<sizeof(Arg) != sizeof(Target) &&
          std::is_same<Arg, bool>::value, int> = 0>
ENOKI_INLINE Target reinterpret_array(const Arg &a) {
    using Scalar = scalar_t<Target>;
    using Int = typename detail::type_chooser<sizeof(Scalar)>::Int;
    Int value = a ? Int(-1) : Int(0);
    return Target(memcpy_cast<Scalar>(value));
}

template <typename T1, typename T2,
          typename Array = detail::extract_array_t<T1, T2>,
          std::enable_if_t<!std::is_void<Array>::value, int> = 0>
ENOKI_INLINE bool operator==(const T1 &a1, const T2 &a2) {
    return all_nested(eq(a1, a2));
}

template <typename T1, typename T2,
          typename Array = detail::extract_array_t<T1, T2>,
          std::enable_if_t<!std::is_void<Array>::value, int> = 0>
ENOKI_INLINE bool operator!=(const T1 &a1, const T2 &a2) {
    return any_nested(neq(a1, a2));
}

template <typename Array1, typename Array2>
ENOKI_INLINE auto abs_dot(const Array1 &a1, const Array2 &a2) {
    return abs(dot(a1, a2));
}

template <typename Array>
ENOKI_INLINE auto mean(const Array &a) {
    return hsum(a) * (1.f / array_size<Array>::value);
}

template <size_t Imm, typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg roli(const Arg &a) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a << (Imm & mask)) | (a >> ((~Imm + 1) & mask));
}

template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg rol(const Arg &a, size_t amt) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a << (amt & mask)) | (a >> ((~amt + 1) & mask));
}

template <size_t Imm, typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg rori(const Arg &a) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a >> (Imm & mask)) | (a << ((~Imm + 1) & mask));
}

template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg ror(const Arg &a, size_t amt) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a >> (amt & mask)) | (a << ((~amt + 1) & mask));
}

template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE auto ror(const Array &a, size_t value) {
    return a.derived().ror_(value);
}

template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE auto rol(const Array &a, size_t value) {
    return a.derived().rol_(value);
}

template <typename T,
          std::enable_if_t<is_array<T>::value && std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto operator<<(const T &a, size_t value) {
    return a.derived().sl_(value);
}
template <typename T,
          std::enable_if_t<is_array<T>::value && !std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto operator<<(const T &a, size_t value) {
    return operator<<((expr_t<T>) a, value);
}

template <typename T,
          std::enable_if_t<is_array<T>::value && std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto operator>>(const T &a, size_t value) {
    return a.derived().sr_(value);
}
template <typename T,
          std::enable_if_t<is_array<T>::value && !std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto operator>>(const T &a, size_t value) {
    return operator>>((expr_t<T>) a, value);
}

ENOKI_ROUTE_UNARY_SCALAR_IMM(sli, sli, a << Imm)
ENOKI_ROUTE_UNARY_SCALAR_IMM(sri, sri, a >> Imm)

ENOKI_ROUTE_UNARY_IMM(roli, roli)
ENOKI_ROUTE_UNARY_IMM(rori, rori)

ENOKI_ROUTE_UNARY_IMM(rol_array, rol_array)
ENOKI_ROUTE_UNARY_IMM(ror_array, ror_array)

ENOKI_ROUTE_BINARY(ror, rorv)
ENOKI_ROUTE_BINARY(rol, rolv)

template <bool = false, typename T,
          std::enable_if_t<is_array<T>::value && std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto rcp(const T &a) {
    return a.derived().rcp_();
}
template <bool = false, typename T,
          std::enable_if_t<is_array<T>::value && !std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto rcp(const T &a) {
    return rcp((expr_t<T>) a);
}

/// Reciprocal (scalar fallback)
template <bool Approx = false, typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg rcp(const Arg &a) {
#if defined(__AVX512ER__)
    if (std::is_same<Arg, float>::value) {
        __m128 v = _mm_set_ss((float) a);
        return Arg(_mm_cvtss_f32(_mm_rcp28_ss(v, v))); /* rel error < 2^-28 */
    }
#endif

#if defined(__SSE4_2__)
    if (Approx && std::is_same<Arg, float>::value) {
        __m128 v = _mm_set_ss((float) a), r;

        #if defined(__AVX512F__)
            r = _mm_rcp14_ss(v, v); /* rel error < 2^-14 */
        #else
            r = _mm_rcp_ss(v);      /* rel error < 1.5*2^-12 */
        #endif

        /* Refine using one Newton-Raphson iteration */
        __m128 ro = r;

        __m128 t0 = _mm_add_ss(r, r);
        __m128 t1 = _mm_mul_ss(r, v);

        #if defined(__FMA__)
            r = _mm_fnmadd_ss(r, t1, t0);
        #else
            r = _mm_sub_ss(t0, _mm_mul_ss(r, t1));
        #endif

        r = _mm_blendv_ps(r, ro, t1); /* mask bit is '1' iff t1 == nan */

        return Arg(_mm_cvtss_f32(r));
    }
#endif

#if defined(__AVX512F__) || defined(__AVX512ER__)
    if (Approx && std::is_same<Arg, double>::value) {
        __m128d v = _mm_set_sd((double) a), r;

        #if defined(__AVX512ER__)
            r = _mm_rcp28_sd(v, v);   /* rel error < 2^-28 */
        #elif defined(__AVX512F__)
            r = _mm_rcp14_sd(v, v);  /* rel error < 2^-14 */
        #endif

        __m128d ro = r, t0, t1;

        /* Refine using 1-2 Newton-Raphson iterations */
        ENOKI_UNROLL for (int i = 0; i < (has_avx512er ? 1 : 2); ++i) {
            t0 = _mm_add_sd(r, r);
            t1 = _mm_mul_sd(r, v);

            #if defined(__FMA__)
                r = _mm_fnmadd_sd(t1, r, t0);
            #else
                r = _mm_sub_sd(t0, _mm_mul_sd(r, t1));
            #endif
        }

        r = _mm_blendv_pd(r, ro, t1); /* mask bit is '1' iff t1 == nan */

        return Arg(_mm_cvtsd_f64(r));
    }
#endif

    return Arg(1) / a;
}

template <bool = false, typename T,
          std::enable_if_t<is_array<T>::value && std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto rsqrt(const T &a) {
    return a.derived().rsqrt_();
}
template <bool = false, typename T,
          std::enable_if_t<is_array<T>::value && !std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto rsqrt(const T &a) {
    return rsqrt((expr_t<T>) a);
}

/// Reciprocal square root (scalar fallback)
template <bool Approx = false, typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg rsqrt(const Arg &a) {
#if defined(__AVX512ER__)
    if (std::is_same<Arg, float>::value) {
        __m128 v = _mm_set_ss((float) a);
        return Arg(_mm_cvtss_f32(_mm_rsqrt28_ss(v, v))); /* rel error < 2^-28 */
    }
#endif

#if defined(__SSE4_2__)
    if (Approx && std::is_same<Arg, float>::value) {
        __m128 v = _mm_set_ss((float) a), r;

        #if defined(__AVX512F__)
            r = _mm_rsqrt14_ss(v, v);  /* rel error < 2^-14 */
        #else
            r = _mm_rsqrt_ss(v);       /* rel error < 1.5*2^-12 */
        #endif

        /* Refine using one Newton-Raphson iteration */
        const __m128 c0 = _mm_set_ss(0.5f),
                     c1 = _mm_set_ss(3.0f);

        __m128 t0 = _mm_mul_ss(r, c0),
               t1 = _mm_mul_ss(r, v),
               ro = r;

        #if defined(__FMA__)
            r = _mm_mul_ss(_mm_fnmadd_ss(t1, r, c1), t0);
        #else
            r = _mm_mul_ss(_mm_sub_ss(c1, _mm_mul_ss(t1, r)), t0);
        #endif

        r = _mm_blendv_ps(r, ro, t1); /* mask bit is '1' iff t1 == nan */

        return Arg(_mm_cvtss_f32(r));
    }
#endif

#if defined(__AVX512F__) || defined(__AVX512ER__)
    if (Approx && std::is_same<Arg, double>::value) {
        __m128d v = _mm_set_sd((double) a), r;

        #if defined(__AVX512ER__)
            r = _mm_rsqrt28_sd(v, v);  /* rel error < 2^-28 */
        #elif defined(__AVX512F__)
            r = _mm_rsqrt14_sd(v, v);  /* rel error < 2^-14 */
        #endif

        const __m128d c0 = _mm_set_sd(0.5),
                      c1 = _mm_set_sd(3.0);

        __m128d ro = r, t0, t1;

        /* Refine using 1-2 Newton-Raphson iterations */
        ENOKI_UNROLL for (int i = 0; i < (has_avx512er ? 1 : 2); ++i) {
            t0 = _mm_mul_sd(r, c0);
            t1 = _mm_mul_sd(r, v);

            #if defined(__FMA__)
                r = _mm_mul_sd(_mm_fnmadd_sd(t1, r, c1), t0);
            #else
                r = _mm_mul_sd(_mm_sub_sd(c1, _mm_mul_sd(t1, r)), t0);
            #endif
        }

        r = _mm_blendv_pd(r, ro, t1); /* mask bit is '1' iff t1 == nan */

        return Arg(_mm_cvtsd_f64(r));
    }
#endif

    return Arg(1) / std::sqrt(a);
}

template <bool = false, typename T,
          std::enable_if_t<is_array<T>::value && std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto exp(const T &a) {
    return a.derived().exp_();
}
template <bool = false, typename T,
          std::enable_if_t<is_array<T>::value && !std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto exp(const T &a) {
    return exp((expr_t<T>) a);
}

template <bool Approx = false, typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg exp(const Arg &a) {
#if defined(__AVX512ER__)
    if (std::is_same<Arg, float>::value && Approx) {
        __m128 v = _mm512_castps512_ps128(
            _mm512_exp2a23_ps(_mm512_castps128_ps512(_mm_mul_ps(
                _mm_set_ss((float) a), _mm_set1_ps(1.4426950408889634074f)))));
        return Arg(_mm_cvtss_f32(v));
    }
#endif
    return std::exp(a);
}

/* operator/, operator%: case 1: use array-specific implementation of operation */
template <typename T,
          std::enable_if_t<is_array<T>::value && std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto operator/(const T &a1, const T &a2) {
    return a1.derived().div_(a2.derived());
}

template <typename T,
          std::enable_if_t<is_array<T>::value && std::is_same<expr_t<T>, T>::value, int> = 0>
ENOKI_INLINE auto operator%(const T &a1, const T &a2) {
    return a1.derived().mod_(a2.derived());
}

/* operator/, operator%: case 2: broadcast/evaluate input arrays and try again */
template <typename T1, typename T2,
          typename Array = detail::extract_array_t<T1, T2>,
          std::enable_if_t<!std::is_void<Array>::value
                           &&!(std::is_integral<scalar_t<T1>>::value && std::is_integral<T2>::value), int> = 0>
ENOKI_INLINE auto operator/(const T1 &a1, const T2 &a2) {
    using Cast   = expr_t<scalar_t<T1>, T2>;
    using Output = expr_t<T1, T2>;

    Output result;
    if (array_depth<T1>::value > array_depth<T2>::value && Output::Approx)
        result = (detail::ref_cast_t<T1, Output>) a1 *
                 rcp<Output::Approx>((Cast) a2);
    else
        result = (detail::ref_cast_t<T1, Output>) a1 /
                 (detail::ref_cast_t<T2, Output>) a2;
    return result;
}

template <typename T1, typename T2,
          typename Array = detail::extract_array_t<T1, T2>,
          std::enable_if_t<!std::is_void<Array>::value
                           &&!(std::is_integral<scalar_t<T1>>::value && std::is_integral<T2>::value), int> = 0>
ENOKI_INLINE auto operator%(const T1 &a1, const T2 &a2) {
    using Output = expr_t<T1, T2>;
    return (detail::ref_cast_t<T1, Output>) a1 %
           (detail::ref_cast_t<T2, Output>) a2;
}

/// Shuffle the entries of an array
template <size_t... Args, typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE auto shuffle(const Array &in) { return in.derived().template shuffle_<Args...>(); }

/// Shuffle the entries of an array (scalar fallback)
template <size_t Index, typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg shuffle(const Arg &arg) {
    static_assert(Index == 0, "Invalid argument to shuffle");
    return arg;
}

//// Convert radians to degrees
template <typename Arg, typename E = expr_t<Arg>> ENOKI_INLINE E rad_to_deg(const Arg &value) {
    return scalar_t<E>(180 / M_PI) * value;
}

/// Convert degrees to radians
template <typename Arg, typename E = expr_t<Arg>> ENOKI_INLINE E deg_to_rad(const Arg &value) {
    return scalar_t<E>(M_PI / 180) * value;
}

NAMESPACE_BEGIN(detail)
template <typename Arg>
ENOKI_INLINE auto sign_mask(const Arg &a) {
    using UInt = scalar_t<uint_array_t<Arg>>;
    using Float = scalar_t<Arg>;
    const Float mask = memcpy_cast<Float>(UInt(1) << (sizeof(UInt) * 8 - 1));
    return detail::and_(a, expr_t<Arg>(mask));
}
NAMESPACE_END(detail)

template <typename Array, enable_if_static_array_t<Array> = 0, typename Expr = expr_t<Array>>
ENOKI_INLINE Expr sign(const Array &a) {
    using Scalar = scalar_t<Expr>;

    if (!std::is_signed<Scalar>::value)
        return Expr(Scalar(1));
    else if (std::is_floating_point<Scalar>::value)
        return detail::sign_mask(a) | Expr(Scalar(1));
    else
        return select(a < Scalar(0), Expr(Scalar(-1)), Expr(Scalar(1)));
}

template <typename Array1, typename Array2, typename Result = expr_t<Array1, Array2>,
          enable_if_array_t<Result> = 0>
ENOKI_INLINE Result copysign(const Array1 &a, const Array2 &b) {
    static_assert(std::is_same<scalar_t<Array1>, scalar_t<Array2>>::value, "Mismatched argument types!");
    static_assert(std::is_signed<scalar_t<Array1>>::value, "copysign() expects signed arguments!");

    if (std::is_floating_point<scalar_t<Result>>::value) {
        return abs(a) | detail::sign_mask(b);
    } else {
        Result result(a);
        result[(a ^ b) < 0] = -a;
        return result;
    }
}

template <typename Array1, typename Array2, typename Result = expr_t<Array1, Array2>,
          enable_if_array_t<Result> = 0>
ENOKI_INLINE Result mulsign(const Array1 &a, const Array2 &b) {
    static_assert(std::is_same<scalar_t<Array1>, scalar_t<Array2>>::value, "Mismatched argument types!");
    static_assert(std::is_signed<scalar_t<Array1>>::value, "mulsign() expects signed arguments!");

    if (std::is_floating_point<scalar_t<Result>>::value) {
        return a ^ detail::sign_mask(b);
    } else {
        Result result(a);
        result[b < 0] = -a;
        return result;
    }
}

template <typename Arg, enable_if_not_array_t<Arg> = 0>
inline Arg sign(const Arg &a) {
    return std::copysign(Arg(1), a);
}

template <typename Arg, enable_if_not_array_t<Arg> = 0>
inline Arg copysign(const Arg &a, const Arg &b) {
    return std::copysign(a, b);
}

template <typename Arg, enable_if_not_array_t<Arg> = 0>
inline Arg mulsign(const Arg &a, const Arg &b) {
    return a * std::copysign(Arg(1), b);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
ENOKI_INLINE T popcnt(T v) {
#if defined(__SSE4_2__)
    if (sizeof(T) <= 4)
        return (T) _mm_popcnt_u32((unsigned int) v);
    #if defined(__x86_64__) || defined(_M_X64)
        return (T) _mm_popcnt_u64((unsigned long long) v);
    #else
        unsigned long long v_ = (unsigned long long) v;
        unsigned int lo = (unsigned int) v_;
        unsigned int hi = (unsigned int) (v_ >> 32);
        return (T) (_mm_popcnt_u32(lo) + _mm_popcnt_u32(hi));
    #endif
#elif defined(_MSC_VER)
    if (sizeof(T) <= 4) {
        uint32_t w = (uint32_t) v;
        w -= (w >> 1) & 0x55555555;
        w = (w & 0x33333333) + ((w >> 2) & 0x33333333);
        w = (w + (w >> 4)) & 0x0F0F0F0F;
        w = (w * 0x01010101) >> 24;
        return (T) w;
    } else {
        uint64_t w = (uint64_t) v;
        w -= (w >> 1) & 0x5555555555555555ull;
        w = (w & 0x3333333333333333ull) + ((w >> 2) & 0x3333333333333333ull);
        w = (w + (w >> 4)) & 0x0F0F0F0F0F0F0F0Full;
        w = (w * 0x0101010101010101ull) >> 56;
        return (T) w;
    }
#else
    if (sizeof(T) <= 4)
        return (T) __builtin_popcount((unsigned int) v);
    else
        return (T) __builtin_popcountll((unsigned long long) v);
#endif
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
ENOKI_INLINE T lzcnt(T v) {
#if defined(__AVX2__)
    if (sizeof(T) <= 4)
        return (T) _lzcnt_u32((unsigned int) v);
    #if defined(__x86_64__) || defined(_M_X64)
        return (T) _lzcnt_u64((unsigned long long) v);
    #else
        unsigned long long v_ = (unsigned long long) v;
        unsigned int lo = (unsigned int) v_;
        unsigned int hi = (unsigned int) (v_ >> 32);
        return (T) (hi != 0 ? _lzcnt_u32(hi) : (_lzcnt_u32(lo) + 32));
    #endif
#elif defined(_MSC_VER)
    unsigned long result;
    if (sizeof(T) <= 4) {
        _BitScanReverse(&result, (unsigned long) v);
        return (v != 0) ? (31 - result) : 32;
    } else {
        _BitScanReverse64(&result, (unsigned long long) v);
        return (v != 0) ? (63 - result) : 64;
    }
#else
    if (sizeof(T) <= 4)
        return (T) (v != 0 ? __builtin_clz((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __builtin_clzll((unsigned long long) v) : 64);
#endif
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
ENOKI_INLINE T tzcnt(T v) {
#if defined(__AVX2__)
    if (sizeof(T) <= 4)
        return (T) _tzcnt_u32((unsigned int) v);
    #if defined(__x86_64__) || defined(_M_X64)
        return (T) _tzcnt_u64((unsigned long long) v);
    #else
        unsigned long long v_ = (unsigned long long) v;
        unsigned int lo = (unsigned int) v_;
        unsigned int hi = (unsigned int) (v_ >> 32);
        return (T) (lo != 0 ? _tzcnt_u32(lo) : (_tzcnt_u32(hi) + 32));
    #endif
#elif defined(_MSC_VER)
    unsigned long result;
    if (sizeof(T) <= 4) {
        _BitScanForward(&result, (unsigned long) v);
        return (v != 0) ? result : 32;
    } else {
        _BitScanForward64(&result, (unsigned long long) v);
        return (v != 0) ? result: 64;
    }
#else
    if (sizeof(T) <= 4)
        return (T) (v != 0 ? __builtin_ctz((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __builtin_ctzll((unsigned long long) v) : 64);
#endif
}

/// Fast implementation for computing the base 2 log of an integer.
template <typename T> ENOKI_INLINE T log2i(T value) {
    return scalar_t<T>(sizeof(scalar_t<T>) * 8 - 1) - lzcnt(value);
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Initialization, loading/writing data
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)
template <typename Array, size_t... Args>
ENOKI_INLINE Array index_sequence_(std::index_sequence<Args...>) {
    return Array(((value_t<Array>) Args)...);
}

template <typename Array, typename Value, size_t... Args>
ENOKI_INLINE Array linspace_(std::index_sequence<Args...>, Value offset, Value step) {
    return Array(((Value) Args * step + offset)...);
}
NAMESPACE_END(detail)

/// Construct a zero-initialized array
template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE Array zero() { return Array::zero_(); }

/// Construct a zero-initialized array
template <typename Array, enable_if_dynamic_array_t<Array> = 0>
ENOKI_INLINE Array zero(size_t size) { return Array::zero_(size); }

/// Construct a zero-initialized array (scalar fallback)
template <typename Arg, std::enable_if_t<std::is_arithmetic<Arg>::value ||
                                         std::is_pointer<Arg>::value ||
                                         std::is_enum<Arg>::value, int> = 0>
ENOKI_INLINE Arg zero() {
    return Arg(0);
}

template <typename Arg, size_t... Indices> auto zero(std::index_sequence<Indices...>) {
    return Arg(zero<std::tuple_element_t<Indices, Arg>>()...);
}

template <typename Arg, size_t Size = std::tuple_size<Arg>::value> auto zero() {
    return zero<Arg>(std::make_index_sequence<Size>());
}

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE Array index_sequence() {
    return detail::index_sequence_<Array>(
        std::make_index_sequence<Array::Size>());
}

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_dynamic_array_t<Array> = 0>
ENOKI_INLINE Array index_sequence(size_t size) {
    return Array::index_sequence_(size);
}

/// Construct an index sequence, i.e. 0, 1, 2, .. (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg index_sequence() {
    return Arg(0);
}

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE Array linspace(scalar_t<Array> min, scalar_t<Array> max) {
    return detail::linspace_<Array>(
        std::make_index_sequence<Array::Size>(), min,
        (max - min) / (scalar_t<Array>) (Array::Size - 1));
}

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_dynamic_array_t<Array> = 0>
ENOKI_INLINE Array linspace(size_t size, scalar_t<Array> min, scalar_t<Array> max) {
    return Array::linspace_(size, min, max);
}

/// Construct an index sequence, i.e. 0, 1, 2, .. (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg linspace() {
    return Arg(0);
}

/// Load an array from aligned memory
template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE Array load(const void *mem) {
    return Array::load_(mem);
}

/// Load an array from aligned memory (masked)
template <typename Array, typename Mask,
          enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size == Mask::Size, int> = 0>
ENOKI_INLINE Array load(const void *mem, const Mask &mask) {
    return Array::load_(mem, reinterpret_array<mask_t<Array>>(mask));
}

/// Load an array from aligned memory (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg load(const void *mem) {
    assert((uintptr_t) mem % alignof(Arg) == 0);
    return *static_cast<const Arg *>(mem);
}

/// Load an array from aligned memory (scalar fallback, masked)
template <typename Arg, typename Mask, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg load(const void *mem, const Mask &mask) {
    assert((uintptr_t) mem % alignof(Arg) == 0);
    return detail::mask_active(mask) ? *static_cast<const Arg *>(mem) : Arg(0);
}

/// Load an array from unaligned memory
template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE Array load_unaligned(const void *mem) {
    return Array::load_unaligned_(mem);
}

/// Load an array from aligned memory (masked)
template <typename Array, typename Mask,
          enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size == Mask::Size, int> = 0>
ENOKI_INLINE Array load_unaligned(const void *mem, const Mask &mask) {
    return Array::load_unaligned_(mem, reinterpret_array<mask_t<Array>>(mask));
}

/// Load an array from unaligned memory (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg load_unaligned(const void *mem) {
    return *static_cast<const Arg *>(mem);
}

/// Load an array from unaligned memory (scalar fallback, masked)
template <typename Arg, typename Mask, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg load_unaligned(const void *mem, Mask mask) {
    return detail::mask_active(mask) ? *static_cast<const Arg *>(mem) : Arg(0);
}

/// Store an array to aligned memory
template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE void store(void *mem, const Array &a) {
    a.store_(mem);
}

/// Store an array to aligned memory (masked)
template <typename Array, typename Mask, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size == Mask::Size, int> = 0>
ENOKI_INLINE void store(void *mem, const Array &a, const Mask &mask) {
    a.store_(mem, reinterpret_array<mask_t<Array>>(mask));
}

/// Store an array to aligned memory (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE void store(void *mem, const Arg &a) {
    assert((uintptr_t) mem % alignof(Arg) == 0);
    *static_cast<Arg *>(mem) = a;
}

/// Store an array to aligned memory (scalar fallback, masked)
template <typename Arg, typename Mask, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE void store(void *mem, const Arg &a, const Mask &mask) {
    assert((uintptr_t) mem % alignof(Arg) == 0);
    if (detail::mask_active(mask))
        *static_cast<Arg *>(mem) = a;
}

/// Store an array to unaligned memory
template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE void store_unaligned(void *mem, const Array &a) {
    a.store_unaligned_(mem);
}

/// Store an array to unaligned memory (masked)
template <typename Array, typename Mask, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size == Mask::Size, int> = 0>
ENOKI_INLINE void store_unaligned(void *mem, const Array &a, const Mask &mask) {
    a.store_unaligned_(mem, reinterpret_array<mask_t<Array>>(mask));
}

/// Store an array to unaligned memory (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE void store_unaligned(void *mem, const Arg &a) {
    *static_cast<Arg *>(mem) = a;
}

/// Store an array to unaligned memory (scalar fallback)
template <typename Arg, typename Mask, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE void store_unaligned(void *mem, const Arg &a, const Mask &mask) {
    if (detail::mask_active(mask))
        *static_cast<Arg *>(mem) = a;
}

/// Prefetch operation
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          bool Write = false, size_t Level = 2, typename Index, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<std::is_integral<scalar_t<Index>>::value &&
                           Index::Size == Array::Size, int> = 0>
ENOKI_INLINE void prefetch(const void *mem, const Index &index) {
    Array::template prefetch_<Stride, Write, Level>(mem, index);
}

/// Prefetch operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), bool Write = false,
          size_t Level = 2, typename Index, enable_if_not_array_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0>
ENOKI_INLINE void prefetch(const void *mem, const Index &index) {
    auto ptr = (const Arg *) ((const uint8_t *) mem + index * Index(Stride));
#if defined(__GNUC__)
    __builtin_prefetch(ptr, Write ? 1 : 0);
#else
    (void) ptr;
#endif
}

/// Masked prefetch operation
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          bool Write = false, size_t Level = 2, typename Index, typename Mask,
          enable_if_static_array_t<Array> = 0,
          std::enable_if_t<std::is_integral<scalar_t<Index>>::value &&
                           Index::Size == Array::Size &&
                           Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE void prefetch(const void *mem, const Index &index,
                           const Mask &mask) {
    Array::template prefetch_<Stride, Write, Level>(
        mem, index, reinterpret_array<mask_t<Array>>(mask));
}

/// Masked prefetch operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), bool Write = false,
          size_t Level = 2, typename Index, enable_if_not_array_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0, typename Mask>
ENOKI_INLINE void prefetch(const void *mem, const Index &index, const Mask &mask) {
    auto ptr = (const Arg *) ((const uint8_t *) mem + index * Index(Stride));
#if defined(__GNUC__)
    if (detail::mask_active(mask))
        __builtin_prefetch(ptr, Write ? 1 : 0);
#else
    (void) ptr;
#endif
}

/// Gather operation
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Index::Size == Array::Size, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index) {
    return Array::template gather_<Stride>(mem, index);
}

template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size != array_size<Index>::value &&
                           array_depth<Index>::value != 0, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index) {
    using Offset = enoki::Array<Index, Array::Size>;
    return Array::template gather_<Stride>(mem, index + index_sequence<Offset>());
}

template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size != array_size<Index>::value &&
                           array_depth<Index>::value == 0, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index) {
    return load_unaligned<Array>((uint8_t *) mem + index * Stride);
}

/// Gather operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg),
          typename Index, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg gather(const void *mem, const Index &index) {
    return *((const Arg *) ((const uint8_t *) mem + index * Index(Stride)));
}

/// Masked gather operation
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Mask, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size == Index::Size &&
                           Array::Size == Mask::Size, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index, const Mask &mask) {
    return Array::template gather_<Stride>(
        mem, index, reinterpret_array<mask_t<Array>>(mask));
}

template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Mask, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size != array_size<Index>::value &&
                           array_depth<Index>::value != 0, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index, const Mask &mask) {
    using Offset = enoki::Array<Index, Array::Size>;
    using EntryMask = mask_t<value_t<Array>>;
    return Array::template gather_<Stride>(
        mem, index + index_sequence<Offset>(),
        mask_t<Array>(reinterpret_array<EntryMask>(mask)));
}

template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Mask, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Array::Size != array_size<Index>::value &&
                           array_depth<Index>::value == 0, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index, const Mask &mask) {
    if (ENOKI_LIKELY(detail::mask_active(mask)))
        return load_unaligned<Array>((uint8_t *) mem + index * Stride);
    else
        return zero<Array>();
}

/// Masked gather operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg),
          typename Index, typename Mask, enable_if_not_array_t<Arg> = 0>
ENOKI_INLINE Arg gather(const void *mem, const Index &index, const Mask &mask) {
    return detail::mask_active(mask)
               ? *((const Arg *) ((const uint8_t *) mem +
                                  index * Index(Stride))) : Arg(0);
}

/// Scatter operation
template <size_t Stride_ = 0, typename Array,
          typename Index, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<std::is_integral<scalar_t<Index>>::value &&
                           Index::Size == Array::Size, int> = 0>
ENOKI_INLINE void scatter(void *mem, const Array &value, const Index &index) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(std::decay_t<value_t<Array>>);
    value.template scatter_<Stride>(mem, index);
}

/// Scatter operation (scalar fallback)
template <size_t Stride_ = 0, typename Arg,
          typename Index, enable_if_not_array_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0>
ENOKI_INLINE void scatter(void *mem, const Arg &value, const Index &index) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(Arg);
    auto ptr = (Arg *) ((uint8_t *) mem + index * Index(Stride));
    *ptr = value;
}

/// Masked scatter operation
template <size_t Stride_ = 0, typename Array, typename Index, typename Mask,
          enable_if_static_array_t<Array> = 0,
          std::enable_if_t<std::is_integral<scalar_t<Index>>::value &&
                           Index::Size == Array::Size &&
                           Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE void scatter(void *mem, const Array &value, const Index &index,
                          const Mask &mask) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(std::decay_t<value_t<Array>>);
    value.template scatter_<Stride>(mem, index,
                                    reinterpret_array<mask_t<Array>>(mask));
}

/// Masked scatter operation (scalar fallback)
template <size_t Stride_ = 0, typename Arg, typename Index,
          enable_if_not_array_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0, typename Mask>
ENOKI_INLINE void scatter(void *mem, const Arg &value, const Index &index, const Mask &mask) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(Arg);
    auto ptr = (Arg *) ((uint8_t *) mem + index * Index(Stride));
    if (detail::mask_active(mask))
        *ptr = value;
}

/// Combined gather-modify-scatter operation without conflicts
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Func, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<std::is_integral<scalar_t<Index>>::value &&
                           Index::Size == Array::Size, int> = 0,
          typename... Args,
          typename = decltype(std::declval<const Func &>()(
              std::declval<Array &>(), std::declval<const Args &>()...))>
ENOKI_INLINE void transform(void *mem, const Index &index, const Func &func,
                            const Args &... args) {
    Array::template transform_<Stride>(mem, index, func, args...);
}

/// Combined gather-modify-scatter operation without conflicts
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Mask, typename Func,
          enable_if_static_array_t<Array> = 0,
          std::enable_if_t<std::is_integral<scalar_t<Index>>::value &&
                           Index::Size == Array::Size &&
                           Mask::Size == Array::Size, int> = 0,
          typename... Args,
          typename = decltype(std::declval<const Func &>()(
              std::declval<Array &>(), std::declval<const Args &>()...))>
ENOKI_INLINE void transform(void *mem, const Index &index, const Mask &mask, const Func &func,
                            const Args &... args) {
    Array::template transform_masked_<Stride>(
        mem, index, reinterpret_array<mask_t<Array>>(mask), func, args...);
}

/// Combined gather-modify-scatter operation without conflicts (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), typename Index, typename Func,
          typename Mask, enable_if_not_array_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0,
          typename... Args,
          typename = decltype(std::declval<const Func &>()(
              std::declval<Arg &>(), std::declval<const Args &>()...))>
ENOKI_INLINE void transform(void *mem, const Index &index, const Mask &mask,
                            const Func &func, const Args&... args) {
    Arg& ptr = *(Arg *) ((uint8_t *) mem + index * Index(Stride));
    if (detail::mask_active(mask))
        func(ptr, args...);
}

/// Combined gather-modify-scatter operation without conflicts (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), typename Index, typename Func,
          enable_if_not_array_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0,
          typename... Args,
          typename = decltype(std::declval<const Func &>()(
              std::declval<Arg &>(), std::declval<const Args &>()...))>
ENOKI_INLINE void transform(void *mem, const Index &index,
                            const Func &func, const Args&... args) {
    Arg &ptr = *(Arg *) ((uint8_t *) mem + index * Index(Stride));
    func(ptr, args...);
}

/// Mask extraction operation
template <typename Array, typename Mask, enable_if_static_array_t<Array> = 0,
          std::enable_if_t<Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE value_t<Array> extract(const Array &value, const Mask &mask) {
    return (value_t<Array>) value.extract_(reinterpret_array<mask_t<Array>>(mask));
}

/// Mask extraction operation (scalar fallback)
template <typename Arg, enable_if_not_array_t<Arg> = 0, typename Mask>
ENOKI_INLINE Arg extract(const Arg &value, const Mask &) {
    return value;
}
template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T fmadd(const T1 &t1, const T2 &t2, const T3 &t3) {
    return std::fma((T) t1, (T) t2, (T) t3);
}

template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<!std::is_floating_point<T>::value && !is_array<T>::value, int> = 0>
ENOKI_INLINE T fmadd(const T1 &t1, const T2 &t2, const T3 &t3) {
    return (T) t1 * (T) t2 + (T) t3;
}

template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T fmsub(const T1 &t1, const T2 &t2, const T3 &t3) {
    return std::fma((T) t1, (T) t2, -(T) t3);
}

template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<!std::is_floating_point<T>::value && !is_array<T>::value, int> = 0>
ENOKI_INLINE T fmsub(const T1 &t1, const T2 &t2, const T3 &t3) {
    return (T) t1 * (T) t2 - (T) t3;
}

template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T fnmadd(const T1 &t1, const T2 &t2, const T3 &t3) {
    return std::fma(-(T) t1, (T) t2, (T) t3);
}

template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<!std::is_floating_point<T>::value && !is_array<T>::value, int> = 0>
ENOKI_INLINE T fnmadd(const T1 &t1, const T2 &t2, const T3 &t3) {
    return -(T) t1 * (T) t2 + (T) t3;
}

template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
ENOKI_INLINE T fnmsub(const T1 &t1, const T2 &t2, const T3 &t3) {
    return std::fma(-(T) t1, (T) t2, -(T) t3);
}

template <typename T1, typename T2, typename T3, typename T = expr_t<T1, T2, T3>,
          std::enable_if_t<!std::is_floating_point<T>::value && !is_array<T>::value, int> = 0>
ENOKI_INLINE T fnmsub(const T1 &t1, const T2 &t2, const T3 &t3) {
    return -(T) t1 * (T) t2 - (T) t3;
}

template <typename T1, typename T2, typename T3,
          typename T = expr_t<T1, T2, T3>, enable_if_not_array_t<T> = 0>
ENOKI_INLINE T fmaddsub(const T1 &t1, const T2 &t2, const T3 &t3) {
    return fmsub(t1, t2, t3);
}

template <typename T1, typename T2, typename T3,
          typename T = expr_t<T1, T2, T3>, enable_if_not_array_t<T> = 0>
ENOKI_INLINE T fmsubadd(const T1 &t1, const T2 &t2, const T3 &t3) {
    return fmadd(t1, t2, t3);
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name "Safe" functions that avoid domain errors due to rounding
// -----------------------------------------------------------------------

template <typename T> ENOKI_INLINE auto safe_sqrt(const T &a) {
    return sqrt(max(a, zero<T>()));
}

template <typename T> ENOKI_INLINE auto safe_rsqrt(const T &a) {
    return rsqrt(max(a, zero<T>()));
}

template <typename T> ENOKI_INLINE auto safe_asin(const T &a) {
    return asin(min(T(1), max(T(-1), a)));
}

template <typename T> ENOKI_INLINE auto safe_acos(const T &a) {
    return acos(min(T(1), max(T(-1), a)));
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Miscellaneous functions
// -----------------------------------------------------------------------

/// Extract the low elements from an array of even size
template <typename Array, enable_if_static_array_t<Array> = 0>
auto low(const Array &a) { return a.derived().low_(); }

/// Extract the high elements from an array of even size
template <typename Array, enable_if_static_array_t<Array> = 0>
auto high(const Array &a) { return a.derived().high_(); }

template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE auto norm(const Array &v) {
    return sqrt(dot(v, v));
}

template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE auto squared_norm(const Array &v) {
    return dot(v, v);
}

template <typename Array, enable_if_static_array_t<Array> = 0>
ENOKI_INLINE auto normalize(const Array &v) {
    return v * Array(rsqrt<Array::Approx>(squared_norm(v)));
}

template <typename Array1, typename Array2,
          enable_if_static_array_t<Array1> = 0,
          enable_if_static_array_t<Array2> = 0>
ENOKI_INLINE auto cross(const Array1 &v1, const Array2 &v2) {
    static_assert(Array1::Derived::Size == 3 && Array2::Derived::Size == 3,
                  "cross(): requires Size = 3");

    return fmsub(shuffle<1, 2, 0>(v1),  shuffle<2, 0, 1>(v2),
                 shuffle<2, 0, 1>(v1) * shuffle<1, 2, 0>(v2));
}

/// Generic range clamping function
template <typename Value1, typename Value2, typename Value3>
auto clamp(const Value1 &value, const Value2 &min_, const Value3 &max_) {
    return max(min(value, max_), min_);
}

template <typename Array1, typename Array2>
ENOKI_INLINE auto hypot(const Array1 &a, const Array2 &b) {
    auto abs_a = abs(a);
    auto abs_b = abs(b);
    auto max   = enoki::max(abs_a, abs_b);
    auto min   = enoki::min(abs_a, abs_b);
    auto ratio = min / max;

    using Scalar = scalar_t<decltype(ratio)>;
    const Scalar inf = std::numeric_limits<Scalar>::infinity();

    return select(
        abs_a < inf & abs_b < inf & ratio < inf,
        max * sqrt(Scalar(1) + ratio*ratio),
        abs_a + abs_b
    );
}

template <typename Value, typename Expr = expr_t<Value>>
ENOKI_INLINE Expr prev_float(const Value &value) {
    using Int = int_array_t<Expr>;
    using IntScalar = scalar_t<Int>;

    const Int exponent_mask = sizeof(IntScalar) == 4
                                  ? IntScalar(0x7f800000)
                                  : IntScalar(0x7ff0000000000000ll);

    const Int pos_denorm = sizeof(IntScalar) == 4
                              ? IntScalar(0x80000001)
                              : IntScalar(0x8000000000000001ll);

    Int i = reinterpret_array<Int>(value);

    auto is_nan_inf = eq(i & exponent_mask, exponent_mask);
    auto is_pos_0   = eq(i, 0);
    auto is_gt_0    = i >= 0;
    auto is_special = is_nan_inf | is_pos_0;

    Int j1 = i + select(is_gt_0, Int(-1), Int(1));
    Int j2 = select(is_pos_0, pos_denorm, i);

    return reinterpret_array<Expr>(select(is_special, j2, j1));
}

template <typename Value, typename Expr = expr_t<Value>>
ENOKI_INLINE Expr next_float(const Value &value) {
    using Int = int_array_t<Expr>;
    using IntScalar = scalar_t<Int>;

    const Int exponent_mask = sizeof(IntScalar) == 4
                                  ? IntScalar(0x7f800000)
                                  : IntScalar(0x7ff0000000000000ll);

    const Int sign_mask = sizeof(IntScalar) == 4
                              ? IntScalar(0x80000000)
                              : IntScalar(0x8000000000000000ll);

    Int i = reinterpret_array<Int>(value);

    auto is_nan_inf = eq(i & exponent_mask, exponent_mask);
    auto is_neg_0   = eq(i, sign_mask);
    auto is_gt_0    = i >= 0;
    auto is_special = is_nan_inf | is_neg_0;

    Int j1 = i + select(is_gt_0, Int(1), Int(-1));
    Int j2 = select(is_neg_0, Int(1), i);

    return reinterpret_array<Expr>(select(is_special, j2, j1));
}

/**
 * Broadcast the given array to the entries of an array of
 * shape (<shape of Other>, <shape of Array>)
 *
 * \tparam Other Denotes the desired shape of the leading
 *         dimensions of the output array
 *
 * \tparam Array Scalar/Array type of the argument.
 */
template <typename Other, typename Array>
ENOKI_INLINE auto broadcast(const Array &value) {
    return like_t<Other, Array>(value);
}

//! @}
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)

template <typename T> struct MaskedScalar {
    MaskedScalar(T &d, bool m) : d(d), m(m) { }

    template <typename T2> ENOKI_INLINE void operator =(const T2 &value) { if (m) d = value; }
    template <typename T2> ENOKI_INLINE void operator+=(const T2 &value) { if (m) d += value; }
    template <typename T2> ENOKI_INLINE void operator-=(const T2 &value) { if (m) d -= value; }
    template <typename T2> ENOKI_INLINE void operator*=(const T2 &value) { if (m) d *= value; }
    template <typename T2> ENOKI_INLINE void operator/=(const T2 &value) { if (m) d /= value; }
    template <typename T2> ENOKI_INLINE void operator|=(const T2 &value) { if (m) d |= value; }
    template <typename T2> ENOKI_INLINE void operator&=(const T2 &value) { if (m) d &= value; }
    template <typename T2> ENOKI_INLINE void operator^=(const T2 &value) { if (m) d ^= value; }

    T &d;
    bool m;
};

template <typename T> struct MaskedArray : ArrayBase<value_t<T>, MaskedArray<T>> {
    static constexpr bool Approx = T::Approx;
    using Mask = mask_t<T>;
    using Scalar = scalar_t<T>;

    MaskedArray(T &d, const Mask &m) : d(d), m(m) { }

    template <typename T2> ENOKI_INLINE void operator =(const T2 &value) { d.massign_(value, m); }
    template <typename T2> ENOKI_INLINE void operator+=(const T2 &value) { d.madd_(value, m); }
    template <typename T2> ENOKI_INLINE void operator-=(const T2 &value) { d.msub_(value, m); }
    template <typename T2> ENOKI_INLINE void operator*=(const T2 &value) { d.mmul_(value, m); }
    template <typename T2> ENOKI_INLINE void operator/=(const T2 &value) { d.mdiv_(value, m); }
    template <typename T2> ENOKI_INLINE void operator|=(const T2 &value) { d.mor_(value, m); }
    template <typename T2> ENOKI_INLINE void operator&=(const T2 &value) { d.mand_(value, m); }
    template <typename T2> ENOKI_INLINE void operator^=(const T2 &value) { d.mxor_(value, m); }

    /// Type alias for a similar-shaped array over a different type
    template <typename T2> using ReplaceType = MaskedArray<typename T::template ReplaceType<T2>>;

    T &d;
    Mask m;
};

NAMESPACE_END(detail)


// -----------------------------------------------------------------------
//! @{ \name Adapter and routing functions for dynamic data structures
// -----------------------------------------------------------------------

template <typename T, typename = int>
struct struct_support {
    static constexpr bool is_dynamic_nested = false;
    using dynamic_t = T;

    template <typename T2> static ENOKI_INLINE size_t slices(const T2 &) { return 0; }
    template <typename T2> static ENOKI_INLINE size_t packets(const T2 &) { return 0; }
    template <typename T2> static ENOKI_INLINE void set_slices(const T2 &, size_t) { }

    template <typename T2> static ENOKI_INLINE decltype(auto) ref_wrap(T2&& value) { return value; }
    template <typename T2> static ENOKI_INLINE decltype(auto) packet(T2&& value, size_t) { return value; }
    template <typename T2> static ENOKI_INLINE decltype(auto) slice(T2&& value, size_t) { return value; }
    template <typename T2> static ENOKI_INLINE decltype(auto) slice_ptr(T2&& value, size_t) { return value; }
    template <typename Mem, typename T2, typename Mask> static ENOKI_INLINE size_t compress(Mem& mem, const T2& value, const Mask &mask) {
        size_t count = detail::mask_active(mask) ? 1 : 0;
        *mem = value;
        mem += count;
        return count;
    }
    template <typename T2> static ENOKI_INLINE auto masked(T2 &value, bool mask) {
        return detail::MaskedScalar<T2>{value, mask};
    }
};

template <typename T> ENOKI_INLINE size_t packets(const T &value) {
    return struct_support<std::decay_t<T>>::packets(value);
}

template <typename T> ENOKI_INLINE size_t slices(const T &value) {
    return struct_support<std::decay_t<T>>::slices(value);
}

template <typename T> ENOKI_NOINLINE void set_slices(T &value, size_t size) {
    struct_support<std::decay_t<T>>::set_slices(value, size);
}

template <typename T>
ENOKI_INLINE decltype(auto) packet(T &&value, size_t i) {
    return struct_support<std::decay_t<T>>::packet(value, i);
}

template <typename T>
ENOKI_INLINE decltype(auto) slice(T &&value, size_t i) {
    return struct_support<std::decay_t<T>>::slice(value, i);
}

template <typename T>
ENOKI_INLINE decltype(auto) slice_ptr(T &&value, size_t i) {
    return struct_support<std::decay_t<T>>::slice_ptr(value, i);
}

template <typename T>
ENOKI_INLINE decltype(auto) ref_wrap(T &&value) {
    return struct_support<std::decay_t<T>>::ref_wrap(value);
}

template <typename Mem, typename Value, typename Mask>
ENOKI_INLINE size_t compress(Mem &mem, const Value &value, const Mask& mask) {
    return struct_support<std::decay_t<Value>>::compress(mem, value, mask);
}

template <typename T>
using is_dynamic_nested =
    std::integral_constant<bool, struct_support<std::decay_t<T>>::is_dynamic_nested>;

template <typename Array, typename Mask>
ENOKI_INLINE auto masked(Array &array, const Mask &mask) {
    return struct_support<std::decay_t<Array>>::masked(array, mask);
}

template <typename T>
using make_dynamic_t = typename struct_support<std::decay_t<T>>::dynamic_t;

template <typename T>
using enable_if_dynamic_nested_t =
    std::enable_if_t<is_dynamic_nested<T>::value, int>;

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Operations to query the depth and shape of nested arrays
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)
template <typename T, std::enable_if_t<!is_array<T>::value, int> = 0>
ENOKI_INLINE void get_shape_recursive(const T &, size_t *) { }

template <typename T, std::enable_if_t<is_array<T>::value, int> = 0>
ENOKI_INLINE void get_shape_recursive(const T &a, size_t *out) {
    *out++ = a.derived().size();
    get_shape_recursive(a.derived().coeff(0), out);
}

template <typename T, std::enable_if_t<!is_array<T>::value, int> = 0>
ENOKI_INLINE bool check_shape_recursive(const T &, const size_t *) { return true; }

template <typename T, std::enable_if_t<is_array<T>::value, int> = 0>
ENOKI_INLINE bool check_shape_recursive(const T &a, const size_t *shape) {
    size_t size = a.derived().size();
    if (*shape != size)
        return false;
    bool match = true;
    if (is_dynamic_nested<value_t<T>>::value) {
        for (size_t i = 0; i < size; ++i)
            match &= check_shape_recursive(a.derived().coeff(i), shape + 1);
    } else {
        check_shape_recursive(a.derived().coeff(0), shape + 1);
    }
    return match;
}

template <typename T, std::enable_if_t<!is_array<T>::value, int> = 0>
ENOKI_INLINE void set_shape_recursive(const T &, const size_t *) { }

template <typename T, std::enable_if_t<is_array<T>::value, int> = 0>
ENOKI_INLINE void set_shape_recursive(T &a, const size_t *shape) {
    size_t size = a.derived().size();
    a.resize_(*shape);
    if (is_dynamic_nested<value_t<T>>::value) {
        for (size_t i = 0; i < size; ++i)
            set_shape_recursive(a.derived().coeff(i), shape + 1);
    } else {
        set_shape_recursive(a.derived().coeff(0), shape + 1);
    }
}

NAMESPACE_END(detail)

template <typename T> std::array<size_t, array_depth<T>::value> shape(const T &a) {
    std::array<size_t, array_depth<T>::value> result;
    detail::get_shape_recursive(a, result.data());
    return result;
}

template <typename T>
void resize(T &a, const std::array<size_t, array_depth<T>::value> &value) {
    detail::set_shape_recursive(a, value.data());
}

template <typename T> bool ragged(const T &a) {
    auto shape = enoki::shape(a);
    return !detail::check_shape_recursive(a, shape.data());
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Polynomial evaluation with short dependency chains and
//           fused multply-adds based on Estrin's scheme
// -----------------------------------------------------------------------

template <typename T1, typename T2, typename T = expr_t<T1>, typename S = scalar_t<T1>>
ENOKI_INLINE T poly2(T1 x, T2 c0, T2 c1, T2 c2) {
    T x2 = x * x;
    return fmadd(x2, S(c2), fmadd(x, S(c1), S(c0)));
}

template <typename T1, typename T2, typename T = expr_t<T1>, typename S = scalar_t<T1>>
ENOKI_INLINE T poly3(T1 x, T2 c0, T2 c1, T2 c2, T2 c3) {
    T x2 = x * x;
    return fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)));
}

template <typename T1, typename T2, typename T = expr_t<T1>, typename S = scalar_t<T1>>
ENOKI_INLINE T poly4(T1 x, T2 c0, T2 c1, T2 c2, T2 c3, T2 c4) {
    T x2 = x * x, x4 = x2 * x2;
    return fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)) + S(c4) * x4);
}

template <typename T1, typename T2, typename T = expr_t<T1>, typename S = scalar_t<T1>>
ENOKI_INLINE T poly5(T1 x, T2 c0, T2 c1, T2 c2, T2 c3, T2 c4, T2 c5) {
    T x2 = x * x, x4 = x2 * x2;
    return fmadd(x2, fmadd(x, S(c3), S(c2)),
                     fmadd(x4, fmadd(x, S(c5), S(c4)), fmadd(x, S(c1), S(c0))));
}

template <typename T1, typename T2, typename T = expr_t<T1>, typename S = scalar_t<T1>>
ENOKI_INLINE T poly6(T1 x, T2 c0, T2 c1, T2 c2, T2 c3, T2 c4, T2 c5, T2 c6) {
    T x2 = x * x, x4 = x2 * x2;
    return fmadd(x4, fmadd(x2, S(c6), fmadd(x, S(c5), S(c4))),
                     fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0))));
}

template <typename T1, typename T2, typename T = expr_t<T1>, typename S = scalar_t<T1>>
ENOKI_INLINE T poly7(T1 x, T2 c0, T2 c1, T2 c2, T2 c3, T2 c4, T2 c5, T2 c6, T2 c7) {
    T x2 = x * x, x4 = x2 * x2;
    return fmadd(x4, fmadd(x2, fmadd(x, S(c7), S(c6)), fmadd(x, S(c5), S(c4))),
                     fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0))));
}

template <typename T1, typename T2, typename T = expr_t<T1>, typename S = scalar_t<T1>>
ENOKI_INLINE T poly8(T1 x, T2 c0, T2 c1, T2 c2, T2 c3, T2 c4, T2 c5, T2 c6, T2 c7, T2 c8) {
    T x2 = x * x, x4 = x2 * x2, x8 = x4 * x4;
    return fmadd(x4, fmadd(x2, fmadd(x, S(c7), S(c6)), fmadd(x, S(c5), S(c4))),
                     fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)) + S(c8) * x8));
}

//! @}
// -----------------------------------------------------------------------

#undef ENOKI_ROUTE_UNARY
#undef ENOKI_ROUTE_UNARY_SCALAR
#undef ENOKI_ROUTE_UNARY_IMM
#undef ENOKI_ROUTE_UNARY_SCALAR_IMM
#undef ENOKI_ROUTE_BINARY
#undef ENOKI_ROUTE_BINARY_SCALAR
#undef ENOKI_ROUTE_SHIFT
#undef ENOKI_ROUTE_TERNARY
#undef ENOKI_ROUTE_COMPOUND_OPERATOR

NAMESPACE_END(enoki)
