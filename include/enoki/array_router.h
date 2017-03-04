/*
    enoki/array_router.h -- Helper functions which route function calls
    in the enoki namespace to the intended recipients

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include "common.h"
#include <iostream>

NAMESPACE_BEGIN(enoki)

// -----------------------------------------------------------------------
//! @{ \name Vertical and horizontal operations
// -----------------------------------------------------------------------

/**
 * Binary operator macro: forward an operation (e.g. "abs()") to the specialized
 * implementation provided by the array partial specialization (e.g. "abs_()")
 */
#define ENOKI_ROUTE_UNARY(name, func)                                          \
    template <typename Type, size_t Size, bool Approx, RoundingMode Mode,      \
              typename Derived>                                                \
    ENOKI_INLINE auto name(                                                    \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a) {         \
        return a.derived().func##_();                                          \
    }

#define ENOKI_ROUTE_UNARY_WITH_FALLBACK(name, expr)                            \
    ENOKI_ROUTE_UNARY(name, name)                                              \
    template <typename Arg, enable_if_notarray_t<Arg> = 0>                     \
    ENOKI_INLINE auto name(const Arg &a) {                                     \
        return expr;                                                           \
    }

/**
 * Binary operator macro: dispatch operator+ to add_(), etc. if both arguments
 * have the same type. Otherwise perform an explicit loop over the array
 * elements and dispatch to the entry's operator+() implementation.
 */
#define ENOKI_ROUTE_BINARY(name, func, expr1, expr2)                           \
    template <typename Type, size_t Size, bool Approx, RoundingMode Mode,      \
              typename Derived1, typename Derived2,                            \
              std::enable_if_t<Derived1::Size == Derived2::Size, int> = 0>     \
    ENOKI_INLINE auto name(                                                    \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived1> &a1,         \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived2> &a2) {       \
        return a1.derived().func##_(a2.derived());                             \
    }                                                                          \
    template <typename Type1, typename Type2, size_t Size1, size_t Size2,      \
              bool Approx1, bool Approx2, RoundingMode Mode1,                  \
              RoundingMode Mode2, typename Derived1, typename Derived2,        \
              std::enable_if_t<Derived1::Size == Derived2::Size &&             \
                              !Derived1::IsMask && !Derived2::IsMask, int> = 0,\
              typename Value = decltype(expr1)>                                \
    ENOKI_INLINE auto name(                                                    \
        const StaticArrayBase<Type1, Size1, Approx1, Mode1, Derived1> &a1,     \
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a2) {   \
        Array<Value, Derived1::Size> r;                                        \
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived1::Size; ++i)            \
            r.coeff(i) = expr2;                                                \
        return r;                                                              \
    }

#define ENOKI_ROUTE_TERNARY(name, func, expr1, expr2)                          \
    template <typename Type, size_t Size, bool Approx, RoundingMode Mode,      \
              typename Derived1, typename Derived2, typename Derived3,         \
              std::enable_if_t<Derived1::Size == Derived2::Size &&             \
                               Derived2::Size == Derived3::Size, int> = 0>     \
    ENOKI_INLINE auto name(                                                    \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived1> &a1,         \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived2> &a2,         \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived3> &a3) {       \
        return a1.derived().func##_(a2.derived(), a3.derived());               \
    }                                                                          \
    template <typename Type1, typename Type2, typename Type3, size_t Size1,    \
              size_t Size2, size_t Size3, bool Approx1, bool Approx2,          \
              bool Approx3, RoundingMode Mode1, RoundingMode Mode2,            \
              RoundingMode Mode3, typename Derived1, typename Derived2,        \
              typename Derived3,                                               \
              std::enable_if_t<Derived1::Size == Derived2::Size &&             \
                               Derived2::Size == Derived3::Size &&             \
                               !Derived1::IsMask && !Derived2::IsMask &&       \
                               !Derived3::IsMask, int> = 0,                    \
              typename Value = decltype(expr1)>                                \
    ENOKI_INLINE auto name(                                                    \
        const StaticArrayBase<Type1, Size1, Approx1, Mode1, Derived1> &a1,     \
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a2,     \
        const StaticArrayBase<Type3, Size3, Approx3, Mode3, Derived3> &a3) {   \
        Array<Value, Derived1::Size> r;                                        \
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived1::Size; ++i)            \
            r.coeff(i) = expr2;                                                \
        return r;                                                              \
    }

#define ENOKI_ROUTE_BCAST(name)                                                \
    template <typename Type, size_t Size, bool Approx, RoundingMode Mode,      \
              typename Derived, typename Arg,                                  \
              std::enable_if_t<detail::bcast<Derived, Arg>::value, int> = 0,   \
              typename = decltype(typename Derived::Expr(std::declval<Arg>()))>\
    ENOKI_INLINE auto name(                                                    \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1,          \
        const Arg &a2) {                                                       \
        return name(a1.derived(), typename Derived::Expr(a2));                 \
    }                                                                          \
    template <typename Type, size_t Size, bool Approx, RoundingMode Mode,      \
              typename Derived, typename Arg,                                  \
              std::enable_if_t<detail::bcast<Derived, Arg>::value, int> = 0,   \
              typename = decltype(typename Derived::Expr(std::declval<Arg>()))>\
    ENOKI_INLINE auto name(                                                    \
        const Arg &a1,                                                         \
        const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a2) {        \
        return name(typename Derived::Expr(a1), a2.derived());                 \
    }

#define ENOKI_ROUTE_BINARY_OPERATOR(name, func)                                \
    ENOKI_ROUTE_BINARY(operator name, func,                                    \
                       std::declval<Type1>() name std::declval<Type2>(),       \
                       a1.derived().coeff(i) name a2.derived().coeff(i))

#define ENOKI_ROUTE_BINARY_FUNCTION(name, func)                                \
    ENOKI_ROUTE_BINARY(name, func,                                             \
                       name(std::declval<Type1>(), std::declval<Type2>()),     \
                       name(a1.derived().coeff(i), a2.derived().coeff(i)))

#define ENOKI_ROUTE_BINARY_OPERATOR_BCAST(name, func)                          \
    ENOKI_ROUTE_BINARY_OPERATOR(name, func)                                    \
    ENOKI_ROUTE_BCAST(operator name)

#define ENOKI_ROUTE_BINARY_FUNCTION_BCAST(name, func)                          \
    ENOKI_ROUTE_BINARY_FUNCTION(name, func)                                    \
    ENOKI_ROUTE_BCAST(name)

/// Macro for compound assignment operators (operator+=, etc.)
#define ENOKI_ROUTE_COMPOUND_OPERATOR(op)                                      \
    template <typename Type, size_t Size, bool Approx, RoundingMode Mode,      \
              typename Derived, typename Arg>                                  \
    ENOKI_INLINE Derived &operator op##=(                                      \
        StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1,                \
        const Arg &a2) {                                                       \
        a1.derived() = a1.derived() op a2;                                     \
        return a1.derived();                                                   \
    }

ENOKI_ROUTE_UNARY(operator-, neg)
ENOKI_ROUTE_UNARY(operator~, not)

ENOKI_ROUTE_BINARY_OPERATOR_BCAST(+,  add)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(-,  sub)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(*,  mul)
ENOKI_ROUTE_BINARY_OPERATOR      (/,  div)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(^,  xor)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(|,  or)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(&,  and)
ENOKI_ROUTE_BINARY_OPERATOR      (<<, slv)
ENOKI_ROUTE_BINARY_OPERATOR      (>>, srv)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(<,  lt)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(<=, le)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(>,  gt)
ENOKI_ROUTE_BINARY_OPERATOR_BCAST(>=, ge)

ENOKI_ROUTE_COMPOUND_OPERATOR(+)
ENOKI_ROUTE_COMPOUND_OPERATOR(-)
ENOKI_ROUTE_COMPOUND_OPERATOR(*)
ENOKI_ROUTE_COMPOUND_OPERATOR(/)
ENOKI_ROUTE_COMPOUND_OPERATOR(^)
ENOKI_ROUTE_COMPOUND_OPERATOR(|)
ENOKI_ROUTE_COMPOUND_OPERATOR(&)
ENOKI_ROUTE_COMPOUND_OPERATOR(<<)
ENOKI_ROUTE_COMPOUND_OPERATOR(>>)

ENOKI_ROUTE_BINARY_FUNCTION_BCAST(max,   max)
ENOKI_ROUTE_BINARY_FUNCTION_BCAST(min,   min)
ENOKI_ROUTE_BINARY_FUNCTION_BCAST(ldexp, ldexp)
ENOKI_ROUTE_BINARY_FUNCTION_BCAST(mulhi, mulhi)
ENOKI_ROUTE_BINARY_FUNCTION_BCAST(pow,   pow)
ENOKI_ROUTE_BINARY_FUNCTION_BCAST(atan2, atan2)
ENOKI_ROUTE_BINARY_FUNCTION_BCAST(eq,    eq)
ENOKI_ROUTE_BINARY_FUNCTION_BCAST(neq,   neq)
ENOKI_ROUTE_BINARY_FUNCTION      (ror,   rorv)
ENOKI_ROUTE_BINARY_FUNCTION      (rol,   rolv)

ENOKI_ROUTE_UNARY(abs, abs)

ENOKI_ROUTE_TERNARY(fmadd, fmadd,
                    fmadd(std::declval<Type1>(),
                          std::declval<Type2>(),
                          std::declval<Type3>()),
                    fmadd(a1.derived().coeff(i),
                          a2.derived().coeff(i),
                          a3.derived().coeff(i))
)

ENOKI_ROUTE_TERNARY(fmsub, fmsub,
                    fmsub(std::declval<Type1>(),
                          std::declval<Type2>(),
                          std::declval<Type3>()),
                    fmsub(a1.derived().coeff(i),
                          a2.derived().coeff(i),
                          a3.derived().coeff(i))
)

ENOKI_ROUTE_TERNARY(fmaddsub, fmaddsub,
                    fmaddsub(std::declval<Type1>(),
                             std::declval<Type2>(),
                             std::declval<Type3>()),
                    fmaddsub(a1.derived().coeff(i),
                             a2.derived().coeff(i),
                             a3.derived().coeff(i))
)

ENOKI_ROUTE_TERNARY(fmsubadd, fmsubadd,
                    fmsubadd(std::declval<Type1>(),
                             std::declval<Type2>(),
                             std::declval<Type3>()),
                    fmsubadd(a1.derived().coeff(i),
                             a2.derived().coeff(i),
                             a3.derived().coeff(i))
)

ENOKI_ROUTE_UNARY(frexp, frexp)
ENOKI_ROUTE_UNARY(sincos, sincos)
ENOKI_ROUTE_UNARY(sincosh, sincosh)
ENOKI_ROUTE_UNARY(all, all)
ENOKI_ROUTE_UNARY(any, any)
ENOKI_ROUTE_UNARY(none, none)
ENOKI_ROUTE_UNARY(count, count)
ENOKI_ROUTE_UNARY(all_nested, all_nested)
ENOKI_ROUTE_UNARY(any_nested, any_nested)
ENOKI_ROUTE_UNARY(none_nested, none_nested)
ENOKI_ROUTE_UNARY(count_nested, count_nested)

ENOKI_ROUTE_UNARY_WITH_FALLBACK(hmin, a)
ENOKI_ROUTE_UNARY_WITH_FALLBACK(hmax, a)
ENOKI_ROUTE_UNARY_WITH_FALLBACK(hprod, a)
ENOKI_ROUTE_UNARY_WITH_FALLBACK(hsum, a)
ENOKI_ROUTE_UNARY_WITH_FALLBACK(hmin_nested, a)
ENOKI_ROUTE_UNARY_WITH_FALLBACK(hmax_nested, a)
ENOKI_ROUTE_UNARY_WITH_FALLBACK(hprod_nested, a)
ENOKI_ROUTE_UNARY_WITH_FALLBACK(hsum_nested, a)

ENOKI_ROUTE_UNARY_WITH_FALLBACK(sqrt,  std::sqrt(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(floor, std::floor(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(ceil,  std::ceil(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(round, std::rint(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(exp,   std::exp(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(log,   std::log(a))

ENOKI_ROUTE_UNARY_WITH_FALLBACK(sin,   std::sin(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(cos,   std::cos(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(tan,   std::tan(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(csc,   Arg(1) / std::sin(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(sec,   Arg(1) / std::cos(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(cot,   Arg(1) / std::tan(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(asin,  std::asin(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(acos,  std::acos(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(atan,  std::atan(a))

ENOKI_ROUTE_UNARY_WITH_FALLBACK(sinh,  std::sinh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(cosh,  std::cosh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(tanh,  std::tanh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(csch,  Arg(1) / std::sinh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(sech,  Arg(1) / std::cosh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(coth,  Arg(1) / std::tanh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(asinh, std::asinh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(acosh, std::acosh(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(atanh, std::atanh(a))

ENOKI_ROUTE_UNARY_WITH_FALLBACK(erf,   std::erf(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(isnan, std::isnan(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(isinf, std::isinf(a))
ENOKI_ROUTE_UNARY_WITH_FALLBACK(isfinite, std::isfinite(a))

template <typename Type1, size_t Size1, bool Approx1, RoundingMode Mode1,
          typename Derived1, typename Type2, size_t Size2, bool Approx2,
          RoundingMode Mode2, typename Derived2>
ENOKI_INLINE auto operator&&(
    const StaticArrayBase<Type1, Size1, Approx1, Mode1, Derived1> &a1,
    const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a2) {
    return a1.derived() & a2.derived();
}

template <typename Type1, size_t Size1, bool Approx1, RoundingMode Mode1,
          typename Derived1, typename Type2, size_t Size2, bool Approx2,
          RoundingMode Mode2, typename Derived2>
ENOKI_INLINE auto
operator||(const StaticArrayBase<Type1, Size1, Approx1, Mode1, Derived1> &a1,
           const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a2) {
    return a1.derived() | a2.derived();
}

template <typename Type, bool Approx, RoundingMode Mode, typename Derived, size_t Size>
ENOKI_INLINE auto operator!(
    const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a) {
    return ~a.derived();
}

/// Arithmetic AND operator involving an array and a mask
template <typename Array, typename Mask, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<!std::is_same<Array, typename Array::Mask>::value &&
                            std::is_same<Mask, typename Array::Mask>::value, int> = 0>
ENOKI_INLINE typename Array::Expr operator&(const Array &a1, const Mask &a2) {
    return a1.derived().and_(a2);
}

/// Arithmetic OR operator involving an array and a mask
template <typename Array, typename Mask, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<!std::is_same<Array, typename Array::Mask>::value &&
                            std::is_same<Mask, typename Array::Mask>::value, int> = 0>
ENOKI_INLINE typename Array::Expr operator|(const Array &a1, const Mask &a2) {
    return a1.derived().or_(a2);
}

template <typename Array, typename Mask, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<!std::is_same<Array, typename Array::Mask>::value &&
                            std::is_same<Mask, typename Array::Mask>::value, int> = 0>
ENOKI_INLINE typename Array::Expr operator^(const Array &a1, const Mask &a2) {
    return a1.derived().xor_(a2);
}

template <typename Arg, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_signed<Arg>::value, int> = 0>
ENOKI_INLINE Arg abs(const Arg &a) { return std::abs(a); }

template <typename Arg, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<!std::is_signed<Arg>::value, int> = 0>
ENOKI_INLINE Arg abs(const Arg &a) { return a; }

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE bool eq(const Arg &a1, const Arg &a2) { return a1 == a2; }

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE bool neq(const Arg &a1, const Arg &a2) { return a1 != a2; }

/// Equality operator
template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename Arg,
          std::enable_if_t<Arg::Derived::Size == Derived::Size, int> = 0>
ENOKI_INLINE bool operator==(
    const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1,
    const Arg &a2) {
    return all_nested(eq(a1.derived(), a2.derived()));
}

/// Inequality operator
template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename Arg,
          std::enable_if_t<Arg::Derived::Size == Derived::Size, int> = 0>
ENOKI_INLINE bool operator!=(
    const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1,
    const Arg &a2) {
    return any_nested(neq(a1.derived(), a2.derived()));
}

ENOKI_ROUTE_BCAST(operator==)
ENOKI_ROUTE_BCAST(operator!=)

template <typename Value, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived1, typename Derived2,
          std::enable_if_t<Derived1::Size == Derived2::Size, int> = 0>
ENOKI_INLINE auto
dot(const StaticArrayBase<Value, Size, Approx, Mode, Derived1> &a1,
    const StaticArrayBase<Value, Size, Approx, Mode, Derived2> &a2) {
    return a1.derived().dot_(a2.derived());
}

template <typename Value1, typename Value2, size_t Size1, size_t Size2,
          bool Approx1, bool Approx2, RoundingMode Mode1, RoundingMode Mode2,
          typename Derived1, typename Derived2,
          std::enable_if_t<Derived1::Size == Derived2::Size, int> = 0>
ENOKI_INLINE auto
dot(const StaticArrayBase<Value1, Size1, Approx1, Mode1, Derived1> &a1,
    const StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a2) {
    return hsum(a1.derived() * a2.derived());
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg dot(const Arg &a, const Arg &b) { return a*b; }

template <typename Array1, typename Array2>
ENOKI_INLINE auto abs_dot(const Array1 &a1, const Array2 &a2) {
    return abs(dot(a1, a2));
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg fmadd(const Arg &a1, const Arg &a2, const Arg &a3) {
    return a1 * a2 + a3;
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg fmsub(const Arg &a1, const Arg &a2, const Arg &a3) {
    return a1 * a2 - a3;
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg fmsubadd(const Arg &a1, const Arg &a2, const Arg &a3) {
    return a1 * a2 + a3;
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg fmaddsub(const Arg &a1, const Arg &a2, const Arg &a3) {
    return a1 * a2 - a3;
}

template <typename Mask, typename Type, size_t Size, bool Approx,
          RoundingMode Mode, typename Derived1, typename Derived2,
          std::enable_if_t<Derived1::Size == Derived2::Size &&
                           Mask::Size == Derived1::Size, int> = 0>
ENOKI_INLINE auto
select(const Mask &a1,
       const StaticArrayBase<Type, Size, Approx, Mode, Derived1> &a2,
       const StaticArrayBase<Type, Size, Approx, Mode, Derived2> &a3) {
    return Derived1::select_(a1.derived(), a2.derived(), a3.derived());
}

template <typename Mask, typename Type, size_t Size, bool Approx,
          RoundingMode Mode, typename Derived1, typename Derived2,
          std::enable_if_t<Derived1::Size == Derived2::Size &&
                           Mask::Size != Derived1::Size, int> = 0>
ENOKI_INLINE auto
select(const Mask &a1,
       const StaticArrayBase<Type, Size, Approx, Mode, Derived1> &a2,
       const StaticArrayBase<Type, Size, Approx, Mode, Derived2> &a3) {
    return Derived1::select_(typename Derived1::Mask(a1.derived()),
                             a2.derived(), a3.derived());
}

template <typename Type1, typename Type2, typename Type3, size_t Size1,
          size_t Size2, size_t Size3, bool Approx1, bool Approx2,
          bool Approx3, RoundingMode Mode1, RoundingMode Mode2,
          RoundingMode Mode3, typename Derived1, typename Derived2,
          typename Derived3,
          std::enable_if_t<Derived1::Size == Derived2::Size &&
                           Derived2::Size == Derived3::Size, int> = 0,
          typename Value = decltype(select(std::declval<Type1>(),
                                            std::declval<Type2>(),
                                            std::declval<Type3>()))>
ENOKI_INLINE auto select(
    const StaticArrayBase<Type1, Size1, Approx1, Mode1, Derived1> &a1,
    const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a2,
    const StaticArrayBase<Type3, Size3, Approx3, Mode3, Derived3> &a3) {
    Array<Value, Derived1::Size> r;
    ENOKI_CHKSCALAR for (size_t i = 0; i < Derived1::Size; ++i)
        r.coeff(i) = select(a1.derived().coeff(i),
                            a2.derived().coeff(i),
                            a3.derived().coeff(i));
    return r;
}

template <typename Arg>
ENOKI_INLINE Arg select(bool a1, const Arg &a2, const Arg &a3) {
    return a1 ? a2 : a3;
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
ENOKI_INLINE auto
operator<<(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a,
           size_t value) {
    return a.derived().sl_(value);
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
ENOKI_INLINE auto
operator>>(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a,
           size_t value) {
    return a.derived().sr_(value);
}

template <size_t Imm, typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto sli(const Array &a) { return a.template sli_<Imm>(); }

template <size_t Imm, typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg sli(const Arg &a) { return a << Imm; }

template <size_t Imm, typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto sri(const Array &a) { return a.template sri_<Imm>(); }

template <size_t Imm, typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg sri(const Arg &a) { return a >> Imm; }

template <size_t Imm, typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto roli(const Array &a) { return a.template roli_<Imm>(); }

template <size_t Imm, typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg roli(const Arg &a) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a << (Imm & mask)) | (a >> ((~Imm + 1) & mask));
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg rol(const Arg &a, const Arg &amt) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a << (amt & mask)) | (a >> ((~amt + 1) & mask));
}

template <size_t Imm, typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto rori(const Array &a) { return a.template rori_<Imm>(); }

template <size_t Imm, typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg rori(const Arg &a) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a >> (Imm & mask)) | (a << ((~Imm + 1) & mask));
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg ror(const Arg &a, const Arg &amt) {
    size_t mask = 8 * sizeof(Arg) - 1;
    return (a >> (amt & mask)) | (a << ((~amt + 1) & mask));
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
ENOKI_INLINE auto
rol(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1, size_t a2) {
    return a1.derived().rol_(a2);
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived,
          std::enable_if_t<Derived::Value::Size != Size, int> = 0>
ENOKI_INLINE auto
rol(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1,
    const typename Derived::Value &a2) {
    return rol(a1.derived(), typename Derived::Expr(a2));
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
ENOKI_INLINE auto
ror(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1, size_t a2) {
    return a1.derived().ror_(a2);
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived,
          std::enable_if_t<Derived::Value::Size != Size, int> = 0>
ENOKI_INLINE auto
ror(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1,
    const typename Derived::Value &a2) {
    return ror(a1.derived(), typename Derived::Expr(a2));
}

template <typename Target, typename Type, size_t Size, bool Approx,
          RoundingMode Mode, typename Derived>
ENOKI_INLINE Target
reinterpret_array(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a) {
    return Target(a.derived(), detail::reinterpret_flag());
}

template <typename Target, typename Arg, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<!std::is_same<Target, bool>::value, int> = 0>
ENOKI_INLINE Target reinterpret_array(const Arg &a) {
    return memcpy_cast<Target>(a);
}

template <typename Target, typename Arg, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_same<Target, bool>::value, int> = 0>
ENOKI_INLINE Target reinterpret_array(const Arg &a) {
    using Int = typename detail::type_chooser<sizeof(Arg)>::Int;
    return memcpy_cast<Int>(a) != 0;
}

/// Return the element-wise maximum of two arrays (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg max(const Arg &a1, const Arg &a2) {
    return std::max(a1, a2);
}

/// Return the element-wise minimum of two arrays (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg min(const Arg &a1, const Arg &a2) {
    return std::min(a1, a2);
}

template <bool = false, typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
ENOKI_INLINE auto
rcp(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a) {
    return a.derived().rcp_();
}

/// Reciprocal (scalar fallback)
template <bool ForceApprox = false, typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg rcp(const Arg &a) {
#if defined(__AVX512ER__)
    if (std::is_same<Arg, float>::value) {
        __m128 v = _mm_set_ss((float) a);
        return Arg(_mm_cvtss_f32(_mm_rcp28_ss(v, v))); /* rel error < 2^-28 */
    }
#endif

#if defined(__SSE4_2__)
    if (ForceApprox && std::is_same<Arg, float>::value) {
        float v = (float) a;
        __m128 v_ = _mm_set_ss(v), r_;

        #if defined(__AVX512F__)
            r_ = _mm_rcp14_ss(v_, v_); /* rel error < 2^-14 */
        #else
            r_ = _mm_rcp_ss(v_);       /* rel error < 1.5*2^-12 */
        #endif

        float r = _mm_cvtss_f32(r_);

        /* Refine using one Newton-Raphson iteration */
        const float c0 = r + r, c1 = r * r;
        r = c0 - c1 * v;

        return Arg(r);
    }
#endif

#if defined(__AVX512F__) || defined(__AVX512ER__)
    if (ForceApprox && std::is_same<Arg, double>::value) {
        double v = (double) a;
        __m128d v_ = _mm_set_sd((double) v), r_;

        #if defined(__AVX512ER__)
            r_ = _mm_rcp28_sd(v_, v_); /* rel error < 2^-18 */
        #elif defined(__AVX512F__)
            r_ = _mm_rcp14_sd(v_, v_); /* rel error < 2^-14 */
        #endif

        double r = _mm_cvtsd_f64(r_);

        #if !defined(__AVX512ER__)
            /* Newton-Raphson iteration 1 */ {
                const double c0 = r + r, c1 = r * r;
                r = c0 - c1 * v;
            }
            /* Newton-Raphson iteration 2 */ {
                const double c0 = r + r, c1 = r * r;
                r = c0 - c1 * v;
            }
        #endif

        return Arg(r);
    }
#endif

    return Arg(1) / a;
}

template <bool = false, typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived>
ENOKI_INLINE auto
rsqrt(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a) {
    return a.derived().rsqrt_();
}

/// Reciprocal square root (scalar fallback)
template <bool ForceApprox = false, typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg rsqrt(const Arg &a) {
#if defined(__AVX512ER__)
    if (std::is_same<Arg, float>::value) {
        __m128 v = _mm_set_ss((float) a);
        return Arg(_mm_cvtss_f32(_mm_rsqrt28_ss(v, v))); /* rel error < 2^-28 */
    }
#endif

#if defined(__SSE4_2__)
    if (ForceApprox && std::is_same<Arg, float>::value) {
        float v = (float) a;
        __m128 v_ = _mm_set_ss(v), r_;

        #if defined(__AVX512F__)
            r_ = _mm_rsqrt14_ss(v_, v_); /* rel error < 2^-14 */
        #else
            r_ = _mm_rsqrt_ss(v_);       /* rel error < 1.5*2^-12 */
        #endif

        float r = _mm_cvtss_f32(r_);

        const float c0 = -.5f, c1 = -3.f;
        /* Refine using one Newton-Raphson iteration */
        float r2 = r * r, rh = r * c0;
        r = rh * (r2 * v + c1);

        return Arg(r);
    }
#endif

#if defined(__AVX512F__) || defined(__AVX512ER__)
    if (ForceApprox && std::is_same<Arg, double>::value) {
        double v = (double) a;
        __m128d v_ = _mm_set_sd((double) v), r_;

        #if defined(__AVX512ER__)
            r_ = _mm_rsqrt28_sd(v_, v_); /* rel error < 2^-18 */
        #elif defined(__AVX512F__)
            r_ = _mm_rsqrt14_sd(v_, v_); /* rel error < 2^-14 */
        #endif

        double r = _mm_cvtsd_f64(r_);

        #if !defined(__AVX512ER__)
            const double c0 = -.5, c1 = -3.;
            /* Newton-Raphson iteration 1 */ {
                double r2 = r * r, rh = r * c0;
                r = rh * (r2 * v + c1);
            }
            /* Newton-Raphson iteration 2 */ {
                double r2 = r * r, rh = r * c0;
                r = rh * (r2 * v + c1);
            }
        #endif

        return Arg(r);
    }
#endif

    return Arg(1) / std::sqrt(a);
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename Arg,
          std::enable_if_t<detail::bcast<Derived, Arg>::value &&
          std::is_floating_point<typename Derived::Scalar>::value, int> = 0>
ENOKI_INLINE auto operator/(
    const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a1,
    const Arg &a2) {
    if (Derived::Approx) /* Fast approximate division using reciprocals */
        return a1.derived() * rcp<true>(a2);
    else
        return a1.derived() / typename Derived::Expr(a2);
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode,
          typename Derived, typename Arg,
          std::enable_if_t<detail::bcast<Derived, Arg>::value &&
          std::is_floating_point<typename Derived::Scalar>::value , int> = 0>
ENOKI_INLINE auto operator/(
    const Arg &a1,
    const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a2) {
    return operator/(typename Derived::Expr(a1), a2.derived());
}

/// Multiply by integer power of 2 (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg ldexp(const Arg &a1, const Arg &a2) {
    return std::ldexp(a1, (int) a2);
}

/// Power function (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg pow(const Arg &a1, const Arg &a2) {
    return std::pow(a1, a2);
}

/// Arc tangent function (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg atan2(const Arg &a1, const Arg &a2) {
    return std::atan2(a1, a2);
}

/// Shuffle the entries of an array
template <size_t... Args, typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto shuffle(const Array &in) { return in.derived().template shuffle_<Args...>(); }

/// Shuffle the entries of an array (scalar fallback)
template <size_t Index, typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg shuffle(const Arg &arg) {
    static_assert(Index == 0, "Invalid argument to shuffle");
    return arg;
}

template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto erfi(const Array &a) {
    return a.erfi_();
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg erfi(const Arg &x) {
    // Based on "Approximating the erfi function" by Mark Giles
    Arg w = -log((Arg(1) - x) * (Arg(1) + x));
    Arg w1 = w - Arg(2.5);
    Arg w2 = sqrt(w) - Arg(3);

    Arg p1 = Arg(2.81022636e-08);
    Arg p2 = Arg(-0.000200214257);
    p1 = fmadd(p1, w1, Arg(3.43273939e-07));
    p2 = fmadd(p2, w2, Arg(0.000100950558));
    p1 = fmadd(p1, w1, Arg(-3.5233877e-06));
    p2 = fmadd(p2, w2, Arg(0.00134934322));
    p1 = fmadd(p1, w1, Arg(-4.39150654e-06));
    p2 = fmadd(p2, w2, Arg(-0.00367342844));
    p1 = fmadd(p1, w1, Arg(0.00021858087));
    p2 = fmadd(p2, w2, Arg(0.00573950773));
    p1 = fmadd(p1, w1, Arg(-0.00125372503));
    p2 = fmadd(p2, w2, Arg(-0.0076224613));
    p1 = fmadd(p1, w1, Arg(-0.00417768164));
    p2 = fmadd(p2, w2, Arg(0.00943887047));
    p1 = fmadd(p1, w1, Arg(0.246640727));
    p2 = fmadd(p2, w2, Arg(1.00167406));
    p1 = fmadd(p1, w1, Arg(1.50140941));
    p2 = fmadd(p2, w2, Arg(2.83297682));

    return select(w < 5, p1, p2) * x;
}

NAMESPACE_BEGIN(detail)
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE typename Array::Expr sign_mask(const Array &a) {
    using UInt = typename uint_array_t<Array>::Scalar;
    using Float = scalar_t<Array>;
    const Float mask = memcpy_cast<Float>(UInt(1) << (sizeof(UInt) * 8 - 1));
    return a.derived() & expr_t<Array>(mask);
}
NAMESPACE_END(detail)

template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE expr_t<Array> sign(const Array &a) {
    using Expr = expr_t<Array>;
    using Scalar = scalar_t<Array>;

    if (!std::is_signed<Scalar>::value)
        return Expr(1);
    else if (std::is_floating_point<Scalar>::value)
        return detail::sign_mask(a) | Expr(Scalar(1));
    else
        return select(a < Scalar(0), Expr(Scalar(-1)), Expr(Scalar(1)));
}

template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE expr_t<Array> copysign(const Array &a, const Array &b) {
    using Scalar = scalar_t<Array>;

    if (!std::is_signed<Scalar>::value) {
        return a;
    } else if (std::is_floating_point<Scalar>::value) {
        return abs(a) | detail::sign_mask(b);
    } else {
        auto abs_a = abs(a);
        return select(b > 0, abs_a, -abs_a);
    }
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
inline Arg sign(const Arg &a) {
    return std::copysign(Arg(1), a);
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
inline Arg copysign(const Arg &a, const Arg &b) {
    return std::copysign(a, b);
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
inline std::pair<Arg, Arg> sincos(const Arg &a) {
    return std::make_pair(std::sin(a), std::cos(a));
}

template <typename Arg, enable_if_notarray_t<Arg> = 0>
inline std::pair<Arg, Arg> sincosh(const Arg &a) {
    return std::make_pair(std::sinh(a), std::cosh(a));
}

/// Break floating-point number into normalized fraction and power of 2 (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE std::pair<Arg, Arg> frexp(const Arg &a) {
    int tmp;
    Arg result = std::frexp(a, &tmp);
    return std::make_pair(result, Arg(tmp));
}

ENOKI_INLINE bool all(bool value) { return value; }
ENOKI_INLINE bool all_nested(bool value) { return value; }
ENOKI_INLINE bool any(bool value) { return value; }
ENOKI_INLINE bool any_nested(bool value) { return value; }
ENOKI_INLINE bool none(bool value) { return !value; }
ENOKI_INLINE bool none_nested(bool value) { return !value; }
ENOKI_INLINE size_t count(bool value) { return value ? 1 : 0; }
ENOKI_INLINE size_t count_nested(bool value) { return value ? 1 : 0; }

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Initialization, loading/writing data
// -----------------------------------------------------------------------

/// Construct a zero-initialized array
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE Array zero() {
    return Array::zero_();
}

/// Construct a zero-initialized array
template <typename Array, enable_if_darray_t<Array> = 0>
ENOKI_INLINE Array zero(size_t size) {
    return Array::zero_(size);
}

/// Construct a zero-initialized array (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg zero() {
    return Arg(0);
}

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

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE Array index_sequence() {
    return detail::index_sequence_<Array>(
        std::make_index_sequence<Array::Size>());
}

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_darray_t<Array> = 0>
ENOKI_INLINE Array index_sequence(size_t size) {
    return Array::index_sequence_(size);
}

/// Construct an index sequence, i.e. 0, 1, 2, .. (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg index_sequence() {
    return Arg(0);
}

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE Array linspace(value_t<Array> min,
                            value_t<Array> max) {
    return detail::linspace_<Array>(
        std::make_index_sequence<Array::Size>(), min,
        (max - min) / (value_t<Array>)(Array::Size - 1));
}

/// Construct an index sequence, i.e. 0, 1, 2, ..
template <typename Array, enable_if_darray_t<Array> = 0,
          typename Value = value_t<Array>>
ENOKI_INLINE Array linspace(size_t size, Value min, Value max) {
    return Array::linspace_(size, min, max);
}

/// Construct an index sequence, i.e. 0, 1, 2, .. (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg linspace() {
    return Arg(0);
}

/// Load an array from aligned memory
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE Array load(const void *mem) {
    return Array::load_(mem);
}

/// Load an array from aligned memory (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg load(const void *mem) {
    assert((uintptr_t) mem % alignof(Arg) == 0);
    return *static_cast<const Arg *>(mem);
}

/// Load an array from unaligned memory
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE Array load_unaligned(const void *mem) {
    return Array::load_unaligned_(mem);
}

/// Load an array from unaligned memory (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE Arg load_unaligned(const void *mem) {
    return *static_cast<const Arg *>(mem);
}

/// Store an array to aligned memory
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE void store(void *mem, const Array &a) {
    a.store_(mem);
}

/// Store an array to aligned memory (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE void store(void *mem, const Arg &a) {
    assert((uintptr_t) mem % alignof(Arg) == 0);
    *static_cast<Arg *>(mem) = a;
}

/// Store an array to unaligned memory
template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE void store_unaligned(void *mem, const Array &a) {
    a.store_unaligned_(mem);
}

/// Store an array to unaligned memory (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE void store_unaligned(void *mem, const Arg &a) {
    *static_cast<Arg *>(mem) = a;
}

/// Prefetch operation
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          bool Write = false, size_t Level = 2, typename Index, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                           Index::Size == Array::Size, int> = 0>
ENOKI_INLINE void prefetch(const void *mem, const Index &index) {
    Array::template prefetch_<Stride, Write, Level>(mem, index);
}

/// Prefetch operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), bool Write = false,
          size_t Level = 2, typename Index, enable_if_notarray_t<Arg> = 0,
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
          enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                           Index::Size == Array::Size &&
                           Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE void prefetch(const void *mem, const Index &index,
                           const Mask &mask) {
    Array::template prefetch_<Stride, Write, Level>(mem, index, mask);
}

/// Masked prefetch operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), bool Write = false,
          size_t Level = 2, typename Index, enable_if_notarray_t<Arg> = 0,
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
          typename Index, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                           Index::Size == Array::Size, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index) {
    return Array::template gather_<Stride>(mem, index);
}

/// Gather operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg),
          typename Index, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0>
ENOKI_INLINE Arg gather(const void *mem, const Index &index) {
    return *((const Arg *) ((const uint8_t *) mem + index * Index(Stride)));
}

/// Masked gather operation
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Mask, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                           Index::Size == Array::Size &&
                           Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE Array gather(const void *mem, const Index &index,
                          const Mask &mask) {
    return Array::template gather_<Stride>(mem, index, mask);
}

/// Masked gather operation (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg),
          typename Index, typename Mask, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0>
ENOKI_INLINE Arg gather(const void *mem, const Index &index, const Mask &mask) {
    return detail::mask_active(mask)
               ? *((const Arg *) ((const uint8_t *) mem +
                                  index * Index(Stride))) : Arg(0);
}

/// Scatter operation
template <size_t Stride_ = 0, typename Array,
          typename Index, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                           Index::Size == Array::Size, int> = 0>
ENOKI_INLINE void scatter(void *mem, const Array &value, const Index &index) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(value_t<Array>);
    value.template scatter_<Stride>(mem, index);
}

/// Scatter operation (scalar fallback)
template <size_t Stride_ = 0, typename Arg,
          typename Index, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0>
ENOKI_INLINE void scatter(void *mem, const Arg &value, const Index &index) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(Arg);
    auto ptr = (Arg *) ((uint8_t *) mem + index * Index(Stride));
    *ptr = value;
}

/// Masked scatter operation
template <size_t Stride_ = 0, typename Array, typename Index, typename Mask,
          enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                           Index::Size == Array::Size &&
                           Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE void scatter(void *mem, const Array &value, const Index &index,
                          const Mask &mask) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(value_t<Array>);
    value.template scatter_<Stride>(mem, index, mask);
}

/// Masked scatter operation (scalar fallback)
template <size_t Stride_ = 0, typename Arg, typename Index,
          enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0, typename Mask>
ENOKI_INLINE void scatter(void *mem, const Arg &value, const Index &index, const Mask &mask) {
    constexpr size_t Stride = (Stride_ != 0) ? Stride_ : sizeof(Arg);
    auto ptr = (Arg *) ((uint8_t *) mem + index * Index(Stride));
    if (detail::mask_active(mask))
        *ptr = value;
}

/// Combined gather-modify-scatter operation without conflicts
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Func, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                           Index::Size == Array::Size, int> = 0>
ENOKI_INLINE void transform(void *mem, const Index &index, const Func &func) {
    Array::template transform_<Stride>(mem, index, func);
}

/// Combined gather-modify-scatter operation without conflicts (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), typename Index, typename Func,
          enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0>
ENOKI_INLINE void transform(void *mem, const Index &index, const Func &func) {
    auto ptr = (Arg *) ((uint8_t *) mem + index * Index(Stride));
    *ptr = func(*ptr);
}

/// Combined gather-modify-scatter operation without conflicts
template <typename Array, size_t Stride = sizeof(scalar_t<Array>),
          typename Index, typename Func, typename Mask,
          enable_if_sarray_t<Array> = 0,
          std::enable_if_t<std::is_integral<typename Index::Value>::value &&
                               Index::Size == Array::Size && Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE void transform(void *mem, const Index &index,
                            const Func &func, const Mask &mask) {
    Array::template transform_<Stride>(mem, index, func, mask);
}

/// Combined gather-modify-scatter operation without conflicts (scalar fallback)
template <typename Arg, size_t Stride = sizeof(Arg), typename Index, typename Func,
          typename Mask, enable_if_notarray_t<Arg> = 0,
          std::enable_if_t<std::is_integral<Index>::value, int> = 0>
ENOKI_INLINE void transform(void *mem, const Index &index,
                            const Func &func, const Mask &mask) {
    auto ptr = (Arg *) ((uint8_t *) mem + index * Index(Stride));
    if (detail::mask_active(mask))
        *ptr = func(*ptr);
}

/// Compressing store operation
template <typename Array, typename Mask, enable_if_sarray_t<Array> = 0,
          std::enable_if_t<Mask::Size == Array::Size, int> = 0>
ENOKI_INLINE void store_compress(void *&mem, const Array &value, const Mask &mask) {
    value.store_compress_(mem, mask);
}

/// Compressing store operation (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0, typename Mask>
ENOKI_INLINE void store_compress(void *&mem, const Arg &value, const Mask &mask) {
    if (detail::mask_active(mask))
        *((Arg *&) mem)++ = value;
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name "Safe" functions that avoid domain errors due to rounding
// -----------------------------------------------------------------------

template <typename T> ENOKI_INLINE auto safe_sqrt(T a) {
    return sqrt(max(a, zero<T>()));
}

template <typename T> ENOKI_INLINE auto safe_asin(T a) {
    return asin(min(T(1), max(T(-1), a)));
}

template <typename T> ENOKI_INLINE auto safe_acos(T a) {
    return acos(min(T(1), max(T(-1), a)));
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Miscellaneous functions
// -----------------------------------------------------------------------

/// Extract the low elements from an array of even size
template <typename Array, enable_if_sarray_t<Array> = 0>
auto low(const Array &a) { return a.derived().low_(); }

/// Extract the high elements from an array of even size
template <typename Array, enable_if_sarray_t<Array> = 0>
auto high(const Array &a) { return a.derived().high_(); }

template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto norm(const Array &v) {
    return sqrt(dot(v, v));
}

template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto squared_norm(const Array &v) {
    return dot(v, v);
}

template <typename Array, enable_if_sarray_t<Array> = 0>
ENOKI_INLINE auto normalize(const Array &v) {
    return v * rsqrt<Array::Approx>(squared_norm(v));
}

template <typename Array1, typename Array2,
          enable_if_sarray_t<Array1> = 0,
          enable_if_sarray_t<Array2> = 0>
ENOKI_INLINE auto cross(const Array1 &v1, const Array2 &v2) {
    static_assert(Array1::Derived::Size == 3 && Array2::Derived::Size == 3,
                  "cross(): requires Size = 3");

    return fmsub(shuffle<1, 2, 0>(v1),  shuffle<2, 0, 1>(v2),
                 shuffle<2, 0, 1>(v1) * shuffle<1, 2, 0>(v2));
}

/// Generic range clamping function
template <typename Value>
Value clamp(Value value, Value min_, Value max_) {
    return max(min(value, max_), min_);
}

/// Access an array element by index
template <typename Array, enable_if_array_t<Array> = 0>
ENOKI_INLINE auto& array_coeff(Array &array, size_t index) {
    return array.coeff(index);
}

/// Access an array element by index (const)
template <typename Array, enable_if_array_t<Array> = 0>
ENOKI_INLINE const auto& array_coeff(const Array &array, size_t index) {
    return array.coeff(index);
}

/// Access an array element by index (scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE auto& array_coeff(Arg &array, size_t) {
    return array;
}

/// Access an array element by index (const, scalar fallback)
template <typename Arg, enable_if_notarray_t<Arg> = 0>
ENOKI_INLINE const auto& array_coeff(const Arg &array, size_t) {
    return array;
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

NAMESPACE_BEGIN(detail)

template <typename Return, size_t Offset, typename T, size_t... Index>
static ENOKI_INLINE Return slice(const T &value, std::index_sequence<Index...>) {
    return Return(value.coeff(Index + Offset)...);
}

NAMESPACE_END(detail)

template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>,
          std::enable_if_t<T::ActualSize == Return::ActualSize, int> = 0>
static ENOKI_INLINE Return head(const T &value) { return value; }

template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>,
          std::enable_if_t<T::ActualSize != Return::ActualSize, int> = 0>
static ENOKI_INLINE Return head(const T &value) {
    static_assert(Size <= array_size<T>::value, "Array size mismatch");
    return detail::slice<Return, 0>(value, std::make_index_sequence<Size>());
}

template <size_t Size, typename T,
          typename Return = Array<value_t<T>, Size, T::Approx, T::Mode>>
static ENOKI_INLINE Return tail(const T &value) {
    static_assert(Size <= array_size<T>::value, "Array size mismatch");
    return detail::slice<Return, T::Size - Size>(value, std::make_index_sequence<Size>());
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Routing functions for dynamic arrays
// -----------------------------------------------------------------------

template <typename T, std::enable_if_t<is_dynamic<T>::value && is_array<T>::value, int> = 0>
ENOKI_INLINE size_t dynamic_size(const T &value) { return value.dynamic_size_(); }

template <typename T, std::enable_if_t<!is_dynamic<T>::value, int> = 0>
ENOKI_INLINE size_t dynamic_size(const T &) { return 0; }

template <typename T, std::enable_if_t<is_dynamic<T>::value && is_array<T>::value, int> = 0>
ENOKI_INLINE void dynamic_resize(T &value, size_t size) { value.dynamic_resize_(size); }

template <typename T, std::enable_if_t<!is_dynamic<T>::value, int> = 0>
ENOKI_INLINE void dynamic_resize(T &, size_t) { }

template <typename T, std::enable_if_t<is_dynamic<T>::value && is_array<T>::value, int> = 0>
ENOKI_INLINE size_t packets(const T &value) { return value.packets_(); }

template <typename T, std::enable_if_t<!is_dynamic<T>::value, int> = 0>
ENOKI_INLINE size_t packets(const T &) { return 0; }

template <typename T, std::enable_if_t<is_dynamic<T>::value && is_array<T>::value, int> = 0>
ENOKI_INLINE auto packet(T &&value, size_t i) -> decltype(value.packet_(i)) { return value.packet_(i); }

template <typename T, std::enable_if_t<!is_dynamic<T>::value, int> = 0>
ENOKI_INLINE T packet(T &&value, size_t) { return value; }

template <typename T, std::enable_if_t<is_dynamic<T>::value && is_array<T>::value, int> = 0>
ENOKI_INLINE auto ref_wrap(T &&value) -> decltype(value.ref_wrap_()) { return value.ref_wrap_(); }

template <typename T, std::enable_if_t<!is_dynamic<T>::value, int> = 0>
ENOKI_INLINE T ref_wrap(T &&value) { return value; }

//! @}
// -----------------------------------------------------------------------

#define ENOKI_MASKED_OPERATOR(name, expr)                                      \
    template <typename Arg, enable_if_notarray_t<Arg> = 0>                     \
    ENOKI_INLINE void name(Arg &a, const Arg &b, bool m) {                     \
        if (m)                                                                 \
            a = expr;                                                          \
    }                                                                          \
    template <typename Array, enable_if_sarray_t<Array> = 0>                   \
    ENOKI_INLINE void name(Array &a, const Array &b,                           \
                           const typename Array::Mask &m) {                    \
        a.name##_(b, m);                                                       \
    }

ENOKI_MASKED_OPERATOR(madd, a + b)
ENOKI_MASKED_OPERATOR(msub, a - b)
ENOKI_MASKED_OPERATOR(mmul, a * b)
ENOKI_MASKED_OPERATOR(mdiv, a / b)
ENOKI_MASKED_OPERATOR(mor, a | b)
ENOKI_MASKED_OPERATOR(mand, a & b)
ENOKI_MASKED_OPERATOR(mxor, a ^ b)

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
    if (is_dynamic<typename T::Value>::value) {
        for (size_t i = 0; i < size; ++i)
            match &= check_shape_recursive(a.derived().coeff(i), shape + 1);
    } else {
        check_shape_recursive(a.derived().coeff(0), shape + 1);
    }
    return match;
}

template <typename T, std::enable_if_t<!is_array<T>::value, int> = 0>
ENOKI_INLINE void set_shape_recursive(T &, const size_t *) { }

template <typename T, std::enable_if_t<is_array<T>::value, int> = 0>
ENOKI_INLINE void set_shape_recursive(T &a, const size_t *shape) {
    size_t size = a.derived().size();
    a.resize_(*shape);
    if (is_dynamic<typename T::Value>::value) {
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

#undef ENOKI_ROUTE_BINARY
#undef ENOKI_ROUTE_BCAST
#undef ENOKI_ROUTE_BINARY_OPERATOR
#undef ENOKI_ROUTE_BINARY_FUNCTION
#undef ENOKI_ROUTE_BINARY_OPERATOR_BCAST
#undef ENOKI_ROUTE_BINARY_FUNCTION_BCAST
#undef ENOKI_ROUTE_COMPOUND_OPERATOR
#undef ENOKI_ROUTE_UNARY_WITH_FALLBACK

NAMESPACE_END(enoki)
