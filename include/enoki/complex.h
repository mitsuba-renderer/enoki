/*
    enoki/comples.h -- Complex number data structure

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

template <typename Type_>
struct Complex
    : StaticArrayImpl<Type_, 2, detail::approx_default<Type_>::value,
                      RoundingMode::Default, Complex<Type_>> {
    using Type = Type_;
    using Base =
        StaticArrayImpl<Type, 2, detail::approx_default<Type>::value,
                        RoundingMode::Default, Complex<Type>>;

    template <typename T> using ReplaceType = Complex<T>;

    ENOKI_DECLARE_CUSTOM_ARRAY(Base, Complex)

    Complex(Type f) : Base(f, zero<Type>()) { }

    template <typename T = Type,
              std::enable_if_t<!std::is_same<T, scalar_t<T>>::value, int> = 0>
    Complex(scalar_t<T> f) : Base(f, zero<Type>()) { }

    ENOKI_ALIGNED_OPERATOR_NEW()
};

template <typename T> ENOKI_INLINE expr_t<T> real(const Complex<T> &z) { return z.x(); }
template <typename T> ENOKI_INLINE expr_t<T> imag(const Complex<T> &z) { return z.y(); }

template <typename T> ENOKI_INLINE Complex<expr_t<T>> rcp(const Complex<T> &z) {
    auto scale = rcp(squared_norm(z));
    return Complex<expr_t<T>>(
         real(z) * scale,
        -imag(z) * scale
    );
}

NAMESPACE_BEGIN(detail)

template <typename T0, typename T1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Complex<T> complex_mul(const Complex<T0> &z0, const Complex<T1> &z1) {
    using Base = Array<T, 2>;
    Base z1_perm = shuffle<1, 0>(z1);
    Base z0_im   = shuffle<1, 1>(z0);
    Base z0_re   = shuffle<0, 0>(z0);
    return fmaddsub(z0_re, z1, z0_im * z1_perm);
}

template <typename T0, typename T1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Complex<T> complex_div(const Complex<T0> &z0, const Complex<T1> &z1) {
    return z0 * rcp(z1);
}

NAMESPACE_END(detail)

template <typename T0, typename T1>
ENOKI_INLINE auto operator*(const Complex<T0> &z0, const Complex<T1> &z1) {
    return detail::complex_mul(z0, z1);
}

template <typename T0>
ENOKI_INLINE auto operator*(const Complex<T0> &z0, const Complex<T0> &z1) {
    return detail::complex_mul(z0, z1);
}

template <typename T0, typename T1>
ENOKI_INLINE Complex<expr_t<T0, T1>> operator*(const Complex<T0> &z, const T1 &s) {
    return Array<expr_t<T0>, 4>(z) * s;
}

template <typename T0, typename T1>
ENOKI_INLINE Complex<expr_t<T0, T1>> operator*(const T0 &s, const Complex<T1> &z) {
    return s * Array<expr_t<T1>, 4>(z);
}

template <typename T0, typename T1>
ENOKI_INLINE auto operator/(const Complex<T0> &z0, const Complex<T1> &z1) {
    return detail::complex_div(z0, z1);
}

template <typename T0>
ENOKI_INLINE auto operator/(const Complex<T0> &z0, const Complex<T0> &z1) {
    return detail::complex_div(z0, z1);
}

template <typename T0, typename T1>
ENOKI_INLINE Complex<expr_t<T0, T1>> operator/(const Complex<T0> &z, const T1 &s) {
    return Array<expr_t<T0>, 4>(z) / s;
}

template <typename T> ENOKI_INLINE Complex<expr_t<T>> conj(const Complex<T> &z) {
    const Complex<expr_t<T>> mask(0.f, -0.f);
    return z ^ mask;
}

template <typename T> ENOKI_INLINE expr_t<T> abs(const Complex<T> &z) { return norm(z); }

template <typename T> Complex<expr_t<T>> exp(const Complex<T> &z) {
    auto exp_r = exp(real(z));
    auto sc_i = sincos(imag(z));
    return Complex<expr_t<T>>(exp_r * sc_i.second, exp_r * sc_i.first);
}

template <typename T> expr_t<T> arg(const Complex<T> &z) {
    return atan2(imag(z), real(z));
}

template <typename T> Complex<expr_t<T>> log(const Complex<T> &z) {
    return Complex<expr_t<T>>(0.5f * log(squared_norm(z)), arg(z));
}

template <typename T0, typename T1>
auto pow(const Complex<T0> &z0, const Complex<T1> &z1) {
    return exp(log(z0) * z1);
}

template <typename T> Complex<expr_t<T>> sqrt(const Complex<T> &z) {
    auto sc = sincos(arg(z) * 0.5f);
    auto r = sqrt(abs(z));
    return Complex<expr_t<T>>(sc.second * r, sc.first * r);
}

template <typename T> Complex<expr_t<T>> sin(const Complex<T> &z) {
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return Complex<expr_t<T>>(sc.first * sch.second, sc.second * sch.first);
}

template <typename T> Complex<expr_t<T>> cos(const Complex<T> &z) {
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return Complex<expr_t<T>>(sc.second * sch.second, -(sc.first * sch.first));
}

template <typename T, typename R = Complex<expr_t<T>>>
std::pair<R, R> sincos(const Complex<T> &z) {
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return std::make_pair<R, R>(
        R(sc.first * sch.second, sc.second * sch.first),
        R(sc.second * sch.second, -(sc.first * sch.first))
    );
}

template <typename T> Complex<expr_t<T>> tan(const Complex<T> &z) {
    using R = Complex<expr_t<T>>;
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return R(sc.first * sch.second, sc.second * sch.first)
         / R(sc.second * sch.second, -(sc.first * sch.first));
}

template <typename T> Complex<expr_t<T>> asin(const Complex<T> &z) {
    using R = Complex<expr_t<T>>;
    auto tmp = log(R(-imag(z), real(z)) + sqrt(1 - z*z));
    return R(imag(tmp), -real(tmp));
}

template <typename T> Complex<expr_t<T>> acos(const Complex<T> &z) {
    using R = Complex<expr_t<T>>;
    auto tmp = sqrt(1 - z*z);
    tmp = log(z + R(-imag(tmp), real(tmp)));
    return R(imag(tmp), -real(tmp));
}

template <typename T> Complex<expr_t<T>> atan(const Complex<T> &z) {
    using R = Complex<expr_t<T>>;
    const R I(0.f, 1.f);
    auto tmp = log((I-z) / (I+z));
    return R(imag(tmp) * 0.5f, -real(tmp) * 0.5f);
}

template <typename T> Complex<expr_t<T>> sinh(const Complex<T> &z) {
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return Complex<expr_t<T>>(sch.first * sc.second, sch.second * sc.first);
}

template <typename T> Complex<expr_t<T>> cosh(const Complex<T> &z) {
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return Complex<expr_t<T>>(sch.second * sc.second, sch.first * sc.first);
}

template <typename T, typename R = Complex<expr_t<T>>>
std::pair<R, R> sincosh(const Complex<T> &z) {
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return std::make_pair<R, R>(
        R(sch.first * sc.second, sch.second * sc.first),
        R(sch.second * sc.second, sch.first * sc.first)
    );
}

template <typename T> Complex<expr_t<T>> tanh(const Complex<T> &z) {
    using R = Complex<expr_t<T>>;
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return
        R(sch.first * sc.second, sch.second * sc.first) /
        R(sch.second * sc.second, sch.first * sc.first);
}

template <typename T> Complex<expr_t<T>> asinh(const Complex<T> &z) {
    return log(z + sqrt(z*z + 1.f));
}

template <typename T> Complex<expr_t<T>> acosh(const Complex<T> &z) {
    return log(z + sqrt(z*z - 1.f));
}

template <typename T> Complex<expr_t<T>> atanh(const Complex<T> &z) {
    using R = Complex<expr_t<T>>;
    return log((R(1.f) + z) / (R(1.f) - z)) * R(0.5f);
}

template <typename T, std::enable_if_t<!is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Complex<T> &z) {
    os << z.x();
    os << (z.y() < 0 ? " - " : " + ") << abs(z.y()) << "i";
    return os;
}

template <typename T, std::enable_if_t<is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Complex<T> &z) {
    os << "[";
    size_t size = z.x().size();
    for (size_t i = 0; i < size; ++i) {
        os << z.x().coeff(i);
        os << (z.y().coeff(i) < 0 ? " - " : " + ") << abs(z.y().coeff(i)) << "i";
        if (i + 1 < size)
            os << ",\n ";
    }
    os << "]";
    return os;
}

NAMESPACE_END(enoki)
