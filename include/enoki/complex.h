/*
    enoki/comples.h -- Complex number data structure

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array.h"

NAMESPACE_BEGIN(enoki)

template <typename Value_, bool Approx_ = detail::approx_default<Value_>::value>
struct Complex : StaticArrayImpl<Value_, 2, Approx_, RoundingMode::Default, Complex<Value_, Approx_>> {
    using Base = StaticArrayImpl<Value_, 2, Approx_, RoundingMode::Default, Complex<Value_, Approx_>>;
    using MaskType = Mask<Value_, 2, Approx_, RoundingMode::Default>;

    static constexpr bool CustomBroadcast = true; // This class provides a custom broadcast operator

    using typename Base::Value;
    using typename Base::Scalar;

    template <typename T, typename T2 = Complex>
    using ReplaceType = Complex<T,
        detail::is_std_float<scalar_t<T>>::value ? T2::Approx
                                                 : detail::approx_default<T>::value>;

    template <typename T, typename T2 = Value_,
              std::enable_if_t<broadcast<T>::value &&
                               std::is_default_constructible<T2>::value &&
                               std::is_constructible<T2, T>::value, int> = 0>
    ENOKI_INLINE Complex(const T &f) : Base(f, zero<Value>()) { }

    template <typename T>
    ENOKI_INLINE static Complex fill_(const T &value) { return Array<Value, 2>::fill_(value); }

    ENOKI_DECLARE_ARRAY(Base, Complex)
};

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> real(const Complex<T, Approx> &z) { return z.x(); }
template <typename T, bool Approx> ENOKI_INLINE expr_t<T> imag(const Complex<T, Approx> &z) { return z.y(); }

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> squared_norm(const Complex<T, Approx> &z) {
    return squared_norm(Array<expr_t<T>, 2>(z));
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> norm(const Complex<T, Approx> &z) {
    return norm(Array<expr_t<T>, 2>(z));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> normalize(const Complex<T, Approx> &q) {
    return enoki::normalize(Array<expr_t<T>, 2>(q));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> rcp(const Complex<T, Approx> &z) {
    auto scale = rcp<Approx>(squared_norm(z));
    return Complex<expr_t<T>, Approx>(
         real(z) * scale,
        -imag(z) * scale
    );
}

NAMESPACE_BEGIN(detail)

template <typename T0, typename T1, bool Approx0, bool Approx1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Complex<T, Approx0 && Approx1> complex_mul(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    using Base   = Array<T, 2>;
    Base z1_perm = shuffle<1, 0>(z1);
    Base z0_im   = shuffle<1, 1>(z0);
    Base z0_re   = shuffle<0, 0>(z0);
    return fmaddsub(z0_re, z1, z0_im * z1_perm);
}

template <typename T0, typename T1, bool Approx0, bool Approx1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Complex<T, Approx0 && Approx1> complex_div(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    return z0 * rcp(z1);
}

NAMESPACE_END(detail)

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto operator*(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    return detail::complex_mul(z0, z1);
}

template <typename T0, bool Approx>
ENOKI_INLINE auto operator*(const Complex<T0, Approx> &z0, const Complex<T0, Approx> &z1) {
    return detail::complex_mul(z0, z1);
}

template <typename T0, typename T1, bool Approx, std::enable_if_t<broadcast<T1>::value, int> = 0>
ENOKI_INLINE Complex<expr_t<T0, T1>, Approx> operator*(const Complex<T0, Approx> &z, const T1 &s) {
    return Array<expr_t<T0>, 2>(z) * fill<Array<scalar_t<T1>, 2>>(s);
}

template <typename T0, typename T1, bool Approx, std::enable_if_t<broadcast<T0>::value, int> = 0>
ENOKI_INLINE Complex<expr_t<T0, T1>, Approx> operator*(const T0 &s, const Complex<T1, Approx> &z) {
    return fill<Array<scalar_t<T0>, 2>>(s) * Array<expr_t<T1>, 2>(z);
}

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto operator/(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    return detail::complex_div(z0, z1);
}

template <typename T0, bool Approx>
ENOKI_INLINE auto operator/(const Complex<T0, Approx> &z0, const Complex<T0, Approx> &z1) {
    return detail::complex_div(z0, z1);
}

template <typename T0, typename T1, bool Approx, std::enable_if_t<broadcast<T1>::value, int> = 0>
ENOKI_INLINE Complex<expr_t<T0, T1>, Approx> operator/(const Complex<T0, Approx> &z, const T1 &s) {
    return Array<expr_t<T0>, 2>(z) / s;
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> conj(const Complex<T, Approx> &z) {
    const Complex<expr_t<T>> mask(0.f, -0.f);
    return z ^ mask;
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> abs(const Complex<T, Approx> &z) { return norm(z); }

template <typename T, bool Approx> Complex<expr_t<T>, Approx> exp(const Complex<T, Approx> &z) {
    auto exp_r = exp(real(z));
    auto sc_i = sincos(imag(z));
    return Complex<expr_t<T>, Approx>(exp_r * sc_i.second, exp_r * sc_i.first);
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> arg(const Complex<T, Approx> &z) {
    return atan2(imag(z), real(z));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> log(const Complex<T, Approx> &z) {
    return Complex<expr_t<T>>(0.5f * log(squared_norm(z)), arg(z));
}

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto pow(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    return exp(log(z0) * z1);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> sqrt(const Complex<T, Approx> &z) {
    auto sc = sincos(arg(z) * 0.5f);
    auto r = sqrt(abs(z));
    return Complex<expr_t<T>, Approx>(sc.second * r, sc.first * r);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> sin(const Complex<T, Approx> &z) {
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return Complex<expr_t<T>, Approx>(sc.first * sch.second, sc.second * sch.first);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> cos(const Complex<T, Approx> &z) {
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return Complex<expr_t<T>, Approx>(sc.second * sch.second, -(sc.first * sch.first));
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE std::pair<R, R> sincos(const Complex<T, Approx> &z) {
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return std::make_pair<R, R>(
        R(sc.first * sch.second, sc.second * sch.first),
        R(sc.second * sch.second, -(sc.first * sch.first))
    );
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> tan(const Complex<T, Approx> &z) {
    using R = Complex<expr_t<T>, Approx>;
    auto sc  = sincos(real(z));
    auto sch = sincosh(imag(z));
    return R(sc.first * sch.second, sc.second * sch.first)
         / R(sc.second * sch.second, -(sc.first * sch.first));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> asin(const Complex<T, Approx> &z) {
    using R = Complex<expr_t<T>, Approx>;
    auto tmp = log(R(-imag(z), real(z)) + sqrt(1 - z*z));
    return R(imag(tmp), -real(tmp));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> acos(const Complex<T, Approx> &z) {
    using R = Complex<expr_t<T>, Approx>;
    auto tmp = sqrt(1 - z*z);
    tmp = log(z + R(-imag(tmp), real(tmp)));
    return R(imag(tmp), -real(tmp));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> atan(const Complex<T, Approx> &z) {
    using R = Complex<expr_t<T>, Approx>;
    const R I(0.f, 1.f);
    auto tmp = log((I-z) / (I+z));
    return R(imag(tmp) * 0.5f, -real(tmp) * 0.5f);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> sinh(const Complex<T, Approx> &z) {
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return Complex<expr_t<T>, Approx>(sch.first * sc.second, sch.second * sc.first);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> cosh(const Complex<T, Approx> &z) {
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return Complex<expr_t<T>, Approx>(sch.second * sc.second, sch.first * sc.first);
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE std::pair<R, R> sincosh(const Complex<T, Approx> &z) {
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return std::make_pair<R, R>(
        R(sch.first * sc.second, sch.second * sc.first),
        R(sch.second * sc.second, sch.first * sc.first)
    );
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> tanh(const Complex<T, Approx> &z) {
    using R = Complex<expr_t<T>, Approx>;
    auto sc  = sincos(imag(z));
    auto sch = sincosh(real(z));
    return R(sch.first * sc.second, sch.second * sc.first) /
           R(sch.second * sc.second, sch.first * sc.first);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> asinh(const Complex<T, Approx> &z) {
    return log(z + sqrt(z*z + 1.f));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> acosh(const Complex<T, Approx> &z) {
    return log(z + sqrt(z*z - 1.f));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> atanh(const Complex<T, Approx> &z) {
    using R = Complex<expr_t<T>, Approx>;
    return log((R(1.f) + z) / (R(1.f) - z)) * R(0.5f);
}

template <typename T, bool Approx, std::enable_if_t<!is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Complex<T, Approx> &z) {
    os << z.x();
    os << (z.y() < 0 ? " - " : " + ") << abs(z.y()) << "i";
    return os;
}

template <typename T, bool Approx, std::enable_if_t<is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Complex<T, Approx> &z) {
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
