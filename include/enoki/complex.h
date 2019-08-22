/*
    enoki/complex.h -- Complex number data structure

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2019 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>

NAMESPACE_BEGIN(enoki)

/// SFINAE helper for complex numbers
template <typename T> using is_complex_helper = enable_if_t<std::decay_t<T>::IsComplex>;
template <typename T> constexpr bool is_complex_v = is_detected_v<is_complex_helper, T>;
template <typename T> using enable_if_complex_t = enable_if_t<is_complex_v<T>>;
template <typename T> using enable_if_not_complex_t = enable_if_t<!is_complex_v<T>>;

template <typename Value_, bool Approx_>
struct Complex : StaticArrayImpl<Value_, 2, Approx_, RoundingMode::Default, false, Complex<Value_, Approx_>> {
    using Base = StaticArrayImpl<Value_, 2, Approx_, RoundingMode::Default, false, Complex<Value_, Approx_>>;
    ENOKI_ARRAY_IMPORT_BASIC(Base, Complex);
    using Base::operator=;

    static constexpr bool IsComplex = true;

    using ArrayType = Complex;
    using MaskType = Mask<Value_, 2, Approx_, RoundingMode::Default>;

    template <typename T>
    using ReplaceValue = Complex<T, is_std_float_v<scalar_t<T>> && is_std_float_v<scalar_t<Value_>>
                                    ? Approx_ : array_approx_v<T>>;

    Complex() = default;

    template <typename T, enable_if_complex_t<T> = 0>
    ENOKI_INLINE Complex(T&& z) : Base(z) { }

    template <typename T, enable_if_t<(array_depth_v<T> < Base::Depth && (is_scalar_v<T> || is_array_v<T>))> = 0,
                          enable_if_not_complex_t<T> = 0>
    ENOKI_INLINE Complex(T &&v) : Base(v, zero<Value_>()) { }

    template <typename T, enable_if_t<(array_depth_v<T> == Base::Depth || !(is_scalar_v<T> || is_array_v<T>))> = 0,
                          enable_if_not_complex_t<T> = 0>
    ENOKI_INLINE Complex(T &&v) : Base(std::forward<T>(v)) { }

    ENOKI_INLINE Complex(const Value_ &v1, const Value_ &v2) : Base(v1, v2) { }

};

template <typename T, enable_if_complex_t<T> = 0>
ENOKI_INLINE T identity() {
    return T(1.f, 0.f);
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> real(const Complex<T, Approx> &z) { return z.x(); }
template <typename T, bool Approx> ENOKI_INLINE expr_t<T> imag(const Complex<T, Approx> &z) { return z.y(); }

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> squared_norm(const Complex<T, Approx> &z) {
    return squared_norm(Array<expr_t<T>, 2, Approx>(z));
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> norm(const Complex<T, Approx> &z) {
    return norm(Array<expr_t<T>, 2, Approx>(z));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> normalize(const Complex<T, Approx> &q) {
    return enoki::normalize(Array<expr_t<T>, 2, Approx>(q));
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> rcp(const Complex<T, Approx> &z) {
    auto scale = rcp<Approx>(squared_norm(z));
    return Complex<expr_t<T>, Approx>(
         real(z) * scale,
        -imag(z) * scale
    );
}

template <typename T0, typename T1, bool Approx0, bool Approx1,
          typename Value = expr_t<T0, T1>, typename Result = Complex<Value, Approx0 && Approx1>>
ENOKI_INLINE Result operator*(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    using Base   = Array<Value, 2, Approx0 && Approx1>;
    Base z1_perm = shuffle<1, 0>(z1);
    Base z0_im   = shuffle<1, 1>(z0);
    Base z0_re   = shuffle<0, 0>(z0);
    return fmaddsub(z0_re, z1, z0_im * z1_perm);
}

template <typename T0, typename T1, bool Approx0,
          typename Value = expr_t<T0, T1>, typename Result = Complex<Value, Approx0>>
ENOKI_INLINE Result operator*(const Complex<T0, Approx0> &z0, const T1 &v1) {
    return Array<expr_t<T0>, 2, Approx0>(z0) * v1;
}

template <typename T0, typename T1, bool Approx1,
          typename Value = expr_t<T0, T1>, typename Result = Complex<Value, Approx1>>
ENOKI_INLINE Result operator*(const T0 &v0, const Complex<T1, Approx1> &z1) {
    return v0 * Array<expr_t<T0>, 2, Approx1>(z1);
}

template <typename T0, typename T1, bool Approx0, bool Approx1,
          typename Value = expr_t<T0, T1>, typename Result = Complex<Value, Approx0 && Approx1>>
ENOKI_INLINE Result operator/(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    return z0 * rcp(z1);
}

template <typename T0, typename T1, bool Approx0,
          typename Value = expr_t<T0, T1>, typename Result = Complex<Value, Approx0>>
ENOKI_INLINE Result operator/(const Complex<T0, Approx0> &z0, const T1 &v1) {
    return Array<expr_t<T0>, 2, Approx0>(z0) / v1;
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> conj(const Complex<T, Approx> &z) {
    const Complex<expr_t<T>> mask(0.f, -0.f);
    return z ^ mask;
}

template <typename T, bool Approx>
ENOKI_INLINE expr_t<T> abs(const Complex<T, Approx> &z) {
    return norm(z);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> exp(const Complex<T, Approx> &z) {
    auto exp_r = exp(real(z));
    auto [s, c] = sincos(imag(z));
    return { exp_r * c, exp_r * s };
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> log(const Complex<T, Approx> &z) {
    return { .5f * log(squared_norm(z)), arg(z) };
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> arg(const Complex<T, Approx> &z) {
    return atan2(imag(z), real(z));
}

template <typename T1, typename T2, typename Expr = expr_t<T1, T2>> std::pair<Expr, Expr>
sincos_arg_diff(const Complex<T1> &z1, const Complex<T2> &z2) {
    Expr normalization = rsqrt(squared_norm(z1) * squared_norm(z2));
    Complex<Expr> value = z1 * conj(z2) * normalization;
    return { imag(value), real(value) };
}

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto pow(const Complex<T0, Approx0> &z0, const Complex<T1, Approx1> &z1) {
    return exp(log(z0) * z1);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> sqrt(const Complex<T, Approx> &z) {
    auto [s, c] = sincos(arg(z) * .5f);
    auto r = sqrt(abs(z));
    return Complex<expr_t<T>, Approx>(c * r, s * r);
}

template <typename T, bool Approx = array_approx_v<T>>
ENOKI_INLINE Complex<expr_t<T>, Approx> sqrtz(const T &x) {
    auto r = sqrt(abs(x)), z = zero<T>();
    auto is_real = x >= 0;
    return { select(is_real, r, z), select(is_real, z, r) };
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> sin(const Complex<T, Approx> &z) {
    auto [s, c]   = sincos(real(z));
    auto [sh, ch] = sincosh(imag(z));
    return Complex<expr_t<T>, Approx>(s * ch, c * sh);
}

template <typename T, bool Approx> ENOKI_INLINE Complex<expr_t<T>, Approx> cos(const Complex<T, Approx> &z) {
    auto [s, c]   = sincos(real(z));
    auto [sh, ch] = sincosh(imag(z));
    return Complex<expr_t<T>, Approx>(c * ch, -s * sh);
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE std::pair<R, R> sincos(const Complex<T, Approx> &z) {
    auto [s, c]   = sincos(real(z));
    auto [sh, ch] = sincosh(imag(z));
    return std::make_pair<R, R>(
        R(s * ch, c * sh),
        R(c * ch, -s * sh)
    );
}

template <typename T, bool Approx>
ENOKI_INLINE Complex<expr_t<T>, Approx> tan(const Complex<T, Approx> &z) {
    auto [s, c] = sincos(z);
    return s / c;
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE R asin(const Complex<T, Approx> &z) {
    auto tmp = log(R(-imag(z), real(z)) + sqrt(1.f - z*z));
    return R(imag(tmp), -real(tmp));
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE R acos(const Complex<T, Approx> &z) {
    auto tmp = sqrt(1.f - z*z);
    tmp = log(z + R(-imag(tmp), real(tmp)));
    return R(imag(tmp), -real(tmp));
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE R atan(const Complex<T, Approx> &z) {
    const R I(0.f, 1.f);
    auto tmp = log((I-z) / (I+z));
    return R(imag(tmp) * .5f, -real(tmp) * .5f);
}

template <typename T, bool Approx>
ENOKI_INLINE Complex<expr_t<T>, Approx> sinh(const Complex<T, Approx> &z) {
    auto [s, c]  = sincos(imag(z));
    auto [sh, ch] = sincosh(real(z));
    return { sh * c, ch * s };
}

template <typename T, bool Approx>
ENOKI_INLINE Complex<expr_t<T>, Approx> cosh(const Complex<T, Approx> &z) {
    auto [s, c]   = sincos(imag(z));
    auto [sh, ch] = sincosh(real(z));
    return { ch * c, sh * s };
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE std::pair<R, R> sincosh(const Complex<T, Approx> &z) {
    auto [s, c] = sincos(imag(z));
    auto [sh, ch]  = sincosh(real(z));
    return std::make_pair<R, R>(
        R(sh * c, ch * s),
        R(ch * c, sh * s)
    );
}

template <typename T, bool Approx>
ENOKI_INLINE Complex<expr_t<T>, Approx> tanh(const Complex<T, Approx> &z) {
    auto [sh, ch] = sincosh(z);
    return sh / ch;
}

template <typename T, bool Approx>
ENOKI_INLINE Complex<expr_t<T>, Approx> asinh(const Complex<T, Approx> &z) {
    return log(z + sqrt(z*z + 1.f));
}

template <typename T, bool Approx>
ENOKI_INLINE Complex<expr_t<T>, Approx> acosh(const Complex<T, Approx> &z) {
    return log(z + sqrt(z*z - 1.f));
}

template <typename T, bool Approx, typename R = Complex<expr_t<T>, Approx>>
ENOKI_INLINE R atanh(const Complex<T, Approx> &z) {
    return log((R(1.f) + z) / (R(1.f) - z)) * R(.5f);
}

template <typename T, bool Approx, enable_if_not_array_t<T> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Complex<T, Approx> &z) {
    os << z.x();
    os << (z.y() < 0 ? " - " : " + ") << abs(z.y()) << "i";
    return os;
}

template <typename T, bool Approx, enable_if_array_t<T> = 0, enable_if_not_array_t<value_t<T>> = 0>
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
