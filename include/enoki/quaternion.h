/*
    enoki/quaternion.h -- Quaternion data structure

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "complex.h"
#include "matrix.h"

NAMESPACE_BEGIN(enoki)

template <typename Value_, bool Approx_ = detail::approx_default<Value_>::value>
struct Quaternion : StaticArrayImpl<Value_, 4, Approx_, RoundingMode::Default, Quaternion<Value_, Approx_>> {
    using Base = StaticArrayImpl<Value_, 4, Approx_, RoundingMode::Default, Quaternion<Value_, Approx_>>;
    using MaskType = Mask<Value_, 4, Approx_, RoundingMode::Default>;

    static constexpr bool CustomBroadcast = true; // This class provides a custom broadcast operator

    using typename Base::Value;
    using typename Base::Scalar;

    template <typename T, typename T2 = Quaternion>
    using ReplaceType = Quaternion<T,
        detail::is_std_float<scalar_t<T>>::value ? T2::Approx
                                                 : detail::approx_default<T>::value>;

    template <typename T, typename T2 = Value_,
              std::enable_if_t<broadcast<T>::value &&
                               std::is_default_constructible<T2>::value &&
                               std::is_constructible<T2, T>::value, int> = 0>
    ENOKI_INLINE Quaternion(const T &f) : Base(zero<Value>(), zero<Value>(), zero<Value>(), f) { }

    template <typename Im, typename Re, std::enable_if_t<Im::Size == 3, int> = 0>
    ENOKI_INLINE Quaternion(const Im &im, const Re &re)
        : Base(im.x(), im.y(), im.z(), re) { }

    template <typename T>
    ENOKI_INLINE static Quaternion fill_(const T &value) { return Array<Value, 4>::fill_(value); }

    ENOKI_DECLARE_ARRAY(Base, Quaternion)
};

template <typename Quat, std::enable_if_t<Quat::IsQuaternion, int> = 0> ENOKI_INLINE Quat identity() {
    return Quat(0.f, 0.f, 0.f, 1.f);
}

template <typename T0, typename T1, bool Approx0, bool Approx1, typename T = expr_t<T0, T1>>
ENOKI_INLINE T dot(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    using Base = Array<T, 4>;
    return dot(Base(q0), Base(q1));
}

template <typename T0, bool Approx, typename T = expr_t<T0>>
ENOKI_INLINE T dot(const Quaternion<T0, Approx> &q0, const Quaternion<T0, Approx> &q1) {
    using Base = Array<T, 4>;
    return dot(Base(q0), Base(q1));
}

template <typename T, bool Approx> ENOKI_INLINE Quaternion<expr_t<T>, Approx> conj(const Quaternion<T, Approx> &q) {
    const Quaternion<expr_t<T>, Approx> mask(-0.f, -0.f, -0.f, 0.f);
    return q ^ mask;
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> squared_norm(const Quaternion<T, Approx> &q) {
    return enoki::squared_norm(Array<expr_t<T>, 4>(q));
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> norm(const Quaternion<T, Approx> &q) {
    return enoki::norm(Array<expr_t<T>, 4>(q));
}

template <typename T, bool Approx> ENOKI_INLINE Quaternion<expr_t<T>, Approx> normalize(const Quaternion<T, Approx> &q) {
    return enoki::normalize(Array<expr_t<T>, 4>(q));
}

template <typename T, bool Approx> ENOKI_INLINE Quaternion<expr_t<T>, Approx> rcp(const Quaternion<T, Approx> &q) {
    return conj(q) * (1 / squared_norm(q));
}

NAMESPACE_BEGIN(detail)

template <typename T0, typename T1, bool Approx0, bool Approx1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Quaternion<T, Approx0 && Approx1> quat_mul(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    using Base = Array<T, 4>;
    const Base sign_mask(0.f, 0.f, 0.f, -0.f);
    Base q0_xyzx = shuffle<0, 1, 2, 0>(q0);
    Base q0_yzxy = shuffle<1, 2, 0, 1>(q0);
    Base q1_wwwx = shuffle<3, 3, 3, 0>(q1);
    Base q1_zxyy = shuffle<2, 0, 1, 1>(q1);
    Base t1 = fmadd(q0_xyzx, q1_wwwx, q0_yzxy * q1_zxyy) ^ sign_mask;

    Base q0_zxyz = shuffle<2, 0, 1, 2>(q0);
    Base q1_yzxz = shuffle<1, 2, 0, 2>(q1);
    Base q0_wwww = shuffle<3, 3, 3, 3>(q0);
    Base t2 = fmsub(q0_wwww, q1, q0_zxyz * q1_yzxz);
    return t1 + t2;
}

template <typename T0, typename T1, bool Approx0, bool Approx1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Quaternion<T, Approx0 && Approx1> quat_div(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    return q0 * rcp(q1);
}

NAMESPACE_END(detail)

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto operator*(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    return detail::quat_mul(q0, q1);
}

template <typename T0, bool Approx>
ENOKI_INLINE auto operator*(const Quaternion<T0, Approx> &q0, const Quaternion<T0, Approx> &q1) {
    return detail::quat_mul(q0, q1);
}

template <typename T0, typename T1, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T0, T1>> operator*(const Quaternion<T0, Approx> &q, const T1 &s) {
    return Array<expr_t<T0>, 4>(q) * fill<Array<scalar_t<T1>, 4>>(s);
}

template <typename T0, typename T1, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T0, T1>> operator*(const T0 &s, const Quaternion<T1, Approx> &q) {
    return fill<Array<scalar_t<T0>, 4>>(s) * Array<expr_t<T1>, 4>(q);
}

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto operator/(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    return detail::quat_div(q0, q1);
}

template <typename T0, bool Approx>
ENOKI_INLINE auto operator/(const Quaternion<T0, Approx> &q0, const Quaternion<T0, Approx> &q1) {
    return detail::quat_div(q0, q1);
}

template <typename T0, typename T1, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T0, T1>> operator/(const Quaternion<T0, Approx> &q, const T1 &s) {
    return Array<expr_t<T0>, 4>(q) / s;
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> real(const Quaternion<T, Approx> &q) { return q.w(); }
template <typename T, bool Approx> ENOKI_INLINE auto imag(const Quaternion<T, Approx> &q) { return head<3>(q); }

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> abs(const Quaternion<T, Approx> &q) { return norm(q); }

template <typename T, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T>, Approx> exp(const Quaternion<T, Approx> &q) {
    auto qi    = imag(q);
    auto ri    = norm(qi);
    auto exp_w = exp(real(q));
    auto sc    = sincos(ri);

    return Quaternion<expr_t<T>, Approx>(qi * (sc.first * exp_w / ri),
                                 sc.second * exp_w);
}

template <typename T, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T>, Approx> log(const Quaternion<T, Approx> &q) {
    auto qi_n    = normalize(imag(q));
    auto rq      = norm(q);
    auto acos_rq = acos(real(q) / rq);
    auto log_rq  = log(rq);

    return Quaternion<expr_t<T>, Approx>(qi_n * acos_rq, log_rq);
}

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto pow(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    return exp(log(q0) * q1);
}

template <typename T, bool Approx>
Quaternion<expr_t<T>, Approx> sqrt(const Quaternion<T, Approx> &q) {
    auto ri = norm(imag(q));
    auto cs = sqrt(Complex<expr_t<T>, Approx>(real(q), ri));
    return Quaternion<expr_t<T>, Approx>(imag(q) * (rcp(ri) * imag(cs)), real(cs));
}

template <typename T, bool Approx, std::enable_if_t<!is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Quaternion<T, Approx> &q) {
    os << q.w();
    os << (q.x() < 0 ? " - " : " + ") << abs(q.x()) << "i";
    os << (q.y() < 0 ? " - " : " + ") << abs(q.y()) << "j";
    os << (q.z() < 0 ? " - " : " + ") << abs(q.z()) << "k";
    return os;
}

template <typename T, bool Approx, std::enable_if_t<is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Quaternion<T, Approx> &q) {
    os << "[";
    size_t size = q.x().size();
    for (size_t i = 0; i < size; ++i) {
        os << q.w().coeff(i);
        os << (q.x().coeff(i) < 0 ? " - " : " + ") << abs(q.x().coeff(i)) << "i";
        os << (q.y().coeff(i) < 0 ? " - " : " + ") << abs(q.y().coeff(i)) << "j";
        os << (q.z().coeff(i) < 0 ? " - " : " + ") << abs(q.z().coeff(i)) << "k";
        if (i + 1 < size)
            os << ",\n ";
    }
    os << "]";
    return os;
}

template <typename Matrix, bool Approx, typename T, typename Expr = expr_t<T>,
          std::enable_if_t<Matrix::Size == 4, int> = 0>
ENOKI_INLINE Matrix quat_to_matrix(const Quaternion<T, Approx> &q_) {
    auto q = q_ * scalar_t<T>(M_SQRT2);

    Expr xx = q.x() * q.x(), yy = q.y() * q.y(), zz = q.z() * q.z();
    Expr xy = q.x() * q.y(), xz = q.x() * q.z(), yz = q.y() * q.z();
    Expr xw = q.x() * q.w(), yw = q.y() * q.w(), zw = q.z() * q.w();

    return Matrix(
         1.f - (yy + zz), xy - zw, xz + yw, 0.f,
         xy + zw, 1.f - (xx + zz), yz - xw, 0.f,
         xz - yw, yz + xw, 1.f - (xx + yy), 0.f,
         0.f, 0.f, 0.f, 1.f
    );
}

template <typename Matrix, bool Approx, typename T, typename Expr = expr_t<T>,
          std::enable_if_t<Matrix::Size == 3, int> = 0>
ENOKI_INLINE Matrix quat_to_matrix(const Quaternion<T, Approx> &q_) {
    auto q = q_ * scalar_t<T>(M_SQRT2);

    Expr xx = q.x() * q.x(), yy = q.y() * q.y(), zz = q.z() * q.z();
    Expr xy = q.x() * q.y(), xz = q.x() * q.z(), yz = q.y() * q.z();
    Expr xw = q.x() * q.w(), yw = q.y() * q.w(), zw = q.z() * q.w();

    return Matrix(
         1.f - (yy + zz), xy - zw, xz + yw,
         xy + zw, 1.f - (xx + zz), yz - xw,
         xz - yw,  yz + xw, 1.f - (xx + yy)
    );
}

template <typename T, size_t Size, bool Approx,
          typename Expr = expr_t<T>,
          typename Quat = Quaternion<Expr, Approx>,
          std::enable_if_t<Size == 3 || Size == 4, int> = 0>
ENOKI_INLINE Quat matrix_to_quat(const Matrix<T, Size, Approx> &mat) {
    const Expr c0(0), c1(1), ch(0.5f);

    // Converting a Rotation Matrix to a Quaternion
    // Mike Day, Insomniac Games
    Expr t0(c1 + mat(0, 0) - mat(1, 1) - mat(2, 2));
    Quat q0(t0,
            mat(1, 0) + mat(0, 1),
            mat(0, 2) + mat(2, 0),
            mat(2, 1) - mat(1, 2));

    Expr t1(c1 - mat(0, 0) + mat(1, 1) - mat(2, 2));
    Quat q1(mat(1, 0) + mat(0, 1),
            t1,
            mat(2, 1) + mat(1, 2),
            mat(0, 2) - mat(2, 0));

    Expr t2(c1 - mat(0, 0) - mat(1, 1) + mat(2, 2));
    Quat q2(mat(0, 2) + mat(2, 0),
            mat(2, 1) + mat(1, 2),
            t2,
            mat(1, 0) - mat(0, 1));

    Expr t3(c1 + mat(0, 0) + mat(1, 1) + mat(2, 2));
    Quat q3(mat(2, 1) - mat(1, 2),
            mat(0, 2) - mat(2, 0),
            mat(1, 0) - mat(0, 1),
            t3);

    auto mask0 = mat(0, 0) > mat(1, 1);
    Expr t01 = select(mask0, t0, t1);
    Quat q01 = select(mask0, q0, q1);

    auto mask1 = mat(0, 0) < -mat(1, 1);
    Expr t23 = select(mask1, t2, t3);
    Quat q23 = select(mask1, q2, q3);

    auto mask2 = mat(2, 2) < c0;
    Expr t0123 = select(mask2, t01, t23);
    Quat q0123 = select(mask2, q01, q23);

    return q0123 * (rsqrt(t0123) * ch);
}

template <typename T0, typename T1, bool Approx0, bool Approx1, typename Float,
          typename E      = expr_t<T0, T1>,
          typename Return = Quaternion<E, Approx0 && Approx1>>
ENOKI_INLINE Return slerp(const Quaternion<T0, Approx0> &q0,
                          const Quaternion<T1, Approx1> &q1_, Float t) {
    using Base = Array<E, 4>;

    auto cos_theta = dot(q0, q1_);
    Return q1 = mulsign(Base(q1_), cos_theta);
    cos_theta = mulsign(cos_theta, cos_theta);

    auto theta = acos(cos_theta);
    auto sc = sincos(theta * t);
    auto close_mask = cos_theta > 0.9995f;

    Return qperp = normalize(q1 - q0 * cos_theta);
    Return result = q0 * sc.second + qperp * sc.first;

    if (ENOKI_UNLIKELY(any_nested(close_mask)))
        masked(result, close_mask) = normalize(q0 * (1.f - t) + q1 * t);

    return result;
}

template <typename Quat, typename Vector3, std::enable_if_t<Quat::IsQuaternion, int> = 0>
ENOKI_INLINE Quat rotate(const Vector3 &axis, const value_t<Quat> &angle) {
    auto sc = sincos(angle * .5f);
    return Quat(concat(axis * sc.first, sc.second));
}

NAMESPACE_END(enoki)
