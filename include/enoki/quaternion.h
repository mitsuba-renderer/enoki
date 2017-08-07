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

template <typename Type_>
struct Quaternion
    : StaticArrayImpl<Type_, 4, detail::approx_default<Type_>::value,
                      RoundingMode::Default, Quaternion<Type_>> {

    static constexpr bool IsQuaternion = true;

    using Type = Type_;
    using Base = StaticArrayImpl<Type, 4, detail::approx_default<Type>::value,
                                 RoundingMode::Default, Quaternion<Type>>;

    template <typename T> using ReplaceType = Quaternion<T>;

    ENOKI_DECLARE_CUSTOM_ARRAY(Base, Quaternion)

    Quaternion(Type f) : Base(zero<Type>(), zero<Type>(), zero<Type>(), f) { }

    template <typename T = Type,
              std::enable_if_t<!std::is_same<T, scalar_t<T>>::value, int> = 0>
    Quaternion(scalar_t<T> f) : Base(zero<Type>(), zero<Type>(), zero<Type>(), f) { }

    template <typename Array,
              std::enable_if_t<array_size<Array>::value == 3, int> = 0>
    Quaternion(const Array &imag, const Type &real)
        : Base(imag.x(), imag.y(), imag.z(), real) { }

    ENOKI_ALIGNED_OPERATOR_NEW()
};

template <typename Quat, std::enable_if_t<Quat::IsQuaternion, int> = 0> ENOKI_INLINE Quat identity() {
    return Quat(0.f, 0.f, 0.f, 1.f);
}

template <typename T0, typename T1, typename T = expr_t<T0, T1>>
ENOKI_INLINE T dot(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
    using Base = Array<T, 4>;
    return dot(Base(q0), Base(q1));
}

template <typename T0, typename T = expr_t<T0>>
ENOKI_INLINE T dot(const Quaternion<T0> &q0, const Quaternion<T0> &q1) {
    using Base = Array<T, 4>;
    return dot(Base(q0), Base(q1));
}

template <typename T> ENOKI_INLINE Quaternion<expr_t<T>> conj(const Quaternion<T> &q) {
    const Quaternion<expr_t<T>> mask(-0.f, -0.f, -0.f, 0.f);
    return q ^ mask;
}

template <typename T> ENOKI_INLINE Quaternion<expr_t<T>> rcp(const Quaternion<T> &q) {
    return conj(q) * (1 / squared_norm(q));
}

NAMESPACE_BEGIN(detail)

template <typename T0, typename T1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Quaternion<T> quat_mul(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
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

template <typename T0, typename T1, typename T = expr_t<T0, T1>>
ENOKI_INLINE Quaternion<T> quat_div(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
    return q0 * rcp(q1);
}

NAMESPACE_END(detail)

template <typename T0, typename T1>
ENOKI_INLINE auto operator*(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
    return detail::quat_mul(q0, q1);
}

template <typename T0>
ENOKI_INLINE auto operator*(const Quaternion<T0> &q0, const Quaternion<T0> &q1) {
    return detail::quat_mul(q0, q1);
}

template <typename T0, typename T1>
ENOKI_INLINE Quaternion<expr_t<T0, T1>> operator*(const Quaternion<T0> &q, const T1 &s) {
    return Array<expr_t<T0>, 4>(q) * s;
}

template <typename T0, typename T1>
ENOKI_INLINE Quaternion<expr_t<T0, T1>> operator*(const T0 &s, const Quaternion<T1> &q) {
    return s * Array<expr_t<T1>, 4>(q);
}

template <typename T0, typename T1>
ENOKI_INLINE auto operator/(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
    return detail::quat_div(q0, q1);
}

template <typename T0>
ENOKI_INLINE auto operator/(const Quaternion<T0> &q0, const Quaternion<T0> &q1) {
    return detail::quat_div(q0, q1);
}

template <typename T0, typename T1>
ENOKI_INLINE Quaternion<expr_t<T0, T1>> operator/(const Quaternion<T0> &q, const T1 &s) {
    return Array<expr_t<T0>, 4>(q) / s;
}

template <typename T> ENOKI_INLINE expr_t<T> real(const Quaternion<T> &q) { return q.w(); }
template <typename T> ENOKI_INLINE auto imag(const Quaternion<T> &q) { return head<3>(q); }
template <typename T> ENOKI_INLINE expr_t<T> abs(const Quaternion<T> &q) { return norm(q); }

template <typename T> Quaternion<expr_t<T>> exp(const Quaternion<T> &q) {
    auto qi    = imag(q);
    auto ri    = norm(qi);
    auto exp_w = exp(real(q));
    auto sc    = sincos(ri);

    return Quaternion<expr_t<T>>(qi * (sc.first * exp_w / ri),
                                 sc.second * exp_w);
}

template <typename T> Quaternion<expr_t<T>> log(const Quaternion<T> &q) {
    auto qi_n    = normalize(imag(q));
    auto rq      = norm(q);
    auto acos_rq = acos(real(q) / rq);
    auto log_rq  = log(rq);

    return Quaternion<expr_t<T>>(qi_n * acos_rq, log_rq);
}

template <typename T1, typename T2>
auto pow(const Quaternion<T1> &q0, const Quaternion<T2> &q1) {
    return exp(log(q0) * q1);
}

template <typename T>
Quaternion<expr_t<T>> sqrt(const Quaternion<T> &q) {
    auto ri = norm(imag(q));
    auto cs = sqrt(Complex<expr_t<T>>(real(q), ri));
    return Quaternion<expr_t<T>>(imag(q) * (rcp(ri) * imag(cs)), real(cs));
}

template <typename T, std::enable_if_t<!is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Quaternion<T> &q) {
    os << q.w();
    os << (q.x() < 0 ? " - " : " + ") << abs(q.x()) << "i";
    os << (q.y() < 0 ? " - " : " + ") << abs(q.y()) << "j";
    os << (q.z() < 0 ? " - " : " + ") << abs(q.z()) << "k";
    return os;
}

template <typename T, std::enable_if_t<is_array<std::decay_t<T>>::value, int> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Quaternion<T> &q) {
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

template <typename Matrix, typename T, typename Expr = expr_t<T>,
          std::enable_if_t<Matrix::Size == 4, int> = 0>
Matrix quat_to_matrix(Quaternion<T> q_) {
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

template <typename Matrix, typename T, typename Expr = expr_t<T>,
          std::enable_if_t<Matrix::Size == 3, int> = 0>
Matrix quat_to_matrix(Quaternion<T> q_) {
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

template <typename T, size_t Size,
          typename Expr = expr_t<T>,
          typename Quat = Quaternion<Expr>,
          std::enable_if_t<Size == 3 || Size == 4, int> = 0>
Quat matrix_to_quat(const Matrix<T, Size> &mat) {
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

template <typename T0, typename T1, typename Float, typename T = expr_t<T0, T1>, typename Return = Quaternion<T>>
Return slerp(const Quaternion<T0> &q0, const Quaternion<T1> &q1_, Float t) {
    using Base = Array<T, 4>;
    using Scalar = scalar_t<T>;

    auto cos_theta = dot(q0, q1_);
    Return q1 = mulsign(Base(q1_), cos_theta);
    cos_theta = abs(cos_theta);

    auto theta = safe_acos(cos_theta);
    auto sc = sincos(theta * t);
    Return qperp = normalize(q1 - q0 * cos_theta);

    return select(
        cos_theta > Scalar(0.9995),
        normalize(q0 * (Scalar(1.0) - t) + q1 * t),
        q0 * sc.second + qperp * sc.first
    );
}

template <typename Quat, typename Vector3, std::enable_if_t<Quat::IsQuaternion, int> = 0>
ENOKI_INLINE Quat rotate(const Vector3 &axis, const value_t<Quat> &angle) {
    using Scalar = scalar_t<Quat>;
    auto sc = sincos(angle * Scalar(.5f));
    return Quat(concat(axis * sc.first, sc.second));
}

NAMESPACE_END(enoki)
