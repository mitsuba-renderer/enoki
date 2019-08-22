/*
    enoki/quaternion.h -- Quaternion data structure

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2019 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/complex.h>
#include <enoki/matrix.h>

NAMESPACE_BEGIN(enoki)

/// SFINAE helper for quaternions
template <typename T> using is_quaternion_helper = enable_if_t<std::decay_t<T>::IsQuaternion>;
template <typename T> constexpr bool is_quaternion_v = is_detected_v<is_quaternion_helper, T>;
template <typename T> using enable_if_quaternion_t = enable_if_t<is_quaternion_v<T>>;
template <typename T> using enable_if_not_quaternion_t = enable_if_t<!is_quaternion_v<T>>;

template <typename Value_, bool Approx_>
struct Quaternion : StaticArrayImpl<Value_, 4, Approx_, RoundingMode::Default, false, Quaternion<Value_, Approx_>> {
    using Base = StaticArrayImpl<Value_, 4, Approx_, RoundingMode::Default, false, Quaternion<Value_, Approx_>>;
    ENOKI_ARRAY_IMPORT_BASIC(Base, Quaternion);
    using Base::operator=;

    static constexpr bool IsQuaternion = true;

    using ArrayType = Quaternion;
    using MaskType = Mask<Value_, 4, Approx_, RoundingMode::Default>;

    template <typename T>
    using ReplaceValue = Quaternion<T, is_std_float_v<scalar_t<T>> && is_std_float_v<scalar_t<Value_>>
                                    ? Approx_ : array_approx_v<T>>;

    Quaternion() = default;

    template <typename Value2, bool Approx2>
    ENOKI_INLINE Quaternion(const Quaternion<Value2, Approx2> &z) : Base(z) { }

    template <typename T, enable_if_t<(array_depth_v<T> < Base::Depth && (is_scalar_v<T> || is_array_v<T>))> = 0,
              enable_if_not_quaternion_t<T> = 0>
    ENOKI_INLINE Quaternion(T &&v) : Base(zero<Value_>(), zero<Value_>(), zero<Value_>(), v) { }

    template <typename T, enable_if_t<(array_depth_v<T> == Base::Depth || !(is_scalar_v<T> || is_array_v<T>))> = 0,
              enable_if_not_quaternion_t<T> = 0>
    ENOKI_INLINE Quaternion(T &&v) : Base(std::forward<T>(v)) { }

    ENOKI_INLINE Quaternion(const Value_ &vi, const Value_ &vj,
                            const Value_ &vk, const Value_ &vr)
        : Base(vi, vj, vk, vr) { }

    template <typename Im, typename Re, enable_if_t<array_size_v<Im> == 3> = 0>
    ENOKI_INLINE Quaternion(const Im &im, const Re &re)
        : Base(im.x(), im.y(), im.z(), re) { }

    /// Construct from sub-arrays
    template <typename T1, typename T2, typename T = Quaternion, enable_if_t<
              array_depth_v<T1> == array_depth_v<T> && array_size_v<T1> == 2 &&
              array_depth_v<T2> == array_depth_v<T> && array_size_v<T2> == 2> = 0>
    Quaternion(const T1 &a1, const T2 &a2)
        : Base(a1, a2) { }
};

template <typename T, enable_if_quaternion_t<T> = 0>
ENOKI_INLINE T identity() {
    return T(0.f, 0.f, 0.f, 1.f);
}

template <typename T, bool Approx> ENOKI_INLINE expr_t<T> real(const Quaternion<T, Approx> &q) { return q.w(); }
template <typename T, bool Approx> ENOKI_INLINE auto imag(const Quaternion<T, Approx> &q) { return head<3>(q); }

template <typename T0, typename T1, bool Approx0, bool Approx1, typename T = expr_t<T0, T1>>
ENOKI_INLINE T dot(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    using Base = Array<T, 4, Approx0 && Approx1>;
    return dot(Base(q0), Base(q1));
}

template <typename T, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T>, Approx> conj(const Quaternion<T, Approx> &q) {
    const Quaternion<expr_t<T>, Approx> mask(-0.f, -0.f, -0.f, 0.f);
    return q ^ mask;
}

template <typename T, bool Approx>
ENOKI_INLINE expr_t<T> squared_norm(const Quaternion<T, Approx> &q) {
    return enoki::squared_norm(Array<expr_t<T>, 4, Approx>(q));
}

template <typename T, bool Approx>
ENOKI_INLINE expr_t<T> norm(const Quaternion<T, Approx> &q) {
    return enoki::norm(Array<expr_t<T>, 4, Approx>(q));
}

template <typename T, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T>, Approx> normalize(const Quaternion<T, Approx> &q) {
    return enoki::normalize(Array<expr_t<T>, 4, Approx>(q));
}

template <typename T, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T>, Approx> rcp(const Quaternion<T, Approx> &q) {
    return conj(q) * (1 / squared_norm(q));
}

template <typename T0, typename T1, bool Approx0, bool Approx1,
          typename Value = expr_t<T0, T1>, typename Result = Quaternion<Value, Approx0 && Approx1>>
ENOKI_INLINE Result operator*(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    using Base   = Array<Value, 4, Approx0 && Approx1>;
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

template <typename T0, typename T1, bool Approx0,
          typename Value = expr_t<T0, T1>, typename Result = Quaternion<Value, Approx0>>
ENOKI_INLINE Result operator*(const Quaternion<T0, Approx0> &q0, const T1 &v1) {
    return Array<expr_t<T0>, 4, Approx0>(q0) * v1;
}

template <typename T0, typename T1, bool Approx1,
          typename Value = expr_t<T0, T1>, typename Result = Quaternion<Value, Approx1>>
ENOKI_INLINE Result operator*(const T0 &v0, const Quaternion<T1, Approx1> &q1) {
    return v0 * Array<expr_t<T0>, 4, Approx1>(q1);
}

template <typename T0, typename T1, bool Approx0, bool Approx1,
          typename Value = expr_t<T0, T1>, typename Result = Quaternion<Value, Approx0 && Approx1>>
ENOKI_INLINE Result operator/(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    return q0 * rcp(q1);
}

template <typename T0, typename T1, bool Approx0,
          typename Value = expr_t<T0, T1>, typename Result = Quaternion<Value, Approx0>>
ENOKI_INLINE Result operator/(const Quaternion<T0, Approx0> &z0, const T1 &v1) {
    return Array<expr_t<T0>, 4, Approx0>(z0) / v1;
}

template <typename T, bool Approx>
ENOKI_INLINE expr_t<T> abs(const Quaternion<T, Approx> &z) {
    return norm(z);
}

template <typename T, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T>, Approx> exp(const Quaternion<T, Approx> &q) {
    auto qi     = imag(q);
    auto ri     = norm(qi);
    auto exp_w  = exp(real(q));
    auto [s, c] = sincos(ri);

    return { qi * (s * exp_w / ri), c * exp_w };
}

template <typename T, bool Approx>
ENOKI_INLINE Quaternion<expr_t<T>, Approx> log(const Quaternion<T, Approx> &q) {
    auto qi_n    = normalize(imag(q));
    auto rq      = norm(q);
    auto acos_rq = acos(real(q) / rq);
    auto log_rq  = log(rq);

    return { qi_n * acos_rq, log_rq };
}

template <typename T0, typename T1, bool Approx0, bool Approx1>
ENOKI_INLINE auto pow(const Quaternion<T0, Approx0> &q0, const Quaternion<T1, Approx1> &q1) {
    return exp(log(q0) * q1);
}

template <typename T, bool Approx>
Quaternion<expr_t<T>, Approx> sqrt(const Quaternion<T, Approx> &q) {
    auto ri = norm(imag(q));
    auto cs = sqrt(Complex<expr_t<T>, Approx>(real(q), ri));
    return { imag(q) * (rcp(ri) * imag(cs)), real(cs) };
}

template <typename Matrix, bool Approx, typename T, typename Expr = expr_t<T>,
          enable_if_t<Matrix::Size == 4> = 0>
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
          enable_if_t<Matrix::Size == 3> = 0>
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
          enable_if_t<Size == 3 || Size == 4> = 0>
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

template <typename T0, typename T1, bool Approx0, bool Approx1, typename T2,
          typename Value  = expr_t<T0, T1, T2>,
          typename Return = Quaternion<Value, Approx0 && Approx1>>
ENOKI_INLINE Return slerp(const Quaternion<T0, Approx0> &q0,
                          const Quaternion<T1, Approx1> &q1_, const T2 &t) {
    using Base = Array<Value, 4, Approx0 && Approx1>;

    Value cos_theta = dot(q0, q1_);
    Return q1 = mulsign(Base(q1_), cos_theta);
    cos_theta = mulsign(cos_theta, cos_theta);

    Value theta = acos(cos_theta);
    auto [s, c] = sincos(theta * t);
    auto close_mask = cos_theta > 0.9995f;

    Return qperp  = normalize(q1 - q0 * cos_theta),
           result = q0 * c + qperp * s;

    if (ENOKI_UNLIKELY(any_nested(close_mask)))
        result[mask_t<Base>(close_mask)] =
            Base(normalize(q0 * (1.f - t) + q1 * t));

    return result;
}

template <typename Quat, typename Vector3, enable_if_t<Quat::IsQuaternion> = 0>
ENOKI_INLINE Quat rotate(const Vector3 &axis, const value_t<Quat> &angle) {
    auto [s, c] = sincos(angle * .5f);
    return concat(axis * s, c);
}

template <typename T, bool Approx, enable_if_not_array_t<T> = 0>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os, const Quaternion<T, Approx> &q) {
    os << q.w();
    os << (q.x() < 0 ? " - " : " + ") << abs(q.x()) << "i";
    os << (q.y() < 0 ? " - " : " + ") << abs(q.y()) << "j";
    os << (q.z() < 0 ? " - " : " + ") << abs(q.z()) << "k";
    return os;
}

template <typename T, bool Approx, enable_if_array_t<T> = 0>
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

NAMESPACE_END(enoki)
