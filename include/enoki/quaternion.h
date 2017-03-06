/*
    enoki/quaternion.h -- Quaternion data structure

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "complex.h"
#include "matrix.h"

NAMESPACE_BEGIN(enoki)

template <typename Type_>
struct Quaternion
    : StaticArrayImpl<Type_, 4, detail::approx_default<Type_>::value,
                      RoundingMode::Default, Quaternion<Type_>> {

    using Type = Type_;
    using Base = StaticArrayImpl<Type, 4, detail::approx_default<Type>::value,
                                 RoundingMode::Default, Quaternion<Type>>;

    template <typename T> using ReplaceType = Quaternion<T>;

    using Base::Base;
    using Base::operator=;

    Quaternion() = default;
    Quaternion(Type f) : Base(Type(0), Type(0), Type(0), f) { }

    template <typename T = Type,
              std::enable_if_t<!std::is_same<T, scalar_t<T>>::value, int> = 0>
    Quaternion(scalar_t<T> f) : Base(Type(0), Type(0), Type(0), f) { }

    template <typename Array,
              std::enable_if_t<array_size<Array>::value == 3, int> = 0>
    Quaternion(const Array &imag, const Type &real)
        : Base(imag.x(), imag.y(), imag.z(), real) { }
};

template <typename T1, typename T2,
          typename Type = decltype(std::declval<T1>() + std::declval<T2>())>
ENOKI_INLINE Quaternion<Type> operator*(const Quaternion<T1> &q0,
                                        const Quaternion<T2> &q1) {
    using Base = Array<Type, 4>;
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

template <typename T, typename T2>
ENOKI_INLINE Quaternion<expr_t<T>> operator*(const Quaternion<T> &q0,
                                             const T2 &other) {
    return Array<expr_t<T>, 4>(q0) * Array<expr_t<T>, 4>(other);
}

template <typename T1, typename T2,
          typename Type = decltype(std::declval<T1>() + std::declval<T2>())>
ENOKI_INLINE Type dot(const Quaternion<T1> &q0, const Quaternion<T2> &q1) {
    using Base = Array<Type, 4>;
    return dot(Base(q0), Base(q1));
}

template <typename T> ENOKI_INLINE Quaternion<expr_t<T>> conj(const Quaternion<T> &q) {
    const Quaternion<expr_t<T>> mask(-0.f, -0.f, -0.f, 0.f);
    return q ^ mask;
}

template <typename T> ENOKI_INLINE Quaternion<expr_t<T>> rcp(const Quaternion<T> &q) {
    return conj(q) * (1 / squared_norm(q));
}

template <typename T1, typename T2,
          typename Type = decltype(std::declval<T1>() + std::declval<T2>())>
ENOKI_INLINE Quaternion<Type> operator/(const Quaternion<T1> &q0,
                                        const Quaternion<T2> &q1) {
    return q0 * rcp(q1);
}

template <typename T, typename T2>
ENOKI_INLINE Quaternion<expr_t<T>> operator/(const Quaternion<T> &q0,
                                             const T2 &other) {
    return Array<expr_t<T>, 4>(q0) / Array<expr_t<T>, 4>(other);
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

template <typename T>
Matrix<expr_t<T>, 4> to_matrix(const Quaternion<T> &q) {
    auto xx = q.x() * q.x(), yy = q.y() * q.y(), zz = q.z() * q.z();
    auto xy = q.x() * q.y(), xz = q.x() * q.z(), yz = q.y() * q.z();
    auto xw = q.x() * q.w(), yw = q.y() * q.w(), zw = q.z() * q.w();

    return Matrix<expr_t<T>, 4>(
         1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (xz - yw), 0,
         2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw), 0,
         2 * (xz + yw), 2 * (yz - xw), 1 - 2 * (xx + yy), 0,
         0, 0, 0, 1
    );
}

template <typename T,
          typename Expr = expr_t<T>,
          typename Quat = Quaternion<Expr>>
Quat from_matrix(const Matrix<T, 4> &mat) {
    //  Converting a Rotation Matrix to a Quaternion
    //  Mike Day, Insomniac Games
    Expr t0(1 + mat(0, 0) - mat(1, 1) - mat(2, 2));
    Quat q0(t0, mat(0, 1) + mat(1, 0), mat(2, 0) + mat(0, 2),
            mat(1, 2) - mat(2, 1));

    Expr t1(1 - mat(0, 0) + mat(1, 1) - mat(2, 2));
    Quat q1(mat(0, 1) + mat(1, 0), t1, mat(1, 2) + mat(2, 1),
            mat(2, 0) - mat(0, 2));

    Expr t2(1 - mat(0, 0) - mat(1, 1) + mat(2, 2));
    Quat q2(mat(2, 0) + mat(0, 2), mat(1, 2) + mat(2, 1), t2,
            mat(0, 1) - mat(1, 0));

    Expr t3(1 + mat(0, 0) + mat(1, 1) + mat(2, 2));
    Quat q3(mat(1, 2) - mat(2, 1), mat(2, 0) - mat(0, 2),
            mat(0, 1) - mat(1, 0), t3);

    auto mask0 = mat(0, 0) > mat(1, 1);
    Expr t01 = select(mask0, t0, t1);
    Quat q01 = select(typename Quat::Mask(mask0), q0, q1);

    auto mask1 = mat(0, 0) < -mat(1, 1);
    Expr t23 = select(mask1, t2, t3);
    Quat q23 = select(typename Quat::Mask(mask1), q2, q3);

    auto mask2 = mat(2, 2) < 0;
    Expr t0123 = select(mask2, t01, t23);
    Quat q0123 = select(typename Quat::Mask(mask2), q01, q23);

    return q0123 * Quat(rsqrt(t0123) * 0.5f);
}

template <typename T, typename Float, typename Return = Quaternion<expr_t<T>>>
Return slerp(const Quaternion<T> &q1, const Quaternion<T> &q2_, Float t) {
    auto cos_theta = dot(q1, q2_), sign = detail::sign_mask(cos_theta);
    Return q2 = q2_ ^ sign;
    cos_theta = cos_theta ^ sign;

    auto theta = safe_acos(cos_theta);
    auto sc = sincos(theta * t);
    Return qperp = normalize(q2 - q1 * cos_theta);

    return select(
        typename Return::Mask(cos_theta > 0.9995f),
        normalize(q1 * (1 - t) +  q2 * t),
        q1 * sc.second + qperp * sc.first
    );
}

// =======================================================================
//! @{ \name Enoki accessors for static & dynamic vectorization
// =======================================================================

/* Is this type dynamic? */
template <typename T> struct is_dynamic_impl<Quaternion<T>> {
    static constexpr bool value = is_dynamic<T>::value;
};

/* Create a dynamic version of this type on demand */
template <typename T> struct dynamic_impl<Quaternion<T>> {
    using type = Quaternion<dynamic_t<T>>;
};

/* How many packets are stored in this instance? */
template <typename T>
size_t packets(const Quaternion<T> &v) {
    return packets(v.x());
}

/* What is the size of the dynamic dimension of this instance? */
template <typename T>
size_t dynamic_size(const Quaternion<T> &v) {
    return dynamic_size(v.x());
}

/* Resize the dynamic dimension of this instance */
template <typename T>
void dynamic_resize(Quaternion<T> &v, size_t size) {
    for (size_t i = 0; i < 4; ++i)
        dynamic_resize(v.coeff(i), size);
}

/* Construct a wrapper that references the data of this instance */
template <typename T> auto ref_wrap(Quaternion<T> &v) {
    using T2 = decltype(ref_wrap(v.x()));
    return Quaternion<T2>{ ref_wrap(v.x()), ref_wrap(v.y()),
                           ref_wrap(v.z()), ref_wrap(v.w()) };
}

/* Construct a wrapper that references the data of this instance (const) */
template <typename T> auto ref_wrap(const Quaternion<T> &v) {
    using T2 = decltype(ref_wrap(v.x()));
    return Quaternion<T2>{ ref_wrap(v.x()), ref_wrap(v.y()),
                           ref_wrap(v.z()), ref_wrap(v.w()) };
}

/* Return the i-th packet */
template <typename T> auto packet(Quaternion<T> &v, size_t i) {
    using T2 = decltype(packet(v.x(), i));
    return Quaternion<T2>{ packet(v.x(), i), packet(v.y(), i),
                           packet(v.z(), i), packet(v.w(), i) };
}

/* Return the i-th packet (const) */
template <typename T> auto packet(const Quaternion<T> &v, size_t i) {
    using T2 = decltype(packet(v.x(), i));
    return Quaternion<T2>{ packet(v.x(), i), packet(v.y(), i),
                           packet(v.z(), i), packet(v.w(), i) };
}

/* Return the i-th slice */
template <typename T> auto slice(Quaternion<T> &v, size_t i) {
    using T2 = decltype(slice(v.x(), i));
    return Quaternion<T2>{ slice(v.x(), i), slice(v.y(), i),
                           slice(v.z(), i), slice(v.w(), i) };
}

/* Return the i-th slice (const) */
template <typename T> auto slice(const Quaternion<T> &v, size_t i) {
    using T2 = decltype(slice(v.x(), i));
    return Quaternion<T2>{ slice(v.x(), i), slice(v.y(), i),
                           slice(v.z(), i), slice(v.w(), i) };
}

//! @}
// =======================================================================

NAMESPACE_END(enoki)
