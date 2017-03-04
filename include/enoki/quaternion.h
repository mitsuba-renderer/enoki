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

NAMESPACE_BEGIN(enoki)

template <typename Scalar_>
struct Quaternion
    : StaticArrayImpl<Scalar_, 4, detail::approx_default<Scalar_>::value,
                      RoundingMode::Default, Quaternion<Scalar_>> {

    using Scalar = Scalar_;
    using Base =
        StaticArrayImpl<Scalar, 4, detail::approx_default<Scalar>::value,
                        RoundingMode::Default, Quaternion<Scalar>>;

    using Base::Base;
    using Base::operator=;

    Quaternion(Scalar f) : Base(Scalar(0), Scalar(0), Scalar(0), f) { }

    template <typename T = Scalar,
              std::enable_if_t<!std::is_same<T, scalar_t<T>>::value, int> = 0>
    Quaternion(scalar_t<T> f) : Base(Scalar(0), Scalar(0), Scalar(0), f) { }

    template <typename Array,
              std::enable_if_t<array_size<Array>::value == 3, int> = 0>
    Quaternion(const Array &imag, const Scalar &real)
        : Base(imag.x(), imag.y(), imag.z(), real) { }
};

template <typename T1, typename T2,
          typename Scalar = decltype(std::declval<T1>() + std::declval<T2>())>
ENOKI_INLINE Quaternion<Scalar> operator*(const Quaternion<T1> &q0,
                                          const Quaternion<T2> &q1) {
    using Base = Array<Scalar, 4>;
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

template <typename T1, typename T2,
          typename Scalar = decltype(std::declval<T1>() + std::declval<T2>())>
ENOKI_INLINE Scalar dot(const Quaternion<T1> &q0, const Quaternion<T2> &q1) {
    using Base = Array<Scalar, 4>;
    return dot(Base(q0), Base(q1));
}

template <typename T>
ENOKI_INLINE Quaternion<expr_t<T>> operator*(const Quaternion<T> &q0, const value_t<T> &other) {
    return Array<expr_t<T>, 4>(q0) * other;
}

template <typename T, std::enable_if_t<!std::is_same<value_t<T>, scalar_t<T>>::value, int> = 0>
ENOKI_INLINE Quaternion<expr_t<T>> operator*(const Quaternion<T> &q0, const scalar_t<T> &other) {
    return Array<expr_t<T>, 4>(q0) * other;
}

template <typename T> ENOKI_INLINE Quaternion<expr_t<T>> conj(const Quaternion<T> &q) {
    const Quaternion<expr_t<T>> mask(-0.f, -0.f, -0.f, 0.f);
    return q ^ mask;
}

template <typename T> ENOKI_INLINE Quaternion<expr_t<T>> rcp(const Quaternion<T> &q) {
    return conj(q) * (1 / squared_norm(q));
}

template <typename T1, typename T2,
          typename Scalar = decltype(std::declval<T1>() + std::declval<T2>())>
ENOKI_INLINE Quaternion<Scalar> operator/(const Quaternion<T1> &q0,
                                          const Quaternion<T2> &q1) {
    return q0 * rcp(q1);
}

template <typename T>
ENOKI_INLINE Quaternion<expr_t<T>> operator/(const Quaternion<T> &q0, const value_t<T> &other) {
    return Array<T, 4>(q0) / other;
}

template <typename T, std::enable_if_t<!std::is_same<value_t<T>, scalar_t<T>>::value, int> = 0>
ENOKI_INLINE Quaternion<expr_t<T>> operator/(const Quaternion<T> &q0, const scalar_t<T> &other) {
    return Array<T, 4>(q0) / other;
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

//! @}
// =======================================================================

NAMESPACE_END(enoki)
