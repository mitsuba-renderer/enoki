/*
    enoki/transform.h -- 3D homogeneous coordinate transformations

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "matrix.h"
#include "quaternion.h"

NAMESPACE_BEGIN(enoki)

template <typename Matrix4, typename Vector3> ENOKI_INLINE Matrix4 translate(const Vector3 &v) {
    Matrix4 trafo = identity<Matrix4>();
    trafo.coeff(3) = concat(v, 1.f);
    return trafo;
}

template <typename Matrix4, typename Vector3> ENOKI_INLINE Matrix4 scale(const Vector3 &v) {
    return diag<Matrix4>(concat(v, 1.f));
}

template <typename Matrix4, typename Vector3, std::enable_if_t<Matrix4::IsMatrix, int> = 0>
ENOKI_INLINE Matrix4 rotate(const Vector3 &axis, const entry_t<Matrix4> &angle) {
    using Float = entry_t<Matrix4>;
    using Vector4 = column_t<Matrix4>;

    Float sin_theta, cos_theta;
    std::tie(sin_theta, cos_theta) = sincos(angle);
    Float cos_theta_m = 1.f - cos_theta;

    auto shuf1 = shuffle<1, 2, 0>(axis),
         shuf2 = shuffle<2, 0, 1>(axis),
         tmp0  = axis * axis * cos_theta_m + cos_theta,
         tmp1  = axis * shuf1 * cos_theta_m + shuf2 * sin_theta,
         tmp2  = axis * shuf2 * cos_theta_m - shuf1 * sin_theta;

    return Matrix4(
        Vector4(tmp0.x(), tmp1.x(), tmp2.x(), 0.f),
        Vector4(tmp2.y(), tmp0.y(), tmp1.y(), 0.f),
        Vector4(tmp1.z(), tmp2.z(), tmp0.z(), 0.f),
        Vector4(0.f, 0.f, 0.f, 1.f)
    );
}

template <typename Matrix4>
ENOKI_INLINE Matrix4 perspective(const entry_t<Matrix4> &fov,
                                 const entry_t<Matrix4> &near,
                                 const entry_t<Matrix4> &far) {
    auto recip = rcp(near - far);
    auto c = cot(.5f * fov);

    Matrix4 trafo = diag<Matrix4>(
        column_t<Matrix4>(c, c, (near + far) * recip, 0.f));

    trafo(2, 3) = 2.f * near * far * recip;
    trafo(3, 2) = -1.f;

    return trafo;
}

template <typename Matrix4>
ENOKI_INLINE Matrix4 frustum(entry_t<Matrix4> left,
                             entry_t<Matrix4> right,
                             entry_t<Matrix4> bottom,
                             entry_t<Matrix4> top,
                             entry_t<Matrix4> near,
                             entry_t<Matrix4> far) {

    auto rl = rcp(right - left),
         tb = rcp(top - bottom),
         fn = rcp(far - near);

    Matrix4 trafo = zero<Matrix4>();
    trafo(0, 0) = (2.f * near) * rl;
    trafo(1, 1) = (2.f * near) * tb;
    trafo(0, 2) = (right + left) * rl;
    trafo(1, 2) = (top + bottom) * tb;
    trafo(2, 2) = -(far + near) * fn;
    trafo(3, 2) = -1.f;
    trafo(2, 3) = -2.f * far * near * fn;

    return trafo;
}

template <typename Matrix4>
ENOKI_INLINE Matrix4 ortho(const entry_t<Matrix4> &left,
                           const entry_t<Matrix4> &right,
                           const entry_t<Matrix4> &bottom,
                           const entry_t<Matrix4> &top,
                           const entry_t<Matrix4> &near,
                           const entry_t<Matrix4> &far) {

    auto rl = rcp(right - left),
         tb = rcp(top - bottom),
         fn = rcp(far - near);

    Matrix4 trafo = zero<Matrix4>();

    trafo(0, 0) = 2.f * rl;
    trafo(1, 1) = 2.f * tb;
    trafo(2, 2) = -2.f * fn;
    trafo(3, 3) = 1.f;
    trafo(0, 3) = -(right + left) * rl;
    trafo(1, 3) = -(top + bottom) * tb;
    trafo(2, 3) = -(far + near) * fn;

    return trafo;
}

template <typename Matrix4, typename Point, typename Vector>
Matrix4 look_at(const Point &origin, const Point &target, const Vector &up) {
    auto dir = normalize(target - origin);
    auto left = normalize(cross(dir, up));
    auto new_up = cross(left, dir);

    return Matrix4(
        concat(left, 0.f),
        concat(new_up, 0.f),
        concat(-dir, 0.f),
        column_t<Matrix4>(
            -dot(left, origin),
            -dot(up, origin),
             dot(dir, origin),
             1.f
        )
    );
}

template <typename T, bool Approx,
          typename E       = expr_t<T>,
          typename Matrix3 = Matrix<E, 3, Approx>,
          typename Vector3 = Array<E, 3, Approx>,
          typename Quat    = Quaternion<E, Approx>>
std::tuple<Matrix3, Quat, Vector3> transform_decompose(const Matrix<T, 4, Approx> &A) {
    Matrix3 A_sub(A), Q, P;
    std::tie(Q, P) = polar_decomp(A_sub);

    if (ENOKI_UNLIKELY(any(enoki::isnan(Q(0, 0)))))
        Q = identity<Matrix3>();

    auto sign_q = det(Q);
    Q = mulsign(Array<Vector3, 3>(Q), sign_q);
    P = mulsign(Array<Vector3, 3>(P), sign_q);

    return std::make_tuple(
        P,
        matrix_to_quat(Q),
        head<3>(A.col(3))
    );
}

template <typename T, bool Approx,
          typename E = expr_t<T>,
          typename Matrix3 = Matrix<E, 3, Approx>,
          typename Matrix4 = Matrix<E, 4, Approx>,
          typename Vector3>
Matrix4 transform_compose(const Matrix<T, 3, Approx> &S,
                          const Quaternion<T, Approx> &q,
                          const Vector3 &t) {
    Matrix4 result = Matrix4(quat_to_matrix<Matrix3>(q) * S);
    result.coeff(3) = concat(t, 1.f);
    return result;
}

template <typename T, bool Approx,
          typename E = expr_t<T>,
          typename Matrix3 = Matrix<E, 3, Approx>,
          typename Matrix4 = Matrix<E, 4, Approx>,
          typename Vector3>
Matrix4 transform_compose_inverse(const Matrix<T, 3, Approx> &S,
                                  const Quaternion<T, Approx> &q,
                                  const Vector3 &t) {
    auto inv_m = inverse(quat_to_matrix<Matrix3>(q) * S);
    Matrix4 result = Matrix4(inv_m);
    result.coeff(3) = concat(inv_m * -t, 1.f);
    return result;
}

NAMESPACE_END(enoki)
