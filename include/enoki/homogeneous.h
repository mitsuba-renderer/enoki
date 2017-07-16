/*
    enoki/homogeneous.h -- 3D homogeneous coordinate transformations

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "matrix.h"

NAMESPACE_BEGIN(enoki)

template <typename Matrix4, typename Vector3> ENOKI_INLINE Matrix4 translate(Vector3 v) {
    Matrix4 trafo = identity<Matrix4>();
    trafo.coeff(3) = concat(v, 1.f);
    return trafo;
}

template <typename Matrix4, typename Vector3> ENOKI_INLINE Matrix4 scale(Vector3 v) {
    return diag<Matrix4>(concat(v, 1.f));
}

template <typename Matrix4, typename Vector3>
ENOKI_INLINE Matrix4 rotate(Vector3 axis, entry_t<Matrix4> angle) {
    using Float = entry_t<Matrix4>;
    using Vector4 = column_t<Matrix4>;

    Float sin_theta, cos_theta;
    std::tie(sin_theta, cos_theta) = sincos(deg_to_rad(angle));
    Float cos_theta_m = 1.f - cos_theta;

    auto naxis = normalize(axis),
         shuf1 = shuffle<1, 2, 0>(naxis),
         shuf2 = shuffle<2, 0, 1>(naxis),
         tmp0  = naxis * naxis * cos_theta_m + cos_theta,
         tmp1  = naxis * shuf1 * cos_theta_m + shuf2 * sin_theta,
         tmp2  = naxis * shuf2 * cos_theta_m - shuf1 * sin_theta;

    return Matrix4(
        Vector4(tmp0.x(), tmp1.x(), tmp2.x(), 0.f),
        Vector4(tmp2.y(), tmp0.y(), tmp1.y(), 0.f),
        Vector4(tmp1.z(), tmp2.z(), tmp0.z(), 0.f),
        Vector4(0.f, 0.f, 0.f, 1.f)
    );
}

template <typename Matrix4>
ENOKI_INLINE Matrix4 perspective(entry_t<Matrix4> fov,
                                 entry_t<Matrix4> near,
                                 entry_t<Matrix4> far) {
    auto recip = rcp(far - near);
    auto c = cot(deg_to_rad(.5f * fov));

    Matrix4 trafo =
        diag<Matrix4>(column_t<Matrix4>(c, c, far * recip, 0.f));
    trafo(2, 3) = -near * far * recip;
    trafo(3, 2) = 1.f;

    return trafo;
}

template <typename Matrix4>
ENOKI_INLINE Matrix4 perspective_gl(entry_t<Matrix4> fov,
                                    entry_t<Matrix4> near,
                                    entry_t<Matrix4> far) {
    auto recip = rcp(near - far);
    auto c = cot(deg_to_rad(.5f * fov));

    Matrix4 trafo = diag<Matrix4>(
        column_t<Matrix4>(c, c, (near + far) * recip, 0.f));
    trafo(2, 3) = 2.f * near * far * recip;
    trafo(3, 2) = -1.f;

    return trafo;
}

NAMESPACE_END(enoki)
