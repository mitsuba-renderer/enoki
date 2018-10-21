/*
    tests/autodiff.cpp -- tests the reverse-mode automatic differentation layer

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <enoki/dynamic.h>
#include <enoki/autodiff.h>
#include <enoki/color.h>

using Float  = float;
using FloatP = Packet<Float>;
using FloatX = DynamicArray<FloatP>;
using FloatD = AutoDiffArray<FloatX>;

using UInt32  = uint32_t;
using UInt32P = Packet<UInt32>;
using UInt32X = DynamicArray<UInt32P>;
using UInt32D = AutoDiffArray<UInt32X>;

using Vector2f  = Array<Float, 2>;
using Vector2fP = Array<FloatP, 2>;
using Vector2fX = Array<FloatX, 2>;
using Vector2fD = Array<FloatD, 2>;

ENOKI_TEST(test00_identity) {
    clear_graph<FloatD>();
    FloatD x = 2.f;
    requires_gradient(x);
    backward(x);
    assert(gradient(x)[0] == 1.f);
}

ENOKI_TEST(test01_add) {
    clear_graph<FloatD>();
    FloatD x = 2.f, y = 3.f;
    requires_gradient(x, y);
    FloatD z = x + y;
    backward(z);
    assert(gradient(x)[0] == 1.f);
    assert(gradient(y)[0] == 1.f);
}

ENOKI_TEST(test02_sub) {
    clear_graph<FloatD>();
    FloatD x = 2.f, y = 3.f;
    requires_gradient(x);
    requires_gradient(y);
    FloatD z = x - y;
    backward(z);
    assert(gradient(x)[0] == 1.f);
    assert(gradient(y)[0] == -1.f);
}

ENOKI_TEST(test03_mul) {
    clear_graph<FloatD>();
    FloatD x = 2.f, y = 3.f;
    requires_gradient(x);
    requires_gradient(y);
    FloatD z = x * y;
    backward(z);
    assert(gradient(x)[0] == 3.f);
    assert(gradient(y)[0] == 2.f);
}

ENOKI_TEST(test04_div) {
    clear_graph<FloatD>();
    FloatD x = 2.f, y = 3.f;
    requires_gradient(x);
    requires_gradient(y);
    FloatD z = x / y;
    backward(z);
    assert(std::abs(gradient(x)[0] - 1.f / 3.f) < 1e-6f);
    assert(std::abs(gradient(y)[0] + 2.f / 9.f) < 1e-6f);
}

ENOKI_TEST(test05_hsum) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(0.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = hsum(x*x);
    backward(y);
    assert(y.size() == 1 && allclose(y.coeff(0), 95.0/27.0));
    assert(allclose(gradient(x), 2 * x));
}

ENOKI_TEST(test06_hprod) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.f, 2.f, 10);
    requires_gradient(x);
    FloatD y = hprod(x);
    backward(y);
    assert(allclose(gradient(x), hprod(x) / x) &&
           y.size() == 1 && allclose(y.coeff(0), 45.5402f));
}

ENOKI_TEST(test07_sqrt) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.f, 2.f, 10);
    requires_gradient(x);
    FloatD y = sqrt(x);
    backward(y);
    assert(allclose(y, sqrt(x)));
    assert(allclose(gradient(x), .5f*rsqrt(x)));
}

ENOKI_TEST(test08_rsqrt) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.f, 2.f, 10);
    requires_gradient(x);
    FloatD y = rsqrt(x);
    backward(y);
    assert(allclose(y, rsqrt(x)));
    assert(allclose(gradient(x), -.5f * pow(x, -3.f / 2.f)));
}

ENOKI_TEST(test09_exp) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(0.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = exp(x*x);
    backward(y);
    assert(allclose(y, exp(sqr(detach(x)))));
    assert(allclose(gradient(x), 2.f * x * exp(sqr(x))));
}

ENOKI_TEST(test10_log) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.f, 2.f, 10);
    requires_gradient(x);
    FloatD y = log(x*x);
    backward(y);
    assert(allclose(y, log(sqr(detach(x)))));
    assert(allclose(gradient(x), 2.f / x));
}

ENOKI_TEST(test11_sin) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(0.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = sin(x*x);
    backward(y);
    assert(allclose(y, sin(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x)*cos(sqr(detach(x)))));
}

ENOKI_TEST(test12_cos) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(0.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = cos(x*x);
    backward(y);
    assert(allclose(y, cos(sqr(detach(x)))));
    assert(allclose(gradient(x), -2*detach(x)*sin(sqr(detach(x)))));
}

ENOKI_TEST(test13_tan) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(0.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = tan(x*x);
    backward(y);
    assert(allclose(y, tan(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x)*sqr(sec(sqr(detach(x))))));
}

ENOKI_TEST(test14_csc) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.f, 2.f, 10);
    requires_gradient(x);
    FloatD y = csc(x*x);
    backward(y);
    assert(allclose(y, csc(sqr(detach(x)))));
    assert(allclose(gradient(x), -2 * detach(x) * cot(sqr(detach(x))) *
                                     csc(sqr(detach(x)))));
}

ENOKI_TEST(test15_sec) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.f, 2.f, 10);
    requires_gradient(x);
    FloatD y = sec(x*x);
    backward(y);
    assert(allclose(y, sec(sqr(detach(x)))));
    assert(allclose(gradient(x), 2 * detach(x) * sec(sqr(detach(x))) *
                                    tan(sqr(detach(x)))));
}

ENOKI_TEST(test16_cot) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.f, 2.f, 10);
    requires_gradient(x);
    FloatD y = cot(x*x);
    backward(y);
    assert(allclose(y, cot(sqr(detach(x)))));
    assert(allclose(gradient(x), -2 * detach(x) * sqr(csc(sqr(detach(x))))));
}

ENOKI_TEST(test17_asin) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-.8f, .8f, 10);
    requires_gradient(x);
    FloatD y = asin(x*x);
    backward(y);
    assert(allclose(y, asin(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x) / sqrt(1-sqr(sqr(detach(x))))));
}

ENOKI_TEST(test18_acos) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-.8f, .8f, 10);
    requires_gradient(x);
    FloatD y = acos(x*x);
    backward(y);
    assert(allclose(y, acos(sqr(detach(x)))));
    assert(allclose(gradient(x), -2*detach(x) / sqrt(1-sqr(sqr(detach(x))))));
}

ENOKI_TEST(test19_atan) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-.8f, .8f, 10);
    requires_gradient(x);
    FloatD y = atan(x*x);
    backward(y);
    assert(allclose(y, atan(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x) / (1+sqr(sqr(detach(x))))));
}

ENOKI_TEST(test20_sinh) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = sinh(x*x);
    backward(y);
    assert(allclose(y, sinh(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x) * cosh(sqr(detach(x)))));
}

ENOKI_TEST(test21_cosh) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = cosh(x*x);
    backward(y);
    assert(allclose(y, cosh(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x) * sinh(sqr(detach(x)))));
}

ENOKI_TEST(test22_tanh) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = tanh(x*x);
    backward(y);
    assert(allclose(y, tanh(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x) * sqr(sech(sqr(detach(x))))));
}

ENOKI_TEST(test23_csch) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = csch(x*x);
    backward(y);
    assert(allclose(y, csch(sqr(detach(x)))));
    assert(allclose(gradient(x), -2*detach(x) * csch(sqr(detach(x))) * coth(sqr(detach(x)))));
}

ENOKI_TEST(test24_sech) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = sech(x*x);
    backward(y);
    assert(allclose(y, sech(sqr(detach(x)))));
    assert(allclose(gradient(x), -2*detach(x) * sech(sqr(detach(x))) * tanh(sqr(detach(x)))));
}

ENOKI_TEST(test25_coth) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = coth(x*x);
    backward(y);
    assert(allclose(y, coth(sqr(detach(x)))));
    assert(allclose(gradient(x), -2*detach(x) * sqr(csch(sqr(detach(x))))));
}

ENOKI_TEST(test25_asinh) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = asinh(x*x);
    backward(y);
    assert(allclose(y, asinh(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x) * rsqrt(1 + sqr(sqr(detach(x))))));
}

ENOKI_TEST(test26_acosh) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(1.01f, 2.f, 10);
    requires_gradient(x);
    FloatD y = acosh(x*x);
    backward(y);
    assert(allclose(y, acosh(sqr(detach(x)))));
    assert(allclose(gradient(x), 2*detach(x) * rsqrt(sqr(sqr(detach(x))) - 1)));
}

ENOKI_TEST(test27_atanh) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(-.99f, .99f, 10);
    requires_gradient(x);
    FloatD y = atanh(x*x);
    backward(y);
    assert(allclose(y, atanh(sqr(detach(x)))));
    assert(allclose(gradient(x), -2*detach(x) * rcp(sqr(sqr(detach(x))) - 1)));
}

ENOKI_TEST(test28_linear_to_srgb) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(0.f, 1.f, 10);
    requires_gradient(x);
    FloatD y = linear_to_srgb(x);

    backward(y);
    std::cout << graphviz(y) << std::endl;
    /// from mathematica
    FloatX ref{ 12.92f,   1.58374f,  1.05702f, 0.834376f, 0.705474f,
                0.61937f, 0.556879f, 0.50899f, 0.470847f, 0.439583f };
    assert(hmax(abs(gradient(x) - ref)) < 1e-5f);
}

ENOKI_TEST(test29_gather) {
    clear_graph<FloatD>();
    FloatD x = linspace<FloatD>(0.f, 1.f, 11);
    requires_gradient(x);
    UInt32D i(3, 5, 6);
    FloatD y = gather<FloatD>(x, i);
    FloatD z = hsum(y);
    backward(z);
    assert(allclose(gradient(x), FloatX(0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0)));
}

template <typename Vector2>
Vector2 square_to_uniform_disk_concentric(Vector2 sample) {
    using Value  = value_t<Vector2>;
    using Mask   = mask_t<Value>;
    using Scalar = scalar_t<Value>;

    Value x = fmsub(Scalar(2), sample.x(), Scalar(1)),
          y = fmsub(Scalar(2), sample.y(), Scalar(1));

    Mask is_zero         = eq(x, zero<Value>()) &&
                           eq(y, zero<Value>()),
         quadrant_1_or_3 = abs(x) < abs(y);

    Value r  = select(quadrant_1_or_3, y, x),
          rp = select(quadrant_1_or_3, x, y);

    Value phi = rp / r * Scalar(.25f * Float(M_PI));
    masked(phi, quadrant_1_or_3) = Scalar(.5f * Float(M_PI)) - phi;
    masked(phi, is_zero) = zero<Value>();

    auto [sin_phi, cos_phi] = sincos(phi);

    return Vector2(r * cos_phi, r * sin_phi);
}


/// Warp a uniformly distributed square sample to a Beckmann distribution
template <typename Vector2,
          typename Value   = expr_t<value_t<Vector2>>,
          typename Vector3 = Array<Value, 3>>
Vector3 square_to_beckmann(const Vector2 &sample, const value_t<Vector2> &alpha) {
    Vector2 p = square_to_uniform_disk_concentric(sample);
    Value r2 = squared_norm(p);

    Value tan_theta_m_sqr = -alpha * alpha * log(1 - r2);
    Value cos_theta_m = rsqrt(1 + tan_theta_m_sqr);
    p *= safe_sqrt((1 - cos_theta_m * cos_theta_m) / r2);

    return Vector3(p.x(), p.y(), cos_theta_m);
}

ENOKI_TEST(test30_sample_disk) {
    clear_graph<FloatD>();
    Vector2f x(.2f, .3f);
    std::cout << square_to_beckmann(x, .4f) << std::endl;

    Vector2fD y(.2f, .3f);
    requires_gradient(y);
    auto sum = hsum(square_to_beckmann(y, .4f));
    std::cout << graphviz(sum) << std::endl;
    backward(sum);
}
