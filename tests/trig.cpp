#include "test.h"

ENOKI_TEST_FLOAT(test01_sin) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sin(a); },
        [](double a) { return std::sin(a); },
        Scalar(-8192), Scalar(8192),
        19
    );

    Array<T, 4> x((Scalar) M_PI);
    Array<T&, 4> y(x);
    assert(sin(x) == sin(y));
}

ENOKI_TEST_FLOAT(test02_cos) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return cos(a); },
        [](double a) { return std::cos(a); },
        Scalar(-8192), Scalar(8192),
        47
    );

    Array<T, 4> x((Scalar) M_PI);
    Array<T&, 4> y(x);
    assert(cos(x) == cos(y));
}

ENOKI_TEST_FLOAT(test03_sincos_sin) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sincos(a).first; },
        [](double a) { return std::sin(a); },
        Scalar(-8192), Scalar(8192),
        19
    );
}

ENOKI_TEST_FLOAT(test04_sincos_cos) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sincos(a).second; },
        [](double a) { return std::cos(a); },
        Scalar(-8192), Scalar(8192),
        47
    );

    Array<T, 4> x((Scalar) M_PI), s, c;
    Array<T&, 4> y(x);
    auto result = sincos(y);
    assert(result.first == sin(y) && result.second == cos(y));
}

ENOKI_TEST_FLOAT(test05_tan) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return tan(a); },
        [](double a) { return std::tan(a); },
        Scalar(-8192), Scalar(8192),
        30
    );

    Array<T, 4> x((Scalar) M_PI);
    Array<T&, 4> y(x);
    assert(tan(x) == tan(y));
}

ENOKI_TEST_FLOAT(test06_asin) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return asin(a); },
        [](double a) { return std::asin(a); },
        Scalar(-1), Scalar(1),
        61
    );

    Array<T, 4> x((Scalar) 0.5);
    Array<T&, 4> y(x);
    assert(asin(x) == asin(y));
}

ENOKI_TEST_FLOAT(test07_acos) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return acos(a); },
        [](double a) { return std::acos(a); },
        Scalar(-1), Scalar(1),
        4
    );

    Array<T, 4> x((Scalar) 0.5);
    Array<T&, 4> y(x);
    assert(acos(x) == acos(y));
}

ENOKI_TEST_FLOAT(test08_atan) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return atan(a); },
        [](double a) { return std::atan(a); },
        Scalar(-1), Scalar(1),
        12
    );

    Array<T, 4> x((Scalar) 0.5);
    Array<T&, 4> y(x);
    assert(atan(x) == atan(y));
}

ENOKI_TEST_FLOAT(test09_atan2) {
    for (int ix = 0; ix <= 100; ++ix) {
        for (int iy = 0; iy <= 100; ++iy) {
            Scalar x = Scalar(ix) / Scalar(100) * 2 - 1;
            Scalar y = Scalar(iy) / Scalar(100) * 2 - 1;
            T atan2_ = T(atan2(T(y), T(x)))[0];
            Scalar atan2_ref = std::atan2(y, x);
            if (x == 0 || y == 0)
                continue;
            assert(std::abs(atan2_[0] - atan2_ref) < 3.58e-6f);
        }
    }
}

ENOKI_TEST_FLOAT(test10_remainder) {
    assert(std::abs(T(csc(T(1.f)) - 1 / std::sin(1.f))[0]) < 1e-6f);
    assert(std::abs(T(sec(T(1.f)) - 1 / std::cos(1.f))[0]) < 1e-6f);
    assert(std::abs(T(cot(T(1.f)) - 1 / std::tan(1.f))[0]) < 1e-6f);
}

ENOKI_TEST_FLOAT(test11_safe_math) {
    assert(all(abs(safe_asin(T(Scalar(-10))) - Scalar(-M_PI / 2)) < 1e-6f));
    assert(all(abs(safe_asin(T(Scalar( 10))) - Scalar( M_PI / 2)) < 1e-6f));
    assert(all(abs(safe_acos(T(Scalar(-10))) - Scalar(M_PI)) < 1e-6f));
    assert(all(abs(safe_acos(T(Scalar( 10))) - Scalar(0)) < 1e-6f));
    assert(all(abs(safe_sqrt(T(Scalar(4)))   - Scalar(2)) < 1e-6f));
    assert(all(abs(safe_sqrt(T(Scalar(-1)))  - Scalar(0)) < 1e-6f));
}
