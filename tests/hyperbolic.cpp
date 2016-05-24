#include "test.h"

ENOKI_TEST_FLOAT(test01_sinh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sinh(a); },
        [](double a) { return std::sinh(a); },
        Scalar(-10), Scalar(10),
        178
    );


    Array<T, 4> x((Scalar) 1);
    Array<T&, 4> y(x);
    assert(sinh(x) == sinh(y));
}

ENOKI_TEST_FLOAT(test02_cosh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return cosh(a); },
        [](double a) { return std::cosh(a); },
        Scalar(-10), Scalar(10),
#if defined(__AVX512ER__)
        8
#else
        3
#endif
    );

    Array<T, 4> x((Scalar) 1);
    Array<T&, 4> y(x);
    assert(cosh(x) == cosh(y));
}

ENOKI_TEST_FLOAT(test03_sincosh_sin) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sincosh(a).first; },
        [](double a) { return std::sinh(a); },
        Scalar(-10), Scalar(10),
        178
    );

    Array<T, 4> x((Scalar) 1), s, c;
    Array<T&, 4> y(x);
    auto result = sincosh(y);
    assert(result.first == sinh(y) && result.second == cosh(y));
}

ENOKI_TEST_FLOAT(test04_sincosh_cos) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sincosh(a).second; },
        [](double a) { return std::cosh(a); },
        Scalar(-10), Scalar(10),
#if defined(__AVX512ER__)
        8
#else
        3
#endif
    );
}

ENOKI_TEST_FLOAT(test05_tanh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return tanh(a); },
        [](double a) { return std::tanh(a); },
        Scalar(-10), Scalar(10),
        357, false
    );
    Array<T, 4> x((Scalar) 1);
    Array<T&, 4> y(x);
    assert(tanh(x) == tanh(y));
}

ENOKI_TEST_FLOAT(test06_asinh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return asinh(a); },
        [](double a) { return std::asinh(a); },
        Scalar(-10), Scalar(10),
        178, false
    );
    Array<T, 4> x((Scalar) 2);
    Array<T&, 4> y(x);
    assert(asinh(x) == asinh(y));
}

ENOKI_TEST_FLOAT(test07_acosh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return acosh(a); },
        [](double a) { return std::acosh(a); },
        Scalar(1), Scalar(10),
        123
    );
    Array<T, 4> x((Scalar) 2);
    Array<T&, 4> y(x);
    assert(acosh(x) == acosh(y));
}

ENOKI_TEST_FLOAT(test08_atanh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return atanh(a); },
        [](double a) { return std::atanh(a); },
        Scalar(-1 + 0.001), Scalar(1 - 0.001),
        358
    );
    Array<T, 4> x((Scalar) 0.5);
    Array<T&, 4> y(x);
    assert(atanh(x) == atanh(y));
}

ENOKI_TEST_FLOAT(test09_remainder) {
    assert(std::abs(T(csch(T(1.f)) - 1 / std::sinh(1.f))[0]) < 1e-6f);
    assert(std::abs(T(sech(T(1.f)) - 1 / std::cosh(1.f))[0]) < 1e-6f);
    assert(std::abs(T(coth(T(1.f)) - 1 / std::tanh(1.f))[0]) < 1e-6f);
}
