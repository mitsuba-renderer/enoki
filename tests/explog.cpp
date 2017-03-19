#include "test.h"

ENOKI_TEST_FLOAT(test01_ldexp) {
    const Value inf = std::numeric_limits<Value>::infinity();
    const Value nan = std::numeric_limits<Value>::quiet_NaN();

    for (int i = -10; i < 10; ++i) {
        for (int j = -10; j < 10; ++j) {
            T f = T(std::ldexp(Value(i), j));
            T f2 = ldexp(T(Value(i)), T(Value(j)));
            assert(f == f2);
        }
    }

    assert(T(ldexp(T(inf), T(Value(2)))) == T(inf));
    assert(T(ldexp(T(-inf), T(Value(2)))) == T(-inf));
    assert(all(enoki::isnan(ldexp(T(nan), T(Value(2))))));
}

#if !defined(__AVX512F__)
// AVX512F frexp() uses slightly different conventions
// It is used by log() where this is not a problem though
ENOKI_TEST_FLOAT(test02_frexp) {
    const Value inf = std::numeric_limits<Value>::infinity();
    const Value nan = std::numeric_limits<Value>::quiet_NaN();
    using int_array_t = enoki::int_array_t<T>;
    using Int = typename int_array_t::Value;

    for (int i = -10; i < 10; ++i) {
        int e;
        Value f = std::frexp(Value(i), &e);
        T e2, f2;
        std::tie(f2, e2) = frexp(T(Value(i)));
        assert(T(Value(e)) == e2);
        assert(T(f) == f2);
    }

    T e, f;

    std::tie(f, e) = frexp(T(inf));
    assert(e == T(0.f) || e == T(-1.f));
    assert(f == T(inf));

    std::tie(f, e) = frexp(T(-inf));
    assert(e == T(0.f) || e == T(-1.f));
    assert(f == T(-inf));

    std::tie(f, e) = frexp(T(+0.f));
    assert(e == T(0.f));
    assert((reinterpret_array<int_array_t>(f) == int_array_t(memcpy_cast<Int>(Value(+0.f)))));

    std::tie(f, e) = frexp(T(-0.f));
    assert(e == T(0.f));
    assert((reinterpret_array<int_array_t>(f) == int_array_t(memcpy_cast<Int>(Value(-0.f)))));

    std::tie(f, e) = frexp(T(nan));
    assert(e == T(0.f) || e == T(-1.f));
    assert(all(enoki::isnan(f)));
}
#endif

ENOKI_TEST_FLOAT(test03_exp) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return exp(a); },
        [](double a) { return std::exp(a); },
        Value(-20), Value(30),
#if defined(__AVX512ER__)
        27
#else
        1
#endif
    );

    Array<T, 4> x((Value) M_PI);
    Array<T&, 4> y(x);
    assert(exp(x) == exp(y));
}

ENOKI_TEST_FLOAT(test04_log) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return log(a); },
        [](double a) { return std::log(a); },
        Value(1e-20), Value(1000),
        1
    );

    Array<T, 4> x((Value) M_PI);
    Array<T&, 4> y(x);
    assert(log(x) == log(y));
}

ENOKI_TEST_FLOAT(test05_pow) {
    assert(T(abs(pow(T(Value(M_PI)), T(Value(-2))) -
               T(Value(0.101321183642338))))[0] < 1e-6f);
}

ENOKI_TEST_FLOAT(test06_erf) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return erf(a); },
        [](double a) { return std::erf(a); },
        Value(-1), Value(1), 72
    );

    Array<T, 4> x((Value) 0.5);
    Array<T&, 4> y(x);
    assert(erf(x) == erf(y));
}

ENOKI_TEST_FLOAT(test07_erfi) {
    for (int i = 0; i < 1000; ++i) {
        auto f = T((float) i / 1000.0f * 2 - 1 + 1e-6f);
        auto inv = erfi(f);
        auto f2 = erf(inv);
        assert(std::abs(T(f-f2)[0]) < 1e-6f);
    }
}
