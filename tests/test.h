/*
    test/test.h -- Rudimentary test runner framework

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#if defined(NDEBUG)
#  undef NDEBUG
#endif

namespace test {
    /* Global parameters and statistics */
    static bool detailed = false;
    static int nonvectorized_count = 0;
}

#define ENOKI_SCALAR { ++test::nonvectorized_count; }
#define ENOKI_TEST(name) void name(); static test::Test name##_test{#name, &name}; void name()

#include <enoki/array.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <random>

using namespace enoki;

NAMESPACE_BEGIN(test)

bool replace(std::string &str, const std::string &from, const std::string &to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

class Test {
public:
    using TestStorage = std::vector<std::pair<std::string, void (*)()>>;

    Test(const char *name, void (*test)()) {
        if (!registered_tests)
            registered_tests = new TestStorage();
        registered_tests->push_back(std::make_pair(name, test));
    }

    static void run_all() {
        if (!registered_tests)
            return;

        for (auto &item : *registered_tests) {
            std::string &name = item.first;
            replace(name, "uint32_t", "uint32");
            replace(name, "int32_t", "int32");
            replace(name, "uint64_t", "uint64");
            replace(name, "int64_t", "int64");
            replace(name, "float", "float32");
            replace(name, "half", "float16");
            replace(name, "double", "float64");
        }

        std::sort(
            registered_tests->begin(), registered_tests->end(),
            [](const auto &t1, const auto &t2) { return t1.first < t2.first; });

        std::string last_prefix;
        size_t nonvectorized_tests = 0, total_tests = 0;
        for (auto& test : *registered_tests) {
            std::string name = test.first, prefix = "";
            if (name.substr(0, 6) == "array_") {
                name = name.substr(6);
                auto offset = name.find("_");
                if (offset == std::string::npos)
                    throw std::out_of_range("Invalid test name");
                std::string type = name.substr(0, offset);
                name = name.substr(offset+1);
                offset = name.find("_");
                if (offset == std::string::npos)
                    throw std::out_of_range("Invalid test name");
                std::string length = name.substr(0, offset);
                while (!length.empty() && length[0] == '0')
                    length = length.substr(1);
                name = name.substr(offset+1);
                replace(length, "acc", ", approx=false");
                prefix = "Array<" + type + ", " + length + ">";
            }

            if (prefix != last_prefix) {
                if (!last_prefix.empty()) {
                    std::cout << std::endl
                              << "Done. (" << total_tests - nonvectorized_tests << "/"
                              << total_tests << " tests were vectorized)" << std::endl
                              << std::endl;
                    nonvectorized_tests = total_tests = 0;
                }

                if (prefix.empty())
                    std::cout << "Remaining tests .." << std::endl;
                else
                    std::cout << "Testing \"" << prefix << "\" .." << std::endl;
            }
            last_prefix = prefix;

            std::cout << "    " << name << ".. ";
            std::cout.flush();
            int nvi = test::nonvectorized_count;
            test.second();
            std::cout << "ok.";
            if (test::nonvectorized_count != nvi) {
                std::cout << " [non-vectorized]";
                nonvectorized_tests++;
            }
            std::cout << std::endl;
            total_tests++;
        }

        if (total_tests > 0) {
            std::cout << std::endl
                      << "Done. (" << total_tests - nonvectorized_tests << "/"
                      << total_tests << " tests were vectorized)" << std::endl
                      << std::endl;
        }
    }

private:
    static TestStorage *registered_tests;
};

Test::TestStorage *Test::registered_tests = nullptr;

/// Compute the difference in ULPs between two values
template <typename T> T ulpdiff(T ref, T val) {
    const T eps = std::numeric_limits<T>::epsilon() / 2;

    /* Express mantissa wrt. same exponent */
    int e_ref, e_val;
    T m_ref = std::frexp(ref, &e_ref);
    T m_val = std::frexp(val, &e_val);

    T diff;
    if (e_ref == e_val)
        diff = m_ref - m_val;
    else
        diff = m_ref - std::ldexp(m_val, e_val - e_ref);

    return std::abs(diff) / eps;
}

template <typename Scalar>
void assert_close(Scalar result, Scalar result_ref, Scalar eps) {
    using UInt = std::conditional_t<sizeof(Scalar) == 4, uint32_t, uint64_t>;
    using Float = std::conditional_t<sizeof(Scalar) == 4, float, double>;
    bool fail = false;

    if (eps == 0 || !std::isfinite(Float(result_ref))) {
        fail = memcpy_cast<UInt>(result) != memcpy_cast<UInt>(result_ref)
            && !(std::isnan(Float(result_ref)) && std::isnan(Float(result)));
    } else {
        fail = !(std::abs(Float(result - result_ref)) /
                  std::max(Float(1e-6), std::abs(Float(result_ref))) < Float(eps));
    }

    if (fail) {
        std::cerr << std::endl << "    * Failure: got "
            << result << " (0x" << std::hex << memcpy_cast<UInt>(result) << std::dec << "), expected "
            << result_ref << " (0x" << std::hex << memcpy_cast<UInt>(result_ref) << ")!" << std::endl;
        abort();
    }
}

template <typename T>
void probe_accuracy(T (*func)(const T &), double (*ref)(double),
                    typename T::Scalar min, typename T::Scalar max,
                    typename T::Scalar max_ulp_assert = 0,
                    bool check_special = true) {
    using Scalar = typename T::Scalar;

    Scalar max_err = 0, avg_err = 0;
    Scalar max_err_rel = 0, avg_err_rel = 0;
    Scalar avg_ulp = 0, max_ulp = 0;
    Scalar max_ulp_pos = -1;
    Scalar max_err_pos = -1;

    auto update_errors = [&](Scalar x, Scalar v, Scalar v_ref) {
        Scalar err = std::abs(v - v_ref);
        Scalar err_rel = err / std::abs(v_ref);
        if (v_ref == 0 && v == 0)
            err_rel = 0;

        Scalar ulp = ulpdiff(v_ref, v);
        if (ulp > max_ulp)
            max_ulp_pos = x;
        if (err > max_err)
            max_err_pos = x;

        avg_err += err;
        max_err = std::max(max_err, err);
        avg_err_rel += err_rel;
        max_err_rel = std::max(max_err_rel, err_rel);
        avg_ulp += ulp;
        max_ulp = std::max(max_ulp, ulp);
    };

    T value, result_ref;

    size_t idx = 0;
    size_t nTries = test::detailed ? 1000000 : 10000;
    for (size_t i = 0; i < nTries; i++) {
        value[idx] = min + (Scalar(i) / Scalar(nTries - 1)) * (max - min);
        result_ref[idx] = Scalar(ref(double(value[idx])));
        idx++;

        if (idx == value.size()) {
            auto&& result = func(value);
            for (size_t k = 0; k < idx; ++k)
                update_errors(value[k], result[k], result_ref[k]);
            idx = 0;
        }
    }

    if (idx > 0) {
        auto&& result = func(value);
        for (size_t k = 0; k < idx; ++k)
            update_errors(value[k], result[k], result_ref[k]);
    }

    if (check_special) {
        /// Test function behavior for +/- inf and NaN-valued inputs
        const Scalar inf = std::numeric_limits<Scalar>::infinity();
        const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
        assert_close(func(T(inf))[0], (Scalar) ref(double(inf)), Scalar(1e-6));
        assert_close(func(T(-inf))[0], (Scalar) ref(double(-inf)), Scalar(1e-6));
        assert_close(func(T(nan))[0], (Scalar) ref(double(nan)), Scalar(1e-6));
    }

    bool success = max_ulp <= max_ulp_assert || max_ulp_assert == 0;

    if (test::detailed || !success) {
        std::cout << "(in [" << min << ", " << max << "]):" << std::endl
              << "     * avg abs. err = " << avg_err / float(nTries) << std::endl
              << "     * avg rel. err = " << avg_err_rel / float(nTries) << std::endl
              << "        -> in ULPs  = " << avg_ulp / float(nTries) << std::endl
              << "     * max abs. err = " << max_err << std::endl
              << "       (at x=" << max_err_pos << ")" << std::endl
              << "     * max rel. err = " << max_err_rel << std::endl
              << "       -> in ULPs   = " << max_ulp << std::endl
              << "       (at x=" << max_ulp_pos << ")" << std::endl;
    }
    assert(success);
}

template <typename T>
void validate_unary(const std::vector<typename T::Scalar> &args,
                    T (*func)(const T &),
                    typename T::Scalar (*ref)(typename T::Scalar),
                    typename T::Scalar eps = 0) {
    T value, result_ref;
    size_t idx = 0;

    for (size_t i = 0; i < args.size(); ++i) {
        typename T::Scalar arg_i = args[i];
        value[idx] = arg_i;
        result_ref[idx] = ref(arg_i);
        idx++;

        if (idx == value.size()) {
            auto&& result = func(value);
            for (size_t k = 0; k < idx; ++k)
                assert_close(result[k], result_ref[k], eps);
            idx = 0;
        }
    }

    if (idx > 0) {
        auto&& result = func(value);
        for (size_t k = 0; k < idx; ++k)
            assert_close(result[k], result_ref[k], eps);
    }
}


template <typename T>
void validate_binary(const std::vector<typename T::Scalar> &args,
                     T (*func)(const T &, const T &),
                     typename T::Scalar (*ref)(typename T::Scalar,
                                               typename T::Scalar),
                     typename T::Scalar eps = 0) {
    using Scalar = typename T::Scalar;
    T value1, value2, result_ref;
    size_t idx = 0;

    for (size_t i=0; i<args.size(); ++i) {
        for (size_t j=0; j<args.size(); ++j) {
            Scalar arg_i = args[i], arg_j = args[j];
            value1[idx] = arg_i; value2[idx] = arg_j;
            result_ref[idx] = ref(arg_i, arg_j);
            idx++;

            if (idx == value1.size()) {
                auto&& result = func(value1, value2);
                for (size_t k = 0; k < idx; ++k)
                    assert_close(result[k], result_ref[k], eps);
                idx = 0;
            }
        }
    }

    if (idx > 0) {
        auto&& result = func(value1, value2);
        for (size_t k = 0; k < idx; ++k)
            assert_close(result[k], result_ref[k], eps);
    }
}

/// Generate an array containing values that are used to test various operations
template <typename Scalar> std::vector<Scalar> sample_values(bool has_nan = true) {
    std::vector<Scalar> args;
    if (std::is_integral<Scalar>::value) {
        args = { Scalar(0), Scalar(1),    Scalar(5),
                 Scalar(7), Scalar(1234), Scalar(1000000) };
        if (std::is_signed<Scalar>::value) {
            args.push_back(Scalar(-1));
            args.push_back(Scalar(-5));
            args.push_back(Scalar(-7));
            args.push_back(Scalar(-1234));
            args.push_back(Scalar(-1000000));
        }
    } else if (std::is_floating_point<Scalar>::value) {
        args = { Scalar(0), Scalar(0.5), Scalar(0.6), Scalar(1), Scalar(2),
                 Scalar(3), Scalar(M_PI), Scalar(-0), Scalar(-0.5), Scalar(-0.6),
                 Scalar(-1), Scalar(-2), Scalar(-3),
                 Scalar(std::numeric_limits<float>::infinity()),
                 Scalar(-std::numeric_limits<float>::infinity())
        };
        if (has_nan)
            args.push_back(Scalar(std::numeric_limits<float>::quiet_NaN()));
    }
    return args;
}

template <typename T>
void validate_ternary(const std::vector<typename T::Scalar> &args,
                      T (*func)(const T &, const T &, const T &),
                      typename T::Scalar (*refFunc)(typename T::Scalar,
                                                    typename T::Scalar,
                                                    typename T::Scalar),
                      typename T::Scalar eps = 0) {
    T value1, value2, value3, result_ref;
    size_t idx = 0;

    for (size_t i = 0; i < args.size(); ++i) {
        for (size_t j = 0; j < args.size(); ++j) {
            for (size_t k = 0; k < args.size(); ++k) {
                typename T::Scalar arg_i = args[i], arg_j = args[j],
                                   arg_k = args[k];
                value1[idx] = arg_i;
                value2[idx] = arg_j;
                value3[idx] = arg_k;
                result_ref[idx] = refFunc(arg_i, arg_j, arg_k);
                idx++;

                if (idx == value1.size()) {
                    auto &&result = func(value1, value2, value3);
                    for (size_t k = 0; k < idx; ++k)
                        assert_close(result[k], result_ref[k], eps);
                    idx = 0;
                }
            }
        }
    }

    if (idx > 0) {
        auto &&result = func(value1, value2, value3);
        for (size_t k = 0; k < idx; ++k)
            assert_close(result[k], result_ref[k], eps);
    }
}

template <typename T>
void validate_horizontal(const std::vector<typename T::Scalar> &args,
                         typename T::Scalar (*func)(const T &),
                         typename T::Scalar (*refFunc)(const T &),
                         typename T::Scalar eps = 0) {
    std::mt19937 gen;
    std::uniform_int_distribution<> dis(0, (int) args.size()-1);
    T value;

    for (int i=0; i<1000; ++i) {
        for (size_t i=0; i<value.size(); ++i)
            value[i] = args[(size_t) dis(gen)];

        assert_close(func(value), refFunc(value), eps);
    }
}

NAMESPACE_END(test)

#define ENOKI_TEST_HELPER(name, type)                                           \
    ENOKI_TEST(array_##type##_01##_##name) { name<type, 1>();  }                \
    ENOKI_TEST(array_##type##_02##_##name) { name<type, 2>();  }                \
    ENOKI_TEST(array_##type##_03##_##name) { name<type, 3>();  }                \
    ENOKI_TEST(array_##type##_04##_##name) { name<type, 4>();  }                \
    ENOKI_TEST(array_##type##_08##_##name) { name<type, 8>();  }                \
    ENOKI_TEST(array_##type##_16##_##name) { name<type, 16>(); }                \
    ENOKI_TEST(array_##type##_31##_##name) { name<type, 31>(); }                \
    ENOKI_TEST(array_##type##_32##_##name) { name<type, 32>(); }

#define ENOKI_TEST_TYPE(name, type)                                             \
    template <typename Scalar, size_t Size,                                     \
              bool Approx = std::is_same<Scalar, float>::value,                 \
              typename T = enoki::Array<Scalar, Size, Approx>>                  \
    void name##_##type();                                                       \
    ENOKI_TEST_HELPER(name##_##type, type)                                      \
    template <typename Scalar, size_t Size, bool Approx, typename T>            \
    void name##_##type()

#define ENOKI_TEST_FLOAT(name)                                                  \
    template <typename Scalar, size_t Size,                                     \
              bool Approx = std::is_same<Scalar, float>::value,                 \
              typename T = enoki::Array<Scalar, Size, Approx>>                  \
    void name();                                                                \
    ENOKI_TEST(array_float_01acc_##name) { name<float, 1, false>();  }          \
    ENOKI_TEST_HELPER(name, float)                                              \
    ENOKI_TEST_HELPER(name, double)                                             \
    template <typename Scalar, size_t Size, bool Approx, typename T>            \
    void name()

#define ENOKI_TEST_INT(name)                                                    \
    template <typename Scalar, size_t Size,                                     \
              typename T = enoki::Array<Scalar, Size>>                          \
    void name();                                                                \
    ENOKI_TEST_HELPER(name, int32_t)                                            \
    ENOKI_TEST_HELPER(name, uint32_t)                                           \
    ENOKI_TEST_HELPER(name, int64_t)                                            \
    ENOKI_TEST_HELPER(name, uint64_t)                                           \
    template <typename Scalar, size_t Size, typename T>                         \
    void name()

#define ENOKI_TEST_ALL(name)                                                    \
    template <typename Scalar, size_t Size,                                     \
              bool Approx = std::is_same<Scalar, float>::value,                 \
              typename T = enoki::Array<Scalar, Size, Approx>>                  \
    void name();                                                                \
    ENOKI_TEST(array_float_01acc_##name) { name<float, 1, false>();  }          \
    ENOKI_TEST_HELPER(name, float)                                              \
    ENOKI_TEST_HELPER(name, double)                                             \
    ENOKI_TEST_HELPER(name, int32_t)                                            \
    ENOKI_TEST_HELPER(name, uint32_t)                                           \
    ENOKI_TEST_HELPER(name, int64_t)                                            \
    ENOKI_TEST_HELPER(name, uint64_t)                                           \
    template <typename Scalar, size_t Size, bool Approx, typename T>            \
    void name()

int main(int argc, char** argv) {
    std::cout << "=== Enoki test suite ===" << std::endl;
    std::cout << "Enabled compiler features: ";

    if (has_avx512dq) std::cout << "avx512dq ";
    if (has_avx512bw) std::cout << "avx512bw ";
    if (has_avx512vl) std::cout << "avx512vl ";
    if (has_avx512er) std::cout << "avx512eri ";
    if (has_avx512pf) std::cout << "avx512pfi ";
    if (has_avx512cd) std::cout << "avx512cdi ";
    if (has_avx512f)  std::cout << "avx512f ";
    if (has_avx2)     std::cout << "avx2 ";
    if (has_avx)      std::cout << "avx ";
    if (has_fma)      std::cout << "fma ";
    if (has_f16c)     std::cout << "f16c ";
    if (has_sse42)    std::cout << "sse4.2 ";

    /* Turn on verbose mode if requested */
    if (argc == 2 && strcmp(argv[1], "-v") == 0)
        test::detailed = true;

    std::cout << std::endl;
    std::cout << std::endl;
    test::Test::run_all();
    return 0;
}
