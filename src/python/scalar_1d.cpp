#include "common.h"
#include <pybind11/functional.h>

void bind_scalar_1d(py::module& m) {
    using Float = float;

    m.def("fmadd", [](Float a, Float b, Float c) {
        return enoki::fmadd(a, b, c);
    });
    m.def("fmsub", [](Float a, Float b, Float c) {
        return enoki::fmsub(a, b, c);
    });
    m.def("fnmadd", [](Float a, Float b, Float c) {
        return enoki::fnmadd(a, b, c);
    });
    m.def("fnmsub", [](Float a, Float b, Float c) {
        return enoki::fnmsub(a, b, c);
    });

    m.def("abs",   [](Float a) { return enoki::abs(a); });
    m.def("sqr",   [](Float a) { return enoki::sqr(a); });
    m.def("sqrt",  [](Float a) { return enoki::sqrt(a); });
    m.def("cbrt",  [](Float a) { return enoki::cbrt(a); });
    m.def("rcp",   [](Float a) { return enoki::rcp(a); });
    m.def("rsqrt", [](Float a) { return enoki::rsqrt(a); });

    m.def("ceil",  [](Float a) { return enoki::ceil(a); });
    m.def("floor", [](Float a) { return enoki::floor(a); });
    m.def("round", [](Float a) { return enoki::round(a); });
    m.def("trunc", [](Float a) { return enoki::trunc(a); });

    m.def("sin",    [](Float a) { return enoki::sin(a); });
    m.def("cos",    [](Float a) { return enoki::cos(a); });
    m.def("sincos", [](Float a) { return enoki::sincos(a); });
    m.def("tan",    [](Float a) { return enoki::tan(a); });
    m.def("sec",    [](Float a) { return enoki::sec(a); });
    m.def("csc",    [](Float a) { return enoki::csc(a); });
    m.def("cot",    [](Float a) { return enoki::cot(a); });
    m.def("asin",   [](Float a) { return enoki::asin(a); });
    m.def("acos",   [](Float a) { return enoki::acos(a); });
    m.def("atan",   [](Float a) { return enoki::atan(a); });
    m.def("atan2",  [](Float a, Float b) {
        return enoki::atan2(a, b);
    });

    m.def("sinh",    [](Float a) { return enoki::sinh(a); });
    m.def("cosh",    [](Float a) { return enoki::cosh(a); });
    m.def("sincosh", [](Float a) { return enoki::sincosh(a); });
    m.def("tanh",    [](Float a) { return enoki::tanh(a); });
    m.def("sech",    [](Float a) { return enoki::sech(a); });
    m.def("csch",    [](Float a) { return enoki::csch(a); });
    m.def("coth",    [](Float a) { return enoki::coth(a); });
    m.def("asinh",   [](Float a) { return enoki::asinh(a); });
    m.def("acosh",   [](Float a) { return enoki::acosh(a); });
    m.def("atanh",   [](Float a) { return enoki::atanh(a); });

    m.def("hsum",    [](Float a) { return enoki::hsum(a); });
    m.def("hprod",   [](Float a) { return enoki::hprod(a); });
    m.def("hmin",    [](Float a) { return enoki::hmin(a); });
    m.def("hmax",    [](Float a) { return enoki::hmax(a); });
    m.def("psum",    [](Float a) { return enoki::psum(a); });
    m.def("reverse", [](Float a) { return enoki::reverse(a); });

    m.def("log",    [](Float a) { return enoki::log(a); });
    m.def("exp",    [](Float a) { return enoki::exp(a); });
    m.def("erfinv", [](Float a) { return enoki::erfinv(a); });
    m.def("erf",    [](Float a) { return enoki::erf(a); });
    m.def("pow",    [](Float a, Float b) {
        return enoki::pow(a, b);
    });

    m.def("lerp", [](Float a, Float b, Float t) {
        return enoki::lerp(a, b, t);
    });
    m.def("clamp", [](Float value, Float min, Float max) {
        return enoki::clamp(value, min, max);
    });

    m.def("isfinite", [](Float a) { return enoki::isfinite(a); });
    m.def("isnan", [](Float a) { return enoki::isnan(a); });
    m.def("isinf", [](Float a) { return enoki::isinf(a); });

    m.def("tzcnt",  [](size_t a) { return enoki::tzcnt(a); });
    m.def("lzcnt",  [](size_t a) { return enoki::lzcnt(a); });
    m.def("popcnt", [](size_t a) { return enoki::popcnt(a); });
    m.def("log2i",  [](size_t a) { return enoki::log2i(a); });

    bind<Vector0m>(m, "Vector0m");
    bind<Vector0f>(m, "Vector0f");
    bind<Vector1m>(m, "Vector1m");
    bind<Vector1f>(m, "Vector1f");

    m.def(
        "binary_search",
        [](uint32_t start,
           uint32_t end,
           const std::function<bool(uint32_t, bool)> &pred,
           bool mask) {
            return enoki::binary_search(start, end, pred, mask);
        },
        "start"_a, "end"_a, "pred"_a, "mask"_a = true);
}
