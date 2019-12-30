#include "common.h"
#include <pybind11/functional.h>

bool __implicit_conversion = false;

bool allclose_py(const py::object &a, const py::object &b,
                 const py::float_ &rtol, const py::float_ &atol,
                 bool equal_nan) {
    ssize_t la = PyObject_Length(a.ptr()),
            lb = PyObject_Length(b.ptr());

    const char *tp_name_a = a.ptr()->ob_type->tp_name,
               *tp_name_b = b.ptr()->ob_type->tp_name;

    if (la == -1 || lb == -1)
        PyErr_Clear();

    if (la != lb)
        throw std::runtime_error("enoki.allclose(): length mismatch!");

    bool ok_a = PyNumber_Check(a.ptr()) || strncmp(tp_name_a, "enoki.", 6) == 0;
    bool ok_b = PyNumber_Check(b.ptr()) || strncmp(tp_name_b, "enoki.", 6) == 0;

    if (ok_a && ok_b) {
        py::module ek = py::module::import("enoki");

        py::object abs        = ek.attr("abs"),
                   eq         = ek.attr("eq"),
                   isnan      = ek.attr("isnan"),
                   all_nested = ek.attr("all_nested");

        py::object lhs = abs(a - b),
                   rhs = atol + abs(b) * rtol;

        py::object cond =
            py::reinterpret_steal<py::object>(PyObject_RichCompare(lhs.ptr(), rhs.ptr(), Py_LE));

        if (!cond)
            throw py::error_already_set();

        if (equal_nan)
            cond = cond | (isnan(a) & isnan(b));

        return py::cast<bool>(all_nested(cond));
    } else if (la >= 0) {
        for (size_t i = 0; i < (size_t) la; ++i) {
            py::int_ key(i);
            if (!allclose_py(a[key], b[key], rtol, atol, equal_nan))
                return false;
        }
    } else {
        throw std::runtime_error("enoki.allclose(): unsupported type!");
    }

    return true;
}

PYBIND11_MODULE(core, m_) {
    ENOKI_MARK_USED(m_);
    py::module m = py::module::import("enoki");

    m.attr("__version__") = ENOKI_VERSION;
    py::set_shared_data("implicit_conversion", &__implicit_conversion);

    py::class_<Buffer<false>>(m, "CPUBuffer");

    m.def("zero",
        [](py::handle h, size_t size) {
            if (size == 1)
                return h(0);
            else
                return h.attr("zero")(size);
        },
        "type"_a, "size"_a = 1);

    m.def("empty",
        [](py::handle h, size_t size) {
            if (size == 1)
                return h();
            else
                return h.attr("empty")(size);
        },
        "type"_a, "size"_a = 1);

    m.def("linspace",
        [](py::handle h, py::handle start, py::handle end, size_t size) {
            if (size == 1)
                return h(start);
            else
                return h.attr("linspace")(start, end, size);
        },
        "type"_a, "start"_a, "end"_a, "size"_a = 1);

    m.def("allclose", &allclose_py,
        "a"_a, "b"_a, "rtol"_a = 1e-5, "atol"_a = 1e-8,
        "equal_nan"_a = false
    );
}
