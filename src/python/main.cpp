#include "common.h"
#include <pybind11/functional.h>

bool __implicit_conversion = false;

bool allclose_py(py::object value, py::object ref, double relerr, double abserr) {
    ssize_t l1 = PyObject_Length(value.ptr()),
            l2 = PyObject_Length(ref.ptr());

    const char *tp_name_1 = value.ptr()->ob_type->tp_name,
               *tp_name_2 =   ref.ptr()->ob_type->tp_name;

    if (l1 == -1 || l2 == -1)
        PyErr_Clear();

    if (l1 != l2)
        throw std::runtime_error("enoki.allclose(): length mismatch!");

    bool ok_1 = PyNumber_Check(value.ptr()) || strncmp(tp_name_1, "enoki.", 6) == 0;
    bool ok_2 = PyNumber_Check(ref.ptr())   || strncmp(tp_name_2, "enoki.", 6) == 0;

    if (ok_1 && ok_2) {
        py::module ek = py::module::import("enoki");

        py::object abs         = ek.attr("abs"),
                   hmax_nested = ek.attr("hmax_nested");

        py::object abserr_value = hmax_nested(abs(value - ref)),
                   relerr_value = abserr_value / hmax_nested(abs(ref));

        if (!PyNumber_Check(abserr_value.ptr()))
            abserr_value = abserr_value[py::int_(0)];

        if (!PyNumber_Check(relerr_value.ptr()))
            relerr_value = relerr_value[py::int_(0)];

        return py::cast<double>(abserr_value) < abserr ||
               py::cast<double>(relerr_value) < relerr;
    } else if (l1 >= 0) {
        for (size_t i = 0; i < (size_t) l1; ++i) {
            py::int_  key(i);
            if (!allclose_py(value[key], ref[key], relerr, abserr))
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
        "value"_a, "ref"_a, "relerr"_a = 1e-5, "abserr"_a = 1e-5
    );
}
