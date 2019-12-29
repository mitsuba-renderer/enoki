#include "common.h"
#include <pybind11/functional.h>

bool __implicit_conversion = false;

PYBIND11_MODULE(core, m_) {
    ENOKI_MARK_USED(m_);
    py::module m = py::module::import("enoki");

    m.attr("__version__") = ENOKI_VERSION;
    py::set_shared_data("implicit_conversion", &__implicit_conversion);

    py::class_<Buffer>(m, "Buffer");

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
}
