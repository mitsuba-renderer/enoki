#include "common.h"

void bind_autodiff_1d(py::module& m) {
    bind<FloatD>(m, "FloatD")
        .def(py::init<const FloatC &>())
        .def(py::init<const UInt32D &>())
        .def("set_log_level",
             [](int log_level) { FloatD::set_log_level_(log_level); },
             "Sets the current log level (0 == none, 1 == minimal, 2 == moderate, 3 == high, 4 == everything)");

    bind<UInt32D>(m, "UInt32D")
        .def(py::init<const UInt32C &>())
        .def(py::init<const FloatD &>());

    bind<mask_t<FloatD>>(m, "BoolD")
        .def(py::init<const BoolC &>());
}
