#include "common.h"

void bind_autodiff_1d(py::module& m) {
    bind<FloatD>(m, "FloatD")
        .def(py::init<const FloatC &>())
        .def(py::init<const UInt32D &>());

    bind<UInt32D>(m, "UInt32D")
        .def(py::init<const UInt32C &>())
        .def(py::init<const FloatD &>());

    bind<mask_t<FloatD>>(m, "BoolD")
        .def(py::init<const BoolC &>());
}
