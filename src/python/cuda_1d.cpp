#include "common.h"

void bind_cuda_1d(py::module& m) {
    bind<mask_t<FloatC>>(m, "BoolC");

    bind<FloatC>(m, "FloatC")
        .def(py::init<const UInt32C &>())
        .def(py::init<const UInt64C &>());

    bind<UInt32C>(m, "UInt32C")
        .def(py::init<const FloatC &>())
        .def(py::init<const UInt64C &>());

    bind<UInt64C>(m, "UInt64C")
        .def(py::init<const FloatC &>())
        .def(py::init<const UInt32C &>());
}
