#include "common.h"

void bind_cuda_1d(py::module& m) {
    bind<FloatC>(m, "FloatC")
        .def(py::init<const UInt32C &>());

    bind<UInt32C>(m, "UInt32C")
        .def(py::init<const FloatC &>());

    bind<mask_t<FloatC>>(m, "BoolC");
}
