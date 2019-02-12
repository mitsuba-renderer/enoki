#include "common.h"

void bind_cuda_4d(py::module& m) {
    bind<Vector4bC>(m, "Vector4bC");

    bind<Vector4fC>(m, "Vector4fC")
        .def(py::init<const Vector4uC &>());

    bind<Vector4uC>(m, "Vector4uC")
        .def(py::init<const Vector4fC &>());
}
