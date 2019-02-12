#include "common.h"

void bind_autodiff_4d(py::module& m) {
    bind<Vector4bD>(m, "Vector4bD")
        .def(py::init<const Vector4bC &>());

    bind<Vector4fD>(m, "Vector4fD")
        .def(py::init<const Vector4fC &>())
        .def(py::init<const Vector4uD &>());

    bind<Vector4uD>(m, "Vector4uD")
        .def(py::init<const Vector4uC &>())
        .def(py::init<const Vector4fD &>());
}
