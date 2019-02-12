#include "common.h"

void bind_autodiff_3d(py::module& m) {
    bind<Vector3bD>(m, "Vector3bD")
        .def(py::init<const Vector3bC &>());

    bind<Vector3fD>(m, "Vector3fD")
        .def(py::init<const Vector3fC &>())
        .def(py::init<const Vector3uD &>());

    bind<Vector3uD>(m, "Vector3uD")
        .def(py::init<const Vector3uC &>())
        .def(py::init<const Vector3fD &>());
}
