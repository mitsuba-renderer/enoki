#include "common.h"

void bind_autodiff_2d(py::module& m) {
    bind<Vector2bD>(m, "Vector2bD")
        .def(py::init<const Vector2bC &>());

    bind<Vector2fD>(m, "Vector2fD")
        .def(py::init<const Vector2fC &>())
        .def(py::init<const Vector2uD &>());

    bind<Vector2uD>(m, "Vector2uD")
        .def(py::init<const Vector2uC &>())
        .def(py::init<const Vector2fD &>());
}
