#include "common.h"

void bind_cuda_2d(py::module& m) {
    bind<Vector2bC>(m, "Vector2bC");

    bind<Vector2fC>(m, "Vector2fC")
        .def(py::init<const Vector2uC &>());

    bind<Vector2uC>(m, "Vector2uC")
        .def(py::init<const Vector2fC &>());
}
