#include "common.h"

void bind_cuda_3d(py::module& m) {
    bind<Vector3bC>(m, "Vector3bC");

    bind<Vector3fC>(m, "Vector3fC")
        .def(py::init<const Vector3uC &>());

    bind<Vector3uC>(m, "Vector3uC")
        .def(py::init<const Vector3fC &>());
}
