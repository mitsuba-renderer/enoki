#include "common.h"

void bind_cuda_autodiff_3d(py::module& m) {
    auto vector3m_class = bind<Vector3mD>(m, "Vector3mD");
    auto vector3f_class = bind<Vector3fD>(m, "Vector3fD");
    auto vector3i_class = bind<Vector3iD>(m, "Vector3iD");
    auto vector3u_class = bind<Vector3uD>(m, "Vector3uD");

    vector3m_class
        .def(py::init<const Vector3mC &>());

    vector3f_class
        .def(py::init<const Vector3fC &>())
        .def(py::init<const Vector3uD &>())
        .def(py::init<const Vector3iD &>());

    vector3i_class
        .def(py::init<const Vector3iC &>())
        .def(py::init<const Vector3fD &>())
        .def(py::init<const Vector3uD &>());

    vector3u_class
        .def(py::init<const Vector3uC &>())
        .def(py::init<const Vector3fD &>())
        .def(py::init<const Vector3iD &>());
}
