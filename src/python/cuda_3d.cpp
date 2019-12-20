#include "common.h"

void bind_cuda_3d(py::module& m) {
    auto vector3m_class = bind<Vector3mC>(m, "Vector3mC");
    auto vector3f_class = bind<Vector3fC>(m, "Vector3fC");
    auto vector3i_class = bind<Vector3iC>(m, "Vector3iC");
    auto vector3u_class = bind<Vector3uC>(m, "Vector3uC");

    vector3f_class
        .def(py::init<const Vector3f &>())
        .def(py::init<const Vector3iC &>())
        .def(py::init<const Vector3uC &>());

    vector3i_class
        .def(py::init<const Vector3i &>())
        .def(py::init<const Vector3uC &>())
        .def(py::init<const Vector3fC &>());

    vector3u_class
        .def(py::init<const Vector3u &>())
        .def(py::init<const Vector3iC &>())
        .def(py::init<const Vector3fC &>());
}
