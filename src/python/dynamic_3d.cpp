#include "common.h"

void bind_dynamic_3d(py::module& m) {
    auto vector3m_class = bind<Vector3mX>(m, "Vector3mX");
    auto vector3f_class = bind<Vector3fX>(m, "Vector3fX");
    auto vector3i_class = bind<Vector3iX>(m, "Vector3iX");
    auto vector3u_class = bind<Vector3uX>(m, "Vector3uX");

    vector3f_class
        .def(py::init<const Vector3uX &>())
        .def(py::init<const Vector3iX &>());

    vector3i_class
        .def(py::init<const Vector3fX &>())
        .def(py::init<const Vector3uX &>());

    vector3u_class
        .def(py::init<const Vector3fX &>())
        .def(py::init<const Vector3iX &>());
}
