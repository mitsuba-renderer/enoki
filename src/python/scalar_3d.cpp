#include "common.h"

void bind_scalar_3d(py::module& m) {
    auto vector3m_class = bind<Vector3m>(m, "Vector3m");
    auto vector3f_class = bind<Vector3f>(m, "Vector3f");
    auto vector3i_class = bind<Vector3i>(m, "Vector3i");
    auto vector3u_class = bind<Vector3u>(m, "Vector3u");

    vector3f_class
        .def(py::init<const Vector3u &>())
        .def(py::init<const Vector3i &>());

    vector3i_class
        .def(py::init<const Vector3f &>())
        .def(py::init<const Vector3u &>());

    vector3u_class
        .def(py::init<const Vector3f &>())
        .def(py::init<const Vector3i &>());
}
