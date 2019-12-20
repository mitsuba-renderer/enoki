#include "common.h"

void bind_dynamic_4d(py::module& m) {
    auto vector4m_class = bind<Vector4mX>(m, "Vector4mX");
    auto vector4f_class = bind<Vector4fX>(m, "Vector4fX");
    auto vector4i_class = bind<Vector4iX>(m, "Vector4iX");
    auto vector4u_class = bind<Vector4uX>(m, "Vector4uX");

    vector4f_class
        .def(py::init<const Vector4uX &>())
        .def(py::init<const Vector4iX &>());

    vector4i_class
        .def(py::init<const Vector4fX &>())
        .def(py::init<const Vector4uX &>());

    vector4u_class
        .def(py::init<const Vector4fX &>())
        .def(py::init<const Vector4iX &>());
}
