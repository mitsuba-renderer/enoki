#include "common.h"

void bind_cuda_4d(py::module& m) {
    auto vector4m_class = bind<Vector4mC>(m, "Vector4mC");
    auto vector4f_class = bind<Vector4fC>(m, "Vector4fC");
    auto vector4i_class = bind<Vector4iC>(m, "Vector4iC");
    auto vector4u_class = bind<Vector4uC>(m, "Vector4uC");

    vector4f_class
        .def(py::init<const Vector4uC &>())
        .def(py::init<const Vector4iC &>());

    vector4i_class
        .def(py::init<const Vector4fC &>())
        .def(py::init<const Vector4uC &>());

    vector4u_class
        .def(py::init<const Vector4fC &>())
        .def(py::init<const Vector4iC &>());
}
