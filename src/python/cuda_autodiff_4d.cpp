#include "common.h"

void bind_cuda_autodiff_4d(py::module& m) {
    auto vector4m_class = bind<Vector4mD>(m, "Vector4mD");
    auto vector4f_class = bind<Vector4fD>(m, "Vector4fD");
    auto vector4i_class = bind<Vector4iD>(m, "Vector4iD");
    auto vector4u_class = bind<Vector4uD>(m, "Vector4uD");

    vector4m_class
        .def(py::init<const Vector4mC &>());

    vector4f_class
        .def(py::init<const Vector4fC &>())
        .def(py::init<const Vector4uD &>())
        .def(py::init<const Vector4iD &>());

    vector4i_class
        .def(py::init<const Vector4iC &>())
        .def(py::init<const Vector4fD &>())
        .def(py::init<const Vector4uD &>());

    vector4u_class
        .def(py::init<const Vector4uC &>())
        .def(py::init<const Vector4fD &>())
        .def(py::init<const Vector4iD &>());
}
