#include "common.h"

void bind_scalar_4d(py::module& m) {
    auto vector4b_class = bind<Vector4b>(m, "Vector4b");
    auto vector4f_class = bind<Vector4f>(m, "Vector4f");
    auto vector4i_class = bind<Vector4i>(m, "Vector4i");
    auto vector4u_class = bind<Vector4u>(m, "Vector4u");

    vector4f_class
        .def(py::init<const Vector4u &>())
        .def(py::init<const Vector4i &>());

    vector4i_class
        .def(py::init<const Vector4f &>())
        .def(py::init<const Vector4u &>());

    vector4u_class
        .def(py::init<const Vector4f &>())
        .def(py::init<const Vector4i &>());
}
