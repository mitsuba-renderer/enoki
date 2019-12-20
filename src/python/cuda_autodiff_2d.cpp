#include "common.h"

void bind_cuda_autodiff_2d(py::module& m) {
    auto vector2m_class = bind<Vector2mD>(m, "Vector2mD");
    auto vector2f_class = bind<Vector2fD>(m, "Vector2fD");
    auto vector2i_class = bind<Vector2iD>(m, "Vector2iD");
    auto vector2u_class = bind<Vector2uD>(m, "Vector2uD");

    vector2m_class
        .def(py::init<const Vector2mC &>());

    vector2f_class
        .def(py::init<const Vector2fC &>())
        .def(py::init<const Vector2uD &>())
        .def(py::init<const Vector2iD &>());

    vector2i_class
        .def(py::init<const Vector2iC &>())
        .def(py::init<const Vector2fD &>())
        .def(py::init<const Vector2uD &>());

    vector2u_class
        .def(py::init<const Vector2uC &>())
        .def(py::init<const Vector2fD &>())
        .def(py::init<const Vector2iD &>());
}
