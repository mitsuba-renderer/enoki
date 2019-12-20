#include "common.h"

void bind_cuda_2d(py::module& m) {
    auto vector2m_class = bind<Vector2mC>(m, "Vector2mC");
    auto vector2f_class = bind<Vector2fC>(m, "Vector2fC");
    auto vector2i_class = bind<Vector2iC>(m, "Vector2iC");
    auto vector2u_class = bind<Vector2uC>(m, "Vector2uC");

    vector2f_class
        .def(py::init<const Vector2f &>())
        .def(py::init<const Vector2uC &>())
        .def(py::init<const Vector2iC &>());

    vector2i_class
        .def(py::init<const Vector2i &>())
        .def(py::init<const Vector2fC &>())
        .def(py::init<const Vector2uC &>());

    vector2u_class
        .def(py::init<const Vector2u &>())
        .def(py::init<const Vector2fC &>())
        .def(py::init<const Vector2iC &>());
}
