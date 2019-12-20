#include "common.h"

void bind_scalar_2d(py::module& m) {
    auto vector2m_class = bind<Vector2m>(m, "Vector2m");
    auto vector2f_class = bind<Vector2f>(m, "Vector2f");
    auto vector2i_class = bind<Vector2i>(m, "Vector2i");
    auto vector2u_class = bind<Vector2u>(m, "Vector2u");

    vector2f_class
        .def(py::init<const Vector2u &>())
        .def(py::init<const Vector2i &>());

    vector2i_class
        .def(py::init<const Vector2f &>())
        .def(py::init<const Vector2u &>());

    vector2u_class
        .def(py::init<const Vector2f &>())
        .def(py::init<const Vector2i &>());
}
