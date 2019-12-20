#include "common.h"

void bind_dynamic_2d(py::module& m) {
    auto vector2b_class = bind<Vector2bX>(m, "Vector2bX");
    auto vector2f_class = bind<Vector2fX>(m, "Vector2fX");
    auto vector2i_class = bind<Vector2iX>(m, "Vector2iX");
    auto vector2u_class = bind<Vector2uX>(m, "Vector2uX");

    vector2f_class
        .def(py::init<const Vector2uX &>())
        .def(py::init<const Vector2iX &>());

    vector2i_class
        .def(py::init<const Vector2fX &>())
        .def(py::init<const Vector2uX &>());

    vector2u_class
        .def(py::init<const Vector2fX &>())
        .def(py::init<const Vector2iX &>());
}
