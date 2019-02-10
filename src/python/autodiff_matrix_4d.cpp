#include "common.h"

void bind_autodiff_matrix_4d(py::module& m) {
    bind_matrix<Matrix4fD>(m, "Matrix4fD");

    m.def("detach", [](const Matrix4fD &a) { return detach(a); });
    m.def("requires_gradient",
          [](const Matrix4fD &a) { return requires_gradient(a); },
          "array"_a);

    m.def("set_requires_gradient",
          [](Matrix4fD &a, bool value) { set_requires_gradient(a, value); },
          "array"_a, "value"_a = true);

    m.def("gradient", [](Matrix4fD &a) { return gradient(a); });
    m.def("set_gradient",
          [](Matrix4fD &a, const Matrix4fD &g) { set_gradient(a, detach(g)); });

    m.def("graphviz", [](const Matrix4fD &a) { return graphviz(a); });

    m.def("set_label", [](const Matrix4fD &a, const char *label) {
        set_label(a, label);
    });
}
