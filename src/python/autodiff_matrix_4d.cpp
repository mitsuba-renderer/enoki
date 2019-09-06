#include "common.h"

void bind_autodiff_matrix_4d(py::module& m) {
    bind_matrix<Matrix4fD>(m, "Matrix4fD");

    m.def("detach", [](const Matrix4fD &a) -> Matrix4fC { return detach(a); });
    m.def("requires_gradient",
          [](const Matrix4fD &a) { return requires_gradient(a); },
          "array"_a);

    m.def("set_requires_gradient",
          [](Matrix4fD &a, bool value) { set_requires_gradient(a, value); },
          "array"_a, "value"_a = true);

    m.def("gradient", [](Matrix4fD &a) { return eval(gradient(a)); });
    m.def("set_gradient",
          [](Matrix4fD &a, const Matrix4fC &g, bool b) { set_gradient(a, g, b); },
          "array"_a, "gradient"_a, "backward"_a = true);

    m.def("graphviz", [](const Matrix4fD &a) { return graphviz(a); });

    m.def("set_label", [](const Matrix4fD &a, const char *label) {
        set_label(a, label);
    });
}
