#include "common.h"

extern void bind_dynamic_0d(py::module&, py::module&);
extern void bind_dynamic_1d(py::module&, py::module&);
extern void bind_dynamic_2d(py::module&, py::module&);
extern void bind_dynamic_3d(py::module&, py::module&);
extern void bind_dynamic_4d(py::module&, py::module&);
extern void bind_dynamic_complex(py::module&, py::module&);
extern void bind_dynamic_matrix(py::module&, py::module&);
extern void bind_dynamic_pcg32(py::module&, py::module&);

bool *disable_print_flag = nullptr;

PYBIND11_MODULE(dynamic, s) {
    py::module m = py::module::import("enoki");

    disable_print_flag = (bool *) py::get_shared_data("disable_print_flag");

    bind_dynamic_0d(m, s);
    bind_dynamic_1d(m, s);
    bind_dynamic_2d(m, s);
    bind_dynamic_3d(m, s);
    bind_dynamic_4d(m, s);
    bind_dynamic_complex(m, s);
    bind_dynamic_matrix(m, s);
    bind_dynamic_pcg32(m, s);
}
