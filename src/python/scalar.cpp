#include "common.h"

extern void bind_scalar_0d(py::module&, py::module&);
extern void bind_scalar_1d(py::module&, py::module&);
extern void bind_scalar_2d(py::module&, py::module&);
extern void bind_scalar_3d(py::module&, py::module&);
extern void bind_scalar_4d(py::module&, py::module&);
extern void bind_scalar_complex(py::module&, py::module&);
extern void bind_scalar_matrix(py::module&, py::module&);
extern void bind_scalar_pcg32(py::module&, py::module&);

bool *disable_print_flag = nullptr;

PYBIND11_MODULE(scalar, s) {
    py::module m = py::module::import("enoki");

    disable_print_flag = (bool *) py::get_shared_data("disable_print_flag");

    bind_scalar_0d(m, s);
    bind_scalar_1d(m, s);
    bind_scalar_2d(m, s);
    bind_scalar_3d(m, s);
    bind_scalar_4d(m, s);
    bind_scalar_complex(m, s);
    bind_scalar_matrix(m, s);
    bind_scalar_pcg32(m, s);
}
