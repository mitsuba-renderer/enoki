#include "common.h"

extern void bind_scalar_0d(py::module&, py::module&);
extern void bind_scalar_1d(py::module&, py::module&);
extern void bind_scalar_2d(py::module&, py::module&);
extern void bind_scalar_3d(py::module&, py::module&);
extern void bind_scalar_4d(py::module&, py::module&);
extern void bind_scalar_complex(py::module&, py::module&);
extern void bind_scalar_matrix(py::module&, py::module&);
extern void bind_scalar_pcg32(py::module&, py::module&);

bool *implicit_conversion = nullptr;

PYBIND11_MODULE(scalar, s) {
    py::module m = py::module::import("enoki");

    implicit_conversion = (bool *) py::get_shared_data("implicit_conversion");

    bind_scalar_0d(m, s);
    bind_scalar_1d(m, s);
    bind_scalar_2d(m, s);
    bind_scalar_3d(m, s);
    bind_scalar_4d(m, s);
    bind_scalar_complex(m, s);
    bind_scalar_matrix(m, s);
    bind_scalar_pcg32(m, s);
}
