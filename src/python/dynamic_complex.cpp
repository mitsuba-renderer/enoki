#include "common.h"

void bind_dynamic_complex(py::module& m) {
    bind_complex<Complex2fX>(m, "Complex2fX")
        .def(py::init<const Complex2f &>());

    bind_complex<Complex24fX>(m, "Complex24fX")
        .def(py::init<const Complex24f &>());
}
