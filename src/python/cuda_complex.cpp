#include "common.h"

void bind_cuda_complex(py::module& m) {
    bind_complex<Complex2fC>(m, "Complex2fC")
        .def(py::init<const Complex2f &>());

    bind_complex<Complex24fC>(m, "Complex24fC")
        .def(py::init<const Complex24f &>());
}
