#include "common.h"

void bind_cuda_autodiff_complex(py::module& m) {
    bind_complex<Complex2fD>(m, "Complex2fD")
        .def(py::init<const Complex2f &>())
        .def(py::init<const Complex2fC &>());

    bind_complex<Complex24fD>(m, "Complex24fD")
        .def(py::init<const Complex24f &>())
        .def(py::init<const Complex24fC &>());
}
