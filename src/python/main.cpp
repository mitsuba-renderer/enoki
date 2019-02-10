#include "common.h"

extern void bind_cuda_1d(py::module&);
extern void bind_cuda_2d(py::module&);
extern void bind_cuda_3d(py::module&);
extern void bind_cuda_4d(py::module&);
extern void bind_cuda_matrix_4d(py::module&);
extern void bind_autodiff_1d(py::module&);
extern void bind_autodiff_2d(py::module&);
extern void bind_autodiff_3d(py::module&);
extern void bind_autodiff_4d(py::module&);
extern void bind_autodiff_matrix_4d(py::module&);
extern void bind_pcg32(py::module&);

PYBIND11_MODULE(enoki, m) {
    bind_cuda_1d(m);
    bind_cuda_2d(m);
    bind_cuda_3d(m);
    bind_cuda_4d(m);
    bind_cuda_matrix_4d(m);
    bind_autodiff_1d(m);
    bind_autodiff_2d(m);
    bind_autodiff_3d(m);
    bind_autodiff_4d(m);
    bind_autodiff_matrix_4d(m);
    bind_pcg32(m);

    m.def("cuda_eval",
          [](bool log_assembly) { enoki::cuda_eval(log_assembly); },
          py::arg("log_assembly") = false);

    m.def("cuda_set_log_level",
          [](int log_level) { enoki::cuda_set_log_level(log_level); },
          "Sets the current log level (0 == none, 1 == minimal, 2 == moderate, 3 == high, 4 == everything)");

    py::class_<CUDAManagedBuffer>(m, "CUDAManagedBuffer");
}
