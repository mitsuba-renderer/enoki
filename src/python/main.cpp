#include "common.h"

extern void bind_cuda_1d(py::module&);
extern void bind_cuda_2d(py::module&);
extern void bind_cuda_3d(py::module&);
extern void bind_autodiff_1d(py::module&);
extern void bind_autodiff_2d(py::module&);
extern void bind_autodiff_3d(py::module&);

PYBIND11_MODULE(enoki, m) {
    bind_cuda_1d(m);
    bind_cuda_2d(m);
    bind_cuda_3d(m);
    bind_autodiff_1d(m);
    bind_autodiff_2d(m);
    bind_autodiff_3d(m);

    m.def("cuda_eval",
          [](bool log_assembly) { enoki::cuda_eval(log_assembly); },
          py::arg("log_assembly") = false);
}
