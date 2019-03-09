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

bool disable_print_flag = false;

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

    m.def("cuda_eval", &cuda_eval, "log_assembly"_a = false,
          py::call_guard<py::gil_scoped_release>());

    m.def("cuda_sync", &cuda_sync,
          py::call_guard<py::gil_scoped_release>());

    m.def("cuda_malloc_trim", &cuda_malloc_trim);

    m.def("cuda_whos", []() { char *w = cuda_whos(); py::print(w); free(w); });

    m.def("cuda_set_log_level", &cuda_set_log_level,
          "Sets the current log level (0: none, 1: kernel launches, 2: +ptxas "
          "statistics, 3: +ptx source, 4: +jit trace, 5: +ref counting)");

    m.def("cuda_log_level", &cuda_log_level);

    py::class_<CUDAManagedBuffer>(m, "CUDAManagedBuffer");
}
