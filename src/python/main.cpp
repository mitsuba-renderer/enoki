#include "common.h"

extern void bind_scalar_1d(py::module&);
extern void bind_scalar_2d(py::module&);
extern void bind_scalar_3d(py::module&);
extern void bind_scalar_4d(py::module&);
extern void bind_scalar_complex(py::module&);
extern void bind_scalar_matrix(py::module&);
extern void bind_scalar_pcg32(py::module&);

extern void bind_dynamic_1d(py::module&);
extern void bind_dynamic_2d(py::module&);
extern void bind_dynamic_3d(py::module&);
extern void bind_dynamic_4d(py::module&);
extern void bind_dynamic_complex(py::module&);
extern void bind_dynamic_matrix(py::module&);
extern void bind_dynamic_pcg32(py::module&);

extern void bind_cuda_1d(py::module&);
extern void bind_cuda_2d(py::module&);
extern void bind_cuda_3d(py::module&);
extern void bind_cuda_4d(py::module&);
extern void bind_cuda_complex(py::module&);
extern void bind_cuda_matrix(py::module&);
extern void bind_cuda_pcg32(py::module&);

extern void bind_cuda_autodiff_1d(py::module&);
extern void bind_cuda_autodiff_2d(py::module&);
extern void bind_cuda_autodiff_3d(py::module&);
extern void bind_cuda_autodiff_4d(py::module&);
extern void bind_cuda_autodiff_complex(py::module&);
extern void bind_cuda_autodiff_matrix(py::module&);

bool disable_print_flag = false; // used in common.h

PYBIND11_MODULE(enoki, m) {
    m.attr("__version__") = ENOKI_VERSION;

    m.attr("Mask")   = py::handle((PyObject *) &PyBool_Type);
    m.attr("Float")  = py::handle((PyObject *) &PyFloat_Type);
    m.attr("Int32")  = py::handle((PyObject *) &PyLong_Type);
    m.attr("UInt32") = py::handle((PyObject *) &PyLong_Type);
    m.attr("Int64")  = py::handle((PyObject *) &PyLong_Type);
    m.attr("UInt64") = py::handle((PyObject *) &PyLong_Type);

    bind_scalar_1d(m);
    bind_scalar_2d(m);
    bind_scalar_3d(m);
    bind_scalar_4d(m);
    bind_scalar_complex(m);
    bind_scalar_matrix(m);
    bind_scalar_pcg32(m);

    bind_dynamic_1d(m);
    bind_dynamic_2d(m);
    bind_dynamic_3d(m);
    bind_dynamic_4d(m);
    bind_dynamic_complex(m);
    bind_dynamic_matrix(m);
    bind_dynamic_pcg32(m);

#if defined(ENOKI_CUDA)
    cuda_sync();

    bind_cuda_1d(m);
    bind_cuda_2d(m);
    bind_cuda_3d(m);
    bind_cuda_4d(m);
    bind_cuda_complex(m);
    bind_cuda_matrix(m);
    bind_cuda_pcg32(m);

#if defined(ENOKI_AUTODIFF)
    bind_cuda_autodiff_1d(m);
    bind_cuda_autodiff_2d(m);
    bind_cuda_autodiff_3d(m);
    bind_cuda_autodiff_4d(m);
    bind_cuda_autodiff_complex(m);
    bind_cuda_autodiff_matrix(m);
#endif

    m.def("cuda_eval", &cuda_eval, "log_assembly"_a = false,
          py::call_guard<py::gil_scoped_release>());

    m.def("cuda_sync", &cuda_sync,
          py::call_guard<py::gil_scoped_release>());

    m.def("cuda_malloc_trim", &cuda_malloc_trim);

    m.def("cuda_whos", []() { char *w = cuda_whos(); py::print(w); free(w); });

    m.def("cuda_mem_get_info", []() {
        size_t free = 0, total = 0;
        cuda_mem_get_info(&free, &total);
        return std::make_pair(free, total);
    });

    m.def("cuda_set_log_level", &cuda_set_log_level,
          "Sets the current log level (0: none, 1: kernel launches, 2: +ptxas "
          "statistics, 3: +ptx source, 4: +jit trace, 5: +ref counting)");

    m.def("cuda_log_level", &cuda_log_level);
#endif

    py::class_<Buffer>(m, "Buffer");

    m.def("set_requires_gradient",
          [](py::object o, bool value) {
              throw py::type_error("set_requires_gradient(): requires a differentiable type as input!");
          }, "array"_a, "value"_a = true);

    m.def("zero",
        [](py::handle h, size_t size) {
            if (size == 1)
                return h(0);
            else
                return h.attr("zero")(size);
        },
        "type"_a, "size"_a = 1);

    m.def("empty",
        [](py::handle h, size_t size) {
            if (size == 1)
                return h();
            else
                return h.attr("empty")(size);
        },
        "type"_a, "size"_a = 1);
}
