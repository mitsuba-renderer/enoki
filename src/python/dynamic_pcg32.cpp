#include <enoki/random.h>
#include "common.h"

void bind_dynamic_pcg32(py::module& m) {
    using PCG32X = PCG32<FloatX, 1>;

    py::class_<PCG32X>(m, "PCG32X")
        .def(py::init<UInt64X, UInt64X>(), "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("seed", &PCG32X::seed, "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("next_uint32", [](PCG32X &pcg) { return pcg.next_uint32(); })
        .def("next_uint32",
             [](PCG32X &pcg, const MaskX &mask) {
                 return pcg.next_uint32(mask);
             },
             "mask"_a)
        .def("next_uint64", [](PCG32X &pcg) { return pcg.next_uint64(); })
        .def("next_uint64",
             [](PCG32X &pcg, const MaskX &mask) {
                 return pcg.next_uint64(mask);
             },
             "mask"_a)
        .def("next_uint32_bounded",
             [](PCG32X &pcg, uint64_t bound) {
                 return pcg.next_uint32_bounded(bound);
             },
             "bound"_a)
        .def("next_uint32_bounded",
             [](PCG32X &pcg, uint64_t bound, const MaskX &mask) {
                 return pcg.next_uint32_bounded(bound, mask);
             },
             "bound"_a, "mask"_a)
        .def("next_uint64_bounded",
             [](PCG32X &pcg, uint64_t bound) {
                 return pcg.next_uint64_bounded(bound);
             },
             "bound"_a)
        .def("next_uint64_bounded",
             [](PCG32X &pcg, uint64_t bound, const MaskX &mask) {
                 return pcg.next_uint64_bounded(bound, mask);
             },
             "bound"_a, "mask"_a)
        .def("next_float32", [](PCG32X &pcg) { return pcg.next_float32(); })
        .def("next_float32",
             [](PCG32X &pcg, const MaskX &mask) {
                 return pcg.next_float32(mask);
             },
             "mask"_a)
        .def_readwrite("state", &PCG32X::state)
        .def_readwrite("inc", &PCG32X::inc);
}
