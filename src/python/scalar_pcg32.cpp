#include <enoki/random.h>
#include "common.h"

void bind_scalar_pcg32(py::module& m) {
    using PCG32 = PCG32<Float>;

    py::class_<PCG32>(m, "PCG32")
        .def(py::init<UInt64, UInt64>(), "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("seed", &PCG32::seed, "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("next_uint32", [](PCG32 &pcg) { return pcg.next_uint32(); })
        .def("next_uint32",
             [](PCG32 &pcg, bool mask) {
                 return pcg.next_uint32(mask);
             },
             "mask"_a)
        .def("next_uint64", [](PCG32 &pcg) { return pcg.next_uint64(); })
        .def("next_uint64",
             [](PCG32 &pcg, bool mask) {
                 return pcg.next_uint64(mask);
             },
             "mask"_a)
        .def("next_uint32_bounded",
             [](PCG32 &pcg, uint64_t bound) {
                 return pcg.next_uint32_bounded(bound);
             },
             "bound"_a)
        .def("next_uint32_bounded",
             [](PCG32 &pcg, uint64_t bound, bool mask) {
                 return pcg.next_uint32_bounded(bound, mask);
             },
             "bound"_a, "mask"_a)
        .def("next_uint64_bounded",
             [](PCG32 &pcg, uint64_t bound) {
                 return pcg.next_uint64_bounded(bound);
             },
             "bound"_a)
        .def("next_uint64_bounded",
             [](PCG32 &pcg, uint64_t bound, bool mask) {
                 return pcg.next_uint64_bounded(bound, mask);
             },
             "bound"_a, "mask"_a)
        .def("next_float32", [](PCG32 &pcg) { return pcg.next_float32(); })
        .def("next_float32",
             [](PCG32 &pcg, bool mask) {
                 return pcg.next_float32(mask);
             },
             "mask"_a)
        .def_readwrite("state", &PCG32::state)
        .def_readwrite("inc", &PCG32::inc);
}
