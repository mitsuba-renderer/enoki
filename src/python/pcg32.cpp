#include <enoki/random.h>
#include "common.h"

void bind_pcg32(py::module& m) {
    using PCG32C = PCG32<FloatC, 1>;

    py::class_<PCG32C>(m, "PCG32C")
        .def(py::init<UInt64C, UInt64C>(),
             "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("seed", &PCG32C::seed,
             "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("next_uint32",
             (UInt32C (PCG32C::*) (const BoolC &)) & PCG32C::next_uint32,
             "mask"_a = true)
        .def("next_uint32",
             (UInt32C (PCG32C::*) (const UInt32C &index, const BoolC &)) & PCG32C::next_uint32,
             "index"_a, "mask"_a = true)
        .def("next_uint32_bounded",
             (UInt32C (PCG32C::*) (uint32_t, const BoolC &)) & PCG32C::next_uint32_bounded,
             "bound"_a, "mask"_a = true)
        .def("next_uint64",
             (UInt64C (PCG32C::*) (const BoolC &)) & PCG32C::next_uint64,
             "mask"_a = true)
        .def("next_uint64",
             (UInt64C (PCG32C::*) (const UInt32C &index, const BoolC &)) & PCG32C::next_uint64,
             "index"_a, "mask"_a = true)
        .def("next_uint64_bounded",
             (UInt64C (PCG32C::*) (uint64_t, const BoolC &)) & PCG32C::next_uint64_bounded,
             "bound"_a, "mask"_a = true)
        .def("next_float32",
             (FloatC (PCG32C::*) (const BoolC &)) & PCG32C::next_float32,
             "mask"_a = true)
        .def("next_float32",
             (FloatC (PCG32C::*) (const UInt32C &index, const BoolC &)) & PCG32C::next_float32,
             "index"_a, "mask"_a = true)
        .def_readwrite("state", &PCG32C::state)
        .def_readwrite("inc", &PCG32C::inc);
}
