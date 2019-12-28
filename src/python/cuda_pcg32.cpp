#include "random.h"

void bind_cuda_pcg32(py::module& m) {
    bind_pcg32<PCG32<FloatC, 1>>(m, "PCG32C");
}
