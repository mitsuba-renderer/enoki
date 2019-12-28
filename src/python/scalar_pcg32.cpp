#include "random.h"

void bind_scalar_pcg32(py::module& m) {
    bind_pcg32<PCG32<float, 1>>(m, "PCG32");
}
