#include "random.h"

void bind_dynamic_pcg32(py::module& m) {
    bind_pcg32<PCG32<FloatX, 1>>(m, "PCG32X");
}
