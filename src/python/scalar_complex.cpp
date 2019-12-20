#include "common.h"

void bind_scalar_complex(py::module& m) {
    bind_complex<Complex2f>(m, "Complex2f");
    bind_complex<Complex24f>(m, "Complex24f");
}
