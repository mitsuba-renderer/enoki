#include "common.h"

void bind_scalar_matrix_4d(py::module& m) {
    bind_matrix<Matrix4f>(m, "Matrix4f");
}
