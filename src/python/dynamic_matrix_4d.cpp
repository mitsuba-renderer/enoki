#include "common.h"

void bind_dynamic_matrix_4d(py::module& m) {
    bind_matrix<Matrix4fX>(m, "Matrix4fX");
}
