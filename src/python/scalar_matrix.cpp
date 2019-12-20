#include "common.h"

void bind_scalar_matrix(py::module& m) {
    bind_matrix<Matrix2f>(m, "Matrix2f");
    bind_matrix<Matrix3f>(m, "Matrix3f");
    bind_matrix<Matrix4f>(m, "Matrix4f");
}
