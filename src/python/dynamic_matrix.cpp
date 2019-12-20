#include "common.h"

void bind_dynamic_matrix(py::module& m) {
    bind_matrix<Matrix2fX>(m, "Matrix2fX");
    bind_matrix<Matrix3fX>(m, "Matrix3fX");
    bind_matrix<Matrix4fX>(m, "Matrix4fX");
    bind_matrix<Matrix44fX>(m, "Matrix44fX");
}
