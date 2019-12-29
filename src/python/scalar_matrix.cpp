#include "common.h"

void bind_scalar_matrix(py::module& m, py::module& s) {
    bind_matrix<Matrix2f>(m, s, "Matrix2f");
    bind_matrix<Matrix3f>(m, s, "Matrix3f");
    bind_matrix<Matrix4f>(m, s, "Matrix4f");
    bind_matrix<Matrix44f>(m, s, "Matrix44f");

    bind_matrix<Matrix2d>(m, s, "Matrix2d");
    bind_matrix<Matrix3d>(m, s, "Matrix3d");
    bind_matrix<Matrix4d>(m, s, "Matrix4d");
    bind_matrix<Matrix44d>(m, s, "Matrix44d");
}
