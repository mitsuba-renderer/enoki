#include "matrix.h"

void bind_scalar_matrix(py::module& m, py::module& s) {
    bind_matrix_mask<Matrix2m>(m, s, "Matrix2m");
    bind_matrix_mask<Matrix3m>(m, s, "Matrix3m");
    bind_matrix_mask<Matrix4m>(m, s, "Matrix4m");

    bind_matrix<Matrix2f>(m, s, "Matrix2f");
    bind_matrix<Matrix3f>(m, s, "Matrix3f");
    bind_matrix<Matrix4f>(m, s, "Matrix4f");

    bind_matrix<Matrix2d>(m, s, "Matrix2d");
    bind_matrix<Matrix3d>(m, s, "Matrix3d");
    bind_matrix<Matrix4d>(m, s, "Matrix4d");
}
