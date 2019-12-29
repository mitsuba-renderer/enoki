#include "common.h"

void bind_cuda_autodiff_matrix(py::module& m, py::module& s) {
    bind_matrix<Matrix2fD>(m, s, "Matrix2f");
    bind_matrix<Matrix3fD>(m, s, "Matrix3f");
    bind_matrix<Matrix4fD>(m, s, "Matrix4f");
    bind_matrix<Matrix44fD>(m, s, "Matrix44f");

    bind_matrix<Matrix2dD>(m, s, "Matrix2d");
    bind_matrix<Matrix3dD>(m, s, "Matrix3d");
    bind_matrix<Matrix4dD>(m, s, "Matrix4d");
    bind_matrix<Matrix44dD>(m, s, "Matrix44d");
}
