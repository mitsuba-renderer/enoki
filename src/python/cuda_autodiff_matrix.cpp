#include "common.h"

void bind_cuda_autodiff_matrix(py::module& m) {
    bind_matrix<Matrix2fD>(m, "Matrix2fD");
    bind_matrix<Matrix3fD>(m, "Matrix3fD");
    bind_matrix<Matrix4fD>(m, "Matrix4fD");
}
