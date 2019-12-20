#include "common.h"

void bind_cuda_matrix(py::module& m) {
    bind_matrix<Matrix2fC>(m, "Matrix2fC");
    bind_matrix<Matrix3fC>(m, "Matrix3fC");
    bind_matrix<Matrix4fC>(m, "Matrix4fC");
}
