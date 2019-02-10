#include "common.h"

void bind_cuda_matrix_4d(py::module& m) {
    bind_matrix<Matrix4fC>(m, "Matrix4fC");
}
