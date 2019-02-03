#include <enoki/pytorch.h>
#include <enoki/autodiff.h>

using namespace enoki;

namespace py = pybind11;

using Float     = float;
using FloatC    = CUDAArray<Float>;
using FloatD    = DiffArray<FloatC>;
using Vector3fC = Array<FloatC, 3>;
using Vector3fD = Array<FloatD, 3>;


PYBIND11_MODULE(torch_test, m) {
    pytorch_register_function(
        m,
        "Normalize",
        "normalize",

        [](py::object ctx, const Vector3fD &in) {
            // Perform the (differentiable) operation using Enoki
            Vector3fD out = normalize(in);

            // Record the indices of input and output gradients for the reverse impl.
            ctx.attr("in_indices")  = gradient_index(in);
            ctx.attr("out_indices") = gradient_index(out);

            // Debug output of the computation graph
            std::cout << graphviz(out) << std::endl;

            // In the case of multiple outputs, return std::make_tuple(...)
            return out;
        },

        [](py::object ctx, const Vector3fC &grad_output_1/* Potentially more output parameters */) {
            // Look up stored context fields
            auto in_indices = gradient_index<Vector3fD>(ctx.attr("in_indices"));
            auto out_indices = gradient_index<Vector3fD>(ctx.attr("out_indices"));

            // Forward output gradients from PyTorch to Enoki
            out_indices.set_gradient(grad_output_1);

            // Propagate derivatives through the tape recorded by Enoki
            backward<FloatD>();

            return in_indices.gradient();
        }
    );
}

