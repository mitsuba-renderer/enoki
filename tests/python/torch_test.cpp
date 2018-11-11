#include <enoki/pytorch.h>
#include <enoki/autodiff.h>
#include <enoki/dynamic.h>

using namespace enoki;

namespace py = pybind11;

using Float  = float;
using FloatX = DynamicArray<Packet<Float>>;
using FloatD = DiffArray<FloatX>;
using Vector3fX = Array<FloatX, 3>;
using Vector3fD = Array<FloatD, 3>;

PYBIND11_MODULE(torch_test, m) {
    using IndexIn = std::array<uint32_t, 3>;
    using IndexOut = std::array<uint32_t, 3>;

    m.def("clear_graph", [](){ clear_graph<FloatD>(); });

    pytorch_register_function(
        m,
        "Normalize",
        "normalize",

        [](py::object ctx, py::object in_ /* Potentially more input parameters */) {
            // PyTorch tensor -> Enoki array
            Vector3fD in = torch_to_enoki<Vector3fD>(in_);

            if (py::cast<bool>(in_.attr("requires_grad")))
                requires_gradient(in, "in");

            // Perform the (differentiable) operation using Enoki
            Vector3fD out = normalize(in);

            // Record the output indices for the reverse impl.
            IndexIn  in_indices  = gradient_index(in);
            IndexOut out_indices = gradient_index(out);

            ctx.attr("in_indices")  = in_indices;
            ctx.attr("out_indices") = out_indices;
            ctx.attr("tape_ptr") = FloatD::get_tape_ptr();

            // Increase the reference count keep the associated tape nodes from being collected
            gradient_inc_ref<FloatD>(in_indices);
            gradient_inc_ref<FloatD>(out_indices);

            // Debug output of the computation graph
            std::cout << graphviz(out) << std::endl;

            // In the case of multiple outputs, return std::make_tuple(...)
            return enoki_to_torch(out);
        },

        [](py::object ctx, py::object grad_output_1 /* Potentially more output parameters */) {
            // Look up stored context fields
            IndexIn  in_indices  = py::cast<IndexIn> (ctx.attr("in_indices"));
            IndexOut out_indices = py::cast<IndexOut>(ctx.attr("out_indices"));

            // PyTorch runs the backward pass on a different thread. The following ensures
            // that Enoki accesses the right tape data structure
            TapeScope<FloatD> scope(py::cast<void *>(ctx.attr("tape_ptr")));

            // Clear all gradients
            clear_gradients<FloatD>();

            // Forward output gradients from PyTorch to Enoki
            set_gradient(out_indices, torch_to_enoki<Vector3fD>(grad_output_1));

            // Propagate derivatives through the tape recorded by Enoki
            backward<FloatD>();

            Vector3fD result = gradient<Vector3fD>(in_indices);

            // Undo the previous reference count change
            gradient_dec_ref<FloatD>(out_indices);
            gradient_dec_ref<FloatD>(in_indices);

            return enoki_to_torch(result);
        }
    );
}

