#include <enoki/python.h>
#include <enoki/autodiff.h>
#include <enoki/dynamic.h>
#include <pybind11/stl.h>

using namespace enoki;

namespace py = pybind11;

namespace enoki {

template <typename Forward, typename Backward>
void pytorch_register_function(pybind11::module &m, const std::string &op_name,
                               const std::string &fn_name, Forward forward,
                               Backward backward) {
    namespace py = pybind11;

    py::object autograd = py::module::import("torch.autograd");
    py::object parent_class = autograd.attr("Function");
    py::object parent_metaclass =
        autograd.attr("function").attr("FunctionMeta");
    py::object staticmethod =
        py::reinterpret_borrow<py::object>((PyObject *) &PyStaticMethod_Type);
    py::dict attributes;

    attributes["forward"] = staticmethod(py::cpp_function(forward));
    attributes["backward"] = staticmethod(py::cpp_function(backward));

    auto cls = parent_metaclass(op_name, py::make_tuple(parent_class), attributes);

    m.add_object(op_name.c_str(), cls);
    m.add_object(fn_name.c_str(), cls.attr("apply"));
}

namespace detail {
    template <typename T> py::object torch_dtype() {
        auto torch = py::module::import("torch");
        const char *name = nullptr;

        if (std::is_same_v<T, half>) {
            name = "float16";
        } else if (std::is_same_v<T, float>) {
            name = "float32";
        } else if (std::is_same_v<T, double>) {
            name = "float64";
        } else if (std::is_integral_v<T>) {
            if (sizeof(T) == 1)
                name = std::is_signed_v<T> ? "int8" : "uint8";
            else if (sizeof(T) == 2)
                name = "int16";
            else if (sizeof(T) == 4)
                name = "int32";
            else if (sizeof(T) == 8)
                name = "int64";
        }

        if (name == nullptr)
            throw std::runtime_error("pytorch_dtype(): Unsupported type");

        return torch.attr(name);
    }
}


template <size_t Index, size_t Dim, typename Source, typename Target>
void copy_array(const std::array<size_t, Dim> &shape,
                const std::array<size_t, Dim> &strides,
                const Source *source,
                Target &target) {
    if constexpr (Index == Dim) {
        target = *source;
    } else {
        const size_t step = strides[Index];
        for (size_t i = 0; i < shape[Index]; ++i) {
            copy_array<Index + 1, Dim>(shape, strides, source, target.coeff(i));
            source += step;
        }
    }
}

template <size_t Index, size_t Dim, typename Source, typename Target>
void copy_array(const std::array<size_t, Dim> &shape,
                const std::array<size_t, Dim> &strides,
                const Source &source,
                Target *target) {
    if constexpr (Index == Dim) {
        *target = source;
    } else {
        const size_t step = strides[Index];
        for (size_t i = 0; i < shape[Index]; ++i) {
            copy_array<Index + 1, Dim>(shape, strides, source.coeff(i), target);
            target += step;
        }
    }
}

template <typename T> T torch_to_enoki(py::object x) {
    using Scalar = scalar_t<T>;
    constexpr size_t Depth = array_depth_v<T>;

    py::tuple shape_obj = x.attr("shape");
    py::object dtype_obj = x.attr("dtype");
    py::object target_dtype = detail::torch_dtype<Scalar>();

    if (shape_obj.size() != Depth)
        throw std::runtime_error("torch_to_enoki(): Input array is of invalid dimension!");

    if (!dtype_obj.is(target_dtype))
        throw std::runtime_error("torch_to_enoki(): Input array has an invalid dtype!");

    auto shape = py::cast<std::array<size_t, Depth>>(shape_obj);
    auto strides = py::cast<std::array<size_t, Depth>>(x.attr("stride")());
    std::reverse(shape.begin(), shape.end());
    std::reverse(strides.begin(), strides.end());

    T result;
    set_shape(result, shape);

    const Scalar *source = (const Scalar *) py::cast<uintptr_t>(x.attr("data_ptr")());
    copy_array<0>(shape, strides, source, result);
    return result;
}

template <typename T> py::object enoki_to_torch(const T &t) {
    using Scalar = scalar_t<T>;
    constexpr size_t Depth = array_depth_v<T>;

    std::array<size_t, Depth> shape = enoki::shape(t), shape_rev = shape, strides;
    std::reverse(shape_rev.begin(), shape_rev.end());

    auto torch = py::module::import("torch");
    py::object dtype = detail::torch_dtype<scalar_t<T>>();
    auto result = torch.attr("empty")(py::cast(shape_rev), py::arg("dtype") = dtype);
    strides = py::cast<std::array<size_t, Depth>>(result.attr("stride")());
    std::reverse(strides.begin(), strides.end());
    Scalar *target = (Scalar *) py::cast<uintptr_t>(result.attr("data_ptr")());

    copy_array<0>(shape, strides, t, target);
    return result;
}
}

using Float  = float;
using FloatX = DynamicArray<Packet<Float>>;
using Vector3fX = Array<FloatX, 3>;


PYBIND11_MODULE(torch_test, m) {
    pytorch_register_function(
        m,
        "Square",
        "square",

        [](py::object ctx, py::object x_) {
            Vector3fX x = torch_to_enoki<Vector3fX>(x_);
            enoki::requires_gradient(x);
            return enoki_to_torch(x * x);
        },

        [](py::object ctx, py::object grad_output) {
            //py::object y = ctx.attr("saved_tensors")[py::int_(0)];
            //return grad_output * y * py::int_(2);
        }
    );
}

