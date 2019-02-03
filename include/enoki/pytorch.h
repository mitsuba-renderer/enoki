/*
    enoki/pytorch.h -- Integration between PyTorch and Enoki's autodiff backend

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyrighe (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/python.h>
#include <enoki/cuda.h>
#include <enoki/autodiff.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename Value>
struct type_caster<Value, std::enable_if_t<enoki::is_cuda_array_v<Value> &&
                                          !enoki::is_diff_array_v<Value>>> {
    using Scalar = enoki::scalar_t<Value>;
    static constexpr size_t Depth = enoki::array_depth_v<Value>;

    bool load(handle src, bool) {
        using namespace enoki;

        if (src.is_none()) {
            is_none = true;
            return true;
        }

        tuple shape_obj = src.attr("shape");
        object dtype_obj = src.attr("dtype");
        object target_dtype = torch_dtype();

        if (shape_obj.size() != Depth)
            throw std::runtime_error(
                "torch_to_enoki(): Input array is of invalid dimension!");

        if (!dtype_obj.is(target_dtype))
            throw std::runtime_error(
                "torch_to_enoki(): Input array has an invalid dtype!");

        auto shape = pybind11::cast<std::array<size_t, Depth>>(shape_obj);
        auto strides = pybind11::cast<std::array<size_t, Depth>>(src.attr("stride")());
        std::reverse(shape.begin(), shape.end());
        std::reverse(strides.begin(), strides.end());

        size_t size = 1;
        for (size_t i : shape)
            size *= i;

        CUDAArray<Scalar> source = CUDAArray<Scalar>::map(
            (Scalar *) pybind11::cast<uintptr_t>(src.attr("data_ptr")()), size);

        copy_array_gather<0>(0, shape, strides, source, value);
        return true;
    }

    static handle cast(const Value *src, return_value_policy policy, handle parent) {
        if (!src)
            return pybind11::none();
        return cast(*src, policy, parent);
    }

    static handle cast(const Value &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace enoki;

        std::array<size_t, Depth> shape = enoki::shape(src),
                                  shape_rev = shape,
                                  strides;
        std::reverse(shape_rev.begin(), shape_rev.end());

        object torch = module::import("torch");
        object dtype_obj = torch_dtype();

        object result =
            torch.attr("empty")(pybind11::cast(shape_rev), arg("dtype") = dtype_obj,
                                arg("device") = "cuda");
        size_t size = 1;
        for (size_t i : shape)
            size *= i;

        strides = pybind11::cast<std::array<size_t, Depth>>(result.attr("stride")());
        std::reverse(strides.begin(), strides.end());
        CUDAArray<Scalar> target = CUDAArray<Scalar>::map(
            (Scalar *) pybind11::cast<uintptr_t>(result.attr("data_ptr")()), size);
        copy_array_scatter<0>(0, shape, strides, src, target);
        return result.inc_ref();
    }

    template <typename T_> using cast_op_type = pybind11::detail::cast_op_type<T_>;

    static constexpr auto name =
            _("torch.Tensor[dtype=") +
            npy_format_descriptor<Scalar>::name + _(", shape=(") +
            array_shape_descr<Value>::name() + _(")]");

    operator Value*() { if (is_none) return nullptr; else return &value; }
    operator Value&() {
        #if !defined(NDEBUG)
            if (is_none)
                throw pybind11::cast_error("Cannot cast None or nullptr to an"
                                           " Enoki array.");
        #endif
        return value;
    }

private:
    static object torch_dtype() {
        object torch = module::import("torch");
        const char *name = nullptr;

        if (std::is_same_v<Scalar, enoki::half>) {
            name = "float16";
        } else if (std::is_same_v<Scalar, float>) {
            name = "float32";
        } else if (std::is_same_v<Scalar, double>) {
            name = "float64";
        } else if (std::is_integral_v<Scalar>) {
            if (sizeof(Scalar) == 1)
                name = std::is_signed_v<Scalar> ? "int8" : "uint8";
            else if (sizeof(Scalar) == 2)
                name = "int16";
            else if (sizeof(Scalar) == 4)
                name = "int32";
            else if (sizeof(Scalar) == 8)
                name = "int64";
        }

        if (name == nullptr)
            throw std::runtime_error("pytorch_dtype(): Unsupported type");

        return torch.attr(name);
    }

    template <size_t Index, size_t Dim, typename Source, typename Target>
    static void copy_array_gather(size_t offset,
                                  const std::array<size_t, Dim> &shape,
                                  const std::array<size_t, Dim> &strides,
                                  const Source &source, Target &target) {
        using namespace enoki;
        if constexpr (Index == Dim - 1) {
            using UInt32 = uint32_array_t<Source>;
            UInt32 index = fmadd(arange<UInt32>((uint32_t) shape[Index]),
                                 (uint32_t) strides[Index], (uint32_t) offset);
            target = gather<Target>(source, index);
        } else {
            const size_t step = strides[Index];
            for (size_t i = 0; i < shape[Index]; ++i) {
                copy_array_gather<Index + 1, Dim>(offset, shape, strides, source,
                                                  target.coeff(i));
                offset += step;
            }
        }
    }

    template <size_t Index, size_t Dim, typename Source, typename Target>
    static void copy_array_scatter(size_t offset,
                                   const std::array<size_t, Dim> &shape,
                                   const std::array<size_t, Dim> &strides,
                                   const Source &source, Target &target) {
        using namespace enoki;
        if constexpr (Index == Dim - 1) {
            using UInt32 = uint32_array_t<Source>;
            UInt32 index = fmadd(arange<UInt32>((uint32_t) shape[Index]),
                                 (uint32_t) strides[Index], (uint32_t) offset);
            scatter(target, source, index);
        } else {
            const size_t step = strides[Index];
            for (size_t i = 0; i < shape[Index]; ++i) {
                copy_array_scatter<Index + 1, Dim>(offset, shape, strides,
                                                   source.coeff(i), target);
                offset += step;
            }
        }
    }

    bool is_none = false;
    Value value;
};

template <typename Value>
struct type_caster<Value, std::enable_if_t<enoki::is_diff_array_v<Value>>> {
    using UnderlyingType = decltype(enoki::eval(enoki::detach(std::declval<Value>())));
    using Caster = type_caster<UnderlyingType>;

    static constexpr auto name = Caster::name;
    template <typename T_> using cast_op_type = pybind11::detail::cast_op_type<T_>;

    bool load(handle src, bool convert) {
        if (src.is_none()) {
            is_none = true;
            return true;
        }
        Caster caster;
        if (!caster.load(src, convert))
            return false;
        value = (UnderlyingType &) caster;

        if (pybind11::cast<bool>(src.attr("requires_grad")))
            enoki::requires_gradient(value);

        return true;
    }

    static handle cast(const Value *src, return_value_policy policy, handle parent) {
        if (!src)
            return pybind11::none();
        return cast(*src, policy, parent);
    }

    static handle cast(const Value &src, return_value_policy policy, handle parent) {
        return Caster::cast(enoki::detach(src), policy, parent);
    }

    operator Value*() { if (is_none) return nullptr; else return &value; }
    operator Value&() {
        #if !defined(NDEBUG)
            if (is_none)
                throw pybind11::cast_error("Cannot cast None or nullptr to an"
                                           " Enoki array.");
        #endif
        return value;
    }

    bool is_none;
    Value value;
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

NAMESPACE_BEGIN(enoki)

struct GradientIndexBase {
    virtual ~GradientIndexBase() = default;
};

template <typename T, typename = int>
struct GradientIndex;

template <typename T> struct GradientIndex<T, enable_if_static_array_t<T>> {
    GradientIndex() = default;
    GradientIndex(const T &value) {
        for (size_t i = 0; i < array_size_v<T>; ++i)
            nested[i] = GradientIndex<value_t<T>>(value.coeff(i));
    }
    GradientIndex(const GradientIndex &) = default;
    GradientIndex(GradientIndex &&) = default;
    GradientIndex& operator=(const GradientIndex &) = default;
    GradientIndex& operator=(GradientIndex &&) = default;

    auto gradient() {
        Array<decltype(nested[0].gradient()), array_size_v<T>> result;
        for (size_t i = 0; i < array_size_v<T>; ++i)
            result.coeff(i) = nested[i].gradient();
        return result;
    }

    template <typename T2>
    void set_gradient(const T2 &value) {
        for (size_t i = 0; i < array_size_v<T>; ++i)
            nested[i].set_gradient(value.coeff(i));
    }

    virtual ~GradientIndex() = default;
    std::array<GradientIndex<value_t<T>>, array_size_v<T>> nested;
};

template <typename T> struct GradientIndex<T, enable_if_dynamic_array_t<T>> {
    GradientIndex() = default;
    GradientIndex(const T &value) {
        m_index = value.index_();
        T::inc_ref_(m_index);
    }

    GradientIndex(const GradientIndex& g) : m_index(g.m_index) {
        T::inc_ref_(m_index);
    }

    GradientIndex(GradientIndex&& g) {
        std::swap(m_index, g.m_index);
    }

    GradientIndex &operator=(const GradientIndex &g) {
        T::dec_ref_(m_index);
        m_index = g.m_index;
        return *this;
    }

    GradientIndex &operator=(GradientIndex &&g) {
        std::swap(m_index, g.m_index);
        return *this;
    }

    virtual ~GradientIndex() {
        T::dec_ref_(m_index);
    }

    auto gradient() const {
        return T::gradient_static_(m_index);
    }

    template <typename T2>
    void set_gradient(const T2 &value) {
        T::set_gradient_static_(m_index, value);
    }

    uint32_t m_index = 0;
};

template <typename T, enable_if_array_t<T> = 0>
pybind11::object gradient_index(const T &value) {
    return pybind11::cast((GradientIndexBase *) new GradientIndex<T>(value),
                          pybind11::return_value_policy::take_ownership);
}

template <typename T> auto& gradient_index(const pybind11::handle handle) {
    return (GradientIndex<T> &) pybind11::cast<GradientIndexBase&>(handle);
}

template <typename Forward, typename Backward>
void pytorch_register_function(pybind11::module &m, const std::string &op_name,
                               const std::string &fn_name, Forward forward,
                               Backward backward) {
    namespace py = pybind11;

    if (!pybind11::detail::get_type_info(typeid(GradientIndexBase), false)) {
        py::class_<GradientIndexBase>(m, "GradientIndexBase");
    }

    py::object autograd = py::module::import("torch.autograd");
    py::object parent_class = autograd.attr("Function");
    py::object parent_metaclass =
        autograd.attr("function").attr("FunctionMeta");
    py::object staticmethod =
        py::reinterpret_borrow<py::object>((PyObject *) &PyStaticMethod_Type);
    py::dict attributes;

    attributes["cuda_eval"] = staticmethod(py::cpp_function([]{ cuda_eval(); }));
    attributes["forward_impl"] = staticmethod(py::cpp_function(forward));
    attributes["backward_impl"] = staticmethod(py::cpp_function(backward));
    py::eval<py::eval_statements>(R"(
@classmethod
def forward(cls, *args, **kwargs):
    result = cls.forward_impl(*args, **kwargs)
    print('*** CUDA flush')
    cls.cuda_eval()
    return result

@classmethod
def backward(cls, *args, **kwargs):
    result = cls.backward_impl(*args, **kwargs)
    cls.cuda_eval()
    return result
)", py::globals(), attributes);

    auto cls = parent_metaclass(op_name, py::make_tuple(parent_class), attributes);

    m.add_object(op_name.c_str(), cls);
    m.add_object(fn_name.c_str(), cls.attr("apply"));
}

NAMESPACE_END(enoki)
