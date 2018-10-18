/*
    enoki/torch.h -- PyTorch backend

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array.h"
#include <pybind11/numpy.h>

NAMESPACE_BEGIN(enoki)

namespace py = pybind11;

namespace detail {
    struct internals {
        internals() = default;
        internals(const internals&) = delete;

        py::object module;

        // Tensor construction functions
        py::object tensor, empty, full, clone, zeros,
                   linspace, arange;

        // dtypes
        py::object uint8, int8, int16, int32, int64,
                   float16, float32, float64;

        // autograd
        py::object backward, grad;

        // arithmetic
        py::object addcmul, abs, sqrt, ceil, floor, round,
                   trunc, sin, cos, tan, asin, acos, atan,
                   atan2, sinh, cosh, tanh, exp, log, pow,
                   rsqrt, reciprocal;

        // miscellaneous
        py::object where, max, min, all, any;

        template <typename T>
        py::handle dtype() {
            if (std::is_same<T, half>::value)
                return float16;
            else if (std::is_same<T, float>::value)
                return float32;
            else if (std::is_same<T, double>::value)
                return float64;
            else if (sizeof(T) == 1)
                return std::is_signed<T>::value ? int8 : uint8;
            else if (sizeof(T) == 2)
                return int16;
            else if (sizeof(T) == 4)
                return int32;
            else if (sizeof(T) == 8)
                return int64;
            else
                return py::none();
        }

    };

    internals& torch() {
        static internals v;
        if (!v.module) {
            py::object m = py::module::import("torch");
            py::object ag = m.attr("autograd");

            v.module = m;

            // Tensor construction functions
            v.tensor = m.attr("tensor");
            v.empty = m.attr("empty");
            v.full = m.attr("full");
            v.clone = m.attr("clone");
            v.zeros = m.attr("zeros");
            v.linspace = m.attr("linspace");
            v.arange = m.attr("arange");

            // dtypes
            v.uint8 = m.attr("uint8");
            v.int8 = m.attr("int8");
            v.int16 = m.attr("int16");
            v.int32 = m.attr("int32");
            v.int64 = m.attr("int64");
            v.float16 = m.attr("float16");
            v.float32 = m.attr("float32");
            v.float64 = m.attr("float64");

            // autograd
            v.backward = ag.attr("backward");
            v.grad = ag.attr("grad");

            // arithmetic
            v.addcmul = m.attr("addcmul");
            v.abs = m.attr("abs");
            v.sqrt = m.attr("sqrt");
            v.ceil = m.attr("ceil");
            v.floor = m.attr("floor");
            v.trunc = m.attr("trunc");
            v.round = m.attr("round");
            v.sin = m.attr("sin");
            v.cos = m.attr("cos");
            v.tan = m.attr("tan");
            v.asin = m.attr("asin");
            v.acos = m.attr("acos");
            v.atan = m.attr("atan");
            v.atan2 = m.attr("atan2");
            v.sinh = m.attr("sinh");
            v.cosh = m.attr("cosh");
            v.tanh = m.attr("tanh");

            v.exp = m.attr("exp");
            v.log = m.attr("log");
            v.pow = m.attr("pow");
            v.rsqrt = m.attr("rsqrt");
            v.reciprocal = m.attr("reciprocal");

            // miscellaneous
            v.where = m.attr("where");
            v.max = m.attr("max");
            v.min = m.attr("min");
            v.all = m.attr("all");
            v.any = m.attr("any");
        }

        return v;
    }

    /* PyTorch currently does not support unsigned integer tensors (other than uint8_t) */
    template <typename T, typename SFINAE = void> struct torch_type { using type = T; };
    template <> struct torch_type<uint16_t> { using type = int16_t; };
    template <> struct torch_type<uint32_t> { using type = int32_t; };
    template <> struct torch_type<uint64_t> { using type = int64_t; };
    template <typename T>
    struct torch_type<T, std::enable_if_t<std::is_pointer<T>::value>> {
        using type = std::conditional_t<sizeof(void *) == 8, int64_t, int32_t>;
    };
}

template <typename Value_, size_t Size_>
struct TorchArray
    : StaticArrayBase<Value_, Size_, detail::is_std_float<Value_>::value,
                      RoundingMode::Default, TorchArray<Value_, Size_>> {
    using Base =
        StaticArrayBase<Value_, Size_, detail::is_std_float<Value_>::value,
                        RoundingMode::Default, TorchArray<Value_, Size_>>;
    using Base::operator[];
    using typename Base::Value;
    using TorchValue = typename detail::torch_type<Value>::type;

    template <typename T>
    using ReplaceType = TorchArray<T, Size_>;
    using MaskType = TorchArray<uint8_t, Size_>;
    using ScalarArray = TorchArray<Value_, 1>;

    static constexpr bool IsMask = std::is_same<Value_, uint8_t>::value;
    static constexpr bool IsTorch = true;

    explicit TorchArray(const py::object &o) : m_tensor(o) { }
    explicit TorchArray(py::object&& o) : m_tensor(std::move(o)) { }

    TorchArray() {
        auto &dt = detail::torch();
        m_tensor = dt.empty(Size_, py::arg("dtype") = dt.dtype<Value>());
    }

    TorchArray(Value value) {
        auto &dt = detail::torch();
        m_tensor = dt.full(py::make_tuple(Size_), TorchValue(value),
                           py::arg("dtype") = dt.dtype<Value>());
    }

    template <size_t S = Size_, std::enable_if_t<S != 1, int> = 0>
    TorchArray(const TorchArray<Value_, 1> &other) {
        m_tensor = other.handle().attr("repeat")(Size_);
    }

    TorchArray(const TorchArray<Value_, Size_> &other) {
        m_tensor = detail::torch().clone(other.handle());
    }

    template <typename Value2>
    TorchArray(const TorchArray<Value2, Size_> &other) {
        m_tensor = other.handle().attr("type")(detail::torch().dtype<Value>());
    }

    template <typename T, std::enable_if_t<!T::IsTorch, int> = 0>
    TorchArray(const T &other) {
        using Value2 = value_t<T>;
        py::array_t<Value2> array({ other.size() }, { sizeof(Value2) }, other.data(), py::none());
        auto &dt = detail::torch();
        m_tensor = dt.tensor(array, py::arg("dtype") = dt.dtype<Value>());
    }

    template <typename Array>
    ENOKI_INLINE TorchArray(const Array &other, detail::reinterpret_flag)  {
        py::object tensor = other.handle().attr("numpy")();
        tensor = tensor.attr("view")(py::dtype::of<Value>());

        auto &dt = detail::torch();
        m_tensor = dt.tensor(tensor, py::arg("dtype") = dt.dtype<Value>());
    }

    bool requires_grad_() const { return py::cast<bool>(m_tensor.attr("requires_grad")); }
    void set_requires_grad_(bool value) const { m_tensor.attr("requires_grad") = value; }
    void detach_() const { m_tensor.attr("detach")(); }
    TorchArray gradient_() const { return TorchArray(m_tensor.attr("grad")); }
    void zero_grad_() const {
        py::object grad = m_tensor.attr("grad");
        if (grad.is(py::none()))
            return;
        grad.attr("zero_")();
    }

    TorchArray neg_() const { return TorchArray(-m_tensor); }
    TorchArray not_() const {
        if (std::is_same<Value, uint8_t>::value) // ~ (operator.invert) is only implemented on byte tensors
            return TorchArray(~m_tensor);
        else
            return xor_(TorchArray(Value(-1)));
    }

    TorchArray add_(const TorchArray &o) const { return TorchArray(m_tensor + o.m_tensor); }
    TorchArray sub_(const TorchArray &o) const { return TorchArray(m_tensor - o.m_tensor); }
    TorchArray mul_(const TorchArray &o) const { return TorchArray(m_tensor * o.m_tensor); }
    TorchArray div_(const TorchArray &o) const { return TorchArray(m_tensor / o.m_tensor); }

    template <size_t Size>
    TorchArray sli_() const { return TorchArray(m_tensor << py::int_(Size)); }
    template <size_t Size>
    TorchArray sri_() const { return TorchArray(m_tensor >> py::int_(Size)); }
    TorchArray slv_(const TorchArray &o) const { return TorchArray(m_tensor << o.m_tensor); }
    TorchArray srv_(const TorchArray &o) const { return TorchArray(m_tensor >> o.m_tensor); }

    TorchArray fmadd_(const TorchArray &a, const TorchArray &b) const {
        return TorchArray(detail::torch().addcmul(b.m_tensor, 1, m_tensor, a.m_tensor));
    }

    TorchArray fnmadd_(const TorchArray &a, const TorchArray &b) const {
        return TorchArray(detail::torch().addcmul(b.m_tensor, -1, m_tensor, a.m_tensor));
    }

    TorchArray and_(const TorchArray &o) const { return TorchArray(m_tensor & o.m_tensor); }
    TorchArray or_(const TorchArray &o)  const { return TorchArray(m_tensor | o.m_tensor); }
    TorchArray xor_(const TorchArray &o) const { return TorchArray(m_tensor ^ o.m_tensor); }

    TorchArray max_(const TorchArray &o) const { return TorchArray(detail::torch().max(m_tensor, o.m_tensor)); }
    TorchArray min_(const TorchArray &o) const { return TorchArray(detail::torch().min(m_tensor, o.m_tensor)); }

    ScalarArray hsum_() const { return ScalarArray(m_tensor.attr("sum")()); }
    ScalarArray hprod_() const { return ScalarArray(m_tensor.attr("prod")()); }
    ScalarArray hmin_() const { return ScalarArray(m_tensor.attr("min")()); }
    ScalarArray hmax_() const { return ScalarArray(m_tensor.attr("max")()); }

    bool all_() const { return py::cast<bool>(detail::torch().all(m_tensor)); }
    bool any_() const { return py::cast<bool>(detail::torch().any(m_tensor)); }
    bool none_() const { return !py::cast<bool>(detail::torch().any(m_tensor)); }

    MaskType eq_ (const TorchArray &o) const { return rich_compare(o, Py_EQ); }
    MaskType neq_(const TorchArray &o) const { return rich_compare(o, Py_NE); }
    MaskType gt_ (const TorchArray &o) const { return rich_compare(o, Py_GT); }
    MaskType ge_ (const TorchArray &o) const { return rich_compare(o, Py_GE); }
    MaskType lt_ (const TorchArray &o) const { return rich_compare(o, Py_LT); }
    MaskType le_ (const TorchArray &o) const { return rich_compare(o, Py_LE); }

    static ENOKI_INLINE TorchArray select_(const MaskType &m, const TorchArray &t, const TorchArray &f) {
        return TorchArray(detail::torch().where(m.handle(), t.handle(), f.handle()));
    }

    TorchArray abs_() const { return TorchArray(detail::torch().abs(m_tensor)); }
    TorchArray sqrt_() const { return TorchArray(detail::torch().sqrt(m_tensor)); }
    TorchArray ceil_() const { return TorchArray(detail::torch().ceil(m_tensor)); }
    TorchArray floor_() const { return TorchArray(detail::torch().floor(m_tensor)); }
    TorchArray round_() const { return TorchArray(detail::torch().round(m_tensor)); }
    TorchArray trunc_() const { return TorchArray(detail::torch().trunc(m_tensor)); }
    TorchArray rsqrt_() const { return TorchArray(detail::torch().rsqrt(m_tensor)); }
    TorchArray rcp_() const { return TorchArray(detail::torch().reciprocal(m_tensor)); }

    TorchArray sin_() const { return TorchArray(detail::torch().sin(m_tensor)); }
    TorchArray cos_() const { return TorchArray(detail::torch().cos(m_tensor)); }
    std::pair<TorchArray, TorchArray> sincos_() const {
        return {
            TorchArray(detail::torch().sin(m_tensor)),
            TorchArray(detail::torch().cos(m_tensor))
        };
    }
    TorchArray tan_() const { return TorchArray(detail::torch().tan(m_tensor)); }
    TorchArray asin_() const { return TorchArray(detail::torch().asin(m_tensor)); }
    TorchArray acos_() const { return TorchArray(detail::torch().acos(m_tensor)); }
    TorchArray atan_() const { return TorchArray(detail::torch().atan(m_tensor)); }
    TorchArray atan2_(const TorchArray &other) const {
        return TorchArray(detail::torch().atan2(m_tensor, other.m_tensor));
    }

    TorchArray exp_() const { return TorchArray(detail::torch().exp(m_tensor)); }
    TorchArray log_() const { return TorchArray(detail::torch().log(m_tensor)); }
    TorchArray pow_(const TorchArray &other) const { return TorchArray(detail::torch().pow(m_tensor, other.m_tensor)); }

    Value *data() const {
        using Int = std::conditional_t<sizeof(void *) == 8, int64_t, int32_t>;
        return (Value *) py::cast<Int>(m_tensor.attr("cpu")().attr("data_ptr")());
    }

    const Value coeff(size_t index) const { return (Value) py::cast<TorchValue>(m_tensor[py::int_(index)]); }
    Value &coeff(size_t) {
        throw std::runtime_error("TorchArray::operator[]: non-const indexing is not allowed!");
    }

    static TorchArray zero_() {
        auto &dt = detail::torch();
        return TorchArray(dt.zeros(Size_, py::arg("dtype") = dt.dtype<Value>()));
    }
    static TorchArray linspace_(Value a, Value b) {
        auto &dt = detail::torch();
        return TorchArray(
            dt.linspace(a, b, Size_, py::arg("dtype") = dt.dtype<Value>()));
    }
    static TorchArray index_sequence_() {
        auto &dt = detail::torch();
        return TorchArray(
            dt.arange(Size_, py::arg("dtype") = dt.dtype<Value>()));
    }
    template <typename Mask>
    ENOKI_INLINE auto extract_(const Mask &mask) const {
        return (Value) py::cast<TorchValue>(m_tensor[mask.handle()][py::int_(0)]);
    }

    template <typename T = Value_, std::enable_if_t<std::is_pointer<T>::value, int> = 0>
    call_support<TorchArray, TorchArray> operator->() const {
        return call_support<TorchArray, TorchArray>(*this);
    }

    py::object tensor() const { return m_tensor; }
    py::handle handle() const { return m_tensor; }

protected:

    MaskType rich_compare(const TorchArray &other, int op) const {
        PyObject *out = PyObject_RichCompare(m_tensor.ptr(), other.m_tensor.ptr(), op);
        if (!out)
            throw py::error_already_set();
        return MaskType(py::reinterpret_steal<py::object>(out));
    }

protected:
    py::object m_tensor;
};


template <typename... Args> void autograd_backward(const Args&... args) {
    detail::torch().backward(py::make_tuple(args.handle()...));
}

template <typename... Args> py::object autograd_forward(const Args&... args) {
    py::tuple py_args = py::make_tuple(args.handle()...);
    size_t size = sizeof...(Args);
    if (size % 2 == 1)
        throw std::runtime_error("autograd_forward(): must have an even number of arguments!");
    return detail::torch().grad(
        py_args[py::slice(0, size / 2, 1)],
        py_args[py::slice(size / 2, size, 1)]
    );
}

template <typename T, std::enable_if_t<T::IsTorch == 0, int> = 0>
void set_requires_grad(T& array, bool value = true) {
    for (size_t i = 0; i < array.size(); ++i)
        set_requires_grad(array.coeff(i), value);
}


template <typename Value, size_t Size>
void set_requires_grad(TorchArray<Value, Size>& array, bool value = true) {
    array.set_requires_grad_(value);
}

template <typename T, std::enable_if_t<T::IsTorch == 0, int> = 0>
void zero_grad(T& array) {
    for (size_t i = 0; i < array.size(); ++i)
        zero_grad(array.coeff(i));
}

template <typename... Args, std::enable_if_t<(sizeof...(Args) > 1), int> = 0>
void zero_grad(Args&... array) {
    bool unused[] = {(zero_grad(array), false)...};
    (void) unused;
}


template <typename Value, size_t Size>
void zero_grad(TorchArray<Value, Size>& array) {
    array.zero_grad_();
}

template <typename T, std::enable_if_t<T::IsTorch == 0, int> = 0>
T gradient(const T& array) {
    T result;
    for (size_t i = 0; i < array.size(); ++i)
        result.coeff(i) = gradient(array.coeff(i));
    return result;
}


template <typename Value, size_t Size>
TorchArray<Value, Size> gradient(const TorchArray<Value, Size>& array) {
    return array.gradient_();
}

NAMESPACE_END(enoki)
