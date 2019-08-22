/*
    enoki/python.h -- pybind11 support for Enoki types

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyrighe (c) 2019 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/complex.h>
#include <pybind11/numpy.h>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename T, typename = void> struct array_shape_descr {
    static constexpr auto name() { return _(""); }
    static constexpr auto name_cont() { return _(""); }
};

template <typename T>
struct array_shape_descr<T, std::enable_if_t<enoki::is_static_array_v<T>>> {
    static constexpr auto name() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _<T::Size>();
    }
    static constexpr auto name_cont() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _<T::Size>() + _(", ");
    }
};

template <typename T>
struct array_shape_descr<T, std::enable_if_t<enoki::is_dynamic_array_v<T>>> {
    static constexpr auto name() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _("n");
    }
    static constexpr auto name_cont() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _("n, ");
    }
};

template <typename Value>
struct type_caster<Value, std::enable_if_t<enoki::is_array_v<Value> &&
                                          !enoki::is_cuda_array_v<Value>>> {
    using Scalar = std::conditional_t<Value::IsMask, bool, enoki::scalar_t<Value>>;
    static constexpr bool IsComplex = Value::IsComplex;

    bool load(handle src, bool convert) {
        if (src.is_none()) {
            is_none = true;
            return true;
        }

        if (!isinstance<array_t<Scalar>>(src)) {
            if (!convert)
                return false;

            /// Don't cast enoki CUDA/autodiff types
            if (strncmp(((PyTypeObject *) src.get_type().ptr())->tp_name, "enoki.", 6) == 0)
                return false;
        }

        constexpr size_t ndim = enoki::array_depth_v<Value>;

        array arr = reinterpret_borrow<array>(src);
        if constexpr (IsComplex) {
            auto np = module::import("numpy");
            try {
                arr = np.attr("asarray")(arr, sizeof(Scalar) == 4 ? "c8" : "c16", "F");
                arr = np.attr("expand_dims")(arr, -1).attr("view")(
                    sizeof(Scalar) == 4 ? "f4" : "f8");
            } catch (const error_already_set &) {
                return false;
            }
        }

        arr = array_t<Scalar, array::f_style | array::forcecast>::ensure(arr);
        if (!arr)
            return false;

        if (ndim != arr.ndim() && !((arr.ndim() == 0 || (arr.ndim() == 1 && IsComplex)) && convert))
            return false;

        std::array<size_t, ndim> shape;
        std::fill(shape.begin(), shape.end(), (size_t) 1);
        std::reverse_copy(arr.shape(), arr.shape() + arr.ndim(), shape.begin());

        try {
            enoki::set_shape(value, shape);
        } catch (const std::length_error &) {
            return false;
        }

        const Scalar *buf = static_cast<const Scalar *>(arr.data());
        read_buffer(buf, value);

        return true;
    }

    static handle cast(const Value *src, return_value_policy policy, handle parent) {
        if (!src)
            return pybind11::none();
        return cast(*src, policy, parent);
    }

    static handle cast(const Value &src, return_value_policy /* policy */, handle /* parent */) {
        if (enoki::ragged(src))
            throw type_error("Ragged arrays are not supported!");

        auto shape = enoki::shape(src);
        std::reverse(shape.begin(), shape.end());
        decltype(shape) stride;

        stride[0] = sizeof(Scalar);
        for (size_t i = 1; i < shape.size(); ++i)
            stride[i] = shape[i - 1] * stride[i - 1];

        array arr(pybind11::dtype::of<Scalar>(),
                  std::vector<ssize_t>(shape.begin(), shape.end()),
                  std::vector<ssize_t>(stride.begin(), stride.end()));

        Scalar *buf = static_cast<Scalar *>(arr.mutable_data());
        write_buffer(buf, src);

        if constexpr (IsComplex) {
            auto np = module::import("numpy");
            arr = np.attr("ascontiguousarray")(arr).attr("view")(
                        sizeof(Scalar) == 4 ? "c8" : "c16").attr("squeeze")(-1);
        }

        return arr.release();
    }

    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

    static constexpr auto name_default =
            _("numpy.ndarray[dtype=") +
            npy_format_descriptor<Scalar>::name + _(", shape=(") +
            array_shape_descr<Value>::name() + _(")]");

    static constexpr auto name_complex =
            _("numpy.ndarray[dtype=Complex[") +
            npy_format_descriptor<Scalar>::name + _("], shape=(") +
            array_shape_descr<enoki::value_t<Value>>::name() + _(")]");

    static constexpr auto name = _<IsComplex>(name_complex, name_default);

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
    template <typename T> static ENOKI_INLINE void write_buffer(Scalar *&buf, const T &value) {
        if constexpr (!enoki::is_array_v<enoki::value_t<T>>) {
            if constexpr (!enoki::is_mask_v<T>) {
                memcpy(buf, value.data(), sizeof(enoki::value_t<T>) * value.size());
                buf += value.size();
            } else {
                for (size_t i = 0, size = value.size(); i < size; ++i)
                    *buf++ = value.coeff(i);
            }
        } else {
            for (size_t i = 0, size = value.size(); i < size; ++i)
                write_buffer(buf, value.coeff(i));
        }
    }

    template <typename T>
    static ENOKI_INLINE void read_buffer(const Scalar *&buf, T &value) {
        if constexpr (!enoki::is_array_v<enoki::value_t<T>>) {
            if constexpr (!enoki::is_mask_v<T>) {
                memcpy(value.data(), buf, sizeof(enoki::value_t<T>) * value.size());
                buf += value.size();
            } else {
                if constexpr (!enoki::is_dynamic_array_v<T>) {
                    enoki::Array<bool, T::Size> value2;
                    for (size_t i = 0, size = value2.size(); i < size; ++i)
                        value2.coeff(i) = *buf++;
                    value = enoki::reinterpret_array<T>(value2);
                } else {
                    const Scalar *end = buf + value.size();
                    for (size_t i = 0; i < enoki::packets(value); ++i) {
                        enoki::Array<bool, T::Packet::Size> value2;
                        for (size_t j = 0; j < T::Packet::Size && buf != end; ++j)
                            value2.coeff(j) = *buf++;
                        enoki::packet(value, i) = enoki::reinterpret_array<typename T::Packet>(value2);
                    }
                }
            }
        } else {
            for (size_t i = 0, size = value.size(); i < size; ++i)
                read_buffer(buf, value.coeff(i));
        }
    }

private:
    Value value;
    bool is_none = false;
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

NAMESPACE_BEGIN(enoki)

namespace detail {
    template <typename T, typename = void> struct reference_dynamic { using type = T; };
    template <typename T> struct reference_dynamic<T, std::enable_if_t<is_dynamic_v<T>>> {
        using type = std::add_lvalue_reference_t<T>;
    };
    template <typename T>
    using reference_dynamic_t = typename reference_dynamic<T>::type;
}

template <typename Func, typename Return, typename... Args>
auto vectorize_wrapper_detail(Func &&f_, Return (*)(Args...)) {
    return [f = std::forward<Func>(f_)](detail::reference_dynamic_t<enoki::make_dynamic_t<Args>>... args) {
        return vectorize_safe(f, args...);
    };
}

/// Vctorize a vanilla function pointer
template <typename Return, typename... Args>
auto vectorize_wrapper(Return (*f)(Args...)) {
    return vectorize_wrapper_detail(f, f);
}

/// Vectorize a lambda function method (possibly with internal state)
template <typename Func,
          typename FuncType = typename pybind11::detail::remove_class<
              decltype(&std::remove_reference<Func>::type::operator())>::type>
auto vectorize_wrapper(Func &&f) {
    return vectorize_wrapper_detail(std::forward<Func>(f), (FuncType *) nullptr);
}

/// Vectorize a class method (non-const)
template <typename Return, typename Class, typename... Arg>
auto vectorize_wrapper(Return (Class::*f)(Arg...)) {
    return vectorize_wrapper_detail(
        [f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
        (Return(*)(Class *, Arg...)) nullptr);
}

/// Vectorize a class method (const)
template <typename Return, typename Class, typename... Arg>
auto vectorize_wrapper(Return (Class::*f)(Arg...) const) {
    return vectorize_wrapper_detail(
        [f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
        (Return(*)(const Class *, Arg...)) nullptr);
}

NAMESPACE_END(enoki)
