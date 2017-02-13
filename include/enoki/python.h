/*
    enoki/python.h -- pybind11 type casters for static and dynamic arrays

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "array.h"
#include <pybind11/numpy.h>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename T, typename = void> struct array_shape_descr {
    static PYBIND11_DESCR name() { return _(""); }
    static PYBIND11_DESCR name_cont() { return _(""); }
};

template <typename T> struct array_shape_descr<T, std::enable_if_t<enoki::is_sarray<T>::value>> {
    static PYBIND11_DESCR name() {
        return array_shape_descr<typename T::Scalar>::name_cont() + _<T::Size>();
    }
    static PYBIND11_DESCR name_cont() {
        return array_shape_descr<typename T::Scalar>::name_cont() + _<T::Size>() + _(", ");
    }
};

template <typename T> struct array_shape_descr<T, std::enable_if_t<enoki::is_darray<T>::value>> {
    static PYBIND11_DESCR name() {
        return array_shape_descr<typename T::Scalar>::name_cont() + _("n");
    }
    static PYBIND11_DESCR name_cont() {
        return array_shape_descr<typename T::Scalar>::name_cont() + _("n, ");
    }
};

template<typename Type> struct type_caster<Type, std::enable_if_t<enoki::is_array<Type>::value>> {
    typedef typename Type::Scalar     Scalar;
    typedef typename Type::BaseScalar BaseScalar;

    bool load(handle src, bool) {
        auto arr = array_t<BaseScalar, array::f_style | array::forcecast>::ensure(src);
        if (!arr)
            return false;

        constexpr size_t ndim = enoki::array_depth<Type>::value;
        if (ndim != arr.ndim())
            return false;

        std::array<size_t, ndim> shape;
        std::reverse_copy(arr.shape(), arr.shape() + ndim, shape.begin());

        try {
            enoki::resize(value, shape);
        } catch (std::length_error) {
            return false;
        }

        const BaseScalar *buf = static_cast<const BaseScalar *>(arr.data());
        read_buffer(buf, value);

        return true;
    }

    static handle cast(const Type *src, return_value_policy policy, handle parent) {
        return cast(*src, policy, parent);
    }

    static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */) {
        if (enoki::ragged(src))
            throw type_error("Ragged arrays are not supported!");

        auto shape = enoki::shape(src);
        std::reverse(shape.begin(), shape.end());
        decltype(shape) stride;

        stride[0] = sizeof(BaseScalar);
        for (size_t i = 1; i < shape.size(); ++i)
            stride[i] = shape[i - 1] * stride[i - 1];

        buffer_info info(nullptr, sizeof(BaseScalar),
                         format_descriptor<BaseScalar>::value, shape.size(),
                         std::vector<size_t>(shape.begin(), shape.end()),
                         std::vector<size_t>(stride.begin(), stride.end()));

        array arr(info);
        BaseScalar *buf = static_cast<BaseScalar *>(arr.mutable_data());
        write_buffer(buf, src);
        return arr.release();
    }

    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

    static PYBIND11_DESCR name() {
        return pybind11::detail::type_descr(
            _("numpy.ndarray[dtype=") +
            npy_format_descriptor<BaseScalar>::name() + _(", shape=(") +
            array_shape_descr<Type>::name() + _(")]"));
    }

    operator Type*() { return &value; }
    operator Type&() { return value; }

private:
    template <typename T, std::enable_if_t<!enoki::is_array<T>::value, int> = 0>
    ENOKI_INLINE static void write_buffer(BaseScalar *&, const T &) { }

    template <typename T, std::enable_if_t<enoki::is_array<T>::value, int> = 0>
    ENOKI_INLINE static void write_buffer(BaseScalar *&buf, const T &value_) {
        const auto &value = value_.derived();
        size_t size = value.size();

        if (std::is_arithmetic<typename T::Scalar>::value) {
            memcpy(buf, &value.coeff(0), sizeof(typename T::Scalar) * size);
            buf += size;
        } else {
            for (size_t i = 0; i < size; ++i)
                write_buffer(buf, value.coeff(i));
        }
    }

    template <typename T, std::enable_if_t<!enoki::is_array<T>::value, int> = 0>
    ENOKI_INLINE static void read_buffer(const BaseScalar *&, T &) { }

    template <typename T, std::enable_if_t<enoki::is_array<T>::value, int> = 0>
    ENOKI_INLINE static void read_buffer(const BaseScalar *&buf, T &value_) {
        auto &value = value_.derived();
        size_t size = value.size();

        if (std::is_arithmetic<typename T::Scalar>::value) {
            memcpy(&value.coeff(0), buf, sizeof(typename T::Scalar) * size);
            buf += size;
        } else {
            for (size_t i = 0; i < size; ++i)
                read_buffer(buf, value.coeff(i));
        }
    }

private:
    Type value;
};


NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

NAMESPACE_BEGIN(enoki)

template <typename Func, typename Return, typename... Args /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
auto vectorize_wrapper_detail(Func &&f_, Return (*)(Args...) PYBIND11_NOEXCEPT_SPECIFIER) {
    return [f = std::forward<Func>(f_)](enoki::detail::vectorize_ref_t<Args>... args) {
        return vectorize_safe(f, args...);
    };
}

/// Construct a vectorize_wrapper from a vanilla function pointer
template <typename Return, typename... Args /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
auto vectorize_wrapper(Return (*f)(Args...) PYBIND11_NOEXCEPT_SPECIFIER) {
    return vectorize_wrapper_detail(f, f);
}

/// Construct a vectorize_wrapper from a lambda function (possibly with internal state)
template <typename Func> auto vectorize_wrapper(Func &&f) {
    return vectorize_wrapper_detail(
        std::forward<Func>(f),
        (typename pybind11::detail::remove_class<decltype(
             &std::remove_reference<Func>::type::operator())>::type *) nullptr);
}

/// Construct a vectorize_wrapper from a class method (non-const)
template <typename Return, typename Class, typename... Arg /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
auto vectorize_wrapper(Return (Class::*f)(Arg...) PYBIND11_NOEXCEPT_SPECIFIER) {
    return vectorize_wrapper_detail(
        [f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
        (Return(*)(Class *, Arg...) PYBIND11_NOEXCEPT_SPECIFIER) nullptr);
}

/// Construct a vectorize_wrapper from a class method (const)
template <typename Return, typename Class, typename... Arg /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
auto vectorize_wrapper(Return (Class::*f)(Arg...) const PYBIND11_NOEXCEPT_SPECIFIER) {
    return vectorize_wrapper_detail(
        [f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
        (Return(*)(const Class *, Arg...) PYBIND11_NOEXCEPT_SPECIFIER) nullptr);
}

NAMESPACE_END(enoki)
