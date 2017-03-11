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

template <typename T> struct array_shape_descr<T, std::enable_if_t<enoki::is_static_array<T>::value>> {
    static PYBIND11_DESCR name() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _<T::Size>();
    }
    static PYBIND11_DESCR name_cont() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _<T::Size>() + _(", ");
    }
};

template <typename T> struct array_shape_descr<T, std::enable_if_t<enoki::is_dynamic_array<T>::value>> {
    static PYBIND11_DESCR name() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _("n");
    }
    static PYBIND11_DESCR name_cont() {
        return array_shape_descr<enoki::value_t<T>>::name_cont() + _("n, ");
    }
};

template<typename Type> struct type_caster<Type, std::enable_if_t<enoki::is_array<Type>::value>> {
    typedef typename Type::Value     Value;
    typedef typename Type::Scalar Scalar;

    bool load(handle src, bool) {
        auto arr = array_t<Scalar, array::f_style | array::forcecast>::ensure(src);
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

        const Scalar *buf = static_cast<const Scalar *>(arr.data());
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

        stride[0] = sizeof(Scalar);
        for (size_t i = 1; i < shape.size(); ++i)
            stride[i] = shape[i - 1] * stride[i - 1];

        buffer_info info(nullptr, sizeof(Scalar),
                         format_descriptor<Scalar>::value, shape.size(),
                         std::vector<size_t>(shape.begin(), shape.end()),
                         std::vector<size_t>(stride.begin(), stride.end()));

        array arr(info);
        Scalar *buf = static_cast<Scalar *>(arr.mutable_data());
        write_buffer(buf, src);
        return arr.release();
    }

    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

    static PYBIND11_DESCR name() {
        return pybind11::detail::type_descr(
            _("numpy.ndarray[dtype=") +
            npy_format_descriptor<Scalar>::name() + _(", shape=(") +
            array_shape_descr<Type>::name() + _(")]"));
    }

    operator Type*() { return &value; }
    operator Type&() { return value; }

private:
    template <typename T, std::enable_if_t<!enoki::is_array<T>::value, int> = 0>
    static ENOKI_INLINE void write_buffer(Scalar *&, const T &) { }

    template <typename T, std::enable_if_t<enoki::is_array<T>::value, int> = 0>
    static ENOKI_INLINE void write_buffer(Scalar *&buf, const T &value_) {
        const auto &value = value_.derived();
        size_t size = value.size();

        if (std::is_arithmetic<enoki::value_t<T>>::value) {
            memcpy(buf, &value.coeff(0), sizeof(enoki::value_t<T>) * size);
            buf += size;
        } else {
            for (size_t i = 0; i < size; ++i)
                write_buffer(buf, value.coeff(i));
        }
    }

    template <typename T, std::enable_if_t<!enoki::is_array<T>::value, int> = 0>
    static ENOKI_INLINE void read_buffer(const Scalar *&, T &) { }

    template <typename T, std::enable_if_t<enoki::is_array<T>::value, int> = 0>
    static ENOKI_INLINE void read_buffer(const Scalar *&buf, T &value_) {
        auto &value = value_.derived();
        size_t size = value.size();

        if (std::is_arithmetic<enoki::value_t<T>>::value) {
            memcpy(&value.coeff(0), buf, sizeof(enoki::value_t<T>) * size);
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

template <typename T, typename = void> struct reference_dynamic { using type = T; };
template <typename T>
struct reference_dynamic<T, std::enable_if_t<is_dynamic_nested<T>::value>> {
    using type = std::add_lvalue_reference_t<T>;
};
template <typename T>
using reference_dynamic_t = typename reference_dynamic<T>::type;

template <typename Func, typename Return, typename... Args>
auto vectorize_wrapper_detail(Func &&f_, Return (*)(Args...)) {
    return [f = std::forward<Func>(f_)](reference_dynamic_t<enoki::dynamic_t<Args>>... args) {
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

// -----------------------------------------------------------------------
//! @{ \name Enoki accessors for static & dynamic vectorization over tuples
// -----------------------------------------------------------------------

/* Is this type dynamic? */
template <typename... Arg> struct is_dynamic_nested_impl<std::tuple<Arg...>> {
    static constexpr bool value = !enoki::detail::all_of<(!is_dynamic_nested<Arg>::value)...>::value;
};

/* Create a dynamic version of this type on demand */
template <typename... Arg> struct dynamic_impl<std::tuple<Arg...>> {
    using type = std::tuple<dynamic_t<Arg>...>;
};

/* How many packets are stored in this instance? */
template <typename... Arg> size_t packets(const std::tuple<Arg...> &t) {
    return packets(std::get<0>(t));
}

/* What is the size of the dynamic dimension of this instance? */
template <typename... Arg> size_t dynamic_size(const std::tuple<Arg...> &t) {
    return dynamic_size(std::get<0>(t));
}

template <typename... Arg, size_t... Index>
void dynamic_resize(std::tuple<Arg...> &t, size_t size, std::index_sequence<Index...>) {
    bool unused[] = { (dynamic_resize(std::get<Index>(t), size), false)... };
    (void) unused;
}

/* Resize the dynamic dimension of this instance */
template <typename... Arg>
void dynamic_resize(std::tuple<Arg...> &t, size_t size) {
    dynamic_resize(t, size, std::make_index_sequence<sizeof...(Arg)>());
}

template <typename... Arg, size_t... Index>
auto ref_wrap(std::tuple<Arg...> &t, std::index_sequence<Index...>) {
    return std::tuple<decltype(ref_wrap(std::declval<Arg&>()))...>(
        ref_wrap(std::get<Index>(t))...);
}

/* Construct a wrapper that references the data of this instance */
template <typename... Arg> auto ref_wrap(std::tuple<Arg...> &t) {
    return ref_wrap(t, std::make_index_sequence<sizeof...(Arg)>());
}

template <typename... Arg, size_t... Index>
auto ref_wrap(const std::tuple<Arg...> &t, std::index_sequence<Index...>) {
    return std::tuple<decltype(ref_wrap(std::declval<const Arg&>()))...>(
        ref_wrap(std::get<Index>(t))...);
}

/* Construct a wrapper that references the data of this instance (const) */
template <typename... Arg> auto ref_wrap(const std::tuple<Arg...> &t) {
    return ref_wrap(t, std::make_index_sequence<sizeof...(Arg)>());
}

template <typename... Arg, size_t... Index>
auto packet(std::tuple<Arg...> &t, size_t i, std::index_sequence<Index...>) {
    return std::tuple<decltype(packet(std::declval<Arg&>(), i))...>(
        packet(std::get<Index>(t), i)...);
}

/* Return the i-th packet */
template <typename... Arg> auto packet(std::tuple<Arg...> &t, size_t i) {
    return packet(t, i, std::make_index_sequence<sizeof...(Arg)>());
}

template <typename... Arg, size_t... Index>
auto packet(const std::tuple<Arg...> &t, size_t i, std::index_sequence<Index...>) {
    return std::tuple<decltype(packet(std::declval<const Arg&>(), i))...>(
        packet(std::get<Index>(t), i)...);
}

/* Return the i-th packet (const) */
template <typename... Arg> auto packet(const std::tuple<Arg...> &t, size_t i) {
    return packet(t, i, std::make_index_sequence<sizeof...(Arg)>());
}

template <typename... Arg, size_t... Index>
auto slice(std::tuple<Arg...> &t, size_t i, std::index_sequence<Index...>) {
    return std::tuple<decltype(slice(std::declval<Arg&>(), i))...>(
        slice(std::get<Index>(t), i)...);
}

/* Return the i-th slice */
template <typename... Arg> auto slice(std::tuple<Arg...> &t, size_t i) {
    return slice(t, i, std::make_index_sequence<sizeof...(Arg)>());
}

template <typename... Arg, size_t... Index>
auto slice(const std::tuple<Arg...> &t, size_t i, std::index_sequence<Index...>) {
    return std::tuple<decltype(slice(std::declval<const Arg&>(), i))...>(
        slice(std::get<Index>(t), i)...);
}

/* Return the i-th slice (const) */
template <typename... Arg> auto slice(const std::tuple<Arg...> &t, size_t i) {
    return slice(t, i, std::make_index_sequence<sizeof...(Arg)>());
}

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
