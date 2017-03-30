/*
    enoki/stl.h -- vectorization support for STL pairs, tuples, and arrays

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array.h"

NAMESPACE_BEGIN(enoki)

template <typename Arg0, typename Arg1> struct dynamic_support<std::pair<Arg0, Arg1>> {
    static constexpr bool is_dynamic_nested =
        enoki::is_dynamic_nested<Arg0>::value || enoki::is_dynamic_nested<Arg1>::value;
    using dynamic_t = std::pair<enoki::make_dynamic_t<Arg0>, enoki::make_dynamic_t<Arg1>>;
    using Value = std::pair<Arg0, Arg1>;

    static ENOKI_INLINE size_t dynamic_size(const Value &value) {
        return enoki::dynamic_size(value.first);
    }

    static ENOKI_INLINE size_t packets(const Value &value) {
        return enoki::packets(value.first);
    }

    static ENOKI_INLINE void dynamic_resize(Value &value, size_t size) {
        enoki::dynamic_resize(value.first, size);
        enoki::dynamic_resize(value.second, size);
    }

    template <typename T2>
    static ENOKI_INLINE auto packet(T2 &&value, size_t i) {
        return std::pair<decltype(enoki::packet(value.first, i)),
                         decltype(enoki::packet(value.second, i))>(
            enoki::packet(value.first, i), enoki::packet(value.second, i));
    }

    template <typename T2>
    static ENOKI_INLINE auto slice(T2 &&value, size_t i) {
        return std::pair<decltype(enoki::slice(value.first, i)),
                         decltype(enoki::slice(value.second, i))>(
            enoki::slice(value.first, i), enoki::slice(value.second, i));
    }

    template <typename T2>
    static ENOKI_INLINE auto ref_wrap(T2 &&value) {
        return std::pair<decltype(enoki::ref_wrap(value.first)),
                         decltype(enoki::ref_wrap(value.second))>(
            enoki::ref_wrap(value.first), enoki::ref_wrap(value.second));
    }
};

template <typename... Args> struct dynamic_support<std::tuple<Args...>> {
    static constexpr bool is_dynamic_nested = !enoki::detail::all_of<(!enoki::is_dynamic_nested<Args>::value)...>::value;
    using dynamic_t = std::tuple<enoki::make_dynamic_t<Args>...>;
    using Value = std::tuple<Args...>;

    static ENOKI_INLINE size_t dynamic_size(const Value &value) {
        return enoki::dynamic_size(std::get<0>(value));
    }

    static ENOKI_INLINE size_t packets(const Value &value) {
        return enoki::packets(std::get<0>(value));
    }

    static ENOKI_INLINE void dynamic_resize(Value &value, size_t size) {
        dynamic_resize(value, size, std::make_index_sequence<sizeof...(Args)>());
    }

    template <typename T2>
    static ENOKI_INLINE auto packet(T2 &&value, size_t i) {
        return packet(std::forward<T2>(value), i, std::make_index_sequence<sizeof...(Args)>());
    }

    template <typename T2>
    static ENOKI_INLINE auto slice(T2 &&value, size_t i) {
        return slice(std::forward<T2>(value), i, std::make_index_sequence<sizeof...(Args)>());
    }

    template <typename T2>
    static ENOKI_INLINE auto ref_wrap(T2 &&value) {
        return ref_wrap(std::forward<T2>(value), std::make_index_sequence<sizeof...(Args)>());
    }

private:
    template <size_t... Index>
    static ENOKI_INLINE void dynamic_resize(Value &value, size_t i, std::index_sequence<Index...>) {
        bool unused[] = { (enoki::dynamic_resize(std::get<Index>(value), i), false)... };
        (void) unused;
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto packet(T2 &&value, size_t i, std::index_sequence<Index...>) {
        return std::tuple<decltype(enoki::packet(std::get<Index>(value), i))...>(
            enoki::packet(std::get<Index>(value), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice(T2 &&value, size_t i, std::index_sequence<Index...>) {
        return std::tuple<decltype(enoki::slice(std::get<Index>(value), i))...>(
            enoki::slice(std::get<Index>(value), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto ref_wrap(T2 &&value, std::index_sequence<Index...>) {
        return std::tuple<decltype(enoki::ref_wrap(std::get<Index>(value)))...>(
            enoki::ref_wrap(std::get<Index>(value))...);
    }
};

template <typename T, size_t Size> struct dynamic_support<std::array<T, Size>> {
    static constexpr bool is_dynamic_nested = enoki::is_dynamic_nested<T>::value;
    using dynamic_t = std::array<enoki::make_dynamic_t<T>, Size>;
    using Value = std::array<T, Size>;

    static ENOKI_INLINE size_t dynamic_size(const Value &value) {
        return enoki::dynamic_size(value[0]);
    }

    static ENOKI_INLINE size_t packets(const Value &value) {
        return enoki::packets(value[0]);
    }

    static ENOKI_INLINE void dynamic_resize(Value &value, size_t size) {
        for (size_t i = 0; i < Size; ++i)
            enoki::dynamic_resize(value[i], size);
    }

    template <typename T2>
    static ENOKI_INLINE auto packet(T2 &&value, size_t i) {
        return packet(std::forward<T2>(value), i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto slice(T2 &&value, size_t i) {
        return slice(std::forward<T2>(value), i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto ref_wrap(T2 &&value) {
        return ref_wrap(std::forward<T2>(value), std::make_index_sequence<Size>());
    }

private:
    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto packet(T2 &&value, size_t i, std::index_sequence<Index...>) {
        return std::array<decltype(enoki::packet(value[0], i)), Size>{{
            enoki::packet(value[Index], i)...}};
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice(T2 &&value, size_t i, std::index_sequence<Index...>) {
        return std::array<decltype(enoki::slice(value[0], i)), Size>{{
            enoki::slice(value[Index], i)...}};
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto ref_wrap(T2 &&value, std::index_sequence<Index...>) {
        return std::array<decltype(enoki::ref_wrap(value[0])), Size>{{
            enoki::ref_wrap(value[Index])...}};
    }
};

NAMESPACE_END(enoki)
