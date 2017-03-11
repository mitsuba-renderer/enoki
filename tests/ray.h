/*
    tests/ray.h -- showcases how to extend Enoki vectorization to custom
    data types

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#include <enoki/array.h>

using namespace enoki;

/**
 * Generic 3D ray class: can represent either a single ray, a static ray
 * bundle, or a dynamic heap-allocated bundle of rays
 */
template <typename Vector_> struct Ray {
    using Vector = Vector_;
    using Value = value_t<Vector>;

    Vector o;
    Vector d;

    // -----------------------------------------------------------------------
    //! @{ \name Constructors & assignment operators
    // -----------------------------------------------------------------------

    Ray() { }
    Ray(const Vector &o, const Vector &d) : o(o), d(d) { }

    template <typename T> Ray(const Ray<T> &r) : o(r.o), d(r.d) { }
    template <typename T> Ray(Ray<T> &&r) : o(std::move(r.o)), d(std::move(r.d)) { }

    template <typename T> Ray &operator=(const Ray<T> &r) {
        o = r.o;
        d = r.d;
        return *this;
    }

    template <typename T> Ray &operator=(Ray<T> &&r) {
        o = std::move(r.o);
        d = std::move(r.d);
        return *this;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Miscellaneous
    // -----------------------------------------------------------------------

    /// Compute a position along a ray
    Vector operator()(Value t) const { return o + t*d; }

    //! @}
    // -----------------------------------------------------------------------
};


NAMESPACE_BEGIN(enoki)

// -----------------------------------------------------------------------
//! @{ \name Enoki accessors for static & dynamic vectorization
// -----------------------------------------------------------------------

template <typename T> struct dynamic_support<Ray<T>> {
    static constexpr bool is_dynamic_nested = enoki::is_dynamic_nested<T>::value;
    using dynamic_t = Ray<enoki::make_dynamic_t<T>>;
    using Value = Ray<T>;

    static ENOKI_INLINE size_t dynamic_size(const Value &value) {
        return enoki::dynamic_size(value.o);
    }

    static ENOKI_INLINE size_t packets(const Value &value) {
        return enoki::packets(value.o);
    }

    static ENOKI_INLINE void dynamic_resize(Value &value, size_t size) {
        enoki::dynamic_resize(value.o, size);
        enoki::dynamic_resize(value.d, size);
    }

    template <typename T2>
    static ENOKI_INLINE auto packet(T2 &&value, size_t i) {
        return Ray<decltype(enoki::packet(value.o, i))>(
            enoki::packet(value.o, i), enoki::packet(value.d, i));
    }

    template <typename T2>
    static ENOKI_INLINE auto slice(T2 &&value, size_t i) {
        return Ray<decltype(enoki::slice(value.o, i))>(
            enoki::slice(value.o, i), enoki::slice(value.d, i));
    }

    template <typename T2> static ENOKI_INLINE auto ref_wrap(T2 &&value) {
        return Ray<decltype(enoki::ref_wrap(value.o))>(
            enoki::ref_wrap(value.o), enoki::ref_wrap(value.d));
    }
};

//! @}
// =======================================================================

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
