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

/// Is this type dynamic?
template <typename T> struct is_dynamic_impl<Ray<T>> {
    static constexpr bool value = is_dynamic<T>::value;
};

/// Create a dynamic version of this type on demand
template <typename T> struct dynamic_impl<Ray<T>> {
    using type = Ray<dynamic_t<T>>;
};

/// How many packets are stored in this instance?
template <typename T> size_t packets(const Ray<T> &r) {
    return packets(r.o);
}

/// What is the size of the dynamic dimension of this instance?
template <typename T> size_t dynamic_size(const Ray<T> &r) {
    return dynamic_size(r.o);
}

/// Resize the dynamic dimension of this instance
template <typename T> void dynamic_resize(Ray<T> &r, size_t size) {
    dynamic_resize(r.o, size);
    dynamic_resize(r.d, size);
}

/// Return the i-th packet
template <typename T> auto packet(Ray<T> &r, size_t i) {
    using T2 = decltype(packet(r.o, i));
    return Ray<T2> { packet(r.o, i), packet(r.d, i) };
}

/// Return the i-th packet (const version)
template <typename T> auto packet(const Ray<T> &r, size_t i) {
    using T2 = decltype(packet(r.o, i));
    return Ray<T2> { packet(r.o, i), packet(r.d, i) };
}

/// Construct a wrapper that references the data of this instance
template <typename T> auto ref_wrap(Ray<T> &r) {
    using T2 = decltype(ref_wrap(r.o));
    return Ray<T2> { ref_wrap(r.o), ref_wrap(r.d) };
}

/// Construct a wrapper that references the data of this instance (const version)
template <typename T> auto ref_wrap(const Ray<T> &r) {
    using T2 = decltype(ref_wrap(r.o));
    return Ray<T2> { ref_wrap(r.o), ref_wrap(r.d) };
}

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
