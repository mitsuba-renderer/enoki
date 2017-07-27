/*
    enoki/alloc.h -- Aligned memory allocator

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "definitions.h"
#include <cstdlib>
#include <new>

#if defined(__linux__)
#  include <malloc.h>
#endif

#if defined(ENOKI_USE_MEMKIND)
#  include <hbwmalloc.h>
#endif

NAMESPACE_BEGIN(enoki)

// -----------------------------------------------------------------------
//! @{ \name Memory allocation (heap)
// -----------------------------------------------------------------------

/// Allocate a suitably aligned memory block
inline ENOKI_MALLOC void *alloc(size_t size) {
    constexpr size_t align = sizeof(void *) > size_t(max_packet_size)
        ? sizeof(void *) : max_packet_size;
    ENOKI_TRACK_ALLOC

    void *ptr;
    #if defined(ENOKI_USE_MEMKIND)
        (void) align;
        ptr = hbw_malloc(size);
    #elif defined(_WIN32)
        ptr = _aligned_malloc(size, align);
    #else
        if (posix_memalign(&ptr, align, size) != 0)
            ptr = nullptr;
    #endif

    if (!ptr && size != 0)
        throw std::bad_alloc();

    return ptr;
}

/// Re-allocate a suitably aligned memory block
inline ENOKI_MALLOC void *realloc(void *cur, size_t size) {
    constexpr size_t align = sizeof(void *) > size_t(max_packet_size)
        ? sizeof(void *) : max_packet_size;
    ENOKI_TRACK_ALLOC

    void *ptr;
    #if defined(ENOKI_USE_MEMKIND)
        (void) align;
        ptr = hbw_realloc(cur, size);
    #elif defined(_WIN32)
        ptr = _aligned_realloc(cur, size, align);
    #else
        ptr = ::realloc(cur, size);
        if (((uintptr_t) ptr) % align != 0) {
            /* Oops, we lost the alignment :( */
            cur = ptr;
            if (posix_memalign(&ptr, align, size) == 0) {
                memcpy(ptr, cur, size);
                free(cur);
            } else {
                ptr = nullptr;
            }
        }
    #endif

    if (!ptr && size != 0)
        throw std::bad_alloc();

    return ptr;
}

/// Allocate a suitably aligned memory block of the given type
template <typename T> static ENOKI_INLINE ENOKI_MALLOC T *alloc(size_t size) {
    return (T *) enoki::alloc(sizeof(T) * size);
}

/// Re-allocate a suitably aligned memory block of the given type
template <typename T> static ENOKI_INLINE ENOKI_MALLOC T *realloc(void *cur, size_t size) {
    return (T *) enoki::realloc(cur, sizeof(T) * size);
}

/// Release aligned memory
static ENOKI_INLINE void dealloc(void *ptr) {
    ENOKI_TRACK_DEALLOC
    #if defined(ENOKI_USE_MEMKIND)
        hbw_free(ptr);
    #elif defined(_WIN32)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

/// Aligned memory allocator for STL data structures
template <typename T> struct aligned_allocator : public std::allocator<T> {
    using Base = std::allocator<T>;
    using typename Base::pointer;
    using typename Base::size_type;

    template <typename U> struct rebind { using other = aligned_allocator<U>; };

    aligned_allocator() { }
    template <class U> aligned_allocator(const aligned_allocator<U> &) { }

    pointer allocate(size_type n, const void * = nullptr) { return enoki::alloc<T>(n); }
    void deallocate(pointer p, size_type) { enoki::dealloc(p); }
};

/// Aligned memory deallocator (e.g. for std::unique_ptr)
struct aligned_deleter {
    void operator()(void *ptr) { dealloc(ptr); }
};

#define ENOKI_ALIGNED_OPERATOR_NEW() \
    ENOKI_INLINE void *operator new(size_t size) { return ::enoki::alloc(size); } \
    ENOKI_INLINE void *operator new[](size_t size) { return ::enoki::alloc(size); } \
    ENOKI_INLINE void *operator new(size_t, void *ptr) { return ptr; } \
    ENOKI_INLINE void *operator new[](size_t, void *ptr) { return ptr; } \
    ENOKI_INLINE void operator delete(void *ptr) { return ::enoki::dealloc(ptr); } \
    ENOKI_INLINE void operator delete[](void *ptr) { return ::enoki::dealloc(ptr); } \
    ENOKI_INLINE void operator delete(void *ptr, size_t) { return ::enoki::dealloc(ptr); } \
    ENOKI_INLINE void operator delete[](void *ptr, size_t) { return ::enoki::dealloc(ptr); } \
    ENOKI_INLINE void operator delete(void *, void*) { } \
    ENOKI_INLINE void operator delete[](void *, void*) { }

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
