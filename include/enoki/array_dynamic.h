/*
    enoki/array_dynamic.h -- Dynamic heap-allocated array

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "array_generic.h"
#include <memory>

#if defined(__linux__)
#  include <malloc.h>
#endif

NAMESPACE_BEGIN(enoki)

/// Allocate a suitably aligned memory block
inline ENOKI_MALLOC void *alloc(size_t size) {
    constexpr size_t align = std::max((size_t) ENOKI_MAX_PACKET_SIZE, sizeof(void *));

    void *ptr;
    #if defined(_WIN32)
        ptr = _aligned_malloc(size, align);
    #elif defined(__APPLE__)
        if (posix_memalign(&ptr, align, size) != 0)
            ptr = nullptr;
    #else
        ptr = memalign(align, size);
    #endif

    if (!ptr)
        throw std::bad_alloc();

    return ptr;
}

/// Allocate a suitably aligned memory block of the given type
template <typename T> static ENOKI_INLINE ENOKI_MALLOC T *alloc(size_t size) {
    return (T *) enoki::alloc(sizeof(T) * size);
}

/// Release aligned memory
inline void dealloc(void *ptr) {
    #if defined(_WIN32)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

/// Aligned memory deallocator
struct aligned_deleter {
    void operator()(void *ptr) { dealloc(ptr); }
};

template <typename Packet_, typename Derived_>
struct DynamicArrayBase : ArrayBase<scalar_t<Packet_>, Derived_> {
    using Packet                              = Packet_;
    using Scalar                              = scalar_t<Packet_>;
    using BaseScalar                          = base_scalar_t<Packet_>;
    static constexpr size_t       PacketSize  = Packet::Size;
    static constexpr bool         Approx      = Packet::Approx;
    static constexpr RoundingMode Mode        = Packet::Mode;
    static constexpr bool         Dynamic     = true;
};

template <typename Packet>
struct DynamicArrayReference : DynamicArrayBase<Packet, DynamicArrayReference<Packet>> {
    using Base         = DynamicArrayBase<Packet, DynamicArrayReference<Packet>>;

    DynamicArrayReference(Packet *packets) : m_packets(packets) { }

    ENOKI_INLINE Packet &packet(size_t i) {
        Packet *packets = (Packet *) ENOKI_ASSUME_ALIGNED(m_packets);
        return packets[i];
    }

    ENOKI_INLINE const Packet &packet(size_t i) const {
        const Packet *packets = (const Packet *) ENOKI_ASSUME_ALIGNED(m_packets);
        return packets[i];
    }

private:
    Packet *m_packets;
};

template <typename Packet_, typename Derived_>
struct DynamicArrayImpl : DynamicArrayBase<Packet_, Derived_> {
    // -----------------------------------------------------------------------
    //! @{ \name Aliases and constants
    // -----------------------------------------------------------------------

    using Base         = DynamicArrayBase<Packet_, Derived_>;

    using typename Base::Packet;
    using typename Base::Scalar;
    using typename Base::Derived;
    using Base::derived;
    using Base::PacketSize;

    using PacketHolder = std::unique_ptr<Packet[], aligned_deleter>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors
    // -----------------------------------------------------------------------

    DynamicArrayImpl() { }

    DynamicArrayImpl(const DynamicArrayImpl &other) { operator=(other); }

    DynamicArrayImpl(size_t size) { resize_(size); }

    DynamicArrayImpl(Scalar *ptr, size_t size)
        : m_packets((Packet *) ptr), m_packets_allocated(0), m_size(size) { }

    DynamicArrayImpl(DynamicArrayImpl &&other)
        :  m_packets(std::move(other.m_packets)),
           m_packets_allocated(other.m_packets_allocated),
           m_size(other.m_size) {
        other.m_packets_allocated = other.m_size = 0;
    }

    ~DynamicArrayImpl() {
        /* Don't deallocate mapped memory */
        if (m_packets_allocated == 0)
            m_packets.release();
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Dynamic array properties
    // -----------------------------------------------------------------------

    ENOKI_INLINE bool empty() const { return m_size == 0; }
    ENOKI_INLINE size_t size() const { return m_size; }
    ENOKI_INLINE size_t dynamic_size() const { return m_size; }
    ENOKI_INLINE size_t packets() const { return (m_size + PacketSize - 1) / PacketSize; }
    ENOKI_INLINE size_t capacity() const { return m_packets_allocated * PacketSize; }
    ENOKI_INLINE bool is_mapped() const { return !empty() && m_packets_allocated == 0; }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Component and packet access
    // -----------------------------------------------------------------------

    ENOKI_INLINE const Scalar *data() const {
        return (const Scalar *) ENOKI_ASSUME_ALIGNED(m_packets.get());
    }

    ENOKI_INLINE Scalar *data() {
        return (Scalar *) ENOKI_ASSUME_ALIGNED(m_packets.get());
    }

    ENOKI_INLINE Packet &packet(size_t i) {
        Packet *packets = (Packet *) ENOKI_ASSUME_ALIGNED(m_packets.get());
        return packets[i];
    }

    ENOKI_INLINE const Packet &packet(size_t i) const {
        const Packet *packets = (const Packet *) ENOKI_ASSUME_ALIGNED(m_packets.get());
        return packets[i];
    }

    ENOKI_INLINE Scalar& coeff(size_t i) {
        return m_packets[i / PacketSize][i % PacketSize];
    }

    ENOKI_INLINE Scalar& coeff(size_t i) const {
        return m_packets[i / PacketSize][i % PacketSize];
    }

    ENOKI_INLINE DynamicArrayReference<Packet> ref_() {
        return DynamicArrayReference<Packet>(m_packets.get());
    }

    ENOKI_INLINE DynamicArrayReference<const Packet> ref_() const {
        return DynamicArrayReference<const Packet>(m_packets.get());
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Assignment operators
    // -----------------------------------------------------------------------

    DynamicArrayImpl &operator=(const DynamicArrayImpl &other) {
        resize_(other.size());
        memcpy(m_packets.get(), other.m_packets.get(),
               m_packets_allocated * sizeof(Packet));
        return derived();
    }

    DynamicArrayImpl &operator=(DynamicArrayImpl &&value) {
        m_packets = std::move(value.m_packets);
        m_packets_allocated = value.m_packets_allocated;
        m_size = value.m_size;
        value.m_packets_allocated = value.m_size = 0;
        return derived();
    }

    //! @}
    // -----------------------------------------------------------------------

    void dynamic_resize(size_t size) { resize_(size); }

    /**
     * \brief Resize the buffer to the desired size
     *
     * When the capacity is insufficient, the implementation destroys the
     * current contents and allocates a new (uninitialized) buffer
     *
     * When compiled in debug mode, newly allocated memory (if any) will be
     * initialized with NaNs.
     */
    ENOKI_NOINLINE void resize_(size_t size) {
        if (size <= m_packets_allocated * PacketSize) {
            m_size = size;
            return;
        }

        if (m_packets_allocated == 0)
            m_packets.release();

        m_packets_allocated = (size + PacketSize - 1) / PacketSize;
        m_packets = PacketHolder(
            enoki::alloc<Packet>(m_packets_allocated));
        m_size = size;

        #if !defined(NDEBUG)
            for (size_t i = 0; i < m_packets_allocated; ++i)
                new (&m_packets[i]) Packet();
        #endif
    }

protected:
    PacketHolder m_packets;
    size_t m_packets_allocated = 0;
    size_t m_size = 0;
};

template <typename Type_>
struct DynamicArray : DynamicArrayImpl<Type_, DynamicArray<Type_>> {
    using Base = DynamicArrayImpl<Type_, DynamicArray<Type_>>;
    using Base::Base;
    using Base::operator=;
};

NAMESPACE_BEGIN(detail)

/// Type trait to access the packet type underlying a potentially nested array
template <typename T, typename = void>
struct packet_ {
    using type = T;
};

template <typename T>
struct packet_<T, std::enable_if_t<is_array<std::decay_t<typename std::decay_t<T>::Scalar>>::value>> {
    using type = typename packet_<typename std::decay_t<T>::Scalar>::type;
};

template <typename T>
using packet_t = typename packet_<T>::type;

/// Helper data structure to replace the packet type underlying a potentially nested array
template <typename T, typename Packet, typename = void>
struct replace_packet {
    using type = Packet;
};

template <typename T, typename Packet>
struct replace_packet<T, Packet, std::enable_if_t<is_array<std::decay_t<typename std::decay_t<T>::Scalar>>::value>> {
    using type =
        Array<typename replace_packet<typename std::decay_t<T>::Scalar, Packet>::type,
              std::decay_t<T>::Size>;
};

template <typename T, typename Packet>
using replace_packet_t = typename replace_packet<T, Packet>::type;



template <typename T>
using vectorize_t = replace_packet_t< T, DynamicArray<packet_t<T>>>;

/// Vectorized inner loop (void return value)
template <typename Func, typename... Args, size_t... Index>
ENOKI_INLINE void vectorize_inner_1(std::index_sequence<Index...>, Func &&f,
                                    size_t packet_count, Args &&... args) {

    ENOKI_IVDEP ENOKI_NOUNROLL for (size_t i = 0; i < packet_count; ++i)
        f(packet(args, i)...);
}

/// Vectorized inner loop (non-void return value)
template <typename Func, typename Out, typename... Args, size_t... Index>
ENOKI_INLINE void vectorize_inner_2(std::index_sequence<Index...>, Func &&f,
                                    size_t packet_count, Out& out, Args &&... args) {

    ENOKI_IVDEP ENOKI_NOUNROLL for (size_t i = 0; i < packet_count; ++i)
        packet(out, i) = f(packet(args, i)...);
}

template <bool Check, typename Return, typename Func, typename... Args,
    std::enable_if_t<std::is_void<Return>::value, int> = 0>
ENOKI_INLINE void vectorize(Func&& f, Args&&... args) {
    size_t packet_count = 0;

    bool unused[] = { (
        (packet_count = (is_dynamic<Args>::value ? packets(args) : packet_count)),
        false)... };
    (void) unused;

    if (Check) {
        size_t dynamic_size = 0;
        bool unused2[] = { (
            (dynamic_size = (is_dynamic<Args>::value ? enoki::dynamic_size(args) : dynamic_size)),
            false)... };
        (void) unused2;

        bool status[] = { (!is_dynamic<Args>::value ||
                           (enoki::dynamic_size(args) == dynamic_size))... };
        for (bool flag : status)
            if (!flag)
                throw std::length_error("vectorize(): vector arguments have incompatible lengths");
    }

    vectorize_inner_1(std::make_index_sequence<sizeof...(Args)>(),
                      std::forward<Func>(f), packet_count, detail::ref(args)...);
}

template <bool Check, typename Return, typename Func, typename... Args,
    std::enable_if_t<!std::is_void<Return>::value, int> = 0>
ENOKI_INLINE auto vectorize(Func&& f, Args&&... args) {
    size_t packet_count = 0, dynamic_size = 0;

    vectorize_t<Return> out;

    bool unused[] = { (
        (packet_count = (is_dynamic<Args>::value ? packets(args) : packet_count)),
        false)... };

    bool unused2[] = { (
        (dynamic_size = (is_dynamic<Args>::value ? enoki::dynamic_size(args) : dynamic_size)),
        false)... };

    (void) unused;
    (void) unused2;

    out.dynamic_resize(dynamic_size);

    if (Check) {
        bool status[] = { (!is_dynamic<Args>::value ||
                           (enoki::dynamic_size(args) == dynamic_size))... };
        for (bool flag : status)
            if (!flag)
                throw std::length_error("vectorize(): vector arguments have incompatible lengths");
    }

    vectorize_inner_2(std::make_index_sequence<sizeof...(Args)>(),
                      std::forward<Func>(f), packet_count, out, detail::ref(args)...);

    return out;
}

NAMESPACE_END(detail)

template <typename Func, typename... Args>
ENOKI_INLINE auto vectorize(Func&& f, Args&&... args) {
#if defined(NDEBUG)
    constexpr bool Check = false;
#else
    constexpr bool Check = true;
#endif
    using Return = decltype(f(packet(args, 0)...));
    return detail::vectorize<Check, Return>(std::forward<Func>(f), args...);
}

template <typename Func, typename... Args>
ENOKI_INLINE auto vectorize_safe(Func&& f, Args&&... args) {
    using Return = decltype(f(packet(args, 0)...));
    return detail::vectorize<true, Return>(std::forward<Func>(f), args...);
}

NAMESPACE_END(enoki)
