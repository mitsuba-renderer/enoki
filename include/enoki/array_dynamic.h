/*
    enoki/array_dynamic.h -- Dynamic heap-allocated array

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array_generic.h"

NAMESPACE_BEGIN(enoki)

template <typename T> struct dynamic_support<T, enable_if_static_array_t<T>> {
    static constexpr size_t Size = array_size<T>::value;
    using Value = value_t<T>;

    static constexpr bool is_dynamic_nested = enoki::is_dynamic_nested<Value>::value;

    using dynamic_t = std::conditional_t<
        array_depth<T>::value == 1, DynamicArray<std::decay_t<T>>,
        typename T::template ReplaceType<make_dynamic_t<Value>>>;

    static ENOKI_INLINE size_t slices(const T &value) { return enoki::slices(value.x()); }
    static ENOKI_INLINE size_t packets(const T& value) { return enoki::packets(value.x()); }

    static ENOKI_INLINE void set_slices(T &value, size_t size) {
        for (size_t i = 0; i < Size; ++i)
            enoki::set_slices(value.coeff(i), size);
    }

    template <typename T2>
    static ENOKI_INLINE auto packet(T2&& value, size_t i) {
        return packet(value, i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto slice(T2&& value, size_t i) {
        return slice(value, i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto slice_ptr(T2&& value, size_t i) {
        return slice_ptr(value, i, std::make_index_sequence<Size>());
    }

    template <typename T2>
    static ENOKI_INLINE auto ref_wrap(T2&& value) {
        return ref_wrap(value, std::make_index_sequence<Size>());
    }

    template <typename Mem, typename T2, typename Mask, enable_if_array_t<Mem> = 0>
    static ENOKI_INLINE size_t compress(Mem &mem, const T2& value, const Mask &mask) {
        size_t result = 0;
        for (size_t i = 0; i < Size; ++i)
            result = enoki::compress(mem.coeff(i), value.coeff(i), mask);
        return result;
    }

    template <typename Mem, typename T2, typename Mask, enable_if_not_array_t<Mem> = 0>
    static ENOKI_INLINE size_t compress(Mem &mem, const T2& value, const Mask &mask) {
        return value.compress_(mem, mask);
    }

private:
    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto packet(T2&& value, size_t i, std::index_sequence<Index...>) {
        using Entry = decltype(enoki::packet(value.coeff(0), i));
        using Return = typename T::template ReplaceType<Entry>;
        return Return(enoki::packet(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice(T2&& value, size_t i, std::index_sequence<Index...>) {
        using Entry = decltype(enoki::slice(value.coeff(0), i));
        using Return = typename T::template ReplaceType<Entry>;
        return Return(enoki::slice(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice_ptr(T2&& value, size_t i, std::index_sequence<Index...>) {
        using Entry = decltype(enoki::slice_ptr(value.coeff(0), i));
        using Return = typename T::template ReplaceType<Entry>;
        return Return(enoki::slice_ptr(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto ref_wrap(T2&& value, std::index_sequence<Index...>) {
        using Entry = decltype(enoki::ref_wrap(value.coeff(0)));
        using Return = typename T::template ReplaceType<Entry>;
        return Return(enoki::ref_wrap(value.coeff(Index))...);
    }
};

template <typename T> struct dynamic_support<T, enable_if_dynamic_array_t<T>> {
    static constexpr bool is_dynamic_nested = true;
    using dynamic_t = T;

    static ENOKI_INLINE size_t slices(const T &value) { return value.slices_(); }
    static ENOKI_INLINE size_t packets(const T& value) { return value.packets_(); }

    static ENOKI_INLINE void set_slices(T &value, size_t size) {
        value.set_slices_(size);
    }

    template <typename T2> static ENOKI_INLINE decltype(auto) packet(T2 &&value, size_t i) {
        return value.packet_(i);
    }

    template <typename T2> static ENOKI_INLINE decltype(auto) slice(T2 &&value, size_t i) {
        return value.slice_(i);
    }

    template <typename T2> static ENOKI_INLINE decltype(auto) slice_ptr(T2 &&value, size_t i) {
        return value.slice_ptr_(i);
    }

    template <typename T2> static ENOKI_INLINE decltype(auto) ref_wrap(T2 &&value) {
        return value.ref_wrap_();
    }
};

template <typename Packet_, typename Derived_>
struct DynamicArrayBase : ArrayBase<value_t<Packet_>, Derived_> {
    using Packet                              = Packet_;
    using Value                               = value_t<Packet_>;
    using Scalar                              = scalar_t<Packet_>;
    static constexpr size_t       PacketSize  = Packet::Size;
    static constexpr bool         Approx      = Packet::Approx;
    static constexpr RoundingMode Mode        = Packet::Mode;
    static constexpr bool IsMask              = Packet::IsMask;
};

template <typename Packet>
struct DynamicArrayReference : DynamicArrayBase<Packet, DynamicArrayReference<Packet>> {
    using Base = DynamicArrayBase<Packet, DynamicArrayReference<Packet>>;
    using Mask = DynamicArrayReference<mask_t<Packet>>;

    DynamicArrayReference(Packet *packets) : m_packets(packets) { }

    ENOKI_INLINE Packet &packet_(size_t i) {
        return ((Packet *) ENOKI_ASSUME_ALIGNED(m_packets))[i];
    }

    ENOKI_INLINE const Packet &packet_(size_t i) const {
        return ((const Packet *) ENOKI_ASSUME_ALIGNED(m_packets))[i];
    }

    template <typename T>
    using ReplaceType = DynamicArrayReference<T>;

private:
    Packet *m_packets;
};

template <typename Packet_, typename Derived_>
struct DynamicArrayImpl : DynamicArrayBase<Packet_, Derived_> {
    // -----------------------------------------------------------------------
    //! @{ \name Aliases and constants
    // -----------------------------------------------------------------------

    using Base = DynamicArrayBase<Packet_, Derived_>;

    using typename Base::Packet;
    using typename Base::Value;
    using typename Base::Scalar;
    using typename Base::Derived;
    using Base::derived;
    using Base::PacketSize;

    using Mask = DynamicArray<mask_t<Packet_>>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors
    // -----------------------------------------------------------------------

    DynamicArrayImpl() { }

    DynamicArrayImpl(const DynamicArrayImpl &other) { operator=(other); }

    DynamicArrayImpl(Value value, size_t size = 1) {
        resize_(size);
        operator=(value);
    }

    DynamicArrayImpl(Value *ptr, size_t size)
        : m_packets((Packet *) ptr), m_packets_allocated(0), m_size(size) { }

    DynamicArrayImpl(DynamicArrayImpl &&other)
        :  m_packets(other.m_packets),
           m_packets_allocated(other.m_packets_allocated),
           m_size(other.m_size) {
        other.m_packets_allocated = other.m_size = 0;
        other.m_packets = nullptr;
    }

    ~DynamicArrayImpl() {
        /* Don't deallocate mapped memory */
        if (m_packets_allocated)
            dealloc(m_packets);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Dynamic array properties
    // -----------------------------------------------------------------------

    ENOKI_INLINE bool empty() const { return m_size == 0; }
    ENOKI_INLINE size_t size() const { return m_size; }
    ENOKI_INLINE size_t capacity() const { return m_packets_allocated * PacketSize; }
    ENOKI_INLINE bool is_mapped() const { return !empty() && m_packets_allocated == 0; }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Helper routines used to implement enoki::vectorize()
    // -----------------------------------------------------------------------

    ENOKI_INLINE size_t slices_() const { return m_size; }

    ENOKI_INLINE size_t packets_() const { return (m_size + PacketSize - 1) / PacketSize; }

    ENOKI_INLINE Packet &packet_(size_t i) {
        return ((Packet *) ENOKI_ASSUME_ALIGNED(m_packets))[i];
    }

    ENOKI_INLINE const Packet &packet_(size_t i) const {
        return ((const Packet *) ENOKI_ASSUME_ALIGNED(m_packets))[i];
    }

    ENOKI_INLINE Value &slice_(size_t i) {
        return m_packets[i / PacketSize][i % PacketSize];
    }

    ENOKI_INLINE const Value &slice_(size_t i) const {
        return m_packets[i / PacketSize][i % PacketSize];
    }

    ENOKI_INLINE Value *slice_ptr_(size_t i) {
        return &m_packets[i / PacketSize][i % PacketSize];
    }

    ENOKI_INLINE const Value *slice_ptr_(size_t i) const {
        return &m_packets[i / PacketSize][i % PacketSize];
    }

    ENOKI_INLINE DynamicArrayReference<Packet> ref_wrap_() {
        return DynamicArrayReference<Packet>(m_packets);
    }

    ENOKI_INLINE DynamicArrayReference<const Packet> ref_wrap_() const {
        return DynamicArrayReference<const Packet>(m_packets);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Functions to access the array contents
    // -----------------------------------------------------------------------

    ENOKI_INLINE const Value *data() const {
        return (const Value *) ENOKI_ASSUME_ALIGNED(m_packets);
    }

    ENOKI_INLINE Value *data() {
        return (Value *) ENOKI_ASSUME_ALIGNED(m_packets);
    }

    ENOKI_INLINE decltype(auto) coeff(size_t i) {
        return m_packets[i / PacketSize].coeff(i % PacketSize);
    }

    ENOKI_INLINE decltype(auto) coeff(size_t i) const {
        return m_packets[i / PacketSize].coeff(i % PacketSize);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization helper functions
    // -----------------------------------------------------------------------

    static Derived zero_(size_t size) {
        Derived result(zero<Value>(), size);
        return result;
    }

    static Derived index_sequence_(size_t size) {
        Derived result;
        result.resize_(size);
        Packet packet = index_sequence<Packet>(),
               shift = Value(PacketSize);
        for (size_t i = 0; i < result.packets_(); ++i) {
            result.packet_(i) = packet;
            packet += shift;
        }
        return result;
    }

    static Derived linspace_(size_t size, Value min, Value max) {
        Derived result;
        result.resize_(size);

        Value step = (max - min) / Value(size - 1);

        Packet packet = linspace<Packet>(min, min + step * (PacketSize - 1)),
               shift = Value(step * PacketSize);

        for (size_t i = 0; i < result.packets_(); ++i) {
            result.packet_(i) = packet;
            packet += shift;
        }
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Assignment operators
    // -----------------------------------------------------------------------

    DynamicArrayImpl &operator=(Value value) {
        Packet packet(value);
        for (size_t i  = 0; i < packets_(); ++i)
            packet_(i) = packet;
        return derived();
    }

    DynamicArrayImpl &operator=(const DynamicArrayImpl &other) {
        resize_(other.size());
        memcpy(m_packets, other.m_packets,
               m_packets_allocated * sizeof(Packet));
        return derived();
    }

    DynamicArrayImpl &operator=(DynamicArrayImpl &&value) {
        m_packets = value.m_packets;
        m_packets_allocated = value.m_packets_allocated;
        m_size = value.m_size;
        value.m_packets_allocated = value.m_size = 0;
        value.m_packets = nullptr;
        return derived();
    }

    //! @}
    // -----------------------------------------------------------------------

    void set_slices_(size_t size) { resize_(size); }

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
        using Index = Array<uint32_t, Packet::Size>;
        bool scalar_flag = m_size == 1;
        Value scalar = scalar_flag ? m_packets[0].coeff(0) : Value();

        if (size <= m_packets_allocated * PacketSize) {
            size_t packets = (size + PacketSize - 1) / PacketSize;
            m_size = size;

            if (scalar_flag && size > 1) {
                /* Broadcast scalars to the entire array */
                Packet p(scalar);
                for (size_t i = 0; i < packets; ++i)
                    store(m_packets + i, p);
            }

            uint32_t remainder = (uint32_t) (size % PacketSize);
            if (remainder > 0) {
                /* There is a partially used packet at the end */
                size_t idx = m_packets_allocated - 1;
                auto keep_mask = index_sequence<Index>() < Index(remainder);
                store(m_packets + idx, load<Packet>(m_packets + idx) & keep_mask);
            }

            return;
        }

        if (m_packets_allocated > 0)
            dealloc(m_packets);

        m_packets_allocated = (size + PacketSize - 1) / PacketSize;
        m_packets = enoki::alloc<Packet>(m_packets_allocated);
        m_size = size;

        if (scalar_flag && size > 1) {
            Packet p(scalar);
            for (size_t i = 0; i < m_packets_allocated; ++i)
                store(m_packets + i, p);
        } else {
            #if !defined(NDEBUG)
                for (size_t i = 0; i < m_packets_allocated; ++i)
                    new (&m_packets[i]) Packet();
            #endif
        }

        uint32_t remainder = (uint32_t) (size % PacketSize);
        if (remainder > 0) {
            /* There is a partially used packet at the end */
            size_t idx = m_packets_allocated - 1;
            auto keep_mask = index_sequence<Index>() < Index(remainder);
            store(m_packets + idx, load<Packet>(m_packets + idx) & keep_mask);
        }
    }

protected:
    Packet *m_packets;
    size_t m_packets_allocated = 0;
    size_t m_size = 0;
};

template <typename Type_>
struct DynamicArray : DynamicArrayImpl<Type_, DynamicArray<Type_>> {
    using Base = DynamicArrayImpl<Type_, DynamicArray<Type_>>;
    using Base::Base;
    using Base::operator=;

    template <typename T>
    using ReplaceType = DynamicArray<T>;
};

NAMESPACE_BEGIN(detail)

/// Vectorized inner loop (void return value)
template <typename Func, typename... Args, size_t... Index>
ENOKI_INLINE void vectorize_inner_1(std::index_sequence<Index...>, Func &&f,
                                    size_t packet_count, Args&&... args) {
    ENOKI_NOUNROLL ENOKI_IVDEP for (size_t i = 0; i < packet_count; ++i)
        f(packet(args, i)...);
}

/// Vectorized inner loop (non-void return value)
template <typename Func, typename Out, typename... Args, size_t... Index>
ENOKI_INLINE void vectorize_inner_2(std::index_sequence<Index...>, Func &&f,
                                    size_t packet_count, Out&& out, Args&&... args) {
    ENOKI_NOUNROLL ENOKI_IVDEP for (size_t i = 0; i < packet_count; ++i)
        packet(out, i) = f(packet(args, i)...);
}

template <typename T> struct mutable_ref { using type = std::add_lvalue_reference_t<T>; };
template <typename T> struct mutable_ref<const T &> { using type = T &; };

template <typename T>
using mutable_ref_t = typename mutable_ref<T>::type;

template <bool Check, bool Resize, typename Return, typename Func, typename... Args,
    std::enable_if_t<std::is_void<Return>::value, int> = 0>
ENOKI_INLINE void vectorize(Func&& f, Args&&... args) {
    size_t packet_count = 0;

    bool unused[] = { (
        (packet_count = !is_dynamic_nested<Args>::value
                        ? packet_count : (Resize ? std::max(packet_count, packets(args))
                                                 : packets(args))),
        false)... };

    (void) unused;

    if (Check || Resize) {
        size_t slice_count = 0;

        bool unused2[] = { (
            (slice_count = !is_dynamic_nested<Args>::value
                            ? slice_count : (Resize ? std::max(slice_count, slices(args))
                                                     : slices(args))),
            false)... };

        (void) unused2;

        size_t status[] = { (!is_dynamic_nested<Args>::value ||
                            ((slice_count != 1 && slices(args) == 1 && Resize) ?
                                (set_slices((mutable_ref_t<decltype(args)>) args, slice_count), true) :
                                (slices(args) == slice_count)))... };

        for (bool flag : status)
            if (!flag)
                throw std::length_error("vectorize(): vector arguments have incompatible lengths");
    }

    vectorize_inner_1(std::make_index_sequence<sizeof...(Args)>(),
                      std::forward<Func>(f), packet_count, ref_wrap(args)...);
}

template <bool Check, bool Resize, typename Return, typename Func, typename... Args,
    std::enable_if_t<!std::is_void<Return>::value, int> = 0>
ENOKI_INLINE auto vectorize(Func&& f, Args&&... args) {
    size_t packet_count = 0, slice_count = 0;

    bool unused[] = { (
        (packet_count = !is_dynamic_nested<Args>::value
                        ? packet_count : (Resize ? std::max(packet_count, packets(args))
                                                 : packets(args))),
        false)... };

    bool unused2[] = { (
        (slice_count = !is_dynamic_nested<Args>::value
                        ? slice_count : (Resize ? std::max(slice_count, slices(args))
                                                 : slices(args))),
        false)... };

    (void) unused;
    (void) unused2;

    make_dynamic_t<Return> out;
    set_slices(out, slice_count);

    if (Check || Resize) {
        size_t status[] = { (!is_dynamic_nested<Args>::value ||
                            ((slice_count != 1 && slices(args) == 1 && Resize) ?
                                (set_slices((mutable_ref_t<decltype(args)>) args, slice_count), true) :
                                (slices(args) == slice_count)))... };

        for (bool flag : status)
            if (!flag)
                throw std::length_error("vectorize(): vector arguments have incompatible lengths");
    }

    vectorize_inner_2(std::make_index_sequence<sizeof...(Args)>(),
                      std::forward<Func>(f), packet_count, ref_wrap(out),
                      ref_wrap(args)...);

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
    return detail::vectorize<Check, false, Return>(std::forward<Func>(f), args...);
}

template <typename Func, typename... Args>
ENOKI_INLINE auto vectorize_safe(Func&& f, Args&&... args) {
    using Return = decltype(f(packet(args, 0)...));
    return detail::vectorize<true, true, Return>(std::forward<Func>(f), args...);
}

NAMESPACE_END(enoki)
