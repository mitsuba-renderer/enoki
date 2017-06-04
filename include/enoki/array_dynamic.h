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

    static ENOKI_INLINE size_t dynamic_size(const T &value) { return enoki::dynamic_size(value.x()); }
    static ENOKI_INLINE size_t packets(const T& value) { return enoki::packets(value.x()); }

    static ENOKI_INLINE void dynamic_resize(T &value, size_t size) {
        for (size_t i = 0; i < Size; ++i)
            enoki::dynamic_resize(value.coeff(i), size);
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
    static ENOKI_INLINE auto ref_wrap(T2&& value) {
        return ref_wrap(value, std::make_index_sequence<Size>());
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
    static ENOKI_INLINE auto ref_wrap(T2&& value, std::index_sequence<Index...>) {
        using Entry = decltype(enoki::ref_wrap(value.coeff(0)));
        using Return = typename T::template ReplaceType<Entry>;
        return Return(enoki::ref_wrap(value.coeff(Index))...);
    }
};

template <typename T> struct dynamic_support<T, enable_if_dynamic_array_t<T>> {
    static constexpr bool is_dynamic_nested = true;
    using dynamic_t = T;

    static ENOKI_INLINE size_t dynamic_size(const T &value) { return value.dynamic_size_(); }
    static ENOKI_INLINE size_t packets(const T& value) { return value.packets_(); }

    static ENOKI_INLINE void dynamic_resize(T &value, size_t size) {
        value.dynamic_resize_(size);
    }

    template <typename T2> static ENOKI_INLINE decltype(auto) packet(T2 &&value, size_t i) {
        return value.packet_(i);
    }

    template <typename T2> static ENOKI_INLINE decltype(auto) slice(T2 &&value, size_t i) {
        return value.slice_(i);
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
    using Mask                                = DynamicArray<mask_t<Packet_>>;
    static constexpr size_t       PacketSize  = Packet::Size;
    static constexpr bool         Approx      = Packet::Approx;
    static constexpr RoundingMode Mode        = Packet::Mode;
    static constexpr bool IsMask              = Packet::IsMask;
};

template <typename Packet>
struct DynamicArrayReference : DynamicArrayBase<Packet, DynamicArrayReference<Packet>> {
    using Base = DynamicArrayBase<Packet, DynamicArrayReference<Packet>>;

    DynamicArrayReference(Packet *packets) : m_packets(packets) { }

    ENOKI_INLINE Packet &packet_(size_t i) {
        return ((Packet *) ENOKI_ASSUME_ALIGNED(m_packets))[i];
    }

    ENOKI_INLINE const Packet &packet_(size_t i) const {
        return ((const Packet *) ENOKI_ASSUME_ALIGNED(m_packets))[i];
    }

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

    ENOKI_INLINE size_t dynamic_size_() const { return m_size; }

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

    void dynamic_resize_(size_t size) { resize_(size); }

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

        if (m_packets_allocated > 0)
            dealloc(m_packets);

        m_packets_allocated = (size + PacketSize - 1) / PacketSize;
        m_packets = enoki::alloc<Packet>(m_packets_allocated);
        m_size = size;

        #if !defined(NDEBUG)
            for (size_t i = 0; i < m_packets_allocated; ++i)
                new (&m_packets[i]) Packet();
        #endif

        /* Clear unused entries */
        size_t used = sizeof(Value) * size;
        size_t allocated =
            sizeof(Value) * m_packets_allocated * PacketSize;
        memset((uint8_t *) m_packets + used, 0, allocated - used);
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
};

NAMESPACE_BEGIN(detail)

template <typename T> struct arg_type {
    using type = std::conditional_t<std::is_const<std::remove_reference_t<T>>::value, expr_t<T>, T>;
};

template <typename T>
using arg_type_t = typename arg_type<T>::type;

/// Vectorized inner loop (void return value)
template <typename Func, typename... Args, size_t... Index>
ENOKI_INLINE void vectorize_inner_1(std::index_sequence<Index...>, Func &&f,
                                    size_t packet_count, Args&&... args) {
    ENOKI_NOUNROLL ENOKI_IVDEP for (size_t i = 0; i < packet_count; ++i)
        f((arg_type_t<decltype(packet(args, i))>) packet(args, i)...);
}

/// Vectorized inner loop (non-void return value)
template <typename Func, typename Out, typename... Args, size_t... Index>
ENOKI_INLINE void vectorize_inner_2(std::index_sequence<Index...>, Func &&f,
                                    size_t packet_count, Out&& out, Args&&... args) {
    ENOKI_NOUNROLL ENOKI_IVDEP for (size_t i = 0; i < packet_count; ++i)
        packet(out, i) = f(((arg_type_t<decltype(packet(args, i))>) packet(args, i))...);
}

template <bool Check, typename Return, typename Func, typename... Args,
    std::enable_if_t<std::is_void<Return>::value, int> = 0>
ENOKI_INLINE void vectorize(Func&& f, Args&&... args) {
    size_t packet_count = 0;

    bool unused[] = { (
        (packet_count = (is_dynamic_nested<Args>::value ? packets(args) : packet_count)),
        false)... };
    (void) unused;

    if (Check) {
        size_t dsize = 0;
        bool unused2[] = { (
            (dsize = (is_dynamic_nested<Args>::value ? dynamic_size(args) : dsize)),
            false)... };
        (void) unused2;

        bool status[] = { (!is_dynamic_nested<Args>::value ||
                           (dynamic_size(args) == dsize))... };
        for (bool flag : status)
            if (!flag)
                throw std::length_error("vectorize(): vector arguments have incompatible lengths");
    }

    vectorize_inner_1(std::make_index_sequence<sizeof...(Args)>(),
                      std::forward<Func>(f), packet_count, ref_wrap(args)...);
}

template <bool Check, typename Return, typename Func, typename... Args,
    std::enable_if_t<!std::is_void<Return>::value, int> = 0>
ENOKI_INLINE auto vectorize(Func&& f, Args&&... args) {
    size_t packet_count = 0, dsize = 0;

    bool unused[] = { (
        (packet_count = (is_dynamic_nested<Args>::value ? packets(args) : packet_count)),
        false)... };

    bool unused2[] = { (
        (dsize = (is_dynamic_nested<Args>::value ? dynamic_size(args) : dsize)),
        false)... };

    (void) unused;
    (void) unused2;

    make_dynamic_t<Return> out;
    dynamic_resize(out, dsize);

    if (Check) {
        bool status[] = { (!is_dynamic_nested<Args>::value ||
                           (dynamic_size(args) == dsize))... };
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
    return detail::vectorize<Check, Return>(std::forward<Func>(f), args...);
}

template <typename Func, typename... Args>
ENOKI_INLINE auto vectorize_safe(Func&& f, Args&&... args) {
    using Return = decltype(f(packet(args, 0)...));
    return detail::vectorize<true, Return>(std::forward<Func>(f), args...);
}

// The main idea of this macro is borrowed from https://github.com/swansontec/map-macro
// (C) William Swanson, Paul Fultz
#define ENOKI_EVAL_0(...) __VA_ARGS__
#define ENOKI_EVAL_1(...) ENOKI_EVAL_0(ENOKI_EVAL_0(ENOKI_EVAL_0(__VA_ARGS__)))
#define ENOKI_EVAL_2(...) ENOKI_EVAL_1(ENOKI_EVAL_1(ENOKI_EVAL_1(__VA_ARGS__)))
#define ENOKI_EVAL_3(...) ENOKI_EVAL_2(ENOKI_EVAL_2(ENOKI_EVAL_2(__VA_ARGS__)))
#define ENOKI_EVAL_4(...) ENOKI_EVAL_3(ENOKI_EVAL_3(ENOKI_EVAL_3(__VA_ARGS__)))
#define ENOKI_EVAL(...)   ENOKI_EVAL_4(ENOKI_EVAL_4(ENOKI_EVAL_4(__VA_ARGS__)))
#define ENOKI_MAP_END(...)
#define ENOKI_MAP_OUT
#define ENOKI_MAP_COMMA ,
#define ENOKI_MAP_GET_END() 0, ENOKI_MAP_END
#define ENOKI_MAP_NEXT_0(test, next, ...) next ENOKI_MAP_OUT
#define ENOKI_MAP_NEXT_1(test, next) ENOKI_MAP_NEXT_0(test, next, 0)
#define ENOKI_MAP_NEXT(test, next) ENOKI_MAP_NEXT_1(ENOKI_MAP_GET_END test, next)
#define ENOKI_EXTRACT_0(next, ...) next

#if defined(_MSC_VER) // MSVC is not as eager to expand macros, hence this workaround
#define ENOKI_MAP_LIST_NEXT_1(test, next) \
    ENOKI_EVAL_0(ENOKI_MAP_NEXT_0(test, ENOKI_MAP_COMMA next, 0))
#else
#define ENOKI_MAP_LIST_NEXT_1(test, next) \
    ENOKI_MAP_NEXT_0(test, ENOKI_MAP_COMMA next, 0)
#endif

#define ENOKI_MAP_LIST_NEXT(test, next) \
    ENOKI_MAP_LIST_NEXT_1 (ENOKI_MAP_GET_END test, next)

#define ENOKI_MAP_LIST_1_0(f, v, x, peek, ...) \
    f(v.x) ENOKI_MAP_LIST_NEXT(peek, ENOKI_MAP_LIST_1_1)(f, v, peek, __VA_ARGS__)
#define ENOKI_MAP_LIST_1_1(f, v, x, peek, ...) \
    f(v.x) ENOKI_MAP_LIST_NEXT(peek, ENOKI_MAP_LIST_1_0)(f, v, peek, __VA_ARGS__)

#define ENOKI_MAP_LIST_2_0(f, v, t, x, peek, ...) \
    f(v.x, t) ENOKI_MAP_LIST_NEXT(peek, ENOKI_MAP_LIST_2_1)(f, v, t, peek, __VA_ARGS__)
#define ENOKI_MAP_LIST_2_1(f, v, t, x, peek, ...) \
    f(v.x, t) ENOKI_MAP_LIST_NEXT(peek, ENOKI_MAP_LIST_2_0)(f, v, t, peek, __VA_ARGS__)

// ENOKI_MAP_LIST_1(f, v, a1, a2, ...) expands to f(v.a1), f(v.a2), ...
#define ENOKI_MAP_LIST_1(f, v, ...) \
    ENOKI_EVAL(ENOKI_MAP_LIST_1_0(f, v, __VA_ARGS__, (), 0))

// ENOKI_MAP_LIST_2(f, v, t, a1, a2, ...) expands to f(v.a1, t), f(v.a2, t), ...
#define ENOKI_MAP_LIST_2(f, v, t, ...) \
    ENOKI_EVAL(ENOKI_MAP_LIST_2_0(f, v, t, __VA_ARGS__, (), 0))

#define ENOKI_DYNAMIC_SUPPORT(Name, ...)                                       \
    NAMESPACE_BEGIN(enoki)                                                     \
    template <typename T> struct dynamic_support<Name<T>> {                    \
        static constexpr bool is_dynamic_nested =                              \
            enoki::is_dynamic_nested<T>::value;                                \
        using dynamic_t = Name<enoki::make_dynamic_t<T>>;                      \
        using Value     = Name<T>;                                             \
        static ENOKI_INLINE size_t dynamic_size(const Value &value) {          \
            return enoki::dynamic_size(value.ENOKI_EXTRACT_0(__VA_ARGS__));    \
        }                                                                      \
        static ENOKI_INLINE size_t packets(const Value &value) {               \
            return enoki::packets(value.ENOKI_EXTRACT_0(__VA_ARGS__));         \
        }                                                                      \
        static ENOKI_INLINE void dynamic_resize(Value &value, size_t size) {   \
            ENOKI_MAP_LIST_2(enoki::dynamic_resize, value, size, __VA_ARGS__); \
        }                                                                      \
        template <typename T2> static ENOKI_INLINE auto ref_wrap(T2 &&value) { \
            using Type = decltype(enoki::ref_wrap(std::declval<T>()));         \
            return Name<Type>(                                                 \
                ENOKI_MAP_LIST_1(enoki::ref_wrap, value, __VA_ARGS__));        \
        }                                                                      \
        template <typename T2>                                                 \
        static ENOKI_INLINE auto packet(T2 &&value, size_t index) {            \
            using Type = decltype(enoki::packet(std::declval<T>(), index));    \
            return Name<Type>(                                                 \
                ENOKI_MAP_LIST_2(enoki::packet, value, index, __VA_ARGS__));   \
        }                                                                      \
        template <typename T2>                                                 \
        static ENOKI_INLINE auto slice(T2 &&value, size_t index) {             \
            using Type = decltype(enoki::slice(std::declval<T>(), index));     \
            return Name<Type>(                                                 \
                ENOKI_MAP_LIST_2(enoki::slice, value, index, __VA_ARGS__));    \
        }                                                                      \
    };                                                                         \
    NAMESPACE_END(enoki)

NAMESPACE_END(enoki)
