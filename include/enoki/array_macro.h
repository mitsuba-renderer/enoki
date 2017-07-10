/*
    enoki/array_macro.h -- Code generation macros for custom data structures

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

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
#define ENOKI_MAP_EXPR_NEXT_1(test, next) \
    ENOKI_EVAL_0(ENOKI_MAP_NEXT_0(test, ENOKI_MAP_COMMA next, 0))
#define ENOKI_MAP_STMT_NEXT_1(test, next) \
    ENOKI_EVAL_0(ENOKI_MAP_NEXT_0(test, next, 0))
#else
#define ENOKI_MAP_EXPR_NEXT_1(test, next) \
    ENOKI_MAP_NEXT_0(test, ENOKI_MAP_COMMA next, 0)
#define ENOKI_MAP_STMT_NEXT_1(test, next) \
    ENOKI_MAP_NEXT_0(test, next, 0)
#endif

#define ENOKI_MAP_EXPR_NEXT(test, next) \
    ENOKI_MAP_EXPR_NEXT_1 (ENOKI_MAP_GET_END test, next)
#define ENOKI_MAP_STMT_NEXT(test, next) \
    ENOKI_MAP_STMT_NEXT_1 (ENOKI_MAP_GET_END test, next)

#define ENOKI_MAP_EXPR_DECL_0(x, peek, ...) \
    const decltype(x) &x ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_DECL_1)(peek, __VA_ARGS__)
#define ENOKI_MAP_EXPR_DECL_1(x, peek, ...) \
    const decltype(x) &x ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_DECL_0)(peek, __VA_ARGS__)

#define ENOKI_MAP_EXPR_COPY_0(x, peek, ...) \
    x(x) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_COPY_1)(peek, __VA_ARGS__)
#define ENOKI_MAP_EXPR_COPY_1(x, peek, ...) \
    x(x) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_COPY_0)(peek, __VA_ARGS__)

#define ENOKI_MAP_EXPR_COPY_V_0(v, x, peek, ...) \
    x(v.x) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_COPY_V_1)(v, peek, __VA_ARGS__)
#define ENOKI_MAP_EXPR_COPY_V_1(v, x, peek, ...) \
    x(v.x) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_COPY_V_0)(v, peek, __VA_ARGS__)

#define ENOKI_MAP_EXPR_MOVE_V_0(v, x, peek, ...) \
    x(std::move(v.x)) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_MOVE_V_1)(v, peek, __VA_ARGS__)
#define ENOKI_MAP_EXPR_MOVE_V_1(v, x, peek, ...) \
    x(std::move(v.x)) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_MOVE_V_0)(v, peek, __VA_ARGS__)

#define ENOKI_MAP_STMT_ASSIGN_0(v, x, peek, ...)                               \
    this->x = v.x;                                                             \
    ENOKI_MAP_STMT_NEXT(peek, ENOKI_MAP_STMT_ASSIGN_1)(v, peek, __VA_ARGS__)
#define ENOKI_MAP_STMT_ASSIGN_1(v, x, peek, ...)                               \
    this->x = v.x;                                                             \
    ENOKI_MAP_STMT_NEXT(peek, ENOKI_MAP_STMT_ASSIGN_0)(v, peek, __VA_ARGS__)

#define ENOKI_MAP_STMT_MOVE_0(v, x, peek, ...)                                 \
    this->x = std::move(v.x);                                                  \
    ENOKI_MAP_STMT_NEXT(peek, ENOKI_MAP_STMT_MOVE_1)(v, peek, __VA_ARGS__)
#define ENOKI_MAP_STMT_MOVE_1(v, x, peek, ...)                                 \
    this->x = std::move(v.x);                                                  \
    ENOKI_MAP_STMT_NEXT(peek, ENOKI_MAP_STMT_MOVE_0)(v, peek, __VA_ARGS__)

#define ENOKI_MAP_EXPR_F1_0(f, v, x, peek, ...) \
    f(v.x) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_F1_1)(f, v, peek, __VA_ARGS__)
#define ENOKI_MAP_EXPR_F1_1(f, v, x, peek, ...) \
    f(v.x) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_F1_0)(f, v, peek, __VA_ARGS__)

#define ENOKI_MAP_EXPR_F2_0(f, v, t, x, peek, ...) \
    f(v.x, t) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_F2_1)(f, v, t, peek, __VA_ARGS__)
#define ENOKI_MAP_EXPR_F2_1(f, v, t, x, peek, ...) \
    f(v.x, t) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_F2_0)(f, v, t, peek, __VA_ARGS__)

#define ENOKI_MAP_EXPR_F3_0(f, m, v, t, x, peek, ...) \
    f(m.x, v.x, t) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_F3_1)(f, m, v, t, peek, __VA_ARGS__)
#define ENOKI_MAP_EXPR_F3_1(f, m, v, t, x, peek, ...) \
    f(m.x, v.x, t) ENOKI_MAP_EXPR_NEXT(peek, ENOKI_MAP_EXPR_F3_0)(f, m, v, t, peek, __VA_ARGS__)

// ENOKI_MAP_EXPR_DECL(a1, a2, ...) expands to const decltype(a1) &a1, ...
#define ENOKI_MAP_EXPR_DECL(...) \
    ENOKI_EVAL(ENOKI_MAP_EXPR_DECL_0(__VA_ARGS__, (), 0))

// ENOKI_MAP_EXPR_COPY(a1, a2, ...) expands to a1(a1), a2(a2), ...
#define ENOKI_MAP_EXPR_COPY(...) \
    ENOKI_EVAL(ENOKI_MAP_EXPR_COPY_0(__VA_ARGS__, (), 0))

// ENOKI_MAP_EXPR_COPY_V(v, a1, a2, ...) expands to a1(v.a1), a2(v.a2), ...
#define ENOKI_MAP_EXPR_COPY_V(v, ...) \
    ENOKI_EVAL(ENOKI_MAP_EXPR_COPY_V_0(v, __VA_ARGS__, (), 0))

// ENOKI_MAP_EXPR_MOVE_V(v, a1, a2, ...) expands to a1(std::move(v.a1)), a2(std::move(v.a2)), ...
#define ENOKI_MAP_EXPR_MOVE_V(v, ...) \
    ENOKI_EVAL(ENOKI_MAP_EXPR_MOVE_V_0(v, __VA_ARGS__, (), 0))

// ENOKI_MAP_STMT_ASSIGN(v, a1, a2, ...) expands to this->a1 = v.a1; ..
#define ENOKI_MAP_STMT_ASSIGN(v, ...) \
    ENOKI_EVAL(ENOKI_MAP_STMT_ASSIGN_0(v, __VA_ARGS__, (), 0))

// ENOKI_MAP_STMT_MOVE(v, a1, a2, ...) expands to this->a1 = std::move(v.a1); ..
#define ENOKI_MAP_STMT_MOVE(v, ...) \
    ENOKI_EVAL(ENOKI_MAP_STMT_MOVE_0(v, __VA_ARGS__, (), 0))

// ENOKI_MAP_EXPR_F1(f, v, a1, a2, ...) expands to f(v.a1), f(v.a2), ...
#define ENOKI_MAP_EXPR_F1(f, v, ...) \
    ENOKI_EVAL(ENOKI_MAP_EXPR_F1_0(f, v, __VA_ARGS__, (), 0))

// ENOKI_MAP_EXPR_F2(f, v, t, a1, a2, ...) expands to f(v.a1, t), f(v.a2, t), ...
#define ENOKI_MAP_EXPR_F2(f, v, t, ...) \
    ENOKI_EVAL(ENOKI_MAP_EXPR_F2_0(f, v, t, __VA_ARGS__, (), 0))

// ENOKI_MAP_EXPR_F3(f, m, v, t, a1, a2, ...) expands to f(m.a1, v.a1, t), f(m.a2, v.a2, t), ...
#define ENOKI_MAP_EXPR_F3(f, v, t, ...) \
    ENOKI_EVAL(ENOKI_MAP_EXPR_F3_0(f, v, t, __VA_ARGS__, (), 0))


#define ENOKI_STRUCT(Struct, ...)                                              \
    Struct()  = default;                                                       \
    Struct(ENOKI_MAP_EXPR_DECL(__VA_ARGS__))                                   \
        : ENOKI_MAP_EXPR_COPY(__VA_ARGS__) {}                                  \
    template <typename... Args>                                                \
    Struct(const Struct<Args...> &value)                                       \
        : ENOKI_MAP_EXPR_COPY_V(value, __VA_ARGS__) {}                         \
    template <typename... Args>                                                \
    Struct(Struct<Args...> &&value)                                            \
        : ENOKI_MAP_EXPR_MOVE_V(value, __VA_ARGS__) {}                         \
    template <typename... Args>                                                \
    Struct &operator=(const Struct<Args...> &value) {                          \
        ENOKI_MAP_STMT_ASSIGN(value, __VA_ARGS__)                              \
        return *this;                                                          \
    }                                                                          \
    template <typename... Args>                                                \
    Struct &operator=(Struct<Args...> &&value) {                               \
        ENOKI_MAP_STMT_MOVE(value, __VA_ARGS__)                                \
        return *this;                                                          \
    }

#define ENOKI_STRUCT_DYNAMIC(Struct, ...)                                      \
    NAMESPACE_BEGIN(enoki)                                                     \
    template <typename... Args> struct dynamic_support<Struct<Args...>> {      \
        static constexpr bool is_dynamic_nested = enoki::detail::any_of<       \
            enoki::is_dynamic_nested<Args>::value...>::value;                  \
        using dynamic_t = Struct<enoki::make_dynamic_t<Args>...>;              \
        using Value = Struct<Args...>;                                         \
        static ENOKI_INLINE size_t packets(const Value &value) {               \
            return enoki::packets(value.ENOKI_EXTRACT_0(__VA_ARGS__));         \
        }                                                                      \
        static ENOKI_INLINE size_t slices(const Value &value) {                \
            return enoki::slices(value.ENOKI_EXTRACT_0(__VA_ARGS__));          \
        }                                                                      \
        static ENOKI_INLINE void set_slices(Value &value, size_t size) {       \
            ENOKI_MAP_EXPR_F2(enoki::set_slices, value, size, __VA_ARGS__);    \
        }                                                                      \
        template <typename Mem, typename Mask>                                 \
        static ENOKI_INLINE size_t compress(Mem &mem, const Value &value,      \
                                          const Mask &mask) {                  \
            return ENOKI_MAP_EXPR_F3(enoki::compress, mem, value,              \
                                     mask, __VA_ARGS__);                       \
        }                                                                      \
        template <typename T>                                                  \
        static ENOKI_INLINE auto slice(T &&value, size_t index) {              \
            constexpr static bool co_ = std::is_const<                         \
                std::remove_reference_t<T>>::value;                            \
            using Type = Struct<decltype(enoki::slice(std::declval<            \
                std::conditional_t<co_, const Args &, Args &>>(), index))...>; \
            return Type{ ENOKI_MAP_EXPR_F2(enoki::slice, value, index,         \
                                           __VA_ARGS__) };                     \
        }                                                                      \
        template <typename T>                                                  \
        static ENOKI_INLINE auto slice_ptr(T &&value, size_t index) {          \
            constexpr static bool co_ = std::is_const<                         \
                std::remove_reference_t<T>>::value;                            \
            using Type = Struct<decltype(enoki::slice_ptr(std::declval<        \
                std::conditional_t<co_, const Args &, Args &>>(), index))...>; \
            return Type{ ENOKI_MAP_EXPR_F2(enoki::slice_ptr, value, index,     \
                                           __VA_ARGS__) };                     \
        }                                                                      \
        template <typename T>                                                  \
        static ENOKI_INLINE auto packet(T &&value, size_t index) {             \
            constexpr static bool co_ = std::is_const<                         \
                std::remove_reference_t<T>>::value;                            \
            using Type = Struct<decltype(enoki::packet(std::declval<           \
                std::conditional_t<co_, const Args &, Args &>>(), index))...>; \
            return Type{ ENOKI_MAP_EXPR_F2(enoki::packet, value, index,        \
                                           __VA_ARGS__) };                     \
        }                                                                      \
        template <typename T> static ENOKI_INLINE auto ref_wrap(T &&value) {   \
            constexpr static bool co_ = std::is_const<                         \
                std::remove_reference_t<T>>::value;                            \
            using Type = Struct<decltype(enoki::ref_wrap(std::declval<         \
                std::conditional_t<co_, const Args &, Args &>>()))...>;        \
            return Type{ ENOKI_MAP_EXPR_F1(enoki::ref_wrap, value,             \
                                           __VA_ARGS__) };                     \
        }                                                                      \
    };                                                                         \
    NAMESPACE_END(enoki)
