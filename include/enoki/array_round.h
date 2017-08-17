/*
    enoki/array_round.h -- Fallback for nonstandard rounding modes

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using ENOKI instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array_generic.h"

NAMESPACE_BEGIN(enoki)

#if defined(ENOKI_X86_64) || defined(ENOKI_X86_32)
/// RAII wrapper that saves and restores the FP Control/Status Register
template <RoundingMode Mode> struct set_rounding_mode {
    set_rounding_mode() : value(_mm_getcsr()) {
        unsigned int csr = value & ~(unsigned int) _MM_ROUND_MASK;
        switch (Mode) {
            case RoundingMode::Nearest: csr |= _MM_ROUND_NEAREST; break;
            case RoundingMode::Down: csr |= _MM_ROUND_DOWN; break;
            case RoundingMode::Up: csr |= _MM_ROUND_UP; break;
            case RoundingMode::Zero: csr |= _MM_ROUND_TOWARD_ZERO; break;
        }
        _mm_setcsr(csr);
    }

    ~set_rounding_mode() {
        _mm_setcsr(value);
    }

    unsigned int value;
};
#else
template <RoundingMode Mode> struct set_rounding_mode {
    // Don't know how to change rounding modes on this platform :(
};
#endif

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived>
struct StaticArrayImpl<Value_, Size_, Approx_, Mode_, Derived,
    std::enable_if_t<detail::rounding_fallback<Value_, Size_, Mode_>::value>>
    : StaticArrayImpl<Value_, Size_, Approx_, RoundingMode::Default, Derived> {

    using Base = StaticArrayImpl<Value_, Size_, Approx_, RoundingMode::Default, Derived>;

    using Expr = std::conditional_t<std::is_same<expr_t<Value_>, Value_>::value, Derived,
                                    Array<expr_t<Value_>, Size_, Approx_, Mode_>>;

    /// Rounding mode of arithmetic operations
    static constexpr RoundingMode Mode = Mode_;

    StaticArrayImpl() { }

    template <typename... Args>
    ENOKI_NOINLINE StaticArrayImpl(Args&&... args) {
        set_rounding_mode<Mode_> mode; (void) mode;
        *this = Base(args...);
    }

    template <typename... Args>
    ENOKI_NOINLINE StaticArrayImpl& operator=(Args&&... args) {
        set_rounding_mode<Mode_> mode; (void) mode;
        Base::operator=(args...);
        return *this;
    }

    ENOKI_NOINLINE auto add_(const Derived &a) const {
        set_rounding_mode<Mode_> mode; (void) mode;
        return Base::add_(a);
    }

    ENOKI_NOINLINE auto sub_(const Derived &a) const {
        set_rounding_mode<Mode_> mode; (void) mode;
        return Base::sub_(a);
    }

    ENOKI_NOINLINE auto mul_(const Derived &a) const {
        set_rounding_mode<Mode_> mode; (void) mode;
        return Base::mul_(a);
    }

    ENOKI_NOINLINE auto div_(const Derived &a) const {
        set_rounding_mode<Mode_> mode; (void) mode;
        return Base::div_(a);
    }

    ENOKI_NOINLINE auto sqrt_() const {
        set_rounding_mode<Mode_> mode; (void) mode;
        return Base::sqrt_();
    }

    ENOKI_NOINLINE auto fmadd_(const Derived &b, const Derived &c) const {
        set_rounding_mode<Mode_> mode; (void) mode;
        return Base::fmadd_(b, c);
    }

    ENOKI_NOINLINE auto fmsub_(const Derived &b, const Derived &c) const {
        set_rounding_mode<Mode_> mode; (void) mode;
        return Base::fmsub_(b, c);
    }
};

NAMESPACE_END(enoki)
