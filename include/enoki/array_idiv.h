/*
    enoki/array_idiv.h -- fast precomputed integer division by constants based
    on libdivide (https://github.com/ridiculousfish/libdivide)

    Copyright (C) 2010 ridiculous_fish

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.

    libdivide@ridiculousfish.com

*/

#pragma once

#include <enoki/array_generic.h>

NAMESPACE_BEGIN(enoki)

// -----------------------------------------------------------------------
//! @{ \name Precomputation for division by integer constants
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)

/// 64 by 32 bit integer division
inline uint32_t div_wide(uint32_t u1, uint32_t u0, uint32_t v, uint32_t *r) {
#if 0 // defined(__GNUC__)
    /* This is very efficient but cannot be evaluated at compile time */
    uint32_t result;
    __asm__("divl %[v]"
            : "=a"(result), "=d"(*r)
            : [v] "r"(v), "a"(u0), "d"(u1));
    return result;
#else
    uint64_t n = (((uint64_t) u1) << 32) | u0;
    uint32_t result = (uint32_t)(n / v);
    *r = (uint32_t)(n - result * (uint64_t) v);
    return result;
#endif
}

/// 128 by 64 bit integer division
inline uint64_t div_wide(uint64_t u1, uint64_t u0, uint64_t v, uint64_t *r) {
#if 0// defined(__GNUC__)
    /* This is very efficient but cannot be evaluated at compile time */
    uint64_t result;
    __asm__("divq %[v]"
            : "=a"(result), "=d"(*r)
            : [v] "r"(v), "a"(u0), "d"(u1));
    return result;
#elif defined(__SIZEOF_INT128__)
    __uint128_t n = (((__uint128_t) u1) << 64) | u0;
    uint64_t result = (uint64_t)(n / v);
    *r = (uint64_t)(n - result * (__uint128_t) v);
    return result;
#else
    // Code taken from Hacker's Delight:
    // http://www.hackersdelight.org/HDcode/divlu.c.
    // License permits inclusion here per:
    // http://www.hackersdelight.org/permissions.htm

    const uint64_t b = (1ULL << 32); // Number base (16 bits).
    uint64_t un1, un0,  // Norm. dividend LSD's.
    vn1, vn0,           // Norm. divisor digits.
    q1, q0,             // Quotient digits.
    un64, un21, un10,   // Dividend digit pairs.
    rhat;               // A remainder.
    int s;              // Shift amount for norm.

    if (u1 >= v) { // overflow
        *r = (uint64_t) -1;
        return (uint64_t) -1;
    }

    // count leading zeros
    s = (int) (63 - log2i(v)); // 0 <= s <= 63.
    if (s > 0) {
        v = v << s;         // Normalize divisor.
        un64 = (u1 << s) | ((u0 >> (64 - s)) & uint64_t(-s >> 31));
        un10 = u0 << s;     // Shift dividend left.
    } else {
        // Avoid undefined behavior.
        un64 = u1 | u0;
        un10 = u0;
    }

    vn1 = v >> 32;            // Break divisor up into
    vn0 = v & 0xFFFFFFFF;     // two 32-bit digits.

    un1 = un10 >> 32;         // Break right half of
    un0 = un10 & 0xFFFFFFFF;  // dividend into two digits.

    q1 = un64/vn1;            // Compute the first
    rhat = un64 - q1*vn1;     // quotient digit, q1.

again1:
    if (q1 >= b || q1*vn0 > b*rhat + un1) {
        q1 = q1 - 1;
        rhat = rhat + vn1;
        if (rhat < b)
            goto again1;
    }

    un21 = un64*b + un1 - q1*v;  // Multiply and subtract.

    q0 = un21/vn1;            // Compute the second
    rhat = un21 - q0*vn1;     // quotient digit, q0.

again2:
    if (q0 >= b || q0 * vn0 > b * rhat + un0) {
        q0 = q0 - 1;
        rhat = rhat + vn1;
        if (rhat < b)
            goto again2;
    }

    *r = (un21*b + un0 - q0*v) >> s;
    return q1*b + q0;
#endif
}

NAMESPACE_END(detail)

template <typename T, typename SFINAE = void> struct divisor { };

template <typename T> struct divisor<T, std::enable_if_t<!std::is_signed<T>::value>> {
    T multiplier;
    uint8_t shift;

    divisor() { }

    divisor(T d) {
        /* Division by 1 is not supported by the precomputation-based approach */
        assert(d != 1);
        shift = (uint8_t) log2i(d);

        if ((d & (d - 1)) == 0) {
            /* Power of two */
            multiplier = 0;
            shift--;
        } else {
            /* General case */
            T rem;
            multiplier = detail::div_wide(T(1) << shift, 0, d, &rem) * 2 + 1;
            assert(rem > 0 && rem < d);

            T twice_rem = rem + rem;
            if (twice_rem >= d || twice_rem < rem)
                multiplier += 1;
        }
    }

    template <typename T2>
    ENOKI_INLINE auto operator()(const T2 &value) const {
        using Expr = decltype(value + value);
        auto q = mulhi(Expr(multiplier), value);
        auto t = sri<1>(value - q) + q;
        return t >> shift;
    }
} ENOKI_PACK;

#if defined(_MSC_VER)
#  pragma pack(push)
#  pragma pack(1)
#endif

template <typename T> struct divisor<T, std::enable_if_t<std::is_signed<T>::value>> {
    using U = std::make_unsigned_t<T>;

    T multiplier;
    uint8_t shift;

    divisor() { }

    divisor(T d) {
        /* Division by +/-1 is not supported by the precomputation-based approach */
        assert(d != 1 && d != -1);

        U ud = (U) d, ad = d < 0 ? (U) -ud : ud;
        shift = (uint8_t) log2i(ad);

        if ((ad & (ad - 1)) == 0) {
            /* Power of two */
            multiplier = 0;
        } else {
            /* General case */
            U rem;
            multiplier = T(detail::div_wide(U(1) << (shift - 1), 0, ad, &rem) * 2 + 1);

            U twice_rem = rem + rem;
            if (twice_rem >= ad || twice_rem < rem)
                multiplier += 1;
        }
        if (d < 0)
            shift |= 0x80;
    }

    template <typename T2>
    ENOKI_INLINE auto operator()(const T2 &value) const {
        using Expr = decltype(value + value);
        uint8_t shift_ = shift & 0x3f;
        Expr sign(int8_t(shift) >> 7);

        auto q = mulhi(Expr(multiplier), value) + value;
        auto q_sign = sri<sizeof(T) * 8 - 1>(q);
        q += q_sign & ((T(1) << shift_) - (multiplier == 0 ? 1 : 0));

        return ((q >> shift_) ^ sign) - sign;
    }
} ENOKI_PACK;

#if defined(_MSC_VER)
#  pragma pack(pop)
#endif

template <typename Type, size_t Size, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<!std::is_floating_point<base_scalar_t<Type>>::value, int> = 0>
ENOKI_INLINE auto
operator/(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a, divisor<base_scalar_t<Type>> div) {
    return div(a.derived());
}

template <typename Type, size_t Size, bool Approx, RoundingMode Mode, typename Derived,
          std::enable_if_t<!std::is_floating_point<base_scalar_t<Type>>::value, int> = 0>
ENOKI_INLINE auto
operator%(const StaticArrayBase<Type, Size, Approx, Mode, Derived> &a, base_scalar_t<Type> v) {
    return a - divisor<base_scalar_t<Type>>(v)(a) * Derived(v);
}

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(enoki)
