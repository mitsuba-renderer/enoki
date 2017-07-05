/*
    enoki/array_recursive.h -- Template specialization that recursively
    instantiates Array instances with smaller sizes when the requested packet
    float array size is not directly supported by the processor's SIMD
    instructions

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

template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_,
          typename Derived>
struct StaticArrayImpl<Type_, Size_, Approx_, Mode_, Derived,
    std::enable_if_t<detail::is_recursive<Type_, Size_, Mode_>::value>>
    : StaticArrayBase<Type_, Size_, Approx_, Mode_, Derived> {

    using Base = StaticArrayBase<Type_, Size_, Approx_, Mode_, Derived>;
    using Expr = Derived;
    using typename Base::Value;
    using typename Base::Scalar;
    using typename Base::Array1;
    using typename Base::Array2;
    using Base::Size;
    using Base::Size1;
    using Base::Size2;
    using Base::operator=;

    using Mask1 = typename Array1::Mask;
    using Mask2 = typename Array2::Mask;

    using Arg = const Derived &;

    struct RMask : StaticArrayBase<Type_, Size_, false, RoundingMode::Default, RMask> {
    public:
        static constexpr size_t Size = Size_;
        static constexpr size_t ActualSize = Size_;

        static constexpr bool IsMask = true;
        using Expr = RMask;
        using Mask = RMask;

        Mask1 m1;
        Mask2 m2;

        ENOKI_INLINE RMask() : m1(), m2() { }
        ENOKI_INLINE RMask(Mask1 m1, Mask2 m2) : m1(m1), m2(m2) { }
        template <typename T, std::enable_if_t<std::is_same<T, bool>::value, int> = 0>
        ENOKI_INLINE RMask(T b) : m1(b), m2(b) { }

        /// Cast another array
        template <typename Other, enable_if_static_array_t<Other> = 0>
        ENOKI_INLINE RMask(const Other &m) : m1(low(m)), m2(high(m)) { }

        /// Reinterpret another array
        template <typename Other>
        ENOKI_INLINE RMask(const Other &m, detail::reinterpret_flag)
            : m1(reinterpret_array<Mask1>(low(m))), m2(reinterpret_array<Mask2>(high(m))) { }

        ENOKI_INLINE explicit operator Derived() const { return Derived((Array1) m1, (Array2) m2); }

        ENOKI_INLINE RMask or_ (const RMask &m) const { return RMask(m1 | m.m1, m2 | m.m2); }
        ENOKI_INLINE RMask and_(const RMask &m) const { return RMask(m1 & m.m1, m2 & m.m2); }
        ENOKI_INLINE RMask xor_(const RMask &m) const { return RMask(m1 ^ m.m1, m2 ^ m.m2); }
        ENOKI_INLINE RMask eq_ (const RMask &m) const { return RMask(eq(m1, m.m1), eq(m2, m.m2)); }
        ENOKI_INLINE RMask neq_(const RMask &m) const { return RMask(neq(m1, m.m1), neq(m2, m.m2)); }
        ENOKI_INLINE RMask not_() const { return RMask(~m1, ~m2); }

        static ENOKI_INLINE RMask select_(const RMask &m, const RMask &t, const RMask &f) {
            return Mask(select(m.m1, t.m1, f.m1),
                        select(m.m2, t.m2, f.m2));
        }

        template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
        ENOKI_INLINE bool all_() const { return all(m1 & m2); }
        template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
        ENOKI_INLINE bool all_() const { return all(m1) && all(m2); }

        template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
        ENOKI_INLINE bool any_() const { return any(m1 | m2); }
        template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
        ENOKI_INLINE bool any_() const { return any(m1) || any(m2); }

        template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
        ENOKI_INLINE bool none_() const { return none(m1 | m2); }
        template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
        ENOKI_INLINE bool none_() const { return none(m1) && none(m2); }

        ENOKI_INLINE size_t count_() const { return count(m1) + count(m2); }

        ENOKI_INLINE const Mask1& low_() const { return m1; }
        ENOKI_INLINE const Mask2& high_() const { return m2; }

        ENOKI_INLINE Value coeff(size_t i) const {
            if (i < Size1)
                return m1.coeff(i);
            else
                return m2.coeff(i - Size1);
        }
    };

    using Mask = RMask;

    /// Default constructor
    ENOKI_INLINE StaticArrayImpl() : a1(), a2() { }

    /// Initialize all entries with a constant
    ENOKI_INLINE StaticArrayImpl(Value value) : a1(value), a2(value) { }

    /// Initialize from a list of component values
    template <typename... Args,
              std::enable_if_t<sizeof...(Args) == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(Args... args) {
        alignas(alignof(Array1)) Value storage[Size] = { (Value) args... };
        a1 = load<Array1>(storage);
        a2 = load<Array2>(storage + Size1);
    }

    /// Construct from the two sub-components
    ENOKI_INLINE StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : a1(a1), a2(a2) { }

    /// Cast another array
    template <size_t Size2, typename Type2, bool Approx2, RoundingMode Mode2,
              typename Derived2, std::enable_if_t<Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a)
        : a1(low(a)), a2(high(a)) { }

    /// Reinterpret another array
    template <size_t Size2, typename Type2, bool Approx2, RoundingMode Mode2,
              typename Derived2, std::enable_if_t<Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a,
        detail::reinterpret_flag)
        : a1(reinterpret_array<Array1>(low(a))), a2(reinterpret_array<Array2>(high(a))) { }

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return Derived(a1 + a.a1, a2 + a.a2); }
    ENOKI_INLINE Derived sub_(Arg a) const { return Derived(a1 - a.a1, a2 - a.a2); }
    ENOKI_INLINE Derived mul_(Arg a) const { return Derived(a1 * a.a1, a2 * a.a2); }
    ENOKI_INLINE Derived div_(Arg a) const { return Derived(a1 / a.a1, a2 / a.a2); }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        return Derived(mulhi(a1, a.a1), mulhi(a2, a.a2));
    }

    ENOKI_INLINE Mask lt_ (Arg a) const { return Mask(a1 <  a.a1, a2 <  a.a2); }
    ENOKI_INLINE Mask gt_ (Arg a) const { return Mask(a1 >  a.a1, a2 >  a.a2); }
    ENOKI_INLINE Mask le_ (Arg a) const { return Mask(a1 <= a.a1, a2 <= a.a2); }
    ENOKI_INLINE Mask ge_ (Arg a) const { return Mask(a1 >= a.a1, a2 >= a.a2); }
    ENOKI_INLINE Mask eq_ (Arg a) const { return Mask(eq(a1, a.a1), eq(a2, a.a2)); }
    ENOKI_INLINE Mask neq_(Arg a) const { return Mask(neq(a1, a.a1), neq(a2, a.a2)); }

    ENOKI_INLINE Derived min_(Arg a) const { return Derived(min(a1, a.a1), min(a2, a.a2)); }
    ENOKI_INLINE Derived max_(Arg a) const { return Derived(max(a1, a.a1), max(a2, a.a2)); }
    ENOKI_INLINE Derived abs_() const { return Derived(abs(a1), abs(a2)); }
    ENOKI_INLINE Derived ceil_() const { return Derived(ceil(a1), ceil(a2)); }
    ENOKI_INLINE Derived floor_() const { return Derived(floor(a1), floor(a2)); }
    ENOKI_INLINE Derived sqrt_() const { return Derived(sqrt(a1), sqrt(a2)); }
    ENOKI_INLINE Derived round_() const { return Derived(round(a1), round(a2)); }
    ENOKI_INLINE Derived rcp_() const { return Derived(rcp(a1), rcp(a2)); }
    ENOKI_INLINE Derived rsqrt_() const { return Derived(rsqrt(a1), rsqrt(a2)); }

    ENOKI_INLINE Derived fmadd_(Arg b, Arg c) const {
        return Derived(fmadd(a1, b.a1, c.a1), fmadd(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fnmadd_(Arg b, Arg c) const {
        return Derived(fnmadd(a1, b.a1, c.a1), fnmadd(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fmsub_(Arg b, Arg c) const {
        return Derived(fmsub(a1, b.a1, c.a1), fmsub(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fnmsub_(Arg b, Arg c) const {
        return Derived(fnmsub(a1, b.a1, c.a1), fnmsub(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fmaddsub_(Arg b, Arg c) const {
        return Derived(fmaddsub(a1, b.a1, c.a1), fmaddsub(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fmsubadd_(Arg b, Arg c) const {
        return Derived(fmsubadd(a1, b.a1, c.a1), fmsubadd(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived or_(Arg arg) const {
        return Derived(a1 | arg.a1, a2 | arg.a2);
    }

    ENOKI_INLINE Derived or_(const Mask &arg) const {
        return Derived(a1 | arg.m1, a2 | arg.m2);
    }

    ENOKI_INLINE Derived and_(Arg arg) const {
        return Derived(a1 & arg.a1, a2 & arg.a2);
    }

    ENOKI_INLINE Derived and_(const Mask &arg) const {
        return Derived(a1 & arg.m1, a2 & arg.m2);
    }

    ENOKI_INLINE Derived xor_(Arg arg) const {
        return Derived(a1 ^ arg.a1, a2 ^ arg.a2);
    }

    ENOKI_INLINE Derived xor_(const Mask &arg) const {
        return Derived(a1 ^ arg.m1, a2 ^ arg.m2);
    }

    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived sli_() const {
        return Derived(sli<Imm>(a1), sli<Imm>(a2));
    }

    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived sri_() const {
        return Derived(sri<Imm>(a1), sri<Imm>(a2));
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived sl_(size_t k) const {
        return Derived(a1 << k, a2 << k);
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived sr_(size_t k) const {
        return Derived(a1 >> k, a2 >> k);
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived slv_(Arg arg) const {
        return Derived(a1 << arg, a2 << arg);
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived srv_(Arg arg) const {
        return Derived(a1 >> arg, a2 >> arg);
    }

    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived roli_() const {
        return Derived(roli<Imm>(a1), roli<Imm>(a2));
    }

    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived rori_() const {
        return Derived(rori<Imm>(a1), rori<Imm>(a2));
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived rol_(size_t k) const {
        return Derived(rol(a1, k), rol(a2, k));
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived ror_(size_t k) const {
        return Derived(ror(a1, k), ror(a2, k));
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived rolv_(Arg arg) const {
        return Derived(rol(a1, arg.a1), rol(a2, arg.a2));
    }

    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE Derived rorv_(Arg arg) const {
        return Derived(ror(a1, arg.a1), ror(a2, arg.a2));
    }

    static ENOKI_INLINE Derived select_(const Mask &m, Arg t, Arg f) {
        return Derived(select(m.m1, t.a1, f.a1),
                       select(m.m2, t.a2, f.a2));
    }

    template <size_t Imm, typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE auto ror_array_() const {
        const Mask1 mask = index_sequence<Array1>() >= Scalar(Imm);

        Array1 a1_r = ror_array<Imm>(a1);
        Array1 a2_r = ror_array<Imm>(a2);

        return Derived(
            select(mask, a1_r, a2_r),
            select(mask, a2_r, a1_r)
        );
    }

    template <size_t Imm, typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE auto ror_array_() const {
        return Base::template ror_array_<Imm>();
    }

    template <size_t Imm, typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE auto rol_array_() const {
        const Mask1 mask = index_sequence<Array1>() < Scalar(Size1 - Imm);

        Array1 a1_r = rol_array<Imm>(a1);
        Array1 a2_r = rol_array<Imm>(a2);

        return Derived(
            select(mask, a1_r, a2_r),
            select(mask, a2_r, a1_r)
        );
    }

    template <size_t Imm, typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE auto rol_array_() const {
        return Base::template rol_array_<Imm>();
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Higher-level functions
    // -----------------------------------------------------------------------

    Derived exp_() const { return Derived(exp(a1), exp(a2));   }
    Derived log_() const { return Derived(log(a1), log(a2));   }

    Derived pow_(Arg a) const {
        return Derived(pow(a1, a.a1), pow(a2, a.a2));
    }

    Derived ldexp_(Arg a) const {
        return Derived(ldexp(a1, a.a1), ldexp(a2, a.a2));
    }

    std::pair<Derived, Derived> frexp_() const {
        auto r1 = frexp(a1);
        auto r2 = frexp(a2);
        return std::make_pair<Derived, Derived>(
            Derived(r1.first, r2.first),
            Derived(r1.second, r2.second)
        );
    }

    Derived sin_()   const { return Derived(sin(a1),   sin(a2));   }
    Derived sinh_()  const { return Derived(sinh(a1),  sinh(a2));  }
    Derived cos_()   const { return Derived(cos(a1),   cos(a2));   }
    Derived cosh_()  const { return Derived(cosh(a1),  cosh(a2));  }
    Derived tan_()   const { return Derived(tan(a1),   tan(a2));   }
    Derived tanh_()  const { return Derived(tanh(a1),  tanh(a2));  }
    Derived csc_()   const { return Derived(csc(a1),   csc(a2));   }
    Derived csch_()  const { return Derived(csch(a1),  csch(a2));  }
    Derived sec_()   const { return Derived(sec(a1),   sec(a2));   }
    Derived sech_()  const { return Derived(sech(a1),  sech(a2));  }
    Derived cot_()   const { return Derived(cot(a1),   cot(a2));   }
    Derived coth_()  const { return Derived(coth(a1),  coth(a2));  }
    Derived asin_()  const { return Derived(asin(a1),  asin(a2));  }
    Derived asinh_() const { return Derived(asinh(a1), asinh(a2)); }
    Derived acos_()  const { return Derived(acos(a1),  acos(a2));  }
    Derived acosh_() const { return Derived(acosh(a1), acosh(a2)); }
    Derived atan_()  const { return Derived(atan(a1),  atan(a2));  }
    Derived atanh_() const { return Derived(atanh(a1), atanh(a2)); }

    std::pair<Derived, Derived> sincos_() const {
        auto r1 = sincos(a1);
        auto r2 = sincos(a2);
        return std::make_pair<Derived, Derived>(
            Derived(r1.first, r2.first),
            Derived(r1.second, r2.second)
        );
    }

    std::pair<Derived, Derived> sincosh_() const {
        auto r1 = sincosh(a1);
        auto r2 = sincosh(a2);
        return std::make_pair<Derived, Derived>(
            Derived(r1.first, r2.first),
            Derived(r1.second, r2.second)
        );
    }

    Derived erf_() const  { return Derived(erf(a1),  erf(a2));  }
    Derived erfinv_() const { return Derived(erfinv(a1), erfinv(a2)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE Value hsum_() const { return hsum(a1 + a2); }
    template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE Value hsum_() const { return hsum(a1) + hsum(a2); }

    template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE Value hprod_() const { return hprod(a1 * a2); }
    template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE Value hprod_() const { return hprod(a1) * hprod(a2); }

    template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE Value hmin_() const { return hmin(min(a1, a2)); }
    template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE Value hmin_() const { return std::min(hmin(a1), hmin(a2)); }

    template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE Value hmax_() const { return hmax(max(a1, a2)); }
    template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE Value hmax_() const { return std::max(hmax(a1), hmax(a2)); }

    template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE auto dot_(Arg arg) const { return hsum(a1*arg.a1 + a2*arg.a2); }
    template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE auto dot_(Arg arg) const { return dot(a1, arg.a1) + dot(a2, arg.a2); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *out) const {
        store((Value *) out, a1);
        store((Value *) out + Size1, a2);
    }

    ENOKI_INLINE void store_(void *out, const Mask &mask) const {
        store((Value *) out, a1, low(mask));
        store((Value *) out + Size1, a2, high(mask));
    }

    ENOKI_INLINE void store_unaligned_(void *out) const {
        store_unaligned((Value *) out, a1);
        store_unaligned((Value *) out + Size1, a2);
    }

    ENOKI_INLINE void store_unaligned_(void *out, const Mask &mask) const {
        store_unaligned((Value *) out, a1, low(mask));
        store_unaligned((Value *) out + Size1, a2, high(mask));
    }

    static ENOKI_INLINE Derived load_(const void *a) {
        return Derived(
            load<Array1>((Value *) a),
            load<Array2>((Value *) a + Size1)
        );
    }

    static ENOKI_INLINE Derived load_(const void *a, const Mask &mask) {
        return Derived(
            load<Array1>((Value *) a, low(mask)),
            load<Array2>((Value *) a + Size1, high(mask))
        );
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *a) {
        return Derived(
            load_unaligned<Array1>((Value *) a),
            load_unaligned<Array2>((Value *) a + Size1)
        );
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *a, const Mask &mask) {
        return Derived(
            load_unaligned<Array1>((Value *) a, low(mask)),
            load_unaligned<Array2>((Value *) a + Size1, high(mask))
        );
    }

    static ENOKI_INLINE Derived zero_() {
        return Derived(zero<Array1>(), zero<Array2>());
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        prefetch<Array1, Stride, Write, Level>(ptr, low(index));
        prefetch<Array2, Stride, Write, Level>(ptr, high(index));
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index, const Mask &mask) {
        prefetch<Array1, Stride, Write, Level>(ptr, low(index), low(mask));
        prefetch<Array2, Stride, Write, Level>(ptr, high(index), high(mask));
    }

    template <size_t Stride, typename Index>
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return Derived(
            gather<Array1, Stride>(ptr, low(index)),
            gather<Array2, Stride>(ptr, high(index))
        );
    }

    template <size_t Stride, typename Index>
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Derived(
            gather<Array1, Stride>(ptr, low(index), low(mask)),
            gather<Array2, Stride>(ptr, high(index), high(mask))
        );
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        scatter<Stride>(ptr, a1, low(index));
        scatter<Stride>(ptr, a2, high(index));
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        scatter<Stride>(ptr, a1, low(index), low(mask));
        scatter<Stride>(ptr, a2, high(index), high(mask));
    }

    template <typename T = Derived, std::enable_if_t<T::Size1 == T::Size2, int> = 0>
    ENOKI_INLINE Value extract_(const Mask &mask) const {
        return extract(select(low(mask), a1, a2), low(mask) | high(mask));
    }

    template <typename T = Derived, std::enable_if_t<T::Size1 != T::Size2, int> = 0>
    ENOKI_INLINE Value extract_(const Mask &mask) const {
        if (ENOKI_LIKELY(any(low(mask))))
            return extract(a1, low(mask));
        else
            return extract(a2, high(mask));
    }

    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        store_compress(ptr, a1, low(mask));
        store_compress(ptr, a2, high(mask));
    }

    template <size_t Stride, typename Index, typename Func, typename... Args>
    static ENOKI_INLINE void transform_masked_(void *ptr, const Index &index, const Mask &mask,
                                               const Func &func, const Args &... args) {
        transform<Array1, Stride>(ptr, low(index),  low(mask),  func, low(args)...);
        transform<Array2, Stride>(ptr, high(index), high(mask), func, high(args)...);
    }

    template <size_t Stride, typename Index, typename Func, typename... Args>
    static ENOKI_INLINE void transform_(void *ptr, const Index &index,
                                        const Func &func, const Args &... args) {
        transform<Array1, Stride>(ptr, low(index),  func, low(args)...);
        transform<Array2, Stride>(ptr, high(index), func, high(args)...);
    }

    // -----------------------------------------------------------------------
    //! @{ \name Component access
    // -----------------------------------------------------------------------

    ENOKI_INLINE const Array1& low_() const { return a1; }
    ENOKI_INLINE const Array2& high_() const { return a2; }

    ENOKI_INLINE const Value& coeff(size_t i) const {
        #if defined(__clang__) || defined(NDEBUG)
            union Helper { Derived t; Value x[Size]; };
            return ((const Helper *) this)->x[i];
        #else
            if (i < Size1)
                return a1.coeff(i);
            else
                return a2.coeff(i - Size1);
        #endif
    }

    ENOKI_INLINE Value& coeff(size_t i) {
        #if defined(__clang__) || defined(NDEBUG)
            union Helper { Derived t; Value x[Size]; };
            return ((Helper *) this)->x[i];
        #else
            if (i < Size1)
                return a1.coeff(i);
            else
                return a2.coeff(i - Size1);
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------

    Array1 a1;
    Array2 a2;
};

NAMESPACE_END(enoki)
