/*
    enoki/array_neon.h -- Packed SIMD array (ARM 64 bit NEON specialization)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyrighe (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array_generic.h"

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

template <> struct is_native<float, 4> : std::true_type { };
template <> struct is_native<double, 2> : std::true_type { };
template <typename T> struct is_native<T, 4, is_int32_t<T>> : std::true_type { };
template <typename T> struct is_native<T, 4, is_int64_t<T>> : std::true_type { };

NAMESPACE_END(detail)

ENOKI_INLINE uint64x2_t vmvnq_u64(uint64x2_t a) {
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a)));
}

ENOKI_INLINE int64x2_t vmvnq_s64(int64x2_t a) {
    return vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(a)));
}

/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (single precision)
template <bool Approx, typename Derived> struct ENOKI_MAY_ALIAS alignas(16)
    StaticArrayImpl<float, 4, Approx, RoundingMode::Default, Derived>
    : StaticArrayBase<float, 4, Approx, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY_CLASSIC(float, 4, Approx, float32x4_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(const Value &value) : m(vdupq_n_f32(value)) { }
    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m{v0, v1, v2, v3} { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(float) : m(a.derived().m) { }
    ENOKI_CONVERT(int32_t) : m(vcvtq_f32_s32(vreinterpretq_s32_u32(a.derived().m))) { }
    ENOKI_CONVERT(uint32_t) : m(vcvtq_f32_u32(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) : m(a.derived().m) { }
    ENOKI_REINTERPRET(int32_t) : m(vreinterpretq_f32_u32(a.derived().m)) { }
    ENOKI_REINTERPRET(uint32_t) : m(vreinterpretq_f32_u32(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_ps(a1.coeff(0), a1.coeff(1), a2.coeff(0), a2.coeff(1))) { }

    ENOKI_INLINE Array1 low_()  const { return Array1(coeff(0), coeff(1)); }
    ENOKI_INLINE Array2 high_() const { return Array2(coeff(2), coeff(3)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return vaddq_f32(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return vsubq_f32(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const { return vmulq_f32(m, a.m); }
    ENOKI_INLINE Derived div_(Arg a) const { return vdivq_f32(m, a.m); }

    ENOKI_INLINE Derived fmadd_(Arg b, Arg c) const { return vfmaq_f32(c.m, m, b.m); }
    ENOKI_INLINE Derived fmsub_(Arg b, Arg c) const { return vfmsq_f32(c.m, m, b.m); }

    ENOKI_INLINE Derived or_ (Arg a) const { return vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(m), vreinterpretq_s32_f32(a.m))); }
    ENOKI_INLINE Derived and_(Arg a) const { return vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(m), vreinterpretq_s32_f32(a.m))); }
    ENOKI_INLINE Derived xor_(Arg a) const { return vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(m), vreinterpretq_s32_f32(a.m))); }

    ENOKI_INLINE auto lt_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcltq_f32(m, a.m))); }
    ENOKI_INLINE auto gt_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcgtq_f32(m, a.m))); }
    ENOKI_INLINE auto le_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcleq_f32(m, a.m))); }
    ENOKI_INLINE auto ge_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vcgeq_f32(m, a.m))); }
    ENOKI_INLINE auto eq_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vceqq_f32(m, a.m))); }
    ENOKI_INLINE auto neq_(Arg a) const { return mask_t<Derived>(vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(m, a.m)))); }

    ENOKI_INLINE Derived abs_()      const { return vabsq_f32(m); }
    ENOKI_INLINE Derived neg_()      const { return vnegq_f32(m); }
    ENOKI_INLINE Derived not_()      const { return vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(m))); }

    ENOKI_INLINE Derived min_(Arg b) const { return vminq_f32(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return vmaxq_f32(b.m, m); }
    ENOKI_INLINE Derived sqrt_()     const { return vsqrtq_f32(m);     }
    ENOKI_INLINE Derived round_()    const { return vrndnq_f32(m);     }
    ENOKI_INLINE Derived floor_()    const { return vrndmq_f32(m);     }
    ENOKI_INLINE Derived ceil_()     const { return vrndpq_f32(m);     }

    ENOKI_INLINE Derived rcp_() const {
        if (Approx) {
            float32x4_t r = vrecpeq_f32(m);
            r = vmulq_f32(r, vrecpsq_f32(r, m));
            return r;
        } else {
            return Base::rcp_();
        }
    }

    ENOKI_INLINE Derived rsqrt_() const {
        if (Approx) {
            float32x4_t r = vrsqrteq_f32(m);
            r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(r, m), r));
            return r;
        } else {
            return Base::rcp_();
        }
    }

    template <typename Mask_>
    static ENOKI_INLINE Derived select_(const Mask_ &m, Arg t, Arg f) {
        return vbslq_f32(vreinterpretq_u32_f32(m.m), t.m, f.m);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hmax_() { return vmaxvq_f32(m); }
    ENOKI_INLINE Value hmin_() { return vminvq_f32(m); }
    ENOKI_INLINE Value hsum_() { return vaddvq_f32(m); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    using Base::load_;
    using Base::store_;
    using Base::load_unaligned_;
    using Base::store_unaligned_;

    ENOKI_INLINE void store_(void *ptr) const {
        vst1q_f32((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16), m);
    }

    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        vst1q_f32((Value *) ptr, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return vld1_f32((const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16));
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return vld1_f32((const Value *) ptr);
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (32-bit integers)
template <typename Value_, typename Derived>
struct ENOKI_MAY_ALIAS alignas(16) StaticArrayImpl<Value_, 4, false, RoundingMode::Default,
                                                   Derived, detail::is_int32_t<Value_>>
    : StaticArrayBase<Value_, 4, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY_CLASSIC(Value_, 4, false, uint32x4_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(const Value &value) : m(vdupq_n_u32((uint32_t) value)) { }
    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m{v0, v1, v2, v3} { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(int32_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint32_t) : m(a.derived().m) { }
    ENOKI_CONVERT(float) : m(std::is_signed<Value>::value ?
          vreinterpretq_u32_s32(vcvtq_s32_f32(a.derived().m))
        : vcvtq_u32_f32(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(int32_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint32_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(float) : m(vreinterpretq_u32_f32(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------


    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_ps(a1.coeff(0), a1.coeff(1), a2.coeff(0), a2.coeff(1))) { }

    ENOKI_INLINE Array1 low_()  const { return Array1(coeff(0), coeff(1)); }
    ENOKI_INLINE Array2 high_() const { return Array2(coeff(2), coeff(3)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return vaddq_u32(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return vsubq_u32(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u32_s32(vmulq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return vmulq_u32(m, a.m);
    }

    ENOKI_INLINE Derived or_ (Arg a) const { return vorrq_u32(m, a.m); }
    ENOKI_INLINE Derived and_(Arg a) const { return vandq_u32(m, a.m); }
    ENOKI_INLINE Derived xor_(Arg a) const { return veorq_u32(m, a.m); }

    ENOKI_INLINE auto lt_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcltq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcltq_u32(m, a.m));
    }

    ENOKI_INLINE auto gt_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcgtq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcgtq_u32(m, a.m));
    }

    ENOKI_INLINE auto le_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcleq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcleq_u32(m, a.m));
    }

    ENOKI_INLINE auto ge_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcgeq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m)));
        else
            return mask_t<Derived>(vcgeq_u32(m, a.m));
    }

    ENOKI_INLINE auto eq_ (Arg a) const { return mask_t<Derived>(vceqq_u32(m, a.m)); }
    ENOKI_INLINE auto neq_(Arg a) const { return mask_t<Derived>(vmvnq_u32(vceqq_u32(m, a.m))); }

    ENOKI_INLINE Derived abs_() const {
        if (!std::is_signed<Value>())
            return m;
        return vreinterpretq_u32_s32(vabsq_s32(vreinterpretq_s32_u32(m)));
    }

    ENOKI_INLINE Derived neg_() const {
        static_assert(std::is_signed<Value>::value);
        return vreinterpretq_u32_s32(vnegq_s32(vreinterpretq_s32_u32(m)));
    }

    ENOKI_INLINE Derived not_()      const { return vmvnq_u32(m); }

    ENOKI_INLINE Derived min_(Arg b) const { return vminq_u32(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return vmaxq_u32(b.m, m); }

    template <typename Mask_>
    static ENOKI_INLINE Derived select_(const Mask_ &m, Arg t, Arg f) {
        return vbslq_u32(m.m, t.m, f.m);
    }

    template <size_t Imm> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u32_s32(
                vshrq_n_s32(vreinterpretq_s32_u32(m), (int) Imm));
        else
            return vshrq_n_u32(m, (int) Imm);
    }

    template <size_t Imm> ENOKI_INLINE Derived sli_() const {
        return vshlq_n_u32(m, (int) Imm);
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(m), vdupq_n_s32(-(int) k)));
        else
            return vshlq_u32(m, vdupq_n_s32(-(int) k));
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        return vshlq_u32(m, vdupq_n_s32((int) k));
    }

    ENOKI_INLINE Derived sr_(const Value &a) const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(m),
                          vnegq_s32(vreinterpretq_s32_u32(a.m))));
        else
            return vshlq_u32(m, vnegq_s32(vreinterpretq_s32_u32(a.m)));
    }

    ENOKI_INLINE Derived sl_(const Value &a) const {
        return vshlq_u32(m, vreinterpretq_s32_u32(a.m));
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        if (std::is_signed<Value>::value) {
            int64x2_t l = vmull_s32(vreinterpret_s32_u32(vget_low_u32(m)), vreinterpret_s32_u32(vget_low_u32(a.m)));
            int64x2_t h = vmull_high_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(a.m));
            return vreinterpretq_u32_s32(vuzp2q_s32(vreinterpretq_s32_s64(l), vreinterpretq_s32_s64(h)));
        } else {
            uint64x2_t l = vmull_u32(vget_low_u32(m), vget_low_u32(a.m));
            uint64x2_t h = vmull_high_u32(m, a.m);
            return vuzp2q_u32(vreinterpretq_u32_u64(l), vreinterpretq_u32_u64(h));
        }
    }

    ENOKI_INLINE Derived lzcnt_() const { return vclzq_u32(m); }
    ENOKI_INLINE Derived tzcnt_() const { return Value(32) - lzcnt(~derived() & (derived() - Value(1))); }
    ENOKI_INLINE Derived popcnt_() const { return vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(m)))); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hmax_() { return vmaxvq_u32(m); }
    ENOKI_INLINE Value hmin_() { return vminvq_u32(m); }
    ENOKI_INLINE Value hsum_() { return vaddvq_u32(m); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    using Base::load_;
    using Base::store_;
    using Base::load_unaligned_;
    using Base::store_unaligned_;

    ENOKI_INLINE void store_(void *ptr) const {
        vst1q_u32((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16), m);
    }

    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        vst1q_u32((Value *) ptr, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return vld1_u32((const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16));
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return vld1_u32((const Value *) ptr);
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (double precision)
template <bool Approx, typename Derived> struct ENOKI_MAY_ALIAS alignas(16)
    StaticArrayImpl<double, 2, Approx, RoundingMode::Default, Derived>
    : StaticArrayBase<double, 2, Approx, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY_CLASSIC(double, 2, Approx, float64x2_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(const Value &value) : m(vdupq_n_f64(value)) { }
    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1) : m{v0, v1} { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(double) : m(a.derived().m) { }
    ENOKI_CONVERT(int64_t) : m(vcvtq_f64_s64(vreinterpretq_s64_u64(a.derived().m))) { }
    ENOKI_CONVERT(uint64_t) : m(vcvtq_f64_u64(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(double) : m(a.derived().m) { }
    ENOKI_REINTERPRET(int64_t) : m(vreinterpretq_f64_u64(a.derived().m)) { }
    ENOKI_REINTERPRET(uint64_t) : m(vreinterpretq_f64_u64(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_ps(a1.coeff(0), a2.coeff(0))) { }

    ENOKI_INLINE Array1 low_()  const { return Array1(coeff(0)); }
    ENOKI_INLINE Array2 high_() const { return Array2(coeff(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return vaddq_f64(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return vsubq_f64(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const { return vmulq_f64(m, a.m); }
    ENOKI_INLINE Derived div_(Arg a) const { return vdivq_f64(m, a.m); }

    ENOKI_INLINE Derived fmadd_(Arg b, Arg c) const { return vfmaq_f64(c.m, m, b.m); }
    ENOKI_INLINE Derived fmsub_(Arg b, Arg c) const { return vfmsq_f64(c.m, m, b.m); }

    ENOKI_INLINE Derived or_ (Arg a) const { return vreinterpretq_f64_s64(vorrq_s64(vreinterpretq_s64_f64(m), vreinterpretq_s64_f64(a.m))); }
    ENOKI_INLINE Derived and_(Arg a) const { return vreinterpretq_f64_s64(vandq_s64(vreinterpretq_s64_f64(m), vreinterpretq_s64_f64(a.m))); }
    ENOKI_INLINE Derived xor_(Arg a) const { return vreinterpretq_f64_s64(veorq_s64(vreinterpretq_s64_f64(m), vreinterpretq_s64_f64(a.m))); }

    ENOKI_INLINE auto lt_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcltq_f64(m, a.m))); }
    ENOKI_INLINE auto gt_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcgtq_f64(m, a.m))); }
    ENOKI_INLINE auto le_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcleq_f64(m, a.m))); }
    ENOKI_INLINE auto ge_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vcgeq_f64(m, a.m))); }
    ENOKI_INLINE auto eq_ (Arg a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vceqq_f64(m, a.m))); }
    ENOKI_INLINE auto neq_(Arg a) const { return mask_t<Derived>(vreinterpretq_f64_u64(vmvnq_u64(vceqq_f64(m, a.m)))); }

    ENOKI_INLINE Derived abs_()      const { return vabsq_f64(m); }
    ENOKI_INLINE Derived neg_()      const { return vnegq_f64(m); }
    ENOKI_INLINE Derived not_()      const { return vreinterpretq_f64_s64(vmvnq_s64(vreinterpretq_s64_f64(m))); }

    ENOKI_INLINE Derived min_(Arg b) const { return vminq_f64(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return vmaxq_f64(b.m, m); }
    ENOKI_INLINE Derived sqrt_()     const { return vsqrtq_f64(m);     }
    ENOKI_INLINE Derived round_()    const { return vrndnq_f64(m);     }
    ENOKI_INLINE Derived floor_()    const { return vrndmq_f64(m);     }
    ENOKI_INLINE Derived ceil_()     const { return vrndpq_f64(m);     }

    ENOKI_INLINE Derived rcp_() const {
        if (Approx) {
            float64x2_t r = vrecpeq_f64(m);
            r = vmulq_f64(r, vrecpsq_f64(r, m));
            return r;
        } else {
            return Base::rcp_();
        }
    }

    ENOKI_INLINE Derived rsqrt_() const {
        if (Approx) {
            float64x2_t r = vrsqrteq_f64(m);
            r = vmulq_f64(r, vrsqrtsq_f64(vmulq_f64(r, m), r));
            return r;
        } else {
            return Base::rcp_();
        }
    }

    template <typename Mask_>
    static ENOKI_INLINE Derived select_(const Mask_ &m, Arg t, Arg f) {
        return vbslq_f64(vreinterpretq_u64_f64(m.m), t.m, f.m);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hmax_() { return vmaxvq_f64(m); }
    ENOKI_INLINE Value hmin_() { return vminvq_f64(m); }
    ENOKI_INLINE Value hsum_() { return vaddvq_f64(m); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    using Base::load_;
    using Base::store_;
    using Base::load_unaligned_;
    using Base::store_unaligned_;

    ENOKI_INLINE void store_(void *ptr) const {
        vst1q_f64((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16), m);
    }

    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        vst1q_f64((Value *) ptr, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return vld1_f64((const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16));
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return vld1_f64((const Value *) ptr);
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using ARM NEON intrinsics (64-bit integers)
template <typename Value_, typename Derived>
struct ENOKI_MAY_ALIAS alignas(16) StaticArrayImpl<Value_, 2, false, RoundingMode::Default,
                                                   Derived, detail::is_int64_t<Value_>>
    : StaticArrayBase<Value_, 2, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY_CLASSIC(Value_, 2, false, uint64x2_t)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(const Value &value) : m(vdupq_n_u64((uint64_t) value)) { }
    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1) : m{v0, v1} { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(int64_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint64_t) : m(a.derived().m) { }
    ENOKI_CONVERT(double) : m(std::is_signed<Value>::value ?
          vreinterpretq_u64_s64(vcvtq_s64_f64(a.derived().m))
        : vcvtq_u64_f64(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(int64_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint64_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(double) : m(vreinterpretq_u64_f64(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------


    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_ps(a1.coeff(0), a2.coeff(0))) { }

    ENOKI_INLINE Array1 low_()  const { return Array1(coeff(0)); }
    ENOKI_INLINE Array2 high_() const { return Array2(coeff(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return vaddq_u64(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return vsubq_u64(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u64_s64(vmulq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return vmulq_u64(m, a.m);
    }

    ENOKI_INLINE Derived or_ (Arg a) const { return vorrq_u64(m, a.m); }
    ENOKI_INLINE Derived and_(Arg a) const { return vandq_u64(m, a.m); }
    ENOKI_INLINE Derived xor_(Arg a) const { return veorq_u64(m, a.m); }

    ENOKI_INLINE auto lt_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcltq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcltq_u64(m, a.m));
    }

    ENOKI_INLINE auto gt_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcgtq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcgtq_u64(m, a.m));
    }

    ENOKI_INLINE auto le_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcleq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcleq_u64(m, a.m));
    }

    ENOKI_INLINE auto ge_(Arg a) const {
        if (std::is_signed<Value>::value)
            return mask_t<Derived>(vcgeq_s64(vreinterpretq_s64_u64(m), vreinterpretq_s64_u64(a.m)));
        else
            return mask_t<Derived>(vcgeq_u64(m, a.m));
    }

    ENOKI_INLINE auto eq_ (Arg a) const { return mask_t<Derived>(vceqq_u64(m, a.m)); }
    ENOKI_INLINE auto neq_(Arg a) const { return mask_t<Derived>(vmvnq_u64(vceqq_u64(m, a.m))); }

    ENOKI_INLINE Derived abs_() const {
        if (!std::is_signed<Value>())
            return m;
        return vreinterpretq_u64_s64(vabsq_s64(vreinterpretq_s64_u64(m)));
    }

    ENOKI_INLINE Derived neg_() const {
        static_assert(std::is_signed<Value>::value);
        return vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(m)));
    }

    ENOKI_INLINE Derived not_()      const { return vmvnq_u64(m); }

    ENOKI_INLINE Derived min_(Arg b) const { return vminq_u64(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return vmaxq_u64(b.m, m); }

    template <typename Mask_>
    static ENOKI_INLINE Derived select_(const Mask_ &m, Arg t, Arg f) {
        return vbslq_u64(m.m, t.m, f.m);
    }

    template <size_t Imm> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u64_s64(
                vshrq_n_s64(vreinterpretq_s64_u64(m), (int) Imm));
        else
            return vshrq_n_u64(m, (int) Imm);
    }

    template <size_t Imm> ENOKI_INLINE Derived sli_() const {
        return vshlq_n_u64(m, (int) Imm);
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u64_s64(
                vshlq_s64(vreinterpretq_s64_u64(m), vdupq_n_s64(-(int) k)));
        else
            return vshlq_u64(m, vdupq_n_s64(-(int) k));
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        return vshlq_u64(m, vdupq_n_s64((int) k));
    }

    ENOKI_INLINE Derived sr_(const Value &a) const {
        if (std::is_signed<Value>::value)
            return vreinterpretq_u64_s64(
                vshlq_s64(vreinterpretq_s64_u64(m),
                          vnegq_s64(vreinterpretq_s64_u64(a.m))));
        else
            return vshlq_u64(m, vnegq_s64(vreinterpretq_s64_u64(a.m)));
    }

    ENOKI_INLINE Derived sl_(const Value &a) const {
        return vshlq_u64(m, vreinterpretq_s64_u64(a.m));
    }

    ENOKI_INLINE Derived popcnt_() const { return vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(m))))); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_() { return vaddvq_u64(m); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    using Base::load_;
    using Base::store_;
    using Base::load_unaligned_;
    using Base::store_unaligned_;

    ENOKI_INLINE void store_(void *ptr) const {
        vst1q_u64((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16), m);
    }

    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        vst1q_u64((Value *) ptr, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return vld1_u64((const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 16));
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return vld1_u64((const Value *) ptr);
    }

    //! @}
    // -----------------------------------------------------------------------
};

NAMESPACE_END(enoki)
