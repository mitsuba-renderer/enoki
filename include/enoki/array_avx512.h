/*
    enoki/array_avx512.h -- Packed SIMD array (AVX512 specialization)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array_avx2.h"

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

template <> struct is_native<float, 16> : std::true_type { };
template <> struct is_native<double, 8> : std::true_type { };
template <typename T> struct is_native<T, 16, is_int32_t<T>> : std::true_type { };
template <typename T> struct is_native<T, 8,  is_int64_t<T>> : std::true_type { };

/// Wraps an individual bit of a mask register
struct KMaskBit {
    bool value : 1;

    operator bool() const { return value; }

    friend std::ostream &operator<<(std::ostream &os, const KMaskBit &b) {
        os << (b.value ? '1' : '0');
        return os;
    }
};

/// Wrapper for AVX512 k0-k7 mask registers
template <typename Type>
struct KMask : StaticArrayBase<detail::KMaskBit, sizeof(Type) * 8, false,
                               RoundingMode::Default, KMask<Type>> {
    using Base = StaticArrayBase<detail::KMaskBit, sizeof(Type) * 8, false,
                                 RoundingMode::Default, KMask<Type>>;
    static constexpr bool IsMask = true;
    static constexpr bool IsNative = true;
    using Base::Size;
    using Expr = KMask;
    using Mask = KMask;
    using HalfType = __mmask8;
    Type k;

    ENOKI_INLINE KMask() { }
    ENOKI_INLINE explicit KMask(Type k) : k(k) { }
    template <typename T, std::enable_if_t<std::is_same<T, bool>::value, int> = 0>
    ENOKI_INLINE KMask(T b) : k(b ? Type(-1) : Type(0)) { }
    /// Convert a compatible mask
    template <typename T, std::enable_if_t<T::IsMask, int> = 0>
    ENOKI_INLINE KMask(T value) : k(reinterpret_array<KMask>(value).k) { }

    ENOKI_INLINE KMask(KMask k, reinterpret_flag) : k(k.k) { }

    ENOKI_REINTERPRET_KMASK(bool, 16) {
        __m128i value = _mm_loadu_si128((__m128i *) a.data());
        #if defined(__AVX512VL__) && defined(__AVX512BW__)
            k = _mm_test_epi8_mask(value, _mm_set1_epi8((char) 0xFF));
        #else
            k = _mm512_test_epi32_mask(_mm512_cvtepi8_epi32(value), _mm512_set1_epi8((char) 0xFF));
        #endif
    }

    ENOKI_REINTERPRET_KMASK(bool, 8) {
        __m128i value = _mm_loadl_epi64((const __m128i *) a.data());
        #if defined(__AVX512VL__) && defined(__AVX512BW__)
            k = (__mmask8) _mm_test_epi8_mask(value, _mm_set1_epi8((char) 0xFF));
        #else
            k = _mm512_test_epi64_mask(_mm512_cvtepi8_epi64(value), _mm512_set1_epi8((char) 0xFF));
        #endif
    }

#if defined(__AVX512VL__)
    ENOKI_REINTERPRET_KMASK(float, 8)
        : k(_mm256_test_epi32_mask(_mm256_castps_si256(a.derived().m),
                                   _mm256_castps_si256(a.derived().m))) { }
    ENOKI_REINTERPRET_KMASK(int32_t, 8)  : k(_mm256_test_epi32_mask(a.derived().m, a.derived().m)) { }
    ENOKI_REINTERPRET_KMASK(uint32_t, 8) : k(_mm256_test_epi32_mask(a.derived().m, a.derived().m)) { }
#else
    ENOKI_REINTERPRET_KMASK(float, 8)    : k(Type(_mm256_movemask_ps(a.derived().m))) { }
    ENOKI_REINTERPRET_KMASK(int32_t, 8)  : k(Type(_mm256_movemask_ps(_mm256_castsi256_ps(a.derived().m)))) { }
    ENOKI_REINTERPRET_KMASK(uint32_t, 8) : k(Type(_mm256_movemask_ps(_mm256_castsi256_ps(a.derived().m)))) { }
#endif

    ENOKI_REINTERPRET_KMASK(double, 16)   { k = _mm512_kunpackb(high(a).k, low(a).k); }
    ENOKI_REINTERPRET_KMASK(int64_t, 16)  { k = _mm512_kunpackb(high(a).k, low(a).k); }
    ENOKI_REINTERPRET_KMASK(uint64_t, 16) { k = _mm512_kunpackb(high(a).k, low(a).k); }

    ENOKI_INLINE KMask eq_(KMask a) const {
        if (Size == 16) /* Use intrinsic if possible */
            return KMask(Type(_mm512_kxnor((__mmask16) k, (__mmask16) (a.k))));
        else
            return KMask(Type(~(k ^ a.k)));
    }

    ENOKI_INLINE KMask neq_(KMask a) const {
        if (Size == 16) /* Use intrinsic if possible */
            return KMask(Type(_mm512_kxor((__mmask16) k, (__mmask16) (a.k))));
        else
            return KMask(Type(k ^ a.k));
    }

    ENOKI_INLINE KMask or_(KMask a) const {
        if (Size == 16) /* Use intrinsic if possible */
            return KMask(Type(_mm512_kor((__mmask16) k, (__mmask16) (a.k))));
        else
            return KMask(Type(k | a.k));
    }

    ENOKI_INLINE KMask and_(KMask a) const {
        if (Size == 16) /* Use intrinsic if possible */
            return KMask(Type(_mm512_kand((__mmask16) k, (__mmask16) (a.k))));
        else
            return KMask(Type(k & a.k));
    }

    ENOKI_INLINE KMask xor_(KMask a) const {
        if (Size == 16) /* Use intrinsic if possible */
            return KMask(Type(_mm512_kxor((__mmask16) k, (__mmask16) (a.k))));
        else
            return KMask(Type(k ^ a.k));
    }

    ENOKI_INLINE KMask not_() const {
        if (Size == 16) /* Use intrinsic if possible */
            return KMask(Type(_mm512_knot((__mmask16) k)));
        else
            return KMask(Type(~k));
    }

    ENOKI_INLINE bool all_() const {
        if (std::is_same<Type, __mmask16>::value)
            return _mm512_kortestc((__mmask16) k, (__mmask16) k);
        else
            return k == Type((1 << Size) - 1);
    }

    ENOKI_INLINE bool none_() const {
        if (std::is_same<Type, __mmask16>::value)
            return _mm512_kortestz((__mmask16) k, (__mmask16) k);
        else
            return k == Type(0);
    }

    ENOKI_INLINE bool any_() const {
        if (std::is_same<Type, __mmask16>::value)
            return !_mm512_kortestz((__mmask16) k, (__mmask16) k);
        else
            return k != Type(0);
    }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) k);
    }

    ENOKI_INLINE KMaskBit coeff(size_t i) const {
        assert(i < Size);
        return KMaskBit { (k & (1 << i)) != 0 };
    }

    static ENOKI_INLINE KMask select_(const Mask &m, const KMask &t,
                                      const KMask &f) {
        if (std::is_same<Type, __mmask16>::value)
            return KMask(Type(_mm512_kor(_mm512_kand((__mmask16) m.k, (__mmask16) t.k),
                                         _mm512_kandn((__mmask16) m.k, (__mmask16) f.k))));
        else
            return KMask((m & t) | (~m & f));
    }

    template <typename T>
    ENOKI_INLINE size_t compress_(T *&ptr, const KMask &mask) const {
        store_unaligned(ptr, KMask((Type) _pext_u32((unsigned int) k, (unsigned int) mask.k)));
        return count(mask);
    }

    ENOKI_INLINE void store_unaligned_(void *mem) const { memcpy(mem, &k, sizeof(Type)); }

    KMask<HalfType> low_()  const { return KMask<HalfType>(HalfType(k)); }
    KMask<HalfType> high_() const { return KMask<HalfType>(HalfType(k >> (Size/2))); }
};

NAMESPACE_END(detail)

/// Partial overload of StaticArrayImpl using AVX512 intrinsics (single precision)
template <bool Approx, RoundingMode Mode, typename Derived> struct alignas(64)
    StaticArrayImpl<float, 16, Approx, Mode, Derived>
    : StaticArrayBase<float, 16, Approx, Mode, Derived> {

    ENOKI_NATIVE_ARRAY(float, 16, Approx, __m512, Mode)
    using Mask = detail::KMask<__mmask16>;

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value) : m(_mm512_set1_ps(value)) { }
    ENOKI_INLINE StaticArrayImpl(Value f0, Value f1, Value f2, Value f3,
                                 Value f4, Value f5, Value f6, Value f7,
                                 Value f8, Value f9, Value f10, Value f11,
                                 Value f12, Value f13, Value f14, Value f15)
        : m(_mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8,
                           f9, f10, f11, f12, f13, f14, f15)) { }


    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(half)
        : m(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) a.data()))) { }

    ENOKI_CONVERT(float) : m(a.derived().m) { }

    ENOKI_CONVERT(int32_t) : m(_mm512_cvt_roundepi32_ps(a.derived().m, (int) Mode)) { }

    ENOKI_CONVERT(uint32_t) : m(_mm512_cvt_roundepu32_ps(a.derived().m, (int) Mode)) { }

    ENOKI_CONVERT(double)
        : m(detail::concat(_mm512_cvt_roundpd_ps(low(a).m, (int) Mode),
                           _mm512_cvt_roundpd_ps(high(a).m, (int) Mode))) { }

#if defined(__AVX512DQ__)
    ENOKI_CONVERT(int64_t)
        : m(detail::concat(_mm512_cvt_roundepi64_ps(low(a).m, (int) Mode),
                           _mm512_cvt_roundepi64_ps(high(a).m, (int) Mode))) { }

    ENOKI_CONVERT(uint64_t)
        : m(detail::concat(_mm512_cvt_roundepu64_ps(low(a).m, (int) Mode),
                           _mm512_cvt_roundepu64_ps(high(a).m, (int) Mode))) { }
#elif defined(__AVX512CD__)
    ENOKI_CONVERT(uint64_t) {
        /* Emulate uint64_t -> float conversion using other intrinsics
           instead of falling back to scalar operations. This is quite
           a bit faster. */

        __m512i v0 = low(a).m,
                v1 = high(a).m,
                lz0 = _mm512_lzcnt_epi64(v0),
                lz1 = _mm512_lzcnt_epi64(v1);

        __mmask8 zero0 =
            _mm512_cmp_epi64_mask(v0, _mm512_setzero_si512(), _MM_CMPINT_NE);

        __mmask8 zero1 =
            _mm512_cmp_epi64_mask(v1, _mm512_setzero_si512(), _MM_CMPINT_NE);

        __m512i mant0 = _mm512_mask_blend_epi64(
            _mm512_cmp_epi64_mask(lz0, _mm512_set1_epi64(63-24), _MM_CMPINT_GT),
            _mm512_srlv_epi64(v0, _mm512_sub_epi64(_mm512_set1_epi64(63 - 23), lz0)),
            _mm512_sllv_epi64(v0, _mm512_add_epi64(_mm512_set1_epi64(23 - 63), lz0))
        );

        __m512i mant1 = _mm512_mask_blend_epi64(
            _mm512_cmp_epi64_mask(lz1, _mm512_set1_epi64(63-24), _MM_CMPINT_GT),
            _mm512_srlv_epi64(v1, _mm512_sub_epi64(_mm512_set1_epi64(63 - 23), lz1)),
            _mm512_sllv_epi64(v1, _mm512_add_epi64(_mm512_set1_epi64(23 - 63), lz1))
        );

        __m512i exp0 = _mm512_slli_epi64(
            _mm512_sub_epi64(_mm512_set1_epi64(127 + 63), lz0), 23);

        __m512i exp1 = _mm512_slli_epi64(
            _mm512_sub_epi64(_mm512_set1_epi64(127 + 63), lz1), 23);

        __m512i comb0 = _mm512_or_epi64(exp0, _mm512_and_epi64(
            mant0, _mm512_set1_epi64(0b0'00000000'11111111111111111111111)));

        __m512i comb1 = _mm512_or_epi64(exp1, _mm512_and_epi64(
            mant1, _mm512_set1_epi64(0b0'00000000'11111111111111111111111)));

        __m256 flt0 =
            _mm256_castsi256_ps(_mm512_maskz_cvtepi64_epi32(zero0, comb0));

        __m256 flt1 =
            _mm256_castsi256_ps(_mm512_maskz_cvtepi64_epi32(zero1, comb1));

        m = detail::concat(flt0, flt1);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) : m(a.derived().m) { }

    ENOKI_REINTERPRET(int32_t) : m(_mm512_castsi512_ps(a.derived().m)) { }
    ENOKI_REINTERPRET(uint32_t) : m(_mm512_castsi512_ps(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm512_castps512_ps256(m); }
    ENOKI_INLINE Array2 high_() const {
        #if defined(__AVX512DQ__)
            return _mm512_extractf32x8_ps(m, 1);
        #else
            return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(m), 1));
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm512_add_round_ps(m, a.m, (int) Mode); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm512_sub_round_ps(m, a.m, (int) Mode); }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm512_mul_round_ps(m, a.m, (int) Mode); }
    ENOKI_INLINE Derived div_(Arg a) const { return _mm512_div_round_ps(m, a.m, (int) Mode); }

    ENOKI_INLINE Derived or_ (Arg a) const {
        #if defined(__AVX512DQ__)
            return _mm512_or_ps(m, a.m);
        #else
            return _mm512_castsi512_ps(
                _mm512_or_si512(_mm512_castps_si512(m), _mm512_castps_si512(a.m)));
        #endif
    }

    ENOKI_INLINE Derived or_ (Mask a) const {
        return _mm512_mask_mov_ps(m, a.k, _mm512_set1_ps(memcpy_cast<Value>(int32_t(-1))));
    }

    ENOKI_INLINE Derived and_(Arg a) const {
        #if defined(__AVX512DQ__)
            return _mm512_and_ps(m, a.m);
        #else
            return _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(m), _mm512_castps_si512(a.m)));
        #endif
    }

    ENOKI_INLINE Derived and_ (Mask a) const {
        return _mm512_maskz_mov_ps(a.k, m);
    }

    ENOKI_INLINE Derived xor_(Arg a) const {
        #if defined(__AVX512DQ__)
            return _mm512_xor_ps(m, a.m);
        #else
            return _mm512_castsi512_ps(
                _mm512_xor_si512(_mm512_castps_si512(m), _mm512_castps_si512(a.m)));
        #endif
    }

    ENOKI_INLINE Derived xor_ (Mask a) const {
        #if defined(__AVX512DQ__)
            const __m512 v1 = _mm512_set1_ps(memcpy_cast<Value>(int32_t(-1)));
            return _mm512_mask_xor_ps(m, a.k, m, v1);
        #else
            const __m512i v0 = _mm512_castps_si512(m);
            const __m512i v1 = _mm512_set1_epi32(int32_t(-1));
            return _mm512_castsi512_ps(_mm512_mask_xor_epi32(v0, a.k, v0, v1));
        #endif
    }

    ENOKI_INLINE Mask lt_ (Arg a) const { return Mask(_mm512_cmp_ps_mask(m, a.m, _CMP_LT_OQ));  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return Mask(_mm512_cmp_ps_mask(m, a.m, _CMP_GT_OQ));  }
    ENOKI_INLINE Mask le_ (Arg a) const { return Mask(_mm512_cmp_ps_mask(m, a.m, _CMP_LE_OQ));  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return Mask(_mm512_cmp_ps_mask(m, a.m, _CMP_GE_OQ));  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return Mask(_mm512_cmp_ps_mask(m, a.m, _CMP_EQ_OQ));  }
    ENOKI_INLINE Mask neq_(Arg a) const { return Mask(_mm512_cmp_ps_mask(m, a.m, _CMP_NEQ_UQ)); }

    ENOKI_INLINE Derived abs_() const {
        #if defined(__AVX512DQ__)
            return _mm512_andnot_ps(_mm512_set1_ps(-0.f), m);
        #else
            return _mm512_castsi512_ps(
                _mm512_andnot_si512(_mm512_set1_epi32(memcpy_cast<int32_t>(-0.f)),
                                    _mm512_castps_si512(m)));
        #endif
    }

    ENOKI_INLINE Derived min_(Arg b) const { return _mm512_min_ps(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return _mm512_max_ps(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm512_ceil_ps(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm512_floor_ps(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm512_sqrt_round_ps(m, (int) Mode); }

    ENOKI_INLINE Derived round_() const {
        return _mm512_roundscale_ps(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE Derived fmadd_   (Arg b, Arg c) const { return _mm512_fmadd_round_ps   (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fmsub_   (Arg b, Arg c) const { return _mm512_fmsub_round_ps   (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fnmadd_  (Arg b, Arg c) const { return _mm512_fnmadd_round_ps  (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fnmsub_  (Arg b, Arg c) const { return _mm512_fnmsub_round_ps  (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fmsubadd_(Arg b, Arg c) const { return _mm512_fmsubadd_round_ps(m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fmaddsub_(Arg b, Arg c) const { return _mm512_fmaddsub_round_ps(m, b.m, c.m, (int) Mode); }

    static ENOKI_INLINE Derived select_(const Mask &m, const Derived &t,
                                        const Derived &f) {
        return _mm512_mask_blend_ps(m.k, f.m, t.m);
    }

    template <size_t I0, size_t I1, size_t I2, size_t I3, size_t I4, size_t I5,
              size_t I6, size_t I7, size_t I8, size_t I9, size_t I10,
              size_t I11, size_t I12, size_t I13, size_t I14, size_t I15>
    ENOKI_INLINE Derived shuffle_() const {
        const __m512i idx =
            _mm512_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7, I8,
                              I9, I10, I11, I12, I13, I14, I15);
        return _mm512_permutexvar_ps(idx, m);
    }

    ENOKI_INLINE Derived rcp_() const {
        #if defined(__AVX512ER__)
            /* rel err < 2^28, use as is */
            return _mm512_rcp28_ps(m);
        #else
            if (Approx) {
                /* Use best reciprocal approximation available on the current
                   hardware and refine */
                __m512 r = _mm512_rcp14_ps(m); /* rel error < 2^-14 */

                /* Refine using one Newton-Raphson iteration */
                __m512 t0 = _mm512_add_ps(r, r),
                       t1 = _mm512_mul_ps(r, m),
                       ro = r;

                __mmask16 k = _mm512_cmp_ps_mask(t1, t1, _CMP_UNORD_Q);

                r = _mm512_fnmadd_ps(t1, r, t0);

                return _mm512_mask_mov_ps(r, k, ro);
            } else {
                return Base::rcp_();
            }
        #endif
    }

    ENOKI_INLINE Derived rsqrt_() const {
        #if defined(__AVX512ER__)
            /* rel err < 2^28, use as is */
            return _mm512_rsqrt28_ps(m);
        #else
            if (Approx) {
                __m512 r = _mm512_rsqrt14_ps(m); /* rel error < 2^-14 */

                /* Refine using one Newton-Raphson iteration */
                const __m512 c0 = _mm512_set1_ps(0.5f),
                             c1 = _mm512_set1_ps(3.0f);

                __m512 t0 = _mm512_mul_ps(r, c0),
                       t1 = _mm512_mul_ps(r, m),
                       ro = r;

                __mmask16 k = _mm512_cmp_ps_mask(t1, t1, _CMP_UNORD_Q);

                r = _mm512_mul_ps(_mm512_sub_ps(c1, _mm512_mul_ps(t1, r)), t0);

                return _mm512_mask_mov_ps(r, k, ro);
            } else {
                return Base::rsqrt_();
            }
        #endif
    }

#if defined(__AVX512ER__)
    ENOKI_INLINE Derived exp_() const {
        return _mm512_exp2a23_ps(
            _mm512_mul_ps(m, _mm512_set1_ps(1.4426950408889634074f)));
    }
#endif

    ENOKI_INLINE Derived ldexp_(Arg arg) const { return _mm512_scalef_ps(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm512_getmant_ps(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm512_add_ps(_mm512_getexp_ps(m), _mm512_set1_ps(1.f)));
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    //! @}
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const {
        _mm512_store_ps((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), m);
    }
    ENOKI_INLINE void store_(void *ptr, const Mask &mask) const {
        _mm512_mask_store_ps((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), mask.k, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        _mm512_storeu_ps((Value *) ptr, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr, const Mask &mask) const {
        _mm512_mask_storeu_ps((Value *) ptr, mask.k, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return _mm512_load_ps((const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_load_ps(mask.k, (const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return _mm512_loadu_ps((const Value *) ptr);
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_loadu_ps(mask.k, (const Value *) ptr);
    }

    static ENOKI_INLINE Derived zero_() { return _mm512_setzero_ps(); }

#if defined(__AVX512PF__)
    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write)
            _mm512_prefetch_i32scatter_ps(ptr, index.m, Stride, Level);
        else
            _mm512_prefetch_i32gather_ps(index.m, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write)
            _mm512_mask_prefetch_i32scatter_ps(ptr, mask.k, index.m, Stride, Level);
        else
            _mm512_mask_prefetch_i32gather_ps(index.m, mask.k, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write) {
            _mm512_prefetch_i64scatter_ps(ptr, low(index).m, Stride, Level);
            _mm512_prefetch_i64scatter_ps(ptr, high(index).m, Stride, Level);
        } else {
            _mm512_prefetch_i64gather_ps(low(index).m, ptr, Stride, Level);
            _mm512_prefetch_i64gather_ps(high(index).m, ptr, Stride, Level);
        }
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write) {
            _mm512_mask_prefetch_i64scatter_ps(ptr, low(mask).k, low(index).m, Stride, Level);
            _mm512_mask_prefetch_i64scatter_ps(ptr, high(mask).k, high(index).m, Stride, Level);
        } else {
            _mm512_mask_prefetch_i64gather_ps(low(index).m, low(mask).k, ptr, Stride, Level);
            _mm512_mask_prefetch_i64gather_ps(high(index).m, high(mask).k, ptr, Stride, Level);
        }
    }
#endif

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm512_i32gather_ps(index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask.k, index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return detail::concat(
            _mm512_i64gather_ps(low(index).m, (const float *) ptr, Stride),
            _mm512_i64gather_ps(high(index).m, (const float *) ptr, Stride));
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return detail::concat(
            _mm512_mask_i64gather_ps(_mm256_setzero_ps(),  low(mask).k,  low(index).m, (const float *) ptr, Stride),
            _mm512_mask_i64gather_ps(_mm256_setzero_ps(), high(mask).k, high(index).m, (const float *) ptr, Stride));
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i32scatter_ps(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i32scatter_ps(ptr, mask.k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i64scatter_ps(ptr, low(index).m,  low(derived()).m,  Stride);
        _mm512_i64scatter_ps(ptr, high(index).m, high(derived()).m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i64scatter_ps(ptr, low(mask).k,   low(index).m,  low(derived()).m,  Stride);
        _mm512_mask_i64scatter_ps(ptr, high(mask).k, high(index).m, high(derived()).m, Stride);
    }

    ENOKI_INLINE Value extract_(const Mask &mask) const {
        return _mm_cvtss_f32(_mm512_castps512_ps128(_mm512_maskz_compress_ps(mask.k, m)));
    }

    ENOKI_INLINE size_t compress_(float *&ptr, const Mask &mask) const {
        __mmask16 k = mask.k;
        _mm512_storeu_ps(ptr, _mm512_maskz_compress_ps(k, m));
        size_t kn = (size_t) _mm_popcnt_u32(k);
        ptr += kn;
        return kn;
    }

#if defined(__AVX512CD__)
    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int32_t)
    static ENOKI_INLINE void transform_(void *mem, Index index,
                                        const Func &func,
                                        const Args &... args) {
        transform_masked_<Stride>(mem, index, Mask(true), func, args...);
    }

    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int32_t)
    static ENOKI_INLINE void transform_masked_(void *mem, Index index, Mask mask,
                                               const Func &func,
                                               const Args &... args) {
        Derived values = _mm512_mask_i32gather_ps(
            _mm512_undefined_ps(), mask.k, index.m, mem, (int) Stride);

        index.m = _mm512_mask_mov_epi32(_mm512_set1_epi32(-1), mask.k, index.m);

        __m512i conflicts = _mm512_conflict_epi32(index.m);
        __m512i perm_idx  = _mm512_sub_epi32(_mm512_set1_epi32(31), _mm512_lzcnt_epi32(conflicts));
        __mmask16 todo    = _mm512_mask_test_epi32_mask(mask.k, conflicts, _mm512_set1_epi32(-1));

        func(values, args...);

        while (ENOKI_UNLIKELY(!_mm512_kortestz(todo, todo))) {
            __mmask16 cur = _mm512_mask_testn_epi32_mask(
                todo, conflicts, _mm512_broadcastmw_epi32(todo));
            values.m = _mm512_mask_permutexvar_ps(values.m, cur, perm_idx, values.m);

            __m512 backup(values.m);
            func(values, args...);

            values.m = _mm512_mask_mov_ps(backup, cur, values.m);
            todo = _mm512_kxor(todo, cur);
        }

        _mm512_mask_i32scatter_ps(mem, mask.k, index.m, values.m, (int) Stride);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Masked versions of key operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE void massign_(const Derived &a, const Mask &mask) { m = _mm512_mask_mov_ps(m, mask.k, a.m); }
    ENOKI_INLINE void madd_   (const Derived &a, const Mask &mask) { m = _mm512_mask_add_round_ps(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void msub_   (const Derived &a, const Mask &mask) { m = _mm512_mask_sub_round_ps(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void mmul_   (const Derived &a, const Mask &mask) { m = _mm512_mask_mul_round_ps(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void mdiv_   (const Derived &a, const Mask &mask) { m = _mm512_mask_div_round_ps(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void mor_    (const Derived &a, const Mask &mask) {
        #if defined(__AVX512DQ__)
            m = _mm512_mask_or_ps(m, mask.k, m, a.m);
        #else
            m = _mm512_castsi512_ps(
                _mm512_or_si512(_mm512_castps_si512(m), mask.k,
                                _mm512_castps_si512(m), _mm512_castps_si512(a.m)));
        #endif
    }

    ENOKI_INLINE void mand_   (const Derived &a, const Mask &mask) {
        #if defined(__AVX512DQ__)
            m = _mm512_mask_and_ps(m, mask.k, m, a.m);
        #else
            m = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(m), mask.k,
                                                     _mm512_castps_si512(m),
                                                     _mm512_castps_si512(a.m)));
        #endif
    }

    ENOKI_INLINE void mxor_   (const Derived &a, const Mask &mask) {
        #if defined(__AVX512DQ__)
            m = _mm512_mask_xor_ps(m, mask.k, m, a.m);
        #else
            m = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(m), mask.k,
                                                     _mm512_castps_si512(m),
                                                     _mm512_castps_si512(a.m)));
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using AVX512 intrinsics (double precision)
template <bool Approx, RoundingMode Mode, typename Derived> struct alignas(64)
    StaticArrayImpl<double, 8, Approx, Mode, Derived>
    : StaticArrayBase<double, 8, Approx, Mode, Derived> {

    ENOKI_NATIVE_ARRAY(double, 8, Approx, __m512d, Mode)
    using Mask = detail::KMask<__mmask8>;

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value) : m(_mm512_set1_pd(value)) { }
    ENOKI_INLINE StaticArrayImpl(Value f0, Value f1, Value f2, Value f3,
                                 Value f4, Value f5, Value f6, Value f7)
        : m(_mm512_setr_pd(f0, f1, f2, f3, f4, f5, f6, f7)) { }


    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(half)
        : m(_mm512_cvtps_pd(
              _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) a.data())))) { }

    ENOKI_CONVERT(float) : m(_mm512_cvtps_pd(a.derived().m)) { }

    ENOKI_CONVERT(double) : m(a.derived().m) { }

    ENOKI_CONVERT(int32_t) : m(_mm512_cvtepi32_pd(a.derived().m)) { }

    ENOKI_CONVERT(uint32_t) : m(_mm512_cvtepu32_pd(a.derived().m)) { }

#if defined(__AVX512DQ__)
    ENOKI_CONVERT(int64_t)
        : m(_mm512_cvt_roundepi64_pd(a.derived().m, (int) Mode)) { }

    ENOKI_CONVERT(uint64_t)
        : m(_mm512_cvt_roundepu64_pd(a.derived().m, (int) Mode)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(double) : m(a.derived().m) { }

    ENOKI_REINTERPRET(int64_t) : m(_mm512_castsi512_pd(a.derived().m)) { }
    ENOKI_REINTERPRET(uint64_t) : m(_mm512_castsi512_pd(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm512_castpd512_pd256(m); }
    ENOKI_INLINE Array2 high_() const { return _mm512_extractf64x4_pd(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm512_add_round_pd(m, a.m, (int) Mode); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm512_sub_round_pd(m, a.m, (int) Mode); }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm512_mul_round_pd(m, a.m, (int) Mode); }
    ENOKI_INLINE Derived div_(Arg a) const { return _mm512_div_round_pd(m, a.m, (int) Mode); }
    ENOKI_INLINE Derived or_ (Arg a) const {
        #if defined(__AVX512DQ__)
            return _mm512_or_pd(m, a.m);
        #else
            return _mm512_castsi512_pd(
                _mm512_or_si512(_mm512_castpd_si512(m), _mm512_castpd_si512(a.m)));
        #endif
    }

    ENOKI_INLINE Derived or_ (Mask a) const {
        return _mm512_mask_mov_pd(m, a.k, _mm512_set1_pd(memcpy_cast<Value>(int64_t(-1))));
    }

    ENOKI_INLINE Derived and_(Arg a) const {
        #if defined(__AVX512DQ__)
            return _mm512_and_pd(m, a.m);
        #else
            return _mm512_castsi512_pd(
                _mm512_and_si512(_mm512_castpd_si512(m), _mm512_castpd_si512(a.m)));
        #endif
    }

    ENOKI_INLINE Derived and_ (Mask a) const {
        return _mm512_maskz_mov_pd(a.k, m);
    }

    ENOKI_INLINE Derived xor_(Arg a) const {
        #if defined(__AVX512DQ__)
            return _mm512_xor_pd(m, a.m);
        #else
            return _mm512_castsi512_pd(
                _mm512_xor_si512(_mm512_castpd_si512(m), _mm512_castpd_si512(a.m)));
        #endif
    }

    ENOKI_INLINE Derived xor_ (Mask a) const {
        #if defined(__AVX512DQ__)
            const __m512d v1 = _mm512_set1_pd(memcpy_cast<Value>(int64_t(-1)));
            return _mm512_mask_xor_pd(m, a.k, m, v1);
        #else
            const __m512i v0 = _mm512_castpd_si512(m);
            const __m512i v1 = _mm512_set1_epi64(int64_t(-1));
            return _mm512_castsi512_pd(_mm512_mask_xor_epi32(v0, a.k, v0, v1));
        #endif
    }

    ENOKI_INLINE Mask lt_ (Arg a) const { return Mask(_mm512_cmp_pd_mask(m, a.m, _CMP_LT_OQ));  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return Mask(_mm512_cmp_pd_mask(m, a.m, _CMP_GT_OQ));  }
    ENOKI_INLINE Mask le_ (Arg a) const { return Mask(_mm512_cmp_pd_mask(m, a.m, _CMP_LE_OQ));  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return Mask(_mm512_cmp_pd_mask(m, a.m, _CMP_GE_OQ));  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return Mask(_mm512_cmp_pd_mask(m, a.m, _CMP_EQ_OQ));  }
    ENOKI_INLINE Mask neq_(Arg a) const { return Mask(_mm512_cmp_pd_mask(m, a.m, _CMP_NEQ_UQ)); }

    ENOKI_INLINE Derived abs_() const {
        #if defined(__AVX512DQ__)
            return _mm512_andnot_pd(_mm512_set1_pd(-0.), m);
        #else
            return _mm512_castsi512_pd(
                _mm512_andnot_si512(_mm512_set1_epi64(memcpy_cast<int64_t>(-0.)),
                                    _mm512_castpd_si512(m)));
        #endif
    }

    ENOKI_INLINE Derived min_(Arg b) const { return _mm512_min_pd(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return _mm512_max_pd(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm512_ceil_pd(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm512_floor_pd(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm512_sqrt_round_pd(m, (int) Mode); }

    ENOKI_INLINE Derived round_() const {
        return _mm512_roundscale_pd(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE Derived fmadd_   (Arg b, Arg c) const { return _mm512_fmadd_round_pd   (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fmsub_   (Arg b, Arg c) const { return _mm512_fmsub_round_pd   (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fnmadd_  (Arg b, Arg c) const { return _mm512_fnmadd_round_pd  (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fnmsub_  (Arg b, Arg c) const { return _mm512_fnmsub_round_pd  (m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fmsubadd_(Arg b, Arg c) const { return _mm512_fmsubadd_round_pd(m, b.m, c.m, (int) Mode); }
    ENOKI_INLINE Derived fmaddsub_(Arg b, Arg c) const { return _mm512_fmaddsub_round_pd(m, b.m, c.m, (int) Mode); }

    static ENOKI_INLINE Derived select_(const Mask &m, const Derived &t,
                                        const Derived &f) {
        return _mm512_mask_blend_pd(m.k, f.m, t.m);
    }

    template <size_t I0, size_t I1, size_t I2, size_t I3, size_t I4, size_t I5,
              size_t I6, size_t I7>
    ENOKI_INLINE Derived shuffle_() const {
        const __m512i idx =
            _mm512_setr_epi64(I0, I1, I2, I3, I4, I5, I6, I7);
        return _mm512_permutexvar_pd(idx, m);
    }

    ENOKI_INLINE Derived rcp_() const {
        if (Approx) {
            /* Use best reciprocal approximation available on the current
               hardware and refine */
            __m512d r;

            #if defined(__AVX512ER__)
                r = _mm512_rcp28_pd(m); /* rel err < 2^28 */
            #else
                r = _mm512_rcp14_pd(m); /* rel error < 2^-14 */
            #endif

            __m512d ro = r, t0, t1;
            __mmask8 k;

            /* Refine using 1-2 Newton-Raphson iterations */
            ENOKI_UNROLL for (int i = 0; i < (has_avx512er ? 1 : 2); ++i) {
                t0 = _mm512_add_pd(r, r);
                t1 = _mm512_mul_pd(r, m);

                if (i == 0)
                    k = _mm512_cmp_pd_mask(t1, t1, _CMP_UNORD_Q);

                r = _mm512_fnmadd_pd(t1, r, t0);
            }

            return _mm512_mask_mov_ps(r, k, ro);
        } else {
            return Base::rcp_();
        }
    }

    ENOKI_INLINE Derived rsqrt_() const {
        if (Approx) {
            /* Use best reciprocal square root approximation available
               on the current hardware and refine */
            __m512d r;
            #if defined(__AVX512ER__)
                r = _mm512_rsqrt28_pd(m); /* rel err < 2^28 */
            #else
                r = _mm512_rsqrt14_pd(m); /* rel error < 2^-14 */
            #endif

            const __m512d c0 = _mm512_set1_pd(0.5),
                          c1 = _mm512_set1_pd(3.0);

            __m512d ro = r, t0, t1;
            __mmask8 k;

            /* Refine using 1-2 Newton-Raphson iterations */
            ENOKI_UNROLL for (int i = 0; i < (has_avx512er ? 1 : 2); ++i) {
                t0 = _mm512_mul_pd(r, c0);
                t1 = _mm512_mul_pd(r, m);

                if (i == 0)
                    k = _mm512_cmp_pd_mask(t1, t1, _CMP_UNORD_Q);

                r = _mm512_mul_pd(_mm512_fnmadd_pd(t1, r, c1), t0);
            }

            return _mm512_mask_mov_ps(r, k, ro);
        } else {
            return Base::rsqrt_();
        }
    }


#if defined(__AVX512ER__)
    ENOKI_INLINE Derived exp_() const {
        if (Approx) {
            return _mm512_exp2a23_pd(
                _mm512_mul_pd(m, _mm512_set1_pd(1.4426950408889634074f)));
        } else {
            return Base::exp_();
        }
    }
#endif

    ENOKI_INLINE Derived ldexp_(Arg arg) const { return _mm512_scalef_pd(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm512_getmant_pd(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm512_add_pd(_mm512_getexp_pd(m), _mm512_set1_pd(1.f)));
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    //! @}
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const {
        _mm512_store_pd((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), m);
    }
    ENOKI_INLINE void store_(void *ptr, const Mask &mask) const {
        _mm512_mask_store_pd((Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), mask.k, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        _mm512_storeu_pd((Value *) ptr, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr, const Mask &mask) const {
        _mm512_mask_storeu_pd((Value *) ptr, mask.k, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return _mm512_load_pd((const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_load_pd(mask.k, (const Value *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return _mm512_loadu_pd((const Value *) ptr);
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_loadu_pd(mask.k, (const Value *) ptr);
    }

    static ENOKI_INLINE Derived zero_() { return _mm512_setzero_pd(); }

#if defined(__AVX512PF__)
    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write)
            _mm512_prefetch_i32scatter_pd(ptr, index.m, Stride, Level);
        else
            _mm512_prefetch_i32gather_pd(index.m, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write)
            _mm512_mask_prefetch_i32scatter_pd(ptr, mask.k, index.m, Stride, Level);
        else
            _mm512_mask_prefetch_i32gather_pd(index.m, mask.k, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write)
            _mm512_prefetch_i64scatter_pd(ptr, index.m, Stride, Level);
        else
            _mm512_prefetch_i64gather_pd(index.m, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write)
            _mm512_mask_prefetch_i64scatter_pd(ptr, mask.k, index.m, Stride, Level);
        else
            _mm512_mask_prefetch_i64gather_pd(index.m, mask.k, ptr, Stride, Level);
    }
#endif

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm512_i32gather_pd(index.m, (const double *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm512_mask_i32gather_pd(_mm512_setzero_pd(), mask.k, index.m, (const double *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm512_i64gather_pd(index.m, (const double *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm512_mask_i64gather_pd(_mm512_setzero_pd(), mask.k, index.m, (const double *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i32scatter_pd(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i32scatter_pd(ptr, mask.k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i64scatter_pd(ptr, index.m, derived().m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i64scatter_pd(ptr, mask.k, index.m, m, Stride);
    }

    ENOKI_INLINE Value extract_(const Mask &mask) const {
        return _mm_cvtsd_f64(_mm512_castpd512_pd128(_mm512_maskz_compress_pd(mask.k, m)));
    }

    ENOKI_INLINE size_t compress_(double *&ptr, const Mask &mask) const {
        __mmask8 k = mask.k;
        _mm512_storeu_pd(ptr, _mm512_maskz_compress_pd(k, m));
        size_t kn = (size_t) _mm_popcnt_u32(k);
        ptr += kn;
        return kn;
    }

#if defined(__AVX512CD__)
    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int64_t)
    static ENOKI_INLINE void transform_(void *mem, Index index,
                                        const Func &func,
                                        const Args &... args) {
        transform_masked_<Stride>(mem, index, Mask(true), func, args...);
    }

    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int64_t)
    static ENOKI_INLINE void transform_masked_(void *mem, Index index,
                                               Mask mask, const Func &func,
                                               const Args &... args) {
        Derived values = _mm512_mask_i64gather_pd(
            _mm512_undefined_pd(), mask.k, index.m, mem, (int) Stride);

        index.m = _mm512_mask_mov_epi64(_mm512_set1_epi64(-1), mask.k, index.m);

        __m512i conflicts = _mm512_conflict_epi64(index.m);
        __m512i perm_idx  = _mm512_sub_epi64(_mm512_set1_epi64(63), _mm512_lzcnt_epi64(conflicts));
        __mmask8 todo     = _mm512_mask_test_epi64_mask(mask.k, conflicts, _mm512_set1_epi64(-1));

        func(values, args...);

        while (ENOKI_UNLIKELY(todo)) {
            __mmask8 cur = _mm512_mask_testn_epi64_mask(
                todo, conflicts, _mm512_broadcastmb_epi64(todo));
            values.m = _mm512_mask_permutexvar_pd(values.m, cur, perm_idx, values.m);

            __m512d backup(values.m);
            func(values, args...);

            values.m = _mm512_mask_mov_pd(backup, cur, values.m);
            todo ^= cur;
        }

        _mm512_mask_i64scatter_pd(mem, mask.k, index.m, values.m, (int) Stride);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Masked versions of key operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE void massign_(const Derived &a, const Mask &mask) { m = _mm512_mask_mov_pd(m, mask.k, a.m); }
    ENOKI_INLINE void madd_   (const Derived &a, const Mask &mask) { m = _mm512_mask_add_round_pd(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void msub_   (const Derived &a, const Mask &mask) { m = _mm512_mask_sub_round_pd(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void mmul_   (const Derived &a, const Mask &mask) { m = _mm512_mask_mul_round_pd(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void mdiv_   (const Derived &a, const Mask &mask) { m = _mm512_mask_div_round_pd(m, mask.k, m, a.m, (int) Mode); }
    ENOKI_INLINE void mor_    (const Derived &a, const Mask &mask) {
        #if defined(__AVX512DQ__)
            m = _mm512_mask_or_pd(m, mask.k, m, a.m);
        #else
            m = _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(m), mask.k,
                                                    _mm512_castpd_si512(m),
                                                    _mm512_castpd_si512(a.m)));
        #endif
    }

    ENOKI_INLINE void mand_   (const Derived &a, const Mask &mask) {
        #if defined(__AVX512DQ__)
            m = _mm512_mask_and_pd(m, mask.k, m, a.m);
        #else
            m = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(m), mask.k,
                                                     _mm512_castpd_si512(m),
                                                     _mm512_castpd_si512(a.m)));
        #endif
    }

    ENOKI_INLINE void mxor_   (const Derived &a, const Mask &mask) {
        #if defined(__AVX512DQ__)
            m = _mm512_mask_xor_pd(m, mask.k, m, a.m);
        #else
            m = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(m), mask.k,
                                                     _mm512_castpd_si512(m),
                                                     _mm512_castpd_si512(a.m)));
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using AVX512 intrinsics (32 bit integers)
template <typename Value_, typename Derived> struct alignas(64)
    StaticArrayImpl<Value_, 16, false, RoundingMode::Default, Derived, detail::is_int32_t<Value_>>
    : StaticArrayBase<Value_, 16, false, RoundingMode::Default, Derived> {

    ENOKI_NATIVE_ARRAY(Value_, 16, false, __m512i, RoundingMode::Default)
    using Mask = detail::KMask<__mmask16>;

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value) : m(_mm512_set1_epi32((int32_t) value)) { }
    ENOKI_INLINE StaticArrayImpl(Value f0, Value f1, Value f2, Value f3,
                                 Value f4, Value f5, Value f6, Value f7,
                                 Value f8, Value f9, Value f10, Value f11,
                                 Value f12, Value f13, Value f14, Value f15)
        : m(_mm512_setr_epi32(
              (int32_t) f0, (int32_t) f1, (int32_t) f2, (int32_t) f3,
              (int32_t) f4, (int32_t) f5, (int32_t) f6, (int32_t) f7,
              (int32_t) f8, (int32_t) f9, (int32_t) f10, (int32_t) f11,
              (int32_t) f12, (int32_t) f13, (int32_t) f14, (int32_t) f15)) { }

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(int32_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint32_t) : m(a.derived().m) { }

    ENOKI_CONVERT(float) {
        if (std::is_signed<Value>::value) {
            m = _mm512_cvttps_epi32(a.derived().m);
        } else {
            m = _mm512_cvttps_epu32(a.derived().m);
        }
    }

    ENOKI_CONVERT(double) {
        if (std::is_signed<Value>::value) {
            m = detail::concat(_mm512_cvttpd_epi32(low(a).m),
                               _mm512_cvttpd_epi32(high(a).m));
        } else {
            m = detail::concat(_mm512_cvttpd_epu32(low(a).m),
                               _mm512_cvttpd_epu32(high(a).m));
        }
    }

    ENOKI_CONVERT(int64_t)
        : m(detail::concat(_mm512_cvtepi64_epi32(low(a).m),
                           _mm512_cvtepi64_epi32(high(a).m))) { }

    ENOKI_CONVERT(uint64_t)
        : m(detail::concat(_mm512_cvtepi64_epi32(low(a).m),
                           _mm512_cvtepi64_epi32(high(a).m))) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) : m(_mm512_castps_si512(a.derived().m)) { }
    ENOKI_REINTERPRET(int32_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint32_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm512_castsi512_si256(m); }
    ENOKI_INLINE Array2 high_() const {
        #if defined(__AVX512DQ__)
            return _mm512_extracti32x8_epi32(m, 1);
        #else
            return _mm512_extracti64x4_epi64(m, 1);
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm512_add_epi32(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm512_sub_epi32(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm512_mullo_epi32(m, a.m); }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm512_or_epi32(m, a.m); }

    ENOKI_INLINE Derived or_ (Mask a) const {
        return _mm512_mask_mov_epi32(m, a.k, _mm512_set1_epi32(int32_t(-1)));
    }

    ENOKI_INLINE Derived and_(Arg a) const { return _mm512_and_epi32(m, a.m); }

    ENOKI_INLINE Derived and_ (Mask a) const {
        return _mm512_maskz_mov_epi32(a.k, m);
    }

    ENOKI_INLINE Derived xor_(Arg a) const { return _mm512_xor_epi32(m, a.m); }

    ENOKI_INLINE Derived xor_ (Mask a) const {
        return _mm512_mask_xor_epi32(m, a.k, m, _mm512_set1_epi32(int32_t(-1)));
    }

    template <size_t k> ENOKI_INLINE Derived sli_() const {
        return _mm512_slli_epi32(m, (int) k);
    }

    template <size_t k> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Value>::value)
            return _mm512_srai_epi32(m, (int) k);
        else
            return _mm512_srli_epi32(m, (int) k);
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        #if 0
            return _mm512_sll_epi32(m, _mm_set1_epi64x((long long) k));
        #else
            /* This is not strictly correct (k may not be a compile-time constant),
               but all targeted compilers figure it out and generate better code */
            return _mm512_slli_epi32(m, (int) k);
        #endif
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        #if 0
            if (std::is_signed<Value>::value)
                return _mm512_sra_epi32(m, _mm_set1_epi64x((long long) k));
            else
                return _mm512_srl_epi32(m, _mm_set1_epi64x((long long) k));
        #else
            /* This is not strictly correct (k may not be a compile-time constant),
               but all targeted compilers figure it out and generate better code */
            if (std::is_signed<Value>::value)
                return _mm512_srai_epi32(m, (int) k);
            else
                return _mm512_srli_epi32(m, (int) k);
        #endif
    }

    ENOKI_INLINE Derived slv_(Arg k) const {
        return _mm512_sllv_epi32(m, k.m);
    }

    ENOKI_INLINE Derived srv_(Arg k) const {
        if (std::is_signed<Value>::value)
            return _mm512_srav_epi32(m, k.m);
        else
            return _mm512_srlv_epi32(m, k.m);
    }

    ENOKI_INLINE Derived rolv_(Arg k) const { return _mm512_rolv_epi32(m, k.m); }
    ENOKI_INLINE Derived rorv_(Arg k) const { return _mm512_rorv_epi32(m, k.m); }

    ENOKI_INLINE Derived rol_(size_t k) const { return rolv_(_mm512_set1_epi32((int32_t) k)); }
    ENOKI_INLINE Derived ror_(size_t k) const { return rorv_(_mm512_set1_epi32((int32_t) k)); }

    template <size_t Imm>
    ENOKI_INLINE Derived roli_() const { return _mm512_rol_epi32(m, (int) Imm); }

    template <size_t Imm>
    ENOKI_INLINE Derived rori_() const { return _mm512_ror_epi32(m, (int) Imm); }

    ENOKI_INLINE Mask lt_ (Arg a) const { return Mask(_mm512_cmp_epi32_mask(m, a.m, _MM_CMPINT_LT));  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return Mask(_mm512_cmp_epi32_mask(m, a.m, _MM_CMPINT_GT));  }
    ENOKI_INLINE Mask le_ (Arg a) const { return Mask(_mm512_cmp_epi32_mask(m, a.m, _MM_CMPINT_LE));  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return Mask(_mm512_cmp_epi32_mask(m, a.m, _MM_CMPINT_GE));  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return Mask(_mm512_cmp_epi32_mask(m, a.m, _MM_CMPINT_EQ));  }
    ENOKI_INLINE Mask neq_(Arg a) const { return Mask(_mm512_cmp_epi32_mask(m, a.m, _MM_CMPINT_NE)); }

    ENOKI_INLINE Derived min_(Arg a) const {
        if (std::is_signed<Value>::value)
            return _mm512_min_epi32(a.m, m);
        else
            return _mm512_min_epu32(a.m, m);
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        if (std::is_signed<Value>::value)
            return _mm512_max_epi32(a.m, m);
        else
            return _mm512_max_epu32(a.m, m);
    }

    ENOKI_INLINE Derived abs_() const {
        return std::is_signed<Value>::value ? _mm512_abs_epi32(m) : m;
    }

    static ENOKI_INLINE Derived select_(const Mask &m, const Derived &t,
                                        const Derived &f) {
        return _mm512_mask_blend_epi32(m.k, f.m, t.m);
    }

    template <size_t I0, size_t I1, size_t I2, size_t I3, size_t I4, size_t I5,
              size_t I6, size_t I7, size_t I8, size_t I9, size_t I10,
              size_t I11, size_t I12, size_t I13, size_t I14, size_t I15>
    ENOKI_INLINE Derived shuffle_() const {
        const __m512i idx =
            _mm512_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7, I8,
                              I9, I10, I11, I12, I13, I14, I15);
        return _mm512_permutexvar_epi32(idx, m);
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        const Mask blend(__mmask16(0b0101010101010101));

        if (std::is_signed<Value>::value) {
            Derived even(_mm512_srli_epi64(_mm512_mul_epi32(m, a.m), 32));
            Derived odd(_mm512_mul_epi32(_mm512_srli_epi64(m, 32),
                                         _mm512_srli_epi64(a.m, 32)));
            return select(blend, even, odd);
        } else {
            Derived even(_mm512_srli_epi64(_mm512_mul_epu32(m, a.m), 32));
            Derived odd(_mm512_mul_epu32(_mm512_srli_epi64(m, 32),
                                         _mm512_srli_epi64(a.m, 32)));
            return select(blend, even, odd);
        }
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    //! @}
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const {
        _mm512_store_si512((__m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), m);
    }
    ENOKI_INLINE void store_(void *ptr, const Mask &mask) const {
        _mm512_mask_store_epi32((__m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), mask.k, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        _mm512_storeu_si512((__m512i *) ptr, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr, const Mask &mask) const {
        _mm512_mask_storeu_epi32((__m512i *) ptr, mask.k, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return _mm512_load_si512((const __m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_load_epi32(mask.k, (const __m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return _mm512_loadu_si512((const __m512i *) ptr);
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_loadu_epi32(mask.k, (const __m512i *) ptr);
    }

    static ENOKI_INLINE Derived zero_() { return _mm512_setzero_si512(); }

#if defined(__AVX512PF__)
    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write)
            _mm512_prefetch_i32scatter_ps(ptr, index.m, Stride, Level);
        else
            _mm512_prefetch_i32gather_ps(index.m, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write)
            _mm512_mask_prefetch_i32scatter_ps(ptr, mask.k, index.m, Stride, Level);
        else
            _mm512_mask_prefetch_i32gather_ps(index.m, mask.k, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write) {
            _mm512_prefetch_i64scatter_ps(ptr, low(index).m, Stride, Level);
            _mm512_prefetch_i64scatter_ps(ptr, high(index).m, Stride, Level);
        } else {
            _mm512_prefetch_i64gather_ps(low(index).m, ptr, Stride, Level);
            _mm512_prefetch_i64gather_ps(high(index).m, ptr, Stride, Level);
        }
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write) {
            _mm512_mask_prefetch_i64scatter_ps(ptr, low(mask).k, low(index).m, Stride, Level);
            _mm512_mask_prefetch_i64scatter_ps(ptr, high(mask).k, high(index).m, Stride, Level);
        } else {
            _mm512_mask_prefetch_i64gather_ps(low(index).m, low(mask).k, ptr, Stride, Level);
            _mm512_mask_prefetch_i64gather_ps(high(index).m, high(mask).k, ptr, Stride, Level);
        }
    }
#endif

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm512_i32gather_epi32(index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask.k, index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return detail::concat(
            _mm512_i64gather_epi32(low(index).m, (const float *) ptr, Stride),
            _mm512_i64gather_epi32(high(index).m, (const float *) ptr, Stride));
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return detail::concat(
            _mm512_mask_i64gather_epi32(_mm256_setzero_si256(),  low(mask).k,  low(index).m, (const float *) ptr, Stride),
            _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), high(mask).k, high(index).m, (const float *) ptr, Stride));
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i32scatter_epi32(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i32scatter_epi32(ptr, mask.k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i64scatter_epi32(ptr, low(index).m,  low(derived()).m,  Stride);
        _mm512_i64scatter_epi32(ptr, high(index).m, high(derived()).m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i64scatter_epi32(ptr, low(mask).k,   low(index).m,  low(derived()).m,  Stride);
        _mm512_mask_i64scatter_epi32(ptr, high(mask).k, high(index).m, high(derived()).m, Stride);
    }

    ENOKI_INLINE Value extract_(const Mask &mask) const {
        return (Value) _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_maskz_compress_epi32(mask.k, m)));
    }

    template <typename T>
    ENOKI_INLINE size_t compress_(T *&ptr, const Mask &mask) const {
        __mmask16 k = mask.k;
        _mm512_storeu_si512((__m512i *) ptr, _mm512_maskz_compress_epi32(k, m));
        size_t kn = (size_t) _mm_popcnt_u32(k);
        ptr += kn;
        return kn;
    }

#if defined(__AVX512CD__)
    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int32_t)
    static ENOKI_INLINE void transform_(void *mem, Index index,
                                        const Func &func,
                                        const Args &... args) {
        transform_masked_<Stride>(mem, index, Mask(true), func, args...);
    }


    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int32_t)
    static ENOKI_INLINE void transform_masked_(void *mem, Index index,
                                               Mask mask, const Func &func,
                                               const Args &... args) {
        Derived values = _mm512_mask_i32gather_epi32(
            _mm512_undefined_epi32(), mask.k, index.m, mem, (int) Stride);

        index.m = _mm512_mask_mov_epi32(_mm512_set1_epi32(-1), mask.k, index.m);

        __m512i conflicts = _mm512_conflict_epi32(index.m);
        __m512i perm_idx  = _mm512_sub_epi32(_mm512_set1_epi32(31), _mm512_lzcnt_epi32(conflicts));
        __mmask16 todo    = _mm512_mask_test_epi32_mask(mask.k, conflicts, _mm512_set1_epi32(-1));

        func(values, args...);

        while (ENOKI_UNLIKELY(!_mm512_kortestz(todo, todo))) {
            __mmask16 cur = _mm512_mask_testn_epi32_mask(
                todo, conflicts, _mm512_broadcastmw_epi32(todo));
            values.m = _mm512_mask_permutexvar_epi32(values.m, cur, perm_idx, values.m);

            __m512i backup(values.m);
            func(values, args...);

            values.m = _mm512_mask_mov_epi32(backup, cur, values.m);
            todo = _mm512_kxor(todo, cur);
        }

        _mm512_mask_i32scatter_epi32(mem, mask.k, index.m, values.m, (int) Stride);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Masked versions of key operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE void massign_(const Derived &a, const Mask &mask) { m = _mm512_mask_mov_epi32(m, mask.k, a.m); }
    ENOKI_INLINE void madd_   (const Derived &a, const Mask &mask) { m = _mm512_mask_add_epi32(m, mask.k, m, a.m); }
    ENOKI_INLINE void msub_   (const Derived &a, const Mask &mask) { m = _mm512_mask_sub_epi32(m, mask.k, m, a.m); }
    ENOKI_INLINE void mmul_   (const Derived &a, const Mask &mask) { m = _mm512_mask_mullo_epi32(m, mask.k, m, a.m); }
    ENOKI_INLINE void mor_    (const Derived &a, const Mask &mask) { m = _mm512_mask_or_epi32(m, mask.k, m, a.m); }
    ENOKI_INLINE void mand_   (const Derived &a, const Mask &mask) { m = _mm512_mask_and_epi32(m, mask.k, m, a.m); }
    ENOKI_INLINE void mxor_   (const Derived &a, const Mask &mask) { m = _mm512_mask_xor_epi32(m, mask.k, m, a.m); }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using AVX512 intrinsics (64 bit integers)
template <typename Value_, typename Derived> struct alignas(64)
    StaticArrayImpl<Value_, 8, false, RoundingMode::Default, Derived, detail::is_int64_t<Value_>>
    : StaticArrayBase<Value_, 8, false, RoundingMode::Default, Derived> {

    ENOKI_NATIVE_ARRAY(Value_, 8, false, __m512i, RoundingMode::Default)
    using Mask = detail::KMask<__mmask8>;

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value) : m(_mm512_set1_epi64((long long) value)) { }
    ENOKI_INLINE StaticArrayImpl(Value f0, Value f1, Value f2, Value f3,
                                 Value f4, Value f5, Value f6, Value f7)
        : m(_mm512_setr_epi64((long long) f0, (long long) f1, (long long) f2,
                              (long long) f3, (long long) f4, (long long) f5,
                              (long long) f6, (long long) f7)) { }

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

#if defined(__AVX512DQ__)
    ENOKI_CONVERT(float) {
        if (std::is_signed<Value>::value) {
            m = _mm512_cvttps_epi64(a.derived().m);
        } else {
            m = _mm512_cvttps_epu64(a.derived().m);
        }
    }
#endif

    ENOKI_CONVERT(int32_t)
        : m(_mm512_cvtepi32_epi64(a.derived().m)) { }

    ENOKI_CONVERT(uint32_t)
        : m(_mm512_cvtepu32_epi64(a.derived().m)) { }

#if defined(__AVX512DQ__)
    ENOKI_CONVERT(double) {
        if (std::is_signed<Value>::value)
            m = _mm512_cvttpd_epi64(a.derived().m);
        else
            m = _mm512_cvttpd_epu64(a.derived().m);
    }
#endif

    ENOKI_CONVERT(int64_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(double) : m(_mm512_castpd_si512(a.derived().m)) { }
    ENOKI_REINTERPRET(int64_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm512_castsi512_si256(m); }
    ENOKI_INLINE Array2 high_() const { return _mm512_extracti64x4_epi64(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm512_add_epi64(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm512_sub_epi64(m, a.m); }

    ENOKI_INLINE Derived mul_(Arg a) const {
        #if defined(__AVX512DQ__) && defined(__AVX512VL__)
            return _mm512_mullo_epi64(m, a.m);
        #else
            __m512i h0    = _mm512_srli_epi64(m, 32);
            __m512i h1    = _mm512_srli_epi64(a.m, 32);
            __m512i low   = _mm512_mul_epu32(m, a.m);
            __m512i mix0  = _mm512_mul_epu32(m, h1);
            __m512i mix1  = _mm512_mul_epu32(h0, a.m);
            __m512i mix   = _mm512_add_epi64(mix0, mix1);
            __m512i mix_s = _mm512_slli_epi64(mix, 32);
            return  _mm512_add_epi64(mix_s, low);
        #endif
    }

    ENOKI_INLINE Derived mulhi_(Arg b) const {
        if (std::is_unsigned<Value>::value) {
            const __m512i low_bits = _mm512_set1_epi64(0xffffffffu);
            __m512i al = m, bl = b.m;
            __m512i ah = _mm512_srli_epi64(al, 32);
            __m512i bh = _mm512_srli_epi64(bl, 32);

            // 4x unsigned 32x32->64 bit multiplication
            __m512i albl = _mm512_mul_epu32(al, bl);
            __m512i albh = _mm512_mul_epu32(al, bh);
            __m512i ahbl = _mm512_mul_epu32(ah, bl);
            __m512i ahbh = _mm512_mul_epu32(ah, bh);

            // Calculate a possible carry from the low bits of the multiplication.
            __m512i carry = _mm512_add_epi64(
                _mm512_srli_epi64(albl, 32),
                _mm512_add_epi64(_mm512_and_epi64(albh, low_bits),
                                 _mm512_and_epi64(ahbl, low_bits)));

            __m512i s0 = _mm512_add_epi64(ahbh, _mm512_srli_epi64(carry, 32));
            __m512i s1 = _mm512_add_epi64(_mm512_srli_epi64(albh, 32),
                                          _mm512_srli_epi64(ahbl, 32));

            return _mm512_add_epi64(s0, s1);
        } else {
            const Derived mask(0xffffffff);
            const Derived a = derived();
            Derived ah = sri<32>(a), bh = sri<32>(b),
                    al = a & mask, bl = b & mask;

            Derived albl_hi = _mm512_srli_epi64(_mm512_mul_epu32(m, b.m), 32);

            Derived t = ah * bl + albl_hi;
            Derived w1 = al * bh + (t & mask);

            return ah * bh + sri<32>(t) + sri<32>(w1);
        }
    }

    ENOKI_INLINE Derived or_ (Arg a) const { return _mm512_or_epi64(m, a.m); }

    ENOKI_INLINE Derived or_ (Mask a) const {
        return _mm512_mask_mov_epi64(m, a.k, _mm512_set1_epi64(int32_t(-1)));
    }

    ENOKI_INLINE Derived and_(Arg a) const { return _mm512_and_epi64(m, a.m); }

    ENOKI_INLINE Derived and_ (Mask a) const {
        return _mm512_maskz_mov_epi64(a.k, m);
    }

    ENOKI_INLINE Derived xor_(Arg a) const { return _mm512_xor_epi64(m, a.m); }

    ENOKI_INLINE Derived xor_ (Mask a) const {
        return _mm512_mask_xor_epi64(m, a.k, m, _mm512_set1_epi64(int32_t(-1)));
    }

    template <size_t k> ENOKI_INLINE Derived sli_() const {
        return _mm512_slli_epi64(m, (int) k);
    }

    template <size_t k> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Value>::value)
            return _mm512_srai_epi64(m, (int) k);
        else
            return _mm512_srli_epi64(m, (int) k);
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        #if 0
            return _mm512_sll_epi64(m, _mm_set1_epi64x((long long) k));
        #else
            /* This is not strictly correct (k may not be a compile-time constant),
               but all targeted compilers figure it out and generate better code */
            return _mm512_slli_epi64(m, (int) k);
        #endif
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        #if 0
            if (std::is_signed<Value>::value)
                return _mm512_sra_epi64(m, _mm_set1_epi64x((long long) k));
            else
                return _mm512_srl_epi64(m, _mm_set1_epi64x((long long) k));
        #else
            /* This is not strictly correct (k may not be a compile-time constant),
               but all targeted compilers figure it out and generate better code */
            if (std::is_signed<Value>::value)
                return _mm512_srai_epi64(m, (int) k);
            else
                return _mm512_srli_epi64(m, (int) k);
        #endif
    }

    ENOKI_INLINE Derived slv_(Arg k) const {
        return _mm512_sllv_epi64(m, k.m);
    }

    ENOKI_INLINE Derived srv_(Arg k) const {
        if (std::is_signed<Value>::value)
            return _mm512_srav_epi64(m, k.m);
        else
            return _mm512_srlv_epi64(m, k.m);
    }

    ENOKI_INLINE Derived rolv_(Arg k) const { return _mm512_rolv_epi64(m, k.m); }
    ENOKI_INLINE Derived rorv_(Arg k) const { return _mm512_rorv_epi64(m, k.m); }

    ENOKI_INLINE Derived rol_(size_t k) const { return rolv_(_mm512_set1_epi64((int32_t) k)); }
    ENOKI_INLINE Derived ror_(size_t k) const { return rorv_(_mm512_set1_epi64((int32_t) k)); }

    template <size_t Imm>
    ENOKI_INLINE Derived roli_() const { return _mm512_rol_epi64(m, (int) Imm); }

    template <size_t Imm>
    ENOKI_INLINE Derived rori_() const { return _mm512_ror_epi64(m, (int) Imm); }

    ENOKI_INLINE Mask lt_ (Arg a) const { return Mask(_mm512_cmp_epi64_mask(m, a.m, _MM_CMPINT_LT));  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return Mask(_mm512_cmp_epi64_mask(m, a.m, _MM_CMPINT_GT));  }
    ENOKI_INLINE Mask le_ (Arg a) const { return Mask(_mm512_cmp_epi64_mask(m, a.m, _MM_CMPINT_LE));  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return Mask(_mm512_cmp_epi64_mask(m, a.m, _MM_CMPINT_GE));  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return Mask(_mm512_cmp_epi64_mask(m, a.m, _MM_CMPINT_EQ));  }
    ENOKI_INLINE Mask neq_(Arg a) const { return Mask(_mm512_cmp_epi64_mask(m, a.m, _MM_CMPINT_NE)); }

    ENOKI_INLINE Derived min_(Arg a) const {
        if (std::is_signed<Value>::value)
            return _mm512_min_epi64(a.m, m);
        else
            return _mm512_min_epu32(a.m, m);
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        if (std::is_signed<Value>::value)
            return _mm512_max_epi64(a.m, m);
        else
            return _mm512_max_epu32(a.m, m);
    }

    ENOKI_INLINE Derived abs_() const {
        return std::is_signed<Value>::value ? _mm512_abs_epi64(m) : m;
    }

    static ENOKI_INLINE Derived select_(const Mask &m, const Derived &t,
                                        const Derived &f) {
        return _mm512_mask_blend_epi64(m.k, f.m, t.m);
    }

    template <size_t I0, size_t I1, size_t I2, size_t I3, size_t I4, size_t I5,
              size_t I6, size_t I7>
    ENOKI_INLINE Derived shuffle_() const {
        const __m512i idx =
            _mm512_setr_epi64(I0, I1, I2, I3, I4, I5, I6, I7);
        return _mm512_permutexvar_epi64(idx, m);
    }


    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    //! @}
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const {
        _mm512_store_si512((__m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), m);
    }
    ENOKI_INLINE void store_(void *ptr, const Mask &mask) const {
        _mm512_mask_store_epi64((__m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64), mask.k, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        _mm512_storeu_si512((__m512i *) ptr, m);
    }
    ENOKI_INLINE void store_unaligned_(void *ptr, const Mask &mask) const {
        _mm512_mask_storeu_epi64((__m512i *) ptr, mask.k, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr) {
        return _mm512_load_si512((const __m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_load_epi64(mask.k, (const __m512i *) ENOKI_ASSUME_ALIGNED_S(ptr, 64));
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        return _mm512_loadu_si512((const __m512i *) ptr);
    }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, const Mask &mask) {
        return _mm512_maskz_loadu_epi64(mask.k, (const __m512i *) ptr);
    }

    static ENOKI_INLINE Derived zero_() { return _mm512_setzero_si512(); }

#if defined(__AVX512PF__)
    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write)
            _mm512_prefetch_i32scatter_pd(ptr, index.m, Stride, Level);
        else
            _mm512_prefetch_i32gather_pd(index.m, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int32_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write)
            _mm512_mask_prefetch_i32scatter_pd(ptr, mask.k, index.m, Stride, Level);
        else
            _mm512_mask_prefetch_i32gather_pd(index.m, mask.k, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        if (Write)
            _mm512_prefetch_i64scatter_pd(ptr, index.m, Stride, Level);
        else
            _mm512_prefetch_i64gather_pd(index.m, ptr, Stride, Level);
    }

    ENOKI_REQUIRE_INDEX_PF(Index, int64_t)
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index,
                                       const Mask &mask) {
        if (Write)
            _mm512_mask_prefetch_i64scatter_pd(ptr, mask.k, index.m, Stride, Level);
        else
            _mm512_mask_prefetch_i64gather_pd(index.m, mask.k, ptr, Stride, Level);
    }
#endif

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm512_i32gather_epi64(index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), mask.k, index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm512_i64gather_epi64(index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), mask.k, index.m, (const float *) ptr, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i32scatter_epi64(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i32scatter_epi64(ptr, mask.k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i64scatter_epi64(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        _mm512_mask_i64scatter_epi64(ptr, mask.k, index.m, m, Stride);
    }

    ENOKI_INLINE Value extract_(const Mask &mask) const {
        return (Value) _mm_cvtsi128_si64(_mm512_castsi512_si128(_mm512_maskz_compress_epi64(mask.k, m)));
    }

    template <typename T>
    ENOKI_INLINE size_t compress_(T *&ptr, const Mask &mask) const {
        __mmask8 k = mask.k;
        _mm512_storeu_si512((__m512i *) ptr, _mm512_maskz_compress_epi64(k, m));
        size_t kn = (size_t) _mm_popcnt_u32(k);
        ptr += kn;
        return kn;
    }

#if defined(__AVX512CD__)
    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int64_t)
    static ENOKI_INLINE void transform_(void *mem, Index index,
                                        const Func &func,
                                        const Args &... args) {
        transform_masked_<Stride>(mem, index, Mask(true), func, args...);
    }

    ENOKI_REQUIRE_INDEX_TRANSFORM(Index, int64_t)
    static ENOKI_INLINE void transform_masked_(void *mem, Index index,
                                               Mask mask, const Func &func,
                                               const Args &... args) {
        Derived values = _mm512_mask_i64gather_epi64(
            _mm512_undefined_epi32(), mask.k, index.m, mem, (int) Stride);

        index.m = _mm512_mask_mov_epi64(_mm512_set1_epi64(-1), mask.k, index.m);

        __m512i conflicts = _mm512_conflict_epi64(index.m);
        __m512i perm_idx  = _mm512_sub_epi64(_mm512_set1_epi64(63), _mm512_lzcnt_epi64(conflicts));
        __mmask8 todo     = _mm512_mask_test_epi64_mask(mask.k, conflicts, _mm512_set1_epi64(-1));

        func(values, args...);

        while (ENOKI_UNLIKELY(todo)) {
            __mmask8 cur = _mm512_mask_testn_epi64_mask(
                todo, conflicts, _mm512_broadcastmb_epi64(todo));
            values.m = _mm512_mask_permutexvar_epi64(values.m, cur, perm_idx, values.m);

            __m512i backup(values.m);
            func(values, args...);

            values.m = _mm512_mask_mov_epi64(backup, cur, values.m);
            todo ^= cur;
        }

        _mm512_mask_i64scatter_epi64(mem, mask.k, index.m, values.m, (int) Stride);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Masked versions of key operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE void massign_(const Derived &a, const Mask &mask) { m = _mm512_mask_mov_epi64(m, mask.k, a.m); }
    ENOKI_INLINE void madd_   (const Derived &a, const Mask &mask) { m = _mm512_mask_add_epi64(m, mask.k, m, a.m); }
    ENOKI_INLINE void msub_   (const Derived &a, const Mask &mask) { m = _mm512_mask_sub_epi64(m, mask.k, m, a.m); }
    ENOKI_INLINE void mmul_   (const Derived &a, const Mask &mask) {
        #if defined(__AVX512DQ__) && defined(__AVX512VL__)
            m = _mm512_mask_mullo_epi64(m, mask.k, m, a.m);
        #else
            m = select(mask, a * derived(), derived()).m;
        #endif
    }
    ENOKI_INLINE void mor_    (const Derived &a, const Mask &mask) { m = _mm512_mask_or_epi64(m, mask.k, m, a.m); }
    ENOKI_INLINE void mand_   (const Derived &a, const Mask &mask) { m = _mm512_mask_and_epi64(m, mask.k, m, a.m); }
    ENOKI_INLINE void mxor_   (const Derived &a, const Mask &mask) { m = _mm512_mask_xor_epi64(m, mask.k, m, a.m); }

    //! @}
    // -----------------------------------------------------------------------
};

NAMESPACE_END(enoki)
