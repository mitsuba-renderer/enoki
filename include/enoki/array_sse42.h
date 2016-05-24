/*
    enoki/array_sse42.h -- Packed SIMD array (SSE4.2 specialization)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "array_generic.h"

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

/// Compressed look-up table for the store_compress() operation [256 bytes]
alignas(16) const uint8_t compress_lut_128[16*16] = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x04, 0x05, 0x06, 0x07,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x08, 0x09, 0x0a, 0x0b, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x02, 0x03,
    0x08, 0x09, 0x0a, 0x0b, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x80, 0x80, 0x80, 0x80, 0x0c, 0x0d, 0x0e, 0x0f,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x01, 0x02, 0x03, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x04, 0x05, 0x06, 0x07, 0x0c, 0x0d, 0x0e, 0x0f,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f
};

NAMESPACE_END(detail)

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (single precision)
template <bool Approx, typename Derived> struct alignas(16)
    StaticArrayImpl<float, 4, Approx, RoundingMode::Default, Derived>
    : StaticArrayBase<float, 4, Approx, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(float, 4, Approx, __m128)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value) : m(_mm_set1_ps(value)) { }
    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1, Scalar v2, Scalar v3)
        : m(_mm_setr_ps(v0, v1, v2, v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(float) : m(a.derived().m) { }
    ENOKI_CONVERT(int32_t) : m(_mm_cvtepi32_ps(a.derived().m)) { }
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(uint32_t) : m(_mm_cvtepu32_ps(a.derived().m)) { }
#endif
#if defined(__AVX__)
    ENOKI_CONVERT(double) : m(_mm256_cvtpd_ps(a.derived().m)) { }
#else
    ENOKI_CONVERT(double)
        : m(_mm_shuffle_ps(_mm_cvtpd_ps(low(a).m), _mm_cvtpd_ps(high(a).m),
                           _MM_SHUFFLE(1, 0, 1, 0))) { }
#endif
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(int64_t) : m(_mm256_cvtepi64_ps(a.derived().m)) { }
    ENOKI_CONVERT(uint64_t) : m(_mm256_cvtepu64_ps(a.derived().m)) { }
#endif
    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) : m(a.derived().m) { }
    ENOKI_REINTERPRET(int32_t) : m(_mm_castsi128_ps(a.derived().m)) { }
    ENOKI_REINTERPRET(uint32_t) : m(_mm_castsi128_ps(a.derived().m)) { }

#if defined(__AVX__)
    ENOKI_REINTERPRET(double)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(_mm256_castpd_si256(a.derived().m)))) { }
#else
    ENOKI_REINTERPRET(double)
        : m(_mm_castsi128_ps(detail::mm256_cvtepi64_epi32(
              _mm_castpd_si128(low(a).m), _mm_castpd_si128(high(a).m)))) { }
#endif

#if defined(__AVX2__)
    ENOKI_REINTERPRET(uint64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(a.derived().m))) { }
    ENOKI_REINTERPRET(int64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(a.derived().m))) { }
#else
    ENOKI_REINTERPRET(uint64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(low(a).m, high(a).m))) { }
    ENOKI_REINTERPRET(int64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(low(a).m, high(a).m))) { }
#endif

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

    ENOKI_INLINE Derived add_(Arg a) const { return _mm_add_ps(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm_sub_ps(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm_mul_ps(m, a.m); }
    ENOKI_INLINE Derived div_(Arg a) const { return _mm_div_ps(m, a.m); }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm_or_ps (m, a.m); }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm_and_ps(m, a.m); }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm_xor_ps(m, a.m); }

#if defined(__AVX__)
    ENOKI_INLINE Mask lt_ (Arg a) const { return _mm_cmp_ps(m, a.m, _CMP_LT_OQ);  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return _mm_cmp_ps(m, a.m, _CMP_GT_OQ);  }
    ENOKI_INLINE Mask le_ (Arg a) const { return _mm_cmp_ps(m, a.m, _CMP_LE_OQ);  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return _mm_cmp_ps(m, a.m, _CMP_GE_OQ);  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return _mm_cmp_ps(m, a.m, _CMP_EQ_OQ);  }
    ENOKI_INLINE Mask neq_(Arg a) const { return _mm_cmp_ps(m, a.m, _CMP_NEQ_UQ); }
#else
    ENOKI_INLINE Mask lt_ (Arg a) const { return _mm_cmplt_ps(m, a.m);  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return _mm_cmpgt_ps(m, a.m);  }
    ENOKI_INLINE Mask le_ (Arg a) const { return _mm_cmple_ps(m, a.m);  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return _mm_cmpge_ps(m, a.m);  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return _mm_cmpeq_ps(m, a.m);  }
    ENOKI_INLINE Mask neq_(Arg a) const { return _mm_cmpneq_ps(m, a.m); }
#endif

    ENOKI_INLINE Derived abs_()      const { return _mm_andnot_ps(_mm_set1_ps(-0.f), m); }
    ENOKI_INLINE Derived min_(Arg b) const { return _mm_min_ps(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return _mm_max_ps(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm_ceil_ps(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm_floor_ps(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm_sqrt_ps(m);     }

    ENOKI_INLINE Derived round_() const {
        return _mm_round_ps(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm_blendv_ps(f.m, t.m, m.m);
    }

#if defined(__FMA__)
    ENOKI_INLINE Derived fmadd_(Arg b, Arg c) const {
        return _mm_fmadd_ps(m, b.m, c.m);
    }

    ENOKI_INLINE Derived fmsub_(Arg b, Arg c) const {
        return _mm_fmsub_ps(m, b.m, c.m);
    }
#endif

    template <int I0, int I1, int I2, int I3>
    ENOKI_INLINE Derived shuffle_() const {
        #if defined(__AVX__)
            return _mm_permute_ps(m, _MM_SHUFFLE(I3, I2, I1, I0));
        #else
            return _mm_shuffle_ps(m, m, _MM_SHUFFLE(I3, I2, I1, I0));
        #endif
    }

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived ldexp_(Arg arg) const { return _mm_scalef_ps(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm_getmant_ps(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm_add_ps(_mm_getexp_ps(m), _mm_set1_ps(1.f)));
    }
#endif

    ENOKI_INLINE Derived rcp_() const {
        if (Approx) {
            /* Use best reciprocal approximation available on the current
               hardware and potentially refine */

            __m128 r;
            #if defined(__AVX512ER__)
                /* rel err < 2^28, use as is */
                r = _mm512_castps512_ps128(
                    _mm512_rcp28_ps(_mm512_castps128_ps512(m)));
            #elif defined(__AVX512F__)
                r = _mm_rcp14_ps(m); /* rel error < 2^-14 */
            #else
                r = _mm_rcp_ps(m);   /* rel error < 1.5*2^-12 */
            #endif

            #if !defined(__AVX512ER__)
                /* Refine using one Newton-Raphson iteration */

                #if defined(__FMA__)
                    const __m128 two = _mm_set1_ps(2.f);
                    r = _mm_mul_ps(r, _mm_fnmadd_ps(r, m, two));
                #else
                    r = _mm_sub_ps(_mm_add_ps(r, r),
                                   _mm_mul_ps(_mm_mul_ps(r, r), m));
                #endif
            #endif

            return r;
        } else {
            return Base::rcp_();
        }
    }

    ENOKI_INLINE Derived rsqrt_() const {
        if (Approx) {
            /* Use best reciprocal square root approximation available
               on the current hardware and potentially refine */
            __m128 r;
            #if defined(__AVX512ER__)
                /* rel err < 2^28, use as is */
                r = _mm512_castps512_ps128(
                    _mm512_rsqrt28_ps(_mm512_castps128_ps512(m)));
            #elif defined(__AVX512VL__)
                r = _mm_rsqrt14_ps(m); /* rel error < 2^-14 */
            #else
                r = _mm_rsqrt_ps(m);   /* rel error < 1.5*2^-12 */
            #endif

            #if !defined(__AVX512ER__)
                /* Refine using one Newton-Raphson iteration */

                const __m128 c0 = _mm_set1_ps(1.5f);
                const __m128 c1 = _mm_set1_ps(-0.5f);

                #if defined(__FMA__)
                    r = _mm_fmadd_ps(r, c0,
                                      _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(m, c1), r),
                                                 _mm_mul_ps(r, r)));
                #else
                    r = _mm_add_ps(_mm_mul_ps(c0, r),
                                    _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(m, c1), r),
                                               _mm_mul_ps(r, r)));
                #endif
            #endif

            return r;
        } else {
            return Base::rsqrt_();
        }
    }

#if defined(__AVX512ER__)
    ENOKI_INLINE Derived exp_() const {
        if (Approx) {
            return _mm512_castps512_ps128(
                _mm512_exp2a23_ps(_mm512_castps128_ps512(
                    _mm_mul_ps(m, _mm_set1_ps(1.4426950408889634074f)))));
        } else {
            return Base::exp_();
        }
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128 t1 = _mm_movehdup_ps(m); \
            __m128 t2 = _mm_##op##_ps(m, t1); \
            t1        = _mm_movehl_ps(t1, t2); \
            t2        = _mm_##op##_ss(t2, t1); \
            return _mm_cvtss_f32(t2); \
        }

    ENOKI_HORIZONTAL_OP(hsum, add)
    ENOKI_HORIZONTAL_OP(hprod, mul)
    ENOKI_HORIZONTAL_OP(hmin, min)
    ENOKI_HORIZONTAL_OP(hmax, max)

    #undef ENOKI_HORIZONTAL_OP

    ENOKI_INLINE bool all_() const { return _mm_movemask_ps(m) == 0xF;}
    ENOKI_INLINE bool any_() const { return _mm_movemask_ps(m) != 0x0; }
    ENOKI_INLINE bool none_() const { return _mm_movemask_ps(m) == 0x0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm_movemask_ps(m));
    }

    ENOKI_INLINE Scalar dot_(Arg a) const {
        return _mm_cvtss_f32(_mm_dp_ps(m, a.m, 0b11110001));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm_store_ps((Scalar *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm_storeu_ps((Scalar *) ptr, m); }

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm_load_ps((const Scalar *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm_loadu_ps((const Scalar *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm_setzero_ps(); }

#if defined(__AVX2__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm_i32gather_ps((const float *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm_mask_i32gather_ps(_mm_setzero_ps(), (const float *) ptr, index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i64gather_ps((const float *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i64gather_ps(_mm_setzero_ps(), (const float *) ptr, index.m, mask.m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm_i32scatter_ps(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask_) const {
        __m128i mask = _mm_castps_si128(mask_.m);
        __mmask8 k = _mm_test_epi32_mask(mask, mask);
        _mm_mask_i32scatter_ps(ptr, k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm256_i64scatter_ps(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask_) const {
        __m128i mask = _mm_castps_si128(mask_.m);
        __mmask8 k = _mm_test_epi32_mask(mask, mask);
        _mm256_mask_i64scatter_ps(ptr, k, index.m, m, Stride);
    }
#endif

    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        unsigned int offset = (unsigned int) _mm_movemask_ps(mask.m);

        #if !defined(__AVX512VL__)
            /** Fancy LUT-based partitioning algorithm, see
                https://deplinenoise.files.wordpress.com/2015/03/gdc2015_afredriksson_simd.pdf */

            __m128i shuf = _mm_load_si128(((const __m128i *) detail::compress_lut_128) + offset),
                    perm = _mm_shuffle_epi8(_mm_castps_si128(m), shuf);

            _mm_storeu_si128((__m128i *) ptr, perm);
        #else
            __mmask8 k = _mm_test_epi32_mask(_mm_castps_si128(mask.m),
                                             _mm_castps_si128(mask.m));
            _mm_storeu_ps((float *) ptr,
                          _mm_mask_compress_ps(_mm_setzero_ps(), k, m));
        #endif

        (float *&) ptr += _mm_popcnt_u32(offset);
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (double precision)
template <bool Approx, typename Derived> struct alignas(16)
    StaticArrayImpl<double, 2, Approx, RoundingMode::Default, Derived>
    : StaticArrayBase<double, 2, Approx, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(double, 2, Approx, __m128d)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value) : m(_mm_set1_pd(value)) { }
    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1)
        : m(_mm_setr_pd(v0, v1)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    /* No vectorized conversions from float/[u]int32_t (too small) */

    ENOKI_CONVERT(double) : m(a.derived().m) { }
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(int64_t) : m(_mm_cvtepi64_pd(a.derived().m)) { }
    ENOKI_CONVERT(uint64_t) : m(_mm_cvtepu64_pd(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1);
        m = _mm_castps_pd(_mm_setr_ps(v0, v0, v1, v1));
    }

    ENOKI_REINTERPRET(int32_t) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1);
        m = _mm_castsi128_pd(_mm_setr_epi32(v0, v0, v1, v1));
    }

    ENOKI_REINTERPRET(uint32_t) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1);
        m = _mm_castsi128_pd(_mm_setr_epi32((int32_t) v0, (int32_t) v0,
                                            (int32_t) v1, (int32_t) v1));
    }

    ENOKI_REINTERPRET(double) : m(a.derived().m) { }
    ENOKI_REINTERPRET(int64_t) : m(_mm_castsi128_pd(a.derived().m)) { }
    ENOKI_REINTERPRET(uint64_t) : m(_mm_castsi128_pd(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_pd(a1.coeff(0), a2.coeff(0))) { }

    ENOKI_INLINE Array1 low_()  const { return Array1(coeff(0)); }
    ENOKI_INLINE Array2 high_() const { return Array2(coeff(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm_add_pd(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm_sub_pd(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm_mul_pd(m, a.m); }
    ENOKI_INLINE Derived div_(Arg a) const { return _mm_div_pd(m, a.m); }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm_or_pd (m, a.m); }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm_and_pd(m, a.m); }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm_xor_pd(m, a.m); }

#if defined(__AVX__)
    ENOKI_INLINE Mask lt_ (Arg a) const { return _mm_cmp_pd(m, a.m, _CMP_LT_OQ);  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return _mm_cmp_pd(m, a.m, _CMP_GT_OQ);  }
    ENOKI_INLINE Mask le_ (Arg a) const { return _mm_cmp_pd(m, a.m, _CMP_LE_OQ);  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return _mm_cmp_pd(m, a.m, _CMP_GE_OQ);  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return _mm_cmp_pd(m, a.m, _CMP_EQ_OQ);  }
    ENOKI_INLINE Mask neq_(Arg a) const { return _mm_cmp_pd(m, a.m, _CMP_NEQ_UQ); }
#else
    ENOKI_INLINE Mask lt_ (Arg a) const { return _mm_cmplt_pd(m, a.m);  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return _mm_cmpgt_pd(m, a.m);  }
    ENOKI_INLINE Mask le_ (Arg a) const { return _mm_cmple_pd(m, a.m);  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return _mm_cmpge_pd(m, a.m);  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return _mm_cmpeq_pd(m, a.m);  }
    ENOKI_INLINE Mask neq_(Arg a) const { return _mm_cmpneq_pd(m, a.m); }
#endif

    ENOKI_INLINE Derived abs_()      const { return _mm_andnot_pd(_mm_set1_pd(-0.), m); }
    ENOKI_INLINE Derived min_(Arg b) const { return _mm_min_pd(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return _mm_max_pd(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm_ceil_pd(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm_floor_pd(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm_sqrt_pd(m);     }

    ENOKI_INLINE Derived round_() const {
        return _mm_round_pd(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm_blendv_pd(f.m, t.m, m.m);
    }

#if defined(__FMA__)
    ENOKI_INLINE Derived fmadd_(Arg b, Arg c) const {
        return _mm_fmadd_pd(m, b.m, c.m);
    }

    ENOKI_INLINE Derived fmsub_(Arg b, Arg c) const {
        return _mm_fmsub_pd(m, b.m, c.m);
    }
#endif

    #if defined(__AVX__)
        #define ENOKI_SHUFFLE_PD(m, flags) _mm_permute_pd(m, flags)
    #else
        #define ENOKI_SHUFFLE_PD(m, flags) _mm_shuffle_pd(m, m, flags)
    #endif

    template <int I0, int I1>
    ENOKI_INLINE Derived shuffle_() const {
        return ENOKI_SHUFFLE_PD(m, (I1 << 1) | I0);
    }

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived ldexp_(Arg arg) const { return _mm_scalef_pd(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm_getmant_pd(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm_add_pd(_mm_getexp_pd(m), _mm_set1_pd(1.0)));
    }
#endif

#if defined(__AVX512VL__) || defined(__AVX512ER__)
    ENOKI_INLINE Derived rcp_() const {
        if (Approx) {
            /* Use best reciprocal approximation available on the current
               hardware and potentially refine */
            __m256 r;
            #if defined(__AVX512ER__)
                /* rel err < 2^28, use as is */
                r = _mm512_castpd512_pd256(
                    _mm512_rcp28_pd(_mm512_castpd256_pd512(m)));
            #elif defined(__AVX512VL__)
                r = _mm256_rcp14_pd(m); /* rel error < 2^-14 */
            #endif

            #if !defined(__AVX512ER__)
                /* Refine using two Newton-Raphson iterations */

                for (int i = 0; i < 2; ++i) {
                    #if defined(__FMA__)
                        const __m256 two = _mm256_set1_pd(2.);
                        r = _mm256_mul_pd(r, _mm256_fnmadd_pd(r, m, two));
                    #else
                        r = _mm256_sub_pd(_mm256_add_pd(r, r),
                                          _mm256_mul_pd(_mm256_mul_pd(r, r), m));
                    #endif
                }
            #endif

            return r;
        } else {
            return Base::rcp_();
        }
    }

    ENOKI_INLINE Derived rsqrt_() const {
        if (Approx) {
            /* Use best reciprocal square root approximation available
               on the current hardware and potentially refine */
            __m128 r;
            #if defined(__AVX512ER__)
                /* rel err < 2^28, use as is */
                r = _mm512_castpd512_pd128(
                    _mm512_rsqrt28_pd(_mm512_castpd128_pd512(m)));
            #elif defined(__AVX512VL__)
                r = _mm_rsqrt14_pd(m); /* rel error < 2^-14 */
            #endif

            #if !defined(__AVX512ER__)
                /* Refine using two Newton-Raphson iterations */
                const __m128 c0 = _mm_set1_pd(1.5);
                const __m128 c1 = _mm_set1_pd(-0.5);

                for (int i = 0; i < 2; ++i) {
                    #if defined(__FMA__)
                        r = _mm_fmadd_pd(r, c0,
                            _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(m, c1), r),
                                       _mm_mul_pd(r, r)));
                    #else
                        r = _mm_add_pd(
                            _mm_mul_pd(c0, r),
                            _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(m, c1), r),
                                       _mm_mul_pd(r, r)));
                    #endif
                }
            #endif

            return r;
        } else {
            return Base::rsqrt_();
        }
    }
#endif

#if defined(__AVX512ER__)
    ENOKI_INLINE Derived exp_() const {
        if (Approx) {
            return _mm512_castpd512_pd128(
                _mm512_exp2a23_pd(_mm512_castpd128_pd512(
                    _mm_mul_pd(m, _mm_set1_pd(1.4426950408889634074)))));
        } else {
            return Base::exp_();
        }
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128d t0 = ENOKI_SHUFFLE_PD(m, 1); \
            __m128d t1 = _mm_##op##_sd(t0, m); \
            return  _mm_cvtsd_f64(t1); \
        }

    ENOKI_HORIZONTAL_OP(hsum, add)
    ENOKI_HORIZONTAL_OP(hprod, mul)
    ENOKI_HORIZONTAL_OP(hmin, min)
    ENOKI_HORIZONTAL_OP(hmax, max)

    #undef ENOKI_HORIZONTAL_OP
    #undef ENOKI_SHUFFLE_PD

    ENOKI_INLINE bool all_() const { return _mm_movemask_pd(m) == 0x3;}
    ENOKI_INLINE bool any_() const { return _mm_movemask_pd(m) != 0x0; }
    ENOKI_INLINE bool none_() const { return _mm_movemask_pd(m) == 0x0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm_movemask_pd(m));
    }

    ENOKI_INLINE Scalar dot_(Arg a) const {
        return _mm_cvtsd_f64(_mm_dp_pd(m, a.m, 0b00110001));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm_store_pd((Scalar *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm_storeu_pd((Scalar *) ptr, m); }

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm_load_pd((const Scalar *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm_loadu_pd((const Scalar *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm_setzero_pd(); }

#if defined(__AVX2__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return Base::template gather_<Stride>(ptr, index);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Base::template gather_<Stride>(ptr, index, mask);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm_i64gather_pd((const double *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm_mask_i64gather_pd(_mm_setzero_pd(), (const double *) ptr, index.m, mask.m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        Base::template scatter_<Stride>(ptr, index);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        Base::template scatter_<Stride>(ptr, index, mask);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm_i64scatter_pd(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask_) const {
        __m128i mask = _mm_castpd_si128(mask_.m);
        __mmask8 k = _mm_test_epi64_mask(mask, mask);
        _mm_mask_i64scatter_pd(ptr, k, index.m, m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        unsigned int offset = (unsigned int) _mm_movemask_pd(mask.m);
        __mmask8 k = _mm_test_epi64_mask(_mm_castpd_si128(mask.m),
                                         _mm_castpd_si128(mask.m));
        _mm_storeu_pd((double *) ptr,
                      _mm_mask_compress_pd(_mm_setzero_pd(), k, m));

        (float *&) ptr += _mm_popcnt_u32(offset);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (32 bit integers)
template <typename Scalar_, typename Derived>
struct alignas(16) StaticArrayImpl<Scalar_, 4, false, RoundingMode::Default,
                                   Derived, detail::is_int32_t<Scalar_>>
    : StaticArrayBase<Scalar_, 4, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(Scalar_, 4, false, __m128i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value) : m(_mm_set1_epi32((int32_t) value)) { }
    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1, Scalar v2, Scalar v3)
        : m(_mm_setr_epi32((int32_t) v0, (int32_t) v1, (int32_t) v2, (int32_t) v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(float) {
        if (std::is_signed<Scalar>::value) {
            m = _mm_cvttps_epi32(a.derived().m);
        } else {
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
            m = _mm_cvttps_epu32(a.derived().m);
#else
            ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                coeff(i) = Scalar(a.derived().coeff(i));
#endif

        }
    }

    ENOKI_CONVERT(int32_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint32_t) : m(a.derived().m) { }

#if defined(__AVX__)
    ENOKI_CONVERT(double) {
        if (std::is_signed<Scalar>::value) {
            m = _mm256_cvttpd_epi32(a.derived().m);
        } else {
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
            m = _mm256_cvttpd_epu32(a.derived().m);
#else
            ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                coeff(i) = Scalar(a.derived().coeff(i));
#endif
        }
    }
#endif

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(int64_t) { m = _mm256_cvtepi64_epi32(a.derived().m); }
    ENOKI_CONVERT(uint64_t) { m = _mm256_cvtepi64_epi32(a.derived().m); }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) : m(_mm_castps_si128(a.derived().m)) { }
    ENOKI_REINTERPRET(int32_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint32_t) : m(a.derived().m) { }

#if defined(__AVX__)
    ENOKI_REINTERPRET(double)
        : m(detail::mm256_cvtepi64_epi32(_mm256_castpd_si256(a.derived().m))) { }
#else
    ENOKI_REINTERPRET(double)
        : m(detail::mm256_cvtepi64_epi32(_mm_castpd_si128(low(a).m),
                                         _mm_castpd_si128(high(a).m))) { }
#endif

#if defined(__AVX2__)
    ENOKI_REINTERPRET(uint64_t)
        : m(detail::mm256_cvtepi64_epi32(a.derived().m)) { }
    ENOKI_REINTERPRET(int64_t)
        : m(detail::mm256_cvtepi64_epi32(a.derived().m)) {}
#else
    ENOKI_REINTERPRET(uint64_t)
        : m(detail::mm256_cvtepi64_epi32(low(a).m, high(a).m)) { }
    ENOKI_REINTERPRET(int64_t)
        : m(detail::mm256_cvtepi64_epi32(low(a).m, high(a).m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_epi32((int32_t) a1.coeff(0), (int32_t) a1.coeff(1),
                           (int32_t) a2.coeff(0), (int32_t) a2.coeff(1))) { }

    ENOKI_INLINE Array1 low_()  const { return Array1(coeff(0), coeff(1)); }
    ENOKI_INLINE Array2 high_() const { return Array2(coeff(2), coeff(3)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm_add_epi32(m, a.m);   }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm_sub_epi32(m, a.m);   }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm_mullo_epi32(m, a.m); }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm_or_si128(m, a.m);    }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm_and_si128(m, a.m);   }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm_xor_si128(m, a.m);   }

    template <size_t k> ENOKI_INLINE Derived sli_() const {
        return _mm_slli_epi32(m, (int) k);
    }

    template <size_t k> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Scalar>::value)
            return _mm_srai_epi32(m, (int) k);
        else
            return _mm_srli_epi32(m, (int) k);
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        return _mm_sll_epi32(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        if (std::is_signed<Scalar>::value)
            return _mm_sra_epi32(m, _mm_set1_epi64x((long long) k));
        else
            return _mm_srl_epi32(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived slv_(Arg k) const {
        #if defined(__AVX2__)
            return _mm_sllv_epi32(m, k.m);
        #else
            Derived out;
            ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                out.coeff(i) = coeff(i) << (size_t) k.coeff(i);
            return out;
        #endif
    }

    ENOKI_INLINE Derived srv_(Arg k) const {
        #if defined(__AVX2__)
            if (std::is_signed<Scalar>::value)
                return _mm_srav_epi32(m, k.m);
            else
                return _mm_srlv_epi32(m, k.m);
        #else
            Derived out;
            ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                out.coeff(i) = coeff(i) >> (size_t) k.coeff(i);
            return out;
        #endif
    }

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived rolv_(Arg k) const { return _mm_rolv_epi32(m, k.m); }
    ENOKI_INLINE Derived rorv_(Arg k) const { return _mm_rorv_epi32(m, k.m); }
    ENOKI_INLINE Derived rol_(size_t k) const { return rolv_(_mm_set1_epi32((int32_t) k)); }
    ENOKI_INLINE Derived ror_(size_t k) const { return rorv_(_mm_set1_epi32((int32_t) k)); }
    template <size_t Imm>
    ENOKI_INLINE Derived roli_(Arg k) const { return _mm_rol_epi32(m, (int) k); }
    template <size_t Imm>
    ENOKI_INLINE Derived rori_(Arg k) const { return _mm_ror_epi32(m, (int) k); }
#endif

    ENOKI_INLINE Mask lt_(Arg a) const {
        if (std::is_signed<Scalar>::value) {
            return _mm_cmpgt_epi32(a.m, m);
        } else {
            const __m128i offset = _mm_set1_epi32((int32_t) 0x80000000ul);
            return _mm_cmpgt_epi32(_mm_sub_epi32(a.m, offset),
                                   _mm_sub_epi32(m, offset));
        }
    }

    ENOKI_INLINE Mask gt_(Arg a) const {
        if (std::is_signed<Scalar>::value) {
            return _mm_cmpgt_epi32(m, a.m);
        } else {
            const __m128i offset = _mm_set1_epi32((int32_t) 0x80000000ul);
            return _mm_cmpgt_epi32(_mm_sub_epi32(m, offset),
                                   _mm_sub_epi32(a.m, offset));
        }
    }

    ENOKI_INLINE Mask le_(Arg a) const { return ~gt_(a); }
    ENOKI_INLINE Mask ge_(Arg a) const { return ~lt_(a); }

    ENOKI_INLINE Mask eq_(Arg a)  const { return _mm_cmpeq_epi32(m, a.m); }
    ENOKI_INLINE Mask neq_(Arg a) const { return ~eq_(a); }

    ENOKI_INLINE Derived min_(Arg a) const {
        if (std::is_signed<Scalar>::value)
            return _mm_min_epi32(a.m, m);
        else
            return _mm_min_epu32(a.m, m);
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        if (std::is_signed<Scalar>::value)
            return _mm_max_epi32(a.m, m);
        else
            return _mm_max_epu32(a.m, m);
    }

    ENOKI_INLINE Derived abs_() const {
        return std::is_signed<Scalar>::value ? _mm_abs_epi32(m) : m;
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm_blendv_epi8(f.m, t.m, m.m);
    }

    template <int I0, int I1, int I2, int I3>
    ENOKI_INLINE Derived shuffle_() const {
        return _mm_shuffle_epi32(m, _MM_SHUFFLE(I3, I2, I1, I0));
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        const Mask blend(Scalar(-1), 0, Scalar(-1), 0);

        if (std::is_signed<Scalar>::value) {
            Derived even(_mm_srli_epi64(_mm_mul_epi32(m, a.m), 32));
            Derived odd(_mm_mul_epi32(_mm_srli_epi64(m, 32), _mm_srli_epi64(a.m, 32)));
            return select(blend, even, odd);
        } else {
            Derived even(_mm_srli_epi64(_mm_mul_epu32(m, a.m), 32));
            Derived odd(_mm_mul_epu32(_mm_srli_epi64(m, 32), _mm_srli_epi64(a.m, 32)));
            return select(blend, even, odd);
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128i t1 = _mm_shuffle_epi32(m, 0x4e); \
            __m128i t2 = _mm_##op##_epi32(m, t1);  \
            t1         = _mm_shufflelo_epi16(t2, 0x4e); \
            t2         = _mm_##op##_epi32(t2, t1); \
            return (Scalar) _mm_cvtsi128_si32(t2); \
        }

    #define ENOKI_HORIZONTAL_OP_SIGNED(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128i t2, t1 = _mm_shuffle_epi32(m, 0x4e); \
            if (std::is_signed<Scalar>::value) \
                t2 = _mm_##op##_epi32(m, t1);  \
            else \
                t2 = _mm_##op##_epu32(m, t1);  \
            t1 = _mm_shufflelo_epi16(t2, 0x4e); \
            if (std::is_signed<Scalar>::value) \
                t2 = _mm_##op##_epi32(t2, t1); \
            else \
                t2 = _mm_##op##_epu32(t2, t1);  \
            return (Scalar) _mm_cvtsi128_si32(t2); \
        }

    ENOKI_HORIZONTAL_OP(hsum, add)
    ENOKI_HORIZONTAL_OP(hprod, mullo)
    ENOKI_HORIZONTAL_OP_SIGNED(hmin, min)
    ENOKI_HORIZONTAL_OP_SIGNED(hmax, max)

    #undef ENOKI_HORIZONTAL_OP
    #undef ENOKI_HORIZONTAL_OP_SIGNED

    ENOKI_INLINE bool all_() const { return _mm_movemask_ps(_mm_castsi128_ps(m)) == 0xF;}
    ENOKI_INLINE bool any_() const { return _mm_movemask_ps(_mm_castsi128_ps(m)) != 0x0; }
    ENOKI_INLINE bool none_() const { return _mm_movemask_ps(_mm_castsi128_ps(m)) == 0x0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm_movemask_ps(_mm_castsi128_ps(m)));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm_store_si128((__m128i *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm_storeu_si128((__m128i *) ptr, m); }

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm_load_si128((const __m128i *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm_loadu_si128((const __m128i *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm_setzero_si128(); }

#if defined(__AVX2__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm_i32gather_epi32((const int *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm_mask_i32gather_epi32(_mm_setzero_si128(), (const int *) ptr,
                                        index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i64gather_epi32((const int *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i64gather_epi32(_mm_setzero_si128(), (const int *) ptr,
                                           index.m, mask.m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm_i32scatter_epi32(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        __mmask8 k = _mm_test_epi32_mask(mask.m, mask.m);
        _mm_mask_i32scatter_epi32(ptr, k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm256_i64scatter_epi32(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        __mmask8 k = _mm_test_epi32_mask(mask.m, mask.m);
        _mm256_mask_i64scatter_epi32(ptr, k, index.m, m, Stride);
    }
#endif

    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        unsigned int offset = (unsigned int) _mm_movemask_ps(_mm_castsi128_ps(mask.m));

        #if !defined(__AVX512VL__)
            /** Fancy LUT-based partitioning algorithm, see
                https://deplinenoise.files.wordpress.com/2015/03/gdc2015_afredriksson_simd.pdf */

            __m128i shuf = _mm_load_si128(((const __m128i *) detail::compress_lut_128) + offset),
                    perm = _mm_shuffle_epi8(m, shuf);

            _mm_storeu_si128((__m128i *) ptr, perm);
        #else
            __mmask8 k = _mm_test_epi32_mask(mask.m, mask.m);
            _mm_storeu_si128((__m128i *) ptr,
                             _mm_mask_compress_epi32(_mm_setzero_si128(), k, m));
        #endif

        (uint32_t *&) ptr += _mm_popcnt_u32(offset);
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (64 bit integers)
template <typename Scalar_, typename Derived>
struct alignas(16) StaticArrayImpl<Scalar_, 2, false, RoundingMode::Default,
                                   Derived, detail::is_int64_t<Scalar_>>
    : StaticArrayBase<Scalar_, 2, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(Scalar_, 2, false, __m128i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value) : m(_mm_set1_epi64x((int64_t) value)) { }
    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1) {
        alignas(16) Scalar data[2];
        data[0] = (Scalar) v0;
        data[1] = (Scalar) v1;
        m = _mm_load_si128((__m128i *) data);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(double) {
        if (std::is_signed<Scalar>::value)
            m = _mm_cvttpd_epi64(a.derived().m);
        else
            m = _mm_cvttpd_epu64(a.derived().m);
    }
#endif

    ENOKI_CONVERT(int64_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1);
        m = _mm_castps_si128(_mm_setr_ps(v0, v0, v1, v1));
    }

    ENOKI_REINTERPRET(int32_t) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1);
        m = _mm_setr_epi32(v0, v0, v1, v1);
    }

    ENOKI_REINTERPRET(uint32_t) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1);
        m = _mm_setr_epi32((int32_t) v0, (int32_t) v0, (int32_t) v1,
                           (int32_t) v1);
    }

    ENOKI_REINTERPRET(double) : m(_mm_castpd_si128(a.derived().m)) { }
    ENOKI_REINTERPRET(int64_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2) {
        alignas(16) Scalar data[2];
        data[0] = (Scalar) a1.coeff(0);
        data[1] = (Scalar) a2.coeff(0);
        m = _mm_load_si128((__m128i *) data);
    }

    ENOKI_INLINE Array1 low_()  const { return Array1(coeff(0)); }
    ENOKI_INLINE Array2 high_() const { return Array2(coeff(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm_add_epi64(m, a.m);   }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm_sub_epi64(m, a.m);   }
    ENOKI_INLINE Derived mul_(Arg a) const {
        #if defined(__AVX512DQ__) && defined(__AVX512VL__)
            return _mm_mullo_epi64(m, a.m);
        #else
            Derived result;
            ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                result.coeff(i) = coeff(i) * a.coeff(i);
            return result;
        #endif
    }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm_or_si128(m, a.m);    }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm_and_si128(m, a.m);   }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm_xor_si128(m, a.m);   }

    template <size_t k> ENOKI_INLINE Derived sli_() const {
        return _mm_slli_epi64(m, k);
    }

    template <size_t k> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Scalar>::value) {
            #if defined(__AVX512VL__)
                return _mm_srai_epi64(m, k);
            #else
                Derived result;
                ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                    result.coeff(i) = coeff(i) >> k;
                return result;
            #endif
        } else {
            return _mm_srli_epi64(m, k);
        }
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        return _mm_sll_epi64(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        if (std::is_signed<Scalar>::value) {
            #if defined(__AVX512VL__)
                return _mm_sra_epi64(m, _mm_set1_epi64x((long long) k));
            #else
                Derived result;
                ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                    result.coeff(i) = coeff(i) >> k;
                return result;
            #endif
        } else {
            return _mm_srl_epi64(m, _mm_set1_epi64x((long long) k));
        }
    }

    ENOKI_INLINE Derived slv_(Arg k) const {
        #if defined(__AVX2__)
            return _mm_sllv_epi64(m, k.m);
        #else
            Derived out;
            ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                out.coeff(i) = coeff(i) << (unsigned int) k.coeff(i);
            return out;
        #endif
    }

    ENOKI_INLINE Derived srv_(Arg k) const {
        if (std::is_signed<Scalar>::value) {
            #if defined(__AVX512VL__)
                return _mm_srav_epi64(m, k.m);
            #endif
        } else {
            #if defined(__AVX2__)
                return _mm_srlv_epi64(m, k.m);
            #endif
        }
        Derived out;
        ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
            out.coeff(i) = coeff(i) >> (unsigned int) k.coeff(i);
        return out;
    }

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived rolv_(Arg k) const { return _mm_rolv_epi64(m, k.m); }
    ENOKI_INLINE Derived rorv_(Arg k) const { return _mm_rorv_epi64(m, k.m); }
    ENOKI_INLINE Derived rol_(size_t k) const { return rolv_(_mm_set1_epi64x((long long) k)); }
    ENOKI_INLINE Derived ror_(size_t k) const { return rorv_(_mm_set1_epi64x((long long) k)); }
    template <size_t Imm>
    ENOKI_INLINE Derived roli_(Arg k) const { return _mm_rol_epi64(m, (int) k); }
    template <size_t Imm>
    ENOKI_INLINE Derived rori_(Arg k) const { return _mm_ror_epi64(m, (int) k); }
#endif

    ENOKI_INLINE Mask lt_(Arg a) const {
        if (std::is_signed<Scalar>::value) {
            return _mm_cmpgt_epi64(a.m, m);
        } else {
            const __m128i offset = _mm_set1_epi64x((long long) 0x8000000000000000ull);
            return _mm_cmpgt_epi64(
                _mm_sub_epi64(a.m, offset),
                _mm_sub_epi64(m, offset)
            );
        }
    }

    ENOKI_INLINE Mask gt_(Arg a) const {
        if (std::is_signed<Scalar>::value) {
            return _mm_cmpgt_epi64(m, a.m);
        } else {
            const __m128i offset = _mm_set1_epi64x((long long) 0x8000000000000000ull);
            return _mm_cmpgt_epi64(
                _mm_sub_epi64(m, offset),
                _mm_sub_epi64(a.m, offset)
            );
        }
    }

    ENOKI_INLINE Mask le_(Arg a) const { return ~gt_(a); }
    ENOKI_INLINE Mask ge_(Arg a) const { return ~lt_(a); }

    ENOKI_INLINE Mask eq_(Arg a)  const { return _mm_cmpeq_epi64(m, a.m); }
    ENOKI_INLINE Mask neq_(Arg a) const { return ~eq_(a); }

    ENOKI_INLINE Derived min_(Arg a) const {
        #if defined(__AVX512VL__)
            if (std::is_signed<Scalar>::value)
                return _mm_min_epi64(a.m, m);
            else
                return _mm_min_epu32(a.m, m);
        #else
            return select(derived() < a, derived(), a);
        #endif
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        #if defined(__AVX512VL__)
            if (std::is_signed<Scalar>::value)
                return _mm_max_epi64(a.m, m);
            else
                return _mm_max_epu32(a.m, m);
        #else
            return select(derived() > a, derived(), a);
        #endif
    }

    ENOKI_INLINE Derived abs_() const {
        if (!std::is_signed<Scalar>::value)
            return m;
        #if defined(__AVX512VL__)
            return _mm_abs_epi64(m);
        #else
        return select(derived() < zero<Derived>(),
                      ~derived() + Derived(Scalar(1)), derived());
        #endif
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm_blendv_epi8(f.m, t.m, m.m);
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        ENOKI_SCALAR return Derived(
            mulhi(coeff(0), a.coeff(0)),
            mulhi(coeff(1), a.coeff(1))
        );
    }

    template <int I0, int I1>
    ENOKI_INLINE Derived shuffle_() const {
        return _mm_shuffle_epi32(
            m, _MM_SHUFFLE(I1 * 2 + 1, I1 * 2, I0 * 2 + 1, I0 * 2));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            Scalar t1 = Scalar(_mm_extract_epi64(m, 1)); \
            Scalar t2 = Scalar(_mm_cvtsi128_si64(m)); \
            return op; \
        }

    ENOKI_HORIZONTAL_OP(hsum,  t1 + t2)
    ENOKI_HORIZONTAL_OP(hprod, t1 * t2)
    ENOKI_HORIZONTAL_OP(hmin,  min(t1, t2))
    ENOKI_HORIZONTAL_OP(hmax,  max(t1, t2))

    #undef ENOKI_HORIZONTAL_OP

    ENOKI_INLINE bool all_() const { return _mm_movemask_pd(_mm_castsi128_pd(m)) == 0x3;}
    ENOKI_INLINE bool any_() const { return _mm_movemask_pd(_mm_castsi128_pd(m)) != 0x0; }
    ENOKI_INLINE bool none_() const { return _mm_movemask_pd(_mm_castsi128_pd(m)) == 0x0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm_movemask_pd(_mm_castsi128_pd(m)));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm_store_si128((__m128i *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm_storeu_si128((__m128i *) ptr, m); }

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm_load_si128((const __m128i *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm_loadu_si128((const __m128i *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm_setzero_si128(); }

#if defined(__AVX2__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return Base::template gather_<Stride>(ptr, index);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Base::template gather_<Stride>(ptr, index, mask);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm_i64gather_epi64((const long long *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm_mask_i64gather_epi64(_mm_setzero_si128(), (const long long *) ptr,
                                        index.m, mask.m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        Base::template scatter_<Stride>(ptr, index);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        Base::template scatter_<Stride>(ptr, index, mask);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm_i64scatter_epi64(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        __mmask8 k = _mm_test_epi64_mask(mask.m, mask.m);
        _mm_mask_i64scatter_epi64(ptr, k, index.m, m, Stride);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl for the n=3 case (single precision)
template <bool Approx, typename Derived> struct alignas(16)
    StaticArrayImpl<float, 3, Approx, RoundingMode::Default, Derived>
    : StaticArrayImpl<float, 4, Approx, RoundingMode::Default, Derived> {
    using Base = StaticArrayImpl<float, 4, Approx, RoundingMode::Default, Derived>;

    using typename Base::Scalar;
    using typename Base::Mask;
    using Arg = const Base &;
    using Base::Base;
    using Base::m;
    using Base::operator=;
    using Base::coeff;
    static constexpr size_t Size = 3;

    ENOKI_INLINE StaticArrayImpl(Scalar f0, Scalar f1, Scalar f2) : Base(f0, f1, f2, Scalar(0)) { }
    ENOKI_INLINE StaticArrayImpl() : Base() { }

    StaticArrayImpl(const StaticArrayImpl &) = default;
    StaticArrayImpl &operator=(const StaticArrayImpl &) = default;

    template <
        typename Type2, bool Approx2, RoundingMode Mode2, typename Derived2>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, 3, Approx2, Mode2, Derived2> &a) {
        ENOKI_SCALAR for (size_t i = 0; i < 3; ++i)
            coeff(i) = Scalar(a.derived().coeff(i));
    }

    template <int I0, int I1, int I2>
    ENOKI_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128 t1 = _mm_movehl_ps(m, m); \
            __m128 t2 = _mm_##op##_ss(m, t1); \
            t1        = _mm_movehdup_ps(m); \
            t1        = _mm_##op##_ss(t1, t2); \
            return _mm_cvtss_f32(t1); \
        }

    ENOKI_HORIZONTAL_OP(hsum, add)
    ENOKI_HORIZONTAL_OP(hprod, mul)
    ENOKI_HORIZONTAL_OP(hmin, min)
    ENOKI_HORIZONTAL_OP(hmax, max)

    #undef ENOKI_HORIZONTAL_OP

    ENOKI_INLINE Scalar dot_(Arg a) const {
        return _mm_cvtss_f32(_mm_dp_ps(m, a.m, 0b01110001));
    }

    ENOKI_INLINE bool all_() const { return (_mm_movemask_ps(m) & 7) == 7; }
    ENOKI_INLINE bool any_() const { return (_mm_movemask_ps(m) & 7) != 0; }
    ENOKI_INLINE bool none_() const { return (_mm_movemask_ps(m) & 7) == 0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) (_mm_movemask_ps(m) & 7));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Loading/writing data (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { memcpy(ptr, &m, sizeof(Scalar)*3); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { store_(ptr); }
    ENOKI_INLINE static Derived load_(const void *ptr) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Scalar) * 3);
        return result;
    }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return load_(ptr); }

    static ENOKI_INLINE auto mask_() {
        return typename Derived::Mask(
            _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0)));
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    ENOKI_INLINE static void prefetch_(const void *ptr, const Index &index) {
        Base::template prefetch_<Stride, Write, Level>(ptr, index, mask_());
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    ENOKI_INLINE static void prefetch_(const void *ptr, const Index &index, const Mask &mask) {
        Base::template prefetch_<Stride, Write, Level>(ptr, index, mask & mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return Base::template gather_<Stride>(ptr, index, mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Base::template gather_<Stride>(ptr, index, mask & mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        Base::template scatter_<Stride>(ptr, index, mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        Base::template scatter_<Stride>(ptr, index, mask & mask_());
    }

    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        return Base::store_compress_(ptr, mask & mask_());
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl for the n=3 case (32 bit integers)
template <typename Scalar_, typename Derived> struct alignas(16)
    StaticArrayImpl<Scalar_, 3, false, RoundingMode::Default, Derived,detail::is_int32_t<Scalar_>>
    : StaticArrayImpl<Scalar_, 4, false, RoundingMode::Default, Derived> {
    using Base = StaticArrayImpl<Scalar_, 4, false, RoundingMode::Default, Derived>;

    using typename Base::Scalar;
    using typename Base::Mask;
    using Arg = const Base &;
    using Base::Base;
    using Base::m;
    using Base::operator=;
    using Base::coeff;
    static constexpr size_t Size = 3;

    ENOKI_INLINE StaticArrayImpl(Scalar f0, Scalar f1, Scalar f2) : Base(f0, f1, f2, Scalar(0)) { }
    ENOKI_INLINE StaticArrayImpl() : Base() { }

    StaticArrayImpl(const StaticArrayImpl &) = default;
    StaticArrayImpl &operator=(const StaticArrayImpl &) = default;

    template <
        typename Type2, bool Approx2, RoundingMode Mode2, typename Derived2>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, 3, Approx2, Mode2, Derived2> &a) {
        ENOKI_SCALAR for (size_t i = 0; i < 3; ++i)
            coeff(i) = Scalar(a.derived().coeff(i));
    }

    template <int I0, int I1, int I2>
    ENOKI_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128i t1 = _mm_unpackhi_epi32(m, m); \
            __m128i t2 = _mm_##op##_epi32(m, t1); \
            t1         = _mm_shuffle_epi32(m, 1); \
            t1         = _mm_##op##_epi32(t1, t2); \
            return (Scalar) _mm_cvtsi128_si32(t1); \
        }

    #define ENOKI_HORIZONTAL_OP_SIGNED(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128i t2, t1 = _mm_unpackhi_epi32(m, m); \
            if (std::is_signed<Scalar>::value) \
                t2 = _mm_##op##_epi32(m, t1); \
            else \
                t2 = _mm_##op##_epu32(m, t1); \
            t1 = _mm_shuffle_epi32(m, 1); \
            if (std::is_signed<Scalar>::value) \
                t1 = _mm_##op##_epi32(t1, t2); \
            else \
                t1 = _mm_##op##_epu32(t1, t2); \
            return (Scalar) _mm_cvtsi128_si32(t1); \
        }

    ENOKI_HORIZONTAL_OP(hsum, add)
    ENOKI_HORIZONTAL_OP(hprod, mullo)
    ENOKI_HORIZONTAL_OP_SIGNED(hmin, min)
    ENOKI_HORIZONTAL_OP_SIGNED(hmax, max)

    #undef ENOKI_HORIZONTAL_OP
    #undef ENOKI_HORIZONTAL_OP_SIGNED

    ENOKI_INLINE bool all_() const { return (_mm_movemask_ps(_mm_castsi128_ps(m)) & 7) == 7;}
    ENOKI_INLINE bool any_() const { return (_mm_movemask_ps(_mm_castsi128_ps(m)) & 7) != 0; }
    ENOKI_INLINE bool none_() const { return (_mm_movemask_ps(_mm_castsi128_ps(m)) & 7) == 0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) (_mm_movemask_ps(_mm_castsi128_ps(m)) & 7));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Loading/writing data (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { memcpy(ptr, &m, sizeof(Scalar)*3); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { store_(ptr); }
    ENOKI_INLINE static Derived load_(const void *ptr) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Scalar) * 3);
        return result;
    }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return load_(ptr); }

    static ENOKI_INLINE auto mask_() {
        return typename Derived::Mask(_mm_setr_epi32(-1, -1, -1, 0));
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    ENOKI_INLINE static void prefetch_(const void *ptr, const Index &index) {
        Base::template prefetch_<Stride, Write, Level>(ptr, index, mask_());
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    ENOKI_INLINE static void prefetch_(const void *ptr, const Index &index, const Mask &mask) {
        Base::template prefetch_<Stride, Write, Level>(ptr, index, mask & mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return Base::template gather_<Stride>(ptr, index, mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Base::template gather_<Stride>(ptr, index, mask & mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        Base::template scatter_<Stride>(ptr, index, mask_());
    }

    template <size_t Stride, typename Index>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        Base::template scatter_<Stride>(ptr, index, mask & mask_());
    }

    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        return Base::store_compress_(ptr, mask & mask_());
    }

    //! @}
    // -----------------------------------------------------------------------
};

NAMESPACE_END(enoki)
