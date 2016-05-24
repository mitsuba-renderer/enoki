/*
    enoki/array_avx.h -- Packed SIMD array (AVX specialization)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "array_sse42.h"

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

/// Compressed look-up table for the store_compress() operation [769 bytes]
alignas(32) const uint8_t compress_lut_256[3 * 256 + 1] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x08, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x10, 0x00, 0x00, 0x11, 0x00, 0x00, 0x88, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x18, 0x00, 0x00, 0x19, 0x00, 0x00, 0xc8, 0x00, 0x00,
    0x1a, 0x00, 0x00, 0xd0, 0x00, 0x00, 0xd1, 0x00, 0x00, 0x88, 0x06, 0x00,
    0x04, 0x00, 0x00, 0x20, 0x00, 0x00, 0x21, 0x00, 0x00, 0x08, 0x01, 0x00,
    0x22, 0x00, 0x00, 0x10, 0x01, 0x00, 0x11, 0x01, 0x00, 0x88, 0x08, 0x00,
    0x23, 0x00, 0x00, 0x18, 0x01, 0x00, 0x19, 0x01, 0x00, 0xc8, 0x08, 0x00,
    0x1a, 0x01, 0x00, 0xd0, 0x08, 0x00, 0xd1, 0x08, 0x00, 0x88, 0x46, 0x00,
    0x05, 0x00, 0x00, 0x28, 0x00, 0x00, 0x29, 0x00, 0x00, 0x48, 0x01, 0x00,
    0x2a, 0x00, 0x00, 0x50, 0x01, 0x00, 0x51, 0x01, 0x00, 0x88, 0x0a, 0x00,
    0x2b, 0x00, 0x00, 0x58, 0x01, 0x00, 0x59, 0x01, 0x00, 0xc8, 0x0a, 0x00,
    0x5a, 0x01, 0x00, 0xd0, 0x0a, 0x00, 0xd1, 0x0a, 0x00, 0x88, 0x56, 0x00,
    0x2c, 0x00, 0x00, 0x60, 0x01, 0x00, 0x61, 0x01, 0x00, 0x08, 0x0b, 0x00,
    0x62, 0x01, 0x00, 0x10, 0x0b, 0x00, 0x11, 0x0b, 0x00, 0x88, 0x58, 0x00,
    0x63, 0x01, 0x00, 0x18, 0x0b, 0x00, 0x19, 0x0b, 0x00, 0xc8, 0x58, 0x00,
    0x1a, 0x0b, 0x00, 0xd0, 0x58, 0x00, 0xd1, 0x58, 0x00, 0x88, 0xc6, 0x02,
    0x06, 0x00, 0x00, 0x30, 0x00, 0x00, 0x31, 0x00, 0x00, 0x88, 0x01, 0x00,
    0x32, 0x00, 0x00, 0x90, 0x01, 0x00, 0x91, 0x01, 0x00, 0x88, 0x0c, 0x00,
    0x33, 0x00, 0x00, 0x98, 0x01, 0x00, 0x99, 0x01, 0x00, 0xc8, 0x0c, 0x00,
    0x9a, 0x01, 0x00, 0xd0, 0x0c, 0x00, 0xd1, 0x0c, 0x00, 0x88, 0x66, 0x00,
    0x34, 0x00, 0x00, 0xa0, 0x01, 0x00, 0xa1, 0x01, 0x00, 0x08, 0x0d, 0x00,
    0xa2, 0x01, 0x00, 0x10, 0x0d, 0x00, 0x11, 0x0d, 0x00, 0x88, 0x68, 0x00,
    0xa3, 0x01, 0x00, 0x18, 0x0d, 0x00, 0x19, 0x0d, 0x00, 0xc8, 0x68, 0x00,
    0x1a, 0x0d, 0x00, 0xd0, 0x68, 0x00, 0xd1, 0x68, 0x00, 0x88, 0x46, 0x03,
    0x35, 0x00, 0x00, 0xa8, 0x01, 0x00, 0xa9, 0x01, 0x00, 0x48, 0x0d, 0x00,
    0xaa, 0x01, 0x00, 0x50, 0x0d, 0x00, 0x51, 0x0d, 0x00, 0x88, 0x6a, 0x00,
    0xab, 0x01, 0x00, 0x58, 0x0d, 0x00, 0x59, 0x0d, 0x00, 0xc8, 0x6a, 0x00,
    0x5a, 0x0d, 0x00, 0xd0, 0x6a, 0x00, 0xd1, 0x6a, 0x00, 0x88, 0x56, 0x03,
    0xac, 0x01, 0x00, 0x60, 0x0d, 0x00, 0x61, 0x0d, 0x00, 0x08, 0x6b, 0x00,
    0x62, 0x0d, 0x00, 0x10, 0x6b, 0x00, 0x11, 0x6b, 0x00, 0x88, 0x58, 0x03,
    0x63, 0x0d, 0x00, 0x18, 0x6b, 0x00, 0x19, 0x6b, 0x00, 0xc8, 0x58, 0x03,
    0x1a, 0x6b, 0x00, 0xd0, 0x58, 0x03, 0xd1, 0x58, 0x03, 0x88, 0xc6, 0x1a,
    0x07, 0x00, 0x00, 0x38, 0x00, 0x00, 0x39, 0x00, 0x00, 0xc8, 0x01, 0x00,
    0x3a, 0x00, 0x00, 0xd0, 0x01, 0x00, 0xd1, 0x01, 0x00, 0x88, 0x0e, 0x00,
    0x3b, 0x00, 0x00, 0xd8, 0x01, 0x00, 0xd9, 0x01, 0x00, 0xc8, 0x0e, 0x00,
    0xda, 0x01, 0x00, 0xd0, 0x0e, 0x00, 0xd1, 0x0e, 0x00, 0x88, 0x76, 0x00,
    0x3c, 0x00, 0x00, 0xe0, 0x01, 0x00, 0xe1, 0x01, 0x00, 0x08, 0x0f, 0x00,
    0xe2, 0x01, 0x00, 0x10, 0x0f, 0x00, 0x11, 0x0f, 0x00, 0x88, 0x78, 0x00,
    0xe3, 0x01, 0x00, 0x18, 0x0f, 0x00, 0x19, 0x0f, 0x00, 0xc8, 0x78, 0x00,
    0x1a, 0x0f, 0x00, 0xd0, 0x78, 0x00, 0xd1, 0x78, 0x00, 0x88, 0xc6, 0x03,
    0x3d, 0x00, 0x00, 0xe8, 0x01, 0x00, 0xe9, 0x01, 0x00, 0x48, 0x0f, 0x00,
    0xea, 0x01, 0x00, 0x50, 0x0f, 0x00, 0x51, 0x0f, 0x00, 0x88, 0x7a, 0x00,
    0xeb, 0x01, 0x00, 0x58, 0x0f, 0x00, 0x59, 0x0f, 0x00, 0xc8, 0x7a, 0x00,
    0x5a, 0x0f, 0x00, 0xd0, 0x7a, 0x00, 0xd1, 0x7a, 0x00, 0x88, 0xd6, 0x03,
    0xec, 0x01, 0x00, 0x60, 0x0f, 0x00, 0x61, 0x0f, 0x00, 0x08, 0x7b, 0x00,
    0x62, 0x0f, 0x00, 0x10, 0x7b, 0x00, 0x11, 0x7b, 0x00, 0x88, 0xd8, 0x03,
    0x63, 0x0f, 0x00, 0x18, 0x7b, 0x00, 0x19, 0x7b, 0x00, 0xc8, 0xd8, 0x03,
    0x1a, 0x7b, 0x00, 0xd0, 0xd8, 0x03, 0xd1, 0xd8, 0x03, 0x88, 0xc6, 0x1e,
    0x3e, 0x00, 0x00, 0xf0, 0x01, 0x00, 0xf1, 0x01, 0x00, 0x88, 0x0f, 0x00,
    0xf2, 0x01, 0x00, 0x90, 0x0f, 0x00, 0x91, 0x0f, 0x00, 0x88, 0x7c, 0x00,
    0xf3, 0x01, 0x00, 0x98, 0x0f, 0x00, 0x99, 0x0f, 0x00, 0xc8, 0x7c, 0x00,
    0x9a, 0x0f, 0x00, 0xd0, 0x7c, 0x00, 0xd1, 0x7c, 0x00, 0x88, 0xe6, 0x03,
    0xf4, 0x01, 0x00, 0xa0, 0x0f, 0x00, 0xa1, 0x0f, 0x00, 0x08, 0x7d, 0x00,
    0xa2, 0x0f, 0x00, 0x10, 0x7d, 0x00, 0x11, 0x7d, 0x00, 0x88, 0xe8, 0x03,
    0xa3, 0x0f, 0x00, 0x18, 0x7d, 0x00, 0x19, 0x7d, 0x00, 0xc8, 0xe8, 0x03,
    0x1a, 0x7d, 0x00, 0xd0, 0xe8, 0x03, 0xd1, 0xe8, 0x03, 0x88, 0x46, 0x1f,
    0xf5, 0x01, 0x00, 0xa8, 0x0f, 0x00, 0xa9, 0x0f, 0x00, 0x48, 0x7d, 0x00,
    0xaa, 0x0f, 0x00, 0x50, 0x7d, 0x00, 0x51, 0x7d, 0x00, 0x88, 0xea, 0x03,
    0xab, 0x0f, 0x00, 0x58, 0x7d, 0x00, 0x59, 0x7d, 0x00, 0xc8, 0xea, 0x03,
    0x5a, 0x7d, 0x00, 0xd0, 0xea, 0x03, 0xd1, 0xea, 0x03, 0x88, 0x56, 0x1f,
    0xac, 0x0f, 0x00, 0x60, 0x7d, 0x00, 0x61, 0x7d, 0x00, 0x08, 0xeb, 0x03,
    0x62, 0x7d, 0x00, 0x10, 0xeb, 0x03, 0x11, 0xeb, 0x03, 0x88, 0x58, 0x1f,
    0x63, 0x7d, 0x00, 0x18, 0xeb, 0x03, 0x19, 0xeb, 0x03, 0xc8, 0x58, 0x1f,
    0x1a, 0xeb, 0x03, 0xd0, 0x58, 0x1f, 0xd1, 0x58, 0x1f, 0x88, 0xc6, 0xfa,
    0x00
};

NAMESPACE_END(detail)

/// Partial overload of StaticArrayImpl using AVX intrinsics (single precision)
template <bool Approx, typename Derived> struct alignas(32)
    StaticArrayImpl<float, 8, Approx, RoundingMode::Default, Derived>
    : StaticArrayBase<float, 8, Approx, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(float, 8, Approx, __m256)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value) : m(_mm256_set1_ps(value)) { }
    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1, Scalar v2, Scalar v3,
                                 Scalar v4, Scalar v5, Scalar v6, Scalar v7)
        : m(_mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(float) : m(a.derived().m) { }

#if defined(__AVX2__)
    ENOKI_CONVERT(int32_t) : m(_mm256_cvtepi32_ps(a.derived().m)) { }
#endif

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(uint32_t) : m(_mm256_cvtepu32_ps(a.derived().m)) { }
#endif

    ENOKI_CONVERT(double)
        : m(_mm256_setr_m128(_mm256_cvtpd_ps(low(a).m),
                             _mm256_cvtpd_ps(high(a).m))) { }

#if defined(__AVX512DQ__)
    ENOKI_CONVERT(int64_t) : m(_mm512_cvtepi64_ps(a.derived().m)) { }
    ENOKI_CONVERT(uint64_t) : m(_mm512_cvtepu64_ps(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) : m(a.derived().m) { }

#if defined(__AVX2__)
    ENOKI_REINTERPRET(int32_t) : m(_mm256_castsi256_ps(a.derived().m)) { }
    ENOKI_REINTERPRET(uint32_t) : m(_mm256_castsi256_ps(a.derived().m)) { }
#else
    ENOKI_REINTERPRET(int32_t)
        : m(_mm256_setr_m128(_mm_castsi128_ps(low(a).m),
                             _mm_castsi128_ps(high(a).m))) { }

    ENOKI_REINTERPRET(uint32_t)
        : m(_mm256_setr_m128(_mm_castsi128_ps(low(a).m),
                             _mm_castsi128_ps(high(a).m))) { }
#endif

#if defined(__AVX512F__)
    // XXX this all needs to be replaced by masks
    ENOKI_REINTERPRET(double) :
        m(_mm256_castsi256_ps(_mm512_cvtepi64_epi32(_mm512_castpd_si512(a.derived().m)))) { }
#else
    ENOKI_REINTERPRET(double)
        : m(_mm256_castsi256_ps(
              detail::mm512_cvtepi64_epi32(_mm256_castpd_si256(low(a).m),
                                           _mm256_castpd_si256(high(a).m)))) { }
#endif

#if defined(__AVX512F__)
    // XXX this all needs to be replaced by masks
    ENOKI_REINTERPRET(uint64_t) :
        m(_mm256_castsi256_ps(_mm512_cvtepi64_epi32(a.derived().m))) { }
    ENOKI_REINTERPRET(int64_t) :
        m(_mm256_castsi256_ps(_mm512_cvtepi64_epi32(a.derived().m))) { }
#elif defined(__AVX2__)
    ENOKI_REINTERPRET(int64_t)
        : m(_mm256_castsi256_ps(
              detail::mm512_cvtepi64_epi32(low(a).m,
                                           high(a).m))) { }
    ENOKI_REINTERPRET(uint64_t)
        : m(_mm256_castsi256_ps(
              detail::mm512_cvtepi64_epi32(low(a).m,
                                           high(a).m))) { }
#else
    ENOKI_REINTERPRET(int64_t)
        : m(_mm256_castsi256_ps(detail::mm512_cvtepi64_epi32(
             low(low(a)).m, high(low(a)).m,
             low(high(a)).m, high(high(a)).m))) { }
    ENOKI_REINTERPRET(uint64_t)
        : m(_mm256_castsi256_ps(detail::mm512_cvtepi64_epi32(
             low(low(a)).m, high(low(a)).m,
             low(high(a)).m, high(high(a)).m))) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm256_setr_m128(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm256_castps256_ps128(m); }
    ENOKI_INLINE Array2 high_() const { return _mm256_extractf128_ps(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm256_add_ps(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm256_sub_ps(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm256_mul_ps(m, a.m); }
    ENOKI_INLINE Derived div_(Arg a) const { return _mm256_div_ps(m, a.m); }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm256_or_ps (m, a.m); }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm256_and_ps(m, a.m); }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm256_xor_ps(m, a.m); }

    ENOKI_INLINE Mask lt_ (Arg a) const { return _mm256_cmp_ps(m, a.m, _CMP_LT_OQ);  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return _mm256_cmp_ps(m, a.m, _CMP_GT_OQ);  }
    ENOKI_INLINE Mask le_ (Arg a) const { return _mm256_cmp_ps(m, a.m, _CMP_LE_OQ);  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return _mm256_cmp_ps(m, a.m, _CMP_GE_OQ);  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return _mm256_cmp_ps(m, a.m, _CMP_EQ_OQ);  }
    ENOKI_INLINE Mask neq_(Arg a) const { return _mm256_cmp_ps(m, a.m, _CMP_NEQ_UQ); }

    ENOKI_INLINE Derived abs_()      const { return _mm256_andnot_ps(_mm256_set1_ps(-0.f), m); }
    ENOKI_INLINE Derived min_(Arg b) const { return _mm256_min_ps(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return _mm256_max_ps(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm256_ceil_ps(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm256_floor_ps(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm256_sqrt_ps(m);     }

    ENOKI_INLINE Derived round_() const {
        return _mm256_round_ps(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm256_blendv_ps(f.m, t.m, m.m);
    }

#if defined(__FMA__)
    ENOKI_INLINE Derived fmadd_(Arg b, Arg c) const {
        return _mm256_fmadd_ps(m, b.m, c.m);
    }

    ENOKI_INLINE Derived fmsub_(Arg b, Arg c) const {
        return _mm256_fmsub_ps(m, b.m, c.m);
    }
#endif

#if defined(__AVX2__)
    template <int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7>
    ENOKI_INLINE Derived shuffle_() const {
        return _mm256_permutevar8x32_ps(m,
            _mm256_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7));
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived ldexp_(Arg arg) const { return _mm256_scalef_ps(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm256_getmant_ps(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm256_add_ps(_mm256_getexp_ps(m), _mm256_set1_ps(1.f)));
    }
#endif

    ENOKI_INLINE Derived rcp_() const {
        if (Approx) {
            /* Use best reciprocal approximation available on the current
               hardware and potentially refine */
            __m256 r;
            #if defined(__AVX512ER__)
                /* rel err < 2^28, use as is */
                r = _mm512_castps512_ps256(
                    _mm512_rcp28_ps(_mm512_castps256_ps512(m)));
            #elif defined(__AVX512VL__)
                r = _mm256_rcp14_ps(m); /* rel error < 2^-14 */
            #else
                r = _mm256_rcp_ps(m);   /* rel error < 1.5*2^-12 */
            #endif

            #if !defined(__AVX512ER__)
                /* Refine using one Newton-Raphson iteration */

                #if defined(__FMA__)
                    const __m256 two = _mm256_set1_ps(2.f);
                    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(r, m, two));
                #else
                    r = _mm256_sub_ps(_mm256_add_ps(r, r),
                                      _mm256_mul_ps(_mm256_mul_ps(r, r), m));
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
            __m256 r;
            #if defined(__AVX512ER__)
                /* rel err < 2^28, use as is */
                r = _mm512_castps512_ps256(
                    _mm512_rsqrt28_ps(_mm512_castps256_ps512(m)));
            #elif defined(__AVX512VL__)
                r = _mm256_rsqrt14_ps(m); /* rel error < 2^-14 */
            #else
                r = _mm256_rsqrt_ps(m);   /* rel error < 1.5*2^-12 */
            #endif

            #if !defined(__AVX512ER__)
                /* Refine using one Newton-Raphson iteration */

                const __m256 c0 = _mm256_set1_ps(1.5f);
                const __m256 c1 = _mm256_set1_ps(-0.5f);

                #if defined(__FMA__)
                    r = _mm256_fmadd_ps(
                        r, c0,
                        _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(m, c1), r),
                                      _mm256_mul_ps(r, r)));
                #else
                    r = _mm256_add_ps(
                        _mm256_mul_ps(c0, r),
                        _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(m, c1), r),
                                      _mm256_mul_ps(r, r)));
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
            return _mm512_castps512_ps256(
                _mm512_exp2a23_ps(_mm512_castps256_ps512(
                    _mm256_mul_ps(m, _mm256_set1_ps(1.4426950408889634074f)))));
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

    ENOKI_INLINE Scalar hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Scalar hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Scalar hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Scalar hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_() const { return _mm256_movemask_ps(m) == 0xFF; }
    ENOKI_INLINE bool any_() const { return _mm256_movemask_ps(m) != 0x00; }
    ENOKI_INLINE bool none_() const { return _mm256_movemask_ps(m) == 0x00; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm256_movemask_ps(m));
    }

    ENOKI_INLINE Scalar dot_(Arg a) const {
        __m256 dp = _mm256_dp_ps(m, a.m, 0b11110001);
        __m128 m0 = _mm256_castps256_ps128(dp);
        __m128 m1 = _mm256_extractf128_ps(dp, 1);
        __m128 m = _mm_add_ss(m0, m1);
        return _mm_cvtss_f32(m);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm256_store_ps((Scalar *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm256_storeu_ps((Scalar *) ptr, m); }

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm256_load_ps((const Scalar *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm256_loadu_ps((const Scalar *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm256_setzero_ps(); }

#if defined(__AVX2__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i32gather_ps((const float *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i32gather_ps(_mm256_setzero_ps(), (const float *) ptr, index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        #if defined(__AVX512F__)
            return _mm512_i64gather_ps(index.m, ptr, Stride);
        #else
            return Derived(
                _mm256_i64gather_ps((const float *) ptr, low(index).m, Stride),
                _mm256_i64gather_ps((const float *) ptr, high(index).m, Stride)
            );
        #endif
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask_) {
        #if defined(__AVX512F__)
            __m512i mask = _mm512_castps_si512(mask_.m);
            __mmask8 k = _mm512_test_epi64_mask(mask, mask);
            return _mm512_mask_i64gather_ps(_mm256_setzero_ps(), k, index.m, (const float *) ptr, Stride);
        #else
            return Derived(
                _mm256_mask_i64gather_ps(_mm_setzero_ps(), (const float *) ptr, low(index).m, low(mask_).m, Stride),
                _mm256_mask_i64gather_ps(_mm_setzero_ps(), (const float *) ptr, high(index).m, high(mask_).m, Stride)
            );
        #endif
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm256_i32scatter_ps(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask_) const {
        __m256i mask = _mm256_castps_si256(mask_.m);
        __mmask8 k = _mm256_test_epi32_mask(mask, mask);
        _mm256_mask_i32scatter_ps(ptr, k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i64scatter_ps(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask_) const {
        __m512i mask = _mm512_castps_si512(mask_.m);
        __mmask8 k = _mm512_test_epi64_mask(mask, mask);
        _mm256_mask_i64scatter_ps(ptr, k, index.m, m, Stride);
    }
#endif

    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        #if defined(__AVX512VL__)
            __mmask8 k = _mm_test_epi32_mask(_mm256_castps_si256(mask.m),
                                             _mm256_castps_si256(mask.m));
            _mm256_storeu_ps((float *) ptr,
                             _mm256_mask_compress_ps(_mm256_setzero_ps(), k, m));
            (float *&) ptr += _mm_popcnt_u32(k);
        #elif defined(__AVX2__)
            /** Fancy LUT-based partitioning algorithm, see http://stackoverflow.com/a/36949578/1130282 */
            const __m256i shift = _mm256_setr_epi32(29, 26, 23, 20, 17, 14, 11, 8);
            unsigned int offset = (unsigned int) _mm256_movemask_ps(mask.m);

            __m256i tmp  = _mm256_set1_epi32(*((int32_t *) (detail::compress_lut_256 + offset*3)));
            __m256i shuf = _mm256_srli_epi32(_mm256_sllv_epi32(tmp, shift), 29);
            __m256  perm  = _mm256_permutevar8x32_ps(m, shuf);

            _mm256_storeu_ps((float *) ptr, perm);
            (float *&) ptr += _mm_popcnt_u32(offset);
        #else
            store_compress(ptr, low(derived()), low(mask));
            store_compress(ptr, high(derived()), high(mask));
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using AVX intrinsics (double precision)
template <bool Approx, typename Derived> struct alignas(32)
    StaticArrayImpl<double, 4, Approx, RoundingMode::Default, Derived>
    : StaticArrayBase<double, 4, Approx, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(double, 4, Approx, __m256d)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value) : m(_mm256_set1_pd(value)) { }
    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1, Scalar v2, Scalar v3)
        : m(_mm256_setr_pd(v0, v1, v2, v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(float) : m(_mm256_cvtps_pd(a.derived().m)) { }

#if defined(__AVX2__)
    ENOKI_CONVERT(int32_t) : m(_mm256_cvtepi32_pd(a.derived().m)) { }
#endif

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(uint32_t) : m(_mm256_cvtepu32_pd(a.derived().m)) { }
#endif

    ENOKI_CONVERT(double) : m(a.derived().m) { }

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(int64_t) : m(_mm256_cvtepi64_pd(a.derived().m)) { }
    ENOKI_CONVERT(uint64_t) : m(_mm256_cvtepu64_pd(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float)
        : m(_mm256_castsi256_pd(
              detail::mm256_cvtepi32_epi64(_mm_castps_si128(a.derived().m)))) { }

    ENOKI_REINTERPRET(int32_t)
        : m(_mm256_castsi256_pd(detail::mm256_cvtepi32_epi64(a.derived().m))) { }

    ENOKI_REINTERPRET(uint32_t)
        : m(_mm256_castsi256_pd(detail::mm256_cvtepi32_epi64(a.derived().m))) { }

    ENOKI_REINTERPRET(double) : m(a.derived().m) { }

#if defined(__AVX2__)
    ENOKI_REINTERPRET(int64_t) : m(_mm256_castsi256_pd(a.derived().m)) { }
    ENOKI_REINTERPRET(uint64_t) : m(_mm256_castsi256_pd(a.derived().m)) { }
#else
    ENOKI_REINTERPRET(int64_t)
        : m(_mm256_setr_m128d(_mm_castsi128_pd(low(a).m),
                              _mm_castsi128_pd(high(a).m))) { }
    ENOKI_REINTERPRET(uint64_t)
        : m(_mm256_setr_m128d(_mm_castsi128_pd(low(a).m),
                              _mm_castsi128_pd(high(a).m))) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm256_setr_m128d(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm256_castpd256_pd128(m); }
    ENOKI_INLINE Array2 high_() const { return _mm256_extractf128_pd(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm256_add_pd(m, a.m); }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm256_sub_pd(m, a.m); }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm256_mul_pd(m, a.m); }
    ENOKI_INLINE Derived div_(Arg a) const { return _mm256_div_pd(m, a.m); }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm256_or_pd (m, a.m); }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm256_and_pd(m, a.m); }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm256_xor_pd(m, a.m); }

    ENOKI_INLINE Mask lt_ (Arg a) const { return _mm256_cmp_pd(m, a.m, _CMP_LT_OQ);  }
    ENOKI_INLINE Mask gt_ (Arg a) const { return _mm256_cmp_pd(m, a.m, _CMP_GT_OQ);  }
    ENOKI_INLINE Mask le_ (Arg a) const { return _mm256_cmp_pd(m, a.m, _CMP_LE_OQ);  }
    ENOKI_INLINE Mask ge_ (Arg a) const { return _mm256_cmp_pd(m, a.m, _CMP_GE_OQ);  }
    ENOKI_INLINE Mask eq_ (Arg a) const { return _mm256_cmp_pd(m, a.m, _CMP_EQ_OQ);  }
    ENOKI_INLINE Mask neq_(Arg a) const { return _mm256_cmp_pd(m, a.m, _CMP_NEQ_UQ); }

    ENOKI_INLINE Derived abs_()      const { return _mm256_andnot_pd(_mm256_set1_pd(-0.), m); }
    ENOKI_INLINE Derived min_(Arg b) const { return _mm256_min_pd(b.m, m); }
    ENOKI_INLINE Derived max_(Arg b) const { return _mm256_max_pd(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm256_ceil_pd(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm256_floor_pd(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm256_sqrt_pd(m);     }

    ENOKI_INLINE Derived round_() const {
        return _mm256_round_pd(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm256_blendv_pd(f.m, t.m, m.m);
    }

#if defined(__FMA__)
    ENOKI_INLINE Derived fmadd_(Arg b, Arg c) const {
        return _mm256_fmadd_pd(m, b.m, c.m);
    }

    ENOKI_INLINE Derived fmsub_(Arg b, Arg c) const {
        return _mm256_fmsub_pd(m, b.m, c.m);
    }
#endif

#if defined(__AVX2__)
    template <int I0, int I1, int I2, int I3>
    ENOKI_INLINE Derived shuffle_() const {
        return _mm256_permute4x64_pd(m, _MM_SHUFFLE(I3, I2, I1, I0));
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived ldexp_(Arg arg) const { return _mm256_scalef_pd(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm256_getmant_pd(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm256_add_pd(_mm256_getexp_pd(m), _mm256_set1_pd(1.0)));
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
            __m256 r;
            #if defined(__AVX512ER__)
                /* rel err < 2^28, use as is */
                r = _mm512_castpd512_pd256(
                    _mm512_rsqrt28_pd(_mm512_castpd256_pd512(m)));
            #elif defined(__AVX512VL__)
                r = _mm256_rsqrt14_pd(m); /* rel error < 2^-14 */
            #endif

            #if !defined(__AVX512ER__)
                /* Refine using two Newton-Raphson iterations */
                const __m256 c0 = _mm256_set1_pd(1.5);
                const __m256 c1 = _mm256_set1_pd(-0.5);

                for (int i = 0; i < 2; ++i) {
                    #if defined(__FMA__)
                        r = _mm256_fmadd_pd(r, c0,
                            _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(m, c1), r),
                                          _mm256_mul_pd(r, r)));
                    #else
                        r = _mm256_add_pd(
                            _mm256_mul_pd(c0, r),
                            _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(m, c1), r),
                                          _mm256_mul_pd(r, r)));
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
            return _mm512_castpd512_pd256(
                _mm512_exp2a23_pd(_mm512_castpd256_pd512(
                    _mm256_mul_pd(m, _mm256_set1_pd(1.4426950408889634074)))));
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

    ENOKI_INLINE Scalar hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Scalar hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Scalar hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Scalar hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_() const { return _mm256_movemask_pd(m) == 0xF; }
    ENOKI_INLINE bool any_() const { return _mm256_movemask_pd(m) != 0x0; }
    ENOKI_INLINE bool none_() const { return _mm256_movemask_pd(m) == 0x0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm256_movemask_pd(m));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm256_store_pd((Scalar *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm256_storeu_pd((Scalar *) ptr, m); }

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm256_load_pd((const Scalar *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm256_loadu_pd((const Scalar *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm256_setzero_pd(); }

#if defined(__AVX2__)
    ENOKI_REQUIRE_INDEX(T, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const T &index) {
        return _mm256_i32gather_pd((const double *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(T, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const T &index, const Mask &mask) {
        return _mm256_mask_i32gather_pd(_mm256_setzero_pd(), (const double *) ptr, index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(T, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const T &index) {
        return _mm256_i64gather_pd((const double *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(T, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const T &index, const Mask &mask) {
        return _mm256_mask_i64gather_pd(_mm256_setzero_pd(), (const double *) ptr, index.m, mask.m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(T, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const T &index) const {
        _mm256_i32scatter_pd(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(T, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const T &index, const Mask &mask_) const {
        __m256i mask = _mm256_castpd_si256(mask_.m);
        __mmask8 k = _mm256_test_epi64_mask(mask, mask);
        _mm256_mask_i32scatter_pd(ptr, k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(T, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const T &index) const {
        _mm256_i64scatter_pd(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(T, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const T &index, const Mask &mask_) const {
        __m256i mask = _mm256_castpd_si256(mask_.m);
        __mmask8 k = _mm256_test_epi64_mask(mask, mask);
        _mm256_mask_i64scatter_pd(ptr, k, index.m, m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        unsigned int offset = (unsigned int) _mm256_movemask_pd(mask.m);
        __mmask8 k = _mm256_test_epi64_mask(_mm256_castpd_si256(mask.m),
                                            _mm256_castpd_si256(mask.m));
        _mm256_storeu_pd((double *) ptr,
                      _mm256_mask_compress_pd(_mm256_setzero_pd(), k, m));

        (float *&) ptr += _mm_popcnt_u32(offset);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl for the n=3 case (double precision)
template <bool Approx, typename Derived> struct alignas(32)
    StaticArrayImpl<double, 3, Approx, RoundingMode::Default, Derived>
    : StaticArrayImpl<double, 4, Approx, RoundingMode::Default, Derived> {
    using Base = StaticArrayImpl<double, 4, Approx, RoundingMode::Default, Derived>;

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

#if !defined(__AVX2__)
    ENOKI_REINTERPRET(uint64_t) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1), v2 = a.derived().coeff(2);
        m = _mm256_castsi256_pd(_mm256_setr_epi64x((int64_t) v0, (int64_t) v1, (int64_t) v2, 0));
    }

    ENOKI_REINTERPRET(int64_t) {
        ENOKI_SCALAR
        auto v0 = a.derived().coeff(0), v1 = a.derived().coeff(1), v2 = a.derived().coeff(2);
        m = _mm256_castsi256_pd(_mm256_setr_epi64x((int64_t) v0, (int64_t) v1, (int64_t) v2, 0));
    }
#endif

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op) \
        ENOKI_INLINE Scalar name##_() const { \
            __m128d t1 = _mm256_extractf128_pd(m, 1); \
            __m128d t2 = _mm256_castpd256_pd128(m); \
            t1 = _mm_##op##_sd(t1, t2); \
            t2 = _mm_permute_pd(t2, 1); \
            t2 = _mm_##op##_sd(t2, t1); \
            return _mm_cvtsd_f64(t2); \
        }

    ENOKI_HORIZONTAL_OP(hsum, add)
    ENOKI_HORIZONTAL_OP(hprod, mul)
    ENOKI_HORIZONTAL_OP(hmin, min)
    ENOKI_HORIZONTAL_OP(hmax, max)

    #undef ENOKI_HORIZONTAL_OP

    ENOKI_INLINE bool all_() const { return (_mm256_movemask_pd(m) & 7) == 7; }
    ENOKI_INLINE bool any_() const { return (_mm256_movemask_pd(m) & 7) != 0; }
    ENOKI_INLINE bool none_() const { return (_mm256_movemask_pd(m) & 7) == 0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) (_mm256_movemask_pd(m) & 7));
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
            _mm256_castsi256_pd(_mm256_setr_epi64x(-1, -1, -1, 0)));
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
