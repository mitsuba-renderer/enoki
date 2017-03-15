/*
    enoki/array_avx.h -- Packed SIMD array (AVX2 specialization)

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "array_avx.h"

NAMESPACE_BEGIN(enoki)

NAMESPACE_BEGIN(detail)

template <typename T> struct is_native<T, 8, is_int32_t<T>> : std::true_type { };
template <typename T> struct is_native<T, 4, is_int64_t<T>> : std::true_type { };
template <typename T> struct is_native<T, 3, is_int64_t<T>> : std::true_type { };

NAMESPACE_END(detail)

/// Partial overload of StaticArrayImpl using AVX2 intrinsics (32 bit integers)
template <typename Value_, typename Derived>
struct alignas(32) StaticArrayImpl<Value_, 8, false, RoundingMode::Default,
                                   Derived, detail::is_int32_t<Value_>>
    : StaticArrayBase<Value_, 8, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY_CLASSIC(Value_, 8, false, __m256i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value) : m(_mm256_set1_epi32((int32_t) value)) { }
    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3,
                                 Value v4, Value v5, Value v6, Value v7)
        : m(_mm256_setr_epi32((int32_t) v0, (int32_t) v1, (int32_t) v2, (int32_t) v3,
                              (int32_t) v4, (int32_t) v5, (int32_t) v6, (int32_t) v7)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(float) {
        if (std::is_signed<Value>::value) {
            m = _mm256_cvttps_epi32(a.derived().m);
        } else {
            #if defined(__AVX512VL__)
                m = _mm256_cvttps_epu32(a.derived().m);
            #else
                ENOKI_TRACK_SCALAR for (size_t i = 0; i < Size; ++i)
                    coeff(i) = Value(a.derived().coeff(i));
            #endif
        }
    }

    ENOKI_CONVERT(int32_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint32_t) : m(a.derived().m) { }

    ENOKI_CONVERT(double) {
        if (std::is_signed<Value>::value) {
            #if defined(__AVX512F__)
                m = _mm512_cvttpd_epi32(a.derived().m);
            #else
                m = detail::concat(_mm256_cvttpd_epi32(low(a).m),
                                   _mm256_cvttpd_epi32(high(a).m));
            #endif
        } else {
            #if defined(__AVX512F__)
                m = _mm512_cvttpd_epu32(a.derived().m);
            #else
                ENOKI_TRACK_SCALAR for (size_t i = 0; i < Size; ++i)
                    coeff(i) = Value(a.derived().coeff(i));
            #endif
        }
    }

    ENOKI_CONVERT(int64_t) {
        #if defined(__AVX512F__)
            m = _mm512_cvtepi64_epi32(a.derived().m);
        #else
            m = detail::mm512_cvtepi64_epi32(low(a).m, high(a).m);
        #endif
    }

    ENOKI_CONVERT(uint64_t) {
        #if defined(__AVX512F__)
            m = _mm512_cvtepi64_epi32(a.derived().m);
        #else
            m = detail::mm512_cvtepi64_epi32(low(a).m, high(a).m);
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float) : m(_mm256_castps_si256(a.derived().m)) { }
    ENOKI_REINTERPRET(int32_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint32_t) : m(a.derived().m) { }

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_REINTERPRET(detail::KMaskBit) : m(_mm256_movm_epi32(a.derived().k)) { }
#elif defined(__AVX512F__)
    ENOKI_REINTERPRET(detail::KMaskBit)
        : m(_mm512_castsi512_si256(_mm512_maskz_mov_epi32(
              (__mmask16) a.derived().k, _mm512_set1_epi32(int32_t(-1))))) { }
#else
    ENOKI_REINTERPRET(double)
        : m(detail::mm512_cvtepi64_epi32(_mm256_castpd_si256(low(a).m),
                                         _mm256_castpd_si256(high(a).m))) { }
    ENOKI_REINTERPRET(int64_t)
        : m(detail::mm512_cvtepi64_epi32(low(a).m, high(a).m)) { }
    ENOKI_REINTERPRET(uint64_t)
        : m(detail::mm512_cvtepi64_epi32(low(a).m, high(a).m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm256_castsi256_si128(m); }
    ENOKI_INLINE Array2 high_() const { return _mm256_extractf128_si256(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm256_add_epi32(m, a.m);   }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm256_sub_epi32(m, a.m);   }
    ENOKI_INLINE Derived mul_(Arg a) const { return _mm256_mullo_epi32(m, a.m); }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm256_or_si256(m, a.m);    }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm256_and_si256(m, a.m);   }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm256_xor_si256(m, a.m);   }

    template <size_t k> ENOKI_INLINE Derived sli_() const {
        return _mm256_slli_epi32(m, (int) k);
    }

    template <size_t k> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Value>::value)
            return _mm256_srai_epi32(m, (int) k);
        else
            return _mm256_srli_epi32(m, (int) k);
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        return _mm256_sll_epi32(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        if (std::is_signed<Value>::value)
            return _mm256_sra_epi32(m, _mm_set1_epi64x((long long) k));
        else
            return _mm256_srl_epi32(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived slv_(Arg k) const {
        return _mm256_sllv_epi32(m, k.m);
    }

    ENOKI_INLINE Derived srv_(Arg k) const {
        if (std::is_signed<Value>::value)
            return _mm256_srav_epi32(m, k.m);
        else
            return _mm256_srlv_epi32(m, k.m);
    }

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived rolv_(Arg k) const { return _mm256_rolv_epi32(m, k.m); }
    ENOKI_INLINE Derived rorv_(Arg k) const { return _mm256_rorv_epi32(m, k.m); }
    ENOKI_INLINE Derived rol_(size_t k) const { return rolv_(_mm256_set1_epi32((int32_t) k)); }
    ENOKI_INLINE Derived ror_(size_t k) const { return rorv_(_mm256_set1_epi32((int32_t) k)); }
    template <size_t Imm>
    ENOKI_INLINE Derived roli_() const { return _mm256_rol_epi32(m, (int) Imm); }
    template <size_t Imm>
    ENOKI_INLINE Derived rori_() const { return _mm256_ror_epi32(m, (int) Imm); }
#endif

    ENOKI_INLINE Mask lt_(Arg a) const {
        if (std::is_signed<Value>::value) {
            return _mm256_cmpgt_epi32(a.m, m);
        } else {
            const __m256i offset = _mm256_set1_epi32((int32_t) 0x80000000ul);
            return _mm256_cmpgt_epi32(_mm256_sub_epi32(a.m, offset),
                                      _mm256_sub_epi32(m, offset));
        }
    }

    ENOKI_INLINE Mask gt_(Arg a) const {
        if (std::is_signed<Value>::value) {
            return _mm256_cmpgt_epi32(m, a.m);
        } else {
            const __m256i offset = _mm256_set1_epi32((int32_t) 0x80000000ul);
            return _mm256_cmpgt_epi32(_mm256_sub_epi32(m, offset),
                                      _mm256_sub_epi32(a.m, offset));
        }
    }

    ENOKI_INLINE Mask le_(Arg a) const { return ~gt_(a); }
    ENOKI_INLINE Mask ge_(Arg a) const { return ~lt_(a); }

    ENOKI_INLINE Mask eq_(Arg a)  const { return _mm256_cmpeq_epi32(m, a.m); }
    ENOKI_INLINE Mask neq_(Arg a) const { return ~eq_(a); }

    ENOKI_INLINE Derived min_(Arg a) const {
        if (std::is_signed<Value>::value)
            return _mm256_min_epi32(a.m, m);
        else
            return _mm256_min_epu32(a.m, m);
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        if (std::is_signed<Value>::value)
            return _mm256_max_epi32(a.m, m);
        else
            return _mm256_max_epu32(a.m, m);
    }

    ENOKI_INLINE Derived abs_() const {
        return std::is_signed<Value>::value ? _mm256_abs_epi32(m) : m;
    }

    static ENOKI_INLINE Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm256_blendv_epi8(f.m, t.m, m.m);
    }

    template <int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7>
    ENOKI_INLINE Derived shuffle_() const {
        return _mm256_permutevar8x32_epi32(m,
            _mm256_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7));
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        const Mask blend(Value(-1), 0, Value(-1), 0, Value(-1), 0, Value(-1), 0);

        if (std::is_signed<Value>::value) {
            Derived even(_mm256_srli_epi64(_mm256_mul_epi32(m, a.m), 32));
            Derived odd(_mm256_mul_epi32(_mm256_srli_epi64(m, 32), _mm256_srli_epi64(a.m, 32)));
            return select(blend, even, odd);
        } else {
            Derived even(_mm256_srli_epi64(_mm256_mul_epu32(m, a.m), 32));
            Derived odd(_mm256_mul_epu32(_mm256_srli_epi64(m, 32), _mm256_srli_epi64(a.m, 32)));
            return select(blend, even, odd);
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_()  const { return _mm256_testc_si256(m, _mm256_set1_epi32(-1)); }
    ENOKI_INLINE bool any_()  const { return !_mm256_testz_si256(m, m); }
    ENOKI_INLINE bool none_() const { return _mm256_testz_si256(m, m); }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm256_movemask_ps(_mm256_castsi256_ps(m)));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm256_store_si256((__m256i *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm256_storeu_si256((__m256i *) ptr, m); }

    static ENOKI_INLINE Derived load_(const void *ptr) { return _mm256_load_si256((const __m256i *) ptr); }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) { return _mm256_loadu_si256((const __m256i *) ptr); }

    static ENOKI_INLINE Derived zero_() { return _mm256_setzero_si256(); }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i32gather_epi32((const int *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i32gather_epi32(
            _mm256_setzero_si256(), (const int *) ptr, index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        #if defined(__AVX512F__)
            return _mm512_i64gather_epi32(index.m, ptr, Stride);
        #else
            return Derived(
                _mm256_i64gather_epi32((const int *) ptr, low(index).m, Stride),
                _mm256_i64gather_epi32((const int *) ptr, high(index).m, Stride)
            );
        #endif
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask_) {
        #if defined(__AVX512VL__) && defined(__AVX512DQ__)
            __mmask8 k = _mm256_movepi32_mask(mask_.m);
            return _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), k, index.m, (const float *) ptr, Stride);
        #else
            return Derived(
                _mm256_mask_i64gather_epi32(_mm_setzero_si128(), (const int *) ptr, low(index).m, low(mask_).m, Stride),
                _mm256_mask_i64gather_epi32(_mm_setzero_si128(), (const int *) ptr, high(index).m, high(mask_).m, Stride)
            );
        #endif
    }

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm256_i32scatter_epi32(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        __mmask8 k = _mm256_test_epi32_mask(mask.m, mask.m);
        _mm256_mask_i32scatter_epi32(ptr, k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm512_i64scatter_epi32(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        __mmask8 k = _mm256_test_epi32_mask(mask.m, mask.m);
        _mm512_mask_i64scatter_epi32(ptr, k, index.m, m, Stride);
    }
#endif

    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        #if defined(__AVX512VL__)
            __mmask8 k = _mm256_test_epi32_mask(mask.m, mask.m);
            _mm256_storeu_si256((__m256i *) ptr,
                             _mm256_mask_compress_epi32(_mm256_setzero_si256(), k, m));
            (Value *&) ptr += _mm_popcnt_u32(k);
        #else
            /** Fancy LUT-based partitioning algorithm, see http://stackoverflow.com/a/36949578/1130282 */
            const __m256i shift = _mm256_setr_epi32(29, 26, 23, 20, 17, 14, 11, 8);
            unsigned int k = (unsigned int) _mm256_movemask_ps(_mm256_castsi256_ps(mask.m));

            __m256i tmp  = _mm256_set1_epi32(*((int32_t *) (detail::compress_lut_256 + k*3)));
            __m256i shuf = _mm256_srli_epi32(_mm256_sllv_epi32(tmp, shift), 29);
            __m256i perm  = _mm256_permutevar8x32_epi32(m, shuf);

            _mm256_storeu_si256((__m256i *) ptr, perm);
            (Value *&) ptr += _mm_popcnt_u32(k);
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using AVX2 intrinsics (64 bit integers)
template <typename Value_, typename Derived>
struct alignas(32) StaticArrayImpl<Value_, 4, false, RoundingMode::Default,
                                   Derived, detail::is_int64_t<Value_>>
    : StaticArrayBase<Value_, 4, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY_CLASSIC(Value_, 4, false, __m256i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value)
        : m(_mm256_set1_epi64x((int64_t) value)) { }

    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m(_mm256_setr_epi64x((int64_t) v0, (int64_t) v1,
                               (int64_t) v2, (int64_t) v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(float) {
        if (std::is_signed<Value>::value)
            m = _mm256_cvttps_epi64(a.derived().m);
        else
            m = _mm256_cvttps_epu64(a.derived().m);
    }

    ENOKI_CONVERT(double) {
        if (std::is_signed<Value>::value)
            m = _mm256_cvttpd_epi64(a.derived().m);
        else
            m = _mm256_cvttpd_epu64(a.derived().m);
    }
#endif
    ENOKI_CONVERT(int32_t)  : m(_mm256_cvtepi32_epi64(a.derived().m)) { }
    ENOKI_CONVERT(uint32_t) : m(_mm256_cvtepu32_epi64(a.derived().m)) { }

    ENOKI_CONVERT(int64_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET(float)
        : m(_mm256_cvtepi32_epi64(_mm_castps_si128(a.derived().m))) { }
    ENOKI_REINTERPRET(int32_t) : m(_mm256_cvtepi32_epi64(a.derived().m)) { }
    ENOKI_REINTERPRET(uint32_t) : m(_mm256_cvtepi32_epi64(a.derived().m)) { }

    ENOKI_REINTERPRET(double) : m(_mm256_castpd_si256(a.derived().m)) { }
    ENOKI_REINTERPRET(int64_t) : m(a.derived().m) { }
    ENOKI_REINTERPRET(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm256_castsi256_si128(m); }
    ENOKI_INLINE Array2 high_() const { return _mm256_extractf128_si256(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Arg a) const { return _mm256_add_epi64(m, a.m);   }
    ENOKI_INLINE Derived sub_(Arg a) const { return _mm256_sub_epi64(m, a.m);   }
    ENOKI_INLINE Derived mul_(Arg a) const {
        #if defined(__AVX512DQ__) && defined(__AVX512VL__)
            return _mm256_mullo_epi64(m, a.m);
        #else
            __m256i h0    = _mm256_srli_epi64(m, 32);
            __m256i h1    = _mm256_srli_epi64(a.m, 32);
            __m256i low   = _mm256_mul_epu32(m, a.m);
            __m256i mix0  = _mm256_mul_epu32(m, h1);
            __m256i mix1  = _mm256_mul_epu32(h0, a.m);
            __m256i mix   = _mm256_add_epi64(mix0, mix1);
            __m256i mix_s = _mm256_slli_epi64(mix, 32);
            return  _mm256_add_epi64(mix_s, low);
        #endif
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        /* Signed high multiplication is too costly to emulate
           using intrinsics, fall back to scalar version */

        if (std::is_signed<Value>::value) {
            Derived result;
            for (size_t i = 0; i < Size; ++i)
                result.coeff(i) = mulhi(coeff(i), a.coeff(i));
            return result;
        }

        const __m256i low_bits = _mm256_set1_epi64x(0xffffffffu);
        __m256i al = m, bl = a.m;

        __m256i ah = _mm256_srli_epi64(al, 32);
        __m256i bh = _mm256_srli_epi64(bl, 32);

        // 4x unsigned 32x32->64 bit multiplication
        __m256i albl = _mm256_mul_epu32(al, bl);
        __m256i albh = _mm256_mul_epu32(al, bh);
        __m256i ahbl = _mm256_mul_epu32(ah, bl);
        __m256i ahbh = _mm256_mul_epu32(ah, bh);

        // Calculate a possible carry from the low bits of the multiplication.
        __m256i carry = _mm256_add_epi64(
            _mm256_srli_epi64(albl, 32),
            _mm256_add_epi64(_mm256_and_si256(albh, low_bits),
                             _mm256_and_si256(ahbl, low_bits)));

        __m256i s0 = _mm256_add_epi64(ahbh, _mm256_srli_epi64(carry, 32));
        __m256i s1 = _mm256_add_epi64(_mm256_srli_epi64(albh, 32),
                                      _mm256_srli_epi64(ahbl, 32));

        return _mm256_add_epi64(s0, s1);
    }

    ENOKI_INLINE Derived or_ (Arg a) const { return _mm256_or_si256(m, a.m);    }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm256_and_si256(m, a.m);   }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm256_xor_si256(m, a.m);   }

    template <size_t k> ENOKI_INLINE Derived sli_() const {
        return _mm256_slli_epi64(m, (int) k);
    }

    template <size_t k> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Value>::value) {
            #if defined(__AVX512VL__)
                return _mm256_srai_epi64(m, (int) k);
            #else
                Derived result;
                ENOKI_TRACK_SCALAR for (size_t i = 0; i < Size; ++i)
                    result.coeff(i) = coeff(i) >> k;
                return result;
            #endif
        } else {
            return _mm256_srli_epi64(m, (int) k);
        }
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        return _mm256_sll_epi64(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        if (std::is_signed<Value>::value) {
            #if defined(__AVX512VL__)
                return _mm256_sra_epi64(m, _mm_set1_epi64x((long long) k));
            #else
                Derived result;
                ENOKI_TRACK_SCALAR for (size_t i = 0; i < Size; ++i)
                    result.coeff(i) = coeff(i) >> k;
                return result;
            #endif
        } else {
            return _mm256_srl_epi64(m, _mm_set1_epi64x((long long) k));
        }
    }

    ENOKI_INLINE Derived slv_(Arg k) const {
        return _mm256_sllv_epi64(m, k.m);
    }

    ENOKI_INLINE Derived srv_(Arg k) const {
        if (std::is_signed<Value>::value) {
            #if defined(__AVX512VL__)
                return _mm256_srav_epi64(m, k.m);
            #else
                Derived out;
                ENOKI_TRACK_SCALAR for (size_t i = 0; i < Size; ++i)
                    out.coeff(i) = coeff(i) >> (size_t) k.coeff(i);
                return out;
            #endif
        } else {
            return _mm256_srlv_epi64(m, k.m);
        }
    }

#if defined(__AVX512VL__)
    ENOKI_INLINE Derived rolv_(Arg k) const { return _mm256_rolv_epi64(m, k.m); }
    ENOKI_INLINE Derived rorv_(Arg k) const { return _mm256_rorv_epi64(m, k.m); }
    ENOKI_INLINE Derived rol_(size_t k) const { return rolv_(_mm256_set1_epi64x((long long) k)); }
    ENOKI_INLINE Derived ror_(size_t k) const { return rorv_(_mm256_set1_epi64x((long long) k)); }
    template <size_t Imm>
    ENOKI_INLINE Derived roli_() const { return _mm256_rol_epi64(m, (int) Imm); }
    template <size_t Imm>
    ENOKI_INLINE Derived rori_() const { return _mm256_ror_epi64(m, (int) Imm); }
#endif

    ENOKI_INLINE Mask lt_(Arg a) const {
        if (std::is_signed<Value>::value) {
            return _mm256_cmpgt_epi64(a.m, m);
        } else {
            const __m256i offset = _mm256_set1_epi64x((long long) 0x8000000000000000ull);
            return _mm256_cmpgt_epi64(
                _mm256_sub_epi64(a.m, offset),
                _mm256_sub_epi64(m, offset)
            );
        }
    }

    ENOKI_INLINE Mask gt_(Arg a) const {
        if (std::is_signed<Value>::value) {
            return _mm256_cmpgt_epi64(m, a.m);
        } else {
            const __m256i offset = _mm256_set1_epi64x((long long) 0x8000000000000000ull);
            return _mm256_cmpgt_epi64(
                _mm256_sub_epi64(m, offset),
                _mm256_sub_epi64(a.m, offset)
            );
        }
    }

    ENOKI_INLINE Mask le_(Arg a) const { return ~gt_(a); }
    ENOKI_INLINE Mask ge_(Arg a) const { return ~lt_(a); }

    ENOKI_INLINE Mask eq_(Arg a)  const { return _mm256_cmpeq_epi64(m, a.m); }
    ENOKI_INLINE Mask neq_(Arg a) const { return ~eq_(a); }

    ENOKI_INLINE Derived min_(Arg a) const {
        #if defined(__AVX512VL__)
            if (std::is_signed<Value>::value)
                return _mm256_min_epi64(a.m, m);
            else
                return _mm256_min_epu32(a.m, m);
        #else
            return select(derived() < a, derived(), a);
        #endif
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        #if defined(__AVX512VL__)
            if (std::is_signed<Value>::value)
                return _mm256_max_epi64(a.m, m);
            else
                return _mm256_max_epu32(a.m, m);
        #else
            return select(derived() > a, derived(), a);
        #endif
    }

    ENOKI_INLINE Derived abs_() const {
        if (!std::is_signed<Value>::value)
            return m;
        #if defined(__AVX512VL__)
            return _mm256_abs_epi64(m);
        #else
            return select(derived() < zero<Derived>(),
                          ~derived() + Derived(Value(1)), derived());
        #endif
    }

    static ENOKI_INLINE Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm256_blendv_epi8(f.m, t.m, m.m);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_()  const { return _mm256_testc_si256(m, _mm256_set1_epi32(-1)); }
    ENOKI_INLINE bool any_()  const { return !_mm256_testz_si256(m, m); }
    ENOKI_INLINE bool none_() const { return _mm256_testz_si256(m, m); }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) _mm256_movemask_pd(_mm256_castsi256_pd(m)));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { _mm256_store_si256((__m256i *) ptr, m); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { _mm256_storeu_si256((__m256i *) ptr, m); }

    static ENOKI_INLINE Derived load_(const void *ptr) { return _mm256_load_si256((const __m256i *) ptr); }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) { return _mm256_loadu_si256((const __m256i *) ptr); }

    static ENOKI_INLINE Derived zero_() { return _mm256_setzero_si256(); }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i32gather_epi64((const long long *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i32gather_epi64(
            _mm256_setzero_si256(), (const long long *) ptr, index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i64gather_epi64((const long long *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i64gather_epi64(_mm256_setzero_si256(), (const long long *) ptr,
                                        index.m, mask.m, Stride);
    }

#if defined(__AVX512VL__)
    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm256_i32scatter_epi64(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        __mmask8 k = _mm256_test_epi64_mask(mask.m, mask.m);
        _mm256_mask_i32scatter_epi64(ptr, k, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index) const {
        _mm256_i64scatter_epi64(ptr, index.m, m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        __mmask8 k = _mm256_test_epi64_mask(mask.m, mask.m);
        _mm256_mask_i64scatter_epi64(ptr, k, index.m, m, Stride);
    }
#endif

#if defined(__AVX512VL__)
    ENOKI_INLINE void store_compress_(void *&ptr, const Mask &mask) const {
        __mmask8 k = _mm256_test_epi64_mask(mask.m, mask.m);
        _mm256_storeu_si256((__m256i *) ptr, _mm256_mask_compress_epi64(
                                                 _mm256_setzero_si256(), k, m));
        (Value *&) ptr += _mm_popcnt_u32(k);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl for the n=3 case (64 bit integers)
template <typename Value_, typename Derived> struct alignas(32)
    StaticArrayImpl<Value_, 3, false, RoundingMode::Default, Derived, detail::is_int64_t<Value_>>
    : StaticArrayImpl<Value_, 4, false, RoundingMode::Default, Derived> {
    using Base = StaticArrayImpl<Value_, 4, false, RoundingMode::Default, Derived>;

    using typename Base::Value;
    using typename Base::Mask;
    using Arg = const Base &;
    using Base::Base;
    using Base::m;
    using Base::operator=;
    using Base::coeff;
    static constexpr size_t Size = 3;

    ENOKI_INLINE StaticArrayImpl(Value f0, Value f1, Value f2) : Base(f0, f1, f2, Value(0)) { }
    ENOKI_INLINE StaticArrayImpl() : Base() { }

    StaticArrayImpl(const StaticArrayImpl &) = default;
    StaticArrayImpl &operator=(const StaticArrayImpl &) = default;

    template <
        typename Type2, bool Approx2, RoundingMode Mode2, typename Derived2>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, 3, Approx2, Mode2, Derived2> &a) {
        ENOKI_TRACK_SCALAR for (size_t i = 0; i < 3; ++i)
            coeff(i) = Value(a.derived().coeff(i));
    }

    template <int I0, int I1, int I2>
    ENOKI_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_() const {
        Value result = coeff(0);
        for (size_t i = 1; i < 3; ++i)
            result += coeff(i);
        return result;
    }

    ENOKI_INLINE Value hprod_() const {
        Value result = coeff(0);
        for (size_t i = 1; i < 3; ++i)
            result *= coeff(i);
        return result;
    }

    ENOKI_INLINE Value hmin_() const {
        Value result = coeff(0);
        for (size_t i = 1; i < 3; ++i)
            result = std::min(result, coeff(i));
        return result;
    }

    ENOKI_INLINE Value hmax_() const {
        Value result = coeff(0);
        for (size_t i = 1; i < 3; ++i)
            result = std::max(result, coeff(i));
        return result;
    }

    ENOKI_INLINE bool all_()  const { return (_mm256_movemask_pd(_mm256_castsi256_pd(m)) & 7) == 7;}
    ENOKI_INLINE bool any_()  const { return (_mm256_movemask_pd(_mm256_castsi256_pd(m)) & 7) != 0; }
    ENOKI_INLINE bool none_() const { return (_mm256_movemask_pd(_mm256_castsi256_pd(m)) & 7) == 0; }

    ENOKI_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32((unsigned int) (_mm256_movemask_pd(_mm256_castsi256_pd(m)) & 7));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Loading/writing data (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const { memcpy(ptr, &m, sizeof(Value)*3); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { store_(ptr); }
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Value) * 3);
        return result;
    }
    static ENOKI_INLINE Derived load_(const void *ptr) { return Base::load_unaligned_(ptr); }

    static ENOKI_INLINE auto mask_() {
        return typename Derived::Mask(_mm256_setr_epi64x(
            (int64_t) -1, (int64_t) -1, (int64_t) -1, (int64_t) 0));
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index) {
        Base::template prefetch_<Stride, Write, Level>(ptr, index, mask_());
    }

    template <size_t Stride, bool Write, size_t Level, typename Index>
    static ENOKI_INLINE void prefetch_(const void *ptr, const Index &index, const Mask &mask) {
        Base::template prefetch_<Stride, Write, Level>(ptr, index, mask & mask_());
    }

    template <size_t Stride, typename Index>
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index) {
        return Base::template gather_<Stride>(ptr, index, mask_());
    }

    template <size_t Stride, typename Index>
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
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
