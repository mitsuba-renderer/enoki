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

/// Partial overload of StaticArrayImpl using AVX2 intrinsics (32 bit integers)
template <typename Scalar_, typename Derived>
struct alignas(32) StaticArrayImpl<Scalar_, 8, false, RoundingMode::Default,
                                   Derived, detail::is_int32_t<Scalar_>>
    : StaticArrayBase<Scalar_, 8, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(Scalar_, 8, false, __m256i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value) : m(_mm256_set1_epi32((int32_t) value)) { }
    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1, Scalar v2, Scalar v3,
                                 Scalar v4, Scalar v5, Scalar v6, Scalar v7)
        : m(_mm256_setr_epi32((int32_t) v0, (int32_t) v1, (int32_t) v2, (int32_t) v3,
                              (int32_t) v4, (int32_t) v5, (int32_t) v6, (int32_t) v7)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    ENOKI_CONVERT(float) {
        if (std::is_signed<Scalar>::value) {
            m = _mm256_cvttps_epi32(a.derived().m);
        } else {
            #if defined(__AVX512VL__)
                m = _mm256_cvttps_epu32(a.derived().m);
            #else
                ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                    coeff(i) = Scalar(a.derived().coeff(i));
            #endif
        }
    }

    ENOKI_CONVERT(int32_t) : m(a.derived().m) { }
    ENOKI_CONVERT(uint32_t) : m(a.derived().m) { }

    ENOKI_CONVERT(double) {
        if (std::is_signed<Scalar>::value) {
            #if defined(__AVX512F__)
                m = _mm512_cvttpd_epi32(a.derived().m);
            #else
                m = _mm256_setr_m128i(_mm256_cvttpd_epi32(low(a).m),
                                      _mm256_cvttpd_epi32(high(a).m));
            #endif
        } else {
            #if defined(__AVX512F__)
                m = _mm512_cvttpd_epu32(a.derived().m);
            #else
                ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                    coeff(i) = Scalar(a.derived().coeff(i));
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

#if defined(__AVX512F__)
    // XXX this all needs to be replaced by masks
    ENOKI_REINTERPRET(double) :
        m(_mm512_cvtepi64_epi32(_mm512_castpd_si512(a.derived().m))) { }
#else
    ENOKI_REINTERPRET(double)
        : m(detail::mm512_cvtepi64_epi32(_mm256_castpd_si256(low(a).m),
                                         _mm256_castpd_si256(high(a).m))) { }
#endif

#if defined(__AVX512F__)
    // XXX this all needs to be replaced by masks
    ENOKI_REINTERPRET(uint64_t) :
        m(_mm512_cvtepi64_epi32(a.derived().m)) { }
    ENOKI_REINTERPRET(int64_t) :
        m(_mm512_cvtepi64_epi32(a.derived().m)) { }
#else
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
        : m(_mm256_setr_m128i(a1.m, a2.m)) { }

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
        if (std::is_signed<Scalar>::value)
            return _mm256_srai_epi32(m, (int) k);
        else
            return _mm256_srli_epi32(m, (int) k);
    }

    ENOKI_INLINE Derived sl_(size_t k) const {
        return _mm256_sll_epi32(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived sr_(size_t k) const {
        if (std::is_signed<Scalar>::value)
            return _mm256_sra_epi32(m, _mm_set1_epi64x((long long) k));
        else
            return _mm256_srl_epi32(m, _mm_set1_epi64x((long long) k));
    }

    ENOKI_INLINE Derived slv_(Arg k) const {
        return _mm256_sllv_epi32(m, k.m);
    }

    ENOKI_INLINE Derived srv_(Arg k) const {
        if (std::is_signed<Scalar>::value)
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
    ENOKI_INLINE Derived roli_(Arg k) const { return _mm256_rol_epi32(m, (int) k); }
    template <size_t Imm>
    ENOKI_INLINE Derived rori_(Arg k) const { return _mm256_ror_epi32(m, (int) k); }
#endif

    ENOKI_INLINE Mask lt_(Arg a) const {
        if (std::is_signed<Scalar>::value) {
            return _mm256_cmpgt_epi32(a.m, m);
        } else {
            const __m256i offset = _mm256_set1_epi32((int32_t) 0x80000000ul);
            return _mm256_cmpgt_epi32(_mm256_sub_epi32(a.m, offset),
                                      _mm256_sub_epi32(m, offset));
        }
    }

    ENOKI_INLINE Mask gt_(Arg a) const {
        if (std::is_signed<Scalar>::value) {
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
        if (std::is_signed<Scalar>::value)
            return _mm256_min_epi32(a.m, m);
        else
            return _mm256_min_epu32(a.m, m);
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        if (std::is_signed<Scalar>::value)
            return _mm256_max_epi32(a.m, m);
        else
            return _mm256_max_epu32(a.m, m);
    }

    ENOKI_INLINE Derived abs_() const {
        return std::is_signed<Scalar>::value ? _mm256_abs_epi32(m) : m;
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm256_blendv_epi8(f.m, t.m, m.m);
    }

    template <int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7>
    ENOKI_INLINE Derived shuffle_() const {
        return _mm256_permutevar8x32_epi32(m,
            _mm256_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7));
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        const Mask blend(Scalar(-1), 0, Scalar(-1), 0, Scalar(-1), 0, Scalar(-1), 0);

        if (std::is_signed<Scalar>::value) {
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

    ENOKI_INLINE Scalar hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Scalar hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Scalar hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Scalar hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_()  const { return _mm256_movemask_ps(_mm256_castsi256_ps(m)) == 0xFF; }
    ENOKI_INLINE bool any_()  const { return _mm256_movemask_ps(_mm256_castsi256_ps(m)) != 0x00; }
    ENOKI_INLINE bool none_() const { return _mm256_movemask_ps(_mm256_castsi256_ps(m)) == 0x00; }

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

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm256_load_si256((const __m256i *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm256_loadu_si256((const __m256i *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm256_setzero_si256(); }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i32gather_epi32((const int *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i32gather_epi32(
            _mm256_setzero_si256(), (const int *) ptr, index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
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
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask_) {
        #if defined(__AVX512F__)
            __m512i mask = _mm512_castps_si512(mask_.m);
            __mmask8 k = _mm512_test_epi64_mask(mask, mask);
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
            __mmask8 k = _mm_test_epi32_mask(mask.m, mask.m);
            _mm256_storeu_si256((float *) ptr, _mm256_mask_compress_epi32(
                                                   _mm256_setzero_si256(), k, m));
            (float *&) ptr += _mm_popcnt_u32(k);
        #else
            store_compress(ptr, low(derived()), low(mask));
            store_compress(ptr, high(derived()), high(mask));
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl using AVX2 intrinsics (64 bit integers)
template <typename Scalar_, typename Derived>
struct alignas(32) StaticArrayImpl<Scalar_, 4, false, RoundingMode::Default,
                                   Derived, detail::is_int64_t<Scalar_>>
    : StaticArrayBase<Scalar_, 4, false, RoundingMode::Default, Derived> {
    ENOKI_NATIVE_ARRAY(Scalar_, 4, false, __m256i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Scalar value)
        : m(_mm256_set1_epi64x((int64_t) value)) { }

    ENOKI_INLINE StaticArrayImpl(Scalar v0, Scalar v1, Scalar v2, Scalar v3)
        : m(_mm256_setr_epi64x((int64_t) v0, (int64_t) v1,
                               (int64_t) v2, (int64_t) v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    ENOKI_CONVERT(float) {
        if (std::is_signed<Scalar>::value)
            m = _mm256_cvttps_epi64(a.derived().m);
        else
            m = _mm256_cvttps_epu64(a.derived().m);
    }

    ENOKI_CONVERT(double) {
        if (std::is_signed<Scalar>::value)
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
     : m(_mm256_setr_m128i(a1.m, a2.m)) { }

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
            Derived result;
            ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
                result.coeff(i) = coeff(i) * a.coeff(i);
            return result;
        #endif
    }
    ENOKI_INLINE Derived or_ (Arg a) const { return _mm256_or_si256(m, a.m);    }
    ENOKI_INLINE Derived and_(Arg a) const { return _mm256_and_si256(m, a.m);   }
    ENOKI_INLINE Derived xor_(Arg a) const { return _mm256_xor_si256(m, a.m);   }

    template <size_t k> ENOKI_INLINE Derived sli_() const {
        return _mm256_slli_epi64(m, (int) k);
    }

    template <size_t k> ENOKI_INLINE Derived sri_() const {
        if (std::is_signed<Scalar>::value) {
            #if defined(__AVX512VL__)
                return _mm256_srai_epi64(m, (int) k);
            #else
                Derived result;
                ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
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
        if (std::is_signed<Scalar>::value) {
            #if defined(__AVX512VL__)
                return _mm256_sra_epi64(m, _mm_set1_epi64x((long long) k));
            #else
                Derived result;
                ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
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
        if (std::is_signed<Scalar>::value) {
            #if defined(__AVX512VL__)
                return _mm256_srav_epi64(m, k.m);
            #else
                Derived out;
                ENOKI_SCALAR for (size_t i = 0; i < Size; ++i)
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
    ENOKI_INLINE Derived roli_(Arg k) const { return _mm256_rol_epi64(m, (int) k); }
    template <size_t Imm>
    ENOKI_INLINE Derived rori_(Arg k) const { return _mm256_ror_epi64(m, (int) k); }
#endif

    ENOKI_INLINE Mask lt_(Arg a) const {
        if (std::is_signed<Scalar>::value) {
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
        if (std::is_signed<Scalar>::value) {
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
            if (std::is_signed<Scalar>::value)
                return _mm256_min_epi64(a.m, m);
            else
                return _mm256_min_epu32(a.m, m);
        #else
            return select(derived() < a, derived(), a);
        #endif
    }

    ENOKI_INLINE Derived max_(Arg a) const {
        #if defined(__AVX512VL__)
            if (std::is_signed<Scalar>::value)
                return _mm256_max_epi64(a.m, m);
            else
                return _mm256_max_epu32(a.m, m);
        #else
            return select(derived() > a, derived(), a);
        #endif
    }

    ENOKI_INLINE Derived abs_() const {
        if (!std::is_signed<Scalar>::value)
            return m;
        #if defined(__AVX512VL__)
            return _mm256_abs_epi64(m);
        #else
            return select(derived() < zero<Derived>(),
                          ~derived() + Derived(Scalar(1)), derived());
        #endif
    }

    ENOKI_INLINE static Derived select_(const Mask &m, Arg t, Arg f) {
        return _mm256_blendv_epi8(f.m, t.m, m.m);
    }

    ENOKI_INLINE Derived mulhi_(Arg a) const {
        ENOKI_SCALAR return Derived(
            mulhi(coeff(0), a.coeff(0)),
            mulhi(coeff(1), a.coeff(1)),
            mulhi(coeff(2), a.coeff(2)),
            mulhi(coeff(3), a.coeff(3))
        );
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Scalar hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Scalar hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Scalar hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Scalar hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_()  const { return _mm256_movemask_pd(_mm256_castsi256_pd(m)) == 0xF; }
    ENOKI_INLINE bool any_()  const { return _mm256_movemask_pd(_mm256_castsi256_pd(m)) != 0x0; }
    ENOKI_INLINE bool none_() const { return _mm256_movemask_pd(_mm256_castsi256_pd(m)) == 0x0; }

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

    ENOKI_INLINE static Derived load_(const void *ptr) { return _mm256_load_si256((const __m256i *) ptr); }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return _mm256_loadu_si256((const __m256i *) ptr); }

    ENOKI_INLINE static Derived zero_() { return _mm256_setzero_si256(); }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i32gather_epi64((const long long *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int32_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return _mm256_mask_i32gather_epi64(
            _mm256_setzero_si256(), (const long long *) ptr, index.m, mask.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index) {
        return _mm256_i64gather_epi64((const long long *) ptr, index.m, Stride);
    }

    ENOKI_REQUIRE_INDEX(Index, int64_t)
    ENOKI_INLINE static Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
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

    //! @}
    // -----------------------------------------------------------------------
};

/// Partial overload of StaticArrayImpl for the n=3 case (64 bit integers)
template <typename Scalar_, typename Derived> struct alignas(32)
    StaticArrayImpl<Scalar_, 3, false, RoundingMode::Default, Derived, detail::is_int64_t<Scalar_>>
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

    ENOKI_INLINE Scalar hsum_() const {
        Scalar result = coeff(0);
        for (size_t i = 1; i < 3; ++i)
            result += coeff(i);
        return result;
    }

    ENOKI_INLINE Scalar hprod_() const {
        Scalar result = coeff(0);
        for (size_t i = 1; i < 3; ++i)
            result *= coeff(i);
        return result;
    }

    ENOKI_INLINE Scalar hmin_() const {
        Scalar result = coeff(0);
        for (size_t i = 1; i < 3; ++i)
            result = std::min(result, coeff(i));
        return result;
    }

    ENOKI_INLINE Scalar hmax_() const {
        Scalar result = coeff(0);
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

    ENOKI_INLINE void store_(void *ptr) const { memcpy(ptr, &m, sizeof(Scalar)*3); }
    ENOKI_INLINE void store_unaligned_(void *ptr) const { store_(ptr); }
    ENOKI_INLINE static Derived load_(const void *ptr) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Scalar) * 3);
        return result;
    }
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) { return load_(ptr); }

    static ENOKI_INLINE auto mask_() {
        return typename Derived::Mask(_mm256_setr_epi64x(
            (int64_t) -1, (int64_t) -1, (int64_t) -1, (int64_t) 0));
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
