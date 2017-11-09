/*
    enoki/array_generic.h -- Generic array implementation that forwards
    all operations to the underlying data type (usually without making use of
    hardware vectorization)

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array_base.h"

NAMESPACE_BEGIN(enoki)

NAMESPACE_BEGIN(detail)

#define ENOKI_DECLARE_ARRAY(Base, Derived)                                     \
    static constexpr bool Is##Derived = true;                                  \
    using Base::Base;                                                          \
    using Base::operator=;                                                     \
    Derived() = default;                                                       \
    Derived(const Derived &) = default;                                        \
    Derived(Derived &&) = default;                                             \
    Derived &operator=(const Derived &) = default;                             \
    Derived &operator=(Derived &&) = default;                                  \
    template <typename T, std::enable_if_t<                                    \
              std::is_constructible<Derived, T>::value, int> = 0>              \
    Derived& operator=(const T &value) { return operator=(Derived(value)); }

NAMESPACE_END(detail)

template <typename, size_t, bool, RoundingMode, typename, typename>
struct StaticArrayImpl { /* Will never be instantiated */ };

template <typename Value_, size_t Size_, bool Approx_, typename Derived_>
struct StaticArrayImpl<
    Value_, Size_, Approx_, RoundingMode::Default, Derived_,
    std::enable_if_t<!detail::is_native<Value_, Size_>::value &&
                     !std::is_enum<Value_>::value &&
                     !(std::is_pointer<Value_>::value && !std::is_arithmetic<std::remove_pointer_t<Value_>>::value) &&
                     !detail::is_recursive<Value_, Size_, RoundingMode::Default>::value>>
    : StaticArrayBase<Value_, Size_, Approx_, RoundingMode::Default, Derived_> {

    using Base = StaticArrayBase<Value_, Size_, Approx_, RoundingMode::Default, Derived_>;

    using Base::operator=;
    using typename Base::Value;
    using typename Base::Element;
    using typename Base::Derived;
    using typename Base::Scalar;
    using typename Base::Array1;
    using typename Base::Array2;
    using Base::Size;
    using Base::Size1;
    using Base::Size2;
    using Base::data;

    using StorageType =
        std::conditional_t<std::is_reference<Value_>::value,
                           std::reference_wrapper<Element>, Value>;

    // -----------------------------------------------------------------------
    //! @{ \name Default constructors and assignment operators
    // -----------------------------------------------------------------------

    ENOKI_TRIVIAL_CONSTRUCTOR(Value_)

    /// Default copy constructor
    StaticArrayImpl(const StaticArrayImpl &t) = default;

    /// Default move constructor
    StaticArrayImpl(StaticArrayImpl &&t) = default;

    /// Default copy assignment operator
    StaticArrayImpl& operator=(const StaticArrayImpl &) = default;

    /// Default move assignment operator
    StaticArrayImpl& operator=(StaticArrayImpl &&) = default;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors that convert from compatible representations
    // -----------------------------------------------------------------------

    /// Initialize the components individually
    template <typename Arg, typename... Args,
              /* Ugly, works around a compiler ICE in MSVC */
              std::enable_if_t<
                  detail::all_of<std::is_constructible<StorageType, Arg>::value,
                                 (std::is_constructible<StorageType, Args>::value && !std::is_same<Args, detail::reinterpret_flag>::value)...,
                                 sizeof...(Args) + 1 == Size_ && (sizeof...(Args) > 0)>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(Arg&& arg, Args&&... args)
        : m_data{{ (StorageType) std::forward<Arg>(arg),
                   (StorageType) std::forward<Args>(args)... }} { ENOKI_CHKSCALAR }

private:
    template <typename T, size_t... Index>
    ENOKI_INLINE StaticArrayImpl(T &&t, std::index_sequence<Index...>)
        : m_data{{ (detail::ref_cast_t<std::decay_t<value_t<T>>, StorageType>) t.derived().coeff(Index)... }} { }

public:
    /// Convert a compatible array type (const, non-recursive)
    template <
        typename Value2, size_t Size2, bool Approx2, RoundingMode Mode2,
        typename Derived2, typename T = Derived,
        std::enable_if_t<std::is_constructible<Value_, const Value2 &>::value &&
                        !T::IsMask && Derived2::Size == T::Size && !Derived2::IsRecursive, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(a, std::make_index_sequence<Size>()) { }

    /// Convert a compatible array type (const, recursive)
    template <
        typename Value2, size_t Size2, bool Approx2, RoundingMode Mode2,
        typename Derived2, typename T = Derived,
        std::enable_if_t<std::is_constructible<Value_, const Value2 &>::value &&
                        !T::IsMask && Derived2::Size == T::Size && Derived2::IsRecursive, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(Array1(low(a)), Array2(high(a))) { }

    /// Convert a compatible array type (non-const, useful when storing references)
    template <typename Value2, size_t Size2, bool Approx2, RoundingMode Mode2,
              typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_constructible<Value_, Value2 &>::value &&
                              !std::is_constructible<Value_, const Value2 &>::value &&
                              Derived2::Size == T::Size && !T::IsMask, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
              StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(a, std::make_index_sequence<Size>()) { }

    /// Reinterpret a compatible array (non-recursive)
    template <typename Value2, size_t Size2, bool Approx2, RoundingMode Mode2, typename Derived2,
              typename T = Derived, std::enable_if_t<Derived2::Size == T::Size && !Derived2::IsRecursive
                  && is_reinterpretable<Value_, Value2>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a,
        detail::reinterpret_flag) {
        for (size_t i = 0; i < Size; ++i)
            coeff(i) = reinterpret_array<Value>(a.derived().coeff(i));
    }

    /// Reinterpret another array (recursive)
    template <typename Value2, size_t Size2, bool Approx2, RoundingMode Mode2, typename Derived2,
              typename T = Derived, std::enable_if_t<Derived2::Size == T::Size && Derived2::IsRecursive, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a,
                                 detail::reinterpret_flag)
        : StaticArrayImpl(reinterpret_array<Array1>(low(a)),
                          reinterpret_array<Array2>(high(a))) { }


    /// Convert a compatible mask -- only used if this is an Array<bool, ...> or another kind of mask
    template <typename T, typename T2 = Derived, std::enable_if_t<T2::IsMask, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const T &value)
        : StaticArrayImpl(value, detail::reinterpret_flag()) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors that convert from incompatible representations
    // -----------------------------------------------------------------------

    /// Broadcast a scalar or packet type
    template <typename T, typename T2 = Derived, typename T3 = Value_,
              std::enable_if_t<broadcast<T>::value && !T2::IsMask && !T2::CustomBroadcast &&
                               std::is_default_constructible<T3>::value &&
                               std::is_constructible<T3, const T &>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const T &a) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = Value(a);
    }

    /// Broadcast a scalar (reinterpret cast)
    template <typename T, typename T2 = Value_,
              std::enable_if_t<broadcast<T>::value &&
                               std::is_default_constructible<T2>::value &&
                               is_reinterpretable<T2, T>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const T &a, detail::reinterpret_flag) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = reinterpret_array<Value>(a);
    }

    /// Broadcast an incompatible array type
    template <typename Value2, size_t Size2, bool Approx2, RoundingMode Mode2, typename Derived2,
              typename T = Derived, std::enable_if_t<Derived2::Size != T::Size &&
                                                     std::is_constructible<Value_, const Derived2 &>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = Value(a.derived());
    }

    /// Broadcast an incompatible array type (reinterpret cast)
    template <typename Value2, size_t Size2, bool Approx2, RoundingMode Mode2, typename Derived2,
              typename T = Derived, std::enable_if_t<Derived2::Size != T::Size &&
                                                     is_reinterpretable<Value_, Derived2>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<Value2, Size2, Approx2, Mode2, Derived2> &a, detail::reinterpret_flag) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = reinterpret_array<Value>(a.derived());
    }

    /// Construct an array of masked arrays from a masked array of arrays.. :)
    template <typename T> ENOKI_INLINE StaticArrayImpl(detail::MaskedArray<T> &value)
        : StaticArrayImpl(value, std::make_index_sequence<Size>()) { }

    template <typename T, size_t... Index>
    ENOKI_INLINE StaticArrayImpl(detail::MaskedArray<T> &value,
                                 std::index_sequence<Index...>)
        : m_data{ detail::MaskedArray<value_t<T>>(value.d.coeff(Index),
                                                  value.m.coeff(Index))... } { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    template <size_t S = Size2, std::enable_if_t<S != 0, int> = 0>
    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : StaticArrayImpl(a1, a2, std::make_index_sequence<Size1>(),
                                  std::make_index_sequence<Size2>()) { }

private:
    template <size_t... Index1, size_t... Index2>
    ENOKI_INLINE StaticArrayImpl(const Array1 &a1, const Array2 &a2,
                                 std::index_sequence<Index1...>,
                                 std::index_sequence<Index2...>)
        : m_data{ { (StorageType) a1.coeff(Index1)...,
                    (StorageType) a2.coeff(Index2)... } } { ENOKI_CHKSCALAR }

public:
    Array1 low_() const {
        Array1 result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size1; ++i)
            result.coeff(i) = coeff(i);
        return result;
    }

    Array2 high_() const {
        Array2 result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size2; ++i)
            result.coeff(i) = coeff(Size1 + i);
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name SSE/AVX/AVX2/AVX512/.. mask to Array<bool, ..> conversions
    // -----------------------------------------------------------------------

    #define ENOKI_REINTERPRET_BOOL(Count, ValueSize)                           \
        template <typename Value2, bool Approx2, RoundingMode Mode2,           \
                  typename Derived2, typename T = Derived_,                    \
                  std::enable_if_t<std::is_same<value_t<T>, bool>::value &&    \
                      Derived2::IsNative && Derived2::Size == T::Size &&       \
                      sizeof(Value2) == ValueSize, int> = 0>                   \
        ENOKI_INLINE StaticArrayImpl(                                          \
            const StaticArrayBase<Value2, Count, Approx2, Mode2, Derived2> &a, \
            detail::reinterpret_flag)

#if defined(ENOKI_X86_SSE42)
    ENOKI_REINTERPRET_BOOL(2, 8) {
        __m128i m = _mm_and_si128((__m128i &) a.derived().m, _mm_set1_epi8(1));
        uint16_t result = (uint16_t) _mm_cvtsi128_si32(_mm_shuffle_epi8(
            m, _mm_set1_epi32(0 + (8 << 8))));
        memcpy(data(), &result, T::Size);
    }

    ENOKI_REINTERPRET_BOOL(4, 4) {
        __m128i m = _mm_and_si128((__m128i &) a.derived().m, _mm_set1_epi8(1));
        uint32_t result = (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi8(
            m, _mm_set1_epi32((0 << 0) + (4 << 8) + (8 << 16) + (12 << 24))));
        memcpy(data(), &result, T::Size);
    }
#endif

#if defined(ENOKI_X86_AVX)
    ENOKI_REINTERPRET_BOOL(4, 8) {
        __m128i hi, lo;
        #if defined(ENOKI_X86_AVX2)
            __m256i m = _mm256_and_si256((__m256i &) a.derived().m, _mm256_set1_epi8(1));
            m = _mm256_shuffle_epi8(
                m, _mm256_set1_epi32((0 << 0) + (8 << 8) + (0 << 16) + (8 << 24)));
            hi = _mm256_castsi256_si128(m);
            lo = _mm256_extracti128_si256(m, 1);

        #else
            const __m128i shufmask = _mm_set1_epi32((0 << 0) + (8 << 8) + (0 << 16) + (8 << 24));
            __m256i m = _mm256_castps_si256(_mm256_and_ps((__m256 &) a.derived().m,
                                            _mm256_castsi256_ps(_mm256_set1_epi8(1))));
            hi = _mm256_castsi256_si128(m);
            lo = _mm256_extractf128_si256(m, 1);
            lo = _mm_shuffle_epi8(lo, shufmask);
            hi = _mm_shuffle_epi8(hi, shufmask);
        #endif
        uint32_t result = (uint32_t) _mm_cvtsi128_si32(_mm_unpacklo_epi16(hi, lo));
        memcpy(data(), &result, T::Size);
    }

    ENOKI_REINTERPRET_BOOL(8, 4) {
        __m128i hi, lo;
        #if defined(ENOKI_X86_AVX2)
            __m256i m = _mm256_and_si256((__m256i &) a.derived().m, _mm256_set1_epi8(1));
            m = _mm256_shuffle_epi8(
                m, _mm256_set1_epi32((0 << 0) + (4 << 8) + (8 << 16) + (12 << 24)));
            hi = _mm256_castsi256_si128(m);
            lo = _mm256_extracti128_si256(m, 1);

        #else
            const __m128i shufmask = _mm_set1_epi32((0 << 0) + (4 << 8) + (8 << 16) + (12 << 24));
            __m256i m = _mm256_castps_si256(_mm256_and_ps((__m256 &) a.derived().m,
                                            _mm256_castsi256_ps(_mm256_set1_epi8(1))));
            hi = _mm256_castsi256_si128(m);
            lo = _mm256_extractf128_si256(m, 1);
            lo = _mm_shuffle_epi8(lo, shufmask);
            hi = _mm_shuffle_epi8(hi, shufmask);
        #endif
        uint64_t result = (uint64_t) detail::mm_cvtsi128_si64(_mm_unpacklo_epi32(hi, lo));
        memcpy(data(), &result, T::Size);
    }
#endif

#if defined(ENOKI_X86_AVX512F)
    ENOKI_REINTERPRET_BOOL(16, 1) {
        #if defined(ENOKI_X86_AVX512BW) && defined(ENOKI_X86_AVX512VL)
            __m128i value = _mm_maskz_set1_epi8(a.derived().k, 1);
            _mm_storeu_si128((__m128i *) data(), value);
        #else
            uint64_t low = (uint64_t) _pdep_u64(a.derived().k,      0x0101010101010101ull);
            uint64_t hi  = (uint64_t) _pdep_u64(a.derived().k >> 8, 0x0101010101010101ull);
            memcpy(data(), &low, 8);
            memcpy(data() + 8, &hi, 8);
        #endif
    }

    ENOKI_REINTERPRET_BOOL(8, 1) {
        #if defined(ENOKI_X86_AVX512BW) && defined(ENOKI_X86_AVX512VL)
            __m128i value = _mm_maskz_set1_epi8(a.derived().k, 1);
            uint64_t result = (uint64_t) _mm_cvtsi128_si64(value);
        #else
            uint64_t result = (uint64_t) _pdep_u64(a.derived().k, 0x0101010101010101ull);
        #endif
        memcpy(data(), &result, T::Size);
    }

#endif

    #undef ENOKI_REINTERPRET_BOOL

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Half-precision conversions
    // -----------------------------------------------------------------------

    #define ENOKI_CONVERT_GENERIC(InputT, OutputT, Count)                      \
        template <bool Approx2, RoundingMode Mode2, typename Derived2,         \
                  typename T = Derived,                                        \
                  std::enable_if_t<std::is_same<value_t<T>, OutputT>::value && \
                                   Derived2::Size == T::Size, int> = 0>        \
        ENOKI_INLINE StaticArrayImpl(                                          \
            const StaticArrayBase<InputT, Count, Approx2, Mode2, Derived2> &a)

#if defined(ENOKI_X86_F16C)
    ENOKI_CONVERT_GENERIC(float, half, 4) {
        __m128i value = _mm_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION);
        memcpy(m_data.data(), &value, sizeof(uint16_t) * Derived::Size);
    }

#if defined(ENOKI_X86_AVX)
    ENOKI_CONVERT_GENERIC(double, half, 4) {
        __m128i value = _mm_cvtps_ph(_mm256_cvtpd_ps(a.derived().m), _MM_FROUND_CUR_DIRECTION);
        memcpy(m_data.data(), &value, sizeof(uint16_t) * Derived::Size);
    }

    ENOKI_CONVERT_GENERIC(float, half, 8) {
        _mm_storeu_si128((__m128i *) m_data.data(), _mm256_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION));
    }
#endif

#if defined(ENOKI_X86_AVX512F)
    ENOKI_CONVERT_GENERIC(double, half, 8) {
        _mm_storeu_si128((__m128i *) m_data.data(), _mm256_cvtps_ph(_mm512_cvtpd_ps(a.derived().m), _MM_FROUND_CUR_DIRECTION));
    }

    ENOKI_CONVERT_GENERIC(float, half, 16) {
        _mm256_storeu_si256((__m256i *) m_data.data(), _mm512_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION));
    }
#endif
#endif

#if defined(ENOKI_ARM_NEON)
    ENOKI_CONVERT_GENERIC(float, half, 4) {
        float16x4_t value = vcvt_f16_f32(a.derived().m);
        memcpy(m_data.data(), &value, sizeof(uint16_t) * Derived::Size);
    }
#endif

    #undef ENOKI_CONVERT_GENERIC

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Generic implementations of vertical operations
    // -----------------------------------------------------------------------

    /// Addition
    ENOKI_INLINE auto add_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            (Value &) result.coeff(i) = (const Value &) coeff(i) + (const Value &) d.coeff(i);
        return result;
    }

    /// Subtraction
    ENOKI_INLINE auto sub_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            (Value &) result.coeff(i) = (const Value &) coeff(i) - (const Value &) d.coeff(i);
        return result;
    }

    /// Multiplication
    ENOKI_INLINE auto mul_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) * d.coeff(i);
        return result;
    }

    /// Integer high multiplication
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto mulhi_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = mulhi(coeff(i), d.coeff(i));
        return result;
    }

    /// Division
    ENOKI_INLINE auto div_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) / d.coeff(i);
        return result;
    }

    /// Modulo
    ENOKI_INLINE auto mod_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) % d.coeff(i);
        return result;
    }

    /// Arithmetic unary NOT operation
    ENOKI_INLINE auto not_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.m_data[i] = detail::not_(coeff(i));
        return result;
    }

    /// Arithmetic OR operation
    template <typename Array>
    ENOKI_INLINE auto or_(const Array &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.m_data[i] = detail::or_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic AND operation
    template <typename Array>
    ENOKI_INLINE auto and_(const Array &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.m_data[i] = detail::and_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic ANDNOT operation
    template <typename Array>
    ENOKI_INLINE auto andnot_(const Array &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.m_data[i] = detail::andnot_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic XOR operation
    template <typename Array>
    ENOKI_INLINE auto xor_(const Array &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.m_data[i] = detail::xor_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic unary negation operation
    ENOKI_INLINE auto neg_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = -coeff(i);
        return result;
    }

    /// Left shift operator
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto sl_(size_t value) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) << value;
        return result;
    }

    /// Left shift operator
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto slv_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) << d.coeff(i);
        return result;
    }

    /// Right shift operator
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto sr_(size_t value) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) >> value;
        return result;
    }

    /// Right shift operator
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto srv_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) >> d.coeff(i);
        return result;
    }

    /// Left shift operator (immediate)
    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto sli_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = sli<Imm>(coeff(i));
        return result;
    }

    /// Right shift operator (immediate)
    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto sri_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = sri<Imm>(coeff(i));
        return result;
    }

    /// Equality comparison operation
    ENOKI_INLINE auto eq_(const Derived &d) const {
        mask_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = eq(Value(coeff(i)), Value(d.coeff(i)));
        return result;
    }

    /// Inequality comparison operation
    ENOKI_INLINE auto neq_(const Derived &d) const {
        mask_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = neq(Value(coeff(i)), Value(d.coeff(i)));
        return result;
    }

    /// Less than comparison operation
    ENOKI_INLINE auto lt_(const Derived &d) const {
        mask_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) < d.coeff(i);
        return result;
    }

    /// Less than or equal comparison operation
    ENOKI_INLINE auto le_(const Derived &d) const {
        mask_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) <= d.coeff(i);
        return result;
    }

    /// Greater than comparison operation
    ENOKI_INLINE auto gt_(const Derived &d) const {
        mask_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) > d.coeff(i);
        return result;
    }

    /// Greater than or equal comparison operation
    ENOKI_INLINE auto ge_(const Derived &d) const {
        mask_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) >= d.coeff(i);
        return result;
    }

    /// Absolute value
    ENOKI_INLINE auto abs_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = abs(coeff(i));
        return result;
    }

    /// Square root
    ENOKI_INLINE auto sqrt_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = sqrt(coeff(i));
        return result;
    }

    /// Round to smallest integral value not less than argument
    ENOKI_INLINE auto ceil_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = ceil(coeff(i));
        return result;
    }

    /// Round to largest integral value not greater than argument
    ENOKI_INLINE auto floor_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = floor(coeff(i));
        return result;
    }

    /// Round to integral value
    ENOKI_INLINE auto round_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = round(coeff(i));
        return result;
    }

    /// Element-wise maximum
    ENOKI_INLINE auto max_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = max(coeff(i), d.coeff(i));
        return result;
    }

    /// Element-wise minimum
    ENOKI_INLINE auto min_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = min(coeff(i), d.coeff(i));
        return result;
    }

    /// Fused multiply-add
    ENOKI_INLINE auto fmadd_(const Derived &d1, const Derived &d2) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = fmadd(coeff(i), d1.coeff(i), d2.coeff(i));
        return result;
    }

    /// Fused negative multiply-add
    ENOKI_INLINE auto fnmadd_(const Derived &d1, const Derived &d2) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = fnmadd(coeff(i), d1.coeff(i), d2.coeff(i));
        return result;
    }

    /// Fused multiply-subtract
    ENOKI_INLINE auto fmsub_(const Derived &d1, const Derived &d2) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = fmsub(coeff(i), d1.coeff(i), d2.coeff(i));
        return result;
    }

    /// Fused negative multiply-subtract
    ENOKI_INLINE auto fnmsub_(const Derived &d1, const Derived &d2) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = fnmsub(coeff(i), d1.coeff(i), d2.coeff(i));
        return result;
    }

    /// Square root of the reciprocal
    ENOKI_INLINE auto rsqrt_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = rsqrt<Approx_>(coeff(i));
        return result;
    }

    /// Reciprocal
    ENOKI_INLINE auto rcp_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = rcp<Approx_>(coeff(i));
        return result;
    }

    /// Exponential function
    ENOKI_INLINE auto exp_() const {
        if (std::is_arithmetic<Value>::value && !has_avx512er) {
            return Base::exp_();
        } else {
            expr_t<Derived> result;
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                result.coeff(i) = exp<Approx_>(coeff(i));
            return result;
        }
    }

    /// Ternary operator -- select between to values based on mask
    template <typename Mask>
    static ENOKI_INLINE auto select_(const Mask &m, const Derived &t, const Derived &f) {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = select(m.coeff(i), t.coeff(i), f.coeff(i));
        return result;
    }

    /// Population count
    ENOKI_INLINE auto popcnt_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = popcnt(coeff(i));
        return result;
    }

    /// Leading zero count
    ENOKI_INLINE auto lzcnt_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = lzcnt(coeff(i));
        return result;
    }

    /// Trailing zero count
    ENOKI_INLINE auto tzcnt_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = tzcnt(coeff(i));
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Generic implementations of horizontal operations
    // -----------------------------------------------------------------------

    /// Horizontal sum
    ENOKI_INLINE Value hsum_() const {
        Value result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result += coeff(i);
        return result;
    }

    /// Horizontal product
    ENOKI_INLINE Value hprod_() const {
        Value result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result *= coeff(i);
        return result;
    }

    /// Horizontal maximum
    ENOKI_INLINE Value hmax_() const {
        Value result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = max(result, coeff(i));
        return result;
    }

    /// Horizontal minimum
    ENOKI_INLINE Value hmin_() const {
        Value result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = min(result, coeff(i));
        return result;
    }

    /// Check if all mask bits are set
    ENOKI_INLINE Value all_() const {
        Value result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = result & coeff(i);
        return result;
    }

    /// Check if any mask bits are set
    ENOKI_INLINE Value any_() const {
        Value result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = result | coeff(i);
        return result;
    }

    /// Check if none of the mask bits are set
    ENOKI_INLINE Value none_() const {
        Value result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = result | coeff(i);
        return !result;
    }

    /// Count the number of active mask bits
    ENOKI_INLINE auto count_() const {
        using Int = value_t<size_array_t<array_t<Derived>>>;
        const Int one(1);
        Int result(0);
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            masked(result, coeff(i)) += one;
        return result;
    }

    /// Dot product
    ENOKI_INLINE Value dot_(const Derived &arg) const {
        Value result = coeff(0) * arg.coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = fmadd(coeff(i), arg.coeff(i), result);
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Element access
    // -----------------------------------------------------------------------

    ENOKI_INLINE Element &coeff(size_t i) { ENOKI_CHKSCALAR return m_data[i]; }
    ENOKI_INLINE const Element &coeff(size_t i) const { ENOKI_CHKSCALAR return m_data[i]; }

    /// Recursive array indexing operator (const)
    template <typename... Args, std::enable_if_t<(sizeof...(Args) >= 1), int> = 0>
    ENOKI_INLINE decltype(auto) coeff(size_t i0, Args... other) const {
        return coeff(i0).coeff(size_t(other)...);
    }

    /// Recursive array indexing operator
    template <typename... Args, std::enable_if_t<(sizeof...(Args) >= 1), int> = 0>
    ENOKI_INLINE decltype(auto) coeff(size_t i0, Args... other) {
        return coeff(i0).coeff(size_t(other)...);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Higher-level functions
    // -----------------------------------------------------------------------

    #define ENOKI_FORWARD_FUNCTION(name)                                     \
        auto name##_() const {                                               \
            expr_t<Derived> result;                                          \
            if (std::is_arithmetic<Value>::value) {                          \
                result = Base::name##_();                                    \
            } else {                                                         \
                ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)            \
                    result.coeff(i) = name(coeff(i));                        \
            }                                                                \
            return result;                                                   \
        }

    auto pow_(const Derived &arg) const {
        expr_t<Derived> result;
        if (std::is_arithmetic<Value>::value) {
            result = Base::pow_(arg);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                result.coeff(i) = pow(coeff(i), arg.coeff(i));
        }
        return result;
    }

    auto ldexp_(const Derived &arg) const {
        expr_t<Derived> result;
        if (std::is_arithmetic<Value>::value && !has_avx512f) {
            result = Base::ldexp_(arg);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                result.coeff(i) = ldexp(coeff(i), arg.coeff(i));
        }
        return result;
    }

    auto frexp_() const {
        std::pair<expr_t<Derived>, expr_t<Derived>> result;
        if (std::is_arithmetic<Value>::value && !has_avx512f) {
            result = Base::frexp_();
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                std::tie(result.first.coeff(i), result.second.coeff(i)) = frexp(coeff(i));
        }
        return result;
    }

    auto sincos_() const {
        std::pair<expr_t<Derived>, expr_t<Derived>> result;
        if (std::is_arithmetic<Value>::value) {
            result = Base::sincos_();
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                std::tie(result.first.coeff(i), result.second.coeff(i)) = sincos(coeff(i));
        }
        return result;
    }

    auto sincosh_() const {
        std::pair<expr_t<Derived>, expr_t<Derived>> result;
        if (std::is_arithmetic<Value>::value) {
            result = Base::sincosh_();
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                std::tie(result.first.coeff(i), result.second.coeff(i)) = sincosh(coeff(i));
        }
        return result;
    }

    ENOKI_FORWARD_FUNCTION(log)
    ENOKI_FORWARD_FUNCTION(sin)
    ENOKI_FORWARD_FUNCTION(sinh)
    ENOKI_FORWARD_FUNCTION(cos)
    ENOKI_FORWARD_FUNCTION(cosh)
    ENOKI_FORWARD_FUNCTION(tan)
    ENOKI_FORWARD_FUNCTION(tanh)
    ENOKI_FORWARD_FUNCTION(csc)
    ENOKI_FORWARD_FUNCTION(csch)
    ENOKI_FORWARD_FUNCTION(sec)
    ENOKI_FORWARD_FUNCTION(sech)
    ENOKI_FORWARD_FUNCTION(cot)
    ENOKI_FORWARD_FUNCTION(coth)
    ENOKI_FORWARD_FUNCTION(asin)
    ENOKI_FORWARD_FUNCTION(asinh)
    ENOKI_FORWARD_FUNCTION(acos)
    ENOKI_FORWARD_FUNCTION(acosh)
    ENOKI_FORWARD_FUNCTION(atan)
    ENOKI_FORWARD_FUNCTION(atanh)

    #undef ENOKI_FORWARD_FUNCTION

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    static ENOKI_INLINE auto zero_() { return expr_t<Derived>(Value(0)); }

    template <typename T = Derived,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_(const void *ptr) {
        Derived result;
        if (std::is_arithmetic<Value>::value) {
            memcpy(result.m_data.data(), ptr, sizeof(Value) * Size);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                result.m_data[i] = load<Value>(static_cast<const Value *>(ptr) + i);
        }
        return result;
    }

    template <typename T = Derived, typename Mask,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_(const void *ptr, const Mask &mask) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.m_data[i] = load<Value>(static_cast<const Value *>(ptr) + i, mask.coeff(i));
        return result;
    }

    template <typename T = Derived,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        Derived result;
        if (std::is_arithmetic<Value>::value) {
            memcpy(result.m_data.data(), ptr, sizeof(Value) * Size);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                result.m_data[i] = load_unaligned<Value>(static_cast<const Value *>(ptr) + i);
        }
        return result;
    }

    template <typename T = Derived, typename Mask,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, const Mask &mask) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.m_data[i] = load_unaligned<Value>(static_cast<const Value *>(ptr) + i, mask.coeff(i));
        return result;
    }

    void store_(void *ptr) const {
        if (std::is_arithmetic<Value>::value) {
            memcpy(ptr, m_data.data(), sizeof(Value) * Size);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                store<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i));
        }
    }

    template <typename Mask>
    void store_(void *ptr, const Mask &mask) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i), mask.coeff(i));
    }

    void store_unaligned_(void *ptr) const {
        if (std::is_arithmetic<Value>::value) {
            memcpy(ptr, m_data.data(), sizeof(Value) * Size);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                store_unaligned<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i));
        }
    }

    template <typename Mask>
    void store_unaligned_(void *ptr, const Mask &mask) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store_unaligned<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i), mask.coeff(i));
    }

    //! @}
    // -----------------------------------------------------------------------

private:
    std::array<StorageType, Size> m_data;
};

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_, typename SFINAE = void>
struct StaticMaskImpl : StaticArrayImpl<mask_t<Value_>, Size_, false, RoundingMode::Default, Derived_> {
    using Base = StaticArrayImpl<mask_t<Value_>, Size_, false, RoundingMode::Default, Derived_>;
    using Base::Base;
    using Base::operator=;
    using Base::data;

    StaticMaskImpl() = default;
};

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticMaskImpl<Value_, Size_, Approx_, Mode_, Derived_,
                      std::enable_if_t<Array<Value_, Size_, Approx_, Mode_>::IsNative &&
                                       std::is_arithmetic<Value_>::value &&
                                       sizeof(Value_) * Size_ * 8 != 512 /* AVX512 is handled specially */ >>
    : StaticArrayImpl<Value_, Size_, Approx_, Mode_, Derived_> {
    StaticMaskImpl() = default;
    using Base = StaticArrayImpl<Value_, Size_, Approx_, Mode_, Derived_>;
    using Base::Base;
    using Base::operator=;
};

/// Enumeration support
template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticArrayImpl<Value_, Size_, Approx_, Mode_, Derived_,
                       std::enable_if_t<std::is_enum<Value_>::value>>
    : StaticArrayImpl<std::underlying_type_t<Value_>, Size_, Approx_, Mode_, Derived_> {

    using UnderlyingType = std::underlying_type_t<Value_>;
    using Base = StaticArrayImpl<UnderlyingType, Size_, Approx_, Mode_, Derived_>;
    using Base::Base;
    using Base::operator=;

    using Value = Value_;
    using Scalar = Value_;

    StaticArrayImpl()  = default;
    StaticArrayImpl(Value value) : Base(UnderlyingType(value)) { }

    ENOKI_INLINE const Value& coeff(size_t i) const { return (Value &) Base::coeff(i); }
    ENOKI_INLINE Value& coeff(size_t i) { return (Value &) Base::coeff(i); }
};

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticMaskImpl<Value_, Size_, Approx_, Mode_, Derived_, std::enable_if_t<std::is_enum<Value_>::value>>
    : StaticMaskImpl<std::underlying_type_t<Value_>, Size_, Approx_, Mode_, Derived_> {
    StaticMaskImpl() = default;
    using Base = StaticMaskImpl<std::underlying_type_t<Value_>, Size_, Approx_, Mode_, Derived_>;
    using Base::Base;
    using Base::operator=;
};

/// Pointer support
template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticArrayImpl<Value_, Size_, Approx_, Mode_, Derived_,
    std::enable_if_t<std::is_pointer<Value_>::value && !std::is_arithmetic<std::remove_pointer_t<Value_>>::value>>
    : StaticArrayImpl<std::uintptr_t, Size_, Approx_, Mode_, Derived_> {

    using UnderlyingType = std::uintptr_t;
    using Base = StaticArrayImpl<UnderlyingType, Size_, Approx_, Mode_, Derived_>;

    using Base::Base;
    using Base::operator=;
    using Base::derived;
    using Value = Value_;
    using Scalar = Value_;

    StaticArrayImpl() = default;
    StaticArrayImpl(Value value) : Base(UnderlyingType(value)) { }

    ENOKI_INLINE const Value& coeff(size_t i) const { return (Value &) Base::coeff(i); }
    ENOKI_INLINE Value& coeff(size_t i) { return (Value &) Base::coeff(i); }

    call_support<Derived_, Derived_> operator->() const {
        return call_support<Derived_, Derived_>(derived());
    }
};

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticMaskImpl<Value_, Size_, Approx_, Mode_, Derived_,
    std::enable_if_t<std::is_pointer<Value_>::value && !std::is_arithmetic<std::remove_pointer_t<Value_>>::value>>
    : StaticMaskImpl<std::uintptr_t, Size_, Approx_, Mode_, Derived_> {
    StaticMaskImpl() = default;
    using Base = StaticMaskImpl<std::uintptr_t, Size_, Approx_, Mode_, Derived_>;
    using Base::Base;
    using Base::operator=;
};

template <typename PacketType_, typename Storage_> struct call_support_base {
    using PacketType = PacketType_;
    using Storage = Storage_;
    using Instance = value_t<PacketType>;
    using Mask = mask_t<PacketType>;
    call_support_base(const Storage &self) : self(self) { }
    const Storage &self;

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Instance>(), std::declval<Args>()...,
                  std::declval<Mask>())),
              std::enable_if_t<!std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE Result dispatch_2(Func&& func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);
        Result result = zero<Result>();

        while (any(mask)) {
            Instance value         = extract(self, mask);
            Mask active            = mask & eq(self, PacketType(value));
            mask                   = andnot(mask, active);
            masked(result, active) = func(value, args..., active);
        }

        return result;
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Instance>(), std::declval<Args>()...,
                  std::declval<Mask>())),
              std::enable_if_t<std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE void dispatch_2(Func &&func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);

        while (any(mask)) {
            Instance value = extract(self, mask);
            Mask active    = mask & eq(self, PacketType(value));
            mask           = andnot(mask, active);
            func(value, args..., active);
        }
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Instance>(), std::declval<Args>()...)),
              typename ResultArray = array_t<like_t<Mask, Result>>,
              std::enable_if_t<!std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE ResultArray dispatch_scalar_2(Func &&func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);
        ResultArray result = zero<Result>();

        while (any(mask)) {
            Instance value          = extract(self, mask);
            Mask active             = mask & eq(self, PacketType(value));
            mask                    = andnot(mask, active);
            masked(result, active)  = func(value, args...);
        }

        return result;
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Instance>(), std::declval<Args>()...)),
              std::enable_if_t<std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE void dispatch_scalar_2(Func &&func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);

        while (any(mask)) {
            Instance value  = extract(self, mask);
            mask           &= neq(self, PacketType(value));
            func(value, args...);
        }
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        (void) args; /* Don't warn if the function takes no arguments */
        return dispatch_2(func, std::get<(Indices + sizeof...(Indices) - 1) %
                                  sizeof...(Indices)>(args)...);
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<!is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        (void) args; /* Don't warn if the function takes no arguments */
        return dispatch_2(func, true, std::get<Indices>(args)...);
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_scalar_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        (void) args; /* Don't warn if the function takes no arguments */
        return dispatch_scalar_2(func, std::get<(Indices + sizeof...(Indices) - 1) %
                                 sizeof...(Indices)>(args)...);
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<!is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_scalar_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        (void) args; /* Don't warn if the function takes no arguments */
        return dispatch_scalar_2(func, true, std::get<Indices>(args)...);
    }
};

#define ENOKI_CALL_SUPPORT_BEGIN(PacketType)                                   \
    namespace enoki {                                                          \
    template <typename Storage>                                                \
    struct call_support<PacketType, Storage>                                   \
        : call_support_base<PacketType, Storage> {                             \
        using Base = call_support_base<PacketType, Storage>;                   \
        using Base::Base;                                                      \
        using Base::self;                                                      \
        using Instance = std::remove_pointer_t<typename Base::Instance>;       \
        auto operator-> () { return this; }

#define ENOKI_CALL_SUPPORT_GENERIC(name, dispatch)                             \
    template <typename... Args, typename T = Storage,                          \
              enable_if_static_array_t<T> = 0>                                 \
    ENOKI_INLINE decltype(auto) name(Args&&... args) {                         \
        return Base::dispatch(                                                 \
            [](Instance *s, auto&&... args2) { return s->name(args2...); },    \
            std::forward_as_tuple(args...),                                    \
            std::make_index_sequence<sizeof...(Args)>()                        \
        );                                                                     \
    }                                                                          \
    template <typename... Args, typename T = Storage,                          \
              enable_if_dynamic_array_t<T> = 0>                                \
    ENOKI_INLINE decltype(auto) name(Args&&... args) {                         \
        return vectorize([](auto self_p, auto&&... args_p) {                   \
                             return self_p->name(args_p...);                   \
                         }, self, args...);                                    \
    }

#define ENOKI_CALL_SUPPORT(name)                                               \
    ENOKI_CALL_SUPPORT_GENERIC(name, dispatch_1)
#define ENOKI_CALL_SUPPORT_SCALAR(name)                                        \
    ENOKI_CALL_SUPPORT_GENERIC(name, dispatch_scalar_1)

#define ENOKI_CALL_SUPPORT_END(PacketType)                                     \
        };                                                                     \
    }

//! @}
// -----------------------------------------------------------------------

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct Array : StaticArrayImpl<Value_, Size_, Approx_, Mode_,
                               Array<Value_, Size_, Approx_, Mode_>> {

    using Base = StaticArrayImpl<Value_, Size_, Approx_, Mode_,
                                 Array<Value_, Size_, Approx_, Mode_>>;
    using MaskType = Mask<Value_, Size_, Approx_, Mode_>;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T>
    using ReplaceType = Array<T, Size_,
          detail::is_std_float<scalar_t<T>>::value ? Approx_ : detail::approx_default<T>::value,
          detail::is_std_float<scalar_t<T>>::value ? Mode_ : RoundingMode::Default>;

    ENOKI_DECLARE_ARRAY(Base, Array)
};

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct Packet : StaticArrayImpl<Value_, Size_, Approx_, Mode_,
                               Packet<Value_, Size_, Approx_, Mode_>> {

    using Base = StaticArrayImpl<Value_, Size_, Approx_, Mode_,
                                 Packet<Value_, Size_, Approx_, Mode_>>;
    using MaskType = PacketMask<Value_, Size_, Approx_, Mode_>;
    static constexpr bool PrefersBroadcast = true;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T>
    using ReplaceType = Packet<T, Size_,
          detail::is_std_float<scalar_t<T>>::value ? Approx_ : detail::approx_default<T>::value,
          detail::is_std_float<scalar_t<T>>::value ? Mode_ : RoundingMode::Default>;

    ENOKI_DECLARE_ARRAY(Base, Packet)
};

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct Mask : StaticMaskImpl<Value_, Size_, Approx_, Mode_, Mask<Value_, Size_, Approx_, Mode_>> {
    using Base = StaticMaskImpl<Value_, Size_, Approx_, Mode_, Mask<Value_, Size_, Approx_, Mode_>>;

    using ArrayType = Array<Value_, Size_, Approx_, Mode_>;
    using MaskType = Mask;
    using typename Base::Scalar;
    static constexpr bool BaseIsMask = Base::IsMask;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceType = Mask<T, Size_>;

    /// Turn mask casts into reinterpreting casts
    template <typename T
#if !defined(__clang__)
    /* Clang handles resolution order of imported constructors
       differently and doesn't require this extra check to avoid
       ambiguity errors */
    , typename T2 = Mask, std::enable_if_t<!T2::BaseIsMask, int> = 0
#endif
    >
    ENOKI_INLINE Mask(const T &value)
        : Mask(value, detail::reinterpret_flag()) { }

    ENOKI_INLINE Mask(const int &value)
        : Mask((bool) value, detail::reinterpret_flag()) { }

    /// Initialize mask with a uniform constant
    template <typename T, typename T2 = Mask,
              std::enable_if_t<std::is_arithmetic<T>::value
#if !defined(__clang__)
        && !T2::BaseIsMask
#endif
    , int> = 0>
    ENOKI_INLINE Mask(const T &value, detail::reinterpret_flag, int = 0)
        : Base(reinterpret_array<Scalar>(value)) { }

    auto eq_(const Mask &other) const {
        return ~(other ^ *this);
    }

    auto neq_(const Mask &other) const {
        return other ^ *this;
    }

    ENOKI_DECLARE_ARRAY(Base, Mask)
};

template <typename Value_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct PacketMask : StaticMaskImpl<Value_, Size_, Approx_, Mode_, PacketMask<Value_, Size_, Approx_, Mode_>> {
    using Base = StaticMaskImpl<Value_, Size_, Approx_, Mode_, PacketMask<Value_, Size_, Approx_, Mode_>>;
    using ArrayType = Array<Value_, Size_, Approx_, Mode_>;
    using MaskType = PacketMask;
    using typename Base::Scalar;
    static constexpr bool IsMask = true;
    static constexpr bool BaseIsMask = Base::IsMask;
    static constexpr bool PrefersBroadcast = true;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceType = PacketMask<T, Size_>;

    /// Turn mask casts into reinterpreting casts
    template <typename T
#if !defined(__clang__)
    /* Clang handles resolution order of imported constructors
       differently and doesn't require this extra check to avoid
       ambiguity errors */
    , typename T2 = PacketMask, std::enable_if_t<!T2::BaseIsMask, int> = 0
#endif
    >
    ENOKI_INLINE PacketMask(const T &value)
        : PacketMask(value, detail::reinterpret_flag()) { }

    ENOKI_INLINE PacketMask(const int &value)
        : PacketMask((bool) value, detail::reinterpret_flag()) { }

    /// Initialize mask with a uniform constant
    template <typename T, typename T2 = PacketMask,
              std::enable_if_t<std::is_arithmetic<T>::value
#if !defined(__clang__)
        && !T2::BaseIsMask
#endif
    , int> = 0>
    ENOKI_INLINE PacketMask(const T &value, detail::reinterpret_flag, int = 0)
        : Base(reinterpret_array<Scalar>(value)) { }

    auto eq_(const PacketMask &other) const {
        return ~(other ^ *this);
    }

    auto neq_(const PacketMask &other) const {
        return other ^ *this;
    }

    ENOKI_DECLARE_ARRAY(Base, PacketMask)
};

NAMESPACE_END(enoki)
