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

#define ENOKI_DECLARE_CUSTOM_ARRAY(Base, Array)                                \
    using Base::Base;                                                          \
    using Base::operator=;                                                     \
    Array() = default;                                                         \
    Array(const Array &) = default;                                            \
    Array(Array &&) = default;                                                 \
    Array &operator=(const Array &) = default;                                 \
    Array &operator=(Array &&) = default;

template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct ArrayMask : StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                                   ArrayMask<Type_, Size_, Approx_, Mode_>> {
    using Base = StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                                 ArrayMask<Type_, Size_, Approx_, Mode_>>;
    static constexpr bool IsMask = true;
    using typename Base::Value;
    using typename Base::Scalar;

    /// Turn mask casts into reinterpreting casts
    template <typename T, typename T2 = Type_,
              std::enable_if_t<is_mask<T>::value
#if defined(_MSC_VER)
              /* Visual studio handles resolution order of imported constructors differently and
                 requires this extra check to avoid ambiguous constructor errors */
              && !std::is_same<T2, bool>::value
#endif
        , int> = 0> /* only enable constructor if not already provided by base class */
    ENOKI_INLINE ArrayMask(const T &value)
        : ArrayMask(value, detail::reinterpret_flag()) { }

    /// Initialize mask with a boolean scalar
    ENOKI_INLINE ArrayMask(bool value_, detail::reinterpret_flag)
        : Base(reinterpret_array<Scalar>(value_)) { }

    ENOKI_DECLARE_CUSTOM_ARRAY(Base, ArrayMask)
    ENOKI_ALIGNED_OPERATOR_NEW()
};

NAMESPACE_END(detail)

/**
 * \brief Generic SIMD packed number data type
 *
 * This class provides an abstration for a packed SIMD number array. For
 * specific arguments, the class will switch to an implementation based on
 * SSE4.2, AVX/AVX2, or AVX512 intrinsics. The generic implementation in this
 * partial template specialization just uses plain C++ but generally also
 * vectorizes quite well on current compilers.
 */
template <typename, size_t, bool, RoundingMode, typename, typename>
struct StaticArrayImpl { /* Will never be instantiated */ };

template <typename Type_, size_t Size_, bool Approx_, typename Derived_>
struct StaticArrayImpl<
    Type_, Size_, Approx_, RoundingMode::Default, Derived_,
    std::enable_if_t<!detail::is_native<Type_, Size_>::value &&
                     !std::is_enum<Type_>::value &&
                     !(std::is_pointer<Type_>::value && !std::is_arithmetic<std::remove_pointer_t<Type_>>::value) &&
                     !detail::is_recursive<Type_, Size_, RoundingMode::Default>::value>>
    : StaticArrayBase<Type_, Size_, Approx_, RoundingMode::Default, Derived_> {

    using Base =
        StaticArrayBase<Type_, Size_, Approx_, RoundingMode::Default, Derived_>;

    using Base::operator=;
    using typename Base::Value;
    using typename Base::Derived;
    using typename Base::Scalar;
    using typename Base::Array1;
    using typename Base::Array2;
    using Base::Size;
    using Base::Size1;
    using Base::Size2;
    using Base::data;
    using Mask = detail::ArrayMask<mask_t<Value>, Size, false, RoundingMode::Default>;

    using StorageType =
        std::conditional_t<std::is_reference<Type_>::value,
                           std::reference_wrapper<Value>, Value>;

    // -----------------------------------------------------------------------
    //! @{ \name Default constructors and assignment operators
    // -----------------------------------------------------------------------

    ENOKI_TRIVIAL_CONSTRUCTOR(Type_)

    /// Default copy constructor
    StaticArrayImpl(const StaticArrayImpl &t) = default;

    /// Default move constructor
    StaticArrayImpl(StaticArrayImpl &&t) = default;

    /// Default copy assignment operator
    StaticArrayImpl& operator=(const StaticArrayImpl &) = default;

    /// Default move assignment operator
    StaticArrayImpl& operator=(StaticArrayImpl &&) = default;

    /// Reinterpret an array (without any changes)
    StaticArrayImpl(const StaticArrayImpl &a, detail::reinterpret_flag)
        : StaticArrayImpl(a) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Additional constructors and assignment operators
    // -----------------------------------------------------------------------

    /// Initialize all components from a scalar
    template <typename T = Type_, std::enable_if_t<!std::is_reference<T>::value, int> = 0>
    #if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && __GNUC__ < 7
        /// Work around a bug in GCC: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72824
        __attribute__((optimize("no-tree-loop-distribute-patterns")))
    #endif
    ENOKI_INLINE StaticArrayImpl(const Value &value) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            m_data[i] = value;
    }

    ENOKI_INLINE StaticArrayImpl& operator=(const Value &value) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            m_data[i] = value;
        return *this;
    }

    /// Initialize all components from a scalar
    template <typename T = Type_, std::enable_if_t<!std::is_reference<T>::value &&
                                                   !std::is_same<Value, Scalar>::value, int> = 0>
    #if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && __GNUC__ < 7
        /// Work around a bug in GCC: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72824
        __attribute__((optimize("no-tree-loop-distribute-patterns")))
    #endif
    ENOKI_INLINE StaticArrayImpl(const Scalar &value) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            m_data[i] = Value(value);
    }

    /// Initialize the individual components
    template <typename Arg, typename... Args,
              /* Ugly, works around a compiler ICE in MSVC */
              std::enable_if_t<
                  detail::all_of<std::is_constructible<StorageType, Arg>::value,
                                 std::is_constructible<StorageType, Args>::value...,
                                 sizeof...(Args) + 1 == Size_ && (sizeof...(Args) > 0)>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(Arg&& arg, Args&&... args)
        : m_data{{ (StorageType) std::forward<Arg>(arg),
                   (StorageType) std::forward<Args>(args)... }} { ENOKI_CHKSCALAR }

private:
    template <typename T, size_t... Index>
    ENOKI_INLINE StaticArrayImpl(T &&t, std::index_sequence<Index...>)
        : m_data{{ (StorageType) t.derived().coeff(Index)... }} { }

public:
    /// Convert a compatible mask
    template <typename T, typename T2 = Type_,
              std::enable_if_t<is_mask<T>::value &&
                               std::is_same<bool, T2>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const T &value)
        : StaticArrayImpl(value, detail::reinterpret_flag()) { }

    /// Convert a compatible array type (const, non-recursive)
    template <
        typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
        typename Derived2, typename T = Derived,
        std::enable_if_t<std::is_constructible<Type_, const Type2 &>::value &&
                        !T::IsMask && Derived2::Size == Size_ && !Derived2::IsRecursive, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(a, std::make_index_sequence<Size>()) { }

    /// Convert a compatible array type (const, recursive)
    template <
        typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
        typename Derived2, typename T = Derived,
        std::enable_if_t<std::is_constructible<Type_, const Type2 &>::value &&
                        !T::IsMask && Derived2::Size == Size_ && Derived2::IsRecursive, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(Array1(low(a)), Array2(high(a))) { }

    /// Convert a compatible array type (non-const, useful when storing references)
    template <typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
              typename Derived2,
              std::enable_if_t<std::is_constructible<Type_, Type2 &>::value &&
                              !std::is_constructible<Type_, const Type2 &>::value &&
                              Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
              StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(a, std::make_index_sequence<Size>()) { }

    /// Reinterpret another array (non-recursive)
    template <typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
              typename Derived2, typename T2 = Derived,
              std::enable_if_t<Derived2::Size == Size_ && !Derived2::IsRecursive, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a,
        detail::reinterpret_flag) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = reinterpret_array<Value>(a.derived().coeff(i));
    }

    /// Reinterpret another array (recursive)
    template <typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
              typename Derived2, typename T2 = Derived,
              std::enable_if_t<Derived2::Size == Size_ && Derived2::IsRecursive, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a,
        detail::reinterpret_flag)
        : StaticArrayImpl(reinterpret_array<Array1>(low(a)),
                          reinterpret_array<Array2>(high(a))) { }

    /// Reinterpret another array (size mismatch)
    template <typename T, std::enable_if_t<array_size<T>::value != Size_ &&
                               std::is_constructible<Type_, T>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const T &a, detail::reinterpret_flag) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = reinterpret_array<Value>(a);
    }

    template <typename T> ENOKI_INLINE StaticArrayImpl(detail::MaskedArray<T> &value)
        : StaticArrayImpl(value, std::make_index_sequence<Size>()) { }

    template <typename T, size_t... Index>
    ENOKI_INLINE StaticArrayImpl(detail::MaskedArray<T> &value,
                                 std::index_sequence<Index...>)
        : m_data{ detail::MaskedArray<value_t<T>>(value.d.coeff(Index),
                                                  value.m.coeff(Index))... } { }

    //! @}
    // -----------------------------------------------------------------------

    #define ENOKI_CONVERT_GENERIC(InputT, OutputT, Count)                     \
        template <bool Approx2, RoundingMode Mode2, typename Derived2,        \
                  typename T = Derived,                                       \
                  std::enable_if_t<std::is_same<value_t<T>, OutputT>::value && \
                                   Derived2::Size == T::Size, int> = 0>       \
        ENOKI_INLINE StaticArrayImpl(                                         \
            const StaticArrayBase<InputT, Count, Approx2, Mode2, Derived2> &a)

    #define ENOKI_REINTERPRET_MASK(Count, TypeSize)                           \
        template <typename Type2, bool Approx2, RoundingMode Mode2,           \
                  typename Derived2, typename T = Derived,                    \
                  std::enable_if_t<T::IsMask && Derived2::IsNative &&         \
                                   Derived2::Size == T::Size &&               \
                                   sizeof(Type2) == TypeSize, int> = 0>       \
        ENOKI_INLINE StaticArrayImpl(                                         \
            const StaticArrayBase<Type2, Count, Approx2, Mode2, Derived2> &a, \
            detail::reinterpret_flag)

#if defined(__F16C__)
    // -----------------------------------------------------------------------
    //! @{ \name Half-precision conversions
    // -----------------------------------------------------------------------

    ENOKI_CONVERT_GENERIC(float, half, 4) {
        __m128i value = _mm_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION);
        memcpy(m_data.data(), &value, sizeof(uint16_t) * Derived::Size);
    }


#if defined(__AVX__)
    ENOKI_CONVERT_GENERIC(double, half, 4) {
        __m128i value = _mm_cvtps_ph(_mm256_cvtpd_ps(a.derived().m), _MM_FROUND_CUR_DIRECTION);
        memcpy(m_data.data(), &value, sizeof(uint16_t) * Derived::Size);
    }

    ENOKI_CONVERT_GENERIC(float, half, 8) {
        _mm_storeu_si128((__m128i *) m_data.data(), _mm256_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION));
    }
#endif

#if defined(__AVX512F__)
    ENOKI_CONVERT_GENERIC(double, half, 8) {
        _mm_storeu_si128((__m128i *) m_data.data(), _mm256_cvtps_ph(_mm512_cvtpd_ps(a.derived().m), _MM_FROUND_CUR_DIRECTION));
    }

    ENOKI_CONVERT_GENERIC(float, half, 16) {
        _mm256_storeu_si256((__m256i *) m_data.data(), _mm512_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION));
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
#endif

    // -----------------------------------------------------------------------
    //! @{ \name Mask conversions
    // -----------------------------------------------------------------------

#if defined(__SSE4_2__)
    ENOKI_REINTERPRET_MASK(2, 8) {
        __m128i m = _mm_and_si128((__m128i &) a.derived().m, _mm_set1_epi8(1));
        uint16_t result = (uint16_t) _mm_cvtsi128_si32(_mm_shuffle_epi8(
            m, _mm_set1_epi32(0 + (8 << 8))));
        memcpy(m_data.data(), &result, T::Size);
    }

    ENOKI_REINTERPRET_MASK(4, 4) {
        __m128i m = _mm_and_si128((__m128i &) a.derived().m, _mm_set1_epi8(1));
        uint32_t result = (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi8(
            m, _mm_set1_epi32((0 << 0) + (4 << 8) + (8 << 16) + (12 << 24))));
        memcpy(m_data.data(), &result, T::Size);
    }
#endif

#if defined(__AVX__)
    ENOKI_REINTERPRET_MASK(4, 8) {
        __m128i hi, lo;
        #if defined(__AVX2__)
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
        memcpy(m_data.data(), &result, T::Size);
    }

    ENOKI_REINTERPRET_MASK(8, 4) {
        __m128i hi, lo;
        #if defined(__AVX2__)
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
        memcpy(m_data.data(), &result, T::Size);
    }
#endif

#if defined(__AVX512F__)
    ENOKI_REINTERPRET_MASK(16, 1) {
        #if defined(__AVX512BW__) && defined(__AVX512VL__)
            __m128i value = _mm_maskz_set1_epi8(a.derived().k, 1);
            _mm_storeu_si128((__m128i *) m_data.data(), value);
        #else
            uint64_t low = (uint64_t) _pdep_u64(a.derived().k,      0x0101010101010101ull);
            uint64_t hi  = (uint64_t) _pdep_u64(a.derived().k >> 8, 0x0101010101010101ull);
            memcpy(m_data.data(), &low, 8);
            memcpy(m_data.data() + 8, &hi, 8);
        #endif
    }

    ENOKI_REINTERPRET_MASK(8, 1) {
        #if defined(__AVX512BW__) && defined(__AVX512VL__)
            __m128i value = _mm_maskz_set1_epi8(a.derived().k, 1);
            uint64_t result = (uint64_t) _mm_cvtsi128_si64(value);
        #else
            uint64_t result = (uint64_t) _pdep_u64(a.derived().k, 0x0101010101010101ull);
        #endif
        memcpy(m_data.data(), &result, T::Size);
    }

#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Generic implementations of vertical operations
    // -----------------------------------------------------------------------

    /// Addition
    ENOKI_INLINE auto add_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) + d.coeff(i);
        return result;
    }

    /// Subtraction
    ENOKI_INLINE auto sub_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) - d.coeff(i);
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
    ENOKI_INLINE Mask eq_(const Derived &d) const {
        Mask result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = eq(Value(coeff(i)), Value(d.coeff(i)));
        return result;
    }

    /// Inequality comparison operation
    ENOKI_INLINE Mask neq_(const Derived &d) const {
        Mask result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = neq(Value(coeff(i)), Value(d.coeff(i)));
        return result;
    }

    /// Less than comparison operation
    ENOKI_INLINE Mask lt_(const Derived &d) const {
        Mask result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) < d.coeff(i);
        return result;
    }

    /// Less than or equal comparison operation
    ENOKI_INLINE Mask le_(const Derived &d) const {
        Mask result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) <= d.coeff(i);
        return result;
    }

    /// Greater than comparison operation
    ENOKI_INLINE Mask gt_(const Derived &d) const {
        Mask result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) > d.coeff(i);
        return result;
    }

    /// Greater than or equal comparison operation
    ENOKI_INLINE Mask ge_(const Derived &d) const {
        Mask result;
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
    ENOKI_INLINE size_array_t<Value> count_() const {
        using Int = size_array_t<Value>;
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
            result += coeff(i) * arg.coeff(i);
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Element access
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value &coeff(size_t i) { ENOKI_CHKSCALAR return m_data[i]; }
    ENOKI_INLINE const Value &coeff(size_t i) const { ENOKI_CHKSCALAR return m_data[i]; }

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
        if (std::is_arithmetic<Value>::value) {
            result = Base::ldexp_(arg);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
                result.coeff(i) = ldexp(coeff(i), arg.coeff(i));
        }
        return result;
    }

    auto frexp_() const {
        std::pair<expr_t<Derived>, expr_t<Derived>> result;
        if (std::is_arithmetic<Value>::value) {
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

    template <typename T = Derived,
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

    template <typename T = Derived,
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

    void store_unaligned_(void *ptr, const Mask &mask) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store_unaligned<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i), mask.coeff(i));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : StaticArrayImpl(a1, a2, std::make_index_sequence<Size1>(),
                          std::make_index_sequence<Size2>()) { }

private:
    template <size_t... Index1, size_t... Index2>
    ENOKI_INLINE StaticArrayImpl(const Array1 &a1, const Array2 &a2,
                                 std::index_sequence<Index1...>,
                                 std::index_sequence<Index2...>)
        : m_data{ { StorageType(a1.coeff(Index1))...,
                    StorageType(a2.coeff(Index2))... } } { ENOKI_CHKSCALAR }

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

private:
    std::array<StorageType, Size> m_data;
};

/// Enumeration support
template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticArrayImpl<Type_, Size_, Approx_, Mode_, Derived_,
    std::enable_if_t<std::is_enum<Type_>::value>>
    : StaticArrayImpl<std::underlying_type_t<Type_>, Size_, Approx_, Mode_, Derived_> {

    using UnderlyingType = std::underlying_type_t<Type_>;
    using Base = StaticArrayImpl<UnderlyingType, Size_, Approx_, Mode_, Derived_>;

    using Base::Base;
    using Base::operator=;
    using Type = Type_;
    using Value = Type_;
    using Scalar = Type_;
    using Base::operator[];

    StaticArrayImpl() : Base() { }
    StaticArrayImpl(Type value) : Base(UnderlyingType(value)) { }

    ENOKI_INLINE const Type& coeff(size_t i) const { return (Type &) Base::coeff(i); }
    ENOKI_INLINE Type& coeff(size_t i) { return (Type &) Base::coeff(i); }
    ENOKI_INLINE const Type& operator[](size_t i) const { return (Type &) Base::operator[](i); }
    ENOKI_INLINE Type& operator[](size_t i) { return (Type &) Base::operator[](i); }
};

template <typename Packet, typename Storage> struct call_support {
    call_support(const Storage &) { }
};

/// Pointer support
template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticArrayImpl<Type_, Size_, Approx_, Mode_, Derived_,
    std::enable_if_t<std::is_pointer<Type_>::value && !std::is_arithmetic<std::remove_pointer_t<Type_>>::value>>
    : StaticArrayImpl<std::uintptr_t, Size_, Approx_, Mode_, Derived_> {

    using UnderlyingType = std::uintptr_t;
    using Base = StaticArrayImpl<UnderlyingType, Size_, Approx_, Mode_, Derived_>;

    using Base::Base;
    using Base::operator=;
    using Base::operator[];
    using Base::derived;
    using Type = Type_;
    using Value = Type_;
    using Scalar = Type_;

    StaticArrayImpl() : Base() { }
    StaticArrayImpl(Type value) : Base(UnderlyingType(value)) { }

    /// Initialize the individual components
    template <typename Arg, typename... Args,
              /* Ugly, works around a compiler ICE in MSVC */
              std::enable_if_t<
                  detail::all_of<std::is_constructible<Type, Arg>::value,
                                 std::is_constructible<Type, Args>::value...,
                                 sizeof...(Args) + 1 == Size_ && (sizeof...(Args) > 0)>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(Arg&& arg, Args&&... args)
        : Base{ UnderlyingType(std::forward<Arg>(arg)),
                UnderlyingType(std::forward<Args>(args))... } { ENOKI_CHKSCALAR }

    ENOKI_INLINE const Type& coeff(size_t i) const { return (Type &) Base::coeff(i); }
    ENOKI_INLINE Type& coeff(size_t i) { return (Type &) Base::coeff(i); }
    ENOKI_INLINE const Type& operator[](size_t i) const { return (Type &) Base::operator[](i); }
    ENOKI_INLINE Type& operator[](size_t i) { return (Type &) Base::operator[](i); }

    call_support<Derived_, Derived_> operator->() const {
        return call_support<Derived_, Derived_>(derived());
    }
};

template <typename Packet_, typename Storage_> struct call_support_base {
    using Packet = Packet_;
    using Storage = Storage_;
    using Value = value_t<Packet>;
    using Mask = mask_t<Packet>;
    call_support_base(const Storage &self) : self(self) { }
    const Storage &self;

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Value>(), std::declval<Args>()...,
                  std::declval<Mask>())),
              std::enable_if_t<!std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE Result dispatch_2(Func&& func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);
        Result result = zero<Result>();

        while (any(mask)) {
            Value value            = extract(self, mask);
            Mask active            = mask & eq(self, Packet(value));
            mask                  &= ~active;
            masked(result, active) = func(value, args..., active);
        }

        return result;
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Value>(), std::declval<Args>()...,
                  std::declval<Mask>())),
              std::enable_if_t<std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE void dispatch_2(Func &&func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);

        while (any(mask)) {
            Value value    = extract(self, mask);
            Mask active    = mask & eq(self, Packet(value));
            mask          &= ~active;
            func(value, args..., active);
        }
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Value>(), std::declval<Args>()...)),
              typename ResultArray = like_t<Mask, Result>,
              std::enable_if_t<!std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE ResultArray dispatch_scalar_2(Func &&func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);
        ResultArray result = zero<Result>();

        while (any(mask)) {
            Value value             = extract(self, mask);
            Mask active             = mask & eq(self, Packet(value));
            mask                   &= ~active;
            masked(result, active)  = func(value, args...);
        }

        return result;
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype(std::declval<Func>()(
                  std::declval<Value>(), std::declval<Args>()...)),
              std::enable_if_t<std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE void dispatch_scalar_2(Func &&func, InputMask&& mask_, Args&&... args) {
        Mask mask(mask_);

        while (any(mask)) {
            Value value    = extract(self, mask);
            mask          &= neq(self, Packet(value));
            func(value, args...);
        }
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        return dispatch_2(func, std::get<(Indices + sizeof...(Indices) - 1) %
                                  sizeof...(Indices)>(args)...);
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<!is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        return dispatch_2(func, true, std::get<Indices>(args)...);
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_scalar_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        return dispatch_scalar_2(func, std::get<(Indices + sizeof...(Indices) - 1) %
                                 sizeof...(Indices)>(args)...);
    }

    template <typename Func, typename Tuple, size_t... Indices,
              std::enable_if_t<!is_mask<detail::tuple_tail_t<Tuple>>::value, int> = 0>
    decltype(auto) dispatch_scalar_1(Func &&func, Tuple &&args, std::index_sequence<Indices...>) {
        return dispatch_scalar_2(func, true, std::get<Indices>(args)...);
    }
};

#define ENOKI_CALL_SUPPORT_BEGIN(Packet)                                       \
    namespace enoki {                                                          \
    template <typename Storage>                                                \
    struct call_support<Packet, Storage>                                       \
        : call_support_base<Packet, Storage> {                                 \
        using Base = call_support_base<Packet, Storage>;                       \
        using Base::Base;                                                      \
        using Base::self;                                                      \
        using Type = std::remove_pointer_t<typename Base::Value>;              \
        auto operator-> () { return this; }

#define ENOKI_CALL_SUPPORT_GENERIC(name, dispatch)                             \
    template <typename... Args, typename T = Storage,                          \
              enable_if_static_array_t<T> = 0>                                 \
    ENOKI_INLINE decltype(auto) name(Args&&... args) {                         \
        return Base::dispatch(                                                 \
            [](Type *self, auto&&... args2) { return self->name(args2...); },  \
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

#define ENOKI_CALL_SUPPORT_END(Packet)                                         \
        };                                                                     \
    }

//! @}
// -----------------------------------------------------------------------

template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct Array : StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                               Array<Type_, Size_, Approx_, Mode_>> {
    using Base = StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                                 Array<Type_, Size_, Approx_, Mode_>>;

    ENOKI_DECLARE_CUSTOM_ARRAY(Base, Array)
    ENOKI_ALIGNED_OPERATOR_NEW()
};

NAMESPACE_END(enoki)
