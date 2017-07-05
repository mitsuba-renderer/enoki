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
                     !std::is_pointer<Type_>::value &&
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
    using Mask = Array<mask_t<Value>, Size>;

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

    template <typename T, typename T2 = Scalar,
              std::enable_if_t<std::is_same<T, bool>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(T value) {
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
        : m_data{{ StorageType(std::forward<Arg>(arg)),
                   StorageType(std::forward<Args>(args))... }} { ENOKI_CHKSCALAR }

    /// Convert a compatible array type (const)
    template <
        typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
        typename Derived2,
        std::enable_if_t<std::is_constructible<Type_, const Type2 &>::value &&
                        !std::is_same<bool, Type2>::value &&
                         Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(a, std::make_index_sequence<Size>()) { }

    /// Convert a compatible array type (non-const, useful when storing references)
    template <typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
              typename Derived2,
              std::enable_if_t<std::is_constructible<Type_, Type2 &>::value &&
                              !std::is_constructible<Type_, const Type2 &>::value &&
                              !std::is_same<bool, Type2>::value &&
                               Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
              StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a)
        : StaticArrayImpl(a, std::make_index_sequence<Size>()) { }

    /// Construct from a mask
    template <size_t Size2, bool Approx2, RoundingMode Mode2, typename Derived2,
              std::enable_if_t<Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<bool, Size2, Approx2, Mode2, Derived2> &a) {
        using Int = typename int_array_t<Derived>::Scalar;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = a.derived().coeff(i) ? memcpy_cast<Scalar>(Int(-1))
                                            : memcpy_cast<Scalar>(Int(0));
    }

    /// Reinterpret another array
    template <typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
              typename Derived2, std::enable_if_t<Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a,
        detail::reinterpret_flag) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = reinterpret_array<Value>(a.derived().coeff(i));
    }

    //! @}
    // -----------------------------------------------------------------------


#if defined(__F16C__)
    // -----------------------------------------------------------------------
    //! @{ \name Half-precision conversions
    // -----------------------------------------------------------------------

    template <bool Approx2, RoundingMode Mode2, typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_same<typename T::Value, half>::value &&
                               Derived2::Size == T::Size, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<float, 4, Approx2, Mode2, Derived2> &a) {
        __m128i value = _mm_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION);
        memcpy(data(), &value, sizeof(uint16_t) * Derived::Size);
    }

#if defined(__AVX__)
    template <bool Approx2, RoundingMode Mode2, typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_same<typename T::Value, half>::value &&
                               Derived2::Size == T::Size, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<double, 4, Approx2, Mode2, Derived2> &a) {
        __m128i value = _mm_cvtps_ph(_mm256_cvtpd_ps(a.derived().m), _MM_FROUND_CUR_DIRECTION);
        memcpy(data(), &value, sizeof(uint16_t) * Derived::Size);
    }

    template <bool Approx2, RoundingMode Mode2, typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_same<typename T::Value, half>::value &&
                               Derived2::Size == T::Size, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<float, 8, Approx2, Mode2, Derived2> &a) {
        _mm_storeu_si128((__m128i *) data(), _mm256_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION));
    }
#endif

#if defined(__AVX512F__)
    template <bool Approx2, RoundingMode Mode2, typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_same<typename T::Value, half>::value &&
                               Derived2::Size == T::Size, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<double, 8, Approx2, Mode2, Derived2> &a) {
        _mm_storeu_si128((__m128i *) data(), _mm256_cvtps_ph(_mm512_cvtpd_ps(a.derived().m), _MM_FROUND_CUR_DIRECTION));
    }

    template <bool Approx2, RoundingMode Mode2, typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_same<typename T::Value, half>::value &&
                               Derived2::Size == T::Size, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<float, 16, Approx2, Mode2, Derived2> &a) {
        _mm256_storeu_si256((__m256i *) data(), _mm512_cvtps_ph(a.derived().m, _MM_FROUND_CUR_DIRECTION));
    }
#endif

    template <size_t Size, bool Approx2, RoundingMode Mode2, typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_same<typename T::Value, half>::value && (Size > 4) &&
                               Derived2::Size == T::Size, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<double, Size, Approx2, Mode2, Derived2> &a)
       : StaticArrayImpl(
               Array<half, Size1, false, Mode2>(low(a)),
               Array<half, Size2, false, Mode2>(high(a))) { }

    template <size_t Size, bool Approx2, RoundingMode Mode2, typename Derived2, typename T = Derived,
              std::enable_if_t<std::is_same<typename T::Value, half>::value && (Size > 4) &&
                               Derived2::Size == T::Size, int> = 0>
    ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<float, Size, Approx2, Mode2, Derived2> &a)
       : StaticArrayImpl(
               Array<half, Size1, false, Mode2>(low(a)),
               Array<half, Size2, false, Mode2>(high(a))) { }

    //! @}
    // -----------------------------------------------------------------------
#endif

private:
    template <typename T, size_t... Index>
    ENOKI_INLINE StaticArrayImpl(T &&t, std::index_sequence<Index...>)
        : m_data{{ StorageType(t.derived().coeff(Index))... }} { }

public:

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

    /// Arithmetic unary NOT operation
    ENOKI_INLINE auto not_() const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::not_(coeff(i));
        return result;
    }

    /// Arithmetic OR operation
    template <typename Array>
    ENOKI_INLINE auto or_(const Array &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::or_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic AND operation
    template <typename Array>
    ENOKI_INLINE auto and_(const Array &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::and_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic XOR operation
    template <typename Array>
    ENOKI_INLINE auto xor_(const Array &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::xor_(coeff(i), d.coeff(i));
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

    /// Ternary operator -- select between to values based on mask
    static ENOKI_INLINE auto select_(const Mask &m, const Derived &t, const Derived &f) {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = select(m.coeff(i), t.coeff(i), f.coeff(i));
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
    ENOKI_INLINE uint32_array_t<Value> count_() const {
        using Int = uint32_array_t<Value>;
        const Int one(1);
        Int result(0);
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            madd(result, one, reinterpret_array<mask_t<Int>>(coeff(i)));
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

    ENOKI_FORWARD_FUNCTION(exp)
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

    ENOKI_FORWARD_FUNCTION(erf)
    ENOKI_FORWARD_FUNCTION(erfinv)

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
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = load<Value>(static_cast<const Value *>(ptr) + i);
        return result;
    }

    template <typename T = Derived,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_(const void *ptr, const Mask &mask) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = load<Value>(static_cast<const Value *>(ptr) + i, mask.coeff(i));
        return result;
    }

    template <typename T = Derived,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = load_unaligned<Value>(static_cast<const Value *>(ptr) + i);
        return result;
    }

    template <typename T = Derived,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, const Mask &mask) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = load_unaligned<Value>(static_cast<const Value *>(ptr) + i, mask.coeff(i));
        return result;
    }

    void store_(void *ptr) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i));
    }

    void store_(void *ptr, const Mask &mask) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i), mask.coeff(i));
    }

    void store_unaligned_(void *ptr) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store_unaligned<Value>((void *) (static_cast<Value *>(ptr) + i), coeff(i));
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

    StaticArrayImpl(Type value) : Base(UnderlyingType(value)) { }

    ENOKI_INLINE const Type& coeff(size_t i) const { return (Type &) Base::coeff(i); }
    ENOKI_INLINE Type& coeff(size_t i) { return (Type &) Base::coeff(i); }
    ENOKI_INLINE const Type& operator[](size_t i) const { return (Type &) Base::operator[](i); }
    ENOKI_INLINE Type& operator[](size_t i) { return (Type &) Base::operator[](i); }
};

template <typename T> struct call_support {
    call_support(T &) { }
};

template <typename T> struct call_support_base {
    using Value = value_t<T>;
    using Mask = mask_t<T>;
    call_support_base(T &self) : self(self) {}
    T &self;

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype((std::declval<Value>()->*std::declval<Func>())(
                  std::declval<Args>()..., std::declval<Mask>())),
              std::enable_if_t<!std::is_void<Result>::value && sizeof...(Args) != 0, int> = 0>
    ENOKI_INLINE Result dispatch_(Func func, Args&&... args, InputMask mask_) {
        Mask mask = reinterpret_array<Mask>(mask_);
        Result result = zero<Result>();

        while (any(mask)) {
            Value value    = extract(self, mask);
            Mask active    = mask & eq(self, T(value));
            mask          &= ~active;
            result[active] = (value->*func)(args..., active);
        }

        return result;
    }

    template <typename Func, typename InputMask,
        typename Result = decltype((std::declval<Value>()->*std::declval<Func>())(
            std::declval<Mask>())),
        std::enable_if_t<!std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE Result dispatch_(Func func, InputMask mask_) {
        Mask mask = reinterpret_array<Mask>(mask_);
        Result result = zero<Result>();

        while (any(mask)) {
            Value value = extract(self, mask);
            Mask active = mask & eq(self, T(value));
            mask &= ~active;
            result[active] = (value->*func)(active);
        }

        return result;
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype((std::declval<Value>()->*std::declval<Func>())(
                  std::declval<Args>()..., std::declval<Mask>())),
              std::enable_if_t<std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE void dispatch_(Func func, Args&&... args, InputMask mask_) {
        Mask mask = reinterpret_array<Mask>(mask_);

        while (any(mask)) {
            Value value    = extract(self, mask);
            Mask active    = mask & eq(self, T(value));
            mask          &= ~active;
            (value->*func)(args..., active);
        }
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype((std::declval<Value>()->*std::declval<Func>())(
                  std::declval<Args>()...)),
              typename ResultArray = like_t<T, Result>,
              std::enable_if_t<!std::is_void<Result>::value && sizeof...(Args) != 0, int> = 0>
    ENOKI_INLINE ResultArray dispatch_scalar_(Func func, Args&&... args, InputMask mask_) {
        Mask mask = reinterpret_array<Mask>(mask_);
        ResultArray result = zero<Result>();

        while (any(mask)) {
            Value value    = extract(self, mask);
            Mask active    = mask & eq(self, T(value));
            mask          &= ~active;
            result[active] = (value->*func)(args...);
        }

        return result;
    }

    template <typename Func, typename InputMask,
        typename Result = decltype((std::declval<Value>()->*std::declval<Func>())()),
        typename ResultArray = like_t<T, Result>,
        std::enable_if_t<!std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE ResultArray dispatch_scalar_(Func func, InputMask mask_) {
        Mask mask = reinterpret_array<Mask>(mask_);
        ResultArray result = zero<Result>();

        while (any(mask)) {
            Value value = extract(self, mask);
            Mask active = mask & eq(self, T(value));
            mask &= ~active;
            result[active] = (value->*func)();
        }

        return result;
    }

    template <typename Func, typename InputMask, typename... Args,
              typename Result = decltype((std::declval<Value>()->*std::declval<Func>())(
                  std::declval<Args>()...)),
              std::enable_if_t<std::is_void<Result>::value, int> = 0>
    ENOKI_INLINE void dispatch_scalar_(Func func, Args&&... args, InputMask mask_) {
        Mask mask = reinterpret_array<Mask>(mask_);

        while (any(mask)) {
            Value value    = extract(self, mask);
            mask          &= neq(self, T(value));
            (value->*func)(args...);
        }
    }

    template <typename Func, size_t... Index, typename... Args,
              typename Last = detail::nth_t<sizeof...(Index), Args...>,
              enable_if_mask_t<Last> = 0>
    ENOKI_INLINE decltype(auto)
    dispatch_scalar_(Func func, std::index_sequence<Index...>, Args&&... args) {
        return dispatch_scalar_<Func, Last, detail::nth_t<Index, Args...>...>(
            func, std::forward<Args>(args)...);
    }

    template <typename Func, size_t... Index, typename... Args,
              typename Last = detail::nth_t<sizeof...(Index), Args...>,
              enable_if_not_mask_t<Last> = 0>
    ENOKI_INLINE decltype(auto)
    dispatch_scalar_(Func func, std::index_sequence<Index...>, Args&&... args) {
        return dispatch_scalar_<Func, Mask, Args...>(
            func, std::forward<Args>(args)..., Mask(true));
    }

    template <typename Func, size_t... Index, typename... Args,
              typename Last = detail::nth_t<sizeof...(Index), Args...>,
              enable_if_mask_t<Last> = 0>
    ENOKI_INLINE decltype(auto)
    dispatch_(Func func, std::index_sequence<Index...>, Args&&... args) {
        return dispatch_<Func, Last, detail::nth_t<Index, Args...>...>(
            func, std::forward<Args>(args)...);
    }

    template <typename Func, size_t... Index, typename... Args,
              typename Last = detail::nth_t<sizeof...(Index), Args...>,
              enable_if_not_mask_t<Last> = 0>
    ENOKI_INLINE decltype(auto)
    dispatch_(Func func, std::index_sequence<Index...>, Args&&... args) {
        return dispatch_<Func, Mask, Args...>(
            func, std::forward<Args>(args)..., Mask(true));
    }
};

/// Pointer support
template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_, typename Derived_>
struct StaticArrayImpl<Type_, Size_, Approx_, Mode_, Derived_,
    std::enable_if_t<std::is_pointer<Type_>::value>>
    : StaticArrayImpl<std::uintptr_t, Size_, Approx_, Mode_, Derived_> {

    using UnderlyingType = std::uintptr_t;
    using Base = StaticArrayImpl<UnderlyingType, Size_, Approx_, Mode_, Derived_>;

    using Base::Base;
    using Base::operator=;
    using Base::derived;
    using Type = Type_;
    using Value = Type_;
    using Scalar = Type_;

    StaticArrayImpl(Type value) : Base(UnderlyingType(value)) { }

    ENOKI_INLINE const Type& coeff(size_t i) const { return (Type &) Base::coeff(i); }
    ENOKI_INLINE Type& coeff(size_t i) { return (Type &) Base::coeff(i); }
    ENOKI_INLINE const Type& operator[](size_t i) const { return (Type &) Base::operator[](i); }
    ENOKI_INLINE Type& operator[](size_t i) { return (Type &) Base::operator[](i); }

    call_support<Derived_> operator->() { return call_support<Derived_>(derived()); }
    call_support<const Derived_> operator->() const { return call_support<const Derived_>(derived()); }
};

#define ENOKI_CALL_SUPPORT_BEGIN(T)                                            \
    namespace enoki {                                                          \
    template <> struct call_support<T> : call_support_base<T> {                \
        using Base = call_support_base<T>;                                     \
        using Base::Base;                                                      \
        using Base::dispatch_;                                                 \
        using Base::dispatch_scalar_;                                          \
        using Type = std::remove_pointer_t<Base::Value>;                       \
        auto operator-> () { return this; }

#define ENOKI_CALL_SUPPORT(name)                                               \
    template <typename... Args>                                                \
    ENOKI_INLINE decltype(auto) name(Args &&... args) {                        \
        constexpr size_t Size = sizeof...(Args) > 0 ? sizeof...(Args) - 1 : 0; \
        return dispatch_(&Type::name, std::make_index_sequence<Size>(),        \
                         std::forward<Args>(args)...);                         \
    }

#define ENOKI_CALL_SUPPORT_SCALAR(name)                                        \
    template <typename... Args>                                                \
    ENOKI_INLINE decltype(auto) name(Args &&... args) {                        \
        constexpr size_t Size = sizeof...(Args) > 0 ? sizeof...(Args) - 1 : 0; \
        return dispatch_scalar_(&Type::name, std::make_index_sequence<Size>(), \
                                std::forward<Args>(args)...);                  \
    }

#define ENOKI_CALL_SUPPORT_END(T)                                              \
        };                                                                     \
    }

//! @}
// -----------------------------------------------------------------------

#define ENOKI_DECLARE_CUSTOM_ARRAY(Base, Array)                                \
    using Base::Base;                                                          \
    using Base::operator=;                                                     \
    Array() = default;                                                         \
    Array(const Array &) = default;                                            \
    Array(Array &&) = default;                                                 \
    Array &operator=(const Array &) = default;                                 \
    Array &operator=(Array &&) = default;

template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct Array : StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                               Array<Type_, Size_, Approx_, Mode_>> {
    using Base = StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                                 Array<Type_, Size_, Approx_, Mode_>>;

    ENOKI_DECLARE_CUSTOM_ARRAY(Base, Array)
    ENOKI_ALIGNED_OPERATOR_NEW()
};

NAMESPACE_BEGIN(detail)

template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct MaskWrapper : StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                                     MaskWrapper<Type_, Size_, Approx_, Mode_>> {
    using Base = StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                                 MaskWrapper<Type_, Size_, Approx_, Mode_>>;
    static constexpr bool IsMask = true;

    ENOKI_DECLARE_CUSTOM_ARRAY(Base, MaskWrapper)
    ENOKI_ALIGNED_OPERATOR_NEW()
};

NAMESPACE_END(detail)

NAMESPACE_END(enoki)
