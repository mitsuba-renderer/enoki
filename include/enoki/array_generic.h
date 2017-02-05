/*
    enoki/array_generic.h -- Generic array implementation that forwards
    all operations to the underlying data type (usually without making use of
    hardware vectorization)

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
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
                     !detail::is_recursive<Type_, Size_, RoundingMode::Default>::value>>
    : StaticArrayBase<Type_, Size_, Approx_, RoundingMode::Default, Derived_> {

    using Base =
        StaticArrayBase<Type_, Size_, Approx_, RoundingMode::Default, Derived_>;
    using Base::operator=;
    using typename Base::Scalar;
    using typename Base::Derived;
    using typename Base::BaseScalar;
    using typename Base::Array1;
    using typename Base::Array2;
    using Base::Size;
    using Base::Size1;
    using Base::Size2;
    using Mask = Array<mask_t<Scalar>, Size>;
    static constexpr bool Native = false;

    /// Denotes the type of an unary expression involving this type
    using Expr = std::conditional_t<std::is_same<expr_t<Type_>, Type_>::value, Derived,
                                    Array<expr_t<Type_>, Size, Approx_, RoundingMode::Default>>;

    using StorageType =
        std::conditional_t<std::is_reference<Type_>::value,
                           std::reference_wrapper<Scalar>, Scalar>;

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
    ENOKI_INLINE StaticArrayImpl(const Scalar &value) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            m_data[i] = value;
    }

    /// Initialize all components from a scalar
    template <typename T = Type_, std::enable_if_t<!std::is_reference<T>::value &&
                                                  !std::is_same<Scalar, BaseScalar>::value, int> = 0>
    #if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && __GNUC__ < 7
        /// Work around a bug in GCC: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=72824
        __attribute__((optimize("no-tree-loop-distribute-patterns")))
    #endif
    ENOKI_INLINE StaticArrayImpl(const BaseScalar &value) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            m_data[i] = Scalar(value);
    }

    template <typename T, typename T2 = BaseScalar,
              std::enable_if_t<std::is_same<T, bool>::value, int> = 0,
              typename Int = typename detail::type_chooser<sizeof(T2)>::Int>
    ENOKI_INLINE StaticArrayImpl(T b)
        : StaticArrayImpl(b ? memcpy_cast<Scalar>(Int(-1))
                            : memcpy_cast<Scalar>(Int(0))) { }

    /// Initialize the individual components
    template <typename Arg, typename... Args,
              /* Ugly, works around a compiler ICE in MSVC */
              std::enable_if_t<
                  detail::all_of<std::is_constructible<StorageType, Arg>::value,
                                 std::is_constructible<StorageType, Args>::value...,
                                 sizeof...(Args) + 1 == Size_ && (sizeof...(Args) > 0)>::value, int> = 0>
    ENOKI_INLINE StaticArrayImpl(Arg &&arg, Args &&... args)
        : m_data{{ StorageType(arg), StorageType(args)... }} { ENOKI_CHKSCALAR }

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
        using Int = typename int_array_t<Derived>::BaseScalar;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = a.derived().coeff(i) ? memcpy_cast<BaseScalar>(Int(-1))
                                            : memcpy_cast<BaseScalar>(Int(0));
    }

    /// Reinterpret another array
    template <typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
              typename Derived2, std::enable_if_t<Derived2::Size == Size_, int> = 0>
    ENOKI_INLINE StaticArrayImpl(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a,
        detail::reinterpret_flag) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            coeff(i) = reinterpret_array<Scalar>(a.derived().coeff(i));
    }

    //! @}
    // -----------------------------------------------------------------------

private:
    template <typename T, size_t... Index>
    ENOKI_INLINE StaticArrayImpl(T &&t, std::index_sequence<Index...>)
        : m_data{{ StorageType(t.derived().coeff(Index))... }} { }

public:

    // -----------------------------------------------------------------------
    //! @{ \name Generic implementations of vertical operations
    // -----------------------------------------------------------------------

    /// Addition
    ENOKI_INLINE Expr add_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) + d.coeff(i);
        return result;
    }

    /// Subtraction
    ENOKI_INLINE Expr sub_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) - d.coeff(i);
        return result;
    }

    /// Multiplication
    ENOKI_INLINE Expr mul_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) * d.coeff(i);
        return result;
    }

    /// High multiplication (integer)
    ENOKI_INLINE Expr mulhi_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = mulhi(coeff(i), d.coeff(i));
        return result;
    }

    /// Division
    ENOKI_INLINE Expr div_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) / d.coeff(i);
        return result;
    }

    /// Arithmetic unary NOT operation
    ENOKI_INLINE Expr not_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::not_(coeff(i));
        return result;
    }

    /// Arithmetic OR operation
    template <typename Array>
    ENOKI_INLINE Expr or_(const Array &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::or_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic AND operation
    template <typename Array>
    ENOKI_INLINE Expr and_(const Array &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::and_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic XOR operation
    template <typename Array>
    ENOKI_INLINE Expr xor_(const Array &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = detail::xor_(coeff(i), d.coeff(i));
        return result;
    }

    /// Arithmetic unary negation operation
    ENOKI_INLINE Expr neg_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = -coeff(i);
        return result;
    }

    /// Left shift operator
    ENOKI_INLINE Expr sl_(size_t value) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) << value;
        return result;
    }

    /// Left shift operator
    ENOKI_INLINE Expr slv_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) << d.coeff(i);
        return result;
    }

    /// Right shift operator
    ENOKI_INLINE Expr sr_(size_t value) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) >> value;
        return result;
    }

    /// Right shift operator
    ENOKI_INLINE Expr srv_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) >> d.coeff(i);
        return result;
    }

    /// Left shift operator (immediate)
    template <size_t Imm> ENOKI_INLINE Expr sli_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = sli<Imm>(coeff(i));
        return result;
    }

    /// Right shift operator (immediate)
    template <size_t Imm> ENOKI_INLINE Expr sri_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = sri<Imm>(coeff(i));
        return result;
    }

    /// Equality comparison operation
    ENOKI_INLINE Mask eq_(const Derived &d) const {
        Mask result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = eq(coeff(i), d.coeff(i));
        return result;
    }

    /// Inequality comparison operation
    ENOKI_INLINE Mask neq_(const Derived &d) const {
        Mask result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = neq(coeff(i), d.coeff(i));
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
    ENOKI_INLINE Expr abs_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = abs(coeff(i));
        return result;
    }

    /// Square root
    ENOKI_INLINE Expr sqrt_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = sqrt(coeff(i));
        return result;
    }

    /// Round to smallest integral value not less than argument
    ENOKI_INLINE Expr ceil_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = ceil(coeff(i));
        return result;
    }

    /// Round to largest integral value not greater than argument
    ENOKI_INLINE Expr floor_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = floor(coeff(i));
        return result;
    }

    /// Round to integral value
    ENOKI_INLINE Expr round_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = round(coeff(i));
        return result;
    }

    /// Element-wise maximum
    ENOKI_INLINE Expr max_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = max(coeff(i), d.coeff(i));
        return result;
    }

    /// Element-wise minimum
    ENOKI_INLINE Expr min_(const Derived &d) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = min(coeff(i), d.coeff(i));
        return result;
    }

    /// Fused multiply-add
    ENOKI_INLINE Expr fmadd_(const Derived &d1, const Derived &d2) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) * d1.coeff(i) + d2.coeff(i);
        return result;
    }

    /// Fused multiply-subtract
    ENOKI_INLINE Expr fmsub_(const Derived &d1, const Derived &d2) const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = coeff(i) * d1.coeff(i) - d2.coeff(i);
        return result;
    }

    /// Square root of the reciprocal
    ENOKI_INLINE Expr rsqrt_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = rsqrt<Approx_>(coeff(i));
        return result;
    }

    /// Reciprocal
    ENOKI_INLINE Expr rcp_() const {
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = rcp<Approx_>(coeff(i));
        return result;
    }

    /// Ternary operator -- select between to values based on mask
    ENOKI_INLINE static Expr select_(const Mask &m, const Derived &t, const Derived &f) {
        Expr r;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            r.coeff(i) = select(m.coeff(i), t.coeff(i), f.coeff(i));
        return r;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Generic implementations of horizontal operations
    // -----------------------------------------------------------------------

    /// Horizontal sum
    ENOKI_INLINE Scalar hsum_() const {
        Scalar result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result += coeff(i);
        return result;
    }

    /// Horizontal product
    ENOKI_INLINE Scalar hprod_() const {
        Scalar result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result *= coeff(i);
        return result;
    }

    /// Horizontal maximum
    ENOKI_INLINE Scalar hmax_() const {
        Scalar result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = max(result, coeff(i));
        return result;
    }

    /// Horizontal minimum
    ENOKI_INLINE Scalar hmin_() const {
        Scalar result = coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result = min(result, coeff(i));
        return result;
    }

    /// Check if all mask bits are set
    ENOKI_INLINE bool all_() const {
        bool result = all(coeff(0));
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result &= all(coeff(i));
        return result;
    }

    /// Check if any mask bits are set
    ENOKI_INLINE bool any_() const {
        bool result = any(coeff(0));
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result |= any(coeff(i));
        return result;
    }

    /// Check if none of the mask bits are set
    ENOKI_INLINE bool none_() const {
        bool result = none(coeff(0));
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result &= none(coeff(i));
        return result;
    }

    /// Count the number of active mask bits
    ENOKI_INLINE size_t count_() const {
        size_t result = count(coeff(0));
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result += count(coeff(i));
        return result;
    }

    /// Dot product
    ENOKI_INLINE Scalar dot_(const Derived &arg) const {
        Scalar result = coeff(0) * arg.coeff(0);
        ENOKI_CHKSCALAR for (size_t i = 1; i < Size; ++i)
            result += coeff(i) * arg.coeff(i);
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Element access
    // -----------------------------------------------------------------------

    ENOKI_INLINE Scalar &coeff(size_t i) { ENOKI_CHKSCALAR return m_data[i]; }
    ENOKI_INLINE const Scalar &coeff(size_t i) const { ENOKI_CHKSCALAR return m_data[i]; }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Higher-level functions
    // -----------------------------------------------------------------------

#define ENOKI_FORWARD_FUNCTION(name)                                           \
    Expr name##_() const {                                                     \
        if (std::is_arithmetic<Scalar>::value)                                 \
            return Base::name##_();                                            \
        Expr result;                                                           \
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)                      \
            result.coeff(i) = name(coeff(i));                                  \
        return result;                                                         \
    }

    Expr pow_(const Derived &arg) const {
        if (std::is_arithmetic<Scalar>::value)
            return Base::pow_(arg);
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = pow(coeff(i), arg.coeff(i));
        return result;
    }

    Expr ldexp_(const Derived &arg) const {
        if (std::is_arithmetic<Scalar>::value)
            return Base::ldexp_(arg);
        Expr result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = ldexp(coeff(i), arg.coeff(i));
        return result;
    }

    std::pair<Expr, Expr> frexp_() const {
        if (std::is_arithmetic<Scalar>::value)
            return Base::frexp_();
        std::pair<Expr, Expr> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            std::tie(result.first.coeff(i), result.second.coeff(i)) =
                frexp(coeff(i));
        return result;
    }

    std::pair<Expr, Expr> sincos_() const {
        if (std::is_arithmetic<Scalar>::value)
            return Base::sincos_();
        std::pair<Expr, Expr> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            std::tie(result.first.coeff(i), result.second.coeff(i)) =
                sincos(coeff(i));
        return result;
    }

    std::pair<Expr, Expr> sincosh_() const {
        if (std::is_arithmetic<Scalar>::value)
            return Base::sincosh_();
        std::pair<Expr, Expr> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            std::tie(result.first.coeff(i), result.second.coeff(i)) =
                sincosh(coeff(i));
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
    ENOKI_FORWARD_FUNCTION(erfi)

    #undef ENOKI_FORWARD_FUNCTION

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE static Expr zero_() { return Expr(Scalar(0)); }

    template <typename T = Derived, std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    ENOKI_INLINE static Derived load_(const void *ptr) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = load<Scalar>(static_cast<const Scalar *>(ptr) + i);
        return result;
    }

    template <typename T = Derived, std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    ENOKI_INLINE static Derived load_unaligned_(const void *ptr) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = load_unaligned<Scalar>(static_cast<const Scalar *>(ptr) + i);
        return result;
    }

    void store_(void *ptr) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store<Scalar>((void *) (static_cast<Scalar *>(ptr) + i), coeff(i));
    }

    void store_unaligned_(void *ptr) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            store_unaligned<Scalar>((void *) (static_cast<Scalar *>(ptr) + i), coeff(i));
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

    // -----------------------------------------------------------------------
    //! @{ \name Operations for dynamic arrays
    // -----------------------------------------------------------------------

    template <typename T = Scalar, std::enable_if_t<is_dynamic<T>::value, int> = 0>
    void dynamic_resize(size_t size) {
        for (size_t i = 0; i < Size; ++i)
            m_data[i].dynamic_resize(size);
    }

    template <typename T = Scalar, std::enable_if_t<is_dynamic<T>::value, int> = 0>
    ENOKI_INLINE size_t dynamic_size() const { return m_data[0].dynamic_size(); }

    template <typename T = Scalar, std::enable_if_t<is_dynamic<T>::value, int> = 0>
    ENOKI_INLINE size_t packets() const { return m_data[0].packets(); }

    template <typename T = Scalar, std::enable_if_t<is_dynamic<T>::value, int> = 0>
    ENOKI_INLINE auto packet(size_t i) {
        return packet(i, std::make_index_sequence<Size>());
    }

    template <typename T = Scalar, std::enable_if_t<is_dynamic<T>::value, int> = 0>
    ENOKI_INLINE auto packet(size_t i) const {
        return packet(i, std::make_index_sequence<Size>());
    }

    template <typename T = Scalar, std::enable_if_t<is_dynamic<T>::value, int> = 0>
    ENOKI_INLINE auto ref_() {
        return ref_(std::make_index_sequence<Size>());
    }

    template <typename T = Scalar, std::enable_if_t<is_dynamic<T>::value, int> = 0>
    ENOKI_INLINE auto ref_() const {
        return ref_(std::make_index_sequence<Size>());
    }

private:
    template <size_t... Index>
    ENOKI_INLINE auto packet(size_t i, std::index_sequence<Index...>) {
        return Array<decltype(enoki::packet(std::declval<Scalar>(), 0)), Size>(
            enoki::packet(m_data[Index], i)...
        );
    }

    template <size_t... Index>
    ENOKI_INLINE auto packet(size_t i, std::index_sequence<Index...>) const {
        return Array<decltype(enoki::packet(std::declval<const Scalar &>(), 0)), Size>(
            enoki::packet(m_data[Index], i)...
        );
    }

    template <size_t... Index>
    ENOKI_INLINE auto ref_(std::index_sequence<Index...>) {
        return Array<decltype(detail::ref(std::declval<Scalar>())), Size>(
            detail::ref(m_data[Index])...
        );
    }

    template <size_t... Index>
    ENOKI_INLINE auto ref_(std::index_sequence<Index...>) const {
        return Array<decltype(detail::ref(std::declval<const Scalar &>())), Size>(
            detail::ref(m_data[Index])...
        );
    }

    //! @}
    // -----------------------------------------------------------------------

private:
    std::array<StorageType, Size> m_data;
};

//! @}
// -----------------------------------------------------------------------

template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_>
struct Array : StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                               Array<Type_, Size_, Approx_, Mode_>> {
    using Base = StaticArrayImpl<Type_, Size_, Approx_, Mode_,
                                 Array<Type_, Size_, Approx_, Mode_>>;
    using Base::Base;
    using Base::operator=;
};

NAMESPACE_END(enoki)
