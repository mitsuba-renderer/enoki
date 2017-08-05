/*
    enoki/array_base.h -- Base classes of all Enoki array data structures

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "array_router.h"

NAMESPACE_BEGIN(enoki)

NAMESPACE_BEGIN(detail)

template <typename Derived> struct MaskedElement {
    Derived &d;
    typename Derived::Mask m;
    template <typename T> ENOKI_INLINE void operator=(T value) { d.massign_(value, m); }
    template <typename T> ENOKI_INLINE void operator+=(T value) { d.madd_(value, m); }
    template <typename T> ENOKI_INLINE void operator-=(T value) { d.msub_(value, m); }
    template <typename T> ENOKI_INLINE void operator*=(T value) { d.mmul_(value, m); }
    template <typename T> ENOKI_INLINE void operator/=(T value) { d.mdiv_(value, m); }
    template <typename T> ENOKI_INLINE void operator|=(T value) { d.mor_(value, m); }
    template <typename T> ENOKI_INLINE void operator&=(T value) { d.mand_(value, m); }
    template <typename T> ENOKI_INLINE void operator^=(T value) { d.mxor_(value, m); }
};

NAMESPACE_END(detail)

template <typename Type_, typename Derived_> struct ArrayBase {
    // -----------------------------------------------------------------------
    //! @{ \name Curiously Recurring Template design pattern
    // -----------------------------------------------------------------------

    /// Alias to the derived type
    using Derived = Derived_;

    /// Cast to derived type
    ENOKI_INLINE Derived &derived()             { return (Derived &) *this; }

    /// Cast to derived type (const version)
    ENOKI_INLINE const Derived &derived() const { return (Derived &) *this; }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Basic declarations
    // -----------------------------------------------------------------------

    /// Actual type underlying the derived array
    using Type = Type_;

    /// Base type underlying the derived array (i.e. without references etc.)
    using Value = std::remove_reference_t<Type>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Iterators
    // -----------------------------------------------------------------------

    ENOKI_INLINE const Value *begin() const {
        ENOKI_CHKSCALAR return derived().data();
    }

    ENOKI_INLINE Value *begin() {
        ENOKI_CHKSCALAR return derived().data();
    }

    ENOKI_INLINE const Value *end() const {
        ENOKI_CHKSCALAR return derived().data() + derived().size();
    }

    ENOKI_INLINE Value *end() {
        ENOKI_CHKSCALAR return derived().data() + derived().size();
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Element access
    // -----------------------------------------------------------------------

    /// Array indexing operator with bounds checks in debug mode
    ENOKI_INLINE Value &operator[](size_t i) {
        #if !defined(NDEBUG) && !defined(ENOKI_DISABLE_RANGE_CHECK)
            if (i >= derived().size())
                throw std::out_of_range(
                    "ArrayBase: out of range access (tried to access index " +
                    std::to_string(i) + " in an array of size " +
                    std::to_string(derived().size()) + ")");
        #endif
        return (Value &) derived().coeff(i);
    }

    /// Array indexing operator with bounds checks in debug mode, const version
    ENOKI_INLINE const Value &operator[](size_t i) const {
        #if !defined(NDEBUG) && !defined(ENOKI_DISABLE_RANGE_CHECK)
            if (i >= derived().size())
                throw std::out_of_range(
                    "ArrayBase: out of range access (tried to access index " +
                    std::to_string(i) + " in an array of size " +
                    std::to_string(derived().size()) + ")");
        #endif
        return (const Value &) derived().coeff(i);
    }

    //! @}
    // -----------------------------------------------------------------------
};

/**
 * \brief Base class containing rudimentary operations and type aliases used by
 * all static and dynamic array implementations
 *
 * This data structure provides various rudimentary operations that are implemented
 * using functionality provided by the target-specific 'Derived' subclass (e.g.
 * ``operator+=`` using ``operator+`` and ``operator=``). This avoids a
 * considerable amount of repetition in target-specific specializations. The
 * implementation makes use of the Curiously Recurring Template design pattern,
 * which enables inlining and other compiler optimizations.
 */
template <typename Type_, size_t Size_, bool Approx_, RoundingMode Mode_,
          typename Derived_>
struct StaticArrayBase : ArrayBase<Type_, Derived_> {
    using Base = ArrayBase<Type_, Derived_>;
    using typename Base::Derived;
    using typename Base::Type;
    using typename Base::Value;
    using Base::derived;

    /// Size of the first sub-array (used to split this array into two parts)
    static constexpr size_t Size1 = detail::lpow2(Size_);

    /// Size of the second sub-array (used to split this array into two parts)
    static constexpr size_t Size2 = Size_ - Size1;

    /// First sub-array type (used to split this array into two parts)
    using Array1 = Array<Type_, Size1, Approx_, Mode_>;

    /// Second sub-array type (used to split this array into two parts)
    using Array2 = Array<Type_, Size2, Approx_, Mode_>;

    /// Value data type all the way at the lowest level
    using Scalar = scalar_t<Value>;

    /// Is this array exclusively for mask usage? (overridden in some subclasses)
    static constexpr bool IsMask = is_mask<Type_>::value;

    /// Can this array be represented using a processor vector register? (no by default)
    static constexpr bool IsNative = false;

    /// Does this array instantiate itself recursively? (see 'array_recursive.h')
    static constexpr bool IsRecursive = false;

    /// Number of array entries
    static constexpr size_t Size = Size_;
    static constexpr size_t ActualSize = Size;

    /// Are arithmetic operations approximate?
    static constexpr bool Approx = Approx_;

    /// Rounding mode of arithmetic operations
    static constexpr RoundingMode Mode = Mode_;

    /// Type alias for a similar-shaped array over a different type
    template <typename T, typename T2 = Derived>
    using ReplaceType = Array<T, T2::Size,
          detail::is_std_float<scalar_t<T>>::value ? T2::Approx : detail::approx_default<T>::value,
          detail::is_std_float<scalar_t<T>>::value ? T2::Mode : RoundingMode::Default
    >;

    static_assert(detail::is_std_float<Scalar>::value || !Approx,
                  "Approximate math library functions are only supported for "
                  "single and double precision arrays!");

    static_assert(!std::is_integral<Value>::value || Mode == RoundingMode::Default,
                  "Integer arrays require Mode == RoundingMode::Default");

    StaticArrayBase() = default;

    /// Assign from a compatible array type
    template <
        typename Type2, size_t Size2, bool Approx2, RoundingMode Mode2,
        typename Derived2,
        std::enable_if_t<std::is_assignable<Value &, Type2>::value, int> = 0>
    ENOKI_INLINE Derived &operator=(
        const StaticArrayBase<Type2, Size2, Approx2, Mode2, Derived2> &a) {
        static_assert(Derived2::Size == Derived::Size, "Size mismatch!");
        if (std::is_arithmetic<Value>::value) { ENOKI_TRACK_SCALAR }
        for (size_t i = 0; i < Size; ++i)
            derived().coeff(i) = a.derived().coeff(i);
        return derived();
    }

    ENOKI_INLINE Derived &operator=(const Scalar &value) {
        derived() = Derived(value);
        return derived();
    }

    using Base::operator[];
    template <typename T = Derived, typename Mask, enable_if_mask_t<Mask> = 0>
    ENOKI_INLINE detail::MaskedElement<T> operator[](Mask m) {
        return detail::MaskedElement<T>{ derived(), reinterpret_array<mask_t<Derived>>(m) };
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Mathematical support library
    // -----------------------------------------------------------------------

    /// Element-wise test for NaN values
    ENOKI_INLINE auto isnan_() const {
        return !eq(derived(), derived());
    }

    /// Element-wise test for +/- infinity
    ENOKI_INLINE auto isinf_() const {
        return eq(abs(derived()), std::numeric_limits<Scalar>::infinity());
    }

    /// Element-wise test for finiteness
    ENOKI_INLINE auto isfinite_() const {
        return abs(derived()) < std::numeric_limits<Scalar>::max();
    }

    /// Division fallback implementation
    ENOKI_INLINE auto div_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = derived().coeff(i) / d.coeff(i);
        return result;
    }

    /// Modulo fallback implementation
    ENOKI_INLINE auto mod_(const Derived &d) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = derived().coeff(i) % d.coeff(i);
        return result;
    }

    /// Left rotation operation fallback implementation
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto rol_(size_t k) const {
        using Expr = expr_t<Derived>;
        if (!std::is_signed<Scalar>::value) {
            constexpr size_t mask = 8 * sizeof(Scalar) - 1u;
            return Expr((derived() << (k & mask)) | (derived() >> ((~k + 1u) & mask)));
        } else {
            return Expr(uint_array_t<Expr>(derived()).rol_(k));
        }
    }

    /// Right rotation operation fallback implementation
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto ror_(size_t k) const {
        using Expr = expr_t<Derived>;
        if (!std::is_signed<Scalar>::value) {
            constexpr size_t mask = 8 * sizeof(Scalar) - 1u;
            return Expr((derived() >> (k & mask)) | (derived() << ((~k + 1u) & mask)));
        } else {
            return Expr(uint_array_t<Expr>(derived()).ror_(k));
        }
    }

    /// Left rotation operation fallback implementation
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto rolv_(const Derived &d) const {
        using Expr = expr_t<Derived>;
        if (!std::is_signed<Scalar>::value) {
            Expr mask(Scalar(8 * sizeof(Scalar) - 1u));
            return Expr((derived() << (d & mask)) | (derived() >> ((~d + Scalar(1)) & mask)));
        } else {
            return Expr(uint_array_t<Expr>(derived()).rolv_(d));
        }
    }

    /// Right rotation operation fallback implementation
    template <typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto rorv_(const Derived &d) const {
        using Expr = expr_t<Derived>;
        if (!std::is_signed<Scalar>::value) {
            Expr mask(Scalar(8 * sizeof(Scalar) - 1u));
            return Expr((derived() >> (d & mask)) | (derived() << ((~d + Scalar(1)) & mask)));
        } else {
            return Expr(uint_array_t<Expr>(derived()).rorv_(d));
        }
    }

    /// Left rotation operation fallback implementation (immediate)
    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto roli_() const {
        using Expr = expr_t<Derived>;
        if (!std::is_signed<Scalar>::value) {
            constexpr size_t mask = 8 * sizeof(Scalar) - 1u;
            return Expr(sli<Imm & mask>(derived()) | sri<((~Imm + 1u) & mask)>(derived()));
        } else {
            return Expr(uint_array_t<Expr>(derived()).template roli_<Imm>());
        }
    }

    /// Right rotation operation fallback implementation (immediate)
    template <size_t Imm, typename T = Scalar, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    ENOKI_INLINE auto rori_() const {
        using Expr = expr_t<Derived>;
        if (!std::is_signed<Scalar>::value) {
            constexpr size_t mask = 8 * sizeof(Scalar) - 1u;
            return Expr(sri<Imm & mask>(derived()) | sli<((~Imm + 1u) & mask)>(derived()));
        } else {
            return Expr(uint_array_t<Expr>(derived()).template rori_<Imm>());
        }
    }

    /// Rotate the entries of the array right
    template <size_t Imm>
    ENOKI_INLINE auto ror_array_() const {
        return ror_array_<Imm>(std::make_index_sequence<Derived::Size>());
    }

    template <size_t Imm, size_t... Index>
    ENOKI_INLINE auto ror_array_(std::index_sequence<Index...>) const {
        return shuffle<(Derived::Size + Index - Imm) % Derived::Size...>(derived());
    }

    /// Rotate the entries of the array left
    template <size_t Imm>
    ENOKI_INLINE auto rol_array_() const {
        return rol_array_<Imm>(std::make_index_sequence<Derived::Size>());
    }

    template <size_t Imm, size_t... Index>
    ENOKI_INLINE auto rol_array_(std::index_sequence<Index...>) const {
        return shuffle<(Index + Imm) % Derived::Size...>(derived());
    }

    /// Arithmetic NOT operation fallback
    ENOKI_INLINE auto not_() const {
        using Expr = expr_t<Derived>;
        const Expr mask(memcpy_cast<Scalar>(typename int_array_t<Expr>::Scalar(-1)));
        return Expr(derived() ^ mask);
    }

    /// Arithmetic unary negation operation fallback
    ENOKI_INLINE auto neg_() const {
        using Expr = expr_t<Derived>;
        if (std::is_floating_point<Value>::value)
            return Expr(derived() ^ Scalar(-0.f));
        else
            return Expr(~derived() + Scalar(1));
    }

    /// Reciprocal fallback implementation
    ENOKI_INLINE auto rcp_() const {
        return expr_t<Derived>(Scalar(1)) / derived();
    }

    /// Reciprocal square root fallback implementation
    ENOKI_INLINE auto rsqrt_() const {
        return expr_t<Derived>(Scalar(1)) / sqrt(derived());
    }

    /// Fused multiply-add fallback implementation
    ENOKI_INLINE auto fmadd_(const Derived &b, const Derived &c) const {
        return derived() * b + c;
    }

    /// Fused negative multiply-add fallback implementation
    ENOKI_INLINE auto fnmadd_(const Derived &b, const Derived &c) const {
        return -derived() * b + c;
    }

    /// Fused multiply-subtract fallback implementation
    ENOKI_INLINE auto fmsub_(const Derived &b, const Derived &c) const {
        return derived() * b - c;
    }

    /// Fused negative multiply-subtract fallback implementation
    ENOKI_INLINE auto fnmsub_(const Derived &b, const Derived &c) const {
        return -derived() * b - c;
    }

    /// Fused multiply-add/subtract fallback implementation
    ENOKI_INLINE auto fmaddsub_(const Derived &b, const Derived &c) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i) {
            if (i % 2 == 0)
                result.coeff(i) = fmsub(derived().coeff(i), b.coeff(i), c.coeff(i));
            else
                result.coeff(i) = fmadd(derived().coeff(i), b.coeff(i), c.coeff(i));
        }
        return result;
    }

    /// Fused multiply-subtract/add fallback implementation
    ENOKI_INLINE auto fmsubadd_(const Derived &b, const Derived &c) const {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i) {
            if (i % 2 == 0)
                result.coeff(i) = fmadd(derived().coeff(i), b.coeff(i), c.coeff(i));
            else
                result.coeff(i) = fmsub(derived().coeff(i), b.coeff(i), c.coeff(i));
        }
        return result;
    }

    /// Dot product fallback implementation
    ENOKI_INLINE Value dot_(const Derived &a) const { return hsum(derived() * a); }

    /// Nested horizontal sum
    ENOKI_INLINE auto hsum_nested_() const { return hsum_nested(hsum(derived())); }

    /// Nested horizontal product
    ENOKI_INLINE auto hprod_nested_() const { return hprod_nested(hprod(derived())); }

    /// Nested horizontal minimum
    ENOKI_INLINE auto hmin_nested_() const { return hmin_nested(hmin(derived())); }

    /// Nested horizontal maximum
    ENOKI_INLINE auto hmax_nested_() const { return hmax_nested(hmax(derived())); }

    /// Nested all() mask operation
    ENOKI_INLINE bool all_nested_() const { return all_nested(all(derived())); }

    /// Nested any() mask operation
    ENOKI_INLINE bool any_nested_() const { return any_nested(any(derived())); }

    /// Nested none() mask operation
    ENOKI_INLINE bool none_nested_() const { return !any_nested(any(derived())); }

    /// Nested count() mask operation
    ENOKI_INLINE auto count_nested_() const { return hsum_nested(count(derived())); }

    /// Shuffle operation fallback implementation
    template <size_t ... Indices> ENOKI_INLINE auto shuffle_() const {
        static_assert(sizeof...(Indices) == Size ||
                      sizeof...(Indices) == Derived::Size, "shuffle(): Invalid size!");
        expr_t<Derived> out;
        size_t idx = 0;
        ENOKI_CHKSCALAR bool result[] = { (out.coeff(idx++) = derived().coeff(Indices), false)... };
        (void) idx; (void) result;
        return out;
    }

    /// Extract fallback implementation
    template <typename Mask>
    ENOKI_INLINE Value extract_(const Mask &mask) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            if (bool(mask.coeff(i)))
                return (Value) derived().coeff(i);
        return Value(0);
    }

    /// Prefetch operation fallback implementation
    template <size_t Stride, bool Write, size_t Level, typename Index>
    static ENOKI_INLINE void prefetch_(const void *mem, const Index &index) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            prefetch<Value, Stride, Write, Level>(mem, index.coeff(i));
    }

    /// Masked prefetch operation fallback implementation
    template <size_t Stride, bool Write, size_t Level, typename Index, typename Mask>
    static ENOKI_INLINE void prefetch_(const void *mem, const Index &index,
                                       const Mask &mask) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            prefetch<Value, Stride, Write, Level>(mem, index.coeff(i), mask.coeff(i));
    }

    /// Gather operation fallback implementation
    template <size_t Stride, typename Index>
    static ENOKI_INLINE auto gather_(const void *mem, const Index &index) {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            result.coeff(i) = gather<Value, Stride>(mem, index.coeff(i));
        return result;
    }

    /// Masked gather operation fallback implementation
    template <size_t Stride, typename Index, typename Mask>
    static ENOKI_INLINE auto gather_(const void *mem, const Index &index,
                                     const Mask &mask) {
        expr_t<Derived> result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            result.coeff(i) = gather<Value, Stride>(mem, index.coeff(i), mask.coeff(i));
        return result;
    }

    /// Scatter operation fallback implementation
    template <size_t Stride, typename Index>
    ENOKI_INLINE void scatter_(void *mem, const Index &index) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            scatter<Stride>(mem, derived().coeff(i), index.coeff(i));
    }

    /// Masked scatter operation fallback implementation
    template <size_t Stride, typename Index, typename Mask>
    ENOKI_INLINE void scatter_(void *mem, const Index &index,
                               const Mask &mask) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            scatter<Stride>(mem, derived().coeff(i), index.coeff(i), mask.coeff(i));
    }

    /// Compressing store fallback implementation
    template <
        typename Mask, typename T = Derived,
        std::enable_if_t<(has_avx2 && T::Size >= 8 && std::is_same<value_t<T>, bool>::value), int> = 0>
    ENOKI_INLINE size_t compress_(Value *&mem, const Mask &mask) const {
        using Mask2 = mask_t<Array<uint32_t, Derived::Size>>;
        Mask2 arr_out;
        auto *arr_out_ptr = (value_t<Mask2>*) &arr_out;
        Mask2 input = reinterpret_array<Mask2>(*this);
        Mask2 input_mask = reinterpret_array<Mask2>(mask);
        size_t size = compress(arr_out_ptr, input, input_mask);
        Derived output_mask = reinterpret_array<Derived>(arr_out);
        store_unaligned(mem, output_mask);
        mem += size;
        return size;
    }

    template <
        typename Mask, typename T = Derived,
        std::enable_if_t<!(has_avx2 && T::Size >= 8 && std::is_same<value_t<T>, bool>::value), int> = 0>
    ENOKI_INLINE size_t compress_(Value *&mem, const Mask &mask) const {
        size_t result = 0;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            result += compress(mem, derived().coeff(i), mask.coeff(i));
        return result;
    }

    void resize_(size_t size) {
        if (size != Derived::Size)
            throw std::length_error("Incompatible size for static array");
    }

    /// Combined gather-modify-scatter operation without conflicts (fallback implementation)
    template <size_t Stride, typename Index, typename Func, typename Mask, typename... Args>
    static ENOKI_INLINE void transform_masked_(void *mem, const Index &index,
                                        const Mask &mask, const Func &func,
                                        const Args &... args) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            transform<Value, Stride>(mem, index.coeff(i), mask.coeff(i),
                                     func, args.coeff(i)...);
    }

    /// Combined gather-modify-scatter operation without conflicts (fallback implementation)
    template <size_t Stride, typename Index, typename Func, typename... Args>
    static ENOKI_INLINE void transform_(void *mem, const Index &index,
                                        const Func &func, const Args &... args) {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            transform<Value, Stride>(mem, index.coeff(i),
                                     func, args.coeff(i)...);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fallback implementations of masked operations
    // -----------------------------------------------------------------------

    #define ENOKI_MASKED_OPERATOR_FALLBACK(name, expr) \
        template <typename T = Derived> \
        ENOKI_INLINE void m##name##_(const expr_t<T> &e, const mask_t<T> &m) { \
            derived() = select(m, expr, derived()); \
        }

    ENOKI_MASKED_OPERATOR_FALLBACK(assign, e)
    ENOKI_MASKED_OPERATOR_FALLBACK(add, derived() + e)
    ENOKI_MASKED_OPERATOR_FALLBACK(sub, derived() - e)
    ENOKI_MASKED_OPERATOR_FALLBACK(mul, derived() * e)
    ENOKI_MASKED_OPERATOR_FALLBACK(div, derived() / e)
    ENOKI_MASKED_OPERATOR_FALLBACK(or, derived() | e)
    ENOKI_MASKED_OPERATOR_FALLBACK(and, derived() & e)
    ENOKI_MASKED_OPERATOR_FALLBACK(xor, derived() ^ e)

    #undef ENOKI_MASKED_OPERATOR_FALLBACK

    template <typename Mask>
    void store_(void *ptr, const Mask &mask) const {
        ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
            store<Value>((void *) (static_cast<Value *>(ptr) + i),
                         derived().coeff(i), mask.derived().coeff(i));
    }

    template <typename T = Derived, typename Mask,
              std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
    static ENOKI_INLINE Derived load_(const void *ptr, const Mask &mask) {
        Derived result;
        ENOKI_CHKSCALAR for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = load<Value>(static_cast<const Value *>(ptr) + i, mask.coeff(i));
        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Trigonometric and inverse trigonometric functions
    // -----------------------------------------------------------------------

    template <bool Sin, bool Cos, typename T>
    ENOKI_INLINE auto sincos_approx_(T &s_out, T &c_out) const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        using IntArray = int_array_t<Expr>;
        using Int = scalar_t<IntArray>;

        /* Joint sine & cosine function approximation based on CEPHES.
           Excellent accuracy in the domain |x| < 8192

           Redistributed under a BSD license with permission of the author, see
           https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

         - sin (in [-8192, 8192]):
           * avg abs. err = 6.61896e-09
           * avg rel. err = 1.37888e-08
              -> in ULPs  = 0.166492
           * max abs. err = 5.96046e-08
             (at x=-8191.31)
           * max rel. err = 1.76826e-06
             -> in ULPs   = 19
             (at x=-6374.29)

         - cos (in [-8192, 8192]):
           * avg abs. err = 6.59965e-09
           * avg rel. err = 1.37432e-08
              -> in ULPs  = 0.166141
           * max abs. err = 5.96046e-08
             (at x=-8191.05)
           * max rel. err = 3.13993e-06
             -> in ULPs   = 47
             (at x=-6199.93)
        */

        Expr x = abs(derived());

        /* Scale by 4/Pi and get the integer part */
        IntArray j(x * Scalar(1.2732395447351626862));

        /* Map zeros to origin; if (j & 1) j += 1 */
        j = (j + Int(1)) & Int(~1u);

        /* Cast back to a floating point value */
        Expr y(j);

        /* Determine sign of result */
        Expr sign_sin, sign_cos;
        constexpr size_t Shift = sizeof(Scalar) * 8 - 3;

        if (Sin)
            sign_sin = detail::sign_mask(reinterpret_array<Expr>(sli<Shift>(j)) ^ derived());

        if (Cos)
            sign_cos = detail::sign_mask(reinterpret_array<Expr>(sli<Shift>(~(j - Int(2)))));

        /* Extended precision modular arithmetic */
        if (Single) {
            y = x - y * Scalar(0.78515625)
                  - y * Scalar(2.4187564849853515625e-4)
                  - y * Scalar(3.77489497744594108e-8);
        } else {
            y = x - y * Scalar(7.85398125648498535156e-1)
                  - y * Scalar(3.77489470793079817668e-8)
                  - y * Scalar(2.69515142907905952645e-15);
        }

        Expr z = y * y, s, c;
        z |= eq(x, std::numeric_limits<Scalar>::infinity());

        if (Single) {
            s = poly2(z, -1.6666654611e-1,
                          8.3321608736e-3,
                         -1.9515295891e-4) * z;

            c = poly2(z,  4.166664568298827e-2,
                         -1.388731625493765e-3,
                          2.443315711809948e-5) * z;
        } else {
            s = poly5(z, -1.66666666666666307295e-1,
                          8.33333333332211858878e-3,
                         -1.98412698295895385996e-4,
                          2.75573136213857245213e-6,
                         -2.50507477628578072866e-8,
                          1.58962301576546568060e-10) * z;

            c = poly5(z,  4.16666666666665929218e-2,
                         -1.38888888888730564116e-3,
                          2.48015872888517045348e-5,
                         -2.75573141792967388112e-7,
                          2.08757008419747316778e-9,
                         -1.13585365213876817300e-11) * z;
        }

        s = fmadd(s, y, y);
        c = fmadd(c, z, fmadd(z, Scalar(-0.5), Scalar(1)));

        mask_t<Expr> polymask(eq(j & Int(2), zero<IntArray>()));

        if (Sin)
            s_out = select(polymask, s, c) ^ sign_sin;

        if (Cos)
            c_out = select(polymask, c, s) ^ sign_cos;
    }

    template <bool Tan, typename T>
    ENOKI_INLINE auto tancot_approx_(T &r) const {
        using Expr = expr_t<Derived>;
        using IntArray = int_array_t<Expr>;
        using Int = scalar_t<IntArray>;
        constexpr bool Single = std::is_same<Scalar, float>::value;

        /*
         - tan (in [-8192, 8192]):
           * avg abs. err = 4.63693e-06
           * avg rel. err = 3.60191e-08
              -> in ULPs  = 0.435442
           * max abs. err = 0.8125
             (at x=-6199.93)
           * max rel. err = 3.12284e-06
             -> in ULPs   = 30
             (at x=-7406.3)
        */

        Expr x = abs(derived());

        /* Scale by 4/Pi and get the integer part */
        IntArray j(x * Scalar(1.2732395447351626862));

        /* Map zeros to origin; if (j & 1) j += 1 */
        j = (j + Int(1)) & Int(~1u);

        /* Cast back to a floating point value */
        Expr y(j);

        /* Extended precision modular arithmetic */
        if (Single) {
            y = x - y * Scalar(0.78515625)
                  - y * Scalar(2.4187564849853515625e-4)
                  - y * Scalar(3.77489497744594108e-8);
        } else {
            y = x - y * Scalar(7.85398125648498535156e-1)
                  - y * Scalar(3.77489470793079817668e-8)
                  - y * Scalar(2.69515142907905952645e-15);
        }

        Expr z = y * y;
        z |= eq(x, std::numeric_limits<Scalar>::infinity());

        constexpr size_t Shift = sizeof(Scalar) * 8 - 2;

        auto sign_tan = detail::sign_mask(
            reinterpret_array<Expr>(sli<Shift>(j)) ^ derived());

        if (Single) {
            r = poly5(z, 3.33331568548e-1,
                         1.33387994085e-1,
                         5.34112807005e-2,
                         2.44301354525e-2,
                         3.11992232697e-3,
                         9.38540185543e-3);
        } else {
            r = poly2(z, -1.79565251976484877988e7,
                          1.15351664838587416140e6,
                         -1.30936939181383777646e4) /
                poly4(z, -5.38695755929454629881e7,
                          2.50083801823357915839e7,
                         -1.32089234440210967447e6,
                          1.36812963470692954678e4,
                          1.00000000000000000000e0);
        }

        r = fmadd(r, z * y, y);

        auto recip_mask = Tan ? neq(j & Int(2), Int(0)) : eq(j & Int(2), Int(0));
        r[x < Scalar(1e-4)] = y;
        r[recip_mask] = rcp(r);
        r = (r ^ sign_tan);
    }

    auto sin_() const {
        expr_t<Derived> r;
        if (Approx) {
            sincos_approx_<true, false>(r, r);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = sin(derived().coeff(i));
        }
        return r;
    }

    auto cos_() const {
        expr_t<Derived> r;
        if (Approx) {
            sincos_approx_<false, true>(r, r);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = cos(derived().coeff(i));
        }
        return r;
    }

    auto sincos_() const {
        using Expr = expr_t<Derived>;

        Expr s_out, c_out;

        if (Approx) {
            sincos_approx_<true, true>(s_out, c_out);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                std::tie(s_out.coeff(i), c_out.coeff(i)) = sincos(derived().coeff(i));
        }

        return std::make_pair(s_out, c_out);
    }

    auto csc_() const { return rcp(sin(derived())); }
    auto sec_() const { return rcp(cos(derived())); }

    auto tan_() const {
        expr_t<Derived> r;
        if (Approx) {
            tancot_approx_<true>(r);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = tan(derived().coeff(i));
        }
        return r;
    }

    auto cot_() const {
        expr_t<Derived> r;
        if (Approx) {
            tancot_approx_<false>(r);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = cot(derived().coeff(i));
        }
        return r;
    }

    auto asin_() const {
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;

        Expr r;
        if (Approx) {
            /*
               Arc sine function approximation based on CEPHES.

             - asin (in [-1, 1]):
               * avg abs. err = 2.25422e-08
               * avg rel. err = 2.85777e-08
                  -> in ULPs  = 0.331032
               * max abs. err = 1.19209e-07
                 (at x=-0.999998)
               * max rel. err = 2.27663e-07
                 -> in ULPs   = 2
                 (at x=-0.841416)
            */

            Expr x_          = derived(),
                 xa          = abs(x_),
                 x2          = x_*x_;

            constexpr bool Single = std::is_same<Scalar, float>::value;

            if (Single) {
                Mask mask_big = xa > Scalar(0.5);

                Expr x1 = Scalar(0.5) * (Scalar(1) - xa);
                Expr x3 = select(mask_big, x1, x2);
                Expr x4 = select(mask_big, sqrt(x1), xa);

                Expr z1 = poly4(x3, 1.6666752422e-1f,
                                    7.4953002686e-2f,
                                    4.5470025998e-2f,
                                    2.4181311049e-2f,
                                    4.2163199048e-2f);

                z1 = fmadd(z1, x3*x4, x4);

                r = select(mask_big, Scalar(M_PI_2) - (z1 + z1), z1);
            } else {
                Mask mask_big = xa > Scalar(0.625);

                if (any_nested(mask_big)) {
                    const Scalar pio4 = Scalar(0.78539816339744830962);
                    const Scalar more_bits = Scalar(6.123233995736765886130e-17);

                    /* arcsin(1-x) = pi/2 - sqrt(2x)(1+R(x))  */
                    Expr zz = Scalar(1) - xa;
                    Expr p = poly4(zz, 2.853665548261061424989e1,
                                      -2.556901049652824852289e1,
                                       6.968710824104713396794e0,
                                      -5.634242780008963776856e-1,
                                       2.967721961301243206100e-3) /
                             poly4(zz, 3.424398657913078477438e2,
                                      -3.838770957603691357202e2,
                                       1.470656354026814941758e2,
                                      -2.194779531642920639778e1,
                                       1.000000000000000000000e0) * zz;
                    zz = sqrt(zz + zz);
                    Expr z = pio4 - zz;
                    r[mask_big] = z - fmsub(zz, p, more_bits) + pio4;
                }

                if (!all_nested(mask_big)) {
                    Expr z = poly5(x2, -8.198089802484824371615e0,
                                        1.956261983317594739197e1,
                                       -1.626247967210700244449e1,
                                        5.444622390564711410273e0,
                                       -6.019598008014123785661e-1,
                                        4.253011369004428248960e-3) /
                             poly5(x2, -4.918853881490881290097e1,
                                        1.395105614657485689735e2,
                                       -1.471791292232726029859e2,
                                        7.049610280856842141659e1,
                                       -1.474091372988853791896e1,
                                        1.000000000000000000000e0) * x2;
                    z = fmadd(xa, z, xa);
                    z = select(xa < Scalar(1e-8), xa, z);
                    r[~mask_big] = z;
                }
            }
            r = copysign(r, x_);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = asin(derived().coeff(i));
        }
        return r;
    }

    auto acos_() const {
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;
        constexpr bool Single = std::is_same<Scalar, float>::value;

        Expr r;
        if (Approx) {
            /*
               Arc cosine function approximation based on CEPHES.

             - acos (in [-1, 1]):
               * avg abs. err = 4.72002e-08
               * avg rel. err = 2.85612e-08
                  -> in ULPs  = 0.33034
               * max abs. err = 2.38419e-07
                 (at x=-0.99999)
               * max rel. err = 1.19209e-07
                 -> in ULPs   = 1
                 (at x=-0.99999)
            */

            Expr x = derived();

            if (Single) {
                Expr xa         = abs(x),
                     x2         = x*x;

                Mask mask_big = xa > Scalar(0.5);

                Expr x1 = Scalar(0.5) * (Scalar(1) - xa);
                Expr x3 = select(mask_big, x1, x2);
                Expr x4 = select(mask_big, sqrt(x1), xa);

                Expr z1 = poly4(x3, 1.6666752422e-1f,
                                    7.4953002686e-2f,
                                    4.5470025998e-2f,
                                    2.4181311049e-2f,
                                    4.2163199048e-2f);

                z1 = fmadd(z1, x3*x4, x4);
                Expr z2 = z1 + z1;
                z2 = select(x < Scalar(0), Scalar(M_PI) - z2, z2);

                Expr z3 = float(M_PI_2) - copysign(z1, x);
                r = select(mask_big , z2, z3);
            } else {
                const Scalar pio4 = Scalar(0.78539816339744830962);
                const Scalar more_bits = Scalar(6.123233995736765886130e-17);
                const Scalar h = Scalar(0.5);

                auto mask = x > h;

                Expr y = asin(select(mask, sqrt(fnmadd(h, x, h)), x));
                r = select(mask, y + y, pio4 - y + more_bits + pio4);
            }
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = acos(derived().coeff(i));
        }
        return r;
    }

    auto atan2_(const Derived &x) const {
        using Expr = expr_t<Derived>;
        constexpr bool Single = std::is_same<Scalar, float>::value;

        Expr r;
        if (Approx) {
            /*
               MiniMax fit by Wenzel Jakob, May 2016

             - atan2() tested via atan() (in [-1, 1]):
               * avg abs. err = 1.81543e-07
               * avg rel. err = 4.15224e-07
                  -> in ULPs  = 4.9197
               * max abs. err = 5.96046e-07
                 (at x=-0.976062)
               * max rel. err = 7.73931e-07
                 -> in ULPs   = 12
                 (at x=-0.015445)
            */
            Expr y          = derived(),
                 abs_x      = abs(x),
                 abs_y      = abs(y),
                 min_val    = min(abs_y, abs_x),
                 max_val    = max(abs_x, abs_y),
                 scale      = Scalar(1) / max_val,
                 scaled_min = min_val * scale,
                 z          = scaled_min * scaled_min;

            // How to find these:
            // f[x_] = MiniMaxApproximation[ArcTan[Sqrt[x]]/Sqrt[x],
            //         {x, {1/10000, 1}, 6, 0}, WorkingPrecision->20][[2, 1]]

            Expr t;
            if (Single) {
                t = poly6(z, 0.99999934166683966009,
                            -0.33326497518773606976,
                            +0.19881342388439013552,
                            -0.13486708938456973185,
                            +0.083863120428809689910,
                            -0.037006525670417265220,
                             0.0078613793713198150252);
            } else {
                t = poly6(z, 9.9999999999999999419e-1,
                             2.50554429737833465113e0,
                             2.28289058385464073556e0,
                             9.20960512187107069075e-1,
                             1.59189681028889623410e-1,
                             9.35911604785115940726e-3,
                             8.07005540507283419124e-5) /
                    poly6(z, 1.00000000000000000000e0,
                             2.83887763071166519407e0,
                             3.02918312742541450749e0,
                             1.50576983803701596773e0,
                             3.49719171130492192607e-1,
                             3.29968942624402204199e-2,
                             8.26619391703564168942e-4);
            }

            t = t * scaled_min;

            t = select(abs_y > abs_x, Scalar(M_PI_2) - t, t);
            t = select(x < zero<Expr>(), Scalar(M_PI) - t, t);
            r = select(y < zero<Expr>(), -t, t);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = atan2(derived().coeff(i), x.coeff(i));
        }
        return r;
    }

    auto atan_() const {
        expr_t<Derived> r;

        if (Approx) {
            r = atan2(derived(), Scalar(1));
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = atan(derived().coeff(i));
        }
        return r;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Exponential function, logarithm, power
    // -----------------------------------------------------------------------

    auto exp_() const {
        using Expr = expr_t<Derived>;
        constexpr bool Single = std::is_same<Scalar, float>::value;

        Expr r;
        if (Approx) {
            /* Exponential function approximation based on CEPHES

               Redistributed under a BSD license with permission of the author, see
               https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

             - exp (in [-20, 30]):
               * avg abs. err = 7155.01
               * avg rel. err = 2.35929e-08
                  -> in ULPs  = 0.273524
               * max abs. err = 1.04858e+06
                 (at x=29.8057)
               * max rel. err = 1.192e-07
                 -> in ULPs   = 1
                 (at x=-19.9999)
            */

            const Expr inf(std::numeric_limits<Scalar>::infinity());
            const Expr max_range(Scalar(Single ? +88.3762588501 : +7.0943613930310391424428e2));
            const Expr min_range(Scalar(Single ? -88.3762588501 : -7.0943613930310391424428e2));

            Expr x(derived());

            auto mask_overflow  = x > max_range,
                 mask_underflow = x < min_range;

            /* Express e^x = e^g 2^n
                 = e^g e^(n loge(2))
                 = e^(g + n loge(2))
            */
            Expr n = floor(fmadd(Scalar(1.4426950408889634073599), x, Scalar(0.5)));
            if (Single) {
                x = fnmadd(n, Scalar(0.693359375), x);
                x = fnmadd(n, Scalar(-2.12194440e-4), x);
            } else {
                x = fnmadd(n, Scalar(6.93145751953125e-1), x);
                x = fnmadd(n, Scalar(1.42860682030941723212e-6), x);
            }

            Expr z = x*x;

            if (Single) {
                z = poly5(x, 5.0000001201e-1, 1.6666665459e-1,
                             4.1665795894e-2, 8.3334519073e-3,
                             1.3981999507e-3, 1.9875691500e-4);
                z = fmadd(z, x * x, x + Scalar(1));
            } else {
                /* Rational approximation for exponential
                   of the fractional part:
                      e^x = 1 + 2x P(x^2) / (Q(x^2) - P(x^2))
                 */
                Expr p = poly2(z, 9.99999999999999999910e-1,
                                  3.02994407707441961300e-2,
                                  1.26177193074810590878e-4) * x;

                Expr q = poly3(z, 2.00000000000000000009e0,
                                  2.27265548208155028766e-1,
                                  2.52448340349684104192e-3,
                                  3.00198505138664455042e-6);

                Expr pq = p / (q-p);
                z = pq + pq + Scalar(1);
            }

            r = select(mask_overflow, inf,
                       select(mask_underflow, zero<Expr>(), ldexp(z, n)));
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = exp(derived().coeff(i));
        }
        return r;
    }

    auto log_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;

        Expr r;
        if (Approx) {
            /* Logarithm function approximation based on CEPHES

               Redistributed under a BSD license with permission of the author, see
               https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

             - log (in [1e-20, 1000]):
               * avg abs. err = 8.8672e-09
               * avg rel. err = 1.57541e-09
                  -> in ULPs  = 0.020038
               * max abs. err = 4.76837e-07
                 (at x=54.7661)
               * max rel. err = 1.19194e-07
                 -> in ULPs   = 1
                 (at x=0.021)
            */
            using UInt = scalar_t<int_array_t<Expr>>;

            const Expr inf(std::numeric_limits<Scalar>::infinity());

            Expr x(derived());

            /* Catch negative and NaN values */
            auto valid_mask = x >= Scalar(0);

            /* Cut off denormalized values (our frexp does not handle them) */
            if (std::is_same<Scalar, float>::value)
                x = max(x, memcpy_cast<Scalar>(UInt(0x00800000u)));
            else
                x = max(x, memcpy_cast<Scalar>(UInt(0x0010000000000000ull)));

            Expr e;
            std::tie(x, e) = frexp(x);

            if (Single) {
                auto lt_inv_sqrt2 = x < Scalar(0.707106781186547524);

                e -= Expr(Scalar(1.f)) & lt_inv_sqrt2;
                x += (x & lt_inv_sqrt2) - Scalar(1);

                Expr z = x * x;
                Expr y = poly8(x, 3.3333331174e-1, -2.4999993993e-1,
                                  2.0000714765e-1, -1.6668057665e-1,
                                  1.4249322787e-1, -1.2420140846e-1,
                                  1.1676998740e-1, -1.1514610310e-1,
                                  7.0376836292e-2);

                y *= x * z;

                y = fmadd(e, Scalar(-2.12194440e-4), y);
                z = fmadd(z, Scalar(-0.5), x + y);
                r = fmadd(e, Scalar(0.693359375), z);
            } else {
                const Scalar half = Scalar(0.5),
                             sqrt_half = Scalar(0.70710678118654752440);

                auto mask_big = abs(e) > Scalar(2);
                auto mask1 = x < sqrt_half;

                e[mask1] -= Scalar(1);

                Expr r_big, r_small;

                if (any_nested(mask_big)) {
                    /* logarithm using log(x) = z + z**3 P(z)/Q(z), where z = 2(x-1)/x+1) */

                    Expr z = x - half;

                    z[~mask1] -= half;

                    Expr y = half * select(mask1, z, x) + half;
                    Expr x2 = z / y;

                    z = x2 * x2;
                    z = x2 * (z * poly2(z, -6.41409952958715622951e1,
                                            1.63866645699558079767e1,
                                           -7.89580278884799154124e-1) /
                                  poly3(z, -7.69691943550460008604e2,
                                            3.12093766372244180303e2,
                                           -3.56722798256324312549e1,
                                            1.00000000000000000000e0));

                    r_big = fnmadd(e, Scalar(2.121944400546905827679e-4), z) + x2;
                }

                if (!all_nested(mask_big)) {
                    /* logarithm using log(1+x) = x - .5x**2 + x**3 P(x)/Q(x) */

                    Expr x2 = select(mask1, x + x, x) - Scalar(1);

                    Expr z = x2*x2;
                    Expr y = x2 * (z * poly5(x2, 7.70838733755885391666e0,
                                                1.79368678507819816313e1,
                                                1.44989225341610930846e1,
                                                4.70579119878881725854e0,
                                                4.97494994976747001425e-1,
                                                1.01875663804580931796e-4) /
                                       poly5(x2, 2.31251620126765340583e1,
                                                7.11544750618563894466e1,
                                                8.29875266912776603211e1,
                                                4.52279145837532221105e1,
                                                1.12873587189167450590e1,
                                                1.00000000000000000000e0));

                    y = fnmadd(e, Scalar(2.121944400546905827679e-4), y);

                    r_small = x2 + fnmadd(half, z, y);
                }

                r = select(mask_big, r_big, r_small);
                r = fmadd(e, Scalar(0.693359375), r);
            }
            r = select(eq(derived(), inf), inf, r | ~valid_mask);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = log(derived().coeff(i));
        }
        return r;
    }

    /// Multiply by integer power of 2
    auto ldexp_(const Derived &n) const {
        using Expr = expr_t<Derived>;

        Expr r;
        if (Approx) {
            constexpr bool Single = std::is_same<Scalar, float>::value;
            r = derived() * reinterpret_array<Expr>(sli<Single ? 23 : 52>(int_array_t<Expr>(n) + (Single ? 0x7f : 0x3ff)));
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = ldexp(derived().coeff(i), n.coeff(i));
        }
        return r;
    }

    /// Break floating-point number into normalized fraction and power of 2
    auto frexp_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        Expr result_m, result_e;

        /// Caveat: does not handle denormals correctly
        if (Approx) {
            using IntArray = int_array_t<Expr>;
            using Int = scalar_t<IntArray>;
            using IntMask = mask_t<IntArray>;

            const IntArray
                exponent_mask(Int(Single ? 0x7f800000ull : 0x7ff0000000000000ull)),
                mantissa_sign_mask(Int(Single ? ~0x7f800000ull : ~0x7ff0000000000000ull)),
                bias_minus_1(Int(Single ? 0x7e : 0x3fe));

            IntArray x = reinterpret_array<IntArray>(derived());
            IntArray exponent_bits = x & exponent_mask;

            /* Detect zero/inf/NaN */
            IntMask is_normal =
                IntMask(neq(derived(), zero<Expr>())) &
                neq(exponent_bits, exponent_mask);

            IntArray exponent_i = (sri<Single ? 23 : 52>(exponent_bits)) - bias_minus_1;

            IntArray mantissa = (x & mantissa_sign_mask) |
                                IntArray(memcpy_cast<Int>(Scalar(.5f)));

            result_e = Expr(exponent_i & is_normal);
            result_m = reinterpret_array<Expr>(select(is_normal, mantissa, x));
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                std::tie(result_m.coeff(i), result_e.coeff(i)) = frexp(derived().coeff(i));
        }
        return std::make_pair(result_m, result_e);
    }

    auto pow_(const Derived &y) const {
        using Expr = expr_t<Derived>;

        Expr r;
        if (Approx) {
            r = exp(log(derived()) * y);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = pow(derived().coeff(i), y.coeff(i));
        }
        return r;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Hyperbolic and inverse hyperbolic functions
    // -----------------------------------------------------------------------

    auto sinh_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;

        Expr r;

        if (Approx) {
            /*
             - sinh (in [-10, 10]):
               * avg abs. err = 2.92524e-05
               * avg rel. err = 2.80831e-08
                  -> in ULPs  = 0.336485
               * max abs. err = 0.00195312
                 (at x=-9.99894)
               * max rel. err = 2.36862e-07
                 -> in ULPs   = 3
                 (at x=-9.69866)
            */
            Expr x  = derived(),
                 xa = abs(x),
                 r_small, r_big;

            Mask mask_big = xa > Scalar(1);

            if (any_nested(mask_big)) {
                Expr exp0 = exp(x),
                     exp1 = rcp(exp0);

                r_big = (exp0 - exp1) * Scalar(0.5);
            }

            if (!all_nested(mask_big)) {
                Expr x2 = x * x;

                if (Single) {
                    r_small = fmadd(poly2(x2, 1.66667160211e-1,
                                              8.33028376239e-3,
                                              2.03721912945e-4),
                                    x2 * x, x);
                } else {
                    r_small = fmadd(poly3(x2, -3.51754964808151394800e5,
                                              -1.15614435765005216044e4,
                                              -1.63725857525983828727e2,
                                              -7.89474443963537015605e-1) /
                                    poly3(x2, -2.11052978884890840399e6,
                                               3.61578279834431989373e4,
                                              -2.77711081420602794433e2,
                                               1.00000000000000000000e0),
                                    x2 * x, x);
                }
            }

            r = select(mask_big, r_big, r_small);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = sinh(derived().coeff(i));
        }
        return r;
    }

    auto cosh_() const {
        using Expr = expr_t<Derived>;

        Expr r;
        if (Approx) {
            /*
             - cosh (in [-10, 10]):
               * avg abs. err = 4.17738e-05
               * avg rel. err = 3.15608e-08
                  -> in ULPs  = 0.376252
               * max abs. err = 0.00195312
                 (at x=-9.99894)
               * max rel. err = 2.38001e-07
                 -> in ULPs   = 3
                 (at x=-9.70164)
            */

            Expr exp0 = exp(derived()),
                 exp1 = rcp(exp0);

            r = (exp0 + exp1) * Scalar(.5f);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = cosh(derived().coeff(i));
        }
        return r;
    }

    auto sincosh_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;

        Expr s_out, c_out;

        if (Approx) {
            /*
             - sinh (in [-10, 10]):
               * avg abs. err = 2.92524e-05
               * avg rel. err = 2.80831e-08
                  -> in ULPs  = 0.336485
               * max abs. err = 0.00195312
                 (at x=-9.99894)
               * max rel. err = 2.36862e-07
                 -> in ULPs   = 3
                 (at x=-9.69866)

             - cosh (in [-10, 10]):
               * avg abs. err = 4.17738e-05
               * avg rel. err = 3.15608e-08
                  -> in ULPs  = 0.376252
               * max abs. err = 0.00195312
                 (at x=-9.99894)
               * max rel. err = 2.38001e-07
                 -> in ULPs   = 3
                 (at x=-9.70164)
            */

            const Scalar half = Scalar(0.5);

            Expr x     = derived(),
                 xa    = abs(x),
                 exp0  = exp(x),
                 exp1  = rcp(exp0),
                 r_big = (exp0 - exp1) * half,
                 r_small;

            Mask mask_big = xa > Scalar(1);

            if (!all_nested(mask_big)) {
                Expr x2 = x * x;

                if (Single) {
                    r_small = fmadd(poly2(x2, 1.66667160211e-1,
                                              8.33028376239e-3,
                                              2.03721912945e-4),
                                    x2 * x, x);
                } else {
                    r_small = fmadd(poly3(x2, -3.51754964808151394800e5,
                                              -1.15614435765005216044e4,
                                              -1.63725857525983828727e2,
                                              -7.89474443963537015605e-1) /
                                    poly3(x2, -2.11052978884890840399e6,
                                               3.61578279834431989373e4,
                                              -2.77711081420602794433e2,
                                               1.00000000000000000000e0),
                                    x2 * x, x);
                }
            }

            s_out = select(mask_big, r_big, r_small);
            c_out = half * (exp0 + exp1);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                std::tie(s_out.coeff(i), c_out.coeff(i)) = sincosh(derived().coeff(i));
        }

        return std::make_pair(s_out, c_out);
    }

    auto tanh_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;

        Expr r;
        if (Approx) {
            /*
               Hyperbolic tangent function approximation based on CEPHES.

             - tanh (in [-10, 10]):
               * avg abs. err = 4.44655e-08
               * avg rel. err = 4.58074e-08
                  -> in ULPs  = 0.698044
               * max abs. err = 3.57628e-07
                 (at x=-2.12867)
               * max rel. err = 4.1006e-07
                 -> in ULPs   = 6
                 (at x=-2.12867)
            */

            Expr x = derived(),
                 r_big, r_small;

            Mask mask_big = abs(x) >= Scalar(0.625);

            if (!all_nested(mask_big)) {
                Expr x2 = x*x;

                if (Single) {
                    r_small = poly4(x2, -3.33332819422e-1,
                                         1.33314422036e-1,
                                        -5.37397155531e-2,
                                         2.06390887954e-2,
                                        -5.70498872745e-3);
                } else {
                    r_small = poly2(x2, -1.61468768441708447952e3,
                                        -9.92877231001918586564e1,
                                        -9.64399179425052238628e-1) /
                              poly3(x2,  4.84406305325125486048e3,
                                         2.23548839060100448583e3,
                                         1.12811678491632931402e2,
                                         1.00000000000000000000e0);
                }

                r_small = fmadd(r_small, x2 * x, x);
            }

            if (any_nested(mask_big)) {
                Expr e  = exp(x + x),
                     e2 = rcp(e + Scalar(1));
                r_big = Scalar(1) - (e2 + e2);
            }

            r = select(mask_big, r_big, r_small);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = tanh(derived().coeff(i));
        }
        return r;
    }

    auto csch_() const { return rcp(sinh(derived())); }

    auto sech_() const {
        using Expr = expr_t<Derived>;

        Expr r;
        if (Approx) {
            Expr exp0 = exp(derived()),
                 exp1 = rcp(exp0);

            r = rcp(exp0 + exp1);
            r = r + r;
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = sech(derived().coeff(i));
        }
        return r;
    }

    auto coth_() const { return rcp(tanh(derived())); }

    auto asinh_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;

        Expr r;
        if (Approx) {
            /*
               Hyperbolic arc sine function approximation based on CEPHES.

             - asinh (in [-10, 10]):
               * avg abs. err = 2.75626e-08
               * avg rel. err = 1.51762e-08
                  -> in ULPs  = 0.178341
               * max abs. err = 2.38419e-07
                 (at x=-10)
               * max rel. err = 1.71857e-07
                 -> in ULPs   = 2
                 (at x=-1.17457)
            */

            Expr x   = derived(),
                 x2 = x*x,
                 xa = abs(x),
                 r_big, r_small;

            Mask mask_big  = xa >= Scalar(Single ? 0.51 : 0.533),
                 mask_huge = xa >= Scalar(Single ? 1e10 : 1e20);

            if (!all_nested(mask_big)) {
                if (Single) {
                    r_small = poly3(x2, -1.6666288134e-1,
                                         7.4847586088e-2,
                                        -4.2699340972e-2,
                                         2.0122003309e-2);
                } else {
                    r_small = poly4(x2, -5.56682227230859640450e0,
                                        -9.09030533308377316566e0,
                                        -4.37390226194356683570e0,
                                        -5.91750212056387121207e-1,
                                        -4.33231683752342103572e-3) /
                              poly4(x2, 3.34009336338516356383e1,
                                        6.95722521337257608734e1,
                                        4.86042483805291788324e1,
                                        1.28757002067426453537e1,
                                        1.00000000000000000000e0);
                }
                r_small = fmadd(r_small, x2 * x, x);
            }

            if (any_nested(mask_big)) {
                r_big = log(xa + (sqrt(x2 + Scalar(1)) & ~mask_huge));
                r_big[mask_huge] += Scalar(M_LN2);
                r_big = copysign(r_big, x);
            }

            r = select(mask_big, r_big, r_small);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = asinh(derived().coeff(i));
        }
        return r;
    }

    auto acosh_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Derived>;

        Expr r;
        if (Approx) {
            /*
               Hyperbolic arc cosine function approximation based on CEPHES.

             - acosh (in [-10, 10]):
               * avg abs. err = 2.8897e-08
               * avg rel. err = 1.49658e-08
                  -> in ULPs  = 0.175817
               * max abs. err = 2.38419e-07
                 (at x=3.76221)
               * max rel. err = 2.35024e-07
                 -> in ULPs   = 3
                 (at x=1.02974)
            */

            Expr x  = derived(),
                 x1 = x - Scalar(1),
                 r_big, r_small;

            Mask mask_big  = x1 >= Scalar(0.49),
                 mask_huge = x1 >= Scalar(1e10);

            if (!all_nested(mask_big)) {
                if (Single) {
                    r_small = poly4(x1,  1.4142135263e+0,
                                        -1.1784741703e-1,
                                         2.6454905019e-2,
                                        -7.5272886713e-3,
                                         1.7596881071e-3);
                } else {
                    r_small = poly4(x1, 1.10855947270161294369E5,
                                        1.08102874834699867335E5,
                                        3.43989375926195455866E4,
                                        3.94726656571334401102E3,
                                        1.18801130533544501356E2) /
                              poly5(x1, 7.83869920495893927727E4,
                                        8.29725251988426222434E4,
                                        2.97683430363289370382E4,
                                        4.15352677227719831579E3,
                                        1.86145380837903397292E2,
                                        1.00000000000000000000E0);
                }

                r_small *= sqrt(x1);
                r_small |= x1 < zero<Expr>();
            }

            if (any_nested(mask_big)) {
                r_big = log(x + (sqrt(fmsub(x, x, Scalar(1))) & ~mask_huge));
                r_big[mask_huge] += Scalar(M_LN2);
            }

            r = select(mask_big, r_big, r_small);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = acosh(derived().coeff(i));
        }
        return r;
    }

    auto atanh_() const {
        constexpr bool Single = std::is_same<Scalar, float>::value;
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;

        Expr r;
        if (Approx) {
            /*
               Hyperbolic arc tangent function approximation based on CEPHES.


             - acosh (in [-10, 10]):
               * avg abs. err = 9.87529e-09
               * avg rel. err = 1.52741e-08
                  -> in ULPs  = 0.183879
               * max abs. err = 2.38419e-07
                 (at x=-0.998962)
               * max rel. err = 1.19209e-07
                 -> in ULPs   = 1
                 (at x=-0.998962)
            */

            Expr x  = derived(),
                 xa = abs(x),
                 r_big, r_small;

            Mask mask_big  = xa >= Scalar(0.5);

            if (!all_nested(mask_big)) {
                Expr x2 = x*x;
                if (Single) {
                    r_small = poly4(x2, 3.33337300303e-1,
                                        1.99782164500e-1,
                                        1.46691431730e-1,
                                        8.24370301058e-2,
                                        1.81740078349e-1);
                } else {
                    r_small = poly4(x2, -3.09092539379866942570e1,
                                         6.54566728676544377376e1,
                                        -4.61252884198732692637e1,
                                         1.20426861384072379242e1,
                                        -8.54074331929669305196e-1) /
                              poly5(x2, -9.27277618139601130017e1,
                                         2.52006675691344555838e2,
                                        -2.49839401325893582852e2,
                                         1.08938092147140262656e2,
                                        -1.95638849376911654834e1,
                                         1.00000000000000000000e0);
                }
                r_small = fmadd(r_small, x2*x, x);
            }

            if (any_nested(mask_big)) {
                r_big = log((Scalar(1) + xa) / (Scalar(1) - xa)) * Scalar(0.5);
                r_big = copysign(r_big, x);
            }

            r = select(mask_big, r_big, r_small);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = atanh(derived().coeff(i));
        }
        return r;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Bit counting fallback implementations
    // -----------------------------------------------------------------------

    Derived popcnt_() const {
        if (sizeof(Scalar) <= 4) {
            using UInt = uint32_array_t<expr_t<Derived>>;
            UInt w = (UInt) derived();

            w -= sri<1>(w) & 0x55555555u;
            w = (w & 0x33333333u) + ((sri<2>(w)) & 0x33333333u);
            w = (w + sri<4>(w)) & 0x0F0F0F0Fu;
            w = sri<24>(w * 0x01010101u);

            return Derived(w);
        } else {
            using UInt = uint64_array_t<expr_t<Derived>>;
            UInt w = (UInt) derived();

            w -= sri<1>(w) & 0x5555555555555555ull;
            w = (w & 0x3333333333333333ull) + (sri<2>(w) & 0x3333333333333333ull);
            w = (w + sri<4>(w)) & 0x0F0F0F0F0F0F0F0Full;
            w = sri<56>(w * 0x0101010101010101ull);

            return Derived(w);
        }
    }

    Derived lzcnt_() const {
        using UInt = uint_array_t<expr_t<Derived>>;
        UInt w = (UInt) derived();
        w |= sri<1>(w);
        w |= sri<2>(w);
        w |= sri<4>(w);
        w |= sri<8>(w);
        w |= sri<16>(w);
        if (sizeof(Scalar) > 4)
            w |= sri<32>(w);
        return popcnt(~w);
    }

    Derived tzcnt_() const {
        using UInt = uint_array_t<expr_t<Derived>>;
        UInt w = (UInt) derived();
        w |= sli<1>(w);
        w |= sli<2>(w);
        w |= sli<4>(w);
        w |= sli<8>(w);
        w |= sli<16>(w);
        if (sizeof(Scalar) > 4)
            w |= sli<32>(w);
        return popcnt(~w);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Component access
    // -----------------------------------------------------------------------

    ENOKI_INLINE const Value &x() const {
        static_assert(Derived::ActualSize >= 1,
                      "StaticArrayBase::x(): requires Size >= 1");
        return derived().coeff(0);
    }

    ENOKI_INLINE Value& x() {
        static_assert(Derived::ActualSize >= 1,
                      "StaticArrayBase::x(): requires Size >= 1");
        return derived().coeff(0);
    }

    ENOKI_INLINE const Value &y() const {
        static_assert(Derived::ActualSize >= 2,
                      "StaticArrayBase::y(): requires Size >= 2");
        return derived().coeff(1);
    }

    ENOKI_INLINE Value& y() {
        static_assert(Derived::ActualSize >= 2,
                      "StaticArrayBase::y(): requires Size >= 2");
        return derived().coeff(1);
    }

    ENOKI_INLINE const Value& z() const {
        static_assert(Derived::ActualSize >= 3,
                      "StaticArrayBase::z(): requires Size >= 3");
        return derived().coeff(2);
    }

    ENOKI_INLINE Value& z() {
        static_assert(Derived::ActualSize >= 3,
                      "StaticArrayBase::z(): requires Size >= 3");
        return derived().coeff(2);
    }

    ENOKI_INLINE const Value& w() const {
        static_assert(Derived::ActualSize >= 4,
                      "StaticArrayBase::w(): requires Size >= 4");
        return derived().coeff(3);
    }

    ENOKI_INLINE Value& w() {
        static_assert(Derived::ActualSize >= 4,
                      "StaticArrayBase::w(): requires Size >= 4");
        return derived().coeff(3);
    }

    ENOKI_INLINE Value *data() { return &derived().coeff(0); }
    ENOKI_INLINE const Value *data() const { return &derived().coeff(0); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Other Methods
    // -----------------------------------------------------------------------

    /// Return the array size
    constexpr size_t size() const { return Derived::Size; }

    //! @}
    // -----------------------------------------------------------------------

};

NAMESPACE_BEGIN(detail)

template <typename Array, size_t N, typename... Indices,
          std::enable_if_t<sizeof...(Indices) == N && !Array::Derived::IsMask, int> = 0>
std::ostream &print(std::ostream &os, const Array &a,
                    const std::array<size_t, N> &, Indices... indices) {
    os << a.derived().coeff(indices...);
    return os;
}

template <typename Array, size_t N, typename... Indices,
          std::enable_if_t<sizeof...(Indices) == N && Array::Derived::IsMask, int> = 0>
std::ostream &print(std::ostream &os, const Array &a,
                    const std::array<size_t, N> &, Indices... indices) {
    os << mask_active(a.derived().coeff(indices...));
    return os;
}

template <typename Array, size_t N, typename... Indices,
          std::enable_if_t<sizeof...(Indices) != N, int> = 0>
std::ostream &print(std::ostream &os, const Array &a,
                    const std::array<size_t, N> &size,
                    Indices... indices) {
    constexpr size_t k = N - sizeof...(Indices) - 1;
    os << "[";
    for (size_t i = 0; i < size[k]; ++i) {
        print(os, a, size, i, indices...);
        if (i + 1 < size[k]) {
            if (k == 0) {
                os << ", ";
            } else {
                os << ",\n";
                for (size_t j = 0; j <= sizeof...(Indices); ++j)
                    os << " ";
            }
        }
    }
    os << "]";
    return os;
}

NAMESPACE_END(detail)

template <typename Value, typename Derived>
ENOKI_NOINLINE std::ostream &operator<<(std::ostream &os,
                                        const ArrayBase<Value, Derived> &a) {
    return detail::print(os, a, shape(a.derived()));
}

/// Macro to initialize uninitialized floating point arrays with NaNs in debug mode
#if defined(NDEBUG)
#define ENOKI_TRIVIAL_CONSTRUCTOR(Type_)                                       \
    template <typename T = Type_,                                              \
         std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>   \
    ENOKI_INLINE StaticArrayImpl() { }
#else
#define ENOKI_TRIVIAL_CONSTRUCTOR(Type_)                                       \
    template <typename T = Type_,                                              \
         std::enable_if_t<std::is_floating_point<T>::value &&                  \
                          std::is_default_constructible<T>::value, int> = 0>   \
    ENOKI_INLINE StaticArrayImpl()                                             \
        : StaticArrayImpl(std::numeric_limits<scalar_t<T>>::quiet_NaN()) { }   \
    template <typename T = Type_,                                              \
         std::enable_if_t<!std::is_floating_point<T>::value &&                 \
                          std::is_default_constructible<T>::value, int> = 0>   \
    ENOKI_INLINE StaticArrayImpl() { }
#endif

/// SFINAE macro for constructors that convert from another type
#define ENOKI_CONVERT(Type)                                                    \
    template <typename Type2, bool Approx2, RoundingMode Mode2,                \
              typename Derived2,                                               \
              std::enable_if_t<detail::is_same<Type2, Type>::value, int> = 0>  \
    ENOKI_INLINE StaticArrayImpl(                                              \
        const StaticArrayBase<Type2, Size, Approx2, Mode2, Derived2> &a)

/// SFINAE macro for constructors that reinterpret another type
#define ENOKI_REINTERPRET(Type)                                                \
    template <typename Type2, bool Approx2, RoundingMode Mode2,                \
              typename Derived2,                                               \
              std::enable_if_t<detail::is_same<Type2, Type>::value, int> = 0>  \
    ENOKI_INLINE StaticArrayImpl(                                              \
        const StaticArrayBase<Type2, Size, Approx2, Mode2, Derived2> &a,       \
        detail::reinterpret_flag)

/// SFINAE macro for constructors that reinterpret another type (K mask registers)
#define ENOKI_REINTERPRET_KMASK(Type, Size)                                    \
    template <typename Type2, bool Approx2, RoundingMode Mode2,                \
              typename Derived2,                                               \
              std::enable_if_t<detail::is_same<Type2, Type>::value, int> = 0>  \
    ENOKI_INLINE KMask(                                                        \
        const StaticArrayBase<Type2, Size, Approx2, Mode2, Derived2> &a,       \
        detail::reinterpret_flag)

/// SFINAE macro for strided operations (scatter, gather)
#define ENOKI_REQUIRE_INDEX(T, Index)                                          \
    template <                                                                 \
        size_t Stride, typename T, typename Mask_ = Mask,                      \
        std::enable_if_t<std::is_integral<typename T::Value>::value &&         \
                         sizeof(typename T::Value) == sizeof(Index), int> = 0>

/// SFINAE macro for strided operations (prefetch)
#define ENOKI_REQUIRE_INDEX_PF(T, Index)                                       \
    template <                                                                 \
        size_t Stride, bool Write, size_t Level, typename T,                   \
        std::enable_if_t<std::is_integral<typename T::Value>::value &&         \
                         sizeof(typename T::Value) == sizeof(Index), int> = 0>

/// SFINAE macro for strided operations (transform)
#define ENOKI_REQUIRE_INDEX_TRANSFORM(T, Index)                                \
    template <                                                                 \
        size_t Stride, typename T, typename Func,                              \
        std::enable_if_t<std::is_integral<typename T::Value>::value &&         \
                         sizeof(typename T::Value) == sizeof(Index), int> = 0, \
        typename... Args>

#define ENOKI_NATIVE_ARRAY(Value_, Size_, Approx_, Register, Mode)             \
    static constexpr bool IsNative = true;                                     \
    using Base = StaticArrayBase<Value_, Size_, Approx_, Mode, Derived>;       \
    using Arg = Derived;                                                       \
    using Expr = Derived;                                                      \
    using Base::operator=;                                                     \
    using typename Base::Value;                                                \
    using typename Base::Array1;                                               \
    using typename Base::Array2;                                               \
    using Base::Size;                                                          \
    using Base::ActualSize;                                                    \
    using Base::derived;                                                       \
    Register m;                                                                \
    ENOKI_TRIVIAL_CONSTRUCTOR(Value_)                                          \
    ENOKI_INLINE StaticArrayImpl(Register value) : m(value) { }                \
    StaticArrayImpl(const StaticArrayImpl &a) = default;                       \
    StaticArrayImpl &operator=(const StaticArrayImpl &a) = default;            \
    ENOKI_INLINE Value &coeff(size_t i) {                                      \
        union Data { Register value; Value data[Size_]; };                     \
        return ((Data *) this)->data[i];                                       \
    }                                                                          \
    ENOKI_INLINE const Value &coeff(size_t i) const {                          \
        union Data { Register value; Value data[Size_]; };                     \
        return ((const Data *) this)->data[i];                                 \
    }                                                                          \
    template <typename Type2, typename Derived2, typename T = Derived,         \
              std::enable_if_t<std::is_assignable<Value_ &, Type2>::value &&   \
                               Derived2::Size == T::Size, int> = 0>            \
    ENOKI_INLINE StaticArrayImpl(const ArrayBase<Type2, Derived2> &a) {        \
        ENOKI_TRACK_SCALAR for (size_t i = 0; i < Derived2::Size; ++i)         \
            derived().coeff(i) = Value(a.derived().coeff(i));                  \
    }

#define ENOKI_NATIVE_ARRAY_CLASSIC(Value_, Size_, Approx_, Register)           \
    ENOKI_NATIVE_ARRAY(Value_, Size_, Approx_, Register,                       \
                       RoundingMode::Default)                                  \
    using Mask =                                                               \
        detail::ArrayMask<Value_, Size_, Approx_, RoundingMode::Default>;

NAMESPACE_END(enoki)
