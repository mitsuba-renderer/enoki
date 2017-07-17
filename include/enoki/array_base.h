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
          std::is_same<scalar_t<T>, float>::value ? T2::Approx : detail::approx_default<T>::value,
          std::is_floating_point<scalar_t<T>>::value ? T2::Mode : RoundingMode::Default
    >;

    static_assert(std::is_same<Scalar, float>::value || !Approx,
                  "Approximate math library functions are only supported in "
                  "single precision mode!");

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
            return derived() ^ Expr(Scalar(-0.f));
        else
            return ~derived() + Expr(Scalar(1));
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

    auto sin_() const {
        using Expr = expr_t<Derived>;
        using IntArray = int_array_t<Expr>;
        using Int = scalar_t<IntArray>;

        Expr r;
        if (Approx) {
            /* Sine function approximation based on CEPHES.
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
            */

            Expr x = abs(derived());

            /* Scale by 4/Pi and get the integer part */
            IntArray j(x * Scalar(1.27323954473516));

            /* Map zeros to origin; if (j & 1) j += 1 */
            j = (j + Int(1)) & Int(~1u);

            /* Cast back to a floating point value */
            Expr y(j);

            /* Determine sign of result */
            Expr sign = detail::sign_mask(reinterpret_array<Expr>(sli<29>(j)) ^ derived());

            /* Extended precision modular arithmetic */
            x = x - y * Scalar(0.78515625)
                  - y * Scalar(2.4187564849853515625e-4)
                  - y * Scalar(3.77489497744594108e-8);

            Expr z = x * x;

            Expr s = poly2(z, -1.6666654611e-1,
                               8.3321608736e-3,
                              -1.9515295891e-4) * z;

            Expr c = poly2(z,  4.166664568298827e-2,
                              -1.388731625493765e-3,
                               2.443315711809948e-5) * z;

            s = fmadd(s, x, x);
            c = fmadd(c, z, fmadd(z, Expr(Scalar(-0.5)), Expr(Scalar(1))));

            auto polymask = mask_t<Expr>(
                eq(j & Int(2), zero<IntArray>()));

            r = select(polymask, s, c) ^ sign;
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = sin(derived().coeff(i));
        }
        return r;
    }

    auto cos_() const {
        using Expr = expr_t<Derived>;
        using IntArray = int_array_t<Expr>;
        using Int = scalar_t<IntArray>;

        Expr r;
        if (Approx) {
            /* Cosine function approximation based on CEPHES.
               Excellent accuracy in the domain |x| < 8192

               Redistributed under a BSD license with permission of the author, see
               https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

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
            IntArray j(x * Scalar(1.27323954473516));

            /* Map zeros to origin; if (j & 1) j += 1 */
            j = (j + Int(1)) & Int(~1u);

            /* Cast back to a floating point value */
            Expr y(j);

            /* Determine sign of result */
            Expr sign = detail::sign_mask(reinterpret_array<Expr>(sli<29>(~(j - Int(2)))));

            /* Extended precision modular arithmetic */
            x = x - y * Scalar(0.78515625)
                  - y * Scalar(2.4187564849853515625e-4)
                  - y * Scalar(3.77489497744594108e-8);

            Expr z = x * x;

            Expr s = poly2(z, -1.6666654611e-1,
                               8.3321608736e-3,
                              -1.9515295891e-4) * z;

            Expr c = poly2(z,  4.166664568298827e-2,
                              -1.388731625493765e-3,
                               2.443315711809948e-5) * z;

            s = fmadd(s, x, x);
            c = fmadd(c, z, fmadd(z, Expr(Scalar(-0.5)), Expr(Scalar(1))));


            auto polymask = mask_t<Expr>(
                eq(j & Int(2), zero<IntArray>()));

            r = select(polymask, c, s) ^ sign;
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = cos(derived().coeff(i));
        }
        return r;
    }

    auto sincos_() const {
        using Expr = expr_t<Derived>;
        using IntArray = int_array_t<Expr>;
        using Int = scalar_t<IntArray>;

        Expr s_out, c_out;

        if (Approx) {
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
            IntArray j(x * Scalar(1.27323954473516));

            /* Map zeros to origin; if (j & 1) j += 1 */
            j = (j + Int(1)) & Int(~1u);

            /* Cast back to a floating point value */
            Expr y(j);

            /* Determine sign of result */
            Expr sign_sin = detail::sign_mask(reinterpret_array<Expr>(sli<29>(j)) ^ derived());
            Expr sign_cos = detail::sign_mask(reinterpret_array<Expr>(sli<29>(~(j - Int(2)))));

            /* Extended precision modular arithmetic */
            x = x - y * Scalar(0.78515625)
                  - y * Scalar(2.4187564849853515625e-4)
                  - y * Scalar(3.77489497744594108e-8);

            Expr z = x * x;

            Expr s = poly2(z, -1.6666654611e-1,
                               8.3321608736e-3,
                              -1.9515295891e-4) * z;

            Expr c = poly2(z,  4.166664568298827e-2,
                              -1.388731625493765e-3,
                               2.443315711809948e-5) * z;

            s = fmadd(s, x, x);
            c = fmadd(c, z, fmadd(z, Expr(Scalar(-0.5)), Expr(Scalar(1))));

            auto polymask = mask_t<Expr>(
                eq(j & Int(2), zero<IntArray>()));

            s_out = select(polymask, s, c) ^ sign_sin;
            c_out = select(polymask, c, s) ^ sign_cos;
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i) {
                c_out.coeff(i) = cos(derived().coeff(i));
                s_out.coeff(i) = sin(derived().coeff(i));
            }
        }

        return std::make_pair(s_out, c_out);
    }

    auto tan_() const {
        using Expr = expr_t<Derived>;

        Expr r;
        if (Approx) {
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

            auto sc = sincos(derived());
            r = sc.first / sc.second;
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = tan(derived().coeff(i));
        }
        return r;
    }

    auto csc_() const { return rcp(sin(derived())); }

    auto sec_() const { return rcp(cos(derived())); }

    auto cot_() const {
        auto sc = sincos(derived());
        return sc.second / sc.first;
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

            r = copysign(select(mask_big, Scalar(M_PI_2) - (z1 + z1), z1), x_);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = asin(derived().coeff(i));
        }
        return r;
    }

    auto acos_() const {
        using Expr = expr_t<Derived>;
        using Mask = mask_t<Expr>;

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
            Expr x_          = derived(),
                 xa          = abs(x_),
                 x2          = x_*x_;

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
            z2 = select(x_ < Scalar(0), Scalar(M_PI) - z2, z2);

            Expr z3 = float(M_PI_2) - copysign(z1, x_);
            r = select(mask_big , z2, z3);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = acos(derived().coeff(i));
        }
        return r;
    }

    auto atan2_(const Derived &x) const {
        using Expr = expr_t<Derived>;

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

            Expr t = poly6(z, 0.99999934166683966009,
                             -0.33326497518773606976,
                             +0.19881342388439013552,
                             -0.13486708938456973185,
                             +0.083863120428809689910,
                             -0.037006525670417265220,
                              0.0078613793713198150252);

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
        using Expr = expr_t<Derived>;

        if (Approx) {
            return atan2(derived(), Expr(Scalar(1)));
        } else {
            Expr r;
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = atan(derived().coeff(i));
            return r;
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Exponential function, logarithm, power
    // -----------------------------------------------------------------------

    auto exp_() const {
        using Expr = expr_t<Derived>;

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
            const Expr max_range(Scalar(+88.3762626647949));
            const Expr min_range(Scalar(-88.3762626647949));

            Expr x(derived());

            auto mask_overflow = x > max_range;
            auto mask_underflow = x < min_range;

            /* Express e^x = e^g 2^n
                 = e^g e^(n loge(2))
                 = e^(g + n loge(2))
            */
            Expr n = floor(Scalar(1.44269504088896341) * x + Scalar(0.5));
            x -= n * Scalar(0.693359375);
            x -= n * Scalar(-2.12194440e-4);

            /* Rational approximation for exponential
               of the fractional part:
                  e^x = 1 + 2x P(x^2) / (Q(x^2) - P(x^2))
             */
            Expr z = poly5(x, 5.0000001201e-1, 1.6666665459e-1,
                              4.1665795894e-2, 8.3334519073e-3,
                              1.3981999507e-3, 1.9875691500e-4);

            z = fmadd(z, x*x, x + Scalar(1));

            r = select(mask_overflow, inf,
                       select(mask_underflow, zero<Expr>(), ldexp(z, n)));
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = exp(derived().coeff(i));
        }
        return r;
    }

    auto log_() const {
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
                x = max(x, Expr(memcpy_cast<Scalar>(UInt(0x00800000u))));
            else
                x = max(x, Expr(memcpy_cast<Scalar>(UInt(0x0010000000000000ull))));

            Expr e;
            std::tie(x, e) = frexp(x);

            auto lt_inv_sqrt2 = x < Scalar(0.707106781186547524);

            e -= Expr(Scalar(1.f)) & lt_inv_sqrt2;
            x += (x & lt_inv_sqrt2) - Expr(Scalar(1));

            Expr z = x * x;
            Expr y = poly8(x, 3.3333331174e-1, -2.4999993993e-1,
                              2.0000714765e-1, -1.6668057665e-1,
                              1.4249322787e-1, -1.2420140846e-1,
                              1.1676998740e-1, -1.1514610310e-1,
                              7.0376836292e-2);

            y *= x * z;

            y = fmadd(e, Expr(Scalar(-2.12194440e-4)), y);
            z = fmadd(z, Expr(Scalar(-0.5)), x + y);
            z = fmadd(e, Expr(Scalar(0.693359375)), z);

            r = select(eq(derived(), inf), inf, z | ~valid_mask);
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
            r = derived() * reinterpret_array<Expr>(sli<23>(int_array_t<Expr>(n) + 0x7f));
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = ldexp(derived().coeff(i), n.coeff(i));
        }
        return r;
    }

    /// Break floating-point number into normalized fraction and power of 2
    auto frexp_() const {
        using Expr = expr_t<Derived>;
        Expr result_m, result_e;

        /// Caveat: does not handle denormals correctly
        if (Approx) {
            using IntArray = int_array_t<Expr>;
            using Int = scalar_t<IntArray>;
            using IntMask = mask_t<IntArray>;

            const IntArray
                exponent_mask(Int(0x7f800000u)),
                mantissa_sign_mask(Int(~0x7f800000u)),
                bias_minus_1(Int(0x7e));

            IntArray x = reinterpret_array<IntArray>(derived());
            IntArray exponent_bits = x & exponent_mask;

            /* Detect zero/inf/NaN */
            IntMask is_normal =
                IntMask(neq(derived(), zero<Expr>())) &
                neq(exponent_bits, exponent_mask);

            IntArray exponent_i = (sri<23>(exponent_bits)) - bias_minus_1;

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
            Expr x = derived(), r_small, r_big;

            Mask mask_big = abs(x) > Scalar(1);

            if (!all_nested(mask_big)) {
                Expr x2 = x*x;
                r_small = fmadd(
                    poly2(x2, 1.66667160211e-1,
                              8.33028376239e-3,
                              2.03721912945e-4),
                    x2 * x, x);
            }

            if (any_nested(mask_big)) {
                Expr exp0 = exp(x),
                     exp1 = rcp(exp0);

                r_big = (exp0 - exp1) * Scalar(0.5);
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

            Expr x = derived(),
                 exp0 = exp(x),
                 exp1 = rcp(exp0);

            const Expr half((Scalar(0.5f)));

            Expr x2 = x*x;
            Expr r_small = fmadd(
                poly2(x2, 1.66667160211e-1,
                          8.33028376239e-3,
                          2.03721912945e-4),
                x2 * x, x);

            Mask mask_big = abs(x) > Scalar(1);

            s_out = select(mask_big, half * (exp0 - exp1), r_small);

            c_out = half * (exp0 + exp1);
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i) {
                s_out.coeff(i) = sinh(derived().coeff(i));
                c_out.coeff(i) = cosh(derived().coeff(i));
            }
        }

        return std::make_pair(s_out, c_out);
    }

    auto tanh_() const {
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

                r_small = fmadd(
                    poly4(x2, -3.33332819422e-1,
                               1.33314422036e-1,
                              -5.37397155531e-2,
                               2.06390887954e-2,
                              -5.70498872745e-3),
                    x2 * x, x);
            }

            if (any_nested(mask_big)) {
                Expr e  = exp(x+x),
                     e2 = rcp(e + Expr(Scalar(1.0f)));
                r_big = 1.f - (e2 + e2);
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

            r = rcp(exp0 + exp1) * Expr(Scalar(2));
        } else {
            ENOKI_CHKSCALAR for (size_t i = 0; i < Derived::Size; ++i)
                r.coeff(i) = sech(derived().coeff(i));
        }
        return r;
    }

    auto coth_() const { return rcp(tanh(derived())); }

    auto asinh_() const {
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

            Expr x = derived(),
                 x2 = x*x,
                 xa = abs(x),
                 r_big, r_small;

            Mask mask_big  = xa >= Scalar(0.51),
                 mask_huge = xa >= Scalar(1e10);

            if (!all_nested(mask_big)) {
                r_small = fmadd(
                    poly3(x2, -1.6666288134e-1,
                               7.4847586088e-2,
                              -4.2699340972e-2,
                               2.0122003309e-2),
                    x2 * x, x);
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

            Expr x = derived(),
                 x1 = x - Scalar(1),
                 r_big, r_small;

            Mask mask_big  = x1 >= Scalar(0.49),
                 mask_huge = x1 >= Scalar(1e10);

            if (!all_nested(mask_big)) {
                r_small = poly4(x1,  1.4142135263e+0,
                                    -1.1784741703e-1,
                                     2.6454905019e-2,
                                    -7.5272886713e-3,
                                     1.7596881071e-3);
                r_small *= sqrt(x1);
                r_small |= x1 < zero<Expr>();
            }

            if (any_nested(mask_big)) {
                r_big = log(x + (sqrt(fmadd(x, x, Expr(Scalar(-1)))) & ~mask_huge));
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

            Expr x = derived(),
                 xa = abs(x),
                 r_big, r_small;

            Mask mask_big  = xa >= Scalar(0.5);

            if (!all_nested(mask_big)) {
                Expr x2 = x*x;
                r_small = fmadd(
                    poly4(x2, 3.33337300303e-1,
                              1.99782164500e-1,
                              1.46691431730e-1,
                              8.24370301058e-2,
                              1.81740078349e-1),
                    x2*x, x);
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
        detail::MaskWrapper<Value_, Size_, Approx_, RoundingMode::Default>;

NAMESPACE_END(enoki)
