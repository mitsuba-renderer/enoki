/*
    enoki/autodiff.h -- Reverse mode automatic differentiation

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>
#include <vector>
#include <thread>

#define ENOKI_AUTODIFF_EXPAND_LIMIT 0

NAMESPACE_BEGIN(enoki)

namespace detail {
    template <typename Base, typename Index>
    struct ReferenceCountedIndex {
        ReferenceCountedIndex() = default;
        ReferenceCountedIndex(Index index) : index(index) {
            Base::inc_ref_(index);
        }

        ReferenceCountedIndex(const ReferenceCountedIndex &o) : index(o.index) {
            Base::inc_ref_(index);
        }

        ReferenceCountedIndex(ReferenceCountedIndex &&o) : index(o.index) {
            o.index = 0;
        }

        ReferenceCountedIndex &operator=(const ReferenceCountedIndex &o) {
            Base::dec_ref_(index);
            index = o.index;
            Base::inc_ref_(index);
            return *this;
        }

        ReferenceCountedIndex &operator=(Index index2) {
            Base::dec_ref_(index);
            index = index2;
            Base::inc_ref_(index);
            return *this;
        }

        ReferenceCountedIndex &operator=(ReferenceCountedIndex &&o) {
            Base::dec_ref_(index);
            index = o.index;
            o.index = 0;
            return *this;
        }

        ~ReferenceCountedIndex() {
            Base::dec_ref_(index);
        }

        void swap(ReferenceCountedIndex &o) { std::swap(index, o.index); }

        operator Index () const { return index; }

    private:
        Index index = 0;
    };
};

template <typename T> struct TapeScope;

template <typename Value>
struct DiffArray : ArrayBase<value_t<Value>, DiffArray<Value>> {
    friend struct TapeScope<DiffArray<Value>>;
private:
    // -----------------------------------------------------------------------
    //! @{ \name Forward declarations
    // -----------------------------------------------------------------------

    struct Tape;

    /// Encodes information about less commonly used operations (e.g. scatter/gather)
    struct Special {
        virtual void compute_gradients(Tape &tape) const = 0;
        virtual std::string graphviz(const Tape &tape) const = 0;
        virtual size_t nbytes() const = 0;
        virtual ~Special() = default;
    };

#if defined(NDEBUG)
    struct Label { Label(const char *) { } };
#else
    using Label = const char *;
#endif

    //! @}
    // -----------------------------------------------------------------------

public:
    static_assert(array_depth_v<Value> <= 1,
                  "DiffArray requires a scalar or (non-nested) static or "
                  "dynamic Enoki array as template parameter.");

    // -----------------------------------------------------------------------
    //! @{ \name Basic declarations
    // -----------------------------------------------------------------------

    using Base = ArrayBase<value_t<Value>, DiffArray<Value>>;
    using typename Base::Scalar;

    using MaskType = DiffArray<mask_t<Value>>;
    using UnderlyingType = Value;
    using Index = uint32_t;
    using RefIndex = detail::ReferenceCountedIndex<DiffArray, Index>;

    static constexpr bool Approx = array_approx_v<Value>;
    static constexpr bool IsMask = is_mask_v<Value>;
    static constexpr bool IsDiff = true;
    static constexpr bool ComputeGradients = std::is_floating_point_v<scalar_t<Value>> &&
                                            !is_mask_v<Value>;
    static constexpr size_t Size = is_scalar_v<Value> ? 1 : array_size_v<Value>;
    static constexpr size_t Depth = is_scalar_v<Value> ? 1 : array_depth_v<Value>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Array interface (constructors, component access, ...)
    // -----------------------------------------------------------------------

    ENOKI_INLINE DiffArray() { }

    ENOKI_NOINLINE ~DiffArray() { }

    ENOKI_NOINLINE DiffArray(const DiffArray &a)
        : value(a.value), index(a.index) { }

    ENOKI_INLINE DiffArray(DiffArray &&a)
        : value(std::move(a.value)), index(a.index) {
        a.index = 0;
    }

    ENOKI_NOINLINE DiffArray &operator=(const DiffArray &a) {
        value = a.value;
        index = a.index;
        return *this;
    }

    ENOKI_INLINE DiffArray &operator=(DiffArray &&a) {
        value = std::move(a.value);
        index.swap(a.index);
        return *this;
    }

    template <typename Value2, enable_if_t<!std::is_same_v<Value, Value2>> = 0>
    DiffArray(const DiffArray<Value2> &a) : value(a.value_()) { }

    template <typename Value2, enable_if_t<!std::is_same_v<Value, Value2>> = 0>
    DiffArray(DiffArray<Value2> &&a) : value(std::move(a.value_())) { }

    template <typename Value2>
    DiffArray(const DiffArray<Value2> &a, detail::reinterpret_flag)
        : value(a.value_(), detail::reinterpret_flag()) { }

    template <typename... Args,
              enable_if_t<std::conjunction_v<
                  std::bool_constant<!is_diff_array_v<Args>>...>> = 0>
    DiffArray(Args&&... args) : value(std::forward<Args>(args)...) { }

    ENOKI_NOINLINE DiffArray(const Value &value) : value(value) { }
    ENOKI_NOINLINE DiffArray(Value &&value) : value(std::move(value)) { }

    template <typename T>
    using ReplaceValue = DiffArray<replace_scalar_t<Value, T, false>>;

    ENOKI_INLINE size_t size() const {
        if constexpr (is_scalar_v<Value>)
            return 1;
        else
            return value.size();
    }

    ENOKI_NOINLINE void resize(size_t size) {
        ENOKI_MARK_USED(size);
        if constexpr (!is_scalar_v<Value>)
            value.resize(size);
    }

    template <typename... Args>
    ENOKI_INLINE decltype(auto) coeff(Args... args) {
        static_assert(sizeof...(Args) == Depth, "coeff(): Invalid number of arguments!");
        if constexpr (is_scalar_v<Value>)
            return value;
        else
            return value.coeff((size_t) args...);
    }

    template <typename... Args>
    ENOKI_INLINE decltype(auto) coeff(Args... args) const {
        static_assert(sizeof...(Args) == Depth, "coeff(): Invalid number of arguments!");
        if constexpr (is_scalar_v<Value>)
            return value;
        else
            return value.coeff((size_t) args...);
    }

    ENOKI_INLINE decltype(auto) data() {
        if constexpr (is_scalar_v<Value>)
            return &value;
        else
            return value.data();
    }

    ENOKI_INLINE decltype(auto) data() const {
        if constexpr (is_scalar_v<Value>)
            return &value;
        else
            return value.data();
    }

    template <typename... Args>
    static DiffArray empty_(Args... args) { return empty<Value>(args...); }
    template <typename... Args>
    static DiffArray zero_(Args... args) { return zero<Value>(args...); }
    template <typename... Args>
    static DiffArray arange_(Args... args) { return arange<Value>(args...); }
    template <typename... Args>
    static DiffArray linspace_(Args... args) { return linspace<Value>(args...); }
    template <typename... Args>
    static DiffArray full_(Args... args) { return full<Value>(args...); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    template <typename... Args>
    ENOKI_INLINE bool needs_gradients_(const Args&...args) const {
        return ((index != 0) || ... || (args.index != 0));
    }

    DiffArray add_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::add_(): unsupported operation!");
        } else {
            Value result = value + a.value;
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_(a))
                    index_new = binary("add", index, a.index, 1, 1, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sub_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::sub_(): unsupported operation!");
        } else {
            Value result = value - a.value;
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_(a))
                    index_new = binary("sub", index, a.index, 1, -1, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray mul_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::mul_(): unsupported operation!");
        } else {
            Value result = value * a.value;
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_(a))
                    index_new = binary("mul", index, a.index, a.value, value, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray div_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::div_(): unsupported operation!");
        } else {
            Value result = value / a.value;
            Index index_new = 0;
            if constexpr (ComputeGradients) {
                if (needs_gradients_(a)) {
                    Value rcp_a = rcp(a.value);
                    index_new = binary("div", index, a.index, rcp_a, -value*sqr(rcp_a), result.size());
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sqrt_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::sqrt_(): unsupported operation!");
        } else {
            Value result = sqrt(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("sqrt", index, .5f / result, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray floor_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>)
            throw std::runtime_error("DiffArray::floor_(): unsupported operation!");
        else
            return DiffArray::create(0, floor(value));
    }

    DiffArray ceil_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>)
            throw std::runtime_error("DiffArray::ceil_(): unsupported operation!");
        else
            return DiffArray::create(0, ceil(value));
    }

    template <typename T> T ceil2int_() const {
        return T(ceil2int<typename T::UnderlyingType>(value));
    }

    template <typename T> T floor2int_() const {
        return T(floor2int<typename T::UnderlyingType>(value));
    }

    DiffArray trunc_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>)
            throw std::runtime_error("DiffArray::trunc_(): unsupported operation!");
        else
            return DiffArray::create(0, trunc(value));
    }

    DiffArray round_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>)
            throw std::runtime_error("DiffArray::round_(): unsupported operation!");
        else
            return DiffArray::create(0, round(value));
    }

    DiffArray abs_() const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::abs_(): unsupported operation!");
        } else {
            Value result = abs(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("abs", index, sign(value), result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray neg_() const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::abs_(): unsupported operation!");
        } else {
            Value result = -value;
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("neg", index, -1, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rcp_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::rcp_(): unsupported operation!");
        } else {
            Value result = rcp(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("rcp", index, -sqr(result), result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rsqrt_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::rsqrt_(): unsupported operation!");
        } else {
            Value result = rsqrt(value);
            Index index_new = 0;
            if constexpr (ComputeGradients) {
                if (needs_gradients_()) {
                    Value rsqrt_2 = sqr(result), rsqrt_3 = result * rsqrt_2;
                    index_new = unary("rsqrt", index, -.5f * rsqrt_3, result.size());
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::fmadd_(): unsupported operation!");
        } else {
            Value result = fmadd(value, a.value, b.value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_(a, b))
                    index_new = ternary("fmadd", index, a.index, b.index, a.value, value, 1, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::fmsub_(): unsupported operation!");
        } else {
            Value result = fmsub(value, a.value, b.value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_(a, b))
                    index_new = ternary("fmsub", index, a.index, b.index, a.value, value, -1, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::fmsub_(): unsupported operation!");
        } else {
            Value result = fnmadd(value, a.value, b.value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_(a, b))
                    index_new = ternary("fnmadd", index, a.index, b.index, -a.value, -value, 1, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::fnmsub_(): unsupported operation!");
        } else {
            Value result = fnmsub(value, a.value, b.value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_(a, b))
                    index_new = ternary("fnmsub", index, a.index, b.index, -a.value, -value, -1, result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sin_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::sin(): unsupported operation!");
        } else {
            auto [s, c] = sincos(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("sin", index, c, value.size());
            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cos_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::cos_(): unsupported operation!");
        } else {
            auto [s, c] = sincos(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("cos", index, -s, value.size());
            return DiffArray::create(index_new, std::move(c));
        }
    }

    std::pair<DiffArray, DiffArray> sincos_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::sincos_(): unsupported operation!");
        } else {
            auto [s, c] = sincos(value);
            Index index_new_s = 0;
            Index index_new_c = 0;
            if constexpr (ComputeGradients) {
                if (needs_gradients_()) {
                    index_new_s = unary("sin", index,  c, value.size());
                    index_new_c = unary("cos", index, -s, value.size());
                }
            }
            return {
                DiffArray::create(index_new_s, std::move(s)),
                DiffArray::create(index_new_c, std::move(c))
            };
        }
    }

    DiffArray tan_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::tan_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("tan", index, sqr(sec(value)), value.size());
            return DiffArray::create(index_new, tan(value));
        }
    }

    DiffArray csc_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::csc_(): unsupported operation!");
        } else {
            Index index_new = 0;
            Value csc_value = csc(value);
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("csc", index, -csc_value * cot(value), value.size());
            return DiffArray::create(index_new, std::move(csc_value));
        }
    }

    DiffArray sec_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::sec_(): unsupported operation!");
        } else {
            Index index_new = 0;
            Value sec_value = sec(value);
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("sec", index, sec_value * tan(value), value.size());
            return DiffArray::create(index_new, std::move(sec_value));
        }
    }

    DiffArray cot_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::cot_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("cot", index, -sqr(csc(value)), value.size());
            return DiffArray::create(index_new, cot(value));
        }
    }

    DiffArray asin_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::asin_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("asin", index, rsqrt(1 - sqr(value)), value.size());
            return DiffArray::create(index_new, asin(value));
        }
    }

    DiffArray acos_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::acos_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("acos", index, -rsqrt(1 - sqr(value)), value.size());
            return DiffArray::create(index_new, acos(value));
        }
    }

    DiffArray atan_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::atan_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("atan", index, rcp(1 + sqr(value)), value.size());
            return DiffArray::create(index_new, atan(value));
        }
    }

    DiffArray sinh_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::sinh_(): unsupported operation!");
        } else {
            auto [s, c] = sincosh(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("sinh", index, c, value.size());
            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cosh_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::cosh_(): unsupported operation!");
        } else {
            auto [s, c] = sincosh(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("cosh", index, s, value.size());
            return DiffArray::create(index_new, std::move(c));
        }
    }

    DiffArray csch_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::csch_(): unsupported operation!");
        } else {
            Value result = csch(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("csch", index, -result * coth(value), result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sech_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::sech_(): unsupported operation!");
        } else {
            Value result = sech(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("sech", index, -result * tanh(value), result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray tanh_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::tanh_(): unsupported operation!");
        } else {
            Value result = tanh(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("index", index, sqr(sech(value)), result.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray asinh_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::asinh_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("asinh", index, rsqrt(1 + sqr(value)), value.size());
            return DiffArray::create(index_new, asinh(value));
        }
    }

    DiffArray acosh_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::acosh_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("acosh", index, rsqrt(sqr(value) - 1), value.size());
            return DiffArray::create(index_new, acosh(value));
        }
    }

    DiffArray atanh_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::atanh_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("atanh", index, rcp(1 - sqr(value)), value.size());
            return DiffArray::create(index_new, atanh(value));
        }
    }

    DiffArray exp_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::exp_(): unsupported operation!");
        } else {
            Value result = exp(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("exp", index, result, value.size());
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray log_() const {
        if constexpr (is_mask_v<Value> || !std::is_floating_point_v<Scalar>) {
            throw std::runtime_error("DiffArray::log_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("log", index, rcp(value), value.size());
            return DiffArray::create(index_new, log(value));
        }
    }

    DiffArray or_(const DiffArray &m) const {
        if constexpr (!is_mask_v<Value> && !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::or_(): unsupported operation!");
        else
            return DiffArray::create(0, value | m.value_());
    }

    template <typename Mask> DiffArray or_(const Mask &m) const {
        Index index_new = 0;
        if constexpr (ComputeGradients && is_mask_v<Mask>)
            if (needs_gradients_())
                index_new = unary("or", index, 1, value.size());
        return DiffArray::create(index_new, value | m.value_());
    }

    DiffArray and_(const DiffArray &m) const {
        if constexpr (!is_mask_v<Value> && !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::and_(): unsupported operation!");
        else
            return DiffArray::create(0, value & m.value_());
    }

    template <typename Mask>
    DiffArray and_(const Mask &m) const {
        Index index_new = 0;
        if constexpr (ComputeGradients && is_mask_v<Mask>)
            if (needs_gradients_())
                index_new = unary("and", index, select(m.value_(), Value(1), Value(0)), value.size());
        return DiffArray::create(index_new, value & m.value_());
    }

    DiffArray xor_(const DiffArray &m) const {
        if constexpr (!is_mask_v<Value> && !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::xor_(): unsupported operation!");
        else
            return DiffArray::create(0, value ^ m.value_());
    }

    template <typename Mask>
    DiffArray xor_(const Mask &m) const {
        if (ComputeGradients && index != 0)
            throw std::runtime_error("DiffArray::xor_(): gradients are not implemented!");
        return DiffArray(value ^ m.value_());
    }

    DiffArray andnot_(const DiffArray &m) const {
        if constexpr (!is_mask_v<Value> && !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::andnot_(): unsupported operation!");
        else
            return DiffArray::create(0, andnot(value, m.value_()));
    }

    template <typename Mask>
    DiffArray andnot_(const Mask &m) const {
        if (ComputeGradients && index != 0)
            throw std::runtime_error("DiffArray::andnot_(): gradients are not implemented!");
        return DiffArray(andnot(value, m.value_()));
    }

    DiffArray max_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value>) {
            throw std::runtime_error("DiffArray::max_(): unsupported operation!");
        } else {
            Value result = max(value, a.value);
            Index index_new = 0;
            if constexpr (ComputeGradients) {
                mask_t<Value> m = value > a.value;
                index_new = binary("max", index, a.index,
                                   select(m, Value(1), Value(0)),
                                   select(m, Value(0), Value(1)),
                                   result.size());
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray min_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value>) {
            throw std::runtime_error("DiffArray::min_(): unsupported operation!");
        } else {
            Value result = min(value, a.value);
            Index index_new = 0;
            if constexpr (ComputeGradients) {
                if (needs_gradients_(a)) {
                    mask_t<Value> m = value < a.value;
                    index_new = binary("min", index, a.index,
                                       select(m, Value(1), Value(0)),
                                       select(m, Value(0), Value(1)),
                                       result.size());
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    static DiffArray select_(const DiffArray<mask_t<Value>> &m,
                             const DiffArray &t,
                             const DiffArray &f) {
        Value result = select(m.value_(), t.value, f.value);
        Index index_new = 0;
        if constexpr (ComputeGradients) {
            if (t.index != 0 || f.index != 0) {
                index_new = binary("select", t.index, f.index,
                                   select(m.value_(), Value(1), Value(0)),
                                   select(m.value_(), Value(0), Value(1)),
                                   m.size());
            }
        }
        return DiffArray::create(index_new, std::move(result));
    }


    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------


    auto all_() const {
        if constexpr (!is_mask_v<Value>)
            throw std::runtime_error("DiffArray::all_(): unsupported operation!");
        else
            return all(value);
    }

    auto any_() const {
        if constexpr (!is_mask_v<Value>)
            throw std::runtime_error("DiffArray::any_(): unsupported operation!");
        else
            return any(value);
    }

    auto count_() const {
        if constexpr (!is_mask_v<Value>)
            throw std::runtime_error("DiffArray::count_(): unsupported operation!");
        else
            return count(value);
    }

    //! @}
    // -----------------------------------------------------------------------

    DiffArray hsum_() const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::hsum_(): unsupported operation!");
        } else {
            Index index_new = 0;
            if (ComputeGradients && index != 0) {
                struct HorizontalAddition : Special {
                    RefIndex source;
                    Index target, size;

                    void compute_gradients(Tape &tape) const {
                        const Value &grad = tape.node(target).gradient;
                        Value &grad_source = tape.node(source).gradient;
                        Value grad_hsum = hsum(grad);
                        grad_hsum.resize(size);
                        grad_source += grad_hsum;
                    }

                    size_t nbytes() const { return sizeof(HorizontalAddition); }

                    std::string graphviz(const Tape &tape) const {
                        return std::to_string(target - tape.offset) + " [shape=doubleoctagon];\n" +
                               std::to_string(target - tape.offset) + " -> " +
                               std::to_string(source - tape.offset) + ";\n";
                    }
                };

                HorizontalAddition *ha = new HorizontalAddition();
                ha->size = (Index) value.size();
                ha->source = index;
                ha->target = index_new = special("hadd", ha, 1);
            }

            return DiffArray::create(index_new, hsum(value));
        }
    }

    DiffArray hprod_() const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::hprod_(): unsupported operation!");
        } else {
            Value result = hprod(value);
            Index index_new = 0;
            if constexpr (ComputeGradients)
                if (needs_gradients_())
                    index_new = unary("hprod", index, select(eq(value, 0), 0.f, result / value), 1);
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray hmax_() const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::hmax_(): unsupported operation!");
        } else {
            if (ComputeGradients && index != 0)
                throw std::runtime_error("hmax(): gradients not yet implemented!");
            return DiffArray::create(0, hmax(value));
        }
    }

    DiffArray hmin_() const {
        if constexpr (is_mask_v<Value> || std::is_pointer_v<Scalar>) {
            throw std::runtime_error("DiffArray::hmin_(): unsupported operation!");
        } else {
            if (ComputeGradients && index != 0)
                throw std::runtime_error("hmin(): gradients not yet implemented!");
            return DiffArray::create(0, hmin(value));
        }
    }

    template <size_t Stride, typename Offset, typename Mask>
    static ENOKI_NOINLINE DiffArray gather_(const void *ptr,
                                            const Offset &offset,
                                            const Mask &mask) {
        using OffsetType = typename Offset::UnderlyingType;
        using MaskType = typename Mask::UnderlyingType;

        Index index_new = 0;


        if constexpr (ComputeGradients) {
            const Tape &tape = get_tape();

            if (tape.scatter_gather_source != nullptr &&
                tape.scatter_gather_source->index != 0) {
                if (Stride != sizeof(Scalar))
                    throw std::runtime_error("Differentiable gather: unsupported stride!");
                struct Gather : Special {
                    RefIndex source;
                    Index target, size;
                    OffsetType offset;
                    MaskType mask;

                    void compute_gradients(Tape &tape) const {
                        const Value &grad_target = tape.node(target).gradient;
                        Value &grad_source = tape.node(source).gradient;
                        grad_source.resize(size);
                        scatter_add(grad_source.data(), offset, grad_target, mask);
                    }

                    size_t nbytes() const { return sizeof(Gather); }

                    std::string graphviz(const Tape &tape) const {
                        return std::to_string(target - tape.offset) + " [shape=doubleoctagon];\n    " +
                               std::to_string(target - tape.offset) + " -> " +
                               std::to_string(source - tape.offset) + ";\n";
                    }
                };

                Gather *g = new Gather();
                g->source = tape.scatter_gather_source->index;
                g->size = (Index) tape.scatter_gather_source->size();
                g->offset = offset.value_();
                g->mask = mask.value_();
                g->target = index_new = special("gather", g, offset.size());
            }
        }

        return DiffArray::create(
            index_new, gather<Value>(ptr, offset.value_(), mask.value_()));
    }

    template <size_t Stride, typename Offset, typename Mask>
    ENOKI_NOINLINE void scatter_add_(void *ptr, const Offset &offset, const Mask &mask) const {
        static_assert(Stride == sizeof(Scalar), "Unsupported stride!");
        using OffsetType = typename Offset::UnderlyingType;
        using MaskType = typename Mask::UnderlyingType;

        if constexpr (ComputeGradients) {
            const Tape &tape = get_tape();

            if (tape.scatter_gather_source != nullptr &&
                index != 0) {
                struct ScatterAdd : Special {
                    RefIndex source;
                    Index target, size;
                    OffsetType offset;
                    MaskType mask;

                    void compute_gradients(Tape &tape) const {
                        Value &grad_target = tape.node(target).gradient;
                        const Value &grad_source = tape.node(source).gradient;
                        grad_target = gather<Value>(grad_source.data(), offset, mask);
                    }

                    size_t nbytes() const { return sizeof(ScatterAdd); }

                    std::string graphviz(const Tape &tape) const {
                        return std::to_string(target - tape.offset) + " [shape=doubleoctagon];\n    " +
                               std::to_string(target - tape.offset) + " -> " +
                               std::to_string(source - tape.offset) + ";\n";
                    }
                };

                DiffArray& target = const_cast<DiffArray &>(*tape.scatter_gather_source);
                ScatterAdd *sa = new ScatterAdd();
                sa->source = special("scatter_add", sa, target.size());
                sa->target = index;
                sa->offset = offset.value_();
                sa->mask = mask.value_();
                if (target.index == 0)
                    target.index = sa->source;
                else
                    target.index = binary("add", sa->source, target.index, 1, 1, target.size());
            }
        }

        scatter_add(ptr, offset.value_(), value_(), mask.value_());
    }

    // -----------------------------------------------------------------------
    //! @{ \name s that don't require derivatives
    // -----------------------------------------------------------------------

    DiffArray not_() const {
        if constexpr ((!is_mask_v<Value> && !std::is_integral_v<Scalar>) ||
                      std::is_pointer_v<Scalar>)
            throw std::runtime_error("DiffArray::not_(): unsupported operation!");
        else
            return DiffArray::create(0, ~value);
    }

    template <typename Mask>
    ENOKI_INLINE value_t<Value> extract_(const Mask &mask) const {
        if constexpr (is_mask_v<Value> || ComputeGradients)
            throw std::runtime_error("DiffArray::extract_(): unsupported operation!");
        else
            return extract(value, mask.value_());
    }

    template <size_t Imm>
    DiffArray sl_() const { return DiffArray::create(0, sl<Imm>(value)); }

    template <size_t Imm>
    DiffArray sr_() const { return DiffArray::create(0, sr<Imm>(value)); }

    template <size_t Imm>
    DiffArray rol_() const { return DiffArray::create(0, rol<Imm>(value)); }

    template <size_t Imm>
    DiffArray ror_() const { return DiffArray::create(0, ror<Imm>(value)); }

    DiffArray sl_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::sl_(): unsupported operation!");
        else
            return DiffArray::create(0, value << a.value);
    }

    DiffArray sr_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::sr_(): unsupported operation!");
        else
            return DiffArray::create(0, value >> a.value);
    }

    DiffArray rol_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::rol_(): unsupported operation!");
        else
            return DiffArray::create(0, rol(value, a.value));
    }

    DiffArray ror_(const DiffArray &a) const {
        if constexpr (is_mask_v<Value> || !std::is_integral_v<Scalar>)
            throw std::runtime_error("DiffArray::ror_(): unsupported operation!");
        else
            return DiffArray::create(0, ror(value, a.value));
    }

    auto eq_ (const DiffArray &d) const { return MaskType(eq(value, d.value)); }
    auto neq_(const DiffArray &d) const { return MaskType(neq(value, d.value)); }
    auto lt_ (const DiffArray &d) const { return MaskType(value < d.value); }
    auto le_ (const DiffArray &d) const { return MaskType(value <= d.value); }
    auto gt_ (const DiffArray &d) const { return MaskType(value > d.value); }
    auto ge_ (const DiffArray &d) const { return MaskType(value >= d.value); }

    //! @}
    // -----------------------------------------------------------------------

private:
    // -----------------------------------------------------------------------
    //! @{ \name Wengert tape for reverse-mode automatic differentiation
    // -----------------------------------------------------------------------

    struct Edge {
        using UInt = uint_array_t<Scalar, false>;

        /// A quiet NaN with a distinct mantissa bit pattern is used to mark 'special' nodes
        static constexpr UInt special_flag =
            sizeof(UInt) == 4 ? UInt(0b0'11111111'10101010101010101010101u) :
            UInt(0b0'11111111111'10101010101010101010101010101010101010101010ull);

        union {
            struct {
                Index source;
                Index target;
            };
            Special *special;
        };

        Value weight;

        Edge() : source(0), target(0) { }

        Edge(Index source, Index target, Value weight)
            : source(source), target(target), weight(weight) { }

        Edge(Edge &&e) : source(e.source), target(e.target), weight(e.weight) {
            e.special = nullptr;
        }

        Edge(Special *special)
            : special(special), weight(memcpy_cast<Scalar>(special_flag)) { }

        ~Edge() {
            if (is_special()) {
                delete special;
            }
        }

        Edge& operator=(Edge &&e) {
            source = e.source;
            target = e.target;
            weight = e.weight;
            e.special = nullptr;
            return *this;
        }

        size_t nbytes() const {
            size_t result = sizeof(Edge);
            if constexpr (is_array_v<Value>)
                result += + weight.nbytes() - sizeof(Value);
            if (is_special())
                result += special->nbytes();
            return result;
        }

        bool is_special() const {
            if constexpr (is_scalar_v<Value>)
                return reinterpret_array<UInt>(weight) == UInt(special_flag);
            else
                return !weight.empty() &&
                       reinterpret_array<UInt>(weight.coeff(0)) == UInt(special_flag);
        }

        bool is_collected() const { return source == 0 && target == 0; }
    };

    enum NodeFlags {
        ScalarNode = 0x10
    };

    struct Node {
        Value gradient;
        uint16_t ref_count = 0;
        uint16_t flags = 0;
        Index edge_offset = 0;

        #if !defined(NDEBUG)
            std::string label;
        #endif

        size_t nbytes() const {
            size_t result = sizeof(Node);
            if constexpr (is_array_v<Value>)
                result += + gradient.nbytes() - sizeof(Value);
            return result;
        }

        bool is_collected() const { return ref_count == 0; }
    };

    struct Tape {
        std::vector<Edge> edges;
        std::vector<Node> nodes;
        std::vector<Index> free_nodes;
        std::mutex mutex;

        Index offset = 0;
        size_t active_node_count = 0;

        const DiffArray *scatter_gather_source = nullptr;

        size_t operations = 0,
               contractions = 0;
        bool processed;

        size_t nbytes() const {
            size_t result = 0;
            for (const auto &e: edges)
                result += e.nbytes();
            for (const auto &n: nodes)
                result += n.nbytes();
            return result;
        }

        size_t edge_count(Index k) const {
            size_t count = 0;
            for (size_t i = node(k).edge_offset;
                 i < edges.size() && edges[i].target == k; ++i)
                ++count;
            return count;
        }

        Node &node(Index i) {
            if (i == 0)
                return nodes[0];
            assert(i - offset < nodes.size());
            return nodes[i - offset];
        }

        const Node &node(Index i) const {
            if (i == 0)
                return nodes[0];
            assert(i - offset < nodes.size());
            return nodes[i - offset];
        }
    };

    ENOKI_NOINLINE static Tape &get_tape() {
        if constexpr (ComputeGradients) {
            if (ENOKI_UNLIKELY(!s_tape)) {
                s_tape = new Tape();
                clear_graph_();
            }
            return *s_tape;
        } else {
            throw std::runtime_error("DiffArray::get_tape(): unsupported operation!");
        }
    }

    ENOKI_NOINLINE static void add_edge(Tape &tape, Index source, Index target,
                                        const Value &weight) {
        if constexpr (ComputeGradients) {
            if (source == 0)
                return;

            assert(!tape.node(source).is_collected());
            auto &edges = tape.edges;

            size_t deg = tape.edge_count(source);
            if (deg != 0 && deg <= ENOKI_AUTODIFF_EXPAND_LIMIT) {
                for (size_t i = 0; i < tape.edge_count(source); ++i) {
                    const Edge &e = edges[tape.node(source).edge_offset + i];
                    assert (!e.is_special());
                    add_edge(tape, e.source, target,
                             select(eq(weight, 0) || eq(e.weight, 0), 0,
                                    weight * e.weight));
                    ++tape.contractions;
                }
                return;
            }

            if (edges.size() == edges.capacity()) {
                /* Edge list compaction */
                Index size = 0;
                for (size_t i = 0; i < edges.size(); ++i) {
                    Edge &es = edges[i];
                    if (es.is_collected())
                        continue;
                    if (i != size) {
                        Edge &et = edges[size];
                        et = std::move(es);
                        Index &ei = tape.node(et.target).edge_offset;
                        ei = std::min(ei, size);
                    }
                    size++;
                }
                edges.resize(size);

                if (edges.size() * 2 > edges.capacity())
                    edges.reserve(edges.capacity() * 2);
            }

            for (ssize_t i = (ssize_t) edges.size() - 1; i >= 0; --i) {
                Edge &e = edges[(size_t) i];
                if (e.target != target) {
                    break;
                } else if (!e.is_special() && e.source == source && e.target == target) {
                    e.weight += weight;
                    return;
                }
            }

            tape.node(source).ref_count++;
            edges.emplace_back(source, target, weight);
        } else {
            throw std::runtime_error("DiffArray::add_edge(): unsupported operation!");
        }
    }

    ENOKI_INLINE static Index add_node(Label label, Tape &tape, size_t size) {
        if constexpr (ComputeGradients) {
            Index node_index = (Index) tape.nodes.size() + tape.offset;
            if (!tape.free_nodes.empty()) {
                node_index = tape.free_nodes.back();
                tape.free_nodes.pop_back();
            } else {
                tape.nodes.emplace_back();
            }

            Node &node = tape.node(node_index);
            node.edge_offset = (Index) tape.edges.size();
            node.flags = size == 1 ? ScalarNode : 0;

            ENOKI_MARK_USED(label);
            #if !defined(NDEBUG)
                node.label = label;
            #endif

            return node_index;
        } else {
            throw std::runtime_error("DiffArray::add_edge(): unsupported operation!");
        }
    }

    ENOKI_NOINLINE static Index unary(Label label, Index i0, const Value &w0, size_t size) {
        if (ENOKI_LIKELY(i0 == 0)) {
            return 0;
        } else {
            Tape &tape = get_tape();
            Index node_index = add_node(label, tape, size);
            add_edge(tape, i0, node_index, w0);
            tape.operations++;
            return node_index;
        }
    }

    ENOKI_NOINLINE static Index binary(Label label, uint32_t i0, uint32_t i1,
                                       const Value &w0, const Value &w1,
                                       size_t size) {
        if (ENOKI_LIKELY(i0 == 0 && i1 == 0)) {
            return 0;
        } else {
            Tape &tape = get_tape();
            Index node_index = add_node(label, tape, size);
            add_edge(tape, i0, node_index, w0);
            add_edge(tape, i1, node_index, w1);
            tape.operations++;
            return node_index;
        }
    }

    ENOKI_NOINLINE static Index ternary(Label label, uint32_t i0, uint32_t i1, uint32_t i2,
                                        const Value &w0, const Value &w1, const Value &w2,
                                        size_t size) {
        if (ENOKI_LIKELY(i0 == 0 && i1 == 0 && i2 == 0)) {
            return 0;
        } else {
            Tape &tape = get_tape();
            Index node_index = add_node(label, tape, size);
            add_edge(tape, i0, node_index, w0);
            add_edge(tape, i1, node_index, w1);
            add_edge(tape, i2, node_index, w2);
            tape.operations++;
            return node_index;
        }
    }

    ENOKI_NOINLINE static Index special(Label label, Special *special, size_t size) {
        Tape &tape = get_tape();
        Index node_index = add_node(label, tape, size);
        tape.edges.emplace_back(special);
        return node_index;
    }

    ENOKI_INLINE static DiffArray create(Index index, Value &&value) {
        DiffArray result(std::move(value));
        result.index = index;
        return result;
    }

    inline static __thread Tape *s_tape = nullptr;

    //! @}
    // -----------------------------------------------------------------------

public:
    static void *get_tape_ptr() { return s_tape; }
    static void set_tape_ptr(void *ptr) { s_tape = (Tape *) ptr; }
    static void ensure_(size_t size) {
        if constexpr (ComputeGradients) {
            Tape &tape = get_tape();
            tape.nodes.reserve(size);
        }
    }

    static void set_scatter_gather_source_(const DiffArray &t) {
        if constexpr (ComputeGradients) {
            Tape &tape = get_tape();
            tape.scatter_gather_source = &t;
        }
    }

    static void clear_scatter_gather_source_() {
        if constexpr (ComputeGradients) {
            Tape &tape = get_tape();
            tape.scatter_gather_source = nullptr;
        }
    }

    static Index tape_offset_() { return get_tape().offset; }

    static void inc_ref_(Index index) {
        ENOKI_MARK_USED(index);

        if constexpr (ComputeGradients) {
            if (index == 0)
                return;
            Tape &tape = get_tape();
            tape.node(index).ref_count++;
        }
    }

    static void dec_ref_(Index index) {
        ENOKI_MARK_USED(index);

        if constexpr (ComputeGradients) {
            if (index == 0)
                return;
            Tape &tape = get_tape();
            if (index < tape.offset)
                return;
            Node &node = tape.node(index);
            if (node.ref_count == 0)
                throw std::runtime_error("Reference counting error!");
            if (--node.ref_count == 0) {
                if constexpr (is_dynamic_array_v<Value>)
                    node.gradient.reset();
                tape.free_nodes.push_back(index);
                for (size_t i = node.edge_offset; i < tape.edges.size(); ++i) {
                    Edge &edge = tape.edges[i];
                    if (edge.is_special() || edge.target != index)
                        break;
                    dec_ref_(edge.source);
                    edge.source = edge.target = 0;
                    if constexpr (is_dynamic_array_v<Value>)
                        edge.weight.reset();
                }
            }
        }
    }

    ENOKI_NOINLINE static void clear_graph_() {
        if constexpr (ComputeGradients) {
            Tape &tape = get_tape();
            tape.offset += tape.nodes.size();
            tape.edges.clear();
            tape.nodes.clear();
            tape.nodes.emplace_back();
            tape.free_nodes.clear();
            tape.operations = 0;
            tape.contractions = 0;
            tape.processed = false;
        } else {
            throw std::runtime_error("DiffArray::clear_graph_(): unsupported operation!");
        }
    }

    static void clear_gradients_() {
        if constexpr (ComputeGradients) {
            Tape &tape = get_tape();
            size_t node_count = 0;
            Value zero = enoki::zero<Value>();

            for (size_t i = 1; i < tape.nodes.size(); ++i) {
                if (!tape.nodes[i].is_collected()) {
                    tape.nodes[i].gradient = zero;
                    node_count++;
                }
            }

            tape.active_node_count = node_count;
        } else {
            throw std::runtime_error("DiffArray::clear_gradients_(): unsupported operation!");
        }
    }

    ENOKI_NOINLINE void requires_gradient_(Label label = nullptr) {
        ENOKI_MARK_USED(label);

        if constexpr (ComputeGradients) {
            #if !defined(NDEBUG)
                std::string label_quotes =
                    "\\\"" + std::string(label ? label : "unnamed") + "\\\"";
                index = add_node(label_quotes.c_str(), get_tape(), value.size());
            #else
                index = add_node(label, get_tape(), value.size());
            #endif
        }
    }

    static void set_gradient_(Index i, const Value &grad) {
        auto &tape = get_tape();
        tape.nodes.at(i - tape.offset).gradient = grad;
    }

    static void backward_static_() {
        if constexpr (ComputeGradients) {
            Tape &tape = get_tape();

            size_t edge_count = 0;
            Value zero = enoki::zero<Value>();

            for (ssize_t i = (ssize_t) tape.edges.size() - 1; i >= 0; --i) {
                const Edge &edge = tape.edges[(size_t) i];
                if (edge.is_collected())
                    continue;

                if (ENOKI_LIKELY(!edge.is_special())) {
                    const Value &weight = edge.weight;
                    const Node &target  = tape.node(edge.target);
                    Node       &source  = tape.node(edge.source);
                    assert(!source.is_collected());

                    masked(source.gradient, neq(weight, zero) & neq(target.gradient, zero)) =
                        fmadd(weight, target.gradient, source.gradient);

                    if (ENOKI_UNLIKELY((source.flags & NodeFlags::ScalarNode) != 0 &&
                                        source.gradient.size() != 1)) {
                        source.gradient = hsum(source.gradient);
                    }
                } else {
                    edge.special->compute_gradients(tape);
                }
                edge_count++;
            }

            std::cout << "Processed " << tape.active_node_count << "/" << tape.nodes.size() - 1
                      << " nodes, " << edge_count << "/" << tape.edges.size()
                      << " edges [" << tape.nbytes() << " bytes, "
                      << tape.operations << " ops and " << tape.contractions
                      << " contractions].. " << std::endl;
        } else {
            throw std::runtime_error("DiffArray::backward_static_(): unsupported operation!");
        }
    }

    ENOKI_NOINLINE void backward_() const {
        if constexpr (ComputeGradients) {
            if (index == 0)
                return;

            Tape &tape = get_tape();
            if (tape.processed)
                throw std::runtime_error(
                    "Error: backward() was used twice in a row. A "
                    "prior call to clear_graph() is necessary!");
            tape.processed = true;

            clear_gradients_();
            tape.node(index).gradient = 1;
            backward_static_();
        } else {
            throw std::runtime_error("DiffArray::backward_(): unsupported operation!");
        }
    }

    ENOKI_NOINLINE static std::string graphviz_() {
        if constexpr (ComputeGradients) {
            Tape &tape = get_tape();
            std::string result;

            #if !defined(NDEBUG)
            size_t index = 0;
            for (const auto &node : tape.nodes) {
                if (!node.is_collected() && !node.label.empty()) {
                    result += "  ";
                    result += std::to_string(index) + " [label=\"" + node.label;
                    if (node.flags & NodeFlags::ScalarNode)
                        result += " [s]";

                    result += "\\n#" + std::to_string(index) + "\"";
                    if (node.label[0] == '\\')
                        result += " fillcolor=salmon style=filled";
                    result += "];\n";
                }

                index++;
            }
            #endif

            for (const auto &edge : tape.edges) {
                if (edge.is_collected())
                    continue;

                result += "  ";
                if (!edge.is_special())
                    result += std::to_string(edge.target - tape.offset) + " -> " +
                              std::to_string(edge.source - tape.offset) + ";\n";
                else
                    result += edge.special->graphviz(tape);
            }
            return result;
        } else {
            throw std::runtime_error("DiffArray::graphviz_(): unsupported operation!");
        }
    }

    ENOKI_INLINE Index index_() const { return index; }
    ENOKI_INLINE const Value &value_() const { return value; }
    ENOKI_INLINE Value &value_() { return value; }

    size_t nbytes() const {
        if constexpr (is_array_v<Value>)
            return sizeof(DiffArray) - sizeof(Value) + value.nbytes();
        else
            return sizeof(DiffArray);
    }

    const Value &gradient_() const { return gradient_static_(index); }

    ENOKI_NOINLINE static const Value &gradient_static_(Index index) {
        Tape &tape = get_tape();
        if (index == 0)
            throw std::runtime_error(
                "No gradient was computed for this variable! (a call to "
                "requires_gradient() is necessary.)");
        else if (index - tape.offset >= tape.nodes.size())
            throw std::runtime_error("Gradient index is out of bounds!");

        return tape.node(index).gradient;
    }

    auto operator->() const {
        using BaseType = std::decay_t<std::remove_pointer_t<Scalar>>;
        return call_support<BaseType, DiffArray>(*this);
    }

private:
    // -----------------------------------------------------------------------
    //! @{ \name Internal state
    // -----------------------------------------------------------------------

    Value value;
    RefIndex index = 0;

    //! @}
    // -----------------------------------------------------------------------
};

template <typename T> struct TapeScope {
    TapeScope(void *ptr) : ptr(ptr) {
        backup = T::get_tape_ptr();
        T::set_tape_ptr(ptr);
        ((typename T::Tape *) ptr)->mutex.lock();
    }

    ~TapeScope() {
        ((typename T::Tape *) ptr)->mutex.unlock();
        T::set_tape_ptr(backup);
    }

    void *ptr;
    void *backup;
};

template <typename T> void clear_graph() { T::clear_graph_(); }
template <typename T> void clear_gradients() { T::clear_gradients_(); }

template <typename T, typename T2> void gradient_inc_ref(const T2 &index) {
    if constexpr (std::is_scalar_v<T2>) {
        T::inc_ref_(index);
    } else {
        for (auto i : index)
            gradient_inc_ref<T>(i);
    }
}

template <typename T, typename T2> void gradient_dec_ref(const T2 &index) {
    if constexpr (std::is_scalar_v<T2>) {
        T::dec_ref_(index);
    } else {
        for (auto i : index)
            gradient_dec_ref<T>(i);
    }
}

template <typename T> ENOKI_INLINE void requires_gradient(T& a, const char *label) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> >= 2) {
            for (size_t i = 0; i < a.size(); ++i) {
                #if defined(NDEBUG)
                    requires_gradient(a.coeff(i), label);
                #else
                    requires_gradient(a.coeff(i), (std::string(label) + "." +
                                                   std::to_string(i)).c_str());
                #endif
            }
        } else {
            a.requires_gradient_(label);
        }
    }
}

template <typename T> ENOKI_INLINE void requires_gradient(T& a) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> >= 2) {
            for (size_t i = 0; i < a.size(); ++i)
                requires_gradient(a.coeff(i));
        } else {
            a.requires_gradient_();
        }
    }
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
void requires_gradient(Args &... args) {
    bool unused[] = { (requires_gradient(args), false)... };
    (void) unused;
}

template <typename T> void backward(const T& a) { a.backward_(); }
template <typename T> void backward() { T::backward_static_(); }

template <typename T1 = void, typename T2> decltype(auto) gradient(const T2 &a) {
    using Target = std::conditional_t<std::is_void_v<T1>, T2, T1>;

    if constexpr (array_depth_v<Target> >= 2) {
        Target result;
        for (size_t i = 0; i < array_size_v<Target>; ++i)
            result.coeff(i) = gradient<value_t<Target>>(a[i]);
        return result;
    } else if constexpr (is_diff_array_v<T2>) {
        return a.gradient_();
    } else if constexpr (std::is_integral_v<T2>) {
        return T1::gradient_static_(a);
    } else {
        static_assert(detail::false_v<T1, T2>, "The given array does not have derivatives.");
    }
}


namespace detail {
    template <typename T> std::string graphviz_edges() {
        if constexpr (array_depth_v<T> >= 2)
            return graphviz_edges<value_t<T>>();
        else
            return T::graphviz_();
    }

    template <typename T> std::string graphviz_nodes(const T &a) {
        if constexpr (array_depth_v<T> >= 2) {
            std::string result;
            for (size_t i = 0; i < T::Size; ++i)
                result += graphviz_nodes(a.coeff(i));
            return result;
        } else {
            return "  " + std::to_string(a.index_() - a.tape_offset_()) + " [fillcolor=cornflowerblue style=filled];\n";
        }
    }
};

template <typename T> std::string graphviz(const T &value) {
    std::string result;
    result += "digraph {\n";
    //result += "  rankdir=BT;\n";
    result += "  rankdir=RL;\n";
    result += "  node [shape=record fontname=Consolas];\n";
    result += detail::graphviz_edges<T>();
    result += detail::graphviz_nodes(value);
    return result + "}";
}

template <typename T>
auto gradient_index(const T &a) {
    if constexpr (is_static_array_v<T>) {
        using Index = decltype(gradient_index(a.coeff(0)));
        std::array<Index, T::Size> result;
        for (int i = 0; i < a.size(); ++i)
            result[i] = gradient_index(a.coeff(i));
        return result;
    } else {
        return a.index_();
    }
}

template <typename T, typename Index>
auto set_gradient(const Index &index, const T &a) {
    if constexpr (is_static_array_v<T>) {
        for (int i = 0; i < a.size(); ++i)
            set_gradient(index[i], a.coeff(i));
    } else {
        T::set_gradient_(index, a);
    }
}

NAMESPACE_END(enoki)
