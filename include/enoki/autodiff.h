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

#define ENOKI_AUTODIFF_EXPAND_LIMIT 3

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

        operator Index () const { return index; }

    private:
        Index index = 0;
    };
};

template <typename Value>
struct AutoDiffArray : ArrayBase<value_t<Value>, AutoDiffArray<Value>> {
private:
    // -----------------------------------------------------------------------
    //! @{ \name Forward declarations
    // -----------------------------------------------------------------------

    struct Tape;

    /// Encodes information about less commonly used operations (e.g. scatter/gather)
    struct Special {
        virtual void compute_gradients(Tape &tape) const = 0;
        virtual std::string graphviz() const = 0;
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
                  "AutoDiffArray requires a scalar or (non-nested) static or "
                  "dynamic Enoki array as template parameter.");

    // -----------------------------------------------------------------------
    //! @{ \name Basic declarations
    // -----------------------------------------------------------------------

    using Base = ArrayBase<value_t<Value>, AutoDiffArray<Value>>;
    using typename Base::Scalar;

    using MaskType = AutoDiffArray<mask_t<Value>>;
    using UnderlyingType = Value;
    using Index = uint32_t;
    using RefIndex = detail::ReferenceCountedIndex<AutoDiffArray, Index>;

    static constexpr bool Approx = array_approx_v<Value>;
    static constexpr bool IsMask = is_mask_v<Value>;
    static constexpr bool IsAutoDiff = true;
    static constexpr bool ComputeGradients = std::is_floating_point_v<scalar_t<Value>>;
    static constexpr size_t Size = is_scalar_v<Value> ? 1 : array_size_v<Value>;
    static constexpr size_t Depth = is_scalar_v<Value> ? 1 : array_depth_v<Value>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Array interface (constructors, component access, ...)
    // -----------------------------------------------------------------------

    AutoDiffArray() = default;
    AutoDiffArray(const AutoDiffArray &) = default;
    AutoDiffArray(AutoDiffArray &&) = default;
    AutoDiffArray& operator=(const AutoDiffArray &) = default;
    AutoDiffArray& operator=(AutoDiffArray &&) = default;

    template <typename Value2, enable_if_t<!std::is_same_v<Value, Value2>> = 0>
    AutoDiffArray(const AutoDiffArray<Value2> &a) : value(a.value_()) { }

    template <typename Value2>
    AutoDiffArray(const AutoDiffArray<Value2> &a, detail::reinterpret_flag)
        : value(a.value_(), detail::reinterpret_flag()) { }

    template <typename... Args,
              enable_if_t<std::conjunction_v<
                  std::bool_constant<!is_autodiff_array_v<Args>>...>> = 0>
    AutoDiffArray(Args &&... args) : value(std::forward<Args>(args)...) { }

    template <typename T>
    using ReplaceValue = AutoDiffArray<replace_scalar_t<Value, T>>;

    ENOKI_INLINE size_t size() const {
        if constexpr (is_scalar_v<Value>)
            return 1;
        else
            return value.size();
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
    static AutoDiffArray empty_(Args... args) { return empty<Value>(args...); }
    template <typename... Args>
    static AutoDiffArray zero_(Args... args) { return zero<Value>(args...); }
    template <typename... Args>
    static AutoDiffArray arange_(Args... args) { return arange<Value>(args...); }
    template <typename... Args>
    static AutoDiffArray linspace_(Args... args) { return linspace<Value>(args...); }
    template <typename... Args>
    static AutoDiffArray full_(Args... args) { return full<Value>(args...); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    AutoDiffArray add_(const AutoDiffArray &a) const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = binary("add", index, a.index, 1.f, 1.f);
        return AutoDiffArray::create(index_new, value + a.value);
    }

    AutoDiffArray sub_(const AutoDiffArray &a) const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = binary("sub", index, a.index, 1.f, -1.f);
        return AutoDiffArray::create(index_new, value - a.value);
    }

    AutoDiffArray mul_(const AutoDiffArray &a) const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = binary("mul", index, a.index, a.value, value);
        return AutoDiffArray::create(index_new, value * a.value);
    }

    AutoDiffArray div_(const AutoDiffArray &a) const {
        Value rcp_a = rcp(a.value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = binary("div", index, a.index, rcp_a, -value*sqr(rcp_a));
        return AutoDiffArray::create(index_new, value / a.value);
    }

    AutoDiffArray sqrt_() const {
        Value sqrt_value = sqrt(value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("sqrt", index, .5f / sqrt_value);
        return AutoDiffArray::create(index_new, sqrt_value);
    }

    AutoDiffArray floor_() const {
        return AutoDiffArray::create(0, floor(value));
    }

    AutoDiffArray ceil_() const {
        return AutoDiffArray::create(0, ceil(value));
    }

    AutoDiffArray trunc_() const {
        return AutoDiffArray::create(0, trunc(value));
    }

    AutoDiffArray round_() const {
        return AutoDiffArray::create(0, round(value));
    }

    AutoDiffArray abs_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("abs", index, sign(value));
        return AutoDiffArray::create(index_new, abs(value));
    }

    AutoDiffArray neg_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("neg", index, -1.f);
        return AutoDiffArray::create(index_new, -value);
    }

    AutoDiffArray rcp_() const {
        Value rcp_value = rcp(value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("rcp", index, -sqr(rcp_value));
        return AutoDiffArray::create(index_new, rcp_value);
    }

    AutoDiffArray rsqrt_() const {
        Value rsqrt_1 = rsqrt(value),
              rsqrt_2 = sqr(rsqrt_1),
              rsqrt_3 = rsqrt_1 * rsqrt_2;
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("rsqrt", index, -.5f * rsqrt_3);
        return AutoDiffArray::create(index_new, rsqrt_1);
    }

    AutoDiffArray fmadd_(const AutoDiffArray &a, const AutoDiffArray &b) const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = ternary("fmadd", index, a.index, b.index, a.value, value, 1.f);
        return AutoDiffArray::create(index_new, fmadd(value, a.value, b.value));
    }

    AutoDiffArray fmsub_(const AutoDiffArray &a, const AutoDiffArray &b) const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = ternary("fmsub", index, a.index, b.index, a.value, value, -1.f);
        return AutoDiffArray::create(index_new, fmsub(value, a.value, b.value));
    }

    AutoDiffArray fnmadd_(const AutoDiffArray &a, const AutoDiffArray &b) const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = ternary("fnmadd", index, a.index, b.index, -a.value, -value, 1.f);
        return AutoDiffArray::create(index_new, fnmadd(value, a.value, b.value));
    }

    AutoDiffArray fnmsub_(const AutoDiffArray &a, const AutoDiffArray &b) const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = ternary("fnmsub", index, a.index, b.index, -a.value, -value, -1.f);
        return AutoDiffArray::create(index_new, fnmsub(value, a.value, b.value));
    }

    AutoDiffArray sin_() const {
        auto [s, c] = sincos(value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("sin", index, c);
        return AutoDiffArray::create(index_new, s);
    }

    AutoDiffArray cos_() const {
        auto [s, c] = sincos(value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("cos", index, -s);
        return AutoDiffArray::create(index_new, c);
    }

    std::pair<AutoDiffArray, AutoDiffArray> sincos_() const {
        auto [s, c] = sincos(value);
        Index index_new_s = 0;
        Index index_new_c = 0;
        if constexpr (ComputeGradients) {
            index_new_s = unary("sin", index,  c);
            index_new_c = unary("cos", index, -s);
        }
        return {
            AutoDiffArray::create(index_new_s, s),
            AutoDiffArray::create(index_new_c, c)
        };
    }

    AutoDiffArray tan_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("tan", index, sqr(sec(value)));
        return AutoDiffArray::create(index_new, tan(value));
    }

    AutoDiffArray csc_() const {
        Index index_new = 0;
        Value csc_value = csc(value);
        if constexpr (ComputeGradients)
            index_new = unary("csc", index, -csc_value * cot(value));
        return AutoDiffArray::create(index_new, csc_value);
    }

    AutoDiffArray sec_() const {
        Index index_new = 0;
        Value sec_value = sec(value);
        if constexpr (ComputeGradients)
            index_new = unary("sec", index, sec_value * tan(value));
        return AutoDiffArray::create(index_new, sec_value);
    }

    AutoDiffArray cot_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("cot", index, -sqr(csc(value)));
        return AutoDiffArray::create(index_new, cot(value));
    }

    AutoDiffArray asin_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("asin", index, rsqrt(1-sqr(value)));
        return AutoDiffArray::create(index_new, asin(value));
    }

    AutoDiffArray acos_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("acos", index, -rsqrt(1-sqr(value)));
        return AutoDiffArray::create(index_new, acos(value));
    }

    AutoDiffArray atan_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("atan", index, rcp(1 + sqr(value)));
        return AutoDiffArray::create(index_new, atan(value));
    }

    AutoDiffArray sinh_() const {
        auto [s, c] = sincosh(value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("sinh", index, c);
        return AutoDiffArray::create(index_new, s);
    }

    AutoDiffArray cosh_() const {
        auto [s, c] = sincosh(value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("cosh", index, s);
        return AutoDiffArray::create(index_new, c);
    }

    AutoDiffArray csch_() const {
        Index index_new = 0;
        Value csch_value = csch(value);
        if constexpr (ComputeGradients)
            index_new = unary("csch", index, -csch_value * coth(value));
        return AutoDiffArray::create(index_new, csch_value);
    }

    AutoDiffArray sech_() const {
        Index index_new = 0;
        Value sech_value = sech(value);
        if constexpr (ComputeGradients)
            index_new = unary("sech", index, -sech_value * tanh(value));
        return AutoDiffArray::create(index_new, sech_value);
    }

    AutoDiffArray tanh_() const {
        Index index_new = 0;
        Value tanh_value = tanh(value);
        if constexpr (ComputeGradients)
            index_new = unary("index", index, sqr(sech(value)));
        return AutoDiffArray::create(index_new, tanh_value);
    }

    AutoDiffArray asinh_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("asinh", index, rsqrt(1 + sqr(value)));
        return AutoDiffArray::create(index_new, asinh(value));
    }

    AutoDiffArray acosh_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("acosh", index, rsqrt(sqr(value) - 1));
        return AutoDiffArray::create(index_new, acosh(value));
    }

    AutoDiffArray atanh_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("atanh", index, rcp(1 - sqr(value)));
        return AutoDiffArray::create(index_new, atanh(value));
    }

    AutoDiffArray exp_() const {
        Value exp_value = exp(value);
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("exp", index, exp_value);
        return AutoDiffArray::create(index_new, exp_value);
    }

    AutoDiffArray log_() const {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = unary("log", index, rcp(value));
        return AutoDiffArray::create(index_new, log(value));
    }

    template <typename Mask>
    AutoDiffArray or_(const Mask &m) const {
        Index index_new = 0;
        if constexpr (ComputeGradients && is_mask_v<Mask>)
            index_new = unary("or", index, 1.f);
        return AutoDiffArray::create(index_new, value | m.value_());
    }

    template <typename Mask>
    AutoDiffArray and_(const Mask &m) const {
        Index index_new = 0;
        if constexpr (ComputeGradients && is_mask_v<Mask>)
            index_new = unary("and", index, select(m.value_(), Value(1), Value(0)));
        return AutoDiffArray::create(index_new, value & m.value_());
    }

    template <typename Mask>
    AutoDiffArray xor_(const Mask &m) const {
        if (ComputeGradients && index != 0)
            throw std::runtime_error("AutoDiffArray::xor_(): gradients are not implemented!");
        return AutoDiffArray(value ^ m.value_());
    }

    template <typename Mask>
    AutoDiffArray andnot_(const Mask &m) const {
        if (ComputeGradients && index != 0)
            throw std::runtime_error("AutoDiffArray::andnot_(): gradients are not implemented!");
        return AutoDiffArray(andnot(value, m.value_()));
    }

    AutoDiffArray max_(const AutoDiffArray &a) const {
        Index index_new = 0;
        if constexpr (ComputeGradients) {
            mask_t<Value> m = value > a.value;
            index_new = binary("max", index, a.index,
                               select(m, Value(1), Value(0)),
                               select(m, Value(0), Value(1)));
        }
        return AutoDiffArray::create(index_new, max(value, a.value));
    }

    AutoDiffArray min_(const AutoDiffArray &a) const {
        Index index_new = 0;
        if constexpr (ComputeGradients) {
            mask_t<Value> m = value < a.value;
            index_new = binary("min", index, a.index,
                               select(m, Value(1), Value(0)),
                               select(m, Value(0), Value(1)));
        }
        return AutoDiffArray::create(index_new, min(value, a.value));
    }

    template <typename Mask>
    static AutoDiffArray select_(const Mask &m,
                                 const AutoDiffArray &t,
                                 const AutoDiffArray &f) {
        Index index_new = 0;
        if constexpr (ComputeGradients)
            index_new = binary("select", t.index, f.index,
                               select(m.value_(), Value(1), Value(0)),
                               select(m.value_(), Value(0), Value(1)));
        return AutoDiffArray::create(index_new,
                               select(m.value_(), t.value, f.value));
    }


    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------


    auto all_() const { return all(value); }
    auto any_() const { return any(value); }
    auto count_() const { return count(value); }


    //! @}
    // -----------------------------------------------------------------------

    AutoDiffArray hsum_() const {
        Index index_new = 0;
        if (ComputeGradients && index != 0) {
            struct HorizontalAddition : Special {
                RefIndex source;
                Index target, size;

                void compute_gradients(Tape &tape) const {
                    const Value &grad = tape.nodes[target].gradient;
                    Value &grad_source = tape.nodes[source].gradient;
                    Value grad_hsum = hsum(grad);
                    grad_hsum.resize(size);
                    grad_source += grad_hsum;
                }

                size_t nbytes() const { return sizeof(HorizontalAddition); }

                std::string graphviz() const {
                    return std::to_string(target) + " [shape=doubleoctagon];\n" +
                           std::to_string(target) + " -> " +
                           std::to_string(source) + ";\n";
                }
            };

            HorizontalAddition *ha = new HorizontalAddition();
            ha->size = (Index) value.size();
            ha->source = index;
            ha->target = index_new = special("hadd", ha);
        }

        return AutoDiffArray::create(index_new, hsum(value));
    }

    AutoDiffArray hprod_() const {
        Index index_new = 0;
        Value prod = hprod(value);
        if constexpr (ComputeGradients)
            index_new = unary(index, select(eq(value, 0.f), 0.f, prod / value));
        return AutoDiffArray::create(index_new, prod);
    }

    AutoDiffArray hmax_() const {
        if (ComputeGradients && index != 0)
            throw std::runtime_error("hmax(): gradients not yet implemented!");
        return AutoDiffArray::create(0, hmax(value));
    }

    AutoDiffArray hmin_() const {
        if (ComputeGradients && index != 0)
            throw std::runtime_error("hmin(): gradients not yet implemented!");
        return AutoDiffArray::create(0, hmin(value));
    }

    template <size_t Stride, typename Offset, typename Mask>
    static ENOKI_NOINLINE AutoDiffArray gather_(const void *ptr,
                                              const Offset &offset,
                                              const Mask &mask) {
        static_assert(Stride == sizeof(Scalar), "Unsupported stride!");
        using OffsetType = typename Offset::UnderlyingType;
        using MaskType = typename Mask::UnderlyingType;

        Index index_new = 0;


        if constexpr (ComputeGradients) {
            const Tape &tape = get_tape();

            if (tape.scatter_gather_source != 0) {
                struct Gather : Special {
                    RefIndex source;
                    Index target, size;
                    OffsetType offset;
                    MaskType mask;

                    void compute_gradients(Tape &tape) const {
                        const Value &grad_target = tape.nodes[target].gradient;
                        Value &grad_source = tape.nodes[source].gradient;
                        grad_source.resize(size);
                        scatter_add(grad_source, grad_target, offset, mask);
                    }

                    size_t nbytes() const { return sizeof(Gather); }

                    std::string graphviz() const {
                        return std::to_string(target) + " [shape=doubleoctagon];\n    " +
                               std::to_string(target) + " -> " +
                               std::to_string(source) + ";\n";
                    }
                };

                Gather *g = new Gather();
                g->source = tape.scatter_gather_source;
                g->size = tape.scatter_gather_size;
                g->offset = offset.value_();
                g->mask = mask.value_();
                g->target = index_new = special("gather", g);
            }
        }

        return AutoDiffArray::create(
            index_new, gather<Value>(ptr, offset.value_(), mask.value_()));
    }

    // -----------------------------------------------------------------------
    //! @{ \name s that don't require derivatives
    // -----------------------------------------------------------------------

    AutoDiffArray not_() const { return AutoDiffArray::create(0, ~value); }

    template <size_t Imm> AutoDiffArray sl_() const { return AutoDiffArray::create(0, sl<Imm>(value)); }
    template <size_t Imm> AutoDiffArray sr_() const { return AutoDiffArray::create(0, sr<Imm>(value)); }
    template <size_t Imm> AutoDiffArray rol_() const { return AutoDiffArray::create(0, rol<Imm>(value)); }
    template <size_t Imm> AutoDiffArray ror_() const { return AutoDiffArray::create(0, ror<Imm>(value)); }

    AutoDiffArray sl_(const AutoDiffArray &a) const { return AutoDiffArray::create(0, sl(value, a)); }
    AutoDiffArray sr_(const AutoDiffArray &a) const { return AutoDiffArray::create(0, sr(value, a)); }
    AutoDiffArray rol_(const AutoDiffArray &a) const { return AutoDiffArray::create(0, rol(value, a)); }
    AutoDiffArray ror_(const AutoDiffArray &a) const { return AutoDiffArray::create(0, ror(value, a)); }

    auto eq_ (const AutoDiffArray &d) const { return MaskType(eq(value, d.value)); }
    auto neq_(const AutoDiffArray &d) const { return MaskType(neq(value, d.value)); }
    auto lt_ (const AutoDiffArray &d) const { return MaskType(value < d.value); }
    auto le_ (const AutoDiffArray &d) const { return MaskType(value <= d.value); }
    auto gt_ (const AutoDiffArray &d) const { return MaskType(value > d.value); }
    auto ge_ (const AutoDiffArray &d) const { return MaskType(value >= d.value); }

    //! @}
    // -----------------------------------------------------------------------

private:
    // -----------------------------------------------------------------------
    //! @{ \name Wengert tape for reverse-mode automatic differentiation
    // -----------------------------------------------------------------------

    struct Edge {
        using UInt = uint_array_t<Scalar>;
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

    struct Node {
        Value gradient;
        Index ref_count = 0;
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
        Index scatter_gather_source = 0,
              scatter_gather_size = 0;
        size_t operations = 0,
               contractions = 0;
        bool ready;

        size_t nbytes() const {
            size_t result = 0;
            for (const auto &e: edges)
                result += e.nbytes();
            for (const auto &n: nodes)
                result += n.nbytes();
            return result;
        }

        size_t edge_count(size_t k) const {
            size_t count = 0;
            for (size_t i = nodes[k].edge_offset;
                 i < edges.size() && edges[i].target == k; ++i)
                ++count;
            return count;
        }
    };

    ENOKI_NOINLINE static Tape &get_tape() {
        if (!s_tape) {
            s_tape = new Tape();
            clear_graph_();
        }
        return *s_tape;
    }

    ENOKI_NOINLINE static void add_edge(Tape &tape, Index source, Index target,
                                        const Value &weight) {
        if (source == 0)
            return;

        assert(!tape.nodes[source].is_collected());
        auto &edges = tape.edges;

        size_t deg = tape.edge_count(source);
        if (deg != 0 && deg <= ENOKI_AUTODIFF_EXPAND_LIMIT) {
            for (size_t i = 0; i < tape.edge_count(source); ++i) {
                const Edge &e = edges[tape.nodes[source].edge_offset + i];
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
                    Index &ei = tape.nodes[et.target].edge_offset;
                    ei = std::min(ei, size);
                }
                size++;
            }
            edges.resize(size);

            if (edges.size() * 2 > edges.capacity())
                edges.reserve(edges.capacity() * 2);
        }

        for (ssize_t i = (ssize_t) edges.size() - 1; i > 0; --i) {
            Edge &e = edges[(size_t) i];
            if (e.target != target) {
                break;
            } else if (!e.is_special() && e.source == source && e.target == target) {
                e.weight += weight;
                return;
            }
        }

        tape.nodes[source].ref_count++;
        edges.emplace_back(source, target, weight);
    }

    ENOKI_INLINE static Index add_node(Label label, Tape &tape) {
        Index node_index = (Index) tape.nodes.size();
        if (!tape.free_nodes.empty()) {
            node_index = tape.free_nodes.back();
            tape.free_nodes.pop_back();
        } else {
            tape.nodes.emplace_back();
        }

        tape.nodes[node_index].edge_offset = (Index) tape.edges.size();

        #if !defined(NDEBUG)
            tape.nodes[node_index].label = label;
        #endif

        return node_index;
    }

    ENOKI_NOINLINE static Index unary(Label label, Index i0, const Value &w0) {
        if (ENOKI_LIKELY(i0 == 0)) {
            return 0;
        } else {
            Tape &tape = get_tape();
            Index node_index = add_node(label, tape);
            add_edge(tape, i0, node_index, w0);
            tape.operations++;
            return node_index;
        }
    }

    ENOKI_NOINLINE static Index binary(Label label, uint32_t i0, uint32_t i1,
                                       const Value &w0, const Value &w1) {
        if (ENOKI_LIKELY(i0 == 0 && i1 == 0)) {
            return 0;
        } else {
            Tape &tape = get_tape();
            Index node_index = add_node(label, tape);
            add_edge(tape, i0, node_index, w0);
            add_edge(tape, i1, node_index, w1);
            tape.operations++;
            return node_index;
        }
    }

    ENOKI_NOINLINE static Index ternary(Label label, uint32_t i0, uint32_t i1, uint32_t i2,
                                        const Value &w0, const Value &w1, const Value &w2) {
        if (ENOKI_LIKELY(i0 == 0 && i1 == 0 && i2 == 0)) {
            return 0;
        } else {
            Tape &tape = get_tape();
            Index node_index = add_node(label, tape);
            add_edge(tape, i0, node_index, w0);
            add_edge(tape, i1, node_index, w1);
            add_edge(tape, i2, node_index, w2);
            tape.operations++;
            return node_index;
        }
    }

    ENOKI_NOINLINE static Index special(Label label, Special *special) {
        Tape &tape = get_tape();
        Index node_index = add_node(label, tape);
        tape.edges.emplace_back(special);
        return node_index;
    }

    ENOKI_INLINE static AutoDiffArray create(Index index, const Value &value) {
        AutoDiffArray result(value);
        result.index = index;
        return result;
    }

    inline static __thread Tape *s_tape = nullptr;

    //! @}
    // -----------------------------------------------------------------------

public:
    static void set_scatter_gather_source_(Index index, size_t size = 0) {
        Tape &tape = get_tape();
        tape.scatter_gather_source = index;
        tape.scatter_gather_size = (Index) size;
    }

    static void inc_ref_(Index index) {
        if (index == 0)
            return;
        Tape &tape = get_tape();
        tape.nodes[index].ref_count++;
    }

    static void dec_ref_(Index index) {
        if (index == 0)
            return;
        Tape &tape = get_tape();
        Node &node = tape.nodes[index];
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

    ENOKI_NOINLINE static void clear_graph_() {
        Tape &tape = get_tape();
        tape.edges.clear();
        tape.nodes.clear();
        tape.nodes.emplace_back();
        tape.free_nodes.clear();
        tape.operations = 0;
        tape.contractions = 0;
        tape.ready = false;
    }

    ENOKI_NOINLINE void requires_gradient_(Label label = nullptr) {
        if (index != 0)
            return;
        #if !defined(NDEBUG)
            std::string label_quotes =
                "\\\"" + std::string(label ? label : "unnamed") + "\\\"";
            index = add_node(label_quotes.c_str(), get_tape());
        #else
            index = add_node(label, get_tape());
        #endif
    }

    ENOKI_NOINLINE void backward_() const {
        if (index == 0)
            return;

        Tape &tape = get_tape();
        if (tape.ready)
            throw std::runtime_error(
                "Error: backward() was used twice in a row. A "
                "prior call to clear_graph() is necessary!");
        tape.ready = true;


        Value zero = enoki::zero<Value>();
        size_t node_count = 0, edge_count = 0;
        for (size_t i = 1; i < tape.nodes.size(); ++i) {
            if (!tape.nodes[i].is_collected()) {
                tape.nodes[i].gradient = zero;
                node_count++;
            }
        }

        tape.nodes[index].gradient = 1.f;

        for (ssize_t i = (ssize_t) tape.edges.size() - 1; i >= 0; --i) {
            const Edge &edge = tape.edges[(size_t) i];
            if (edge.is_collected())
                continue;

            if (ENOKI_LIKELY(!edge.is_special())) {
                const Node &target  = tape.nodes[edge.target];
                Node       &source  = tape.nodes[edge.source];
                const Value &weight = edge.weight;
                assert(!source.is_collected());

                masked(source.gradient, neq(weight, zero) & neq(target.gradient, zero)) =
                    fmadd(weight, target.gradient, source.gradient);
            } else {
                edge.special->compute_gradients(tape);
            }
            edge_count++;
        }

        std::cout << "Processed " << node_count << "/" << tape.nodes.size() - 1
                  << " nodes, " << edge_count << "/" << tape.edges.size()
                  << " edges [" << tape.nbytes() << " bytes, "
                  << tape.operations << " ops and " << tape.contractions
                  << " contractions].. " << std::endl;
    }

    ENOKI_NOINLINE static std::string graphviz_() {
        Tape &tape = get_tape();
        std::string result;

        size_t index = 0;
        for (const auto &node : tape.nodes) {
            #if !defined(NDEBUG)
                if (!node.is_collected() && !node.label.empty()) {
                    result += std::to_string(index) + " [label=\"" + node.label +
                              "\\n#" + std::to_string(index) + "\"";
                    if (node.label[0] == '\\')
                        result += " fillcolor=salmon style=filled";
                    result += "];\n";
                }
            #endif

            index++;
        }

        for (const auto &edge : tape.edges) {
            if (edge.is_collected())
                continue;

            if (!edge.is_special())
                result += std::to_string(edge.target) + " -> " +
                          std::to_string(edge.source) + ";\n";
            else
                result += edge.special->graphviz();
        }
        return result;
    }

    ENOKI_NOINLINE Index index_() const { return index; }
    ENOKI_NOINLINE const Value &value_() const { return value; }

    ENOKI_NOINLINE const Value &gradient_() const {
        Tape &tape = get_tape();
        if (index == 0)
            throw std::runtime_error(
                "No gradient was computed for this variable! (a call to "
                "requires_gradient() is necessary.)");
        else if (index >= tape.nodes.size())
            throw std::runtime_error("Gradient index is out of bounds!");

        return tape.nodes[index].gradient;
    }

    // -----------------------------------------------------------------------
    //! @{ \name Internal state
    // -----------------------------------------------------------------------

private:
    Value value;
    RefIndex index = 0;

    //! @}
    // -----------------------------------------------------------------------
};

template <typename T> void clear_graph() {
    T::clear_graph_();
}

template <typename T> ENOKI_INLINE void requires_gradient(T& a, const char *label) {
    if constexpr (is_autodiff_array_v<T>) {
        if constexpr (array_depth_v<T> >= 2) {
            for (size_t i = 0; i < a.size(); ++i) {
                #if defined(NDEBUG)
                    requires_gradient(a.coeff(i), label);
                #else
                    requires_gradient(a.coeff(i), (std::string(label) + " [" +
                                                   std::to_string(i) + "]").c_str());
                #endif
            }
        } else {
            a.requires_gradient_(label);
        }
    }
}

template <typename T> ENOKI_INLINE void requires_gradient(T& a) {
    if constexpr (is_autodiff_array_v<T>) {
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

template <typename T> void backward(const T& a) {
    a.backward_();
}

template <typename T> auto gradient(const T &a) {
    if constexpr (array_depth_v<T> >= 2) {
        using Value = decltype(gradient(a.coeff(0)));
        Array<Value, T::Size> result;
        for (size_t i = 0; i < T::Size; ++i)
            result.coeff(i) = gradient(a.coeff(i));
        return result;
    } else if constexpr (is_autodiff_array_v<T>) {
        return a.gradient_();
    } else {
        static_assert(detail::false_v<T>, "The given array does not have derivatives.");
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
            return std::to_string(a.index_()) + " [fillcolor=cornflowerblue style=filled];\n";
        }
    }
};

template <typename T> std::string graphviz(const T &value) {
    std::string result;
    result += "digraph {\n";
    //result += "rankdir=BT;\n";
    result += "rankdir=RL;\n";
    result += "node [shape=record fontname=Consolas];\n";
    result += detail::graphviz_edges<T>();
    result += detail::graphviz_nodes(value);
    return result + "}";
}

NAMESPACE_END(enoki)
