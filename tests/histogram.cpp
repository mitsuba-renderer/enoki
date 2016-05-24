#include <iostream>
#include <enoki/array.h>

using namespace enoki;

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

/// PCG32 Pseudorandom number generator
template <size_t Size_> struct PCG32 {
    static constexpr size_t Size = Size_;
    using UInt64  = Array<uint64_t, Size>;
    using UInt32  = Array<uint32_t, Size>;
    using Float32 = Array<float, Size>;
    using Float64 = Array<double, Size>;

    /// Initialize the pseudorandom number generator with the \ref seed() function
    PCG32(UInt64 initstate = PCG32_DEFAULT_STATE,
          UInt64 initseq = index_sequence<UInt64>()) {
        seed(initstate, initseq);
    }

    /**
     * \brief Seed the pseudorandom number generator
     *
     * Specified in two parts: a state initializer and a sequence selection
     * constant (a.k.a. stream id)
     */
    void seed(UInt64 initstate, UInt64 initseq = 1) {
        state = zero<UInt64>();
        inc = sli<1>(initseq) | UInt64(1u);
        next_uint();
        state += initstate;
        next_uint();
    }

    /// Generate a uniformly distributed unsigned 32-bit random number
    UInt32 ENOKI_API next_uint() {
        UInt64 oldstate = state;
        state = oldstate * UInt64(PCG32_MULT) + inc;
        UInt32 xorshifted = UInt32(sri<27>(sri<18>(oldstate) ^ oldstate));
        UInt32 rotOffset = UInt32(sri<59>(oldstate));
        return ror(xorshifted, rotOffset);
    }

    /// Generate a single precision floating point value on the interval [0, 1)
    Float32 ENOKI_API next_float() {
        return reinterpret_array<Float32>(sri<9>(next_uint()) | UInt32(0x3f800000)) - 1.f;
    }

    /**
     * \brief Generate a double precision floating point value on the interval [0, 1)
     *
     * \remark Since the underlying random number generator produces 32 bit output,
     * only the first 32 mantissa bits will be filled (however, the resolution is still
     * finer than in \ref next_float(), which only uses 23 mantissa bits)
     */
    Float64 ENOKI_API nextDouble() {
        /* Trick from MTGP: generate an uniformly distributed
           double precision number in [1,2) and subtract 1. */
        return reinterpret_array<Float64>(sli<20>(UInt64(next_uint())) |
                                    UInt64(0x3ff0000000000000)) - 1.0;
    }

    /// Equality operator
    bool operator==(const PCG32 &other) const { return state == other.state && inc == other.inc; }

    /// Inequality operator
    bool operator!=(const PCG32 &other) const { return state != other.state || inc != other.inc; }

    UInt64 state;  // RNG state.  All values are possible.
    UInt64 inc;    // Controls which RNG sequence (stream) is selected. Must *always* be odd.
};

int main(int /* argc */, char * /* argv */[]) {
    using RNG     = PCG32<16>;
    using UInt32  = RNG::UInt32;
    using Float32 = RNG::Float32;

    RNG rng;

    /* Bin configuration */
    const float minValue = -5;
    const float maxValue =  5;
    const int   binCount = 31;
    uint32_t bins[binCount] { };

    for (size_t i = 0; i < 10 * 1024 * 1024 / RNG::Size; ++i) {
        /* Generate a uniform variate */
        Float32 x = rng.next_float();

        /* Importance sample a normal distribution */
        Float32 y = float(M_SQRT2) * erfi(2*x - 1.f);

        /* Compute bin index */
        UInt32 idx((y - minValue) * binCount / (maxValue - minValue));
        idx = min(max(idx, zero<UInt32>()), UInt32(binCount - 1));

        /* Increment the bin indices */
        apply<UInt32>(
            bins, idx, [](auto x) {
                return x + 1;
            }
        );
    }

    for (int i=0; i<binCount; ++i) {
        std::cout << "bin[" << i << "] = ";
        for (uint32_t j = 0; j < bins[i] / 50000; ++j)
            std::cout << "*";
        std::cout << " " << bins[i] << std::endl;
    }

    return 0;
}
