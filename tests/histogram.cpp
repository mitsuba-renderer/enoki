/*
    test/histogram.cpp -- Test which uses transform_<> to build a histogram
    of a set of normally distributed pseudorandom samples

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE.txt file.
*/

#if defined(NDEBUG)
#  undef NDEBUG
#endif

#include "random.h"

int main(int /* argc */, char * /* argv */[]) {
    using UInt32      = Array<uint32_t>;
    using UInt32Mask  = typename UInt32::Mask;
    using RNG         = PCG32<UInt32>;
    using Float32     = RNG::Float32;
    using UInt64      = RNG::UInt64;

    /* Bin configuration */
    const float min_value = -4;
    const float max_value =  4;
    const uint32_t bin_count = 31;
    uint32_t bins[bin_count] { };

    for (size_t j = 0; j < 16 / RNG::Size; ++j) {
        RNG rng(PCG32_DEFAULT_STATE, index_sequence<UInt64>() + (j * RNG::Size));

        for (size_t i = 0; i < 1024 * 1024; ++i) {
            /* Generate a uniform variate */
            Float32 x = rng.next_float32();

            /* Importance sample a normal distribution */
            Float32 y = float(M_SQRT2) * erfi(2.f*x - 1.f);

            /* Compute bin index */
            UInt32 idx((y - min_value) * float(bin_count) / (max_value - min_value));

            /* Discard samples that are out of bounds */
            UInt32Mask mask = idx >= zero<UInt32>() && idx < bin_count;

            /* Increment the bin indices */
            transform<UInt32>(
                bins, idx, [](auto x) {
                    return x + 1u;
                },
                mask
            );
        }
    }

    uint32_t sum = 0;
    for (uint32_t i = 0; i < bin_count; ++i) {
        std::cout << "bin[" << i << "] = ";
        for (uint32_t j = 0; j < bins[i] / 50000; ++j)
            std::cout << "*";
        std::cout << " " << bins[i] << std::endl;
        sum += bins[i];
    }

    assert((16 * 1024 * 1024 - sum) == 743);
    assert(bins[0] == 1342);
    assert(bins[1] == 2558);

    return 0;
}
