Benchmark
=========

We now turn to the results of a microbenchmark which runs the previously
discussed GPS record distance function on a dynamic array with 10 million
entries.

.. container:: toggle

    .. container:: header

        **Show/Hide Code**



    .. code-block:: cpp
       :linenos:

        /* Compilation flags:
           $ clang++ benchmark.cpp -o benchmark -std=c++14 -I include -O3
                     -march=native -fomit-frame-pointer -fno-stack-protector -DNDEBUG
         */

        #include <enoki/array.h>
        #include <enoki/random.h>
        #include <chrono>

        using namespace enoki;

        auto clk() { return std::chrono::high_resolution_clock::now(); }

        template <typename T> float clkdiff(T a, T b) {
            return std::chrono::duration<float>(b - a).count() * 1000;
        }

        template <typename Value> struct GPSCoord2 {
            using Vector2 = Array<Value, 2>;
            using UInt64  = uint64_array_t<Value>;
            using Bool    = bool_array_t<Value>;

            UInt64 time;
            Vector2 pos;
            Bool reliable;

            ENOKI_STRUCT(GPSCoord2,           /* <- name of this class */
                         time, pos, reliable  /* <- list of all attributes in correct order */)
        };

        ENOKI_STRUCT_DYNAMIC(GPSCoord2, time, pos, reliable)

        using FloatP       = Array<float, SIMD_WIDTH, false>;
        using FloatX       = DynamicArray<FloatP>;
        using GPSCoord2fX  = GPSCoord2<FloatX>;
        using GPSCoord2fP  = GPSCoord2<FloatP>;
        using GPSCoord2f   = GPSCoord2<float>;

        using RNG = PCG32<FloatP>;

        /// Calculate the distance in kilometers between 'r1' and 'r2' using the haversine formula
        template <typename Value_, typename Value = expr_t<Value_>>
        ENOKI_INLINE Value distance(const GPSCoord2<Value_> &r1, const GPSCoord2<Value_> &r2) {
            using Scalar = scalar_t<Value>;

            const Value deg_to_rad = Scalar(M_PI / 180.0);

            auto sin_diff_h = sin(deg_to_rad * Scalar(.5) * (r2.pos - r1.pos));
            sin_diff_h *= sin_diff_h;

            Value a = sin_diff_h.x() + sin_diff_h.y() *
                      cos(r1.pos.x() * deg_to_rad) *
                      cos(r2.pos.x() * deg_to_rad);

            return select(
                r1.reliable & r2.reliable,
                Scalar(6371.0 * 2.0) * atan2(sqrt(a), sqrt(Scalar(1.0) - a)),
                Value(std::numeric_limits<Scalar>::quiet_NaN())
            );
        }

        int main(int argc, char *argv[]) {
            for (int i =0; i<3; ++i) {
                GPSCoord2fX coord1;
                GPSCoord2fX coord2;
                FloatX result;

                auto clk0 = clk();

                size_t size = 10000000;
                dynamic_resize(coord1, size);
                dynamic_resize(coord2, size);
                dynamic_resize(result, size);

                auto clk1 = clk();

                RNG rng;

                for (size_t j = 0; j < packets(coord1); ++j) {
                    packet(coord1, j) = GPSCoord2fP {
                        0,
                        { rng.next_float32() * 180.f - 90, rng.next_float32() * 360.f - 180.f},
                        true
                    };
                    packet(coord2, j) = GPSCoord2fP {
                        0,
                        { rng.next_float32() * 180.f - 90, rng.next_float32() * 360.f - 180.f},
                        true
                    };
                }

                auto clk2 = clk();

                vectorize([](auto &&result, auto &&coord1, auto &&coord2) {
                              result = distance<FloatP>(coord1, coord2);
                          },
                          result, coord1, coord2);

                auto clk3 = clk();
                std::cout << clkdiff(clk2, clk3) << " (alloc = " << clkdiff(clk0, clk1)
                          << ", fill = " << clkdiff(clk1, clk2) << ")" << std::endl;
            }

            return 0;
        }

The plots shows the measured speedup relative to a scalar baseline
implementation. We consider two different microarchitectures:

Knight's Landing microarchitecture (Xeon Phi 7210)
--------------------------------------------------

The Knight's Landing architecture provides hardware support for SIMD arithmetic
using 16 single precision point values. Interestingly, the best performance is
reached when working with arrays of 32 entries, which can be interpreted as a
type of loop unrolling. The ability of issuing wide memory operations,
performing branchless arithmetic using vector registers, and keeping two
independent instructions in flight for each arithmetic operation leads to a
total speedup of 23.5x (i.e. considerably exceeding the expected maximum
speedup of 16 from the vectorized instructions alone!).

Relative to the C math library, Enoki obtains an even larger speedup of
**38.7x**. Using the standard C math library on this platform is fairly
expensive, presumably because of function call overheads (whereas Enoki
generally inlines functions), and because it is compiled for a generic x86_64
machine rather than the native architecture.

*Platform details*: clang trunk rev. 304711 on Linux 64 bit (RHEL 7.3)

.. image:: dynamic-04.svg
    :width: 600px
    :align: center

Skylake microarchitecture (i7-6920HQ)
-------------------------------------

The Skylake architecture provides hardware support for SIMD arithmetic using 8
single precision point values. Significant speedups are observed for packets of
8 and 16 entries. It is likely that more involved functions (i.e. with a higher
register pressure) will have a sharper performance drop after :math:`n=16` due
to the relatively small number of registers on this platform. Enoki
single-precision transcendentals only slightly faster than the standard C math
library on this platform. The max. speedup relative to the standard C math
library is **10.0x**.

*Platform details*: clang trunk rev. 304711 on macOS 10.12.5

.. image:: dynamic-05.svg
    :width: 600px
    :align: center
