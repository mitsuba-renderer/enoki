Advanced topics
===============

This section is still under construction.

TODO: Python integration, for loops,
bool_array_t, like_t, etc. ENOKI_UNLIKELY, meshgrid,
set_flush_denormals, memory allocator, low(), high(), head<>(), copysign,
mulsign, concat, function calls,
stl.h
compress in operations

dynamic array class & set_slices, slices, packet, slice

- broadcasting
- arithmetic involving arrays of references

Undocumented: reinterpret_array, shuffle

Compressing arrays
------------------

A common design pattern in vectorized code involves *compressing* arrays, i.e.
selectively writing only masked parts of an array so that the selected entries
become densely packed in memory (e.g. to improve resource usage when only parts
of an array participate in a computation).

The function :cpp:func:`compress` efficiently maps this operation onto the
targeted hardware (SSE4.2, AVX2, and AVX512 implementations are provided). The
function also automatically advances the pointer by the amount of written
entries.

.. code-block:: cpp

    Array<float, 16> input = ...;
    auto mask = input < 0;

    float output[16];
    size_t count = compress(output, input, mask);
    std::cout << count << " entries were written." << std::endl;
    ...

Custom data structures such as the GPS record class discussed in previous
chapters are transparently supported by :cpp:func:`compress`---in this case,
the mask applies to each vertical slice through the data structure as
illustrated in the following figure:

.. image:: advanced-01.svg
    :width: 800px
    :align: center

The :cpp:func:`slice_ptr` function is used to acquire a pointer to the
beginning of the output array. It returns a value of type ``GPSRecord2<float
*>``, which is composed of multiple pointers (one for each component). The
following snippet illustrates how an arbitrarily long list of records can be
filtered:

.. code-block:: cpp

    GPSCoord2fX input = /* .. input data to be compressed .. */;

    /* Make sure there is enough space to store all data */
    GPSCoord2fX output;
    set_slices(output, slices(input));

    /* Structure composed of pointers to the output arrays */
    GPSRecord2<float *> ptr = slice_ptr(output, 0);

    /* Counter used to keep track of the number of collected elements */
    size_t final_size = 0;

    /* Go through all packets, compress, and append */
    for (size_t i = 0; i < packets(input); ++i) {
        /* Let/s filter out the records with input.reliable == true */
        auto input_p = packet(input, i);
        final_size += compress(ptr, input_p, input_p.reliable);
    }

    /* Now that the final number of slices is known, adjust the output array size */
    set_slices(output, final_size);

.. _integer-division:

Integer division
----------------
TBD

Reinterpreting the contents of arrays
-------------------------------------

In additions to casts between different types, it is possible to reinterpret
the bit-level representation as a different type when both source and target
types have matching sizes and layouts:

.. code-block:: cpp

    using Source = Array<int64_t, 32>;
    using Target = Array<double, 32>;

    Source source = /* ... integer vector which makes sense when interpreted as a double value ... */;
    Target target = reinterpret_array<Target>(source);

This feature can also be used to convert between mask types.

.. _platform-differences:

Architectural differences handled by Enoki
------------------------------------------

Note that the AVX512 back-end is special and instead uses eight dedicated mask
registers to store masks compactly (allocating only a single bit per mask
entry). Such tedious differences between platforms are invisible in user code
that uses the abstractions of Enoki.

for instance, machines with AVX (but no AVX2)
don't have an 8-wide integer vector unit. This means that an ``Array<float,
8>`` can be represented using a single AVX ``ymm`` register, but casting it to
an ``Array<int32_t, 8>`` entails switching to a pair of half width SSE4.2
``xmm`` integer registers, etc.

---for instance, AVX512 uses special mask
registers, while older Intel machines use normal vector registers that have all
bits set to ``1`` for entries where the comparison was true and ``0``
elsewhere. Such tedious platform differences are hidden when using the
abstractions of Enoki.

- Enoki provides control over the rounding mode of elementary arithmetic
  operations. The AVX512 back-end can translate this into particularly
  efficient instruction sequences with embedded rounding flags.

The histogram problem and conflict detection
--------------------------------------------

Consider vectorizing a function that increments the entries of a histogram
given a SIMD vector with histogram bin indices. It is impossible to do this
kind of indirect update using a normal pair of gather and scatter operations,
since incorrect updates occur whenever the ``indices`` array contains an index
multiple times:

.. code-block:: cpp

    using Float = Array<float, 16>;
    using Index = Array<int32_t, 16>;

    float hist[1000] = { 0.f }; /* Histogram entries */

    Index indices = /* .. bin indices whose value should be increased .. */;

    /* Ooops, don't do this. Some entries may have to be incremented multiple time.. */
    scatter(hist, gather<Float>(hist, indices) + 1, indices);

Enoki provides a function named :cpp:func:`enoki::transform`, which modifies an
indirect memory location in a way that is not susceptible to conflicts. The
function takes an arbitrary function as parameter and applies it to the
specified memory location, which allows this approach to generalize to
situations other than just building histograms.

.. code-block:: cpp

    /* Unmasked version */
    transform<Float>(hist, indices, [](auto x) { return x + 1; });

    /* Masked version */
    transform<Float>(hist, indices, [](auto x) { return x + 1; }, mask);

Internally, :cpp:func:`enoki::transform` detects and processes conflicts using
the AVX512CDI instruction set. When conflicts are present, the function
provided as an argument may be applied multiple times in a row. When AVX512CDI
is not available, a (slower) scalar fallback implementation is used.

.. _custom-arrays:

Defining custom array types
---------------------------

Enoki provides a mechanism for declaring custom array types using the
`Curiously recurring template pattern
<https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern>`_. The
following snippet shows a declaration of a hypothetical type named ``Spectrum``
representing a discretized color spectrum. ``Spectrum`` behaves the same way as
:cpp:class:`Array` and supports all regular Enoki operations.

.. code-block:: cpp

    template <typename Type, size_t Size>
    struct Spectrum : enoki::StaticArrayImpl<Type, Size, false,
                                            RoundingMode::Default,
                                            Spectrum<Type, Size>> {

        /// Base class
        using Base = enoki::StaticArrayImpl<Type, Size, false,
                                            RoundingMode::Default,
                                            Spectrum<Type, Size>>;

        /// Import constructors, assignment operators, etc.
        ENOKI_DECLARE_CUSTOM_ARRAY(Base, Spectrum)

        /// Helper alias used to transition between vector types (used by enoki::vectorize)
        template <typename T> using ReplaceType = Spectrum<T, Size>;
    };

The main reason for declaring custom arrays is to tag (and preserve)
the type of arrays within expressions. For instance, the type of ``value2``
in the following snippet is ``Spectrum<float, 8>``.

.. code-block:: cpp

    Spectrum<float, 8> value = { ... };
    auto value2 = exp(-value);

Adding backends for new instruction sets
----------------------------------------

Adding a new Enoki array type involves creating a new partial overload of the
``StaticArrayImpl<>`` template that derives from ``StaticArrayBase``. To
support the full feature set of Enoki, overloads must provide at least a set of
core methods shown below. The underscores in the function names indicate that
this is considered non-public API that should only be accessed indirectly via
the routing templates in ``enoki/enoki_router.h``.

* The following core operations must be provided by every implementation.

  * Loads and stores: ``store_``, ``store_unaligned_``, ``load_``,
    ``load_unaligned_``.

  * Arithmetic and bit-level operations: ``add_``, ``sub_``, ``mul_``, ``mulhi_``
    (signed/unsigned high integer multiplication), ``div_``, ``and_``, ``or_``,
    ``xor_``.

  * Unary operators: ``neg_``, ``not_``.

  * Comparison operators that produce masks: ``ge_``, ``gt_``, ``lt_``, ``le_``,
    ``eq_``, ``neq_``.

  * Other elementary operations: ``abs_``, ``ceil_``, ``floor_``, ``max_``,
    ``min_``, ``round_``, ``sqrt_``.

  * Shift operations for integers: ``sl_``, ``sli_``, ``slv_``, ``sr_``, ``sri_``,
    ``srv_``.

  * Horizontal operations: ``none_``, ``all_``, ``any_``, ``hprod_``, ``hsum_``,
    ``hmax_``, ``hmin_``, ``count_``.

  * Masked blending operation: ``select_``.

  * Access to low and high part (if applicable): ``high_``, ``low_``.

  * Zero-valued array creation: ``zero_``.

* The following operations all have default implementations in Enoki's
  mathematical support library, hence overriding them is optional.

  However, doing so may be worthwile if efficient hardware-level support exists
  on the target platform.

  * Shuffle operation (emulated using scalar operations by default):
    ``shuffle_``.

  * Compressed stores (emulated using scalar operations by default):
    ``store_compress_``.

  * Extracting an element based on a mask (emulated using scalar operations by default):
    ``extract_``.

  * Scatter/gather operations (emulated using scalar operations by default):
    ``scatter_``, ``gather_``.

  * Prefetch operations (no-op by default): ``prefetch_``.

  * Trigonometric and hyperbolic functions: ``sin_``, ``sinh_``, ``sincos_``,
    ``sincosh_``, ``cos_``, ``cosh_``, ``tan_``, ``tanh_``, ``csc_``,
    ``csch_``, ``sec_``, ``sech_``, ``cot_``, ``coth_``, ``asin_``,
    ``asinh_``, ``acos_``, ``acosh_``, ``atan_``, ``atanh_``.

  * Fused multiply-add routines (reduced to ``add_``/``sub_`` and ``mul_`` by
    default): ``fmadd_``, ``fmsub_``, ``fnmadd_``, ``fnmsub_``,
    ``fmaddsub_``, ``fmsubadd_``.

  * Reciprocal and reciprocal square root (reduced to ``div_`` and ``sqrt_``
    by default): ``rcp_``, ``rsqrt_``.

  * Dot product (reduced to ``mul_`` and ``hsum_`` by default): ``dot_``.

  * Exponentials, logarithms, powers, floating point exponent manipulation
    functions: ``log_``, ``exp_``, ``pow_`` ``frexp_``, ``ldexp_``.

  * Error function and its inverse: ``erf_``, ``erfinv_``.

  * Optional bit-level rotation operations (reduced to shifts by default):
    ``rol_``, ``roli_``, ``rolv_``, ``ror_``, ``rori_``, ``rorv_``.
