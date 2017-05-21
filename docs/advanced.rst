Advanced topics
===============

The histogram problem and conflict detection
--------------------------------------------

Consider vectorizing a function that increments the entries of a histogram
given a SIMD vector with histogram bin indices. It is impossible to do this
kind of indirect update using a normal pair of gather and scatter operations,
since incorrect updates occur whenever the ``indices`` array contains an index
multiple times:

```cpp
using Float = Array<float, 16>;
using Index = Array<int32_t, 16>;

float hist[1000] = { 0.f }; /* Histogram entries */

Index indices = /* .. bin indices whose value should be increased .. */;

/* Ooops, don't do this. Some entries may have to be incremented multiple time.. */
scatter(hist, gather<Float>(hist, indices) + 1, indices);
```

Enoki provides a function named ``transform``, which modifies an indirect
memory location in a way that is not susceptible to conflicts. The function
takes an arbitrary function as parameter and applies it to the specified memory
location, which allows this approach to generalize to situations other than
just building histograms.

```
/* Unmasked version */
transform<Float>(hist, indices, [](auto x) { return x + 1; });

/* Masked version */
transform<Float>(hist, indices, [](auto x) { return x + 1; }, mask);
```

Internally, ``transform`` detects and processes conflicts using the AVX512CDI
instruction set. When conflicts are present, the function provided as an
argument may be applied multiple times in a row. When AVX512CDI is not
available, a (slower) scalar fallback implementation is used.

Supporting new types
--------------------

Adding a new Enoki array type involves creating a new partial overload of the
``StaticArrayImpl<>`` template that derives from ``StaticArrayBase``. To
support the full feature set of Enoki, overloads must provide at least a set of
core methods shown below. The underscores in the function names indicate that
this is considered non-public API that should only be accessed indirectly via
the routing templates in ``enoki/enoki_router.h``.

- Required operations:

    - Loads and stores: ``store_``, ``store_unaligned_``, ``load_``,
      ``load_unaligned_``.

    - Arithmetic and bit-level operations: ``add_``, ``sub_``, ``mul_``, ``mulhi_``
      (signed/unsigned high integer multiplication), ``div_``, ``and_``, ``or_``,
      ``xor_``.

    - Unary operators: ``neg_``, ``not_``.

    - Comparison operators that produce masks: ``ge_``, ``gt_``, ``lt_``, ``le_``,
      ``eq_``, ``neq_``.

    - Other elementary operations: ``abs_``, ``ceil_``, ``floor_``, ``max_``,
      ``min_``, ``round_``, ``sqrt_``.

    - Shift operations for integers: ``sl_``, ``sli_``, ``slv_``, ``sr_``, ``sri_``,
      ``srv_``.

    - Horizontal operations: ``none_``, ``all_``, ``any_``, ``hprod_``, ``hsum_``,
      ``hmax_``, ``hmin_``, ``count_``.

    - Masked blending operation: ``select_``.

    - Access to low and high part (if applicable): ``high_``, ``low_``.

    - Zero-valued array creation: ``zero_``.

- The following operations all have default implementations in Enoki's
  mathematical support library, hence overriding them is optional. However,
  doing so may be worthwile if efficient hardware-level support exists on
  the target platform.

    - Shuffle operation (emulated using scalar operations by default):
      ``shuffle_``.

    - Compressed stores (emulated using scalar operations by default):
      ``store_compress_``.

    - Extracting an element based on a mask (emulated using scalar operations by default):
      ``extract_``.

    - Scatter/gather operations (emulated using scalar operations by default):
      ``scatter_``, ``gather_``.

    - Prefetch operations (no-op by default): ``prefetch_``.

    - Trigonometric and hyperbolic functions: ``sin_``, ``sinh_``, ``sincos_``,
      ``sincosh_``, ``cos_``, ``cosh_``, ``tan_``, ``tanh_``, ``csc_``,
      ``csch_``, ``sec_``, ``sech_``, ``cot_``, ``coth_``, ``asin_``,
      ``asinh_``, ``acos_``, ``acosh_``, ``atan_``, ``atanh_``.

    - Fused multiply-add routines (reduced to ``add_``/``sub_`` and ``mul_`` by
      default): ``fmadd_``, ``fmsub_``, ``fmaddsub_``, ``fmsubadd_``.

    - Reciprocal and reciprocal square root (reduced to ``div_`` and ``sqrt_``
      by default): ``rcp_``, ``rsqrt_``.

    - Dot product (reduced to ``mul_`` and ``hsum_`` by default): ``dot_``.

    - Exponentials, logarithms, powers, floating point exponent manipulation
      functions: ``log_``, ``exp_``, ``pow_`` ``frexp_``, ``ldexp_``.

    - Error function and its inverse: ``erf_``, ``erfinv_``.

    - Optional bit-level rotation operations (reduced to shifts by default):
      ``rol_``, ``roli_``, ``rolv_``, ``ror_``, ``rori_``, ``rorv_``.
