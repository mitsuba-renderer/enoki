<p align="center">
    <img src="https://github.com/mitsuba-renderer/enoki/raw/master/resources/enoki-logo.png" alt="Enoki logo" width="400"/>
</p>

# Enoki — fast vectorized arithmetic on modern processors

**Enoki** is a C++ template library that enables transparent vectorization
of numerical code on Intel processors. It is implemented as a set of header
files and has no dependencies other than a sufficiently C++14-capable compiler
(e.g. GCC, Clang, Intel C++ Compiler 2016, Visual Studio 2017).

Enoki is split into two major components: the front-end provides various
elementary and transcendental math functions and generally resembles a standard
math library implementation. The back-end provides the basic ingredients that
are needed to realize these operations using the SIMD instruction set(s)
supported by the target hardware (e.g. AVX512, AVX2, AVX, FMA, F16C, and
SSE4.2).

There are a number of differences in comparison to existing vectorized math
libraries (e.g. Intel MKL, AMD ACML,
[ssemath](http://gruntthepeon.free.fr/ssemath) by Julien Pommier,
[avx_mathfun](http://software-lisc.fbk.eu/avx_mathfun) by Giovanni Garberoglio,
and Agner Fog's [vector library](http://www.agner.org/optimize/#vectorclassg)):

- Enoki has good coverage of elementary and transcendental mathematical
  functions; specifically, ``cos``, ``sin``, ``sincos``, ``tan``, ``csc``,
  ``sec``, ``cot``, ``acos``, ``asin``, ``atan``, ``atan2``, ``exp``, ``log``,
  ``pow``, ``sinh``, ``cosh``, ``sincosh``, ``tanh``, ``csch``, ``sech``,
  ``coth``, ``asinh``, ``acosh``, ``atanh``, ``frexp``, and ``ldexp``, ``erf``,
  and ``erfi`` are supported.

  Efficient vectorized branch-free implementations of all of the above
  functions are available. It is worth noting that these are less accurate than
  their standard C math library counterparts: depending on the function, the
  approximations have an average relative error between 0.1 and 4 ULPs. The C
  math library can be used as a fallback when higher precision transcendental
  functions are needed.

- Enoki makes heavy use of SIMD intrinsics to ensure that compilers generate
  efficient machine code. The intrinsics are contained in the different
  back-end header files (e.g. ``array_avx.h`` for AVX intrinsics), which
  provide rudimentary arithmetic and bit-level operations. Fancier operations
  (e.g. ``atan2``) use the back-ends as an abstract interface to the hardware,
  which means that it's simple to support other instruction sets such as a
  hypothetical future AVX1024 or even an entirely different architecture (ARM?)
  by just adding a new back-end.

- Enoki supports arbitrary array sizes that don't necessarily match what is
  supported by the underlying hardware (e.g. 16 x single precision on a machine
  whose SSE vector only has hardware support for 4 x single precision
  operands). The library uses template metaprogramming techniques to
  efficiently map array expressions onto the available hardware resources.

  This greatly simplifies development because it's enough to write a single
  implementation of a numerical algorithm that can then be deployed on any
  target architecture.

- Enoki provides control over the rounding mode of elementary arithmetic
  operations. The AVX512 back-end can translate this into particularly
  efficient instruction sequences with embedded rounding flags.

- Enoki has native support for 32 and 64 bit integers and single and double
  precision floating point operands. It also exposes a limited amount of
  hardware support for half precision operands (mainly conversion operations).

- In addition to simple static arrays, Enoki also provides support for
  dynamically allocated arrays and SoA-style vectors (more on this below).

- There are non-vectorized fallbacks for everything, thus programs will run
  even on unsupported architectures (albeit without the performance benefits of
  vectorization).

- Enoki is available under a non-viral open source license (3-clause BSD).

The project is named after Enokitake, a type of mushroom with many long and
parallel stalks reminiscent of data flow in SIMD arithmetic.

## License

Copyright (c) 2017 Wenzel Jakob. All rights reserved. Use of this source code
is governed by a BSD-style license that can be found in the
[LICENSE.txt](LICENSE.txt) file.

# Basic overview

The remainder of this document provides a basic overview of the Enoki library.
All code snippets assume that the following lines are present:

```cpp
#include <iostream>
#include <enoki/array.h>

/* Don't forget to include the 'enoki' namespace */
using namespace enoki;
```

## Static arrays: ``enoki::Array<Type, Size>``

The most important data structure in this library is the ``Array`` class, which is
a generic container that encapsulates a fixed-size array of an arbitrary data
type. An important difference that distinguishes ``enoki::Array`` from the
superficially similar ``std::array`` container is that Enoki arrays forward all
C++ operators (and other standard mathematical functions) to the contained
elements. For instance, the following somewhat contrived piece of code is valid
since ``Array::operator+()`` can carry out the desired addition by invoking
``std::string::operator+()``.

```cpp
using StringArray = Array<std::string, 2>;

StrArray a1("Hello ", "How are "),
         a2("world!", "you?");

// Prints: "[Hello world!,  How are you?]"
std::cout << a1 + a2 << std::endl;
```

The real use case of Enoki arrays, however, involves packed arrays of integer
or floating point values, for which arithmetic operations can often be reduced
to fast vectorized instructions provided by current processor architectures.
The library ships with a large number of partial template overloads that become
active when the ``Type`` and ``Size`` parameters supplied to the
``enoki::Array<Type, Size>`` template correspond to combinations that are
natively supported by the targeted hardware.

In addition to ``Type`` and ``Size``, ``enoki::Array`` supports two additional
template parameters.  We will explicitly specify them all below to define a new
type named ``MyFloat``:

```cpp
using MyFloat = Array<
    float,                 // Type:   Underlying scalar data type
    4,                     // Size:   Number of packed float elements
    true,                  // Approx: Use approximate math library?
    RoundingMode::Default  // Mode:   Rounding mode (Default/Up/Down/Zero/Nearest)
>;
```

Some of the parameters can be omitted: if ``Size`` is not specified, the
implementation chooses the largest value that is natively supported by the
target hardware. The ``Approx`` and ``Mode`` template parameters only make
sense when dealing with floating point types. In that case, approximate math is
activated by default when ``Type`` is a single precision ``float``. The default
rounding mode ``RoundingMode::Default`` means that the library won't interfere
with the hardware's currently selected rounding mode.

It is worth pointing out that that ``Array`` does *not* require ``Size`` to
exactly match what is supported by the hardware to benefit from vectorization.
Enoki relies on template metaprogramming techniques to ensure optimal code
generation even in such challenging situations. For instance, on a machine with
SSE4.2 (4-wide single precision arrays) and AVX (8-wide single precision
arrays), any arithmetic operations involving a ``Array<float, 21>`` will create
two AVX instructions, one SSE4.2 instruction, and one scalar instruction. A
perhaps more sensible use of this feature is to instantiate packed arrays with
a ``Size`` that is an integer multiple of what is supported natively as a way
of aggressively unrolling the underlying computations.

#### Initializing, reading, and writing data

Arrays can be initialized by broadcasting a scalar value, or by
specifying the values of the individual entries.

```cpp
/* Initialize all entries with a constant */
MyFloat f1(1.f);

/* Initialize the entries individually */
MyFloat f2(1.f, 2.f, 3.f, 4.f);
```

Explicit load and store operations (especially aligned ones) are preferable
when accessing data via pointers.

``` cpp
float *mem = /* ... pointer to floating point data ... */;
MyFloat f3;

/* Load entries of 'f3' from 'mem' */
f3 = load<MyFloat>(mem);           /* if known that 'mem' is aligned */
f3 = load_unaligned<MyFloat>(mem); /* otherwise */


/* Store entries of 'f3' to 'mem' */
store(mem, f3);                    /* if known that 'mem' is aligned */
store_unaligned(mem, f3)           /* otherwise */
```

Scatter and gather operations are also supported:

``` cpp
/* 32 and 64 bit integers allowed as indices for scatter/gather operations */
Array<int, 4> idx(1, 2, 3, 4);

/* Load f3[i] from mem[idx[i]] (i = 1, 2, ..)*/
f3 = gather<MyFloat>(mem, idx);

/* Write f3[i] to mem[idx[i]] (i = 1, 2, ..)*/
scatter(mem, f3, idx);
```

All scatter/gather operations accept an extra masking parameter to disable
memory accesses for some of the components. Mask values are discussed shortly.

Finally, the following initialization methods also exist:

``` cpp
/* More efficient way to create an array with zero entries */
f1 = zero<MyFloat>();

/* Initialize entries with index sequence 0, 1, 2, 3, ... */
f1 = index_sequence<MyFloat>();
```

#### A brief interlude: simultaneous scalar and vector implementations

When vectorizing a numerical routine in an existing codebase, it's useful and
often necessary to keep the original scalar version of the code around rather
than simply replacing it with its vectorized equivalent. Needless to say, this
can lead to a considerable amount of code duplication. When using Enoki, this
problem can be avoided by replacing scalar functions with *function templates*
that support instantiation using both scalar and array arguments. An example
of this is given below:

```cpp
/* Contrived scalar code that accumulates a Gaussian kernel into a floating
   point buffer. Only accepts float arguments :( */
void kernel_scalar(float x, float y, float sigma, float *buf) {
    buf[0] += std::exp(-0.5f * (x*x + y*y) / sigma);
}

/* This verson other hand can be instantiated as *both* kernel<float> and kernel<MyFloat>!*/
template <typename T> void kernel(T x, T y, T sigma, float *buffer) {
    store(buf,
          load<T>(buf) + exp(-0.5f * (x*x + y*y) / sigma));
}
```

#### ``enoki::Array`` discussion continued: element access and streams

The components of ``Array`` can be accessed via ``operator[]``. If you find
yourself using this much, your code is likely not making good use of the vector
units.

```cpp
f2[2] = 1.f;
```

Alternatively, the functions ``x()``, ``y()``, ``z()``, and ``w()`` can be used
to access the first four components. The following line is equivalent to the
one above.

```cpp
f2.z() = 1.f;
```

Enoki provides an overloaded ``operator<<(std::ostream&, ...)`` stream
insertion operator to facilitate inspection of array contents:

```cpp
/* The line below prints: [1, 2, 3, 4] */
std::cout << MyFloat(1.f, 2.f, 3.f, 4.f) << std::endl;
```

#### Vertical operations

Enoki provides the following *vertical* operations. The word vertical implies
that they are independently applied to all array elements.

```cpp
/* Basic arithmetic operations*/
f1 *= (f2 + 1.f) / (f2 - 1.f);

/* Basic math library functions */
f2 = ceil(f1); f2 = floor(f1); f2 = round(f1);
f2 = abs(f1);  f2 = sqrt(f1); f2 = sign(f1);
f2 = min(f1, f2); f2 = max(f1, f2);

/* Fused multiply-add/subtract */
f1 = fmadd(f1, f2, f3); /* f1 * f2 + f3 */
f1 = fmsub(f1, f2, f3); /* f1 * f2 - f3 */

/* Efficient reciprocal and reciprocal square root */
f1 = rcp(f1);
f1 = rsqrt(f1);

/* Trigonometric and inverse trigonometric functions */
f2 = sin(f1);   f2 = cos(f1);    f2 = tan(f1);
f2 = csc(f1);   f2 = sec(f1);    f2 = cot(f1);
f2 = asin(f1);  f2 = acos(f1);   f2 = atan(f2);
f2 = atan2(f1, f2);
std::tie(f1, f2) = sincos(f1);

/* Hyperbolic and inverse hyperbolic functions */
f2 = sinh(f1);  f2 = cosh(f1);  f2 = tanh(f1);
f2 = csch(f1);  f2 = sech(f1);  f2 = coth(f1);
f2 = asinh(f1); f2 = acosh(f1); f2 = atanh(f2);
std::tie(f1, f2) = sincosh(f1);

/* Exponential function, natural logarithm, power function */
f2 = exp(f1);   f2 = log(f1);   f2 = pow(f1, f2);

/* Error function and its inverse */
f2 = erf(f1);   f2 = erfi(f1);

/* Exponent/mantissa manipulation */
f1 = ldexp(f1, f2);
std::tie(f1, f2) = frexp(f1);

/* Bit shifts and rotations (only integer arrays) */
i1 = sli<3>(i1);   i1 = sri<3>(i1);   /* Shift by a compile-time constant ("immediate") */
i1 = i1 >> i2;     i1 = i1 << i2;     /* Element-wise shift by a variable amount */
i1 = roli<3>(i1);  i1 = rori<3>(i1);  /* Rotate by a compile-time constant ("immediate") */
i1 = rol(i1, i2);  i1 = ror(i1, i2);  /* Element-wise rotation by a variable amount */
```

#### Shuffle

Components of a vector can be reordered using the following syntax:

```
f2 = shuffle<0, 2, 1, 4>(f1);
```

#### Horizontal operations

In contrast to the above vertical operations, the following *horizontal*
operations consider the entries of a packed array jointly and return a scalar.
Depending on the size of the array, these are implemented using between
log<sub>2</sub>(*N*) and *N*-1 vertical reduction operations and shuffles.
Horizontal operations should generally be avoided since they don't fully
utilize the hardware vector units (ways of avoiding them are discussed later).

```cpp
/* Horizontal sum, equivalent to f1[0] + f1[1] + f1[2] + f1[3] */
float s0 = hsum(f1);

/* Horizontal product, equivalent to f1[0] * f1[1] * f1[2] * f1[3] */
float s1 = hprod(f1);

/* Horizontal minimum, equivalent to std::min({ f1[0], f1[1], f1[2], f1[3] }) */
float s2 = hmin(f1);

/* Horizontal maximum, equivalent to std::max({ f1[0], f1[1], f1[2], f1[3] }) */
float s3 = hmax(f1);
```

The following linear algebra primitives are also realized in terms of horizontal operations:

```cpp
/* Dot product of two float arrays */
float dp = dot(f1, f2);

/* For convenience: absolute value of the dot product */
float adp = abs_dot(f1, f2);

/* Squared 2-norm of a vector */
float sqn = squared_norm(f1);

/* 2-norm of a vector */
float nrm = norm(f1);
```

#### Mask and bit-level operations

Comparisons involving Enoki types are generally applied component-wise and
produce a *mask*, i.e. a packed array that will have all bits set to ``1`` for
each entry where the comparison was true, and ``0`` everywhere else. These
masks don't make sense as floating point or integer values, but they enable
powerful branchless logic in comparison with a range of other bit-level
operations. Note that the AVX512 back-end is special and instead uses eight
dedicated mask registers to store masks compactly (allocating only a single bit
per mask entry). Such tedious differences between platforms are invisible in
user code that uses the abstractions of Enoki.

The following snippets show some example usage of mask types:

```cpp
auto mask = f1 > 1;

/* Bit-level and operation: Zero out entries where the comparison was false */
f1 &= mask;
```

Masks can be combined in various ways

```cpp
mask ^= (f1 > cos(f2)) | ~(f2 <= f1);
```

The following range tests also generate masks

```cpp
mask = isnan(f1);    /* Per-component NaN test */
mask = isinf(f1);    /* Per-compoment +/- infinity test */
mask = isfinite(f1); /* Per-component test for finite values */
```

As with floating point values, there are also horizontal operations for masks:

```cpp
/* Do all entries have a mask value corresponding to 'true'? */
bool mask_all_true  = all(mask);

/* Do some entries have a mask value corresponding to 'true'? */
bool mask_some_true = any(mask);

/* Do none of the entries have a mask value corresponding to 'true'? */
bool mask_none_true = none(mask);

/* Count how many entries have a mask value corresponding to 'true'? */
size_t true_count = count(mask);

```

Following the principle of least surprise, ``operator==`` and ``operator!=``
are horizontal operations that return a boolean value; vertical alternatives
named ``eq()`` and ``neq()`` are also available. The following pairs of operations
are equivalent:

```cpp
bool b1 = (f1 == f2);
bool b2 = all(eq(f1, f2));

bool b3 = (f1 != f2);
bool b4 = any(neq(f1, f2));
```

One of the most useful bit-level operation is ``select()`` which chooses
between two arguments using a mask.  This is extremely useful for writing
branch-free code.  Argument order matches the C ternary operator, i.e.
``condition ? true_value : false_value``.

```cpp
f1 = select(f1 < 0, f1, f2);

/* The above select() statement is equivalent to the following less efficient expression */
f1 = ((f1 < 0) & f1) | (~(f1 < 0) & f2);
```

#### Casting, half precision

Enoki supports conversion between any pair of types using builtin constructors:

```cpp
using Source = Array<int64_t, 32>;
using Target = Array<double, 32>;

Source source = ...;
Target target(source);
```

Enoki reduces the conversion to accelerated hardware operations whenever
possible while hiding ugly platform-specific details--for instance, machines
with AVX but no AVX2 don't have an 8-wide integer vector unit. This means that
an ``Array<float, 8>`` can be represented using a single AVX "ymm" register,
but casting it to an ``Array<int32_t, 8>`` entails switching to a pair of half
width SSE4.2 "xmm" integer registers, etc. In addition to all of the standard
types, fast vectorized conversions between single precision values and a
special 16-bit half precision floating point type ``enoki::half`` are available
if the host processor supports the F16C instruction set.

When the types have matched sizes and layouts, it is also possible to
reinterpret the bit-level representation as a different type:

```cpp
using Source = Array<int64_t, 32>;
using Target = Array<double, 32>;

Source source = /* ... integer vector which makes sense when interpreted as a double value ... */;
Target target = reinterpret<Target>(source);
```

#### The histogram problem and conflict detection

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

Enoki provides a function named ``apply``, which modifies an indirect memory
location in a way that is not susceptible to conflicts. The function takes an
arbitrary function as parameter and applies it to the specified memory
location, which allows this approach to generalize to situations other than
just building histograms.

```
apply<Float>(hist, indices, [](auto x) { return x + 1; });
```

Internally, ``apply`` detects and processes conflicts using the AVX512CDI
instruction set. When conflicts are present, the function provided as an
argument may be applied multiple times in a row. When AVX512CDI is not
available, a (slower) scalar fallback implementation is used.


#### Compressing arrays

It is sometimes helpful to be able to selectively write only the masked parts
of an array so that they become densely packed in memory. The
``store_compress`` function efficiently maps this operation onto the targeted
hardware. The function also automatically advances the pointer by the amount
of written entries.

```cpp
float *mem;
auto mask = f1 < 0;
store_compress(mem, f1, mask);
mem += count(mask);
...
```

#### The special case of 3-Vectors

Because information with ``Size == 3`` occurs frequently (3D coordinates, color
information, ...) and generally also benefits very slightly from vectorization,
there is a special case which processes 3-vectors in packed float arrays of
size 4, leaving the last component unused. Any vertical operations are applied
to the entire float array including the fourth component, while horizontal
operations ignore the last component. An efficient cross product operation
realized using shuffles is available for 3-vectors:

```cpp
f1 = cross(f1, f2);
```

Generally, a better way to work with 3D data while achieving much greater
instruction level parallelism is via the SoA approach discussed next.

## Structure of Arrays (SoA), or "vectorized vectors"

Application data is often not arranged in a way that is conductive to efficient
vectorization. For instance, vectors in a 3D dataset have too few dimensions to
fully utilize the SIMD lanes of modern hardware. Scalar or horizontal
operations like dot products lead to similar inefficiencies if used frequently.
In such situations, it is preferable to use a *Structure of Arrays* (SoA)
approach, which provides a way of converting scalar, horizontal, and
low-dimensional vector arithmetic into vertical operations with 100% SIMD lane
utilization.

Let's consider the following scalar example code, which computes the normalized
cross product of a pair of vectors:

```cpp
struct Vector3f {
   float x;
   float y;
   float z;

   friend Vector3f cross(const Vector3f &v1, const Vector3f &v2) {
       return Vector3f{
           (v1.y * v2.z) - (v1.z * v2.y),
           (v1.z * v2.x) - (v1.x * v2.z),
           (v1.x * v2.y) - (v1.y * v2.x)
       };
   }

   friend Vector3f normalize(const Vector3f &v) {
       float scale = 1.0f / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
       return Vector3f{v.x * scale, v.y * scale, v.z * scale};
   }
};

Vector3f test(Vector3f a, Vector3f b) {
    return normalize(cross(a, b));
}
```

Clang compiles the ``test()`` function into the following fairly decent scalar
assembly (``clang -O3 -msse4.2 -mfma -ffp-contract=fast -fomit-frame-pointer``):

```Assembly
__Z4test8Vector3fS_:
        vmovshdup   xmm4, xmm0           ## xmm4 = xmm0[1, 1, 3, 3]
        vmovshdup   xmm5, xmm2           ## xmm5 = xmm2[1, 1, 3, 3]
        vinsertps   xmm6, xmm3, xmm1, 16 ## xmm6 = xmm3[0], xmm1[0], xmm3[2, 3]
        vblendps    xmm7, xmm2, xmm0, 2  ## xmm7 = xmm2[0], xmm0[1], xmm2[2, 3]
        vpermilps   xmm7, xmm7, 225      ## xmm7 = xmm7[1, 0, 2, 3]
        vinsertps   xmm1, xmm1, xmm3, 16 ## xmm1 = xmm1[0], xmm3[0], xmm1[2, 3]
        vblendps    xmm3, xmm0, xmm2, 2  ## xmm3 = xmm0[0], xmm2[1], xmm0[2, 3]
        vpermilps   xmm3, xmm3, 225      ## xmm3 = xmm3[1, 0, 2, 3]
        vmulps      xmm1, xmm1, xmm3
        vfmsub231ps xmm1, xmm6, xmm7
        vmulss      xmm2, xmm4, xmm2
        vfmsub231ss xmm2, xmm0, xmm5
        vmovshdup   xmm0, xmm1           ## xmm0 = xmm1[1, 1, 3, 3]
        vmulss      xmm0, xmm0, xmm0
        vfmadd231ss xmm0, xmm1, xmm1
        vfmadd231ss xmm0, xmm2, xmm2
        vsqrtss     xmm0, xmm0, xmm0
        vmovss      xmm3, dword ptr [rip + LCPI0_0] ## xmm3 = 1.f, 0.f, 0.f, 0.f
        vdivss      xmm3, xmm3, xmm0
        vmovsldup   xmm0, xmm3           ## xmm0 = xmm3[0, 0, 2, 2]
        vmulps      xmm0, xmm1, xmm0
        vmulss      xmm1, xmm2, xmm3
        ret
```

However, note that only 50% of the above 22 instruction perform actual
arithmetic (which is scalar, i.e. low throughput), with the remainder being
spent on unpacking and re-shuffling data. Simply rewriting this code using
Enoki leads to considerable improvements:

```cpp
/* Enoki version */
using Vector3f = Array<float, 3>;

Vector3f test(Vector3f a, Vector3f b) {
    return normalize(cross(a, b));
}
```

```Assembly
# Assembly for Enoki version
__Z4test8Vector3fS_:
    vpermilps   xmm2, xmm0, 201 ## xmm2 = xmm0[1, 2, 0, 3]
    vpermilps   xmm3, xmm1, 210 ## xmm3 = xmm1[2, 0, 1, 3]
    vpermilps   xmm0, xmm0, 210 ## xmm0 = xmm0[2, 0, 1, 3]
    vpermilps   xmm1, xmm1, 201 ## xmm1 = xmm1[1, 2, 0, 3]
    vmulps      xmm0, xmm0, xmm1
    vfmsub231ps xmm0, xmm2, xmm3
    vdpps       xmm1, xmm0, xmm0, 113
    vsqrtss     xmm1, xmm1, xmm1
    vmovss      xmm2, dword ptr [rip + LCPI0_0] ## xmm2 = 1.f, 0.f, 0.f, 0.f
    vdivss      xmm1, xmm2, xmm1
    vpermilps   xmm1, xmm1, 0   ## xmm1 = xmm1[0, 0, 0, 0]
    vmulps      xmm0, xmm0, xmm1
    ret
```

This is better but still not ideal: of the 12 instructions (a reduction by 50%
compared to the previous example), 3 are vectorized, 2 are scalar, and 1 is a
(slow) horizontal reduction. The remaining 6 are shuffle and move instructions.

The key idea that enables further vectorization of this code is to work on
groups of vectors at once, whose components are stored contiguously in memory
(or registers). This is known as SoA-style data organization. One chunk of
component entries is referred to as a *packet*.

Since Enoki arrays support arbitrary nesting, it's straightforward to wrap an
existing ``Array`` representing a packet of data into another array with the
semantics of an ``N``-dimensional vector. As before, all mathematical
operations discussed so far are supported by simply forwarding them to the
underlying packet type. The following snippet demonstrates the basic usage of
such an approach.

```cpp
/* Declare an underlying packet type with 4 floats (let's try non-approximate math mode first) */
using FloatP = Array<float, 4, /* Approx = */ false>;

/* NEW: Packet containing four separate three-dimensional vectors */
using Vector3fP = Array<FloatP, 3>;

Vector3fP vec(
   FloatP(1, 2, 3, 4),    /* X components */
   FloatP(5, 6, 7, 8),    /* Y components */
   FloatP(9, 10, 11, 12)  /* Z components */
);

/* Enoki's stream insertion operator automatically detects the 
   SoA case and (more conveniently) prints the array contents as 
   "[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]" */
std::cout << vec << std::endl;

/* Element access using operator[] and x()/y()/z()/w() now return size-4 packets */
vec.x() = vec[1];

/* Arithmetic operations applied to all components */
vec += vec * sin(vec);

/* Initialize component packets with constant values */
Vector3fP vec2(1.f, 2.f, 3.f);

/* Dot product (caution: this now creates a size-4 packet of
   dot products corresponding to each of the 3D vectors) */
FloatP dp = dot(vec, vec2);

.. etc ..

```

With the above type aliases, the ``test()`` function now looks as
follows:

```cpp
Vector3fP test(Vector3fP a, Vector3fP b) {
    return normalize(cross(a, b));
}
```

Disregarding the loads and stores that are needed to fetch the operands and
write the results, this generates the following assembly:

```Assembly
# Assembly for SoA-style version
__Z4test8Vector3fS_:
    vmulps       xmm6, xmm2, xmm4
    vfmsub231ps  xmm6, xmm1, xmm5
    vmulps       xmm5, xmm0, xmm5
    vfmsub213ps  xmm2, xmm3, xmm5
    vmulps       xmm1, xmm1, xmm3
    vfmsub231ps  xmm1, xmm0, xmm4
    vmulps       xmm0, xmm2, xmm2
    vfmadd231ps  xmm0, xmm6, xmm6
    vfmadd231ps  xmm0, xmm1, xmm1
    vsqrtps      xmm0, xmm0
    vbroadcastss xmm3, dword ptr [rip + LCPI0_0]
    vdivps       xmm0, xmm3, xmm0
    vmulps       xmm3, xmm6, xmm0
    vmulps       xmm2, xmm2, xmm0
    vmulps       xmm0, xmm1, xmm0
```

This is *much* better: 15 vectorized operations which process four vectors at
the same time, while fully utilizing the underlying SSE4.2 vector units. If
wider vector units are available, it's of course possible to process many more
vectors at the same time. Enoki will also avoid costly high-latency operations
like division and square root if the user indicates that minor approximations
are tolerable. The following snippet demonstrates how to simultaneously process
16 vectors on a machine which supports the AVX512ER instruction set:

```cpp
/* Packet of 16 single precision floats (approximate mode now enabled) */
using FloatP = enoki::Array<float, 16>;

/* Packet of 16 3D vectors */
using Vector3fP = Vector<FloatP, 3>;

Vector3fP test(Vector3fP a, Vector3fP b) {
    return normalize(cross(a, b));
}
```

```Assembly
# Assembly for AVX512ER SoA-style version
__Z4test8Vector3fS_:
    vmulps       zmm6, zmm2, zmm4
    vfmsub231ps  zmm6, zmm1, zmm5
    vmulps       zmm5, zmm0, zmm5
    vfmsub213ps  zmm2, zmm3, zmm5
    vmulps       zmm1, zmm1, zmm3
    vfmsub231ps  zmm1, zmm0, zmm4
    vmulps       zmm0, zmm2, zmm2
    vfmadd231ps  zmm0, zmm6, zmm6
    vfmadd231ps  zmm0, zmm1, zmm1
    vrsqrt28ps   zmm0, zmm0        # <-- Fast reciprocal square root instruction
    vmulps       zmm3, zmm6, zmm0
    vmulps       zmm2, zmm2, zmm0
    vmulps       zmm0, zmm1, zmm0
```

Note that it can be advantageous to use an integer multiple of the system's
SIMD width (e.g. 2x) to further increase the amount of arithmetic that occurs
between memory accesses. In the above example, Enoki would then unroll every
32x-wide operation into a pair of 16x-wide AVX512 instructions.

While the ``Array`` class makes it straightforward to implement SoA-style
vectors types, it can be awkward to work with (fairly small) fixed packet
sizes---the number of data points that will need to be processed in actual
applications is generally much larger and on the order of thousands or
millions, i.e. too large to be processed using registers and branch&loop-free
code. The remainder of this document therefore discusses classes that can be
used to realize SoA-style computations over arrays with arbitrary lengths that
are dynamically allocated.

## Dynamic vectors: ``DynamicArray<Packet>``

``DynamicArray`` is a smart pointer class that manages the lifetime of a
dynamically allocated memory region. It is the exclusive owner of this data and
is also responsible for its destruction when the dynamic array goes out of
scope (the class thus resembles ``std::unique_ptr`` in C++11 somewhat). Dynamic
arrays can be used to realize arithmetic involving data that is much larger
than the maximum SIMD width supported by the underlying hardware (e.g. arrays
with thousands or millions of entries).

The class requires a (static) ``Array<>`` type as a template parameter named
``Packet``. This packet type will later be used to realize vectorized
computations involving the dynamic array's contents.

```cpp
/* Static float array ("P" means packet) */
using FloatP = Array<float>;

/* Dynamic float array (vectorized via FloatP, "X" means arbitrarily many) */
using FloatX = DynamicArray<FloatP>;
```

Note that ``DynamicArray`` is a holder type that *cannot* be used for
arithmetic operations (for instance, there is no ``operator+`` to add two
``DynamicArray`` instances). Efficient implementation of such operators for
dynamic memory regions generally involves a technique named [expression
templates](https://en.wikipedia.org/wiki/Expression_templates)---unfortunately,
this kind of approach tends to produce intermediate code with an extremely
large number of common subexpressions that exceeds the capabilities of the
*common subexpression elimination* (CSE) stage of current compilers (the first
version of Enoki in fact used expression templates, and it was due to the
difficulties with them that an alternative was developed).

Vectorized computations involving dynamic arrays are implemented using the
``vectorize`` function, which implements a sliding window over a larger amount
of data. ``vectorize`` takes a computational kernel as the first argument,
which is generally a polymorphic lambda function, i.e. a lambda function that
uses the ``auto`` keyword to leave the specific type of its arguments
undefined. Arguments that are read should use the ``auto`` keyword, while
arguments that are written (and potentially also read) should use the
``auto&&`` keyword. The remaining parameters to ``vectorize`` are dynamic
arrays  that are processed by the kernel (or other parameters which are simply
forwarded). The number of these parameters must exactly match the number of
lambda function parameters. ``vectorize`` then automatically instantiates the
kernel for the underlying packet type and executes it as many times as is
necessary to process all of the data.

For instance, the following snippet adds a scaled multiple of a dynamic float
array to another dynamic float array (their sizes must match):

```cpp
void scaleAdd(const FloatX &vec1, float scale, FloatX &vec2) {
    vectorize(
        /* Function to execute over vector arguments */
        [](auto v1, float s, auto &&v2) { v2 += s * v1; },

        /* Specify input arguments to above lamba fct. */
        vec1, scale, vec2
    );
}
```

It is not necessary to "route" all parameters through vectorize. Auxiliary data
structures or constants are easily accessible via the lamda capture object using
the ``[&]`` notation:

```cpp
void scale(FloatX &vec, float scale) {
    /* Scale all entries of 'vec' */
    vectorize(
        [&](auto &&v) { v *= scale; },
        vec
    );
}
```

## Putting everything together: Vectorized dynamic vectors

The final combination that is likely the most useful in practice entails
building SoA-style vectors from dynamic arrays. The following list shows the
various possible type definitions -- the last entry is the one of interest.

```cpp
/* FloatP: Packet of floats (size based on hardware capabilities) */
using FloatP = Array<float>;

/* Dynamic float array ("X" means arbitrarily many, vectorized via FloatP) */
using FloatX = DynamicArray<FloatP>;

/* Single 3D vector */
using Vector3f = Array<float, 3>;

/* Fixed size packet of 3D vectors (SoA data organization, vectorized using FloatP) */
using Vector3fP = Vector<FloatP, 3>;

/* === NEW ===: Arbitrarily many 3D vectors (SoA data organization, vectorized using FloatP) */
using Vector3fX = Vector<FloatX, 3>;
```

The prior example of computing a normalized cross product (plus some extra code
for illustrative purposes) now looks as follows:

```cpp
Vector3fX vec1 = Vector3fX::Zero(1000),
          vec2 = Vector3fX::Constant(1000, 0.f),
          result;

/* The scatter() and gather() methods can be used to convert between AoS and
   SoA representations. This step resembles a big matrix transpose and is not
   particularly efficient. */
for (size_t i=0; i<vec1.size(); ++i) {
    Vector3f data = vec1.gather(i);
    vec1.scatter(i, Vector3f(....));
    vec2.scatter(i, Vector3f(....));
}

/* Ensure that the output array has the right size */
result.resize(vec1.size());

vectorize(
    [](auto v1, auto v2, auto &&r) {
       r = normalize(cross(v1, v2));
    },
    vec1, vec2, result
);
```

## Supporting new types

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

    - Scatter/gather operations (emulated using scalar operations by default):
      ``scatter_``, ``gather_``.

    - Prefetch operations (no-op by default): ``prefetch_``.

    - Trigonometric and hyperbolic functions: ``sin_``, ``sinh_``, ``sincos_``,
      ``sincosh_``, ``cos_``, ``cosh_``, ``tan_``, ``tanh_``, ``csc_``,
      ``csch_``, ``sec_``, ``sech_``, ``cot_``, ``coth_``, ``asin_``,
      ``asinh_``, ``acos_``, ``acosh_``, ``atan_``, ``atanh_``.

    - Fused multiply-add routines (reduced to ``add_``/``sub_`` and ``mul_`` by
      default): ``fmadd_``, ``fmsub_``.

    - Reciprocal and reciprocal square root (reduced to ``div_`` and ``sqrt_``
      by default): ``rcp_``, ``rsqrt_``.

    - Dot product (reduced to ``mul_`` and ``hsum_`` by default): ``dot_``.

    - Exponentials, logarithms, powers, floating point exponent manipulation
      functions: ``log_``, ``exp_``, ``pow_`` ``frexp_``, ``ldexp_``.

    - Error function and its inverse: ``erf_``, ``erfi_``.

    - Optional bit-level rotation operations (reduced to shifts by default):
      ``rol_``, ``roli_``, ``rolv_``, ``ror_``, ``rori_``, ``rorv_``.

