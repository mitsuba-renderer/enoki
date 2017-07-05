.. image:: enoki-logo.svg
    :width: 400px
    :align: center

Introduction
============

**Enoki** is a C++ template library that enables transparent vectorization of
numerical code. It is implemented as a set of header files with no dependencies
other than a sufficiently C++14-capable compiler (GCC, Clang, Intel C++
Compiler 2016, Visual Studio 2017). Enoki code reduces to efficient SIMD
instructions available on modern processor architectures (AVX512, AVX2, AVX,
FMA, BMI, F16C, and SSE4.2), and it generates scalar fallback code if no vector
instructions are present.

Motivation
----------

The development of this library was prompted by the by the author's frustration
with the current vectorization landscape:

1. Auto-vectorization in state-of-the-art compilers is inherently local. A
   computation whose call graph spans separate compilation units (e.g. multiple
   shared libraries) simply can't be vectorized.

2. Data structures must be converted into a *Structure of Arrays* (SoA) layout
   to be eligible for vectorization.

   .. image:: intro-01.svg
       :width: 400px
       :align: center

   This is analogous to performing a matrix transpose of an application's
   entire memory layout---an intrusive change that is likely to touch almost
   every line of code.

3. Parts of the application likely have to be rewritten using `intrinsic
   instructions <https://software.intel.com/sites/landingpage/IntrinsicsGuide>`_,
   which is going to look something like this:

   .. image:: intro-02.svg
       :width: 400px
       :align: center

   Intrinsics-heavy code is challenging to read and modify once written, and
   it is inherently non-portable.

4. Vectorized transcendental functions (*exp*, *sin*, *cos*, ..) are not widely
   available. Intel and AMD provide proprietary implementations, but most
   compilers don't include them by default.

5. It is desirable to retain both scalar and vector versions of an algorithm,
   but ensuring their consistency throughout the development cycle becomes a
   maintenance nightmare.

6. *Domain-specific languages* (DSLs) for vectorization such as `ISPC
   <https://ispc.github.io>`_ address many of the above issues but assume that
   the main computation underlying an application can be condensed into a
   compact kernel that is implementable using the limited language subset of
   the DSL.

   This is not the case for complex applications, where the "kernel" may be
   spread out over many separate modules involving high-level language features
   such as functional or object-oriented programming.

What Enoki does differently
---------------------------

Enoki addresses these issues and provides a *complete* solution for vectorizing
modern C++ applications with nontrivial control flow and data structures,
dynamic memory allocation, virtual function calls, and vector calls across
module boundaries. It has the following design goals:

1. **Unobtrusive**: Only minor modifications are necessary to convert existing
   C++ code into its Enoki-vectorized equivalent, which remains readable and
   maintainable.

2. **No code duplication**: It is generally desirable to provide both scalar
   and vectorized versions of an API, e.g. for debugging, and to preserve
   compatibility with legacy code. Enoki code extensively relies on class and
   function templates to achieve this goal without any code duplication---the
   same implementation works with both scalar and vector arguments.

3. **Complex data structures**: Converting complex data structures to SoA
   layout is a breeze using Enoki. All the hard work is handled by the C++14
   type system.

4. **Function calls**: vectorized calls to functions in other compilation units
   (e.g. a dynamically loaded plugin) are possible. Enoki can even vectorize
   method or virtual method calls (e.g. ``instance->my_function(arg1, arg2,
   ...);`` when ``instance`` turns out to be a SIMD array of instances).

5. **Transcendentals**: Enoki provides branch-free vectorized elementary and
   transcendental functions, including *cos*, *sin*, *sincos*, *tan*, *csc*,
   *sec*, *cot*, *acos*, *asin*, *atan*, *atan2*, *exp*, *log*, *pow*, *sinh*,
   *cosh*, *sincosh*, *tanh*, *csch*, *sech*, *coth*, *asinh*, *acosh*,
   *atanh*, *i0e*, *frexp*, *ldexp*, *erf*, and *erfinv*.

   .. image:: intro-03.png
       :width: 720px
       :align: center

   They are slightly less accurate than their standard C math library
   counterparts: depending on the function, the approximations have an average
   relative error between 0.1 and 4 ULPs. The C math library can be used as a
   fallback when higher precision transcendental functions are needed.

6. **Portability**: Enoki supports arbitrary array sizes that don't necessarily
   match what is supported by the underlying hardware (e.g. 16 x single
   precision on a machine whose SSE vector only has hardware support for 4 x
   single precision operands). The library uses template metaprogramming
   techniques to efficiently map array expressions onto the available hardware
   resources. This greatly simplifies development because it's enough to write
   a single implementation of a numerical algorithm that can then be deployed
   on any target architecture. There are non-vectorized fallbacks for
   everything, thus programs will run even on unsupported architectures (albeit
   without the performance benefits of vectorization).

7. **Modular architecture**: Enoki is split into two major components: the
   front-end provides various high-level array operations, while the back-end
   provides the basic ingredients that are needed to realize these operations
   using the SIMD instruction set(s) supported by the target architecture.

   The back-end makes heavy use of SIMD intrinsics to ensure that compilers
   generate efficient machine code. The intrinsics are contained in separate
   back-end header files (e.g. ``array_avx.h`` for AVX intrinsics), which
   provide rudimentary arithmetic and bit-level operations. Fancier operations
   (e.g. *atan2*) use the back-ends as an abstract interface to the hardware,
   which means that it's simple to support other instruction sets such as a
   hypothetical future AVX1024 or even an entirely different architecture (e.g.
   ARM) by just adding a new back-end.

9. **License**: Enoki is available under a non-viral open source license
   (3-clause BSD).

The project is named after `Enokitake <https://en.wikipedia.org/wiki/Enokitake>`_,
a type of mushroom with many long and parallel stalks reminiscent of data flow
in SIMD arithmetic.
