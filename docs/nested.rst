Nested arrays
=============

Motivation
----------

Application data is often not arranged in a way that is conductive to
efficient vectorization. For instance, vectors in a 3D dataset have too few
dimensions to fully utilize the SIMD lanes of modern hardware. Scalar or
horizontal operations like dot products lead to similar inefficiencies if
used frequently. In such situations, it is preferable to use *nested* arrays
using a technique that is known as a *Structure of Arrays* (SoA)
representation, which provides a way of converting scalar, horizontal, and
low-dimensional vector arithmetic into vertical operations that fully
utilize all SIMD lanes.

To understand the fundamental problem, consider the following basic example
code, which computes the normalized cross product of a pair of 3D vectors.
Without Enoki, this might be done as follows:

.. code-block:: cpp

    struct Vector3f {
       float x;
       float y;
       float z;
    };

   Vector3f normalize(const Vector3f &v) {
       float scale = 1.0f / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
       return Vector3f{v.x * scale, v.y * scale, v.z * scale};
   }

   Vector3f cross(const Vector3f &v1, const Vector3f &v2) {
       return Vector3f{
           (v1.y * v2.z) - (v1.z * v2.y),
           (v1.z * v2.x) - (v1.x * v2.z),
           (v1.x * v2.y) - (v1.y * v2.x)
       };
   }

    Vector3f test(Vector3f a, Vector3f b) {
        return normalize(cross(a, b));
    }

Clang compiles the ``test()`` function into the following fairly decent scalar
assembly (``clang -O3 -msse4.2 -mfma -ffp-contract=fast -fomit-frame-pointer``):

.. code-block:: nasm

    __Z4test8Vector3fS_:
            vmovshdup   xmm4, xmm0           ; xmm4 = xmm0[1, 1, 3, 3]
            vmovshdup   xmm5, xmm2           ; xmm5 = xmm2[1, 1, 3, 3]
            vinsertps   xmm6, xmm3, xmm1, 16 ; xmm6 = xmm3[0], xmm1[0], xmm3[2, 3]
            vblendps    xmm7, xmm2, xmm0, 2  ; xmm7 = xmm2[0], xmm0[1], xmm2[2, 3]
            vpermilps   xmm7, xmm7, 225      ; xmm7 = xmm7[1, 0, 2, 3]
            vinsertps   xmm1, xmm1, xmm3, 16 ; xmm1 = xmm1[0], xmm3[0], xmm1[2, 3]
            vblendps    xmm3, xmm0, xmm2, 2  ; xmm3 = xmm0[0], xmm2[1], xmm0[2, 3]
            vpermilps   xmm3, xmm3, 225      ; xmm3 = xmm3[1, 0, 2, 3]
            vmulps      xmm1, xmm1, xmm3
            vfmsub231ps xmm1, xmm6, xmm7
            vmulss      xmm2, xmm4, xmm2
            vfmsub231ss xmm2, xmm0, xmm5
            vmovshdup   xmm0, xmm1           ; xmm0 = xmm1[1, 1, 3, 3]
            vmulss      xmm0, xmm0, xmm0
            vfmadd231ss xmm0, xmm1, xmm1
            vfmadd231ss xmm0, xmm2, xmm2
            vsqrtss     xmm0, xmm0, xmm0
            vmovss      xmm3, dword ptr [rip + LCPI0_0] ; xmm3 = 1.f, 0.f, 0.f, 0.f
            vdivss      xmm3, xmm3, xmm0
            vmovsldup   xmm0, xmm3           ; xmm0 = xmm3[0, 0, 2, 2]
            vmulps      xmm0, xmm1, xmm0
            vmulss      xmm1, xmm2, xmm3
            ret

However, note that only 50% of the above 22 instruction perform actual
arithmetic (which is scalar, i.e. low throughput), with the remainder being
spent on unpacking and re-shuffling data.

A first attempt
---------------

Simply rewriting this code using Enoki leads to considerable improvements:

.. code-block:: cpp

    /* Enoki version */
    using Vector3f = Array<float, 3>;

    Vector3f test(Vector3f a, Vector3f b) {
        return normalize(cross(a, b));
    }

.. code-block:: nasm

    ; Assembly for Enoki version
    __Z4test8Vector3fS_:
        vpermilps   xmm2, xmm0, 201 ; xmm2 = xmm0[1, 2, 0, 3]
        vpermilps   xmm3, xmm1, 210 ; xmm3 = xmm1[2, 0, 1, 3]
        vpermilps   xmm0, xmm0, 210 ; xmm0 = xmm0[2, 0, 1, 3]
        vpermilps   xmm1, xmm1, 201 ; xmm1 = xmm1[1, 2, 0, 3]
        vmulps      xmm0, xmm0, xmm1
        vfmsub231ps xmm0, xmm2, xmm3
        vdpps       xmm1, xmm0, xmm0, 113
        vsqrtss     xmm1, xmm1, xmm1
        vmovss      xmm2, dword ptr [rip + LCPI0_0] ; xmm2 = 1.f, 0.f, 0.f, 0.f
        vdivss      xmm1, xmm2, xmm1
        vpermilps   xmm1, xmm1, 0   ; xmm1 = xmm1[0, 0, 0, 0]
        vmulps      xmm0, xmm0, xmm1
        ret

Enoki uses SSE4.2 instructions that "waste" the last component, leading to more
compact code in this case. This is better but still not ideal: of the 12
instructions (a reduction by 50% compared to the previous example), 3 are
vectorized, 2 are scalar, and 1 is a (slow) horizontal reduction. The remaining
6 are shuffle and move instructions.

A better solution
-----------------

The key idea that enables further vectorization of this code is to work on 3D
arrays, whose components are themselves arrays. This is known as SoA-style data
organization. One group of multiple 3D vectors represented in this way is
referred to as a *packet*.

.. image:: nested-01.svg
    :width: 400px
    :align: center

Since Enoki arrays support arbitrary nesting, it's straightforward to wrap an
existing ``Array`` representing a packet of data into another array with the
semantics of an ``N``-dimensional vector. As before, all mathematical
operations discussed so far are trivially supported due to the semantics of an
Enoki array: all operations are simply forwarded to the contained entries
(which are themselves arrays now, so the procedure continues recursively). The
following snippet demonstrates the basic usage of such an approach.

.. code-block:: cpp

    /* Declare an underlying packet type with 4 floats (let's try non-approximate math mode first) */
    using FloatP = Array<float, 4, /* Approx = */ false>;

    /* NEW: Packet containing four separate three-dimensional vectors */
    using Vector3fP = Array<FloatP, 3>;

    Vector3fP vec(
       FloatP(1, 2, 3, 4),    /* X components */
       FloatP(5, 6, 7, 8),    /* Y components */
       FloatP(9, 10, 11, 12)  /* Z components */
    );

    /* Enoki's stream insertion operator detects the recursive array and
       prints the contents as a list of 3D vectors
       "[[1, 5, 9],
         [2, 6, 10],
         [3, 7, 11],
         [4, 8, 12]]" */
    std::cout << vec << std::endl;

    /* Element access using operator[] and x()/y()/z()/w() now return size-4 packets */
    vec.x() = vec[1];

    /* Transcendental functions applied to all components */
    Vector3fP vec2 = sin(vec);

The behavior of horizontal operations changes as well--for instance, the dot
product

.. code-block:: cpp

    FloatP dp = dot(vec, vec2);

now creates a size-4 packet of dot products: one for each pair of input 3D
vectors. This is simply a consequence of applying the definition of the dot
product to the components of the array (which are now arrays). Another
consequence is that an inefficient horizontal operation was converted into a
series of vertical operations that make better use of the processor's vector
units.

.. image:: nested-02.svg
    :width: 600px
    :align: center

With the above type aliases, the ``test()`` function now looks as
follows:

.. code-block:: cpp

    Vector3fP test(Vector3fP a, Vector3fP b) {
        return normalize(cross(a, b));
    }

Disregarding the loads and stores that are needed to fetch the operands and
write the results, this generates the following assembly:

.. code-block:: nasm

    ; Assembly for SoA-style version
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

This is *much* better: 15 vectorized operations which process four vectors at
the same time, while fully utilizing the underlying SSE4.2 vector units. If
wider arithmetic is available, it's of course possible to process many more
vectors at the same time.

Enoki will also avoid costly high-latency operations like division and square
root if the user indicates that minor approximations are tolerable. The
following snippet demonstrates how to simultaneously process 16 vectors on a
machine which supports the AVX512ER instruction set:

.. code-block:: cpp

    /* Packet of 16 single precision floats (approximate mode now enabled) */
    using FloatP = Array<float, 16>;

    /* Packet of 16 3D vectors */
    using Vector3fP = Vector<FloatP, 3>;

    Vector3fP test(Vector3fP a, Vector3fP b) {
        return normalize(cross(a, b));
    }

.. code-block:: nasm

    ; Assembly for AVX512ER SoA-style version
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
        vrsqrt28ps   zmm0, zmm0        ; <-- Fast reciprocal square root instruction
        vmulps       zmm3, zmm6, zmm0
        vmulps       zmm2, zmm2, zmm0
        vmulps       zmm0, zmm1, zmm0

Note that it can be advantageous to use an integer multiple of the system's
SIMD width (e.g. 2x) to further increase the amount of arithmetic that
occurs between memory accesses. In the above example, Enoki would then
unroll every 32x-wide operation into a pair of 16x-wide AVX512 instructions.

Nested horizontal operations
----------------------------

It was mentioned earlier that horizontal operations involving nested arrays
return arrays instead of scalars (the same is also true for horizontal mask
operations such as :cpp:func:`any`).

Sometimes this is not desirable, and Enoki thus also provides nested versions
all of horizontal operations that can be accessed via the ``_nested`` suffix.
These functions recursively apply horizontal reductions until the result ceases
to be an array. For instance, the following function ensures that no element of
a packet of 3-vectors contains a Not-a-Number floating point value.

.. code-block:: cpp

    bool check(Vector3fP x) {
        return none_nested(isnan(x));
    }
