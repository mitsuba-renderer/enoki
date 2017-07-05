.. cpp:namespace:: enoki

Operations
==========

The following operations are available in the ``enoki`` namespace.

Memory operations
-----------------

.. cpp:function:: template <typename Array> Array load(const void *mem, mask_t<Array> mask = true)

    Loads an array of type ``Array`` from the memory address ``mem`` (which is
    assumed to be aligned on a multiple of ``alignof(Array)`` bytes). No loads
    are performed for entries whose mask bit is ``false``---instead, these
    entries are initialized with zero.

    .. warning::
        Performing an aligned load from an unaligned memory address will cause a
        general protection fault that immediately terminates the application.

.. cpp:function:: template <typename Array> Array load_unaligned(const void *mem, mask_t<Array> mask = true)

    Loads an array of type ``Array`` from the memory address ``mem`` (which is
    not required to be aligned). No loads are performed for entries whose mask
    bit is ``false``---instead, these entries are initialized with zero.

.. cpp:function:: template <typename Array> void store(const void *mem, Array array, mask_t<Array> mask = true)

    Stores an array of type ``Array`` at the memory address ``mem`` (which is
    assumed to be aligned on a multiple of ``alignof(Array)`` bytes). No stores
    are performed for entries whose mask bit is ``false``.

    .. warning::
        Performing an aligned storefrom an unaligned memory address will cause a
        general protection fault that immediately terminates the application.

.. cpp:function:: template <typename Array> void store_unaligned(const void *mem, Array array, mask_t<Array> mask = true)

    Stores an array of type ``Array`` at the memory address ``mem`` (which is
    not required to be aligned). No stores are performed for entries whose mask
    bit is ``false``.

.. cpp:function:: template <typename Array, size_t Stride = sizeof(scalar_t<Array>), \
                            typename Index> \
                  Array gather(const void *mem, Index index, mask_t<Array> mask = true)

    Loads an array of type ``Array`` using a masked gather operation. This is
    equivalent to the following scalar loop (which is mapped to efficient
    hardware instructions if supported by the target hardware).

    .. code-block:: cpp

        Array result;
        for (size_t i = 0; i < Array::Size; ++i)
            if (mask[i])
                result[i] = ((Type *) mem)[index[i]];
            else
                result[i] = Type(0);

    The ``index`` parameter must be a 32 or 64 bit integer array having the
    same number of entries. It will be interpreted as a signed array regardless
    of whether the provided array is signed or unsigned.

    The default value of the ``Stride`` parameter indicates that the data at
    ``mem`` uses a packed memory layout (i.e. a stride value of
    ``sizeof(Type)``); other values override this behavior.

.. cpp:function:: template <size_t Stride = 0, typename Array, typename Index> \
                  void scatter(const void *mem, Array array, Index index, mask_t<Array> mask = true)

    Stores an array of type ``Array`` using a scatter operation. This is
    equivalent to the following scalar loop (which is mapped to efficient
    hardware instructions if supported by the target hardware).

    .. code-block:: cpp

        for (size_t i = 0; i < Array::Size; ++i)
            if (mask[i])
                ((Type *) mem)[index[i]] = array[i];

    The ``index`` parameter must be a 32 or 64 bit integer array having the
    same number of entries. It will be interpreted as a signed array regardless
    of whether the provided array is signed or unsigned.

    The default value of the ``Stride`` parameter indicates that the data at
    ``mem`` uses a packed memory layout (i.e. a stride value of
    ``sizeof(Type)``); other values override this behavior.

.. cpp:function:: template <typename Array, size_t Stride = sizeof(scalar_t<Array>), \
                            bool Write = false, size_t Level = 2, typename Index> \
                  void prefetch(const void *mem, Index index, mask_t<Array> mask = true)

    Pre-fetches an array of type ``Array`` into the L1 or L2 cache (as
    indicated via the ``Level`` template parameter) to reduce the latency of a
    future gather or scatter operation. If ``Write = true``, the
    the associated cache line should be acquired for write access (i.e. a
    *scatter* rather than a *gather* operation).

    The ``index`` parameter must be a 32 or 64 bit integer array having the
    same number of entries. It will be interpreted as a signed array regardless
    of whether the provided array is signed or unsigned.

    If provided, the mask parameter specifies which of the pre-fetches should
    actually be performed.

    The default value of the ``Stride`` parameter indicates that the data at
    ``mem`` uses a packed memory layout (i.e. a stride value of
    ``sizeof(Type)``); other values override this behavior.

Miscellaneous initialization
----------------------------

.. cpp:function:: template <typename Array> Array zero()

    Returns an array filled with zeros. This is analogous to writing
    ``Array(0)`` but makes it more explicit to the compiler that a specific
    efficient instruction sequence should be used for zero-initialization.

.. cpp:function:: template <typename Array> Array index_sequence()

    Return an array initialized with an index sequence, i.e. ``0, 1, .., Array::Size-1``.

.. cpp:function:: template <typename Array> Array linspace(scalar_t<Array> min, scalar_t<Array> max)

    Return an array initialized with linear linearly spaced entries including
    the endpoints ``min`` and ``max``.

Elementary Arithmetic Operators
-------------------------------

.. cpp:function:: template <typename Array> Array operator+(Array x, Array y)

    Binary addition operator.

.. cpp:function:: template <typename Array> Array operator-(Array x, Array y)

    Binary subtraction operator.

.. cpp:function:: template <typename Array> Array operator-(Array x)

    Unary minus operator.

.. cpp:function:: template <typename Array> Array operator*(Array x, Array y)

    Binary multiplication operator.

.. cpp:function:: template <typename Array> Array operator/(Array x, Array y)

    Binary division operator. A special overload to multiply by the reciprocal
    when the second argument is a scalar.

    Integer division is handled specially, see :ref:`integer-division` for
    details.


Elementary Arithmetic Functions
-------------------------------

.. cpp:function:: template <typename Array> Array rcp(Array x)

    Computes the reciprocal :math:`\frac{1}{x}`. A slightly less accurate (but
    more efficient) implementation is used when approximate mode is enabled for
    ``Array``. Relies on AVX512ER instructions if available.


.. cpp:function:: template <typename Array> Array rsqrt(Array x)

    Computes the reciprocal square root :math:`\frac{1}{\sqrt{x}}`. A slightly
    less accurate (but more efficient) implementation is used when approximate
    mode is enabled for ``Array``. Relies on AVX512ER instructions if available.

.. cpp:function:: template <typename Array> Array abs(Array x)

    Computes the absolute value :math:`|x|` (analogous to ``std::abs``).

.. cpp:function:: template <typename Array> Array max(Array x, Array y)

    Returns the maximum of :math:`x` and :math:`y` (analogous to ``std::max``).

.. cpp:function:: template <typename Array> Array min(Array x, Array y)

    Returns the minimum of :math:`x` and :math:`y` (analogous to ``std::min``).

.. cpp:function:: template <typename Array> Array sign(Array x)

    Computes the signum function :math:`\begin{cases}1,&\mathrm{if}\ x\ge 0\\0,&\mathrm{otherwise}\end{cases}`

    Analogous to ``std::copysign(1.f, x)``.

.. cpp:function:: template <typename Array> Array sqrt(Array x)

    Computes the square root of :math:`x` (analogous to ``std::sqrt``).

.. cpp:function:: template <typename Array> Array ceil(Array x)

    Computes the ceiling of :math:`x` (analogous to ``std::ceil``).

.. cpp:function:: template <typename Array> Array floor(Array x)

    Computes the floor of :math:`x` (analogous to ``std::floor``).

.. cpp:function:: template <typename Array> Array round(Array x)

    Rounds :math:`x` to the nearest integer using Banker's rounding for
    half-way values.

    .. note::

        This is analogous to ``std::rint``, not ``std::round``.

.. cpp:function:: template <typename Array> Array fmadd(Array x, Array y, Array z)

    Performs a fused multiply-add operation if supported by the target
    hardware. Otherwise, the operation is emulated using conventional
    multiplication and addition (i.e. ``x * y + z``).

.. cpp:function:: template <typename Array> Array fnmadd(Array x, Array y, Array z)

    Performs a fused negative multiply-add operation if supported by the target
    hardware. Otherwise, the operation is emulated using conventional
    multiplication and addition (i.e. ``-x * y + z``).

.. cpp:function:: template <typename Array> Array fmsub(Array x, Array y, Array z)

    Performs a fused multiply-subtract operation if supported by the target
    hardware. Otherwise, the operation is emulated using conventional
    multiplication and subtraction (i.e. ``x * y - z``).

.. cpp:function:: template <typename Array> Array fnmsub(Array x, Array y, Array z)

    Performs a fused negative multiply-subtract operation if supported by the
    target hardware. Otherwise, the operation is emulated using conventional
    multiplication and subtraction (i.e. ``-x * y - z``).

.. cpp:function:: template <typename Array> Array ldexp(Array x, Array n)

    Multiplies :math:`x` by :math:`2^n`. Analogous to ``std::ldexp`` except
    that ``n`` is a floating point argument.

.. cpp:function:: template <typename Array> std::pair<Array, Array> frexp(Array x)

    Breaks the floating-point number :math:`x` into a normalized fraction and
    power of 2. Analogous to ``std::frexp`` except that both return values are
    floating point values.

Mask-related operations
-----------------------

.. cpp:function:: template <typename Array> mask_t<Array> isnan(Array x)

    Checks for NaN values and returns a mask, analogous to ``std::isnan``.

.. cpp:function:: template <typename Array> mask_t<Array> isinf(Array x)

    Checks for infinite values and returns a mask, analogous to ``std::isinf``.

.. cpp:function:: template <typename Array> mask_t<Array> isfinite(Array x)

    Checks for finite values and returns a mask, analogous to ``std::isfinite``.

Transcendental functions
------------------------

Accuracy of transcendental function approximations
**************************************************

Most approximations of transcendental functions are based on routines in the
CEPHES math library. The table below provides some statistics on their absolute
and relative error.

The CEPHES approximations are only used when approximate mode is enabled;
otherwise, the functions below will invoke the corresponding non-vectorized
standard C library routines.

.. note::

    The forward trigonometric functions (*sin*, *cos*, *tan*) are optimized for
    low error on the domain :math:`|x| < 8192` and don't perform as well beyond
    this range.

.. list-table::
    :widths: 5 8 8 10 8 10
    :header-rows: 1
    :align: center

    * - Function
      - Tested domain
      - Abs. error (mean)
      - Abs. error (max)
      - Rel. error (mean)
      - Rel. error (max)
    * - :math:`sin(x)`
      - :math:`-8192 < x < 8192`
      - :math:`1.2 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`1.9 \cdot 10^{-8}\,(0.25\,\mathrm{ulp})`
      - :math:`1.8 \cdot 10^{-6}\,(19\,\mathrm{ulp})`
    * - :math:`cos(x)`
      - :math:`-8192 < x < 8192`
      - :math:`1.2 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`1.9 \cdot 10^{-8}\,(0.25\,\mathrm{ulp})`
      - :math:`3.1 \cdot 10^{-6}\,(47\,\mathrm{ulp})`
    * - :math:`tan(x)`
      - :math:`-8192 < x < 8192`
      - :math:`4.6 \cdot 10^{-6}`
      - :math:`8.1 \cdot 10^{-1}`
      - :math:`3.9 \cdot 10^{-8}\,(0.47\,\mathrm{ulp})`
      - :math:`3.1 \cdot 10^{-6}\,(30\,\mathrm{ulp})`
    * - :math:`asin(x)`
      - :math:`-1 < x < 1`
      - :math:`2.3 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`2.9 \cdot 10^{-8}\,(0.33\,\mathrm{ulp})`
      - :math:`2.3 \cdot 10^{-7}\,(2\,\mathrm{ulp})`
    * - :math:`acos(x)`
      - :math:`-1 < x < 1`
      - :math:`4.7 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`2.9 \cdot 10^{-8}\,(0.33\,\mathrm{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\mathrm{ulp})`
    * - :math:`atan(x)`
      - :math:`-1 < x < 1`
      - :math:`1.8 \cdot 10^{-7}`
      - :math:`6.6 \cdot 10^{-7}`
      - :math:`4.2 \cdot 10^{-7}\,(4.9\,\mathrm{ulp})`
      - :math:`8.5 \cdot 10^{-7}\,(12\,\mathrm{ulp})`
    * - :math:`sinh(x)`
      - :math:`-10 < x < 10`
      - :math:`2.7 \cdot 10^{-5}`
      - :math:`9.8 \cdot 10^{-4}`
      - :math:`2.6 \cdot 10^{-8}\,(0.31\,\mathrm{ulp})`
      - :math:`2 \cdot 10^{-7}\,(2\,\mathrm{ulp})`
    * - :math:`cosh(x)`
      - :math:`-10 < x < 10`
      - :math:`4.1 \cdot 10^{-5}`
      - :math:`9.8 \cdot 10^{-4}`
      - :math:`2.9 \cdot 10^{-8}\,(0.35\,\mathrm{ulp})`
      - :math:`2 \cdot 10^{-7}\,(2\,\mathrm{ulp})`
    * - :math:`tanh(x)`
      - :math:`-10 < x < 10`
      - :math:`2.6 \cdot 10^{-8}`
      - :math:`1.8 \cdot 10^{-7}`
      - :math:`2.8 \cdot 10^{-8}\,(0.44\,\mathrm{ulp})`
      - :math:`3.1 \cdot 10^{-7}\,(3\,\mathrm{ulp})`
    * - :math:`asinh(x)`
      - :math:`-10 < x < 10`
      - :math:`2.8 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`1.5 \cdot 10^{-8}\,(0.18\,\mathrm{ulp})`
      - :math:`1.7 \cdot 10^{-7}\,(2\,\mathrm{ulp})`
    * - :math:`acosh(x)`
      - :math:`1 < x < 10`
      - :math:`2.9 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`1.5 \cdot 10^{-8}\,(0.18\,\mathrm{ulp})`
      - :math:`2.4 \cdot 10^{-7}\,(3\,\mathrm{ulp})`
    * - :math:`atanh(x)`
      - :math:`-1 < x < 1`
      - :math:`9.9 \cdot 10^{-9}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`1.5 \cdot 10^{-8}\,(0.18\,\mathrm{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\mathrm{ulp})`
    * - :math:`exp(x)`
      - :math:`-20 < x < 30`
      - :math:`7.2 \cdot 10^{3}`
      - :math:`0.1 \cdot 10^{7}`
      - :math:`2.4 \cdot 10^{-8}\,(0.27\,\mathrm{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\mathrm{ulp})`
    * - :math:`log(x)`
      - :math:`10^{-20} < x < 10^{30}`
      - :math:`9.8 \cdot 10^{-9}`
      - :math:`7.6 \cdot 10^{-6}`
      - :math:`1.4 \cdot 10^{-10}\,(0.0013\,\mathrm{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\mathrm{ulp})`
    * - :math:`erf(x)`
      - :math:`-1 < x < 1`
      - :math:`1.0 \cdot 10^{-7}`
      - :math:`5.6 \cdot 10^{-7}`
      - :math:`3.3 \cdot 10^{-7}\,(4\,\mathrm{ulp})`
      - :math:`6.2 \cdot 10^{-6}\,(75\,\mathrm{ulp})`

Trigonometric functions
***********************

.. cpp:function:: template <typename Array> Array sin(Array x)

    Sine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array cos(Array x)

    Cosine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> std::pair<Array, Array> sincos(Array x)

    Simultaneous sine and cosine function approximation based on the CEPHES
    library.

.. cpp:function:: template <typename Array> Array tan(Array x)

    Tangent function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array csc(Array x)

    Cosecant convenience function implemented as ``rcp(sin(x))``.

.. cpp:function:: template <typename Array> Array sec(Array x)

    Cosecant convenience function implemented as ``rcp(cos(x))``.

.. cpp:function:: template <typename Array> Array cot(Array x)

    Cotangent convenience function implemented as ``rcp(tan(x))``.

.. cpp:function:: template <typename Array> Array asin(Array x)

    Arcsine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array acos(Array x)

    Arccosine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array atan(Array x)

    Arctangent function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array atan2(Array y, Array x)

    Arctangent function of two variables.

Hyperbolic functions
********************

.. cpp:function:: template <typename Array> Array sinh(Array x)

    Hyperbolic sine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array cosh(Array x)

    Hyperbolic cosine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> std::pair<Array, Array> sincosh(Array x)

    Simultaneous hyperbolic sine and cosine function approximation based on the
    CEPHES library.

.. cpp:function:: template <typename Array> Array tanh(Array x)

    Hyperbolic tangent function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array csch(Array x)

    Hyperbolic cosecant convenience function implemented as ``rcp(sinh(x))``.

.. cpp:function:: template <typename Array> Array sech(Array x)

    Hyperbolic secant convenience function.

.. cpp:function:: template <typename Array> Array coth(Array x)

    Hyperbolic cotangent convenience function implemented as ``rcp(tanh(x))``.

.. cpp:function:: template <typename Array> Array asinh(Array x)

    Hyperbolic arcsine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array acosh(Array x)

    Hyperbolic arccosine function approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array atanh(Array x)

    Hyperbolic arctangent function approximation based on the CEPHES library.

Exponential, logarithm, and others
**********************************

.. cpp:function:: template <typename Array> Array exp(Array x)

   Base-:math:`e` exponential function approximation based on the CEPHES
   library. Relies on AVX512ER instructions if available.

.. cpp:function:: template <typename Array> Array log(Array x)

    Natural logarithm approximation based on the CEPHES library.

.. cpp:function:: template <typename Array> Array pow(Array x, Array y)

    Computes the power function :math:`x^y`.

.. cpp:function:: template <typename Array> Array erf(Array x)

    Error function approximation.

.. cpp:function:: template <typename Array> Array erfinv(Array x)

    Inverse error function approximation.

.. cpp:function:: template <typename Array> Array i0e(Array x)

    Approximation of the exponentially scaled modified Bessel function of order
    zero.
