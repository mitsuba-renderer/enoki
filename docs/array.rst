.. cpp:namespace:: enoki

Static arrays
=============

Global definitions
------------------

.. cpp:var:: static constexpr size_t max_packet_size

   Denotes the maximal packet size (in bytes) that can be mapped to native
   vector registers. It is equal to 64 if AVX512 is present, 32 if AVX is
   present, and 16 for machines with only SSE 4.2.

.. cpp:var:: static constexpr bool has_avx512dq

    Specifies whether AVX512DQ instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx512vl

    Specifies whether AVX512VL instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx512bw

    Specifies whether AVX512BW instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx512cd

    Specifies whether AVX512CD instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx512pf

    Specifies whether AVX512PF instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx512er

    Specifies whether AVX512ER instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx512f

    Specifies whether AVX512F instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx2

    Specifies whether AVX2 instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_avx

    Specifies whether AVX instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_fma

    Specifies whether FMA instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_f16c

    Specifies whether F16C instructions are available on the target architecture.

.. cpp:var:: static constexpr bool has_sse42

    Specifies whether SSE 4.2 instructions are available on the target architecture.

Rounding modes
--------------

.. cpp:enum:: RoundingMode

    Enumeration defining the choice of rounding modes for floating point
    operations. :cpp:enumerator:`RoundingMode::Default` must be used for integer
    arrays.

    .. cpp:enumerator:: Default

        Don't interfere with the rounding mode that is currently configured in
        the hardware's status register.

    .. cpp:enumerator:: Nearest

        Round to the nearest representable value (the tie-breaking method is
        hardware dependent)

    .. cpp:enumerator:: Down

        Always round to negative infinity

    .. cpp:enumerator:: Up

        Always round to positive infinity

    .. cpp:enumerator:: Zero

        Always round to zero

Static arrays
-------------

.. cpp:class:: template <typename Type, size_t Size = max_packet_size / sizeof(Type), \
                         bool Approx = detail::approx_default<Type>::value, \
                         RoundingMode Mode = RoundingMode::Default> \
               Array : StaticArrayImpl<Type, Size, Approx, Mode, Array<Type, Size, Approx, Mode>>

    The default Enoki array class -- a generic container that stores a
    fixed-size array of an arbitrary data type similar to the standard template
    library class ``std::array``. The main distinction between the two is that
    :cpp:class:`enoki::Array` forwards all arithmetic operations (and other
    standard mathematical functions) to the contained elements.

    It has several template parameters:

    :tparam typename Type: The underlying scalar data type
    :tparam size_t Size: Number of packed array entries
    :tparam bool Approx:
        Use the vectorized approximate math library? In this case,
        transcendental operations like ``sin``, ``atanh``, etc. will run using
        a fast vectorized implementation that is slightly more approximate than
        the (scalar) implementation provided by the C math library.

        The default is to enable the approximate math library for single
        precision floats. It is not supported for other types, and a
        compile-time assertion will be raised in this case.
    :tparam RoundingMode Mode:
        Specifies the rounding mode used for elementary arithmetic operations.
        Must be set to :any:`RoundingMode::Default` for integer types or a
        compile-time assertion will be raised.

    This class is just a small wrapper that instantiates
    :cpp:class:`enoki::StaticArrayImpl` using the Curiously Recurring Template
    Pattern (CRTP). The latter provides the actual machinery that is needed to
    evaluate array expressions. See :ref:`custom-arrays` for details.

.. cpp:class:: template <typename Type, size_t Size, bool Approx, \
                         RoundingMode Mode, typename Derived> StaticArrayImpl

    This base class provides the core implementation of an Enoki array. It
    cannot be instantiated directly and is used via the Curiously Recurring
    Template Pattern (CRTP). See :cpp:class:`Array` and :ref:`custom-arrays`
    for details on how to create custom array types.

    .. cpp:function:: StaticArrayImpl()

        Create an unitialized array. Floating point arrays are initialized
        using ``std::numeric_limits<Type>::quiet_NaN()`` when the application
        is compiled in debug mode.

    .. cpp:function:: StaticArrayImpl(Type type)

        Broadcast a constant value to all entries of the array.

    .. cpp:function:: template<typename... Args> StaticArrayImpl(Args... args)

        Initialize the individual array entries with ``args`` (where
        ``sizeof...(args) == Size``).

    .. cpp:function:: template<typename Type2, bool Approx2, RoundingMode Mode2, typename Derived2> \
                      StaticArrayImpl(const StaticArrayImpl<Type2, Size, Approx2, Mode2, Derived2> &other)

        Initialize the array with the contents of another given array that
        potentially has a different underlying type. Enoki will perform a
        vectorized type conversion if this is supported by the target
        processor.

    .. cpp:function:: const Type& operator[](size_t index) const

        Return a reference to an array element (const version). When the
        application is compiled in debug mode, the function performs a range
        check and throws ``std::out_of_range`` in case of an out-of-range
        access. This behavior can be disabled by defining
        ``ENOKI_DISABLE_RANGE_CHECK``.

    .. cpp:function:: Type& operator[](size_t index)

        Return a reference to an array element. When the application is
        compiled in debug mode, the function performs a range check and throws
        ``std::out_of_range`` in case of an out-of-range access. This behavior
        can be disabled by defining ``ENOKI_DISABLE_RANGE_CHECK``.

    .. cpp:function:: const Type& coeff(size_t index) const

        Just like :cpp:func:`operator[]`, but without the range check (const
        version).

    .. cpp:function:: Type& coeff(size_t index)

        Just like :cpp:func:`operator[]`, but without the range check.

    .. cpp:function:: Type& x()

        Access the first component.

    .. cpp:function:: const Type& x() const

        Access the first component (const version).

    .. cpp:function:: Type& y()

        Access the second component.

    .. cpp:function:: const Type& y() const

        Access the second component (const version).

    .. cpp:function:: Type& z()

        Access the third component.

    .. cpp:function:: const Type& z() const

        Access the third component (const version).

    .. cpp:function:: Type& w()

        Access the fourth component.

    .. cpp:function:: const Type& w() const

        Access the fourth component (const version).
