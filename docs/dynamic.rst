.. _dynamic:

Dynamic arrays
==============

Arrays and nested arrays facilitate the development of vectorized code that
processes multiple values at once. However, it can be awkward to work with
small fixed packet sizes when the underlying application must process millions
or billions of data points. The remainder of this document discusses
infrastructure that can be used to realize computations involving dynamically
allocated arrays of arbitrary length.

One of the core ingredients is :cpp:class:`enoki::DynamicArray`, which is a
smart pointer that manages the lifetime of a dynamically allocated memory
region. It is the exclusive owner of this data and is also responsible for its
destruction when the dynamic array goes out of scope (similar to
``std::unique_ptr``). Dynamic arrays can be used to realize arithmetic
involving data that is much larger than the maximum SIMD width supported by the
underlying hardware.

The class requires a single template argument, which can be any kind of
:cpp:class:`enoki::Array`. This is the *packet type* that will be used to used
to realize vectorized computations involving the array contents. The following
code snippet illustrates the creation of a dynamic floating point array that
vectorizes using 4-wide SSE arithmetic.

.. code-block:: cpp

    /* Static float array (the suffix "P" indicates that this is a fixed-size packet) */
    using FloatP = Packet<float, 4>;

    /* Dynamic float array (vectorized via FloatP, the suffix "X" indicates arbitrary length) */
    using FloatX = DynamicArray<FloatP>;

.. note::

    In contrast to the array types discussed so far, a
    :cpp:class:`enoki::DynamicArray` instance *should never* be part of an
    arithmetic expression. For instance, the following will compile and yield
    the expected result, but this style of using dynamic arrays is *strongly*
    disouraged.

    .. code-block:: cpp
        :emphasize-lines: 4

        FloatX in1 = ... , in2 = ... , in3 = ... , in4 = ...;

        /* Add the dynamic arrays using operator+() */
        FloatX out = in1 + in2 + in3 + in4;

    Why that is the case requires a longer explanation on the design of this
    library.

    At a high level, there are two "standard" ways of implementing arithmetic
    for dynamically allocated memory regions.

    1. A common approach (used e.g. by most ``std::valarray`` implementations)
       is to evaluate partial expressions in place, creating a large number
       of temporaries in the process.

       This is unsatisfactory since the amount of computation is very small
       compared to the resulting memory traffic.

    2. The second is a technique named `expression templates
       <https://en.wikipedia.org/wiki/Expression_templates>`_ that is used
       in libraries such as `Eigen <https://eigen.tuxfamily.org>`_.
       Expression templates construct complex graphs describing the inputs
       and operations of a mathematical expression using C++ templates.
       The underlying motivation is to avoid numerous memory allocations
       for temporaries by postponing evaluation until the point where the
       expression template is assigned to a storage container.

       Unfortunately, pushed to the scale of entire programs, this approach
       tends to produce intermediate code with an extremely large number of
       common subexpressions that exceeds the capabilities of the *common
       subexpression elimination* (CSE) stage of current compilers. The
       first version of Enoki in fact used expression templates, and it was
       due to the difficulties with them that an alternative was developed.

    The key idea of vectorizing over dynamic Enoki arrays is to iterate over
    packets (i.e. static arrays) that represent a sliding window into the
    dynamic array's contents. Packets, in turn, are easily supported using the
    tools discussed in the previous sections. Enoki provides a powerful
    operation named :cpp:func:`enoki::vectorize`, discussed later, that
    implements this sliding window technique automatically.

    That said, for convenience, arithmetic operations like ``operator+`` *are*
    implemented for dynamic arrays, and they are realized using approach 1 of
    the above list (i.e. with copious amounts of memory allocation for
    temporaries). Using them in performance-critical code is unadvisable.


Allocating dynamic arrays
-------------------------

The dynamic array API is minimalistic. Arrays can be created, resized, and
queried for their size---that's mostly it (after all, they are only meant to be
used as holder types).

The allocated memory region is always fully aligned according to the
requirements of the packet type. Enoki may sometimes allocate a partially used
packet at the end, which eliminates the need for special end-of-array handling.
The following code snippet allocates an array of size 5 using 4-wide packets,
which means that 3 entries at the end are unused.

.. image:: dynamic-01.svg
    :width: 400px
    :align: center

.. code-block:: cpp

    /* Creates a dynamic array that is initially empty */
    FloatX x;

    /* Allocate memory for at least 5 entries */
    set_slices(x, 5);

    /* Query the size (a.k.a number of "slices") of the dynamic array */
    size_t slice_count = slices(x);
    assert(slice_count == 5);

    /* Query the number of packets */
    size_t packet_count = packets(x);
    assert(packet_count == 2);

A few convenience initialization methods also exist:

.. code-block:: cpp

    /* Efficient way to create an array filled with zero entries */
    x = zero<FloatX>(size);

    /* Initialize entries with index sequence 0, 1, 2, ... */
    x = index_sequence<FloatX>(size);

    /* Initialize entries with a linearly increasing sequence with endpoints 0 and 1 */
    x = linspace<FloatX>(size, 0.f, 1.f);

Custom dynamic data structures
------------------------------

The :ref:`previous section <custom-structures>` used the example of a GPS
record to show how Enoki can create packet versions of a type. The same
approach also generalizes to dynamic arrays, allowing an arbitrarily long
sequence of records to be represented. This requires two small additions to the
original type declaration:

.. code-block:: cpp
    :emphasize-lines: 10, 11, 14
    :linenos:

    template <typename Value> struct GPSCoord2 {
        using Vector2 = Array<Value, 2>;
        using UInt64  = uint64_array_t<Value>;
        using Bool    = bool_array_t<Value>;

        UInt64 time;
        Vector2 pos;
        Bool reliable;

        ENOKI_STRUCT(GPSCoord2,           /* <- name of this class */
                     time, pos, reliable  /* <- list of all attributes in layout order */)
    };

    ENOKI_STRUCT_SUPPORT(GPSCoord2, time, pos, reliable)

The two highlighted changes play the following roles:

1. The macro on lines 10 and 11 declares copy and assignment constructors that
   are able to convert between different types of records.

2. The macro on line 14 declares a partial template overload that makes Enoki
   aware of ``GPSCoord2`` for the purposes of dynamic vectorization.

It is possible but fairly tedious to write these declarations by hand, hence
the code generation macros.

With these declarations, we can now allocate a dynamic array of 1000
coordinates that will be processed in packets of 4 (or more, depending on the
definition of ``FloatP``):

.. code-block:: cpp

   using GPSCoord2fX = GPSCoord2<FloatX>;

   GPSCoord2fX coord;
   set_slices(coord, 1000);

In memory, this data will be arranged as follows:

.. image:: dynamic-02.svg
    :width: 600px
    :align: center

In other words: each field references a dynamic array that contiguously stores
the contents in a SoA organization.

Accessing array packets
-----------------------

The :cpp:func:`enoki::packet` function can be used to create a reference to the
:math:`i`-th packet of a dynamic array or a custom dynamic data structure.
For instance, the following code iterates over all packets and resets their
time values:

.. code-block:: cpp

    /* Reset the time value of all records */
    for (size_t i = 0; i < packets(coord); ++i) {
        auto &&ref = packet(coord, n);
        ref.time = 0;
    }

When used with a dynamic data structure, ``packet()`` function is interesting
because it returns an instance of a new type ``GPSRecord2<FloatP&>`` that was
not discussed yet (note the ampersand in the template argument). Instead of
directly storing data, all fields of a ``GPSRecord2<FloatP&>`` are references
pointing to packets of data elsewhere in memory. In this case, assigning
(writing) to a field of this structure of references will change the
corresponding entry *of the dynamic array*. Conceptually, this looks as follows:

.. image:: dynamic-03.svg
    :width: 600px
    :align: center

References can also be cast into their associated packet types and vice versa:

.. code-block:: cpp

    /* Read a GPSRecord2<FloatP&> and convert to GPSRecord2<FloatP> */
    GPSCoord2fP cp = packet(coord, n);

    /* Assign a GPSRecord2<FloatP> to a GPSRecord2<FloatP&> */
    packet(coord, n + 1) = cp;

.. note::

    For non-nested dynamic arrays such as ``FloatX = DynamicArray<FloatP>``,
    calling ``packet()`` simply returns a reference to the right ``FloatP``
    entry in that array of packets. Note this is a reference type
    (``FloatP&``), not a structure (``GPSRecord2<FloatP&>``).
    This is why we strongly encourage using universal references (``auto &&``)
    to hold the result of ``packet()``:

    .. code-block:: cpp

        auto   ref = packet(coord, i);   // Only works for dynamic structures
        auto  &ref = packet(numbers, i); // Only works for non-nested arrays
        auto &&ref = packet(coord, i);   // Works for both

Accessing array slices
----------------------

Enoki provides a second way of indexing into dynamic arrays: the
:cpp:func:`enoki::slice` function creates a reference to the
:math:`i`-th *slice* of a dynamic array or a custom dynamic data
structure. Elements of a slice store references to *scalar*
elements representing a vertical slice through the data structure.

The following code iterates over all slices and initializes the time values to
an increasing sequence:

.. code-block:: cpp

    /* Set the i-th time value to 'i' */
    for (size_t i = 0; i < slices(coord); ++i) {
        auto ref = slice(coord, n);
        ref.time = i;
    }

Here, the :cpp:func:`enoki::slice()` function returns an instance
of a new type ``GPSRecord2<float&>`` (again, note the ampersand),
Conceptually, this looks as follows:

.. image:: dynamic-06.svg
    :width: 600px
    :align: center

Slice reference types can also be cast into their associated scalar data types
and vice versa:

.. code-block:: cpp

    /* Read a GPSRecord2<float&> and convert to GPSRecord2<float> */
    GPSCoord2f c = slice(coord, n);

    /* Assign a GPSRecord2<float> to a GPSRecord2<float&> */
    slice(coord, n + 1) = c;


Dynamic vectorization
---------------------

Now suppose that we'd like to compute the pairwise distance between records
organized in two dynamically allocated lists. Direct application of the
discussed ingredients leads to the following overall structure:

.. code-block:: cpp

    GPSCoord2fX coord1;
    GPSCoord2fX coord2;
    FloatX result;

    // Allocate memory and fill input arrays with contents (e.g. using slice(...))
    ...

    // Call SIMD-vectorized function for each packet
    for (size_t i = 0; i < packets(coord1); ++i)
        packet(result, i) = distance(packet(coord1, i),
                                     packet(coord2, i));

This does not quite compile (yet)---a minor modification of the ``distance()``
function is required:

.. code-block:: cpp
    :emphasize-lines: 2, 3
    :linenos:

    /// Calculate the distance in kilometers between 'r1' and 'r2' using the haversine formula
    template <typename Value_, typename Value = expr_t<Value_>>
    Value distance(const GPSCoord2<Value_> &r1, const GPSCoord2<Value_> &r2) {
        using Scalar = scalar_t<Value>;
        const Value deg_to_rad = Scalar(M_PI / 180.0);

        auto sin_diff_h = sin(deg_to_rad * .5f * (r2.pos - r1.pos));
        sin_diff_h *= sin_diff_h;

        Value a = sin_diff_h.x() + sin_diff_h.y() *
                  cos(r1.pos.x() * deg_to_rad) *
                  cos(r2.pos.x() * deg_to_rad);

        return select(
            r1.reliable & r2.reliable,
            Scalar(6371.f * 2.f) * atan2(sqrt(a), sqrt(1.f - a)),
            std::numeric_limits<Scalar>::quiet_NaN()
        );
    }

The modified version above uses the :cpp:type:`enoki::expr_t` type trait to
determine a suitable type that is able to hold the result of an expression
involving its argument (which turns ``FloatP&`` into ``FloatP`` in this case).

.. note::

    The issue with the original code was that it was called with a
    ``GPSRecord2<FloatP&>`` instance, i.e. with a template parameter ``Value =
    FloatP&``. However, the ``Value`` type is also used for the return value as
    well as various intermediate computations, which is illegal since these
    temporaries are not associated with an address in memory.

With these modifications, we are now finally able to vectorize over the dynamic
array:

.. code-block:: cpp

    // Call SIMD-vectorized function for each packet -- yay!
    for (size_t i = 0; i < packets(coord1); ++i)
        packet(result, i) = distance(packet(coord1, i),
                                     packet(coord2, i));

Shorthand notation
------------------

Extracting individual packets as shown in the snippet above can become fairly
tedious when a function takes many arguments. Enoki offers a convenient helper
function named :cpp:func:`enoki::vectorize` that automates this process. It
takes a function and a number of dynamic arrays as input and calls the function
once for each set of input packets.

.. code-block:: cpp

    FloatX result = vectorize(
        distance<FloatP>, // Function to call
        coord1,           // Input argument 1
        coord2            // Input argument 2
                          // ...
    );

Here, the returned float packets are stored in a dynamic array of type
``FloatX``.

When the output array is already allocated, it is also possible to write the
results directly into the array. The snippet below shows how to do this by
calling call :cpp:func:`enoki::vectorize` with a lambda function.

.. code-block:: cpp

    vectorize(
        [](auto&& result, auto&& coord1, auto &&coord2) {
            result = distance<FloatP>(coord1, coord2);
        },
        result,
        coord1,
        coord2
    );

Note the use of a variadic lambda with ``auto&&`` arguments: it would be
redundant to specify the argument types since they are automatically inferred
from the function inputs.

Naturally, we could also perform the complete calculation within the lambda function:

.. code-block:: cpp

    vectorize(
        [](auto&& result, auto&& coord1, auto&& coord2) {
            using Value = FloatP;
            using Scalar = float;

            const Value deg_to_rad = Scalar(M_PI / 180.0);

            auto sin_diff_h = sin(deg_to_rad * .5f * (coord2.pos - coord1.pos));
            sin_diff_h *= sin_diff_h;

            Value a = sin_diff_h.x() + sin_diff_h.y() *
                      cos(coord1.pos.x() * deg_to_rad) *
                      cos(coord2.pos.x() * deg_to_rad);

            result = select(
                coord1.reliable & coord2.reliable,
                (6371.f * 2.f) * atan2(sqrt(a), sqrt(1.f - a)),
                std::numeric_limits<Scalar>::quiet_NaN()
            );
        },

        result,
        coord1,
        coord2
    );

It is not necessary to "route" all parameters through
:cpp:func:`enoki::vectorize`. Auxiliary data structures or constants are easily
accessible via the lambda capture object using the standard ``[&]`` notation.

