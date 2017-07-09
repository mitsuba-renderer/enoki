.. _custom-structures:

Custom data structures
======================

The previous sections introduced Enoki arrays as a powerful ingredient for
designing vectorized algorithms that are capable of processing multiple inputs
at the same time. However, in many cases, vectorizing an algorithm will also
require a corresponding change to the data structures that underlie it. This
section demonstrates how C++ templates provide a natural framework for
achieving this goal while satisfying the desiderata of Enoki (readability,
portability, no code duplication, etc.).

We will focus on an simple example data structure that represents a position
record acquired by a GPS tracker along with auxiliary information (time, and a
flag stating whether the data is considered reliable).

.. code-block:: cpp

    using Vector2f = Array<float, 2>;

    /// Simple 2D GPS record tagged with auxiliary information
    struct GPSCoord2f {
        /// UNIX time when the data point was acquired (seconds since Jan 1970)
        uint64_t time;

        /// Latitude, longitude as a 2D vector
        Vector2f pos;

        /// Is the data point reliable (e.g. enough GPS satellites in sensor's field of view)
        bool reliable;
    };

Next, consider the following function, which computes the distance between two
GPS coordinates using the haversine formula. When either of the two positions
is deemed unreliable, it returns a *NaN* value to inform the caller about this.

.. code-block:: cpp

    /// Calculate the distance in kilometers between 'r1' and 'r2' using the haversine formula
    float distance(const GPSCoord2f &r1, const GPSCoord2f &r2) {
        const float deg_to_rad = (float) (M_PI / 180.0);

        if (!r1.reliable || !r2.reliable)
            return std::numeric_limits<float>::quiet_NaN();

        Vector2f sin_diff_h = sin(deg_to_rad * .5f * (r2.pos - r1.pos));
        sin_diff_h *= sin_diff_h;

        float a = sin_diff_h.x() + sin_diff_h.y() *
                  cos(r1.pos.x() * deg_to_rad) *
                  cos(r2.pos.x() * deg_to_rad);

        return 6371.f * 2.f * atan2(sqrt(a), sqrt(1.f - a));
    }

Suppose that we would like to add a second vectorized version of this function,
which works with packets of GPS coordinates. Instead of defining yet another
data structure for coordinates packets in addition to the existing
``GPSCoord2f``, our approach will be to rely on a single template data
structure that subsumes both cases. It is parameterized by the type of a GPS
position component (e.g. latitude) named ``Value``.

.. code-block:: cpp

    template <typename Value> struct GPSCoord2 {
        using Vector2 = Array<Value, 2>;
        using UInt64  = uint64_array_t<Value>;
        using Bool    = bool_array_t<Value>;

        UInt64 time;
        Vector2 pos;
        Bool reliable;
    };

The ``using`` declarations at the beginning require an explanation: they
involve the type traits :cpp:type:`enoki::uint64_array_t` and
:cpp:type:`enoki::bool_array_t`, which "compute" the type of an Enoki array
that has the same configuration as their ``Value`` parameter, but with
``uint64_t``- and boolean-valued entries, respectively. Both are
specializations of the more general :cpp:type:`enoki::like_t` trait that works
for any type.

With these declarations, we can now create a packet type ``GPSCoord2fP`` that
stores 16 GPS positions in a convenient SoA representation.

.. code-block:: cpp

    using FloatP      = Array<float, 16>;
    using GPSCoord2fP = GPSCoord2<FloatP>;

An important aspect of the type calculations mentioned above is that they
also generalize to non-array arguments. In particular, ``uint64_array_t<float>`` and
``bool_array_t<float>`` simply turn into ``uint64_t`` and ``bool``, respectively,
hence the type alias

.. code-block:: cpp

    using GPSCoord2f  = GPSCoord2<float>;

perfectly reproduces the original (scalar) GPS record definition. Having
defined the GPS record type, it is time to update the function definition as
well. Once more, we will rely on C++ templates to do so.

The new ``distance`` function shown below is similarly templated with respect
to the ``Value`` type, and it works for both scalar and vector arguments.

.. code-block:: cpp
    :linenos:

    /// Calculate the distance in kilometers between 'r1' and 'r2' using the haversine formula
    template <typename Value>
    Value distance(const GPSCoord2<Value> &r1, const GPSCoord2<Value> &r2) {
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

Note how the overall structure is preserved. There are two noteworthy changes:

1. Control flow such as ``if`` statements must be replaced by branchless code
   involving masks (see the :cpp:func:`enoki::select` statement on line 15).
   Separate entries may have a different control flow, which is not possible
   with standard C++ language constructs, hence the need for masks.

   If desired, the early-out optimization from the previous snippet can be
   preserved for the special case that *all* records are unreliable:

   .. code-block:: cpp

       if (ENOKI_UNLIKELY(none(r1.reliable & r2.reliable)))
           return Value(std::numeric_limits<Scalar>::quiet_NaN())

   The :cpp:func:`ENOKI_UNLIKELY` macro signals that the branch is rarely
   taken, which can be used for improved code layout if supported by the
   compiler.

2. The :cpp:type:`enoki::scalar_t` type alias on line 4 is used to extract the
   elementary arithmetic type underlying an Enoki array---this results in the
   type ``float`` in our example, which is used to cast various constants to
   the right precision.

   It is sometimes useful to be able to work with a higher precision. Our
   templated ``distance`` function can nicely accommodate this need by simply
   switching to the following types:

   .. code-block:: cpp

       using GPSCoord2d   = GPSCoord2<double>;
       using DoubleP      = Array<double, 16>;
       using GPSCoord2dP  = GPSCoord2<DoubleP>;

   The ``distance`` function requires no changes. When working with double
   precision GPS records, all constants used in the algorithm automatically
   adapt to the higher precision due to the casts to the ``Scalar`` type.
