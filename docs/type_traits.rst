cpp:namespace:: enoki

Type traits
===========

The following type traits are available to query the properties of arrays at
compile time.

Replacing the scalar type of an array
-------------------------------------

The :cpp:class:`enoki::like_t` type trait and various aliases construct arrays
matching a certain layout, but with different-flavored data. This is often
helpful when defining custom data structures or function inputs. See the
section on :ref:`custom data structures <custom-structures>` for an example
usage.

.. cpp:type:: template <typename Array, typename Scalar> like_t

    Replaces the scalar type underlying an array. For instance,
    ``like_t<Array<Array<float, 16>, 32>, int>`` is equal to ``Array<Array<int,
    16>, 32>``.

    Also works for scalar arguments; pointers and reference arguments are
    copied---for instance, ``like_t<const float *, int>`` is equal to ``const
    int *``.

.. cpp:type:: template <typename Array> uint32_array_t = like_t<Array, uint32_t>

    Create a 32-bit unsigned integer array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> int32_array_t = like_t<Array, int32_t>

    Create a 32-bit signed integer array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> uint64_array_t = like_t<Array, uint64_t>

    Create a 64-bit unsigned integer array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> int64_array_t = like_t<Array, int64_t>

    Create a 64-bit signed integer array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> int_array_t

    Create a signed integer array (with the same number of bits per entry as
    the input) matching the layout of ``Array``.

.. cpp:type:: template <typename Array> uint_array_t

    Create an unsigned integer array (with the same number of bits per entry as
    the input) matching the layout of ``Array``.

.. cpp:type:: template <typename Array> float16_array_t = like_t<Array, half>

    Create a half precision array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> float32_array_t = like_t<Array, float>

    Create a single precision array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> float64_array_t = like_t<Array, double>

    Create a double precision array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> float_array_t

    Create a floating point array (with the same number of bits per entry as
    the input) matching the layout of ``Array``.

.. cpp:type:: template <typename Array> bool_array_t = like_t<Array, bool>

    Create a boolean array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> size_array_t = like_t<Array, size_t>

    Create a ``size_t``-valued array matching the layout of ``Array``.

.. cpp:type:: template <typename Array> ssize_array_t = like_t<Array, ssize_t>

    Create a ``ssize_t``-valued array matching the layout of ``Array``.

Accessing types related to Enoki arrays
---------------------------------------

.. cpp:class:: template <typename T> mask

    .. cpp:type:: type

        Given an Enoki array *T*, :cpp:type:`mask\<T>::type` provides access to the
        underlying mask type (i.e. the type that would result from a comparison
        operation such as ``array < 0``). For non-array types *T*,
        :cpp:type:`type` is set to *bool*.


.. cpp:type:: template <typename T> mask_t = typename mask<T>::type

   Convenience type alias for :cpp:class:`mask`.

.. cpp:class:: template <typename T> value

    .. cpp:type:: type

        Given an Enoki array *T*, :cpp:type:`value\<T>::type` provides access to the type of
        the individual array entries. For non-array types *T*, :cpp:type:`type` is
        simply set to the template parameter *T*.

.. cpp:type:: template <typename T> value_t = typename value<T>::type

   Convenience type alias for :cpp:class:`value`.

.. cpp:class:: template <typename T> scalar

    Given a (potentially nested) Enoki array *T*, this trait class provides
    access to the scalar type underlying the array. For a nested array such as
    ``Array<Array<float, 4>, 4>``, the scalar type is ``float``, while the value
    type returned by :cpp:type:`value_t` is ``Array<float, 4>``. For non-array
    types *T*, :cpp:type:`type` is simply set to the template parameter *T*.

    .. cpp:type:: type

.. cpp:type:: template <typename T> scalar_t = typename scalar<T>::type

   Convenience type alias for :cpp:class:`scalar`.

.. cpp:class:: template <typename T> array_depth

    .. cpp:member:: static constexpr size_t value

        Given a type :cpp:any:`T` (which could be a nested Enoki array),
        :cpp:member:`value` specifies the nesting level and stores it in the
        :cpp:var:`value` member. Non-array types (e.g. ``int32_t``) have a
        nesting level of 0, a type such as ``Array<float>`` has nesting level
        1, and so on.


SFINAE helper types
-------------------

The following section discusses helper types that can be used to selectively
enable or disable template functions for Enoki arrays, e.g. like so:

.. code-block:: cpp

    template <typename Value, enable_if_array_t<Value> = 0>
    void f(Value value) {
        /* Invoked if 'Value' is an Enoki array */
    }

    template <typename Value, enable_if_not_array_t<Value> = 0>
    void f(Value value) {
        /* Invoked if 'Value' is *not* an Enoki array */
    }


Detecting Enoki arrays
**********************

.. cpp:class:: template <typename T> is_array

    .. cpp:member:: static constexpr bool value

        Equal to ``true`` iff *T* is a static or dynamic Enoki array type.

.. cpp:type:: template <typename T> enable_if_array_t = std::enable_if_t<is_array<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for Enoki
    array types.

.. cpp:type:: template <typename T> enable_if_not_array_t = std::enable_if_t<!is_array<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for types
    that are not Enoki arrays.


Detecting Enoki masks
*********************

.. cpp:class:: template <typename T> is_mask

    .. cpp:member:: static constexpr bool value

        Equal to ``true`` iff *T* is a static or dynamic Enoki mask type.

.. cpp:type:: template <typename T> enable_if_mask_t = std::enable_if_t<is_mask<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for Enoki
    mask types.

.. cpp:type:: template <typename T> enable_if_not_mask_t = std::enable_if_t<!is_mask<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for types
    that are not Enoki masks.

Detecting static Enoki arrays
*****************************

.. cpp:class:: template <typename T> is_static_array

    .. cpp:member:: static constexpr bool value

        Equal to ``true`` iff *T* is a static Enoki array type.

.. cpp:type:: template <typename T> enable_if_static_array_t = std::enable_if_t<is_static_array<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for
    static Enoki array types.

.. cpp:type:: template <typename T> enable_if_not_static_array_t = std::enable_if_t<!is_static_array<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for
    static Enoki array types.

Detecting dynamic Enoki arrays
******************************

.. cpp:class:: template <typename T> is_dynamic_array

    .. cpp:member:: static constexpr bool value

        Equal to ``true`` iff *T* is a dynamic Enoki array type.

.. cpp:type:: template <typename T> enable_if_dynamic_array_t = std::enable_if_t<is_dynamic_array<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for
    dynamic Enoki array types.

.. cpp:type:: template <typename T> enable_if_not_dynamic_array_t = std::enable_if_t<!is_dynamic_array<T>::value, int>

    SFINAE alias to selectively enable a class or function definition for
    dynamic Enoki array types.

.. cpp:class:: template <typename T> is_dynamic_nested

    .. cpp:member:: static constexpr bool value

        Equal to ``true`` iff *T* (which could be a nested Enoki array) contains
        a dynamic array at *any* level.

        This is different from :cpp:class:`is_dynamic_array`, which only cares
        about the outermost level -- for instance, given static array *T*
        containing a nested dynamic array, ``is_dynamic_array<T>::value ==
        false``, while ``is_dynamic_nested<T>::value == true``.


