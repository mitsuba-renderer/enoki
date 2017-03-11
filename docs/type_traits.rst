Type traits
===========

The following type traits are available to query the properties of arrays at
compile time.

1. To detect Enoki arrays
-------------------------

.. cpp:namespace:: enoki 

.. cpp:class:: template <typename T> is_array

    Checks whether *T* is a static or dynamic Enoki array -- if so, the the
    :cpp:var:`value` member is set to *true*. Otherwise :cpp:var:`value` is
    equal to *false*.

    .. cpp:member:: static constexpr bool value

.. cpp:type:: template <typename T> enable_if_array_t = std::enable_if_t<is_array<T>::value, int>

    This convenience type alias can be used to selectively enable a class or
    function definition for Enoki arrays using SFINAE.

.. cpp:type:: template <typename T> enable_if_not_array_t = std::enable_if_t<!is_array<T>::value, int>
    
    This convenience type alias can be used to selectively disable a class or
    function definition for Enoki arrays using SFINAE.

.. cpp:class:: template <typename T> is_static_array

    Checks whether *T* is a static Enoki array -- if so, the the
    :cpp:var:`value` member is set to *true*. Otherwise :cpp:var:`value` is
    equal to *false*.

    .. cpp:member:: static constexpr bool value

.. cpp:type:: template <typename T> enable_if_static_array_t = std::enable_if_t<is_static_array<T>::value, int>

    This convenience type alias can be used to selectively enable a class or
    function definition for static Enoki arrays using SFINAE.

.. cpp:class:: template <typename T> is_dynamic_array

    Checks whether *T* is a dynamic Enoki array -- if so, the the
    :cpp:var:`value` member is set to *true*. Otherwise :cpp:var:`value` is
    equal to *false*.

    .. cpp:member:: static constexpr bool value

.. cpp:type:: template <typename T> enable_if_dynamic_array_t = std::enable_if_t<is_dynamic_array<T>::value, int>
    
    This convenience type alias can be used to selectively enable a class or
    function definition for dynamic Enoki arrays using SFINAE.

.. cpp:class:: template <typename T> is_dynamic_nested

    Checks whether *T* (which could be a nested Enoki array) contains a
    dynamic array at *any* level. If so, the the :cpp:var:`value` member is set to
    *true*. Otherwise :cpp:var:`value` is equal to *false*.
    
    This is different from :cpp:class:`is_dynamic_array`, which only cares
    about the outermost level -- for instance, given static array *T*
    containing a nested dynamic array, ``is_dynamic_array<T> == false`` while
    ``is_dynamic_nested<T> == true``.

    .. cpp:member:: static constexpr bool value

.. cpp:type:: template <typename T> enable_if_dynamic_nested_t = std::enable_if_t<is_dynamic_nested<T>::value, int>
    
    This convenience type alias can be used to selectively enable a class or
    function definition for a type containing a dynamic Enoki array using SFINAE.


2. To inspect Enoki arrays
--------------------------

.. cpp:class:: template <typename T> mask

    Given an Enoki array *T*, this trait class provides access to the
    underlying mask type (i.e. the type that would result from a comparison
    operation such as ``array < 0``). For non-array types *T*, :cpp:type:`type`
    is set to *bool*.

    .. cpp:type:: type

.. cpp:type:: template <typename T> mask_t = typename mask<T>::type

   Convenience type alias for :cpp:class:`mask`.

.. cpp:class:: template <typename T> value

    Given an Enoki array *T*, this trait class provides access to the type of
    the individual array entries. For non-array types *T*, :cpp:type:`type` is
    simply set to the template parameter *T*.

    .. cpp:type:: type

.. cpp:type:: template <typename T> value_t = typename value<T>::type

   Convenience type alias for :cpp:class:`mask`.

.. cpp:class:: template <typename T> scalar

    Given a (potentially nested) Enoki array *T*, this trait class provides
    access to the scalar type underlying the array. For a nested array such as
    *Array<Array<float, 4>, 4>*, the scalar type is *float*, while the value
    type returned by :cpp:type:`value_t` is *Array<float, 4>*. For non-array
    types *T*, :cpp:type:`type` is simply set to the template parameter *T*.

    .. cpp:type:: type

.. cpp:type:: template <typename T> scalar_t = typename scalar<T>::type

   Convenience type alias for :cpp:class:`scalar`.

.. cpp:class:: template <typename T> array_depth

    Given a type *T* (which could be a nested Enoki array), this trait computes
    the nesting level and stores it in the :cpp:var:`value` member. Non-array
    types (e.g. *int32_t*) have a nesting level of 0, a type such as
    *Array<float>* has nesting level 1, and so on.

    .. cpp:member:: static constexpr size_t value

