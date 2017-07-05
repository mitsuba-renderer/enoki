.. cpp:namespace:: enoki

Fixed-size matrices
===================

Enoki provides a vectorizable type for small fixed-size square matrices (e.g.
:math:`2\times 2`, :math:`3\times 3`, or :math:`4\times 4`) that are commonly
found in computer graphics and vision applications.

.. note::

    Enoki attempts to store matrix entries in processor registers---it is
    unwise to work with large matrices since this will cause considerable
    pressure on the register allocator. Instead, consider using specialized
    libraries for large-scale linear algebra (e.g. Eigen or Intel MKL) for
    larger matrices.

To use this feature, include the following header:

.. code-block:: cpp

    #include <enoki/matrix.h>

Usage
-----

The following example shows how to define and perform basic arithmetic using
:cpp:class:`enoki::Matrix` to construct a :math:`4\times 4` homogeneous
coordinate "look-at" camera matrix and its inverse:

.. code-block:: cpp

    using Vector3f = Matrix<float, 3>;
    using Vector4f = Matrix<float, 4>;
    using Matrix4f = Matrix<float, 4>;

    std::pair<Matrix4f, Matrix4f> look_at(const Vector3f &origin,
                                          const Vector3f &target,
                                          const Vector3f &up) {
        Vector3f dir = normalize(target - origin);
        Vector3f left = normalize(cross(up, dir));
        Vector3f new_up = cross(dir, left);

        Matrix4f result = Matrix4f::from_cols(
            concat(left, 0.f),
            concat(new_up, 0.f),
            concat(dir, 0.f),
            concat(origin, 1.f)
        );

        /* The following two statements efficiently compute
           the inverse. Alternatively, we could have written

            Matrix4f inverse = invert(result);
        */
        Matrix4f inverse = Matrix4f::from_rows(
            concat(left, 0.f),
            concat(new_up, 0.f),
            concat(dir, 0.f),
            Vector4f(0.f, 0.f, 0.f, 1.f)
        );

        inverse[3] = inverse * concat(-origin, 1.f);

        return std::make_pair(result, inverse);
    }

Enoki can also work with fixed-size packets and dynamic arrays of matrices. The
following example shows how to compute the inverse of a matrix array.

.. code-block:: cpp

    using FloatP = Array<float, 4>;      // 4-wide packet type
    using FloatX = DynamicArray<FloatP>; // arbitrary-length array

    using MatrixP = Matrix<FloatP, 4>;   // a packet of 4x4 matrices
    using MatrixX = Matrix<FloatX, 4>;   // arbitrarily many 4x4 matrices

    MatrixX matrices;
    dynamic_resize(matrices, 1000);

    // .. fill matrices with contents ..

    // Invert all matrices
    vectorize(
        [](auto &&m) {
            m = invert(MatrixP(m));
        },
        matrices
    );

Reference
---------

.. cpp:class:: template <typename Type, size_t Size> Matrix : StaticArrayImpl<Array<Type, Size>, Size>

    The class :cpp:class:`enoki::Matrix` represents a dense square matrix of
    fixed size as a ``Size`` :math:`\times` ``Size`` Enoki array whose
    components are of type ``Type``. The implementation relies on a
    column-major storage order to enable a particularly efficient
    implementation of vectorized matrix multiplication.

    .. cpp:function:: template <typename... Columns> Matrix(Columns... columns)

        Creates a new :cpp:class:`enoki::Matrix` instance with the
        given set of columns (where ``Size == sizeof...(Columns)``)

    .. cpp:function:: Matrix(Type f)

        Creates a :cpp:class:`enoki::Matrix` instance which has the value ``f``
        on the diagonal and zeroes elsewhere.

    .. cpp:function:: Type& operator()(size_t i, size_t j)

        Returns a reference to the matrix entry :math:`(i, j)`.

    .. cpp:function:: const Type& operator()(size_t i, size_t j) const

        Returns a const reference to the matrix entry :math:`(i, j)`.

    .. cpp:function:: template <typename... Columns> static Matrix from_columns(Columns... columns)

        Creates a new :cpp:class:`enoki::Matrix` instance with the given set of
        columns (where ``Size == sizeof...(Columns)``). This is identical to
        the :cpp:func:`Matrix::Matrix()` constructor but makes it more explicit
        that the input are columns.

    .. cpp:function:: template <typename... Rows> static Matrix from_rows(Rows... rows)

        Creates a new :cpp:class:`enoki::Matrix` instance with the given set of
        rows (where ``Size == sizeof...(Rows)``).

Supported operations
********************

.. cpp:function:: template <typename T, size_t Size> Matrix<T, Size> operator*(Matrix<T, Size> m, Matrix<T, Size> v)

    Efficient vectorized matrix-matrix multiplication operation. On AVX512VL, a
    :math:`4\times 4` matrix multiplication reduces to 4 multiplications and 12 fused
    multiply-adds with embedded broadcasts.

.. cpp:function:: template <typename T, size_t Size> Array<T, Size> operator*(Matrix<T, Size> m, Array<T, Size> v)

    Matrix-vector multiplication operation.

.. cpp:function:: template <typename T, size_t Size> T trace(Matrix<T, Size> m)

    Computes the trace (i.e. sum of the diagonal elements) of the given matrix.

.. cpp:function:: template <typename T, size_t Size> Matrix<T, Size> identity()

    Returns the identity matrix.

.. cpp:function:: template <typename T, size_t Size> Matrix<T, Size> transpose(Matrix<T, Size> m)

    Computes the transpose of ``m`` using an efficient set of shuffles.

.. cpp:function:: template <typename T, size_t Size> Matrix<T, Size> invert(Matrix<T, Size> m)

    Computes the inverse of ``m`` using an efficient vectorized form of
    Cramer's rule

    .. warning::

         This function is only implemented for :math:`1\times 1`,
         :math:`2\times 2`, :math:`3\times 3`, and :math:`4\times 4` matrices.
