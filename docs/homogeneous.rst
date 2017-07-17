.. cpp:namespace:: enoki

Homogeneous transformations
===========================

Enoki provides a number of convenience functions to construct 3D homogeneous
coordinate transformations (rotations, translations, scales, perspective
transformation matrices, etc.). To use them, include the following header file:

.. code-block:: cpp

    #include <enoki/homogeneous.h>

Reference
---------

.. cpp:function:: template <typename Matrix, typename Vector3> Matrix translate(Vector3 v)

    Constructs a homogeneous transformation, which translates points by ``v``.

.. cpp:function:: template <typename Matrix, typename Vector3> Matrix scale(Vector3 v)

    Constructs a homogeneous transformation, which scales points by ``v``.

.. cpp:function:: template <typename Matrix, typename Vector3, typename Float> Matrix rotate(Vector3 v, Float angle)

    Constructs a homogeneous transformation, which rotates by ``angle`` radians
    around the axis ``v``. The function requires ``v`` to be normalized.

.. cpp:function:: template <typename Matrix, typename Float> Matrix perspective(Float fov, Float near, Float far)

    Constructs a perspective projection matrix with the specified field of view
    (in radians) and near and far clip planes. The returned matrix performs the
    transformation

    .. math::

        \begin{pmatrix}
        x\\y\\z\end{pmatrix}
        \mapsto
        \begin{pmatrix}
        c\,x/z\\ c\,x/z\\
        \frac{\mathrm{far}\,(z-\mathrm{near})}{z\, (\mathrm{far}-\mathrm{near})}
        \end{pmatrix},

    where

    .. math::

        c = \mathrm{cot}\left(0.5\, \textrm{fov}\right),

    which maps :math:`(0, 0, \mathrm{near})^T` to :math:`(0, 0, 0)^T` and
    :math:`(0, 0, \mathrm{far})^T` to :math:`(0, 0, 1)^T`. See also
    :cpp:func:`perspective_gl` for an OpenGL-style perspective matrix.


.. cpp:function:: template <typename Matrix, typename Float> Matrix perspective_gl(Float fov, Float near, Float far)

    Constructs an OpenGL-compatible perspective projection matrix with the
    specified field of view (in radians) and near and far clip planes. The
    returned matrix performs the transformation

    .. math::

        \begin{pmatrix}
        x\\y\\z\end{pmatrix}
        \mapsto
        \begin{pmatrix}
        -c\,x/z\\ -c\,x/z\\
        \frac{2\,\mathrm{far}\,\mathrm{near}+z\,(\mathrm{far}+\mathrm{near})}{z\, (\mathrm{far}-\mathrm{near})}
        \end{pmatrix},

    where

    .. math::

        c = \mathrm{cot}\left(0.5\, \textrm{fov}\right),

    which maps :math:`(0, 0, -\mathrm{near})^T` to :math:`(0, 0, -1)^T` and
    :math:`(0, 0, -\mathrm{far})^T` to :math:`(0, 0, 1)^T`. See also
    :cpp:func:`perspective` for a different convention used in some
    rendering systems.

