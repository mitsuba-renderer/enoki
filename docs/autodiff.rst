.. cpp:namespace:: enoki
.. _autodiff:

Automatic differentiation
=========================

Automatic differentiation (AD) broadly refers to a set of techniques that
numerically evaluate the gradient of a computer program. Two variants of AD are
widely used:

1. **Forward mode**. Starting with a set of inputs (e.g. ``a`` and ``b``) and
   associated derivatives (``da`` and ``db``), *forward-mode* AD instruments
   every operation (e.g. ``c = a * b``) with an additional step that tracks the
   evolution of derivatives (``dc = a*db + b*da``). Forward mode is ideal for
   functions with a single input and many outputs. When the function has
   multiple inputs, a separate propagation pass is needed per parameter, which
   can become very costly.

2. **Reverse mode**. On the other hand, *reverse-mode* AD specifically targets
   the case where the function to be differentiated has one (or few) outputs
   and potentially many inputs. It traverses the computational graph from
   outputs to inputs, repeatedly evaluating the chain rule in reverse. This
   approach is also known as *backpropagation* in the context of neural
   networks.

   One tricky aspect of reverse-mode is that the backward traversal can only
   begin once the output has been computed. A partial record of intermediate
   computations must furthermore be kept in memory, which can become costly for
   long-running computations.

The implementation in Enoki is realized via the special ``DiffArray<T>`` array
type and supports both of the above variants, though it is particularly
optimized for reverse mode operation. The template argument ``T`` refers to an
arithmetic type (e.g. ``float`` or an Enoki array) that is used to carry out
the underlying primal and derivative computation.

Due to the need to maintain a computation graph for the reverse-mode traversal,
AD tends to involve massive amounts of memory traffic, making it an ideal fit
for GPU-resident arrays that provide a higher amount of memory bandwidth. For
this reason, combinations such as ``DiffArray<CUDAArray<float>>`` should be
considered the default choice. Enoki's Python bindings expose this type as
``FloatD``. As in the previous section, we will stick to the interactive Python
interface and postpone a discussion of the C++ side until the end of this
section.

A ``FloatD`` instance consists of two parts: a floating point array that is
used during the original computation (of type ``FloatC``), and an index that
refers to a node in a separately maintained directed acyclic graph capturing
the structure of the differentiable portion of the computation. By default, the
index is set to zero, which indicates that the variable does not participate in
automatic differentiation.

The following two example snippets demonstrate usage of automatic
differentiation. In both cases, the :cpp:func:`set_requires_gradient` function
is used to mark a variable as being part of a differentiable computation.

.. code-block:: python

    >>> from enoki import *

    >>> # Create a differentiable variable
    >>> a = FloatD(2.0)
    >>> set_requires_gradient(a)

    >>> # Arithmetic with one input ('a') and multiple outputs ('b', 'c')
    >>> b = a * a
    >>> c = sqrt(a)

    >>> # Forward-propagate gradients to outputs
    >>> forward(a)
    autodiff: forward(): processed 3/5 nodes.

    >>> gradient(b), gradient(c)
    ([4], [0.353553])

The :cpp:func:`forward` and :cpp:func:`backward` function realize the two
previously discussed AD variants, and :cpp:func:`gradient` extracts the
gradient associated with a differentiable variable. An example of reverse-mode
traversal is shown next:

.. code-block:: python

    >>> from enoki import *

    >>> # Create multiple differentiable input variables
    >>> a, b = FloatD(2.0), FloatD(3.0)
    >>> set_requires_gradient(a)
    >>> set_requires_gradient(b)

    >>> # Arithmetic with two inputs ('a', 'b') and a single output ('c')
    >>> c = a * sqrt(b)

    >>> # Backward-propagate gradients to inputs
    >>> backward(c)
    autodiff: backward(): processed 3/4 nodes.

    >>> gradient(a), gradient(b)
    ([1.73205], [0.57735])

Note that :cpp:func:`gradient` returns the gradient using the wrapped arithmetic
type, which is a ``FloatC`` instance in this case. Another function named
:cpp:func:`detach` can be used to extract the value using the underlying
(non-differentiable) array type. Using these two operations, a gradient descent
step on a parameter ``a`` would be realized as follows:

.. code-block:: python

    >>> a = FloatD(detach(a) + step_size * gradient(a))

Visualizing computation graphs
------------------------------

It is possible to visualize the graph of the currently active computation using
the :cpp:func:`graphviz` function. You may also want to assign explicit
variable names via  :cpp:func:`set_label` to make the visualization easier to
parse. An example is shown below:

.. code-block:: python

    >>> a = FloatD(1.0)
    >>> set_requires_gradient(a)
    >>> b = erf(a)
    >>> set_label(a, 'a')
    >>> set_label(b, 'b')

    >>> print(graphviz(b))
    digraph {
      rankdir=RL;
      fontname=Consolas;
      node [shape=record fontname=Consolas];
      1 [label="'a' [s]\n#1 [E/I: 1/5]" fillcolor=salmon style=filled];
      3 [label="mul [s]\n#3 [E/I: 0/4]"];
      ... 111 lines skipped ...
      46 -> 12;
      46 [fillcolor=cornflowerblue style=filled];
    }

The resulting string can be visualized via Graphviz, which reveals the
numerical approximation used to evaluate the error function :cpp:func:`erf`.

.. figure:: autodiff-01.svg
    :width: 800px
    :align: center

The combination of Enoki's JIT compiler and AD has interesting consequences:
computation related to derivatives is queued up along with primal arithmetic
and can thus be compiled to into a joint GPU kernel. 

For example, if a forward computation evaluates the expression :math:`\sin(x)`,
the weight of the associated backward edge in the computation graph is given by
:math:`\cos(x)`. The computation of both of these quantities is automatically
merged into a single joint kernel, leveraging subexpression elimination and
constant folding to further improve efficiency.

For the previous example involving the error function, :cpp:func:`cuda_whos`
introduced in the last section reveals that many variables relating to both
primal and gradient computations have been scheduled (but not executed yet).

.. code-block:: python

    >>> cuda_whos()

      ID        Type   E/I Refs   Size        Memory     Ready    Label
      =================================================================
      10        f32    3 / 11     1           4 B         [ ]     a
      11        f32    1 / 0      1           4 B         [ ]     a.grad
      16        f32    0 / 1      1           4 B         [ ]     
      17        f32    0 / 1      1           4 B         [ ]     
      ... 117 lines skipped ...
      150       f32    1 / 0      1           4 B         [ ]     b
      151       f32    0 / 1      1           4 B         [ ]     
      152       f32    0 / 1      1           4 B         [ ]     
      153       f32    1 / 0      1           4 B         [ ]     
      154       f32    0 / 1      1           4 B         [ ]     
      155       f32    0 / 1      1           4 B         [ ]     
      156       f32    1 / 0      1           4 B         [ ]     
      =================================================================

      Memory usage (ready)     : 0 B
      Memory usage (scheduled) : 0 B + 268 B = 268 B
      Memory savings           : 235 B

..

    TODO: Graph simplification, larger rotation example with FloatD.scope,
    autodiffing scatter/gathers? Efficiency difference to PyTorch. back-propagating
    multiple weighted variables.
