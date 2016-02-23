r"""Symbolic helper for :mod:`assignment1.dg1`.

Provides exact values for stiffness and mass matrices using
symbolic algebra.

For example, :mod:`assignment1.dg1` previously used pre-computed mass and
stiffness matrices from this module. These were created using evenly spaced
points on :math:`\left[0, 1\right]` for small :math:`p`. These values
can be verified by :func:`find_matrices_symbolic` below.
"""


import six
import sympy


def get_symbolic_vandermonde(p_order, x_vals=None):
    r"""Get symbolic Vandermonde matrix of evenly spaced points.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type x_vals: list
    :param x_vals: (Optional) The list of :math:`x`-values to use. If not
                   given, defaults to ``p_order + 1`` evenly spaced points
                   on :math:`\left[0, 1\right]`.

    :rtype: tuple
    :returns: Pair of vector of powers of :math:`x` and Vandermonde matrix.
              Both are type
              :class:`sympy.Matrix <sympy.matrices.dense.MutableDenseMatrix>`,
              the ``x_vec`` is a row vector with ``p_order + 1`` columns and
              the Vandermonde matrix is square of dimension ``p_order + 1``.
    """
    x_symb = sympy.Symbol('x')
    if x_vals is None:
        x_vals = sympy.Matrix(six.moves.xrange(p_order + 1)) / p_order
    vand_mat = sympy.zeros(p_order + 1, p_order + 1)
    x_vec = sympy.zeros(1, p_order + 1)
    for i in six.moves.xrange(p_order + 1):
        x_vec[i] = x_symb**i
        for j in six.moves.xrange(p_order + 1):
            vand_mat[i, j] = x_vals[i]**j
    return x_vec, vand_mat


# pylint: disable=too-many-locals
def find_matrices_symbolic(p_order, start=0, stop=1, x_vals=None):
    r"""Find mass and stiffness matrices using symbolic algebra.

    We do this on the reference interval :math:`\left[0, 1\right]`
    with the evenly spaced points

    .. math::

       x_0 = 0, x_1 = 1/p, \ldots, x_p = 1

    and compute the polynomials :math:`\varphi_j(x)` such that
    :math:`\varphi_j\left(x_i\right) = \delta_{ij}`. Since we
    are using symbolic rationals numbers, we do this directly by
    inverting the Vandermonde matrix :math:`V` such that

        .. math::

           \left[ \begin{array}{c c c c}
                    1 & x_0 & \cdots & x_0^p  \\
                    1 & x_1 & \cdots & x_1^p  \\
               \vdots &     &        & \vdots \\
                    1 & x_p & \cdots & x_p^p
           \end{array}\right]
           \left[ \begin{array}{c}
               c_0 \\ c_1 \\ \vdots \\ c_p
           \end{array}\right]
           = \left(\delta_{ij}\right) = I_{p + 1}

    Then use these to compute the mass matrix

    .. math::

        M_{ij} = \int_0^1 \varphi_i(x) \varphi_j(x) \, dx

    and the stiffness matrix

    .. math::

        K_{ij} = \int_0^1 \varphi_i'(x) \varphi_j(x) \, dx

    Some previously used precomputed values for evenly spaced
    points on :math:`\left[0, 1\right]` are

    .. math::

       \begin{align*}
       M_1 = \frac{1}{6} \left[ \begin{array}{c c}
                          2 & 1 \\
                          1 & 2
                        \end{array}\right]
       & \qquad
       K_1 = \frac{1}{2} \left[ \begin{array}{c c}
                          -1 & -1 \\
                           1 &  1
                        \end{array}\right]
       \\
       M_2 = \frac{1}{30} \left[ \begin{array}{c c c}
                            4 &  2 & -1 \\
                            2 & 16 &  2 \\
                           -1 &  2 &  4
                         \end{array}\right]
       & \qquad
       K_2 = \frac{1}{6} \left[ \begin{array}{c c c}
                          -3 & -4 &  1 \\
                           4 &  0 & -4 \\
                          -1 &  4 &  3
                        \end{array}\right]
       \\
       M_3 = \frac{1}{1680} \left[ \begin{array}{c c c c}
                              128 &   99 &  -36 &   19 \\
                               99 &  648 &  -81 &  -36 \\
                              -36 &  -81 &  648 &   99 \\
                               19 &  -36 &   99 &  128
                           \end{array}\right]
       & \qquad
       K_3 = \frac{1}{80} \left[ \begin{array}{c c c c}
                            -40 &  -57 &   24 &   -7 \\
                             57 &    0 &  -81 &   24 \\
                            -24 &   81 &    0 &  -57 \\
                              7 &  -24 &   57 &   40
                         \end{array}\right]
       \end{align*}

    In addition, when :math:`p = 3`, the Gauss-Lobatto nodes

    .. math::

        x_0 = -1, \; x_1 = -\frac{1}{\sqrt{5}}, \;
        x_2 = \frac{1}{\sqrt{5}}, \; x_4 = 1

    are **not** evenly spaced for the first time. These produce

    .. math::

       M_3 = \frac{1}{42} \left[ \begin{array}{c c c c}
                                     6 &  \sqrt{5} & -\sqrt{5} &  1        \\
                              \sqrt{5} &        30 &         5 & -\sqrt{5} \\
                             -\sqrt{5} &         5 &        30 &  \sqrt{5} \\
                                     1 & -\sqrt{5} &  \sqrt{5} &  6
                           \end{array}\right]

    and

    .. math::

       K_3 = \frac{1}{24} \left[ \begin{array}{c c c c}
                              -12 & -5 & -5 & -2 \\
                                5 &  0 &  0 & -5 \\
                                5 &  0 &  0 & -5 \\
                                2 &  5 &  5 & 12
                           \end{array}\right]
       + \frac{\sqrt{5}}{24} \left[ \begin{array}{c c c c}
                                0 & -5 &   5 &  0 \\
                                5 &  0 & -10 &  5 \\
                               -5 & 10 &   0 & -5 \\
                                0 & -5 &   5 &  0
                           \end{array}\right]

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type start: :class:`sympy.core.expr.Expr`
    :param start: (Optional) The beginning of the interval.
                  Defaults to 0.

    :type stop: :class:`sympy.core.expr.Expr`
    :param stop: (Optional) The end of the interval. Defaults to 1.

    :type x_vals: list
    :param x_vals: (Optional) The list of :math:`x`-values to use. If not
                   given, defaults to ``p_order + 1`` evenly spaced points
                   on :math:`\left[0, 1\right]`.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, square
              :class:`sympy.Matrix <sympy.matrices.dense.MutableDenseMatrix>`
              with rows/columns equal to ``p_order + 1``.
    """
    x_symb = sympy.Symbol('x')
    x_vec, vand_mat = get_symbolic_vandermonde(p_order, x_vals=x_vals)
    coeff_mat = vand_mat**(-1)
    phi_funcs = x_vec * coeff_mat
    phi_funcs = phi_funcs.expand()
    phi_funcs.simplify()

    mass_mat = sympy.zeros(p_order + 1, p_order + 1)
    stiffness_mat = sympy.zeros(p_order + 1, p_order + 1)
    for i in six.moves.xrange(p_order + 1):
        phi_i = phi_funcs[i]
        phi_i_prime = sympy.diff(phi_i, x_symb)
        for j in six.moves.xrange(i, p_order + 1):
            phi_j = phi_funcs[j]
            integral_m = sympy.integrate(phi_i * phi_j, x_symb)
            integral_k = sympy.integrate(phi_i_prime * phi_j, x_symb)
            mass_mat[i, j] = (integral_m.subs({x_symb: stop}) -
                              integral_m.subs({x_symb: start}))
            stiffness_mat[i, j] = (integral_k.subs({x_symb: stop}) -
                                   integral_k.subs({x_symb: start}))
            if j > i:
                mass_mat[j, i] = mass_mat[i, j]
                stiffness_mat[j, i] = -stiffness_mat[i, j]

    return mass_mat, stiffness_mat
# pylint: enable=too-many-locals
