"""Helper functions for ``weno_computations`` notebook."""


from __future__ import print_function

import six
import sympy


def to_latex(value, replace_dict):
    """Convert an expression to LaTeX.

    This method is required so we can get a specified ordering
    for terms that may not have the desired lexicographic
    ordering.

    :type value: :class:`sympy.core.expr.Expr`
    :param value: A

    :type replace_dict: dict
    :param replace_dict: Dictionary where keys are old variable names (as
                         strings) and values are the new variable names
                         to replace them with.

    :rtype: str
    :returns: The value as LaTeX, with all variables replaced.
    """
    result = sympy.latex(value)
    for old, new_ in six.iteritems(replace_dict):
        result = result.replace(old, new_)
    return result


def interp_three_points():
    r"""Return interpolated values for :math:`x_{j+1/2}` using three points.

    Uses three sets of interpolating points, :math:`x_{j-2}, x_{j-1}, x_j`,
    :math:`x_{j-1}, x_j, x_{j+1}` and :math:`x_{j}, x_{j+1}, x_{j+2}`.

    :rtype: tuple
    :returns: Triple of LaTeX strings, one for each set of
              interpolating points.
    """
    one_half = sympy.Rational(1, 2)
    # Intentionally use values which are simple to replace and
    # ordered lexicographically.
    u_minus2, u_minus1, u_zero, u_plus1, u_plus2 = sympy.symbols(
        'A, B, C, D, E')
    replace_dict = {
        'A': 'u_{j-2}',
        'B': 'u_{j-1}',
        'C': 'u_{j}',
        'D': 'u_{j+1}',
        'E': 'u_{j+2}',
    }
    # Approximate with [-2, -1, 0].
    approx_minus2 = sympy.interpolating_poly(
        3, one_half, X=[-2, -1, 0],
        Y=[u_minus2, u_minus1, u_zero])
    approx_minus2 = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(1)}'),
                                   approx_minus2)
    approx_minus2 = to_latex(approx_minus2, replace_dict)
    # Approximate with [-1, 0, 1].
    approx_minus1 = sympy.interpolating_poly(
        3, one_half, X=[-1, 0, 1],
        Y=[u_minus1, u_zero, u_plus1])
    approx_minus1 = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(2)}'),
                                   approx_minus1)
    approx_minus1 = to_latex(approx_minus1, replace_dict)
    # Approximate with [0, 1, 2].
    approx_zero = sympy.interpolating_poly(
        3, one_half, X=[0, 1, 2],
        Y=[u_zero, u_plus1, u_plus2])
    approx_zero = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(3)}'),
                                 approx_zero)
    approx_zero = to_latex(approx_zero, replace_dict)
    return approx_minus2, approx_minus1, approx_zero
