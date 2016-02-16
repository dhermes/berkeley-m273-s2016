"""Helper functions for ``weno_computations`` notebook."""


from __future__ import print_function

import matplotlib.pyplot as plt
import six
import sympy


X_SYMB = sympy.Symbol('X')


def poly_interp(values, start_index):
    r"""Interpolate a polynomial at ``values``.

    Assumes :math:`\Delta x = 1` and uses ``start_index``
    as the beginning value of :math:`j`.

    :type values: list
    :param values: List of values being interpolated.

    :type start_index: int
    :param start_index: The index where the stencil starts.

    :rtype: :class:`sympy.core.expr.Expr`
    :returns: The interpolated function.
    """
    num_values = len(values)
    interp_poly = sympy.interpolating_poly(num_values, X_SYMB)
    var_subs = {}
    for index in six.moves.xrange(num_values):
        x_var = sympy.Symbol('x%d' % (index,))
        var_subs[x_var] = index + start_index
        y_var = sympy.Symbol('y%d' % (index,))
        var_subs[y_var] = values[index]
    return interp_poly.subs(var_subs)


def interp_three_points():
    r"""Return interpolated values for :math:`x_{j+1/2}` using three points.

    Uses three sets of interpolating points, :math:`x_{j-2}, x_{j-1}, x_j`,
    :math:`x_{j-1}, x_j, x_{j+1}` and :math:`x_{j}, x_{j+1}, x_{j+2}`.

    :rtype: tuple
    :returns: Triple of :mod:`sympy` expressions, one for each set of
              interpolating points.
    """
    one_half = sympy.Rational(1, 2)
    u_minus2, u_minus1, u0, u1, u2 = sympy.symbols(
        'u_{j-2}, u_{j-1}, u_{j}, u_{j+1}, u_{j+2}')
    # Approximate with [-2, -1, 0].
    p2_minus2 = poly_interp([u_minus2, u_minus1, u0], -2)
    approx_minus2 = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(1)}'),
                                   p2_minus2.subs({X_SYMB: one_half}))
    # Approximate with [-1, 0, 1].
    p2_minus1 = poly_interp([u_minus1, u0, u1], -1)
    approx_minus1 = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(2)}'),
                                   p2_minus1.subs({X_SYMB: one_half}))
    # Approximate with [0, 1, 2].
    p2_0 = poly_interp([u0, u1, u2], 0)
    approx_0 = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(3)}'),
                              p2_0.subs({X_SYMB: one_half}))
    return approx_minus2, approx_minus1, approx_0
