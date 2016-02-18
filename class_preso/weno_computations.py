"""Helper functions for ``weno_computations`` notebook."""


from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import seaborn
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


def interp_simple_stencils():
    r"""Return interpolated values for :math:`x_{j+1/2}` using simple stencils.

    First uses three sets of interpolating points,
    :math:`\left\{x_{j-2}, x_{j-1}, x_j\right\}`,
    :math:`\left\{x_{j-1}, x_j, x_{j+1}\right\}` and
    :math:`\left\{x_{j}, x_{j+1}, x_{j+2}\right\}`
    to give local order three approximations.

    Then uses all five points
    :math:`\left\{x_{j-2}, x_{j-1}, x_j, x_{j+1}, x_{j+2}\right\}` to give
    an order five approximation on the whole stencil.

    :rtype: tuple
    :returns: Quadraple of LaTeX strings, one for each set of
              interpolating points, in the order described above.
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
    # Approximate with [0, 1, 2].
    approx_all = sympy.interpolating_poly(
        5, one_half, X=[-2, -1, 0, 1, 2],
        Y=[u_minus2, u_minus1, u_zero, u_plus1, u_plus2])
    approx_all = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}'),
                                approx_all)
    approx_all = to_latex(approx_all, replace_dict)
    return approx_minus2, approx_minus1, approx_zero, approx_all


def make_intro_plots(stopping_point=None):
    """Make introductory plots.

    0, 3, 2, -1, 2
    """
    colors = seaborn.color_palette('husl')[:4]
    fontsize = 16
    num_pts = 100
    rows = cols = 2
    fig, ax_vals = plt.subplots(rows, cols, sharex=True, sharey=True)
    fig.set_size_inches(15, 8)
    ax_vals[0, 0].set_xlim(-3, 3)
    ax_vals[0, 0].set_ylim(-3, 4)

    # Top left plot (-2, -1, and 0)
    top_left = ax_vals[0, 0]
    top_left.plot(np.linspace(-2.5, -1.5, num_pts),
                  0 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    top_left.plot(np.linspace(-1.5, -0.5, num_pts),
                  3 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    top_left.plot(np.linspace(-0.5, 0.5, num_pts),
                  2 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    if stopping_point == 0:
        return
    label1 = (r'$u_{-2} \left(\frac{x^2 + x}{2}\right) +'
              r'u_{-1} \left(-x^2 - 2x\right) + '
              r'u_{0} \left(\frac{x^2 + 3x + 2}{2}\right)$')
    top_left.text(-2.5, -2, label1, fontsize=fontsize)
    x_vals = np.linspace(-2, 0, num_pts)
    y_vals = -2 * x_vals**2 - 3 * x_vals + 2
    top_left.plot(x_vals, y_vals, color=colors[0])
    if stopping_point == 1:
        return
    # Do a little bit extra (a dashed line to 0.5 and then
    # a dot at 0.5).
    x_vals = np.linspace(0, 0.5, num_pts)
    y_vals = -2 * x_vals**2 - 3 * x_vals + 2
    top_left.plot(x_vals, y_vals, color=colors[0],
                  linestyle='dashed')
    y_val1 = y_vals[-1]
    top_left.plot([0.5], [y_val1], color=colors[0],
                  linestyle='None', marker='o')
    label1_half = (r'$\frac{3}{8}u_{-2} - \frac{5}{4}u_{-1} + '
                   r'\frac{15}{8}u_{0}$')
    top_left.text(0.75, 1.9, label1_half, fontsize=fontsize)
    if stopping_point == 2:
        return

    # Top right plot (-1, 0, and 1)
    top_right = ax_vals[0, 1]
    top_right.plot(np.linspace(-1.5, -0.5, num_pts),
                   3 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    top_right.plot(np.linspace(-0.5, 0.5, num_pts),
                   2 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    top_right.plot(np.linspace(0.5, 1.5, num_pts),
                   -1 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    if stopping_point == 3:
        return
    label2 = (r'$u_{-1} \left(\frac{x^2 - x}{2}\right) +'
              r'u_{0} \left(1 -x^2\right) + '
              r'u_{1} \left(\frac{x^2 + x}{2}\right)$')
    top_right.text(-2.5, -2, label2, fontsize=fontsize)
    x_vals = np.linspace(-1, 1, num_pts)
    y_vals = -x_vals**2 - 2 * x_vals + 2
    top_right.plot(x_vals, y_vals, color=colors[1])
    if stopping_point == 4:
        return
    # Do a little bit extra: a dot at 0.5.
    y_val2 = -0.5**2 - 2 * 0.5 + 2
    top_right.plot([0.5], [y_val2], color=colors[1],
                   linestyle='None', marker='o')
    label2_half = (r'$-\frac{1}{8}u_{-1} + \frac{3}{4}u_{0} + '
                   r'\frac{3}{8}u_{1}$')
    top_right.text(0.75, 1.9, label2_half, fontsize=fontsize)
    if stopping_point == 5:
        return

    # Bottom left plot (0, 1, 2)
    bottom_left = ax_vals[1, 0]
    bottom_left.plot(np.linspace(-0.5, 0.5, num_pts),
                     2 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    bottom_left.plot(np.linspace(0.5, 1.5, num_pts),
                     -1 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    bottom_left.plot(np.linspace(1.5, 2.5, num_pts),
                     2 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    if stopping_point == 6:
        return
    label3 = (r'$u_{0} \left(\frac{x^2 - 3x + 2}{2}\right) +'
              r'u_{1} \left(-x^2 + 2x\right) + '
              r'u_{2} \left(\frac{x^2 - x}{2}\right)$')
    bottom_left.text(-2.5, -2, label3, fontsize=fontsize)
    x_vals = np.linspace(0, 2, num_pts)
    y_vals = 3 * x_vals**2 - 6 * x_vals + 2
    bottom_left.plot(x_vals, y_vals, color=colors[2])
    if stopping_point == 7:
        return
    # Do a little bit extra: a dot at 0.5.
    y_val3 = 3 * 0.5**2 - 6 * 0.5 + 2
    bottom_left.plot([0.5], [y_val3], color=colors[2],
                     linestyle='None', marker='o')
    label3_half = (r'$\frac{3}{8}u_{-1} + \frac{3}{4}u_{0} - '
                   r'\frac{1}{8}u_{1}$')
    bottom_left.text(-2.5, 2.75, label3_half, fontsize=fontsize)
    if stopping_point == 8:
        return

    # Bottom right plot (-2, -1, 0, 1 and 2)
    bottom_right = ax_vals[1, 1]
    bottom_right.plot(np.linspace(-2.5, -1.5, num_pts),
                      0 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(-1.5, -0.5, num_pts),
                      3 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(-0.5, 0.5, num_pts),
                      2 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(0.5, 1.5, num_pts),
                      -1 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(1.5, 2.5, num_pts),
                      2 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    if stopping_point == 9:
        return
    x_vals = np.linspace(-2, 2, num_pts)
    y_vals = (3 * x_vals**4 + 10 * x_vals**3 - 15 * x_vals**2 -
              34 * x_vals + 24) / 12.0
    bottom_right.plot(x_vals, y_vals, color=colors[3])
    if stopping_point == 10:
        return
    # Do a little bit extra: a dot at 0.5.
    y_val4 = (3 * 0.5**4 + 10 * 0.5**3 - 15 * 0.5**2 -
              34 * 0.5 + 24) / 12.0
    bottom_right.plot([0.5], [y_val4], color=colors[3],
                      linestyle='None', marker='o')
    label4_half = (r'$\frac{3}{128}u_{-2} - \frac{5}{32}u_{-1} + '
                   r'\frac{45}{64}u_{0} + \frac{15}{32}u_{1} -'
                   r'\frac{5}{128}u_{2}$')
    bottom_right.text(-2.5, -2, label4_half, fontsize=fontsize)
    if stopping_point == 11:
        return
    # Do a little bit more: add all the other dots.
    bottom_right.plot([0.5], [y_val1], color=colors[0],
                      linestyle='None', marker='o')
    bottom_right.plot([0.5], [y_val2], color=colors[1],
                      linestyle='None', marker='o')
    bottom_right.plot([0.5], [y_val3], color=colors[2],
                      linestyle='None', marker='o')
