"""Helper functions for ``weno_computations`` notebook.

Slides can be seen on `nbviewer`_.

.. _nbviewer: https://nbviewer.jupyter.org/format/slides/github/\
              dhermes/berkeley-m273-s2016/blob/master/class_preso/\
              weno_computations.ipynb
"""


from __future__ import print_function

# pylint: disable=wrong-import-order
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import six
import sympy
# pylint: enable=wrong-import-order
# NOTE: The import order fails on Travis but I can't reproduce locally.
#       This seems to be a confusion about the order of precedence
#       1. std lib 2. external libs 3. projects modules.


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


# pylint: disable=too-many-locals
def interp_simple_stencils():
    r"""Return interpolated values for :math:`u_{j+1/2}` using simple stencils.

    First uses three sets of interpolating values,

    .. math::

        \left\{\overline{u}_{j-2}, \overline{u}_{j-1}, \overline{u}_j\right\},
        \left\{\overline{u}_{j-1}, \overline{u}_j,
            \overline{u}_{j+1}\right\},
        \left\{\overline{u}_j, \overline{u}_{j+1}, \overline{u}_{j+2}\right\},

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
    x_symb = sympy.Symbol('X')
    u_minus2, u_minus1, u_zero, u_plus1, u_plus2 = sympy.symbols(
        'A, B, C, D, E')
    replace_dict = {
        'A': r'\overline{u}_{j-2}',
        'B': r'\overline{u}_{j-1}',
        'C': r'\overline{u}_{j}',
        'D': r'\overline{u}_{j+1}',
        'E': r'\overline{u}_{j+2}',
    }

    # Approximate with [-2, -1, 0].
    x_vals = [-3 + one_half, -2 + one_half, -1 + one_half, one_half]
    y_vals = [
        0,
        u_minus2,
        u_minus2 + u_minus1,
        u_minus2 + u_minus1 + u_zero,
    ]
    anti_derivative_minus2 = sympy.interpolating_poly(
        4, x_symb, X=x_vals, Y=y_vals)
    # Evaluate derivative at one-half.
    approx_minus2 = anti_derivative_minus2.diff(x_symb).subs(
        {x_symb: one_half})
    approx_minus2 = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(1)}'),
                                   approx_minus2)
    approx_minus2 = to_latex(approx_minus2, replace_dict)

    # Approximate with [-1, 0, 1].
    x_vals = [-2 + one_half, -1 + one_half, one_half, 1 + one_half]
    y_vals = [
        0,
        u_minus1,
        u_minus1 + u_zero,
        u_minus1 + u_zero + u_plus1,
    ]
    anti_derivative_minus1 = sympy.interpolating_poly(
        4, x_symb, X=x_vals, Y=y_vals)
    # Evaluate derivative at one-half.
    approx_minus1 = anti_derivative_minus1.diff(x_symb).subs(
        {x_symb: one_half})
    approx_minus1 = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(2)}'),
                                   approx_minus1)
    approx_minus1 = to_latex(approx_minus1, replace_dict)

    # Approximate with [0, 1, 2].
    x_vals = [-1 + one_half, one_half, 1 + one_half, 2 + one_half]
    y_vals = [
        0,
        u_zero,
        u_zero + u_plus1,
        u_zero + u_plus1 + u_plus2,
    ]
    anti_derivative_zero = sympy.interpolating_poly(
        4, x_symb, X=x_vals, Y=y_vals)
    # Evaluate derivative at one-half.
    approx_zero = anti_derivative_zero.diff(x_symb).subs(
        {x_symb: one_half})
    approx_zero = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}^{(3)}'),
                                 approx_zero)
    approx_zero = to_latex(approx_zero, replace_dict)

    # Approximate with [-2, -1, 0, 1, 2].
    x_vals = [-3 + one_half, -2 + one_half, -1 + one_half, one_half,
              1 + one_half, 2 + one_half]
    y_vals = [
        0,
        u_minus2,
        u_minus2 + u_minus1,
        u_minus2 + u_minus1 + u_zero,
        u_minus2 + u_minus1 + u_zero + u_plus1,
        u_minus2 + u_minus1 + u_zero + u_plus1 + u_plus2,
    ]
    anti_derivative_all = sympy.interpolating_poly(
        6, x_symb, X=x_vals, Y=y_vals)
    # Evaluate derivative at one-half.
    approx_all = anti_derivative_all.diff(x_symb).subs(
        {x_symb: one_half})
    approx_all = sympy.Equality(sympy.Symbol(r'u_{j + \frac{1}{2}}'),
                                approx_all)
    approx_all = to_latex(approx_all, replace_dict)
    return approx_minus2, approx_minus1, approx_zero, approx_all
# pylint: enable=too-many-locals


# pylint: disable=too-many-locals,too-many-return-statements
# pylint: disable=too-many-statements
def make_intro_plots(stopping_point=None):
    r"""Make introductory plots.

    Uses

    .. math::

        \overline{u}_{-2} = 0,
        \overline{u}_{-1} = 3,
        \overline{u}_{0} = 2,
        \overline{u}_{1} = -1,
        \overline{u}_{2} = 2

    And plots the interpolations by quadratics (on the three contiguous
    subregions) and by a quartic that preserve the interval.
    """
    colors = seaborn.color_palette('husl')[:4]
    fontsize = 16
    num_pts = 100
    rows = cols = 2
    fig, ax_vals = plt.subplots(rows, cols, sharex=True, sharey=True)
    fig.set_size_inches(15, 8)
    ax_vals[0, 0].set_xlim(-3, 3)
    ax_vals[0, 0].set_ylim(-4, 6)

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
    label1 = (r'$\overline{u}_{-2} \left(\frac{12x^2 + 12x - 1}{24}\right) + '
              r'\overline{u}_{-1} \left(\frac{-12x^2 - 24x + 1}{12}\right) + '
              r'\overline{u}_{0} \left(\frac{12x^2 + 36x + 23}{24}\right)$')
    top_left.text(-3, -3, label1, fontsize=fontsize)
    x_vals = np.linspace(-2.5, 0.5, num_pts)
    y_vals = (-12 * x_vals**2 - 18 * x_vals + 13) / 6.0
    top_left.fill_between(x_vals, 0, y_vals,
                          color=colors[0], alpha=0.5)
    if stopping_point == 1:
        return
    # Do a little bit extra: a dot at 0.5.
    y_val1 = (-12 * 0.5**2 - 18 * 0.5 + 13) / 6.0
    top_left.plot([0.5], [y_val1], color=colors[0],
                  linestyle='None', marker='o')
    label1_half = (r'$\frac{1}{3}\overline{u}_{-2} - '
                   r'\frac{7}{6}\overline{u}_{-1} + '
                   r'\frac{11}{6}\overline{u}_{0}$')
    top_left.text(0.75, 2.5, label1_half, fontsize=fontsize)
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
    x_vals = np.linspace(-1.5, 1.5, num_pts)
    y_vals = (-12 * x_vals**2 - 24 * x_vals + 25) / 12.0
    top_right.fill_between(x_vals, 0, y_vals,
                           color=colors[1], alpha=0.5)
    if stopping_point == 4:
        return
    # Do a little bit extra: a dot at 0.5.
    y_val2 = (-12 * 0.5**2 - 24 * 0.5 + 25) / 12.0
    top_right.plot([0.5], [y_val2], color=colors[1],
                   linestyle='None', marker='o')
    label2_half = (r'$-\frac{1}{6}\overline{u}_{-1} + '
                   r'\frac{5}{6}\overline{u}_{0} + '
                   r'\frac{1}{3}\overline{u}_{1}$')
    top_right.text(0.75, 2.5, label2_half, fontsize=fontsize)
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
    x_vals = np.linspace(-0.5, 2.5, num_pts)
    y_vals = (12 * x_vals**2 - 24 * x_vals + 7) / 4.0
    bottom_left.fill_between(x_vals, y_vals,
                             color=colors[2], alpha=0.5)
    if stopping_point == 7:
        return
    # Do a little bit extra: a dot at 0.5.
    y_val3 = (12 * 0.5**2 - 24 * 0.5 + 7) / 4.0
    bottom_left.plot([0.5], [y_val3], color=colors[2],
                     linestyle='None', marker='o')
    label3_half = (r'$\frac{1}{3}\overline{u}_{0} + '
                   r'\frac{5}{6}\overline{u}_{1} - '
                   r'\frac{1}{6}\overline{u}_{2}$')
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
    x_vals = np.linspace(-2.5, 2.5, num_pts)
    y_vals = (240 * x_vals**4 + 800 * x_vals**3 - 1320 * x_vals**2 -
              2920 * x_vals + 2027) / 960.0
    bottom_right.fill_between(x_vals, y_vals,
                              color=colors[3], alpha=0.5)
    if stopping_point == 10:
        return
    # Do a little bit extra: a dot at 0.5.
    y_val4 = (240 * 0.5**4 + 800 * 0.5**3 - 1320 * 0.5**2 -
              2920 * 0.5 + 2027) / 960.0
    bottom_right.plot([0.5], [y_val4], color=colors[3],
                      linestyle='None', marker='o')
    label4_half = (r'$\frac{1}{30}\overline{u}_{-2} - '
                   r'\frac{13}{60}\overline{u}_{-1} + '
                   r'\frac{47}{60}\overline{u}_{0} + '
                   r'\frac{9}{20}\overline{u}_{1} -'
                   r'\frac{1}{20}\overline{u}_{2}$')
    bottom_right.text(-2.5, -3, label4_half, fontsize=fontsize)
    if stopping_point == 11:
        return
    # Do a little bit more: add all the other dots.
    bottom_right.plot([0.5], [y_val1], color=colors[0],
                      linestyle='None', marker='o')
    bottom_right.plot([0.5], [y_val2], color=colors[1],
                      linestyle='None', marker='o')
    bottom_right.plot([0.5], [y_val3], color=colors[2],
                      linestyle='None', marker='o')
# pylint: enable=too-many-locals,too-many-return-statements
# pylint: enable=too-many-statements


def discontinuity_to_volume():
    """Make plots similar to introductory, but with a discontinuity."""
    colors = seaborn.color_palette('husl')[:2]
    fontsize = 16
    num_pts = 100
    rows, cols = 1, 2
    fig, (ax1, ax2) = plt.subplots(rows, cols, sharex=True, sharey=True)
    fig.set_size_inches(15, 8)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.25, 2.25)

    # Set u(x).
    ax1.fill_between(np.linspace(-2.5, 0, num_pts), 2 * np.ones(num_pts),
                     color=colors[0], alpha=0.5)
    ax1.fill_between(np.linspace(0, 2.5, num_pts), np.ones(num_pts),
                     color=colors[0], alpha=0.5)
    ax1.set_title('$u(x)$', fontsize=fontsize)

    # Set cell average u(x).
    for begin, val in zip([-2.5, -1.5, -0.5, 0.5, 1.5],
                          [2, 2, 1.5, 1, 1]):
        ax2.plot(np.linspace(begin, begin + 1, num_pts),
                 val * np.ones(num_pts),
                 color=colors[1], linestyle='dashed')
        ax2.plot([begin, begin + 1], [val, val],
                 color=colors[1], marker='o', linestyle='None')
    ax2.set_title(r'$\overline{u}(x)$', fontsize=fontsize)


# pylint: disable=too-many-locals
def make_shock_plot():
    """Make plots similar to introductory, but with a discontinuity."""
    colors = seaborn.color_palette('husl')[:4]
    fontsize = 20
    num_pts = 100
    rows = cols = 2
    fig, ax_vals = plt.subplots(rows, cols, sharex=True, sharey=True)
    fig.set_size_inches(15, 8)
    ax_vals[0, 0].set_xlim(-3, 3)
    ax_vals[0, 0].set_ylim(-0.5, 2.5)

    # Top left plot (-2, -1, and 0)
    top_left = ax_vals[0, 0]
    top_left.plot(np.linspace(-2.5, -1.5, num_pts),
                  2 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    top_left.plot(np.linspace(-1.5, -0.5, num_pts),
                  2 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    top_left.plot(np.linspace(-0.5, 0.5, num_pts),
                  1.5 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    x_vals = np.linspace(-2.5, 0.5, num_pts)
    y_vals = -x_vals**2 / 4 - 3 * x_vals / 4 + 73.0 / 48
    top_left.fill_between(x_vals, 0, y_vals,
                          color=colors[0], alpha=0.5)
    label1 = r'$\frac{-12x^2 - 36x + 73}{48}$'
    top_left.text(1, 1.5, label1, fontsize=fontsize)

    # Top right plot (-1, 0, and 1)
    top_right = ax_vals[0, 1]
    top_right.plot(np.linspace(-1.5, -0.5, num_pts),
                   2 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    top_right.plot(np.linspace(-0.5, 0.5, num_pts),
                   1.5 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    top_right.plot(np.linspace(0.5, 1.5, num_pts),
                   1 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    x_vals = np.linspace(-1.5, 1.5, num_pts)
    y_vals = -x_vals / 2 + 1.5
    top_right.fill_between(x_vals, 0, y_vals,
                           color=colors[1], alpha=0.5)
    label2 = r'$\frac{-x + 3}{2}$'
    top_right.text(1, 1.5, label2, fontsize=fontsize)

    # Bottom left plot (0, 1, 2)
    bottom_left = ax_vals[1, 0]
    bottom_left.plot(np.linspace(-0.5, 0.5, num_pts),
                     1.5 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    bottom_left.plot(np.linspace(0.5, 1.5, num_pts),
                     1 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    bottom_left.plot(np.linspace(1.5, 2.5, num_pts),
                     1 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    x_vals = np.linspace(-0.5, 2.5, num_pts)
    y_vals = x_vals**2 / 4 - 3 * x_vals / 4 + 71.0 / 48
    bottom_left.fill_between(x_vals, y_vals,
                             color=colors[2], alpha=0.5)
    label3 = r'$\frac{12x^2 - 36x + 71}{48}$'
    bottom_left.text(1, 1.5, label3, fontsize=fontsize)

    # Bottom right plot (-2, -1, 0, 1 and 2)
    bottom_right = ax_vals[1, 1]
    bottom_right.plot(np.linspace(-2.5, -1.5, num_pts),
                      2 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(-1.5, -0.5, num_pts),
                      2 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(-0.5, 0.5, num_pts),
                      1.5 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(0.5, 1.5, num_pts),
                      1 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(1.5, 2.5, num_pts),
                      1 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    x_vals = np.linspace(-2.5, 2.5, num_pts)
    y_vals = x_vals**3 / 12 - 29 * x_vals / 48 + 1.5
    bottom_right.fill_between(x_vals, y_vals,
                              color=colors[3], alpha=0.5)
    label4 = r'$\frac{4x^3 - 29x + 72}{48}$'
    bottom_right.text(1, 1.5, label4, fontsize=fontsize)
# pylint: enable=too-many-locals


# pylint: disable=invalid-name
def discontinuity_to_volume_single_cell(stopping_point=None):
    """Plot a piecewise constant function w/discontinuity towards the left.

    :type stopping_point: int
    :param stopping_point: (Optional) The transition point to stop at when
                           creating the plot. By passing in 0, 1, 2, ...
                           this allows us to create a short slide-show.
    """
    colors = seaborn.color_palette('husl')[:2]
    fontsize = 16
    num_pts = 100
    rows, cols = 1, 2
    fig, (ax1, ax2) = plt.subplots(rows, cols, sharex=True, sharey=True)
    fig.set_size_inches(15, 8)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.25, 2.25)

    # Set u(x).
    ax1.fill_between(np.linspace(-2.5, -2, num_pts), 2 * np.ones(num_pts),
                     color=colors[0], alpha=0.5)
    ax1.fill_between(np.linspace(-2, 2.5, num_pts), np.ones(num_pts),
                     color=colors[0], alpha=0.5)
    ax1.set_title('$u(x)$', fontsize=fontsize)
    if stopping_point == 0:
        return

    # Set cell average u(x).
    for begin, val in zip([-2.5, -1.5, -0.5, 0.5, 1.5],
                          [1.5, 1, 1, 1, 1]):
        ax2.plot(np.linspace(begin, begin + 1, num_pts),
                 val * np.ones(num_pts),
                 color=colors[1], linestyle='dashed')
        ax2.plot([begin, begin + 1], [val, val],
                 color=colors[1], marker='o', linestyle='None')
    ax2.set_title(r'$\overline{u}(x)$', fontsize=fontsize)
# pylint: enable=invalid-name


# pylint: disable=too-many-locals
def make_shock_plot_single_cell():
    """Plot the reconstructed polynomials that occur near a shock."""
    colors = seaborn.color_palette('husl')[:4]
    fontsize = 20
    num_pts = 100
    rows = cols = 2
    fig, ax_vals = plt.subplots(rows, cols, sharex=True, sharey=True)
    fig.set_size_inches(15, 8)
    ax_vals[0, 0].set_xlim(-3, 3)
    ax_vals[0, 0].set_ylim(-0.5, 2.5)

    # Top left plot (-2, -1, and 0)
    top_left = ax_vals[0, 0]
    top_left.plot(np.linspace(-2.5, -1.5, num_pts),
                  1.5 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    top_left.plot(np.linspace(-1.5, -0.5, num_pts),
                  1 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    top_left.plot(np.linspace(-0.5, 0.5, num_pts),
                  1 * np.ones(num_pts),
                  color='black', linestyle='dashed')
    x_vals = np.linspace(-2.5, 0.5, num_pts)
    y_vals = x_vals**2 / 4 + x_vals / 4 + 47.0 / 48
    top_left.fill_between(x_vals, 0, y_vals,
                          color=colors[0], alpha=0.5)
    label1 = r'$\frac{12x^2 + 12x + 47}{48}$'
    top_left.text(1, 1.5, label1, fontsize=fontsize)

    # Top right plot (-1, 0, and 1)
    top_right = ax_vals[0, 1]
    top_right.plot(np.linspace(-1.5, -0.5, num_pts),
                   1 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    top_right.plot(np.linspace(-0.5, 0.5, num_pts),
                   1 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    top_right.plot(np.linspace(0.5, 1.5, num_pts),
                   1 * np.ones(num_pts),
                   color='black', linestyle='dashed')
    x_vals = np.linspace(-1.5, 1.5, num_pts)
    y_vals = np.ones(num_pts)
    top_right.fill_between(x_vals, 0, y_vals,
                           color=colors[1], alpha=0.5)
    label2 = r'$1$'
    top_right.text(1, 1.5, label2, fontsize=fontsize)

    # Bottom left plot (0, 1, 2)
    bottom_left = ax_vals[1, 0]
    bottom_left.plot(np.linspace(-0.5, 0.5, num_pts),
                     1 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    bottom_left.plot(np.linspace(0.5, 1.5, num_pts),
                     1 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    bottom_left.plot(np.linspace(1.5, 2.5, num_pts),
                     1 * np.ones(num_pts),
                     color='black', linestyle='dashed')
    x_vals = np.linspace(-0.5, 2.5, num_pts)
    y_vals = np.ones(num_pts)
    bottom_left.fill_between(x_vals, y_vals,
                             color=colors[2], alpha=0.5)
    label3 = r'$1$'
    bottom_left.text(1, 1.5, label3, fontsize=fontsize)

    # Bottom right plot (-2, -1, 0, 1 and 2)
    bottom_right = ax_vals[1, 1]
    bottom_right.plot(np.linspace(-2.5, -1.5, num_pts),
                      1.5 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(-1.5, -0.5, num_pts),
                      1 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(-0.5, 0.5, num_pts),
                      1 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(0.5, 1.5, num_pts),
                      1 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    bottom_right.plot(np.linspace(1.5, 2.5, num_pts),
                      1 * np.ones(num_pts),
                      color='black', linestyle='dashed')
    x_vals = np.linspace(-2.5, 2.5, num_pts)
    y_vals = (x_vals**4 / 48 - x_vals**3 / 24 - x_vals**2 / 32 +
              5 * x_vals / 96 + 1283.0 / 1280)
    bottom_right.fill_between(x_vals, y_vals,
                              color=colors[3], alpha=0.5)
    label4 = r'$\frac{80x^4 - 160x^3 - 120x^2 + 200x + 3849}{3840}$'
    bottom_right.text(-0.75, 1.5, label4, fontsize=fontsize)
# pylint: enable=too-many-locals
