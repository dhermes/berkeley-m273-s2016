"""Module for solving a 1D conservation law via DG.

Adapted from a Discontinuous Galerkin (DG) solver written
by Per Olof-Persson.

Check out an example `notebook`_ using these utilities to
solve the problem.

.. _notebook: http://nbviewer.jupyter.org/github/dhermes/\
              berkeley-m273-s2016/blob/master/assignment1/dg1.ipynb
"""


import numpy as np
from numpy.polynomial import legendre
import six
import sympy


_RK_STEPS = (4, 3, 2, 1)


def get_symbolic_vandermonde(p_order):
    """Get symbolic Vandermonde matrix of evenly spaced points.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :rtype: tuple
    :returns: Pair of vector of powers of :math:`x` and Vandermonde matrix.
              Both are type
              :class:`sympy.Matrix <sympy.matrices.dense.MutableDenseMatrix>`,
              the ``x_vec`` is a row vector with ``p_order + 1`` columns and
              the Vandermonde matrix is square of dimension ``p_order + 1``.
    """
    x_symb = sympy.Symbol('x')
    x_vals = sympy.Matrix(six.moves.xrange(p_order + 1)) / p_order
    vand_mat = sympy.zeros(p_order + 1, p_order + 1)
    x_vec = sympy.zeros(1, p_order + 1)
    for i in six.moves.xrange(p_order + 1):
        x_vec[i] = x_symb**i
        for j in six.moves.xrange(p_order + 1):
            vand_mat[i, j] = x_vals[i]**j
    return x_vec, vand_mat


def find_matrices_symbolic(p_order):
    """Find mass and stiffness matrices using symbolic algebra.

    We do this on the reference interval :math:`\\left[0, 1\\right]`
    with the evenly spaced points

    .. math::

       x_0 = 0, x_1 = 1/p, \\ldots, x_p = 1

    and compute the polynomials :math:`\\varphi_j(x)` such that
    :math:`\\varphi_j\\left(x_i\\right) = \\delta_{ij}`. Since we
    are using symbolic rationals numbers, we do this directly by
    inverting the Vandermonde matrix :math:`V` such that

        .. math::

           \\left[ \\begin{array}{c c c c}
                    1 & x_0 & \\cdots & x_0^p \\\\
                    1 & x_1 & \\cdots & x_1^p \\\\
                    \\vdots & & & \\vdots \\\\
                    1 & x_p & \\cdots & x_p^p
                    \\end{array}\\right]
           \\left[ \\begin{array}{c} c_0 \\\\
                                     c_1 \\\\
                                   \\vdots \\\\
                                     c_p \\end{array}\\right]
           = \\left(\\delta_{ij}\\right) = I_{p + 1}

    Then use these to compute the mass matrix

    .. math::

        M_{ij} = \\int_0^1 \\varphi_i(x) \\varphi_j(x) \\, dx

    and the stiffness matrix

    .. math::

        K_{ij} = \\int_0^1 \\varphi_i'(x) \\varphi_j(x) \\, dx

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, square
              :class:`sympy.Matrix <sympy.matrices.dense.MutableDenseMatrix>`
              with rows/columns equal to ``p_order + 1``.
    """
    x_symb = sympy.Symbol('x')
    x_vec, vand_mat = get_symbolic_vandermonde(p_order)
    coeff_mat = vand_mat**(-1)
    phi_funcs = x_vec * coeff_mat

    mass_mat = sympy.zeros(p_order + 1, p_order + 1)
    stiffness_mat = sympy.zeros(p_order + 1, p_order + 1)
    for i in six.moves.xrange(p_order + 1):
        phi_i = phi_funcs[i]
        phi_i_prime = sympy.diff(phi_i, x_symb)
        for j in six.moves.xrange(i, p_order + 1):
            phi_j = phi_funcs[j]
            integral_m = sympy.integrate(phi_i * phi_j, x_symb)
            integral_k = sympy.integrate(phi_i_prime * phi_j, x_symb)
            mass_mat[i, j] = (integral_m.subs({x_symb: 1}) -
                              integral_m.subs({x_symb: 0}))
            stiffness_mat[i, j] = (integral_k.subs({x_symb: 1}) -
                                   integral_k.subs({x_symb: 0}))
            if j > i:
                mass_mat[j, i] = mass_mat[i, j]
                stiffness_mat[j, i] = -stiffness_mat[i, j]

    return mass_mat, stiffness_mat


def mass_and_stiffness_matrices_p1():
    """Get mass and stiffness matrices for :math:`p = 1`.

    These are for the interval :math:`\\left[0, 1\\right]`.

    .. math::

       M = \\frac{1}{6} \\left[ \\begin{array}{c c}
                          2 & 1 \\\\
                          1 & 2
                        \\end{array}\\right], \\qquad
       K = \\frac{1}{2} \\left[ \\begin{array}{c c}
                          -1 & -1 \\\\
                           1 &  1
                        \\end{array}\\right]

    These values can be verified by :func:`find_matrices_symbolic`.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, :math:`2 \\times 2`
              :class:`numpy.ndarray`.
    """
    mass_mat = np.array([
        [2, 1],
        [1, 2],
    ]) / 6.0
    stiffness_mat = np.array([
        [-1, -1],
        [1, 1],
    ]) / 2.0
    return mass_mat, stiffness_mat


def mass_and_stiffness_matrices_p2():
    """Get mass and stiffness matrices for :math:`p = 2`.

    These are for the interval :math:`\\left[0, 1\\right]`.

    .. math::

       M = \\frac{1}{30} \\left[ \\begin{array}{c c c}
                            4 &  2 & -1 \\\\
                            2 & 16 &  2 \\\\
                           -1 &  2 &  4
                         \\end{array}\\right], \\qquad
       K = \\frac{1}{6} \\left[ \\begin{array}{c c c}
                          -3 & -4 &  1 \\\\
                           4 &  0 & -4 \\\\
                          -1 &  4 &  3
                        \\end{array}\\right]

    These values can be verified by :func:`find_matrices_symbolic`.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, :math:`3 \\times 3`
              :class:`numpy.ndarray`.
    """
    mass_mat = np.array([
        [4, 2, -1],
        [2, 16, 2],
        [-1, 2, 4],
    ]) / 30.0
    stiffness_mat = np.array([
        [-3, -4, 1],
        [4, 0, -4],
        [-1, 4, 3],
    ]) / 6.0
    return mass_mat, stiffness_mat


def mass_and_stiffness_matrices_p3():
    """Get mass and stiffness matrices for :math:`p = 3`.

    These are for the interval :math:`\\left[0, 1\\right]`.

    .. math::

       M = \\frac{1}{1680} \\left[ \\begin{array}{c c c c}
                              128 &   99 &  -36 &   19 \\\\
                               99 &  648 &  -81 &  -36 \\\\
                              -36 &  -81 &  648 &   99 \\\\
                               19 &  -36 &   99 &  128
                           \\end{array}\\right], \\qquad
       K = \\frac{1}{80} \\left[ \\begin{array}{c c c c}
                            -40 &  -57 &   24 &   -7 \\\\
                             57 &    0 &  -81 &   24 \\\\
                            -24 &   81 &    0 &  -57 \\\\
                              7 &  -24 &   57 &   40
                         \\end{array}\\right]

    These values can be verified by :func:`find_matrices_symbolic`.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, :math:`4 \\times 4`
              :class:`numpy.ndarray`.
    """
    mass_mat = np.array([
        [128, 99, -36, 19],
        [99, 648, -81, -36],
        [-36, -81, 648, 99],
        [19, -36, 99, 128],
    ]) / 1680.0
    stiffness_mat = np.array([
        [-40, -57, 24, -7],
        [57, 0, -81, 24],
        [-24, 81, 0, -57],
        [7, -24, 57, 40],
    ]) / 80.0
    return mass_mat, stiffness_mat


def gauss_lobatto_points(num_points):
    """Get the node points for Gauss-Lobatto quadrature.

    Using :math:`n` points, this quadrature is accurate to degree
    :math:`2n - 3`. The node points are :math:`x_1 = -1`,
    :math:`x_n = 1` and the interior are :math:`n - 2` roots of
    :math:`P'_{n - 1}(x)`.

    Though we don't compute them here, the weights are
    :math:`w_1 = w_n = \\frac{2}{n(n - 1)}` and for the interior points

    .. math::

       w_j = \\frac{2}{n(n - 1) \\left[P_{n - 1}\\left(x_j\\right)\\right]^2}

    This is in contrast to the scheme used in Gaussian quadrature, which
    use roots of :math:`P_n(x)` as nodes and use the weights

    .. math::

       w_j = \\frac{2}{\\left(1 - x_j\\right)^2
                \\left[P'_n\\left(x_j\\right)\\right]^2}

    :type num_points: int
    :param num_points: The number of points to use.

    :rtype: :class:`numpy.ndarray`
    :returns: 1D array, the interior quadrature nodes.
    """
    p_n_minus1 = [0] * (num_points - 1) + [1]
    inner_nodes = legendre.legroots(legendre.legder(p_n_minus1))
    # Utilize symmetry about 0.
    inner_nodes = 0.5 * (inner_nodes - inner_nodes[::-1])
    return inner_nodes


def get_legendre_matrix(points, max_degree=None):
    """Evaluate Legendre polynomials at a set of points.

    If our points are :math:`x_0, \\ldots, x_p`, this computes

    .. math::

       \\left[ \\begin{array}{c c c c}
         L_0(x_0) & L_1(x_0) & \\cdots & L_d(x_0) \\\\
         L_0(x_1) & L_1(x_1) & \\cdots & L_d(x_p) \\\\
         \\vdots  & \\vdots  & \\ddots & \\vdots  \\\\
         L_0(x_p) & L_1(x_p) & \\cdots & L_d(x_p)
       \\end{array}\\right]

    by utilizing the recurrence

    .. math::

        n L_n(x) = (2n - 1) x L_{n - 1}(x) - (n - 1) L_{n - 2}(x)

    :type points: :class:`numpy.ndarray`
    :param points: 1D array. The points at which to evaluate Legendre
                   polynomials.

    :type max_degree: int
    :param max_degree: (Optional) The maximum degree of Legendre
                       polynomial to use. Defaults to one less than the
                       number of points (which will produce a square
                       output).

    :rtype: :class:`numpy.ndarray`
    :returns: The 2D array containing the Legendre polynomials evaluated
              at our input points.
    """
    num_points, = points.shape
    if max_degree is None:
        max_degree = num_points - 1
    # Use Fortran order since we operator on columns.
    result = np.zeros((num_points, max_degree + 1), order='F')
    result[:, 0] = 1.0
    result[:, 1] = points
    for degree in six.moves.xrange(2, max_degree + 1):
        result[:, degree] = (
            (2 * degree - 1) * points * result[:, degree - 1] -
            (degree - 1) * result[:, degree - 2]) / degree
    return result


def _find_matrices_helper(vals1, vals2):
    """Helper for :func:`find_matrices`.

    Computes a shoelace-like product of two vectors :math:`u, v`
    via

    .. math::

        u_0 (v_1 + v_3 + \\cdots) + u_1 (v_2 + v_4 + \\cdots) +
            u_{p - 1} v_p

    :type vals1: :class:`numpy.ndarray`.
    :param vals1: The vector :math`u`.

    :type vals2: :class:`numpy.ndarray`.
    :param vals2: The vector :math`v`.

    :rtype: float
    :returns: The shoelace-like product of the vectors.
    """
    result = 0
    for i, val in enumerate(vals1):
        result += val * np.sum(vals2[i + 1::2])
    return result


def find_matrices(p_order):
    """Find mass and stiffness matrices.

    We do this on the reference interval :math:`\\left[-1, 1\\right]`
    with the evenly spaced points

    .. math::

       x_0 = -1, x_1 = -(p - 2)/p, \\ldots, x_p = 1

    and compute the polynomials :math:`\\varphi_j(x)` such that
    :math:`\\varphi_j\\left(x_i\\right) = \\delta_{ij}`. We do this by
    writing

    .. math::

       \\varphi_j(x) = \\sum_{n = 0}^p c_n^{(j)} L_n(x)

    where :math:`L_n(x)` is the Legendre polynomial of degree :math:`n`.
    With this representation, we need to solve

    .. math::

       \\left[ \\begin{array}{c c c c}
         L_0(x_0) & L_1(x_0) & \\cdots & L_p(x_0) \\\\
         L_0(x_1) & L_1(x_1) & \\cdots & L_p(x_p) \\\\
         \\vdots  & \\vdots  & \\ddots & \\vdots  \\\\
         L_0(x_p) & L_1(x_p) & \\cdots & L_p(x_p)
       \\end{array}\\right]
       \\left[ \\begin{array}{c c c c}
         c_0^{(0)} & c_0^{(1)} & \\cdots & c_0^{(p)} \\\\
         c_1^{(0)} & c_1^{(1)} & \\cdots & c_1^{(p)} \\\\
           \\vdots &   \\vdots & \\ddots & \\vdots   \\\\
         c_p^{(0)} & c_p^{(1)} & \\cdots & c_p^{(p)}
       \\end{array}\\right]
       = \\left(\\delta_{ij}\\right) = I_{p + 1}

    Then use these to compute the mass matrix

    .. math::

        M_{ij} = \\int_{-1}^1 \\varphi_i(x) \\varphi_j(x) \\, dx

    and the stiffness matrix

    .. math::

        K_{ij} = \\int_{-1}^1 \\varphi_i'(x) \\varphi_j(x) \\, dx

    Utilizing the fact that

    .. math::

        \\left\\langle L_n, L_m \\right\\rangle =
            \\int_{-1}^1 L_n(x) L_m(x) \\, dx =
            \\frac{2}{2n + 1} \\delta_{nm}

    we can compute

    .. math::

        M_{ij} = \\left\\langle \\varphi_i, \\varphi_j \\right\\rangle =
            \\sum_{n, m} \\left\\langle c_n^{(i)} L_n,
                c_m^{(j)} L_m \\right\\rangle =
            \\sum_{n = 0}^p \\frac{2}{2n + 1} c_n^{(i)} c_n^{(j)}.

    Similarly

    .. math::

        \\left\\langle L_n'(x), L_m(x) \\right\\rangle =
          \\begin{cases}
            2 & \\text{ if } n > m \\text{ and }
                n - m \\equiv 1 \\bmod{2} \\\\
            0 & \\text{ otherwise}.
          \\end{cases}

    gives

    .. math::

        \\begin{align*}
        K_{ij} &= \\left\\langle \\varphi_i', \\varphi_j \\right\\rangle
                = \\sum_{n, m} \\left\\langle c_n^{(i)} L_n',
                      c_m^{(j)} L_m \\right\\rangle \\\\
               &= 2 \\left(c_0^{(j)} \\left(c_1^{(i)} + c_3^{(i)} +
                      \\cdots\\right)
                         + c_1^{(j)} \\left(c_2^{(i)} + c_4^{(i)} +
                      \\cdots\\right)
                         + \\cdots
                         + c_{p - 1}^{(j)} c_p^{(i)}\\right) \\\\
        \\end{align*}

    (For more general integrals, one might use Gaussian quadrature.
    The largest degree integrand :math:`\\varphi_i \\varphi_j` has
    degree :math:`2 p` so this would require :math:`n = p + 1` points
    to be exact up to degree :math:`2(p + 1) - 1 = 2p + 1`.)

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, square
              :class:`numpy.ndarray` of dimension ``p_order + 1``.
    """
    # Find the coefficients of the L_n(x) for each basis function.
    x_vals = np.linspace(-1, 1, p_order + 1)
    coeff_mat = np.linalg.inv(get_legendre_matrix(x_vals))

    # Populate the mass and stiffness matrices.
    legendre_norms = 2.0 / np.arange(1, 2 * p_order + 2, 2)
    mass_mat = np.zeros((p_order + 1, p_order + 1))
    stiffness_mat = np.zeros((p_order + 1, p_order + 1))
    for i in six.moves.xrange(p_order + 1):
        mass_mat_base = legendre_norms * coeff_mat[:, i]
        for j in six.moves.xrange(i, p_order + 1):
            mass_mat[i, j] = np.sum(mass_mat_base * coeff_mat[:, j])
            stiffness_mat[i, j] = 2.0 * _find_matrices_helper(
                coeff_mat[:, j], coeff_mat[:, i])
            if j > i:
                mass_mat[j, i] = mass_mat[i, j]
                stiffness_mat[j, i] = -stiffness_mat[i, j]

    return mass_mat, stiffness_mat


def low_storage_rk(ode_func, u_val, dt):
    """Update an ODE solutuon with an order 2/4 Runge-Kutta function.

    The method is given by the following Butcher array:

    .. math::

           \\begin{array}{c | c c c c}
             0 &   0 &     &     &   \\\\
           1/4 & 1/4 &   0 &     &   \\\\
           1/3 &   0 & 1/3 &   0 &   \\\\
           1/2 &   0 &   0 & 1/2 & 0 \\\\
           \\hline
               &   0 &   0 &   0 & 1
           \\end{array}

    It is advantageous because the updates :math:`k_j` can be over-written at
    each step, since they are never re-used.

    One can see that this method is **order 2** for general
    :math:`\\dot{u} = f(u)` by verifying that not all order 3 node
    conditions are satisfied. For example:

    .. math::

       \\frac{1}{3} \\neq \\sum_i b_i c_i^2 = 0 + 0 + 0 +
       1 \\cdot \\left(\\frac{1}{2}\\right)^2

    However, for linear ODEs, the method is **order 4**. To see this, note
    that the test problem :math:`\\dot{u} = \\lambda u` gives the stability
    function

    .. math::

        R\\left(\\lambda \\Delta t\\right) = R(z) =
        1 + z + \\frac{z^2}{2} + \\frac{z^3}{6} + \\frac{z^4}{24}

    which matches the Taylor series for :math:`e^z` to order 4.

    See `Problem Set 3`_ from Persson's Math 228A for more details.

    .. _Problem Set 3: http://persson.berkeley.edu/228A/ps3.pdf

    :type ode_func: callable
    :param ode_func: The RHS in the ODE :math:`\\dot{u} = f(u)`.

    :type u_val: :class:`numpy.ndarray`
    :param u_val: The input to :math:`f(u)`.

    :type dt: float
    :param dt: The timestep to use.

    :rtype: :class:`numpy.ndarray`
    :returns: The updated solution value.
    """
    u_prev = u_val
    # NOTE: We may not actually need to copy this if the ``ode_func`` does
    #       not mutate ``u_val`` in place, but we copy anyhow to avoid any
    #       accidental mutation.
    u_curr = u_val.copy()
    for irk in _RK_STEPS:
        u_curr = u_prev + dt / irk * ode_func(u_curr)
    return u_curr


def get_node_points(num_points, p_order, step_size=None):
    """Return node points to splitting unit interval for DG.

    :type num_points: int
    :param num_points: The number :math:`n` of intervals to divide
                       :math:`\\left[0, 1\\right]` into.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type step_size: float
    :param step_size: (Optional) The step size :math:`1 / n`.

    :rtype: :class:`numpy.ndarray`
    :returns: The :math:`x`-values for the node points, with
              ``p_order + 1`` rows and :math:`n` columns. The columns
              correspond to each sub-interval and the rows correspond
              to the node points within each sub-interval.
    """
    if step_size is None:
        step_size = 1.0 / num_points
    interval_starts = np.linspace(0, 1 - step_size, num_points)
    # Split the first interval [0, h] in ``p_order + 1`` points
    first_interval = np.linspace(0, step_size, p_order + 1)
    # Broadcast the values with ``first_interval`` as rows and
    # columns as ``interval_starts``.
    return (first_interval[:, np.newaxis] +
            interval_starts[np.newaxis, :])


def get_gaussian_like_initial_data(node_points):
    """Get the default initial solution data.

    In this case it is

    .. math::

       u(x, 0) = \\exp\\left(-\\left(\\frac{x - \\frac{1}{2}}{0.1}
                             \\right)^2\\right)

    :type node_points: :class:`numpy.ndarray`
    :param node_points: Points at which evaluate the initial data function.

    :rtype: :class:`numpy.ndarray`
    :returns: The :math:`u`-values at each node point.
    """
    return np.exp(-(node_points - 0.5)**2 / 0.01)


class DG1Solver(object):
    """Discontinuous Galerkin (DG) solver for the 1D conservation law

    .. math::

       \\frac{\\partial u}{\\partial t} + \\frac{\\partial u}{\\partial x} = 0

    on the interval :math:`\\left[0, 1\\right]`. By default, uses
    Gaussian-like initial data

    .. math::

       u(x, 0) = \\exp\\left(-\\left(\\frac{x - \\frac{1}{2}}{0.1}
                             \\right)^2\\right)

    but :math:`u(x, 0)` can be specified via ``get_initial_data``.

    We represent our solution via the :math:`(p + 1) \\times n` rectangular
    matrix:

        .. math::

           \\mathbf{u} = \\left[ \\begin{array}{c c c c}
                 u_0^1 &   u_0^2 & \\cdots &   u_0^n \\\\
                 u_1^1 &   u_1^2 & \\cdots &   u_1^n \\\\
               \\vdots & \\vdots & \\ddots & \\vdots \\\\
                 u_p^1 &   u_p^2 & \\cdots &   u_p^n
               \\end{array}\\right]

    where each column represents one of :math:`n` sub-intervals and each row
    represents one of the :math:`p + 1` node points within each sub-interval.

    :type num_intervals: int
    :param num_intervals: The number :math:`n` of intervals to divide
                          :math:`\\left[0, 1\\right]` into.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type total_time: float
    :param total_time: The amount of time to run the solver for (starts at
                       :math:`t = 0`).

    :type dt: float
    :param dt: The timestep to use in the solver.

    :type get_initial_data: callable
    :param get_initial_data: (Optional) The function to use to evaluate
                             :math:`u(x, 0)` at the points in our solution.
                             Defaults to
                             :func:`get_gaussian_like_initial_data`.
    """

    def __init__(self, num_intervals, p_order, total_time, dt,
                 get_initial_data=None):
        self.num_intervals = num_intervals
        self.p_order = p_order
        self.total_time = total_time
        self.dt = dt
        self.current_step = 0
        # Computed values.
        self.num_steps = int(np.round(self.total_time / self.dt))
        self.step_size = 1.0 / self.num_intervals
        self.node_points = get_node_points(self.num_intervals, self.p_order,
                                           step_size=self.step_size)
        # The solution: u(x, t).
        if get_initial_data is None:
            get_initial_data = get_gaussian_like_initial_data
        self.solution = get_initial_data(self.node_points)
        (self.mass_mat,
         self.stiffness_mat) = self.get_mass_and_stiffness_matrices()

    def get_mass_and_stiffness_matrices(self):
        """Get the mass and stiffness matrices for the current solver.

        Uses pre-computed mass matrix and stiffness matrix for :math:`p = 1`,
        :math:`p = 2` and :math:`p = 3` degree polynomials and computes
        the matrices on the fly for larger :math:`p`.

        Depends on the sub-interval width ``h`` and the order of accuracy
        ``p_order``.

        :rtype: tuple
        :returns: Pair of mass and stiffness matrices, both with
                  ``p_order + 1`` rows and columns.
        """
        if self.p_order == 1:
            mass_mat, stiffness_mat = mass_and_stiffness_matrices_p1()
        elif self.p_order == 2:
            mass_mat, stiffness_mat = mass_and_stiffness_matrices_p2()
        elif self.p_order == 3:
            mass_mat, stiffness_mat = mass_and_stiffness_matrices_p3()
        else:
            mass_mat, stiffness_mat = find_matrices(self.p_order)
            # We are solving on [0, 1] but ``find_matrices`` is
            # on [-1, 1], and the mass matrix is translation invariant
            # but scales with interval length.
            mass_mat = 0.5 * mass_mat

        return self.step_size * mass_mat, stiffness_mat

    def ode_func(self, u_val):
        """Compute the right-hand side for the ODE.

        When we write

        .. math::

           M \\dot{\\mathbf{u}} = K \\mathbf{u} +
           \\left[ \\begin{array}{c c c c c}
                      u_p^2 &   u_p^3 & \\cdots &      u_p^n & u_p^1   \\\\
                          0 &       0 & \\cdots &          0 & 0       \\\\
                    \\vdots & \\vdots & \\ddots &    \\vdots & \\vdots \\\\
                          0 &       0 & \\cdots &          0 & 0       \\\\
                     -u_p^1 &  -u_p^2 & \\cdots & -u_p^{n-1} & -u_p^n  \\\\
                    \\end{array}\\right]

        we specify a RHS :math:`f(u)` via solving the system.

        :type u_val: :class:`numpy.ndarray`
        :param u_val: The input to :math:`f(u)`.

        :rtype: :class:`numpy.ndarray`
        :returns: The value of the slope function evaluated at ``u_val``.
        """
        rhs = np.dot(self.stiffness_mat, u_val)

        # First we modify
        #    K u^k --> K u^k - up^k ep
        # so we just take the final row of ``u``
        #     [up^0, up^1, ..., up^{n-1}]
        # and subtract it from the last component of ``r``.
        rhs[-1, :] -= u_val[-1, :]
        # Then we modify
        #    K u^k - up^k ep --> K u^k - up^k ep + up^{k-1} e0
        # with the assumption that up^{-1} = up^{n-1}, i.e. we
        # assume the solution is periodic, so we just roll
        # the final row of ``u`` around to
        #     [up^1, ..., up^{n-1}, up^0]
        # and add it to the first component of ``r``.
        rhs[0, :] += np.roll(u_val[-1, :], shift=1)
        return np.linalg.solve(self.mass_mat, rhs)

    def update(self):
        """Update the solution for a single time step.

        We use :meth:`ode_func` to compute :math:`\\dot{u} = f(u)` and
        pair it with an RK method (:func:`low_storage_rk`) to compute
        the updated value.
        """
        self.solution = low_storage_rk(self.ode_func, self.solution, self.dt)
        self.current_step += 1
