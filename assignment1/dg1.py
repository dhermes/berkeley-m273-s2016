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
from numpy.polynomial import polynomial
import six
import sympy


_RK_STEPS = (4, 3, 2, 1)

INTERVAL_POINTS = 10
"""Number of points to use when plotting a polynomial on an interval."""


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


def find_matrices(p_order):
    """Find mass and stiffness matrices.

    We do this on the reference interval :math:`\\left[0, 1\\right]`
    with the evenly spaced points

    .. math::

       x_0 = 0, x_1 = 1/p, \\ldots, x_p = 1

    and compute the polynomials :math:`\\varphi_j(x)` such that
    :math:`\\varphi_j\\left(x_i\\right) = \\delta_{ij}`. We do this directly
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

    This method uses Gaussian quadrature to evaluate the integrals.
    The largest degree integrand :math:`\\varphi_i \\varphi_j` has
    degree :math:`2 p` so we use :math:`n = p + 1` points to ensure
    that the quadrature is exact.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, square
              :class:`numpy.ndarray` of dimension ``p_order + 1``.
    """
    num_leg = p_order + 1
    leg_pts, leg_weights = legendre.leggauss(num_leg)
    # Shift points and weights from [-1, 1] to [0, 1]
    leg_pts += 1
    leg_pts *= 0.5
    leg_weights *= 0.5

    # Now create the Vandermonde matrix and invert.
    x_vals = np.arange(p_order + 1, dtype=np.float64) / p_order
    vand_mat = np.zeros((p_order + 1, p_order + 1))
    for i in six.moves.xrange(p_order + 1):
        for j in six.moves.xrange(p_order + 1):
            vand_mat[i, j] = x_vals[i]**j
    coeff_mat = np.linalg.inv(vand_mat)

    # Evaluate the PHI_i at the Legendre points.
    phi_vals = polynomial.polyval(leg_pts, coeff_mat)
    # The rows correspond to the polynomials while the
    # columns correspond to the values in ``leg_pts``.

    # Populate the mass and stiffness matrices.
    mass_mat = np.zeros((p_order + 1, p_order + 1))
    stiffness_mat = np.zeros((p_order + 1, p_order + 1))
    for i in six.moves.xrange(p_order + 1):
        phi_i = phi_vals[i, :]
        phi_i_prime = polynomial.polyval(
            leg_pts, polynomial.polyder(coeff_mat[:, i]))
        for j in six.moves.xrange(i, p_order + 1):
            phi_j = phi_vals[j, :]
            mass_mat[i, j] = (phi_i * phi_j).dot(leg_weights)
            stiffness_mat[i, j] = (phi_i_prime * phi_j).dot(leg_weights)
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


class PolynomialInterpolate(object):
    """Polynomial interpolation from node points.

    Assumes the first and last :math:`x`-value are the endpoints of
    the interval.

    Using Lagrange basis polynomials we can write our polynomial as

    .. math::

       p(x) = \\sum_{j} y_j \\ell_j(x)

    and we can compute :math:`\\ell_j(x)` of our data without ever computing
    the coefficients. We do this by computing all pairwise differences of
    our :math:`x`-values and the interpolating values. Then we take the
    products of these differences (leaving out one of the interpolating
    values).

    :type x_vals: :class:`numpy.ndarray`
    :param x_vals: List of :math:`x`-values that uniquely define a
                   polynomial. The degree is one less than the number
                   of points.

    :type num_points: int
    :param num_points: The number of points to use to represent
                       the polynomial.
    """

    def __init__(self, x_vals, num_points=INTERVAL_POINTS):
        self.x_vals = x_vals
        # Computed values.
        min_x = x_vals[0]
        max_x = x_vals[-1]
        self.all_x = np.linspace(min_x, max_x, num_points)
        self.lagrange_matrix = self.make_lagrange_matrix()

    def make_lagrange_matrix(self):
        """Make matrix where :math:`M_{ij} = \\ell_j(x_i)`.

        This matrix contains the Lagrange interpolating polynomials evaluated
        on the interval given by ``x_vals``. The :math:`x_i` (corresponding to
        rows in :math:`M`) are the ``num_points`` possible :math:`x`-values in
        ``all_x`` and the :math:`\\ell_j` (corresponding to columns in
        :math:`M`) are the Lagrange interpolating polynomials interpolated
        on the points in ``x_vals``.

        :rtype: :class:`numpy.ndarray`
        :returns: The matrix :math:`M`.
        """
        # First compute the denominators of the Lagrange polynomials.
        pairwise_diff = self.x_vals[:, np.newaxis] - self.x_vals[np.newaxis, :]
        # Put 1's on the diagonal (instead of zeros) before taking product.
        np.fill_diagonal(pairwise_diff, 1.0)
        lagrange_denoms = np.prod(pairwise_diff, axis=1)  # Row products.

        num_x = self.x_vals.size
        # Now compute the differences of our x-values for plotting
        # and the x-values used to interpolate.
        new_x_diff = self.all_x[:, np.newaxis] - self.x_vals[np.newaxis, :]
        result = np.zeros((self.all_x.size, num_x))

        for index in six.moves.xrange(num_x):
            curr_slice = np.hstack([new_x_diff[:, :index],
                                    new_x_diff[:, index + 1:]])
            result[:, index] = (np.prod(curr_slice, axis=1) /
                                lagrange_denoms[index])

        return result

    def interpolate(self, y_vals):
        """Evaluate interpolated polynomial given :math:`y`-values.

        We've already pre-computed the values :math:`\\ell_j(x)` for
        all the :math:`x`-values we use in our interval (``num_points`` in
        all, using the interpolating :math:`x`-values to compute the
        :math:`\\ell_j(x)`). So we simply use them to compute

        .. math::

           p(x) = \\sum_{j} y_j \\ell_j(x)

        using the :math:`y_j` from ``y_vals``.

        :type y_vals: :class:`numpy.ndarray`
        :param y_vals: 1D array of :math:`y`-values that uniquely define
                       our interpolating polynomial.

        :rtype: :class:`numpy.ndarray`
        :returns: 1D array containing :math:`p(x)` for each :math:`x`-value
                  in the interval (``num_points`` in all).
        """
        if len(y_vals.shape) == 1:
            # Make into a column vector before applying matrix.
            y_vals = y_vals[:, np.newaxis]
        return self.lagrange_matrix.dot(y_vals)


class DG1Solver(object):
    """Discontinuous Galerkin (DG) solver for the 1D conservation law

    .. math::

       \\frac{\\partial u}{\\partial t} + \\frac{\\partial u}{\\partial x} = 0

    on the interval :math:`\\left[0, 1\\right]` with Gaussian-like
    initial data

    .. math::

       u(x, 0) = \\exp\\left(-\\left(\\frac{x - \\frac{1}{2}}{0.1}
                             \\right)^2\\right)

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
                       :math:`t = 0`.

    :type dt: float
    :param dt: The timestep to use in the solver.
    """

    def __init__(self, num_intervals, p_order, total_time, dt):
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
        self.solution = self._get_initial_data()
        (self.mass_mat,
         self.stiffness_mat) = self.get_mass_and_stiffness_matrices()

    def _get_initial_data(self):
        """Get the initial solution data.

        In this case it is

        .. math::

           u(x, 0) = \\exp\\left(-\\left(\\frac{x - \\frac{1}{2}}{0.1}
                                 \\right)^2\\right)

        :rtype: :class:`numpy.ndarray`
        :returns: The :math:`u`-values at each node point.
        """
        return np.exp(-(self.node_points - 0.5)**2 / 0.01)

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


class DG1Animate(object):
    """Helper for animating a solution.

    Assumes the ``solver`` operates on :math:`x \\in \\left[0, 1\\right]`
    and the solution :math:`u(x, t) \\in \\left[0, 1\\right]` (give
    or take some noise).

    :type solver: :class:`DG1Solver`
    :param solver: The solver which computes and updates the solution.

    :type fig: :class:`matplotlib.figure.Figure`
    :param fig: (Optional) A figure to use for plotting. Intended to be passed
                when creating a :class:`matplotlib.animation.FuncAnimation`.

    :type ax: :class:`matplotlib.artist.Artist`
    :param ax: (Optional) An axis to be used for plotting.

    :raises: :class:`ValueError <exceptions.ValueError>` if one of ``fig``
             or ``ax`` is passed, but not both.
    """

    def __init__(self, solver, fig=None, ax=None):
        self.solver = solver
        # Computed values.
        self.poly_interp_func = self._get_interpolation_func()
        self.plot_lines = None  # Will be updated in ``init_func``.
        self.fig = fig
        self.ax = ax
        # Give defaults for fig and ax if not set.
        if self.fig is None:
            if self.ax is not None:
                raise ValueError('Received an axis but no figure.')
            import matplotlib.pyplot as plt
            self.fig, self.ax = plt.subplots(1, 1)
        # At this point both fig and ax should be set, but if fig
        # was not None, then it's possible ax **was** None.
        if self.ax is None:
            raise ValueError('Received a figure but no axis.')
        self._configure_axis()
        # Plot the initial data (in red) to compare against.
        self._plot_solution('red')

    def _configure_axis(self):
        """Configure the axis for plotting."""
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0 - 0.1, 1 + 0.1)
        self.ax.grid(b=True)  # b == boolean, 'on'/'off'

    def _get_interpolation_func(self):
        """Get polynomial interpolation object for reference interval.

        :rtype: :class:`PolynomialInterpolate`
        :returns: Interpolation object for the reference
        """
        # Reference ``x``-values are in the first column.
        x_vals = self.solver.node_points[:, 0]
        return PolynomialInterpolate(x_vals)

    def _plot_solution(self, color):
        """Plot the solution and return the newly created lines.

        :type color: str
        :param color: The color to use in plotting the solution.

        :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
        :returns: List of the updated matplotlib line objects.
        """
        _, num_cols = self.solver.node_points.shape
        plot_lines = []
        interp_func = self.poly_interp_func
        all_y = interp_func.interpolate(self.solver.solution)
        for index in six.moves.xrange(num_cols):
            x_start = self.solver.node_points[0, index],
            line, = self.ax.plot(x_start + interp_func.all_x,
                                 all_y[:, index],
                                 color=color, linewidth=2)
            plot_lines.append(line)

        return plot_lines

    def init_func(self):
        """An initialization function for the animation.

        Plots the initial data **and** stores the lines created.

        :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
        :returns: List of the updated matplotlib line objects,
                  with length equal to :math:`n` (coming from ``solver``).
        """
        # Plot the same data (in blue) and store the lines.
        self.plot_lines = self._plot_solution('blue')
        # For ``init_func`` with ``blit`` turned on, the initial
        # frame should not have visible lines. See
        # http://stackoverflow.com/q/21439489/1068170 for more info.
        for line in self.plot_lines:
            line.set_visible(False)
        return self.plot_lines

    def update_plot(self, frame_number):
        """Updates the lines in the plot.

        First advances the solver and then uses the updated value
        to update the :class:`matplotlib.lines.Line2D` objects
        associated to each interval.

        :type frame_number: int
        :param frame_number: (Unused) The current frame.

        :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
        :returns: List of the updated matplotlib line objects,
                  with length equal to :math:`n` (coming from ``solver``).
        :raises: :class:`ValueError <exceptions.ValueError>` if the
                 frame number doesn't make the current step on the
                 solver.
        """
        if frame_number == 0:
            # ``init_func`` creates lines that are not visible, to
            # address http://stackoverflow.com/q/21439489/1068170.
            # So in the initial frame, we make them visible.
            for line in self.plot_lines:
                line.set_visible(True)
        if self.solver.current_step != frame_number:
            raise ValueError('Solver current step does not match '
                             'frame number', self.solver.current_step,
                             frame_number)
        self.solver.update()
        all_y = self.poly_interp_func.interpolate(self.solver.solution)
        for index, line in enumerate(self.plot_lines):
            line.set_ydata(all_y[:, index])
        return self.plot_lines
