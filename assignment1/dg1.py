"""Module for solving a 1D conservation law via DG.

Adapted from a Discontinuous Galerkin (DG) solver written
by Per Olof-Persson.

Check out an example `notebook`_ using these utilities to
solve the problem.

.. _notebook: http://nbviewer.jupyter.org/github/dhermes/berkeley-m273-s2016/blob/master/assignment1/dg1.ipynb
"""


import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre
from numpy.polynomial import polynomial
import six
import sympy


MASS_MAT_P1 = np.array([
    [2, 1],
    [1, 2],
]) / 6.0
"""Pre-computed mass matrix for :math:`p = 1`."""

STIFFNESS_MAT_P1 = 0.5 * np.array([
    [-1, -1],
    [ 1,  1],
])
"""Pre-computed stiffness matrix for :math:`p = 1`."""

MASS_MAT_P2 = np.array([
    [ 4,  2, -1],
    [ 2, 16,  2],
    [-1,  2,  4],
]) / 30.0
"""Pre-computed mass matrix for :math:`p = 2`."""

STIFFNESS_MAT_P2 = np.array([
    [-3, -4,  1],
    [ 4,  0, -4],
    [-1,  4,  3],
]) / 6.0
"""Pre-computed stiffness matrix for :math:`p = 2`."""

MASS_MAT_P3 = np.array([
    [128,  99, -36,  19],
    [ 99, 648, -81, -36],
    [-36, -81, 648,  99],
    [ 19, -36,  99, 128],
]) / 1680.0
"""Pre-computed mass matrix for :math:`p = 3`."""

STIFFNESS_MAT_P3 = np.array([
    [-40, -57,  24,  -7],
    [ 57,   0, -81,  24],
    [-24,  81,   0, -57],
    [  7, -24,  57,  40],
]) / 80.0
"""Pre-computed stiffness matrix for :math:`p = 3`."""

_RK_STEPS = (4, 3, 2, 1)

INTERVAL_POINTS = 10
"""Number of points to use when plotting a polynomial on an interval."""


def find_matrices_symbolic(p_order):
    """Find mass and stiffness matrices using symbolic algebra.

    We do this on the reference interval [0, 1] with the points

    .. math::

       x_0 = 0, x_1 = \\frac{1}{p}, \\ldots, x_p = 1

    and compute the polynomials :math:`\\varphi_j(x)` such that
    :math:`\\varphi_j\\left(x_i\\right) = \\delta_{ij}`. Since we
    are using rationals, we do this directly by inverting the
    Vandermonde matrix.

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

    Then uses these to compute the mass matrix

    .. math::

        M_{ij} = \\int_0^1 \\varphi_i(x) \\varphi_j(x) \\, dx

    and the stiffness matrix

    .. math::

        K_{ij} = \\int_0^1 \\varphi_i'(x) \\varphi_j(x) \\, dx

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrix, square
              :class:`sympy.Matrix <sympy.matrices.dense.MutableDenseMatrix>`
              with rows/columns equal to ``p_order + 1``.
    """
    x_symb = sympy.Symbol('x')
    x_vals = sympy.Matrix(six.moves.xrange(p_order + 1)) / p_order
    V = sympy.zeros(p_order + 1, p_order + 1)
    x_vec = sympy.zeros(1, p_order + 1)
    for i in six.moves.xrange(p_order + 1):
        x_vec[i] = x_symb**i
        for j in six.moves.xrange(p_order + 1):
            V[i, j] = x_vals[i]**j
    coeff_mat = V**(-1)
    phi_funcs = x_vec * coeff_mat

    M = sympy.zeros(p_order + 1, p_order + 1)
    K = sympy.zeros(p_order + 1, p_order + 1)
    for i in six.moves.xrange(p_order + 1):
        phi_i = phi_funcs[i]
        phi_i_prime = sympy.diff(phi_i, x_symb)
        for j in six.moves.xrange(i, p_order + 1):
            phi_j = phi_funcs[j]
            I_M = sympy.integrate(phi_i * phi_j, x_symb)
            I_K = sympy.integrate(phi_i_prime * phi_j, x_symb)
            M[i, j] = I_M.subs({x_symb: 1}) - I_M.subs({x_symb: 0})
            K[i, j] = I_K.subs({x_symb: 1}) - I_K.subs({x_symb: 0})
            if j > i:
                M[j, i] = M[i, j]
                K[j, i] = -K[i, j]

    return M, K


def find_matrices(p_order):
    """Find mass and stiffness matrices.

    We do this on the reference interval [0, 1] with the points

    .. math::

       x_0 = 0, x_1 = \\frac{1}{p}, \\ldots, x_p = 1

    and compute the polynomials :math:`\\varphi_j(x)` such that
    :math:`\\varphi_j\\left(x_i\\right) = \\delta_{ij}`. Since we
    are using rationals, we do this directly by inverting the
    Vandermonde matrix.

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

    Then uses these to compute the mass matrix

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
    :returns: Pair of mass and stiffness matrix, square :class:`numpy.ndarray`
              with rows/columns equal to ``p_order + 1``.
    """
    num_leg = p_order + 1
    leg_pts, leg_weights = legendre.leggauss(num_leg)
    # Shift points and weights from [-1, 1] to [0, 1]
    leg_pts += 1
    leg_pts *= 0.5
    leg_weights *= 0.5

    # Now create the Vandermonde matrix and invert.
    x_vals = np.arange(p_order + 1, dtype=np.float64) / p_order
    V = np.zeros((p_order + 1, p_order + 1))
    for i in six.moves.xrange(p_order + 1):
        for j in six.moves.xrange(p_order + 1):
            V[i, j] = x_vals[i]**j
    coeff_mat = np.linalg.inv(V)

    # Evaluate the PHI_i at the Legendre points.
    phi_vals = polynomial.polyval(leg_pts, coeff_mat)
    # The rows correspond to the polynomials while the
    # columns correspond to the values in ``leg_pts``.

    # Populate the mass and stiffness matrices.
    M = np.zeros((p_order + 1, p_order + 1))
    K = np.zeros((p_order + 1, p_order + 1))
    for i in six.moves.xrange(p_order + 1):
        phi_i = phi_vals[i, :]
        phi_i_prime = polynomial.polyval(
            leg_pts, polynomial.polyder(coeff_mat[:, i]))
        for j in six.moves.xrange(i, p_order + 1):
            phi_j = phi_vals[j, :]
            M[i, j] = (phi_i * phi_j).dot(leg_weights)
            K[i, j] = (phi_i_prime * phi_j).dot(leg_weights)
            if j > i:
                M[j, i] = M[i, j]
                K[j, i] = -K[i, j]

    return M, K


def low_storage_rk(ode_func, u_val, dt):
    """Update an ODE solutuon with an order 2/4 Runge-Kutta function.

    The method is based on the following Butcher array:

    .. math::

           \\begin{array}{c | c c c c}
             0 &   0 &     &     &   \\\\
           1/4 & 1/4 &   0 &     &   \\\\
           1/3 &   0 & 1/3 &   0 &   \\\\
           1/2 &   0 &   0 & 1/2 & 0 \\\\
           \\hline
               &   0 &   0 &   0 & 1
           \\end{array}

    It is advantageous because the update can be over-written at each
    step, since updates are never re-used.

    One can see that this method is order ``2`` for general
    :math:`\\dot{u} = f(u)` by verifying that not all order 3 node
    conditions are satisfied

    .. math::

       \\frac{1}{3} \\neq \\sum_i b_i c_i^2 = 0 + 0 + 0 +
       1 \\cdot \\left(\\frac{1}{2}\\right)^2

    However, for linear ODEs, the method is order ``4``. To see this, note
    that the test problem :math:`\\dot{u} = \\lambda u` gives the stability
    function

    .. math::

        R\\left(\\lambda \\Delta t\\right) = R(z) =
        1 + z + \\frac{z^2}{2} + \\frac{z^3}{3} + \\frac{z^4}{4}

    which matches the Taylor series for :math:`e^z` to order ``4``.

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


def get_node_points(n, p_order, h=None):
    """Return node points to splitting unit interval for DG.

    :type n: int
    :param n: The number of intervals to divide :math:`\\left[0, 1\\right]`
              into.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type h: float
    :param h: (Optional) The step size ``1 / n``.

    :rtype: :class:`numpy.ndarray`
    :returns: The ``x``-values for the node points, with
              ``p_order + 1`` rows and ``n`` columns. The columns
              correspond to each sub-interval and the rows correspond
              to the node points within each sub-interval.
    """
    if h is None:
        h = 1.0 / n
    interval_starts = np.linspace(0, 1 - h, n)
    # Split the first interval [0, h] in ``p_order + 1`` points
    first_interval = np.linspace(0, h, p_order + 1)
    # Broadcast the values with ``first_interval`` as rows and
    # columns as ``interval_starts``.
    return (first_interval[:, np.newaxis] +
            interval_starts[np.newaxis, :])


class PolynomialInterpolate(object):
    """Polynomial interpolation from node points.

    Assumes the first and last ``x``-value are the endpoints of
    the interval.

    Using Lagrange basis polynomials we can write our polynomial as

    .. math::

       p(x) = \\sum_{j} y_j \\ell_j(x)

    and we can compute :math:`\\ell_j(x)`` of our data without ever computing
    the coefficients. We do this by computing all pairwise differences of
    our ``x``-values and the interpolating values. Then we take the products
    of these differences (leaving out one of the interpolating values).

    :type x_vals: :class:`numpy.ndarray`
    :param x_vals: List of ``x``-values that uniquely define a
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
        """Make matrix of Lagrange interp. polys evaluated on interval.

        :rtype: :class:`numpy.ndarray`
        :returns: The matrix :math:`\\ell_j(x)` where the rows correspond to
                  the ``num_points`` possible ``x``-values and the columns
                  correspond to the ``p_order + 1`` possible ``j``-values.
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
        """Evaluate interpolated polynomial given ``y``-values.

        We've already pre-computed the values :math:`\\ell_j(x)` for
        all the ``x``-values we use in our interval (``num_points`` in
        all, using the interpolating ``x``-values to compute the
        :math:`\\ell_j(x)`). So we simply use them to compute

        .. math::

           p(x) = \\sum_{j} y_j \\ell_j(x)

        using the :math:`y_j` from ``y_vals``.

        :type y_vals: :class:`numpy.ndarray`
        :param y_vals: 1D array of ``y``-values that uniquely define
                       our interpolating polynomial.

        :rtype: :class:`numpy.ndarray`
        :returns: 1D array containing :math:`\\p(x)` for each ``x``-value in
                  the interval (``num_points`` in all).
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

    Uses pre-computed mass matrix and stiffness matrix for :math:`p = 1`,
    :math:`p = 2` and :math:`p = 3` degree polynomials and computes
    the matrices on the fly for larger :math:`p`.

    We represent our solution via the :math:`(p + 1) \\times n` rectangular
    matrix:

        .. math::

           \\mathbf{u} = \\left[ \\begin{array}{c c c c}
                      u_0^1 & u_0^2 & \\cdots & u_0^n \\\\
                      u_1^1 & u_1^2 & \\cdots & u_1^n \\\\
                    \\vdots & & \\ddots & \\vdots \\\\
                      u_p^1 & u_p^2 & \\cdots & u_p^n \\\\
                    \\end{array}\\right]

    where each column represents one of :math:`n` sub-intervals and each row
    represents one of the :math:`p + 1` node points within each sub-interval.

    :type n: int
    :param n: The number of intervals to divide :math:`\\left[0, 1\\right]`
              into.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type T: float
    :param T: The amount of time to run the solver for (starts at
              :math:`t = 0`.

    :type dt: float
    :param dt: The timestep to use in the solver.
    """

    def __init__(self, n, p_order, T, dt):
        self.n = n
        self.p_order = p_order
        self.T = T
        self.dt = dt
        self.current_step = 0
        # Computed values.
        self.num_steps = int(np.round(self.T / self.dt))
        self.h = 1.0 / self.n
        self.x = get_node_points(self.n, self.p_order, h=self.h)
        self.u = self._get_initial_data()
        M, K = self._get_mass_and_stiffness_matrices()
        self.mass_mat = M
        self.stiffness_mat = K

    def _get_initial_data(self):
        """Get the initial solution data.

        In this case it is

        .. math::

           u(x, 0) = \\exp\\left(-\\left(\\frac{x - \\frac{1}{2}}{0.1}
                                 \\right)^2\\right)

        :rtype: :class:`numpy.ndarray`
        :returns: The ``u``-values at each point in ``x``.
        """
        return np.exp(-(self.x - 0.5)**2 / 0.01)

    def _get_mass_and_stiffness_matrices(self):
        """Get the mass and stiffness matrices for the current solver.

        Depends on the sub-interval width ``h`` and the order of accuracy
        ``p_order``. Comes from the pre-computed constant matrices in
        the current module.

        :rtype: tuple
        :returns: Pair of mass and stiffness matric, both with ``p_order + 1``
                  rows and columns.
        """
        if self.p_order == 1:
            return self.h * MASS_MAT_P1, STIFFNESS_MAT_P1
        elif self.p_order == 2:
            return self.h * MASS_MAT_P2, STIFFNESS_MAT_P2
        elif self.p_order == 3:
            return self.h * MASS_MAT_P3, STIFFNESS_MAT_P3
        else:
            M, K = find_matrices(p_order)
            return self.h * M, K

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
        r = np.dot(self.stiffness_mat, u_val)

        # First we modify
        #    K u^k --> K u^k - up^k ep
        # so we just take the final row of ``u``
        #     [up^0, up^1, ..., up^{n-1}]
        # and subtract it from the last component of ``r``.
        r[-1, :] -= u_val[-1, :]
        # Then we modify
        #    K u^k - up^k ep --> K u^k - up^k ep + up^{k-1} e0
        # with the assumption that up^{-1} = up^{n-1}, i.e. we
        # assume the solution is periodic, so we just roll
        # the final row of ``u`` around to
        #     [up^1, ..., up^{n-1}, up^0]
        # and add it to the first component of ``r``.
        r[0, :] += np.roll(u_val[-1, :], shift=1)
        return np.linalg.solve(self.mass_mat, r)

    def update(self):
        """Update the solution for a single time step.

        We use :meth:`ode_func` to compute :math:`\\dot{u} = f(u)` and
        pair it with an RK method (:func:`low_storage_rk`) to compute
        the updated value.
        """
        self.u = low_storage_rk(self.ode_func, self.u, self.dt)
        self.current_step += 1


class DG1Animate(object):
    """Helper for animating a solution.

    Assumes the ``solver`` operates on :math:`x \\in \\left[0, 1\\right]`
    and the solution :math:`u(x, t) \\in \\left[0, 1\\right]` (give
    or take some noise).

    :type solver: :class:`DG1Solver`
    :param solver: The solver which computes and updates the solution.
    """

    def __init__(self, solver):
        self.solver = solver
        # Computed values.
        self.poly_interp_func = self._get_interpolation_func()
        self.plot_lines = None  # Will be updated in ``init_func``.
        self.fig, self.ax = plt.subplots(1, 1)
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
        x_vals = self.solver.x[:, 0]
        return PolynomialInterpolate(x_vals)

    def _plot_solution(self, color):
        """Plot the solution and return the newly created lines.

        :type color: str
        :param color: The color to use in plotting the solution.

        :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
        :returns: List of the updated matplotlib line objects.
        """
        _, num_cols = self.solver.x.shape
        plot_lines = []
        interp_func = self.poly_interp_func
        all_y = interp_func.interpolate(self.solver.u)
        for index in six.moves.xrange(num_cols):
            x_start = self.solver.x[0, index],
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
        all_y = self.poly_interp_func.interpolate(self.solver.u)
        for index, line in enumerate(self.plot_lines):
            line.set_ydata(all_y[:, index])
        return self.plot_lines
