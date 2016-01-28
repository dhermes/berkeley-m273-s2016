"""Module for solving a 1D conservation law via DG.

Adapted from a Discontinuous Galerkin (DG) solver written
by Per Olof-Persson.
"""


import matplotlib.pyplot as plt
import numpy as np
import six


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

_RK_STEPS = (4, 3, 2, 1)

INTERVAL_POINTS = 10
"""Number of points to use when plotting a polynomial on an interval."""


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

    Uses pre-computed mass matrix (``Mel``) and stiffness matrix (``Kel``)
    for :math:`p = 1` and :math:`p = 2` degree polynomials.

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

    :raises: :class:`ValueError <exceptions.ValueError>` if ``p_order``
             is not ``1`` or ``2``.
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
        self.mass_mat = self._get_mass_matrix()
        self.stiffness_mat = self._get_stiffness_matrix()

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

    def _get_mass_matrix(self):
        """Get the mass matrix for the current solver.

        Depends on the sub-interval width ``h`` and the order of accuracy
        ``p_order``. Comes from the pre-computed constant matrices in
        the current module.

        :rtype: :class:`numpy.ndarray`
        :returns: The mass matrix for the solver, with ``p_order + 1``
                  rows and columns.
        :raises: :class:`ValueError <exceptions.ValueError>` if
                 ``p_order`` is not ``1`` or ``2``.
        """
        if self.p_order == 1:
            return self.h * MASS_MAT_P1
        elif self.p_order == 2:
            return self.h * MASS_MAT_P2
        else:
            raise ValueError('Error: p_order not implemented', self.p_order)

    def _get_stiffness_matrix(self):
        """Get the stiffness matrix for the current solver.

        Depends on the order of accuracy ``p_order``. Comes from the
        pre-computed constant matrices in the current module.

        :rtype: :class:`numpy.ndarray`
        :returns: The stiffness matrix for the solver, with ``p_order + 1``
                  rows and columns.
        :raises: :class:`ValueError <exceptions.ValueError>` if
                 ``p_order`` is not ``1`` or ``2``.
        """
        if self.p_order == 1:
            return STIFFNESS_MAT_P1
        elif self.p_order == 2:
            return STIFFNESS_MAT_P2
        else:
            raise ValueError('Error: p_order not implemented', self.p_order)

    def update(self):
        """Update the solution for a single time step.

        For each sub-interval :math:`k` of :math:`\\left[0, 1\\right]`, we
        use the current solution

        .. math::

           \\mathbf{u}^k = \\left[ \\begin{array}{c} u_0^k \\\\
                                   u_1^k \\\\
                                   \\vdots \\\\
                                   u_p^k \\end{array}\\right]

        and update it via

        .. math::

           M \\dot{\\mathbf{u}}^k = K \\mathbf{u}^k + u_p^{k - 1} e_0 -
                                    u_p^k e_{p}

        with the periodic assumption :math:`u_p^{0-1} = u_p^{n-1}`.
        Once we can find :math:`\\dot{\\mathbf{u}}^k` we can use RK4
        to compute the updated value

        .. math::

           \\mathbf{u}^k + \\frac{\\Delta t}{6} \\left(\\text{slope}_1 +
                       2 \\text{slope}_2 + 2 \\text{slope}_3 +
                       \\text{slope}_4\\right).
        """
        u_prev = self.u
        # NOTE: We don't actually need to copy this since the values of
        #       ``u_curr`` aren't mutated in place, but we copy anyhow to
        #       avoid any accidental mutation.
        u_curr = self.u.copy()
        for irk in _RK_STEPS:
            # u = [u0^0, u0^1, ..., u0^{n-1}]
            #     [u1^0, u1^1, ..., u1^{n-1}]
            #     [...                   ...]
            #     [up^0, up^1, ..., up^{n-1}]
            r = np.dot(self.stiffness_mat, u_curr)
            # At this point, the columns of ``r`` are
            #    r = [K u^0, K u^1, ..., K u^{n-1}]
            # and we seek to have each column contain
            #    K u^k + up^{k-1} e0 - up^k ep

            # So the first thing we do is modify
            #    K u^k --> K u^k - up^k ep
            # so we just take the final row of ``u``
            #     [up^0, up^1, ..., up^{n-1}]
            # and subtract it from the last component of ``r``.
            r[-1, :] -= u_curr[-1, :]
            # Then we modify
            #    K u^k - up^k ep --> K u^k - up^k ep + up^{k-1} e0
            # with the assumption that up^{-1} = up^{n-1}, i.e. we
            # assume the solution is periodic, so we just roll
            # the final row of ``u`` around to
            #     [up^1, ..., up^{n-1}, up^0]
            # and add it to the first component of ``r``.
            r[0, :] += np.roll(u_curr[-1, :], shift=1)
            # Here, we solve M u' = K u^k + up^{k-1} e0 - up^k ep
            # in each column of ``r`` and then use the ``u'``
            # estimates to update the value.
            u_curr = u_prev + self.dt / irk * np.linalg.solve(
                self.mass_mat, r)
        self.u = u_curr
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
        self.poly_interp_funcs = self._get_interpolation_funcs()
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

    def _get_interpolation_funcs(self):
        """Get polynomial interpolation objects for each interval.

        :rtype: list
        :returns: List of :class:`PolynomialInterpolate` objects, one
                  for each sub-interval.
        """
        _, num_cols = self.solver.x.shape
        interp_funcs = []
        for i in six.moves.xrange(num_cols):
            x_vals = self.solver.x[:, i]
            interp_funcs.append(PolynomialInterpolate(x_vals))
        return interp_funcs

    def _plot_solution(self, color):
        """Plot the solution and return the newly created lines.

        :type color: str
        :param color: The color to use in plotting the solution.

        :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
        :returns: List of the updated ``matplotlib`` line objects.
        """
        _, num_cols = self.solver.x.shape
        plot_lines = []
        for index in six.moves.xrange(num_cols):
            interp_func = self.poly_interp_funcs[index]
            y_vals = self.solver.u[:, index]
            all_y = interp_func.interpolate(y_vals)
            line, = self.ax.plot(interp_func.all_x, all_y,
                                 color=color, linewidth=2)
            plot_lines.append(line)

        return plot_lines

    def init_func(self):
        """An initialization function for the animation.

        Plots the initial data **and** stores the lines created.

        :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
        :returns: List of the updated ``matplotlib`` line objects,
                  with length equal to ``n`` (coming from ``solver``).
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
        :returns: List of the updated ``matplotlib`` line objects,
                  with length equal to ``n`` (coming from ``solver``).
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
        for index, line in enumerate(self.plot_lines):
            interp_func = self.poly_interp_funcs[index]
            y_vals = self.solver.u[:, index]
            all_y = interp_func.interpolate(y_vals)
            line.set_ydata(all_y)
        return self.plot_lines
