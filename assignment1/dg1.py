"""Simple dg1 script from Persson."""


import matplotlib.pyplot as plt
import numpy as np


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


def dg1(n, p_order, T, dt):
    """Basic dg1 solver.

    A Discontinuous Galerkin (DG) solver to solve the 1D conservation law

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

    :rtype: tuple
    :returns: Pair of positional and keyword arguments to
              :class:`matplotlib.animation.FuncAnimation`. The
              positional arguments contain a :class:`matplotlib.figure.Figure`
              pre-populated with the initial data.
    :raises: :class:`ValueError <exceptions.ValueError>` if ``p_order``
             is not ``1`` or ``2``.
    """
    h = 1.0 / n
    nsteps = int(np.round(T / dt))

    if p_order == 1:
        Mel = h * MASS_MAT_P1
        Kel = STIFFNESS_MAT_P1
        x = get_node_points(n, p_order, h=h)
    elif p_order == 2:
        Mel = h * MASS_MAT_P2
        Kel = STIFFNESS_MAT_P2
        x = get_node_points(n, p_order, h=h)
    else:
        raise ValueError('Error: p_order not implemented', p_order)

    u = np.exp(-(x - 0.5)**2 / 0.01)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-.1, 1.1))
    ax.plot(x, u, 'r', lw=2)
    plt.grid(True)
    plot_lines = ax.plot(x, u, 'b', lw=2)

    animation_args = (fig, animate)
    animation_kwargs = {
        'frames': nsteps + 1,
        'interval': 20,
        'blit': True,
        'fargs': [u, plot_lines, Kel, Mel, dt],
    }
    return animation_args, animation_kwargs


def animate(frame_number, u, plot_lines, Kel, Mel, dt):
    """Update solution and use new solution to update plotted data.

    Uses the pre-computed mass matrix (``Mel``) and stiffness matrix (``Kel``)
    from :func:`dg1` and a modified RK4 to update the solution. This is
    done by using the following to compute :math:`\\dot{u}` via

    .. math::

       M \\dot{\\mathbf{u}}^k = K \\mathbf{u}^k + u_p^{k - 1} e_0 - u_p^k e_{p}

    and using it as a black box for RK4 via :math:`\\dot{u} = f(t, u)`.

    :type frame_number: int
    :param frame_number: (Unused) The current frame.

    :type u: :class:`numpy.ndarray`
    :param u: The current solution, with ``p_order + 1`` rows and ``n``
              columns. The columns correspond to each interval and the
              rows correspond to the nodes of the polynomial within
              each interval.

    :type plot_lines: :class:`list` of :class:`matplotlib.lines.Line2D`
    :param plot_lines: List of ``matplotlib`` line objects which need to
                       be updated with new solution values.

    :type Kel: :class:`numpy.ndarray`
    :param Kel: The stiffness matrix, with ``p_order + 1`` rows and columns.

    :type Mel: :class:`numpy.ndarray`
    :param Mel: The mass matrix, scaled by the interval width
                :math:`1 / n`, with ``p_order + 1`` rows and columns.

    :type dt: float
    :param dt: The timestep to use in the solver.

    :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
    :returns: List of the updated ``matplotlib`` line objects.
    """
    u_orig = u
    u0 = u
    for irk in _RK_STEPS:
        # u = [u0^0, u0^1, ..., u0^{n-1}]
        #     [u1^0, u1^1, ..., u1^{n-1}]
        #     [...                   ...]
        #     [up^0, up^1, ..., up^{n-1}]
        r = np.dot(Kel, u)
        # At this point, the columns of ``r`` are
        #    r = [K u^0, K u^1, ..., K u^{n-1}]
        # and we seek to have each column contain
        #    K u^k + up^{k-1} e0 - up^k ep

        # So the first thing we do is modify
        #    K u^k --> K u^k - up^k ep
        # so we just take the final row of ``u``
        #     [up^0, up^1, ..., up^{n-1}]
        # and subtract it from the last component of ``r``.
        r[-1, :] -= u[-1, :]
        # Then we modify
        #    K u^k - up^k ep --> K u^k - up^k ep + up^{k-1} e0
        # with the assumption that up^{-1} = up^{n-1}, i.e. we
        # assume the solution is periodic, so we just roll
        # the final row of ``u`` around to
        #     [up^1, ..., up^{n-1}, up^0]
        # and add it to the first component of ``r``.
        r[0, :] += np.roll(u[-1, :], shift=1)
        # Here, we solve M u' = K u^k + up^{k-1} e0 - up^k ep
        # in each column of ``r`` and then use the ``u'``
        # estimates to update the value.
        u = u0 + dt / irk * np.linalg.solve(Mel, r)
    for index, line in enumerate(plot_lines):
        line.set_ydata(u[:, index])
    u_orig[:] = u  # Update the original data.
    return plot_lines


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

        with the periodic assumption
        :math:`\\mathbf{u}^{-1} = \\mathbf{u}^{n-1}`. Once we can find
        :math:`\\dot{\\mathbf{u}}^k` we can use RK4 to compute the
        updated value

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
            u_curr = u_prev + dt / irk * np.linalg.solve(
                self.mass_mat, r)
        self.u = u_curr
        self.current_step += 1
