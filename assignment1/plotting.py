"""Plotting helpers for :mod:`dg1` solver."""


import numpy as np
import six


INTERVAL_POINTS = 10
"""Number of points to use when plotting a polynomial on an interval."""


def make_lagrange_matrix(x_vals, all_x):
    r"""Make matrix where :math:`M_{ij} = \ell_j(x_i)`.

    This matrix contains the Lagrange interpolating polynomials evaluated
    on the interval given by ``x_vals``. The :math:`x_i` (corresponding to
    rows in :math:`M`) are the ``num_points`` possible :math:`x`-values in
    ``all_x`` and the :math:`\ell_j` (corresponding to columns in
    :math:`M`) are the Lagrange interpolating polynomials interpolated
    on the points in ``x_vals``.

    :type x_vals: :class:`numpy.ndarray`
    :param x_vals: 1D array of :math:`x`-values used to interpolate data via
                   Lagrange basis functions.

    :type all_x: :class:`numpy.ndarray`
    :param all_x: 1D array of points to evaluate the :math:`\ell_j(x)`` at.

    :rtype: :class:`numpy.ndarray`
    :returns: The matrix :math:`M`.
    """
    # First compute the denominators of the Lagrange polynomials.
    pairwise_diff = x_vals[:, np.newaxis] - x_vals[np.newaxis, :]
    # Put 1's on the diagonal (instead of zeros) before taking product.
    np.fill_diagonal(pairwise_diff, 1.0)
    lagrange_denoms = np.prod(pairwise_diff, axis=1)  # Row products.

    num_x = x_vals.size
    # Now compute the differences of our x-values for plotting
    # and the x-values used to interpolate.
    new_x_diff = all_x[:, np.newaxis] - x_vals[np.newaxis, :]
    result = np.zeros((all_x.size, num_x))

    for index in six.moves.xrange(num_x):
        curr_slice = np.hstack([new_x_diff[:, :index],
                                new_x_diff[:, index + 1:]])
        result[:, index] = (np.prod(curr_slice, axis=1) /
                            lagrange_denoms[index])

    return result


class PolynomialInterpolate(object):
    r"""Polynomial interpolation from node points.

    Assumes the first and last :math:`x`-value are the endpoints of
    the interval.

    Using Lagrange basis polynomials we can write our polynomial as

    .. math::

       p(x) = \sum_{j} y_j \ell_j(x)

    and we can compute :math:`\ell_j(x)` of our data without ever computing
    the coefficients. We do this by computing all pairwise differences of
    our :math:`x`-values and the interpolating values. Then we take the
    products of these differences (leaving out one of the interpolating
    values).

    :type x_vals: :class:`numpy.ndarray`
    :param x_vals: List of :math:`x`-values that uniquely define a
                   polynomial. The degree is one less than the number
                   of points.

    :type num_points: int
    :param num_points: (Optional) The number of points to use to represent
                       the polynomial. Defaults to :data:`INTERVAL_POINTS`.
    """

    def __init__(self, x_vals, num_points=None):
        self.x_vals = x_vals
        # Computed values.
        min_x = x_vals[0]
        max_x = x_vals[-1]
        if num_points is None:
            num_points = INTERVAL_POINTS
        self.all_x = np.linspace(min_x, max_x, num_points)
        self.lagrange_matrix = make_lagrange_matrix(self.x_vals, self.all_x)

    @classmethod
    def from_solver(cls, solver, num_points=None):
        """Polynomial interpolation factory from a solver.

        The reference interval for the interpolation is assumed
        to be in the first column of the :math:`x`-values stored
        on the solver.

        :type solver: :class:`.dg1.DG1Solver`
        :param solver: A solver containing :math:`x`-values.

        :type num_points: int
        :param num_points: (Optional) The number of points to use to represent
                           the polynomial. Defaults to :data:`INTERVAL_POINTS`.

        :rtype: :class:`PolynomialInterpolate`
        :returns: Interpolation object for the reference
        """
        # Reference ``x``-values are in the first column.
        x_vals = solver.node_points[:, 0]
        return cls(x_vals, num_points=num_points)

    def interpolate(self, y_vals):
        r"""Evaluate interpolated polynomial given :math:`y`-values.

        We've already pre-computed the values :math:`\ell_j(x)` for
        all the :math:`x`-values we use in our interval (``num_points`` in
        all, using the interpolating :math:`x`-values to compute the
        :math:`\ell_j(x)`). So we simply use them to compute

        .. math::

           p(x) = \sum_{j} y_j \ell_j(x)

        using the :math:`y_j` from ``y_vals``.

        :type y_vals: :class:`numpy.ndarray`
        :param y_vals: Array of :math:`y`-values that uniquely define
                       our interpolating polynomial. If 1D, converted into
                       a column vector before returning.

        :rtype: :class:`numpy.ndarray`
        :returns: 2D array containing :math:`p(x)` for each :math:`x`-value
                  in the interval (``num_points`` in all). If there are
                  multiple columns in ``y_vals`` (i.e. multiple :math:`p(x)`)
                  then each column of the result will corresponding to each
                  of these polynomials evaluated at ``all_x``.
        """
        if len(y_vals.shape) == 1:
            # Make into a column vector before applying matrix.
            y_vals = y_vals[:, np.newaxis]
        return self.lagrange_matrix.dot(y_vals)


def plot_solution(color, num_cols, interp_func, solver, ax):
    """Plot the solution and return the newly created lines.

    Helper for :class:`DG1Animate`.

    :type color: str
    :param color: The color to use in plotting the solution.

    :type num_cols: int
    :param num_cols: The number of columsn in the ``solution``.

    :type interp_func: :class:`PolynomialInterpolate`
    :param interp_func: The polynomial interpolation object used to map
                        a solution onto a set of points.

    :type solver: :class:`.dg1.DG1Solver`
    :param solver: A solver containing a ``solution`` and ``node_points``.

    :type ax: :class:`matplotlib.artist.Artist`
    :param ax: An axis to be used for plotting.

    :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
    :returns: List of the updated matplotlib line objects.
    """
    plot_lines = []
    all_y = interp_func.interpolate(solver.solution)
    for index in six.moves.xrange(num_cols):
        x_start = solver.node_points[0, index]
        line, = ax.plot(x_start + interp_func.all_x,
                        all_y[:, index],
                        color=color, linewidth=2)
        plot_lines.append(line)

    return plot_lines


def _configure_axis(ax, x_min=0.0, x_max=1.0, y_min=0.0,
                    y_max=1.0, y_buffer=0.1):
    """Configure an axis for plotting.

    Sets the (buffered) bounding box and turns on the grid.

    Helper for :class:`DG1Animate`.

    :type ax: :class:`matplotlib.artist.Artist`
    :param ax: An axis to be used for plotting.

    :type x_min: float
    :param x_min: The minimum :math:`x`-value in the plot.

    :type x_max: float
    :param x_max: The maximum :math:`x`-value in the plot.

    :type y_min: float
    :param y_min: The minimum :math:`y`-value in the plot.

    :type y_max: float
    :param y_max: The maximum :math:`y`-value in the plot.

    :type y_buffer: float
    :param y_buffer: A buffer to allow for noise in a solution
                     in the :math:`y`-direction.
    """
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
    ax.grid(b=True)  # b == boolean, 'on'/'off'


class DG1Animate(object):
    """Helper for animating a solution.

    Assumes the solution (which is updated via ``solver``) produces
    a solution that remains in the same bounding box as :math:`u(x, 0)` (give
    or take some noise).

    :type solver: :class:`.dg1.DG1Solver`
    :param solver: The solver which computes and updates the solution.

    :type fig: :class:`matplotlib.figure.Figure`
    :param fig: (Optional) A figure to use for plotting. Intended to be passed
                when creating a :class:`matplotlib.animation.FuncAnimation`.

    :type ax: :class:`matplotlib.artist.Artist`
    :param ax: (Optional) An axis to be used for plotting.

    :type interp_points: int
    :param interp_points: (Optional) The number of points to use to represent
                          polynomials on an interval. Defaults to
                          :data:`INTERVAL_POINTS`.

    :raises: :class:`ValueError <exceptions.ValueError>` if one of ``fig``
             or ``ax`` is passed, but not both.
    """

    def __init__(self, solver, fig=None, ax=None, interp_points=None):
        self.solver = solver
        # Computed values.
        self.poly_interp_func = PolynomialInterpolate.from_solver(
            solver, num_points=interp_points)
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

    def init_func(self):
        """An initialization function for the animation.

        Plots the initial data **and** stores the lines created.

        :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
        :returns: List of the updated matplotlib line objects,
                  with length equal to :math:`n` (coming from ``solver``).
        """
        # Pre-configure the axes and the background data.
        x_min = np.min(self.solver.node_points)
        x_max = np.max(self.solver.node_points)
        y_min = np.min(self.solver.solution)
        y_max = np.max(self.solver.solution)
        _configure_axis(self.ax, x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max)
        # Plot the initial data (in red) to compare against.
        _, num_cols = self.solver.node_points.shape
        plot_solution('red', num_cols, self.poly_interp_func,
                      self.solver, self.ax)
        # Plot the same data (in blue) and store the lines.
        self.plot_lines = plot_solution(
            'blue', num_cols, self.poly_interp_func, self.solver, self.ax)
        # For ``init_func`` with ``blit`` turned on, the initial
        # frame should not have visible lines. See
        # http://stackoverflow.com/q/21439489/1068170 for more info.
        for line in self.plot_lines:
            line.set_visible(False)
        return self.plot_lines

    def update_plot(self, frame_number):
        """Update the lines in the plot.

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


def plot_convergence(p_order, interval_sizes, colors, solver_factory,
                     interval_width=1.0, total_time=1.0):  #  pragma: NO COVER
    """Plots a convergence plot for a given order.

    Creates a side-by-side of error plots and the solutions as the mesh
    is refined.

    :type p_order: int
    :param p_order: The order of accuracy desired.

    :type interval_sizes: :class:`numpy.ndarray`
    :param interval_sizes: Array of :math:`n` values to use for the number
                           of sub-intervals.

    :type colors: list
    :param colors: List of triples RGB (each a color). Expected to be the
                   same length as ``interval_sizes``.

    :type solver_factory: type
    :param solver_factory: Class that can be used to construct a solver.

    :type interval_width: float
    :param interval_width: (Optional) The width of the interval where the
                           solver works. Defaults to 1.0.

    :type total_time: float
    :param total_time: (Optional) The total time to run the solver. Defaults
                       to 1.0.
    """
    import matplotlib.pyplot as plt
    # Prepare plots.
    rows, cols = 1, 2
    fig, (ax1, ax2) = plt.subplots(rows, cols)

    # Prepare mesh sizes.
    interval_sizes = np.array(interval_sizes)
    dx_vals = interval_width / interval_sizes
    # For the CFL condition: dt = dx / (3 * p * p)
    dt_vals = dx_vals / (3.0 * p_order * p_order)

    # Compute solution on various meshes.
    log2_h = []
    log2_errs = []
    for num_intervals, dt, color in zip(interval_sizes, dt_vals, colors):
        solver = solver_factory(num_intervals=num_intervals,
                                p_order=p_order,
                                total_time=total_time, dt=dt)
        # Save initial solution for later comparison (though a copy is
        # not strictly needed).
        init_soln = solver.solution.copy()
        while solver.current_step != solver.num_steps:
            solver.update()

        frob_err = np.linalg.norm(init_soln - solver.solution, ord='fro')
        log2_h.append(np.log2(interval_width / num_intervals))
        log2_errs.append(np.log2(frob_err))
        interp_func = PolynomialInterpolate.from_solver(solver)
        plotted_lines = plot_solution(color, num_intervals,
                                      interp_func, solver, ax2)
        plt_label = '$n = %d$' % (num_intervals,)
        # Just label the first line (they'll all have the same color).
        plotted_lines[0].set_label(plt_label)

    # Plot the errors.
    ax1.plot(log2_h, log2_errs, label='errors')
    conv_rate, fit_const = np.polyfit(log2_h, log2_errs, deg=1)
    fit_line = conv_rate * np.array(log2_h) + fit_const
    ax1.plot(log2_h, fit_line, label='fit line')

    # Configure the plot.
    fig.set_size_inches(15, 8)
    fig_title = r'$p = %d, rate = %g$' % (p_order, conv_rate)
    fig.suptitle(fig_title, fontsize=20)
    ax1.legend(loc='upper left', fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)
    plt.show()
