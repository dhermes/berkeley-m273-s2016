"""Simple dg1 script from Persson."""


import matplotlib.pyplot as plt
import numpy as np
import six


def dg1(n, p_order, T, dt):
    """Basic dg1 solver.

    A Discontinuous Galerkin (DG) solver to solve the 1D conservation law

    .. math::

       \\frac{\\partial u}{\\partial t} + \\frac{\\partial u}{\\partial x} = 0

    on the interval :math:`\\left[0, 1\\right]` with initial condition

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
    h  = 1.0 / n
    nsteps = int(np.round(T / dt))

    if p_order == 1:
        Mel = h / 6.0 * np.array([
            [2, 1],
            [1, 2],
        ])
        Kel = 0.5 * np.array([
            [-1, -1],
            [ 1,  1],
        ])
        x = np.array([np.linspace(0, 1 - h, n),
                      np.linspace(h, 1, n)])
    elif p_order == 2:
        Mel = h / 30.0 * np.array([
            [ 4,  2, -1],
            [ 2, 16,  2],
            [-1,  2,  4],
        ])
        Kel = np.array([
            [-3, -4,  1],
            [ 4,  0, -4],
            [-1,  4,  3],
        ]) / 6.0
        x = np.array([np.linspace(0, 1 - h, n),
                      np.linspace(0.5 * h, 1 - 0.5 * h, n),
                      np.linspace(h, 1, n)])
    else:
        raise ValueError('Error: p_order not implemented', p_order)

    u = np.exp(-(x - 0.5)**2 / 0.01)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-.1, 1.1))
    ax.plot(x, u, 'r', lw=2)
    plt.grid(True)
    plot_lines = ax.plot(x, u, 'b', lw=2)

    animation_args = (fig, animate)
    wrapped_indices = np.hstack([n - 1, np.arange(n - 1)])
    animation_kwargs = {
        'frames': nsteps + 1,
        'interval': 20,
        'blit': True,
        'fargs': [u, plot_lines, Kel, Mel, dt, wrapped_indices],
    }
    return animation_args, animation_kwargs


def animate(frame_number, u, plot_lines, Kel, Mel, dt, wrapped_indices):
    """Update solution and use new solution to update plotted data.

    Uses the pre-computed mass matrix (``Mel``) and stiffness matrix (``Kel``)
    from :func:`dg1` and a modified RK4 to update the solution. This is
    done by using the following to compute :math:`\dot{u}` via

    .. math::

       M \dot{\mathbf{u}}^k = K \mathbf{u}^k + u_p^{k - 1} e_0 - u_p^k e_{p}

    and using it as a black box for RK4 via :math:`\dot{u} = f(t, u)`.

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

    :type wrapped_indices: :class:`numpy.ndarray`
    :param wrapped_indices: The indices ``[n - 1, 0, 1, ..., n - 2]``.

    :rtype: :class:`list` of :class:`matplotlib.lines.Line2D`
    :returns: List of the updated ``matplotlib`` line objects.
    """
    u_orig = u
    u0 = u
    for irk in six.moves.xrange(4, 0, -1):
        r = np.dot(Kel, u)
        r[-1, :] -= u[-1, :]
        r[0, :] += u[-1, wrapped_indices]
        u = u0 + dt / irk * np.linalg.solve(Mel, r)
    for index, line in enumerate(plot_lines):
        line.set_ydata(u[:, index])
    u_orig[:] = u  # Update the original data.
    return plot_lines
