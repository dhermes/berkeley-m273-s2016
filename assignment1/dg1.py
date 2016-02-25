r"""Module for solving a 1D conservation law via DG.

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


_RK_STEPS = (4, 3, 2, 1)


# pylint: disable=too-few-public-methods
class MathProvider(object):
    """Mutable settings for doing math.

    For callers that wish to swap out the default behavior -- for example,
    to use high-precision values instead of floats -- this class can
    just be monkey patched on the module.

    The module assumes through-out that solution data is in NumPy arrays,
    but the data inside those arrays may be any type.

    .. note::

       The callers assume :data:`exp_func` is a vectorized exponential
       that can act on a NumPy array containing elements of the relevant
       type.

    .. note::

       The :data:`zeros` constructor should also be able to take the
       ``order`` argument (and should produce a NumPy array).
    """
    exp_func = staticmethod(np.exp)
    linspace = staticmethod(np.linspace)
    num_type = staticmethod(float)
    mat_inv = staticmethod(np.linalg.inv)
    mat_mul = staticmethod(np.dot)
    solve = staticmethod(np.linalg.solve)
    zeros = staticmethod(np.zeros)
# pylint: enable=too-few-public-methods


def gauss_lobatto_points(start, stop, num_points):
    r"""Get the node points for Gauss-Lobatto quadrature.

    Using :math:`n` points, this quadrature is accurate to degree
    :math:`2n - 3`. The node points are :math:`x_1 = -1`,
    :math:`x_n = 1` and the interior are :math:`n - 2` roots of
    :math:`P'_{n - 1}(x)`.

    Though we don't compute them here, the weights are
    :math:`w_1 = w_n = \frac{2}{n(n - 1)}` and for the interior points

    .. math::

       w_j = \frac{2}{n(n - 1) \left[P_{n - 1}\left(x_j\right)\right]^2}

    This is in contrast to the scheme used in Gaussian quadrature, which
    use roots of :math:`P_n(x)` as nodes and use the weights

    .. math::

       w_j = \frac{2}{\left(1 - x_j\right)^2
                \left[P'_n\left(x_j\right)\right]^2}

    .. note::

       This method is **not** generic enough to accommodate non-NumPy
       types as it relies on the :mod:`numpy.polynomial.legendre`.

    :type start: float
    :param start: The beginning of the interval.

    :type stop: float
    :param stop: The end of the interval.

    :type num_points: int
    :param num_points: The number of points to use.

    :rtype: :class:`numpy.ndarray`
    :returns: 1D array, the interior quadrature nodes.
    """
    p_n_minus1 = [0] * (num_points - 1) + [1]
    inner_nodes = legendre.legroots(legendre.legder(p_n_minus1))
    # Utilize symmetry about 0.
    inner_nodes = 0.5 * (inner_nodes - inner_nodes[::-1])
    if start != -1.0 or stop != 1.0:
        # [-1, 1] --> [0, 2] --> [0, stop - start] --> [start, stop]
        inner_nodes = start + (inner_nodes + 1.0) * 0.5 * (stop - start)
    return np.hstack([[start], inner_nodes, [stop]])


def get_legendre_matrix(points, max_degree=None):
    r"""Evaluate Legendre polynomials at a set of points.

    If our points are :math:`x_0, \ldots, x_p`, this computes

    .. math::

       \left[ \begin{array}{c c c c}
         L_0(x_0) & L_1(x_0) & \cdots & L_d(x_0) \\
         L_0(x_1) & L_1(x_1) & \cdots & L_d(x_p) \\
           \vdots &   \vdots & \ddots & \vdots   \\
         L_0(x_p) & L_1(x_p) & \cdots & L_d(x_p)
       \end{array}\right]

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
    num_points, = np.shape(points)
    if max_degree is None:
        max_degree = num_points - 1
    # Use Fortran order since we operate on columns.
    result = MathProvider.zeros((num_points, max_degree + 1), order='F')
    result[:, 0] = MathProvider.num_type(1.0)
    result[:, 1] = points
    for degree in six.moves.xrange(2, max_degree + 1):
        result[:, degree] = (
            (2 * degree - 1) * points * result[:, degree - 1] -
            (degree - 1) * result[:, degree - 2]) / degree
    return result


def _find_matrices_helper(vals1, vals2):
    r"""Helper for :func:`find_matrices`.

    Computes a shoelace-like product of two vectors :math:`u, v`
    via

    .. math::

        u_0 (v_1 + v_3 + \cdots) + u_1 (v_2 + v_4 + \cdots) +
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


def get_evenly_spaced_points(start, stop, num_points):
    """Get points on an interval that are evenly spaced.

    This is intended to be used to give points on a reference
    interval when using DG on the 1D problem.

    :type start: float
    :param start: The beginning of the interval.

    :type stop: float
    :param stop: The end of the interval.

    :type num_points: int
    :param num_points: The number of points to use on the interval.

    :rtype: :class:`numpy.ndarray`
    :returns: The evenly spaced points on the interval.
    """
    return MathProvider.linspace(start, stop, num_points)


def find_matrices(p_order, points_on_ref_int=None):
    r"""Find mass and stiffness matrices.

    We do this on the reference interval :math:`\left[-1, 1\right]`.
    By default we use the evenly spaced points

    .. math::

       x_0 = -1, x_1 = -(p - 2)/p, \ldots, x_p = 1

    but the set of nodes to use on the reference interval can be specified
    via the ``points_on_ref_int`` argument. With our points, we
    compute the polynomials :math:`\varphi_j(x)` such that
    :math:`\varphi_j\left(x_i\right) = \delta_{ij}`. We do this by
    writing

    .. math::

       \varphi_j(x) = \sum_{n = 0}^p c_n^{(j)} L_n(x)

    where :math:`L_n(x)` is the Legendre polynomial of degree :math:`n`.
    With this representation, we need to solve

    .. math::

       \left[ \begin{array}{c c c c}
         L_0(x_0) & L_1(x_0) & \cdots & L_p(x_0) \\
         L_0(x_1) & L_1(x_1) & \cdots & L_p(x_p) \\
          \vdots  & \vdots   & \ddots & \vdots   \\
         L_0(x_p) & L_1(x_p) & \cdots & L_p(x_p)
       \end{array}\right]
       \left[ \begin{array}{c c c c}
         c_0^{(0)} & c_0^{(1)} & \cdots & c_0^{(p)} \\
         c_1^{(0)} & c_1^{(1)} & \cdots & c_1^{(p)} \\
            \vdots &    \vdots & \ddots & \vdots    \\
         c_p^{(0)} & c_p^{(1)} & \cdots & c_p^{(p)}
       \end{array}\right]
       = \left(\delta_{ij}\right) = I_{p + 1}

    Then use these to compute the mass matrix

    .. math::

        M_{ij} = \int_{-1}^1 \varphi_i(x) \varphi_j(x) \, dx

    and the stiffness matrix

    .. math::

        K_{ij} = \int_{-1}^1 \varphi_i'(x) \varphi_j(x) \, dx

    Utilizing the fact that

    .. math::

        \left\langle L_n, L_m \right\rangle =
            \int_{-1}^1 L_n(x) L_m(x) \, dx =
            \frac{2}{2n + 1} \delta_{nm}

    we can compute

    .. math::

        M_{ij} = \left\langle \varphi_i, \varphi_j \right\rangle =
            \sum_{n, m} \left\langle c_n^{(i)} L_n,
                c_m^{(j)} L_m \right\rangle =
            \sum_{n = 0}^p \frac{2}{2n + 1} c_n^{(i)} c_n^{(j)}.

    Similarly

    .. math::

        \left\langle L_n'(x), L_m(x) \right\rangle =
          \begin{cases}
            2 & \text{ if } n > m \text{ and }
                n - m \equiv 1 \bmod{2} \\
            0 & \text{ otherwise}.
          \end{cases}

    gives

    .. math::

        \begin{align*}
        K_{ij} &= \left\langle \varphi_i', \varphi_j \right\rangle
                = \sum_{n, m} \left\langle c_n^{(i)} L_n',
                      c_m^{(j)} L_m \right\rangle \\
               &= 2 \left(c_0^{(j)} \left(c_1^{(i)} + c_3^{(i)} +
                      \cdots\right)
                         + c_1^{(j)} \left(c_2^{(i)} + c_4^{(i)} +
                      \cdots\right)
                         + \cdots
                         + c_{p - 1}^{(j)} c_p^{(i)}\right) \\
        \end{align*}

    (For more general integrals, one might use Gaussian quadrature.
    The largest degree integrand :math:`\varphi_i \varphi_j` has
    degree :math:`2 p` so this would require :math:`n = p + 1` points
    to be exact up to degree :math:`2(p + 1) - 1 = 2p + 1`.)

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type points_on_ref_int: :data:`function <types.FunctionType>`
    :param points_on_ref_int: (Optional) The method used to partition the
                              reference interval :math:`\left[0, h\right]`
                              into node points. Defaults to
                              :func:`get_evenly_spaced_points`.

    :rtype: tuple
    :returns: Pair of mass and stiffness matrices, square
              :class:`numpy.ndarray` of dimension ``p_order + 1``.
    """
    if points_on_ref_int is None:
        points_on_ref_int = get_evenly_spaced_points
    # Find the coefficients of the L_n(x) for each basis function.
    x_vals = points_on_ref_int(-1, 1, p_order + 1)
    coeff_mat = MathProvider.mat_inv(get_legendre_matrix(x_vals))

    # Populate the mass and stiffness matrices.
    legendre_norms = (MathProvider.num_type(2.0) /
                      np.arange(1, 2 * p_order + 2, 2))
    mass_mat = MathProvider.zeros((p_order + 1, p_order + 1))
    stiffness_mat = MathProvider.zeros((p_order + 1, p_order + 1))
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
    r"""Update an ODE solutuon with an order 2/4 Runge-Kutta function.

    The method is given by the following Butcher array:

    .. math::

           \begin{array}{c | c c c c}
             0 &   0 &     &     &   \\
           1/4 & 1/4 &   0 &     &   \\
           1/3 &   0 & 1/3 &   0 &   \\
           1/2 &   0 &   0 & 1/2 & 0 \\
           \hline
               &   0 &   0 &   0 & 1
           \end{array}

    It is advantageous because the updates :math:`k_j` can be over-written at
    each step, since they are never re-used.

    One can see that this method is **order 2** for general
    :math:`\dot{u} = f(u)` by verifying that not all order 3 node
    conditions are satisfied. For example:

    .. math::

       \frac{1}{3} \neq \sum_i b_i c_i^2 = 0 + 0 + 0 +
       1 \cdot \left(\frac{1}{2}\right)^2

    However, for linear ODEs, the method is **order 4**. To see this, note
    that the test problem :math:`\dot{u} = \lambda u` gives the stability
    function

    .. math::

        R\left(\lambda \Delta t\right) = R(z) =
        1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}

    which matches the Taylor series for :math:`e^z` to order 4.

    See `Problem Set 3`_ from Persson's Math 228A for more details.

    .. _Problem Set 3: http://persson.berkeley.edu/228A/ps3.pdf

    :type ode_func: callable
    :param ode_func: The RHS in the ODE :math:`\dot{u} = f(u)`.

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


def get_node_points(num_points, p_order, step_size=None,
                    points_on_ref_int=None):
    r"""Return node points to split unit interval for DG.

    :type num_points: int
    :param num_points: The number :math:`n` of intervals to divide
                       :math:`\left[0, 1\right]` into.

    :type p_order: int
    :param p_order: The degree of precision for the method.

    :type step_size: float
    :param step_size: (Optional) The step size :math:`1 / n`.

    :type points_on_ref_int: :data:`function <types.FunctionType>`
    :param points_on_ref_int: (Optional) The method used to partition the
                              reference interval :math:`\left[0, h\right]`
                              into node points. Defaults to
                              :func:`get_evenly_spaced_points`.

    :rtype: :class:`numpy.ndarray`
    :returns: The :math:`x`-values for the node points, with
              ``p_order + 1`` rows and :math:`n` columns. The columns
              correspond to each sub-interval and the rows correspond
              to the node points within each sub-interval.
    """
    if step_size is None:
        step_size = MathProvider.num_type(1.0) / num_points
    if points_on_ref_int is None:
        points_on_ref_int = get_evenly_spaced_points
    interval_starts = MathProvider.linspace(0, 1 - step_size, num_points)
    # Split the first interval [0, h] in ``p_order + 1`` points.
    first_interval = points_on_ref_int(0, step_size, p_order + 1)
    # Broadcast the values with ``first_interval`` as rows and
    # columns as ``interval_starts``.
    return (first_interval[:, np.newaxis] +
            interval_starts[np.newaxis, :])


def get_gaussian_like_initial_data(node_points):
    r"""Get the default initial solution data.

    In this case it is

    .. math::

       u(x, 0) = \exp\left(-\left(\frac{x - \frac{1}{2}}{0.1}
                             \right)^2\right)

    :type node_points: :class:`numpy.ndarray`
    :param node_points: Points at which evaluate the initial data function.

    :rtype: :class:`numpy.ndarray`
    :returns: The :math:`u`-values at each node point.
    """
    return MathProvider.exp_func(- 25 * (2 * node_points - 1)**2)


class DG1Solver(object):
    r"""Discontinuous Galerkin (DG) solver for the 1D conservation law.

    .. math::

       \frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0

    on the interval :math:`\left[0, 1\right]`. By default, uses
    Gaussian-like initial data

    .. math::

       u(x, 0) = \exp\left(-\left(\frac{x - \frac{1}{2}}{0.1}
                             \right)^2\right)

    but :math:`u(x, 0)` can be specified via ``get_initial_data``.

    We represent our solution via the :math:`(p + 1) \times n` rectangular
    matrix:

        .. math::

           \mathbf{u} = \left[ \begin{array}{c c c c}
                 u_0^1 &  u_0^2 & \cdots &  u_0^n \\
                 u_1^1 &  u_1^2 & \cdots &  u_1^n \\
                \vdots & \vdots & \ddots & \vdots \\
                 u_p^1 &  u_p^2 & \cdots &  u_p^n
               \end{array}\right]

    where each column represents one of :math:`n` sub-intervals and each row
    represents one of the :math:`p + 1` node points within each sub-interval.

    :type num_intervals: int
    :param num_intervals: The number :math:`n` of intervals to divide
                          :math:`\left[0, 1\right]` into.

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

    :type points_on_ref_int: :data:`function <types.FunctionType>`
    :param points_on_ref_int: (Optional) The method used to partition the
                              reference interval :math:`\left[0, h\right]`
                              into node points. Defaults to
                              :func:`get_evenly_spaced_points`.
    """

    def __init__(self, num_intervals, p_order, total_time, dt,
                 get_initial_data=None, points_on_ref_int=None):
        self.num_intervals = num_intervals
        self.p_order = p_order
        self.total_time = total_time
        self.dt = dt
        self.current_step = 0
        self.points_on_ref_int = points_on_ref_int
        # Computed values.
        self.num_steps = int(round(self.total_time / self.dt))
        self.step_size = MathProvider.num_type(1.0) / self.num_intervals
        self.node_points = get_node_points(
            self.num_intervals, self.p_order,
            step_size=self.step_size,
            points_on_ref_int=self.points_on_ref_int)
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
        mass_mat, stiffness_mat = find_matrices(
            self.p_order, points_on_ref_int=self.points_on_ref_int)
        # We are solving on [0, 1] but ``find_matrices`` is
        # on [-1, 1], and the mass matrix is translation invariant
        # but scales with interval length.
        scaled_mass_mat = (self.step_size *
                           MathProvider.num_type(0.5) * mass_mat)
        return scaled_mass_mat, stiffness_mat

    def ode_func(self, u_val):
        r"""Compute the right-hand side for the ODE.

        When we write

        .. math::

           M \dot{\mathbf{u}} = K \mathbf{u} +
           \left[ \begin{array}{c c c c c}
                     u_p^2 &  u_p^3 & \cdots &      u_p^n & u_p^1  \\
                         0 &      0 & \cdots &          0 & 0      \\
                    \vdots & \vdots & \ddots &     \vdots & \vdots \\
                         0 &      0 & \cdots &          0 & 0      \\
                    -u_p^1 & -u_p^2 & \cdots & -u_p^{n-1} & -u_p^n \\
                    \end{array}\right]

        we specify a RHS :math:`f(u)` via solving the system.

        :type u_val: :class:`numpy.ndarray`
        :param u_val: The input to :math:`f(u)`.

        :rtype: :class:`numpy.ndarray`
        :returns: The value of the slope function evaluated at ``u_val``.
        """
        rhs = MathProvider.mat_mul(self.stiffness_mat, u_val)

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
        return MathProvider.solve(self.mass_mat, rhs)

    def update(self):
        r"""Update the solution for a single time step.

        We use :meth:`ode_func` to compute :math:`\dot{u} = f(u)` and
        pair it with an RK method (:func:`low_storage_rk`) to compute
        the updated value.
        """
        self.solution = low_storage_rk(self.ode_func, self.solution, self.dt)
        self.current_step += 1
