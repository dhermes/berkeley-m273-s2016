"""Helpers to use :mod:`assignment1.dg1` with high-precision numbers.

High-precision is achieved by using :mod:`mpmath`.
"""

import mpmath
import numpy as np
import six


_VECTORIZED_EXP = np.vectorize(mpmath.exp)


def _forward_substitution(lower_tri, rhs_mat, pivots):
    r"""Solve a lower triangular system with forward substitution.

    Solves :math:`Lx = PR` for :math:`x` where :math:`P` is a
    permutation matrix and :math:`L` is lower triangular.

    This method is based on
    :meth:`mpmath.matrices.linalg.LinearAlgebraMethods.L_solve`
    and
    :meth:`mpmath.matrices.matrices.MatrixMethods.swap_row` but
    uses NumPy matrices instead and allows the right-hand side to
    be a 2D matrix.

    .. note::

       We assume the caller has verified that ``lower_tri`` is square
       and that ``rhs_mat`` has the same number of rows as ``lower_tri``.

    :type lower_tri: :class:`np.ndarray`
    :param lower_tri: A square :math:`n \times n` matrix.

    :type rhs_mat: :class:`np.ndarray`
    :param rhs_mat: A :math:`n \times m` matrix.

    :type pivots: list
    :param pivots: List of pivots needed to apply the permutation
                   matrix :math:`P`.

    :rtype: :class:`np.ndarray`
    :returns: The solution :math:`x`.
    """
    solution = rhs_mat.copy()
    # Apply the pivots.
    for k, pivot_val in enumerate(pivots):
        solution[[k, pivot_val], :] = solution[[pivot_val, k], :]

    # Now carry out the forward substitution.
    problem_size, _ = lower_tri.shape
    for i in six.moves.xrange(1, problem_size):
        for j in six.moves.xrange(i):
            solution[i, :] -= lower_tri[i, j] * solution[j, :]
    return solution


def _back_substitution(upper_tri, rhs_mat):
    r"""Solve an upper triangular system with back substitution.

    Solves :math:`Ux = R` for :math:`x` where :math:`U` is upper triangular.

    This method is based on
    :meth:`mpmath.matrices.linalg.LinearAlgebraMethods.U_solve`
    but uses NumPy matrices instead and allows the right-hand side to
    be a 2D matrix.

    .. note::

       We assume the caller has verified that ``upper_tri`` is square
       and that ``rhs_mat`` has the same number of rows as ``upper_tri``.

    :type upper_tri: :class:`np.ndarray`
    :param upper_tri: A square :math:`n \times n` matrix.

    :type rhs_mat: :class:`np.ndarray`
    :param rhs_mat: A :math:`n \times m` matrix.

    :rtype: :class:`np.ndarray`
    :returns: The solution :math:`x`.
    """
    problem_size, _ = upper_tri.shape
    solution = rhs_mat.copy()
    for i in six.moves.xrange(problem_size - 1, -1, -1):
        for j in six.moves.xrange(i + 1, problem_size):
            solution[i, :] -= upper_tri[i, j] * solution[j, :]
        solution[i, :] /= upper_tri[i, i]
    return solution


class HighPrecProvider(object):
    """High-precision replacement for :class:`assignment1.dg1.MathProvider`.

    Implements interfaces that are essentially identical (at least up to the
    usage in :mod:`dg1 <assignment1.dg1>`) as those provided by NumPy.

    All matrices returned are :class:`numpy.ndarray` with :class:`mpmath.mpf`
    as the data type and all matrix inputs are assumed to be of the same form.
    """

    _solve_lu_cache = {}

    @staticmethod
    def exp_func(value):
        """Vectorized exponential function."""
        return _VECTORIZED_EXP(value)

    @staticmethod
    def linspace(start, stop, num=50):
        """Linearly spaced points.

        Points are computed with :func:`mpmath.linspace` but the
        output (a ``list``) is converted back to a :class:`numpy.ndarray`.
        """
        return np.array(mpmath.linspace(start, stop, num))

    @staticmethod
    def num_type(value):
        """The high-precision numerical type: :class:`mpmath.mpf`."""
        return mpmath.mpf(value)

    @staticmethod
    def mat_inv(mat):
        """Matrix inversion, using :mod:`mpmath`."""
        inv_mpmath = mpmath.matrix(mat.tolist())**(-1)
        return np.array(inv_mpmath.tolist())

    @classmethod
    def solve(cls, left_mat, right_mat):
        """Solve ``Ax = b`` for ``x``.

        ``A`` is given by ``left_mat`` and ``b`` by ``right_mat``.

        This method seeks to mirror
        :meth:`mpmath.matrices.linalg.LinearAlgebraMethods.lu_solve`,
        which uses
        :meth:`mpmath.matrices.linalg.LinearAlgebraMethods.LU_decomp`,
        :meth:`mpmath.matrices.linalg.LinearAlgebraMethods.L_solve` and
        :meth:`mpmath.matrices.linalg.LinearAlgebraMethods.U_solve`. Due
        to limitations of :mod:`mpmath` we use modified helpers to
        accomplish the upper- and lower-triangular solves. We also cache
        the LU-factorization for future uses.
        """
        left_mat_id = id(left_mat)
        if left_mat_id not in cls._solve_lu_cache:
            # left_mat is a NumPy matrix, so must first be
            # converted to a mpmath matrix.
            lu_parts, pivots = mpmath.mp.LU_decomp(
                mpmath.matrix(left_mat.tolist()))
            # Convert back to a NumPy matrix.
            lu_parts = np.array(lu_parts.tolist())
            cls._solve_lu_cache[left_mat_id] = lu_parts, pivots

        lu_parts, pivots = cls._solve_lu_cache[left_mat_id]
        solution = _forward_substitution(lu_parts, right_mat, pivots)
        solution = _back_substitution(lu_parts, solution)
        return solution

    @staticmethod
    def zeros(shape, **kwargs):
        """Produce a matrix of zeros of a given shape."""
        result = np.empty(shape, dtype=mpmath.mpf, **kwargs)
        result.fill(mpmath.mpf('0.0'))
        return result


def gauss_lobatto_points(start, stop, num_points):
    """Get the node points for Gauss-Lobatto quadrature.

    Rather than using the optimizations in
    :func:`.dg1.gauss_lobatto_points`, this uses :mod:`mpmath` utilities
    directly to find the roots of :math:`P_n'(x)` (where :math:`n` is equal
    to ``num_points - 1``).

    :type start: :class:`mpmath.mpf` (or ``float``)
    :param start: The beginning of the interval.

    :type stop: :class:`mpmath.mpf` (or ``float``)
    :param stop: The end of the interval.

    :type num_points: int
    :param num_points: The number of points to use.

    :rtype: :class:`numpy.ndarray`
    :returns: 1D array, the interior quadrature nodes.
    """
    def leg_poly(value):
        """Legendre polynomial :math:`P_n(x)`."""
        return mpmath.legendre(num_points - 1, value)

    def leg_poly_diff(value):
        """Legendre polynomial derivative :math:`P_n'(x)`."""
        return mpmath.diff(leg_poly, value)

    poly_coeffs = mpmath.taylor(leg_poly_diff, 0, num_points - 2)[::-1]
    inner_roots = mpmath.polyroots(poly_coeffs)
    # Create result.
    start = mpmath.mpf(start)
    stop = mpmath.mpf(stop)
    result = [start]
    # Convert the inner nodes to the interval at hand.
    half_width = (stop - start) / 2
    for index in six.moves.xrange(num_points - 2):
        result.append(start + (inner_roots[index] + 1) * half_width)
    result.append(stop)
    return np.array(result)
