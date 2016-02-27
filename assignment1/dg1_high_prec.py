"""Helpers to use :mod:`assignment1.dg1` with high-precision numbers.

High-precision is achieved by using :mod:`mpmath`.
"""

import mpmath
import numpy as np
import six


_VECTORIZED_EXP = np.vectorize(mpmath.exp)


class HighPrecProvider(object):
    """High-precision replacement for :class:`assignment1.dg1.MathProvider`.

    Implements interfaces that are essentially identical (at least up to the
    usage in :mod:`dg1 <assignment1.dg1>`) as those provided by NumPy.

    All matrices returned are :class:`numpy.ndarray` with :class:`mpmath.mpf`
    as the data type and all matrix inputs are assumed to be of the same form.
    """

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

    @staticmethod
    def solve(left_mat, right_mat):
        """Solve ``Ax = b`` for ``x``.

        ``A`` is given by ``left_mat`` and ``b`` by ``right_mat``.
        """
        raise NotImplementedError

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
