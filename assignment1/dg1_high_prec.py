"""Helpers to use :mod:`assignment1.dg1` with high-precision numbers.

High-precision is achieved by using :mod:`mpmath`.
"""

import mpmath
import numpy as np


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
