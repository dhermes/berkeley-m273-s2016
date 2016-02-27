import unittest

import mock


class Test__forward_substitution(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(lower_tri, rhs_mat, pivots):
        from assignment1.dg1_high_prec import _forward_substitution
        return _forward_substitution(lower_tri, rhs_mat, pivots)

    def test_it(self):
        import mpmath
        import numpy as np

        pivots = [1, 2]
        left_mat = np.array([
            [mpmath.mpf('1.0'), mpmath.mpf('0.0'), mpmath.mpf('0.0')],
            [mpmath.mpf('2.0'), mpmath.mpf('1.0'), mpmath.mpf('0.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('-1.0'), mpmath.mpf('1.0')],
        ])
        # Start with a known solution and work backwards.
        solution = np.array([
            [mpmath.mpf('2.0'), mpmath.mpf('0.0')],
            [mpmath.mpf('-1.0'), mpmath.mpf('-1.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('4.0')],
        ])
        # Compute the corresponding RHS.
        rhs_mat = left_mat.dot(solution)
        # Apply the pivots in reverse (i.e. to un-pivot).
        for index in (1, 0):
            pivot_val = pivots[index]
            rhs_mat[[index, pivot_val], :] = rhs_mat[[pivot_val, index], :]

        with mpmath.mp.workprec(100):
            result = self._call_func_under_test(left_mat, rhs_mat, pivots)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, rhs_mat.shape)
            self.assertTrue(np.all(result == solution))


class Test__back_substitution(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(upper_tri, rhs_mat):
        from assignment1.dg1_high_prec import _back_substitution
        return _back_substitution(upper_tri, rhs_mat)


    def test_it(self):
        import mpmath
        import numpy as np

        left_mat = np.array([
            [mpmath.mpf('3.0'), mpmath.mpf('4.0'), mpmath.mpf('5.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('-1.0'), mpmath.mpf('1.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('0.0'), mpmath.mpf('3.0')],
        ])
        # Start with a known solution and work backwards.
        solution = np.array([
            [mpmath.mpf('4.0'), mpmath.mpf('9.0')],
            [mpmath.mpf('2.0'), mpmath.mpf('1.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('1.0')],
        ])
        # Compute the corresponding RHS.
        rhs_mat = left_mat.dot(solution)

        with mpmath.mp.workprec(100):
            result = self._call_func_under_test(left_mat, rhs_mat)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, rhs_mat.shape)
            self.assertTrue(np.all(result == solution))


class TestHighPrecProvider(unittest.TestCase):

    def setUp(self):
        # Clean out the solve() LU cache.
        klass = self._get_target_class()
        klass._solve_lu_cache.clear()

    @staticmethod
    def _get_target_class():
        from assignment1.dg1_high_prec import HighPrecProvider
        return HighPrecProvider

    def test_exp_func(self):
        import mpmath
        import numpy as np

        exp_func = self._get_target_class().exp_func
        with mpmath.mp.workprec(100):
            scalar_val = mpmath.log('2.0')
            result = exp_func(scalar_val)
            self.assertEqual(result, mpmath.mpf('2.0'))
            mat_val = np.array([
                [mpmath.log('2.0'), mpmath.log('3.0'), mpmath.log('4.0')],
                [mpmath.log('5.0'), mpmath.log('6.0'), mpmath.log('7.0')],
            ])
            result = exp_func(mat_val)
            expected_result = np.array([
                [mpmath.mpf('2.0'), mpmath.mpf('3.0'), mpmath.mpf('4.0')],
                [mpmath.mpf('5.0'), mpmath.mpf('6.0'), mpmath.mpf('7.0')],
            ])
            self.assertTrue(np.all(result == expected_result))

    def test_linspace(self):
        import mpmath
        import numpy as np

        linspace = self._get_target_class().linspace

        result1 = linspace(0, 1, 5)
        self.assertTrue(np.all(result1 == [0, 0.25, 0.5, 0.75, 1.0]))

        with mpmath.mp.workprec(100):
            result2 = linspace(0, 1, 12)
            result3 = linspace(mpmath.mpf('0'), mpmath.mpf('1'), 12)
            self.assertTrue(np.all(result2 == result3))
            expected_result = np.array([
                mpmath.mpf('0/11'), mpmath.mpf('1/11'), mpmath.mpf('2/11'),
                mpmath.mpf('3/11'), mpmath.mpf('4/11'), mpmath.mpf('5/11'),
                mpmath.mpf('6/11'), mpmath.mpf('7/11'), mpmath.mpf('8/11'),
                mpmath.mpf('9/11'), mpmath.mpf('10/11'), mpmath.mpf('11/11'),
            ])
            self.assertTrue(np.all(result2 == expected_result))

    def test_num_type(self):
        import mpmath

        num_type = self._get_target_class().num_type
        self.assertIsInstance(num_type(0), mpmath.mpf)
        self.assertIsInstance(num_type(1.0), mpmath.mpf)
        self.assertIsInstance(num_type('2.1'), mpmath.mpf)

    def test_mat_inv(self):
        import mpmath
        import numpy as np

        mat_inv = self._get_target_class().mat_inv
        sq_mat = np.array([
            [mpmath.mpf('1'), mpmath.mpf('2')],
            [mpmath.mpf('3'), mpmath.mpf('4')],
        ])
        inv_val = mat_inv(sq_mat)
        # Check the type of the output.
        self.assertIsInstance(inv_val, np.ndarray)
        self.assertEqual(inv_val.shape, (2, 2))
        all_types = set([type(val) for val in inv_val.flatten()])
        self.assertEqual(all_types, set([mpmath.mpf]))

        # Check the actual result.
        expected_result = np.array([
            [mpmath.mpf('-2.0'), mpmath.mpf('1.0')],
            [mpmath.mpf('1.5'), mpmath.mpf('-0.5')],
        ])
        delta = np.abs(inv_val - expected_result)
        self.assertLess(np.max(delta), 1e-10)

    def test_solve(self):
        import mpmath
        import numpy as np

        solve = self._get_target_class().solve
        left_mat = np.array([
            [mpmath.mpf('0.0'), mpmath.mpf('1.0'), mpmath.mpf('2.0')],
            [mpmath.mpf('3.0'), mpmath.mpf('4.0'), mpmath.mpf('5.0')],
            [mpmath.mpf('6.0'), mpmath.mpf('7.0'), mpmath.mpf('11.0')],
        ])
        # Start with a known solution and work backwards.
        solution = np.array([
            [mpmath.mpf('2.0'), mpmath.mpf('0.0')],
            [mpmath.mpf('-1.0'), mpmath.mpf('-1.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('4.0')],
        ])
        # Compute the corresponding RHS.
        right_mat = left_mat.dot(solution)

        with mpmath.mp.workprec(100):
            result = solve(left_mat, right_mat)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, right_mat.shape)
            self.assertTrue(np.all(result == solution))

    def test_solve_cache(self):
        import mpmath
        import numpy as np

        klass = self._get_target_class()
        left_mat = np.array([
            [mpmath.mpf('0.0'), mpmath.mpf('1.0'), mpmath.mpf('2.0')],
            [mpmath.mpf('3.0'), mpmath.mpf('4.0'), mpmath.mpf('5.0')],
            [mpmath.mpf('6.0'), mpmath.mpf('7.0'), mpmath.mpf('11.0')],
        ])
        # Start with a known solution and work backwards.
        solution = np.array([
            [mpmath.mpf('2.0'), mpmath.mpf('0.0')],
            [mpmath.mpf('-1.0'), mpmath.mpf('-1.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('4.0')],
        ])
        # Compute the corresponding RHS.
        right_mat = left_mat.dot(solution)

        self.assertEqual(klass._solve_lu_cache, {})
        with mpmath.mp.workprec(100):
            result = klass.solve(left_mat, right_mat)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, right_mat.shape)
            self.assertTrue(np.all(result == solution))

        id_left = id(left_mat)
        self.assertEqual(klass._solve_lu_cache.keys(), [id_left])
        cached_values = klass._solve_lu_cache[id_left]

        lu_parts = np.array([
            [mpmath.mpf('3.0'), mpmath.mpf('4.0'), mpmath.mpf('5.0')],
            [mpmath.mpf('2.0'), mpmath.mpf('-1.0'), mpmath.mpf('1.0')],
            [mpmath.mpf('0.0'), mpmath.mpf('-1.0'), mpmath.mpf('3.0')],
        ])
        self.assertTrue(np.all(cached_values[0] == lu_parts))
        pivots = [1, 2]
        self.assertEqual(cached_values[1], pivots)

        # Call solve() again and verify that LU_decomp() is never used.
        with mock.patch('mpmath.mp.LU_decomp') as lu_decomp:
            with mpmath.mp.workprec(100):
                result = klass.solve(left_mat, right_mat)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, right_mat.shape)
                self.assertTrue(np.all(result == solution))
            lu_decomp.assert_not_called()

    def test_zeros(self):
        import mpmath
        import numpy as np

        zeros = self._get_target_class().zeros
        mat1 = zeros(3)
        self.assertIsInstance(mat1, np.ndarray)
        self.assertEqual(mat1.shape, (3,))
        self.assertEqual(mat1.dtype, object)
        self.assertTrue(np.all(mat1 == mpmath.mpf('0.0')))
        all_types = set([type(val) for val in mat1.flatten()])
        self.assertEqual(all_types, set([mpmath.mpf]))

        mat2 = zeros((3, 7), order='F')
        self.assertIsInstance(mat2, np.ndarray)
        self.assertEqual(mat2.shape, (3, 7))
        self.assertEqual(mat2.dtype, object)
        self.assertTrue(np.all(mat2 == mpmath.mpf('0.0')))
        all_types = set([type(val) for val in mat2.flatten()])
        self.assertEqual(all_types, set([mpmath.mpf]))
        self.assertTrue(mat2.flags.f_contiguous)


class Test_gauss_lobatto_points(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(start, stop, num_points):
        from assignment1.dg1_high_prec import gauss_lobatto_points
        return gauss_lobatto_points(start, stop, num_points)

    def test_it(self):
        import mpmath
        import numpy as np
        from assignment1.dg1 import gauss_lobatto_points as low_prec

        with mpmath.mp.workprec(100):
            result1 = low_prec(-1, 1, 5)
            result2 = self._call_func_under_test(-1, 1, 5)
            self.assertTrue(np.allclose(result1, result2.astype(float)))

            result3 = low_prec(3, 7, 19)
            result4 = self._call_func_under_test(
                mpmath.mpf('3'), mpmath.mpf('7'), 19)
            self.assertTrue(np.allclose(result3, result4.astype(float)))
