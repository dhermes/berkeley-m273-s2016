import unittest


class TestHighPrecProvider(unittest.TestCase):

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
        solve = self._get_target_class().solve
        with self.assertRaises(NotImplementedError):
            solve(None, None)

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
