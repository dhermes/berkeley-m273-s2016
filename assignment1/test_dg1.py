import unittest

import mock


class Test_get_symbolic_vandermonde(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order):
        from assignment1.dg1 import get_symbolic_vandermonde
        return get_symbolic_vandermonde(p_order)

    def test_p1(self):
        import sympy

        x_symb = sympy.Symbol('x')
        x_vec, vand_mat = self._call_func_under_test(1)
        self.assertTrue(isinstance(x_vec, sympy.Matrix))
        self.assertEqual(x_vec.tolist(), [[1, x_symb]])
        self.assertTrue(isinstance(vand_mat, sympy.Matrix))
        expected_vand = [
            [1, 0],
            [1, 1],
        ]
        self.assertEqual(vand_mat.tolist(), expected_vand)

    def test_p4(self):
        import sympy

        x_symb = sympy.Symbol('x')
        x_vec, vand_mat = self._call_func_under_test(4)
        self.assertTrue(isinstance(x_vec, sympy.Matrix))
        self.assertEqual(x_vec.tolist(),
                         [[1, x_symb, x_symb**2, x_symb**3, x_symb**4]])
        self.assertTrue(isinstance(vand_mat, sympy.Matrix))
        expected_vand = [
            [256, 0, 0, 0, 0],
            [256, 64, 16, 4, 1],
            [256, 128, 64, 32, 16],
            [256, 192, 144, 108, 81],
            [256, 256, 256, 256, 256],
        ]
        self.assertEqual((256 * vand_mat).tolist(), expected_vand)


class Test_find_matrices_symbolic(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order):
        from assignment1.dg1 import find_matrices_symbolic
        return find_matrices_symbolic(p_order)

    def test_p1(self):
        import sympy

        mass_mat, stiffness_mat = self._call_func_under_test(1)
        self.assertTrue(isinstance(mass_mat, sympy.Matrix))
        self.assertEqual((6 * mass_mat).tolist(),
                         [[2, 1], [1, 2]])
        self.assertTrue(isinstance(stiffness_mat, sympy.Matrix))
        self.assertEqual((2 * stiffness_mat).tolist(),
                         [[-1, -1], [1, 1]])

    def test_p2(self):
        import sympy

        mass_mat, stiffness_mat = self._call_func_under_test(2)
        self.assertTrue(isinstance(mass_mat, sympy.Matrix))
        self.assertEqual((30 * mass_mat).tolist(),
                         [[4, 2, -1], [2, 16, 2], [-1, 2, 4]])
        self.assertTrue(isinstance(stiffness_mat, sympy.Matrix))
        self.assertEqual((6 * stiffness_mat).tolist(),
                         [[-3, -4, 1], [4, 0, -4], [-1, 4, 3]])


class Test_mass_and_stiffness_matrices_p1(unittest.TestCase):

    @staticmethod
    def _call_func_under_test():
        from assignment1.dg1 import mass_and_stiffness_matrices_p1
        return mass_and_stiffness_matrices_p1()

    def test_value(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat, stiffness_mat = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))

        self.assertTrue(np.all(6 * mass_mat == [[2, 1], [1, 2]]))
        self.assertTrue(np.all(2 * stiffness_mat == [[-1, -1], [1, 1]]))

    def test_identical(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat1, stiffness_mat1 = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat1, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat1, np.ndarray))
        mass_mat2, stiffness_mat2 = self._call_func_under_test()

        self.assertFalse(mass_mat1 is mass_mat2)
        self.assertTrue(np.all(mass_mat1 == mass_mat2))
        self.assertFalse(stiffness_mat1 is stiffness_mat2)
        self.assertTrue(np.all(stiffness_mat1 == stiffness_mat2))


class Test_mass_and_stiffness_matrices_p2(unittest.TestCase):

    @staticmethod
    def _call_func_under_test():
        from assignment1.dg1 import mass_and_stiffness_matrices_p2
        return mass_and_stiffness_matrices_p2()

    def test_value(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat, stiffness_mat = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))

        expected_mass_mat = [[4, 2, -1], [2, 16, 2], [-1, 2, 4]]
        self.assertTrue(np.all(30 * mass_mat == expected_mass_mat))
        expected_stiffness_mat = [[-3, -4, 1], [4, 0, -4], [-1, 4, 3]]
        self.assertTrue(np.all(6 * stiffness_mat == expected_stiffness_mat))

    def test_identical(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat1, stiffness_mat1 = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat1, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat1, np.ndarray))
        mass_mat2, stiffness_mat2 = self._call_func_under_test()

        self.assertFalse(mass_mat1 is mass_mat2)
        self.assertTrue(np.all(mass_mat1 == mass_mat2))
        self.assertFalse(stiffness_mat1 is stiffness_mat2)
        self.assertTrue(np.all(stiffness_mat1 == stiffness_mat2))


class Test_mass_and_stiffness_matrices_p3(unittest.TestCase):

    @staticmethod
    def _call_func_under_test():
        from assignment1.dg1 import mass_and_stiffness_matrices_p3
        return mass_and_stiffness_matrices_p3()

    def test_value(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat, stiffness_mat = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))

        expected_mass_mat = [
            [128, 99, -36, 19],
            [99, 648, -81, -36],
            [-36, -81, 648, 99],
            [19, -36, 99, 128],
        ]
        self.assertTrue(np.all(1680 * mass_mat == expected_mass_mat))
        expected_stiffness_mat = [
            [-40, -57, 24, -7],
            [57, 0, -81, 24],
            [-24, 81, 0, -57],
            [7, -24, 57, 40],
        ]
        self.assertTrue(np.all(80 * stiffness_mat == expected_stiffness_mat))

    def test_identical(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat1, stiffness_mat1 = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat1, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat1, np.ndarray))
        mass_mat2, stiffness_mat2 = self._call_func_under_test()

        self.assertFalse(mass_mat1 is mass_mat2)
        self.assertTrue(np.all(mass_mat1 == mass_mat2))
        self.assertFalse(stiffness_mat1 is stiffness_mat2)
        self.assertTrue(np.all(stiffness_mat1 == stiffness_mat2))


class Test_find_matrices(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order):
        from assignment1.dg1 import find_matrices
        return find_matrices(p_order)

    def test_p1(self):
        import numpy as np
        from assignment1.dg1 import mass_and_stiffness_matrices_p1

        mass_mat, stiffness_mat = self._call_func_under_test(1)
        (expected_mass_mat,
         expected_stiffness_mat) = mass_and_stiffness_matrices_p1()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        self.assertTrue(np.allclose(mass_mat, expected_mass_mat))
        self.assertTrue(np.allclose(stiffness_mat, expected_stiffness_mat))

    def test_p2(self):
        import numpy as np
        from assignment1.dg1 import mass_and_stiffness_matrices_p2

        mass_mat, stiffness_mat = self._call_func_under_test(2)
        (expected_mass_mat,
         expected_stiffness_mat) = mass_and_stiffness_matrices_p2()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        self.assertTrue(np.allclose(mass_mat, expected_mass_mat))
        self.assertTrue(np.allclose(stiffness_mat, expected_stiffness_mat))


class Test_low_storage_rk(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(ode_func, u_val, dt):
        from assignment1.dg1 import low_storage_rk
        return low_storage_rk(ode_func, u_val, dt)

    @staticmethod
    def _sympy_copyable_symbol():
        import sympy

        class DoNothingCopy(sympy.Symbol):

            def copy(self):
                return self

        return DoNothingCopy

    def test_symbolic_ode_func(self):
        import sympy

        dt = sympy.Symbol('dt')
        u_val = self._sympy_copyable_symbol()('u')
        ode_func = sympy.Function('f')

        # Check that the linear method acts as truncated exponential.
        result = self._call_func_under_test(ode_func, u_val, dt)
        val0 = u_val
        val1 = u_val + dt/4 * ode_func(val0)
        val2 = u_val + dt/3 * ode_func(val1)
        val3 = u_val + dt/2 * ode_func(val2)
        val4 = u_val + dt * ode_func(val3)
        self.assertEqual(result, val4)

    def test_matches_exponential(self):
        import sympy

        dt, lambda_ = sympy.symbols('lambda, dt')
        u_val = self._sympy_copyable_symbol()('u')

        ode_called = []

        def mock_ode_func(value):
            ode_called.append(value)
            return lambda_ * value

        # Check that the linear method acts as truncated exponential.
        result = self._call_func_under_test(mock_ode_func, u_val, dt)
        z = dt * lambda_
        expected_result = u_val * (1 + z + z**2/2 + z**3/6 + z**4/24)
        self.assertEqual(result.expand(), expected_result.expand())

        # Check that the terms are built up as described in the Butcher array.
        self.assertEqual(ode_called[0], u_val)
        expected_called = u_val
        for called_val, irk in zip(ode_called[1:], (4, 3, 2, 1)):
            expected_called = u_val + z/irk * expected_called
            self.assertEqual(called_val.expand(), expected_called.expand())


class Test_get_node_points(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(num_points, p_order, step_size=None):
        from assignment1.dg1 import get_node_points
        return get_node_points(num_points, p_order, step_size=step_size)

    def test_default_step_size(self):
        import numpy as np

        num_points = 4
        p_order = 2
        result = self._call_func_under_test(num_points, p_order)
        self.assertTrue(isinstance(result, np.ndarray))
        expected_result = [
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [2, 4, 6, 8],
        ]
        self.assertTrue(np.all(8 * result == expected_result))

    def test_explicit_step_size(self):
        import numpy as np

        num_points = 4
        p_order = 2
        step_size = 1.0 / 3
        result = self._call_func_under_test(num_points, p_order,
                                            step_size=step_size)
        self.assertTrue(isinstance(result, np.ndarray))
        expected_result = [
            [0, 4, 8, 12],
            [3, 7, 11, 15],
            [6, 10, 14, 18],
        ]
        self.assertTrue(np.allclose(18 * result, expected_result))


class Test_make_lagrange_matrix(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(x_vals, all_x):
        from assignment1.dg1 import make_lagrange_matrix
        return make_lagrange_matrix(x_vals, all_x)

    def test_it(self):
        import numpy as np

        x_vals = np.array([0., 1., 2.])
        # The Lagrange functions for the interpolating x values 0, 1, 2
        # are simply L0(x) = (x - 1)(x - 2)/2, L1(x) = x(2 - x) and
        # L2(x) = x(x - 1)/2.
        all_x = np.array([3., 5., 8., 13.])
        result = self._call_func_under_test(x_vals, all_x)
        expected_result = [
            [1, -3, 3],
            [6, -15, 10],
            [21, -48, 28],
            [66, -143, 78],
        ]
        self.assertTrue(np.all(result == expected_result))


class TestPolynomialInterpolate(unittest.TestCase):

    @staticmethod
    def _get_target_class():
        from assignment1.dg1 import PolynomialInterpolate
        return PolynomialInterpolate

    def _make_one(self, x_vals, num_points=None):
        return self._get_target_class()(x_vals, num_points=num_points)

    def test_constructor_defaults(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0])
        num_points = 3
        with mock.patch('assignment1.dg1.INTERVAL_POINTS', new=num_points):
            interp_func = self._make_one(x_vals)

        self.assertTrue(interp_func.x_vals is x_vals)
        # Split the interval [0, 1] into ``num_points`` == 3 points.
        self.assertTrue(np.all(interp_func.all_x == [0, 0.5, 1.0]))
        # The Lagrange functions for the interpolating x values 0, 1
        # are simply L0(x) = 1 - x and L1(x) = x. Evaluating these
        # at [0, 0.5, 1] is straightforward.
        expected_lagrange = [
            [1, 0],
            [0.5, 0.5],
            [0, 1],
        ]
        self.assertTrue(
            np.all(interp_func.lagrange_matrix == expected_lagrange))

    def test_constructor_explicit(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0, 2.0])
        num_points = 5
        interp_func = self._make_one(x_vals, num_points=num_points)

        self.assertTrue(interp_func.x_vals is x_vals)
        # Split the interval [0, 2] into ``num_points`` == 5 points.
        self.assertTrue(np.all(interp_func.all_x == [0, 0.5, 1.0, 1.5, 2.0]))
        # The Lagrange functions for the interpolating x values 0, 1, 2
        # are simply L0(x) = (x - 1)(x - 2)/2, L1(x) = x(2 - x) and
        # L2(x) = x(x - 1)/2. Evaluating these at [0, 0.5, 1, 1.5, 2] gives.
        expected_lagrange = [
            [8, 0, 0],
            [3, 6, -1],
            [0, 8, 0],
            [-1, 6, 3],
            [0, 0, 8],
        ]
        self.assertTrue(
            np.all(8 * interp_func.lagrange_matrix == expected_lagrange))

    def test_interpolate_1D_data(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0, 2.0])
        interp_func = self._make_one(x_vals, num_points=5)
        y_vals = np.array([2., 3., 2.])
        # Our polynomial is
        #        2 L0(x) + 3 L1(x) + 2 L2(x)
        #     == (x - 1)(x - 2) + 3x(2 - x) + x(x - 1)
        #     == - x^2 + 2x + 2
        # and we evaluate it at the 5 points [0, 0.5, 1.0, 1.5, 2.0].
        self.assertEqual(y_vals.shape, (3,))
        result = interp_func.interpolate(y_vals)
        # Verify it becomes 2D.
        self.assertEqual(result.shape, (5, 1))
        expected_result = [
            [2.],
            [2.75],
            [3.],
            [2.75],
            [2.],
        ]
        self.assertTrue(np.all(result == expected_result))

    def test_interpolate_2D_data(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0])
        interp_func = self._make_one(x_vals, num_points=3)
        y_vals = np.array([
            [10., 11., 5.],
            [20., 12., 4.],
        ])
        # Our polynomials are
        #     10 L0(x) + 20 L1(x) = 10(1 - x) + 20x = 10 + 10x
        #     11 L0(x) + 12 L1(x) = 11(1 - x) + 12x = 11 +   x
        #      5 L0(x) +  4 L1(x) =  5(1 - x) +  4x =  5 -   x
        # and we evaluate them at the 3 points [0, 0.5, 2.0]
        self.assertEqual(y_vals.shape, (2, 3))
        result = interp_func.interpolate(y_vals)
        # Verify it becomes 2D.
        self.assertEqual(result.shape, (3, 3))
        expected_result = [
            [10, 11, 5],
            [15, 11.5, 4.5],
            [20, 12, 4],
        ]
        self.assertTrue(np.all(result == expected_result))
