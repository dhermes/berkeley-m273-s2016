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


class Test_gauss_lobatto_info(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(num_points):
        from assignment1.dg1 import gauss_lobatto_info
        return gauss_lobatto_info(num_points)

    def test_five_points(self):
        import numpy as np
        import sympy

        t_symb = sympy.Symbol('t')
        num_points = 5
        p4 = sympy.legendre(num_points - 1, t_symb)
        p4_prime = p4.diff(t_symb)
        inner_roots, inner_weights = self._call_func_under_test(num_points)
        # Make sure the computed roots are actually roots.
        self.assertTrue(np.allclose(
            [float(p4_prime.subs({t_symb: root}))
             for root in inner_roots], 0))

        symbolic_roots = sorted(sympy.roots(p4_prime, multiple=True))
        base_weight = sympy.Rational(2, num_points * (num_points - 1))
        sym_weights = [base_weight / p4.subs({t_symb: sym_root})**2
                       for sym_root in symbolic_roots]
        self.assertTrue(np.all(inner_weights == sym_weights))

    def test_symmetry(self):
        import numpy as np

        inner_roots, inner_weights = self._call_func_under_test(15)
        self.assertTrue(np.all(inner_roots == -inner_roots[::-1]))
        self.assertTrue(np.all(inner_weights == inner_weights[::-1]))

    @staticmethod
    def _accuracy_helper(degree, all_roots, all_weights):
        import numpy as np
        from numpy.polynomial import polynomial

        if degree % 2 == 0:
            expected_value = 2.0 / (degree + 1)
        else:
            # Odd-function integrate to 0.
            expected_value = 0.0
        curr_poly = [0.0] * degree + [1.0]
        poly_vals = polynomial.polyval(all_roots, curr_poly)
        quadrature_val = np.dot(poly_vals, all_weights)
        return expected_value, quadrature_val

    def test_accuracy(self):
        import numpy as np
        import six

        num_points = 4
        inner_roots, inner_weights = self._call_func_under_test(num_points)
        all_roots = np.hstack([[-1], inner_roots, [1]])
        base_weight = 2.0 / (num_points * (num_points - 1))
        all_weights = np.hstack([[base_weight], inner_weights,
                                 [base_weight]])
        for degree in six.moves.xrange(2 * num_points - 2):
            # Verify x^degree is exact.
            expected_value, quadrature_val = self._accuracy_helper(
                degree, all_roots, all_weights)
            self.assertTrue(np.allclose(quadrature_val, expected_value))

        # Verify that the next degree does not integrate exactly.
        expected_value, quadrature_val = self._accuracy_helper(
            2 * num_points - 2, all_roots, all_weights)
        self.assertFalse(np.allclose(expected_value, quadrature_val))


class Test_get_legendre_matrix(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(points, max_degree=None):
        from assignment1.dg1 import get_legendre_matrix
        return get_legendre_matrix(points, max_degree=max_degree)

    def test_degree0(self):
        import numpy as np

        with self.assertRaises(IndexError):
            self._call_func_under_test(np.array([1]))
        with self.assertRaises(IndexError):
            self._call_func_under_test(np.array([1, 2]), max_degree=0)

    def test_evenly_spaced(self):
        import numpy as np
        from numpy.polynomial import legendre
        import six

        num_points = 17
        points = np.linspace(0, 1, num_points)
        result = self._call_func_under_test(points)
        self.assertEqual(result.shape, (num_points, num_points))
        self.assertTrue(result.flags.f_contiguous)
        expected_result = np.zeros((num_points, num_points))
        for n in six.moves.xrange(num_points):
            leg_coeffs = [0] * n + [1]
            expected_result[:, n] = legendre.legval(points, leg_coeffs)
        self.assertTrue(np.allclose(result, expected_result))
        frob_err = np.linalg.norm(result - expected_result, ord='fro')
        self.assertTrue(frob_err < 1e-13)

    def test_chebyshev_explicit_degree(self):
        import numpy as np
        from numpy.polynomial import legendre
        import six

        num_nodes = 8
        theta = np.pi * np.arange(2 * num_nodes - 1, 0, -2,
                                  dtype=np.float64) / (2 * num_nodes)
        points = np.cos(theta)
        max_degree = 40
        result = self._call_func_under_test(points,
                                            max_degree=max_degree)
        self.assertEqual(result.shape, (num_nodes, max_degree + 1))
        self.assertTrue(result.flags.f_contiguous)
        expected_result = np.zeros((num_nodes, max_degree + 1))
        for n in six.moves.xrange(max_degree + 1):
            leg_coeffs = [0] * n + [1]
            expected_result[:, n] = legendre.legval(points, leg_coeffs)
        self.assertTrue(np.allclose(result, expected_result))
        frob_err = np.linalg.norm(result - expected_result, ord='fro')
        self.assertTrue(frob_err < 1e-13)

    def _evenly_spaced_condition_num_helper(self, p_order):
        import numpy as np

        x_vals = np.linspace(-1, 1, p_order + 1)
        leg_mat = self._call_func_under_test(x_vals)
        kappa2 = np.linalg.cond(leg_mat, p=2)
        # This gives the exponent of kappa2.
        base_exponent = np.log2(np.spacing(1.0))
        return int(np.round(np.log2(np.spacing(kappa2)) - base_exponent))

    def test_evenly_spaced_ill_conditioning(self):
        import six

        # 2^exponent <= kappa2 < 2^(exponent + 1)
        kappa_exponents = {
            p_order: self._evenly_spaced_condition_num_helper(p_order)
            for p_order in six.moves.xrange(1, 15 + 1)}
        self.assertEqual(kappa_exponents, {
            1: 0,
            2: 0,
            3: 1,
            4: 2,
            5: 2,
            6: 2,
            7: 3,
            8: 4,
            9: 4,
            10: 5,
            11: 6,
            12: 6,
            13: 7,
            14: 8,
            15: 9,
        })

    def _chebyshev_conditioning_helper(self, num_nodes):
        import numpy as np

        theta = np.pi * np.arange(2 * num_nodes - 1, 0, -2,
                                  dtype=np.float64) / (2 * num_nodes)
        x_vals = np.cos(theta)
        leg_mat = self._call_func_under_test(x_vals)
        kappa2 = np.linalg.cond(leg_mat, p=2)
        # This gives the exponent of kappa2.
        base_exponent = np.log2(np.spacing(1.0))
        return int(np.round(np.log2(np.spacing(kappa2)) - base_exponent))

    def test_chebyshev_conditioning(self):
        # Demonstrate not so bad conditioning
        import six

        # 2^exponent <= kappa2 < 2^(exponent + 1)
        kappa_exponents = {
            num_nodes: self._chebyshev_conditioning_helper(num_nodes)
            for num_nodes in six.moves.xrange(2, 15 + 1)}
        self.assertEqual(kappa_exponents, {
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 2,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
        })

    def _gauss_lobatto_conditioning_helper(self, num_nodes,
                                           zero_centered=False):
        import numpy as np
        from assignment1.dg1 import gauss_lobatto_info

        inner_nodes, _ = gauss_lobatto_info(num_nodes)
        if zero_centered:
            x_vals = np.hstack([[-1], inner_nodes, [1]])
        else:
            # Translate [-1, 1] --> [0, 2] --> [0, 1]
            x_vals = np.hstack([[0], 0.5 * (1 + inner_nodes), [1]])
        leg_mat = self._call_func_under_test(x_vals)
        kappa2 = np.linalg.cond(leg_mat, p=2)
        # This gives the exponent of kappa2.
        base_exponent = np.log2(np.spacing(1.0))
        return int(np.round(np.log2(np.spacing(kappa2)) - base_exponent))

    def test_gauss_lobatto_conditioning_wrong_interval(self):
        import six

        # 2^exponent <= kappa2 < 2^(exponent + 1)
        kappa_exponents = {
            num_nodes: self._gauss_lobatto_conditioning_helper(
                num_nodes, zero_centered=False)
            for num_nodes in six.moves.xrange(2, 15 + 1)}
        self.assertEqual(kappa_exponents, {
            2: 1,
            3: 3,
            4: 6,
            5: 8,
            6: 11,
            7: 13,
            8: 16,
            9: 18,
            10: 21,
            11: 23,
            12: 26,
            13: 28,
            14: 31,
            15: 33,
        })

    def test_gauss_lobatto_conditioning_same_interval(self):
        import six

        # 2^exponent <= kappa2 < 2^(exponent + 1)
        kappa_exponents = {
            num_nodes: self._gauss_lobatto_conditioning_helper(
                num_nodes, zero_centered=True)
            for num_nodes in six.moves.xrange(2, 15 + 1)}
        self.assertEqual(kappa_exponents, {
            2: 0,
            3: 0,
            4: 1,
            5: 1,
            6: 1,
            7: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
        })


class Test__find_matrices_helper(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(vals1, vals2):
        from assignment1.dg1 import _find_matrices_helper
        return _find_matrices_helper(vals1, vals2)

    def test_it(self):
        import sympy

        # NOTE: Could also use sympy.Matrix([sympy.symbols('u:4')]).
        #       but will refer to the variables.
        u0, u1, u2, u3 = sympy.symbols('u0, u1, u2, u3')
        v0, v1, v2, v3 = sympy.symbols('v0, v1, v2, v3')
        u = sympy.Matrix([[u0, u1, u2, u3]])
        v = sympy.Matrix([[v0, v1, v2, v3]])
        result = self._call_func_under_test(u, v)
        self.assertEqual(result, u0 * (v1 + v3) + u1 * v2 + u2 * v3)


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
        # We are solving on [0, 1] but ``find_matrices`` is
        # on [-1, 1], and the mass matrix is translation invariant
        # but scales with interval length.
        self.assertTrue(np.allclose(0.5 * mass_mat, expected_mass_mat))
        # The stiffness matrix is scale and translation invariant.
        self.assertTrue(np.allclose(stiffness_mat, expected_stiffness_mat))

    def test_p2(self):
        import numpy as np
        from assignment1.dg1 import mass_and_stiffness_matrices_p2

        mass_mat, stiffness_mat = self._call_func_under_test(2)
        (expected_mass_mat,
         expected_stiffness_mat) = mass_and_stiffness_matrices_p2()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        # We are solving on [0, 1] but ``find_matrices`` is
        # on [-1, 1], and the mass matrix is translation invariant
        # but scales with interval length.
        self.assertTrue(np.allclose(0.5 * mass_mat, expected_mass_mat))
        # The stiffness matrix is scale and translation invariant.
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


class Test_get_gaussian_like_initial_data(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(node_points):
        from assignment1.dg1 import get_gaussian_like_initial_data
        return get_gaussian_like_initial_data(node_points)

    def test_it(self):
        import numpy as np
        # We cheat and use the inverse.
        #           E == exp(-25(2N - 1)^2)
        # <==> log(E) == -25(2N - 1)^2
        # <==> 2N - 1 == +/- SQRT(0.04 * log(1/E))
        # <==>      N == 0.5 +/- SQRT(0.01 * log(1/E))
        expected_result = 1.0 / np.array([
            [3, 12, 8, 1],
            [7, 7, 4, 9],
            [2, 1, 5, 11],
        ])
        node_points = 0.5 + np.sqrt(0.01 * np.log(1.0/expected_result))
        result = self._call_func_under_test(node_points)
        self.assertTrue(np.allclose(result, expected_result))


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

    def test_from_solver_factory(self):
        import numpy as np

        klass = self._get_target_class()
        solver = mock.MagicMock(node_points=np.array([
            [0., 2., 4.],
            [1., 3., 5.],
        ]))
        num_points = 5
        interp_func = klass.from_solver(solver, num_points=num_points)
        # x_vals are first column of mock node points
        self.assertTrue(np.all(interp_func.x_vals == [0, 1]))
        # Split the interval [0, 1] (given by the first column of x_vals)
        # into num_points = 5 pieces.
        self.assertTrue(np.all(interp_func.all_x ==
                               [0.0, 0.25, 0.5, 0.75, 1.0]))
        # The interpolating functions are L0(x) = 1 - x, L1(x) = x, evaluated
        # at the points above.
        expected_lagrange = [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.5, 0.5],
            [0.25, 0.75],
            [0.0, 1.0],
        ]
        self.assertTrue(np.all(interp_func.lagrange_matrix ==
                               expected_lagrange))

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


class TestDG1Solver(unittest.TestCase):

    def setUp(self):
        self.num_intervals = 10
        self.step_size = 0.1
        self.num_steps = 500
        self.dt = 2e-3
        self.total_time = self.num_steps * self.dt

    @staticmethod
    def _get_target_class():
        from assignment1.dg1 import DG1Solver
        return DG1Solver

    def _make_one(self, num_intervals, p_order, total_time, dt,
                  get_initial_data=None):
        return self._get_target_class()(num_intervals, p_order,
                                        total_time, dt,
                                        get_initial_data=get_initial_data)

    def _constructor_helper(self, p_order, nodes, soln, mass_mat,
                            stiffness_mat, get_initial_data=None):
        solver = self._make_one(self.num_intervals, p_order, self.total_time,
                                self.dt,
                                get_initial_data=get_initial_data)
        self.assertIsInstance(solver, self._get_target_class())
        self.assertEqual(solver.num_intervals, self.num_intervals)
        self.assertEqual(solver.p_order, p_order)
        self.assertEqual(solver.total_time, self.total_time)
        self.assertEqual(solver.dt, self.dt)
        self.assertEqual(solver.current_step, 0)
        self.assertEqual(solver.num_steps, self.num_steps)
        self.assertEqual(solver.step_size, self.step_size)
        self.assertEqual(solver.node_points, nodes)
        self.assertEqual(solver.solution, soln)
        self.assertEqual(solver.mass_mat, mass_mat)
        self.assertEqual(solver.stiffness_mat, stiffness_mat)

    @mock.patch('assignment1.dg1.get_node_points')
    @mock.patch('assignment1.dg1.get_gaussian_like_initial_data')
    def _constructor_small_p_helper(self, p_order, init_data, get_nodes):
        # Set-up mocks.
        get_nodes.return_value = nodes = object()
        init_data.return_value = object()
        mock_mass_base = mock.MagicMock()
        mock_stiffness = object()
        mock_mass_base.__rmul__.return_value = mock_mass = object()

        patch_name = ('assignment1.dg1.'
                      'mass_and_stiffness_matrices_p%d' % (p_order,))
        with mock.patch(patch_name) as get_mats:
            get_mats.return_value = mock_mass_base, mock_stiffness
            self._constructor_helper(p_order, nodes, init_data.return_value,
                                     mock_mass, mock_stiffness)
        # Verify mocks were called.
        get_nodes.assert_called_once_with(self.num_intervals, p_order,
                                          step_size=self.step_size)
        init_data.assert_called_once_with(nodes)
        get_mats.assert_called_once_with()
        mock_mass_base.__rmul__.assert_called_once_with(self.step_size)

    def test_constructor_small_p(self):
        self._constructor_small_p_helper(1)
        self._constructor_small_p_helper(2)
        self._constructor_small_p_helper(3)

    @mock.patch('assignment1.dg1.get_node_points')
    @mock.patch('assignment1.dg1.mass_and_stiffness_matrices_p1')
    def test_constructor_explicit_init_data(self, get_mats, get_nodes):
        # Set-up mocks.
        get_nodes.return_value = nodes = object()
        mock_mass_base = mock.MagicMock()
        mock_stiffness = object()
        get_mats.return_value = mock_mass_base, mock_stiffness
        mock_mass_base.__rmul__.return_value = mock_mass = object()

        init_data_obj = object()
        init_data_points = []

        def init_data(node_points):
            init_data_points.append(node_points)
            return init_data_obj

        # Construct the object.
        p_order = 1
        self._constructor_helper(p_order, nodes, init_data_obj,
                                 mock_mass, mock_stiffness,
                                 get_initial_data=init_data)
        # Verify mocks were called.
        get_nodes.assert_called_once_with(self.num_intervals, p_order,
                                          step_size=self.step_size)
        self.assertEqual(init_data_points, [nodes])
        get_mats.assert_called_once_with()
        mock_mass_base.__rmul__.assert_called_once_with(self.step_size)

    @mock.patch('assignment1.dg1.get_node_points')
    @mock.patch('assignment1.dg1.find_matrices')
    @mock.patch('assignment1.dg1.get_gaussian_like_initial_data')
    def test_constructor_large_p(self, init_data, find_mats, get_nodes):
        # Set-up mocks.
        get_nodes.return_value = nodes = object()
        init_data.return_value = object()
        mock_mass1 = mock.MagicMock()
        mock_stiffness = object()
        find_mats.return_value = mock_mass1, mock_stiffness
        mock_mass1.__rmul__.return_value = mock_mass2 = mock.MagicMock()
        mock_mass2.__rmul__.return_value = mock_mass3 = object()
        # Construct the object.
        p_order = 10
        self._constructor_helper(p_order, nodes, init_data.return_value,
                                 mock_mass3, mock_stiffness)
        # Verify mocks were called.
        get_nodes.assert_called_once_with(self.num_intervals, p_order,
                                          step_size=self.step_size)
        init_data.assert_called_once_with(nodes)
        find_mats.assert_called_once_with(p_order)
        mock_mass1.__rmul__.assert_called_once_with(0.5)
        mock_mass2.__rmul__.assert_called_once_with(self.step_size)

    def test_constructor_no_mocks(self):
        import numpy as np

        def init_data(node_points):
            return 2 * node_points

        self.num_intervals = 2
        self.step_size = 0.5
        solver = self._make_one(self.num_intervals, 1,
                                self.total_time, self.dt,
                                get_initial_data=init_data)
        self.assertIsInstance(solver, self._get_target_class())
        self.assertEqual(solver.num_intervals, self.num_intervals)
        self.assertEqual(solver.p_order, 1)
        self.assertEqual(solver.total_time, self.total_time)
        self.assertEqual(solver.dt, self.dt)
        self.assertEqual(solver.current_step, 0)
        self.assertEqual(solver.num_steps, self.num_steps)
        self.assertEqual(solver.step_size, self.step_size)
        expected_nodes = [
            [0, 0.5],
            [0.5, 1],
        ]
        self.assertTrue(np.all(solver.node_points == expected_nodes))
        self.assertTrue(np.all(solver.solution == 2 * solver.node_points))
        # No need to check the matrices.

    def test_ode_func(self):
        import numpy as np

        solver = self._make_one(2, 1, self.total_time, self.dt)
        # Artificially patch the mass_mat and stiffness_mat.
        solver.mass_mat = np.array([
            [4., 0.],
            [0., 5.],
        ])
        solver.stiffness_mat = np.array([
            [5., 1.],
            [-2., 0.],
        ])
        u_val = np.array([
            [0., 0.1, 0.2, 0.7, 1.1],
            [0.4, 0.8, 1.2, 1.9, 5.0],
        ])
        expected_result = np.array([
            [1.35, 0.425, 0.75, 1.65, 3.1],
            [-0.08, -0.2, -0.32, -0.66, -1.44],
        ])
        self.assertTrue(np.allclose(solver.ode_func(u_val), expected_result))

    @mock.patch('assignment1.dg1.low_storage_rk', return_value=object())
    def test_update(self, rk_method):
        solver = self._make_one(2, 1, self.total_time, self.dt)
        # Artificially patch the pieces used by update.
        solver.current_step = current_step = 121
        solver.solution = orig_soln = object()
        solver.dt = object()
        result = solver.update()
        self.assertEqual(result, None)
        # Check updated state.
        self.assertEqual(solver.current_step, current_step + 1)
        self.assertEqual(solver.solution, rk_method.return_value)
        # Verify mock.
        rk_method.assert_called_once_with(solver.ode_func,
                                          orig_soln, solver.dt)


class Test__plot_solution(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(color, num_cols, interp_func,
                              solver, ax):
        from assignment1.dg1 import _plot_solution
        return _plot_solution(color, num_cols, interp_func, solver, ax)

    def test_it(self):
        import sympy

        # Create mock DG1Solver.
        node_points = sympy.Matrix([
            [42, 1337],
        ])
        solution = object()
        solver = mock.MagicMock(
            node_points=node_points,
            solution=solution,
        )
        # Create mock PolynomialInterpolate.
        all_x = 99
        interp_func = mock.MagicMock(
            all_x=all_x,
        )
        y_vals = sympy.Matrix([
            [10, 100],
            [11, 200],
        ])
        interp_func.interpolate.return_value = y_vals
        # Create mock axis.
        ax = mock.MagicMock()
        line_obj = object()
        ax.plot.return_value = (line_obj,)
        # Call the method.
        num_cols = 2
        color = 'green'
        result = self._call_func_under_test(
            color, num_cols, interp_func, solver, ax)
        # Verify result (lines come from ax.plot mock).
        self.assertEqual(result, [line_obj] * num_cols)
        # Verify mocks were called.
        interp_func.interpolate.assert_called_once_with(solution)
        plot_calls = ax.plot.mock_calls
        self.assertEqual(len(plot_calls), 2)
        self.assertEqual(
            plot_calls[0],
            mock.call(
                all_x + node_points[0, 0],
                y_vals[:, 0],
                color=color, linewidth=2,
            ),
        )
        self.assertEqual(
            plot_calls[1],
            mock.call(
                all_x + node_points[0, 1],
                y_vals[:, 1],
                color=color, linewidth=2,
            ),
        )


class Test__configure_axis(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(ax, **kwargs):
        from assignment1.dg1 import _configure_axis
        return _configure_axis(ax, **kwargs)

    def test_defaults(self):
        # Make ax a mock.
        ax = mock.MagicMock()
        # Call function.
        result = self._call_func_under_test(ax)
        self.assertEqual(result, None)
        # Verify mocks were called.
        ax.set_xlim.assert_called_once_with(0.0, 1.0)
        ax.set_ylim.assert_called_once_with(-0.1, 1.1)
        ax.grid.assert_called_once_with(b=True)

    def test_explicit(self):
        # Make ax a mock.
        ax = mock.MagicMock()
        # Call function.
        x_min = 1.2
        x_max = 3.4
        y_min = 5.6
        y_max = 7.8
        y_buffer = 1000.0
        result = self._call_func_under_test(ax, x_min=x_min, x_max=x_max,
                                            y_min=y_min, y_max=y_max,
                                            y_buffer=y_buffer)
        self.assertEqual(result, None)
        # Verify mocks were called.
        ax.set_xlim.assert_called_once_with(x_min, x_max)
        ax.set_ylim.assert_called_once_with(y_min - y_buffer,
                                            y_max + y_buffer)
        ax.grid.assert_called_once_with(b=True)


class TestDG1Animate(unittest.TestCase):

    @staticmethod
    def _get_target_class():
        from assignment1.dg1 import DG1Animate
        return DG1Animate

    def _make_one(self, solver, fig=None, ax=None, interp_points=None):
        return self._get_target_class()(solver, fig=fig, ax=ax,
                                        interp_points=interp_points)

    @mock.patch('matplotlib.pyplot.subplots', create=True)
    @mock.patch('assignment1.dg1.PolynomialInterpolate.from_solver',
                return_value=object())
    def test_constructor_defaults(self, interp_factory, subplots):
        # Set-up mocks.
        fig = object()
        ax = object()
        subplots.return_value = fig, ax
        # Construct the object.
        solver = object()
        animate_obj = self._make_one(solver)
        # Verify properties
        self.assertEqual(animate_obj.solver, solver)
        self.assertEqual(animate_obj.poly_interp_func,
                         interp_factory.return_value)
        self.assertEqual(animate_obj.fig, fig)
        self.assertEqual(animate_obj.ax, ax)
        # Verify mock.
        subplots.assert_called_once_with(1, 1)
        interp_factory.assert_called_once_with(solver, num_points=None)

    @mock.patch('assignment1.dg1.PolynomialInterpolate.from_solver')
    def test_constructor_axis_with_no_figure(self, interp_factory):
        # Make a subclass which mocks out the constructor helpers.
        solver = object()
        # Fail to construct the object.
        with self.assertRaises(ValueError):
            self._make_one(solver, ax=object())
        interp_factory.assert_called_once_with(solver, num_points=None)

    @mock.patch('assignment1.dg1.PolynomialInterpolate.from_solver')
    def test_constructor_figure_with_no_axis(self, interp_factory):
        # Make a subclass which mocks out the constructor helpers.
        solver = object()
        # Fail to construct the object.
        with self.assertRaises(ValueError):
            self._make_one(solver, fig=object())
        interp_factory.assert_called_once_with(solver, num_points=None)

    @mock.patch('assignment1.dg1.PolynomialInterpolate.from_solver',
                return_value=object())
    @mock.patch('assignment1.dg1._configure_axis')
    @mock.patch('assignment1.dg1._plot_solution')
    def test_init_func(self, plot_soln, config_ax, interp_factory):
        import numpy as np

        # Set-up mocks.
        num_cols = 1
        solver = mock.MagicMock(
            node_points=np.array([0, 1]).reshape(2, num_cols),
            solution=[2, 3],
        )
        fig = object()
        ax = object()
        plot_line = mock.MagicMock()
        plot_soln.return_value = [plot_line]
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        plot_lines = animate_obj.init_func()
        self.assertEqual(plot_lines, plot_soln.return_value)
        # Verify mocks were called.
        config_ax.assert_called_once_with(ax, x_min=0, x_max=1,
                                          y_min=2, y_max=3)
        self.assertEqual(len(plot_soln.mock_calls), 2)
        self.assertEqual(
            plot_soln.mock_calls[0],
            mock.call('red', num_cols, interp_factory.return_value,
                      solver, ax),
        )
        interp_factory.assert_called_once_with(solver, num_points=None)
        self.assertEqual(
            plot_soln.mock_calls[1],
            mock.call('blue', num_cols, interp_factory.return_value,
                      solver, ax),
        )
        interp_factory.assert_called_once_with(solver, num_points=None)
        plot_line.set_visible.assert_called_once_with(False)

    @mock.patch('assignment1.dg1.PolynomialInterpolate.from_solver')
    def test_update_plot_frame_zero(self, interp_factory):
        import sympy

        # Set-up mocks.
        solver = mock.MagicMock(
            current_step=0,
            solution=object(),
        )
        fig = ax = object()
        poly_interp = mock.MagicMock()
        poly_interp.interpolate.return_value = sympy.Matrix([[0, 0]])
        interp_factory.return_value = poly_interp
        plot_line = mock.MagicMock()
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        animate_obj.plot_lines = [plot_line]
        plot_lines = animate_obj.update_plot(solver.current_step)
        self.assertEqual(plot_lines, animate_obj.plot_lines)
        # Verify mocks were called.
        plot_line.set_visible.assert_called_once_with(True)
        solver.update.assert_called_once_with()
        poly_interp.interpolate.assert_called_once_with(solver.solution)
        plot_line.set_ydata.assert_called_once_with(sympy.Matrix([[0]]))

    @mock.patch('assignment1.dg1.PolynomialInterpolate.from_solver')
    def test_update_plot_frame_nonzero(self, interp_factory):
        import sympy

        # Set-up mocks.
        solver = mock.MagicMock(
            current_step=10203,
            solution=object(),
        )
        fig = ax = object()
        poly_interp = mock.MagicMock()
        poly_interp.interpolate.return_value = sympy.Matrix([[0, 0]])
        interp_factory.return_value = poly_interp
        plot_line = mock.MagicMock()
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        animate_obj.plot_lines = [plot_line]
        plot_lines = animate_obj.update_plot(solver.current_step)
        self.assertEqual(plot_lines, animate_obj.plot_lines)
        # Verify mocks were called.
        plot_line.set_visible.assert_not_called()
        solver.update.assert_called_once_with()
        poly_interp.interpolate.assert_called_once_with(solver.solution)
        plot_line.set_ydata.assert_called_once_with(sympy.Matrix([[0]]))

    @mock.patch('assignment1.dg1.PolynomialInterpolate.from_solver')
    def test_update_plot_mismatch_frame(self, interp_factory):
        # Set-up mocks.
        solver = mock.MagicMock(
            current_step=10203,
            solution=object(),
        )
        fig = ax = object()
        poly_interp = mock.MagicMock()
        interp_factory.return_value = poly_interp
        plot_line = mock.MagicMock()
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        animate_obj.plot_lines = [plot_line]
        with self.assertRaises(ValueError):
            animate_obj.update_plot(-1)
        # Verify mocks were called.
        plot_line.set_visible.assert_not_called()
        solver.update.assert_not_called()
        poly_interp.interpolate.assert_not_called()
        plot_line.set_ydata.assert_not_called()
