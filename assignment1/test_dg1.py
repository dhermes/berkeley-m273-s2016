import unittest

import mock


class Test_gauss_lobatto_points(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(start, stop, num_points):
        from assignment1.dg1 import gauss_lobatto_points
        return gauss_lobatto_points(start, stop, num_points)

    def test_five_points(self):
        import numpy as np
        import sympy

        t_symb = sympy.Symbol('t')
        num_points = 5
        p4 = sympy.legendre(num_points - 1, t_symb)
        p4_prime = p4.diff(t_symb)
        start = -1.0
        stop = 1.0
        all_nodes = self._call_func_under_test(start, stop, num_points)
        self.assertEqual(all_nodes[0], start)
        self.assertEqual(all_nodes[-1], stop)
        inner_nodes = all_nodes[1:-1]
        # Make sure the computed roots are actually roots.
        self.assertTrue(np.allclose(
            [float(p4_prime.subs({t_symb: root}))
             for root in inner_nodes], 0))

    def test_symmetry(self):
        import numpy as np

        all_nodes = self._call_func_under_test(-1.0, 1.0, 15)
        self.assertTrue(np.all(all_nodes == -all_nodes[::-1]))

    def test_non_standard_interval(self):
        import numpy as np

        all_nodes = self._call_func_under_test(0.0, 10.0, 4)
        # [-1, 1/sqrt(5), 1/sqrt(5), 1] --> [0, 1-1/sqrt(5), 1+1/sqrt(5), 2]
        self.assertTrue(np.allclose(
            all_nodes, [0.0, 5 - np.sqrt(5), 5.0 + np.sqrt(5), 10.0]))

    @staticmethod
    def _accuracy_helper(degree, all_nodes, all_weights):
        import numpy as np
        from numpy.polynomial import polynomial

        if degree % 2 == 0:
            expected_value = 2.0 / (degree + 1)
        else:
            # Odd-function integrate to 0.
            expected_value = 0.0
        curr_poly = [0.0] * degree + [1.0]
        poly_vals = polynomial.polyval(all_nodes, curr_poly)
        quadrature_val = np.dot(poly_vals, all_weights)
        return expected_value, quadrature_val

    @staticmethod
    def _get_weights(num_points, inner_nodes):
        from numpy.polynomial import legendre

        base_weight = 2.0 / (num_points * (num_points - 1))
        p_n_minus1 = [0] * (num_points - 1) + [1]
        p_n_minus1_at_nodes = legendre.legval(inner_nodes, p_n_minus1)
        return base_weight / p_n_minus1_at_nodes**2

    def test_accuracy(self):
        import numpy as np
        import six

        num_points = 4
        all_nodes = self._call_func_under_test(-1.0, 1.0, num_points)
        self.assertEqual(all_nodes[0], -1.0)
        self.assertEqual(all_nodes[-1], 1.0)
        inner_nodes = all_nodes[1:-1]
        inner_weights = self._get_weights(num_points, inner_nodes)
        all_nodes = np.hstack([[-1], inner_nodes, [1]])
        base_weight = 2.0 / (num_points * (num_points - 1))
        all_weights = np.hstack([[base_weight], inner_weights,
                                 [base_weight]])
        for degree in six.moves.xrange(2 * num_points - 2):
            # Verify x^degree is exact.
            expected_value, quadrature_val = self._accuracy_helper(
                degree, all_nodes, all_weights)
            self.assertTrue(np.allclose(quadrature_val, expected_value))

        # Verify that the next degree does not integrate exactly.
        expected_value, quadrature_val = self._accuracy_helper(
            2 * num_points - 2, all_nodes, all_weights)
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
        from assignment1.dg1 import gauss_lobatto_points

        all_nodes = gauss_lobatto_points(-1.0, 1.0, num_nodes)
        self.assertEqual(all_nodes[0], -1.0)
        self.assertEqual(all_nodes[-1], 1.0)
        inner_nodes = all_nodes[1:-1]
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


class Test_get_evenly_spaced_points(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(start, stop, num_points):
        from assignment1.dg1 import get_evenly_spaced_points
        return get_evenly_spaced_points(start, stop, num_points)

    @mock.patch('assignment1.dg1.MathProvider.linspace',
                return_value=object())
    def test_shadows_np_linspace(self, linspace_mock):
        start = object()
        stop = object()
        num_points = object()
        result = self._call_func_under_test(start, stop, num_points)
        self.assertEqual(result, linspace_mock.return_value)
        linspace_mock.assert_called_once_with(start, stop, num_points)

    def test_simple_interval(self):
        import numpy as np

        result = self._call_func_under_test(0, 1, 5)
        self.assertTrue(np.all(result == [0, 0.25, 0.5, 0.75, 1]))


class Test_find_matrices(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order, points_on_ref_int=None):
        from assignment1.dg1 import find_matrices
        return find_matrices(p_order, points_on_ref_int=points_on_ref_int)

    def test_p1(self):
        import numpy as np

        mass_mat, stiffness_mat = self._call_func_under_test(1)
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        # We are solving on [0, 1] but ``find_matrices`` is
        # on [-1, 1], and the mass matrix is translation invariant
        # but scales with interval length.
        mass_mat_p1 = np.array([
            [2, 1],
            [1, 2],
        ]) / 6.0
        self.assertTrue(np.allclose(0.5 * mass_mat, mass_mat_p1))
        # The stiffness matrix is scale and translation invariant.
        stiffness_mat_p1 = np.array([
            [-1, -1],
            [1, 1],
        ]) / 2.0
        self.assertTrue(np.allclose(stiffness_mat, stiffness_mat_p1))

    def test_p1_explicit_points_on_ref(self):
        import numpy as np

        explicit_points_called = []
        p_order = 1

        def explicit_points(start, stop, loc_num_points):
            explicit_points_called.append((start, stop, loc_num_points))
            # NOTE: This result assumes we know p_order == 1.
            self.assertEqual(loc_num_points, 2)
            return np.array([1.0, 4.0])

        mass_mat, stiffness_mat = self._call_func_under_test(
            p_order, points_on_ref_int=explicit_points)
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        self.assertEqual(explicit_points_called, [(-1, 1, p_order + 1)])
        # We have the basis functions PHI_0 = (4 - x) / 3 and
        # PHI_1 = (x - 1) / 3. When we integrate them:
        # M_{00} = 1 / 9 INTEGRAL_{-1}^1 (4 - x)^2 dx
        #        = (1 - 4)^3 / 27 - ((-1) - 4)^3 / 27 = 98 / 27
        # M_{11} = 1 / 9 INTEGRAL_{-1}^1 (x - 1)^2 dx
        #        = (1 - 1)^3/27 - ((-1) - 1)^3/27 = 8 / 27
        # M_{01} = 1 / 9 INTEGRAL_{-1}^1 (4 - x)(x - 1) dx = -26 / 27
        self.assertTrue(np.allclose(27 * mass_mat, [[98, -26], [-26, 8]]))
        # Since PHI_0' = -1/3, PHI_1' = 1/3
        # K_{00} = -1 / 9 INTEGRAL_{-1}^1 (4 - x) dx
        #        = (1 - 4)^2/18 - ((-1) - 4)^2/18 = -16 / 18
        # K_{11} = 1 / 9 INTEGRAL_{-1}^1 (x - 1) dx
        #        = (1 - 1)^2/18 - ((-1) - 1)^2/18 = -4 / 18
        # K_{01} = - K_{11}
        self.assertTrue(np.all(9 * stiffness_mat == [[-8, 2], [-2, -2]]))

    def test_p2(self):
        import numpy as np

        mass_mat, stiffness_mat = self._call_func_under_test(2)
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        # We are solving on [0, 1] but ``find_matrices`` is
        # on [-1, 1], and the mass matrix is translation invariant
        # but scales with interval length.
        mass_mat_p2 = np.array([
            [4, 2, -1],
            [2, 16, 2],
            [-1, 2, 4],
        ]) / 30.0
        self.assertTrue(np.allclose(0.5 * mass_mat, mass_mat_p2))
        # The stiffness matrix is scale and translation invariant.
        stiffness_mat_p2 = np.array([
            [-3, -4, 1],
            [4, 0, -4],
            [-1, 4, 3],
        ]) / 6.0
        self.assertTrue(np.allclose(stiffness_mat, stiffness_mat_p2))

    def test_p3(self):
        import numpy as np

        mass_mat, stiffness_mat = self._call_func_under_test(3)
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        # We are solving on [0, 1] but ``find_matrices`` is
        # on [-1, 1], and the mass matrix is translation invariant
        # but scales with interval length.
        mass_mat_p3 = np.array([
            [128, 99, -36, 19],
            [99, 648, -81, -36],
            [-36, -81, 648, 99],
            [19, -36, 99, 128],
        ]) / 1680.0
        self.assertTrue(np.allclose(0.5 * mass_mat, mass_mat_p3))
        # The stiffness matrix is scale and translation invariant.
        stiffness_mat_p3 = np.array([
            [-40, -57, 24, -7],
            [57, 0, -81, 24],
            [-24, 81, 0, -57],
            [7, -24, 57, 40],
        ]) / 80.0
        self.assertTrue(np.allclose(stiffness_mat, stiffness_mat_p3))


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
    def _call_func_under_test(num_points, p_order, step_size=None,
                              points_on_ref_int=None):
        from assignment1.dg1 import get_node_points
        return get_node_points(num_points, p_order, step_size=step_size,
                               points_on_ref_int=points_on_ref_int)

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

    def test_explicit_points_on_ref(self):
        import numpy as np

        explicit_points_called = []
        p_order = 1
        num_points = 4

        def explicit_points(start, stop, loc_num_points):
            explicit_points_called.append((start, stop, loc_num_points))
            # NOTE: This result assumes we know p_order = 1
            self.assertEqual(loc_num_points, 2)
            return np.array([3.0, 7.0])

        result = self._call_func_under_test(num_points, p_order,
                                            points_on_ref_int=explicit_points)
        self.assertEqual(explicit_points_called,
                         [(0, 1.0/num_points, p_order + 1)])
        self.assertEqual(result.shape, (p_order + 1, num_points))
        self.assertTrue(np.all(result == [
            [3.0, 3.25, 3.5, 3.75],
            [7.0, 7.25, 7.5, 7.75],
        ]))


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
                  get_initial_data=None, points_on_ref_int=None):
        return self._get_target_class()(num_intervals, p_order,
                                        total_time, dt,
                                        get_initial_data=get_initial_data,
                                        points_on_ref_int=points_on_ref_int)

    # pylint: disable=too-many-arguments
    def _constructor_helper(self, p_order, nodes, soln, mass_mat,
                            stiffness_mat, get_initial_data=None,
                            points_on_ref_int=None):
        solver = self._make_one(self.num_intervals, p_order, self.total_time,
                                self.dt, get_initial_data=get_initial_data,
                                points_on_ref_int=points_on_ref_int)
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
    # pylint: enable=too-many-arguments

    @mock.patch('assignment1.dg1.get_node_points')
    @mock.patch('assignment1.dg1.find_matrices')
    def test_constructor_explicit_init_data(self, find_mats, get_nodes):
        # Set-up mocks.
        get_nodes.return_value = nodes = object()
        mock_mass_base = mock.MagicMock()
        mock_stiffness = object()
        find_mats.return_value = mock_mass_base, mock_stiffness
        mock_mass_base.__rmul__.return_value = mock_mass = object()

        init_data_obj = object()
        init_data_points = []

        def init_data(node_points):
            init_data_points.append(node_points)
            return init_data_obj

        # Construct the object.
        p_order = 1
        get_points = object()
        self._constructor_helper(p_order, nodes, init_data_obj,
                                 mock_mass, mock_stiffness,
                                 get_initial_data=init_data,
                                 points_on_ref_int=get_points)
        # Verify mocks were called.
        get_nodes.assert_called_once_with(self.num_intervals, p_order,
                                          step_size=self.step_size,
                                          points_on_ref_int=get_points)
        self.assertEqual(init_data_points, [nodes])
        find_mats.assert_called_once_with(p_order,
                                          points_on_ref_int=get_points)
        mock_mass_base.__rmul__.assert_called_once_with(0.5 * self.step_size)

    @mock.patch('assignment1.dg1.get_node_points')
    @mock.patch('assignment1.dg1.find_matrices')
    @mock.patch('assignment1.dg1.get_gaussian_like_initial_data')
    def _constructor_defaults_helper(self, p_order, init_data,
                                     find_mats, get_nodes):
        # Set-up mocks.
        get_nodes.return_value = nodes = object()
        init_data.return_value = object()
        mock_mass1 = mock.MagicMock()
        mock_stiffness = object()
        find_mats.return_value = mock_mass1, mock_stiffness
        mock_mass1.__rmul__.return_value = mock_mass2 = object()
        # Construct the object.
        self._constructor_helper(p_order, nodes, init_data.return_value,
                                 mock_mass2, mock_stiffness)
        # Verify mocks were called.
        get_nodes.assert_called_once_with(self.num_intervals, p_order,
                                          step_size=self.step_size,
                                          points_on_ref_int=None)
        init_data.assert_called_once_with(nodes)
        find_mats.assert_called_once_with(p_order, points_on_ref_int=None)
        mock_mass1.__rmul__.assert_called_once_with(0.5 * self.step_size)

    def test_constructor_small_and_large_p(self):
        self._constructor_defaults_helper(1)
        self._constructor_defaults_helper(2)
        self._constructor_defaults_helper(3)
        self._constructor_defaults_helper(4)
        self._constructor_defaults_helper(10)

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

    def _get_error(self, num_intervals, p_order, T, dt):
        import numpy as np

        solver = self._make_one(num_intervals, p_order, T, dt)
        # Save initial solution for later comparison (though a copy is
        # not strictly needed).
        init_soln = solver.solution.copy()
        while solver.current_step != solver.num_steps:
            solver.update()
        return np.linalg.norm(init_soln - solver.solution, ord='fro')

    def _convergence_helper(self, p_order, drop_off=0.0):
        import numpy as np
        import six

        T = 1.0
        num_intervals = 4
        # NOTE: dt / dx == 1 / (3 p^2)
        dx = 1.0 / num_intervals
        dt = dx / (3 * p_order * p_order)
        h_vals = []
        errors = []
        for _ in six.moves.xrange(5):
            err_frob = self._get_error(num_intervals, p_order, T, dt)
            errors.append(err_frob)
            h_vals.append(dx)
            # Update the values used. Preserve the CFL condition by
            # leaving dt / dx CONSTANT.
            num_intervals *= 2
            dx *= 0.5
            dt *= 0.5

        conv_rate, _ = np.polyfit(np.log2(h_vals), np.log2(errors), deg=1)
        self.assertTrue(conv_rate >= p_order - drop_off)
        self.assertTrue(conv_rate < p_order + 1 - drop_off)

    def test_linear_convergence(self):
        self._convergence_helper(1)

    def test_quadratic_convergence(self):
        self._convergence_helper(2)

    def test_cubic_convergence(self):
        self._convergence_helper(3)

    def test_quartic_convergence(self):
        self._convergence_helper(4)

    def test_quintic_convergence(self):
        self._convergence_helper(5)
